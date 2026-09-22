"""Inference for multi-label checkpoints: every label of a state decided in one forward pass."""
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch

from ..agent import Agent
from ..common import collate_items
from .calibration import sigmoid
from .data import Budget, build_items
from .rlcd import label_logits
from .schema import LabelSchema

State = Union[str, dict, list]


@torch.no_grad()
def score_states(
    model: torch.nn.Module,
    tok,
    schema: LabelSchema,
    states: Sequence[State],
    budget: Budget,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    batch_size: int = 32,
) -> np.ndarray:
    """Label decision logits [n, K] (label marker minus threshold marker), before temperature."""
    items = []
    for i, s in enumerate(states):
        items.extend(build_items(tok, schema, s, budget, ex=i))
    order = sorted(range(len(items)), key=lambda i: len(items[i]["ids"]))

    out = np.zeros((len(states), len(schema)), dtype=np.float32)
    was_training = model.training
    model.eval()
    for s in range(0, len(order), batch_size):
        sel = [items[i] for i in order[s:s + batch_size]]
        b = collate_items([sel], tok.pad_token_id)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
            logits, _ = model(
                b["input_ids"].to(device),
                b["attention_mask"].to(device),
                b["marker_pos"].to(device),
                b["marker_mask"].to(device),
                b["qtype"].to(device),
            )
        d = label_logits(logits.float()).cpu().numpy()
        for r, it in enumerate(sel):
            out[it["ex"], it["label_idx"]] = d[r, :len(it["label_idx"])]
    model.train(was_training)
    return out


class MultiLabelAgent:
    """Runtime for a checkpoint written by `laya.multilabel.train`.

    The checkpoint is an ordinary Laya checkpoint (same architecture, same files) whose
    `rl_agent_config.json` carries a `multilabel` section: label schema, fitted temperature and
    decision thresholds. `self.agent` is the regular `laya.Agent` over the same weights.
    """

    def __init__(
        self,
        model_id_or_path: str,
        device: Optional[str] = None,
        token: Optional[str] = None,
        subfolder: Optional[str] = None,
    ):
        self.agent = Agent(model_id_or_path, device=device, token=token, subfolder=subfolder)
        ml = self.agent.cfg.get("multilabel")
        if not ml:
            raise ValueError(
                "%r has no 'multilabel' section in rl_agent_config.json; train one with "
                "`python -m laya.multilabel train`." % model_id_or_path
            )
        self.schema = LabelSchema.from_dict(ml)
        self.temperature = float(ml.get("temperature", 1.0))
        thr = ml.get("thresholds", {})
        self.thresholds = np.array([float(thr.get(n, 0.5)) for n in self.schema.names])
        self.budget = Budget(
            max_len=self.agent.cfg.get("max_len", 512),
            head_max_len=self.agent.cfg.get("head_max_len", 192),
            labels_per_seq=int(ml.get("labels_per_seq", len(self.schema))),
        )

    def probabilities(self, states: Sequence[State], batch_size: int = 32) -> np.ndarray:
        """Calibrated P(label) for every state and label, [n, K] in `schema.names` order."""
        a = self.agent
        d = score_states(a.model, a.tok, self.schema, states, self.budget, a.device, a.dtype, batch_size)
        return sigmoid(d / max(1e-3, self.temperature))

    def predict_batch(
        self, states: Sequence[State], threshold: Optional[float] = None, batch_size: int = 32
    ) -> List[Dict[str, Any]]:
        p = self.probabilities(states, batch_size)
        thr = self.thresholds if threshold is None else np.full(len(self.schema), float(threshold))
        results = []
        for row in p:
            pred = row >= thr
            results.append({
                "type": "multi",
                "labels": [n for n, on in zip(self.schema.names, pred) if on],
                "probabilities": {n: round(float(v), 4) for n, v in zip(self.schema.names, row)},
                # probability that every label decision is right, treating them as independent
                "confidence": round(float(np.where(pred, row, 1.0 - row).prod()), 4),
            })
        return results

    def predict(self, state: State, threshold: Optional[float] = None) -> Dict[str, Any]:
        return self.predict_batch([state], threshold)[0]


def load(model_id_or_path: str, device: Optional[str] = None, token: Optional[str] = None,
         subfolder: Optional[str] = None) -> MultiLabelAgent:
    return MultiLabelAgent(model_id_or_path, device=device, token=token, subfolder=subfolder)
