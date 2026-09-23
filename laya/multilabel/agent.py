"""Inference for multi-label checkpoints: every label of a state decided in one forward pass."""
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch

from ..agent import Agent, _fix_tokenizer_config
from ..common import DecisionModel, collate_items
from .backbone import attach_adapter, ensure_pad, head_state_dict, load_backbone
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


def _resolve_dir(model_id_or_path: str, token: Optional[str], subfolder: Optional[str]) -> str:
    local = os.path.join(model_id_or_path, subfolder) if subfolder else model_id_or_path
    if os.path.isdir(local):
        return local
    if model_id_or_path.startswith(("/", "./", "../")):
        raise FileNotFoundError("Local model path not found: %r" % model_id_or_path)
    from huggingface_hub import snapshot_download

    kw = {"token": token or os.environ.get("HF_TOKEN")}
    if subfolder:
        kw["allow_patterns"] = ["%s/*" % subfolder]
    root = snapshot_download(model_id_or_path, **kw)
    return os.path.join(root, subfolder) if subfolder else root


def _pick_device(device: Optional[str]) -> torch.device:
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class MultiLabelAgent:
    """Runtime for a checkpoint written by `laya.multilabel.train`.

    Two checkpoint shapes are understood. A full one is an ordinary Laya checkpoint (same
    architecture, same files) whose `rl_agent_config.json` carries a `multilabel` section; it is
    loaded through the regular `laya.Agent`, kept as `self.agent`. An adapter-only one, written
    by LoRA runs, holds the LoRA adapter and the decision head; the backbone is fetched from its
    original hub id, the adapter merged into it, and `self.agent` is None.
    """

    def __init__(
        self,
        model_id_or_path: str,
        device: Optional[str] = None,
        token: Optional[str] = None,
        subfolder: Optional[str] = None,
    ):
        from transformers import AutoTokenizer

        model_dir = _resolve_dir(model_id_or_path, token, subfolder)
        cfg_path = os.path.join(model_dir, "rl_agent_config.json")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError("%r is not a Laya checkpoint (no rl_agent_config.json)" % model_id_or_path)
        with open(cfg_path) as f:
            cfg = json.load(f)
        ml = cfg.get("multilabel")
        if not ml:
            raise ValueError(
                "%r has no 'multilabel' section in rl_agent_config.json; train one with "
                "`python -m laya.multilabel train`." % model_id_or_path
            )
        self.cfg = cfg
        self.schema = LabelSchema.from_dict(ml)
        self.layout = ml.get("layout", "encoder")

        if ml.get("adapter"):
            from safetensors.torch import load_file

            base = ml["base"]
            self.device = _pick_device(device)
            load_dtype = base.get("load_dtype", "bf16") if self.device.type != "cpu" else "fp32"
            enc, _, _, _ = load_backbone(base["init"], base.get("subfolder"), token, load_dtype)
            _fix_tokenizer_config(model_dir)
            self.tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
            ensure_pad(self.tok)
            if enc.get_input_embeddings().num_embeddings < len(self.tok):
                enc.resize_token_embeddings(len(self.tok))
            enc = attach_adapter(enc, os.path.join(model_dir, ml["adapter"]), merge=True)
            self.model = DecisionModel(enc, cfg.get("head_layers", 0), len(cfg.get("act_costs", {})) + 1)
            head = load_file(os.path.join(model_dir, "model.safetensors"))
            missing = set(head_state_dict(self.model)) - set(head)
            if missing:
                raise ValueError("%r lacks decision-head weights %s" % (model_id_or_path, sorted(missing)[:3]))
            self.model.load_state_dict(head, strict=False)
            self.model.to(self.device).eval()
            self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
            self.agent = None
        else:
            self.agent = Agent(model_dir, device=device, token=token)
            self.model, self.tok, self.device, self.dtype = self.agent.model, self.agent.tok, self.agent.device, self.agent.dtype

        self.temperature = float(ml.get("temperature", 1.0))
        thr = ml.get("thresholds", {})
        self.thresholds = np.array([float(thr.get(n, 0.5)) for n in self.schema.names])
        marker = ml.get("marker_token") if self.layout == "causal" else None
        self.budget = Budget(
            max_len=cfg.get("max_len", 512),
            head_max_len=cfg.get("head_max_len", 192),
            labels_per_seq=int(ml.get("labels_per_seq", len(self.schema))),
            layout=self.layout,
            marker_id=self.tok.convert_tokens_to_ids(marker) if marker else None,
            bos_id=self.tok.bos_token_id if self.layout == "causal" else None,
        )

    def probabilities(self, states: Sequence[State], batch_size: int = 32) -> np.ndarray:
        """Calibrated P(label) for every state and label, [n, K] in `schema.names` order."""
        d = score_states(self.model, self.tok, self.schema, states, self.budget, self.device, self.dtype, batch_size)
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
