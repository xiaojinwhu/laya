"""Multi-label examples, token budgets, and sequence items.

Two sequence layouts, chosen by the backbone:

  encoder   Laya's own, from `laya.common.build_sequence`:
            [CLS] choice question: <ins> [SEP] [MASK] none: .. [MASK] label: .. [SEP] state [SEP]
            The marker precedes its option and reads it through bidirectional attention.
  causal    for decoders, where a position only sees what came before it:
            <bos>? state \\n choice question: <ins> \\n - none: .. <m> \\n - label: .. <m>
            The state comes first and the marker <m> follows its option, so by the time the
            backbone reaches a marker it has read the state, the question and that option.

In both, marker 0 is the threshold option and markers 1.. are the labels of the sequence.
"""
import json
import random
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

from ..common import QTYPES, build_sequence, render_options, serialize_state
from .schema import LabelSchema

OPTION_TOKEN_CAP = 48  # build_sequence keeps at most this many tokens of each option text
LAYOUTS = ("encoder", "causal")


@dataclass
class Example:
    state: Union[str, dict, list]
    target: List[float]  # dense over schema.names, each in [0, 1]
    uid: Optional[str] = None


@dataclass
class Budget:
    max_len: int
    head_max_len: int          # encoder layout: the option budget of build_sequence; causal: informational
    labels_per_seq: int
    layout: str = "encoder"
    marker_id: Optional[int] = None  # causal layout only
    bos_id: Optional[int] = None     # causal layout only; None when the tokenizer has no bos


def build_causal_sequence(
    tok,
    state: Union[str, dict, list],
    q: Dict,
    max_len: int,
    marker_id: int,
    bos_id: Optional[int] = None,
    option_order: Optional[List[int]] = None,
) -> Tuple[List[int], List[int]]:
    """Causal layout; same contract as `laya.common.build_sequence` (ids, marker positions)."""
    opts = render_options(q)
    order = option_order if option_order is not None else list(range(len(opts)))
    head_ids = tok("\nchoice question: %s" % q["ins"], add_special_tokens=False)["input_ids"]
    opt_ids = [tok("\n- " + opts[i], add_special_tokens=False)["input_ids"][:OPTION_TOKEN_CAP] + [marker_id]
               for i in order]
    prefix = [bos_id] if bos_id is not None else []
    room = max_len - len(prefix) - len(head_ids) - sum(len(o) for o in opt_ids)
    if room < 8:
        raise ValueError("max_len=%d leaves no room for the state next to %d options" % (max_len, len(opts)))
    st = tok(serialize_state(state), add_special_tokens=False)["input_ids"][:room]
    ids = prefix + st + head_ids
    markers = []
    for o in opt_ids:
        ids.extend(o)
        markers.append(len(ids) - 1)
    return ids, markers


class CachedTokenizer:
    """Tokenizer proxy that memoises `tok(text)`.

    `build_sequence` tokenizes the instructions and every option separately, and those strings
    are the same in every sequence; only the state is new. Anything else is forwarded.
    """

    def __init__(self, tok, maxsize: int = 8192):
        self._tok = tok
        self._ids = lru_cache(maxsize=maxsize)(
            lambda text: tuple(tok(text, add_special_tokens=False)["input_ids"]))

    def __call__(self, text: str, add_special_tokens: bool = False):
        if add_special_tokens:
            return self._tok(text, add_special_tokens=True)
        return {"input_ids": list(self._ids(text))}

    def __getattr__(self, name):
        return getattr(self._tok, name)


def _row_labels(row: Dict) -> Dict[str, float]:
    labels = row.get("labels", row.get("intents"))
    if labels is None:
        raise ValueError("row has no 'labels' field: %s" % json.dumps(row, ensure_ascii=False)[:200])
    if isinstance(labels, dict):
        return {str(k): float(v) for k, v in labels.items()}
    if isinstance(labels, str):
        labels = [labels]
    return {str(k): 1.0 for k in labels}


def iter_rows(path: str) -> Iterable[Dict]:
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError("%s:%d is not valid JSON (%s)" % (path, n, e))


def infer_labels(paths: Sequence[str]) -> Dict[str, None]:
    """Label set of a dataset that ships without a labels file: every name seen, sorted."""
    seen = set()
    for p in paths:
        for row in iter_rows(p):
            seen.update(_row_labels(row))
    return {n: None for n in sorted(seen)}


def read_jsonl(path: str, schema: LabelSchema) -> List[Example]:
    """One JSON object per line: {"text": ... | "state": ..., "labels": [names] | {name: prob}}."""
    out = []
    for i, row in enumerate(iter_rows(path)):
        state = row["state"] if "state" in row else row.get("text")
        if state is None:
            raise ValueError("%s row %d has neither 'text' nor 'state'" % (path, i + 1))
        target = [0.0] * len(schema)
        for name, p in _row_labels(row).items():
            if name not in schema.index:
                raise ValueError("%s row %d: label %r is not in the label schema" % (path, i + 1, name))
            target[schema.index[name]] = min(1.0, max(0.0, p))
        out.append(Example(state=state, target=target, uid=str(row.get("id", i))))
    return out


def _ntok(tok, text: str) -> int:
    return len(tok(text, add_special_tokens=False)["input_ids"])


def plan_budget(
    tok,
    schema: LabelSchema,
    max_positions: int,
    state_budget: int = 128,
    base_max_len: int = 512,
    base_head_max_len: int = 192,
    labels_per_seq: Optional[int] = None,
    layout: str = "encoder",
    marker_id: Optional[int] = None,
    bos_id: Optional[int] = None,
) -> Budget:
    """Size the sequence so no option is truncated, chunking labels if they cannot fit.

    Options share `head_max_len`; when they overflow it `build_sequence` cuts every option down to
    a few tokens and labels become indistinguishable (the Banking77 failure in the README). Each
    label is decided against the threshold option of its own sequence, so unlike a softmax `choice`
    the label set can be split across sequences without changing what a probability means.

    The causal layout has no option budget of its own; `max_len` is simply what the parts add up
    to, capped by the backbone's positions (decoders have tens of thousands, so chunking there is
    a memory choice made with `labels_per_seq`, not a necessity).
    """
    if layout not in LAYOUTS:
        raise ValueError("layout must be one of %s, got %r" % (LAYOUTS, layout))
    opts = render_options(schema.question())
    if layout == "causal":
        if marker_id is None:
            raise ValueError("the causal layout needs a marker_id")
        lens = [1 + min(OPTION_TOKEN_CAP, _ntok(tok, "\n- " + o)) for o in opts]
        fixed = (1 if bos_id is not None else 0) + _ntok(tok, "\nchoice question: " + schema.instructions)
    else:
        lens = [1 + min(OPTION_TOKEN_CAP, _ntok(tok, " " + o)) for o in opts]
        # build_sequence starts truncating once fewer than 16 tokens are left for the instructions
        fixed = max(16, _ntok(tok, "choice question: " + schema.instructions))
    none_len, lab_lens = lens[0], sorted(lens[1:], reverse=True)

    def head_need(c: int) -> int:
        return none_len + sum(lab_lens[:c]) + fixed

    c = len(lab_lens) if labels_per_seq is None else max(1, min(labels_per_seq, len(lab_lens)))
    while c > 1 and head_need(c) + 4 + state_budget > max_positions:
        c -= 1
    if head_need(c) + 4 + state_budget > max_positions:
        raise ValueError(
            "one label plus %d state tokens does not fit in %d positions; shorten the label "
            "descriptions or lower state_budget" % (state_budget, max_positions)
        )
    if layout == "causal":
        head = head_need(c)
        return Budget(max_len=min(max_positions, head + 4 + state_budget), head_max_len=head, labels_per_seq=c,
                      layout=layout, marker_id=marker_id, bos_id=bos_id)
    head = max(base_head_max_len, -(-head_need(c) // 8) * 8)
    head = min(head, max_positions - 4 - state_budget)
    max_len = min(max_positions, max(base_max_len, head + 4 + state_budget))
    return Budget(max_len=max_len, head_max_len=head, labels_per_seq=c)


def chunk_labels(n_labels: int, labels_per_seq: int, rng: Optional[random.Random] = None) -> List[List[int]]:
    """Label ids split into sequences; shuffled when `rng` is given (training), in order otherwise."""
    ids = list(range(n_labels))
    if rng is not None:
        rng.shuffle(ids)
    n_chunks = -(-n_labels // labels_per_seq)
    size = -(-n_labels // n_chunks)  # even chunks rather than one short tail
    return [ids[i:i + size] for i in range(0, n_labels, size)]


def build_items(
    tok,
    schema: LabelSchema,
    state: Union[str, dict, list],
    budget: Budget,
    target: Optional[Sequence[float]] = None,
    rng: Optional[random.Random] = None,
    ex: int = 0,
) -> List[Dict]:
    """Sequences for one example. Marker 0 is the threshold option, markers 1.. are `label_idx`."""
    items = []
    for chunk in chunk_labels(len(schema), budget.labels_per_seq, rng):
        if budget.layout == "causal":
            ids, markers = build_causal_sequence(
                tok, schema.state(state), schema.question(chunk), budget.max_len, budget.marker_id, budget.bos_id
            )
        else:
            ids, markers = build_sequence(
                tok, schema.state(state), schema.question(chunk), budget.max_len, budget.head_max_len
            )
        if len(markers) != len(chunk) + 1:
            raise ValueError(
                "%d labels do not fit in max_len=%d; raise max_len or lower labels_per_seq"
                % (len(chunk), budget.max_len)
            )
        item = {"ids": ids, "markers": markers, "qtype": QTYPES["choice"], "label_idx": chunk, "ex": ex}
        if target is not None:
            item["target"] = [0.0] + [float(target[j]) for j in chunk]
        items.append(item)
    return items
