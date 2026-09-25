"""Label schema for the multi-label primitive: which labels exist and how the question is asked."""
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Union

DEFAULT_INSTRUCTIONS = "Which intents does the user express in `utterance`? Several may apply."
DEFAULT_NONE_KEY = "none"
DEFAULT_NONE_DESC = "no listed intent is expressed"


@dataclass
class LabelSchema:
    """The label set plus the fixed wording of the multi-label question.

    A multi-label question is laid out exactly like a Laya `choice` question, with one extra
    option in front: the threshold option (`none_key`). Every label is then decided against
    that option, so the sequence reads

        [CLS] choice question: <instructions> [SEP] [MASK] none: ... [MASK] label_0: ... [SEP] state [SEP]
    """

    labels: Dict[str, Optional[str]]
    instructions: str = DEFAULT_INSTRUCTIONS
    none_key: str = DEFAULT_NONE_KEY
    none_desc: Optional[str] = DEFAULT_NONE_DESC
    state_key: str = "utterance"

    def __post_init__(self):
        if not self.labels:
            raise ValueError("LabelSchema needs at least one label")
        if self.none_key in self.labels:
            raise ValueError(
                "label %r collides with the threshold option; pass a different none_key" % self.none_key
            )
        self.names: List[str] = list(self.labels)
        self.index: Dict[str, int] = {n: i for i, n in enumerate(self.names)}

    def __len__(self) -> int:
        return len(self.names)

    def question(self, label_ids: Optional[Sequence[int]] = None) -> Dict:
        """Internal `choice` question over the threshold option followed by `label_ids` in order."""
        ids = range(len(self.names)) if label_ids is None else label_ids
        crit = {self.none_key: self.none_desc}
        for j in ids:
            crit[self.names[j]] = self.labels[self.names[j]]
        return {"t": "choice", "ins": self.instructions, "crit": crit}

    def state(self, x: Union[str, dict, list]) -> Union[dict, list]:
        """A bare string is wrapped under `state_key` so the instructions can refer to it."""
        return {self.state_key: x} if isinstance(x, str) else x

    def to_dict(self) -> Dict:
        return {
            "labels": dict(self.labels),
            "instructions": self.instructions,
            "none_key": self.none_key,
            "none_desc": self.none_desc,
            "state_key": self.state_key,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "LabelSchema":
        return cls(
            labels=dict(d["labels"]),
            instructions=d.get("instructions", DEFAULT_INSTRUCTIONS),
            none_key=d.get("none_key", DEFAULT_NONE_KEY),
            none_desc=d.get("none_desc", DEFAULT_NONE_DESC),
            state_key=d.get("state_key", "utterance"),
        )


def load_labels(path: str) -> Dict:
    """Read a labels file: a list of names, a {name: description} map, or a full schema dict."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, list):
        return {"labels": {str(n): None for n in raw}}
    if isinstance(raw, dict) and isinstance(raw.get("labels"), (dict, list)):
        labels = raw["labels"]
        if isinstance(labels, list):
            labels = {str(n): None for n in labels}
        return dict(raw, labels=labels)
    if isinstance(raw, dict):
        return {"labels": raw}
    raise ValueError("%s: expected a list of label names or a {name: description} object" % path)
