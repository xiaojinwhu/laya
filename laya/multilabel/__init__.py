"""Multi-label classification on Laya: every label decided in one forward pass, trained with RLCD.

    python -m laya.multilabel train --train-file train.jsonl --dev-file dev.jsonl --output-dir out
    python -m laya.multilabel predict --model out --text "play some jazz and book a table for two"
"""
from .agent import MultiLabelAgent, load, score_states
from .calibration import fit_temperature, tune_thresholds
from .data import Budget, Example, build_items, plan_budget, read_jsonl
from .metrics import multilabel_metrics
from .rlcd import label_logits, multilabel_pairs, multilabel_rlcd_loss, rlcd_loss
from .schema import LabelSchema, load_labels
from .trainer import TrainConfig, train

__all__ = [
    "MultiLabelAgent",
    "load",
    "score_states",
    "LabelSchema",
    "load_labels",
    "Example",
    "Budget",
    "read_jsonl",
    "plan_budget",
    "build_items",
    "rlcd_loss",
    "multilabel_rlcd_loss",
    "multilabel_pairs",
    "label_logits",
    "fit_temperature",
    "tune_thresholds",
    "multilabel_metrics",
    "TrainConfig",
    "train",
]
