"""Multi-label classification and calibration metrics (numpy only)."""
from typing import Dict, List, Sequence

import numpy as np

from ..common import ece_score


def _prf(tp: float, fp: float, fn: float):
    p = tp / max(1.0, tp + fp)
    r = tp / max(1.0, tp + fn)
    return p, r, 2 * tp / max(1.0, 2 * tp + fp + fn)


def multilabel_metrics(p: np.ndarray, y: np.ndarray, thresholds: Sequence[float], names: List[str]) -> Dict:
    """p: probabilities [n, K]; y: gold in [0, 1] (soft gold is binarised at 0.5 for the F1 family).

    Calibration is reported twice: `ece` over the n*K individual label decisions, and `set_ece`
    over whole predictions, whose confidence is the product of its label decisions'.
    """
    thr = np.asarray(thresholds, dtype=float)[None, :]
    gold, pred = y >= 0.5, p >= thr
    tp, fp, fn = (pred & gold).sum(0), (pred & ~gold).sum(0), (~pred & gold).sum(0)

    micro_p, micro_r, micro_f = _prf(float(tp.sum()), float(fp.sum()), float(fn.sum()))
    per_label = {}
    for j, n in enumerate(names):
        lp, lr, lf = _prf(float(tp[j]), float(fp[j]), float(fn[j]))
        per_label[n] = {"precision": round(lp, 4), "recall": round(lr, 4), "f1": round(lf, 4),
                        "support": int(gold[:, j].sum())}
    seen = [n for n in names if per_label[n]["support"] > 0]
    exact = (pred == gold).all(1)

    pc = np.clip(p, 1e-7, 1 - 1e-7)
    decision_conf = np.where(pred, p, 1.0 - p)
    return {
        "n": int(p.shape[0]),
        "micro_f1": round(micro_f, 4),
        "micro_precision": round(micro_p, 4),
        "micro_recall": round(micro_r, 4),
        "macro_f1": round(float(np.mean([per_label[n]["f1"] for n in seen])) if seen else 0.0, 4),
        "exact_match": round(float(exact.mean()), 4),
        "hamming_loss": round(float((pred != gold).mean()), 4),
        "ece": round(ece_score(decision_conf.ravel(), (pred == gold).ravel().astype(float)), 4),
        "set_ece": round(ece_score(decision_conf.prod(1), exact.astype(float)), 4),
        "brier": round(float(((p - y) ** 2).mean()), 4),
        "nll": round(float(-(y * np.log(pc) + (1 - y) * np.log(1 - pc)).mean()), 4),
        "labels_per_example": {"gold": round(float(gold.sum(1).mean()), 3),
                               "predicted": round(float(pred.sum(1).mean()), 3)},
        "per_label": per_label,
    }
