"""Post-training calibration: one temperature on the label logits, then decision thresholds."""
from typing import List, Optional

import numpy as np
import torch


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def fit_temperature(d: np.ndarray, y: np.ndarray, valid: Optional[np.ndarray] = None) -> float:
    """Temperature T minimising the log loss of sigmoid(d / T) against (soft) targets y.

    Same fit as the fine-tuning notebook's `fit_one_temp` (LBFGS on log T, clamped to
    [0.1, 10]), applied to the two-way [threshold, label] logits.
    """
    sel = np.ones_like(d, dtype=bool) if valid is None else valid
    if int(sel.sum()) < 10:
        return 1.0
    dd = torch.tensor(d[sel], dtype=torch.float32)
    yy = torch.tensor(y[sel], dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(dd / log_t.exp(), yy)
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.clamp(log_t.exp(), 0.1, 10.0).item())


def _f1_counts(pred: np.ndarray, gold: np.ndarray):
    tp = float((pred & gold).sum())
    return tp, float((pred & ~gold).sum()), float((~pred & gold).sum())


def _best_threshold(p: np.ndarray, gold: np.ndarray, grid: np.ndarray, default: float) -> float:
    best_f, best_t = -1.0, default
    for t in grid:
        tp, fp, fn = _f1_counts(p >= t, gold)
        f = 2 * tp / max(1.0, 2 * tp + fp + fn)
        # ties go to the threshold nearest the default, which is the better-calibrated choice
        if f > best_f + 1e-12 or (abs(f - best_f) <= 1e-12 and abs(t - default) < abs(best_t - default)):
            best_f, best_t = f, float(t)
    return best_t


def tune_thresholds(p: np.ndarray, y: np.ndarray, mode: str = "global", min_pos: int = 5) -> List[float]:
    """Decision thresholds per label, tuned for F1 on held-out probabilities `p` [n, K].

    fixed      0.5 everywhere
    global     one threshold maximising micro-F1
    per_label  each label maximises its own F1, falling back to the global threshold when the
               held-out set has fewer than `min_pos` positives for it
    """
    k = p.shape[1]
    if mode == "fixed":
        return [0.5] * k
    gold = y >= 0.5
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    g = _best_threshold(p.ravel(), gold.ravel(), grid, 0.5)
    if mode == "global":
        return [g] * k
    if mode != "per_label":
        raise ValueError("threshold mode must be fixed, global or per_label, got %r" % mode)
    return [_best_threshold(p[:, j], gold[:, j], grid, g) if int(gold[:, j].sum()) >= min_pos else g
            for j in range(k)]
