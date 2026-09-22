"""Figures for the ablation report, from results.jsonl and the trackio store.

    python plot_ablation.py --out runs/ablation --groups groups.ablation.json --assets assets \\
        --project laya-multilabel-ablation --curves A_laya_en_rlcd,A_modernbert_large_bare,...

Writes assets/ablation_test_f1.png (test micro-F1 and exact match per variant, grouped by axis)
and assets/ablation_curves.png (training loss and reward per step for the chosen runs).
"""
import argparse
import json
import os
import sqlite3

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BAR = "#2a78d6"
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASE)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def bars(rows, groups, path):
    order, labels, seps = [], [], []
    for title, names in groups.items():
        sel = [n for n in names if n in rows]
        if not sel:
            continue
        if order:
            seps.append(len(order) - 0.5)
        order += sel
        labels += [n for n in sel]
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.34 * len(order) + 1.2), sharey=True)
    y = list(range(len(order)))[::-1]
    for ax, key, title in zip(axes, ("test_micro_f1", "test_exact_match"), ("Test micro-F1", "Test exact match")):
        vals = [rows[n].get(key) or 0.0 for n in order]
        ax.barh(y, vals, height=0.62, color=BAR, linewidth=0)
        for yi, v in zip(y, vals):
            ax.text(v + 0.006, yi, "%.3f" % v, va="center", ha="left", fontsize=8.5, color=INK2)
        for s in seps:
            ax.axhline(len(order) - 1 - s, color=GRID, linewidth=0.8)
        ax.set_xlim(0, 1.08)
        ax.set_title(title, fontsize=11, color=INK, loc="left")
        style(ax)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def curves(db, names, path):
    con = sqlite3.connect(db)
    series = {}
    for name in names:
        pts = con.execute("select step, metrics from metrics where run_name = ? order by step, id", (name,)).fetchall()
        xs, loss, rew = [], [], []
        for step, m in pts:
            m = json.loads(m)
            if "train/loss_ce" in m:
                xs.append(step)
                loss.append(m["train/loss_ce"])
                rew.append(m.get("train/reward"))
        if xs:
            series[name] = (xs, loss, rew)
    con.close()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for (name, (xs, loss, rew)), color in zip(series.items(), CATEGORICAL):
        axes[0].plot(xs, loss, color=color, linewidth=2, label=name)
        axes[1].plot(xs, rew, color=color, linewidth=2, label=name)
    for ax, title in zip(axes, ("Soft cross-entropy per label decision (train)", "Proper-scoring reward (train)")):
        ax.set_title(title, fontsize=11, color=INK, loc="left")
        ax.set_xlabel("micro-batches", fontsize=9, color=MUTED)
        style(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.grid(axis="x", visible=False)
    axes[0].legend(frameon=False, fontsize=8.5, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="ablation output dir holding results.jsonl")
    ap.add_argument("--groups", required=True)
    ap.add_argument("--assets", required=True)
    ap.add_argument("--project", default="laya-multilabel-ablation")
    ap.add_argument("--trackio-dir", default=os.path.expanduser("~/.cache/huggingface/trackio"))
    ap.add_argument("--curves", default="", help="comma list of run names for the training curves")
    args = ap.parse_args()

    rows = {}
    with open(os.path.join(args.out, "results.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            rows[r["name"]] = r
    with open(args.groups, encoding="utf-8") as f:
        groups = json.load(f)
    os.makedirs(args.assets, exist_ok=True)
    bars(rows, groups, os.path.join(args.assets, "ablation_test_f1.png"))
    print("wrote", os.path.join(args.assets, "ablation_test_f1.png"))
    if args.curves:
        db = os.path.join(args.trackio_dir, args.project + ".db")
        curves(db, [c.strip() for c in args.curves.split(",") if c.strip()], os.path.join(args.assets, "ablation_curves.png"))
        print("wrote", os.path.join(args.assets, "ablation_curves.png"))


if __name__ == "__main__":
    main()
