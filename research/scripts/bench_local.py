"""Local extensive benchmark of the three Laya checkpoints.

Part A  MASSIVE intent across every language the dataset ships (~51), 20-option choice.
Part B  typed-decisions (400 cases / 2,000 decisions) on all three checkpoints, so the
        fine-tuned laya-typed-decisions can be compared with Jev's published 0.727 on the
        same benchmark.

Writes local_benchmark_results.json.

  USE_TF=0 python3 research/scripts/bench_local.py [--langs N] [--per-lang N] [--skip-a] [--skip-b]
      [--distractors random|scenario] [--criteria rewrite|key]
"""
import argparse
import gc
import json
import math
import os
import random
import re
import sys
import time

os.environ.setdefault("USE_TF", "0")          # TensorFlow's abseil runtime deadlocks model build
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import torch  # noqa: E402

RESEARCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(RESEARCH)
sys.path.insert(0, REPO)

import laya  # noqa: E402
from laya.common import QTYPES, build_sequence, collate_items, render_options, temp_bucket  # noqa: E402

# Local checkpoint dirs are used when present, otherwise the Hugging Face repo is downloaded.
ROOT = os.path.expanduser(os.environ.get("LAYA_MODELS", "~/laya_models"))
MODELS = {"english": (os.path.join(ROOT, "laya"), "convaiinnovations/laya"),
          "multilingual": (os.path.join(ROOT, "laya-multilingual"), "convaiinnovations/laya-multilingual"),
          "typed-decisions": (os.path.join(ROOT, "laya-typed-decisions"), "convaiinnovations/laya-typed-decisions")}
OUT = os.path.join(RESEARCH, "results", "local_benchmark_results.json")
TYPED_DECISIONS = os.environ.get(
    "TYPED_DECISIONS_PARQUET", os.path.join(RESEARCH, "typed-decisions", "all", "test-00000-of-00001.parquet"))
SEED, N_OPTS = 13, 20
N_BOOT = 1000


# ------------------------------------------------------------------ sampling
def sample_rows(rows, n, label_fn, seed=SEED):
    """Up to n rows, stratified by label: shuffle each label's rows, then take them round-robin.

    Hugging Face test splits are often sorted by label (banking77 is 40 rows per label in label
    order), so `rows[:n]` would evaluate only the first few labels.
    """
    rng = random.Random(seed)
    by_label = {}
    for r in rows:
        by_label.setdefault(label_fn(r), []).append(r)
    pools = [by_label[k] for k in sorted(by_label, key=str)]
    for p in pools:
        rng.shuffle(p)
    rng.shuffle(pools)
    out, i = [], 0
    while len(out) < n and any(i < len(p) for p in pools):
        out.extend(p[i] for p in pools if i < len(p))
        i += 1
    out = out[:n]
    rng.shuffle(out)
    return out


def pick_distractors(rng, gold, labels, n_opts, mode="random"):
    """gold + n_opts-1 distractors. mode="scenario" fills first from labels sharing gold's
    scenario prefix (MASSIVE `alarm_set` -> `alarm_*`), the confusable negatives in real traffic."""
    pool = [x for x in labels if x != gold]
    k = min(n_opts - 1, len(pool))
    if mode == "scenario":
        scen = gold.split("_", 1)[0]
        near = [x for x in pool if x.split("_", 1)[0] == scen]
        far = [x for x in pool if x.split("_", 1)[0] != scen]
        near = rng.sample(near, min(k, len(near)))
        keys = [gold] + near + rng.sample(far, k - len(near))
    else:
        keys = [gold] + rng.sample(pool, k)
    rng.shuffle(keys)
    return keys


# ------------------------------------------------------------------ engine
def to_internal(qdef):
    t = qdef["type"]
    crit = qdef.get("criteria")
    if t == "choice" and isinstance(crit, list):
        crit = {c: None for c in crit}
    ins = qdef["instructions"]
    return {"t": t, "ins": ins if isinstance(ins, str) else json.dumps(ins), "crit": crit}


@torch.no_grad()
def score_cases(agent, cases, max_tokens=8192, max_seqs=64, tag=""):
    max_len = agent.cfg.get("max_len", 512)
    hml = agent.cfg.get("head_max_len", 192)
    items, index, dropped = [], [], 0
    for ci, (state, questions) in enumerate(cases):
        for qid, qdef in questions.items():
            q = to_internal(qdef)
            try:
                ids, mk = build_sequence(agent.tok, state, q, max_len, hml)
            except Exception:
                index.append((ci, qid, QTYPES[q["t"]], 0)); items.append(None); dropped += 1; continue
            if len(mk) != len(render_options(q)):
                index.append((ci, qid, QTYPES[q["t"]], 0)); items.append(None); dropped += 1; continue
            items.append({"ids": ids, "markers": mk, "qtype": QTYPES[q["t"]]})
            index.append((ci, qid, QTYPES[q["t"]], len(mk)))
    order = sorted([i for i, it in enumerate(items) if it is not None],
                   key=lambda i: len(items[i]["ids"]))
    out = [None] * len(items)
    t0, done, i = time.time(), 0, 0
    while i < len(order):
        j, L = i, 0
        while j < len(order) and j - i < max_seqs and \
                max(L, len(items[order[j]]["ids"])) * (j - i + 1) <= max_tokens:
            L = max(L, len(items[order[j]]["ids"])); j += 1
        j = max(j, i + 1)
        sel = [items[order[t]] for t in range(i, j)]
        b = collate_items([sel], agent.tok.pad_token_id)
        lg, _ = agent.model(b["input_ids"].to(agent.device), b["attention_mask"].to(agent.device),
                            b["marker_pos"].to(agent.device), b["marker_mask"].to(agent.device),
                            b["qtype"].to(agent.device))
        lg = lg.float().cpu().numpy()
        for r in range(j - i):
            out[order[i + r]] = lg[r, :len(sel[r]["markers"])]
        done += j - i
        if tag and done % 500 < (j - i):
            el = time.time() - t0
            sys.stderr.write("\r   [%s] %d/%d %.0f q/s ETA %ds    "
                             % (tag, done, len(order), done / max(el, 1e-9),
                                int(el * (len(order) - done) / max(1, done))))
            sys.stderr.flush()
        i = j
    if tag:
        sys.stderr.write("\r" + " " * 70 + "\r")
    return out, index, time.time() - t0, dropped


def softmax_t(z, t=1.0):
    z = np.asarray(z, float) / max(1e-3, float(t))
    e = np.exp(z - z.max())
    return e / e.sum()


def temp_for(agent, qt, k):
    return float(agent.temperature_by_options.get(temp_bucket(qt, k), agent.temperature[qt]))


def ece_score(conf, corr, bins=15):
    conf, corr = np.asarray(conf, float), np.asarray(corr, float)
    if not len(conf):
        return float("nan")
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (conf > lo) & (conf <= hi)
        if s.any():
            e += s.mean() * abs(conf[s].mean() - corr[s].mean())
    return float(e)


def ece_equal_mass(conf, corr, bins=10):
    """ECE with equal-count bins; stabler than fixed-width bins at a few hundred samples."""
    conf, corr = np.asarray(conf, float), np.asarray(corr, float)
    if not len(conf):
        return float("nan")
    o = np.argsort(conf)
    return float(sum(len(b) / len(o) * abs(conf[b].mean() - corr[b].mean())
                     for b in np.array_split(o, min(bins, len(o))) if len(b)))


def bootstrap_ci(corr, n_boot=N_BOOT, seed=SEED):
    corr = np.asarray(corr, float)
    if not len(corr):
        return [float("nan"), float("nan")]
    rs = np.random.RandomState(seed)
    means = corr[rs.randint(0, len(corr), (n_boot, len(corr)))].mean(1)
    return [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]


def mcnemar(corr_a, corr_b):
    """Exact two-sided McNemar test on paired per-item correctness (same questions, two models)."""
    a, b = np.asarray(corr_a, bool), np.asarray(corr_b, bool)
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    n = n01 + n10
    if n == 0:
        return {"a_only": n10, "b_only": n01, "p_value": 1.0}
    k = min(n01, n10)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"a_only": n10, "b_only": n01, "p_value": round(p, 6)}


def correctness(rows):
    """Per-item 0/1 correctness; a dropped question (probs None) counts as wrong."""
    return np.array([0.0 if p is None else float(int(np.argmax(p)) == g) for g, p in rows])


def macro_f1(g, p):
    g, p = np.asarray(g), np.asarray(p)
    f = []
    for c in sorted(set(g.tolist()) | set(p.tolist())):
        tp = int(((p == c) & (g == c)).sum()); fp = int(((p == c) & (g != c)).sum())
        fn = int(((p != c) & (g == c)).sum())
        f.append(2 * tp / max(1, 2 * tp + fp + fn))
    return float(np.mean(f))


def metrics(rows):
    """Accuracy (with bootstrap 95% CI) counts dropped questions as wrong; calibration metrics
    are over the scored questions only."""
    all_corr = correctness(rows)
    n_dropped = sum(1 for r in rows if r[1] is None)
    rows = [r for r in rows if r[1] is not None]
    if not rows:
        return {"n": 0, "n_dropped": n_dropped}
    g = np.array([x[0] for x in rows]); p = np.array([int(np.argmax(x[1])) for x in rows])
    c = np.array([float(np.max(x[1])) for x in rows]); corr = (p == g).astype(float)
    return {"n": len(rows) + n_dropped, "n_dropped": n_dropped,
            "accuracy": round(float(all_corr.mean()), 4),
            "accuracy_ci95": bootstrap_ci(all_corr),
            "accuracy_scored_only": round(float(corr.mean()), 4),
            "ece_equal_mass": round(ece_equal_mass(c, corr), 4),
            "macro_f1": round(macro_f1(g, p), 4), "ece": round(ece_score(c, corr), 4),
            "brier": round(float(np.mean([((np.asarray(x[1]) - np.eye(len(x[1]))[x[0]]) ** 2).sum()
                                          for x in rows])), 4),
            "nll": round(float(np.mean([-math.log(max(float(x[1][x[0]]), 1e-12)) for x in rows])), 4),
            "mean_confidence": round(float(c.mean()), 4),
            "acc_at_50_coverage": round(float(corr[np.argsort(-c)[:max(1, len(c)//2)]].mean()), 4)}


def load(name):
    local, repo = MODELS[name]
    ag = laya.load(local if os.path.isdir(local) else repo, device="cpu")
    ag.model.eval()
    return ag


def option_flip_rate(agent, cases, limit=200, seed=SEED):
    """Fraction of choice questions whose chosen label changes when the options are permuted."""
    rng = random.Random(seed)
    base, perm = [], []
    for state, qs in cases[:limit]:
        for qid, qd in qs.items():
            if qd["type"] != "choice" or not isinstance(qd.get("criteria"), dict):
                continue
            keys = list(qd["criteria"])
            shuffled = keys[:]
            rng.shuffle(shuffled)
            base.append((state, {qid: qd}))
            perm.append((state, {qid: {**qd, "criteria": {k: qd["criteria"][k] for k in shuffled}}}))
    if not base:
        return None
    la, ia, _, _ = score_cases(agent, base)
    lb, ib, _, _ = score_cases(agent, perm)
    flips, n = 0, 0
    for (sa, qa), (sb, qb), za, zb in zip(base, perm, la, lb):
        if za is None or zb is None:
            continue
        qid = next(iter(qa))
        flips += list(qa[qid]["criteria"])[int(np.argmax(za))] != list(qb[qid]["criteria"])[int(np.argmax(zb))]
        n += 1
    return {"n": n, "flip_rate": round(flips / max(1, n), 4)}


# ------------------------------------------------------------------ part A
def massive_languages():
    from huggingface_hub import HfApi
    files = [s.rfilename for s in (HfApi().dataset_info("mteb/amazon_massive_intent").siblings or [])]
    return sorted({m.group(1) for f in files for m in [re.match(r"test/([A-Za-z\-]+)\.json", f)] if m})


def render_intent_criteria(keys, style):
    if style == "key":
        return {k: None for k in keys}
    return {k: k.replace("_", " ").replace(".", ": ") for k in keys}


def build_massive(langs, per_lang, distractors="random", criteria="rewrite"):
    from datasets import load_dataset
    suites, en_ids = {}, None
    for lg in sorted(langs, key=lambda x: x != "en"):
        try:
            d = load_dataset("mteb/amazon_massive_intent", lg, split="test")
            labels = sorted(set(d["label_text"]))
            rng = random.Random(SEED)
            rows = list(d)
            # MASSIVE is parallel: evaluate the same utterance ids in every language
            if en_ids is not None and "id" in d.column_names:
                keep = {r["id"]: r for r in rows}
                picked = [keep[i] for i in en_ids if i in keep]
            else:
                picked = sample_rows(rows, per_lang, lambda r: r["label_text"])
                if lg == "en" and "id" in d.column_names:
                    en_ids = [r["id"] for r in picked]
            cases, gold = [], []
            for r in picked:
                keys = pick_distractors(rng, r["label_text"], labels, N_OPTS, distractors)
                cases.append(({"utterance": r["text"]},
                              {"intent": {"type": "choice",
                                          "instructions": "What is the user asking for in `utterance`?",
                                          "criteria": render_intent_criteria(keys, criteria)}}))
                gold.append(keys.index(r["label_text"]))
            suites[lg] = (cases, gold, len(labels))
            print("   built %-8s %d cases (%d labels)" % (lg, len(cases), len(labels)), flush=True)
        except Exception as e:
            print("   FAIL %-8s %s" % (lg, str(e)[:70]), flush=True)
    return suites


def run_part_a(results, langs, per_lang, distractors="random", criteria="rewrite"):
    print("\n=== PART A: MASSIVE intent, %d languages, %d cases each, %d options (%s distractors, %s criteria) ===\n"
          % (len(langs), per_lang, N_OPTS, distractors, criteria), flush=True)
    suites = build_massive(langs, per_lang, distractors, criteria)
    results["part_a"] = {"config": {"languages": sorted(suites), "per_lang": per_lang,
                                    "n_options": N_OPTS, "seed": SEED, "distractors": distractors,
                                    "criteria": criteria, "sampling": "stratified by label, parallel ids"},
                         "by_model": {}, "paired_english_vs_multilingual": {}}
    corr_by = {}
    for mname in ("english", "multilingual"):
        print("\n--- %s ---" % mname, flush=True)
        ag = load(mname)
        per = {}
        for lg, (cases, gold, _) in sorted(suites.items()):
            lgs, idx, secs, dropped = score_cases(ag, cases, tag="%s/%s" % (mname, lg))
            rows = [(gold[ci], softmax_t(z, temp_for(ag, qt, k)) if z is not None else None)
                    for (ci, _, qt, k), z in zip(idx, lgs)]
            m = metrics(rows); m["seconds"] = round(secs, 1); m["dropped"] = dropped
            corr_by.setdefault(mname, {})[lg] = correctness(rows)
            if lg == "en":
                m["option_order"] = option_flip_rate(ag, cases)
            per[lg] = m
            print("   %-8s acc %.3f  f1 %.3f  ECE %.3f  conf %.3f  (%.0f q/s)"
                  % (lg, m["accuracy"], m["macro_f1"], m["ece"], m["mean_confidence"],
                     m["n"] / max(secs, 1e-9)), flush=True)
        accs = [v["accuracy"] for v in per.values()]
        results["part_a"]["by_model"][mname] = {
            "per_language": per,
            "macro_accuracy": round(float(np.mean(accs)), 4),
            "macro_ece": round(float(np.mean([v["ece"] for v in per.values()])), 4),
            "n_languages": len(per),
            "languages_above_random": int(sum(1 for a in accs if a > 3.0 / N_OPTS)),
        }
        print("   MACRO acc %.4f | ECE %.4f | %d/%d langs > 3x random"
              % (results["part_a"]["by_model"][mname]["macro_accuracy"],
                 results["part_a"]["by_model"][mname]["macro_ece"],
                 results["part_a"]["by_model"][mname]["languages_above_random"], len(per)), flush=True)
        del ag; gc.collect()
        json.dump(results, open(OUT, "w"), indent=2)
    for lg in sorted(suites):
        results["part_a"]["paired_english_vs_multilingual"][lg] = mcnemar(
            corr_by["english"][lg], corr_by["multilingual"][lg])
    json.dump(results, open(OUT, "w"), indent=2)


# ------------------------------------------------------------------ part B
def build_typed_decisions():
    import pandas as pd
    df = pd.read_parquet(TYPED_DECISIONS)
    cases, gold, wfs = [], [], []
    for _, r in df.iterrows():
        qs = json.loads(r["questions"]); g = json.loads(r["gold"])
        st = r["state"]
        try:
            st = json.loads(st)
        except Exception:
            pass
        gm = {}
        for qid, qd in qs.items():
            gg = g[qid]
            if qd["type"] == "choice":
                keys = list(qd["criteria"].keys())
                gm[qid] = {"idx": keys.index(str(gg["label"])),
                           "soft": [float(gg.get("probabilities", {}).get(k, 0.0)) for k in keys]}
            elif qd["type"] == "noul":
                pt = float(gg.get("probabilities", {}).get("true", gg.get("noul", 0.5)))
                gm[qid] = {"idx": 1 if str(gg["label"]).lower() == "true" else 0, "soft": [1 - pt, pt]}
            else:
                n = len(qd["criteria"])
                gm[qid] = {"idx": int(gg["label"]),
                           "soft": [float(gg.get("probabilities", {}).get(str(i), 0.0)) for i in range(n)],
                           "gold_score": float(gg.get("score", float(gg["label"])))}
        cases.append((st, qs)); gold.append(gm); wfs.append(r["workflow"])
    return cases, gold, wfs


def run_part_b(results):
    print("\n=== PART B: typed-decisions, 400 cases / 2,000 decisions, all 3 checkpoints ===\n",
          flush=True)
    cases, gold, wfs = build_typed_decisions()
    # reference points that are NOT measured here - published / independently measured
    results["part_b"] = {"reference_points": {
        "jev_1.13.0_published": {"accuracy": 0.727, "soft_accuracy": 0.580, "brier": 0.148,
                                 "ece": 0.144, "score_mae": 0.391, "ms_per_case": 710,
                                 "source": "figure quoted in the laya repo's own comparison table"},
        "teacher_self_agreement_ceiling": {"accuracy": 0.735},
        "modernbert_base_specialist": {"accuracy": 0.646},
        "random_guess": {"accuracy": 0.3175},
        "per_question_majority_class": {"accuracy": 0.4610},
        "note": "Jev was NOT run here - no TypeSafe API credential is available. These are "
                "published numbers reproduced for context, not a measured head-to-head.",
    }, "by_model": {}}
    for mname in ("english", "multilingual", "typed-decisions"):
        print("--- %s ---" % mname, flush=True)
        ag = load(mname)
        lgs, idx, secs, dropped = score_cases(ag, cases, tag=mname)
        rows, soft, brier_s, mae, w1, by_wf, by_qt = [], [], [], [], [], {}, {}
        for (ci, qid, qt, k), z in zip(idx, lgs):
            g = gold[ci][qid]
            if z is None:
                rows.append((g["idx"], None)); continue
            p = softmax_t(z, temp_for(ag, qt, k))
            rows.append((g["idx"], p))
            by_wf.setdefault(wfs[ci], []).append((g["idx"], p))
            by_qt.setdefault({0: "choice", 1: "score", 2: "noul"}[qt], []).append((g["idx"], p))
            gp = np.asarray(g["soft"], float)
            if gp.sum() > 0:
                gp = gp / gp.sum()
                pp = p[:len(gp)] if len(p) >= len(gp) else np.pad(p, (0, len(gp) - len(p)))
                pp = pp / max(pp.sum(), 1e-12)
                soft.append(float((pp * gp).sum())); brier_s.append(float(((pp - gp) ** 2).sum()))
            if "gold_score" in g:
                exp = float((np.arange(len(p)) * p).sum())
                mae.append(abs(exp - g["gold_score"])); w1.append(float(abs(exp - g["gold_score"]) <= 1))
        m = metrics(rows)
        m.update(soft_accuracy=round(float(np.mean(soft)), 4) if soft else None,
                 brier_vs_soft=round(float(np.mean(brier_s)), 4) if brier_s else None,
                 score_mae=round(float(np.mean(mae)), 4) if mae else None,
                 within_1_level=round(float(np.mean(w1)), 4) if w1 else None,
                 seconds=round(secs, 1), dropped=dropped,
                 ms_per_case=round(1000 * secs / len(cases), 1),
                 by_workflow={k: metrics(v) for k, v in sorted(by_wf.items())},
                 by_question_type={k: metrics(v) for k, v in sorted(by_qt.items())})
        results["part_b"]["by_model"][mname] = m
        print("   acc %.4f | soft %.4f | brier(soft) %s | ECE %.4f | MAE %s | %.0f ms/case"
              % (m["accuracy"], m["soft_accuracy"] or 0, m["brier_vs_soft"], m["ece"],
                 m["score_mae"], m["ms_per_case"]), flush=True)
        for wf, v in m["by_workflow"].items():
            print("      %-28s acc %.3f (n=%d)" % (wf, v["accuracy"], v["n"]), flush=True)
        del ag; gc.collect()
        json.dump(results, open(OUT, "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", type=int, default=0, help="cap number of languages (0 = all)")
    ap.add_argument("--per-lang", type=int, default=120)
    ap.add_argument("--skip-a", action="store_true")
    ap.add_argument("--skip-b", action="store_true")
    ap.add_argument("--distractors", choices=("random", "scenario"), default="random",
                    help="scenario = same-scenario confusable intents first")
    ap.add_argument("--criteria", choices=("rewrite", "key"), default="rewrite",
                    help="option text: rewritten label or bare key (description ablation)")
    a = ap.parse_args()

    results = {"meta": {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "device": "cpu",
                        "torch": torch.__version__, "laya": laya.__version__,
                        "threads": torch.get_num_threads()}}
    if os.path.exists(OUT):
        try:
            results = json.load(open(OUT)); results["meta"]["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass
    if not a.skip_a:
        langs = massive_languages()
        if a.langs:
            langs = langs[:a.langs]
        run_part_a(results, langs, a.per_lang, a.distractors, a.criteria)
    if not a.skip_b:
        run_part_b(results)
    json.dump(results, open(OUT, "w"), indent=2)
    print("\nwrote %s" % OUT, flush=True)


if __name__ == "__main__":
    main()
