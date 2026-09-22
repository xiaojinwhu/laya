"""Run a grid of `laya.multilabel` training variants and tabulate them.

    python ablate.py run    --base config.mixsnips.json --grid grid.json --out runs/ablation
    python ablate.py run    --base config.mixsnips.json --out runs/ablation \\
        --set bce rl_weight=0 --set laya_lora lora_r=16 --set qwen init=Qwen/Qwen3-0.6B-Base,lora_r=16
    python ablate.py report --out runs/ablation

A grid file is a JSON list of {"name": ..., "overrides": {TrainConfig field: value}}. Every
variant trains in its own process (`python -m laya.multilabel train --config BASE --<override> ..`)
so memory is returned between runs, writes to OUT/NAME, and is skipped when OUT/NAME/metrics.json
already exists (delete it to rerun). `report` gathers the metrics into results.jsonl + results.md.
"""
import argparse
import json
import os
import subprocess
import sys
import time

COLUMNS = [
    ("name", "variant", "{}"),
    ("init", "init", "{}"),
    ("layout", "layout", "{}"),
    ("lora_r", "lora", "{}"),
    ("params_m", "params (M)", "{:.0f}"),
    ("trainable_m", "trainable (M)", "{:.1f}"),
    ("dev_micro_f1", "dev F1", "{:.4f}"),
    ("test_micro_f1", "test F1", "{:.4f}"),
    ("test_macro_f1", "test macro-F1", "{:.4f}"),
    ("test_exact_match", "exact", "{:.4f}"),
    ("test_ece", "ECE", "{:.4f}"),
    ("best_epoch", "best ep", "{}"),
    ("train_minutes", "train min", "{:.1f}"),
    ("peak_memory_gb", "peak GB", "{:.1f}"),
    ("ms_batch1", "ms/ex (b=1)", "{:.0f}"),
    ("ms_batched", "ms/ex (batched)", "{:.1f}"),
]


def parse_value(v: str):
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    if v.lower() in ("none", "null"):
        return None
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def parse_set(spec: str):
    """`name key=value,key=value` -> {"name": name, "overrides": {...}}."""
    name, _, rest = spec.partition(" ")
    if not rest:
        return {"name": name, "overrides": {}}
    ov = {}
    for kv in rest.split(","):
        k, _, v = kv.partition("=")
        ov[k.strip().replace("-", "_")] = parse_value(v.strip())
    return {"name": name, "overrides": ov}


def flags(overrides: dict):
    out = []
    for k, v in overrides.items():
        if v is None:
            continue
        out += ["--" + k.replace("_", "-"), str(v).lower() if isinstance(v, bool) else str(v)]
    return out


def run_variant(base: str, variant: dict, out_dir: str, python: str, dry: bool) -> dict:
    name, ov = variant["name"], dict(variant.get("overrides", {}))
    vdir = os.path.join(out_dir, name)
    if os.path.exists(os.path.join(vdir, "metrics.json")):
        print("== %s: done already, skipping" % name, flush=True)
        return {"name": name, "status": "cached"}
    cmd = [python, "-m", "laya.multilabel", "train", "--config", base] + flags(ov) + ["--output-dir", vdir]
    print("== %s\n   %s" % (name, " ".join(cmd)), flush=True)
    if dry:
        return {"name": name, "status": "dry"}
    os.makedirs(vdir, exist_ok=True)
    t0 = time.time()
    env = dict(os.environ, PYTHONUNBUFFERED="1", USE_TF="0", TOKENIZERS_PARALLELISM="false")
    with open(os.path.join(vdir, "train.log"), "w") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        for line in proc.stdout:
            log.write(line)
            if line.startswith(("init:", "data:", "sequence:", "schedule:", "epoch", "calibration", "dev ", "test ", "saved", "Traceback")):
                print("   " + line.rstrip(), flush=True)
        rc = proc.wait()
    status = "ok" if rc == 0 and os.path.exists(os.path.join(vdir, "metrics.json")) else "failed (rc=%s)" % rc
    print("   -> %s in %.1f min" % (status, (time.time() - t0) / 60), flush=True)
    return {"name": name, "status": status}


def collect(out_dir: str, names=None):
    rows = []
    for name in sorted(os.listdir(out_dir)) if names is None else names:
        mp = os.path.join(out_dir, name, "metrics.json")
        cp = os.path.join(out_dir, name, "rl_agent_config.json")
        if not os.path.exists(mp):
            continue
        with open(mp) as f:
            m = json.load(f)
        run, cfg = m.get("run", {}), {}
        if os.path.exists(cp):
            with open(cp) as f:
                cfg = json.load(f).get("training", {}).get("config", {})
        row = {"name": name, "init": run.get("init", cfg.get("init")), "layout": run.get("layout"),
               "lora_r": run.get("lora_r", cfg.get("lora_r")), "params_m": run.get("params_m"),
               "trainable_m": run.get("trainable_m"), "best_epoch": m.get("best_epoch"),
               "train_minutes": run.get("train_minutes"), "peak_memory_gb": run.get("peak_memory_gb"),
               "ms_batch1": run.get("ms_per_example_batch1"),
               "ms_batched": next((v for k, v in run.items() if k.startswith("ms_per_example_batch") and not k.endswith("batch1")), None),
               "rl_weight": cfg.get("rl_weight"), "epochs": cfg.get("epochs"), "limit_train": cfg.get("limit_train")}
        for split in ("dev", "test"):
            for key in ("micro_f1", "macro_f1", "exact_match", "ece", "brier"):
                row["%s_%s" % (split, key)] = m.get(split, {}).get(key)
        rows.append(row)
    return rows


def fmt(row: dict, key: str, spec: str) -> str:
    v = row.get(key)
    if v is None:
        return "–"
    try:
        return spec.format(v)
    except (ValueError, TypeError):
        return str(v)


def write_report(out_dir: str, rows):
    with open(os.path.join(out_dir, "results.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    lines = ["| " + " | ".join(h for _, h, _ in COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    for r in rows:
        lines.append("| " + " | ".join(fmt(r, k, s) for k, _, s in COLUMNS) + " |")
    table = "\n".join(lines)
    with open(os.path.join(out_dir, "results.md"), "w") as f:
        f.write("# Ablation results\n\n%s\n" % table)
    print("\n" + table)
    print("\nwritten: %s/results.md, results.jsonl" % out_dir)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--base", required=True, help="TrainConfig JSON every variant starts from")
    r.add_argument("--grid", help="JSON list of {name, overrides}")
    r.add_argument("--set", action="append", default=[], metavar="'NAME key=value,key=value'",
                   help="add a variant inline; repeatable")
    r.add_argument("--out", required=True)
    r.add_argument("--python", default=sys.executable)
    r.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("report")
    p.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.cmd == "report":
        write_report(args.out, collect(args.out))
        return
    variants = []
    if args.grid:
        with open(args.grid) as f:
            variants += json.load(f)
    variants += [parse_set(s) for s in args.set]
    if not variants:
        sys.exit("no variants: give --grid and/or --set")
    os.makedirs(args.out, exist_ok=True)
    statuses = [run_variant(args.base, v, args.out, args.python, args.dry_run) for v in variants]
    if not args.dry_run:
        write_report(args.out, collect(args.out, [v["name"] for v in variants]))
        failed = [s["name"] for s in statuses if s["status"].startswith("failed")]
        if failed:
            print("FAILED: " + ", ".join(failed))
            sys.exit(1)


if __name__ == "__main__":
    main()
