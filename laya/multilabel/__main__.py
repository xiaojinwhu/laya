"""Command line: python -m laya.multilabel {train,eval,predict}."""
import argparse
import json
import os
import sys
from dataclasses import fields
from typing import Optional, get_type_hints

# see tests/test_local_e2e.py: a TensorFlow install can deadlock model construction
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from .trainer import TrainConfig, train  # noqa: E402


def _bool(v: str) -> bool:
    if v.lower() in ("1", "true", "yes", "y", "on"):
        return True
    if v.lower() in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError("expected true or false, got %r" % v)


def _add_config_args(p: argparse.ArgumentParser):
    hints = get_type_hints(TrainConfig)
    for f in fields(TrainConfig):
        t = hints[f.name]
        if getattr(t, "__origin__", None) is not None:  # Optional[X] -> X
            t = [a for a in t.__args__ if a is not type(None)][0]
        p.add_argument("--" + f.name.replace("_", "-"), dest=f.name, type=_bool if t is bool else t,
                       default=argparse.SUPPRESS, help="default: %r" % (f.default,))


def cmd_train(args):
    values = {}
    if args.config:
        with open(args.config, encoding="utf-8") as f:
            values.update(json.load(f))
    known = {f.name for f in fields(TrainConfig)}
    unknown = sorted(set(values) - known)
    if unknown:
        sys.exit("unknown keys in %s: %s" % (args.config, ", ".join(unknown)))
    values.update({k: v for k, v in vars(args).items() if k in known})  # flags win over the file
    train(TrainConfig(**values))


def cmd_eval(args):
    import numpy as np

    from .agent import MultiLabelAgent
    from .data import read_jsonl
    from .metrics import multilabel_metrics

    agent = MultiLabelAgent(args.model, device=args.device)
    examples = read_jsonl(args.data, agent.schema)
    p = agent.probabilities([e.state for e in examples], args.batch_size)
    y = np.array([e.target for e in examples], dtype=np.float32)
    thr = agent.thresholds if args.threshold is None else [args.threshold] * len(agent.schema)
    m = multilabel_metrics(p, y, thr, agent.schema.names)
    if args.output:
        with open(args.output, "w") as f:
            json.dump(m, f, indent=2, ensure_ascii=False)
    per_label = m.pop("per_label")
    print(json.dumps(m, indent=2))
    print("\n%-32s %9s %9s %9s %8s" % ("label", "precision", "recall", "f1", "support"))
    for name, s in per_label.items():
        print("%-32s %9.4f %9.4f %9.4f %8d" % (name[:32], s["precision"], s["recall"], s["f1"], s["support"]))


def cmd_predict(args):
    from .agent import MultiLabelAgent
    from .data import iter_rows

    agent = MultiLabelAgent(args.model, device=args.device)
    if args.text:
        for text, res in zip(args.text, agent.predict_batch(args.text, args.threshold)):
            top = sorted(res["probabilities"].items(), key=lambda kv: -kv[1])[:args.top]
            print("%s\n  -> %s  (confidence %.2f)\n     %s" % (
                text, ", ".join(res["labels"]) or "(none)", res["confidence"],
                "  ".join("%s %.2f" % kv for kv in top)))
        return
    if not args.input:
        sys.exit("give --text or --input")
    rows = list(iter_rows(args.input))
    states = [r["state"] if "state" in r else r["text"] for r in rows]
    out = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout
    for row, res in zip(rows, agent.predict_batch(states, args.threshold, args.batch_size)):
        out.write(json.dumps(dict(row, prediction=res), ensure_ascii=False) + "\n")
    if args.output:
        out.close()


def main(argv: Optional[list] = None):
    ap = argparse.ArgumentParser(prog="python -m laya.multilabel", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="RLCD fine-tuning; every TrainConfig field is a flag")
    t.add_argument("--config", help="JSON file of TrainConfig fields; flags override it")
    _add_config_args(t)
    t.set_defaults(fn=cmd_train)

    e = sub.add_parser("eval", help="score a labelled JSONL file with a trained checkpoint")
    e.add_argument("--model", required=True)
    e.add_argument("--data", required=True)
    e.add_argument("--output", help="write the full metrics JSON here")
    e.set_defaults(fn=cmd_eval)

    p = sub.add_parser("predict", help="label texts, or a JSONL file of {text|state} rows")
    p.add_argument("--model", required=True)
    p.add_argument("--text", nargs="+")
    p.add_argument("--input")
    p.add_argument("--output")
    p.add_argument("--top", type=int, default=5, help="probabilities to show per text")
    p.set_defaults(fn=cmd_predict)

    for s in (e, p):
        s.add_argument("--threshold", type=float, help="override the checkpoint's tuned thresholds")
        s.add_argument("--device")
        s.add_argument("--batch-size", type=int, default=32)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
