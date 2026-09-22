"""RLCD fine-tuning pipeline for multi-label classification.

Follows `notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb`: two learning rates
(encoder / decision head), cosine decay, gradient accumulation, annealed exploration noise,
DDP under `torchrun`, post-training temperature fitting, and a checkpoint in the standard Laya
layout. What it adds is a held-out split for model selection and calibration, decision
thresholds, and the multi-label question itself (see `rlcd.py`).
"""
import contextlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ..agent import _fix_tokenizer_config
from ..common import build_model, collate_items
from .agent import score_states
from .calibration import fit_temperature, sigmoid, tune_thresholds
from .data import Budget, CachedTokenizer, Example, build_items, infer_labels, plan_budget, read_jsonl
from .metrics import multilabel_metrics
from .rlcd import multilabel_rlcd_loss
from .schema import LabelSchema, load_labels

CHECKPOINT_FILES = ["rl_agent_config.json", "model.safetensors", "encoder/*", "tokenizer/*"]


@dataclass
class TrainConfig:
    # data
    train_file: str = ""
    dev_file: Optional[str] = None      # held out for model selection and calibration
    test_file: Optional[str] = None
    labels_file: Optional[str] = None   # label names/descriptions; inferred from the data if absent
    dev_ratio: float = 0.1              # carved from train when there is no dev_file
    limit_train: Optional[int] = None
    limit_eval: Optional[int] = None
    output_dir: str = "laya_multilabel"

    # what to start from: a Laya checkpoint (hub id or directory) or any HF masked-LM encoder
    init: str = "convaiinnovations/laya"
    subfolder: Optional[str] = None
    head_layers: int = 2                # only used when `init` is a bare encoder

    # question wording; each overrides the labels file
    instructions: Optional[str] = None
    none_key: Optional[str] = None
    none_desc: Optional[str] = None
    state_key: Optional[str] = None

    # sequence budget; None = sized from the label set
    max_len: Optional[int] = None
    head_max_len: Optional[int] = None
    state_budget: int = 128
    labels_per_seq: Optional[int] = None
    shuffle_labels: bool = True         # new option order every epoch

    # RLCD (defaults are the fine-tuning notebook's)
    epochs: int = 4
    micro_batch: int = 8
    grad_accum: int = 4
    group_size: int = 4
    lr_encoder: float = 2.5e-5
    lr_head: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.0
    max_grad_norm: float = 1.0
    sigma_start: float = 0.4
    sigma_end: float = 0.1
    w_sph: float = 0.75
    rl_weight: float = 1.0
    ce_weight: float = 1.0
    pos_weight: float = 1.0
    freeze_encoder: bool = False
    gradient_checkpointing: bool = True  # as in the notebook: full fine-tuning of 421M params on 16 GB

    # selection, calibration, runtime
    threshold_mode: str = "global"      # fixed | global | per_label
    patience: int = 0                   # stop after this many epochs without dev improvement; 0 = off
    eval_before_train: bool = True
    eval_batch_size: int = 32
    device: Optional[str] = None
    amp: str = "auto"                   # auto | bf16 | fp16 | off
    save_dtype: str = "fp16"            # fp16 | fp32
    seed: int = 42
    log_every: int = 50
    token: Optional[str] = None


# ---------------------------------------------------------------------------- setup
def resolve_init(init: str, subfolder: Optional[str], token: Optional[str], head_layers: int):
    """(checkpoint_dir, config) for a Laya checkpoint, or (None, fresh config) for a bare encoder."""
    fresh = {"encoder": init, "head_layers": head_layers, "max_len": 512, "head_max_len": 192,
             "act_costs": {"escalate": 0.5}, "amp_dtype": "bf16"}
    local = os.path.join(init, subfolder) if subfolder else init
    if os.path.isdir(local):
        cfg_path = os.path.join(local, "rl_agent_config.json")
        if not os.path.exists(cfg_path):
            return None, fresh
        with open(cfg_path) as f:
            return local, json.load(f)

    from huggingface_hub import hf_hub_download, snapshot_download
    from huggingface_hub.utils import EntryNotFoundError

    token = token or os.environ.get("HF_TOKEN")
    try:
        hf_hub_download(init, "rl_agent_config.json", subfolder=subfolder, token=token)
    except EntryNotFoundError:
        return None, fresh
    # only this checkpoint: the hub repo bundles three
    patterns = ["%s/%s" % (subfolder, p) for p in CHECKPOINT_FILES] if subfolder else CHECKPOINT_FILES
    root = snapshot_download(init, allow_patterns=patterns, token=token)
    ckpt = os.path.join(root, subfolder) if subfolder else root
    with open(os.path.join(ckpt, "rl_agent_config.json")) as f:
        return ckpt, json.load(f)


def load_base(cfg: TrainConfig):
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    ckpt, mcfg = resolve_init(cfg.init, cfg.subfolder, cfg.token, cfg.head_layers)
    if ckpt:
        _fix_tokenizer_config(ckpt)
        tok = AutoTokenizer.from_pretrained(os.path.join(ckpt, "tokenizer"))
        model = build_model(mcfg, encoder_dir=os.path.join(ckpt, "encoder"))
        model.load_state_dict(load_file(os.path.join(ckpt, "model.safetensors")), strict=True)
    else:
        tok = AutoTokenizer.from_pretrained(cfg.init, token=cfg.token)
        model = build_model(mcfg)
    missing = [n for n in ("mask_token_id", "cls_token_id", "sep_token_id", "pad_token_id")
               if getattr(tok, n, None) is None]
    if missing:
        raise ValueError("the tokenizer of %r lacks %s; Laya needs a masked-LM style encoder"
                         % (cfg.init, ", ".join(missing)))
    try:
        model.encoder.config.reference_compile = False  # see laya.agent: keep ModernBERT eager
    except Exception:
        pass
    return tok, model, mcfg, bool(ckpt)


def build_schema(cfg: TrainConfig) -> LabelSchema:
    files = [f for f in (cfg.train_file, cfg.dev_file, cfg.test_file) if f]
    d = load_labels(cfg.labels_file) if cfg.labels_file else {"labels": infer_labels(files)}
    for key in ("instructions", "none_key", "none_desc", "state_key"):
        if getattr(cfg, key) is not None:
            d[key] = getattr(cfg, key)
    return LabelSchema.from_dict(d)


def pick_device(name: Optional[str], local_rank: int) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda", local_rank)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def amp_setup(device: torch.device, amp: str) -> Tuple[torch.dtype, bool]:
    """(autocast dtype, use GradScaler). Mixed precision is CUDA-only, as in `laya.Agent`."""
    if device.type != "cuda" or amp == "off":
        return torch.float32, False
    if amp == "auto":
        amp = "bf16" if torch.cuda.get_device_capability(device)[0] >= 8 else "fp16"
    return (torch.bfloat16, False) if amp == "bf16" else (torch.float16, True)


def lr_factor(base_lr: float, total: int, warmup: int, eta_min: float = 1e-6):
    floor = min(1.0, eta_min / base_lr)

    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        prog = min(1.0, (step - warmup) / max(1, total - warmup))
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * prog))
    return f


def save_checkpoint(model, tok, mcfg: Dict, out_dir: str, save_dtype: str):
    from safetensors.torch import save_file

    os.makedirs(out_dir, exist_ok=True)
    sd = {}
    for k, v in model.state_dict().items():
        v = v.detach().cpu()
        sd[k] = (v.half() if save_dtype == "fp16" and v.is_floating_point() else v).contiguous()
    save_file(sd, os.path.join(out_dir, "model.safetensors"))
    model.encoder.config.save_pretrained(os.path.join(out_dir, "encoder"))
    tok.save_pretrained(os.path.join(out_dir, "tokenizer"))
    with open(os.path.join(out_dir, "rl_agent_config.json"), "w") as f:
        json.dump(mcfg, f, indent=2, ensure_ascii=False)


def evaluate_split(model, tok, schema, examples: List[Example], budget: Budget, device, dtype,
                   batch_size: int, temperature: float, thresholds) -> Dict:
    d = score_states(model, tok, schema, [e.state for e in examples], budget, device, dtype, batch_size)
    y = np.array([e.target for e in examples], dtype=np.float32)
    return multilabel_metrics(sigmoid(d / temperature), y, thresholds, schema.names)


# ---------------------------------------------------------------------------- training
def train(cfg: TrainConfig) -> Dict:
    if not cfg.train_file:
        raise ValueError("train_file is required")

    world, rank, local_rank = int(os.environ.get("WORLD_SIZE", "1")), 0, 0
    distributed = world > 1 and "RANK" in os.environ
    if distributed:
        import torch.distributed as dist
        # rank 0 evaluates dev while the others wait in a collective; do not let that time out
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo", timeout=timedelta(hours=3))
        rank, local_rank = dist.get_rank(), int(os.environ.get("LOCAL_RANK", "0"))
    device = pick_device(cfg.device, local_rank)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    main = rank == 0

    def say(msg: str):
        if main:
            print(msg, flush=True)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed + rank)  # exploration noise differs per rank, data order does not

    # ---- data
    schema = build_schema(cfg)
    train_ex = read_jsonl(cfg.train_file, schema)
    if cfg.dev_file:
        dev_ex = read_jsonl(cfg.dev_file, schema)
    else:
        random.Random(cfg.seed).shuffle(train_ex)
        n_dev = max(1, int(len(train_ex) * cfg.dev_ratio))
        dev_ex, train_ex = train_ex[:n_dev], train_ex[n_dev:]
    test_ex = read_jsonl(cfg.test_file, schema) if cfg.test_file else []
    if cfg.limit_train:
        train_ex = train_ex[:cfg.limit_train]
    if cfg.limit_eval:
        dev_ex, test_ex = dev_ex[:cfg.limit_eval], test_ex[:cfg.limit_eval]
    y_dev = np.array([e.target for e in dev_ex], dtype=np.float32)

    # ---- model
    raw_tok, model, mcfg, from_laya = load_base(cfg)
    tok = CachedTokenizer(raw_tok)
    max_positions = int(getattr(model.encoder.config, "max_position_embeddings", 512))
    budget = plan_budget(tok, schema, max_positions, cfg.state_budget, mcfg.get("max_len", 512),
                         mcfg.get("head_max_len", 192), cfg.labels_per_seq)
    if cfg.head_max_len:
        budget.head_max_len = cfg.head_max_len
    if cfg.max_len:
        budget.max_len = cfg.max_len
    mcfg.update(max_len=budget.max_len, head_max_len=budget.head_max_len)

    if cfg.freeze_encoder:
        model.encoder.requires_grad_(False)
    if cfg.gradient_checkpointing and not cfg.freeze_encoder:
        try:
            model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except ValueError as e:  # a few encoders do not support it; training still works, with more memory
            say("gradient checkpointing unavailable for this encoder (%s)" % e)
    model.to(device).train()
    net = model
    if distributed:
        from torch.nn.parallel import DistributedDataParallel as DDP
        net = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None,
                  find_unused_parameters=True)

    groups = [{"params": [p for n, p in model.named_parameters() if not n.startswith("encoder.")],
               "lr": cfg.lr_head}]
    if not cfg.freeze_encoder:
        groups.insert(0, {"params": [p for n, p in model.named_parameters() if n.startswith("encoder.")],
                          "lr": cfg.lr_encoder})
    optimizer = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)

    n_chunks = -(-len(schema) // budget.labels_per_seq)
    per_rank = -(-len(train_ex) * n_chunks // world)
    n_batches = -(-per_rank // cfg.micro_batch)
    total_updates = -(-n_batches // cfg.grad_accum) * cfg.epochs
    warmup = int(total_updates * cfg.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, [lr_factor(g["lr"], total_updates, warmup) for g in groups])
    dtype, use_scaler = amp_setup(device, cfg.amp)
    scaler = (torch.amp.GradScaler("cuda", enabled=use_scaler) if hasattr(torch.amp, "GradScaler")
              else torch.cuda.amp.GradScaler(enabled=use_scaler))  # torch < 2.3

    say("init: %s%s (%s) | %.0fM params | device %s x%d | autocast %s"
        % (cfg.init, "/" + cfg.subfolder if cfg.subfolder else "",
           "Laya checkpoint" if from_laya else "bare encoder, new decision head",
           sum(p.numel() for p in model.parameters()) / 1e6, device, world,
           str(dtype).replace("torch.", "") if device.type == "cuda" else "off"))
    say("data: %d train / %d dev / %d test | %d labels, %.2f per example"
        % (len(train_ex), len(dev_ex), len(test_ex), len(schema),
           float(np.mean([sum(t >= 0.5 for t in e.target) for e in train_ex]))))
    say("sequence: max_len %d, head_max_len %d, %d labels per sequence (%d sequence%s per example)"
        % (budget.max_len, budget.head_max_len, budget.labels_per_seq, n_chunks, "" if n_chunks == 1 else "s"))
    say("schedule: %d epochs x %d micro-batches of %d, %d updates" %
        (cfg.epochs, n_batches, cfg.micro_batch, total_updates))

    os.makedirs(cfg.output_dir, exist_ok=True)
    log_path = os.path.join(cfg.output_dir, "train_log.jsonl")
    if main:
        open(log_path, "w").close()

    def log(rec: Dict):
        if main:
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")

    def dev_score() -> Dict:
        d = score_states(model, tok, schema, [e.state for e in dev_ex], budget, device, dtype,
                         cfg.eval_batch_size)
        if device.type == "mps":
            torch.mps.empty_cache()  # evaluation buffers would otherwise stay cached through training
        elif device.type == "cuda":
            torch.cuda.empty_cache()
        p = sigmoid(d)
        thr = tune_thresholds(p, y_dev, "global")
        m = multilabel_metrics(p, y_dev, thr, schema.names)
        at_half = multilabel_metrics(p, y_dev, [0.5] * len(schema), schema.names)
        return {"micro_f1": m["micro_f1"], "macro_f1": m["macro_f1"], "exact_match": m["exact_match"],
                "threshold": thr[0], "micro_f1@0.5": at_half["micro_f1"], "ece@0.5": at_half["ece"]}

    best, best_epoch, stale, t0 = -1.0, 0, 0, time.time()

    def checkpoint_if_best(epoch: int, dev: Dict):
        nonlocal best, best_epoch, stale
        if dev["micro_f1"] > best:
            best, best_epoch, stale = dev["micro_f1"], epoch, 0
            save_checkpoint(model, raw_tok, mcfg, cfg.output_dir, cfg.save_dtype)
        else:
            stale += 1

    # the starting weights are a candidate too, so the result is never worse on dev than the init
    if main and (cfg.eval_before_train or cfg.epochs == 0):
        dev = dev_score()
        checkpoint_if_best(0, dev)
        say("epoch 0 (before training) | dev micro-F1 %.4f @%.2f | exact %.4f"
            % (dev["micro_f1"], dev["threshold"], dev["exact_match"]))
        log({"epoch": 0, "dev": dev})

    # ---- RLCD
    update, epoch = 0, 0
    for epoch in range(1, cfg.epochs + 1):
        rng = random.Random(cfg.seed + epoch)  # identical on every rank so the shards line up
        items = []
        for i, ex in enumerate(train_ex):
            items.extend(build_items(tok, schema, ex.state, budget, target=ex.target,
                                     rng=rng if cfg.shuffle_labels else None, ex=i))
        rng.shuffle(items)
        items += items[: per_rank * world - len(items)]  # every rank runs the same number of steps
        mine = items[rank::world]
        batches = [mine[i:i + cfg.micro_batch] for i in range(0, len(mine), cfg.micro_batch)]

        sums, seen = {"loss": 0.0, "loss_rl": 0.0, "loss_ce": 0.0, "reward": 0.0}, 0
        optimizer.zero_grad(set_to_none=True)
        for bi, chunk in enumerate(batches):
            sigma = cfg.sigma_start + (cfg.sigma_end - cfg.sigma_start) * update / max(1, total_updates - 1)
            boundary = (bi + 1) % cfg.grad_accum == 0 or bi + 1 == len(batches)
            b = collate_items([chunk], tok.pad_token_id)
            with contextlib.nullcontext() if boundary or not distributed else net.no_sync():
                with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"
                                    and dtype != torch.float32):
                    logits, act = net(
                        b["input_ids"].to(device),
                        b["attention_mask"].to(device),
                        b["marker_pos"].to(device),
                        b["marker_mask"].to(device),
                        b["qtype"].to(device),
                        detach_encoder=cfg.freeze_encoder,
                    )
                loss, stats = multilabel_rlcd_loss(
                    logits.float(), b["marker_mask"].to(device), b["target"].to(device), sigma,
                    cfg.group_size, w_sph=cfg.w_sph, rl_weight=cfg.rl_weight, ce_weight=cfg.ce_weight,
                    pos_weight=cfg.pos_weight)
                # the act head takes no part in this objective; keep it in the graph for DDP
                scaler.scale(loss / cfg.grad_accum + 0.0 * act.sum()).backward()

            if boundary:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update += 1

            stats["loss"] = loss.item()
            for key in sums:
                sums[key] += stats[key]
            seen += 1
            if main and seen % cfg.log_every == 0:
                say("  epoch %d/%d | step %d/%d | loss %.4f (rl %.4f, ce %.4f) | reward %.3f | sigma %.3f | lr %.2e"
                    % (epoch, cfg.epochs, seen, len(batches), stats["loss"], stats["loss_rl"],
                       stats["loss_ce"], stats["reward"], sigma, scheduler.get_last_lr()[0]))

        rec = {"epoch": epoch, "minutes": round((time.time() - t0) / 60, 2),
               "train": {k: round(v / max(1, seen), 4) for k, v in sums.items()}}
        if main:
            rec["dev"] = dev_score()
            checkpoint_if_best(epoch, rec["dev"])
            say("epoch %d/%d | loss %.4f | reward %.3f | dev micro-F1 %.4f @%.2f | exact %.4f | %.1f min%s"
                % (epoch, cfg.epochs, rec["train"]["loss"], rec["train"]["reward"], rec["dev"]["micro_f1"],
                   rec["dev"]["threshold"], rec["dev"]["exact_match"], rec["minutes"],
                   "  <- best" if best_epoch == epoch else ""))
            log(rec)
        if distributed:
            flag = torch.tensor([int(cfg.patience > 0 and stale >= cfg.patience)], device=device)
            dist.broadcast(flag, 0)
            stop = bool(flag.item())
        else:
            stop = cfg.patience > 0 and stale >= cfg.patience
        if stop:
            say("no dev improvement for %d epochs, stopping" % cfg.patience)
            break

    # ---- calibrate and evaluate the best checkpoint, exactly as it was written to disk
    report = {}
    if main:
        from safetensors.torch import load_file

        del optimizer, scaler, scheduler
        model.load_state_dict(load_file(os.path.join(cfg.output_dir, "model.safetensors")), strict=True)
        model.eval()
        d_dev = score_states(model, tok, schema, [e.state for e in dev_ex], budget, device, dtype,
                             cfg.eval_batch_size)
        temperature = fit_temperature(d_dev, y_dev)
        thresholds = tune_thresholds(sigmoid(d_dev / temperature), y_dev, cfg.threshold_mode)
        say("\ncalibration on dev: temperature %.3f | thresholds (%s) %s"
            % (temperature, cfg.threshold_mode,
               "%.2f" % thresholds[0] if cfg.threshold_mode != "per_label"
               else "%.2f-%.2f" % (min(thresholds), max(thresholds))))

        mcfg.update({
            "fine_tuned": True,
            "model_name": "laya-multilabel",
            "multilabel": dict(schema.to_dict(), labels_per_seq=budget.labels_per_seq,
                               temperature=temperature, threshold_mode=cfg.threshold_mode,
                               thresholds={n: t for n, t in zip(schema.names, thresholds)}),
            "training": {"objective": "multilabel-rlcd", "init": cfg.init, "subfolder": cfg.subfolder,
                         "best_epoch": best_epoch, "epochs_completed": epoch, "updates": update,
                         "hours": round((time.time() - t0) / 3600, 3), "world_size": world,
                         "config": {k: v for k, v in asdict(cfg).items() if k != "token"}},
        })
        with open(os.path.join(cfg.output_dir, "rl_agent_config.json"), "w") as f:
            json.dump(mcfg, f, indent=2, ensure_ascii=False)

        args = (budget, device, dtype, cfg.eval_batch_size, temperature, thresholds)
        report = {"best_epoch": best_epoch, "temperature": temperature,
                  "dev": evaluate_split(model, tok, schema, dev_ex, *args)}
        if test_ex:
            report["test"] = evaluate_split(model, tok, schema, test_ex, *args)
        with open(os.path.join(cfg.output_dir, "metrics.json"), "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        for split in ("dev", "test"):
            if split in report:
                m = report[split]
                say("%-4s | micro-F1 %.4f (P %.4f, R %.4f) | macro-F1 %.4f | exact match %.4f | ECE %.4f | Brier %.4f"
                    % (split, m["micro_f1"], m["micro_precision"], m["micro_recall"], m["macro_f1"],
                       m["exact_match"], m["ece"], m["brier"]))
        say("saved to %s (best epoch %d)" % (cfg.output_dir, best_epoch))

    if distributed:
        dist.barrier()
        dist.destroy_process_group()
    return report
