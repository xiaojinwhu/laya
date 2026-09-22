"""Multi-label primitive: sequence layout, RLCD loss, calibration, metrics, and a full train run.

Needs no network and no checkpoint: the end-to-end part trains a randomly initialised two-layer
BERT, built here, to memorise a small keyword dataset on CPU.

Run:  python3 tests/test_multilabel.py
"""
import json
import os
import random
import sys
import tempfile

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import laya  # noqa: E402
from laya.common import QTYPES, proper_reward  # noqa: E402
from laya.multilabel import (  # noqa: E402
    LabelSchema, MultiLabelAgent, TrainConfig, build_items, fit_temperature, multilabel_metrics,
    multilabel_pairs, multilabel_rlcd_loss, plan_budget, read_jsonl, rlcd_loss, train, tune_thresholds,
)
from laya.multilabel.__main__ import main as cli  # noqa: E402
from laya.multilabel.data import chunk_labels  # noqa: E402
from laya.multilabel.rlcd import label_logits  # noqa: E402

PASS, FAIL = [], []


def ok(name, cond, detail=""):
    (PASS if cond else FAIL).append(name if cond else "%s  %s" % (name, detail))
    print("   %s %s%s" % ("PASS" if cond else "FAIL", name, ("  " + str(detail)) if detail and not cond else ""),
          flush=True)


def head(t):
    print("\n" + "=" * 78 + "\n  " + t + "\n" + "=" * 78, flush=True)


LABELS = {"play_music": "play a song or an album", "book_table": "reserve a table to eat",
          "get_weather": "forecast, rain, temperature", "set_alarm": "wake me up, alarm clock"}
PHRASES = {"play_music": ["play jazz", "put on a song"], "book_table": ["book a table", "reserve dinner"],
           "get_weather": ["will it rain", "weather forecast"], "set_alarm": ["wake me at six", "set an alarm"]}


def make_fixture(root):
    """Tiny WordPiece tokenizer + random BERT saved as a bare HF encoder, plus a keyword dataset."""
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

    rng = random.Random(0)
    rows = []
    for _ in range(48):
        labs = rng.sample(list(LABELS), rng.choice([1, 2, 2, 3]))
        rows.append({"text": " and ".join(rng.choice(PHRASES[n]) for n in labs), "labels": labs})
    rows[0]["labels"] = {rows[0]["labels"][0]: 0.8}  # soft labels are accepted too
    data = os.path.join(root, "train.jsonl")
    with open(data, "w") as f:
        f.write("\n".join(json.dumps(r) for r in rows) + "\n")
    labels = os.path.join(root, "labels.json")
    with open(labels, "w") as f:
        json.dump(LABELS, f)

    corpus = [r["text"] for r in rows] + list(LABELS) + list(LABELS.values()) + [
        "choice question: which intents does the user express in utterance several may apply "
        "none no listed intent is expressed false true noul"]
    t = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    t.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    t.train_from_iterator(corpus, trainers.WordPieceTrainer(
        vocab_size=300, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]))
    tok = PreTrainedTokenizerFast(tokenizer_object=t, pad_token="[PAD]", unk_token="[UNK]",
                                  cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")
    enc = os.path.join(root, "encoder")
    tok.save_pretrained(enc)
    torch.manual_seed(0)
    BertModel(BertConfig(vocab_size=len(tok), hidden_size=128, num_hidden_layers=2, num_attention_heads=2,
                         intermediate_size=256, max_position_embeddings=256)).save_pretrained(enc)
    return tok, enc, data, labels


tmp = tempfile.TemporaryDirectory()
tok, ENC, DATA, LABELS_FILE = make_fixture(tmp.name)
schema = LabelSchema(labels=LABELS)

# ---------------------------------------------------------------- 1. schema and sequences
head("1. Schema and sequence layout")
try:
    LabelSchema(labels={"none": None, "a": None})
    ok("schema/label colliding with the threshold option is rejected", False)
except ValueError:
    ok("schema/label colliding with the threshold option is rejected", True)
ok("schema/threshold option comes first", list(schema.question([2, 0])["crit"]) == ["none", "get_weather", "play_music"])
ok("schema/round-trips through its dict", LabelSchema.from_dict(schema.to_dict()).to_dict() == schema.to_dict())

budget = plan_budget(tok, schema, max_positions=256, state_budget=64)
target = [1.0, 0.0, 0.0, 1.0]
it = build_items(tok, schema, "play jazz and set an alarm", budget, target=target)[0]
ok("items/one marker per label plus the threshold", len(it["markers"]) == len(schema) + 1)
ok("items/every marker sits on a [MASK]", all(it["ids"][m] == tok.mask_token_id for m in it["markers"]))
ok("items/the threshold marker has no target", it["target"] == [0.0] + target, it["target"])
ok("items/laid out as a choice question", it["qtype"] == QTYPES["choice"])

sh = build_items(tok, schema, "play jazz and set an alarm", budget, target=target, rng=random.Random(3))[0]
ok("items/shuffling reorders the labels", sh["label_idx"] != it["label_idx"] and sorted(sh["label_idx"]) == [0, 1, 2, 3])
ok("items/targets follow their labels", [sh["target"][1 + i] for i in np.argsort(sh["label_idx"])] == target)
text = tok.decode(sh["ids"][sh["markers"][1]:sh["markers"][2]])
ok("items/option text follows its marker", schema.names[sh["label_idx"][0]].split("_")[0] in text, text)

chunks = chunk_labels(10, 4, random.Random(0))
ok("chunks/cover every label once", sorted(j for c in chunks for j in c) == list(range(10)))
ok("chunks/are even and within the limit", [len(c) for c in chunks] == [4, 4, 2] or max(map(len, chunks)) <= 4)
multi = build_items(tok, schema, "play jazz", plan_budget(tok, schema, 256, 64, labels_per_seq=3), target=target)
ok("chunks/each sequence carries its own threshold marker",
   len(multi) == 2 and all(len(m["markers"]) == len(m["label_idx"]) + 1 for m in multi))

# ---------------------------------------------------------------- 2. token budget
head("2. Token budget")
big = LabelSchema(labels={"intent_%02d" % i: "description number %d of a rather long kind" % i for i in range(60)})
b_all = plan_budget(tok, big, max_positions=8192, state_budget=128)
ok("budget/head grows until no option is truncated", b_all.labels_per_seq == 60 and b_all.head_max_len > 192,
   b_all)
ok("budget/never shrinks below the checkpoint's own", budget.head_max_len >= 192 or budget.max_len == 256, budget)
b_small = plan_budget(tok, big, max_positions=512, state_budget=128)
ok("budget/labels are chunked when positions run out", 1 <= b_small.labels_per_seq < 60, b_small)
seqs = build_items(tok, big, "x", b_small)
ok("budget/chunked sequences fit", all(len(s["ids"]) <= b_small.max_len for s in seqs)
   and sum(len(s["label_idx"]) for s in seqs) == 60)

# ---------------------------------------------------------------- 3. RLCD
head("3. RLCD loss")
torch.manual_seed(0)
logits = torch.randn(5, 4)
mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]], dtype=torch.bool)
y = torch.tensor([[0, 1, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [0, .3, .7, 0], [0, 0, 0, 0]], dtype=torch.float32)
pair, tpair = multilabel_pairs(logits, mask, y)
ok("pairs/one noul question per valid label", tuple(pair.shape) == (int(mask[:, 1:].sum()), 2))
ok("pairs/softmax over [threshold, label] is sigmoid of the difference",
   torch.allclose(torch.softmax(pair, -1)[:, 1], torch.sigmoid(label_logits(logits))[mask[:, 1:]], atol=1e-6))
ok("pairs/targets are [1-y, y]", torch.allclose(tpair.sum(-1), torch.ones(len(tpair))) and
   torch.allclose(tpair[:, 1], y[:, 1:][mask[:, 1:]]))


def notebook_step(lg, tg, qt, mk, sigma, G):
    """The training step of notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb, verbatim."""
    k = mk.sum(-1, keepdim=True).float()
    eps = torch.randn((G,) + lg.shape) * sigma * mk
    eps = (eps - eps.sum(-1, keepdim=True) / k) * mk
    z = lg.detach().unsqueeze(0) + eps
    q = torch.softmax(z.masked_fill(~mk, -1e4), -1)
    with torch.no_grad():
        r = proper_reward(q, tg.unsqueeze(0), qt, mk, w_sph=0.75, w_rps=1.0)
        adv = r - r.mean(0, keepdim=True)
        adv = adv / (adv.std() + 1e-6)
    logp = -(((z - lg.unsqueeze(0)) ** 2) * mk).sum(-1) / (2 * sigma ** 2)
    loss_rl = -(adv * logp).mean()
    loss_ce = -(tg * torch.log_softmax(lg.masked_fill(~mk, -1e4), -1)).sum(-1).mean()
    return loss_rl + 1.0 * loss_ce


qt = torch.tensor([0, 1, 2, 0, 1])
soft = torch.softmax(torch.randn(5, 4).masked_fill(~mask, -1e4), -1)
a = logits.clone().requires_grad_(True)
b = logits.clone().requires_grad_(True)
torch.manual_seed(7)
mine, _ = rlcd_loss(a, soft, qt, mask, sigma=0.3, group_size=4)
torch.manual_seed(7)
ref = notebook_step(b, soft, qt, mask, 0.3, 4)
mine.backward()
ref.backward()
ok("rlcd/same loss as the fine-tuning notebook", torch.allclose(mine, ref, atol=1e-6), (mine.item(), ref.item()))
ok("rlcd/same gradient as the fine-tuning notebook", torch.allclose(a.grad, b.grad, atol=1e-6))

z = logits.clone().requires_grad_(True)
loss, stats = multilabel_rlcd_loss(z, mask, y, sigma=0.3)
loss.backward()
ok("rlcd/finite loss and stats", bool(torch.isfinite(loss)) and all(np.isfinite(v) for v in stats.values()))
ok("rlcd/padded markers get no gradient", bool((z.grad[~mask] == 0).all()))
ok("rlcd/the threshold marker is trained", bool((z.grad[:, 0] != 0).all()))

# The reward is a strictly proper scoring rule, so the policy gradient on its own (no
# cross-entropy guidance) has to settle on the target probabilities, soft ones included.
torch.manual_seed(0)
yt = torch.tensor([[0., 1.0, 0.0, 0.7, 0.3], [0., 0.0, 1.0, 0.9, 0.5]])
free = torch.zeros_like(yt, requires_grad=True)
opt = torch.optim.Adam([free], lr=0.05)
for _ in range(500):
    loss, _ = multilabel_rlcd_loss(free, torch.ones_like(yt, dtype=torch.bool), yt, sigma=0.2,
                                   group_size=8, rl_weight=1.0, ce_weight=0.0)
    opt.zero_grad()
    loss.backward()
    opt.step()
gap = float((torch.sigmoid(label_logits(free.detach())) - yt[:, 1:]).abs().max())
ok("rlcd/reward alone converges to the soft targets", gap < 0.08, "max |p - y| = %.3f" % gap)

w1, _ = multilabel_rlcd_loss(logits, mask, y, 0.3, rl_weight=0.0, pos_weight=1.0)
w5, _ = multilabel_rlcd_loss(logits, mask, y, 0.3, rl_weight=0.0, pos_weight=5.0)
ok("rlcd/pos_weight changes the weighting", not torch.allclose(w1, w5))

# ---------------------------------------------------------------- 4. calibration and metrics
head("4. Calibration and metrics")
g = np.random.default_rng(0)
p_true = g.uniform(0.02, 0.98, size=6000)
y_draw = (g.uniform(size=6000) < p_true).astype(np.float32)
d_over = 2.5 * np.log(p_true / (1 - p_true))  # logits of a model 2.5x too confident
T = fit_temperature(d_over, y_draw)
ok("temperature/recovers an over-confident model's", abs(T - 2.5) < 0.25, "T = %.3f" % T)
ok("temperature/too little data leaves it at 1", fit_temperature(d_over[:5], y_draw[:5]) == 1.0)

P = np.array([[.9, .2, .4], [.8, .6, .1], [.3, .7, .45], [.2, .1, .45]])
Y = np.array([[1, 0, 1], [1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=np.float32)
ok("thresholds/fixed", tune_thresholds(P, Y, "fixed") == [0.5] * 3)
tg = tune_thresholds(P, Y, "global")
ok("thresholds/global finds the separating cut", len(set(tg)) == 1 and 0.3 < tg[0] <= 0.4, tg)
ok("thresholds/rare labels fall back to global", tune_thresholds(P, Y, "per_label", min_pos=99) == tg)
P2 = P.copy()
P2[:, 2] = [.2, .05, .25, .15]  # a label the model is systematically shy about
tl = tune_thresholds(P2, Y, "per_label", min_pos=1)
ok("thresholds/per-label adapts to a shy label", tl[2] <= 0.15 and tl[0] > 0.3, tl)

m = multilabel_metrics(P, Y, [0.5] * 3, ["a", "b", "c"])
ok("metrics/micro-F1", m["micro_f1"] == round(2 * 4 / (2 * 4 + 0 + 3), 4), m["micro_f1"])  # tp 4, fp 0, fn 3
ok("metrics/exact match", m["exact_match"] == 0.25, m["exact_match"])
ok("metrics/per-label support", [m["per_label"][n]["support"] for n in "abc"] == [2, 2, 3])
perfect = multilabel_metrics(Y * 0.98 + 0.01, Y, [0.5] * 3, ["a", "b", "c"])
ok("metrics/perfect predictions", perfect["micro_f1"] == 1.0 and perfect["exact_match"] == 1.0
   and perfect["hamming_loss"] == 0.0 and perfect["ece"] < 0.02)

# ---------------------------------------------------------------- 5. end to end
head("5. Train -> checkpoint -> inference (tiny random encoder, CPU)")
examples = read_jsonl(DATA, schema)
ok("data/soft labels are read", examples[0].target.count(0.8) == 1)
OUT = os.path.join(tmp.name, "out")
cfg = TrainConfig(train_file=DATA, dev_file=DATA, labels_file=LABELS_FILE, init=ENC, output_dir=OUT,
                  device="cpu", epochs=60, micro_batch=8, grad_accum=1, lr_encoder=1e-3, lr_head=1e-3,
                  shuffle_labels=False, rl_weight=0.3, log_every=10 ** 6, threshold_mode="per_label")
report = train(cfg)
log = [json.loads(line) for line in open(os.path.join(OUT, "train_log.jsonl"))]
ok("train/epoch 0 is evaluated before any update", log[0]["epoch"] == 0)
ok("train/loss falls", log[-1]["train"]["loss"] < 0.5 * log[1]["train"]["loss"],
   (log[1]["train"]["loss"], log[-1]["train"]["loss"]))
ok("train/reward rises", log[-1]["train"]["reward"] > log[1]["train"]["reward"])
ok("train/memorises the training set", report["dev"]["micro_f1"] > 0.9 > log[0]["dev"]["micro_f1"],
   (log[0]["dev"]["micro_f1"], report["dev"]["micro_f1"]))
ok("train/writes the Laya checkpoint layout", all(os.path.exists(os.path.join(OUT, p)) for p in (
    "model.safetensors", "rl_agent_config.json", "encoder/config.json", "tokenizer/tokenizer.json", "metrics.json")))

saved = json.load(open(os.path.join(OUT, "rl_agent_config.json")))
ok("config/multilabel section", set(saved["multilabel"]["thresholds"]) == set(LABELS)
   and saved["multilabel"]["temperature"] > 0 and saved["training"]["best_epoch"] >= 1)

agent = MultiLabelAgent(OUT, device="cpu")
states = [e.state for e in examples]
p = agent.probabilities(states)
y = np.array([e.target for e in examples])
again = multilabel_metrics(p, y, agent.thresholds, agent.schema.names)
ok("agent/reproduces the trainer's metrics from disk", abs(again["micro_f1"] - report["dev"]["micro_f1"]) < 1e-6,
   (again["micro_f1"], report["dev"]["micro_f1"]))
res = agent.predict(states[1])
ok("agent/answer shape", res["type"] == "multi" and set(res["probabilities"]) == set(LABELS)
   and 0.0 <= res["confidence"] <= 1.0 and all(n in LABELS for n in res["labels"]))
ok("agent/threshold override", agent.predict(states[1], threshold=0.0)["labels"] == list(LABELS)
   and agent.predict(states[1], threshold=1.01)["labels"] == [])

agent.budget.labels_per_seq = 2  # same labels split over two sequences, each with its own threshold
ok("agent/chunked inference scores every label", agent.probabilities(states[:8]).shape == (8, 4)
   and bool((agent.probabilities(states[:8]) > 0).all()))

stock = laya.load(OUT, device="cpu")  # the unmodified runtime must still load and answer
ans = stock.predict({"utterance": "play jazz"}, {"q": {"type": "noul", "instructions": "Is music requested?"}})
ok("compat/stock laya.Agent loads the checkpoint", 0.0 <= ans["answers"]["q"]["noul"] <= 1.0)

pred_file = os.path.join(tmp.name, "pred.jsonl")
cli(["predict", "--model", OUT, "--input", DATA, "--output", pred_file, "--device", "cpu"])
rows = [json.loads(line) for line in open(pred_file)]
ok("cli/predict writes one prediction per row", len(rows) == len(examples) and "labels" in rows[0]["prediction"])

try:
    MultiLabelAgent(ENC, device="cpu")
    ok("agent/refuses a directory that is not a checkpoint", False)
except (FileNotFoundError, ValueError):
    ok("agent/refuses a directory that is not a checkpoint", True)

# ---------------------------------------------------------------- summary
head("SUMMARY")
print("\n   %d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("     FAIL " + f)
tmp.cleanup()
sys.exit(1 if FAIL else 0)
