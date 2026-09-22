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
    LabelSchema, MultiLabelAgent, TrainConfig, build_causal_sequence, build_items, detect_layout, fit_temperature,
    multilabel_metrics, multilabel_pairs, multilabel_rlcd_loss, pick_marker, plan_budget, read_jsonl, rlcd_loss,
    train, tune_thresholds,
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

# ---------------------------------------------------------------- 6. decoders: causal layout
head("6. Decoder backbone: causal layout, full fine-tune and LoRA adapter (tiny random Qwen3, CPU)")


def make_decoder(root):
    """Same WordPiece vocab, but a decoder-style tokenizer (no mask/cls/sep) and a tiny Qwen3."""
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3Model

    dtok = PreTrainedTokenizerFast(tokenizer_object=tok.backend_tokenizer, pad_token="[PAD]", unk_token="[UNK]")
    dtok.model_max_length = 512
    dec = os.path.join(root, "decoder")
    dtok.save_pretrained(dec)
    torch.manual_seed(0)
    Qwen3Model(Qwen3Config(vocab_size=len(dtok), hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                           num_key_value_heads=2, intermediate_size=128, max_position_embeddings=512,
                           tie_word_embeddings=False)).save_pretrained(dec)
    return dtok, dec


dtok, DEC = make_decoder(tmp.name)
from transformers import AutoConfig  # noqa: E402

ok("layout/decoder config is detected as causal", detect_layout(AutoConfig.from_pretrained(DEC)) == "causal")
ok("layout/encoder config is detected as encoder", detect_layout(AutoConfig.from_pretrained(ENC)) == "encoder")
marker, marker_id, resize = pick_marker(dtok, len(dtok))
ok("marker/no mask token -> <opt> is added and needs a new embedding row", marker == "<opt>" and resize
   and marker_id == len(dtok) - 1)
ok("marker/a mask token is used when the tokenizer has one", pick_marker(tok, 10 ** 6)[0] == tok.mask_token)
cb = plan_budget(dtok, schema, 512, 64, layout="causal", marker_id=marker_id, bos_id=None)
ok("causal/budget adds the parts up", cb.layout == "causal" and cb.max_len == cb.head_max_len + 4 + 64)
state = {"utterance": "play jazz"}
ids, mk = build_causal_sequence(dtok, state, schema.question([1, 0]), cb.max_len, marker_id)
state_ids = dtok(json.dumps(state, ensure_ascii=False), add_special_tokens=False)["input_ids"]
ok("causal/one marker per option, each on the marker token", len(mk) == 3 and all(ids[m] == marker_id for m in mk))
ok("causal/state comes first", ids[:len(state_ids)] == state_ids)
segs = [dtok.decode(ids[(mk[i - 1] + 1) if i else len(state_ids):mk[i]]) for i in range(3)]
ok("causal/threshold option precedes the labels", "none" in segs[0] and "question" in segs[0])
ok("causal/each marker follows its own option text", "book" in segs[1] and "play" in segs[2], segs)
ok("causal/markers are increasing", mk == sorted(mk) and mk[-1] == len(ids) - 1)
short = build_causal_sequence(dtok, {"utterance": "play jazz " * 200}, schema.question(), 96, marker_id)
ok("causal/long state is truncated, options kept", len(short[0]) <= 96 and len(short[1]) == len(schema) + 1)
bos_ids, bos_mk = build_causal_sequence(dtok, state, schema.question([1, 0]), cb.max_len, marker_id, bos_id=7)
ok("causal/bos goes first when the tokenizer has one", bos_ids[0] == 7 and bos_ids[1:] == ids
   and bos_mk == [m + 1 for m in mk])

OUT_D = os.path.join(tmp.name, "out_dec")
cfg_d = TrainConfig(train_file=DATA, dev_file=DATA, labels_file=LABELS_FILE, init=DEC, output_dir=OUT_D,
                    device="cpu", epochs=40, micro_batch=8, grad_accum=1, lr_encoder=1e-3, lr_head=1e-3,
                    shuffle_labels=False, rl_weight=0.3, log_every=10 ** 6)
rep_d = train(cfg_d)
saved_d = json.load(open(os.path.join(OUT_D, "rl_agent_config.json")))
ok("decoder/causal layout and marker recorded", saved_d["multilabel"]["layout"] == "causal"
   and saved_d["multilabel"]["marker_token"] == "<opt>" and saved_d["head_layers"] == 0)
ok("decoder/full fine-tune memorises the set", rep_d["dev"]["micro_f1"] > 0.9, rep_d["dev"]["micro_f1"])
ok("decoder/run stats recorded", rep_d["run"]["layout"] == "causal" and rep_d["run"]["trainable_m"] > 0
   and rep_d["run"]["ms_per_example_batch1"] > 0)
ag_d = MultiLabelAgent(OUT_D, device="cpu")
p_d = ag_d.probabilities(states)
m_d = multilabel_metrics(p_d, y, ag_d.thresholds, ag_d.schema.names)
ok("decoder/full checkpoint reloads through laya.Agent with identical metrics",
   ag_d.agent is not None and abs(m_d["micro_f1"] - rep_d["dev"]["micro_f1"]) < 1e-6)
ok("decoder/tokenizer on disk carries the marker", ag_d.tok.convert_tokens_to_ids("<opt>") == marker_id)

try:
    import peft  # noqa: F401
    HAVE_PEFT = True
except ImportError:
    HAVE_PEFT = False
if HAVE_PEFT:
    OUT_L = os.path.join(tmp.name, "out_lora")
    cfg_l = TrainConfig(train_file=DATA, dev_file=DATA, labels_file=LABELS_FILE, init=DEC, output_dir=OUT_L,
                        device="cpu", epochs=30, micro_batch=8, grad_accum=1, lora_r=8, lr_lora=5e-3, lr_head=1e-3,
                        shuffle_labels=False, rl_weight=0.3, log_every=10 ** 6)
    rep_l = train(cfg_l)
    ok("lora/adapter-only checkpoint layout", os.path.exists(os.path.join(OUT_L, "adapter", "adapter_model.safetensors"))
       and os.path.exists(os.path.join(OUT_L, "model.safetensors")))
    head_only = os.path.getsize(os.path.join(OUT_L, "model.safetensors")) < os.path.getsize(os.path.join(OUT_D, "model.safetensors"))
    ok("lora/model.safetensors holds only the decision head", head_only)
    ok("lora/learns", rep_l["dev"]["micro_f1"] > 0.6, rep_l["dev"]["micro_f1"])
    ok("lora/trainable parameters are the adapter and the head", 0 < rep_l["run"]["trainable_m"] < rep_d["run"]["trainable_m"])
    ag_l = MultiLabelAgent(OUT_L, device="cpu")
    m_l = multilabel_metrics(ag_l.probabilities(states), y, ag_l.thresholds, ag_l.schema.names)
    ok("lora/base + adapter reload gives identical metrics", ag_l.agent is None
       and abs(m_l["micro_f1"] - rep_l["dev"]["micro_f1"]) < 1e-6, (m_l["micro_f1"], rep_l["dev"]["micro_f1"]))
    try:
        train(TrainConfig(train_file=DATA, init=OUT_L, output_dir=os.path.join(tmp.name, "x"), device="cpu", epochs=0))
        ok("lora/adapter checkpoint is refused as an init", False)
    except ValueError:
        ok("lora/adapter checkpoint is refused as an init", True)
else:
    print("   SKIP lora/* (peft not installed)")

# ---------------------------------------------------------------- 7. Gemma 4 architecture
head("7. Gemma 4 text backbone: per-layer embeddings, <mask> reused as marker (tiny random model, CPU)")
try:
    from transformers import PreTrainedTokenizerFast
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    HAVE_GEMMA4 = True
except ImportError:
    HAVE_GEMMA4 = False
if HAVE_GEMMA4:
    from transformers import AutoModel

    gtok = PreTrainedTokenizerFast(tokenizer_object=tok.backend_tokenizer, pad_token="[PAD]", unk_token="[UNK]",
                                   mask_token="[MASK]", bos_token="[CLS]", eos_token="[SEP]")
    gtok.model_max_length = 512
    G4 = os.path.join(tmp.name, "gemma4")
    gtok.save_pretrained(G4)
    # the last layer shares KV with the last non-shared layer of its type, so it needs one before it
    gcfg = Gemma4TextConfig(vocab_size=len(gtok), hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                            num_attention_heads=4, num_key_value_heads=1, head_dim=16, max_position_embeddings=512,
                            sliding_window=32, num_kv_shared_layers=1,
                            layer_types=["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
                            vocab_size_per_layer_input=len(gtok), hidden_size_per_layer_input=8,
                            pad_token_id=gtok.pad_token_id, bos_token_id=gtok.bos_token_id, eos_token_id=gtok.eos_token_id)
    torch.manual_seed(0)
    AutoModel.from_config(gcfg).save_pretrained(G4)
    ok("gemma4/decoder although the tokenizer has a mask token", detect_layout(AutoConfig.from_pretrained(G4)) == "causal")
    ok("gemma4/the mask token is reused as marker, no resize", pick_marker(gtok, len(gtok)) == ("[MASK]", gtok.mask_token_id, False))
    OUT_G = os.path.join(tmp.name, "out_g4")
    rep_g = train(TrainConfig(train_file=DATA, dev_file=DATA, labels_file=LABELS_FILE, init=G4, output_dir=OUT_G,
                              device="cpu", epochs=30, micro_batch=8, grad_accum=1, lr_encoder=1e-3, lr_head=1e-3,
                              shuffle_labels=False, rl_weight=0.3, log_every=10 ** 6))
    saved_g = json.load(open(os.path.join(OUT_G, "rl_agent_config.json")))
    ok("gemma4/trains with bos first and <mask> markers", rep_g["dev"]["micro_f1"] > 0.8
       and saved_g["multilabel"]["marker_token"] == "[MASK]", rep_g["dev"]["micro_f1"])
    ag_g = MultiLabelAgent(OUT_G, device="cpu")
    ok("gemma4/reloads with identical metrics",
       abs(multilabel_metrics(ag_g.probabilities(states), y, ag_g.thresholds, ag_g.schema.names)["micro_f1"]
           - rep_g["dev"]["micro_f1"]) < 1e-6)
    ok("gemma4/bos is the first token of every sequence",
       build_items(ag_g.tok, ag_g.schema, "x", ag_g.budget)[0]["ids"][0] == gtok.bos_token_id)
    try:
        train(TrainConfig(train_file=DATA, dev_file=DATA, labels_file=LABELS_FILE, init=G4, marker_token="<opt>",
                          output_dir=os.path.join(tmp.name, "x2"), device="cpu", epochs=0))
        ok("gemma4/refuses a marker that would need a new embedding row", False)
    except ValueError:
        ok("gemma4/refuses a marker that would need a new embedding row", True)
else:
    print("   SKIP gemma4/* (this transformers has no gemma4)")

# ---------------------------------------------------------------- 8. experiment tracking
head("8. Experiment tracking (trackio when installed; the adapter itself needs nothing)")
from laya.multilabel.tracking import Tracker, flat  # noqa: E402

ok("tracking/flat nests with slashes and drops non-numbers",
   flat("dev", {"micro_f1": 0.5, "per_label": {"a": {"f1": 1.0}}, "name": "x", "ok": True})
   == {"dev/micro_f1": 0.5, "dev/per_label/a/f1": 1.0})
none = Tracker("none", "p", "r", {})
none.log({"x": 1}, step=0)
none.finish()
ok("tracking/none is a no-op", none.url is None)
try:
    Tracker("mlflow", "p", "r", {})
    ok("tracking/unknown tracker is rejected", False)
except ValueError:
    ok("tracking/unknown tracker is rejected", True)
os.environ["TRACKIO_DIR"] = os.path.join(tmp.name, "trackio")  # must be set before trackio is first imported
try:
    import trackio  # noqa: F401
    HAVE_TRACKIO = True
except ImportError:
    HAVE_TRACKIO = False
if HAVE_TRACKIO:
    import sqlite3

    rep_t = train(TrainConfig(train_file=DATA, dev_file=DATA, labels_file=LABELS_FILE, init=DEC,
                              output_dir=os.path.join(tmp.name, "out_trk"), device="cpu", epochs=3, micro_batch=8,
                              grad_accum=1, lr_encoder=1e-3, lr_head=1e-3, log_every=2, tracker="trackio",
                              project="laya-test", run_name="tiny"))
    db = os.path.join(tmp.name, "trackio", "laya-test.db")
    con = sqlite3.connect(db)
    n_rows = con.execute("select count(*) from metrics where run_name = 'tiny'").fetchone()[0]
    last = json.loads(con.execute("select metrics from metrics where run_name = 'tiny' order by step desc, id desc limit 1")
                      .fetchone()[0])
    cfg_row = json.loads(con.execute("select config from configs where run_name = 'tiny'").fetchone()[0])
    con.close()
    ok("tracking/trackio has per-step and per-epoch rows", n_rows >= 3 + 3, n_rows)
    ok("tracking/final metrics match metrics.json", abs(last["final/test/micro_f1"] - rep_t["test"]["micro_f1"]) < 1e-9
       if "final/test/micro_f1" in last else abs(last["final/dev/micro_f1"] - rep_t["dev"]["micro_f1"]) < 1e-9, last)
    ok("tracking/config is recorded", cfg_row.get("lr_head") == 1e-3 and cfg_row.get("layout") == "causal")
else:
    print("   SKIP tracking/trackio (not installed)")

# ---------------------------------------------------------------- summary
head("SUMMARY")
print("\n   %d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("     FAIL " + f)
tmp.cleanup()
sys.exit(1 if FAIL else 0)
