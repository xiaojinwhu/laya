# 多标签意图分类训练 pipeline

一句话里可以同时包含多个意图（"放首爵士乐，**再**帮我订个两人位"）。Laya 原生的三种决策原语里，
`choice` 是 softmax 单选，`noul` 是单个二值判断，都不能在**一次前向**里输出"每个意图是否出现"。
`laya.multilabel` 在不改动模型结构的前提下补上了这个原语，并沿用仓库里的 RLCD 训练算法
（`notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb`）。

```
laya/multilabel/
  schema.py        标签体系 + 问题措辞
  data.py          JSONL 读取、token 预算规划、两种序列布局（encoder / causal）
  backbone.py      任意 HF 主干的接入：布局判断、marker 选择、多模态 checkpoint 取文本主干、LoRA
  rlcd.py          RLCD 损失（notebook 训练步的函数化移植）+ 多标签形式
  calibration.py   温度拟合（LBFGS）+ 判定阈值搜索
  metrics.py       micro/macro-F1、exact match、ECE、Brier …
  trainer.py       训练主流程（单卡 / torchrun 多卡），输出标准 Laya checkpoint 或 adapter-only checkpoint
  agent.py         推理：MultiLabelAgent（两种 checkpoint 都能加载）
  __main__.py      命令行：train / eval / predict
examples/multilabel_intent/
  prepare_data.py  MixSNIPS / MixATIS / 由 MASSIVE 合成的多意图数据（51 种语言，含中文）
  ablate.py        消融跑批：一份基础配置 × 一组覆盖 → 逐个训练 → 汇总成表
tests/test_multilabel.py   无需联网的单元 + 端到端测试（encoder 与 decoder 两条路径）
```

## 算法：如何把多标签放进 Laya

**序列布局**与 `choice` 完全相同，只是在最前面多放一个**阈值选项**（`none`）：

```
[CLS] choice question: <instructions> [SEP] [MASK] none: … [MASK] intent_0: … [MASK] intent_1: … [SEP] utterance [SEP]
```

模型在每个 `[MASK]` 位置打一个分 `z`。标签 *i* 的概率定义为它与阈值选项的**两两 softmax**：

```
P(intent_i) = softmax([z_none, z_i])[1] = sigmoid(z_i − z_none)
```

这恰好就是一个 logits 为 `[false, true] = [z_none, z_i]` 的 `noul` 问题。所以一条含 K 个标签的序列
＝ **在同一次前向里回答的 K 个 noul 问题**，RLCD 算法可以原封不动地套在每个 (样本, 标签) 决策上：

| 步骤 | notebook（单问题） | 这里（每个标签决策） |
|---|---|---|
| 策略 | 选项 logits 上的高斯 `N(z, σ²)`，噪声做零均值投影 | 同左，作用在 `[z_none, z_i]` 这一对上 |
| 采样 | 每题 `G=4` 个带噪分布 `q = softmax(z+ε)` | 同左 |
| 奖励 | `proper_reward`：log score + 0.75 × spherical score（严格 proper） | 直接调用 `laya.common.proper_reward`，qtype=noul |
| 优势 | 组内相对基线（GRPO 风格），全 batch 标准化 | **每个标签决策各自一组基线** → 信用分配精确到标签 |
| 损失 | `−adv · log N(z+ε; z, σ²)` + 1.0 × soft CE | 同左（soft CE 即逐标签 BCE，支持软标签） |
| 探索 | σ 0.4 → 0.1 | 同左（按 update 线性退火） |
| 优化 | AdamW，encoder 2.5e-5 / head 1e-4，cosine，梯度累积，clip 1.0 | 同左，另有可选 warmup |
| 校准 | 训练后 LBFGS 拟合温度 | 同左，但在**留出的 dev** 上拟合，并额外搜索判定阈值 |

`tests/test_multilabel.py` 里有一项测试把 notebook 的训练步逐字抄下来，验证 `rlcd_loss` 在相同随机种子下
**损失与梯度完全一致**；另一项验证只用 RL 项（关掉 CE）时概率会收敛到软标签本身（0.7 → 0.70），
这正是"严格 proper scoring rule"应有的性质。

RLCD 常被当成 GRPO 的变体，其实两者只共享"组内相对基线"这一个部件；逐阶段的对照和为什么它能得到校准的概率，
见 [RLCD_vs_GRPO.md](RLCD_vs_GRPO.md)。

这样设计带来的几个好处：

- **结构零改动**：权重、配置、目录布局都是标准 Laya checkpoint，`laya.load(out_dir)` 照常可用，
  原有的 choice / score / noul 问题仍能回答。
- **零样本可用**：Laya 预训练时见过大量 "other / none of the above" 选项，阈值选项的语义与之一致
  （下方实测：未训练时 micro-F1 已有 0.73）。
- **标签可以分块**：softmax 单选一旦把选项拆到多条序列，概率就失去归一化；而这里每个标签只和
  **自己序列里的阈值**比较，所以标签很多、放不进一条序列时可以自动拆成多条
  （README 里 Banking77 那类"77 个选项挤在 256 token 里"的问题因此不存在）。
- **选项顺序增强**：每个 epoch 重新打乱标签顺序（阈值选项固定在首位），抑制位置偏置。

### decoder 基模（Qwen / Gemma）怎么接

上面的布局把选项放在 state **前面**，靠双向注意力让 `[MASK]` 读到后面的 state。因果注意力下这行不通：marker 位置根本看不到
state。所以 decoder 用第二种布局 `causal`，把顺序反过来，marker 放在**每个选项文本之后**：

```
{"utterance": "放首爵士乐，再帮我订个两人位"}
choice question: Which intents does the user express in `utterance`? Several may apply.
- none: no listed intent is expressed<m>
- play_music: play a song, album or artist<m>
- book_restaurant: reserve a table<m>
```

走到 `<m>` 时主干已经读过 state、问题和这个选项本身。逐标签决策 `sigmoid(z_i − z_none)`、RLCD 损失、校准、指标全都不变，
只是序列构造换了一个函数（`data.build_causal_sequence`）。`none` 仍在最前，只看到 state + 问题，是一个干净的基线；后面的标签
能看到前面的标签，靠每个 epoch 打乱顺序来消位置依赖。不套 chat template。

接入时几件事是自动处理的（`backbone.py`）：

- **布局判断按模型类型**，不按 tokenizer：Gemma 的 tokenizer 有 `<mask>`，但它是 decoder。
- **marker 复用现有 token，尽量不 resize**：Gemma 用它的 `<mask>`，Qwen 用 `<|fim_pad|>`，都没有才新增 `<opt>`。
  Gemma 3n / 4 带 per-layer embedding（按 token id 查第二张表），resize 主 embedding 会把它撑爆，所以对这类模型直接拒绝新增。
- **多模态 checkpoint 只取文本主干**：Qwen3.5、Gemma 4 的 hub 权重是 `*ForConditionalGeneration`，用 `text_config` 走
  `AutoModelForCausalLM` 只加载语言模型部分，视觉/音频塔不进内存；失败时退回整模加载再取 `language_model`。
- **决策头默认 0 层**：2 层随机初始化的 transformer 在 d=2560 时是 150M 新参数，小数据上大概率有害；`head_layers` 可作消融轴。
- **LoRA**（`peft`）：`lora_r > 0` 时主干 bf16 冻结、adapter 和决策头 fp32 训练；只存 adapter。

限制：decoder checkpoint 只能走 `MultiLabelAgent`，原版 `laya.Agent` 的 choice / score / noul 不适用（布局不同）。

## 快速开始

```bash
uv venv .venv && uv pip install --python .venv/bin/python -e . tokenizers
```

### 1. 准备数据

```bash
python examples/multilabel_intent/prepare_data.py mixsnips --out examples/multilabel_intent/data/mixsnips
python examples/multilabel_intent/prepare_data.py mixatis  --out examples/multilabel_intent/data/mixatis
python examples/multilabel_intent/prepare_data.py massive  --out examples/multilabel_intent/data/massive_zh --lang zh-CN
```

自有数据只需要两种文件：

```jsonl
{"text": "放首爵士乐，再帮我订个两人位", "labels": ["play_music", "book_restaurant"]}
{"text": "明天会下雨吗", "labels": ["weather_query"]}
{"state": {"utterance": "…", "channel": "app"}, "labels": {"refund": 1.0, "complaint": 0.6}}
```

- `text`（字符串）或 `state`（任意 JSON，和 Laya 的 state 一样）；
- `labels` 是名字列表，或 `{名字: 概率}` 的**软标签**（RLCD 的奖励和 CE 都支持）；没有任何意图就给 `[]`。

`labels.json`（可选，但**强烈建议写描述**——模型是靠读选项文字来判断的）：

```json
{
  "instructions": "Which intents does the user express in `utterance`? Several may apply.",
  "labels": {"play_music": "play a song, album or artist", "book_restaurant": "reserve a table"}
}
```

也可以只是名字列表 `["a", "b"]` 或 `{名字: 描述}`；不提供时从数据里自动收集标签名。

### 2. 训练

```bash
# 从 Laya 英文 checkpoint 微调
python -m laya.multilabel train --config examples/multilabel_intent/config.mixsnips.json

# 中文 / 多语言：用 multilingual 子目录（mmBERT-base）
python -m laya.multilabel train --config examples/multilabel_intent/config.massive_zh.json

# 任意 HF 掩码语言模型 encoder 也可以（决策头随机初始化）
python -m laya.multilabel train --init hfl/chinese-roberta-wwm-ext --lr-head 5e-4 --warmup-ratio 0.05 \
    --train-file train.jsonl --dev-file dev.jsonl --labels-file labels.json --output-dir out

# decoder 基模（Qwen3 / Qwen3.5 / Gemma 4 …）：自动切到 causal 布局；LoRA 只存 adapter
python -m laya.multilabel train --init Qwen/Qwen3-0.6B-Base --lora-r 16 \
    --train-file train.jsonl --dev-file dev.jsonl --labels-file labels.json --output-dir out_qwen

# 多卡（与 notebook 相同的 DDP 方式）
torchrun --standalone --nproc_per_node=2 -m laya.multilabel train --config cfg.json
```

`--config` 里的每个字段都可以用同名命令行参数覆盖（`--epochs 3 --pos-weight 2`）。常用字段：

| 字段 | 默认 | 说明 |
|---|---|---|
| `init` / `subfolder` | `convaiinnovations/laya` | Laya checkpoint（hub id 或目录）、任意 HF 掩码 encoder，或 decoder（Qwen / Gemma …） |
| `layout` | 自动 | `encoder` / `causal`；按模型类型判断（在 MaskedLM 映射里的是 encoder，其余 causal） |
| `marker_token` | 自动 | causal 布局的选项 marker：依次取 tokenizer 的 mask token → `<\|fim_pad\|>` → 新增 `<opt>` |
| `head_layers` | 自动 | 裸主干上的决策头层数：encoder 2 层，decoder 0 层（scorer 直接接 marker 的 hidden state） |
| `load_dtype` | 自动 | 主干权重精度：LoRA + GPU/MPS 时 bf16，否则 fp32 |
| `lora_r` `lora_alpha` `lora_dropout` `lora_targets` `lr_lora` | 0 / 32 / 0.05 / all-linear / 2e-4 | `lora_r > 0` 启用 LoRA（需要 `peft`），主干冻结，只训 adapter + 决策头；checkpoint 只存 adapter |
| `tracker` `project` `run_name` `tracker_space` | none / laya-multilabel / 输出目录名 / – | 实验记录：`trackio` 或 `wandb`（见下文「实验记录」） |
| `dev_file` / `dev_ratio` | – / 0.1 | 选模型、拟合温度和阈值用；不给 dev 就从 train 切 10% |
| `epochs` `micro_batch` `grad_accum` | 4 / 8 / 4 | 与 notebook 一致 |
| `group_size` `sigma_start` `sigma_end` `w_sph` | 4 / 0.4 / 0.1 / 0.75 | RLCD 超参，与 notebook 一致 |
| `rl_weight` `ce_weight` | 1.0 / 1.0 | 两项损失的权重；`rl_weight=0` 退化为纯 BCE 基线 |
| `pos_weight` | 1.0 | 标签很稀疏（几十上百个标签）时调大，给正例更高权重 |
| `labels_per_seq` | 自动 | 每条序列放多少标签；默认在位置上限内尽量全放，放不下自动分块 |
| `state_budget` | 128 | 给输入文本预留的 token 数，长文本调大 |
| `threshold_mode` | `global` | `fixed`(0.5) / `global`(一个阈值, 最大化 micro-F1) / `per_label` |
| `freeze_encoder` | false | 只训决策头，显存小很多 |
| `gradient_checkpointing` | true | 与 notebook 一致。421M 全参数 fp32 微调、micro-batch 8：开 ≈ 9 GB，关 ≈ 15 GB；显存富余时关掉可快约 30% |
| `patience` | 0 | dev 连续 N 个 epoch 不涨就停 |
| `limit_train` `limit_eval` | – | 调试用的小规模运行 |

训练过程：epoch 0 先评一次初始权重（也作为候选）→ 每个 epoch 在 dev 上选最优 → 重新载入最优权重 →
在 dev 上拟合温度、搜索阈值 → 评 dev / test。输出目录：

```
out/model.safetensors  encoder/  tokenizer/      标准 Laya checkpoint
out/rl_agent_config.json                         多了 "multilabel"（标签、布局、marker、温度、阈值）和 "training" 两节
out/metrics.json  out/train_log.jsonl            metrics.json 里的 "run" 一节记录参数量、峰值显存、耗时、推理延迟
```

LoRA 运行（`lora_r > 0`）写的是 **adapter-only checkpoint**：`adapter/`（peft 格式）+ `model.safetensors`（只有决策头，几 MB）+
配置和 tokenizer。加载时从 `multilabel.base.init` 记录的 hub id 重新取主干，把 adapter merge 进去；原版 `laya.Agent` 不认这种目录，
`MultiLabelAgent` 两种都认。

### 3. 评估与预测

```bash
python -m laya.multilabel eval    --model out --data test.jsonl
python -m laya.multilabel predict --model out --text "play some jazz and book a table for two"
python -m laya.multilabel predict --model out --input in.jsonl --output pred.jsonl
```

```python
from laya.multilabel import MultiLabelAgent

agent = MultiLabelAgent("out")            # 本地目录或 hub id
res = agent.predict("play some jazz by miles davis and book a table for two")
res["labels"]          # ['BookRestaurant', 'PlayMusic']
res["probabilities"]   # {'AddToPlaylist': 0.003, 'BookRestaurant': 0.98, ...}  已做温度校准
res["confidence"]      # 0.95，所有标签判定同时正确的概率（独立性假设），可用于转人工的门控

agent.predict_batch(list_of_texts, batch_size=32)
agent.agent.predict(state, questions)     # 同一份权重仍是普通的 laya.Agent
```

### 4. 实验记录

每个 run 目录里始终有 `train_log.jsonl`（逐 epoch）和 `metrics.json`（最终指标 + 参数量、显存、耗时、延迟），
不依赖任何服务。要对比多次实验，接 **trackio**（本地 SQLite + 网页 dashboard，`pip install trackio`）或 wandb，
两者 API 相同：

```bash
python -m laya.multilabel train --config cfg.json --tracker trackio --project my-intents --run-name qwen_lora16
trackio show --project my-intents          # 打开 dashboard；多个 run 叠在一张图上比较
```

记录的内容：`train/*`（每 `log_every` 步一次：loss、rl、ce、reward、σ、lr）、`dev/*` 和 `epoch_train/*`（每 epoch）、
`final/dev|test/*`（校准后的最终指标）、`run/*`（参数量、显存、耗时、延迟），config 里是完整的 TrainConfig。
`--tracker-space <user/space>` 可以把 trackio dashboard 同步到一个 HF Space（wandb 下这个参数是 entity）。
DDP 下只有 rank 0 记录。

### 5. 消融跑批

```bash
# 内联定义变体：名字 + 覆盖 TrainConfig 字段
python examples/multilabel_intent/ablate.py run --base examples/multilabel_intent/config.mixsnips.json \
    --out examples/multilabel_intent/runs/ablation \
    --set "laya_rlcd" \
    --set "laya_bce rl_weight=0" \
    --set "qwen3_0.6b_lora init=Qwen/Qwen3-0.6B-Base,lora_r=16" \
    --set "gemma4_e2b_lora init=google/gemma-4-E2B,lora_r=16"

# 或者写成 grid.json：[{"name": "...", "overrides": {...}}, ...]
python examples/multilabel_intent/ablate.py run --base cfg.json --grid grid.json --out runs/ablation
python examples/multilabel_intent/ablate.py report --out runs/ablation      # 只重新汇总
```

每个变体在独立进程里训练（内存在变体之间归还），写到 `OUT/NAME/`，已有 `metrics.json` 的变体自动跳过（删掉即重跑）。
结束后生成 `results.md` / `results.jsonl`：变体、基模、布局、LoRA、参数量、dev/test F1、exact match、ECE、最佳 epoch、
训练时长、峰值显存、推理延迟（batch 1 / batched）。加 `--tracker trackio` 则每个变体以自己的名字记成一个 run，
project 默认取 `--out` 的目录名，`trackio show --project <名字>` 就能把整张矩阵的曲线叠起来看。

## 实测

全部在一台 M4 MacBook（16 GB，MPS）上完成，只为验证 pipeline，不是调参后的最优结果。

**MixSNIPS**（7 个意图，每句 1–3 个）。dev / test 各取前 500 条；exact match = 整个意图集合完全正确。

| 初始化 | 训练量 | dev micro-F1 | test micro-F1 | test exact match | test ECE |
|---|---|---|---|---|---|
| Laya 英文 checkpoint（421M），**零样本** | 0 | 0.706 | – | – | – |
| Laya 英文 checkpoint，RLCD 微调 | 1,600 条（训练集的 4%）× 1 epoch，100 次更新，约 10 分钟 | **0.980** | **0.955** | **0.880** | 0.013 |
| BERT-mini（11M，裸 encoder + 新决策头），RLCD | 4,000 条 × 3 epoch，约 2 分钟 | 0.952 | 0.921 | 0.760 | 0.017 |
| 同上，但 `rl_weight=0`（纯 BCE 对照） | 同上 | 0.953 | 0.919 | 0.768 | 0.015 |

- 从 Laya checkpoint 出发收益很明显：零样本就有 0.71，用 4% 的数据训 100 步，test micro-F1 0.955。
  这些数字来自极小的训练预算，**不能**和论文里全量训练的结果直接比较。
- **RLCD 与纯 BCE 在这个规模上打平**（0.921 vs 0.919，差异在噪声范围内）。RL 项在这里没有带来可见的
  精度提升；它的理论价值在校准和软标签上（奖励的最优点就是真实概率），需要更大规模的对照才能下结论。
  想要更快更稳的基线，把 `rl_weight` 设为 0 即可。
- 校准：温度拟合后 ECE 在 0.013–0.017，`confidence` 可以直接用来做转人工门控
  （域外句子 "what is the capital of france" → 不输出任何意图，confidence 0.57）。
- 微调后同一份权重交给原版 `laya.Agent` 回答普通问题，抽查结果与微调前几乎一致
  （noul 0.44 → 0.40、0.92 → 0.91，score 1.27 → 1.20）。长时间微调后是否仍然如此没有验证。
- BERT-mini 那个 checkpoint 在**完整** test（2,199 条）上：micro-F1 0.924，exact match 0.764，ECE 0.015；
  最弱的是 `SearchCreativeWork`（F1 0.78），它和 `PlayMusic` / `SearchScreeningEvent` 语义重叠最大。

**MASSIVE zh-CN 合成多意图**（60 个意图，每句 1–3 个，dev / test 各取前 300–600 条）。这组难得多：标签多、
每个标签只有几百条样本，而且标签描述是英文（`alarm set`）、句子是中文。

| 初始化 | 训练量 | 序列 | dev micro-F1 | test micro-F1 | test exact match | test ECE |
|---|---|---|---|---|---|---|
| Laya multilingual（322M），**零样本** | 0 | 60 标签放进 1 条（`head_max_len` 自动 256 → 536） | 0.316 | 0.271 | 0.037 | 0.012 |
| Laya multilingual，RLCD 微调 | 800 条 × 1 epoch，**50 次更新**，约 15 分钟 | 同上 | 0.556 | 0.561 | 0.140 | 0.003 |
| `hfl/rbt3`（3 层中文 RoBERTa，裸 encoder） | 6,000 条 × 3 epoch，3,375 次更新，约 58 分钟 | BERT 只有 512 位置 → **自动分成 3 条 × 21 标签** | 0.617 | 0.601 | 0.142 | 0.004 |

- 两条路径（长上下文单序列 / 短上下文自动分块）都能正常训练、校准、保存和推理。
- 这里的分数都远未收敛（rbt3 每个 epoch 还在涨：0.51 → 0.59 → 0.62；multilingual 只跑了 50 步），
  只说明 pipeline 在 60 标签、中文场景下工作正常，**不代表这个任务能达到的水平**。要认真做中文，建议：
  multilingual checkpoint + 全量数据 + 多个 epoch + **中文标签描述** + 一张 CUDA 卡。
- `epochs=0` 就是"只校准不训练"：零样本评估一个 checkpoint 并写出温度和阈值。

**decoder 基模**（每家最小尺寸，MixSNIPS，LoRA r=16 打全部线性层，主干 bf16）：

| 基模 | 训练量 | test micro-F1 | exact match | ECE | 峰值内存 | checkpoint | 推理 ms/条（MPS） |
|---|---|---|---|---|---|---|---|
| Qwen3-0.6B-Base，LoRA（11.4M 可训练 / 607M） | 1,600 条 × 1 epoch，100 次更新，13 分钟 | **0.948**（全量 2,199 条）；前 500 条 0.938 | 0.838 | 0.013 | ≈2.5 GB | 55 MB（adapter 40 MB + 决策头 5 MB） | 92（batch 8）/ 112（batch 1） |
| Qwen3.5-0.8B-Base，LoRA（765M 文本主干，视觉塔不加载） | 只跑了 64 条验证通路 | – | – | – | – | – | 线性注意力走 torch 慢路径，每步约为 Qwen3 的 2 倍；CUDA 上装 `flash-linear-attention` + `causal-conv1d` |
| Gemma 4 E2B | 本机未跑 | – | – | – | 文本主干 4.65B 参数（其中 per-layer embedding 2.35B），bf16 就要 9.3 GB | – | 需要 CUDA 卡；架构路径（PLE、KV 共享、滑窗、`<mask>` 做 marker、`<bos>` 在首）用随机初始化的小模型在测试里走通 |

同样的 100 次更新预算下：Laya 英文 checkpoint 全参 0.955 > Qwen3-0.6B LoRA 0.938 > BERT-mini 全参 0.921（前 500 条 test）。
Qwen 只训了 1.9% 的参数、显存不到 Laya 全参微调的三分之一。

`ablate.py` 的一次输出示例（BERT-mini，1,000 条 × 2 epoch，test 前 300 条，噪声大约 ±0.03，只用来看格式）：

| variant | layout | lora | params (M) | trainable (M) | dev F1 | test F1 | exact | ECE | train min | peak GB |
|---|---|---|---|---|---|---|---|---|---|---|
| bertmini_rlcd | encoder | 0 | 13 | 12.9 | 0.847 | 0.760 | 0.360 | 0.048 | 0.4 | 1.6 |
| bertmini_bce | encoder | 0 | 13 | 12.9 | 0.875 | 0.791 | 0.450 | 0.038 | 0.4 | 1.6 |
| bertmini_lora16 | encoder | 16 | 13 | 2.0 | 0.868 | 0.804 | 0.440 | 0.037 | 0.5 | 1.7 |

`grid.smallest.json` 里写好了"每家最小尺寸"的变体（Laya RLCD / BCE、Qwen3-0.6B、Qwen3.5-0.8B、Gemma 4 E2B），
在 CUDA 机器上：

```bash
python examples/multilabel_intent/ablate.py run --base examples/multilabel_intent/config.mixsnips.json \
    --grid examples/multilabel_intent/grid.smallest.json --out examples/multilabel_intent/runs/smallest
```

**显存**（421M 全参数、fp32、AdamW、micro-batch 8、序列约 170 token，MPS 实测）：开梯度检查点峰值约 9 GB，
不开约 15 GB（16 GB 的机器会直接开始换页，看起来像"卡死"）。CUDA 上有混合精度，会更省。
MPS 上报告的"峰值"是按步采样的 `driver_allocated_memory`，是下界不是严格峰值；CUDA 上是 `max_memory_allocated`。

## 注意事项

- **标签描述很重要**。模型通过阅读 `名字: 描述` 来判断，`atis_flight` 这种名字最好配上一句话描述。
- **中文及其他非英文数据请用 `subfolder: "multilingual"`**（英文 checkpoint 读不了非拉丁文字，见主 README）。
- **标签很多时**（60+）：序列会变长，`head_max_len` / `max_len` 会自动放大并写回 checkpoint 配置；
  encoder 位置上限不够（如 BERT 的 512）时自动分块，推理时各块仍在同一个 batch 里一次前向完成。
- macOS 上 `torchrun --standalone` 可能因主机名解析失败而卡住，改用
  `torchrun --nnodes=1 --nproc_per_node=2 --master-addr=127.0.0.1 --master-port=29533 …`。
- **decoder 基模的显存**：LoRA 时主干 bf16 冻结，显存 ≈ 主干权重 + 激活。Qwen3-0.6B 约 2.5 GB、Qwen3.5-0.8B 约 3 GB、
  Gemma 4 E2B 光文本主干就 9.3 GB（16 GB 的 Mac 放不下训练，上 CUDA）。全参微调 decoder 走 fp32 权重 + autocast，
  1.7B 以上就需要 24 GB 以上的卡或 LoRA。
- **Qwen3.5** 的线性注意力层没有 CUDA 内核时退回 torch 实现，能跑但慢一倍以上；CUDA 上 `pip install flash-linear-attention causal-conv1d`。
- **Gemma 3 系列在 Hub 上是 gated**，需要 `HF_TOKEN` 并在页面接受许可；Gemma 4 目前不是。
- decoder 类 checkpoint（含 adapter-only）只能用 `MultiLabelAgent` 加载，`laya.load()` 会报错或答非所问。
- 混合精度只在 CUDA 上启用（T4 用 fp16 + GradScaler，Ampere 及以上用 bf16）；MPS / CPU 走 fp32。
