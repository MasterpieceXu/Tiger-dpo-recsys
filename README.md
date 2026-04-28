# Tiger-DPO-RecSys

> RQ-VAE 语义 ID + TIGER 生成式推荐 + DPO 偏好对齐，在 MovieLens-32M 上的端到端实现。

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MasterpieceXu/Tiger-dpo-recsys/blob/main/notebooks/colab_train.ipynb)
[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/downloads/release/python-3119/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5-EE4C2C)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 这个仓库做了什么

把"用大语言模型做推荐"这条路径用最朴素的方式走通：

1. **RQ-VAE** 把每部电影压缩成 2 个离散 token（"语义 ID"）— 论文里叫 codebook。
2. **TIGER**（基于 T5-small）用 seq2seq 监督微调（SFT），学习"用户历史 → 下一个语义 ID"的生成。
3. **OneRec-lite + DPO**（[Rafailov et al., NeurIPS 2023](https://arxiv.org/abs/2305.18290)）拿真实下一个观看记录当 *chosen*、模型自己 beam-search 出来的非真实候选当 *rejected*，做偏好对齐。
4. 用同一份测试集，把 **TIGER + DPO / TIGER (SFT) / ItemKNN / Popular / Random** 五个模型放在一起评 `Recall@K` 和 `NDCG@K`。

最后由 [`src/report.py`](src/report.py) 自动渲染一份 [`outputs/REPORT.md`](outputs/REPORT.md)，可以直接贴到简历或文档里。

## Highlights

- ✅ **DPO 算法独立模块** [`src/dpo.py`](src/dpo.py)：正确的 sequence-level log-probability + Rafailov 原始 loss + chosen / rejected reward / margin / accuracy 五项训练指标，policy 路径**带梯度**（这是原代码的核心 bug）。
- ✅ **可扩展的 baseline 评测** [`src/evaluation.py`](src/evaluation.py)：把原来 39 GB 的稠密 cosine 矩阵换成稀疏 user×item + 每个 item 只留 Top-50 邻居，内存从 39 GB 降到 ~30 MB。
- ✅ **自动报告** [`src/report.py`](src/report.py)：从 JSON 一键渲染对比表，包含 DPO 消融行 + 训练动力学曲线。
- ✅ **三档预设** (`local_smoke` / `free_colab_safe` / `pro_colab_full`)：免费 Colab 也能完整跑一次，T4 上 3-4 小时。
- ✅ **现代依赖栈**：Python 3.11 / PyTorch 2.5 / Transformers 4.45。原 fork 上游的代码用的是已经被 4.46 删除的 API（`evaluation_strategy`、`T5Tokenizer` 等），全部修复。详见 [CHANGELOG.md](CHANGELOG.md)。

## 实验结果

> 跑完 `pro_colab_full` 预设后由 `outputs/REPORT.md` 自动填入。下面是占位符，第一次 Colab 跑完后会被脚本自动覆盖。

| Model               | Recall@10 | Recall@50 | NDCG@10 | NDCG@50 |
| ------------------- | :-------: | :-------: | :-----: | :-----: |
| **TIGER + DPO**     |     —     |     —     |    —    |    —    |
| **TIGER (SFT)**     |     —     |     —     |    —    |    —    |
| ItemKNN (Top-50)    |     —     |     —     |    —    |    —    |
| Popular             |     —     |     —     |    —    |    —    |
| Random              |     —     |     —     |    —    |    —    |

## 项目结构

```
Tiger-dpo-recsys/
├── config.py                       # 所有超参 + 预设系统
├── utils.py                        # 数据 / 指标小工具
├── requirements.txt                # 钉了版本，与 Colab 兼容
├── src/
│   ├── data_preprocessing.py       # MovieLens 清洗 + TF-IDF
│   ├── rqvae.py                    # RQ-VAE 模型
│   ├── train_rqvae.py              # Stage 1 训练入口
│   ├── sequence_generator.py       # Stage 2 用户序列生成
│   ├── tiger_model.py              # T5-small 包装 + 自定义 tokenizer
│   ├── train_tiger.py              # Stage 3 SFT 训练入口
│   ├── dpo.py                      # ★ DPO 算法模块（独立、可复用）
│   ├── onerec_lite.py              # Stage 5：多项目 SFT + 偏好对构造 + DPOTrainer
│   ├── evaluation.py               # Stage 4：sparse baselines + TIGER 多变体评测
│   └── report.py                   # Stage 6：JSON → Markdown 报告
├── scripts/
│   ├── run_pipeline.py             # 一条命令跑所有 stage
│   └── activate_venv.ps1           # PowerShell 激活本地 venv
├── notebooks/
│   └── colab_train.ipynb           # Colab 一键训练 + Drive 备份
├── dataset/ml-32m/                 # 数据集（gitignored）
├── models/                         # 训练产物（gitignored）
├── outputs/
│   ├── evaluation_results.json
│   ├── dpo_metrics.json
│   └── REPORT.md                   # 最终对比报告
└── logs/
```

## 在 Colab 上跑（推荐）

点上面那个 "Open in Colab" 按钮，然后：

1. **`Runtime → Change runtime type → T4 GPU`**
2. 在第一个 cell 里按需修改 `PRESET`：
   - `pro_colab_full`：Pro 账号、全量 ml-32m，~8-10h
   - `free_colab_safe`：免费 T4、150k 用户，~3-4h
3. **`Runtime → Run all`**

跑完之后 `outputs/REPORT.md` 里就有完整的对比表 + DPO 消融 + 训练动力学。同时所有产物自动备份到你 Drive 的 `MyDrive/tiger-dpo-recsys-runs/`。

### Colab Free vs Pro

| 项目 | Free T4 (16 GB) | Pro (T4/V100/A100) |
| --- | --- | --- |
| 推荐预设 | `free_colab_safe` | `pro_colab_full` |
| 训练数据规模 | 150k 用户 | 全量 (~200k 用户) |
| 一次会话能跑完吗 | ✅ 3-4 小时 | ✅ 8-10 小时 |
| OOM 风险 | 低（已调过 batch） | 低 |

## 在本地跑

```powershell
# Windows PowerShell
git clone https://github.com/MasterpieceXu/Tiger-dpo-recsys.git
cd Tiger-dpo-recsys

py -3.11 -m venv .venv
. .\scripts\activate_venv.ps1
pip install -r requirements.txt

# 下载数据
mkdir dataset
curl.exe -L -o dataset/ml-32m.zip https://files.grouplens.org/datasets/movielens/ml-32m.zip
Expand-Archive dataset/ml-32m.zip dataset/

# 跑流水线（CPU 上用 local_smoke 预设确认代码无 bug）
python scripts/run_pipeline.py --preset local_smoke
```

```bash
# Linux / macOS
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

wget https://files.grouplens.org/datasets/movielens/ml-32m.zip -P dataset/
unzip dataset/ml-32m.zip -d dataset/

python scripts/run_pipeline.py --preset pro_colab_full
```

## 流水线阶段

```
stage 0  环境/数据自检
stage 1  数据预处理 + RQ-VAE → outputs/item_semantic_ids.jsonl
stage 2  用户序列生成 → outputs/sequences/*.json
stage 3  TIGER SFT → models/tiger_final/
stage 4  评测：TIGER vs Popular vs ItemKNN vs Random
         → outputs/evaluation_results.json
stage 5  OneRec-lite (multi-item SFT + DPO) → models/onerec_lite_dpo/
         + outputs/dpo_preference_data.json
         + outputs/dpo_metrics.json
stage 6  渲染 outputs/REPORT.md（含 DPO 消融行）
```

可以单独跑某些 stage：

```bash
python scripts/run_pipeline.py --preset free_colab_safe --stages 3,4   # 只重训 TIGER 并评测
python scripts/run_pipeline.py --stages 6                              # 只重新渲染报告
```

## 算法细节

### 1. RQ-VAE 语义 ID（Stage 1）

每部电影的 TF-IDF 文本向量经过编码器 → 残差量化（`levels=2`）→ 得到一个二维 codebook 索引 `(c1, c2)`，作为该电影的"语义 ID"。

实现：[`src/rqvae.py`](src/rqvae.py)。

### 2. TIGER 生成式推荐（Stage 3）

用 T5-small 做 seq2seq：输入是用户历史的语义 ID 序列 `<id_a> <id_b> ...`，输出是下一部电影的语义 ID。tokenizer 在 T5 原表上额外加了 `vocab_size = 16384` 个 `<id_*>` token。

实现：[`src/tiger_model.py`](src/tiger_model.py)、[`src/train_tiger.py`](src/train_tiger.py)。

### 3. DPO 偏好对齐（Stage 5）⭐

DPO loss 实现在 [`src/dpo.py`](src/dpo.py)：

```python
# Sequence-level log-probability under any seq2seq model
def compute_sequence_logprob(model, input_ids, attention_mask, labels):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    log_probs = F.log_softmax(outputs.logits, dim=-1)
    label_mask = (labels != -100).float()
    safe_labels = labels.clamp(min=0)
    token_logp = log_probs.gather(2, safe_labels.unsqueeze(-1)).squeeze(-1)
    return (token_logp * label_mask).sum(dim=1)   # [B]

# DPO loss (Rafailov et al., 2023, Eq. 7)
def dpo_loss(pi_pos, pi_neg, ref_pos, ref_neg, beta=0.1):
    margin = (pi_pos - ref_pos) - (pi_neg - ref_neg)
    loss = -F.logsigmoid(beta * margin).mean()
    return {"loss": loss, "reward_margin": margin.mean(), "accuracy": (margin > 0).float().mean()}
```

偏好对的构造（[`src/onerec_lite.py::PreferenceDataBuilder`](src/onerec_lite.py)）：

> 对每个测试用户，把他真实的最后 N 部电影当作 *chosen*；让当前 SFT 模型对前缀做 beam search，凡是 beam 命中但**不在** chosen 集合里的候选 → *rejected*。

这样 negatives 是模型自己采样分布上的"近真"样本，而不是均匀随机噪声，是 DPO 设计目标的工作区。

### 4. Baseline & 评测（Stage 4）

[`src/evaluation.py`](src/evaluation.py) 实现三个经典 baseline 用作对照：

- **Popular**：全局热门 Top-K
- **ItemKNN**：基于 cosine 的 item-item 协同过滤，用 `sklearn.NearestNeighbors` + 每个 item Top-50 邻居（避免 O(I²) 矩阵）
- **Random**：纯随机（Recall 下界）

## 配置 / 预设

所有可调参数在 [`config.py`](config.py) 里。预设系统通过 `--preset` 或 `GR_PRESET` 环境变量切换：

| 预设 | 训练用户 | TIGER epochs | TIGER batch | DPO epochs | 评测采样 | 估计 T4 时长 |
| --- | --- | --- | --- | --- | --- | --- |
| `default` | all | 5 | 32 | 2 | 5k | 论文级，本机难跑 |
| `local_smoke` | 5k | 1 | 8 | 1 | 500 | <30 min CPU |
| `free_colab_safe` | 150k | 3 | 16 (×2 grad acc) | 2 | 5k | ~3-4h T4 |
| `pro_colab_full` | all | 5 | 24 (×2 grad acc) | 3 | 10k | ~8-10h T4 |

## 致谢

- **Upstream**：本仓库 fork 自 [xkx-youcha/GR-movie-recommendation](https://github.com/xkx-youcha/GR-movie-recommendation)，并在其基础上做了大量重写（详见 [CHANGELOG.md](CHANGELOG.md)）。
- **TIGER**：[Rajput et al., *Recommender Systems with Generative Retrieval*, NeurIPS 2023](https://arxiv.org/abs/2305.05065)
- **DPO**：[Rafailov et al., *Direct Preference Optimization*, NeurIPS 2023](https://arxiv.org/abs/2305.18290)
- **MovieLens-32M**：[GroupLens Research](https://grouplens.org/datasets/movielens/32m/)
- **T5**：[Raffel et al., 2020](https://arxiv.org/abs/1910.10683)

## License

MIT —— inherited from upstream.
