# Tiger-DPO-RecSys

在 MovieLens-32M 上用 RQ-VAE 把电影压成"语义 ID"，让 T5-small 像写句子一样
生成下一部电影，再用 DPO 拿用户真实喜好做偏好对齐。最后跟 ItemKNN / Popular /
Random 几个老牌 baseline 摆在一起评。

> 这套代码原始版本来自 [xkx-youcha/GR-movie-recommendation](https://github.com/xkx-youcha/GR-movie-recommendation)。
> 我接手以后：
>
> 1. 把它从已经过期的依赖里拉回到当前能跑的栈（Python 3.11 / PyTorch 2.5 / Transformers 4.45）；
> 2. 把原本写错的 DPO 训练循环重新实现了一遍，单独抽成一个 `src/dpo.py` 模块；
> 3. 把会爆 39 GB 内存的 baseline 评测改成稀疏版本；
> 4. 加了从 JSON 直接渲染对比表的报告生成器，跑完即出可以贴到任何文档里的结果。
>
> 详细变更见 [CHANGELOG.md](CHANGELOG.md)。

## 30 秒看明白做了啥

```
ratings.csv  ──►  RQ-VAE  ──►  每部电影 = (id_a, id_b)
                                       │
用户历史(电影序列) ─►  转成 <id_*> token 序列 ──┐
                                              ▼
                                      T5-small 监督微调 (Stage 3)
                                              │
                            真实下一部电影 = chosen ─┐
                  模型自己 beam 出来的 ≠ chosen = rejected
                                              ▼
                                          DPO (Stage 5)
                                              │
                                              ▼
   评测：TIGER+DPO  vs  TIGER(SFT only)  vs  ItemKNN  vs  Popular  vs  Random
                                              │
                                              ▼
                                  outputs/REPORT.md (Markdown 对比表)
```

## 实验结果

> 第一次跑完 `pro_colab_full` 后，下面这张表会被
> [`src/report.py`](src/report.py) 自动覆盖成真实数字。

| 模型              | Recall@10 | Recall@50 | NDCG@10 | NDCG@50 |
| ----------------- | :-------: | :-------: | :-----: | :-----: |
| **TIGER + DPO**   |    —      |    —      |    —    |    —    |
| **TIGER (SFT)**   |    —      |    —      |    —    |    —    |
| ItemKNN (Top-50)  |    —      |    —      |    —    |    —    |
| Popular           |    —      |    —      |    —    |    —    |
| Random            |    —      |    —      |    —    |    —    |

跑完之后 `outputs/REPORT.md` 还会有：

- 一段一句话 headline，可以直接抄进简历项目说明
- DPO 消融行（SFT vs SFT+DPO），每个指标的 percentage point 提升
- DPO 训练动力学（chosen reward / rejected reward / margin / accuracy 的逐 epoch 曲线）

---

## 复现 — 三种姿势挑一个

### 姿势 1：本地 Windows + 自家 NVIDIA 显卡

> 在 RTX 4060 (8 GB) 上验证过，免费 Colab T4 有的活儿这里都能干，而且没有 12 小时上限。

```powershell
git clone https://github.com/MasterpieceXu/Tiger-dpo-recsys.git
cd Tiger-dpo-recsys

# 装 Python 3.11 venv（Python 必须是 3.11）
py -3.11 -m venv .venv
. .\scripts\activate_venv.ps1

# 必须用 CUDA 版 torch；下面这条会自动从官方 wheel 仓装匹配版本
pip install --upgrade pip
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 验证 GPU
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 拉数据集（240 MB）
mkdir dataset
curl.exe -L -o dataset/ml-32m.zip https://files.grouplens.org/datasets/movielens/ml-32m.zip
Expand-Archive dataset/ml-32m.zip dataset/

# 跑流水线（4060 8GB 显存推荐用 local_4060 预设；显存更大可以直接 pro_colab_full）
python scripts/run_pipeline.py --preset local_4060
```

跑完之后看：

- `outputs/REPORT.md` —— 带表格的对比报告（最有用）
- `models/tiger_final/` —— 监督微调后的 TIGER 检查点
- `models/onerec_lite_dpo/` —— DPO 对齐后的 TIGER 检查点
- `outputs/evaluation_results.json` —— 原始评测数字
- `outputs/dpo_metrics.json` —— DPO 训练逐 epoch 指标

### 姿势 2：Colab 一键

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MasterpieceXu/Tiger-dpo-recsys/blob/main/notebooks/colab_train.ipynb)

点这个按钮 → `Runtime → Change runtime type → T4 GPU` (Free) 或 `A100/V100`
(Pro) → `Runtime → Run all`。

Notebook 第 1 个 cell 控制预设：

| 预设 | 适用 | 时长 |
| --- | --- | --- |
| `local_smoke` | 只想验证代码不出错 | <30 分钟，CPU 也行 |
| `free_colab_safe` | 免费 T4 | ~3-4 小时 |
| `pro_colab_full` | Pro 账号、要论文级数字 | ~8-10 小时 |

跑完后所有产物会自动备份到 `Drive/MyDrive/tiger-dpo-recsys-runs/<preset>-<时间戳>/`。

### 姿势 3：单步调试（懒得跑全程）

每个 stage 都可以独立跑，方便挂断点：

```bash
python scripts/run_pipeline.py --stages 0       # 环境/数据自检
python scripts/run_pipeline.py --stages 1       # 数据预处理 + RQ-VAE
python scripts/run_pipeline.py --stages 2       # 用户序列生成
python scripts/run_pipeline.py --stages 3       # TIGER 监督微调
python scripts/run_pipeline.py --stages 4       # 评测（含 baseline）
python scripts/run_pipeline.py --stages 5       # OneRec-lite + DPO
python scripts/run_pipeline.py --stages 6       # 仅渲染 outputs/REPORT.md
```

也可以混合：`--stages 4,6`（重新评测后立即重出报告）。

---

## 这个项目的几个"做对了"的地方

聊三个简历里值得拎出来说的细节，刚好对应代码里的三个文件：

### 1. DPO 写对了 — `src/dpo.py`

原版 `OneRecLiteTrainer.train_dpo` 的训练循环里，policy 模型的前向被
`with torch.no_grad():` 包住了。这意味着 `loss.backward()` 算出来的梯度全是
0，DPO "训练" 实际上是个空操作。修复时要做两件事：

```python
def compute_sequence_logprob(model, input_ids, attention_mask, labels):
    """给定 (prompt, response)，返回每个样本的 log P(response | prompt)。"""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    log_probs = F.log_softmax(outputs.logits, dim=-1)         # [B, L, V]
    label_mask = (labels != -100).float()                     # 屏蔽 padding
    safe_labels = labels.clamp(min=0)                         # gather 不能见 -100
    token_logp = log_probs.gather(2, safe_labels.unsqueeze(-1)).squeeze(-1)
    return (token_logp * label_mask).sum(dim=1)               # [B]

# DPO loss (Rafailov et al., NeurIPS 2023)
def dpo_loss(pi_pos, pi_neg, ref_pos, ref_neg, beta=0.1):
    margin = (pi_pos - ref_pos) - (pi_neg - ref_neg)
    loss = -F.logsigmoid(beta * margin).mean()
    return {"loss": loss, "reward_margin": margin.mean(),
            "accuracy": (margin > 0).float().mean()}
```

训练循环里 policy 路径 **不能** 套 `no_grad`，reference 路径 **必须** 套：

```python
pi_pos  = compute_sequence_logprob(policy.model, x, mask, y_pos)   # 带梯度
pi_neg  = compute_sequence_logprob(policy.model, x, mask, y_neg)
with torch.no_grad():
    ref_pos = compute_sequence_logprob(reference.model, x, mask, y_pos)
    ref_neg = compute_sequence_logprob(reference.model, x, mask, y_neg)
loss = dpo_loss(pi_pos, pi_neg, ref_pos, ref_neg, beta=0.1)["loss"]
loss.backward()
```

`DPOTrainer` 还会顺便记录每个 batch 的 chosen reward、rejected reward、
margin 和 accuracy（chosen reward > rejected reward 的比例），这四个指标
是 DPO 训练健康度的标准看板。

### 2. Baseline 评测不爆内存 — `src/evaluation.py`

原版 `BaselineRecommender._prepare_data()` 里有这么一行：

```python
self.item_similarity = cosine_similarity(item_features)   # 70k × 70k 稠密矩阵
```

ml-32m 上 ≈ **39 GB float64**。免费 Colab 12 GB RAM 必爆，Pro 25 GB 也爆，
所有人都一样的死。

修复思路：根本就不需要稠密矩阵，每个 item 只需要它最相近的 k 个邻居。改成
稀疏 user-item 矩阵 + `sklearn.neighbors.NearestNeighbors`，每个 item
只存 Top-50 邻居：

```python
self.user_item = csr_matrix(...)                                     # 稀疏 user × item
knn = NearestNeighbors(n_neighbors=51, metric='cosine', n_jobs=-1)
knn.fit(self.user_item.T)                                            # item × user
distances, indices = knn.kneighbors(self.user_item.T)
self._neighbor_idx = indices[:, 1:]                                  # [I, 50]
self._neighbor_sim = (1 - distances)[:, 1:]
```

内存从 **39 GB → 30 MB**。

### 3. 报告自动生成 — `src/report.py`

跑完所有 stage 后通过这一个脚本一键产出 Markdown：

```bash
python scripts/run_pipeline.py --stages 6
# 输出: outputs/REPORT.md
```

里面分四节：

1. **Headline**：一句话 "TIGER+DPO 在 ml-32m 测试集 Recall@50 = X.XXXX，比
   ItemKNN baseline 高 +Y.YY pp"
2. **Full comparison**：5 个模型 × 6 个指标的完整表
3. **DPO ablation**：SFT vs SFT+DPO 的逐指标 pp 提升
4. **DPO training dynamics**：每 epoch 的 loss / margin / accuracy

这个报告就是为了**直接贴到简历或文档里**而设计的。

---

## 项目结构

```
Tiger-dpo-recsys/
├── README.md                    # 你正在看
├── CHANGELOG.md                 # v0.1 / v0.2 改了什么
├── config.py                    # 所有超参 + 4 档预设
├── requirements.txt             # 钉了版本
├── utils.py                     # 数据 / 指标小工具
├── src/
│   ├── data_preprocessing.py    # ratings.csv 清洗 + 文本特征
│   ├── rqvae.py                 # 残差量化 VAE
│   ├── train_rqvae.py           # Stage 1 训练
│   ├── sequence_generator.py    # Stage 2 用户序列
│   ├── tiger_model.py           # T5-small 包装 + 自定义 tokenizer
│   ├── train_tiger.py           # Stage 3 SFT
│   ├── dpo.py                   # ★ DPO 算法核心
│   ├── onerec_lite.py           # Stage 5 多项目 SFT + 偏好对构造 + 调 dpo.py
│   ├── evaluation.py            # Stage 4 稀疏 baselines + 多 TIGER 变体评测
│   └── report.py                # Stage 6 渲染 REPORT.md
├── scripts/
│   ├── run_pipeline.py          # 一条命令跑完所有 stage
│   └── activate_venv.ps1        # Windows PowerShell 激活 venv
├── notebooks/
│   └── colab_train.ipynb        # Colab 一键训练 + Drive 备份
├── dataset/ml-32m/              # 数据（gitignored）
├── models/                      # 检查点（gitignored）
├── outputs/
│   ├── evaluation_results.json
│   ├── dpo_metrics.json
│   └── REPORT.md
└── logs/
```

## 预设有什么差别

`config.py::apply_preset` 控制四档。看着像四份配置，其实就是改了三个钮：
**用多少用户**、**TIGER 训几个 epoch / 多大 batch**、**DPO 训几个 epoch**。

| Preset | 训练用户 | TIGER epochs | TIGER batch (×grad_accum) | DPO epochs | 评测采样 |
| --- | --- | --- | --- | --- | --- |
| `default` | 全量 | 5 | 32 (×1) | 2 | 5k |
| `local_smoke` | 5k | 1 | 8 (×1) | 1 | 500 |
| `local_4060` | 100k | 3 | 8 (×4) | 2 | 5k |
| `free_colab_safe` | 150k | 3 | 16 (×2) | 2 | 5k |
| `pro_colab_full` | 全量 | 5 | 24 (×2) | 3 | 10k |

切换：

```bash
python scripts/run_pipeline.py --preset free_colab_safe
# 或者用环境变量：
GR_PRESET=free_colab_safe python scripts/run_pipeline.py
```

## 已知的几个坑

- **必须 Python 3.11**：3.13 上 sentencepiece 还没有官方 wheel，会失败。
- **必须 CUDA 版 torch**：`pip install -r requirements.txt` 默认装的是
  CPU 版，在本地 GPU 上跑要按上面"姿势 1"那条 cu121 wheel 重装。
- **第一次跑会从 HuggingFace Hub 下 t5-small**（~250 MB），跑之前确认有
  网络访问。
- **fp16 在 CPU 上会报错**：代码已经做了 `if torch.cuda.is_available()` 自
  动回退，但如果你在没有 GPU 的机器上手动改 config 把 `fp16=True`，仍然会
  炸。

## 致谢

- 上游基线：[xkx-youcha/GR-movie-recommendation](https://github.com/xkx-youcha/GR-movie-recommendation)
- TIGER：[Rajput et al., *Recommender Systems with Generative Retrieval*, NeurIPS 2023](https://arxiv.org/abs/2305.05065)
- DPO：[Rafailov et al., *Direct Preference Optimization*, NeurIPS 2023](https://arxiv.org/abs/2305.18290)
- 数据：[MovieLens-32M, GroupLens Research](https://grouplens.org/datasets/movielens/32m/)

## License

MIT。承袭自上游。
