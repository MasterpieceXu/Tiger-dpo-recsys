# MovieLens-32M 生成式推荐系统

这是一个基于MovieLens-32M数据集的完整生成式推荐系统实现，包含RQ-VAE语义ID构建、TIGER生成式模型训练、以及OneRec-lite多项目生成等功能。

## 🚀 项目特点

- **完整的5阶段实现**：从数据预处理到端到端训练
- **低成本高效**：基于T5-small，单GPU即可训练
- **可复现**：详细的配置和脚本，4周内可完成
- **模块化设计**：每个阶段独立，便于调试和扩展

## 📁 项目结构

```
GR-movie-recommendation/
├── dataset/ml-32m/          # MovieLens-32M数据集
├── src/                     # 源代码
│   ├── rqvae.py            # RQ-VAE模型实现
│   ├── data_preprocessing.py # 数据预处理
│   ├── train_rqvae.py      # RQ-VAE训练脚本
│   ├── sequence_generator.py # 序列生成器
│   ├── tiger_model.py      # TIGER模型实现
│   ├── train_tiger.py      # TIGER训练脚本
│   ├── evaluation.py       # 评测模块
│   └── onerec_lite.py      # OneRec-lite实现
├── scripts/                 # 运行脚本
│   ├── run_pipeline.py     # 完整流程运行脚本
│   └── demo.py             # 演示脚本
├── models/                  # 保存的模型
├── outputs/                 # 输出文件
├── logs/                    # 日志文件
├── config.py               # 配置文件
├── utils.py                # 工具函数
└── requirements.txt        # 依赖包
```

## 🛠️ 环境设置

> **建议**：本仓库的代码已对齐 Python 3.11 + PyTorch 2.5 + Transformers 4.45 这套
> 与 Google Colab 完全兼容的栈。下面给出了「本地 venv」和「Colab 训练」两条路径。

### 1. 本地：创建项目内独立 venv

仓库要求至少安装一个 Python 3.11 解释器（Windows 可用 [python.org 安装包](https://www.python.org/downloads/release/python-3119/)，并勾选 *Add to PATH*）。

```powershell
# 在仓库根目录执行（PowerShell）
py -3.11 -m venv .venv
. .\scripts\activate_venv.ps1     # 或: . .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
```

```bash
# Linux / macOS
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
```

激活后 `which python` / `Get-Command python` 应当指向 `.venv` 下，而不是系统 Python。

### 2. Colab：一键训练 notebook

仓库根目录里有 [`notebooks/colab_train.ipynb`](notebooks/colab_train.ipynb)，按顺序执行 cell 即可：

1. 切到 GPU runtime（`Runtime → Change runtime type → T4 GPU`）。
2. 第一段 cell 修改 `GITHUB_URL` 为你的仓库地址，需要的话开启 `PERSIST_TO_DRIVE` 把模型回写到 Google Drive。
3. 后续 cell 会依次完成：克隆代码 → 安装依赖 → 下载 MovieLens-32M → 跑完整 pipeline → 备份产物。

完整 Stage 1~4 在 T4 上大约需 2~3 小时；想先跑通流程可以把 cell 里的 `SMOKE_TEST = True`，10 分钟内即可走完。

### 3. 下载数据（仅本地训练时需要手动下）

```bash
wget https://files.grouplens.org/datasets/movielens/ml-32m.zip
unzip ml-32m.zip -d dataset/
```

## 🎯 使用方法

### 快速开始

运行完整的5阶段流程：

```bash
python scripts/run_pipeline.py --stages 0,1,2,3,4
```

### 分阶段运行

#### 阶段0：环境检查
```bash
python scripts/run_pipeline.py --stages 0
```

#### 阶段1：构建语义ID (RQ-VAE)
```bash
python scripts/run_pipeline.py --stages 1
```

#### 阶段2：生成训练语料
```bash
python scripts/run_pipeline.py --stages 2
```

#### 阶段3：训练TIGER模型
```bash
python scripts/run_pipeline.py --stages 3
```

#### 阶段4：离线评测
```bash
python scripts/run_pipeline.py --stages 4
```

#### 阶段5：OneRec-lite (可选)
```bash
python scripts/run_pipeline.py --stages 5
```

### 演示和测试

训练完成后，可以运行演示脚本：

```bash
python scripts/demo.py
```

## 📊 5个阶段详解

### 阶段0：环境 & 数据（0.5天）
- 检查数据集是否存在
- 验证环境配置

### 阶段1：构建语义ID（RQ-VAE，2-3天）
- **目标**：将70k部电影转成固定长度的离散token
- **实现**：RQ-VAE模型，8×8两层码本
- **输出**：每部电影的语义ID序列

### 阶段2：生成训练语料（1天）
- **目标**：将用户历史交互转成「输入序列 → 下一部影片」格式
- **实现**：序列生成器，支持多种格式
- **输出**：训练/验证/测试序列文件

### 阶段3：训练轻量级TIGER（2-3天）
- **目标**：基于T5-small训练生成式推荐模型
- **实现**：自定义tokenizer + T5模型
- **输出**：训练好的TIGER模型

### 阶段4：离线评测（1天）
- **目标**：评估模型性能，对比基线方法
- **指标**：Recall@50, NDCG@50
- **基线**：Popular, ItemKNN, Random

### 阶段5：端到端OneRec-lite（+2周，可选）
- **目标**：实现会话级一次生成多部电影
- **实现**：多项目生成 + DPO偏好对齐
- **输出**：增强的推荐模型

## ⚙️ 配置说明

主要配置在`config.py`中：

```python
@dataclass
class Config:
    # 数据配置
    data: DataConfig = DataConfig()
    # RQ-VAE配置
    rqvae: RQVAEConfig = RQVAEConfig()
    # TIGER配置
    tiger: TIGERConfig = TIGERConfig()
    # 评测配置
    eval: EvalConfig = EvalConfig()
```

关键参数：
- `vocab_size`: 语义ID词汇表大小 (默认16384)
- `levels`: RQ-VAE量化层数 (默认2)
- `max_seq_length`: 最大序列长度 (默认50)
- `num_train_epochs`: 训练轮数 (默认5)

## 📈 预期结果

### 性能指标
- **Recall@50**: 0.15-0.25
- **NDCG@50**: 0.08-0.15
- **训练时间**: 2-3小时 (单张3090)

### 模型大小
- **RQ-VAE**: ~10MB
- **TIGER**: ~60MB (T5-small基础)
- **总存储**: <100MB

## 🔧 故障排除

### 常见问题

1. **内存不足**
   - 减小batch_size
   - 使用fp16训练
   - 减少max_seq_length

2. **CUDA错误**
   - 检查GPU内存
   - 设置device="cpu"进行CPU训练

3. **数据加载错误**
   - 确认数据路径正确
   - 检查文件权限

### 调试技巧

1. **单步调试**：每个阶段单独运行
2. **日志查看**：检查logs/目录下的日志文件
3. **小数据测试**：减少数据量进行快速测试

### 已知的依赖兼容性说明

仓库对应的依赖栈：

| 组件 | 版本 |
| --- | --- |
| Python | 3.11.x |
| torch | 2.2 ~ 2.5 |
| transformers | 4.41 ~ 4.45 |
| accelerate | 0.30 ~ 0.x |
| numpy | 1.26 ~ 2.0 |

代码里已经做了如下兼容修复：

- `TrainingArguments` 使用 `eval_strategy`（替代旧的 `evaluation_strategy`，4.46+ 已删除）。
- `T5Tokenizer` 替换为 `AutoTokenizer(use_fast=True)`，避免 `T5Tokenizer` 在新版本被废弃。
- T5 生成时去掉了 `do_sample=True` 与 `num_beams>1` 的冲突组合。
- Seq2Seq 训练 label padding 用 `-100` 屏蔽，避免计入交叉熵。
- `fp16` 仅在检测到 CUDA 时启用，CPU 上自动回退。
- `src/*.py` 在文件顶部统一把项目根目录插入到 `sys.path`，无论以模块还是脚本方式启动都能 import。

## 📚 参考文献

1. RQ-VAE: [Residual Quantization for Recommender Systems](https://github.com/EdoardoBotta/RQ-VAE-Recommender)
2. T5: [Exploring the Limits of Transfer Learning](https://arxiv.org/abs/1910.10683)
3. DPO: [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

MIT License

---

**注意**：这是一个研究项目，建议在学术环境中使用。生产环境使用请进行充分测试。
