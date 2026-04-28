# Changelog

This file documents the work done on top of the upstream
[`xkx-youcha/GR-movie-recommendation`](https://github.com/xkx-youcha/GR-movie-recommendation)
when building **Tiger-DPO-RecSys**.

## v0.2 — DPO + sparse baselines + auto report

### Added

- **`src/dpo.py`** — model-agnostic DPO module:
  - `compute_sequence_logprob()`: per-sample sequence log-probability over a
    seq2seq model, with `-100` mask handling for padded targets.
  - `dpo_loss()`: Rafailov et al. (2023) DPO objective + chosen / rejected
    reward / margin / accuracy diagnostics.
  - `PreferencePairDataset`: tokenizer-aware loader for
    `(prompt, chosen, rejected)` triples.
  - `DPOTrainer`: training loop with policy + frozen reference, gradient
    clipping, per-epoch metrics dump to `dpo_metrics.json`.
- **`src/report.py`** — renders `outputs/REPORT.md` from the JSON outputs.
  Includes:
  - Headline blurb (resume-friendly one-liner).
  - Full comparison table over all configured `Recall@K` / `NDCG@K`.
  - DPO ablation row (SFT vs SFT+DPO with `pp` deltas).
  - Per-epoch DPO training dynamics (loss / margin / accuracy).
- **Preset system in `config.py`** — `default`, `local_smoke`,
  `free_colab_safe`, `pro_colab_full`. Switchable via `--preset` or the
  `GR_PRESET` env var.
- **`DPOConfig` and `EvalConfig.max_test_users` / `EvalConfig.knn_top_n`** —
  surface-level knobs for stage 5 / 4 respectively.
- **Pipeline stage 6**: render `REPORT.md` after stages 4/5 finish.

### Changed

- **`src/onerec_lite.py`** rewritten: now only orchestrates multi-item SFT +
  preference-pair construction + delegates DPO to `src/dpo.py`. Removed the
  in-file `DPODataset` / `DPOLoss` / inlined trainer.
- **`src/evaluation.py`** rewritten:
  - `BaselineRecommender` switched from a dense
    ``cosine_similarity(items × items)`` matrix (≈ 39 GB on ml-32m) to a
    sparse `csr_matrix` + `sklearn.NearestNeighbors` with `knn_top_n=50`.
    Memory now scales as O(items × 50) instead of O(items²).
  - `RecommendationEvaluator` accepts `tiger_model_paths` so it can evaluate
    multiple TIGER variants (e.g. SFT and SFT+DPO) and surface them in the
    final report.
  - Test set is sub-sampled to `EvalConfig.max_test_users` for tractable
    evaluation on the full ml-32m.
- **`scripts/run_pipeline.py`** rewritten: clean stage banners, `--preset`
  flag, automatic stage-6 invocation.
- **`notebooks/colab_train.ipynb`** rewritten:
  - Pre-flight cell now checks GPU presence, VRAM, free disk, preset name.
  - Preset selector (`local_smoke` / `free_colab_safe` / `pro_colab_full`).
  - Renders `REPORT.md` directly via `IPython.display.Markdown`.
  - Drive backup path includes preset + timestamp.

### Removed

- The `DPODataset` / `DPOLoss` / `OneRecLiteTrainer.train_dpo` implementations
  inside `src/onerec_lite.py`. They had two correctness bugs:
  1. The policy forward was wrapped in `torch.no_grad()`, so backward pass
     produced no gradients (the trainer was secretly a no-op).
  2. `outputs.loss` (a batch-averaged cross-entropy) was used as a stand-in
     for sequence log-probability, which is mathematically wrong.

  Both are fixed by the new `src/dpo.py`.

## v0.1 — Modernization for Python 3.11 / Transformers 4.45

Compatibility fixes so the codebase actually runs on the current PyTorch + HF
stack and can be deployed on Colab.

### Compatibility

- `TrainingArguments(evaluation_strategy=...)` → `eval_strategy=...`
  (the old kwarg was removed in `transformers>=4.46`).
- `T5Tokenizer.from_pretrained(...)` → `AutoTokenizer.from_pretrained(..., use_fast=True)`
  to use the recommended fast tokenizer path.
- Seq2Seq `labels` now have padding tokens replaced by `-100` so they are
  ignored by cross-entropy loss.
- `TIGERModel.generate()` no longer mixes `do_sample=True` with `num_beams>1`
  (an incompatible combination in modern transformers).
- `TIGERModel.recommend()` no longer slices the output by encoder length —
  T5 is encoder-decoder, so `generate(...)` already returns only the decoder
  sequence.
- `fp16` auto-disabled when CUDA is unavailable (CPU/MPS would otherwise
  hard-error on training start).
- `TIGERModel.from_pretrained` rewritten to avoid double-init and to verify
  embedding size matches the (possibly extended) tokenizer.
- Every `src/*.py` now has an idempotent `sys.path` injection at the top, so
  files run as either `python src/foo.py` or `python -m src.foo`.

### Repo plumbing

- `requirements.txt` pinned to a Python 3.11 / Colab-compatible version range
  (torch 2.5.1, transformers 4.45.2, etc.).
- `.gitignore` cleaned up: keeps `notebooks/*.ipynb` tracked, ignores HF
  cache.
- `scripts/activate_venv.ps1` helper for PowerShell users.

### Colab

- Initial `notebooks/colab_train.ipynb` (later replaced in v0.2 with the
  preset-aware version).
