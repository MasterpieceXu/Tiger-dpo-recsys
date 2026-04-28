"""
Configuration file for the MovieLens-32M generative recommendation system.

The single :class:`Config` dataclass owns every knob the pipeline cares about.
For Colab / local trade-offs, use :func:`apply_preset` (or set the
``GR_PRESET`` environment variable) to flip between a handful of well-tested
defaults instead of editing fields by hand.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

# Re-exported so callers can build a DPOConfig without reaching into src/.
from src.dpo import DPOConfig as _DPOAlgoConfig


@dataclass
class DataConfig:
    data_dir: str = "dataset/ml-32m"
    ratings_file: str = "ratings.csv"
    movies_file: str = "movies.csv"
    min_rating: float = 4.0          # ratings >= this are positive
    max_seq_length: int = 50         # max user history per sample
    min_interactions: int = 5
    test_ratio: float = 0.2
    val_ratio: float = 0.1
    # Subsample top-N most active users (None = use all). Useful for
    # Colab-friendly presets that don't want to chew through 200k users.
    max_users: Optional[int] = None


@dataclass
class RQVAEConfig:
    vocab_size: int = 16384
    levels: int = 2
    dim: int = 256
    hidden_dim: int = 512
    num_layers: int = 4
    dropout: float = 0.1
    commitment_cost: float = 0.25
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-3
    warmup_steps: int = 1000
    # Use 0 on Windows or anywhere `multiprocessing.spawn` doesn't work for
    # the DataLoader (some scipy sparse matrices fail to pickle); 4 is fine
    # on Colab / Linux servers.
    dataloader_num_workers: int = 4


@dataclass
class TIGERConfig:
    model_name: str = "t5-small"
    max_length: int = 512
    num_train_epochs: int = 5
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 64
    learning_rate: float = 5e-4
    warmup_steps: int = 500
    weight_decay: float = 0.01
    logging_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 1000
    gradient_accumulation_steps: int = 1
    fp16: bool = True
    dataloader_num_workers: int = 4


@dataclass
class DPOConfig:
    """OneRec-lite DPO settings (kept separate from the algorithm-level config)."""

    enabled: bool = True
    beta: float = 0.1
    learning_rate: float = 1e-6
    weight_decay: float = 0.0
    num_epochs: int = 2
    batch_size: int = 8
    grad_clip: float = 1.0
    log_every: int = 25
    # Knobs for ``PreferenceDataBuilder``.
    num_beams_for_pairs: int = 20
    num_candidates_for_pairs: int = 10


@dataclass
class EvalConfig:
    recall_k: List[int] = field(default_factory=lambda: [10, 20, 50])
    ndcg_k: List[int] = field(default_factory=lambda: [10, 20, 50])
    num_candidates: int = 1000
    # Cap test users to keep evaluation under control on big datasets.
    max_test_users: Optional[int] = 5000
    # ItemKNN: only keep top-N nearest neighbors per item to avoid an O(I^2)
    # similarity matrix (~39 GB on ml-32m).
    knn_top_n: int = 50


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    rqvae: RQVAEConfig = field(default_factory=RQVAEConfig)
    tiger: TIGERConfig = field(default_factory=TIGERConfig)
    dpo: DPOConfig = field(default_factory=DPOConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    output_dir: str = "outputs"
    model_dir: str = "models"
    log_dir: str = "logs"

    device: str = "cuda"
    seed: int = 42

    preset: str = "default"

    def __post_init__(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        env_preset = os.environ.get("GR_PRESET")
        if env_preset:
            apply_preset(self, env_preset)

    # ---- helpers ---------------------------------------------------------

    def to_dpo_config(self) -> _DPOAlgoConfig:
        """Convert the user-facing :class:`DPOConfig` into the algorithm-level
        :class:`src.dpo.DPOConfig` consumed by :class:`src.dpo.DPOTrainer`."""
        return _DPOAlgoConfig(
            beta=self.dpo.beta,
            learning_rate=self.dpo.learning_rate,
            weight_decay=self.dpo.weight_decay,
            num_epochs=self.dpo.num_epochs,
            batch_size=self.dpo.batch_size,
            grad_clip=self.dpo.grad_clip,
            log_every=self.dpo.log_every,
            metrics_path=os.path.join(self.output_dir, "dpo_metrics.json"),
            save_dir=os.path.join(self.model_dir, "onerec_lite_dpo"),
        )


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

def apply_preset(cfg: Config, name: str) -> Config:
    """Mutate ``cfg`` in-place to match a named preset and return it.

    Available presets:

    * ``default``         – original paper-ish settings (full ml-32m).
    * ``local_smoke``     – tiny subset, runs in <30 min on CPU. Numbers are
                            meaningless – use this only to validate code paths
                            before launching a real run.
    * ``free_colab_safe`` – fits on a Colab free T4 (16 GB) and finishes in
                            ~3-4 hours. Also a good fit for any local 12-20 GB
                            GPU (e.g. RTX 3060 12 GB, RTX 4070, RTX 4060 Ti).
    * ``pro_colab_full``  – full ml-32m on Colab Pro (T4 / V100), ~8-10 hours.
                            Also a good fit for any local 22 GB+ GPU
                            (e.g. RTX 3090, RTX 4090, A5000, A100).
    * ``local_gpu``       – meta-preset that detects the largest local CUDA
                            device and dispatches to one of the above. Falls
                            back to ``local_smoke`` if no GPU is found.
    """
    name = name.lower()
    cfg.preset = name

    if name == "default":
        return cfg

    if name == "local_gpu":
        resolved = _resolve_local_gpu_preset()
        cfg.preset = resolved
        return apply_preset(cfg, resolved)

    if name == "local_smoke":
        cfg.data.max_users = 2_000
        cfg.data.max_seq_length = 30
        cfg.rqvae.epochs = 3
        cfg.rqvae.batch_size = 64
        cfg.rqvae.dataloader_num_workers = 0
        cfg.tiger.num_train_epochs = 1
        cfg.tiger.per_device_train_batch_size = 8
        cfg.tiger.gradient_accumulation_steps = 1
        cfg.tiger.max_length = 256
        # Bumped from 50/200 because at 12k train steps an eval-every-50
        # config triggers ~250 evals, each one running over the val split.
        # 1000 keeps `load_best_model_at_end` working with ~12 evals/epoch.
        cfg.tiger.eval_steps = 1000
        cfg.tiger.save_steps = 1000
        cfg.tiger.dataloader_num_workers = 0
        # Note: ``train_tiger.py`` already gates fp16 on ``torch.cuda.is_available()``,
        # so leaving it at the default ``True`` is safe on CPU too — it just gets
        # disabled there. On a local GPU this roughly halves Stage 3 wall-clock.
        cfg.dpo.num_epochs = 1
        cfg.dpo.batch_size = 4
        cfg.eval.max_test_users = 500
        return cfg

    if name == "free_colab_safe":
        cfg.data.max_users = 150_000
        cfg.rqvae.epochs = 30
        cfg.tiger.num_train_epochs = 3
        cfg.tiger.per_device_train_batch_size = 16
        cfg.tiger.gradient_accumulation_steps = 2
        cfg.tiger.max_length = 384
        cfg.tiger.dataloader_num_workers = 2
        cfg.dpo.num_epochs = 2
        cfg.dpo.batch_size = 8
        cfg.eval.max_test_users = 5_000
        return cfg

    if name == "pro_colab_full":
        cfg.data.max_users = None  # full dataset
        cfg.rqvae.epochs = 50
        cfg.tiger.num_train_epochs = 5
        cfg.tiger.per_device_train_batch_size = 24
        cfg.tiger.gradient_accumulation_steps = 2
        cfg.tiger.max_length = 512
        cfg.tiger.dataloader_num_workers = 4
        cfg.dpo.num_epochs = 3
        cfg.dpo.batch_size = 8
        cfg.eval.max_test_users = 10_000
        return cfg

    raise ValueError(
        f"Unknown preset {name!r}. Choose from: default, local_smoke, "
        "free_colab_safe, pro_colab_full, local_gpu"
    )


# Floor of (largest) GPU VRAM in GB at which each base preset becomes the
# recommended default for a local / on-prem run. Picked conservatively so
# that fp16 activations + optimiser state still fit at the configured batch
# sizes; nudge upward if you are hitting OOM.
_LOCAL_GPU_VRAM_FLOORS_GB = (
    (22.0, "pro_colab_full"),    # 3090 / 4090 / A5000 / A100 / H100
    (12.0, "free_colab_safe"),   # T4 / 3060 12GB / 4070 / 4060 Ti 16GB
    (0.0, "local_smoke"),        # tiny GPUs, integrated, or CPU-only
)


def _resolve_local_gpu_preset() -> str:
    """Pick a base preset for the current host based on detected VRAM.

    Imports ``torch`` lazily so that ``apply_preset`` stays importable in
    environments where torch isn't installed yet (e.g. running the env-check
    script before ``pip install``).
    """
    try:
        import torch
    except ImportError:
        return "local_smoke"

    if not torch.cuda.is_available():
        return "local_smoke"

    max_vram_gb = 0.0
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        max_vram_gb = max(max_vram_gb, props.total_memory / 1024**3)

    for floor_gb, preset_name in _LOCAL_GPU_VRAM_FLOORS_GB:
        if max_vram_gb >= floor_gb:
            return preset_name
    return "local_smoke"
