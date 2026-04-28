"""
Pipeline driver for the MovieLens-32M generative recommendation system.

Stages
------
0. Environment & data sanity check
1. Data preprocessing + RQ-VAE training + semantic-ID generation
2. User sequence generation
3. TIGER (T5-small) supervised fine-tuning
4. Offline evaluation (TIGER + classical baselines) -> evaluation_results.json
5. OneRec-lite multi-item generation + DPO preference alignment
6. Render REPORT.md (always run last, even if stage 5 was skipped)

Examples
--------
::

    # The default settings (paper-ish full ml-32m)
    python scripts/run_pipeline.py --stages 0,1,2,3,4

    # The Colab-Pro full preset, including DPO and the auto-generated report
    python scripts/run_pipeline.py --preset pro_colab_full --stages 0,1,2,3,4,5,6

    # Just regenerate the report from existing JSON
    python scripts/run_pipeline.py --stages 6
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Optional

# Project root on path so absolute imports work whether the user runs us as
# a module or a script.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import Config, apply_preset  # noqa: E402
from utils import setup_logging  # noqa: E402

from src.data_preprocessing import MovieLensPreprocessor  # noqa: E402
from src.evaluation import RecommendationEvaluator  # noqa: E402
from src.onerec_lite import OneRecLiteTrainer  # noqa: E402
from src.report import generate_report  # noqa: E402
from src.sequence_generator import SequenceGenerator  # noqa: E402
from src.train_rqvae import RQVAETrainer  # noqa: E402
from src.train_tiger import TIGERTrainer  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage helpers
# ---------------------------------------------------------------------------

def _banner(title: str) -> None:
    print("=" * 60)
    print(title)
    print("=" * 60)


def run_stage_0(config: Config) -> bool:
    _banner("Stage 0: Environment & data sanity check")

    # --- Python / platform ------------------------------------------------
    import platform
    import shutil

    print(f"[ok] Python      : {platform.python_version()} on {platform.platform()}")

    # --- Torch / CUDA / GPU ----------------------------------------------
    try:
        import torch

        print(f"[ok] torch       : {torch.__version__} (CUDA build: {torch.version.cuda or 'cpu-only'})")
        if torch.cuda.is_available():
            for idx in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(idx)
                vram = props.total_memory / 1024**3
                print(
                    f"[ok]   cuda:{idx} {props.name} | VRAM {vram:.1f} GB | "
                    f"sm_{props.major}{props.minor}"
                )
            try:
                x = torch.randn(512, 512, device="cuda", requires_grad=True)
                ((x @ x.t()).sum()).backward()
                torch.cuda.synchronize()
                print("[ok] CUDA smoke test passed (matmul + backward)")
            except Exception as exc:  # pragma: no cover - hardware-specific
                print(f"[!] CUDA smoke test failed: {exc}")
                print("    torch sees CUDA but the kernel call crashed — likely a driver/runtime mismatch.")
                return False
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            print("[ok] Apple MPS backend available (training works, fp16 disabled).")
        else:
            print("[!] No GPU detected — only the `local_smoke` preset is realistic on CPU.")
            if config.preset not in {"local_smoke", "default"}:
                print(f"    Current preset `{config.preset}` will be very slow without a GPU.")
    except ImportError:
        print("[x] torch is not installed. Run `pip install -r requirements.txt`.")
        return False

    # --- Disk space -------------------------------------------------------
    free_gb = shutil.disk_usage(_PROJECT_ROOT).free / 1024**3
    if free_gb < 5:
        print(f"[!] Free disk: {free_gb:.1f} GB at {_PROJECT_ROOT} (need >= 5 GB).")
        return False
    print(f"[ok] Free disk   : {free_gb:.1f} GB at {_PROJECT_ROOT}")

    # --- Dataset ----------------------------------------------------------
    ratings_path = os.path.join(config.data.data_dir, config.data.ratings_file)
    if not os.path.exists(ratings_path):
        print("[x] MovieLens-32M data not found.")
        print(f"    Expected: {ratings_path}")
        print("    Get it: https://files.grouplens.org/datasets/movielens/ml-32m.zip")
        return False
    size_mb = os.path.getsize(ratings_path) / 1024**2
    print(f"[ok] Dataset     : {ratings_path} ({size_mb:.0f} MB)")

    # --- Run config -------------------------------------------------------
    print(f"[ok] Output dir  : {config.output_dir}")
    print(f"[ok] Model dir   : {config.model_dir}")
    print(f"[ok] Preset      : {config.preset}")
    print(f"[ok] Max users   : {config.data.max_users or 'all'}")
    print(f"[ok] TIGER epochs: {config.tiger.num_train_epochs} | "
          f"DPO epochs: {config.dpo.num_epochs}")
    return True


def run_stage_1(config: Config) -> bool:
    _banner("Stage 1: Build semantic IDs (preprocess + RQ-VAE)")

    preprocessor = MovieLensPreprocessor(
        data_dir=config.data.data_dir,
        min_rating=config.data.min_rating,
        min_interactions=config.data.min_interactions,
        max_users=config.data.max_users,
    )
    ratings, movies, _ = preprocessor.process_data(config.output_dir)
    print(f"[ok] preprocessed {len(ratings)} ratings, {len(movies)} movies")

    trainer = RQVAETrainer(config)
    trainer.train()
    semantic_ids = trainer.generate_semantic_ids()
    print(f"[ok] generated semantic IDs for {len(semantic_ids)} items")
    return True


def run_stage_2(config: Config) -> bool:
    _banner("Stage 2: User sequences for next-item prediction")
    generator = SequenceGenerator(
        min_rating=config.data.min_rating,
        max_seq_length=config.data.max_seq_length,
        min_interactions=config.data.min_interactions,
    )
    sequences = generator.process_sequences(
        data_dir=config.output_dir,
        output_dir=os.path.join(config.output_dir, "sequences"),
    )
    print(
        f"[ok] train: {len(sequences['train'])} | "
        f"val: {len(sequences['val'])} | test: {len(sequences['test'])} users"
    )
    return True


def run_stage_3(config: Config) -> bool:
    _banner("Stage 3: TIGER (T5-small) supervised fine-tuning")
    trainer = TIGERTrainer(config)
    sequences_dir = os.path.join(config.output_dir, "sequences")
    trainer.train(sequences_dir, training_mode="seq2seq")
    print("[ok] TIGER SFT model saved to models/tiger_final")
    return True


def run_stage_4(config: Config) -> bool:
    _banner("Stage 4: Offline evaluation (TIGER vs classical baselines)")
    evaluator = RecommendationEvaluator(config)
    sequences_dir = os.path.join(config.output_dir, "sequences")

    tiger_paths = {
        "TIGER (SFT)": os.path.join(config.model_dir, "tiger_final"),
    }
    dpo_path = os.path.join(config.model_dir, "onerec_lite_dpo")
    if os.path.isdir(dpo_path):
        tiger_paths["TIGER + DPO"] = dpo_path

    results = evaluator.run_evaluation(
        sequences_dir=sequences_dir,
        data_dir=config.output_dir,
        tiger_model_paths=tiger_paths,
    )

    print("\nEvaluation summary:")
    for model_name, metrics in results.items():
        print(f"  {model_name}")
        for metric, value in metrics.items():
            print(f"    {metric}: {value:.4f}")
    return True


def run_stage_5(config: Config) -> bool:
    _banner("Stage 5: OneRec-lite (multi-item SFT + DPO)")
    trainer = OneRecLiteTrainer(config)
    base_model_path = os.path.join(config.model_dir, "tiger_final")
    sequences_dir = os.path.join(config.output_dir, "sequences")
    trainer.train_complete_pipeline(base_model_path, sequences_dir)
    print("[ok] OneRec-lite + DPO completed")
    return True


def run_stage_6(config: Config) -> bool:
    _banner("Stage 6: Render REPORT.md")
    md = generate_report(
        eval_path=os.path.join(config.output_dir, "evaluation_results.json"),
        dpo_path=os.path.join(config.output_dir, "dpo_metrics.json"),
        output_path=os.path.join(config.output_dir, "REPORT.md"),
        preset=config.preset,
        extras={
            "Max users (training)": config.data.max_users or "all",
            "Max test users (eval)": config.eval.max_test_users or "all",
            "TIGER epochs": config.tiger.num_train_epochs,
            "DPO epochs": config.dpo.num_epochs,
        },
    )
    print(md)
    print(f"\n[ok] REPORT.md written to {config.output_dir}/REPORT.md")
    return True


STAGE_FUNCS = {
    0: run_stage_0,
    1: run_stage_1,
    2: run_stage_2,
    3: run_stage_3,
    4: run_stage_4,
    5: run_stage_5,
    6: run_stage_6,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MovieLens-32M generative recommendation pipeline"
    )
    parser.add_argument(
        "--stages",
        type=str,
        default="0,1,2,3,4,5,6",
        help="Comma-separated list of stages to run (0-6)",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=os.environ.get("GR_PRESET", "default"),
        help=(
            "default | local_smoke | free_colab_safe | pro_colab_full | "
            "local_gpu (auto-pick by detected VRAM)"
        ),
    )
    args = parser.parse_args()

    config = Config()
    apply_preset(config, args.preset)

    setup_logging(os.path.join(config.log_dir, "pipeline.log"))

    stages_to_run = [int(s.strip()) for s in args.stages.split(",")]

    print("MovieLens-32M Generative Recommendation Pipeline")
    print("=" * 60)
    print(f"Preset       : {config.preset}")
    print(f"Stages       : {stages_to_run}")
    print(f"Output dir   : {config.output_dir}")
    print(f"Model dir    : {config.model_dir}")
    print("=" * 60)

    for stage in stages_to_run:
        fn: Optional[callable] = STAGE_FUNCS.get(stage)
        if fn is None:
            print(f"[!] Unknown stage: {stage}")
            return
        try:
            ok = fn(config)
            if not ok:
                print(f"[x] Stage {stage} failed.")
                return
            print(f"[ok] Stage {stage} done.")
        except Exception:
            logger.exception("Stage %d crashed", stage)
            print(f"[x] Stage {stage} crashed (see logs).")
            return

    print("\n" + "=" * 60)
    print("Pipeline finished.")
    print("=" * 60)


if __name__ == "__main__":
    main()
