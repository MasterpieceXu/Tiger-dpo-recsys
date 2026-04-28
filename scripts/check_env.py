"""
Standalone environment preflight for Tiger-DPO-RecSys.

Run this *before* kicking off ``scripts/run_pipeline.py`` to confirm that
the host can actually train the model end-to-end. It is safe to run on
laptops, workstations, headless servers, and Colab alike.

What it checks
--------------
* Python interpreter version and platform.
* PyTorch / Transformers versions.
* CUDA / cuDNN availability and per-device VRAM (or Apple MPS).
* A tiny CUDA matmul + backward pass to make sure the GPU stack is wired
  up correctly (catches "torch sees CUDA but kernel mismatch" early).
* Free disk space at the project root.
* Whether the MovieLens-32M ``ratings.csv`` is in place.
* Recommends a ``--preset`` for ``run_pipeline.py`` based on detected VRAM.

Usage
-----
::

    python scripts/check_env.py
    python scripts/check_env.py --json          # machine-readable output
    python scripts/check_env.py --no-cuda-test  # skip the matmul smoke test
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Pretty printing helpers (no external deps so this script also runs before
# `pip install -r requirements.txt`).
# ---------------------------------------------------------------------------

_OK = "[ok]"
_WARN = "[!]"
_ERR = "[x]"

# Toggled off by ``--json`` so that machine-readable mode emits *only* JSON
# on stdout and stays pipe-friendly.
_VERBOSE = True


def _say(msg: str = "") -> None:
    if _VERBOSE:
        print(msg)


def _print_section(title: str) -> None:
    bar = "-" * 60
    _say(f"\n{bar}\n{title}\n{bar}")


# ---------------------------------------------------------------------------
# Individual checks. Each returns a dict that is added to the JSON report,
# and prints a human-friendly summary line as a side effect.
# ---------------------------------------------------------------------------

def check_python() -> Dict[str, Any]:
    info = {
        "python_version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "executable": sys.executable,
    }
    major, minor = sys.version_info[:2]
    if (major, minor) < (3, 9):
        _say(f"{_ERR} Python {info['python_version']} is too old (need >= 3.9, recommended 3.11).")
        info["status"] = "error"
    elif (major, minor) >= (3, 13):
        _say(f"{_WARN} Python {info['python_version']}: sentencepiece may not have a wheel yet. 3.11 is the tested version.")
        info["status"] = "warn"
    else:
        _say(f"{_OK} Python {info['python_version']} on {platform.platform()}")
        info["status"] = "ok"
    return info


def check_torch(run_cuda_test: bool = True) -> Dict[str, Any]:
    info: Dict[str, Any] = {"installed": False}
    try:
        import torch  # noqa: F401
    except ImportError:
        _say(f"{_ERR} torch is not installed. Run `pip install -r requirements.txt`.")
        info["status"] = "error"
        return info

    import torch

    info["installed"] = True
    info["torch_version"] = torch.__version__
    info["cuda_available"] = bool(torch.cuda.is_available())
    info["mps_available"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    info["cuda_built"] = torch.version.cuda
    info["cudnn_version"] = torch.backends.cudnn.version() if torch.cuda.is_available() else None
    info["devices"] = []

    _say(f"{_OK} torch {torch.__version__} (CUDA build: {torch.version.cuda or 'cpu-only'})")

    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            vram_gb = props.total_memory / 1024**3
            dev = {
                "index": idx,
                "name": props.name,
                "vram_gb": round(vram_gb, 2),
                "compute_capability": f"{props.major}.{props.minor}",
            }
            info["devices"].append(dev)
            _say(
                f"{_OK}   cuda:{idx} {props.name} | VRAM {vram_gb:.1f} GB | "
                f"sm_{props.major}{props.minor}"
            )
    elif info["mps_available"]:
        info["devices"].append({"index": 0, "name": "Apple MPS", "vram_gb": None})
        _say(f"{_OK} Apple MPS backend available (training works, fp16 will be disabled).")
    else:
        _say(f"{_WARN} No GPU detected — only the `local_smoke` preset is realistic on CPU.")

    if run_cuda_test and torch.cuda.is_available():
        try:
            x = torch.randn(1024, 1024, device="cuda", requires_grad=True)
            y = (x @ x.t()).sum()
            y.backward()
            torch.cuda.synchronize()
            info["cuda_smoke_test"] = "ok"
            _say(f"{_OK} CUDA smoke test passed (1024x1024 matmul + backward).")
        except Exception as exc:
            info["cuda_smoke_test"] = f"failed: {exc}"
            _say(f"{_ERR} CUDA smoke test failed: {exc}")
            _say(
                f"{_WARN} torch sees CUDA but the kernel call crashed — "
                f"often a driver/runtime mismatch. Try a fresh `pip install torch`."
            )

    info["status"] = "ok"
    return info


def check_transformers() -> Dict[str, Any]:
    info: Dict[str, Any] = {"installed": False}
    try:
        import transformers
    except ImportError:
        _say(f"{_ERR} transformers is not installed. Run `pip install -r requirements.txt`.")
        info["status"] = "error"
        return info

    info["installed"] = True
    info["transformers_version"] = transformers.__version__

    parts = transformers.__version__.split(".")
    try:
        major = int(parts[0])
        minor = int(parts[1])
    except (IndexError, ValueError):
        major = minor = 0

    if (major, minor) >= (4, 46):
        _say(f"{_WARN} transformers {transformers.__version__}: 4.46+ removed `evaluation_strategy`; pin to <4.46.")
        info["status"] = "warn"
    elif (major, minor) < (4, 41):
        _say(f"{_WARN} transformers {transformers.__version__}: trainer expects >= 4.41.")
        info["status"] = "warn"
    else:
        _say(f"{_OK} transformers {transformers.__version__}")
        info["status"] = "ok"
    return info


def check_disk(min_free_gb: float = 5.0) -> Dict[str, Any]:
    free_gb = shutil.disk_usage(_PROJECT_ROOT).free / 1024**3
    info = {"free_gb": round(free_gb, 2), "min_required_gb": min_free_gb}
    if free_gb < min_free_gb:
        _say(f"{_ERR} Free disk: {free_gb:.1f} GB at {_PROJECT_ROOT} (need >= {min_free_gb} GB).")
        info["status"] = "error"
    else:
        _say(f"{_OK} Free disk: {free_gb:.1f} GB at {_PROJECT_ROOT}")
        info["status"] = "ok"
    return info


def check_dataset() -> Dict[str, Any]:
    # We avoid importing config.Config because that has heavier (transitive)
    # imports and we want this script runnable even if the optional deps
    # aren't installed yet. Fall back to the documented default path.
    try:
        from config import Config  # noqa: WPS433

        cfg = Config()
        ratings_path = os.path.join(cfg.data.data_dir, cfg.data.ratings_file)
        movies_path = os.path.join(cfg.data.data_dir, cfg.data.movies_file)
    except Exception:
        ratings_path = os.path.join(_PROJECT_ROOT, "dataset", "ml-32m", "ratings.csv")
        movies_path = os.path.join(_PROJECT_ROOT, "dataset", "ml-32m", "movies.csv")

    info: Dict[str, Any] = {
        "ratings_path": ratings_path,
        "movies_path": movies_path,
        "ratings_present": os.path.exists(ratings_path),
        "movies_present": os.path.exists(movies_path),
    }

    if info["ratings_present"] and info["movies_present"]:
        size_mb = os.path.getsize(ratings_path) / 1024**2
        _say(f"{_OK} MovieLens-32M found at {os.path.dirname(ratings_path)} "
              f"(ratings.csv ~ {size_mb:.0f} MB)")
        info["status"] = "ok"
    else:
        missing = [p for p, ok in [(ratings_path, info["ratings_present"]),
                                    (movies_path, info["movies_present"])] if not ok]
        _say(f"{_WARN} Missing dataset files: {missing}")
        _say("    Download:  https://files.grouplens.org/datasets/movielens/ml-32m.zip")
        info["status"] = "warn"
    return info


# ---------------------------------------------------------------------------
# Preset suggestion: pick the largest preset that fits the detected hardware.
# ---------------------------------------------------------------------------

_PRESET_VRAM_FLOOR = {
    "local_smoke": 0.0,        # any environment, including CPU
    "free_colab_safe": 12.0,   # T4 / RTX 3060 12GB / RTX 4060 Ti 16GB ...
    "pro_colab_full": 22.0,    # 3090 / 4090 / A5000 / A100 ...
}


def suggest_preset(torch_info: Dict[str, Any]) -> Dict[str, Any]:
    if not torch_info.get("cuda_available"):
        recommended = "local_smoke"
        reason = "no CUDA GPU detected (CPU/MPS run)"
    else:
        max_vram = max((d["vram_gb"] or 0.0) for d in torch_info["devices"])
        recommended = "local_smoke"
        for name, floor in _PRESET_VRAM_FLOOR.items():
            if max_vram >= floor:
                recommended = name
        reason = f"largest GPU has {max_vram:.1f} GB VRAM"

    _say(f"{_OK} Recommended preset: --preset {recommended}  ({reason})")
    _say(f"     python scripts/run_pipeline.py --preset {recommended}")
    return {"recommended_preset": recommended, "reason": reason}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pre-flight environment check for Tiger-DPO-RecSys."
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--no-cuda-test", action="store_true", help="skip the CUDA matmul smoke test")
    parser.add_argument("--min-disk-gb", type=float, default=5.0, help="minimum free disk in GB (default: 5)")
    args = parser.parse_args(argv)

    global _VERBOSE
    _VERBOSE = not args.json

    _say("=" * 60)
    _say("Tiger-DPO-RecSys :: environment check")
    _say("=" * 60)

    _print_section("1. Python")
    py_info = check_python()

    _print_section("2. PyTorch / GPU")
    torch_info = check_torch(run_cuda_test=not args.no_cuda_test)

    _print_section("3. Transformers")
    tf_info = check_transformers()

    _print_section("4. Disk")
    disk_info = check_disk(min_free_gb=args.min_disk_gb)

    _print_section("5. Dataset")
    data_info = check_dataset()

    _print_section("6. Preset suggestion")
    preset_info = suggest_preset(torch_info)

    report = {
        "python": py_info,
        "torch": torch_info,
        "transformers": tf_info,
        "disk": disk_info,
        "dataset": data_info,
        "preset": preset_info,
    }

    has_error = any(s.get("status") == "error" for s in (py_info, torch_info, tf_info, disk_info))

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        _say()
        _say("=" * 60)
        if has_error:
            _say(f"{_ERR} One or more blocking issues detected. Fix them before running the pipeline.")
        else:
            _say(f"{_OK} Environment ready. Next:")
            _say(f"     python scripts/run_pipeline.py --preset {preset_info['recommended_preset']}")
        _say("=" * 60)

    return 1 if has_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
