"""
Render a human-readable Markdown report from the JSON outputs of the pipeline.

Inputs
------
* ``outputs/evaluation_results.json``   – {model_name: {metric: value}}
* ``outputs/dpo_metrics.json``          – DPO training history (optional)

Output
------
* ``outputs/REPORT.md`` – a Markdown file you can paste into the README or a
  resume project section. It contains:

    1. Run metadata (preset, timestamp, dataset stats if available).
    2. A side-by-side comparison table of every model on Recall@K / NDCG@K.
    3. The DPO ablation: SFT vs SFT+DPO with deltas in pp.
    4. DPO training dynamics (per-epoch loss / reward margin / accuracy).

The script is intentionally dependency-free (only ``json`` + stdlib) so it can
be invoked from a Colab cell without needing pandas/torch.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
from typing import Any, Dict, List, Optional


METRIC_ORDER = ["Recall@10", "Recall@20", "Recall@50", "NDCG@10", "NDCG@20", "NDCG@50"]
TIGER_LABELS_PRIORITY = ["TIGER + DPO", "TIGER (SFT)", "TIGER"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _format_value(v: float) -> str:
    return f"{v:.4f}" if isinstance(v, (int, float)) else str(v)


def _format_pp_delta(new: float, old: float) -> str:
    delta = (new - old) * 100  # percentage points
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.2f} pp"


def _tiger_label(results: Dict[str, Dict[str, float]]) -> Optional[str]:
    """Return the best-available TIGER variant label for headline reporting."""
    for label in TIGER_LABELS_PRIORITY:
        if label in results:
            return label
    return None


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def _render_metadata(preset: Optional[str], extras: Optional[Dict[str, Any]]) -> str:
    rows = [
        f"- **Generated:** {_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Preset:** `{preset or 'unknown'}`",
    ]
    if extras:
        for k, v in extras.items():
            rows.append(f"- **{k}:** {v}")
    return "\n".join(rows)


def _render_comparison_table(results: Dict[str, Dict[str, float]]) -> str:
    if not results:
        return "_No evaluation results available._"

    metrics_present = [m for m in METRIC_ORDER
                       if any(m in r for r in results.values())]
    header = "| Model | " + " | ".join(metrics_present) + " |"
    sep = "| --- | " + " | ".join(":---:" for _ in metrics_present) + " |"

    # Order: TIGER variants first (DPO before SFT), then baselines.
    tiger_rows = [m for m in TIGER_LABELS_PRIORITY if m in results]
    baseline_rows = [m for m in results if m not in tiger_rows]
    ordered = tiger_rows + baseline_rows

    lines = [header, sep]
    for name in ordered:
        cells = [f"**{name}**" if name in tiger_rows else name]
        for m in metrics_present:
            cells.append(_format_value(results[name].get(m, "—")))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _render_dpo_ablation(results: Dict[str, Dict[str, float]]) -> str:
    if "TIGER + DPO" not in results or "TIGER (SFT)" not in results:
        return "_DPO ablation not available – run stage 5 to populate this section._"

    sft = results["TIGER (SFT)"]
    dpo = results["TIGER + DPO"]
    metrics_present = [m for m in METRIC_ORDER if m in sft and m in dpo]

    header = "| Metric | TIGER (SFT) | TIGER + DPO | Δ |"
    sep = "| --- | :---: | :---: | :---: |"
    lines = [header, sep]
    for m in metrics_present:
        lines.append(
            f"| {m} | {_format_value(sft[m])} | {_format_value(dpo[m])} "
            f"| {_format_pp_delta(dpo[m], sft[m])} |"
        )
    return "\n".join(lines)


def _render_dpo_history(dpo_payload: Optional[Dict[str, Any]]) -> str:
    if not dpo_payload or not dpo_payload.get("history"):
        return "_No DPO training metrics found (`outputs/dpo_metrics.json`)._"

    cfg = dpo_payload.get("config", {})
    history: List[Dict[str, float]] = dpo_payload["history"]

    lines = []
    if cfg:
        lines.append(
            f"DPO trained with `beta={cfg.get('beta')}`, "
            f"`lr={cfg.get('learning_rate')}`, "
            f"`epochs={cfg.get('num_epochs')}`, "
            f"`batch_size={cfg.get('batch_size')}`."
        )
        lines.append("")

    lines.append("| Epoch | Loss | Reward (chosen) | Reward (rejected) | Margin | Accuracy |")
    lines.append("| :---: | :---: | :---: | :---: | :---: | :---: |")
    for row in history:
        lines.append(
            f"| {row.get('epoch', '?')} "
            f"| {_format_value(row.get('loss', 0.0))} "
            f"| {_format_value(row.get('reward_chosen', 0.0))} "
            f"| {_format_value(row.get('reward_rejected', 0.0))} "
            f"| {_format_value(row.get('reward_margin', 0.0))} "
            f"| {_format_value(row.get('accuracy', 0.0))} |"
        )
    return "\n".join(lines)


def _render_resume_blurb(results: Dict[str, Dict[str, float]]) -> str:
    """A short paragraph that's safe to copy into a CV bullet."""
    headline = _tiger_label(results)
    if not headline:
        return ""
    pop = results.get("Popular", {})
    knn = results.get("ItemKNN", {})
    tiger = results[headline]

    if "Recall@50" not in tiger:
        return ""

    lines = [
        f"On the held-out MovieLens-32M test set, **{headline}** reached "
        f"Recall@50 = {tiger['Recall@50']:.4f} and NDCG@50 = "
        f"{tiger.get('NDCG@50', 0.0):.4f}.",
    ]
    if knn and "Recall@50" in knn:
        lift = (tiger["Recall@50"] - knn["Recall@50"]) * 100
        lines.append(
            f"This is **{lift:+.2f} pp** above the ItemKNN baseline "
            f"({knn['Recall@50']:.4f})"
        )
        if pop and "Recall@50" in pop:
            lift_pop = (tiger["Recall@50"] - pop["Recall@50"]) * 100
            lines[-1] += f" and **{lift_pop:+.2f} pp** above Popular ({pop['Recall@50']:.4f})."
        else:
            lines[-1] += "."
    return " ".join(lines)


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------

def generate_report(
    eval_path: str = "outputs/evaluation_results.json",
    dpo_path: str = "outputs/dpo_metrics.json",
    output_path: str = "outputs/REPORT.md",
    preset: Optional[str] = None,
    extras: Optional[Dict[str, Any]] = None,
) -> str:
    """Render the report and return its Markdown text. Also writes to disk."""
    results = _read_json(eval_path) or {}
    dpo_payload = _read_json(dpo_path)

    sections = [
        "# Experiment Report",
        "",
        "## Run metadata",
        "",
        _render_metadata(preset, extras),
        "",
        "## Headline result",
        "",
        _render_resume_blurb(results) or "_No headline result available – evaluation incomplete._",
        "",
        "## Full comparison",
        "",
        _render_comparison_table(results),
        "",
        "## DPO ablation: SFT vs SFT+DPO",
        "",
        _render_dpo_ablation(results),
        "",
        "## DPO training dynamics",
        "",
        _render_dpo_history(dpo_payload),
        "",
        "---",
        "",
        "_Auto-generated by `src/report.py` from "
        "`outputs/evaluation_results.json` and `outputs/dpo_metrics.json`._",
        "",
    ]

    md = "\n".join(sections)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)

    return md


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Render outputs/REPORT.md")
    p.add_argument("--eval", default="outputs/evaluation_results.json")
    p.add_argument("--dpo", default="outputs/dpo_metrics.json")
    p.add_argument("--out", default="outputs/REPORT.md")
    p.add_argument("--preset", default=None)
    return p


if __name__ == "__main__":
    args = _build_argparser().parse_args()
    md = generate_report(
        eval_path=args.eval,
        dpo_path=args.dpo,
        output_path=args.out,
        preset=args.preset,
    )
    print(md)
