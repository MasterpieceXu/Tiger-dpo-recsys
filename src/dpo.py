"""
Direct Preference Optimization (DPO) primitives for sequence-to-sequence models.

This module is intentionally model-agnostic: it operates on any
``transformers``-style encoder-decoder model that accepts
``(input_ids, attention_mask, labels)`` and returns ``logits``. We use it to
align the TIGER (T5-small) recommender against preference data, but the same
``DPOTrainer`` could be dropped onto any ``T5ForConditionalGeneration`` /
``BartForConditionalGeneration`` policy.

Reference
---------
Rafailov, R., Sharma, A., Mitchell, E., Manning, C. D., Ermon, S., & Finn, C.
(2023). *Direct Preference Optimization: Your Language Model is Secretly a
Reward Model.* NeurIPS 2023.

Notes on this implementation
----------------------------
The original ``onerec_lite.OneRecLiteTrainer.train_dpo`` had two correctness
bugs that this module fixes:

1. The policy forward was wrapped in ``torch.no_grad()``, so ``loss.backward()``
   silently produced no gradients and "training" was a no-op.
2. ``outputs.loss`` (a batch-averaged cross-entropy) was used in place of a
   per-sample sequence log-probability ``log P(y | x)``.

Here we compute ``log P(y | x) = sum_t log P(y_t | y_<t, x)`` per sample,
mask padding (-100) before summing, and only wrap the *reference* model with
``no_grad`` so the policy receives gradients.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------

def compute_sequence_logprob(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Return per-sample ``log P(labels | input_ids)`` under ``model``.

    Parameters
    ----------
    model
        Any seq2seq model whose forward signature accepts
        ``(input_ids, attention_mask, labels)`` and returns an object with a
        ``.logits`` field of shape ``[B, L_out, V]``. ``T5ForConditionalGeneration``
        is the obvious target.
    input_ids, attention_mask
        Encoder inputs, shapes ``[B, L_in]``.
    labels
        Decoder targets, shape ``[B, L_out]``. Padding tokens **must** be
        replaced with ``-100`` upstream so they do not contribute to the
        sequence log-prob.

    Returns
    -------
    Tensor of shape ``[B]`` – the (un-normalized) sequence log-probability for
    each sample.
    """
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )

    logits = outputs.logits                                   # [B, L_out, V]
    log_probs = F.log_softmax(logits, dim=-1)                 # [B, L_out, V]

    # Replace -100 with 0 so gather doesn't hit an out-of-range index. We mask
    # those positions out again right after gather.
    label_mask = (labels != -100).float()                     # [B, L_out]
    safe_labels = labels.clamp(min=0)                         # [B, L_out]

    token_logp = log_probs.gather(2, safe_labels.unsqueeze(-1)).squeeze(-1)
    return (token_logp * label_mask).sum(dim=1)               # [B]


def dpo_loss(
    pi_pos: torch.Tensor,
    pi_neg: torch.Tensor,
    ref_pos: torch.Tensor,
    ref_neg: torch.Tensor,
    beta: float = 0.1,
) -> Dict[str, torch.Tensor]:
    """Compute the DPO loss and a few diagnostic statistics.

    All four inputs are 1-D tensors of shape ``[B]`` containing per-sample
    sequence log-probabilities (``compute_sequence_logprob``).

    The DPO objective from Rafailov et al. (2023, Eq. 7) is

    .. math::
        L = - \\mathbb{E} \\big[ \\log \\sigma(\\beta \\cdot
            (\\log \\pi_\\theta(y_w|x) - \\log \\pi_\\text{ref}(y_w|x)
            - \\log \\pi_\\theta(y_l|x) + \\log \\pi_\\text{ref}(y_l|x))) \\big]

    Returns a dict with keys ``loss``, ``reward_chosen``, ``reward_rejected``,
    ``reward_margin`` and ``accuracy`` (fraction of pairs where the chosen
    reward exceeds the rejected reward).
    """
    # Implicit reward used by DPO: log policy - log reference.
    reward_chosen = pi_pos - ref_pos                          # [B]
    reward_rejected = pi_neg - ref_neg                        # [B]

    margin = reward_chosen - reward_rejected                  # [B]
    loss = -F.logsigmoid(beta * margin).mean()
    accuracy = (margin > 0).float().mean()

    return {
        "loss": loss,
        "reward_chosen": reward_chosen.mean().detach(),
        "reward_rejected": reward_rejected.mean().detach(),
        "reward_margin": margin.mean().detach(),
        "accuracy": accuracy.detach(),
    }


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class PreferencePairDataset(Dataset):
    """Tokenizes ``(prompt, chosen, rejected)`` triples for DPO training.

    The on-disk format is a single JSON file whose top level is a list of dicts
    with keys ``input``, ``positive``, ``negative`` (matching what
    ``onerec_lite.PreferenceDataBuilder`` produces).

    The dataset returns four tensors per sample:

    * ``input_ids``      – encoder input
    * ``attention_mask`` – encoder mask
    * ``chosen_labels``  – decoder labels for the preferred response (pad -> -100)
    * ``rejected_labels`` – decoder labels for the dispreferred response
    """

    def __init__(
        self,
        preference_data_path: str,
        tokenizer: Any,
        max_input_length: int = 512,
        max_target_length: int = 64,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

        with open(preference_data_path, "r", encoding="utf-8") as f:
            self.records: List[Dict[str, str]] = json.load(f)

        logger.info(
            "Loaded %d preference pairs from %s",
            len(self.records),
            preference_data_path,
        )

    def __len__(self) -> int:
        return len(self.records)

    def _encode_target(self, text: str) -> torch.Tensor:
        enc = self.tokenizer.base_tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_target_length,
            return_tensors="pt",
        )
        labels = enc["input_ids"].squeeze(0).clone()
        # Padding tokens are ignored during loss/log-prob computation.
        labels[labels == self.tokenizer.base_tokenizer.pad_token_id] = -100
        return labels

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]

        input_enc = self.tokenizer.base_tokenizer(
            rec["input"],
            truncation=True,
            padding="max_length",
            max_length=self.max_input_length,
            return_tensors="pt",
        )

        return {
            "input_ids": input_enc["input_ids"].squeeze(0),
            "attention_mask": input_enc["attention_mask"].squeeze(0),
            "chosen_labels": self._encode_target(rec["positive"]),
            "rejected_labels": self._encode_target(rec["negative"]),
        }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

@dataclass
class DPOConfig:
    """All knobs for ``DPOTrainer`` in one place so they can be serialized."""

    beta: float = 0.1
    learning_rate: float = 1e-6
    weight_decay: float = 0.0
    num_epochs: int = 2
    batch_size: int = 8
    grad_clip: float = 1.0
    log_every: int = 25
    metrics_path: str = "outputs/dpo_metrics.json"
    save_dir: str = "models/onerec_lite_dpo"


@dataclass
class _EpochStats:
    epoch: int = 0
    steps: int = 0
    loss: float = 0.0
    reward_chosen: float = 0.0
    reward_rejected: float = 0.0
    reward_margin: float = 0.0
    accuracy: float = 0.0

    def update(self, batch_metrics: Dict[str, torch.Tensor]) -> None:
        self.steps += 1
        self.loss += float(batch_metrics["loss"].detach())
        self.reward_chosen += float(batch_metrics["reward_chosen"])
        self.reward_rejected += float(batch_metrics["reward_rejected"])
        self.reward_margin += float(batch_metrics["reward_margin"])
        self.accuracy += float(batch_metrics["accuracy"])

    def average(self) -> Dict[str, float]:
        n = max(self.steps, 1)
        return {
            "epoch": self.epoch,
            "loss": self.loss / n,
            "reward_chosen": self.reward_chosen / n,
            "reward_rejected": self.reward_rejected / n,
            "reward_margin": self.reward_margin / n,
            "accuracy": self.accuracy / n,
        }


class DPOTrainer:
    """Train a policy seq2seq model against a frozen reference using DPO.

    The trainer is intentionally minimalist – it takes already-instantiated
    policy and reference models (so callers can choose how to load them), a
    ``PreferencePairDataset``, and a ``DPOConfig``. It handles:

    * Freezing the reference model.
    * Building a ``DataLoader`` and an ``AdamW`` optimizer.
    * Running ``num_epochs`` of training while logging ``loss``,
      ``reward_chosen``, ``reward_rejected``, ``reward_margin`` and
      ``accuracy`` per step.
    * Persisting per-epoch metrics to ``cfg.metrics_path`` so the eventual
      ``REPORT.md`` can render a chart of training dynamics.

    Saving the policy checkpoint is left to the caller (the policy model is
    usually a thin wrapper such as ``TIGERModel`` that knows how to save its
    custom tokenizer alongside the weights).
    """

    def __init__(
        self,
        policy: nn.Module,
        reference: nn.Module,
        dataset: Dataset,
        cfg: Optional[DPOConfig] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.cfg = cfg or DPOConfig()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.policy = policy.to(self.device)
        self.reference = reference.to(self.device)
        self.reference.eval()
        for p in self.reference.parameters():
            p.requires_grad_(False)

        self.dataset = dataset
        self.history: List[Dict[str, float]] = []

    # ------------------------------------------------------------------ utils

    def _collate(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        return {k: torch.stack([b[k] for b in batch], dim=0) for k in batch[0]}

    def _step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        chosen_labels = batch["chosen_labels"].to(self.device)
        rejected_labels = batch["rejected_labels"].to(self.device)

        # Policy: gradients flow.
        pi_pos = compute_sequence_logprob(
            self.policy, input_ids, attention_mask, chosen_labels
        )
        pi_neg = compute_sequence_logprob(
            self.policy, input_ids, attention_mask, rejected_labels
        )

        # Reference: frozen, no gradients ever.
        with torch.no_grad():
            ref_pos = compute_sequence_logprob(
                self.reference, input_ids, attention_mask, chosen_labels
            )
            ref_neg = compute_sequence_logprob(
                self.reference, input_ids, attention_mask, rejected_labels
            )

        return dpo_loss(pi_pos, pi_neg, ref_pos, ref_neg, beta=self.cfg.beta)

    # ------------------------------------------------------------------- train

    def train(self) -> List[Dict[str, float]]:
        """Run the configured number of epochs and return the metric history."""
        loader = DataLoader(
            self.dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            collate_fn=self._collate,
            drop_last=False,
        )

        optimizer = torch.optim.AdamW(
            (p for p in self.policy.parameters() if p.requires_grad),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )

        os.makedirs(os.path.dirname(self.cfg.metrics_path) or ".", exist_ok=True)

        for epoch in range(1, self.cfg.num_epochs + 1):
            self.policy.train()
            stats = _EpochStats(epoch=epoch)

            pbar = tqdm(loader, desc=f"DPO epoch {epoch}/{self.cfg.num_epochs}")
            for step, batch in enumerate(pbar, start=1):
                metrics = self._step(batch)
                loss = metrics["loss"]

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.cfg.grad_clip and self.cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.policy.parameters(), self.cfg.grad_clip
                    )
                optimizer.step()

                stats.update(metrics)

                if step % self.cfg.log_every == 0:
                    pbar.set_postfix({
                        "loss": f"{float(loss):.4f}",
                        "margin": f"{float(metrics['reward_margin']):.3f}",
                        "acc": f"{float(metrics['accuracy']):.2f}",
                    })

            avg = stats.average()
            self.history.append(avg)
            logger.info(
                "[DPO] epoch %d | loss=%.4f | margin=%.3f | acc=%.3f",
                avg["epoch"], avg["loss"], avg["reward_margin"], avg["accuracy"],
            )

            self._dump_metrics()

        return self.history

    def _dump_metrics(self) -> None:
        """Persist per-epoch metrics + the config used for training."""
        payload = {
            "config": asdict(self.cfg),
            "history": self.history,
        }
        with open(self.cfg.metrics_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        logger.info("DPO metrics written to %s", self.cfg.metrics_path)


# ---------------------------------------------------------------------------
# Convenience: load JSON metrics back for downstream reporting
# ---------------------------------------------------------------------------

def load_dpo_metrics(path: str) -> Dict[str, Any]:
    """Read the JSON written by :meth:`DPOTrainer._dump_metrics`."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
