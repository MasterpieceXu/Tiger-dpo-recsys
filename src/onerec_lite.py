"""
OneRec-lite: session-level multi-item generation + DPO preference alignment.

This module is the *orchestration* layer for stage 5 of the pipeline. The
heavy algorithmic lifting is delegated:

* Sequence-level log-probabilities, the DPO loss and the actual training loop
  live in :mod:`src.dpo` (model-agnostic, reusable).
* The policy / reference seq2seq models come from :class:`src.tiger_model.TIGERModel`.

What stays here:

1. :class:`MultiItemDataset` – the multi-item supervised fine-tuning dataset
   used to teach the SFT model to emit several next-items in a single decode.
2. :class:`PreferenceDataBuilder` – constructs ``(prompt, chosen, rejected)``
   triples from the user's true future items vs. the SFT model's
   beam-search candidates.
3. :class:`OneRecLiteTrainer` – wires steps 1+2 together with the SFT trainer
   and the DPO trainer.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import Trainer, TrainingArguments

# Make the project root importable in any launch mode.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.dpo import DPOConfig, DPOTrainer, PreferencePairDataset  # noqa: E402
from src.tiger_model import TIGERModel  # noqa: E402
from config import Config  # noqa: E402
from utils import set_seed, setup_logging  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Multi-item generation dataset (kept from the original implementation).
# ---------------------------------------------------------------------------

class MultiItemDataset(Dataset):
    """Builds (input_seq, next-N-items) supervised samples from user sequences."""

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_input_length: int = 512,
        max_target_length: int = 64,
        num_target_items: int = 5,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.num_target_items = num_target_items
        self.data: List[Dict] = []

        with open(data_path, "r", encoding="utf-8") as f:
            sequences = json.load(f)

        for user_id, sequence in sequences.items():
            if len(sequence) < num_target_items + 2:
                continue
            for i in range(len(sequence) - num_target_items):
                input_seq = sequence[: i + 1]
                target_seq = sequence[i + 1 : i + 1 + num_target_items]
                if len(target_seq) == num_target_items:
                    self.data.append({
                        "user_id": user_id,
                        "input": input_seq,
                        "target": target_seq,
                    })

        logger.info("MultiItemDataset: built %d samples", len(self.data))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        sample = self.data[idx]
        input_text = f"<user_{sample['user_id']}> " + " ".join(sample["input"])
        target_text = " ".join(sample["target"]) + " <eos>"

        input_enc = self.tokenizer.base_tokenizer(
            input_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_input_length,
            return_tensors="pt",
        )
        target_enc = self.tokenizer.base_tokenizer(
            target_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_target_length,
            return_tensors="pt",
        )

        labels = target_enc["input_ids"].squeeze(0).clone()
        labels[labels == self.tokenizer.base_tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_enc["input_ids"].squeeze(0),
            "attention_mask": input_enc["attention_mask"].squeeze(0),
            "labels": labels,
        }


# ---------------------------------------------------------------------------
# Building DPO preference pairs from an SFT model.
# ---------------------------------------------------------------------------

class PreferenceDataBuilder:
    """Generate ``(prompt, chosen, rejected)`` triples for DPO.

    Strategy
    --------
    For each user sequence in the test set:

    * The user's actual last-N items become the *chosen* response.
    * We ask the current SFT model to beam-search several candidates from the
      same prefix; any candidate that is **not** in the chosen set becomes a
      *rejected* response.

    This is the construction recipe used in OneRec / DPO-Rec style work and
    keeps the negatives close to the model's own decoding distribution, which
    is the regime DPO is designed for (rather than uniform random negatives).
    """

    def __init__(
        self,
        sft_model: TIGERModel,
        num_target_items: int = 5,
        num_beams: int = 20,
        num_candidates: int = 10,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = sft_model
        self.model.eval()
        self.num_target_items = num_target_items
        self.num_beams = num_beams
        self.num_candidates = num_candidates
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

    def build(self, sequences_dir: str, output_path: str) -> List[Dict[str, str]]:
        test_path = os.path.join(sequences_dir, "test_sequences.json")
        with open(test_path, "r", encoding="utf-8") as f:
            test_sequences = json.load(f)

        records: List[Dict[str, str]] = []
        for user_id, sequence in tqdm(test_sequences.items(), desc="Building DPO pairs"):
            if len(sequence) < self.num_target_items + 2:
                continue

            input_seq = sequence[: -self.num_target_items]
            chosen = sequence[-self.num_target_items :]
            chosen_set = set(chosen)

            try:
                semantic_recs = self.model.recommend(
                    input_seq,
                    num_recommendations=self.num_candidates,
                    num_beams=max(self.num_beams, self.num_candidates),
                )
            except Exception as exc:
                logger.warning("recommend() failed for user %s: %s", user_id, exc)
                continue

            rejected: List[str] = []
            for rec in semantic_recs:
                for sid in rec:
                    token = f"<id_{sid}>"
                    if token not in chosen_set and token not in rejected:
                        rejected.append(token)
                        if len(rejected) >= self.num_target_items:
                            break
                if len(rejected) >= self.num_target_items:
                    break

            if len(rejected) < self.num_target_items:
                continue

            records.append({
                "user_id": user_id,
                "input": f"<user_{user_id}> " + " ".join(input_seq),
                "positive": " ".join(chosen),
                "negative": " ".join(rejected[: self.num_target_items]),
            })

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

        logger.info("Wrote %d preference pairs -> %s", len(records), output_path)
        return records


# ---------------------------------------------------------------------------
# Pipeline trainer
# ---------------------------------------------------------------------------

class OneRecLiteTrainer:
    """Stage 5 pipeline: multi-item SFT -> preference pairs -> DPO."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )
        set_seed(config.seed)

    # --------------------------------------------- multi-item SFT (T5 next-N)

    def train_multi_item_generation(
        self,
        base_model_path: str,
        sequences_dir: str,
    ) -> TIGERModel:
        logger.info("Multi-item generation fine-tuning...")

        model = TIGERModel.from_pretrained(base_model_path).to(self.device)

        train_dataset = MultiItemDataset(
            os.path.join(sequences_dir, "train_sequences.json"),
            model.tokenizer,
            max_input_length=self.config.tiger.max_length,
            num_target_items=5,
        )
        val_dataset = MultiItemDataset(
            os.path.join(sequences_dir, "val_sequences.json"),
            model.tokenizer,
            max_input_length=self.config.tiger.max_length,
            num_target_items=5,
        )

        use_fp16 = bool(self.config.tiger.fp16 and torch.cuda.is_available())
        training_args = TrainingArguments(
            output_dir=os.path.join(self.config.model_dir, "onerec_lite"),
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            learning_rate=1e-5,
            warmup_steps=100,
            weight_decay=0.01,
            logging_steps=50,
            eval_steps=200,
            save_steps=500,
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=use_fp16,
            remove_unused_columns=False,
            report_to="none",
        )

        trainer = Trainer(
            model=model.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )
        trainer.train()

        save_path = os.path.join(self.config.model_dir, "onerec_lite_multi")
        model.save_pretrained(save_path)
        logger.info("Saved multi-item model to %s", save_path)
        return model

    # --------------------------------------------------------- DPO alignment

    def run_dpo(
        self,
        sft_model_path: str,
        preference_data_path: str,
    ) -> TIGERModel:
        logger.info("Loading policy + reference TIGER models for DPO...")
        policy = TIGERModel.from_pretrained(sft_model_path).to(self.device)
        reference = TIGERModel.from_pretrained(sft_model_path).to(self.device)

        # Make `policy.parameters()` walk the underlying T5 weights.
        # Note: TIGERModel itself is an nn.Module wrapper around model + tokenizer,
        # but DPOTrainer talks to the *T5 model* directly because
        # ``compute_sequence_logprob`` calls ``model(...)`` and expects an
        # ``outputs.logits`` field (TIGERModel's __call__ returns a HuggingFace
        # Seq2SeqLMOutput too, but using .model is more explicit).
        dpo_cfg = self.config.to_dpo_config()
        dataset = PreferencePairDataset(
            preference_data_path=preference_data_path,
            tokenizer=policy.tokenizer,
            max_input_length=self.config.tiger.max_length,
            max_target_length=64,
        )

        trainer = DPOTrainer(
            policy=policy.model,          # the actual nn.Module that has logits
            reference=reference.model,
            dataset=dataset,
            cfg=dpo_cfg,
            device=self.device,
        )
        trainer.train()

        # Save the fine-tuned policy via the TIGERModel wrapper so the custom
        # tokenizer & semantic-id mappings are preserved.
        save_path = dpo_cfg.save_dir
        policy.save_pretrained(save_path)
        logger.info("Saved DPO-aligned policy to %s", save_path)
        return policy

    # ----------------------------------------------------- end-to-end driver

    def train_complete_pipeline(
        self,
        base_model_path: str,
        sequences_dir: str,
    ) -> TIGERModel:
        logger.info("Starting OneRec-lite end-to-end pipeline...")

        # Step 1: multi-item SFT.
        self.train_multi_item_generation(base_model_path, sequences_dir)

        # Step 2: build preference pairs from the SFT model.
        sft_path = os.path.join(self.config.model_dir, "onerec_lite_multi")
        sft_for_pairs = TIGERModel.from_pretrained(sft_path)
        builder = PreferenceDataBuilder(
            sft_model=sft_for_pairs,
            num_target_items=5,
            num_beams=self.config.dpo.num_beams_for_pairs,
            num_candidates=self.config.dpo.num_candidates_for_pairs,
            device=self.device,
        )
        pref_path = os.path.join(self.config.output_dir, "dpo_preference_data.json")
        builder.build(sequences_dir, pref_path)

        # Step 3: DPO.
        return self.run_dpo(sft_path, pref_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config = Config()
    setup_logging(os.path.join(config.log_dir, "onerec_lite.log"))

    trainer = OneRecLiteTrainer(config)

    base_model_path = os.path.join(config.model_dir, "tiger_final")
    sequences_dir = os.path.join(config.output_dir, "sequences")

    trainer.train_complete_pipeline(base_model_path, sequences_dir)
    print("OneRec-lite training pipeline finished!")
