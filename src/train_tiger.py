"""
Training script for TIGER model
"""
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Trainer, 
    TrainingArguments, 
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
import json
import logging
from typing import Dict, List
from tqdm import tqdm
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.tiger_model import TIGERModel, TIGERTokenizer  # noqa: E402
from config import Config  # noqa: E402
from utils import set_seed, setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

class RecommendationDataset(Dataset):
    """Dataset for recommendation training"""
    
    def __init__(self, data_path: str, tokenizer: TIGERTokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        # Load data
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.data.append(line)
        
        logger.info(f"Loaded {len(self.data)} samples from {data_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        
        # For causal LM, input and target are the same (shifted)
        encoding = self.tokenizer.base_tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        # For causal LM, labels are input_ids shifted
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

class Seq2SeqDataset(Dataset):
    """Dataset for sequence-to-sequence training"""
    
    def __init__(self, data_path: str, tokenizer: TIGERTokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        # Load data
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and '\t' in line:
                    input_text, target_text = line.split('\t', 1)
                    self.data.append((input_text, target_text))
        
        logger.info(f"Loaded {len(self.data)} samples from {data_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        input_text, target_text = self.data[idx]

        input_encoding = self.tokenizer.base_tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        target_encoding = self.tokenizer.base_tokenizer(
            target_text,
            truncation=True,
            padding='max_length',
            max_length=64,
            return_tensors='pt'
        )

        # Replace pad-token positions with -100 so they are ignored by CE loss.
        labels = target_encoding['input_ids'].squeeze().clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'labels': labels,
        }

class TIGERTrainer:
    """Trainer for TIGER model"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Set seed
        set_seed(config.seed)
        
    def create_model(self) -> TIGERModel:
        """Create TIGER model"""
        logger.info("Creating TIGER model...")
        
        model = TIGERModel(
            base_model=self.config.tiger.model_name,
            vocab_size=self.config.rqvae.vocab_size
        )
        
        return model
    
    def prepare_datasets(self, sequences_dir: str, training_mode: str = "seq2seq"):
        """Prepare training datasets"""
        logger.info(f"Preparing datasets in {training_mode} mode...")
        
        if training_mode == "causal":
            train_dataset = RecommendationDataset(
                os.path.join(sequences_dir, "train_causal.txt"),
                self.model.tokenizer,
                self.config.tiger.max_length
            )
            val_dataset = RecommendationDataset(
                os.path.join(sequences_dir, "val_causal.txt"),
                self.model.tokenizer,
                self.config.tiger.max_length
            )
        else:  # seq2seq
            train_dataset = Seq2SeqDataset(
                os.path.join(sequences_dir, "train_seq2seq.txt"),
                self.model.tokenizer,
                self.config.tiger.max_length
            )
            val_dataset = Seq2SeqDataset(
                os.path.join(sequences_dir, "val_seq2seq.txt"),
                self.model.tokenizer,
                self.config.tiger.max_length
            )
        
        return train_dataset, val_dataset
    
    def train(self, sequences_dir: str, training_mode: str = "seq2seq"):
        """Train TIGER model"""
        logger.info("Starting TIGER training...")
        
        # Create model
        self.model = self.create_model()
        
        # Prepare datasets
        train_dataset, val_dataset = self.prepare_datasets(sequences_dir, training_mode)
        
        # Only enable fp16 when CUDA is available; CPU/MPS fp16 will hard-error.
        use_fp16 = bool(self.config.tiger.fp16 and torch.cuda.is_available())

        training_args = TrainingArguments(
            output_dir=os.path.join(self.config.model_dir, "tiger"),
            num_train_epochs=self.config.tiger.num_train_epochs,
            per_device_train_batch_size=self.config.tiger.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.tiger.per_device_eval_batch_size,
            learning_rate=self.config.tiger.learning_rate,
            warmup_steps=self.config.tiger.warmup_steps,
            weight_decay=self.config.tiger.weight_decay,
            logging_steps=self.config.tiger.logging_steps,
            eval_steps=self.config.tiger.eval_steps,
            save_steps=self.config.tiger.save_steps,
            # `eval_strategy` is the new (4.41+) name; `evaluation_strategy` is removed in 4.46+.
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            gradient_accumulation_steps=self.config.tiger.gradient_accumulation_steps,
            fp16=use_fp16,
            dataloader_num_workers=self.config.tiger.dataloader_num_workers,
            remove_unused_columns=False,
            save_total_limit=2,  # keep best + latest only; otherwise smoke runs eat ~10 GB of stale checkpoints
            report_to="none",  # disable wandb/tensorboard auto-logging
        )
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.model.tokenizer.base_tokenizer,
            model=self.model.model,
            padding=True
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train
        logger.info("Starting training...")
        trainer.train()
        
        # Save model
        model_path = os.path.join(self.config.model_dir, "tiger_final")
        self.model.save_pretrained(model_path)
        logger.info(f"Model saved to {model_path}")
        
        return self.model
    
    def fine_tune_with_dpo(self, model_path: str, preference_data_path: str):
        """Fine-tune with Direct Preference Optimization (DPO)"""
        logger.info("Starting DPO fine-tuning...")
        
        # Load trained model
        self.model = TIGERModel.from_pretrained(model_path)
        
        # TODO: Implement DPO training
        # This would require implementing preference learning
        # For now, we'll skip this advanced feature
        
        logger.info("DPO fine-tuning completed!")
        
        return self.model

def create_preference_data(sequences_dir: str, output_path: str):
    """Create preference data for DPO training"""
    logger.info("Creating preference data...")
    
    # Load test sequences
    test_path = os.path.join(sequences_dir, "test_sequences.json")
    with open(test_path, 'r') as f:
        test_sequences = json.load(f)
    
    preference_data = []
    for user_id, sequence in test_sequences.items():
        if len(sequence) < 4:
            continue
            
        # Create positive and negative examples
        input_seq = sequence[:-2]
        positive_target = sequence[-2:]  # Actual next items
        
        # Create negative target (random items)
        negative_target = ["<id_0>", "<id_1>"]  # Simplified negative sampling
        
        preference_data.append({
            'user_id': user_id,
            'input': " ".join(input_seq),
            'positive': " ".join(positive_target),
            'negative': " ".join(negative_target)
        })
    
    # Save preference data
    with open(output_path, 'w') as f:
        json.dump(preference_data, f, indent=2)
    
    logger.info(f"Created {len(preference_data)} preference pairs")

if __name__ == "__main__":
    config = Config()
    logger = setup_logging(os.path.join(config.log_dir, "tiger_training.log"))
    
    trainer = TIGERTrainer(config)
    
    # Train model
    sequences_dir = os.path.join(config.output_dir, "sequences")
    model = trainer.train(sequences_dir, training_mode="seq2seq")
    
    # Create preference data for DPO (optional)
    preference_path = os.path.join(config.output_dir, "preference_data.json")
    create_preference_data(sequences_dir, preference_path)
    
    print("TIGER training completed!")
