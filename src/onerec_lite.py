"""
OneRec-lite: Session-level multi-item generation with DPO
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm
from transformers import Trainer, TrainingArguments

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.tiger_model import TIGERModel  # noqa: E402
from config import Config  # noqa: E402
from utils import set_seed, setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

class MultiItemDataset(Dataset):
    """Dataset for multi-item generation training"""
    
    def __init__(self, data_path: str, tokenizer, max_input_length: int = 512, 
                 max_target_length: int = 64, num_target_items: int = 5):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.num_target_items = num_target_items
        self.data = []
        
        # Load sequences
        with open(data_path, 'r') as f:
            sequences = json.load(f)
        
        # Create training samples
        for user_id, sequence in sequences.items():
            if len(sequence) < num_target_items + 2:
                continue
            
            # Create multiple samples per user
            for i in range(len(sequence) - num_target_items):
                input_seq = sequence[:i+1]
                target_seq = sequence[i+1:i+1+num_target_items]
                
                if len(target_seq) == num_target_items:
                    self.data.append({
                        'user_id': user_id,
                        'input': input_seq,
                        'target': target_seq
                    })
        
        logger.info(f"Created {len(self.data)} multi-item training samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Prepare input text
        input_text = f"<user_{sample['user_id']}> " + " ".join(sample['input'])
        target_text = " ".join(sample['target']) + " <eos>"
        
        # Tokenize
        input_encoding = self.tokenizer.base_tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_input_length,
            return_tensors='pt'
        )
        
        target_encoding = self.tokenizer.base_tokenizer(
            target_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_target_length,
            return_tensors='pt'
        )

        labels = target_encoding['input_ids'].squeeze().clone()
        labels[labels == self.tokenizer.base_tokenizer.pad_token_id] = -100

        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'labels': labels,
        }

class DPODataset(Dataset):
    """Dataset for Direct Preference Optimization"""
    
    def __init__(self, preference_data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        # Load preference data
        with open(preference_data_path, 'r') as f:
            preference_data = json.load(f)
        
        for item in preference_data:
            self.data.append({
                'input': item['input'],
                'positive': item['positive'],
                'negative': item['negative']
            })
        
        logger.info(f"Loaded {len(self.data)} preference pairs")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Tokenize input
        input_encoding = self.tokenizer.base_tokenizer(
            sample['input'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Tokenize positive and negative targets
        pos_encoding = self.tokenizer.base_tokenizer(
            sample['positive'],
            truncation=True,
            padding='max_length',
            max_length=64,
            return_tensors='pt'
        )
        
        neg_encoding = self.tokenizer.base_tokenizer(
            sample['negative'],
            truncation=True,
            padding='max_length',
            max_length=64,
            return_tensors='pt'
        )
        
        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'positive_labels': pos_encoding['input_ids'].squeeze(),
            'negative_labels': neg_encoding['input_ids'].squeeze()
        }

class DPOLoss(nn.Module):
    """Direct Preference Optimization Loss"""
    
    def __init__(self, beta: float = 0.1):
        super().__init__()
        self.beta = beta
    
    def forward(self, policy_pos_logprobs: torch.Tensor, policy_neg_logprobs: torch.Tensor,
                reference_pos_logprobs: torch.Tensor, reference_neg_logprobs: torch.Tensor):
        """
        Compute DPO loss
        
        Args:
            policy_pos_logprobs: Log probabilities of positive examples under policy model
            policy_neg_logprobs: Log probabilities of negative examples under policy model
            reference_pos_logprobs: Log probabilities of positive examples under reference model
            reference_neg_logprobs: Log probabilities of negative examples under reference model
        """
        # Calculate log ratios
        pos_ratio = policy_pos_logprobs - reference_pos_logprobs
        neg_ratio = policy_neg_logprobs - reference_neg_logprobs
        
        # DPO loss
        loss = -F.logsigmoid(self.beta * (pos_ratio - neg_ratio)).mean()
        
        return loss

class OneRecLiteTrainer:
    """Trainer for OneRec-lite model"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Set seed
        set_seed(config.seed)
    
    def train_multi_item_generation(self, base_model_path: str, sequences_dir: str):
        """Train multi-item generation capability"""
        logger.info("Training multi-item generation...")
        
        # Load base model
        model = TIGERModel.from_pretrained(base_model_path)
        model.to(self.device)
        
        # Prepare dataset
        train_dataset = MultiItemDataset(
            os.path.join(sequences_dir, "train_sequences.json"),
            model.tokenizer,
            max_input_length=self.config.tiger.max_length,
            num_target_items=5
        )
        
        val_dataset = MultiItemDataset(
            os.path.join(sequences_dir, "val_sequences.json"),
            model.tokenizer,
            max_input_length=self.config.tiger.max_length,
            num_target_items=5
        )
        
        use_fp16 = bool(self.config.tiger.fp16 and torch.cuda.is_available())

        training_args = TrainingArguments(
            output_dir=os.path.join(self.config.model_dir, "onerec_lite"),
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            learning_rate=1e-5,  # lower lr for fine-tuning
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
        
        # Trainer
        trainer = Trainer(
            model=model.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )
        
        # Train
        trainer.train()
        
        # Save model
        model_path = os.path.join(self.config.model_dir, "onerec_lite_multi")
        model.save_pretrained(model_path)
        
        logger.info("Multi-item generation training completed!")
        return model
    
    def create_preference_data(self, sequences_dir: str, model_path: str, output_path: str):
        """Create preference data for DPO training"""
        logger.info("Creating preference data...")
        
        # Load model for generating negative examples
        model = TIGERModel.from_pretrained(model_path)
        model.eval()
        model.to(self.device)
        
        # Load test sequences
        test_path = os.path.join(sequences_dir, "test_sequences.json")
        with open(test_path, 'r') as f:
            test_sequences = json.load(f)
        
        preference_data = []
        
        for user_id, sequence in tqdm(test_sequences.items(), desc="Creating preference data"):
            if len(sequence) < 8:  # Need enough items
                continue
            
            # Split sequence
            input_seq = sequence[:-5]
            positive_target = sequence[-5:]  # Last 5 items as positive
            
            # Generate negative examples using the model
            try:
                input_text = f"<user_{user_id}> " + " ".join(input_seq)
                
                # Generate multiple candidates
                semantic_recs = model.recommend(
                    input_seq,
                    num_recommendations=10,
                    num_beams=20
                )
                
                # Create negative target (different from positive)
                negative_candidates = []
                for semantic_rec in semantic_recs:
                    for semantic_id in semantic_rec:
                        token = f"<id_{semantic_id}>"
                        if token not in positive_target:
                            negative_candidates.append(token)
                
                if len(negative_candidates) >= 5:
                    negative_target = negative_candidates[:5]
                    
                    preference_data.append({
                        'user_id': user_id,
                        'input': " ".join(input_seq),
                        'positive': " ".join(positive_target),
                        'negative': " ".join(negative_target)
                    })
                
            except Exception as e:
                logger.warning(f"Error creating preference data for user {user_id}: {e}")
                continue
        
        # Save preference data
        with open(output_path, 'w') as f:
            json.dump(preference_data, f, indent=2)
        
        logger.info(f"Created {len(preference_data)} preference pairs")
        return preference_data
    
    def train_dpo(self, model_path: str, preference_data_path: str):
        """Train with Direct Preference Optimization"""
        logger.info("Starting DPO training...")
        
        # Load policy model
        policy_model = TIGERModel.from_pretrained(model_path)
        policy_model.to(self.device)
        
        # Load reference model (frozen copy)
        reference_model = TIGERModel.from_pretrained(model_path)
        reference_model.to(self.device)
        reference_model.eval()
        
        # Freeze reference model
        for param in reference_model.parameters():
            param.requires_grad = False
        
        # Prepare dataset
        dpo_dataset = DPODataset(preference_data_path, policy_model.tokenizer)
        dataloader = DataLoader(dpo_dataset, batch_size=8, shuffle=True)
        
        # Optimizer
        optimizer = torch.optim.AdamW(policy_model.parameters(), lr=1e-6)
        
        # DPO loss
        dpo_loss_fn = DPOLoss(beta=0.1)
        
        # Training loop
        policy_model.train()
        num_epochs = 2
        
        for epoch in range(num_epochs):
            total_loss = 0
            
            for batch in tqdm(dataloader, desc=f"DPO Epoch {epoch+1}"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                positive_labels = batch['positive_labels'].to(self.device)
                negative_labels = batch['negative_labels'].to(self.device)
                
                # Get log probabilities from policy model
                with torch.no_grad():
                    policy_pos_outputs = policy_model.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=positive_labels
                    )
                    policy_neg_outputs = policy_model.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=negative_labels
                    )
                
                # Get log probabilities from reference model
                with torch.no_grad():
                    ref_pos_outputs = reference_model.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=positive_labels
                    )
                    ref_neg_outputs = reference_model.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=negative_labels
                    )
                
                # Calculate DPO loss
                loss = dpo_loss_fn(
                    -policy_pos_outputs.loss,  # Convert loss to log prob
                    -policy_neg_outputs.loss,
                    -ref_pos_outputs.loss,
                    -ref_neg_outputs.loss
                )
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            logger.info(f"DPO Epoch {epoch+1}: Average Loss = {avg_loss:.4f}")
        
        # Save DPO model
        dpo_model_path = os.path.join(self.config.model_dir, "onerec_lite_dpo")
        policy_model.save_pretrained(dpo_model_path)
        
        logger.info("DPO training completed!")
        return policy_model
    
    def train_complete_pipeline(self, base_model_path: str, sequences_dir: str):
        """Train complete OneRec-lite pipeline"""
        logger.info("Starting complete OneRec-lite training pipeline...")
        
        # Step 1: Train multi-item generation
        multi_model = self.train_multi_item_generation(base_model_path, sequences_dir)
        
        # Step 2: Create preference data
        multi_model_path = os.path.join(self.config.model_dir, "onerec_lite_multi")
        preference_data_path = os.path.join(self.config.output_dir, "dpo_preference_data.json")
        
        self.create_preference_data(sequences_dir, multi_model_path, preference_data_path)
        
        # Step 3: Train with DPO
        final_model = self.train_dpo(multi_model_path, preference_data_path)
        
        logger.info("OneRec-lite training pipeline completed!")
        return final_model

if __name__ == "__main__":
    config = Config()
    logger = setup_logging(os.path.join(config.log_dir, "onerec_lite.log"))
    
    trainer = OneRecLiteTrainer(config)
    
    # Train complete pipeline
    base_model_path = os.path.join(config.model_dir, "tiger_final")
    sequences_dir = os.path.join(config.output_dir, "sequences")
    
    final_model = trainer.train_complete_pipeline(base_model_path, sequences_dir)
    
    print("OneRec-lite training completed!")
