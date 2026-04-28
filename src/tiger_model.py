"""
TIGER: T5-based Generative Recommendation Model
"""
import json
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    T5Config,
    T5ForConditionalGeneration,
)


class TIGERTokenizer:
    """Custom tokenizer for TIGER model with semantic IDs"""

    def __init__(self, base_model: str = "t5-small", vocab_size: int = 16384):
        # AutoTokenizer transparently picks the fast tokenizer when available and
        # is the recommended replacement for the deprecated `T5Tokenizer` class.
        self.base_tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
        self.vocab_size = vocab_size

        # Add semantic ID tokens
        semantic_tokens = [f"<id_{i}>" for i in range(vocab_size)]
        special_tokens = ["<user>", "<eos>", "<unk>", "<pad>", "<mask>"]

        new_tokens = semantic_tokens + special_tokens
        self.base_tokenizer.add_tokens(new_tokens)
        
        # Create token mappings
        self.semantic_id_to_token = {i: f"<id_{i}>" for i in range(vocab_size)}
        self.token_to_semantic_id = {f"<id_{i}>": i for i in range(vocab_size)}
        
    def encode(self, text: str, **kwargs) -> List[int]:
        """Encode text to token IDs"""
        return self.base_tokenizer.encode(text, **kwargs)
    
    def decode(self, token_ids: List[int], **kwargs) -> str:
        """Decode token IDs to text"""
        return self.base_tokenizer.decode(token_ids, **kwargs)
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text"""
        return self.base_tokenizer.tokenize(text)
    
    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Convert tokens to IDs"""
        return self.base_tokenizer.convert_tokens_to_ids(tokens)
    
    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """Convert IDs to tokens"""
        return self.base_tokenizer.convert_ids_to_tokens(ids)
    
    def __len__(self):
        return len(self.base_tokenizer)
    
    @property
    def pad_token_id(self):
        return self.base_tokenizer.pad_token_id
    
    @property
    def eos_token_id(self):
        return self.base_tokenizer.eos_token_id
    
    def save_pretrained(self, save_directory: str):
        """Save tokenizer"""
        self.base_tokenizer.save_pretrained(save_directory)
        
        # Save custom mappings
        mappings = {
            'semantic_id_to_token': self.semantic_id_to_token,
            'token_to_semantic_id': self.token_to_semantic_id,
            'vocab_size': self.vocab_size
        }
        
        with open(os.path.join(save_directory, 'custom_mappings.json'), 'w') as f:
            json.dump(mappings, f)
    
    @classmethod
    def from_pretrained(cls, load_directory: str):
        """Load tokenizer"""
        base_tokenizer = AutoTokenizer.from_pretrained(load_directory, use_fast=True)

        mappings_path = os.path.join(load_directory, 'custom_mappings.json')
        with open(mappings_path, 'r') as f:
            mappings = json.load(f)

        tokenizer = cls.__new__(cls)
        tokenizer.base_tokenizer = base_tokenizer
        tokenizer.vocab_size = mappings['vocab_size']
        # JSON keys are always strings; restore the int -> str mapping we use at runtime.
        tokenizer.semantic_id_to_token = {int(k): v for k, v in mappings['semantic_id_to_token'].items()}
        tokenizer.token_to_semantic_id = {k: int(v) for k, v in mappings['token_to_semantic_id'].items()}

        return tokenizer

class TIGERModel(nn.Module):
    """TIGER: T5-based Generative Recommendation Model"""
    
    def __init__(self, base_model: str = "t5-small", vocab_size: int = 16384):
        super().__init__()
        
        # Load base T5 model
        self.config = T5Config.from_pretrained(base_model)
        self.model = T5ForConditionalGeneration.from_pretrained(base_model)
        
        # Resize embeddings for new tokens
        self.tokenizer = TIGERTokenizer(base_model, vocab_size)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Store vocab size
        self.vocab_size = vocab_size
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None,
                labels: torch.Tensor = None, **kwargs):
        """Forward pass"""
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
    
    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None,
                max_new_tokens: int = 10, num_beams: int = 5,
                num_return_sequences: int = 1, **kwargs):
        """Generate recommendations.

        Uses deterministic beam search (sampling is incompatible with `num_beams>1`
        in current `transformers` releases). Pass `do_sample=True` and
        `num_beams=1` via kwargs if you want stochastic generation instead.
        """
        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            do_sample=False,
            early_stopping=True,
        )
        gen_kwargs.update(kwargs)
        return self.model.generate(**gen_kwargs)
    
    def recommend(self, user_sequence: List[str], num_recommendations: int = 10,
                 num_beams: int = 10, tokens_per_item: int = 2) -> List[List[int]]:
        """Generate recommendations for a user sequence.

        Args:
            user_sequence: list of semantic-id tokens (e.g. ``["<id_12>", "<id_7>", ...]``).
            num_recommendations: how many distinct candidate items to generate.
            num_beams: beam width — must be >= ``num_recommendations``.
            tokens_per_item: number of semantic tokens per item (matches
                ``RQVAEConfig.levels``; default 2).
        """
        self.eval()
        device = next(self.parameters()).device
        num_beams = max(num_beams, num_recommendations)

        input_text = " ".join(user_sequence)
        encoding = self.tokenizer.base_tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            outputs = self.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=tokens_per_item,
                num_beams=num_beams,
                num_return_sequences=num_recommendations,
            )

        # T5 is seq2seq: ``outputs`` already contains only the decoder output
        # (starting with the decoder-start token). Decode each beam directly.
        recommendations: List[List[int]] = []
        for output in outputs:
            decoded = self.tokenizer.base_tokenizer.decode(
                output, skip_special_tokens=False
            )
            semantic_ids: List[int] = []
            for token in decoded.split():
                if token in self.tokenizer.token_to_semantic_id:
                    semantic_ids.append(self.tokenizer.token_to_semantic_id[token])
            if semantic_ids:
                recommendations.append(semantic_ids)

        return recommendations
    
    def save_pretrained(self, save_directory: str):
        """Save model and tokenizer"""
        os.makedirs(save_directory, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(save_directory)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_directory)
        
        # Save config
        config_dict = {
            'vocab_size': self.vocab_size,
            'base_model': self.config.name_or_path if hasattr(self.config, 'name_or_path') else 't5-small'
        }
        
        with open(os.path.join(save_directory, 'tiger_config.json'), 'w') as f:
            json.dump(config_dict, f)
    
    @classmethod
    def from_pretrained(cls, load_directory: str):
        """Load model and tokenizer"""
        config_path = os.path.join(load_directory, 'tiger_config.json')
        with open(config_path, 'r') as f:
            config_dict = json.load(f)

        instance = cls.__new__(cls)
        nn.Module.__init__(instance)
        instance.vocab_size = config_dict['vocab_size']
        instance.tokenizer = TIGERTokenizer.from_pretrained(load_directory)
        instance.model = T5ForConditionalGeneration.from_pretrained(load_directory)
        instance.config = instance.model.config
        # Make sure token embeddings line up with the (possibly extended) tokenizer.
        if instance.model.get_input_embeddings().num_embeddings != len(instance.tokenizer):
            instance.model.resize_token_embeddings(len(instance.tokenizer))

        return instance
