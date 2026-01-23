"""
Configuration file for MovieLens-32M Generative Recommendation System
"""
import os
from dataclasses import dataclass,field
from typing import List, Optional

@dataclass
class DataConfig:
    """Data configuration"""
    data_dir: str = "dataset/ml-32m"
    ratings_file: str = "ratings.csv"
    movies_file: str = "movies.csv"
    min_rating: float = 4.0  # Minimum rating for positive samples
    max_seq_length: int = 50  # Maximum sequence length per user
    min_interactions: int = 5  # Minimum interactions per user
    test_ratio: float = 0.2  # Test set ratio
    val_ratio: float = 0.1   # Validation set ratio

@dataclass
class RQVAEConfig:
    """RQ-VAE configuration"""
    vocab_size: int = 16384
    levels: int = 2  # Number of quantization levels
    dim: int = 256   # Embedding dimension
    hidden_dim: int = 512
    num_layers: int = 4
    dropout: float = 0.1
    commitment_cost: float = 0.25
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-3
    warmup_steps: int = 1000

@dataclass
class TIGERConfig:
    """TIGER model configuration"""
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
class EvalConfig:
    """Evaluation configuration"""
    recall_k: List[int] = None
    ndcg_k: List[int] = None
    num_candidates: int = 1000
    
    def __post_init__(self):
        if self.recall_k is None:
            self.recall_k = [10, 20, 50]
        if self.ndcg_k is None:
            self.ndcg_k = [10, 20, 50]

@dataclass
class Config:
    """Main configuration"""
    data: DataConfig = field(default_factory=DataConfig)
    rqvae: RQVAEConfig = field(default_factory=RQVAEConfig)
    tiger: TIGERConfig = field(default_factory=TIGERConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    
    # Paths
    output_dir: str = "outputs"
    model_dir: str = "models"
    log_dir: str = "logs"
    
    # Device
    device: str = "cuda"
    seed: int = 42
    
    def __post_init__(self):
        # Create directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
