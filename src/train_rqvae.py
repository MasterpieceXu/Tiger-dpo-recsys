"""
Training script for RQ-VAE model
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from tqdm import tqdm
import logging
from typing import List, Dict

# Add project root to path so the absolute imports below work whether the
# script is invoked as a module or directly.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.rqvae import RQVAE  # noqa: E402
from src.data_preprocessing import MovieLensPreprocessor, load_corpus  # noqa: E402
from config import Config  # noqa: E402
from utils import set_seed, setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

class ItemCorpusDataset(Dataset):
    """Dataset for item corpus"""
    
    def __init__(self, corpus_data: List[Dict], tfidf_features: np.ndarray):
        self.corpus_data = corpus_data
        self.tfidf_features = tfidf_features
    
    def __len__(self):
        return len(self.corpus_data)
    
    def __getitem__(self, idx):
        return {
            'movieId': self.corpus_data[idx]['movieId'],
            'encoded_movieId': self.corpus_data[idx]['encoded_movieId'],
            'features': torch.FloatTensor(self.tfidf_features[idx])
        }

class RQVAETrainer:
    """Trainer for RQ-VAE model"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Set seed
        set_seed(config.seed)
        
    def prepare_data(self):
        """Prepare training data"""
        logger.info("Preparing data...")
        
        # Check if processed data exists
        corpus_path = os.path.join(self.config.output_dir, "item_corpus.jsonl")
        
        if not os.path.exists(corpus_path):
            logger.info("Processed data not found. Running preprocessing...")
            preprocessor = MovieLensPreprocessor(
                data_dir=self.config.data.data_dir,
                min_rating=self.config.data.min_rating,
                min_interactions=self.config.data.min_interactions
            )
            preprocessor.process_data(self.config.output_dir)
        
        # Load corpus
        corpus_data, tfidf_features = load_corpus(corpus_path)
        
        # Create dataset
        dataset = ItemCorpusDataset(corpus_data, tfidf_features)
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.rqvae.batch_size,
            shuffle=True,
            num_workers=self.config.rqvae.dataloader_num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        
        return dataloader, tfidf_features.shape[1]
    
    def create_model(self, input_dim: int):
        """Create RQ-VAE model"""
        logger.info("Creating RQ-VAE model...")
        
        model = RQVAE(
            vocab_size=input_dim,
            embedding_dim=self.config.rqvae.dim,
            hidden_dim=self.config.rqvae.hidden_dim,
            num_levels=self.config.rqvae.levels,
            num_layers=self.config.rqvae.num_layers,
            dropout=self.config.rqvae.dropout,
            commitment_cost=self.config.rqvae.commitment_cost
        ).to(self.device)
        
        return model
    
    def train_epoch(self, model: nn.Module, dataloader: DataLoader, 
                   optimizer: optim.Optimizer, epoch: int):
        """Train one epoch"""
        model.train()
        total_loss = 0
        total_recon_loss = 0
        total_vq_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            features = batch['features'].to(self.device)
            
            # Forward pass
            reconstructed, vq_loss, _ = model(features)
            
            # Reconstruction loss
            recon_loss = nn.MSELoss()(reconstructed, features)
            
            # Total loss
            loss = recon_loss + vq_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_vq_loss += vq_loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Recon': f"{recon_loss.item():.4f}",
                'VQ': f"{vq_loss.item():.4f}"
            })
        
        avg_loss = total_loss / len(dataloader)
        avg_recon_loss = total_recon_loss / len(dataloader)
        avg_vq_loss = total_vq_loss / len(dataloader)
        
        logger.info(f"Epoch {epoch}: Loss={avg_loss:.4f}, Recon={avg_recon_loss:.4f}, VQ={avg_vq_loss:.4f}")
        
        return avg_loss
    
    def train(self):
        """Train RQ-VAE model"""
        logger.info("Starting RQ-VAE training...")
        
        # Prepare data
        dataloader, input_dim = self.prepare_data()
        
        # Create model
        model = self.create_model(input_dim)
        
        # Create optimizer
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config.rqvae.learning_rate,
            weight_decay=1e-5
        )
        
        # Create scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.rqvae.epochs
        )
        
        # Training loop
        best_loss = float('inf')
        for epoch in range(1, self.config.rqvae.epochs + 1):
            loss = self.train_epoch(model, dataloader, optimizer, epoch)
            scheduler.step()
            
            # Save best model
            if loss < best_loss:
                best_loss = loss
                model_path = os.path.join(self.config.model_dir, "rqvae_best.pt")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'config': self.config.rqvae,
                    'input_dim': input_dim,
                    'epoch': epoch,
                    'loss': loss
                }, model_path)
                logger.info(f"Saved best model with loss {loss:.4f}")
        
        logger.info("RQ-VAE training completed!")
        return model
    
    def generate_semantic_ids(self, model_path: str = None):
        """Generate semantic IDs for all items"""
        logger.info("Generating semantic IDs...")
        
        # Load model
        if model_path is None:
            model_path = os.path.join(self.config.model_dir, "rqvae_best.pt")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        input_dim = checkpoint['input_dim']
        
        model = self.create_model(input_dim)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Load data
        corpus_path = os.path.join(self.config.output_dir, "item_corpus.jsonl")
        corpus_data, tfidf_features = load_corpus(corpus_path)
        
        # Generate semantic IDs
        semantic_ids = []
        with torch.no_grad():
            for i in tqdm(range(len(corpus_data)), desc="Generating semantic IDs"):
                features = torch.FloatTensor(tfidf_features[i]).unsqueeze(0).to(self.device)
                indices = model.get_semantic_ids(features)
                
                semantic_ids.append({
                    'movieId': corpus_data[i]['movieId'],
                    'encoded_movieId': corpus_data[i]['encoded_movieId'],
                    'semantic_ids': [idx.cpu().item() for idx in indices]
                })
        
        # Save semantic IDs
        output_path = os.path.join(self.config.output_dir, "item_semantic_ids.jsonl")
        with open(output_path, 'w') as f:
            for item in semantic_ids:
                f.write(json.dumps(item) + '\n')
        
        logger.info(f"Semantic IDs saved to {output_path}")
        return semantic_ids

if __name__ == "__main__":
    config = Config()
    logger = setup_logging(os.path.join(config.log_dir, "rqvae_training.log"))
    
    trainer = RQVAETrainer(config)
    
    # Train model
    model = trainer.train()
    
    # Generate semantic IDs
    semantic_ids = trainer.generate_semantic_ids()
    
    print(f"Generated semantic IDs for {len(semantic_ids)} items")
