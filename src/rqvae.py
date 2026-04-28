"""
Residual Quantized Variational AutoEncoder (RQ-VAE) for generating semantic IDs
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
import numpy as np

class VectorQuantizer(nn.Module):
    """Vector Quantization layer"""
    
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # Initialize embeddings
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
    
    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            inputs: [B, D] tensor
        Returns:
            quantized: [B, D] quantized tensor
            loss: quantization loss
            indices: [B] quantization indices
        """
        # Flatten input
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embeddings.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embeddings.weight.t()))
        
        # Get closest embeddings
        indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, indices, 1)
        
        # Quantize
        quantized = torch.matmul(encodings, self.embeddings.weight).view_as(inputs)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight through estimator
        quantized = inputs + (quantized - inputs).detach()
        
        return quantized, loss, indices.squeeze()

class ResidualQuantizer(nn.Module):
    """Residual Quantizer with multiple levels"""
    
    def __init__(self, num_embeddings: int, embedding_dim: int, num_levels: int = 2, 
                 commitment_cost: float = 0.25):
        super().__init__()
        self.num_levels = num_levels
        self.quantizers = nn.ModuleList([
            VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
            for _ in range(num_levels)
        ])
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: [B, D] input tensor
        Returns:
            quantized: [B, D] quantized tensor
            total_loss: total quantization loss
            all_indices: list of [B] indices for each level
        """
        # NOTE: use *out-of-place* ops below. ``residual = x`` is just a
        # name binding, so ``residual -= q`` would mutate the encoder
        # output in-place and break autograd on PyTorch >= 2.4.
        residual = x
        quantized = torch.zeros_like(x)
        total_loss = 0
        all_indices = []

        for quantizer in self.quantizers:
            q, loss, indices = quantizer(residual)
            quantized = quantized + q
            residual = residual - q
            total_loss = total_loss + loss
            all_indices.append(indices)

        return quantized, total_loss, all_indices

class RQVAE(nn.Module):
    """RQ-VAE model for item representation learning"""
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, 
                 num_levels: int = 2, num_layers: int = 4, dropout: float = 0.1,
                 commitment_cost: float = 0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_levels = num_levels
        
        # Text encoder (for movie titles, genres, etc.)
        self.text_encoder = nn.Sequential(
            nn.Linear(vocab_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            *[nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for _ in range(num_layers - 2)],
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # Residual quantizer
        self.quantizer = ResidualQuantizer(
            num_embeddings=int(vocab_size ** (1/num_levels)),
            embedding_dim=embedding_dim,
            num_levels=num_levels,
            commitment_cost=commitment_cost
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            *[nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for _ in range(num_layers - 2)],
            nn.Linear(hidden_dim, vocab_size)
        )
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Encode input to semantic IDs"""
        # Encode to continuous representation
        encoded = self.text_encoder(x)
        
        # Quantize
        quantized, _, indices = self.quantizer(encoded)
        
        return quantized, indices
    
    def decode(self, quantized: torch.Tensor) -> torch.Tensor:
        """Decode quantized representation"""
        return self.decoder(quantized)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Forward pass"""
        # Encode
        encoded = self.text_encoder(x)
        
        # Quantize
        quantized, vq_loss, indices = self.quantizer(encoded)
        
        # Decode
        reconstructed = self.decoder(quantized)
        
        return reconstructed, vq_loss, indices
    
    def get_semantic_ids(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Get semantic IDs for items"""
        with torch.no_grad():
            _, indices = self.encode(x)
        return indices
