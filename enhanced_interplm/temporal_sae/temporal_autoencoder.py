# Temporal-SAE实现 # enhanced_interplm/temporal_sae/temporal_autoencoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
import numpy as np


class TemporalSAE(nn.Module):
    """
    Temporal-aware Sparse Autoencoder that captures feature evolution across layers.
    
    Key innovations:
    1. LSTM-based encoding to capture temporal dependencies
    2. Multi-head attention for cross-layer interactions
    3. Time decay factors for direct/indirect influence modeling
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dict_size: int,
        num_layers: int,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        time_decay_factor: float = 0.9
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dict_size = dict_size
        self.num_layers = num_layers
        self.time_decay_factor = time_decay_factor
        
        # Temporal encoder using LSTM
        self.temporal_encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        # Project LSTM output to dictionary size
        self.feature_projection = nn.Linear(hidden_dim * 2, dict_size)
        
        # Multi-head attention for cross-layer interactions
        self.cross_layer_attention = nn.MultiheadAttention(
            embed_dim=dict_size,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Temporal decoder
        self.temporal_decoder = nn.LSTM(
            input_size=dict_size,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, input_dim)
        
        # Layer-specific biases
        self.layer_biases = nn.Parameter(torch.zeros(num_layers, input_dim))
        
        # Time decay attention weights
        self.register_buffer(
            'time_decay_weights',
            self._compute_time_decay_weights(num_layers)
        )
        
    def _compute_time_decay_weights(self, num_layers: int) -> torch.Tensor:
        """Compute time decay weights for layer interactions."""
        weights = torch.zeros(num_layers, num_layers)
        for i in range(num_layers):
            for j in range(i + 1):
                weights[i, j] = self.time_decay_factor ** (i - j)
        return weights
        
    def encode(
        self, 
        embeddings_sequence: torch.Tensor,
        return_all_features: bool = False
    ) -> torch.Tensor:
        """
        Encode a sequence of embeddings from multiple layers.
        
        Args:
            embeddings_sequence: [batch_size, num_layers, seq_len, embed_dim]
            return_all_features: If True, return features for all layers
            
        Returns:
            Encoded features: [batch_size, seq_len, dict_size] or
                            [batch_size, num_layers, seq_len, dict_size]
        """
        batch_size, num_layers, seq_len, embed_dim = embeddings_sequence.shape
        
        # Reshape for LSTM processing
        embeddings_flat = embeddings_sequence.view(
            batch_size * seq_len, num_layers, embed_dim
        )
        
        # Temporal encoding
        lstm_out, _ = self.temporal_encoder(embeddings_flat)
        
        # Project to feature space
        features = self.feature_projection(lstm_out)
        features = F.relu(features)
        
        # Apply cross-layer attention
        attended_features = []
        for t in range(num_layers):
            # Create query from current layer
            query = features[:, t:t+1, :]
            
            # Create keys and values from all previous layers (including current)
            keys = features[:, :t+1, :]
            values = features[:, :t+1, :]
            
            # Apply time decay weights
            attn_mask = self.time_decay_weights[t, :t+1].unsqueeze(0)
            
            # Compute attention
            attn_out, _ = self.cross_layer_attention(
                query, keys, values,
                attn_mask=attn_mask.repeat(batch_size * seq_len, 1, 1)
            )
            
            attended_features.append(attn_out)
            
        # Stack attended features
        attended_features = torch.cat(attended_features, dim=1)
        
        # Reshape back
        attended_features = attended_features.view(
            batch_size, num_layers, seq_len, self.dict_size
        )
        
        if return_all_features:
            return attended_features
        else:
            # Return only the last layer's features
            return attended_features[:, -1, :, :]
            
    def decode(
        self,
        features: torch.Tensor,
        target_layer: Optional[int] = None
    ) -> torch.Tensor:
        """
        Decode features back to embedding space.
        
        Args:
            features: [batch_size, seq_len, dict_size] or 
                     [batch_size, num_layers, seq_len, dict_size]
            target_layer: If specified, decode for specific layer
            
        Returns:
            Reconstructed embeddings
        """
        if features.dim() == 4:
            # Full layer sequence provided
            batch_size, num_layers, seq_len, _ = features.shape
            features_flat = features.view(batch_size * seq_len, num_layers, -1)
        else:
            # Single layer features
            batch_size, seq_len, _ = features.shape
            features_flat = features.view(batch_size * seq_len, 1, -1)
            
        # Temporal decoding
        decoder_out, _ = self.temporal_decoder(features_flat)
        
        # Project to output space
        reconstructed = self.output_projection(decoder_out)
        
        if features.dim() == 4:
            reconstructed = reconstructed.view(
                batch_size, num_layers, seq_len, self.input_dim
            )
            
            # Add layer-specific biases
            reconstructed = reconstructed + self.layer_biases.unsqueeze(0).unsqueeze(2)
            
            if target_layer is not None:
                return reconstructed[:, target_layer, :, :]
            else:
                return reconstructed
        else:
            reconstructed = reconstructed.view(batch_size, seq_len, self.input_dim)
            if target_layer is not None:
                reconstructed = reconstructed + self.layer_biases[target_layer]
            return reconstructed
            
    def forward(
        self,
        embeddings_sequence: torch.Tensor,
        return_features: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the Temporal-SAE.
        
        Args:
            embeddings_sequence: [batch_size, num_layers, seq_len, embed_dim]
            return_features: If True, also return encoded features
            
        Returns:
            Tuple of (reconstructed_embeddings, features)
        """
        # Encode
        features = self.encode(embeddings_sequence, return_all_features=True)
        
        # Decode
        reconstructed = self.decode(features)
        
        if return_features:
            return reconstructed, features
        else:
            return reconstructed, None
            
    def compute_circuit_importance(
        self,
        features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute importance scores for feature circuits across layers.
        
        Args:
            features: [batch_size, num_layers, seq_len, dict_size]
            
        Returns:
            Circuit importance matrix: [dict_size, dict_size, num_layers-1]
        """
        batch_size, num_layers, seq_len, dict_size = features.shape
        
        # Compute feature transitions between consecutive layers
        importance_scores = torch.zeros(dict_size, dict_size, num_layers - 1)
        
        for layer in range(num_layers - 1):
            curr_features = features[:, layer, :, :].mean(dim=0)  # [seq_len, dict_size]
            next_features = features[:, layer + 1, :, :].mean(dim=0)
            
            # Compute correlation between features across layers
            curr_norm = F.normalize(curr_features, p=2, dim=0)
            next_norm = F.normalize(next_features, p=2, dim=0)
            
            # Feature correlation matrix
            correlation = torch.matmul(curr_norm.T, next_norm)
            importance_scores[:, :, layer] = correlation
            
        return importance_scores


class TemporalFeatureTracker(nn.Module):
    """
    Tracks feature evolution across layers with causal analysis.
    """
    
    def __init__(self, feature_dim: int, num_layers: int):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        
        # Feature evolution GRU
        self.evolution_gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=feature_dim,
            num_layers=1,
            batch_first=True
        )
        
        # Causal strength predictor
        self.causal_predictor = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid()
        )
        
    def track_feature_evolution(
        self,
        features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Track how features evolve across layers.
        
        Args:
            features: [batch_size, num_layers, seq_len, feature_dim]
            
        Returns:
            Dictionary containing evolution metrics
        """
        batch_size, num_layers, seq_len, feature_dim = features.shape
        
        # Process each sequence position
        evolution_metrics = {
            'feature_stability': [],
            'feature_emergence': [],
            'feature_decay': []
        }
        
        for pos in range(seq_len):
            pos_features = features[:, :, pos, :]  # [batch_size, num_layers, feature_dim]
            
            # Track evolution through GRU
            evolution_out, _ = self.evolution_gru(pos_features)
            
            # Compute stability (how much features persist)
            stability = F.cosine_similarity(
                pos_features[:, :-1, :],
                pos_features[:, 1:, :],
                dim=-1
            ).mean(dim=0)  # [num_layers-1]
            
            # Compute emergence (new features appearing)
            emergence = (pos_features[:, 1:, :] > 0.1).float().sum(dim=-1) - \
                       (pos_features[:, :-1, :] > 0.1).float().sum(dim=-1)
            emergence = emergence.mean(dim=0) / feature_dim
            
            # Compute decay (features disappearing)
            decay = (pos_features[:, :-1, :] > 0.1).float().sum(dim=-1) - \
                   (pos_features[:, 1:, :] > 0.1).float().sum(dim=-1)
            decay = decay.mean(dim=0) / feature_dim
            
            evolution_metrics['feature_stability'].append(stability)
            evolution_metrics['feature_emergence'].append(emergence)
            evolution_metrics['feature_decay'].append(decay)
            
        # Stack metrics
        for key in evolution_metrics:
            evolution_metrics[key] = torch.stack(evolution_metrics[key], dim=-1)
            
        return evolution_metrics
        
    def compute_causal_strength(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute causal strength between feature sets.
        
        Args:
            source_features: [batch_size, seq_len, feature_dim]
            target_features: [batch_size, seq_len, feature_dim]
            
        Returns:
            Causal strength matrix: [feature_dim, feature_dim]
        """
        # Concatenate features
        combined = torch.cat([
            source_features.unsqueeze(2).expand(-1, -1, self.feature_dim, -1),
            target_features.unsqueeze(3).expand(-1, -1, -1, self.feature_dim)
        ], dim=-1)
        
        # Predict causal strength
        causal_strength = self.causal_predictor(combined)
        
        # Average over batch and sequence
        causal_matrix = causal_strength.mean(dim=[0, 1])
        
        return causal_matrix