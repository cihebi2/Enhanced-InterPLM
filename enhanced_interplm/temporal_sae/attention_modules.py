# 注意力机制 
# enhanced_interplm/temporal_sae/attention_modules.py

"""
Attention mechanisms for temporal-aware sparse autoencoders.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict
import numpy as np


class TemporalCrossAttention(nn.Module):
    """
    Cross-attention mechanism for temporal feature interactions across layers.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_bias: bool = True,
        use_temporal_bias: bool = True
    ):
        super().__init__()
        
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        
        self.dropout = nn.Dropout(dropout)
        
        # Temporal bias for layer-aware attention
        if use_temporal_bias:
            self.temporal_bias = nn.Parameter(torch.zeros(1, num_heads, 1, 1))
            
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        temporal_mask: Optional[torch.Tensor] = None,
        layer_distances: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of temporal cross-attention.
        
        Args:
            query: Query tensor [batch_size, seq_len, embed_dim]
            key: Key tensor [batch_size, seq_len, embed_dim]
            value: Value tensor [batch_size, seq_len, embed_dim]
            temporal_mask: Temporal attention mask
            layer_distances: Distance between layers for temporal weighting
            
        Returns:
            Output tensor and attention weights
        """
        batch_size, seq_len, _ = query.shape
        
        # Project and reshape
        q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # [batch, heads, seq_len, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply temporal bias based on layer distances
        if layer_distances is not None and hasattr(self, 'temporal_bias'):
            # layer_distances shape: [seq_len, seq_len]
            temporal_weight = torch.exp(-layer_distances.unsqueeze(0).unsqueeze(0))
            scores = scores + self.temporal_bias * temporal_weight
            
        # Apply mask if provided
        if temporal_mask is not None:
            scores = scores.masked_fill(temporal_mask == 0, -1e9)
            
        # Compute attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        
        return attn_output, attn_weights


class HierarchicalAttention(nn.Module):
    """
    Hierarchical attention mechanism for multi-scale feature integration.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        num_levels: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        
        # Level-specific attention modules
        self.level_attentions = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(num_levels)
        ])
        
        # Cross-level attention
        self.cross_level_attention = TemporalCrossAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Level embeddings
        self.level_embeddings = nn.Parameter(
            torch.randn(num_levels, embed_dim) * 0.02
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * num_levels, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
    def forward(
        self,
        features: Dict[str, torch.Tensor],
        return_attention_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass of hierarchical attention.
        
        Args:
            features: Dictionary of features at different levels
                     e.g., {'micro': ..., 'meso': ..., 'macro': ...}
            return_attention_weights: Whether to return attention weights
            
        Returns:
            Integrated features and optionally attention weights
        """
        level_outputs = []
        attention_weights = {} if return_attention_weights else None
        
        # Process each level
        for i, (level_name, level_features) in enumerate(features.items()):
            # Add level embedding
            level_features = level_features + self.level_embeddings[i]
            
            # Self-attention within level
            attn_output, attn_weights = self.level_attentions[i](
                level_features, level_features, level_features,
                need_weights=True
            )
            
            level_outputs.append(attn_output)
            
            if return_attention_weights:
                attention_weights[f'{level_name}_self'] = attn_weights
                
        # Stack level outputs
        stacked_outputs = torch.stack(level_outputs, dim=1)  # [batch, levels, seq, embed]
        
        # Cross-level attention
        batch_size, num_levels, seq_len, embed_dim = stacked_outputs.shape
        
        # Reshape for cross-attention
        stacked_flat = stacked_outputs.view(batch_size, -1, embed_dim)
        
        # Use middle level as query
        middle_idx = num_levels // 2
        query = stacked_outputs[:, middle_idx]
        
        # Cross-level integration
        cross_output, cross_weights = self.cross_level_attention(
            query=query,
            key=stacked_flat,
            value=stacked_flat
        )
        
        if return_attention_weights:
            attention_weights['cross_level'] = cross_weights
            
        # Combine all level outputs
        all_features = torch.cat([*level_outputs, cross_output], dim=-1)
        
        # Final fusion
        integrated = self.fusion(all_features)
        
        return integrated, attention_weights


class SparseAttention(nn.Module):
    """
    Sparse attention mechanism for efficient feature selection.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        sparsity_ratio: float = 0.1,
        temperature: float = 1.0
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.sparsity_ratio = sparsity_ratio
        self.temperature = temperature
        
        # Attention components
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
        # Top-k selection
        self.top_k = None  # Will be set based on sequence length
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with sparse attention.
        
        Args:
            x: Input tensor [batch_size, seq_len, embed_dim]
            mask: Optional attention mask
            
        Returns:
            Output tensor and sparsity mask
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute queries, keys, values
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        # Reshape for multi-head attention
        q = self._reshape_multihead(q, batch_size, seq_len)
        k = self._reshape_multihead(k, batch_size, seq_len)
        v = self._reshape_multihead(v, batch_size, seq_len)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.embed_dim // self.num_heads)
        
        # Apply temperature scaling
        scores = scores / self.temperature
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        # Sparse selection
        top_k = int(self.sparsity_ratio * seq_len * seq_len)
        top_k = max(1, min(top_k, seq_len * seq_len))
        
        # Flatten scores for top-k selection
        flat_scores = scores.view(batch_size * self.num_heads, -1)
        
        # Get top-k indices
        _, top_indices = torch.topk(flat_scores, top_k, dim=-1)
        
        # Create sparse mask
        sparse_mask = torch.zeros_like(flat_scores)
        sparse_mask.scatter_(-1, top_indices, 1)
        sparse_mask = sparse_mask.view_as(scores)
        
        # Apply sparse mask
        sparse_scores = scores * sparse_mask - 1e9 * (1 - sparse_mask)
        
        # Compute attention weights
        attn_weights = F.softmax(sparse_scores, dim=-1)
        
        # Apply attention
        output = torch.matmul(attn_weights, v)
        
        # Reshape back
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.embed_dim)
        
        return output, sparse_mask
        
    def _reshape_multihead(self, x: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
        """Reshape tensor for multi-head attention."""
        x = x.view(batch_size, seq_len, self.num_heads, self.embed_dim // self.num_heads)
        return x.transpose(1, 2)


class CausalTemporalAttention(nn.Module):
    """
    Causal attention for temporal dependencies with decay.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        max_layers: int = 12,
        decay_factor: float = 0.9,
        use_learned_decay: bool = True
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.max_layers = max_layers
        self.decay_factor = decay_factor
        
        # Standard attention components
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Learned decay parameters
        if use_learned_decay:
            self.decay_params = nn.Parameter(
                torch.ones(num_heads, max_layers, max_layers) * decay_factor
            )
        else:
            # Fixed exponential decay
            decay_matrix = torch.zeros(max_layers, max_layers)
            for i in range(max_layers):
                for j in range(i + 1):
                    decay_matrix[i, j] = decay_factor ** (i - j)
            self.register_buffer('decay_matrix', decay_matrix)
            
    def forward(
        self,
        x: torch.Tensor,
        layer_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with causal temporal attention.
        
        Args:
            x: Input tensor [batch_size, num_layers, seq_len, embed_dim]
            layer_indices: Optional layer indices for attention
            
        Returns:
            Output tensor with same shape as input
        """
        batch_size, num_layers, seq_len, _ = x.shape
        
        # Reshape for processing all positions
        x_flat = x.view(batch_size * num_layers, seq_len, self.embed_dim)
        
        # Compute QKV
        qkv = self.qkv(x_flat).chunk(3, dim=-1)
        q, k, v = [t.view(batch_size, num_layers, seq_len, self.num_heads, -1).transpose(2, 3)
                   for t in qkv]
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.embed_dim // self.num_heads)
        
        # Apply causal mask
        causal_mask = torch.triu(torch.ones(num_layers, num_layers), diagonal=1).bool()
        causal_mask = causal_mask.to(scores.device)
        
        # Expand mask for batch and heads
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        scores = scores.masked_fill(causal_mask, -1e9)
        
        # Apply temporal decay
        if hasattr(self, 'decay_params'):
            decay = self.decay_params[:, :num_layers, :num_layers]
            decay = decay.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        else:
            decay = self.decay_matrix[:num_layers, :num_layers]
            decay = decay.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            
        scores = scores * decay
        
        # Compute attention weights
        attn_weights = F.softmax(scores, dim=2)
        
        # Apply attention
        output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        output = output.transpose(2, 3).contiguous()
        output = output.view(batch_size * num_layers, seq_len, self.embed_dim)
        output = self.out_proj(output)
        output = output.view(batch_size, num_layers, seq_len, self.embed_dim)
        
        return output


class FeatureRoutingAttention(nn.Module):
    """
    Attention mechanism for routing features through different pathways.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_experts: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.num_heads = num_heads
        
        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 2, embed_dim)
            )
            for _ in range(num_experts)
        ])
        
        # Routing attention
        self.router = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Expert embeddings
        self.expert_embeddings = nn.Parameter(
            torch.randn(num_experts, embed_dim) * 0.02
        )
        
        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(embed_dim, num_experts),
            nn.Softmax(dim=-1)
        )
        
    def forward(
        self,
        x: torch.Tensor,
        return_routing_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with feature routing.
        
        Args:
            x: Input tensor [batch_size, seq_len, embed_dim]
            return_routing_weights: Whether to return routing weights
            
        Returns:
            Output tensor and optionally routing weights
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute routing weights
        routing_weights = self.gate(x)  # [batch, seq, num_experts]
        
        # Prepare expert queries
        expert_queries = self.expert_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Attend to input features
        attended_features, _ = self.router(expert_queries, x, x)
        
        # Apply experts
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            expert_input = attended_features[:, i]  # [batch, embed_dim]
            expert_output = expert(expert_input)
            expert_outputs.append(expert_output)
            
        # Stack expert outputs
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [batch, experts, embed_dim]
        
        # Apply routing weights
        expert_outputs = expert_outputs.unsqueeze(1).expand(-1, seq_len, -1, -1)
        routing_weights = routing_weights.unsqueeze(-1)
        
        # Weighted sum of expert outputs
        output = (expert_outputs * routing_weights).sum(dim=2)
        
        if return_routing_weights:
            return output, routing_weights.squeeze(-1)
        else:
            return output, None