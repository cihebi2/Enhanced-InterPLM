# 功能域映射 
# enhanced_interplm/hierarchical_mapping/cross_level_integrator.py

"""
Advanced cross-level integration for hierarchical protein analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import networkx as nx
from scipy.spatial.distance import cdist


class HierarchicalGraphIntegrator(nn.Module):
    """
    Integrates features across hierarchical levels using graph neural networks.
    """
    
    def __init__(
        self,
        aa_dim: int = 64,
        ss_dim: int = 128,
        domain_dim: int = 256,
        protein_dim: int = 512,
        hidden_dim: int = 256,
        num_heads: int = 8
    ):
        super().__init__()
        
        self.level_dims = {
            'aa': aa_dim,
            'ss': ss_dim,
            'domain': domain_dim,
            'protein': protein_dim
        }
        
        # Level-specific encoders with different architectures
        self.level_encoders = nn.ModuleDict({
            'aa': self._build_aa_encoder(aa_dim, hidden_dim),
            'ss': self._build_ss_encoder(ss_dim, hidden_dim),
            'domain': self._build_domain_encoder(domain_dim, hidden_dim),
            'protein': self._build_protein_encoder(protein_dim, hidden_dim)
        })
        
        # Cross-level message passing
        self.bottom_up_gnn = HierarchicalGNN(
            hidden_dim, num_heads, direction='bottom_up'
        )
        
        self.top_down_gnn = HierarchicalGNN(
            hidden_dim, num_heads, direction='top_down'
        )
        
        # Bidirectional integration
        self.bidirectional_integrator = BidirectionalIntegrator(
            hidden_dim, num_heads
        )
        
        # Multi-scale fusion
        self.scale_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
    def _build_aa_encoder(self, input_dim: int, hidden_dim: int) -> nn.Module:
        """Build amino acid level encoder."""
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
    def _build_ss_encoder(self, input_dim: int, hidden_dim: int) -> nn.Module:
        """Build secondary structure encoder."""
        return nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.LayerNorm([hidden_dim])
        )
        
    def _build_domain_encoder(self, input_dim: int, hidden_dim: int) -> nn.Module:
        """Build domain level encoder."""
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
    def _build_protein_encoder(self, input_dim: int, hidden_dim: int) -> nn.Module:
        """Build protein level encoder."""
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
    def forward(
        self,
        features: Dict[str, torch.Tensor],
        hierarchy: Dict[str, List[Tuple[int, int]]]
    ) -> Dict[str, torch.Tensor]:
        """
        Integrate features across hierarchical levels.
        
        Args:
            features: Dictionary of features at each level
            hierarchy: Hierarchical relationships between levels
            
        Returns:
            Integrated features at each level
        """
        # Encode features at each level
        encoded = {}
        
        for level, feat in features.items():
            if level in self.level_encoders:
                encoder = self.level_encoders[level]
                
                if level == 'ss':
                    # Handle conv1d input
                    feat = feat.transpose(1, 2)
                    encoded[level] = encoder(feat).transpose(1, 2)
                elif feat.dim() == 3:
                    # Flatten batch for linear layers
                    batch_size, seq_len, feat_dim = feat.shape
                    feat_flat = feat.view(-1, feat_dim)
                    encoded_flat = encoder(feat_flat)
                    encoded[level] = encoded_flat.view(batch_size, seq_len, -1)
                else:
                    encoded[level] = encoder(feat)
                    
        # Bottom-up message passing
        bottom_up_features = self.bottom_up_gnn(encoded, hierarchy)
        
        # Top-down message passing
        top_down_features = self.top_down_gnn(bottom_up_features, hierarchy)
        
        # Bidirectional integration
        integrated = self.bidirectional_integrator(
            bottom_up_features, top_down_features, hierarchy
        )
        
        # Multi-scale fusion for each level
        fused = {}
        
        for level in features.keys():
            if level in integrated:
                # Concatenate all scale information
                level_features = []
                
                # Add features from all levels
                for scale in ['aa', 'ss', 'domain', 'protein']:
                    if scale in integrated:
                        scale_feat = integrated[scale]
                        
                        # Adjust dimensions to match
                        if scale != level:
                            scale_feat = self._adjust_scale(
                                scale_feat, integrated[level], hierarchy, scale, level
                            )
                            
                        level_features.append(scale_feat)
                        
                # Fuse multi-scale features
                if len(level_features) == 4:
                    concat_features = torch.cat(level_features, dim=-1)
                    fused[level] = self.scale_fusion(concat_features)
                else:
                    fused[level] = integrated[level]
                    
        return fused
        
    def _adjust_scale(
        self,
        source_feat: torch.Tensor,
        target_feat: torch.Tensor,
        hierarchy: Dict[str, List[Tuple[int, int]]],
        source_level: str,
        target_level: str
    ) -> torch.Tensor:
        """Adjust feature dimensions between different scales."""
        if source_feat.shape == target_feat.shape:
            return source_feat
            
        # Simple interpolation for now
        if source_feat.dim() == 3 and target_feat.dim() == 3:
            # Interpolate sequence dimension
            source_feat = F.interpolate(
                source_feat.transpose(1, 2),
                size=target_feat.shape[1],
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
            
        elif source_feat.dim() == 2 and target_feat.dim() == 3:
            # Expand protein-level to sequence
            source_feat = source_feat.unsqueeze(1).expand(
                -1, target_feat.shape[1], -1
            )
            
        return source_feat


class HierarchicalGNN(nn.Module):
    """
    Graph neural network for hierarchical message passing.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        direction: str = 'bottom_up'
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.direction = direction
        
        # Message functions for each level transition
        self.message_functions = nn.ModuleDict({
            'aa_to_ss': self._build_message_function(),
            'ss_to_domain': self._build_message_function(),
            'domain_to_protein': self._build_message_function(),
            'protein_to_domain': self._build_message_function(),
            'domain_to_ss': self._build_message_function(),
            'ss_to_aa': self._build_message_function()
        })
        
        # Update functions
        self.update_functions = nn.ModuleDict({
            level: self._build_update_function()
            for level in ['aa', 'ss', 'domain', 'protein']
        })
        
    def _build_message_function(self) -> nn.Module:
        """Build message passing function."""
        return nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
    def _build_update_function(self) -> nn.Module:
        """Build node update function."""
        return nn.GRUCell(self.hidden_dim, self.hidden_dim)
        
    def forward(
        self,
        features: Dict[str, torch.Tensor],
        hierarchy: Dict[str, List[Tuple[int, int]]]
    ) -> Dict[str, torch.Tensor]:
        """
        Perform hierarchical message passing.
        """
        updated_features = features.copy()
        
        if self.direction == 'bottom_up':
            # AA -> SS -> Domain -> Protein
            transitions = [
                ('aa', 'ss', 'aa_to_ss'),
                ('ss', 'domain', 'ss_to_domain'),
                ('domain', 'protein', 'domain_to_protein')
            ]
        else:
            # Protein -> Domain -> SS -> AA
            transitions = [
                ('protein', 'domain', 'protein_to_domain'),
                ('domain', 'ss', 'domain_to_ss'),
                ('ss', 'aa', 'ss_to_aa')
            ]
            
        for source, target, message_key in transitions:
            if source in features and target in features:
                messages = self._compute_messages(
                    updated_features[source],
                    updated_features[target],
                    hierarchy.get(f'{source}_to_{target}', []),
                    self.message_functions[message_key]
                )
                
                # Update target features
                updated_features[target] = self._update_nodes(
                    updated_features[target],
                    messages,
                    self.update_functions[target]
                )
                
        return updated_features
        
    def _compute_messages(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        mappings: List[Tuple[int, int]],
        message_fn: nn.Module
    ) -> torch.Tensor:
        """Compute messages from source to target nodes."""
        # Initialize messages
        messages = torch.zeros_like(target_features)
        
        if not mappings:
            # If no explicit mapping, use averaging
            if source_features.dim() == target_features.dim():
                messages = source_features
            elif source_features.dim() > target_features.dim():
                messages = source_features.mean(dim=1, keepdim=True)
            else:
                messages = source_features.unsqueeze(1).expand_as(target_features)
        else:
            # Use provided mappings
            for source_idx, target_idx in mappings:
                if source_features.dim() == 3 and target_features.dim() == 3:
                    combined = torch.cat([
                        source_features[:, source_idx],
                        target_features[:, target_idx]
                    ], dim=-1)
                    
                    message = message_fn(combined)
                    messages[:, target_idx] += message
                    
        return messages
        
    def _update_nodes(
        self,
        features: torch.Tensor,
        messages: torch.Tensor,
        update_fn: nn.Module
    ) -> torch.Tensor:
        """Update node features with messages."""
        if features.dim() == 3:
            batch_size, seq_len, hidden_dim = features.shape
            features_flat = features.view(-1, hidden_dim)
            messages_flat = messages.view(-1, hidden_dim)
            
            updated_flat = update_fn(messages_flat, features_flat)
            updated = updated_flat.view(batch_size, seq_len, hidden_dim)
        else:
            updated = update_fn(messages, features)
            
        return updated


class BidirectionalIntegrator(nn.Module):
    """
    Integrates bottom-up and top-down information flows.
    """
    
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Attention for combining bottom-up and top-down
        self.integration_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Gating mechanism
        self.bottom_up_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        self.top_down_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        # Final integration
        self.final_integration = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
    def forward(
        self,
        bottom_up: Dict[str, torch.Tensor],
        top_down: Dict[str, torch.Tensor],
        hierarchy: Dict[str, List[Tuple[int, int]]]
    ) -> Dict[str, torch.Tensor]:
        """
        Integrate bidirectional information flows.
        """
        integrated = {}
        
        for level in bottom_up.keys():
            if level in top_down:
                bu_features = bottom_up[level]
                td_features = top_down[level]
                
                # Ensure same dimensions
                if bu_features.shape != td_features.shape:
                    continue
                    
                # Compute gates
                combined = torch.cat([bu_features, td_features], dim=-1)
                
                bu_gate = self.bottom_up_gate(combined)
                td_gate = self.top_down_gate(combined)
                
                # Gated combination
                gated_bu = bu_features * bu_gate
                gated_td = td_features * td_gate
                
                # Attention-based integration
                if bu_features.dim() == 3:
                    integrated_attn, _ = self.integration_attention(
                        gated_bu, gated_td, gated_td
                    )
                else:
                    integrated_attn = (gated_bu + gated_td) / 2
                    
                # Final integration
                final_combined = torch.cat([integrated_attn, gated_bu + gated_td], dim=-1)
                integrated[level] = self.final_integration(final_combined)
                
        return integrated


class MultiScaleAttentionIntegrator(nn.Module):
    """
    Attention-based integration across multiple scales.
    """
    
    def __init__(
        self,
        scale_dims: Dict[str, int],
        output_dim: int,
        num_heads: int = 8
    ):
        super().__init__()
        
        self.scale_dims = scale_dims
        self.output_dim = output_dim
        
        # Scale-specific projections
        self.scale_projections = nn.ModuleDict({
            scale: nn.Linear(dim, output_dim)
            for scale, dim in scale_dims.items()
        })
        
        # Cross-scale attention
        self.cross_scale_attention = nn.ModuleDict({
            scale: nn.MultiheadAttention(
                embed_dim=output_dim,
                num_heads=num_heads,
                batch_first=True
            )
            for scale in scale_dims.keys()
        })
        
        # Scale importance weights
        self.scale_importance = nn.Parameter(
            torch.ones(len(scale_dims)) / len(scale_dims)
        )
        
        # Output network
        self.output_network = nn.Sequential(
            nn.Linear(output_dim * len(scale_dims), output_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(
        self,
        multi_scale_features: Dict[str, torch.Tensor],
        target_scale: str = 'aa'
    ) -> torch.Tensor:
        """
        Integrate features from multiple scales.
        
        Args:
            multi_scale_features: Features at different scales
            target_scale: Target scale for output
            
        Returns:
            Integrated features at target scale
        """
        # Project all scales to common dimension
        projected = {}
        
        for scale, features in multi_scale_features.items():
            if scale in self.scale_projections:
                projected[scale] = self.scale_projections[scale](features)
                
        # Get target scale shape
        target_shape = multi_scale_features[target_scale].shape[:-1]
        
        # Attend from target scale to all scales
        attended_features = []
        
        target_proj = projected[target_scale]
        
        for i, (scale, features) in enumerate(projected.items()):
            # Adjust dimensions if necessary
            if features.shape[:-1] != target_shape:
                features = self._adjust_dimensions(features, target_shape)
                
            # Cross-scale attention
            if target_proj.dim() == 3:
                attended, _ = self.cross_scale_attention[target_scale](
                    target_proj, features, features
                )
            else:
                attended = features
                
            # Apply scale importance
            attended = attended * self.scale_importance[i]
            attended_features.append(attended)
            
        # Concatenate and process
        concatenated = torch.cat(attended_features, dim=-1)
        integrated = self.output_network(concatenated)
        
        return integrated
        
    def _adjust_dimensions(
        self,
        features: torch.Tensor,
        target_shape: torch.Size
    ) -> torch.Tensor:
        """Adjust feature dimensions to match target."""
        if len(features.shape) == len(target_shape) + 1:
            # Same number of dimensions
            if features.dim() == 3 and len(target_shape) == 2:
                # Interpolate sequence dimension
                features = F.interpolate(
                    features.transpose(1, 2),
                    size=target_shape[1],
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)
        elif len(features.shape) < len(target_shape) + 1:
            # Add dimensions
            while len(features.shape) < len(target_shape) + 1:
                features = features.unsqueeze(1)
            # Expand to target shape
            expand_shape = list(target_shape) + [features.shape[-1]]
            features = features.expand(*expand_shape)
            
        return features


class HierarchicalPooling(nn.Module):
    """
    Pooling operations for hierarchical feature aggregation.
    """
    
    def __init__(self, method: str = 'attention'):
        super().__init__()
        
        self.method = method
        
        if method == 'attention':
            self.attention_pooling = AttentionPooling()
        elif method == 'graph':
            self.graph_pooling = GraphPooling()
            
    def forward(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        graph: Optional[nx.Graph] = None
    ) -> torch.Tensor:
        """
        Pool features hierarchically.
        """
        if self.method == 'attention':
            return self.attention_pooling(features, mask)
        elif self.method == 'graph' and graph is not None:
            return self.graph_pooling(features, graph)
        else:
            # Default mean pooling
            if mask is not None:
                features = features * mask.unsqueeze(-1)
                pooled = features.sum(dim=1) / mask.sum(dim=1, keepdim=True)
            else:
                pooled = features.mean(dim=1)
            return pooled


class AttentionPooling(nn.Module):
    """Attention-based pooling."""
    
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply attention pooling."""
        # Compute attention weights
        attn_weights = self.attention(features).squeeze(-1)
        
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, -float('inf'))
            
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # Weighted sum
        pooled = torch.matmul(attn_weights.unsqueeze(1), features).squeeze(1)
        
        return pooled


class GraphPooling(nn.Module):
    """Graph-based hierarchical pooling."""
    
    def __init__(self):
        super().__init__()
        
    def forward(
        self,
        features: torch.Tensor,
        graph: nx.Graph
    ) -> torch.Tensor:
        """Apply graph-based pooling."""
        # Simplified graph pooling
        # In practice, would use more sophisticated graph pooling methods
        
        pooled_features = []
        
        # Pool based on graph communities
        communities = nx.community.greedy_modularity_communities(graph)
        
        for community in communities:
            community_indices = list(community)
            if community_indices:
                community_features = features[:, community_indices].mean(dim=1)
                pooled_features.append(community_features)
                
        if pooled_features:
            return torch.stack(pooled_features, dim=1)
        else:
            return features.mean(dim=1, keepdim=True)