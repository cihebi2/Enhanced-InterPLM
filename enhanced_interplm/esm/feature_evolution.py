# 特征演化追踪 
# enhanced_interplm/esm/feature_evolution.py

"""
Feature evolution tracking across transformer layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mutual_info_score
import pandas as pd


@dataclass
class FeatureEvolutionMetrics:
    """Container for feature evolution metrics."""
    layer_similarities: np.ndarray  # Similarity between consecutive layers
    feature_magnitudes: np.ndarray  # Feature magnitude per layer
    feature_diversity: np.ndarray  # Feature diversity per layer
    residual_strengths: np.ndarray  # Strength of residual connections
    emergence_patterns: Dict[int, List[int]]  # Features emerging at each layer
    decay_patterns: Dict[int, List[int]]  # Features decaying at each layer
    stable_features: List[int]  # Features stable across layers
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'layer_similarities': self.layer_similarities.tolist(),
            'feature_magnitudes': self.feature_magnitudes.tolist(),
            'feature_diversity': self.feature_diversity.tolist(),
            'residual_strengths': self.residual_strengths.tolist(),
            'emergence_patterns': {k: v for k, v in self.emergence_patterns.items()},
            'decay_patterns': {k: v for k, v in self.decay_patterns.items()},
            'stable_features': self.stable_features
        }


class FeatureEvolutionAnalyzer:
    """
    Analyzes how features evolve across transformer layers.
    """
    
    def __init__(
        self,
        num_layers: int,
        feature_dim: int,
        stability_threshold: float = 0.8,
        emergence_threshold: float = 0.3,
        device: torch.device = torch.device('cpu')
    ):
        self.num_layers = num_layers
        self.feature_dim = feature_dim
        self.stability_threshold = stability_threshold
        self.emergence_threshold = emergence_threshold
        self.device = device
        
        # Storage for accumulated statistics
        self.reset_statistics()
        
    def reset_statistics(self):
        """Reset accumulated statistics."""
        self.feature_activations = []
        self.layer_representations = []
        self.sample_count = 0
        
    def track_batch(self, embeddings: torch.Tensor):
        """
        Track feature evolution for a batch of embeddings.
        
        Args:
            embeddings: [batch_size, num_layers, seq_len, feature_dim]
        """
        batch_size = embeddings.shape[0]
        
        # Store batch statistics
        self.feature_activations.append(embeddings.detach().cpu())
        self.sample_count += batch_size
        
        # Keep only recent batches to manage memory
        if len(self.feature_activations) > 100:
            self.feature_activations.pop(0)
            
    def analyze_evolution(self) -> FeatureEvolutionMetrics:
        """
        Analyze accumulated feature evolution patterns.
        
        Returns:
            FeatureEvolutionMetrics object
        """
        if not self.feature_activations:
            raise ValueError("No data tracked yet. Call track_batch first.")
            
        # Concatenate all tracked batches
        all_features = torch.cat(self.feature_activations, dim=0)
        
        # Compute various metrics
        layer_similarities = self._compute_layer_similarities(all_features)
        feature_magnitudes = self._compute_feature_magnitudes(all_features)
        feature_diversity = self._compute_feature_diversity(all_features)
        residual_strengths = self._compute_residual_strengths(all_features)
        
        # Identify feature patterns
        emergence_patterns = self._identify_emergence_patterns(all_features)
        decay_patterns = self._identify_decay_patterns(all_features)
        stable_features = self._identify_stable_features(all_features)
        
        return FeatureEvolutionMetrics(
            layer_similarities=layer_similarities,
            feature_magnitudes=feature_magnitudes,
            feature_diversity=feature_diversity,
            residual_strengths=residual_strengths,
            emergence_patterns=emergence_patterns,
            decay_patterns=decay_patterns,
            stable_features=stable_features
        )
        
    def _compute_layer_similarities(self, features: torch.Tensor) -> np.ndarray:
        """Compute similarity between consecutive layers."""
        similarities = []
        
        for layer in range(self.num_layers - 1):
            curr_layer = features[:, layer].reshape(-1, self.feature_dim)
            next_layer = features[:, layer + 1].reshape(-1, self.feature_dim)
            
            # Cosine similarity
            cos_sim = F.cosine_similarity(curr_layer, next_layer, dim=-1)
            similarities.append(cos_sim.mean().item())
            
        return np.array(similarities)
        
    def _compute_feature_magnitudes(self, features: torch.Tensor) -> np.ndarray:
        """Compute average feature magnitude per layer."""
        magnitudes = features.norm(dim=-1).mean(dim=(0, 2))
        return magnitudes.numpy()
        
    def _compute_feature_diversity(self, features: torch.Tensor) -> np.ndarray:
        """Compute feature diversity (std) per layer."""
        diversity = features.std(dim=-1).mean(dim=(0, 2))
        return diversity.numpy()
        
    def _compute_residual_strengths(self, features: torch.Tensor) -> np.ndarray:
        """Compute strength of residual connections."""
        if self.num_layers < 2:
            return np.array([])
            
        first_layer = features[:, 0]
        residual_strengths = []
        
        for layer in range(1, self.num_layers):
            curr_layer = features[:, layer]
            
            # Flatten and compute correlation
            first_flat = first_layer.reshape(-1, self.feature_dim)
            curr_flat = curr_layer.reshape(-1, self.feature_dim)
            
            # Cosine similarity as proxy for residual strength
            residual = F.cosine_similarity(first_flat, curr_flat, dim=-1)
            residual_strengths.append(residual.mean().item())
            
        return np.array(residual_strengths)
        
    def _identify_emergence_patterns(self, features: torch.Tensor) -> Dict[int, List[int]]:
        """Identify features that emerge at each layer."""
        emergence_patterns = {}
        
        # Compute feature importance per layer
        feature_importance = features.abs().mean(dim=(0, 2))  # [num_layers, feature_dim]
        
        for layer in range(1, self.num_layers):
            prev_importance = feature_importance[layer - 1]
            curr_importance = feature_importance[layer]
            
            # Features with significant increase in importance
            importance_increase = curr_importance - prev_importance
            emerging = torch.where(importance_increase > self.emergence_threshold)[0]
            
            emergence_patterns[layer] = emerging.tolist()
            
        return emergence_patterns
        
    def _identify_decay_patterns(self, features: torch.Tensor) -> Dict[int, List[int]]:
        """Identify features that decay at each layer."""
        decay_patterns = {}
        
        feature_importance = features.abs().mean(dim=(0, 2))
        
        for layer in range(1, self.num_layers):
            prev_importance = feature_importance[layer - 1]
            curr_importance = feature_importance[layer]
            
            # Features with significant decrease in importance
            importance_decrease = prev_importance - curr_importance
            decaying = torch.where(importance_decrease > self.emergence_threshold)[0]
            
            decay_patterns[layer] = decaying.tolist()
            
        return decay_patterns
        
    def _identify_stable_features(self, features: torch.Tensor) -> List[int]:
        """Identify features that remain stable across layers."""
        stable_features = []
        
        # Compute feature correlations across layers
        for feat_idx in range(self.feature_dim):
            is_stable = True
            
            for layer in range(self.num_layers - 1):
                curr_feat = features[:, layer, :, feat_idx].flatten()
                next_feat = features[:, layer + 1, :, feat_idx].flatten()
                
                # Compute correlation
                if curr_feat.std() > 0 and next_feat.std() > 0:
                    corr = torch.corrcoef(torch.stack([curr_feat, next_feat]))[0, 1]
                    
                    if corr < self.stability_threshold:
                        is_stable = False
                        break
                        
            if is_stable:
                stable_features.append(feat_idx)
                
        return stable_features
        
    def visualize_evolution(
        self,
        metrics: FeatureEvolutionMetrics,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Create comprehensive visualization of feature evolution."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Layer similarities
        ax = axes[0, 0]
        ax.plot(range(1, len(metrics.layer_similarities) + 1), metrics.layer_similarities, 'o-')
        ax.set_xlabel('Layer Transition')
        ax.set_ylabel('Cosine Similarity')
        ax.set_title('Layer-to-Layer Similarity')
        ax.grid(True, alpha=0.3)
        
        # Feature magnitudes
        ax = axes[0, 1]
        ax.plot(metrics.feature_magnitudes, 'o-', color='orange')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Average Magnitude')
        ax.set_title('Feature Magnitude Evolution')
        ax.grid(True, alpha=0.3)
        
        # Feature diversity
        ax = axes[0, 2]
        ax.plot(metrics.feature_diversity, 'o-', color='green')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Feature Diversity (Std)')
        ax.set_title('Feature Diversity Evolution')
        ax.grid(True, alpha=0.3)
        
        # Residual strengths
        ax = axes[1, 0]
        if len(metrics.residual_strengths) > 0:
            ax.plot(range(1, len(metrics.residual_strengths) + 1), 
                   metrics.residual_strengths, 'o-', color='red')
            ax.set_xlabel('Layer')
            ax.set_ylabel('Residual Strength')
            ax.set_title('Residual Connection Strength')
            ax.grid(True, alpha=0.3)
        
        # Emergence/Decay patterns
        ax = axes[1, 1]
        layers = list(range(1, self.num_layers))
        emergence_counts = [len(metrics.emergence_patterns.get(l, [])) for l in layers]
        decay_counts = [len(metrics.decay_patterns.get(l, [])) for l in layers]
        
        x = np.arange(len(layers))
        width = 0.35
        ax.bar(x - width/2, emergence_counts, width, label='Emerging', color='blue', alpha=0.7)
        ax.bar(x + width/2, decay_counts, width, label='Decaying', color='red', alpha=0.7)
        ax.set_xlabel('Layer')
        ax.set_ylabel('Number of Features')
        ax.set_title('Feature Emergence and Decay')
        ax.set_xticks(x)
        ax.set_xticklabels(layers)
        ax.legend()
        
        # Stable features histogram
        ax = axes[1, 2]
        ax.bar(['Stable', 'Dynamic'], 
               [len(metrics.stable_features), self.feature_dim - len(metrics.stable_features)],
               color=['green', 'orange'], alpha=0.7)
        ax.set_ylabel('Number of Features')
        ax.set_title(f'Feature Stability ({len(metrics.stable_features)} stable features)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig


class LayerTransitionAnalyzer:
    """
    Analyzes transitions between specific layers.
    """
    
    def __init__(self, source_layer: int, target_layer: int):
        self.source_layer = source_layer
        self.target_layer = target_layer
        
    def analyze_transition(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Analyze transition from source to target layer.
        
        Args:
            source_features: [batch_size, seq_len, feature_dim]
            target_features: [batch_size, seq_len, feature_dim]
            
        Returns:
            Dictionary of transition metrics
        """
        batch_size, seq_len, feature_dim = source_features.shape
        
        # Flatten for analysis
        source_flat = source_features.reshape(-1, feature_dim)
        target_flat = target_features.reshape(-1, feature_dim)
        
        # Compute transition matrix (feature correlation)
        transition_matrix = self._compute_transition_matrix(source_flat, target_flat)
        
        # Identify transformation patterns
        transformation_patterns = self._identify_transformations(transition_matrix)
        
        # Compute information flow
        information_flow = self._compute_information_flow(source_flat, target_flat)
        
        return {
            'transition_matrix': transition_matrix,
            'transformation_patterns': transformation_patterns,
            'information_flow': information_flow,
            'source_activation_rate': (source_flat.abs() > 0.1).float().mean(dim=0),
            'target_activation_rate': (target_flat.abs() > 0.1).float().mean(dim=0)
        }
        
    def _compute_transition_matrix(
        self,
        source: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute feature transition matrix."""
        # Normalize features
        source_norm = F.normalize(source, p=2, dim=0)
        target_norm = F.normalize(target, p=2, dim=0)
        
        # Compute correlation matrix
        transition_matrix = torch.matmul(source_norm.T, target_norm) / source.shape[0]
        
        return transition_matrix
        
    def _identify_transformations(
        self,
        transition_matrix: torch.Tensor
    ) -> Dict[str, List[Tuple[int, int]]]:
        """Identify different types of feature transformations."""
        patterns = {
            'preserved': [],  # Features that map strongly to themselves
            'split': [],      # Features that split into multiple
            'merged': [],     # Multiple features merging into one
            'transformed': [] # Features that map to different features
        }
        
        # Identify preserved features (high diagonal values)
        diagonal = transition_matrix.diagonal()
        preserved_mask = diagonal > 0.8
        patterns['preserved'] = torch.where(preserved_mask)[0].tolist()
        
        # Identify split features (one source to many targets)
        for i in range(transition_matrix.shape[0]):
            row = transition_matrix[i]
            strong_connections = torch.where(row > 0.5)[0]
            
            if len(strong_connections) > 1:
                patterns['split'].append((i, strong_connections.tolist()))
                
        # Identify merged features (many sources to one target)
        for j in range(transition_matrix.shape[1]):
            col = transition_matrix[:, j]
            strong_connections = torch.where(col > 0.5)[0]
            
            if len(strong_connections) > 1:
                patterns['merged'].append((strong_connections.tolist(), j))
                
        return patterns
        
    def _compute_information_flow(
        self,
        source: torch.Tensor,
        target: torch.Tensor
    ) -> float:
        """Compute information flow from source to target."""
        # Use mutual information as a measure
        # Discretize features for MI computation
        source_discrete = (source > source.median(dim=0)[0]).float()
        target_discrete = (target > target.median(dim=0)[0]).float()
        
        # Compute average MI across features
        mi_scores = []
        
        for i in range(min(source.shape[1], 100)):  # Limit for efficiency
            for j in range(min(target.shape[1], 100)):
                mi = mutual_info_score(
                    source_discrete[:, i].cpu().numpy(),
                    target_discrete[:, j].cpu().numpy()
                )
                mi_scores.append(mi)
                
        return np.mean(mi_scores) if mi_scores else 0.0


class FeatureLineageTracker:
    """
    Tracks the lineage of individual features across layers.
    """
    
    def __init__(self, num_layers: int, feature_dim: int):
        self.num_layers = num_layers
        self.feature_dim = feature_dim
        self.lineages = {}
        
    def trace_feature_lineage(
        self,
        features: torch.Tensor,
        feature_idx: int
    ) -> Dict[str, np.ndarray]:
        """
        Trace the lineage of a specific feature across layers.
        
        Args:
            features: [batch_size, num_layers, seq_len, feature_dim]
            feature_idx: Index of feature to trace
            
        Returns:
            Dictionary containing lineage information
        """
        lineage = {
            'activation_pattern': [],
            'magnitude_evolution': [],
            'correlation_with_first': [],
            'layer_contributions': []
        }
        
        # Extract feature across all layers
        feature_evolution = features[:, :, :, feature_idx]
        
        # First layer as reference
        first_layer = feature_evolution[:, 0].flatten()
        
        for layer in range(self.num_layers):
            layer_feature = feature_evolution[:, layer]
            
            # Activation pattern
            activation_rate = (layer_feature.abs() > 0.1).float().mean().item()
            lineage['activation_pattern'].append(activation_rate)
            
            # Magnitude evolution
            avg_magnitude = layer_feature.abs().mean().item()
            lineage['magnitude_evolution'].append(avg_magnitude)
            
            # Correlation with first layer
            if layer > 0:
                curr_flat = layer_feature.flatten()
                if first_layer.std() > 0 and curr_flat.std() > 0:
                    corr = torch.corrcoef(torch.stack([first_layer, curr_flat]))[0, 1].item()
                else:
                    corr = 0.0
                lineage['correlation_with_first'].append(corr)
                
        # Compute layer contributions (how much each layer changes the feature)
        for layer in range(1, self.num_layers):
            prev = feature_evolution[:, layer - 1].flatten()
            curr = feature_evolution[:, layer].flatten()
            
            # Compute change magnitude
            change = (curr - prev).abs().mean().item()
            lineage['layer_contributions'].append(change)
            
        return {k: np.array(v) for k, v in lineage.items()}