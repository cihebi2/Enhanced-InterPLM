# 层间特征追踪 
# enhanced_interplm/temporal_sae/layer_tracking.py

"""
Layer-wise feature tracking for temporal SAE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
import networkx as nx


@dataclass
class LayerFeatureState:
    """State of features at a specific layer."""
    layer_idx: int
    active_features: Set[int]
    feature_magnitudes: Dict[int, float]
    feature_frequencies: Dict[int, float]
    total_activation: float
    sparsity: float
    
    def get_top_features(self, k: int = 10) -> List[Tuple[int, float]]:
        """Get top k features by magnitude."""
        sorted_features = sorted(
            self.feature_magnitudes.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_features[:k]


@dataclass
class FeatureTransition:
    """Transition of a feature between layers."""
    feature_idx: int
    source_layer: int
    target_layer: int
    magnitude_change: float
    correlation: float
    transformation_type: str  # 'preserved', 'amplified', 'diminished', 'emerged', 'vanished'
    
    
class LayerTracker:
    """
    Tracks feature evolution across transformer layers.
    """
    
    def __init__(
        self,
        num_layers: int,
        feature_dim: int,
        activation_threshold: float = 0.1,
        correlation_threshold: float = 0.7
    ):
        self.num_layers = num_layers
        self.feature_dim = feature_dim
        self.activation_threshold = activation_threshold
        self.correlation_threshold = correlation_threshold
        
        # Storage for layer states
        self.layer_states: List[LayerFeatureState] = []
        self.transitions: List[FeatureTransition] = []
        
        # Feature lineage tracking
        self.feature_lineages: Dict[int, List[int]] = defaultdict(list)
        
        # Statistics accumulation
        self.reset_statistics()
        
    def reset_statistics(self):
        """Reset accumulated statistics."""
        self.feature_activation_counts = np.zeros((self.num_layers, self.feature_dim))
        self.feature_magnitude_sums = np.zeros((self.num_layers, self.feature_dim))
        self.layer_transition_matrix = np.zeros((self.num_layers - 1, self.feature_dim, self.feature_dim))
        self.sample_count = 0
        
    def track_forward_pass(
        self,
        layer_features: List[torch.Tensor],
        return_detailed: bool = False
    ) -> Dict[str, Any]:
        """
        Track features through a forward pass.
        
        Args:
            layer_features: List of feature tensors for each layer [seq_len, feature_dim]
            return_detailed: Whether to return detailed tracking info
            
        Returns:
            Dictionary containing tracking results
        """
        assert len(layer_features) == self.num_layers
        
        # Clear previous tracking
        self.layer_states = []
        self.transitions = []
        
        # Track each layer
        for layer_idx, features in enumerate(layer_features):
            state = self._analyze_layer_state(features, layer_idx)
            self.layer_states.append(state)
            
            # Track transitions
            if layer_idx > 0:
                transitions = self._analyze_transitions(
                    layer_features[layer_idx - 1],
                    features,
                    layer_idx - 1,
                    layer_idx
                )
                self.transitions.extend(transitions)
                
        # Update statistics
        self._update_statistics(layer_features)
        
        # Analyze feature lineages
        lineages = self._trace_feature_lineages()
        
        results = {
            'layer_states': self.layer_states,
            'transitions': self.transitions,
            'lineages': lineages,
            'summary': self._compute_summary_statistics()
        }
        
        if return_detailed:
            results['detailed'] = self._compute_detailed_analysis()
            
        return results
        
    def _analyze_layer_state(
        self,
        features: torch.Tensor,
        layer_idx: int
    ) -> LayerFeatureState:
        """Analyze the state of features at a specific layer."""
        # Compute activation mask
        active_mask = features.abs() > self.activation_threshold
        active_features = set(torch.where(active_mask.any(dim=0))[0].tolist())
        
        # Compute feature statistics
        feature_magnitudes = {}
        feature_frequencies = {}
        
        for feat_idx in active_features:
            feat_values = features[:, feat_idx]
            feature_magnitudes[feat_idx] = feat_values.abs().mean().item()
            feature_frequencies[feat_idx] = active_mask[:, feat_idx].float().mean().item()
            
        # Overall statistics
        total_activation = features.abs().sum().item()
        sparsity = len(active_features) / self.feature_dim
        
        return LayerFeatureState(
            layer_idx=layer_idx,
            active_features=active_features,
            feature_magnitudes=feature_magnitudes,
            feature_frequencies=feature_frequencies,
            total_activation=total_activation,
            sparsity=sparsity
        )
        
    def _analyze_transitions(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        source_layer: int,
        target_layer: int
    ) -> List[FeatureTransition]:
        """Analyze transitions between consecutive layers."""
        transitions = []
        
        # Get active features in both layers
        source_active = (source_features.abs() > self.activation_threshold).any(dim=0)
        target_active = (target_features.abs() > self.activation_threshold).any(dim=0)
        
        # Analyze each feature
        for feat_idx in range(self.feature_dim):
            source_vals = source_features[:, feat_idx]
            target_vals = target_features[:, feat_idx]
            
            # Skip if feature is inactive in both layers
            if not source_active[feat_idx] and not target_active[feat_idx]:
                continue
                
            # Compute correlation if both are active
            if source_vals.std() > 0 and target_vals.std() > 0:
                correlation = torch.corrcoef(
                    torch.stack([source_vals, target_vals])
                )[0, 1].item()
            else:
                correlation = 0.0
                
            # Determine transformation type
            source_mag = source_vals.abs().mean().item()
            target_mag = target_vals.abs().mean().item()
            
            if not source_active[feat_idx] and target_active[feat_idx]:
                trans_type = 'emerged'
            elif source_active[feat_idx] and not target_active[feat_idx]:
                trans_type = 'vanished'
            elif correlation > self.correlation_threshold:
                if target_mag > source_mag * 1.2:
                    trans_type = 'amplified'
                elif target_mag < source_mag * 0.8:
                    trans_type = 'diminished'
                else:
                    trans_type = 'preserved'
            else:
                trans_type = 'transformed'
                
            transitions.append(FeatureTransition(
                feature_idx=feat_idx,
                source_layer=source_layer,
                target_layer=target_layer,
                magnitude_change=target_mag - source_mag,
                correlation=correlation,
                transformation_type=trans_type
            ))
            
        return transitions
        
    def _update_statistics(self, layer_features: List[torch.Tensor]):
        """Update accumulated statistics."""
        self.sample_count += 1
        
        for layer_idx, features in enumerate(layer_features):
            # Update activation counts
            active_mask = (features.abs() > self.activation_threshold)
            self.feature_activation_counts[layer_idx] += active_mask.sum(dim=0).cpu().numpy()
            
            # Update magnitude sums
            self.feature_magnitude_sums[layer_idx] += features.abs().sum(dim=0).cpu().numpy()
            
            # Update transition matrix
            if layer_idx > 0:
                prev_features = layer_features[layer_idx - 1]
                
                # Normalize features
                prev_norm = F.normalize(prev_features, p=2, dim=0)
                curr_norm = F.normalize(features, p=2, dim=0)
                
                # Compute correlation matrix
                corr_matrix = torch.matmul(prev_norm.T, curr_norm).cpu().numpy()
                self.layer_transition_matrix[layer_idx - 1] += corr_matrix
                
    def _trace_feature_lineages(self) -> Dict[int, List[Tuple[int, int]]]:
        """Trace the lineage of features across layers."""
        lineages = defaultdict(list)
        
        # Start from first layer active features
        for feat_idx in self.layer_states[0].active_features:
            lineage = [(0, feat_idx)]
            current_feat = feat_idx
            
            # Trace through layers
            for layer_idx in range(1, self.num_layers):
                # Find strongest connection in next layer
                transitions = [t for t in self.transitions 
                             if t.source_layer == layer_idx - 1 and 
                             t.feature_idx == current_feat and
                             t.correlation > self.correlation_threshold]
                
                if transitions:
                    # Follow strongest correlation
                    best_transition = max(transitions, key=lambda t: t.correlation)
                    
                    # Find corresponding feature in target layer
                    for next_feat in self.layer_states[layer_idx].active_features:
                        next_transitions = [t for t in self.transitions
                                          if t.target_layer == layer_idx and
                                          t.feature_idx == next_feat and
                                          t.source_layer == layer_idx - 1]
                        
                        if next_transitions and next_transitions[0].correlation > self.correlation_threshold:
                            lineage.append((layer_idx, next_feat))
                            current_feat = next_feat
                            break
                else:
                    break
                    
            if len(lineage) > 1:
                lineages[feat_idx] = lineage
                
        return dict(lineages)
        
    def _compute_summary_statistics(self) -> Dict[str, Any]:
        """Compute summary statistics from tracking."""
        summary = {
            'avg_sparsity_per_layer': [state.sparsity for state in self.layer_states],
            'total_activation_per_layer': [state.total_activation for state in self.layer_states],
            'num_active_features_per_layer': [len(state.active_features) for state in self.layer_states],
            'feature_emergence_rate': [],
            'feature_preservation_rate': []
        }
        
        # Compute emergence and preservation rates
        for layer_idx in range(1, self.num_layers):
            prev_active = self.layer_states[layer_idx - 1].active_features
            curr_active = self.layer_states[layer_idx].active_features
            
            emerged = len(curr_active - prev_active)
            preserved = len(curr_active & prev_active)
            
            summary['feature_emergence_rate'].append(emerged / max(len(curr_active), 1))
            summary['feature_preservation_rate'].append(preserved / max(len(prev_active), 1))
            
        return summary
        
    def _compute_detailed_analysis(self) -> Dict[str, Any]:
        """Compute detailed analysis of feature dynamics."""
        detailed = {
            'transition_type_counts': defaultdict(int),
            'layer_similarity_matrix': np.zeros((self.num_layers, self.num_layers)),
            'feature_importance_scores': {},
            'critical_transitions': []
        }
        
        # Count transition types
        for trans in self.transitions:
            detailed['transition_type_counts'][trans.transformation_type] += 1
            
        # Compute layer similarity matrix
        for i in range(self.num_layers):
            for j in range(i, self.num_layers):
                if i == j:
                    detailed['layer_similarity_matrix'][i, j] = 1.0
                else:
                    # Jaccard similarity of active features
                    active_i = self.layer_states[i].active_features
                    active_j = self.layer_states[j].active_features
                    
                    if active_i or active_j:
                        similarity = len(active_i & active_j) / len(active_i | active_j)
                        detailed['layer_similarity_matrix'][i, j] = similarity
                        detailed['layer_similarity_matrix'][j, i] = similarity
                        
        # Compute feature importance scores
        for layer_idx, state in enumerate(self.layer_states):
            for feat_idx, magnitude in state.feature_magnitudes.items():
                if feat_idx not in detailed['feature_importance_scores']:
                    detailed['feature_importance_scores'][feat_idx] = []
                    
                detailed['feature_importance_scores'][feat_idx].append({
                    'layer': layer_idx,
                    'magnitude': magnitude,
                    'frequency': state.feature_frequencies[feat_idx]
                })
                
        # Identify critical transitions (large magnitude changes)
        critical_transitions = sorted(
            self.transitions,
            key=lambda t: abs(t.magnitude_change),
            reverse=True
        )[:10]
        
        detailed['critical_transitions'] = critical_transitions
        
        return detailed
        
    def visualize_tracking(
        self,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Visualize feature tracking results."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Sparsity evolution
        ax = axes[0, 0]
        sparsity = [state.sparsity for state in self.layer_states]
        ax.plot(sparsity, 'o-', linewidth=2, markersize=8)
        ax.set_xlabel('Layer')
        ax.set_ylabel('Sparsity')
        ax.set_title('Feature Sparsity Across Layers')
        ax.grid(True, alpha=0.3)
        
        # 2. Feature activation heatmap
        ax = axes[0, 1]
        activation_matrix = np.zeros((self.num_layers, min(100, self.feature_dim)))
        
        for layer_idx, state in enumerate(self.layer_states):
            for feat_idx in list(state.active_features)[:100]:
                if feat_idx < activation_matrix.shape[1]:
                    activation_matrix[layer_idx, feat_idx] = state.feature_magnitudes.get(feat_idx, 0)
                    
        sns.heatmap(activation_matrix, ax=ax, cmap='YlOrRd', cbar=True)
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Layer')
        ax.set_title('Feature Activation Patterns')
        
        # 3. Transition type distribution
        ax = axes[0, 2]
        trans_counts = defaultdict(int)
        for trans in self.transitions:
            trans_counts[trans.transformation_type] += 1
            
        if trans_counts:
            ax.bar(trans_counts.keys(), trans_counts.values())
            ax.set_xlabel('Transition Type')
            ax.set_ylabel('Count')
            ax.set_title('Feature Transition Types')
            ax.tick_params(axis='x', rotation=45)
            
        # 4. Layer similarity matrix
        ax = axes[1, 0]
        similarity_matrix = np.zeros((self.num_layers, self.num_layers))
        
        for i in range(self.num_layers):
            for j in range(self.num_layers):
                active_i = self.layer_states[i].active_features
                active_j = self.layer_states[j].active_features
                
                if active_i or active_j:
                    similarity_matrix[i, j] = len(active_i & active_j) / max(len(active_i | active_j), 1)
                else:
                    similarity_matrix[i, j] = 0
                    
        sns.heatmap(similarity_matrix, ax=ax, cmap='Blues', vmin=0, vmax=1, 
                   annot=True, fmt='.2f', square=True)
        ax.set_xlabel('Layer')
        ax.set_ylabel('Layer')
        ax.set_title('Layer Feature Similarity')
        
        # 5. Feature preservation flow
        ax = axes[1, 1]
        preservation_rates = []
        emergence_rates = []
        
        for i in range(1, self.num_layers):
            prev_active = self.layer_states[i-1].active_features
            curr_active = self.layer_states[i].active_features
            
            if prev_active:
                preservation_rates.append(len(prev_active & curr_active) / len(prev_active))
            else:
                preservation_rates.append(0)
                
            if curr_active:
                emergence_rates.append(len(curr_active - prev_active) / len(curr_active))
            else:
                emergence_rates.append(0)
                
        x = range(1, self.num_layers)
        ax.plot(x, preservation_rates, 'o-', label='Preservation Rate', linewidth=2)
        ax.plot(x, emergence_rates, 's-', label='Emergence Rate', linewidth=2)
        ax.set_xlabel('Layer Transition')
        ax.set_ylabel('Rate')
        ax.set_title('Feature Dynamics')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. Top feature trajectories
        ax = axes[1, 2]
        
        # Select top features from first layer
        top_features = sorted(
            self.layer_states[0].feature_magnitudes.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        for feat_idx, _ in top_features:
            trajectory = []
            
            for layer_idx, state in enumerate(self.layer_states):
                if feat_idx in state.feature_magnitudes:
                    trajectory.append(state.feature_magnitudes[feat_idx])
                else:
                    trajectory.append(0)
                    
            ax.plot(trajectory, 'o-', label=f'Feature {feat_idx}', linewidth=2)
            
        ax.set_xlabel('Layer')
        ax.set_ylabel('Magnitude')
        ax.set_title('Top Feature Trajectories')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def get_feature_importance_ranking(
        self,
        criterion: str = 'persistence'
    ) -> List[Tuple[int, float]]:
        """
        Rank features by importance based on various criteria.
        
        Args:
            criterion: 'persistence', 'magnitude', 'emergence', or 'influence'
            
        Returns:
            List of (feature_idx, score) tuples
        """
        scores = {}
        
        if criterion == 'persistence':
            # Score based on how many layers feature remains active
            for feat_idx in range(self.feature_dim):
                active_layers = sum(1 for state in self.layer_states 
                                  if feat_idx in state.active_features)
                scores[feat_idx] = active_layers / self.num_layers
                
        elif criterion == 'magnitude':
            # Score based on average magnitude across layers
            for feat_idx in range(self.feature_dim):
                magnitudes = [state.feature_magnitudes.get(feat_idx, 0)
                            for state in self.layer_states]
                scores[feat_idx] = np.mean(magnitudes)
                
        elif criterion == 'emergence':
            # Score based on when feature first emerges
            for feat_idx in range(self.feature_dim):
                for layer_idx, state in enumerate(self.layer_states):
                    if feat_idx in state.active_features:
                        # Earlier emergence = higher score
                        scores[feat_idx] = 1.0 - (layer_idx / self.num_layers)
                        break
                else:
                    scores[feat_idx] = 0.0
                    
        elif criterion == 'influence':
            # Score based on downstream effects
            for trans in self.transitions:
                if trans.transformation_type in ['amplified', 'preserved']:
                    feat_idx = trans.feature_idx
                    scores[feat_idx] = scores.get(feat_idx, 0) + abs(trans.correlation)
                    
        else:
            raise ValueError(f"Unknown criterion: {criterion}")
            
        # Sort by score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        return ranked