# 特征交互图构建 # enhanced_interplm/circuit_discovery/graph_builder.py

import torch
import torch.nn as nn
import networkx as nx
import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy import stats
import pandas as pd


class DynamicGraphBuilder:
    """
    Builds dynamic interaction graphs between features across layers.
    """
    
    def __init__(
        self,
        mutual_info_threshold: float = 0.1,
        causal_threshold: float = 0.3,
        use_gpu: bool = True
    ):
        self.mutual_info_threshold = mutual_info_threshold
        self.causal_threshold = causal_threshold
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        
    def build_feature_interaction_graph(
        self,
        features: Dict[int, torch.Tensor],
        feature_metadata: Optional[Dict] = None
    ) -> nx.DiGraph:
        """
        Build a directed graph representing feature interactions.
        
        Args:
            features: Dictionary mapping layer indices to feature tensors
            feature_metadata: Optional metadata for features
            
        Returns:
            NetworkX directed graph with feature interactions
        """
        graph = nx.DiGraph()
        
        # Add nodes for each feature in each layer
        for layer_idx, layer_features in features.items():
            n_features = layer_features.shape[-1]
            for feat_idx in range(n_features):
                node_id = f"L{layer_idx}_F{feat_idx}"
                graph.add_node(
                    node_id,
                    layer=layer_idx,
                    feature=feat_idx,
                    activation_mean=float(layer_features[:, :, feat_idx].mean()),
                    activation_std=float(layer_features[:, :, feat_idx].std())
                )
                
        # Compute interactions between features
        layers = sorted(features.keys())
        for i, layer_i in enumerate(layers):
            for layer_j in layers[i:]:  # Include same-layer and cross-layer
                interactions = self._compute_feature_interactions(
                    features[layer_i],
                    features[layer_j],
                    same_layer=(layer_i == layer_j)
                )
                
                # Add edges based on interaction strength
                for (feat_i, feat_j), strength in interactions.items():
                    if strength['causal_strength'] > self.causal_threshold:
                        source = f"L{layer_i}_F{feat_i}"
                        target = f"L{layer_j}_F{feat_j}"
                        
                        if source != target:  # Avoid self-loops
                            graph.add_edge(
                                source, target,
                                weight=strength['causal_strength'],
                                mutual_info=strength['mutual_info'],
                                correlation=strength['correlation']
                            )
                            
        return graph
        
    def _compute_feature_interactions(
        self,
        features_1: torch.Tensor,
        features_2: torch.Tensor,
        same_layer: bool = False
    ) -> Dict[Tuple[int, int], Dict[str, float]]:
        """
        Compute pairwise interactions between features.
        
        Args:
            features_1: [batch, seq_len, n_features_1]
            features_2: [batch, seq_len, n_features_2]
            same_layer: Whether features are from the same layer
            
        Returns:
            Dictionary mapping feature pairs to interaction metrics
        """
        interactions = {}
        
        batch_size, seq_len, n_feat_1 = features_1.shape
        _, _, n_feat_2 = features_2.shape
        
        # Flatten batch and sequence dimensions
        feat_1_flat = features_1.view(-1, n_feat_1)
        feat_2_flat = features_2.view(-1, n_feat_2)
        
        # Compute interactions for each feature pair
        for i in range(n_feat_1):
            for j in range(n_feat_2):
                if same_layer and i >= j:  # Skip redundant pairs in same layer
                    continue
                    
                # Extract feature activations
                f1 = feat_1_flat[:, i].cpu().numpy()
                f2 = feat_2_flat[:, j].cpu().numpy()
                
                # Compute interaction metrics
                metrics = {
                    'mutual_info': self._compute_mutual_info(f1, f2),
                    'correlation': float(np.corrcoef(f1, f2)[0, 1]),
                    'causal_strength': self._compute_causal_strength(f1, f2)
                }
                
                interactions[(i, j)] = metrics
                
        return interactions
        
    def _compute_mutual_info(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute mutual information between two features."""
        # Discretize continuous values
        x_discrete = np.digitize(x, bins=np.percentile(x, np.linspace(0, 100, 10)))
        y_discrete = np.digitize(y, bins=np.percentile(y, np.linspace(0, 100, 10)))
        
        # Compute joint and marginal probabilities
        joint_hist = np.histogram2d(x_discrete, y_discrete, bins=10)[0]
        joint_prob = joint_hist / joint_hist.sum()
        
        x_prob = joint_prob.sum(axis=1)
        y_prob = joint_prob.sum(axis=0)
        
        # Compute mutual information
        mi = 0
        for i in range(len(x_prob)):
            for j in range(len(y_prob)):
                if joint_prob[i, j] > 0:
                    mi += joint_prob[i, j] * np.log(
                        joint_prob[i, j] / (x_prob[i] * y_prob[j] + 1e-10)
                    )
                    
        return mi
        
    def _compute_causal_strength(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute causal strength using Granger causality test.
        Simplified version for demonstration.
        """
        # Simple lagged correlation as proxy for causality
        if len(x) < 2:
            return 0.0
            
        # Compute correlation between x[t-1] and y[t]
        lagged_corr = np.corrcoef(x[:-1], y[1:])[0, 1]
        
        # Normalize to [0, 1]
        causal_strength = (lagged_corr + 1) / 2
        
        return float(causal_strength)


class CircuitMotifDetector:
    """
    Detects recurring circuit motifs in feature interaction graphs.
    """
    
    def __init__(self, min_motif_size: int = 3, max_motif_size: int = 8):
        self.min_motif_size = min_motif_size
        self.max_motif_size = max_motif_size
        
    def find_circuit_motifs(
        self,
        graph: nx.DiGraph,
        min_frequency: int = 2
    ) -> List[Dict]:
        """
        Find recurring circuit motifs in the feature interaction graph.
        
        Args:
            graph: Feature interaction graph
            min_frequency: Minimum frequency for a motif to be considered
            
        Returns:
            List of detected motifs with their properties
        """
        motifs = []
        
        # Find all simple paths of different lengths
        for size in range(self.min_motif_size, self.max_motif_size + 1):
            size_motifs = self._find_motifs_of_size(graph, size)
            
            # Filter by frequency
            frequent_motifs = [m for m in size_motifs if m['frequency'] >= min_frequency]
            motifs.extend(frequent_motifs)
            
        # Rank motifs by importance
        motifs = self._rank_motifs(motifs)
        
        return motifs
        
    def _find_motifs_of_size(self, graph: nx.DiGraph, size: int) -> List[Dict]:
        """Find all motifs of a specific size."""
        motifs = {}
        nodes = list(graph.nodes())
        
        # Find all subgraphs of given size
        from itertools import combinations
        for node_subset in combinations(nodes, size):
            subgraph = graph.subgraph(node_subset)
            
            if subgraph.number_of_edges() >= size - 1:  # Connected subgraph
                # Compute canonical form (isomorphism class)
                canonical = self._get_canonical_form(subgraph)
                
                if canonical not in motifs:
                    motifs[canonical] = {
                        'structure': canonical,
                        'instances': [],
                        'frequency': 0,
                        'avg_weight': 0
                    }
                    
                # Record this instance
                motifs[canonical]['instances'].append(node_subset)
                motifs[canonical]['frequency'] += 1
                
                # Compute average edge weight
                edge_weights = [graph[u][v]['weight'] for u, v in subgraph.edges()]
                if edge_weights:
                    motifs[canonical]['avg_weight'] = np.mean(edge_weights)
                    
        return list(motifs.values())
        
    def _get_canonical_form(self, subgraph: nx.DiGraph) -> str:
        """Get canonical representation of a graph structure."""
        # Simple adjacency matrix representation
        nodes = sorted(subgraph.nodes())
        n = len(nodes)
        adj_matrix = np.zeros((n, n))
        
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        
        for u, v in subgraph.edges():
            adj_matrix[node_to_idx[u], node_to_idx[v]] = 1
            
        # Convert to string representation
        return str(adj_matrix.flatten().tolist())
        
    def _rank_motifs(self, motifs: List[Dict]) -> List[Dict]:
        """Rank motifs by importance."""
        for motif in motifs:
            # Compute importance score
            motif['importance'] = (
                motif['frequency'] * 0.4 +
                motif['avg_weight'] * 0.6
            )
            
        # Sort by importance
        motifs.sort(key=lambda x: x['importance'], reverse=True)
        
        return motifs


class CircuitValidator:
    """
    Validates discovered circuits through various tests.
    """
    
    def __init__(self, validation_threshold: float = 0.7):
        self.validation_threshold = validation_threshold
        
    def validate_circuits(
        self,
        circuits: List[Dict],
        features: Dict[int, torch.Tensor],
        task_labels: Optional[torch.Tensor] = None
    ) -> List[Dict]:
        """
        Validate discovered circuits.
        
        Args:
            circuits: List of discovered circuits
            features: Feature activations by layer
            task_labels: Optional task-specific labels
            
        Returns:
            Validated circuits with scores
        """
        validated_circuits = []
        
        for circuit in circuits:
            validation_scores = {
                'coherence': self._compute_coherence_score(circuit, features),
                'stability': self._compute_stability_score(circuit, features),
                'task_relevance': 0.0
            }
            
            if task_labels is not None:
                validation_scores['task_relevance'] = self._compute_task_relevance(
                    circuit, features, task_labels
                )
                
            # Compute overall validation score
            overall_score = np.mean(list(validation_scores.values()))
            
            if overall_score >= self.validation_threshold:
                circuit['validation_scores'] = validation_scores
                circuit['overall_score'] = overall_score
                validated_circuits.append(circuit)
                
        return validated_circuits
        
    def _compute_coherence_score(
        self,
        circuit: Dict,
        features: Dict[int, torch.Tensor]
    ) -> float:
        """Compute how coherently circuit components activate together."""
        coherence_scores = []
        
        for instance in circuit['instances']:
            # Extract features for this circuit instance
            instance_features = []
            for node in instance:
                layer, feat_idx = self._parse_node_id(node)
                feat_acts = features[layer][:, :, feat_idx]
                instance_features.append(feat_acts.flatten())
                
            # Compute pairwise correlations
            instance_features = torch.stack(instance_features)
            corr_matrix = torch.corrcoef(instance_features)
            
            # Average correlation as coherence
            n = corr_matrix.shape[0]
            if n > 1:
                upper_triangle = corr_matrix[torch.triu_indices(n, n, offset=1)]
                coherence = upper_triangle.mean().item()
                coherence_scores.append(coherence)
                
        return np.mean(coherence_scores) if coherence_scores else 0.0
        
    def _compute_stability_score(
        self,
        circuit: Dict,
        features: Dict[int, torch.Tensor]
    ) -> float:
        """Compute stability of circuit activation patterns."""
        stability_scores = []
        
        for instance in circuit['instances']:
            # Extract activation patterns
            patterns = []
            for node in instance:
                layer, feat_idx = self._parse_node_id(node)
                pattern = features[layer][:, :, feat_idx] > 0.1  # Binary activation
                patterns.append(pattern.float())
                
            # Compute consistency across different samples
            patterns = torch.stack(patterns)
            pattern_std = patterns.std(dim=0).mean()
            stability = 1.0 - pattern_std.item()
            stability_scores.append(stability)
            
        return np.mean(stability_scores) if stability_scores else 0.0
        
    def _compute_task_relevance(
        self,
        circuit: Dict,
        features: Dict[int, torch.Tensor],
        task_labels: torch.Tensor
    ) -> float:
        """Compute relevance of circuit to specific task."""
        relevance_scores = []
        
        for instance in circuit['instances']:
            # Aggregate circuit activation
            circuit_activation = []
            for node in instance:
                layer, feat_idx = self._parse_node_id(node)
                activation = features[layer][:, :, feat_idx].mean(dim=1)  # Average over sequence
                circuit_activation.append(activation)
                
            circuit_activation = torch.stack(circuit_activation).mean(dim=0)  # Average over features
            
            # Compute correlation with task labels
            if task_labels.dim() > 1:
                task_labels = task_labels.float().mean(dim=1)
                
            correlation = torch.corrcoef(
                torch.stack([circuit_activation, task_labels])
            )[0, 1].item()
            
            relevance_scores.append(abs(correlation))
            
        return np.mean(relevance_scores) if relevance_scores else 0.0
        
    def _parse_node_id(self, node_id: str) -> Tuple[int, int]:
        """Parse node ID to extract layer and feature indices."""
        parts = node_id.split('_')
        layer = int(parts[0][1:])  # Remove 'L' prefix
        feature = int(parts[1][1:])  # Remove 'F' prefix
        return layer, feature