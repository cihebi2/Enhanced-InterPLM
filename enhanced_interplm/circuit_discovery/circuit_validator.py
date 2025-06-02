# 回路验证 
"""Advanced circuit validation with biological relevance scoring."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import networkx as nx
from sklearn.metrics import mutual_info_score
from scipy.stats import pearsonr, spearmanr


@dataclass
class CircuitMetrics:
    """Comprehensive metrics for circuit evaluation."""
    coherence: float
    stability: float
    task_relevance: float
    biological_plausibility: float
    information_flow: float
    modularity: float
    
    @property
    def overall_score(self) -> float:
        """Compute weighted overall score."""
        weights = {
            'coherence': 0.2,
            'stability': 0.2,
            'task_relevance': 0.2,
            'biological_plausibility': 0.2,
            'information_flow': 0.1,
            'modularity': 0.1
        }
        
        score = (
            self.coherence * weights['coherence'] +
            self.stability * weights['stability'] +
            self.task_relevance * weights['task_relevance'] +
            self.biological_plausibility * weights['biological_plausibility'] +
            self.information_flow * weights['information_flow'] +
            self.modularity * weights['modularity']
        )
        
        return score


class AdvancedCircuitValidator:
    """
    Advanced validation of discovered circuits with multiple criteria.
    """
    
    def __init__(
        self,
        validation_threshold: float = 0.6,
        min_circuit_size: int = 3,
        max_circuit_size: int = 20,
        use_biological_priors: bool = True
    ):
        self.validation_threshold = validation_threshold
        self.min_circuit_size = min_circuit_size
        self.max_circuit_size = max_circuit_size
        self.use_biological_priors = use_biological_priors
        
        # Load biological priors if available
        if use_biological_priors:
            self._load_biological_priors()
            
    def validate_circuits(
        self,
        circuits: List[Dict],
        features: Dict[int, torch.Tensor],
        task_labels: Optional[torch.Tensor] = None,
        protein_annotations: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Comprehensive circuit validation.
        
        Args:
            circuits: List of discovered circuits
            features: Feature activations by layer
            task_labels: Optional task-specific labels
            protein_annotations: Optional protein annotations
            
        Returns:
            Validated circuits with detailed metrics
        """
        validated_circuits = []
        
        for circuit in circuits:
            # Skip circuits outside size bounds
            circuit_size = len(circuit.get('instances', [[]])[0])
            if circuit_size < self.min_circuit_size or circuit_size > self.max_circuit_size:
                continue
                
            # Compute comprehensive metrics
            metrics = self._compute_circuit_metrics(
                circuit, features, task_labels, protein_annotations
            )
            
            # Check if circuit passes validation
            if metrics.overall_score >= self.validation_threshold:
                circuit['metrics'] = metrics
                circuit['validation_score'] = metrics.overall_score
                validated_circuits.append(circuit)
                
        # Rank by validation score
        validated_circuits.sort(key=lambda x: x['validation_score'], reverse=True)
        
        return validated_circuits
        
    def _compute_circuit_metrics(
        self,
        circuit: Dict,
        features: Dict[int, torch.Tensor],
        task_labels: Optional[torch.Tensor],
        annotations: Optional[Dict]
    ) -> CircuitMetrics:
        """Compute all validation metrics for a circuit."""
        
        # Basic metrics
        coherence = self._compute_coherence(circuit, features)
        stability = self._compute_stability(circuit, features)
        
        # Task relevance
        if task_labels is not None:
            task_relevance = self._compute_task_relevance(circuit, features, task_labels)
        else:
            task_relevance = 0.5  # Neutral score if no labels
            
        # Biological plausibility
        if self.use_biological_priors and annotations:
            biological_plausibility = self._compute_biological_plausibility(
                circuit, features, annotations
            )
        else:
            biological_plausibility = 0.5
            
        # Information flow
        information_flow = self._compute_information_flow(circuit, features)
        
        # Modularity
        modularity = self._compute_modularity(circuit, features)
        
        return CircuitMetrics(
            coherence=coherence,
            stability=stability,
            task_relevance=task_relevance,
            biological_plausibility=biological_plausibility,
            information_flow=information_flow,
            modularity=modularity
        )
        
    def _compute_coherence(self, circuit: Dict, features: Dict[int, torch.Tensor]) -> float:
        """
        Compute circuit coherence - how well components activate together.
        """
        coherence_scores = []
        
        for instance in circuit.get('instances', []):
            if not instance:
                continue
                
            # Extract features for this circuit instance
            instance_features = []
            
            for node in instance:
                layer, feat_idx = self._parse_node_id(node)
                if layer in features:
                    feat_acts = features[layer][:, :, feat_idx].flatten()
                    instance_features.append(feat_acts)
                    
            if len(instance_features) < 2:
                continue
                
            # Compute pairwise correlations
            instance_tensor = torch.stack(instance_features)
            
            # Handle case where features have low variance
            valid_features = []
            for feat in instance_tensor:
                if feat.std() > 1e-6:
                    valid_features.append(feat)
                    
            if len(valid_features) >= 2:
                valid_tensor = torch.stack(valid_features)
                corr_matrix = torch.corrcoef(valid_tensor)
                
                # Average correlation (excluding diagonal)
                n = corr_matrix.shape[0]
                mask = ~torch.eye(n, dtype=bool)
                avg_corr = corr_matrix[mask].mean().item()
                
                coherence_scores.append(avg_corr)
                
        return np.mean(coherence_scores) if coherence_scores else 0.0
        
    def _compute_stability(self, circuit: Dict, features: Dict[int, torch.Tensor]) -> float:
        """
        Compute circuit stability across different samples.
        """
        stability_scores = []
        
        for instance in circuit.get('instances', []):
            if not instance:
                continue
                
            # Get activation patterns for each node
            patterns = []
            
            for node in instance:
                layer, feat_idx = self._parse_node_id(node)
                if layer in features:
                    # Binary activation pattern
                    pattern = (features[layer][:, :, feat_idx] > 0.1).float()
                    patterns.append(pattern.flatten())
                    
            if len(patterns) < 2:
                continue
                
            # Compute consistency of co-activation
            patterns_tensor = torch.stack(patterns)
            
            # Variance of co-activation patterns
            co_activation = patterns_tensor.prod(dim=0)  # All must be active
            
            if co_activation.sum() > 0:
                # Stability is high if co-activation is consistent
                stability = co_activation.mean().item()
                stability_scores.append(stability)
                
        return np.mean(stability_scores) if stability_scores else 0.0
        
    def _compute_task_relevance(
        self,
        circuit: Dict,
        features: Dict[int, torch.Tensor],
        task_labels: torch.Tensor
    ) -> float:
        """
        Compute relevance of circuit to specific task.
        """
        relevance_scores = []
        
        for instance in circuit.get('instances', []):
            if not instance:
                continue
                
            # Aggregate circuit activation
            circuit_activations = []
            
            for node in instance:
                layer, feat_idx = self._parse_node_id(node)
                if layer in features:
                    activation = features[layer][:, :, feat_idx].mean(dim=1)
                    circuit_activations.append(activation)
                    
            if not circuit_activations:
                continue
                
            # Average activation across circuit
            circuit_activation = torch.stack(circuit_activations).mean(dim=0)
            
            # Compute correlation with task labels
            if task_labels.dim() > 1:
                # Multi-class: use mutual information
                mi_score = self._compute_mutual_information(
                    circuit_activation, task_labels
                )
                relevance_scores.append(mi_score)
            else:
                # Binary/regression: use correlation
                if circuit_activation.std() > 0 and task_labels.std() > 0:
                    corr = torch.corrcoef(
                        torch.stack([circuit_activation, task_labels.float()])
                    )[0, 1].abs().item()
                    relevance_scores.append(corr)
                    
        return np.mean(relevance_scores) if relevance_scores else 0.0
        
    def _compute_biological_plausibility(
        self,
        circuit: Dict,
        features: Dict[int, torch.Tensor],
        annotations: Dict
    ) -> float:
        """
        Assess biological plausibility of circuit.
        """
        plausibility_scores = []
        
        # Check if circuit components map to known biological features
        for instance in circuit.get('instances', []):
            if not instance:
                continue
                
            biological_associations = 0
            total_nodes = len(instance)
            
            for node in instance:
                layer, feat_idx = self._parse_node_id(node)
                
                # Check various biological associations
                if self._is_associated_with_function(feat_idx, annotations):
                    biological_associations += 1
                elif self._is_associated_with_structure(feat_idx, annotations):
                    biological_associations += 0.8
                elif self._is_associated_with_conservation(feat_idx, annotations):
                    biological_associations += 0.6
                    
            if total_nodes > 0:
                plausibility = biological_associations / total_nodes
                plausibility_scores.append(plausibility)
                
        return np.mean(plausibility_scores) if plausibility_scores else 0.5
        
    def _compute_information_flow(
        self,
        circuit: Dict,
        features: Dict[int, torch.Tensor]
    ) -> float:
        """
        Compute information flow through circuit.
        """
        flow_scores = []
        
        for instance in circuit.get('instances', []):
            if not instance:
                continue
                
            # Order nodes by layer
            ordered_nodes = sorted(instance, key=lambda x: int(x.split('_')[0][1:]))
            
            if len(ordered_nodes) < 2:
                continue
                
            # Compute information transfer between consecutive nodes
            transfers = []
            
            for i in range(len(ordered_nodes) - 1):
                source_layer, source_feat = self._parse_node_id(ordered_nodes[i])
                target_layer, target_feat = self._parse_node_id(ordered_nodes[i + 1])
                
                if source_layer in features and target_layer in features:
                    source_acts = features[source_layer][:, :, source_feat].flatten()
                    target_acts = features[target_layer][:, :, target_feat].flatten()
                    
                    # Compute transfer entropy or mutual information
                    if source_acts.std() > 0 and target_acts.std() > 0:
                        mi = self._compute_mutual_information(source_acts, target_acts)
                        transfers.append(mi)
                        
            if transfers:
                # Average information transfer
                flow_scores.append(np.mean(transfers))
                
        return np.mean(flow_scores) if flow_scores else 0.0
        
    def _compute_modularity(self, circuit: Dict, features: Dict[int, torch.Tensor]) -> float:
        """
        Compute circuit modularity - internal cohesion vs external separation.
        """
        modularity_scores = []
        
        for instance in circuit.get('instances', []):
            if not instance:
                continue
                
            # Get features for circuit nodes
            circuit_features = []
            circuit_indices = []
            
            for node in instance:
                layer, feat_idx = self._parse_node_id(node)
                if layer in features:
                    circuit_features.append(features[layer][:, :, feat_idx])
                    circuit_indices.append((layer, feat_idx))
                    
            if len(circuit_features) < 2:
                continue
                
            # Compute internal cohesion
            internal_corrs = []
            for i in range(len(circuit_features)):
                for j in range(i + 1, len(circuit_features)):
                    feat_i = circuit_features[i].flatten()
                    feat_j = circuit_features[j].flatten()
                    
                    if feat_i.std() > 0 and feat_j.std() > 0:
                        corr = torch.corrcoef(torch.stack([feat_i, feat_j]))[0, 1].abs()
                        internal_corrs.append(corr.item())
                        
            internal_cohesion = np.mean(internal_corrs) if internal_corrs else 0
            
            # Compute external separation (simplified)
            # In practice, would compare to features outside circuit
            external_separation = 1.0 - internal_cohesion * 0.5
            
            modularity = internal_cohesion * external_separation
            modularity_scores.append(modularity)
            
        return np.mean(modularity_scores) if modularity_scores else 0.0
        
    def _parse_node_id(self, node_id: str) -> Tuple[int, int]:
        """Parse node ID to extract layer and feature indices."""
        parts = node_id.split('_')
        if len(parts) >= 2:
            layer = int(parts[0][1:])  # Remove 'L' prefix
            feature = int(parts[1][1:])  # Remove 'F' prefix
            return layer, feature
        else:
            raise ValueError(f"Invalid node ID format: {node_id}")
            
    def _compute_mutual_information(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        bins: int = 10
    ) -> float:
        """Compute mutual information between two variables."""
        x_np = x.cpu().numpy()
        y_np = y.cpu().numpy()
        
        # Discretize continuous variables
        x_discrete = np.digitize(x_np, bins=np.percentile(x_np, np.linspace(0, 100, bins)))
        
        if y.dim() > 1:
            # Multi-class
            y_discrete = y.argmax(dim=-1).cpu().numpy()
        else:
            y_discrete = np.digitize(y_np, bins=np.percentile(y_np, np.linspace(0, 100, bins)))
            
        return mutual_info_score(x_discrete, y_discrete)
        
    def _load_biological_priors(self):
        """Load biological priors for validation."""
        # In practice, would load from database
        self.known_motifs = {
            'helix_cap': {'size': 4, 'conservation': 0.8},
            'zinc_finger': {'size': 4, 'conservation': 0.9},
            'catalytic_triad': {'size': 3, 'conservation': 0.95},
            'binding_pocket': {'size': 5, 'conservation': 0.85}
        }
        
    def _is_associated_with_function(self, feat_idx: int, annotations: Dict) -> bool:
        """Check if feature is associated with known function."""
        if 'functional_features' in annotations:
            return feat_idx in annotations['functional_features']
        return False
        
    def _is_associated_with_structure(self, feat_idx: int, annotations: Dict) -> bool:
        """Check if feature is associated with structural element."""
        if 'structural_features' in annotations:
            return feat_idx in annotations['structural_features']
        return False
        
    def _is_associated_with_conservation(self, feat_idx: int, annotations: Dict) -> bool:
        """Check if feature is associated with conserved regions."""
        if 'conserved_features' in annotations:
            return feat_idx in annotations['conserved_features']
        return False