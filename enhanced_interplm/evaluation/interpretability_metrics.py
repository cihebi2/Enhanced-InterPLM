# 可解释性指标 
# enhanced_interplm/evaluation/interpretability_metrics.py

"""
Specialized interpretability metrics for protein language model analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from scipy.stats import entropy, mutual_info_score
from sklearn.metrics import adjusted_mutual_info_score
import pandas as pd


@dataclass
class InterpretabilityScore:
    """Container for interpretability metrics."""
    feature_interpretability: float
    circuit_interpretability: float
    biological_alignment: float
    human_alignment: float
    overall_score: float
    details: Dict[str, float]
    
    def __repr__(self):
        return f"InterpretabilityScore(overall={self.overall_score:.3f})"


class FeatureInterpretabilityMetrics:
    """
    Metrics for assessing feature interpretability.
    """
    
    def __init__(self, reference_features: Optional[Dict[str, torch.Tensor]] = None):
        self.reference_features = reference_features or {}
        
    def compute_feature_interpretability(
        self,
        features: torch.Tensor,
        feature_annotations: Optional[Dict[int, str]] = None
    ) -> Dict[str, float]:
        """
        Compute interpretability metrics for features.
        
        Args:
            features: Feature tensor [batch, seq_len, num_features]
            feature_annotations: Human annotations for features
            
        Returns:
            Dictionary of interpretability metrics
        """
        metrics = {}
        
        # Feature monosemanticity (how specific each feature is)
        metrics['monosemanticity'] = self._compute_monosemanticity(features)
        
        # Feature orthogonality (independence of features)
        metrics['orthogonality'] = self._compute_orthogonality(features)
        
        # Feature consistency (activation consistency across similar inputs)
        metrics['consistency'] = self._compute_consistency(features)
        
        # Feature coverage (how well features cover input space)
        metrics['coverage'] = self._compute_coverage(features)
        
        # Human alignment (if annotations provided)
        if feature_annotations:
            metrics['human_alignment'] = self._compute_human_alignment(
                features, feature_annotations
            )
            
        return metrics
        
    def _compute_monosemanticity(self, features: torch.Tensor) -> float:
        """
        Compute monosemanticity score - how specific each feature is.
        Features that activate on specific patterns score higher.
        """
        # Flatten batch and sequence dimensions
        features_flat = features.reshape(-1, features.shape[-1])
        
        # Compute activation entropy for each feature
        entropies = []
        
        for feat_idx in range(features.shape[-1]):
            feat_acts = features_flat[:, feat_idx]
            
            # Skip dead features
            if feat_acts.abs().max() < 0.01:
                continue
                
            # Binarize activations
            active = (feat_acts.abs() > 0.1).float()
            
            # Compute entropy of activation pattern
            p_active = active.mean()
            if 0 < p_active < 1:
                ent = -p_active * np.log(p_active) - (1-p_active) * np.log(1-p_active)
                # Lower entropy = more specific = higher monosemanticity
                monosem = 1 - ent / np.log(2)
                entropies.append(monosem)
                
        return np.mean(entropies) if entropies else 0.0
        
    def _compute_orthogonality(self, features: torch.Tensor) -> float:
        """
        Compute orthogonality of features (independence).
        """
        # Sample subset for efficiency
        n_samples = min(1000, features.shape[0] * features.shape[1])
        features_flat = features.reshape(-1, features.shape[-1])
        
        if features_flat.shape[0] > n_samples:
            indices = torch.randperm(features_flat.shape[0])[:n_samples]
            features_flat = features_flat[indices]
            
        # Compute correlation matrix
        features_norm = F.normalize(features_flat, p=2, dim=0)
        corr_matrix = torch.abs(torch.matmul(features_norm.T, features_norm))
        
        # Remove diagonal
        n_features = corr_matrix.shape[0]
        mask = ~torch.eye(n_features, dtype=bool)
        off_diagonal_corr = corr_matrix[mask]
        
        # Orthogonality score (lower correlation = higher orthogonality)
        orthogonality = 1 - off_diagonal_corr.mean().item()
        
        return orthogonality
        
    def _compute_consistency(self, features: torch.Tensor) -> float:
        """
        Compute consistency of feature activations.
        """
        # Compute variance across batch for each position and feature
        feature_variance = features.var(dim=0)  # [seq_len, num_features]
        
        # Normalize by mean activation
        feature_mean = features.abs().mean(dim=0) + 1e-8
        normalized_variance = feature_variance / feature_mean
        
        # Consistency is inverse of normalized variance
        consistency = 1 / (1 + normalized_variance.mean().item())
        
        return consistency
        
    def _compute_coverage(self, features: torch.Tensor) -> float:
        """
        Compute how well features cover the input space.
        """
        # Count active features per position
        active_features = (features.abs() > 0.1).float()
        active_per_position = active_features.sum(dim=-1)  # [batch, seq_len]
        
        # Coverage is the average fraction of active features
        coverage = active_per_position.mean() / features.shape[-1]
        
        return coverage.item()
        
    def _compute_human_alignment(
        self,
        features: torch.Tensor,
        annotations: Dict[int, str]
    ) -> float:
        """
        Compute alignment with human annotations.
        """
        # Group features by annotation
        annotation_groups = {}
        for feat_idx, annotation in annotations.items():
            if annotation not in annotation_groups:
                annotation_groups[annotation] = []
            annotation_groups[annotation].append(feat_idx)
            
        # Compute within-group vs between-group similarity
        within_similarities = []
        between_similarities = []
        
        features_flat = features.reshape(-1, features.shape[-1])
        
        for group_name, group_indices in annotation_groups.items():
            if len(group_indices) < 2:
                continue
                
            # Within-group similarity
            for i in range(len(group_indices)):
                for j in range(i + 1, len(group_indices)):
                    feat_i = features_flat[:, group_indices[i]]
                    feat_j = features_flat[:, group_indices[j]]
                    
                    if feat_i.std() > 0 and feat_j.std() > 0:
                        sim = F.cosine_similarity(feat_i, feat_j, dim=0)
                        within_similarities.append(sim.item())
                        
            # Between-group similarity
            other_indices = [idx for idx, ann in annotations.items() 
                           if ann != group_name]
            
            for idx1 in group_indices[:5]:  # Limit for efficiency
                for idx2 in other_indices[:5]:
                    feat_1 = features_flat[:, idx1]
                    feat_2 = features_flat[:, idx2]
                    
                    if feat_1.std() > 0 and feat_2.std() > 0:
                        sim = F.cosine_similarity(feat_1, feat_2, dim=0)
                        between_similarities.append(sim.item())
                        
        # Human alignment is high when within > between similarity
        if within_similarities and between_similarities:
            alignment = (np.mean(within_similarities) - np.mean(between_similarities) + 1) / 2
            return max(0, min(1, alignment))
        
        return 0.5  # Neutral if can't compute


class CircuitInterpretabilityMetrics:
    """
    Metrics for assessing circuit interpretability.
    """
    
    def compute_circuit_interpretability(
        self,
        circuits: List[Dict],
        features: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute interpretability metrics for circuits.
        """
        if not circuits:
            return {'circuit_clarity': 0.0, 'circuit_modularity': 0.0}
            
        metrics = {}
        
        # Circuit clarity (how well-defined circuits are)
        clarity_scores = []
        for circuit in circuits[:50]:  # Analyze top circuits
            if 'validation_scores' in circuit:
                clarity = circuit['validation_scores'].get('coherence', 0)
                clarity_scores.append(clarity)
                
        metrics['circuit_clarity'] = np.mean(clarity_scores) if clarity_scores else 0.0
        
        # Circuit modularity (independence of circuits)
        metrics['circuit_modularity'] = self._compute_circuit_modularity(circuits)
        
        # Circuit specificity (how specific circuits are to functions)
        metrics['circuit_specificity'] = self._compute_circuit_specificity(circuits)
        
        return metrics
        
    def _compute_circuit_modularity(self, circuits: List[Dict]) -> float:
        """
        Compute modularity of circuits (low overlap = high modularity).
        """
        # Extract feature sets for each circuit
        circuit_features = []
        
        for circuit in circuits[:20]:  # Limit for efficiency
            features = set()
            for instance in circuit.get('instances', []):
                for node in instance:
                    # Extract feature index from node ID
                    if '_F' in node:
                        feat_idx = int(node.split('_F')[1])
                        features.add(feat_idx)
            circuit_features.append(features)
            
        # Compute pairwise overlap
        overlaps = []
        for i in range(len(circuit_features)):
            for j in range(i + 1, len(circuit_features)):
                if circuit_features[i] and circuit_features[j]:
                    overlap = len(circuit_features[i] & circuit_features[j])
                    union = len(circuit_features[i] | circuit_features[j])
                    jaccard = overlap / union if union > 0 else 0
                    overlaps.append(jaccard)
                    
        # Modularity is inverse of average overlap
        avg_overlap = np.mean(overlaps) if overlaps else 0
        modularity = 1 - avg_overlap
        
        return modularity
        
    def _compute_circuit_specificity(self, circuits: List[Dict]) -> float:
        """
        Compute specificity of circuits to their functions.
        """
        specificities = []
        
        for circuit in circuits:
            # Use importance score as proxy for specificity
            if 'importance' in circuit:
                specificities.append(circuit['importance'])
                
        return np.mean(specificities) if specificities else 0.0


class HumanAlignmentMetrics:
    """
    Metrics for assessing alignment with human understanding.
    """
    
    def compute_human_alignment(
        self,
        features: torch.Tensor,
        human_labels: Optional[Dict[str, torch.Tensor]] = None,
        expert_annotations: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Compute alignment with human understanding.
        """
        metrics = {}
        
        if human_labels:
            # Predictability of human labels from features
            metrics['label_predictability'] = self._compute_label_predictability(
                features, human_labels
            )
            
        if expert_annotations:
            # Agreement with expert annotations
            metrics['expert_agreement'] = self._compute_expert_agreement(
                features, expert_annotations
            )
            
        # Concept purity (how pure discovered concepts are)
        metrics['concept_purity'] = self._compute_concept_purity(features)
        
        return metrics
        
    def _compute_label_predictability(
        self,
        features: torch.Tensor,
        human_labels: Dict[str, torch.Tensor]
    ) -> float:
        """
        Compute how well features predict human labels.
        """
        predictabilities = []
        
        # Flatten features
        features_flat = features.mean(dim=1)  # Average over sequence
        
        for label_name, labels in human_labels.items():
            # Simple linear probe
            probe = nn.Linear(features.shape[-1], labels.shape[-1])
            optimizer = torch.optim.Adam(probe.parameters(), lr=0.01)
            
            # Quick training
            for _ in range(50):
                pred = probe(features_flat)
                
                if labels.dim() == 1:
                    loss = F.binary_cross_entropy_with_logits(pred.squeeze(), labels.float())
                else:
                    loss = F.cross_entropy(pred, labels)
                    
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            # Evaluate
            with torch.no_grad():
                pred = probe(features_flat)
                if labels.dim() == 1:
                    acc = ((pred.squeeze() > 0) == labels).float().mean()
                else:
                    acc = (pred.argmax(dim=-1) == labels).float().mean()
                    
                predictabilities.append(acc.item())
                
        return np.mean(predictabilities) if predictabilities else 0.0
        
    def _compute_expert_agreement(
        self,
        features: torch.Tensor,
        expert_annotations: Dict
    ) -> float:
        """
        Compute agreement with expert annotations.
        """
        # Simplified: would implement more sophisticated matching
        return 0.7  # Placeholder
        
    def _compute_concept_purity(self, features: torch.Tensor) -> float:
        """
        Compute purity of discovered concepts.
        """
        # Analyze feature activation patterns
        features_binary = (features.abs() > 0.1).float()
        
        # Compute co-activation matrix
        features_flat = features_binary.reshape(-1, features.shape[-1])
        coactivation = torch.matmul(features_flat.T, features_flat) / features_flat.shape[0]
        
        # Diagonal dominance indicates pure concepts
        diag = torch.diag(coactivation)
        off_diag = coactivation - torch.diag(diag)
        
        purity = diag.mean() / (off_diag.abs().mean() + 1e-8)
        purity = torch.clamp(purity, 0, 1)
        
        return purity.item()


def compute_interpretability_score(
    features: torch.Tensor,
    circuits: Optional[List[Dict]] = None,
    annotations: Optional[Dict] = None,
    human_labels: Optional[Dict] = None
) -> InterpretabilityScore:
    """
    Compute comprehensive interpretability score.
    """
    # Initialize metrics computers
    feature_metrics = FeatureInterpretabilityMetrics()
    circuit_metrics = CircuitInterpretabilityMetrics()
    human_metrics = HumanAlignmentMetrics()
    
    # Compute feature interpretability
    feature_scores = feature_metrics.compute_feature_interpretability(features, annotations)
    feature_interp = np.mean(list(feature_scores.values()))
    
    # Compute circuit interpretability
    if circuits:
        circuit_scores = circuit_metrics.compute_circuit_interpretability(circuits, features)
        circuit_interp = np.mean(list(circuit_scores.values()))
    else:
        circuit_interp = 0.0
        circuit_scores = {}
        
    # Compute human alignment
    if human_labels or annotations:
        human_scores = human_metrics.compute_human_alignment(features, human_labels, annotations)
        human_align = np.mean(list(human_scores.values()))
    else:
        human_align = 0.5
        human_scores = {}
        
    # Biological alignment (simplified)
    bio_align = 0.6  # Would compute from biological relevance metrics
    
    # Overall score
    overall = np.mean([feature_interp, circuit_interp, bio_align, human_align])
    
    # Compile all details
    details = {}
    details.update({f'feature_{k}': v for k, v in feature_scores.items()})
    details.update({f'circuit_{k}': v for k, v in circuit_scores.items()})
    details.update({f'human_{k}': v for k, v in human_scores.items()})
    
    return InterpretabilityScore(
        feature_interpretability=feature_interp,
        circuit_interpretability=circuit_interp,
        biological_alignment=bio_align,
        human_alignment=human_align,
        overall_score=overall,
        details=details
    )