# enhanced_interplm/evaluation/comprehensive_metrics.py

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
from scipy.stats import spearmanr, pearsonr
import pandas as pd
from dataclasses import dataclass


@dataclass
class MetricResult:
    """Container for metric results with metadata."""
    value: float
    std: Optional[float] = None
    details: Optional[Dict] = None
    
    def __repr__(self):
        if self.std is not None:
            return f"{self.value:.4f} ± {self.std:.4f}"
        return f"{self.value:.4f}"


class InterpretabilityMetrics:
    """
    Comprehensive metrics for evaluating PLM interpretability.
    """
    
    def __init__(self):
        self.results = {}
        
    def compute_all_metrics(
        self,
        features: torch.Tensor,
        reconstructed: torch.Tensor,
        original: torch.Tensor,
        circuits: Optional[List[Dict]] = None,
        annotations: Optional[Dict] = None
    ) -> Dict[str, MetricResult]:
        """Compute all interpretability metrics."""
        
        # Basic reconstruction metrics
        self.results['reconstruction'] = self._compute_reconstruction_metrics(
            reconstructed, original
        )
        
        # Feature quality metrics
        self.results['features'] = self._compute_feature_metrics(features)
        
        # Circuit quality metrics
        if circuits is not None:
            self.results['circuits'] = self._compute_circuit_metrics(circuits, features)
            
        # Biological relevance metrics
        if annotations is not None:
            self.results['biological'] = self._compute_biological_metrics(
                features, annotations
            )
            
        return self.results
        
    def _compute_reconstruction_metrics(
        self,
        reconstructed: torch.Tensor,
        original: torch.Tensor
    ) -> Dict[str, MetricResult]:
        """Compute reconstruction quality metrics."""
        metrics = {}
        
        # MSE
        mse = F.mse_loss(reconstructed, original)
        metrics['mse'] = MetricResult(value=mse.item())
        
        # Normalized reconstruction error
        norm_error = torch.norm(reconstructed - original, dim=-1) / (
            torch.norm(original, dim=-1) + 1e-8
        )
        metrics['normalized_error'] = MetricResult(
            value=norm_error.mean().item(),
            std=norm_error.std().item()
        )
        
        # Cosine similarity
        cos_sim = F.cosine_similarity(
            reconstructed.flatten(start_dim=2),
            original.flatten(start_dim=2),
            dim=-1
        )
        metrics['cosine_similarity'] = MetricResult(
            value=cos_sim.mean().item(),
            std=cos_sim.std().item()
        )
        
        # Variance explained
        total_var = original.var()
        residual_var = (original - reconstructed).var()
        var_explained = 1 - (residual_var / total_var)
        metrics['variance_explained'] = MetricResult(value=var_explained.item())
        
        return metrics
        
    def _compute_feature_metrics(
        self,
        features: torch.Tensor
    ) -> Dict[str, MetricResult]:
        """Compute feature quality metrics."""
        metrics = {}
        
        # Sparsity metrics
        # L0 sparsity (percentage of non-zero features)
        l0_sparsity = (features.abs() > 0.01).float().mean()
        metrics['l0_sparsity'] = MetricResult(value=l0_sparsity.item())
        
        # L1 sparsity
        l1_sparsity = features.abs().mean()
        metrics['l1_sparsity'] = MetricResult(value=l1_sparsity.item())
        
        # Feature activation frequency
        activation_freq = (features.abs() > 0.1).float().mean(dim=(0, 1, 2))
        metrics['activation_frequency'] = MetricResult(
            value=activation_freq.mean().item(),
            std=activation_freq.std().item(),
            details={'per_feature': activation_freq.cpu().numpy()}
        )
        
        # Feature diversity (entropy of activation patterns)
        feature_probs = F.softmax(features.abs().mean(dim=(0, 1)), dim=-1)
        entropy = -(feature_probs * torch.log(feature_probs + 1e-8)).sum()
        metrics['feature_entropy'] = MetricResult(value=entropy.item())
        
        # Dead neurons
        dead_neurons = (features.abs().max(dim=(0, 1, 2))[0] < 0.01).float().mean()
        metrics['dead_neurons_ratio'] = MetricResult(value=dead_neurons.item())
        
        # Feature correlation
        if features.shape[-1] < 10000:  # Avoid memory issues
            feat_flat = features.flatten(0, 2)  # [batch*layers*seq, features]
            feat_corr = torch.corrcoef(feat_flat.T)
            
            # Average off-diagonal correlation
            mask = ~torch.eye(feat_corr.shape[0], dtype=bool)
            avg_corr = feat_corr[mask].abs().mean()
            metrics['feature_correlation'] = MetricResult(value=avg_corr.item())
            
        return metrics
        
    def _compute_circuit_metrics(
        self,
        circuits: List[Dict],
        features: torch.Tensor
    ) -> Dict[str, MetricResult]:
        """Compute circuit quality metrics."""
        metrics = {}
        
        if not circuits:
            return metrics
            
        # Number of circuits
        metrics['num_circuits'] = MetricResult(value=float(len(circuits)))
        
        # Circuit sizes
        circuit_sizes = [len(c['instances'][0]) if c['instances'] else 0 for c in circuits]
        metrics['avg_circuit_size'] = MetricResult(
            value=np.mean(circuit_sizes) if circuit_sizes else 0,
            std=np.std(circuit_sizes) if circuit_sizes else 0
        )
        
        # Circuit importance scores
        importance_scores = [c.get('importance', 0) for c in circuits]
        metrics['avg_circuit_importance'] = MetricResult(
            value=np.mean(importance_scores) if importance_scores else 0,
            std=np.std(importance_scores) if importance_scores else 0
        )
        
        # Circuit coherence
        coherence_scores = []
        for circuit in circuits[:10]:  # Evaluate top 10 circuits
            if 'validation_scores' in circuit:
                coherence_scores.append(circuit['validation_scores'].get('coherence', 0))
                
        if coherence_scores:
            metrics['circuit_coherence'] = MetricResult(
                value=np.mean(coherence_scores),
                std=np.std(coherence_scores)
            )
            
        # Circuit coverage (percentage of features involved in circuits)
        all_features_in_circuits = set()
        for circuit in circuits:
            for instance in circuit.get('instances', []):
                for node in instance:
                    # Extract feature index from node ID
                    if '_F' in node:
                        feat_idx = int(node.split('_F')[1])
                        all_features_in_circuits.add(feat_idx)
                        
        circuit_coverage = len(all_features_in_circuits) / features.shape[-1]
        metrics['circuit_coverage'] = MetricResult(value=circuit_coverage)
        
        return metrics
        
    def _compute_biological_metrics(
        self,
        features: torch.Tensor,
        annotations: Dict
    ) -> Dict[str, MetricResult]:
        """Compute biological relevance metrics."""
        metrics = {}
        
        # Secondary structure prediction accuracy
        if 'secondary_structure' in annotations:
            ss_metrics = self._evaluate_secondary_structure(features, annotations['secondary_structure'])
            metrics.update(ss_metrics)
            
        # Domain prediction metrics
        if 'domains' in annotations:
            domain_metrics = self._evaluate_domains(features, annotations['domains'])
            metrics.update(domain_metrics)
            
        # GO term prediction metrics
        if 'go_terms' in annotations:
            go_metrics = self._evaluate_go_terms(features, annotations['go_terms'])
            metrics.update(go_metrics)
            
        # Conservation correlation
        if 'conservation' in annotations:
            conservation_corr = self._evaluate_conservation(features, annotations['conservation'])
            metrics['conservation_correlation'] = conservation_corr
            
        return metrics
        
    def _evaluate_secondary_structure(
        self,
        features: torch.Tensor,
        true_ss: torch.Tensor
    ) -> Dict[str, MetricResult]:
        """Evaluate secondary structure prediction."""
        metrics = {}
        
        # Use a simple linear probe for evaluation
        # In practice, you would use the trained SS mapper
        probe = nn.Linear(features.shape[-1], 8).to(features.device)
        
        # Simple optimization
        optimizer = torch.optim.Adam(probe.parameters(), lr=0.01)
        
        for _ in range(100):
            pred_ss = probe(features.flatten(0, 2))
            true_ss_flat = true_ss.flatten(0, 1)
            
            # Mask padding
            mask = true_ss_flat.sum(dim=-1) > 0
            
            if mask.sum() > 0:
                loss = F.cross_entropy(
                    pred_ss[mask],
                    true_ss_flat[mask].argmax(dim=-1)
                )
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
        # Evaluate
        with torch.no_grad():
            pred_ss = probe(features.flatten(0, 2))
            pred_classes = pred_ss.argmax(dim=-1)
            true_classes = true_ss.flatten(0, 1).argmax(dim=-1)
            
            mask = true_ss.flatten(0, 1).sum(dim=-1) > 0
            
            if mask.sum() > 0:
                accuracy = (pred_classes[mask] == true_classes[mask]).float().mean()
                metrics['ss_accuracy'] = MetricResult(value=accuracy.item())
                
                # Per-class accuracy
                class_acc = []
                for i in range(8):
                    class_mask = true_classes[mask] == i
                    if class_mask.sum() > 0:
                        acc = (pred_classes[mask][class_mask] == i).float().mean()
                        class_acc.append(acc.item())
                        
                metrics['ss_class_accuracy'] = MetricResult(
                    value=np.mean(class_acc),
                    std=np.std(class_acc)
                )
                
        return metrics
        
    def _evaluate_domains(
        self,
        features: torch.Tensor,
        true_domains: torch.Tensor
    ) -> Dict[str, MetricResult]:
        """Evaluate domain prediction."""
        metrics = {}
        
        # Simplified domain detection evaluation
        # Aggregate features within predicted domains
        domain_features = []
        domain_labels = []
        
        batch_size, num_layers, seq_len, _ = features.shape
        
        for b in range(batch_size):
            # Find domain boundaries in ground truth
            domain_mask = true_domains[b] > 0
            
            if domain_mask.sum() > 0:
                # Get average features within domain
                domain_feat = features[b, :, domain_mask].mean(dim=1).mean(dim=0)
                domain_features.append(domain_feat)
                domain_labels.append(1)
                
                # Get average features outside domain
                non_domain_feat = features[b, :, ~domain_mask].mean(dim=1).mean(dim=0)
                domain_features.append(non_domain_feat)
                domain_labels.append(0)
                
        if domain_features:
            domain_features = torch.stack(domain_features)
            domain_labels = torch.tensor(domain_labels)
            
            # Simple separability test
            # In practice, use trained domain classifier
            probe = nn.Linear(features.shape[-1], 1).to(features.device)
            
            # Quick training
            optimizer = torch.optim.Adam(probe.parameters(), lr=0.01)
            for _ in range(50):
                pred = probe(domain_features).squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(pred, domain_labels.float())
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            # Evaluate
            with torch.no_grad():
                pred = torch.sigmoid(probe(domain_features).squeeze(-1))
                auc = roc_auc_score(domain_labels.cpu(), pred.cpu())
                metrics['domain_auc'] = MetricResult(value=auc)
                
        return metrics
        
    def _evaluate_go_terms(
        self,
        features: torch.Tensor,
        true_go: torch.Tensor
    ) -> Dict[str, MetricResult]:
        """Evaluate GO term prediction."""
        metrics = {}
        
        # Aggregate features across sequence
        seq_features = features.mean(dim=2)  # [batch, layers, features]
        
        # Flatten batch and layers
        seq_features_flat = seq_features.flatten(0, 1)
        true_go_flat = true_go.repeat(features.shape[1], 1)
        
        # For each GO term, evaluate prediction
        go_aucs = []
        go_aps = []
        
        for i in range(true_go.shape[-1]):
            if true_go_flat[:, i].sum() > 0:  # At least one positive example
                # Simple linear probe
                probe = nn.Linear(features.shape[-1], 1).to(features.device)
                
                # Quick optimization
                optimizer = torch.optim.Adam(probe.parameters(), lr=0.01)
                for _ in range(50):
                    pred = probe(seq_features_flat).squeeze(-1)
                    loss = F.binary_cross_entropy_with_logits(pred, true_go_flat[:, i])
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                # Evaluate
                with torch.no_grad():
                    pred = torch.sigmoid(probe(seq_features_flat).squeeze(-1))
                    
                    try:
                        auc = roc_auc_score(true_go_flat[:, i].cpu(), pred.cpu())
                        ap = average_precision_score(true_go_flat[:, i].cpu(), pred.cpu())
                        
                        go_aucs.append(auc)
                        go_aps.append(ap)
                    except:
                        pass
                        
        if go_aucs:
            metrics['go_term_auc'] = MetricResult(
                value=np.mean(go_aucs),
                std=np.std(go_aucs)
            )
            metrics['go_term_ap'] = MetricResult(
                value=np.mean(go_aps),
                std=np.std(go_aps)
            )
            
        return metrics
        
    def _evaluate_conservation(
        self,
        features: torch.Tensor,
        conservation: torch.Tensor
    ) -> MetricResult:
        """Evaluate correlation with evolutionary conservation."""
        # Compute feature activation strength
        feature_strength = features.abs().mean(dim=-1)  # [batch, layers, seq]
        
        correlations = []
        
        for b in range(features.shape[0]):
            for l in range(features.shape[1]):
                # Mask padding
                mask = conservation[b] >= 0
                
                if mask.sum() > 10:  # Need enough data points
                    corr, _ = spearmanr(
                        feature_strength[b, l][mask].cpu(),
                        conservation[b][mask].cpu()
                    )
                    
                    if not np.isnan(corr):
                        correlations.append(corr)
                        
        if correlations:
            return MetricResult(
                value=np.mean(correlations),
                std=np.std(correlations)
            )
        else:
            return MetricResult(value=0.0)


class BiologicalRelevanceScorer:
    """
    Scores features based on their biological relevance.
    """
    
    def __init__(self, known_motifs: Optional[Dict[str, torch.Tensor]] = None):
        self.known_motifs = known_motifs or self._load_default_motifs()
        
    def _load_default_motifs(self) -> Dict[str, torch.Tensor]:
        """Load default protein motifs."""
        # Simplified motif patterns
        motifs = {
            'helix_cap': torch.tensor([1, 0, 0, 0, 1, 1, 1, 1]),  # N-cap pattern
            'beta_turn': torch.tensor([0, 1, 1, 0, 0, 1, 1, 0]),
            'zinc_finger': torch.tensor([1, 0, 0, 1, 0, 0, 1, 0]),  # C-X-X-C pattern
        }
        return motifs
        
    def score_features(
        self,
        features: torch.Tensor,
        annotations: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Score features based on biological relevance."""
        scores = {}
        
        # Motif detection score
        if self.known_motifs:
            motif_scores = self._score_motif_detection(features, annotations)
            scores.update(motif_scores)
            
        # Functional site enrichment
        if 'functional_sites' in annotations:
            site_scores = self._score_functional_sites(features, annotations['functional_sites'])
            scores.update(site_scores)
            
        # Structure-function correlation
        if 'secondary_structure' in annotations and 'conservation' in annotations:
            struct_func_score = self._score_structure_function(
                features,
                annotations['secondary_structure'],
                annotations['conservation']
            )
            scores['structure_function_correlation'] = struct_func_score
            
        return scores
        
    def _score_motif_detection(
        self,
        features: torch.Tensor,
        annotations: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Score ability to detect known motifs."""
        scores = {}
        
        # For each known motif, check if any features specifically activate on it
        for motif_name, motif_pattern in self.known_motifs.items():
            motif_score = self._compute_motif_enrichment(features, motif_pattern)
            scores[f'{motif_name}_enrichment'] = motif_score
            
        return scores
        
    def _compute_motif_enrichment(
        self,
        features: torch.Tensor,
        motif_pattern: torch.Tensor
    ) -> float:
        """Compute enrichment of features on a specific motif."""
        # Simplified: correlation between feature activation and motif presence
        # In practice, use more sophisticated pattern matching
        
        batch_size, num_layers, seq_len, num_features = features.shape
        motif_len = len(motif_pattern)
        
        enrichment_scores = []
        
        for feat_idx in range(num_features):
            feat_activations = features[:, :, :, feat_idx]
            
            # Compute correlation with motif pattern
            # This is a simplified version
            correlations = []
            
            for b in range(batch_size):
                for l in range(num_layers):
                    for i in range(seq_len - motif_len + 1):
                        window = feat_activations[b, l, i:i+motif_len]
                        
                        if window.sum() > 0:
                            corr = torch.corrcoef(
                                torch.stack([window, motif_pattern.float()])
                            )[0, 1]
                            
                            if not torch.isnan(corr):
                                correlations.append(corr.item())
                                
            if correlations:
                enrichment_scores.append(np.mean(correlations))
                
        return max(enrichment_scores) if enrichment_scores else 0.0
        
    def _score_functional_sites(
        self,
        features: torch.Tensor,
        functional_sites: torch.Tensor
    ) -> Dict[str, float]:
        """Score enrichment at functional sites."""
        scores = {}
        
        # Categories: active, binding, allosteric, catalytic, regulatory
        site_names = ['active', 'binding', 'allosteric', 'catalytic', 'regulatory']
        
        for i, site_name in enumerate(site_names):
            if i < functional_sites.shape[-1]:
                site_mask = functional_sites[:, :, i] > 0.5
                
                # Average feature activation at functional sites
                site_activation = features[site_mask].abs().mean()
                
                # Compare to background
                background_activation = features[~site_mask].abs().mean()
                
                enrichment = (site_activation / (background_activation + 1e-8)).item()
                scores[f'{site_name}_site_enrichment'] = enrichment
                
        return scores
        
    def _score_structure_function(
        self,
        features: torch.Tensor,
        secondary_structure: torch.Tensor,
        conservation: torch.Tensor
    ) -> float:
        """Score correlation between structure and function."""
        # Identify conserved structural elements
        # High conservation + specific secondary structure
        
        # Get helix and sheet masks
        helix_mask = secondary_structure[:, :, 0] > 0.5  # Alpha helix
        sheet_mask = secondary_structure[:, :, 2] > 0.5  # Beta sheet
        
        # Conserved helices and sheets
        conserved_helix = helix_mask & (conservation > 0.7)
        conserved_sheet = sheet_mask & (conservation > 0.7)
        
        # Check if features specifically activate on conserved structural elements
        helix_enrichment = features[conserved_helix].abs().mean() / (
            features[helix_mask & ~conserved_helix].abs().mean() + 1e-8
        )
        
        sheet_enrichment = features[conserved_sheet].abs().mean() / (
            features[sheet_mask & ~conserved_sheet].abs().mean() + 1e-8
        )
        
        return max(helix_enrichment.item(), sheet_enrichment.item())