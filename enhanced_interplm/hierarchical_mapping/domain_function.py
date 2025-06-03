# 功能域映射 
# enhanced_interplm/hierarchical_mapping/domain_function.py

"""
Advanced domain and functional region mapping.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Union
from dataclasses import dataclass
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
import networkx as nx


@dataclass
class ProteinDomain:
    """Container for protein domain information."""
    domain_id: str
    domain_type: str
    start: int
    end: int
    confidence: float
    subfamily: Optional[str] = None
    functional_sites: Optional[List[int]] = None
    interactions: Optional[List[str]] = None


class AdvancedDomainMapper(nn.Module):
    """
    Advanced protein domain detection and functional mapping.
    """
    
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 512,
        num_domain_types: int = 500,  # Extended domain families
        use_domain_grammar: bool = True,
        use_coevolution: bool = True
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_domain_types = num_domain_types
        self.use_domain_grammar = use_domain_grammar
        self.use_coevolution = use_coevolution
        
        # Multi-resolution domain detection
        self.domain_convs = nn.ModuleList([
            # Short domains (20-50 residues)
            nn.Conv1d(feature_dim, hidden_dim // 4, kernel_size=25, padding=12, stride=5),
            # Medium domains (50-150 residues)
            nn.Conv1d(feature_dim, hidden_dim // 4, kernel_size=75, padding=37, stride=10),
            # Long domains (150-300 residues)
            nn.Conv1d(feature_dim, hidden_dim // 4, kernel_size=150, padding=75, stride=20),
            # Very long domains (300+ residues)
            nn.Conv1d(feature_dim, hidden_dim // 4, kernel_size=300, padding=150, stride=30)
        ])
        
        # Domain boundary refinement
        self.boundary_refiner = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # Domain type classifier with hierarchical structure
        self.domain_classifier = HierarchicalDomainClassifier(
            input_dim=hidden_dim,
            num_classes=num_domain_types
        )
        
        # Functional site predictor
        self.functional_site_detector = FunctionalSiteDetector(
            feature_dim=hidden_dim,
            num_site_types=10
        )
        
        # Domain grammar module
        if use_domain_grammar:
            self.domain_grammar = DomainGrammarModule(
                hidden_dim=hidden_dim,
                num_domain_types=num_domain_types
            )
            
        # Coevolution analyzer
        if use_coevolution:
            self.coevolution_analyzer = CoevolutionAnalyzer(
                feature_dim=hidden_dim
            )
            
        # Domain interaction predictor
        self.interaction_predictor = DomainInteractionPredictor(
            domain_dim=hidden_dim
        )
        
    def forward(
        self,
        features: torch.Tensor,
        evolutionary_features: Optional[torch.Tensor] = None,
        return_intermediates: bool = False
    ) -> Dict[str, Any]:
        """
        Detect and classify protein domains.
        
        Args:
            features: [batch_size, seq_len, feature_dim]
            evolutionary_features: Optional coevolution/conservation data
            return_intermediates: Return intermediate results
            
        Returns:
            Dictionary with domain predictions and analyses
        """
        batch_size, seq_len, _ = features.shape
        
        # Multi-resolution domain detection
        domain_features = self._multi_resolution_detection(features)
        
        # Refine domain boundaries
        refined_features, boundary_scores = self._refine_boundaries(domain_features)
        
        # Detect domain regions
        domain_regions = self._detect_domain_regions(boundary_scores)
        
        # Classify domains
        domain_predictions = self._classify_domains(refined_features, domain_regions)
        
        # Detect functional sites within domains
        functional_sites = self._detect_functional_sites(
            refined_features, domain_regions
        )
        
        # Apply domain grammar constraints
        if self.use_domain_grammar:
            grammar_scores = self.domain_grammar(
                domain_predictions, domain_regions
            )
            domain_predictions = self._apply_grammar_constraints(
                domain_predictions, grammar_scores
            )
            
        # Analyze coevolution patterns
        if self.use_coevolution and evolutionary_features is not None:
            coevolution_patterns = self.coevolution_analyzer(
                refined_features, evolutionary_features, domain_regions
            )
        else:
            coevolution_patterns = None
            
        # Predict domain interactions
        interactions = self.interaction_predictor(
            refined_features, domain_regions
        )
        
        results = {
            'domain_regions': domain_regions,
            'domain_predictions': domain_predictions,
            'functional_sites': functional_sites,
            'domain_interactions': interactions,
            'boundary_scores': boundary_scores
        }
        
        if coevolution_patterns is not None:
            results['coevolution_patterns'] = coevolution_patterns
            
        if return_intermediates:
            results['multi_res_features'] = domain_features
            results['refined_features'] = refined_features
            
        return results
        
    def _multi_resolution_detection(self, features: torch.Tensor) -> torch.Tensor:
        """Detect domains at multiple resolutions."""
        # Transpose for convolution
        features_t = features.transpose(1, 2)  # [batch, feature_dim, seq_len]
        
        multi_res_features = []
        
        for conv in self.domain_convs:
            conv_out = F.relu(conv(features_t))
            
            # Upsample to original resolution
            if conv_out.shape[-1] != features.shape[1]:
                conv_out = F.interpolate(
                    conv_out,
                    size=features.shape[1],
                    mode='linear',
                    align_corners=False
                )
                
            multi_res_features.append(conv_out)
            
        # Concatenate multi-resolution features
        combined = torch.cat(multi_res_features, dim=1)  # [batch, hidden_dim, seq_len]
        combined = combined.transpose(1, 2)  # [batch, seq_len, hidden_dim]
        
        return combined
        
    def _refine_boundaries(
        self,
        domain_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Refine domain boundaries using LSTM."""
        # Process with bidirectional LSTM
        refined, _ = self.boundary_refiner(domain_features)
        
        # Compute boundary scores
        # High score indicates domain boundary
        diff_features = refined[:, 1:] - refined[:, :-1]
        boundary_scores = torch.norm(diff_features, dim=-1)
        
        # Pad boundary scores
        boundary_scores = F.pad(boundary_scores, (0, 1), value=0)
        
        return refined, boundary_scores
        
    def _detect_domain_regions(
        self,
        boundary_scores: torch.Tensor,
        min_domain_length: int = 30
    ) -> List[List[Tuple[int, int]]]:
        """Detect domain regions from boundary scores."""
        batch_size = boundary_scores.shape[0]
        domain_regions = []
        
        for b in range(batch_size):
            scores = boundary_scores[b].cpu().numpy()
            
            # Find peaks in boundary scores
            from scipy.signal import find_peaks
            peaks, properties = find_peaks(
                scores,
                height=np.percentile(scores, 75),
                distance=min_domain_length
            )
            
            # Convert peaks to domain regions
            regions = []
            if len(peaks) > 0:
                # Add start if first peak is not at beginning
                if peaks[0] > min_domain_length:
                    regions.append((0, peaks[0]))
                    
                # Add regions between peaks
                for i in range(len(peaks) - 1):
                    if peaks[i+1] - peaks[i] >= min_domain_length:
                        regions.append((peaks[i], peaks[i+1]))
                        
                # Add end if last peak is not at end
                if len(scores) - peaks[-1] > min_domain_length:
                    regions.append((peaks[-1], len(scores)))
            else:
                # No clear boundaries, treat as single domain
                regions.append((0, len(scores)))
                
            domain_regions.append(regions)
            
        return domain_regions
        
    def _classify_domains(
        self,
        features: torch.Tensor,
        domain_regions: List[List[Tuple[int, int]]]
    ) -> List[List[Dict]]:
        """Classify each detected domain."""
        batch_predictions = []
        
        for b, regions in enumerate(domain_regions):
            predictions = []
            
            for start, end in regions:
                # Extract domain features
                domain_feat = features[b, start:end].mean(dim=0)
                
                # Classify domain
                class_logits = self.domain_classifier(domain_feat.unsqueeze(0))
                class_probs = F.softmax(class_logits, dim=-1)
                
                top_class = class_probs.argmax(dim=-1).item()
                confidence = class_probs.max(dim=-1)[0].item()
                
                predictions.append({
                    'start': start,
                    'end': end,
                    'domain_class': top_class,
                    'confidence': confidence,
                    'class_probs': class_probs.squeeze(0)
                })
                
            batch_predictions.append(predictions)
            
        return batch_predictions
        
    def _detect_functional_sites(
        self,
        features: torch.Tensor,
        domain_regions: List[List[Tuple[int, int]]]
    ) -> List[List[Dict]]:
        """Detect functional sites within domains."""
        batch_sites = []
        
        for b, regions in enumerate(domain_regions):
            domain_sites = []
            
            for start, end in regions:
                # Analyze domain for functional sites
                domain_features = features[b, start:end]
                
                site_predictions = self.functional_site_detector(
                    domain_features.unsqueeze(0)
                )
                
                # Find high-confidence sites
                for site_type in range(site_predictions.shape[-1]):
                    site_probs = site_predictions[0, :, site_type]
                    site_positions = torch.where(site_probs > 0.7)[0]
                    
                    for pos in site_positions:
                        domain_sites.append({
                            'position': start + pos.item(),
                            'site_type': site_type,
                            'confidence': site_probs[pos].item(),
                            'domain_region': (start, end)
                        })
                        
            batch_sites.append(domain_sites)
            
        return batch_sites
        
    def _apply_grammar_constraints(
        self,
        predictions: List[List[Dict]],
        grammar_scores: torch.Tensor
    ) -> List[List[Dict]]:
        """Apply domain grammar constraints to refine predictions."""
        # Adjust domain confidences based on grammar scores
        for b in range(len(predictions)):
            for i, domain in enumerate(predictions[b]):
                if i < grammar_scores.shape[1]:
                    grammar_weight = grammar_scores[b, i].item()
                    domain['confidence'] *= (0.5 + 0.5 * grammar_weight)
                    
        return predictions


class HierarchicalDomainClassifier(nn.Module):
    """
    Hierarchical domain classification with family/subfamily structure.
    """
    
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        
        # Coarse classification (families)
        self.family_classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, num_classes // 10)  # 10 subfamilies per family
        )
        
        # Fine classification (subfamilies)
        self.subfamily_classifier = nn.Sequential(
            nn.Linear(input_dim + num_classes // 10, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Hierarchical classification."""
        # First level: family
        family_logits = self.family_classifier(x)
        family_probs = F.softmax(family_logits, dim=-1)
        
        # Second level: subfamily conditioned on family
        combined = torch.cat([x, family_probs], dim=-1)
        subfamily_logits = self.subfamily_classifier(combined)
        
        return subfamily_logits


class FunctionalSiteDetector(nn.Module):
    """
    Detect various types of functional sites within domains.
    """
    
    def __init__(self, feature_dim: int, num_site_types: int = 10):
        super().__init__()
        
        self.site_types = [
            'active_site',
            'binding_site',
            'allosteric_site',
            'catalytic_residue',
            'regulatory_site',
            'ptm_site',
            'metal_binding',
            'nucleotide_binding',
            'protein_interface',
            'dna_binding'
        ]
        
        # Site-specific detectors
        self.site_detectors = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(feature_dim, 64, kernel_size=7, padding=3),
                nn.ReLU(),
                nn.Conv1d(64, 32, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Conv1d(32, 1, kernel_size=3, padding=1),
                nn.Sigmoid()
            )
            for _ in range(num_site_types)
        ])
        
    def forward(self, domain_features: torch.Tensor) -> torch.Tensor:
        """Detect functional sites."""
        # domain_features: [batch, seq_len, feature_dim]
        
        features_t = domain_features.transpose(1, 2)  # [batch, feature_dim, seq_len]
        
        site_predictions = []
        
        for detector in self.site_detectors:
            site_prob = detector(features_t)  # [batch, 1, seq_len]
            site_predictions.append(site_prob)
            
        # Stack predictions
        sites = torch.cat(site_predictions, dim=1)  # [batch, num_sites, seq_len]
        sites = sites.transpose(1, 2)  # [batch, seq_len, num_sites]
        
        return sites


class DomainGrammarModule(nn.Module):
    """
    Learn and apply domain arrangement grammar rules.
    """
    
    def __init__(self, hidden_dim: int, num_domain_types: int):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_domain_types = num_domain_types
        
        # Domain arrangement scorer
        self.arrangement_lstm = nn.LSTM(
            input_size=num_domain_types,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )
        
        # Grammar rule network
        self.grammar_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Common domain patterns (learned)
        self.pattern_embeddings = nn.Embedding(100, hidden_dim)
        
    def forward(
        self,
        domain_predictions: List[List[Dict]],
        domain_regions: List[List[Tuple[int, int]]]
    ) -> torch.Tensor:
        """Score domain arrangements based on grammar."""
        batch_size = len(domain_predictions)
        max_domains = max(len(preds) for preds in domain_predictions)
        
        # Convert predictions to sequence
        domain_sequences = torch.zeros(batch_size, max_domains, self.num_domain_types)
        
        for b, predictions in enumerate(domain_predictions):
            for i, pred in enumerate(predictions):
                if i < max_domains:
                    domain_sequences[b, i] = pred['class_probs']
                    
        # Process with LSTM
        lstm_out, _ = self.arrangement_lstm(domain_sequences)
        
        # Score each domain in context
        grammar_scores = self.grammar_scorer(lstm_out).squeeze(-1)
        
        return grammar_scores


class CoevolutionAnalyzer(nn.Module):
    """
    Analyze coevolution patterns within and between domains.
    """
    
    def __init__(self, feature_dim: int):
        super().__init__()
        
        self.feature_dim = feature_dim
        
        # Coevolution pattern detector
        self.pattern_detector = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        domain_features: torch.Tensor,
        evolutionary_features: torch.Tensor,
        domain_regions: List[List[Tuple[int, int]]]
    ) -> Dict[str, torch.Tensor]:
        """Analyze coevolution patterns."""
        batch_size = domain_features.shape[0]
        coevolution_results = []
        
        for b in range(batch_size):
            domain_coevo = []
            
            for start, end in domain_regions[b]:
                # Extract domain-specific features
                domain_feat = domain_features[b, start:end]
                evo_feat = evolutionary_features[b, start:end]
                
                # Compute pairwise coevolution scores
                n_positions = end - start
                coevo_matrix = torch.zeros(n_positions, n_positions)
                
                for i in range(n_positions):
                    for j in range(i + 1, n_positions):
                        combined = torch.cat([
                            domain_feat[i] * evo_feat[i],
                            domain_feat[j] * evo_feat[j]
                        ])
                        
                        coevo_score = self.pattern_detector(combined.unsqueeze(0))
                        coevo_matrix[i, j] = coevo_score
                        coevo_matrix[j, i] = coevo_score
                        
                domain_coevo.append({
                    'region': (start, end),
                    'coevolution_matrix': coevo_matrix,
                    'coevolution_strength': coevo_matrix.mean().item()
                })
                
            coevolution_results.append(domain_coevo)
            
        return {'domain_coevolution': coevolution_results}


class DomainInteractionPredictor(nn.Module):
    """
    Predict interactions between domains.
    """
    
    def __init__(self, domain_dim: int):
        super().__init__()
        
        self.interaction_scorer = nn.Sequential(
            nn.Linear(domain_dim * 2, domain_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(domain_dim, domain_dim // 2),
            nn.ReLU(),
            nn.Linear(domain_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        domain_features: torch.Tensor,
        domain_regions: List[List[Tuple[int, int]]]
    ) -> List[Dict]:
        """Predict domain-domain interactions."""
        batch_interactions = []
        
        for b, regions in enumerate(domain_regions):
            interactions = []
            
            # Get average features for each domain
            domain_avg_features = []
            for start, end in regions:
                avg_feat = domain_features[b, start:end].mean(dim=0)
                domain_avg_features.append(avg_feat)
                
            # Predict pairwise interactions
            for i in range(len(regions)):
                for j in range(i + 1, len(regions)):
                    combined = torch.cat([
                        domain_avg_features[i],
                        domain_avg_features[j]
                    ])
                    
                    interaction_score = self.interaction_scorer(
                        combined.unsqueeze(0)
                    ).item()
                    
                    if interaction_score > 0.5:  # Threshold
                        interactions.append({
                            'domain1': regions[i],
                            'domain2': regions[j],
                            'interaction_score': interaction_score,
                            'interaction_type': self._classify_interaction(
                                domain_avg_features[i],
                                domain_avg_features[j]
                            )
                        })
                        
            batch_interactions.append(interactions)
            
        return batch_interactions
        
    def _classify_interaction(self, feat1: torch.Tensor, feat2: torch.Tensor) -> str:
        """Classify type of domain interaction."""
        # Simplified classification based on feature similarity
        similarity = F.cosine_similarity(feat1, feat2, dim=0).item()
        
        if similarity > 0.8:
            return 'homotypic'  # Similar domains
        elif similarity > 0.5:
            return 'heterotypic_cooperative'  # Different but cooperative
        else:
            return 'heterotypic_regulatory'  # Regulatory interaction