# 生物学相关性 
# enhanced_interplm/evaluation/biological_relevance.py

"""
Comprehensive biological relevance scoring for discovered features and circuits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import pandas as pd
from scipy.stats import hypergeom, fisher_exact
from sklearn.metrics import matthews_corrcoef, roc_auc_score
import json
from pathlib import Path
from Bio import SeqIO, Align
from Bio.SubsMat import MatrixInfo
import re


@dataclass
class BiologicalAnnotation:
    """Container for biological annotations."""
    feature_id: int
    annotation_type: str  # 'function', 'structure', 'evolution', 'disease'
    annotation: str
    confidence: float
    evidence: List[str]
    references: Optional[List[str]] = None
    
    def to_dict(self) -> dict:
        return {
            'feature_id': self.feature_id,
            'annotation_type': self.annotation_type,
            'annotation': self.annotation,
            'confidence': self.confidence,
            'evidence': self.evidence,
            'references': self.references
        }


class BiologicalRelevanceScorer:
    """
    Advanced biological relevance scoring for features.
    """
    
    def __init__(
        self,
        go_annotation_path: Optional[Path] = None,
        pfam_database_path: Optional[Path] = None,
        disease_variant_path: Optional[Path] = None,
        conservation_data_path: Optional[Path] = None
    ):
        # Load biological databases
        self.go_annotations = self._load_go_annotations(go_annotation_path)
        self.pfam_database = self._load_pfam_database(pfam_database_path)
        self.disease_variants = self._load_disease_variants(disease_variant_path)
        self.conservation_data = self._load_conservation_data(conservation_data_path)
        
        # Initialize scoring components
        self.functional_scorer = FunctionalRelevanceScorer(self.go_annotations)
        self.structural_scorer = StructuralRelevanceScorer(self.pfam_database)
        self.evolutionary_scorer = EvolutionaryRelevanceScorer(self.conservation_data)
        self.clinical_scorer = ClinicalRelevanceScorer(self.disease_variants)
        
    def score_features(
        self,
        features: torch.Tensor,
        sequences: List[str],
        annotations: Optional[Dict[str, torch.Tensor]] = None,
        structures: Optional[torch.Tensor] = None,
        evolution_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive biological relevance scoring.
        
        Args:
            features: Feature activations [batch, seq_len, num_features]
            sequences: List of protein sequences
            annotations: Optional annotations (GO terms, domains, etc.)
            structures: Optional 3D structures
            evolution_data: Optional evolutionary data (MSAs, conservation)
            
        Returns:
            Dictionary containing relevance scores and annotations
        """
        num_features = features.shape[-1]
        biological_annotations = []
        
        # Score each feature
        feature_scores = {}
        
        for feat_idx in range(num_features):
            feat_activation = features[:, :, feat_idx]
            
            # Skip inactive features
            if feat_activation.abs().max() < 0.1:
                continue
                
            # 1. Functional relevance
            func_score, func_annot = self.functional_scorer.score_feature(
                feat_activation, sequences, annotations
            )
            
            # 2. Structural relevance
            struct_score, struct_annot = self.structural_scorer.score_feature(
                feat_activation, sequences, structures
            )
            
            # 3. Evolutionary relevance
            evo_score, evo_annot = self.evolutionary_scorer.score_feature(
                feat_activation, sequences, evolution_data
            )
            
            # 4. Clinical relevance
            clin_score, clin_annot = self.clinical_scorer.score_feature(
                feat_activation, sequences
            )
            
            # Combine scores
            total_score = self._combine_scores({
                'functional': func_score,
                'structural': struct_score,
                'evolutionary': evo_score,
                'clinical': clin_score
            })
            
            feature_scores[feat_idx] = {
                'total_score': total_score,
                'component_scores': {
                    'functional': func_score,
                    'structural': struct_score,
                    'evolutionary': evo_score,
                    'clinical': clin_score
                }
            }
            
            # Collect annotations
            for annot in [func_annot, struct_annot, evo_annot, clin_annot]:
                if annot:
                    biological_annotations.append(annot)
                    
        # Identify biologically important features
        important_features = self._identify_important_features(feature_scores)
        
        # Generate biological insights
        insights = self._generate_biological_insights(
            important_features, biological_annotations
        )
        
        return {
            'feature_scores': feature_scores,
            'annotations': biological_annotations,
            'important_features': important_features,
            'insights': insights,
            'summary': self._summarize_biological_relevance(feature_scores)
        }
        
    def _load_go_annotations(self, path: Optional[Path]) -> Dict:
        """Load Gene Ontology annotations."""
        if path and path.exists():
            with open(path, 'r') as f:
                return json.load(f)
        else:
            # Default GO terms
            return {
                'molecular_function': {
                    'catalytic_activity': ['GO:0003824'],
                    'binding': ['GO:0005488'],
                    'transporter': ['GO:0005215']
                },
                'biological_process': {
                    'metabolism': ['GO:0008152'],
                    'regulation': ['GO:0065007'],
                    'signaling': ['GO:0023052']
                },
                'cellular_component': {
                    'membrane': ['GO:0016020'],
                    'nucleus': ['GO:0005634'],
                    'cytoplasm': ['GO:0005737']
                }
            }
            
    def _load_pfam_database(self, path: Optional[Path]) -> Dict:
        """Load Pfam domain database."""
        if path and path.exists():
            with open(path, 'r') as f:
                return json.load(f)
        else:
            # Simplified Pfam entries
            return {
                'PF00001': {'name': 'GPCR', 'function': 'G-protein coupled receptor'},
                'PF00018': {'name': 'SH3', 'function': 'SH3 domain'},
                'PF00169': {'name': 'PH', 'function': 'PH domain'},
                'PF00595': {'name': 'PDZ', 'function': 'PDZ domain'},
                'PF00069': {'name': 'Pkinase', 'function': 'Protein kinase domain'}
            }
            
    def _load_disease_variants(self, path: Optional[Path]) -> Dict:
        """Load disease-associated variants."""
        if path and path.exists():
            with open(path, 'r') as f:
                return json.load(f)
        else:
            # Example disease variants
            return {
                'pathogenic': [
                    {'pos': 53, 'wt': 'R', 'mut': 'H', 'disease': 'cancer'},
                    {'pos': 175, 'wt': 'R', 'mut': 'H', 'disease': 'cancer'}
                ],
                'benign': [
                    {'pos': 72, 'wt': 'P', 'mut': 'R', 'disease': None}
                ]
            }
            
    def _load_conservation_data(self, path: Optional[Path]) -> Dict:
        """Load conservation data."""
        if path and path.exists():
            with open(path, 'r') as f:
                return json.load(f)
        else:
            return {}
            
    def _combine_scores(self, scores: Dict[str, float]) -> float:
        """Combine multiple relevance scores."""
        # Weighted combination
        weights = {
            'functional': 0.3,
            'structural': 0.25,
            'evolutionary': 0.25,
            'clinical': 0.2
        }
        
        total = sum(scores.get(k, 0) * w for k, w in weights.items())
        return total
        
    def _identify_important_features(
        self,
        feature_scores: Dict[int, Dict]
    ) -> List[int]:
        """Identify biologically important features."""
        # Sort by total score
        sorted_features = sorted(
            feature_scores.items(),
            key=lambda x: x[1]['total_score'],
            reverse=True
        )
        
        # Apply threshold
        threshold = 0.6
        important = [
            feat_id for feat_id, scores in sorted_features
            if scores['total_score'] > threshold
        ]
        
        return important
        
    def _generate_biological_insights(
        self,
        important_features: List[int],
        annotations: List[BiologicalAnnotation]
    ) -> List[str]:
        """Generate biological insights from analysis."""
        insights = []
        
        # Group annotations by type
        annot_by_type = defaultdict(list)
        for annot in annotations:
            if annot.feature_id in important_features:
                annot_by_type[annot.annotation_type].append(annot)
                
        # Generate insights
        if annot_by_type['function']:
            func_annotations = annot_by_type['function']
            unique_functions = set(a.annotation for a in func_annotations)
            insights.append(
                f"Identified {len(unique_functions)} distinct functional features, "
                f"including: {', '.join(list(unique_functions)[:3])}"
            )
            
        if annot_by_type['structure']:
            struct_annotations = annot_by_type['structure']
            insights.append(
                f"Found {len(struct_annotations)} structure-specific features "
                f"that may represent key structural elements"
            )
            
        if annot_by_type['evolution']:
            evo_annotations = annot_by_type['evolution']
            highly_conserved = [a for a in evo_annotations if 'highly conserved' in a.annotation]
            if highly_conserved:
                insights.append(
                    f"{len(highly_conserved)} features show high evolutionary conservation, "
                    f"suggesting functional importance"
                )
                
        if annot_by_type['disease']:
            disease_annotations = annot_by_type['disease']
            insights.append(
                f"Discovered {len(disease_annotations)} features associated with "
                f"disease-relevant positions"
            )
            
        return insights
        
    def _summarize_biological_relevance(
        self,
        feature_scores: Dict[int, Dict]
    ) -> Dict[str, Any]:
        """Summarize biological relevance analysis."""
        if not feature_scores:
            return {'status': 'No biologically relevant features found'}
            
        all_scores = [s['total_score'] for s in feature_scores.values()]
        component_scores = defaultdict(list)
        
        for scores in feature_scores.values():
            for comp, score in scores['component_scores'].items():
                component_scores[comp].append(score)
                
        summary = {
            'num_relevant_features': len(feature_scores),
            'avg_relevance_score': np.mean(all_scores),
            'max_relevance_score': np.max(all_scores),
            'component_averages': {
                comp: np.mean(scores) for comp, scores in component_scores.items()
            },
            'highly_relevant': sum(1 for s in all_scores if s > 0.7),
            'moderately_relevant': sum(1 for s in all_scores if 0.4 < s <= 0.7),
            'weakly_relevant': sum(1 for s in all_scores if s <= 0.4)
        }
        
        return summary


class FunctionalRelevanceScorer:
    """Scores features based on functional relevance."""
    
    def __init__(self, go_annotations: Dict):
        self.go_annotations = go_annotations
        
    def score_feature(
        self,
        feature_activation: torch.Tensor,
        sequences: List[str],
        annotations: Optional[Dict] = None
    ) -> Tuple[float, Optional[BiologicalAnnotation]]:
        """Score functional relevance of a feature."""
        score = 0.0
        evidence = []
        
        # Check correlation with GO terms if available
        if annotations and 'go_terms' in annotations:
            go_correlation = self._compute_go_correlation(
                feature_activation, annotations['go_terms']
            )
            if go_correlation > 0.5:
                score += go_correlation * 0.4
                evidence.append(f"GO term correlation: {go_correlation:.3f}")
                
        # Check for catalytic residue patterns
        catalytic_score = self._check_catalytic_patterns(
            feature_activation, sequences
        )
        if catalytic_score > 0:
            score += catalytic_score * 0.3
            evidence.append("Catalytic residue pattern detected")
            
        # Check for binding site patterns
        binding_score = self._check_binding_patterns(
            feature_activation, sequences
        )
        if binding_score > 0:
            score += binding_score * 0.3
            evidence.append("Binding site pattern detected")
            
        annotation = None
        if score > 0.3 and evidence:
            annotation = BiologicalAnnotation(
                feature_id=0,  # Will be set by caller
                annotation_type='function',
                annotation='; '.join(evidence),
                confidence=score,
                evidence=evidence
            )
            
        return score, annotation
        
    def _compute_go_correlation(
        self,
        activation: torch.Tensor,
        go_terms: torch.Tensor
    ) -> float:
        """Compute correlation with GO terms."""
        # Average activation per protein
        protein_activation = activation.mean(dim=1)
        
        # Compute correlation with each GO term
        correlations = []
        
        for i in range(go_terms.shape[-1]):
            go_vector = go_terms[:, i]
            
            if go_vector.sum() > 0:
                corr = torch.corrcoef(
                    torch.stack([protein_activation, go_vector])
                )[0, 1]
                
                if not torch.isnan(corr):
                    correlations.append(abs(corr.item()))
                    
        return max(correlations) if correlations else 0.0
        
    def _check_catalytic_patterns(
        self,
        activation: torch.Tensor,
        sequences: List[str]
    ) -> float:
        """Check for catalytic residue patterns."""
        catalytic_residues = {'H', 'D', 'S', 'C', 'E', 'K'}
        
        scores = []
        
        for i, seq in enumerate(sequences):
            # Find positions with high activation
            high_activation = activation[i] > activation[i].mean() + activation[i].std()
            
            # Check if high activation positions are catalytic residues
            catalytic_count = 0
            total_high = high_activation.sum().item()
            
            if total_high > 0:
                for j, is_high in enumerate(high_activation):
                    if is_high and j < len(seq) and seq[j] in catalytic_residues:
                        catalytic_count += 1
                        
                enrichment = catalytic_count / total_high
                expected = len(catalytic_residues) / 20  # Expected by chance
                
                if enrichment > expected * 1.5:
                    scores.append(enrichment / expected)
                    
        return np.mean(scores) if scores else 0.0
        
    def _check_binding_patterns(
        self,
        activation: torch.Tensor,
        sequences: List[str]
    ) -> float:
        """Check for binding site patterns."""
        # Simplified binding site detection
        binding_patterns = [
            'GxGxxG',  # Nucleotide binding
            'CxxC',    # Zinc binding
            'DxD',     # Metal binding
            '[ST]x[RK]',  # Kinase substrate
        ]
        
        scores = []
        
        for i, seq in enumerate(sequences):
            pattern_matches = 0
            
            for pattern in binding_patterns:
                # Check if pattern positions have high activation
                for match in re.finditer(pattern.replace('x', '.'), seq):
                    start, end = match.span()
                    
                    if activation[i, start:end].mean() > activation[i].mean():
                        pattern_matches += 1
                        
            if pattern_matches > 0:
                scores.append(min(1.0, pattern_matches / len(binding_patterns)))
                
        return np.mean(scores) if scores else 0.0


class StructuralRelevanceScorer:
    """Scores features based on structural relevance."""
    
    def __init__(self, pfam_database: Dict):
        self.pfam_database = pfam_database
        
    def score_feature(
        self,
        feature_activation: torch.Tensor,
        sequences: List[str],
        structures: Optional[torch.Tensor] = None
    ) -> Tuple[float, Optional[BiologicalAnnotation]]:
        """Score structural relevance of a feature."""
        score = 0.0
        evidence = []
        
        # Check for domain-specific activation
        domain_score = self._check_domain_specificity(
            feature_activation, sequences
        )
        if domain_score > 0:
            score += domain_score * 0.4
            evidence.append("Domain-specific activation")
            
        # Check for secondary structure preferences
        if structures is not None:
            ss_score = self._check_secondary_structure_preference(
                feature_activation, structures
            )
            if ss_score > 0:
                score += ss_score * 0.3
                evidence.append("Secondary structure preference")
                
        # Check for structural motifs
        motif_score = self._check_structural_motifs(
            feature_activation, sequences
        )
        if motif_score > 0:
            score += motif_score * 0.3
            evidence.append("Structural motif detected")
            
        annotation = None
        if score > 0.3 and evidence:
            annotation = BiologicalAnnotation(
                feature_id=0,
                annotation_type='structure',
                annotation='; '.join(evidence),
                confidence=score,
                evidence=evidence
            )
            
        return score, annotation
        
    def _check_domain_specificity(
        self,
        activation: torch.Tensor,
        sequences: List[str]
    ) -> float:
        """Check if feature is specific to protein domains."""
        # Simplified domain detection
        # In practice, would use HMM profiles
        
        domain_patterns = {
            'SH3': 'Y...[PV]..P',
            'PDZ': '[ST].[VIL]',
            'WW': 'W...W',
            'RING': 'C.{2}C.{9,39}C.{1,3}H.{2,3}C.{2}C.{4,48}C.{2}C'
        }
        
        domain_scores = []
        
        for i, seq in enumerate(sequences):
            for domain_name, pattern in domain_patterns.items():
                matches = list(re.finditer(pattern, seq))
                
                if matches:
                    # Check if activation is high in domain regions
                    domain_activation = []
                    
                    for match in matches:
                        start, end = match.span()
                        domain_activation.append(
                            activation[i, start:end].mean().item()
                        )
                        
                    if domain_activation:
                        avg_domain_act = np.mean(domain_activation)
                        avg_total_act = activation[i].mean().item()
                        
                        if avg_domain_act > avg_total_act * 1.5:
                            domain_scores.append(avg_domain_act / (avg_total_act + 1e-10))
                            
        return min(1.0, np.mean(domain_scores)) if domain_scores else 0.0
        
    def _check_secondary_structure_preference(
        self,
        activation: torch.Tensor,
        structures: torch.Tensor
    ) -> float:
        """Check for secondary structure preferences."""
        # Assuming structures contains secondary structure assignments
        # structures shape: [batch, seq_len, 8] (8-state DSSP)
        
        preferences = []
        
        for i in range(activation.shape[0]):
            # Get SS assignments
            ss_probs = structures[i] if structures.dim() == 3 else structures
            
            # Compute activation by SS type
            ss_activation = {}
            
            for ss_type in range(ss_probs.shape[-1]):
                ss_mask = ss_probs[:, ss_type] > 0.5
                
                if ss_mask.sum() > 0:
                    ss_activation[ss_type] = activation[i][ss_mask].mean().item()
                    
            if len(ss_activation) > 1:
                # Check for strong preferences
                values = list(ss_activation.values())
                max_act = max(values)
                mean_act = np.mean(values)
                
                if max_act > mean_act * 1.5:
                    preferences.append(max_act / (mean_act + 1e-10))
                    
        return min(1.0, np.mean(preferences) - 1.0) if preferences else 0.0
        
    def _check_structural_motifs(
        self,
        activation: torch.Tensor,
        sequences: List[str]
    ) -> float:
        """Check for known structural motifs."""
        structural_motifs = {
            'helix_cap': 'N[TS]',
            'beta_turn': 'PG',
            'proline_kink': 'P',
            'glycine_hinge': 'G'
        }
        
        motif_scores = []
        
        for i, seq in enumerate(sequences):
            for motif_name, pattern in structural_motifs.items():
                for match in re.finditer(pattern, seq):
                    pos = match.start()
                    
                    # Check activation at motif position
                    if activation[i, pos] > activation[i].mean() + activation[i].std():
                        motif_scores.append(1.0)
                        
        return min(1.0, len(motif_scores) / (len(sequences) * len(structural_motifs)))


class EvolutionaryRelevanceScorer:
    """Scores features based on evolutionary relevance."""
    
    def __init__(self, conservation_data: Dict):
        self.conservation_data = conservation_data
        
    def score_feature(
        self,
        feature_activation: torch.Tensor,
        sequences: List[str],
        evolution_data: Optional[Dict] = None
    ) -> Tuple[float, Optional[BiologicalAnnotation]]:
        """Score evolutionary relevance of a feature."""
        score = 0.0
        evidence = []
        
        # Check correlation with conservation
        if evolution_data and 'conservation' in evolution_data:
            cons_corr = self._compute_conservation_correlation(
                feature_activation, evolution_data['conservation']
            )
            if cons_corr > 0.5:
                score += cons_corr * 0.5
                evidence.append(f"Conservation correlation: {cons_corr:.3f}")
                
        # Check for coevolution patterns
        if evolution_data and 'coevolution' in evolution_data:
            coevo_score = self._check_coevolution_patterns(
                feature_activation, evolution_data['coevolution']
            )
            if coevo_score > 0:
                score += coevo_score * 0.3
                evidence.append("Coevolution pattern detected")
                
        # Check for phylogenetic specificity
        phylo_score = self._check_phylogenetic_patterns(
            feature_activation, sequences
        )
        if phylo_score > 0:
            score += phylo_score * 0.2
            evidence.append("Phylogenetic specificity")
            
        annotation = None
        if score > 0.3 and evidence:
            annotation = BiologicalAnnotation(
                feature_id=0,
                annotation_type='evolution',
                annotation='; '.join(evidence),
                confidence=score,
                evidence=evidence
            )
            
        return score, annotation
        
    def _compute_conservation_correlation(
        self,
        activation: torch.Tensor,
        conservation: torch.Tensor
    ) -> float:
        """Compute correlation with conservation scores."""
        correlations = []
        
        for i in range(activation.shape[0]):
            if conservation[i].sum() > 0:
                corr = torch.corrcoef(
                    torch.stack([activation[i], conservation[i]])
                )[0, 1]
                
                if not torch.isnan(corr):
                    correlations.append(abs(corr.item()))
                    
        return max(correlations) if correlations else 0.0
        
    def _check_coevolution_patterns(
        self,
        activation: torch.Tensor,
        coevolution: torch.Tensor
    ) -> float:
        """Check for coevolution patterns."""
        # Find positions with high activation
        high_activation_mask = activation > activation.mean() + activation.std()
        
        scores = []
        
        for i in range(activation.shape[0]):
            high_positions = torch.where(high_activation_mask[i])[0]
            
            if len(high_positions) > 1:
                # Check if high activation positions coevolve
                coevo_scores = []
                
                for j, pos1 in enumerate(high_positions):
                    for pos2 in high_positions[j+1:]:
                        if pos1 < coevolution.shape[0] and pos2 < coevolution.shape[1]:
                            coevo_scores.append(coevolution[pos1, pos2].item())
                            
                if coevo_scores:
                    avg_coevo = np.mean(coevo_scores)
                    
                    # Compare to background
                    background_coevo = coevolution.mean().item()
                    
                    if avg_coevo > background_coevo * 1.5:
                        scores.append(avg_coevo / (background_coevo + 1e-10))
                        
        return min(1.0, np.mean(scores) - 1.0) if scores else 0.0
        
    def _check_phylogenetic_patterns(
        self,
        activation: torch.Tensor,
        sequences: List[str]
    ) -> float:
        """Check for phylogenetic-specific patterns."""
        # Simplified phylogenetic analysis
        # Would use actual phylogenetic tree in practice
        
        # Group sequences by similarity
        sequence_groups = self._cluster_sequences(sequences)
        
        if len(sequence_groups) < 2:
            return 0.0
            
        # Check if activation patterns differ between groups
        group_activations = []
        
        for group_indices in sequence_groups:
            if len(group_indices) > 0:
                group_act = activation[group_indices].mean(dim=0)
                group_activations.append(group_act)
                
        if len(group_activations) > 1:
            # Compute variance between groups
            group_tensor = torch.stack(group_activations)
            between_group_var = group_tensor.var(dim=0).mean()
            
            # Compute variance within groups
            within_group_var = activation.var(dim=0).mean()
            
            if within_group_var > 0:
                f_statistic = between_group_var / within_group_var
                return min(1.0, f_statistic / 10.0)
                
        return 0.0
        
    def _cluster_sequences(self, sequences: List[str]) -> List[List[int]]:
        """Simple sequence clustering by similarity."""
        if len(sequences) < 2:
            return [list(range(len(sequences)))]
            
        # Compute pairwise similarities
        similarities = np.zeros((len(sequences), len(sequences)))
        
        for i in range(len(sequences)):
            for j in range(i+1, len(sequences)):
                # Simple identity-based similarity
                sim = sum(a == b for a, b in zip(sequences[i], sequences[j]))
                sim = sim / max(len(sequences[i]), len(sequences[j]))
                similarities[i, j] = similarities[j, i] = sim
                
        # Simple clustering: sequences > 80% similar are in same group
        groups = []
        assigned = set()
        
        for i in range(len(sequences)):
            if i not in assigned:
                group = [i]
                assigned.add(i)
                
                for j in range(i+1, len(sequences)):
                    if j not in assigned and similarities[i, j] > 0.8:
                        group.append(j)
                        assigned.add(j)
                        
                groups.append(group)
                
        return groups


class ClinicalRelevanceScorer:
    """Scores features based on clinical/disease relevance."""
    
    def __init__(self, disease_variants: Dict):
        self.disease_variants = disease_variants
        
    def score_feature(
        self,
        feature_activation: torch.Tensor,
        sequences: List[str]
    ) -> Tuple[float, Optional[BiologicalAnnotation]]:
        """Score clinical relevance of a feature."""
        score = 0.0
        evidence = []
        
        # Check for disease variant enrichment
        variant_score = self._check_disease_variant_enrichment(
            feature_activation, sequences
        )
        if variant_score > 0:
            score += variant_score * 0.5
            evidence.append("Disease variant enrichment")
            
        # Check for PTM site patterns (often disease-relevant)
        ptm_score = self._check_ptm_patterns(
            feature_activation, sequences
        )
        if ptm_score > 0:
            score += ptm_score * 0.3
            evidence.append("PTM site pattern")
            
        # Check for intrinsically disordered regions (disease-associated)
        disorder_score = self._check_disorder_patterns(
            feature_activation, sequences
        )
        if disorder_score > 0:
            score += disorder_score * 0.2
            evidence.append("Disorder region pattern")
            
        annotation = None
        if score > 0.3 and evidence:
            annotation = BiologicalAnnotation(
                feature_id=0,
                annotation_type='disease',
                annotation='; '.join(evidence),
                confidence=score,
                evidence=evidence
            )
            
        return score, annotation
        
    def _check_disease_variant_enrichment(
        self,
        activation: torch.Tensor,
        sequences: List[str]
    ) -> float:
        """Check enrichment at disease variant positions."""
        if 'pathogenic' not in self.disease_variants:
            return 0.0
            
        pathogenic_variants = self.disease_variants['pathogenic']
        
        enrichment_scores = []
        
        for i, seq in enumerate(sequences):
            variant_positions = []
            
            # Find variant positions in this sequence
            for variant in pathogenic_variants:
                pos = variant['pos']
                
                if pos < len(seq) and seq[pos] == variant['wt']:
                    variant_positions.append(pos)
                    
            if variant_positions:
                # Check activation at variant positions
                variant_activation = activation[i, variant_positions].mean()
                background_activation = activation[i].mean()
                
                if background_activation > 0:
                    enrichment = variant_activation / background_activation
                    
                    if enrichment > 1.5:
                        enrichment_scores.append(enrichment.item())
                        
        if enrichment_scores:
            # Statistical test for enrichment
            # Simplified - would use proper statistics in practice
            avg_enrichment = np.mean(enrichment_scores)
            return min(1.0, (avg_enrichment - 1.0) / 2.0)
            
        return 0.0
        
    def _check_ptm_patterns(
        self,
        activation: torch.Tensor,
        sequences: List[str]
    ) -> float:
        """Check for post-translational modification patterns."""
        ptm_patterns = {
            'phosphorylation': '[ST][RK].|[ST]P',
            'ubiquitination': 'K',
            'methylation': '[RK]',
            'acetylation': 'K',
            'sumoylation': '[VI]K.E'
        }
        
        ptm_scores = []
        
        for i, seq in enumerate(sequences):
            for ptm_type, pattern in ptm_patterns.items():
                for match in re.finditer(pattern, seq):
                    pos = match.start()
                    
                    # Check if PTM site has high activation
                    if activation[i, pos] > activation[i].mean() + 0.5 * activation[i].std():
                        ptm_scores.append(1.0)
                        
        if ptm_scores:
            return min(1.0, len(ptm_scores) / (len(sequences) * 3))
            
        return 0.0
        
    def _check_disorder_patterns(
        self,
        activation: torch.Tensor,
        sequences: List[str]
    ) -> float:
        """Check for intrinsically disordered region patterns."""
        # Disorder-promoting residues
        disorder_residues = {'P', 'E', 'S', 'K', 'R', 'Q', 'A'}
        
        disorder_scores = []
        
        for i, seq in enumerate(sequences):
            # Find regions enriched in disorder-promoting residues
            window_size = 20
            
            for j in range(len(seq) - window_size):
                window = seq[j:j+window_size]
                disorder_content = sum(1 for aa in window if aa in disorder_residues) / window_size
                
                if disorder_content > 0.6:
                    # Check if this region has distinct activation
                    window_activation = activation[i, j:j+window_size].mean()
                    
                    if window_activation > activation[i].mean():
                        disorder_scores.append(1.0)
                        
        if disorder_scores:
            return min(1.0, len(disorder_scores) / (len(sequences) * 5))
            
        return 0.0