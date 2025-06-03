# 氨基酸属性映射 
# enhanced_interplm/hierarchical_mapping/aa_property_mapper.py

"""
Advanced amino acid property mapping and analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import pandas as pd
from scipy.stats import pearsonr, spearmanr


@dataclass
class AminoAcidProperties:
    """Comprehensive amino acid properties."""
    # Hydrophobicity scales
    hydrophobicity_kd = {  # Kyte-Doolittle
        'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
        'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
        'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
        'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
    }
    
    # Charge at pH 7
    charge = {
        'A': 0, 'C': 0, 'D': -1, 'E': -1, 'F': 0,
        'G': 0, 'H': 0.1, 'I': 0, 'K': 1, 'L': 0,
        'M': 0, 'N': 0, 'P': 0, 'Q': 0, 'R': 1,
        'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0
    }
    
    # Molecular weight
    molecular_weight = {
        'A': 89.1, 'C': 121.2, 'D': 133.1, 'E': 147.1, 'F': 165.2,
        'G': 75.1, 'H': 155.2, 'I': 131.2, 'K': 146.2, 'L': 131.2,
        'M': 149.2, 'N': 132.1, 'P': 115.1, 'Q': 146.2, 'R': 174.2,
        'S': 105.1, 'T': 119.1, 'V': 117.1, 'W': 204.2, 'Y': 181.2
    }
    
    # Volume
    volume = {
        'A': 88.6, 'C': 108.5, 'D': 111.1, 'E': 138.4, 'F': 189.9,
        'G': 60.1, 'H': 153.2, 'I': 166.7, 'K': 168.6, 'L': 166.7,
        'M': 162.9, 'N': 114.1, 'P': 112.7, 'Q': 143.8, 'R': 173.4,
        'S': 89.0, 'T': 116.1, 'V': 140.0, 'W': 227.8, 'Y': 193.6
    }
    
    # Secondary structure propensity
    helix_propensity = {
        'A': 1.42, 'C': 0.70, 'D': 1.01, 'E': 1.51, 'F': 1.13,
        'G': 0.57, 'H': 1.00, 'I': 1.08, 'K': 1.16, 'L': 1.21,
        'M': 1.45, 'N': 0.67, 'P': 0.57, 'Q': 1.11, 'R': 0.98,
        'S': 0.77, 'T': 0.83, 'V': 1.06, 'W': 1.08, 'Y': 0.69
    }
    
    sheet_propensity = {
        'A': 0.83, 'C': 1.19, 'D': 0.54, 'E': 0.37, 'F': 1.38,
        'G': 0.75, 'H': 0.87, 'I': 1.60, 'K': 0.74, 'L': 1.30,
        'M': 1.05, 'N': 0.89, 'P': 0.55, 'Q': 1.10, 'R': 0.93,
        'S': 0.75, 'T': 1.19, 'V': 1.70, 'W': 1.37, 'Y': 1.47
    }
    
    # Flexibility
    flexibility = {
        'A': 0.357, 'C': 0.346, 'D': 0.511, 'E': 0.497, 'F': 0.314,
        'G': 0.544, 'H': 0.323, 'I': 0.462, 'K': 0.466, 'L': 0.365,
        'M': 0.295, 'N': 0.463, 'P': 0.509, 'Q': 0.493, 'R': 0.529,
        'S': 0.507, 'T': 0.444, 'V': 0.386, 'W': 0.305, 'Y': 0.420
    }


class DetailedAAPropertyMapper(nn.Module):
    """
    Detailed amino acid property mapping with multiple scales and properties.
    """
    
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 128,
        num_properties: int = 7,  # Number of property types
        use_evolutionary_info: bool = True
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_properties = num_properties
        self.use_evolutionary_info = use_evolutionary_info
        
        self.aa_props = AminoAcidProperties()
        
        # Property-specific encoders
        self.property_encoders = nn.ModuleDict({
            'hydrophobicity': self._build_property_encoder(),
            'charge': self._build_property_encoder(),
            'size': self._build_property_encoder(),
            'flexibility': self._build_property_encoder(),
            'helix_propensity': self._build_property_encoder(),
            'sheet_propensity': self._build_property_encoder(),
            'conservation': self._build_property_encoder()
        })
        
        # Cross-property attention
        self.cross_property_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True
        )
        
        # Property interaction network
        self.property_interaction = nn.Sequential(
            nn.Linear(hidden_dim * num_properties, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Output heads for different properties
        self.output_heads = nn.ModuleDict({
            'hydrophobicity': nn.Linear(hidden_dim, 1),
            'charge': nn.Linear(hidden_dim, 3),  # +, -, neutral
            'size': nn.Linear(hidden_dim, 3),  # small, medium, large
            'flexibility': nn.Linear(hidden_dim, 1),
            'secondary_structure': nn.Linear(hidden_dim, 3),  # helix, sheet, coil
            'functional_class': nn.Linear(hidden_dim, 6),  # hydrophobic, polar, charged, etc.
            'conservation': nn.Linear(hidden_dim, 1)
        })
        
    def _build_property_encoder(self) -> nn.Module:
        """Build encoder for a specific property."""
        return nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )
        
    def forward(
        self,
        features: torch.Tensor,
        sequences: List[str],
        evolutionary_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Map features to detailed amino acid properties.
        
        Args:
            features: [batch_size, seq_len, feature_dim]
            sequences: List of amino acid sequences
            evolutionary_features: Optional evolutionary information
            
        Returns:
            Dictionary of predicted properties
        """
        batch_size, seq_len, _ = features.shape
        
        # Encode each property type
        property_encodings = {}
        
        for prop_name, encoder in self.property_encoders.items():
            # Reshape for batch norm
            features_flat = features.view(-1, self.feature_dim)
            encoded = encoder(features_flat)
            encoded = encoded.view(batch_size, seq_len, self.hidden_dim)
            property_encodings[prop_name] = encoded
            
        # Stack property encodings
        stacked_encodings = torch.stack(list(property_encodings.values()), dim=2)
        # Shape: [batch_size, seq_len, num_properties, hidden_dim]
        
        # Cross-property attention
        batch_seq = batch_size * seq_len
        stacked_flat = stacked_encodings.view(batch_seq, self.num_properties, self.hidden_dim)
        
        attended, _ = self.cross_property_attention(
            stacked_flat, stacked_flat, stacked_flat
        )
        attended = attended.view(batch_size, seq_len, self.num_properties, self.hidden_dim)
        
        # Combine all property information
        combined = attended.reshape(batch_size, seq_len, -1)
        integrated = self.property_interaction(combined)
        
        # Generate predictions for each property
        predictions = {}
        
        # Basic properties
        predictions['hydrophobicity'] = torch.sigmoid(
            self.output_heads['hydrophobicity'](integrated)
        )
        
        predictions['charge'] = F.softmax(
            self.output_heads['charge'](integrated), dim=-1
        )
        
        predictions['size'] = F.softmax(
            self.output_heads['size'](integrated), dim=-1
        )
        
        predictions['flexibility'] = torch.sigmoid(
            self.output_heads['flexibility'](integrated)
        )
        
        predictions['secondary_structure'] = F.softmax(
            self.output_heads['secondary_structure'](integrated), dim=-1
        )
        
        predictions['functional_class'] = F.softmax(
            self.output_heads['functional_class'](integrated), dim=-1
        )
        
        # Conservation (if evolutionary features provided)
        if evolutionary_features is not None and self.use_evolutionary_info:
            # Combine with evolutionary information
            evo_combined = torch.cat([integrated, evolutionary_features], dim=-1)
            predictions['conservation'] = torch.sigmoid(
                self.output_heads['conservation'](evo_combined[:, :, :self.hidden_dim])
            )
        else:
            predictions['conservation'] = torch.sigmoid(
                self.output_heads['conservation'](integrated)
            )
            
        # Add ground truth comparisons
        predictions['ground_truth'] = self._compute_ground_truth_properties(sequences)
        
        # Add property correlations
        predictions['property_correlations'] = self._compute_property_correlations(
            predictions, sequences
        )
        
        return predictions
        
    def _compute_ground_truth_properties(
        self,
        sequences: List[str]
    ) -> Dict[str, torch.Tensor]:
        """Compute ground truth properties from sequences."""
        batch_size = len(sequences)
        max_len = max(len(seq) for seq in sequences)
        
        gt_properties = {
            'hydrophobicity': torch.zeros(batch_size, max_len),
            'charge': torch.zeros(batch_size, max_len),
            'size': torch.zeros(batch_size, max_len),
            'flexibility': torch.zeros(batch_size, max_len),
            'helix_propensity': torch.zeros(batch_size, max_len),
            'sheet_propensity': torch.zeros(batch_size, max_len)
        }
        
        for i, seq in enumerate(sequences):
            for j, aa in enumerate(seq):
                if aa in self.aa_props.hydrophobicity_kd:
                    # Normalize properties to [0, 1]
                    gt_properties['hydrophobicity'][i, j] = (
                        self.aa_props.hydrophobicity_kd[aa] + 5
                    ) / 10
                    
                    gt_properties['charge'][i, j] = self.aa_props.charge.get(aa, 0)
                    
                    gt_properties['size'][i, j] = self.aa_props.volume.get(aa, 100) / 250
                    
                    gt_properties['flexibility'][i, j] = self.aa_props.flexibility.get(aa, 0.4)
                    
                    gt_properties['helix_propensity'][i, j] = self.aa_props.helix_propensity.get(aa, 1.0)
                    
                    gt_properties['sheet_propensity'][i, j] = self.aa_props.sheet_propensity.get(aa, 1.0)
                    
        return gt_properties
        
    def _compute_property_correlations(
        self,
        predictions: Dict[str, torch.Tensor],
        sequences: List[str]
    ) -> Dict[str, float]:
        """Compute correlations between predicted and true properties."""
        correlations = {}
        gt_properties = predictions['ground_truth']
        
        for prop_name in ['hydrophobicity', 'flexibility']:
            if prop_name in predictions and prop_name in gt_properties:
                pred_flat = predictions[prop_name].flatten()
                gt_flat = gt_properties[prop_name].flatten()
                
                # Only compute on valid positions
                mask = gt_flat != 0
                if mask.sum() > 10:
                    corr, _ = pearsonr(
                        pred_flat[mask].detach().cpu().numpy(),
                        gt_flat[mask].cpu().numpy()
                    )
                    correlations[f'{prop_name}_correlation'] = corr
                    
        return correlations


class AAPropertyAnalyzer:
    """
    Analyzer for amino acid property patterns and distributions.
    """
    
    def __init__(self):
        self.aa_props = AminoAcidProperties()
        
    def analyze_property_patterns(
        self,
        sequences: List[str],
        predictions: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """
        Analyze patterns in amino acid properties.
        """
        analysis = {}
        
        # Hydrophobic cluster analysis
        analysis['hydrophobic_clusters'] = self._analyze_hydrophobic_clusters(
            sequences, predictions.get('hydrophobicity')
        )
        
        # Charge distribution analysis
        analysis['charge_distribution'] = self._analyze_charge_distribution(
            sequences, predictions.get('charge')
        )
        
        # Secondary structure preferences
        analysis['structure_preferences'] = self._analyze_structure_preferences(
            sequences, predictions.get('secondary_structure')
        )
        
        # Functional motif detection
        analysis['functional_motifs'] = self._detect_functional_motifs(
            sequences, predictions
        )
        
        return analysis
        
    def _analyze_hydrophobic_clusters(
        self,
        sequences: List[str],
        hydrophobicity: Optional[torch.Tensor]
    ) -> Dict[str, Any]:
        """Analyze hydrophobic clustering patterns."""
        clusters = {
            'cluster_sizes': [],
            'cluster_positions': [],
            'cluster_scores': []
        }
        
        if hydrophobicity is None:
            return clusters
            
        for i, seq in enumerate(sequences):
            seq_hydro = hydrophobicity[i, :len(seq)]
            
            # Find hydrophobic regions (high hydrophobicity)
            hydro_mask = seq_hydro > 0.6
            
            # Find clusters
            in_cluster = False
            cluster_start = 0
            
            for j in range(len(seq)):
                if hydro_mask[j] and not in_cluster:
                    cluster_start = j
                    in_cluster = True
                elif not hydro_mask[j] and in_cluster:
                    cluster_size = j - cluster_start
                    if cluster_size >= 3:  # Minimum cluster size
                        clusters['cluster_sizes'].append(cluster_size)
                        clusters['cluster_positions'].append((i, cluster_start, j))
                        clusters['cluster_scores'].append(
                            seq_hydro[cluster_start:j].mean().item()
                        )
                    in_cluster = False
                    
        return clusters
        
    def _analyze_charge_distribution(
        self,
        sequences: List[str],
        charge: Optional[torch.Tensor]
    ) -> Dict[str, Any]:
        """Analyze charge distribution patterns."""
        if charge is None:
            return {}
            
        distribution = {
            'positive_fraction': [],
            'negative_fraction': [],
            'charge_clusters': [],
            'net_charge': []
        }
        
        for i, seq in enumerate(sequences):
            seq_charge = charge[i, :len(seq)]
            
            # Charge fractions
            pos_frac = seq_charge[:, 0].mean().item()  # Positive
            neg_frac = seq_charge[:, 1].mean().item()  # Negative
            
            distribution['positive_fraction'].append(pos_frac)
            distribution['negative_fraction'].append(neg_frac)
            distribution['net_charge'].append(pos_frac - neg_frac)
            
            # Detect charge clusters
            for charge_type in [0, 1]:  # Positive, negative
                charge_mask = seq_charge.argmax(dim=-1) == charge_type
                clusters = self._find_clusters(charge_mask)
                distribution['charge_clusters'].extend([
                    (i, start, end, 'positive' if charge_type == 0 else 'negative')
                    for start, end in clusters
                ])
                
        return distribution
        
    def _analyze_structure_preferences(
        self,
        sequences: List[str],
        structure: Optional[torch.Tensor]
    ) -> Dict[str, Any]:
        """Analyze secondary structure preferences."""
        if structure is None:
            return {}
            
        preferences = {
            'helix_regions': [],
            'sheet_regions': [],
            'structure_transitions': []
        }
        
        for i, seq in enumerate(sequences):
            seq_struct = structure[i, :len(seq)].argmax(dim=-1)
            
            # Find continuous structure regions
            for struct_type, struct_name in [(0, 'helix'), (1, 'sheet')]:
                struct_mask = seq_struct == struct_type
                regions = self._find_clusters(struct_mask)
                
                for start, end in regions:
                    if end - start >= 3:  # Minimum structure length
                        preferences[f'{struct_name}_regions'].append((i, start, end))
                        
            # Find structure transitions
            for j in range(1, len(seq)):
                if seq_struct[j] != seq_struct[j-1]:
                    preferences['structure_transitions'].append(
                        (i, j, seq_struct[j-1].item(), seq_struct[j].item())
                    )
                    
        return preferences
        
    def _detect_functional_motifs(
        self,
        sequences: List[str],
        predictions: Dict[str, torch.Tensor]
    ) -> List[Dict]:
        """Detect functional motifs based on property patterns."""
        motifs = []
        
        # Define property-based motifs
        motif_patterns = {
            'catalytic_triad': {
                'pattern': ['charged', 'small', 'charged'],
                'properties': ['charge', 'size', 'charge']
            },
            'zinc_finger': {
                'pattern': ['C', 'any', 'any', 'C'],
                'sequence_based': True
            },
            'phosphorylation_site': {
                'pattern': ['S/T', 'any', 'charged'],
                'sequence_based': True,
                'properties': ['charge']
            }
        }
        
        # Detect each motif type
        for motif_name, motif_def in motif_patterns.items():
            if motif_def.get('sequence_based', False):
                # Sequence-based detection
                detected = self._detect_sequence_motifs(sequences, motif_def)
            else:
                # Property-based detection
                detected = self._detect_property_motifs(
                    predictions, motif_def, sequences
                )
                
            motifs.extend([{
                'type': motif_name,
                'position': pos,
                'sequence': seq_id,
                'confidence': conf
            } for seq_id, pos, conf in detected])
            
        return motifs
        
    def _find_clusters(self, mask: torch.Tensor) -> List[Tuple[int, int]]:
        """Find continuous True regions in a boolean mask."""
        clusters = []
        in_cluster = False
        start = 0
        
        for i in range(len(mask)):
            if mask[i] and not in_cluster:
                start = i
                in_cluster = True
            elif not mask[i] and in_cluster:
                clusters.append((start, i))
                in_cluster = False
                
        if in_cluster:
            clusters.append((start, len(mask)))
            
        return clusters
        
    def _detect_sequence_motifs(
        self,
        sequences: List[str],
        motif_def: Dict
    ) -> List[Tuple[int, int, float]]:
        """Detect sequence-based motifs."""
        detected = []
        
        # Simplified motif detection
        for i, seq in enumerate(sequences):
            if 'pattern' in motif_def:
                pattern = motif_def['pattern']
                
                # Simple pattern matching
                for j in range(len(seq) - len(pattern) + 1):
                    match = True
                    for k, pat in enumerate(pattern):
                        if pat == 'any':
                            continue
                        elif '/' in pat:  # Multiple options
                            options = pat.split('/')
                            if seq[j + k] not in options:
                                match = False
                                break
                        elif seq[j + k] != pat:
                            match = False
                            break
                            
                    if match:
                        detected.append((i, j, 0.9))  # High confidence for exact match
                        
        return detected
        
    def _detect_property_motifs(
        self,
        predictions: Dict[str, torch.Tensor],
        motif_def: Dict,
        sequences: List[str]
    ) -> List[Tuple[int, int, float]]:
        """Detect property-based motifs."""
        detected = []
        
        # Simplified property motif detection
        pattern = motif_def.get('pattern', [])
        properties = motif_def.get('properties', [])
        
        if not pattern or not properties:
            return detected
            
        for i in range(len(sequences)):
            seq_len = len(sequences[i])
            
            for j in range(seq_len - len(pattern) + 1):
                match_score = 0
                
                for k, (pat, prop) in enumerate(zip(pattern, properties)):
                    if prop in predictions:
                        prop_pred = predictions[prop][i, j + k]
                        
                        # Check if property matches pattern
                        if pat == 'charged':
                            if prop == 'charge':
                                # Check if position is charged (not neutral)
                                if prop_pred[2] < 0.5:  # Not neutral
                                    match_score += 1
                        elif pat == 'small':
                            if prop == 'size':
                                if prop_pred[0] > 0.5:  # Small
                                    match_score += 1
                                    
                confidence = match_score / len(pattern)
                if confidence > 0.7:
                    detected.append((i, j, confidence))
                    
        return detected