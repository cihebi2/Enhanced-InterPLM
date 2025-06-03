# 二级结构映射 
# enhanced_interplm/hierarchical_mapping/secondary_structure.py

"""
Advanced secondary structure mapping and analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import pandas as pd
from scipy.signal import find_peaks
from sklearn.metrics import precision_recall_fscore_support


@dataclass 
class SecondaryStructureElement:
    """Container for secondary structure element."""
    type: str  # 'helix', 'sheet', 'turn', 'coil'
    start: int
    end: int
    confidence: float
    subtype: Optional[str] = None  # e.g., 'alpha', '310', 'pi' for helices


class AdvancedSSMapper(nn.Module):
    """
    Advanced secondary structure prediction with structural constraints.
    """
    
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        use_structural_constraints: bool = True
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.use_structural_constraints = use_structural_constraints
        
        # Multi-scale convolutional layers for different SS patterns
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(feature_dim, hidden_dim, kernel_size=3, padding=1),  # Local
            nn.Conv1d(feature_dim, hidden_dim, kernel_size=7, padding=3),  # Medium
            nn.Conv1d(feature_dim, hidden_dim, kernel_size=15, padding=7)  # Long-range
        ])
        
        # Bidirectional LSTM for sequence context
        self.lstm = nn.LSTM(
            input_size=hidden_dim * 3,  # Concatenated conv outputs
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        # Attention mechanism for long-range interactions
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # SS type classifier
        self.ss_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 8)  # 8-state DSSP
        )
        
        # Helix subtype classifier
        self.helix_subtype = nn.Linear(hidden_dim * 2, 3)  # alpha, 3-10, pi
        
        # Beta sheet orientation classifier
        self.sheet_orientation = nn.Linear(hidden_dim * 2, 2)  # parallel, antiparallel
        
        # Structural constraint module
        if use_structural_constraints:
            self.structural_constraints = StructuralConstraintModule(hidden_dim * 2)
            
    def forward(
        self,
        features: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
        return_details: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Predict secondary structure with optional structural constraints.
        
        Args:
            features: [batch_size, seq_len, feature_dim]
            coords: Optional 3D coordinates [batch_size, seq_len, 3]
            return_details: Return detailed predictions
            
        Returns:
            Dictionary with SS predictions
        """
        batch_size, seq_len, _ = features.shape
        
        # Multi-scale convolutions
        features_t = features.transpose(1, 2)  # [batch, feature_dim, seq_len]
        conv_outputs = []
        
        for conv in self.conv_layers:
            conv_out = F.relu(conv(features_t))
            conv_outputs.append(conv_out)
            
        # Concatenate multi-scale features
        multi_scale = torch.cat(conv_outputs, dim=1)  # [batch, hidden*3, seq_len]
        multi_scale = multi_scale.transpose(1, 2)  # [batch, seq_len, hidden*3]
        
        # LSTM processing
        lstm_out, _ = self.lstm(multi_scale)
        
        # Self-attention for long-range dependencies
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Combine LSTM and attention
        combined = lstm_out + attn_out
        
        # Apply structural constraints if available
        if self.use_structural_constraints and coords is not None:
            structural_features = self.structural_constraints(combined, coords)
            combined = combined + structural_features
            
        # SS classification
        ss_logits = self.ss_classifier(combined)
        ss_probs = F.softmax(ss_logits, dim=-1)
        
        results = {
            'ss_logits': ss_logits,
            'ss_probs': ss_probs,
            'ss_pred': ss_probs.argmax(dim=-1)
        }
        
        if return_details:
            # Helix subtype prediction for helix positions
            helix_mask = ss_probs[:, :, 0] > 0.5  # Alpha helix positions
            if helix_mask.any():
                helix_features = combined[helix_mask]
                helix_subtype_logits = self.helix_subtype(helix_features)
                results['helix_subtypes'] = {
                    'mask': helix_mask,
                    'logits': helix_subtype_logits,
                    'probs': F.softmax(helix_subtype_logits, dim=-1)
                }
                
            # Sheet orientation for sheet positions
            sheet_mask = ss_probs[:, :, 2] > 0.5  # Beta sheet positions
            if sheet_mask.any():
                sheet_features = combined[sheet_mask]
                sheet_orient_logits = self.sheet_orientation(sheet_features)
                results['sheet_orientation'] = {
                    'mask': sheet_mask,
                    'logits': sheet_orient_logits,
                    'probs': F.softmax(sheet_orient_logits, dim=-1)
                }
                
            results['attention_weights'] = attn_weights
            
        return results


class StructuralConstraintModule(nn.Module):
    """
    Apply structural constraints to SS prediction.
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Distance-based constraints
        self.distance_encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 64)
        )
        
        # Angle-based constraints
        self.angle_encoder = nn.Sequential(
            nn.Linear(2, 32),  # phi, psi
            nn.ReLU(),
            nn.Linear(32, 64)
        )
        
        # Combine structural features
        self.structural_fusion = nn.Sequential(
            nn.Linear(128 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(
        self,
        features: torch.Tensor,
        coords: torch.Tensor
    ) -> torch.Tensor:
        """Apply structural constraints."""
        batch_size, seq_len, hidden_dim = features.shape
        
        # Compute pairwise distances
        distances = torch.cdist(coords, coords)
        
        # Local distance features (i to i+1, i+2, i+3, i+4)
        local_distances = []
        for offset in range(1, 5):
            if offset < seq_len:
                diag_distances = torch.diagonal(distances, offset=offset, dim1=-2, dim2=-1)
                local_distances.append(diag_distances)
                
        # Pad and stack
        local_dist_features = []
        for i, dist in enumerate(local_distances):
            # Pad to match seq_len
            pad_size = i + 1
            padded = F.pad(dist, (0, pad_size), value=0)
            encoded = self.distance_encoder(padded.unsqueeze(-1))
            local_dist_features.append(encoded)
            
        # Average distance features
        if local_dist_features:
            distance_features = torch.stack(local_dist_features).mean(dim=0)
        else:
            distance_features = torch.zeros(batch_size, seq_len, 64).to(coords.device)
            
        # Compute backbone angles
        angles = self._compute_backbone_angles(coords)
        angle_features = self.angle_encoder(angles)
        
        # Combine all structural features
        structural_features = torch.cat([
            features,
            distance_features,
            angle_features
        ], dim=-1)
        
        # Final fusion
        constrained_features = self.structural_fusion(structural_features)
        
        return constrained_features
        
    def _compute_backbone_angles(self, coords: torch.Tensor) -> torch.Tensor:
        """Compute approximate backbone dihedral angles."""
        batch_size, seq_len, _ = coords.shape
        
        # Initialize angles
        angles = torch.zeros(batch_size, seq_len, 2).to(coords.device)
        
        # Compute pseudo phi/psi angles from CA coordinates
        for i in range(1, seq_len - 1):
            # Vectors
            v1 = coords[:, i] - coords[:, i-1]
            v2 = coords[:, i+1] - coords[:, i]
            
            # Angle between vectors (pseudo-phi)
            cos_angle = F.cosine_similarity(v1, v2, dim=-1)
            angles[:, i, 0] = torch.acos(torch.clamp(cos_angle, -1, 1))
            
            # Dihedral approximation (pseudo-psi)
            if i < seq_len - 2:
                v3 = coords[:, i+2] - coords[:, i+1]
                
                # Normal vectors
                n1 = torch.cross(v1, v2, dim=-1)
                n2 = torch.cross(v2, v3, dim=-1)
                
                # Dihedral angle
                cos_dihedral = F.cosine_similarity(n1, n2, dim=-1)
                angles[:, i, 1] = torch.acos(torch.clamp(cos_dihedral, -1, 1))
                
        return angles


class SSMotifDetector:
    """
    Detect and analyze secondary structure motifs.
    """
    
    def __init__(self):
        # Define common SS motifs
        self.motif_patterns = {
            'helix_cap_n': {
                'pattern': ['C', 'H', 'H', 'H'],
                'description': 'N-terminal helix cap'
            },
            'helix_cap_c': {
                'pattern': ['H', 'H', 'H', 'C'],
                'description': 'C-terminal helix cap'
            },
            'beta_hairpin': {
                'pattern': ['E', 'E', 'T', 'T', 'E', 'E'],
                'description': 'Beta hairpin'
            },
            'beta_bulge': {
                'pattern': ['E', 'E', 'S', 'E', 'E'],
                'description': 'Beta bulge'
            },
            'helix_kink': {
                'pattern': ['H', 'H', 'T', 'H', 'H'],
                'description': 'Helix kink'
            },
            'beta_alpha_beta': {
                'pattern': ['E', 'C', 'H', 'H', 'C', 'E'],
                'description': 'Beta-alpha-beta motif'
            }
        }
        
    def detect_motifs(
        self,
        ss_sequence: Union[str, torch.Tensor],
        confidence_threshold: float = 0.7
    ) -> List[Dict]:
        """
        Detect SS motifs in sequence.
        
        Args:
            ss_sequence: Secondary structure sequence (string or tensor)
            confidence_threshold: Minimum confidence for detection
            
        Returns:
            List of detected motifs
        """
        # Convert to string if tensor
        if isinstance(ss_sequence, torch.Tensor):
            ss_string = self._tensor_to_ss_string(ss_sequence)
        else:
            ss_string = ss_sequence
            
        detected_motifs = []
        
        # Search for each motif pattern
        for motif_name, motif_info in self.motif_patterns.items():
            pattern = motif_info['pattern']
            pattern_len = len(pattern)
            
            # Sliding window search
            for i in range(len(ss_string) - pattern_len + 1):
                window = ss_string[i:i + pattern_len]
                
                # Check if pattern matches (with some flexibility)
                if self._matches_pattern(window, pattern):
                    detected_motifs.append({
                        'type': motif_name,
                        'description': motif_info['description'],
                        'start': i,
                        'end': i + pattern_len,
                        'sequence': window,
                        'confidence': self._compute_match_confidence(window, pattern)
                    })
                    
        # Filter by confidence
        detected_motifs = [m for m in detected_motifs if m['confidence'] >= confidence_threshold]
        
        # Remove overlapping motifs (keep highest confidence)
        detected_motifs = self._remove_overlaps(detected_motifs)
        
        return detected_motifs
        
    def _tensor_to_ss_string(self, ss_tensor: torch.Tensor) -> str:
        """Convert SS prediction tensor to DSSP string."""
        dssp_codes = ['H', 'B', 'E', 'G', 'I', 'T', 'S', '-']
        
        if ss_tensor.dim() == 2:  # Probabilities
            ss_indices = ss_tensor.argmax(dim=-1)
        else:  # Already indices
            ss_indices = ss_tensor
            
        ss_string = ''.join([dssp_codes[idx] for idx in ss_indices])
        return ss_string
        
    def _matches_pattern(self, window: str, pattern: List[str]) -> bool:
        """Check if window matches pattern with flexibility."""
        # Simplified SS mapping for pattern matching
        ss_groups = {
            'H': ['H', 'G', 'I'],  # All helices
            'E': ['E', 'B'],        # All sheets
            'T': ['T', 'S'],        # Turns and bends
            'C': ['-', 'T', 'S']    # Coil/loop
        }
        
        for w_char, p_char in zip(window, pattern):
            if p_char in ss_groups:
                if w_char not in ss_groups[p_char]:
                    return False
            elif w_char != p_char:
                return False
                
        return True
        
    def _compute_match_confidence(self, window: str, pattern: List[str]) -> float:
        """Compute confidence of pattern match."""
        exact_matches = sum(1 for w, p in zip(window, pattern) if w == p)
        return exact_matches / len(pattern)
        
    def _remove_overlaps(self, motifs: List[Dict]) -> List[Dict]:
        """Remove overlapping motifs, keeping highest confidence."""
        if not motifs:
            return motifs
            
        # Sort by confidence (descending)
        sorted_motifs = sorted(motifs, key=lambda x: x['confidence'], reverse=True)
        
        kept_motifs = []
        used_positions = set()
        
        for motif in sorted_motifs:
            positions = set(range(motif['start'], motif['end']))
            
            # Check for overlap
            if not positions & used_positions:
                kept_motifs.append(motif)
                used_positions.update(positions)
                
        # Sort by position
        kept_motifs.sort(key=lambda x: x['start'])
        
        return kept_motifs


class SSQualityAssessor:
    """
    Assess quality of secondary structure predictions.
    """
    
    def __init__(self):
        # Physical constraints for SS elements
        self.min_helix_length = 4
        self.min_sheet_length = 3
        self.max_loop_in_helix = 1
        
    def assess_prediction(
        self,
        predicted_ss: torch.Tensor,
        confidence_scores: Optional[torch.Tensor] = None,
        coordinates: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Assess quality of SS predictions.
        
        Args:
            predicted_ss: Predicted SS (indices or one-hot)
            confidence_scores: Optional confidence scores
            coordinates: Optional 3D coordinates
            
        Returns:
            Dictionary of quality metrics
        """
        metrics = {}
        
        # Convert to indices if needed
        if predicted_ss.dim() == 2:
            ss_indices = predicted_ss.argmax(dim=-1)
        else:
            ss_indices = predicted_ss
            
        # Basic statistics
        metrics['helix_fraction'] = (ss_indices == 0).float().mean().item()
        metrics['sheet_fraction'] = (ss_indices == 2).float().mean().item()
        metrics['coil_fraction'] = (ss_indices == 7).float().mean().item()
        
        # Check physical constraints
        metrics['valid_helix_fraction'] = self._check_helix_validity(ss_indices)
        metrics['valid_sheet_fraction'] = self._check_sheet_validity(ss_indices)
        
        # Confidence analysis
        if confidence_scores is not None:
            metrics['avg_confidence'] = confidence_scores.max(dim=-1)[0].mean().item()
            metrics['low_confidence_fraction'] = (confidence_scores.max(dim=-1)[0] < 0.5).float().mean().item()
            
        # Structural consistency
        if coordinates is not None:
            metrics['structural_consistency'] = self._check_structural_consistency(
                ss_indices, coordinates
            )
            
        # Complexity metrics
        metrics['ss_complexity'] = self._compute_ss_complexity(ss_indices)
        metrics['transition_rate'] = self._compute_transition_rate(ss_indices)
        
        return metrics
        
    def _check_helix_validity(self, ss_indices: torch.Tensor) -> float:
        """Check fraction of helices meeting minimum length."""
        helix_mask = (ss_indices == 0) | (ss_indices == 3) | (ss_indices == 4)
        
        valid_helices = 0
        total_helices = 0
        
        # Find continuous helix regions
        in_helix = False
        helix_start = 0
        
        for i in range(len(ss_indices)):
            if helix_mask[i] and not in_helix:
                helix_start = i
                in_helix = True
            elif not helix_mask[i] and in_helix:
                helix_length = i - helix_start
                total_helices += 1
                if helix_length >= self.min_helix_length:
                    valid_helices += 1
                in_helix = False
                
        # Handle helix at end
        if in_helix:
            helix_length = len(ss_indices) - helix_start
            total_helices += 1
            if helix_length >= self.min_helix_length:
                valid_helices += 1
                
        return valid_helices / max(total_helices, 1)
        
    def _check_sheet_validity(self, ss_indices: torch.Tensor) -> float:
        """Check fraction of sheets meeting minimum length."""
        sheet_mask = (ss_indices == 2) | (ss_indices == 1)
        
        valid_sheets = 0
        total_sheets = 0
        
        in_sheet = False
        sheet_start = 0
        
        for i in range(len(ss_indices)):
            if sheet_mask[i] and not in_sheet:
                sheet_start = i
                in_sheet = True
            elif not sheet_mask[i] and in_sheet:
                sheet_length = i - sheet_start
                total_sheets += 1
                if sheet_length >= self.min_sheet_length:
                    valid_sheets += 1
                in_sheet = False
                
        if in_sheet:
            sheet_length = len(ss_indices) - sheet_start
            total_sheets += 1
            if sheet_length >= self.min_sheet_length:
                valid_sheets += 1
                
        return valid_sheets / max(total_sheets, 1)
        
    def _check_structural_consistency(
        self,
        ss_indices: torch.Tensor,
        coordinates: torch.Tensor
    ) -> float:
        """Check if SS assignments are consistent with 3D structure."""
        consistency_scores = []
        
        # Check helix consistency (should have regular i to i+4 distances)
        helix_positions = torch.where((ss_indices == 0) | (ss_indices == 3))[0]
        
        for i in range(len(helix_positions) - 4):
            pos_i = helix_positions[i]
            pos_i4 = pos_i + 4
            
            if pos_i4 < len(coordinates):
                # Ideal i to i+4 distance in alpha helix is ~5.4 Å
                distance = torch.norm(coordinates[pos_i] - coordinates[pos_i4])
                consistency = torch.exp(-((distance - 5.4) / 2.0) ** 2)
                consistency_scores.append(consistency.item())
                
        # Check sheet consistency (extended conformation)
        sheet_positions = torch.where(ss_indices == 2)[0]
        
        for i in range(len(sheet_positions) - 2):
            pos_i = sheet_positions[i]
            pos_i2 = pos_i + 2
            
            if pos_i2 < len(coordinates):
                # Extended conformation has larger i to i+2 distance
                distance = torch.norm(coordinates[pos_i] - coordinates[pos_i2])
                consistency = 1.0 if distance > 6.0 else distance / 6.0
                consistency_scores.append(consistency)
                
        return np.mean(consistency_scores) if consistency_scores else 0.0
        
    def _compute_ss_complexity(self, ss_indices: torch.Tensor) -> float:
        """Compute complexity of SS assignment."""
        # Count unique SS types
        unique_ss = len(torch.unique(ss_indices))
        
        # Count transitions
        transitions = torch.sum(ss_indices[1:] != ss_indices[:-1]).item()
        
        # Normalized complexity
        complexity = (unique_ss / 8.0) * (transitions / len(ss_indices))
        
        return complexity
        
    def _compute_transition_rate(self, ss_indices: torch.Tensor) -> float:
        """Compute rate of SS transitions."""
        transitions = torch.sum(ss_indices[1:] != ss_indices[:-1]).item()
        return transitions / (len(ss_indices) - 1)