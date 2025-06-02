# 疏水性分析 
"""Hydrophobic interaction analysis module."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import DBSCAN


class HydrophobicAnalyzer(nn.Module):
    """
    Analyzes hydrophobic interactions and clusters in protein features.
    """
    
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 128,
        hydrophobicity_scale: str = 'kyte_doolittle'
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # Hydrophobicity scales
        self.hydrophobicity_scales = {
            'kyte_doolittle': {
                'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
                'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
                'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
                'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
            },
            'hopp_woods': {
                'A': -0.5, 'C': -1.0, 'D': 3.0, 'E': 3.0, 'F': -2.5,
                'G': 0.0, 'H': -0.5, 'I': -1.8, 'K': 3.0, 'L': -1.8,
                'M': -1.3, 'N': 0.2, 'P': 0.0, 'Q': 0.2, 'R': 3.0,
                'S': 0.3, 'T': -0.4, 'V': -1.5, 'W': -3.4, 'Y': -2.3
            }
        }
        
        self.scale = self.hydrophobicity_scales[hydrophobicity_scale]
        
        # Neural network for hydrophobic feature extraction
        self.hydrophobic_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()  # Output in [-1, 1]
        )
        
        # Hydrophobic cluster detector
        self.cluster_detector = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(8, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        features: torch.Tensor,
        sequences: List[str],
        structures: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Analyze hydrophobic properties and interactions.
        
        Args:
            features: [batch_size, seq_len, feature_dim]
            sequences: List of amino acid sequences
            structures: Optional 3D coordinates [batch_size, seq_len, 3]
            
        Returns:
            Dictionary containing hydrophobic analysis results
        """
        batch_size, seq_len, _ = features.shape
        
        # Get true hydrophobicity from sequences
        true_hydrophobicity = self._compute_true_hydrophobicity(sequences)
        
        # Predict hydrophobicity from features
        features_flat = features.view(-1, self.feature_dim)
        predicted_hydro = self.hydrophobic_encoder(features_flat)
        predicted_hydro = predicted_hydro.view(batch_size, seq_len, 1)
        
        # Detect hydrophobic clusters
        clusters = self.cluster_detector(predicted_hydro.transpose(1, 2))
        clusters = clusters.transpose(1, 2)
        
        results = {
            'predicted_hydrophobicity': predicted_hydro,
            'true_hydrophobicity': true_hydrophobicity,
            'hydrophobic_clusters': clusters,
            'cluster_locations': self._identify_cluster_locations(clusters)
        }
        
        # If structures provided, analyze 3D hydrophobic cores
        if structures is not None:
            hydrophobic_cores = self._analyze_hydrophobic_cores(
                predicted_hydro,
                structures
            )
            results['hydrophobic_cores'] = hydrophobic_cores
            
        return results
        
    def _compute_true_hydrophobicity(
        self,
        sequences: List[str]
    ) -> torch.Tensor:
        """Compute true hydrophobicity values from sequences."""
        batch_size = len(sequences)
        max_len = max(len(seq) for seq in sequences)
        
        hydro_tensor = torch.zeros(batch_size, max_len, 1)
        
        for i, seq in enumerate(sequences):
            for j, aa in enumerate(seq):
                if aa in self.scale:
                    # Normalize to [-1, 1]
                    hydro_tensor[i, j, 0] = self.scale[aa] / 5.0
                    
        return hydro_tensor.to(next(self.parameters()).device)
        
    def _identify_cluster_locations(
        self,
        clusters: torch.Tensor,
        threshold: float = 0.5
    ) -> List[List[Tuple[int, int]]]:
        """Identify start and end positions of hydrophobic clusters."""
        batch_size, seq_len, _ = clusters.shape
        cluster_locations = []
        
        for b in range(batch_size):
            sequence_clusters = []
            cluster_probs = clusters[b, :, 0].cpu().numpy()
            
            # Find continuous regions above threshold
            in_cluster = False
            start = 0
            
            for i in range(seq_len):
                if cluster_probs[i] > threshold:
                    if not in_cluster:
                        start = i
                        in_cluster = True
                else:
                    if in_cluster:
                        sequence_clusters.append((start, i))
                        in_cluster = False
                        
            # Handle cluster at end
            if in_cluster:
                sequence_clusters.append((start, seq_len))
                
            cluster_locations.append(sequence_clusters)
            
        return cluster_locations
        
    def _analyze_hydrophobic_cores(
        self,
        hydrophobicity: torch.Tensor,
        structures: torch.Tensor,
        distance_threshold: float = 8.0
    ) -> Dict[str, torch.Tensor]:
        """Analyze 3D hydrophobic cores in protein structures."""
        batch_size, seq_len, _ = hydrophobicity.shape
        
        cores = []
        core_scores = []
        
        for b in range(batch_size):
            # Get hydrophobic residues
            hydro_mask = hydrophobicity[b, :, 0] > 0.3
            hydro_indices = torch.where(hydro_mask)[0]
            
            if len(hydro_indices) < 3:
                cores.append(torch.zeros(seq_len))
                core_scores.append(0.0)
                continue
                
            # Get coordinates of hydrophobic residues
            hydro_coords = structures[b, hydro_indices].cpu().numpy()
            
            # Cluster hydrophobic residues in 3D space
            if len(hydro_coords) > 0:
                clustering = DBSCAN(
                    eps=distance_threshold,
                    min_samples=3
                ).fit(hydro_coords)
                
                # Create core mask
                core_mask = torch.zeros(seq_len)
                
                for label in set(clustering.labels_):
                    if label != -1:  # Not noise
                        cluster_indices = hydro_indices[clustering.labels_ == label]
                        core_mask[cluster_indices] = 1
                        
                cores.append(core_mask)
                
                # Score based on compactness
                if len(set(clustering.labels_)) > 0:
                    core_score = len(clustering.labels_[clustering.labels_ != -1]) / len(hydro_indices)
                else:
                    core_score = 0.0
                    
                core_scores.append(core_score)
            else:
                cores.append(torch.zeros(seq_len))
                core_scores.append(0.0)
                
        return {
            'core_masks': torch.stack(cores).to(hydrophobicity.device),
            'core_scores': torch.tensor(core_scores).to(hydrophobicity.device)
        }
        
    def compute_hydrophobic_moment(
        self,
        sequence: str,
        window_size: int = 11,
        angle: float = 100.0
    ) -> torch.Tensor:
        """
        Compute hydrophobic moment for detecting amphipathic structures.
        
        Args:
            sequence: Amino acid sequence
            window_size: Window size for moment calculation
            angle: Angle between residues (100° for alpha helix)
            
        Returns:
            Hydrophobic moment values
        """
        moments = []
        
        for i in range(len(sequence) - window_size + 1):
            window = sequence[i:i + window_size]
            
            # Compute moment components
            sum_x = 0
            sum_y = 0
            
            for j, aa in enumerate(window):
                if aa in self.scale:
                    hydro = self.scale[aa]
                    theta = np.radians(j * angle)
                    sum_x += hydro * np.cos(theta)
                    sum_y += hydro * np.sin(theta)
                    
            # Compute magnitude
            moment = np.sqrt(sum_x**2 + sum_y**2) / window_size
            moments.append(moment)
            
        return torch.tensor(moments)