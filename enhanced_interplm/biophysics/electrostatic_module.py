# 静电相互作用 
"""Electrostatic interaction analysis module."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.spatial.distance import cdist


class ElectrostaticAnalyzer(nn.Module):
    """
    Analyzes electrostatic interactions and charge distributions.
    """
    
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 128,
        dielectric_constant: float = 80.0
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.dielectric_constant = dielectric_constant
        
        # Amino acid charges at pH 7
        self.aa_charges = {
            'D': -1.0,  # Aspartate
            'E': -1.0,  # Glutamate
            'K': 1.0,   # Lysine
            'R': 1.0,   # Arginine
            'H': 0.1,   # Histidine (partial)
            # Neutral amino acids
            'A': 0, 'C': 0, 'F': 0, 'G': 0, 'I': 0,
            'L': 0, 'M': 0, 'N': 0, 'P': 0, 'Q': 0,
            'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0
        }
        
        # Neural network for charge prediction
        self.charge_predictor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),  # Positive, negative, neutral
            nn.Softmax(dim=-1)
        )
        
        # Salt bridge detector
        self.salt_bridge_detector = nn.Sequential(
            nn.Linear(feature_dim * 2 + 1, hidden_dim),  # Features + distance
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Electrostatic potential calculator
        self.potential_calculator = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(8, 1, kernel_size=3, padding=1)
        )
        
    def forward(
        self,
        features: torch.Tensor,
        sequences: List[str],
        structures: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Analyze electrostatic properties and interactions.
        
        Args:
            features: [batch_size, seq_len, feature_dim]
            sequences: List of amino acid sequences
            structures: Optional 3D coordinates [batch_size, seq_len, 3]
            
        Returns:
            Dictionary containing electrostatic analysis results
        """
        batch_size, seq_len, _ = features.shape
        
        # Get true charges from sequences
        true_charges = self._compute_true_charges(sequences)
        
        # Predict charge distribution
        features_flat = features.view(-1, self.feature_dim)
        predicted_charges = self.charge_predictor(features_flat)
        predicted_charges = predicted_charges.view(batch_size, seq_len, 3)
        
        # Calculate electrostatic potential
        net_charge = predicted_charges[:, :, 0:1] - predicted_charges[:, :, 1:2]
        potential = self.potential_calculator(net_charge.transpose(1, 2))
        potential = potential.transpose(1, 2)
        
        results = {
            'predicted_charges': predicted_charges,
            'true_charges': true_charges,
            'net_charge': net_charge,
            'electrostatic_potential': potential
        }
        
        # If structures provided, detect salt bridges and compute interactions
        if structures is not None:
            salt_bridges = self._detect_salt_bridges(
                features, predicted_charges, structures
            )
            results['salt_bridges'] = salt_bridges
            
            interaction_energy = self._compute_electrostatic_energy(
                predicted_charges, structures
            )
            results['electrostatic_energy'] = interaction_energy
            
        return results
        
    def _compute_true_charges(
        self,
        sequences: List[str]
    ) -> torch.Tensor:
        """Compute true charge values from sequences."""
        batch_size = len(sequences)
        max_len = max(len(seq) for seq in sequences)
        
        charges = torch.zeros(batch_size, max_len, 3)  # [positive, negative, neutral]
        
        for i, seq in enumerate(sequences):
            for j, aa in enumerate(seq):
                if aa in self.aa_charges:
                    charge = self.aa_charges[aa]
                    if charge > 0:
                        charges[i, j, 0] = 1  # Positive
                    elif charge < 0:
                        charges[i, j, 1] = 1  # Negative
                    else:
                        charges[i, j, 2] = 1  # Neutral
                        
        return charges.to(next(self.parameters()).device)
        
    def _detect_salt_bridges(
        self,
        features: torch.Tensor,
        charges: torch.Tensor,
        structures: torch.Tensor,
        distance_threshold: float = 4.0
    ) -> Dict[str, torch.Tensor]:
        """Detect salt bridges between charged residues."""
        batch_size, seq_len, _ = features.shape
        
        salt_bridge_masks = []
        salt_bridge_pairs = []
        
        for b in range(batch_size):
            # Find charged residues
            positive_mask = charges[b, :, 0] > charges[b, :, 2]  # More positive than neutral
            negative_mask = charges[b, :, 1] > charges[b, :, 2]  # More negative than neutral
            
            positive_indices = torch.where(positive_mask)[0]
            negative_indices = torch.where(negative_mask)[0]
            
            # Check distances between oppositely charged residues
            bridge_mask = torch.zeros(seq_len, seq_len)
            bridge_pairs = []
            
            for pos_idx in positive_indices:
                for neg_idx in negative_indices:
                    # Compute distance
                    dist = torch.norm(
                        structures[b, pos_idx] - structures[b, neg_idx]
                    )
                    
                    if dist < distance_threshold:
                        # Check if it's a salt bridge using neural network
                        combined_features = torch.cat([
                            features[b, pos_idx],
                            features[b, neg_idx],
                            dist.unsqueeze(0)
                        ])
                        
                        bridge_prob = self.salt_bridge_detector(
                            combined_features.unsqueeze(0)
                        )
                        
                        if bridge_prob > 0.5:
                            bridge_mask[pos_idx, neg_idx] = 1
                            bridge_mask[neg_idx, pos_idx] = 1
                            bridge_pairs.append((pos_idx.item(), neg_idx.item()))
                            
            salt_bridge_masks.append(bridge_mask)
            salt_bridge_pairs.append(bridge_pairs)
            
        return {
            'salt_bridge_masks': torch.stack(salt_bridge_masks),
            'salt_bridge_pairs': salt_bridge_pairs
        }
        
    def _compute_electrostatic_energy(
        self,
        charges: torch.Tensor,
        structures: torch.Tensor,
        cutoff_distance: float = 15.0
    ) -> torch.Tensor:
        """
        Compute electrostatic interaction energy.
        
        Uses Coulomb's law with distance-dependent dielectric.
        """
        batch_size, seq_len, _ = charges.shape
        energies = []
        
        # Coulomb constant in kcal/mol
        k_coulomb = 332.0
        
        for b in range(batch_size):
            # Get net charges
            net_charges = charges[b, :, 0] - charges[b, :, 1]
            
            # Compute pairwise distances
            coords = structures[b].cpu().numpy()
            distances = cdist(coords, coords)
            
            # Initialize energy
            energy = 0.0
            
            # Compute pairwise electrostatic energy
            for i in range(seq_len):
                for j in range(i + 1, seq_len):
                    if distances[i, j] < cutoff_distance and distances[i, j] > 0:
                        # Distance-dependent dielectric
                        dielectric = self.dielectric_constant * distances[i, j] / 4.0
                        
                        # Coulomb energy
                        e_ij = k_coulomb * net_charges[i] * net_charges[j] / (
                            dielectric * distances[i, j]
                        )
                        
                        energy += e_ij.item()
                        
            energies.append(energy)
            
        return torch.tensor(energies).to(charges.device)
        
    def compute_dipole_moment(
        self,
        charges: torch.Tensor,
        structures: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute protein dipole moment and direction.
        
        Returns:
            Dipole magnitude and direction vector
        """
        batch_size = charges.shape[0]
        
        dipole_magnitudes = []
        dipole_directions = []
        
        for b in range(batch_size):
            # Get net charges
            net_charges = charges[b, :, 0] - charges[b, :, 1]
            
            # Compute center of charge
            charge_weighted_coords = structures[b] * net_charges.unsqueeze(-1)
            total_charge = net_charges.sum()
            
            if abs(total_charge) > 1e-6:
                center_of_charge = charge_weighted_coords.sum(dim=0) / total_charge
            else:
                center_of_charge = structures[b].mean(dim=0)
                
            # Compute dipole moment
            dipole = torch.zeros(3).to(charges.device)
            
            for i in range(len(net_charges)):
                if abs(net_charges[i]) > 1e-6:
                    r_vec = structures[b, i] - center_of_charge
                    dipole += net_charges[i] * r_vec
                    
            # Magnitude and direction
            magnitude = torch.norm(dipole)
            direction = dipole / (magnitude + 1e-8)
            
            dipole_magnitudes.append(magnitude)
            dipole_directions.append(direction)
            
        return (
            torch.stack(dipole_magnitudes),
            torch.stack(dipole_directions)
        )