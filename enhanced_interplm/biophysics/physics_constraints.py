# 物理约束模块 
# enhanced_interplm/biophysics/physics_constraints.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class AminoAcidProperties:
    """Physical and chemical properties of amino acids."""
    hydrophobicity = {
        'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
        'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
        'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
        'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
    }
    
    charge = {
        'A': 0, 'C': 0, 'D': -1, 'E': -1, 'F': 0,
        'G': 0, 'H': 0.5, 'I': 0, 'K': 1, 'L': 0,
        'M': 0, 'N': 0, 'P': 0, 'Q': 0, 'R': 1,
        'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0
    }
    
    size = {
        'A': 88.6, 'C': 108.5, 'D': 111.1, 'E': 138.4, 'F': 189.9,
        'G': 60.1, 'H': 153.2, 'I': 166.7, 'K': 168.6, 'L': 166.7,
        'M': 162.9, 'N': 114.1, 'P': 112.7, 'Q': 143.8, 'R': 173.4,
        'S': 89.0, 'T': 116.1, 'V': 140.0, 'W': 227.8, 'Y': 193.6
    }
    
    # Hydrogen bond donor/acceptor capabilities
    h_bond_donor = {
        'A': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0,
        'G': 0, 'H': 1, 'I': 0, 'K': 1, 'L': 0,
        'M': 0, 'N': 1, 'P': 0, 'Q': 1, 'R': 2,
        'S': 1, 'T': 1, 'V': 0, 'W': 1, 'Y': 1
    }
    
    h_bond_acceptor = {
        'A': 0, 'C': 0, 'D': 2, 'E': 2, 'F': 0,
        'G': 0, 'H': 1, 'I': 0, 'K': 0, 'L': 0,
        'M': 1, 'N': 1, 'P': 0, 'Q': 1, 'R': 0,
        'S': 1, 'T': 1, 'V': 0, 'W': 0, 'Y': 1
    }


class BiophysicsConstraintModule(nn.Module):
    """
    Module that incorporates biophysical constraints into feature learning.
    """
    
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 256,
        constraint_weight: float = 0.1
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.constraint_weight = constraint_weight
        self.aa_properties = AminoAcidProperties()
        
        # Feature extractors for different physical properties
        self.hydrophobic_extractor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.charge_extractor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),  # Positive, negative, neutral
            nn.Softmax(dim=-1)
        )
        
        self.hbond_extractor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)  # Donor and acceptor propensities
        )
        
        # Spatial interaction predictor
        self.spatial_predictor = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        features: torch.Tensor,
        sequences: List[str],
        structures: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Apply biophysical constraints to features.
        
        Args:
            features: [batch_size, seq_len, feature_dim]
            sequences: List of amino acid sequences
            structures: Optional 3D coordinates [batch_size, seq_len, 3]
            
        Returns:
            Constrained features and constraint losses
        """
        batch_size, seq_len, _ = features.shape
        
        # Extract biophysical features
        hydrophobic_pred = self.hydrophobic_extractor(features)
        charge_pred = self.charge_extractor(features)
        hbond_pred = self.hbond_extractor(features)
        
        # Compute constraint losses
        losses = {}
        
        # Hydrophobicity constraint
        losses['hydrophobic'] = self._hydrophobic_constraint_loss(
            hydrophobic_pred, sequences
        )
        
        # Charge constraint
        losses['charge'] = self._charge_constraint_loss(
            charge_pred, sequences
        )
        
        # Hydrogen bond constraint
        losses['hbond'] = self._hbond_constraint_loss(
            hbond_pred, sequences, structures
        )
        
        # If structures are provided, add spatial constraints
        if structures is not None:
            losses['spatial'] = self._spatial_constraint_loss(
                features, structures, sequences
            )
            
        # Apply constraints to features
        constrained_features = self._apply_constraints(
            features, hydrophobic_pred, charge_pred, hbond_pred
        )
        
        return constrained_features, losses
        
    def _hydrophobic_constraint_loss(
        self,
        predictions: torch.Tensor,
        sequences: List[str]
    ) -> torch.Tensor:
        """Compute loss for hydrophobicity predictions."""
        batch_size, seq_len, _ = predictions.shape
        
        # Get true hydrophobicity values
        true_hydrophobic = torch.zeros_like(predictions)
        
        for b, seq in enumerate(sequences):
            for i, aa in enumerate(seq):
                if aa in self.aa_properties.hydrophobicity:
                    # Normalize to [0, 1]
                    hydro_value = (self.aa_properties.hydrophobicity[aa] + 5) / 10
                    true_hydrophobic[b, i, 0] = hydro_value
                    
        # MSE loss
        loss = F.mse_loss(predictions, true_hydrophobic)
        
        return loss
        
    def _charge_constraint_loss(
        self,
        predictions: torch.Tensor,
        sequences: List[str]
    ) -> torch.Tensor:
        """Compute loss for charge predictions."""
        batch_size, seq_len, _ = predictions.shape
        
        # Get true charge categories
        true_charge = torch.zeros(batch_size, seq_len, 3)
        
        for b, seq in enumerate(sequences):
            for i, aa in enumerate(seq):
                if aa in self.aa_properties.charge:
                    charge = self.aa_properties.charge[aa]
                    if charge > 0:
                        true_charge[b, i, 0] = 1  # Positive
                    elif charge < 0:
                        true_charge[b, i, 1] = 1  # Negative
                    else:
                        true_charge[b, i, 2] = 1  # Neutral
                        
        true_charge = true_charge.to(predictions.device)
        
        # Cross-entropy loss
        loss = F.cross_entropy(
            predictions.view(-1, 3),
            true_charge.view(-1, 3).argmax(dim=-1)
        )
        
        return loss
        
    def _hbond_constraint_loss(
        self,
        predictions: torch.Tensor,
        sequences: List[str],
        structures: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute loss for hydrogen bond predictions."""
        batch_size, seq_len, _ = predictions.shape
        
        # Get true H-bond capabilities
        true_hbond = torch.zeros_like(predictions)
        
        for b, seq in enumerate(sequences):
            for i, aa in enumerate(seq):
                if aa in self.aa_properties.h_bond_donor:
                    true_hbond[b, i, 0] = self.aa_properties.h_bond_donor[aa]
                    true_hbond[b, i, 1] = self.aa_properties.h_bond_acceptor[aa]
                    
        # MSE loss for H-bond propensities
        loss = F.mse_loss(predictions, true_hbond)
        
        # If structures provided, add geometric constraints
        if structures is not None:
            geometric_loss = self._hbond_geometric_loss(
                predictions, structures, sequences
            )
            loss = loss + 0.5 * geometric_loss
            
        return loss
        
    def _hbond_geometric_loss(
        self,
        hbond_pred: torch.Tensor,
        structures: torch.Tensor,
        sequences: List[str]
    ) -> torch.Tensor:
        """Compute geometric constraints for hydrogen bonds."""
        batch_size, seq_len, _ = structures.shape
        
        # Compute pairwise distances
        distances = torch.cdist(structures, structures)
        
        # H-bond distance range: 2.5-3.5 Å
        hbond_mask = (distances > 2.5) & (distances < 3.5)
        
        # Compute H-bond compatibility
        donor_pred = hbond_pred[:, :, 0:1]  # [batch, seq, 1]
        acceptor_pred = hbond_pred[:, :, 1:2]
        
        # Predicted H-bond probability between pairs
        hbond_prob = donor_pred @ acceptor_pred.transpose(-2, -1)
        
        # Loss: predicted H-bonds should align with geometric constraints
        loss = F.binary_cross_entropy_with_logits(
            hbond_prob,
            hbond_mask.float()
        )
        
        return loss
        
    def _spatial_constraint_loss(
        self,
        features: torch.Tensor,
        structures: torch.Tensor,
        sequences: List[str]
    ) -> torch.Tensor:
        """Compute spatial interaction constraints."""
        batch_size, seq_len, _ = features.shape
        
        # Compute pairwise distances
        distances = torch.cdist(structures, structures)
        
        # Define contact threshold (8 Å)
        contacts = (distances < 8.0).float()
        
        # Predict spatial interactions from features
        spatial_pred = torch.zeros(batch_size, seq_len, seq_len).to(features.device)
        
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                pair_features = torch.cat([features[:, i], features[:, j]], dim=-1)
                spatial_pred[:, i, j] = self.spatial_predictor(pair_features).squeeze(-1)
                spatial_pred[:, j, i] = spatial_pred[:, i, j]
                
        # Loss: predicted interactions should match spatial contacts
        loss = F.binary_cross_entropy(spatial_pred, contacts)
        
        return loss
        
    def _apply_constraints(
        self,
        features: torch.Tensor,
        hydrophobic_pred: torch.Tensor,
        charge_pred: torch.Tensor,
        hbond_pred: torch.Tensor
    ) -> torch.Tensor:
        """Apply learned constraints to modulate features."""
        # Create constraint modulation factors
        hydro_modulation = 1.0 + self.constraint_weight * (hydrophobic_pred - 0.5)
        
        # Charge-based modulation
        charge_modulation = 1.0 + self.constraint_weight * (
            charge_pred[:, :, 0:1] - charge_pred[:, :, 1:2]
        )
        
        # H-bond modulation
        hbond_modulation = 1.0 + self.constraint_weight * (
            hbond_pred.mean(dim=-1, keepdim=True) - 0.5
        )
        
        # Apply modulations
        constrained_features = features * hydro_modulation * charge_modulation * hbond_modulation
        
        return constrained_features


class BiophysicsGuidedSAE(nn.Module):
    """
    Sparse Autoencoder with integrated biophysical constraints.
    """
    
    def __init__(
        self,
        activation_dim: int,
        dict_size: int,
        hidden_dim: int = 256,
        physics_weight: float = 0.1
    ):
        super().__init__()
        
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.physics_weight = physics_weight
        
        # Standard SAE components
        self.encoder = nn.Linear(activation_dim, dict_size, bias=True)
        self.decoder = nn.Linear(dict_size, activation_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(activation_dim))
        
        # Biophysics constraint module
        self.physics_module = BiophysicsConstraintModule(
            feature_dim=dict_size,
            hidden_dim=hidden_dim,
            constraint_weight=physics_weight
        )
        
        # Initialize decoder with unit norm
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(
                self.decoder.weight.data, p=2, dim=0
            )
            
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode inputs to sparse features."""
        return F.relu(self.encoder(x - self.bias))
        
    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """Decode features back to input space."""
        return self.decoder(f) + self.bias
        
    def forward(
        self,
        x: torch.Tensor,
        sequences: Optional[List[str]] = None,
        structures: Optional[torch.Tensor] = None,
        return_physics_loss: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Dict]]:
        """
        Forward pass with optional biophysical constraints.
        
        Args:
            x: Input activations [batch_size, seq_len, activation_dim]
            sequences: Amino acid sequences (required for physics constraints)
            structures: 3D coordinates (optional)
            return_physics_loss: Whether to compute and return physics losses
            
        Returns:
            Reconstructed activations, features, and optional physics losses
        """
        # Encode
        features = self.encode(x)
        
        # Apply biophysical constraints if sequences provided
        if sequences is not None and return_physics_loss:
            constrained_features, physics_losses = self.physics_module(
                features, sequences, structures
            )
            # Weighted combination
            features = (1 - self.physics_weight) * features + \
                      self.physics_weight * constrained_features
        else:
            physics_losses = None
            
        # Decode
        x_reconstructed = self.decode(features)
        
        return x_reconstructed, features, physics_losses