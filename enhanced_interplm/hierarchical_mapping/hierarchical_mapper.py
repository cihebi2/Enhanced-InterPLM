# enhanced_interplm/hierarchical_mapping/hierarchical_mapper.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import pandas as pd
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist


@dataclass
class ProteinAnnotation:
    """Container for protein functional annotations."""
    sequence: str
    aa_properties: Dict[int, Dict[str, float]]  # Position -> properties
    secondary_structure: str  # DSSP string
    domains: List[Tuple[int, int, str]]  # (start, end, domain_type)
    go_terms: List[str]  # Gene Ontology terms
    ec_number: Optional[str] = None  # Enzyme classification


class AminoAcidPropertyMapper(nn.Module):
    """
    Maps SAE features to amino acid level properties.
    """
    
    def __init__(
        self,
        feature_dim: int,
        property_dim: int = 64,
        num_properties: int = 20  # Standard amino acids
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.property_dim = property_dim
        
        # Amino acid property embeddings
        self.aa_embeddings = nn.Embedding(num_properties, property_dim)
        
        # Feature to property mapper
        self.feature_to_property = nn.Sequential(
            nn.Linear(feature_dim, property_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(property_dim * 2, property_dim),
            nn.LayerNorm(property_dim)
        )
        
        # Property predictor heads
        self.hydrophobicity_head = nn.Linear(property_dim, 1)
        self.charge_head = nn.Linear(property_dim, 3)  # +, -, neutral
        self.size_head = nn.Linear(property_dim, 1)
        self.polarity_head = nn.Linear(property_dim, 2)  # polar, nonpolar
        self.aromaticity_head = nn.Linear(property_dim, 2)  # aromatic, non-aromatic
        
    def forward(
        self,
        features: torch.Tensor,
        sequences: List[str]
    ) -> Dict[str, torch.Tensor]:
        """
        Map features to amino acid properties.
        
        Args:
            features: [batch_size, seq_len, feature_dim]
            sequences: List of amino acid sequences
            
        Returns:
            Dictionary of predicted properties
        """
        batch_size, seq_len, _ = features.shape
        
        # Map features to property space
        property_features = self.feature_to_property(features)
        
        # Predict various properties
        predictions = {
            'hydrophobicity': torch.sigmoid(self.hydrophobicity_head(property_features)),
            'charge': F.softmax(self.charge_head(property_features), dim=-1),
            'size': torch.sigmoid(self.size_head(property_features)),
            'polarity': F.softmax(self.polarity_head(property_features), dim=-1),
            'aromaticity': F.softmax(self.aromaticity_head(property_features), dim=-1)
        }
        
        # Compute property conservation scores
        conservation_scores = self._compute_conservation(predictions, sequences)
        predictions['conservation'] = conservation_scores
        
        return predictions
        
    def _compute_conservation(
        self,
        predictions: Dict[str, torch.Tensor],
        sequences: List[str]
    ) -> torch.Tensor:
        """Compute conservation scores based on predicted properties."""
        # Simplified conservation score based on property consistency
        hydro_var = predictions['hydrophobicity'].var(dim=1)
        charge_entropy = -(predictions['charge'] * torch.log(predictions['charge'] + 1e-8)).sum(dim=-1).mean(dim=1)
        
        # Lower variance and entropy indicate higher conservation
        conservation = 1.0 - (hydro_var + charge_entropy) / 2
        
        return conservation.unsqueeze(1).expand(-1, predictions['hydrophobicity'].shape[1], -1)


class SecondaryStructureMapper(nn.Module):
    """
    Maps features to secondary structure elements.
    """
    
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 128,
        num_ss_types: int = 8  # DSSP 8-state
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_ss_types = num_ss_types
        
        # Bidirectional LSTM for capturing structural context
        self.structure_lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        
        # Secondary structure classifier
        self.ss_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_ss_types)
        )
        
        # Structure motif detector
        self.motif_conv = nn.Conv1d(
            in_channels=hidden_dim * 2,
            out_channels=64,
            kernel_size=5,
            padding=2
        )
        
        # Common motif patterns
        self.register_buffer('helix_pattern', torch.tensor([1, 1, 1, 1, 0, 0, 0, 0]))
        self.register_buffer('sheet_pattern', torch.tensor([0, 0, 0, 0, 1, 1, 0, 0]))
        self.register_buffer('turn_pattern', torch.tensor([0, 0, 0, 0, 0, 0, 1, 0]))
        
    def forward(
        self,
        features: torch.Tensor,
        true_ss: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Map features to secondary structures.
        
        Args:
            features: [batch_size, seq_len, feature_dim]
            true_ss: Optional true secondary structures for training
            
        Returns:
            Dictionary with structure predictions and motifs
        """
        batch_size, seq_len, _ = features.shape
        
        # Get contextual representations
        lstm_out, _ = self.structure_lstm(features)
        
        # Predict secondary structure
        ss_logits = self.ss_classifier(lstm_out)
        ss_probs = F.softmax(ss_logits, dim=-1)
        
        # Detect structural motifs
        lstm_features = lstm_out.transpose(1, 2)  # [batch, hidden*2, seq_len]
        motif_features = self.motif_conv(lstm_features)
        motif_features = motif_features.transpose(1, 2)  # [batch, seq_len, 64]
        
        # Identify specific motif types
        motifs = self._detect_motifs(ss_probs)
        
        results = {
            'ss_probabilities': ss_probs,
            'ss_predictions': ss_probs.argmax(dim=-1),
            'motif_features': motif_features,
            'helix_score': motifs['helix'],
            'sheet_score': motifs['sheet'],
            'turn_score': motifs['turn'],
            'disorder_score': motifs['disorder']
        }
        
        return results
        
    def _detect_motifs(self, ss_probs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Detect specific structural motifs from SS probabilities."""
        batch_size, seq_len, _ = ss_probs.shape
        
        # Compute motif scores using pattern matching
        helix_score = F.conv1d(
            ss_probs.transpose(1, 2),
            self.helix_pattern.view(1, -1, 1).float(),
            padding=0
        ).transpose(1, 2)
        
        sheet_score = F.conv1d(
            ss_probs.transpose(1, 2),
            self.sheet_pattern.view(1, -1, 1).float(),
            padding=0
        ).transpose(1, 2)
        
        turn_score = F.conv1d(
            ss_probs.transpose(1, 2),
            self.turn_pattern.view(1, -1, 1).float(),
            padding=0
        ).transpose(1, 2)
        
        # Disorder score (low confidence in any structure)
        disorder_score = 1.0 - ss_probs.max(dim=-1)[0].unsqueeze(-1)
        
        # Pad to match sequence length
        pad_size = seq_len - helix_score.shape[1]
        if pad_size > 0:
            helix_score = F.pad(helix_score, (0, 0, 0, pad_size))
            sheet_score = F.pad(sheet_score, (0, 0, 0, pad_size))
            turn_score = F.pad(turn_score, (0, 0, 0, pad_size))
            
        return {
            'helix': helix_score,
            'sheet': sheet_score,
            'turn': turn_score,
            'disorder': disorder_score
        }


class DomainFunctionMapper(nn.Module):
    """
    Maps features to protein domains and functional regions.
    """
    
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 256,
        num_domain_types: int = 100  # Common domain families
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # Domain boundary detector
        self.boundary_detector = nn.Sequential(
            nn.Conv1d(feature_dim, hidden_dim, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim // 2, 2, kernel_size=3, padding=1)  # Start/end probabilities
        )
        
        # Domain classifier
        self.domain_classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_domain_types)
        )
        
        # Functional site predictor
        self.functional_site_predictor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 5)  # Active, binding, allosteric, catalytic, regulatory
        )
        
        # Domain interaction network
        self.domain_interaction = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True
        )
        
    def forward(
        self,
        features: torch.Tensor,
        annotations: Optional[List[ProteinAnnotation]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Map features to domains and functional regions.
        
        Args:
            features: [batch_size, seq_len, feature_dim]
            annotations: Optional protein annotations for training
            
        Returns:
            Dictionary with domain and function predictions
        """
        batch_size, seq_len, _ = features.shape
        
        # Detect domain boundaries
        features_conv = features.transpose(1, 2)  # [batch, feature_dim, seq_len]
        boundary_logits = self.boundary_detector(features_conv)
        boundary_probs = F.softmax(boundary_logits, dim=1).transpose(1, 2)
        
        # Identify domains from boundaries
        domains = self._extract_domains(boundary_probs)
        
        # Classify each domain
        domain_predictions = []
        domain_features = []
        
        for b in range(batch_size):
            seq_domains = []
            seq_domain_features = []
            
            for start, end in domains[b]:
                # Average features within domain
                domain_feat = features[b, start:end].mean(dim=0)
                domain_class = self.domain_classifier(domain_feat)
                
                seq_domains.append({
                    'start': start,
                    'end': end,
                    'class_probs': F.softmax(domain_class, dim=-1),
                    'class': domain_class.argmax()
                })
                seq_domain_features.append(domain_feat)
                
            domain_predictions.append(seq_domains)
            if seq_domain_features:
                domain_features.append(torch.stack(seq_domain_features))
            else:
                domain_features.append(torch.zeros(1, self.feature_dim).to(features.device))
                
        # Predict functional sites
        functional_sites = F.sigmoid(self.functional_site_predictor(features))
        
        # Analyze domain interactions
        domain_interactions = self._analyze_domain_interactions(domain_features)
        
        return {
            'boundary_probabilities': boundary_probs,
            'domains': domain_predictions,
            'functional_sites': functional_sites,
            'domain_interactions': domain_interactions,
            'active_sites': functional_sites[:, :, 0],
            'binding_sites': functional_sites[:, :, 1],
            'catalytic_sites': functional_sites[:, :, 4]
        }
        
    def _extract_domains(
        self,
        boundary_probs: torch.Tensor,
        threshold: float = 0.5
    ) -> List[List[Tuple[int, int]]]:
        """Extract domain regions from boundary probabilities."""
        batch_size, seq_len, _ = boundary_probs.shape
        all_domains = []
        
        for b in range(batch_size):
            domains = []
            start_probs = boundary_probs[b, :, 0].cpu().numpy()
            end_probs = boundary_probs[b, :, 1].cpu().numpy()
            
            # Find peaks in start/end probabilities
            in_domain = False
            current_start = 0
            
            for i in range(seq_len):
                if not in_domain and start_probs[i] > threshold:
                    current_start = i
                    in_domain = True
                elif in_domain and end_probs[i] > threshold:
                    domains.append((current_start, i + 1))
                    in_domain = False
                    
            # Close any open domain
            if in_domain:
                domains.append((current_start, seq_len))
                
            all_domains.append(domains)
            
        return all_domains
        
    def _analyze_domain_interactions(
        self,
        domain_features: List[torch.Tensor]
    ) -> torch.Tensor:
        """Analyze interactions between domains."""
        max_domains = max(len(df) for df in domain_features)
        batch_size = len(domain_features)
        
        # Pad domain features to same length
        padded_features = torch.zeros(
            batch_size, max_domains, domain_features[0].shape[-1]
        ).to(domain_features[0].device)
        
        for b, df in enumerate(domain_features):
            padded_features[b, :len(df)] = df
            
        # Project to hidden dimension
        domain_hidden = F.relu(
            F.linear(padded_features, self.domain_classifier[0].weight, self.domain_classifier[0].bias)
        )
        
        # Compute domain interactions
        interactions, _ = self.domain_interaction(
            domain_hidden, domain_hidden, domain_hidden
        )
        
        return interactions


class CrossLevelIntegrator(nn.Module):
    """
    Integrates features across different hierarchical levels.
    """
    
    def __init__(
        self,
        aa_dim: int = 64,
        ss_dim: int = 128,
        domain_dim: int = 256
    ):
        super().__init__()
        
        # Level-specific encoders
        self.aa_encoder = nn.Sequential(
            nn.Linear(aa_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128)
        )
        
        self.ss_encoder = nn.Sequential(
            nn.Linear(ss_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128)
        )
        
        self.domain_encoder = nn.Sequential(
            nn.Linear(domain_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128)
        )
        
        # Cross-level attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=128,
            num_heads=8,
            batch_first=True
        )
        
        # Integration network
        self.integrator = nn.Sequential(
            nn.Linear(128 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
    def forward(
        self,
        aa_features: torch.Tensor,
        ss_features: torch.Tensor,
        domain_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Integrate features across hierarchical levels.
        
        Args:
            aa_features: Amino acid level features
            ss_features: Secondary structure features
            domain_features: Domain level features
            
        Returns:
            Integrated feature representation
        """
        # Encode each level
        aa_encoded = self.aa_encoder(aa_features)
        ss_encoded = self.ss_encoder(ss_features)
        
        # Handle variable-length domain features
        if domain_features.dim() == 2:
            domain_features = domain_features.unsqueeze(0)
        domain_encoded = self.domain_encoder(domain_features).mean(dim=1, keepdim=True)
        
        # Expand domain features to sequence length
        seq_len = aa_encoded.shape[1]
        domain_encoded = domain_encoded.expand(-1, seq_len, -1)
        
        # Cross-level attention
        # AA attends to SS
        aa_ss_attn, _ = self.cross_attention(aa_encoded, ss_encoded, ss_encoded)
        
        # SS attends to domain
        ss_domain_attn, _ = self.cross_attention(ss_encoded, domain_encoded, domain_encoded)
        
        # Concatenate all representations
        combined = torch.cat([aa_ss_attn, ss_domain_attn, domain_encoded], dim=-1)
        
        # Final integration
        integrated = self.integrator(combined)
        
        return {
            'integrated_features': integrated,
            'aa_ss_attention': aa_ss_attn,
            'ss_domain_attention': ss_domain_attn,
            'hierarchical_embedding': combined
        }