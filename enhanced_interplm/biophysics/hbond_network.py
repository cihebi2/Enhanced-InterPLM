# 氢键网络 
"""Hydrogen bond network analysis module."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
import networkx as nx


class HBondNetworkAnalyzer(nn.Module):
    """
    Analyzes hydrogen bond networks in protein structures.
    """
    
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 128
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # H-bond donor/acceptor capabilities
        self.donors = {
            'N': True,  # Backbone N
            'R': True,  # Arg
            'K': True,  # Lys
            'W': True,  # Trp
            'Q': True,  # Gln
            'N': True,  # Asn
            'H': True,  # His
            'S': True,  # Ser
            'T': True,  # Thr
            'Y': True,  # Tyr
            'C': True,  # Cys
        }
        
        self.acceptors = {
            'O': True,  # Backbone O
            'D': True,  # Asp
            'E': True,  # Glu
            'Q': True,  # Gln
            'N': True,  # Asn
            'H': True,  # His
            'S': True,  # Ser
            'T': True,  # Thr
            'Y': True,  # Tyr
            'C': True,  # Cys
            'M': True,  # Met
        }
        
        # Neural network for H-bond prediction
        self.hbond_predictor = nn.Sequential(
            nn.Linear(feature_dim * 2 + 3, hidden_dim),  # Features + geometry
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # H-bond strength estimator
        self.strength_estimator = nn.Sequential(
            nn.Linear(feature_dim * 2 + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Network analyzer
        self.network_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
    def forward(
        self,
        features: torch.Tensor,
        sequences: List[str],
        structures: torch.Tensor,
        backbone_coords: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Analyze hydrogen bond networks.
        
        Args:
            features: [batch_size, seq_len, feature_dim]
            sequences: List of amino acid sequences
            structures: CA coordinates [batch_size, seq_len, 3]
            backbone_coords: Optional backbone coords [batch_size, seq_len, 3, 3] (N, CA, C)
            
        Returns:
            Dictionary containing H-bond analysis results
        """
        batch_size, seq_len, _ = features.shape
        
        # Predict H-bond network
        hbond_network = self._predict_hbond_network(
            features, sequences, structures, backbone_coords
        )
        
        # Analyze network properties
        network_properties = self._analyze_network_properties(hbond_network)
        
        # Identify H-bond motifs
        hbond_motifs = self._identify_hbond_motifs(hbond_network, sequences)
        
        # Compute H-bond energy
        hbond_energy = self._compute_hbond_energy(hbond_network)
        
        results = {
            'hbond_network': hbond_network['adjacency'],
            'hbond_strengths': hbond_network['strengths'],
            'network_properties': network_properties,
            'hbond_motifs': hbond_motifs,
            'hbond_energy': hbond_energy,
            'donor_sites': self._identify_donor_sites(sequences),
            'acceptor_sites': self._identify_acceptor_sites(sequences)
        }
        
        return results
        
    def _predict_hbond_network(
        self,
        features: torch.Tensor,
        sequences: List[str],
        structures: torch.Tensor,
        backbone_coords: Optional[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Predict hydrogen bond network."""
        batch_size, seq_len, _ = features.shape
        
        adjacency_matrices = []
        strength_matrices = []
        
        for b in range(batch_size):
            adjacency = torch.zeros(seq_len, seq_len)
            strengths = torch.zeros(seq_len, seq_len)
            
            # Get potential donors and acceptors
            donors = self._get_donor_indices(sequences[b])
            acceptors = self._get_acceptor_indices(sequences[b])
            
            # Check all donor-acceptor pairs
            for donor_idx in donors:
                for acceptor_idx in acceptors:
                    if abs(donor_idx - acceptor_idx) < 2:  # Skip adjacent residues
                        continue
                        
                    # Compute geometric features
                    geometry = self._compute_hbond_geometry(
                        structures[b, donor_idx],
                        structures[b, acceptor_idx],
                        backbone_coords[b] if backbone_coords is not None else None,
                        donor_idx,
                        acceptor_idx
                    )
                    
                    # Combine features and geometry
                    combined = torch.cat([
                        features[b, donor_idx],
                        features[b, acceptor_idx],
                        geometry
                    ])
                    
                    # Predict H-bond
                    hbond_prob = self.hbond_predictor(combined.unsqueeze(0))
                    
                    if hbond_prob > 0.5:
                        adjacency[donor_idx, acceptor_idx] = 1
                        
                        # Estimate strength
                        strength = self.strength_estimator(combined.unsqueeze(0))
                        strengths[donor_idx, acceptor_idx] = strength
                        
            adjacency_matrices.append(adjacency)
            strength_matrices.append(strengths)
            
        return {
            'adjacency': torch.stack(adjacency_matrices),
            'strengths': torch.stack(strength_matrices)
        }
        
    def _compute_hbond_geometry(
        self,
        donor_pos: torch.Tensor,
        acceptor_pos: torch.Tensor,
        backbone_coords: Optional[torch.Tensor],
        donor_idx: int,
        acceptor_idx: int
    ) -> torch.Tensor:
        """Compute H-bond geometric features."""
        # Distance
        distance = torch.norm(acceptor_pos - donor_pos)
        
        # Simplified angle calculation (would use actual H positions in practice)
        if backbone_coords is not None:
            # Use backbone geometry for angle estimation
            angle = torch.tensor(120.0)  # Placeholder
        else:
            angle = torch.tensor(180.0)  # Assume linear
            
        # Distance feature (normalized)
        dist_feat = torch.exp(-distance / 3.5)  # Optimal H-bond distance ~2.8-3.5 Å
        
        # Angle feature (normalized)
        angle_feat = torch.cos(torch.deg2rad(180 - angle))  # Optimal angle ~180°
        
        # Sequence separation feature
        seq_sep = torch.tensor(abs(acceptor_idx - donor_idx) / 10.0)
        
        return torch.tensor([dist_feat, angle_feat, seq_sep])
        
    def _analyze_network_properties(
        self,
        hbond_network: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Analyze properties of H-bond network."""
        adjacency = hbond_network['adjacency']
        strengths = hbond_network['strengths']
        
        batch_size, seq_len, _ = adjacency.shape
        
        properties = {
            'degree': [],
            'clustering_coefficient': [],
            'betweenness_centrality': [],
            'network_density': [],
            'avg_path_length': []
        }
        
        for b in range(batch_size):
            # Convert to networkx graph
            G = nx.from_numpy_array(adjacency[b].cpu().numpy())
            
            # Degree
            degrees = dict(G.degree())
            properties['degree'].append(list(degrees.values()))
            
            # Clustering coefficient
            clustering = nx.clustering(G)
            properties['clustering_coefficient'].append(list(clustering.values()))
            
            # Betweenness centrality
            if G.number_of_edges() > 0:
                betweenness = nx.betweenness_centrality(G)
                properties['betweenness_centrality'].append(list(betweenness.values()))
            else:
                properties['betweenness_centrality'].append([0] * seq_len)
                
            # Network density
            density = nx.density(G)
            properties['network_density'].append(density)
            
            # Average path length
            if nx.is_connected(G):
                avg_path = nx.average_shortest_path_length(G)
            else:
                # Use largest connected component
                largest_cc = max(nx.connected_components(G), key=len)
                if len(largest_cc) > 1:
                    subG = G.subgraph(largest_cc)
                    avg_path = nx.average_shortest_path_length(subG)
                else:
                    avg_path = 0
                    
            properties['avg_path_length'].append(avg_path)
            
        # Convert to tensors
        for key in ['degree', 'clustering_coefficient', 'betweenness_centrality']:
            properties[key] = torch.tensor(properties[key]).to(adjacency.device)
            
        properties['network_density'] = torch.tensor(
            properties['network_density']
        ).to(adjacency.device)
        
        properties['avg_path_length'] = torch.tensor(
            properties['avg_path_length']
        ).to(adjacency.device)
        
        return properties
        
    def _identify_hbond_motifs(
        self,
        hbond_network: Dict[str, torch.Tensor],
        sequences: List[str]
    ) -> Dict[str, List]:
        """Identify common H-bond motifs."""
        adjacency = hbond_network['adjacency']
        batch_size = adjacency.shape[0]
        
        motifs = {
            'alpha_helix': [],
            'beta_sheet': [],
            'beta_turn': [],
            'helix_cap': []
        }
        
        for b in range(batch_size):
            adj = adjacency[b]
            seq = sequences[b]
            
            # Alpha helix: i to i+4 H-bonds
            helix_bonds = []
            for i in range(len(seq) - 4):
                if adj[i, i+4] > 0:
                    helix_bonds.append((i, i+4))
                    
            if len(helix_bonds) >= 2:
                motifs['alpha_helix'].append(helix_bonds)
                
            # Beta sheet: long-range H-bonds
            sheet_bonds = []
            for i in range(len(seq)):
                for j in range(i + 5, len(seq)):
                    if adj[i, j] > 0:
                        sheet_bonds.append((i, j))
                        
            if len(sheet_bonds) >= 2:
                motifs['beta_sheet'].append(sheet_bonds)
                
            # Beta turn: i to i+3 H-bonds
            turn_bonds = []
            for i in range(len(seq) - 3):
                if adj[i, i+3] > 0:
                    turn_bonds.append((i, i+3))
                    
            motifs['beta_turn'].append(turn_bonds)
            
            # Helix cap: specific H-bond patterns at helix termini
            cap_bonds = []
            for bonds in helix_bonds:
                start = bonds[0]
                if start > 0 and adj[start-1, start+2] > 0:
                    cap_bonds.append((start-1, start+2))
                    
            motifs['helix_cap'].append(cap_bonds)
            
        return motifs
        
    def _compute_hbond_energy(
        self,
        hbond_network: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Estimate total H-bond energy."""
        strengths = hbond_network['strengths']
        
        # Typical H-bond energy: -1 to -5 kcal/mol
        # Strength 1.0 corresponds to -5 kcal/mol
        energy_per_bond = -5.0
        
        # Sum all H-bond energies
        total_energies = []
        
        for b in range(strengths.shape[0]):
            total_energy = (strengths[b] * energy_per_bond).sum()
            total_energies.append(total_energy)
            
        return torch.tensor(total_energies).to(strengths.device)
        
    def _get_donor_indices(self, sequence: str) -> List[int]:
        """Get indices of potential H-bond donors."""
        donors = []
        
        for i, aa in enumerate(sequence):
            if aa in self.donors:
                donors.append(i)
                
        # Add backbone N (all residues except first)
        donors.extend(range(1, len(sequence)))
        
        return list(set(donors))
        
    def _get_acceptor_indices(self, sequence: str) -> List[int]:
        """Get indices of potential H-bond acceptors."""
        acceptors = []
        
        for i, aa in enumerate(sequence):
            if aa in self.acceptors:
                acceptors.append(i)
                
        # Add backbone O (all residues except last)
        acceptors.extend(range(len(sequence) - 1))
        
        return list(set(acceptors))
        
    def _identify_donor_sites(self, sequences: List[str]) -> torch.Tensor:
        """Identify all donor sites in sequences."""
        batch_size = len(sequences)
        max_len = max(len(seq) for seq in sequences)
        
        donor_mask = torch.zeros(batch_size, max_len)
        
        for b, seq in enumerate(sequences):
            donors = self._get_donor_indices(seq)
            for idx in donors:
                if idx < max_len:
                    donor_mask[b, idx] = 1
                    
        return donor_mask.to(next(self.parameters()).device)
        
    def _identify_acceptor_sites(self, sequences: List[str]) -> torch.Tensor:
        """Identify all acceptor sites in sequences."""
        batch_size = len(sequences)
        max_len = max(len(seq) for seq in sequences)
        
        acceptor_mask = torch.zeros(batch_size, max_len)
        
        for b, seq in enumerate(sequences):
            acceptors = self._get_acceptor_indices(seq)
            for idx in acceptors:
                if idx < max_len:
                    acceptor_mask[b, idx] = 1
                    
        return acceptor_mask.to(next(self.parameters()).device)