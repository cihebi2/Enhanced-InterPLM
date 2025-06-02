# 进化数据处理 

# enhanced_interplm/data_processing/evolution_data.py

"""
Processing and analysis of evolutionary data (MSAs) for Enhanced InterPLM.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import pandas as pd
from Bio import SeqIO, AlignIO
from Bio.Align import MultipleSeqAlignment
from scipy.stats import entropy
import logging

logger = logging.getLogger(__name__)


class EvolutionaryDataProcessor:
    """
    Processes Multiple Sequence Alignments (MSAs) and extracts evolutionary features.
    """
    
    def __init__(
        self,
        gap_cutoff: float = 0.5,
        similarity_threshold: float = 0.62,
        use_weights: bool = True
    ):
        """
        Initialize evolutionary data processor.
        
        Args:
            gap_cutoff: Maximum fraction of gaps allowed per position
            similarity_threshold: Threshold for sequence similarity weighting
            use_weights: Whether to use sequence weights
        """
        self.gap_cutoff = gap_cutoff
        self.similarity_threshold = similarity_threshold
        self.use_weights = use_weights
        
        # Amino acid alphabet
        self.aa_to_idx = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY-')}
        self.idx_to_aa = {i: aa for aa, i in self.aa_to_idx.items()}
        self.num_amino_acids = len(self.aa_to_idx)
        
    def process_msa(
        self,
        msa_path: Path,
        target_seq: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Process MSA file and extract evolutionary features.
        
        Args:
            msa_path: Path to MSA file
            target_seq: Target sequence (if None, use first sequence)
            
        Returns:
            Dictionary containing evolutionary features
        """
        # Load MSA
        alignment = self._load_msa(msa_path)
        
        if target_seq is None:
            target_seq = str(alignment[0].seq).replace('-', '')
            
        # Filter and process alignment
        filtered_alignment = self._filter_alignment(alignment, target_seq)
        
        # Compute sequence weights
        if self.use_weights:
            weights = self._compute_sequence_weights(filtered_alignment)
        else:
            weights = np.ones(len(filtered_alignment)) / len(filtered_alignment)
            
        # Extract features
        features = {
            'pssm': self._compute_pssm(filtered_alignment, weights),
            'conservation': self._compute_conservation(filtered_alignment, weights),
            'coevolution': self._compute_coevolution(filtered_alignment, weights),
            'gap_statistics': self._compute_gap_statistics(filtered_alignment),
            'effective_sequences': self._compute_effective_sequences(filtered_alignment, weights)
        }
        
        # Add metadata
        features['num_sequences'] = len(filtered_alignment)
        features['alignment_depth'] = self._compute_alignment_depth(filtered_alignment)
        features['sequence_diversity'] = self._compute_sequence_diversity(filtered_alignment)
        
        return features
        
    def _load_msa(self, msa_path: Path) -> MultipleSeqAlignment:
        """Load MSA from file."""
        # Try different formats
        for fmt in ['fasta', 'stockholm', 'clustal']:
            try:
                alignment = AlignIO.read(str(msa_path), fmt)
                logger.info(f"Loaded MSA with {len(alignment)} sequences")
                return alignment
            except:
                continue
                
        raise ValueError(f"Could not load MSA from {msa_path}")
        
    def _filter_alignment(
        self,
        alignment: MultipleSeqAlignment,
        target_seq: str
    ) -> MultipleSeqAlignment:
        """Filter alignment based on target sequence."""
        # Find target sequence in alignment
        target_idx = self._find_target_sequence(alignment, target_seq)
        
        if target_idx is None:
            logger.warning("Target sequence not found in alignment")
            return alignment
            
        # Get positions without gaps in target
        target_aligned = str(alignment[target_idx].seq)
        non_gap_positions = [i for i, aa in enumerate(target_aligned) if aa != '-']
        
        # Filter columns
        filtered_records = []
        for record in alignment:
            filtered_seq = ''.join([str(record.seq)[i] for i in non_gap_positions])
            filtered_records.append(
                SeqIO.SeqRecord(
                    seq=filtered_seq,
                    id=record.id,
                    description=record.description
                )
            )
            
        return MultipleSeqAlignment(filtered_records)
        
    def _find_target_sequence(
        self,
        alignment: MultipleSeqAlignment,
        target_seq: str
    ) -> Optional[int]:
        """Find target sequence in alignment."""
        target_nogap = target_seq.replace('-', '')
        
        for i, record in enumerate(alignment):
            seq_nogap = str(record.seq).replace('-', '')
            if seq_nogap == target_nogap:
                return i
                
        return None
        
    def _compute_sequence_weights(
        self,
        alignment: MultipleSeqAlignment
    ) -> np.ndarray:
        """Compute sequence weights using Henikoff weighting."""
        num_seqs = len(alignment)
        seq_len = alignment.get_alignment_length()
        
        # Count amino acids at each position
        counts = np.zeros((seq_len, self.num_amino_acids))
        
        for record in alignment:
            for i, aa in enumerate(str(record.seq)):
                if aa in self.aa_to_idx:
                    counts[i, self.aa_to_idx[aa]] += 1
                    
        # Compute weights
        weights = np.zeros(num_seqs)
        
        for seq_idx, record in enumerate(alignment):
            weight = 0
            for pos, aa in enumerate(str(record.seq)):
                if aa in self.aa_to_idx:
                    aa_idx = self.aa_to_idx[aa]
                    # Number of different amino acids at this position
                    num_aa_types = np.sum(counts[pos] > 0)
                    if num_aa_types > 1 and counts[pos, aa_idx] > 0:
                        weight += 1.0 / (num_aa_types * counts[pos, aa_idx])
                        
            weights[seq_idx] = weight
            
        # Normalize weights
        weights = weights / weights.sum()
        
        return weights
        
    def _compute_pssm(
        self,
        alignment: MultipleSeqAlignment,
        weights: np.ndarray
    ) -> torch.Tensor:
        """Compute Position-Specific Scoring Matrix."""
        seq_len = alignment.get_alignment_length()
        pssm = np.zeros((seq_len, self.num_amino_acids))
        
        # Count weighted amino acids
        for seq_idx, record in enumerate(alignment):
            for pos, aa in enumerate(str(record.seq)):
                if aa in self.aa_to_idx:
                    pssm[pos, self.aa_to_idx[aa]] += weights[seq_idx]
                    
        # Add pseudocounts
        pseudocount = 0.01
        pssm = pssm + pseudocount
        
        # Normalize to probabilities
        pssm = pssm / pssm.sum(axis=1, keepdims=True)
        
        # Convert to log-odds
        background_freq = np.ones(self.num_amino_acids) / 20  # Exclude gap
        background_freq[-1] = 0  # Gap frequency
        
        with np.errstate(divide='ignore', invalid='ignore'):
            log_odds = np.log2(pssm / background_freq)
            log_odds[np.isnan(log_odds)] = 0
            log_odds[np.isinf(log_odds)] = 0
            
        return torch.tensor(log_odds, dtype=torch.float32)
        
    def _compute_conservation(
        self,
        alignment: MultipleSeqAlignment,
        weights: np.ndarray
    ) -> torch.Tensor:
        """Compute per-position conservation scores."""
        seq_len = alignment.get_alignment_length()
        conservation = np.zeros(seq_len)
        
        for pos in range(seq_len):
            # Get amino acid distribution at this position
            aa_counts = np.zeros(self.num_amino_acids)
            
            for seq_idx, record in enumerate(alignment):
                aa = str(record.seq)[pos]
                if aa in self.aa_to_idx:
                    aa_counts[self.aa_to_idx[aa]] += weights[seq_idx]
                    
            # Compute entropy
            aa_probs = aa_counts / aa_counts.sum()
            conservation[pos] = 1.0 - entropy(aa_probs) / np.log(20)  # Normalized
            
        return torch.tensor(conservation, dtype=torch.float32)
        
    def _compute_coevolution(
        self,
        alignment: MultipleSeqAlignment,
        weights: np.ndarray,
        max_pairs: int = 1000
    ) -> torch.Tensor:
        """Compute coevolution scores using mutual information."""
        seq_len = alignment.get_alignment_length()
        
        # For efficiency, limit to top varying positions
        conservation = self._compute_conservation(alignment, weights).numpy()
        variable_positions = np.where(conservation < 0.9)[0]
        
        if len(variable_positions) > 100:
            # Select most variable positions
            variable_positions = variable_positions[
                np.argsort(conservation[variable_positions])[:100]
            ]
            
        # Compute mutual information between pairs
        mi_matrix = np.zeros((seq_len, seq_len))
        
        for i, pos_i in enumerate(variable_positions):
            for j, pos_j in enumerate(variable_positions):
                if i < j:
                    mi = self._mutual_information(
                        alignment, weights, pos_i, pos_j
                    )
                    mi_matrix[pos_i, pos_j] = mi
                    mi_matrix[pos_j, pos_i] = mi
                    
        # Apply APC correction
        mi_corrected = self._apc_correction(mi_matrix)
        
        return torch.tensor(mi_corrected, dtype=torch.float32)
        
    def _mutual_information(
        self,
        alignment: MultipleSeqAlignment,
        weights: np.ndarray,
        pos_i: int,
        pos_j: int
    ) -> float:
        """Compute mutual information between two positions."""
        # Joint distribution
        joint_counts = np.zeros((self.num_amino_acids, self.num_amino_acids))
        
        for seq_idx, record in enumerate(alignment):
            aa_i = str(record.seq)[pos_i]
            aa_j = str(record.seq)[pos_j]
            
            if aa_i in self.aa_to_idx and aa_j in self.aa_to_idx:
                joint_counts[self.aa_to_idx[aa_i], self.aa_to_idx[aa_j]] += weights[seq_idx]
                
        # Normalize
        joint_probs = joint_counts / joint_counts.sum()
        
        # Marginal distributions
        marginal_i = joint_probs.sum(axis=1)
        marginal_j = joint_probs.sum(axis=0)
        
        # Compute MI
        mi = 0
        for i in range(self.num_amino_acids):
            for j in range(self.num_amino_acids):
                if joint_probs[i, j] > 0 and marginal_i[i] > 0 and marginal_j[j] > 0:
                    mi += joint_probs[i, j] * np.log(
                        joint_probs[i, j] / (marginal_i[i] * marginal_j[j])
                    )
                    
        return mi
        
    def _apc_correction(self, mi_matrix: np.ndarray) -> np.ndarray:
        """Apply Average Product Correction to MI matrix."""
        # Compute row and column means
        row_means = mi_matrix.mean(axis=1)
        col_means = mi_matrix.mean(axis=0)
        overall_mean = mi_matrix.mean()
        
        # Apply correction
        corrected = mi_matrix.copy()
        n = mi_matrix.shape[0]
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    correction = (row_means[i] * col_means[j]) / overall_mean
                    corrected[i, j] = max(0, mi_matrix[i, j] - correction)
                    
        return corrected
        
    def _compute_gap_statistics(
        self,
        alignment: MultipleSeqAlignment
    ) -> torch.Tensor:
        """Compute gap statistics per position."""
        seq_len = alignment.get_alignment_length()
        gap_freq = np.zeros(seq_len)
        
        for record in alignment:
            for pos, aa in enumerate(str(record.seq)):
                if aa == '-':
                    gap_freq[pos] += 1
                    
        gap_freq = gap_freq / len(alignment)
        
        return torch.tensor(gap_freq, dtype=torch.float32)
        
    def _compute_effective_sequences(
        self,
        alignment: MultipleSeqAlignment,
        weights: np.ndarray
    ) -> float:
        """Compute effective number of sequences."""
        # Using exponential of entropy of weights
        weight_entropy = -np.sum(weights * np.log(weights + 1e-10))
        return np.exp(weight_entropy)
        
    def _compute_alignment_depth(
        self,
        alignment: MultipleSeqAlignment
    ) -> np.ndarray:
        """Compute alignment depth (non-gap sequences) per position."""
        seq_len = alignment.get_alignment_length()
        depth = np.zeros(seq_len)
        
        for record in alignment:
            for pos, aa in enumerate(str(record.seq)):
                if aa != '-':
                    depth[pos] += 1
                    
        return depth
        
    def _compute_sequence_diversity(
        self,
        alignment: MultipleSeqAlignment
    ) -> float:
        """Compute overall sequence diversity in alignment."""
        # Compute pairwise sequence identities
        num_seqs = len(alignment)
        if num_seqs < 2:
            return 0.0
            
        identities = []
        
        for i in range(num_seqs):
            for j in range(i + 1, num_seqs):
                seq_i = str(alignment[i].seq)
                seq_j = str(alignment[j].seq)
                
                # Compute identity
                matches = sum(1 for a, b in zip(seq_i, seq_j) if a == b and a != '-')
                length = sum(1 for a, b in zip(seq_i, seq_j) if a != '-' or b != '-')
                
                if length > 0:
                    identities.append(matches / length)
                    
        # Diversity is 1 - average identity
        return 1.0 - np.mean(identities) if identities else 0.0


class CoevolutionNetwork:
    """
    Builds and analyzes coevolution networks from MSA data.
    """
    
    def __init__(
        self,
        contact_threshold: float = 8.0,
        coevolution_threshold: float = 0.5
    ):
        self.contact_threshold = contact_threshold
        self.coevolution_threshold = coevolution_threshold
        
    def build_network(
        self,
        coevolution_scores: torch.Tensor,
        structure: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Build coevolution network from scores.
        
        Args:
            coevolution_scores: Coevolution matrix [seq_len, seq_len]
            structure: Optional 3D coordinates [seq_len, 3]
            
        Returns:
            Network analysis results
        """
        seq_len = coevolution_scores.shape[0]
        
        # Threshold coevolution scores
        adjacency = (coevolution_scores > self.coevolution_threshold).float()
        
        # Remove diagonal
        adjacency.fill_diagonal_(0)
        
        # Compute network metrics
        metrics = {
            'adjacency': adjacency,
            'degree': adjacency.sum(dim=1),
            'clustering_coefficient': self._compute_clustering_coefficient(adjacency),
            'betweenness_centrality': self._compute_betweenness_centrality(adjacency)
        }
        
        # If structure available, compute contact enrichment
        if structure is not None:
            metrics['contact_enrichment'] = self._compute_contact_enrichment(
                adjacency, structure
            )
            
        # Identify coevolution modules
        metrics['modules'] = self._identify_modules(adjacency)
        
        return metrics
        
    def _compute_clustering_coefficient(
        self,
        adjacency: torch.Tensor
    ) -> torch.Tensor:
        """Compute clustering coefficient for each node."""
        n = adjacency.shape[0]
        clustering = torch.zeros(n)
        
        for i in range(n):
            neighbors = torch.where(adjacency[i] > 0)[0]
            k = len(neighbors)
            
            if k >= 2:
                # Count edges between neighbors
                neighbor_edges = 0
                for j in range(len(neighbors)):
                    for l in range(j + 1, len(neighbors)):
                        if adjacency[neighbors[j], neighbors[l]] > 0:
                            neighbor_edges += 1
                            
                # Clustering coefficient
                clustering[i] = 2 * neighbor_edges / (k * (k - 1))
                
        return clustering
        
    def _compute_betweenness_centrality(
        self,
        adjacency: torch.Tensor
    ) -> torch.Tensor:
        """Compute betweenness centrality (simplified)."""
        # This is a simplified version
        # For full implementation, use networkx
        n = adjacency.shape[0]
        
        # Use degree centrality as approximation
        degree = adjacency.sum(dim=1)
        centrality = degree / (n - 1)
        
        return centrality
        
    def _compute_contact_enrichment(
        self,
        adjacency: torch.Tensor,
        structure: torch.Tensor
    ) -> float:
        """Compute enrichment of coevolution edges in structural contacts."""
        # Compute distance matrix
        distances = torch.cdist(structure, structure)
        contacts = (distances < self.contact_threshold).float()
        contacts.fill_diagonal_(0)
        
        # Compute enrichment
        coevolution_edges = adjacency.sum()
        contact_edges = contacts.sum()
        overlap = (adjacency * contacts).sum()
        
        # Expected overlap by chance
        total_possible = adjacency.shape[0] * (adjacency.shape[0] - 1) / 2
        expected = coevolution_edges * contact_edges / total_possible
        
        # Enrichment ratio
        enrichment = overlap / (expected + 1e-10)
        
        return enrichment.item()
        
    def _identify_modules(
        self,
        adjacency: torch.Tensor,
        min_module_size: int = 3
    ) -> List[List[int]]:
        """Identify coevolution modules using community detection."""
        # Simple connected components for now
        # For better results, use spectral clustering or Louvain
        
        n = adjacency.shape[0]
        visited = torch.zeros(n, dtype=torch.bool)
        modules = []
        
        for start in range(n):
            if not visited[start]:
                # BFS to find connected component
                module = []
                queue = [start]
                visited[start] = True
                
                while queue:
                    node = queue.pop(0)
                    module.append(node)
                    
                    # Add unvisited neighbors
                    neighbors = torch.where(adjacency[node] > 0)[0]
                    for neighbor in neighbors:
                        if not visited[neighbor]:
                            visited[neighbor] = True
                            queue.append(neighbor.item())
                            
                if len(module) >= min_module_size:
                    modules.append(module)
                    
        return modules