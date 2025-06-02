# 结构数据处理 
# enhanced_interplm/data_processing/structure_processor.py

"""
Processing and analysis of protein 3D structures for Enhanced InterPLM.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import pandas as pd
from Bio.PDB import PDBParser, DSSP, PPBuilder, NeighborSearch
from Bio.PDB.Structure import Structure
from Bio.PDB.Chain import Chain
import warnings
from scipy.spatial.distance import cdist
from scipy.spatial import distance_matrix
import logging

warnings.filterwarnings('ignore', category=UserWarning)
logger = logging.getLogger(__name__)


class StructureProcessor:
    """
    Processes protein 3D structures and extracts structural features.
    """
    
    def __init__(
        self,
        contact_threshold: float = 8.0,
        neighbor_threshold: float = 10.0,
        dssp_path: Optional[str] = None
    ):
        """
        Initialize structure processor.
        
        Args:
            contact_threshold: Distance threshold for contacts (Angstroms)
            neighbor_threshold: Distance threshold for neighbors
            dssp_path: Path to DSSP executable (if None, will try to find it)
        """
        self.contact_threshold = contact_threshold
        self.neighbor_threshold = neighbor_threshold
        self.dssp_path = dssp_path
        self.parser = PDBParser(QUIET=True)
        
    def process_structure(
        self,
        structure_path: Union[str, Path],
        chain_id: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Process a PDB structure and extract features.
        
        Args:
            structure_path: Path to PDB file
            chain_id: Specific chain to process (if None, use first chain)
            
        Returns:
            Dictionary containing structural features
        """
        # Parse structure
        structure = self.parser.get_structure('protein', str(structure_path))
        
        # Get chain
        if chain_id is None:
            chain = list(structure.get_chains())[0]
        else:
            chain = structure[0][chain_id]
            
        # Extract features
        features = {
            'coordinates': self._extract_coordinates(chain),
            'distances': self._compute_distance_matrix(chain),
            'contacts': self._compute_contact_map(chain),
            'secondary_structure': self._extract_secondary_structure(structure, chain),
            'phi_psi': self._extract_phi_psi_angles(chain),
            'sasa': self._compute_sasa(structure),
            'local_geometry': self._extract_local_geometry(chain),
            'residue_depth': self._compute_residue_depth(chain)
        }
        
        # Add derived features
        features['contact_order'] = self._compute_contact_order(features['contacts'])
        features['structural_motifs'] = self._identify_structural_motifs(features)
        
        return features
        
    def _extract_coordinates(self, chain: Chain) -> torch.Tensor:
        """Extract CA coordinates from chain."""
        coords = []
        
        for residue in chain:
            if residue.id[0] == ' ':  # Standard residue
                if 'CA' in residue:
                    coords.append(residue['CA'].get_coord())
                else:
                    # Use centroid if CA missing
                    atoms = [atom.get_coord() for atom in residue.get_atoms()]
                    coords.append(np.mean(atoms, axis=0))
                    
        return torch.tensor(np.array(coords), dtype=torch.float32)
        
    def _compute_distance_matrix(self, chain: Chain) -> torch.Tensor:
        """Compute pairwise distance matrix."""
        coords = self._extract_coordinates(chain)
        distances = torch.cdist(coords, coords)
        return distances
        
    def _compute_contact_map(self, chain: Chain) -> torch.Tensor:
        """Compute residue contact map."""
        distances = self._compute_distance_matrix(chain)
        contacts = (distances < self.contact_threshold).float()
        
        # Remove diagonal and nearby residues
        seq_len = contacts.shape[0]
        for i in range(seq_len):
            for j in range(max(0, i-2), min(seq_len, i+3)):
                contacts[i, j] = 0
                
        return contacts
        
    def _extract_secondary_structure(
        self,
        structure: Structure,
        chain: Chain
    ) -> torch.Tensor:
        """Extract secondary structure using DSSP."""
        try:
            if self.dssp_path:
                dssp = DSSP(structure[0], str(structure.id), dssp=self.dssp_path)
            else:
                dssp = DSSP(structure[0], str(structure.id))
                
            # Map DSSP codes to 8-state
            dssp_to_idx = {
                'H': 0,  # Alpha helix
                'B': 1,  # Beta bridge
                'E': 2,  # Extended strand
                'G': 3,  # 3-10 helix
                'I': 4,  # Pi helix
                'T': 5,  # Turn
                'S': 6,  # Bend
                '-': 7   # Coil
            }
            
            ss_sequence = []
            for residue in chain:
                if residue.id[0] == ' ':
                    key = (chain.id, residue.id)
                    if key in dssp:
                        ss_code = dssp[key][2]
                        ss_idx = dssp_to_idx.get(ss_code, 7)
                    else:
                        ss_idx = 7  # Default to coil
                    ss_sequence.append(ss_idx)
                    
            # One-hot encode
            ss_tensor = torch.zeros(len(ss_sequence), 8)
            for i, idx in enumerate(ss_sequence):
                ss_tensor[i, idx] = 1
                
            return ss_tensor
            
        except Exception as e:
            logger.warning(f"DSSP failed: {e}. Using default coil assignment.")
            seq_len = len(list(chain.get_residues()))
            ss_tensor = torch.zeros(seq_len, 8)
            ss_tensor[:, 7] = 1  # All coil
            return ss_tensor
            
    def _extract_phi_psi_angles(self, chain: Chain) -> torch.Tensor:
        """Extract phi and psi angles."""
        ppb = PPBuilder()
        angles = []
        
        for pp in ppb.build_peptides(chain):
            for residue, phi_psi in zip(pp, pp.get_phi_psi_list()):
                phi, psi = phi_psi
                
                # Convert None to 0
                phi = phi if phi is not None else 0.0
                psi = psi if psi is not None else 0.0
                
                angles.append([phi, psi])
                
        return torch.tensor(angles, dtype=torch.float32)
        
    def _compute_sasa(self, structure: Structure) -> torch.Tensor:
        """Compute solvent accessible surface area."""
        # Simplified SASA calculation
        # For full implementation, use FreeSASA or similar
        
        atoms = list(structure.get_atoms())
        coords = np.array([atom.get_coord() for atom in atoms])
        
        # Use neighbor search for efficiency
        ns = NeighborSearch(atoms)
        
        sasa_per_residue = []
        
        for chain in structure.get_chains():
            for residue in chain:
                if residue.id[0] == ' ':
                    # Count exposed atoms (simplified)
                    exposed_count = 0
                    total_atoms = 0
                    
                    for atom in residue:
                        neighbors = ns.search(atom.get_coord(), 5.0)
                        # Atom is exposed if few neighbors
                        if len(neighbors) < 10:
                            exposed_count += 1
                        total_atoms += 1
                        
                    # Normalized exposure
                    exposure = exposed_count / max(total_atoms, 1)
                    sasa_per_residue.append(exposure)
                    
        return torch.tensor(sasa_per_residue, dtype=torch.float32)
        
    def _extract_local_geometry(self, chain: Chain) -> Dict[str, torch.Tensor]:
        """Extract local geometric features."""
        coords = self._extract_coordinates(chain).numpy()
        seq_len = len(coords)
        
        # Local curvature
        curvature = np.zeros(seq_len)
        for i in range(1, seq_len - 1):
            # Angle between consecutive CA-CA vectors
            v1 = coords[i] - coords[i-1]
            v2 = coords[i+1] - coords[i]
            
            v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
            v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)
            
            # Curvature as angle
            cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1, 1)
            curvature[i] = np.arccos(cos_angle)
            
        # Local density (number of residues within threshold)
        density = np.zeros(seq_len)
        dist_matrix = cdist(coords, coords)
        
        for i in range(seq_len):
            density[i] = np.sum(dist_matrix[i] < self.neighbor_threshold) - 1
            
        # Local compactness
        compactness = np.zeros(seq_len)
        for i in range(seq_len):
            neighbors = np.where(dist_matrix[i] < self.neighbor_threshold)[0]
            if len(neighbors) > 1:
                neighbor_coords = coords[neighbors]
                # Radius of gyration of local neighborhood
                center = neighbor_coords.mean(axis=0)
                rg = np.sqrt(np.mean(np.sum((neighbor_coords - center)**2, axis=1)))
                compactness[i] = 1.0 / (1.0 + rg)
                
        return {
            'curvature': torch.tensor(curvature, dtype=torch.float32),
            'density': torch.tensor(density, dtype=torch.float32),
            'compactness': torch.tensor(compactness, dtype=torch.float32)
        }
        
    def _compute_residue_depth(self, chain: Chain) -> torch.Tensor:
        """Compute residue depth (distance from surface)."""
        coords = self._extract_coordinates(chain).numpy()
        seq_len = len(coords)
        
        # Compute centroid
        centroid = coords.mean(axis=0)
        
        # Distance from centroid
        distances_from_center = np.linalg.norm(coords - centroid, axis=1)
        
        # Normalize by maximum distance
        max_dist = distances_from_center.max()
        normalized_depth = 1.0 - (distances_from_center / max_dist)
        
        return torch.tensor(normalized_depth, dtype=torch.float32)
        
    def _compute_contact_order(self, contacts: torch.Tensor) -> float:
        """Compute relative contact order."""
        seq_len = contacts.shape[0]
        
        total_contacts = 0
        total_separation = 0
        
        for i in range(seq_len):
            for j in range(i + 3, seq_len):  # Skip nearby residues
                if contacts[i, j] > 0:
                    total_contacts += 1
                    total_separation += abs(j - i)
                    
        if total_contacts > 0:
            avg_separation = total_separation / total_contacts
            relative_co = avg_separation / seq_len
            return relative_co
        else:
            return 0.0
            
    def _identify_structural_motifs(
        self,
        features: Dict[str, torch.Tensor]
    ) -> Dict[str, List[Tuple[int, int]]]:
        """Identify common structural motifs."""
        motifs = {
            'beta_hairpins': [],
            'helix_pairs': [],
            'beta_sheets': [],
            'loops': []
        }
        
        ss = features['secondary_structure'].argmax(dim=-1)
        contacts = features['contacts']
        
        seq_len = len(ss)
        
        # Beta hairpins (antiparallel beta strands connected by turn)
        for i in range(seq_len - 5):
            # Look for E-T-E pattern
            if (ss[i] == 2 and  # Extended strand
                ss[i+1:i+3].max() >= 5 and  # Turn or bend
                ss[i+3] == 2 and  # Extended strand
                contacts[i, i+3] > 0):  # Contact between strands
                motifs['beta_hairpins'].append((i, i+3))
                
        # Helix pairs (helices in contact)
        helix_regions = self._find_continuous_regions(ss, 0)  # Alpha helix
        
        for i, (start1, end1) in enumerate(helix_regions):
            for j, (start2, end2) in enumerate(helix_regions[i+1:], i+1):
                # Check if helices are in contact
                if contacts[start1:end1, start2:end2].sum() > 0:
                    motifs['helix_pairs'].append(((start1, end1), (start2, end2)))
                    
        # Beta sheets (multiple strands in contact)
        strand_regions = self._find_continuous_regions(ss, 2)  # Extended strand
        
        if len(strand_regions) >= 2:
            # Group strands that are in contact
            strand_groups = []
            used = set()
            
            for i, (start1, end1) in enumerate(strand_regions):
                if i in used:
                    continue
                    
                group = [(start1, end1)]
                used.add(i)
                
                for j, (start2, end2) in enumerate(strand_regions):
                    if j != i and j not in used:
                        if contacts[start1:end1, start2:end2].sum() > 0:
                            group.append((start2, end2))
                            used.add(j)
                            
                if len(group) >= 2:
                    motifs['beta_sheets'].append(group)
                    
        # Loops (coil regions between secondary structures)
        coil_regions = self._find_continuous_regions(ss, 7)  # Coil
        
        for start, end in coil_regions:
            if 3 <= end - start <= 12:  # Typical loop length
                motifs['loops'].append((start, end))
                
        return motifs
        
    def _find_continuous_regions(
        self,
        sequence: torch.Tensor,
        target_value: int
    ) -> List[Tuple[int, int]]:
        """Find continuous regions of a specific value."""
        regions = []
        in_region = False
        start = 0
        
        for i, val in enumerate(sequence):
            if val == target_value:
                if not in_region:
                    start = i
                    in_region = True
            else:
                if in_region:
                    regions.append((start, i))
                    in_region = False
                    
        if in_region:
            regions.append((start, len(sequence)))
            
        return regions


class StructuralFeatureExtractor:
    """
    Extracts advanced structural features for model input.
    """
    
    def __init__(self, num_radial_bins: int = 20, num_angular_bins: int = 18):
        self.num_radial_bins = num_radial_bins
        self.num_angular_bins = num_angular_bins
        self.max_radius = 20.0  # Angstroms
        
    def extract_features(
        self,
        coordinates: torch.Tensor,
        sequence: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Extract comprehensive structural features.
        
        Args:
            coordinates: 3D coordinates [seq_len, 3]
            sequence: Optional amino acid sequence
            
        Returns:
            Dictionary of structural features
        """
        features = {}
        
        # Basic geometric features
        features.update(self._extract_geometric_features(coordinates))
        
        # Radial distribution functions
        features['rdf'] = self._compute_rdf(coordinates)
        
        # Angular distribution functions
        features['adf'] = self._compute_adf(coordinates)
        
        # Moment invariants
        features['moments'] = self._compute_moment_invariants(coordinates)
        
        # If sequence provided, add sequence-structure features
        if sequence is not None:
            features.update(self._extract_sequence_structure_features(
                coordinates, sequence
            ))
            
        return features
        
    def _extract_geometric_features(
        self,
        coordinates: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Extract basic geometric features."""
        seq_len = len(coordinates)
        
        # End-to-end distance
        end_to_end = torch.norm(coordinates[-1] - coordinates[0])
        
        # Radius of gyration
        center = coordinates.mean(dim=0)
        rg = torch.sqrt(torch.mean(torch.sum((coordinates - center)**2, dim=1)))
        
        # Asphericity and shape descriptors
        inertia_tensor = self._compute_inertia_tensor(coordinates)
        eigenvalues = torch.linalg.eigvals(inertia_tensor).real
        eigenvalues = torch.sort(eigenvalues)[0]
        
        asphericity = 1.5 * (eigenvalues[2]**2 - eigenvalues.mean()**2) / eigenvalues.mean()**2
        acylindricity = eigenvalues[1] - eigenvalues[0]
        
        # Local backbone geometry
        bond_lengths = torch.norm(coordinates[1:] - coordinates[:-1], dim=1)
        bond_angles = []
        
        for i in range(1, seq_len - 1):
            v1 = coordinates[i] - coordinates[i-1]
            v2 = coordinates[i+1] - coordinates[i]
            
            cos_angle = torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2) + 1e-8)
            angle = torch.acos(torch.clamp(cos_angle, -1, 1))
            bond_angles.append(angle)
            
        bond_angles = torch.tensor(bond_angles)
        
        # Dihedral angles
        dihedral_angles = []
        
        for i in range(seq_len - 3):
            dihedral = self._compute_dihedral(
                coordinates[i],
                coordinates[i+1],
                coordinates[i+2],
                coordinates[i+3]
            )
            dihedral_angles.append(dihedral)
            
        dihedral_angles = torch.tensor(dihedral_angles)
        
        return {
            'end_to_end_distance': end_to_end,
            'radius_of_gyration': rg,
            'asphericity': torch.tensor(asphericity),
            'acylindricity': torch.tensor(acylindricity),
            'bond_lengths': bond_lengths,
            'bond_angles': bond_angles,
            'dihedral_angles': dihedral_angles
        }
        
    def _compute_rdf(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Compute radial distribution function."""
        distances = torch.cdist(coordinates, coordinates)
        
        # Remove diagonal
        mask = ~torch.eye(len(coordinates), dtype=bool)
        distances = distances[mask]
        
        # Bin distances
        bins = torch.linspace(0, self.max_radius, self.num_radial_bins + 1)
        hist = torch.histc(distances, bins=self.num_radial_bins, min=0, max=self.max_radius)
        
        # Normalize by shell volume
        dr = self.max_radius / self.num_radial_bins
        shell_volumes = 4 * np.pi * (bins[1:]**2) * dr
        
        rdf = hist / shell_volumes
        rdf = rdf / rdf.sum()  # Normalize
        
        return rdf
        
    def _compute_adf(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Compute angular distribution function."""
        seq_len = len(coordinates)
        angles = []
        
        # Sample triplets of residues
        for i in range(seq_len):
            for j in range(i + 1, min(i + 10, seq_len)):
                for k in range(j + 1, min(j + 10, seq_len)):
                    # Compute angle i-j-k
                    v1 = coordinates[i] - coordinates[j]
                    v2 = coordinates[k] - coordinates[j]
                    
                    cos_angle = torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2) + 1e-8)
                    angle = torch.acos(torch.clamp(cos_angle, -1, 1))
                    angles.append(angle)
                    
        angles = torch.tensor(angles)
        
        # Bin angles
        bins = torch.linspace(0, np.pi, self.num_angular_bins + 1)
        adf = torch.histc(angles, bins=self.num_angular_bins, min=0, max=np.pi)
        adf = adf / adf.sum()  # Normalize
        
        return adf
        
    def _compute_moment_invariants(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Compute geometric moment invariants."""
        # Center coordinates
        centered = coordinates - coordinates.mean(dim=0)
        
        # Compute moments up to order 3
        moments = []
        
        for order in range(1, 4):
            for i in range(order + 1):
                for j in range(order + 1 - i):
                    k = order - i - j
                    
                    moment = torch.mean(
                        centered[:, 0]**i * centered[:, 1]**j * centered[:, 2]**k
                    )
                    moments.append(moment)
                    
        return torch.tensor(moments)
        
    def _compute_inertia_tensor(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Compute inertia tensor."""
        centered = coordinates - coordinates.mean(dim=0)
        
        I = torch.zeros(3, 3)
        
        for coord in centered:
            x, y, z = coord
            I[0, 0] += y**2 + z**2
            I[1, 1] += x**2 + z**2
            I[2, 2] += x**2 + y**2
            I[0, 1] -= x * y
            I[0, 2] -= x * z
            I[1, 2] -= y * z
            
        I[1, 0] = I[0, 1]
        I[2, 0] = I[0, 2]
        I[2, 1] = I[1, 2]
        
        return I / len(coordinates)
        
    def _compute_dihedral(
        self,
        p1: torch.Tensor,
        p2: torch.Tensor,
        p3: torch.Tensor,
        p4: torch.Tensor
    ) -> torch.Tensor:
        """Compute dihedral angle between four points."""
        # Vectors between points
        v1 = p2 - p1
        v2 = p3 - p2
        v3 = p4 - p3
        
        # Normal vectors to planes
        n1 = torch.cross(v1, v2)
        n2 = torch.cross(v2, v3)
        
        # Normalize
        n1 = n1 / (torch.norm(n1) + 1e-8)
        n2 = n2 / (torch.norm(n2) + 1e-8)
        
        # Compute angle
        cos_angle = torch.dot(n1, n2)
        angle = torch.acos(torch.clamp(cos_angle, -1, 1))
        
        # Determine sign
        if torch.dot(torch.cross(n1, n2), v2) < 0:
            angle = -angle
            
        return angle
        
    def _extract_sequence_structure_features(
        self,
        coordinates: torch.Tensor,
        sequence: str
    ) -> Dict[str, torch.Tensor]:
        """Extract features combining sequence and structure."""
        # Hydrophobic cluster analysis
        hydrophobic_aas = set('AILMFVWY')
        hydrophobic_mask = torch.tensor([aa in hydrophobic_aas for aa in sequence])
        
        # Find hydrophobic clusters
        distances = torch.cdist(coordinates, coordinates)
        hydrophobic_contacts = distances * hydrophobic_mask.unsqueeze(0) * hydrophobic_mask.unsqueeze(1)
        hydrophobic_contacts = (hydrophobic_contacts < 8.0) & (hydrophobic_contacts > 0)
        
        # Cluster statistics
        cluster_sizes = []
        visited = torch.zeros(len(sequence), dtype=torch.bool)
        
        for i in range(len(sequence)):
            if hydrophobic_mask[i] and not visited[i]:
                # BFS to find cluster
                cluster = [i]
                queue = [i]
                visited[i] = True
                
                while queue:
                    node = queue.pop(0)
                    neighbors = torch.where(hydrophobic_contacts[node])[0]
                    
                    for neighbor in neighbors:
                        if not visited[neighbor]:
                            visited[neighbor] = True
                            cluster.append(neighbor.item())
                            queue.append(neighbor.item())
                            
                cluster_sizes.append(len(cluster))
                
        # Compute features
        features = {
            'num_hydrophobic_clusters': torch.tensor(len(cluster_sizes)),
            'avg_cluster_size': torch.tensor(np.mean(cluster_sizes)) if cluster_sizes else torch.tensor(0.0),
            'max_cluster_size': torch.tensor(max(cluster_sizes)) if cluster_sizes else torch.tensor(0),
            'hydrophobic_core_fraction': hydrophobic_mask.float().mean()
        }
        
        return features