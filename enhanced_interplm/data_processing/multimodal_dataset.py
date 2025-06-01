# 多模态数据集 
# enhanced_interplm/data_processing/multimodal_dataset.py

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import h5py
from Bio import SeqIO
from Bio.PDB import PDBParser
import warnings
warnings.filterwarnings('ignore')


class MultiModalProteinDataset(Dataset):
    """
    A comprehensive dataset class for multi-modal protein data including:
    - Sequences
    - Multi-layer ESM embeddings
    - 3D structures
    - Evolutionary data (MSAs)
    - Functional annotations
    - Biophysical properties
    """
    
    def __init__(
        self,
        data_root: Path,
        split: str = 'train',
        layers: List[int] = [1, 2, 3, 4, 5, 6],
        max_seq_len: int = 1022,
        include_structures: bool = True,
        include_evolution: bool = True,
        include_annotations: bool = True,
        cache_embeddings: bool = True
    ):
        super().__init__()
        
        self.data_root = Path(data_root)
        self.split = split
        self.layers = layers
        self.max_seq_len = max_seq_len
        self.include_structures = include_structures
        self.include_evolution = include_evolution
        self.include_annotations = include_annotations
        self.cache_embeddings = cache_embeddings
        
        # Load metadata
        self.metadata = self._load_metadata()
        
        # Filter by split
        self.data = self.metadata[self.metadata['split'] == split].reset_index(drop=True)
        
        # Initialize caches
        self.embedding_cache = {} if cache_embeddings else None
        self.structure_cache = {}
        
        # Load annotation mappings
        if include_annotations:
            self.annotation_vocab = self._load_annotation_vocab()
            
        print(f"Loaded {len(self.data)} proteins for {split} split")
        
    def _load_metadata(self) -> pd.DataFrame:
        """Load protein metadata from CSV."""
        metadata_path = self.data_root / 'metadata.csv'
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
            
        return pd.read_csv(metadata_path)
        
    def _load_annotation_vocab(self) -> Dict[str, int]:
        """Load vocabulary for functional annotations."""
        vocab_path = self.data_root / 'annotation_vocab.json'
        if vocab_path.exists():
            import json
            with open(vocab_path, 'r') as f:
                return json.load(f)
        return {}
        
    def __len__(self) -> int:
        return len(self.data)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single protein sample with all modalities."""
        row = self.data.iloc[idx]
        protein_id = row['protein_id']
        
        # Load sequence
        sequence = row['sequence']
        if len(sequence) > self.max_seq_len:
            sequence = sequence[:self.max_seq_len]
            
        # Load multi-layer embeddings
        embeddings = self._load_embeddings(protein_id, len(sequence))
        
        # Initialize sample
        sample = {
            'protein_id': protein_id,
            'sequence': sequence,
            'embeddings': embeddings,
            'length': len(sequence)
        }
        
        # Load structure if available
        if self.include_structures and pd.notna(row.get('structure_path')):
            structure = self._load_structure(row['structure_path'], len(sequence))
            sample['structure'] = structure
            sample['has_structure'] = True
        else:
            sample['structure'] = torch.zeros(len(sequence), 3)
            sample['has_structure'] = False
            
        # Load evolutionary data
        if self.include_evolution and pd.notna(row.get('msa_path')):
            evolution_features = self._load_evolution_features(row['msa_path'], len(sequence))
            sample['evolution_features'] = evolution_features
        else:
            sample['evolution_features'] = torch.zeros(len(sequence), 21)  # 20 AAs + gap
            
        # Load annotations
        if self.include_annotations:
            annotations = self._load_annotations(row)
            sample.update(annotations)
            
        # Add biophysical properties
        biophysics = self._compute_biophysical_properties(sequence)
        sample['biophysics'] = biophysics
        
        return sample
        
    def _load_embeddings(self, protein_id: str, seq_len: int) -> torch.Tensor:
        """Load multi-layer ESM embeddings."""
        if self.cache_embeddings and protein_id in self.embedding_cache:
            return self.embedding_cache[protein_id]
            
        embeddings = []
        
        for layer in self.layers:
            emb_path = self.data_root / 'embeddings' / f'layer_{layer}' / f'{protein_id}.npy'
            
            if emb_path.exists():
                layer_emb = np.load(emb_path)
                if layer_emb.shape[0] > seq_len:
                    layer_emb = layer_emb[:seq_len]
                embeddings.append(torch.tensor(layer_emb, dtype=torch.float32))
            else:
                # Use zero embeddings if file not found
                embeddings.append(torch.zeros(seq_len, 320))  # ESM-2 8M dimension
                
        embeddings = torch.stack(embeddings)  # [num_layers, seq_len, embed_dim]
        
        if self.cache_embeddings:
            self.embedding_cache[protein_id] = embeddings
            
        return embeddings
        
    def _load_structure(self, structure_path: str, seq_len: int) -> torch.Tensor:
        """Load 3D structure coordinates."""
        if structure_path in self.structure_cache:
            return self.structure_cache[structure_path][:seq_len]
            
        full_path = self.data_root / 'structures' / structure_path
        
        if full_path.suffix == '.pdb':
            coords = self._parse_pdb(full_path)
        elif full_path.suffix == '.npz':
            coords = np.load(full_path)['coords']
        else:
            coords = np.zeros((seq_len, 3))
            
        coords_tensor = torch.tensor(coords[:seq_len], dtype=torch.float32)
        self.structure_cache[structure_path] = coords_tensor
        
        return coords_tensor
        
    def _parse_pdb(self, pdb_path: Path) -> np.ndarray:
        """Parse PDB file to extract CA coordinates."""
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', str(pdb_path))
        
        coords = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if 'CA' in residue:
                        coords.append(residue['CA'].get_coord())
                        
        return np.array(coords)
        
    def _load_evolution_features(self, msa_path: str, seq_len: int) -> torch.Tensor:
        """Load evolutionary features from MSA."""
        full_path = self.data_root / 'evolution' / msa_path
        
        if full_path.exists():
            # Load MSA and compute position-specific scoring matrix
            msa = self._read_msa(full_path)
            pssm = self._compute_pssm(msa)
            
            if pssm.shape[0] > seq_len:
                pssm = pssm[:seq_len]
                
            return torch.tensor(pssm, dtype=torch.float32)
        else:
            return torch.zeros(seq_len, 21)
            
    def _read_msa(self, msa_path: Path) -> List[str]:
        """Read MSA file."""
        sequences = []
        
        for record in SeqIO.parse(str(msa_path), 'fasta'):
            sequences.append(str(record.seq))
            
        return sequences
        
    def _compute_pssm(self, msa: List[str]) -> np.ndarray:
        """Compute position-specific scoring matrix from MSA."""
        if not msa:
            return np.zeros((1, 21))
            
        seq_len = len(msa[0])
        aa_to_idx = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY-')}
        
        pssm = np.zeros((seq_len, 21))
        
        for pos in range(seq_len):
            counts = np.zeros(21)
            for seq in msa:
                if pos < len(seq):
                    aa = seq[pos]
                    if aa in aa_to_idx:
                        counts[aa_to_idx[aa]] += 1
                        
            # Normalize to frequencies
            if counts.sum() > 0:
                pssm[pos] = counts / counts.sum()
                
        return pssm
        
    def _load_annotations(self, row: pd.Series) -> Dict[str, torch.Tensor]:
        """Load functional annotations."""
        annotations = {}
        
        # Secondary structure
        if pd.notna(row.get('secondary_structure')):
            ss = row['secondary_structure']
            ss_encoded = self._encode_secondary_structure(ss)
            annotations['secondary_structure'] = ss_encoded
        else:
            annotations['secondary_structure'] = torch.zeros(len(row['sequence']), 8)
            
        # GO terms
        if pd.notna(row.get('go_terms')):
            go_terms = row['go_terms'].split(';')
            go_encoded = self._encode_go_terms(go_terms)
            annotations['go_terms'] = go_encoded
        else:
            annotations['go_terms'] = torch.zeros(len(self.annotation_vocab.get('go', [])))
            
        # Domain annotations
        if pd.notna(row.get('domains')):
            domains = self._parse_domains(row['domains'], len(row['sequence']))
            annotations['domains'] = domains
        else:
            annotations['domains'] = torch.zeros(len(row['sequence']))
            
        return annotations
        
    def _encode_secondary_structure(self, ss_string: str) -> torch.Tensor:
        """Encode DSSP secondary structure string."""
        ss_to_idx = {
            'H': 0,  # Alpha helix
            'B': 1,  # Beta bridge
            'E': 2,  # Extended strand
            'G': 3,  # 3-10 helix
            'I': 4,  # Pi helix
            'T': 5,  # Turn
            'S': 6,  # Bend
            '-': 7   # Coil
        }
        
        encoded = torch.zeros(len(ss_string), 8)
        for i, ss in enumerate(ss_string):
            if ss in ss_to_idx:
                encoded[i, ss_to_idx[ss]] = 1
                
        return encoded
        
    def _encode_go_terms(self, go_terms: List[str]) -> torch.Tensor:
        """Encode GO terms as multi-hot vector."""
        if 'go' not in self.annotation_vocab:
            return torch.zeros(1)
            
        go_vocab = self.annotation_vocab['go']
        encoded = torch.zeros(len(go_vocab))
        
        for term in go_terms:
            if term in go_vocab:
                encoded[go_vocab[term]] = 1
                
        return encoded
        
    def _parse_domains(self, domain_string: str, seq_len: int) -> torch.Tensor:
        """Parse domain annotations."""
        domain_mask = torch.zeros(seq_len)
        
        # Format: "10-50:PF00001;60-120:PF00002"
        for domain in domain_string.split(';'):
            if ':' in domain:
                range_str, domain_id = domain.split(':')
                start, end = map(int, range_str.split('-'))
                domain_mask[start:end] = 1
                
        return domain_mask
        
    def _compute_biophysical_properties(self, sequence: str) -> torch.Tensor:
        """Compute biophysical properties for each residue."""
        # Hydrophobicity scale (Kyte-Doolittle)
        hydrophobicity = {
            'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
            'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
            'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
            'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
        }
        
        # Charge
        charge = {
            'D': -1, 'E': -1, 'K': 1, 'R': 1, 'H': 0.5
        }
        
        # Size (molecular weight in Daltons)
        size = {
            'A': 89, 'C': 121, 'D': 133, 'E': 147, 'F': 165,
            'G': 75, 'H': 155, 'I': 131, 'K': 146, 'L': 131,
            'M': 149, 'N': 132, 'P': 115, 'Q': 146, 'R': 174,
            'S': 105, 'T': 119, 'V': 117, 'W': 204, 'Y': 181
        }
        
        properties = torch.zeros(len(sequence), 3)
        
        for i, aa in enumerate(sequence):
            # Hydrophobicity (normalized to [-1, 1])
            properties[i, 0] = hydrophobicity.get(aa, 0) / 5.0
            
            # Charge
            properties[i, 1] = charge.get(aa, 0)
            
            # Size (normalized to [0, 1])
            properties[i, 2] = size.get(aa, 100) / 200.0
            
        return properties


class CollateFunction:
    """Custom collate function for multi-modal protein data."""
    
    def __init__(self, pad_value: float = 0.0):
        self.pad_value = pad_value
        
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate batch of protein samples."""
        # Find max sequence length in batch
        max_len = max(sample['length'] for sample in batch)
        
        # Initialize batch dictionary
        collated = {
            'protein_ids': [s['protein_id'] for s in batch],
            'sequences': [s['sequence'] for s in batch],
            'lengths': torch.tensor([s['length'] for s in batch]),
            'has_structure': torch.tensor([s['has_structure'] for s in batch])
        }
        
        # Pad embeddings
        embeddings = []
        for sample in batch:
            emb = sample['embeddings']
            if emb.shape[1] < max_len:
                pad_size = max_len - emb.shape[1]
                emb = F.pad(emb, (0, 0, 0, pad_size), value=self.pad_value)
            embeddings.append(emb)
        collated['embeddings'] = torch.stack(embeddings)
        
        # Pad structures
        structures = []
        for sample in batch:
            struct = sample['structure']
            if struct.shape[0] < max_len:
                pad_size = max_len - struct.shape[0]
                struct = F.pad(struct, (0, 0, 0, pad_size), value=self.pad_value)
            structures.append(struct)
        collated['structures'] = torch.stack(structures)
        
        # Pad other features similarly
        for key in ['evolution_features', 'biophysics', 'secondary_structure', 'domains']:
            if key in batch[0]:
                features = []
                for sample in batch:
                    feat = sample[key]
                    if feat.shape[0] < max_len:
                        pad_size = max_len - feat.shape[0]
                        padding = (0, 0, 0, pad_size) if feat.dim() == 2 else (0, pad_size)
                        feat = F.pad(feat, padding, value=self.pad_value)
                    features.append(feat)
                collated[key] = torch.stack(features)
                
        # Handle GO terms (not sequence-length dependent)
        if 'go_terms' in batch[0]:
            collated['go_terms'] = torch.stack([s['go_terms'] for s in batch])
            
        return collated


def create_protein_dataloaders(
    data_root: Path,
    batch_size: int = 32,
    num_workers: int = 4,
    layers: List[int] = [1, 2, 3, 4, 5, 6],
    include_structures: bool = True,
    include_evolution: bool = True,
    include_annotations: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders."""
    
    # Create datasets
    train_dataset = MultiModalProteinDataset(
        data_root=data_root,
        split='train',
        layers=layers,
        include_structures=include_structures,
        include_evolution=include_evolution,
        include_annotations=include_annotations
    )
    
    val_dataset = MultiModalProteinDataset(
        data_root=data_root,
        split='val',
        layers=layers,
        include_structures=include_structures,
        include_evolution=include_evolution,
        include_annotations=include_annotations
    )
    
    test_dataset = MultiModalProteinDataset(
        data_root=data_root,
        split='test',
        layers=layers,
        include_structures=include_structures,
        include_evolution=include_evolution,
        include_annotations=include_annotations
    )
    
    # Create collate function
    collate_fn = CollateFunction()
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader