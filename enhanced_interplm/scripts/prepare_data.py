#!/usr/bin/env python3
"""
Prepare protein data for Enhanced InterPLM training.

This script:
1. Downloads protein sequences from UniProt
2. Extracts ESM embeddings
3. Downloads AlphaFold structures
4. Computes MSAs and evolutionary features
5. Organizes data into training format
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import requests
import torch
from Bio import SeqIO, SwissProt
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from enhanced_interplm.esm import extract_multilayer_embeddings
from enhanced_interplm.data_processing import (
    EvolutionaryDataProcessor,
    StructureProcessor
)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProteinDataPreparer:
    """Prepares protein data for Enhanced InterPLM."""
    
    def __init__(
        self,
        output_dir: Path,
        esm_model: str = "esm2_t6_8M_UR50D",
        max_seq_length: int = 1022
    ):
        self.output_dir = Path(output_dir)
        self.esm_model = esm_model
        self.max_seq_length = max_seq_length
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'sequences').mkdir(exist_ok=True)
        (self.output_dir / 'embeddings').mkdir(exist_ok=True)
        (self.output_dir / 'structures').mkdir(exist_ok=True)
        (self.output_dir / 'evolution').mkdir(exist_ok=True)
        (self.output_dir / 'annotations').mkdir(exist_ok=True)
        
        # Initialize processors
        self.structure_processor = StructureProcessor()
        self.evolution_processor = EvolutionaryDataProcessor()
        
    def prepare_dataset(
        self,
        protein_list_file: Optional[Path] = None,
        uniprot_query: Optional[str] = None,
        max_proteins: int = 1000
    ):
        """Main preparation pipeline."""
        
        # Step 1: Get protein list
        logger.info("Getting protein list...")
        if protein_list_file:
            proteins = self._load_protein_list(protein_list_file)
        elif uniprot_query:
            proteins = self._query_uniprot(uniprot_query, max_proteins)
        else:
            raise ValueError("Either protein_list_file or uniprot_query must be provided")
            
        logger.info(f"Processing {len(proteins)} proteins")
        
        # Step 2: Download sequences and annotations
        logger.info("Downloading sequences and annotations...")
        protein_data = self._download_protein_data(proteins)
        
        # Step 3: Extract ESM embeddings
        logger.info("Extracting ESM embeddings...")
        self._extract_embeddings(protein_data)
        
        # Step 4: Download and process structures
        logger.info("Processing structures...")
        self._process_structures(protein_data)
        
        # Step 5: Compute evolutionary features
        logger.info("Computing evolutionary features...")
        self._compute_evolution_features(protein_data)
        
        # Step 6: Create metadata
        logger.info("Creating metadata...")
        self._create_metadata(protein_data)
        
        # Step 7: Split dataset
        logger.info("Creating train/val/test splits...")
        self._create_splits(protein_data)
        
        logger.info("Data preparation complete!")
        
    def _load_protein_list(self, file_path: Path) -> List[str]:
        """Load protein IDs from file."""
        with open(file_path, 'r') as f:
            proteins = [line.strip() for line in f if line.strip()]
        return proteins
        
    def _query_uniprot(self, query: str, max_proteins: int) -> List[str]:
        """Query UniProt for proteins."""
        url = "https://rest.uniprot.org/uniprotkb/search"
        params = {
            'query': query,
            'format': 'list',
            'size': max_proteins
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        proteins = response.text.strip().split('\n')
        return proteins[:max_proteins]
        
    def _download_protein_data(self, proteins: List[str]) -> Dict:
        """Download protein sequences and annotations."""
        protein_data = {}
        
        for protein_id in tqdm(proteins, desc="Downloading proteins"):
            try:
                # Get UniProt entry
                url = f"https://rest.uniprot.org/uniprotkb/{protein_id}.txt"
                response = requests.get(url)
                response.raise_for_status()
                
                # Parse SwissProt format
                from io import StringIO
                record = SwissProt.read(StringIO(response.text))
                
                # Extract data
                sequence = record.sequence
                
                # Skip if too long
                if len(sequence) > self.max_seq_length:
                    logger.warning(f"Skipping {protein_id}: sequence too long ({len(sequence)})")
                    continue
                    
                protein_data[protein_id] = {
                    'sequence': sequence,
                    'length': len(sequence),
                    'organism': record.organism,
                    'gene_name': record.gene_name,
                    'description': record.description,
                    'keywords': record.keywords,
                    'go_terms': self._extract_go_terms(record),
                    'ec_number': self._extract_ec_number(record),
                    'pfam_domains': self._extract_pfam_domains(record),
                    'secondary_structure': self._extract_secondary_structure(record)
                }
                
                # Save sequence
                seq_file = self.output_dir / 'sequences' / f'{protein_id}.fasta'
                with open(seq_file, 'w') as f:
                    f.write(f">{protein_id}\n{sequence}\n")
                    
            except Exception as e:
                logger.error(f"Error processing {protein_id}: {e}")
                continue
                
        return protein_data
        
    def _extract_embeddings(self, protein_data: Dict):
        """Extract multi-layer ESM embeddings."""
        sequences = []
        protein_ids = []
        
# 继续 _extract_embeddings 方法
        for protein_id, data in protein_data.items():
            sequences.append(data['sequence'])
            protein_ids.append(protein_id)
            
        # Extract embeddings in batches
        logger.info(f"Extracting embeddings for {len(sequences)} sequences")
        
        # Define layers to extract based on model
        if "8M" in self.esm_model:
            layers = [1, 2, 3, 4, 5, 6]
        elif "35M" in self.esm_model:
            layers = [1, 3, 6, 9, 12]
        elif "150M" in self.esm_model:
            layers = list(range(1, 31, 5))
        elif "650M" in self.esm_model:
            layers = list(range(1, 34, 5))
        else:
            layers = [1, 2, 3, 4, 5, 6]
            
        # Extract embeddings
        embeddings_dict = extract_multilayer_embeddings(
            sequences=sequences,
            model_name=self.esm_model,
            layers=layers,
            batch_size=8,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Save embeddings for each layer
        for layer_idx in layers:
            layer_dir = self.output_dir / 'embeddings' / f'layer_{layer_idx}'
            layer_dir.mkdir(exist_ok=True)
            
            for i, protein_id in enumerate(protein_ids):
                embeddings = embeddings_dict['embeddings'][i, layer_idx - 1]
                
                # Save as numpy array
                emb_file = layer_dir / f'{protein_id}.npy'
                np.save(emb_file, embeddings.cpu().numpy())
                
                # Update protein data
                protein_data[protein_id]['embeddings_computed'] = True
                
    def _process_structures(self, protein_data: Dict):
        """Download and process protein structures."""
        for protein_id, data in tqdm(protein_data.items(), desc="Processing structures"):
            try:
                # Try to download AlphaFold structure
                af_url = f"https://alphafold.ebi.ac.uk/files/AF-{protein_id}-F1-model_v4.pdb"
                response = requests.get(af_url)
                
                if response.status_code == 200:
                    # Save PDB file
                    pdb_file = self.output_dir / 'structures' / f'{protein_id}.pdb'
                    with open(pdb_file, 'w') as f:
                        f.write(response.text)
                        
                    # Process structure
                    struct_features = self.structure_processor.process_structure(pdb_file)
                    
                    # Save processed features
                    struct_file = self.output_dir / 'structures' / f'{protein_id}_features.npz'
                    np.savez_compressed(
                        struct_file,
                        coordinates=struct_features['coordinates'].cpu().numpy(),
                        distances=struct_features['distances'].cpu().numpy(),
                        contacts=struct_features['contacts'].cpu().numpy(),
                        secondary_structure=struct_features['secondary_structure'].cpu().numpy()
                    )
                    
                    protein_data[protein_id]['structure_available'] = True
                    protein_data[protein_id]['structure_path'] = f'{protein_id}.pdb'
                    
            except Exception as e:
                logger.warning(f"Could not process structure for {protein_id}: {e}")
                protein_data[protein_id]['structure_available'] = False
                
    def _compute_evolution_features(self, protein_data: Dict):
        """Compute evolutionary features from MSAs."""
        # This would typically involve running HHblits or similar
        # For now, we'll create placeholder evolutionary features
        
        for protein_id, data in tqdm(protein_data.items(), desc="Computing evolution features"):
            try:
                # In practice, you would:
                # 1. Run HHblits against UniRef30 or similar database
                # 2. Parse the resulting MSA
                # 3. Compute conservation, coevolution, etc.
                
                # Placeholder: create mock evolutionary features
                seq_len = len(data['sequence'])
                
                # Mock conservation scores
                conservation = np.random.beta(2, 1, seq_len)  # Skewed towards conserved
                
                # Mock PSSM
                pssm = np.random.randn(seq_len, 21)
                pssm = F.softmax(torch.tensor(pssm), dim=-1).numpy()
                
                # Save evolution features
                evo_file = self.output_dir / 'evolution' / f'{protein_id}.npz'
                np.savez_compressed(
                    evo_file,
                    conservation=conservation,
                    pssm=pssm,
                    num_sequences=100,  # Mock value
                    sequence_diversity=0.7  # Mock value
                )
                
                protein_data[protein_id]['evolution_computed'] = True
                
            except Exception as e:
                logger.warning(f"Could not compute evolution features for {protein_id}: {e}")
                protein_data[protein_id]['evolution_computed'] = False
                
    def _extract_go_terms(self, record) -> List[str]:
        """Extract GO terms from SwissProt record."""
        go_terms = []
        for ref in record.cross_references:
            if ref[0] == 'GO':
                go_terms.append(ref[1])
        return go_terms
        
    def _extract_ec_number(self, record) -> Optional[str]:
        """Extract EC number from SwissProt record."""
        for comment in record.comments:
            if comment.startswith('EC='):
                return comment.split('=')[1].strip()
        return None
        
    def _extract_pfam_domains(self, record) -> List[Dict]:
        """Extract Pfam domain annotations."""
        domains = []
        for feature in record.features:
            if feature.type == 'DOMAIN' and 'Pfam' in feature.qualifiers.get('db_xref', []):
                domains.append({
                    'start': feature.location.start,
                    'end': feature.location.end,
                    'pfam_id': feature.qualifiers.get('db_xref', [''])[0],
                    'description': feature.qualifiers.get('note', [''])[0]
                })
        return domains
        
    def _extract_secondary_structure(self, record) -> str:
        """Extract secondary structure if available."""
        # This would typically come from DSSP or predicted
        # For now, return empty string
        return ""
        
    def _create_metadata(self, protein_data: Dict):
        """Create metadata file for the dataset."""
        metadata_rows = []
        
        for protein_id, data in protein_data.items():
            row = {
                'protein_id': protein_id,
                'sequence': data['sequence'],
                'length': data['length'],
                'organism': data.get('organism', ''),
                'gene_name': data.get('gene_name', ''),
                'description': data.get('description', ''),
                'keywords': ';'.join(data.get('keywords', [])),
                'go_terms': ';'.join(data.get('go_terms', [])),
                'ec_number': data.get('ec_number', ''),
                'has_structure': data.get('structure_available', False),
                'has_evolution': data.get('evolution_computed', False),
                'has_embeddings': data.get('embeddings_computed', False)
            }
            
            # Add domain information
            if 'pfam_domains' in data and data['pfam_domains']:
                domain_strings = []
                for domain in data['pfam_domains']:
                    domain_strings.append(f"{domain['start']}-{domain['end']}:{domain['pfam_id']}")
                row['domains'] = ';'.join(domain_strings)
            else:
                row['domains'] = ''
                
            metadata_rows.append(row)
            
        # Create DataFrame and save
        metadata_df = pd.DataFrame(metadata_rows)
        metadata_file = self.output_dir / 'metadata.csv'
        metadata_df.to_csv(metadata_file, index=False)
        
        # Save annotation vocabulary
        self._create_annotation_vocab(protein_data)
        
    def _create_annotation_vocab(self, protein_data: Dict):
        """Create vocabulary for annotations."""
        vocab = {
            'go': {},
            'ec': {},
            'pfam': {},
            'keywords': {}
        }
        
        # Collect all unique annotations
        all_go_terms = set()
        all_ec_numbers = set()
        all_pfam_ids = set()
        all_keywords = set()
        
        for data in protein_data.values():
            all_go_terms.update(data.get('go_terms', []))
            if data.get('ec_number'):
                all_ec_numbers.add(data['ec_number'])
            for domain in data.get('pfam_domains', []):
                all_pfam_ids.add(domain['pfam_id'])
            all_keywords.update(data.get('keywords', []))
            
        # Create vocabularies
        for i, term in enumerate(sorted(all_go_terms)):
            vocab['go'][term] = i
            
        for i, ec in enumerate(sorted(all_ec_numbers)):
            vocab['ec'][ec] = i
            
        for i, pfam in enumerate(sorted(all_pfam_ids)):
            vocab['pfam'][pfam] = i
            
        for i, keyword in enumerate(sorted(all_keywords)):
            vocab['keywords'][keyword] = i
            
        # Save vocabulary
        vocab_file = self.output_dir / 'annotation_vocab.json'
        with open(vocab_file, 'w') as f:
            json.dump(vocab, f, indent=2)
            
    def _create_splits(self, protein_data: Dict):
        """Create train/validation/test splits."""
        # Get all protein IDs
        all_proteins = list(protein_data.keys())
        np.random.shuffle(all_proteins)
        
        # Split ratios
        train_ratio = 0.8
        val_ratio = 0.1
        test_ratio = 0.1
        
        n_proteins = len(all_proteins)
        n_train = int(n_proteins * train_ratio)
        n_val = int(n_proteins * val_ratio)
        
        # Assign splits
        train_proteins = all_proteins[:n_train]
        val_proteins = all_proteins[n_train:n_train + n_val]
        test_proteins = all_proteins[n_train + n_val:]
        
        # Update metadata with splits
        metadata_file = self.output_dir / 'metadata.csv'
        metadata_df = pd.read_csv(metadata_file)
        
        def get_split(protein_id):
            if protein_id in train_proteins:
                return 'train'
            elif protein_id in val_proteins:
                return 'val'
            else:
                return 'test'
                
        metadata_df['split'] = metadata_df['protein_id'].apply(get_split)
        metadata_df.to_csv(metadata_file, index=False)
        
        # Save split lists
        splits = {
            'train': train_proteins,
            'val': val_proteins,
            'test': test_proteins
        }
        
        splits_file = self.output_dir / 'splits.json'
        with open(splits_file, 'w') as f:
            json.dump(splits, f, indent=2)
            
        logger.info(f"Created splits: train={len(train_proteins)}, "
                   f"val={len(val_proteins)}, test={len(test_proteins)}")


def main():
    parser = argparse.ArgumentParser(description="Prepare protein data for Enhanced InterPLM")
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Output directory for processed data'
    )
    parser.add_argument(
        '--protein-list',
        type=Path,
        help='File containing protein IDs (one per line)'
    )
    parser.add_argument(
        '--uniprot-query',
        type=str,
        help='UniProt query string (e.g., "organism:9606 AND reviewed:true")'
    )
    parser.add_argument(
        '--max-proteins',
        type=int,
        default=1000,
        help='Maximum number of proteins to process'
    )
    parser.add_argument(
        '--esm-model',
        type=str,
        default='esm2_t6_8M_UR50D',
        help='ESM model to use for embeddings'
    )
    parser.add_argument(
        '--max-seq-length',
        type=int,
        default=1022,
        help='Maximum sequence length'
    )
    
    args = parser.parse_args()
    
    # Initialize preparer
    preparer = ProteinDataPreparer(
        output_dir=args.output_dir,
        esm_model=args.esm_model,
        max_seq_length=args.max_seq_length
    )
    
    # Run preparation
    preparer.prepare_dataset(
        protein_list_file=args.protein_list,
        uniprot_query=args.uniprot_query,
        max_proteins=args.max_proteins
    )


if __name__ == '__main__':
    main()        
