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
        
        for protein_id, data in protein_data.items():
            sequences