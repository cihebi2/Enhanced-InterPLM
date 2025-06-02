# enhanced_interplm/esm/extract_embeddings_cli.py

"""
Command-line interface for extracting ESM embeddings with layer-wise tracking.
"""

import typer
from pathlib import Path
import torch
from typing import List, Optional
import pandas as pd
from rich.console import Console
from rich.progress import track
import h5py
import json

from enhanced_interplm.esm.layerwise_embeddings import (
    LayerwiseEmbeddingExtractor,
    LayerwiseEmbeddingConfig
)

app = typer.Typer()
console = Console()


@app.command()
def extract(
    input_file: Path = typer.Argument(
        ...,
        help="Input file containing sequences (FASTA or CSV)"
    ),
    output_file: Path = typer.Argument(
        ...,
        help="Output HDF5 file for embeddings"
    ),
    model_name: str = typer.Option(
        "esm2_t6_8M_UR50D",
        "--model", "-m",
        help="ESM model name"
    ),
    layers: Optional[str] = typer.Option(
        None,
        "--layers", "-l",
        help="Comma-separated layer indices (e.g., '1,3,5')"
    ),
    batch_size: int = typer.Option(
        8,
        "--batch-size", "-b",
        help="Batch size for processing"
    ),
    max_seq_length: int = typer.Option(
        1022,
        "--max-length",
        help="Maximum sequence length"
    ),
    device: str = typer.Option(
        "cuda",
        "--device", "-d",
        help="Device to use (cuda/cpu)"
    ),
    save_attention: bool = typer.Option(
        False,
        "--attention",
        help="Save attention weights"
    ),
    normalize: bool = typer.Option(
        False,
        "--normalize",
        help="Normalize embeddings"
    ),
    chunk_size: int = typer.Option(
        100,
        "--chunk-size",
        help="Number of sequences to process at once"
    )
):
    """Extract layer-wise ESM embeddings from protein sequences."""
    
    console.print(f"[bold blue]Enhanced InterPLM - ESM Embedding Extractor[/bold blue]")
    console.print(f"Model: {model_name}")
    console.print(f"Input: {input_file}")
    console.print(f"Output: {output_file}")
    
    # Load sequences
    sequences, seq_ids = load_sequences(input_file)
    console.print(f"Loaded {len(sequences)} sequences")
    
    # Parse layers
    if layers is not None:
        layer_list = [int(l.strip()) for l in layers.split(',')]
    else:
        layer_list = None
        
    # Configure extractor
    config = LayerwiseEmbeddingConfig(
        model_name=model_name,
        layers=layer_list,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
        device=device,
        save_attention_weights=save_attention,
        normalize_embeddings=normalize
    )
    
    # Initialize extractor
    with console.status("[bold green]Loading ESM model..."):
        extractor = LayerwiseEmbeddingExtractor(config)
        
    # Extract embeddings
    console.print("[bold green]Extracting embeddings...")
    extractor.extract_and_save(
        sequences=sequences,
        sequence_ids=seq_ids,
        output_path=output_file,
        chunk_size=chunk_size
    )
    
    console.print(f"[bold green]✓[/bold green] Embeddings saved to {output_file}")
    
    # Print summary
    with h5py.File(output_file, 'r') as f:
        console.print("\n[bold]Summary:[/bold]")
        console.print(f"  Model: {f.attrs['model']}")
        console.print(f"  Layers: {f.attrs['layers']}")
        console.print(f"  Sequences: {f.attrs['num_sequences']}")
        
        # Sample sequence info
        first_id = seq_ids[0]
        if first_id in f:
            emb_shape = f[first_id]['embeddings'].shape
            console.print(f"  Embedding shape: {emb_shape}")


@app.command()
def analyze(
    embedding_file: Path = typer.Argument(
        ...,
        help="HDF5 file containing embeddings"
    ),
    output_dir: Path = typer.Option(
        Path("embedding_analysis"),
        "--output", "-o",
        help="Output directory for analysis results"
    ),
    num_samples: int = typer.Option(
        100,
        "--samples", "-n",
        help="Number of samples to analyze"
    )
):
    """Analyze extracted embeddings."""
    
    console.print(f"[bold blue]Analyzing embeddings from {embedding_file}[/bold blue]")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load embeddings
    with h5py.File(embedding_file, 'r') as f:
        # Get metadata
        model_name = f.attrs.get('model', 'unknown')
        layers = f.attrs.get('layers', [])
        num_sequences = f.attrs.get('num_sequences', 0)
        
        console.print(f"Model: {model_name}")
        console.print(f"Layers: {layers}")
        console.print(f"Total sequences: {num_sequences}")
        
        # Analyze subset
        seq_ids = list(f.keys())[:num_samples]
        
        # Collect statistics
        stats = {
            'seq_id': [],
            'seq_len': [],
            'mean_activation': [],
            'std_activation': [],
            'sparsity': []
        }
        
        for seq_id in track(seq_ids, description="Analyzing sequences"):
            if seq_id in f:
                embeddings = f[seq_id]['embeddings'][:]
                
                stats['seq_id'].append(seq_id)
                stats['seq_len'].append(f[seq_id].attrs['length'])
                stats['mean_activation'].append(embeddings.mean())
                stats['std_activation'].append(embeddings.std())
                stats['sparsity'].append((embeddings.abs() < 0.01).mean())
                
    # Save statistics
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(output_dir / 'embedding_statistics.csv', index=False)
    
    # Create summary report
    report = {
        'model': model_name,
        'layers': list(layers),
        'num_sequences_analyzed': len(seq_ids),
        'avg_sequence_length': stats_df['seq_len'].mean(),
        'avg_mean_activation': stats_df['mean_activation'].mean(),
        'avg_std_activation': stats_df['std_activation'].mean(),
        'avg_sparsity': stats_df['sparsity'].mean()
    }
    
    with open(output_dir / 'analysis_report.json', 'w') as f:
        json.dump(report, f, indent=2)
        
    console.print(f"\n[bold green]✓[/bold green] Analysis complete!")
    console.print(f"Results saved to {output_dir}")


@app.command()
def merge(
    input_files: List[Path] = typer.Argument(
        ...,
        help="Input HDF5 files to merge"
    ),
    output_file: Path = typer.Option(
        ...,
        "--output", "-o",
        help="Output merged HDF5 file"
    )
):
    """Merge multiple embedding files."""
    
    console.print(f"[bold blue]Merging {len(input_files)} files[/bold blue]")
    
    with h5py.File(output_file, 'w') as out_f:
        total_sequences = 0
        
        # Copy metadata from first file
        with h5py.File(input_files[0], 'r') as first_f:
            for attr_name, attr_value in first_f.attrs.items():
                out_f.attrs[attr_name] = attr_value
                
        # Merge all files
        for input_file in track(input_files, description="Merging files"):
            with h5py.File(input_file, 'r') as in_f:
                for seq_id in in_f.keys():
                    if seq_id not in out_f:
                        in_f.copy(seq_id, out_f)
                        total_sequences += 1
                        
        # Update total sequences
        out_f.attrs['num_sequences'] = total_sequences
        
    console.print(f"[bold green]✓[/bold green] Merged {total_sequences} sequences")
    console.print(f"Output saved to {output_file}")


def load_sequences(input_file: Path) -> tuple:
    """Load sequences from FASTA or CSV file."""
    sequences = []
    seq_ids = []
    
    if input_file.suffix in ['.fasta', '.fa']:
        from Bio import SeqIO
        for record in SeqIO.parse(str(input_file), 'fasta'):
            sequences.append(str(record.seq))
            seq_ids.append(record.id)
            
    elif input_file.suffix == '.csv':
        df = pd.read_csv(input_file)
        if 'sequence' in df.columns and 'id' in df.columns:
            sequences = df['sequence'].tolist()
            seq_ids = df['id'].tolist()
        else:
            raise ValueError("CSV must have 'sequence' and 'id' columns")
            
    else:
        raise ValueError(f"Unsupported file format: {input_file.suffix}")
        
    return sequences, seq_ids


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()