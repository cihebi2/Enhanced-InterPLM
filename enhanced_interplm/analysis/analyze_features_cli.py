# enhanced_interplm/analysis/analyze_features_cli.py

"""
Command-line interface for analyzing discovered features and circuits.
"""

import typer
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from typing import Optional, List
import json
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.table import Table
from rich.progress import track
import networkx as nx

from enhanced_interplm.utils.helpers import load_checkpoint
from enhanced_interplm.circuit_discovery import (
    DynamicGraphBuilder, CircuitMotifDetector, CircuitValidator
)
from enhanced_interplm.evaluation import InterpretabilityMetrics
from enhanced_interplm.visualization import (
    CircuitVisualizer, TemporalFlowVisualizer
)

app = typer.Typer()
console = Console()


@app.command()
def analyze_checkpoint(
    checkpoint_path: Path = typer.Argument(
        ...,
        help="Path to model checkpoint"
    ),
    data_path: Path = typer.Argument(
        ...,
        help="Path to data for analysis"
    ),
    output_dir: Path = typer.Option(
        Path("feature_analysis"),
        "--output", "-o",
        help="Output directory for results"
    ),
    num_samples: int = typer.Option(
        100,
        "--samples", "-n",
        help="Number of samples to analyze"
    ),
    top_features: int = typer.Option(
        50,
        "--top-features", "-t",
        help="Number of top features to analyze"
    ),
    circuit_analysis: bool = typer.Option(
        True,
        "--circuits/--no-circuits",
        help="Perform circuit analysis"
    ),
    device: str = typer.Option(
        "cuda",
        "--device", "-d",
        help="Device to use"
    )
):
    """Analyze features from a trained Enhanced InterPLM model."""
    
    console.print("[bold blue]Enhanced InterPLM - Feature Analysis[/bold blue]")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint
    console.print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = load_checkpoint(checkpoint_path, torch.device(device))
    
    # Extract model configuration
    config = checkpoint.get('config', {})
    model_config = config.get('model', {})
    
    # Initialize temporal SAE
    from enhanced_interplm.temporal_sae import TemporalSAE
    
    temporal_sae = TemporalSAE(
        input_dim=model_config['temporal_sae']['input_dim'],
        hidden_dim=model_config['temporal_sae']['hidden_dim'],
        dict_size=model_config['temporal_sae']['dict_size'],
        num_layers=len(model_config.get('esm_layers', [1, 2, 3, 4, 5, 6])),
        num_attention_heads=model_config['temporal_sae'].get('num_attention_heads', 8),
        dropout=0.0  # No dropout for analysis
    ).to(device)
    
    temporal_sae.load_state_dict(checkpoint['model_states']['temporal_sae'])
    temporal_sae.eval()
    
    # Load data
    console.print(f"Loading data from {data_path}")
    features_dict = load_features_for_analysis(data_path, num_samples, device)
    
    # Extract features using the model
    all_features = []
    
    with torch.no_grad():
        for embeddings in track(features_dict['embeddings'], description="Extracting features"):
            _, features = temporal_sae(embeddings.unsqueeze(0), return_features=True)
            all_features.append(features.squeeze(0))
            
    all_features = torch.stack(all_features)
    
    # Feature statistics
    console.print("\n[bold]Computing feature statistics...[/bold]")
    feature_stats = compute_feature_statistics(all_features)
    
    # Save statistics
    save_feature_statistics(feature_stats, output_dir / "feature_statistics.json")
    
    # Identify top features
    top_feature_indices = identify_top_features(feature_stats, top_features)
    console.print(f"Identified top {len(top_feature_indices)} features")
    
    # Visualize feature activation patterns
    console.print("\n[bold]Creating feature visualizations...[/bold]")
    visualize_feature_patterns(
        all_features,
        top_feature_indices,
        output_dir / "feature_patterns"
    )
    
    # Circuit analysis
    if circuit_analysis:
        console.print("\n[bold]Performing circuit analysis...[/bold]")
        circuits = analyze_circuits(all_features, output_dir / "circuits")
        
        # Print circuit summary
        print_circuit_summary(circuits)
        
    # Generate report
    generate_analysis_report(
        feature_stats,
        top_feature_indices,
        circuits if circuit_analysis else None,
        config,
        output_dir / "analysis_report.html"
    )
    
    console.print(f"\n[bold green]✓[/bold green] Analysis complete! Results saved to {output_dir}")


@app.command()
def compare_checkpoints(
    checkpoint1: Path = typer.Argument(..., help="First checkpoint"),
    checkpoint2: Path = typer.Argument(..., help="Second checkpoint"),
    output_dir: Path = typer.Option(
        Path("checkpoint_comparison"),
        "--output", "-o",
        help="Output directory"
    ),
    metric: str = typer.Option(
        "cosine",
        "--metric", "-m",
        help="Similarity metric (cosine/correlation/l2)"
    )
):
    """Compare features between two checkpoints."""
    
    console.print("[bold blue]Comparing checkpoints[/bold blue]")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load both checkpoints
    ckpt1 = load_checkpoint(checkpoint1)
    ckpt2 = load_checkpoint(checkpoint2)
    
    # Extract decoder weights (these define features)
    decoder1 = ckpt1['model_states']['temporal_sae']['decoder.weight']
    decoder2 = ckpt2['model_states']['temporal_sae']['decoder.weight']
    
    # Compute similarity matrix
    if metric == "cosine":
        # Normalize features
        decoder1_norm = decoder1 / decoder1.norm(dim=0, keepdim=True)
        decoder2_norm = decoder2 / decoder2.norm(dim=0, keepdim=True)
        similarity = torch.matmul(decoder1_norm.T, decoder2_norm)
        
    elif metric == "correlation":
        # Compute correlation
        decoder1_centered = decoder1 - decoder1.mean(dim=0)
        decoder2_centered = decoder2 - decoder2.mean(dim=0)
        similarity = torch.matmul(decoder1_centered.T, decoder2_centered)
        similarity = similarity / (decoder1_centered.std(dim=0).unsqueeze(1) * decoder2_centered.std(dim=0).unsqueeze(0))
        
    else:  # L2 distance
        similarity = -torch.cdist(decoder1.T, decoder2.T)
        
    # Find best matches
    best_matches = similarity.argmax(dim=1)
    match_scores = similarity.max(dim=1)[0]
    
    # Save results
    results = {
        'metric': metric,
        'num_features_1': decoder1.shape[1],
        'num_features_2': decoder2.shape[1],
        'avg_match_score': float(match_scores.mean()),
        'min_match_score': float(match_scores.min()),
        'max_match_score': float(match_scores.max())
    }
    
    with open(output_dir / 'comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)
        
    # Visualize similarity matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        similarity.cpu().numpy()[:100, :100],  # Show first 100 features
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1
    )
    plt.xlabel('Checkpoint 2 Features')
    plt.ylabel('Checkpoint 1 Features')
    plt.title(f'Feature Similarity ({metric})')
    plt.tight_layout()
    plt.savefig(output_dir / 'similarity_matrix.png')
    
    console.print(f"[bold green]✓[/bold green] Comparison complete!")
    console.print(f"Average match score: {results['avg_match_score']:.3f}")


@app.command()
def export_features(
    checkpoint_path: Path = typer.Argument(..., help="Model checkpoint"),
    output_file: Path = typer.Option(
        Path("features.npz"),
        "--output", "-o",
        help="Output file for features"
    ),
    format: str = typer.Option(
        "npz",
        "--format", "-f",
        help="Output format (npz/csv/parquet)"
    )
):
    """Export learned features for external analysis."""
    
    console.print("[bold blue]Exporting features[/bold blue]")
    
    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path)
    
    # Extract all feature-related weights
    features = {
        'decoder_weights': checkpoint['model_states']['temporal_sae']['decoder.weight'].cpu().numpy(),
        'encoder_weights': checkpoint['model_states']['temporal_sae']['encoder.weight'].cpu().numpy(),
        'encoder_bias': checkpoint['model_states']['temporal_sae']['encoder.bias'].cpu().numpy(),
        'layer_biases': checkpoint['model_states']['temporal_sae']['layer_biases'].cpu().numpy()
    }
    
    # Add metadata
    metadata = {
        'dict_size': features['decoder_weights'].shape[1],
        'input_dim': features['decoder_weights'].shape[0],
        'num_layers': features['layer_biases'].shape[0],
        'epoch': checkpoint.get('epoch', -1),
        'global_step': checkpoint.get('global_step', -1)
    }
    
    # Save in requested format
    if format == "npz":
        np.savez(output_file, **features, metadata=metadata)
        
    elif format == "csv":
        # Convert to DataFrame
        df = pd.DataFrame(features['decoder_weights'].T)
        df.columns = [f'dim_{i}' for i in range(df.shape[1])]
        df['feature_id'] = range(len(df))
        df.to_csv(output_file, index=False)
        
    elif format == "parquet":
        df = pd.DataFrame(features['decoder_weights'].T)
        df.columns = [f'dim_{i}' for i in range(df.shape[1])]
        df['feature_id'] = range(len(df))
        df.to_parquet(output_file)
        
    console.print(f"[bold green]✓[/bold green] Features exported to {output_file}")


def load_features_for_analysis(data_path: Path, num_samples: int, device: torch.device) -> dict:
    """Load features from data file."""
    if data_path.suffix == '.h5':
        import h5py
        with h5py.File(data_path, 'r') as f:
            # Load first num_samples
            embeddings = []
            sequences = []
            
            for i, seq_id in enumerate(f.keys()):
                if i >= num_samples:
                    break
                    
                embeddings.append(torch.tensor(f[seq_id]['embeddings'][:]))
                sequences.append(f[seq_id].attrs.get('sequence', ''))
                
            return {
                'embeddings': torch.stack(embeddings).to(device),
                'sequences': sequences
            }
            
    else:
        raise ValueError(f"Unsupported data format: {data_path.suffix}")


def compute_feature_statistics(features: torch.Tensor) -> dict:
    """Compute comprehensive statistics for features."""
    # features shape: [batch, layers, seq_len, dict_size]
    
    stats = {}
    
    # Activation frequency (how often each feature is active)
    activation_freq = (features.abs() > 0.1).float().mean(dim=(0, 1, 2))
    stats['activation_frequency'] = activation_freq.cpu().numpy()
    
    # Mean activation magnitude
    mean_activation = features.abs().mean(dim=(0, 1, 2))
    stats['mean_activation'] = mean_activation.cpu().numpy()
    
    # Activation variance
    activation_std = features.std(dim=(0, 1, 2))
    stats['activation_std'] = activation_std.cpu().numpy()
    
    # Layer-specific statistics
    layer_stats = []
    for layer in range(features.shape[1]):
        layer_features = features[:, layer]
        layer_stats.append({
            'layer': layer,
            'mean_sparsity': (layer_features.abs() > 0.1).float().mean().item(),
            'mean_magnitude': layer_features.abs().mean().item(),
            'max_activation': layer_features.abs().max().item()
        })
    stats['layer_statistics'] = layer_stats
    
    # Feature co-activation
    # Sample subset for efficiency
    sample_features = features[:10].reshape(-1, features.shape[-1])
    if sample_features.shape[0] > 1000:
        sample_features = sample_features[:1000]
        
    feature_active = (sample_features.abs() > 0.1).float()
    coactivation = torch.matmul(feature_active.T, feature_active) / feature_active.shape[0]
    stats['coactivation_matrix'] = coactivation.cpu().numpy()
    
    return stats


def identify_top_features(stats: dict, top_k: int) -> List[int]:
    """Identify top features based on statistics."""
    # Score features by activation frequency and magnitude
    scores = stats['activation_frequency'] * stats['mean_activation']
    
    # Get top k features
    top_indices = np.argsort(scores)[-top_k:][::-1]
    
    return top_indices.tolist()


def visualize_feature_patterns(features: torch.Tensor, feature_indices: List[int], output_dir: Path):
    """Visualize activation patterns for selected features."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create activation heatmap for top features
    fig, axes = plt.subplots(len(feature_indices[:10]), 1, figsize=(12, 2 * len(feature_indices[:10])))
    
    if len(feature_indices) == 1:
        axes = [axes]
        
    for idx, feat_idx in enumerate(feature_indices[:10]):
        # Average over batch and layers
        feat_pattern = features[:, :, :, feat_idx].mean(dim=(0, 1)).cpu().numpy()
        
        ax = axes[idx]
        ax.plot(feat_pattern)
        ax.set_title(f'Feature {feat_idx}')
        ax.set_xlabel('Position')
        ax.set_ylabel('Activation')
        
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_patterns.png')
    plt.close()
    
    # Create layer evolution plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for feat_idx in feature_indices[:5]:
        layer_means = features[:, :, :, feat_idx].mean(dim=(0, 2)).cpu().numpy()
        ax.plot(layer_means, label=f'Feature {feat_idx}')
        
    ax.set_xlabel('Layer')
    ax.set_ylabel('Mean Activation')
    ax.set_title('Feature Evolution Across Layers')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'layer_evolution.png')
    plt.close()


def analyze_circuits(features: torch.Tensor, output_dir: Path) -> List[dict]:
    """Analyze functional circuits in features."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to format for circuit analysis
    features_dict = {}
    for layer in range(features.shape[1]):
        features_dict[layer] = features[:, layer].detach()
        
    # Build interaction graph
    graph_builder = DynamicGraphBuilder()
    interaction_graph = graph_builder.build_feature_interaction_graph(features_dict)
    
    # Detect motifs
    motif_detector = CircuitMotifDetector()
    motifs = motif_detector.find_circuit_motifs(interaction_graph, min_frequency=2)
    
    # Validate circuits
    validator = CircuitValidator()
    validated_circuits = validator.validate_circuits(motifs[:50], features_dict)
    
    # Save circuit graph
    nx.write_graphml(interaction_graph, output_dir / 'interaction_graph.graphml')
    
    # Visualize top circuits
    visualizer = CircuitVisualizer()
    for i, circuit in enumerate(validated_circuits[:5]):
        visualizer.visualize_circuit_graph(
            circuit,
            interaction_graph,
            save_path=output_dir / f'circuit_{i}.html'
        )
        
    return validated_circuits


def print_circuit_summary(circuits: List[dict]):
    """Print summary of discovered circuits."""
    table = Table(title="Discovered Circuits")
    
    table.add_column("Circuit", style="cyan")
    table.add_column("Size", style="magenta")
    table.add_column("Frequency", style="green")
    table.add_column("Importance", style="yellow")
    table.add_column("Coherence", style="blue")
    
    for i, circuit in enumerate(circuits[:10]):
        size = len(circuit['instances'][0]) if circuit['instances'] else 0
        freq = circuit.get('frequency', 0)
        importance = circuit.get('importance', 0)
        coherence = circuit.get('validation_scores', {}).get('coherence', 0)
        
        table.add_row(
            f"Circuit {i+1}",
            str(size),
            str(freq),
            f"{importance:.3f}",
            f"{coherence:.3f}"
        )
        
    console.print(table)


def generate_analysis_report(
    feature_stats: dict,
    top_features: List[int],
    circuits: Optional[List[dict]],
    config: dict,
    output_path: Path
):
    """Generate HTML analysis report."""
    html = f"""
    <html>
    <head>
        <title>Enhanced InterPLM Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1, h2 {{ color: #333; }}
            .metric {{ background: #f0f0f0; padding: 10px; margin: 10px 0; }}
            .feature {{ background: #e8f4f8; padding: 5px; margin: 5px 0; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background: #4CAF50; color: white; }}
        </style>
    </head>
    <body>
        <h1>Enhanced InterPLM Analysis Report</h1>
        
        <h2>Model Configuration</h2>
        <div class="metric">
            <strong>Model:</strong> {config.get('model', {}).get('esm_model', 'Unknown')}<br>
            <strong>Dictionary Size:</strong> {config.get('model', {}).get('temporal_sae', {}).get('dict_size', 'Unknown')}<br>
            <strong>Layers:</strong> {config.get('model', {}).get('esm_layers', [])}
        </div>
        
        <h2>Feature Statistics</h2>
        <div class="metric">
            <strong>Total Features:</strong> {len(feature_stats['activation_frequency'])}<br>
            <strong>Active Features:</strong> {sum(feature_stats['activation_frequency'] > 0.01)}<br>
            <strong>Mean Activation Frequency:</strong> {np.mean(feature_stats['activation_frequency']):.4f}<br>
            <strong>Mean Activation Magnitude:</strong> {np.mean(feature_stats['mean_activation']):.4f}
        </div>
        
        <h2>Top Features</h2>
        <table>
            <tr>
                <th>Feature ID</th>
                <th>Activation Frequency</th>
                <th>Mean Magnitude</th>
                <th>Std Deviation</th>
            </tr>
    """
    
    for feat_idx in top_features[:20]:
        html += f"""
            <tr>
                <td>{feat_idx}</td>
                <td>{feature_stats['activation_frequency'][feat_idx]:.4f}</td>
                <td>{feature_stats['mean_activation'][feat_idx]:.4f}</td>
                <td>{feature_stats['activation_std'][feat_idx]:.4f}</td>
            </tr>
        """
        
    html += """
        </table>
        
        <h2>Layer Statistics</h2>
        <table>
            <tr>
                <th>Layer</th>
                <th>Mean Sparsity</th>
                <th>Mean Magnitude</th>
                <th>Max Activation</th>
            </tr>
    """
    
    for layer_stat in feature_stats['layer_statistics']:
        html += f"""
            <tr>
                <td>{layer_stat['layer']}</td>
                <td>{layer_stat['mean_sparsity']:.4f}</td>
                <td>{layer_stat['mean_magnitude']:.4f}</td>
                <td>{layer_stat['max_activation']:.4f}</td>
            </tr>
        """
        
    if circuits:
        html += f"""
        </table>
        
        <h2>Circuit Analysis</h2>
        <div class="metric">
            <strong>Total Circuits:</strong> {len(circuits)}<br>
            <strong>Average Circuit Size:</strong> {np.mean([len(c['instances'][0]) for c in circuits if c['instances']]):.2f}<br>
            <strong>Max Circuit Importance:</strong> {max([c['importance'] for c in circuits]):.4f}
        </div>
        """
        
    html += """
    </body>
    </html>
    """
    
    with open(output_path, 'w') as f:
        f.write(html)


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()