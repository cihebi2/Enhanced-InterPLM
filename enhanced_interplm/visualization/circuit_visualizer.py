# 回路可视化 
# enhanced_interplm/visualization/circuit_visualizer.py

"""
Advanced visualization tools for functional circuits and feature interactions.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
import colorcet as cc
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class CircuitVisualizer:
    """
    Comprehensive visualization toolkit for functional circuits.
    """
    
    def __init__(
        self,
        style: str = 'dark',
        color_palette: Optional[List[str]] = None,
        fig_size: Tuple[int, int] = (12, 8),
        dpi: int = 300
    ):
        self.style = style
        self.fig_size = fig_size
        self.dpi = dpi
        
        # Set style
        if style == 'dark':
            plt.style.use('dark_background')
            self.bg_color = '#1e1e1e'
            self.grid_color = '#333333'
            self.text_color = '#ffffff'
        else:
            plt.style.use('seaborn-v0_8-whitegrid')
            self.bg_color = '#ffffff'
            self.grid_color = '#cccccc'
            self.text_color = '#000000'
            
        # Color palette
        if color_palette is None:
            self.color_palette = [
                '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57',
                '#FF9FF3', '#54A0FF', '#48C9B0', '#FD79A8', '#A29BFE'
            ]
        else:
            self.color_palette = color_palette
            
    def visualize_circuit_graph(
        self,
        circuit: Dict[str, Any],
        interaction_graph: nx.Graph,
        save_path: Optional[Union[str, Path]] = None,
        interactive: bool = True,
        layout: str = 'spring'
    ) -> Union[plt.Figure, go.Figure]:
        """
        Visualize a circuit as an interactive network graph.
        
        Args:
            circuit: Circuit dictionary with nodes and edges
            interaction_graph: NetworkX graph of feature interactions
            save_path: Optional path to save visualization
            interactive: Use plotly for interactive visualization
            layout: Graph layout algorithm
            
        Returns:
            Figure object (matplotlib or plotly)
        """
        if interactive:
            return self._create_interactive_circuit_graph(
                circuit, interaction_graph, save_path, layout
            )
        else:
            return self._create_static_circuit_graph(
                circuit, interaction_graph, save_path, layout
            )
            
    def _create_interactive_circuit_graph(
        self,
        circuit: Dict,
        graph: nx.Graph,
        save_path: Optional[Path],
        layout: str
    ) -> go.Figure:
        """Create interactive circuit visualization with Plotly."""
        
        # Extract circuit nodes
        circuit_nodes = set()
        for instance in circuit.get('instances', []):
            circuit_nodes.update(instance)
            
        # Create subgraph
        subgraph = graph.subgraph(circuit_nodes)
        
        # Compute layout
        if layout == 'spring':
            pos = nx.spring_layout(subgraph, k=2, iterations=50)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(subgraph)
        elif layout == 'circular':
            pos = nx.circular_layout(subgraph)
        else:
            pos = nx.spring_layout(subgraph)
            
        # Create edge traces
        edge_trace = []
        for edge in subgraph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            edge_trace.append(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=2, color='#888'),
                hoverinfo='none',
                showlegend=False
            ))
            
        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        
        for node in subgraph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Extract node info
            layer = int(node.split('_')[0][1:]) if '_' in node else 0
            feature = int(node.split('_F')[1]) if '_F' in node else 0
            
            node_text.append(f"Layer: {layer}<br>Feature: {feature}")
            node_color.append(layer)
            
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                showscale=True,
                colorscale='Viridis',
                size=20,
                color=node_color,
                colorbar=dict(
                    thickness=15,
                    title='Layer',
                    xanchor='left',
                    titleside='right'
                ),
                line_width=2
            )
        )
        
        # Create figure
        fig = go.Figure(
            data=edge_trace + [node_trace],
            layout=go.Layout(
                title=f"Circuit {circuit.get('id', 'Unknown')} - "
                      f"Importance: {circuit.get('importance', 0):.3f}",
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor=self.bg_color,
                paper_bgcolor=self.bg_color,
                font=dict(color=self.text_color)
            )
        )
        
        if save_path:
            fig.write_html(str(save_path))
            
        return fig
        
    def _create_static_circuit_graph(
        self,
        circuit: Dict,
        graph: nx.Graph,
        save_path: Optional[Path],
        layout: str
    ) -> plt.Figure:
        """Create static circuit visualization with matplotlib."""
        
        fig, ax = plt.subplots(figsize=self.fig_size)
        
        # Extract circuit nodes
        circuit_nodes = set()
        for instance in circuit.get('instances', []):
            circuit_nodes.update(instance)
            
        # Create subgraph
        subgraph = graph.subgraph(circuit_nodes)
        
        # Compute layout
        if layout == 'spring':
            pos = nx.spring_layout(subgraph, k=2, iterations=50)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(subgraph)
        else:
            pos = nx.spring_layout(subgraph)
            
        # Extract node colors based on layer
        node_colors = []
        for node in subgraph.nodes():
            if '_' in node:
                layer = int(node.split('_')[0][1:])
                node_colors.append(layer)
            else:
                node_colors.append(0)
                
        # Draw graph
        nx.draw_networkx_nodes(
            subgraph, pos, ax=ax,
            node_color=node_colors,
            node_size=500,
            cmap='viridis',
            alpha=0.9
        )
        
        nx.draw_networkx_edges(
            subgraph, pos, ax=ax,
            edge_color='gray',
            width=2,
            alpha=0.5
        )
        
        nx.draw_networkx_labels(
            subgraph, pos, ax=ax,
            font_size=8,
            font_color=self.text_color
        )
        
        ax.set_title(
            f"Circuit {circuit.get('id', 'Unknown')} - "
            f"Importance: {circuit.get('importance', 0):.3f}",
            fontsize=14
        )
        ax.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            
        return fig
        
    def plot_circuit_activation(
        self,
        circuit: Dict,
        features: torch.Tensor,
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Plot activation patterns of circuit features.
        
        Args:
            circuit: Circuit information
            features: Feature tensor [batch, layers, seq_len, features]
            save_path: Optional save path
            
        Returns:
            Matplotlib figure
        """
        # Extract circuit feature indices
        circuit_features = set()
        for instance in circuit.get('instances', []):
            for node in instance:
                if '_F' in node:
                    feat_idx = int(node.split('_F')[1])
                    circuit_features.add(feat_idx)
                    
        circuit_features = sorted(list(circuit_features))
        
        # Create subplots
        n_features = len(circuit_features)
        n_cols = min(4, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 2))
        if n_features == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
            
        # Plot each feature
        for idx, feat_idx in enumerate(circuit_features):
            ax = axes[idx]
            
            # Extract feature activations across layers
            feat_acts = features[:, :, :, feat_idx].mean(dim=0)  # Average over batch
            
            # Plot heatmap
            im = ax.imshow(
                feat_acts.cpu().numpy(),
                aspect='auto',
                cmap='hot',
                interpolation='nearest'
            )
            
            ax.set_title(f'Feature {feat_idx}', fontsize=10)
            ax.set_xlabel('Position')
            ax.set_ylabel('Layer')
            
            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
        # Remove empty subplots
        for idx in range(n_features, len(axes)):
            fig.delaxes(axes[idx])
            
        plt.suptitle(f'Circuit Activation Patterns', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            
        return fig
        
    def visualize_circuit_hierarchy(
        self,
        circuits: List[Dict],
        features: torch.Tensor,
        max_circuits: int = 50,
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Visualize hierarchical clustering of circuits.
        
        Args:
            circuits: List of circuit dictionaries
            features: Feature tensor
            max_circuits: Maximum number of circuits to display
            save_path: Optional save path
            
        Returns:
            Matplotlib figure
        """
        # Limit number of circuits
        circuits = circuits[:max_circuits]
        
        # Compute circuit similarity matrix
        similarity_matrix = self._compute_circuit_similarity(circuits, features)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Dendrogram
        linkage_matrix = linkage(squareform(1 - similarity_matrix), method='ward')
        dendrogram(
            linkage_matrix,
            ax=ax1,
            labels=[f"C{i}" for i in range(len(circuits))],
            color_threshold=0.7 * max(linkage_matrix[:, 2])
        )
        ax1.set_title('Circuit Hierarchy')
        ax1.set_xlabel('Circuit ID')
        ax1.set_ylabel('Distance')
        
        # Similarity heatmap
        sns.heatmap(
            similarity_matrix,
            ax=ax2,
            cmap='RdBu_r',
            vmin=0,
            vmax=1,
            square=True,
            cbar_kws={'label': 'Similarity'},
            xticklabels=[f"C{i}" for i in range(len(circuits))],
            yticklabels=[f"C{i}" for i in range(len(circuits))]
        )
        ax2.set_title('Circuit Similarity Matrix')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            
        return fig
        
    def _compute_circuit_similarity(
        self,
        circuits: List[Dict],
        features: torch.Tensor
    ) -> np.ndarray:
        """Compute pairwise similarity between circuits."""
        n_circuits = len(circuits)
        similarity_matrix = np.zeros((n_circuits, n_circuits))
        
        for i in range(n_circuits):
            for j in range(i, n_circuits):
                # Extract feature sets
                features_i = self._get_circuit_features(circuits[i])
                features_j = self._get_circuit_features(circuits[j])
                
                # Jaccard similarity
                intersection = len(features_i & features_j)
                union = len(features_i | features_j)
                
                similarity = intersection / union if union > 0 else 0
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity
                
        return similarity_matrix
        
    def _get_circuit_features(self, circuit: Dict) -> set:
        """Extract feature indices from circuit."""
        features = set()
        
        for instance in circuit.get('instances', []):
            for node in instance:
                if '_F' in node:
                    feat_idx = int(node.split('_F')[1])
                    features.add(feat_idx)
                    
        return features
        
    def plot_circuit_evolution(
        self,
        circuit: Dict,
        temporal_features: List[torch.Tensor],
        timestamps: Optional[List[str]] = None,
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Plot evolution of circuit activation over time/conditions.
        
        Args:
            circuit: Circuit information
            temporal_features: List of feature tensors at different timepoints
            timestamps: Optional timestamp labels
            save_path: Optional save path
            
        Returns:
            Matplotlib figure
        """
        # Extract circuit features
        circuit_features = self._get_circuit_features(circuit)
        circuit_features = sorted(list(circuit_features))
        
        if not timestamps:
            timestamps = [f"T{i}" for i in range(len(temporal_features))]
            
        # Compute circuit activation strength over time
        activation_strengths = []
        
        for features in temporal_features:
            # Average activation of circuit features
            circuit_acts = []
            for feat_idx in circuit_features:
                if feat_idx < features.shape[-1]:
                    act = features[:, :, :, feat_idx].abs().mean().item()
                    circuit_acts.append(act)
                    
            activation_strengths.append(np.mean(circuit_acts) if circuit_acts else 0)
            
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Line plot of activation strength
        ax1.plot(timestamps, activation_strengths, 'o-', linewidth=2, markersize=8)
        ax1.set_xlabel('Time/Condition')
        ax1.set_ylabel('Average Activation Strength')
        ax1.set_title('Circuit Activation Over Time')
        ax1.grid(True, alpha=0.3)
        
        # Heatmap of individual feature evolution
        feature_evolution = np.zeros((len(circuit_features), len(temporal_features)))
        
        for t_idx, features in enumerate(temporal_features):
            for f_idx, feat_idx in enumerate(circuit_features):
                if feat_idx < features.shape[-1]:
                    feature_evolution[f_idx, t_idx] = features[:, :, :, feat_idx].abs().mean().item()
                    
        im = ax2.imshow(
            feature_evolution,
            aspect='auto',
            cmap='viridis',
            interpolation='nearest'
        )
        
        ax2.set_xlabel('Time/Condition')
        ax2.set_ylabel('Circuit Features')
        ax2.set_xticks(range(len(timestamps)))
        ax2.set_xticklabels(timestamps)
        ax2.set_yticks(range(len(circuit_features)))
        ax2.set_yticklabels([f"F{idx}" for idx in circuit_features])
        ax2.set_title('Individual Feature Evolution')
        
        plt.colorbar(im, ax=ax2, label='Activation Strength')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            
        return fig
        
    def create_circuit_summary_dashboard(
        self,
        circuits: List[Dict],
        features: torch.Tensor,
        save_path: Optional[Path] = None
    ) -> go.Figure:
        """
        Create comprehensive dashboard for circuit analysis.
        
        Args:
            circuits: List of circuits
            features: Feature tensor
            save_path: Optional save path
            
        Returns:
            Plotly figure
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                'Circuit Importance Distribution',
                'Circuit Size Distribution',
                'Feature Coverage',
                'Layer Distribution',
                'Circuit Coherence',
                'Top Circuit Features'
            ),
            specs=[
                [{'type': 'histogram'}, {'type': 'histogram'}, {'type': 'bar'}],
                [{'type': 'histogram'}, {'type': 'scatter'}, {'type': 'heatmap'}]
            ]
        )
        
        # 1. Circuit importance distribution
        importances = [c.get('importance', 0) for c in circuits]
        fig.add_trace(
            go.Histogram(x=importances, nbinsx=30, name='Importance'),
            row=1, col=1
        )
        
        # 2. Circuit size distribution
        sizes = [len(c.get('instances', [[]])[0]) if c.get('instances') else 0 for c in circuits]
        fig.add_trace(
            go.Histogram(x=sizes, nbinsx=20, name='Size'),
            row=1, col=2
        )
        
        # 3. Feature coverage
        all_features = set()
        for circuit in circuits:
            all_features.update(self._get_circuit_features(circuit))
            
        coverage = len(all_features) / features.shape[-1]
        fig.add_trace(
            go.Bar(
                x=['Used', 'Unused'],
                y=[len(all_features), features.shape[-1] - len(all_features)],
                name='Features'
            ),
            row=1, col=3
        )
        
        # 4. Layer distribution
        layer_counts = {}
        for circuit in circuits:
            for instance in circuit.get('instances', []):
                for node in instance:
                    if '_' in node:
                        layer = int(node.split('_')[0][1:])
                        layer_counts[layer] = layer_counts.get(layer, 0) + 1
                        
        fig.add_trace(
            go.Histogram(
                x=list(layer_counts.keys()),
                y=list(layer_counts.values()),
                name='Layer Usage'
            ),
            row=2, col=1
        )
        
        # 5. Circuit coherence scatter
        coherences = []
        sizes_plot = []
        
        for circuit in circuits[:100]:  # Limit for performance
            if 'validation_scores' in circuit:
                coherence = circuit['validation_scores'].get('coherence', 0)
                size = len(circuit.get('instances', [[]])[0]) if circuit.get('instances') else 0
                coherences.append(coherence)
                sizes_plot.append(size)
                
        fig.add_trace(
            go.Scatter(
                x=sizes_plot,
                y=coherences,
                mode='markers',
                marker=dict(
                    size=8,
                    color=importances[:len(coherences)],
                    colorscale='Viridis',
                    showscale=True
                ),
                name='Circuits'
            ),
            row=2, col=2
        )
        
        # 6. Top circuit features heatmap
        top_circuits = sorted(circuits, key=lambda x: x.get('importance', 0), reverse=True)[:10]
        
        feature_matrix = np.zeros((10, min(50, features.shape[-1])))
        
        for i, circuit in enumerate(top_circuits):
            circuit_feats = list(self._get_circuit_features(circuit))[:50]
            for feat_idx in circuit_feats:
                if feat_idx < feature_matrix.shape[1]:
                    feature_matrix[i, feat_idx] = 1
                    
        fig.add_trace(
            go.Heatmap(
                z=feature_matrix,
                colorscale='Blues',
                showscale=False
            ),
            row=2, col=3
        )
        
        # Update layout
        fig.update_layout(
            title_text="Circuit Analysis Dashboard",
            showlegend=False,
            height=800,
            plot_bgcolor=self.bg_color,
            paper_bgcolor=self.bg_color,
            font=dict(color=self.text_color)
        )
        
        # Update axes
        fig.update_xaxes(title_text="Importance", row=1, col=1)
        fig.update_xaxes(title_text="Circuit Size", row=1, col=2)
        fig.update_xaxes(title_text="Feature Type", row=1, col=3)
        fig.update_xaxes(title_text="Layer", row=2, col=1)
        fig.update_xaxes(title_text="Circuit Size", row=2, col=2)
        fig.update_xaxes(title_text="Feature Index", row=2, col=3)
        
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=2)
        fig.update_yaxes(title_text="Number of Features", row=1, col=3)
        fig.update_yaxes(title_text="Count", row=2, col=1)
        fig.update_yaxes(title_text="Coherence", row=2, col=2)
        fig.update_yaxes(title_text="Top Circuits", row=2, col=3)
        
        if save_path:
            fig.write_html(str(save_path))
            
        return fig
        
    def visualize_circuit_3d_embedding(
        self,
        circuits: List[Dict],
        features: torch.Tensor,
        method: str = 'tsne',
        save_path: Optional[Path] = None
    ) -> go.Figure:
        """
        Create 3D embedding visualization of circuits.
        
        Args:
            circuits: List of circuits
            features: Feature tensor
            method: Embedding method ('tsne', 'pca', 'mds')
            save_path: Optional save path
            
        Returns:
            Plotly 3D figure
        """
        # Create circuit feature vectors
        circuit_vectors = []
        circuit_labels = []
        circuit_sizes = []
        
        for i, circuit in enumerate(circuits[:100]):  # Limit for performance
            circuit_feats = self._get_circuit_features(circuit)
            
            # Create binary feature vector
            feat_vector = np.zeros(min(features.shape[-1], 1000))
            for feat_idx in circuit_feats:
                if feat_idx < len(feat_vector):
                    feat_vector[feat_idx] = 1
                    
            circuit_vectors.append(feat_vector)
            circuit_labels.append(f"Circuit {i}")
            circuit_sizes.append(len(circuit_feats))
            
        circuit_vectors = np.array(circuit_vectors)
        
        # Compute embedding
        if method == 'tsne':
            embedder = TSNE(n_components=3, random_state=42)
        elif method == 'pca':
            embedder = PCA(n_components=3)
        elif method == 'mds':
            embedder = MDS(n_components=3, random_state=42)
        else:
            raise ValueError(f"Unknown embedding method: {method}")
            
        coords = embedder.fit_transform(circuit_vectors)
        
        # Create 3D scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode='markers+text',
            marker=dict(
                size=np.array(circuit_sizes) * 2,
                color=[c.get('importance', 0) for c in circuits[:len(coords)]],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Importance"),
                line=dict(color='white', width=0.5)
            ),
            text=circuit_labels,
            textposition="top center",
            hovertemplate='%{text}<br>Size: %{marker.size}<br>Importance: %{marker.color:.3f}'
        )])
        
        fig.update_layout(
            title=f'Circuit 3D Embedding ({method.upper()})',
            scene=dict(
                xaxis_title='Component 1',
                yaxis_title='Component 2',
                zaxis_title='Component 3',
                bgcolor=self.bg_color
            ),
            paper_bgcolor=self.bg_color,
            font=dict(color=self.text_color)
        )
        
        if save_path:
            fig.write_html(str(save_path))
            
        return fig
        
    def create_feature_flow_diagram(
        self,
        features: torch.Tensor,
        circuit: Dict,
        save_path: Optional[Path] = None
    ) -> go.Figure:
        """
        Create Sankey diagram showing feature flow through layers.
        
        Args:
            features: Feature tensor [batch, layers, seq_len, features]
            circuit: Circuit to visualize
            save_path: Optional save path
            
        Returns:
            Plotly figure
        """
        # Extract circuit features per layer
        layer_features = {}
        
        for instance in circuit.get('instances', []):
            for node in instance:
                if '_' in node and '_F' in node:
                    layer = int(node.split('_')[0][1:])
                    feat_idx = int(node.split('_F')[1])
                    
                    if layer not in layer_features:
                        layer_features[layer] = set()
                    layer_features[layer].add(feat_idx)
                    
        # Create nodes and links for Sankey
        labels = []
        source = []
        target = []
        value = []
        
        # Create nodes
        node_map = {}
        node_idx = 0
        
        layers = sorted(layer_features.keys())
        for layer in layers:
            for feat in sorted(layer_features[layer]):
                node_id = f"L{layer}_F{feat}"
                labels.append(node_id)
                node_map[node_id] = node_idx
                node_idx += 1
                
        # Create links between consecutive layers
        for i in range(len(layers) - 1):
            curr_layer = layers[i]
            next_layer = layers[i + 1]
            
            for feat1 in layer_features[curr_layer]:
                for feat2 in layer_features[next_layer]:
                    # Compute connection strength
                    if feat1 < features.shape[-1] and feat2 < features.shape[-1]:
                        # Use feature correlation as connection strength
                        corr = torch.corrcoef(torch.stack([
                            features[:, curr_layer, :, feat1].flatten(),
                            features[:, next_layer, :, feat2].flatten()
                        ]))[0, 1].abs().item()
                        
                        if corr > 0.3:  # Threshold
                            source_id = f"L{curr_layer}_F{feat1}"
                            target_id = f"L{next_layer}_F{feat2}"
                            
                            source.append(node_map[source_id])
                            target.append(node_map[target_id])
                            value.append(corr)
                            
        # Create Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=labels,
                color=[self.color_palette[i % len(self.color_palette)] for i in range(len(labels))]
            ),
            link=dict(
                source=source,
                target=target,
                value=value,
                color='rgba(100, 100, 100, 0.4)'
            )
        )])
        
        fig.update_layout(
            title=f"Feature Flow in Circuit",
            font_size=10,
            height=600,
            paper_bgcolor=self.bg_color,
            font=dict(color=self.text_color)
        )
        
        if save_path:
            fig.write_html(str(save_path))
            
        return fig