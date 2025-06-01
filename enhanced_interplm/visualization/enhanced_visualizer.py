# enhanced_interplm/visualization/enhanced_visualizer.py

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from typing import Dict, List, Tuple, Optional
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import py3Dmol


class CircuitVisualizer:
    """
    Visualizes discovered functional circuits and their activation patterns.
    """
    
    def __init__(self, style: str = 'dark'):
        self.style = style
        self._setup_style()
        
    def _setup_style(self):
        """Set up visualization style."""
        if self.style == 'dark':
            plt.style.use('dark_background')
            self.bg_color = '#1e1e1e'
            self.text_color = '#ffffff'
            self.accent_color = '#00d4ff'
        else:
            plt.style.use('seaborn-v0_8-whitegrid')
            self.bg_color = '#ffffff'
            self.text_color = '#000000'
            self.accent_color = '#0066cc'
            
    def visualize_circuit_graph(
        self,
        circuit: Dict,
        graph: nx.DiGraph,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """Visualize a circuit as an interactive graph."""
        
        # Extract subgraph for this circuit
        circuit_nodes = circuit['instances'][0]  # Use first instance
        subgraph = graph.subgraph(circuit_nodes)
        
        # Calculate layout
        pos = nx.spring_layout(subgraph, k=2, iterations=50)
        
        # Create edge traces
        edge_traces = []
        for edge in subgraph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(
                    width=subgraph[edge[0]][edge[1]]['weight'] * 5,
                    color=self.accent_color
                ),
                hoverinfo='none'
            )
            edge_traces.append(edge_trace)
            
        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        
        for node in subgraph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Extract layer and feature info
            layer, feature = node.split('_')
            node_text.append(f'{layer}<br>Feature {feature}')
            
            # Color by layer
            layer_num = int(layer[1:])
            node_color.append(layer_num)
            
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            text=node_text,
            textposition='top center',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='Viridis',
                color=node_color,
                size=20,
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
            data=edge_traces + [node_trace],
            layout=go.Layout(
                title=f'Circuit Visualization (Importance: {circuit["importance"]:.3f})',
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
            fig.write_html(save_path)
            
        return fig
        
    def plot_circuit_activation_heatmap(
        self,
        circuits: List[Dict],
        features: torch.Tensor,
        max_circuits: int = 20
    ) -> plt.Figure:
        """Plot heatmap of circuit activation patterns."""
        
        # Select top circuits
        top_circuits = sorted(circuits, key=lambda x: x['importance'], reverse=True)[:max_circuits]
        
        # Extract activation patterns
        circuit_activations = []
        circuit_labels = []
        
        for i, circuit in enumerate(top_circuits):
            # Get features for circuit nodes
            circuit_features = []
            
            for node in circuit['instances'][0]:
                layer, feat_idx = self._parse_node_id(node)
                feat_activation = features[:, layer, :, feat_idx].mean(dim=0)
                circuit_features.append(feat_activation.cpu().numpy())
                
            # Average activation across circuit
            avg_activation = np.mean(circuit_features, axis=0)
            circuit_activations.append(avg_activation)
            circuit_labels.append(f'Circuit {i+1}')
            
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        
        activation_matrix = np.array(circuit_activations)
        
        # Normalize each circuit's activation
        activation_matrix = (activation_matrix - activation_matrix.mean(axis=1, keepdims=True)) / (
            activation_matrix.std(axis=1, keepdims=True) + 1e-8
        )
        
        sns.heatmap(
            activation_matrix,
            cmap='RdBu_r',
            center=0,
            yticklabels=circuit_labels,
            xticklabels=False,
            cbar_kws={'label': 'Normalized Activation'},
            ax=ax
        )
        
        ax.set_xlabel('Sequence Position')
        ax.set_ylabel('Circuit')
        ax.set_title('Circuit Activation Patterns')
        
        plt.tight_layout()
        
        return fig
        
    def _parse_node_id(self, node_id: str) -> Tuple[int, int]:
        """Parse node ID to extract layer and feature indices."""
        parts = node_id.split('_')
        layer = int(parts[0][1:])
        feature = int(parts[1][1:])
        return layer, feature


class TemporalFlowVisualizer:
    """
    Visualizes feature evolution across transformer layers.
    """
    
    def __init__(self):
        self.colors = px.colors.qualitative.Set3
        
    def plot_feature_flow(
        self,
        features: torch.Tensor,
        feature_indices: List[int],
        protein_idx: int = 0
    ) -> go.Figure:
        """
        Plot how specific features evolve across layers.
        
        Args:
            features: [batch, layers, seq_len, features]
            feature_indices: List of feature indices to track
            protein_idx: Which protein in batch to visualize
        """
        
        num_layers = features.shape[1]
        seq_len = features.shape[2]
        
        # Create subplots for each feature
        fig = make_subplots(
            rows=len(feature_indices),
            cols=1,
            subplot_titles=[f'Feature {idx}' for idx in feature_indices],
            vertical_spacing=0.05
        )
        
        for i, feat_idx in enumerate(feature_indices):
            # Extract feature activation across layers
            feat_data = features[protein_idx, :, :, feat_idx].cpu().numpy()
            
            # Create heatmap
            heatmap = go.Heatmap(
                z=feat_data,
                x=list(range(seq_len)),
                y=list(range(num_layers)),
                colorscale='Viridis',
                showscale=(i == 0),
                colorbar=dict(title='Activation')
            )
            
            fig.add_trace(heatmap, row=i+1, col=1)
            
            # Update axes
            fig.update_xaxes(title_text='Position' if i == len(feature_indices)-1 else '', row=i+1, col=1)
            fig.update_yaxes(title_text='Layer', row=i+1, col=1)
            
        fig.update_layout(
            title='Feature Evolution Across Layers',
            height=200 * len(feature_indices),
            showlegend=False
        )
        
        return fig
        
    def plot_layer_transition_matrix(
        self,
        importance_scores: torch.Tensor
    ) -> go.Figure:
        """
        Plot transition matrix showing feature relationships between layers.
        
        Args:
            importance_scores: [features, features, num_layer_transitions]
        """
        
        num_features = importance_scores.shape[0]
        num_transitions = importance_scores.shape[2]
        
        # Average importance across all transitions
        avg_importance = importance_scores.mean(dim=2).cpu().numpy()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=avg_importance,
            x=[f'F{i}' for i in range(num_features)],
            y=[f'F{i}' for i in range(num_features)],
            colorscale='Hot',
            colorbar=dict(title='Avg Importance')
        ))
        
        fig.update_layout(
            title='Feature Transition Matrix (Averaged Across Layers)',
            xaxis_title='Target Feature',
            yaxis_title='Source Feature',
            width=800,
            height=800
        )
        
        return fig
        
    def animate_feature_evolution(
        self,
        features: torch.Tensor,
        feature_idx: int,
        protein_idx: int = 0,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """Create animated visualization of feature evolution."""
        
        num_layers = features.shape[1]
        seq_len = features.shape[2]
        
        # Extract feature data
        feat_data = features[protein_idx, :, :, feature_idx].cpu().numpy()
        
        # Create frames for animation
        frames = []
        for layer in range(num_layers):
            frame_data = go.Scatter(
                x=list(range(seq_len)),
                y=feat_data[layer],
                mode='lines+markers',
                name=f'Layer {layer}',
                line=dict(width=3)
            )
            
            frames.append(go.Frame(
                data=[frame_data],
                name=str(layer)
            ))
            
        # Create initial plot
        fig = go.Figure(
            data=[frames[0].data[0]],
            frames=frames
        )
        
        # Add animation controls
        fig.update_layout(
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {'label': 'Play',
                     'method': 'animate',
                     'args': [None, {'frame': {'duration': 500}}]},
                    {'label': 'Pause',
                     'method': 'animate',
                     'args': [[None], {'frame': {'duration': 0}, 'mode': 'immediate'}]}
                ]
            }],
            sliders=[{
                'active': 0,
                'steps': [{'label': f'Layer {i}', 'method': 'animate', 'args': [[str(i)]]}
                         for i in range(num_layers)]
            }],
            title=f'Feature {feature_idx} Evolution Across Layers',
            xaxis_title='Sequence Position',
            yaxis_title='Activation Value'
        )
        
        if save_path:
            fig.write_html(save_path)
            
        return fig


class HierarchicalMappingVisualizer:
    """
    Visualizes hierarchical feature-function mappings.
    """
    
    def __init__(self):
        self.level_colors = {
            'amino_acid': '#FF6B6B',
            'secondary_structure': '#4ECDC4',
            'domain': '#45B7D1'
        }
        
    def plot_hierarchical_features(
        self,
        aa_features: Dict[str, torch.Tensor],
        ss_features: Dict[str, torch.Tensor],
        domain_features: List[Dict],
        protein_idx: int = 0
    ) -> plt.Figure:
        """
        Plot multi-level feature analysis.
        """
        
        fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
        
        seq_len = aa_features['hydrophobicity'].shape[1]
        positions = np.arange(seq_len)
        
        # Plot amino acid properties
        ax = axes[0]
        hydro = aa_features['hydrophobicity'][protein_idx].squeeze().cpu().numpy()
        ax.plot(positions, hydro, label='Hydrophobicity', linewidth=2)
        ax.fill_between(positions, 0, hydro, alpha=0.3)
        ax.set_ylabel('Hydrophobicity')
        ax.legend()
        ax.set_title('Amino Acid Properties')
        
        # Plot charge distribution
        ax = axes[1]
        charge_probs = aa_features['charge'][protein_idx].cpu().numpy()
        ax.bar(positions, charge_probs[:, 0], label='Positive', alpha=0.7, color='red')
        ax.bar(positions, -charge_probs[:, 1], label='Negative', alpha=0.7, color='blue')
        ax.set_ylabel('Charge')
        ax.legend()
        ax.set_title('Charge Distribution')
        
        # Plot secondary structure
        ax = axes[2]
        ss_pred = ss_features['ss_predictions'][protein_idx].cpu().numpy()
        
        # Create color map for SS types
        ss_colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink', 'gray']
        ss_labels = ['Helix', 'Bridge', 'Strand', '3-10 Helix', 'Pi Helix', 'Turn', 'Bend', 'Coil']
        
        for i in range(8):
            mask = ss_pred == i
            ax.scatter(positions[mask], np.ones(mask.sum()) * i, 
                      c=ss_colors[i], label=ss_labels[i], s=50)
            
        ax.set_ylabel('SS Type')
        ax.set_ylim(-0.5, 7.5)
        ax.set_yticks(range(8))
        ax.set_yticklabels(ss_labels)
        ax.set_title('Secondary Structure Prediction')
        
        # Plot domains
        ax = axes[3]
        if domain_features[protein_idx]:
            for domain in domain_features[protein_idx]:
                start, end = domain['start'], domain['end']
                ax.axvspan(start, end, alpha=0.3, label=f'Domain {domain["class"]}')
                
        ax.set_ylabel('Domains')
        ax.set_xlabel('Sequence Position')
        ax.set_title('Domain Prediction')
        
        plt.tight_layout()
        
        return fig
        
    def plot_cross_level_attention(
        self,
        attention_weights: Dict[str, torch.Tensor]
    ) -> go.Figure:
        """
        Visualize attention between hierarchical levels.
        """
        
        # Create Sankey diagram showing information flow
        source = []
        target = []
        value = []
        
        # Define node indices
        aa_nodes = list(range(20))  # 20 amino acids
        ss_nodes = list(range(20, 28))  # 8 SS types
        domain_nodes = list(range(28, 33))  # 5 domain types
        
        # AA to SS attention
        if 'aa_ss_attention' in attention_weights:
            attn = attention_weights['aa_ss_attention'].mean(dim=0).cpu().numpy()
            for i in range(20):
                for j in range(8):
                    if attn[i, j] > 0.1:
                        source.append(i)
                        target.append(20 + j)
                        value.append(float(attn[i, j]))
                        
        # SS to Domain attention
        if 'ss_domain_attention' in attention_weights:
            attn = attention_weights['ss_domain_attention'].mean(dim=0).cpu().numpy()
            for i in range(8):
                for j in range(5):
                    if attn[i, j] > 0.1:
                        source.append(20 + i)
                        target.append(28 + j)
                        value.append(float(attn[i, j]))
                        
        # Create labels
        aa_labels = list('ACDEFGHIKLMNPQRSTVWY')
        ss_labels = ['Helix', 'Bridge', 'Strand', '3-10', 'Pi', 'Turn', 'Bend', 'Coil']
        domain_labels = ['Domain1', 'Domain2', 'Domain3', 'Domain4', 'Domain5']
        all_labels = aa_labels + ss_labels + domain_labels
        
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color='black', width=0.5),
                label=all_labels,
                color=['#FF6B6B']*20 + ['#4ECDC4']*8 + ['#45B7D1']*5
            ),
            link=dict(
                source=source,
                target=target,
                value=value
            )
        )])
        
        fig.update_layout(
            title='Cross-Level Attention Flow',
            font_size=10,
            height=600
        )
        
        return fig


class BiophysicsConstraintVisualizer:
    """
    Visualizes biophysical constraints and their effects.
    """
    
    def __init__(self):
        self.property_colors = {
            'hydrophobic': 'blue',
            'charged': 'red', 
            'polar': 'green',
            'aromatic': 'purple'
        }
        
    def visualize_3d_constraints(
        self,
        structure: torch.Tensor,
        features: torch.Tensor,
        constraints: Dict[str, torch.Tensor],
        feature_idx: int
    ) -> str:
        """
        Create 3D visualization with py3Dmol showing constraints.
        
        Returns HTML string for embedding.
        """
        
        # Extract coordinates and feature activations
        coords = structure.cpu().numpy()
        activations = features[:, :, feature_idx].mean(dim=0).cpu().numpy()
        
        # Create py3Dmol view
        view = py3Dmol.view(width=800, height=600)
        
        # Add protein backbone
        for i in range(len(coords) - 1):
            view.addCylinder({
                'start': {'x': coords[i, 0], 'y': coords[i, 1], 'z': coords[i, 2]},
                'end': {'x': coords[i+1, 0], 'y': coords[i+1, 1], 'z': coords[i+1, 2]},
                'radius': 0.3,
                'color': 'gray'
            })
            
        # Add spheres colored by feature activation
        for i, (coord, activation) in enumerate(zip(coords, activations)):
            # Color based on activation strength
            color = self._activation_to_color(activation)
            
            view.addSphere({
                'center': {'x': coord[0], 'y': coord[1], 'z': coord[2]},
                'radius': 0.5 + activation * 2,
                'color': color
            })
            
        # Add constraint visualizations
        if 'hydrophobic' in constraints:
            hydro_patches = self._find_hydrophobic_patches(
                coords, constraints['hydrophobic']
            )
            for patch in hydro_patches:
                center = coords[patch].mean(axis=0)
                view.addSphere({
                    'center': {'x': center[0], 'y': center[1], 'z': center[2]},
                    'radius': 3.0,
                    'color': 'blue',
                    'opacity': 0.3
                })
                
        view.zoomTo()
        view.spin(True)
        
        return view._make_html()
        
    def _activation_to_color(self, activation: float) -> str:
        """Convert activation value to color."""
        # Normalize to [0, 1]
        activation = np.clip(activation, 0, 1)
        
        # Create gradient from blue to red
        r = int(255 * activation)
        b = int(255 * (1 - activation))
        
        return f'#{r:02x}00{b:02x}'
        
    def _find_hydrophobic_patches(
        self,
        coords: np.ndarray,
        hydrophobicity: torch.Tensor,
        threshold: float = 0.7
    ) -> List[np.ndarray]:
        """Find hydrophobic patches in structure."""
        hydro_mask = hydrophobicity.squeeze().cpu().numpy() > threshold
        hydro_indices = np.where(hydro_mask)[0]
        
        if len(hydro_indices) < 3:
            return []
            
        # Cluster nearby hydrophobic residues
        from sklearn.cluster import DBSCAN
        
        hydro_coords = coords[hydro_indices]
        clustering = DBSCAN(eps=5.0, min_samples=3).fit(hydro_coords)
        
        patches = []
        for label in set(clustering.labels_):
            if label != -1:  # Not noise
                cluster_mask = clustering.labels_ == label
                cluster_indices = hydro_indices[cluster_mask]
                patches.append(cluster_indices)
                
        return patches
        
    def plot_constraint_losses(
        self,
        loss_history: Dict[str, List[float]]
    ) -> plt.Figure:
        """Plot training history of constraint losses."""
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for constraint_type, losses in loss_history.items():
            ax.plot(losses, label=constraint_type, linewidth=2)
            
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Loss')
        ax.set_title('Biophysical Constraint Losses During Training')
        ax.legend()
        ax.set_yscale('log')
        
        plt.tight_layout()
        
        return fig