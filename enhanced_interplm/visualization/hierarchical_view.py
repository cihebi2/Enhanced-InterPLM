# 层次化视图 
# enhanced_interplm/visualization/hierarchical_view.py

"""
Hierarchical visualization of multi-scale feature-function mappings.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch, Wedge
from matplotlib.collections import PatchCollection
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, List, Tuple, Optional, Union
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
import networkx as nx
from collections import defaultdict


class HierarchicalMappingVisualizer:
    """
    Visualizes hierarchical feature-function mappings across multiple scales.
    """
    
    def __init__(
        self,
        style: str = 'modern',
        color_scheme: Optional[Dict[str, str]] = None
    ):
        self.style = style
        
        # Default color scheme for different levels
        if color_scheme is None:
            self.color_scheme = {
                'amino_acid': '#FF6B6B',
                'secondary_structure': '#4ECDC4',
                'domain': '#45B7D1',
                'function': '#96CEB4',
                'interaction': '#DDA0DD',
                'conservation': '#FFD93D'
            }
        else:
            self.color_scheme = color_scheme
            
        self._setup_style()
        
    def _setup_style(self):
        """Set up visualization style."""
        if self.style == 'modern':
            plt.style.use('seaborn-v0_8-darkgrid')
            self.bg_color = '#f8f9fa'
            self.text_color = '#2c3e50'
            self.edge_color = '#34495e'
        elif self.style == 'classic':
            plt.style.use('seaborn-v0_8-whitegrid')
            self.bg_color = '#ffffff'
            self.text_color = '#000000'
            self.edge_color = '#333333'
        else:  # dark
            plt.style.use('dark_background')
            self.bg_color = '#1e1e1e'
            self.text_color = '#ffffff'
            self.edge_color = '#cccccc'
            
    def visualize_hierarchical_features(
        self,
        aa_features: Dict[str, torch.Tensor],
        ss_features: Dict[str, torch.Tensor],
        domain_features: Dict[str, List],
        integrated_features: Optional[torch.Tensor] = None,
        sample_idx: int = 0,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create comprehensive hierarchical visualization.
        
        Args:
            aa_features: Amino acid level features
            ss_features: Secondary structure features
            domain_features: Domain level features
            integrated_features: Cross-level integrated features
            sample_idx: Which sample to visualize
            save_path: Path to save figure
            
        Returns:
            Figure object
        """
        # Create figure with custom layout
        fig = plt.figure(figsize=(16, 12))
        
        # Define grid layout
        gs = fig.add_gridspec(4, 3, height_ratios=[1, 1.5, 1.5, 1],
                            width_ratios=[1, 2, 1], hspace=0.3, wspace=0.2)
        
        # Top row: Summary statistics
        ax_summary = fig.add_subplot(gs[0, :])
        self._plot_level_summary(ax_summary, aa_features, ss_features, domain_features)
        
        # Second row: AA features
        ax_aa_props = fig.add_subplot(gs[1, 0])
        ax_aa_main = fig.add_subplot(gs[1, 1])
        ax_aa_cons = fig.add_subplot(gs[1, 2])
        self._plot_aa_level(ax_aa_props, ax_aa_main, ax_aa_cons,
                          aa_features, sample_idx)
        
        # Third row: SS and domains
        ax_ss = fig.add_subplot(gs[2, :2])
        ax_domains = fig.add_subplot(gs[2, 2])
        self._plot_ss_level(ax_ss, ss_features, sample_idx)
        self._plot_domain_level(ax_domains, domain_features, sample_idx)
        
        # Bottom row: Integration
        ax_integration = fig.add_subplot(gs[3, :])
        if integrated_features is not None:
            self._plot_integration(ax_integration, integrated_features, sample_idx)
        else:
            ax_integration.text(0.5, 0.5, 'No integrated features available',
                              ha='center', va='center', transform=ax_integration.transAxes)
            ax_integration.axis('off')
            
        # Main title
        fig.suptitle('Hierarchical Feature-Function Mapping', fontsize=16, y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=self.bg_color)
            
        return fig
        
    def create_interactive_hierarchy(
        self,
        features: Dict[str, torch.Tensor],
        annotations: Dict[str, List],
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create interactive hierarchical visualization with Plotly.
        
        Args:
            features: Dictionary of features at different levels
            annotations: Annotations for features
            save_path: Path to save HTML
            
        Returns:
            Plotly figure
        """
        # Create sunburst chart for hierarchical data
        data = self._prepare_hierarchical_data(features, annotations)
        
        fig = go.Figure(go.Sunburst(
            ids=data['ids'],
            labels=data['labels'],
            parents=data['parents'],
            values=data['values'],
            branchvalues="total",
            marker=dict(
                colors=data['colors'],
                colorscale='Viridis',
                cmid=0.5
            ),
            hovertemplate='<b>%{label}</b><br>Value: %{value:.3f}<br>Level: %{customdata}<extra></extra>',
            customdata=data['levels']
        ))
        
        fig.update_layout(
            title='Hierarchical Feature Organization',
            width=900,
            height=900,
            margin=dict(t=50, l=0, r=0, b=0)
        )
        
        if save_path:
            fig.write_html(save_path)
            
        return fig
        
    def visualize_cross_level_attention(
        self,
        attention_weights: Dict[str, torch.Tensor],
        level_names: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize attention patterns between hierarchical levels.
        
        Args:
            attention_weights: Dictionary of attention matrices
            level_names: Names for each level
            save_path: Path to save figure
            
        Returns:
            Figure object
        """
        if level_names is None:
            level_names = ['Amino Acid', 'Secondary Structure', 'Domain', 'Function']
            
        num_levels = len(level_names)
        
        fig, axes = plt.subplots(1, num_levels-1, figsize=(15, 5))
        
        # Plot attention between consecutive levels
        for i in range(num_levels - 1):
            ax = axes[i] if num_levels > 2 else axes
            
            # Get attention matrix for this level pair
            key = f'{level_names[i].lower()}_{level_names[i+1].lower()}'
            if key in attention_weights:
                attn = attention_weights[key].cpu().numpy()
                
                # Plot heatmap
                sns.heatmap(attn, ax=ax, cmap='YlOrRd', cbar=True,
                          xticklabels=False, yticklabels=False)
                ax.set_title(f'{level_names[i]} → {level_names[i+1]}')
                ax.set_xlabel(level_names[i+1])
                ax.set_ylabel(level_names[i])
                
        plt.suptitle('Cross-Level Attention Patterns', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def create_feature_hierarchy_tree(
        self,
        features: Dict[str, torch.Tensor],
        method: str = 'ward',
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create hierarchical clustering tree of features.
        
        Args:
            features: Feature dictionary
            method: Clustering method
            save_path: Path to save figure
            
        Returns:
            Figure object
        """
        # Combine features from different levels
        all_features = []
        feature_labels = []
        
        for level_name, level_features in features.items():
            if isinstance(level_features, torch.Tensor):
                # Flatten and sample features
                flat_features = level_features.reshape(-1, level_features.shape[-1])
                
                # Sample subset for visualization
                n_samples = min(100, flat_features.shape[0])
                indices = np.random.choice(flat_features.shape[0], n_samples, replace=False)
                
                sampled = flat_features[indices].cpu().numpy()
                all_features.append(sampled)
                
                # Create labels
                for i in range(n_samples):
                    feature_labels.append(f'{level_name[:3]}_{i}')
                    
        # Combine all features
        combined_features = np.vstack(all_features)
        
        # Compute linkage
        distances = pdist(combined_features, metric='cosine')
        linkage_matrix = linkage(distances, method=method)
        
        # Create dendrogram
        fig, ax = plt.subplots(figsize=(12, 8))
        
        dendrogram(
            linkage_matrix,
            labels=feature_labels,
            ax=ax,
            leaf_rotation=90,
            leaf_font_size=8
        )
        
        ax.set_title('Hierarchical Feature Clustering', fontsize=14)
        ax.set_xlabel('Feature')
        ax.set_ylabel('Distance')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def visualize_level_transitions(
        self,
        features: Dict[str, torch.Tensor],
        transitions: Dict[str, np.ndarray],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize transitions between hierarchical levels.
        
        Args:
            features: Features at each level
            transitions: Transition probabilities between levels
            save_path: Path to save figure
            
        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create circular layout for levels
        levels = list(features.keys())
        n_levels = len(levels)
        
        # Position levels in a circle
        angles = np.linspace(0, 2 * np.pi, n_levels, endpoint=False)
        positions = {}
        
        for i, (level, angle) in enumerate(zip(levels, angles)):
            x = np.cos(angle) * 3
            y = np.sin(angle) * 3
            positions[level] = (x, y)
            
            # Draw level node
            circle = Circle((x, y), 0.8, color=self.color_scheme.get(level, 'gray'),
                          alpha=0.7, ec=self.edge_color, linewidth=2)
            ax.add_patch(circle)
            
            # Add label
            ax.text(x, y, level.replace('_', '\n'), ha='center', va='center',
                   fontsize=10, fontweight='bold')
                   
        # Draw transitions
        for (src, dst), weight in transitions.items():
            if src in positions and dst in positions:
                x1, y1 = positions[src]
                x2, y2 = positions[dst]
                
                # Draw arrow
                ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                          arrowprops=dict(
                              arrowstyle='->',
                              color=self.edge_color,
                              alpha=min(1.0, weight),
                              lw=weight * 3,
                              connectionstyle="arc3,rad=0.2"
                          ))
                          
        # Set limits and remove axes
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_aspect('equal')
        ax.axis('off')
        
        ax.set_title('Level Transition Network', fontsize=14, pad=20)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def create_3d_hierarchy(
        self,
        features: Dict[str, torch.Tensor],
        sample_idx: int = 0,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create 3D visualization of hierarchical features.
        
        Args:
            features: Feature dictionary
            sample_idx: Sample to visualize
            save_path: Path to save HTML
            
        Returns:
            Plotly 3D figure
        """
        # Extract features for visualization
        traces = []
        
        # Z-levels for each hierarchy level
        z_levels = {'amino_acid': 0, 'secondary_structure': 1, 'domain': 2, 'function': 3}
        
        for level_name, level_features in features.items():
            if isinstance(level_features, torch.Tensor) and level_features.dim() >= 2:
                # Get features for sample
                if level_features.dim() > 2:
                    feat = level_features[sample_idx].cpu().numpy()
                else:
                    feat = level_features.cpu().numpy()
                    
                # Reduce to 2D for visualization
                if feat.shape[-1] > 2:
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=2)
                    feat_2d = pca.fit_transform(feat.reshape(-1, feat.shape[-1]))
                else:
                    feat_2d = feat.reshape(-1, 2)
                    
                # Create 3D scatter
                z_level = z_levels.get(level_name, 0)
                
                trace = go.Scatter3d(
                    x=feat_2d[:, 0],
                    y=feat_2d[:, 1],
                    z=np.full(len(feat_2d), z_level),
                    mode='markers',
                    name=level_name,
                    marker=dict(
                        size=5,
                        color=self.color_scheme.get(level_name, 'gray'),
                        opacity=0.8
                    )
                )
                traces.append(trace)
                
        # Create figure
        fig = go.Figure(data=traces)
        
        fig.update_layout(
            title='3D Hierarchical Feature Space',
            scene=dict(
                xaxis_title='Component 1',
                yaxis_title='Component 2',
                zaxis_title='Hierarchy Level',
                zaxis=dict(
                    tickmode='array',
                    tickvals=[0, 1, 2, 3],
                    ticktext=['AA', 'SS', 'Domain', 'Function']
                )
            ),
            width=900,
            height=700
        )
        
        if save_path:
            fig.write_html(save_path)
            
        return fig
        
    # Helper methods for plotting
    
    def _plot_level_summary(self, ax, aa_features, ss_features, domain_features):
        """Plot summary statistics for each level."""
        levels = ['Amino Acid', 'Secondary Structure', 'Domain']
        
        # Compute statistics
        stats = []
        
        # AA level
        aa_active = sum(v.abs().mean() > 0.1 for v in aa_features.values()) / len(aa_features)
        stats.append(aa_active)
        
        # SS level
        ss_active = ss_features.get('ss_predictions', torch.zeros(1)).unique().numel() / 8
        stats.append(ss_active)
        
        # Domain level
        domain_count = len(domain_features.get(0, []))
        stats.append(min(1.0, domain_count / 5))  # Normalize to [0, 1]
        
        # Create bar plot
        bars = ax.bar(levels, stats, color=[self.color_scheme.get(l.lower().replace(' ', '_'), 'gray')
                                           for l in levels])
        
        # Add value labels
        for bar, stat in zip(bars, stats):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{stat:.2f}', ha='center', va='bottom')
                   
        ax.set_ylabel('Activity Level')
        ax.set_title('Feature Activity by Hierarchical Level')
        ax.set_ylim(0, 1.2)
        
    def _plot_aa_level(self, ax_props, ax_main, ax_cons, aa_features, sample_idx):
        """Plot amino acid level features."""
        # Properties plot
        if 'hydrophobicity' in aa_features:
            hydro = aa_features['hydrophobicity'][sample_idx].squeeze().cpu().numpy()
            positions = np.arange(len(hydro))
            
            ax_props.plot(positions, hydro, color=self.color_scheme['amino_acid'], lw=2)
            ax_props.fill_between(positions, 0, hydro, alpha=0.3,
                                color=self.color_scheme['amino_acid'])
            ax_props.set_ylabel('Hydrophobicity')
            ax_props.set_title('AA Properties')
            ax_props.grid(True, alpha=0.3)
            
        # Main features plot
        if 'charge' in aa_features:
            charge = aa_features['charge'][sample_idx].cpu().numpy()
            
            # Stacked bar chart for charge distribution
            ax_main.bar(positions, charge[:, 0], label='Positive',
                      color='red', alpha=0.7)
            ax_main.bar(positions, -charge[:, 1], label='Negative',
                      color='blue', alpha=0.7)
            
            ax_main.set_ylabel('Charge')
            ax_main.set_title('Charge Distribution')
            ax_main.legend()
            ax_main.grid(True, alpha=0.3, axis='y')
            
        # Conservation plot
        if 'conservation' in aa_features:
            cons = aa_features['conservation'][sample_idx].squeeze().cpu().numpy()
            
            im = ax_cons.imshow(cons.reshape(1, -1), aspect='auto',
                              cmap='YlOrRd', vmin=0, vmax=1)
            ax_cons.set_yticks([])
            ax_cons.set_xlabel('Position')
            ax_cons.set_title('Conservation')
            
            # Add colorbar
            plt.colorbar(im, ax=ax_cons, orientation='horizontal', pad=0.1)
            
    def _plot_ss_level(self, ax, ss_features, sample_idx):
        """Plot secondary structure features."""
        if 'ss_predictions' not in ss_features:
            ax.text(0.5, 0.5, 'No SS predictions available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            return
            
        ss_pred = ss_features['ss_predictions'][sample_idx].cpu().numpy()
        seq_len = len(ss_pred)
        
        # Create SS visualization
        ss_colors = {
            0: '#ff0000',  # Helix - red
            1: '#ff8800',  # Bridge - orange
            2: '#ffff00',  # Strand - yellow
            3: '#00ff00',  # 3-10 helix - green
            4: '#00ffff',  # Pi helix - cyan
            5: '#0000ff',  # Turn - blue
            6: '#ff00ff',  # Bend - magenta
            7: '#888888'   # Coil - gray
        }
        
        # Plot SS as colored blocks
        for i in range(seq_len):
            ss_type = int(ss_pred[i])
            rect = Rectangle((i, 0), 1, 1, facecolor=ss_colors.get(ss_type, 'gray'),
                           edgecolor='black', linewidth=0.5)
            ax.add_patch(rect)
            
        # Add legend
        legend_elements = [
            mpatches.Patch(color=ss_colors[i], label=['H', 'B', 'E', 'G', 'I', 'T', 'S', 'C'][i])
            for i in range(8)
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1),
                 ncol=1, title='SS Type')
                 
        ax.set_xlim(0, seq_len)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Position')
        ax.set_yticks([])
        ax.set_title('Secondary Structure Prediction')
        
    def _plot_domain_level(self, ax, domain_features, sample_idx):
        """Plot domain features."""
        if sample_idx not in domain_features or not domain_features[sample_idx]:
            ax.text(0.5, 0.5, 'No domains detected',
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            return
            
        domains = domain_features[sample_idx]
        
        # Create pie chart of domain types
        domain_counts = defaultdict(int)
        for domain in domains:
            domain_counts[f"Domain {domain['class']}"] += 1
            
        if domain_counts:
            labels = list(domain_counts.keys())
            sizes = list(domain_counts.values())
            colors = [self.color_scheme.get('domain', 'gray') for _ in labels]
            
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                  startangle=90)
            ax.set_title('Domain Distribution')
            
    def _plot_integration(self, ax, integrated_features, sample_idx):
        """Plot integrated features."""
        integrated = integrated_features[sample_idx].cpu().numpy()
        
        # Use PCA for visualization
        if integrated.shape[-1] > 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            integrated_2d = pca.fit_transform(integrated)
            
            ax.scatter(integrated_2d[:, 0], integrated_2d[:, 1],
                      c=np.arange(len(integrated_2d)), cmap='viridis',
                      alpha=0.6, s=50)
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        else:
            # Direct plot if already 2D
            ax.scatter(integrated[:, 0], integrated[:, 1],
                      c=np.arange(len(integrated)), cmap='viridis',
                      alpha=0.6, s=50)
            ax.set_xlabel('Dimension 1')
            ax.set_ylabel('Dimension 2')
            
        ax.set_title('Integrated Cross-Level Features')
        ax.grid(True, alpha=0.3)
        
    def _prepare_hierarchical_data(self, features, annotations):
        """Prepare data for hierarchical visualization."""
        # Build hierarchical structure
        data = {
            'ids': ['protein'],
            'labels': ['Protein'],
            'parents': [''],
            'values': [1.0],
            'colors': [0.5],
            'levels': ['root']
        }
        
        # Add levels
        level_idx = 1
        for level_name, level_features in features.items():
            if isinstance(level_features, dict):
                for feat_name, feat_tensor in level_features.items():
                    feat_id = f'{level_name}_{feat_name}'
                    data['ids'].append(feat_id)
                    data['labels'].append(feat_name)
                    data['parents'].append('protein')
                    data['values'].append(feat_tensor.abs().mean().item())
                    data['colors'].append(level_idx)
                    data['levels'].append(level_name)
                    
            level_idx += 1
            
        return data