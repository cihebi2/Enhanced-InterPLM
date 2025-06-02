# 时序流动可视化 
# enhanced_interplm/visualization/temporal_flow.py

"""
Temporal flow visualization for feature evolution across transformer layers.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
from matplotlib.collections import LineCollection
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union
import pandas as pd
from pathlib import Path
import colorcet as cc


class TemporalFlowVisualizer:
    """
    Advanced visualization of feature evolution and temporal flow across layers.
    """
    
    def __init__(
        self,
        style: str = 'dark',
        colormap: str = 'viridis',
        dpi: int = 300
    ):
        self.style = style
        self.colormap = colormap
        self.dpi = dpi
        
        # Set style
        self._setup_style()
        
    def _setup_style(self):
        """Set up visualization style."""
        if self.style == 'dark':
            plt.style.use('dark_background')
            self.bg_color = '#1e1e1e'
            self.text_color = '#ffffff'
            self.accent_color = '#00d4ff'
            self.grid_color = '#333333'
        else:
            plt.style.use('seaborn-v0_8-whitegrid')
            self.bg_color = '#ffffff'
            self.text_color = '#000000'
            self.accent_color = '#0066cc'
            self.grid_color = '#cccccc'
            
    def visualize_feature_flow(
        self,
        features: torch.Tensor,
        feature_indices: Optional[List[int]] = None,
        layer_names: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        interactive: bool = True
    ) -> Union[plt.Figure, go.Figure]:
        """
        Visualize how features flow and transform across layers.
        
        Args:
            features: Feature tensor [batch, layers, seq_len, num_features]
            feature_indices: Specific features to visualize (None for auto-select)
            layer_names: Names for each layer
            save_path: Path to save visualization
            interactive: Whether to create interactive plot
            
        Returns:
            Figure object
        """
        if interactive:
            return self._create_interactive_flow(
                features, feature_indices, layer_names, save_path
            )
        else:
            return self._create_static_flow(
                features, feature_indices, layer_names, save_path
            )
            
    def _create_interactive_flow(
        self,
        features: torch.Tensor,
        feature_indices: Optional[List[int]],
        layer_names: Optional[List[str]],
        save_path: Optional[str]
    ) -> go.Figure:
        """Create interactive flow visualization with Plotly."""
        batch_size, num_layers, seq_len, num_features = features.shape
        
        # Select features to visualize
        if feature_indices is None:
            # Auto-select most active features
            feature_activity = features.abs().mean(dim=(0, 1, 2))
            feature_indices = torch.topk(feature_activity, min(10, num_features))[1].tolist()
            
        if layer_names is None:
            layer_names = [f'Layer {i}' for i in range(num_layers)]
            
        # Create subplots
        fig = make_subplots(
            rows=len(feature_indices),
            cols=1,
            subplot_titles=[f'Feature {idx}' for idx in feature_indices],
            vertical_spacing=0.02
        )
        
        # Plot each feature
        for i, feat_idx in enumerate(feature_indices):
            # Extract feature across layers
            feat_data = features[:, :, :, feat_idx].mean(dim=0).cpu().numpy()
            
            # Create heatmap
            heatmap = go.Heatmap(
                z=feat_data,
                x=list(range(seq_len)),
                y=layer_names,
                colorscale='Viridis',
                showscale=(i == 0),
                hovertemplate='Layer: %{y}<br>Position: %{x}<br>Activation: %{z:.3f}<extra></extra>'
            )
            
            fig.add_trace(heatmap, row=i+1, col=1)
            
            # Add flow lines
            for layer in range(num_layers - 1):
                # Find peaks in current and next layer
                curr_peaks = self._find_peaks(feat_data[layer])
                next_peaks = self._find_peaks(feat_data[layer + 1])
                
                # Draw connections
                for curr_peak in curr_peaks:
                    for next_peak in next_peaks:
                        weight = feat_data[layer, curr_peak] * feat_data[layer + 1, next_peak]
                        
                        if weight > 0.1:
                            fig.add_shape(
                                type="line",
                                x0=curr_peak, y0=layer,
                                x1=next_peak, y1=layer + 1,
                                line=dict(
                                    color="rgba(255,255,255,0.3)",
                                    width=weight * 5
                                ),
                                row=i+1, col=1
                            )
                            
        # Update layout
        fig.update_layout(
            title='Feature Evolution Across Layers',
            height=200 * len(feature_indices),
            showlegend=False,
            plot_bgcolor=self.bg_color,
            paper_bgcolor=self.bg_color,
            font=dict(color=self.text_color)
        )
        
        fig.update_xaxes(title_text='Sequence Position')
        fig.update_yaxes(title_text='Layer')
        
        if save_path:
            fig.write_html(save_path)
            
        return fig
        
    def _create_static_flow(
        self,
        features: torch.Tensor,
        feature_indices: Optional[List[int]],
        layer_names: Optional[List[str]],
        save_path: Optional[str]
    ) -> plt.Figure:
        """Create static flow visualization with matplotlib."""
        batch_size, num_layers, seq_len, num_features = features.shape
        
        # Select features
        if feature_indices is None:
            feature_activity = features.abs().mean(dim=(0, 1, 2))
            feature_indices = torch.topk(feature_activity, min(5, num_features))[1].tolist()
            
        fig, axes = plt.subplots(
            len(feature_indices), 1,
            figsize=(12, 3 * len(feature_indices)),
            sharex=True
        )
        
        if len(feature_indices) == 1:
            axes = [axes]
            
        for ax_idx, feat_idx in enumerate(feature_indices):
            ax = axes[ax_idx]
            
            # Extract feature data
            feat_data = features[:, :, :, feat_idx].mean(dim=0).cpu().numpy()
            
            # Plot heatmap
            im = ax.imshow(
                feat_data,
                aspect='auto',
                cmap=self.colormap,
                interpolation='bilinear'
            )
            
            # Add flow arrows
            self._add_flow_arrows(ax, feat_data)
            
            # Formatting
            ax.set_ylabel('Layer')
            ax.set_title(f'Feature {feat_idx} Flow')
            
            if layer_names:
                ax.set_yticks(range(num_layers))
                ax.set_yticklabels(layer_names)
                
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Activation')
            
        axes[-1].set_xlabel('Sequence Position')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            
        return fig
        
    def visualize_layer_transitions(
        self,
        transition_matrices: List[torch.Tensor],
        layer_names: Optional[List[str]] = None,
        top_k: int = 20,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize feature transition matrices between layers.
        
        Args:
            transition_matrices: List of transition matrices between consecutive layers
            layer_names: Names for each layer
            top_k: Number of top connections to show
            save_path: Path to save visualization
            
        Returns:
            Figure object
        """
        num_transitions = len(transition_matrices)
        
        fig, axes = plt.subplots(
            1, num_transitions,
            figsize=(5 * num_transitions, 6)
        )
        
        if num_transitions == 1:
            axes = [axes]
            
        for i, (ax, trans_matrix) in enumerate(zip(axes, transition_matrices)):
            # Get top connections
            trans_numpy = trans_matrix.cpu().numpy()
            
            # Find top k connections
            flat_indices = np.argpartition(trans_numpy.ravel(), -top_k)[-top_k:]
            top_indices = np.unravel_index(flat_indices, trans_numpy.shape)
            
            # Create connection plot
            self._plot_layer_connections(
                ax, trans_numpy, top_indices,
                f'{layer_names[i] if layer_names else f"Layer {i}"}',
                f'{layer_names[i+1] if layer_names else f"Layer {i+1}"}'
            )
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            
        return fig
        
    def create_flow_animation(
        self,
        features: torch.Tensor,
        feature_idx: int,
        save_path: str,
        fps: int = 10
    ):
        """
        Create animated visualization of feature flow.
        
        Args:
            features: Feature tensor [batch, layers, seq_len, num_features]
            feature_idx: Feature index to animate
            save_path: Path to save animation
            fps: Frames per second
        """
        batch_size, num_layers, seq_len, _ = features.shape
        
        # Extract feature data
        feat_data = features[:, :, :, feature_idx].mean(dim=0).cpu().numpy()
        
        # Set up figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Initialize plot
        im = ax.imshow(
            np.zeros((num_layers, seq_len)),
            aspect='auto',
            cmap=self.colormap,
            vmin=feat_data.min(),
            vmax=feat_data.max()
        )
        
        ax.set_xlabel('Sequence Position')
        ax.set_ylabel('Layer')
        ax.set_title(f'Feature {feature_idx} Evolution')
        
        # Animation function
        def animate(frame):
            # Progressive reveal of layers
            data = np.zeros((num_layers, seq_len))
            
            if frame < num_layers:
                data[:frame+1] = feat_data[:frame+1]
            else:
                # After revealing all layers, highlight flow
                data = feat_data.copy()
                
                # Add flow highlighting
                flow_frame = frame - num_layers
                if flow_frame < seq_len:
                    for layer in range(num_layers):
                        data[layer, flow_frame] *= 1.5
                        
            im.set_data(data)
            return [im]
            
        # Create animation
        anim = animation.FuncAnimation(
            fig, animate,
            frames=num_layers + seq_len,
            interval=1000/fps,
            blit=True
        )
        
        # Save animation
        anim.save(save_path, writer='pillow', fps=fps)
        plt.close()
        
    def visualize_feature_trajectories(
        self,
        features: torch.Tensor,
        feature_indices: List[int],
        sample_idx: int = 0,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize trajectories of specific features across layers.
        
        Args:
            features: Feature tensor [batch, layers, seq_len, num_features]
            feature_indices: Features to track
            sample_idx: Which sample in batch to visualize
            save_path: Path to save visualization
            
        Returns:
            Figure object
        """
        num_layers = features.shape[1]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Extract trajectories
        trajectories = []
        for feat_idx in feature_indices:
            traj = features[sample_idx, :, :, feat_idx].cpu().numpy()
            trajectories.append(traj)
            
        # Plot 1: Average activation across sequence
        for i, (feat_idx, traj) in enumerate(zip(feature_indices, trajectories)):
            avg_activation = traj.mean(axis=1)
            color = plt.cm.tab10(i % 10)
            
            ax1.plot(range(num_layers), avg_activation, 'o-',
                    label=f'Feature {feat_idx}', color=color, linewidth=2)
            
        ax1.set_ylabel('Average Activation')
        ax1.set_title('Feature Trajectories Across Layers')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Activation variance
        for i, (feat_idx, traj) in enumerate(zip(feature_indices, trajectories)):
            activation_std = traj.std(axis=1)
            color = plt.cm.tab10(i % 10)
            
            ax2.plot(range(num_layers), activation_std, 's-',
                    label=f'Feature {feat_idx}', color=color, linewidth=2)
            
        ax2.set_xlabel('Layer')
        ax2.set_ylabel('Activation Std Dev')
        ax2.set_title('Feature Activation Variability')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            
        return fig
        
    def visualize_information_flow(
        self,
        features: torch.Tensor,
        method: str = 'mutual_information',
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize information flow between layers.
        
        Args:
            features: Feature tensor [batch, layers, seq_len, num_features]
            method: Method to measure information flow
            save_path: Path to save visualization
            
        Returns:
            Figure object
        """
        num_layers = features.shape[1]
        
        # Compute information flow
        flow_matrix = self._compute_information_flow(features, method)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot flow matrix
        im = ax.imshow(flow_matrix, cmap='YlOrRd', aspect='auto')
        
        # Add text annotations
        for i in range(num_layers):
            for j in range(num_layers):
                text = ax.text(j, i, f'{flow_matrix[i, j]:.2f}',
                             ha='center', va='center',
                             color='black' if flow_matrix[i, j] > 0.5 else 'white')
                             
        # Formatting
        ax.set_xticks(range(num_layers))
        ax.set_yticks(range(num_layers))
        ax.set_xticklabels([f'L{i}' for i in range(num_layers)])
        ax.set_yticklabels([f'L{i}' for i in range(num_layers)])
        ax.set_xlabel('Target Layer')
        ax.set_ylabel('Source Layer')
        ax.set_title(f'Information Flow ({method})')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Information Transfer')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            
        return fig
        
    def create_sankey_diagram(
        self,
        features: torch.Tensor,
        feature_indices: List[int],
        threshold: float = 0.3,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create Sankey diagram showing feature flow across layers.
        
        Args:
            features: Feature tensor
            feature_indices: Features to include
            threshold: Minimum connection strength to show
            save_path: Path to save visualization
            
        Returns:
            Plotly figure
        """
        num_layers = features.shape[1]
        
        # Build node labels
        labels = []
        for layer in range(num_layers):
            for feat_idx in feature_indices:
                labels.append(f'L{layer}_F{feat_idx}')
                
        # Build connections
        source = []
        target = []
        value = []
        
        for layer in range(num_layers - 1):
            for i, feat_i in enumerate(feature_indices):
                for j, feat_j in enumerate(feature_indices):
                    # Compute connection strength
                    strength = self._compute_feature_connection(
                        features, layer, feat_i, feat_j
                    )
                    
                    if strength > threshold:
                        source_idx = layer * len(feature_indices) + i
                        target_idx = (layer + 1) * len(feature_indices) + j
                        
                        source.append(source_idx)
                        target.append(target_idx)
                        value.append(float(strength))
                        
        # Create Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=labels,
                color=self.accent_color
            ),
            link=dict(
                source=source,
                target=target,
                value=value,
                color='rgba(0,212,255,0.4)'
            )
        )])
        
        fig.update_layout(
            title="Feature Flow Across Layers",
            font_size=10,
            height=600,
            plot_bgcolor=self.bg_color,
            paper_bgcolor=self.bg_color,
            font=dict(color=self.text_color)
        )
        
        if save_path:
            fig.write_html(save_path)
            
        return fig
        
    # Helper methods
    
    def _find_peaks(self, signal: np.ndarray, threshold: float = 0.5) -> List[int]:
        """Find peaks in 1D signal."""
        peaks = []
        
        for i in range(1, len(signal) - 1):
            if signal[i] > threshold and signal[i] > signal[i-1] and signal[i] > signal[i+1]:
                peaks.append(i)
                
        return peaks
        
    def _add_flow_arrows(self, ax, feat_data: np.ndarray):
        """Add flow arrows to heatmap."""
        num_layers, seq_len = feat_data.shape
        
        for layer in range(num_layers - 1):
            # Find strong activations
            curr_peaks = self._find_peaks(feat_data[layer])
            next_peaks = self._find_peaks(feat_data[layer + 1])
            
            for curr_peak in curr_peaks[:5]:  # Limit arrows
                for next_peak in next_peaks[:5]:
                    weight = feat_data[layer, curr_peak] * feat_data[layer + 1, next_peak]
                    
                    if weight > 0.2:
                        ax.annotate('', xy=(next_peak, layer + 1),
                                  xytext=(curr_peak, layer),
                                  arrowprops=dict(
                                      arrowstyle='->',
                                      color='white',
                                      alpha=min(1.0, weight),
                                      lw=1
                                  ))
                                  
    def _plot_layer_connections(
        self,
        ax,
        trans_matrix: np.ndarray,
        top_indices: Tuple[np.ndarray, np.ndarray],
        source_name: str,
        target_name: str
    ):
        """Plot connections between two layers."""
        # Create bipartite layout
        num_source = trans_matrix.shape[0]
        num_target = trans_matrix.shape[1]
        
        # Plot nodes
        y_source = np.linspace(0, 1, num_source)
        y_target = np.linspace(0, 1, num_target)
        
        # Source nodes
        ax.scatter(np.zeros(num_source), y_source, s=50, c='blue', zorder=2)
        
        # Target nodes
        ax.scatter(np.ones(num_target), y_target, s=50, c='red', zorder=2)
        
        # Plot connections
        for src, tgt in zip(top_indices[0], top_indices[1]):
            weight = trans_matrix[src, tgt]
            
            ax.plot([0, 1], [y_source[src], y_target[tgt]],
                   'k-', alpha=min(1.0, weight), lw=weight*3)
                   
        # Labels
        ax.text(-0.1, 0.5, source_name, rotation=90, va='center', ha='center')
        ax.text(1.1, 0.5, target_name, rotation=90, va='center', ha='center')
        
        ax.set_xlim(-0.2, 1.2)
        ax.set_ylim(-0.1, 1.1)
        ax.axis('off')
        
    def _compute_information_flow(
        self,
        features: torch.Tensor,
        method: str
    ) -> np.ndarray:
        """Compute information flow between layers."""
        num_layers = features.shape[1]
        flow_matrix = np.zeros((num_layers, num_layers))
        
        for i in range(num_layers):
            for j in range(num_layers):
                if i != j:
                    if method == 'mutual_information':
                        # Simplified MI calculation
                        feat_i = features[:, i].flatten().cpu().numpy()
                        feat_j = features[:, j].flatten().cpu().numpy()
                        
                        # Discretize
                        bins = 10
                        hist, _, _ = np.histogram2d(feat_i, feat_j, bins=bins)
                        
                        # Compute MI
                        pxy = hist / hist.sum()
                        px = pxy.sum(axis=1)
                        py = pxy.sum(axis=0)
                        
                        mi = 0
                        for x in range(bins):
                            for y in range(bins):
                                if pxy[x, y] > 0:
                                    mi += pxy[x, y] * np.log(pxy[x, y] / (px[x] * py[y] + 1e-10))
                                    
                        flow_matrix[i, j] = mi
                        
                    elif method == 'correlation':
                        feat_i = features[:, i].flatten()
                        feat_j = features[:, j].flatten()
                        
                        corr = torch.corrcoef(torch.stack([feat_i, feat_j]))[0, 1]
                        flow_matrix[i, j] = abs(corr.item())
                        
        return flow_matrix
        
    def _compute_feature_connection(
        self,
        features: torch.Tensor,
        layer: int,
        feat_i: int,
        feat_j: int
    ) -> float:
        """Compute connection strength between features across layers."""
        curr_feat = features[:, layer, :, feat_i].flatten()
        next_feat = features[:, layer + 1, :, feat_j].flatten()
        
        # Use correlation as connection strength
        if curr_feat.std() > 0 and next_feat.std() > 0:
            corr = torch.corrcoef(torch.stack([curr_feat, next_feat]))[0, 1]
            return abs(corr.item())
        else:
            return 0.0