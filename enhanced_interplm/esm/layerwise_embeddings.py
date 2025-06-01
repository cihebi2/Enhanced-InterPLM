# 多层embedding提取 
# enhanced_interplm/esm/layerwise_embeddings.py

"""
Extended ESM embedding extraction with layer-wise tracking and feature evolution analysis.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
from pathlib import Path
from tqdm import tqdm
import h5py
from esm import pretrained
from transformers import EsmModel, EsmTokenizer, EsmForMaskedLM
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LayerwiseEmbeddingConfig:
    """Configuration for layer-wise embedding extraction."""
    model_name: str = "esm2_t6_8M_UR50D"
    layers: List[int] = None
    batch_size: int = 8
    max_seq_length: int = 1022
    device: str = "cuda"
    save_attention_weights: bool = False
    save_hidden_states: bool = True
    normalize_embeddings: bool = False
    dtype: torch.dtype = torch.float32
    
    def __post_init__(self):
        if self.layers is None:
            # Default layers based on model
            if "8M" in self.model_name:
                self.layers = list(range(1, 7))  # 6 layers
            elif "35M" in self.model_name:
                self.layers = list(range(1, 13))  # 12 layers
            elif "150M" in self.model_name:
                self.layers = list(range(1, 31))  # 30 layers
            elif "650M" in self.model_name:
                self.layers = list(range(1, 34))  # 33 layers
            else:
                self.layers = [1, 2, 3, 4, 5, 6]  # Default


class LayerwiseEmbeddingExtractor:
    """
    Extract and analyze embeddings from all layers of ESM models.
    """
    
    def __init__(self, config: LayerwiseEmbeddingConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        # Load model and tokenizer
        self._load_model()
        
        # Initialize tracking
        self.embedding_stats = {}
        
    def _load_model(self):
        """Load ESM model and tokenizer."""
        logger.info(f"Loading ESM model: {self.config.model_name}")
        
        # Use transformers library for more control
        model_name = f"facebook/{self.config.model_name}"
        self.tokenizer = EsmTokenizer.from_pretrained(model_name)
        self.model = EsmForMaskedLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        # Get model info
        self.num_layers = self.model.config.num_hidden_layers
        self.hidden_size = self.model.config.hidden_size
        
        logger.info(f"Model loaded: {self.num_layers} layers, {self.hidden_size} hidden size")
        
    def extract_embeddings(
        self,
        sequences: Union[str, List[str]],
        return_contacts: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Extract embeddings from specified layers.
        
        Args:
            sequences: Single sequence or list of sequences
            return_contacts: Whether to compute and return contact predictions
            
        Returns:
            Dictionary containing:
                - 'embeddings': [batch, layers, seq_len, hidden_size]
                - 'attention': [batch, layers, heads, seq_len, seq_len] (if enabled)
                - 'contacts': [batch, seq_len, seq_len] (if requested)
        """
        if isinstance(sequences, str):
            sequences = [sequences]
            
        all_embeddings = []
        all_attention = [] if self.config.save_attention_weights else None
        
        # Process in batches
        for i in range(0, len(sequences), self.config.batch_size):
            batch_seqs = sequences[i:i + self.config.batch_size]
            batch_results = self._process_batch(batch_seqs)
            
            all_embeddings.append(batch_results['embeddings'])
            if all_attention is not None:
                all_attention.append(batch_results['attention'])
                
        # Concatenate results
        results = {
            'embeddings': torch.cat(all_embeddings, dim=0)
        }
        
        if all_attention is not None:
            results['attention'] = torch.cat(all_attention, dim=0)
            
        # Compute contacts if requested
        if return_contacts:
            results['contacts'] = self._compute_contacts(results['attention'])
            
        return results
        
    def _process_batch(self, sequences: List[str]) -> Dict[str, torch.Tensor]:
        """Process a batch of sequences."""
        # Tokenize
        inputs = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_length
        ).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                output_attentions=self.config.save_attention_weights
            )
            
        # Extract embeddings from specified layers
        hidden_states = outputs.hidden_states  # Tuple of [batch, seq, hidden]
        
        # Stack selected layers
        selected_embeddings = []
        for layer_idx in self.config.layers:
            if layer_idx <= len(hidden_states) - 1:
                layer_embeddings = hidden_states[layer_idx]
                
                # Remove special tokens (CLS, SEP, PAD)
                batch_size, seq_len, hidden_size = layer_embeddings.shape
                
                # Get actual sequence lengths
                seq_lens = inputs.attention_mask.sum(dim=1) - 2  # Subtract CLS and SEP
                
                # Extract only sequence positions
                clean_embeddings = []
                for b in range(batch_size):
                    seq_emb = layer_embeddings[b, 1:seq_lens[b]+1]  # Skip CLS, stop before SEP
                    clean_embeddings.append(seq_emb)
                    
                # Pad to max length in batch
                max_len = max(emb.shape[0] for emb in clean_embeddings)
                padded_embeddings = []
                for emb in clean_embeddings:
                    if emb.shape[0] < max_len:
                        padding = torch.zeros(max_len - emb.shape[0], hidden_size).to(emb.device)
                        emb = torch.cat([emb, padding], dim=0)
                    padded_embeddings.append(emb)
                    
                layer_embeddings = torch.stack(padded_embeddings)
                
                if self.config.normalize_embeddings:
                    layer_embeddings = F.normalize(layer_embeddings, p=2, dim=-1)
                    
                selected_embeddings.append(layer_embeddings)
                
        embeddings = torch.stack(selected_embeddings, dim=1)  # [batch, layers, seq, hidden]
        
        results = {'embeddings': embeddings}
        
        # Extract attention if requested
        if self.config.save_attention_weights and outputs.attentions is not None:
            selected_attention = []
            for layer_idx in self.config.layers:
                if layer_idx <= len(outputs.attentions) - 1:
                    selected_attention.append(outputs.attentions[layer_idx])
            results['attention'] = torch.stack(selected_attention, dim=1)
            
        return results
        
    def extract_and_save(
        self,
        sequences: List[str],
        sequence_ids: List[str],
        output_path: Path,
        chunk_size: int = 100
    ):
        """
        Extract embeddings and save to HDF5 file.
        
        Args:
            sequences: List of sequences
            sequence_ids: List of sequence identifiers
            output_path: Path to save HDF5 file
            chunk_size: Number of sequences to process at once
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(output_path, 'w') as f:
            # Create datasets
            f.attrs['model'] = self.config.model_name
            f.attrs['layers'] = self.config.layers
            f.attrs['num_sequences'] = len(sequences)
            
            # Process in chunks
            for chunk_start in tqdm(range(0, len(sequences), chunk_size), desc="Extracting embeddings"):
                chunk_end = min(chunk_start + chunk_size, len(sequences))
                chunk_seqs = sequences[chunk_start:chunk_end]
                chunk_ids = sequence_ids[chunk_start:chunk_end]
                
                # Extract embeddings
                results = self.extract_embeddings(chunk_seqs)
                
                # Save to HDF5
                for i, seq_id in enumerate(chunk_ids):
                    grp = f.create_group(seq_id)
                    grp.create_dataset(
                        'embeddings',
                        data=results['embeddings'][i].cpu().numpy(),
                        compression='gzip'
                    )
                    grp.attrs['sequence'] = chunk_seqs[i]
                    grp.attrs['length'] = len(chunk_seqs[i])
                    
                    if 'attention' in results:
                        grp.create_dataset(
                            'attention',
                            data=results['attention'][i].cpu().numpy(),
                            compression='gzip'
                        )
                        
        logger.info(f"Saved embeddings to {output_path}")
        
    def analyze_layer_evolution(
        self,
        embeddings: torch.Tensor
    ) -> Dict[str, np.ndarray]:
        """
        Analyze how embeddings evolve across layers.
        
        Args:
            embeddings: [batch, layers, seq_len, hidden_size]
            
        Returns:
            Dictionary with evolution metrics
        """
        batch_size, num_layers, seq_len, hidden_size = embeddings.shape
        
        metrics = {}
        
        # Layer-wise similarity
        layer_similarities = []
        for i in range(num_layers - 1):
            curr_layer = embeddings[:, i].reshape(-1, hidden_size)
            next_layer = embeddings[:, i + 1].reshape(-1, hidden_size)
            
            # Cosine similarity
            cos_sim = F.cosine_similarity(curr_layer, next_layer, dim=-1)
            layer_similarities.append(cos_sim.mean().item())
            
        metrics['layer_similarities'] = np.array(layer_similarities)
        
        # Feature magnitude evolution
        magnitudes = embeddings.norm(dim=-1).mean(dim=(0, 2))  # Average over batch and seq
        metrics['feature_magnitudes'] = magnitudes.cpu().numpy()
        
        # Feature diversity (measured by std)
        diversity = embeddings.std(dim=-1).mean(dim=(0, 2))
        metrics['feature_diversity'] = diversity.cpu().numpy()
        
        # Residual connections strength
        if num_layers > 1:
            first_layer = embeddings[:, 0]
            residual_strengths = []
            
            for i in range(1, num_layers):
                curr_layer = embeddings[:, i]
                # Project first layer to match dimensions if needed
                residual = F.cosine_similarity(
                    first_layer.reshape(-1, hidden_size),
                    curr_layer.reshape(-1, hidden_size),
                    dim=-1
                )
                residual_strengths.append(residual.mean().item())
                
            metrics['residual_strengths'] = np.array(residual_strengths)
            
        return metrics
        
    def _compute_contacts(
        self,
        attention: torch.Tensor,
        min_separation: int = 6
    ) -> torch.Tensor:
        """
        Compute contact predictions from attention weights.
        
        Args:
            attention: [batch, layers, heads, seq_len, seq_len]
            min_separation: Minimum sequence separation for contacts
            
        Returns:
            Contact predictions: [batch, seq_len, seq_len]
        """
        # Average over layers and heads
        avg_attention = attention.mean(dim=(1, 2))
        
        # Symmetrize
        contacts = (avg_attention + avg_attention.transpose(-1, -2)) / 2
        
        # Apply APC (Average Product Correction)
        a_i = contacts.sum(dim=-1, keepdim=True)
        a_j = contacts.sum(dim=-2, keepdim=True)
        a = contacts.sum(dim=(-1, -2), keepdim=True)
        
        apc = (a_i * a_j) / a
        contacts = contacts - apc
        
        # Mask diagonal and short-range contacts
        batch_size, seq_len, _ = contacts.shape
        mask = torch.ones_like(contacts)
        
        for i in range(seq_len):
            for j in range(seq_len):
                if abs(i - j) < min_separation:
                    mask[:, i, j] = 0
                    
        contacts = contacts * mask
        
        return contacts


class FeatureEvolutionTracker:
    """
    Track and analyze feature evolution across transformer layers.
    """
    
    def __init__(self, num_layers: int, feature_dim: int):
        self.num_layers = num_layers
        self.feature_dim = feature_dim
        
        # Initialize tracking matrices
        self.reset_tracking()
        
    def reset_tracking(self):
        """Reset all tracking statistics."""
        self.feature_trajectories = []
        self.layer_transitions = torch.zeros(self.num_layers - 1, self.feature_dim, self.feature_dim)
        self.feature_importance = torch.zeros(self.num_layers, self.feature_dim)
        self.sample_count = 0
        
    def update(self, embeddings: torch.Tensor):
        """
        Update tracking with new embeddings.
        
        Args:
            embeddings: [batch, layers, seq_len, feature_dim]
        """
        batch_size = embeddings.shape[0]
        
        # Update feature importance (average activation magnitude)
        self.feature_importance += embeddings.abs().mean(dim=(0, 2))
        
        # Update layer transitions
        for layer in range(self.num_layers - 1):
            curr_features = embeddings[:, layer].reshape(-1, self.feature_dim)
            next_features = embeddings[:, layer + 1].reshape(-1, self.feature_dim)
            
            # Compute feature correlation matrix
            corr = torch.matmul(curr_features.T, next_features) / curr_features.shape[0]
            self.layer_transitions[layer] += corr
            
        self.sample_count += batch_size
        
    def get_statistics(self) -> Dict[str, torch.Tensor]:
        """Get accumulated statistics."""
        if self.sample_count == 0:
            return {}
            
        return {
            'feature_importance': self.feature_importance / self.sample_count,
            'layer_transitions': self.layer_transitions / self.sample_count,
            'transition_entropy': self._compute_transition_entropy()
        }
        
    def _compute_transition_entropy(self) -> torch.Tensor:
        """Compute entropy of layer transitions."""
        # Normalize transition matrices
        transitions = F.softmax(self.layer_transitions.abs(), dim=-1)
        
        # Compute entropy
        entropy = -(transitions * torch.log(transitions + 1e-8)).sum(dim=-1)
        
        return entropy
        
    def identify_stable_features(self, threshold: float = 0.8) -> List[int]:
        """
        Identify features that remain stable across layers.
        
        Args:
            threshold: Correlation threshold for stability
            
        Returns:
            List of stable feature indices
        """
        stable_features = []
        
        for feat_idx in range(self.feature_dim):
            is_stable = True
            
            for layer in range(self.num_layers - 1):
                # Check if feature maintains high self-correlation
                self_corr = self.layer_transitions[layer, feat_idx, feat_idx]
                if self_corr < threshold:
                    is_stable = False
                    break
                    
            if is_stable:
                stable_features.append(feat_idx)
                
        return stable_features
        
    def identify_emerging_features(self, layer: int, threshold: float = 0.5) -> List[int]:
        """
        Identify features that emerge at a specific layer.
        
        Args:
            layer: Layer index
            threshold: Importance threshold
            
        Returns:
            List of emerging feature indices
        """
        if layer == 0:
            return []
            
        prev_importance = self.feature_importance[layer - 1]
        curr_importance = self.feature_importance[layer]
        
        # Features with large increase in importance
        importance_increase = curr_importance - prev_importance
        emerging = torch.where(importance_increase > threshold)[0].tolist()
        
        return emerging


def extract_multilayer_embeddings(
    sequences: List[str],
    model_name: str = "esm2_t6_8M_UR50D",
    layers: Optional[List[int]] = None,
    batch_size: int = 8,
    device: str = "cuda"
) -> Dict[str, torch.Tensor]:
    """
    Convenience function to extract embeddings from multiple layers.
    
    Args:
        sequences: List of protein sequences
        model_name: ESM model name
        layers: List of layer indices (None for all layers)
        batch_size: Batch size for processing
        device: Device to use
        
    Returns:
        Dictionary with embeddings and metadata
    """
    config = LayerwiseEmbeddingConfig(
        model_name=model_name,
        layers=layers,
        batch_size=batch_size,
        device=device
    )
    
    extractor = LayerwiseEmbeddingExtractor(config)
    results = extractor.extract_embeddings(sequences)
    
    # Add layer evolution analysis
    evolution_metrics = extractor.analyze_layer_evolution(results['embeddings'])
    results['evolution_metrics'] = evolution_metrics
    
    return results