# enhanced_interplm/utils/helpers.py

"""
Utility functions for Enhanced InterPLM framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import json
import yaml
import pickle
import h5py
import pandas as pd
from datetime import datetime
import hashlib
import logging
from contextlib import contextmanager
import time
import psutil
import GPUtil

logger = logging.getLogger(__name__)


# ============================================================================
# Device and Memory Management
# ============================================================================

def get_optimal_device(gpu_id: Optional[int] = None) -> torch.device:
    """
    Get optimal device for computation with memory checking.
    
    Args:
        gpu_id: Specific GPU ID to use
        
    Returns:
        torch.device object
    """
    if not torch.cuda.is_available():
        logger.info("CUDA not available, using CPU")
        return torch.device('cpu')
        
    if gpu_id is not None:
        device = torch.device(f'cuda:{gpu_id}')
        logger.info(f"Using specified GPU: {gpu_id}")
        return device
        
    # Find GPU with most free memory
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            best_gpu = max(gpus, key=lambda x: x.memoryFree)
            device = torch.device(f'cuda:{best_gpu.id}')
            logger.info(f"Selected GPU {best_gpu.id} with {best_gpu.memoryFree}MB free memory")
            return device
    except:
        pass
        
    # Fallback to default GPU
    device = torch.device('cuda')
    logger.info("Using default CUDA device")
    return device


@contextmanager
def torch_memory_debug(device: torch.device, tag: str = ""):
    """Context manager for debugging GPU memory usage."""
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        start_memory = torch.cuda.memory_allocated(device)
        
    yield
    
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        end_memory = torch.cuda.memory_allocated(device)
        memory_diff = (end_memory - start_memory) / 1024 / 1024  # MB
        
        if memory_diff > 0:
            logger.debug(f"[{tag}] Memory increased by {memory_diff:.2f} MB")
            

def clear_gpu_cache():
    """Clear GPU cache and log memory status."""
    if torch.cuda.is_available():
        before = torch.cuda.memory_allocated() / 1024 / 1024
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        after = torch.cuda.memory_allocated() / 1024 / 1024
        logger.debug(f"Cleared GPU cache: {before:.1f}MB -> {after:.1f}MB")


# ============================================================================
# Data I/O Utilities
# ============================================================================

def save_checkpoint(
    model_dict: Dict[str, Any],
    path: Union[str, Path],
    metadata: Optional[Dict] = None
):
    """
    Save model checkpoint with metadata.
    
    Args:
        model_dict: Dictionary containing model states
        path: Save path
        metadata: Additional metadata to save
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model_dict,
        'timestamp': datetime.now().isoformat(),
        'pytorch_version': torch.__version__,
    }
    
    if metadata:
        checkpoint['metadata'] = metadata
        
    # Add checksum for integrity
    state_bytes = pickle.dumps(model_dict)
    checkpoint['checksum'] = hashlib.md5(state_bytes).hexdigest()
    
    torch.save(checkpoint, path)
    logger.info(f"Saved checkpoint to {path}")
    

def load_checkpoint(
    path: Union[str, Path],
    device: Optional[torch.device] = None,
    verify_checksum: bool = True
) -> Dict[str, Any]:
    """
    Load model checkpoint with verification.
    
    Args:
        path: Checkpoint path
        device: Device to load to
        verify_checksum: Whether to verify checkpoint integrity
        
    Returns:
        Checkpoint dictionary
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
        
    checkpoint = torch.load(path, map_location=device or 'cpu')
    
    if verify_checksum and 'checksum' in checkpoint:
        state_bytes = pickle.dumps(checkpoint['model_state_dict'])
        checksum = hashlib.md5(state_bytes).hexdigest()
        
        if checksum != checkpoint['checksum']:
            logger.warning("Checkpoint checksum mismatch - file may be corrupted")
            
    logger.info(f"Loaded checkpoint from {path}")
    return checkpoint


def save_features_hdf5(
    features: Dict[str, torch.Tensor],
    metadata: Dict[str, Any],
    path: Union[str, Path],
    compression: str = 'gzip'
):
    """
    Save features to HDF5 file with metadata.
    
    Args:
        features: Dictionary of feature tensors
        metadata: Metadata dictionary
        path: Save path
        compression: HDF5 compression type
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(path, 'w') as f:
        # Save metadata as attributes
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                f.attrs[key] = value
            else:
                f.attrs[key] = json.dumps(value)
                
        # Save features
        for name, tensor in features.items():
            f.create_dataset(
                name,
                data=tensor.cpu().numpy(),
                compression=compression
            )
            
    logger.info(f"Saved features to {path}")
    

def load_features_hdf5(
    path: Union[str, Path],
    feature_names: Optional[List[str]] = None
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Load features from HDF5 file.
    
    Args:
        path: HDF5 file path
        feature_names: Specific features to load (None for all)
        
    Returns:
        Features dictionary and metadata
    """
    path = Path(path)
    features = {}
    metadata = {}
    
    with h5py.File(path, 'r') as f:
        # Load metadata
        for key, value in f.attrs.items():
            try:
                metadata[key] = json.loads(value)
            except:
                metadata[key] = value
                
        # Load features
        if feature_names is None:
            feature_names = list(f.keys())
            
        for name in feature_names:
            if name in f:
                features[name] = f[name][:]
                
    return features, metadata


# ============================================================================
# Sequence and Structure Utilities
# ============================================================================

def validate_sequence(sequence: str) -> bool:
    """Validate protein sequence."""
    valid_aas = set('ACDEFGHIKLMNPQRSTVWY')
    return all(aa in valid_aas for aa in sequence.upper())


def pad_sequences(
    sequences: List[torch.Tensor],
    max_length: Optional[int] = None,
    padding_value: float = 0.0
) -> torch.Tensor:
    """
    Pad sequences to same length.
    
    Args:
        sequences: List of tensors with shape [seq_len, ...]
        max_length: Maximum length (None to use longest)
        padding_value: Value to use for padding
        
    Returns:
        Padded tensor with shape [batch, max_length, ...]
    """
    if not sequences:
        return torch.empty(0)
        
    # Find maximum length
    lengths = [seq.shape[0] for seq in sequences]
    if max_length is None:
        max_length = max(lengths)
        
    # Pad sequences
    padded = []
    for seq, length in zip(sequences, lengths):
        if length < max_length:
            pad_size = [max_length - length] + [0] * (seq.dim() - 1)
            seq = F.pad(seq, (0, 0) * (seq.dim() - 1) + tuple(pad_size), value=padding_value)
        elif length > max_length:
            seq = seq[:max_length]
        padded.append(seq)
        
    return torch.stack(padded)


def compute_sequence_similarity(seq1: str, seq2: str) -> float:
    """Compute sequence similarity using Blosum62."""
    from Bio import pairwise2
    from Bio.Align import substitution_matrices
    
    blosum62 = substitution_matrices.load("BLOSUM62")
    alignments = pairwise2.align.globalds(seq1, seq2, blosum62, -10, -0.5)
    
    if alignments:
        best_alignment = alignments[0]
        score = best_alignment.score
        max_score = max(
            sum(blosum62[aa, aa] for aa in seq1),
            sum(blosum62[aa, aa] for aa in seq2)
        )
        return score / max_score
    return 0.0


def compute_contact_map(
    coords: torch.Tensor,
    threshold: float = 8.0
) -> torch.Tensor:
    """
    Compute contact map from 3D coordinates.
    
    Args:
        coords: 3D coordinates [seq_len, 3]
        threshold: Distance threshold for contacts
        
    Returns:
        Binary contact map [seq_len, seq_len]
    """
    distances = torch.cdist(coords, coords)
    contacts = (distances < threshold).float()
    
    # Remove self-contacts
    contacts.fill_diagonal_(0)
    
    return contacts


# ============================================================================
# Feature Analysis Utilities
# ============================================================================

def compute_feature_statistics(
    features: torch.Tensor,
    dim: Optional[Union[int, Tuple[int, ...]]] = None
) -> Dict[str, torch.Tensor]:
    """
    Compute comprehensive statistics for features.
    
    Args:
        features: Feature tensor
        dim: Dimension(s) to compute statistics over
        
    Returns:
        Dictionary of statistics
    """
    stats = {
        'mean': features.mean(dim=dim),
        'std': features.std(dim=dim),
        'min': features.min(dim=dim)[0] if dim is not None else features.min(),
        'max': features.max(dim=dim)[0] if dim is not None else features.max(),
        'l1_norm': features.abs().mean(dim=dim),
        'l2_norm': features.norm(p=2, dim=dim),
        'sparsity': (features.abs() > 0.01).float().mean(dim=dim),
    }
    
    # Add percentiles
    if dim is not None:
        for p in [25, 50, 75, 90, 95, 99]:
            stats[f'p{p}'] = torch.quantile(features, p/100, dim=dim)
            
    return stats


def compute_feature_correlation(
    features1: torch.Tensor,
    features2: torch.Tensor,
    method: str = 'pearson'
) -> torch.Tensor:
    """
    Compute correlation between feature sets.
    
    Args:
        features1: First feature set [n_samples, n_features]
        features2: Second feature set [n_samples, n_features]
        method: Correlation method ('pearson' or 'spearman')
        
    Returns:
        Correlation matrix [n_features1, n_features2]
    """
    if method == 'pearson':
        # Standardize features
        features1 = (features1 - features1.mean(dim=0)) / (features1.std(dim=0) + 1e-8)
        features2 = (features2 - features2.mean(dim=0)) / (features2.std(dim=0) + 1e-8)
        
        # Compute correlation
        correlation = torch.matmul(features1.T, features2) / features1.shape[0]
        
    elif method == 'spearman':
        # Convert to ranks
        features1_rank = features1.argsort(dim=0).argsort(dim=0).float()
        features2_rank = features2.argsort(dim=0).argsort(dim=0).float()
        
        # Compute correlation on ranks
        correlation = compute_feature_correlation(features1_rank, features2_rank, 'pearson')
        
    else:
        raise ValueError(f"Unknown correlation method: {method}")
        
    return correlation


def identify_dead_features(
    features: torch.Tensor,
    threshold: float = 0.01,
    min_samples: int = 10
) -> List[int]:
    """
    Identify dead (never activated) features.
    
    Args:
        features: Feature tensor [batch, ..., n_features]
        threshold: Activation threshold
        min_samples: Minimum samples to consider
        
    Returns:
        List of dead feature indices
    """
    # Reshape to [n_samples, n_features]
    features_flat = features.reshape(-1, features.shape[-1])
    
    if features_flat.shape[0] < min_samples:
        logger.warning(f"Too few samples ({features_flat.shape[0]}) to identify dead features")
        return []
        
    # Count activations per feature
    activation_counts = (features_flat.abs() > threshold).sum(dim=0)
    
    # Dead features have zero activations
    dead_features = torch.where(activation_counts == 0)[0].tolist()
    
    return dead_features


# ============================================================================
# Evaluation Utilities
# ============================================================================

def compute_reconstruction_metrics(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """
    Compute comprehensive reconstruction metrics.
    
    Args:
        original: Original tensor
        reconstructed: Reconstructed tensor
        mask: Optional mask for valid positions
        
    Returns:
        Dictionary of metrics
    """
    if mask is not None:
        original = original[mask]
        reconstructed = reconstructed[mask]
        
    metrics = {}
    
    # MSE
    metrics['mse'] = F.mse_loss(reconstructed, original).item()
    
    # MAE
    metrics['mae'] = F.l1_loss(reconstructed, original).item()
    
    # Normalized error
    norm_error = torch.norm(reconstructed - original, dim=-1) / (torch.norm(original, dim=-1) + 1e-8)
    metrics['normalized_error'] = norm_error.mean().item()
    
    # Cosine similarity
    cos_sim = F.cosine_similarity(
        original.reshape(-1, original.shape[-1]),
        reconstructed.reshape(-1, reconstructed.shape[-1]),
        dim=-1
    )
    metrics['cosine_similarity'] = cos_sim.mean().item()
    
    # Variance explained
    total_var = original.var()
    residual_var = (original - reconstructed).var()
    metrics['variance_explained'] = (1 - residual_var / total_var).item()
    
    # R-squared
    ss_res = ((original - reconstructed) ** 2).sum()
    ss_tot = ((original - original.mean()) ** 2).sum()
    metrics['r_squared'] = (1 - ss_res / ss_tot).item()
    
    return metrics


def bootstrap_confidence_interval(
    data: np.ndarray,
    statistic: callable,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: Optional[int] = None
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval.
    
    Args:
        data: Input data
        statistic: Function to compute statistic
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level
        random_state: Random seed
        
    Returns:
        (statistic, lower_bound, upper_bound)
    """
    if random_state is not None:
        np.random.seed(random_state)
        
    # Original statistic
    original_stat = statistic(data)
    
    # Bootstrap
    bootstrap_stats = []
    n_samples = len(data)
    
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(data, size=n_samples, replace=True)
        bootstrap_stats.append(statistic(bootstrap_sample))
        
    # Compute confidence interval
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_bound = np.percentile(bootstrap_stats, lower_percentile)
    upper_bound = np.percentile(bootstrap_stats, upper_percentile)
    
    return original_stat, lower_bound, upper_bound


# ============================================================================
# Configuration Utilities
# ============================================================================

def merge_configs(base_config: Dict, update_config: Dict) -> Dict:
    """
    Recursively merge configuration dictionaries.
    
    Args:
        base_config: Base configuration
        update_config: Configuration updates
        
    Returns:
        Merged configuration
    """
    merged = base_config.copy()
    
    for key, value in update_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
            
    return merged


def flatten_dict(d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
    """
    Flatten nested dictionary.
    
    Args:
        d: Dictionary to flatten
        parent_key: Parent key prefix
        sep: Separator for keys
        
    Returns:
        Flattened dictionary
    """
    items = []
    
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
            
    return dict(items)


# ============================================================================
# Timing and Profiling
# ============================================================================

class Timer:
    """Simple timer context manager."""
    
    def __init__(self, name: str = "Operation", logger: Optional[logging.Logger] = None):
        self.name = name
        self.logger = logger or logging.getLogger(__name__)
        
    def __enter__(self):
        self.start_time = time.time()
        self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        return self
        
    def __exit__(self, *args):
        elapsed_time = time.time() - self.start_time
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_used = end_memory - self.start_memory
        
        self.logger.info(
            f"{self.name} took {elapsed_time:.2f}s, "
            f"memory change: {memory_used:+.1f}MB"
        )


@contextmanager
def set_random_seed(seed: int):
    """Context manager for setting random seeds."""
    # Save current state
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    
    if torch.cuda.is_available():
        cuda_state = torch.cuda.get_rng_state()
        
    # Set new seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    try:
        yield
    finally:
        # Restore state
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(cuda_state)