# 时序SAE模块（核心创新1） 
"""Temporal-aware Sparse Autoencoder module."""

from .temporal_autoencoder import TemporalSAE, TemporalFeatureTracker
from .attention_modules import (
    TemporalCrossAttention,
    HierarchicalAttention,
    SparseAttention,
    CausalTemporalAttention,
    FeatureRoutingAttention
)
from .layer_tracking import (
    LayerTracker,
    LayerFeatureState,
    FeatureTransition
)

__all__ = [
    "TemporalSAE",
    "TemporalFeatureTracker",
    "TemporalCrossAttention",
    "HierarchicalAttention",
    "SparseAttention",
    "CausalTemporalAttention",
    "FeatureRoutingAttention",
    "LayerTracker",
    "LayerFeatureState",
    "FeatureTransition",
]