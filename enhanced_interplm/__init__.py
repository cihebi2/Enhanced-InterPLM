# Enhanced InterPLM 主代码模块 
"""
Enhanced InterPLM: Advanced Interpretability for Protein Language Models

A comprehensive framework for understanding protein language models through:
- Temporal-aware sparse autoencoders
- Functional circuit discovery
- Biophysics-guided learning
- Hierarchical feature-function mapping
"""

__version__ = "0.1.0"
__author__ = "Enhanced InterPLM Team"

# Core modules
from enhanced_interplm.temporal_sae.temporal_autoencoder import (
    TemporalSAE,
    TemporalFeatureTracker
)

from enhanced_interplm.circuit_discovery.graph_builder import (
    DynamicGraphBuilder,
    CircuitMotifDetector,
    CircuitValidator
)

from enhanced_interplm.biophysics.physics_constraints import (
    BiophysicsConstraintModule,
    BiophysicsGuidedSAE,
    AminoAcidProperties
)

from enhanced_interplm.hierarchical_mapping.hierarchical_mapper import (
    AminoAcidPropertyMapper,
    SecondaryStructureMapper,
    DomainFunctionMapper,
    CrossLevelIntegrator
)

# Utilities
from enhanced_interplm.utils.helpers import (
    get_optimal_device,
    save_checkpoint,
    load_checkpoint,
    compute_reconstruction_metrics
)

__all__ = [
    # Core classes
    "TemporalSAE",
    "TemporalFeatureTracker",
    "DynamicGraphBuilder",
    "CircuitMotifDetector",
    "CircuitValidator",
    "BiophysicsConstraintModule",
    "BiophysicsGuidedSAE",
    "AminoAcidPropertyMapper",
    "SecondaryStructureMapper",
    "DomainFunctionMapper",
    "CrossLevelIntegrator",
    # Version
    "__version__",
]