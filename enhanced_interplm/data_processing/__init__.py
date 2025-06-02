# 数据处理（继承InterPLM） 
"""Data processing and loading utilities."""

from .multimodal_dataset import (
    MultiModalProteinDataset,
    CollateFunction,
    create_protein_dataloaders
)
from .structure_processor import (
    StructureProcessor,
    StructuralFeatureExtractor
)
from .evolution_data import (
    EvolutionaryDataProcessor,
    CoevolutionNetwork
)

__all__ = [
    "MultiModalProteinDataset",
    "CollateFunction",
    "create_protein_dataloaders",
    "StructureProcessor",
    "StructuralFeatureExtractor",
    "EvolutionaryDataProcessor",
    "CoevolutionNetwork",
]