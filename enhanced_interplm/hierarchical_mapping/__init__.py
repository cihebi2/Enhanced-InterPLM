# 层次化映射（核心创新4） 
"""Hierarchical feature-function mapping modules."""

from .hierarchical_mapper import (
    AminoAcidPropertyMapper,
    SecondaryStructureMapper,
    DomainFunctionMapper,
    CrossLevelIntegrator,
    ProteinAnnotation
)

__all__ = [
    "AminoAcidPropertyMapper",
    "SecondaryStructureMapper",
    "DomainFunctionMapper",
    "CrossLevelIntegrator",
    "ProteinAnnotation",
]