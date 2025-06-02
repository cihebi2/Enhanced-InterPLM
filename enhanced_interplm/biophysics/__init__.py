# 生物物理约束（核心创新3） 
"""Biophysics-guided learning modules."""

from .physics_constraints import (
    BiophysicsConstraintModule,
    BiophysicsGuidedSAE,
    AminoAcidProperties
)
from .hydrophobic_module import HydrophobicAnalyzer
from .electrostatic_module import ElectrostaticAnalyzer
from .hbond_network import HBondNetworkAnalyzer

__all__ = [
    "BiophysicsConstraintModule",
    "BiophysicsGuidedSAE",
    "AminoAcidProperties",
    "HydrophobicAnalyzer",
    "ElectrostaticAnalyzer",
    "HBondNetworkAnalyzer",
]