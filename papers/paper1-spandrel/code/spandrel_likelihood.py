"""
Spandrel Likelihood Module - Compatibility Shim

This module re-exports likelihood classes from spandrel_core for backward compatibility.
The canonical source is now spandrel_core.likelihood.

Prefer importing directly from spandrel_core:
    from spandrel_core import SpandrelLikelihood, CosmologyParams
"""

# Re-export all likelihood components from spandrel_core
from spandrel_core.likelihood import (
    CosmologyParams,
    StandardizationParams,
    EvolutionParams,
    SpandrelLikelihood,
    compute_model_comparison,
)

__all__ = [
    "CosmologyParams",
    "StandardizationParams",
    "EvolutionParams",
    "SpandrelLikelihood",
    "compute_model_comparison",
]
