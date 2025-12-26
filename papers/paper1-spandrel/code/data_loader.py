"""
Data Loader for Spandrel Analysis - Compatibility Shim

This module re-exports data structures from spandrel_core for backward compatibility.
The canonical source is now spandrel_core.data.

Prefer importing directly from spandrel_core:
    from spandrel_core import SNDataset, PantheonPlusLoader
"""

# Re-export all data components from spandrel_core
from spandrel_core.data import (
    SNDataset,
    PantheonPlusLoader,
    SimulatedDataLoader,
    split_by_host_mass,
    split_by_redshift,
)

__all__ = [
    "SNDataset",
    "PantheonPlusLoader",
    "SimulatedDataLoader",
    "split_by_host_mass",
    "split_by_redshift",
]
