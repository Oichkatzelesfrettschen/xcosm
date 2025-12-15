"""
CCF Parameters Module
=====================

Defines the fundamental parameters of the Computational Cosmogenesis Framework.

DEPRECATED: This module now imports from cosmos.core.parameters for backwards compatibility.
Please import directly from cosmos.core.parameters instead.
"""

# Import from the canonical location
from cosmos.core.parameters import CCFParameters, SimulationConfig

# Re-export for backwards compatibility
__all__ = ['CCFParameters', 'SimulationConfig']
