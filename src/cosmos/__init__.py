"""
COSMOS - A Research Physics Simulatorium
=========================================

The COSMOS package implements the Algebraic Emergence Geometry (AEG) framework
and the Spandrel Framework for cosmological and astrophysical simulations.

Key Components:
--------------
- core: Fundamental algebraic structures (Octonions, QCD running, entropic cosmology)
- models: Physical models (Spandrel cosmology, Helmholtz EOS, fractal flames)
- engines: Computational engines (CCF bigraph, flame simulations, Riemann hydro)
- analysis: Analysis scripts and derivations

Version: 0.1.0
"""

__version__ = "0.1.0"

# Core exports
from cosmos.core import (
    Octonion,
    Jordan3O,
    alpha_s_4loop,
    running_mass,
    PartitionFunction,
    DiscreteSpacetime,
    E_z_entropic,
    luminosity_distance,
    distance_modulus,
)

# Model exports
from cosmos.models import (
    SpandrelEvolutionEquation,
    DESIConstraints,
    PhantomCrossingAnalysis,
)

# Engine exports
from cosmos.engines import (
    CosmologicalBigraphEngine,
    CCFParameters,
    SpectralNSSolver,
    PhysicalParameters,
)

__all__ = [
    "__version__",
    # Core
    "Octonion",
    "Jordan3O",
    "alpha_s_4loop",
    "running_mass",
    "PartitionFunction",
    "DiscreteSpacetime",
    "E_z_entropic",
    "luminosity_distance",
    "distance_modulus",
    # Models
    "SpandrelEvolutionEquation",
    "DESIConstraints",
    "PhantomCrossingAnalysis",
    # Engines
    "CosmologicalBigraphEngine",
    "CCFParameters",
    "SpectralNSSolver",
    "PhysicalParameters",
]
