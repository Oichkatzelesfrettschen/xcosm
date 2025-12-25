"""
XCOSM - eXceptional COSMological Framework
===========================================

XCOSM explores connections between exceptional algebraic structures (octonions,
Jordan algebras, F₄/E₆/G₂ symmetries), discrete pregeometry (bigraph dynamics
with Ollivier-Ricci curvature), and observational cosmology.

Key Components:
--------------
- core: Fundamental algebraic structures (Octonions, QCD running, entropic cosmology)
- models: Physical models (Spandrel cosmology, Helmholtz EOS, fractal flames)
- engines: Computational engines (CCF bigraph, flame simulations, Riemann hydro)
- data: Unified data access (Pantheon+, DESI BAO, LHC, GW sensitivity)
- analysis: Analysis scripts and derivations

Research Program:
-----------------
- Paper 1: Spandrel - SNe Ia progenitor evolution mimicking dark energy
- Paper 2: Scale-dependent H₀ smoothing estimator
- Paper 3: CCF curvature convergence (Ollivier-Ricci → Ricci)

Version: 0.2.0
"""

__version__ = "0.2.0"

# Core exports
from xcosm.core import (
    DiscreteSpacetime,
    E_z_entropic,
    Jordan3O,
    Octonion,
    PartitionFunction,
    alpha_s_4loop,
    distance_modulus,
    luminosity_distance,
    running_mass,
)

# Data exports (convenience loaders)
from xcosm.data import (
    load_cms_flow,
    load_desi_bao,
    load_et_sensitivity,
    load_lisa_sensitivity,
    load_pantheon,
    load_sdss_bao,
    load_w_mass,
)

# Engine exports
from xcosm.engines import (
    CCFParameters,
    CosmologicalBigraphEngine,
    PhysicalParameters,
    SpectralNSSolver,
)

# Model exports
from xcosm.models import (
    DESIConstraints,
    PhantomCrossingAnalysis,
    SpandrelEvolutionEquation,
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
    # Data loaders
    "load_pantheon",
    "load_desi_bao",
    "load_sdss_bao",
    "load_cms_flow",
    "load_w_mass",
    "load_lisa_sensitivity",
    "load_et_sensitivity",
]
