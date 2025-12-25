"""
COSMOS Models Module
====================

Physical models implementing the Spandrel Framework and related cosmological
and astrophysical phenomena.

Components:
----------
- spandrel_cosmology: Spandrel bias function and phantom dark energy mimicry
- D_z_model: Fractal dimension evolution with metallicity and age
- helmholtz_eos: Helmholtz equation of state for white dwarf matter
- gravitational_fractal: Fractal flame gravitational wave emission
- jordan_spectrum: Jordan algebra fermion mass spectrum
- crystallization_model: White dwarf crystallization physics
- spandrel_equations: Deflagration and detonation wave equations
- spectral_fractal: Spectral properties of fractal flames

Key Classes and Functions:
--------------------------
Spandrel Cosmology:
    - SpandrelEvolutionEquation: Complete bias evolution chain
    - DESIConstraints: DESI DR2 observational constraints
    - PhantomCrossingAnalysis: Phantom crossing analysis
    - JWSTHighzSN: JWST high-z supernova predictions
    - StretchEvolutionData: Stretch parameter evolution with redshift

Fractal Dimension Model:
    - D_from_metallicity: D(Z) from cosmic metallicity evolution
    - D_from_age: D(age) from progenitor age
    - D_total: Combined D(Z, age) model
    - D_of_z: Fractal dimension as function of redshift
    - D_from_x1: Invert stretch parameter to fractal dimension

Helmholtz EOS:
    - EOSResult: Equation of state result data structure

Gravitational Fractal:
    - FractalSurface: Fractal flame surface representation
    - GravitationalWaveEmission: GW emission from fractal flames
    - SpandrelGWEquation: Spandrel GW prediction equations

Jordan Spectrum:
    - JordanAlgebra: Jordan algebra structure and fermion masses
    - RGRunning: Renormalization group running
    - AEGAnalysis: AEG framework mass analysis
"""

# Spandrel cosmology
# Crystallization model
from xcosm.models.crystallization_model import (
    CrystallizationState,
)

# D(z) model
from xcosm.models.D_z_model import (
    D_BASELINE,
    Z_SOLAR,
    D_from_age,
    D_from_metallicity,
    D_from_x1,
    D_of_z,
    D_total,
)

# Gravitational fractal
from xcosm.models.gravitational_fractal import (
    DetectorSensitivity,
    ExpandingFractalFlame,
    FractalSurface,
    GravitationalWaveEmission,
    QuadrupoleTensor,
    SpandrelGWEquation,
)

# Helmholtz EOS
from xcosm.models.helmholtz_eos import (
    EOSResult,
)

# Jordan spectrum
from xcosm.models.jordan_spectrum import (
    AEGAnalysis,
    FermionMass,
    JordanAlgebra,
    RGRunning,
)
from xcosm.models.spandrel_cosmology import (
    DESIConstraints,
    JWSTFrontierData,
    JWSTHighzSN,
    PhantomCrossingAnalysis,
    SpandrelEvolutionEquation,
    SpandrelEvolutionParameters,
    StretchEvolutionData,
)

__all__ = [
    # Spandrel cosmology
    "SpandrelEvolutionParameters",
    "SpandrelEvolutionEquation",
    "DESIConstraints",
    "PhantomCrossingAnalysis",
    "JWSTHighzSN",
    "JWSTFrontierData",
    "StretchEvolutionData",
    # D(z) model
    "D_from_metallicity",
    "D_from_age",
    "D_total",
    "D_of_z",
    "D_from_x1",
    "D_BASELINE",
    "Z_SOLAR",
    # Helmholtz EOS
    "EOSResult",
    # Gravitational fractal
    "FractalSurface",
    "QuadrupoleTensor",
    "ExpandingFractalFlame",
    "GravitationalWaveEmission",
    "DetectorSensitivity",
    "SpandrelGWEquation",
    # Jordan spectrum
    "FermionMass",
    "JordanAlgebra",
    "RGRunning",
    "AEGAnalysis",
    # Crystallization model
    "CrystallizationState",
]
