"""
COSMOS Core Module
==================

Fundamental algebraic structures and physics implementations for the AEG framework.

Components:
----------
- planck_units: Planck unit system with SI/natural unit conversions
- octonion_algebra: Octonion and J₃(O) Jordan algebra implementation
- qcd_running: QCD running coupling and mass evolution (4-loop)
- entropic_cosmology: Entropic dark energy model and cosmological distances
- partition_function: Discrete to continuum bridge via partition functions
- spacetime_projection: Spacetime projection and gravitational field mapping

Key Classes and Functions:
--------------------------
Planck Units:
    - Planck: Base Planck units (length, time, mass, temperature, charge)
    - SI: SI fundamental constants (c, hbar, G, k_B, etc.)
    - DIMENSIONLESS: Dimensionless fundamental constants (α, α_s, etc.)
    - MASS_SCALES: SM particle masses in Planck units
    - Unit conversion functions: to_planck_*, from_planck_*

Octonion Algebra:
    - Octonion: 8-dimensional octonion algebra
    - Jordan3O: J₃(O) exceptional Jordan algebra (27-dimensional)

QCD Running:
    - alpha_s_4loop: 4-loop running strong coupling
    - alpha_s_1loop: 1-loop running strong coupling
    - running_mass: Quark mass running with RG flow
    - beta_coefficients: QCD β-function coefficients
    - gamma_coefficients: Mass anomalous dimension coefficients

Entropic Cosmology:
    - E_z_entropic: Dimensionless Hubble parameter for entropic model
    - w_entropic: Entropic equation of state parameter
    - luminosity_distance: Luminosity distance with entropic corrections
    - distance_modulus: Distance modulus for supernova fitting
    - run_mcmc_simple: MCMC parameter estimation

Partition Function:
    - PartitionFunction: Discrete-to-continuum partition function bridge
    - DiscreteSpacetime: Discrete spacetime representation
"""

# Planck units system (fundamental)
# Entropic cosmology
from xcosm.core.entropic_cosmology import (
    E_z_entropic,
    E_z_LCDM,
    chi_squared,
    comoving_distance,
    compute_evidence_ratio,
    distance_modulus,
    distance_modulus_array,
    generate_synthetic_pantheon,
    grid_search,
    log_likelihood,
    log_posterior,
    log_prior,
    luminosity_distance,
    rho_de_ratio,
    run_mcmc_simple,
    w_entropic,
)

# Octonion algebra
from xcosm.core.octonion_algebra import (
    FANO_LINES,
    Jordan3O,
    Octonion,
    classify_j3o_components,
)

# Partition function
from xcosm.core.partition_function import (
    DiscreteSpacetime,
    PartitionFunction,
    continuum_partition_function,
    verify_continuum_limit,
)
from xcosm.core.planck_units import (
    COSMOLOGY_PLANCK,
    DIMENSIONLESS,
    LENGTH_SCALES,
    MASS_SCALES,
    SI,
    CosmologyPlanck,
    DimensionlessConstants,
    H0_from_planck,
    H0_to_planck,
    LengthScalesPlanck,
    MassScalesPlanck,
    Planck,
    from_planck_energy,
    from_planck_length,
    from_planck_mass,
    from_planck_time,
    natural_to_planck,
    planck_to_natural,
    to_planck_energy,
    to_planck_length,
    to_planck_mass,
    to_planck_time,
    validate_planck_units,
)

# QCD running
from xcosm.core.qcd_running import (
    ALPHA_S_MZ,
    M_ELECTRON,
    M_MUON,
    M_TAU,
    MZ,
    alpha_s_1loop,
    alpha_s_4loop,
    beta_coefficients,
    compute_mass_ratios,
    find_unification_scale,
    gamma_coefficients,
    running_mass,
)

__all__ = [
    # Planck units system
    "SI",
    "Planck",
    "DimensionlessConstants",
    "MassScalesPlanck",
    "LengthScalesPlanck",
    "CosmologyPlanck",
    "DIMENSIONLESS",
    "MASS_SCALES",
    "LENGTH_SCALES",
    "COSMOLOGY_PLANCK",
    "to_planck_mass",
    "from_planck_mass",
    "to_planck_length",
    "from_planck_length",
    "to_planck_time",
    "from_planck_time",
    "to_planck_energy",
    "from_planck_energy",
    "natural_to_planck",
    "planck_to_natural",
    "H0_to_planck",
    "H0_from_planck",
    "validate_planck_units",
    # Octonion algebra
    "Octonion",
    "Jordan3O",
    "FANO_LINES",
    "classify_j3o_components",
    # QCD running
    "alpha_s_1loop",
    "alpha_s_4loop",
    "running_mass",
    "beta_coefficients",
    "gamma_coefficients",
    "compute_mass_ratios",
    "find_unification_scale",
    "ALPHA_S_MZ",
    "MZ",
    "M_ELECTRON",
    "M_MUON",
    "M_TAU",
    # Entropic cosmology
    "w_entropic",
    "rho_de_ratio",
    "E_z_entropic",
    "E_z_LCDM",
    "comoving_distance",
    "luminosity_distance",
    "distance_modulus",
    "distance_modulus_array",
    "chi_squared",
    "log_likelihood",
    "log_prior",
    "log_posterior",
    "generate_synthetic_pantheon",
    "grid_search",
    "run_mcmc_simple",
    "compute_evidence_ratio",
    # Partition function
    "DiscreteSpacetime",
    "PartitionFunction",
    "continuum_partition_function",
    "verify_continuum_limit",
]
