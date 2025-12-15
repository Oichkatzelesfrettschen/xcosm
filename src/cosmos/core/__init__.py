"""
COSMOS Core Module
==================

Fundamental algebraic structures and physics implementations for the AEG framework.

Components:
----------
- octonion_algebra: Octonion and J₃(O) Jordan algebra implementation
- qcd_running: QCD running coupling and mass evolution (4-loop)
- entropic_cosmology: Entropic dark energy model and cosmological distances
- partition_function: Discrete to continuum bridge via partition functions
- spacetime_projection: Spacetime projection and gravitational field mapping

Key Classes and Functions:
--------------------------
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

# Octonion algebra
from cosmos.core.octonion_algebra import (
    Octonion,
    Jordan3O,
    FANO_LINES,
    classify_j3o_components,
)

# QCD running
from cosmos.core.qcd_running import (
    alpha_s_1loop,
    alpha_s_4loop,
    running_mass,
    beta_coefficients,
    gamma_coefficients,
    compute_mass_ratios,
    find_unification_scale,
    ALPHA_S_MZ,
    MZ,
    M_ELECTRON,
    M_MUON,
    M_TAU,
)

# Entropic cosmology
from cosmos.core.entropic_cosmology import (
    w_entropic,
    rho_de_ratio,
    E_z_entropic,
    E_z_LCDM,
    comoving_distance,
    luminosity_distance,
    distance_modulus,
    distance_modulus_array,
    chi_squared,
    log_likelihood,
    log_prior,
    log_posterior,
    generate_synthetic_pantheon,
    grid_search,
    run_mcmc_simple,
    compute_evidence_ratio,
)

# Partition function
from cosmos.core.partition_function import (
    DiscreteSpacetime,
    PartitionFunction,
    continuum_partition_function,
    verify_continuum_limit,
)

__all__ = [
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
