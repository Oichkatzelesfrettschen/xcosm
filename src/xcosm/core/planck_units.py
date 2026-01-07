"""
Planck Units System
===================

Fundamental unit system for COSMOS physics framework.

In Planck units, the following fundamental constants are set to unity:
    c = 1       (speed of light)
    ℏ = 1       (reduced Planck constant)
    G = 1       (gravitational constant)
    k_B = 1     (Boltzmann constant)
    ε_0 = 1/4π  (permittivity, via Lorentz-Heaviside convention)

This gives natural units where:
    [Length] = [Time] = [Mass]⁻¹ = [Energy]⁻¹ = [Temperature]⁻¹

Conversions:
    1 Planck length  ℓ_P = 1.616255 × 10⁻³⁵ m
    1 Planck time    t_P = 5.391247 × 10⁻⁴⁴ s
    1 Planck mass    M_P = 2.176434 × 10⁻⁸ kg = 1.220890 × 10¹⁹ GeV
    1 Planck temp    T_P = 1.416784 × 10³² K
    1 Planck charge  q_P = 1.875546 × 10⁻¹⁸ C (= √(4πε_0ℏc) = e/√α)

All physics in this framework should be expressed in Planck units internally,
with conversions applied only at I/O boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

# =============================================================================
# FUNDAMENTAL CONSTANTS IN SI UNITS (CODATA 2018)
# =============================================================================


class SI:
    """SI values of fundamental constants (CODATA 2018)."""

    # Defining constants (exact by definition in SI)
    c: float = 299_792_458.0  # m/s (exact)
    h: float = 6.62607015e-34  # J⋅s (exact)
    hbar: float = 1.054571817e-34  # J⋅s (= h/2π)
    e: float = 1.602176634e-19  # C (exact)
    k_B: float = 1.380649e-23  # J/K (exact)
    N_A: float = 6.02214076e23  # mol⁻¹ (exact)

    # Measured constants
    G: float = 6.67430e-11  # m³/(kg⋅s²) ± 0.00015
    alpha: float = 7.2973525693e-3  # fine structure (≈ 1/137.036)
    alpha_s_MZ: float = 0.1179  # strong coupling at M_Z

    # Masses
    m_e: float = 9.1093837015e-31  # kg (electron)
    m_p: float = 1.67262192369e-27  # kg (proton)
    m_n: float = 1.67492749804e-27  # kg (neutron)

    # Derived electromagnetic
    mu_0: float = 1.25663706212e-6  # N/A² (magnetic permeability)
    eps_0: float = 8.8541878128e-12  # F/m (electric permittivity)


# =============================================================================
# PLANCK UNITS (derived from SI)
# =============================================================================


class Planck:
    """
    Planck units derived from fundamental constants.

    In these units: c = ℏ = G = k_B = 1
    """

    # Base Planck units
    length: float = np.sqrt(SI.hbar * SI.G / SI.c**3)  # ℓ_P = 1.616255e-35 m
    time: float = np.sqrt(SI.hbar * SI.G / SI.c**5)  # t_P = 5.391247e-44 s
    mass: float = np.sqrt(SI.hbar * SI.c / SI.G)  # M_P = 2.176434e-8 kg
    temperature: float = np.sqrt(SI.hbar * SI.c**5 / (SI.G * SI.k_B**2))  # T_P = 1.416784e32 K
    charge: float = np.sqrt(4 * np.pi * SI.eps_0 * SI.hbar * SI.c)  # q_P = 1.875546e-18 C

    # Derived Planck units
    energy: float = mass * SI.c**2  # E_P = M_P c² = 1.956e9 J
    momentum: float = mass * SI.c  # p_P = M_P c
    force: float = SI.c**4 / SI.G  # F_P = c⁴/G = 1.210e44 N
    power: float = SI.c**5 / SI.G  # P_P = c⁵/G = 3.628e52 W
    density: float = SI.c**5 / (SI.hbar * SI.G**2)  # ρ_P = c⁵/(ℏG²) = 5.155e96 kg/m³

    # Energy scale in GeV (useful for particle physics)
    mass_GeV: float = 1.220890e19  # M_P in GeV
    energy_GeV: float = 1.220890e19  # E_P in GeV


# =============================================================================
# DIMENSIONLESS CONSTANTS (fundamental, no unit dependence)
# =============================================================================


@dataclass(frozen=True)
class DimensionlessConstants:
    """
    Truly fundamental dimensionless constants.

    These are pure numbers that do not depend on unit choice.
    They are the ONLY fundamental parameters in physics.
    """

    # Electromagnetic coupling (fine structure constant)
    alpha: float = 7.2973525693e-3  # ≈ 1/137.036

    # Strong coupling at Z mass
    alpha_s_MZ: float = 0.1179  # α_s(M_Z)

    # Weak mixing angle (sin²θ_W at M_Z, MS-bar)
    sin2_theta_W: float = 0.23121  # Weinberg angle

    # Mass ratios (dimensionless)
    m_e_over_m_p: float = 5.446170214e-4  # electron/proton mass ratio
    m_n_over_m_p: float = 1.00137841931  # neutron/proton mass ratio
    m_mu_over_m_e: float = 206.7682830  # muon/electron mass ratio
    m_tau_over_m_e: float = 3477.23  # tau/electron mass ratio

    # Gravitational coupling (at Planck scale, = 1 by definition)
    # At low energies: α_G = G m_p² / (ℏc) ≈ 5.9e-39
    alpha_G_proton: float = 5.9e-39  # gravitational "fine structure"

    # Cosmological parameters (dimensionless)
    Omega_m: float = 0.315  # matter density fraction
    Omega_Lambda: float = 0.685  # dark energy density fraction
    Omega_b: float = 0.0493  # baryon density fraction

    # J₃(O) framework parameters
    dim_J3O: int = 27  # dimension of J₃(O)
    dim_F4: int = 52  # dimension of F₄
    dim_E6: int = 78  # dimension of E₆
    dim_E7: int = 133  # dimension of E₇
    dim_E8: int = 248  # dimension of E₈

    @property
    def alpha_inv(self) -> float:
        """Inverse fine structure constant."""
        return 1.0 / self.alpha

    @property
    def hierarchy_ratio(self) -> float:
        """M_P / M_EW hierarchy ratio."""
        return Planck.mass_GeV / 246.0  # v_EW = 246 GeV

    @property
    def log_hierarchy(self) -> float:
        """Natural log of hierarchy ratio."""
        return float(np.log(self.hierarchy_ratio))


# Global instance for convenient access
DIMENSIONLESS = DimensionlessConstants()


# =============================================================================
# ENERGY/MASS SCALES IN PLANCK UNITS
# =============================================================================


@dataclass(frozen=True)
class MassScalesPlanck:
    """
    Standard Model mass scales normalized to Planck mass.

    All values are m/M_P (dimensionless in Planck units).
    """

    # Leptons
    electron: float = 0.51099895e-3 / Planck.mass_GeV  # m_e/M_P
    muon: float = 0.1056583755 / Planck.mass_GeV  # m_μ/M_P
    tau: float = 1.77686 / Planck.mass_GeV  # m_τ/M_P

    # Light quarks (MS-bar at 2 GeV)
    up: float = 2.16e-3 / Planck.mass_GeV  # m_u/M_P
    down: float = 4.67e-3 / Planck.mass_GeV  # m_d/M_P
    strange: float = 93.4e-3 / Planck.mass_GeV  # m_s/M_P

    # Heavy quarks
    charm: float = 1.27 / Planck.mass_GeV  # m_c/M_P
    bottom: float = 4.18 / Planck.mass_GeV  # m_b/M_P
    top: float = 172.76 / Planck.mass_GeV  # m_t/M_P

    # Gauge bosons
    W: float = 80.377 / Planck.mass_GeV  # M_W/M_P
    Z: float = 91.1876 / Planck.mass_GeV  # M_Z/M_P
    higgs: float = 125.25 / Planck.mass_GeV  # M_H/M_P

    # Electroweak scale
    v_EW: float = 246.0 / Planck.mass_GeV  # v/M_P (Higgs vev)

    # QCD scale
    Lambda_QCD: float = 0.217 / Planck.mass_GeV  # Λ_QCD/M_P

    # Proton/neutron
    proton: float = 0.93827 / Planck.mass_GeV  # m_p/M_P
    neutron: float = 0.93957 / Planck.mass_GeV  # m_n/M_P


MASS_SCALES = MassScalesPlanck()


# =============================================================================
# LENGTH SCALES IN PLANCK UNITS
# =============================================================================


@dataclass(frozen=True)
class LengthScalesPlanck:
    """
    Standard length scales normalized to Planck length.

    All values are L/ℓ_P (dimensionless in Planck units).
    In Planck units, length has same dimension as 1/mass.
    """

    # Quantum scales
    compton_electron: float = Planck.mass_GeV / (0.51099895e-3)  # ℏ/(m_e c) in ℓ_P
    compton_proton: float = Planck.mass_GeV / 0.93827  # ℏ/(m_p c) in ℓ_P

    # Nuclear scales
    bohr_radius: float = 5.29177210903e-11 / Planck.length  # a_0/ℓ_P
    classical_electron_radius: float = 2.8179403262e-15 / Planck.length  # r_e/ℓ_P

    # Cosmological scales (approximate)
    hubble_radius: float = 4.4e26 / Planck.length  # c/H_0 in ℓ_P

    @property
    def hierarchy_length(self) -> float:
        """Electroweak length scale ℏ/(v_EW c) in Planck units."""
        return Planck.mass_GeV / 246.0  # Same as inverse mass ratio


LENGTH_SCALES = LengthScalesPlanck()


# =============================================================================
# UNIT CONVERSION FUNCTIONS
# =============================================================================


class UnitSystem(Enum):
    """Available unit systems."""

    PLANCK = "planck"
    SI = "si"
    NATURAL = "natural"  # c = ℏ = k_B = 1, but G ≠ 1, energies in GeV
    CGS = "cgs"


def to_planck_mass(value: float, unit: str) -> float:
    """
    Convert mass to Planck units (m/M_P).

    Parameters
    ----------
    value : float
        Mass value in specified units
    unit : str
        One of: 'kg', 'g', 'GeV', 'MeV', 'eV', 'solar'

    Returns
    -------
    float
        Mass in Planck units (dimensionless ratio m/M_P)
    """
    conversions = {
        "kg": 1.0 / Planck.mass,
        "g": 1e-3 / Planck.mass,
        "GeV": 1.0 / Planck.mass_GeV,
        "MeV": 1e-3 / Planck.mass_GeV,
        "eV": 1e-9 / Planck.mass_GeV,
        "solar": 1.989e30 / Planck.mass,
    }
    if unit not in conversions:
        raise ValueError(f"Unknown mass unit: {unit}. Use one of {list(conversions.keys())}")
    return value * conversions[unit]


def from_planck_mass(value: float, unit: str) -> float:
    """
    Convert mass from Planck units to specified units.

    Parameters
    ----------
    value : float
        Mass in Planck units (m/M_P)
    unit : str
        Target unit: 'kg', 'g', 'GeV', 'MeV', 'eV', 'solar'

    Returns
    -------
    float
        Mass in specified units
    """
    conversions = {
        "kg": Planck.mass,
        "g": Planck.mass * 1e3,
        "GeV": Planck.mass_GeV,
        "MeV": Planck.mass_GeV * 1e3,
        "eV": Planck.mass_GeV * 1e9,
        "solar": Planck.mass / 1.989e30,
    }
    if unit not in conversions:
        raise ValueError(f"Unknown mass unit: {unit}. Use one of {list(conversions.keys())}")
    return value * conversions[unit]


def to_planck_length(value: float, unit: str) -> float:
    """
    Convert length to Planck units (L/ℓ_P).

    Parameters
    ----------
    value : float
        Length value in specified units
    unit : str
        One of: 'm', 'cm', 'fm', 'Mpc', 'kpc', 'pc', 'AU', 'ly'

    Returns
    -------
    float
        Length in Planck units (dimensionless ratio L/ℓ_P)
    """
    conversions = {
        "m": 1.0 / Planck.length,
        "cm": 1e-2 / Planck.length,
        "fm": 1e-15 / Planck.length,
        "Mpc": 3.0857e22 / Planck.length,
        "kpc": 3.0857e19 / Planck.length,
        "pc": 3.0857e16 / Planck.length,
        "AU": 1.496e11 / Planck.length,
        "ly": 9.461e15 / Planck.length,
    }
    if unit not in conversions:
        raise ValueError(f"Unknown length unit: {unit}. Use one of {list(conversions.keys())}")
    return value * conversions[unit]


def from_planck_length(value: float, unit: str) -> float:
    """
    Convert length from Planck units to specified units.

    Parameters
    ----------
    value : float
        Length in Planck units (L/ℓ_P)
    unit : str
        Target unit: 'm', 'cm', 'fm', 'Mpc', 'kpc', 'pc', 'AU', 'ly'

    Returns
    -------
    float
        Length in specified units
    """
    conversions = {
        "m": Planck.length,
        "cm": Planck.length * 1e2,
        "fm": Planck.length * 1e15,
        "Mpc": Planck.length / 3.0857e22,
        "kpc": Planck.length / 3.0857e19,
        "pc": Planck.length / 3.0857e16,
        "AU": Planck.length / 1.496e11,
        "ly": Planck.length / 9.461e15,
    }
    if unit not in conversions:
        raise ValueError(f"Unknown length unit: {unit}. Use one of {list(conversions.keys())}")
    return value * conversions[unit]


def to_planck_time(value: float, unit: str) -> float:
    """
    Convert time to Planck units (t/t_P).

    Parameters
    ----------
    value : float
        Time value in specified units
    unit : str
        One of: 's', 'ms', 'us', 'ns', 'yr', 'Gyr'

    Returns
    -------
    float
        Time in Planck units (dimensionless ratio t/t_P)
    """
    conversions = {
        "s": 1.0 / Planck.time,
        "ms": 1e-3 / Planck.time,
        "us": 1e-6 / Planck.time,
        "ns": 1e-9 / Planck.time,
        "yr": 3.1557e7 / Planck.time,
        "Gyr": 3.1557e16 / Planck.time,
    }
    if unit not in conversions:
        raise ValueError(f"Unknown time unit: {unit}. Use one of {list(conversions.keys())}")
    return value * conversions[unit]


def from_planck_time(value: float, unit: str) -> float:
    """
    Convert time from Planck units to specified units.

    Parameters
    ----------
    value : float
        Time in Planck units (t/t_P)
    unit : str
        Target unit: 's', 'ms', 'us', 'ns', 'yr', 'Gyr'

    Returns
    -------
    float
        Time in specified units
    """
    conversions = {
        "s": Planck.time,
        "ms": Planck.time * 1e3,
        "us": Planck.time * 1e6,
        "ns": Planck.time * 1e9,
        "yr": Planck.time / 3.1557e7,
        "Gyr": Planck.time / 3.1557e16,
    }
    if unit not in conversions:
        raise ValueError(f"Unknown time unit: {unit}. Use one of {list(conversions.keys())}")
    return value * conversions[unit]


def to_planck_energy(value: float, unit: str) -> float:
    """
    Convert energy to Planck units (E/E_P).

    Parameters
    ----------
    value : float
        Energy value in specified units
    unit : str
        One of: 'J', 'GeV', 'MeV', 'keV', 'eV', 'erg'

    Returns
    -------
    float
        Energy in Planck units (dimensionless ratio E/E_P)
    """
    # E_P in various units
    E_P_J = Planck.energy
    E_P_GeV = Planck.energy_GeV

    conversions = {
        "J": 1.0 / E_P_J,
        "GeV": 1.0 / E_P_GeV,
        "MeV": 1e-3 / E_P_GeV,
        "keV": 1e-6 / E_P_GeV,
        "eV": 1e-9 / E_P_GeV,
        "erg": 1e-7 / E_P_J,
    }
    if unit not in conversions:
        raise ValueError(f"Unknown energy unit: {unit}. Use one of {list(conversions.keys())}")
    return value * conversions[unit]


def from_planck_energy(value: float, unit: str) -> float:
    """
    Convert energy from Planck units to specified units.

    Parameters
    ----------
    value : float
        Energy in Planck units (E/E_P)
    unit : str
        Target unit: 'J', 'GeV', 'MeV', 'keV', 'eV', 'erg'

    Returns
    -------
    float
        Energy in specified units
    """
    E_P_J = Planck.energy
    E_P_GeV = Planck.energy_GeV

    conversions = {
        "J": E_P_J,
        "GeV": E_P_GeV,
        "MeV": E_P_GeV * 1e3,
        "keV": E_P_GeV * 1e6,
        "eV": E_P_GeV * 1e9,
        "erg": E_P_J * 1e7,
    }
    if unit not in conversions:
        raise ValueError(f"Unknown energy unit: {unit}. Use one of {list(conversions.keys())}")
    return value * conversions[unit]


# =============================================================================
# NATURAL UNITS (GeV-based) <-> PLANCK CONVERSIONS
# =============================================================================


def natural_to_planck(value: float, dimension: str) -> float:
    """
    Convert from natural units (GeV-based) to Planck units.

    In natural units (c=ℏ=1): [E] = [M] = [T]⁻¹ = [L]⁻¹ = GeV
    In Planck units (c=ℏ=G=1): Everything dimensionless

    Parameters
    ----------
    value : float
        Value in natural units
    dimension : str
        Physical dimension: 'mass', 'energy', 'length', 'time', 'momentum'

    Returns
    -------
    float
        Value in Planck units
    """
    M_P = Planck.mass_GeV  # GeV

    if dimension in ("mass", "energy", "momentum"):
        return value / M_P
    elif dimension in ("length", "time"):
        # In natural units, length ~ 1/GeV, so L_planck ~ M_P GeV⁻¹
        return value * M_P
    else:
        raise ValueError(f"Unknown dimension: {dimension}")


def planck_to_natural(value: float, dimension: str) -> float:
    """
    Convert from Planck units to natural units (GeV-based).

    Parameters
    ----------
    value : float
        Value in Planck units (dimensionless)
    dimension : str
        Physical dimension: 'mass', 'energy', 'length', 'time', 'momentum'

    Returns
    -------
    float
        Value in natural units (GeV-based)
    """
    M_P = Planck.mass_GeV  # GeV

    if dimension in ("mass", "energy", "momentum"):
        return value * M_P
    elif dimension in ("length", "time"):
        return value / M_P
    else:
        raise ValueError(f"Unknown dimension: {dimension}")


# =============================================================================
# HUBBLE PARAMETER IN PLANCK UNITS
# =============================================================================


def H0_to_planck(H0_km_s_Mpc: float) -> float:
    """
    Convert Hubble constant from km/s/Mpc to Planck units (H₀ t_P).

    Parameters
    ----------
    H0_km_s_Mpc : float
        Hubble constant in km/s/Mpc

    Returns
    -------
    float
        Hubble constant in Planck units (dimensionless)
    """
    # Convert to 1/s first
    H0_per_s = H0_km_s_Mpc * 1000 / (3.0857e22)  # km/s/Mpc -> 1/s
    # Then to Planck units
    return H0_per_s * Planck.time


def H0_from_planck(H0_planck: float) -> float:
    """
    Convert Hubble constant from Planck units to km/s/Mpc.

    Parameters
    ----------
    H0_planck : float
        Hubble constant in Planck units

    Returns
    -------
    float
        Hubble constant in km/s/Mpc
    """
    H0_per_s = H0_planck / Planck.time
    return H0_per_s * 3.0857e22 / 1000


# =============================================================================
# COSMOLOGICAL PARAMETERS IN PLANCK UNITS
# =============================================================================


@dataclass
class CosmologyPlanck:
    """
    Cosmological parameters expressed in Planck units.

    All quantities are dimensionless ratios in Planck units.
    """

    # Hubble parameter (H₀ t_P)
    H0: float = H0_to_planck(67.4)  # Planck 2018 value

    # Critical density in Planck units (ρ_c / ρ_P where ρ_P = c⁵/(ℏG²))
    # ρ_c = 3H₀²/(8πG) = 3H₀²/(8π) in Planck units
    @property
    def rho_critical(self) -> float:
        """Critical density in Planck units."""
        return 3 * self.H0**2 / (8 * np.pi)

    # Cosmological constant (in Planck units)
    # Λ ~ 3H₀² Ω_Λ in natural units
    @property
    def Lambda(self) -> float:
        """Cosmological constant in Planck units."""
        Omega_Lambda = 0.685
        return 3 * self.H0**2 * Omega_Lambda

    # Hubble length in Planck units
    @property
    def L_H(self) -> float:
        """Hubble length c/H₀ in Planck units (= 1/H₀ since c=1)."""
        return 1.0 / self.H0

    # Hubble time in Planck units
    @property
    def t_H(self) -> float:
        """Hubble time 1/H₀ in Planck units."""
        return 1.0 / self.H0


COSMOLOGY_PLANCK = CosmologyPlanck()


# =============================================================================
# VALIDATION AND TESTING
# =============================================================================


def validate_planck_units():
    """
    Validate Planck unit definitions by checking consistency relations.

    Returns
    -------
    dict
        Dictionary with validation results
    """
    results = {}

    # Check ℓ_P = √(ℏG/c³)
    l_P_computed = np.sqrt(SI.hbar * SI.G / SI.c**3)
    results["length_check"] = {
        "computed": l_P_computed,
        "stored": Planck.length,
        "relative_error": abs(l_P_computed - Planck.length) / Planck.length,
    }

    # Check t_P = √(ℏG/c⁵) = ℓ_P/c
    t_P_computed = np.sqrt(SI.hbar * SI.G / SI.c**5)
    results["time_check"] = {
        "computed": t_P_computed,
        "stored": Planck.time,
        "relative_error": abs(t_P_computed - Planck.time) / Planck.time,
    }

    # Check M_P = √(ℏc/G)
    M_P_computed = np.sqrt(SI.hbar * SI.c / SI.G)
    results["mass_check"] = {
        "computed": M_P_computed,
        "stored": Planck.mass,
        "relative_error": abs(M_P_computed - Planck.mass) / Planck.mass,
    }

    # Check consistency: ℓ_P × M_P × c/ℏ = 1
    consistency = Planck.length * Planck.mass * SI.c / SI.hbar
    results["lm_consistency"] = {
        "computed": consistency,
        "expected": 1.0,
        "relative_error": abs(consistency - 1.0),
    }

    # Check G = ℓ_P² c³ / ℏ (definition)
    G_from_planck = Planck.length**2 * SI.c**3 / SI.hbar
    results["G_check"] = {
        "computed": G_from_planck,
        "stored": SI.G,
        "relative_error": abs(G_from_planck - SI.G) / SI.G,
    }

    return results


def print_planck_scales():
    """Print summary of Planck scales and comparisons."""
    print("=" * 70)
    print("PLANCK UNIT SYSTEM")
    print("=" * 70)

    print("\n  Base Planck Units:")
    print(f"    ℓ_P = {Planck.length:.6e} m")
    print(f"    t_P = {Planck.time:.6e} s")
    print(f"    M_P = {Planck.mass:.6e} kg = {Planck.mass_GeV:.6e} GeV")
    print(f"    T_P = {Planck.temperature:.6e} K")
    print(f"    q_P = {Planck.charge:.6e} C")

    print("\n  Derived Planck Units:")
    print(f"    E_P = {Planck.energy:.6e} J")
    print(f"    F_P = {Planck.force:.6e} N")
    print(f"    ρ_P = {Planck.density:.6e} kg/m³")

    print("\n  Dimensionless Constants:")
    print(f"    α = {DIMENSIONLESS.alpha:.10f} = 1/{1/DIMENSIONLESS.alpha:.3f}")
    print(f"    α_s(M_Z) = {DIMENSIONLESS.alpha_s_MZ}")
    print(f"    sin²θ_W = {DIMENSIONLESS.sin2_theta_W}")
    print(f"    M_P/v_EW = {DIMENSIONLESS.hierarchy_ratio:.3e}")

    print("\n  Mass Scales (in Planck units):")
    print(f"    m_e/M_P = {MASS_SCALES.electron:.6e}")
    print(f"    m_p/M_P = {MASS_SCALES.proton:.6e}")
    print(f"    M_W/M_P = {MASS_SCALES.W:.6e}")
    print(f"    M_Z/M_P = {MASS_SCALES.Z:.6e}")
    print(f"    v_EW/M_P = {MASS_SCALES.v_EW:.6e}")

    print("\n  Cosmological Scales (in Planck units):")
    print(f"    H₀ t_P = {COSMOLOGY_PLANCK.H0:.6e}")
    print(f"    L_H/ℓ_P = {COSMOLOGY_PLANCK.L_H:.6e}")


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Classes
    "SI",
    "Planck",
    "DimensionlessConstants",
    "MassScalesPlanck",
    "LengthScalesPlanck",
    "CosmologyPlanck",
    "UnitSystem",
    # Instances
    "DIMENSIONLESS",
    "MASS_SCALES",
    "LENGTH_SCALES",
    "COSMOLOGY_PLANCK",
    # Conversion functions
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
    # Utilities
    "validate_planck_units",
    "print_planck_scales",
]


if __name__ == "__main__":
    print_planck_scales()

    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    results = validate_planck_units()
    all_pass = True
    for check, data in results.items():
        status = "PASS" if data["relative_error"] < 1e-6 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  {check}: {status} (rel. error: {data['relative_error']:.2e})")

    if all_pass:
        print("\n  All validation checks passed.")
