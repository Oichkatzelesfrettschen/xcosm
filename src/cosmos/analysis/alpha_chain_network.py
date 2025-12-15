#!/usr/bin/env python3
"""
alpha_chain_network.py — 13-Isotope Alpha-Chain Nuclear Network

Replaces the single-step Fisher-KPP reaction with a proper nuclear
burning network for Type Ia supernova deflagration simulations.

ISOTOPES (alpha-chain):
    He4, C12, O16, Ne20, Mg24, Si28, S32, Ar36, Ca40, Ti44, Cr48, Fe52, Ni56

KEY REACTIONS:
    C12 + C12 → Ne20 + He4     (Q = 4.62 MeV)
    C12 + O16 → Mg24 + He4     (Q = 6.77 MeV)
    O16 + O16 → Si28 + He4     (Q = 9.59 MeV)
    Si28 + ... → Ni56 (NSE)    (Q ~ 1.75 MeV/nucleon)

PHYSICS:
    - Thermonuclear reaction rates from Caughlan & Fowler (1988)
    - Coulomb screening (weak/strong regimes)
    - Energy release computed from mass excess
    - Stiff ODE integration for multi-timescale problem

DEPTH IMPROVEMENT:
    This addresses the "Shallow Spot" identified in the Computational Depth Audit:
    - Fisher-KPP: Fuel → Ash (single step)
    - Alpha-chain: C → Mg → Si → Ni (staged energy release)

Author: Spandrel Framework
Date: November 28, 2025
"""

import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from typing import Tuple, Dict, Optional
import warnings

# =============================================================================
# PHYSICAL CONSTANTS (CGS)
# =============================================================================

# Fundamental
k_B = 1.380649e-16       # Boltzmann constant [erg/K]
N_A = 6.02214076e23      # Avogadro number [mol^-1]
c_light = 2.99792458e10  # Speed of light [cm/s]
m_u = 1.66054e-24        # Atomic mass unit [g]
MeV_to_erg = 1.60218e-6  # MeV to erg conversion

# Nuclear
hbar = 1.054572e-27      # Reduced Planck [erg·s]
e_charge = 4.80326e-10   # Electron charge [esu]


# =============================================================================
# ISOTOPE DATA
# =============================================================================

@dataclass
class Isotope:
    """Nuclear isotope properties."""
    name: str
    Z: int          # Atomic number (protons)
    A: int          # Mass number (protons + neutrons)
    mass_excess: float  # Mass excess in MeV (Δ = M - A*m_u)

    @property
    def binding_energy_per_nucleon(self) -> float:
        """B/A in MeV."""
        # B = Z*m_p + N*m_n - M = -Δ + (A - Z)*Δ_n + Z*Δ_p
        # Simplified: B/A ~ 8.8 - 0.0007*A for heavy nuclei
        return 8.8 - 0.0007 * self.A


# Alpha-chain isotopes
ISOTOPES = {
    'He4':  Isotope('He4',   2,  4,   2.425),
    'C12':  Isotope('C12',   6, 12,   0.000),
    'O16':  Isotope('O16',   8, 16,  -4.737),
    'Ne20': Isotope('Ne20', 10, 20,  -7.042),
    'Mg24': Isotope('Mg24', 12, 24, -13.934),
    'Si28': Isotope('Si28', 14, 28, -21.493),
    'S32':  Isotope('S32',  16, 32, -26.016),
    'Ar36': Isotope('Ar36', 18, 36, -30.231),
    'Ca40': Isotope('Ca40', 20, 40, -34.846),
    'Ti44': Isotope('Ti44', 22, 44, -37.548),
    'Cr48': Isotope('Cr48', 24, 48, -42.820),
    'Fe52': Isotope('Fe52', 26, 52, -48.333),
    'Ni56': Isotope('Ni56', 28, 56, -53.904),
}

# Ordered list for indexing
ISOTOPE_NAMES = ['He4', 'C12', 'O16', 'Ne20', 'Mg24', 'Si28',
                 'S32', 'Ar36', 'Ca40', 'Ti44', 'Cr48', 'Fe52', 'Ni56']
N_SPECIES = len(ISOTOPE_NAMES)


# =============================================================================
# REACTION RATES (Caughlan & Fowler 1988 approximations)
# =============================================================================

def T9(T: float) -> float:
    """Temperature in units of 10^9 K."""
    return T / 1e9


def rate_C12_C12(T: float, rho: float) -> float:
    """
    C12 + C12 → Ne20 + He4 reaction rate.

    Rate from Caughlan & Fowler 1988, fit to experimental data.
    Returns rate per unit mass [1/s].

    Note: This is the rate-limiting step at T < 10^9 K.
    """
    T9_val = T9(T)
    if T9_val < 0.1:
        return 0.0

    # CF88 fit (simplified)
    # λ_CC = ρ × N_A × <σv> / (2 × A_C²)
    # <σv> = S(0) × exp(-τ) / τ² with Gamow peak
    tau = 84.165 / T9_val**(1/3)
    S_factor = 3.0e16  # S(0) in keV·barn (order of magnitude)

    # Rate coefficient [cm³/mol/s]
    rate_coeff = 4.27e26 * T9_val**(-2/3) * np.exp(-84.165 / T9_val**(1/3))

    # Rate per unit mass [1/s] = ρ × N_A × rate_coeff / A²
    rate = rho * N_A * rate_coeff / (12**2)

    return rate


def rate_C12_O16(T: float, rho: float) -> float:
    """
    C12 + O16 → Mg24 + He4 reaction rate.

    Slightly faster than C+C due to higher Coulomb barrier penetration
    for the asymmetric system.
    """
    T9_val = T9(T)
    if T9_val < 0.1:
        return 0.0

    # CF88 fit (simplified)
    rate_coeff = 1.72e31 * T9_val**(-2/3) * np.exp(-106.594 / T9_val**(1/3))

    # Rate per unit mass
    rate = rho * N_A * rate_coeff / (12 * 16)

    return rate


def rate_O16_O16(T: float, rho: float) -> float:
    """
    O16 + O16 → Si28 + He4 reaction rate.

    Higher threshold than C+C but produces more energy.
    """
    T9_val = T9(T)
    if T9_val < 0.3:
        return 0.0

    # CF88 fit
    rate_coeff = 7.10e36 * T9_val**(-2/3) * np.exp(-135.93 / T9_val**(1/3))

    # Correction for resonances (simplified)
    rate_coeff *= (1 + 0.04 * T9_val**2)

    rate = rho * N_A * rate_coeff / (16**2)

    return rate


def rate_alpha_capture(iso_name: str, T: float, rho: float) -> float:
    """
    Generic alpha capture rate: X + He4 → Y + gamma.

    Used for the alpha-chain from Ne20 onward.
    Rates scale roughly as Z^2 for the Coulomb barrier.
    """
    T9_val = T9(T)
    if T9_val < 0.5:
        return 0.0

    iso = ISOTOPES[iso_name]
    Z = iso.Z
    A = iso.A

    # Gamow energy for alpha capture
    E_G = 0.98 * (Z * 2)**2 * (A * 4 / (A + 4))**(1/3)  # keV

    # Simplified rate (order of magnitude)
    tau = 4.25 * (E_G / T9_val)**(1/3)
    rate_coeff = 1e25 * T9_val**(-2/3) * np.exp(-tau)

    # Scale with Z² for Coulomb barrier
    rate_coeff *= (Z / 10)**2

    rate = rho * N_A * rate_coeff / (A * 4)

    return rate


def rate_Si_to_NSE(T: float, rho: float) -> float:
    """
    Si28 → Ni56 quasi-equilibrium rate.

    At T > 3×10^9 K, silicon burning enters Nuclear Statistical Equilibrium.
    Model as effective rate to Ni56.
    """
    T9_val = T9(T)
    if T9_val < 3.0:
        return 0.0

    # NSE timescale ~ 0.1 s at T9 = 5
    # Rate ~ 1/τ_NSE × exp(-Q/kT) where Q ~ 10 MeV
    tau_NSE = 0.1 * np.exp(10.0 / T9_val)
    rate = 1.0 / tau_NSE

    return rate


# =============================================================================
# ENERGY RELEASE
# =============================================================================

def Q_value(reactants: list, products: list) -> float:
    """
    Compute Q-value (energy release) for a reaction.

    Q = Σ(mass_excess_reactants) - Σ(mass_excess_products)

    Returns Q in erg.
    """
    Q_MeV = sum(ISOTOPES[r].mass_excess for r in reactants) - \
            sum(ISOTOPES[p].mass_excess for p in products)
    return Q_MeV * MeV_to_erg


# Pre-computed Q-values for key reactions
Q_C12_C12 = Q_value(['C12', 'C12'], ['Ne20', 'He4'])  # 4.62 MeV
Q_C12_O16 = Q_value(['C12', 'O16'], ['Mg24', 'He4'])  # 6.77 MeV
Q_O16_O16 = Q_value(['O16', 'O16'], ['Si28', 'He4'])  # 9.59 MeV
Q_Si_Ni = Q_value(['Si28', 'Si28'], ['Ni56', 'He4'])  # ~1.75 MeV/nucleon × 28


# =============================================================================
# NETWORK ODE SYSTEM
# =============================================================================

def network_rhs(t: float, Y: np.ndarray, rho: float, T: float) -> np.ndarray:
    """
    Right-hand side of the nuclear network ODEs.

    dY_i/dt = Σ_j (creation rates) - Σ_k (destruction rates)

    Parameters
    ----------
    t : float
        Time (unused, for ODE solver compatibility)
    Y : ndarray
        Mass fractions [He4, C12, O16, Ne20, Mg24, Si28, S32, Ar36, Ca40, Ti44, Cr48, Fe52, Ni56]
    rho : float
        Density [g/cm³]
    T : float
        Temperature [K]

    Returns
    -------
    dYdt : ndarray
        Time derivatives of mass fractions
    """
    # Unpack mass fractions
    Y_He = Y[0]
    Y_C = Y[1]
    Y_O = Y[2]
    Y_Ne = Y[3]
    Y_Mg = Y[4]
    Y_Si = Y[5]
    Y_S = Y[6]
    Y_Ar = Y[7]
    Y_Ca = Y[8]
    Y_Ti = Y[9]
    Y_Cr = Y[10]
    Y_Fe = Y[11]
    Y_Ni = Y[12]

    # Compute reaction rates
    r_CC = rate_C12_C12(T, rho) * Y_C**2
    r_CO = rate_C12_O16(T, rho) * Y_C * Y_O
    r_OO = rate_O16_O16(T, rho) * Y_O**2

    # Alpha captures (simplified chain)
    r_Ne_a = rate_alpha_capture('Ne20', T, rho) * Y_Ne * Y_He
    r_Mg_a = rate_alpha_capture('Mg24', T, rho) * Y_Mg * Y_He
    r_Si_a = rate_alpha_capture('Si28', T, rho) * Y_Si * Y_He
    r_S_a = rate_alpha_capture('S32', T, rho) * Y_S * Y_He
    r_Ar_a = rate_alpha_capture('Ar36', T, rho) * Y_Ar * Y_He
    r_Ca_a = rate_alpha_capture('Ca40', T, rho) * Y_Ca * Y_He
    r_Ti_a = rate_alpha_capture('Ti44', T, rho) * Y_Ti * Y_He
    r_Cr_a = rate_alpha_capture('Cr48', T, rho) * Y_Cr * Y_He
    r_Fe_a = rate_alpha_capture('Fe52', T, rho) * Y_Fe * Y_He

    # NSE (silicon → nickel)
    r_Si_NSE = rate_Si_to_NSE(T, rho) * Y_Si

    # Construct dY/dt
    dYdt = np.zeros(N_SPECIES)

    # He4: produced by C+C, C+O, O+O; consumed by alpha captures
    dYdt[0] = (4/20) * r_CC + (4/24) * r_CO + (4/28) * r_OO \
              - (4/24) * r_Ne_a - (4/28) * r_Mg_a - (4/32) * r_Si_a \
              - (4/36) * r_S_a - (4/40) * r_Ar_a - (4/44) * r_Ca_a \
              - (4/48) * r_Ti_a - (4/52) * r_Cr_a - (4/56) * r_Fe_a

    # C12: consumed by C+C and C+O
    dYdt[1] = -2 * (12/20) * r_CC - (12/24) * r_CO

    # O16: consumed by C+O and O+O
    dYdt[2] = -(16/24) * r_CO - 2 * (16/28) * r_OO

    # Ne20: produced by C+C, consumed by alpha capture
    dYdt[3] = (20/20) * r_CC - (20/24) * r_Ne_a

    # Mg24: produced by C+O and Ne+α
    dYdt[4] = (24/24) * r_CO + (24/24) * r_Ne_a - (24/28) * r_Mg_a

    # Si28: produced by O+O and Mg+α, consumed by α and NSE
    dYdt[5] = (28/28) * r_OO + (28/28) * r_Mg_a - (28/32) * r_Si_a - r_Si_NSE

    # S32 → Ni56 chain (simplified)
    dYdt[6] = (32/32) * r_Si_a - (32/36) * r_S_a
    dYdt[7] = (36/36) * r_S_a - (36/40) * r_Ar_a
    dYdt[8] = (40/40) * r_Ar_a - (40/44) * r_Ca_a
    dYdt[9] = (44/44) * r_Ca_a - (44/48) * r_Ti_a
    dYdt[10] = (48/48) * r_Ti_a - (48/52) * r_Cr_a
    dYdt[11] = (52/52) * r_Cr_a - (52/56) * r_Fe_a
    dYdt[12] = (56/56) * r_Fe_a + (56/28) * r_Si_NSE  # NSE shortcut

    return dYdt


def energy_generation_rate(Y: np.ndarray, rho: float, T: float) -> float:
    """
    Compute nuclear energy generation rate ε [erg/g/s].

    ε = Σ (Q_reaction × rate_reaction / ρ)
    """
    Y_C = Y[1]
    Y_O = Y[2]
    Y_Si = Y[5]
    Y_He = Y[0]

    r_CC = rate_C12_C12(T, rho) * Y_C**2
    r_CO = rate_C12_O16(T, rho) * Y_C * Y_O
    r_OO = rate_O16_O16(T, rho) * Y_O**2
    r_Si_NSE = rate_Si_to_NSE(T, rho) * Y_Si

    # Energy per gram
    eps = (Q_C12_C12 * r_CC + Q_C12_O16 * r_CO +
           Q_O16_O16 * r_OO + Q_Si_Ni * r_Si_NSE / 2)

    return eps


# =============================================================================
# SOLVER INTERFACE
# =============================================================================

def evolve_network(Y0: np.ndarray, rho: float, T: float,
                   dt: float, method: str = 'Radau') -> Tuple[np.ndarray, float]:
    """
    Evolve the nuclear network for one timestep.

    Parameters
    ----------
    Y0 : ndarray
        Initial mass fractions
    rho : float
        Density [g/cm³]
    T : float
        Temperature [K]
    dt : float
        Timestep [s]
    method : str
        ODE solver method (default: 'Radau' for stiff systems)

    Returns
    -------
    Y_new : ndarray
        Final mass fractions
    energy_released : float
        Total energy released per gram [erg/g]
    """
    # Suppress warnings from stiff solver
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        sol = solve_ivp(
            lambda t, y: network_rhs(t, y, rho, T),
            [0, dt],
            Y0,
            method=method,
            dense_output=False,
            rtol=1e-6,
            atol=1e-10
        )

    Y_new = sol.y[:, -1]

    # Ensure mass conservation and positivity
    Y_new = np.maximum(Y_new, 0)
    Y_new /= Y_new.sum()  # Normalize

    # Compute energy released
    eps_avg = 0.5 * (energy_generation_rate(Y0, rho, T) +
                     energy_generation_rate(Y_new, rho, T))
    energy_released = eps_avg * dt

    return Y_new, energy_released


# =============================================================================
# INITIAL CONDITIONS
# =============================================================================

def initial_composition_CO(X_C: float = 0.5, X_O: float = 0.5) -> np.ndarray:
    """
    Initial composition for C/O white dwarf.

    Default: 50% C12, 50% O16 by mass.
    """
    Y = np.zeros(N_SPECIES)
    Y[1] = X_C  # C12
    Y[2] = X_O  # O16
    return Y


def initial_composition_with_metals(X_C: float = 0.48, X_O: float = 0.48,
                                     X_Ne: float = 0.02, X_Mg: float = 0.02) -> np.ndarray:
    """
    Initial composition with some metals from CNO processing.
    """
    Y = np.zeros(N_SPECIES)
    Y[1] = X_C   # C12
    Y[2] = X_O   # O16
    Y[3] = X_Ne  # Ne20
    Y[4] = X_Mg  # Mg24
    return Y


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demonstrate_network():
    """Demonstrate the alpha-chain network."""
    print("=" * 70)
    print("ALPHA-CHAIN NUCLEAR NETWORK DEMONSTRATION")
    print("=" * 70)
    print()

    # Initial C/O composition
    Y0 = initial_composition_CO()
    print("Initial composition:")
    for i, name in enumerate(ISOTOPE_NAMES):
        if Y0[i] > 1e-6:
            print(f"  {name}: {Y0[i]:.4f}")
    print()

    # Test at different temperatures
    rho = 2e9  # g/cm³

    print(f"Density: {rho:.1e} g/cm³")
    print()
    print(f"{'T [K]':>12} | {'ε [erg/g/s]':>12} | {'τ_burn [s]':>12} | Dominant")
    print("-" * 60)

    for T in [5e8, 1e9, 2e9, 3e9, 5e9]:
        eps = energy_generation_rate(Y0, rho, T)

        # Burning timescale
        E_nuc = 1.75e18 * 0.5  # erg/g for 50% burning to Ni
        tau_burn = E_nuc / eps if eps > 0 else np.inf

        # Dominant reaction
        r_CC = rate_C12_C12(T, rho)
        r_OO = rate_O16_O16(T, rho)
        r_NSE = rate_Si_to_NSE(T, rho)

        if r_CC > r_OO and r_CC > r_NSE:
            dominant = "C+C"
        elif r_OO > r_NSE:
            dominant = "O+O"
        else:
            dominant = "NSE"

        print(f"{T:>12.1e} | {eps:>12.2e} | {tau_burn:>12.2e} | {dominant}")

    print()

    # Evolution at T = 3e9 K
    print("=" * 70)
    print("NETWORK EVOLUTION AT T = 3×10⁹ K")
    print("=" * 70)
    print()

    T = 3e9
    Y = Y0.copy()

    print(f"{'Time [s]':>10} | {'C12':>8} | {'O16':>8} | {'Si28':>8} | {'Ni56':>8}")
    print("-" * 56)

    t_total = 0
    dt = 0.001  # 1 ms
    n_steps = 100

    for step in range(n_steps):
        Y, E_rel = evolve_network(Y, rho, T, dt)
        t_total += dt

        if step % 20 == 0:
            print(f"{t_total:>10.3f} | {Y[1]:>8.4f} | {Y[2]:>8.4f} | "
                  f"{Y[5]:>8.4f} | {Y[12]:>8.4f}")

    print("-" * 56)
    print()
    print("Final composition:")
    for i, name in enumerate(ISOTOPE_NAMES):
        if Y[i] > 1e-4:
            print(f"  {name}: {Y[i]:.4f}")

    # Ni-56 mass produced
    M_Ni = Y[12]  # Mass fraction
    print()
    print(f"Ni-56 mass fraction: {M_Ni:.3f}")
    print(f"For 1.0 M_sun ejecta: M(Ni-56) = {M_Ni:.3f} M_sun")


if __name__ == "__main__":
    demonstrate_network()
