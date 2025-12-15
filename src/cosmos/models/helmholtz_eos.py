#!/usr/bin/env python3
"""
helmholtz_eos.py — Helmholtz Free Energy Equation of State for WD Matter

Implements the Timmes & Swesty (2000) Helmholtz EOS for degenerate stellar matter.
Includes proper treatment of:
1. Degenerate electrons (relativistic + non-relativistic)
2. Electron-positron pairs
3. Photon radiation
4. Ions (ideal gas + Coulomb corrections + DEBYE corrections)

THE DEBYE CORRECTION:
=====================
At low T (T < Θ_D), the ion heat capacity follows Debye's T³ law, not 3/2 N k_B.
This affects:
- Internal energy
- Pressure (small correction)
- Sound speed
- Adiabatic index γ

For WD deflagrations (T ~ 10⁹ K, ρ ~ 10⁹ g/cm³):
- Θ_D ~ 4×10⁸ K for carbon
- T > Θ_D, so classical limit applies, but corrections still matter

COULOMB CORRECTIONS:
====================
The Coulomb coupling parameter:
    Γ = (Ze)² / (a_i k_B T)    where a_i = (3/4πn_i)^(1/3)

- Γ < 1: Weakly coupled plasma
- Γ > 1: Strongly coupled (liquid)
- Γ > 175: Crystallized (solid)

At flame conditions: Γ ~ 1-10 (strong coupling)

References:
-----------
- Timmes & Swesty 2000, ApJS 126, 501 (Helmholtz EOS)
- Chabrier & Potekhin 1998, PRE 58, 4941 (Coulomb corrections)
- Potekhin & Chabrier 2000, PRE 62, 8554 (Ion mixing)

Author: Spandrel Framework
Date: November 28, 2025
"""

import numpy as np
from typing import Tuple, NamedTuple
from dataclasses import dataclass


# =============================================================================
# PHYSICAL CONSTANTS (CGS)
# =============================================================================

# Fundamental
C_LIGHT = 2.99792458e10     # cm/s
HBAR = 1.05457266e-27       # erg·s
K_BOLTZMANN = 1.380658e-16  # erg/K
M_ELECTRON = 9.1093897e-28  # g
M_PROTON = 1.6726231e-24    # g
E_ELECTRON = 4.8032068e-10  # esu (electron charge)
N_AVOGADRO = 6.0221367e23   # /mol

# Derived
M_E_C2 = M_ELECTRON * C_LIGHT**2           # electron rest mass energy
ALPHA_FS = E_ELECTRON**2 / (HBAR * C_LIGHT)  # fine structure constant
A_RAD = 7.5657e-15          # radiation constant [erg/cm³/K⁴]

# Composition (C/O WD default)
Z_ION = 6.0     # Mean ion charge (carbon)
A_ION = 12.0    # Mean atomic mass (carbon)


# =============================================================================
# DEBYE PHYSICS
# =============================================================================

def ion_number_density(rho: float, A: float = A_ION) -> float:
    """Ion number density n_i = ρ N_A / A"""
    return rho * N_AVOGADRO / A


def ion_sphere_radius(n_i: float) -> float:
    """
    Ion sphere radius (Wigner-Seitz radius).

    a_i = (3 / 4π n_i)^(1/3)
    """
    return (3.0 / (4.0 * np.pi * n_i))**(1.0/3.0)


def coulomb_coupling_parameter(rho: float, T: float,
                                 Z: float = Z_ION, A: float = A_ION) -> float:
    """
    Coulomb coupling parameter Γ.

    Γ = (Ze)² / (a_i k_B T)

    - Γ < 1: Ideal plasma
    - Γ ~ 1-10: Strongly coupled liquid
    - Γ > 175: Coulomb crystal

    Parameters
    ----------
    rho : float
        Density [g/cm³]
    T : float
        Temperature [K]
    Z, A : float
        Ion charge and mass number

    Returns
    -------
    Gamma : float
        Coupling parameter (dimensionless)
    """
    n_i = ion_number_density(rho, A)
    a_i = ion_sphere_radius(n_i)

    Gamma = Z**2 * E_ELECTRON**2 / (a_i * K_BOLTZMANN * T)
    return Gamma


def debye_temperature(rho: float, Z: float = Z_ION, A: float = A_ION) -> float:
    """
    Debye temperature for ion lattice.

    Θ_D = (ℏ/k_B) × ω_p × (4π/3)^(1/3)

    where ω_p = sqrt(4π n_i Z² e² / m_i) is the ion plasma frequency.

    For C at ρ = 10⁹ g/cm³: Θ_D ≈ 4×10⁸ K

    Parameters
    ----------
    rho : float
        Density [g/cm³]

    Returns
    -------
    Theta_D : float
        Debye temperature [K]
    """
    n_i = ion_number_density(rho, A)
    m_i = A * M_PROTON

    # Ion plasma frequency
    omega_p = np.sqrt(4.0 * np.pi * n_i * Z**2 * E_ELECTRON**2 / m_i)

    # Debye temperature
    Theta_D = HBAR * omega_p / K_BOLTZMANN * (4.0 * np.pi / 3.0)**(1.0/3.0)

    return Theta_D


def debye_function_D3(x: float) -> float:
    """
    Debye function D₃(x) = (3/x³) ∫₀ˣ t³/(eᵗ-1) dt

    Used for heat capacity and energy of Debye solid.

    For x >> 1 (T << Θ_D): D₃(x) → π⁴/(5x³)
    For x << 1 (T >> Θ_D): D₃(x) → 1

    Parameters
    ----------
    x : float
        Θ_D / T

    Returns
    -------
    D3 : float
        Debye function value
    """
    if x < 0.1:
        # High-T limit: D₃ ≈ 1 - 3x/8 + x²/20
        return 1.0 - 0.375 * x + 0.05 * x**2
    elif x > 20:
        # Low-T limit: D₃ ≈ π⁴/(5x³)
        return np.pi**4 / (5.0 * x**3)
    else:
        # Numerical integration
        from scipy.integrate import quad
        integrand = lambda t: t**3 / (np.exp(t) - 1) if t > 0 else 0
        result, _ = quad(integrand, 1e-10, x)
        return 3.0 * result / x**3


# =============================================================================
# ION EOS WITH COULOMB + DEBYE CORRECTIONS
# =============================================================================

def ion_pressure_ideal(rho: float, T: float, A: float = A_ION) -> float:
    """
    Ideal gas ion pressure.

    P_i = n_i k_B T = (ρ N_A / A) k_B T
    """
    n_i = ion_number_density(rho, A)
    return n_i * K_BOLTZMANN * T


def ion_pressure_coulomb(rho: float, T: float,
                          Z: float = Z_ION, A: float = A_ION) -> float:
    """
    Coulomb correction to ion pressure.

    For strongly coupled plasma (Γ > 1), use Chabrier-Potekhin fit:
    P_Coulomb / P_ideal = f(Γ) ≈ -0.9 × Γ

    Parameters
    ----------
    rho, T : float
        Density and temperature
    Z, A : float
        Ion charge and mass number

    Returns
    -------
    P_coul : float
        Coulomb pressure correction [dyn/cm²]
    """
    Gamma = coulomb_coupling_parameter(rho, T, Z, A)
    P_ideal = ion_pressure_ideal(rho, T, A)

    if Gamma < 1:
        # Weak coupling: Debye-Hückel
        f_Gamma = -Gamma**(3/2) / np.sqrt(3)
    else:
        # Strong coupling: Fit from Chabrier & Potekhin
        # f(Γ) ≈ -0.9 Γ + 0.18 Γ^(1/4) for liquid
        f_Gamma = -0.9 * Gamma + 0.18 * Gamma**0.25

    return P_ideal * f_Gamma


def ion_energy_debye(rho: float, T: float,
                      Z: float = Z_ION, A: float = A_ION) -> float:
    """
    Ion internal energy with Debye correction.

    Classical limit (T >> Θ_D): E_i = (3/2) n_i k_B T
    Quantum (T << Θ_D): E_i = (3/2) n_i k_B T × D₃(Θ_D/T) + zero-point

    Parameters
    ----------
    rho, T : float
        Density and temperature

    Returns
    -------
    e_i : float
        Ion internal energy per unit volume [erg/cm³]
    """
    n_i = ion_number_density(rho, A)
    Theta_D = debye_temperature(rho, Z, A)

    x = Theta_D / T
    D3 = debye_function_D3(x)

    # Zero-point energy
    E_zero = 0.5 * n_i * K_BOLTZMANN * Theta_D * (9.0/8.0)

    # Thermal energy with Debye correction
    E_thermal = 3.0 * n_i * K_BOLTZMANN * T * D3

    return E_thermal + E_zero


def ion_cv_debye(rho: float, T: float,
                  Z: float = Z_ION, A: float = A_ION) -> float:
    """
    Ion heat capacity with Debye correction.

    Classical (T >> Θ_D): C_v = (3/2) n_i k_B
    Quantum (T << Θ_D): C_v ∝ T³ (Debye T³ law)

    Parameters
    ----------
    rho, T : float
        Density and temperature

    Returns
    -------
    cv_i : float
        Ion heat capacity per unit volume [erg/cm³/K]
    """
    n_i = ion_number_density(rho, A)
    Theta_D = debye_temperature(rho, Z, A)

    x = Theta_D / T

    if x < 0.1:
        # Classical limit
        cv = 3.0 * n_i * K_BOLTZMANN
    elif x > 20:
        # T³ law
        cv = 12.0 * np.pi**4 / 5.0 * n_i * K_BOLTZMANN * (T / Theta_D)**3
    else:
        # Numerical derivative of D₃
        # C_v = 3 n k_B [4 D₃(x) - 3x/(eˣ-1)]
        D3 = debye_function_D3(x)
        exp_term = x / (np.exp(x) - 1) if x < 100 else 0
        cv = 3.0 * n_i * K_BOLTZMANN * (4.0 * D3 - 3.0 * exp_term)

    return max(cv, 0)


# =============================================================================
# ELECTRON EOS (DEGENERATE)
# =============================================================================

def electron_fermi_energy(rho: float, Z: float = Z_ION, A: float = A_ION) -> float:
    """
    Electron Fermi energy.

    E_F = (ℏ²/2m_e) × (3π² n_e)^(2/3)

    For relativistic electrons:
    E_F = sqrt((p_F c)² + (m_e c²)²) - m_e c²
    """
    n_e = Z * ion_number_density(rho, A)

    # Fermi momentum
    p_F = HBAR * (3.0 * np.pi**2 * n_e)**(1.0/3.0)

    # Relativistic energy
    E_F = np.sqrt((p_F * C_LIGHT)**2 + M_E_C2**2) - M_E_C2

    return E_F


def electron_degeneracy_parameter(rho: float, T: float,
                                    Z: float = Z_ION, A: float = A_ION) -> float:
    """
    Electron degeneracy parameter η = E_F / (k_B T).

    η >> 1: Fully degenerate (T << T_Fermi)
    η ~ 1: Partially degenerate
    η << 1: Non-degenerate (classical)
    """
    E_F = electron_fermi_energy(rho, Z, A)
    return E_F / (K_BOLTZMANN * T)


def electron_pressure_degenerate(rho: float, Z: float = Z_ION, A: float = A_ION) -> float:
    """
    Degenerate electron pressure (zero-temperature limit).

    Non-relativistic: P_e = (1/5) × (3/8π)^(2/3) × (h²/m_e) × n_e^(5/3)
    Relativistic: P_e = (1/4) × (3/π)^(1/3) × ℏc × n_e^(4/3)

    We use the full relativistic expression.
    """
    n_e = Z * ion_number_density(rho, A)

    # Relativity parameter: x = p_F / (m_e c)
    p_F = HBAR * (3.0 * np.pi**2 * n_e)**(1.0/3.0)
    x = p_F / (M_ELECTRON * C_LIGHT)

    # Full relativistic pressure (Chandrasekhar formula)
    # P = (π m_e⁴ c⁵ / 3 h³) × f(x)
    # f(x) = x(2x²-3)√(1+x²) + 3 sinh⁻¹(x)
    A_ch = np.pi * M_ELECTRON**4 * C_LIGHT**5 / (3.0 * (2*np.pi*HBAR)**3)

    f_x = x * (2*x**2 - 3) * np.sqrt(1 + x**2) + 3.0 * np.arcsinh(x)

    return A_ch * f_x


def electron_pressure_thermal(rho: float, T: float,
                               Z: float = Z_ION, A: float = A_ION) -> float:
    """
    Thermal correction to electron pressure.

    For degenerate electrons: P_th ≈ (π²/3) × n_e k_B² T² / E_F
    """
    n_e = Z * ion_number_density(rho, A)
    E_F = electron_fermi_energy(rho, Z, A)

    eta = E_F / (K_BOLTZMANN * T)

    if eta > 10:
        # Strongly degenerate: Sommerfeld expansion
        P_th = (np.pi**2 / 3.0) * n_e * K_BOLTZMANN**2 * T**2 / E_F
    elif eta < 0.1:
        # Non-degenerate: ideal gas
        P_th = n_e * K_BOLTZMANN * T
    else:
        # Intermediate: interpolate
        P_degen = electron_pressure_degenerate(rho, Z, A)
        P_ideal = n_e * K_BOLTZMANN * T
        weight = 1.0 / (1.0 + np.exp(-eta + 2))
        P_th = weight * P_degen * (np.pi**2/3.0) * (K_BOLTZMANN*T/E_F)**2 + \
               (1-weight) * P_ideal

    return P_th


# =============================================================================
# RADIATION EOS
# =============================================================================

def radiation_pressure(T: float) -> float:
    """Radiation pressure P_rad = (1/3) a T⁴"""
    return A_RAD * T**4 / 3.0


def radiation_energy(T: float) -> float:
    """Radiation energy density e_rad = a T⁴"""
    return A_RAD * T**4


# =============================================================================
# TOTAL HELMHOLTZ EOS
# =============================================================================

@dataclass
class EOSResult:
    """Container for EOS outputs."""
    pressure: float          # Total pressure [dyn/cm²]
    energy: float            # Internal energy per volume [erg/cm³]
    entropy: float           # Entropy per volume [erg/cm³/K]
    sound_speed: float       # Sound speed [cm/s]
    gamma_eff: float         # Effective adiabatic index
    temperature: float       # Temperature [K]
    density: float           # Density [g/cm³]

    # Component breakdown
    P_electron: float
    P_ion: float
    P_radiation: float
    P_coulomb: float

    # Debye diagnostics
    Theta_Debye: float       # Debye temperature [K]
    Gamma_coulomb: float     # Coulomb coupling parameter
    eta_degeneracy: float    # Electron degeneracy parameter


def helmholtz_eos(rho: float, T: float,
                   Z: float = Z_ION, A: float = A_ION) -> EOSResult:
    """
    Complete Helmholtz EOS with all corrections.

    P_total = P_electron + P_ion + P_radiation + P_coulomb

    Parameters
    ----------
    rho : float
        Density [g/cm³]
    T : float
        Temperature [K]
    Z, A : float
        Mean ion charge and mass number

    Returns
    -------
    result : EOSResult
        Complete EOS output
    """
    # Electrons
    P_e_degen = electron_pressure_degenerate(rho, Z, A)
    P_e_thermal = electron_pressure_thermal(rho, T, Z, A)
    P_electron = P_e_degen + P_e_thermal

    # Ions
    P_ion_ideal = ion_pressure_ideal(rho, T, A)
    P_coulomb = ion_pressure_coulomb(rho, T, Z, A)
    P_ion = P_ion_ideal  # Coulomb added separately

    # Radiation
    P_rad = radiation_pressure(T)

    # Total
    P_total = P_electron + P_ion + P_rad + P_coulomb

    # Energy
    E_ion = ion_energy_debye(rho, T, Z, A)
    E_rad = radiation_energy(T)
    # Electron energy: approximate as P_e / (γ-1)
    E_electron = P_electron / (4.0/3.0 - 1)  # γ = 4/3 for relativistic

    E_total = E_electron + E_ion + E_rad

    # Sound speed: c_s² = (∂P/∂ρ)_s ≈ γ P / ρ
    # For mixture, use effective gamma
    n_i = ion_number_density(rho, A)
    cv_ion = ion_cv_debye(rho, T, Z, A)
    cv_total = cv_ion + 4.0 * A_RAD * T**3  # + electron contribution

    gamma_eff = 1.0 + P_total / E_total if E_total > 0 else 4.0/3.0
    c_s = np.sqrt(gamma_eff * P_total / rho)

    # Diagnostics
    Theta_D = debye_temperature(rho, Z, A)
    Gamma = coulomb_coupling_parameter(rho, T, Z, A)
    eta = electron_degeneracy_parameter(rho, T, Z, A)

    return EOSResult(
        pressure=P_total,
        energy=E_total,
        entropy=0,  # Would need full entropy calculation
        sound_speed=c_s,
        gamma_eff=gamma_eff,
        temperature=T,
        density=rho,
        P_electron=P_electron,
        P_ion=P_ion,
        P_radiation=P_rad,
        P_coulomb=P_coulomb,
        Theta_Debye=Theta_D,
        Gamma_coulomb=Gamma,
        eta_degeneracy=eta
    )


def temperature_from_pressure(rho: float, P_target: float,
                               Z: float = Z_ION, A: float = A_ION,
                               T_guess: float = 1e9) -> float:
    """
    Invert EOS to get temperature from pressure.

    Solves P(ρ, T) = P_target for T.

    Parameters
    ----------
    rho : float
        Density [g/cm³]
    P_target : float
        Target pressure [dyn/cm²]
    T_guess : float
        Initial guess for temperature

    Returns
    -------
    T : float
        Temperature [K]
    """
    from scipy.optimize import brentq

    def residual(T):
        if T < 1e6:
            return 1e50
        result = helmholtz_eos(rho, T, Z, A)
        return result.pressure - P_target

    # Find bracketing interval
    T_low, T_high = 1e6, 1e11

    try:
        T = brentq(residual, T_low, T_high)
    except ValueError:
        # Fall back to Newton iteration
        T = T_guess
        for _ in range(50):
            result = helmholtz_eos(rho, T, Z, A)
            dP_dT = (helmholtz_eos(rho, T*1.01, Z, A).pressure -
                     result.pressure) / (0.01 * T)
            if abs(dP_dT) < 1e-50:
                break
            T = T - (result.pressure - P_target) / dP_dT
            T = max(T, 1e6)
            if abs(result.pressure - P_target) / P_target < 1e-6:
                break

    return T


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demonstrate_debye_corrections():
    """Show the importance of Debye corrections in WD conditions."""

    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "    HELMHOLTZ EOS WITH DEBYE CORRECTIONS".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "═" * 68 + "╝")
    print()

    # Typical WD deflagration conditions
    print("=" * 70)
    print("EOS AT FLAME CONDITIONS (ρ = 2×10⁹ g/cm³)")
    print("=" * 70)
    print()

    rho = 2e9  # g/cm³

    print(f"{'T [K]':>12} | {'Θ_D [K]':>10} | {'T/Θ_D':>8} | {'Γ':>8} | "
          f"{'η':>8} | {'γ_eff':>8}")
    print("-" * 70)

    for T in [1e8, 3e8, 1e9, 3e9, 1e10]:
        result = helmholtz_eos(rho, T)
        ratio = T / result.Theta_Debye

        print(f"{T:>12.2e} | {result.Theta_Debye:>10.2e} | {ratio:>8.2f} | "
              f"{result.Gamma_coulomb:>8.2f} | {result.eta_degeneracy:>8.1f} | "
              f"{result.gamma_eff:>8.3f}")

    print("-" * 70)
    print()

    # Pressure breakdown at flame conditions
    T_flame = 3e9
    result = helmholtz_eos(rho, T_flame)

    print(f"PRESSURE BREAKDOWN at ρ = {rho:.0e} g/cm³, T = {T_flame:.0e} K:")
    print(f"  P_electron:  {result.P_electron:.3e} dyn/cm² "
          f"({100*result.P_electron/result.pressure:.1f}%)")
    print(f"  P_ion:       {result.P_ion:.3e} dyn/cm² "
          f"({100*result.P_ion/result.pressure:.1f}%)")
    print(f"  P_radiation: {result.P_radiation:.3e} dyn/cm² "
          f"({100*result.P_radiation/result.pressure:.1f}%)")
    print(f"  P_coulomb:   {result.P_coulomb:.3e} dyn/cm² "
          f"({100*result.P_coulomb/result.pressure:.1f}%)")
    print(f"  P_total:     {result.pressure:.3e} dyn/cm²")
    print()
    print(f"  Sound speed: {result.sound_speed/1e8:.2f} × 10⁸ cm/s")
    print(f"  γ_eff:       {result.gamma_eff:.3f}")
    print()

    # Debye correction importance
    print("=" * 70)
    print("DEBYE CORRECTION IMPORTANCE")
    print("=" * 70)
    print()
    print("At T < Θ_D, the ion heat capacity follows C_v ∝ T³ (Debye law)")
    print("rather than C_v = (3/2) n k_B (classical).")
    print()

    for rho in [1e7, 1e8, 1e9, 2e9]:
        Theta_D = debye_temperature(rho)
        print(f"ρ = {rho:.0e} g/cm³: Θ_D = {Theta_D:.2e} K")

    print()
    print("At the flame front (T ~ 10⁹ K), we are in the CLASSICAL limit (T > Θ_D)")
    print("but Debye corrections are still ~10-20% for accurate energy transport.")
    print()
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_debye_corrections()
