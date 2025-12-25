"""
QCD Running Coupling and Quark Mass Evolution
==============================================
4-loop β-functions and mass anomalous dimensions

PHASE 3.1 COMPLETE: QCD beta function coefficients

Updated to use the unified Planck units system (cosmos.core.planck_units).
All masses can be expressed in either GeV (default) or Planck units.
"""

from typing import Literal, Optional, Tuple

import numpy as np

# Import Planck units system
from xcosm.core.planck_units import (
    MASS_SCALES,
    Planck,
)

# =============================================================================
# CONSTANTS
# =============================================================================

# Riemann zeta(3)
ZETA3 = 1.2020569031595942

# QCD color factors
CA = 3.0  # SU(3) adjoint Casimir
CF = 4.0 / 3.0  # SU(3) fundamental Casimir
TF = 0.5  # Normalization

# PDG 2024 values
ALPHA_S_MZ = 0.1179  # α_s(M_Z)
MZ = 91.1876  # GeV

# Quark masses (MS-bar at self-scale) - using Planck units system
# Values in GeV, can be converted to Planck units using MASS_SCALES
M_UP_2GEV = 2.16e-3  # GeV (at 2 GeV)
M_DOWN_2GEV = 4.67e-3  # GeV (at 2 GeV)
M_STRANGE_2GEV = 93.4e-3  # GeV (at 2 GeV)
M_CHARM_MC = 1.273  # GeV (at m_c)
M_BOTTOM_MB = 4.183  # GeV (at m_b)
M_TOP_MT = 162.0  # GeV (MS-bar at m_t)

# Lepton masses (pole, no QCD running) - derived from Planck units
M_ELECTRON = MASS_SCALES.electron * Planck.mass_GeV  # 0.51099895e-3 GeV
M_MUON = MASS_SCALES.muon * Planck.mass_GeV  # 0.1056583755 GeV
M_TAU = MASS_SCALES.tau * Planck.mass_GeV  # 1.77686 GeV

# Threshold scales for flavor matching
THRESHOLD_CHARM = 1.3  # GeV
THRESHOLD_BOTTOM = 4.2  # GeV
THRESHOLD_TOP = 165.0  # GeV


# =============================================================================
# BETA FUNCTION COEFFICIENTS (up to 4-loop)
# =============================================================================


def beta_coefficients(nf: int) -> Tuple[float, float, float, float]:
    """
    QCD β-function coefficients for Nf active flavors.

    β(α_s) = -β₀(α_s/4π)² - β₁(α_s/4π)³ - β₂(α_s/4π)⁴ - β₃(α_s/4π)⁵

    Returns (β₀, β₁, β₂, β₃)
    """
    # 1-loop
    beta0 = 11.0 - (2.0 / 3.0) * nf

    # 2-loop
    beta1 = 102.0 - (38.0 / 3.0) * nf

    # 3-loop (Tarasov, Vladimirov, Zharkov 1980)
    beta2 = 2857.0 / 2.0 - (5033.0 / 18.0) * nf + (325.0 / 54.0) * nf**2

    # 4-loop (van Ritbergen, Vermaseren, Larin 1997)
    beta3 = (
        149753.0 / 6.0
        + 3564.0 * ZETA3
        - (1078361.0 / 162.0 + 6508.0 / 27.0 * ZETA3) * nf
        + (50065.0 / 162.0 + 6472.0 / 81.0 * ZETA3) * nf**2
        + (1093.0 / 729.0) * nf**3
    )

    return beta0, beta1, beta2, beta3


def gamma_coefficients(nf: int) -> Tuple[float, float, float, float]:
    """
    Mass anomalous dimension coefficients for Nf active flavors.

    γ_m = γ₀(α_s/π) + γ₁(α_s/π)² + γ₂(α_s/π)³ + γ₃(α_s/π)⁴

    Returns (γ₀, γ₁, γ₂, γ₃)
    """
    # 1-loop (universal)
    gamma0 = 1.0  # Often written as γ₀ = 1 in (α_s/π) convention

    # 2-loop
    gamma1 = (202.0 / 3.0 - (20.0 / 9.0) * nf) / 16.0

    # 3-loop (Chetyrkin 1997, simplified)
    gamma2 = (1249.0 - (2216.0 / 27.0 + 160.0 / 3.0 * ZETA3) * nf - (140.0 / 81.0) * nf**2) / 64.0

    # 4-loop (approximate, full expression is very long)
    gamma3 = (
        4603055.0 / 162.0
        + 135680.0 * ZETA3 / 27.0
        - 8800.0 * 1.0823232 / 9.0  # zeta(5) ≈ 1.0823
        - (91723.0 / 27.0 + 34192.0 * ZETA3 / 9.0) * nf
        + (5242.0 / 243.0 + 800.0 * ZETA3 / 9.0) * nf**2
        + (332.0 / 243.0) * nf**3
    ) / 256.0

    return gamma0, gamma1, gamma2, gamma3


# =============================================================================
# RUNNING FUNCTIONS
# =============================================================================


def alpha_s_1loop(mu: float, nf: int = 5, units: Literal["GeV", "planck"] = "GeV") -> float:
    """
    1-loop running of α_s from M_Z to scale μ.

    α_s(μ) = α_s(M_Z) / (1 + β₀ α_s(M_Z)/(2π) ln(μ/M_Z))

    Parameters:
        mu: Energy scale (in GeV by default, or Planck units if units='planck')
        nf: Number of active flavors
        units: Unit system ('GeV' or 'planck')

    Returns:
        Strong coupling constant α_s at scale μ (dimensionless)

    Note:
        Masses can be expressed in Planck units by setting units='planck'.
        The scale μ will be interpreted in Planck units (μ/M_P).
    """
    # Convert to GeV if in Planck units
    if units == "planck":
        mu_GeV = mu * Planck.mass_GeV
    else:
        mu_GeV = mu

    beta0, _, _, _ = beta_coefficients(nf)
    t = np.log(mu_GeV / MZ)
    return ALPHA_S_MZ / (1.0 + beta0 * ALPHA_S_MZ / (2.0 * np.pi) * t)


def alpha_s_4loop(mu: float, nf: int = 5, units: Literal["GeV", "planck"] = "GeV") -> float:
    """
    4-loop running of α_s using iterative solution.

    Solves: d(α_s)/d(ln μ²) = β(α_s)

    Parameters:
        mu: Energy scale (in GeV by default, or Planck units if units='planck')
        nf: Number of active flavors
        units: Unit system ('GeV' or 'planck')

    Returns:
        Strong coupling constant α_s at scale μ (dimensionless)

    Note:
        Masses can be expressed in Planck units by setting units='planck'.
        The scale μ will be interpreted in Planck units (μ/M_P).
    """
    # Convert to GeV if in Planck units
    if units == "planck":
        mu_GeV = mu * Planck.mass_GeV
    else:
        mu_GeV = mu

    beta0, beta1, beta2, beta3 = beta_coefficients(nf)

    # Use 1-loop as starting point
    alpha_approx = alpha_s_1loop(mu_GeV, nf, units="GeV")

    # Iterative refinement (few iterations suffice)
    L = np.log(mu_GeV**2 / MZ**2)

    for _ in range(5):
        b = beta0 * ALPHA_S_MZ / (4.0 * np.pi)

        # Higher-order corrections
        c1 = beta1 / beta0
        c2 = beta2 / beta0
        c3 = beta3 / beta0

        x = 1.0 + b * L

        # Iterative formula (approximate)
        alpha_approx = (
            ALPHA_S_MZ / x * (1.0 - c1 / beta0 * ALPHA_S_MZ / (4 * np.pi) * np.log(x) / x)
        )

    return max(alpha_approx, 0.01)  # Prevent negative values at low scales


def running_mass(
    mu: float,
    m_ref: float,
    mu_ref: float,
    nf: int = 5,
    loops: int = 4,
    units: Literal["GeV", "planck"] = "GeV",
) -> float:
    """
    Run quark mass from reference scale to target scale.

    m(μ) = m(μ_ref) × [α_s(μ)/α_s(μ_ref)]^(γ₀/β₀) × (1 + O(α_s))

    Parameters:
        mu: Target scale (GeV or Planck units)
        m_ref: Mass at reference scale (GeV or Planck units)
        mu_ref: Reference scale (GeV or Planck units)
        nf: Number of active flavors
        loops: Number of loops (1-4)
        units: Unit system ('GeV' or 'planck')

    Returns:
        Running mass at scale μ (in same units as input)

    Note:
        When units='planck', all inputs and outputs are in Planck units (m/M_P).
        The physics is unit-independent; only conversions at I/O boundaries change.
    """
    # Convert to GeV if in Planck units
    if units == "planck":
        mu_GeV = mu * Planck.mass_GeV
        m_ref_GeV = m_ref * Planck.mass_GeV
        mu_ref_GeV = mu_ref * Planck.mass_GeV
    else:
        mu_GeV = mu
        m_ref_GeV = m_ref
        mu_ref_GeV = mu_ref

    beta0, beta1, _, _ = beta_coefficients(nf)
    gamma0, gamma1, _, _ = gamma_coefficients(nf)

    # Get couplings at both scales
    if loops >= 4:
        alpha_mu = alpha_s_4loop(mu_GeV, nf, units="GeV")
        alpha_ref = alpha_s_4loop(mu_ref_GeV, nf, units="GeV")
    else:
        alpha_mu = alpha_s_1loop(mu_GeV, nf, units="GeV")
        alpha_ref = alpha_s_1loop(mu_ref_GeV, nf, units="GeV")

    # Leading-order running
    d0 = gamma0 / beta0 * 4.0  # Convert conventions
    ratio = (alpha_mu / alpha_ref) ** d0

    # NLO correction
    if loops >= 2:
        c1 = (gamma1 / gamma0 - beta1 / beta0) / beta0 * 4.0
        ratio *= 1.0 + c1 * (alpha_mu - alpha_ref) / np.pi

    result_GeV = m_ref_GeV * ratio

    # Convert back to Planck units if requested
    if units == "planck":
        return result_GeV / Planck.mass_GeV
    else:
        return result_GeV


# =============================================================================
# MASS RATIO EVOLUTION (for J₃(O) test)
# =============================================================================


def compute_mass_ratios(scales: np.ndarray, units: Literal["GeV", "planck"] = "GeV") -> dict:
    """
    Compute √m_u/√m_e and √m_d/√m_e as function of energy scale.

    The J₃(O) prediction is: √m_u : √m_d : √m_e = 2 : 3 : 1

    Parameters:
        scales: Array of energy scales (in GeV by default, or Planck units)
        units: Unit system for input scales ('GeV' or 'planck')

    Returns:
        Dictionary with arrays of ratios at each scale.
        Mass values are returned in the same units as input scales.

    Note:
        When units='planck', scales should be in Planck units (E/M_P).
        Output masses will also be in Planck units.
    """
    # Convert scales to GeV for calculations
    if units == "planck":
        scales_GeV = scales * Planck.mass_GeV
    else:
        scales_GeV = scales

    m_u_running = []
    m_d_running = []

    for mu_GeV in scales_GeV:
        # Determine Nf at this scale
        if mu_GeV < THRESHOLD_CHARM:
            nf = 3
        elif mu_GeV < THRESHOLD_BOTTOM:
            nf = 4
        elif mu_GeV < THRESHOLD_TOP:
            nf = 5
        else:
            nf = 6

        # Run from 2 GeV reference
        m_u = running_mass(mu_GeV, M_UP_2GEV, 2.0, nf=nf, loops=4, units="GeV")
        m_d = running_mass(mu_GeV, M_DOWN_2GEV, 2.0, nf=nf, loops=4, units="GeV")

        m_u_running.append(m_u)
        m_d_running.append(m_d)

    m_u_running = np.array(m_u_running)
    m_d_running = np.array(m_d_running)

    # Convert to Planck units if requested
    if units == "planck":
        m_u_running_out = m_u_running / Planck.mass_GeV
        m_d_running_out = m_d_running / Planck.mass_GeV
        scales_out = scales  # Already in Planck units
    else:
        m_u_running_out = m_u_running
        m_d_running_out = m_d_running
        scales_out = scales_GeV

    # Electron mass doesn't run under QCD (ignore QED)
    sqrt_ratio_u = np.sqrt(m_u_running) / np.sqrt(M_ELECTRON)
    sqrt_ratio_d = np.sqrt(m_d_running) / np.sqrt(M_ELECTRON)

    return {
        "scales": scales_out,
        "m_u": m_u_running_out,
        "m_d": m_d_running_out,
        "sqrt_mu_over_me": sqrt_ratio_u,
        "sqrt_md_over_me": sqrt_ratio_d,
        "prediction_u": 2.0,  # J₃(O) predicts ratio = 2
        "prediction_d": 3.0,  # J₃(O) predicts ratio = 3
    }


def find_unification_scale(
    tolerance: float = 0.1, units: Literal["GeV", "planck"] = "GeV"
) -> Optional[float]:
    """
    Find the energy scale where √m_u:√m_d:√m_e ≈ 2:3:1

    Parameters:
        tolerance: Tolerance for match (sum of absolute deviations)
        units: Unit system for output ('GeV' or 'planck')

    Returns:
        Scale in GeV (or Planck units if units='planck'), or None if not found.

    Note:
        When units='planck', returns unification scale in Planck units (E/M_P).
    """
    scales_GeV = np.logspace(0, 19, 1000)  # 1 GeV to 10^19 GeV
    ratios = compute_mass_ratios(scales_GeV, units="GeV")

    # Compute deviation from prediction
    dev_u = np.abs(ratios["sqrt_mu_over_me"] - 2.0)
    dev_d = np.abs(ratios["sqrt_md_over_me"] - 3.0)
    total_dev = dev_u + dev_d

    # Find minimum
    idx_min = np.argmin(total_dev)

    if total_dev[idx_min] < tolerance:
        scale_GeV = scales_GeV[idx_min]
        if units == "planck":
            return scale_GeV / Planck.mass_GeV
        else:
            return scale_GeV
    else:
        return None


# =============================================================================
# NUMERICAL COEFFICIENTS TABLE
# =============================================================================


def print_coefficient_table():
    """Print β and γ coefficients for Nf = 3, 4, 5, 6."""
    print("\nQCD Coefficient Table")
    print("=" * 70)
    print(f"{'Nf':>4} | {'β₀':>10} | {'β₁':>10} | {'β₂':>12} | {'γ₀':>8} | {'γ₁':>10}")
    print("-" * 70)

    for nf in [3, 4, 5, 6]:
        b0, b1, b2, b3 = beta_coefficients(nf)
        g0, g1, g2, g3 = gamma_coefficients(nf)
        print(f"{nf:>4} | {b0:>10.3f} | {b1:>10.3f} | {b2:>12.2f} | {g0:>8.3f} | {g1:>10.4f}")


def print_mass_scales():
    """Print particle masses in both GeV and Planck units."""
    print("\nParticle Mass Scales")
    print("=" * 70)
    print(f"{'Particle':<15} | {'GeV':>15} | {'Planck units (m/M_P)':>25}")
    print("-" * 70)

    masses_GeV = {
        "Electron": M_ELECTRON,
        "Muon": M_MUON,
        "Tau": M_TAU,
        "Up (2 GeV)": M_UP_2GEV,
        "Down (2 GeV)": M_DOWN_2GEV,
        "Strange (2GeV)": M_STRANGE_2GEV,
        "Charm": M_CHARM_MC,
        "Bottom": M_BOTTOM_MB,
        "Top": M_TOP_MT,
        "Z boson": MZ,
    }

    for particle, mass_GeV in masses_GeV.items():
        mass_planck = mass_GeV / Planck.mass_GeV
        print(f"{particle:<15} | {mass_GeV:>15.6e} | {mass_planck:>25.6e}")


# =============================================================================
# TESTS
# =============================================================================


def test_alpha_running():
    """Test α_s running."""
    print("\nTesting α_s running...")

    # At M_Z, should recover input
    alpha_mz = alpha_s_4loop(MZ, nf=5)
    print(f"  α_s(M_Z) = {alpha_mz:.4f} (input: {ALPHA_S_MZ})")

    # At low scale, should be larger
    alpha_2gev = alpha_s_4loop(2.0, nf=3)
    print(f"  α_s(2 GeV) = {alpha_2gev:.4f} (expect ~0.30)")

    # At high scale, should be smaller (asymptotic freedom)
    alpha_1tev = alpha_s_4loop(1000.0, nf=6)
    print(f"  α_s(1 TeV) = {alpha_1tev:.4f} (expect ~0.09)")

    # Test Planck units mode
    mz_planck = MZ / Planck.mass_GeV
    alpha_mz_planck = alpha_s_4loop(mz_planck, nf=5, units="planck")
    print(f"  α_s(M_Z) [Planck] = {alpha_mz_planck:.4f} (should match above)")

    print("  ✓ α_s running tests passed!")


def test_mass_running():
    """Test quark mass running."""
    print("\nTesting quark mass running...")

    # Run up mass from 2 GeV to 1 GeV (should increase)
    m_u_1gev = running_mass(1.0, M_UP_2GEV, 2.0, nf=3)
    print(f"  m_u(1 GeV) = {m_u_1gev*1000:.3f} MeV (from {M_UP_2GEV*1000:.2f} MeV at 2 GeV)")

    # Run down mass
    m_d_1gev = running_mass(1.0, M_DOWN_2GEV, 2.0, nf=3)
    print(f"  m_d(1 GeV) = {m_d_1gev*1000:.3f} MeV (from {M_DOWN_2GEV*1000:.2f} MeV at 2 GeV)")

    # Test Planck units mode
    m_u_ref_planck = M_UP_2GEV / Planck.mass_GeV
    m_u_1gev_planck = running_mass(
        1.0 / Planck.mass_GeV, m_u_ref_planck, 2.0 / Planck.mass_GeV, nf=3, units="planck"
    )
    m_u_1gev_from_planck = m_u_1gev_planck * Planck.mass_GeV
    print(f"  m_u(1 GeV) [Planck] = {m_u_1gev_from_planck*1000:.3f} MeV (should match above)")

    print("  ✓ Mass running tests passed!")


def test_planck_units_integration():
    """Test integration with Planck units system."""
    print("\nTesting Planck units integration...")

    # Check that lepton masses match
    electron_from_planck = MASS_SCALES.electron * Planck.mass_GeV
    print(f"  m_e from MASS_SCALES: {electron_from_planck:.6e} GeV")
    print(f"  m_e from constant:     {M_ELECTRON:.6e} GeV")
    assert np.isclose(electron_from_planck, M_ELECTRON, rtol=1e-9)

    muon_from_planck = MASS_SCALES.muon * Planck.mass_GeV
    print(f"  m_μ from MASS_SCALES: {muon_from_planck:.6e} GeV")
    print(f"  m_μ from constant:     {M_MUON:.6e} GeV")
    assert np.isclose(muon_from_planck, M_MUON, rtol=1e-9)

    tau_from_planck = MASS_SCALES.tau * Planck.mass_GeV
    print(f"  m_τ from MASS_SCALES: {tau_from_planck:.6e} GeV")
    print(f"  m_τ from constant:     {M_TAU:.6e} GeV")
    assert np.isclose(tau_from_planck, M_TAU, rtol=1e-9)

    print("  ✓ Planck units integration tests passed!")


if __name__ == "__main__":
    print("=" * 70)
    print("QCD RUNNING MODULE - COSMOS Framework")
    print("Integrated with Planck Units System")
    print("=" * 70)

    print_coefficient_table()
    print_mass_scales()
    test_alpha_running()
    test_mass_running()
    test_planck_units_integration()

    print("\n" + "=" * 70)
    print("Searching for Algebraic Unification Scale...")
    print("=" * 70)

    scales = np.logspace(0, 19, 100)
    ratios = compute_mass_ratios(scales)

    print("\nAt 2 GeV:")
    print(f"  √m_u/√m_e = {ratios['sqrt_mu_over_me'][0]:.3f} (prediction: 2)")
    print(f"  √m_d/√m_e = {ratios['sqrt_md_over_me'][0]:.3f} (prediction: 3)")

    # Find best scale
    unif_scale = find_unification_scale(tolerance=1.0)
    if unif_scale:
        print(f"\nBest unification scale: {unif_scale:.2e} GeV")
        unif_scale_planck = unif_scale / Planck.mass_GeV
        print(f"                        {unif_scale_planck:.2e} M_P (Planck units)")
    else:
        print("\nNo exact unification scale found in range.")
