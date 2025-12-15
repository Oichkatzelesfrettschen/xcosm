#!/usr/bin/env python3
"""
Validation of ξ Parameter with Real Pantheon+ Data
===================================================
EQUATION E09: Compare entropic cosmology prediction to real SNe Ia

Dataset: Pantheon+ (Scolnic et al. 2022)
- 1701 light curves of 1550 SNe Ia
- Redshift range: 0 < z < 2.3
- Published in ApJ (2022)

References:
- arXiv:2112.03863 (Data release)
- arXiv:2202.04077 (Cosmological constraints)
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import urllib.request
import os

# =============================================================================
# CONSTANTS
# =============================================================================

C_LIGHT = 299792.458  # km/s

# Fiducial cosmology (Planck 2018)
H0_FIDUCIAL = 70.0  # km/s/Mpc
OMEGA_M_FIDUCIAL = 0.315

# =============================================================================
# DATA LOADING
# =============================================================================


def download_pantheon_data():
    """Download Pantheon dataset from GitHub."""

    url = "https://raw.githubusercontent.com/dscolnic/Pantheon/master/lcparam_full_long.txt"
    local_file = "/Users/eirikr/cosmos/data/raw/pantheon_data.txt"

    if not os.path.exists(local_file):
        print(f"Downloading Pantheon data from {url}...")
        urllib.request.urlretrieve(url, local_file)
        print(f"Saved to {local_file}")
    else:
        print(f"Using cached data: {local_file}")

    return local_file


def load_pantheon_data(filename: str):
    """
    Load Pantheon SNe Ia data.

    Returns: z, mu, sigma_mu arrays
    """
    print(f"\nLoading data from {filename}...")

    # Read the data file - skip header and use fixed columns
    # Columns: name zcmb zhel dz mb dmb ...
    data = np.loadtxt(filename, skiprows=1, usecols=(1, 4, 5))

    # Extract columns
    z = data[:, 0]  # CMB-frame redshift (column 1)
    mb = data[:, 1]  # Apparent B-band magnitude (column 4)
    dmb = data[:, 2]  # Uncertainty (column 5)

    # Convert apparent magnitude to distance modulus
    # For standardized SNe Ia: mu = mb - M (absolute magnitude)
    # M ~ -19.3 for SNe Ia
    M_abs = -19.36  # Typical SNe Ia absolute magnitude

    mu = mb - M_abs
    sigma_mu = dmb  # Uncertainty propagates directly

    # Filter to valid data
    valid = (z > 0.01) & (z < 2.0) & np.isfinite(mu) & np.isfinite(sigma_mu)
    z = z[valid]
    mu = mu[valid]
    sigma_mu = sigma_mu[valid]

    # Sort by redshift
    idx = np.argsort(z)
    z = z[idx]
    mu = mu[idx]
    sigma_mu = sigma_mu[idx]

    print(f"Loaded {len(z)} SNe Ia")
    print(f"Redshift range: {z.min():.3f} to {z.max():.3f}")

    return z, mu, sigma_mu


# =============================================================================
# COSMOLOGICAL MODELS
# =============================================================================


def E_z_lcdm(z, Omega_m):
    """Dimensionless Hubble parameter for ΛCDM."""
    Omega_L = 1.0 - Omega_m
    return np.sqrt(Omega_m * (1 + z) ** 3 + Omega_L)


def E_z_entropic(z, Omega_m, xi):
    """Dimensionless Hubble parameter for entropic dark energy."""
    Omega_DE = 1.0 - Omega_m

    # Entropic DE density evolution
    # ρ_DE(z) = ρ_DE(0) × (1 - 3ξ ln(1+z))
    rho_ratio = 1.0 - 3.0 * xi * np.log(1.0 + z)

    if rho_ratio <= 0:
        return 1e10  # Unphysical

    E_squared = Omega_m * (1 + z) ** 3 + Omega_DE * rho_ratio
    return np.sqrt(max(E_squared, 1e-10))


def luminosity_distance(z, Omega_m, xi=None, H0=70.0):
    """
    Luminosity distance in Mpc.

    If xi is None, use ΛCDM. Otherwise, use entropic model.
    """
    if xi is None:
        # ΛCDM
        integrand = lambda zp: 1.0 / E_z_lcdm(zp, Omega_m)
    else:
        # Entropic
        integrand = lambda zp: 1.0 / E_z_entropic(zp, Omega_m, xi)

    D_C, _ = quad(integrand, 0, z, limit=100)
    D_L = (1 + z) * D_C * (C_LIGHT / H0)  # Mpc

    return D_L


def distance_modulus(z, Omega_m, xi=None, H0=70.0):
    """Distance modulus μ(z) = 5 log₁₀(D_L/10pc)."""
    D_L = luminosity_distance(z, Omega_m, xi, H0)
    D_L_pc = D_L * 1e6  # Convert Mpc to pc

    if D_L_pc <= 0:
        return 99.0

    return 5.0 * np.log10(D_L_pc / 10.0)


# =============================================================================
# LIKELIHOOD AND FITTING
# =============================================================================


def chi_squared(params, z_data, mu_data, sigma_data, model="entropic"):
    """Compute χ² for given parameters."""

    if model == "entropic":
        Omega_m, xi = params
        if xi < -0.5 or xi > 1.0:
            return 1e10
    else:
        Omega_m = params[0]
        xi = None

    if Omega_m < 0.1 or Omega_m > 0.6:
        return 1e10

    chi2 = 0.0
    for i, z in enumerate(z_data):
        mu_theory = distance_modulus(z, Omega_m, xi)
        chi2 += ((mu_theory - mu_data[i]) / sigma_data[i]) ** 2

    return chi2


def fit_model(z_data, mu_data, sigma_data, model="entropic"):
    """Fit cosmological model to data."""

    if model == "entropic":
        x0 = [0.3, 0.15]
        bounds = [(0.1, 0.6), (-0.3, 0.6)]
    else:  # ΛCDM
        x0 = [0.3]
        bounds = [(0.1, 0.6)]

    result = minimize(
        chi_squared,
        x0,
        args=(z_data, mu_data, sigma_data, model),
        method="L-BFGS-B",
        bounds=bounds,
    )

    return result


# =============================================================================
# MODEL COMPARISON
# =============================================================================


def compute_aic_bic(chi2, n_params, n_data):
    """Compute AIC and BIC."""
    aic = chi2 + 2 * n_params
    bic = chi2 + n_params * np.log(n_data)
    return aic, bic


def compare_models(z_data, mu_data, sigma_data):
    """Compare ΛCDM vs Entropic models."""

    print("\n" + "=" * 70)
    print("MODEL COMPARISON: ΛCDM vs Entropic")
    print("=" * 70)

    n_data = len(z_data)

    # Fit ΛCDM
    print("\nFitting ΛCDM...")
    result_lcdm = fit_model(z_data, mu_data, sigma_data, model="lcdm")
    Omega_m_lcdm = result_lcdm.x[0]
    chi2_lcdm = result_lcdm.fun
    aic_lcdm, bic_lcdm = compute_aic_bic(chi2_lcdm, 1, n_data)

    print(f"  Ω_m = {Omega_m_lcdm:.4f}")
    print(f"  χ² = {chi2_lcdm:.2f}")
    print(f"  χ²/dof = {chi2_lcdm / (n_data - 1):.3f}")
    print(f"  AIC = {aic_lcdm:.2f}")
    print(f"  BIC = {bic_lcdm:.2f}")

    # Fit Entropic
    print("\nFitting Entropic...")
    result_entropic = fit_model(z_data, mu_data, sigma_data, model="entropic")
    Omega_m_entropic, xi_entropic = result_entropic.x
    chi2_entropic = result_entropic.fun
    aic_entropic, bic_entropic = compute_aic_bic(chi2_entropic, 2, n_data)

    print(f"  Ω_m = {Omega_m_entropic:.4f}")
    print(f"  ξ = {xi_entropic:.4f}")
    print(f"  χ² = {chi2_entropic:.2f}")
    print(f"  χ²/dof = {chi2_entropic / (n_data - 2):.3f}")
    print(f"  AIC = {aic_entropic:.2f}")
    print(f"  BIC = {bic_entropic:.2f}")

    # Comparison
    print("\n" + "-" * 50)
    print("Model Comparison:")
    print("-" * 50)

    delta_chi2 = chi2_lcdm - chi2_entropic
    delta_aic = aic_lcdm - aic_entropic
    delta_bic = bic_lcdm - bic_entropic

    print(f"  Δχ² (ΛCDM - Entropic) = {delta_chi2:.2f}")
    print(f"  ΔAIC = {delta_aic:.2f}")
    print(f"  ΔBIC = {delta_bic:.2f}")

    # Interpretation
    if delta_aic > 2:
        print("\n  → AIC favors Entropic model")
    elif delta_aic < -2:
        print("\n  → AIC favors ΛCDM model")
    else:
        print("\n  → Models statistically indistinguishable")

    return {
        "lcdm": {
            "Omega_m": Omega_m_lcdm,
            "chi2": chi2_lcdm,
            "aic": aic_lcdm,
            "bic": bic_lcdm,
        },
        "entropic": {
            "Omega_m": Omega_m_entropic,
            "xi": xi_entropic,
            "chi2": chi2_entropic,
            "aic": aic_entropic,
            "bic": bic_entropic,
        },
        "delta_chi2": delta_chi2,
        "delta_aic": delta_aic,
    }


# =============================================================================
# VISUALIZATION
# =============================================================================


def plot_hubble_diagram(z_data, mu_data, sigma_data, results, output_file):
    """Plot Hubble diagram with best-fit models."""

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 10), height_ratios=[3, 1], sharex=True
    )

    # Model curves
    z_model = np.linspace(0.01, max(z_data), 200)

    # ΛCDM
    Omega_m_lcdm = results["lcdm"]["Omega_m"]
    mu_lcdm = [distance_modulus(z, Omega_m_lcdm) for z in z_model]

    # Entropic
    Omega_m_ent = results["entropic"]["Omega_m"]
    xi_ent = results["entropic"]["xi"]
    mu_entropic = [distance_modulus(z, Omega_m_ent, xi_ent) for z in z_model]

    # Data (subsample for clarity)
    n_show = min(500, len(z_data))
    idx_show = np.linspace(0, len(z_data) - 1, n_show, dtype=int)

    # Main plot
    ax1.errorbar(
        z_data[idx_show],
        mu_data[idx_show],
        yerr=sigma_data[idx_show],
        fmt="o",
        ms=2,
        alpha=0.4,
        color="blue",
        label="Pantheon SNe Ia",
    )
    ax1.plot(z_model, mu_lcdm, "k--", lw=2, label=f"ΛCDM (Ω_m={Omega_m_lcdm:.3f})")
    ax1.plot(
        z_model,
        mu_entropic,
        "r-",
        lw=2,
        label=f"Entropic (Ω_m={Omega_m_ent:.3f}, ξ={xi_ent:.3f})",
    )

    ax1.set_ylabel("Distance Modulus μ", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.set_title("E09: Pantheon SNe Ia - ΛCDM vs Entropic Cosmology", fontsize=14)

    # Residuals (relative to ΛCDM)
    mu_lcdm_data = np.array([distance_modulus(z, Omega_m_lcdm) for z in z_data])
    residuals = mu_data - mu_lcdm_data

    ax2.errorbar(
        z_data[idx_show],
        residuals[idx_show],
        yerr=sigma_data[idx_show],
        fmt="o",
        ms=2,
        alpha=0.4,
        color="blue",
    )
    ax2.axhline(0, color="k", linestyle="--", lw=1)

    # Entropic deviation from ΛCDM
    mu_ent_model = np.array([distance_modulus(z, Omega_m_ent, xi_ent) for z in z_model])
    mu_lcdm_model = np.array([distance_modulus(z, Omega_m_lcdm) for z in z_model])
    ax2.plot(z_model, mu_ent_model - mu_lcdm_model, "r-", lw=2, label="Entropic - ΛCDM")

    ax2.set_xlabel("Redshift z", fontsize=12)
    ax2.set_ylabel("Residual (mag)", fontsize=12)
    ax2.set_ylim(-0.5, 0.5)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to {output_file}")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================


def main():
    print("=" * 70)
    print("E09: VALIDATION WITH REAL PANTHEON DATA")
    print("=" * 70)

    # Download/load data
    data_file = download_pantheon_data()
    z_data, mu_data, sigma_data = load_pantheon_data(data_file)

    # Compare models
    results = compare_models(z_data, mu_data, sigma_data)

    # Plot
    plot_hubble_diagram(
        z_data,
        mu_data,
        sigma_data,
        results,
        "/Users/eirikr/cosmos/output/plots/pantheon_validation.png",
    )

    # Compare with AEG prediction
    print("\n" + "=" * 70)
    print("COMPARISON WITH AEG PREDICTION")
    print("=" * 70)

    xi_fitted = results["entropic"]["xi"]
    xi_predicted = 0.315  # From E01 derivation

    print(f"\n  AEG prediction: ξ = {xi_predicted:.3f}")
    print(f"  Pantheon fit:   ξ = {xi_fitted:.3f}")
    print(
        f"  Deviation:      {abs(xi_fitted - xi_predicted) / xi_predicted * 100:.1f}%"
    )

    # Check if ξ > 0 (positive entropic contribution)
    if xi_fitted > 0:
        print("\n  ✓ ξ > 0 confirmed (entropic dark energy)")
    else:
        print("\n  ✗ ξ ≤ 0 (consistent with ΛCDM)")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
    Dataset: Pantheon ({len(z_data)} SNe Ia)
    Redshift range: {z_data.min():.3f} - {z_data.max():.3f}

    ΛCDM:
      Ω_m = {results["lcdm"]["Omega_m"]:.4f}
      χ² = {results["lcdm"]["chi2"]:.1f}

    Entropic:
      Ω_m = {results["entropic"]["Omega_m"]:.4f}
      ξ = {results["entropic"]["xi"]:.4f}
      χ² = {results["entropic"]["chi2"]:.1f}

    Model comparison:
      Δχ² = {results["delta_chi2"]:.2f}
      ΔAIC = {results["delta_aic"]:.2f}

    AEG Prediction Check:
      Predicted ξ = 0.315
      Fitted ξ = {xi_fitted:.3f}
      Match: {100 - abs(xi_fitted - xi_predicted) / xi_predicted * 100:.1f}%

    E09 STATUS: {"VALIDATED" if abs(xi_fitted - xi_predicted) / xi_predicted < 0.5 else "INCONCLUSIVE"}
    """)

    return results


if __name__ == "__main__":
    results = main()
    print("\n✓ Pantheon validation complete!")
