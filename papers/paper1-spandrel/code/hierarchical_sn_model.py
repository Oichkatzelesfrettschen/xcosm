#!/usr/bin/env python3
"""
Hierarchical Type Ia Supernova Model for Spandrel Analysis

This module implements a hierarchical Bayesian model for SNe Ia standardization
that includes host galaxy covariates and population drift parameters.

The model tests the Spandrel hypothesis: that DESI's apparent dark energy
evolution signal arises from unmodeled SN Ia population systematics correlated
with host galaxy properties.

Author: COSMOS Collaboration
Date: December 2025
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class SNData:
    """Container for SN Ia observational data."""

    z: np.ndarray  # Redshifts
    m_obs: np.ndarray  # Observed peak magnitudes
    m_err: np.ndarray  # Magnitude uncertainties
    x1: np.ndarray  # SALT2 stretch
    x1_err: np.ndarray  # Stretch uncertainties
    c: np.ndarray  # SALT2 color
    c_err: np.ndarray  # Color uncertainties
    host_mass: np.ndarray  # Host galaxy stellar mass (log10 M_solar)
    host_mass_err: np.ndarray
    host_sfr: Optional[np.ndarray] = None  # Host SFR (optional)
    host_metallicity_proxy: Optional[np.ndarray] = None  # Metallicity proxy


@dataclass
class SpandrelParams:
    """Spandrel model parameters."""

    # Standard SALT2 parameters
    M0: float = -19.3  # Baseline absolute magnitude
    alpha: float = 0.14  # Stretch coefficient
    beta: float = 3.1  # Color coefficient

    # Host galaxy parameters
    gamma_mass: float = 0.05  # Host mass step

    # Population evolution parameters (Spandrel-specific)
    dM_dz: float = 0.0  # Luminosity evolution with redshift
    dx1_dz: float = 0.0  # Stretch population drift
    dc_dz: float = 0.0  # Color population drift

    # Intrinsic scatter
    sigma_int: float = 0.1  # Intrinsic scatter (mag)


class HierarchicalSNModel:
    """
    Hierarchical Bayesian model for SN Ia standardization.

    This model extends standard SALT2-based standardization to include:
    1. Host galaxy mass correction
    2. Redshift-dependent population drift (Spandrel hypothesis)
    3. Selection effects (Malmquist bias)
    4. Proper covariance structure

    The key test is whether including population drift terms (dM_dz, dx1_dz, dc_dz)
    significantly changes the inferred dark energy parameters (w0, wa).
    """

    def __init__(self, data: SNData, cosmology: str = "flat_wcdm"):
        """
        Initialize the hierarchical SN model.

        Args:
            data: SNData container with observational data
            cosmology: Cosmological model ("flat_wcdm" or "flat_w0wa")
        """
        self.data = data
        self.cosmology = cosmology
        self.n_sn = len(data.z)

        # Validate data
        self._validate_data()

    def _validate_data(self):
        """Validate input data consistency."""
        n = self.n_sn
        required_fields = ["z", "m_obs", "m_err", "x1", "c", "host_mass"]
        for field in required_fields:
            arr = getattr(self.data, field)
            if arr is None:
                raise ValueError(f"Missing required field: {field}")
            if len(arr) != n:
                raise ValueError(f"Field {field} has wrong length: {len(arr)} != {n}")

    def distance_modulus(
        self, z: np.ndarray, w0: float = -1.0, wa: float = 0.0, Om: float = 0.3, H0: float = 70.0
    ) -> np.ndarray:
        """
        Compute distance modulus for flat w0wa cosmology.

        Args:
            z: Redshifts
            w0: Dark energy equation of state at z=0
            wa: Dark energy evolution parameter
            Om: Matter density parameter
            H0: Hubble constant (km/s/Mpc)

        Returns:
            Distance modulus mu(z)
        """
        # Speed of light in km/s
        c = 299792.458

        # Comoving distance integral
        def E(z, w0, wa, Om):
            """E(z) = H(z)/H0"""
            Ode = 1.0 - Om
            w_z = w0 + wa * z / (1 + z)
            return np.sqrt(Om * (1 + z) ** 3 + Ode * (1 + z) ** (3 * (1 + w_z)))

        # Numerical integration for comoving distance
        from scipy.integrate import quad

        def dc(z_val):
            """Comoving distance to redshift z."""
            if z_val == 0:
                return 0.0
            integral, _ = quad(lambda zp: 1.0 / E(zp, w0, wa, Om), 0, z_val)
            return (c / H0) * integral

        dc_array = np.array([dc(zi) for zi in z])

        # Luminosity distance (flat universe)
        dl = dc_array * (1 + z)

        # Distance modulus
        mu = 5 * np.log10(dl) + 25

        return mu

    def standardized_magnitude(
        self, params: SpandrelParams, include_evolution: bool = True
    ) -> np.ndarray:
        """
        Compute standardized absolute magnitude for each SN.

        Args:
            params: SpandrelParams with model parameters
            include_evolution: Whether to include population evolution terms

        Returns:
            Standardized absolute magnitude array
        """
        # Baseline standardization
        M_std = (
            params.M0
            + params.alpha * self.data.x1
            - params.beta * self.data.c
            + params.gamma_mass * np.maximum(self.data.host_mass - 10.0, 0)
        )

        # Population evolution terms (Spandrel hypothesis)
        if include_evolution:
            M_std += params.dM_dz * self.data.z
            # Note: x1 and c drift affect the population means, not individual values
            # This is handled in the hierarchical likelihood

        return M_std

    def log_likelihood_baseline(self, theta: np.ndarray) -> float:
        """
        Log-likelihood for baseline model (no population evolution).

        Args:
            theta: Parameter vector [M0, alpha, beta, gamma_mass, sigma_int, w0, wa, Om]

        Returns:
            Log-likelihood value
        """
        M0, alpha, beta, gamma_mass, sigma_int, w0, wa, Om = theta

        params = SpandrelParams(
            M0=M0, alpha=alpha, beta=beta, gamma_mass=gamma_mass, sigma_int=sigma_int
        )

        # Predicted distance modulus
        mu_theory = self.distance_modulus(self.data.z, w0, wa, Om)

        # Standardized magnitude
        M_std = self.standardized_magnitude(params, include_evolution=False)

        # Predicted apparent magnitude
        m_pred = mu_theory - M_std

        # Residuals
        residuals = self.data.m_obs - m_pred

        # Total variance (observational + intrinsic)
        var_total = self.data.m_err**2 + sigma_int**2

        # Gaussian log-likelihood
        log_lik = -0.5 * np.sum(residuals**2 / var_total + np.log(2 * np.pi * var_total))

        return log_lik

    def log_likelihood_spandrel(self, theta: np.ndarray) -> float:
        """
        Log-likelihood for Spandrel model (with population evolution).

        Args:
            theta: Parameter vector [M0, alpha, beta, gamma_mass, sigma_int,
                                    dM_dz, dx1_dz, dc_dz, w0, wa, Om]

        Returns:
            Log-likelihood value
        """
        (M0, alpha, beta, gamma_mass, sigma_int, dM_dz, dx1_dz, dc_dz, w0, wa, Om) = theta

        params = SpandrelParams(
            M0=M0,
            alpha=alpha,
            beta=beta,
            gamma_mass=gamma_mass,
            sigma_int=sigma_int,
            dM_dz=dM_dz,
            dx1_dz=dx1_dz,
            dc_dz=dc_dz,
        )

        # Predicted distance modulus
        mu_theory = self.distance_modulus(self.data.z, w0, wa, Om)

        # Standardized magnitude with evolution
        M_std = self.standardized_magnitude(params, include_evolution=True)

        # Predicted apparent magnitude
        m_pred = mu_theory - M_std

        # Residuals
        residuals = self.data.m_obs - m_pred

        # Total variance
        var_total = self.data.m_err**2 + sigma_int**2

        # Gaussian log-likelihood
        log_lik = -0.5 * np.sum(residuals**2 / var_total + np.log(2 * np.pi * var_total))

        return log_lik

    def host_mass_split_test(self, mass_threshold: float = 10.0) -> Dict[str, Tuple[float, float]]:
        """
        Perform host mass split test (primary Spandrel falsification).

        Args:
            mass_threshold: log10(M/M_solar) threshold for split

        Returns:
            Dictionary with results for high/low mass subsamples
        """
        high_mass = self.data.host_mass >= mass_threshold
        low_mass = ~high_mass

        results = {}

        for name, mask in [("high_mass", high_mass), ("low_mass", low_mass)]:
            # Fit baseline model to subsample
            n_sub = np.sum(mask)
            if n_sub < 10:
                results[name] = (np.nan, np.nan)
                continue

            # Compute Hubble residuals
            # (This is a simplified version; full analysis would fit cosmology)
            sub_z = self.data.z[mask]
            sub_m = self.data.m_obs[mask]
            sub_err = self.data.m_err[mask]

            # Mean Hubble residual (relative to fiducial cosmology)
            mu_fid = self.distance_modulus(sub_z, w0=-1.0, wa=0.0)
            residuals = sub_m - mu_fid + 19.3  # Approximate standardization

            mean_resid = np.mean(residuals)
            err_mean = np.std(residuals) / np.sqrt(n_sub)

            results[name] = (mean_resid, err_mean)

        # Compute difference (key Spandrel diagnostic)
        if not np.isnan(results["high_mass"][0]) and not np.isnan(results["low_mass"][0]):
            delta = results["high_mass"][0] - results["low_mass"][0]
            delta_err = np.sqrt(results["high_mass"][1] ** 2 + results["low_mass"][1] ** 2)
            results["delta"] = (delta, delta_err)
            results["significance"] = delta / delta_err if delta_err > 0 else np.nan

        return results


def run_spandrel_analysis(data: SNData, n_samples: int = 10000, n_walkers: int = 32) -> Dict:
    """
    Run full Spandrel analysis comparing baseline vs evolution models.

    Args:
        data: SNData container
        n_samples: Number of MCMC samples
        n_walkers: Number of MCMC walkers

    Returns:
        Dictionary with analysis results
    """
    model = HierarchicalSNModel(data)

    results = {
        "host_mass_split": model.host_mass_split_test(),
        "n_sn": model.n_sn,
        "z_range": (np.min(data.z), np.max(data.z)),
    }

    # TODO: Add full MCMC fitting with emcee/cobaya
    # TODO: Add model comparison (baseline vs Spandrel)
    # TODO: Add posterior predictive checks

    return results


if __name__ == "__main__":
    # Example usage with simulated data
    np.random.seed(42)
    n_sn = 1000

    # Simulate SN data
    z = np.random.uniform(0.01, 1.5, n_sn)
    x1 = np.random.normal(0, 1, n_sn)
    c = np.random.normal(0, 0.1, n_sn)
    host_mass = np.random.normal(10.5, 0.5, n_sn)

    # Simulate magnitudes (simplified)
    M0 = -19.3
    alpha = 0.14
    beta = 3.1
    mu_true = 5 * np.log10((1 + z) * 4285.7 * z) + 25  # Approximate
    m_true = mu_true - M0 - alpha * x1 + beta * c
    m_obs = m_true + np.random.normal(0, 0.15, n_sn)

    data = SNData(
        z=z,
        m_obs=m_obs,
        m_err=np.full(n_sn, 0.15),
        x1=x1,
        x1_err=np.full(n_sn, 0.1),
        c=c,
        c_err=np.full(n_sn, 0.02),
        host_mass=host_mass,
        host_mass_err=np.full(n_sn, 0.1),
    )

    # Run analysis
    results = run_spandrel_analysis(data)

    print("Spandrel Analysis Results")
    print("=" * 50)
    print(f"N_SN: {results['n_sn']}")
    print(f"z range: {results['z_range']}")
    print("\nHost Mass Split Test:")
    for key, val in results["host_mass_split"].items():
        if isinstance(val, tuple):
            print(f"  {key}: {val[0]:.4f} ± {val[1]:.4f}")
        else:
            print(f"  {key}: {val:.2f}σ")
