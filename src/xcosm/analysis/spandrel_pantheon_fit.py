#!/usr/bin/env python3
"""
Spandrel Framework: Fitting metallicity evolution model to Pantheon+ SN Ia data.

This script tests whether the apparent phantom dark energy signal (w < -1) from
DESI can be explained by systematic evolution of SN Ia progenitor metallicity
with redshift.

The Spandrel hypothesis: Lower metallicity at high-z leads to brighter SNe Ia
due to higher C/O ratios and more Ni-56 production.

Model: mu_corrected = mu_observed - delta_mu(Z)
       delta_mu(Z) = xi * log10(Z/Z_solar)
       Z(z) = Z_solar * 10^(-0.15 * z)  [cosmic metallicity evolution]

Author: COSMOS Analysis Pipeline
Date: 2025-12-16
"""

import json
import warnings
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2

warnings.filterwarnings("ignore")

# Output directory
OUTPUT_DIR = Path("/Users/eirikr/1_Workspace/cosmos/paper/output")
OUTPUT_DIR.mkdir(exist_ok=True)


class PantheonPlusData:
    """
    Simulated Pantheon+ data structure for testing Spandrel model.

    Note: This uses a representative sample based on Pantheon+ statistics,
    not the actual proprietary data. For publication, real data access
    would be required.
    """

    def __init__(self, n_sne: int = 1701):
        """
        Initialize with simulated Pantheon+-like dataset.

        Parameters
        ----------
        n_sne : int
            Number of supernovae (Pantheon+ has ~1701)
        """
        self.n_sne = n_sne
        np.random.seed(42)  # Reproducibility

        # Generate redshift distribution matching Pantheon+
        # Most SNe at z < 0.1, tail extending to z ~ 2.3
        z_low = np.random.exponential(0.05, int(0.6 * n_sne))
        z_mid = np.random.uniform(0.1, 0.8, int(0.35 * n_sne))
        z_high = np.random.uniform(0.8, 2.3, int(0.05 * n_sne))

        self.redshifts = np.clip(np.concatenate([z_low, z_mid, z_high]), 0.01, 2.3)
        self.redshifts = np.sort(self.redshifts)[:n_sne]

        # Cosmological parameters (Planck 2018 + phantom signal)
        self.H0 = 73.0  # km/s/Mpc (SH0ES-like for local calibration)
        self.Om = 0.315
        self.w0_apparent = -1.1  # Apparent phantom value from naive fit
        self.wa_apparent = -0.5

        # Generate distance moduli with phantom cosmology
        self.mu_observed = self._compute_distance_modulus(
            self.redshifts, self.w0_apparent, self.wa_apparent
        )

        # Add realistic scatter (intrinsic + measurement)
        self.mu_err = 0.1 + 0.05 * self.redshifts  # Error increases with z
        self.mu_observed += np.random.normal(0, self.mu_err)

        # True underlying cosmology (LCDM)
        self.mu_true_lcdm = self._compute_distance_modulus(self.redshifts, -1.0, 0.0)

    def _compute_distance_modulus(self, z: np.ndarray, w0: float, wa: float) -> np.ndarray:
        """
        Compute distance modulus for w0-wa cosmology.

        Uses numerical integration of:
        D_L = (1+z) * c/H0 * integral[0 to z] dz'/E(z')

        where E(z) = sqrt(Om*(1+z)^3 + (1-Om)*exp(3*int[w(z')/(1+z')]dz'))
        """
        from scipy.integrate import quad

        c_km_s = 299792.458  # km/s

        def w_z(z_prime):
            """Dark energy equation of state w(z) = w0 + wa * z/(1+z)"""
            return w0 + wa * z_prime / (1 + z_prime)

        def integrand_w(z_prime):
            """Integrand for dark energy density evolution."""
            return (1 + w_z(z_prime)) / (1 + z_prime)

        def E_z(z_val):
            """Dimensionless Hubble parameter E(z) = H(z)/H0."""
            if z_val == 0:
                return 1.0

            # Integrate w(z') from 0 to z
            w_integral, _ = quad(integrand_w, 0, z_val)

            # Dark energy density evolution
            rho_de = (1 - self.Om) * np.exp(3 * w_integral)

            # Matter + dark energy
            return np.sqrt(self.Om * (1 + z_val) ** 3 + rho_de)

        def comoving_distance(z_val):
            """Comoving distance in Mpc."""
            if z_val == 0:
                return 0.0
            result, _ = quad(lambda zp: 1.0 / E_z(zp), 0, z_val)
            return (c_km_s / self.H0) * result

        # Compute luminosity distance
        d_L = np.array([(1 + zi) * comoving_distance(zi) for zi in z])

        # Distance modulus: mu = 5 * log10(d_L / 10pc)
        mu = 5 * np.log10(d_L) + 25

        return mu


class SpandrelModel:
    """
    Spandrel Framework: SN Ia luminosity correction for metallicity evolution.

    The model posits that observed "phantom" dark energy is actually an
    artifact of uncorrected metallicity evolution in SN Ia progenitors.
    """

    def __init__(self):
        """Initialize Spandrel model parameters."""
        # Metallicity evolution: Z(z) = Z_solar * 10^(-alpha_Z * z)
        self.alpha_Z = 0.15  # Metallicity evolution slope (dex per unit z)

        # Luminosity-metallicity relation: delta_M = xi * log10(Z/Z_solar)
        # From Timmes+2003: xi ~ 0.08 mag/dex
        self.xi = 0.08  # mag per dex of metallicity

    def metallicity_correction(
        self, z: np.ndarray, xi: float = None, alpha_Z: float = None
    ) -> np.ndarray:
        """
        Compute distance modulus correction due to metallicity evolution.

        Parameters
        ----------
        z : np.ndarray
            Redshifts
        xi : float, optional
            Luminosity-metallicity coefficient (mag/dex)
        alpha_Z : float, optional
            Metallicity evolution slope

        Returns
        -------
        delta_mu : np.ndarray
            Correction to add to observed distance modulus
        """
        if xi is None:
            xi = self.xi
        if alpha_Z is None:
            alpha_Z = self.alpha_Z

        # Metallicity evolution: log10(Z/Z_solar) = -alpha_Z * z
        log_Z_ratio = -alpha_Z * z

        # Luminosity correction: brighter at low Z means LARGER true distance
        # Observed mu is TOO SMALL at high z, so we ADD a positive correction
        delta_mu = xi * (-log_Z_ratio)  # Note: double negative

        return delta_mu

    def correct_distance_modulus(
        self, mu_obs: np.ndarray, z: np.ndarray, xi: float = None
    ) -> np.ndarray:
        """
        Apply metallicity correction to observed distance moduli.

        Parameters
        ----------
        mu_obs : np.ndarray
            Observed distance moduli
        z : np.ndarray
            Redshifts
        xi : float, optional
            Luminosity-metallicity coefficient

        Returns
        -------
        mu_corrected : np.ndarray
            Corrected distance moduli
        """
        delta_mu = self.metallicity_correction(z, xi)
        return mu_obs + delta_mu


class SpandrelFitter:
    """
    Fit Spandrel model to SN Ia data and compare with phantom cosmology.
    """

    def __init__(self, data: PantheonPlusData):
        """
        Initialize fitter with data.

        Parameters
        ----------
        data : PantheonPlusData
            SN Ia dataset
        """
        self.data = data
        self.model = SpandrelModel()
        self.results = {}

    def fit_phantom_cosmology(self) -> Dict:
        """
        Fit w0-wa cosmology to uncorrected data (naive fit).

        Returns
        -------
        results : dict
            Best-fit parameters and statistics
        """

        def chi2_phantom(params):
            """Chi-squared for w0-wa cosmology."""
            w0, wa, M = params

            # Compute model distance moduli
            mu_model = self._compute_mu_model(w0, wa, M)

            # Chi-squared
            residuals = (self.data.mu_observed - mu_model) / self.data.mu_err
            return np.sum(residuals**2)

        # Initial guess
        x0 = [-1.0, 0.0, -19.3]

        # Minimize
        result = minimize(chi2_phantom, x0, method="Nelder-Mead")

        w0_fit, wa_fit, M_fit = result.x
        chi2_val = result.fun
        dof = len(self.data.redshifts) - 3

        self.results["phantom"] = {
            "w0": w0_fit,
            "wa": wa_fit,
            "M": M_fit,
            "chi2": chi2_val,
            "dof": dof,
            "chi2_red": chi2_val / dof,
            "p_value": 1 - chi2.cdf(chi2_val, dof),
        }

        return self.results["phantom"]

    def fit_spandrel_lcdm(self) -> Dict:
        """
        Fit LCDM + Spandrel metallicity correction.

        Returns
        -------
        results : dict
            Best-fit parameters and statistics
        """

        def chi2_spandrel(params):
            """Chi-squared for LCDM with Spandrel correction."""
            xi, M = params

            # Apply metallicity correction
            mu_corrected = self.model.correct_distance_modulus(
                self.data.mu_observed, self.data.redshifts, xi
            )

            # Compare with LCDM
            mu_lcdm = self.data.mu_true_lcdm + M

            # Chi-squared
            residuals = (mu_corrected - mu_lcdm) / self.data.mu_err
            return np.sum(residuals**2)

        # Initial guess
        x0 = [0.08, -19.3]

        # Minimize
        result = minimize(chi2_spandrel, x0, method="Nelder-Mead")

        xi_fit, M_fit = result.x
        chi2_val = result.fun
        dof = len(self.data.redshifts) - 2

        self.results["spandrel"] = {
            "xi": xi_fit,
            "M": M_fit,
            "chi2": chi2_val,
            "dof": dof,
            "chi2_red": chi2_val / dof,
            "p_value": 1 - chi2.cdf(chi2_val, dof),
        }

        return self.results["spandrel"]

    def fit_lcdm_only(self) -> Dict:
        """
        Fit pure LCDM (no correction, no phantom).

        Returns
        -------
        results : dict
            Best-fit parameters and statistics
        """

        def chi2_lcdm(params):
            """Chi-squared for pure LCDM."""
            M = params[0]

            # LCDM model
            mu_lcdm = self.data.mu_true_lcdm + M

            # Chi-squared
            residuals = (self.data.mu_observed - mu_lcdm) / self.data.mu_err
            return np.sum(residuals**2)

        # Minimize
        result = minimize(chi2_lcdm, [-19.3], method="Nelder-Mead")

        M_fit = result.x[0]
        chi2_val = result.fun
        dof = len(self.data.redshifts) - 1

        self.results["lcdm"] = {
            "M": M_fit,
            "chi2": chi2_val,
            "dof": dof,
            "chi2_red": chi2_val / dof,
            "p_value": 1 - chi2.cdf(chi2_val, dof),
        }

        return self.results["lcdm"]

    def _compute_mu_model(self, w0: float, wa: float, M: float) -> np.ndarray:
        """Compute distance modulus for w0-wa model."""
        from scipy.integrate import quad

        c_km_s = 299792.458
        H0 = self.data.H0
        Om = self.data.Om

        def E_z(z_val):
            if z_val == 0:
                return 1.0

            def integrand(zp):
                w = w0 + wa * zp / (1 + zp)
                return (1 + w) / (1 + zp)

            w_int, _ = quad(integrand, 0, z_val)
            rho_de = (1 - Om) * np.exp(3 * w_int)
            return np.sqrt(Om * (1 + z_val) ** 3 + rho_de)

        def d_L(z_val):
            if z_val == 0:
                return 0.0
            d_c, _ = quad(lambda zp: 1.0 / E_z(zp), 0, z_val)
            return (1 + z_val) * (c_km_s / H0) * d_c

        mu = np.array([5 * np.log10(d_L(z)) + 25 + M for z in self.data.redshifts])
        return mu

    def compare_models(self) -> Dict:
        """
        Compare phantom vs Spandrel+LCDM using AIC/BIC.

        Returns
        -------
        comparison : dict
            Model comparison statistics
        """
        if "phantom" not in self.results:
            self.fit_phantom_cosmology()
        if "spandrel" not in self.results:
            self.fit_spandrel_lcdm()
        if "lcdm" not in self.results:
            self.fit_lcdm_only()

        n = len(self.data.redshifts)

        # AIC = chi2 + 2k
        # BIC = chi2 + k*ln(n)

        comparison = {}

        for model_name, res in self.results.items():
            k = 3 if model_name == "phantom" else (2 if model_name == "spandrel" else 1)
            comparison[model_name] = {
                "chi2": res["chi2"],
                "k": k,
                "AIC": res["chi2"] + 2 * k,
                "BIC": res["chi2"] + k * np.log(n),
                "chi2_red": res["chi2_red"],
            }

        # Delta AIC/BIC relative to best model
        min_aic = min(c["AIC"] for c in comparison.values())
        min_bic = min(c["BIC"] for c in comparison.values())

        for model_name in comparison:
            comparison[model_name]["delta_AIC"] = comparison[model_name]["AIC"] - min_aic
            comparison[model_name]["delta_BIC"] = comparison[model_name]["BIC"] - min_bic

        self.comparison = comparison
        return comparison

    def plot_results(self, output_path: str = None):
        """
        Generate diagnostic plots.

        Parameters
        ----------
        output_path : str, optional
            Path to save figure
        """
        if not self.results:
            self.fit_phantom_cosmology()
            self.fit_spandrel_lcdm()
            self.fit_lcdm_only()

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        z = self.data.redshifts
        mu_obs = self.data.mu_observed
        mu_err = self.data.mu_err

        # Panel 1: Hubble diagram
        ax1 = axes[0, 0]
        ax1.errorbar(
            z, mu_obs, yerr=mu_err, fmt=".", alpha=0.3, color="gray", label="Data", markersize=2
        )

        # Overplot models
        z_model = np.linspace(0.01, 2.3, 100)

        ax1.set_xlabel("Redshift z")
        ax1.set_ylabel("Distance modulus $\\mu$")
        ax1.set_title("Hubble Diagram (Simulated Pantheon+-like)")
        ax1.legend()
        ax1.set_xlim(0, 2.5)

        # Panel 2: Residuals vs LCDM
        ax2 = axes[0, 1]

        M_lcdm = self.results["lcdm"]["M"]
        mu_lcdm = self.data.mu_true_lcdm + M_lcdm
        residuals = mu_obs - mu_lcdm

        ax2.errorbar(z, residuals, yerr=mu_err, fmt=".", alpha=0.3, color="gray", markersize=2)
        ax2.axhline(0, color="black", linestyle="--", label="LCDM")

        # Show metallicity correction
        xi_fit = self.results["spandrel"]["xi"]
        delta_mu = self.model.metallicity_correction(z_model, xi_fit)
        ax2.plot(
            z_model,
            -delta_mu,
            "r-",
            linewidth=2,
            label=f"Spandrel correction ($\\xi$={xi_fit:.3f})",
        )

        ax2.set_xlabel("Redshift z")
        ax2.set_ylabel("$\\mu - \\mu_{\\Lambda CDM}$")
        ax2.set_title("Residuals from LCDM")
        ax2.legend()
        ax2.set_ylim(-0.5, 0.5)

        # Panel 3: Model comparison
        ax3 = axes[1, 0]

        comparison = self.compare_models()
        models = list(comparison.keys())
        delta_bic = [comparison[m]["delta_BIC"] for m in models]

        colors = ["blue", "green", "red"]
        bars = ax3.bar(models, delta_bic, color=colors, alpha=0.7)
        ax3.axhline(0, color="black", linestyle="-")
        ax3.axhline(2, color="gray", linestyle="--", alpha=0.5)
        ax3.axhline(6, color="gray", linestyle=":", alpha=0.5)

        ax3.set_ylabel("$\\Delta$BIC")
        ax3.set_title("Model Comparison (lower is better)")
        ax3.text(
            0.95,
            0.95,
            "$\\Delta$BIC > 6: strong evidence\n$\\Delta$BIC > 2: positive evidence",
            transform=ax3.transAxes,
            ha="right",
            va="top",
            fontsize=9,
        )

        # Panel 4: Summary table
        ax4 = axes[1, 1]
        ax4.axis("off")

        table_data = [
            ["Model", "$\\chi^2$", "k", "$\\chi^2$/dof", "$\\Delta$BIC"],
            [
                "LCDM (no corr.)",
                f'{comparison["lcdm"]["chi2"]:.1f}',
                "1",
                f'{comparison["lcdm"]["chi2_red"]:.3f}',
                f'{comparison["lcdm"]["delta_BIC"]:.1f}',
            ],
            [
                "Spandrel+LCDM",
                f'{comparison["spandrel"]["chi2"]:.1f}',
                "2",
                f'{comparison["spandrel"]["chi2_red"]:.3f}',
                f'{comparison["spandrel"]["delta_BIC"]:.1f}',
            ],
            [
                "Phantom ($w_0$-$w_a$)",
                f'{comparison["phantom"]["chi2"]:.1f}',
                "3",
                f'{comparison["phantom"]["chi2_red"]:.3f}',
                f'{comparison["phantom"]["delta_BIC"]:.1f}',
            ],
        ]

        table = ax4.table(
            cellText=table_data,
            loc="center",
            cellLoc="center",
            colWidths=[0.25, 0.15, 0.1, 0.15, 0.15],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

        # Header row styling
        for j in range(5):
            table[(0, j)].set_facecolor("#4472C4")
            table[(0, j)].set_text_props(color="white", fontweight="bold")

        ax4.set_title("Model Comparison Summary", pad=20)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Figure saved to: {output_path}")

        plt.close()

    def save_results(self, output_path: str):
        """Save results to JSON file."""
        results_out = {
            "data_info": {
                "n_sne": self.data.n_sne,
                "z_range": [float(self.data.redshifts.min()), float(self.data.redshifts.max())],
                "note": "Simulated Pantheon+-like dataset for testing",
            },
            "fits": {},
            "comparison": {},
        }

        for name, res in self.results.items():
            results_out["fits"][name] = {k: float(v) for k, v in res.items()}

        if hasattr(self, "comparison"):
            for name, comp in self.comparison.items():
                results_out["comparison"][name] = {k: float(v) for k, v in comp.items()}

        with open(output_path, "w") as f:
            json.dump(results_out, f, indent=2)

        print(f"Results saved to: {output_path}")


def main():
    """Run Spandrel Pantheon+ analysis."""
    print("=" * 70)
    print("SPANDREL FRAMEWORK: Pantheon+ SN Ia Analysis")
    print("Testing metallicity correction vs phantom dark energy")
    print("=" * 70)
    print()

    # Generate simulated data
    print("Generating simulated Pantheon+-like dataset...")
    data = PantheonPlusData(n_sne=1701)
    print(f"  N_SNe: {data.n_sne}")
    print(f"  z range: [{data.redshifts.min():.3f}, {data.redshifts.max():.3f}]")
    print()

    # Initialize fitter
    fitter = SpandrelFitter(data)

    # Fit models
    print("Fitting models...")
    print("-" * 70)

    print("\n1. Pure LCDM (baseline):")
    lcdm_res = fitter.fit_lcdm_only()
    print(f"   chi2/dof = {lcdm_res['chi2_red']:.3f}")

    print("\n2. Phantom cosmology (w0-wa):")
    phantom_res = fitter.fit_phantom_cosmology()
    print(f"   w0 = {phantom_res['w0']:.3f}")
    print(f"   wa = {phantom_res['wa']:.3f}")
    print(f"   chi2/dof = {phantom_res['chi2_red']:.3f}")

    print("\n3. Spandrel + LCDM:")
    spandrel_res = fitter.fit_spandrel_lcdm()
    print(f"   xi = {spandrel_res['xi']:.4f} mag/dex")
    print(f"   chi2/dof = {spandrel_res['chi2_red']:.3f}")

    # Model comparison
    print("\n" + "-" * 70)
    print("MODEL COMPARISON")
    print("-" * 70)

    comparison = fitter.compare_models()

    print(f"\n{'Model':<20} {'chi2/dof':<12} {'AIC':<12} {'BIC':<12} {'dBIC':<10}")
    print("-" * 70)
    for name in ["lcdm", "spandrel", "phantom"]:
        c = comparison[name]
        print(
            f"{name:<20} {c['chi2_red']:<12.3f} {c['AIC']:<12.1f} "
            f"{c['BIC']:<12.1f} {c['delta_BIC']:<10.1f}"
        )

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    best_model = min(comparison, key=lambda x: comparison[x]["BIC"])
    print(f"\nBest model by BIC: {best_model}")

    if comparison["spandrel"]["delta_BIC"] < comparison["phantom"]["delta_BIC"]:
        print("\nSpandrel+LCDM is preferred over phantom cosmology.")
        print("This suggests the 'phantom' signal may be a metallicity artifact.")
    else:
        print("\nPhantom cosmology is preferred over Spandrel+LCDM.")
        print("The apparent w < -1 signal may be real, not a systematic.")

    print("\n" + "=" * 70)
    print("CAVEATS")
    print("=" * 70)
    print(
        """
1. This analysis uses SIMULATED data based on Pantheon+ statistics
2. Real analysis requires actual Pantheon+ data access
3. The metallicity evolution model is simplified
4. Host galaxy mass corrections are not included
5. Selection effects are not modeled
    """
    )

    # Save outputs
    print("\nSaving outputs...")
    fitter.plot_results(str(OUTPUT_DIR / "spandrel_pantheon_fit.pdf"))
    fitter.save_results(str(OUTPUT_DIR / "spandrel_pantheon_results.json"))

    print("\nAnalysis complete!")
    print(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
