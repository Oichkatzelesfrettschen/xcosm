#!/usr/bin/env python3
"""
Rigorous H₀ Covariance Analysis
================================

Performs proper statistical analysis of H₀ measurements including:
1. Full covariance matrix accounting for systematic correlations
2. χ² fit for constant H₀ (null hypothesis)
3. χ² fit for H₀ gradient model: H₀(k) = H₀_0 + m·log₁₀(k)
4. Proper significance testing with degrees of freedom
5. Visualization of covariance structure and fit results

References:
- Verde et al. (2019) "Tensions between the Early and Late Universe"
- Handley & Lemos (2019) "Quantifying tensions in cosmological parameters"
- Planck Collaboration VI (2020) for CMB systematics
- Riess et al. (2022) SH0ES for distance ladder correlations
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


@dataclass
class H0Measurement:
    """Single H₀ measurement with metadata."""

    name: str
    method: str
    value: float  # km/s/Mpc
    error: float  # km/s/Mpc (1-sigma)
    redshift_regime: str  # 'early', 'intermediate', 'late'
    scale: float  # characteristic k scale in h/Mpc


class H0CovarianceAnalysis:
    """
    Rigorous statistical analysis of H₀ measurements with full covariance.

    The covariance matrix accounts for:
    1. Statistical uncertainties (diagonal)
    2. Systematic correlations within method classes
    3. Cosmological model dependencies
    4. Shared calibration sources
    """

    def __init__(self):
        """Initialize measurements and construct covariance matrix."""
        self.measurements = self._define_measurements()
        self.num_measurements = len(self.measurements)
        self.covariance_matrix = self._construct_covariance_matrix()
        self.inverse_covariance = np.linalg.inv(self.covariance_matrix)

    def _define_measurements(self) -> List[H0Measurement]:
        """
        Define the 15 H₀ measurements with proper categorization.

        Redshift regimes:
        - early: CMB (z ~ 1100)
        - intermediate: BAO, WL (0.1 < z < 3)
        - late: Distance ladder (z < 0.01)

        Scale assignments based on characteristic modes:
        - CMB: k ~ 0.01 h/Mpc (acoustic horizon)
        - BAO: k ~ 0.05-0.1 h/Mpc (sound horizon)
        - WL: k ~ 0.1-1 h/Mpc (nonlinear scales)
        - Distance ladder: k ~ 10 h/Mpc (local)
        """
        return [
            # Early universe (CMB)
            H0Measurement(
                name="Planck CMB",
                method="CMB",
                value=67.4,
                error=0.5,
                redshift_regime="early",
                scale=0.01,
            ),
            H0Measurement(
                name="ACT DR6",
                method="CMB",
                value=67.9,
                error=1.5,
                redshift_regime="early",
                scale=0.01,
            ),
            # Intermediate universe (BAO)
            H0Measurement(
                name="BOSS BAO",
                method="BAO",
                value=67.6,
                error=0.5,
                redshift_regime="intermediate",
                scale=0.075,
            ),
            H0Measurement(
                name="eBOSS",
                method="BAO",
                value=68.2,
                error=0.8,
                redshift_regime="intermediate",
                scale=0.08,
            ),
            # Intermediate universe (Weak Lensing)
            H0Measurement(
                name="DES Y3 WL",
                method="WL",
                value=68.2,
                error=1.5,
                redshift_regime="intermediate",
                scale=0.3,
            ),
            H0Measurement(
                name="DES Y3 combined",
                method="WL",
                value=68.0,
                error=0.9,
                redshift_regime="intermediate",
                scale=0.3,
            ),
            H0Measurement(
                name="KiDS",
                method="WL",
                value=67.5,
                error=2.0,
                redshift_regime="intermediate",
                scale=0.3,
            ),
            # Late universe (Distance ladder - TRGB)
            H0Measurement(
                name="TRGB", method="TRGB", value=69.8, error=1.7, redshift_regime="late", scale=5.0
            ),
            # Late universe (Geometric)
            H0Measurement(
                name="Megamasers",
                method="Geometric",
                value=73.9,
                error=3.0,
                redshift_regime="late",
                scale=10.0,
            ),
            H0Measurement(
                name="SBF", method="SBF", value=70.5, error=2.4, redshift_regime="late", scale=5.0
            ),
            # Late universe (Lensing Time Delay)
            H0Measurement(
                name="Lensing TD",
                method="LensingTD",
                value=73.3,
                error=1.8,
                redshift_regime="late",
                scale=1.0,
            ),
            # Late universe (Cepheid calibrated)
            H0Measurement(
                name="SH0ES",
                method="Cepheid",
                value=73.04,
                error=1.04,
                redshift_regime="late",
                scale=10.0,
            ),
        ]

    def _construct_covariance_matrix(self) -> np.ndarray:
        """
        Construct full covariance matrix with systematic correlations.

        Correlation structure:
        1. CMB experiments: ρ = 0.4 (shared foreground/systematics)
        2. BAO experiments: ρ = 0.3 (shared reconstruction methods)
        3. DES measurements: ρ = 0.6 (same survey, overlapping data)
        4. Distance ladder (Cepheid-based): ρ = 0.5 (shared calibration)
        5. Cross-method (same regime): ρ = 0.1 (cosmological model)

        References:
        - Planck/ACT correlation: Aiola et al. (2020) ACT DR4
        - DES internal: Abbott et al. (2022) DES Y3
        - Distance ladder: Riess et al. (2022) Section 8
        """
        covariance_matrix = np.zeros((self.num_measurements, self.num_measurements))

        # Start with diagonal (statistical uncertainties)
        for i, measurement in enumerate(self.measurements):
            covariance_matrix[i, i] = measurement.error**2

        # Add off-diagonal correlations
        for i in range(self.num_measurements):
            for j in range(i + 1, self.num_measurements):
                measurement_i = self.measurements[i]
                measurement_j = self.measurements[j]

                correlation_coefficient = self._get_correlation(measurement_i, measurement_j)

                covariance_matrix[i, j] = (
                    correlation_coefficient * measurement_i.error * measurement_j.error
                )
                covariance_matrix[j, i] = covariance_matrix[i, j]  # Symmetric

        return covariance_matrix

    def _get_correlation(self, measurement_i: H0Measurement, measurement_j: H0Measurement) -> float:
        """
        Determine correlation coefficient between two measurements.

        Correlation hierarchy:
        1. Same experiment/survey: 0.6-0.8
        2. Same method class: 0.3-0.5
        3. Same redshift regime: 0.05-0.15
        4. Different regimes: 0.0
        """
        # CMB experiments (shared foreground models, calibration)
        if measurement_i.method == "CMB" and measurement_j.method == "CMB":
            return 0.4

        # BAO experiments (shared fiber collision corrections, reconstruction)
        if measurement_i.method == "BAO" and measurement_j.method == "BAO":
            return 0.3

        # DES internal correlations (same survey)
        if "DES" in measurement_i.name and "DES" in measurement_j.name:
            return 0.6

        # Cepheid-calibrated distance ladder (shared LMC/MW Cepheid calibration)
        cepheid_methods = {"Cepheid", "TRGB", "SBF"}
        if measurement_i.method in cepheid_methods and measurement_j.method in cepheid_methods:
            return 0.5

        # Same redshift regime (shared cosmological model assumptions)
        if measurement_i.redshift_regime == measurement_j.redshift_regime:
            if measurement_i.redshift_regime == "early":
                return 0.1  # CMB-CMB already handled
            elif measurement_i.redshift_regime == "intermediate":
                return 0.1
            elif measurement_i.redshift_regime == "late":
                return 0.15  # Late universe more correlated (local environment)

        # Different regimes: no correlation
        return 0.0

    def fit_constant_h0(self) -> Tuple[float, float, float, int]:
        """
        Fit constant H₀ model (null hypothesis).

        Returns:
            h0_best: Best-fit H₀ value
            h0_error: 1-sigma error on H₀
            chi2: χ² value
            dof: Degrees of freedom
        """
        # Best-fit H₀ is weighted mean with inverse covariance weighting
        values = np.array([m.value for m in self.measurements])

        # H₀_best = (1^T C^{-1} 1)^{-1} (1^T C^{-1} H)
        ones = np.ones(self.num_measurements)
        denominator = ones @ self.inverse_covariance @ ones
        h0_best = (ones @ self.inverse_covariance @ values) / denominator
        h0_error = np.sqrt(1.0 / denominator)

        # χ² = (H - H₀_best)^T C^{-1} (H - H₀_best)
        residuals = values - h0_best
        chi2 = residuals @ self.inverse_covariance @ residuals

        # Degrees of freedom: N_measurements - N_parameters
        dof = self.num_measurements - 1

        return h0_best, h0_error, chi2, dof

    def fit_gradient_model(self) -> Tuple[float, float, float, float, float, int]:
        """
        Fit gradient model: H₀(k) = H₀_0 + m·log₁₀(k/(h/Mpc))

        This is the alternative hypothesis testing for scale-dependent H₀.

        Returns:
            h0_0: Intercept (H₀ at k = 1 h/Mpc)
            h0_0_error: Error on intercept
            slope: Gradient m
            slope_error: Error on slope
            chi2: χ² value
            dof: Degrees of freedom
        """
        values = np.array([m.value for m in self.measurements])
        log_scales = np.array([np.log10(m.scale) for m in self.measurements])

        # Design matrix: [1, log₁₀(k)]
        design_matrix = np.column_stack([np.ones(self.num_measurements), log_scales])

        # Generalized least squares: β = (X^T C^{-1} X)^{-1} X^T C^{-1} y
        xt_cinv = design_matrix.T @ self.inverse_covariance
        xt_cinv_x = xt_cinv @ design_matrix
        parameter_covariance = np.linalg.inv(xt_cinv_x)
        parameters = parameter_covariance @ xt_cinv @ values

        h0_0 = parameters[0]
        slope = parameters[1]
        h0_0_error = np.sqrt(parameter_covariance[0, 0])
        slope_error = np.sqrt(parameter_covariance[1, 1])

        # χ²
        model_predictions = design_matrix @ parameters
        residuals = values - model_predictions
        chi2 = residuals @ self.inverse_covariance @ residuals

        # Degrees of freedom: N_measurements - N_parameters
        dof = self.num_measurements - 2

        return h0_0, h0_0_error, slope, slope_error, chi2, dof

    def calculate_tension_significance(
        self, chi2_null: float, chi2_alt: float, dof_null: int, dof_alt: int
    ) -> Dict[str, float]:
        """
        Calculate statistical significance of tension between models.

        Uses likelihood ratio test:
        Δχ² = χ²_null - χ²_alt ~ χ²(Δdof) under null hypothesis

        Args:
            chi2_null: χ² for null hypothesis (constant H₀)
            chi2_alt: χ² for alternative hypothesis (gradient)
            dof_null: Degrees of freedom for null
            dof_alt: Degrees of freedom for alternative

        Returns:
            Dictionary with significance metrics
        """
        delta_chi2 = chi2_null - chi2_alt
        delta_dof = dof_null - dof_alt

        # p-value from χ² distribution
        p_value = 1.0 - stats.chi2.cdf(delta_chi2, delta_dof)

        # Convert to sigma (one-sided Gaussian equivalent)
        if p_value > 0 and p_value < 1:
            significance_sigma = stats.norm.ppf(1 - p_value)
        else:
            significance_sigma = 0.0

        # AIC and BIC for model comparison
        num_data = len(self.measurements)
        aic_null = chi2_null + 2 * 1  # 1 parameter
        aic_alt = chi2_alt + 2 * 2  # 2 parameters
        bic_null = chi2_null + np.log(num_data) * 1
        bic_alt = chi2_alt + np.log(num_data) * 2

        return {
            "delta_chi2": delta_chi2,
            "delta_dof": delta_dof,
            "p_value": p_value,
            "significance_sigma": significance_sigma,
            "aic_null": aic_null,
            "aic_alt": aic_alt,
            "delta_aic": aic_null - aic_alt,
            "bic_null": bic_null,
            "bic_alt": bic_alt,
            "delta_bic": bic_null - bic_alt,
        }

    def run_full_analysis(self, output_dir: Optional[Union[str, Path]] = None):
        """
        Run complete analysis and generate outputs.

        Args:
            output_dir: Directory for saving results
        """
        from xcosm_common.paths import default_output_dir

        output_path = default_output_dir() if output_dir is None else Path(output_dir)
        output_path = output_path.expanduser().resolve()
        output_path.mkdir(parents=True, exist_ok=True)

        print("=" * 80)
        print("H₀ COVARIANCE ANALYSIS")
        print("=" * 80)
        print(f"\nAnalyzing {self.num_measurements} H₀ measurements")
        print(f"Covariance matrix dimension: {self.covariance_matrix.shape}")

        # Fit null hypothesis (constant H₀)
        print("\n" + "-" * 80)
        print("NULL HYPOTHESIS: H₀ = constant")
        print("-" * 80)
        h0_const, h0_const_err, chi2_null, dof_null = self.fit_constant_h0()
        p_value_null = 1.0 - stats.chi2.cdf(chi2_null, dof_null)

        print(f"Best-fit H₀: {h0_const:.2f} ± {h0_const_err:.2f} km/s/Mpc")
        print(f"χ² = {chi2_null:.2f}")
        print(f"DOF = {dof_null}")
        print(f"Reduced χ² = {chi2_null/dof_null:.2f}")
        print(f"p-value = {p_value_null:.4e}")
        print(f"Goodness of fit: {self._interpret_gof(chi2_null, dof_null)}")

        # Fit alternative hypothesis (gradient model)
        print("\n" + "-" * 80)
        print("ALTERNATIVE HYPOTHESIS: H₀(k) = H₀_0 + m·log₁₀(k)")
        print("-" * 80)
        h0_0, h0_0_err, slope, slope_err, chi2_alt, dof_alt = self.fit_gradient_model()
        p_value_alt = 1.0 - stats.chi2.cdf(chi2_alt, dof_alt)

        print(f"H₀_0 (at k=1 h/Mpc): {h0_0:.2f} ± {h0_0_err:.2f} km/s/Mpc")
        print(f"Gradient m: {slope:.2f} ± {slope_err:.2f} km/s/Mpc/dex")
        print(f"χ² = {chi2_alt:.2f}")
        print(f"DOF = {dof_alt}")
        print(f"Reduced χ² = {chi2_alt/dof_alt:.2f}")
        print(f"p-value = {p_value_alt:.4e}")
        print(f"Goodness of fit: {self._interpret_gof(chi2_alt, dof_alt)}")

        # Model comparison
        print("\n" + "-" * 80)
        print("MODEL COMPARISON")
        print("-" * 80)
        significance = self.calculate_tension_significance(chi2_null, chi2_alt, dof_null, dof_alt)

        print(f"Δχ² = {significance['delta_chi2']:.2f}")
        print(f"ΔDOF = {significance['delta_dof']}")
        print(f"p-value (gradient improves fit): {significance['p_value']:.4e}")
        print(f"Significance: {significance['significance_sigma']:.2f}σ")
        print(f"\nΔAIC = {significance['delta_aic']:.2f} (negative favors gradient)")
        print(f"ΔBIC = {significance['delta_bic']:.2f} (negative favors gradient)")
        print(f"\nInterpretation: {self._interpret_model_comparison(significance)}")

        # Print measurement details
        print("\n" + "-" * 80)
        print("MEASUREMENTS")
        print("-" * 80)
        print(
            f"{'Name':<20} {'Value':>8} {'Error':>6} {'Method':<12} {'Regime':<12} {'log₁₀(k)':<8}"
        )
        print("-" * 80)
        for m in self.measurements:
            print(
                f"{m.name:<20} {m.value:>8.2f} {m.error:>6.2f} "
                f"{m.method:<12} {m.redshift_regime:<12} {np.log10(m.scale):>8.2f}"
            )

        # Save results
        self._save_results(
            h0_const,
            h0_const_err,
            chi2_null,
            dof_null,
            h0_0,
            h0_0_err,
            slope,
            slope_err,
            chi2_alt,
            dof_alt,
            significance,
            output_path,
        )

        # Generate visualizations
        self._plot_covariance_matrix(output_path)
        self._plot_correlation_matrix(output_path)
        self._plot_fit_results(h0_const, h0_0, slope, output_path)
        self._plot_residuals(h0_const, h0_0, slope, output_path)

        print(f"\nResults saved to: {output_path}")
        print("=" * 80)

    def _interpret_gof(self, chi2: float, dof: int) -> str:
        """Interpret goodness of fit."""
        p_value = 1.0 - stats.chi2.cdf(chi2, dof)
        reduced_chi2 = chi2 / dof

        if reduced_chi2 < 0.5:
            return "Overfit (reduced χ² too small)"
        elif reduced_chi2 < 1.5 and p_value > 0.05:
            return "Good fit"
        elif reduced_chi2 < 2.0 and p_value > 0.01:
            return "Acceptable fit"
        elif p_value > 0.001:
            return "Marginal fit"
        else:
            return "Poor fit - significant tension"

    def _interpret_model_comparison(self, significance: Dict[str, float]) -> str:
        """Interpret model comparison results."""
        delta_bic = significance["delta_bic"]
        sigma = significance["significance_sigma"]

        interpretation_parts = []

        # BIC interpretation (Kass & Raftery 1995)
        if delta_bic < -10:
            interpretation_parts.append("Very strong evidence for gradient (ΔBIC < -10)")
        elif delta_bic < -6:
            interpretation_parts.append("Strong evidence for gradient (ΔBIC < -6)")
        elif delta_bic < -2:
            interpretation_parts.append("Positive evidence for gradient (ΔBIC < -2)")
        elif delta_bic < 2:
            interpretation_parts.append("Inconclusive (|ΔBIC| < 2)")
        elif delta_bic < 6:
            interpretation_parts.append("Positive evidence against gradient")
        else:
            interpretation_parts.append("Strong evidence against gradient")

        # Sigma interpretation
        if sigma > 5:
            interpretation_parts.append(f"{sigma:.1f}σ detection of gradient")
        elif sigma > 3:
            interpretation_parts.append(f"{sigma:.1f}σ evidence for gradient")
        elif sigma > 2:
            interpretation_parts.append(f"{sigma:.1f}σ hint of gradient")
        else:
            interpretation_parts.append(f"No significant gradient ({sigma:.1f}σ)")

        return "; ".join(interpretation_parts)

    def _save_results(
        self,
        h0_const: float,
        h0_const_err: float,
        chi2_null: float,
        dof_null: int,
        h0_0: float,
        h0_0_err: float,
        slope: float,
        slope_err: float,
        chi2_alt: float,
        dof_alt: int,
        significance: Dict[str, float],
        output_path: Path,
    ):
        """Save numerical results to JSON."""
        results = {
            "measurements": [
                {
                    "name": m.name,
                    "method": m.method,
                    "value": m.value,
                    "error": m.error,
                    "redshift_regime": m.redshift_regime,
                    "scale_h_per_mpc": m.scale,
                    "log10_scale": np.log10(m.scale),
                }
                for m in self.measurements
            ],
            "null_hypothesis": {
                "model": "H0 = constant",
                "h0": h0_const,
                "h0_error": h0_const_err,
                "chi2": chi2_null,
                "dof": dof_null,
                "reduced_chi2": chi2_null / dof_null,
                "p_value": float(1.0 - stats.chi2.cdf(chi2_null, dof_null)),
            },
            "alternative_hypothesis": {
                "model": "H0(k) = H0_0 + m*log10(k)",
                "h0_0": h0_0,
                "h0_0_error": h0_0_err,
                "gradient_m": slope,
                "gradient_m_error": slope_err,
                "chi2": chi2_alt,
                "dof": dof_alt,
                "reduced_chi2": chi2_alt / dof_alt,
                "p_value": float(1.0 - stats.chi2.cdf(chi2_alt, dof_alt)),
            },
            "model_comparison": {
                "delta_chi2": significance["delta_chi2"],
                "delta_dof": significance["delta_dof"],
                "p_value": significance["p_value"],
                "significance_sigma": significance["significance_sigma"],
                "delta_aic": significance["delta_aic"],
                "delta_bic": significance["delta_bic"],
                "interpretation": self._interpret_model_comparison(significance),
            },
            "covariance_matrix": self.covariance_matrix.tolist(),
            "correlation_matrix": self._get_correlation_matrix().tolist(),
        }

        with open(output_path / "h0_covariance_results.json", "w") as f:
            json.dump(results, f, indent=2)

    def _get_correlation_matrix(self) -> np.ndarray:
        """Convert covariance to correlation matrix."""
        std_devs = np.sqrt(np.diag(self.covariance_matrix))
        correlation = self.covariance_matrix / np.outer(std_devs, std_devs)
        return correlation

    def _plot_covariance_matrix(self, output_path: Path):
        """Visualize covariance matrix."""
        fig, ax = plt.subplots(figsize=(12, 10))

        labels = [m.name for m in self.measurements]

        im = ax.imshow(self.covariance_matrix, cmap="RdBu_r", aspect="auto")

        ax.set_xticks(range(self.num_measurements))
        ax.set_yticks(range(self.num_measurements))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(labels, fontsize=9)

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Covariance [(km/s/Mpc)²]", fontsize=11)

        ax.set_title("H₀ Measurement Covariance Matrix", fontsize=13, fontweight="bold")

        plt.tight_layout()
        plt.savefig(output_path / "h0_covariance_matrix.png", dpi=300, bbox_inches="tight")
        plt.savefig(output_path / "h0_covariance_matrix.pdf", bbox_inches="tight")
        plt.close()

    def _plot_correlation_matrix(self, output_path: Path):
        """Visualize correlation matrix."""
        fig, ax = plt.subplots(figsize=(12, 10))

        correlation = self._get_correlation_matrix()
        labels = [m.name for m in self.measurements]

        im = ax.imshow(correlation, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)

        ax.set_xticks(range(self.num_measurements))
        ax.set_yticks(range(self.num_measurements))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(labels, fontsize=9)

        # Add correlation values
        for i in range(self.num_measurements):
            for j in range(self.num_measurements):
                if i != j and abs(correlation[i, j]) > 0.1:
                    text = ax.text(
                        j,
                        i,
                        f"{correlation[i, j]:.2f}",
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=7,
                    )

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Correlation Coefficient", fontsize=11)

        ax.set_title("H₀ Measurement Correlation Matrix", fontsize=13, fontweight="bold")

        plt.tight_layout()
        plt.savefig(output_path / "h0_correlation_matrix.png", dpi=300, bbox_inches="tight")
        plt.savefig(output_path / "h0_correlation_matrix.pdf", bbox_inches="tight")
        plt.close()

    def _plot_fit_results(self, h0_const: float, h0_0: float, slope: float, output_path: Path):
        """Plot measurements with both model fits."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        values = np.array([m.value for m in self.measurements])
        errors = np.array([m.error for m in self.measurements])
        log_scales = np.array([np.log10(m.scale) for m in self.measurements])

        # Color by regime
        regime_colors = {"early": "blue", "intermediate": "green", "late": "red"}
        colors = [regime_colors[m.redshift_regime] for m in self.measurements]

        # Upper panel: Measurements and constant fit
        ax1.errorbar(
            log_scales,
            values,
            yerr=errors,
            fmt="o",
            capsize=5,
            capthick=2,
            markersize=8,
            alpha=0.7,
            ecolor="gray",
            markerfacecolor="none",
            markeredgewidth=2,
            label="Measurements",
        )

        for i, (log_k, val, err, color) in enumerate(zip(log_scales, values, errors, colors)):
            ax1.scatter(log_k, val, c=color, s=100, alpha=0.7, edgecolors="black", linewidths=1)

        # Constant H₀ line
        ax1.axhline(
            h0_const,
            color="black",
            linestyle="--",
            linewidth=2,
            label=f"Constant: H₀ = {h0_const:.2f} km/s/Mpc",
        )
        ax1.fill_between(
            ax1.get_xlim(),
            h0_const - 2,
            h0_const + 2,
            alpha=0.2,
            color="gray",
            label="±2 km/s/Mpc band",
        )

        ax1.set_ylabel("H₀ [km/s/Mpc]", fontsize=12)
        ax1.set_title("H₀ Measurements vs Scale", fontsize=13, fontweight="bold")
        ax1.legend(loc="best", fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Lower panel: Measurements and gradient fit
        ax2.errorbar(
            log_scales,
            values,
            yerr=errors,
            fmt="o",
            capsize=5,
            capthick=2,
            markersize=8,
            alpha=0.7,
            ecolor="gray",
            markerfacecolor="none",
            markeredgewidth=2,
        )

        for i, (log_k, val, err, color) in enumerate(zip(log_scales, values, errors, colors)):
            ax2.scatter(log_k, val, c=color, s=100, alpha=0.7, edgecolors="black", linewidths=1)

        # Gradient model line
        log_k_model = np.linspace(log_scales.min() - 0.5, log_scales.max() + 0.5, 100)
        h0_model = h0_0 + slope * log_k_model
        ax2.plot(
            log_k_model,
            h0_model,
            "r-",
            linewidth=2,
            label=f"Gradient: H₀ = {h0_0:.2f} + {slope:.2f}×log₁₀(k)",
        )

        ax2.set_xlabel("log₁₀(k [h/Mpc])", fontsize=12)
        ax2.set_ylabel("H₀ [km/s/Mpc]", fontsize=12)
        ax2.legend(loc="best", fontsize=10)
        ax2.grid(True, alpha=0.3)

        # Add regime labels
        regime_patches = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color,
                markersize=10,
                label=regime.capitalize(),
            )
            for regime, color in regime_colors.items()
        ]
        fig.legend(
            handles=regime_patches,
            loc="upper right",
            bbox_to_anchor=(0.98, 0.98),
            title="Redshift Regime",
            fontsize=10,
        )

        plt.tight_layout()
        plt.savefig(output_path / "h0_fit_results.png", dpi=300, bbox_inches="tight")
        plt.savefig(output_path / "h0_fit_results.pdf", bbox_inches="tight")
        plt.close()

    def _plot_residuals(self, h0_const: float, h0_0: float, slope: float, output_path: Path):
        """Plot residuals for both models."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        values = np.array([m.value for m in self.measurements])
        errors = np.array([m.error for m in self.measurements])
        log_scales = np.array([np.log10(m.scale) for m in self.measurements])
        labels = [m.name for m in self.measurements]

        # Constant model residuals
        residuals_const = values - h0_const
        pull_const = residuals_const / errors

        x_positions = np.arange(self.num_measurements)
        ax1.errorbar(
            x_positions,
            residuals_const,
            yerr=errors,
            fmt="o",
            capsize=5,
            markersize=8,
            color="blue",
            alpha=0.7,
        )
        ax1.axhline(0, color="black", linestyle="--", linewidth=2)
        ax1.fill_between(x_positions, -2, 2, alpha=0.2, color="gray", label="±2 km/s/Mpc")
        ax1.set_xticks(x_positions)
        ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax1.set_ylabel("Residual [km/s/Mpc]", fontsize=11)
        ax1.set_title(
            f"Constant Model Residuals\nχ²/dof = {np.sum(pull_const**2):.1f}/{len(pull_const)-1}",
            fontsize=12,
            fontweight="bold",
        )
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=9)

        # Gradient model residuals
        h0_gradient_predictions = h0_0 + slope * log_scales
        residuals_gradient = values - h0_gradient_predictions
        pull_gradient = residuals_gradient / errors

        ax2.errorbar(
            x_positions,
            residuals_gradient,
            yerr=errors,
            fmt="o",
            capsize=5,
            markersize=8,
            color="red",
            alpha=0.7,
        )
        ax2.axhline(0, color="black", linestyle="--", linewidth=2)
        ax2.fill_between(x_positions, -2, 2, alpha=0.2, color="gray", label="±2 km/s/Mpc")
        ax2.set_xticks(x_positions)
        ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax2.set_ylabel("Residual [km/s/Mpc]", fontsize=11)
        ax2.set_title(
            f"Gradient Model Residuals\nχ²/dof = {np.sum(pull_gradient**2):.1f}/{len(pull_gradient)-2}",
            fontsize=12,
            fontweight="bold",
        )
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=9)

        plt.tight_layout()
        plt.savefig(output_path / "h0_residuals.png", dpi=300, bbox_inches="tight")
        plt.savefig(output_path / "h0_residuals.pdf", bbox_inches="tight")
        plt.close()


def main():
    """Run full H₀ covariance analysis."""
    analysis = H0CovarianceAnalysis()
    analysis.run_full_analysis()


if __name__ == "__main__":
    main()
