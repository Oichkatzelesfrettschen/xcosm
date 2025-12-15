"""
Analysis Module
===============

Statistical analysis tools for H0 gradient and cosmological observables.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np
from scipy import stats
from scipy.optimize import minimize


@dataclass
class H0Measurement:
    """
    Represents a single H0 measurement.

    Attributes
    ----------
    name : str
        Dataset/experiment name
    h0 : float
        Hubble constant value (km/s/Mpc)
    sigma : float
        1-sigma uncertainty
    k_eff : float
        Effective wavenumber probed (Mpc^-1)
    z_range : Tuple[float, float]
        Redshift range
    method : str
        Measurement method
    """

    name: str
    h0: float
    sigma: float
    k_eff: float
    z_range: Tuple[float, float] = (0.0, 0.0)
    method: str = ""


class H0GradientAnalysis:
    """
    Analyze scale-dependence of H0 measurements.

    Fits the model: H0(k) = H0_0 + m * log10(k)
    and tests for significant gradient.
    """

    def __init__(self, measurements: Optional[List[H0Measurement]] = None):
        self.measurements = measurements or self._default_measurements()
        self._fit_result = None

    def _default_measurements(self) -> List[H0Measurement]:
        """Return default compilation of H0 measurements."""
        return [
            H0Measurement("Planck 2018", 67.36, 0.54, 2e-4, (1100, 1100), "Primary CMB"),
            H0Measurement("ACT DR6", 67.9, 1.1, 5e-4, (0.5, 5), "CMB lensing"),
            H0Measurement("DESI DR1", 68.52, 0.62, 0.01, (0.1, 2.1), "BAO"),
            H0Measurement("DESI DR2", 68.7, 0.55, 0.015, (0.1, 4.2), "BAO+RSD"),
            H0Measurement("DES Y3", 69.1, 1.2, 0.05, (0.15, 1.0), "Weak lensing"),
            H0Measurement("KiDS-1000", 69.5, 1.8, 0.08, (0.1, 1.2), "Cosmic shear"),
            H0Measurement("TRGB", 69.8, 1.7, 0.1, (0, 0.01), "TRGB"),
            H0Measurement("CCHP 2024", 69.96, 1.05, 0.15, (0, 0.02), "TRGB+JAGB"),
            H0Measurement("H0LiCOW", 71.8, 2.0, 0.25, (0.3, 1.0), "Time delay"),
            H0Measurement("Megamaser", 73.0, 2.5, 0.3, (0, 0.05), "Megamaser"),
            H0Measurement("SH0ES 2024", 73.17, 0.86, 0.5, (0, 0.15), "Cepheid-SN"),
        ]

    def fit_gradient(self) -> Dict[str, float]:
        """
        Fit linear gradient model in log(k).

        Returns
        -------
        Dict[str, float]
            Fit results including intercept, slope, and uncertainties
        """
        log_k = np.array([np.log10(m.k_eff) for m in self.measurements])
        h0 = np.array([m.h0 for m in self.measurements])
        sigma = np.array([m.sigma for m in self.measurements])

        weights = 1.0 / sigma**2

        def chi_squared(params):
            intercept, slope = params
            model = intercept + slope * log_k
            return np.sum(weights * (h0 - model)**2)

        result = minimize(chi_squared, [70.0, 1.0], method='Nelder-Mead')
        intercept, slope = result.x

        residuals = h0 - (intercept + slope * log_k)
        chi2 = np.sum(weights * residuals**2)
        dof = len(self.measurements) - 2

        S_ww = np.sum(weights)
        S_wx = np.sum(weights * log_k)
        S_wxx = np.sum(weights * log_k**2)
        delta = S_ww * S_wxx - S_wx**2

        sigma_intercept = np.sqrt(S_wxx / delta)
        sigma_slope = np.sqrt(S_ww / delta)

        self._fit_result = {
            "intercept": intercept,
            "slope": slope,
            "sigma_intercept": sigma_intercept,
            "sigma_slope": sigma_slope,
            "chi2": chi2,
            "dof": dof,
            "chi2_reduced": chi2 / dof,
            "significance": slope / sigma_slope
        }

        return self._fit_result

    def test_flat_model(self) -> Dict[str, float]:
        """
        Test flat (no gradient) model.

        Returns
        -------
        Dict[str, float]
            Flat model fit and comparison statistics
        """
        h0 = np.array([m.h0 for m in self.measurements])
        sigma = np.array([m.sigma for m in self.measurements])
        weights = 1.0 / sigma**2

        h0_flat = np.sum(weights * h0) / np.sum(weights)
        sigma_flat = 1.0 / np.sqrt(np.sum(weights))

        residuals = h0 - h0_flat
        chi2_flat = np.sum(weights * residuals**2)
        dof_flat = len(self.measurements) - 1

        if self._fit_result is None:
            self.fit_gradient()

        delta_chi2 = chi2_flat - self._fit_result["chi2"]

        f_stat = delta_chi2 / self._fit_result["chi2"] * self._fit_result["dof"]
        p_value = 1 - stats.f.cdf(f_stat, 1, self._fit_result["dof"])

        return {
            "h0_flat": h0_flat,
            "sigma_flat": sigma_flat,
            "chi2_flat": chi2_flat,
            "dof_flat": dof_flat,
            "delta_chi2": delta_chi2,
            "f_statistic": f_stat,
            "p_value": p_value
        }

    def ccf_comparison(
        self,
        ccf_intercept: float = 71.87,
        ccf_slope: float = 1.15
    ) -> Dict[str, float]:
        """
        Compare fit to CCF predictions.

        Parameters
        ----------
        ccf_intercept : float
            CCF predicted intercept
        ccf_slope : float
            CCF predicted slope (km/s/Mpc per decade)

        Returns
        -------
        Dict[str, float]
            Comparison statistics
        """
        if self._fit_result is None:
            self.fit_gradient()

        intercept_diff = abs(self._fit_result["intercept"] - ccf_intercept)
        slope_diff = abs(self._fit_result["slope"] - ccf_slope)

        intercept_tension = intercept_diff / self._fit_result["sigma_intercept"]
        slope_tension = slope_diff / self._fit_result["sigma_slope"]

        combined_tension = np.sqrt(intercept_tension**2 + slope_tension**2)

        return {
            "ccf_intercept": ccf_intercept,
            "ccf_slope": ccf_slope,
            "fitted_intercept": self._fit_result["intercept"],
            "fitted_slope": self._fit_result["slope"],
            "intercept_tension_sigma": intercept_tension,
            "slope_tension_sigma": slope_tension,
            "combined_tension_sigma": combined_tension,
            "ccf_consistent": combined_tension < 2.0
        }

    def summary(self) -> str:
        """Generate summary report."""
        if self._fit_result is None:
            self.fit_gradient()

        flat = self.test_flat_model()
        ccf = self.ccf_comparison()

        lines = [
            "=" * 60,
            "H0 GRADIENT ANALYSIS SUMMARY",
            "=" * 60,
            f"",
            f"Data: {len(self.measurements)} measurements",
            f"Scale range: {np.log10(min(m.k_eff for m in self.measurements)):.1f} to "
            f"{np.log10(max(m.k_eff for m in self.measurements)):.1f} (log10 k)",
            f"",
            f"Gradient Fit:",
            f"  H0(k) = ({self._fit_result['intercept']:.2f} +/- "
            f"{self._fit_result['sigma_intercept']:.2f}) + "
            f"({self._fit_result['slope']:.2f} +/- "
            f"{self._fit_result['sigma_slope']:.2f}) * log10(k)",
            f"  chi2/dof = {self._fit_result['chi2_reduced']:.2f}",
            f"  Gradient significance: {self._fit_result['significance']:.1f} sigma",
            f"",
            f"Flat Model Comparison:",
            f"  Delta chi2 = {flat['delta_chi2']:.1f}",
            f"  p-value = {flat['p_value']:.2e}",
            f"",
            f"CCF Consistency:",
            f"  Tension with CCF prediction: {ccf['combined_tension_sigma']:.1f} sigma",
            f"  CCF consistent: {ccf['ccf_consistent']}",
            "=" * 60,
        ]

        return "\n".join(lines)
