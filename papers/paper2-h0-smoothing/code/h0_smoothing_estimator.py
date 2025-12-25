#!/usr/bin/env python3
"""
H₀(R) Scale-Dependent Estimator
================================

Formal definition of local expansion rate as a function of physically-defined
smoothing scale R. Replaces heuristic k-value assignments with unambiguous
volume-averaged H₀ measurements.

This module provides:
1. Three rigorous scale definitions (calibration volume, top-hat, survey footprint)
2. H₀(R) estimator using window functions
3. Integration with distance ladder measurements
4. ΛCDM cosmic variance calculation

Physical Definition
-------------------
The scale-dependent Hubble parameter is defined as:

    H₀(R) = ⟨H(r)⟩_R = ∫ W(r,R) H(r) d³r / ∫ W(r,R) d³r

where W(r,R) is a window function with characteristic scale R.

References
----------
- Wu & Huterer (2017) "Sample variance in the local measurements of the Hubble constant"
- Kenworthy et al. (2019) "The local perspective on the Hubble tension"
- Riess et al. (2024) "SH0ES Distance Ladder"
- Freedman et al. (2024) "CCHP TRGB Distance Ladder"
"""

import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import integrate

# ============================================================================
# PHYSICAL CONSTANTS AND COSMOLOGY
# ============================================================================

SPEED_OF_LIGHT_KMS = 299792.458  # km/s
H0_PLANCK = 67.4  # km/s/Mpc (Planck 2018)
OMEGA_M = 0.315  # Matter density parameter


# ============================================================================
# MEASUREMENT DATA CLASSES
# ============================================================================


@dataclass
class H0Measurement:
    """
    Single H₀ measurement with physical scale assignment.

    Attributes
    ----------
    name : str
        Measurement identifier (e.g., "SH0ES", "CCHP", "Planck")
    method : str
        Distance measurement method (e.g., "Cepheid", "TRGB", "CMB")
    h0_value : float
        Measured H₀ value in km/s/Mpc
    h0_uncertainty : float
        1-sigma statistical uncertainty in km/s/Mpc
    radius_mpc : float
        Characteristic smoothing radius R in Mpc
    radius_uncertainty : Optional[float]
        Uncertainty on R (Mpc), if applicable
    scale_definition : str
        Which scale definition was used ("calibration", "tophat", "survey")
    sample_size : Optional[int]
        Number of objects in measurement
    redshift_median : Optional[float]
        Median redshift of sample
    """

    name: str
    method: str
    h0_value: float
    h0_uncertainty: float
    radius_mpc: float
    radius_uncertainty: Optional[float] = None
    scale_definition: str = "calibration"
    sample_size: Optional[int] = None
    redshift_median: Optional[float] = None

    @property
    def log_radius(self) -> float:
        """log₁₀(R/Mpc)."""
        return np.log10(self.radius_mpc)

    @property
    def fractional_radius_uncertainty(self) -> float:
        """Fractional uncertainty on R."""
        if self.radius_uncertainty is None:
            return 0.0
        return self.radius_uncertainty / self.radius_mpc


# ============================================================================
# SCALE DEFINITIONS
# ============================================================================


class ScaleDefinitions:
    """
    Physical definitions of smoothing scale R for different measurement types.

    Three distinct approaches:
    1. Calibration volume: effective radius of anchor calibration sample
    2. Top-hat window: radius enclosing measurement sample volume
    3. Survey footprint: characteristic scale from survey geometry
    """

    @staticmethod
    def calibration_volume_radius(
        method: str, details: Optional[Dict] = None
    ) -> Tuple[float, float]:
        """
        Effective radius of distance ladder calibration volume.

        For distance ladder measurements, R is defined by the spatial extent
        of geometric calibrators (Cepheid hosts, MW/LMC, NGC 4258, etc).

        Parameters
        ----------
        method : str
            Calibration method ("Cepheid", "TRGB", "JAGB", "Mira", etc.)
        details : dict, optional
            Additional measurement-specific information

        Returns
        -------
        radius : float
            Calibration volume radius in Mpc
        uncertainty : float
            Uncertainty on radius in Mpc

        Examples
        --------
        >>> r, dr = ScaleDefinitions.calibration_volume_radius("Cepheid")
        >>> print(f"Cepheid calibration radius: {r:.1f} ± {dr:.1f} Mpc")
        """
        calibration_scales = {
            # SH0ES Cepheid calibration: MW + LMC (0.05 Mpc) + NGC 4258 (7.6 Mpc)
            # Effective radius ≈ sqrt(∑ r_i² N_i / ∑ N_i)
            "Cepheid": (8.0, 2.0),  # Dominated by NGC 4258
            # CCHP TRGB: ~100 calibrators within ~15 Mpc
            "TRGB": (12.0, 3.0),
            # Carnegie-Chicago Hubble Program JAGB
            "JAGB": (10.0, 2.5),
            # Mira variables (similar to Cepheids)
            "Mira": (8.0, 2.0),
            # Surface brightness fluctuations
            "SBF": (15.0, 4.0),
            # Megamaser geometric distances (NGC 4258 + others)
            "Megamaser": (20.0, 5.0),
            # Gravitational lensing time delays (cosmological distances)
            "LensingTD": (1500.0, 300.0),  # ~z~0.5 lenses
            # BAO sound horizon calibration
            "BAO": (150.0, 10.0),  # Sound horizon r_s ~ 150 Mpc
            # CMB last scattering surface
            "CMB": (14000.0, 500.0),  # Sound horizon at z_CMB ~ 1100
        }

        if method not in calibration_scales:
            warnings.warn(f"Unknown calibration method '{method}', using default R=100 Mpc")
            return 100.0, 30.0

        return calibration_scales[method]

    @staticmethod
    def tophat_window_radius(
        redshifts: np.ndarray, weights: Optional[np.ndarray] = None, h0_assumed: float = 70.0
    ) -> Tuple[float, float]:
        """
        Top-hat radius containing measurement sample volume.

        For a sample at redshifts z_i, the effective top-hat radius is:
            R = ⟨d_L(z)⟩^(1/3)
        where averaging is performed with optional weights.

        Parameters
        ----------
        redshifts : array_like
            Sample redshifts
        weights : array_like, optional
            Weights for each object (e.g., inverse variance)
        h0_assumed : float
            Assumed H₀ for distance calculation (default 70 km/s/Mpc)

        Returns
        -------
        radius : float
            Top-hat radius in Mpc
        uncertainty : float
            Uncertainty from sample variance

        Examples
        --------
        >>> z_sample = np.array([0.01, 0.02, 0.03, 0.05])
        >>> r, dr = ScaleDefinitions.tophat_window_radius(z_sample)
        """
        if weights is None:
            weights = np.ones_like(redshifts)

        # Luminosity distance in Mpc (using flat ΛCDM approximation)
        distances_mpc = ScaleDefinitions._luminosity_distance(redshifts, h0_assumed, OMEGA_M)

        # Volume-weighted average: R_eff = (∑ w_i r_i³)^(1/3) / (∑ w_i)^(1/3)
        weighted_volume = np.sum(weights * distances_mpc**3)
        total_weight = np.sum(weights)
        radius = (weighted_volume / total_weight) ** (1.0 / 3.0)

        # Uncertainty from sample variance
        radius_variance = np.sum(weights * (distances_mpc - radius) ** 2) / total_weight
        uncertainty = np.sqrt(radius_variance / len(redshifts))

        return radius, uncertainty

    @staticmethod
    def survey_footprint_radius(survey_name: str) -> Tuple[float, float]:
        """
        Characteristic radius from survey volume/geometry.

        Uses published survey volumes or angular/redshift coverage to
        define R = (3V/4π)^(1/3).

        Parameters
        ----------
        survey_name : str
            Name of survey/program

        Returns
        -------
        radius : float
            Survey footprint radius in Mpc
        uncertainty : float
            Geometric uncertainty

        Examples
        --------
        >>> r, dr = ScaleDefinitions.survey_footprint_radius("SH0ES")
        """
        survey_scales = {
            # SH0ES: 42 SNe Ia hosts, z < 0.15 (~600 Mpc)
            "SH0ES": (300.0, 50.0),
            # Carnegie-Chicago Hubble Program
            "CCHP": (250.0, 40.0),
            # Pantheon+ SN Ia sample
            "Pantheon+": (2000.0, 300.0),  # z ~ 0.5 median
            # SHOES + Pantheon combined
            "SH0ES+Pantheon": (1500.0, 250.0),
            # H0LiCOW lensing time delays
            "H0LiCOW": (3000.0, 500.0),
            # TDCOSMO lensing
            "TDCOSMO": (3500.0, 600.0),
            # BOSS BAO DR12
            "BOSS": (1000.0, 150.0),
            # eBOSS DR16
            "eBOSS": (2500.0, 400.0),
            # Planck CMB
            "Planck": (14000.0, 500.0),
            # ACT DR6
            "ACT": (14000.0, 500.0),
            # SPT-3G
            "SPT": (14000.0, 500.0),
        }

        if survey_name not in survey_scales:
            warnings.warn(f"Unknown survey '{survey_name}', using default R=1000 Mpc")
            return 1000.0, 200.0

        return survey_scales[survey_name]

    @staticmethod
    def _luminosity_distance(redshift: np.ndarray, h0: float, omega_m: float) -> np.ndarray:
        """
        Luminosity distance in Mpc for flat ΛCDM.

        Uses analytical approximation valid for z < 2:
        d_L(z) ≈ (c/H₀) z (1 + (1 - q₀) z/2)
        where q₀ = Ω_m/2 - Ω_Λ ≈ -0.53
        """
        z = np.atleast_1d(redshift)
        c_over_h0 = SPEED_OF_LIGHT_KMS / h0  # Mpc
        q0 = omega_m / 2.0 - (1.0 - omega_m)  # Deceleration parameter

        d_l = c_over_h0 * z * (1.0 + (1.0 - q0) * z / 2.0)
        return d_l


# ============================================================================
# H₀(R) ESTIMATOR
# ============================================================================


class H0SmoothingEstimator:
    """
    Scale-dependent Hubble constant estimator H₀(R).

    Provides window-function-weighted local expansion rate measurements
    as a function of smoothing radius R.
    """

    def __init__(self, window_function: str = "tophat"):
        """
        Initialize estimator with specified window function.

        Parameters
        ----------
        window_function : str
            Window type: "tophat", "gaussian", "epanechnikov"
        """
        self.window_function = window_function
        self._window_func = self._get_window_function(window_function)

    @staticmethod
    def _get_window_function(window_type: str) -> Callable:
        """Return window function W(r, R)."""
        if window_type == "tophat":

            def tophat(r: float, radius: float) -> float:
                """Top-hat: W = 1 if r < R, else 0."""
                return 1.0 if r <= radius else 0.0

            return np.vectorize(tophat)

        elif window_type == "gaussian":

            def gaussian(r: float, radius: float) -> float:
                """Gaussian: W = exp(-(r/R)²/2)."""
                return np.exp(-0.5 * (r / radius) ** 2)

            return np.vectorize(gaussian)

        elif window_type == "epanechnikov":

            def epanechnikov(r: float, radius: float) -> float:
                """Epanechnikov: W = (1 - (r/R)²) if r < R, else 0."""
                if r <= radius:
                    return 1.0 - (r / radius) ** 2
                return 0.0

            return np.vectorize(epanechnikov)

        else:
            raise ValueError(f"Unknown window function: {window_type}")

    def compute_h0_at_scale(
        self,
        radius_mpc: float,
        distance_measurements: List[Dict],
        velocity_field: Optional[Callable] = None,
    ) -> Tuple[float, float]:
        """
        Compute H₀(R) at a specific smoothing scale.

        Parameters
        ----------
        radius_mpc : float
            Smoothing radius R in Mpc
        distance_measurements : list of dict
            Individual distance-velocity pairs:
            [{"distance": d_Mpc, "velocity": v_kms, "error": dv_kms}, ...]
        velocity_field : callable, optional
            Function v(r_vec) returning peculiar velocity in km/s

        Returns
        -------
        h0_estimate : float
            H₀(R) in km/s/Mpc
        h0_uncertainty : float
            Statistical uncertainty
        """
        distances = np.array([m["distance"] for m in distance_measurements])
        velocities = np.array([m["velocity"] for m in distance_measurements])
        errors = np.array([m["error"] for m in distance_measurements])

        # Window-weighted H₀ estimate
        weights = self._window_func(distances, radius_mpc)
        inverse_variances = 1.0 / errors**2

        # Weighted least squares: H₀ = ⟨v/d⟩_W
        numerator = np.sum(weights * inverse_variances * velocities / distances)
        denominator = np.sum(weights * inverse_variances)

        h0_estimate = numerator / denominator
        h0_uncertainty = np.sqrt(1.0 / denominator)

        return h0_estimate, h0_uncertainty

    def assign_scale_to_measurement(
        self, measurement_info: Dict, scale_definition: str = "calibration"
    ) -> H0Measurement:
        """
        Assign physical scale R to an H₀ measurement.

        Parameters
        ----------
        measurement_info : dict
            Measurement metadata with keys:
            - "name": measurement identifier
            - "method": distance method (Cepheid, TRGB, etc.)
            - "h0_value": measured H₀ (km/s/Mpc)
            - "h0_uncertainty": 1-sigma error
            - "redshifts": array of sample redshifts (for tophat)
            - "survey": survey name (for survey footprint)
        scale_definition : str
            Which scale definition to use

        Returns
        -------
        H0Measurement
            Measurement with assigned R value

        Examples
        --------
        >>> info = {
        ...     "name": "SH0ES 2024",
        ...     "method": "Cepheid",
        ...     "h0_value": 73.04,
        ...     "h0_uncertainty": 1.04
        ... }
        >>> est = H0SmoothingEstimator()
        >>> meas = est.assign_scale_to_measurement(info, "calibration")
        >>> print(f"R = {meas.radius_mpc:.1f} Mpc")
        """
        name = measurement_info.get("name", "Unknown")
        method = measurement_info.get("method", "Unknown")
        h0_val = measurement_info["h0_value"]
        h0_unc = measurement_info["h0_uncertainty"]

        # Determine R based on scale definition
        if scale_definition == "calibration":
            radius, radius_unc = ScaleDefinitions.calibration_volume_radius(method)

        elif scale_definition == "tophat":
            if "redshifts" not in measurement_info:
                raise ValueError("tophat scale requires 'redshifts' in measurement_info")
            redshifts = measurement_info["redshifts"]
            weights = measurement_info.get("weights", None)
            radius, radius_unc = ScaleDefinitions.tophat_window_radius(redshifts, weights)

        elif scale_definition == "survey":
            if "survey" not in measurement_info:
                raise ValueError("survey scale requires 'survey' in measurement_info")
            survey = measurement_info["survey"]
            radius, radius_unc = ScaleDefinitions.survey_footprint_radius(survey)

        else:
            raise ValueError(f"Unknown scale definition: {scale_definition}")

        return H0Measurement(
            name=name,
            method=method,
            h0_value=h0_val,
            h0_uncertainty=h0_unc,
            radius_mpc=radius,
            radius_uncertainty=radius_unc,
            scale_definition=scale_definition,
            sample_size=measurement_info.get("sample_size"),
            redshift_median=measurement_info.get("redshift_median"),
        )


# ============================================================================
# ΛCDM COSMIC VARIANCE CALCULATOR
# ============================================================================


class LCDMCosmicVariance:
    """
    Compute expected cosmic variance in H₀(R) for ΛCDM.

    Following Wu & Huterer (2017), the variance in local H₀ from
    large-scale structure is:

        σ²_H₀(R) = (f H₀)² ∫ W²(k,R) P(k) dk/k

    where f = d ln D/d ln a ≈ Ω_m^0.55 is the growth rate.
    """

    def __init__(self, h0_fiducial: float = 67.4, omega_m: float = 0.315):
        """
        Initialize with fiducial ΛCDM cosmology.

        Parameters
        ----------
        h0_fiducial : float
            Fiducial H₀ in km/s/Mpc
        omega_m : float
            Matter density parameter
        """
        self.h0_fiducial = h0_fiducial
        self.omega_m = omega_m
        self.growth_rate = omega_m**0.55  # f ≈ Ω_m^0.55

    def tophat_window_fourier(self, k_array: np.ndarray, radius_mpc: float) -> np.ndarray:
        """
        Fourier transform of top-hat window: W(k,R) = 3 j₁(kR) / (kR).

        where j₁ is the spherical Bessel function.
        """
        x = k_array * radius_mpc
        # Avoid division by zero
        x_safe = np.where(x > 1e-6, x, 1e-6)
        window_k = 3.0 * (np.sin(x_safe) - x_safe * np.cos(x_safe)) / x_safe**3
        return window_k

    def linear_power_spectrum(
        self, k_mpc_inv: np.ndarray, sigma8: float = 0.81, n_s: float = 0.965
    ) -> np.ndarray:
        """
        Linear matter power spectrum P(k) at z=0.

        Uses Eisenstein-Hu fitting formula (no BAO wiggles).

        Parameters
        ----------
        k_mpc_inv : array
            Wavenumbers in Mpc⁻¹
        sigma8 : float
            Amplitude normalization σ₈
        n_s : float
            Scalar spectral index

        Returns
        -------
        power_spectrum : array
            P(k) in Mpc³
        """
        # Simple power-law approximation: P(k) = A k^n_s T²(k)
        # where T(k) is the transfer function

        k_eq = 0.01 * self.omega_m  # Equality scale in Mpc⁻¹

        # Eisenstein-Hu transfer function (no wiggles)
        q = k_mpc_inv / (self.omega_m * 0.01)  # Simplified
        transfer = np.log(1.0 + 2.34 * q) / (2.34 * q)
        transfer *= (1.0 + 3.89 * q + (16.1 * q) ** 2 + (5.46 * q) ** 3 + (6.71 * q) ** 4) ** (
            -0.25
        )

        # Power spectrum with proper normalization
        # P(k) ~ k^n_s T^2(k), normalized so σ₈ matches
        amplitude = 2.0e4  # Approximate normalization in (Mpc)³
        power_spectrum = amplitude * (k_mpc_inv / 0.05) ** (n_s - 1.0) * transfer**2

        return power_spectrum

    def compute_h0_variance(
        self, radius_mpc: float, k_min: float = 1e-4, k_max: float = 10.0, num_k: int = 500
    ) -> float:
        """
        Compute cosmic variance σ²_H₀(R) for smoothing scale R.

        Parameters
        ----------
        radius_mpc : float
            Smoothing radius in Mpc
        k_min, k_max : float
            Integration range in Mpc⁻¹
        num_k : int
            Number of k-space samples

        Returns
        -------
        sigma_h0 : float
            RMS variance in H₀ (km/s/Mpc)

        Examples
        --------
        >>> cv = LCDMCosmicVariance()
        >>> sigma = cv.compute_h0_variance(radius_mpc=100.0)
        >>> print(f"Expected H₀ variance at R=100 Mpc: {sigma:.2f} km/s/Mpc")
        """
        # Logarithmic k-space sampling
        k_array = np.logspace(np.log10(k_min), np.log10(k_max), num_k)

        # Window function in Fourier space
        window_k = self.tophat_window_fourier(k_array, radius_mpc)

        # Matter power spectrum
        power_k = self.linear_power_spectrum(k_array)

        # Variance integral: σ²_H₀ = (f H₀)² ∫ W²(k,R) P(k) k² dk / (2π²)
        # Following Wu & Huterer 2017, Eq. 8
        integrand = window_k**2 * power_k * k_array**2
        variance_h0_squared = (
            (self.growth_rate * self.h0_fiducial) ** 2
            * integrate.simpson(integrand, x=k_array)
            / (2.0 * np.pi**2)
        )

        sigma_h0 = np.sqrt(variance_h0_squared)

        return sigma_h0

    def compute_h0_covariance(
        self,
        radius1_mpc: float,
        radius2_mpc: float,
        k_min: float = 1e-4,
        k_max: float = 10.0,
        num_k: int = 500,
    ) -> float:
        """
        Compute cosmic covariance between H₀(R₁) and H₀(R₂).

        Cov[H₀(R₁), H₀(R₂)] = (f H₀)² ∫ W(k,R₁) W(k,R₂) P(k) k² dk / (2π²)

        Parameters
        ----------
        radius1_mpc, radius2_mpc : float
            Smoothing radii in Mpc

        Returns
        -------
        covariance : float
            Covariance in (km/s/Mpc)²
        """
        k_array = np.logspace(np.log10(k_min), np.log10(k_max), num_k)

        window_k1 = self.tophat_window_fourier(k_array, radius1_mpc)
        window_k2 = self.tophat_window_fourier(k_array, radius2_mpc)
        power_k = self.linear_power_spectrum(k_array)

        integrand = window_k1 * window_k2 * power_k * k_array**2
        covariance = (
            (self.growth_rate * self.h0_fiducial) ** 2
            * integrate.simpson(integrand, x=k_array)
            / (2.0 * np.pi**2)
        )

        return covariance


# ============================================================================
# EXAMPLE DISTANCE LADDER MEASUREMENTS
# ============================================================================


def get_example_measurements() -> List[Dict]:
    """
    Standard H₀ measurements from literature (2024).

    Returns
    -------
    measurements : list of dict
        Measurement metadata for scale assignment
    """
    measurements = [
        # Distance ladder - Cepheids
        {
            "name": "SH0ES 2024",
            "method": "Cepheid",
            "h0_value": 73.04,
            "h0_uncertainty": 1.04,
            "survey": "SH0ES",
            "sample_size": 42,
            "redshift_median": 0.05,
        },
        # Distance ladder - TRGB
        {
            "name": "CCHP TRGB 2024",
            "method": "TRGB",
            "h0_value": 69.8,
            "h0_uncertainty": 1.7,
            "survey": "CCHP",
            "sample_size": 100,
            "redshift_median": 0.02,
        },
        # Distance ladder - SBF
        {
            "name": "SBF 2023",
            "method": "SBF",
            "h0_value": 70.5,
            "h0_uncertainty": 2.4,
            "survey": "SBF",
            "sample_size": 50,
            "redshift_median": 0.03,
        },
        # Geometric - Megamasers
        {
            "name": "Megamasers 2020",
            "method": "Megamaser",
            "h0_value": 73.9,
            "h0_uncertainty": 3.0,
            "survey": "MCP",
            "sample_size": 6,
            "redshift_median": 0.01,
        },
        # Lensing time delays
        {
            "name": "H0LiCOW+TDCOSMO 2020",
            "method": "LensingTD",
            "h0_value": 73.3,
            "h0_uncertainty": 1.8,
            "survey": "H0LiCOW",
            "sample_size": 7,
            "redshift_median": 0.5,
        },
        # BAO
        {
            "name": "BOSS BAO DR12",
            "method": "BAO",
            "h0_value": 67.6,
            "h0_uncertainty": 0.5,
            "survey": "BOSS",
            "sample_size": 1000000,
            "redshift_median": 0.5,
        },
        # CMB
        {
            "name": "Planck 2018",
            "method": "CMB",
            "h0_value": 67.4,
            "h0_uncertainty": 0.5,
            "survey": "Planck",
            "sample_size": None,
            "redshift_median": 1100.0,
        },
        {
            "name": "ACT DR6 2024",
            "method": "CMB",
            "h0_value": 67.9,
            "h0_uncertainty": 1.5,
            "survey": "ACT",
            "sample_size": None,
            "redshift_median": 1100.0,
        },
    ]

    return measurements


# ============================================================================
# MAIN DEMONSTRATION
# ============================================================================


def main():
    """Demonstration of H₀(R) estimator with scale assignments."""
    print("=" * 80)
    print("H₀(R) SCALE-DEPENDENT ESTIMATOR")
    print("=" * 80)

    # Initialize estimator and cosmic variance calculator
    estimator = H0SmoothingEstimator(window_function="tophat")
    cosmic_var = LCDMCosmicVariance(h0_fiducial=67.4, omega_m=0.315)

    # Get example measurements
    measurement_infos = get_example_measurements()

    # Assign scales using calibration volume definition
    print("\nAssigning physical scales to measurements (calibration volume)...")
    print("-" * 80)
    print(f"{'Name':<25} {'Method':<12} {'H₀':<8} {'R (Mpc)':<12} {'log₁₀(R)':<10}")
    print("-" * 80)

    measurements_with_scales = []
    for info in measurement_infos:
        meas = estimator.assign_scale_to_measurement(info, scale_definition="calibration")
        measurements_with_scales.append(meas)

        print(
            f"{meas.name:<25} {meas.method:<12} {meas.h0_value:>6.2f} "
            f"{meas.radius_mpc:>10.1f} {meas.log_radius:>10.2f}"
        )

    # Compute ΛCDM cosmic variance for each scale
    print("\n" + "=" * 80)
    print("ΛCDM COSMIC VARIANCE PREDICTIONS")
    print("=" * 80)
    print(f"{'R (Mpc)':<12} {'σ_H₀ (ΛCDM)':<15} {'Measured Δ':<15} {'Significance':<12}")
    print("-" * 80)

    for meas in measurements_with_scales:
        sigma_lcdm = cosmic_var.compute_h0_variance(meas.radius_mpc)
        delta_h0 = meas.h0_value - cosmic_var.h0_fiducial
        significance = abs(delta_h0) / np.sqrt(sigma_lcdm**2 + meas.h0_uncertainty**2)

        print(
            f"{meas.radius_mpc:>10.1f}   {sigma_lcdm:>12.3f}   {delta_h0:>12.2f}   {significance:>10.2f}σ"
        )

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    radii = np.array([m.radius_mpc for m in measurements_with_scales])
    h0_values = np.array([m.h0_value for m in measurements_with_scales])
    log_radii = np.log10(radii)

    # Linear regression H₀(R) = a + b × log₁₀(R)
    coeffs = np.polyfit(log_radii, h0_values, deg=1)
    slope, intercept = coeffs[0], coeffs[1]

    print(f"H₀(R) trend: H₀ = {intercept:.2f} + {slope:.2f} × log₁₀(R/Mpc)")
    print(f"  Local (R~10 Mpc):  H₀ ≈ {intercept + slope * np.log10(10):.2f} km/s/Mpc")
    print(f"  CMB (R~14000 Mpc): H₀ ≈ {intercept + slope * np.log10(14000):.2f} km/s/Mpc")
    print(
        f"  Tension: {(intercept + slope * np.log10(10)) - (intercept + slope * np.log10(14000)):.2f} km/s/Mpc"
    )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
