#!/usr/bin/env python3
"""
ΛCDM Mock H₀(R) Generator
=========================

Generate mock H₀(R) realizations in ΛCDM to establish null distribution
for hypothesis testing. Includes proper cosmic variance from finite volume
sampling, large-scale structure, and sample variance.

This module provides:
1. Mock generation from ΛCDM velocity field realizations
2. Sample variance for finite samples
3. Different survey geometries (spherical, conical, slab)
4. Null distribution for trend detection

Physical Basis
--------------
In ΛCDM, local H₀ measurements scatter due to:

1. Cosmic variance from large-scale velocity field:
   σ²_cosmic(R) = (f H₀)² ∫ W²(k,R) P(k) dk/k

2. Sample variance from finite number of objects:
   σ²_sample = σ²_intrinsic / N_eff

3. Measurement errors:
   σ²_measurement = individual measurement uncertainties

The mock generator samples realizations of the velocity field consistent
with ΛCDM P(k) and computes H₀(R) for different smoothing scales.

References
----------
- Wu & Huterer (2017) "Sample variance in the local measurements of the Hubble constant"
- Kenworthy et al. (2019) "The local perspective on the Hubble tension"
- Carr et al. (2021) "Local H₀ measurements in light of local gravitational structure"
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

# ============================================================================
# CONSTANTS AND FIDUCIAL COSMOLOGY
# ============================================================================

SPEED_OF_LIGHT_KMS = 299792.458  # km/s
H0_FIDUCIAL = 67.4  # km/s/Mpc (Planck ΛCDM)
OMEGA_M_FIDUCIAL = 0.315
OMEGA_LAMBDA_FIDUCIAL = 0.685
SIGMA8_FIDUCIAL = 0.81
N_S_FIDUCIAL = 0.965


# ============================================================================
# VELOCITY FIELD SAMPLER
# ============================================================================


class VelocityFieldSampler:
    """
    Sample random Gaussian velocity field realizations consistent with ΛCDM P(k).

    The peculiar velocity field is related to matter density by:
        v(k) = i (f H a / k) δ(k) k̂

    where f = d ln D / d ln a ≈ Ω_m^0.55 is the growth rate.
    """

    def __init__(
        self,
        box_size_mpc: float = 1000.0,
        grid_size: int = 128,
        omega_m: float = OMEGA_M_FIDUCIAL,
        sigma8: float = SIGMA8_FIDUCIAL,
        n_s: float = N_S_FIDUCIAL,
        h0: float = H0_FIDUCIAL,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize velocity field sampler.

        Parameters
        ----------
        box_size_mpc : float
            Comoving box size in Mpc
        grid_size : int
            Number of grid points per dimension
        omega_m : float
            Matter density parameter
        sigma8 : float
            σ₈ normalization
        n_s : float
            Spectral index
        h0 : float
            Hubble constant in km/s/Mpc
        random_seed : int, optional
            Random seed for reproducibility
        """
        self.box_size = box_size_mpc
        self.grid_size = grid_size
        self.omega_m = omega_m
        self.sigma8 = sigma8
        self.n_s = n_s
        self.h0 = h0
        self.growth_rate = omega_m**0.55

        if random_seed is not None:
            np.random.seed(random_seed)

        # k-space grid
        self.k_fundamental = 2.0 * np.pi / box_size_mpc
        self.k_nyquist = np.pi * grid_size / box_size_mpc

    def _compute_transfer_function(self, k: np.ndarray) -> np.ndarray:
        """
        Compute the Eisenstein-Hu transfer function T(k).

        Parameters
        ----------
        k : array
            Wavenumbers in Mpc⁻¹

        Returns
        -------
        transfer : array
            Transfer function T(k)
        """
        omega_m = self.omega_m
        h = self.h0 / 100.0

        # Shape parameter
        theta_cmb = 2.726 / 2.7  # CMB temperature ratio
        omega_m_h2 = omega_m * h**2
        omega_b_h2 = 0.022  # Baryon density (approximate)

        # Transfer function parameters
        k_eq = 0.0746 * omega_m_h2 / theta_cmb**2  # Mpc⁻¹

        s = 44.5 * np.log(9.83 / omega_m_h2) / np.sqrt(1.0 + 10.0 * omega_b_h2**0.75)

        # Fitting functions
        alpha_gamma = (
            1.0
            - 0.328 * np.log(431.0 * omega_m_h2) * omega_b_h2 / omega_m_h2
            + 0.38 * np.log(22.3 * omega_m_h2) * (omega_b_h2 / omega_m_h2) ** 2
        )

        gamma_eff = omega_m * h * (alpha_gamma + (1.0 - alpha_gamma) / (1.0 + (0.43 * k * s) ** 4))

        q_eff = k * theta_cmb**2 / gamma_eff

        # Transfer function
        transfer = np.log(np.e + 1.84 * alpha_gamma * q_eff) / (
            np.log(np.e + 1.84 * alpha_gamma * q_eff)
            + (14.4 * q_eff + 325.0 * q_eff**2 + 1706.0 * q_eff**3)
            / (1.0 + 60.5 * q_eff + 162.8 * q_eff**2 + 2118.0 * q_eff**3)
        )

        return transfer

    def eisenstein_hu_power_spectrum(self, k_array: np.ndarray) -> np.ndarray:
        """
        Eisenstein-Hu linear matter power spectrum (no BAO wiggles).

        Parameters
        ----------
        k_array : array
            Wavenumbers in Mpc⁻¹

        Returns
        -------
        power_spectrum : array
            P(k) in (Mpc)³
        """
        k = k_array
        transfer = self._compute_transfer_function(k)

        # Normalize to σ₈ using unnormalized power spectrum
        def unnormalized_power(k_test):
            t = self._compute_transfer_function(k_test)
            return k_test**self.n_s * t**2 / k_test**3

        normalization = self.sigma8**2 / self._compute_sigma8_from_power(unnormalized_power)

        power_spectrum = normalization * k**self.n_s * transfer**2 / k**3

        return power_spectrum

    def _compute_sigma8_from_power(self, power_func: Callable) -> float:
        """Compute σ₈ from power spectrum for normalization."""
        r8_mpc = 8.0  # 8 h⁻¹ Mpc
        k_array = np.logspace(-4, 2, 1000)
        power = power_func(k_array)

        # Window function W(k*R) = 3 j₁(kR) / (kR)
        x = k_array * r8_mpc
        window = 3.0 * (np.sin(x) - x * np.cos(x)) / x**3

        # σ²(R) = ∫ P(k) W²(k,R) k² dk / (2π²)
        integrand = power * window**2 * k_array**2
        variance = integrate.simpson(integrand, x=k_array) / (2.0 * np.pi**2)

        return np.sqrt(variance)

    def generate_gaussian_random_field(self) -> np.ndarray:
        """
        Generate 3D Gaussian random density field δ(x).

        Returns
        -------
        delta_field : ndarray, shape (grid_size, grid_size, grid_size)
            Density perturbation field
        """
        grid = self.grid_size

        # k-space grid
        kx = np.fft.fftfreq(grid, d=self.box_size / grid) * 2.0 * np.pi
        ky = np.fft.fftfreq(grid, d=self.box_size / grid) * 2.0 * np.pi
        kz = np.fft.fftfreq(grid, d=self.box_size / grid) * 2.0 * np.pi

        kx_grid, ky_grid, kz_grid = np.meshgrid(kx, ky, kz, indexing="ij")
        k_magnitude = np.sqrt(kx_grid**2 + ky_grid**2 + kz_grid**2)
        k_magnitude[0, 0, 0] = 1.0  # Avoid division by zero

        # Power spectrum
        power_k = self.eisenstein_hu_power_spectrum(k_magnitude.flatten()).reshape(
            k_magnitude.shape
        )

        # Random phases
        delta_k = np.sqrt(power_k / 2.0) * (
            np.random.randn(grid, grid, grid) + 1j * np.random.randn(grid, grid, grid)
        )

        # Reality condition: δ(-k) = δ*(k)
        delta_k[0, 0, 0] = 0.0  # Mean density is zero

        # Inverse FFT to get real-space field
        delta_field = np.fft.ifftn(delta_k).real

        return delta_field

    def generate_velocity_field(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate 3D peculiar velocity field from density field.

        v(k) = i (f H a / k) δ(k) k̂

        Returns
        -------
        vx, vy, vz : ndarrays, shape (grid_size, grid_size, grid_size)
            Velocity components in km/s
        """
        grid = self.grid_size

        # Generate density field
        delta_field = self.generate_gaussian_random_field()
        delta_k = np.fft.fftn(delta_field)

        # k-space grid
        kx = np.fft.fftfreq(grid, d=self.box_size / grid) * 2.0 * np.pi
        ky = np.fft.fftfreq(grid, d=self.box_size / grid) * 2.0 * np.pi
        kz = np.fft.fftfreq(grid, d=self.box_size / grid) * 2.0 * np.pi

        kx_grid, ky_grid, kz_grid = np.meshgrid(kx, ky, kz, indexing="ij")
        k_magnitude = np.sqrt(kx_grid**2 + ky_grid**2 + kz_grid**2)
        k_magnitude[0, 0, 0] = 1.0  # Avoid division by zero

        # Velocity in Fourier space: v_i(k) = i (f H a / k²) δ(k) k_i
        # At z=0, a=1
        velocity_prefactor = self.growth_rate * self.h0 / k_magnitude**2

        vx_k = 1j * velocity_prefactor * delta_k * kx_grid
        vy_k = 1j * velocity_prefactor * delta_k * ky_grid
        vz_k = 1j * velocity_prefactor * delta_k * kz_grid

        # Inverse FFT to get real-space velocities
        vx = np.fft.ifftn(vx_k).real
        vy = np.fft.ifftn(vy_k).real
        vz = np.fft.ifftn(vz_k).real

        return vx, vy, vz


# ============================================================================
# MOCK H₀(R) GENERATOR
# ============================================================================


@dataclass
class MockH0Curve:
    """
    Single mock H₀(R) realization.

    Attributes
    ----------
    radii_mpc : array
        Smoothing radii in Mpc
    h0_values : array
        H₀(R) values in km/s/Mpc
    h0_uncertainties : array
        Statistical uncertainties at each R
    realization_index : int
        Index of this mock realization
    """

    radii_mpc: np.ndarray
    h0_values: np.ndarray
    h0_uncertainties: np.ndarray
    realization_index: int


class LCDMMockGenerator:
    """
    Generate ensemble of mock H₀(R) curves in ΛCDM.

    Samples velocity field realizations and computes H₀(R) for different
    smoothing scales to build null distribution.
    """

    def __init__(
        self,
        h0_fiducial: float = H0_FIDUCIAL,
        omega_m: float = OMEGA_M_FIDUCIAL,
        sigma8: float = SIGMA8_FIDUCIAL,
        box_size_mpc: float = 1000.0,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize mock generator.

        Parameters
        ----------
        h0_fiducial : float
            True H₀ in ΛCDM (km/s/Mpc)
        omega_m : float
            Matter density parameter
        sigma8 : float
            σ₈ normalization
        box_size_mpc : float
            Simulation box size
        random_seed : int, optional
            Random seed
        """
        self.h0_fiducial = h0_fiducial
        self.omega_m = omega_m
        self.sigma8 = sigma8
        self.box_size = box_size_mpc
        self.random_seed = random_seed

        self.velocity_sampler = VelocityFieldSampler(
            box_size_mpc=box_size_mpc,
            omega_m=omega_m,
            sigma8=sigma8,
            h0=h0_fiducial,
            random_seed=random_seed,
        )

    def sample_observers_in_volume(
        self, num_observers: int = 100, geometry: str = "spherical", max_radius_mpc: float = 500.0
    ) -> np.ndarray:
        """
        Sample observer positions within survey volume.

        Parameters
        ----------
        num_observers : int
            Number of observer positions
        geometry : str
            Survey geometry: "spherical", "conical", "slab"
        max_radius_mpc : float
            Maximum distance from origin

        Returns
        -------
        positions : ndarray, shape (num_observers, 3)
            Observer positions in Mpc
        """
        if geometry == "spherical":
            # Uniform in sphere: r ~ U(0, R_max)^(1/3)
            radii = max_radius_mpc * np.random.rand(num_observers) ** (1.0 / 3.0)
            theta = np.arccos(2.0 * np.random.rand(num_observers) - 1.0)
            phi = 2.0 * np.pi * np.random.rand(num_observers)

            x = radii * np.sin(theta) * np.cos(phi)
            y = radii * np.sin(theta) * np.sin(phi)
            z = radii * np.cos(theta)

        elif geometry == "conical":
            # Conical: uniform in z, declining density with distance
            radii = max_radius_mpc * np.random.rand(num_observers)
            theta = np.arccos(1.0 - 0.5 * np.random.rand(num_observers))  # Half-cone
            phi = 2.0 * np.pi * np.random.rand(num_observers)

            x = radii * np.sin(theta) * np.cos(phi)
            y = radii * np.sin(theta) * np.sin(phi)
            z = radii * np.cos(theta)

        elif geometry == "slab":
            # Slab: uniform in xy, limited z range
            x = max_radius_mpc * (2.0 * np.random.rand(num_observers) - 1.0)
            y = max_radius_mpc * (2.0 * np.random.rand(num_observers) - 1.0)
            z = (max_radius_mpc / 10.0) * (2.0 * np.random.rand(num_observers) - 1.0)

        else:
            raise ValueError(f"Unknown geometry: {geometry}")

        positions = np.column_stack([x, y, z])
        return positions

    def compute_h0_from_velocity_field(
        self,
        observer_positions: np.ndarray,
        velocity_field: Tuple[np.ndarray, np.ndarray, np.ndarray],
        smoothing_radius_mpc: float,
    ) -> Tuple[float, float]:
        """
        Compute H₀(R) from velocity field at given smoothing scale.

        Parameters
        ----------
        observer_positions : ndarray, shape (N, 3)
            Observer positions in Mpc
        velocity_field : tuple of ndarrays
            (vx, vy, vz) velocity components
        smoothing_radius_mpc : float
            Smoothing scale R

        Returns
        -------
        h0_estimate : float
            Estimated H₀(R) in km/s/Mpc
        h0_uncertainty : float
            Statistical uncertainty from sample variance
        """
        vx, vy, vz = velocity_field
        grid_size = vx.shape[0]
        cell_size = self.box_size / grid_size

        # Compute distances and apply top-hat window
        distances = np.linalg.norm(observer_positions, axis=1)
        in_window = distances <= smoothing_radius_mpc

        if np.sum(in_window) < 5:
            # Insufficient sample
            return self.h0_fiducial, 5.0

        positions_windowed = observer_positions[in_window]
        distances_windowed = distances[in_window]

        # Interpolate velocity field at observer positions
        velocities_radial = []
        for pos in positions_windowed:
            # Grid indices (periodic boundary conditions)
            ix = int((pos[0] + self.box_size / 2.0) / cell_size) % grid_size
            iy = int((pos[1] + self.box_size / 2.0) / cell_size) % grid_size
            iz = int((pos[2] + self.box_size / 2.0) / cell_size) % grid_size

            # Velocity at this position
            v_vec = np.array([vx[ix, iy, iz], vy[ix, iy, iz], vz[ix, iy, iz]])

            # Radial component
            r_hat = pos / np.linalg.norm(pos)
            v_radial = np.dot(v_vec, r_hat)

            velocities_radial.append(v_radial)

        velocities_radial = np.array(velocities_radial)

        # Hubble flow + peculiar velocity: v_obs = H₀ r + v_pec
        # Estimate H₀ from linear fit
        weights = np.ones_like(distances_windowed)
        h0_estimate = np.sum(weights * velocities_radial / distances_windowed) / np.sum(weights)

        # Add Hubble flow
        h0_estimate += self.h0_fiducial

        # Uncertainty from sample variance
        residuals = velocities_radial - (h0_estimate - self.h0_fiducial) * distances_windowed
        h0_uncertainty = np.std(residuals / distances_windowed) / np.sqrt(len(distances_windowed))

        return h0_estimate, h0_uncertainty

    def generate_mock_h0_curves(
        self,
        num_mocks: int = 100,
        radii_mpc: Optional[np.ndarray] = None,
        num_observers: int = 50,
        geometry: str = "spherical",
    ) -> List[MockH0Curve]:
        """
        Generate ensemble of mock H₀(R) curves.

        Parameters
        ----------
        num_mocks : int
            Number of mock realizations
        radii_mpc : array, optional
            Smoothing radii to evaluate (default: 10-10000 Mpc)
        num_observers : int
            Number of observer positions per realization
        geometry : str
            Survey geometry

        Returns
        -------
        mock_curves : list of MockH0Curve
            Mock H₀(R) realizations

        Examples
        --------
        >>> gen = LCDMMockGenerator()
        >>> mocks = gen.generate_mock_h0_curves(num_mocks=100)
        >>> print(f"Generated {len(mocks)} mock realizations")
        """
        if radii_mpc is None:
            radii_mpc = np.logspace(1, 4, 20)  # 10 to 10000 Mpc

        print(f"Generating {num_mocks} mock H₀(R) realizations...")
        print(f"  Radii: {radii_mpc.min():.1f} - {radii_mpc.max():.1f} Mpc")
        print(f"  Observers per mock: {num_observers}")
        print(f"  Geometry: {geometry}")

        mock_curves = []

        for i_mock in range(num_mocks):
            if (i_mock + 1) % 10 == 0:
                print(f"  Mock {i_mock + 1}/{num_mocks}...")

            # Generate velocity field realization
            vx, vy, vz = self.velocity_sampler.generate_velocity_field()

            # Sample observer positions
            observers = self.sample_observers_in_volume(
                num_observers=num_observers, geometry=geometry, max_radius_mpc=radii_mpc.max()
            )

            # Compute H₀(R) at each smoothing scale
            h0_values = []
            h0_uncertainties = []

            for radius in radii_mpc:
                h0_r, h0_unc = self.compute_h0_from_velocity_field(observers, (vx, vy, vz), radius)
                h0_values.append(h0_r)
                h0_uncertainties.append(h0_unc)

            mock_curve = MockH0Curve(
                radii_mpc=radii_mpc,
                h0_values=np.array(h0_values),
                h0_uncertainties=np.array(h0_uncertainties),
                realization_index=i_mock,
            )

            mock_curves.append(mock_curve)

        print(f"✓ Generated {num_mocks} mock realizations")
        return mock_curves

    def compute_null_statistics(self, mock_curves: List[MockH0Curve]) -> Dict:
        """
        Compute null distribution statistics from mocks.

        Parameters
        ----------
        mock_curves : list of MockH0Curve
            Mock realizations

        Returns
        -------
        statistics : dict
            Null distribution statistics
        """
        num_mocks = len(mock_curves)
        num_radii = len(mock_curves[0].radii_mpc)

        # Stack all mock H₀ values
        h0_array = np.array(
            [mock.h0_values for mock in mock_curves]
        )  # Shape: (num_mocks, num_radii)

        # Compute mean and scatter at each R
        h0_mean = np.mean(h0_array, axis=0)
        h0_std = np.std(h0_array, axis=0)
        h0_percentiles = {
            "16": np.percentile(h0_array, 16, axis=0),
            "50": np.percentile(h0_array, 50, axis=0),
            "84": np.percentile(h0_array, 84, axis=0),
            "2.5": np.percentile(h0_array, 2.5, axis=0),
            "97.5": np.percentile(h0_array, 97.5, axis=0),
        }

        # Compute slopes for each mock (H₀(R) = a + b × log₁₀(R))
        log_radii = np.log10(mock_curves[0].radii_mpc)
        slopes = []
        intercepts = []

        for h0_values in h0_array:
            coeffs = np.polyfit(log_radii, h0_values, deg=1)
            slopes.append(coeffs[0])
            intercepts.append(coeffs[1])

        slopes = np.array(slopes)
        intercepts = np.array(intercepts)

        statistics = {
            "radii_mpc": mock_curves[0].radii_mpc,
            "h0_mean": h0_mean,
            "h0_std": h0_std,
            "h0_percentiles": h0_percentiles,
            "slopes": slopes,
            "slope_mean": np.mean(slopes),
            "slope_std": np.std(slopes),
            "intercepts": intercepts,
            "intercept_mean": np.mean(intercepts),
            "intercept_std": np.std(intercepts),
            "num_mocks": num_mocks,
        }

        return statistics


# ============================================================================
# VISUALIZATION
# ============================================================================


def plot_mock_h0_ensemble(
    mock_curves: List[MockH0Curve], statistics: Dict, output_path: str = "mock_h0_ensemble.png"
):
    """
    Plot ensemble of mock H₀(R) curves with statistics.

    Parameters
    ----------
    mock_curves : list of MockH0Curve
        Mock realizations
    statistics : dict
        Null distribution statistics
    output_path : str
        Output file path
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    radii = statistics["radii_mpc"]
    log_radii = np.log10(radii)

    # Upper panel: H₀(R) curves
    ax = axes[0]

    # Plot individual mocks (faint)
    for mock in mock_curves[:50]:  # Plot first 50 to avoid clutter
        ax.plot(log_radii, mock.h0_values, color="blue", alpha=0.1, linewidth=0.5)

    # Plot median and percentiles
    ax.plot(log_radii, statistics["h0_percentiles"]["50"], "k-", linewidth=2, label="Median")
    ax.fill_between(
        log_radii,
        statistics["h0_percentiles"]["16"],
        statistics["h0_percentiles"]["84"],
        alpha=0.3,
        color="gray",
        label="68% CI",
    )
    ax.fill_between(
        log_radii,
        statistics["h0_percentiles"]["2.5"],
        statistics["h0_percentiles"]["97.5"],
        alpha=0.15,
        color="gray",
        label="95% CI",
    )

    # Fiducial H₀
    ax.axhline(
        H0_FIDUCIAL,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"True H₀ = {H0_FIDUCIAL} km/s/Mpc",
    )

    ax.set_ylabel("H₀ [km/s/Mpc]", fontsize=12)
    ax.set_title(
        f'ΛCDM Mock H₀(R) Ensemble (N={statistics["num_mocks"]} realizations)',
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Lower panel: Slope distribution
    ax = axes[1]
    ax.hist(statistics["slopes"], bins=30, alpha=0.7, color="blue", edgecolor="black")
    ax.axvline(0, color="red", linestyle="--", linewidth=2, label="Zero slope (no trend)")
    ax.axvline(
        statistics["slope_mean"],
        color="green",
        linestyle="-",
        linewidth=2,
        label=f'Mean slope = {statistics["slope_mean"]:.3f} ± {statistics["slope_std"]:.3f}',
    )

    ax.set_xlabel("Slope [km/s/Mpc per decade in R]", fontsize=12)
    ax.set_ylabel("Number of Mocks", fontsize=12)
    ax.set_title("Null Distribution of H₀(R) Slopes", fontsize=13, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {output_path}")
    plt.close()


# ============================================================================
# MAIN DEMONSTRATION
# ============================================================================


def main():
    """Demonstration of ΛCDM mock generator."""
    print("=" * 80)
    print("ΛCDM MOCK H₀(R) GENERATOR")
    print("=" * 80)

    # Initialize mock generator
    generator = LCDMMockGenerator(
        h0_fiducial=67.4, omega_m=0.315, sigma8=0.81, box_size_mpc=1000.0, random_seed=42
    )

    # Generate mock ensemble
    radii = np.logspace(1, 4, 15)  # 10 to 10000 Mpc
    mocks = generator.generate_mock_h0_curves(
        num_mocks=100, radii_mpc=radii, num_observers=50, geometry="spherical"
    )

    # Compute null statistics
    print("\nComputing null distribution statistics...")
    stats = generator.compute_null_statistics(mocks)

    print("\n" + "=" * 80)
    print("NULL DISTRIBUTION STATISTICS")
    print("=" * 80)
    print(f"Number of mocks: {stats['num_mocks']}")
    print("Slope distribution:")
    print(f"  Mean: {stats['slope_mean']:.4f} km/s/Mpc/decade")
    print(f"  Std:  {stats['slope_std']:.4f} km/s/Mpc/decade")
    print(
        f"  95% range: [{np.percentile(stats['slopes'], 2.5):.4f}, {np.percentile(stats['slopes'], 97.5):.4f}]"
    )
    print("\nIntercept distribution:")
    print(f"  Mean: {stats['intercept_mean']:.2f} km/s/Mpc")
    print(f"  Std:  {stats['intercept_std']:.2f} km/s/Mpc")

    # Visualization
    print("\nGenerating visualization...")
    plot_mock_h0_ensemble(mocks, stats, output_path="/tmp/mock_h0_ensemble.png")

    print("\n" + "=" * 80)
    print("FALSIFICATION CRITERION")
    print("=" * 80)
    print("To claim detection of H₀(R) trend:")
    print(f"  Observed slope must exceed 2σ threshold: |m| > {2 * stats['slope_std']:.4f}")
    print("  Or p-value < 0.05 from null distribution")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
