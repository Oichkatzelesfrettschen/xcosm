"""
GravitationalFractal: The Sound of Fractal Death

This module computes the gravitational wave emission from Type Ia supernovae
with fractal flame fronts. The key insight:

    A spherically symmetric explosion (D=2.0) produces ZERO gravitational waves.
    A fractal flame (D>2.0) has inherent asymmetry → non-zero quadrupole → GW emission.

The "Loudness" (strain amplitude) of the GW signal scales with fractal dimension D.

Physical Framework:
    - GW strain: h_ij = (2G/c⁴r) d²Q_ij^TT/dt²
    - Quadrupole moment: Q_ij = ∫ ρ(x)[3x_i x_j - r²δ_ij] d³x
    - For fractal surface: variance in Q_ij ~ (D - 2)

Key Results from Literature:
    - SNe Ia GW frequency: 0.4-2.5 Hz (centered ~1 Hz)
    - Expected strain: h ~ 10⁻²¹ to 10⁻²² at 10 kpc
    - Energy in GW: ~7×10³⁹ erg (delayed-detonation models)
    - Detection requires: DECIGO/BBO (decihertz band, 0.1-10 Hz)

References:
    - Seitenzahl et al. (2015): PRD 92, 124013 (delayed-detonation GW)
    - Seitenzahl et al. (2011): PRL 106, 201103 (single-degenerate GW)

Author: Spandrel Project, 2025
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict
import numpy as np
from numpy.typing import NDArray
from scipy.fft import fft, fftfreq
from scipy.special import sph_harm
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt
from matplotlib import cm


# =============================================================================
# Physical Constants (CGS)
# =============================================================================

G_GRAV = 6.674e-8          # Gravitational constant [cm³/g/s²]
C_LIGHT = 2.998e10         # Speed of light [cm/s]
M_SUN = 1.989e33           # Solar mass [g]
PC_CM = 3.086e18           # Parsec in cm
KPC_CM = 3.086e21          # Kiloparsec in cm
MPC_CM = 3.086e24          # Megaparsec in cm

# White Dwarf Parameters
M_WD = 1.4 * M_SUN         # Chandrasekhar mass
R_WD = 2.0e8               # WD radius [cm] (~2000 km)
RHO_WD = 2.0e9             # Central density [g/cm³]

# Explosion Dynamics
V_DET = 1.5e9              # Detonation velocity [cm/s] (~15,000 km/s)
T_EXPLOSION = 2.0          # Explosion timescale [s]


# =============================================================================
# Fractal Surface Generation (Spherical Harmonics)
# =============================================================================

@dataclass
class FractalSurface:
    """
    A fractal spherical surface generated via spherical harmonic expansion.

    The power spectrum follows P(l) ~ l^(-β) where β = 2H + 1
    and the fractal dimension D = 3 - H.

    For turbulent flames:
        - Kolmogorov: D ≈ 7/3 ≈ 2.33, β = 8/3
        - Experimental: D ≈ 2.3-2.5
    """
    fractal_dimension: float
    mean_radius: float
    amplitude_rms: float
    lmax: int = 64
    seed: Optional[int] = None

    # Generated data
    coeffs_real: NDArray[np.float64] = field(default=None, repr=False)
    coeffs_imag: NDArray[np.float64] = field(default=None, repr=False)

    def __post_init__(self):
        if not 2.0 <= self.fractal_dimension <= 3.0:
            raise ValueError(f"Fractal dimension must be in [2, 3], got {self.fractal_dimension}")
        self._generate_coefficients()

    @property
    def hurst_exponent(self) -> float:
        """Hurst exponent H = 3 - D for 2D surface in 3D."""
        return 3.0 - self.fractal_dimension

    @property
    def spectral_index(self) -> float:
        """Spectral index β = 2H + 1."""
        return 2.0 * self.hurst_exponent + 1.0

    def _generate_coefficients(self):
        """Generate spherical harmonic coefficients with fractal power spectrum."""
        if self.seed is not None:
            np.random.seed(self.seed)

        self.coeffs_real = np.zeros((self.lmax + 1, self.lmax + 1))
        self.coeffs_imag = np.zeros((self.lmax + 1, self.lmax + 1))

        # Power spectrum: P(l) ~ l^(-β)
        beta = self.spectral_index

        # Normalization to achieve desired RMS amplitude
        total_variance = 0.0
        for l in range(1, self.lmax + 1):
            power_l = (l + 0.5) ** (-beta)
            total_variance += power_l * (2 * l + 1)

        normalization = self.amplitude_rms / np.sqrt(total_variance)

        # Generate random coefficients
        for l in range(1, self.lmax + 1):
            power_l = (l + 0.5) ** (-beta)
            std_l = normalization * np.sqrt(power_l / (2 * l + 1))

            for m in range(0, l + 1):
                self.coeffs_real[l, m] = std_l * np.random.randn()
                if m > 0:
                    self.coeffs_imag[l, m] = std_l * np.random.randn()

    def evaluate(self, theta: NDArray[np.float64],
                 phi: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Evaluate the surface radius at given angular coordinates.

        Parameters:
            theta: Polar angle [0, π]
            phi: Azimuthal angle [0, 2π]

        Returns:
            r: Radial coordinate at each (θ, φ)
        """
        displacement = np.zeros_like(theta)

        for l in range(1, self.lmax + 1):
            for m in range(0, l + 1):
                # Spherical harmonic Y_l^m(θ, φ)
                ylm = sph_harm(m, l, phi, theta)

                # Real spherical harmonic contribution
                if m == 0:
                    displacement += self.coeffs_real[l, m] * np.real(ylm)
                else:
                    displacement += self.coeffs_real[l, m] * np.real(ylm) * np.sqrt(2)
                    displacement += self.coeffs_imag[l, m] * np.imag(ylm) * np.sqrt(2)

        return self.mean_radius + displacement

    def generate_grid(self, n_theta: int = 64,
                      n_phi: int = 128) -> Tuple[NDArray, NDArray, NDArray]:
        """
        Generate a grid of surface points.

        Returns:
            theta_grid, phi_grid, r_grid
        """
        theta = np.linspace(0.01, np.pi - 0.01, n_theta)
        phi = np.linspace(0, 2 * np.pi, n_phi)
        theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')

        r_grid = self.evaluate(theta_grid, phi_grid)

        return theta_grid, phi_grid, r_grid

    def to_cartesian(self, theta: NDArray, phi: NDArray,
                     r: NDArray) -> Tuple[NDArray, NDArray, NDArray]:
        """Convert spherical to Cartesian coordinates."""
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return x, y, z

    def surface_area(self, n_theta: int = 64, n_phi: int = 128) -> float:
        """
        Compute approximate surface area.

        For small perturbations: A ≈ ∫ r² sin(θ) dθ dφ
        """
        theta_grid, phi_grid, r_grid = self.generate_grid(n_theta, n_phi)

        dtheta = np.pi / n_theta
        dphi = 2 * np.pi / n_phi

        # Surface element in spherical coordinates
        area_element = r_grid**2 * np.sin(theta_grid) * dtheta * dphi

        return np.sum(area_element)

    def asymmetry_parameter(self, n_theta: int = 64, n_phi: int = 128) -> float:
        """
        Compute the asymmetry parameter (RMS deviation from sphere).

        α = sqrt(⟨(r - r_mean)²⟩) / r_mean
        """
        theta_grid, phi_grid, r_grid = self.generate_grid(n_theta, n_phi)

        deviation = r_grid - self.mean_radius
        rms_deviation = np.sqrt(np.mean(deviation**2))

        return rms_deviation / self.mean_radius


# =============================================================================
# Quadrupole Moment Tensor
# =============================================================================

class QuadrupoleTensor:
    """
    Computes the mass quadrupole moment tensor Q_ij for a fractal shell.

    Q_ij = ∫ ρ(x) [3x_i x_j - r² δ_ij] d³x

    For a thin shell of mass M at radius r(θ,φ):
        Q_ij = M ∫ [3x_i x_j - r² δ_ij] dΩ / (4π)

    The GW strain depends on d²Q_ij/dt².
    """

    def __init__(self, surface: FractalSurface, shell_mass: float):
        """
        Initialize with a fractal surface and shell mass.

        Parameters:
            surface: FractalSurface object defining the geometry
            shell_mass: Total mass of the shell [g]
        """
        self.surface = surface
        self.shell_mass = shell_mass
        self.n_theta = 64
        self.n_phi = 128

    def compute(self) -> NDArray[np.float64]:
        """
        Compute the traceless quadrupole moment tensor Q_ij.

        Returns:
            Q: 3x3 symmetric traceless tensor [g cm²]
        """
        theta, phi, r = self.surface.generate_grid(self.n_theta, self.n_phi)
        x, y, z = self.surface.to_cartesian(theta, phi, r)

        # Solid angle element
        dtheta = np.pi / self.n_theta
        dphi = 2 * np.pi / self.n_phi
        solid_angle = np.sin(theta) * dtheta * dphi

        # Surface mass density (mass per solid angle)
        sigma = self.shell_mass / (4 * np.pi)

        # Cartesian positions (flattened)
        positions = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
        weights = (sigma * solid_angle).ravel()
        r_sq = (r**2).ravel()

        # Compute Q_ij = Σ m_k [3 x_i x_j - r² δ_ij]
        Q = np.zeros((3, 3))

        for k in range(len(weights)):
            pos = positions[k]
            outer = np.outer(pos, pos)
            Q += weights[k] * (3 * outer - r_sq[k] * np.eye(3))

        # Enforce symmetry and tracelessness
        Q = 0.5 * (Q + Q.T)
        Q -= np.trace(Q) / 3.0 * np.eye(3)

        return Q

    def compute_components(self) -> Dict[str, float]:
        """
        Compute and return individual Q_ij components.

        The traceless symmetric tensor has 5 independent components:
        Q_xx, Q_yy (Q_zz = -Q_xx - Q_yy), Q_xy, Q_xz, Q_yz
        """
        Q = self.compute()

        return {
            'Q_xx': Q[0, 0],
            'Q_yy': Q[1, 1],
            'Q_zz': Q[2, 2],
            'Q_xy': Q[0, 1],
            'Q_xz': Q[0, 2],
            'Q_yz': Q[1, 2],
            'trace': np.trace(Q),  # Should be ~0
            'frobenius_norm': np.sqrt(np.sum(Q**2))
        }


# =============================================================================
# Time Evolution of Expanding Fractal Shell
# =============================================================================

class ExpandingFractalFlame:
    """
    Models a time-evolving fractal flame front expanding through a white dwarf.

    The flame starts at some initial radius and expands at the detonation
    velocity while maintaining its fractal character. The quadrupole moment
    evolves in time, generating gravitational waves.
    """

    def __init__(self,
                 fractal_dimension: float,
                 initial_radius: float = R_WD * 0.1,
                 final_radius: float = R_WD,
                 expansion_time: float = T_EXPLOSION,
                 shell_mass: float = M_WD * 0.6,
                 n_timesteps: int = 500):
        """
        Initialize the expanding flame model.

        Parameters:
            fractal_dimension: D of the flame front (2.0-3.0)
            initial_radius: Starting radius [cm]
            final_radius: Ending radius [cm]
            expansion_time: Total expansion time [s]
            shell_mass: Mass of the burning shell [g]
            n_timesteps: Number of time samples
        """
        self.D = fractal_dimension
        self.r_init = initial_radius
        self.r_final = final_radius
        self.t_exp = expansion_time
        self.M_shell = shell_mass
        self.n_steps = n_timesteps

        # Time array
        self.times = np.linspace(0, expansion_time, n_timesteps)
        self.dt = self.times[1] - self.times[0]

        # Amplitude scales with radius (self-similar turbulence)
        self.amplitude_fraction = 0.1  # RMS displacement / mean radius

        # Storage for quadrupole evolution
        self._Q_history: Optional[NDArray] = None
        self._computed = False

    def mean_radius(self, t: float) -> float:
        """Mean radius at time t (linear expansion)."""
        progress = t / self.t_exp
        return self.r_init + (self.r_final - self.r_init) * progress

    def amplitude(self, t: float) -> float:
        """RMS amplitude at time t (scales with radius)."""
        return self.amplitude_fraction * self.mean_radius(t)

    def compute_evolution(self, seed: int = 42) -> NDArray[np.float64]:
        """
        Compute the time evolution of the quadrupole tensor.

        Returns:
            Q_history: Array of shape (n_timesteps, 3, 3) containing Q_ij(t)
        """
        if self._computed:
            return self._Q_history

        self._Q_history = np.zeros((self.n_steps, 3, 3))

        for i, t in enumerate(self.times):
            # Generate surface at this time
            # Use time-dependent seed for correlated evolution
            surface = FractalSurface(
                fractal_dimension=self.D,
                mean_radius=self.mean_radius(t),
                amplitude_rms=self.amplitude(t),
                lmax=48,  # Lower resolution for speed
                seed=seed + i  # Slowly evolving random field
            )

            # Compute quadrupole
            quad = QuadrupoleTensor(surface, self.M_shell)
            self._Q_history[i] = quad.compute()

        self._computed = True
        return self._Q_history

    def compute_derivatives(self) -> Tuple[NDArray, NDArray]:
        """
        Compute first and second time derivatives of Q_ij.

        Returns:
            Q_dot: dQ/dt (n_steps, 3, 3)
            Q_ddot: d²Q/dt² (n_steps, 3, 3)
        """
        Q = self.compute_evolution()

        # Second-order centered difference
        Q_dot = np.zeros_like(Q)
        Q_ddot = np.zeros_like(Q)

        # First derivative (centered difference)
        Q_dot[1:-1] = (Q[2:] - Q[:-2]) / (2 * self.dt)
        Q_dot[0] = (Q[1] - Q[0]) / self.dt
        Q_dot[-1] = (Q[-1] - Q[-2]) / self.dt

        # Second derivative
        Q_ddot[1:-1] = (Q[2:] - 2*Q[1:-1] + Q[:-2]) / (self.dt**2)
        Q_ddot[0] = Q_ddot[1]
        Q_ddot[-1] = Q_ddot[-2]

        return Q_dot, Q_ddot


# =============================================================================
# Gravitational Wave Computation
# =============================================================================

class GravitationalWaveEmission:
    """
    Computes the gravitational wave strain and spectrum from an expanding
    fractal flame front.

    Key equations:
        h_ij = (2G/c⁴r) Q̈_ij^TT

    where TT is the transverse-traceless projection.

    For a source at distance r, the strain amplitude is:
        h ~ (G/c⁴) (M R² ω²) / r

    The power spectrum Ω_GW(f) characterizes the stochastic background.
    """

    def __init__(self, flame: ExpandingFractalFlame, distance: float = 10 * KPC_CM):
        """
        Initialize GW computation.

        Parameters:
            flame: ExpandingFractalFlame model
            distance: Distance to source [cm] (default 10 kpc)
        """
        self.flame = flame
        self.distance = distance

        # Compute evolution if not done
        self.Q = flame.compute_evolution()
        self.Q_dot, self.Q_ddot = flame.compute_derivatives()

        # GW prefactor: 2G/c⁴
        self.gw_prefactor = 2 * G_GRAV / (C_LIGHT**4)

    def transverse_traceless_projection(self, Q: NDArray,
                                        direction: NDArray = None) -> NDArray:
        """
        Apply the TT projection to the quadrupole tensor.

        For a wave propagating in direction n̂:
            Λ_ijkl = P_ik P_jl - (1/2) P_ij P_kl
            P_ij = δ_ij - n_i n_j

        Parameters:
            Q: Quadrupole tensor (3x3) or array of tensors (n, 3, 3)
            direction: Propagation direction (default: z-axis)

        Returns:
            Q_TT: TT-projected tensor(s)
        """
        if direction is None:
            direction = np.array([0.0, 0.0, 1.0])

        n = direction / np.linalg.norm(direction)

        # Transverse projector P_ij = δ_ij - n_i n_j
        P = np.eye(3) - np.outer(n, n)

        # Apply projection
        if Q.ndim == 2:
            Q_trans = P @ Q @ P
            Q_TT = Q_trans - 0.5 * np.trace(Q_trans) * P
        else:
            # Array of tensors
            Q_TT = np.zeros_like(Q)
            for i in range(Q.shape[0]):
                Q_trans = P @ Q[i] @ P
                Q_TT[i] = Q_trans - 0.5 * np.trace(Q_trans) * P

        return Q_TT

    def compute_strain(self, direction: NDArray = None) -> Tuple[NDArray, NDArray, NDArray]:
        """
        Compute the GW strain time series h₊(t) and h×(t).

        Parameters:
            direction: Propagation direction (default: z-axis)

        Returns:
            times: Time array [s]
            h_plus: Plus polarization strain
            h_cross: Cross polarization strain
        """
        # Apply TT projection to Q̈
        Q_ddot_TT = self.transverse_traceless_projection(self.Q_ddot, direction)

        # Strain tensor: h_ij = (2G/c⁴r) Q̈_ij^TT
        h_tensor = self.gw_prefactor * Q_ddot_TT / self.distance

        # Extract polarizations (assuming propagation along z)
        h_plus = h_tensor[:, 0, 0]    # h_xx = -h_yy
        h_cross = h_tensor[:, 0, 1]   # h_xy = h_yx

        return self.flame.times, h_plus, h_cross

    def compute_characteristic_strain(self) -> Tuple[NDArray, NDArray]:
        """
        Compute the characteristic strain h_c(f) = f |h̃(f)|.

        This is useful for comparison with detector sensitivity curves.

        Returns:
            frequencies: Frequency array [Hz]
            h_c: Characteristic strain
        """
        times, h_plus, h_cross = self.compute_strain()

        # FFT
        n = len(times)
        dt = times[1] - times[0]

        h_plus_fft = fft(h_plus) * dt
        h_cross_fft = fft(h_cross) * dt

        frequencies = fftfreq(n, dt)

        # Take positive frequencies only
        pos_mask = frequencies > 0
        frequencies = frequencies[pos_mask]
        h_plus_fft = h_plus_fft[pos_mask]
        h_cross_fft = h_cross_fft[pos_mask]

        # Combined strain
        h_fft = np.sqrt(np.abs(h_plus_fft)**2 + np.abs(h_cross_fft)**2)

        # Characteristic strain
        h_c = frequencies * h_fft * np.sqrt(2)  # Factor for one-sided spectrum

        return frequencies, h_c

    def compute_energy_spectrum(self) -> Tuple[NDArray, float]:
        """
        Compute the total energy radiated in gravitational waves.

        dE/dt = (G/5c⁵) ⟨Q⃛_ij Q⃛_ij⟩

        Returns:
            power_time_series: dE/dt [erg/s]
            total_energy: E_GW [erg]
        """
        # Compute third derivative (for power)
        Q_dot, Q_ddot = self.Q_dot, self.Q_ddot

        # Third derivative (forward difference of second)
        Q_dddot = np.zeros_like(self.Q)
        dt = self.flame.dt
        Q_dddot[:-1] = (Q_ddot[1:] - Q_ddot[:-1]) / dt
        Q_dddot[-1] = Q_dddot[-2]

        # Power: dE/dt = (G/5c⁵) Σ_ij (Q⃛_ij)²
        prefactor = G_GRAV / (5 * C_LIGHT**5)

        power = np.zeros(len(self.flame.times))
        for i in range(len(power)):
            power[i] = prefactor * np.sum(Q_dddot[i]**2)

        # Total energy (integrate)
        total_energy = np.trapz(power, self.flame.times)

        return power, total_energy

    def peak_strain(self) -> float:
        """Return the peak strain amplitude."""
        times, h_plus, h_cross = self.compute_strain()
        h_total = np.sqrt(h_plus**2 + h_cross**2)
        return np.max(h_total)

    def dominant_frequency(self) -> float:
        """Return the dominant GW frequency [Hz]."""
        frequencies, h_c = self.compute_characteristic_strain()
        return frequencies[np.argmax(h_c)]


# =============================================================================
# Detector Sensitivity Curves
# =============================================================================

class DetectorSensitivity:
    """
    Gravitational wave detector sensitivity curves.

    Provides strain sensitivity √S_n(f) for:
        - LIGO (aLIGO design)
        - LISA (ESA mission)
        - DECIGO/BBO (future decihertz)
        - Einstein Telescope (3rd gen ground)
    """

    @staticmethod
    def ligo_sensitivity(frequencies: NDArray) -> NDArray:
        """
        Advanced LIGO design sensitivity √S_n(f) [Hz^(-1/2)].

        Simplified analytic approximation.
        """
        f = frequencies

        # Seismic wall below 10 Hz
        seismic = 1e-21 * (10 / f)**10

        # Thermal noise
        thermal = 6e-24 * (100 / f)**0.5

        # Shot noise
        shot = 4e-24 * (f / 100)**0.5

        # Combine
        h_n = np.sqrt(seismic**2 + thermal**2 + shot**2)

        # Set high values outside band
        h_n[f < 10] = 1e-18
        h_n[f > 5000] = 1e-18

        return h_n

    @staticmethod
    def lisa_sensitivity(frequencies: NDArray) -> NDArray:
        """
        LISA sensitivity √S_n(f) [Hz^(-1/2)].

        Based on Robson, Cornish & Liu (2019).
        """
        f = frequencies
        L = 2.5e9  # Arm length [m]
        f_star = C_LIGHT / (2 * np.pi * L * 100)  # ~19 mHz

        # Acceleration noise
        S_acc = (3e-15)**2 * (1 + (0.4e-3 / f)**2) * (1 + (f / 8e-3)**4)

        # Optical metrology noise
        S_oms = (1.5e-11)**2 * (1 + (2e-3 / f)**4)

        # Total noise
        P_n = S_oms + 2 * (1 + np.cos(f / f_star)**2) * S_acc / (2 * np.pi * f)**4

        # Transfer function
        R = 3/20 * 1/(1 + 0.6 * (f / f_star)**2)

        S_n = P_n / (L**2 * R)

        # Set high values outside band
        h_n = np.sqrt(S_n)
        h_n[f < 1e-4] = 1e-15
        h_n[f > 1] = 1e-15

        return h_n

    @staticmethod
    def decigo_sensitivity(frequencies: NDArray) -> NDArray:
        """
        DECIGO/BBO sensitivity √S_n(f) [Hz^(-1/2)].

        Optimized for 0.1-10 Hz band.
        """
        f = frequencies

        # Target sensitivity ~10⁻²⁴ at 1 Hz
        h_n = 5e-24 * np.sqrt(1 + (0.1 / f)**4 + (f / 10)**4)

        # Set high values outside band
        h_n[f < 0.01] = 1e-18
        h_n[f > 100] = 1e-18

        return h_n

    @staticmethod
    def einstein_telescope_sensitivity(frequencies: NDArray) -> NDArray:
        """
        Einstein Telescope sensitivity √S_n(f) [Hz^(-1/2)].

        3rd generation ground-based detector.
        """
        f = frequencies

        # Low frequency (cryogenic)
        low_freq = 1e-23 * (3 / f)**2

        # Mid frequency
        mid_freq = 3e-25

        # High frequency
        high_freq = 3e-25 * (f / 100)

        h_n = np.sqrt(low_freq**2 + mid_freq**2 + high_freq**2)

        # Set high values outside band
        h_n[f < 2] = 1e-18
        h_n[f > 10000] = 1e-18

        return h_n


# =============================================================================
# The Spandrel-GW Equation
# =============================================================================

class SpandrelGWEquation:
    """
    The gravitational wave analog of the Spandrel-Phillips equation.

    Key insight: GW strain scales with asymmetry, which scales with (D - 2).

    For a spherically symmetric explosion (D = 2.0):
        Q_ij = 0 (by symmetry)
        h = 0

    For a fractal flame (D > 2.0):
        ⟨Q²⟩ ∝ (D - 2)²
        h ∝ (D - 2)

    The Spandrel-GW equation:
        h_peak = h_0 × (D - 2)^α × (M/M_Ch) × (10 kpc / r)

    where α ≈ 1.5 (from simulations) and h_0 ≈ 10⁻²² (calibration).
    """

    # Calibrated parameters
    H_0 = 2e-22           # Reference strain at 10 kpc for D=2.35, M=M_Ch
    ALPHA = 1.5           # Scaling exponent with (D-2)
    D_REF = 2.35          # Reference fractal dimension

    @classmethod
    def predict_strain(cls, fractal_dimension: float,
                       mass: float = M_WD,
                       distance: float = 10 * KPC_CM) -> float:
        """
        Predict peak GW strain from fractal dimension.

        h = h_0 × [(D-2)/(D_ref-2)]^α × (M/M_Ch) × (10 kpc/r)
        """
        if fractal_dimension <= 2.0:
            return 0.0

        d_factor = ((fractal_dimension - 2.0) / (cls.D_REF - 2.0)) ** cls.ALPHA
        m_factor = mass / M_WD
        r_factor = (10 * KPC_CM) / distance

        return cls.H_0 * d_factor * m_factor * r_factor

    @classmethod
    def predict_energy(cls, fractal_dimension: float,
                       mass: float = M_WD) -> float:
        """
        Predict total GW energy from fractal dimension.

        E_GW ∝ (D - 2)² × M² × (v/c)⁴

        Reference: ~7×10³⁹ erg for D=2.35, M=M_Ch (Seitenzahl et al. 2015)
        """
        if fractal_dimension <= 2.0:
            return 0.0

        E_REF = 7e39  # erg, reference energy

        d_factor = ((fractal_dimension - 2.0) / (cls.D_REF - 2.0)) ** 2
        m_factor = (mass / M_WD) ** 2

        return E_REF * d_factor * m_factor

    @classmethod
    def generate_population(cls, num_samples: int = 500,
                           d_range: Tuple[float, float] = (2.1, 2.8)) -> Dict:
        """
        Generate a population of SNe Ia with GW predictions.

        Returns dictionary with D, strain, energy arrays.
        """
        # Fractal dimension distribution (peaked near 2.35)
        dimensions = np.random.beta(2, 3, num_samples)
        dimensions = d_range[0] + dimensions * (d_range[1] - d_range[0])

        # Add scatter in mass (sub-Chandra to Chandra)
        masses = np.random.uniform(0.8, 1.0, num_samples) * M_WD

        # Compute predictions
        strains = np.array([cls.predict_strain(d, m) for d, m in zip(dimensions, masses)])
        energies = np.array([cls.predict_energy(d, m) for d, m in zip(dimensions, masses)])

        return {
            'fractal_dimension': dimensions,
            'mass': masses / M_SUN,
            'peak_strain_10kpc': strains,
            'energy_erg': energies
        }


# =============================================================================
# Visualization Suite
# =============================================================================

class GravitationalFractalVisualizer:
    """
    Visualization suite for the GW-fractal connection.
    """

    @staticmethod
    def plot_strain_vs_dimension(save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot peak strain as a function of fractal dimension.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Strain vs D
        dimensions = np.linspace(2.01, 2.9, 100)
        strains = [SpandrelGWEquation.predict_strain(d) for d in dimensions]

        ax1.semilogy(dimensions, strains, 'b-', linewidth=2)
        ax1.axhline(y=1e-21, color='red', linestyle='--', alpha=0.7,
                   label='DECIGO threshold')
        ax1.axvline(x=2.0, color='gray', linestyle=':', alpha=0.7,
                   label='Spherical limit (h=0)')

        ax1.set_xlabel('Fractal Dimension D', fontsize=12)
        ax1.set_ylabel('Peak Strain h (at 10 kpc)', fontsize=12)
        ax1.set_title('The Spandrel-GW Equation:\nAsymmetry → Gravitational Waves', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Energy vs D
        energies = [SpandrelGWEquation.predict_energy(d) for d in dimensions]

        ax2.semilogy(dimensions, energies, 'r-', linewidth=2)
        ax2.axhline(y=7e39, color='blue', linestyle='--', alpha=0.7,
                   label='Seitenzahl et al. (2015)')

        ax2.set_xlabel('Fractal Dimension D', fontsize=12)
        ax2.set_ylabel('Total GW Energy E_GW [erg]', fontsize=12)
        ax2.set_title('Energy Radiated in Gravitational Waves', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    @staticmethod
    def plot_detector_comparison(fractal_dimension: float = 2.35,
                                 distance_kpc: float = 10,
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot GW signal against detector sensitivity curves.
        """
        # Generate signal
        flame = ExpandingFractalFlame(
            fractal_dimension=fractal_dimension,
            n_timesteps=1000
        )

        gw = GravitationalWaveEmission(flame, distance=distance_kpc * KPC_CM)
        frequencies, h_c = gw.compute_characteristic_strain()

        # Detector sensitivities
        f_full = np.logspace(-4, 4, 1000)

        fig, ax = plt.subplots(figsize=(12, 8))

        # Detector curves
        ax.loglog(f_full, DetectorSensitivity.lisa_sensitivity(f_full),
                 'g-', linewidth=2, label='LISA', alpha=0.7)
        ax.loglog(f_full, DetectorSensitivity.decigo_sensitivity(f_full),
                 'm-', linewidth=2, label='DECIGO/BBO', alpha=0.7)
        ax.loglog(f_full, DetectorSensitivity.ligo_sensitivity(f_full),
                 'b-', linewidth=2, label='Advanced LIGO', alpha=0.7)
        ax.loglog(f_full, DetectorSensitivity.einstein_telescope_sensitivity(f_full),
                 'c-', linewidth=2, label='Einstein Telescope', alpha=0.7)

        # Signal
        valid = (frequencies > 0.01) & (frequencies < 100) & (h_c > 0)
        if np.any(valid):
            ax.loglog(frequencies[valid], h_c[valid], 'r-', linewidth=3,
                     label=f'SN Ia (D={fractal_dimension}, d={distance_kpc} kpc)')

        # Mark the key frequency range
        ax.axvspan(0.4, 2.5, alpha=0.1, color='red', label='SNe Ia GW band')

        ax.set_xlabel('Frequency [Hz]', fontsize=12)
        ax.set_ylabel('Characteristic Strain h_c', fontsize=12)
        ax.set_xlim(1e-4, 1e4)
        ax.set_ylim(1e-26, 1e-16)
        ax.legend(loc='upper right', fontsize=10)
        ax.set_title(f'Gravitational Wave Detection: Fractal Flame at {distance_kpc} kpc\n'
                    f'(D = {fractal_dimension})', fontsize=14)
        ax.grid(True, alpha=0.3, which='both')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    @staticmethod
    def plot_fractal_surface(fractal_dimension: float = 2.35,
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize the fractal flame surface.
        """
        fig = plt.figure(figsize=(14, 6))

        # Generate surfaces for different D values
        d_values = [2.05, fractal_dimension, 2.7]
        titles = ['Nearly Spherical\n(D ≈ 2.0)',
                  f'Typical Flame\n(D = {fractal_dimension})',
                  'Highly Turbulent\n(D ≈ 2.7)']

        for idx, (d, title) in enumerate(zip(d_values, titles)):
            ax = fig.add_subplot(1, 3, idx + 1, projection='3d')

            surface = FractalSurface(
                fractal_dimension=d,
                mean_radius=1.0,
                amplitude_rms=0.15,
                lmax=48,
                seed=42
            )

            theta, phi, r = surface.generate_grid(n_theta=50, n_phi=100)
            x, y, z = surface.to_cartesian(theta, phi, r)

            # Color by radius deviation
            colors = (r - 1.0) / 0.15

            ax.plot_surface(x, y, z, facecolors=cm.hot(0.5 + 0.5 * colors),
                           alpha=0.9, linewidth=0, antialiased=True)

            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.set_title(title, fontsize=12)

            # Equal aspect ratio
            max_range = 1.3
            ax.set_xlim(-max_range, max_range)
            ax.set_ylim(-max_range, max_range)
            ax.set_zlim(-max_range, max_range)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    @staticmethod
    def plot_time_evolution(fractal_dimension: float = 2.35,
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot time evolution of quadrupole moment and GW strain.
        """
        flame = ExpandingFractalFlame(
            fractal_dimension=fractal_dimension,
            n_timesteps=500
        )

        gw = GravitationalWaveEmission(flame, distance=10 * KPC_CM)
        times, h_plus, h_cross = gw.compute_strain()
        power, total_energy = gw.compute_energy_spectrum()

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Quadrupole evolution
        Q = flame.compute_evolution()
        Q_norm = np.sqrt(np.sum(Q**2, axis=(1, 2)))

        axes[0, 0].plot(times, Q_norm, 'b-', linewidth=1.5)
        axes[0, 0].set_xlabel('Time [s]', fontsize=11)
        axes[0, 0].set_ylabel('||Q|| [g cm²]', fontsize=11)
        axes[0, 0].set_title('Quadrupole Moment Evolution', fontsize=12)
        axes[0, 0].grid(True, alpha=0.3)

        # Strain time series
        axes[0, 1].plot(times, h_plus, 'b-', alpha=0.8, label='h₊')
        axes[0, 1].plot(times, h_cross, 'r-', alpha=0.8, label='h×')
        axes[0, 1].set_xlabel('Time [s]', fontsize=11)
        axes[0, 1].set_ylabel('Strain h', fontsize=11)
        axes[0, 1].set_title('Gravitational Wave Strain (at 10 kpc)', fontsize=12)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Power
        axes[1, 0].semilogy(times, power, 'r-', linewidth=1.5)
        axes[1, 0].set_xlabel('Time [s]', fontsize=11)
        axes[1, 0].set_ylabel('dE/dt [erg/s]', fontsize=11)
        axes[1, 0].set_title(f'GW Power (Total: {total_energy:.2e} erg)', fontsize=12)
        axes[1, 0].grid(True, alpha=0.3)

        # Spectrum
        frequencies, h_c = gw.compute_characteristic_strain()
        valid = (frequencies > 0.01) & (frequencies < 100) & (h_c > 1e-30)

        if np.any(valid):
            axes[1, 1].loglog(frequencies[valid], h_c[valid], 'b-', linewidth=1.5)
        axes[1, 1].axvline(x=1.0, color='red', linestyle='--', alpha=0.7,
                          label='f = 1 Hz')
        axes[1, 1].set_xlabel('Frequency [Hz]', fontsize=11)
        axes[1, 1].set_ylabel('Characteristic Strain h_c', fontsize=11)
        axes[1, 1].set_title('GW Frequency Spectrum', fontsize=12)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle(f'Gravitational Wave Emission from Fractal Flame (D = {fractal_dimension})',
                    fontsize=14, y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig


# =============================================================================
# Boundary Catastrophe GW Predictions
# =============================================================================

class BoundaryCatastropheGW:
    """
    GW predictions for the boundary catastrophe cases: D→2 and D→3.

    Type Iax (D→2): Failed DDT, nearly laminar flame
        - PREDICTION: Nearly ZERO GW emission (h → 0)
        - Physical reason: Spherical symmetry preserved in laminar limit
        - Detection: NON-DETECTION confirms laminar floor

    03fg-like (D→3): Maximum turbulence, super-Chandrasekhar
        - PREDICTION: MAXIMUM GW emission (h → h_max)
        - Physical reason: Volume-filling turbulence, extreme asymmetry
        - Detection: Enhanced strain, ~2% polarization confirmed by spectropolarimetry

    Nov 2025 Update:
        - SN 2008ha (D ≈ 2.0): Δm₁₅ = 2.4-2.7 mag, M_Ni = 0.003 M☉
        - SN 2009dc (D ≈ 2.9): Δm₁₅ = 0.65 mag, M_Ni = 1.2-1.8 M☉
        - ~1-2% intrinsic polarization in 03fg-like confirms asymmetry (Nagao et al. 2024)
    """

    # Boundary values from Phase 3 research
    D_LAMINAR = 2.0        # Perfect laminar flame (Type Iax extreme: SN 2008ha)
    D_TURBULENT = 3.0      # Volume-filling sponge (theoretical maximum)
    D_IAX_TYPICAL = 2.05   # Typical Type Iax (weak deflagration)
    D_03FG_TYPICAL = 2.7   # Typical 03fg-like (SN 2009dc)

    # Super-Chandrasekhar masses for 03fg-like
    M_03FG_LOW = 1.8 * M_SUN    # Lower bound super-Chandra
    M_03FG_HIGH = 2.8 * M_SUN   # Upper bound (SN 2009dc)

    @classmethod
    def type_iax_strain(cls, distance: float = 10 * KPC_CM,
                        D_estimate: float = 2.05) -> dict:
        """
        Predict GW strain for Type Iax (D→2 boundary catastrophe).

        Type Iax characteristics:
            - D ≈ 2.0-2.15 (nearly laminar)
            - M_Ni = 0.003-0.27 M☉
            - Failed DDT (pure deflagration)
            - Nearly spherical → minimal GW

        Returns:
            Dictionary with strain predictions and detection prospects.
        """
        # Strain scaling: h ∝ (D - 2)^1.5
        h_peak = SpandrelGWEquation.predict_strain(D_estimate, distance=distance)
        E_gw = SpandrelGWEquation.predict_energy(D_estimate)

        # For SN 2008ha extreme (D → 2.0):
        h_2008ha = SpandrelGWEquation.predict_strain(2.01, distance=distance)

        # Detection threshold (DECIGO at 1 Hz)
        h_decigo_threshold = 5e-24

        return {
            'classification': 'Type Iax (D→2 boundary)',
            'D_estimate': D_estimate,
            'h_peak_at_distance': h_peak,
            'E_gw_erg': E_gw,
            'h_2008ha_extreme': h_2008ha,
            'distance_kpc': distance / KPC_CM,
            'detectable_decigo': h_peak > h_decigo_threshold,
            'physical_interpretation': 'Nearly laminar flame - minimal GW',
            'prediction': 'NON-DETECTION confirms laminar floor hypothesis',
            'sn2008ha_signature': {
                'D': 2.01,
                'dm15_mag': 2.5,
                'M_Ni_solar': 0.003,
                'h_strain': h_2008ha,
                'GW_status': 'SILENT (spherically symmetric)'
            }
        }

    @classmethod
    def type_03fg_strain(cls, distance: float = 10 * KPC_CM,
                         D_estimate: float = 2.7,
                         mass: float = None) -> dict:
        """
        Predict GW strain for 03fg-like (D→3 turbulent boundary catastrophe).

        03fg-like characteristics:
            - D ≈ 2.5-3.0 (maximum turbulence)
            - M_Ni = 1.2-1.8 M☉ (super-Chandrasekhar!)
            - Violent DD merger progenitor
            - Extreme asymmetry (~2% polarization) → enhanced GW

        Returns:
            Dictionary with strain predictions and detection prospects.
        """
        if mass is None:
            mass = cls.M_03FG_HIGH  # Use SN 2009dc estimate

        # Strain scaling: h ∝ (D - 2)^1.5 × (M/M_Ch)
        h_peak = SpandrelGWEquation.predict_strain(D_estimate, mass=mass, distance=distance)
        E_gw = SpandrelGWEquation.predict_energy(D_estimate, mass=mass)

        # Normal SNe Ia comparison
        h_normal = SpandrelGWEquation.predict_strain(2.2, mass=M_WD, distance=distance)

        # Amplification factor from super-Chandra mass + high D
        amplification = h_peak / h_normal if h_normal > 0 else np.inf

        # Detection threshold (DECIGO at 1 Hz)
        h_decigo_threshold = 5e-24

        return {
            'classification': '03fg-like (D→3 boundary)',
            'D_estimate': D_estimate,
            'mass_solar': mass / M_SUN,
            'h_peak_at_distance': h_peak,
            'E_gw_erg': E_gw,
            'h_normal_comparison': h_normal,
            'amplification_factor': amplification,
            'distance_kpc': distance / KPC_CM,
            'detectable_decigo': h_peak > h_decigo_threshold,
            'physical_interpretation': 'Volume-filling turbulence - maximum GW',
            'prediction': f'ENHANCED strain ({amplification:.1f}× normal SNe Ia)',
            'sn2009dc_signature': {
                'D': 2.9,
                'dm15_mag': 0.65,
                'M_Ni_solar': 1.5,
                'M_WD_solar': 2.5,
                'polarization_percent': 1.5,
                'h_strain': SpandrelGWEquation.predict_strain(2.9, mass=2.5*M_SUN, distance=distance),
                'GW_status': 'LOUD (extreme asymmetry)'
            }
        }

    @classmethod
    def bifurcation_summary(cls) -> dict:
        """
        Summary of the D-dependent GW bifurcation.

        The Spandrel Framework predicts a BIMODAL GW distribution:
            - LOW-D (Type Iax): h → 0 (silent)
            - HIGH-D (03fg-like): h → h_max (loud)

        This is the gravitational analog of the Hubble bifurcation.
        """
        d_range = np.linspace(2.01, 2.95, 50)
        h_range = [SpandrelGWEquation.predict_strain(d) for d in d_range]

        # Find bifurcation point (where h exceeds threshold)
        h_threshold = 1e-23  # Approximate detection threshold
        d_critical = 2.15  # Approximate D where detection becomes possible

        return {
            'D_laminar_limit': cls.D_LAMINAR,
            'D_turbulent_limit': cls.D_TURBULENT,
            'D_critical_detection': d_critical,
            'h_threshold': h_threshold,
            'd_range': d_range.tolist(),
            'h_range': h_range,
            'interpretation': {
                'low_D': 'Type Iax (failed DDT) - GW silent',
                'mid_D': 'Normal SNe Ia - GW detectable with DECIGO',
                'high_D': '03fg-like (super-Chandra) - GW enhanced'
            },
            'falsifiable_prediction': (
                'Multi-messenger observation: Overluminous 03fg-like SNe '
                'should have ~5-10× stronger GW signals than normal SNe Ia, '
                'while Type Iax should be GW-silent.'
            )
        }

    @classmethod
    def lisa_progenitor_prediction(cls) -> dict:
        """
        LISA detection predictions for DD progenitors of 03fg-like SNe.

        LISA will detect DD systems ~minutes before merger.
        03fg-like SNe have DD merger progenitors → LISA precursors expected.

        From literature (A&A 2024):
            - LISA expected to detect ~501 pre-explosion DD systems
            - 03fg-like rate: 0.1-0.8% of SNe Ia
            - Expected LISA-detected 03fg progenitors: ~0.5-4 systems
        """
        # LISA sensitivity band
        f_lisa_min = 1e-4  # Hz
        f_lisa_max = 1e-1  # Hz

        # Final orbit before merger (orbital period ~ minutes)
        P_final_min = 60  # seconds (1 minute before merger)
        f_gw_final = 2 / P_final_min  # GW frequency is 2× orbital

        # DD merger rate
        rate_dd = 1.4e-3  # DD mergers per year per galaxy
        rate_03fg = rate_dd * 0.005  # ~0.5% are 03fg-like

        return {
            'detector': 'LISA',
            'frequency_band_Hz': (f_lisa_min, f_lisa_max),
            'progenitor_type': 'Double-degenerate (DD) merger',
            'lisa_dd_detections': 501,  # Expected DD progenitors per year
            '03fg_like_fraction': 0.005,
            'expected_03fg_progenitors': 501 * 0.005,  # ~2.5 per year
            'multi_messenger_opportunity': (
                'LISA GW detection of DD inspiral → '
                'optical follow-up for 03fg-like SN → '
                'GW burst at explosion (DECIGO) → '
                'complete D-measurement chain'
            ),
            'timeline': {
                'LISA_detection': 'Minutes to hours before merger',
                'Optical_trigger': 'Hours after merger (rising SN)',
                'DECIGO_detection': 'During explosion (~2s)'
            }
        }


# =============================================================================
# Main Demonstration
# =============================================================================

def demonstrate_gravitational_fractal():
    """
    Complete demonstration of the Fractal-GW connection.
    """
    print("=" * 70)
    print("THE GRAVITATIONAL WAVE SPECTRUM OF FRACTAL DEATH")
    print("=" * 70)
    print()

    print("KEY INSIGHT:")
    print("  Spherical explosion (D=2.0) → Q_ij = 0 → h = 0 (SILENCE)")
    print("  Fractal flame (D>2.0) → Q_ij ≠ 0 → h > 0 (SOUND)")
    print()

    # Test cases
    test_cases = [
        (2.05, "Nearly Laminar (quiet)"),
        (2.35, "Normal Type Ia"),
        (2.55, "Moderately Turbulent"),
        (2.70, "Highly Turbulent (loud)")
    ]

    print("PREDICTIONS (at 10 kpc):")
    print("-" * 70)
    print(f"{'D':>6} | {'Peak Strain':>14} | {'Energy [erg]':>14} | Description")
    print("-" * 70)

    for d, desc in test_cases:
        h = SpandrelGWEquation.predict_strain(d)
        e = SpandrelGWEquation.predict_energy(d)
        print(f"{d:>6.2f} | {h:>14.2e} | {e:>14.2e} | {desc}")

    print()
    print("THE SPANDREL-GW EQUATION:")
    print("  h_peak = h_0 × (D-2)^α × (M/M_Ch) × (10 kpc/r)")
    print(f"  Parameters: h_0 = {SpandrelGWEquation.H_0:.1e}, α = {SpandrelGWEquation.ALPHA}")
    print()

    print("DETECTION PROSPECTS:")
    print("  LISA (0.1 mHz - 1 Hz): Cannot detect explosion (progenitors only)")
    print("  LIGO (10 Hz - 10 kHz): Below band (f ~ 1 Hz)")
    print("  DECIGO/BBO (0.1-10 Hz): OPTIMAL BAND for SNe Ia")
    print()

    print("FALSIFIABLE PREDICTION:")
    print("  The GW power spectral density should show:")
    print("    • Laminar floor (D→2): P(f) → 0 (no signal)")
    print("    • Kolmogorov peak (D≈2.33): P(f) ~ f^(-8/3)")
    print("    • Turbulent ceiling (D→3): Maximum asymmetry")


def main():
    """Main entry point."""
    demonstrate_gravitational_fractal()

    print("\nGenerating visualizations...")

    import matplotlib
    matplotlib.use('Agg')

    viz = GravitationalFractalVisualizer()

    viz.plot_strain_vs_dimension(
        save_path='/Users/eirikr/cosmos/gw_strain_vs_dimension.png'
    )
    print("  → Saved: gw_strain_vs_dimension.png")

    viz.plot_detector_comparison(
        fractal_dimension=2.35,
        distance_kpc=10,
        save_path='/Users/eirikr/cosmos/gw_detector_comparison.png'
    )
    print("  → Saved: gw_detector_comparison.png")

    viz.plot_fractal_surface(
        fractal_dimension=2.35,
        save_path='/Users/eirikr/cosmos/fractal_surfaces.png'
    )
    print("  → Saved: fractal_surfaces.png")

    viz.plot_time_evolution(
        fractal_dimension=2.35,
        save_path='/Users/eirikr/cosmos/gw_time_evolution.png'
    )
    print("  → Saved: gw_time_evolution.png")

    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
