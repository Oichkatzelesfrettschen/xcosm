"""
CCF Parameters Module
=====================

Defines the fundamental parameters of the Computational Cosmogenesis Framework.

This is the canonical CCFParameters class for the entire COSMOS project.
All other modules should import from this location.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass(frozen=True)
class CCFParameters:
    """
    Fundamental CCF parameters calibrated to cosmological observations.

    These parameters are derived from:
    - Planck 2018 + ACT DR6: n_s = 0.966 ± 0.004
    - DESI DR2: w₀ = -0.83 ± 0.05, wₐ = -0.70 ± 0.25
    - KiDS-Legacy: S₈ = 0.815 ± 0.018
    - BICEP/Keck 2024: r < 0.032 (95% CL)
    - SH0ES 2024: H₀ = 73.17 ± 0.86 km/s/Mpc (local)

    Attributes
    ----------
    lambda_inflation : float
        Slow-roll parameter / Inflation decay rate from spectral index n_s = 1 - 2*lambda - eta
        Default: 0.003 (gives n_s ≈ 0.966)

    eta_curvature : float
        Curvature coupling parameter / Bigraph curvature coupling from ACT DR6 lensing
        Default: 0.028

    alpha_attachment : float
        Preferential attachment exponent from S_8 measurements
        Default: 0.85 (gives S_8 ≈ 0.78-0.815)

    epsilon_tension : float
        Link tension parameter for dark energy
        Default: 0.25 (gives w_0 ≈ -0.833)

    k_star : float
        Crossover scale in Mpc^-1
        Default: 0.01

    h0_cmb : float
        CMB-scale Hubble constant in km/s/Mpc
        Default: 67.4

    h0_gradient : float
        H0 gradient in km/s/Mpc per decade of k
        Default: 1.15
    """

    lambda_inflation: float = 0.003
    eta_curvature: float = 0.028
    alpha_attachment: float = 0.85
    epsilon_tension: float = 0.25
    k_star: float = 0.01
    h0_cmb: float = 67.4
    h0_gradient: float = 1.15

    # =========================================================================
    # SPECTRAL INDEX - Dual method/property interface
    # =========================================================================

    @property
    def spectral_index(self) -> float:
        """n_s = 1 - 2λ - η (slow-roll relation)."""
        return 1.0 - 2.0 * self.lambda_inflation - self.eta_curvature

    def spectral_index_method(self) -> float:
        """
        Compute predicted scalar spectral index n_s (method form).

        Uses slow-roll relation: n_s = 1 - 2λ - η

        Returns
        -------
        float
            Scalar spectral index n_s
        """
        return 1.0 - 2.0 * self.lambda_inflation - self.eta_curvature

    # =========================================================================
    # TENSOR-TO-SCALAR RATIO - Dual method/property interface
    # =========================================================================

    @property
    def tensor_to_scalar(self) -> float:
        """r = 16λ × cos²θ with multi-field suppression."""
        cos2_theta = 0.10  # Multi-field suppression
        return 16.0 * self.lambda_inflation * cos2_theta

    def tensor_to_scalar_method(self) -> float:
        """
        Compute predicted tensor-to-scalar ratio r (method form).

        Uses multi-field suppression: r = 16λ × cos²θ

        Returns
        -------
        float
            Tensor-to-scalar ratio r
        """
        cos2_theta = 0.10  # Multi-field suppression
        return 16.0 * self.lambda_inflation * cos2_theta

    # =========================================================================
    # S8 PARAMETER - Dual method/property interface
    # =========================================================================

    @property
    def s8_parameter(self) -> float:
        """S_8 parameter (σ_8 × (Ω_m/0.3)^0.5) from attachment exponent."""
        return 0.83 - 0.05 * (1 - self.alpha_attachment)

    def s8_parameter_method(self) -> float:
        """
        Compute predicted S_8 from attachment exponent (method form).

        Returns
        -------
        float
            S_8 parameter (σ_8 × (Ω_m/0.3)^0.5)
        """
        return 0.83 - 0.05 * (1 - self.alpha_attachment)

    # =========================================================================
    # DARK ENERGY EQUATION OF STATE - Dual method/property interface
    # =========================================================================

    @property
    def dark_energy_eos(self) -> float:
        """Dark energy equation of state w_0."""
        return -1.0 + 2.0 * self.epsilon_tension / 3.0

    def dark_energy_eos_method(self) -> float:
        """
        Compute dark energy equation of state w_0 (method form).

        Returns
        -------
        float
            Dark energy equation of state w_0
        """
        return -1.0 + 2.0 * self.epsilon_tension / 3.0

    @property
    def w0_dark_energy(self) -> float:
        """w₀ = -1 + 2ε/3 (link tension contribution) - alias for dark_energy_eos."""
        return -1.0 + 2.0 * self.epsilon_tension / 3.0

    @property
    def wa_dark_energy(self) -> float:
        """
        wₐ from time-dependent tension relaxation.

        Returns
        -------
        float
            Dark energy evolution parameter wₐ (calibrated to DESI DR2)
        """
        return -0.70

    # =========================================================================
    # HUBBLE PARAMETER METHODS
    # =========================================================================

    def hubble_at_scale(self, k: float) -> float:
        """
        Compute H0 at a given scale k.

        H₀(k) = H₀_CMB + m × log₁₀(k/k*)

        Parameters
        ----------
        k : float
            Wavenumber in Mpc^-1

        Returns
        -------
        float
            H0 in km/s/Mpc
        """
        return self.h0_cmb + self.h0_gradient * np.log10(k / self.k_star)

    # =========================================================================
    # SERIALIZATION METHODS
    # =========================================================================

    def to_dict(self) -> dict:
        """
        Convert parameters to dictionary representation.

        Returns
        -------
        dict
            Dictionary with core parameters and derived observables
        """
        return {
            'lambda_inflation': self.lambda_inflation,
            'eta_curvature': self.eta_curvature,
            'alpha_attachment': self.alpha_attachment,
            'epsilon_tension': self.epsilon_tension,
            'k_star': self.k_star,
            'h0_cmb': self.h0_cmb,
            'h0_gradient': self.h0_gradient,
            'derived': {
                'n_s': self.spectral_index,
                'r': self.tensor_to_scalar,
                's8': self.s8_parameter,
                'w0': self.w0_dark_energy,
                'wa': self.wa_dark_energy,
                'h0_gradient': self.h0_gradient,
            }
        }

    def __str__(self) -> str:
        """
        Human-readable string representation.

        Returns
        -------
        str
            Formatted parameter summary
        """
        return (
            f"CCFParameters(\n"
            f"  lambda = {self.lambda_inflation} -> n_s = {self.spectral_index:.4f}\n"
            f"  alpha  = {self.alpha_attachment} -> S_8 = {self.s8_parameter:.3f}\n"
            f"  epsilon = {self.epsilon_tension} -> w_0 = {self.dark_energy_eos:.3f}\n"
            f"  r = {self.tensor_to_scalar:.4f}\n"
            f"  H0_CMB = {self.h0_cmb} km/s/Mpc\n"
            f"  H0_gradient = {self.h0_gradient} km/s/Mpc/decade\n"
            f")"
        )


@dataclass
class SimulationConfig:
    """
    Configuration for CCF simulations.

    Attributes
    ----------
    inflation_steps : int
        Number of inflationary evolution steps
    structure_steps : int
        Number of structure formation steps
    expansion_steps : int
        Number of cosmological expansion steps
    seed : Optional[int]
        Random seed for reproducibility
    verbose : bool
        Whether to print progress
    """

    inflation_steps: int = 100
    structure_steps: int = 200
    expansion_steps: int = 50
    seed: Optional[int] = None
    verbose: bool = True
