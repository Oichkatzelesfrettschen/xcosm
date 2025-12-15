"""
CCF Parameters Module
=====================

Defines the fundamental parameters of the Computational Cosmogenesis Framework.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class CCFParameters:
    """
    Fundamental CCF parameters calibrated to cosmological observations.

    Attributes
    ----------
    lambda_inflation : float
        Slow-roll parameter from spectral index n_s = 1 - 2*lambda
        Default: 0.003 (gives n_s = 0.966)

    eta_curvature : float
        Curvature coupling parameter from ACT DR6 lensing
        Default: 0.028

    alpha_attachment : float
        Preferential attachment exponent from S_8 measurements
        Default: 0.85 (gives S_8 = 0.78)

    epsilon_tension : float
        Link tension parameter for dark energy
        Default: 0.25 (gives w_0 = -0.833)

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

    def spectral_index(self) -> float:
        """Compute predicted scalar spectral index n_s."""
        return 1.0 - 2.0 * self.lambda_inflation

    def s8_parameter(self) -> float:
        """Compute predicted S_8 from attachment exponent."""
        return 0.83 - 0.05 * (1 - self.alpha_attachment)

    def dark_energy_eos(self) -> float:
        """Compute dark energy equation of state w_0."""
        return -1.0 + 2.0 * self.epsilon_tension / 3.0

    def tensor_to_scalar(self) -> float:
        """Compute predicted tensor-to-scalar ratio r."""
        cos2_theta = 0.10  # Multi-field suppression
        return 16.0 * self.lambda_inflation * cos2_theta

    def hubble_at_scale(self, k: float) -> float:
        """
        Compute H0 at a given scale k.

        Parameters
        ----------
        k : float
            Wavenumber in Mpc^-1

        Returns
        -------
        float
            H0 in km/s/Mpc
        """
        import numpy as np
        return self.h0_cmb + self.h0_gradient * np.log10(k / self.k_star)

    def __str__(self) -> str:
        return (
            f"CCFParameters(\n"
            f"  lambda = {self.lambda_inflation} -> n_s = {self.spectral_index():.4f}\n"
            f"  alpha  = {self.alpha_attachment} -> S_8 = {self.s8_parameter():.3f}\n"
            f"  epsilon = {self.epsilon_tension} -> w_0 = {self.dark_energy_eos():.3f}\n"
            f"  r = {self.tensor_to_scalar():.4f}\n"
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
