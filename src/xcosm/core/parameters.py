"""
CCF Parameters Module
=====================

Defines the fundamental parameters of the Computational Cosmogenesis Framework.

This is the canonical CCFParameters class for the entire COSMOS project.
All other modules should import from this location.

Planck Units Interface
----------------------
The CCFParameters class provides dual interfaces for all parameters:
- Standard units (Mpc^-1, km/s/Mpc) for observational cosmology
- Planck units (dimensionless) for fundamental physics

All Planck-unit properties end with '_planck' suffix.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from xcosm.core.planck_units import (
    H0_from_planck,
    H0_to_planck,
    to_planck_length,
)


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

    Planck Unit Interface
    ---------------------
    The following properties provide Planck-unit representations:

    k_star_planck : float (property)
        Crossover scale in Planck units (k/ℓ_P⁻¹)

    h0_cmb_planck : float (property)
        CMB-scale Hubble constant in Planck units (H₀ t_P)

    h0_gradient_planck : float (property)
        H0 gradient in Planck units

    hubble_at_scale_planck(k_planck) : method
        Compute H0 at a given scale in Planck units

    from_planck(...) : class method
        Construct CCFParameters from Planck-unit inputs
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
    # PLANCK UNIT PROPERTIES
    # =========================================================================

    @property
    def k_star_planck(self) -> float:
        """
        Crossover scale in Planck units (k/ℓ_P⁻¹).

        Converts k_star from Mpc⁻¹ to Planck units.

        Returns
        -------
        float
            Crossover scale k_star in Planck units (dimensionless)
        """
        # k in Mpc⁻¹ -> k in ℓ_P⁻¹
        # k[ℓ_P⁻¹] = k[Mpc⁻¹] × (1 Mpc / ℓ_P)
        return self.k_star * to_planck_length(1.0, "Mpc")

    @property
    def h0_cmb_planck(self) -> float:
        """
        CMB-scale Hubble constant in Planck units (H₀ t_P).

        Converts h0_cmb from km/s/Mpc to dimensionless Planck units.

        Returns
        -------
        float
            H₀ in Planck units (dimensionless)
        """
        return H0_to_planck(self.h0_cmb)

    @property
    def h0_gradient_planck(self) -> float:
        """
        H0 gradient in Planck units.

        Converts h0_gradient from km/s/Mpc per decade to Planck units.

        Returns
        -------
        float
            H₀ gradient in Planck units (dimensionless)
        """
        return H0_to_planck(self.h0_gradient)

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

    def hubble_at_scale_planck(self, k_planck: float) -> float:
        """
        Compute H0 at a given scale k in Planck units.

        H₀(k) = H₀_CMB + m × log₁₀(k/k*)

        All quantities are in Planck units (dimensionless).

        Parameters
        ----------
        k_planck : float
            Wavenumber in Planck units (k/ℓ_P⁻¹)

        Returns
        -------
        float
            H0 in Planck units (H₀ t_P)
        """
        return self.h0_cmb_planck + self.h0_gradient_planck * np.log10(
            k_planck / self.k_star_planck
        )

    # =========================================================================
    # CONSTRUCTION FROM PLANCK UNITS
    # =========================================================================

    @classmethod
    def from_planck(
        cls,
        lambda_inflation: float = 0.003,
        eta_curvature: float = 0.028,
        alpha_attachment: float = 0.85,
        epsilon_tension: float = 0.25,
        k_star_planck: Optional[float] = None,
        h0_cmb_planck: Optional[float] = None,
        h0_gradient_planck: Optional[float] = None,
    ) -> "CCFParameters":
        """
        Construct CCFParameters from Planck-unit inputs.

        This class method allows creation of CCFParameters using Planck units
        for cosmological scales while maintaining backward compatibility with
        standard units.

        Parameters
        ----------
        lambda_inflation : float
            Slow-roll parameter (dimensionless)
            Default: 0.003
        eta_curvature : float
            Curvature coupling parameter (dimensionless)
            Default: 0.028
        alpha_attachment : float
            Preferential attachment exponent (dimensionless)
            Default: 0.85
        epsilon_tension : float
            Link tension parameter (dimensionless)
            Default: 0.25
        k_star_planck : float, optional
            Crossover scale in Planck units (k/ℓ_P⁻¹)
            If None, uses default k_star = 0.01 Mpc⁻¹
        h0_cmb_planck : float, optional
            CMB-scale Hubble constant in Planck units (H₀ t_P)
            If None, uses default h0_cmb = 67.4 km/s/Mpc
        h0_gradient_planck : float, optional
            H0 gradient in Planck units
            If None, uses default h0_gradient = 1.15 km/s/Mpc/decade

        Returns
        -------
        CCFParameters
            New parameter instance with values converted from Planck units

        Examples
        --------
        >>> # Create parameters with Planck-unit crossover scale
        >>> params = CCFParameters.from_planck(
        ...     k_star_planck=1e60,  # Some value in Planck units
        ...     h0_cmb_planck=COSMOLOGY_PLANCK.H0
        ... )
        """
        # Use defaults if Planck values not provided
        defaults = cls()

        # Convert k_star from Planck units to Mpc⁻¹
        if k_star_planck is not None:
            # k[Mpc⁻¹] = k[ℓ_P⁻¹] / (1 Mpc / ℓ_P)
            k_star = k_star_planck / to_planck_length(1.0, "Mpc")
        else:
            k_star = defaults.k_star

        # Convert h0_cmb from Planck units to km/s/Mpc
        if h0_cmb_planck is not None:
            h0_cmb = H0_from_planck(h0_cmb_planck)
        else:
            h0_cmb = defaults.h0_cmb

        # Convert h0_gradient from Planck units to km/s/Mpc
        if h0_gradient_planck is not None:
            h0_gradient = H0_from_planck(h0_gradient_planck)
        else:
            h0_gradient = defaults.h0_gradient

        return cls(
            lambda_inflation=lambda_inflation,
            eta_curvature=eta_curvature,
            alpha_attachment=alpha_attachment,
            epsilon_tension=epsilon_tension,
            k_star=k_star,
            h0_cmb=h0_cmb,
            h0_gradient=h0_gradient,
        )

    # =========================================================================
    # SERIALIZATION METHODS
    # =========================================================================

    def to_dict(self) -> dict:
        """
        Convert parameters to dictionary representation.

        Returns
        -------
        dict
            Dictionary with core parameters, derived observables, and Planck-unit values
        """
        return {
            "lambda_inflation": self.lambda_inflation,
            "eta_curvature": self.eta_curvature,
            "alpha_attachment": self.alpha_attachment,
            "epsilon_tension": self.epsilon_tension,
            "k_star": self.k_star,
            "h0_cmb": self.h0_cmb,
            "h0_gradient": self.h0_gradient,
            "derived": {
                "n_s": self.spectral_index,
                "r": self.tensor_to_scalar,
                "s8": self.s8_parameter,
                "w0": self.w0_dark_energy,
                "wa": self.wa_dark_energy,
                "h0_gradient": self.h0_gradient,
            },
            "planck_units": {
                "k_star_planck": self.k_star_planck,
                "h0_cmb_planck": self.h0_cmb_planck,
                "h0_gradient_planck": self.h0_gradient_planck,
            },
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
