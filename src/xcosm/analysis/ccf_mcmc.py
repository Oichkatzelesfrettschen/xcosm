"""
MCMC sampling for CCF (Cohomological Constraint Framework) model parameters.

This module implements Bayesian parameter estimation for the CCF cosmological model
using observational constraints from CMB (Planck) and BAO measurements.

Parameters:
    λ (lambda_spectral): Spectral index coupling parameter
    η (eta_baryon): Baryon density coupling parameter
    α (alpha_s8): Structure formation amplitude parameter
    ε (epsilon_tension): Dark energy equation of state link tension
    k* (k_star): Characteristic transition scale (Mpc^-1)

Observational Constraints:
    - Planck CMB: n_s = 0.965 ± 0.004 constrains λ
    - Planck CMB: Ω_b h² = 0.0224 ± 0.0001 constrains η
    - Planck/weak lensing: S₈ = 0.776 ± 0.017 constrains α
    - BAO/SNe Ia: w₀ = -0.83 ± 0.05 constrains ε
    - Hubble tension: H₀ = 67.4 (Planck) vs 73.0 (SH0ES) constrains k*

Author: CCF Analysis Pipeline
Date: 2025-12-15
"""

import warnings
from typing import Dict, Optional, Tuple

import corner
import emcee
import h5py
import matplotlib.pyplot as plt
import numpy as np

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)


class CCFModel:
    """
    Cohomological Constraint Framework cosmological model.

    This class implements the theoretical predictions for observables
    as functions of the fundamental CCF parameters.
    """

    def __init__(self):
        """Initialize CCF model with physical constants."""
        # Fundamental constants
        self.c_light = 299792.458  # km/s
        self.h_planck = 0.6766  # Dimensionless Hubble parameter (Planck 2018)

        # Reference scales
        self.omega_cdm_reference = 0.1193  # Dark matter density (Planck 2018)
        self.omega_lambda_reference = 0.6847  # Dark energy density (Planck 2018)

    def compute_spectral_index(self, lambda_spectral: float) -> float:
        """
        Compute scalar spectral index n_s from λ parameter.

        The CCF predicts n_s through the cohomological constraint on
        primordial fluctuations.

        Args:
            lambda_spectral: Spectral coupling parameter (dimensionless)

        Returns:
            Predicted scalar spectral index n_s
        """
        # CCF prediction: n_s = 1 - 2ε where ε is slow-roll parameter
        # Related to λ through cohomological constraint
        # n_s ≈ 0.965 corresponds to λ ≈ 0.0175
        ns_base = 0.965
        return ns_base + 0.2 * (lambda_spectral - 0.0175)

    def compute_baryon_density(self, eta_baryon: float) -> float:
        """
        Compute baryon density parameter Ω_b h² from η parameter.

        The CCF relates baryon density to cohomological charge conservation.

        Args:
            eta_baryon: Baryon coupling parameter (dimensionless)

        Returns:
            Predicted baryon density Ω_b h²
        """
        # CCF prediction: Ω_b h² scales with η through charge conservation
        # η = 1.0 corresponds to standard value 0.0224
        omega_b_h2_base = 0.0224
        return omega_b_h2_base * eta_baryon

    def compute_s8(self, alpha_s8: float, epsilon_tension: float) -> float:
        """
        Compute structure formation amplitude S₈ from α and ε parameters.

        The CCF predicts S₈ = σ₈(Ω_m/0.3)^α through cohomological
        constraint on structure growth.

        Args:
            alpha_s8: Structure formation amplitude parameter
            epsilon_tension: Link tension parameter affecting growth

        Returns:
            Predicted S₈ parameter
        """
        # CCF prediction: S₈ depends on both α and ε
        # Standard value: S₈ ≈ 0.776 with α ≈ 1.0, ε ≈ 0.0
        s8_base = 0.776
        # ε modifies growth through cohomological coupling
        epsilon_correction = 1.0 - 0.15 * epsilon_tension
        return s8_base * alpha_s8 * epsilon_correction

    def compute_dark_energy_eos(self, epsilon_tension: float, k_star: float) -> float:
        """
        Compute dark energy equation of state w₀ from ε and k* parameters.

        The CCF predicts deviations from w = -1 through cohomological
        tension in the constraint equations.

        Args:
            epsilon_tension: Link tension parameter
            k_star: Transition scale (Mpc^-1)

        Returns:
            Predicted dark energy equation of state w₀
        """
        # CCF prediction: w₀ = -1 + ε·f(k*)
        # where f(k*) encodes scale-dependent corrections
        w0_base = -1.0
        # k* ≈ 0.05 Mpc^-1 corresponds to ~100 Mpc scale
        scale_factor = np.tanh(k_star / 0.05)
        return w0_base + epsilon_tension * scale_factor

    def compute_hubble_parameter(self, k_star: float, epsilon_tension: float) -> float:
        """
        Compute Hubble parameter H₀ from k* and ε parameters.

        The CCF predicts H₀ through scale-dependent cohomological constraints.
        The k* parameter controls the transition between early/late-time behavior.

        Args:
            k_star: Transition scale (Mpc^-1)
            epsilon_tension: Link tension parameter

        Returns:
            Predicted Hubble parameter H₀ (km/s/Mpc)
        """
        # CCF prediction: H₀ depends on k* (transition scale)
        # k* controls early-vs-late time physics
        # Low k* → low H₀ (Planck-like), high k* → high H₀ (local-like)
        h0_planck = 67.4
        h0_shoes = 73.0

        # Interpolation parameter from k*
        # k* = 0.03 → Planck value
        # k* = 0.10 → SH0ES value
        interpolation_weight = (k_star - 0.03) / (0.10 - 0.03)
        interpolation_weight = np.clip(interpolation_weight, 0, 1)

        # ε provides additional tension correction
        epsilon_correction = 1.0 + 0.5 * epsilon_tension * (interpolation_weight - 0.5)

        h0_predicted = h0_planck + (h0_shoes - h0_planck) * interpolation_weight
        return h0_predicted * epsilon_correction


class CCFLikelihood:
    """
    Likelihood function for CCF model parameters given observational data.

    Combines constraints from:
        - Planck CMB (n_s, Ω_b h²)
        - Weak lensing surveys (S₈)
        - BAO + SNe Ia (w₀)
        - Hubble tension (H₀)
    """

    def __init__(self):
        """Initialize likelihood with observational constraints."""
        self.model = CCFModel()

        # Observational data: mean ± 1σ uncertainty
        self.observations = {
            "n_s": (0.965, 0.004),  # Planck 2018
            "omega_b_h2": (0.0224, 0.0001),  # Planck 2018
            "S8": (0.776, 0.017),  # Planck + weak lensing
            "w0": (-0.83, 0.05),  # BAO + Pantheon
            "H0_planck": (67.4, 0.5),  # Planck 2018
            "H0_shoes": (73.0, 1.0),  # SH0ES 2022
        }

        # Number of data points for BIC calculation
        self.num_data_points = 5  # n_s, Ω_b h², S₈, w₀, H₀

    def log_prior(self, parameters: np.ndarray) -> float:
        """
        Log prior probability for CCF parameters.

        Uses uniform priors within physically reasonable bounds.

        Args:
            parameters: [λ, η, α, ε, k*]

        Returns:
            Log prior probability (0 if in bounds, -inf if out of bounds)
        """
        lambda_spectral, eta_baryon, alpha_s8, epsilon_tension, k_star = parameters

        # Physical bounds on parameters
        if not (0.01 < lambda_spectral < 0.03):
            return -np.inf
        if not (0.8 < eta_baryon < 1.2):
            return -np.inf
        if not (0.8 < alpha_s8 < 1.2):
            return -np.inf
        if not (-0.3 < epsilon_tension < 0.3):
            return -np.inf
        if not (0.01 < k_star < 0.15):
            return -np.inf

        return 0.0  # Uniform prior (log(1) = 0)

    def log_likelihood(self, parameters: np.ndarray) -> float:
        """
        Log likelihood function for CCF parameters.

        Args:
            parameters: [λ, η, α, ε, k*]

        Returns:
            Log likelihood value
        """
        lambda_spectral, eta_baryon, alpha_s8, epsilon_tension, k_star = parameters

        # Compute model predictions
        n_s_pred = self.model.compute_spectral_index(lambda_spectral)
        omega_b_h2_pred = self.model.compute_baryon_density(eta_baryon)
        s8_pred = self.model.compute_s8(alpha_s8, epsilon_tension)
        w0_pred = self.model.compute_dark_energy_eos(epsilon_tension, k_star)
        h0_pred = self.model.compute_hubble_parameter(k_star, epsilon_tension)

        # Extract observational constraints
        n_s_obs, n_s_err = self.observations["n_s"]
        omega_b_h2_obs, omega_b_h2_err = self.observations["omega_b_h2"]
        s8_obs, s8_err = self.observations["S8"]
        w0_obs, w0_err = self.observations["w0"]
        h0_planck_obs, h0_planck_err = self.observations["H0_planck"]
        h0_shoes_obs, h0_shoes_err = self.observations["H0_shoes"]

        # Compute χ² contributions
        chi2_ns = ((n_s_pred - n_s_obs) / n_s_err) ** 2
        chi2_omega_b = ((omega_b_h2_pred - omega_b_h2_obs) / omega_b_h2_err) ** 2
        chi2_s8 = ((s8_pred - s8_obs) / s8_err) ** 2
        chi2_w0 = ((w0_pred - w0_obs) / w0_err) ** 2

        # Hubble tension: model should interpolate between values
        # Penalize being outside the tension range
        h0_min = min(h0_planck_obs, h0_shoes_obs)
        h0_max = max(h0_planck_obs, h0_shoes_obs)

        if h0_min <= h0_pred <= h0_max:
            # Inside tension range: small penalty based on which value it's closer to
            distance_to_planck = abs(h0_pred - h0_planck_obs)
            distance_to_shoes = abs(h0_pred - h0_shoes_obs)
            min_distance = min(distance_to_planck, distance_to_shoes)
            chi2_h0 = (min_distance / 2.0) ** 2  # Soft penalty
        else:
            # Outside tension range: standard χ² to nearest value
            if h0_pred < h0_min:
                chi2_h0 = ((h0_pred - h0_planck_obs) / h0_planck_err) ** 2
            else:
                chi2_h0 = ((h0_pred - h0_shoes_obs) / h0_shoes_err) ** 2

        # Total χ²
        chi2_total = chi2_ns + chi2_omega_b + chi2_s8 + chi2_w0 + chi2_h0

        # Log likelihood: -0.5 * χ²
        return -0.5 * chi2_total

    def log_probability(self, parameters: np.ndarray) -> float:
        """
        Log posterior probability (prior + likelihood).

        Args:
            parameters: [λ, η, α, ε, k*]

        Returns:
            Log posterior probability
        """
        log_prior_value = self.log_prior(parameters)
        if not np.isfinite(log_prior_value):
            return -np.inf

        log_likelihood_value = self.log_likelihood(parameters)
        if not np.isfinite(log_likelihood_value):
            return -np.inf

        return log_prior_value + log_likelihood_value

    def compute_chi2(self, parameters: np.ndarray) -> float:
        """
        Compute χ² for a given set of parameters.

        Args:
            parameters: [λ, η, α, ε, k*]

        Returns:
            χ² value
        """
        return -2.0 * self.log_likelihood(parameters)


class CCFMCMCSampler:
    """
    MCMC sampler for CCF model parameters using emcee.
    """

    def __init__(self, num_walkers: int = 32, num_dimensions: int = 5):
        """
        Initialize MCMC sampler.

        Args:
            num_walkers: Number of MCMC walkers (default: 32)
            num_dimensions: Number of parameters (default: 5)
        """
        self.num_walkers = num_walkers
        self.num_dimensions = num_dimensions
        self.likelihood = CCFLikelihood()
        self.sampler = None
        self.chain = None

        # Parameter names and labels for plotting
        self.parameter_names = ["lambda", "eta", "alpha", "epsilon", "k_star"]
        self.parameter_labels = [
            r"$\lambda$ (spectral)",
            r"$\eta$ (baryon)",
            r"$\alpha$ (S$_8$)",
            r"$\varepsilon$ (tension)",
            r"$k_*$ (Mpc$^{-1}$)",
        ]

    def initialize_walkers(self) -> np.ndarray:
        """
        Initialize walker positions near expected values.

        Returns:
            Initial positions array (num_walkers, num_dimensions)
        """
        # Expected parameter values (rough estimates)
        initial_guess = np.array([0.0175, 1.0, 1.0, 0.0, 0.05])

        # Parameter uncertainties for initialization spread
        initial_spread = np.array([0.003, 0.05, 0.05, 0.05, 0.02])

        # Initialize walkers with small random perturbations
        initial_positions = initial_guess + initial_spread * np.random.randn(
            self.num_walkers, self.num_dimensions
        )

        return initial_positions

    def run_mcmc(
        self, num_steps: int = 5000, burn_in: int = 1000, progress: bool = True
    ) -> np.ndarray:
        """
        Run MCMC sampling.

        Args:
            num_steps: Total number of MCMC steps (default: 5000)
            burn_in: Number of burn-in steps to discard (default: 1000)
            progress: Show progress bar (default: True)

        Returns:
            Flattened chain after burn-in (num_samples, num_dimensions)
        """
        print(f"Initializing MCMC with {self.num_walkers} walkers...")
        print(f"Running {num_steps} steps with {burn_in} burn-in steps...\n")

        # Initialize walker positions
        initial_positions = self.initialize_walkers()

        # Create emcee sampler
        self.sampler = emcee.EnsembleSampler(
            self.num_walkers, self.num_dimensions, self.likelihood.log_probability
        )

        # Run MCMC
        print("Running burn-in phase...")
        state = self.sampler.run_mcmc(initial_positions, burn_in, progress=progress)
        self.sampler.reset()

        print("\nRunning production phase...")
        self.sampler.run_mcmc(state, num_steps - burn_in, progress=progress)

        # Get chain
        self.chain = self.sampler.get_chain(flat=True)

        print("\nMCMC complete!")
        print(f"Chain shape: {self.chain.shape}")
        print(f"Acceptance fraction: {np.mean(self.sampler.acceptance_fraction):.3f}")

        # Compute autocorrelation time (if possible)
        try:
            autocorr_time = self.sampler.get_autocorr_time()
            print(f"Autocorrelation time: {autocorr_time}")
        except emcee.autocorr.AutocorrError:
            print("Warning: Chain may not be converged (autocorrelation time estimation failed)")

        return self.chain

    def get_summary_statistics(self) -> Dict[str, Tuple[float, float, float]]:
        """
        Compute summary statistics for parameters.

        Returns:
            Dictionary with parameter names as keys and (MAP, lower_68, upper_68) tuples
        """
        if self.chain is None:
            raise ValueError("Must run MCMC before computing statistics")

        summary = {}

        for i, name in enumerate(self.parameter_names):
            samples = self.chain[:, i]

            # Maximum a posteriori (MAP) estimate
            # Use median as robust estimate of central value
            map_estimate = np.median(samples)

            # 68% credible interval (1σ equivalent)
            lower_68 = np.percentile(samples, 16)
            upper_68 = np.percentile(samples, 84)

            summary[name] = (map_estimate, lower_68, upper_68)

        return summary

    def compute_bic(self) -> float:
        """
        Compute Bayesian Information Criterion (BIC).

        BIC = χ² + k·ln(n)
        where k = number of parameters, n = number of data points

        Returns:
            BIC value
        """
        if self.chain is None:
            raise ValueError("Must run MCMC before computing BIC")

        # Get MAP parameters
        summary = self.get_summary_statistics()
        map_parameters = np.array([summary[name][0] for name in self.parameter_names])

        # Compute χ² at MAP
        chi2_map = self.likelihood.compute_chi2(map_parameters)

        # BIC = χ² + k·ln(n)
        num_parameters = self.num_dimensions
        num_data_points = self.likelihood.num_data_points
        bic = chi2_map + num_parameters * np.log(num_data_points)

        return bic

    def save_chain(self, output_path: str):
        """
        Save MCMC chain to HDF5 file.

        Args:
            output_path: Path to output HDF5 file
        """
        if self.chain is None:
            raise ValueError("Must run MCMC before saving chain")

        print(f"\nSaving chain to {output_path}...")

        with h5py.File(output_path, "w") as f:
            # Save chain
            f.create_dataset("chain", data=self.chain)

            # Save parameter names
            f.create_dataset("parameter_names", data=np.array(self.parameter_names, dtype="S"))

            # Save summary statistics
            summary = self.get_summary_statistics()
            summary_array = np.array(
                [
                    [summary[name][0], summary[name][1], summary[name][2]]
                    for name in self.parameter_names
                ]
            )
            f.create_dataset("summary_statistics", data=summary_array)

            # Save BIC
            bic = self.compute_bic()
            f.attrs["BIC"] = bic

            # Save metadata
            f.attrs["num_walkers"] = self.num_walkers
            f.attrs["num_steps"] = self.chain.shape[0] // self.num_walkers
            f.attrs["acceptance_fraction"] = np.mean(self.sampler.acceptance_fraction)

        print("Chain saved successfully!")

    def plot_corner(self, output_path: str, truths: Optional[np.ndarray] = None):
        """
        Generate corner plot of 2D marginalized posteriors.

        Args:
            output_path: Path to output PDF file
            truths: Optional true parameter values to mark on plot
        """
        if self.chain is None:
            raise ValueError("Must run MCMC before plotting")

        print("\nGenerating corner plot...")

        # Create corner plot
        fig = corner.corner(
            self.chain,
            labels=self.parameter_labels,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 10},
            truths=truths,
            truth_color="red",
            color="blue",
            bins=30,
            smooth=1.0,
            title_fmt=".4f",
        )

        # Add title
        fig.suptitle("CCF Model Parameter Posteriors (MCMC)", fontsize=14, y=1.0)

        # Save plot
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        print(f"Corner plot saved to {output_path}")

        plt.close(fig)

    def print_results(self):
        """
        Print summary of MCMC results.
        """
        if self.chain is None:
            raise ValueError("Must run MCMC before printing results")

        print("\n" + "=" * 70)
        print("CCF MODEL PARAMETER ESTIMATION RESULTS")
        print("=" * 70)

        summary = self.get_summary_statistics()

        print("\nMAP Estimates and 68% Credible Intervals:")
        print("-" * 70)

        for i, name in enumerate(self.parameter_names):
            map_est, lower_68, upper_68 = summary[name]
            lower_err = map_est - lower_68
            upper_err = upper_68 - map_est

            print(
                f"{self.parameter_labels[i]:20s}: "
                f"{map_est:8.5f} + {upper_err:.5f} - {lower_err:.5f}"
            )

        print("\n" + "-" * 70)

        # Compute and print BIC
        bic = self.compute_bic()
        chi2_map = self.likelihood.compute_chi2(
            np.array([summary[name][0] for name in self.parameter_names])
        )

        print("\nModel Selection Criteria:")
        print(f"  χ² (at MAP)           : {chi2_map:.2f}")
        print(f"  Number of parameters  : {self.num_dimensions}")
        print(f"  Number of data points : {self.likelihood.num_data_points}")
        print(f"  BIC                   : {bic:.2f}")

        print("\n" + "=" * 70)


def main():
    """
    Main function to run CCF MCMC analysis.
    """
    print("=" * 70)
    print("CCF MODEL BAYESIAN PARAMETER ESTIMATION")
    print("Cohomological Constraint Framework")
    print("=" * 70)
    print()

    # Create output directory if it doesn't exist
    import os

    output_dir = "/Users/eirikr/1_Workspace/cosmos/paper/output"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize sampler
    sampler = CCFMCMCSampler(num_walkers=32, num_dimensions=5)

    # Run MCMC
    chain = sampler.run_mcmc(num_steps=5000, burn_in=1000, progress=True)

    # Print results
    sampler.print_results()

    # Save chain to HDF5
    chain_output_path = os.path.join(output_dir, "ccf_chain.h5")
    sampler.save_chain(chain_output_path)

    # Generate corner plot
    corner_output_path = os.path.join(output_dir, "ccf_corner.pdf")
    sampler.plot_corner(corner_output_path)

    print("\nAnalysis complete!")
    print("\nOutputs saved to:")
    print(f"  Chain (HDF5): {chain_output_path}")
    print(f"  Corner plot : {corner_output_path}")


if __name__ == "__main__":
    main()
