"""
Entropic Cosmology - MCMC Parameter Estimation
===============================================
Fitting the Entropic Dark Energy model to Pantheon+ Supernova data

AEG Framework: Phase 1.2-1.6
"""

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from typing import Tuple, Optional, Dict, Callable
import warnings

# Suppress integration warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

C_LIGHT = 299792.458  # km/s


# =============================================================================
# ENTROPIC DARK ENERGY MODEL
# =============================================================================

def w_entropic(z: float, xi: float) -> float:
    """
    Entropic equation of state parameter w(z).

    w(z) = -1 + ξ / (1 - 3ξ ln(1+z))

    Parameters:
        z: Redshift
        xi: Entropic coupling parameter (ξ ≈ 0.25 predicted)

    Returns:
        w(z) - equation of state at redshift z

    Notes:
        - ξ = 0 recovers ΛCDM (w = -1)
        - Singularity at z* where 1 - 3ξ ln(1+z*) = 0
          i.e., z* = exp(1/(3ξ)) - 1
    """
    if xi == 0:
        return -1.0

    denom = 1.0 - 3.0 * xi * np.log(1.0 + z)

    # Check for singularity
    if abs(denom) < 1e-10:
        return -1e6 if denom > 0 else 1e6

    return -1.0 + xi / denom


def rho_de_ratio(z: float, xi: float) -> float:
    """
    Dark energy density evolution: ρ_DE(z) / ρ_DE(0)

    For w(z) = -1 + ξ/(1 - 3ξ ln(1+z)), the integral yields:
    ρ_DE(z)/ρ_DE(0) = (1 - 3ξ ln(1+z))

    Parameters:
        z: Redshift
        xi: Entropic parameter

    Returns:
        ρ_DE(z) / ρ_DE(0)
    """
    if xi == 0:
        return 1.0  # Cosmological constant

    ratio = 1.0 - 3.0 * xi * np.log(1.0 + z)

    # Physical constraint: density must be positive
    if ratio <= 0:
        return 1e-10

    return ratio


def E_z_entropic(z: float, Omega_m: float, xi: float) -> float:
    """
    Dimensionless Hubble parameter E(z) = H(z)/H₀

    E²(z) = Ω_m(1+z)³ + Ω_DE × ρ_DE(z)/ρ_DE(0)

    Parameters:
        z: Redshift
        Omega_m: Matter density parameter
        xi: Entropic parameter

    Returns:
        E(z) = H(z)/H₀
    """
    Omega_DE = 1.0 - Omega_m  # Flat universe

    matter_term = Omega_m * (1.0 + z)**3
    de_term = Omega_DE * rho_de_ratio(z, xi)

    E_squared = matter_term + de_term

    if E_squared <= 0:
        return 1e-10

    return np.sqrt(E_squared)


def E_z_LCDM(z: float, Omega_m: float) -> float:
    """Standard ΛCDM E(z) for comparison."""
    Omega_L = 1.0 - Omega_m
    return np.sqrt(Omega_m * (1.0 + z)**3 + Omega_L)


# =============================================================================
# LUMINOSITY DISTANCE (Phase 1.2)
# =============================================================================

def comoving_distance(z: float, Omega_m: float, xi: float,
                      model: str = 'entropic') -> float:
    """
    Comoving distance D_C(z) in units of c/H₀

    D_C(z) = ∫₀^z dz'/E(z')

    Parameters:
        z: Redshift
        Omega_m: Matter density
        xi: Entropic parameter (ignored for LCDM)
        model: 'entropic' or 'LCDM'

    Returns:
        D_C in units of c/H₀ (multiply by c/H₀ to get Mpc)
    """
    if model == 'LCDM':
        integrand = lambda zp: 1.0 / E_z_LCDM(zp, Omega_m)
    else:
        integrand = lambda zp: 1.0 / E_z_entropic(zp, Omega_m, xi)

    result, _ = quad(integrand, 0, z, limit=100)
    return result


def luminosity_distance(z: float, Omega_m: float, xi: float,
                        H0: float = 70.0, model: str = 'entropic') -> float:
    """
    Luminosity distance D_L(z) in Mpc

    D_L(z) = (1+z) × D_C(z) × (c/H₀)

    Parameters:
        z: Redshift
        Omega_m: Matter density parameter
        xi: Entropic parameter
        H0: Hubble constant in km/s/Mpc
        model: 'entropic' or 'LCDM'

    Returns:
        D_L in Mpc
    """
    D_C = comoving_distance(z, Omega_m, xi, model)
    D_L = (1.0 + z) * D_C * (C_LIGHT / H0)
    return D_L


def distance_modulus(z: float, Omega_m: float, xi: float,
                     H0: float = 70.0, model: str = 'entropic') -> float:
    """
    Distance modulus μ(z) = 5 log₁₀(D_L/10pc)

    Parameters:
        z: Redshift
        Omega_m, xi, H0: Cosmological parameters
        model: 'entropic' or 'LCDM'

    Returns:
        Distance modulus μ
    """
    D_L = luminosity_distance(z, Omega_m, xi, H0, model)

    # D_L in Mpc, convert to pc: 1 Mpc = 10^6 pc
    D_L_pc = D_L * 1e6

    if D_L_pc <= 0:
        return 99.0  # Invalid

    mu = 5.0 * np.log10(D_L_pc / 10.0)
    return mu


# =============================================================================
# VECTORIZED VERSIONS FOR MCMC EFFICIENCY
# =============================================================================

def distance_modulus_array(z_array: np.ndarray, Omega_m: float, xi: float,
                           H0: float = 70.0, model: str = 'entropic') -> np.ndarray:
    """Compute distance modulus for array of redshifts."""
    return np.array([distance_modulus(z, Omega_m, xi, H0, model) for z in z_array])


# =============================================================================
# LIKELIHOOD FUNCTION (Phase 1.3)
# =============================================================================

def chi_squared(params: np.ndarray, z_data: np.ndarray, mu_data: np.ndarray,
                sigma_data: np.ndarray, model: str = 'entropic') -> float:
    """
    Chi-squared statistic for cosmological fit.

    χ² = Σᵢ [(μ_theory(zᵢ) - μ_obs,ᵢ) / σᵢ]²

    Parameters:
        params: [Omega_m, xi] for entropic, [Omega_m] for LCDM
        z_data, mu_data, sigma_data: Observed SN data
        model: 'entropic' or 'LCDM'

    Returns:
        χ² value
    """
    if model == 'LCDM':
        Omega_m = params[0]
        xi = 0.0
    else:
        Omega_m, xi = params[0], params[1]

    # Marginalize over H0 analytically (Goliath et al. 2001 method)
    # or use fixed H0 for simplicity
    H0 = 70.0

    mu_theory = distance_modulus_array(z_data, Omega_m, xi, H0, model)

    residuals = (mu_theory - mu_data) / sigma_data
    chi2 = np.sum(residuals**2)

    return chi2


def log_likelihood(params: np.ndarray, z_data: np.ndarray, mu_data: np.ndarray,
                   sigma_data: np.ndarray, model: str = 'entropic') -> float:
    """
    Log-likelihood for MCMC.

    ln L = -χ²/2
    """
    chi2 = chi_squared(params, z_data, mu_data, sigma_data, model)
    return -0.5 * chi2


def log_prior(params: np.ndarray, model: str = 'entropic') -> float:
    """
    Log-prior for cosmological parameters.

    Priors:
        Omega_m: Uniform [0.1, 0.5]
        xi: Uniform [-0.5, 1.0]

    Returns:
        0 if within bounds, -inf otherwise
    """
    if model == 'LCDM':
        Omega_m = params[0]
        if 0.1 < Omega_m < 0.5:
            return 0.0
        return -np.inf
    else:
        Omega_m, xi = params[0], params[1]
        if 0.1 < Omega_m < 0.5 and -0.5 < xi < 1.0:
            return 0.0
        return -np.inf


def log_posterior(params: np.ndarray, z_data: np.ndarray, mu_data: np.ndarray,
                  sigma_data: np.ndarray, model: str = 'entropic') -> float:
    """
    Log-posterior = log-prior + log-likelihood
    """
    lp = log_prior(params, model)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, z_data, mu_data, sigma_data, model)


# =============================================================================
# SYNTHETIC PANTHEON-LIKE DATA (for testing without network access)
# =============================================================================

def generate_synthetic_pantheon(n_sne: int = 100, Omega_m_true: float = 0.3,
                                xi_true: float = 0.0, noise_level: float = 0.15,
                                seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic Type Ia supernova data mimicking Pantheon+.

    Parameters:
        n_sne: Number of supernovae
        Omega_m_true: True matter density
        xi_true: True entropic parameter (0 = ΛCDM)
        noise_level: Distance modulus uncertainty
        seed: Random seed

    Returns:
        (z_data, mu_data, sigma_data)
    """
    np.random.seed(seed)

    # Redshift distribution similar to Pantheon+
    z_data = np.concatenate([
        np.random.uniform(0.01, 0.1, n_sne // 3),    # Low-z
        np.random.uniform(0.1, 0.5, n_sne // 3),     # Mid-z
        np.random.uniform(0.5, 1.5, n_sne // 3 + n_sne % 3)  # High-z
    ])
    z_data = np.sort(z_data)

    # True distance moduli
    model = 'entropic' if xi_true != 0 else 'LCDM'
    mu_true = distance_modulus_array(z_data, Omega_m_true, xi_true, H0=70.0, model=model)

    # Add Gaussian noise
    sigma_data = np.full_like(z_data, noise_level)
    mu_data = mu_true + np.random.normal(0, noise_level, len(z_data))

    return z_data, mu_data, sigma_data


# =============================================================================
# SIMPLE GRID SEARCH (before full MCMC)
# =============================================================================

def grid_search(z_data: np.ndarray, mu_data: np.ndarray, sigma_data: np.ndarray,
                model: str = 'entropic', n_grid: int = 50) -> Dict:
    """
    Grid search over parameter space to find approximate best-fit.

    Returns:
        Dictionary with best-fit parameters and chi² grid
    """
    Omega_m_range = np.linspace(0.15, 0.45, n_grid)

    if model == 'LCDM':
        chi2_grid = np.zeros(n_grid)
        for i, Om in enumerate(Omega_m_range):
            chi2_grid[i] = chi_squared([Om], z_data, mu_data, sigma_data, 'LCDM')

        best_idx = np.argmin(chi2_grid)
        return {
            'best_Omega_m': Omega_m_range[best_idx],
            'best_chi2': chi2_grid[best_idx],
            'Omega_m_range': Omega_m_range,
            'chi2_grid': chi2_grid,
        }
    else:
        xi_range = np.linspace(-0.2, 0.5, n_grid)
        chi2_grid = np.zeros((n_grid, n_grid))

        for i, Om in enumerate(Omega_m_range):
            for j, xi in enumerate(xi_range):
                chi2_grid[i, j] = chi_squared([Om, xi], z_data, mu_data, sigma_data, 'entropic')

        best_idx = np.unravel_index(np.argmin(chi2_grid), chi2_grid.shape)
        return {
            'best_Omega_m': Omega_m_range[best_idx[0]],
            'best_xi': xi_range[best_idx[1]],
            'best_chi2': chi2_grid[best_idx],
            'Omega_m_range': Omega_m_range,
            'xi_range': xi_range,
            'chi2_grid': chi2_grid,
        }


# =============================================================================
# MCMC RUNNER (Phase 1.4)
# =============================================================================

def run_mcmc_simple(z_data: np.ndarray, mu_data: np.ndarray, sigma_data: np.ndarray,
                    model: str = 'entropic', n_steps: int = 5000,
                    n_walkers: int = 32, burn_in: int = 1000) -> Dict:
    """
    Run MCMC using simple Metropolis-Hastings (no emcee dependency).

    Parameters:
        z_data, mu_data, sigma_data: SN data
        model: 'entropic' or 'LCDM'
        n_steps: Number of MCMC steps
        n_walkers: Number of independent chains
        burn_in: Steps to discard

    Returns:
        Dictionary with chains and diagnostics
    """
    # Initialize
    if model == 'LCDM':
        ndim = 1
        p0 = np.array([0.3])
        proposal_sigma = np.array([0.02])
    else:
        ndim = 2
        p0 = np.array([0.3, 0.1])
        proposal_sigma = np.array([0.02, 0.05])

    chains = []

    for walker in range(n_walkers):
        chain = np.zeros((n_steps, ndim))
        current = p0 + np.random.normal(0, proposal_sigma)
        current_lp = log_posterior(current, z_data, mu_data, sigma_data, model)

        accepted = 0
        for step in range(n_steps):
            # Propose
            proposal = current + np.random.normal(0, proposal_sigma)
            proposal_lp = log_posterior(proposal, z_data, mu_data, sigma_data, model)

            # Accept/reject
            if np.log(np.random.random()) < proposal_lp - current_lp:
                current = proposal
                current_lp = proposal_lp
                accepted += 1

            chain[step] = current

        chains.append(chain)
        print(f"  Walker {walker+1}/{n_walkers}: acceptance = {accepted/n_steps:.2%}")

    chains = np.array(chains)

    # Discard burn-in
    samples = chains[:, burn_in:, :].reshape(-1, ndim)

    # Compute statistics
    if model == 'LCDM':
        result = {
            'Omega_m': np.mean(samples[:, 0]),
            'Omega_m_std': np.std(samples[:, 0]),
            'samples': samples,
            'chains': chains,
        }
    else:
        result = {
            'Omega_m': np.mean(samples[:, 0]),
            'Omega_m_std': np.std(samples[:, 0]),
            'xi': np.mean(samples[:, 1]),
            'xi_std': np.std(samples[:, 1]),
            'samples': samples,
            'chains': chains,
        }

    return result


# =============================================================================
# BAYES FACTOR (Phase 1.5)
# =============================================================================

def compute_evidence_ratio(z_data: np.ndarray, mu_data: np.ndarray,
                           sigma_data: np.ndarray, n_samples: int = 10000) -> float:
    """
    Estimate Bayes factor B = P(data|Entropic) / P(data|ΛCDM)
    using Savage-Dickey density ratio at ξ = 0.

    B ≈ π(ξ=0) / P(ξ=0|data)

    where π is the prior and P is the posterior.

    Returns:
        log(Bayes Factor) - positive favors Entropic, negative favors ΛCDM
    """
    # Prior density at xi=0 (uniform on [-0.5, 1.0])
    prior_density_at_zero = 1.0 / 1.5  # = 0.667

    # Estimate posterior density at xi=0 from MCMC
    result = run_mcmc_simple(z_data, mu_data, sigma_data, model='entropic',
                            n_steps=2000, n_walkers=16, burn_in=500)
    xi_samples = result['samples'][:, 1]

    # Kernel density estimate at xi=0
    from scipy.stats import gaussian_kde
    try:
        kde = gaussian_kde(xi_samples)
        posterior_density_at_zero = kde(0.0)[0]
    except:
        # Fallback: histogram estimate
        hist, edges = np.histogram(xi_samples, bins=50, density=True)
        idx = np.argmin(np.abs(edges[:-1] - 0.0))
        posterior_density_at_zero = hist[idx] if idx < len(hist) else 0.1

    # Savage-Dickey ratio
    if posterior_density_at_zero > 0:
        log_bayes = np.log(prior_density_at_zero / posterior_density_at_zero)
    else:
        log_bayes = np.inf  # Strong evidence against nested model

    return log_bayes


# =============================================================================
# TESTS & DEMOS
# =============================================================================

def test_distance_calculations():
    """Test distance modulus calculations."""
    print("\nTesting distance calculations...")

    z_test = [0.01, 0.1, 0.5, 1.0, 1.5]
    Omega_m = 0.3
    H0 = 70.0

    print(f"\n  z     | μ(ΛCDM)  | μ(Entropic, ξ=0.25)")
    print("-" * 45)
    for z in z_test:
        mu_lcdm = distance_modulus(z, Omega_m, 0.0, H0, 'LCDM')
        mu_ent = distance_modulus(z, Omega_m, 0.25, H0, 'entropic')
        print(f"  {z:.2f}  | {mu_lcdm:8.3f} | {mu_ent:8.3f}")

    print("  ✓ Distance calculations complete!")


def test_synthetic_fit():
    """Test fitting on synthetic data."""
    print("\nTesting synthetic data fit...")

    # Generate ΛCDM data
    z, mu, sigma = generate_synthetic_pantheon(n_sne=100, Omega_m_true=0.3, xi_true=0.0)
    print(f"  Generated {len(z)} synthetic SNe")

    # Grid search
    print("  Running grid search (ΛCDM)...")
    result_lcdm = grid_search(z, mu, sigma, model='LCDM', n_grid=30)
    print(f"    Best Ω_m = {result_lcdm['best_Omega_m']:.3f}, χ² = {result_lcdm['best_chi2']:.1f}")

    print("  Running grid search (Entropic)...")
    result_ent = grid_search(z, mu, sigma, model='entropic', n_grid=30)
    print(f"    Best Ω_m = {result_ent['best_Omega_m']:.3f}, ξ = {result_ent['best_xi']:.3f}, χ² = {result_ent['best_chi2']:.1f}")

    print("  ✓ Synthetic fit test complete!")


def demo_entropic_vs_lcdm():
    """Demonstrate difference between Entropic and ΛCDM predictions."""
    print("\n" + "=" * 60)
    print("ENTROPIC VS ΛCDM COMPARISON")
    print("=" * 60)

    z_range = np.linspace(0.01, 2.0, 100)
    Omega_m = 0.3

    # Compute H(z)/H0
    E_lcdm = [E_z_LCDM(z, Omega_m) for z in z_range]
    E_ent_025 = [E_z_entropic(z, Omega_m, 0.25) for z in z_range]
    E_ent_010 = [E_z_entropic(z, Omega_m, 0.10) for z in z_range]

    print("\n  z     | E(ΛCDM) | E(ξ=0.10) | E(ξ=0.25) | Δ(0.25)")
    print("-" * 60)
    for i in [0, 10, 25, 50, 75, 99]:
        z = z_range[i]
        delta = (E_ent_025[i] - E_lcdm[i]) / E_lcdm[i] * 100
        print(f"  {z:.2f}  | {E_lcdm[i]:.4f}  | {E_ent_010[i]:.4f}    | {E_ent_025[i]:.4f}    | {delta:+.2f}%")

    print("\n  Key: Negative Δ means Entropic predicts LOWER H(z)")
    print("       This can help resolve the Hubble tension!")


if __name__ == "__main__":
    print("=" * 60)
    print("ENTROPIC COSMOLOGY MODULE - AEG Framework")
    print("=" * 60)

    test_distance_calculations()
    test_synthetic_fit()
    demo_entropic_vs_lcdm()
