#!/usr/bin/env python3
"""
Optimized MCMC Analysis for Entropic Cosmology
===============================================
Uses emcee with multiprocessing for M1 optimization

Run: python3 run_mcmc_optimized.py
"""

import numpy as np
import emcee
import corner
import matplotlib.pyplot as plt
from scipy.integrate import quad
from multiprocessing import Pool, cpu_count
import time
import warnings
import os

# Suppress integration warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend for headless operation
import matplotlib
matplotlib.use('Agg')

print("=" * 70)
print("ENTROPIC COSMOLOGY - OPTIMIZED MCMC ANALYSIS")
print("=" * 70)
print(f"CPU Cores: {cpu_count()}")
print(f"emcee version: {emcee.__version__}")

# =============================================================================
# COSMOLOGICAL MODEL
# =============================================================================

C_LIGHT = 299792.458  # km/s

def E_z_entropic(z, Omega_m, xi):
    """Dimensionless Hubble parameter for Entropic Dark Energy."""
    Omega_DE = 1.0 - Omega_m
    rho_ratio = 1.0 - 3.0 * xi * np.log(1.0 + z)
    if rho_ratio <= 0:
        return 1e10
    E_squared = Omega_m * (1.0 + z)**3 + Omega_DE * rho_ratio
    return np.sqrt(max(E_squared, 1e-10))

def comoving_distance(z, Omega_m, xi):
    """Comoving distance in units of c/H₀."""
    integrand = lambda zp: 1.0 / E_z_entropic(zp, Omega_m, xi)
    result, _ = quad(integrand, 0, z, limit=50)
    return result

def distance_modulus(z, Omega_m, xi, H0=70.0):
    """Distance modulus μ(z)."""
    D_C = comoving_distance(z, Omega_m, xi)
    D_L = (1.0 + z) * D_C * (C_LIGHT / H0)
    D_L_pc = D_L * 1e6
    if D_L_pc <= 0:
        return 99.0
    return 5.0 * np.log10(D_L_pc / 10.0)

# =============================================================================
# SYNTHETIC DATA GENERATION
# =============================================================================

def generate_synthetic_data(n_sne=150, Omega_m_true=0.30, xi_true=0.15,
                            sigma=0.12, seed=42):
    """Generate Pantheon-like synthetic supernova data."""
    np.random.seed(seed)

    # Realistic redshift distribution
    z_data = np.concatenate([
        np.random.uniform(0.01, 0.1, n_sne // 3),
        np.random.uniform(0.1, 0.5, n_sne // 3),
        np.random.uniform(0.5, 1.5, n_sne // 3 + n_sne % 3),
    ])
    z_data = np.sort(z_data)

    # True distance moduli
    mu_true = np.array([distance_modulus(z, Omega_m_true, xi_true) for z in z_data])

    # Add noise
    sigma_data = np.full_like(z_data, sigma)
    mu_data = mu_true + np.random.normal(0, sigma, len(z_data))

    return z_data, mu_data, sigma_data

# =============================================================================
# LIKELIHOOD AND PRIOR
# =============================================================================

def log_prior(theta):
    """Flat prior on Ω_m and ξ."""
    Omega_m, xi = theta
    if 0.1 < Omega_m < 0.5 and -0.3 < xi < 0.6:
        return 0.0
    return -np.inf

def log_likelihood(theta, z_data, mu_data, sigma_data):
    """Gaussian log-likelihood."""
    Omega_m, xi = theta

    chi2 = 0.0
    for i, z in enumerate(z_data):
        mu_theory = distance_modulus(z, Omega_m, xi)
        chi2 += ((mu_theory - mu_data[i]) / sigma_data[i])**2

    return -0.5 * chi2

def log_posterior(theta, z_data, mu_data, sigma_data):
    """Log-posterior = log-prior + log-likelihood."""
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, z_data, mu_data, sigma_data)

# =============================================================================
# MCMC RUNNER
# =============================================================================

def run_mcmc(z_data, mu_data, sigma_data, n_walkers=32, n_steps=2000,
             n_cores=None, progress=True):
    """Run parallel MCMC with emcee."""

    ndim = 2
    if n_cores is None:
        n_cores = min(cpu_count(), n_walkers)

    # Initial positions
    p0 = np.array([0.3, 0.1]) + 0.01 * np.random.randn(n_walkers, ndim)

    print(f"\n[MCMC] Configuration:")
    print(f"  Walkers:    {n_walkers}")
    print(f"  Steps:      {n_steps}")
    print(f"  CPU Cores:  {n_cores}")
    print(f"  Data pts:   {len(z_data)}")

    # Run with multiprocessing - pass data via args
    start_time = time.time()

    with Pool(n_cores) as pool:
        sampler = emcee.EnsembleSampler(
            n_walkers, ndim, log_posterior,
            args=(z_data, mu_data, sigma_data),
            pool=pool
        )

        print(f"\n[MCMC] Running...")
        sampler.run_mcmc(p0, n_steps, progress=progress)

    elapsed = time.time() - start_time
    n_samples = n_walkers * n_steps
    print(f"\n[MCMC] Completed in {elapsed:.1f}s ({n_samples/elapsed:.0f} samples/s)")

    # Diagnostics
    acceptance = np.mean(sampler.acceptance_fraction)
    print(f"[MCMC] Acceptance fraction: {acceptance:.2%}")

    # Burn-in and thinning
    burn_in = n_steps // 4
    samples = sampler.get_chain(discard=burn_in, flat=True)
    log_prob = sampler.get_log_prob(discard=burn_in, flat=True)

    # Results
    Omega_m_mean = np.mean(samples[:, 0])
    Omega_m_std = np.std(samples[:, 0])
    xi_mean = np.mean(samples[:, 1])
    xi_std = np.std(samples[:, 1])

    print(f"\n[MCMC] Results:")
    print(f"  Ω_m = {Omega_m_mean:.4f} ± {Omega_m_std:.4f}")
    print(f"  ξ   = {xi_mean:.4f} ± {xi_std:.4f}")

    return {
        'samples': samples,
        'log_prob': log_prob,
        'chain': sampler.get_chain(),
        'Omega_m': (Omega_m_mean, Omega_m_std),
        'xi': (xi_mean, xi_std),
        'acceptance': acceptance,
        'elapsed': elapsed,
    }

# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_results(result, z_data, mu_data, sigma_data, true_params=None,
                 output_dir='.'):
    """Generate corner plot and diagnostic figures."""

    samples = result['samples']
    labels = [r'$\Omega_m$', r'$\xi$']

    # Corner plot
    fig1 = corner.corner(
        samples, labels=labels,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 12}
    )

    if true_params:
        corner.overplot_lines(fig1, true_params, color='red')
        corner.overplot_points(fig1, [true_params], marker='s', color='red')

    fig1.suptitle('Entropic Cosmology: MCMC Posterior', fontsize=14, y=1.02)
    fig1.savefig(f'{output_dir}/plots/mcmc_corner.png', dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved {output_dir}/plots/mcmc_corner.png")

    # Chain evolution
    fig2, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    chain = result['chain']

    for i in range(min(10, chain.shape[1])):
        axes[0].plot(chain[:, i, 0], alpha=0.5, lw=0.5)
        axes[1].plot(chain[:, i, 1], alpha=0.5, lw=0.5)

    axes[0].set_ylabel(r'$\Omega_m$')
    axes[1].set_ylabel(r'$\xi$')
    axes[1].set_xlabel('Step')
    fig2.suptitle('MCMC Chain Evolution', fontsize=14)
    fig2.tight_layout()
    fig2.savefig(f'{output_dir}/plots/mcmc_chains.png', dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved {output_dir}/plots/mcmc_chains.png")

    # Hubble diagram
    fig3, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[3, 1])

    # Best fit
    Om_best, xi_best = result['Omega_m'][0], result['xi'][0]
    z_model = np.linspace(0.01, max(z_data), 100)
    mu_entropic = [distance_modulus(z, Om_best, xi_best) for z in z_model]
    mu_lcdm = [distance_modulus(z, Om_best, 0.0) for z in z_model]

    ax1.errorbar(z_data, mu_data, yerr=sigma_data, fmt='o', ms=3, alpha=0.5,
                 label='Data', color='blue')
    ax1.plot(z_model, mu_entropic, 'r-', lw=2,
             label=f'Entropic (ξ={xi_best:.3f})')
    ax1.plot(z_model, mu_lcdm, 'k--', lw=2, label='ΛCDM (ξ=0)')
    ax1.set_ylabel(r'Distance Modulus $\mu$')
    ax1.legend()
    ax1.set_title('Hubble Diagram: Entropic vs ΛCDM')

    # Residuals
    mu_model = [distance_modulus(z, Om_best, xi_best) for z in z_data]
    residuals = mu_data - np.array(mu_model)
    ax2.errorbar(z_data, residuals, yerr=sigma_data, fmt='o', ms=3, alpha=0.5)
    ax2.axhline(0, color='k', linestyle='--')
    ax2.set_xlabel('Redshift z')
    ax2.set_ylabel('Residual')

    fig3.tight_layout()
    fig3.savefig(f'{output_dir}/plots/hubble_diagram.png', dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved {output_dir}/plots/hubble_diagram.png")

    plt.close('all')

# =============================================================================
# BAYES FACTOR
# =============================================================================

def compute_bayes_factor(samples, prior_range=(-0.3, 0.6)):
    """
    Compute Bayes Factor using Savage-Dickey density ratio.

    B = P(ξ=0 | prior) / P(ξ=0 | posterior)

    Positive log(B) favors Entropic, negative favors ΛCDM.
    """
    from scipy.stats import gaussian_kde

    xi_samples = samples[:, 1]

    # Prior density at ξ=0 (uniform)
    prior_width = prior_range[1] - prior_range[0]
    prior_at_zero = 1.0 / prior_width

    # Posterior density at ξ=0 (KDE estimate)
    try:
        kde = gaussian_kde(xi_samples)
        posterior_at_zero = kde(0.0)[0]
    except:
        # Fallback: histogram
        hist, edges = np.histogram(xi_samples, bins=50, density=True)
        idx = np.argmin(np.abs(edges[:-1] - 0.0))
        posterior_at_zero = hist[idx]

    if posterior_at_zero > 0:
        log_B = np.log(prior_at_zero / posterior_at_zero)
    else:
        log_B = np.inf

    # Interpretation
    if log_B > 3:
        interpretation = "Strong evidence for Entropic"
    elif log_B > 1:
        interpretation = "Moderate evidence for Entropic"
    elif log_B > 0:
        interpretation = "Weak evidence for Entropic"
    elif log_B > -1:
        interpretation = "Weak evidence for ΛCDM"
    elif log_B > -3:
        interpretation = "Moderate evidence for ΛCDM"
    else:
        interpretation = "Strong evidence for ΛCDM"

    return {
        'log_B': log_B,
        'B': np.exp(log_B) if np.isfinite(log_B) else np.inf,
        'interpretation': interpretation,
        'prior_at_zero': prior_at_zero,
        'posterior_at_zero': posterior_at_zero,
    }

# =============================================================================
# MAIN
# =============================================================================

def main():
    output_dir = '/Users/eirikr/cosmos'
    os.makedirs(output_dir, exist_ok=True)

    # True parameters for synthetic data
    TRUE_OMEGA_M = 0.30
    TRUE_XI = 0.15  # Non-zero entropic parameter

    print("\n" + "=" * 70)
    print("GENERATING SYNTHETIC DATA")
    print("=" * 70)
    print(f"  True Ω_m = {TRUE_OMEGA_M}")
    print(f"  True ξ   = {TRUE_XI}")

    z_data, mu_data, sigma_data = generate_synthetic_data(
        n_sne=150, Omega_m_true=TRUE_OMEGA_M, xi_true=TRUE_XI, sigma=0.12
    )
    print(f"  Generated {len(z_data)} synthetic SNe")
    print(f"  Redshift range: {z_data.min():.3f} to {z_data.max():.3f}")

    # Run MCMC
    print("\n" + "=" * 70)
    print("RUNNING MCMC")
    print("=" * 70)

    result = run_mcmc(
        z_data, mu_data, sigma_data,
        n_walkers=32,
        n_steps=1500,  # Reduced for speed, increase for better convergence
        n_cores=8,
        progress=True
    )

    # Compute Bayes Factor
    print("\n" + "=" * 70)
    print("BAYES FACTOR ANALYSIS")
    print("=" * 70)

    bf_result = compute_bayes_factor(result['samples'])
    print(f"  ln(B) = {bf_result['log_B']:.2f}")
    print(f"  B = {bf_result['B']:.2f}")
    print(f"  Interpretation: {bf_result['interpretation']}")

    # Generate plots
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)

    plot_results(result, z_data, mu_data, sigma_data,
                 true_params=[TRUE_OMEGA_M, TRUE_XI],
                 output_dir=output_dir)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
  True Parameters:
    Ω_m = {TRUE_OMEGA_M}
    ξ   = {TRUE_XI}

  Recovered Parameters:
    Ω_m = {result['Omega_m'][0]:.4f} ± {result['Omega_m'][1]:.4f}
    ξ   = {result['xi'][0]:.4f} ± {result['xi'][1]:.4f}

  Bayes Factor:
    ln(B) = {bf_result['log_B']:.2f}
    {bf_result['interpretation']}

  Performance:
    {result['elapsed']:.1f}s total
    {len(result['samples'])} samples
    {result['acceptance']:.1%} acceptance

  Output files:
    {output_dir}/plots/mcmc_corner.png
    {output_dir}/plots/mcmc_chains.png
    {output_dir}/plots/hubble_diagram.png
""")

    return result, bf_result


if __name__ == "__main__":
    try:
        result, bf_result = main()
        print("✓ MCMC analysis completed successfully!")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
