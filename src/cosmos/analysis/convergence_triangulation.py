#!/usr/bin/env python3
"""
convergence_triangulation.py — Resolution Convergence Study

The Crisis of Convergence:
    48³:  β ≈ 0.050
    128³: β ≈ 0.008

Is β → 0 (Turbulent Washout) or β → β_∞ > 0 (Spandrel Survives)?

This script:
1. Runs 64³ sweep to get third data point
2. Fits β(N) = β_∞ + A·N^(-p)
3. Extrapolates to Hero Run resolutions

Author: Spandrel Framework
Date: November 28, 2025
"""

import sys
import time
import numpy as np
from scipy.optimize import curve_fit
from typing import Dict, List, Tuple

from cosmos.engines.flame_box_mps import MPSConfig, MPSSpectralSolver, DEVICE
import torch


# Existing data points
EXISTING_DATA = {
    48: {'D_ref': 2.73, 'beta': 0.050, 'source': 'pilot'},
    128: {'D_ref': 2.81, 'beta': 0.008, 'source': 'production'}
}

# Metallicity values for sweep
METALLICITY_VALUES = [0.1, 0.3, 1.0, 3.0]  # Reduced set for speed


def box_counting_dimension(Y: torch.Tensor, threshold: float = 0.5) -> float:
    """Compute fractal dimension using box-counting."""
    Y_np = Y.cpu().numpy()
    N = Y_np.shape[0]

    flame = (Y_np > threshold - 0.1) & (Y_np < threshold + 0.1)

    box_sizes = []
    for s in [2, 4, 8, 16]:
        if N % s == 0 and s < N:
            box_sizes.append(s)

    if len(box_sizes) < 2:
        box_sizes = [2, 4]

    counts = []
    for size in box_sizes:
        n_boxes_per_dim = N // size
        count = 0
        for i in range(n_boxes_per_dim):
            for j in range(n_boxes_per_dim):
                for k in range(n_boxes_per_dim):
                    box = flame[
                        i*size:(i+1)*size,
                        j*size:(j+1)*size,
                        k*size:(k+1)*size
                    ]
                    if box.any():
                        count += 1
        counts.append(count)

    log_eps = np.log(1.0 / np.array(box_sizes))
    log_N = np.log(np.array(counts) + 1)

    if len(log_eps) >= 2:
        slope, _ = np.polyfit(log_eps, log_N, 1)
        D = slope
    else:
        D = 2.5

    return np.clip(D, 2.0, 3.0)


def run_metallicity_sweep(N: int, n_steps: int = 400) -> Dict:
    """Run D(Z) sweep at given resolution."""
    print(f"\n{'='*60}")
    print(f"  Resolution: {N}³")
    print('='*60)

    results = {}

    for Z in METALLICITY_VALUES:
        print(f"\n  Z = {Z} Z_sun:")

        config = MPSConfig(
            N=N,
            Z_metallicity=Z,
            enable_expansion=True,
            tau_expansion=0.2,
            enable_baroclinic=True,
            enable_stratification=True
        )

        solver = MPSSpectralSolver(config)

        # Track D evolution
        D_samples = []
        sample_interval = 100

        start = time.perf_counter()

        for step in range(n_steps):
            solver.step_forward()

            if (step + 1) % sample_interval == 0:
                D = box_counting_dimension(solver.Y_scalar)
                D_samples.append(D)
                print(f"    Step {step+1}: D = {D:.3f}")

        elapsed = time.perf_counter() - start

        # Final D (average of last samples)
        D_final = np.mean(D_samples[-2:]) if len(D_samples) >= 2 else D_samples[-1]

        results[Z] = {
            'D': D_final,
            'D_samples': D_samples,
            'elapsed': elapsed
        }

        print(f"    Final D = {D_final:.3f} ({elapsed:.1f}s)")

    return results


def fit_DZ_relation(results: Dict) -> Tuple[float, float]:
    """Fit D(Z) = D_ref - β × ln(Z/Z_sun)."""
    Z_arr = np.array(list(results.keys()))
    D_arr = np.array([results[Z]['D'] for Z in Z_arr])

    log_Z = np.log(Z_arr)
    slope, intercept = np.polyfit(log_Z, D_arr, 1)

    D_ref = intercept
    beta = -slope

    return D_ref, beta


def power_law_model(N, beta_inf, A, p):
    """β(N) = β_∞ + A·N^(-p)"""
    return beta_inf + A * np.power(N, -p)


def fit_convergence(N_values: np.ndarray, beta_values: np.ndarray) -> Dict:
    """Fit the resolution convergence power law."""

    # Try to fit the full model
    try:
        # Initial guesses
        p0 = [0.005, 0.5, 0.5]  # β_∞, A, p
        bounds = ([0, 0, 0.1], [0.1, 10, 3])

        popt, pcov = curve_fit(power_law_model, N_values, beta_values,
                               p0=p0, bounds=bounds, maxfev=10000)

        beta_inf, A, p = popt
        perr = np.sqrt(np.diag(pcov))

        return {
            'beta_inf': beta_inf,
            'A': A,
            'p': p,
            'beta_inf_err': perr[0],
            'success': True
        }
    except Exception as e:
        print(f"  Warning: Power law fit failed ({e})")

        # Fallback: linear extrapolation in log-log space
        log_N = np.log(N_values)
        log_beta = np.log(beta_values + 1e-6)

        slope, intercept = np.polyfit(log_N, log_beta, 1)

        # Estimate β at N=2048
        log_beta_2048 = slope * np.log(2048) + intercept
        beta_2048 = np.exp(log_beta_2048)

        return {
            'beta_inf': max(beta_2048, 0),
            'A': np.nan,
            'p': -slope,
            'beta_inf_err': np.nan,
            'success': False,
            'method': 'log-linear extrapolation'
        }


def main():
    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + " CONVERGENCE TRIANGULATION ".center(58) + "║")
    print("║" + " Is β → 0 or β → β_∞ > 0? ".center(58) + "║")
    print("╚" + "═" * 58 + "╝")
    print()
    print(f"Device: {DEVICE}")
    print()

    # =========================================================
    # Step 1: Run 64³ sweep (the missing data point)
    # =========================================================
    print("\n" + "=" * 60)
    print("STEP 1: Running 64³ sweep for triangulation")
    print("=" * 60)

    results_64 = run_metallicity_sweep(N=64, n_steps=400)
    D_ref_64, beta_64 = fit_DZ_relation(results_64)

    print(f"\n  64³ Result: D_ref = {D_ref_64:.3f}, β = {beta_64:.4f}")

    # =========================================================
    # Step 2: Compile all data points
    # =========================================================
    print("\n" + "=" * 60)
    print("STEP 2: Compiling convergence data")
    print("=" * 60)

    convergence_data = {
        48: {'D_ref': 2.73, 'beta': 0.050},
        64: {'D_ref': D_ref_64, 'beta': beta_64},
        128: {'D_ref': 2.81, 'beta': 0.008}
    }

    print("\n  Resolution | D_ref  | β      | Δβ from prev")
    print("  " + "-" * 50)

    prev_beta = None
    for N in sorted(convergence_data.keys()):
        data = convergence_data[N]
        delta = f"{data['beta'] - prev_beta:+.4f}" if prev_beta else "---"
        print(f"  {N:>10}³ | {data['D_ref']:.3f} | {data['beta']:.4f} | {delta}")
        prev_beta = data['beta']

    # =========================================================
    # Step 3: Fit convergence power law
    # =========================================================
    print("\n" + "=" * 60)
    print("STEP 3: Fitting β(N) = β_∞ + A·N^(-p)")
    print("=" * 60)

    N_values = np.array(sorted(convergence_data.keys()))
    beta_values = np.array([convergence_data[N]['beta'] for N in N_values])

    fit_result = fit_convergence(N_values, beta_values)

    print(f"\n  Fit Results:")
    print(f"    β_∞ = {fit_result['beta_inf']:.4f} ± {fit_result.get('beta_inf_err', np.nan):.4f}")
    print(f"    A   = {fit_result['A']:.4f}")
    print(f"    p   = {fit_result['p']:.3f}")

    # =========================================================
    # Step 4: Extrapolate to Hero Run resolutions
    # =========================================================
    print("\n" + "=" * 60)
    print("STEP 4: Extrapolation to Hero Run resolutions")
    print("=" * 60)

    hero_resolutions = [256, 512, 1024, 2048, 4096]

    print("\n  Resolution | β (predicted) | δμ (mag) | Status")
    print("  " + "-" * 55)

    for N in hero_resolutions:
        if fit_result['success']:
            beta_pred = power_law_model(N, fit_result['beta_inf'],
                                       fit_result['A'], fit_result['p'])
        else:
            # Log-linear extrapolation
            log_N = np.log(N_values)
            log_beta = np.log(beta_values + 1e-6)
            slope, intercept = np.polyfit(log_N, log_beta, 1)
            beta_pred = np.exp(slope * np.log(N) + intercept)

        beta_pred = max(beta_pred, 0)

        # Cosmological impact: δμ ≈ 5 × β × ln(3) (for Z range 0.3 to 3.0 Z_sun)
        delta_mu = 5 * beta_pred * np.log(3)

        if beta_pred > 0.02:
            status = "STRONG"
        elif beta_pred > 0.005:
            status = "VIABLE"
        elif beta_pred > 0.001:
            status = "MARGINAL"
        else:
            status = "WASHED OUT"

        print(f"  {N:>10}³ | {beta_pred:13.4f} | {delta_mu:8.3f} | {status}")

    # =========================================================
    # Step 5: Verdict
    # =========================================================
    print("\n" + "=" * 60)
    print("VERDICT: Crisis Assessment")
    print("=" * 60)

    beta_inf = fit_result['beta_inf']

    print(f"\n  Asymptotic β_∞ = {beta_inf:.4f}")
    print()

    if beta_inf > 0.01:
        verdict = "SPANDREL SURVIVES"
        explanation = (
            "The metallicity-fractal coupling persists at high resolution.\n"
            "  The cosmological bias remains significant (δμ > 0.05 mag).\n"
            "  DESI phantom signal CAN be explained by Spandrel mechanism."
        )
    elif beta_inf > 0.003:
        verdict = "SPANDREL MARGINAL"
        explanation = (
            "The effect is weakened but non-zero.\n"
            "  Additional physics (age, CSM) may be needed to reach DESI.\n"
            "  Hero Run CRITICAL to confirm asymptotic behavior."
        )
    else:
        verdict = "TURBULENT WASHOUT"
        explanation = (
            "The metallicity effect is washed out by turbulent diffusion.\n"
            "  The Spandrel mechanism cannot explain DESI alone.\n"
            "  Must invoke progenitor age or alternative mechanisms."
        )

    print(f"  {verdict}")
    print()
    print(f"  {explanation}")

    # Save comprehensive results
    np.savez('data/processed/convergence_triangulation.npz',
             z=z_vals,
             w_eff=w_eff,
             H=H_vals,
             tension_sigma=tension_sigma)

    print("\n  Results saved to: convergence_triangulation.npz")

    return fit_result


if __name__ == "__main__":
    result = main()
    sys.exit(0)
