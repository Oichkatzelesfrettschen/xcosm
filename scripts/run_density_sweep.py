#!/usr/bin/env python3
"""
run_density_sweep.py — Phase 4: The Age of Fire

Test the hypothesis that Progenitor Age (via central density ρ_c)
drives the fractal dimension D independently of metallicity.

Physical Chain:
    Age → ρ_c → gravity (g) → Atwood number (At) → turbulence → D

Key Question:
    Is dD/dρ_c positive, negative, or zero?

    - If dD/dρ_c < 0: Young (low ρ_c) → High D → Fainter → wₐ < 0 ✓
    - If dD/dρ_c > 0: Young (low ρ_c) → Low D → Brighter → wₐ > 0 ✗
    - If dD/dρ_c = 0: Age mechanism fails

Mapping Age to Density:
    - Young progenitors (high z): Accrete fast, ignite early
      → ρ_c ~ 1×10⁹ g/cm³

    - Old progenitors (low z): Simmer longer, Urca cooling
      → ρ_c ~ 5×10⁹ g/cm³

Author: Spandrel Framework
Date: November 28, 2025
"""

import sys
import time
import numpy as np
from typing import Dict, List, Tuple

import torch
from cosmos.engines.flame_box_mps import MPSConfig, MPSSpectralSolver, DEVICE

# Density values to sweep (normalized units, proportional to ρ_c)
# These scale gravity and Atwood number
DENSITY_SCALE_VALUES = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]

# Mapping to physical interpretation
# rho_scale = 1.0 corresponds to ρ_c ~ 2×10⁹ g/cm³ (reference)
def physical_density(rho_scale: float) -> float:
    """Convert normalized density to physical units."""
    return 2e9 * rho_scale  # g/cm³


def age_from_density(rho_c: float) -> float:
    """Estimate progenitor age from central density.

    Higher ρ_c → longer simmering → older progenitor.
    Rough scaling from Piro & Bildsten 2008.
    """
    # Age in Gyr, rough fit
    rho_9 = rho_c / 1e9
    age = 1.0 + 2.0 * np.log(rho_9)  # Gyr
    return max(age, 0.5)


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
        n_boxes = N // size
        count = 0
        for i in range(n_boxes):
            for j in range(n_boxes):
                for k in range(n_boxes):
                    box = flame[i*size:(i+1)*size,
                               j*size:(j+1)*size,
                               k*size:(k+1)*size]
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


def run_single_density(rho_scale: float, N: int = 64, n_steps: int = 400) -> Dict:
    """Run simulation at given density scale."""

    config = MPSConfig(
        N=N,
        Z_metallicity=1.0,  # Fixed at solar (isolate density effect)
        rho_scale=rho_scale,
        enable_expansion=True,
        tau_expansion=0.15,
        enable_baroclinic=True,
        enable_stratification=True
    )

    solver = MPSSpectralSolver(config)

    # Track D evolution
    D_samples = []
    KE_samples = []
    sample_interval = 100

    start = time.perf_counter()

    for step in range(n_steps):
        solver.step_forward()

        if (step + 1) % sample_interval == 0:
            D = box_counting_dimension(solver.Y_scalar)
            KE = 0.5 * (solver.u**2 + solver.v**2 + solver.w**2).mean().item()
            D_samples.append(D)
            KE_samples.append(KE)

    elapsed = time.perf_counter() - start

    D_final = np.mean(D_samples[-2:]) if len(D_samples) >= 2 else D_samples[-1]
    D_std = np.std(D_samples[-2:]) if len(D_samples) >= 2 else 0.0
    KE_final = np.mean(KE_samples[-2:]) if len(KE_samples) >= 2 else KE_samples[-1]

    return {
        'rho_scale': rho_scale,
        'rho_physical': physical_density(rho_scale),
        'age': age_from_density(physical_density(rho_scale)),
        'gravity': config.gravity,
        'D_final': D_final,
        'D_std': D_std,
        'KE_final': KE_final,
        'elapsed': elapsed
    }


def main():
    print()
    print("╔" + "═" * 62 + "╗")
    print("║" + " PHASE 4: THE AGE OF FIRE ".center(62) + "║")
    print("║" + " Testing D(ρ_c) — The Density-Fractal Coupling ".center(62) + "║")
    print("╚" + "═" * 62 + "╝")
    print()
    print(f"Device: {DEVICE}")
    print()

    print("Physical Mapping:")
    print("  Young progenitors (high z) → Low ρ_c → Low gravity")
    print("  Old progenitors (low z)    → High ρ_c → High gravity")
    print()
    print("Hypothesis:")
    print("  If dD/dρ_c < 0: Young → High D → Fainter → wₐ < 0 (GOOD)")
    print("  If dD/dρ_c > 0: Young → Low D → Brighter → wₐ > 0 (BAD)")
    print()

    # Run density sweep
    results = []

    for rho_scale in DENSITY_SCALE_VALUES:
        rho_phys = physical_density(rho_scale)
        age = age_from_density(rho_phys)
        print(f"Running ρ_scale = {rho_scale:.2f} (ρ_c = {rho_phys:.1e} g/cm³, age ~ {age:.1f} Gyr)...")

        result = run_single_density(rho_scale, N=64, n_steps=400)
        results.append(result)

        print(f"  D = {result['D_final']:.3f} ± {result['D_std']:.3f}, "
              f"KE = {result['KE_final']:.4f}, time = {result['elapsed']:.1f}s")

    # Analysis
    print()
    print("=" * 64)
    print("DENSITY SWEEP RESULTS")
    print("=" * 64)
    print()
    print("ρ_scale | ρ_c (g/cm³) | Age (Gyr) | g     | D      | KE")
    print("-" * 64)

    for r in results:
        print(f"{r['rho_scale']:7.2f} | {r['rho_physical']:11.2e} | "
              f"{r['age']:9.1f} | {r['gravity']:.3f} | {r['D_final']:.3f} | {r['KE_final']:.4f}")

    # Fit D(ρ) relation
    rho_arr = np.array([r['rho_scale'] for r in results])
    D_arr = np.array([r['D_final'] for r in results])

    # Fit: D = D_ref + γ × ln(ρ/ρ_ref)
    log_rho = np.log(rho_arr)
    slope, intercept = np.polyfit(log_rho, D_arr, 1)

    gamma = slope  # dD/d(ln ρ)
    D_ref = intercept

    print()
    print("=" * 64)
    print("D(ρ) SCALING RELATION")
    print("=" * 64)
    print()
    print(f"  D(ρ) = {D_ref:.3f} + {gamma:.3f} × ln(ρ/ρ_ref)")
    print()
    print(f"  γ = dD/d(ln ρ) = {gamma:.4f}")
    print()

    # Interpret sign
    print("=" * 64)
    print("INTERPRETATION")
    print("=" * 64)
    print()

    if gamma < -0.01:
        print("  γ < 0: Higher density → LOWER D")
        print()
        print("  Physical chain:")
        print("    Young (high z) → Low ρ_c → High D → More wrinkling")
        print("    → Faster deflagration → DDT at lower density")
        print("    → Less Ni-56 → FAINTER SNe")
        print()
        print("  Cosmological implication:")
        print("    High-z SNe appear FAINTER → wₐ < 0 ✓")
        print()
        print("  ✓ AGE MECHANISM CONFIRMED")
        print("  The Age of Fire can explain the DESI phantom signal!")
        age_works = True

    elif gamma > 0.01:
        print("  γ > 0: Higher density → HIGHER D")
        print()
        print("  Physical chain:")
        print("    Young (high z) → Low ρ_c → Low D → Less wrinkling")
        print("    → Slower deflagration → DDT at higher density")
        print("    → More Ni-56 → BRIGHTER SNe")
        print()
        print("  Cosmological implication:")
        print("    High-z SNe appear BRIGHTER → wₐ > 0 ✗")
        print()
        print("  ✗ AGE MECHANISM HAS WRONG SIGN")
        print("  Need to reconsider the physics!")
        age_works = False

    else:
        print("  γ ≈ 0: Density does NOT affect D significantly")
        print()
        print("  The Age mechanism is INEFFECTIVE.")
        print("  Must rely on Selection Effects alone.")
        age_works = False

    # Compute cosmological contribution
    print()
    print("=" * 64)
    print("COSMOLOGICAL CONTRIBUTION")
    print("=" * 64)
    print()

    # At z=1, progenitors are ~3 Gyr younger → ρ_c is ~0.5× lower
    delta_ln_rho = np.log(0.5)  # ln(ρ_young / ρ_old)
    delta_D_age = gamma * delta_ln_rho

    # Convert to magnitude: δμ ≈ κ × ΔD with κ ≈ 0.5 mag/D-unit
    kappa = 0.5
    delta_mu_age = kappa * delta_D_age

    print(f"  At z=1 vs z=0:")
    print(f"    Δ(ln ρ) = {delta_ln_rho:.2f} (younger progenitors)")
    print(f"    ΔD = γ × Δ(ln ρ) = {delta_D_age:.3f}")
    print(f"    δμ = κ × ΔD = {delta_mu_age:.3f} mag")
    print()

    if abs(delta_mu_age) > 0.02:
        print(f"  Age contributes δμ ~ {abs(delta_mu_age):.2f} mag")
        print("  This is SIGNIFICANT for the multi-mechanism budget!")
    else:
        print(f"  Age contributes δμ ~ {abs(delta_mu_age):.2f} mag")
        print("  This is MARGINAL — Selection Effects must dominate.")

    # Save results
    np.savez('data/processed/density_sweep_results.npz',
             rho_scale=[r['rho_scale'] for r in results],
             rho_physical=[r['rho_physical'] for r in results],
             age=[r['age'] for r in results],
             D=[r['D_final'] for r in results],
             gamma=gamma,
             D_ref=D_ref,
             age_works=age_works)

    print()
    print("  Results saved to: data/processed/density_sweep_results.npz")

    return age_works


if __name__ == "__main__":
    result = main()
    sys.exit(0 if result else 1)
