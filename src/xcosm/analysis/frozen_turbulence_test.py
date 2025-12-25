#!/usr/bin/env python3
"""
frozen_turbulence_test.py — Control: Zero-Turbulence Baseline

If Spandrel physics is real, the MOLECULAR thermal diffusivity κ(Z)
should affect flame propagation even WITHOUT turbulent advection.

Test: Run with velocity_scale = 0 (no Navier-Stokes advection)
Prediction:
    - D = 2.0 (flat flame, no wrinkling)
    - But FLAME SPEED should vary with Z (higher Z → lower κ → slower flame)

This separates "molecular physics works" from "turbulence washes it out."

Author: Spandrel Framework
Date: November 28, 2025
"""

import sys
import time
from typing import Dict

import numpy as np
import torch

from xcosm.engines.flame_box_mps import DEVICE, MPSConfig, MPSSpectralSolver

METALLICITY_VALUES = [0.1, 0.3, 1.0, 3.0]


class FrozenTurbulenceSolver(MPSSpectralSolver):
    """MPS solver with turbulent advection disabled."""

    def __init__(self, config: MPSConfig):
        super().__init__(config)
        # Zero out initial velocities
        self.u = torch.zeros_like(self.u)
        self.v = torch.zeros_like(self.v)
        self.w = torch.zeros_like(self.w)

    def _compute_nonlinear(self):
        """Return zero advection term (frozen turbulence)."""
        zeros = torch.zeros_like(self.u)
        return zeros, zeros, zeros

    def _compute_baroclinic(self):
        """Return zero baroclinic term (no vorticity generation)."""
        zeros = torch.zeros_like(self.u)
        return zeros, zeros, zeros


def measure_flame_speed(solver: FrozenTurbulenceSolver, n_steps: int = 200) -> Dict:
    """Measure flame propagation speed in frozen turbulence."""

    # Track flame position over time
    times = []
    flame_positions = []
    burned_fractions = []

    for step in range(n_steps):
        solver.step_forward()

        if (step + 1) % 20 == 0:
            # Flame position = mean z where Y > 0.5
            mask = solver.Y_scalar > 0.5
            if mask.any():
                flame_z = (solver.Z_coord * mask.float()).sum() / mask.float().sum()
                flame_z = flame_z.item()
            else:
                flame_z = 0.3  # Initial position

            burned = (solver.Y_scalar < 0.5).float().mean().item()

            times.append(solver.t)
            flame_positions.append(flame_z)
            burned_fractions.append(burned)

    # Compute flame speed from position history
    times = np.array(times)
    positions = np.array(flame_positions)

    if len(times) > 2:
        # Linear fit to get speed
        slope, intercept = np.polyfit(times, positions, 1)
        flame_speed = slope
    else:
        flame_speed = 0.0

    return {
        "times": times,
        "positions": positions,
        "burned": burned_fractions,
        "flame_speed": flame_speed,
        "final_position": flame_positions[-1],
        "final_burned": burned_fractions[-1],
    }


def run_frozen_test(Z: float, N: int = 64) -> Dict:
    """Run frozen turbulence test for one metallicity."""

    config = MPSConfig(
        N=N,
        Z_metallicity=Z,
        enable_expansion=False,  # No expansion for clean test
        enable_baroclinic=False,
        enable_stratification=False,
        cfl=0.5,  # Can use larger CFL without advection
    )

    solver = FrozenTurbulenceSolver(config)

    start = time.perf_counter()
    result = measure_flame_speed(solver, n_steps=300)
    elapsed = time.perf_counter() - start

    result["Z"] = Z
    result["kappa"] = config.thermal_diffusivity
    result["elapsed"] = elapsed

    return result


def main():
    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + " FROZEN TURBULENCE CONTROL TEST ".center(58) + "║")
    print("║" + " Does κ(Z) affect flame propagation without turbulence? ".center(58) + "║")
    print("╚" + "═" * 58 + "╝")
    print()
    print(f"Device: {DEVICE}")
    print()

    print("Physics Test:")
    print("  - Navier-Stokes advection: DISABLED (u = 0)")
    print("  - Baroclinic vorticity: DISABLED")
    print("  - Buoyancy: DISABLED")
    print("  - Only: Reaction-Diffusion (Fisher-KPP)")
    print()
    print("Prediction:")
    print("  - Flame should be FLAT (D ≈ 2.0)")
    print("  - Flame SPEED should vary with κ(Z)")
    print("  - Higher Z → Lower κ → SLOWER flame")
    print()

    # Run tests
    results = []
    for Z in METALLICITY_VALUES:
        print(f"Running Z = {Z} Z_sun...")
        result = run_frozen_test(Z, N=64)
        results.append(result)
        print(
            f"  κ = {result['kappa']:.4f}, v_flame = {result['flame_speed']:.4f}, "
            f"burned = {result['final_burned']:.1%}"
        )

    # Analysis
    print()
    print("=" * 60)
    print("FROZEN TURBULENCE RESULTS")
    print("=" * 60)
    print()
    print("Z/Z_sun | κ (thermal) | v_flame | Δv/v_ref | Burned")
    print("-" * 60)

    kappa_ref = results[2]["kappa"]  # Z = 1.0
    v_ref = results[2]["flame_speed"]

    for r in results:
        dv = (r["flame_speed"] - v_ref) / v_ref if v_ref > 0 else 0
        print(
            f"{r['Z']:7.1f} | {r['kappa']:11.4f} | {r['flame_speed']:7.4f} | "
            f"{dv:+8.1%} | {r['final_burned']:6.1%}"
        )

    # Theoretical prediction: v_flame ∝ √(κ × A) for Fisher-KPP
    print()
    print("Theoretical Check (Fisher-KPP: v ∝ √κ):")
    print("-" * 60)

    kappa_values = np.array([r["kappa"] for r in results])
    v_values = np.array([r["flame_speed"] for r in results])

    # Fit v = C × κ^α
    if all(v > 0 for v in v_values):
        log_kappa = np.log(kappa_values)
        log_v = np.log(v_values)
        alpha, log_C = np.polyfit(log_kappa, log_v, 1)
        C = np.exp(log_C)

        print(f"  Fitted: v_flame = {C:.3f} × κ^{alpha:.3f}")
        print("  Theory: v_flame ∝ κ^0.5")
        print(f"  Agreement: {'GOOD' if 0.3 < alpha < 0.7 else 'POOR'}")
    else:
        print("  Insufficient flame propagation for fit")
        alpha = 0

    # Verdict
    print()
    print("=" * 60)
    print("VERDICT: Molecular Physics Check")
    print("=" * 60)
    print()

    # Check if flame speed varies with Z as expected
    v_low_Z = results[0]["flame_speed"]  # Z = 0.1
    v_high_Z = results[-1]["flame_speed"]  # Z = 3.0

    if v_low_Z > v_high_Z * 1.1:  # At least 10% faster at low Z
        print("  ✓ MOLECULAR PHYSICS CONFIRMED")
        print()
        print("  The metallicity effect EXISTS at the molecular level:")
        print(f"    - Low Z (0.1): v = {v_low_Z:.4f}")
        print(f"    - High Z (3.0): v = {v_high_Z:.4f}")
        print(f"    - Ratio: {v_low_Z/v_high_Z:.2f}× faster at low Z")
        print()
        print("  CONCLUSION: The Turbulent Washout is real, but the underlying")
        print("  physics is correct. The molecular-scale metallicity effect")
        print("  is overwhelmed by turbulent diffusion at high Reynolds number.")
        molecular_ok = True
    else:
        print("  ✗ MOLECULAR PHYSICS UNCERTAIN")
        print()
        print("  The flame speed does not vary significantly with metallicity.")
        print("  This suggests our κ(Z) parametrization may be too weak,")
        print("  or the reaction rate dominates over diffusion.")
        molecular_ok = False

    # TODO: Save results once cascade variables are computed
    # np.savez('data/processed/frozen_turbulence_results.npz',
    #          reynolds=reynolds_vals,
    #          energy=energy_vals,
    #          dissipation=dissipation_vals,
    #          kolmogorov=kolmogorov_vals)
    # print()
    # print("  Results saved to: frozen_turbulence_results.npz")

    return molecular_ok


if __name__ == "__main__":
    result = main()
    sys.exit(0 if result else 1)
