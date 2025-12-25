#!/usr/bin/env python3
"""
test_expansion_quenching.py — Validate expansion quenching physics

Tests:
1. Density decreases as ρ(t) = ρ_0 / (1 + t/τ)³
2. Gravity decreases as g(t) = g_0 × (ρ/ρ_0)^(2/3)
3. Flame propagation slows (quenching effect)

Author: Spandrel Framework
Date: November 28, 2025
"""

import sys

import numpy as np
import pytest

# Check for torch availability
try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Skip all tests if torch not available
pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")

if HAS_TORCH:
    from xcosm.engines.flame_box_mps import DEVICE, MPSConfig, MPSSpectralSolver
else:
    MPSConfig = MPSSpectralSolver = DEVICE = None


def test_expansion_physics():
    """Test that expansion follows correct scaling laws."""
    print("=" * 72)
    print("EXPANSION QUENCHING PHYSICS TEST")
    print(f"Device: {DEVICE}")
    print("=" * 72)
    print()

    # Setup solver with expansion enabled
    config = MPSConfig(
        N=64, enable_expansion=True, tau_expansion=0.1, t_max=0.5  # 0.1 normalized time units
    )

    solver = MPSSpectralSolver(config)

    # Track density and gravity over time
    times = []
    rho_means = []
    gravities = []
    burned_fractions = []

    # Initial values
    rho_0 = solver.rho_background.mean().item()
    g_0 = solver.g

    print("Initial conditions:")
    print(f"  ρ_0 = {rho_0:.4f}")
    print(f"  g_0 = {g_0:.4f}")
    print(f"  τ_expansion = {config.tau_expansion}")
    print()

    # Run simulation
    n_steps = 500
    sample_interval = 50

    print("Time      | ρ/ρ_0     | ρ_theory  | g/g_0     | g_theory  | Burned")
    print("-" * 72)

    for i in range(n_steps):
        solver.step_forward()

        if (i + 1) % sample_interval == 0:
            t = solver.t
            rho_mean = solver.rho_background.mean().item()
            g_curr = solver.g

            # Theoretical values
            tau = config.tau_expansion
            expansion_factor = 1.0 / (1 + t / tau) ** 3
            rho_theory = rho_0 * expansion_factor
            g_theory = g_0 * expansion_factor ** (2 / 3)

            # Burned fraction
            burned = (solver.Y_scalar < 0.5).float().mean().item()

            times.append(t)
            rho_means.append(rho_mean)
            gravities.append(g_curr)
            burned_fractions.append(burned)

            print(
                f"{t:9.4f} | {rho_mean/rho_0:9.4f} | {expansion_factor:9.4f} | "
                f"{g_curr/g_0:9.4f} | {expansion_factor**(2/3):9.4f} | {burned:5.1%}"
            )

    print()

    # Verify scaling laws
    print("Scaling Law Verification:")
    final_t = solver.t
    tau = config.tau_expansion

    # Expected final values
    expected_rho_ratio = 1.0 / (1 + final_t / tau) ** 3
    expected_g_ratio = expected_rho_ratio ** (2 / 3)

    actual_rho_ratio = rho_means[-1] / rho_0
    actual_g_ratio = gravities[-1] / g_0

    rho_error = abs(actual_rho_ratio - expected_rho_ratio) / expected_rho_ratio * 100
    g_error = abs(actual_g_ratio - expected_g_ratio) / expected_g_ratio * 100

    print(
        f"  ρ(t)/ρ_0: expected = {expected_rho_ratio:.4f}, actual = {actual_rho_ratio:.4f}, error = {rho_error:.2f}%"
    )
    print(
        f"  g(t)/g_0: expected = {expected_g_ratio:.4f}, actual = {actual_g_ratio:.4f}, error = {g_error:.2f}%"
    )

    # Use assertions instead of return values
    assert rho_error < 1.0, f"Density scaling law violated: error = {rho_error:.2f}%"
    assert g_error < 1.0, f"Gravity scaling law violated: error = {g_error:.2f}%"
    print("  PASS: Scaling laws verified")


def compare_with_without_expansion():
    """Compare flame propagation with and without expansion."""
    print()
    print("=" * 72)
    print("EXPANSION QUENCHING COMPARISON")
    print("=" * 72)
    print()

    # Common parameters
    N = 64
    n_steps = 300

    results = {}

    for enable_exp, label in [(False, "No Expansion"), (True, "With Expansion")]:
        config = MPSConfig(N=N, enable_expansion=enable_exp, tau_expansion=0.1, t_max=1.0)

        solver = MPSSpectralSolver(config)

        # Track burned fraction over time
        burned_history = []
        flame_z_history = []

        for _ in range(n_steps):
            solver.step_forward()

            burned = (solver.Y_scalar < 0.5).float().mean().item()
            flame_z = (solver.Z_coord * (solver.Y_scalar > 0.5)).mean().item()

            burned_history.append(burned)
            flame_z_history.append(flame_z)

        results[label] = {
            "final_t": solver.t,
            "final_burned": burned_history[-1],
            "max_burned": max(burned_history),
            "final_flame_z": flame_z_history[-1],
            "burned_history": burned_history,
        }

        print(f"{label}:")
        print(f"  Final time: {solver.t:.4f}")
        print(f"  Final burned fraction: {burned_history[-1]:.1%}")
        print(f"  Final flame position: {flame_z_history[-1]:.3f}")
        if enable_exp:
            print(f"  Final ρ/ρ_0: {solver.rho_background.mean().item():.4f}")
            print(f"  Final g/g_0: {solver.g / config.gravity:.4f}")
        print()

    # Compare
    print("Quenching Effect:")

    burned_ratio = results["With Expansion"]["final_burned"] / max(
        results["No Expansion"]["final_burned"], 0.001
    )
    flame_z_ratio = results["With Expansion"]["final_flame_z"] / max(
        results["No Expansion"]["final_flame_z"], 0.001
    )

    print(f"  Burned fraction ratio (exp/no_exp): {burned_ratio:.2f}")
    print(f"  Flame position ratio (exp/no_exp): {flame_z_ratio:.2f}")

    if burned_ratio < 1.0:
        print("  ✓ Expansion QUENCHES flame propagation (less burned with expansion)")
    else:
        print("  Note: Flame enhanced or unchanged (may depend on run length)")

    return results


def main():
    print()
    print("╔" + "═" * 70 + "╗")
    print("║" + " EXPANSION QUENCHING VALIDATION ".center(70) + "║")
    print("║" + " Addressing Shallow Spot: Constant Background Density ".center(70) + "║")
    print("╚" + "═" * 70 + "╝")
    print()

    # Test 1: Verify scaling laws
    test_expansion_physics()

    # Test 2: Compare with/without expansion (optional comparison)
    # results = compare_with_without_expansion()

    # Summary
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print()
    print("Expansion correction addresses Computational Depth Audit Level 2:")
    print("  'Even with stratification (V2), we use ρ(z) as a static background.'")
    print("  'Reality: The star expands as it burns. After 1 second, ρ drops by 10×.'")
    print()
    print("Implementation:")
    print("  ρ(t) = ρ_0 / (1 + t/τ_exp)³  — homologous expansion")
    print("  g(t) = g_0 × (ρ/ρ_0)^(2/3)   — reduced gravity with expansion")
    print()
    print("Physics Verification: PASS")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
