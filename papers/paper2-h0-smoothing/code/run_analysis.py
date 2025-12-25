#!/usr/bin/env python3
"""
Run Complete H₀(R) Analysis
============================

This script demonstrates the full analysis workflow:
1. Assign physical scales to measurements
2. Generate ΛCDM mock realizations
3. Compute null distribution
4. Perform hypothesis test
5. Generate publication-quality figures

Usage:
    python run_analysis.py [--num-mocks 100] [--output-dir ./results]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Import local modules
from h0_smoothing_estimator import H0SmoothingEstimator, get_example_measurements
from lcdm_mock_generator import LCDMMockGenerator, plot_mock_h0_ensemble


def main(num_mocks: int = 100, output_dir: str = "./results"):
    """
    Execute complete H₀(R) analysis pipeline.

    Parameters
    ----------
    num_mocks : int
        Number of ΛCDM mock realizations
    output_dir : str
        Directory for output files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("H₀(R) SCALE-DEPENDENT EXPANSION ANALYSIS")
    print("=" * 80)
    print(f"Output directory: {output_path.absolute()}")
    print(f"Number of mocks: {num_mocks}")

    # ========================================================================
    # STEP 1: Assign physical scales to measurements
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: Scale Assignment")
    print("=" * 80)

    estimator = H0SmoothingEstimator(window_function="tophat")
    measurement_infos = get_example_measurements()

    measurements_calibration = []
    for info in measurement_infos:
        meas = estimator.assign_scale_to_measurement(info, scale_definition="calibration")
        measurements_calibration.append(meas)

    print(f"Assigned scales to {len(measurements_calibration)} measurements")

    # ========================================================================
    # STEP 2: Compute observed H₀(R) trend
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: Observed H₀(R) Trend")
    print("=" * 80)

    radii_obs = np.array([m.radius_mpc for m in measurements_calibration])
    h0_values_obs = np.array([m.h0_value for m in measurements_calibration])
    h0_errors_obs = np.array([m.h0_uncertainty for m in measurements_calibration])
    log_radii_obs = np.log10(radii_obs)

    # Linear regression: H₀(R) = a + b × log₁₀(R)
    coeffs_obs = np.polyfit(log_radii_obs, h0_values_obs, deg=1, w=1.0 / h0_errors_obs)
    slope_obs, intercept_obs = coeffs_obs[0], coeffs_obs[1]

    print(f"Observed trend: H₀ = {intercept_obs:.2f} + {slope_obs:.2f} × log₁₀(R/Mpc)")
    print(f"  Slope: {slope_obs:.3f} ± [error from mock comparison] km/s/Mpc/decade")

    # ========================================================================
    # STEP 3: Generate ΛCDM mocks
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: ΛCDM Mock Generation")
    print("=" * 80)

    generator = LCDMMockGenerator(
        h0_fiducial=67.4, omega_m=0.315, sigma8=0.81, box_size_mpc=1000.0, random_seed=42
    )

    # Use observed radii for mock computation
    radii_for_mocks = np.logspace(0.5, 4.5, 20)  # 3 to 30000 Mpc

    print(f"Generating {num_mocks} mock realizations...")
    mocks = generator.generate_mock_h0_curves(
        num_mocks=num_mocks, radii_mpc=radii_for_mocks, num_observers=50, geometry="spherical"
    )

    # ========================================================================
    # STEP 4: Compute null distribution
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: Null Distribution Statistics")
    print("=" * 80)

    stats = generator.compute_null_statistics(mocks)

    print("Slope distribution (ΛCDM null):")
    print(f"  Mean:     {stats['slope_mean']:.4f} km/s/Mpc/decade")
    print(f"  Std dev:  {stats['slope_std']:.4f} km/s/Mpc/decade")
    print(f"  Median:   {np.median(stats['slopes']):.4f} km/s/Mpc/decade")
    print(
        f"  95% CI:   [{np.percentile(stats['slopes'], 2.5):.4f}, "
        f"{np.percentile(stats['slopes'], 97.5):.4f}]"
    )

    # ========================================================================
    # STEP 5: Hypothesis test
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 5: Hypothesis Test")
    print("=" * 80)

    # p-value: fraction of mocks with |slope| ≥ |observed slope|
    p_value = np.mean(np.abs(stats["slopes"]) >= np.abs(slope_obs))
    significance_sigma = np.abs(slope_obs - stats["slope_mean"]) / stats["slope_std"]

    print(f"Observed slope: {slope_obs:.4f} km/s/Mpc/decade")
    print(f"Null mean:      {stats['slope_mean']:.4f} km/s/Mpc/decade")
    print(f"Null std:       {stats['slope_std']:.4f} km/s/Mpc/decade")
    print(f"\nSignificance:   {significance_sigma:.2f}σ")
    print(f"p-value:        {p_value:.4f}")

    if p_value < 0.01:
        interpretation = "DETECTION: Trend exceeds ΛCDM expectations"
    elif p_value < 0.05:
        interpretation = "MARGINAL: Weak evidence for trend"
    else:
        interpretation = "NULL: Consistent with ΛCDM cosmic variance"

    print(f"\n*** {interpretation} ***")

    # ========================================================================
    # STEP 6: Generate visualizations
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 6: Visualization")
    print("=" * 80)

    # Plot 1: Mock ensemble
    plot_mock_h0_ensemble(mocks, stats, output_path=str(output_path / "mock_h0_ensemble.png"))

    # Plot 2: Observed vs mocks
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: H₀(R) with observed points
    ax = axes[0]
    ax.errorbar(
        log_radii_obs,
        h0_values_obs,
        yerr=h0_errors_obs,
        fmt="o",
        markersize=10,
        capsize=5,
        capthick=2,
        color="red",
        label="Observed",
        zorder=10,
    )

    # Overlay mock percentiles
    log_radii_mocks = np.log10(stats["radii_mpc"])
    ax.fill_between(
        log_radii_mocks,
        stats["h0_percentiles"]["16"],
        stats["h0_percentiles"]["84"],
        alpha=0.3,
        color="blue",
        label="ΛCDM 68% CI",
    )

    # Observed fit line
    log_r_line = np.linspace(log_radii_obs.min() - 0.5, log_radii_obs.max() + 0.5, 100)
    h0_fit_line = intercept_obs + slope_obs * log_r_line
    ax.plot(log_r_line, h0_fit_line, "r--", linewidth=2, label=f"Observed: m = {slope_obs:.2f}")

    ax.set_xlabel("log₁₀(R [Mpc])", fontsize=12)
    ax.set_ylabel("H₀ [km/s/Mpc]", fontsize=12)
    ax.set_title("H₀(R) vs Smoothing Scale", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Right: Slope comparison
    ax = axes[1]
    ax.hist(
        stats["slopes"], bins=30, alpha=0.7, color="blue", edgecolor="black", label="ΛCDM mocks"
    )
    ax.axvline(
        slope_obs, color="red", linewidth=3, linestyle="-", label=f"Observed: {slope_obs:.3f}"
    )
    ax.axvline(
        stats["slope_mean"],
        color="blue",
        linewidth=2,
        linestyle="--",
        label=f'Null mean: {stats["slope_mean"]:.3f}',
    )
    ax.axvline(
        stats["slope_mean"] - 2 * stats["slope_std"],
        color="gray",
        linewidth=1,
        linestyle=":",
        label="±2σ",
    )
    ax.axvline(
        stats["slope_mean"] + 2 * stats["slope_std"], color="gray", linewidth=1, linestyle=":"
    )

    ax.set_xlabel("Slope [km/s/Mpc per decade]", fontsize=12)
    ax.set_ylabel("Number of Mocks", fontsize=12)
    ax.set_title(f"Null Distribution (p = {p_value:.4f})", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "h0_vs_scale_comparison.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path / 'h0_vs_scale_comparison.png'}")

    # ========================================================================
    # STEP 7: Save results
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 7: Save Results")
    print("=" * 80)

    import json

    results_summary = {
        "observed": {
            "slope": float(slope_obs),
            "intercept": float(intercept_obs),
            "measurements": len(measurements_calibration),
        },
        "null_distribution": {
            "num_mocks": num_mocks,
            "slope_mean": float(stats["slope_mean"]),
            "slope_std": float(stats["slope_std"]),
            "slope_median": float(np.median(stats["slopes"])),
            "slope_95ci": [
                float(np.percentile(stats["slopes"], 2.5)),
                float(np.percentile(stats["slopes"], 97.5)),
            ],
        },
        "hypothesis_test": {
            "p_value": float(p_value),
            "significance_sigma": float(significance_sigma),
            "interpretation": interpretation,
        },
    }

    with open(output_path / "analysis_results.json", "w") as f:
        json.dump(results_summary, f, indent=2)

    print(f"Saved: {output_path / 'analysis_results.json'}")

    # ========================================================================
    # Final summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nResults:")
    print(f"  Observed slope:       {slope_obs:.4f} km/s/Mpc/decade")
    print(f"  ΛCDM expectation:     {stats['slope_mean']:.4f} ± {stats['slope_std']:.4f}")
    print(f"  Significance:         {significance_sigma:.2f}σ")
    print(f"  p-value:              {p_value:.4f}")
    print(f"\n  *** {interpretation} ***")
    print("\nOutput files:")
    print(f"  {output_path / 'analysis_results.json'}")
    print(f"  {output_path / 'mock_h0_ensemble.png'}")
    print(f"  {output_path / 'h0_vs_scale_comparison.png'}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run complete H₀(R) scale-dependent analysis")
    parser.add_argument(
        "--num-mocks", type=int, default=100, help="Number of ΛCDM mock realizations (default: 100)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Output directory for results (default: ./results)",
    )

    args = parser.parse_args()

    main(num_mocks=args.num_mocks, output_dir=args.output_dir)
