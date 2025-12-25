#!/usr/bin/env python3
"""
Example usage of H₀ covariance analysis.

This script demonstrates how to:
1. Run the full analysis
2. Access specific results
3. Perform custom calculations
4. Modify parameters
"""

import numpy as np
from h0_covariance import H0CovarianceAnalysis

# Example 1: Run full analysis with default settings
print("=" * 80)
print("EXAMPLE 1: Full Analysis")
print("=" * 80)
analysis = H0CovarianceAnalysis()
analysis.run_full_analysis()

# Example 2: Access individual fit results
print("\n" + "=" * 80)
print("EXAMPLE 2: Accessing Fit Results")
print("=" * 80)

h0_const, h0_const_err, chi2_null, dof_null = analysis.fit_constant_h0()
print(f"Constant model: H₀ = {h0_const:.3f} ± {h0_const_err:.3f} km/s/Mpc")
print(f"  χ²/dof = {chi2_null:.2f}/{dof_null} = {chi2_null/dof_null:.2f}")

h0_0, h0_0_err, slope, slope_err, chi2_alt, dof_alt = analysis.fit_gradient_model()
print(f"\nGradient model: H₀(k) = {h0_0:.3f} + {slope:.3f}×log₁₀(k)")
print(f"  H₀₀ = {h0_0:.3f} ± {h0_0_err:.3f} km/s/Mpc")
print(f"  m = {slope:.3f} ± {slope_err:.3f} km/s/Mpc/dex")
print(f"  χ²/dof = {chi2_alt:.2f}/{dof_alt} = {chi2_alt/dof_alt:.2f}")

# Example 3: Predict H₀ at specific scales
print("\n" + "=" * 80)
print("EXAMPLE 3: Predictions at Different Scales")
print("=" * 80)

scales_h_per_mpc = [0.01, 0.1, 1.0, 10.0]  # h/Mpc
scale_names = ["CMB acoustic", "BAO", "Intermediate", "Local"]

print(f"\nUsing gradient model: H₀(k) = {h0_0:.2f} + {slope:.2f}×log₁₀(k)")
print(f"\n{'Scale Name':<20} {'k [h/Mpc]':<12} {'log₁₀(k)':<10} {'H₀ [km/s/Mpc]':<15}")
print("-" * 70)

for k_value, name in zip(scales_h_per_mpc, scale_names):
    log_k = np.log10(k_value)
    h0_prediction = h0_0 + slope * log_k
    print(f"{name:<20} {k_value:<12.3f} {log_k:<10.2f} {h0_prediction:<15.2f}")

# Example 4: Calculate tension between specific measurements
print("\n" + "=" * 80)
print("EXAMPLE 4: Pairwise Tensions")
print("=" * 80)


def calculate_tension(idx1, idx2, analysis):
    """Calculate tension between two measurements accounting for covariance."""
    m1 = analysis.measurements[idx1]
    m2 = analysis.measurements[idx2]

    # Difference in measurements
    delta_h0 = m1.value - m2.value

    # Combined uncertainty from covariance matrix
    var_1 = analysis.covariance_matrix[idx1, idx1]
    var_2 = analysis.covariance_matrix[idx2, idx2]
    covar_12 = analysis.covariance_matrix[idx1, idx2]

    variance_diff = var_1 + var_2 - 2 * covar_12
    sigma_diff = np.sqrt(variance_diff)

    # Tension in sigma
    tension = abs(delta_h0) / sigma_diff

    return delta_h0, sigma_diff, tension


# Famous tensions
print("\nPlanck vs SH0ES (classic H₀ tension):")
planck_idx = 0  # Planck CMB
shoes_idx = 11  # SH0ES
delta, sigma, tension = calculate_tension(planck_idx, shoes_idx, analysis)
print(f"  ΔH₀ = {delta:.2f} ± {sigma:.2f} km/s/Mpc")
print(f"  Tension: {tension:.1f}σ")

print("\nACT vs SH0ES:")
act_idx = 1  # ACT DR6
delta, sigma, tension = calculate_tension(act_idx, shoes_idx, analysis)
print(f"  ΔH₀ = {delta:.2f} ± {sigma:.2f} km/s/Mpc")
print(f"  Tension: {tension:.1f}σ")

print("\nBOSS BAO vs DES combined:")
boss_idx = 2  # BOSS BAO
des_idx = 5  # DES Y3 combined
delta, sigma, tension = calculate_tension(boss_idx, des_idx, analysis)
print(f"  ΔH₀ = {delta:.2f} ± {sigma:.2f} km/s/Mpc")
print(f"  Tension: {tension:.1f}σ")

# Example 5: Examine correlation structure
print("\n" + "=" * 80)
print("EXAMPLE 5: Correlation Structure")
print("=" * 80)

correlation_matrix = analysis._get_correlation_matrix()

print("\nStrongest correlations:")
strong_correlations = []
for i in range(len(analysis.measurements)):
    for j in range(i + 1, len(analysis.measurements)):
        corr = correlation_matrix[i, j]
        if abs(corr) > 0.25:  # Threshold for "strong"
            strong_correlations.append((i, j, corr))

# Sort by absolute correlation
strong_correlations.sort(key=lambda x: abs(x[2]), reverse=True)

for i, j, corr in strong_correlations[:5]:  # Top 5
    print(
        f"  {analysis.measurements[i].name:<20} - {analysis.measurements[j].name:<20}: ρ = {corr:.2f}"
    )

# Example 6: Test significance of gradient detection
print("\n" + "=" * 80)
print("EXAMPLE 6: Gradient Detection Significance")
print("=" * 80)

# Gradient significance = slope / slope_error
gradient_significance = slope / slope_err
print(f"Gradient: m = {slope:.3f} ± {slope_err:.3f} km/s/Mpc/dex")
print(f"Detection significance: {gradient_significance:.1f}σ")
print(f"Null hypothesis (m = 0): rejected at {gradient_significance:.1f}σ")

# Predicted variation across full scale range
k_min = min(m.scale for m in analysis.measurements)
k_max = max(m.scale for m in analysis.measurements)
delta_log_k = np.log10(k_max / k_min)
delta_h0_total = slope * delta_log_k
delta_h0_error = slope_err * delta_log_k

print(f"\nScale range: {k_min:.3f} to {k_max:.1f} h/Mpc ({delta_log_k:.1f} decades)")
print(f"Predicted total variation: ΔH₀ = {delta_h0_total:.2f} ± {delta_h0_error:.2f} km/s/Mpc")

# Example 7: Custom output location
print("\n" + "=" * 80)
print("EXAMPLE 7: Custom Output Directory")
print("=" * 80)

# Uncomment to run with custom output
# custom_output = "/path/to/custom/output"
# analysis.run_full_analysis(output_dir=custom_output)
# print(f"Results saved to: {custom_output}")

print("\nFor custom output, uncomment lines in example and specify directory.")

# Example 8: Access covariance matrix elements
print("\n" + "=" * 80)
print("EXAMPLE 8: Covariance Matrix Structure")
print("=" * 80)

print(f"\nCovariance matrix shape: {analysis.covariance_matrix.shape}")
print(
    f"Matrix symmetry check: {np.allclose(analysis.covariance_matrix, analysis.covariance_matrix.T)}"
)

# Check positive definiteness
eigenvalues = np.linalg.eigvalsh(analysis.covariance_matrix)
print(f"Eigenvalues range: [{eigenvalues.min():.3f}, {eigenvalues.max():.3f}]")
print(f"Positive definite: {np.all(eigenvalues > 0)}")
print(f"Condition number: {eigenvalues.max() / eigenvalues.min():.1f}")

# Determinant
det = np.linalg.det(analysis.covariance_matrix)
print(f"Determinant: {det:.2e}")

print("\n" + "=" * 80)
print("Examples complete!")
print("=" * 80)
