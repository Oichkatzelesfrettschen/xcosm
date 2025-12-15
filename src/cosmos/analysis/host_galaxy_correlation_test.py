#!/usr/bin/env python3
"""
Host Galaxy Correlation Test - Red Team Validation of Spandrel Framework

Tests whether SALT3 stretch (x₁) correlates with host galaxy properties,
and whether the "Spandrel signal" is just known systematics relabeled.

Key Question: Does x₁ (proxy for fractal dimension D) correlate with:
- Host galaxy stellar mass (the known "mass step")?
- Or does it represent independent physics?

References:
- Pantheon+ (Brout et al. 2022, ApJ 938, 110)
- Nicolas et al. 2021 (A&A 649, A74) - 5σ stretch evolution
- Rigault et al. 2020 (A&A 644, A176) - sSFR correlation
- Son et al. 2025 (MNRAS 544, 975) - 5.5σ age correlation
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import urllib.request
import os

# Constants
DATA_URL = "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES.dat"
LOCAL_FILE = "data/raw/Pantheon+SH0ES.dat"

def download_pantheon_data():
    """Download Pantheon+ data if not already present."""
    if not os.path.exists(LOCAL_FILE):
        print("Downloading Pantheon+ data from GitHub...")
        urllib.request.urlretrieve(DATA_URL, LOCAL_FILE)
        print(f"Saved to {LOCAL_FILE}")
    else:
        print(f"Using cached data: {LOCAL_FILE}")
    return LOCAL_FILE

def load_pantheon_data(filepath):
    """Load Pantheon+ data with all columns."""
    print("Loading Pantheon+ data...")

    # Read space-delimited data
    df = pd.read_csv(filepath, delim_whitespace=True, comment='#')

    print(f"Loaded {len(df)} supernovae")
    print(f"Columns: {list(df.columns)}")

    return df

def analyze_stretch_mass_correlation(df):
    """
    Test 1: Does stretch (x₁) correlate with host galaxy mass?

    If yes: Spandrel's "D" may just be a proxy for known mass-step systematics
    If no: Spandrel captures independent physics
    """
    print("\n" + "="*80)
    print("TEST 1: STRETCH (x₁) vs HOST GALAXY STELLAR MASS")
    print("="*80)

    # Filter valid data
    valid_mask = (
        df['HOST_LOGMASS'].notna() &
        df['x1'].notna() &
        (df['HOST_LOGMASS'] > 0) &  # Exclude placeholder values
        (df['HOST_LOGMASS'] < 15)   # Reasonable mass range
    )
    df_valid = df[valid_mask].copy()
    print(f"\nSNe with valid host mass: {len(df_valid)} / {len(df)}")

    # Extract data
    x1_values = df_valid['x1'].values
    host_mass = df_valid['HOST_LOGMASS'].values

    # Compute correlations
    pearson_r, pearson_p = pearsonr(x1_values, host_mass)
    spearman_r, spearman_p = spearmanr(x1_values, host_mass)

    print(f"\nCorrelation Results:")
    print(f"  Pearson r  = {pearson_r:.4f} (p = {pearson_p:.2e})")
    print(f"  Spearman ρ = {spearman_r:.4f} (p = {spearman_p:.2e})")

    # Interpretation
    if abs(pearson_r) > 0.3:
        print(f"\n  ⚠️  STRONG CORRELATION: x₁ is likely a proxy for host mass")
        print(f"     The Spandrel 'D' signal may be the known mass-step relabeled")
    elif abs(pearson_r) > 0.1:
        print(f"\n  ⚠️  MODERATE CORRELATION: x₁ partially traces host mass")
        print(f"     The Spandrel framework captures BOTH mass effects AND additional physics")
    else:
        print(f"\n  ✓  WEAK/NO CORRELATION: x₁ is independent of host mass")
        print(f"     The Spandrel 'D' parameter captures genuinely new physics")

    # Split by mass to compare stretch distributions
    mass_threshold = 10.0  # log10(M/M_sun) = 10, standard mass-step threshold
    low_mass_mask = host_mass < mass_threshold
    high_mass_mask = host_mass >= mass_threshold

    x1_low_mass = x1_values[low_mass_mask]
    x1_high_mass = x1_values[high_mass_mask]

    print(f"\n  Mass Split Analysis (threshold = 10^10 M☉):")
    print(f"    Low-mass hosts:  N={len(x1_low_mass)}, <x₁> = {np.mean(x1_low_mass):.3f} ± {np.std(x1_low_mass):.3f}")
    print(f"    High-mass hosts: N={len(x1_high_mass)}, <x₁> = {np.mean(x1_high_mass):.3f} ± {np.std(x1_high_mass):.3f}")

    # T-test for difference
    t_stat, t_pval = stats.ttest_ind(x1_low_mass, x1_high_mass)
    delta_x1 = np.mean(x1_low_mass) - np.mean(x1_high_mass)

    print(f"    Δx₁ = {delta_x1:.3f} (t = {t_stat:.2f}, p = {t_pval:.2e})")

    if t_pval < 0.001:
        print(f"    ⚠️  Highly significant difference in stretch by host mass (>3σ)")

    return {
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'delta_x1': delta_x1,
        'mass_ttest_p': t_pval,
        'n_valid': len(df_valid)
    }

def analyze_stretch_redshift_evolution(df):
    """
    Test 2: Does stretch evolve with redshift?

    Nicolas et al. 2021 found 5σ evidence for x₁(z) evolution.
    This test reproduces that finding.
    """
    print("\n" + "="*80)
    print("TEST 2: STRETCH (x₁) EVOLUTION WITH REDSHIFT")
    print("="*80)

    # Filter valid data
    valid_mask = df['x1'].notna() & df['zHD'].notna() & (df['zHD'] > 0)
    df_valid = df[valid_mask].copy()

    x1_values = df_valid['x1'].values
    redshift = df_valid['zHD'].values

    # Compute correlation
    pearson_r, pearson_p = pearsonr(x1_values, redshift)
    spearman_r, spearman_p = spearmanr(x1_values, redshift)

    print(f"\nCorrelation with redshift:")
    print(f"  Pearson r  = {pearson_r:.4f} (p = {pearson_p:.2e})")
    print(f"  Spearman ρ = {spearman_r:.4f} (p = {spearman_p:.2e})")

    # Bin by redshift to show evolution
    z_bins = [0, 0.1, 0.3, 0.5, 0.7, 1.0, 2.5]
    print(f"\n  Redshift-binned stretch values:")

    for i in range(len(z_bins)-1):
        z_low, z_high = z_bins[i], z_bins[i+1]
        mask = (redshift >= z_low) & (redshift < z_high)
        if mask.sum() > 10:
            x1_bin = x1_values[mask]
            print(f"    z ∈ [{z_low:.1f}, {z_high:.1f}): N={mask.sum():4d}, <x₁> = {np.mean(x1_bin):+.3f} ± {np.std(x1_bin)/np.sqrt(mask.sum()):.3f}")

    # Compare low-z vs high-z (Nicolas et al. comparison)
    low_z_mask = redshift < 0.1
    high_z_mask = (redshift > 0.5) & (redshift < 1.0)

    x1_low_z = x1_values[low_z_mask]
    x1_high_z = x1_values[high_z_mask]

    if len(x1_high_z) > 10:
        delta_x1_z = np.mean(x1_high_z) - np.mean(x1_low_z)
        t_stat, t_pval = stats.ttest_ind(x1_low_z, x1_high_z)

        print(f"\n  Low-z (z<0.1) vs High-z (0.5<z<1.0):")
        print(f"    Low-z:  <x₁> = {np.mean(x1_low_z):.3f} (N={len(x1_low_z)})")
        print(f"    High-z: <x₁> = {np.mean(x1_high_z):.3f} (N={len(x1_high_z)})")
        print(f"    Δx₁ = {delta_x1_z:.3f} (p = {t_pval:.2e})")

        # Compare to Nicolas et al. prediction
        print(f"\n  Nicolas et al. 2021 prediction:")
        print(f"    Expected: <x₁>(z~0.05) = -0.17, <x₁>(z~0.65) = +0.34")
        print(f"    Expected Δx₁ ≈ 0.51")
        print(f"    Observed Δx₁ = {delta_x1_z:.3f}")

    return {
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p
    }

def analyze_mass_corrected_residuals(df):
    """
    Test 3: After controlling for host mass, does x₁ still predict Hubble residuals?

    This tests whether the Spandrel framework captures physics beyond the mass-step.
    """
    print("\n" + "="*80)
    print("TEST 3: HUBBLE RESIDUALS vs x₁ (CONTROLLING FOR HOST MASS)")
    print("="*80)

    # Filter valid data
    valid_mask = (
        df['HOST_LOGMASS'].notna() &
        df['x1'].notna() &
        df['MU_SH0ES'].notna() &
        df['zHD'].notna() &
        (df['HOST_LOGMASS'] > 0) &
        (df['HOST_LOGMASS'] < 15) &
        (df['zHD'] > 0.01) &
        (df['zHD'] < 0.15)  # Focus on Hubble flow regime
    )
    df_valid = df[valid_mask].copy()
    print(f"\nSNe in Hubble flow (0.01 < z < 0.15) with valid data: {len(df_valid)}")

    if len(df_valid) < 50:
        print("  Insufficient data for this test")
        return {}

    # Compute Hubble residuals (simplified)
    # Expected distance modulus for flat ΛCDM with H₀=70
    H0 = 70.0
    c = 299792.458  # km/s

    z = df_valid['zHD'].values
    mu_obs = df_valid['MU_SH0ES'].values

    # Simple approximation: μ = 5*log10(cz/H0) + 25
    mu_expected = 5 * np.log10(c * z / H0) + 25
    hubble_residual = mu_obs - mu_expected

    x1_values = df_valid['x1'].values
    host_mass = df_valid['HOST_LOGMASS'].values

    # Simple correlation: HR vs x₁
    r_hr_x1, p_hr_x1 = pearsonr(hubble_residual, x1_values)
    print(f"\n  Correlation: Hubble Residual vs x₁")
    print(f"    r = {r_hr_x1:.4f} (p = {p_hr_x1:.2e})")

    # Correlation: HR vs host mass
    r_hr_mass, p_hr_mass = pearsonr(hubble_residual, host_mass)
    print(f"\n  Correlation: Hubble Residual vs Host Mass")
    print(f"    r = {r_hr_mass:.4f} (p = {p_hr_mass:.2e})")

    # Partial correlation: HR vs x₁, controlling for mass
    # Using linear regression residuals
    from numpy.polynomial import polynomial as P

    # Regress out mass from both HR and x₁
    mass_centered = host_mass - np.mean(host_mass)

    # HR residuals after mass correction
    coeffs_hr = np.polyfit(mass_centered, hubble_residual, 1)
    hr_mass_corrected = hubble_residual - np.polyval(coeffs_hr, mass_centered)

    # x₁ residuals after mass correction
    coeffs_x1 = np.polyfit(mass_centered, x1_values, 1)
    x1_mass_corrected = x1_values - np.polyval(coeffs_x1, mass_centered)

    # Partial correlation
    r_partial, p_partial = pearsonr(hr_mass_corrected, x1_mass_corrected)

    print(f"\n  Partial Correlation: HR vs x₁ | controlling for Host Mass")
    print(f"    r_partial = {r_partial:.4f} (p = {p_partial:.2e})")

    if abs(r_partial) > 0.1 and p_partial < 0.05:
        print(f"\n  ✓ x₁ predicts Hubble residuals BEYOND the mass-step effect")
        print(f"    This supports the Spandrel framework capturing new physics")
    else:
        print(f"\n  ⚠️  x₁ effect on HR is fully explained by host mass")
        print(f"    The Spandrel 'D' may be redundant with mass-step correction")

    return {
        'r_hr_x1': r_hr_x1,
        'p_hr_x1': p_hr_x1,
        'r_hr_mass': r_hr_mass,
        'p_hr_mass': p_hr_mass,
        'r_partial': r_partial,
        'p_partial': p_partial
    }

def summarize_results(results):
    """Provide final verdict on the Spandrel Framework validation."""
    print("\n" + "="*80)
    print("FINAL ASSESSMENT: RED TEAM VALIDATION RESULTS")
    print("="*80)

    print("\n╔════════════════════════════════════════════════════════════════════════════╗")
    print("║                    SPANDREL FRAMEWORK v2.0 - RED TEAM VERDICT             ║")
    print("╠════════════════════════════════════════════════════════════════════════════╣")

    # Test 1 verdict
    if 'test1' in results:
        t1 = results['test1']
        if abs(t1.get('pearson_r', 0)) > 0.3:
            print("║ Test 1 (x₁ vs Mass):     ⚠️  CONCERN - Strong mass correlation detected   ║")
        elif abs(t1.get('pearson_r', 0)) > 0.1:
            print("║ Test 1 (x₁ vs Mass):     ⚡ MIXED - Moderate mass correlation             ║")
        else:
            print("║ Test 1 (x₁ vs Mass):     ✓  PASS - x₁ independent of host mass           ║")

    # Test 2 verdict
    if 'test2' in results:
        t2 = results['test2']
        if t2.get('pearson_p', 1) < 0.001:
            print("║ Test 2 (x₁ vs z):        ✓  CONFIRMED - Stretch evolves with redshift    ║")
        else:
            print("║ Test 2 (x₁ vs z):        ⚠️  WEAK - No significant z evolution           ║")

    # Test 3 verdict
    if 'test3' in results:
        t3 = results['test3']
        if t3.get('p_partial', 1) < 0.05 and abs(t3.get('r_partial', 0)) > 0.1:
            print("║ Test 3 (HR | Mass):      ✓  PASS - x₁ effect persists after mass control ║")
        else:
            print("║ Test 3 (HR | Mass):      ⚠️  CONCERN - x₁ effect absorbed by mass-step   ║")

    print("╠════════════════════════════════════════════════════════════════════════════╣")

    # Overall verdict
    concerns = 0
    if 'test1' in results and abs(results['test1'].get('pearson_r', 0)) > 0.3:
        concerns += 1
    if 'test3' in results and results['test3'].get('p_partial', 1) > 0.05:
        concerns += 1

    if concerns == 0:
        print("║ OVERALL VERDICT:         ✓  Framework captures genuinely new physics       ║")
    elif concerns == 1:
        print("║ OVERALL VERDICT:         ⚡ Framework requires refinement                  ║")
    else:
        print("║ OVERALL VERDICT:         ⚠️  Framework may be redundant with known effects ║")

    print("╚════════════════════════════════════════════════════════════════════════════╝")

def main():
    """Execute the full host galaxy correlation test suite."""
    print("╔════════════════════════════════════════════════════════════════════════════╗")
    print("║     RED TEAM VALIDATION: HOST GALAXY CORRELATION TEST (Test A)            ║")
    print("║     Testing whether Spandrel's 'D' is independent of known systematics    ║")
    print("╚════════════════════════════════════════════════════════════════════════════╝")

    # Download/load data
    filepath = download_pantheon_data()
    df = load_pantheon_data(filepath)

    # Run tests
    results = {}

    results['test1'] = analyze_stretch_mass_correlation(df)
    results['test2'] = analyze_stretch_redshift_evolution(df)
    results['test3'] = analyze_mass_corrected_residuals(df)

    # Final verdict
    summarize_results(results)

    return results

if __name__ == '__main__':
    results = main()
