#!/usr/bin/env python3
"""
Pantheon+ Hubble Bifurcation Analysis

Tests whether H₀ varies systematically with SALT3 stretch parameter x₁.
The Spandrel Metric hypothesis predicts:
- Low-stretch SNe (x₁ < 0, D < 2.18) → H₀ ≈ 75.27 km/s/Mpc
- High-stretch SNe (x₁ > 0, D > 2.18) → H₀ ≈ 71.25 km/s/Mpc
- ΔH₀ ≈ 4 km/s/Mpc (matching the Hubble tension!)
"""

import numpy as np
import pandas as pd
from scipy import stats

# Speed of light in km/s
SPEED_OF_LIGHT = 299792.458

def load_pantheon_data(filepath):
    """Load Pantheon+ data and extract relevant columns."""
    # Read the data file
    data = pd.read_csv(filepath, delim_whitespace=True, comment='#')

    # Select only the columns we need
    columns_needed = ['CID', 'zHD', 'x1', 'MU_SH0ES', 'MU_SH0ES_ERR_DIAG']
    df = data[columns_needed].copy()

    # Rename for clarity
    df.columns = ['SN_name', 'redshift', 'x1_stretch', 'distance_modulus', 'mu_error']

    # Remove any rows with missing data
    df = df.dropna()

    return df

def compute_hubble_parameter(redshift, distance_modulus):
    """
    Compute H₀ from redshift and distance modulus.

    For low-z SNe, the distance modulus is:
    μ = 5 log₁₀(d_L) + 25
    where d_L is luminosity distance in Mpc

    For small z, d_L ≈ c·z/H₀
    Therefore: H₀ ≈ c·z / d_L = c·z / 10^((μ-25)/5)
    """
    # Convert distance modulus to luminosity distance
    luminosity_distance = 10**((distance_modulus - 25) / 5)  # in Mpc

    # Compute H₀
    hubble_param = SPEED_OF_LIGHT * redshift / luminosity_distance  # in km/s/Mpc

    return hubble_param

def analyze_bifurcation(df, z_max=0.1):
    """
    Analyze the Hubble bifurcation by splitting on x₁.

    Parameters:
    -----------
    df : DataFrame
        Pantheon+ data
    z_max : float
        Maximum redshift for low-z sample (default 0.1)
    """

    print("=" * 80)
    print("PANTHEON+ HUBBLE BIFURCATION ANALYSIS")
    print("=" * 80)
    print()

    # Filter to low-z sample for H₀ estimation
    df_low_z = df[df['redshift'] <= z_max].copy()

    print(f"Total SNe in Pantheon+: {len(df)}")
    print(f"SNe with z ≤ {z_max}: {len(df_low_z)}")
    print()

    # Compute H₀ for low-z sample
    df_low_z['H0'] = compute_hubble_parameter(df_low_z['redshift'], df_low_z['distance_modulus'])

    # Split into LOW-D and HIGH-D populations based on x₁
    # x₁ = 0 corresponds to D ≈ 2.18 in the Spandrel Metric
    low_d_mask = df_low_z['x1_stretch'] < 0
    high_d_mask = df_low_z['x1_stretch'] > 0

    df_low_d = df_low_z[low_d_mask]
    df_high_d = df_low_z[high_d_mask]

    print("=" * 80)
    print("POPULATION STATISTICS (z ≤ {})".format(z_max))
    print("=" * 80)
    print()

    # LOW-D Population (x₁ < 0, D < 2.18)
    print("LOW-D POPULATION (x₁ < 0, D < 2.18):")
    print(f"  Sample size: {len(df_low_d)}")
    print(f"  Mean x₁: {df_low_d['x1_stretch'].mean():.4f} ± {df_low_d['x1_stretch'].std():.4f}")
    print(f"  Mean z: {df_low_d['redshift'].mean():.6f} ± {df_low_d['redshift'].std():.6f}")
    print(f"  Mean μ: {df_low_d['distance_modulus'].mean():.4f} ± {df_low_d['distance_modulus'].std():.4f}")
    print(f"  Mean H₀: {df_low_d['H0'].mean():.2f} ± {df_low_d['H0'].std():.2f} km/s/Mpc")
    print(f"  Median H₀: {df_low_d['H0'].median():.2f} km/s/Mpc")
    print()

    # HIGH-D Population (x₁ > 0, D > 2.18)
    print("HIGH-D POPULATION (x₁ > 0, D > 2.18):")
    print(f"  Sample size: {len(df_high_d)}")
    print(f"  Mean x₁: {df_high_d['x1_stretch'].mean():.4f} ± {df_high_d['x1_stretch'].std():.4f}")
    print(f"  Mean z: {df_high_d['redshift'].mean():.6f} ± {df_high_d['redshift'].std():.6f}")
    print(f"  Mean μ: {df_high_d['distance_modulus'].mean():.4f} ± {df_high_d['distance_modulus'].std():.4f}")
    print(f"  Mean H₀: {df_high_d['H0'].mean():.2f} ± {df_high_d['H0'].std():.2f} km/s/Mpc")
    print(f"  Median H₀: {df_high_d['H0'].median():.2f} km/s/Mpc")
    print()

    # Compute the difference
    delta_h0 = df_low_d['H0'].mean() - df_high_d['H0'].mean()

    print("=" * 80)
    print("HUBBLE BIFURCATION RESULTS")
    print("=" * 80)
    print()
    print(f"ΔH₀ = H₀(low-D) - H₀(high-D) = {delta_h0:.2f} km/s/Mpc")
    print()

    # Statistical test
    t_stat, p_value = stats.ttest_ind(df_low_d['H0'], df_high_d['H0'])
    print(f"Two-sample t-test:")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.6f}")
    print(f"  Significance: {('YES' if p_value < 0.05 else 'NO')} at 5% level")
    print()

    # Spandrel Metric prediction check
    print("=" * 80)
    print("SPANDREL METRIC PREDICTION CHECK")
    print("=" * 80)
    print()
    print("Predicted:")
    print("  H₀(low-D) ≈ 75.27 km/s/Mpc")
    print("  H₀(high-D) ≈ 71.25 km/s/Mpc")
    print("  ΔH₀ ≈ 4.0 km/s/Mpc")
    print()
    print("Observed:")
    print(f"  H₀(low-D) = {df_low_d['H0'].mean():.2f} km/s/Mpc")
    print(f"  H₀(high-D) = {df_high_d['H0'].mean():.2f} km/s/Mpc")
    print(f"  ΔH₀ = {delta_h0:.2f} km/s/Mpc")
    print()

    # Full sample statistics (all redshifts)
    print("=" * 80)
    print("FULL SAMPLE STATISTICS (all redshifts)")
    print("=" * 80)
    print()

    df_all_low_d = df[df['x1_stretch'] < 0]
    df_all_high_d = df[df['x1_stretch'] > 0]

    print("LOW-D POPULATION (x₁ < 0):")
    print(f"  Sample size: {len(df_all_low_d)}")
    print(f"  Mean x₁: {df_all_low_d['x1_stretch'].mean():.4f} ± {df_all_low_d['x1_stretch'].std():.4f}")
    print(f"  Mean z: {df_all_low_d['redshift'].mean():.4f} ± {df_all_low_d['redshift'].std():.4f}")
    print(f"  Mean μ: {df_all_low_d['distance_modulus'].mean():.4f} ± {df_all_low_d['distance_modulus'].std():.4f}")
    print()

    print("HIGH-D POPULATION (x₁ > 0):")
    print(f"  Sample size: {len(df_all_high_d)}")
    print(f"  Mean x₁: {df_all_high_d['x1_stretch'].mean():.4f} ± {df_all_high_d['x1_stretch'].std():.4f}")
    print(f"  Mean z: {df_all_high_d['redshift'].mean():.4f} ± {df_all_high_d['redshift'].std():.4f}")
    print(f"  Mean μ: {df_all_high_d['distance_modulus'].mean():.4f} ± {df_all_high_d['distance_modulus'].std():.4f}")
    print()

    # Save results to CSV files
    print("=" * 80)
    print("SAVING DATA FILES")
    print("=" * 80)
    print()

    # Save low-z samples
    df_low_d.to_csv('/Users/eirikr/cosmos/pantheon_low_d_population.csv', index=False)
    df_high_d.to_csv('/Users/eirikr/cosmos/pantheon_high_d_population.csv', index=False)

    # Save full samples
    df_all_low_d.to_csv('/Users/eirikr/cosmos/pantheon_all_low_d.csv', index=False)
    df_all_high_d.to_csv('/Users/eirikr/cosmos/pantheon_all_high_d.csv', index=False)

    print("Files saved:")
    print("  - pantheon_low_d_population.csv (low-z, x₁ < 0)")
    print("  - pantheon_high_d_population.csv (low-z, x₁ > 0)")
    print("  - pantheon_all_low_d.csv (all z, x₁ < 0)")
    print("  - pantheon_all_high_d.csv (all z, x₁ > 0)")
    print()

    return df_low_d, df_high_d

def main():
    """Main analysis function."""

    # Load Pantheon+ data
    # (Assuming local file exists, otherwise download)
    try:
        df = load_pantheon_data('data/raw/pantheon_plus_data.dat')
    except FileNotFoundError:
    print(f"Loaded {len(df)} supernovae")
    print()

    # Analyze bifurcation
    df_low_d, df_high_d = analyze_bifurcation(df, z_max=0.1)

    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == '__main__':
    main()
