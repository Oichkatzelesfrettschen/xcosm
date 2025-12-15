#!/usr/bin/env python3
"""
Detailed Pantheon+ Hubble Bifurcation Analysis

More sophisticated analysis including:
- Multiple redshift bins
- Weighted averages
- Distance modulus residuals
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Speed of light in km/s
SPEED_OF_LIGHT = 299792.458

def load_pantheon_data(filepath):
    """Load Pantheon+ data and extract relevant columns."""
    data = pd.read_csv(filepath, sep='\s+', comment='#')

    columns_needed = ['CID', 'zHD', 'x1', 'MU_SH0ES', 'MU_SH0ES_ERR_DIAG']
    df = data[columns_needed].copy()
    df.columns = ['SN_name', 'redshift', 'x1_stretch', 'distance_modulus', 'mu_error']
    df = df.dropna()

    return df

def compute_hubble_parameter_binned(df, z_bins):
    """
    Compute H₀ for different redshift bins.
    """

    results = []

    for i in range(len(z_bins) - 1):
        z_min, z_max = z_bins[i], z_bins[i+1]
        z_center = (z_min + z_max) / 2

        # Filter to this redshift bin
        mask = (df['redshift'] >= z_min) & (df['redshift'] < z_max)
        df_bin = df[mask].copy()

        if len(df_bin) == 0:
            continue

        # Split by x₁
        df_low_d = df_bin[df_bin['x1_stretch'] < 0]
        df_high_d = df_bin[df_bin['x1_stretch'] > 0]

        # Compute H₀ for each population
        for population, df_pop, label in [
            ('LOW-D', df_low_d, 'x₁ < 0'),
            ('HIGH-D', df_high_d, 'x₁ > 0')
        ]:
            if len(df_pop) > 0:
                # Simple H₀ estimation
                luminosity_distance = 10**((df_pop['distance_modulus'] - 25) / 5)
                h0_values = SPEED_OF_LIGHT * df_pop['redshift'] / luminosity_distance

                results.append({
                    'z_min': z_min,
                    'z_max': z_max,
                    'z_center': z_center,
                    'population': population,
                    'label': label,
                    'n_sne': len(df_pop),
                    'mean_x1': df_pop['x1_stretch'].mean(),
                    'mean_z': df_pop['redshift'].mean(),
                    'mean_mu': df_pop['distance_modulus'].mean(),
                    'mean_h0': h0_values.mean(),
                    'std_h0': h0_values.std(),
                    'median_h0': h0_values.median(),
                })

    return pd.DataFrame(results)

def analyze_residuals(df):
    """
    Analyze distance modulus residuals as a function of x₁.

    The key insight: if there's a bifurcation, we should see systematic
    offsets in the Hubble diagram residuals between low-x₁ and high-x₁ SNe.
    """

    print("=" * 80)
    print("DISTANCE MODULUS RESIDUAL ANALYSIS")
    print("=" * 80)
    print()

    # For low-z, compute expected μ from mean H₀
    df_low_z = df[df['redshift'] <= 0.1].copy()

    # Assume a fiducial H₀ = 70 km/s/Mpc for Ω_M = 0.3
    # For low-z: μ ≈ 5 log₁₀(c·z/H₀) + 25
    H0_fiducial = 70.0
    df_low_z['mu_expected'] = 5 * np.log10(SPEED_OF_LIGHT * df_low_z['redshift'] / H0_fiducial) + 25
    df_low_z['mu_residual'] = df_low_z['distance_modulus'] - df_low_z['mu_expected']

    # Split by x₁
    low_d_residuals = df_low_z[df_low_z['x1_stretch'] < 0]['mu_residual']
    high_d_residuals = df_low_z[df_low_z['x1_stretch'] > 0]['mu_residual']

    print(f"Using fiducial H₀ = {H0_fiducial} km/s/Mpc")
    print()
    print(f"LOW-D (x₁ < 0) residuals:")
    print(f"  Mean: {low_d_residuals.mean():.4f} mag")
    print(f"  Std: {low_d_residuals.std():.4f} mag")
    print()
    print(f"HIGH-D (x₁ > 0) residuals:")
    print(f"  Mean: {high_d_residuals.mean():.4f} mag")
    print(f"  Std: {high_d_residuals.std():.4f} mag")
    print()
    print(f"Δμ = μ(low-D) - μ(high-D) = {low_d_residuals.mean() - high_d_residuals.mean():.4f} mag")
    print()

    # Convert to ΔH₀
    # Δμ ≈ -5 Δlog₁₀(H₀) = -5/(ln 10) * ΔH₀/H₀
    # So ΔH₀/H₀ ≈ -Δμ * ln(10) / 5
    delta_mu = low_d_residuals.mean() - high_d_residuals.mean()
    delta_h0_fraction = -delta_mu * np.log(10) / 5
    delta_h0 = delta_h0_fraction * H0_fiducial

    print(f"Implied ΔH₀/H₀ = {delta_h0_fraction * 100:.2f}%")
    print(f"Implied ΔH₀ = {delta_h0:.2f} km/s/Mpc")
    print()

    return df_low_z

def create_plots(df):
    """Create diagnostic plots."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Filter to low-z
    df_plot = df[df['redshift'] <= 0.15].copy()

    # Compute H₀
    luminosity_distance = 10**((df_plot['distance_modulus'] - 25) / 5)
    df_plot['H0'] = SPEED_OF_LIGHT * df_plot['redshift'] / luminosity_distance

    # Plot 1: H₀ vs x₁
    ax = axes[0, 0]
    low_d = df_plot[df_plot['x1_stretch'] < 0]
    high_d = df_plot[df_plot['x1_stretch'] > 0]

    ax.scatter(low_d['x1_stretch'], low_d['H0'], alpha=0.5, s=20, label='x₁ < 0 (LOW-D)', color='blue')
    ax.scatter(high_d['x1_stretch'], high_d['H0'], alpha=0.5, s=20, label='x₁ > 0 (HIGH-D)', color='red')
    ax.axhline(low_d['H0'].mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean LOW-D: {low_d["H0"].mean():.2f}')
    ax.axhline(high_d['H0'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean HIGH-D: {high_d["H0"].mean():.2f}')
    ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    ax.set_xlabel('x₁ (stretch parameter)', fontsize=12)
    ax.set_ylabel('H₀ (km/s/Mpc)', fontsize=12)
    ax.set_title('H₀ vs x₁ (z ≤ 0.15)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(50, 100)

    # Plot 2: H₀ vs redshift
    ax = axes[0, 1]
    ax.scatter(low_d['redshift'], low_d['H0'], alpha=0.5, s=20, label='x₁ < 0 (LOW-D)', color='blue')
    ax.scatter(high_d['redshift'], high_d['H0'], alpha=0.5, s=20, label='x₁ > 0 (HIGH-D)', color='red')
    ax.set_xlabel('Redshift z', fontsize=12)
    ax.set_ylabel('H₀ (km/s/Mpc)', fontsize=12)
    ax.set_title('H₀ vs Redshift', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(50, 100)

    # Plot 3: Distance modulus vs redshift (Hubble diagram)
    ax = axes[1, 0]
    ax.scatter(low_d['redshift'], low_d['distance_modulus'], alpha=0.5, s=20, label='x₁ < 0 (LOW-D)', color='blue')
    ax.scatter(high_d['redshift'], high_d['distance_modulus'], alpha=0.5, s=20, label='x₁ > 0 (HIGH-D)', color='red')
    ax.set_xlabel('Redshift z', fontsize=12)
    ax.set_ylabel('Distance Modulus μ', fontsize=12)
    ax.set_title('Hubble Diagram', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 4: x₁ distribution
    ax = axes[1, 1]
    ax.hist(df_plot['x1_stretch'], bins=50, alpha=0.7, color='gray', edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='x₁ = 0 (D = 2.18)')
    ax.set_xlabel('x₁ (stretch parameter)', fontsize=12)
    ax.set_ylabel('Number of SNe', fontsize=12)
    ax.set_title('x₁ Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output/plots/pantheon_bifurcation_plots.png', dpi=300, bbox_inches='tight')
    print("Plots saved to: output/plots/pantheon_bifurcation_plots.png")
    print()

def main():
    """Main analysis function."""

    # Load data
    print("Loading Pantheon+ data...")
    try:
        df = load_pantheon_data('data/raw/pantheon_plus_data.dat')
    except FileNotFoundError:
        df = load_pantheon_data('/Users/eirikr/cosmos/pantheon_plus_data.dat')
    print(f"Loaded {len(df)} supernovae")
    print()

    # Binned analysis
    print("=" * 80)
    print("BINNED H₀ ANALYSIS")
    print("=" * 80)
    print()

    z_bins = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15]
    df_binned = compute_hubble_parameter_binned(df, z_bins)

    for z_min in sorted(df_binned['z_min'].unique()):
        bin_data = df_binned[df_binned['z_min'] == z_min]
        z_max = bin_data['z_max'].iloc[0]

        print(f"Redshift bin: {z_min:.2f} ≤ z < {z_max:.2f}")

        low_d_row = bin_data[bin_data['population'] == 'LOW-D']
        high_d_row = bin_data[bin_data['population'] == 'HIGH-D']

        if len(low_d_row) > 0:
            row = low_d_row.iloc[0]
            print(f"  LOW-D (x₁ < 0): N={row['n_sne']:3d}, H₀={row['mean_h0']:6.2f} ± {row['std_h0']:5.2f} km/s/Mpc")

        if len(high_d_row) > 0:
            row = high_d_row.iloc[0]
            print(f"  HIGH-D (x₁ > 0): N={row['n_sne']:3d}, H₀={row['mean_h0']:6.2f} ± {row['std_h0']:5.2f} km/s/Mpc")

        if len(low_d_row) > 0 and len(high_d_row) > 0:
            delta_h0 = low_d_row.iloc[0]['mean_h0'] - high_d_row.iloc[0]['mean_h0']
            print(f"  ΔH₀ = {delta_h0:+6.2f} km/s/Mpc")

        print()

    # Residual analysis
    df_low_z = analyze_residuals(df)

    # Create plots
    print("Creating diagnostic plots...")
    create_plots(df)

    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == '__main__':
    main()
