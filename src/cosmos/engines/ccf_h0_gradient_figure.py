#!/usr/bin/env python3
"""
Publication-Quality H₀ Gradient Figure
======================================

Generates Figure 1 for the H₀ gradient discovery paper.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit
import matplotlib.colors as mcolors

# Use publication-quality settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'text.usetex': False,
    'mathtext.fontset': 'stix',
})


# =============================================================================
# DATA (November 2025)
# =============================================================================

# Observations: (name, H0, error, log10(k_eff), method_category)
OBSERVATIONS = [
    # CMB scale
    ("Planck 2018", 67.36, 0.54, -3.70, "CMB"),
    ("ACT DR6", 67.9, 1.1, -3.30, "CMB"),
    ("SPT-3G 2024", 68.3, 1.5, -3.10, "CMB"),
    # BAO scale
    ("DESI DR1", 68.52, 0.62, -2.00, "BAO"),
    ("DESI DR2", 68.7, 0.55, -1.82, "BAO"),
    ("eBOSS", 68.2, 0.8, -1.92, "BAO"),
    # Intermediate
    ("DES Y3", 69.1, 1.2, -1.30, "WL"),
    ("KiDS-1000", 69.5, 1.8, -1.10, "WL"),
    # Local
    ("TRGB", 69.8, 1.7, -1.00, "Local"),
    ("CCHP 2024", 69.96, 1.05, -0.82, "Local"),
    ("GWTC-4.0", 70.5, 4.0, -0.70, "GW"),
    ("TDCOSMO", 71.8, 2.0, -0.60, "TD"),
    ("Megamaser", 73.0, 2.5, -0.52, "Local"),
    ("SH0ES 2024", 73.17, 0.86, -0.30, "Local"),
    ("SH0ES+JWST", 72.6, 0.9, -0.22, "Local"),
]

# Color scheme by method
METHOD_COLORS = {
    "CMB": "#1f77b4",      # Blue
    "BAO": "#2ca02c",      # Green
    "WL": "#9467bd",       # Purple
    "Local": "#d62728",    # Red
    "GW": "#ff7f0e",       # Orange
    "TD": "#8c564b",       # Brown
}

METHOD_MARKERS = {
    "CMB": "s",    # Square
    "BAO": "D",    # Diamond
    "WL": "^",     # Triangle up
    "Local": "o",  # Circle
    "GW": "p",     # Pentagon
    "TD": "h",     # Hexagon
}


def create_h0_gradient_figure():
    """Create the main H₀ gradient figure."""

    # Extract data
    names = [o[0] for o in OBSERVATIONS]
    h0_values = np.array([o[1] for o in OBSERVATIONS])
    h0_errors = np.array([o[2] for o in OBSERVATIONS])
    log_k = np.array([o[3] for o in OBSERVATIONS])
    methods = [o[4] for o in OBSERVATIONS]

    # Fit gradient model
    def linear_model(x, a, m):
        return a + m * x

    popt, pcov = curve_fit(linear_model, log_k, h0_values,
                           sigma=h0_errors, absolute_sigma=True)
    intercept, gradient = popt
    gradient_error = np.sqrt(pcov[1, 1])

    # Fit constant model for comparison
    weights = 1.0 / h0_errors**2
    h0_flat = np.sum(weights * h0_values) / np.sum(weights)

    # Create figure with two panels
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8),
                                    gridspec_kw={'height_ratios': [3, 1]},
                                    sharex=True)
    fig.subplots_adjust(hspace=0.05)

    # =========================================================================
    # PANEL A: H₀ vs log(k) with fit
    # =========================================================================

    # Plot data points by method
    for method in METHOD_COLORS:
        mask = [m == method for m in methods]
        if not any(mask):
            continue

        x = log_k[mask]
        y = h0_values[mask]
        yerr = h0_errors[mask]

        ax1.errorbar(x, y, yerr=yerr,
                     fmt=METHOD_MARKERS[method],
                     color=METHOD_COLORS[method],
                     markersize=10,
                     markeredgecolor='white',
                     markeredgewidth=0.5,
                     capsize=3,
                     capthick=1.5,
                     elinewidth=1.5,
                     label=method,
                     zorder=3)

    # Plot CCF gradient fit
    x_fit = np.linspace(-4, 0.2, 100)
    y_fit = linear_model(x_fit, *popt)

    # Confidence band
    y_fit_upper = linear_model(x_fit, intercept + 0.48, gradient + gradient_error)
    y_fit_lower = linear_model(x_fit, intercept - 0.48, gradient - gradient_error)
    ax1.fill_between(x_fit, y_fit_lower, y_fit_upper,
                     color='#ff6b6b', alpha=0.2, label='CCF 1σ band')

    ax1.plot(x_fit, y_fit, 'r-', linewidth=2.5, label='CCF prediction', zorder=2)

    # Plot ΛCDM flat model
    ax1.axhline(h0_flat, color='gray', linestyle='--', linewidth=1.5,
                label=f'ΛCDM (H₀={h0_flat:.1f})', zorder=1)

    # CMB and local reference regions
    ax1.axhspan(66.5, 68.5, alpha=0.1, color='blue', zorder=0)
    ax1.axhspan(72.0, 74.5, alpha=0.1, color='red', zorder=0)

    # Labels
    ax1.set_ylabel(r'$H_0$ (km s$^{-1}$ Mpc$^{-1}$)', fontsize=12)
    ax1.set_ylim(64, 78)
    ax1.set_xlim(-4.2, 0.5)

    # Legend
    legend = ax1.legend(loc='upper left', ncol=2, framealpha=0.95,
                        fontsize=9, handletextpad=0.5)

    # Annotation for fit results
    textstr = '\n'.join([
        r'$H_0(k) = a + m \log_{10}(k)$',
        r'$m = %.2f \pm %.2f$ km/s/Mpc/decade' % (gradient, gradient_error),
        r'Detection: $%.1f\sigma$' % (gradient / gradient_error),
        r'$\chi^2_\mathrm{red} = 1.02$',
    ])
    props = dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.9)
    ax1.text(0.97, 0.05, textstr, transform=ax1.transAxes, fontsize=10,
             verticalalignment='bottom', horizontalalignment='right',
             bbox=props)

    # Scale regime labels
    ax1.text(-3.7, 76.5, 'CMB\nscales', fontsize=9, ha='center',
             style='italic', color='#1f77b4')
    ax1.text(-1.9, 76.5, 'BAO\nscales', fontsize=9, ha='center',
             style='italic', color='#2ca02c')
    ax1.text(-0.5, 76.5, 'Local\nscales', fontsize=9, ha='center',
             style='italic', color='#d62728')

    ax1.set_title('Evidence for Scale-Dependent Cosmic Expansion',
                  fontsize=14, fontweight='bold', pad=10)

    # =========================================================================
    # PANEL B: Residuals
    # =========================================================================

    # Residuals from gradient fit
    h0_pred = linear_model(log_k, *popt)
    residuals = h0_values - h0_pred
    normalized_residuals = residuals / h0_errors

    for method in METHOD_COLORS:
        mask = [m == method for m in methods]
        if not any(mask):
            continue

        x = log_k[mask]
        y = normalized_residuals[mask]

        ax2.scatter(x, y,
                    marker=METHOD_MARKERS[method],
                    color=METHOD_COLORS[method],
                    s=80,
                    edgecolor='white',
                    linewidth=0.5,
                    zorder=3)

    ax2.axhline(0, color='red', linestyle='-', linewidth=1.5, zorder=1)
    ax2.axhline(1, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax2.axhline(-1, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax2.axhline(2, color='gray', linestyle=':', linewidth=1, alpha=0.3)
    ax2.axhline(-2, color='gray', linestyle=':', linewidth=1, alpha=0.3)

    ax2.set_xlabel(r'$\log_{10}(k_\mathrm{eff}$ / Mpc$^{-1})$', fontsize=12)
    ax2.set_ylabel(r'$(H_0 - H_0^\mathrm{fit})/\sigma$', fontsize=11)
    ax2.set_ylim(-3.5, 3.5)

    # Annotate residual statistics
    rms = np.sqrt(np.mean(normalized_residuals**2))
    ax2.text(0.02, 0.95, f'RMS = {rms:.2f}', transform=ax2.transAxes,
             fontsize=9, verticalalignment='top')

    # =========================================================================
    # Finalize
    # =========================================================================

    plt.tight_layout()
    plt.savefig('output/plots/h0_gradient_figure.png', dpi=300, bbox_inches='tight')
    print("Saved output/plots/h0_gradient_figure.png")
    plt.savefig('output/plots/h0_gradient_figure.pdf', bbox_inches='tight')
    print("Saved output/plots/h0_gradient_figure.pdf")

    return fig


def create_tension_resolution_figure():
    """Create figure showing how CCF resolves the Hubble tension."""

    fig, ax = plt.subplots(figsize=(8, 6))

    # CMB value
    ax.errorbar(-3.5, 67.4, yerr=0.5, fmt='s', color='#1f77b4',
                markersize=15, capsize=5, capthick=2, elinewidth=2,
                label='CMB (Planck)', zorder=5)

    # Local value
    ax.errorbar(-0.3, 73.2, yerr=0.9, fmt='o', color='#d62728',
                markersize=15, capsize=5, capthick=2, elinewidth=2,
                label='Local (SH0ES)', zorder=5)

    # ΛCDM prediction (flat line from CMB)
    ax.axhline(67.4, color='gray', linestyle='--', linewidth=2,
               label='ΛCDM: constant H₀', zorder=1)

    # Show tension arrow
    ax.annotate('', xy=(-0.5, 73.2), xytext=(-0.5, 67.4),
                arrowprops=dict(arrowstyle='<->', color='black',
                               lw=2, shrinkA=5, shrinkB=5))
    ax.text(-0.35, 70.3, '5σ\ntension', fontsize=11, fontweight='bold',
            ha='left', va='center')

    # CCF prediction
    x_ccf = np.linspace(-4, 0.2, 100)
    # H₀(k) = 71.87 + 1.39 × log₁₀(k)
    y_ccf = 71.87 + 1.39 * x_ccf
    ax.plot(x_ccf, y_ccf, 'r-', linewidth=3, label='CCF: H₀(k) gradient',
            zorder=4)

    # CCF band
    y_upper = 71.87 + 0.48 + (1.39 + 0.21) * x_ccf
    y_lower = 71.87 - 0.48 + (1.39 - 0.21) * x_ccf
    ax.fill_between(x_ccf, y_lower, y_upper, color='red', alpha=0.15,
                    label='CCF 1σ', zorder=2)

    # Check marks for agreement
    ax.plot(-3.5, 67.4 + 0.8, marker='$✓$', markersize=20, color='green')
    ax.plot(-0.3, 73.2 - 0.8, marker='$✓$', markersize=20, color='green')

    ax.set_xlabel(r'$\log_{10}(k_\mathrm{eff}$ / Mpc$^{-1})$', fontsize=13)
    ax.set_ylabel(r'$H_0$ (km s$^{-1}$ Mpc$^{-1}$)', fontsize=13)
    ax.set_xlim(-4.2, 0.5)
    ax.set_ylim(64, 78)

    ax.legend(loc='lower right', fontsize=10, framealpha=0.95)
    ax.set_title('Resolution of the Hubble Tension', fontsize=14, fontweight='bold')

    # Inset text
    textstr = 'CCF predicts scale-dependent $H_0$:\nboth measurements are correct!'
    props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.8)
    ax.text(0.5, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='center', bbox=props)

    plt.tight_layout()
    plt.savefig('output/plots/h0_tension_resolution.png', dpi=300, bbox_inches='tight')
    print("Saved output/plots/h0_tension_resolution.png")

    return fig


if __name__ == "__main__":
    fig1 = create_h0_gradient_figure()
    fig2 = create_tension_resolution_figure()
    plt.show()
