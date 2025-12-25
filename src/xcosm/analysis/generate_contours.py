#!/usr/bin/env python3
"""
Generate publication-quality parameter space contour plots from MCMC chains.

This script reads the CCF MCMC chain and generates 2D marginalized contours
showing parameter correlations and constraints.

Author: COSMOS Analysis Pipeline
Date: 2025-12-16
"""

from pathlib import Path

import corner
import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Output directory
OUTPUT_DIR = Path("/Users/eirikr/1_Workspace/cosmos/paper/output")
CHAIN_FILE = OUTPUT_DIR / "ccf_chain.h5"


def load_chain(chain_path: Path) -> dict:
    """Load MCMC chain from HDF5 file."""
    with h5py.File(chain_path, "r") as f:
        chain = f["chain"][:]
        param_names = [name.decode() for name in f["parameter_names"][:]]
        summary = f["summary_statistics"][:]
        bic = f.attrs["BIC"]

    return {"chain": chain, "param_names": param_names, "summary": summary, "bic": bic}


def plot_2d_contours(chain: np.ndarray, param_names: list, output_path: str):
    """
    Generate 2D marginalized contour plots.

    Parameters
    ----------
    chain : np.ndarray
        MCMC chain (n_samples, n_params)
    param_names : list
        Parameter names
    output_path : str
        Output file path
    """
    # Parameter labels for plotting
    labels = [r"$\lambda$", r"$\eta$", r"$\alpha$", r"$\varepsilon$", r"$k_*$"]

    # Create corner plot with custom styling
    fig = corner.corner(
        chain,
        labels=labels,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 12},
        label_kwargs={"fontsize": 14},
        title_fmt=".4f",
        color="#1f77b4",
        truth_color="#d62728",
        levels=[0.68, 0.95],  # 1σ and 2σ contours
        fill_contours=True,
        plot_datapoints=True,
        plot_density=True,
        smooth=1.5,
        bins=40,
    )

    # Title
    fig.suptitle("CCF Parameter Space Constraints (MCMC)", fontsize=16, y=1.02)

    # Save
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Contour plot saved to: {output_path}")


def plot_key_correlations(chain: np.ndarray, param_names: list, output_path: str):
    """
    Plot key parameter correlations individually.
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    # Define interesting pairs
    pairs = [
        (0, 4, r"$\lambda$ vs $k_*$"),
        (3, 4, r"$\varepsilon$ vs $k_*$"),
        (1, 2, r"$\eta$ vs $\alpha$"),
        (0, 3, r"$\lambda$ vs $\varepsilon$"),
        (2, 3, r"$\alpha$ vs $\varepsilon$"),
        (1, 4, r"$\eta$ vs $k_*$"),
    ]

    labels = [r"$\lambda$", r"$\eta$", r"$\alpha$", r"$\varepsilon$", r"$k_*$"]

    for ax, (i, j, title) in zip(axes.flat, pairs):
        # 2D histogram
        H, xedges, yedges = np.histogram2d(chain[:, i], chain[:, j], bins=50)

        # Compute levels for 1σ and 2σ contours
        H_sorted = np.sort(H.flatten())[::-1]
        H_cumsum = np.cumsum(H_sorted) / H_sorted.sum()

        level_68 = H_sorted[np.searchsorted(H_cumsum, 0.68)]
        level_95 = H_sorted[np.searchsorted(H_cumsum, 0.95)]

        # Plot
        ax.contourf(
            xedges[:-1],
            yedges[:-1],
            H.T,
            levels=[level_95, level_68, H.max()],
            colors=["#c6dbef", "#6baed6", "#2171b5"],
            alpha=0.8,
        )
        ax.contour(
            xedges[:-1],
            yedges[:-1],
            H.T,
            levels=[level_95, level_68],
            colors=["#08519c", "#08519c"],
            linewidths=[1, 2],
        )

        # Mark MAP estimate
        map_i = np.median(chain[:, i])
        map_j = np.median(chain[:, j])
        ax.plot(map_i, map_j, "r+", markersize=15, mew=2)

        ax.set_xlabel(labels[i], fontsize=12)
        ax.set_ylabel(labels[j], fontsize=12)
        ax.set_title(title, fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Correlation plot saved to: {output_path}")


def plot_1d_posteriors(chain: np.ndarray, param_names: list, output_path: str):
    """
    Plot 1D marginalized posteriors.
    """
    fig, axes = plt.subplots(1, 5, figsize=(16, 3))

    labels = [r"$\lambda$", r"$\eta$", r"$\alpha$", r"$\varepsilon$", r"$k_*$"]

    for i, (ax, label) in enumerate(zip(axes, labels)):
        samples = chain[:, i]

        # Use numpy histogram directly to avoid matplotlib bug
        counts, bin_edges = np.histogram(samples, bins=50, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax.bar(
            bin_centers,
            counts,
            width=bin_edges[1] - bin_edges[0],
            alpha=0.7,
            color="#1f77b4",
            edgecolor="none",
        )

        # KDE
        kde = stats.gaussian_kde(samples)
        x = np.linspace(samples.min(), samples.max(), 200)
        ax.plot(x, kde(x), "r-", linewidth=2)

        # Mark median and 68% CI
        median = np.median(samples)
        lower = np.percentile(samples, 16)
        upper = np.percentile(samples, 84)

        ax.axvline(median, color="black", linestyle="-", linewidth=2)
        ax.axvline(lower, color="gray", linestyle="--", linewidth=1)
        ax.axvline(upper, color="gray", linestyle="--", linewidth=1)

        ax.set_xlabel(label, fontsize=14)
        ax.set_ylabel("Probability density", fontsize=10)
        ax.set_title(f"{median:.4f}$^{{+{upper-median:.4f}}}_{{-{median-lower:.4f}}}$", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"1D posteriors saved to: {output_path}")


def main():
    """Generate all contour plots."""
    print("=" * 70)
    print("GENERATING PARAMETER SPACE CONTOURS")
    print("=" * 70)
    print()

    # Check if chain exists
    if not CHAIN_FILE.exists():
        print(f"ERROR: Chain file not found: {CHAIN_FILE}")
        print("Run ccf_mcmc.py first to generate the chain.")
        return

    # Load chain
    print(f"Loading chain from: {CHAIN_FILE}")
    data = load_chain(CHAIN_FILE)
    print(f"  Chain shape: {data['chain'].shape}")
    print(f"  Parameters: {data['param_names']}")
    print(f"  BIC: {data['bic']:.2f}")
    print()

    # Generate plots
    print("Generating plots...")

    # Full corner plot
    plot_2d_contours(data["chain"], data["param_names"], str(OUTPUT_DIR / "ccf_contours_full.pdf"))

    # Key correlations
    plot_key_correlations(
        data["chain"], data["param_names"], str(OUTPUT_DIR / "ccf_correlations.pdf")
    )

    # 1D posteriors
    plot_1d_posteriors(
        data["chain"], data["param_names"], str(OUTPUT_DIR / "ccf_posteriors_1d.pdf")
    )

    print("\nAll plots generated successfully!")


if __name__ == "__main__":
    main()
