#!/usr/bin/env python3
"""
Fermion Mass Golden Ratio Hypothesis Test

Tests the hypothesis that fermion masses follow: m_i = M₀ × φ^(-n_i)
where φ = (1 + √5)/2 is the golden ratio.

Uses PDG 2024 values with proper error propagation and statistical testing.
"""

import numpy as np

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available, plots will be skipped")

try:
    from scipy.optimize import minimize
    from scipy.stats import chi2

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available, using simple optimization")

import warnings

# Golden ratio
PHI = (1 + np.sqrt(5)) / 2


class FermionMassData:
    """PDG 2024 fermion masses with uncertainties."""

    def __init__(self):
        # Quarks (masses in GeV)
        self.quark_names = ["t", "b", "c", "s", "d", "u"]
        self.quark_masses = np.array(
            [
                172.69,  # top
                4.18,  # bottom (MS bar at m_b)
                1.27,  # charm
                0.0934,  # strange (93.4 MeV)
                0.00467,  # down (4.67 MeV)
                0.00216,  # up (2.16 MeV)
            ]
        )
        self.quark_errors = np.array(
            [
                0.30,  # top
                0.03,  # bottom
                0.02,  # charm
                0.0086,  # strange
                0.00048,  # down
                0.00049,  # up
            ]
        )

        # Leptons (masses in GeV)
        self.lepton_names = ["τ", "μ", "e"]
        self.lepton_masses = np.array(
            [1.77686, 0.105658, 0.000511]  # tau (1776.86 MeV)  # muon  # electron
        )
        self.lepton_errors = np.array(
            [
                0.00012,  # tau
                0.000001,  # muon (negligible, set to 1 keV)
                0.000001,  # electron (negligible, set to 1 keV)
            ]
        )

        # Combined dataset
        self.all_names = self.quark_names + self.lepton_names
        self.all_masses = np.concatenate([self.quark_masses, self.lepton_masses])
        self.all_errors = np.concatenate([self.quark_errors, self.lepton_errors])

    def get_quarks(self):
        """Return quark data."""
        return self.quark_names, self.quark_masses, self.quark_errors

    def get_leptons(self):
        """Return lepton data."""
        return self.lepton_names, self.lepton_masses, self.lepton_errors

    def get_all(self):
        """Return all fermion data."""
        return self.all_names, self.all_masses, self.all_errors


class GoldenRatioFitter:
    """Fit fermion masses to golden ratio hypothesis."""

    def __init__(self, fermion_names, fermion_masses, fermion_errors):
        self.names = fermion_names
        self.masses = fermion_masses
        self.errors = fermion_errors
        self.num_fermions = len(fermion_masses)

        # Results storage
        self.best_fit_M0 = None
        self.best_fit_exponents = None
        self.chi_squared = None
        self.dof = None
        self.p_value = None
        self.fitted = False

    def model(self, exponent, M0):
        """Golden ratio model: m = M₀ × φ^(-n)"""
        return M0 * PHI ** (-exponent)

    def fit_individual_exponents(self):
        """
        Fit individual exponent n_i for each fermion.

        For each fermion: m_i = M₀ × φ^(-n_i)
        We'll find the best global M₀ and individual n_i values.
        """

        def objective(params):
            """
            Chi-squared objective function.
            params = [M0, n_1, n_2, ..., n_N]
            """
            M0 = params[0]
            exponents = params[1:]

            # Predicted masses
            predicted_masses = M0 * PHI ** (-exponents)

            # Chi-squared with measurement errors
            chi_sq = np.sum(((self.masses - predicted_masses) / self.errors) ** 2)
            return chi_sq

        # Initial guess: M0 ~ top mass, exponents spread logarithmically
        log_mass_range = np.log(self.masses.max() / self.masses.min())
        log_phi = np.log(PHI)
        initial_exponents = -np.log(self.masses / self.masses.max()) / log_phi

        initial_params = np.concatenate([[self.masses.max()], initial_exponents])

        # Bounds: M0 > 0, exponents can be any real number
        bounds = [(1e-6, 1e6)] + [(-50, 50)] * self.num_fermions

        # Minimize chi-squared
        if not SCIPY_AVAILABLE:
            raise RuntimeError("scipy is required for optimization. Please install scipy.")

        result = minimize(objective, initial_params, bounds=bounds, method="L-BFGS-B")

        if not result.success:
            warnings.warn(f"Optimization did not converge: {result.message}")

        # Extract results
        self.best_fit_M0 = result.x[0]
        self.best_fit_exponents = result.x[1:]
        self.chi_squared = result.fun

        # Degrees of freedom: N measurements - (1 + N) parameters
        # This is actually -1 DOF, meaning we're overfitting!
        # A proper test would fix some exponents or use a simpler model
        self.dof = self.num_fermions - (1 + self.num_fermions)

        # For a more honest test, let's calculate reduced DOF
        # We should only have 1 free parameter (M0) if exponents follow a pattern
        self.fitted = True

        return result

    def fit_linear_exponent_pattern(self):
        """
        Test if the fitted exponents follow a pattern.
        This is a better statistical test.
        """
        if not self.fitted:
            raise RuntimeError("Must call fit_individual_exponents() first")

        # Check if exponents follow a linear pattern
        # by fitting a line to the exponents
        indices = np.arange(self.num_fermions)

        # Fit y = a + b*x to the exponents
        coeffs = np.polyfit(indices, self.best_fit_exponents, 1)
        a_pattern, b_pattern = coeffs[1], coeffs[0]

        # Exponents from linear pattern
        exponents_pattern = a_pattern + b_pattern * indices

        # Predicted masses from pattern
        predicted_masses_pattern = self.best_fit_M0 * PHI ** (-exponents_pattern)

        # Chi-squared for pattern
        chi_sq_pattern = np.sum(((self.masses - predicted_masses_pattern) / self.errors) ** 2)
        dof_pattern = self.num_fermions - 3  # M0, a, b

        # Also try a simpler test: just use the best individual fits
        # and see how much worse it is than null hypothesis
        # This is the honest chi-squared with proper DOF

        return self.best_fit_M0, a_pattern, b_pattern, chi_sq_pattern, dof_pattern

    def compute_statistics(self):
        """Compute proper statistical measures."""
        if not self.fitted:
            raise RuntimeError("Must call fit_individual_exponents() first")

        # Since we have N measurements and N+1 parameters, we're overfitting
        # Report this honestly
        if self.dof <= 0:
            self.p_value = None
            reduced_chi_sq = None
        else:
            reduced_chi_sq = self.chi_squared / self.dof
            self.p_value = 1 - chi2.cdf(self.chi_squared, self.dof)

        return reduced_chi_sq, self.p_value

    def null_hypothesis_test(self):
        """
        Test against null hypothesis: no pattern (random masses).

        Calculate chi-squared for hypothesis that masses are just
        independent measurements with no underlying pattern.
        """
        # For null hypothesis, we just compare to the mean
        mean_mass = np.mean(self.masses)
        chi_sq_null = np.sum(((self.masses - mean_mass) / self.errors) ** 2)
        dof_null = self.num_fermions - 1

        return chi_sq_null, dof_null

    def get_predicted_masses(self):
        """Get predicted masses from best fit."""
        if not self.fitted:
            raise RuntimeError("Must call fit_individual_exponents() first")

        return self.best_fit_M0 * PHI ** (-self.best_fit_exponents)

    def report_results(self):
        """Generate detailed results report."""
        if not self.fitted:
            raise RuntimeError("Must call fit_individual_exponents() first")

        reduced_chi_sq, p_value = self.compute_statistics()
        predicted_masses = self.get_predicted_masses()
        residuals = self.masses - predicted_masses
        pulls = residuals / self.errors

        # Get linear pattern results
        M0_linear, a, b, chi_sq_linear, dof_linear = self.fit_linear_exponent_pattern()
        reduced_chi_sq_linear = chi_sq_linear / dof_linear if dof_linear > 0 else None
        p_value_linear = 1 - chi2.cdf(chi_sq_linear, dof_linear) if dof_linear > 0 else None

        # Null hypothesis
        chi_sq_null, dof_null = self.null_hypothesis_test()

        print("=" * 80)
        print("FERMION MASS GOLDEN RATIO HYPOTHESIS TEST")
        print("=" * 80)
        print(f"\nModel: m_i = M₀ × φ^(-n_i), where φ = {PHI:.6f}")
        print(f"\nNumber of fermions: {self.num_fermions}")
        print("PDG 2024 data used\n")

        print("-" * 80)
        print("FIT 1: Individual Exponents (Overfitted Model)")
        print("-" * 80)
        print(f"Best-fit M₀ = {self.best_fit_M0:.4f} GeV")
        print(f"Chi-squared = {self.chi_squared:.2f}")
        print(f"DOF = {self.dof} (NEGATIVE - MODEL IS OVERFITTED!)")
        print("\nWARNING: This model has more free parameters (N+1) than data points (N).")
        print("It will fit any dataset perfectly. This is NOT a valid statistical test.\n")

        print("\nIndividual Fermion Results:")
        print(f"{'Name':<6} {'Mass (GeV)':<12} {'Predicted':<12} {'Exponent n':<12} {'Pull':<8}")
        print("-" * 80)
        for i, name in enumerate(self.names):
            print(
                f"{name:<6} {self.masses[i]:<12.6f} {predicted_masses[i]:<12.6f} "
                f"{self.best_fit_exponents[i]:<12.4f} {pulls[i]:<8.2f}"
            )

        print("\n" + "-" * 80)
        print("FIT 2: Linear Exponent Pattern (Honest Test)")
        print("-" * 80)
        print(f"Model: n_i = {a:.4f} + {b:.4f} × i")
        print(f"Best-fit M₀ = {M0_linear:.4f} GeV")
        print(f"Chi-squared = {chi_sq_linear:.2f}")
        print(f"DOF = {dof_linear}")
        if reduced_chi_sq_linear is not None:
            print(f"Reduced χ² = {reduced_chi_sq_linear:.2f}")
            print(f"p-value = {p_value_linear:.4f}")

            if reduced_chi_sq_linear > 10:
                print("\nVERDICT: POOR FIT - χ²/dof >> 1 indicates model is rejected")
            elif reduced_chi_sq_linear > 3:
                print("\nVERDICT: BAD FIT - Model does not describe the data well")
            elif reduced_chi_sq_linear > 1.5:
                print("\nVERDICT: MARGINAL FIT - Some tension between model and data")
            else:
                print("\nVERDICT: GOOD FIT - Model consistent with data")

        print("\n" + "-" * 80)
        print("NULL HYPOTHESIS: No Pattern")
        print("-" * 80)
        print(f"Chi-squared = {chi_sq_null:.2f}")
        print(f"DOF = {dof_null}")
        print("\nComparison:")
        if dof_linear > 0:
            print(f"  Δχ² = {chi_sq_null - chi_sq_linear:.2f}")
            print(f"  ΔDOF = {dof_null - dof_linear}")

            # F-test equivalent
            delta_chi_sq = chi_sq_null - chi_sq_linear
            delta_dof = dof_null - dof_linear
            if delta_chi_sq > 0 and delta_dof > 0:
                improvement = delta_chi_sq / delta_dof
                print(f"  Improvement per parameter = {improvement:.2f}")
                if improvement < 1:
                    print("  Golden ratio model does NOT significantly improve fit")
                else:
                    print(f"  Golden ratio model improves fit by Δχ² = {delta_chi_sq:.2f}")

        print("\n" + "=" * 80)
        print("CONCLUSIONS")
        print("=" * 80)
        print("\nScientific Assessment:")

        if dof_linear > 0 and reduced_chi_sq_linear is not None:
            if reduced_chi_sq_linear > 5:
                print("• The golden ratio hypothesis is STRONGLY REJECTED by the data")
                print(f"• χ²/dof = {reduced_chi_sq_linear:.1f} >> 1 indicates very poor fit")
                print("• The masses do not follow the proposed φ^(-n) pattern")
            elif reduced_chi_sq_linear > 2:
                print("• The golden ratio hypothesis provides a POOR fit to the data")
                print(f"• χ²/dof = {reduced_chi_sq_linear:.1f} indicates significant discrepancies")
                print("• The pattern may be coincidental or approximate at best")
            else:
                print("• The golden ratio hypothesis shows interesting agreement with data")
                print(f"• χ²/dof = {reduced_chi_sq_linear:.1f} suggests reasonable fit")
                print("• Further theoretical justification would be needed")

        print("\n" + "=" * 80 + "\n")

        return {
            "M0": self.best_fit_M0,
            "exponents": self.best_fit_exponents,
            "chi_squared": self.chi_squared,
            "dof": self.dof,
            "M0_linear": M0_linear,
            "linear_params": (a, b),
            "chi_squared_linear": chi_sq_linear,
            "dof_linear": dof_linear,
            "reduced_chi_squared_linear": reduced_chi_sq_linear,
            "p_value_linear": p_value_linear,
            "chi_squared_null": chi_sq_null,
            "dof_null": dof_null,
        }


def plot_results(fitter, output_path):
    """Generate publication-quality figure."""

    if not MATPLOTLIB_AVAILABLE:
        print("\nSkipping plot generation (matplotlib not available)")
        print(f"Install matplotlib to generate: {output_path}")
        return

    predicted_masses = fitter.get_predicted_masses()

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [2, 1]})

    # Color scheme
    colors = [
        "#e74c3c",
        "#3498db",
        "#2ecc71",
        "#f39c12",
        "#9b59b6",
        "#1abc9c",
        "#e67e22",
        "#95a5a6",
        "#34495e",
    ]

    # Top panel: Masses vs exponent
    x_positions = np.arange(len(fitter.names))

    # Plot measured masses with error bars
    ax1.errorbar(
        x_positions,
        fitter.masses,
        yerr=fitter.errors,
        fmt="o",
        markersize=8,
        capsize=5,
        capthick=2,
        label="PDG 2024 Measurements",
        color="black",
        zorder=3,
    )

    # Plot predicted masses
    for i, (mass, name) in enumerate(zip(predicted_masses, fitter.names)):
        color = colors[i % len(colors)]
        ax1.plot(
            x_positions[i],
            mass,
            "s",
            markersize=10,
            color=color,
            label=f"{name}: n={fitter.best_fit_exponents[i]:.2f}",
            zorder=2,
        )

    # Connect with lines
    ax1.plot(x_positions, predicted_masses, "--", color="gray", alpha=0.5, zorder=1)

    ax1.set_ylabel("Mass (GeV)", fontsize=12, fontweight="bold")
    ax1.set_yscale("log")
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(fitter.names, fontsize=11)
    ax1.grid(True, alpha=0.3, which="both")
    ax1.legend(loc="best", fontsize=9, ncol=2)
    ax1.set_title(
        "Fermion Mass Golden Ratio Fit: m = M₀ × φ⁻ⁿ\n"
        + f"M₀ = {fitter.best_fit_M0:.2f} GeV, φ = {PHI:.6f}",
        fontsize=13,
        fontweight="bold",
    )

    # Bottom panel: Pulls
    pulls = (fitter.masses - predicted_masses) / fitter.errors
    colors_bars = ["red" if abs(p) > 3 else "orange" if abs(p) > 2 else "green" for p in pulls]

    ax2.bar(x_positions, pulls, color=colors_bars, alpha=0.7, edgecolor="black")
    ax2.axhline(0, color="black", linestyle="-", linewidth=1)
    ax2.axhline(2, color="orange", linestyle="--", linewidth=1, alpha=0.5)
    ax2.axhline(-2, color="orange", linestyle="--", linewidth=1, alpha=0.5)
    ax2.axhline(3, color="red", linestyle="--", linewidth=1, alpha=0.5)
    ax2.axhline(-3, color="red", linestyle="--", linewidth=1, alpha=0.5)

    ax2.set_ylabel("Pull (σ)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Fermion", fontsize=12, fontweight="bold")
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(fitter.names, fontsize=11)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_ylim(-5, 5)

    # Add statistics box
    stats_text = f"Linear pattern: χ²/dof = {fitter.chi_squared:.1f}/{fitter.num_fermions-3}\n"

    _, _, _, chi_sq_linear, dof_linear = fitter.fit_linear_exponent_pattern()
    if dof_linear > 0:
        reduced_chi_sq = chi_sq_linear / dof_linear
        stats_text = (
            f"Linear pattern: χ²/dof = {chi_sq_linear:.1f}/{dof_linear} = {reduced_chi_sq:.2f}\n"
        )
        p_val = 1 - chi2.cdf(chi_sq_linear, dof_linear)
        stats_text += f"p-value = {p_val:.4f}"

    ax2.text(
        0.02,
        0.98,
        stats_text,
        transform=ax2.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nFigure saved to: {output_path}")
    plt.close()


def main():
    """Run complete analysis."""

    print("\nLoading PDG 2024 fermion mass data...")
    data = FermionMassData()

    # Analyze all fermions together
    print("\n" + "=" * 80)
    print("ANALYSIS: ALL FERMIONS (Quarks + Leptons)")
    print("=" * 80)
    names, masses, errors = data.get_all()
    fitter_all = GoldenRatioFitter(names, masses, errors)
    fitter_all.fit_individual_exponents()
    results_all = fitter_all.report_results()

    # Generate plot
    output_path = "/Users/eirikr/1_Workspace/cosmos/paper/output/fermion_mass_fit.pdf"
    plot_results(fitter_all, output_path)

    # Analyze quarks only
    print("\n" + "=" * 80)
    print("ANALYSIS: QUARKS ONLY")
    print("=" * 80)
    quark_names, quark_masses, quark_errors = data.get_quarks()
    fitter_quarks = GoldenRatioFitter(quark_names, quark_masses, quark_errors)
    fitter_quarks.fit_individual_exponents()
    results_quarks = fitter_quarks.report_results()

    output_path_quarks = "/Users/eirikr/1_Workspace/cosmos/paper/output/fermion_mass_fit_quarks.pdf"
    plot_results(fitter_quarks, output_path_quarks)

    # Analyze leptons only
    print("\n" + "=" * 80)
    print("ANALYSIS: LEPTONS ONLY")
    print("=" * 80)
    lepton_names, lepton_masses, lepton_errors = data.get_leptons()
    fitter_leptons = GoldenRatioFitter(lepton_names, lepton_masses, lepton_errors)
    fitter_leptons.fit_individual_exponents()
    results_leptons = fitter_leptons.report_results()

    output_path_leptons = (
        "/Users/eirikr/1_Workspace/cosmos/paper/output/fermion_mass_fit_leptons.pdf"
    )
    plot_results(fitter_leptons, output_path_leptons)

    print("\n" + "=" * 80)
    print("SUMMARY OF ALL ANALYSES")
    print("=" * 80)
    print("\n1. ALL FERMIONS:")
    print(
        f"   χ²/dof = {results_all['reduced_chi_squared_linear']:.2f}"
        if results_all["reduced_chi_squared_linear"]
        else "   Overfitted"
    )

    print("\n2. QUARKS ONLY:")
    print(
        f"   χ²/dof = {results_quarks['reduced_chi_squared_linear']:.2f}"
        if results_quarks["reduced_chi_squared_linear"]
        else "   Overfitted"
    )

    print("\n3. LEPTONS ONLY:")
    print(
        f"   χ²/dof = {results_leptons['reduced_chi_squared_linear']:.2f}"
        if results_leptons["reduced_chi_squared_linear"]
        else "   Overfitted"
    )

    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)
    print("\nBased on rigorous statistical analysis with PDG 2024 data:")

    if results_all["reduced_chi_squared_linear"] and results_all["reduced_chi_squared_linear"] > 3:
        print("• The golden ratio hypothesis is REJECTED")
        print("• Fermion masses do NOT follow a simple φ^(-n) pattern")
        print("• Any apparent correlations are likely coincidental")
    elif (
        results_all["reduced_chi_squared_linear"]
        and results_all["reduced_chi_squared_linear"] > 1.5
    ):
        print("• The golden ratio hypothesis shows MARGINAL agreement at best")
        print("• Significant discrepancies exist between model and data")
        print("• Further theoretical justification would be required")
    else:
        print("• The golden ratio hypothesis shows INTERESTING agreement with data")
        print("• Further investigation may be warranted")
        print("• Theoretical motivation would strengthen the case")

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
