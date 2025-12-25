#!/usr/bin/env python3
"""
falsification_table.py

Documents all falsified hypotheses, pending tests, and experimental timelines
for COSMOS research program. Generates LaTeX tables for umbrella note paper.

Usage:
    python falsification_table.py --output tables/falsification_program.tex
"""

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class FalsificationTest:
    """Represents a falsification test for a COSMOS hypothesis."""

    hypothesis_id: str
    hypothesis_name: str
    test_description: str
    data_source: str
    timeline: str
    status: str  # 'falsified', 'pending', 'marginal', 'confirmed', 'under_analysis'
    failure_condition: str
    current_result: Optional[str] = None
    sigma_level: Optional[float] = None
    paper_reference: Optional[str] = None
    notes: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON export."""
        return {
            "hypothesis_id": self.hypothesis_id,
            "hypothesis_name": self.hypothesis_name,
            "test_description": self.test_description,
            "data_source": self.data_source,
            "timeline": self.timeline,
            "status": self.status,
            "failure_condition": self.failure_condition,
            "current_result": self.current_result,
            "sigma_level": self.sigma_level,
            "paper_reference": self.paper_reference,
            "notes": self.notes,
        }


class COSMOSFalsificationProgram:
    """Manages all falsification tests in COSMOS program."""

    def __init__(self):
        """Initialize falsification test database."""
        self.tests: List[FalsificationTest] = []
        self._populate_tests()

    def _populate_tests(self):
        """Define all COSMOS falsification tests."""

        # FALSIFIED HYPOTHESES
        self.tests.extend(
            [
                FalsificationTest(
                    hypothesis_id="H_fermion",
                    hypothesis_name="φ⁻ⁿ fermion mass scaling",
                    test_description="Fit fermion masses to m_f = m_0 * φ⁻ⁿ scaling",
                    data_source="PDG 2024 fermion masses",
                    timeline="Completed 2024",
                    status="falsified",
                    failure_condition="χ²/dof > 3.0 indicates poor fit",
                    current_result="χ²/dof = 35,173 / 12 = 2,931",
                    sigma_level=None,
                    paper_reference="None (exploratory analysis)",
                    notes="CATASTROPHICALLY FALSIFIED. No algebraic structure explains fermion masses.",
                ),
                FalsificationTest(
                    hypothesis_id="H_bigraph_convergence",
                    hypothesis_name="Bigraph κ_OR → Ricci curvature",
                    test_description="Ollivier-Ricci curvature should converge to GR Ricci in continuum limit",
                    data_source="CCF bigraph simulations",
                    timeline="2024-2025",
                    status="falsified",
                    failure_condition="κ_OR diverges rather than converges to R_μν",
                    current_result="κ_OR ~ -N^{0.55} (diverges as N increases)",
                    sigma_level=None,
                    paper_reference="Paper 3",
                    notes="Divergence indicates CCF operates outside proven convergence regime. "
                    "van der Hoorn et al. (2021) convergence requires connected graphs; "
                    "CCF uses disconnected bigraphs.",
                ),
                FalsificationTest(
                    hypothesis_id="H_strong_cp_original",
                    hypothesis_name="Strong CP (original formulation)",
                    test_description="θ_QCD should equal π/8 from octonion geometry",
                    data_source="Neutron EDM limits",
                    timeline="Completed 2024",
                    status="falsified",
                    failure_condition="θ_QCD > 10⁻¹⁰ ruled out by neutron EDM",
                    current_result="π/8 ≈ 0.393 >> 10⁻¹⁰",
                    sigma_level=None,
                    paper_reference="None (recognized immediately)",
                    notes="Original hypothesis abandoned. Could be reframed as θ_eff after "
                    "axion solution, but no longer pursued.",
                ),
            ]
        )

        # PENDING FALSIFICATION TESTS
        self.tests.extend(
            [
                FalsificationTest(
                    hypothesis_id="H_spandrel_host_split",
                    hypothesis_name="Spandrel host galaxy bias",
                    test_description="SN Ia Hubble residuals split by host mass/metallicity/SFR",
                    data_source="Pantheon+, DES-SN5YR, Union3",
                    timeline="2025 Q1-Q2",
                    status="pending",
                    failure_condition="No systematic bias pattern vs redshift after host split",
                    current_result="Model development phase",
                    sigma_level=None,
                    paper_reference="Paper 1 (HIGHEST PRIORITY)",
                    notes="Primary test of whether DESI w₀w_a signal is SN systematic. "
                    "Success: bias detected, correlated with z, Δ(w₀,w_a) > 1σ. "
                    "Failure: no bias pattern or Δ(w₀,w_a) < 0.5σ.",
                ),
                FalsificationTest(
                    hypothesis_id="H_h0_scale_dependence",
                    hypothesis_name="H₀ scale-dependence",
                    test_description="Local H₀(R) varies with smoothing scale R",
                    data_source="Multi-method distance ladder (Cepheids, TRGB, JAGB, SBF, Miras)",
                    timeline="2025-2026",
                    status="pending",
                    failure_condition="H₀(R) trend consistent with ΛCDM cosmic variance",
                    current_result="Heuristic k-mapping suggests trend; requires H₀(R) definition",
                    sigma_level=None,
                    paper_reference="Paper 2",
                    notes="Previous '4.7σ' claims SUSPENDED pending physically defined H₀(R). "
                    "Must compare to ΛCDM mock realizations.",
                ),
                FalsificationTest(
                    hypothesis_id="H_cmb_tensor_modes",
                    hypothesis_name="r = 0.0048 tensor-to-scalar ratio",
                    test_description="CMB B-mode polarization detection",
                    data_source="CMB-S4, LiteBIRD",
                    timeline="2029-2032",
                    status="pending",
                    failure_condition="r < 0.003 (2σ below prediction) or r > 0.007 (2σ above)",
                    current_result="Current limit: r < 0.032 (Planck+BICEP/Keck)",
                    sigma_level=None,
                    paper_reference="Paper 3",
                    notes="Prediction from CCF/bigraph model. Long timeline. "
                    "CMB-S4 sensitivity target: σ(r) ~ 0.001.",
                ),
                FalsificationTest(
                    hypothesis_id="H_cp_phase",
                    hypothesis_name="δ_CP = 67.8° from octonion geometry",
                    test_description="CKM CP-violating phase measurement",
                    data_source="LHCb, Belle II (γ measurement)",
                    timeline="2025-2028",
                    status="marginal",
                    failure_condition="> 3σ deviation from 67.8°",
                    current_result="γ = 62.8° ± 2.6° (LHCb 2025)",
                    sigma_level=1.9,
                    paper_reference="None (exploratory)",
                    notes="Current 1.9σ tension. γ relates to δ_CP via γ ≈ 60° + δ_CP. "
                    "Prediction: γ ≈ 67.8°. Marginal agreement, needs higher precision.",
                ),
                FalsificationTest(
                    hypothesis_id="H_or_convergence_connected",
                    hypothesis_name="OR convergence in connected regime",
                    test_description="Test κ_OR → R_μν for connected bigraphs",
                    data_source="CCF simulations with modified rewriting",
                    timeline="2025 Q2",
                    status="under_analysis",
                    failure_condition="Divergence persists even in connected graphs",
                    current_result="Testing in progress",
                    sigma_level=None,
                    paper_reference="Paper 3",
                    notes="Hypothesis: disconnected graphs cause divergence. "
                    "Test: modify CCF to maintain connectivity. "
                    "If still diverges, bigraph approach fundamentally flawed.",
                ),
            ]
        )

        # CONFIRMED HYPOTHESES (for completeness)
        self.tests.extend(
            [
                FalsificationTest(
                    hypothesis_id="H_bh_entropy",
                    hypothesis_name="BH entropy = A/4",
                    test_description="Bekenstein-Hawking entropy formula",
                    data_source="Theoretical (established)",
                    timeline="1973 (Bekenstein, Hawking)",
                    status="confirmed",
                    failure_condition="N/A (foundational physics)",
                    current_result="S = A/(4G) in Planck units",
                    sigma_level=None,
                    paper_reference="Independent input (not COSMOS prediction)",
                    notes="Used as independent input, not a COSMOS prediction.",
                ),
                FalsificationTest(
                    hypothesis_id="H_percolation",
                    hypothesis_name="Percolation threshold p_c = 1/4",
                    test_description="Bond percolation on specific lattices",
                    data_source="Mathematical proof (lattice theory)",
                    timeline="Established (lattice theory)",
                    status="confirmed",
                    failure_condition="N/A (proven for specific lattices)",
                    current_result="p_c = 1/4 for certain lattice types",
                    sigma_level=None,
                    paper_reference="Independent input (not COSMOS prediction)",
                    notes="Valid for specific lattice structures. Not universal.",
                ),
            ]
        )

    def filter_by_status(self, status: str) -> List[FalsificationTest]:
        """
        Get all tests with specific status.

        Parameters
        ----------
        status : str
            Status to filter ('falsified', 'pending', 'marginal', etc.)

        Returns
        -------
        filtered_tests : List[FalsificationTest]
            Tests matching status
        """
        return [test for test in self.tests if test.status == status]

    def generate_latex_falsified_table(self) -> str:
        """
        Generate LaTeX table of falsified hypotheses.

        Returns
        -------
        latex_code : str
            LaTeX table code
        """
        falsified_tests = self.filter_by_status("falsified")

        lines = []
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"\centering")
        lines.append(r"\caption{Falsified COSMOS Hypotheses}")
        lines.append(r"\label{tab:falsified_hypotheses}")
        lines.append(r"\begin{tabular}{lp{4cm}p{3cm}p{4cm}}")
        lines.append(r"\hline")
        lines.append(r"Hypothesis & Test & Result & Conclusion \\")
        lines.append(r"\hline")

        for test in falsified_tests:
            # Escape special LaTeX characters
            name_safe = test.hypothesis_name.replace("_", r"\_")
            result_safe = test.current_result.replace("_", r"\_") if test.current_result else ""

            lines.append(
                f"{name_safe} & {test.test_description} & "
                f"{result_safe} & {test.failure_condition} \\\\"
            )

        lines.append(r"\hline")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")

        return "\n".join(lines)

    def generate_latex_pending_table(self) -> str:
        """
        Generate LaTeX table of pending falsification tests.

        Returns
        -------
        latex_code : str
            LaTeX table code
        """
        pending_tests = [
            test for test in self.tests if test.status in ["pending", "under_analysis", "marginal"]
        ]

        lines = []
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"\centering")
        lines.append(r"\caption{Pending COSMOS Falsification Tests}")
        lines.append(r"\label{tab:pending_tests}")
        lines.append(r"\begin{tabular}{lp{3.5cm}p{2.5cm}p{2cm}p{3cm}}")
        lines.append(r"\hline")
        lines.append(r"Hypothesis & Test & Data Source & Timeline & Failure Condition \\")
        lines.append(r"\hline")

        for test in pending_tests:
            name_safe = test.hypothesis_name.replace("_", r"\_")
            data_safe = test.data_source.replace("_", r"\_")
            failure_safe = test.failure_condition.replace("_", r"\_")

            lines.append(
                f"{name_safe} & {test.test_description} & "
                f"{data_safe} & {test.timeline} & {failure_safe} \\\\"
            )

        lines.append(r"\hline")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")

        return "\n".join(lines)

    def generate_combined_latex_table(self) -> str:
        """
        Generate comprehensive LaTeX table with all tests.

        Returns
        -------
        latex_code : str
            LaTeX longtable code
        """
        lines = []
        lines.append(r"\begin{longtable}{lp{3cm}p{2.5cm}p{1.5cm}p{3cm}p{2cm}}")
        lines.append(r"\caption{COSMOS Falsification Program} \\")
        lines.append(r"\label{tab:falsification_program}")
        lines.append(r"\hline")
        lines.append(r"Hypothesis & Test & Data & Timeline & Failure Condition & Status \\")
        lines.append(r"\hline")
        lines.append(r"\endfirsthead")
        lines.append(r"\hline")
        lines.append(r"Hypothesis & Test & Data & Timeline & Failure Condition & Status \\")
        lines.append(r"\hline")
        lines.append(r"\endhead")

        # Group by status
        status_order = ["falsified", "pending", "under_analysis", "marginal", "confirmed"]
        status_labels = {
            "falsified": r"\textbf{FALSIFIED}",
            "pending": r"\textit{Pending}",
            "under_analysis": r"\textit{Testing}",
            "marginal": r"\textit{Marginal}",
            "confirmed": r"Established",
        }

        for status in status_order:
            tests_in_status = self.filter_by_status(status)

            if tests_in_status:
                lines.append(r"\hline")
                lines.append(f"\\multicolumn{{6}}{{c}}{{{status_labels[status]}}} \\\\")
                lines.append(r"\hline")

                for test in tests_in_status:
                    name_safe = test.hypothesis_name.replace("_", r"\_")
                    test_safe = test.test_description.replace("_", r"\_")
                    data_safe = test.data_source.replace("_", r"\_")
                    failure_safe = test.failure_condition.replace("_", r"\_")

                    # Add result for falsified/marginal tests
                    if test.current_result:
                        result_safe = test.current_result.replace("_", r"\_")
                        status_display = result_safe
                    else:
                        status_display = status_labels[status]

                    lines.append(
                        f"{name_safe} & {test_safe} & {data_safe} & "
                        f"{test.timeline} & {failure_safe} & {status_display} \\\\"
                    )

        lines.append(r"\hline")
        lines.append(r"\end{longtable}")

        return "\n".join(lines)

    def export_to_json(self, output_path: Path):
        """
        Export all tests to JSON for reproducibility.

        Parameters
        ----------
        output_path : Path
            Where to save JSON file
        """
        data = {
            "meta": {
                "generated": datetime.now().isoformat(),
                "program": "COSMOS Falsification Program",
                "total_tests": len(self.tests),
            },
            "tests": [test.to_dict() for test in self.tests],
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Exported {len(self.tests)} tests to {output_path}")

    def print_summary(self):
        """Print summary of falsification program."""
        print("\n" + "=" * 80)
        print("COSMOS FALSIFICATION PROGRAM SUMMARY")
        print("=" * 80)

        status_counts = {}
        for test in self.tests:
            status_counts[test.status] = status_counts.get(test.status, 0) + 1

        print(f"\nTotal hypotheses tracked: {len(self.tests)}")
        print("\nStatus breakdown:")
        for status, count in sorted(status_counts.items()):
            print(f"  {status.upper()}: {count}")

        # Falsified hypotheses
        print("\n" + "=" * 80)
        print("FALSIFIED HYPOTHESES (Research Graveyard)")
        print("=" * 80)

        falsified = self.filter_by_status("falsified")
        for test in falsified:
            print(f"\n{test.hypothesis_name}:")
            print(f"  Test: {test.test_description}")
            print(f"  Result: {test.current_result}")
            print(f"  Why it failed: {test.failure_condition}")
            if test.notes:
                print(f"  Notes: {test.notes}")

        # Pending tests (highest priority)
        print("\n" + "=" * 80)
        print("PENDING FALSIFICATION TESTS (Active Research)")
        print("=" * 80)

        pending = [t for t in self.tests if t.status in ["pending", "under_analysis"]]
        for test in sorted(pending, key=lambda t: t.timeline):
            print(f"\n{test.hypothesis_name} ({test.timeline}):")
            print(f"  Test: {test.test_description}")
            print(f"  Data: {test.data_source}")
            print(f"  Failure condition: {test.failure_condition}")
            if test.current_result:
                print(f"  Current status: {test.current_result}")
            if test.paper_reference:
                print(f"  Paper: {test.paper_reference}")

        # Marginal tests
        print("\n" + "=" * 80)
        print("MARGINAL TESTS (Currently Viable)")
        print("=" * 80)

        marginal = self.filter_by_status("marginal")
        for test in marginal:
            print(f"\n{test.hypothesis_name}:")
            print(f"  Prediction vs Data: {test.current_result}")
            if test.sigma_level:
                print(f"  Tension: {test.sigma_level}σ")
            print(f"  Failure threshold: {test.failure_condition}")

        print("\n" + "=" * 80)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Generate COSMOS falsification program tables")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("tables"), help="Directory for output files"
    )
    parser.add_argument(
        "--format", choices=["latex", "json", "both"], default="both", help="Output format"
    )

    args = parser.parse_args()

    # Create falsification program
    program = COSMOSFalsificationProgram()

    # Print summary
    program.print_summary()

    # Generate outputs
    if args.format in ["latex", "both"]:
        # Falsified hypotheses table
        falsified_table = program.generate_latex_falsified_table()
        falsified_path = args.output_dir / "falsified_hypotheses.tex"
        falsified_path.parent.mkdir(parents=True, exist_ok=True)
        falsified_path.write_text(falsified_table)
        print(f"\nSaved falsified hypotheses table to {falsified_path}")

        # Pending tests table
        pending_table = program.generate_latex_pending_table()
        pending_path = args.output_dir / "pending_tests.tex"
        pending_path.write_text(pending_table)
        print(f"Saved pending tests table to {pending_path}")

        # Combined comprehensive table
        combined_table = program.generate_combined_latex_table()
        combined_path = args.output_dir / "falsification_program.tex"
        combined_path.write_text(combined_table)
        print(f"Saved combined table to {combined_path}")

    if args.format in ["json", "both"]:
        json_path = args.output_dir / "falsification_program.json"
        program.export_to_json(json_path)

    print("\nFalsification program documentation complete.")


if __name__ == "__main__":
    main()
