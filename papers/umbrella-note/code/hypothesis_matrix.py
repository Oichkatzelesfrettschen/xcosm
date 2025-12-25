#!/usr/bin/env python3
"""
hypothesis_matrix.py

Builds dependency matrix showing which hypotheses are independent vs derived.
Distinguishes calibrated parameters from testable predictions.
Generates visualization of hypothesis relationships for COSMOS program overview.

Usage:
    python hypothesis_matrix.py --output figures/hypothesis_dependency.pdf
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


class HypothesisNode:
    """Represents a hypothesis or parameter in the COSMOS program."""

    def __init__(
        self,
        hypothesis_id: str,
        name: str,
        node_type: str,
        testable: bool,
        status: str,
        description: str = "",
        paper_reference: Optional[str] = None,
    ):
        """
        Initialize hypothesis node.

        Parameters
        ----------
        hypothesis_id : str
            Unique identifier (e.g., 'H1', 'P1', 'I1')
        name : str
            Short name of hypothesis
        node_type : str
            One of: 'independent', 'calibrated', 'prediction', 'derived'
        testable : bool
            Whether hypothesis can be tested with data
        status : str
            Current status (e.g., 'pending', 'falsified', 'marginal', 'confirmed')
        description : str, optional
            Longer description
        paper_reference : str, optional
            Which paper tests this hypothesis
        """
        self.hypothesis_id = hypothesis_id
        self.name = name
        self.node_type = node_type
        self.testable = testable
        self.status = status
        self.description = description
        self.paper_reference = paper_reference


class COSMOSDependencyGraph:
    """Manages dependency graph for COSMOS hypotheses."""

    def __init__(self):
        """Initialize empty dependency graph."""
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, HypothesisNode] = {}
        self._build_cosmos_hypotheses()
        self._build_dependencies()

    def _build_cosmos_hypotheses(self):
        """Define all COSMOS hypotheses and parameters."""

        # Independent inputs (no dependencies)
        independent_inputs = [
            HypothesisNode(
                "I1",
                "BH entropy formula",
                "independent",
                False,
                "established",
                "S = (A/4) in Planck units (Bekenstein-Hawking)",
                None,
            ),
            HypothesisNode(
                "I2",
                "Percolation threshold",
                "independent",
                False,
                "established",
                "p_c = 1/4 for specific lattice types",
                None,
            ),
            HypothesisNode(
                "I3",
                "Octonion structure",
                "independent",
                False,
                "established",
                "Division algebra structure constants",
                None,
            ),
            HypothesisNode(
                "I4",
                "Jordan algebra J3(O)",
                "independent",
                False,
                "established",
                "3x3 Hermitian octonion matrices",
                None,
            ),
        ]

        # Calibrated parameters (fitted to data, NOT predictions)
        calibrated_params = [
            HypothesisNode(
                "C1",
                "w₀ = -5/6",
                "calibrated",
                False,
                "fitted",
                "Dark energy EOS parameter calibrated to DESI DR2",
                None,
            ),
            HypothesisNode(
                "C2",
                "n_s = 0.966",
                "calibrated",
                False,
                "fitted",
                "Scalar spectral index calibrated to Planck",
                None,
            ),
            HypothesisNode(
                "C3",
                "S₈ = 0.78",
                "calibrated",
                False,
                "fitted",
                "Matter clustering amplitude calibrated to weak lensing",
                None,
            ),
            HypothesisNode(
                "C4",
                "H₀ gradient slope",
                "calibrated",
                False,
                "fitted",
                "Scale-dependent expansion rate calibrated to multi-method data",
                "Paper 2",
            ),
        ]

        # Derived theoretical connections
        derived_hypotheses = [
            HypothesisNode(
                "D1",
                "ε = 1/4 motif",
                "derived",
                False,
                "organizing_principle",
                "Organizing motif, NOT independent confirmation",
                None,
            ),
            HypothesisNode(
                "D2",
                "F₄ → J3(O) connection",
                "derived",
                False,
                "theoretical",
                "Exceptional algebra relationship",
                None,
            ),
            HypothesisNode(
                "D3",
                "Bigraph rewriting rules",
                "derived",
                True,
                "under_test",
                "Discrete pregeometry dynamics",
                "Paper 3",
            ),
        ]

        # Testable predictions
        predictions = [
            HypothesisNode(
                "P1",
                "Spandrel host bias",
                "prediction",
                True,
                "pending",
                "SN Ia residuals correlated with host metallicity/mass/SFR",
                "Paper 1",
            ),
            HypothesisNode(
                "P2",
                "r = 0.0048",
                "prediction",
                True,
                "pending",
                "Tensor-to-scalar ratio from CCF",
                "Paper 3",
            ),
            HypothesisNode(
                "P3",
                "δ_CP = 67.8°",
                "prediction",
                True,
                "marginal",
                "CP phase from octonion geometry (now 1.9σ from LHCb 62.8±2.6°)",
                None,
            ),
            HypothesisNode(
                "P4",
                "OR curvature convergence",
                "prediction",
                True,
                "falsified",
                "Ollivier-Ricci curvature should converge to Ricci curvature",
                "Paper 3",
            ),
            HypothesisNode(
                "P5",
                "H₀ scale-dependence",
                "prediction",
                True,
                "pending",
                "Local H₀ varies with smoothing scale (requires H₀(R) definition)",
                "Paper 2",
            ),
        ]

        # Add all nodes to graph
        for node_list in [independent_inputs, calibrated_params, derived_hypotheses, predictions]:
            for node in node_list:
                self.nodes[node.hypothesis_id] = node
                self.graph.add_node(
                    node.hypothesis_id,
                    label=node.name,
                    node_type=node.node_type,
                    testable=node.testable,
                    status=node.status,
                )

    def _build_dependencies(self):
        """Define dependency edges between hypotheses."""

        # Calibrated parameters depend on data (implicit, not shown)
        # Derived hypotheses depend on independent inputs
        dependencies = [
            # ε = 1/4 motif depends on both independent inputs and calibrations
            ("I1", "D1"),  # BH entropy → ε motif
            ("I2", "D1"),  # Percolation → ε motif
            ("C1", "D1"),  # w₀ calibration → ε motif
            ("C2", "D1"),  # n_s calibration → ε motif
            ("C3", "D1"),  # S₈ calibration → ε motif
            # Algebra connections
            ("I3", "D2"),  # Octonions → F₄ connection
            ("I4", "D2"),  # Jordan algebra → F₄ connection
            # Bigraph dynamics
            ("D2", "D3"),  # Algebra → bigraph rules
            # Predictions
            ("D1", "P1"),  # ε motif → Spandrel (weak connection)
            ("D3", "P2"),  # Bigraph → r prediction
            ("D3", "P4"),  # Bigraph → OR convergence test
            ("I3", "P3"),  # Octonions → δ_CP prediction
            ("D2", "P3"),  # F₄ → δ_CP prediction
            ("C4", "P5"),  # H₀ gradient calibration → scale-dependence test
        ]

        for source_id, target_id in dependencies:
            self.graph.add_edge(source_id, target_id)

    def generate_dependency_matrix(self) -> np.ndarray:
        """
        Generate adjacency matrix for dependency graph.

        Returns
        -------
        matrix : np.ndarray
            Dependency matrix (rows: sources, cols: targets)
        node_ids : List[str]
            Ordered list of node IDs
        """
        node_ids = sorted(self.nodes.keys())
        num_nodes = len(node_ids)
        matrix = np.zeros((num_nodes, num_nodes), dtype=int)

        for i, source_id in enumerate(node_ids):
            for j, target_id in enumerate(node_ids):
                if self.graph.has_edge(source_id, target_id):
                    matrix[i, j] = 1

        return matrix, node_ids

    def identify_independent_vs_derived(self) -> Dict[str, List[str]]:
        """
        Classify hypotheses by independence.

        Returns
        -------
        classification : Dict[str, List[str]]
            Keys: 'independent', 'calibrated', 'derived', 'predictions'
        """
        classification = {"independent": [], "calibrated": [], "derived": [], "predictions": []}

        for hypothesis_id, node in self.nodes.items():
            if node.node_type == "independent":
                classification["independent"].append(hypothesis_id)
            elif node.node_type == "calibrated":
                classification["calibrated"].append(hypothesis_id)
            elif node.node_type == "derived":
                classification["derived"].append(hypothesis_id)
            elif node.node_type == "prediction":
                classification["predictions"].append(hypothesis_id)

        return classification

    def count_independent_confirmations(self, hypothesis_id: str) -> int:
        """
        Count genuinely independent inputs to a hypothesis.

        Parameters
        ----------
        hypothesis_id : str
            Target hypothesis to analyze

        Returns
        -------
        count : int
            Number of independent (non-calibrated) ancestors
        """
        # Get all ancestors
        if hypothesis_id not in self.graph:
            return 0

        ancestors = nx.ancestors(self.graph, hypothesis_id)

        # Count only independent inputs (not calibrated)
        independent_count = sum(
            1 for ancestor_id in ancestors if self.nodes[ancestor_id].node_type == "independent"
        )

        return independent_count

    def visualize_dependency_graph(
        self, output_path: Optional[Path] = None, figsize: Tuple[int, int] = (14, 10)
    ):
        """
        Generate visualization of hypothesis dependency graph.

        Parameters
        ----------
        output_path : Path, optional
            Where to save figure
        figsize : Tuple[int, int]
            Figure dimensions
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Define node colors by type
        color_map = {
            "independent": "#2E7D32",  # Dark green
            "calibrated": "#D32F2F",  # Dark red
            "derived": "#1976D2",  # Dark blue
            "prediction": "#F57C00",  # Dark orange
        }

        # Define node shapes by status
        status_markers = {
            "established": "o",
            "fitted": "s",
            "theoretical": "d",
            "organizing_principle": "v",
            "under_test": "^",
            "pending": "p",
            "marginal": "h",
            "falsified": "X",
        }

        # Use hierarchical layout
        pos = self._hierarchical_layout()

        # Draw nodes by type
        for node_type, color in color_map.items():
            node_ids = [
                node_id for node_id, node in self.nodes.items() if node.node_type == node_type
            ]

            nx.draw_networkx_nodes(
                self.graph,
                pos,
                nodelist=node_ids,
                node_color=color,
                node_size=1200,
                alpha=0.9,
                ax=ax,
            )

        # Draw edges
        nx.draw_networkx_edges(
            self.graph,
            pos,
            edge_color="gray",
            arrows=True,
            arrowsize=20,
            width=2,
            alpha=0.6,
            ax=ax,
            connectionstyle="arc3,rad=0.1",
        )

        # Draw labels
        labels = {node_id: node.name for node_id, node in self.nodes.items()}
        nx.draw_networkx_labels(
            self.graph, pos, labels=labels, font_size=8, font_weight="bold", ax=ax
        )

        # Add legend
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor=color_map["independent"], label="Independent Input"),
            Patch(facecolor=color_map["calibrated"], label="Calibrated Parameter"),
            Patch(facecolor=color_map["derived"], label="Derived Hypothesis"),
            Patch(facecolor=color_map["prediction"], label="Testable Prediction"),
        ]
        ax.legend(handles=legend_elements, loc="upper left", fontsize=10)

        ax.set_title(
            "COSMOS Hypothesis Dependency Graph\n" "Red nodes are calibrated (NOT predictions)",
            fontsize=14,
            fontweight="bold",
        )
        ax.axis("off")

        plt.tight_layout()

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Saved dependency graph to {output_path}")

        return fig, ax

    def _hierarchical_layout(self) -> Dict[str, Tuple[float, float]]:
        """
        Create hierarchical layout for dependency graph.

        Returns
        -------
        pos : Dict[str, Tuple[float, float]]
            Node positions
        """
        # Manually define layers
        layers = {
            0: ["I1", "I2", "I3", "I4"],  # Independent inputs
            1: ["C1", "C2", "C3", "C4"],  # Calibrated parameters
            2: ["D1", "D2", "D3"],  # Derived hypotheses
            3: ["P1", "P2", "P3", "P4", "P5"],  # Predictions
        }

        pos = {}
        for layer_idx, node_ids in layers.items():
            num_nodes_in_layer = len(node_ids)
            for i, node_id in enumerate(sorted(node_ids)):
                # Spread nodes horizontally within layer
                x_position = (i - num_nodes_in_layer / 2) * 2
                y_position = -layer_idx * 3
                pos[node_id] = (x_position, y_position)

        return pos

    def generate_latex_table(self, output_path: Optional[Path] = None) -> str:
        """
        Generate LaTeX table of hypothesis dependencies.

        Parameters
        ----------
        output_path : Path, optional
            Where to save LaTeX table

        Returns
        -------
        latex_table : str
            LaTeX table code
        """
        lines = []
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"\centering")
        lines.append(r"\caption{COSMOS Hypothesis Dependency Classification}")
        lines.append(r"\label{tab:hypothesis_dependencies}")
        lines.append(r"\begin{tabular}{llccp{5cm}}")
        lines.append(r"\hline")
        lines.append(r"ID & Name & Type & Testable & Independent Inputs \\")
        lines.append(r"\hline")

        # Group by type
        classification = self.identify_independent_vs_derived()

        for type_name in ["independent", "calibrated", "derived", "predictions"]:
            node_ids = classification[type_name]

            for node_id in sorted(node_ids):
                node = self.nodes[node_id]

                # Count independent confirmations
                num_independent = self.count_independent_confirmations(node_id)

                # Format row
                testable_str = "Yes" if node.testable else "No"
                type_display = node.node_type.replace("_", " ").title()

                lines.append(
                    f"{node_id} & {node.name} & {type_display} & "
                    f"{testable_str} & {num_independent} \\\\"
                )

            if type_name != "predictions":
                lines.append(r"\hline")

        lines.append(r"\hline")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")

        latex_table = "\n".join(lines)

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(latex_table)
            print(f"Saved LaTeX table to {output_path}")

        return latex_table

    def print_summary(self):
        """Print summary of dependency analysis."""
        print("\n" + "=" * 70)
        print("COSMOS HYPOTHESIS DEPENDENCY SUMMARY")
        print("=" * 70)

        classification = self.identify_independent_vs_derived()

        print("\nINDEPENDENT INPUTS (genuinely independent):")
        for node_id in sorted(classification["independent"]):
            node = self.nodes[node_id]
            print(f"  {node_id}: {node.name} - {node.description}")

        print("\nCALIBRATED PARAMETERS (fitted to data, NOT predictions):")
        for node_id in sorted(classification["calibrated"]):
            node = self.nodes[node_id]
            print(f"  {node_id}: {node.name} - {node.description}")

        print("\nDERIVED HYPOTHESES:")
        for node_id in sorted(classification["derived"]):
            node = self.nodes[node_id]
            num_indep = self.count_independent_confirmations(node_id)
            print(f"  {node_id}: {node.name} ({num_indep} independent inputs)")
            print(f"      {node.description}")

        print("\nTESTABLE PREDICTIONS:")
        for node_id in sorted(classification["predictions"]):
            node = self.nodes[node_id]
            num_indep = self.count_independent_confirmations(node_id)
            print(f"  {node_id}: {node.name} (Status: {node.status})")
            print(f"      Independent inputs: {num_indep}")
            print(f"      {node.description}")
            if node.paper_reference:
                print(f"      Tested in: {node.paper_reference}")

        # Critical analysis
        print("\n" + "=" * 70)
        print("CRITICAL ANALYSIS")
        print("=" * 70)

        eps_motif_node = self.nodes["D1"]
        num_independent_eps = self.count_independent_confirmations("D1")

        print("\nε = 1/4 'convergence' analysis:")
        print("  Total appearances: 6 (I1, I2, C1, C2, C3, + Jordan algebra)")
        print(f"  Genuinely independent: {num_independent_eps} (I1: BH entropy, I2: Percolation)")
        print("  Calibrated to data: 3 (C1: w₀, C2: n_s, C3: S₈)")
        print("  Theoretical: 1 (Jordan algebra structure)")
        print("\n  CONCLUSION: ε = 1/4 is an ORGANIZING MOTIF, not convergent evidence.")

        print("\n" + "=" * 70)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Generate COSMOS hypothesis dependency analysis")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("figures"), help="Directory for output files"
    )
    parser.add_argument(
        "--no-plot", action="store_true", help="Skip generating plot (only print summary)"
    )

    args = parser.parse_args()

    # Create dependency graph
    cosmos_graph = COSMOSDependencyGraph()

    # Print summary
    cosmos_graph.print_summary()

    # Generate visualization
    if not args.no_plot:
        output_path = args.output_dir / "hypothesis_dependency.pdf"
        cosmos_graph.visualize_dependency_graph(output_path)

        # Also save as PNG
        output_path_png = args.output_dir / "hypothesis_dependency.png"
        cosmos_graph.visualize_dependency_graph(output_path_png)

    # Generate LaTeX table
    latex_path = args.output_dir / "hypothesis_dependency_table.tex"
    cosmos_graph.generate_latex_table(latex_path)

    # Generate dependency matrix
    matrix, node_ids = cosmos_graph.generate_dependency_matrix()

    print("\nDependency matrix shape:", matrix.shape)
    print("Nodes:", node_ids)

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
