#!/usr/bin/env python3
"""
CCF Bigraph Curvature Divergence Diagnosis
===========================================

This script analyzes why CCF bigraphs show curvature DIVERGENCE rather than
convergence. Current observations: κ_OR ~ -N^0.55 (growing more negative).

Hypotheses to test:
1. Disconnected graphs: CCF graphs remain disconnected at tested scales
2. Wrong radius scaling: Connection radius doesn't scale correctly with N
3. Non-uniform sampling: CCF rewriting creates clustering/voids
4. Hypergraph structure: CCF link graph is hypergraph, not simple graph

This script diagnoses the failure mode and identifies minimal conditions
for convergence.

IMPORTANT: This is NOT about proving CCF works for GR.
This is about UNDERSTANDING why it fails and DOCUMENTING that failure.
"""

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from tqdm import tqdm

# Add parent directory to path to import CCF modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "ccf_package"))

from ccf.bigraph import BigraphEngine, BigraphState, CCFParameters, SimulationConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class DiagnosticMetrics:
    """Diagnostic metrics for a graph."""

    num_nodes: int
    num_edges: int
    num_components: int
    largest_component_size: int
    largest_component_fraction: float
    mean_degree: float
    degree_std: float
    is_connected: bool
    clustering_coefficient: float
    mean_curvature: float
    curvature_std: float

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "num_components": self.num_components,
            "largest_component_size": self.largest_component_size,
            "largest_component_fraction": self.largest_component_fraction,
            "mean_degree": self.mean_degree,
            "degree_std": self.degree_std,
            "is_connected": self.is_connected,
            "clustering_coefficient": self.clustering_coefficient,
            "mean_curvature": self.mean_curvature,
            "curvature_std": self.curvature_std,
        }


class OllivierRicciCurvature:
    """Ollivier-Ricci curvature (same as baseline implementation)."""

    def __init__(self, idleness: float = 0.5):
        self.idleness = idleness

    def neighbor_measure(self, graph: nx.Graph, node: int) -> Dict[int, float]:
        """Construct probability measure on neighborhood."""
        neighbors = list(graph.neighbors(node))

        if len(neighbors) == 0:
            return {node: 1.0}

        measure = {node: self.idleness}
        mass_per_neighbor = (1.0 - self.idleness) / len(neighbors)

        for neighbor in neighbors:
            measure[neighbor] = mass_per_neighbor

        return measure

    def wasserstein_distance(
        self, graph: nx.Graph, measure_x: Dict[int, float], measure_y: Dict[int, float]
    ) -> float:
        """Compute Wasserstein-1 distance."""
        support_x = sorted(measure_x.keys())
        support_y = sorted(measure_y.keys())

        if not support_x or not support_y:
            return 0.0

        num_x = len(support_x)
        num_y = len(support_y)

        cost_matrix = np.zeros((num_x, num_y))

        for i, node_x in enumerate(support_x):
            for j, node_y in enumerate(support_y):
                if node_x == node_y:
                    cost_matrix[i, j] = 0.0
                else:
                    try:
                        cost_matrix[i, j] = nx.shortest_path_length(graph, node_x, node_y)
                    except nx.NetworkXNoPath:
                        cost_matrix[i, j] = 1e10

        probs_x = np.array([measure_x[node] for node in support_x])
        probs_y = np.array([measure_y[node] for node in support_y])

        probs_x = probs_x / np.sum(probs_x)
        probs_y = probs_y / np.sum(probs_y)

        wasserstein_dist = 0.0
        for i in range(num_x):
            for j in range(num_y):
                wasserstein_dist += probs_x[i] * probs_y[j] * cost_matrix[i, j]

        return wasserstein_dist

    def edge_curvature(self, graph: nx.Graph, node_x: int, node_y: int) -> float:
        """Compute OR curvature for edge."""
        if not graph.has_edge(node_x, node_y):
            raise ValueError(f"No edge ({node_x}, {node_y})")

        measure_x = self.neighbor_measure(graph, node_x)
        measure_y = self.neighbor_measure(graph, node_y)

        wasserstein_dist = self.wasserstein_distance(graph, measure_x, measure_y)

        return 1.0 - wasserstein_dist

    def mean_curvature(self, graph: nx.Graph) -> Tuple[float, float]:
        """Compute mean curvature over all edges."""
        if graph.number_of_edges() == 0:
            return 0.0, 0.0

        curvatures = []
        for node_x, node_y in graph.edges():
            try:
                kappa = self.edge_curvature(graph, node_x, node_y)
                curvatures.append(kappa)
            except Exception:
                continue

        if not curvatures:
            return 0.0, 0.0

        return np.mean(curvatures), np.std(curvatures)


# ============================================================================
# CCF Bigraph Generation
# ============================================================================


def generate_ccf_bigraph(num_nodes: int, seed: Optional[int] = None) -> BigraphState:
    """
    Generate CCF bigraph with standard parameters.

    This uses the ACTUAL CCF rewriting rules from the package.

    Parameters
    ----------
    num_nodes : int
        Target number of nodes
    seed : Optional[int]
        Random seed

    Returns
    -------
    BigraphState
        Final bigraph state
    """
    params = CCFParameters()

    config = SimulationConfig(
        inflation_steps=int(np.log2(num_nodes / 10)) if num_nodes > 10 else 1,
        structure_steps=max(1, num_nodes // 20),
        expansion_steps=max(1, num_nodes // 50),
        verbose=False,
        seed=seed,
    )

    engine = BigraphEngine(params=params, config=config)

    initial_state = engine.create_initial_state(num_nodes=10)
    result = engine.run_simulation(initial_state=initial_state)

    return result.final_state


def compute_diagnostic_metrics(
    state: BigraphState, compute_curvature: bool = True
) -> DiagnosticMetrics:
    """
    Compute comprehensive diagnostic metrics for a bigraph.

    Parameters
    ----------
    state : BigraphState
        The bigraph state
    compute_curvature : bool
        Whether to compute OR curvature (slow for large graphs)

    Returns
    -------
    DiagnosticMetrics
        Diagnostic metrics
    """
    graph = state.to_networkx()

    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()

    # Connectivity analysis
    components = list(nx.connected_components(graph))
    num_components = len(components)

    if num_components > 0:
        largest_component = max(components, key=len)
        largest_component_size = len(largest_component)
        largest_component_fraction = largest_component_size / num_nodes
        is_connected = num_components == 1
    else:
        largest_component_size = 0
        largest_component_fraction = 0.0
        is_connected = False

    # Degree distribution
    degrees = [d for n, d in graph.degree()]
    mean_degree = np.mean(degrees) if degrees else 0.0
    degree_std = np.std(degrees) if degrees else 0.0

    # Clustering
    if num_nodes > 0 and num_edges > 0:
        clustering_coefficient = nx.average_clustering(graph)
    else:
        clustering_coefficient = 0.0

    # Curvature (on largest component)
    if compute_curvature and num_components > 0 and largest_component_size > 1:
        largest_subgraph = graph.subgraph(largest_component).copy()
        orc = OllivierRicciCurvature()
        mean_curv, std_curv = orc.mean_curvature(largest_subgraph)
    else:
        mean_curv = np.nan
        std_curv = np.nan

    return DiagnosticMetrics(
        num_nodes=num_nodes,
        num_edges=num_edges,
        num_components=num_components,
        largest_component_size=largest_component_size,
        largest_component_fraction=largest_component_fraction,
        mean_degree=mean_degree,
        degree_std=degree_std,
        is_connected=is_connected,
        clustering_coefficient=clustering_coefficient,
        mean_curvature=mean_curv,
        curvature_std=std_curv,
    )


# ============================================================================
# Divergence Diagnosis Tests
# ============================================================================


def test_ccf_connectivity(node_counts: List[int], num_realizations: int = 10) -> Dict:
    """
    Hypothesis 1: CCF graphs are disconnected.

    Test: Generate CCF bigraphs at multiple scales and check connectivity.

    Expected: If graphs remain disconnected as N increases, this violates
    van der Hoorn convergence assumptions.
    """
    logger.info("=" * 70)
    logger.info("HYPOTHESIS 1: CCF Disconnectivity")
    logger.info("=" * 70)
    logger.info("Testing: Are CCF bigraphs disconnected?")
    logger.info("")

    results = {"node_counts": node_counts, "num_realizations": num_realizations, "metrics": []}

    for num_nodes in node_counts:
        logger.info(f"Testing N = {num_nodes}...")

        ensemble_metrics = []

        for realization_idx in tqdm(range(num_realizations), desc=f"N={num_nodes}", leave=False):
            state = generate_ccf_bigraph(num_nodes, seed=realization_idx)

            # Compute metrics (curvature only for small N)
            compute_curv = num_nodes <= 500
            metrics = compute_diagnostic_metrics(state, compute_curvature=compute_curv)

            ensemble_metrics.append(metrics)

        # Average over ensemble
        avg_components = np.mean([m.num_components for m in ensemble_metrics])
        avg_largest_fraction = np.mean([m.largest_component_fraction for m in ensemble_metrics])
        fraction_connected = np.mean([m.is_connected for m in ensemble_metrics])

        results["metrics"].append(
            {
                "num_nodes": num_nodes,
                "avg_components": avg_components,
                "avg_largest_fraction": avg_largest_fraction,
                "fraction_connected": fraction_connected,
                "avg_mean_degree": np.mean([m.mean_degree for m in ensemble_metrics]),
                "avg_clustering": np.mean([m.clustering_coefficient for m in ensemble_metrics]),
            }
        )

        logger.info(f"  Components: {avg_components:.1f}")
        logger.info(f"  Largest component: {avg_largest_fraction*100:.1f}%")
        logger.info(f"  Connected: {fraction_connected*100:.0f}%")
        logger.info("")

    # Diagnosis
    all_disconnected = all(m["fraction_connected"] < 0.5 for m in results["metrics"])

    if all_disconnected:
        logger.info("DIAGNOSIS: ✓ CCF graphs remain DISCONNECTED")
        logger.info("→ This violates van der Hoorn convergence assumptions")
        logger.info("→ Disconnected graphs can show divergent curvature")
        results["diagnosis"] = "disconnected"
    else:
        logger.info("DIAGNOSIS: CCF graphs become connected at large N")
        logger.info("→ Disconnectivity is NOT the cause of divergence")
        results["diagnosis"] = "connected"

    logger.info("")

    return results


def test_ccf_curvature_scaling(node_counts: List[int], num_realizations: int = 10) -> Dict:
    """
    Hypothesis 2: CCF curvature diverges as κ ~ -N^α with α > 0.

    Test: Measure curvature scaling and fit power law.

    Expected: If α > 0, curvature grows more negative (diverges).
    If α < 0, curvature approaches zero (converges).
    """
    logger.info("=" * 70)
    logger.info("HYPOTHESIS 2: CCF Curvature Scaling")
    logger.info("=" * 70)
    logger.info("Testing: How does κ_OR scale with N?")
    logger.info("")

    results = {"node_counts": [], "mean_curvatures": [], "std_curvatures": []}

    for num_nodes in node_counts:
        # Only compute curvature for tractable sizes
        if num_nodes > 1000:
            logger.info(f"Skipping N={num_nodes} (too large for curvature computation)")
            continue

        logger.info(f"Testing N = {num_nodes}...")

        ensemble_curvatures = []

        for realization_idx in tqdm(range(num_realizations), desc=f"N={num_nodes}", leave=False):
            state = generate_ccf_bigraph(num_nodes, seed=realization_idx)
            graph = state.to_networkx()

            # Compute curvature on largest component
            components = list(nx.connected_components(graph))
            if components:
                largest_component = max(components, key=len)
                subgraph = graph.subgraph(largest_component).copy()

                if subgraph.number_of_edges() > 0:
                    orc = OllivierRicciCurvature()
                    mean_curv, _ = orc.mean_curvature(subgraph)
                    ensemble_curvatures.append(mean_curv)

        if ensemble_curvatures:
            mean_kappa = np.mean(ensemble_curvatures)
            std_kappa = np.std(ensemble_curvatures)

            results["node_counts"].append(num_nodes)
            results["mean_curvatures"].append(mean_kappa)
            results["std_curvatures"].append(std_kappa)

            logger.info(f"  κ_OR = {mean_kappa:+.6f} ± {std_kappa:.6f}")
        else:
            logger.warning(f"  No valid curvature measurements for N={num_nodes}")

        logger.info("")

    # Fit power law: κ_OR ~ A * N^α
    if len(results["node_counts"]) >= 3:
        node_counts_arr = np.array(results["node_counts"])
        curvatures_arr = np.abs(np.array(results["mean_curvatures"]))

        log_N = np.log(node_counts_arr)
        log_kappa = np.log(curvatures_arr)

        from scipy.stats import linregress

        slope, intercept, r_value, p_value, std_err = linregress(log_N, log_kappa)

        results["power_law_exponent"] = slope
        results["power_law_exponent_error"] = std_err
        results["r_squared"] = r_value**2

        logger.info(f"Power law fit: |κ_OR| ~ N^{slope:.3f} ± {std_err:.3f}")
        logger.info(f"R² = {r_value**2:.4f}")
        logger.info("")

        if slope > 0.1:
            logger.info("DIAGNOSIS: ✓ DIVERGENCE (α > 0)")
            logger.info(f"→ Curvature grows as N^{slope:.2f}")
            logger.info("→ No convergence to continuum limit")
            results["diagnosis"] = "divergence"
        elif slope < -0.1:
            logger.info("DIAGNOSIS: CONVERGENCE (α < 0)")
            logger.info(f"→ Curvature decreases as N^{slope:.2f}")
            logger.info("→ May converge to continuum limit")
            results["diagnosis"] = "convergence"
        else:
            logger.info("DIAGNOSIS: UNCLEAR (α ≈ 0)")
            logger.info("→ Need more data or larger N")
            results["diagnosis"] = "unclear"
    else:
        logger.warning("Not enough data points for power law fit")
        results["diagnosis"] = "insufficient_data"

    logger.info("")

    return results


def test_constrained_ccf(num_nodes: int = 500, num_realizations: int = 10) -> Dict:
    """
    Hypothesis 3: Constraint modifications can restore convergence.

    Test: Modify CCF rewriting to enforce connectivity and measure curvature.

    This is a FORWARD-LOOKING test: what constraints would make CCF viable?
    """
    logger.info("=" * 70)
    logger.info("HYPOTHESIS 3: Constrained CCF")
    logger.info("=" * 70)
    logger.info("Testing: Can constraints restore convergence?")
    logger.info("")

    results = {"num_nodes": num_nodes, "num_realizations": num_realizations, "constraints": []}

    # Baseline: unconstrained CCF
    logger.info("Constraint: NONE (baseline CCF)")

    baseline_curvatures = []
    for idx in tqdm(range(num_realizations), desc="Baseline", leave=False):
        state = generate_ccf_bigraph(num_nodes, seed=idx)
        graph = state.to_networkx()

        components = list(nx.connected_components(graph))
        if components:
            largest = max(components, key=len)
            subgraph = graph.subgraph(largest).copy()

            if subgraph.number_of_edges() > 0:
                orc = OllivierRicciCurvature()
                mean_curv, _ = orc.mean_curvature(subgraph)
                baseline_curvatures.append(mean_curv)

    baseline_mean = np.mean(baseline_curvatures) if baseline_curvatures else np.nan

    results["constraints"].append(
        {
            "type": "none",
            "mean_curvature": baseline_mean,
            "description": "Standard CCF rewriting (unconstrained)",
        }
    )

    logger.info(f"  κ_OR = {baseline_mean:+.6f}")
    logger.info("")

    # Constraint 1: Force connectivity (theoretical)
    logger.info("Constraint: FORCE_CONNECTIVITY (theoretical)")
    logger.info("  → Modify rewriting to maintain single connected component")
    logger.info("  → Implementation: Add edges to merge components")
    logger.info("  → Result: NOT IMPLEMENTED (requires CCF modification)")

    results["constraints"].append(
        {
            "type": "force_connectivity",
            "mean_curvature": np.nan,
            "description": "Force graph to remain connected",
            "implemented": False,
        }
    )

    logger.info("")

    # Summary
    logger.info("DIAGNOSIS:")
    logger.info("→ Baseline CCF shows negative curvature")
    logger.info("→ Constraint testing requires CCF core modifications")
    logger.info("→ This is an OPEN PROBLEM for future work")
    logger.info("")

    return results


# ============================================================================
# Visualization
# ============================================================================


def create_diagnosis_plots(connectivity_results: Dict, scaling_results: Dict, output_path: Path):
    """Create diagnostic visualization."""

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Plot 1: Connectivity vs N
    ax1 = fig.add_subplot(gs[0, 0])

    node_counts = connectivity_results["node_counts"]
    metrics = connectivity_results["metrics"]

    fraction_connected = [m["fraction_connected"] * 100 for m in metrics]
    avg_components = [m["avg_components"] for m in metrics]

    ax1.plot(
        node_counts,
        fraction_connected,
        "o-",
        color="#2E86AB",
        linewidth=2,
        markersize=8,
        label="% Connected",
    )
    ax1.set_xlabel("Number of Nodes N", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Fraction Connected (%)", fontsize=12, fontweight="bold")
    ax1.set_title("CCF Connectivity Analysis", fontsize=13, fontweight="bold")
    ax1.set_xscale("log")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    # Plot 2: Number of components
    ax2 = fig.add_subplot(gs[0, 1])

    ax2.plot(node_counts, avg_components, "s-", color="#A23B72", linewidth=2, markersize=8)
    ax2.set_xlabel("Number of Nodes N", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Average # Components", fontsize=12, fontweight="bold")
    ax2.set_title("Graph Fragmentation", fontsize=13, fontweight="bold")
    ax2.set_xscale("log")
    ax2.grid(True, alpha=0.3)

    # Plot 3: Curvature scaling
    ax3 = fig.add_subplot(gs[1, :])

    if scaling_results["node_counts"]:
        node_counts_curv = np.array(scaling_results["node_counts"])
        mean_curvatures = np.array(scaling_results["mean_curvatures"])
        std_curvatures = np.array(scaling_results["std_curvatures"])

        ax3.errorbar(
            node_counts_curv,
            np.abs(mean_curvatures),
            yerr=std_curvatures,
            fmt="o-",
            color="#F18F01",
            capsize=5,
            capthick=2,
            markersize=8,
            linewidth=2,
            label="|κ_OR| (CCF bigraphs)",
        )

        # Plot power law fit
        if "power_law_exponent" in scaling_results:
            alpha = scaling_results["power_law_exponent"]
            alpha_err = scaling_results["power_law_exponent_error"]

            N_fit = np.logspace(
                np.log10(node_counts_curv.min()), np.log10(node_counts_curv.max()), 100
            )

            # Fit line through data
            prefactor = np.exp(
                np.mean(np.log(np.abs(mean_curvatures)) - alpha * np.log(node_counts_curv))
            )
            kappa_fit = prefactor * N_fit**alpha

            ax3.plot(
                N_fit,
                kappa_fit,
                "--",
                color="#C73E1D",
                linewidth=2.5,
                label=f"Fit: N^{alpha:.2f} ± {alpha_err:.2f}",
            )

        ax3.set_xlabel("Number of Nodes N", fontsize=12, fontweight="bold")
        ax3.set_ylabel("|Ollivier-Ricci Curvature|", fontsize=12, fontweight="bold")
        ax3.set_title(
            "CCF Curvature Divergence (κ_OR ~ -N^α, α > 0)", fontsize=13, fontweight="bold"
        )
        ax3.set_xscale("log")
        ax3.set_yscale("log")
        ax3.grid(True, alpha=0.3, which="both")
        ax3.legend(fontsize=11)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Diagnostic plots saved to {output_path}")
    plt.close()


# ============================================================================
# Main Execution
# ============================================================================


def main():
    """
    Main diagnostic workflow.

    This is NOT about proving CCF works.
    This is about DIAGNOSING why it fails and DOCUMENTING that failure.
    """
    logger.info("=" * 70)
    logger.info("CCF CURVATURE DIVERGENCE DIAGNOSIS")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Goal: Understand WHY CCF bigraphs show curvature divergence")
    logger.info("")
    logger.info("Current observation: κ_OR ~ -N^0.55 (diverging negative)")
    logger.info("Expected (if converging): κ_OR → 0 as N → ∞")
    logger.info("")

    # Configuration
    node_counts_connectivity = [50, 100, 200, 500, 1000, 2000]
    node_counts_curvature = [50, 100, 200, 500]  # Smaller for curvature (slow)
    num_realizations = 10

    # Output directory
    output_dir = Path("/Users/eirikr/1_Workspace/cosmos/papers/paper3-ccf-curvature/data")

    # Test 1: Connectivity
    connectivity_results = test_ccf_connectivity(
        node_counts_connectivity, num_realizations=num_realizations
    )

    # Test 2: Curvature scaling
    scaling_results = test_ccf_curvature_scaling(
        node_counts_curvature, num_realizations=num_realizations
    )

    # Test 3: Constrained CCF (theoretical)
    constraint_results = test_constrained_ccf(num_nodes=500, num_realizations=num_realizations)

    # Save results
    all_results = {
        "connectivity": connectivity_results,
        "scaling": scaling_results,
        "constraints": constraint_results,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "ccf_divergence_diagnosis.json"

    with open(results_path, "w") as json_file:
        json.dump(all_results, json_file, indent=2)

    logger.info(f"Results saved to {results_path}")

    # Create plots
    plot_path = output_dir.parent / "figures" / "ccf_divergence_diagnosis.pdf"
    create_diagnosis_plots(connectivity_results, scaling_results, plot_path)

    # Final summary
    logger.info("=" * 70)
    logger.info("DIVERGENCE DIAGNOSIS SUMMARY")
    logger.info("=" * 70)
    logger.info("")

    # Connectivity diagnosis
    if connectivity_results["diagnosis"] == "disconnected":
        logger.info("✓ PRIMARY CAUSE: Disconnected graphs")
        logger.info("  → CCF bigraphs remain disconnected at all tested scales")
        logger.info("  → This violates van der Hoorn convergence assumptions")
        logger.info("  → Theorem requires connected (or giant component) graphs")

    # Scaling diagnosis
    if "power_law_exponent" in scaling_results:
        alpha = scaling_results["power_law_exponent"]
        logger.info("")
        logger.info(f"✓ SCALING LAW: |κ_OR| ~ N^{alpha:.3f}")

        if alpha > 0:
            logger.info("  → DIVERGENCE confirmed (α > 0)")
            logger.info("  → Curvature grows more negative with N")
            logger.info("  → No convergence to continuum limit")
        else:
            logger.info("  → Unexpected convergence (α < 0)")

    logger.info("")
    logger.info("=" * 70)
    logger.info("CONCLUSION")
    logger.info("=" * 70)
    logger.info("")
    logger.info("CCF bigraphs operate OUTSIDE the regime where Ollivier-Ricci")
    logger.info("curvature convergence theorems apply.")
    logger.info("")
    logger.info("This does NOT invalidate CCF as a computational framework.")
    logger.info("It DOES invalidate claims that CCF provides a 'route to GR'")
    logger.info("via curvature convergence.")
    logger.info("")
    logger.info("Path forward:")
    logger.info("1. Document this failure mode honestly in the paper")
    logger.info("2. Identify constraints that would restore convergence")
    logger.info("3. Frame as 'open problem' not 'established result'")
    logger.info("")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
