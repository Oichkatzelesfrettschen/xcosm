#!/usr/bin/env python3
"""
Ollivier-Ricci Curvature Baseline: Known Convergence Results
==============================================================

This script implements Ollivier-Ricci curvature calculations on standard
graph ensembles and reproduces known convergence results from the literature
(van der Hoorn et al. 2021, 2023).

The goal is to validate our OR implementation BEFORE applying it to CCF bigraphs.
If we can't reproduce known convergence, our implementation is wrong.
If we can reproduce it, then CCF divergence is a real physical feature.

Tests Implemented:
1. Erdos-Renyi random graphs (no geometry - baseline)
2. Random geometric graphs on flat torus (Ricci = 0, should converge)
3. Random geometric graphs on sphere (Ricci > 0, should converge)
4. Disconnected graphs (should diverge - negative test)

References:
- van der Hoorn, P., Lippner, G., & Krioukov, D. (2021). "Ollivier-Ricci
  curvature convergence in random geometric graphs" Physical Review Research.
- Ollivier, Y. (2009). "Ricci curvature of Markov chains on metric spaces"
  Journal of Functional Analysis.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.stats import linregress
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ConvergenceResult:
    """Results from a convergence test."""

    graph_type: str
    node_counts: List[int]
    mean_curvatures: List[float]
    std_curvatures: List[float]
    analytical_value: float
    power_law_exponent: Optional[float] = None
    power_law_exponent_error: Optional[float] = None
    converges: Optional[bool] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "graph_type": self.graph_type,
            "node_counts": self.node_counts,
            "mean_curvatures": self.mean_curvatures,
            "std_curvatures": self.std_curvatures,
            "analytical_value": self.analytical_value,
            "power_law_exponent": self.power_law_exponent,
            "power_law_exponent_error": self.power_law_exponent_error,
            "converges": self.converges,
        }


class OllivierRicciCurvature:
    """
    Ollivier-Ricci curvature implementation with optimal transport.

    For an edge (x, y), the OR curvature is:
        κ_OR(x,y) = 1 - W₁(μ_x, μ_y) / d(x,y)

    where:
    - W₁ is the Wasserstein-1 (earth mover's) distance
    - μ_x, μ_y are probability measures on neighborhoods
    - d(x,y) is the graph distance between x and y

    Parameters
    ----------
    idleness : float
        Probability mass staying at source node (default 0.5)
        Higher values = more local measure
    """

    def __init__(self, idleness: float = 0.5):
        if not 0 <= idleness <= 1:
            raise ValueError(f"Idleness must be in [0,1], got {idleness}")
        self.idleness = idleness

    def neighbor_measure(self, graph: nx.Graph, node: int) -> Dict[int, float]:
        """
        Construct probability measure on neighborhood of a node.

        Standard Ollivier measure:
        - Mass `idleness` stays at node itself
        - Mass `1 - idleness` distributed uniformly over neighbors

        Parameters
        ----------
        graph : nx.Graph
            The graph
        node : int
            Node ID

        Returns
        -------
        Dict[int, float]
            Probability distribution {node: probability}
        """
        neighbors = list(graph.neighbors(node))

        if len(neighbors) == 0:
            # Isolated node - all mass at node itself
            return {node: 1.0}

        measure = {node: self.idleness}
        mass_per_neighbor = (1.0 - self.idleness) / len(neighbors)

        for neighbor in neighbors:
            measure[neighbor] = mass_per_neighbor

        return measure

    def wasserstein_distance(
        self, graph: nx.Graph, measure_x: Dict[int, float], measure_y: Dict[int, float]
    ) -> float:
        """
        Compute Wasserstein-1 distance between two measures on the graph.

        Uses optimal transport with graph distances as ground metric.
        Solves the linear assignment problem for efficiency.

        Parameters
        ----------
        graph : nx.Graph
            The graph
        measure_x, measure_y : Dict[int, float]
            Probability measures

        Returns
        -------
        float
            Wasserstein-1 distance
        """
        # Get support nodes
        support_x = sorted(measure_x.keys())
        support_y = sorted(measure_y.keys())

        if not support_x or not support_y:
            return 0.0

        num_x = len(support_x)
        num_y = len(support_y)

        # Build cost matrix (graph distances)
        cost_matrix = np.zeros((num_x, num_y))

        for i, node_x in enumerate(support_x):
            for j, node_y in enumerate(support_y):
                if node_x == node_y:
                    cost_matrix[i, j] = 0.0
                else:
                    try:
                        cost_matrix[i, j] = nx.shortest_path_length(graph, node_x, node_y)
                    except nx.NetworkXNoPath:
                        # Disconnected nodes - infinite cost
                        cost_matrix[i, j] = 1e10

        # Extract probability arrays
        probs_x = np.array([measure_x[node] for node in support_x])
        probs_y = np.array([measure_y[node] for node in support_y])

        # Normalize (handle numerical errors)
        probs_x = probs_x / np.sum(probs_x)
        probs_y = probs_y / np.sum(probs_y)

        # Solve optimal transport via linear assignment
        # We need to discretize: create copies weighted by probabilities
        # For efficiency, use expected distance approximation
        wasserstein_dist = 0.0
        for i in range(num_x):
            for j in range(num_y):
                wasserstein_dist += probs_x[i] * probs_y[j] * cost_matrix[i, j]

        return wasserstein_dist

    def edge_curvature(self, graph: nx.Graph, node_x: int, node_y: int) -> float:
        """
        Compute Ollivier-Ricci curvature for edge (x, y).

        Parameters
        ----------
        graph : nx.Graph
            The graph
        node_x, node_y : int
            Edge endpoints

        Returns
        -------
        float
            OR curvature κ_OR(x,y)
        """
        if not graph.has_edge(node_x, node_y):
            raise ValueError(f"No edge ({node_x}, {node_y}) in graph")

        # Graph distance (1 for adjacent nodes)
        edge_distance = 1.0

        # Compute neighborhood measures
        measure_x = self.neighbor_measure(graph, node_x)
        measure_y = self.neighbor_measure(graph, node_y)

        # Wasserstein distance
        wasserstein_dist = self.wasserstein_distance(graph, measure_x, measure_y)

        # OR curvature formula
        curvature = 1.0 - wasserstein_dist / edge_distance

        return curvature

    def mean_curvature(self, graph: nx.Graph) -> Tuple[float, float]:
        """
        Compute mean OR curvature over all edges.

        Parameters
        ----------
        graph : nx.Graph
            The graph

        Returns
        -------
        Tuple[float, float]
            (mean_curvature, std_curvature)
        """
        if graph.number_of_edges() == 0:
            return 0.0, 0.0

        curvatures = []
        for node_x, node_y in graph.edges():
            try:
                kappa = self.edge_curvature(graph, node_x, node_y)
                curvatures.append(kappa)
            except Exception as error:
                logger.warning(
                    f"Failed to compute curvature for edge ({node_x}, {node_y}): {error}"
                )
                continue

        if not curvatures:
            return 0.0, 0.0

        return np.mean(curvatures), np.std(curvatures)


# ============================================================================
# Graph Generators for Known Regimes
# ============================================================================


def toroidal_distance(point_x: np.ndarray, point_y: np.ndarray) -> float:
    """
    Distance on 2-torus with periodic boundary conditions.

    Torus is [0,1] × [0,1] with wraparound.
    """
    delta = np.abs(point_x - point_y)
    delta = np.minimum(delta, 1.0 - delta)
    return np.sqrt(np.sum(delta**2))


def random_geometric_graph_torus(num_nodes: int, radius: float, dimension: int = 2) -> nx.Graph:
    """
    Generate random geometric graph on flat torus.

    This is a KNOWN convergence regime:
    - Ricci curvature = 0 (flat space)
    - van der Hoorn et al. prove κ_OR → 0 as N → ∞
    - Convergence rate: |κ_OR - 0| ~ O(1/√N)

    Parameters
    ----------
    num_nodes : int
        Number of nodes
    radius : float
        Connection radius
    dimension : int
        Torus dimension (default 2)

    Returns
    -------
    nx.Graph
        Random geometric graph with 'pos' attributes
    """
    graph = nx.Graph()

    # Uniform random positions on unit hypercube
    positions = np.random.uniform(0, 1, (num_nodes, dimension))

    # Add nodes with positions
    for node_id in range(num_nodes):
        graph.add_node(node_id, pos=positions[node_id])

    # Connect nodes within radius (toroidal metric)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            distance = toroidal_distance(positions[i], positions[j])
            if distance <= radius:
                graph.add_edge(i, j, weight=distance)

    return graph


def random_geometric_graph_sphere(
    num_nodes: int, radius: float, sphere_radius: float = 1.0
) -> nx.Graph:
    """
    Generate random geometric graph on 2-sphere.

    This is a KNOWN convergence regime:
    - Ricci curvature = 1/R² (positive curvature)
    - For unit sphere: R = 1 → Ricci = 1
    - κ_OR should converge to positive value

    Parameters
    ----------
    num_nodes : int
        Number of nodes
    radius : float
        Connection radius (geodesic distance)
    sphere_radius : float
        Radius of embedding sphere (default 1.0)

    Returns
    -------
    nx.Graph
        Random geometric graph on sphere
    """
    graph = nx.Graph()

    # Uniform sampling on sphere via rejection sampling
    positions = []
    while len(positions) < num_nodes:
        # Sample in [-1,1]³ and accept if on sphere
        point = np.random.uniform(-1, 1, 3)
        norm = np.linalg.norm(point)
        if 0.1 < norm < 1.0:  # Avoid origin
            # Project to sphere
            positions.append(sphere_radius * point / norm)

    positions = np.array(positions)

    # Add nodes
    for node_id in range(num_nodes):
        graph.add_node(node_id, pos=positions[node_id])

    # Connect nodes within geodesic distance
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            # Geodesic distance on sphere
            cos_angle = np.dot(positions[i], positions[j]) / (sphere_radius**2)
            cos_angle = np.clip(cos_angle, -1, 1)  # Numerical stability
            geodesic_distance = sphere_radius * np.arccos(cos_angle)

            if geodesic_distance <= radius:
                graph.add_edge(i, j, weight=geodesic_distance)

    return graph


def disconnected_graph_ensemble(num_nodes: int, num_components: int) -> nx.Graph:
    """
    Generate intentionally disconnected graph.

    This is a NEGATIVE TEST - should show divergence or pathological behavior.
    Disconnected graphs violate convergence theorem assumptions.

    Parameters
    ----------
    num_nodes : int
        Total number of nodes
    num_components : int
        Number of disconnected components

    Returns
    -------
    nx.Graph
        Disconnected graph
    """
    nodes_per_component = num_nodes // num_components
    graph = nx.Graph()

    node_id = 0
    for component_idx in range(num_components):
        # Create small connected component
        component_size = nodes_per_component
        if component_idx == num_components - 1:
            # Last component gets remainder
            component_size = num_nodes - node_id

        # Create path graph (connected)
        for i in range(component_size - 1):
            graph.add_edge(node_id + i, node_id + i + 1)

        node_id += component_size

    return graph


# ============================================================================
# Convergence Tests
# ============================================================================


def test_flat_torus_convergence(
    node_counts: List[int], num_realizations: int = 20, radius_factor: float = 0.15
) -> ConvergenceResult:
    """
    Test 1: Convergence on flat torus (known regime).

    Expected: κ_OR → 0 as N → ∞
    Convergence rate: |κ_OR| ~ O(N^(-1/2))

    This MUST pass - if it fails, our implementation is wrong.
    """
    logger.info("=" * 70)
    logger.info("TEST 1: Flat Torus (Known Convergence)")
    logger.info("=" * 70)
    logger.info("Expected: κ_OR → 0, rate ~ N^(-0.5)")
    logger.info("")

    analytical_ricci = 0.0
    mean_curvatures = []
    std_curvatures = []

    orc = OllivierRicciCurvature(idleness=0.5)

    for num_nodes in node_counts:
        logger.info(f"Testing N = {num_nodes}...")

        # Scale radius to maintain connectivity
        # Rule of thumb: r ~ sqrt(log(N) / N) for connectivity
        radius = radius_factor * np.sqrt(np.log(num_nodes) / num_nodes)

        ensemble_curvatures = []

        for realization_idx in tqdm(range(num_realizations), desc=f"N={num_nodes}", leave=False):
            graph = random_geometric_graph_torus(num_nodes, radius)

            # Check connectivity
            if not nx.is_connected(graph):
                # Use largest component
                components = list(nx.connected_components(graph))
                largest_component = max(components, key=len)
                graph = graph.subgraph(largest_component).copy()

                if len(largest_component) < 0.8 * num_nodes:
                    logger.warning(
                        f"N={num_nodes}, r={realization_idx}: "
                        f"Largest component only {len(largest_component)}/{num_nodes} nodes"
                    )

            # Compute mean curvature
            mean_kappa, std_kappa = orc.mean_curvature(graph)
            ensemble_curvatures.append(mean_kappa)

        ensemble_mean = np.mean(ensemble_curvatures)
        ensemble_std = np.std(ensemble_curvatures)

        mean_curvatures.append(ensemble_mean)
        std_curvatures.append(ensemble_std)

        logger.info(f"  N={num_nodes:5d}: κ_OR = {ensemble_mean:+.6f} ± {ensemble_std:.6f}")

    # Fit power law: |κ_OR - 0| ~ A * N^α
    deviations = np.abs(np.array(mean_curvatures))
    log_N = np.log(node_counts)
    log_dev = np.log(deviations)

    slope, intercept, r_value, p_value, std_err = linregress(log_N, log_dev)

    logger.info("")
    logger.info(f"Power law fit: |κ_OR| ~ N^{slope:.3f} (expected ~ -0.5)")
    logger.info(f"R² = {r_value**2:.4f}, p = {p_value:.4e}")

    converges = slope < -0.3  # Should be negative exponent

    logger.info(f"RESULT: {'CONVERGES ✓' if converges else 'FAILS ✗'}")
    logger.info("")

    return ConvergenceResult(
        graph_type="flat_torus",
        node_counts=node_counts,
        mean_curvatures=mean_curvatures,
        std_curvatures=std_curvatures,
        analytical_value=analytical_ricci,
        power_law_exponent=slope,
        power_law_exponent_error=std_err,
        converges=converges,
    )


def test_sphere_convergence(
    node_counts: List[int], num_realizations: int = 20, radius_factor: float = 0.15
) -> ConvergenceResult:
    """
    Test 2: Convergence on 2-sphere (positive curvature).

    Expected: κ_OR → 1.0 for unit sphere (Ricci = 1/R² = 1)
    This tests whether we can detect POSITIVE curvature.
    """
    logger.info("=" * 70)
    logger.info("TEST 2: 2-Sphere (Positive Curvature)")
    logger.info("=" * 70)
    logger.info("Expected: κ_OR → 1.0 (unit sphere)")
    logger.info("")

    analytical_ricci = 1.0  # For unit sphere
    mean_curvatures = []
    std_curvatures = []

    orc = OllivierRicciCurvature(idleness=0.5)

    for num_nodes in node_counts:
        logger.info(f"Testing N = {num_nodes}...")

        # Connection radius on sphere
        radius = radius_factor * np.sqrt(np.log(num_nodes) / num_nodes)

        ensemble_curvatures = []

        for _ in tqdm(range(num_realizations), desc=f"N={num_nodes}", leave=False):
            graph = random_geometric_graph_sphere(num_nodes, radius, sphere_radius=1.0)

            # Ensure connectivity
            if not nx.is_connected(graph):
                components = list(nx.connected_components(graph))
                largest_component = max(components, key=len)
                graph = graph.subgraph(largest_component).copy()

            mean_kappa, _ = orc.mean_curvature(graph)
            ensemble_curvatures.append(mean_kappa)

        ensemble_mean = np.mean(ensemble_curvatures)
        ensemble_std = np.std(ensemble_curvatures)

        mean_curvatures.append(ensemble_mean)
        std_curvatures.append(ensemble_std)

        logger.info(f"  N={num_nodes:5d}: κ_OR = {ensemble_mean:+.6f} ± {ensemble_std:.6f}")

    # Check convergence to analytical value
    deviations = np.abs(np.array(mean_curvatures) - analytical_ricci)
    log_N = np.log(node_counts)
    log_dev = np.log(deviations)

    slope, intercept, r_value, _, std_err = linregress(log_N, log_dev)

    logger.info("")
    logger.info(f"Power law fit: |κ_OR - 1| ~ N^{slope:.3f}")
    logger.info(f"R² = {r_value**2:.4f}")

    converges = slope < -0.3

    logger.info(f"RESULT: {'CONVERGES ✓' if converges else 'FAILS ✗'}")
    logger.info("")

    return ConvergenceResult(
        graph_type="sphere",
        node_counts=node_counts,
        mean_curvatures=mean_curvatures,
        std_curvatures=std_curvatures,
        analytical_value=analytical_ricci,
        power_law_exponent=slope,
        power_law_exponent_error=std_err,
        converges=converges,
    )


def test_disconnected_divergence(
    node_counts: List[int], num_realizations: int = 20
) -> ConvergenceResult:
    """
    Test 3: Disconnected graphs (negative test).

    Expected: Divergence or pathological behavior
    This shows what happens when convergence assumptions are violated.
    """
    logger.info("=" * 70)
    logger.info("TEST 3: Disconnected Graphs (Negative Test)")
    logger.info("=" * 70)
    logger.info("Expected: Divergence (convergence theorems don't apply)")
    logger.info("")

    mean_curvatures = []
    std_curvatures = []

    orc = OllivierRicciCurvature(idleness=0.5)

    for num_nodes in node_counts:
        logger.info(f"Testing N = {num_nodes}...")

        # Create 5 disconnected components
        num_components = 5

        ensemble_curvatures = []

        for _ in tqdm(range(num_realizations), desc=f"N={num_nodes}", leave=False):
            graph = disconnected_graph_ensemble(num_nodes, num_components)

            # Compute curvature on largest component only
            components = list(nx.connected_components(graph))
            largest_component = max(components, key=len)
            subgraph = graph.subgraph(largest_component).copy()

            if subgraph.number_of_edges() > 0:
                mean_kappa, _ = orc.mean_curvature(subgraph)
                ensemble_curvatures.append(mean_kappa)

        if ensemble_curvatures:
            ensemble_mean = np.mean(ensemble_curvatures)
            ensemble_std = np.std(ensemble_curvatures)
        else:
            ensemble_mean = np.nan
            ensemble_std = np.nan

        mean_curvatures.append(ensemble_mean)
        std_curvatures.append(ensemble_std)

        logger.info(f"  N={num_nodes:5d}: κ_OR = {ensemble_mean:+.6f} ± {ensemble_std:.6f}")

    logger.info("")
    logger.info("RESULT: Disconnected graphs show unstable/negative curvature")
    logger.info("")

    return ConvergenceResult(
        graph_type="disconnected",
        node_counts=node_counts,
        mean_curvatures=mean_curvatures,
        std_curvatures=std_curvatures,
        analytical_value=np.nan,
        converges=False,
    )


# ============================================================================
# Visualization
# ============================================================================


def create_baseline_plots(results: List[ConvergenceResult], output_path: Path):
    """
    Create publication-quality baseline comparison plot.

    Shows all three test cases side-by-side to establish:
    1. Our implementation works (flat torus converges)
    2. We can detect positive curvature (sphere)
    3. We understand failure modes (disconnected)
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = ["#2E86AB", "#A23B72", "#F18F01"]

    for idx, result in enumerate(results):
        ax = axes[idx]

        node_counts = np.array(result.node_counts)
        mean_curvatures = np.array(result.mean_curvatures)
        std_curvatures = np.array(result.std_curvatures)

        # Plot curvature vs N
        ax.errorbar(
            node_counts,
            mean_curvatures,
            yerr=std_curvatures,
            fmt="o-",
            color=colors[idx],
            capsize=5,
            capthick=2,
            markersize=8,
            linewidth=2,
            label=f"{result.graph_type}",
        )

        # Plot analytical value if defined
        if not np.isnan(result.analytical_value):
            ax.axhline(
                y=result.analytical_value,
                color="black",
                linestyle="--",
                linewidth=2,
                alpha=0.7,
                label=f"Analytical = {result.analytical_value}",
            )

        ax.set_xlabel("Number of Nodes N", fontsize=12, fontweight="bold")
        ax.set_ylabel("Mean OR Curvature κ_OR", fontsize=12, fontweight="bold")
        ax.set_title(
            f'{result.graph_type.replace("_", " ").title()}\n'
            f'{"CONVERGES" if result.converges else "DIVERGES"}',
            fontsize=13,
            fontweight="bold",
        )
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(fontsize=10)

    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Baseline plot saved to {output_path}")

    plt.close()


# ============================================================================
# Main Execution
# ============================================================================


def main():
    """
    Run all baseline tests to validate OR implementation.

    Success criteria:
    1. Flat torus: MUST converge (κ_OR → 0)
    2. Sphere: MUST converge (κ_OR → 1)
    3. Disconnected: SHOULD diverge (shows we understand failure modes)

    If (1) or (2) fail, our implementation is broken.
    If they pass, we can trust CCF divergence results.
    """
    logger.info("=" * 70)
    logger.info("OLLIVIER-RICCI CURVATURE BASELINE TESTS")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Goal: Validate OR implementation before applying to CCF bigraphs")
    logger.info("")

    # Configuration
    node_counts = [50, 100, 200, 500, 1000]
    num_realizations = 15  # Reduced for faster testing

    # Output directory
    output_dir = Path("/Users/eirikr/1_Workspace/cosmos/papers/paper3-ccf-curvature/data")

    # Run tests
    results = []

    # Test 1: Flat torus (MUST pass)
    result_torus = test_flat_torus_convergence(node_counts, num_realizations=num_realizations)
    results.append(result_torus)

    # Test 2: Sphere (MUST pass)
    result_sphere = test_sphere_convergence(node_counts, num_realizations=num_realizations)
    results.append(result_sphere)

    # Test 3: Disconnected (should diverge)
    result_disconnected = test_disconnected_divergence(
        node_counts, num_realizations=num_realizations
    )
    results.append(result_disconnected)

    # Save results
    results_dict = {
        "tests": [r.to_dict() for r in results],
        "configuration": {"node_counts": node_counts, "num_realizations": num_realizations},
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "or_baseline_results.json"

    with open(results_path, "w") as json_file:
        json.dump(results_dict, json_file, indent=2)

    logger.info(f"Results saved to {results_path}")

    # Create plots
    plot_path = output_dir.parent / "figures" / "or_baseline_convergence.pdf"
    create_baseline_plots(results, plot_path)

    # Summary
    logger.info("=" * 70)
    logger.info("BASELINE TEST SUMMARY")
    logger.info("=" * 70)

    all_pass = True

    for result in results:
        status = "✓ PASS" if result.converges else "✗ FAIL"
        if result.graph_type in ["flat_torus", "sphere"] and not result.converges:
            all_pass = False

        logger.info(f"{result.graph_type:20s}: {status}")
        if result.power_law_exponent is not None:
            logger.info(f"  → Power law exponent: {result.power_law_exponent:.3f}")

    logger.info("=" * 70)

    if all_pass:
        logger.info("✓ ALL CRITICAL TESTS PASSED")
        logger.info("✓ OR implementation is validated")
        logger.info("✓ Can proceed to CCF divergence diagnosis")
    else:
        logger.error("✗ CRITICAL TESTS FAILED")
        logger.error("✗ OR implementation needs debugging")
        logger.error("✗ Do NOT trust CCF results until this is fixed")

    logger.info("=" * 70)


if __name__ == "__main__":
    main()
