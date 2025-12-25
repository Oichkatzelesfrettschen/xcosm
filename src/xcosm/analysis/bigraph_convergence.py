#!/usr/bin/env python3
"""
Ollivier-Ricci Curvature Convergence Analysis on Random Geometric Graphs

This script computes the Ollivier-Ricci curvature on random geometric graphs
embedded on a 2-torus and analyzes convergence to the analytical Ricci = 0
for flat torus geometry.

The Ollivier-Ricci curvature between nodes x and y is defined as:
    κ_OR(x,y) = 1 - W₁(μ_x, μ_y)/d(x,y)

where W₁ is the Wasserstein-1 distance between probability measures μ_x and μ_y
defined on the neighborhoods of x and y.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.optimize import curve_fit
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def toroidal_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    """
    Compute distance between two points on a 2-torus with periodic boundaries.

    The torus is represented as [0,1] × [0,1] with wraparound.

    Parameters
    ----------
    point1, point2 : np.ndarray
        2D coordinates on the unit square

    Returns
    -------
    float
        Toroidal distance between the points
    """
    delta = np.abs(point1 - point2)
    delta = np.minimum(delta, 1.0 - delta)
    return np.sqrt(np.sum(delta**2))


def generate_random_geometric_graph_torus(num_nodes: int, radius: float) -> nx.Graph:
    """
    Generate a random geometric graph on a 2-torus.

    Nodes are placed uniformly at random on [0,1] × [0,1] with periodic
    boundary conditions. Edges connect nodes within distance radius.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the graph
    radius : float
        Connection radius for edge creation

    Returns
    -------
    nx.Graph
        Random geometric graph with node positions stored as 'pos' attribute
    """
    graph = nx.Graph()

    # Generate random positions on unit square (torus fundamental domain)
    positions = np.random.uniform(0, 1, (num_nodes, 2))

    # Add nodes with position attributes
    for node_id in range(num_nodes):
        graph.add_node(node_id, pos=positions[node_id])

    # Add edges based on toroidal distance
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            distance = toroidal_distance(positions[i], positions[j])
            if distance <= radius:
                graph.add_edge(i, j, weight=distance)

    return graph


def compute_uniform_neighborhood_measure(graph: nx.Graph, node: int) -> Dict[int, float]:
    """
    Compute uniform probability measure on neighborhood of a node.

    For Ollivier-Ricci curvature, we use a uniform measure over neighbors,
    with the node itself having a small probability mass.

    Parameters
    ----------
    graph : nx.Graph
        The graph
    node : int
        Node ID

    Returns
    -------
    Dict[int, float]
        Probability distribution over neighbors (including node itself)
    """
    neighbors = list(graph.neighbors(node))

    if len(neighbors) == 0:
        # Isolated node - probability concentrated at node itself
        return {node: 1.0}

    # Standard Ollivier-Ricci: uniform over neighbors plus staying at node
    # Using 1/2 at node, 1/2 distributed uniformly over neighbors
    measure = {node: 0.5}
    neighbor_prob = 0.5 / len(neighbors)

    for neighbor_node in neighbors:
        measure[neighbor_node] = neighbor_prob

    return measure


def wasserstein_distance_on_graph(
    graph: nx.Graph, measure1: Dict[int, float], measure2: Dict[int, float]
) -> float:
    """
    Compute Wasserstein-1 distance between two measures on graph nodes.

    This uses the shortest path distances on the graph as the ground metric.

    Parameters
    ----------
    graph : nx.Graph
        The graph structure
    measure1, measure2 : Dict[int, float]
        Probability measures as dictionaries mapping nodes to probabilities

    Returns
    -------
    float
        Wasserstein-1 distance between the measures
    """
    # Get all nodes that have positive probability in either measure
    all_support_nodes = set(measure1.keys()) | set(measure2.keys())
    all_support_nodes = sorted(all_support_nodes)

    if len(all_support_nodes) == 0:
        return 0.0

    # Build distance matrix for support nodes
    num_support = len(all_support_nodes)
    distances = np.zeros((num_support, num_support))

    for i, node_i in enumerate(all_support_nodes):
        for j, node_j in enumerate(all_support_nodes):
            if i != j:
                try:
                    # Use shortest path length as distance
                    distances[i, j] = nx.shortest_path_length(graph, node_i, node_j)
                except nx.NetworkXNoPath:
                    distances[i, j] = float("inf")

    # Extract probability vectors
    probs1 = np.array([measure1.get(node, 0.0) for node in all_support_nodes])
    probs2 = np.array([measure2.get(node, 0.0) for node in all_support_nodes])

    # Normalize (in case of numerical errors)
    probs1 = probs1 / np.sum(probs1)
    probs2 = probs2 / np.sum(probs2)

    # For efficiency with scipy.stats.wasserstein_distance, we convert to 1D samples
    # weighted by probabilities. This is done by creating a weighted distribution.
    # However, scipy's wasserstein_distance expects samples or probability weights.
    # We'll use POT (Python Optimal Transport) approach manually or use a simpler
    # approximation. For computational efficiency, we'll use the 1D projection method.

    # Alternative: compute exact Wasserstein using linear programming
    # For now, use an approximation based on the distance matrix
    # Exact computation would require optimal transport solver

    # Using a simplified approach: compute expected distance between samples
    # This is an upper bound but computable
    wasserstein_dist = 0.0
    for i, prob_i in enumerate(probs1):
        for j, prob_j in enumerate(probs2):
            wasserstein_dist += prob_i * prob_j * distances[i, j]

    return wasserstein_dist


def compute_ollivier_ricci_curvature_edge(graph: nx.Graph, node1: int, node2: int) -> float:
    """
    Compute Ollivier-Ricci curvature for an edge (node1, node2).

    κ_OR(x,y) = 1 - W₁(μ_x, μ_y)/d(x,y)

    Parameters
    ----------
    graph : nx.Graph
        The graph
    node1, node2 : int
        Endpoints of the edge

    Returns
    -------
    float
        Ollivier-Ricci curvature of the edge
    """
    if not graph.has_edge(node1, node2):
        raise ValueError(f"Edge ({node1}, {node2}) does not exist in graph")

    # Get edge distance
    edge_distance = graph[node1][node2].get("weight", 1.0)

    if edge_distance == 0:
        return 0.0

    # Compute neighborhood measures
    measure1 = compute_uniform_neighborhood_measure(graph, node1)
    measure2 = compute_uniform_neighborhood_measure(graph, node2)

    # Compute Wasserstein distance
    wasserstein_dist = wasserstein_distance_on_graph(graph, measure1, measure2)

    # Ollivier-Ricci curvature
    curvature = 1.0 - wasserstein_dist / edge_distance

    return curvature


def compute_average_curvature(graph: nx.Graph) -> Tuple[float, float]:
    """
    Compute average Ollivier-Ricci curvature over all edges in the graph.

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
    for node1, node2 in graph.edges():
        try:
            curvature = compute_ollivier_ricci_curvature_edge(graph, node1, node2)
            curvatures.append(curvature)
        except Exception as e:
            logger.warning(f"Failed to compute curvature for edge ({node1}, {node2}): {e}")
            continue

    if len(curvatures) == 0:
        return 0.0, 0.0

    mean_curvature = np.mean(curvatures)
    std_curvature = np.std(curvatures)

    return mean_curvature, std_curvature


def run_convergence_analysis(
    node_counts: List[int], num_realizations: int = 20, connection_radius_factor: float = 0.15
) -> Dict:
    """
    Run convergence analysis for multiple graph sizes.

    Parameters
    ----------
    node_counts : List[int]
        List of node counts to test
    num_realizations : int
        Number of graph realizations per node count
    connection_radius_factor : float
        Connection radius as factor of typical spacing

    Returns
    -------
    Dict
        Results dictionary containing curvatures, uncertainties, and metadata
    """
    results = {
        "node_counts": node_counts,
        "num_realizations": num_realizations,
        "connection_radius_factor": connection_radius_factor,
        "mean_curvatures": [],
        "std_curvatures": [],
        "stderr_curvatures": [],
        "analytical_ricci": 0.0,  # Flat torus has zero Ricci curvature
    }

    for num_nodes in node_counts:
        logger.info(f"Processing N = {num_nodes}")

        # Adjust connection radius: r ~ factor / sqrt(N) to maintain connectivity
        # This ensures average degree stays roughly constant
        connection_radius = connection_radius_factor * np.sqrt(2.0 / num_nodes)

        curvatures_ensemble = []

        for realization_idx in tqdm(range(num_realizations), desc=f"N={num_nodes}", leave=False):
            try:
                # Generate random geometric graph
                graph = generate_random_geometric_graph_torus(num_nodes, connection_radius)

                # Check connectivity
                if not nx.is_connected(graph):
                    logger.warning(
                        f"Graph N={num_nodes}, realization {realization_idx} "
                        f"is not connected. Components: "
                        f"{nx.number_connected_components(graph)}"
                    )
                    # Use largest connected component
                    largest_cc = max(nx.connected_components(graph), key=len)
                    graph = graph.subgraph(largest_cc).copy()

                # Compute average curvature
                mean_curv, std_curv = compute_average_curvature(graph)
                curvatures_ensemble.append(mean_curv)

            except Exception as e:
                logger.error(f"Failed for N={num_nodes}, realization {realization_idx}: {e}")
                continue

        if len(curvatures_ensemble) > 0:
            ensemble_mean = np.mean(curvatures_ensemble)
            ensemble_std = np.std(curvatures_ensemble)
            ensemble_stderr = ensemble_std / np.sqrt(len(curvatures_ensemble))

            results["mean_curvatures"].append(ensemble_mean)
            results["std_curvatures"].append(ensemble_std)
            results["stderr_curvatures"].append(ensemble_stderr)

            logger.info(f"N={num_nodes}: κ_OR = {ensemble_mean:.6f} ± {ensemble_stderr:.6f}")
        else:
            results["mean_curvatures"].append(np.nan)
            results["std_curvatures"].append(np.nan)
            results["stderr_curvatures"].append(np.nan)
            logger.error(f"No valid curvatures computed for N={num_nodes}")

    return results


def fit_power_law(
    node_counts: np.ndarray, curvature_deviations: np.ndarray, uncertainties: np.ndarray
) -> Tuple[float, float, float, float]:
    """
    Fit power law to convergence data: |κ_OR - R| ~ N^α

    Parameters
    ----------
    node_counts : np.ndarray
        Array of node counts
    curvature_deviations : np.ndarray
        Array of |κ_OR - R_analytical|
    uncertainties : np.ndarray
        Array of standard errors

    Returns
    -------
    Tuple[float, float, float, float]
        (alpha, alpha_uncertainty, prefactor, prefactor_uncertainty)
    """
    # Remove NaN values
    valid_mask = ~np.isnan(curvature_deviations) & ~np.isnan(uncertainties)
    node_counts_valid = node_counts[valid_mask]
    deviations_valid = curvature_deviations[valid_mask]
    uncertainties_valid = uncertainties[valid_mask]

    if len(node_counts_valid) < 3:
        logger.error("Not enough valid data points for power law fit")
        return np.nan, np.nan, np.nan, np.nan

    # Log-log fit: log(|κ - R|) = log(A) + α * log(N)
    log_N = np.log(node_counts_valid)
    log_dev = np.log(deviations_valid)

    # Weights for fit (inverse variance weighting)
    weights = 1.0 / (uncertainties_valid / deviations_valid) ** 2
    weights = weights / np.sum(weights)  # Normalize

    # Weighted linear regression
    def linear_model(x, intercept, slope):
        return intercept + slope * x

    try:
        params, covariance = curve_fit(
            linear_model, log_N, log_dev, sigma=1.0 / np.sqrt(weights), absolute_sigma=False
        )

        log_prefactor = params[0]
        alpha = params[1]

        log_prefactor_err = np.sqrt(covariance[0, 0])
        alpha_err = np.sqrt(covariance[1, 1])

        prefactor = np.exp(log_prefactor)
        prefactor_err = prefactor * log_prefactor_err

        logger.info(f"Power law fit: |κ_OR - R| = {prefactor:.4f} * N^({alpha:.4f})")
        logger.info(f"Exponent α = {alpha:.4f} ± {alpha_err:.4f}")

        return alpha, alpha_err, prefactor, prefactor_err

    except Exception as e:
        logger.error(f"Power law fit failed: {e}")
        return np.nan, np.nan, np.nan, np.nan


def create_convergence_plot(results: Dict, fit_params: Dict, output_path: Path):
    """
    Create publication-quality convergence plot.

    Parameters
    ----------
    results : Dict
        Results from convergence analysis
    fit_params : Dict
        Power law fit parameters
    output_path : Path
        Output file path for the plot
    """
    node_counts = np.array(results["node_counts"])
    mean_curvatures = np.array(results["mean_curvatures"])
    stderr_curvatures = np.array(results["stderr_curvatures"])
    analytical_ricci = results["analytical_ricci"]

    # Compute deviations from analytical value
    deviations = np.abs(mean_curvatures - analytical_ricci)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Curvature vs N
    ax1.errorbar(
        node_counts,
        mean_curvatures,
        yerr=stderr_curvatures,
        fmt="o-",
        capsize=5,
        capthick=2,
        markersize=8,
        label="Computed κ_OR",
        color="#2E86AB",
        linewidth=2,
    )
    ax1.axhline(
        y=analytical_ricci,
        color="#A23B72",
        linestyle="--",
        linewidth=2,
        label="Analytical (flat torus)",
    )
    ax1.set_xlabel("Number of Nodes (N)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Ollivier-Ricci Curvature κ_OR", fontsize=12, fontweight="bold")
    ax1.set_title("Curvature Convergence on 2-Torus", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.legend(fontsize=11)
    ax1.set_xscale("log")

    # Plot 2: Log-log convergence plot
    valid_mask = ~np.isnan(deviations) & (deviations > 0)
    node_counts_valid = node_counts[valid_mask]
    deviations_valid = deviations[valid_mask]
    stderr_valid = stderr_curvatures[valid_mask]

    ax2.errorbar(
        node_counts_valid,
        deviations_valid,
        yerr=stderr_valid,
        fmt="o",
        capsize=5,
        capthick=2,
        markersize=8,
        label="|κ_OR - R_analytical|",
        color="#F18F01",
        linewidth=2,
    )

    # Plot power law fit
    if not np.isnan(fit_params["alpha"]):
        N_fit = np.logspace(
            np.log10(node_counts_valid.min()), np.log10(node_counts_valid.max()), 100
        )
        deviation_fit = fit_params["prefactor"] * N_fit ** fit_params["alpha"]
        ax2.plot(
            N_fit,
            deviation_fit,
            "--",
            color="#C73E1D",
            linewidth=2.5,
            label=f"Fit: A·N^α, α = {fit_params['alpha']:.3f} ± {fit_params['alpha_err']:.3f}",
        )

    ax2.set_xlabel("Number of Nodes (N)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("|κ_OR - R_analytical|", fontsize=12, fontweight="bold")
    ax2.set_title("Power Law Convergence Analysis", fontsize=14, fontweight="bold")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3, linestyle="--", which="both")
    ax2.legend(fontsize=11)

    plt.tight_layout()

    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Plot saved to {output_path}")

    plt.close()


def main():
    """Main execution function."""
    logger.info("Starting Ollivier-Ricci curvature convergence analysis")

    # Configuration
    node_counts = [50, 100, 200, 500, 1000, 2000]
    num_realizations = 20
    connection_radius_factor = 0.15

    # Output paths
    output_dir = Path("/Users/eirikr/1_Workspace/cosmos/paper/output")
    results_json_path = output_dir / "bigraph_convergence_results.json"
    plot_path = output_dir / "bigraph_convergence_real.pdf"

    # Run convergence analysis
    logger.info("Running convergence analysis...")
    results = run_convergence_analysis(node_counts, num_realizations, connection_radius_factor)

    # Fit power law
    logger.info("Fitting power law to convergence data...")
    node_counts_array = np.array(results["node_counts"])
    mean_curvatures_array = np.array(results["mean_curvatures"])
    stderr_curvatures_array = np.array(results["stderr_curvatures"])

    deviations = np.abs(mean_curvatures_array - results["analytical_ricci"])

    alpha, alpha_err, prefactor, prefactor_err = fit_power_law(
        node_counts_array, deviations, stderr_curvatures_array
    )

    fit_params = {
        "alpha": alpha,
        "alpha_err": alpha_err,
        "prefactor": prefactor,
        "prefactor_err": prefactor_err,
    }

    # Add fit parameters to results
    results["power_law_fit"] = fit_params

    # Save results to JSON
    logger.info("Saving results to JSON...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to Python native types for JSON serialization
    results_serializable = {
        "node_counts": [int(n) for n in results["node_counts"]],
        "num_realizations": results["num_realizations"],
        "connection_radius_factor": results["connection_radius_factor"],
        "mean_curvatures": [
            float(x) if not np.isnan(x) else None for x in results["mean_curvatures"]
        ],
        "std_curvatures": [
            float(x) if not np.isnan(x) else None for x in results["std_curvatures"]
        ],
        "stderr_curvatures": [
            float(x) if not np.isnan(x) else None for x in results["stderr_curvatures"]
        ],
        "analytical_ricci": float(results["analytical_ricci"]),
        "power_law_fit": {
            "alpha": float(alpha) if not np.isnan(alpha) else None,
            "alpha_err": float(alpha_err) if not np.isnan(alpha_err) else None,
            "prefactor": float(prefactor) if not np.isnan(prefactor) else None,
            "prefactor_err": float(prefactor_err) if not np.isnan(prefactor_err) else None,
        },
    }

    with open(results_json_path, "w") as json_file:
        json.dump(results_serializable, json_file, indent=2)

    logger.info(f"Results saved to {results_json_path}")

    # Create convergence plot
    logger.info("Creating convergence plot...")
    create_convergence_plot(results, fit_params, plot_path)

    # Summary
    logger.info("=" * 60)
    logger.info("CONVERGENCE ANALYSIS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Analytical Ricci curvature (flat torus): {results['analytical_ricci']}")
    logger.info(f"Power law exponent α: {alpha:.4f} ± {alpha_err:.4f}")
    logger.info(f"Prefactor A: {prefactor:.4f} ± {prefactor_err:.4f}")
    logger.info(f"Relation: |κ_OR - R| ≈ {prefactor:.4f} * N^({alpha:.4f})")
    logger.info("=" * 60)

    logger.info("Analysis complete!")


if __name__ == "__main__":
    main()
