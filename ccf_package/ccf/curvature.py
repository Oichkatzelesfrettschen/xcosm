"""
Curvature Module
================

Discrete curvature computations for bigraph geometry.
"""

import numpy as np
import networkx as nx
from typing import Tuple, Dict, Optional
from scipy.optimize import linear_sum_assignment

from .bigraph import BigraphState


class OllivierRicciCurvature:
    """
    Computes Ollivier-Ricci curvature on graphs.

    The Ollivier-Ricci curvature measures how much mass transport
    between neighbors differs from the graph distance. It converges
    to Ricci curvature in the continuum limit (van der Hoorn et al. 2023).

    Parameters
    ----------
    idleness : float
        Probability mass remaining at source (default 0.5)
    """

    def __init__(self, idleness: float = 0.5):
        self.idleness = idleness

    def compute_edge_curvature(
        self,
        graph: nx.Graph,
        u: int,
        v: int
    ) -> float:
        """
        Compute Ollivier-Ricci curvature for edge (u, v).

        kappa(u,v) = 1 - W_1(mu_u, mu_v) / d(u,v)

        where W_1 is the Wasserstein-1 distance and mu_u, mu_v
        are probability measures on neighbors.

        Parameters
        ----------
        graph : nx.Graph
            The graph
        u, v : int
            Edge endpoints

        Returns
        -------
        float
            Ollivier-Ricci curvature
        """
        if not graph.has_edge(u, v):
            raise ValueError(f"Edge ({u}, {v}) not in graph")

        mu_u = self._neighbor_measure(graph, u)
        mu_v = self._neighbor_measure(graph, v)

        w1 = self._wasserstein_distance(graph, mu_u, mu_v)

        d_uv = 1.0  # graph distance for adjacent nodes

        return 1.0 - w1 / d_uv

    def _neighbor_measure(
        self,
        graph: nx.Graph,
        node: int
    ) -> Dict[int, float]:
        """Compute probability measure on neighbors."""
        neighbors = list(graph.neighbors(node))

        if not neighbors:
            return {node: 1.0}

        mass_per_neighbor = (1.0 - self.idleness) / len(neighbors)

        measure = {node: self.idleness}
        for n in neighbors:
            measure[n] = mass_per_neighbor

        return measure

    def _wasserstein_distance(
        self,
        graph: nx.Graph,
        mu: Dict[int, float],
        nu: Dict[int, float]
    ) -> float:
        """Compute Wasserstein-1 distance between measures."""
        nodes_mu = list(mu.keys())
        nodes_nu = list(nu.keys())

        n_mu = len(nodes_mu)
        n_nu = len(nodes_nu)

        cost = np.zeros((n_mu, n_nu))
        for i, u in enumerate(nodes_mu):
            for j, v in enumerate(nodes_nu):
                if u == v:
                    cost[i, j] = 0
                elif graph.has_edge(u, v):
                    cost[i, j] = 1
                else:
                    try:
                        cost[i, j] = nx.shortest_path_length(graph, u, v)
                    except nx.NetworkXNoPath:
                        cost[i, j] = float('inf')

        supply = np.array([mu[n] for n in nodes_mu])
        demand = np.array([nu[n] for n in nodes_nu])

        total_distance = self._optimal_transport(cost, supply, demand)

        return total_distance

    def _optimal_transport(
        self,
        cost: np.ndarray,
        supply: np.ndarray,
        demand: np.ndarray
    ) -> float:
        """Solve optimal transport problem."""
        n_supply = len(supply)
        n_demand = len(demand)

        expanded_cost = np.zeros((100, 100))
        expanded_cost[:n_supply, :n_demand] = cost[:n_supply, :n_demand]
        expanded_cost[n_supply:, :] = 1e10
        expanded_cost[:, n_demand:] = 1e10

        row_ind, col_ind = linear_sum_assignment(expanded_cost[:n_supply, :n_demand])

        total = 0.0
        for i, j in zip(row_ind, col_ind):
            transported = min(supply[i], demand[j])
            total += transported * cost[i, j]

        return total

    def compute_all_curvatures(
        self,
        graph: nx.Graph
    ) -> Dict[Tuple[int, int], float]:
        """Compute curvature for all edges."""
        curvatures = {}
        for u, v in graph.edges():
            curvatures[(u, v)] = self.compute_edge_curvature(graph, u, v)
        return curvatures

    def scalar_curvature(self, graph: nx.Graph) -> float:
        """
        Compute total scalar curvature (sum over edges).

        In the continuum limit, this converges to the integrated
        Ricci scalar.
        """
        curvatures = self.compute_all_curvatures(graph)
        return sum(curvatures.values())

    def mean_curvature(self, graph: nx.Graph) -> float:
        """Compute mean curvature over all edges."""
        curvatures = self.compute_all_curvatures(graph)
        if not curvatures:
            return 0.0
        return np.mean(list(curvatures.values()))


def compute_bigraph_curvature(state: BigraphState) -> Dict[str, float]:
    """
    Compute curvature metrics for a bigraph state.

    Parameters
    ----------
    state : BigraphState
        The bigraph state

    Returns
    -------
    Dict[str, float]
        Dictionary with scalar, mean, and variance of curvature
    """
    graph = state.to_networkx()

    if graph.number_of_edges() == 0:
        return {"scalar": 0.0, "mean": 0.0, "variance": 0.0}

    orc = OllivierRicciCurvature()
    curvatures = orc.compute_all_curvatures(graph)

    values = list(curvatures.values())

    return {
        "scalar": sum(values),
        "mean": np.mean(values),
        "variance": np.var(values),
        "min": min(values),
        "max": max(values)
    }
