#!/usr/bin/env python3
"""
PHASE F.3: TRIALITY EQUILIBRIUM
================================

Testing the competition between Gravity (Preferential Attachment)
and Gauge Forces (Triadic Closure) to stabilize epsilon = 0.25.

Key Hypothesis:
    P(Link A → B) ∝ k_B^α × (1 - λ) + Δ_AB × λ

Where:
    k_B = degree of target (Gravity)
    Δ_AB = triangles formed by linking A-B (Gauge/Triadic Closure)
    λ = Triality weight (the coupling we're scanning)

Goal: Find λ_c such that clustering stabilizes at ε = 0.25

December 2025 - Critical Coupling Determination
"""

import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Set
import sys


class TrialityEquilibriumGraph:
    """
    Graph with active triadic closure dynamics.

    Unlike passive preferential attachment, this model includes
    a gauge force that prefers completing triangles.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.nodes: Set[int] = set()
        self.edges: Set[Tuple[int, int]] = set()
        self.adjacency: Dict[int, Set[int]] = defaultdict(set)
        self.next_id = 0

    def add_node(self) -> int:
        node_id = self.next_id
        self.nodes.add(node_id)
        self.next_id += 1
        return node_id

    def add_edge(self, u: int, v: int) -> bool:
        if u == v or (min(u,v), max(u,v)) in self.edges:
            return False
        edge = (min(u, v), max(u, v))
        self.edges.add(edge)
        self.adjacency[u].add(v)
        self.adjacency[v].add(u)
        return True

    def degree(self, node: int) -> int:
        return len(self.adjacency[node])

    def neighbors(self, node: int) -> Set[int]:
        return self.adjacency[node]

    def triangles_formed_by_edge(self, u: int, v: int) -> int:
        """Count how many triangles would be formed by adding edge u-v."""
        if u == v:
            return 0
        common = self.adjacency[u] & self.adjacency[v]
        return len(common)

    def count_triangles(self) -> int:
        """Count total triangles in graph."""
        total = 0
        for node in self.nodes:
            neighbors = list(self.adjacency[node])
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    if neighbors[j] in self.adjacency[neighbors[i]]:
                        total += 1
        return total // 3  # Each triangle counted 3 times

    def count_wedges(self) -> int:
        """Count total wedges (open triplets)."""
        total = 0
        for node in self.nodes:
            k = self.degree(node)
            if k >= 2:
                total += k * (k - 1) // 2
        return total

    def clustering_coefficient(self) -> float:
        """Global clustering coefficient = 3 * triangles / wedges."""
        triangles = self.count_triangles()
        wedges = self.count_wedges()
        if wedges == 0:
            return 0.0
        return 3 * triangles / wedges

    def initialize_seed(self, n_seed: int = 4):
        """Initialize with a complete graph (maximum clustering)."""
        for _ in range(n_seed):
            self.add_node()
        # Complete graph
        nodes = list(self.nodes)
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                self.add_edge(nodes[i], nodes[j])


def run_triality_equilibrium(
    n_final: int = 1000,
    m_edges: int = 3,
    triality_weight: float = 0.0,
    alpha: float = 1.0,
    seed: int = 42,
    record_interval: int = 50
) -> Tuple[List[float], List[int]]:
    """
    Run graph growth with mixed Gravity + Gauge dynamics.

    Parameters
    ----------
    n_final : int
        Final number of nodes
    m_edges : int
        Edges added per new node
    triality_weight : float
        λ in [0, 1]. 0 = pure gravity, 1 = pure triadic closure
    alpha : float
        Preferential attachment exponent
    seed : int
        Random seed
    record_interval : int
        Record clustering every N steps

    Returns
    -------
    clustering_history : List[float]
        Clustering coefficient over time
    node_counts : List[int]
        Node counts at each recording
    """
    graph = TrialityEquilibriumGraph(seed=seed)
    graph.initialize_seed(n_seed=4)

    clustering_history = []
    node_counts = []

    for step in range(4, n_final):
        new_node = graph.add_node()

        # Get all existing nodes (excluding new one)
        candidates = [n for n in graph.nodes if n != new_node]

        if len(candidates) < m_edges:
            # Not enough nodes yet, connect to all
            for c in candidates:
                graph.add_edge(new_node, c)
            continue

        # Compute attachment scores for each candidate
        scores = np.zeros(len(candidates))

        for i, candidate in enumerate(candidates):
            # A. Gravity score (preferential attachment)
            deg = graph.degree(candidate)
            gravity_score = (deg + 1) ** alpha  # +1 to avoid zero

            # B. Gauge score (triadic closure potential)
            # How many triangles would form if we connect new_node to candidate?
            # This requires knowing what other edges new_node will have.
            # Approximate: count how many of candidate's neighbors are already
            # connected to new_node (but new_node has no edges yet in this step).
            # Better: count triangles that would form with previously selected targets.
            gauge_score = 1.0  # Base score

            # Enhance: candidate is highly clustered = good gauge target
            local_triangles = 0
            neighbors = list(graph.neighbors(candidate))
            for j in range(len(neighbors)):
                for k in range(j + 1, len(neighbors)):
                    if neighbors[k] in graph.adjacency[neighbors[j]]:
                        local_triangles += 1
            gauge_score = local_triangles + 1  # +1 base

            # Combined score
            scores[i] = (1 - triality_weight) * gravity_score + triality_weight * gauge_score

        # Normalize to probabilities
        total_score = scores.sum()
        if total_score == 0:
            probs = np.ones(len(candidates)) / len(candidates)
        else:
            probs = scores / total_score

        # Select m_edges targets
        targets = graph.rng.choice(
            candidates,
            size=min(m_edges, len(candidates)),
            replace=False,
            p=probs
        )

        for target in targets:
            graph.add_edge(new_node, target)

        # Record clustering periodically
        if step % record_interval == 0:
            cc = graph.clustering_coefficient()
            clustering_history.append(cc)
            node_counts.append(len(graph.nodes))

    return clustering_history, node_counts


def scan_triality_coupling(
    weights: List[float] = None,
    n_final: int = 2000,
    m_edges: int = 3,
    n_seeds: int = 3,
    target_epsilon: float = 0.25
) -> Dict:
    """
    Scan over triality weights to find critical coupling.

    Returns dict with results for each weight.
    """
    if weights is None:
        weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    results = {}

    print("=" * 70)
    print("TRIALITY EQUILIBRIUM SCAN")
    print(f"Target ε = {target_epsilon}")
    print("=" * 70)

    for weight in weights:
        print(f"\n  λ = {weight:.2f}: ", end="", flush=True)

        final_clusterings = []

        for seed in range(n_seeds):
            history, nodes = run_triality_equilibrium(
                n_final=n_final,
                m_edges=m_edges,
                triality_weight=weight,
                seed=seed * 17 + 42
            )
            if history:
                final_clusterings.append(history[-1])
            print(".", end="", flush=True)

        mean_final = np.mean(final_clusterings)
        std_final = np.std(final_clusterings)

        results[weight] = {
            'mean': mean_final,
            'std': std_final,
            'deviation_from_target': abs(mean_final - target_epsilon)
        }

        print(f" ε = {mean_final:.4f} ± {std_final:.4f}")

    # Find best weight
    best_weight = min(results.keys(), key=lambda w: results[w]['deviation_from_target'])

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n  Best λ = {best_weight:.2f} → ε = {results[best_weight]['mean']:.4f}")
    print(f"  Target ε = {target_epsilon}")
    print(f"  Deviation = {results[best_weight]['deviation_from_target']:.4f}")

    return results, best_weight


def detailed_analysis(best_weight: float, n_final: int = 3000, seed: int = 42):
    """Run detailed analysis at best weight."""
    print("\n" + "=" * 70)
    print(f"DETAILED ANALYSIS AT λ = {best_weight:.2f}")
    print("=" * 70)

    history, nodes = run_triality_equilibrium(
        n_final=n_final,
        m_edges=3,
        triality_weight=best_weight,
        seed=seed,
        record_interval=100
    )

    print(f"\n  Initial clustering: {history[0]:.4f}")
    print(f"  Final clustering: {history[-1]:.4f}")
    print(f"  Final nodes: {nodes[-1]}")

    # Check for equilibrium (is it stable?)
    late_values = history[-10:]
    mean_late = np.mean(late_values)
    std_late = np.std(late_values)

    print(f"\n  Late-time mean: {mean_late:.4f}")
    print(f"  Late-time std: {std_late:.4f}")

    is_stable = std_late < 0.01
    print(f"  Equilibrium stable: {'YES ✓' if is_stable else 'NO - still evolving'}")

    return history, nodes


def main():
    """Run full triality equilibrium analysis."""
    print("=" * 70)
    print("PHASE F.3: TRIALITY EQUILIBRIUM ANALYSIS")
    print("Finding Critical Gauge Coupling λ_c for ε = 0.25")
    print("=" * 70)

    # Scan over weights
    results, best_weight = scan_triality_coupling(
        weights=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        n_final=1500,
        m_edges=3,
        n_seeds=3,
        target_epsilon=0.25
    )

    # Detailed analysis at best weight
    detailed_analysis(best_weight, n_final=2000)

    # Physical interpretation
    print("\n" + "=" * 70)
    print("PHYSICAL INTERPRETATION")
    print("=" * 70)
    print(f"""
    The critical coupling λ_c = {best_weight:.2f} represents:

    1. GRAVITY vs GAUGE BALANCE:
       - (1 - λ) = {1 - best_weight:.2f} → Preferential Attachment weight
       - λ = {best_weight:.2f} → Triadic Closure weight

    2. COSMOLOGICAL MEANING:
       If λ_c ≈ 0.25, then the vacuum geometry is stabilized when:

           Gauge Force / Total Force = ε = 1/4

       This would be a remarkable prediction: the same ε = 1/4 that
       appears in F₄ Casimirs also determines the dynamical equilibrium.

    3. RELATION TO w₀:
       w₀ = -1 + 2ε/3 = -1 + 2λ_c/3

       If λ_c ≈ 0.25: w₀ ≈ -0.833 (matches DESI!)
       If λ_c ≈ 0.50: w₀ ≈ -0.667
       If λ_c ≈ 0.75: w₀ ≈ -0.500

    {"=" * 66}
    """)

    # Verdict
    print("VERDICT:")
    if abs(best_weight - 0.25) < 0.1:
        print("  ε = 1/4 is BOTH algebraic (F₄) AND dynamical (equilibrium)!")
        print("  The theory is self-consistent.")
    elif best_weight < 0.2:
        print("  Gravity dominates. ε = 1/4 must be imposed, not emergent.")
    else:
        print(f"  Critical coupling λ_c = {best_weight:.2f} differs from 1/4.")
        print("  Further investigation needed.")

    print("=" * 70)

    return results, best_weight


if __name__ == "__main__":
    results, best_weight = main()
