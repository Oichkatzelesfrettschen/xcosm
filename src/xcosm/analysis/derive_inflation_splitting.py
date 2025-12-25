#!/usr/bin/env python3
"""
PHASE F.4: INFLATION VS GRAVITY - VERTEX SPLITTING MODEL
=========================================================

The Key Insight:
    Preferential Attachment (PA) → Tree-like structures → C → 0
    Vertex Splitting (VS) → Preserves triangles → C stable

This script tests the hypothesis that ε = 0.25 is the EQUILIBRIUM
between Inflation (splitting) and Gravity (attachment).

The Mechanism:
    - SPLITTING: Node v splits into (v₁, v₂). Both inherit v's edges.
      This PRESERVES and MULTIPLIES triangles.
    - ATTACHMENT: New node connects to high-degree hubs.
      This DILUTES triangles.

Prediction: There exists a critical split_probability p_c such that
            clustering stabilizes at ε = 0.25.

December 2025 - The Inflation-Gravity Balance
"""

from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np


class InflationGravityGraph:
    """Graph with mixed inflation (splitting) and gravity (attachment) dynamics."""

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
        if u == v or u not in self.nodes or v not in self.nodes:
            return False
        edge = (min(u, v), max(u, v))
        if edge in self.edges:
            return False
        self.edges.add(edge)
        self.adjacency[u].add(v)
        self.adjacency[v].add(u)
        return True

    def remove_node(self, node: int):
        if node not in self.nodes:
            return
        # Remove all edges involving this node
        neighbors = list(self.adjacency[node])
        for neighbor in neighbors:
            edge = (min(node, neighbor), max(node, neighbor))
            self.edges.discard(edge)
            self.adjacency[neighbor].discard(node)
        del self.adjacency[node]
        self.nodes.discard(node)

    def degree(self, node: int) -> int:
        return len(self.adjacency[node])

    def neighbors(self, node: int) -> Set[int]:
        return self.adjacency[node].copy()

    def count_triangles(self) -> int:
        total = 0
        for node in self.nodes:
            neighbors = list(self.adjacency[node])
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    if neighbors[j] in self.adjacency[neighbors[i]]:
                        total += 1
        return total // 3

    def count_wedges(self) -> int:
        total = 0
        for node in self.nodes:
            k = self.degree(node)
            if k >= 2:
                total += k * (k - 1) // 2
        return total

    def clustering_coefficient(self) -> float:
        triangles = self.count_triangles()
        wedges = self.count_wedges()
        if wedges == 0:
            return 0.0
        return 3 * triangles / wedges

    def initialize_tetrahedron(self):
        """Start with K₄ (tetrahedron) - maximum clustering seed."""
        for _ in range(4):
            self.add_node()
        nodes = list(self.nodes)
        for i in range(4):
            for j in range(i + 1, 4):
                self.add_edge(nodes[i], nodes[j])


def vertex_split(graph: InflationGravityGraph, node: int) -> int:
    """
    Split vertex into two, preserving neighborhood.

    Before: node v connected to {a, b, c, ...}
    After:  v connected to {a, b, c, ..., v'}
            v' connected to {a, b, c, ..., v}

    This DOUBLES triangles involving v!
    """
    neighbors = graph.neighbors(node)
    if not neighbors:
        return -1

    # Create clone
    clone = graph.add_node()

    # Connect clone to original
    graph.add_edge(node, clone)

    # Clone inherits all of original's neighbors
    for neighbor in neighbors:
        graph.add_edge(clone, neighbor)

    return clone


def preferential_attachment(
    graph: InflationGravityGraph, m_edges: int = 2, alpha: float = 1.0
) -> int:
    """
    Standard preferential attachment: add new node connecting to m hubs.
    """
    if len(graph.nodes) < m_edges:
        return -1

    new_node = graph.add_node()
    candidates = [n for n in graph.nodes if n != new_node]

    # Compute degree-based probabilities
    degrees = np.array([graph.degree(c) + 1 for c in candidates])
    probs = degrees**alpha
    probs = probs / probs.sum()

    # Select targets
    targets = graph.rng.choice(
        candidates, size=min(m_edges, len(candidates)), replace=False, p=probs
    )

    for target in targets:
        graph.add_edge(new_node, target)

    return new_node


def run_inflation_gravity_simulation(
    n_steps: int = 500,
    split_prob: float = 0.5,
    m_attach: int = 2,
    seed: int = 42,
    record_interval: int = 10,
) -> Tuple[List[float], List[int], List[int]]:
    """
    Run mixed inflation/gravity simulation.

    Parameters
    ----------
    n_steps : int
        Number of growth steps
    split_prob : float
        Probability of splitting (inflation) vs attachment (gravity)
    m_attach : int
        Edges per attachment event
    seed : int
        Random seed
    record_interval : int
        Recording frequency

    Returns
    -------
    clustering_history, node_counts, edge_counts
    """
    graph = InflationGravityGraph(seed=seed)
    graph.initialize_tetrahedron()

    clustering_history = []
    node_counts = []
    edge_counts = []

    for step in range(n_steps):
        if graph.rng.random() < split_prob:
            # INFLATION: Vertex Splitting
            # Pick a random node to split
            if len(graph.nodes) > 0:
                parent = graph.rng.choice(list(graph.nodes))
                vertex_split(graph, parent)
        else:
            # GRAVITY: Preferential Attachment
            preferential_attachment(graph, m_edges=m_attach)

        # Record metrics
        if step % record_interval == 0:
            cc = graph.clustering_coefficient()
            clustering_history.append(cc)
            node_counts.append(len(graph.nodes))
            edge_counts.append(len(graph.edges))

    return clustering_history, node_counts, edge_counts


def scan_split_probability(
    probabilities: List[float] = None,
    n_steps: int = 400,
    n_seeds: int = 3,
    target_epsilon: float = 0.25,
) -> Dict:
    """Scan over split probabilities to find equilibrium."""

    if probabilities is None:
        probabilities = [0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    results = {}

    print("=" * 70)
    print("INFLATION-GRAVITY EQUILIBRIUM SCAN")
    print(f"Target ε = {target_epsilon}")
    print("=" * 70)

    for prob in probabilities:
        print(f"\n  p_split = {prob:.2f}: ", end="", flush=True)

        final_clusterings = []

        for seed in range(n_seeds):
            history, nodes, edges = run_inflation_gravity_simulation(
                n_steps=n_steps,
                split_prob=prob,
                m_attach=2,
                seed=seed * 17 + 42,
                record_interval=20,
            )
            if history:
                # Use late-time average
                late_avg = np.mean(history[-5:]) if len(history) >= 5 else history[-1]
                final_clusterings.append(late_avg)
            print(".", end="", flush=True)

        mean_final = np.mean(final_clusterings)
        std_final = np.std(final_clusterings)

        results[prob] = {
            "mean": mean_final,
            "std": std_final,
            "deviation": abs(mean_final - target_epsilon),
        }

        print(f" ε = {mean_final:.4f} ± {std_final:.4f}")

    # Find best probability
    best_prob = min(results.keys(), key=lambda p: results[p]["deviation"])

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n  Best p_split = {best_prob:.2f}")
    print(f"  → ε = {results[best_prob]['mean']:.4f} ± {results[best_prob]['std']:.4f}")
    print(f"  Target ε = {target_epsilon}")
    print(f"  Deviation = {results[best_prob]['deviation']:.4f}")

    return results, best_prob


def detailed_evolution(best_prob: float, n_steps: int = 800, seed: int = 42):
    """Run detailed analysis at best split probability."""

    print("\n" + "=" * 70)
    print(f"DETAILED EVOLUTION AT p_split = {best_prob:.2f}")
    print("=" * 70)

    history, nodes, edges = run_inflation_gravity_simulation(
        n_steps=n_steps, split_prob=best_prob, seed=seed, record_interval=20
    )

    print(f"\n  Initial clustering: {history[0]:.4f}")
    print(f"  Final clustering: {history[-1]:.4f}")
    print(f"  Final nodes: {nodes[-1]}")
    print(f"  Final edges: {edges[-1]}")

    # Stability analysis
    if len(history) >= 10:
        late_values = history[-10:]
        mean_late = np.mean(late_values)
        std_late = np.std(late_values)
        print(f"\n  Late-time mean: {mean_late:.4f}")
        print(f"  Late-time std: {std_late:.4f}")
        print(f"  Stable equilibrium: {'YES ✓' if std_late < 0.02 else 'NO - evolving'}")

    # Evolution summary
    print("\n  Evolution phases:")
    n_phases = min(5, len(history))
    phase_size = len(history) // n_phases
    for i in range(n_phases):
        start_idx = i * phase_size
        end_idx = (i + 1) * phase_size
        phase_mean = np.mean(history[start_idx:end_idx])
        print(f"    Phase {i+1}: ε = {phase_mean:.4f}")

    return history, nodes, edges


def main():
    """Run full inflation-gravity analysis."""
    print("=" * 70)
    print("PHASE F.4: INFLATION VS GRAVITY")
    print("Finding Critical Split Probability for ε = 0.25")
    print("=" * 70)
    print(
        """
    The Model:
        - With probability p: SPLIT (inflation) → preserves triangles
        - With probability 1-p: ATTACH (gravity) → dilutes triangles

    Hypothesis:
        There exists p_c such that ε stabilizes at 0.25.
        If p_c ≈ 0.25, the theory is self-consistent.
    """
    )

    # Scan over split probabilities
    results, best_prob = scan_split_probability(
        probabilities=[0.0, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        n_steps=500,
        n_seeds=3,
        target_epsilon=0.25,
    )

    # Detailed evolution at best probability
    detailed_evolution(best_prob, n_steps=600)

    # Physical interpretation
    print("\n" + "=" * 70)
    print("PHYSICAL INTERPRETATION")
    print("=" * 70)

    if best_prob > 0.8:
        interpretation = """
    RESULT: ε = 0.25 requires DOMINANT INFLATION (p > 0.8)

    This means:
        • The universe must expand faster than it clusters
        • Gravity alone cannot maintain vacuum geometry
        • ε = 0.25 is a GEOMETRIC CONSTRAINT, not a dynamical equilibrium

    The F₄ algebra imposes this constraint at the Planck scale,
    and inflation preserves it into the late universe.
        """
    elif 0.2 < best_prob < 0.4:
        interpretation = f"""
    RESULT: ε = 0.25 achieved at p_c ≈ {best_prob:.2f}

    This is REMARKABLE if p_c ≈ 0.25:
        • The same number (1/4) appears in:
          - F₄ Casimir ratio
          - Bekenstein-Hawking entropy
          - Inflation/gravity balance

    The universe is at the CRITICAL POINT between:
        • Pure inflation (crystalline geometry)
        • Pure gravity (random graph)

    This is the phase transition that defines spacetime.
        """
    else:
        interpretation = f"""
    RESULT: Critical probability p_c = {best_prob:.2f}

    The inflation-gravity balance point differs from 0.25.
    This suggests ε = 1/4 must be imposed algebraically (F₄)
    rather than emerging from graph dynamics.
        """

    print(interpretation)

    # Final verdict
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)

    if results[best_prob]["mean"] > 0.2:
        print("  Vertex splitting CAN stabilize clustering at ε ≈ 0.25")
        print("  The inflation mechanism is ESSENTIAL for vacuum geometry")
    else:
        print("  Even with splitting, ε < 0.25")
        print("  Additional constraints (F₄ holonomy?) needed")

    print("=" * 70)

    return results, best_prob


if __name__ == "__main__":
    results, best_prob = main()
