#!/usr/bin/env python3
"""
CCF Enhanced Bigraph Simulation v1.0
=====================================

Extensions to test ChatGPT critique predictions:
1. Spectral dimension estimation via random walks
2. Triangle/motif ratio tracking (does ε = 1/4 emerge?)
3. BAO wave-freeze mechanism
4. Proper place/link graph separation

December 2025 - Addressing theoretical gaps
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict
import json
import math

# Import existing engine components
try:
    from cosmos.engines.ccf_bigraph_engine import (
        CCFParameters, PARAMS, NodeType, LinkType,
        Node, Link, CosmologicalBigraph, CosmologicalBigraphEngine
    )
except ImportError:
    # Fallback definitions if import fails
    @dataclass(frozen=True)
    class CCFParameters:
        lambda_inflation: float = 0.003
        eta_curvature: float = 0.028
        alpha_attachment: float = 0.85
        epsilon_tension: float = 0.25
        k_star: float = 0.01
    PARAMS = CCFParameters()


# =============================================================================
# SPECTRAL DIMENSION ANALYSIS
# =============================================================================

def estimate_spectral_dimension(
    adjacency: Dict[int, Set[int]],
    t_max: int = 10,
    n_walks: int = 5000,
    seed: int = 42
) -> Dict:
    """
    Estimate spectral dimension d_s from random walk return probabilities.

    P_return(t) ~ t^{-d_s/2}

    For 4D manifold-like graphs: d_s → 4
    For scale-free networks: d_s may be fractional

    Args:
        adjacency: Dict mapping node_id -> set of neighbor node_ids
        t_max: Maximum walk length
        n_walks: Number of random walks per starting point
        seed: Random seed

    Returns:
        Dict with return probabilities P(t), estimated d_s, and fit quality
    """
    rng = np.random.default_rng(seed)
    nodes = list(adjacency.keys())

    if len(nodes) < 10:
        return {"P": [], "d_s": None, "r_squared": 0.0, "error": "Too few nodes"}

    returns = [0] * (t_max + 1)
    total_walks = 0

    for _ in range(n_walks):
        start = rng.choice(nodes)
        pos = start

        for t in range(1, t_max + 1):
            neighbors = list(adjacency.get(pos, set()))
            if not neighbors:
                break
            pos = rng.choice(neighbors)
            if pos == start:
                returns[t] += 1

        total_walks += 1

    # Compute return probabilities
    P = [r / total_walks for r in returns]

    # Fit log P(t) = a + b log t; then d_s = -2b
    xs, ys = [], []
    for t in range(2, t_max + 1):  # Start from t=2 to avoid t=1 edge effects
        if P[t] > 0:
            xs.append(math.log(t))
            ys.append(math.log(P[t]))

    if len(xs) < 3:
        return {"P": P, "d_s": None, "r_squared": 0.0, "error": "Insufficient data"}

    # Linear regression
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    ss_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    ss_xx = sum((x - mean_x) ** 2 for x in xs)
    ss_yy = sum((y - mean_y) ** 2 for y in ys)

    if ss_xx == 0:
        return {"P": P, "d_s": None, "r_squared": 0.0, "error": "Zero variance"}

    slope = ss_xy / ss_xx
    intercept = mean_y - slope * mean_x

    # R-squared
    r_squared = (ss_xy ** 2) / (ss_xx * ss_yy) if ss_yy > 0 else 0.0

    # Spectral dimension
    d_s = -2 * slope

    return {
        "P": P,
        "d_s": d_s,
        "r_squared": r_squared,
        "slope": slope,
        "intercept": intercept,
        "t_values": list(range(t_max + 1)),
    }


# =============================================================================
# MOTIF ANALYSIS (Testing ε emergence)
# =============================================================================

def count_triangles(adjacency: Dict[int, Set[int]]) -> int:
    """
    Count total triangles in the graph.
    A triangle exists when nodes (u, v, w) are mutually connected.
    """
    triangles = 0
    nodes = list(adjacency.keys())

    for u in nodes:
        neighbors_u = adjacency.get(u, set())
        for v in neighbors_u:
            if v > u:  # Avoid double counting
                neighbors_v = adjacency.get(v, set())
                # Common neighbors form triangles
                common = neighbors_u & neighbors_v
                triangles += len([w for w in common if w > v])

    return triangles


def count_open_triplets(adjacency: Dict[int, Set[int]]) -> int:
    """
    Count open triplets (wedges): paths u-v-w where u and w are NOT connected.
    """
    triplets = 0

    for v, neighbors in adjacency.items():
        n_neighbors = len(neighbors)
        if n_neighbors >= 2:
            # Number of pairs of neighbors
            pairs = n_neighbors * (n_neighbors - 1) // 2
            triplets += pairs

    return triplets


def compute_motif_ratios(adjacency: Dict[int, Set[int]]) -> Dict:
    """
    Compute motif ratios that might converge to ε = 1/4.

    Candidate ratios:
    1. Triangles / (Triangles + OpenTriplets) - clustering coefficient variant
    2. Triangles / Nodes
    3. Triangles / Edges
    """
    n_nodes = len(adjacency)
    n_edges = sum(len(neighbors) for neighbors in adjacency.values()) // 2
    n_triangles = count_triangles(adjacency)
    n_triplets = count_open_triplets(adjacency)

    # Candidate ε ratios
    ratio_clustering = n_triangles / (n_triangles + n_triplets) if (n_triangles + n_triplets) > 0 else 0.0
    ratio_per_node = n_triangles / n_nodes if n_nodes > 0 else 0.0
    ratio_per_edge = n_triangles / n_edges if n_edges > 0 else 0.0

    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "n_triangles": n_triangles,
        "n_open_triplets": n_triplets,
        "clustering_ratio": ratio_clustering,  # Does this → 1/4?
        "triangles_per_node": ratio_per_node,
        "triangles_per_edge": ratio_per_edge,
    }


# =============================================================================
# BAO WAVE SIMULATION
# =============================================================================

@dataclass
class BAOWaveState:
    """Track BAO-like wave propagation through the graph."""
    seed_node: int
    current_shell: Set[int]
    shell_history: List[Set[int]] = field(default_factory=list)
    step: int = 0
    frozen: bool = False


def simulate_bao_wave(
    adjacency: Dict[int, Set[int]],
    seed_node: int,
    freeze_after: int = 10,
    n_shells: int = 20
) -> Dict:
    """
    Simulate BAO-like wave propagation.

    A wave starts at seed_node and propagates outward.
    At each step, it marks nodes at distance d from seed.
    After freeze_after steps, propagation stops (decoupling).

    Returns:
        Dict with shell sizes at each distance and frozen distance
    """
    if seed_node not in adjacency:
        return {"error": "Seed node not in graph"}

    visited = {seed_node}
    current_shell = {seed_node}
    shell_sizes = [1]  # Shell at d=0

    for d in range(1, n_shells + 1):
        # Find next shell: neighbors of current shell not yet visited
        next_shell = set()
        for node in current_shell:
            for neighbor in adjacency.get(node, set()):
                if neighbor not in visited:
                    next_shell.add(neighbor)

        if not next_shell:
            break

        visited.update(next_shell)
        current_shell = next_shell
        shell_sizes.append(len(next_shell))

        # Check for freeze (decoupling)
        if d == freeze_after:
            break

    # Compute BAO feature: excess at freeze distance
    frozen_distance = min(freeze_after, len(shell_sizes) - 1)

    # In a proper BAO, there should be a bump at the freeze distance
    # Compute ratio to expected (geometric growth)
    if frozen_distance > 2:
        expected_shell = shell_sizes[frozen_distance - 1] * 1.2  # Rough estimate
        actual_shell = shell_sizes[frozen_distance]
        bao_excess = actual_shell / expected_shell if expected_shell > 0 else 0.0
    else:
        bao_excess = 1.0

    return {
        "shell_sizes": shell_sizes,
        "frozen_distance": frozen_distance,
        "bao_excess": bao_excess,
        "total_reached": len(visited),
    }


# =============================================================================
# ENHANCED BIGRAPH WITH PLACE/LINK SEPARATION
# =============================================================================

class DualBigraph:
    """
    Proper bigraph with separate place and link graphs.

    Place graph: Geometric/classical structure (tree/forest)
    Link graph: Entanglement/connectivity (hypergraph)
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

        # Node data
        self.nodes: Set[int] = set()
        self.node_types: Dict[int, str] = {}  # F_4 control labels
        self.next_node_id = 0

        # Place graph (tree structure)
        self.place_parent: Dict[int, Optional[int]] = {}
        self.place_children: Dict[int, Set[int]] = defaultdict(set)

        # Link graph (general connectivity)
        self.link_neighbors: Dict[int, Set[int]] = defaultdict(set)

        # Metrics
        self.step = 0
        self.history = []

    def add_node(self, node_type: str = "vacuum", parent: Optional[int] = None) -> int:
        """Add a node with optional place-graph parent."""
        node_id = self.next_node_id
        self.next_node_id += 1

        self.nodes.add(node_id)
        self.node_types[node_id] = node_type

        # Place graph
        self.place_parent[node_id] = parent
        if parent is not None:
            self.place_children[parent].add(node_id)

        return node_id

    def add_link(self, u: int, v: int):
        """Add an edge to the link graph (entanglement)."""
        if u in self.nodes and v in self.nodes:
            self.link_neighbors[u].add(v)
            self.link_neighbors[v].add(u)

    def place_distance(self, u: int, v: int) -> int:
        """Distance in place graph (tree distance)."""
        # Find path to root for both nodes
        def path_to_root(n):
            path = []
            current = n
            while current is not None:
                path.append(current)
                current = self.place_parent.get(current)
            return path

        path_u = path_to_root(u)
        path_v = path_to_root(v)

        # Find LCA
        set_u = set(path_u)
        for i, node in enumerate(path_v):
            if node in set_u:
                j = path_u.index(node)
                return i + j

        return len(path_u) + len(path_v)  # Different trees

    def link_distance(self, u: int, v: int, max_depth: int = 20) -> int:
        """Distance in link graph (BFS)."""
        if u == v:
            return 0

        visited = {u}
        frontier = {u}

        for d in range(1, max_depth + 1):
            next_frontier = set()
            for node in frontier:
                for neighbor in self.link_neighbors.get(node, set()):
                    if neighbor == v:
                        return d
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.add(neighbor)
            frontier = next_frontier
            if not frontier:
                break

        return max_depth + 1  # Not reachable

    def compute_twist(self) -> float:
        """
        Compute "twist" parameter τ between place and link graphs.

        τ measures how much link connectivity differs from place adjacency.
        High τ → torsion-like effects (Immirzi-like)
        Low τ → close to Riemannian geometry
        """
        if len(self.nodes) < 2:
            return 0.0

        # Sample node pairs and compare distances
        nodes_list = list(self.nodes)
        n_samples = min(1000, len(nodes_list) * (len(nodes_list) - 1) // 2)

        twist_sum = 0.0
        count = 0

        for _ in range(n_samples):
            u, v = self.rng.choice(nodes_list, size=2, replace=False)
            d_place = self.place_distance(u, v)
            d_link = self.link_distance(u, v)

            if d_place > 0 and d_link < 20:
                twist_sum += abs(d_place - d_link) / d_place
                count += 1

        return twist_sum / count if count > 0 else 0.0

    def get_link_adjacency(self) -> Dict[int, Set[int]]:
        """Return link graph adjacency for analysis."""
        return dict(self.link_neighbors)

    def record_state(self):
        """Record current state for history."""
        motifs = compute_motif_ratios(self.get_link_adjacency())
        spectral = estimate_spectral_dimension(self.get_link_adjacency())

        self.history.append({
            "step": self.step,
            "n_nodes": len(self.nodes),
            "n_link_edges": sum(len(n) for n in self.link_neighbors.values()) // 2,
            "d_s": spectral.get("d_s"),
            "clustering_ratio": motifs["clustering_ratio"],
            "triangles_per_node": motifs["triangles_per_node"],
            "twist": self.compute_twist(),
        })
        self.step += 1


# =============================================================================
# ENHANCED SIMULATION ENGINE
# =============================================================================

class EnhancedCosmologySimulator:
    """
    Enhanced cosmology simulator testing ChatGPT critique predictions.
    """

    def __init__(self, seed: int = 42):
        self.bigraph = DualBigraph(seed)
        self.rng = np.random.default_rng(seed)
        self.params = PARAMS

    def initialize(self, n_initial: int = 5):
        """Create initial vacuum nodes."""
        root = self.bigraph.add_node("vacuum", parent=None)

        for i in range(n_initial - 1):
            node = self.bigraph.add_node("vacuum", parent=root)
            # Initial links within vacuum
            if self.rng.random() < 0.5:
                self.bigraph.add_link(root, node)

    def run_inflation(self, n_steps: int = 50, p_inflate: float = 0.5):
        """
        Inflation phase: exponential node growth.

        Each node may spawn a child with probability p_inflate.
        Probability decays with step (graceful exit).
        """
        print(f"[INFLATION] Running {n_steps} steps...")

        for step in range(n_steps):
            # Decay probability (graceful exit)
            p_current = p_inflate * np.exp(-0.02 * step)

            # Snapshot current nodes
            current_nodes = list(self.bigraph.nodes)

            for node_id in current_nodes:
                if self.bigraph.node_types.get(node_id) != "vacuum":
                    continue

                if self.rng.random() < p_current:
                    # Spawn child
                    child = self.bigraph.add_node("vacuum", parent=node_id)

                    # Link to parent (spatial adjacency → link graph)
                    self.bigraph.add_link(node_id, child)

                    # Preferential attachment to high-degree nodes
                    for other in list(self.bigraph.nodes):
                        if other != child and other != node_id:
                            degree = len(self.bigraph.link_neighbors.get(other, set()))
                            p_attach = 0.001 * (degree ** self.params.alpha_attachment)
                            if self.rng.random() < p_attach:
                                self.bigraph.add_link(child, other)

            self.bigraph.record_state()

        print(f"    Created {len(self.bigraph.nodes)} nodes")

    def run_reheating(self):
        """Convert vacuum nodes to matter/dark matter."""
        print("[REHEATING]...")

        for node_id in list(self.bigraph.nodes):
            if self.bigraph.node_types.get(node_id) == "vacuum":
                roll = self.rng.random()
                if roll < 0.30:
                    self.bigraph.node_types[node_id] = "matter"
                elif roll < 0.55:
                    self.bigraph.node_types[node_id] = "dark_matter"
                else:
                    self.bigraph.node_types[node_id] = "dark_energy"

        self.bigraph.record_state()

    def run_structure_formation(self, n_steps: int = 100):
        """
        Structure formation via preferential attachment.

        Matter/dark matter nodes attract links preferentially.
        """
        print(f"[STRUCTURE] Running {n_steps} steps...")

        for step in range(n_steps):
            nodes = list(self.bigraph.nodes)

            # Try to add links via preferential attachment
            for _ in range(len(nodes) // 10):
                if len(nodes) < 2:
                    break

                u, v = self.rng.choice(nodes, size=2, replace=False)

                if v in self.bigraph.link_neighbors.get(u, set()):
                    continue  # Already connected

                # Preferential attachment
                deg_u = len(self.bigraph.link_neighbors.get(u, set()))
                deg_v = len(self.bigraph.link_neighbors.get(v, set()))
                weight = (deg_u + 1) ** 0.85 + (deg_v + 1) ** 0.85

                if self.rng.random() < weight / 100:
                    self.bigraph.add_link(u, v)

            if step % 10 == 0:
                self.bigraph.record_state()

    def run_full_simulation(
        self,
        inflation_steps: int = 50,
        structure_steps: int = 100
    ) -> Dict:
        """Run complete cosmological evolution."""
        print("=" * 60)
        print("ENHANCED CCF BIGRAPH SIMULATION")
        print("Testing ChatGPT critique predictions")
        print("=" * 60)

        self.initialize(n_initial=5)
        self.run_inflation(inflation_steps)
        self.run_reheating()
        self.run_structure_formation(structure_steps)

        # Final analysis
        print("\n" + "=" * 60)
        print("FINAL ANALYSIS")
        print("=" * 60)

        adj = self.bigraph.get_link_adjacency()

        # Spectral dimension
        spectral = estimate_spectral_dimension(adj, t_max=12, n_walks=10000)
        print(f"\n[SPECTRAL DIMENSION]")
        print(f"  d_s = {spectral['d_s']:.2f}" if spectral['d_s'] else "  d_s = N/A")
        print(f"  R² = {spectral['r_squared']:.3f}")
        print(f"  Target: d_s → 4 for emergent 4D spacetime")

        # Motif ratios (ε emergence test)
        motifs = compute_motif_ratios(adj)
        print(f"\n[MOTIF RATIOS - Testing ε = 1/4 emergence]")
        print(f"  Triangles: {motifs['n_triangles']}")
        print(f"  Open triplets: {motifs['n_open_triplets']}")
        print(f"  Clustering ratio: {motifs['clustering_ratio']:.4f}")
        print(f"  Target: clustering_ratio → 0.25?")

        # Twist (γ-like parameter)
        twist = self.bigraph.compute_twist()
        print(f"\n[PLACE/LINK TWIST - γ analog]")
        print(f"  τ = {twist:.4f}")
        print(f"  Interpretation: High τ → torsion/Immirzi effects")

        # BAO wave test
        if len(self.bigraph.nodes) > 10:
            seed_node = list(self.bigraph.nodes)[0]
            bao = simulate_bao_wave(adj, seed_node, freeze_after=8)
            print(f"\n[BAO WAVE TEST]")
            print(f"  Shell sizes: {bao['shell_sizes'][:10]}...")
            print(f"  Frozen distance: {bao['frozen_distance']}")
            print(f"  BAO excess: {bao['bao_excess']:.2f}")

        # Summary
        return {
            "n_nodes": len(self.bigraph.nodes),
            "spectral_dimension": spectral.get("d_s"),
            "clustering_ratio": motifs["clustering_ratio"],
            "twist": twist,
            "history": self.bigraph.history,
        }


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run enhanced simulation."""
    sim = EnhancedCosmologySimulator(seed=42)

    results = sim.run_full_simulation(
        inflation_steps=60,
        structure_steps=120
    )

    # Check if ε = 1/4 emerges
    print("\n" + "=" * 60)
    print("ε EMERGENCE TEST")
    print("=" * 60)

    cr = results["clustering_ratio"]
    target_eps = 0.25
    deviation = abs(cr - target_eps) / target_eps * 100

    print(f"  Measured clustering ratio: {cr:.4f}")
    print(f"  Target ε = 1/4 = 0.25")
    print(f"  Deviation: {deviation:.1f}%")

    if deviation < 20:
        print("  Status: ENCOURAGING - within 20% of target")
    elif deviation < 50:
        print("  Status: SUGGESTIVE - within 50% of target")
    else:
        print("  Status: NOT CONVERGED - requires parameter tuning")

    # Check spectral dimension
    ds = results["spectral_dimension"]
    if ds:
        print(f"\n  Spectral dimension: {ds:.2f}")
        if 3.5 < ds < 4.5:
            print("  Status: 4D-LIKE")
        else:
            print(f"  Status: Non-4D (d_s = {ds:.2f})")

    # Save results
    output_file = "data/processed/ccf_enhanced_results.json"
    with open(output_file, "w") as f:
        # Convert numpy types for JSON
        serializable = {
            k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
            for k, v in results.items()
            if k != "history"
        }
        serializable["history_length"] = len(results["history"])
        json.dump(serializable, f, indent=2)

    print(f"\nResults saved to {output_file}")

    return sim, results


if __name__ == "__main__":
    sim, results = main()
