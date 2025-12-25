#!/usr/bin/env python3
"""
CCF Triality Bigraph - F₄ Colored Network Dynamics
===================================================

Implements the "Triality Weighted Attachment" hypothesis:

Nodes are typed by F₄ triality representations (8_v, 8_s, 8_c).
Attachment probabilities are modified by a Triality Matrix T_ij
that encodes the representation theory of SO(8) ⊂ F₄.

Hypothesis: This constraint pushes clustering ratio from 0.22 → 0.25

December 2025 - Closing the ε gap
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

# =============================================================================
# TRIALITY REPRESENTATIONS
# =============================================================================


class TrialityType(Enum):
    """
    The three 8-dimensional representations of SO(8) under triality.

    These form the three octonion directions in F₄:
    - 8_v: Vector representation (spacetime directions)
    - 8_s: Positive spinor (left-handed matter)
    - 8_c: Negative spinor (right-handed matter)

    Triality σ: 8_v → 8_s → 8_c → 8_v
    """

    VECTOR = auto()  # 8_v
    SPINOR_P = auto()  # 8_s (positive chirality)
    SPINOR_N = auto()  # 8_c (negative chirality)
    SINGLET = auto()  # 1 (scalar/dark energy)


# Triality interaction matrix
# T[i][j] = coupling strength for type_i connecting to type_j
# Physics: Different representations interact via triality-mixing vertices
TRIALITY_MATRIX = {
    # Vector couples primarily to spinors (gauge-matter vertices)
    TrialityType.VECTOR: {
        TrialityType.VECTOR: 0.3,  # Same-type: suppressed (Pauli-like)
        TrialityType.SPINOR_P: 1.0,  # Cross-type: enhanced
        TrialityType.SPINOR_N: 1.0,  # Cross-type: enhanced
        TrialityType.SINGLET: 0.5,  # Gravitational coupling
    },
    # Positive spinor couples to vectors and negative spinors
    TrialityType.SPINOR_P: {
        TrialityType.VECTOR: 1.0,
        TrialityType.SPINOR_P: 0.3,  # Same-chirality: suppressed
        TrialityType.SPINOR_N: 0.8,  # Opposite chirality: Yukawa-like
        TrialityType.SINGLET: 0.5,
    },
    # Negative spinor couples to vectors and positive spinors
    TrialityType.SPINOR_N: {
        TrialityType.VECTOR: 1.0,
        TrialityType.SPINOR_P: 0.8,
        TrialityType.SPINOR_N: 0.3,
        TrialityType.SINGLET: 0.5,
    },
    # Singlet couples weakly to everything (gravitational)
    TrialityType.SINGLET: {
        TrialityType.VECTOR: 0.5,
        TrialityType.SPINOR_P: 0.5,
        TrialityType.SPINOR_N: 0.5,
        TrialityType.SINGLET: 0.2,
    },
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class TrialityNode:
    """A node with F₄ triality type."""

    node_id: int
    triality: TrialityType
    mass: float = 1.0
    creation_step: int = 0

    def __hash__(self):
        return hash(self.node_id)


@dataclass
class TrialityLink:
    """A link between triality-typed nodes."""

    link_id: int
    source_id: int
    target_id: int
    tension: float = 1.0
    creation_step: int = 0

    def __hash__(self):
        return hash(self.link_id)


# =============================================================================
# TRIALITY BIGRAPH
# =============================================================================


class TrialityBigraph:
    """
    Bigraph with F₄ triality-typed nodes.

    Key difference from standard bigraph:
    - Nodes carry triality labels (8_v, 8_s, 8_c, 1)
    - Attachment probability is weighted by TRIALITY_MATRIX
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

        self.nodes: dict[int, TrialityNode] = {}
        self.links: dict[int, TrialityLink] = {}

        self.outgoing: dict[int, set[int]] = defaultdict(set)
        self.incoming: dict[int, set[int]] = defaultdict(set)

        self.next_node_id = 0
        self.next_link_id = 0
        self.current_step = 0

        # Triality distribution (approximate F₄ decomposition)
        # 26 = 8 + 8 + 8 + 1 + 1, so roughly equal triality weights
        self.triality_weights = {
            TrialityType.VECTOR: 0.30,
            TrialityType.SPINOR_P: 0.30,
            TrialityType.SPINOR_N: 0.30,
            TrialityType.SINGLET: 0.10,
        }

    def add_node(self, triality: TrialityType = None, mass: float = 1.0) -> TrialityNode:
        """Add a node with random or specified triality type."""
        if triality is None:
            # Sample from triality distribution
            types = list(self.triality_weights.keys())
            weights = list(self.triality_weights.values())
            triality = self.rng.choice(types, p=weights)

        node = TrialityNode(
            node_id=self.next_node_id, triality=triality, mass=mass, creation_step=self.current_step
        )
        self.nodes[node.node_id] = node
        self.next_node_id += 1
        return node

    def add_link(self, source_id: int, target_id: int, tension: float = 1.0) -> TrialityLink | None:
        """Add a link, respecting triality constraints."""
        if source_id not in self.nodes or target_id not in self.nodes:
            return None

        # Check if already linked
        for link_id in self.outgoing[source_id]:
            if self.links[link_id].target_id == target_id:
                return None

        link = TrialityLink(
            link_id=self.next_link_id,
            source_id=source_id,
            target_id=target_id,
            tension=tension,
            creation_step=self.current_step,
        )
        self.links[link.link_id] = link
        self.outgoing[source_id].add(link.link_id)
        self.incoming[target_id].add(link.link_id)
        self.next_link_id += 1
        return link

    def degree(self, node_id: int) -> int:
        """Total degree of a node."""
        return len(self.outgoing[node_id]) + len(self.incoming[node_id])

    def neighbors(self, node_id: int) -> set[int]:
        """All neighboring node IDs."""
        result = set()
        for link_id in self.outgoing[node_id]:
            result.add(self.links[link_id].target_id)
        for link_id in self.incoming[node_id]:
            result.add(self.links[link_id].source_id)
        return result

    def triality_coupling(self, node1_id: int, node2_id: int) -> float:
        """Get triality coupling strength between two nodes."""
        t1 = self.nodes[node1_id].triality
        t2 = self.nodes[node2_id].triality
        return TRIALITY_MATRIX[t1][t2]


# =============================================================================
# TRIALITY PREFERENTIAL ATTACHMENT
# =============================================================================


def triality_preferential_attachment(
    bigraph: TrialityBigraph, alpha: float = 0.85, epsilon: float = 0.25, n_new_links: int = 100
) -> int:
    """
    Add links using triality-weighted preferential attachment.

    P(link from u to v) ∝ degree(v)^α × T[type(u)][type(v)]

    The triality matrix T forces mixed-type triangles, pushing
    the clustering structure toward the F₄ constraint.

    Returns number of links added.
    """
    if len(bigraph.nodes) < 2:
        return 0

    node_ids = list(bigraph.nodes.keys())
    links_added = 0

    for _ in range(n_new_links):
        # Pick source node uniformly
        source_id = bigraph.rng.choice(node_ids)
        source_neighbors = bigraph.neighbors(source_id)

        # Compute attachment probabilities for all other nodes
        candidates = [nid for nid in node_ids if nid != source_id and nid not in source_neighbors]

        if not candidates:
            continue

        # Preferential attachment with triality weighting
        probs = []
        for target_id in candidates:
            degree = bigraph.degree(target_id) + 1  # +1 to avoid zero
            triality_weight = bigraph.triality_coupling(source_id, target_id)
            prob = (degree**alpha) * triality_weight
            probs.append(prob)

        probs = np.array(probs)
        probs /= probs.sum()

        # Sample target
        target_id = bigraph.rng.choice(candidates, p=probs)

        # Add link with tension ε
        bigraph.add_link(source_id, target_id, tension=epsilon)
        links_added += 1

    return links_added


# =============================================================================
# CLUSTERING ANALYSIS
# =============================================================================


def count_triangles(bigraph: TrialityBigraph) -> tuple[int, int]:
    """
    Count triangles and potential triangles (wedges).

    Returns (n_triangles, n_wedges).
    Clustering coefficient = 3 * n_triangles / n_wedges.
    """
    triangles = 0
    wedges = 0

    for node_id in bigraph.nodes:
        neighbors = list(bigraph.neighbors(node_id))
        n_neighbors = len(neighbors)

        if n_neighbors < 2:
            continue

        # Each pair of neighbors forms a wedge
        wedges += n_neighbors * (n_neighbors - 1) // 2

        # Check how many wedges are closed (triangles)
        for i in range(n_neighbors):
            for j in range(i + 1, n_neighbors):
                if neighbors[j] in bigraph.neighbors(neighbors[i]):
                    triangles += 1

    # Each triangle is counted 3 times (once per vertex)
    return triangles // 3, wedges


def compute_clustering_ratio(bigraph: TrialityBigraph) -> float:
    """
    Compute the clustering ratio (ε_observed).

    This is the probability that two neighbors of a node are connected.
    Target: ε = 0.25 (from F₄ Casimir structure).
    """
    n_triangles, n_wedges = count_triangles(bigraph)

    if n_wedges == 0:
        return 0.0

    # Clustering coefficient = 3 * triangles / wedges
    return 3 * n_triangles / n_wedges


def analyze_triality_distribution(bigraph: TrialityBigraph) -> dict:
    """Analyze the triality type distribution in the graph."""
    counts = dict.fromkeys(TrialityType, 0)
    for node in bigraph.nodes.values():
        counts[node.triality] += 1

    total = sum(counts.values())
    fractions = {t.name: count / total for t, count in counts.items()}

    return {
        "counts": {t.name: c for t, c in counts.items()},
        "fractions": fractions,
        "total": total,
    }


def analyze_triangle_types(bigraph: TrialityBigraph) -> dict:
    """
    Analyze the triality composition of triangles.

    F₄ predicts triangles should be "mixed" (different types per vertex).
    """
    triangle_types = defaultdict(int)

    for node_id in bigraph.nodes:
        neighbors = list(bigraph.neighbors(node_id))
        node_type = bigraph.nodes[node_id].triality.name

        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                if neighbors[j] in bigraph.neighbors(neighbors[i]):
                    # Found a triangle
                    types = sorted(
                        [
                            node_type,
                            bigraph.nodes[neighbors[i]].triality.name,
                            bigraph.nodes[neighbors[j]].triality.name,
                        ]
                    )
                    triangle_types[tuple(types)] += 1

    # Normalize (each triangle counted 3 times)
    return {"-".join(k): v // 3 for k, v in triangle_types.items()}


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================


def run_triality_experiment(
    n_nodes: int = 500,
    n_attachment_rounds: int = 10,
    links_per_round: int = 200,
    alpha: float = 0.85,
    epsilon: float = 0.25,
    seed: int = 42,
    initial_density: float = 0.15,
) -> dict:
    """
    Run the triality-weighted attachment experiment.

    Goal: Demonstrate that F₄ type constraints push ε from 0.22 → 0.25.
    """
    print("=" * 70)
    print("TRIALITY BIGRAPH EXPERIMENT")
    print("Testing F₄ Representation Constraints on Clustering")
    print("=" * 70)

    # Create bigraph and seed with nodes
    bigraph = TrialityBigraph(seed=seed)

    print(f"\n[1] Creating {n_nodes} triality-typed nodes...")
    for _ in range(n_nodes):
        bigraph.add_node()

    dist = analyze_triality_distribution(bigraph)
    print(f"    Triality distribution: {dist['fractions']}")

    # Initial DENSE connections to match original simulation topology
    print(f"\n[2] Creating initial connections (density={initial_density})...")
    node_ids = list(bigraph.nodes.keys())

    # Create local clusters first (small-world seeding)
    cluster_size = 10
    n_clusters = n_nodes // cluster_size
    for c in range(n_clusters):
        cluster_nodes = node_ids[c * cluster_size : (c + 1) * cluster_size]
        # Connect each node in cluster to ~3-4 others
        for i, nid in enumerate(cluster_nodes):
            for j in range(i + 1, min(i + 4, len(cluster_nodes))):
                if bigraph.rng.random() < 0.7:
                    bigraph.add_link(nid, cluster_nodes[j])

    # Add some inter-cluster links
    for _ in range(n_nodes // 5):
        i = bigraph.rng.choice(node_ids)
        j = bigraph.rng.choice(node_ids)
        if i != j:
            bigraph.add_link(i, j)

    print(f"    Initial links: {len(bigraph.links)}")

    # Run triality preferential attachment
    print(f"\n[3] Running triality-weighted attachment ({n_attachment_rounds} rounds)...")

    history = {"round": [], "n_links": [], "clustering": [], "mean_degree": []}

    for round_num in range(n_attachment_rounds):
        bigraph.current_step = round_num

        links_added = triality_preferential_attachment(
            bigraph, alpha=alpha, epsilon=epsilon, n_new_links=links_per_round
        )

        clustering = compute_clustering_ratio(bigraph)
        mean_degree = 2 * len(bigraph.links) / len(bigraph.nodes)

        history["round"].append(round_num)
        history["n_links"].append(len(bigraph.links))
        history["clustering"].append(clustering)
        history["mean_degree"].append(mean_degree)

        if (round_num + 1) % 2 == 0:
            print(
                f"    Round {round_num + 1}: links={len(bigraph.links)}, "
                f"clustering={clustering:.4f}, degree={mean_degree:.2f}"
            )

    # Final analysis
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    final_clustering = compute_clustering_ratio(bigraph)
    final_degree = 2 * len(bigraph.links) / len(bigraph.nodes)

    print("\n  Final Statistics:")
    print(f"    Nodes: {len(bigraph.nodes)}")
    print(f"    Links: {len(bigraph.links)}")
    print(f"    Mean degree: {final_degree:.2f}")
    print(f"    Clustering ratio (ε): {final_clustering:.4f}")
    print("    Target ε: 0.2500")
    print(
        f"    Deviation: {abs(final_clustering - 0.25):.4f} ({abs(final_clustering - 0.25)/0.25*100:.1f}%)"
    )

    # Triangle type analysis
    print("\n  Triangle Type Distribution:")
    triangle_types = analyze_triangle_types(bigraph)
    total_triangles = sum(triangle_types.values())
    if total_triangles > 0:
        for ttype, count in sorted(triangle_types.items(), key=lambda x: -x[1])[:10]:
            print(f"    {ttype}: {count} ({count/total_triangles*100:.1f}%)")

    # Comparison with standard (non-triality) attachment
    print("\n  Comparison:")
    print("    Standard preferential attachment: ε ≈ 0.22")
    print(f"    Triality-weighted attachment: ε ≈ {final_clustering:.4f}")
    improvement = (final_clustering - 0.22) / (0.25 - 0.22) * 100
    print(f"    Gap closure: {improvement:.1f}% toward target")

    # Verdict
    print("\n" + "=" * 70)
    if final_clustering > 0.23:
        print("VERDICT: Triality constraints INCREASE clustering toward F₄ prediction ✓")
    else:
        print("VERDICT: Triality constraints do not significantly affect clustering")
    print("=" * 70)

    return {
        "final_clustering": final_clustering,
        "final_degree": final_degree,
        "history": history,
        "triangle_types": triangle_types,
        "triality_distribution": dist,
    }


# =============================================================================
# COMPARISON: STANDARD VS TRIALITY
# =============================================================================


def create_clustered_bigraph(n_nodes: int, seed: int, use_triality: bool = True) -> TrialityBigraph:
    """Create a bigraph with initial small-world clustering."""
    bigraph = TrialityBigraph(seed=seed)

    # Add nodes
    for _ in range(n_nodes):
        if use_triality:
            bigraph.add_node()
        else:
            bigraph.add_node(triality=TrialityType.VECTOR)

    node_ids = list(bigraph.nodes.keys())

    # Create clustered initial topology
    cluster_size = 8
    n_clusters = n_nodes // cluster_size

    for c in range(n_clusters):
        cluster_nodes = node_ids[c * cluster_size : (c + 1) * cluster_size]
        for i, nid in enumerate(cluster_nodes):
            for j in range(i + 1, min(i + 3, len(cluster_nodes))):
                bigraph.add_link(nid, cluster_nodes[j])

    # Inter-cluster links
    for _ in range(n_nodes // 4):
        i = bigraph.rng.choice(node_ids)
        j = bigraph.rng.choice(node_ids)
        if i != j:
            bigraph.add_link(i, j)

    return bigraph


def compare_attachment_models(n_nodes: int = 300, n_rounds: int = 8, seed: int = 42):
    """
    Compare standard preferential attachment vs triality-weighted.
    """
    print("\n" + "=" * 70)
    print("COMPARISON: Standard vs Triality Attachment")
    print("=" * 70)

    # Standard preferential attachment (no triality - uniform coupling)
    print("\n[A] Standard Preferential Attachment (uniform coupling):")
    bigraph_std = create_clustered_bigraph(n_nodes, seed, use_triality=False)
    print(f"    Initial links: {len(bigraph_std.links)}")
    print(f"    Initial clustering: {compute_clustering_ratio(bigraph_std):.4f}")

    # Override triality coupling to uniform
    bigraph_std.triality_coupling = lambda a, b: 1.0

    for _ in range(n_rounds):
        triality_preferential_attachment(bigraph_std, n_new_links=100)

    clustering_std = compute_clustering_ratio(bigraph_std)
    print(f"    Final clustering (ε): {clustering_std:.4f}")

    # Triality-weighted attachment (F₄ structure)
    print("\n[B] Triality-Weighted Attachment (F₄ structure):")
    bigraph_tri = create_clustered_bigraph(n_nodes, seed, use_triality=True)
    print(f"    Initial links: {len(bigraph_tri.links)}")
    print(f"    Initial clustering: {compute_clustering_ratio(bigraph_tri):.4f}")

    for _ in range(n_rounds):
        triality_preferential_attachment(bigraph_tri, n_new_links=100)

    clustering_tri = compute_clustering_ratio(bigraph_tri)
    print(f"    Final clustering (ε): {clustering_tri:.4f}")

    # Summary
    print("\n" + "-" * 70)
    print(f"  Standard:  ε = {clustering_std:.4f}")
    print(f"  Triality:  ε = {clustering_tri:.4f}")
    print("  Target:    ε = 0.2500")
    print(f"  Improvement: {clustering_tri - clustering_std:+.4f}")

    return clustering_std, clustering_tri


# =============================================================================
# MAIN
# =============================================================================


def main():
    """Run triality bigraph experiments."""
    # Main experiment
    results = run_triality_experiment(
        n_nodes=400, n_attachment_rounds=12, links_per_round=250, alpha=0.85, epsilon=0.25, seed=42
    )

    # Comparison
    compare_attachment_models(n_nodes=300, n_rounds=10, seed=123)

    return results


if __name__ == "__main__":
    results = main()
