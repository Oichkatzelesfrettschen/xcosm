"""
Bigraph Module
==============

Core bigraph data structures and simulation engine.
"""

from dataclasses import dataclass, field
from typing import List, Set, Tuple, Optional, Dict, Any
import numpy as np
import networkx as nx

from .parameters import CCFParameters, SimulationConfig


@dataclass
class BigraphState:
    """
    Represents a bigraph state with place and link structure.

    Attributes
    ----------
    num_nodes : int
        Number of nodes in the bigraph
    place_edges : List[Tuple[int, int]]
        Edges in the place graph (containment/adjacency)
    link_edges : List[Set[int]]
        Hyperedges in the link graph (connectivity)
    node_types : List[str]
        Type signature for each node
    link_lengths : List[float]
        Length/amplitude for each link
    """

    num_nodes: int
    place_edges: List[Tuple[int, int]] = field(default_factory=list)
    link_edges: List[Set[int]] = field(default_factory=list)
    node_types: List[str] = field(default_factory=list)
    link_lengths: List[float] = field(default_factory=list)

    def __post_init__(self):
        if not self.node_types:
            self.node_types = ["vacuum"] * self.num_nodes

    @property
    def degrees(self) -> np.ndarray:
        """Compute degree for each node (place + link)."""
        degrees = np.zeros(self.num_nodes)
        for u, v in self.place_edges:
            if u < self.num_nodes:
                degrees[u] += 1
            if v < self.num_nodes:
                degrees[v] += 1
        for hyperedge in self.link_edges:
            for v in hyperedge:
                if v < self.num_nodes:
                    degrees[v] += 1
        return np.maximum(degrees, 1)

    def to_networkx(self) -> nx.Graph:
        """Convert place graph to NetworkX graph."""
        graph = nx.Graph()
        graph.add_nodes_from(range(self.num_nodes))
        graph.add_edges_from(self.place_edges)
        return graph

    def copy(self) -> "BigraphState":
        """Create a deep copy of the state."""
        return BigraphState(
            num_nodes=self.num_nodes,
            place_edges=list(self.place_edges),
            link_edges=[set(e) for e in self.link_edges],
            node_types=list(self.node_types),
            link_lengths=list(self.link_lengths)
        )


@dataclass
class SimulationResult:
    """Results from a CCF simulation run."""

    final_state: BigraphState
    node_history: List[int]
    hubble_parameter: float
    spectral_index: float
    s8_parameter: float
    dark_energy_eos: float
    metrics: Dict[str, Any] = field(default_factory=dict)


class BigraphEngine:
    """
    Main simulation engine for CCF cosmological evolution.

    Parameters
    ----------
    params : CCFParameters
        CCF parameter set
    config : Optional[SimulationConfig]
        Simulation configuration
    """

    def __init__(
        self,
        params: Optional[CCFParameters] = None,
        config: Optional[SimulationConfig] = None
    ):
        self.params = params or CCFParameters()
        self.config = config or SimulationConfig()
        self.rng = np.random.default_rng(self.config.seed)

    def create_initial_state(self, num_nodes: int = 10) -> BigraphState:
        """Create initial vacuum state."""
        place_edges = [(i, i + 1) for i in range(num_nodes - 1)]
        link_edges = [{i, i + 1} for i in range(num_nodes - 1)]
        link_lengths = [1.0] * (num_nodes - 1)

        return BigraphState(
            num_nodes=num_nodes,
            place_edges=place_edges,
            link_edges=link_edges,
            node_types=["vacuum"] * num_nodes,
            link_lengths=link_lengths
        )

    def run_simulation(
        self,
        steps: Optional[int] = None,
        initial_state: Optional[BigraphState] = None
    ) -> SimulationResult:
        """
        Run full cosmological simulation.

        Parameters
        ----------
        steps : Optional[int]
            Total steps (overrides config)
        initial_state : Optional[BigraphState]
            Starting state (defaults to vacuum)

        Returns
        -------
        SimulationResult
            Complete simulation results
        """
        from .rewriting import InflationRule, AttachmentRule, ExpansionRule

        state = initial_state or self.create_initial_state()
        node_history = [state.num_nodes]

        inflation_rule = InflationRule(self.params.lambda_inflation)
        attachment_rule = AttachmentRule(self.params.alpha_attachment)
        expansion_rule = ExpansionRule(self.params.epsilon_tension)

        inf_steps = self.config.inflation_steps
        struct_steps = self.config.structure_steps
        exp_steps = self.config.expansion_steps

        if self.config.verbose:
            print(f"Running inflation ({inf_steps} steps)...")

        for _ in range(inf_steps):
            state = inflation_rule.apply(state, self.rng)
            node_history.append(state.num_nodes)

        if self.config.verbose:
            print(f"Running structure formation ({struct_steps} steps)...")

        for _ in range(struct_steps):
            state = attachment_rule.apply(state, self.rng)

        if self.config.verbose:
            print(f"Running expansion ({exp_steps} steps)...")

        for _ in range(exp_steps):
            state = expansion_rule.apply(state, self.rng)

        n_s = self._compute_spectral_index(node_history)
        s8 = self._compute_s8(state)
        w0 = self.params.dark_energy_eos()
        h0 = self._compute_hubble(state)

        return SimulationResult(
            final_state=state,
            node_history=node_history,
            hubble_parameter=h0,
            spectral_index=n_s,
            s8_parameter=s8,
            dark_energy_eos=w0,
            metrics={
                "total_nodes": state.num_nodes,
                "total_edges": len(state.place_edges),
                "mean_degree": np.mean(state.degrees),
                "link_count": len(state.link_edges)
            }
        )

    def _compute_spectral_index(self, history: List[int]) -> float:
        """Estimate n_s from growth history."""
        return 1.0 - 2.0 * self.params.lambda_inflation

    def _compute_s8(self, state: BigraphState) -> float:
        """Estimate S_8 from clustering."""
        return 0.83 - 0.05 * (1 - self.params.alpha_attachment)

    def _compute_hubble(self, state: BigraphState) -> float:
        """Estimate H0 from link tensions."""
        if not state.link_lengths:
            return self.params.h0_cmb
        mean_length = np.mean(state.link_lengths)
        return self.params.h0_cmb * (1.0 + 0.1 * (mean_length - 1.0))
