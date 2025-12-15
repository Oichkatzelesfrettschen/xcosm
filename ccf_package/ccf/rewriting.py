"""
Rewriting Rules Module
======================

Defines the cosmological rewriting rules for bigraph evolution.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict
import numpy as np

from .bigraph import BigraphState


class RewritingRule(ABC):
    """Abstract base class for rewriting rules."""

    def __init__(self, name: str, params: Dict[str, float]):
        self.name = name
        self.params = params

    @abstractmethod
    def apply(
        self,
        state: BigraphState,
        rng: np.random.Generator
    ) -> BigraphState:
        """Apply the rewriting rule to the bigraph state."""
        pass


class InflationRule(RewritingRule):
    """
    Inflationary node duplication rule.

    R_inf: O -> O-O

    Each node has probability lambda of duplicating per step.
    """

    def __init__(self, lmbda: float = 0.003):
        super().__init__("inflation", {"lambda": lmbda})

    def apply(
        self,
        state: BigraphState,
        rng: np.random.Generator
    ) -> BigraphState:
        """Apply inflationary expansion."""
        lmbda = self.params["lambda"]

        if rng.random() > lmbda * state.num_nodes:
            return state

        source = rng.integers(0, state.num_nodes)
        new_id = state.num_nodes

        new_place = list(state.place_edges) + [(source, new_id)]
        new_links = list(state.link_edges) + [{source, new_id}]
        new_types = list(state.node_types) + [state.node_types[source]]
        new_lengths = list(state.link_lengths) + [1.0]

        return BigraphState(
            num_nodes=state.num_nodes + 1,
            place_edges=new_place,
            link_edges=new_links,
            node_types=new_types,
            link_lengths=new_lengths
        )


class AttachmentRule(RewritingRule):
    """
    Preferential attachment rule.

    P(link to v) ~ deg(v)^alpha

    Implements gravitational clustering through preferential
    attachment with exponent alpha.
    """

    def __init__(self, alpha: float = 0.85):
        super().__init__("attachment", {"alpha": alpha})

    def apply(
        self,
        state: BigraphState,
        rng: np.random.Generator
    ) -> BigraphState:
        """Apply preferential attachment."""
        if state.num_nodes < 2:
            return state

        alpha = self.params["alpha"]
        degrees = state.degrees

        probs = degrees ** alpha
        probs = probs / np.sum(probs)

        source = rng.integers(0, state.num_nodes)
        target = rng.choice(state.num_nodes, p=probs)

        if source == target:
            return state

        edge = (min(source, target), max(source, target))
        if edge in state.place_edges:
            return state

        new_place = list(state.place_edges) + [edge]

        return BigraphState(
            num_nodes=state.num_nodes,
            place_edges=new_place,
            link_edges=state.link_edges,
            node_types=state.node_types,
            link_lengths=state.link_lengths
        )


class ExpansionRule(RewritingRule):
    """
    Cosmological expansion rule.

    l -> l * (1 + H * dt)

    Links expand according to the Hubble parameter determined
    by link tension epsilon.
    """

    def __init__(self, epsilon: float = 0.25, h0: float = 70.0):
        super().__init__("expansion", {"epsilon": epsilon, "H0": h0})

    def apply(
        self,
        state: BigraphState,
        rng: np.random.Generator
    ) -> BigraphState:
        """Apply cosmological expansion to links."""
        h0 = self.params["H0"]
        dt = 0.01
        factor = 1 + h0 * dt / 3e5

        new_lengths = [l * factor for l in state.link_lengths]

        return BigraphState(
            num_nodes=state.num_nodes,
            place_edges=state.place_edges,
            link_edges=state.link_edges,
            node_types=state.node_types,
            link_lengths=new_lengths
        )


class ReheatRule(RewritingRule):
    """
    Reheating transition rule.

    vacuum -> {matter, radiation, dark}

    Converts vacuum nodes to matter types with specified fractions.
    """

    def __init__(
        self,
        matter_fraction: float = 0.27,
        radiation_fraction: float = 0.0001,
        dark_fraction: float = 0.68
    ):
        super().__init__("reheat", {
            "matter": matter_fraction,
            "radiation": radiation_fraction,
            "dark": dark_fraction
        })

    def apply(
        self,
        state: BigraphState,
        rng: np.random.Generator
    ) -> BigraphState:
        """Apply reheating to convert vacuum nodes."""
        fractions = [
            self.params["matter"],
            self.params["radiation"],
            self.params["dark"]
        ]
        fractions.append(1 - sum(fractions))  # remaining vacuum
        types = ["matter", "radiation", "dark", "vacuum"]

        new_types = []
        for node_type in state.node_types:
            if node_type == "vacuum":
                new_types.append(rng.choice(types, p=fractions))
            else:
                new_types.append(node_type)

        return BigraphState(
            num_nodes=state.num_nodes,
            place_edges=state.place_edges,
            link_edges=state.link_edges,
            node_types=new_types,
            link_lengths=state.link_lengths
        )
