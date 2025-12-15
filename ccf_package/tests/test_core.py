"""
Core CCF Tests
==============

Unit tests for the Computational Cosmogenesis Framework.
"""

import pytest
import numpy as np

from ccf import (
    CCFParameters,
    BigraphState,
    BigraphEngine,
    InflationRule,
    AttachmentRule,
    ExpansionRule,
    OllivierRicciCurvature,
    H0GradientAnalysis,
    H0Measurement,
    compute_spectral_index,
    compute_s8,
    compute_dark_energy_eos,
)


class TestCCFParameters:
    """Test CCF parameter calculations."""

    def test_default_parameters(self):
        """Test default parameter values."""
        params = CCFParameters()
        assert params.lambda_inflation == 0.003
        assert params.alpha_attachment == 0.85
        assert params.epsilon_tension == 0.25

    def test_spectral_index(self):
        """Test n_s calculation."""
        params = CCFParameters()
        n_s = params.spectral_index()
        # n_s = 1 - 2*lambda = 1 - 2*0.003 = 0.994
        assert 0.99 < n_s < 1.0
        assert np.isclose(n_s, 0.994, atol=0.01)

    def test_dark_energy_eos(self):
        """Test w_0 calculation."""
        params = CCFParameters()
        w0 = params.dark_energy_eos()
        assert -1.0 < w0 < -0.8
        assert np.isclose(w0, -0.833, atol=0.01)

    def test_tensor_to_scalar(self):
        """Test r calculation."""
        params = CCFParameters()
        r = params.tensor_to_scalar()
        assert 0.001 < r < 0.01
        assert np.isclose(r, 0.0048, atol=0.001)

    def test_hubble_at_scale(self):
        """Test H0(k) gradient."""
        params = CCFParameters()

        h0_cmb = params.hubble_at_scale(1e-4)
        h0_local = params.hubble_at_scale(1.0)  # Use k=1.0 Mpc^-1 for local

        assert h0_cmb < h0_local
        assert h0_cmb < 68  # CMB scale: ~65 km/s/Mpc
        assert h0_local > 69  # Local scale: ~69.7 km/s/Mpc


class TestBigraphState:
    """Test bigraph state operations."""

    def test_creation(self):
        """Test state creation."""
        state = BigraphState(
            num_nodes=10,
            place_edges=[(0, 1), (1, 2)],
            link_edges=[{0, 1}, {1, 2}],
            node_types=["vacuum"] * 10,
            link_lengths=[1.0, 1.0]
        )
        assert state.num_nodes == 10
        assert len(state.place_edges) == 2

    def test_degrees(self):
        """Test degree computation."""
        state = BigraphState(
            num_nodes=3,
            place_edges=[(0, 1), (1, 2)],
            link_edges=[{0, 1}],
            node_types=["vacuum"] * 3,
            link_lengths=[1.0]
        )
        degrees = state.degrees
        assert len(degrees) == 3
        assert degrees[1] >= degrees[0]

    def test_to_networkx(self):
        """Test NetworkX conversion."""
        state = BigraphState(
            num_nodes=5,
            place_edges=[(0, 1), (1, 2), (2, 3)],
            link_edges=[],
            node_types=["vacuum"] * 5,
            link_lengths=[]
        )
        graph = state.to_networkx()
        assert graph.number_of_nodes() == 5
        assert graph.number_of_edges() == 3

    def test_copy(self):
        """Test state copying."""
        state = BigraphState(
            num_nodes=5,
            place_edges=[(0, 1)],
            link_edges=[{0, 1}],
            node_types=["vacuum"] * 5,
            link_lengths=[1.0]
        )
        copy = state.copy()
        assert copy.num_nodes == state.num_nodes
        assert copy is not state


class TestRewritingRules:
    """Test cosmological rewriting rules."""

    def test_inflation_rule(self):
        """Test inflationary expansion."""
        state = BigraphState(
            num_nodes=10,
            place_edges=[(i, i+1) for i in range(9)],
            link_edges=[{i, i+1} for i in range(9)],
            node_types=["vacuum"] * 10,
            link_lengths=[1.0] * 9
        )

        rule = InflationRule(lmbda=1.0)
        rng = np.random.default_rng(42)

        new_state = rule.apply(state, rng)
        assert new_state.num_nodes >= state.num_nodes

    def test_attachment_rule(self):
        """Test preferential attachment."""
        state = BigraphState(
            num_nodes=20,
            place_edges=[(i, i+1) for i in range(19)],
            link_edges=[],
            node_types=["matter"] * 20,
            link_lengths=[]
        )

        rule = AttachmentRule(alpha=0.85)
        rng = np.random.default_rng(42)

        new_state = rule.apply(state, rng)
        assert len(new_state.place_edges) >= len(state.place_edges)

    def test_expansion_rule(self):
        """Test cosmological expansion."""
        state = BigraphState(
            num_nodes=10,
            place_edges=[],
            link_edges=[{0, 1}],
            node_types=["dark"] * 10,
            link_lengths=[1.0]
        )

        rule = ExpansionRule(epsilon=0.25)
        rng = np.random.default_rng(42)

        new_state = rule.apply(state, rng)
        assert new_state.link_lengths[0] > state.link_lengths[0]


class TestBigraphEngine:
    """Test simulation engine."""

    def test_initial_state(self):
        """Test initial state creation."""
        engine = BigraphEngine()
        state = engine.create_initial_state(20)
        assert state.num_nodes == 20

    def test_simulation_runs(self):
        """Test that simulation completes."""
        from ccf.parameters import SimulationConfig
        config = SimulationConfig(
            inflation_steps=10,
            structure_steps=10,
            expansion_steps=5,
            verbose=False
        )
        engine = BigraphEngine(config=config)
        result = engine.run_simulation()

        assert result.final_state.num_nodes >= 10  # May start with 10 and not grow
        assert result.spectral_index > 0.9
        assert result.dark_energy_eos < 0


class TestOllivierRicciCurvature:
    """Test curvature computations."""

    def test_ring_curvature(self):
        """Test curvature on ring graph."""
        import networkx as nx
        graph = nx.cycle_graph(10)

        orc = OllivierRicciCurvature()
        kappa = orc.compute_edge_curvature(graph, 0, 1)

        assert kappa > 0

    def test_scalar_curvature(self):
        """Test total scalar curvature."""
        import networkx as nx
        graph = nx.complete_graph(5)

        orc = OllivierRicciCurvature()
        scalar = orc.scalar_curvature(graph)

        assert scalar > 0


class TestH0Analysis:
    """Test H0 gradient analysis."""

    def test_fit_gradient(self):
        """Test gradient fitting."""
        analysis = H0GradientAnalysis()
        result = analysis.fit_gradient()

        assert "slope" in result
        assert "intercept" in result
        assert result["slope"] > 0

    def test_flat_comparison(self):
        """Test comparison with flat model."""
        analysis = H0GradientAnalysis()
        flat = analysis.test_flat_model()

        assert "delta_chi2" in flat
        assert flat["delta_chi2"] > 0

    def test_ccf_comparison(self):
        """Test CCF consistency check."""
        analysis = H0GradientAnalysis()
        ccf = analysis.ccf_comparison()

        assert "combined_tension_sigma" in ccf


class TestObservables:
    """Test observable computations."""

    def test_spectral_index(self):
        """Test n_s computation."""
        n_s, sigma = compute_spectral_index(0.003)
        assert 0.99 < n_s < 1.0
        assert sigma > 0

    def test_s8(self):
        """Test S_8 computation."""
        s8, sigma = compute_s8(0.85)
        assert 0.7 < s8 < 0.9
        assert sigma > 0

    def test_dark_energy_eos(self):
        """Test w_0 computation."""
        w0, sigma = compute_dark_energy_eos(0.25)
        assert -1.0 < w0 < -0.8
        assert sigma > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
