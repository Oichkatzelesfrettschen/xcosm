#!/usr/bin/env python3
"""
CCF Quick Start Example
=======================

Demonstrates basic usage of the Computational Cosmogenesis Framework.
"""

import sys
sys.path.insert(0, '..')

from ccf import (
    CCFParameters,
    BigraphEngine,
    H0GradientAnalysis,
    OllivierRicciCurvature,
)
from ccf.parameters import SimulationConfig
from ccf.observables import predict_cmbs4_observables


def main():
    print("=" * 60)
    print("CCF QUICK START DEMO")
    print("=" * 60)

    print("\n1. CCF Parameters")
    print("-" * 40)

    params = CCFParameters()
    print(params)

    print("\n2. Run Simulation")
    print("-" * 40)

    config = SimulationConfig(
        inflation_steps=50,
        structure_steps=100,
        expansion_steps=25,
        seed=42,
        verbose=True
    )

    engine = BigraphEngine(params, config)
    result = engine.run_simulation()

    print(f"\nResults:")
    print(f"  Final nodes:  {result.final_state.num_nodes}")
    print(f"  Final edges:  {len(result.final_state.place_edges)}")
    print(f"  Mean degree:  {result.metrics['mean_degree']:.2f}")
    print(f"  H0:           {result.hubble_parameter:.2f} km/s/Mpc")
    print(f"  n_s:          {result.spectral_index:.4f}")
    print(f"  w_0:          {result.dark_energy_eos:.3f}")

    print("\n3. H0 Gradient Analysis")
    print("-" * 40)

    analysis = H0GradientAnalysis()
    fit = analysis.fit_gradient()

    print(f"  Gradient: {fit['slope']:.2f} +/- {fit['sigma_slope']:.2f} km/s/Mpc/decade")
    print(f"  Significance: {fit['significance']:.1f} sigma")
    print(f"  chi2/dof: {fit['chi2_reduced']:.2f}")

    print("\n4. CMB-S4 Predictions")
    print("-" * 40)

    preds = predict_cmbs4_observables(params)
    print(f"  r = {preds['r']['value']:.4f} (S/N = {preds['r']['detection_significance']:.1f})")
    print(f"  n_t = {preds['n_t']['value']:.5f}")
    print(f"  R = {preds['R']['ccf_prediction']:.2f} (vs standard: {preds['R']['expected_standard']:.0f})")

    print("\n5. Curvature Computation")
    print("-" * 40)

    graph = result.final_state.to_networkx()
    if graph.number_of_edges() > 0:
        orc = OllivierRicciCurvature()
        mean_kappa = orc.mean_curvature(graph)
        print(f"  Mean Ollivier-Ricci curvature: {mean_kappa:.4f}")
    else:
        print("  No edges for curvature computation")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
