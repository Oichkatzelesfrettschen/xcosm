"""
CLI Module
==========

Command-line interface for CCF simulations and analysis.
"""

import argparse
import sys

from .parameters import CCFParameters, SimulationConfig
from .bigraph import BigraphEngine
from .analysis import H0GradientAnalysis
from .observables import predict_cmbs4_observables


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="ccf-simulate",
        description="Computational Cosmogenesis Framework simulation and analysis"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    sim_parser = subparsers.add_parser("simulate", help="Run cosmological simulation")
    sim_parser.add_argument(
        "--steps", type=int, default=350,
        help="Total simulation steps"
    )
    sim_parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility"
    )
    sim_parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output"
    )

    h0_parser = subparsers.add_parser("h0-analysis", help="Run H0 gradient analysis")
    h0_parser.add_argument(
        "--compare-ccf", action="store_true",
        help="Compare with CCF predictions"
    )

    pred_parser = subparsers.add_parser("predict", help="Generate CCF predictions")
    pred_parser.add_argument(
        "--cmbs4", action="store_true",
        help="CMB-S4 tensor predictions"
    )
    pred_parser.add_argument(
        "--all", action="store_true",
        help="All observables"
    )

    params_parser = subparsers.add_parser("params", help="Show CCF parameters")

    args = parser.parse_args()

    if args.command == "simulate":
        run_simulation(args)
    elif args.command == "h0-analysis":
        run_h0_analysis(args)
    elif args.command == "predict":
        run_predictions(args)
    elif args.command == "params":
        show_parameters()
    else:
        parser.print_help()


def run_simulation(args):
    """Run cosmological simulation."""
    params = CCFParameters()
    config = SimulationConfig(
        inflation_steps=100,
        structure_steps=200,
        expansion_steps=50,
        seed=args.seed,
        verbose=not args.quiet
    )

    engine = BigraphEngine(params, config)
    result = engine.run_simulation()

    print("\n" + "=" * 50)
    print("SIMULATION RESULTS")
    print("=" * 50)
    print(f"Final nodes:      {result.final_state.num_nodes}")
    print(f"Final edges:      {len(result.final_state.place_edges)}")
    print(f"Hubble parameter: {result.hubble_parameter:.2f} km/s/Mpc")
    print(f"Spectral index:   {result.spectral_index:.4f}")
    print(f"S_8 parameter:    {result.s8_parameter:.3f}")
    print(f"Dark energy w_0:  {result.dark_energy_eos:.3f}")
    print("=" * 50)


def run_h0_analysis(args):
    """Run H0 gradient analysis."""
    analysis = H0GradientAnalysis()
    print(analysis.summary())

    if args.compare_ccf:
        ccf = analysis.ccf_comparison()
        print("\nCCF Comparison Details:")
        print(f"  Predicted intercept: {ccf['ccf_intercept']:.2f}")
        print(f"  Predicted slope:     {ccf['ccf_slope']:.2f}")
        print(f"  Fitted intercept:    {ccf['fitted_intercept']:.2f}")
        print(f"  Fitted slope:        {ccf['fitted_slope']:.2f}")


def run_predictions(args):
    """Generate CCF predictions."""
    params = CCFParameters()

    if args.cmbs4 or args.all:
        print("\n" + "=" * 50)
        print("CMB-S4 PREDICTIONS")
        print("=" * 50)

        preds = predict_cmbs4_observables(params)

        print(f"\nTensor-to-scalar ratio r:")
        print(f"  CCF prediction: {preds['r']['value']:.4f} +/- {preds['r']['sigma']:.4f}")
        print(f"  CMB-S4 sensitivity: {preds['r']['cmbs4_sensitivity']}")
        print(f"  Expected S/N: {preds['r']['detection_significance']:.1f}")

        print(f"\nTensor tilt n_t:")
        print(f"  CCF prediction: {preds['n_t']['value']:.5f}")

        print(f"\nConsistency ratio R = r/(-8*n_t):")
        print(f"  Standard inflation: {preds['R']['expected_standard']:.2f}")
        print(f"  CCF prediction: {preds['R']['ccf_prediction']:.2f}")
        print(f"  Distinguishing power: {preds['R']['distinguishing_power']:.1f} sigma")

        print("=" * 50)

    if args.all and not args.cmbs4:
        print("\n" + "=" * 50)
        print("ALL CCF PREDICTIONS")
        print("=" * 50)
        print(f"\nScalar spectral index n_s: {params.spectral_index():.4f}")
        print(f"S_8 parameter: {params.s8_parameter():.3f}")
        print(f"Dark energy w_0: {params.dark_energy_eos():.3f}")
        print(f"Tensor-to-scalar r: {params.tensor_to_scalar():.4f}")
        print("=" * 50)


def show_parameters():
    """Display CCF parameters."""
    params = CCFParameters()
    print(params)


if __name__ == "__main__":
    main()
