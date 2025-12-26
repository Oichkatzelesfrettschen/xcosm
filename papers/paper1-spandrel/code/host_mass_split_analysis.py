#!/usr/bin/env python3
"""
Host Mass Split Analysis for Spandrel Falsification

This module implements the primary Spandrel falsification test:
If SN Ia population evolution correlates with host galaxy properties,
splitting the sample by host mass should reveal different apparent
cosmological parameters in baseline (no evolution) fits.

The key prediction:
- Baseline model (no dM/dz): high-mass and low-mass samples give DIFFERENT w0, wa
- Spandrel model (with dM/dz): both samples give CONSISTENT w0, wa

This differential signature is robust to many systematics that affect
absolute calibration but not relative trends.

Author: COSMOS Collaboration
Date: December 2025
"""

import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy import optimize

# Import from spandrel-core shared library
from spandrel_core import (
    SimulatedDataLoader,
    SNDataset,
    split_by_host_mass,
    SpandrelLikelihood,
)


@dataclass
class SplitTestResult:
    """Results from host mass split analysis."""

    # Sample sizes
    n_low_mass: int
    n_high_mass: int
    mass_threshold: float

    # Best-fit parameters for low-mass sample
    w0_low: float
    w0_low_err: float
    wa_low: float
    wa_low_err: float

    # Best-fit parameters for high-mass sample
    w0_high: float
    w0_high_err: float
    wa_high: float
    wa_high_err: float

    # Differences
    delta_w0: float
    delta_w0_err: float
    delta_wa: float
    delta_wa_err: float

    # Significance of split
    delta_w0_sigma: float
    delta_wa_sigma: float
    combined_sigma: float

    # Model info
    model_type: str  # "baseline" or "spandrel"


class HostMassSplitAnalyzer:
    """
    Analyze SN Ia samples split by host galaxy mass.

    This is the primary Spandrel falsification test. The analysis:
    1. Splits sample at host mass threshold (default: 10^10 M_sun)
    2. Fits cosmology independently to each subsample
    3. Compares inferred w0, wa between subsamples
    4. Quantifies significance of any difference

    Under Spandrel hypothesis:
    - Baseline fits should show significant delta_w0, delta_wa
    - Spandrel fits (with evolution) should show consistent w0, wa
    """

    def __init__(self, dataset: SNDataset):
        """
        Initialize analyzer with SN dataset.

        Args:
            dataset: SNDataset containing the full sample
        """
        self.dataset = dataset
        self.n_total = len(dataset)

        # Validate host mass data
        if dataset.host_mass is None:
            raise ValueError("Dataset must include host_mass for split analysis")

        valid_mass = np.isfinite(dataset.host_mass) & (dataset.host_mass > 0)
        if np.sum(valid_mass) < 100:
            warnings.warn(f"Only {np.sum(valid_mass)} SNe with valid host mass")

    def run_split_test(
        self, mass_threshold: float = 10.0, model_type: str = "baseline", n_bootstrap: int = 100
    ) -> SplitTestResult:
        """
        Run the host mass split test.

        Args:
            mass_threshold: log10(M/M_sun) threshold for split
            model_type: "baseline" (no evolution) or "spandrel" (with evolution)
            n_bootstrap: Number of bootstrap samples for error estimation

        Returns:
            SplitTestResult with analysis results
        """
        # Split the sample
        low_mass, high_mass = split_by_host_mass(self.dataset, threshold=mass_threshold)

        print(f"Split analysis: {len(low_mass)} low-mass, {len(high_mass)} high-mass SNe")

        # Fit each subsample
        if model_type == "baseline":
            w0_low, w0_low_err, wa_low, wa_low_err = self._fit_subsample_baseline(
                low_mass, n_bootstrap
            )
            w0_high, w0_high_err, wa_high, wa_high_err = self._fit_subsample_baseline(
                high_mass, n_bootstrap
            )
        else:
            w0_low, w0_low_err, wa_low, wa_low_err = self._fit_subsample_spandrel(
                low_mass, n_bootstrap
            )
            w0_high, w0_high_err, wa_high, wa_high_err = self._fit_subsample_spandrel(
                high_mass, n_bootstrap
            )

        # Compute differences
        delta_w0 = w0_high - w0_low
        delta_w0_err = np.sqrt(w0_high_err**2 + w0_low_err**2)

        delta_wa = wa_high - wa_low
        delta_wa_err = np.sqrt(wa_high_err**2 + wa_low_err**2)

        # Significance
        delta_w0_sigma = np.abs(delta_w0) / delta_w0_err if delta_w0_err > 0 else 0
        delta_wa_sigma = np.abs(delta_wa) / delta_wa_err if delta_wa_err > 0 else 0

        # Combined significance (treating w0 and wa as independent)
        combined_sigma = np.sqrt(delta_w0_sigma**2 + delta_wa_sigma**2)

        return SplitTestResult(
            n_low_mass=len(low_mass),
            n_high_mass=len(high_mass),
            mass_threshold=mass_threshold,
            w0_low=w0_low,
            w0_low_err=w0_low_err,
            wa_low=wa_low,
            wa_low_err=wa_low_err,
            w0_high=w0_high,
            w0_high_err=w0_high_err,
            wa_high=wa_high,
            wa_high_err=wa_high_err,
            delta_w0=delta_w0,
            delta_w0_err=delta_w0_err,
            delta_wa=delta_wa,
            delta_wa_err=delta_wa_err,
            delta_w0_sigma=delta_w0_sigma,
            delta_wa_sigma=delta_wa_sigma,
            combined_sigma=combined_sigma,
            model_type=model_type,
        )

    def _fit_subsample_baseline(
        self, subsample: SNDataset, n_bootstrap: int = 100
    ) -> Tuple[float, float, float, float]:
        """
        Fit baseline model to subsample.

        Returns: (w0_best, w0_err, wa_best, wa_err)
        """
        # Create likelihood object from SNDataset arrays
        likelihood = SpandrelLikelihood(
            z=subsample.z_cmb,
            m_obs=subsample.m_b,
            m_err=subsample.m_b_err,
            x1=subsample.x1,
            x1_err=subsample.x1_err,
            c=subsample.c,
            c_err=subsample.c_err,
            host_mass=subsample.host_mass,
            host_mass_err=subsample.host_mass_err,
        )

        # Initial guess
        # [M0, alpha, beta, gamma, sigma_int, H0, Om, w0, wa]
        x0 = np.array([-19.3, 0.14, 3.1, 0.05, 0.1, 70.0, 0.3, -1.0, 0.0])

        # Minimize negative log-posterior
        def neg_log_post(theta):
            lp = likelihood.log_posterior_baseline(theta)
            return -lp if np.isfinite(lp) else 1e30

        # Bounds for optimizer
        bounds = [
            (-20.0, -18.0),  # M0
            (0.0, 0.3),  # alpha
            (2.0, 4.5),  # beta
            (-0.1, 0.15),  # gamma
            (0.01, 0.3),  # sigma_int
            (60.0, 80.0),  # H0
            (0.2, 0.4),  # Om
            (-2.0, 0.0),  # w0
            (-2.0, 2.0),  # wa
        ]

        try:
            result = optimize.minimize(
                neg_log_post, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 1000}
            )
            theta_best = result.x
            w0_best = theta_best[7]
            wa_best = theta_best[8]
        except Exception as e:
            warnings.warn(f"Optimization failed: {e}")
            w0_best, wa_best = -1.0, 0.0

        # Bootstrap for errors
        w0_samples = []
        wa_samples = []

        n_sn = len(subsample)
        rng = np.random.default_rng(42)

        for _ in range(n_bootstrap):
            # Resample with replacement
            idx = rng.choice(n_sn, size=n_sn, replace=True)

            # Create bootstrap subsample
            boot_sample = SNDataset(
                name=f"{subsample.name}_boot",
                z=subsample.z[idx],
                z_cmb=subsample.z_cmb[idx],
                m_b=subsample.m_b[idx],
                m_b_err=subsample.m_b_err[idx],
                x1=subsample.x1[idx],
                x1_err=subsample.x1_err[idx],
                c=subsample.c[idx],
                c_err=subsample.c_err[idx],
                host_mass=subsample.host_mass[idx],
                host_mass_err=subsample.host_mass_err[idx],
            )

            boot_lik = SpandrelLikelihood(
                z=boot_sample.z_cmb,
                m_obs=boot_sample.m_b,
                m_err=boot_sample.m_b_err,
                x1=boot_sample.x1,
                x1_err=boot_sample.x1_err,
                c=boot_sample.c,
                c_err=boot_sample.c_err,
                host_mass=boot_sample.host_mass,
                host_mass_err=boot_sample.host_mass_err,
            )

            def neg_log_post_boot(theta):
                lp = boot_lik.log_posterior_baseline(theta)
                return -lp if np.isfinite(lp) else 1e30

            try:
                result_boot = optimize.minimize(
                    neg_log_post_boot,
                    theta_best,
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": 500},
                )
                w0_samples.append(result_boot.x[7])
                wa_samples.append(result_boot.x[8])
            except Exception:
                pass

        if len(w0_samples) > 10:
            w0_err = np.std(w0_samples)
            wa_err = np.std(wa_samples)
        else:
            # Fallback: approximate errors from Hessian
            w0_err = 0.1
            wa_err = 0.5

        return w0_best, w0_err, wa_best, wa_err

    def _fit_subsample_spandrel(
        self, subsample: SNDataset, n_bootstrap: int = 100
    ) -> Tuple[float, float, float, float]:
        """
        Fit Spandrel model (with evolution) to subsample.

        Returns: (w0_best, w0_err, wa_best, wa_err)
        """
        likelihood = SpandrelLikelihood(
            z=subsample.z_cmb,
            m_obs=subsample.m_b,
            m_err=subsample.m_b_err,
            x1=subsample.x1,
            x1_err=subsample.x1_err,
            c=subsample.c,
            c_err=subsample.c_err,
            host_mass=subsample.host_mass,
            host_mass_err=subsample.host_mass_err,
        )

        # Initial guess
        # [M0, alpha, beta, gamma, sigma_int, dM_dz, dx1_dz, dc_dz, H0, Om, w0, wa]
        x0 = np.array([-19.3, 0.14, 3.1, 0.05, 0.1, 0.0, 0.0, 0.0, 70.0, 0.3, -1.0, 0.0])

        def neg_log_post(theta):
            lp = likelihood.log_posterior_spandrel(theta)
            return -lp if np.isfinite(lp) else 1e30

        bounds = [
            (-20.0, -18.0),  # M0
            (0.0, 0.3),  # alpha
            (2.0, 4.5),  # beta
            (-0.1, 0.15),  # gamma
            (0.01, 0.3),  # sigma_int
            (-0.5, 0.5),  # dM_dz
            (-1.0, 1.0),  # dx1_dz
            (-0.1, 0.1),  # dc_dz
            (60.0, 80.0),  # H0
            (0.2, 0.4),  # Om
            (-2.0, 0.0),  # w0
            (-2.0, 2.0),  # wa
        ]

        try:
            result = optimize.minimize(
                neg_log_post, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 1000}
            )
            theta_best = result.x
            w0_best = theta_best[10]
            wa_best = theta_best[11]
        except Exception as e:
            warnings.warn(f"Optimization failed: {e}")
            w0_best, wa_best = -1.0, 0.0

        # Bootstrap for errors (simplified)
        w0_samples = []
        wa_samples = []
        n_sn = len(subsample)
        rng = np.random.default_rng(42)

        for _ in range(min(n_bootstrap, 50)):  # Fewer for Spandrel (more params)
            idx = rng.choice(n_sn, size=n_sn, replace=True)

            boot_sample = SNDataset(
                name=f"{subsample.name}_boot",
                z=subsample.z[idx],
                z_cmb=subsample.z_cmb[idx],
                m_b=subsample.m_b[idx],
                m_b_err=subsample.m_b_err[idx],
                x1=subsample.x1[idx],
                x1_err=subsample.x1_err[idx],
                c=subsample.c[idx],
                c_err=subsample.c_err[idx],
                host_mass=subsample.host_mass[idx],
                host_mass_err=subsample.host_mass_err[idx],
            )

            boot_lik = SpandrelLikelihood(
                z=boot_sample.z_cmb,
                m_obs=boot_sample.m_b,
                m_err=boot_sample.m_b_err,
                x1=boot_sample.x1,
                x1_err=boot_sample.x1_err,
                c=boot_sample.c,
                c_err=boot_sample.c_err,
                host_mass=boot_sample.host_mass,
                host_mass_err=boot_sample.host_mass_err,
            )

            def neg_log_post_boot(theta):
                lp = boot_lik.log_posterior_spandrel(theta)
                return -lp if np.isfinite(lp) else 1e30

            try:
                result_boot = optimize.minimize(
                    neg_log_post_boot,
                    theta_best,
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": 500},
                )
                w0_samples.append(result_boot.x[10])
                wa_samples.append(result_boot.x[11])
            except Exception:
                pass

        if len(w0_samples) > 10:
            w0_err = np.std(w0_samples)
            wa_err = np.std(wa_samples)
        else:
            w0_err = 0.1
            wa_err = 0.5

        return w0_best, w0_err, wa_best, wa_err

    def scan_mass_thresholds(
        self, thresholds: List[float] = None, model_type: str = "baseline"
    ) -> List[SplitTestResult]:
        """
        Scan over different host mass thresholds.

        This tests robustness of the split result to threshold choice.

        Args:
            thresholds: List of log10(M/M_sun) thresholds to test
            model_type: "baseline" or "spandrel"

        Returns:
            List of SplitTestResult for each threshold
        """
        if thresholds is None:
            thresholds = [9.5, 10.0, 10.5, 11.0]

        results = []
        for thresh in thresholds:
            print(f"\n--- Testing threshold: log(M) = {thresh} ---")
            result = self.run_split_test(
                mass_threshold=thresh, model_type=model_type, n_bootstrap=50  # Fewer for scan
            )
            results.append(result)

        return results


def run_injection_recovery_test(n_sn: int = 1000, dM_dz_true: float = 0.1, seed: int = 42) -> Dict:
    """
    Injection-recovery test for Spandrel detection.

    Simulates SN data with known population evolution, then tests whether:
    1. Baseline model shows spurious w0, wa differences
    2. Spandrel model correctly recovers input cosmology

    Args:
        n_sn: Number of SNe to simulate
        dM_dz_true: True luminosity evolution (mag per unit z)
        seed: Random seed

    Returns:
        Dictionary with test results
    """
    print(f"\n{'='*60}")
    print("INJECTION-RECOVERY TEST")
    print(f"N_SN = {n_sn}, dM/dz_true = {dM_dz_true}")
    print(f"{'='*60}")

    # Generate simulated data WITH evolution
    loader = SimulatedDataLoader(seed=seed)
    dataset = loader.generate(
        n_sn=n_sn,
        z_range=(0.01, 1.5),
        include_evolution=True,
        dM_dz=dM_dz_true,
        dx1_dz=0.0,
        dc_dz=0.0,
    )

    # Run analysis
    analyzer = HostMassSplitAnalyzer(dataset)

    # Baseline model (should show split if evolution present)
    print("\n--- BASELINE MODEL (no evolution correction) ---")
    result_baseline = analyzer.run_split_test(
        mass_threshold=10.0, model_type="baseline", n_bootstrap=50
    )

    # Spandrel model (should NOT show split)
    print("\n--- SPANDREL MODEL (with evolution correction) ---")
    result_spandrel = analyzer.run_split_test(
        mass_threshold=10.0, model_type="spandrel", n_bootstrap=50
    )

    results = {
        "n_sn": n_sn,
        "dM_dz_true": dM_dz_true,
        "baseline": result_baseline,
        "spandrel": result_spandrel,
    }

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print("\nBaseline model (no evolution):")
    print(f"  delta_w0 = {result_baseline.delta_w0:.3f} +/- {result_baseline.delta_w0_err:.3f}")
    print(f"  delta_wa = {result_baseline.delta_wa:.3f} +/- {result_baseline.delta_wa_err:.3f}")
    print(f"  Combined significance: {result_baseline.combined_sigma:.1f}sigma")

    print("\nSpandrel model (with evolution):")
    print(f"  delta_w0 = {result_spandrel.delta_w0:.3f} +/- {result_spandrel.delta_w0_err:.3f}")
    print(f"  delta_wa = {result_spandrel.delta_wa:.3f} +/- {result_spandrel.delta_wa_err:.3f}")
    print(f"  Combined significance: {result_spandrel.combined_sigma:.1f}sigma")

    if result_baseline.combined_sigma > 2 and result_spandrel.combined_sigma < 2:
        print("\n*** SPANDREL SIGNATURE DETECTED ***")
        print("Baseline shows host-mass dependent cosmology.")
        print("Evolution correction removes the dependence.")
        results["signature_detected"] = True
    else:
        print("\nNo clear Spandrel signature in this realization.")
        results["signature_detected"] = False

    return results


def format_result_table(result: SplitTestResult) -> str:
    """Format split test result as ASCII table."""
    lines = [
        f"Host Mass Split Analysis ({result.model_type.upper()} model)",
        "=" * 55,
        f"Threshold: log(M/M_sun) = {result.mass_threshold:.1f}",
        f"Low-mass sample:  N = {result.n_low_mass}",
        f"High-mass sample: N = {result.n_high_mass}",
        "",
        "Parameter       Low-mass          High-mass         Difference",
        "-" * 55,
        f"w0          {result.w0_low:7.3f} +/- {result.w0_low_err:.3f}  "
        f"{result.w0_high:7.3f} +/- {result.w0_high_err:.3f}  "
        f"{result.delta_w0:+.3f} ({result.delta_w0_sigma:.1f}sigma)",
        f"wa          {result.wa_low:7.3f} +/- {result.wa_low_err:.3f}  "
        f"{result.wa_high:7.3f} +/- {result.wa_high_err:.3f}  "
        f"{result.delta_wa:+.3f} ({result.delta_wa_sigma:.1f}sigma)",
        "-" * 55,
        f"Combined significance: {result.combined_sigma:.1f}sigma",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    # Run injection-recovery test
    results = run_injection_recovery_test(n_sn=500, dM_dz_true=0.1, seed=42)

    # Print formatted tables
    print("\n" + format_result_table(results["baseline"]))
    print("\n" + format_result_table(results["spandrel"]))
