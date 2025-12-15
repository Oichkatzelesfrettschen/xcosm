# Pantheon+ Hubble Bifurcation Analysis Results

**Analysis Date:** 2025-11-28
**Dataset:** Pantheon+ (1701 SNe Ia)
**Hypothesis:** Hubble Bifurcation - H₀ varies with SALT3 stretch parameter x₁

## Executive Summary

This analysis tests whether the Hubble constant H₀ shows systematic variation with the SALT3 stretch parameter x₁, which maps to fractal dimension D via the Spandrel Metric framework. The hypothesis predicts:

- **LOW-D population** (x₁ < 0, D < 2.18): H₀ ≈ 75.27 km/s/Mpc
- **HIGH-D population** (x₁ > 0, D > 2.18): H₀ ≈ 71.25 km/s/Mpc
- **Expected ΔH₀**: ~4 km/s/Mpc (matching the Hubble tension)

## Key Findings

### Overall Results (z ≤ 0.1)

**LOW-D Population (x₁ < 0, D < 2.18):**
- Sample size: 428 SNe
- Mean x₁: -1.083 ± 0.716
- Mean z: 0.0274 ± 0.0184
- Mean μ: 34.80 ± 1.60 mag
- **Mean H₀: 71.66 ± 7.11 km/s/Mpc**
- Median H₀: 71.65 km/s/Mpc

**HIGH-D Population (x₁ > 0, D > 2.18):**
- Sample size: 313 SNe
- Mean x₁: 0.724 ± 0.475
- Mean z: 0.0301 ± 0.0196
- Mean μ: 35.02 ± 1.63 mag
- **Mean H₀: 71.32 ± 6.40 km/s/Mpc**
- Median H₀: 71.33 km/s/Mpc

**Observed ΔH₀: 0.34 km/s/Mpc**
(t-statistic: 0.679, p-value: 0.497 - NOT statistically significant)

### Binned Analysis

Redshift-binned results show variable ΔH₀:

| z Range | LOW-D H₀ | HIGH-D H₀ | ΔH₀ | N (LOW-D) | N (HIGH-D) |
|---------|----------|-----------|-----|-----------|------------|
| 0.01-0.02 | 70.92 ± 6.06 | 72.87 ± 6.02 | -1.95 | 109 | 45 |
| 0.02-0.03 | 73.26 ± 5.63 | 72.34 ± 5.94 | +0.92 | 104 | 99 |
| 0.03-0.05 | 70.41 ± 4.38 | 70.55 ± 4.46 | -0.14 | 100 | 77 |
| 0.05-0.07 | 70.85 ± 3.96 | 70.95 ± 7.06 | -0.10 | 28 | 24 |
| 0.07-0.10 | 70.15 ± 3.43 | 66.62 ± 4.90 | **+3.52** | 22 | 22 |
| 0.10-0.15 | 66.90 ± 3.41 | 65.93 ± 4.02 | +0.97 | 32 | 53 |

**Notable:** The 0.07-0.10 redshift bin shows ΔH₀ = +3.52 km/s/Mpc, approaching the predicted value.

### Distance Modulus Residual Analysis

Using a fiducial H₀ = 70 km/s/Mpc:

- **LOW-D residuals:** -0.041 ± 0.208 mag
- **HIGH-D residuals:** -0.032 ± 0.196 mag
- **Δμ:** -0.009 mag
- **Implied ΔH₀/H₀:** 0.42%
- **Implied ΔH₀:** 0.29 km/s/Mpc

## Full Sample Statistics (All Redshifts)

**LOW-D Population (x₁ < 0):**
- Sample size: 823 SNe
- Mean x₁: -0.917 ± 0.681
- Mean z: 0.187 ± 0.236
- Mean μ: 37.74 ± 3.41 mag

**HIGH-D Population (x₁ > 0):**
- Sample size: 878 SNe
- Mean x₁: 0.712 ± 0.462
- Mean z: 0.253 ± 0.257
- Mean μ: 38.89 ± 3.24 mag

## Interpretation

### Null Result vs. Prediction

The analysis **does not confirm** the strong Hubble bifurcation predicted by the original hypothesis:

1. **Overall ΔH₀ is small:** 0.34 km/s/Mpc vs. predicted 4.0 km/s/Mpc
2. **Not statistically significant:** p = 0.497
3. **Direction inconsistent:** Some bins show negative ΔH₀ (opposite of prediction)

### Possible Explanations

1. **Standardization Success:** The SALT3 light curve fitting already accounts for stretch-dependent variations, removing any intrinsic H₀ dependence on x₁.

2. **Sample Selection:** The Pantheon+ sample uses sophisticated corrections (bias corrections, covariance matrices) that may already correct for any systematic effects related to stretch.

3. **Redshift Dependence:** The hint of a signal in the 0.07-0.10 bin (ΔH₀ = +3.52 km/s/Mpc) suggests possible redshift-dependent effects, but small sample sizes (N=22 each) make this inconclusive.

4. **Cosmological Evolution:** At higher redshifts, the simple H₀ = cz/d_L approximation breaks down and requires proper cosmological modeling with Ω_M, Ω_Λ.

5. **Measurement Method:** The simple low-z H₀ estimation may not be sensitive enough to detect subtle population differences.

### Alternative Hypothesis Test Needed

To properly test the Spandrel Metric hypothesis, one would need to:

1. **Refit the light curves** without stretch corrections to see the "raw" x₁ dependence
2. **Use proper cosmological fitting** (e.g., MCMC with separate H₀ for each population)
3. **Focus on very low-z** (z < 0.03) where approximations are most valid
4. **Include systematic uncertainty analysis** on the standardization procedure itself

## Data Files Generated

1. **pantheon_low_d_population.csv** - Low-z (z ≤ 0.1), x₁ < 0 (428 SNe)
2. **pantheon_high_d_population.csv** - Low-z (z ≤ 0.1), x₁ > 0 (313 SNe)
3. **pantheon_all_low_d.csv** - All redshifts, x₁ < 0 (823 SNe)
4. **pantheon_all_high_d.csv** - All redshifts, x₁ > 0 (878 SNe)
5. **pantheon_bifurcation_plots.png** - Diagnostic visualization

## Conclusions

1. **Primary Finding:** The Pantheon+ data **does not show** the predicted 4 km/s/Mpc Hubble bifurcation between low-stretch and high-stretch SNe Ia.

2. **Observed Effect:** ΔH₀ ≈ 0.34 km/s/Mpc (not statistically significant)

3. **Potential Signal:** One redshift bin (0.07-0.10) shows ΔH₀ = +3.52 km/s/Mpc, but this is based on small samples and requires confirmation.

4. **Next Steps:**
   - Investigate the 0.07-0.10 redshift bin more carefully
   - Perform full cosmological parameter fitting with separate H₀ for each population
   - Examine whether SALT3 standardization removes intrinsic bifurcation
   - Consider alternative fractal dimension proxies beyond x₁

## Technical Notes

- **Method:** Direct H₀ estimation using H₀ = cz/d_L for low-z approximation
- **Distance:** d_L = 10^((μ-25)/5) Mpc
- **Redshift:** zHD (Hubble diagram redshift) from Pantheon+
- **Distance Modulus:** MU_SH0ES from Pantheon+ (includes SH0ES calibration)
- **Stretch Parameter:** SALT3 x₁ from Pantheon+

## References

- **Data Source:** Pantheon+ (Brout et al. 2022, ApJ 938, 110)
- **GitHub:** https://github.com/PantheonPlusSH0ES/DataRelease
- **VizieR:** J/ApJ/938/110
- **Sample Size:** 1701 SNe Ia (0.001 < z < 2.26)
