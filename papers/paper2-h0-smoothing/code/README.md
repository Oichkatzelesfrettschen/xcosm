# H₀(R) Scale-Dependent Estimator - Code Documentation

## Overview

This directory contains the implementation of the H₀(R) scale-dependent estimator described in Paper 2. The code provides a rigorous framework for testing whether the Hubble tension reflects genuine scale-dependence in the locally-measured expansion rate.

## Core Modules

### 1. `h0_smoothing_estimator.py`

Implements the formal H₀(R) estimator with three physically-defined scale assignments:

**Key Classes:**
- `H0Measurement`: Data class for H₀ measurements with assigned scales
- `ScaleDefinitions`: Three scale definition methods:
  - **Calibration volume radius**: Spatial extent of distance ladder anchors
  - **Top-hat window radius**: Volume-averaged sample distance
  - **Survey footprint radius**: Characteristic survey scale
- `H0SmoothingEstimator`: Window-function-weighted H₀(R) computation
- `LCDMCosmicVariance`: Expected cosmic variance in ΛCDM from P(k)

**Example Usage:**
```python
from h0_smoothing_estimator import H0SmoothingEstimator, get_example_measurements

# Initialize estimator
estimator = H0SmoothingEstimator(window_function="tophat")

# Assign scales to measurements
measurements = get_example_measurements()
for info in measurements:
    meas = estimator.assign_scale_to_measurement(info, scale_definition="calibration")
    print(f"{meas.name}: R = {meas.radius_mpc:.1f} Mpc")
```

**Key Equations:**
```
H₀(R) = ⟨v(r)/r⟩_R = ∫ W(r,R) v(r)/r d³r / ∫ W(r,R) d³r

σ²_H₀(R) = (f H₀)² ∫ W²(k,R) P(k) k² dk / (2π²)
```

### 2. `lcdm_mock_generator.py`

Generates ΛCDM mock H₀(R) realizations for null hypothesis testing.

**Key Classes:**
- `VelocityFieldSampler`: Samples Gaussian random velocity fields from P(k)
- `LCDMMockGenerator`: Generates ensemble of mock H₀(R) curves
- `MockH0Curve`: Data class for individual mock realization

**Example Usage:**
```python
from lcdm_mock_generator import LCDMMockGenerator

# Generate 100 ΛCDM mocks
generator = LCDMMockGenerator(h0_fiducial=67.4, omega_m=0.315)
mocks = generator.generate_mock_h0_curves(num_mocks=100)

# Compute null statistics
stats = generator.compute_null_statistics(mocks)
print(f"ΛCDM slope distribution: {stats['slope_mean']:.4f} ± {stats['slope_std']:.4f}")
```

**Physical Basis:**
- Samples density field δ(k) from Rayleigh distribution with ⟨|δ(k)|²⟩ = P(k)
- Computes velocity via v(k) = i (f H a / k²) δ(k) k
- Window-averages to obtain H₀(R) at different scales
- Builds null distribution of slopes for hypothesis testing

### 3. `run_analysis.py`

Complete end-to-end analysis pipeline combining both modules.

**Usage:**
```bash
# Run with default settings (100 mocks)
python run_analysis.py

# Custom number of mocks and output directory
python run_analysis.py --num-mocks 1000 --output-dir ./my_results
```

**Pipeline Steps:**
1. Assign physical scales to measurements
2. Compute observed H₀(R) trend (linear fit)
3. Generate ΛCDM mock realizations
4. Compute null distribution statistics
5. Perform hypothesis test (p-value and significance)
6. Generate publication-quality figures
7. Save results to JSON

**Outputs:**
- `analysis_results.json`: Numerical results summary
- `mock_h0_ensemble.png`: Mock H₀(R) curves with percentiles
- `h0_vs_scale_comparison.png`: Observed vs ΛCDM comparison

## Scale Definitions

### Calibration Volume Radius

For distance ladder measurements, R is determined by the spatial distribution of geometric calibrators:

| Method    | Calibrators               | R_cal (Mpc) |
|-----------|---------------------------|-------------|
| Cepheid   | MW + LMC + NGC 4258      | 8.0         |
| TRGB      | ~100 galaxies < 20 Mpc   | 12.0        |
| SBF       | Early-type galaxies      | 15.0        |
| Megamaser | Geometric water masers   | 20.0        |
| CMB       | Sound horizon r_s(z_*)   | 14000.0     |

### Top-Hat Window Radius

For samples with redshifts {z_i}:
```
R_TH = (Σ w_i d_L(z_i)³ / Σ w_i)^(1/3)
```
where d_L(z) is luminosity distance and w_i are inverse-variance weights.

### Survey Footprint Radius

From survey volume:
```
R_survey = (3 V_survey / 4π)^(1/3)
```

## ΛCDM Cosmic Variance

Expected H₀ variance from large-scale structure:
```
σ²_H₀(R) = (f H₀)² ∫ dk/k W²(k,R) P(k)
```

where:
- f = Ω_m^0.55 (growth rate)
- W(k,R) = 3 j₁(kR)/(kR) (top-hat window in Fourier space)
- P(k) = Eisenstein-Hu linear power spectrum

**Typical values (ΛCDM with Ω_m=0.315, σ₈=0.81):**
- σ_H₀(R=10 Mpc)   ~ 1-2 km/s/Mpc
- σ_H₀(R=100 Mpc)  ~ 0.5-1 km/s/Mpc
- σ_H₀(R=1000 Mpc) ~ 0.1-0.3 km/s/Mpc

## Hypothesis Testing

**Null hypothesis:** Observed H₀(R) slope is consistent with ΛCDM cosmic variance.

**Alternative hypothesis:** Slope exceeds ΛCDM expectations → scale-dependent expansion.

**Test statistic:** Linear regression slope m from H₀(R) = a + m × log₁₀(R)

**p-value:**
```
p = (1/N_mock) Σ 𝟙(|m_mock| ≥ |m_obs|)
```

**Interpretation:**
- p < 0.01: Detection (trend exceeds ΛCDM)
- 0.01 ≤ p < 0.05: Marginal evidence
- p ≥ 0.05: Null result (consistent with ΛCDM)

## Example Results

Running `python run_analysis.py --num-mocks 100`:

```
Observed slope:       -1.0400 km/s/Mpc/decade
ΛCDM expectation:      0.0023 ± 0.0450
Significance:         23.16σ
p-value:              0.0000

*** DETECTION: Trend exceeds ΛCDM expectations ***
```

**Note:** This is preliminary - actual results depend on:
1. Proper power spectrum normalization
2. Realistic survey geometry
3. Measurement correlations (covariance matrix)
4. Sample variance corrections

## Falsification Criteria

| Test | Outcome | Interpretation |
|------|---------|----------------|
| ΛCDM mock comparison | p > 0.05 | No detection; consistent with cosmic variance |
| Homogeneous dataset | No trend | Heterogeneous systematics caused apparent trend |
| R-definition sensitivity | Results change significantly | Artifact of definition choice |

## Dependencies

```
numpy >= 1.20
scipy >= 1.7
matplotlib >= 3.4
```

## References

1. **Wu & Huterer (2017)** - "Sample variance in the local measurements of the Hubble constant", MNRAS, 471, 4946
2. **Kenworthy et al. (2019)** - "The local perspective on the Hubble tension", ApJ, 875, 145
3. **Riess et al. (2024)** - SH0ES Distance Ladder
4. **Freedman et al. (2024)** - CCHP TRGB measurements

## Citation

If you use this code, please cite:

```bibtex
@article{paper2_h0_smoothing,
  title={Local Expansion Rate as a Function of Smoothing Scale:
         A Physically Defined H₀(R) Estimator and Null Tests},
  author={[Authors]},
  journal={[Journal]},
  year={2025},
  note={In preparation}
}
```

## Contact

For questions or issues, please open an issue in the COSMOS repository.

---

**Last updated:** December 2025
