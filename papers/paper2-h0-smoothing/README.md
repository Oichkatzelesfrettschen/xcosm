# Paper 2: Scale-Dependent H₀ Methodology

**Status:** Methodology Development Phase

## Title (Working)
"Local Expansion Rate as a Function of Smoothing Scale: A Physically Defined H₀(R) Estimator and Null Tests"

## Core Hypothesis
The Hubble tension reflects genuine scale-dependence in the locally-inferred expansion rate, detectable when measurements are analyzed as a function of a physically-defined smoothing scale R.

## Critical Problem Being Fixed
**Previous approach:** Heuristic k-values assigned to heterogeneous measurements.
**Problem:** No physical justification; "4.7σ detection" is not scientifically interpretable.
**Solution:** Define H₀(R) where R is an explicit, unambiguous smoothing scale.

## Directory Structure
```
paper2-h0-smoothing/
├── README.md                    # This file
├── manuscript/
│   ├── h0_smoothing.tex         # Main LaTeX document
│   ├── h0_smoothing.bib         # Bibliography
│   └── figures/                 # Paper figures
├── code/
│   ├── h0_smoothing_scale.py    # R definition and assignment
│   ├── lcdm_h0_mocks.py         # ΛCDM null distribution
│   ├── h0_regression.py         # Regression analysis
│   └── run_analysis.py          # Main analysis script
├── data/
│   ├── distance_ladder/         # Cepheid, TRGB, SBF data
│   ├── bao/                     # BAO measurements
│   └── cmb/                     # CMB-derived H₀
└── figures/
    └── generated/               # Analysis output figures
```

## Scale Definition Options

### Option A: Calibration Volume Radius
```python
def R_calibration_volume(method):
    """Effective radius of distance ladder calibration anchors."""
    if method == "Cepheid":
        # Effective radius of LMC + NGC 4258 + MW Cepheids
        return 10  # Mpc
    elif method == "TRGB":
        # Effective radius of TRGB calibrators
        return 15  # Mpc
    elif method == "CMB":
        # Sound horizon scale
        return 150  # Mpc (comoving)
    # ... etc
```

### Option B: Top-Hat Window
```python
def R_tophat(z_sample, method):
    """Top-hat radius containing sample volume."""
    if method == "local_SN":
        # Volume containing local SNe Ia
        return compute_effective_radius(z_sample)
    # ... etc
```

### Option C: Survey Footprint
```python
def R_survey(survey):
    """Characteristic scale of survey volume."""
    volumes = {
        "SH0ES": 40,      # Mpc
        "CCHP": 20,       # Mpc
        "H0LiCOW": 1000,  # Mpc (lensing)
        "Planck": 14000,  # Mpc (last scattering)
    }
    return volumes.get(survey)
```

## ΛCDM Null Distribution

The key test: does H₀(R) trend exceed ΛCDM cosmic variance expectations?

```python
def generate_lcdm_mocks(n_mocks=1000):
    """Generate ΛCDM mock H₀(R) realizations."""
    for i in range(n_mocks):
        # Sample local velocity field
        v_local = sample_velocity_field(P_k_lcdm)

        # Compute H₀ at each R scale
        for R in R_values:
            H0_mock[i, R] = compute_local_H0(v_local, R)

    return H0_mock  # Shape: (n_mocks, n_R)

def compute_null_p_value(H0_obs, H0_mocks):
    """P-value for observed trend against null."""
    slope_obs = fit_slope(H0_obs)
    slope_null = np.array([fit_slope(m) for m in H0_mocks])
    p_value = np.mean(np.abs(slope_null) >= np.abs(slope_obs))
    return p_value
```

## Falsification Criteria

| Test | Outcome | Interpretation |
|------|---------|----------------|
| ΛCDM mock comparison | p > 0.05 | No detection; trend consistent with cosmic variance |
| Homogeneous dataset | No trend | Heterogeneous systematics were the cause |
| R-definition sensitivity | Results change | Artifact of definition choice |

## What This Paper Removes from COSMOS

- [x] "4.7σ detection" language
- [x] Heuristic k-value assignments
- [x] Claims of "significant" scale-dependence
- [x] Any language implying detection before null test

## Deliverables Checklist

- [ ] Formal R definition document
- [ ] R assignment for each H₀ measurement method
- [ ] ΛCDM mock generator
- [ ] Null distribution computation
- [ ] P-value for observed trend
- [ ] Homogeneous dataset test (if data available)
- [ ] Paper draft
- [ ] Reproducibility notebook

## Key References

1. Planck Collaboration 2020 - CMB H₀
2. Riess et al. 2024 - SH0ES Cepheids
3. Freedman et al. 2024 - TRGB
4. Pesce et al. 2020 - Megamasers
5. Wu & Huterer 2017 - Local H₀ variance

## Success Criteria
1. Physically unambiguous R definition for each method
2. ΛCDM null distribution computed with proper cosmic variance
3. Either: p < 0.01 (detection), OR honest null result reported
4. Results do not depend sensitively on R definition details

---
*Last updated: December 2025*
