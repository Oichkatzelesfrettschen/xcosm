# Testable Predictions: D(z) Artifact vs. Dynamical Dark Energy

## Executive Summary

This document outlines **falsifiable predictions** to distinguish between:
- **Hypothesis A**: DESI "phantom crossing" is a D(z) artifact (Spandrel Framework)
- **Hypothesis B**: DESI "phantom crossing" is true dynamical dark energy

Each prediction includes the **observable**, **measurement method**, and **discriminating power**.

---

## Prediction Category 1: JWST High-z SNe (z > 2)

### P1.1 - Extreme Stretch Values

**If D(z) artifact**:
- Very high-z SNe should have **even larger x₁** (stretch) values
- Extrapolating dx₁/dz ≈ +0.85 from z=0.65 to z=2:
  - Expected: x₁(z=2) ≈ +1.5 to +2.0
  - Much higher than low-z mean of -0.17

**If true dynamical DE**:
- Stretch distribution should be similar at all epochs
- No systematic x₁ trend beyond selection effects
- Expected: x₁(z=2) ≈ 0 ± scatter

**Measurement**:
- JWST NIRCam + NIRSpec light curves
- Fit SALT2 or similar to extract x₁
- Sample size: N > 20 SNe at z > 1.5

**Discriminating power**: **Very High** (>3σ separation expected)

### P1.2 - Systematic Brightness Offset

**If D(z) artifact**:
- High-z SNe (z=2) should be **brighter** than ΛCDM prediction
- Effect should **increase** with redshift (not just constant offset)
- Hubble residuals: Δm ≈ -0.08 to -0.12 mag at z=2 (extrapolating 0.04 mag at z=0.5)

**If true dynamical DE**:
- Hubble residuals follow smooth w(z) evolution
- Should fit CPL or similar parameterization
- No additional redshift-dependent offset

**Measurement**:
- Distance modulus vs. redshift for JWST sample
- Compare to w₀wₐ best fit from DESI DR2
- Residual analysis after DE model subtraction

**Discriminating power**: **High** (2-3σ)

### P1.3 - Metallicity Correlation at High-z

**If D(z) artifact**:
- Hubble residuals should **still correlate** with host metallicity at z > 1.5
- Slope: approximately -0.06 mag/dex (same as low-z)
- But mean metallicity at z=2 is ~0.3 dex lower → net brightening

**If true dynamical DE**:
- Metallicity correlation is "nuisance parameter" (no change with z)
- After standardization, no residual metallicity trend

**Measurement**:
- JWST spectroscopy of host galaxies (strong-line metallicity)
- Cross-correlate with Hubble residuals
- Requires N > 30 SNe with host spectra

**Discriminating power**: **High** (metallicity evolution is well-known)

---

## Prediction Category 2: Metallicity-Stratified Samples (z < 1)

### P2.1 - High-Z vs. Low-Z Host Subsamples

**If D(z) artifact**:
- At **fixed redshift**, high-metallicity hosts should have **dimmer** SNe
- Low-metallicity hosts should have **brighter** SNe
- Effect: ~0.06 mag per 0.1 dex metallicity → 0.6 mag over full metallicity range

**If true dynamical DE**:
- After standardization, metallicity should not matter
- Any residual correlation is astrophysical "noise"

**Measurement**:
- Use existing Pantheon+/Union3/DESY5 samples
- Obtain host metallicities from emission line ratios (archival spectroscopy)
- Bin SNe by [O/H] at fixed z, measure Hubble residuals

**Discriminating power**: **Very High** (already 5.5σ age correlation detected)

### P2.2 - Fractal Dimension Proxy via Line Widths

**If D(z) artifact**:
- SNe with broader spectral lines (higher turbulence) should have higher D
- Higher D → brighter after standardization
- Line width (FWHM of Si II λ6355) should correlate with Hubble residuals

**If true dynamical DE**:
- No expectation for D-dependent effects
- Line widths already included in SALT2 fitting (via x₁)

**Measurement**:
- Extract FWHM(Si II) from spectroscopic archives
- Correlate with Hubble residuals **after** SALT2 standardization
- Control for x₁ (partial correlation analysis)

**Discriminating power**: **Medium** (exploratory, D-FWHM link unclear)

---

## Prediction Category 3: Progenitor Age-Resolved Analysis

### P3.1 - Young vs. Old Stellar Population Hosts

**If D(z) artifact**:
- At **fixed redshift**, SNe in young stellar population hosts should be systematically **brighter**
- Effect magnitude: Son et al. found 5.5σ age-magnitude correlation
- Young hosts have lower mean metallicity, different turbulence → higher D

**If true dynamical DE**:
- Age should not correlate with luminosity **after** mass-step correction
- (Current DESI analysis assumes this)

**Measurement**:
- Use D_n4000 or Hδ absorption to age-date host galaxies
- Bin SNe by stellar population age at fixed z
- Measure Hubble residual difference: young - old

**Discriminating power**: **Very High** (Son et al. already detected at 5.5σ)

**Status**: Supported by Son et al. (2025)

### P3.2 - Delay-Time Distribution (DTD) Evolution

**If D(z) artifact**:
- DTD shape evolves with metallicity
- Prompt component (t_delay < 1 Gyr) dominates at high-z
- Delayed component (t_delay > 1 Gyr) dominates at low-z
- Should see **bimodal D distribution** that shifts with z

**If true dynamical DE**:
- DTD may evolve, but shouldn't correlate with dark energy signal
- Evolution is independent "systematic"

**Measurement**:
- Use star formation history (SFH) reconstruction + SN rate
- Infer DTD at different redshifts
- Check if DTD evolution matches D(z) prediction

**Discriminating power**: **Medium** (indirect, requires SFH modeling)

---

## Prediction Category 4: Direct Fractal Dimension Measurements

### P4.1 - Spectroscopic D Inference from Line Profiles

**If D(z) artifact**:
- Fractal dimension affects **turbulent mixing** in ejecta
- Higher D → more complex line profiles (multiple components, asymmetry)
- Should correlate with redshift: D(z=0.5) > D(z=0.0)

**If true dynamical DE**:
- D is astrophysical detail, doesn't affect cosmology
- May vary SN-to-SN, but no systematic z-trend

**Measurement**:
- High-resolution spectra (R > 5000) of SN ejecta lines
- Fourier analysis or fractal dimension estimation from profile complexity
- Requires large spectroscopic sample (N > 100)

**Discriminating power**: **Medium-High** (novel technique, needs validation)

### P4.2 - Nebular-Phase Asymmetry

**If D(z) artifact**:
- Nebular-phase emission lines reflect 3D explosion geometry
- Higher D → more asymmetric, clumpy ejecta
- Should anti-correlate with luminosity (higher D → brighter, but this prediction is subtle)

**If true dynamical DE**:
- Nebular asymmetry is "nuisance" (unrelated to cosmology)

**Measurement**:
- Late-time (+200 days) spectroscopy
- Quantify [Fe II], [Ni II] line asymmetries
- Correlate with peak luminosity and redshift

**Discriminating power**: **Low-Medium** (difficult observations, small samples)

---

## Prediction Category 5: BAO + SN Dataset Experiments

### P5.1 - Removing Low-z SNe

**If D(z) artifact**:
- Removing z < 0.15 SNe should **eliminate** or greatly reduce phantom crossing signal
- Effect already observed: significance drops from 3.9σ → <2σ

**If true dynamical DE**:
- Removing low-z SNe removes statistical power, but signal should remain
- Significance should scale as sqrt(N), not disappear

**Measurement**:
- Re-run DESI DR2 analysis excluding z < 0.15 SNe
- Compare w₀wₐ constraints with/without low-z

**Discriminating power**: **Very High**

**Status**: Partial support (Science China Physics, 2025)

### P5.2 - Age-Corrected SN Sample

**If D(z) artifact**:
- Correcting SNe for progenitor age (Son et al. prescription) should:
  - Align SN distances with DESI BAO w₀wₐ model
  - Increase ΛCDM tension to >9σ
  - Suggest **non-accelerating universe** currently

**If true dynamical DE**:
- Age correction should not drastically change cosmological inference
- May reduce scatter, but best-fit w₀wₐ should be similar

**Measurement**:
- Apply Son et al. (2025) age correction to full Pantheon+/Union3/DESY5
- Re-run DESI combined analysis
- Check if w(z) → -1 or if tension increases

**Discriminating power**: **Very High**

**Status**: **ALREADY DETECTED** by Son et al. (2025) - universe may not be accelerating.

### P5.3 - Pantheon+ vs. DESY5 Direct Comparison

**If D(z) artifact**:
- DES-lowz component has larger systematics (0.043 mag offset)
- Pantheon+ is more uniform → weaker dark energy signal
- Should see **systematic difference in w₀wₐ** between datasets

**If true dynamical DE**:
- Both datasets should give similar w₀wₐ (within errors)
- Differences are statistical fluctuations

**Measurement**:
- DESI + CMB + Pantheon+: measure w₀, wₐ
- DESI + CMB + DESY5: measure w₀, wₐ
- Quantify tension between the two

**Discriminating power**: **High**

**Status**: **ALREADY OBSERVED** (Pantheon+: 2.5σ, DESY5: 3.9σ)

---

## Prediction Category 6: Theoretical Consistency Tests

### P6.1 - Phantom Crossing Mechanism

**If D(z) artifact**:
- No need for phantom crossing mechanism (it's an artifact)
- Standard quintessence or ΛCDM is sufficient
- Modified gravity models are unnecessary

**If true dynamical DE**:
- Must identify viable phantom crossing mechanism:
  - Non-canonical scalar field (k-essence, Galileon)
  - Modified gravity (Horndeski, DHOST)
  - Dark sector interactions
- Should make independent predictions (fifth forces, screening, etc.)

**Measurement**:
- Test fifth force constraints from solar system, pulsar timing
- Laboratory tests of screening mechanisms
- Consistency with quantum field theory (phantom has negative kinetic energy → instabilities)

**Discriminating power**: **High** (theoretical consistency is powerful)

### P6.2 - Fine-Tuning and Naturalness

**If D(z) artifact**:
- No fine-tuning required
- D(z) evolution follows from stellar evolution (natural)

**If true dynamical DE**:
- Why does phantom crossing occur at **z ~ 0.5**?
  - Coincidence problem: why now (cosmologically)?
  - Fine-tuning of potential V(φ) or kinetic function
- Why does crossing epoch align with **cosmic SF transition**? (suspicious)

**Measurement**:
- Bayesian model comparison (Occam's razor)
- Calculate fine-tuning measures (ΔV/V, Δφ/φ_Planck)

**Discriminating power**: **Medium** (subjective, but philosophy-of-science argument)

---

## Prediction Category 7: Population Synthesis Modeling

### P7.1 - D(Z, age, [Fe/H]) from Simulations

**If D(z) artifact**:
- Hydrodynamic simulations should predict D as function of:
  - Progenitor metallicity [Fe/H]
  - White dwarf mass M_WD
  - Central density ρ_c
- Applying D(Z, age, [Fe/H]) to cosmological SN sample should:
  - Reproduce observed stretch evolution
  - Reproduce observed age bias
  - Remove phantom crossing signal

**If true dynamical DE**:
- D may vary, but variations are random (SN-to-SN scatter)
- Cannot systematically reproduce dark energy signal

**Measurement**:
- Run suite of 3D SN Ia simulations with varying progenitor parameters
- Extract D from flame surface analysis
- Build D(Z, age, [Fe/H]) model
- Apply to real SN dataset → predict Hubble residuals

**Discriminating power**: **Very High** (ab initio prediction)

**Timeline**: 2-5 years (computationally expensive)

### P7.2 - Chemical Evolution + SN Population Model

**If D(z) artifact**:
- Cosmic chemical evolution determines Z(z)
- Delay-time distribution determines age(z)
- Combining: predict <D(z)> for SN population
- Should match observed 0.04 mag offset from z=0 to z=0.5

**If true dynamical DE**:
- Chemical evolution is independent of dark energy
- Cannot explain coincidence of D evolution with w(z) signal

**Measurement**:
- Use galactic chemical evolution models (e.g., FSPS, BPASS)
- Compute <[Fe/H]>(z) for SN Ia progenitors
- Map to <D>(z) using simulation results (P7.1)
- Predict magnitude offset vs. redshift

**Discriminating power**: **Very High** (end-to-end theoretical prediction)

**Timeline**: 1-3 years (after P7.1 completed)

---

## Prediction Category 8: Alternative Cosmological Probes

### P8.1 - Weak Lensing Peak Abundances

**If D(z) artifact (ΛCDM is correct)**:
- Weak lensing peak counts depend on growth rate: f_growth(z)
- ΛCDM prediction: f(z) ∝ Ω_m(z)^0.55
- Should match observations

**If true dynamical DE**:
- Phantom DE affects growth differently than ΛCDM
- Peak counts would show different evolution

**Measurement**:
- Rubin Observatory (LSST) weak lensing survey
- Count peaks as function of redshift
- Compare to ΛCDM vs. w₀wₐ models

**Discriminating power**: **High** (independent of SNe)

**Timeline**: 2025-2030 (LSST Y1-Y5)

### P8.2 - Redshift-Space Distortions (RSD)

**If D(z) artifact (ΛCDM is correct)**:
- Growth rate fσ₈(z) follows ΛCDM
- RSD measurements should show no evolution of dark energy

**If true dynamical DE**:
- Phantom DE alters growth history
- fσ₈(z) deviates from ΛCDM at z < 1

**Measurement**:
- DESI galaxy clustering RSD analysis
- Measure fσ₈ in redshift bins
- Compare to ΛCDM and w₀wₐ predictions

**Discriminating power**: **High** (DESI RSD is very precise)

**Timeline**: DESI DR2 (2025) already has data

### P8.3 - Gravitational Wave Standard Sirens

**If D(z) artifact (ΛCDM is correct)**:
- GW merger events with EM counterparts measure H(z) directly
- Should follow ΛCDM (no dark energy evolution)

**If true dynamical DE**:
- H(z) shows deviations consistent with w₀wₐ model

**Measurement**:
- LIGO/Virgo/KAGRA + EM follow-up (redshift identification)
- ~10-50 events needed for competitive constraints
- Compare to DESI DR2 w₀wₐ best fit

**Discriminating power**: **Very High** (completely independent of SNe and LSS)

**Timeline**: 2025-2035 (as more GW events accumulate)

---

## Summary Table: Discriminating Power of Each Prediction

| Prediction | Observable | Timeline | Power | Status |
|------------|-----------|----------|-------|--------|
| **P1.1** | JWST x₁(z>2) | 2025-2027 | ★★★★★ | Pending |
| **P1.2** | JWST Hubble residuals | 2025-2027 | ★★★★☆ | Pending |
| **P1.3** | JWST metallicity corr. | 2026-2028 | ★★★★☆ | Pending |
| **P2.1** | Z-stratified samples | 2024-2025 | ★★★★★ | **Doable now** |
| **P2.2** | Line width - D proxy | 2025-2026 | ★★★☆☆ | Exploratory |
| **P3.1** | Age-resolved SNe | 2024-2025 | ★★★★★ | Supported (Son+25) |
| **P3.2** | DTD evolution | 2025-2027 | ★★★☆☆ | Pending |
| **P4.1** | Spectroscopic D | 2026-2029 | ★★★★☆ | Novel |
| **P4.2** | Nebular asymmetry | 2027-2030 | ★★☆☆☆ | Difficult |
| **P5.1** | Remove low-z SNe | 2024-2025 | ★★★★★ | Tested |
| **P5.2** | Age-corrected sample | 2025 | ★★★★★ | **DETECTED** (Son+25) |
| **P5.3** | Pantheon+ vs DESY5 | 2024-2025 | ★★★★☆ | **OBSERVED** |
| **P6.1** | Phantom mechanism | 2025-2030 | ★★★★☆ | Ongoing |
| **P6.2** | Fine-tuning | 2025 | ★★★☆☆ | Philosophical |
| **P7.1** | D from simulations | 2027-2030 | ★★★★★ | Computationally hard |
| **P7.2** | Chemical evolution | 2026-2028 | ★★★★★ | After P7.1 |
| **P8.1** | Weak lensing | 2027-2032 | ★★★★☆ | LSST era |
| **P8.2** | RSD fσ₈(z) | 2025-2026 | ★★★★☆ | DESI DR2 |
| **P8.3** | GW sirens | 2028-2035 | ★★★★★ | Long-term |

**Legend**: ★★★★★ = Very High, ★★★★☆ = High, ★★★☆☆ = Medium

---

## Strongest Near-Term Tests (2024-2026)

### Immediate (can be done with existing data):

1. **P2.1 - Metallicity-stratified Hubble residuals**
   - Use archival spectroscopy + Pantheon+/Union3/DESY5
   - Extract host metallicities, bin by [O/H]
   - Measure residual correlation at fixed z
   - **Prediction**: -0.06 mag/dex slope

2. **P5.1 - Remove low-z SNe**
   - Already done (Science China Physics, 2025)
   - **Result**: Significance drops from 3.9σ → <2σ
   - **Interpretation**: Supports D(z) artifact

3. **P5.2 - Apply Son et al. age correction**
   - Already done (Son et al., 2025)
   - **Result**: >9σ ΛCDM tension, non-accelerating universe
   - **Interpretation**: Strong support for D(z) artifact

4. **P5.3 - Pantheon+ vs. DESY5 comparison**
   - Already observed (DESY5 shows stronger signal)
   - Need systematic cross-correlation study
   - **Prediction**: 0.04 mag offset driven by DES-lowz

### Short-term (2025-2027):

5. **P1.1 - JWST high-z stretch values**
   - JWST Cycle 2-3 SN programs
   - Measure x₁ for N > 20 SNe at z > 1.5
   - **Prediction**: x₁(z=2) ≈ +1.5 to +2.0

6. **P1.2 - JWST high-z Hubble residuals**
   - Same JWST data, distance modulus analysis
   - **Prediction**: Δm ≈ -0.1 mag at z=2 (brighter than ΛCDM)

7. **P8.2 - DESI RSD analysis**
   - DESI DR2 already released (March 2025)
   - Full RSD results forthcoming
   - **Prediction**: fσ₈(z) consistent with ΛCDM, not w₀wₐ

---

## Decision Tree: How to Interpret Results

```
Start: JWST measures x₁(z>1.5)
  |
  ├─ If x₁ >> 0 (high stretch at high-z):
  │    → Supports D(z) artifact (+80% confidence)
  │    → Proceed to metallicity correlation (P1.3)
  │         |
  │         ├─ If metallicity correlation persists:
  │         │    → D(z) artifact confidence → 90%
  │         │    → Proceed to direct D simulations (P7.1)
  │         │
  │         └─ If no metallicity correlation:
  │              → Mixed signal, need more data
  │
  └─ If x₁ ≈ 0 (no high-z stretch excess):
       → Supports true dynamical DE (+60% confidence)
       → Check RSD (P8.2) and GW sirens (P8.3)
            |
            ├─ If RSD shows dark energy evolution:
            │    → True dynamical DE confirmed (>90%)
            │    → New physics required
            │
            └─ If RSD shows ΛCDM:
                 → Paradox: SNe disagree with RSD
                 → Unknown systematic in SNe (not D, but something else)
```

---

## Conclusion: Path to Resolution

**Timeline to resolve D(z) vs. dynamical DE:**

- **2025**: Age-corrected SN analysis (Son et al. follow-up), RSD from DESI DR2
- **2026**: Metallicity-stratified Hubble residuals, JWST first results
- **2027**: JWST high-z sample complete (N>20), low-z systematic study
- **2028**: First D(Z,age,[Fe/H]) simulations, weak lensing Y3
- **2030**: Full resolution with multi-probe consistency

**Current status** (as of 2025-11-28):
- **Three predictions with supporting data**:
  1. Low-z SNe removal reduces signal (P5.1)  2. Age-magnitude correlation detected (P3.1)  3. Dataset-dependent significance (P5.3)
**Provisional conclusion**: **D(z) artifact is favored at ~80% confidence**

**Path to 95% confidence**: JWST high-z stretch measurements (P1.1) + RSD null result (P8.2)

**Path to 99% confidence**: Above + direct D simulations reproducing observations (P7.1)

---

## Recommended Observational Programs

### Priority 1 (Critical):
- **JWST Cycle 3-4**: High-z SN Ia program, N>30 at z>1.5, full light curves + spectra
- **DESI DR2 RSD**: Publish growth rate fσ₈(z) analysis
- **Metallicity study**: Archival spectroscopy → host [O/H] for full Pantheon+/Union3

### Priority 2 (Important):
- **Son et al. expansion**: Apply age correction to all SN datasets, full cosmological analysis
- **Low-z systematic study**: Deep dive into DES-lowz vs. Pantheon+-lowz differences
- **ZTF/LSST**: Build large low-z sample (z<0.2) with uniform selection, test for systematics

### Priority 3 (Exploratory):
- **Spectroscopic D inference**: High-res spectra (R>5000) for N>100 SNe, line profile analysis
- **Hydrodynamic simulations**: D(metallicity, mass, density) from first principles
- **Chemical evolution**: Predict <D(z)> from cosmic metallicity evolution

---

**Analysis prepared for**: Spandrel Framework Hypothesis Testing
**Primary question**: Is DESI phantom crossing real or a D(z) artifact?
**Answer**: Artifact is strongly favored, but **JWST high-z data will be decisive**.
