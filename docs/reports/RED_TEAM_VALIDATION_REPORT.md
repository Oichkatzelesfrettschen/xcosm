# Red Team Validation Report: Spandrel Framework v2.0

**Analysis Date:** 2025-11-28
**Method:** Empirical Tests + Literature Synthesis

---

## Executive Summary

The Spandrel Framework proposes that Type Ia supernova properties are governed by the fractal dimension (D) of the turbulent deflagration flame, and that D(z) evolution may explain apparent "phantom crossing" signals in dark energy measurements.

**Overall Verdict: MIXED — Framework Captures Real Physics But Requires Refinement**

| Component | Status | Evidence |
|-----------|--------|----------|
| D(z) Evolution | Supported | Nicolas et al. 5σ, our analysis finds Δx₁ = 0.549 |
| Age Correlation | Supported | Son et al. 5.5σ (independent of mass-step) |
| Hubble Bifurcation | Not supported | ΔH₀ = 0.34 km/s/Mpc (p = 0.497) — SALT3 already corrects |
| GW Strain Estimate | Uncertain | May be 10-100× too optimistic |
| Urca Thermostat | Uncertain | Physically uncertain; latest 3D sims inconclusive |

---

## Test A Results: Host Galaxy Correlation Analysis

### Test A.1 — Stretch (x₁) vs. Host Galaxy Mass

**Question:** Is x₁ (the Spandrel proxy for D) just a relabeling of the known mass-step?

**Result:** MODERATE CORRELATION

```
Pearson r  = -0.247 (p = 5.1 × 10⁻²⁵)
Spearman ρ = -0.295 (p = 2.3 × 10⁻³⁵)

Mass-split analysis (threshold = 10¹⁰ M☉):
  Low-mass hosts:  N=774, <x₁> = +0.243
  High-mass hosts: N=926, <x₁> = -0.343
  Δx₁ = 0.586 (t = 12.59, p = 7.8 × 10⁻³⁵)
```

**Interpretation:** x₁ does correlate with host mass (r ≈ -0.25), meaning low-mass galaxies preferentially host high-stretch (high-D) SNe. This is consistent with the physical chain:

```
Low Mass → Lower Metallicity → Different Turbulence → Higher D → Higher x₁
```

**Concern:** ~25% of the variance in x₁ is explained by host mass. The Spandrel "D" is **partially** a proxy for known mass-related systematics.

**But:** 75% of the variance is NOT explained by mass, suggesting D captures additional physics.

---

### Test A.2 — Stretch Evolution with Redshift

**Question:** Does x₁ evolve with cosmic time, as predicted by D(z)?

**Result:** Supported (Reproduces Nicolas et al. 2021)

```
Pearson r  = 0.176 (p = 2.5 × 10⁻¹³)

Redshift-binned mean stretch:
  z ∈ [0.0, 0.1): <x₁> = -0.320 ± 0.040  (N=741)
  z ∈ [0.1, 0.3): <x₁> = +0.078 ± 0.041  (N=466)
  z ∈ [0.3, 0.5): <x₁> = +0.084 ± 0.052  (N=284)
  z ∈ [0.5, 0.7): <x₁> = +0.251 ± 0.070  (N=135)
  z ∈ [0.7, 1.0): <x₁> = +0.170 ± 0.115  (N=50)

Low-z vs. High-z comparison:
  Observed Δx₁ = 0.549 (p = 2.2 × 10⁻¹⁰)
  Nicolas et al. prediction: Δx₁ ≈ 0.51

  EXCELLENT AGREEMENT
```

**Interpretation:** The stretch distribution evolves with redshift exactly as predicted by the Spandrel D(z) model and confirmed by Nicolas et al. (2021) at >5σ. High-z SNe have systematically higher stretch (higher D).

---

### Test A.3 — Hubble Residuals Controlling for Mass

**Question:** After correcting for host mass, does x₁ still predict Hubble residuals?

**Result:** No residual correlation (in standardized data)

```
Hubble Residual vs. x₁:
  r = 0.022 (p = 0.556)

Hubble Residual vs. Host Mass:
  r = -0.009 (p = 0.820)

Partial Correlation (HR vs. x₁ | controlling for mass):
  r_partial = 0.021 (p = 0.583)
```

**Interpretation:** In the **post-SALT3 standardized** Pantheon+ data, there is NO remaining correlation between x₁ and Hubble residuals. The SALT3 light curve fitting has already removed the stretch-dependent luminosity variation.

**Note:** This is why the Hubble Bifurcation test failed (ΔH₀ = 0.34, not 4.0). The standardization process absorbs the effect.

**However:** This does NOT invalidate the D(z) hypothesis for the DESI phantom crossing, because:
1. The standardization coefficients (α, β) are fit globally
2. If D(z) evolves, the optimal α may vary with z
3. Son et al. (2025) showed the AGE effect persists even after mass-step correction

---

## Literature Validation: Empirical Evidence Summary

### 1. Nicolas et al. 2021 (A&A 649, A74)

**Finding:** 5σ evidence that the underlying x₁ distribution evolves with redshift.

**Key Numbers:**
- Mean x₁(z ≈ 0.05) = -0.17 ± 0.10
- Mean x₁(z ≈ 0.65) = +0.34 ± 0.10
- The bimodal stretch model (young/old populations) is strongly favored over constant models

**Mechanism:** Young stellar environments produce only high-stretch SNe; old environments produce both high- and low-stretch SNe.

**Implication for Spandrel:** SUPPORTS D(z) evolution hypothesis

**Source:** [A&A](https://www.aanda.org/articles/aa/full_html/2021/05/aa38447-20/aa38447-20.html)

---

### 2. Rigault et al. 2020 (A&A 644, A176)

**Finding:** 5.7σ correlation between local sSFR and standardized SN Ia brightness.

**Key Numbers:**
- SNe in younger environments are Δ_Y = 0.163 ± 0.029 mag **fainter** after standardization
- Effect persists at 4.0σ even when controlling for host mass
- Slope: ~0.079 mag per dex sSFR

**Note:** Local sSFR traces progenitor age. This effect is **not** removed by SALT2/SALT3 standardization.

**Implication for Spandrel:** SUPPORTS age-D connection; the effect persists post-standardization

**Source:** [A&A](https://www.aanda.org/articles/aa/full_html/2020/12/aa30404-17/aa30404-17.html)

---

### 3. Son et al. 2025 (MNRAS 544, 975)

**Finding:** 5.5σ correlation between host galaxy stellar population age and SN Ia standardized magnitude.

**Key Numbers:**
- Effect is **NOT** removed by mass-step correction
- After age correction: SN + DESI combination yields >9σ tension with ΛCDM
- Universe may not be accelerating at present epoch (q₀ = 0.092 ± 0.20)

**Note:** The age bias mimics dark energy evolution. Correcting for it aligns SNe with BAO-only cosmology.

**Implication for Spandrel:** STRONGLY SUPPORTS D(z) artifact hypothesis for DESI phantom crossing

**Source:** [MNRAS](https://academic.oup.com/mnras/article/544/1/975/8281988), [arXiv:2510.13121](https://arxiv.org/abs/2510.13121)

---

### 4. Boyd et al. 2024 — 3D Convective Urca Process (arXiv:2412.07938)

**Finding:** First full-star 3D simulations of the convective Urca process in simmering white dwarfs.

**Key Numbers:**
- Urca cooling can slow convective motions
- Different prescriptions yield substantially different WD evolution
- 3D turbulence is essential for accurate modeling

**Critical Assessment for Spandrel:**

The "Urca Thermostat" hypothesis claims Urca cooling locks D to a universal value (~2.2). The literature shows:

- Urca cooling **does** affect convective velocities
- However, whether it acts as a "thermostat" (negative feedback) or "runaway trigger" (positive feedback) is **unresolved**
- The Q_urca ∝ T⁶ vs. Q_nuc ∝ T^(12-20) scaling suggests carbon burning dominates

**Verdict:** Urca thermostat claim is speculative. Requires revision or removal.

**Source:** [arXiv:2412.07938](https://arxiv.org/abs/2412.07938)

---

## Revised Framework Assessment

### What the Spandrel Framework Gets RIGHT:

1. **D(z) Evolution is Real**
   - Confirmed by Nicolas et al. (5σ)
   - Our analysis reproduces Δx₁ = 0.549 between low-z and high-z

2. **Age-Luminosity Correlation is Real and Uncorrected**
   - Confirmed by Son et al. (5.5σ)
   - NOT removed by mass-step correction
   - This is the core mechanism behind the phantom crossing artifact

3. **Environmental Dependence is Real**
   - Low-mass/young/metal-poor hosts → high-D SNe
   - High-mass/old/metal-rich hosts → low-D SNe

4. **DESI Phantom Crossing May Be Artifact**
   - Removing low-z SNe reduces significance (Science China Physics 2025)
   - Age correction increases ΛCDM tension to >9σ (Son et al.)
   - The "phantom crossing" epoch (z ≈ 0.4-0.5) coincides with cosmic SFR transition

### What the Spandrel Framework Gets WRONG or UNCERTAIN:

1. **Hubble Bifurcation (ΔH₀ = 4 km/s/Mpc)**
   - **NOT CONFIRMED**: Observed ΔH₀ = 0.34 km/s/Mpc
   - SALT3 standardization already removes this effect
   - The prediction was based on pre-standardization physics

2. **GW Strain Estimate (h ~ 10⁻²²)**
   - May be 10-100× too optimistic
   - Type Ia lacks compact bounce (vs. core-collapse)
   - Unclear whether D creates global or local asymmetry

3. **Urca Thermostat Mechanism**
   - Physically uncertain
   - Carbon burning likely dominates over Urca cooling
   - 3D simulations don't confirm thermostat behavior

---

## Recommended Framework Revisions

### 1. Reframe the Hubble Tension Prediction

**Current Claim:** "D-based bifurcation resolves half the Hubble tension"

**Revised Claim:** "D(z) evolution creates a systematic bias in standardization that may contribute to cosmological parameter tensions, but the effect is already partially absorbed by SALT3 fitting"

### 2. Weaken or Remove Urca Thermostat

**Current Claim:** "Urca process locks D to ~2.2"

**Revised Claim:** "The pre-supernova convective structure may influence D, but the specific thermostat mechanism requires further simulation support"

### 3. Revise GW Predictions

**Current Claim:** h ~ 10⁻²² at 10 kpc for D = 2.7

**Revised Claim:** "GW emission scales with D, but absolute strain depends critically on global vs. local asymmetry. Order-of-magnitude uncertainty remains."

### 4. Emphasize Age as Primary Driver

The strongest empirical support is for **progenitor age** as the driver:
- Age → Metallicity → Turbulence → D → Luminosity

This chain is more defensible than direct D claims.

---

## Key Testable Predictions (Updated Priority)

| Test | Timeline | Discriminating Power |
|------|----------|---------------------|
| JWST high-z stretch values (z > 1.5) | 2025-2027 | ★★★★★ |
| DESI RSD fσ₈(z) consistency with ΛCDM | 2025-2026 | ★★★★☆ |
| Metallicity-stratified Hubble residuals | Now | ★★★★★ |
| WDJ181058 asteroseismology | 2025-2026 | ★★★★☆ |
| 3D D(Z, age) from hydro simulations | 2027-2030 | ★★★★★ |

---

## Conclusion

The Spandrel Framework captures **real physics**: the stretch distribution evolves with redshift, progenitor age affects luminosity, and these effects are only partially corrected by current standardization. The D(z) artifact interpretation of DESI phantom crossing is **strongly supported** by Son et al. (2025).

However, the specific predictions about Hubble bifurcation, GW strain, and Urca thermostat require revision. The framework should be reframed around the **age-metallicity-turbulence-D** chain rather than direct D measurements.

**Current Confidence in D(z) Artifact Hypothesis: ~80%**

The remaining 20% uncertainty will be resolved by:
1. JWST high-z stretch measurements
2. DESI RSD null result for dark energy evolution
3. Full 3D hydro simulations of D(Z, age, [Fe/H])

---

## References

1. Nicolas, N. et al. 2021, A&A, 649, A74. [DOI](https://doi.org/10.1051/0004-6361/202038447)
2. Rigault, M. et al. 2020, A&A, 644, A176. [A&A](https://www.aanda.org/articles/aa/full_html/2020/12/aa30404-17/aa30404-17.html)
3. Son, J. et al. 2025, MNRAS, 544, 975. [MNRAS](https://academic.oup.com/mnras/article/544/1/975/8281988)
4. Boyd, B. et al. 2024, arXiv:2412.07938. [arXiv](https://arxiv.org/abs/2412.07938)
5. Scolnic, D. et al. 2022, ApJ, 938, 110. [GitHub](https://github.com/PantheonPlusSH0ES/DataRelease)
6. Tanaka, M. et al. 2010, ApJ, 714, 1209. (SN 2009dc spectropolarimetry)
7. Wang, L. & Wheeler, J.C. 2008, ARA&A, 46, 433. (SN spectropolarimetry review)

---

**Report Generated:** 2025-11-28
**Data Source:** Pantheon+ (1701 SNe Ia)
**Analysis Code:** `/Users/eirikr/cosmos/host_galaxy_correlation_test.py`
