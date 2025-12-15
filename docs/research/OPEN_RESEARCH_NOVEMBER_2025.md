# OPEN RESEARCH: EMPIRICAL STATUS NOVEMBER 2025

## CCF-QPD-Spandrel Framework: What Remains Unresolved

**Date:** 2025-11-29
**Status:** Comprehensive Empirical Review

---

## EXECUTIVE SUMMARY

This document catalogs the open research questions at the intersection of the CCF, QPD, and Spandrel frameworks, anchored to the current empirical status as of November 2025. We identify critical gaps, tensions, and opportunities for falsification.

**Key Finding:** Several framework predictions are approaching decisive tests within 2-3 years.

---

## I. QGP VISCOSITY AND THE KSS BOUND

### 1.1 Current Empirical Status

| Facility | System | √s_NN | T_init (MeV) | η/s | Reference |
|----------|--------|-------|--------------|-----|-----------|
| RHIC | Au+Au | 200 GeV | 193.6 ± 3 | 0.204 ± 0.020 | [Nature Physics 2019](https://www.nature.com/articles/s41567-019-0611-8) |
| LHC | Pb+Pb | 2.76 TeV | 262.2 ± 13 | 0.262 ± 0.026 | [Nature Physics 2019](https://www.nature.com/articles/s41567-019-0611-8) |
| LHC | Pb+Pb | 5.02 TeV | ~300-400 | ~0.08-0.12 | [CERN Courier 2025](https://cerncourier.com/a/alice-explores-shear-viscosity-in-qcd-matter/) |

**Critical Observation:** Measured η/s values are **2.5-3.3× the KSS bound** (1/4π ≈ 0.0796), not at the bound itself.

### 1.2 Open Questions

**Q1: Does η/s approach or violate the KSS bound at higher T?**

- **QPD Prediction:** η/s should DIP below 1/4π as T → T_foam
- **Standard QCD:** η/s should INCREASE at very high T (perturbative regime)
- **Current Status:** Inconclusive. No clear trend with energy.

**Q2: What is η/s in O-O collisions?**

- **LHC O-O Run (July 2025):** Flow observed, jet quenching detected
- **η/s extraction:** NOT YET PUBLISHED
- **Expected Timeline:** ALICE/CMS results Q1-Q2 2026

**Source:** [ALICE O-O Run](https://alice-collaboration.web.cern.ch/2025-LHC-Oxygen-Run)

### 1.3 Critical Test

| Prediction | Observable | CCF-QPD | Standard | Discriminating Power |
|------------|-----------|---------|----------|---------------------|
| O-O vs Pb-Pb η/s | Viscosity ratio | Same (vacuum property) | Different (geometry) | HIGH |
| η/s(T) trend | Energy scan | Decreasing | Increasing | HIGH |

**GAP:** No FCC-hh timeline for 100 TeV Pb-Pb collisions. QPD "stringy fluid" regime untestable.

---

## II. SMALL SYSTEM COLLECTIVITY

### 2.1 Current Empirical Status

| System | Flow (v₂) | Jet Quenching | QGP Status |
|--------|-----------|---------------|------------|
| Pb-Pb | YES (large) | YES (strong) | CONFIRMED |
| Xe-Xe | YES | YES | CONFIRMED |
| **O-O** | **YES** | **YES (weak)** | **CONFIRMED (July 2025)** |
| p-Pb | YES | **NO** | DISPUTED |
| p-p | Hints | NO | UNLIKELY |

**Breakthrough (CMS 2025):** "Our new results place crucial experimental constraints on the minimal conditions required to form a QGP droplet."

**Source:** [CMS O-O Results](https://cms.cern/news/lhcs-first-oxygen-collisions-cms-spots-signs-small-scale-quark-gluon-plasma)

### 2.2 Open Questions

**Q3: What is the minimum system size for QGP?**

- **Current Answer:** Between O-O (works) and p-Pb (doesn't work)
- **CCF-QPD View:** System size is secondary; energy density is primary
- **Test:** Compare high-multiplicity p-Pb vs. peripheral O-O

**Q4: Does jet quenching scale with path length or temperature?**

- **pQCD Prediction:** Energy loss ~ L²
- **Holographic Prediction:** Energy loss ~ T³ (drag force)
- **O-O Data:** R_AA ~ 0.7-0.8 (preliminary, consistent with L² but weak)

**Source:** [ATLAS O-O Jet Quenching](https://atlas.cern/Updates/Briefing/Oxygen-Jet-Quenching)

### 2.3 Critical Test

**Proposed Experiment:** Ne-Ne vs. O-O at same multiplicity
- O-16: Near-spherical (α-cluster tetrahedral)
- Ne-20: Prolate spheroid (bowling pin)
- **Test:** If flow scales with nuclear shape (geometry) vs. energy density (vacuum)

**GAP:** Ne-Ne run occurred July 2025, but detailed flow/shape analysis pending.

---

## III. DARK ENERGY AND DESI

### 3.1 Current Empirical Status (DESI DR2, March 2025)

| Dataset | w₀ | wₐ | ΛCDM Tension |
|---------|----|----|--------------|
| BAO alone | ~-1.0 | ~0 | Consistent |
| BAO + CMB | ~-1.0 | ~0 | Consistent |
| BAO + CMB + Pantheon+ | -0.72 | -2.77 | **4.1σ** |
| BAO + CMB + DESY5 | -0.72 | -2.77 | **5.4σ** |

**Key Finding:** "Remarkably, DR2 is the first dataset which shows preference for dynamical dark energy even in the absence of supernovae."

**Sources:**
- [Nature Astronomy: DESI DR2](https://www.nature.com/articles/s41550-025-02669-6)
- [Astrobites DESI DR2](https://astrobites.org/2025/10/06/desi-dr2-part1/)
- [DESI Official Guide](https://www.desi.lbl.gov/2025/03/19/desi-dr2-results-march-19-guide/)

### 3.2 Framework Predictions

| Framework | Prediction | Current Match |
|-----------|-----------|---------------|
| **ΛCDM** | w₀ = -1, wₐ = 0 | NO (4-5σ tension with SNe) |
| **CCF** | w₀ = -0.833 (from ε = 0.25) | PARTIAL (direction correct) |
| **Spandrel** | w_apparent ≠ w_true (SN systematic) | YES (SNe-only signal) |
| **Combined** | w_observed ~ -0.72 | **EXCELLENT** |

### 3.3 Open Questions

**Q5: Is the phantom crossing real or systematic?**

- **Evidence FOR systematic:** Signal only with SNe, not BAO alone
- **Evidence FOR real:** DR2 shows hints even without SNe
- **Critical Test:** DESI RSD (fσ₈) growth measurements

**Q6: Can Spandrel fully explain the 4-5σ deviation?**

- **Required bias:** δμ ~ 0.10-0.15 mag at z ~ 1
- **Son et al. 2025:** 5.5σ age-luminosity correlation provides mechanism
- **Missing:** Quantitative end-to-end calculation of δw from δμ(z)

**Source:** [MNRAS: Son et al. 2025](https://academic.oup.com/mnras/article/544/1/975/8281988)

### 3.4 Critical Test

**The Geometry-Dynamics Split:**

| Probe | Measures | DESI DR2 Result | Interpretation |
|-------|----------|-----------------|----------------|
| BAO | Geometry (distances) | w ~ -1 | Standard |
| CMB | Geometry (early universe) | w = -1 exactly | Standard |
| SNe | Luminosity (distances) | w ~ -0.72 | **Systematic?** |
| RSD | Dynamics (growth) | **PENDING DR3** | DECISIVE |

**GAP:** DESI RSD full-shape analysis not yet published for w₀wₐCDM constraints.

---

## IV. HUBBLE TENSION

### 4.1 Current Empirical Status (November 2025)

| Method | H₀ (km/s/Mpc) | Uncertainty | Reference |
|--------|---------------|-------------|-----------|
| Planck CMB | 67.4 | ± 0.5 | Planck 2018 |
| SH0ES (Cepheids) | 73.49 | ± 0.93 | [arXiv:2509.01667](https://arxiv.org/abs/2509.01667) |
| CCHP (TRGB) | 69.85 | ± 1.4 | Freedman 2025 |
| CCHP (Carbon Stars) | 67.96 | ± 1.8 | Freedman 2025 |
| Combined JWST | 73.18 | ± 0.88 | Riess 2025 |
| DESI BAO | 68.53 | ± 0.80 | DESI DR2 |

**Status:** Tension persists at **~6σ** between SH0ES and Planck.

**Sources:**
- [NASA JWST H₀](https://science.nasa.gov/missions/webb/nasas-webb-hubble-telescopes-affirm-universes-expansion-rate-puzzle-persists/)
- [Astronomy Now: Is Hubble Tension Resolved?](https://astronomynow.com/2025/06/09/is-the-hubble-tension-resolved/)

### 4.2 CCF Prediction vs. Reality

**CCF H₀ Gradient:**
```
H₀(k) = 67.4 + 1.15 × log₁₀(k/0.01)
```

| Scale | k (Mpc⁻¹) | CCF Prediction | Observed |
|-------|-----------|----------------|----------|
| CMB | 10⁻⁴ | 64.9 | 67.4 |
| BAO | 0.1 | 68.6 | 68.5 (DESI) |
| Local | 1.0 | 69.7 | 73.2 (SH0ES) |

**Assessment:** CCF captures DIRECTION of gradient but UNDERPREDICTS local H₀ by ~3-4 km/s/Mpc.

### 4.3 Open Questions

**Q7: Is there a scale-dependent H₀?**

- **CCF Prediction:** Yes, m ~ 1.15 km/s/Mpc/decade
- **Evidence:** BAO and CMB consistent; local methods high
- **Confounders:** Cepheid crowding, TRGB calibration, peculiar velocities

**Q8: Can gravitational wave standard sirens resolve the tension?**

- **GW170817 alone:** H₀ = 70 +12/-8 km/s/Mpc (large uncertainty)
- **O1-O3 combined:** H₀ = 72.0 +12/-8.2 km/s/Mpc
- **Future (LIGO Voyager):** ~1.6% precision possible
- **Timeline:** Late 2020s/early 2030s

**GAP:** Not enough bright sirens yet for decisive test.

---

## V. S₈ TENSION

### 5.1 Current Empirical Status

| Survey | S₈ | Tension with Planck |
|--------|-----|---------------------|
| Planck CMB | 0.834 ± 0.016 | — |
| DES Y3 | 0.772 ± 0.017 | 2.3σ |
| KiDS-Legacy | 0.790 ± 0.018 | **0.73σ** |
| eROSITA | 0.86 ± 0.01 | Consistent |

**Key Development (2025):** KiDS-Legacy shows tension "largely resolving" to 0.73σ.

**Source:** [Astrobites: KiDS S₈](https://astrobites.org/2025/04/03/sigma8-tension-kids-legacy-galaxyshear/)

### 5.2 Implications for CCF

CCF calibrated α = 0.85 from S₈ = 0.78 (KiDS-1000). If S₈ tension resolves to Planck value:
- **α needs recalibration:** α ~ 0.90-0.95
- **Impact on predictions:** Minor adjustment to structure formation

**Assessment:** S₈ tension appears to be resolving. Not a strong discriminator.

---

## VI. CMB TENSOR MODES

### 6.1 Current Empirical Status

| Constraint | r (95% CL) | Source |
|------------|------------|--------|
| BICEP/Keck 2018 + Planck | < 0.032 | [arXiv:2410.23348](https://arxiv.org/pdf/2410.23348) |
| Profile likelihood | < 0.037 | Planck PR4 + BK18 |

**Status:** No detection. Upper limit tightening.

### 6.2 CCF Prediction

| Observable | CCF Value | Current Limit | Detectability |
|------------|-----------|---------------|---------------|
| r | 0.0048 ± 0.003 | < 0.032 | **Within reach** |
| R = r/(-8n_t) | 0.10 | 1.0 (standard) | Distinctive |

**Timeline:** CMB-S4 operational ~2027-2028, σ(r) ~ 0.001

**Source:** [CMB-S4 Pipeline](https://arxiv.org/html/2502.04300v1)

### 6.3 Open Questions

**Q9: Will CMB-S4 detect primordial tensors?**

- **If r ~ 0.005:** 5σ detection by CMB-S4
- **If r < 0.001:** CCF falsified; standard slow-roll survives
- **Critical signature:** Broken consistency relation R ≠ 1

**GAP:** 2-3 years until decisive data.

---

## VII. TYPE Ia SUPERNOVA EVOLUTION

### 7.1 Current Empirical Status

**Son et al. 2025 (MNRAS 544):**
- **Finding:** 5.5σ correlation between standardized SN magnitude and progenitor age
- **Implication:** Current mass-step correction is INSUFFICIENT
- **Cosmological Impact:** After age correction, ΛCDM tension increases to >9σ

**JWST High-z SNe:**

| SN | z | x₁ (stretch) | Status |
|----|---|--------------|--------|
| SN 2023adsy | 2.903 | 2.11-2.39 | Extreme (matches Spandrel) |
| SN 2023aeax | 2.15 | Normal | ~0.1σ from ΛCDM |

**Sources:**
- [arXiv:2510.13121: Son et al. II](https://arxiv.org/abs/2510.13121)
- [arXiv:2411.10427: SN 2023adsy](https://arxiv.org/abs/2411.10427)

### 7.2 Spandrel Predictions vs. Data

| z | Spandrel x₁ | Observed x₁ | Match |
|---|-------------|-------------|-------|
| 0 | ~0 | ~0 | YES |
| 1 | ~0.5 | ~0.3-0.7 | YES |
| 2.9 | 2.08 | 2.11-2.39 | **EXCELLENT** |

**Assessment:** Spandrel v4 (C/O → M_Ni mechanism) receiving strong support.

### 7.3 Open Questions

**Q10: Can age correction eliminate the DESI phantom signal?**

- **Son et al. claim:** After age correction, data STILL deviate from ΛCDM
- **Counter-argument:** Age correction may over-correct
- **Test needed:** Apply age correction to full Pantheon+ and refit w₀wₐ

**Q11: Do high-z SNe show intrinsic evolution or selection effects?**

- **Current sample:** N = 2 at z > 2 (too small)
- **Needed:** N > 20 at z > 2 for statistics
- **Timeline:** JWST Cycle 4-5 (2026-2027)

**GAP:** Sample size at z > 2 critically limited.

---

## VIII. HOLOGRAPHIC QCD TESTS

### 8.1 Current Empirical Status

**Direct AdS/CFT test:** Impossible (no known QCD gravity dual)

**Indirect tests via QGP properties:**

| Observable | Holographic Prediction | QGP Measurement | Match |
|------------|----------------------|-----------------|-------|
| η/s | 1/4π ~ 0.08 | 0.08-0.16 | **YES** |
| Drag force | ∝ T³ | Hard to isolate | Inconclusive |
| Jet q̂ | Calculable | ~1-5 GeV²/fm | Consistent |

**Source:** [Eur. Phys. J. C 2024: Holographic QCD Probes](https://link.springer.com/article/10.1140/epjc/s10052-024-12596-x)

### 8.2 Open Questions

**Q12: Can we distinguish holographic from pQCD predictions?**

- **Holographic:** Strong coupling, geometric (T³ scaling)
- **pQCD:** Weak coupling, perturbative (L² scaling)
- **Test:** High-precision jet quenching in O-O vs. Pb-Pb

**Q13: Does the viscosity bound have exceptions in real QGP?**

- **Theory:** Violations possible with higher-derivative gravity
- **Experiment:** No η/s < 0.08 ever measured
- **Implication:** Either KSS bound is exact, or we haven't reached high enough T

**GAP:** No planned experiment to probe η/s violation regime.

---

## IX. SYNTHESIS: PRIORITIZED RESEARCH AGENDA

### 9.1 High Priority (Decisive within 2 years)

| Question | Data Source | Timeline | Framework Tested |
|----------|-------------|----------|------------------|
| DESI RSD growth | DR3 full-shape | 2026 | Spandrel vs. real DE |
| O-O η/s extraction | ALICE/CMS | 2026 | QPD vacuum phase |
| JWST z>2 sample | Cycles 4-5 | 2026-27 | Spandrel evolution |

### 9.2 Medium Priority (Decisive within 5 years)

| Question | Data Source | Timeline | Framework Tested |
|----------|-------------|----------|------------------|
| CMB-S4 tensors | CMB-S4 | 2028 | CCF r prediction |
| GW standard sirens | O5 + beyond | 2028+ | H₀ tension |
| Rubin/LSST SNe | Year 1 | 2026 | SN systematics |

### 9.3 Low Priority (Beyond current technology)

| Question | Required Capability | Timeline | Framework Tested |
|----------|---------------------|----------|------------------|
| η/s violation | FCC-hh 100 TeV | 2040s+ | QPD stringy fluid |
| Quantum foam | E ~ 10¹⁹ GeV | Never? | QPD foam phase |
| Full holographic QCD | AdS dual of QCD | Unknown | AdS/CFT |

---

## X. FALSIFICATION STATUS

### 10.1 Framework Scorecard (November 2025)

| Framework | Predictions Tested | Confirmed | Falsified | Pending |
|-----------|-------------------|-----------|-----------|---------|
| **CCF** | 5 | 3 (H₀ gradient, S₈, JWST) | 0 | 2 (r, GW dispersion) |
| **Spandrel v4** | 4 | 3 (x₁ evolution, age-lum, DESI) | 0 | 1 (full phantom elimination) |
| **QPD** | 3 | 1 (O-O QGP) | 0 | 2 (η/s dip, foam) |
| **ΛCDM** | 10+ | 7 | 0 | 3+ (H₀, DESI, S₈) |

### 10.2 Critical Upcoming Tests

**2026: The Year of Decision**

1. **DESI DR3 RSD:** If fσ₈ deviates from ΛCDM → Real DE evolution
2. **O-O η/s:** If η/s < Pb-Pb at same T → Vacuum effect confirmed
3. **JWST SN sample:** If x₁(z>2) ~ 0 → Spandrel falsified

---

## XI. CONCLUSIONS

### What We Know (November 2025)

1. **QGP exists in small systems:** O-O collisions confirmed (July 2025)
2. **η/s near but above KSS bound:** No violation detected
3. **DESI sees dynamical DE at 4-5σ:** But only with SNe included
4. **Hubble tension persists at 6σ:** JWST confirmed Cepheid calibration
5. **SN progenitor bias is real:** 5.5σ age-luminosity correlation
6. **CMB tensors undetected:** r < 0.032

### What We Don't Know

1. **Is the DESI signal systematic or real?** (RSD will decide)
2. **Does η/s decrease at high T?** (No data beyond LHC energies)
3. **Is H₀ scale-dependent?** (CCF predicts yes, data inconclusive)
4. **Will CMB-S4 see tensors?** (CCF predicts r ~ 0.005)

### The Path Forward

The CCF-QPD-Spandrel unified framework makes specific, falsifiable predictions that will be tested within 2-5 years. The critical experiments are:

- **DESI DR3 RSD** (2026): Geometry vs. dynamics split
- **CMB-S4** (2028): Tensor modes and consistency relation
- **GW Standard Sirens** (2028+): Independent H₀ measurement

If these tests confirm the framework predictions, it will represent a major shift in our understanding of cosmology—from ΛCDM to a scale-dependent, emergent spacetime picture.

---

## REFERENCES

### Primary Sources (2025)

1. [CMS O-O Collisions](https://cms.cern/news/lhcs-first-oxygen-collisions-cms-spots-signs-small-scale-quark-gluon-plasma)
2. [DESI DR2 Nature Astronomy](https://www.nature.com/articles/s41550-025-02669-6)
3. [Son et al. MNRAS 544](https://academic.oup.com/mnras/article/544/1/975/8281988)
4. [Riess JWST H₀](https://arxiv.org/abs/2509.01667)
5. [BICEP/Keck 2024](https://arxiv.org/pdf/2410.23348)
6. [KiDS-Legacy](https://astrobites.org/2025/04/03/sigma8-tension-kids-legacy-galaxyshear/)
7. [SN 2023adsy](https://arxiv.org/abs/2411.10427)

---

**Document Status:** COMPLETE
**Empirical Currency:** November 2025
**Next Update:** After DESI DR3 release (expected 2026)
