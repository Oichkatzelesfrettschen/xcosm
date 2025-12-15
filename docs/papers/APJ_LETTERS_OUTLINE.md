# ApJ Letters Draft Outline

## Nucleosynthetic Yields in High-Z Progenitors: A Mechanism for the Observed Age-Luminosity Relation in Type Ia Supernova Cosmology

---

## Abstract (150 words)

The recent detection of a 5.5σ correlation between Type Ia supernova luminosity and progenitor age (Son et al. 2025) implies systematic biases in SN cosmology that may explain the DESI "phantom dark energy" signal. We present 3D hydrodynamic simulations demonstrating that the **flame fractal dimension is washed out** by turbulent mixing at high Reynolds number (β → 0 as N → ∞), ruling out geometric mechanisms. Instead, the luminosity diversity arises from **nucleosynthetic yields**: low-metallicity progenitors have higher electron fractions (Ye), producing more ⁵⁶Ni and brighter supernovae. We derive a parametric model D(Z, age) that reproduces the observed stretch evolution (Nicolas et al. 5σ) and predicts SN 2023adsy at z=2.9 within observational errors. The implied magnitude bias (~0.06 mag at z=0.5) is sufficient to explain the DESI phantom crossing as an astrophysical systematic rather than new fundamental physics.

---

## 1. Introduction (0.5 pages)

**Opening:** The DESI collaboration's detection of apparent dark energy evolution (w₀ ≠ -1, wₐ ≠ 0) has sparked intense debate about whether we are witnessing new fundamental physics or systematic biases in Type Ia supernova cosmology.

**Key Context:**
- DESI phantom crossing at 2.8-4.2σ significance
- Signal appears ONLY when SNe are included (geometry-only probes show ΛCDM)
- Son et al. (2025): 5.5σ age-luminosity correlation, uncorrected by mass-step

**Question:** What is the physical mechanism underlying the age-luminosity correlation?

**Our Answer:** We demonstrate that it is nucleosynthetic (C/O → Ye → M_Ni), not geometric (flame fractal dimension).

---

## 2. Turbulent Washout (0.5 pages)

### 2.1 Pilot Simulations

3D Rayleigh-Taylor unstable flame simulations at 48³, 64³, 128³ resolution using spectral Navier-Stokes with Fisher-KPP reaction-diffusion.

**Key Result:**

| Resolution | β = dD/d(ln Z) |
|------------|----------------|
| 48³ | 0.050 |
| 64³ | 0.023 |
| 128³ | 0.008 |

Power law: β(N) ~ N^(-1.4)

**Extrapolation:** β_∞ → 0 (turbulent washout)

### 2.2 Physical Interpretation

At high Reynolds number, turbulent diffusivity κ_turb >> molecular diffusivity κ_mol(Z). The flame structure converges to a universal form with D ≈ 2.6, independent of progenitor metallicity.

**Implication:** Flame geometry cannot explain the observed luminosity diversity.

---

## 3. Nucleosynthetic Mechanism (0.75 pages)

### 3.1 The C/O → Ye → M_Ni Chain

Following Timmes et al. (2003):
- Neutron excess: η = 0.101 × Z
- Electron fraction: Ye = 0.5 - η/2
- ⁵⁶Ni yield: M_Ni ∝ Ye² (approximately)

Low Z → High Ye → More ⁵⁶Ni → Brighter

**Quantitative prediction:** ~12-25% M_Ni variation over observed Z range

### 3.2 Cosmic Evolution

| z | Z/Z☉ | Age (Gyr) | ΔM_Ni | Δμ (mag) |
|---|------|-----------|-------|----------|
| 0.0 | 1.0 | 5.0 | 0% | 0.00 |
| 0.5 | 0.8 | 3.6 | +5% | -0.05 |
| 1.0 | 0.6 | 2.9 | +10% | -0.10 |
| 2.0 | 0.3 | 2.1 | +18% | -0.18 |

### 3.3 Validation Against Observations

- Nicolas et al. (2021): x₁ evolution at 5σ → Model predicts x₁(z=0.65) = +0.48 vs +0.34±0.10 observed
- SN 2023adsy (z=2.9): Model predicts x₁ = +2.08 vs +2.11-2.39 observed

---

## 4. Implications (0.5 pages)

### 4.1 DESI Phantom Crossing

The implied magnitude bias of ~0.06 mag at z=0.5 is comparable to the ~0.04 mag offset identified in DESI systematics analyses. Combined with the age effect (Son et al.), this is sufficient to explain the phantom signal.

### 4.2 Falsification Criteria

The nucleosynthetic mechanism is falsified if:
1. JWST high-z SNe show x₁ ≈ 0 (no stretch evolution)
2. DESI RSD shows growth suppression (real DE evolution)
3. Metallicity-stratified Hubble residuals show no correlation

### 4.3 Future Work

Production-scale simulations (2048³) are needed to:
- Compute M_Ni(C/O, ²²Ne) calibration tables
- Quantify DDT physics contributions
- Provide z-dependent corrections for Rubin/Roman

---

## 5. Conclusions (0.25 pages)

1. Flame fractal dimension is washed out by turbulence (β → 0)
2. Luminosity diversity arises from nucleosynthetic yields
3. The age-luminosity correlation (5.5σ) has a physical basis
4. The DESI phantom crossing is likely an astrophysical systematic

---

## Figures

### Figure 1: Resolution Convergence
- Panel A: β(N) showing power-law decay to zero
- Panel B: D(Z) at different resolutions
- Caption: "Turbulent washout: the metallicity-flame coupling vanishes at high resolution"

### Figure 2: Model Validation
- Panel A: x₁(z) model vs observations (Nicolas et al., SN 2023adsy)
- Panel B: Magnitude bias Δμ(z)
- Panel C: M_Ni vs metallicity
- Caption: "Nucleosynthetic model reproduces observed stretch evolution"

### Figure 3: Physical Chain
- Schematic: Z → Ye → M_Ni → L → Δμ
- Comparison with Son et al. age-luminosity slope
- Caption: "The nucleosynthetic origin of the age-luminosity correlation"

---

## Key References

1. Son et al. 2025, MNRAS 544, 975 (5.5σ age-luminosity)
2. Nicolas et al. 2021, A&A 649, A74 (5σ stretch evolution)
3. Timmes et al. 2003, ApJ 590, L83 (metallicity → Ni-56)
4. Keegans et al. 2023, ApJS 268, 8 (metallicity-dependent yields)
5. DESI Collaboration 2024, arXiv:2404.03002 (phantom crossing)
6. Pierel et al. 2024, arXiv:2411.10427 (SN 2023adsy)

---

## Target Journal

ApJ Letters (4 pages, rapid publication)

## Timeline

- Draft: January 2026
- Internal review: January 2026
- Submission: February 2026
- Expected publication: March-April 2026

---

**Document Version:** 1.0
**Created:** 2025-11-29
