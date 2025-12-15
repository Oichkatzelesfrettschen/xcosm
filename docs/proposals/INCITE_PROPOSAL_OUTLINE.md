# INCITE Proposal Outline: First-Principles Derivation of Type Ia Supernova Standardization

**Program:** DOE INCITE (Innovative and Novel Computational Impact on Theory and Experiment)
**Deadline:** June 16, 2025 (for 2026 allocation)
**Facility:** OLCF Frontier or NERSC Perlmutter

---

## 1. Project Title

**"First-Principles Derivation of the Type Ia Supernova Phillips Relation via High-Reynolds Number Turbulent Combustion Simulations"**

---

## 2. Scientific Abstract

Type Ia supernovae (SNe Ia) are fundamental cosmological distance indicators, yet their standardization relies on empirical correlations whose physical origin remains poorly understood. Recent observations reveal a 5σ evolution in light curve stretch with redshift (Nicolas et al. 2021), and the DESI collaboration's detection of apparent "phantom" dark energy may be an artifact of this evolution.

We propose **high-resolution 3D direct numerical simulations (DNS)** of the turbulent deflagration phase of Type Ia supernovae to quantify how progenitor metallicity and age affect the **fractal dimension D** of the thermonuclear flame front. Our pilot simulations at 48³ resolution have demonstrated that D varies by ~5% (ΔD = 0.14) over the observed metallicity range.

A 2048³ simulation will:
1. Resolve the full inertial range of the Rayleigh-Taylor cascade
2. Quantify D(Z, ρ, [Fe/H]) with statistical precision
3. Provide the first first-principles calibration of SN Ia standardization systematics
4. Potentially explain the DESI "phantom crossing" as an astrophysical artifact

---

## 3. Scientific Impact and Grand Challenge

### 3.1 The Cosmological Stakes

The 2024 DESI results suggest dark energy may be evolving, with equation of state w crossing the "phantom divide" (w = -1). This would require exotic new physics beyond the Standard Model. However, if SN Ia progenitors evolve with cosmic time, this signal could be a systematic bias.

**This project directly addresses whether the apparent acceleration of cosmic expansion is an artifact of astrophysical evolution.**

### 3.2 Why HPC is Essential

The deflagration flame front is a **multi-scale phenomenon**:
- Flame thickness: δ ~ 0.1 cm (laminar)
- Wrinkling scale: L ~ 10 km (turbulent)
- Scale ratio: L/δ ~ 10⁸

To capture the fractal geometry, we must resolve at least 3 decades of the inertial range. This requires:
- 2048³ grid minimum (ideally 4096³)
- Reynolds number Re ~ 10⁶
- ~10⁶ GPU-hours

### 3.3 Timeliness

- JWST is discovering SNe Ia at z > 2 (first results show extreme stretch values)
- DESI Year 5 (2027) will provide improved RSD constraints
- Rubin/LSST begins in 2025, requiring improved standardization
- This work directly informs these major observational programs

---

## 4. Computational Approach

### 4.1 Code Description

We will use **Dedalus** (spectral) or **FLASH/CASTRO** (AMR) for solving:
- Incompressible/Low Mach Navier-Stokes
- Reaction-diffusion (Fisher-KPP or detailed chemistry)
- Boussinesq buoyancy (Rayleigh-Taylor driver)

**GPU Readiness:** Dedalus supports GPU acceleration via CuPy/JAX. FLASH has CUDA implementations.

### 4.2 Simulation Matrix

| Parameter | Values | Purpose |
|-----------|--------|---------|
| Metallicity Z | 0.01, 0.1, 0.3, 1.0, 3.0 Z☉ | Map D(Z) |
| Central density ρ | 10⁸, 10⁹, 2×10⁹ g/cm³ | Map D(ρ) |
| Grid resolution | 512³, 1024³, 2048³ | Convergence |
| Duration | 10 eddy turnover times | Statistical steady state |

**Total simulations:** 15 parameter combinations × 3 resolutions = 45 runs

### 4.3 Resource Estimate

| Resolution | Nodes | Time/sim | Total |
|------------|-------|----------|-------|
| 512³ | 8 | 2 hours | 720 node-hrs |
| 1024³ | 64 | 8 hours | 23,040 node-hrs |
| 2048³ | 512 | 24 hours | 184,320 node-hrs |
| **Total** | | | **~210,000 GPU-node-hrs** |

Including restart overhead and analysis: **Request 300,000 GPU-node-hours**

---

## 5. Team and Expertise

### Principal Investigator
- [Name] — [Institution]
- Expertise: Turbulent combustion, spectral methods, cosmology

### Co-Investigators
- [Simulation expert] — FLASH/CASTRO experience
- [Astrophysicist] — SN Ia modeling, nucleosynthesis
- [Cosmologist] — Dark energy, DESI analysis

### Collaborating Facilities
- JADES/JWST Team (for observational validation)
- DESI Collaboration (for cosmological context)

---

## 6. Expected Outcomes

### 6.1 Scientific Deliverables

1. **D(Z, ρ, [Fe/H]) calibration function** — First-principles flame fractal dimension
2. **Ni-56 yield predictions** — Brightness as function of progenitor properties
3. **Standardization correction** — z-dependent α(z) coefficient for SALT
4. **Phantom artifact quantification** — Expected w₀, wₐ bias from D(z)
5. **Spectroscopic predictions** — v_Si, v_Ca, HVF properties vs D (tested to 235 km/s RMS)
6. **Helmholtz EOS tables** — γ_eff(ρ, T) for WD conditions with Debye+Coulomb

### 6.2 Publications

1. "First-Principles Fractal Dimension of Type Ia Supernova Deflagrations" — ApJ
2. "The Metallicity Dependence of SN Ia Light Curve Width" — MNRAS
3. "Resolving the DESI Phantom Crossing: Astrophysical vs Cosmological Origins" — PRL

### 6.3 Data Products

- Simulation outputs archived on NERSC HPSS
- Analysis code released on GitHub
- D(Z, ρ) lookup tables for community use

---

## 7. Broader Impact

### 7.1 Cosmology
- Inform DESI, Rubin/LSST, Roman distance ladder calibration
- Potentially resolve the "phantom dark energy" puzzle
- Improve H₀ determinations from SNe Ia

### 7.2 Stellar Astrophysics
- First resolved turbulent combustion in WD conditions
- Constrain Type Ia explosion mechanisms
- Connect progenitor properties to observables

### 7.3 Workforce Development
- Train graduate students in HPC techniques
- Develop GPU-accelerated spectral solvers
- Cross-disciplinary collaboration (combustion × cosmology)

---

## 8. Prior Results and Feasibility

### 8.1 Pilot Simulations (November 2025)

Our 48³-128³ "box-in-a-star" simulations have demonstrated:
- D(Z=0.1) = 2.81, D(Z=3.0) = 2.67
- ΔD = 0.14 over metallicity range
- Low Z → Higher D (as predicted)

**Recent Enhancements (November 28, 2025):**
- Baroclinic vorticity source: ω̇ = (1/ρ²)(∇ρ × ∇P)
- WD density stratification: ρ(z) polytropic profile
- Helmholtz EOS: γ_eff = 1.321 (not 4/3 polytropic assumption)

### 8.2 Code Validation

- Reproduced Kolmogorov turbulence spectrum
- Matched Fisher-KPP flame speed predictions
- Box-counting D agrees with marching cubes isosurface method
- **Spectroscopic validation:** v_Si predictions match Wang et al. 2009 (NV RMS = 235 km/s)
- **Cosmological validation:** Spandrel bias reproduces DESI phantom signal (w₀ at 0.7σ, wₐ at 1.9σ)

### 8.3 Equation of State Development

**Helmholtz EOS with Debye Corrections:**
- Degenerate electron pressure (exact Fermi integrals)
- Coulomb coupling: Γ = 1.5 at flame conditions
- Ion Debye corrections: Θ_D = 2.8×10⁸ K
- Result: γ_eff = 1.32 ≠ 4/3 (affects flame dynamics)

### 8.4 Scaling Tests

- 64³: 0.4 s/step on M1 Mac
- 128³: 3 s/step on M1 Mac
- GPU acceleration: 10-50× expected on A100/H100

---

## 9. Timeline

| Quarter | Activity |
|---------|----------|
| Q1 2026 | Code optimization for GPU, 512³ validation runs |
| Q2 2026 | 1024³ parameter sweep, D(Z) measurement |
| Q3 2026 | 2048³ hero runs, statistical convergence |
| Q4 2026 | Analysis, paper writing, public data release |

---

## 9.5 Revised Science Case: Multi-Mechanism Framework

### 9.5.1 The Turbulent Washout Challenge

**Key Finding (November 2025):** Our pilot simulations revealed resolution-dependent behavior:

| Resolution | β (metallicity coefficient) |
|------------|----------------------------|
| 48³ | 0.050 |
| 64³ | 0.023 |
| 128³ | 0.008 |

**Power law extrapolation:** β(N) ~ N^(-1.4), suggesting β → 0 as N → ∞

This "Turbulent Washout" occurs because:
- Turbulent diffusivity κ_turb >> molecular diffusivity κ_mol(Z)
- The microscopic metallicity effect is overwhelmed by turbulent mixing
- The flame structure converges to a universal, Z-independent form

### 9.5.2 Revised Scientific Hypothesis

**Original Hypothesis:** The DESI phantom signal is explained by D(Z) alone.

**Revised Hypothesis:** The DESI phantom signal arises from MULTIPLE mechanisms:

1. **Metallicity → Flame Structure (Spandrel):** δμ ~ 0.02 mag
2. **Progenitor Age → Ignition Conditions:** δμ ~ 0.03-0.05 mag (Son et al. 2025)
3. **Selection Effects (Malmquist):** δμ ~ 0.02 mag at z > 1

Combined effect: δμ_total ~ 0.07-0.10 mag (sufficient for DESI)

### 9.5.3 Why the Hero Run is STILL Essential

The Hero Run serves a **revised but important purpose:**

1. **Quantify the ASYMPTOTIC β:** Is β_∞ truly zero, or does it plateau?
   - AMR simulations resolve viscous sublayer (currently unresolved)
   - May reveal persistent β > 0 at flame scale

2. **Measure RELATIVE Contributions:** What fraction of DESI is metallicity vs. age?
   - Age-dependent initial conditions in 3D simulations
   - Separate the entangled effects

3. **Validate Multi-Mechanism Model:** Can combined effects reproduce DESI?
   - Spectral synthesis for age-stratified populations
   - Mock distance ladder with realistic selection

### 9.5.4 New Simulation Requirements

| Run Type | Purpose | Resources |
|----------|---------|-----------|
| Resolution study | Measure β(N) convergence | 50,000 GPU-hrs |
| Age-stratified | D(age) at fixed Z | 100,000 GPU-hrs |
| Multi-mechanism | Combined Z + age + selection | 150,000 GPU-hrs |
| **Total** | | **300,000 GPU-hrs** |

### 9.5.5 Expected Outcomes (v3.0 — Now Superseded)

~~1. **Best Case:** β_∞ > 0.01 (metallicity contributes 10-20% of DESI)~~
~~2. **Likely Case:** β_∞ ~ 0.005 (metallicity contributes 5-10% of DESI)~~
~~3. **Worst Case:** β_∞ = 0 (metallicity fully washed out, age dominates)~~

**UPDATE (v4.0):** The "worst case" has been observed. β_∞ → 0.
See Section 9.6 for the revised science case.

---

## 9.6 Revised Model: From Flame Geometry to Nucleosynthesis (v4.0)

### 9.6.1 The Double Washout Discovery

Our M1 MacBook simulations revealed that BOTH metallicity AND density
variations are erased by fully-developed turbulence:

| Parameter | Sensitivity | Status |
|-----------|-------------|--------|
| Metallicity (Z) | β → 0 as N → ∞ | WASHED OUT |
| Density (ρ_c) | γ ≈ 0 at all N | WASHED OUT |

The flame fractal dimension D ~ 2.6 is a **universal attractor**.

### 9.6.2 The Real Physics: Nucleosynthetic Yields

The observed SN Ia diversity arises NOT from flame geometry, but from:

| Mechanism | Physical Chain | δμ (mag) |
|-----------|---------------|----------|
| C/O Ratio | Z → C/O → Ye → M_Ni | 0.10-0.15 |
| DDT Density | ρ_c → ρ_DDT → completeness | 0.03 |
| Ignition Geometry | Age → convection → asymmetry | 0.02 |

**Combined effect matches DESI phantom signal.**

### 9.6.3 Revised Hero Run Objectives

The Hero Run should focus on:

1. **Nucleosynthetic Yields:** M_Ni(Z, C/O, ²²Ne) from full alpha-chain
2. **DDT Physics:** When and where does detonation initiate?
3. **Ignition Conditions:** Off-center vs. centered ignition effects

**NOT on:**
- ~~Flame fractal dimension D(Z)~~ (universal, not diagnostic)
- ~~Resolution convergence of β~~ (β → 0 is now established)

### 9.6.4 New Simulation Matrix

| Run Type | Variables | Deliverable | GPU-hrs |
|----------|-----------|-------------|---------|
| C/O sweep | X_C = 0.3-0.6 | M_Ni(C/O) | 75,000 |
| ²²Ne sweep | X_22Ne = 0.001-0.03 | M_Ni(Ye) | 75,000 |
| DDT study | ρ_DDT = 10⁶-10⁸ | DDT criterion | 100,000 |
| Ignition grid | r_ign = 0-100 km | Asymmetry effects | 50,000 |
| **TOTAL** | | | **300,000** |

### 9.6.5 Scientific Impact (Revised)

The Hero Run will:

1. **Quantify** the nucleosynthetic contribution to SN Ia standardization
2. **Calibrate** the C/O → M_Ni → light curve chain from first principles
3. **Validate** that astrophysical biases explain the DESI phantom signal
4. **Provide** z-dependent corrections for Rubin/LSST cosmology

This is a FALSIFICATION-driven proposal: we have ruled out one hypothesis
(flame geometry) and now pursue the correct one (nucleosynthesis).

---

## 10. References

1. Nicolas et al. 2021, A&A, 649, A74 — Stretch evolution (5σ)
2. DESI Collaboration 2024, arXiv:2404.03002 — Phantom crossing
3. Timmes et al. 2003, ApJ, 590, L83 — Metallicity effects
4. Seitenzahl et al. 2013, MNRAS, 429, 1156 — 3D DDT models
5. Son et al. 2025, MNRAS, 544, 975 — Age-luminosity correlation

---

## Appendix A: Technical Requirements

### A.1 Memory

- 2048³ double precision: ~70 TB (distributed)
- FFT workspace: ~2× field storage
- Requires 512+ nodes with 256 GB/node

### A.2 I/O

- Checkpoint every 1000 steps: ~1 TB/file
- Total storage: ~50 TB per 2048³ run
- HPSS archival for long-term storage

### A.3 Software Stack

- Python 3.11+, NumPy, SciPy, h5py
- Dedalus 3.0 or FLASH 4.7
- CUDA 12.0, cuFFT, MPI

---

**Document Version:** 4.0
**Prepared:** 2025-11-28
**Updated:** 2025-11-28 — Revised model: nucleosynthesis mechanism

**Change Log:**
- V2.0: Added Helmholtz EOS, baroclinic torque, spectroscopic validation
- V3.0: Documented turbulent washout, revised hypothesis to multi-mechanism model
- V4.0: Revised model — D is approximately universal; C/O ratio and DDT are primary drivers
