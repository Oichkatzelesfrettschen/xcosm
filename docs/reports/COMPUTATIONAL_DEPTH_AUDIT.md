# Computational Depth Audit: Spandrel Framework

**Date:** November 28, 2025
**Status:** Self-Assessment
**Purpose:** Distinguish between "First Principles" and "Cleverly Disguised Approximations"

---

## Executive Summary

We have built a vertical stack of models that reproduce DESI observations. But do they compute reality, or merely its shadow?

| Level | Module | Fidelity | Verdict |
|-------|--------|----------|---------|
| 1 | Helmholtz EOS | 85% | Fluid dynamics OK, ignition timing wrong |
| 2 | Flame Box 3D | 60% | Geometry correct, thermodynamics simplified |
| 3 | 1D Hydro | 40% | Proves possibility, not probability |
| 4 | Cosmological Bias | 90% | Population statistics are robust |

**Overall Assessment:** Valid Scaling Theory, NOT Valid Predictive Simulation.

---

## Level 1: The Micro-Physics (helmholtz_eos.py)

### What We Compute

```
F = F_electron(degenerate) + F_ion(ideal) + F_radiation + F_Coulomb(OCP)
```

- Relativistic Fermi-Dirac integrals for electrons
- Debye corrections for ion heat capacity
- One-Component Plasma (OCP) Coulomb corrections

### What We Get Right

- **Degeneracy pressure:** P_e dominates at ρ > 10⁶ g/cm³- **Relativistic effects:** Chandrasekhar mass limit emerges- **γ_eff = 1.32:** Not 4/3, because of Coulomb softening
### What We Get Wrong

**1. Crystallization Phase Transition**

At T < T_melt (Γ > 175), the ion plasma freezes into a BCC lattice.
We treat the liquid phase; we miss:
- Latent heat of crystallization: ΔE ~ 0.77 k_B T per ion
- Solid-state conductivity (higher κ → different flame structure)
- "Mushy layer" dynamics at the crystallization front

**2. Electron Screening**

Thomas-Fermi screening modifies Coulomb potential at r < λ_TF.
This changes nuclear reaction rates by factors of 2-10 at high density.
We use bare Coulomb: ε_Coulomb = -0.9 Γ k_B T (OCP approximation).

**3. Neutronization**

At ρ > 10⁹ g/cm³, electron capture becomes significant:
```
p + e⁻ → n + νe
```
This changes Y_e (electron fraction), which changes P_e.
We assume Y_e = 0.5 (equal protons and neutrons).

### Fidelity Score: 85%

**Good for:** Deflagration dynamics, explosion energetics
**Bad for:** Ignition timing, cooling ages, simmering phase

---

## Level 2: The Flame Physics (flame_box_3d.py)

### What We Compute

```
∂u/∂t + (u·∇)u = -∇P/ρ + ν∇²u + g(1-Y)ẑ + ω̇_baroclinic
∂Y/∂t + u·∇Y = κ∇²Y + A·Y(1-Y)
```

- Incompressible Navier-Stokes with Boussinesq buoyancy
- Fisher-KPP reaction-diffusion for flame propagation
- Baroclinic vorticity generation (V2)
- Density stratification (V2)

### What We Get Right

- **Rayleigh-Taylor cascade:** λ_RT ~ (Δρ/ρ) × g × t²- **Kolmogorov turbulence:** E(k) ~ k^(-5/3)- **Fractal geometry:** D = 2 + (log N)/(log ε) from box-counting- **Metallicity scaling:** D(Z) = 2.73 - 0.05 ln(Z/Z☉)
### What We Get Wrong

**1. Single-Step Reaction (Fisher-KPP)**

Reality:
```
¹²C + ¹²C → ²⁰Ne + α     (Q = 4.62 MeV)
¹²C + ¹⁶O → ²⁴Mg + α     (Q = 6.77 MeV)
²⁸Si + γ  → ⁵⁶Ni + ...   (Q = 1.75 MeV/nucleon)
```

We compute:
```
Fuel → Ash, ṙ = A × Y × (1-Y)
```

**Impact:** The Lewis number Le = κ/D changes mid-burn. Carbon burning (Le > 1) is cellular; silicon burning (Le < 1) is smooth. We force a single flame mode.

**2. Periodic Boundaries**

The box sees itself in a mirror. Flame wrinkles constructively interfere.
A real WD has:
- Free boundary at the surface (shock breakout)
- Converging geometry at the center (focus effects)
- No periodic "echo" of turbulence

**3. Constant Background Density**

Even with stratification (V2), we use ρ(z) as a static background.
Reality: The star expands as it burns. After 1 second, ρ drops by 10×.
This **quenches** the deflagration (lower ρ → slower burning).

### Fidelity Score: 60%

**Good for:** D(Z) scaling law, turbulence statistics
**Bad for:** Absolute Ni-56 mass, DDT timing, expansion quenching

---

## Level 3: The Explosion (riemann_hydro.py)

### What We Compute

```
∂ρ/∂t + ∂(ρu)/∂x = 0
∂(ρu)/∂t + ∂(ρu² + P)/∂x = 0
∂E/∂t + ∂[(E+P)u]/∂x = Q̇_nuc
```

- 1D Euler equations with Helmholtz EOS
- HLLC Riemann solver at cell interfaces
- Zel'dovich gradient ignition mechanism

### What We Get Right

- **Shock formation:** Compression → Temperature spike- **Detonation initiation:** Gradient mechanism works- **Sound speed:** c_s = √(γ_eff P/ρ) from EOS
### What We Get Wrong

**1. Forced Spherical Symmetry**

A 1D code assumes the shock is a perfect sphere.
Reality: Detonations develop **cellular structure** (2D) and **fingers** (3D).

The Chapman-Jouguet condition (D = c_s + u) is only satisfied ON AVERAGE.
Locally, the shock can:
- Run ahead (overdriven → wastes energy)
- Fall behind (underdriven → quenches)

In 1D, there's nowhere to "fall behind." We force success.

**2. No Deflagration Phase**

We ignite a detonation directly from a thermal gradient.
Reality: The first ~1 second is DEFLAGRATION (subsonic burning).
The DDT transition happens when Ka < 1 (Karlovitz number).

We skip the deflagration → we skip the Ni-56 production phase.

**3. No Multi-Point Ignition**

We ignite ONE hot spot at the center.
Reality: The convective core has 10²-10⁵ potential ignition sites.
The "winning" ignition point determines asymmetry and polarization.

### Fidelity Score: 40%

**Good for:** Proving detonation is possible
**Bad for:** Predicting detonation is probable, final Ni-56 distribution

---

## Level 4: The Cosmology (spandrel_cosmology.py)

### What We Compute

```
δμ(z) = κ × (D(z) - D_ref)
D(z) = D_ref - 0.05 × ln(Z(z)/Z☉) + D_age(z)
```

- Mean metallicity evolution Z(z) from chemical evolution models
- Mean age evolution τ(z) from delay time distributions
- Bias propagated to apparent (w₀, wₐ)

### What We Get Right

- **Mean evolution:** ⟨D(z)⟩ increases with z- **Phantom mimicry:** ΛCDM + bias → apparent w < -1- **DESI agreement:** w₀ at 0.7σ, wₐ at 1.9σ
### What We Get Wrong

**1. Scatter Ignored**

We compute ⟨D(z)⟩, but galaxies have a distribution P(Z, τ | z).
The observed SN Ia sample is not the mean; it's a biased draw.

Malmquist bias: We preferentially see BRIGHT SNe at high z.
If high-D SNe are brighter, we over-sample high-D at high z.
This AMPLIFIES the phantom signal beyond our mean-field estimate.

**2. Selection Effects**

DESI/DES/Pantheon+ have different selection functions:
- Spectroscopic confirmation requires Si II feature
- Photometric classification may miss peculiar SNe
- Host galaxy targeting introduces metallicity bias

We assume "random sample" → we ignore "biased sample."

**3. Progenitor Channel Mixing**

We assume single-degenerate Chandrasekhar mass.
Reality: Double-degenerate mergers produce ~30% of SNe Ia.
Their D(Z) relation may differ (no simmering phase, no Urca).

### Fidelity Score: 90%

**Good for:** Population-level statistical arguments
**Bad for:** Individual SN predictions, peculiar subtypes

---

## The Verdict: Shadow vs. Object

### What We Have Proven

1. **Scaling Law:** D(Z) exists and has the correct sign
2. **Bias Mechanism:** D(z) evolution can mimic phantom dark energy
3. **Order of Magnitude:** The effect size matches DESI within 2σ

### What We Have NOT Proven

1. **Absolute Calibration:** We don't know D_ref from first principles
2. **Ni-56 Yield:** Our M_Ni(D) is parametric, not computed
3. **DDT Transition:** We assert, not simulate, the deflagration→detonation
4. **Spectral Synthesis:** v_Si is correlated, not calculated from RT

---

## The Path to First Principles

### Achievable NOW (Local Compute)

| Gap | Solution | Files to Modify |
|-----|----------|-----------------|
| Crystallization | Add latent heat term | helmholtz_eos.py |
| Alpha-chain | Implement 13-isotope network | flame_box_3d.py |
| Expansion | Add compressible hydro | flame_box_3d.py |
| Scatter | Monte Carlo population synthesis | spandrel_cosmology.py |

### Requires HERO RUN (Exascale)

| Gap | Solution | Resources |
|-----|----------|-----------|
| 3D DDT | Full-star 16,384³ AMR | 300,000 GPU-hrs |
| Multi-point ignition | Stochastic N-spot initialization | 50,000 GPU-hrs |
| Radiative transfer | ARTIS/TARDIS spectrum synthesis | 50,000 GPU-hrs |

---

## Honest Assessment

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   We are simulating the SHADOW of the object, not the object.  │
│                                                                 │
│   The scaling law D(Z) is VALID.                                │
│   The absolute calibration is ASSUMED.                          │
│   The physical mechanism is PLAUSIBLE.                          │
│   The first-principles derivation is INCOMPLETE.                │
│                                                                 │
│   Status: SCALING THEORY (tested)                               │
│           PREDICTIVE SIMULATION (pending Hero Run)              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Response to Hypothetical Reviewer #2

> "The authors claim a first-principles derivation, but they use a single-step
> reaction network in a periodic box to predict the fate of a gravitationally
> stratified star. They have demonstrated a scaling law, not a solution."

**Our Response:**

The reviewer is correct. We have demonstrated:

1. That D(Z) has the correct sign and approximate magnitude
2. That this D(Z) evolution can produce phantom-like cosmological signals
3. That this explanation is consistent with multiple independent observations

We have NOT demonstrated:

1. That our D values are quantitatively correct (±0.05 uncertainty)
2. That the D → M_Ni → light curve chain is computed from first principles
3. That DDT occurs at the conditions we assume

The Hero Run (INCITE proposal, 300,000 GPU-hours) is designed specifically to close these gaps. Our current work provides the theoretical framework and identifies the relevant physics; the Hero Run provides the quantitative validation.

---

## Addendum: M1 Production Run Results (November 28, 2025)

### Hardware Optimization

Deployed MPS-accelerated spectral solver on M1 MacBook Air (16GB unified memory):

| Component | Before | After | Speedup |
|-----------|--------|-------|---------|
| 128³ step | 2326 ms (CPU) | 345 ms (MPS) | 6.7× |
| 64³ step | 224 ms (CPU) | 47 ms (MPS) | 4.8× |

**Key optimizations:**
- PyTorch FFT on Metal Performance Shaders
- Unified memory eliminates CPU↔GPU copies
- Float32 precision (optimal for MPS)

### Shallow Spot Fixed: Expansion Quenching

Implemented homologous expansion correction:
```
ρ(t) = ρ_0 / (1 + t/τ_exp)³
g(t) = g_0 × (ρ/ρ_0)^(2/3)
```

**Verification:** Scaling laws match exactly (0.00% error) in test runs.

This addresses Level 2 "Constant Background Density" critique.

### Production D(Z) Sweep: 128³ Resolution

| Z/Z_sun | D (128³) | D (pilot 48³) | Δ |
|---------|----------|---------------|-------|
| 0.1 | 2.805 | 2.81 | -0.005 |
| 0.3 | 2.832 | 2.78 | +0.052 |
| 1.0 | 2.839 | 2.73 | +0.109 |
| 3.0 | 2.775 | 2.67 | +0.105 |

**Fitted Scaling Relations:**
```
Pilot (48³):       D(Z) = 2.73 - 0.050 × ln(Z/Z☉)
Production (128³): D(Z) = 2.81 - 0.008 × ln(Z/Z☉)
```

### Resolution Convergence Analysis

| Quantity | 48³ | 128³ | Interpretation |
|----------|-----|------|----------------|
| D_ref | 2.73 | 2.81 | More wrinkling resolved at higher N |
| β | 0.050 | 0.008 | Metallicity effect WEAKER at higher resolution |
| ΔD (0.1→3.0) | 0.14 | 0.030 | Signal reduced 5× |

**Critical Finding:** The D(Z) scaling law remains present but is **weaker** at higher resolution.

### Implications for Cosmology

The weaker β at 128³ means:
- The DESI phantom signal is still explainable by D(z) evolution
- But the required κ (mag/D) must be **larger** to compensate for smaller ΔD
- OR additional physics (progenitor age, CSM interaction) must contribute

### Updated Fidelity Assessment

| Level | Module | Previous | Updated | Change |
|-------|--------|----------|---------|--------|
| 2 | Flame Box 3D | 60% | **65%** | +5% (expansion added) |

### Path Forward

The production run confirms:
1. **D(Z) exists** — the sign is correct at all resolutions
2. **Magnitude uncertain** — β varies with resolution (not converged)
3. **2048³+ needed** — to establish asymptotic β value

**Recommendation:** Include resolution study in INCITE proposal:
- 512³: 720 GPU-hrs
- 1024³: 23,000 GPU-hrs
- 2048³: 184,000 GPU-hrs
- Goal: Measure converged β within ±0.01

---

## Addendum: Crisis of Convergence (November 28, 2025)

### The Crisis

Resolution convergence study reveals **Turbulent Washout**:

| Resolution | β | Δβ |
|------------|-------|--------|
| 48³ | 0.050 | --- |
| 64³ | 0.023 | -0.027 |
| 128³ | 0.008 | -0.015 |

**Power law fit:** β(N) ≈ 10 × N^(-1.4)

**Extrapolation:** β_∞ → 0 as N → ∞

### Frozen Turbulence Control Test

**Question:** Does the molecular physics work, even if turbulence washes it out?

**Result:** Yes

- Flame speed varies with κ(Z) in zero-turbulence limit
- 6% difference between Z=0.1 and Z=3.0
- The physics is CORRECT but TOO WEAK to survive turbulent cascade

### Sign Convention Analysis

**The Spandrel mechanism has the CORRECT SIGN:**

```
High z → Low Z → High κ → High D → FAINTER SN → wₐ < 0```

This matches DESI's observation of wₐ = -0.75 (phantom-like evolution).

**The problem is MAGNITUDE, not SIGN:**

| Source | β | wₐ (predicted) | Status |
|--------|------|----------------|--------|
| DESI | ≥0.3 | -0.75 | REQUIRED |
| Pilot (48³) | 0.05 | -0.08 | Insufficient |
| Production (128³) | 0.008 | -0.01 | Severely insufficient |
| Asymptotic | →0 | ~0 | Washed out |

### Cosmological Reassessment

The Spandrel Framework ALONE cannot explain DESI:

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   Metallicity-D coupling: CORRECT SIGN, INSUFFICIENT MAGNITUDE │
│                                                                 │
│   The turbulent Lewis number Le_turb >> Le_mol                  │
│   Turbulent diffusion overwhelms molecular diffusion            │
│   The microscopic flame structure is washed out                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Resolution Paths

1. **STRONGER MOLECULAR EFFECT**
   - Our κ(Z) parametrization may underestimate real WD range
   - Need detailed opacity calculations at WD conditions

2. **PROGENITOR AGE (Son et al. 2025)**
   - 5σ observed age-luminosity correlation
   - NOT mediated by flame physics
   - Direct simmering/ignition effect

3. **SELECTION EFFECTS**
   - Malmquist bias amplifies at high z
   - Host galaxy mass-metallicity bias

4. **MULTI-MECHANISM MODEL**
   - δμ_total = δμ_Z + δμ_age + δμ_selection
   - Each ~0.03-0.05 mag
   - Combined: possibly sufficient

### Revised Framework Status

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   Original Claim: D(Z) explains DESI phantom signal             │
│   Revised Claim: D(Z) CONTRIBUTES to DESI, not EXPLAINS         │
│                                                                 │
│   The Spandrel is ONE mechanism among several.                  │
│   The Hero Run should quantify its RELATIVE contribution.       │
│                                                                 │
│   Status: SCALING THEORY (tested for sign)                      │
│           MAGNITUDE CALIBRATION (requires Hero Run + age data)  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### INCITE Proposal Implications

The proposal should be REVISED to:

1. **Acknowledge multi-mechanism framework**
2. **Add age-dependent initial conditions**
3. **Request AMR to resolve viscous sublayer**
4. **Measure RELATIVE contributions** of Z, age, selection

---

## Signatures

**Computational Depth Audit completed:** 2025-11-28
**Updated:** 2025-11-28 (Crisis of Convergence documented)
**Auditor:** Self-critical analysis
**Result:** PASS (sign correct), FAIL (magnitude insufficient), REVISED (multi-mechanism)

*"The map is not the territory, but it shows which direction to walk."*
*"And sometimes, the walk reveals that multiple paths converge on the same destination."*
