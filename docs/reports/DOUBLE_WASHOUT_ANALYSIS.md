# The Double Washout: Why Turbulence Erases Everything

**Date:** November 28, 2025
**Status:** Complete
**Implication:** The flame box paradigm has fundamental limits

---

## Executive Summary

Two independent sweeps reveal the same phenomenon:

| Mechanism | Variable | γ = dD/d(ln X) | Status |
|-----------|----------|----------------|--------|
| Metallicity | Z | 0.008 → 0 | WASHED OUT |
| Density/Age | ρ_c | -0.0005 ≈ 0 | WASHED OUT |

**Conclusion:** In fully-developed turbulence, the flame structure converges to a
**universal attractor** that is independent of both metallicity AND gravity.

---

## 1. The Physics of the Double Washout

### 1.1 Why Metallicity Washes Out

```
κ_turb = u' × ℓ ~ 10⁻² (normalized units)
κ_mol(Z) ~ 0.03-0.05 (normalized units)

Turbulent Lewis number: Le_turb = κ_turb / κ_mol ~ 0.2-0.5
```

As resolution increases, κ_turb grows (more scales resolved).
Eventually: κ_turb >> κ_mol(Z) for all Z.
The flame "forgets" its molecular thickness.

### 1.2 Why Density/Gravity Washes Out

The Rayleigh-Taylor growth rate scales as:

```
σ_RT ~ √(g × k × At)
```

But in the nonlinear regime:

```
u_turb ~ (g × L × At)^(1/3)  (dimensional analysis)
```

Once turbulence is fully developed:
- The turbulent velocity becomes self-sustaining
- The cascade rate ε ~ u³/L becomes independent of the forcing
- D converges to the Kolmogorov-limited universal value

**Key insight:** Gravity sets the RATE of RT instability growth,
but not the FINAL turbulent state. After saturation, all gravity
values lead to the same statistical equilibrium.

---

## 2. What This Means for Spandrel

### 2.1 The Flame Box Cannot Distinguish Progenitors

Our 3D spectral solver with:
- Incompressible Navier-Stokes
- Boussinesq buoyancy
- Fisher-KPP reaction-diffusion

Cannot capture the mechanisms that differentiate SN Ia progenitors.

### 2.2 The Missing Physics

The **observed** Age-Luminosity correlation (Son et al. 2025, 5σ) must arise from:

1. **Ignition Geometry** (not in our model)
   - Off-center ignition in young progenitors
   - Floating bubble vs. centered ignition
   - Sets DDT location, not flame D

2. **Nucleosynthesis Chain** (not in our model)
   - C/O ratio varies with progenitor mass
   - Different energy release, different expansion
   - Not captured by single-step Fisher-KPP

3. **DDT Transition Physics** (not in our model)
   - Karlovitz number Ka < 1 criterion
   - Depends on turbulence intensity at transition
   - Our box doesn't capture this moment

4. **Simmering Phase** (not in our model)
   - Urca cooling over 10⁵ years
   - Sets core composition and ignition location
   - Not a deflagration property

---

## 3. The Paradigm Limitation

### 3.1 What the Flame Box CAN Do

Validate Kolmogorov turbulence (E(k) ~ k^(-5/3))
Measure box-counting fractal dimension D ~ 2.6
Confirm Rayleigh-Taylor instability
Test numerical convergence

### 3.2 What the Flame Box CANNOT Do

Distinguish Z = 0.1 from Z = 3.0 (washed out)
Distinguish ρ_c = 10⁹ from 10¹⁰ (washed out)
Predict M_Ni (no nucleosynthesis)
Predict DDT timing (no compressibility)
Explain the observed Age-Luminosity relation

---

## 4. The Path Forward

### 4.1 Option A: Abandon Flame Box Paradigm

The observed SN Ia correlations with metallicity and age arise from:
- Nucleosynthesis (C/O ratio → energy release)
- Ignition physics (simmering → hot spot location)
- Global expansion (compressible hydro, not Boussinesq)

These require **full-star 3D simulations** (FLASH/CASTRO), not flame boxes.

### 4.2 Option B: Extract What We Can

The flame box demonstrates:
- Turbulent flame structure is UNIVERSAL at high Re
- D ~ 2.6 is the attractor value
- Microscopic progenitor variations are ERASED

This is itself a scientific result: the Phillips relation does NOT
arise from deflagration geometry, but from something else (DDT, Ni-mass).

### 4.3 Option C: Focus on Non-D Mechanisms

The DESI phantom signal may arise from:
- Direct nucleosynthetic yields (not D)
- Selection effects (Malmquist bias)
- Host galaxy mass-metallicity bias

The Spandrel (D → luminosity) is not the primary driver.

---

## 5. Revised Scientific Conclusion

```
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│   THE DOUBLE WASHOUT                                             │
│                                                                  │
│   Both metallicity and density variations are erased by          │
│   fully-developed turbulence in the deflagration phase.          │
│                                                                  │
│   The flame structure D ~ 2.6 is a UNIVERSAL ATTRACTOR.          │
│                                                                  │
│   The observed Age-Luminosity and Z-Luminosity correlations      │
│   must arise from physics OUTSIDE the deflagration:              │
│     - Ignition geometry                                          │
│     - Nucleosynthetic yields                                     │
│     - DDT transition timing                                      │
│     - Selection effects                                          │
│                                                                  │
│   The "Spandrel" hypothesis (D drives M_Ni) is FALSIFIED         │
│   for turbulent deflagrations.                                   │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 6. What This Means for INCITE

### 6.1 The Original Proposal Premise (Now Invalid)

> "D(Z) varies with metallicity, driving the DESI phantom signal."

This is falsified by the Double Washout.

### 6.2 Revised Proposal Premise (Still Valid)

> "Full-star 3D simulations are needed to capture the DDT transition
> and nucleosynthetic yields that ACTUALLY drive SN Ia diversity."

The Hero Run is still essential, but for different reasons:
- Not to measure D(Z)
- But to capture DDT and M_Ni(Z, age) directly

---

## 7. Honest Assessment

We have discovered that the **Spandrel mechanism** (flame geometry →
luminosity) is a dead end for the deflagration phase. The flame
structure converges to a universal state.

However, the **Age-Luminosity observation** (Son et al. 2025) is real.
It must arise from:
- Ignition conditions (not flame geometry)
- Or DDT transition (not deflagration)
- Or direct nucleosynthesis (C/O ratio effects)

The Hero Run should focus on **DDT physics** and **full-star yields**,
not flame box D(Z) measurements.

---

**Document Version:** 1.0
**Status:** Complete
