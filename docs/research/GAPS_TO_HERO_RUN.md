# From Reduced Models to Hero Runs: The Path to First Principles

**Date:** November 28, 2025
**Status:** SPECIFICATION DOCUMENT
**Purpose:** Define requirements for full *ab initio* validation of the Spandrel Framework

---

## Executive Summary

The Spandrel Framework has been tested at the scaling law level using reduced-order models. Full-star 3D radiation-hydrodynamics simulations are needed to verify results from first principles. This document specifies the four gaps between current models and production runs.

---

## Current Model Hierarchy

| Model | File | What It Captures | What It Misses |
|-------|------|------------------|----------------|
| D(Z) Parametric | `D_z_model.py` | Scaling laws | No physics derivation |
| 3D Box-in-Star | `flame_box_3d.py` | Turbulent flame statistics | Density gradient, expansion |
| 1D Hydro | `riemann_hydro.py` | Macro-scale explosion | Multi-D, nuclear network |
| Cosmological Bias | `spandrel_cosmology.py` | Observable chain | No spectral synthesis |
| Spectral Proxy | `spectral_fractal.py` | v_Si correlation | No radiative transfer |

---

## Gap 1: Box vs. Star (Boundary Conditions)

### The Problem

`flame_box_3d.py` simulates a periodic cube (L ~ 10 km) embedded in the star. It assumes:
- Constant background density ρ = 2×10⁹ g/cm³
- No density gradient (dρ/dr = 0)
- No global expansion

### The Reality

A real WD has:
- Steep density gradient: ρ(r) ∝ (1 - r²/R²)^n
- As bubble rises, it expands adiabatically
- Expansion competes with wrinkling

### The Physics Missing

**Baroclinic Torque:**
```
ω̇_baroclinic = (1/ρ²) × (∇ρ × ∇P)
```

This generates vorticity spontaneously when density and pressure gradients are misaligned. In a stratified WD:
- Creates additional turbulence not captured in periodic boxes
- May INCREASE or DECREASE D depending on alignment

### Simulation Requirements

| Parameter | Current | Required |
|-----------|---------|----------|
| Domain | Periodic box, 10 km | Full star, 2000 km radius |
| Grid | Uniform 48³-128³ | AMR, 10 cm to 100 km |
| Density | Constant | Stratified polytrope |
| Gravity | None | Self-gravity (Poisson) |
| Expansion | None | Compressible hydro |

### Minimum Resolution

To resolve Kolmogorov scale (η ~ 1 cm) in a star (R ~ 2000 km):
```
N_eff = R/η = 2×10⁸
```

With AMR (factor 100 compression): N_eff ~ 2×10⁶ → 16,384³ equivalent

---

## Gap 2: Urca Micro-Physics (The Thermostat)

### The Problem

`URCA_REYNOLDS_ATTRACTOR.md` models Urca cooling as:
```
Q_Urca = ε₀ × ρ² × T⁵
```

This is a bulk approximation.

### The Reality

The Urca process involves specific isotopes:
- ²³Na ↔ ²³Ne (E_th = 4.38 MeV)
- ²⁵Mg ↔ ²⁵Na (E_th = 3.83 MeV)

These isotopes exist in **Urca shells** — thin radial layers where the Fermi energy equals the threshold:
```
E_F(r) = E_th → defines r_Urca
```

### The Physics Missing

1. **Shell Structure:** Cooling is not uniform; it's concentrated in shells
2. **Convective Pile-up:** Does convection stall at Urca shells?
3. **Neutrino Transport:** ν escape is non-local (optically thin)
4. **Network Effects:** Urca pairs are produced by prior burning

### Simulation Requirements

| Parameter | Current | Required |
|-----------|---------|----------|
| Nuclear Network | None | 23+ isotopes (Urca pairs) |
| Neutrino Transport | None | Leakage scheme or flux-limited |
| Radial Structure | None | Resolved Urca shells |
| Electron Capture | None | Tabulated rates (Langanke) |

### The Key Test

Does the deflagration "pile up" at Urca shells, creating a preferred turbulence level?

---

## Gap 3: Spot vs. Swarm (Ignition Statistics)

### The Problem

`riemann_hydro.py` initializes ONE hot spot (Zel'dovich gradient). This ignites the flame.

### The Reality

The convective core has millions of thermal fluctuations:
- PDF of temperature: P(T) ~ Gaussian with σ_T from turbulence
- Any fluctuation > T_crit can ignite
- Multi-point ignition is possible

### The Physics Missing

1. **Stochastic Ignition:** Draw from turbulent PDF
2. **Competition:** First flame expands star, suppressing others
3. **DDT Location:** Where does detonation trigger?
4. **Asymmetry:** Off-center ignition → polarization signature

### Simulation Requirements

| Parameter | Current | Required |
|-----------|---------|----------|
| Ignition | Single point | Stochastic field |
| Number of Spots | 1 | 10-1000 sampled from PDF |
| Competition | None | Full hydro interaction |
| DDT Criterion | None | Karlovitz number tracking |

### The Key Test

Is there a "winning" ignition point that determines explosion outcome?

---

## Gap 4: Synthetic Spectrum (The Final Observable)

### The Problem

`spectral_fractal.py` predicts v_Si from scaling arguments. This is a proxy, not a simulation.

### The Reality

Astronomers measure spectra: flux vs. wavelength, with absorption lines.
- Si II 6355Å: Formed in outer layers
- Ca II H&K: Highest velocities
- Fe II/III: Inner, Ni-56 heated regions

### The Physics Missing

1. **3D Ash Distribution:** Ni-56 is not spherically symmetric
2. **Radiative Transfer:** Monte Carlo photon propagation
3. **NLTE Effects:** Non-equilibrium ionization
4. **Time Evolution:** Spectra change day-by-day

### Simulation Requirements

| Parameter | Current | Required |
|-----------|---------|----------|
| Ash Structure | None | 3D from hydro output |
| Radiative Transfer | None | Monte Carlo (ARTIS/TARDIS) |
| Ionization | None | NLTE solver |
| Time Series | None | -15 to +60 days |

### The Key Test

Does a high-D explosion reproduce the shallow Si II of SN 1991T?
Does a low-D explosion reproduce the Ti II absorption of SN 1991bg?

---

## Hero Run Specification

### Title

*Ab Initio* Derivation of the Type Ia Phillips Relation via Full-Star 3D Radiation-Hydrodynamics

### Code Options

| Code | Pros | Cons |
|------|------|------|
| **FLASH** | AMR, nuclear networks, mature | Limited GPU support |
| **CASTRO** | GPU-native, well-documented | Smaller user base |
| **Arepo** | Moving mesh, accurate advection | Complex setup |

**Recommendation:** CASTRO with NVIDIA GPU acceleration

### Grid Specification

```
Base Grid: 256³
AMR Levels: 6 (refinement factor 2)
Max Resolution: 256 × 2⁶ = 16,384 effective
Physical: 10 cm resolution at flame front
```

### Physics Modules

1. **Equation of State:** Helmholtz (degenerate + radiation + ions)
2. **Nuclear Network:**
   - 13-isotope α-chain (He → Ni-56)
   - 10-isotope Urca network (Na, Ne, Mg, Al)
3. **Gravity:** Monopole + quadrupole (self-gravity)
4. **Neutrinos:** Leakage scheme for Urca cooling
5. **MHD:** Optional (test magnetic damping of D)

### Run Matrix

| Run | Z/Z☉ | Ignition | Purpose |
|-----|------|----------|---------|
| 1 | 1.0 | Single, center | Local calibration baseline |
| 2 | 1.0 | Single, off-center | Asymmetry test |
| 3 | 0.1 | Single, center | High-z analog |
| 4 | 0.1 | Multi-point (N=100) | Stochastic ignition |
| 5 | 3.0 | Single, center | Super-solar test |

### Output Requirements

1. **Hydrodynamic:**
   - D(t) via box-counting on flame surface
   - Ni-56 yield M_Ni(t)
   - Kinetic energy E_kin(t)

2. **Spectroscopic:**
   - 3D ash composition map at t = 100 s
   - Input to ARTIS/TARDIS for spectrum synthesis

3. **Observable Chain:**
   - Light curve (via Arnett model + ARTIS)
   - v_Si at maximum (from synthetic spectrum)
   - x₁ (from light curve fit)

### Resource Estimate

| Component | Cost |
|-----------|------|
| Single 16,384³-equivalent run | 50,000 GPU-hours |
| 5-run matrix | 250,000 GPU-hours |
| Radiative transfer (ARTIS) | 50,000 GPU-hours |
| **Total** | **300,000 GPU-hours** |

**Facility:** NERSC Perlmutter or OLCF Frontier
**Allocation:** INCITE or ALCC proposal required

### Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Code Setup | 3 months | CASTRO + network configured |
| Calibration Runs | 3 months | Resolution convergence |
| Production Runs | 6 months | Full 5-run matrix |
| Analysis | 3 months | D(Z), spectra, light curves |
| **Total** | **15 months** | First-principles Phillips relation |

---

## Immediate Actions (Local)

While awaiting HPC allocation, we can:

1. **Baroclinic Extension:** Add ∇ρ × ∇P term to `flame_box_3d.py`
2. **Urca Network:** Implement simplified 4-isotope network
3. **Spectral Validation:** Compare `spectral_fractal.py` to observed SNe
4. **INCITE Proposal:** Draft for June 2025 deadline

---

## Success Criteria

The Hero Run is successful if:

1. **D(Z=1.0) ≈ 2.2-2.4** (matches local SNe x₁ distribution)
2. **D(Z=0.1) - D(Z=1.0) ≈ 0.1-0.2** (matches JWST high-z)
3. **Synthetic v_Si** matches Wang et al. 2009 correlation
4. **Synthetic light curve** reproduces Phillips relation

If all four criteria are met: **Spandrel Framework is proven from first principles.**

---

## References

1. Reinecke et al. 2002, A&A — First 3D DDT simulations
2. Seitenzahl et al. 2013, MNRAS — Ni-56 yields from DDT
3. Kasen et al. 2009, ApJ — Radiative transfer for SNe Ia
4. Almgren et al. 2010, ApJ — CASTRO code description
5. Fryxell et al. 2000, ApJS — FLASH code description

---

**Document Version:** 1.0
**Status:** READY FOR INCITE PROPOSAL
