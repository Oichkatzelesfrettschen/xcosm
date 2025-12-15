# Gravitational Wave Predictions: Type Ia Supernovae and DECIGO

**Date:** 2025-11-28
**Status:** THEORETICAL FRAMEWORK (requires simulation validation)

---

## Executive Summary

Type Ia supernovae generate gravitational waves through two mechanisms:
1. **Pre-explosion merger** (WD-WD binary inspiral)
2. **Explosion asymmetry** (deflagration/detonation turbulence)

DECIGO (0.1-10 Hz) can detect both signals, providing multi-messenger constraints on progenitor properties and potentially testing the Spandrel D(z) hypothesis.

---

## 1. DECIGO Sensitivity

### 1.1 Frequency Band

| Detector | Frequency Range | Target Sources |
|----------|-----------------|----------------|
| LIGO/Virgo | 10-1000 Hz | Compact mergers, core-collapse |
| **DECIGO** | **0.1-10 Hz** | **WD mergers, SN Ia explosions** |
| LISA | 0.1-100 mHz | Massive BH mergers, EMRIs |

### 1.2 Strain Sensitivity

- DECIGO goal: h ~ 10⁻²⁴ Hz⁻¹/² at 1 Hz
- B-DECIGO (precursor): h ~ 10⁻²² Hz⁻¹/² at 1 Hz
- Launch: B-DECIGO 2030s, DECIGO 2040s+

---

## 2. GW from Pre-Explosion Mergers (WD-WD)

### 2.1 Detection Predictions

From Maselli et al. 2019 (arXiv:1910.01063):

| Quantity | Value |
|----------|-------|
| Expected mergers (z < 0.08) | ~6600/year |
| Host galaxy ID (z < 0.065) | Possible |
| Progenitor masses | 1.0 M☉ + 0.8 M☉ (typical) |
| Merger frequency | 0.01-0.1 Hz |

### 2.2 Connection to Spandrel

If WD mass distribution evolves with z:
- High-z: More sub-Chandrasekhar mergers
- Low-z: More Chandrasekhar-mass singles

This could be detected via GW merger mass statistics vs. optical SN Ia rates.

---

## 3. GW from Explosion Asymmetry

### 3.1 Physical Mechanism

Deflagration/detonation creates asymmetric mass motion:
```
Asymmetric burning → Quadrupole moment Q̈ → GW emission

h ∝ (G/c⁴) × (Q̈/r) ∝ (G/c⁴) × (M × v²/r)
```

### 3.2 Strain Estimates

From Falta et al. 2011 (PRL 106, 201103):

| Distance | Peak Strain h | Frequency |
|----------|---------------|-----------|
| 10 kpc | ~10⁻²² | 1-3 Hz |
| 1 Mpc | ~10⁻²⁴ | 1-3 Hz |
| 10 Mpc | ~10⁻²⁵ | 1-3 Hz |

### 3.3 Spandrel D Prediction

Higher fractal dimension D → More asymmetric explosion → Larger GW amplitude

```
h(D) ∝ (D - 2)^α  where α ~ 1-2

D = 2.0 (smooth):  h = h_min
D = 2.5 (fractal): h ~ 3 × h_min
D = 2.8 (extreme): h ~ 10 × h_min
```

**Caveat:** This scaling is theoretical. Our 3D simulations give D ~ 2.7-2.8, but the GW amplitude depends on whether the asymmetry is global (detectable) or local (cancels out).

---

## 4. D(z) Effect on GW Background

### 4.1 Stochastic Background

The superposition of all SN Ia GW creates a background:

```
Ω_GW(f) = ∫ dz × R(z) × h²(D(z)) / H(z)
```

where R(z) is the SN Ia rate and D(z) evolves per our model.

### 4.2 Spectral Signature

If D evolves with z:
- High-z SNe: Higher D → Stronger individual events
- But: Further away → Lower strain at Earth

Net effect: The background spectrum may show deviations from a simple power law.

### 4.3 Prediction

| Scenario | Spectral Index | Notes |
|----------|----------------|-------|
| D = const | -2/3 | Standard power law |
| D(z) evolves | Varies with f | May show "pink" features |
| Super-Chandra | Different slope | Extreme D → different spectrum |

---

## 5. Multi-Messenger Strategy

### 5.1 Detection Scenarios

**Scenario A: DECIGO detects WD merger → Optical follow-up**
- GW alert triggers SN Ia monitoring
- Light curve gives x₁ (stretch)
- Test: Does high WD mass → high x₁?

**Scenario B: Optical SN Ia → DECIGO searches for explosion GW**
- Nearby SN Ia (< 1 Mpc) detected optically
- DECIGO searches for ~1 Hz burst
- Test: Does high x₁ → high GW amplitude?

### 5.2 Key Observables

| Observable | Probe | Spandrel Connection |
|------------|-------|---------------------|
| Merger GW chirp mass | Progenitor mass | WD mass → D? |
| Explosion GW amplitude | Asymmetry | D → h scaling |
| Optical x₁ | Flame turbulence | D → x₁ calibration |

---

## 6. Timeline and Feasibility

### 6.1 Near-term (2025-2030)

- **Current:** No direct GW detection capability for SN Ia
- **LIGO O5 (2027):** Core-collapse possible, SN Ia unlikely
- **B-DECIGO (2030s):** First decihertz sensitivity

### 6.2 Long-term (2030-2050)

- **B-DECIGO:** Detect nearby (< 100 kpc) SN Ia explosions
- **DECIGO:** Detect WD mergers at cosmological distances
- **Multi-messenger:** Combine GW + optical for D(z) test

### 6.3 Required Developments

1. **Simulation:** 3D simulations of D → GW amplitude mapping
2. **Templates:** Waveform templates for matched filtering
3. **Theory:** Better understanding of global vs. local asymmetry

---

## 7. Caveats and Uncertainties

### 7.1 Conservative Assessment

The Spandrel Framework's GW predictions are **highly uncertain**:

1. **Global vs. Local Asymmetry:**
   - If flame wrinkling is local (cancels at large scales), GW is weak
   - If global asymmetry develops, GW could be strong

2. **Order of Magnitude:**
   - Original estimate: h ~ 10⁻²² at 10 kpc
   - May be 10-100× lower in practice

3. **D → h Scaling:**
   - Theoretical; needs simulation verification
   - Current 3D runs don't resolve GW-relevant scales

### 7.2 What Would Confirm Spandrel GW

A definitive test requires:
- Detection of explosion GW from a nearby SN Ia
- Optical measurement of x₁ for same event
- Correlation between x₁ and GW amplitude

This is a **2040s+ prospect** with DECIGO.

---

## 8. Conclusion

GW observations offer a **future probe** of the Spandrel Framework:

1. **WD merger statistics** can constrain progenitor mass evolution
2. **Explosion GW** may directly measure flame asymmetry
3. **Multi-messenger** x₁-GW correlations would test D → observables

However, this is **long-term science** requiring DECIGO (2040s+). Current priorities should focus on:
- JWST high-z stretch measurements
- DESI RSD null test
- 3D simulations of D(Z)

---

## References

1. Maselli et al. 2019, arXiv:1910.01063 — WD-WD merger GW
2. Falta et al. 2011, PRL 106, 201103 — SN Ia explosion GW
3. DECIGO Collaboration 2021, PTEP 05A105 — DECIGO status

---

**Analysis Date:** 2025-11-28
**Confidence:** Low-Medium (theoretical framework, needs simulation)
