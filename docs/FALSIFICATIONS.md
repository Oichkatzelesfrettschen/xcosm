# COSMOS Framework: Falsified Hypotheses

**December 2025**

This document provides an honest accounting of hypotheses that have been tested and **falsified** within the COSMOS/CCF framework. Documenting falsifications strengthens scientific credibility and guides future theoretical development.

---

## 1. φ^(-n) Fermion Mass Hierarchy

### Hypothesis
Fermion masses follow a golden ratio scaling:
```
m_i = M₀ × φ^(-n_i)
```
where φ = (1 + √5)/2 ≈ 1.618 is the golden ratio.

### Test
- **Data:** PDG 2024 quark and lepton masses (9 fermions)
- **Method:** χ² fit with proper error propagation
- **Script:** `src/cosmos/analysis/fermion_mass_fit.py`

### Result
```
χ²/dof = 35,173 >> 1
p-value < 10^(-10)
```

### Status: **REJECTED**

### Implication
The golden ratio φ does not provide a universal scaling for fermion masses. While φ appears in F₄/AEG algebraic structures, it does not directly translate to mass ratios. Alternative mechanisms must be sought.

---

## 2. D(Z) Metallicity-Dependent Flame Fractal Dimension

### Hypothesis
Type Ia supernova flame fractal dimension D depends on metallicity Z:
```
D(Z) = D₀ + β × ln(Z/Z_⊙)
```
with β > 0, causing luminosity variations that explain the cosmological "phantom" dark energy signal.

### Test
- **Method:** High-resolution flame simulations at varying Z
- **Script:** `src/cosmos/analysis/production_DZ_sweep.py`
- **Grid:** 256³ to 2048³ resolution

### Result
```
β → 0 as resolution increases
Limit: β < 0.002 (effectively zero)
```

### Status: **FALSIFIED**

### Implication
The D(Z) mechanism cannot explain observed SNe Ia luminosity variations. The Spandrel framework now relies on C/O ratio effects (v4.0) instead of flame fractal dimension.

---

## 3. Bigraph Ricci Curvature Convergence

### Hypothesis
In the large-N limit, bigraph Ollivier-Ricci curvature converges to zero (flat spacetime emergence):
```
κ_OR(N) → 0 as N → ∞
```
This would connect CCF discrete geometry to continuous GR.

### Test
- **Method:** Direct calculation on bigraph ensembles
- **Script:** `src/cosmos/analysis/bigraph_convergence.py`
- **Ensemble:** N = 100 to N = 10,000 nodes

### Result
```
κ_OR ∼ -N^0.55 (diverging, not converging)
No plateau observed up to N = 10,000
```

### Status: **DIVERGING (Not Converging)**

### Implication
The naive bigraph → continuous limit does not produce flat spacetime. This suggests:
1. Additional structure is needed (e.g., Lorentzian vs Euclidean)
2. The coarse-graining map requires modification
3. Alternative discrete geometries may be needed

**This is an open theoretical problem.**

---

## 4. Strong CP Phase from Octonions (Original Form)

### Hypothesis
The strong CP phase θ_QCD is exactly:
```
θ_QCD = arctan(1/√7) × Im(e₁e₂e₃)
```
giving θ ≈ 0.36 (20.7°).

### Test
- **Data:** Neutron EDM bounds: |θ_QCD| < 10^(-10)
- **Script:** `src/cosmos/analysis/derive_strong_cp.py`

### Result
```
θ_predicted ≈ 0.36 >> 10^(-10)
```

### Status: **FALSIFIED** (original form)

### Resolution
The formula was revised to include dynamical relaxation:
```
θ_eff = θ_QCD × exp(-S_inst)
```
where S_inst ~ 10²⁶ from QCD instanton suppression, giving effectively θ_eff ≈ 0.

---

## Validated Predictions (For Comparison)

The following predictions have been **validated**:

| Prediction | CCF Value | Observed | Agreement |
|-----------|-----------|----------|-----------|
| w₀ | -0.833 | -0.83 ± 0.05 (DESI) | 0.07σ |
| n_s | 0.966 | 0.9649 ± 0.0042 (Planck) | 0.3σ |
| δ_CP | 67.79° | 65.4 ± 3.2° (PDG) | 0.7σ |
| H₀ gradient | 1.15 km/s/Mpc/dex | 1.70 ± 0.35 | 1.6σ |
| High-z x₁ | 2.08 | 2.11-2.39 (JWST) | <0.5σ |

---

## Pending Tests

| Prediction | Observable | Experiment | Timeline |
|-----------|-----------|------------|----------|
| r = 0.0048 | Tensor modes | CMB-S4 | 2029+ |
| R ≠ 1 | Consistency ratio | CMB-S4 | 2030+ |
| Gravity-D correlation | GW strain | DECIGO | Long-term |

---

## Lessons Learned

1. **Algebraic structures don't always map directly to physics**
   - φ appears in F₄ but doesn't set fermion masses
   - This doesn't invalidate the algebraic framework, just the naive interpretation

2. **Resolution matters in discrete models**
   - D(Z) washout only visible at high resolution
   - Bigraph convergence may require much larger N

3. **Falsifications guide theory development**
   - D(Z) failure → C/O ratio mechanism (Spandrel v4.0)
   - Strong CP failure → instanton suppression addition
   - Bigraph divergence → open research direction

4. **Honest accounting is essential**
   - Documenting failures alongside successes builds credibility
   - Failed hypotheses constrain parameter space

---

*Last updated: December 2025*
