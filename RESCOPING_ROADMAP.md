# COSMOS Framework: Rescoping Roadmap

**Version:** December 2025 (Post-Review Restructuring)

This document defines the transformation of COSMOS from a single "unified framework" paper into a multi-paper research program with explicit dependency tracking and falsification criteria.

---

## Executive Summary

### The Problem
The current COSMOS manuscript bundles three orthogonal research directions (exceptional algebra, discrete cosmogenesis, SN astrophysics) into a single "tension resolution" narrative. This:
- Triggers the field's immune response against "grand unified numerology"
- Makes individual claims non-falsifiable in isolation
- Conflates calibration with prediction
- Presents heuristic analyses as statistical detections

### The Solution
Split into **three focused papers** plus an **umbrella note**, each with:
- Single, well-defined hypothesis
- Explicit likelihood/methodology
- Clear falsification criteria
- Independent reproducibility

---

## Repository Structure (New)

```
cosmos/
├── papers/
│   ├── paper1-spandrel/              # SN Ia systematics (HIGHEST ROI)
│   │   ├── manuscript/
│   │   ├── code/
│   │   ├── data/
│   │   └── figures/
│   ├── paper2-h0-smoothing/          # H₀(R) methodology
│   │   ├── manuscript/
│   │   ├── code/
│   │   ├── data/
│   │   └── figures/
│   ├── paper3-ccf-curvature/         # CCF/OR theory
│   │   ├── manuscript/
│   │   ├── code/
│   │   ├── data/
│   │   └── figures/
│   └── umbrella-note/                # Short program overview
│       └── manuscript/
├── src/cosmos/                       # Shared analysis code
├── data/                             # Shared datasets
├── docs/                             # Documentation
├── legacy/                           # Original unified paper (archived)
└── RESCOPING_ROADMAP.md             # This document
```

---

## Paper 1: Spandrel (HIGHEST PRIORITY)

### Title
"Does SN Ia Progenitor Evolution Mimic Evolving Dark Energy? A Hierarchical Test Using Host Galaxy Proxies"

### Core Hypothesis
The DESI DR2 preference for w₀ > -1, wₐ < 0 (at 2.8-4.2σ) arises primarily from unmodeled SN Ia population evolution correlated with host galaxy properties, not from true dark energy dynamics.

### Deliverables

| Deliverable | Description | Status |
|-------------|-------------|--------|
| Hierarchical SN likelihood | Latent M, SALT2 covariates, host proxies, selection | NOT STARTED |
| Metallicity-split test | Primary falsification: bias vs. host proxy | NOT STARTED |
| Multi-dataset comparison | Pantheon+, DES-SN, Union3 | NOT STARTED |
| DESI impact analysis | Δ(w₀,wₐ) with/without evolution term | NOT STARTED |

### Model Specification

```
Hierarchical Model:
├── Population level:
│   ├── M₀(z) = M₀ + αz × z           # Intrinsic luminosity evolution
│   ├── x₁(z) ~ N(μ_x₁(z), σ_x₁)     # Stretch population drift
│   └── c(z) ~ N(μ_c(z), σ_c)        # Color population drift
├── Individual SN level:
│   ├── M_i = M₀(z_i) + α×x₁_i + β×c_i + γ×M_host_i
│   └── μ_obs_i ~ N(μ_cosmo(z_i) + M_i, σ_obs)
└── Selection:
    └── P(detect | M, z, host) = f(Malmquist, host mass, survey)
```

### Falsification Criteria

| Test | Outcome → Interpretation |
|------|--------------------------|
| Split by host mass proxy | Bias detected → Spandrel supported |
| w₀wₐ shift after correction | Δσ > 1.5 → Spandrel explains signal |
| Cross-dataset consistency | Inconsistent → selection effects dominate |

### Code Requirements
- `src/cosmos/analysis/hierarchical_sn_model.py` (NEW)
- `src/cosmos/analysis/spandrel_likelihood.py` (NEW)
- Integration with `cobaya` or `emcee` for sampling

### Timeline
- Month 1: Hierarchical model implementation
- Month 2: Pantheon+ fit with host covariates
- Month 3: DESI impact analysis
- Month 4: Paper draft

---

## Paper 2: Scale-Dependent H₀

### Title
"Local Expansion Rate as a Function of Smoothing Scale: A Physically Defined H₀(R) Estimator and Null Tests"

### Core Hypothesis
The Hubble tension reflects genuine scale-dependence in the locally-inferred expansion rate, detectable when measurements are binned by a physically-defined smoothing scale R (not heuristic wavenumber k).

### The Critical Fix
**Current problem:** "k" values are assigned heuristically with no physical justification.
**Solution:** Define H₀(R) where R is an explicit smoothing scale:

```
Definition options:
├── Option A: Top-hat radius for SN subsamples
│   └── R = characteristic distance to SN host galaxies in each method
├── Option B: Window function for peculiar velocity field
│   └── R = filter scale in reconstructed v(r) field
└── Option C: Calibration volume
    └── R = effective radius of distance ladder anchors
```

### Deliverables

| Deliverable | Description | Status |
|-------------|-------------|--------|
| H₀(R) definition | Explicit, unambiguous R assignment | NOT STARTED |
| ΛCDM null distribution | Cosmic variance mocks at each R | NOT STARTED |
| Homogeneous dataset test | Single method spanning multiple R | NOT STARTED |
| Significance reassessment | p-value against null, not "4.7σ" | NOT STARTED |

### Falsification Criteria

| Test | Outcome → Interpretation |
|------|--------------------------|
| ΛCDM mock comparison | Trend within null → no detection |
| Homogeneous dataset | No trend → heterogeneous systematics explain |
| Multiple R definitions | Definition-dependent → artifact |

### Code Requirements
- `src/cosmos/analysis/h0_smoothing_scale.py` (NEW)
- `src/cosmos/analysis/lcdm_h0_mocks.py` (NEW)
- Explicit R assignment function

### What Gets Removed from Current Paper
- "4.7σ detection" language
- Any claim of "significant" scale-dependence
- Heuristic k mapping table

### Timeline
- Month 1: R definition and mock pipeline
- Month 2: Reanalysis with proper null
- Month 3: Paper draft

---

## Paper 3: CCF/OR Curvature

### Title
"Ollivier-Ricci Curvature Convergence Under Bigraph Rewriting Dynamics: Conditions, Counterexamples, and Tests"

### Core Hypothesis
Under appropriate conditions, Ollivier-Ricci curvature on evolving bigraphs converges to Ricci curvature in the continuum limit, providing a route from discrete dynamics to emergent gravity.

### The Critical Problem
**Current state:** Simulations show κ_OR ~ -N^{0.55} (diverging), not converging to flat-space expectation (κ → 0).
**Diagnosis:** Operating outside the regime where convergence theorems apply (graphs remain disconnected).

### Deliverables

| Deliverable | Description | Status |
|-------------|-------------|--------|
| Known-regime reproduction | Convergence on connected RGG | NOT STARTED |
| Divergence diagnosis | Why CCF regime fails | PARTIALLY DONE |
| Rewriting constraints | Rules that preserve convergence | NOT STARTED |
| Condition taxonomy | When convergence is/isn't expected | NOT STARTED |

### Required Demonstrations (Before Any "Emergent GR" Language)

```
Step 1: Reproduce known results
├── Connected random geometric graphs
├── Correct radius scaling (r ~ N^{-1/d})
└── Verify κ_OR → κ_Ricci for known manifolds

Step 2: Identify failure modes
├── Why current CCF graphs diverge
├── Connectivity analysis
└── Scaling regime mismatch

Step 3: Derive constraints
├── What rewriting rules preserve convergence
├── What initial conditions are required
└── What coupling to matter/fields is needed
```

### Falsification Criteria

| Test | Outcome → Interpretation |
|------|--------------------------|
| Known regime test | Fails → implementation error |
| CCF regime test | Diverges → CCF needs modification |
| Constrained rewriting | Still diverges → CCF not viable route to GR |

### What Gets Removed from Current Paper
- "Emergent GR" language
- "Route to Einstein equations" claims
- Any implication that convergence is demonstrated

### Honest Replacement Language
> "CCF provides a phenomenological dynamical network model. The connection to continuum gravity remains an open theoretical problem, with current simulations showing divergence rather than convergence in the tested regime."

### Code Requirements
- `src/cosmos/engines/or_convergence_tests.py` (NEW)
- `src/cosmos/engines/connected_rgg_baseline.py` (NEW)
- Comparison with van der Hoorn et al. results

### Timeline
- Month 1: Known-regime baseline
- Month 2: Failure mode analysis
- Month 3: Constrained rewriting exploration
- Month 4: Paper draft

---

## Umbrella Note

### Title
"COSMOS: A Falsification Program Linking Exceptional Algebra Motifs, Discrete Cosmogenesis, and SN Systematics"

### Purpose
Short (6-10 pages) overview that:
- Lists hypotheses and their dependency graph
- Lists falsification tests and timelines
- Does NOT claim "tension resolution"
- Does NOT include BIC/χ² comparisons
- Points to individual papers for details

### Structure

```
1. Introduction (1 page)
   - Three research directions, loosely connected
   - Explicit statement: "not a unified theory"

2. Hypothesis Inventory (2 pages)
   - Table of claims with status and dependencies
   - Distinguish calibration vs. prediction

3. Dependency Graph (1 page)
   - Visual DAG showing what depends on what
   - Identify independent vs. correlated tests

4. Falsification Program (2 pages)
   - Table: Hypothesis | Test | Data | Timeline | Failure condition
   - Emphasis on near-term testable predictions

5. ε = 1/4 as Motif (1 page)
   - Explicitly: "organizing motif, not evidence"
   - Dependency analysis (only 2 independent appearances)

6. Conclusion (1 page)
   - Invitation for critical evaluation
   - Explicit limitations
```

### What This Note Does NOT Contain
- BIC comparisons
- "Preferred over ΛCDM" language
- Statistical significance claims
- "Tension resolved" framing

---

## Dependency Graph

```
                    ε = 1/4 (MOTIF)
                         │
         ┌───────────────┼───────────────┐
         │               │               │
         ▼               ▼               ▼
    ┌─────────┐    ┌──────────┐    ┌──────────┐
    │   AEG   │    │   CCF    │    │ Spandrel │
    │ Algebra │    │ Bigraph  │    │   SNe    │
    └────┬────┘    └────┬─────┘    └────┬─────┘
         │              │               │
         ▼              ▼               ▼
    ┌─────────┐    ┌──────────┐    ┌──────────┐
    │ CP phase│    │  H₀(R)   │    │ DESI w₀wₐ│
    │(1.9σ)   │    │(heuristic│    │ signal   │
    └─────────┘    │  → R)    │    └──────────┘
                   └──────────┘
                        │
                        ▼
                   ┌──────────┐
                   │ OR→Ricci │
                   │(DIVERGES)│
                   └──────────┘
```

### Independence Analysis

| Claim | Depends On | Independent? |
|-------|------------|--------------|
| w₀ = -5/6 | ε calibration | NO (calibrated) |
| n_s = 0.966 | λ, η calibration | NO (calibrated) |
| S₈ = 0.78 | α calibration | NO (calibrated) |
| δ_CP = 67.8° | Octonion geometry | YES (but 1.9σ) |
| r = 0.0048 | CCF dynamics | YES (testable) |
| Spandrel bias | SN population | YES (testable) |
| OR convergence | Graph dynamics | YES (but fails) |

---

## Immediate Actions (This Week)

### 1. Create Directory Structure
```bash
mkdir -p papers/{paper1-spandrel,paper2-h0-smoothing,paper3-ccf-curvature,umbrella-note}/{manuscript,code,data,figures}
```

### 2. Archive Current Paper
```bash
mv paper/ legacy/unified-paper-v1/
```

### 3. Extract Spandrel Content
- Move relevant sections from `cosmos_paper.tex`
- Move `src/cosmos/models/spandrel_*.py`
- Move `docs/research/SPANDREL_*.md`

### 4. Remove/Quarantine from Any Public Version
- [ ] Table 7 (DESI χ² comparison)
- [ ] "4.7σ" H₀ gradient claim
- [ ] "Emergent GR" language
- [ ] "Tension resolved" framing
- [ ] BIC preference claims without reproducibility package

---

## Medium-Term Milestones

| Milestone | Target Date | Deliverable |
|-----------|-------------|-------------|
| Spandrel hierarchical model | +1 month | Working likelihood code |
| H₀(R) definition paper | +2 months | Draft with ΛCDM mocks |
| OR convergence baseline | +2 months | Reproduced known results |
| Spandrel arXiv v1 | +3 months | Full paper submission |
| H₀(R) arXiv v1 | +4 months | Methodology paper |
| CCF curvature arXiv v1 | +5 months | Theory + simulations |
| Umbrella note | +6 months | After individual papers |

---

## Success Criteria

### Paper 1 (Spandrel) Success
- [ ] Hierarchical model fits Pantheon+ with host covariates
- [ ] Metallicity split shows predicted bias pattern
- [ ] DESI+SN w₀wₐ preference reduced by ≥1σ after correction
- [ ] Cross-dataset consistency demonstrated

### Paper 2 (H₀ Smoothing) Success
- [ ] Physically unambiguous R definition
- [ ] ΛCDM null distribution computed
- [ ] Either: trend detected beyond null, OR honest null result reported

### Paper 3 (CCF Curvature) Success
- [ ] Known OR convergence reproduced
- [ ] Failure mode diagnosed
- [ ] Either: convergence achieved with constraints, OR honest "open problem" documented

### Overall Program Success
- [ ] At least one paper survives external reproduction
- [ ] At least one falsifiable prediction tested
- [ ] Framework earns "serious if speculative" rather than "grand numerology" classification

---

## What Changes in Rhetoric

### Remove
- "COSMOS resolves cosmological tensions"
- "Unified framework explains..."
- "Multiple independent confirmations of ε = 1/4"
- "4.7σ detection of scale-dependent H₀"
- "Preferred over ΛCDM"

### Replace With
- "COSMOS proposes testable hypotheses for..."
- "ε = 1/4 is an organizing motif that generates..."
- "Exploratory analysis suggests scale-dependence, pending rigorous definition"
- "Spandrel predicts that host-corrected SNe will show..."
- "CCF dynamics require demonstrated convergence before cosmological interpretation"

---

*Document generated: December 2025*
*Status: Active rescoping in progress*
