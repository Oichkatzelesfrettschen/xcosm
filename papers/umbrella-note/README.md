# Umbrella Note: COSMOS Program Overview

**Status:** To be written AFTER individual papers

## Title (Working)
"COSMOS: A Falsification Program Linking Exceptional Algebra Motifs, Discrete Cosmogenesis, and SN Systematics"

## Purpose
A short (6-10 page) overview that:
- Lists hypotheses and their dependency structure
- Lists falsification tests with explicit timelines
- Points to individual papers for technical details
- Does NOT claim "tension resolution"

## What This Note Is NOT
- A unified theory paper
- A "tension resolved" announcement
- A BIC/χ² comparison paper
- A statistical significance claim paper

## Structure

### 1. Introduction (1 page)
- Three loosely connected research directions
- Explicit framing: "research program with testable hypotheses"
- NOT: "unified framework that resolves tensions"

### 2. Hypothesis Inventory (2 pages)

| ID | Hypothesis | Domain | Testable? | Current Status |
|----|------------|--------|-----------|----------------|
| H1 | Spandrel explains DESI w₀wₐ | SN astrophysics | YES | Paper 1 |
| H2 | H₀ shows scale-dependence | Cosmology | YES (with R def) | Paper 2 |
| H3 | OR curvature → Ricci | Discrete gravity | YES | Paper 3 |
| H4 | δ_CP from octonion geometry | Particle physics | Marginal (1.9σ) | Appendix |
| H5 | ε = 1/4 unifies physics | Theory | NO (mostly calibration) | Section 5 |

### 3. Dependency Graph (1 page)

```
Independent Inputs:
├── BH entropy formula (Bekenstein-Hawking)
├── Percolation threshold (p_c = 1/4 for certain lattices)
└── Octonion structure constants

Calibrated Quantities (NOT predictions):
├── w₀ = -5/6 (calibrated to DESI)
├── n_s = 0.966 (calibrated to Planck)
├── S₈ = 0.78 (calibrated to weak lensing)
└── H₀ gradient slope (calibrated to measurements)

Genuine Predictions:
├── Spandrel: host-correlated SN bias
├── r = 0.0048 (tensor modes)
├── δ_CP = 67.8° (but now 1.9σ from data)
└── OR convergence (currently fails)
```

### 4. Falsification Program (2 pages)

| Hypothesis | Test | Data/Experiment | Timeline | Failure Condition |
|------------|------|-----------------|----------|-------------------|
| Spandrel | Host-split SN residuals | Pantheon+/DES | 2025 | No bias pattern |
| H₀(R) | Trend vs ΛCDM mocks | Multi-method | 2025-2026 | Within null |
| OR convergence | Connected regime test | Simulation | 2025 | Still diverges |
| r = 0.0048 | B-mode detection | CMB-S4 | 2029+ | r < 0.003 |
| δ_CP | γ precision | LHCb/Belle II | 2026+ | > 3σ from 67.8° |

### 5. ε = 1/4 as Organizing Motif (1 page)

**Explicit statement:**
> "The appearance of ε = 1/4 in multiple contexts is used as an organizing motif for hypothesis generation, NOT as evidence of a deep connection. Of the six appearances listed, only two (BH entropy, percolation) are genuinely independent of the cosmological calibration targets."

**Dependency analysis:**
- w₀ = -5/6 → explicitly calibrated to DESI
- n_s = 0.966 → explicitly calibrated to Planck
- S₈ = 0.78 → explicitly calibrated to weak lensing
- BH entropy → independent (Bekenstein-Hawking)
- Percolation → independent (lattice theory)
- Jordan algebra → theoretical connection (unproven)

### 6. Conclusion (1 page)
- Invitation for critical evaluation
- Explicit limitations recap
- Emphasis on falsifiability over confirmation

## Writing Constraints

### Must Include
- [ ] Explicit calibration vs prediction table
- [ ] Dependency DAG
- [ ] Falsification conditions for each hypothesis
- [ ] Timeline for experimental tests
- [ ] Honest assessment of current status

### Must NOT Include
- [ ] BIC comparisons without reproducibility package
- [ ] "Preferred over ΛCDM" language
- [ ] "4.7σ detection" claims
- [ ] "Emergent GR" without convergence demonstration
- [ ] "Tension resolved" framing

## Prerequisites
This note should be written AFTER:
1. Paper 1 (Spandrel) submitted
2. Paper 2 (H₀ smoothing) submitted
3. Paper 3 (CCF curvature) submitted

The umbrella note references and summarizes the individual papers.

## Target Length
6-10 pages including figures and tables

## Target Venue
arXiv:astro-ph.CO with cross-list to hep-th

## Tools Created

### 1. `hypothesis_matrix.py`
Generates hypothesis dependency analysis:
- Classifies hypotheses as independent, calibrated, derived, or predictions
- Counts genuinely independent inputs vs calibrated parameters
- Creates visual dependency graph
- Exports LaTeX tables

**Usage:**
```bash
python hypothesis_matrix.py --output-dir figures/
python hypothesis_matrix.py --no-plot  # Print summary only
```

**Outputs:**
- `figures/hypothesis_dependency.pdf` - Dependency graph visualization
- `figures/hypothesis_dependency.png` - PNG version
- `figures/hypothesis_dependency_table.tex` - LaTeX table
- Console summary with critical analysis

**Key insight:** Shows that only 2 of 6 ε = 1/4 appearances are independent.

### 2. `falsification_table.py`
Documents all falsification tests (falsified, pending, marginal):
- Tracks hypothesis status with explicit failure conditions
- Includes timelines for experimental tests
- Exports to LaTeX and JSON formats

**Usage:**
```bash
python falsification_table.py --output-dir tables/
python falsification_table.py --format latex  # LaTeX only
python falsification_table.py --format json   # JSON only
```

**Outputs:**
- `tables/falsified_hypotheses.tex` - Falsified tests table
- `tables/pending_tests.tex` - Pending tests table
- `tables/falsification_program.tex` - Combined comprehensive table
- `tables/falsification_program.json` - JSON export for reproducibility
- Console summary of all tests

**Falsified hypotheses:**
- φ⁻ⁿ fermion masses (χ²/dof = 2,931)
- Bigraph κ_OR → Ricci convergence (diverges as -N^0.55)
- Strong CP original formulation (θ >> 10⁻¹⁰)

**Pending tests:**
- Spandrel host bias (2025 Q1-Q2, Paper 1)
- H₀ scale-dependence (2025-2026, Paper 2)
- r = 0.0048 tensor modes (2029-2032, CMB-S4)
- δ_CP = 67.8° (2025-2028, LHCb/Belle II, currently 1.9σ)
- OR convergence in connected regime (2025 Q2, Paper 3)

### 3. `outline.tex`
Complete LaTeX outline for 6-10 page umbrella note:
- Structured according to README plan
- Explicit "What We Do NOT Claim" section
- References to individual papers
- NO BIC comparisons or "tension resolved" language

**Compile:**
```bash
cd /Users/eirikr/1_Workspace/cosmos/papers/umbrella-note
pdflatex outline.tex
bibtex outline
pdflatex outline.tex
pdflatex outline.tex
```

**Prerequisites:**
- Individual papers (1-3) must be complete before writing
- Tables generated by `falsification_table.py`
- Figures generated by `hypothesis_matrix.py`

## Installation

Required Python packages:
```bash
pip install numpy matplotlib networkx
```

## Workflow

1. Complete individual papers (Papers 1-3)
2. Run `hypothesis_matrix.py` to generate dependency analysis
3. Run `falsification_table.py` to generate test tables
4. Fill in `outline.tex` with results from individual papers
5. Compile LaTeX document
6. Submit to arXiv after individual papers are published

---
*Last updated: December 2025*
