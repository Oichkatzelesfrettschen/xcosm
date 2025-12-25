# Paper 3: CCF Curvature Analysis Code

This directory contains Python scripts for analyzing Ollivier-Ricci curvature convergence (or divergence) on CCF bigraphs.

## Files

### `or_curvature_baseline.py`

**Purpose:** Validate OR curvature implementation against known convergence results.

**What it does:**
- Implements Ollivier-Ricci curvature calculation from scratch
- Tests on three graph ensembles:
  1. **Flat torus (2D):** Expected convergence to κ_OR → 0
  2. **2-sphere:** Expected convergence to κ_OR → 1 (for unit sphere)
  3. **Disconnected graphs:** Expected divergence (negative control)

**Success criteria:**
- Flat torus MUST show convergence (validates implementation)
- Sphere MUST show convergence to positive value
- Only proceed to CCF tests if these pass

**Usage:**
```bash
python3 or_curvature_baseline.py
```

**Output:**
- `data/or_baseline_results.json` - Convergence data for all three tests
- `figures/or_baseline_convergence.pdf` - Comparison plots

**Runtime:** ~5-15 minutes depending on node counts

---

### `ccf_divergence_diagnosis.py`

**Purpose:** Diagnose WHY CCF bigraphs show curvature divergence.

**What it does:**
- Generates CCF bigraphs using standard rewriting rules
- Computes diagnostic metrics:
  - Connectivity (# components, largest component size)
  - Degree distribution (mean, std)
  - Ollivier-Ricci curvature (on largest component)
- Tests three hypotheses:
  1. **Disconnectivity:** CCF graphs remain disconnected
  2. **Scaling divergence:** κ_OR ~ -N^α with α > 0
  3. **Constraint effects:** Can modifications restore convergence?

**Key finding (expected):**
```
PRIMARY CAUSE: Disconnected graphs
- CCF bigraphs have O(N/k) components
- Violates van der Hoorn connectivity assumption
- Results in divergent curvature κ_OR ~ -N^0.55
```

**Usage:**
```bash
python3 ccf_divergence_diagnosis.py
```

**Output:**
- `data/ccf_divergence_diagnosis.json` - Full diagnostic data
- `figures/ccf_divergence_diagnosis.pdf` - Connectivity and scaling plots

**Runtime:** ~10-30 minutes (curvature computation is expensive)

---

## Dependencies

Required packages:
```bash
pip install numpy scipy networkx matplotlib tqdm
```

CCF package (from parent repository):
```bash
# Already available at /Users/eirikr/1_Workspace/cosmos/ccf_package
# Scripts automatically add to sys.path
```

## Workflow

**Step 1: Validate implementation**
```bash
python3 or_curvature_baseline.py
```

Check that flat torus and sphere tests PASS. If they fail, fix implementation before proceeding.

**Step 2: Diagnose CCF divergence**
```bash
python3 ccf_divergence_diagnosis.py
```

This runs the full diagnostic suite and generates plots.

**Step 3: Analyze results**

Review:
- Connectivity data: Are CCF graphs disconnected?
- Curvature scaling: What is the power law exponent?
- Comparison to baseline: How different is CCF from known regimes?

## Interpretation Guide

### Baseline Tests (or_curvature_baseline.py)

**Flat torus result:**
- Power law exponent α ≈ -0.5 → CONVERGENCE ✓
- κ_OR approaches 0 as N increases
- Validates implementation

**Sphere result:**
- Power law exponent α ≈ -0.5 → CONVERGENCE ✓
- κ_OR approaches 1.0 as N increases
- Confirms positive curvature detection

**Disconnected result:**
- Negative or unstable curvature → DIVERGENCE ✓
- Shows what happens when convergence assumptions fail
- Negative control

### CCF Diagnosis (ccf_divergence_diagnosis.py)

**Connectivity test:**
- If `fraction_connected` < 0.5 for all N → Disconnectivity confirmed
- Number of components should grow with N
- Largest component should shrink (as fraction)

**Curvature scaling:**
- If power law exponent α > 0 → DIVERGENCE
- If power law exponent α < 0 → CONVERGENCE
- Expected: α ≈ +0.55 (diverging negative curvature)

**Constraint test:**
- Currently theoretical (not fully implemented)
- Shows what WOULD be needed to fix CCF
- Future work direction

## Key Metrics

### Convergence Indicators

**CONVERGES if:**
- Power law exponent α < -0.3
- Graph is connected (or has giant component ≥ 80% nodes)
- Curvature magnitude decreases with N

**DIVERGES if:**
- Power law exponent α > 0
- Graph remains disconnected at all scales
- Curvature magnitude increases with N

### CCF Expected Results

Based on preliminary observations:
- Disconnectivity: YES (100% of realizations)
- Components: O(N/k) with k ≈ 5-10
- Power law exponent: α ≈ +0.55 ± 0.05
- Verdict: **DIVERGENCE**

## Troubleshooting

### Import errors
```
ModuleNotFoundError: No module named 'ccf'
```

**Fix:** Scripts should auto-add CCF package to path. If not, manually:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.home() / "1_Workspace/cosmos/ccf_package"))
```

### Memory errors (large N)

Curvature computation is O(E * d_max^2 * SP). For N > 1000:
- Reduce num_realizations
- Skip curvature computation (set `compute_curvature=False`)
- Use only connectivity diagnostics

### Slow runtime

Each curvature computation on N=500 graph takes ~1-5 seconds.
For 10 realizations × 5 node counts = 50-250 seconds.

Speed up:
- Reduce node_counts (fewer test sizes)
- Reduce num_realizations (less averaging)
- Use multiprocessing (not currently implemented)

## Citation

If using this code, cite:
```
[Author names]. "Ollivier-Ricci Curvature Divergence in CCF Bigraph Dynamics:
A Failure Mode Analysis." In preparation, 2025.
```

## License

[TBD - same as parent COSMOS repository]

## Contact

[TBD]
