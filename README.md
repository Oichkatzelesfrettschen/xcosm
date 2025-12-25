# XCOSM: eXceptional COSMological Framework

A multi-paper research program exploring connections between **exceptional algebraic structures** (octonions, Jordan algebras, F₄/E₆/G₂), **discrete pregeometry** (bigraph dynamics with Ollivier-Ricci curvature), and **observational cosmology** (SNe Ia systematics, scale-dependent expansion).

**Status:** Active research with explicit falsification tests.

## Quick Start

```bash
# Clone and setup
git clone https://github.com/eirikr/xcosm.git
cd xcosm

# Option 1: Use virtual environment (recommended)
make venv
source .venv/bin/activate

# Option 2: Install directly
make install-dev

# Run tests
make test

# Build papers
make papers
```

## Repository Structure

```
xcosm/
├── src/xcosm/                    # Python package
│   ├── core/                     # Algebraic structures (octonions, QCD, entropic cosmology)
│   ├── models/                   # Physical models (Spandrel, fractals, EOS)
│   ├── engines/                  # Simulation engines (CCF bigraph, flame, hydro)
│   ├── data/                     # Unified data loaders (Pantheon+, BAO, LHC, GW)
│   └── analysis/                 # Analysis scripts and derivations
├── papers/                       # LaTeX research papers
│   ├── paper1-spandrel/          # SNe Ia population evolution (HIGHEST PRIORITY)
│   ├── paper2-h0-smoothing/      # Scale-dependent H₀ methodology
│   ├── paper3-ccf-curvature/     # CCF/OR curvature convergence
│   └── umbrella-note/            # Program overview
├── data/                         # Datasets (Pantheon+, DESI, Planck, etc.)
├── tests/                        # Test suite
├── docs/                         # Documentation
└── Makefile                      # Build orchestration
```

## Research Papers

### Paper 1: Spandrel (HIGHEST PRIORITY)
**"Does SN Ia Progenitor Evolution Mimic Evolving Dark Energy?"**

Tests whether DESI's apparent dark energy evolution signal arises from unmodeled SN Ia population systematics (metallicity → fractal dimension → luminosity bias).

- **Falsification test:** Host mass/metallicity split
- **Directory:** `papers/paper1-spandrel/`

### Paper 2: Scale-Dependent H₀
**"Local Expansion Rate as a Function of Smoothing Scale"**

Replaces heuristic k-mapping with physically defined H₀(R) estimator.

- **Falsification test:** Trend vs ΛCDM cosmic variance mocks
- **Directory:** `papers/paper2-h0-smoothing/`

### Paper 3: CCF Curvature Convergence
**"Ollivier-Ricci Curvature Under Bigraph Rewriting"**

Investigates when OR curvature converges to Ricci curvature.

- **Current state:** Divergence observed (κ_OR ~ -N^{0.55})
- **Directory:** `papers/paper3-ccf-curvature/`

## Key Features

### Core Module (`xcosm.core`)
- **Octonion algebra** with Fano plane multiplication
- **Jordan algebras** (27-dimensional J₃(O))
- **QCD running** (4-loop α_s, mass evolution)
- **Entropic cosmology** (MCMC fitting)
- **Partition functions** (discrete-to-continuum bridge)

### Data Module (`xcosm.data`)
```python
from xcosm.data import load_pantheon, load_desi_bao

pantheon = load_pantheon()
low_z = pantheon.select(z_max=0.1)
bao = load_desi_bao('dr2')
```

### Engine Module (`xcosm.engines`)
- **CCF Bigraph Engine**: Cosmological bigraph rewriting
- **Flame Box 3D**: Spectral Navier-Stokes turbulence
- **Riemann Hydro**: HLLC Riemann solver

## Make Targets

| Category | Target | Description |
|----------|--------|-------------|
| **Setup** | `make venv` | Create virtual environment |
| | `make install-dev` | Install with dev dependencies |
| **Test** | `make test` | Run all tests |
| | `make test-cov` | Tests with coverage report |
| **Quality** | `make lint` | Run ruff linter |
| | `make format` | Format with black |
| **Papers** | `make papers` | Build all papers |
| | `make paper1` | Build Paper 1 only |
| **Utility** | `make verify` | Check build dependencies |
| | `make stats` | Show project statistics |
| | `make help` | Show all targets |

## Key Distinctions

### What We Claim
- ε = 1/4 is an **organizing motif** for hypothesis generation
- Spandrel proposes a **testable systematic** affecting DESI dark energy inference
- CCF provides a **phenomenological toy model** requiring convergence demonstration

### What We Do NOT Claim
- ~~"XCOSM resolves cosmological tensions"~~ (overstated)
- ~~"4.7σ detection of scale-dependent H₀"~~ (heuristic k, not rigorous)
- ~~"Emergent GR from bigraphs"~~ (convergence not demonstrated)
- ~~"Multiple independent confirmations"~~ (most ε appearances are calibrated)

## Falsified Hypotheses

| Hypothesis | Test Result | Status |
|------------|-------------|--------|
| φ⁻ⁿ fermion masses | χ²/dof = 35,173 | **REJECTED** |
| Bigraph κ_OR → 0 | κ ~ -N^{0.55} | **DIVERGING** |
| Strong CP (original) | θ >> 10⁻¹⁰ | **FALSIFIED** |

See [docs/FALSIFICATIONS.md](docs/FALSIFICATIONS.md) for details.

## Requirements

- Python ≥ 3.8
- NumPy, SciPy, Matplotlib, Pandas, NetworkX
- LaTeX (for papers): latexmk, pdflatex, revtex4-2

## Contributing

This is active research. Critical evaluation welcome. See individual paper READMEs for contribution opportunities.

## License

MIT License. See `LICENSE` for details.

---
*XCOSM v0.2.0 | December 2025*
