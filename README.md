# COSMOS: A Research Physics Simulatorium

## Overview
The COSMOS project is a comprehensive research physics simulatorium designed to explore and validate the Algebraic-Entropic Gravity (AEG) framework, Spandrel Framework, and other cosmological models. It integrates high-performance compute engines, symbolic derivation tools, and data analysis pipelines into a unified ecosystem.

## Directory Structure

### `src/cosmos/` - Core Source Code
*   **`core/`**: Fundamental mathematical libraries (e.g., `octonion_algebra`, `qcd_running`) and core physics modules.
*   **`engines/`**: Heavy-duty simulation engines (e.g., `ccf_bigraph_engine`, `m1_compute_engine`, `flame_box_mps`).
*   **`models/`**: Specific physical models (e.g., `spandrel_cosmology`, `D_z_model`).
*   **`analysis/`**: Analytical scripts, derivation verifications, and data processing tools.

### `data/` - Data Assets
*   **`raw/`**: Raw observational data (e.g., Pantheon+ SNe Ia datasets).
*   **`processed/`**: Processed intermediate files (`.npz`, `.json`).
*   **`references/`**: Bibliographic data.

### `docs/` - Documentation
*   **`papers/`**: Monographs, synthesis papers, and LaTeX source files.
*   **`proposals/`**: Research grant proposals.
*   **`reports/`**: Status reports, audits, and gap analyses.
*   **`research/`**: General research notes and experimental roadmaps.

### `scripts/` - Execution Scripts
Top-level scripts for running analyses and simulations.
*   `run_full_analysis.py`: Main pipeline for AEG framework analysis.
*   `run_density_sweep.py`: Progenitor density sweep analysis.
*   `run_mcmc_optimized.py`: MCMC parameter estimation.

### `tests/` - Validation
Unit tests and physics validation scripts.

### `output/`
*   **`plots/`**: Generated figures and plots.

## Installation
This project is structured as a Python package. To install in editable mode:

```bash
pip install -e .
```

## Usage
To run the full analysis pipeline:

```bash
python3 scripts/run_full_analysis.py
```

To run specific validations:

```bash
python3 tests/validate_full_chain.py
```
