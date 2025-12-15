# Computational Cosmogenesis Framework (CCF)

A Python package for simulating emergent spacetime from bigraphical reactive systems.

## Overview

CCF provides a computational framework for cosmology based on Robin Milner's bigraph theory. The framework models spacetime as an evolving bigraphical reactive system, where:

- **Nodes** represent spacetime points
- **Place graph** encodes geometric structure
- **Link graph** encodes entanglement/connectivity
- **Rewriting rules** drive cosmological evolution

## Key Features

- Bigraph simulation engine with cosmological rewriting rules
- H0 gradient analysis demonstrating scale-dependent expansion
- Ollivier-Ricci curvature computation (converges to GR in continuum limit)
- CMB-S4 tensor mode predictions
- Full Standard Model gauge group emergence from motif automorphisms

## Installation

```bash
pip install ccf-cosmology
```

Or from source:

```bash
git clone https://github.com/ccf-collaboration/ccf-cosmology
cd ccf-cosmology
pip install -e ".[dev]"
```

## Quick Start

```python
from ccf import BigraphEngine, CCFParameters

# Initialize with default parameters
params = CCFParameters()
engine = BigraphEngine(params)

# Run simulation
result = engine.run_simulation()

# Access results
print(f"H0: {result.hubble_parameter:.2f} km/s/Mpc")
print(f"n_s: {result.spectral_index:.4f}")
print(f"w_0: {result.dark_energy_eos:.3f}")
```

## CCF Parameters

| Parameter | Symbol | Default | Observable |
|-----------|--------|---------|------------|
| Slow-roll | lambda | 0.003 | n_s = 0.966 |
| Attachment | alpha | 0.85 | S_8 = 0.78 |
| Tension | epsilon | 0.25 | w_0 = -0.833 |
| Curvature | eta | 0.028 | ACT lensing |

## H0 Gradient Analysis

CCF predicts scale-dependent expansion:

```
H0(k) = 67.4 + 1.15 * log10(k/0.01) km/s/Mpc
```

This resolves the "Hubble tension" - both CMB and local measurements are correct at their respective scales.

```python
from ccf import H0GradientAnalysis

analysis = H0GradientAnalysis()
print(analysis.summary())
```

## CMB-S4 Predictions

CCF makes specific predictions testable by CMB-S4:

- Tensor-to-scalar ratio: r = 0.0048 +/- 0.003
- Broken consistency relation: R = r/(-8*n_t) = 0.10

```python
from ccf import predict_cmbs4_observables

preds = predict_cmbs4_observables()
print(f"r = {preds['r']['value']:.4f}")
print(f"Detection S/N = {preds['r']['detection_significance']:.1f}")
```

## Command Line Interface

```bash
# Run simulation
ccf-simulate simulate --steps 500 --seed 42

# Analyze H0 gradient
ccf-simulate h0-analysis --compare-ccf

# Show predictions
ccf-simulate predict --cmbs4

# Display parameters
ccf-simulate params
```

## Mathematical Foundations

CCF is built on rigorous mathematical foundations:

1. **Lorentz Invariance**: Causal poset structure embeds to Minkowski spacetime (Malament 1977)
2. **Einstein Equations**: Ollivier-Ricci curvature converges to Ricci tensor (van der Hoorn et al. 2023)
3. **Action Principle**: Variational derivation of rewriting rules from bigraph action
4. **Gauge Groups**: Standard Model emerges from motif automorphisms

## References

### Theoretical Foundations
- Milner, R. (2009). *The Space and Motion of Communicating Agents*
- Malament, D. (1977). J. Math. Phys. 18, 1399
- Jacobson, T. (1995). Phys. Rev. Lett. 75, 1260
- van der Hoorn et al. (2023). Discrete Comput. Geom.

### Observational Data
- Planck Collaboration (2020). A&A 641, A6
- DESI Collaboration (2024). arXiv:2404.03002
- Riess et al. (2024). ApJL 962, L17

## Citation

```bibtex
@article{CCF2025,
    author = {{CCF Collaboration}},
    title = "{The Computational Cosmogenesis Framework}",
    journal = {arXiv},
    year = {2025}
}
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please see CONTRIBUTING.md for guidelines.
