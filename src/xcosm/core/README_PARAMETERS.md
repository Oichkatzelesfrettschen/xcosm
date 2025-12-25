# CCFParameters - Canonical Implementation

This is the **single source of truth** for CCFParameters in the COSMOS project.

## Location
`/Users/eirikr/1_Workspace/cosmos/src/cosmos/core/parameters.py`

## Usage

### Import (Recommended)
```python
from cosmos.core.parameters import CCFParameters, SimulationConfig

params = CCFParameters()
print(params.spectral_index)      # Property access: 0.9660
print(params.tensor_to_scalar)    # Property access: 0.0048
```

### Backwards Compatible Import
```python
from ccf.parameters import CCFParameters  # Re-exports from cosmos.core

params = CCFParameters()
```

## Features

### Core Parameters
All fundamental CCF parameters calibrated to November 2025 observations:
- `lambda_inflation = 0.003` - Inflation decay rate
- `eta_curvature = 0.028` - Curvature coupling
- `alpha_attachment = 0.85` - Preferential attachment exponent
- `epsilon_tension = 0.25` - Link tension parameter
- `k_star = 0.01` - Crossover scale (Mpc⁻¹)
- `h0_cmb = 67.4` - CMB-scale Hubble constant (km/s/Mpc)
- `h0_gradient = 1.15` - H₀ gradient (km/s/Mpc/decade)

### Derived Observables (Properties)
Access cosmological observables as properties:
- `spectral_index` → n_s = 0.9660
- `tensor_to_scalar` → r = 0.0048
- `s8_parameter` → S_8 = 0.8225
- `dark_energy_eos` → w₀ = -0.8333
- `w0_dark_energy` → (alias for dark_energy_eos)
- `wa_dark_energy` → wₐ = -0.70

### Methods
- `hubble_at_scale(k)` - Compute H₀ at wavenumber k
- `to_dict()` - Serialize to dictionary
- `__str__()` - Human-readable representation

### Backwards Compatibility Methods
For legacy code that calls methods with parentheses:
- `spectral_index_method()`
- `tensor_to_scalar_method()`
- `s8_parameter_method()`
- `dark_energy_eos_method()`

## Design Philosophy

The class uses a **dual interface**:

1. **Properties** (modern, preferred): `params.spectral_index`
2. **Methods** (legacy support): `params.spectral_index_method()`

This ensures compatibility with both:
- Engine code expecting properties: `self.params.spectral_index`
- Legacy code calling methods: `params.spectral_index()`

## Migration Notes

All previous CCFParameters implementations have been consolidated:

1. ✅ `ccf_package/ccf/parameters.py` - Now imports from here
2. ✅ `cosmos/engines/ccf_bigraph_engine.py` - Now imports from here
3. ✅ `cosmos/engines/ccf_h0_gradient_analysis.py` - Now imports from here
4. ✅ `cosmos/engines/ccf_bigraph_enhanced.py` - Now imports from here

**No code changes required** - all existing imports work through re-exports and backwards compatibility.

## Calibration

Parameters calibrated to:
- **Planck 2018 + ACT DR6**: n_s = 0.966 ± 0.004
- **DESI DR2**: w₀ = -0.83 ± 0.05, wₐ = -0.70 ± 0.25
- **KiDS-Legacy**: S₈ = 0.815 ± 0.018
- **BICEP/Keck 2024**: r < 0.032 (95% CL)
- **SH0ES 2024**: H₀ = 73.17 ± 0.86 km/s/Mpc (local)

## Example

```python
from cosmos.core.parameters import CCFParameters

# Create parameters with defaults
params = CCFParameters()

# Access derived observables
print(f"Spectral index: {params.spectral_index:.4f}")
print(f"Tensor-to-scalar: {params.tensor_to_scalar:.4f}")
print(f"S8 parameter: {params.s8_parameter:.4f}")
print(f"Dark energy w0: {params.w0_dark_energy:.4f}")
print(f"Dark energy wa: {params.wa_dark_energy:.2f}")

# Scale-dependent Hubble constant
for k in [0.0002, 0.01, 0.5]:
    h0 = params.hubble_at_scale(k)
    print(f"H0(k={k}): {h0:.2f} km/s/Mpc")

# Serialize
print(params.to_dict())

# Human-readable
print(params)
```

## Output
```
Spectral index: 0.9660
Tensor-to-scalar: 0.0048
S8 parameter: 0.8225
Dark energy w0: -0.8333
Dark energy wa: -0.70
H0(k=0.0002): 62.63 km/s/Mpc
H0(k=0.01): 67.40 km/s/Mpc
H0(k=0.5): 73.35 km/s/Mpc

CCFParameters(
  lambda = 0.003 -> n_s = 0.9660
  alpha  = 0.85 -> S_8 = 0.823
  epsilon = 0.25 -> w_0 = -0.833
  r = 0.0048
  H0_CMB = 67.4 km/s/Mpc
  H0_gradient = 1.15 km/s/Mpc/decade
)
```

## See Also
- `/Users/eirikr/1_Workspace/cosmos/CONSOLIDATION_SUMMARY.md` - Full consolidation details
- `cosmos.core.spacetime_projection` - Spacetime geometry
- `cosmos.engines.ccf_bigraph_engine` - Bigraph simulation engine
