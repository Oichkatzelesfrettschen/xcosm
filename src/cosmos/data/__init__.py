"""
COSMOS Data Module
==================

Unified data access interface for all COSMOS datasets.

Available Datasets
------------------

**Cosmology / Supernovae:**
- `load_pantheon()` - Pantheon+ SNe Ia (1701 supernovae, 0.001 < z < 2.26)
- `PantheonDataset` - Full dataset class with selection methods

**Baryon Acoustic Oscillations:**
- `load_desi_bao()` - DESI DR1/DR2 BAO measurements
- `load_sdss_bao()` - SDSS DR12/DR16 BAO measurements
- `BAODataset` - Combined BAO dataset class

**Particle Physics (LHC):**
- `load_cms_flow()` - CMS small-system collectivity data
- `load_atlas_quenching()` - ATLAS O-O jet quenching
- `load_w_mass()` - Precision W boson mass measurements

**Gravitational Waves:**
- `load_lisa_sensitivity()` - LISA detector sensitivity curve
- `load_et_sensitivity()` - Einstein Telescope sensitivity curve

Examples
--------
>>> from cosmos.data import load_pantheon, load_desi_bao
>>>
>>> # Load Pantheon+ supernovae
>>> pantheon = load_pantheon()
>>> print(f"Loaded {len(pantheon)} supernovae")
>>> low_z = pantheon.select(z_max=0.1)
>>> print(f"Low-z sample: {len(low_z)} SNe")
>>>
>>> # Load DESI BAO data
>>> bao = load_desi_bao('dr2')
>>> for m in bao:
...     print(f"z={m.z_eff:.2f}: D_M/r_d={m.dm_over_rd:.1f}")

Data Sources
------------
- Pantheon+: https://github.com/PantheonPlusSH0ES/DataRelease
- DESI BAO: https://github.com/CobayaSampler/bao_data
- SDSS BAO: SDSS DR12/DR16 public releases
- CMS/ATLAS: Public data from CERN experiments
- LISA/ET: Consortium sensitivity estimates
"""

# Base utilities
from .base import (
    get_data_root,
    get_raw_data_path,
    get_processed_data_path,
    DatasetInfo,
    SPEED_OF_LIGHT_KM_S,
    H0_PLANCK,
    H0_SHOES,
    OMEGA_M_PLANCK,
)

# Pantheon+ supernovae
from .pantheon import (
    PantheonDataset,
    PantheonSelection,
    SupernovaRecord,
    load_pantheon,
)

# BAO measurements
from .bao import (
    BAODataset,
    BAOMeasurement,
    load_desi_bao,
    load_sdss_bao,
)

# LHC particle physics
from .lhc import (
    CMSFlowDataset,
    ATLASQuenchingDataset,
    WMassDataset,
    FlowMeasurement,
    JetQuenchingMeasurement,
    WMassMeasurement,
    load_cms_flow,
    load_atlas_quenching,
    load_w_mass,
)

# Gravitational wave sensitivity
from .gw_sensitivity import (
    LISASensitivity,
    EinsteinTelescopeSensitivity,
    SensitivityCurve,
    load_lisa_sensitivity,
    load_et_sensitivity,
)

__all__ = [
    # Base
    'get_data_root',
    'get_raw_data_path',
    'get_processed_data_path',
    'DatasetInfo',
    'SPEED_OF_LIGHT_KM_S',
    'H0_PLANCK',
    'H0_SHOES',
    'OMEGA_M_PLANCK',
    # Pantheon
    'PantheonDataset',
    'PantheonSelection',
    'SupernovaRecord',
    'load_pantheon',
    # BAO
    'BAODataset',
    'BAOMeasurement',
    'load_desi_bao',
    'load_sdss_bao',
    # LHC
    'CMSFlowDataset',
    'ATLASQuenchingDataset',
    'WMassDataset',
    'FlowMeasurement',
    'JetQuenchingMeasurement',
    'WMassMeasurement',
    'load_cms_flow',
    'load_atlas_quenching',
    'load_w_mass',
    # GW
    'LISASensitivity',
    'EinsteinTelescopeSensitivity',
    'SensitivityCurve',
    'load_lisa_sensitivity',
    'load_et_sensitivity',
]
