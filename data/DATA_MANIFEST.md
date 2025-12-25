# COSMOS Project Data Manifest

**Generated:** December 2025
**Total Data Volume:** ~61 MB (raw) + ~28 KB (processed)
**Disk Budget Used:** 0.4% of available (14 GB available, 3.5 GB budget)

---

## Data Directory Structure

```
data/
├── raw/                          # 61 MB - Primary observational data
│   ├── bao_data/                 # 58 MB - BAO measurements (DESI, SDSS)
│   ├── nuclear/                  # 2.0 MB - AME2020 atomic mass tables
│   ├── pdg_2024/                 # 40 KB - Particle physics data
│   ├── lhc_data/                 # 20 KB - LHC collision data
│   ├── gw_sensitivity/           # 20 KB - GW detector curves
│   ├── jwst_jades/               # 12 KB - JWST high-z SNe
│   ├── weak_lensing/             # 8 KB - KiDS/DES results
│   ├── cmb_s4/                   # 4 KB - CMB-S4 forecasts
│   ├── planck_2018/              # 4 KB - Planck parameters
│   ├── Pantheon+SH0ES.dat        # 568 KB - Type Ia supernovae
│   ├── lcparam_full_long.txt     # 68 KB - Light curve parameters
│   └── wd_pulsation_periods_table.csv
├── processed/                    # 28 KB - Analysis outputs
│   └── *.npz, *.json             # Regeneratable from scripts
└── references/
    └── references.bib            # BibTeX database
```

---

## Data Sources by Category

### 1. Cosmology & Supernovae

| Dataset | Files | Size | Source | Reference |
|---------|-------|------|--------|-----------|
| **Pantheon+ SNe Ia** | `Pantheon+SH0ES.dat` | 568 KB | Pantheon+ Collaboration | Brout et al. 2022, ApJ 938, 110 |
| Light curve params | `lcparam_full_long.txt` | 68 KB | SH0ES | Riess et al. 2022 |
| Pantheon variants | `pantheon_*.csv` | 136 KB | Derived | Metallicity bins |

### 2. Baryon Acoustic Oscillations

| Dataset | Directory | Size | Source | Reference |
|---------|-----------|------|--------|-----------|
| **DESI DR2** | `bao_data/desi_bao_dr2/` | ~30 KB | DESI Collaboration | DESI 2024-2025 |
| DESI 2024 | `bao_data/desi_2024_*` | ~32 KB | CobayaSampler | DESI Collaboration |
| SDSS DR12/16 | `bao_data/sdss_*` | ~26 MB | SDSS | BOSS/eBOSS |
| BAO consensus | `bao_data/*consensus*` | ~2 KB | Combined | Multiple surveys |

**Tracers included:** LRG (z=0.4-0.8), ELG (z=1.1-1.6), QSO (z=0.8-2.1), Lyman-alpha, BGS (z=0.1-0.4)

### 3. Nuclear Physics

| Dataset | Files | Size | Source | Reference |
|---------|-------|------|--------|-----------|
| **AME2020 masses** | `mass_1.mas20` | 473 KB | AMDC IMPCAS | Huang et al. 2021, CPC 45 |
| AME2020 rounded | `massround.mas20` | 429 KB | AMDC IMPCAS | Wang et al. 2021, CPC 45 |
| AME2020 reactions | `rct1.mas20`, `rct2_1.mas20` | 1.0 MB | AMDC IMPCAS | Wang et al. 2021 |

**URL:** https://amdc.impcas.ac.cn/masstables/Ame2020/

### 4. Particle Physics (PDG 2024)

| Dataset | Files | Size | Source | Reference |
|---------|-------|------|--------|-----------|
| Mass/width table | `mass_width_2024.txt` | 32 KB | PDG | Navas et al. 2024, PRD 110 |
| Summary YAML | `pdg_2024_summary.yaml` | 8 KB | Compiled | PDG 2024 |

**Includes:** Gauge bosons, leptons, quarks, CKM matrix, coupling constants

### 5. LHC Data

| Dataset | Files | Size | Source | Reference |
|---------|-------|------|--------|-----------|
| CMS v2 flow | `cms_small_system_v2.csv` | 614 B | CMS | PRL 132, 172302 (2024) |
| ATLAS O-O quenching | `atlas_oo_quenching_2025.csv` | 577 B | ATLAS | Preliminary 2025 |
| W boson mass | `w_boson_mass_2025.csv` | 517 B | CMS/ATLAS/CDF | CMS 2024 |
| **ALICE O-O flow** | `alice_oo_nene_flow_2025.yaml` | 6 KB | ALICE | arXiv:2509.06428 |

### 6. JWST/JADES High-z Supernovae

| Dataset | Files | Size | Source | Reference |
|---------|-------|------|--------|-----------|
| SN 2023adsy | `sn2023adsy_data.yaml` | 8 KB | JADES | Vinko & Regos 2025, A&A |
| Survey summary | `jades_transient_survey_summary.yaml` | 4 KB | JADES | DeCoursey et al. 2025 |

**Key result:** First Type Ia at z=2.9, SALT3-NIR x1 = 2.11-2.39

### 7. CMB & Planck

| Dataset | Files | Size | Source | Reference |
|---------|-------|------|--------|-----------|
| Planck 2018 | `planck_2018_parameters.yaml` | 4 KB | Planck | A&A 641, A6 (2020) |
| CMB-S4 forecasts | `cmb_s4_forecasts.yaml` | 4 KB | CMB-S4 | arXiv:2008.12619 |

**Note:** Full Planck MCMC chains (~9 GB) available at https://pla.esac.esa.int/

### 8. Weak Lensing

| Dataset | Files | Size | Source | Reference |
|---------|-------|------|--------|-----------|
| Survey compilation | `weak_lensing_surveys.yaml` | 8 KB | KiDS/DES/HSC | Multiple 2021-2023 |

**Key result:** S8 = 0.773 ± 0.012 (combined), 4σ tension with Planck

### 9. Gravitational Wave Detectors

| Dataset | Files | Size | Source | Reference |
|---------|-------|------|--------|-----------|
| LISA sensitivity | `LISA_sensitivity.dat` | 16 KB | LISA Consortium | Robson et al. 2019 |
| Einstein Telescope | `ET-0000A-18.txt` | 4 KB | ET Consortium | ET Design Study |

---

## Data Quality & Provenance

### Validation Status

| Category | Status | Notes |
|----------|--------|-------|
| Pantheon+ SNe | ✓ Validated | Cross-checked with light curves |
| DESI BAO | ✓ Validated | Includes full covariance matrices |
| AME2020 | ✓ Validated | Official AMDC release |
| PDG 2024 | ✓ Validated | Official PDG release |
| ALICE O-O | ⚠ Reconstructed | Awaiting HEPData upload |
| JWST JADES | ✓ Validated | From peer-reviewed papers |
| Weak lensing | ✓ Validated | Published survey results |

### Covariance Matrices

| Dataset | Covariance Available |
|---------|---------------------|
| DESI BAO | ✓ Full covariance (individual + combined) |
| SDSS BAO | ✓ Consensus covariance |
| H0 measurements | ✓ 12×12 with systematics |
| Pantheon+ | ✓ Statistical + systematic |
| KiDS/DES | ✓ Available from survey websites |

---

## Missing Data (Future Work)

| Dataset | Status | Size Estimate | Priority |
|---------|--------|---------------|----------|
| Planck MCMC chains | Not downloaded | 9 GB | Medium |
| JWST Cycle 4 SNe | Future data | TBD | High |
| CMB-S4 real data | Future (~2029) | TBD | Future |
| DESI Y5 RSD | Future (~2029) | TBD | High |
| Roman high-z SNe | Future (~2027) | TBD | High |
| LIGO O5 events | Future (~2027) | TBD | Medium |

---

## Citations

All data usage should cite the original sources. Key citations:

```bibtex
@article{Brout2022,
  author = {Brout, D. and others},
  title = {The Pantheon+ Analysis},
  journal = {ApJ},
  volume = {938},
  pages = {110},
  year = {2022}
}

@article{DESI2024,
  author = {{DESI Collaboration}},
  title = {DESI 2024 VI: Cosmological Constraints},
  journal = {arXiv:2404.03002},
  year = {2024}
}

@article{Planck2020,
  author = {{Planck Collaboration}},
  title = {Planck 2018 results. VI. Cosmological parameters},
  journal = {A\&A},
  volume = {641},
  pages = {A6},
  year = {2020}
}

@article{PDG2024,
  author = {Navas, S. and others},
  title = {Review of Particle Physics},
  journal = {Phys. Rev. D},
  volume = {110},
  pages = {030001},
  year = {2024}
}
```

---

## Regenerating Processed Data

All files in `data/processed/` can be regenerated:

```bash
# From project root
python scripts/run_full_analysis.py
python scripts/run_density_sweep.py
python src/cosmos/analysis/h0_covariance.py
```

---

*Manifest last updated: December 2025*
