# Paper 1: Spandrel - SN Ia Population Evolution

**Status:** HIGHEST PRIORITY | Development Phase

## Title (Working)
"Does SN Ia Progenitor Evolution Mimic Evolving Dark Energy? A Hierarchical Test Using Host Galaxy Proxies"

## Core Hypothesis
The DESI DR2 preference for evolving dark energy (w₀ > -1, wₐ < 0 at 2.8-4.2σ) arises primarily from unmodeled Type Ia supernova population evolution correlated with host galaxy properties, not from true cosmological dynamics.

## Falsification Test
**Primary:** Split SN sample by host metallicity proxy (mass, SFR, sSFR) and check for systematic bias in Hubble residuals.

| Outcome | Interpretation |
|---------|---------------|
| Bias detected, correlated with z | Spandrel supported |
| No bias after host correction | Spandrel falsified |
| Δ(w₀,wₐ) > 1σ after correction | Spandrel explains DESI signal |

## Directory Structure
```
paper1-spandrel/
├── README.md                    # This file
├── manuscript/
│   ├── spandrel_paper.tex       # Main LaTeX document
│   ├── spandrel.bib             # Bibliography
│   └── figures/                 # Paper figures
├── code/
│   ├── hierarchical_sn_model.py # Stan/PyMC model
│   ├── spandrel_likelihood.py   # Likelihood implementation
│   ├── host_proxy_analysis.py   # Host galaxy analysis
│   └── run_analysis.py          # Main analysis script
├── data/
│   ├── pantheon_plus/           # Pantheon+ data
│   ├── des_sn/                  # DES SN data
│   └── host_properties/         # Host galaxy catalogs
└── figures/
    └── generated/               # Analysis output figures
```

## Hierarchical Model Specification

```python
# Population-level parameters
M0 = Normal(mu=-19.3, sigma=0.1)           # Baseline absolute magnitude
alpha_z = Normal(mu=0, sigma=0.1)          # Luminosity evolution with z
alpha = Normal(mu=0.14, sigma=0.01)        # Stretch coefficient
beta = Normal(mu=3.1, sigma=0.1)           # Color coefficient
gamma = Normal(mu=0.05, sigma=0.02)        # Host mass coefficient

# Population drift
mu_x1_z = a_x1 + b_x1 * z                  # Stretch population drift
mu_c_z = a_c + b_c * z                     # Color population drift

# Individual SN
for i in range(N_sn):
    x1[i] ~ Normal(mu_x1_z[i], sigma_x1)
    c[i] ~ Normal(mu_c_z[i], sigma_c)
    M[i] = M0 + alpha_z * z[i] + alpha * x1[i] - beta * c[i] + gamma * log10(M_host[i]/1e10)
    mu_obs[i] ~ Normal(mu_cosmo(z[i], w0, wa) - M[i], sigma_obs[i])
```

## Required Data

| Dataset | Source | Status |
|---------|--------|--------|
| Pantheon+ | Scolnic et al. 2022 | Available |
| DES-SN5YR | DES Collaboration | Available |
| Union3 | Rubin et al. 2023 | TBD |
| Host masses | GAMA, SDSS | Partial |
| Host metallicities | MaNGA proxies | TBD |

## Deliverables Checklist

- [ ] Hierarchical model in Stan/PyMC
- [ ] Pantheon+ fit with baseline model
- [ ] Host mass split analysis
- [ ] Metallicity proxy implementation
- [ ] DESI BAO + corrected SN joint fit
- [ ] Δ(w₀,wₐ) quantification
- [ ] Cross-dataset validation (Pantheon+ vs DES)
- [ ] Paper draft
- [ ] Reproducibility package

## Key References

1. Scolnic et al. 2022 - Pantheon+ sample
2. DESI Collaboration 2024 - BAO + dark energy
3. Rigault et al. 2020 - Host galaxy effects
4. Rose et al. 2021 - SN standardization systematics
5. Popovic et al. 2023 - Selection effects

## Success Criteria
1. Hierarchical model converges on real data
2. Host mass split shows statistically significant bias pattern
3. w₀wₐ constraints shift by ≥1σ after host correction
4. Results consistent across Pantheon+ and DES-SN

---
*Last updated: December 2025*
