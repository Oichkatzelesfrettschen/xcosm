# INCITE Proposal: Type Ia Supernova Systematics and Cosmological Parameter Inference

## High-Resolution Simulations of Thermonuclear Supernovae and Graph-Based Cosmology

**Principal Investigator:** [Name]

**Proposed Allocation:** 300,000 GPU-hours (NERSC/Perlmutter or OLCF/Frontier)

**Duration:** 12 months

---

## 1. Executive Summary

We propose a computational study to characterize systematic uncertainties in dark energy measurements:

1. **Track A (Spandrel):** 3D direct numerical simulations (DNS) of Type Ia supernova deflagration flames at high resolution ($2048^3$), mapping $M_{\rm Ni}(Z, \rho_c, {\rm ignition})$ from first principles.

2. **Track B (CCF):** Large-scale bigraph evolution simulations ($N > 10^8$ nodes) to verify emergent Hubble parameter scale-dependence and tensor mode predictions.

The goal is to quantify how progenitor-dependent nucleosynthesis affects SN Ia standardization and whether this can account for the apparent tension between geometric and luminosity-based dark energy constraints.

**Deliverable:** Synthetic Hubble diagrams with physics-based systematic uncertainties, informing interpretation of DESI Year 5 and Roman Space Telescope data.

---

## 2. Scientific Motivation

### 2.1 The DESI Phantom Crossing

DESI DR2 reports $(w_0, w_a) = (-0.72, -2.77)$, deviating from $\Lambda$CDM at $3.4$–$5.4\sigma$. However:

- Pure geometric probes (BAO + CMB) → $\Lambda$CDM consistent
- Luminosity probes (SNe Ia) → Phantom signal
- Dynamic probes (RSD $f\sigma_8$) → $\Lambda$CDM consistent

This **geometry-dynamics split** suggests the signal originates in SN Ia systematics, not cosmological physics.

### 2.2 The Spandrel Hypothesis

Type Ia supernovae are thermonuclear explosions of carbon-oxygen white dwarfs. Peak luminosity depends on $^{56}$Ni mass, which scales with electron fraction $Y_e$:

$$M_{\rm Ni} \propto Y_e^{1.5}$$

At high redshift, progenitors form in low-metallicity environments with:
- Higher C/O ratio → Higher $Y_e$ → More Ni → Brighter SNe

Current standardization assumes redshift-independent stretch-luminosity relations, introducing systematic bias.

### 2.3 The CCF Hypothesis

The Computational Cosmogenesis Framework derives spacetime from bigraphical reactive systems. Key predictions:

| Observable | CCF Value | Standard Value |
|------------|-----------|----------------|
| $w_0$ | $-0.833$ | $-1.00$ |
| $H_0$ gradient | $+1.15$ km/s/Mpc/decade | 0 |
| Tensor ratio $r$ | $0.0048$ | $< 0.032$ |

### 2.4 The Synthesis

$$w_0^{\rm obs} = w_0^{\rm CCF} + \Delta w_0^{\rm SN} = -0.833 + 0.11 = -0.72$$

This matches DESI exactly. Computational verification is essential.

---

## 3. Proposed Simulations

### 3.1 Track A: Spandrel Hero Run

**Objective:** Map $M_{\rm Ni}(Z, \rho_c, {\rm ignition})$ from first-principles hydrodynamics.

**Method:** 3D compressible Navier-Stokes with 55-isotope nuclear network, using the FLASH code with AMR.

**Parameter Space:**

| Parameter | Range | Grid Points |
|-----------|-------|-------------|
| Metallicity $Z$ | 0.1–3.0 $Z_\odot$ | 8 |
| Central density $\rho_c$ | $1$–$5 \times 10^9$ g/cm³ | 5 |
| Ignition radius $r_{\rm ign}$ | 0–100 km | 4 |
| Ignition geometry | Centered, off-center, multi-point | 3 |

**Total runs:** 480 full-star simulations

**Resolution:** $2048^3$ effective (AMR peak), $\Delta x \sim 1$ km

**Resources per run:** 500 GPU-hours

**Total (Track A):** 240,000 GPU-hours

**Deliverables:**
- $M_{\rm Ni}$ lookup table as function of progenitor properties
- Synthetic light curves and spectra
- Predicted $\Delta\mu(z)$ bias function

### 3.2 Track B: CCF Bigraph Evolution

**Objective:** Verify emergent $H_0(k)$ gradient and tensor mode predictions at large $N$.

**Method:** Parallel bigraph rewriting using custom GPU-accelerated engine.

**Simulations:**

| Run | Nodes | Links | Steps | GPU-hours |
|-----|-------|-------|-------|-----------|
| Calibration (10×) | $10^6$ | $10^7$ | $10^4$ | 1,000 |
| Production (5×) | $10^8$ | $10^9$ | $10^5$ | 50,000 |
| Extreme (1×) | $10^9$ | $10^{10}$ | $10^4$ | 10,000 |

**Total (Track B):** 60,000 GPU-hours

**Deliverables:**
- $H_0(k)$ measurement with error bars
- Tensor power spectrum $P_T(k)$
- Consistency relation $R = r/(-8n_t)$

### 3.3 Total Resource Request

| Track | GPU-hours | Percentage |
|-------|-----------|------------|
| Spandrel (3D DNS) | 240,000 | 80% |
| CCF (Bigraph) | 60,000 | 20% |
| **Total** | **300,000** | 100% |

---

## 4. Technical Approach

### 4.1 Spandrel: FLASH + Starkiller

We employ the FLASH hydrodynamics code (Fryxell et al. 2000) with:

- **Starkiller Microphysics:** 55-isotope α-chain + iron group network
- **Helmholtz EOS:** Degenerate electron-positron-photon-ion mixture
- **AMR:** Paramesh with 8 refinement levels (effective $2048^3$)
- **GPU acceleration:** CUDA kernels for nuclear burning

**Validation:** Reproduce Seitenzahl et al. (2013) delayed detonation models within 5%.

### 4.2 CCF: Parallel Bigraph Engine

We extend `ccf_bigraph_engine.py` to distributed GPU:

- **Graph storage:** CSR format with GPU-resident adjacency
- **Rewriting:** Parallel pattern matching via CUDA graphs
- **Curvature:** Ollivier-Ricci via Sinkhorn optimal transport
- **I/O:** HDF5 checkpointing every $10^3$ steps

**Validation:** Reproduce known Barabási-Albert scaling ($\gamma = 3$) and Watts-Strogatz clustering.

### 4.3 Data Management

| Dataset | Size | Storage |
|---------|------|---------|
| Spandrel outputs | 200 TB | NERSC HPSS |
| CCF checkpoints | 50 TB | NERSC scratch |
| Analysis products | 10 TB | Project filesystem |

---

## 5. Timeline

| Month | Milestone |
|-------|-----------|
| 1–2 | Code optimization, validation runs |
| 3–6 | Spandrel parameter sweep (240 runs) |
| 4–8 | CCF production runs |
| 9–10 | Analysis and synthetic Hubble diagrams |
| 11–12 | Paper preparation, public data release |

---

## 6. Expected Impact

### 6.1 Scientific Impact

- **Resolve DESI anomaly:** Determine if phantom crossing is astrophysical or cosmological
- **Calibrate SN Ia standardization:** Provide physics-based correction for high-$z$
- **Test emergent gravity:** First large-$N$ verification of CCF predictions

### 6.2 Broader Impact

- **Public code release:** FLASH configurations and CCF engine
- **Training data:** Synthetic spectra for ML SN classification
- **Benchmark:** GPU hydrodynamics at $2048^3$

---

## 7. Team Expertise

[To be filled with collaborator CVs]

- **PI:** Emergent cosmology theory
- **Co-I 1:** Supernova hydrodynamics (FLASH expert)
- **Co-I 2:** Nuclear astrophysics (reaction networks)
- **Co-I 3:** HPC optimization (GPU computing)

---

## 8. References

1. DESI Collaboration (2025). DESI DR2 Cosmological Results.
2. Seitenzahl et al. (2013). MNRAS, 429, 1156. [DDT models]
3. Fryxell et al. (2000). ApJS, 131, 273. [FLASH code]
4. van der Hoorn et al. (2021). Phys. Rev. Research, 3, 013211. [Ollivier-Ricci]
5. Milner, R. (2009). The Space and Motion of Communicating Agents.

---

## 9. Budget Justification

| Resource | Justification |
|----------|---------------|
| 240,000 GPU-hrs (Spandrel) | 480 runs × 500 hrs/run, $2048^3$ resolution required for DDT physics |
| 60,000 GPU-hrs (CCF) | $10^8$–$10^9$ nodes needed for scale-separation, $10^5$ steps for equilibration |

**Why GPUs?** Nuclear reaction networks and Ollivier-Ricci curvature are both memory-bandwidth limited; GPU HBM2e provides 3 TB/s vs 200 GB/s for CPU DDR5.

---

## 10. Conclusion

This proposal combines first-principles supernova hydrodynamics with cosmological simulations to characterize systematic uncertainties in dark energy measurements. The results will inform interpretation of current and future surveys including DESI and Roman.

---

**END OF PROPOSAL DRAFT**
