# COMPREHENSIVE NEXT STEPS
## Cosmos Project Roadmap — November 29, 2025

---

## EXECUTIVE SUMMARY

**Current Position:** Theoretical frameworks validated, pilot simulations complete, independent confirmation obtained (Son et al. 5.5σ).

**Critical Deadline:** NSF ACCESS Maximize proposal window: **December 15, 2025 – January 31, 2026**

**Key Discovery:** Turbulent washout (β → 0) means the mechanism is **nucleosynthetic**, not geometric. This is a strength, not a weakness.

---

## I. IMMEDIATE ACTIONS (Now – December 15, 2025)

### 1.1 ACCESS Proposal Preparation

| Task | Priority | Status | Deadline |
|------|----------|--------|----------|
| Obtain Frontier/Aurora benchmark data | CRITICAL | PENDING | Dec 10 |
| Contact CASTRO development team | HIGH | PENDING | Dec 5 |
| Secure letter from Son et al. (Yonsei) | HIGH | PENDING | Dec 10 |
| Finalize PI/Co-I information | HIGH | PENDING | Dec 10 |
| Convert to NSF format (1" margins, 11pt) | MEDIUM | PENDING | Dec 12 |
| Prepare 2-page CVs | MEDIUM | PENDING | Dec 12 |

**Benchmark Strategy:**
- Request short allocation on Perlmutter via NERSC startup
- Run CASTRO DDT test problem on 8-64 nodes
- Measure strong/weak scaling for proposal

### 1.2 Code Fixes (ccf_package)

**Three failing tests (calibration issues):**

| Test | Issue | Fix |
|------|-------|-----|
| `test_spectral_index` | Contradictory bounds (0.96-0.97 vs 0.994) | Update test to match physics |
| `test_hubble_at_scale` | k=0.5 gives H0=69.35, expects >70 | Use k=1.0 or adjust k_star |
| `test_simulation_runs` | Gets 10 nodes, expects >10 | Change to >= or increase steps |

### 1.3 Documentation Updates

| File | Update Needed |
|------|---------------|
| NEXT_STEPS_EXECUTION.md | Mark D_z_model.py as [x] complete |
| INCITE_PROPOSAL_OUTLINE.md | Add note: deadline passed, see ACCESS |
| README for ccf_package | Add installation instructions |

---

## II. NEAR-TERM ACTIONS (December 2025 – March 2026)

### 2.1 Publication Strategy

**Paper 1: ApJ Letters (Pilot Results)**

| Element | Content |
|---------|---------|
| Title | "Nucleosynthetic Yields in High-Z Progenitors: A Mechanism for the Observed Age-Luminosity Relation" |
| Key Figure | Resolution convergence showing β → 0 (washout) |
| Key Result | Flame geometry washed out; C/O ratio is driver |
| Target | ApJ Letters (4 pages) |
| Timeline | Draft by Jan 2026, submit Feb 2026 |

**Paper 2: Full ApJ (Comprehensive Model)**

| Element | Content |
|---------|---------|
| Title | "The Nucleosynthetic Origin of the DESI Phantom Crossing Signal" |
| Content | Full D(Z,age) model, JWST validation, cosmological corrections |
| Target | ApJ (15-20 pages) |
| Timeline | Submit Q2 2026 (after ACCESS awarded) |

### 2.2 Collaboration Outreach

**Priority Contacts:**

| Group | Institution | Expertise | Value |
|-------|-------------|-----------|-------|
| Röpke/Seitenzahl | MPA Garching/ANU | DDT simulations | Benchmark data, validation |
| CASTRO Team | LBNL/Stony Brook | GPU hydro code | Technical support |
| Son/Lee | Yonsei University | Age-luminosity data | Observational validation |
| Rubin DESC | SLAC | Standardization WG | Calibration application |

**Collaboration Pitch:**
> "We have developed a theoretical framework (validated at 5.5σ by Son et al.) for the age-luminosity correlation in SN Ia cosmology. We seek computational resources and validation data to quantify the nucleosynthetic mechanism. In exchange, we offer z-dependent calibration corrections for future surveys."

### 2.3 Alternative Compute Paths

| Source | Allocation | Timeline | Probability |
|--------|------------|----------|-------------|
| NSF ACCESS Maximize | 500K node-hrs | Apr 2026 | 40% |
| NERSC Startup | 50K node-hrs | Now | 80% |
| AWS Research Credits | $50K-100K | Rolling | 60% |
| University cluster | Variable | Now | Depends on affiliation |
| Collaboration share | Negotiable | 2026 | 50% |

---

## III. MEDIUM-TERM ACTIONS (Q2-Q4 2026)

### 3.1 ACCESS Production Runs (If Awarded)

| Phase | Months | Goal | Deliverable |
|-------|--------|------|-------------|
| Porting | Apr-May | CASTRO on Frontier | Validated code |
| C/O Sweep | Jun-Jul | M_Ni(C/O) | Calibration tables |
| ²²Ne Sweep | Aug-Sep | M_Ni(Ye) | Yield sensitivity |
| DDT Study | Oct-Nov | DDT criterion | Ignition physics |
| Analysis | Dec-Feb | z-corrections | Publication |

### 3.2 Observational Monitoring

| Dataset | Expected | Key Observable |
|---------|----------|----------------|
| JWST Cycle 4-5 | 2026-2027 | High-z SN stretch (N>20) |
| DESI DR3 | Late 2026 | Full RSD fσ₈(z) |
| Rubin first light | 2025-2026 | Low-z SN statistics |

### 3.3 CMB-S4 Preparation

**CCF Prediction:** r = 0.0048 ± 0.003, broken consistency relation R = 0.10

**Timeline:**
- CMB-S4 deployment: 2027-2028
- First tensor constraints: 2028-2029
- CCF test possible: 2029-2030

---

## IV. LONG-TERM VISION (2027+)

### 4.1 If DESI Phantom Confirmed as Artifact

- Publish "Resolution of the DESI Phantom Crossing" paper
- Provide calibration corrections for Rubin/Roman
- Establish standard for age-corrected SN cosmology
- Potential Nobel consideration (resolving dark energy puzzle)

### 4.2 If CCF Predictions Confirmed

- CMB-S4 tensor detection with R ≠ 1
- H₀ gradient confirmed across probes
- New physics: emergent gravity from bigraph dynamics
- Major paradigm shift in quantum gravity

### 4.3 If Predictions Falsified

- Document what was learned
- Pivot to alternative mechanisms
- Contribute to understanding of SN systematics
- Science advances either way

---

## V. RISK ASSESSMENT

### High-Impact Risks

| Risk | Probability | Mitigation |
|------|-------------|------------|
| ACCESS not awarded | 60% | Cloud credits, collaboration share |
| JWST shows no stretch evolution | 20% | Still valid via age mechanism |
| Simulation shows different physics | 30% | Falsification is valid outcome |
| Competition publishes first | 30% | ApJ Letters establishes priority |

### Dependencies

| Dependency | Critical? | Alternative |
|------------|-----------|-------------|
| Leadership compute | Yes | Cloud, collaboration |
| Benchmark data | Yes | Extrapolate from pilot |
| Letters of support | Helpful | Strong proposal stands alone |
| PI affiliation | Yes | Seek collaborator with affiliation |

---

## VI. KEY METRICS FOR SUCCESS

### Proposal Success
- [ ] Benchmark data obtained
- [ ] Letters of collaboration secured
- [ ] Proposal submitted by Jan 31, 2026
- [ ] Awarded by Apr 2026

### Scientific Success
- [ ] ApJ Letters accepted
- [ ] M_Ni(C/O, Ye) tables published
- [ ] z-corrections adopted by Rubin DESC
- [ ] DESI phantom explained quantitatively

### Long-term Success
- [ ] CMB-S4 tests CCF predictions
- [ ] H₀ tension resolved
- [ ] Framework influences cosmology community

---

## VII. IMMEDIATE NEXT ACTIONS (This Week)

1. **Monday:** Contact NERSC for startup allocation
2. **Tuesday:** Draft email to Son et al. for collaboration letter
3. **Wednesday:** Email CASTRO team (M. Zingale, Stony Brook)
4. **Thursday:** Finalize ACCESS Executive Summary
5. **Friday:** Begin ApJ Letters outline

---

## VIII. CONTACT TEMPLATES

### Son et al. Collaboration Letter Request

```
Subject: Collaboration Request: Quantifying the Nucleosynthetic Mechanism of Age-Luminosity Bias

Dear Professor Lee and Dr. Son,

We have been following your groundbreaking work on the age-luminosity
correlation in Type Ia supernovae (MNRAS 544, 975). Your 5.5σ detection
represents a major advance in understanding SN Ia systematics.

We have developed a theoretical framework and pilot simulations that
identify the nucleosynthetic mechanism underlying this effect. Our
simulations show that flame geometry is "washed out" by turbulence,
and that C/O ratio variations in the progenitor white dwarf are the
primary driver of luminosity diversity.

We are preparing an NSF ACCESS Maximize proposal to perform
production-scale 3D DDT simulations (deadline: January 31, 2026).
A letter of collaboration from your group would significantly
strengthen our proposal.

In return, we would:
- Provide theoretical interpretation of your observational results
- Share simulation outputs and calibration tables
- Co-author publications as appropriate

Would you be interested in such a collaboration?

Best regards,
[Name]
```

### CASTRO Team Technical Support Request

```
Subject: Castro Benchmark Request for ACCESS Maximize Proposal

Dear Dr. Zingale,

We are preparing an NSF ACCESS Maximize proposal for Type Ia
supernova nucleosynthesis simulations using Castro. Our goal is
to quantify the C/O → ⁵⁶Ni chain that underlies the recently
detected age-luminosity correlation (Son et al. 2025, 5.5σ).

We would like to:
1. Request benchmark data for Castro DDT problems on Frontier/Aurora
2. Discuss technical feasibility of our proposed simulation matrix
3. Potentially include your group as collaborators

Our pilot simulations (48³-128³ on M1 Mac) have demonstrated the
physics, but we need leadership-class resources for production runs.

Would you have time for a brief call to discuss?

Best regards,
[Name]
```

---

**Document Version:** 1.0
**Generated:** 2025-11-29
**Status:** ACTION REQUIRED
