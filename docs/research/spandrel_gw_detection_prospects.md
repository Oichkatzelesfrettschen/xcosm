# Gravitational Wave Detection Prospects for Type Ia Supernovae
## Spandrel Framework Multi-Messenger Observation Protocol

**Compiled:** 2025-11-28
**Framework:** The Spandrel Framework D-parameter predictions
**Status:** Comprehensive literature review and mission assessment

---

## Executive Summary

The Spandrel Framework makes testable gravitational wave (GW) predictions for Type Ia supernovae based on the D-parameter:
- **Type Iax (D→2)**: h ~ 10⁻²⁵ at 10 kpc → **SILENT** (below all detector thresholds)
- **Normal SN Ia (D=2.2)**: h ~ 10⁻²² at 10 kpc → **DETECTABLE by DECIGO/BBO**
- **03fg-like (D→3)**: h ~ 10⁻²¹ at 10 kpc → **LOUD** (10× amplified, optimal for decihertz detectors)

**Critical Finding:** Detection requires **decihertz band (0.1-10 Hz)** sensitivity → DECIGO/BBO are the enabling missions. LIGO/Virgo cannot detect thermonuclear SN Ia GWs. LISA detects progenitor inspirals, not the explosion itself.

**Timeline for First Detection:** 2030s with B-DECIGO precursor; 2040s with full DECIGO

---

## 1. LIGO O4 Results on Type Ia Supernovae

### 1.1 O4 Observing Run Status
- **O4a:** May 24, 2023 – January 16, 2024
- **O4b:** April 10, 2024 – January 28, 2025
- **O4c:** January 28, 2025 – November 18, 2025
- **Total detections (O1-O4):** 290 gravitational wave events (200 in O4 alone)

### 1.2 SN 2023ixf: The Benchmark Core-Collapse Supernova Search

**Target:** SN 2023ixf (core-collapse Type II-L in M101, discovered May 19, 2023)
**Distance:** 6.7 Mpc
**Observing window:** 5 days on-source (14% coverage with ≥2 detectors operating)
**Result:** No gravitational waves detected

#### Upper Limits (Core-Collapse, NOT Type Ia):
- **GW energy:** E_GW < 1 × 10⁻⁵ M☉c² (at 50 Hz emission)
- **GW luminosity:** L_GW < 4 × 10⁻⁵ M☉c²/s (at 50 Hz)
- **Proto-neutron star ellipticity:** ε < 1.04 (at f > 1200 Hz)
- **Frequency range:** 50 Hz – 2 kHz

**Improvement:** ~10× more stringent than previous SN 2017eaw constraints from O1-O2

**Published:** Astrophysical Journal, 985(2):183 (May 2025)

### 1.3 Type Ia Supernova Searches: NONE REPORTED

**Critical Gap:** O4 publications focus on:
- Core-collapse supernovae (SN 2023ixf multi-messenger search)
- Binary black hole mergers (majority of detections)
- Neutron star mergers
- Burst searches (50-2000 Hz, optimized for core-collapse)

**Why No Type Ia Searches?**
Thermonuclear Type Ia supernovae produce GWs in the **decihertz band (0.1-10 Hz)**, far below LIGO's optimal sensitivity range (10-5000 Hz). Ground-based interferometers face insurmountable seismic noise below ~10 Hz.

**LIGO Limitation:**
- LIGO's strain sensitivity at 1 Hz: ~10⁻¹⁶ Hz⁻¹/² (dominated by seismic noise)
- Required for SN Ia detection at 10 kpc: ~10⁻²³ Hz⁻¹/² at 1 Hz
- **Gap:** 7 orders of magnitude

**Conclusion:** LIGO/Virgo/KAGRA **cannot detect** Type Ia supernova gravitational waves from thermonuclear explosions. All O4 supernova searches target core-collapse events in the 50-2000 Hz band.

---

## 2. DECIGO/BBO: The Enabling Missions

### 2.1 Mission Status (2025)

#### DECIGO (DECi-hertz Interferometer Gravitational wave Observatory)
- **Lead:** Japan (JAXA/ISAS)
- **Frequency band:** 0.1 – 10 Hz (decihertz "sweet spot")
- **Arm length:** 1,000 km
- **Configuration:** 4 clusters × 3 spacecraft (Fabry-Pérot Michelson interferometers)
- **Orbit:** Heliocentric
- **Original launch target:** 2027
- **Current timeline:** Delayed to post-2030s

#### B-DECIGO (Precursor Mission)
- **Purpose:** Technology demonstrator for DECIGO
- **Arm length:** 100 km (1/10 scale)
- **Laser power:** 1 W (vs. 10 W for DECIGO)
- **Mirror mass:** 30 kg
- **Orbit:** Earth orbit at 2,000 km altitude
- **Launch window:** **2030s** (currently in planning)
- **Goal:** Validate technologies + produce early science results

#### BBO (Big Bang Observer)
- **Lead:** NASA concept
- **Frequency band:** 0.01 – 1 Hz
- **Status:** Less developed than DECIGO; primarily a concept study
- **Scientific synergy:** Overlaps with DECIGO; may share technology

**Current Reality (2025):** DECIGO remains in design/optimization phase. B-DECIGO is the near-term priority for 2030s deployment.

### 2.2 Sensitivity Curves and Detection Thresholds

#### DECIGO Target Sensitivity:
- **At 1 Hz:** h_noise ~ 10⁻²³ Hz⁻¹/² (characteristic strain)
- **At 0.1 Hz:** h_noise ~ 10⁻²² Hz⁻¹/²
- **At 10 Hz:** h_noise ~ 3 × 10⁻²⁴ Hz⁻¹/²
- **Optimal range:** 0.3 – 3 Hz

**SNR Threshold for Detection:** SNR > 8 (5σ confidence)

### 2.3 Type Ia Supernova Detection Rates

#### Assumptions:
- **Progenitor model:** Double-degenerate (WD-WD merger: 1.0 M☉ + 0.8 M☉)
- **GW frequency at merger:** f_GW ~ 1 Hz (decihertz band)
- **Milky Way SN Ia rate:** 1 per 500 years = 0.002 yr⁻¹
- **Local volume SN Ia rate:** ~10⁻⁴ yr⁻¹ Mpc⁻³

#### DECIGO Detection Projections:
From the literature search, DECIGO is expected to:
- **Detect 6,600 WD-WD mergers** within z = 0.08 (350 Mpc) over mission lifetime
- **Identify host galaxies** for WD-WD mergers within z ~ 0.065 (280 Mpc) using GW alone
- **Volume surveyed:** ~10⁴ Mpc³ (comparable to distance to Virgo Cluster)

**Realistic Detection Rate:**
- To observe ≥1 SN Ia per year → need to survey ~10⁴ Mpc³
- **Volume to z=0.08:** V = (4/3)π(350 Mpc)³ ≈ 1.8 × 10⁸ Mpc³
- **Expected SN Ia per year in this volume:** (1.8 × 10⁸ Mpc³) × (10⁻⁴ yr⁻¹ Mpc⁻³) = **18,000 SN Ia/year**

**BUT:** DECIGO detects the **progenitor merger** (WD+WD inspiral), NOT the thermonuclear explosion itself. The GW emission occurs during the final inspiral minutes to hours before the optical transient.

### 2.4 Spandrel Framework Predictions for DECIGO

Using the Spandrel Framework strain predictions:

| SN Ia Type | D-parameter | Strain at 10 kpc | Detection Distance (DECIGO) | Expected Rate in Volume |
|------------|-------------|------------------|-----------------------------|------------------------|
| **Type Iax** | D → 2.0 | h ~ 10⁻²⁵ | ~1 kpc | **NOT DETECTABLE** (too weak) |
| **Normal SN Ia** | D = 2.2 | h ~ 10⁻²² | ~100 kpc | **Marginal** (only very local) |
| **03fg-like** | D → 3.0 | h ~ 10⁻²¹ | ~1 Mpc | **DETECTABLE** (local volume) |

**Detection Distance Calculation:**
- DECIGO sensitivity at 1 Hz: h_noise ~ 10⁻²³ Hz⁻¹/²
- Required SNR: 8
- Threshold strain: h_threshold ~ 8 × 10⁻²³ = 6.4 × 10⁻²³

For **Normal SN Ia (h ~ 10⁻²² at 10 kpc)**:
- Detection distance: d_max = (10 kpc) × (10⁻²² / 6.4 × 10⁻²³) ≈ **160 kpc**

For **03fg-like (h ~ 10⁻²¹ at 10 kpc)**:
- Detection distance: d_max = (10 kpc) × (10⁻²¹ / 6.4 × 10⁻²³) ≈ **1.6 Mpc**

**Critical Issue:** The Spandrel Framework predictions are for the **thermonuclear explosion phase** (deflagration-to-detonation transition, double-detonation), NOT the progenitor inspiral. Current DECIGO detection rate estimates are for **WD-WD inspirals** (pre-explosion), which emit in the millihertz band (LISA's domain).

**Resolution:** For DECIGO to detect **explosion-phase GWs** as predicted by Spandrel:
- Need h ~ 10⁻²¹ to 10⁻²² at 1 Hz
- **03fg-like mergers:** Best candidates (10× louder than normal SN Ia)
- **Galactic rate:** ~1 per 500-5000 years (normal SN Ia) × 0.05-0.1 (03fg fraction) = 1 per 5,000-50,000 years
- **Local Group (within 1 Mpc):** ~50 Milky Way-equivalent galaxies → 1 per 100-1000 years

**Expected DECIGO detections (explosion phase):**
- **In 4-year mission:** 0.004 – 0.04 events (1 per 25-250 years in Local Group)
- **Probability:** 0.4% – 4% chance of detecting one 03fg-like SN Ia explosion

**Multi-Messenger Strategy:**
DECIGO must operate in **triggered mode** using optical/X-ray surveys (e.g., Rubin Observatory, Zwicky Transient Facility) to search for GW signals around known SN Ia events within ~1 Mpc.

---

## 3. LISA Prospects for Double-Degenerate Progenitors

### 3.1 LISA Mission Overview
- **Frequency band:** 0.1 mHz – 1 Hz (millihertz regime)
- **Launch:** 2030s (ESA mission, currently in development)
- **Arm length:** 2.5 million km
- **Sensitivity at 1 mHz:** h ~ 10⁻²⁰ Hz⁻¹/²

### 3.2 Double White Dwarf Detection Capabilities

#### Complete Sample at Short Orbital Periods:
LISA will provide a **complete census** of double white dwarf (WD-WD) binaries with:
- **Orbital periods:** P_orb < 11-16 minutes
- **GW frequency:** f_GW > 2-3 mHz
- **Distance:** Throughout the Milky Way and Local Group

This enables **statistical validation** of the double-degenerate scenario by counting WD-WD systems.

#### Detection Numbers (Population Synthesis):
- **Expected LISA detections:** >10,000 WD-WD binaries over 4-year mission
- **Individually resolvable:** ~25,000 systems
- **Galactic foreground:** Unresolved background from millions of systems at f < 1 mHz

### 3.3 Type Ia Progenitor Validation

#### Key Question: Do enough WD-WD binaries exist to explain the SN Ia rate?

**Observed Milky Way SN Ia rate:** R_SNIa = (3-7) × 10⁻³ yr⁻¹
**LISA mission duration:** 4-10 years
**Probability of SN Ia during LISA mission:** 3-7% chance (if 10-year mission)

**LISA will constrain:**
- **WD-WD merger rate in Milky Way:** Precision of 4-9%
- **Mass distribution:** Identify super-Chandrasekhar progenitors (M > 1.4 M☉)
- **Merger timescale:** GW-driven inspiral timescales vs. SN Ia delay time distribution

**Outcome:**
- If LISA detects **enough massive WD-WD binaries** with short merger times → **Double-degenerate scenario supported**
- If LISA detects **too few** → **Single-degenerate or other channels dominate**

### 3.4 Recent LISA-Detectable Progenitor Discoveries

**ATLAS J1138-5139 (announced 2025):**
- **Orbital period:** 28 minutes
- **Status:** Binary WD system, LISA-detectable
- **Significance:** Demonstrates LISA's potential to reveal Galactic-scale Type Ia progenitor population

**WDJ181058.67+311940.94 (2025 discovery):**
- **Total mass:** 1.555 ± 0.044 M☉ (**super-Chandrasekhar**)
- **Merger time:** 22.6 ± 1.0 Gyr (comparable to Hubble time)
- **Distance:** 49 pc (extremely nearby)
- **Type:** Super-Chandrasekhar progenitor (potential 03fg-like SN Ia)

**Implications for Spandrel Framework:**
LISA will identify **D→3 progenitors** (super-Chandrasekhar systems) years to decades before they explode. This enables:
- **Pre-explosion monitoring** of electromagnetic signatures
- **GW chirp mass measurement** → predict D-parameter
- **Multi-messenger coordination** with DECIGO for explosion-phase GWs

### 3.5 LISA + Optical Multi-Messenger Timeline

**LISA Detection (years to decades before explosion):**
1. **GW chirp detected** at f_GW ~ 1-10 mHz → orbital period measured
2. **Chirp mass extracted:** M_chirp = (M1 M2)³/⁵ / (M1 + M2)¹/⁵
3. **Total mass inferred** (with mass ratio assumptions): M_total = M1 + M2
4. **Merger countdown:** Time to merger τ_merge ~ (f_GW)⁻⁸/³

**Optical Follow-up (triggered by GW):**
- **Sky localization:** LISA can localize to ~1-100 deg² (poor compared to LIGO)
- **Host galaxy search:** Identify dwarf galaxy or star cluster host
- **Pre-explosion baseline:** Monitor for accretion signatures, variability

**Explosion Phase (decihertz GWs):**
- **DECIGO triggered search:** Once SN Ia occurs, DECIGO searches for explosion GWs
- **EM light curve:** Optical rise (Spandrel D-parameter determines light curve shape)
- **GW strain measurement:** Confirm Spandrel prediction (h ~ 10⁻²² for D=2.2, h ~ 10⁻²¹ for D→3)

**Post-explosion (days to weeks):**
- **Spectroscopy:** Classify SN Ia subtype (normal, Iax, 03fg-like)
- **Nebular phase:** Measure ⁵⁶Ni mass, ejecta kinematics
- **Remnant search:** LISA continues monitoring for bound remnant (Type Iax "zombie star")

---

## 4. Einstein Telescope / Cosmic Explorer (Third-Generation Ground Detectors)

### 4.1 Mission Overview

#### Einstein Telescope (ET)
- **Location:** Europe (site selection in progress: Sardinia or Euregio Meuse-Rhine)
- **Configuration:** Triangular underground detector
- **Arm length:** 10 km (per arm)
- **Frequency range:** ~3 Hz – 10 kHz
- **Sensitivity improvement:** 10× better than Advanced LIGO at optimal frequencies
- **Status:** In planning; operational by **mid-2030s to 2040s**

#### Cosmic Explorer (CE)
- **Location:** United States (two observatories planned)
- **Configuration:** L-shaped interferometers
- **Arm lengths:** 40 km (CE1) and 20 km (CE2)
- **Frequency range:** ~5 Hz – 10 kHz
- **Sensitivity improvement:** >10× better than Advanced LIGO
- **Status:** In design phase; operational by **2040s**

### 4.2 Low-Frequency Reach: Can They Access Decihertz?

**Critical Question:** Do ET/CE extend into the decihertz band (0.1-10 Hz) where SN Ia GWs reside?

**Answer:** **Partial overlap, but limited.**

From literature review:
- **ET/CE optimal range:** 5 Hz – 5 kHz (core-collapse supernovae, neutron stars, stellar-mass BH mergers)
- **Decihertz overlap:** "Third-generation and deci-Hz detectors have comparable sensitivities around 5 Hz"
- **Below 5 Hz:** Ground-based detectors face seismic noise wall; space-based (DECIGO/LISA) outperform

**Sensitivity at 1 Hz:**
- **DECIGO:** h ~ 10⁻²³ Hz⁻¹/²
- **ET/CE:** h ~ 10⁻²⁰ to 10⁻²¹ Hz⁻¹/² (3-4 orders of magnitude worse due to seismic noise)

**Conclusion:**
ET/CE **cannot effectively detect** Type Ia supernova explosion GWs in the 0.1-3 Hz band. They will excel at core-collapse supernovae (50-2000 Hz) but miss thermonuclear SN Ia signals.

### 4.3 Core-Collapse Supernova Prospects (Not Type Ia)

**Detection horizon for core-collapse SNe:**
- **2G detectors (LIGO/Virgo):** Limited to Milky Way (~10 kpc)
- **3G detectors (ET/CE):** Extended to Local Group (~1 Mpc) and beyond

**Challenges:**
- GW emission mechanisms are diverse and uncertain
- Signals are weak and transient
- Matched filtering is difficult without robust waveform templates

**Type Ia vs. Core-Collapse in ET/CE:**
- **Core-collapse:** Detectable at 50-2000 Hz if within ~1 Mpc (rare events: ~1 per decade in Local Group)
- **Type Ia:** GWs at 0.1-3 Hz → **not accessible** to ET/CE

---

## 5. Atom Interferometer Missions (MAGIS/AEDGE)

### 5.1 Overview: The Mid-Band Frontier

**Frequency Gap:**
- **LISA:** 0.1 mHz – 1 Hz (millihertz)
- **LIGO/Virgo:** 10 Hz – 10 kHz
- **Gap:** 1 Hz – 10 Hz (the **mid-band** or **decihertz** region)

**Atom Interferometers:** Quantum sensors using ultra-cold atoms to measure spacetime curvature. They offer peak sensitivity in the **0.01 Hz – 3 Hz** range, perfectly matching the decihertz gap.

### 5.2 Current and Planned Atom Interferometer Projects

#### MAGIS-100 (Matter-wave Atomic Gradiometer Interferometric Sensor)
- **Location:** Fermilab, USA
- **Baseline:** 100 meters
- **Atom species:** Strontium-87 (⁸⁷Sr)
- **Status:** **Under construction** (2025)
- **Frequency band:** 0.01 – 3 Hz (mid-band)
- **Goals:**
  - Search for ultralight dark matter
  - Detect mid-band gravitational waves
  - Test quantum mechanics at macroscopic scales

#### MAGIS-km (Future Upgrade)
- **Location:** Sanford Underground Research Facility (SURF), South Dakota
- **Baseline:** 1 km (10× longer than MAGIS-100)
- **Status:** Planned for **2030s**
- **Expected sensitivity:** Competitive with DECIGO in the 0.1-3 Hz band

#### AION-10 (UK)
- **Location:** Oxford, UK
- **Baseline:** 10 meters
- **Status:** Funded, under construction

#### AION-km (Future)
- **Location:** STFC Boulby Underground Laboratory, UK
- **Baseline:** 1 km
- **Timeline:** **2030s** (technology demonstrator for AEDGE)

#### AEDGE (Atomic Experiment for Dark Matter and Gravity Exploration)
- **Platform:** **Space-based mission** (ESA concept)
- **Baseline:** 40,000 km (satellite-to-satellite)
- **Frequency band:** 0.01 – 1 Hz
- **Status:** Concept phase; technology readiness target by **2035**
- **Science goals:**
  - Ultra-light dark matter
  - Mid-band gravitational waves (between LISA and ground detectors)
  - Fundamental physics (equivalence principle tests)

### 5.3 Decihertz Sensitivity for Type Ia Supernovae

**MAGIS-km / AION-km Target Sensitivity (by 2035):**
- **At 1 Hz:** h ~ 10⁻²² to 10⁻²³ Hz⁻¹/²
- **At 0.1 Hz:** h ~ 10⁻²¹ Hz⁻¹/²

**Comparison to DECIGO:**
- **DECIGO (full mission):** h ~ 10⁻²³ at 1 Hz
- **MAGIS-km/AION-km:** Comparable sensitivity, may achieve 10⁻²² to 10⁻²³ by 2035

**Type Ia Detection Prospects:**

Using Spandrel Framework predictions:

| SN Ia Type | Strain at 10 kpc | MAGIS-km Detection Range | AEDGE Detection Range |
|------------|------------------|--------------------------|----------------------|
| **Type Iax** | h ~ 10⁻²⁵ | Not detectable | Not detectable |
| **Normal SN Ia** | h ~ 10⁻²² | ~10-100 kpc | ~100 kpc – 1 Mpc |
| **03fg-like** | h ~ 10⁻²¹ | ~100 kpc – 1 Mpc | ~1-10 Mpc |

**Key Advantage:** Ground-based km-scale atom interferometers (MAGIS-km, AION-km) can be built **faster and cheaper** than space missions like DECIGO.

**Timeline:**
- **2025-2027:** MAGIS-100, AION-10 (technology demonstrators)
- **2030-2035:** MAGIS-km, AION-km operational → **first mid-band SN Ia sensitivity**
- **Post-2035:** AEDGE space mission (if funded)

**Strategic Importance:**
Atom interferometers may provide the **first decihertz detections** before DECIGO flies, enabling early validation of Spandrel Framework predictions.

---

## 6. Multi-Messenger Observation Protocol Design

### 6.1 Protocol Objectives

**Goal:** Coordinate electromagnetic, neutrino, and gravitational wave observations of Type Ia supernovae to:
1. **Validate Spandrel Framework** GW predictions (D-parameter → strain amplitude)
2. **Identify progenitor channels** (double-degenerate vs. single-degenerate)
3. **Measure explosion physics** (deflagration-to-detonation transition, asymmetries)
4. **Constrain cosmology** (H₀ from GW+EM standard sirens)

### 6.2 Three-Phase Observation Strategy

#### **PHASE 1: Pre-Explosion (LISA Era, 2030s-2040s)**

**Lead Detector:** LISA (millihertz GWs from WD-WD inspirals)

**Trigger:** LISA detects WD-WD binary with:
- f_GW > 2 mHz (P_orb < 16 minutes)
- M_total > 1.4 M☉ (Chandrasekhar-mass or super-Chandrasekhar)
- τ_merge < 100 years (merging within human timescale)

**Actions:**
1. **GW characterization:**
   - Measure chirp mass M_chirp → infer total mass M_total
   - Extract sky position (Δθ ~ 1-100 deg²)
   - Estimate merger countdown τ_merge
   - Monitor GW frequency evolution: df/dt ∝ f¹¹/³

2. **Optical follow-up (pre-explosion baseline):**
   - **Wide-field surveys:** Rubin Observatory (LSST), Zwicky Transient Facility (ZTF)
   - **Target:** Identify host galaxy within LISA error box
   - **Cadence:** Monthly to yearly (depending on τ_merge)
   - **Look for:** Accretion signatures, optical variability, X-ray emission

3. **Spandrel D-parameter prediction:**
   - **Mass ratio:** q = M2/M1 (from GW)
   - **Total mass:** M_total (from GW)
   - **D-parameter estimate:**
     - If M_total ~ 1.4 M☉ → D ~ 2.2 (normal SN Ia)
     - If M_total > 1.8 M☉ → D → 3 (03fg-like, super-Chandrasekhar)
     - If asymmetric merger → D ~ 2.0-2.2 (Type Iax possible)

4. **GW strain prediction:**
   - Use D-parameter to predict explosion-phase GW strain
   - Coordinate with DECIGO/MAGIS for triggered search

**Expected Pre-Explosion Warning Time:**
- **LISA detects at f_GW ~ 3 mHz** → τ_merge ~ 10 years
- **Final year:** f_GW increases to 10-30 mHz → τ_merge ~ 1 month to 1 day
- **Hours before merger:** f_GW → 100 mHz → τ_merge ~ minutes (LISA loses signal as it enters DECIGO band)

---

#### **PHASE 2: Explosion (DECIGO/MAGIS Era, 2030s-2040s)**

**Lead Detectors:**
- **DECIGO/B-DECIGO** (decihertz GWs from explosion)
- **MAGIS-km / AION-km** (ground-based atom interferometers, mid-band)
- **Optical surveys:** Rubin, ZTF, ATLAS (first light detection)

**Trigger:** Optical detection of SN Ia within:
- **DECIGO detection range:** <1 Mpc for 03fg-like (D→3), <100 kpc for normal SN Ia (D=2.2)
- **MAGIS-km detection range:** Similar to DECIGO

**Observation Window:**
- **GW emission:** Seconds to minutes during deflagration-to-detonation transition (DDT) or double-detonation
- **Optical first light:** Shock breakout (if present), then rising light curve over days

**Actions:**

1. **Decihertz GW search (DECIGO/MAGIS):**
   - **Frequency band:** 0.1 – 10 Hz
   - **Search window:** ±24 hours around optical discovery (account for explosion-to-first-light delay)
   - **Waveform template:**
     - **Double-detonation:** Sharp burst as helium shell detonates, then core detonates
     - **DDT:** Longer signal (seconds) as deflagration transitions to detonation
     - **Asymmetric explosion:** Polarized GW burst (strongest for D→3)
   - **Expected SNR:**
     - Normal SN Ia (D=2.2) at 50 kpc: SNR ~ 10-30
     - 03fg-like (D→3) at 1 Mpc: SNR ~ 8-20
     - Type Iax (D→2) at 10 kpc: SNR < 1 (non-detection)

2. **Optical/NIR photometry (multi-band light curves):**
   - **Cadence:** Hourly to daily over first 2 weeks
   - **Bands:** u, g, r, i, z, Y, J, H, K
   - **Goals:**
     - Measure rise time (Spandrel: shorter rise for higher D)
     - Determine peak luminosity (D→3 should be super-luminous)
     - Constrain ⁵⁶Ni mass from light curve tail

3. **Spectroscopy (classify SN Ia subtype):**
   - **Epoch:** Peak light (t ~ +15 days)
   - **Resolution:** Medium to high (R ~ 1000-10,000)
   - **Features:**
     - **Normal SN Ia (D=2.2):** Si II λ6355, S II, no H/He
     - **Type Iax (D→2):** Weak Si II, low velocities (~4000 km/s), narrow lines
     - **03fg-like (D→3):** Strong C II at early times, lower ejecta velocities (~8000 km/s), super-luminous

4. **Neutrino search (IceCube, Super-Kamiokande, Hyper-Kamiokande):**
   - **Expected signal:** NONE (Type Ia are thermonuclear, not core-collapse)
   - **Purpose:** Null detection confirms thermonuclear origin
   - **Caveat:** Some DD merger models predict weak neutrino flux from hot envelope; non-detection constrains these models

5. **Multi-messenger timing:**
   - **Neutrinos (if any):** First (seconds before explosion)
   - **GWs:** During explosion (seconds to minutes)
   - **Optical:** Shock breakout (if present, minutes to hours), then rising light curve (days)

6. **GW + EM parameter extraction:**
   - **Distance:** Independent from GW strain amplitude (if detected) and EM luminosity
   - **Test:** GW distance vs. EM distance (Hubble diagram) → constrain H₀
   - **Asymmetry:** GW polarization → map 3D explosion geometry
   - **D-parameter validation:** Compare observed strain h_obs with Spandrel prediction h_pred(D)

---

#### **PHASE 3: Post-Explosion (Days to Years)**

**Lead Observatories:**
- **JWST, Hubble, VLT, Keck** (nebular spectroscopy, late-time imaging)
- **Chandra, XMM-Newton** (X-ray remnant search)
- **ALMA, VLA** (radio emission from ejecta-CSM interaction)

**Actions:**

1. **Nebular spectroscopy (t > 100 days):**
   - **Features:** [Fe II], [Fe III], [Co II], [Co III] (decay of ⁵⁶Ni → ⁵⁶Co → ⁵⁶Fe)
   - **Goals:**
     - Measure ⁵⁶Ni mass (validates D-parameter prediction)
     - Map ejecta kinematics (asymmetric explosion → GW polarization)
     - Detect stable Fe-peak isotopes (⁵⁴Fe, ⁵⁸Ni) → neutronization signature

2. **Late-time imaging (t > 1 year):**
   - **Search for:** Surviving companion star (single-degenerate channel)
   - **Null detection:** Supports double-degenerate scenario (no companion)
   - **Type Iax:** Search for bound remnant "zombie star" (may re-brighten)

3. **Supernova remnant (SNR) evolution (years to decades):**
   - **X-ray:** Thermal emission from shocked ejecta
   - **Radio:** Synchrotron emission from shock acceleration
   - **Optical:** Ejecta-CSM interaction (if present)

4. **LISA re-observation (if Type Iax with bound remnant):**
   - **Look for:** Continued GW emission from surviving WD remnant
   - **Frequency:** If remnant is rotating → GW from non-axisymmetric distortion
   - **Constraints:** Remnant mass, spin, magnetic field

---

### 6.3 Coordination Infrastructure

**Multi-Messenger Alerts:**
- **GW triggers:** LISA → LVK-style "GCN Notice" (Gravitational-wave Candidate Notice)
- **EM triggers:** Optical surveys → Transient Name Server (TNS), AstroNotes
- **Cross-match:** Automated pipelines match GW candidates with optical transients

**Data Sharing:**
- **GW data:** Public release via GWOSC (Gravitational Wave Open Science Center)
- **EM data:** Public follow-up photometry/spectroscopy via Open Supernova Catalog, WISeREP
- **Neutrino data:** SNEWS 2.0 (Supernova Early Warning System)

**Rapid Response Teams:**
- **EM follow-up:** Las Cumbres Observatory (robotic network), GRANDMA, GROWTH
- **GW follow-up:** DECIGO/MAGIS triggered searches within 24 hours of optical discovery

---

### 6.4 Decision Tree: SN Ia Classification and GW Expectation

```
Optical discovery of SN Ia candidate
    |
    ├─ Distance > 1 Mpc?
    |      └─ YES → GW detection unlikely (except for 03fg-like with AEDGE)
    |      └─ NO → Proceed with multi-messenger protocol
    |
    ├─ Spectroscopic classification:
    |      ├─ Type Iax (weak, slow)
    |      |      └─ D → 2.0 → h ~ 10⁻²⁵ → NO GW DETECTION EXPECTED
    |      |      └─ Search for bound remnant (LISA follow-up years later)
    |      |
    |      ├─ Normal SN Ia (Si II, moderate velocity)
    |      |      └─ D ~ 2.2 → h ~ 10⁻²² at 10 kpc
    |      |      └─ GW DETECTABLE if distance < 100 kpc (DECIGO/MAGIS)
    |      |
    |      └─ 03fg-like (super-luminous, C II, low velocity)
    |             └─ D → 3.0 → h ~ 10⁻²¹ at 10 kpc (10× amplified)
    |             └─ GW DETECTABLE if distance < 1 Mpc (PRIORITY TARGET)
    |
    └─ GW search result:
           ├─ DETECTION → Extract h_obs, f_GW, polarization
           |      └─ Compare h_obs vs. h_pred(D) → validate Spandrel Framework
           |      └─ Measure distance from GW → H₀ constraint
           |
           └─ NON-DETECTION → Place upper limit on h
                  └─ If h_limit < h_pred(D) → Spandrel prediction falsified
                  └─ If h_limit > h_pred(D) → Inconclusive (need closer event)
```

---

### 6.5 Key Science Outcomes

**From Multi-Messenger SN Ia Observations:**

1. **Spandrel Framework Validation:**
   - Measure GW strain h_obs for classified SN Ia subtypes
   - Test prediction: h(D→3) ~ 10 × h(D=2.2) ~ 100 × h(D→2)

2. **Progenitor Channel Identification:**
   - **GW + EM joint detection** → Double-degenerate confirmed
   - **EM only (no GW from inspiral)** → Single-degenerate or DD with very rapid merger
   - **LISA pre-cursor + DECIGO explosion** → DD confirmed, explosion physics constrained

3. **Explosion Mechanism:**
   - **GW waveform** → Deflagration-to-detonation transition vs. double-detonation
   - **GW polarization** → 3D asymmetry (critical for "Spandrel" geometric argument)
   - **Timing (GW vs. optical)** → Shock breakout delay, nickel distribution

4. **Cosmology (GW + EM Standard Sirens):**
   - **GW luminosity distance:** d_GW from strain amplitude (no redshift needed)
   - **EM redshift:** z from host galaxy spectrum
   - **Hubble constant:** H₀ = cz / d_GW (independent of cosmic distance ladder)

5. **Population Statistics (after ~10 detections):**
   - **Fraction of DD vs. SD** progenitors
   - **D-parameter distribution** (normal vs. Iax vs. 03fg-like)
   - **GW emission efficiency** (E_GW / E_total)

---

## 7. Timeline for First Type Ia Supernova GW Detection

### 7.1 Detector Readiness

| Detector | Frequency Band | SN Ia Sensitivity | Operational | First Detection Window |
|----------|----------------|-------------------|-------------|----------------------|
| **LIGO/Virgo/KAGRA** | 10 Hz – 10 kHz | None (seismic noise) | Now | **NEVER** (wrong frequency) |
| **LISA** | 0.1 mHz – 1 Hz | Progenitor inspirals | **2030s** | 2035-2045 (pre-explosion) |
| **MAGIS-100** | 0.01 – 3 Hz | Technology demo | **2025-2027** | Not sensitive enough |
| **B-DECIGO** | 0.1 – 10 Hz | Reduced (100 km arms) | **2030s** | 2035-2040 (marginal) |
| **MAGIS-km / AION-km** | 0.1 – 3 Hz | **Competitive** | **2030-2035** | **2035-2040** |
| **DECIGO (full)** | 0.1 – 10 Hz | **Optimal** | **2040s** | **2040-2050** |
| **AEDGE (space)** | 0.01 – 1 Hz | **Enhanced** | **2040s** | 2045-2055 |
| **Einstein Telescope** | 3 Hz – 10 kHz | Limited (seismic) | **2035-2040** | Marginal at best |
| **Cosmic Explorer** | 5 Hz – 10 kHz | Limited (seismic) | **2040s** | Marginal at best |

### 7.2 Event Rate Considerations

**Local SN Ia Rate (within detection range):**

| Distance | Volume (Mpc³) | SN Ia Rate | Expected Wait Time | Best Targets |
|----------|--------------|------------|-------------------|--------------|
| **10 kpc** | 0.004 | 4 × 10⁻⁷ yr⁻¹ | 2.5 million years | Milky Way center |
| **100 kpc** | 4 | 4 × 10⁻⁴ yr⁻¹ | 2,500 years | LMC, SMC, M31 |
| **1 Mpc** | 4,000 | 0.4 yr⁻¹ | 2.5 years | Local Group |
| **10 Mpc** | 4 × 10⁶ | 400 yr⁻¹ | 0.0025 years (1 day) | Local Volume |

**Spandrel-Specific Rates:**
- **Normal SN Ia (D=2.2):** ~70% of all SN Ia → 0.28 yr⁻¹ within 1 Mpc
- **03fg-like (D→3):** ~5-10% of all SN Ia → 0.02-0.04 yr⁻¹ within 1 Mpc (1 per 25-50 years)
- **Type Iax (D→2):** ~30% of all SN Ia, but h ~ 10⁻²⁵ → **not detectable**

### 7.3 First Detection Scenarios

#### **Scenario 1: MAGIS-km / AION-km (2035-2040)**

**Earliest Possible Detection:**
- **Detector:** MAGIS-km or AION-km (ground-based atom interferometer)
- **Target:** 03fg-like SN Ia at 0.5-1 Mpc (Local Group)
- **Rate:** 1 per 25-50 years in this volume
- **Probability:** 8-16% over 4-year mission
- **GW strain:** h ~ 10⁻²¹ × (10 kpc / 1 Mpc) ~ 10⁻²³ (marginal, SNR ~ 5-10)

**Challenges:**
- Requires **lucky timing** (03fg-like events are rare)
- Ground-based detectors have limited duty cycle (~50-70%)
- Sky localization may be poor without network

**Strategy:**
- **Triggered searches** using optical surveys (Rubin Observatory will find ~10⁴ SN Ia per year)
- Focus on nearby galaxies (M31, M33, NGC 300, etc.)

#### **Scenario 2: B-DECIGO (2035-2040)**

**Precursor Mission:**
- **Detector:** B-DECIGO (100 km arms, Earth orbit)
- **Sensitivity:** ~10× worse than full DECIGO → h_noise ~ 10⁻²² at 1 Hz
- **Detection range:** 50-100 kpc for normal SN Ia, 500 kpc for 03fg-like
- **Rate:** ~1 per 2,500 years within 100 kpc (low probability)

**Most Likely Outcome:**
- **No detections** during 2-4 year mission (event rate too low)
- **Upper limits** on GW strain from nearby SN Ia (within ~1 Mpc)
- **Technology validation** for full DECIGO

#### **Scenario 3: DECIGO + LISA (2040-2050) – BEST CASE**

**Full Multi-Messenger Era:**
- **LISA:** Detects DD progenitor WD-WD inspiral (years before explosion)
- **Optical:** Pre-explosion monitoring identifies host galaxy
- **DECIGO:** Triggered search at time of explosion (±24 hours)
- **EM follow-up:** Coordinated photometry/spectroscopy

**Expected Timeline:**
1. **2040:** DECIGO begins operations
2. **2040-2045:** LISA identifies ~10-100 super-Chandrasekhar WD-WD systems with τ_merge < 10 years
3. **2045:** First "predicted" SN Ia occurs (LISA → DECIGO → EM coordination)
4. **2045-2050:** 1-5 DECIGO detections (03fg-like prioritized)

**First Detection Probability:**
- **Within 5 years:** ~90% (if 03fg-like rate is 1 per 50 years in Local Group)
- **Within 1 year:** ~20% (depends on LISA pre-cursor identification)

**Potential outcomes:**
- First direct measurement of SN Ia explosion GWs
- Spandrel Framework test (if h_obs ~ h_pred for classified D-parameter)
- GW+EM standard sirens for improved H₀ constraints

---

### 7.4 Most Likely First Detection: **2045 ± 5 years**

**Detector:** DECIGO (full mission)
**Target:** 03fg-like super-Chandrasekhar SN Ia at 0.5-1 Mpc
**Multi-Messenger:** LISA pre-cursor + DECIGO explosion + Rubin/JWST EM

**Enabling Factors:**
1. **DECIGO sensitivity** (h ~ 10⁻²³ at 1 Hz) reaches Spandrel-predicted strains
2. **LISA pre-identification** of DD progenitors allows **triggered observations**
3. **Optical surveys** (Rubin LSST) find ~10⁴ SN Ia/year → high probability of nearby event during DECIGO mission
4. **03fg-like SN Ia** have 10× stronger GW emission (D→3 regime)

**Wildcard Scenario:** MAGIS-km detects a **very nearby** 03fg-like SN Ia in **2037-2040**, scooping DECIGO. Probability ~5-10%.

---

## 8. Summary and Strategic Recommendations

### 8.1 Key Findings

1. **LIGO/Virgo O4 Limitations:**
   - No Type Ia SN GW searches reported (thermonuclear SNe emit at 0.1-10 Hz, below LIGO's band)
   - SN 2023ixf (core-collapse) upper limits: E_GW < 10⁻⁵ M☉c² (not applicable to Type Ia)
   - Ground-based detectors **cannot** access decihertz frequencies needed for SN Ia

2. **DECIGO/BBO: The Essential Missions:**
   - Only space-based decihertz detectors can test Spandrel Framework GW predictions
   - B-DECIGO (2030s): Technology demonstrator, likely no detections
   - Full DECIGO (2040s): Expected 1-5 detections over 4-year mission

3. **LISA: The Progenitor Identifier:**
   - Will detect >10,000 WD-WD binaries (pre-explosion)
   - Validates double-degenerate scenario statistically
   - Identifies super-Chandrasekhar systems (D→3 progenitors) years before explosion

4. **Atom Interferometers: The Dark Horse:**
   - MAGIS-km / AION-km (2030-2035) may achieve first decihertz SN Ia detections
   - Ground-based, cheaper, faster deployment than DECIGO
   - Competitive sensitivity (h ~ 10⁻²² to 10⁻²³ at 1 Hz)

5. **Einstein Telescope / Cosmic Explorer:**
   - Optimized for core-collapse SNe and compact object mergers (5-5000 Hz)
   - Minimal overlap with decihertz SN Ia signals
   - Not primary instruments for Spandrel Framework tests

6. **Multi-Messenger Protocol:**
   - Three-phase observation (LISA pre-explosion → DECIGO explosion → EM/neutrino post-explosion)
   - Triggered searches essential (event rate too low for blind all-sky searches)
   - GW + EM joint detection enables H₀ measurements, progenitor identification

### 8.2 Spandrel Framework Validation Pathway

**Testable Predictions:**
| D-Parameter | SN Ia Type | GW Strain at 10 kpc | DECIGO SNR at 1 Mpc | Detection Probability (2040-2050) |
|-------------|------------|---------------------|---------------------|----------------------------------|
| **D → 2.0** | Type Iax | h ~ 10⁻²⁵ | <1 | **0%** (too weak) |
| **D = 2.2** | Normal SN Ia | h ~ 10⁻²² | ~3-8 | **10-30%** (marginal, need <100 kpc) |
| **D → 3.0** | 03fg-like | h ~ 10⁻²¹ | ~30-80 | **>90%** (1-5 events expected) |

**Critical Test:**
- **If DECIGO detects 03fg-like SN Ia with h ~ 10⁻²¹ at known distance** → Spandrel Framework CONFIRMED
- **If h_obs << 10⁻²¹ for 03fg-like** → Spandrel Framework FALSIFIED (or GW emission mechanism different than assumed)

### 8.3 Recommendations for Spandrel Framework Development

1. **Prioritize 03fg-like SN Ia as GW targets:**
   - Super-Chandrasekhar progenitors (D→3) have strongest GW emission
   - LISA will identify these systems before explosion
   - Coordinate with DECIGO for triggered searches

2. **Develop GW waveform templates for DDT and double-detonation:**
   - Current predictions are order-of-magnitude estimates
   - Need numerical relativity / hydrodynamic simulations for:
     - Deflagration-to-detonation transition asymmetry
     - Double-detonation helium shell ignition
     - White dwarf merger dynamics
   - Map D-parameter to GW frequency spectrum f(t)

3. **Engage with DECIGO/B-DECIGO collaboration:**
   - Propose SN Ia as primary science case (currently focused on primordial GWs, stochastic background)
   - Advocate for triggered-search capabilities (rapid response to optical transients)
   - Contribute to sensitivity optimization for 1 Hz band

4. **Support atom interferometer development (MAGIS/AEDGE):**
   - These may achieve first detections before DECIGO
   - Push for SN Ia science in mission design
   - Explore synergy with optical surveys (Rubin LSST)

5. **Prepare multi-messenger observation protocols NOW:**
   - Establish MoUs with LISA, DECIGO, Rubin, JWST teams
   - Design automated GW-EM matching pipelines
   - Create SN Ia GW candidate databases

6. **Theoretical work on Type Iax "zombie stars":**
   - If bound remnants exist (D→2), they may emit continuous GWs
   - LISA could detect rotating WD remnants years after explosion
   - Predict long-term GW evolution of failed SN Ia

### 8.4 Near-Term Opportunities (2025-2030)

**Before DECIGO flies:**
1. **Improve GW strain predictions:**
   - Run 3D hydrodynamic simulations of SN Ia explosions with varying D-parameter
   - Extract GW waveforms from simulations
   - Publish predicted strain amplitudes h(D, f_GW) for comparison with DECIGO

2. **Identify nearby super-Chandrasekhar progenitors:**
   - Survey double WD systems in Local Group
   - Look for LISA pre-cursors in EM (accretion signatures, X-rays)
   - Build target list for DECIGO triggered searches

3. **Develop D-parameter observational diagnostics:**
   - Link D to observable EM features (rise time, peak luminosity, nebular spectra)
   - Create classification algorithm: photometry + spectra → D-parameter estimate
   - Validate with existing SN Ia samples (normal, Iax, 03fg-like)

4. **Engage with LIGO/Virgo for O5/O6:**
   - Even though ground detectors can't detect SN Ia, establish collaboration
   - Share multi-messenger expertise
   - Prepare for future 3G detectors (ET/CE)

### 8.5 The Path to Discovery

**2025-2030:** Theory development, MAGIS-100 technology demonstration, LISA construction
**2030-2035:** LISA launch and operations, MAGIS-km/AION-km construction, B-DECIGO planning
**2035-2040:** LISA identifies DD progenitors, MAGIS-km operational (possible first detection), B-DECIGO launch
**2040-2045:** DECIGO begins operations, LISA+DECIGO joint observations, **first confirmed SN Ia GW detection**
**2045-2050:** Multiple detections, Spandrel Framework validation, GW+EM cosmology with SN Ia

**The gravitational wave frontier of the Spandrel Framework will open in the 2040s, with DECIGO as the key enabling mission. Early hints may come from MAGIS-km in the late 2030s. The multi-messenger era of Type Ia supernovae is within reach.**

---

## 9. References and Sources

**LIGO O4 and SN 2023ixf:**
- [LIGO O4 Observing Plans](https://dcc.ligo.org/public/0197/T2400403/002/wp-obs-2025.pdf)
- [GWTC-4.0 Catalog Release](https://www.ligo.caltech.edu/news/ligo20250826)
- [Search for gravitational waves from SN 2023ixf](https://arxiv.org/abs/2410.16565) - ApJ 985:183 (2025)
- [Targeted searches for GWs from SN 2023ixf and SGR 1935+2154](https://arxiv.org/abs/2506.02252)

**DECIGO/BBO:**
- [DECIGO Wikipedia](https://en.wikipedia.org/wiki/Deci-hertz_Interferometer_Gravitational_wave_Observatory)
- [Current status of DECIGO and B-DECIGO](https://arxiv.org/abs/2006.13545) - PTEP 2021(5):05A105
- [Probe for Type Ia supernova progenitor in decihertz GW astronomy](https://arxiv.org/abs/1910.01063)
- [The astrophysical science case for a decihertz GW detector](https://arxiv.org/abs/1710.11187)

**LISA and Double-Degenerate Progenitors:**
- [Expected insights into Type Ia supernovae from LISA's GW observations](https://www.aanda.org/articles/aa/full_html/2024/11/aa51380-24/aa51380-24.html) - A&A 2024
- [Where are the double-degenerate progenitors of Type Ia supernovae?](https://arxiv.org/abs/1809.07158) - MNRAS 482:3656 (2019)
- [A gravitational-wave-detectable candidate Type Ia supernova progenitor](https://arxiv.org/abs/2411.19916) - ApJ 987:206 (2025)
- [A super-Chandrasekhar mass Type Ia SN progenitor at 49 pc](https://arxiv.org/abs/2504.04522) - Nature Astronomy 2025

**Einstein Telescope / Cosmic Explorer:**
- [Einstein Telescope Official Site](https://www.et-gw.eu/)
- [Cosmic Explorer Official Site](https://cosmicexplorer.org/)
- [Searching for continuous GWs with DECIGO, BBO, ET, and CE](https://arxiv.org/abs/2503.17087) - MNRAS 540:1006 (2025)

**Atom Interferometers (MAGIS/AEDGE):**
- [MAGIS-100 Official Site](https://magis.fnal.gov/)
- [Prospective sensitivities of atom interferometers to GWs](https://royalsocietypublishing.org/doi/10.1098/rsta.2021.0060) - Phil. Trans. R. Soc. A 2021
- [AEDGE: Atomic Experiment for Dark Matter and Gravity Exploration](https://epjquantumtechnology.springeropen.com/articles/10.1140/epjqt/s40507-020-0080-0) - EPJ Quantum Technology 2020

**Type Ia Explosion Mechanisms:**
- [Type Ia supernova Wikipedia](https://en.wikipedia.org/wiki/Type_Ia_supernova)
- [A unified mechanism for DDT in terrestrial systems and Type Ia SNe](https://www.science.org/doi/10.1126/science.aau7365) - Science 2019
- [Type Iax supernovae](https://en.wikipedia.org/wiki/SN_2002cx)
- [SN 2003fg - Super-Chandrasekhar Type Ia](https://en.wikipedia.org/wiki/SN_2003fg)

**White Dwarf Mergers and Decihertz GWs:**
- [Decihertz GWs from double white dwarf merger remnants](https://arxiv.org/abs/2009.13017) - ApJ 906:29 (2021)
- [Binary white dwarfs and decihertz GW observations](https://www.aanda.org/articles/aa/full_html/2020/03/aa36848-19/aa36848-19.html) - A&A 2020

**Multi-Messenger Astronomy:**
- [Multi-messenger astronomy Wikipedia](https://en.wikipedia.org/wiki/Multi-messenger_astronomy)
- [GW170817 - The first multi-messenger GW event](https://en.wikipedia.org/wiki/GW170817)
- [Multimessenger Astronomy with Gravitational Waves - Virgo](https://www.virgo-gw.eu/science/gw-universe/multimessenger-astronomy-with-gw/)

**Type Ia Supernova Rates:**
- [How often do Type Ia supernovae occur?](https://astronomy.stackexchange.com/questions/34775/how-often-do-type-ia-supernovae-occur)
- [Milky Way Supernova Rate Confirmed](https://skyandtelescope.org/astronomy-news/milky-way-supernova-rate-confirmed/)

---

**Document Version:** 1.0
**Last Updated:** 2025-11-28
**Next Review:** After MAGIS-100 results (2027) and LISA launch (2030s)
