# Ultra-Massive White Dwarf Pulsation Database
## For Spandrel-Asteroseismic Prediction Framework

**Purpose**: Predict fractal dimension D of SN Ia progenitors BEFORE explosion using asteroseismic signatures

**Hypothesis**: D = f(P₁/P₂, Δπ, T_eff, M_WD)
- Convective turbulence in pre-explosion WDs sets initial D
- WD pulsation modes (g-modes) couple to convective regions
- Pulsation period ratios encode information about internal turbulence → D

---

## I. KEY SN Ia PROGENITOR CANDIDATE

### WDJ181058.67+311940.94
**The closest known super-Chandrasekhar SN Ia progenitor**

**Properties**:
- **Distance**: 49 pc
- **Total Mass**: 1.555 ± 0.044 M☉ (super-Chandrasekhar)
- **System Type**: Double white dwarf binary
- **Merger Time**: 22.6 ± 1.0 Gyr (same order as Hubble time)
- **Explosion Type**: Predicted double-detonation (both stars destroyed)
- **Peak Magnitude**: m_V ≈ -16 (200,000× brighter than Jupiter)
- **SN Type**: Subluminous Type Ia

**Status**:
- Published: Munday et al. 2025, Nature Astronomy
- arXiv: 2504.04522
- **PULSATION DATA**: Not yet reported in search results
- **Action needed**: Check if pulsations detected (key target)

**Observational Status**:
- Radial velocities obtained with: ISIS (WHT), IDS (INT), FIES/ALFOSC (NOT), UVES (VLT 8.2m)
- Time-series photometry needed for pulsation detection

**Scientific Significance**:
- When it explodes, will outshine ALL stars in sky
- Birth rate of super-Chandrasekhar DWDs: ≥6.0 × 10⁻⁴ yr⁻¹
- Observed SN Ia rate from such systems: ~4.4 × 10⁻⁵ yr⁻¹

---

## II. ULTRA-MASSIVE PULSATING WHITE DWARFS (M > 1.1 M☉)

### 1. J0959-1828 (CURRENT RECORD HOLDER)
**Most massive pulsating WD currently known**

**Physical Properties**:
- **Mass**: 1.32 M☉ (CO core) or 1.27 M☉ (ONe core)
- **Status**: Confirmed multi-periodic pulsator
- **T_eff**: Within ZZ Ceti strip (10,500-12,300 K)

**Pulsation Periods (converted from frequencies)**:
- **Mode 1**: 85.3 c/d → **P₁ = 1013 s** (amplitude: 2.3 mma)
- **Mode 2**: 91.7 c/d → **P₂ = 942 s** (amplitude: 1.8 mma)
- **Mode 3**: 401.2±0.8 c/d → **P₃ = 215.4 s** (amplitude: 7.4 mma) [APO]
- **Mode 4**: 402.4±0.4 c/d → **P₄ = 214.8 s** [ULTRACAM confirmation]
- **Mode 5**: 404.5 c/d → **P₅ = 213.7 s** (1.3 mma g-band, 5.5 mma u-band) [GTC]
- **Mode 6**: 430.2 c/d → **P₆ = 201.0 s** [GTC]

**Period Ratios** (for Spandrel framework):
- P₁/P₂ = 1.075
- P₁/P₃ = 4.70
- P₅/P₆ = 1.063

**Theoretical Context**:
- Higher mass → higher gravity → higher Brunt-Väisälä frequency → shorter g-mode periods
- Entire spectrum shifted to shorter periods compared to lower-mass WDs

**Reference**: arXiv:2510.09802 (2024)

---

### 2. WD J004917.14-252556.81 (J0049-2525)
**Previous record holder, most massive with detailed asteroseismology**

**Physical Properties**:
- **Mass**: 1.29 M☉ (ONe core, best-fit asteroseismic model)
  - Alternative: 1.31 M☉ (CO core) or 1.26 M☉ (ONe core) from initial estimate
- **T_eff**: 13,020 K (hot edge of ZZ Ceti strip)
- **log g**: 9.34 cm s⁻²
- **Distance**: 326 light years (Gaia agreement)
- **Crystallized Core**: >99%- **H-layer mass**: log(M_H/M_⋆) ≲ -7.5
- **Rotation**: Tentative period of 0.3 or 0.67 days

**Pulsation Modes (13 detected)**:
Period range: **170 to 258 seconds**

**Formation**:
- NO binary merger signatures (no magnetism, no high tangential velocity, no rapid rotation)
- Likely single-star evolution → ONe core

**Observations**:
- Initial: APO + Gemini (2 frequencies detected)
- Extended: 11 nights with NTT + Gemini + APO (13 modes)

**Scientific Importance**:
- Best chance to use asteroseismology on ONe-core WD
- Laboratory for extreme matter (densities impossible on Earth)
- Near Chandrasekhar limit (1.4 M☉)

**References**:
- Kanaan et al. 2023, MNRAS 522, 2181
- Çalışkan et al. 2025, arXiv:2505.17177 (detailed asteroseismology)

---

### 3. WD J0135+5722
**Richest pulsating ultra-massive WD (19 modes)**

**Physical Properties**:
- **Mass**: 1.118 M☉ (ONe core) or 1.135 M☉ (CO core)
- **T_eff**: ~12,000 K (center of ZZ Ceti strip)
- **Core**: Partially crystallized

**Pulsation Modes (19 detected - RECORD)**:
Period range: **137 to 1345 seconds**

**Individual Periods (partial list available)**:
- Shortest: ~137 s
- Longest: ~1345 s
- **Period Spread**: Factor of ~9.8 (large dynamic range)

**Observations**:
- APO 3.5m telescope with ARCTIC imager + BG40 filter
- UT 2023 Dec 23 and 2024 Sep 1
- Confirmed multi-periodic oscillations over several nights

**Comparison**:
- Previous record: BPM 37093 with 8 modes
- This discovery opens door to extending seismic analysis to ultra-massive WDs

**Reference**: De Gerónimo et al. 2025, ApJL 980 L9

---

### 4. BPM 37093 ("Lucy")
**First ultra-massive WD with detailed asteroseismology**

**Physical Properties**:
- **Mass**: 1.10 M☉
- **Crystallized Fraction**: 92% of total mass
- **Core**: Likely ONe composition

**Pulsation Modes (8 detected)**:
Period range: **510 to 660 seconds**

**Period Spacing**:
- **Δπ (ℓ=2)**: 17.6 seconds (average for quadrupole modes)

**Mode Trapping**:
- Evidence of mode trapping from partial crystallization
- Departures from uniform spacing encode internal structure

**Observations**:
- Whole Earth Telescope (WET) campaigns 1998 & 1999
- At least 8 independent pulsation modes detected

**Asteroseismology Results**:
- M_He atmosphere: (2.0 ± 1.0) × 10⁻⁶ M_⋆ (for prototype V777 Her comparison)
- Period spacing directly probes total stellar mass

**Reference**: Kanaan et al. 2005, A&A 432, 219-224

---

### 5. GD 518
**Ultra-massive with high crystallization**

**Properties**:
- **Mass**: 1.24 M☉
- **Crystallized Fraction**: 97%
**Reference**: Córsico et al. 2019, A&A (asteroseismological analysis)

---

### 6. SDSS J084021.23+522217.4 (SDSS J0840+5222)
**Ultra-massive with detailed asteroseismic modeling**

**Properties**:
- **Mass**: 1.16 M☉
- **Crystallized Fraction**: 81%

**Reference**: Córsico et al. 2019, A&A

---

### 7. WD J212402.03
**Properties**:
- **Mass**: 1.16 M☉

---

### 8. WD J0204+8713
**Properties**:
- **Mass**: 1.05 M☉ (lower end of ultra-massive range)

---

## III. ZZ CETI INSTABILITY STRIP PROPERTIES

### General Characteristics
- **Temperature Range**: 10,500 K ≲ T_eff ≲ 12,300 K (standard)
  - Extended range: 10,500 K ≲ T_eff ≲ 13,500 K (includes hot edge)
- **Surface Gravity**: 7.5 ≲ log g ≲ 9.35
- **Period Range**: 70-1500 s (general); 100-1400 s (typical)
- **Harmonic Degree**: ℓ ≤ 2 (low-degree modes)
- **Mode Type**: Non-radial g-modes (spheroidal)

### Massive WD Survey Results
- **Sample**: 31 massive DA WDs with M ≥ 0.9 M☉ from Montreal WD Database (100 pc)
- **Confirmed Pulsators**: 16 out of 31 (52% detection rate)
- **Instability Strip Purity**: May not be pure at high masses (weak magnetism can suppress pulsations)

### Period-Temperature-Mass Relations
- **Mean period vs T_eff**: Periods INCREASE with DECREASING temperature
- **Mean period vs Mass**: Periods DECREASE with INCREASING mass
- **Physical reason**: Higher mass → higher g → higher N (Brunt-Väisälä) → shorter periods

---

## IV. ASYMPTOTIC THEORY & PERIOD SPACING

### Asymptotic Period Spacing (Δπ)
For g-modes with high radial order k, consecutive periods (|Δk|=1) approach constant spacing:

**Formula** (Tassoul et al. 1990):
- Δπ_ℓ ≈ (2π²/√(ℓ(ℓ+1))) × (∫(N/r) dr)⁻¹
- Where N = Brunt-Väisälä frequency

**Key Properties**:
- **Mass Dependence**: Δπ is a DIRECT PROBE of total stellar mass
- **Envelope Dependence**: Longer Δπ for thinner H envelopes
- **Composition**: Weakly depends on luminosity and envelope mass

### Mode Trapping
**Observable**: Departures from uniform period spacing

**Physical Cause**:
- Sharp chemical composition gradients
- Chemical transition regions in stratified WD envelope
- Partial crystallization (especially in ultra-massive WDs)

**Pattern**:
- Δπ_k exhibits maxima and minima
- Trapping cycle and amplitude LARGER for thinner H envelopes
- Can determine H and He envelope thickness without period-by-period fitting

### Measured Values (ℓ=2 quadrupole modes)
- **BPM 37093**: Δπ = 17.6 s (M = 1.10 M☉, 92% crystallized)
- **Expected scaling**: Δπ decreases with increasing mass

---

## V. CONVECTIVE COUPLING TO PULSATIONS

### Convective Driving Mechanism
**Discovery**: Brickhill (early work), formalized by Goldreich & Wu (1999)

**Physics**:
- "Instantaneous adjustment" of convective flux to pulsations
- Quasi-adiabatic calculations
- Convective timescales vs pulsation timescales

**Observables**:
- Explains RED EDGE of ZZ Ceti instability strip (low-T boundary)
- Accounts for pulse shapes of large-amplitude pulsators
- Long-term amplitude/frequency modulations → probe mode coupling

### Convective Blocking Mechanism
**Condition**:
- Sharp base of surface convection zone
- Convective timescales SLOWER than pulsation timescales

**Effect**:
- Convective flows react too slowly to perturbations
- Large, coherent pulsations can build up
- Sets BLUE EDGE of instability strip (high-T boundary)

### Turbulent Convection
**Stochastic vs Self-Excited**:
- WD pulsations: SELF-EXCITED (eigen-modes)
- Solar-like stars: STOCHASTIC (forced by turbulent convection)

**Turbulent Diffusion**:
- Higher-order poloidal magnetic field modes
- Driven by crystallization-induced convection
- Affects surface magnetic field predictions (factor 2-4 variation)

---

## VI. FRACTAL DIMENSION IN WD EXPLOSIONS

### Previous Research (Type Ia SNe simulations)

**Deflagration-to-Detonation Transition (DDT)**:
- Burning starts as subsonic deflagration
- Transitions to supersonic detonation
- Mechanism unknown in detail

**Fractal Flame Front**:
- **Measured D**: ~2.36 (in 3D simulations)
- Used to estimate flame area in subgrid-scale models
- DDT probability calculated from:
  - Distribution of turbulent velocities on grid scale
  - Fractal flame surface area

**Time Evolution**:
- Fractal dimension D varies with time during explosion
- Resolution-dependent measurements
- Higher central density → faster growth of subgrid turbulence → earlier DDT

**References**:
- A&A 559, A117 (2013) - DDT subgrid model
- MNRAS 414, 2709 (2011) - Central density as secondary parameter

---

## VII. SPANDREL-ASTEROSEISMIC FRAMEWORK

### Proposed Connection: D ↔ Pulsation Properties

**Chain of Physical Causation**:
1. **Pre-explosion turbulence** in convective regions → sets fractal dimension D
2. **G-mode pulsations** couple to convective zones (via convective driving/blocking)
3. **Period ratios & spacing** encode turbulent structure:
   - P₁/P₂: Fundamental vs overtone → convective zone depth
   - Δπ: Period spacing → mass, envelope thickness, stratification
   - Mode trapping patterns → chemical gradients from convection
4. **D is predictable** from asteroseismic signature BEFORE explosion

### Observable Parameters (from pulsations)
- **P_i**: Individual mode periods
- **P_i/P_j**: Period ratios between modes
- **Δπ**: Asymptotic period spacing
- **Trapping amplitude**: Departure from uniform spacing
- **T_eff**: Effective temperature
- **M_WD**: Total mass (from Δπ and spectroscopy)
- **M_H, M_He**: Envelope masses (from mode trapping)

### Predicted Relationship
**D = f(P₁/P₂, Δπ, T_eff, M_WD)**

Where:
- Higher turbulence → larger D
- Turbulence couples to g-mode cavity
- Observable in period structure

### Key Testable Predictions
1. **Ultra-massive WDs** (M > 1.2 M☉) should show distinct D signatures
2. **High crystallization** (>90%) may affect turbulent coupling
3. **ONe vs CO cores** may have different convective/turbulent properties → different D
4. **WDJ181058.67+311940.94** (if pulsating) provides calibration for super-Chandrasekhar progenitors

---

## VIII. DBV (V777 Her) STARS - Helium Atmosphere Pulsators

### Why Include DBVs?
- Some SNe Ia may have He-burning phase
- Different convective properties than H-atmosphere WDs
- Complementary mass range

### Properties
- **Spectral Type**: DB (He-dominated atmosphere)
- **Temperature Range**: 22,000-29,000 K (HOTTER than ZZ Ceti)
- **Period Range**: 150-1100 s (100-1400 s in some sources)
- **Mode Type**: Non-radial g-modes
- **Known DBVs**: 47 objects (as of recent count)

### Driving Mechanism
- **Helium recombination** in outer envelope
- κ-γ mechanism in He partial ionization zone (sets BLUE edge)
- Convective driving (sets RED edge once convection zone deepens)
- Huge opacity increase during He recombination phase

### Prototype: V777 Her
- **T_eff**: 25,900 K
- **log g**: 7.9
- **Mass**: 0.63 ± 0.03 M☉
- **M_He**: (2.0 ± 1.0) × 10⁻⁶ M_⋆
- No atmospheric hydrogen

### Scientific Value
- Asteroseismology allows inference of: origin, internal structure, evolution
- Mass, He layer mass, core composition, B-field, rotation
- Outer convection zone properties
- Different turbulent regime than ZZ Ceti stars

---

## IX. PUBLIC DATABASES & RESOURCES

### La Plata Evolutionary Group
- **Database**: Complete bank of chemical profiles and g-mode periods for ZZ Ceti stars
- **Models**: Fully evolutionary computations from ZAMS → WD cooling
- **Coverage**:
  - Wide range of stellar masses
  - Effective temperatures
  - H envelope thicknesses
  - Dipole (ℓ=1) and quadrupole (ℓ=2) periods
- **Availability**: Free download from La Plata website
- **Reference**: arXiv:1204.6101

### Montreal White Dwarf Database
- 100 pc sample with M ≥ 0.9 M☉ massive WDs
- Used for recent ultra-massive ZZ Ceti survey
- Spectroscopic and photometric data

---

## X. NEXT STEPS FOR SPANDREL FRAMEWORK

### Immediate Actions
1. **Check WDJ181058.67+311940.94 for pulsations**
   - Access VLT/WHT/NOT data archives
   - Time-series photometry essential
   - If pulsating: First super-Chandrasekhar pulsator at 49 pc

2. **Download complete period tables**
   - WD J0135+5722: All 19 modes (De Gerónimo et al. 2025)
   - WD J0049-2525: All 13 modes (Çalışkan et al. 2025)
   - BPM 37093: WET data (Kanaan et al. 2005)
   - La Plata database: Full theoretical grid

3. **Calculate period ratios for ALL ultra-massive WDs**
   - P₁/P₂, P₂/P₃, etc. for each object
   - Compare with theoretical models
   - Look for correlations with mass, crystallization

### Theoretical Development
1. **Model convective-turbulent coupling to g-modes**
   - Use existing convective driving theory
   - Connect turbulent velocity field to D
   - Predict how D modulates period structure

2. **Calibrate D from simulations**
   - Extract D from known SNe Ia (your existing work)
   - Match to progenitor properties (if known)
   - Build empirical D(M_WD, T_eff) relation

3. **Test on BPM 37093 & WD J0135+5722**
   - Most modes detected → best period structure
   - Calculate predicted D
   - Compare with D~2.36 from simulations

### Observational Targets
1. **High priority**: WDJ181058.67+311940.94 (closest super-Chandra progenitor)
2. **Medium priority**: All 16 confirmed massive ZZ Ceti pulsators
3. **Future**: DBV ultra-massive pulsators (if any exist)

### Data Products Needed
- **Table 1**: Object | Mass | T_eff | Periods | Δπ | P₁/P₂ | Crystallization
- **Table 2**: Theoretical grid from La Plata (M, T_eff, M_H) → periods
- **Table 3**: SN Ia events with known D → match to progenitor constraints

---

## XI. SUMMARY STATISTICS

### Ultra-Massive Pulsating WDs (M > 1.05 M☉)
- **Confirmed**: 8+ objects
- **Most massive**: J0959-1828 (1.32 M☉)
- **Richest**: WD J0135+5722 (19 modes)
- **Best asteroseismology**: WD J0049-2525 (13 modes, >99% crystallized)
- **Period range**: 137-1345 s (all objects combined)

### Key Period Spacings (ℓ=2)
- **BPM 37093** (1.10 M☉): Δπ = 17.6 s
- **Expected scaling**: Δπ ∝ M⁻¹/² (approximate)

### Fractal Dimension from Simulations
- **D (flame front)**: ~2.36 in 3D DDT models
- **Time-dependent**: Evolves during explosion
- **Resolution-dependent**: Requires subgrid modeling

### Discovery Rate
- **Survey**: 31 candidates (M ≥ 0.9 M☉)
- **Confirmed**: 16 pulsators (52%)
- **Record**: J0959-1828 discovered 2024

---

## XII. REFERENCES

### Key Papers (chronological)

**1990**: Tassoul et al. - Asymptotic theory of stellar pulsations

**1999**: Goldreich & Wu - Convective driving theory

**2005**: Kanaan et al., A&A 432, 219 - BPM 37093 WET observations

**2011**: MNRAS 414, 2709 - Type Ia SN diversity, D~2.36

**2013**: A&A 559, A117 - DDT model with fractal flame

**2019**: Córsico et al., A&A - Asteroseismology of BPM 37093, GD 518, SDSS J0840+5222

**2023**: Kanaan et al., MNRAS 522, 2181 - Discovery of J0049-2525 pulsations

**2024**: arXiv:2510.09802 - ZZ Ceti strip for massive WDs, discovery of J0959-1828

**2025**: De Gerónimo et al., ApJL 980 L9 - WD J0135+5722 (19 modes)

**2025**: Çalışkan et al., arXiv:2505.17177 - Detailed asteroseismology of J0049-2525

**2025**: Munday et al., Nature Astronomy - WDJ181058.67+311940.94 super-Chandra progenitor at 49 pc

### Databases
- La Plata evolutionary models: arXiv:1204.6101
- Montreal White Dwarf Database (100 pc sample)

---

## XIII. CONTACT & COLLABORATION

**Spandrel Framework Lead**: [Your institution/contact]

**Key Collaborators Needed**:
- WD asteroseismology experts (La Plata group, Whole Earth Telescope consortium)
- SN Ia progenitor observers (VLT, Gemini, HST time)
- Convection/turbulence theorists (mixing length theory, 3D simulations)
- DDT simulation experts (fractal flame front modeling)

**Proposed Name**: "Spandrel-Asteroseismic Progenitor Diagnostic (SAPD)"

**Goal**: Predict D for individual SN Ia progenitors from pulsation data, enabling:
- Observational test of Spandrel hypothesis
- Early-warning system for nearby super-Chandrasekhar mergers
- New standard candle calibration (D-corrected luminosity)
