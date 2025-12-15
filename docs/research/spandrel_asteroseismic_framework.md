# Spandrel-Asteroseismic Prediction Framework
## Predicting Fractal Dimension D of SN Ia Progenitors from White Dwarf Pulsations

---

## EXECUTIVE SUMMARY

**Goal**: Predict the fractal dimension D of Type Ia supernova progenitors BEFORE explosion using asteroseismic observations of ultra-massive white dwarf pulsators.

**Physical Basis**:
- Convective turbulence in pre-explosion white dwarfs sets the initial fractal dimension D
- White dwarf g-mode pulsations couple to convective regions via convective driving/blocking
- Pulsation period ratios and spacing patterns encode turbulent structure information
- Therefore: D can be predicted from observable pulsation properties

**Current Status**:
- Ultra-massive WD pulsators identified (8+ objects with M > 1.05 M☉)
- Detailed pulsation data available for 3-4 key objects
- Fractal dimension D ~ 2.36 measured in SN Ia explosion simulations
- **Priority**: WDJ181058.67+311940.94 (closest super-Chandrasekhar progenitor at 49 pc) - pulsation status unknown
- Direct connection between pulsations and D not yet established

---

## I. THEORETICAL FRAMEWORK

### A. Physical Chain of Causation

```
Pre-explosion WD interior:
  ↓
Convective turbulence (κ-γ mechanism, He/H burning)
  ↓
Sets fractal structure of turbulent cascade
  ↓
Fractal dimension D characterizes turbulence
  ↓
G-modes couple to convective zones (coupling strength ~ turbulent velocity)
  ↓
Period structure encodes turbulent properties:
  - Period ratios P_i/P_j → convective zone geometry
  - Period spacing Δπ → mass, stratification, envelope thickness
  - Mode trapping → chemical gradients from convective mixing
  ↓
Observable asteroseismic signature contains D information
  ↓
Explosion preserves D (or transforms it predictably)
  ↓
SN Ia observables (light curve, spectra) → measured D
```

### B. Key Physical Mechanisms

#### 1. Convective Driving (Brickhill, Goldreich & Wu 1999)
- Convective flux adjusts instantaneously to pulsations
- Quasi-adiabatic limit for WD conditions
- **Sets red edge** of instability strip (low T boundary)
- Explains pulse shapes and amplitude modulation

**Coupling Strength**:
```
E_conv ~ (v_conv / v_sound)² × L_conv
```
Where v_conv is turbulent velocity → related to D

#### 2. Convective Blocking
- Base of convection zone sharp
- Convective timescales τ_conv > pulsation periods P
- Allows coherent pulsations to build up
- **Sets blue edge** of instability strip (high T boundary)

#### 3. Mode Trapping
- Chemical gradients create "potential wells" for g-modes
- Departures from uniform Δπ encode gradient structure
- **Gradients set by convective mixing** → turbulent diffusion → D

**Observable**:
```
Δπ_k = π_{k+1} - π_k  [deviates from constant Δπ_asymptotic]
Trapping amplitude ~ gradient sharpness ~ D
```

### C. Proposed Functional Form

**Primary Relation**:
```
D = f(P₁/P₂, Δπ, T_eff, M_WD, [mode trapping amplitude])
```

**Dimensional Analysis**:
```
D ~ 2 + α × (P₁/P₂ - 1) + β × (Δπ/Δπ_ref) + γ × log(M_WD/M_Chandra)
```

Where:
- α, β, γ are coupling coefficients (to be calibrated)
- Δπ_ref ~ 17.6 s (from BPM 37093 at M = 1.10 M☉)
- M_Chandra = 1.4 M☉

**Physical Interpretation**:
- **Baseline D ~ 2**: Laminar/weakly turbulent (Kolmogorov D ~ 5/3 + 1/3 for 2D projection)
- **P₁/P₂ term**: Measures overtone structure → convective zone depth
- **Δπ term**: Encodes mass and stratification → turbulent layer thickness
- **M_WD term**: Near-Chandrasekhar WDs have different convective physics

**Alternative (Empirical)**:
```
D = D₀ × (1 + A × [P₁/P₂]^n₁) × (1 + B × [Δπ]^n₂) × (M_WD/M_☉)^n₃
```
Where D₀ ~ 2.36 (from simulations)

---

## II. OBSERVATIONAL DATA REQUIREMENTS

### A. For Individual Progenitor Candidates

**Minimum Requirements**:
1. **Stellar parameters** (spectroscopy):
   - T_eff (±100 K precision)
   - log g (±0.05 dex precision)
   - Mass M_WD (from g + evolutionary models)

2. **Pulsation detection** (time-series photometry):
   - ≥3 independent modes detected
   - Period precision: Δπ/π < 10⁻⁴ (achievable with multi-night campaigns)
   - Mode identification (ℓ = 1 vs ℓ = 2)

3. **Mode structure**:
   - Period ratios P₁/P₂, P₂/P₃, etc.
   - Asymptotic spacing Δπ
   - Trapping amplitude (deviation from uniform spacing)

**Optimal Requirements** (for best D constraint):
1. **Rich pulsation spectrum**: ≥8 modes (like BPM 37093) or ≥13 modes (like J0049-2525)
2. **Wide period coverage**: Factor of >3 in period range
3. **High-precision photometry**: Space-based (TESS, future missions) or multi-site ground campaigns
4. **Long baseline**: Monitor for rotation, evolution (dπ/dt)

### B. For Framework Calibration

**Need**:
1. **Training set**:
   - 10-20 ultra-massive WDs with detailed asteroseismology
   - Covering range: 1.05 < M/M☉ < 1.35
   - Covering range: 10,500 < T_eff < 13,500 K

2. **Validation set**:
   - Known SN Ia with measured D (from your existing work)
   - Progenitor constraints (M_WD, age, environment)
   - Match observed D to predicted D from progenitor properties

3. **Theoretical models**:
   - La Plata database (full evolutionary sequences)
   - 3D convection simulations → extract D from turbulent velocity field
   - DDT simulations with measured D_flame → connect to pre-explosion D

---

## III. KEY OBSERVATIONAL TARGETS

### Tier 1: High Priority

#### 1. WDJ181058.67+311940.94
**Why**: Closest known super-Chandrasekhar SN Ia progenitor (49 pc)
- Will explode in 22.6 Gyr (measurable evolution)
- M = 1.555 M☉ (well above Chandrasekhar)
- Predicted double-detonation
- **Action**: Check for pulsations
  - Time-series photometry with 1-2 minute cadence
  - Multi-night campaign (≥5 nights)
  - TESS may have data already (check archive)

**Predicted pulsation properties** (if in instability strip):
- Shorter periods than J0959-1828 (higher mass)
- Possibly <200 s for dominant modes
- May have exotic mode structure (double WD system)

**Impact**:
- If pulsating: FIRST Spandrel calibrator for super-Chandrasekhar progenitors
- Unique test of D(M) scaling above M_Chandra
- Will constrain D 23 Gyr before explosion.

---

### Tier 2: High Value Targets

#### 2. WD J0135+5722
**Why**: Richest pulsator (19 modes) = best period structure
- M = 1.12-1.14 M☉
- Periods: 137-1345 s (factor 9.8 range)
- Partially crystallized
- **ACTION**: Obtain complete mode identification
  - Download De Gerónimo et al. 2025 full tables
  - Identify ℓ = 1 vs ℓ = 2 modes
  - Calculate all period ratios
  - Measure Δπ for each ℓ sequence

**Use**: Benchmark for D(period structure) calibration

#### 3. WD J004917.14-252556.81 (J0049-2525)
**Why**: Most massive with detailed asteroseismology
- M = 1.29 M☉ (ONe core likely)
- 13 modes: 170-258 s
- >99% crystallized
- Single-star evolution (no merger)
- **ACTION**: Access Çalışkan et al. 2025 full mode list
  - Extract all 13 periods
  - Calculate Δπ
  - Compare crystallized vs non-crystallized period structure

**Use**:
- Test if high crystallization affects D prediction
- ONe core may have different convective properties than CO

#### 4. J0959-1828
**Why**: Current mass record holder
- M = 1.32 M☉ (highest confirmed)
- 6 modes detected
- Wide period range: 201-1013 s
- **ACTION**: Use existing data from arXiv:2510.09802

**Use**: Anchor high-mass end of D(M) relation

---

### Tier 3: Supporting Targets

#### 5. BPM 37093
- 8 modes, Δπ = 17.6 s well-measured
- 92% crystallized
- Historical WET data
- **Use**: Baseline for Δπ scaling

#### 6. GD 518 & SDSS J0840+5222
- Ultra-massive (1.16-1.24 M☉)
- High crystallization (81-97%)
- **Use**: Fill in mass range

---

## IV. COMPUTATIONAL WORKFLOW

### Step 1: Data Acquisition
```python
# Pseudo-code for data pipeline

import numpy as np
import pandas as pd

# Load observational data
wd_data = pd.read_csv('wd_pulsation_periods_table.csv')

# For each object with ≥3 modes:
for wd in wd_data.iterrows():
    periods = extract_periods(wd)  # Array of periods in seconds
    mass = wd['Mass_Msun']
    teff = wd['Teff_K']

    # Calculate derived quantities
    period_ratios = compute_ratios(periods)  # All P_i/P_j
    delta_pi = asymptotic_spacing(periods, ell=2)  # For ℓ=2 modes
    trapping_amp = measure_trapping(periods, delta_pi)

    # Store for framework
    asteroseismic_params[wd.name] = {
        'P1/P2': period_ratios[0],
        'Delta_Pi': delta_pi,
        'Trapping': trapping_amp,
        'Mass': mass,
        'Teff': teff
    }
```

### Step 2: Fractal Dimension Prediction

**Model 1: Linear (for initial testing)**:
```python
def predict_D_linear(P1_P2, Delta_Pi, M_WD, coeffs):
    """
    Linear model: D = a0 + a1*(P1/P2) + a2*Delta_Pi + a3*M_WD
    """
    a0, a1, a2, a3 = coeffs
    D = a0 + a1*(P1_P2 - 1.0) + a2*(Delta_Pi/17.6) + a3*np.log10(M_WD/1.4)
    return D

# Initial guess (to be calibrated):
coeffs_init = [2.36, 0.5, -0.2, 0.3]

D_predicted = predict_D_linear(
    P1_P2 = asteroseismic_params['J0959']['P1/P2'],
    Delta_Pi = asteroseismic_params['J0959']['Delta_Pi'],
    M_WD = asteroseismic_params['J0959']['Mass'],
    coeffs = coeffs_init
)
```

**Model 2: Nonlinear (physics-motivated)**:
```python
def predict_D_physics(params, asteroseism):
    """
    Physics-based model connecting convection to pulsations

    Parameters:
    - params: [v_conv, L_conv, tau_conv] from stellar models
    - asteroseism: observed periods and spacing

    Returns:
    - D: predicted fractal dimension
    """
    # 1. Estimate convective properties from period structure
    v_conv = estimate_convection_velocity(asteroseism)

    # 2. Calculate turbulent Reynolds number
    Re_turb = v_conv * L_conv / nu_mol

    # 3. Fractal dimension from Kolmogorov scaling
    # For 3D turbulence: D ~ 8/3, but WD convection may differ
    D_3D = 2 + (Re_turb / Re_crit)**0.5  # Example scaling

    # 4. Projection to 2D (flame front is 2D surface in 3D)
    D_flame = D_3D - 1.0  # Rough estimate

    return D_flame
```

### Step 3: Calibration Against Known SNe Ia

```python
# Use your existing D measurements from SN Ia observations
sne_ia_data = load_pantheon_D_measurements()  # Your previous work

# Match SNe to progenitor constraints
matched_systems = match_sne_to_progenitors(sne_ia_data, wd_data)

# Optimize coefficients to minimize prediction error
from scipy.optimize import minimize

def objective(coeffs):
    """Minimize |D_predicted - D_observed| for matched systems"""
    error = 0
    for system in matched_systems:
        D_pred = predict_D_linear(
            system['P1/P2'],
            system['Delta_Pi'],
            system['Mass'],
            coeffs
        )
        D_obs = system['D_measured']
        error += (D_pred - D_obs)**2
    return error

result = minimize(objective, coeffs_init, method='Nelder-Mead')
coeffs_calibrated = result.x

print(f"Calibrated coefficients: {coeffs_calibrated}")
print(f"D = {coeffs_calibrated[0]:.2f} + "
      f"{coeffs_calibrated[1]:.2f}*(P1/P2-1) + "
      f"{coeffs_calibrated[2]:.2f}*(ΔΠ/17.6s) + "
      f"{coeffs_calibrated[3]:.2f}*log(M/1.4M☉)")
```

### Step 4: Application to New Progenitors

```python
# For WDJ181058 (when pulsation data available):
D_WDJ181058 = predict_D_linear(
    P1_P2 = measured_from_observations,
    Delta_Pi = measured_from_observations,
    M_WD = 1.555,  # Known from radial velocities
    coeffs = coeffs_calibrated
)

print(f"Predicted D for WDJ181058: {D_WDJ181058:.2f}")
print(f"Expected SN Ia subluminosity: {calculate_subluminosity(D_WDJ181058)}")
print(f"Time until explosion: 22.6 ± 1.0 Gyr")
```

---

## V. VALIDATION STRATEGY

### A. Internal Consistency Checks

1. **Scaling Relations**:
   - Verify D increases with mass (higher turbulence near M_Chandra)
   - Check D vs T_eff (hotter → stronger convection?)
   - Test D vs crystallization (frozen core = reduced turbulence?)

2. **Mode Identification**:
   - Period ratios for ℓ=1 vs ℓ=2 differ by factor ~√3
   - Ensure correct mode assignment before computing ratios

3. **Theoretical Comparison**:
   - La Plata models: compute periods for given M, T_eff, M_H
   - Compare predicted vs observed period structure
   - Residuals may correlate with D

### B. External Validation

1. **Nearby SNe Ia**:
   - SN 2014J (closest recent SN Ia): D measured from observations
   - Search for pre-explosion imaging → progenitor constraints
   - Compare predicted D (if WD detected) to measured D

2. **Double-Detonation Candidates**:
   - HD 265435 (sdB-WD binary, 70 Myr to explosion)
   - Has tidally-tilted pulsations in sdB component
   - Test framework on sdB pulsations

3. **Statistical Tests**:
   - Pantheon+ sample: 1701 SNe Ia
   - Extract D for each (your existing work)
   - Match to progenitor population statistics
   - Verify D distribution matches predicted distribution from WD demographics

### C. Predictive Tests

**Falsifiable Predictions**:

1. **WDJ181058.67+311940.94** (if pulsating):
   - Predicted: D > 2.4 (higher than normal due to super-Chandrasekhar mass)
   - Predicted: Δπ < 15 s (higher mass → shorter spacing)
   - Predicted: Subluminous SN Ia (already predicted from double-detonation)
   - **Test in 23 Gyr** (or sooner if nearby analog found)

2. **Mass Scaling**:
   - D(1.3 M☉) > D(1.1 M☉) for similar T_eff
   - If violated: convection suppressed in very massive WDs?

3. **Crystallization Effect**:
   - J0049-2525 (>99% crystallized) vs J0959 (less crystallized)
   - Predict: Higher crystallization → lower D (frozen turbulence)
   - Measure from period structure deviations

---

## VI. IMMEDIATE ACTION ITEMS

### Week 1: Data Compilation
- [ ] Download De Gerónimo et al. 2025 (ApJL 980 L9) - WD J0135+5722 full period table
- [ ] Download Çalışkan et al. 2025 (arXiv:2505.17177) - J0049-2525 full period table
- [ ] Access La Plata database (arXiv:1204.6101) - theoretical grid
- [ ] Compile all available periods into master table

### Week 2: Period Analysis
- [ ] Calculate P₁/P₂, P₂/P₃, etc. for all objects
- [ ] Measure Δπ for each object (ℓ=1 and ℓ=2 sequences separately)
- [ ] Quantify mode trapping amplitude
- [ ] Create diagnostic plots: P vs M, Δπ vs M, etc.

### Week 3: WDJ181058 Investigation
- [ ] Check TESS archive for observations of WDJ181058.67+311940.94
- [ ] Search literature for any follow-up photometry
- [ ] If no data: propose observing campaign (NOT, WHT, VLT)
- [ ] Calculate expected periods from theoretical models (M=1.555 M☉, estimate T_eff)

### Week 4: Framework Implementation
- [ ] Code up predict_D_linear() function
- [ ] Load your Pantheon+ D measurements
- [ ] Attempt first calibration (even with sparse data)
- [ ] Generate preliminary predictions for all 8 ultra-massive WDs

### Month 2: Theoretical Development
- [ ] Contact La Plata group for collaboration
- [ ] Obtain 3D convection simulations for massive WDs
- [ ] Extract turbulent velocity fields → calculate D directly
- [ ] Connect simulation D to observable period structure

### Month 3: Observing Proposal
- [ ] Write proposal for WDJ181058 time-series photometry
- [ ] Target: 10 nights over 2-3 months (rotation coverage)
- [ ] Cadence: 1-2 minutes (Nyquist for 200-1000 s periods)
- [ ] Facilities: VLT/FORS2, WHT/ACAM, NOT/ALFOSC, or 2-4m class

---

## VII. EXPECTED OUTCOMES

### Scientific Impact

1. **Pre-Explosion D Prediction**:
   - Enable observational test of Spandrel hypothesis
   - Predict SN Ia properties decades to Gyr before explosion
   - New diagnostic for SN Ia progenitor channels

2. **Improved Standard Candle**:
   - D-corrected luminosity: M_B = f(light curve, D)
   - Reduce scatter in Hubble diagram
   - Better constraints on dark energy equation of state

3. **Asteroseismology of SN Ia Progenitors**:
   - New subfield connecting WD pulsations to explosive outcomes
   - Probe interior turbulence via g-mode period structure
   - Constrain convective physics in extreme conditions

4. **Early Warning System**:
   - WDJ181058 at 49 pc will be BRIGHT (m_V = -16)
   - Could affect Earth if much closer
   - Demographic studies: how many nearby super-Chandrasekhar progenitors?

### Publications Roadmap

**Paper I**: "Ultra-Massive White Dwarf Pulsation Database" (This work)
- Compile all periods, masses, properties
- Calculate period ratios and spacing for all objects
- Identify correlations with mass, T_eff, crystallization

**Paper II**: "Spandrel-Asteroseismic Framework: Theory"
- Develop physical connection: convection → turbulence → D → pulsations
- Predict functional form D = f(periods)
- Benchmark with 3D simulations

**Paper III**: "Calibration with Pantheon+ SNe Ia"
- Match SNe Ia D measurements to progenitor constraints
- Calibrate framework coefficients
- Validate on independent sample

**Paper IV**: "WDJ181058: Predicting D for a Super-Chandrasekhar Progenitor"
- Present pulsation observations (if detected)
- Apply framework to predict D
- Forecast SN Ia properties 23 Gyr in advance

**Paper V**: "Asteroseismic Standard Candles: D-Corrected SN Ia Luminosity"
- Apply D-corrections to Pantheon+ sample
- Reduce Hubble diagram scatter
- Improved cosmological constraints

---

## VIII. POTENTIAL CHALLENGES

### Observational

1. **WDJ181058 may not pulsate**:
   - Could be outside instability strip (too hot or too cold)
   - Double WD system may suppress pulsations (tidal effects)
   - Fallback: Apply framework to other super-Chandrasekhar candidates when found

2. **Period identification difficult**:
   - Need long time baseline for ℓ identification
   - Multi-site campaigns expensive
   - TESS has limited cadence for short periods

3. **Sparse progenitor constraints**:
   - Most SNe Ia have no direct progenitor detection
   - Must rely on statistical matching (uncertain)

### Theoretical

1. **Convection-pulsation coupling poorly understood**:
   - Convective driving is simplified model
   - 3D time-dependent convection needed for full treatment
   - May require MESA simulations with detailed mixing

2. **D may not be set by pre-explosion turbulence**:
   - Could be set during explosion itself (DDT phase)
   - Pre-explosion D may be erased or transformed
   - Need to connect pre- and post-explosion D

3. **Crystallization effects**:
   - >99% crystallized WDs have frozen cores
   - Pulsations may not probe convective regions
   - Framework may only work for <90% crystallization

### Statistical

1. **Small sample size**:
   - Only 8 ultra-massive WDs with pulsations
   - Only 3-4 with rich mode spectra
   - May need 10-20 for robust calibration

2. **Selection effects**:
   - Nearby WDs preferentially detected
   - Pulsation detection biased toward certain modes
   - May miss key progenitor population

---

## IX. LONG-TERM VISION

### 5-Year Goals

1. **Framework Operational**:
   - D predictions for all ultra-massive pulsators
   - Calibrated against ≥10 SNe Ia with known D
   - Published and available to community

2. **WDJ181058 Characterized**:
   - Pulsation status determined
   - If pulsating: D predicted
   - Long-term monitoring established

3. **New Discoveries**:
   - TESS/PLATO find 10-20 more ultra-massive pulsators
   - Expand mass range to 1.35-1.45 M☉ (if exist)
   - Discover more super-Chandrasekhar binaries

### 10-Year Goals

1. **D-Corrected Cosmology**:
   - Apply to full Pantheon+ sample (1701 SNe Ia)
   - Reduce Hubble tension with improved standard candles
   - Constrain w(z) evolution to <1%

2. **Multi-Messenger Progenitor Science**:
   - GW detection of WD mergers (LISA era)
   - Predict D from pre-merger pulsations
   - Test predictions with post-merger SN Ia

3. **Complete Progenitor Census**:
   - Map all super-Chandrasekhar systems in Local Volume
   - Predict which will explode as SNe Ia
   - Establish SN Ia rate from WD demographics

---

## X. SUMMARY

**The Spandrel-Asteroseismic framework offers a new approach to Type Ia supernova progenitor science**:

- **Physics**: Connects pre-explosion turbulence (D) to observable pulsations
- **Observations**: Leverages existing WD asteroseismology techniques
- **Prediction**: Can forecast SN Ia properties Gyr before explosion
- **Validation**: Testable against measured D from SN Ia observations
- **Impact**: Improves standard candles, probes convection physics, early warning

**Next Steps**:
1. Obtain pulsation data for WDJ181058.67+311940.94 (49 pc super-Chandrasekhar progenitor)
2. Compile complete period tables for WD J0135+5722 (19 modes) and J0049-2525 (13 modes)
3. Implement prediction framework and calibrate against Pantheon+ D measurements
4. Publish framework and apply to all ultra-massive WD pulsators

**Timeline**: Framework operational in 12-18 months, first D predictions in 2 years

**Contact**: [Your name/institution for collaboration inquiries]

---

## REFERENCES

See `ultra_massive_wd_pulsation_database.md` for complete reference list.

Key papers:
- Munday et al. 2025, Nat Ast - WDJ181058 discovery
- De Gerónimo et al. 2025, ApJL 980 L9 - WD J0135+5722
- Çalışkan et al. 2025, arXiv:2505.17177 - J0049-2525 asteroseismology
- Goldreich & Wu 1999 - Convective driving theory
- A&A 559, A117 (2013) - Fractal flame D ~ 2.36
