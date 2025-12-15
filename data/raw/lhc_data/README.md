# LHC Data Directory

This directory contains data from Large Hadron Collider (LHC) experiments, particularly focusing on:
1.  **Oxygen-Oxygen (O-O) Collisions (July 2025)**: ALICE and CMS results testing QGP formation in small systems.
2.  **W/Z Boson Precision Measurements**: Tests of the Standard Model and dimensional transmutation.
3.  **Jet Quenching**: ATLAS results on energy loss in small systems.

## Data Sources & Access

### 1. ALICE O-O Collisions (July 2025)
*   **Primary Paper:** *Evidence of nuclear geometry-driven anisotropic flow in OO and Ne-Ne collisions at $\sqrt{s_{\rm NN}}$ = 5.36 TeV* (arXiv:2509.06428).
*   **Data:** See `alice_oo_flow_2025.yaml` (Placeholder/Supplementary).
*   **Key Finding:** Nuclear geometry (alpha-clustering) drives flow even in light systems like O-O and Ne-Ne.

### 2. CMS Small-Scale QGP (PRL 132, 172302)
*   **Primary Paper:** *CMS Collaboration, Phys. Rev. Lett. 132, 172302 (2024)*.
*   **Data File:** `cms_small_system_v2.csv` (Reconstructed from Figure 3).
*   **Key Finding:** v2 > 0 for N_ch > 10, establishing the "turn-on" of collective behavior.

### 3. ATLAS O-O Jet Quenching
*   **Primary Source:** ATLAS Briefings / Initial Stages 2025 (Taipei).
*   **Data File:** `atlas_oo_quenching_2025.csv` (Preliminary).
*   **Key Finding:** Slight suppression (R_AA ~ 0.92) hints at onset of energy loss in O-O.

### 4. W Boson Mass (2025 Updates)
*   **Data File:** `w_boson_mass_2025.csv`.
*   **Key Finding:** CMS (2024/25) and ATLAS (2024) agree with SM (80360 MeV), resolving the CDF II anomaly (80433 MeV). This constrains "dimensional transmutation" models that might have relied on the anomaly.

## Files in this Directory
*   `cms_small_system_v2.csv`: v2 vs Multiplicity.
*   `atlas_oo_quenching_2025.csv`: Jet quenching R_AA values.
*   `w_boson_mass_2025.csv`: Precision mass measurements.
*   `alice_oo_flow_2025.yaml` (to be added).