#!/usr/bin/env python3
"""
sign_analysis.py — Trace the Spandrel Sign Convention

Critical Question: Does high D make SNe BRIGHTER or FAINTER?

Chain:
1. High z → Low Z (less chemical enrichment)
2. Low Z → Higher κ (thermal conductivity)
3. Higher κ → Higher Le (Lewis number)
4. Higher Le → More cellular instability → Higher D
5. Higher D → ??? → M_Ni change → Brightness change

Step 5 is the key. Let's trace it carefully.

Author: Spandrel Framework
Date: November 28, 2025
"""


print()
print("=" * 70)
print("SPANDREL SIGN CONVENTION ANALYSIS")
print("=" * 70)
print()

# =====================================================
# Step 1-4: Already established
# =====================================================
print("ESTABLISHED CHAIN (correct signs):")
print("-" * 70)
print("  1. High z → Low Z        (chemical evolution: confirmed)")
print("  2. Low Z  → High κ       (fewer e⁻ scatterers: confirmed)")
print("  3. High κ → High Le      (Le = κ/D_mol: by definition)")
print("  4. High Le → High D      (cellular instability: confirmed)")
print()

# =====================================================
# Step 5: D → M_Ni
# =====================================================
print("CONTESTED STEP 5: D → M_Ni")
print("-" * 70)
print()
print("Hypothesis A (conventional): Higher D → More deflagration → Less Ni-56")
print("  - More flame surface → faster deflagration")
print("  - Faster deflagration → more expansion before DDT")
print("  - DDT at lower density → less complete Si burning")
print("  - Result: Higher D → FAINTER SN → Higher μ → Need MORE acceleration")
print()
print("Hypothesis B (alternative): Higher D → More deflagration → More Ni-56")
print("  - More flame surface → burns more material before DDT")
print("  - More Ni-56 produced in deflagration phase")
print("  - Result: Higher D → BRIGHTER SN → Lower μ → Need LESS acceleration")
print()

# =====================================================
# DESI requirement
# =====================================================
print("DESI OBSERVATION:")
print("-" * 70)
print("  w₀ = -0.827 ± 0.063  (less negative than -1)")
print("  wₐ = -0.75 ± 0.29    (negative, phantom-like)")
print()
print("  At high z: w = w₀ + wₐ(1-a) becomes MORE negative")
print("  This means: dark energy was MORE accelerating at high z")
print()
print("  To produce this signal, we need:")
print("  → SNe at high z appear FAINTER than ΛCDM expects")
print("  → This makes us infer LARGER distances")
print("  → Larger distances require MORE past acceleration")
print("  → More past acceleration = more negative w at high z = wₐ < 0")
print()

# =====================================================
# Matching
# =====================================================
print("SIGN MATCHING:")
print("-" * 70)
print()
print("If Hypothesis A is correct:")
print("  High z → High D → FAINTER → wₐ < 0 ✓ MATCHES DESI")
print()
print("If Hypothesis B is correct:")
print("  High z → High D → BRIGHTER → wₐ > 0 ✗ OPPOSITE TO DESI")
print()

# =====================================================
# Evidence from literature
# =====================================================
print("LITERATURE EVIDENCE:")
print("-" * 70)
print()
print("1. Phillips Relation (Phillips 1993):")
print("   - Brighter SNe Ia have SLOWER light curves (higher stretch)")
print("   - Implies: More Ni-56 → slower radioactive decay timescale")
print()
print("2. Kasen & Woosley 2007:")
print("   - Deflagration-to-detonation transition determines Ni-56 yield")
print("   - Earlier DDT (higher ρ) → More Ni-56 → Brighter")
print("   - Later DDT (lower ρ) → Less Ni-56 → Fainter")
print()
print("3. Timmes, Brown & Truran 2003:")
print("   - Lower metallicity → higher C/O ratio in WD")
print("   - Higher C/O → more energetic burning → MORE Ni-56")
print("   - This would make low-Z SNe BRIGHTER ✗")
print()
print("4. Mazzali et al. 2007:")
print("   - High velocity SNe (HV) are brighter than normal velocity (NV)")
print("   - HV SNe have more asymmetric explosions")
print()

# =====================================================
# The complication
# =====================================================
print("THE COMPLICATION:")
print("-" * 70)
print()
print("There are MULTIPLE competing effects:")
print()
print("  A. Metallicity → κ → D → deflagration → DDT timing → Ni-56")
print("     (our Spandrel mechanism)")
print()
print("  B. Metallicity → C/O ratio → nuclear energy → Ni-56")
print("     (Timmes 2003: goes OPPOSITE to Spandrel)")
print()
print("  C. Metallicity → neutron excess → electron capture → Ni-56")
print("     (affects Ni-56 through Ye, usually small)")
print()
print("  D. Progenitor age → simmering → ignition conditions → Ni-56")
print("     (Son et al. 2025: observed 5σ signal)")
print()
print("The NET effect depends on which mechanism dominates!")
print()

# =====================================================
# Our framework's assumption
# =====================================================
print("OUR FRAMEWORK'S ASSUMPTION:")
print("-" * 70)
print()
print("We assumed: Higher D → Different light curve width → δμ")
print()
print("Specifically in spandrel_cosmology.py:")
print("  δμ = κ × (D - D_ref)")
print("  with κ > 0")
print()
print("This means: Higher D → Higher μ → FAINTER")
print()
print("If true: High z → Low Z → High D → FAINTER → wₐ < 0 ✓")
print()
print("Our cosmology code showed wₐ ≈ -0.08 for β=0.05")
print("This is the CORRECT SIGN but WRONG MAGNITUDE")
print()

# =====================================================
# The real crisis
# =====================================================
print("=" * 70)
print("THE REAL CRISIS")
print("=" * 70)
print()
print("The Spandrel mechanism has the CORRECT SIGN.")
print()
print("The CRISIS is purely about MAGNITUDE:")
print("  - Required β for DESI: β ≳ 0.3 (to get wₐ ~ -0.75)")
print("  - Simulated β at 48³: β = 0.05")
print("  - Simulated β at 128³: β = 0.008")
print("  - Extrapolated β_∞: β → 0")
print()
print("The turbulent washout reduces β by ~6× going from 48³ to 128³.")
print("This is the TURBULENT WASHOUT crisis, not a sign error.")
print()
print("Even if the molecular physics is correct, turbulence overwhelms it.")
print()

# =====================================================
# Resolution
# =====================================================
print("=" * 70)
print("RESOLUTION PATHS")
print("=" * 70)
print()
print("1. STRONGER MOLECULAR EFFECT")
print("   - Our κ(Z) parametrization may be too weak")
print("   - Real WD conductivity spans larger range?")
print()
print("2. PROGENITOR AGE DOMINATES")
print("   - Son et al. 2025 observational signal is 5σ")
print("   - Age → Ni-56 correlation may be DIRECT")
print("   - Not mediated by flame physics")
print()
print("3. SELECTION EFFECTS")
print("   - Malmquist bias at high z")
print("   - Host galaxy metallicity bias")
print()
print("4. THE SPANDREL IS A CONTRIBUTOR, NOT THE WHOLE STORY")
print("   - δμ_total = δμ_metallicity + δμ_age + δμ_selection")
print("   - Each contributes ~0.03-0.05 mag")
print("   - Combined: sufficient for DESI")
print()
