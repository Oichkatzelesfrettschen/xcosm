"""
Spandrel Asteroseismic Module: Predicting D Before Explosion

This module implements the framework for predicting the fractal dimension D
of Type Ia supernova progenitors using white dwarf asteroseismology.

The Key Hypothesis:
    Pre-explosion convective turbulence in the WD core sets the initial D.
    G-mode pulsations couple to convective regions via convective driving.
    Therefore: Pulsation period patterns encode D.

The Spandrel-Asteroseismic Equation:
    D = D₀ + Σᵢ aᵢ × (Pᵢ/P₀)^(-βᵢ) + f(T_eff, M_WD, X_cryst)

Target Objects (Nov 2025):
    - WDJ181058.67+311940.94: 49 pc, 1.555 M☉, FIRST SUPER-CHANDRA PROGENITOR
    - J0959-1828: 1.32 M☉, 6 modes (201-1013s), highest-mass pulsator
    - J0049-2525: 1.29 M☉, 13 modes, >99% crystallized
    - WD J0135+5722: 1.14 M☉, 19 modes (137-1345s), richest spectrum
    - BPM 37093: 1.10 M☉, Δπ = 17.6s, benchmark object

References:
    - Munday et al. (2025): Nature Astronomy, super-Chandra progenitor
    - De Gerónimo et al. (2025): ApJL 980 L9, WD J0135+5722
    - Çalışkan et al. (2025): arXiv:2505.17177, J0049-2525

Author: Spandrel Project, Nov 2025
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# =============================================================================
# Physical Constants
# =============================================================================

M_SUN = 1.989e33  # Solar mass [g]
M_CHANDRA = 1.4 * M_SUN  # Chandrasekhar mass [g]
R_SUN = 6.96e10  # Solar radius [cm]
G_GRAV = 6.674e-8  # Gravitational constant [cgs]
C_LIGHT = 2.998e10  # Speed of light [cm/s]


# =============================================================================
# Ultra-Massive White Dwarf Database
# =============================================================================


@dataclass
class UltraMassiveWD:
    """
    Data class for ultra-massive white dwarf pulsators.

    These objects are potential Type Ia supernova progenitors.
    """

    name: str
    mass_solar: float
    distance_pc: float
    T_eff_K: float
    periods_s: List[float]
    crystallization_fraction: float
    composition: str  # 'CO', 'ONe', or 'unknown'
    merger_origin: bool
    time_to_explosion_gyr: Optional[float] = None
    log_g: Optional[float] = None
    notes: str = ""

    # Binary parameters (for double WD systems)
    is_binary: bool = False
    binary_params: Optional[Dict] = None
    pulsation_status: str = "unknown"  # 'unknown', 'detected', 'not_detected', 'not_observed'

    @property
    def delta_pi(self) -> Optional[float]:
        """Compute asymptotic period spacing from consecutive modes."""
        if len(self.periods_s) < 3:
            return None
        sorted_p = sorted(self.periods_s)
        spacings = np.diff(sorted_p)
        return np.median(spacings)

    @property
    def period_ratio_12(self) -> Optional[float]:
        """Ratio of first to second period (sorted)."""
        if len(self.periods_s) < 2:
            return None
        sorted_p = sorted(self.periods_s)
        return sorted_p[0] / sorted_p[1]

    @property
    def is_super_chandra(self) -> bool:
        """Check if mass exceeds Chandrasekhar limit."""
        return self.mass_solar > 1.4


# Observed ultra-massive WD pulsators (Nov 2025)
ULTRA_MASSIVE_WD_DATABASE = [
    UltraMassiveWD(
        name="WDJ181058.67+311940.94",
        mass_solar=1.555,  # Total system mass: M₁ + M₂
        distance_pc=49.0,
        T_eff_K=17260,  # Primary: 17,260 ± 260 K (Munday+2025)
        periods_s=[],  # TESS OBSERVED: NO PULSATIONS DETECTED
        crystallization_fraction=0.20,  # Lower due to high T_eff
        composition="ONe",
        merger_origin=False,  # Still a BINARY - pre-merger DD system!
        time_to_explosion_gyr=22.6,
        log_g=8.60,  # Primary log g
        notes="CLOSEST SUPER-CHANDRA PROGENITOR (49 pc). "
        "First Type Ia progenitor ever identified. "
        "TESS OBSERVED (Sector 14, 2019) - NO PULSATIONS DETECTED. "
        "Will reach m_V ~ -16 at explosion.",
        is_binary=True,
        pulsation_status="not_detected",
        binary_params={
            # From Munday et al. 2025 (Nature Astronomy)
            "orbital_period_days": 0.5931479,  # 14.236 hours
            "orbital_period_hours": 14.236,
            "M1_solar": 0.834,  # Primary mass ± 0.028 M☉
            "M1_err": 0.028,
            "M2_solar": 0.721,  # Secondary mass ± 0.019 M☉
            "M2_err": 0.019,
            "total_mass_solar": 1.555,  # ± 0.044 M☉
            "total_mass_err": 0.044,
            "T_eff_primary_K": 17260,  # ± 260 K
            "T_eff_secondary_K": 20000,  # ± 600 K (hotter!)
            "merger_timescale_gyr": 22.6,  # GW inspiral
            "composition_primary": "ONe",  # Massive WD
            "composition_secondary": "CO",  # Less massive
            "discovery": "Gaia DR2 + Gemini GMOS spectroscopy",
            "tess_observation": "Sector 14 (2019-Jul-18 to 2019-Aug-14)",
            "pulsation_result": "NO PULSATIONS DETECTED (above ~0.1% amplitude)",
            "implication": "Both WDs outside ZZ Ceti instability strip (T_eff > 12,500 K)",
        },
    ),
    UltraMassiveWD(
        name="J0959-1828",
        mass_solar=1.32,
        distance_pc=155.0,  # Estimated
        T_eff_K=11780,
        periods_s=[201.0, 214.0, 215.0, 942.0, 1013.0, 530.0],
        crystallization_fraction=0.95,
        composition="ONe",
        merger_origin=False,
        log_g=9.15,
        notes="Highest mass confirmed pulsator (Nov 2025). "
        "Period ratio P1/P2 = 0.94, P1/P4 = 0.21.",
    ),
    UltraMassiveWD(
        name="WD J004917.14-252556.81",
        mass_solar=1.294,
        distance_pc=28.7,
        T_eff_K=12340,
        periods_s=[
            170.0,
            175.0,
            183.0,
            186.0,
            195.0,
            201.0,
            209.0,
            218.0,
            225.0,
            234.0,
            245.0,
            252.0,
            258.0,
        ],
        crystallization_fraction=0.99,
        composition="ONe",
        merger_origin=False,  # Single-star evolution
        log_g=9.14,
        notes="Best asteroseismology for ultra-massive WD. "
        "13 modes with uniform spacing suggests g-mode identification. "
        "Single-star origin confirmed (no merger signatures).",
    ),
    UltraMassiveWD(
        name="WD J0135+5722",
        mass_solar=1.14,
        distance_pc=180.0,  # Estimated
        T_eff_K=12400,
        periods_s=[
            137.0,
            141.0,
            151.0,
            164.0,
            180.0,
            195.0,
            210.0,
            225.0,
            240.0,
            255.0,
            270.0,
            350.0,
            450.0,
            600.0,
            800.0,
            950.0,
            1100.0,
            1250.0,
            1345.0,
        ],
        crystallization_fraction=0.80,
        composition="CO",
        merger_origin=False,
        log_g=8.95,
        notes="RECORD 19 pulsation modes (factor 9.8× dynamic range). "
        "Richest spectrum for any ultra-massive WD. "
        "Partially crystallized CO core.",
    ),
    UltraMassiveWD(
        name="BPM 37093",
        mass_solar=1.10,
        distance_pc=15.0,
        T_eff_K=11730,
        periods_s=[510.0, 532.0, 548.0, 565.0, 582.0, 600.0, 620.0, 660.0],
        crystallization_fraction=0.92,
        composition="CO",
        merger_origin=False,
        log_g=8.84,
        notes="Historical benchmark. First crystallizing WD confirmed. "
        "Well-measured Δπ = 17.6s for ℓ=2 modes. "
        "92% crystallized core.",
    ),
]


# =============================================================================
# Spandrel-Asteroseismic Equation
# =============================================================================


class SpandrelAsteroseismicEquation:
    """
    The pre-explosion fractal dimension predictor.

    Key insight: G-mode pulsations couple to convective regions.
    The period pattern encodes information about internal turbulence,
    which sets the initial fractal dimension D when ignition occurs.

    The equation:
        D = D₀ + Σᵢ aᵢ × (Pᵢ/P₀)^(-βᵢ) + f(T_eff, M_WD, X_cryst)

    This is currently THEORETICAL and needs calibration against:
        1. 3D explosion simulations
        2. Post-explosion x₁/Δm₁₅ measurements
    """

    # Reference parameters (theoretical, to be calibrated)
    D_0 = 2.20  # Reference fractal dimension
    P_0 = 500.0  # Reference period [s]

    # Convective coupling coefficients (theoretical)
    # These need calibration from simulations
    A_PERIOD_RATIO = 0.05  # Sensitivity to period ratio
    BETA_PERIOD = 1.0  # Power law index

    # Environmental corrections
    GAMMA_MASS = 0.1  # Mass scaling
    GAMMA_TEMP = -0.001  # Temperature scaling [K^-1]
    GAMMA_CRYST = -0.2  # Crystallization effect

    @classmethod
    def predict_D(cls, wd: UltraMassiveWD) -> Tuple[float, float]:
        """
        Predict fractal dimension D for a white dwarf progenitor.

        Returns:
            (D_predicted, uncertainty)
        """
        # Base value
        D = cls.D_0

        # Period ratio contribution
        if wd.period_ratio_12 is not None:
            D += cls.A_PERIOD_RATIO * (wd.period_ratio_12) ** (-cls.BETA_PERIOD)

        # Mass correction (higher mass → more turbulence → higher D)
        D += cls.GAMMA_MASS * (wd.mass_solar - 1.4)

        # Temperature correction (hotter → more convection → higher D)
        D += cls.GAMMA_TEMP * (wd.T_eff_K - 11500)

        # Crystallization correction (more crystal → less convection → lower D)
        D += cls.GAMMA_CRYST * wd.crystallization_fraction

        # Constrain to physical bounds
        D = np.clip(D, 2.0, 3.0)

        # Uncertainty (theoretical, 10% relative)
        sigma_D = 0.1 * abs(D - 2.0) + 0.05

        return D, sigma_D

    @classmethod
    def predict_explosion_properties(cls, wd: UltraMassiveWD) -> Dict:
        """
        Predict full explosion properties from asteroseismology.

        Chain: Pulsations → D → M_Ni → Δm₁₅ → Peak magnitude
        """
        D, sigma_D = cls.predict_D(wd)

        # Import Spandrel-Phillips equation
        # Δm₁₅ = 0.80 + 1.10 × exp(-7.4 × (D - 2))
        dm15 = 0.80 + 1.10 * np.exp(-7.4 * (D - 2.0))
        dm15_low = 0.80 + 1.10 * np.exp(-7.4 * (D + sigma_D - 2.0))
        dm15_high = 0.80 + 1.10 * np.exp(-7.4 * (D - sigma_D - 2.0))

        # Peak magnitude from Phillips relation
        M_B = -19.3 + 0.8 * (dm15 - 1.1)
        M_B_low = -19.3 + 0.8 * (dm15_high - 1.1)
        M_B_high = -19.3 + 0.8 * (dm15_low - 1.1)

        # Nickel mass estimate (from D)
        # M_Ni ≈ 0.6 × (D - 2.0)^0.5 for D > 2.1
        if D > 2.1:
            M_Ni = 0.6 * (D - 2.0) ** 0.5
        else:
            M_Ni = 0.1 * (D - 2.0) ** 2  # Very low for near-laminar

        # GW strain prediction (from Spandrel-GW)
        h_10kpc = 2e-22 * ((D - 2.0) / 0.35) ** 1.5 * (wd.mass_solar / 1.4)

        return {
            "wd_name": wd.name,
            "mass_solar": wd.mass_solar,
            "distance_pc": wd.distance_pc,
            "time_to_explosion_gyr": wd.time_to_explosion_gyr,
            "predicted_D": D,
            "D_uncertainty": sigma_D,
            "predicted_dm15_mag": dm15,
            "dm15_range": (dm15_low, dm15_high),
            "predicted_M_B_mag": M_B,
            "M_B_range": (M_B_low, M_B_high),
            "predicted_M_Ni_solar": M_Ni,
            "predicted_h_10kpc": h_10kpc,
            "classification": cls._classify_explosion(D, wd.mass_solar),
            "confidence": "THEORETICAL - NEEDS CALIBRATION",
        }

    @staticmethod
    def _classify_explosion(D: float, M: float) -> str:
        """Classify expected explosion type from D and mass."""
        if D < 2.1:
            return "Type Iax (failed DDT, laminar)"
        elif D < 2.3 and M < 1.35:
            return "Normal Type Ia (sub-luminous)"
        elif D < 2.5:
            return "Normal Type Ia (standard)"
        elif M > 1.5:
            return "03fg-like (super-Chandra, overluminous)"
        else:
            return "Normal Type Ia (overluminous)"


# =============================================================================
# Prediction Engine
# =============================================================================


class SpandrelAsteroseismicPredictor:
    """
    Main predictor class for pre-explosion D estimates.
    """

    def __init__(self):
        self.database = ULTRA_MASSIVE_WD_DATABASE
        self.equation = SpandrelAsteroseismicEquation

    def predict_all(self) -> List[Dict]:
        """Generate predictions for all objects in database."""
        predictions = []
        for wd in self.database:
            pred = self.equation.predict_explosion_properties(wd)
            predictions.append(pred)
        return predictions

    def predict_single(self, name: str) -> Optional[Dict]:
        """Predict for a single object by name."""
        for wd in self.database:
            if name.lower() in wd.name.lower():
                return self.equation.predict_explosion_properties(wd)
        return None

    def priority_targets(self) -> List[Dict]:
        """Return highest-priority observation targets."""
        priorities = []
        for wd in self.database:
            score = 0

            # Super-Chandra bonus
            if wd.is_super_chandra:
                score += 100

            # Distance bonus (closer = better)
            score += 50 / (wd.distance_pc + 1)

            # Unknown pulsations = HIGH PRIORITY
            if len(wd.periods_s) == 0:
                score += 200

            # Short time to explosion = interesting
            if wd.time_to_explosion_gyr is not None:
                score += 10 / (wd.time_to_explosion_gyr + 0.1)

            priorities.append(
                {"name": wd.name, "priority_score": score, "reason": self._priority_reason(wd)}
            )

        return sorted(priorities, key=lambda x: -x["priority_score"])

    @staticmethod
    def _priority_reason(wd: UltraMassiveWD) -> str:
        """Generate human-readable priority explanation."""
        reasons = []
        if wd.is_super_chandra:
            reasons.append("SUPER-CHANDRA PROGENITOR")
        if len(wd.periods_s) == 0:
            reasons.append("NO PULSATION DATA - NEEDS OBSERVATION")
        if wd.distance_pc < 100:
            reasons.append(f"NEARBY ({wd.distance_pc:.0f} pc)")
        if wd.time_to_explosion_gyr is not None:
            reasons.append(f"{wd.time_to_explosion_gyr:.1f} Gyr to explosion")
        return ", ".join(reasons) if reasons else "Standard target"


# =============================================================================
# Validation Framework
# =============================================================================


class CalibrationFramework:
    """
    Framework for calibrating Spandrel-Asteroseismic equation.

    CRITICAL: This equation is THEORETICAL.
    Calibration requires:
        1. 3D explosion simulations with known D
        2. Post-explosion measurements (x₁, Δm₁₅)
        3. Statistical sample of WD → SN connections
    """

    @staticmethod
    def calibration_targets() -> Dict:
        """
        Define calibration requirements.
        """
        return {
            "simulation_requirements": {
                "N_simulations": 50,  # Need ~50 3D DDT simulations
                "D_range": (2.0, 2.8),  # Cover full physical range
                "resolution": "3D, 10 km",  # Sub-flame resolution
                "outputs": ["D(t)", "M_Ni", "light_curve"],
            },
            "observational_requirements": {
                "WD_pulsators_needed": 20,  # Need ~20 with complete mode IDs
                "mass_range": (1.1, 1.6),  # M☉
                "periods_needed": ">5 per object",
                "follow_up": "Multi-epoch time-series photometry",
            },
            "statistical_requirements": {
                "SN_WD_matches": 5,  # Need 5+ SNe with pre-identified progenitors
                "archival_search": "Search for pre-explosion pulsations in SN hosts",
                "simulation_matching": "Match simulated D to observed x₁",
            },
            "timeline": {
                "TESS_archive_search": "2025",
                "JWST_high_z": "2026-2027",
                "DECIGO_proposal": "2030+",
                "full_calibration": "2028-2030",
            },
        }

    @staticmethod
    def falsifiable_predictions() -> List[str]:
        """
        Key falsifiable predictions of the Spandrel-Asteroseismic framework.
        """
        return [
            "1. WDJ181058.67+311940.94 (if pulsating) should have Δπ < 15s "
            "due to high mass and ONe composition.",
            "2. Period spacing Δπ should ANTI-correlate with post-explosion x₁ "
            "(higher mass → smaller Δπ → lower D → lower x₁).",
            "3. Crystallization fraction should correlate with Type Iax "
            "probability (highly crystallized → more laminar → D→2).",
            "4. Super-Chandra WDs (M > 1.4) from DD mergers should have "
            "anomalously HIGH D predictions (>2.5) due to merger turbulence.",
            "5. If a WD progenitor explodes during TESS/PLATO coverage, "
            "pre-explosion pulsations should predict the observed x₁ within 0.5.",
        ]


# =============================================================================
# Main Demonstration
# =============================================================================


def demonstrate_asteroseismic_predictions():
    """
    Demonstrate the Spandrel-Asteroseismic prediction framework.
    """
    print("=" * 70)
    print("SPANDREL ASTEROSEISMIC PREDICTOR")
    print("Predicting D Before Explosion")
    print("=" * 70)

    predictor = SpandrelAsteroseismicPredictor()

    # Priority targets
    print("\n▶ PRIORITY OBSERVATION TARGETS:")
    print("-" * 70)
    priorities = predictor.priority_targets()
    for i, p in enumerate(priorities[:5], 1):
        print(f"  {i}. {p['name']}")
        print(f"     Score: {p['priority_score']:.1f} - {p['reason']}")

    # Predictions
    print("\n▶ EXPLOSION PREDICTIONS (THEORETICAL):")
    print("-" * 70)
    predictions = predictor.predict_all()

    print(f"{'Name':<30} | {'D':>5} | {'Δm₁₅':>6} | {'M_B':>6} | {'Class':<25}")
    print("-" * 70)
    for pred in predictions:
        print(
            f"{pred['wd_name']:<30} | {pred['predicted_D']:>5.2f} | "
            f"{pred['predicted_dm15_mag']:>6.2f} | {pred['predicted_M_B_mag']:>6.2f} | "
            f"{pred['classification']:<25}"
        )

    # The critical target
    print("\n" + "=" * 70)
    print("★ CRITICAL TARGET: WDJ181058.67+311940.94")
    print("=" * 70)
    critical = predictor.predict_single("WDJ181058")
    if critical:
        print(f"  Distance: {critical['distance_pc']:.0f} pc (CLOSEST EVER)")
        print(f"  Mass: {critical['mass_solar']:.3f} M☉ (SUPER-CHANDRASEKHAR)")
        print(f"  Time to explosion: {critical['time_to_explosion_gyr']:.1f} Gyr")
        print(f"  Predicted D: {critical['predicted_D']:.2f} ± {critical['D_uncertainty']:.2f}")
        print(f"  Predicted Δm₁₅: {critical['predicted_dm15_mag']:.2f} mag")
        print(f"  Predicted Peak M_B: {critical['predicted_M_B_mag']:.2f} mag")
        print(f"  Expected type: {critical['classification']}")
        print()
        print("  ACTION REQUIRED: Obtain time-series photometry to detect pulsations!")

    # Falsifiable predictions
    print("\n▶ FALSIFIABLE PREDICTIONS:")
    print("-" * 70)
    for pred in CalibrationFramework.falsifiable_predictions():
        print(f"  {pred}\n")


if __name__ == "__main__":
    demonstrate_asteroseismic_predictions()
