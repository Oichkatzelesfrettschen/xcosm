"""
Unit tests for Planck units system.

Tests cover:
1. Consistency of Planck unit definitions
2. Unit conversion round-trips
3. Dimensionless constant values
4. Physical scale relationships
"""

# Import from the package
import sys

import numpy as np
import pytest

sys.path.insert(0, "/Users/eirikr/1_Workspace/cosmos/src")

from xcosm.core.planck_units import (
    COSMOLOGY_PLANCK,
    DIMENSIONLESS,
    LENGTH_SCALES,
    MASS_SCALES,
    SI,
    H0_from_planck,
    H0_to_planck,
    Planck,
    from_planck_energy,
    from_planck_length,
    from_planck_mass,
    from_planck_time,
    natural_to_planck,
    planck_to_natural,
    to_planck_energy,
    to_planck_length,
    to_planck_mass,
    to_planck_time,
    validate_planck_units,
)


class TestPlanckUnitDefinitions:
    """Test consistency of Planck unit definitions."""

    def test_planck_length_formula(self):
        """ℓ_P = √(ℏG/c³)"""
        computed = np.sqrt(SI.hbar * SI.G / SI.c**3)
        assert abs(computed - Planck.length) / Planck.length < 1e-6

    def test_planck_time_formula(self):
        """t_P = √(ℏG/c⁵) = ℓ_P/c"""
        computed = np.sqrt(SI.hbar * SI.G / SI.c**5)
        assert abs(computed - Planck.time) / Planck.time < 1e-6

        # Also check ℓ_P/c
        computed2 = Planck.length / SI.c
        assert abs(computed2 - Planck.time) / Planck.time < 1e-6

    def test_planck_mass_formula(self):
        """M_P = √(ℏc/G)"""
        computed = np.sqrt(SI.hbar * SI.c / SI.G)
        assert abs(computed - Planck.mass) / Planck.mass < 1e-6

    def test_planck_temperature_formula(self):
        """T_P = √(ℏc⁵/(Gk_B²))"""
        computed = np.sqrt(SI.hbar * SI.c**5 / (SI.G * SI.k_B**2))
        assert abs(computed - Planck.temperature) / Planck.temperature < 1e-6

    def test_lm_consistency(self):
        """ℓ_P × M_P × c/ℏ = 1 (definition check)"""
        product = Planck.length * Planck.mass * SI.c / SI.hbar
        assert abs(product - 1.0) < 1e-6

    def test_G_from_planck(self):
        """G = ℓ_P²c³/ℏ"""
        computed = Planck.length**2 * SI.c**3 / SI.hbar
        assert abs(computed - SI.G) / SI.G < 1e-6

    def test_planck_energy(self):
        """E_P = M_P c²"""
        computed = Planck.mass * SI.c**2
        assert abs(computed - Planck.energy) / Planck.energy < 1e-6

    def test_planck_force(self):
        """F_P = c⁴/G"""
        computed = SI.c**4 / SI.G
        assert abs(computed - Planck.force) / Planck.force < 1e-6


class TestDimensionlessConstants:
    """Test dimensionless constant values."""

    def test_fine_structure_constant(self):
        """α ≈ 1/137.036"""
        assert abs(1 / DIMENSIONLESS.alpha - 137.036) < 0.001

    def test_strong_coupling(self):
        """α_s(M_Z) ≈ 0.118"""
        assert 0.11 < DIMENSIONLESS.alpha_s_MZ < 0.13

    def test_weinberg_angle(self):
        """sin²θ_W ≈ 0.231"""
        assert 0.22 < DIMENSIONLESS.sin2_theta_W < 0.24

    def test_electron_proton_ratio(self):
        """m_e/m_p ≈ 1/1836"""
        assert abs(DIMENSIONLESS.m_e_over_m_p - 1 / 1836.15) < 1e-5

    def test_hierarchy_ratio(self):
        """M_P/v_EW ~ 5×10¹⁶"""
        assert 1e16 < DIMENSIONLESS.hierarchy_ratio < 1e18

    def test_j3o_dimensions(self):
        """dim(J₃(O)) = 27"""
        assert DIMENSIONLESS.dim_J3O == 27

    def test_exceptional_dimensions(self):
        """Check exceptional group dimensions."""
        assert DIMENSIONLESS.dim_F4 == 52
        assert DIMENSIONLESS.dim_E6 == 78
        assert DIMENSIONLESS.dim_E7 == 133
        assert DIMENSIONLESS.dim_E8 == 248


class TestMassScales:
    """Test mass scale relationships."""

    def test_electron_mass_planck(self):
        """m_e/M_P ≈ 4.2×10⁻²³"""
        assert 1e-24 < MASS_SCALES.electron < 1e-21

    def test_proton_mass_planck(self):
        """m_p/M_P ≈ 7.7×10⁻²⁰"""
        assert 1e-21 < MASS_SCALES.proton < 1e-18

    def test_electron_proton_ratio(self):
        """m_e/m_p from mass scales"""
        ratio = MASS_SCALES.electron / MASS_SCALES.proton
        assert abs(ratio - DIMENSIONLESS.m_e_over_m_p) < 1e-5

    def test_higgs_vev(self):
        """v_EW = 246 GeV gives correct Planck ratio"""
        expected = 246 / Planck.mass_GeV
        assert abs(MASS_SCALES.v_EW - expected) / expected < 1e-6

    def test_mass_hierarchy(self):
        """m_e < m_μ < m_τ"""
        assert MASS_SCALES.electron < MASS_SCALES.muon < MASS_SCALES.tau

    def test_quark_hierarchy(self):
        """m_u < m_d < m_s < m_c < m_b < m_t"""
        assert MASS_SCALES.up < MASS_SCALES.down < MASS_SCALES.strange
        assert MASS_SCALES.strange < MASS_SCALES.charm < MASS_SCALES.bottom
        assert MASS_SCALES.bottom < MASS_SCALES.top


class TestUnitConversions:
    """Test unit conversion functions."""

    def test_mass_roundtrip_kg(self):
        """kg -> Planck -> kg roundtrip"""
        mass_kg = 1.0
        planck = to_planck_mass(mass_kg, "kg")
        back = from_planck_mass(planck, "kg")
        assert abs(back - mass_kg) / mass_kg < 1e-10

    def test_mass_roundtrip_gev(self):
        """GeV -> Planck -> GeV roundtrip"""
        mass_gev = 100.0
        planck = to_planck_mass(mass_gev, "GeV")
        back = from_planck_mass(planck, "GeV")
        assert abs(back - mass_gev) / mass_gev < 1e-10

    def test_mass_planck_mass_is_unity(self):
        """M_P in Planck units = 1"""
        planck_in_planck = to_planck_mass(Planck.mass, "kg")
        assert abs(planck_in_planck - 1.0) < 1e-6

    def test_length_roundtrip_m(self):
        """m -> Planck -> m roundtrip"""
        length_m = 1e-10
        planck = to_planck_length(length_m, "m")
        back = from_planck_length(planck, "m")
        assert abs(back - length_m) / length_m < 1e-10

    def test_length_roundtrip_mpc(self):
        """Mpc -> Planck -> Mpc roundtrip"""
        length_mpc = 100.0
        planck = to_planck_length(length_mpc, "Mpc")
        back = from_planck_length(planck, "Mpc")
        assert abs(back - length_mpc) / length_mpc < 1e-10

    def test_time_roundtrip_s(self):
        """s -> Planck -> s roundtrip"""
        time_s = 1.0
        planck = to_planck_time(time_s, "s")
        back = from_planck_time(planck, "s")
        assert abs(back - time_s) / time_s < 1e-10

    def test_time_roundtrip_gyr(self):
        """Gyr -> Planck -> Gyr roundtrip"""
        time_gyr = 13.8
        planck = to_planck_time(time_gyr, "Gyr")
        back = from_planck_time(planck, "Gyr")
        assert abs(back - time_gyr) / time_gyr < 1e-10

    def test_energy_roundtrip_gev(self):
        """GeV -> Planck -> GeV roundtrip"""
        energy_gev = 1000.0
        planck = to_planck_energy(energy_gev, "GeV")
        back = from_planck_energy(planck, "GeV")
        assert abs(back - energy_gev) / energy_gev < 1e-10


class TestNaturalPlanckConversions:
    """Test natural <-> Planck conversions."""

    def test_mass_natural_planck(self):
        """Mass conversion natural <-> Planck"""
        mass_gev = 100.0  # 100 GeV
        planck = natural_to_planck(mass_gev, "mass")
        back = planck_to_natural(planck, "mass")
        assert abs(back - mass_gev) / mass_gev < 1e-10

    def test_length_natural_planck(self):
        """Length conversion natural <-> Planck"""
        # In natural units, length ~ 1/GeV
        length_inv_gev = 1.0 / 100  # 1/(100 GeV)
        planck = natural_to_planck(length_inv_gev, "length")
        back = planck_to_natural(planck, "length")
        assert abs(back - length_inv_gev) / length_inv_gev < 1e-10


class TestHubbleConversions:
    """Test Hubble parameter conversions."""

    def test_H0_roundtrip(self):
        """H₀ conversion roundtrip"""
        H0_km_s_Mpc = 70.0
        planck = H0_to_planck(H0_km_s_Mpc)
        back = H0_from_planck(planck)
        assert abs(back - H0_km_s_Mpc) / H0_km_s_Mpc < 1e-6

    def test_H0_planck_order_of_magnitude(self):
        """H₀ in Planck units ~ 10⁻⁶¹"""
        H0_planck = H0_to_planck(70.0)
        assert 1e-62 < H0_planck < 1e-60


class TestCosmologyPlanck:
    """Test cosmological parameters in Planck units."""

    def test_hubble_length(self):
        """L_H = c/H₀ = 1/H₀ in Planck units"""
        assert abs(COSMOLOGY_PLANCK.L_H - 1 / COSMOLOGY_PLANCK.H0) < 1e-10

    def test_critical_density(self):
        """ρ_c = 3H₀²/(8π) in Planck units"""
        expected = 3 * COSMOLOGY_PLANCK.H0**2 / (8 * np.pi)
        assert abs(COSMOLOGY_PLANCK.rho_critical - expected) < 1e-100


class TestValidation:
    """Test the validation function."""

    def test_validate_all_pass(self):
        """All validation checks should pass."""
        results = validate_planck_units()
        for check, data in results.items():
            assert data["relative_error"] < 1e-6, f"{check} failed"


class TestPhysicalRelationships:
    """Test known physical relationships."""

    def test_compton_wavelength(self):
        """λ_C = ℏ/(mc) = M_P/m in Planck units"""
        # For electron: λ_C/ℓ_P = M_P/m_e
        compton_electron_planck = 1 / MASS_SCALES.electron
        # Should be very large (~10²²)
        assert compton_electron_planck > 1e20

    def test_schwarzschild_radius(self):
        """r_s = 2GM/c² = 2M in Planck units"""
        # For a Planck mass: r_s = 2ℓ_P
        # For proton: r_s/ℓ_P = 2 m_p/M_P
        r_s_proton_planck = 2 * MASS_SCALES.proton
        assert r_s_proton_planck < 1e-18  # Very small

    def test_gravitational_vs_em_coupling(self):
        """α_G << α"""
        assert DIMENSIONLESS.alpha_G_proton < DIMENSIONLESS.alpha * 1e-35


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
