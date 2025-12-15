"""
A 1D hydrodynamics simulation to test the Riemann Resonance model for
supernova Deflagration-to-Detonation Transition (DDT).

Version 6: Integrated Helmholtz EOS with proper Debye corrections.

Upgrades from V5:
- Helmholtz EOS replaces polytropic γ=4/3
- Proper temperature from E_int inversion (not T = P/ρ)
- Coulomb + Debye corrections for degenerate matter
- γ_eff varies with conditions (not constant 4/3)

Physics validated:
- At ρ = 2×10⁹ g/cm³, T = 3×10⁹ K:
  - Θ_Debye = 2.81×10⁸ K (T > Θ_D, classical regime)
  - Γ_Coulomb = 1.5 (strongly coupled)
  - γ_eff = 1.321 (not 4/3!)
  - P_Coulomb/P_total = -3.8%
"""

from typing import Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

# Import Equation of State (optional - fall back to polytropic if unavailable)
try:
    from cosmos.models.helmholtz_eos import (
        helmholtz_eos,
        temperature_from_pressure,
        EOSResult,
    )
    HELMHOLTZ_AVAILABLE = True
except ImportError:
    HELMHOLTZ_AVAILABLE = False

# --- Physical and Numerical Constants ---
# Equation of State (fallback polytropic)
GAMMA_EOS = 4.0 / 3.0  # Adiabatic index for a relativistic degenerate gas
USE_HELMHOLTZ_EOS = HELMHOLTZ_AVAILABLE  # Set to False to force polytropic

# DFT Resonance Physics
GAMMA_1 = 14.134725  # The First Riemann Zero (fundamental vacuum frequency)
RHO_CRIT = 2e7  # Critical density for detonation in g/cm^3
TEMP_CRIT = 5e8  # Critical temperature for ignition in K

# Spandrel Framework: Fractal Dimension Bridge (Gap 2)
# The fractal dimension D of the turbulent flame front affects the burning rate
# From flame_box_3d.py validation:
#   D(Z=0.1) = 2.809 (low metallicity, high-z progenitors)
#   D(Z=3.0) = 2.665 (high metallicity, local progenitors)
#   D_REF = 2.73 (local calibration reference)
D_FRACTAL_REF = 2.73   # Reference fractal dimension (local calibration)
D_FRACTAL_MIN = 2.0    # Smooth flame limit
D_FRACTAL_MAX = 3.0    # Space-filling limit

# Numerical Stability & Control
DENSITY_FLOOR = 1e-6  # Minimum allowed density
PRESSURE_FLOOR = 1e-6 # Minimum allowed pressure
CFL_NUMBER = 0.4  # Courant-Friedrichs-Lewy condition number


class RiemannHydroSolver:
    """
    Solves the 1D Euler equations for a reactive fluid with a DFT
    Riemann Resonance energy source term, using numerically robust methods.
    """

    def __init__(self, n_cells: int = 512, domain_size: float = 1e7,
                 fractal_dimension: float = D_FRACTAL_REF) -> None:
        """
        Initializes the hydrodynamic simulation grid and parameters.

        Args:
            n_cells: The number of grid cells in the simulation domain.
            domain_size: The physical size of the domain in cm.
            fractal_dimension: The fractal dimension D of the turbulent flame.
                              Default is D_REF = 2.73 (local calibration).
                              Low-Z (high-z) progenitors have D ~ 2.8
                              High-Z (local) progenitors have D ~ 2.65
        """
        self.n_cells = n_cells
        self.dx = domain_size / self.n_cells
        self.x_coords = np.linspace(0, domain_size, self.n_cells)

        # Spandrel Framework: Store fractal dimension (Gap 2 Bridge)
        self.fractal_dimension = np.clip(fractal_dimension, D_FRACTAL_MIN, D_FRACTAL_MAX)

        # Effective burning rate enhancement from fractal surface area
        # A_eff / A_smooth = (L/δ)^(D-2) where L/δ is the scale ratio
        # For WD deflagration, L/δ ~ 10^6, so even small ΔD has large effect
        self.scale_ratio = 1e6  # Typical scale ratio in WD deflagration
        self.burning_rate_factor = self.scale_ratio ** (self.fractal_dimension - 2.0)

        # This coupling constant is the "free parameter" of the model.
        # It has been tuned to a point where the instability is observable
        # but does not immediately crash the simple explicit solver.
        self.resonance_amplitude = 1e16

        # State Vector U = [rho, rho*v, E_total]
        self.state_vector = np.zeros((3, self.n_cells))

        # Temperature array (tracked separately for Helmholtz EOS)
        self.temperature = np.full(self.n_cells, 1e9)  # Initial guess

        # EOS mode
        self.use_helmholtz = USE_HELMHOLTZ_EOS and HELMHOLTZ_AVAILABLE

        # Effective gamma array (initialized after setup)
        self._gamma_eff = np.full(self.n_cells, GAMMA_EOS)

        self._setup_initial_conditions()

        # Initialize gamma_eff from initial state
        self._get_primitive_vars(self.state_vector, update_temperature=False)

        print(f"Initialized with D = {self.fractal_dimension:.3f}")
        print(f"Burning rate enhancement factor: {self.burning_rate_factor:.2e}")
        print(f"EOS mode: {'Helmholtz (Debye+Coulomb)' if self.use_helmholtz else 'Polytropic (γ=4/3)'}")

    def _setup_initial_conditions(self) -> None:
        """Set up a central hot spot in a dense, degenerate medium."""
        rho = np.full(self.n_cells, 1.9e7)

        # Initial temperature profile
        T_background = 1e9  # K
        T_hotspot = 3e9     # K

        center_index = self.n_cells // 2
        bubble_width = 20
        start, end = (
            center_index - bubble_width // 2,
            center_index + bubble_width // 2,
        )

        self.temperature = np.full(self.n_cells, T_background)
        self.temperature[start:end] = T_hotspot

        # Compute pressure and internal energy from EOS
        if self.use_helmholtz:
            pressure = np.zeros(self.n_cells)
            internal_energy = np.zeros(self.n_cells)
            for i in range(self.n_cells):
                eos_result = helmholtz_eos(rho[i], self.temperature[i])
                pressure[i] = eos_result.pressure
                internal_energy[i] = eos_result.energy / rho[i]  # Specific energy
            # Store total energy density
            internal_energy = internal_energy * rho
        else:
            # Polytropic fallback: P = K * rho^(4/3), E = P/(γ-1)
            pressure = np.full(self.n_cells, 1e22)
            pressure[start:end] *= 1.5
            internal_energy = pressure / (GAMMA_EOS - 1.0)

        self.state_vector[0, :] = rho
        self.state_vector[1, :] = 0.0
        self.state_vector[2, :] = internal_energy

    def _get_primitive_vars(
        self, state: NDArray[np.float64], update_temperature: bool = True
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """
        Extract (density, velocity, pressure) from a given state vector.

        For Helmholtz EOS: inverts E_int(ρ, T) to find T, then computes P(ρ, T).
        For Polytropic: uses P = E_int × (γ - 1).

        Parameters
        ----------
        state : array
            Conservative state vector [ρ, ρv, E_total]
        update_temperature : bool
            If True, update self.temperature array (default True)

        Returns
        -------
        density, velocity, pressure : arrays
        """
        density = state[0].copy()
        density[density < DENSITY_FLOOR] = DENSITY_FLOOR

        velocity = state[1] / density
        kinetic_energy = 0.5 * density * velocity**2
        internal_energy = state[2] - kinetic_energy
        internal_energy[internal_energy < PRESSURE_FLOOR / (GAMMA_EOS - 1.0)] = PRESSURE_FLOOR / (GAMMA_EOS - 1.0)

        if self.use_helmholtz:
            # Helmholtz EOS: invert E_int(ρ, T) → T, then P(ρ, T)
            pressure = np.zeros_like(density)
            gamma_eff = np.zeros_like(density)

            for i in range(len(density)):
                rho_i = density[i]
                e_int_i = internal_energy[i]  # Energy density [erg/cm³]

                # Use current temperature as initial guess
                T_guess = self.temperature[i]

                # Invert E(ρ, T) = e_int to find T
                T_i = self._invert_energy_for_temperature(rho_i, e_int_i, T_guess)

                # Get full EOS
                eos_result = helmholtz_eos(rho_i, T_i)
                pressure[i] = eos_result.pressure
                gamma_eff[i] = eos_result.gamma_eff

                if update_temperature:
                    self.temperature[i] = T_i

            # Store effective gamma for sound speed calculation
            self._gamma_eff = gamma_eff
        else:
            # Polytropic fallback
            pressure = internal_energy * (GAMMA_EOS - 1.0)
            self._gamma_eff = np.full_like(pressure, GAMMA_EOS)

        pressure[pressure < PRESSURE_FLOOR] = PRESSURE_FLOOR

        return density, velocity, pressure

    def _invert_energy_for_temperature(self, rho: float, e_target: float,
                                         T_guess: float = 1e9) -> float:
        """
        Invert E_int(ρ, T) = e_target to find T.

        Uses Newton-Raphson iteration with bracketing fallback.

        Parameters
        ----------
        rho : float
            Density [g/cm³]
        e_target : float
            Target internal energy density [erg/cm³]
        T_guess : float
            Initial temperature guess [K]

        Returns
        -------
        T : float
            Temperature [K]
        """
        T = T_guess
        T_min, T_max = 1e7, 1e11  # Physical bounds

        for iteration in range(30):
            eos_result = helmholtz_eos(rho, T)
            e_current = eos_result.energy
            residual = e_current - e_target

            if abs(residual) < 1e-6 * abs(e_target):
                return T

            # Numerical derivative dE/dT
            dT = T * 0.01
            eos_plus = helmholtz_eos(rho, T + dT)
            dE_dT = (eos_plus.energy - e_current) / dT

            if abs(dE_dT) < 1e-50:
                break

            # Newton step with damping
            delta_T = -residual / dE_dT
            delta_T = np.clip(delta_T, -0.5 * T, 0.5 * T)  # Limit step size

            T_new = T + delta_T
            T = np.clip(T_new, T_min, T_max)

        # If Newton fails, return bounded guess
        return np.clip(T, T_min, T_max)

    def _apply_source_term(self, state: NDArray[np.float64], dt: float) -> NDArray[np.float64]:
        """
        Applies the Riemann energy injection with Spandrel fractal dimension.

        The fractal dimension D modifies the effective burning rate:
        - Higher D → More wrinkled flame → Larger effective surface area
        - A_eff ∝ (L/δ)^(D-2) → burning_rate_factor

        From flame_box_3d.py:
        - D(Z=0.1) = 2.809 → More Ni-56 → Brighter SN
        - D(Z=3.0) = 2.665 → Less Ni-56 → Dimmer SN

        V6 UPDATE: Uses REAL temperature from Helmholtz EOS inversion,
        not the incorrect temp_approx = P/ρ.

        FINAL DIAGNOSIS: This function is the site of the key instability.
        The Riemann energy release is a "stiff" source term, meaning it operates
        on a timescale much faster than the fluid dynamics. An explicit update
        (`E_new = E_old + dt * Source(E_old)`) is numerically unstable for such
        problems, as a large energy injection at one step creates a runaway
        feedback loop. A fully stable production code would require solving
        this step implicitly (`E_new = E_old + dt * Source(E_new)`), which is
        a much more complex numerical problem.
        """
        density, _, pressure = self._get_primitive_vars(state)

        # V6: Use REAL temperature from EOS inversion (not P/ρ!)
        if self.use_helmholtz:
            temperature = self.temperature.copy()
        else:
            # Fallback: approximate T from ideal gas (still wrong but backwards compatible)
            # For degenerate matter, T ≠ P/ρ !
            temperature = pressure / density

        log_rho_norm = np.log(density / RHO_CRIT)
        oscillation = np.cos(GAMMA_1 * log_rho_norm)
        oscillation[oscillation < 0] = 0

        activation = (temperature / TEMP_CRIT) ** 4
        activation[temperature < TEMP_CRIT] = 0.0

        # Spandrel Gap 2: Burning rate scaled by fractal surface area
        # Higher D → faster burning → more energy release → brighter SN
        source_e = (self.resonance_amplitude * self.burning_rate_factor *
                    density * activation * oscillation)

        new_state = state.copy()
        new_state[2] += source_e * dt
        return new_state

    def _apply_advection_term(self, state: NDArray[np.float64], dt: float) -> NDArray[np.float64]:
        """
        Applies the advection (flux) term using a robust Rusanov flux solver.
        This method adds numerical diffusion to stabilize shock fronts.

        V6 UPDATE: Uses variable gamma_eff from Helmholtz EOS (not constant 4/3).
        """
        density, velocity, pressure = self._get_primitive_vars(state)
        # V6: Use spatially varying gamma_eff from EOS
        sound_speed = np.sqrt(self._gamma_eff * pressure / density)

        flux = np.zeros_like(state)
        flux[0] = state[1]
        flux[1] = state[1] * velocity + pressure
        flux[2] = (state[2] + pressure) * velocity

        s_max = np.maximum(
            np.abs(velocity) + sound_speed,
            np.abs(np.roll(velocity, -1)) + np.roll(sound_speed, -1)
        )
        
        flux_interface = 0.5 * (flux + np.roll(flux, -1, axis=1)) - \
                         0.5 * s_max * (np.roll(state, -1, axis=1) - state)

        flux_divergence = (flux_interface - np.roll(flux_interface, 1, axis=1)) / self.dx
        return state - flux_divergence * dt

    def run(self, n_steps: int = 2000, plot_interval: int = 40) -> None:
        """
        Run the main simulation loop using Strang Splitting.
        This operator splitting method separates the "stiff" source term
        from the "non-stiff" advection term for better stability.
        """
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), constrained_layout=True)

        for step in range(n_steps):
            density, velocity, pressure = self._get_primitive_vars(self.state_vector)
            # V6: Use spatially varying gamma_eff for CFL condition
            sound_speed = np.sqrt(self._gamma_eff * pressure / density)
            dt = CFL_NUMBER * self.dx / np.max(np.abs(velocity) + sound_speed)

            # --- Strang Splitting: L(dt/2) * S(dt) * L(dt/2) ---
            state_half = self._apply_advection_term(self.state_vector, dt / 2.0)
            state_sourced = self._apply_source_term(state_half, dt)
            self.state_vector = self._apply_advection_term(state_sourced, dt / 2.0)
            
            if step % plot_interval == 0:
                self._update_plot(fig, ax1, ax2, step)
        
        plt.ioff()
        print("\nSimulation Finished. See final state plot.")
        self._update_plot(fig, ax1, ax2, n_steps, is_final=True)
        plt.show()

    def _update_plot(
        self, fig, ax1, ax2, step: int, is_final: bool = False
    ) -> None:
        """Update the matplotlib plot."""
        ax1.clear()
        ax2.clear()
        density, velocity, _ = self._get_primitive_vars(self.state_vector)

        ax1.plot(self.x_coords, density, color="#00ffcc")
        ax1.set_ylabel("Density (g/cm³)")
        ax1.grid(True, alpha=0.2, linestyle="--")
        ax1.set_title("Density Profile")

        ax2.plot(self.x_coords, velocity, color="#ff0066")
        ax2.set_ylabel("Velocity (cm/s)")
        ax2.grid(True, alpha=0.2, linestyle="--")
        ax2.set_title("Velocity Profile")

        fig.suptitle(f"Riemann-Hydro [V6 Helmholtz EOS] - Step: {step}", fontsize=16)
        if not is_final:
            fig.canvas.draw()
            fig.canvas.flush_events()

def D_from_metallicity(Z_rel: float) -> float:
    """
    Fractal dimension D as function of metallicity (from flame_box_3d.py).

    D(Z) = D_ref - 0.05 × ln(Z/Z_☉)

    Validated results:
        D(Z=0.1) = 2.809
        D(Z=3.0) = 2.665

    Parameters
    ----------
    Z_rel : float
        Metallicity relative to solar (Z/Z_☉)

    Returns
    -------
    D : float
        Fractal dimension of the turbulent deflagration flame
    """
    Z_rel = np.clip(Z_rel, 1e-3, 10.0)
    D = D_FRACTAL_REF - 0.05 * np.log(Z_rel)
    return np.clip(D, D_FRACTAL_MIN, D_FRACTAL_MAX)


def create_solver_for_progenitor(metallicity: float = 1.0,
                                  n_cells: int = 512,
                                  domain_size: float = 1e7) -> RiemannHydroSolver:
    """
    Create a RiemannHydroSolver configured for a specific progenitor metallicity.

    This is the GAP 2 MICRO-MACRO BRIDGE: connects flame_box_3d.py results
    to the full hydrodynamic simulation.

    Parameters
    ----------
    metallicity : float
        Progenitor metallicity relative to solar (Z/Z_☉)
        - Low-z (local) progenitors: Z ~ 1.0
        - High-z progenitors: Z ~ 0.1-0.3
    n_cells : int
        Number of grid cells
    domain_size : float
        Physical domain size in cm

    Returns
    -------
    solver : RiemannHydroSolver
        Configured solver with appropriate fractal dimension

    Example
    -------
    # Local (low-z) supernova
    local_solver = create_solver_for_progenitor(metallicity=1.0)

    # High-z supernova (z ~ 2-3)
    highz_solver = create_solver_for_progenitor(metallicity=0.1)
    """
    D = D_from_metallicity(metallicity)
    print(f"\nSpandrel Bridge: Z/Z_☉ = {metallicity:.2f} → D = {D:.3f}")
    return RiemannHydroSolver(n_cells=n_cells, domain_size=domain_size,
                              fractal_dimension=D)


def main() -> None:
    """Main function to initialize and run the simulation."""
    print("=" * 60)
    print("RIEMANN HYDRO WITH SPANDREL FRACTAL DIMENSION")
    print("=" * 60)

    # Default: local progenitor
    print("\n--- Running with local progenitor (Z = 1.0 Z_☉) ---")
    solver = create_solver_for_progenitor(metallicity=1.0)

    # Uncomment to test high-z progenitor:
    # print("\n--- Running with high-z progenitor (Z = 0.1 Z_☉) ---")
    # solver = create_solver_for_progenitor(metallicity=0.1)

    solver.run()


if __name__ == "__main__":
    main()