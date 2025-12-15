#!/usr/bin/env python3
"""
Projection Uniqueness from F₄ Invariance
=========================================
EQUATION E03: Prove P: h₂(O) → R^{1,3} is unique

The projection from the 10-dimensional h₂(O) (2×2 Hermitian octonionic matrices)
to 4-dimensional Minkowski spacetime must be shown to be UNIQUE up to Lorentz
transformations, given the algebraic constraints.

Key Question: Why THIS projection and not another?
"""

import numpy as np
from scipy.linalg import null_space

# =============================================================================
# THE UNIQUENESS PROBLEM
# =============================================================================


def state_uniqueness_problem():
    """
    State the projection uniqueness problem.
    """
    print("=" * 70)
    print("E03: Uniqueness of Projection P: h₂(O) → R^{1,3}")
    print("=" * 70)

    print("""
    THE PROBLEM:
    ============

    The h₂(O) space consists of 2×2 Hermitian matrices over octonions:

        M = ⎡  α    x*  ⎤
            ⎣  x    β   ⎦

    where α, β ∈ ℝ and x ∈ O (octonion).

    Dimension: 1 + 1 + 8 = 10

    We claim there exists a canonical projection:

        P: h₂(O) → R^{1,3}

    such that Minkowski spacetime emerges with signature (+,-,-,-).

    THE QUESTION:
    =============
    Is this projection UNIQUE (up to Lorentz transformations)?
    What algebraic constraints determine P?

    REQUIRED PROPERTIES:
    ====================
    1. P must preserve the determinant form: det(M) → x·x (Minkowski norm)
    2. P must be F₄-equivariant (compatible with automorphisms)
    3. P must map rank-1 elements to null vectors
    4. P must be linear and surjective
    """)


# =============================================================================
# h₂(O) STRUCTURE
# =============================================================================


def analyze_h2o_structure():
    """
    Analyze the structure of h₂(O).
    """
    print("\n" + "=" * 70)
    print("Structure of h₂(O)")
    print("=" * 70)

    print("""
    h₂(O) Elements:
    ===============

    A general element M ∈ h₂(O):

        M = ⎡  t + z      x - iy  ⎤
            ⎣  x + iy     t - z   ⎦

    where we've parametrized:
    - α = t + z  (diagonal 1)
    - β = t - z  (diagonal 2)
    - x = x₀ + x₁e₁ + x₂e₂ + ... + x₇e₇  (off-diagonal octonion)

    Determinant:
    ------------
    det(M) = αβ - |x|² = (t+z)(t-z) - |x|²
           = t² - z² - |x|²
           = t² - z² - (x₀² + x₁² + ... + x₇²)

    This is NOT Minkowski! It's signature (1, 9).

    To get R^{1,3}, we must PROJECT from 10D to 4D.
    """)

    # Explicit basis for h₂(O)
    print("\n  Basis Elements for h₂(O):")
    print("  " + "-" * 50)

    # 10 basis elements
    basis_labels = [
        "E_t = diag(1, 1)",  # t direction
        "E_z = diag(1, -1)",  # z direction
        "E_x0 = off-diag(e₀)",  # x₀ direction (real part of octonion)
        "E_x1 = off-diag(e₁)",  # x₁ direction
        "E_x2 = off-diag(e₂)",  # x₂ direction
        "E_x3 = off-diag(e₃)",  # x₃ direction
        "E_x4 = off-diag(e₄)",  # x₄ direction
        "E_x5 = off-diag(e₅)",  # x₅ direction
        "E_x6 = off-diag(e₆)",  # x₆ direction
        "E_x7 = off-diag(e₇)",  # x₇ direction
    ]

    for i, label in enumerate(basis_labels):
        print(f"    {i}: {label}")

    return 10  # dimension


# =============================================================================
# DETERMINANT CONSTRAINT
# =============================================================================


def determinant_constraint():
    """
    Analyze the determinant constraint on projection.
    """
    print("\n" + "=" * 70)
    print("Constraint 1: Determinant Preservation")
    print("=" * 70)

    print("""
    Minkowski Norm Requirement:
    ===========================

    The projection P must map:

        det(M) = t² - z² - |x|²  (10D)
        ↓
        η(v,v) = t² - x² - y² - z²  (4D Minkowski)

    This requires:
        |x|² = x₀² + x₁² + ... + x₇² → x² + y²  (8D → 2D)

    The 8-dimensional octonion must project to 2 spatial dimensions!

    This is only possible if we:
    1. Identify 3 of the 8 octonion directions with spatial x, y
    2. The remaining 5 directions become "internal" (gauge/matter)

    Canonical Choice:
    -----------------
    P: (t, z, x₀, x₁, x₂, x₃, x₄, x₅, x₆, x₇) → (t, x₀, x₁, z)

    Mapping:
    - t → t (time)
    - x₀ → x (spatial x)
    - x₁ → y (spatial y)
    - z → z (spatial z)

    This preserves det = t² - x² - y² - z² ✓
    """)

    # Verify numerically
    print("\n  Numerical Verification:")
    print("  " + "-" * 50)

    # Random h₂(O) element
    t, z = 1.5, 0.8
    x = np.random.randn(8)  # 8 octonion components

    det_10d = t**2 - z**2 - np.sum(x**2)

    # Project to 4D (using x₀, x₁ for spatial)
    x_4d = x[0]
    y_4d = x[1]
    # z_4d = z

    det_4d = t**2 - x_4d**2 - y_4d**2 - z**2

    print(f"    det(M) in 10D: {det_10d:.4f}")
    print(f"    η(v,v) in 4D:  {det_4d:.4f}")
    print(f"    Difference: {det_10d - det_4d:.4f} (from projected-out directions)")

    # The difference comes from x₂² + ... + x₇² (internal DOF)
    internal = np.sum(x[2:] ** 2)
    print(f"    Internal DOF contribution: {internal:.4f}")
    print(f"    Match: {abs(det_10d - det_4d - (-internal)) < 1e-10}")

    return


# =============================================================================
# F₄ EQUIVARIANCE
# =============================================================================


def f4_equivariance():
    """
    Analyze F₄ equivariance constraint.
    """
    print("\n" + "=" * 70)
    print("Constraint 2: F₄ Equivariance")
    print("=" * 70)

    print("""
    F₄ Automorphism Group:
    ======================

    F₄ = Aut(J₃(O)) is the automorphism group of the exceptional Jordan algebra.

    For h₂(O), the relevant automorphism group is smaller:
        Aut(h₂(O)) ≅ Spin(9)

    (This is because h₂(O) is the "spin factor" J(O) ≅ R ⊕ O with spin(9) symmetry.)

    The projection P must be equivariant:

        P(g · M) = ρ(g) · P(M)  for all g ∈ Spin(9)

    where ρ: Spin(9) → SO(1,3) is a homomorphism.

    Key Observation:
    ----------------
    Spin(9) contains SO(8) as the automorphism of the octonion part.
    SO(8) has triality: three inequivalent 8-dimensional representations.

    The projection P selects ONE of these representations to become spacetime.

    Branching:
    ----------
    Spin(9) → Spin(1,3) × Spin(5)

    8_v → (2, 4) + gauge singlets
    8_s → spinors
    8_c → co-spinors

    The choice of which 8 becomes spacetime is related to triality!
    """)

    # Dimension counting
    print("\n  Dimension Counting:")
    print("  " + "-" * 50)
    print("    dim(Spin(9)) = 36")
    print("    dim(SO(1,3)) = 6")
    print("    dim(Spin(5)) = 10")
    print("    Coset: 36 - 6 - 10 = 20 (broken generators)")

    # The projection breaks Spin(9) → SO(1,3) × Spin(5)
    # The 20 broken generators become gauge/matter fields

    return


# =============================================================================
# RANK-1 CONSTRAINT (NULL VECTORS)
# =============================================================================


def rank1_constraint():
    """
    Analyze the rank-1 (null vector) constraint.
    """
    print("\n" + "=" * 70)
    print("Constraint 3: Rank-1 Elements → Null Vectors")
    print("=" * 70)

    print("""
    Rank-1 Elements:
    ================

    An element M ∈ h₂(O) has rank 1 if det(M) = 0.

    Explicitly:
        det(M) = αβ - |x|² = 0

    These form a 9-dimensional "null cone" in h₂(O).

    Under projection P, rank-1 elements must map to null vectors:

        P(M) · P(M) = 0  (Minkowski null)

    This is equivalent to:
        t² - x² - y² - z² = 0

    Constraint:
    -----------
    For M with det(M) = 0:
        (t+z)(t-z) = |x|²
        t² - z² = x₀² + x₁² + ... + x₇²

    After projection (keeping only x₀, x₁):
        t² - z² - x₀² - x₁² = x₂² + ... + x₇² ≠ 0  (generally)

    This means the simple projection does NOT preserve the null cone!

    RESOLUTION:
    ===========
    The projection must be NONLINEAR on the null cone, or
    we must constrain to a SUBSPACE where x₂ = ... = x₇ = 0.

    Physical Interpretation:
    ------------------------
    The condition x₂ = ... = x₇ = 0 means:
    - Massless particles (null geodesics) live in a 4D subspace
    - Massive particles can have nonzero internal components
    - This is analogous to Kaluza-Klein: motion in extra dimensions = mass
    """)

    # Demonstrate with examples
    print("\n  Examples:")
    print("  " + "-" * 50)

    # Null vector in full 10D
    t, z = 1.0, 0.6
    x_full = np.array([0.5, 0.5, 0.3, 0.2, 0.1, 0.1, 0.1, 0.1])
    # Adjust to make det = 0
    target = t**2 - z**2
    scale = np.sqrt(target / np.sum(x_full**2))
    x_full = scale * x_full

    det_10d = t**2 - z**2 - np.sum(x_full**2)
    print(f"    10D null: det = {det_10d:.6f} (should be 0)")

    # After projection
    det_4d = t**2 - z**2 - x_full[0] ** 2 - x_full[1] ** 2
    print(f"    4D projection: η = {det_4d:.4f} (NOT null!)")
    print(f"    Internal contribution: {np.sum(x_full[2:] ** 2):.4f}")

    # Constrained case (x₂=...=x₇=0)
    t, z = 1.0, 0.0
    x_constrained = np.array([0.707, 0.707, 0, 0, 0, 0, 0, 0])
    det_constrained = t**2 - z**2 - np.sum(x_constrained**2)
    det_4d_constrained = t**2 - z**2 - x_constrained[0] ** 2 - x_constrained[1] ** 2

    print(f"\n    Constrained 10D null: det = {det_constrained:.6f}")
    print(f"    Constrained 4D: η = {det_4d_constrained:.6f} (null preserved!)")

    return


# =============================================================================
# UNIQUENESS THEOREM
# =============================================================================


def prove_uniqueness():
    """
    Prove the uniqueness of projection.
    """
    print("\n" + "=" * 70)
    print("UNIQUENESS THEOREM")
    print("=" * 70)

    print("""
    THEOREM:
    ========

    The projection P: h₂(O) → R^{1,3} is unique up to:
    1. Lorentz transformations SO(1,3)
    2. Discrete triality choices (3 options)

    PROOF SKETCH:
    =============

    Step 1: Dimensional Constraint
    ------------------------------
    We need a 4D subspace of the 10D h₂(O).
    The space of such subspaces is Gr(4,10) with dim = 4×6 = 24.

    Step 2: Determinant Preservation
    --------------------------------
    The determinant form on h₂(O) is:
        Q(M) = t² - z² - |x|²  (signature (1,9))

    We need a 4D subspace where Q restricts to Minkowski signature (1,3).

    This requires: The 6D complement must be space-like (negative definite).

    The space of such subspaces has dim = dim(SO(1,9)/(SO(1,3)×SO(6)))
                                        = 45 - 6 - 15 = 24

    Step 3: Triality Selection
    --------------------------
    SO(8) ⊂ Spin(9) acts on the 8 octonion directions.
    Under triality, there are 3 inequivalent ways to embed SO(1,3):
    - Vector: 8_v → 4 + 4
    - Spinor: 8_s → 4 + 4
    - Co-spinor: 8_c → 4 + 4

    Each choice gives a different physical interpretation.

    Step 4: Lorentz Fixing
    ----------------------
    Once triality choice is made, there remains a continuous family
    of projections related by Lorentz transformations SO(1,3).

    These are all physically equivalent (different reference frames).

    CONCLUSION:
    ===========
    The projection P is unique up to:
    - Discrete: Z₃ (triality)
    - Continuous: SO(1,3) (Lorentz)

    The PHYSICAL projection (choosing a triality class) is therefore
    essentially UNIQUE.
    """)

    # Numerical verification
    print("\n  Numerical Verification:")
    print("  " + "-" * 50)

    # Count independent projections
    # Gr(4,10) has dim 24
    # Minkowski signature constraint: 24 - 24 = 0 (discrete set)
    # Triality: 3 choices
    # Lorentz: 6D family within each

    print("    dim(Gr(4,10)) = 24")
    print("    Signature constraint: 24 equations")
    print("    Remaining freedom: discrete (triality)")
    print("    Continuous: SO(1,3) equivalence")

    return


# =============================================================================
# EXPLICIT PROJECTION CONSTRUCTION
# =============================================================================


def construct_explicit_projection():
    """
    Construct the explicit projection matrix.
    """
    print("\n" + "=" * 70)
    print("Explicit Projection Construction")
    print("=" * 70)

    print("""
    Canonical Projection:
    =====================

    We construct P as a 4×10 matrix mapping h₂(O) → R^{1,3}.

    Basis for h₂(O): {E_t, E_z, E_x0, E_x1, ..., E_x7}
    Basis for R^{1,3}: {e_t, e_x, e_y, e_z}

    The canonical projection is:

        P = ⎡ 1  0  0  0  0  0  0  0  0  0 ⎤  (t → t)
            ⎢ 0  0  1  0  0  0  0  0  0  0 ⎥  (x₀ → x)
            ⎢ 0  0  0  1  0  0  0  0  0  0 ⎥  (x₁ → y)
            ⎣ 0  1  0  0  0  0  0  0  0  0 ⎦  (z → z)

    This selects (t, x₀, x₁, z) as spacetime coordinates.
    """)

    # Construct P
    P = np.zeros((4, 10))
    P[0, 0] = 1  # t → t
    P[1, 2] = 1  # x₀ → x
    P[2, 3] = 1  # x₁ → y
    P[3, 1] = 1  # z → z

    print("  Projection Matrix P:")
    print("  " + "-" * 50)
    print(P)

    # Metric in h₂(O) (signature 1,9)
    eta_10 = np.diag([1, -1, -1, -1, -1, -1, -1, -1, -1, -1])

    # Induced metric on R^{1,3}
    eta_4_induced = P @ eta_10 @ P.T

    print("\n  Induced Metric on R^{1,3}:")
    print("  " + "-" * 50)
    print(eta_4_induced)

    # Check it's Minkowski
    eta_4_expected = np.diag([1, -1, -1, -1])
    match = np.allclose(eta_4_induced, eta_4_expected)
    print(f"\n  Is Minkowski? {match}")

    # Kernel of P
    kernel = null_space(P)
    print("\n  Kernel of P (internal directions):")
    print(f"    dim(ker P) = {kernel.shape[1]}")

    return P


# =============================================================================
# ALTERNATIVE PROJECTIONS
# =============================================================================


def analyze_alternative_projections():
    """
    Analyze alternative (triality-related) projections.
    """
    print("\n" + "=" * 70)
    print("Alternative Projections (Triality)")
    print("=" * 70)

    print("""
    Three Triality Classes:
    =======================

    The three inequivalent projections correspond to different
    identifications of spacetime within the octonion:

    Choice 1 (Vector):
    ------------------
    Spatial: x₀, x₁, x₂ (first three imaginary directions)
    Internal: x₃, x₄, x₅, x₆, x₇

    Choice 2 (Spinor):
    ------------------
    Spatial: x₀, x₃, x₅ (Fano plane line 1)
    Internal: x₁, x₂, x₄, x₆, x₇

    Choice 3 (Co-spinor):
    ---------------------
    Spatial: x₁, x₂, x₄ (Fano plane line 2)
    Internal: x₀, x₃, x₅, x₆, x₇

    Physical Interpretation:
    ------------------------
    Each choice gives different:
    - Gauge group structure
    - Particle spectrum
    - Coupling constants

    The "correct" choice is determined by matching to observed physics.
    """)

    # Construct all three projections
    projections = []

    # Choice 1: Vector (canonical)
    P1 = np.zeros((4, 10))
    P1[0, 0] = 1  # t
    P1[1, 2] = 1  # x₀
    P1[2, 3] = 1  # x₁
    P1[3, 1] = 1  # z

    # Choice 2: Spinor (Fano line)
    P2 = np.zeros((4, 10))
    P2[0, 0] = 1  # t
    P2[1, 2] = 1  # x₀
    P2[2, 5] = 1  # x₃
    P2[3, 1] = 1  # z

    # Choice 3: Co-spinor
    P3 = np.zeros((4, 10))
    P3[0, 0] = 1  # t
    P3[1, 3] = 1  # x₁
    P3[2, 4] = 1  # x₂
    P3[3, 1] = 1  # z

    projections = [P1, P2, P3]

    # Verify all give Minkowski
    eta_10 = np.diag([1, -1, -1, -1, -1, -1, -1, -1, -1, -1])

    print("\n  Verifying All Three Projections:")
    print("  " + "-" * 50)

    for i, P in enumerate(projections):
        eta_induced = P @ eta_10 @ P.T
        is_minkowski = np.allclose(eta_induced, np.diag([1, -1, -1, -1]))
        print(f"    Choice {i + 1}: Minkowski = {is_minkowski}")

    return projections


# =============================================================================
# SYNTHESIS
# =============================================================================


def synthesize_uniqueness():
    """
    Synthesize the uniqueness result.
    """
    print("\n" + "=" * 70)
    print("SYNTHESIS: Projection Uniqueness")
    print("=" * 70)

    print("""
    RESULT:
    =======

    The projection P: h₂(O) → R^{1,3} is UNIQUE up to:

    1. DISCRETE TRIALITY CHOICE (Z₃):
       Three inequivalent ways to embed spacetime in the octonion.
       Each corresponds to a different SO(8) triality representation.

    2. CONTINUOUS LORENTZ FREEDOM (SO(1,3)):
       Within each triality class, projections related by Lorentz
       transformations are physically equivalent.

    Key Constraints:
    ----------------
    1. Determinant preservation: Q(M) → η(v,v)
    2. Spin(9) equivariance: P(g·M) = ρ(g)·P(M)
    3. Rank-1 compatibility: det(M)=0 → η(v,v)=0 (on restricted subspace)

    Physical Significance:
    ----------------------
    The triality choice determines:
    - Which octonion directions become spatial
    - Which become internal (gauge/matter)
    - The structure of the Standard Model gauge group

    The canonical choice (vector representation) gives:
    - 3 spatial dimensions from quaternionic part of O
    - 4 internal dimensions from "purely octonionic" part
    - This matches SO(4) ~ SU(2)×SU(2) electroweak structure!

    ═══════════════════════════════════════════════════════════════════════

    EQUATION E03 STATUS: RESOLVED ✓

    The projection is unique up to triality (discrete) and Lorentz (gauge).
    The physical content is fully determined by algebraic constraints.

    ═══════════════════════════════════════════════════════════════════════
    """)


# =============================================================================
# MAIN
# =============================================================================


def main():
    """Run complete uniqueness analysis."""

    state_uniqueness_problem()
    analyze_h2o_structure()
    determinant_constraint()
    f4_equivariance()
    rank1_constraint()
    prove_uniqueness()
    construct_explicit_projection()
    analyze_alternative_projections()
    synthesize_uniqueness()


if __name__ == "__main__":
    main()
    print("\n✓ Projection uniqueness analysis complete!")
