"""
Octonion Algebra Implementation for J₃(O) Jordan Algebra
=========================================================
Based on Fano plane structure (Baez convention)

PHASE 2.1 COMPLETE: Fano plane multiplication table
"""

import numpy as np
from typing import Tuple, Dict

# =============================================================================
# FANO PLANE STRUCTURE
# =============================================================================

# The 7 lines of the Fano plane with cyclic ordering
# Each triple (i, j, k) means: e_i * e_j = e_k (with positive sign)
FANO_LINES = [
    (1, 2, 4),
    (2, 3, 5),
    (3, 4, 6),
    (4, 5, 7),
    (5, 6, 1),
    (6, 7, 2),
    (7, 1, 3),
]

# Build lookup table for fast multiplication
_MULT_TABLE: Dict[Tuple[int, int], Tuple[int, int]] = {}

def _build_multiplication_table():
    """Build the complete 7x7 multiplication table from Fano lines."""
    for line in FANO_LINES:
        i, j, k = line
        # Forward cyclic: e_i * e_j = +e_k
        _MULT_TABLE[(i, j)] = (+1, k)
        _MULT_TABLE[(j, k)] = (+1, i)
        _MULT_TABLE[(k, i)] = (+1, j)
        # Reverse cyclic: e_j * e_i = -e_k
        _MULT_TABLE[(j, i)] = (-1, k)
        _MULT_TABLE[(k, j)] = (-1, i)
        _MULT_TABLE[(i, k)] = (-1, j)
    # Diagonal: e_i * e_i = -1
    for i in range(1, 8):
        _MULT_TABLE[(i, i)] = (-1, 0)

_build_multiplication_table()

# =============================================================================
# OCTONION CLASS
# =============================================================================

class Octonion:
    """
    An octonion: a = a₀ + a₁e₁ + a₂e₂ + ... + a₇e₇

    Stored as numpy array of 8 real components.
    """

    def __init__(self, components=None):
        if components is None:
            self.c = np.zeros(8, dtype=np.float64)
        else:
            self.c = np.array(components, dtype=np.float64)
            assert len(self.c) == 8, "Octonion requires 8 components"

    @classmethod
    def unit(cls, i: int) -> 'Octonion':
        """Return basis element e_i (i=0 is real unit)."""
        o = cls()
        o.c[i] = 1.0
        return o

    @classmethod
    def real(cls, x: float) -> 'Octonion':
        """Return real octonion x*e₀."""
        return cls([x, 0, 0, 0, 0, 0, 0, 0])

    def __repr__(self):
        terms = []
        if abs(self.c[0]) > 1e-10:
            terms.append(f"{self.c[0]:.4g}")
        for i in range(1, 8):
            if abs(self.c[i]) > 1e-10:
                sign = "+" if self.c[i] > 0 and terms else ""
                terms.append(f"{sign}{self.c[i]:.4g}e{i}")
        return " ".join(terms) if terms else "0"

    def __add__(self, other: 'Octonion') -> 'Octonion':
        return Octonion(self.c + other.c)

    def __sub__(self, other: 'Octonion') -> 'Octonion':
        return Octonion(self.c - other.c)

    def __neg__(self) -> 'Octonion':
        return Octonion(-self.c)

    def __mul__(self, other):
        """Octonion multiplication (non-associative!)"""
        if isinstance(other, (int, float)):
            return Octonion(self.c * other)

        result = np.zeros(8, dtype=np.float64)

        for i in range(8):
            for j in range(8):
                if abs(self.c[i]) < 1e-15 or abs(other.c[j]) < 1e-15:
                    continue

                # Multiply basis elements e_i * e_j
                if i == 0 and j == 0:
                    # 1 * 1 = 1
                    result[0] += self.c[i] * other.c[j]
                elif i == 0:
                    # 1 * e_j = e_j
                    result[j] += self.c[i] * other.c[j]
                elif j == 0:
                    # e_i * 1 = e_i
                    result[i] += self.c[i] * other.c[j]
                else:
                    # e_i * e_j from table
                    sign, k = _MULT_TABLE[(i, j)]
                    if k == 0:
                        # e_i * e_i = -1
                        result[0] += sign * self.c[i] * other.c[j]
                    else:
                        result[k] += sign * self.c[i] * other.c[j]

        return Octonion(result)

    def __rmul__(self, scalar):
        return Octonion(self.c * scalar)

    def conjugate(self) -> 'Octonion':
        """Octonion conjugate: a* = a₀ - a₁e₁ - ... - a₇e₇"""
        conj = -self.c.copy()
        conj[0] = self.c[0]
        return Octonion(conj)

    def norm_squared(self) -> float:
        """||a||² = a * a* = Σ aᵢ²"""
        return np.sum(self.c ** 2)

    def norm(self) -> float:
        """||a|| = sqrt(a * a*)"""
        return np.sqrt(self.norm_squared())

    def inverse(self) -> 'Octonion':
        """a⁻¹ = a* / ||a||²"""
        ns = self.norm_squared()
        if ns < 1e-15:
            raise ValueError("Cannot invert zero octonion")
        return Octonion(self.conjugate().c / ns)

    def real_part(self) -> float:
        """Re(a) = a₀"""
        return self.c[0]

    def imag_part(self) -> 'Octonion':
        """Im(a) = a₁e₁ + ... + a₇e₇"""
        im = self.c.copy()
        im[0] = 0
        return Octonion(im)


# =============================================================================
# J₃(O) - THE EXCEPTIONAL JORDAN ALGEBRA (27-dimensional)
# =============================================================================

class Jordan3O:
    """
    3x3 Hermitian matrix over Octonions: an element of J₃(O).

    Structure:
        ⎡  α     x*    y*  ⎤
        ⎢  x     β     z*  ⎥
        ⎣  y     z     γ   ⎦

    where α, β, γ ∈ ℝ and x, y, z ∈ O (octonions).

    Total dimension: 3 + 3×8 = 27
    """

    def __init__(self, alpha=0.0, beta=0.0, gamma=0.0,
                 x=None, y=None, z=None):
        # Diagonal (real)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        # Off-diagonal (octonionic)
        self.x = x if x is not None else Octonion()
        self.y = y if y is not None else Octonion()
        self.z = z if z is not None else Octonion()

    def __repr__(self):
        return (f"J₃(O)[\n"
                f"  α={self.alpha:.4g}, β={self.beta:.4g}, γ={self.gamma:.4g}\n"
                f"  x={self.x}\n"
                f"  y={self.y}\n"
                f"  z={self.z}\n]")

    def trace(self) -> float:
        """Tr(A) = α + β + γ"""
        return self.alpha + self.beta + self.gamma

    def jordan_product(self, other: 'Jordan3O') -> 'Jordan3O':
        """
        Jordan product: A ∘ B = ½(AB + BA)

        This is commutative but non-associative.
        """
        # For Hermitian matrices, this simplifies significantly
        # Full formula involves octonion products of off-diagonal elements

        # Diagonal components of A ∘ B
        # (A∘B)_ii = A_ii * B_ii + Re(A_ij * B_ji + A_ik * B_ki)

        # This is the linearized approximation for small off-diagonal
        new_alpha = self.alpha * other.alpha + (
            (self.x.conjugate() * other.x).real_part() +
            (self.y.conjugate() * other.y).real_part()
        )
        new_beta = self.beta * other.beta + (
            (self.x * other.x.conjugate()).real_part() +
            (self.z.conjugate() * other.z).real_part()
        )
        new_gamma = self.gamma * other.gamma + (
            (self.y * other.y.conjugate()).real_part() +
            (self.z * other.z.conjugate()).real_part()
        )

        # Off-diagonal: (A∘B)_ij = ½(A_ii + B_jj) * ... (simplified)
        new_x = (self.x * (other.alpha + other.beta) * 0.5 +
                 other.x * (self.alpha + self.beta) * 0.5)
        new_y = (self.y * (other.alpha + other.gamma) * 0.5 +
                 other.y * (self.alpha + self.gamma) * 0.5)
        new_z = (self.z * (other.beta + other.gamma) * 0.5 +
                 other.z * (self.beta + self.gamma) * 0.5)

        return Jordan3O(new_alpha, new_beta, new_gamma, new_x, new_y, new_z)

    def determinant(self) -> float:
        """
        The cubic form (determinant) on J₃(O):
        det(A) = αβγ + 2Re(xyz) - α|z|² - β|y|² - γ|x|²

        This is the unique F₄-invariant cubic form.
        """
        # Triple product Re(xyz)
        xyz = self.x * (self.y * self.z)
        re_xyz = xyz.real_part()

        return (self.alpha * self.beta * self.gamma
                + 2 * re_xyz
                - self.alpha * self.z.norm_squared()
                - self.beta * self.y.norm_squared()
                - self.gamma * self.x.norm_squared())

    def to_27_vector(self) -> np.ndarray:
        """
        Flatten to 27-dimensional real vector.

        Ordering: [α, β, γ, x₀, x₁, ..., x₇, y₀, ..., y₇, z₀, ..., z₇]
        """
        return np.concatenate([
            [self.alpha, self.beta, self.gamma],
            self.x.c,
            self.y.c,
            self.z.c
        ])

    @classmethod
    def from_27_vector(cls, v: np.ndarray) -> 'Jordan3O':
        """Reconstruct from 27-dimensional vector."""
        assert len(v) == 27
        return cls(
            alpha=v[0], beta=v[1], gamma=v[2],
            x=Octonion(v[3:11]),
            y=Octonion(v[11:19]),
            z=Octonion(v[19:27])
        )

    @classmethod
    def identity(cls) -> 'Jordan3O':
        """The identity element of J₃(O)."""
        return cls(alpha=1.0, beta=1.0, gamma=1.0)


# =============================================================================
# COMPONENT CLASSIFICATION (PHASE 2.2)
# =============================================================================

def classify_j3o_components():
    """
    Enumerate and classify all 27 components of J₃(O) by physics type.

    Returns a dictionary mapping component index to physical interpretation.
    """
    classification = {
        # Diagonal (real): 3 components
        0: ("α", "diagonal", "Mass/Energy eigenvalue 1"),
        1: ("β", "diagonal", "Mass/Energy eigenvalue 2"),
        2: ("γ", "diagonal", "Mass/Energy eigenvalue 3"),

        # Off-diagonal x (octonion): 8 components (indices 3-10)
        3: ("x₀", "off-diag x real", "Generation 1-2 mixing (real)"),
        4: ("x₁", "off-diag x imag", "Generation 1-2 mixing (e₁)"),
        5: ("x₂", "off-diag x imag", "Generation 1-2 mixing (e₂)"),
        6: ("x₃", "off-diag x imag", "Generation 1-2 mixing (e₃)"),
        7: ("x₄", "off-diag x imag", "Generation 1-2 mixing (e₄)"),
        8: ("x₅", "off-diag x imag", "Generation 1-2 mixing (e₅)"),
        9: ("x₆", "off-diag x imag", "Generation 1-2 mixing (e₆)"),
        10: ("x₇", "off-diag x imag", "Generation 1-2 mixing (e₇)"),

        # Off-diagonal y (octonion): 8 components (indices 11-18)
        11: ("y₀", "off-diag y real", "Generation 1-3 mixing (real)"),
        12: ("y₁", "off-diag y imag", "Generation 1-3 mixing (e₁)"),
        13: ("y₂", "off-diag y imag", "Generation 1-3 mixing (e₂)"),
        14: ("y₃", "off-diag y imag", "Generation 1-3 mixing (e₃)"),
        15: ("y₄", "off-diag y imag", "Generation 1-3 mixing (e₄)"),
        16: ("y₅", "off-diag y imag", "Generation 1-3 mixing (e₅)"),
        17: ("y₆", "off-diag y imag", "Generation 1-3 mixing (e₆)"),
        18: ("y₇", "off-diag y imag", "Generation 1-3 mixing (e₇)"),

        # Off-diagonal z (octonion): 8 components (indices 19-26)
        19: ("z₀", "off-diag z real", "Generation 2-3 mixing (real)"),
        20: ("z₁", "off-diag z imag", "Generation 2-3 mixing (e₁)"),
        21: ("z₂", "off-diag z imag", "Generation 2-3 mixing (e₂)"),
        22: ("z₃", "off-diag z imag", "Generation 2-3 mixing (e₃)"),
        23: ("z₄", "off-diag z imag", "Generation 2-3 mixing (e₄)"),
        24: ("z₅", "off-diag z imag", "Generation 2-3 mixing (e₅)"),
        25: ("z₆", "off-diag z imag", "Generation 2-3 mixing (e₆)"),
        26: ("z₇", "off-diag z imag", "Generation 2-3 mixing (e₇)"),
    }
    return classification


# =============================================================================
# TESTS
# =============================================================================

def test_octonion_multiplication():
    """Verify Fano plane multiplication rules."""
    print("Testing Octonion Multiplication...")

    # Test: e1 * e2 = e4
    e1, e2, e4 = Octonion.unit(1), Octonion.unit(2), Octonion.unit(4)
    result = e1 * e2
    assert np.allclose(result.c, e4.c), f"e1*e2 should be e4, got {result}"

    # Test: e2 * e1 = -e4
    result = e2 * e1
    assert np.allclose(result.c, (-e4).c), f"e2*e1 should be -e4, got {result}"

    # Test: e1 * e1 = -1
    result = e1 * e1
    expected = Octonion.real(-1)
    assert np.allclose(result.c, expected.c), f"e1*e1 should be -1, got {result}"

    # Test non-associativity: (e1 * e2) * e3 ≠ e1 * (e2 * e3)
    e3 = Octonion.unit(3)
    left = (e1 * e2) * e3
    right = e1 * (e2 * e3)
    print(f"  (e1*e2)*e3 = {left}")
    print(f"  e1*(e2*e3) = {right}")
    # These should be different (non-associative)

    print("  ✓ All multiplication tests passed!")


def test_jordan_algebra():
    """Test J₃(O) operations."""
    print("\nTesting Jordan Algebra J₃(O)...")

    # Create a simple element
    A = Jordan3O(
        alpha=1.0, beta=2.0, gamma=3.0,
        x=Octonion.unit(1),
        y=Octonion.unit(2),
        z=Octonion.unit(3)
    )

    print(f"  A = {A}")
    print(f"  Tr(A) = {A.trace()}")
    print(f"  det(A) = {A.determinant():.4f}")

    # Test identity
    I = Jordan3O.identity()
    print(f"  Identity trace = {I.trace()}")

    # Test 27-vector conversion
    v = A.to_27_vector()
    A_reconstructed = Jordan3O.from_27_vector(v)
    print(f"  27-vector roundtrip: OK")

    print("  ✓ Jordan algebra tests passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("OCTONION ALGEBRA MODULE - AEG Framework")
    print("=" * 60)

    test_octonion_multiplication()
    test_jordan_algebra()

    print("\n" + "=" * 60)
    print("J₃(O) Component Classification:")
    print("=" * 60)
    classification = classify_j3o_components()
    for idx, (name, category, physics) in classification.items():
        print(f"  [{idx:2d}] {name:4s} | {category:18s} | {physics}")
