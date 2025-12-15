"""
COSMOS Engines Module
=====================

Computational engines for simulations and numerical analysis.

Components:
----------
- ccf_bigraph_engine: Computational Cosmogenesis Framework bigraph engine
- ccf_bigraph_enhanced: Enhanced CCF with BAO and dual bigraph structures
- ccf_mathematical_foundations: Mathematical foundations for CCF
- ccf_gauge_emergence_verification: Gauge symmetry emergence verification
- ccf_action_principle_verification: Action principle and uniqueness verification
- ccf_triality_bigraph: F₄ triality bigraph implementation
- flame_box_3d: 3D turbulent flame simulation (spectral solver)
- flame_box_mps: Apple Metal GPU-accelerated flame simulation
- riemann_hydro: Riemann solver for hydrodynamics
- m1_compute_engine: Apple M1/M2 GPU compute engine

Key Classes and Functions:
--------------------------
CCF Bigraph Engine:
    - CosmologicalBigraphEngine: Main CCF simulation engine
    - CosmologicalBigraph: Bigraph data structure
    - CCFParameters: CCF model parameters (calibrated to observations)
    - NodeType: Bigraph node types (matter, radiation, dark energy)
    - LinkType: Bigraph link types (spatial, causal)
    - RewritingRule: Base class for rewriting rules
    - InflationRule: Inflation rewriting dynamics
    - ReheatingRule: Reheating rewriting dynamics
    - PreferentialAttachmentRule: Structure formation dynamics
    - ExpansionRule: Cosmic expansion dynamics

Enhanced CCF:
    - EnhancedCosmologySimulator: Enhanced CCF with BAO
    - DualBigraph: Dual bigraph structure
    - BAOWaveState: Baryon acoustic oscillation state

Flame Simulations:
    - SpectralNSSolver: Spectral Navier-Stokes solver
    - PhysicalParameters: Physical parameters for flame simulation
    - SimConfig: Simulation configuration
    - MPSSpectralSolver: Metal Performance Shaders GPU solver

Riemann Hydro:
    - RiemannHydroSolver: HLLC Riemann solver

M1 Compute:
    - CosmologyGPU: GPU-accelerated cosmology calculations
    - ParallelMCMC: Parallel MCMC on GPU
    - QCDGPU: QCD running on GPU
    - OctonionGPU: Octonion algebra on GPU
"""

# CCF Bigraph Engine
from cosmos.engines.ccf_bigraph_engine import (
    CCFParameters,
    NodeType,
    LinkType,
    Node,
    Link,
    CosmologicalBigraph,
    RewritingRule,
    InflationRule,
    ReheatingRule,
    PreferentialAttachmentRule,
    ExpansionRule,
    CosmologicalBigraphEngine,
)

# Enhanced CCF
from cosmos.engines.ccf_bigraph_enhanced import (
    BAOWaveState,
    DualBigraph,
    EnhancedCosmologySimulator,
)

# CCF Mathematical Foundations
from cosmos.engines.ccf_mathematical_foundations import (
    RewritingEvent,
    CausalPoset,
    GraphTopology,
    QuantumBigraphState,
    UnitaryRewriteOperator,
    OllivierRicciCurvature,
    BigraphAction,
)

# CCF Gauge Emergence
from cosmos.engines.ccf_gauge_emergence_verification import (
    Motif,
    AutomorphismGroup,
    U1Verification,
    SU2Verification,
    SU3Verification,
    StandardModelVerification,
)

# CCF Triality
from cosmos.engines.ccf_triality_bigraph import (
    TrialityType,
    TrialityNode,
    TrialityLink,
    TrialityBigraph,
)

# Flame simulations
from cosmos.engines.flame_box_3d import (
    PhysicalParameters,
    SimConfig,
    SpectralNSSolver,
)

# Riemann hydro
from cosmos.engines.riemann_hydro import (
    RiemannHydroSolver,
)

__all__ = [
    # CCF Bigraph Engine
    "CCFParameters",
    "NodeType",
    "LinkType",
    "Node",
    "Link",
    "CosmologicalBigraph",
    "RewritingRule",
    "InflationRule",
    "ReheatingRule",
    "PreferentialAttachmentRule",
    "ExpansionRule",
    "CosmologicalBigraphEngine",
    # Enhanced CCF
    "BAOWaveState",
    "DualBigraph",
    "EnhancedCosmologySimulator",
    # Mathematical Foundations
    "RewritingEvent",
    "CausalPoset",
    "GraphTopology",
    "QuantumBigraphState",
    "UnitaryRewriteOperator",
    "OllivierRicciCurvature",
    "BigraphAction",
    # Gauge Emergence
    "Motif",
    "AutomorphismGroup",
    "U1Verification",
    "SU2Verification",
    "SU3Verification",
    "StandardModelVerification",
    # Triality
    "TrialityType",
    "TrialityNode",
    "TrialityLink",
    "TrialityBigraph",
    # Flame simulations
    "PhysicalParameters",
    "SimConfig",
    "SpectralNSSolver",
    # Riemann hydro
    "RiemannHydroSolver",
]
