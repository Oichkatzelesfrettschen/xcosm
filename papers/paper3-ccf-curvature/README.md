# Paper 3: CCF Ollivier-Ricci Curvature Convergence

**Status:** Theory + Simulation Development Phase

## Title (Working)
"Ollivier-Ricci Curvature Convergence Under Bigraph Rewriting Dynamics: Conditions, Counterexamples, and Tests"

## Core Question
Under what conditions does Ollivier-Ricci curvature on evolving bigraphs converge to Ricci curvature in the continuum limit?

## Critical Problem
**Current state:** CCF simulations show κ_OR ~ -N^{0.55} (diverging negative), NOT converging to flat-space expectation (κ → 0).

**Diagnosis:** Operating outside the regime where convergence theorems apply:
- van der Hoorn et al. (2021, 2023) prove convergence for connected random geometric graphs
- CCF graphs remain disconnected at tested scales (N ≤ 5000)

## Directory Structure
```
paper3-ccf-curvature/
├── README.md                    # This file
├── manuscript/
│   ├── ccf_curvature.tex        # Main LaTeX document
│   ├── ccf_curvature.bib        # Bibliography
│   └── figures/                 # Paper figures
├── code/
│   ├── or_convergence_tests.py  # OR curvature tests
│   ├── connected_rgg_baseline.py # Known regime baseline
│   ├── ccf_rewriting.py         # CCF rewriting dynamics
│   ├── convergence_analysis.py  # Convergence diagnostics
│   └── run_analysis.py          # Main analysis script
├── data/
│   ├── rgg_baselines/           # Known convergence tests
│   ├── ccf_simulations/         # CCF-specific results
│   └── manifold_tests/          # Known manifold comparisons
└── figures/
    └── generated/               # Analysis output figures
```

## Required Demonstrations

### Step 1: Reproduce Known Results
Before any CCF-specific claims, must demonstrate OR convergence in the proven regime.

```python
def test_known_convergence():
    """Reproduce van der Hoorn et al. convergence results."""

    # Test 1: Connected RGG on flat torus (Ricci = 0)
    for N in [100, 500, 1000, 5000, 10000]:
        G = random_geometric_graph(N, r=r_connected(N, d=2))
        kappa_or = ollivier_ricci_curvature(G)
        assert_converges_to_zero(kappa_or, tol=1/sqrt(N))

    # Test 2: Connected RGG on sphere (Ricci = 1/R²)
    for N in [100, 500, 1000, 5000]:
        G = rgg_on_sphere(N, R=1.0)
        kappa_or = ollivier_ricci_curvature(G)
        kappa_ricci = 1.0  # For unit sphere
        assert_converges_to(kappa_or, kappa_ricci, tol=1/sqrt(N))

    # Test 3: Negative curvature (hyperbolic)
    # ... etc
```

### Step 2: Diagnose CCF Failure Mode

```python
def diagnose_ccf_divergence():
    """Understand why CCF regime shows divergence."""

    # Connectivity analysis
    for N in [100, 500, 1000, 5000]:
        G_ccf = ccf_bigraph(N, rules=standard_rules)

        # Check graph properties
        n_components = count_connected_components(G_ccf)
        largest_component = largest_component_size(G_ccf)
        mean_degree = average_degree(G_ccf)

        # Compare to RGG with same N
        G_rgg = random_geometric_graph(N, r=r_connected(N))

        # Log diagnostics
        print(f"N={N}: CCF components={n_components}, "
              f"RGG connected={is_connected(G_rgg)}")
```

### Step 3: Identify Constraints for Convergence

```python
def constrained_rewriting_test():
    """Test if modified rewriting rules preserve convergence."""

    constraints = [
        "maintain_connectivity",
        "preserve_degree_distribution",
        "bounded_curvature_change",
    ]

    for constraint in constraints:
        G = ccf_bigraph_constrained(N=1000, constraint=constraint)
        kappa_or = ollivier_ricci_curvature(G)

        # Does this constraint help?
        converges = check_convergence(kappa_or, target=0)
        print(f"Constraint {constraint}: converges={converges}")
```

## Convergence Theorem Requirements (van der Hoorn et al.)

For OR → Ricci convergence, need:
1. **Connectivity:** Graph must be connected (or have one giant component)
2. **Radius scaling:** r ~ N^{-1/d} where d is manifold dimension
3. **Uniform sampling:** Points uniformly distributed on manifold
4. **Local regularity:** No clustering or voids

CCF graphs violate at least (1) - they remain disconnected.

## Falsification Criteria

| Test | Outcome | Interpretation |
|------|---------|----------------|
| Known regime baseline | Fails | Implementation error - fix before proceeding |
| CCF regime | Diverges | CCF requires modification for GR route |
| Constrained rewriting | All fail | CCF not viable discrete gravity approach |
| Constrained rewriting | Some work | Identify minimal constraints needed |

## What This Paper Removes from COSMOS

- [x] "Emergent GR" language
- [x] "Route to Einstein equations" claims
- [x] Any implication that convergence is demonstrated
- [x] Schematic figures presented as if they show real convergence

## Honest Replacement Language

**Old:**
> "The Ollivier-Ricci curvature κ_OR converges to the Ricci curvature in the continuum limit, providing a route from bigraph dynamics to Einstein gravity."

**New:**
> "Under appropriate conditions (connected graphs with correct radius scaling), Ollivier-Ricci curvature is known to converge to Ricci curvature. Current CCF simulations operate outside this regime (disconnected graphs), showing divergence κ_OR ~ -N^{0.55} rather than convergence. This paper analyzes failure modes and explores constraints under which convergence might be achieved."

## Deliverables Checklist

- [ ] OR curvature implementation validated against literature
- [ ] Connected RGG baseline demonstrating known convergence
- [ ] CCF connectivity analysis at multiple N
- [ ] Divergence scaling law characterization
- [ ] Constraint exploration (≥3 constraint types)
- [ ] Phase diagram: when does CCF converge?
- [ ] Paper draft with honest "open problem" framing
- [ ] Reproducibility package

## Key References

1. van der Hoorn et al. 2021, 2023 - OR convergence proofs
2. Ollivier 2009 - Original OR curvature definition
3. Lin et al. 2011 - OR on graphs
4. Milner 2009 - Bigraph dynamics
5. Konopka et al. 2008 - Quantum graphity

## Success Criteria
1. Known convergence regime reproduced
2. CCF failure mode diagnosed with specific cause
3. Either: constraints found that enable convergence, OR honest "open problem" documented
4. Clear taxonomy: when CCF does/doesn't converge

---
*Last updated: December 2025*
