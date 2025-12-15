"""
Computational Cosmogenesis Framework (CCF)
==========================================

A Python package for simulating emergent spacetime from bigraphical
reactive systems.

Key Features
------------
- Bigraph simulation engine with cosmological rewriting rules
- H0 gradient analysis and scale-dependent expansion
- Mathematical foundations (Ollivier-Ricci curvature, causal posets)
- Gauge group emergence verification
- CMB-S4 tensor predictions

Quick Start
-----------
>>> from ccf import BigraphEngine, CCFParameters
>>> params = CCFParameters()
>>> engine = BigraphEngine(params)
>>> result = engine.run_simulation(steps=1000)
>>> print(f"Final H0: {result.hubble_parameter:.2f} km/s/Mpc")

References
----------
- Milner, R. (2009). The Space and Motion of Communicating Agents
- Planck Collaboration (2020). A&A 641, A6
- DESI Collaboration (2024). arXiv:2404.03002
"""

__version__ = "1.0.0"
__author__ = "CCF Collaboration"

from .parameters import CCFParameters
from .bigraph import BigraphState, BigraphEngine
from .rewriting import InflationRule, AttachmentRule, ExpansionRule
from .curvature import OllivierRicciCurvature
from .analysis import H0GradientAnalysis, H0Measurement
from .observables import compute_spectral_index, compute_s8, compute_dark_energy_eos

__all__ = [
    "CCFParameters",
    "BigraphState",
    "BigraphEngine",
    "InflationRule",
    "AttachmentRule",
    "ExpansionRule",
    "OllivierRicciCurvature",
    "H0GradientAnalysis",
    "H0Measurement",
    "compute_spectral_index",
    "compute_s8",
    "compute_dark_energy_eos",
]
