"""Typed, stable boundary utilities for xcosm.

This package is intended to remain small and dependency-light so that analysis
code can depend on it without pulling in the full research surface area.
"""

from xcosm_common.paths import default_output_dir

__all__ = ["default_output_dir"]

