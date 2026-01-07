from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OutputLayout:
    """Standard output directory layout for xcosm runs."""

    root: Path

    def project_dir(self, project: str) -> Path:
        return self.root / project


def default_output_dir(*, project: str = "xcosm") -> Path:
    """Default output directory for xcosm analyses.

    Precedence:
    1) `XCOSM_OUTPUT_DIR`
    2) `OPENUNIVERSE_OUTPUT_DIR` / `project`
    3) `~/openuniverse-output/<project>`
    """
    explicit = os.environ.get("XCOSM_OUTPUT_DIR")
    if explicit:
        return Path(explicit).expanduser().resolve()

    base = os.environ.get("OPENUNIVERSE_OUTPUT_DIR")
    root = Path(base).expanduser().resolve() if base else (Path.home() / "openuniverse-output")
    return (root / project).resolve()

