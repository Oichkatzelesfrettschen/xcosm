"""
Base Data Module
================

Common utilities, paths, and base classes for COSMOS data access.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def get_data_root() -> Path:
    """
    Get the root path to the data directory.

    Searches in order:
    1. COSMOS_DATA_ROOT environment variable
    2. Relative to this file's location (../../data from src/cosmos/data/)
    3. Current working directory + data/

    Returns
    -------
    Path
        Path to the data directory

    Raises
    ------
    FileNotFoundError
        If no valid data directory is found
    """
    # Check environment variable first
    env_root = os.environ.get("COSMOS_DATA_ROOT")
    if env_root:
        path = Path(env_root)
        if path.exists():
            return path

    # Check relative to module location
    module_dir = Path(__file__).parent
    project_data = module_dir.parent.parent.parent.parent / "data"
    if project_data.exists():
        return project_data

    # Check current working directory
    cwd_data = Path.cwd() / "data"
    if cwd_data.exists():
        return cwd_data

    raise FileNotFoundError(
        "Could not find COSMOS data directory. "
        "Set COSMOS_DATA_ROOT environment variable or run from project root."
    )


def get_raw_data_path() -> Path:
    """Get path to raw data directory."""
    return get_data_root() / "raw"


def get_processed_data_path() -> Path:
    """Get path to processed data directory."""
    return get_data_root() / "processed"


@dataclass
class DatasetInfo:
    """
    Metadata about a dataset.

    Attributes
    ----------
    name : str
        Dataset name (e.g., "Pantheon+SH0ES")
    source : str
        Primary source (paper reference or URL)
    description : str
        Brief description of the data
    n_records : int
        Number of records/rows
    columns : list[str]
        Column names
    citation : Optional[str]
        BibTeX key or citation string
    """

    name: str
    source: str
    description: str
    n_records: int = 0
    columns: list[str] = field(default_factory=list)
    citation: Optional[str] = None


# Standard physical constants used in data analysis
SPEED_OF_LIGHT_KM_S = 299792.458  # km/s
H0_PLANCK = 67.4  # km/s/Mpc (Planck 2018)
H0_SHOES = 73.04  # km/s/Mpc (SH0ES 2022)
OMEGA_M_PLANCK = 0.315  # Matter density (Planck 2018)
