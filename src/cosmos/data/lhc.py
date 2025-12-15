"""
LHC Particle Physics Data Loader
================================

Provides access to LHC experimental data:
- CMS small-system collectivity (QGP flow)
- ATLAS O-O jet quenching
- W boson mass measurements

References:
- CMS PRL 132, 172302 (2024)
- ATLAS O-O runs 2024-2025
- CDF/CMS/ATLAS W mass precision measurements
"""

from dataclasses import dataclass
from typing import Optional, List
import numpy as np

from .base import get_raw_data_path, DatasetInfo


@dataclass
class FlowMeasurement:
    """
    Elliptic flow (v2) measurement.

    Attributes
    ----------
    multiplicity : float
        Charged particle multiplicity (N_ch)
    v2 : float
        Elliptic flow coefficient
    v2_stat_err : float
        Statistical uncertainty
    v2_sys_err : float
        Systematic uncertainty
    collision_system : str
        Collision type (e.g., "pp", "PbPb", "OO")
    sqrt_s : float
        Center-of-mass energy (TeV)
    """
    multiplicity: float
    v2: float
    v2_stat_err: float
    v2_sys_err: float
    collision_system: str = "pp"
    sqrt_s: float = 13.0

    @property
    def v2_total_err(self) -> float:
        """Total uncertainty (stat + sys in quadrature)."""
        return np.sqrt(self.v2_stat_err**2 + self.v2_sys_err**2)


@dataclass
class JetQuenchingMeasurement:
    """
    Jet suppression (RAA) measurement.

    Attributes
    ----------
    pt : float
        Transverse momentum (GeV)
    raa : float
        Nuclear modification factor
    raa_stat_err : float
        Statistical uncertainty
    raa_sys_err : float
        Systematic uncertainty
    centrality : str
        Centrality class (e.g., "0-10%")
    collision_system : str
        Collision type
    """
    pt: float
    raa: float
    raa_stat_err: float
    raa_sys_err: float
    centrality: str = "0-10%"
    collision_system: str = "OO"


@dataclass
class WMassMeasurement:
    """
    W boson mass measurement.

    Attributes
    ----------
    mass : float
        W boson mass (GeV)
    stat_err : float
        Statistical uncertainty
    sys_err : float
        Systematic uncertainty
    experiment : str
        Experiment name (CMS, ATLAS, CDF)
    year : int
        Publication year
    """
    mass: float
    stat_err: float
    sys_err: float
    experiment: str
    year: int

    @property
    def total_err(self) -> float:
        """Total uncertainty."""
        return np.sqrt(self.stat_err**2 + self.sys_err**2)


class CMSFlowDataset:
    """
    CMS small-system collectivity data.

    Reference: CMS PRL 132, 172302 (2024)
    "Evidence for collective flow in high-multiplicity pp collisions"
    """

    def __init__(self):
        self.measurements: List[FlowMeasurement] = []
        self._loaded = False

    def load(self) -> 'CMSFlowDataset':
        """Load CMS flow data."""
        filepath = get_raw_data_path() / 'lhc_data' / 'cms_small_system_v2.csv'

        if not filepath.exists():
            raise FileNotFoundError(f"CMS flow data not found: {filepath}")

        # Load CSV, skip header comments
        data = np.genfromtxt(filepath, delimiter='|', skip_header=6, comments='#')

        self.measurements = []
        for row in data:
            if len(row) >= 4:
                self.measurements.append(FlowMeasurement(
                    multiplicity=row[0],
                    v2=row[1],
                    v2_stat_err=row[2],
                    v2_sys_err=row[3],
                    collision_system='pp',
                    sqrt_s=13.0
                ))

        self._loaded = True
        return self

    def __len__(self) -> int:
        return len(self.measurements)

    def __iter__(self):
        return iter(self.measurements)

    def get_multiplicity(self) -> np.ndarray:
        """Get array of multiplicities."""
        return np.array([m.multiplicity for m in self.measurements])

    def get_v2(self) -> np.ndarray:
        """Get array of v2 values."""
        return np.array([m.v2 for m in self.measurements])

    def info(self) -> DatasetInfo:
        return DatasetInfo(
            name="CMS Small-System Flow",
            source="CMS PRL 132, 172302 (2024)",
            description="Elliptic flow v2 vs multiplicity in pp collisions at 13 TeV",
            n_records=len(self),
            columns=['multiplicity', 'v2', 'v2_stat_err', 'v2_sys_err'],
            citation="@article{CMS2024, doi={10.1103/PhysRevLett.132.172302}}"
        )


class ATLASQuenchingDataset:
    """
    ATLAS O-O jet quenching data.

    Reference: ATLAS O-O runs 2024-2025
    """

    def __init__(self):
        self.measurements: List[JetQuenchingMeasurement] = []
        self._loaded = False

    def load(self) -> 'ATLASQuenchingDataset':
        """Load ATLAS quenching data."""
        filepath = get_raw_data_path() / 'lhc_data' / 'atlas_oo_quenching_2025.csv'

        if not filepath.exists():
            raise FileNotFoundError(f"ATLAS quenching data not found: {filepath}")

        # Load CSV
        data = np.genfromtxt(filepath, delimiter=',', skip_header=1, comments='#')

        self.measurements = []
        for row in data:
            if len(row) >= 4:
                self.measurements.append(JetQuenchingMeasurement(
                    pt=row[0],
                    raa=row[1],
                    raa_stat_err=row[2],
                    raa_sys_err=row[3] if len(row) > 3 else 0.0,
                    collision_system='OO'
                ))

        self._loaded = True
        return self

    def __len__(self) -> int:
        return len(self.measurements)

    def __iter__(self):
        return iter(self.measurements)

    def info(self) -> DatasetInfo:
        return DatasetInfo(
            name="ATLAS O-O Jet Quenching",
            source="ATLAS O-O runs 2024-2025",
            description="Nuclear modification factor RAA in O-O collisions",
            n_records=len(self),
            columns=['pt', 'raa', 'raa_stat_err', 'raa_sys_err'],
            citation="@article{ATLAS2025}"
        )


class WMassDataset:
    """
    Precision W boson mass measurements.

    Includes CMS, ATLAS, and CDF measurements.
    """

    def __init__(self):
        self.measurements: List[WMassMeasurement] = []
        self._loaded = False

    def load(self) -> 'WMassDataset':
        """Load W mass measurements."""
        filepath = get_raw_data_path() / 'lhc_data' / 'w_boson_mass_2025.csv'

        if not filepath.exists():
            # Use hardcoded values as fallback (GeV)
            self.measurements = [
                WMassMeasurement(80.360, 0.010, 0.012, "CMS", 2024),
                WMassMeasurement(80.366, 0.010, 0.013, "ATLAS", 2024),
                WMassMeasurement(80.434, 0.006, 0.007, "CDF", 2022),
            ]
            self._loaded = True
            return self

        # Load from file - format: Experiment, Value (MeV), Stat (MeV), Sys (MeV), Total (MeV), Year
        self.measurements = []

        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue

                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 6:
                    try:
                        experiment = parts[0]
                        mass_mev = float(parts[1])
                        stat_mev = float(parts[2])
                        sys_mev = float(parts[3])
                        # total_mev = float(parts[4])  # We'll compute this
                        year = int(parts[5])

                        # Convert MeV to GeV
                        self.measurements.append(WMassMeasurement(
                            mass=mass_mev / 1000.0,
                            stat_err=stat_mev / 1000.0,
                            sys_err=sys_mev / 1000.0,
                            experiment=experiment,
                            year=year
                        ))
                    except (ValueError, IndexError):
                        continue

        self._loaded = True
        return self

    def __len__(self) -> int:
        return len(self.measurements)

    def __iter__(self):
        return iter(self.measurements)

    def world_average(self) -> tuple:
        """
        Compute weighted average of W mass measurements.

        Returns
        -------
        tuple
            (average_mass, uncertainty)
        """
        if not self.measurements:
            return (0.0, 0.0)

        weights = [1 / m.total_err**2 for m in self.measurements]
        total_weight = sum(weights)

        average = sum(m.mass * w for m, w in zip(self.measurements, weights)) / total_weight
        uncertainty = np.sqrt(1 / total_weight)

        return (average, uncertainty)

    def sm_prediction(self) -> float:
        """Standard Model prediction for W mass."""
        return 80.357  # GeV (SM electroweak fit)

    def info(self) -> DatasetInfo:
        return DatasetInfo(
            name="W Boson Mass Measurements",
            source="CMS, ATLAS, CDF combined",
            description="Precision W boson mass measurements from collider experiments",
            n_records=len(self),
            columns=['mass', 'stat_err', 'sys_err', 'experiment', 'year'],
            citation="@article{CMS2024W}"
        )


# Convenience functions
def load_cms_flow() -> CMSFlowDataset:
    """Load CMS small-system flow data."""
    dataset = CMSFlowDataset()
    return dataset.load()


def load_atlas_quenching() -> ATLASQuenchingDataset:
    """Load ATLAS O-O jet quenching data."""
    dataset = ATLASQuenchingDataset()
    return dataset.load()


def load_w_mass() -> WMassDataset:
    """Load W boson mass measurements."""
    dataset = WMassDataset()
    return dataset.load()
