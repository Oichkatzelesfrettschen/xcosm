"""
BAO Data Loader
===============

Provides access to Baryon Acoustic Oscillation measurements from DESI and SDSS.

Datasets:
- DESI DR1/DR2: Dark Energy Spectroscopic Instrument
- SDSS DR12/DR16: Sloan Digital Sky Survey

References:
- DESI Collaboration 2024, arXiv:2404.03002
- BOSS/eBOSS DR12/DR16 combined analysis
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List
import numpy as np

from .base import get_raw_data_path, DatasetInfo


@dataclass
class BAOMeasurement:
    """
    Single BAO measurement.

    Attributes
    ----------
    z_eff : float
        Effective redshift of measurement
    dm_over_rd : Optional[float]
        D_M / r_d (comoving angular diameter distance / sound horizon)
    dh_over_rd : Optional[float]
        D_H / r_d (Hubble distance / sound horizon) = c/(H(z)*r_d)
    dv_over_rd : Optional[float]
        D_V / r_d (volume-averaged distance / sound horizon)
    f_sigma8 : Optional[float]
        Growth rate times sigma_8 at this redshift
    tracer : str
        Galaxy tracer type (e.g., "LRG", "ELG", "QSO", "Lya")
    survey : str
        Survey name (e.g., "DESI", "SDSS")
    """
    z_eff: float
    dm_over_rd: Optional[float] = None
    dh_over_rd: Optional[float] = None
    dv_over_rd: Optional[float] = None
    f_sigma8: Optional[float] = None
    tracer: str = ""
    survey: str = ""

    @property
    def has_full_ap(self) -> bool:
        """Check if measurement has full Alcock-Paczynski info (D_M and D_H)."""
        return self.dm_over_rd is not None and self.dh_over_rd is not None


class BAODataset:
    """
    Collection of BAO measurements.

    Provides unified access to DESI and SDSS BAO data with covariance matrices.

    Examples
    --------
    >>> bao = BAODataset()
    >>> bao.load_desi_dr2()
    >>> print(f"Loaded {len(bao)} BAO measurements")
    >>> for m in bao.measurements:
    ...     print(f"z={m.z_eff:.2f}: D_M/r_d={m.dm_over_rd:.1f}")
    """

    def __init__(self):
        self.measurements: List[BAOMeasurement] = []
        self.covariance: Optional[np.ndarray] = None
        self._loaded = False

    def __len__(self) -> int:
        return len(self.measurements)

    def __iter__(self):
        return iter(self.measurements)

    def load_desi_dr2(self) -> 'BAODataset':
        """
        Load DESI DR2 BAO measurements.

        Returns
        -------
        BAODataset
            Self for method chaining
        """
        bao_path = get_raw_data_path() / 'bao_data' / 'desi_bao_dr2'

        # DESI DR2 tracers and their redshift bins
        # File naming: desi_gaussian_bao_{tracer}_mean.txt
        tracers = [
            ('BGS_BRIGHT-21.35_GCcomb', 0.30, 'BGS'),
            ('LRG_GCcomb_z0.4-0.6', 0.51, 'LRG'),
            ('LRG_GCcomb_z0.6-0.8', 0.71, 'LRG'),
            ('LRG+ELG_LOPnotqso_GCcomb', 0.93, 'LRG+ELG'),
            ('ELG_LOPnotqso_GCcomb_z1.1-1.6', 1.32, 'ELG'),
            ('QSO_GCcomb', 1.49, 'QSO'),
            ('Lya_GCcomb', 2.33, 'Lya'),
        ]

        self.measurements = []

        for tracer_file, z_eff, tracer_type in tracers:
            mean_file = bao_path / f'desi_gaussian_bao_{tracer_file}_mean.txt'

            if not mean_file.exists():
                continue

            try:
                # Load mean values - file format varies:
                # Some have: z, value, quantity_name
                # Some have: DM_over_rd, DH_over_rd
                with open(mean_file, 'r') as f:
                    lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]

                dm_over_rd = None
                dh_over_rd = None
                dv_over_rd = None
                actual_z = z_eff

                for line in lines:
                    parts = line.split()
                    if len(parts) >= 3:
                        # Format: z value quantity
                        actual_z = float(parts[0])
                        value = float(parts[1])
                        quantity = parts[2] if len(parts) > 2 else ""

                        if 'DM' in quantity.upper():
                            dm_over_rd = value
                        elif 'DH' in quantity.upper():
                            dh_over_rd = value
                        elif 'DV' in quantity.upper():
                            dv_over_rd = value
                    elif len(parts) == 2:
                        # Format: DM_over_rd, DH_over_rd on separate lines
                        if dm_over_rd is None:
                            dm_over_rd = float(parts[0])
                            dh_over_rd = float(parts[1])
                        else:
                            dh_over_rd = float(parts[0])

                measurement = BAOMeasurement(
                    z_eff=actual_z,
                    dm_over_rd=dm_over_rd,
                    dh_over_rd=dh_over_rd,
                    dv_over_rd=dv_over_rd,
                    tracer=tracer_type,
                    survey='DESI_DR2'
                )
                self.measurements.append(measurement)
            except (OSError, ValueError) as e:
                continue

        self._loaded = True
        return self

    def load_desi_dr1(self) -> 'BAODataset':
        """
        Load DESI DR1 (2024) BAO measurements.

        Returns
        -------
        BAODataset
            Self for method chaining
        """
        bao_path = get_raw_data_path() / 'bao_data'

        # DESI 2024 tracers
        tracers = [
            ('BGS_BRIGHT-21.5_GCcomb_z0.1-0.4', 0.30, 'BGS'),
            ('LRG_GCcomb_z0.4-0.6', 0.51, 'LRG'),
            ('LRG_GCcomb_z0.6-0.8', 0.71, 'LRG'),
            ('LRG+ELG_LOPnotqso_GCcomb_z0.8-1.1', 0.93, 'LRG+ELG'),
            ('ELG_LOPnotqso_GCcomb_z1.1-1.6', 1.32, 'ELG'),
            ('QSO_GCcomb_z0.8-2.1', 1.49, 'QSO'),
            ('Lya_GCcomb', 2.33, 'Lya'),
        ]

        self.measurements = []

        for tracer_file, z_eff, tracer_type in tracers:
            mean_file = bao_path / f'desi_2024_gaussian_bao_{tracer_file}_mean.txt'

            if not mean_file.exists():
                continue

            try:
                values = np.loadtxt(mean_file)

                if len(values) >= 2:
                    measurement = BAOMeasurement(
                        z_eff=z_eff,
                        dm_over_rd=values[0],
                        dh_over_rd=values[1],
                        tracer=tracer_type,
                        survey='DESI_DR1'
                    )
                    self.measurements.append(measurement)
            except (OSError, ValueError):
                continue

        self._loaded = True
        return self

    def load_sdss(self) -> 'BAODataset':
        """
        Load SDSS DR12/DR16 BAO measurements.

        Returns
        -------
        BAODataset
            Self for method chaining
        """
        bao_path = get_raw_data_path() / 'bao_data'

        # SDSS DR16 LRG
        lrg_file = bao_path / 'sdss_DR16_LRG_BAO_DMDH.dat'
        if lrg_file.exists():
            try:
                data = np.loadtxt(lrg_file)
                # Typical format: z_eff, D_M/r_d, D_H/r_d
                if len(data) >= 3:
                    self.measurements.append(BAOMeasurement(
                        z_eff=data[0],
                        dm_over_rd=data[1],
                        dh_over_rd=data[2],
                        tracer='LRG',
                        survey='SDSS_DR16'
                    ))
            except (OSError, ValueError):
                pass

        # SDSS DR16 QSO
        qso_file = bao_path / 'sdss_DR16_QSO_BAO_DMDH.txt'
        if qso_file.exists():
            try:
                data = np.loadtxt(qso_file)
                if len(data) >= 3:
                    self.measurements.append(BAOMeasurement(
                        z_eff=data[0],
                        dm_over_rd=data[1],
                        dh_over_rd=data[2],
                        tracer='QSO',
                        survey='SDSS_DR16'
                    ))
            except (OSError, ValueError):
                pass

        self._loaded = True
        return self

    def get_redshifts(self) -> np.ndarray:
        """Get array of effective redshifts."""
        return np.array([m.z_eff for m in self.measurements])

    def get_dm_rd(self) -> np.ndarray:
        """Get array of D_M/r_d values."""
        return np.array([m.dm_over_rd for m in self.measurements if m.dm_over_rd is not None])

    def get_dh_rd(self) -> np.ndarray:
        """Get array of D_H/r_d values."""
        return np.array([m.dh_over_rd for m in self.measurements if m.dh_over_rd is not None])

    def select_survey(self, survey: str) -> List[BAOMeasurement]:
        """
        Select measurements from a specific survey.

        Parameters
        ----------
        survey : str
            Survey name (e.g., "DESI_DR1", "SDSS_DR16")

        Returns
        -------
        List[BAOMeasurement]
            Matching measurements
        """
        return [m for m in self.measurements if survey.upper() in m.survey.upper()]

    def select_tracer(self, tracer: str) -> List[BAOMeasurement]:
        """
        Select measurements by tracer type.

        Parameters
        ----------
        tracer : str
            Tracer type (e.g., "LRG", "ELG", "QSO")

        Returns
        -------
        List[BAOMeasurement]
            Matching measurements
        """
        return [m for m in self.measurements if tracer.upper() in m.tracer.upper()]

    def info(self) -> DatasetInfo:
        """Return dataset metadata."""
        surveys = set(m.survey for m in self.measurements)
        return DatasetInfo(
            name="BAO Measurements",
            source="DESI + SDSS combined",
            description=f"BAO measurements from {', '.join(surveys)}",
            n_records=len(self),
            columns=['z_eff', 'dm_over_rd', 'dh_over_rd', 'tracer', 'survey'],
            citation="@article{DESI2024, doi={10.48550/arXiv.2404.03002}}"
        )


# Convenience functions
def load_desi_bao(release: str = 'dr2') -> BAODataset:
    """
    Load DESI BAO data.

    Parameters
    ----------
    release : str
        Data release ('dr1' or 'dr2')

    Returns
    -------
    BAODataset
        Loaded BAO measurements

    Examples
    --------
    >>> bao = load_desi_bao('dr2')
    >>> for m in bao:
    ...     print(f"z={m.z_eff:.2f}: D_M/r_d={m.dm_over_rd:.1f}")
    """
    dataset = BAODataset()
    if release.lower() == 'dr2':
        return dataset.load_desi_dr2()
    else:
        return dataset.load_desi_dr1()


def load_sdss_bao() -> BAODataset:
    """
    Load SDSS BAO data.

    Returns
    -------
    BAODataset
        Loaded BAO measurements
    """
    dataset = BAODataset()
    return dataset.load_sdss()
