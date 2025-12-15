"""
Pantheon+ SNe Ia Data Loader
============================

Provides clean access to the Pantheon+ Type Ia supernova dataset.

Dataset: Pantheon+SH0ES (1701 supernovae, 0.001 < z < 2.26)
Reference: Brout et al. 2022, ApJ 938, 110; Scolnic et al. 2022, ApJ 938, 113
Source: https://github.com/PantheonPlusSH0ES/DataRelease
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np

from .base import get_raw_data_path, DatasetInfo, SPEED_OF_LIGHT_KM_S


@dataclass
class SupernovaRecord:
    """
    Single supernova observation.

    Attributes
    ----------
    name : str
        Supernova identifier
    z_cmb : float
        CMB-frame redshift
    z_hel : float
        Heliocentric redshift
    z_hd : float
        Hubble diagram redshift (used for cosmology)
    mu : float
        Distance modulus (mag)
    mu_err : float
        Distance modulus uncertainty
    x1 : float
        SALT3 stretch parameter
    x1_err : float
        Stretch uncertainty
    c : float
        SALT3 color parameter
    c_err : float
        Color uncertainty
    host_mass : Optional[float]
        Host galaxy stellar mass (log M_sun)
    ra : float
        Right ascension (degrees)
    dec : float
        Declination (degrees)
    """
    name: str
    z_cmb: float
    z_hel: float
    z_hd: float
    mu: float
    mu_err: float
    x1: float
    x1_err: float
    c: float
    c_err: float
    host_mass: Optional[float]
    ra: float
    dec: float


class PantheonDataset:
    """
    Pantheon+ SNe Ia dataset.

    Provides structured access to the Pantheon+SH0ES supernova compilation.

    Examples
    --------
    >>> pantheon = PantheonDataset()
    >>> pantheon.load()
    >>> print(f"Loaded {len(pantheon)} supernovae")
    >>> low_z = pantheon.select(z_max=0.1)
    >>> print(f"Low-z sample: {len(low_z.z)} SNe")
    """

    def __init__(self, filepath: Optional[str] = None):
        """
        Initialize the dataset.

        Parameters
        ----------
        filepath : Optional[str]
            Path to Pantheon data file. If None, uses default location.
        """
        if filepath is None:
            self.filepath = get_raw_data_path() / 'pantheon_data.txt'
        else:
            self.filepath = filepath

        self._loaded = False
        self._data = None

        # Column arrays (populated on load)
        self.names = None
        self.z_cmb = None
        self.z_hel = None
        self.z_hd = None
        self.mu = None
        self.mu_err = None
        self.x1 = None
        self.x1_err = None
        self.c = None
        self.c_err = None
        self.ra = None
        self.dec = None

    def load(self) -> 'PantheonDataset':
        """
        Load the Pantheon+ dataset from file.

        Returns
        -------
        PantheonDataset
            Self, for method chaining
        """
        try:
            import pandas as pd
            # Read file, handle #-prefixed header line
            # First, read the header to get column names
            with open(self.filepath, 'r') as f:
                header_line = f.readline().strip()

            # Parse header: starts with #name, so strip # and split
            if header_line.startswith('#'):
                columns = header_line[1:].split()
            else:
                columns = header_line.split()

            # Load data, skip header line
            data = pd.read_csv(self.filepath, sep=r'\s+', skiprows=1, header=None)

            # Trim columns to match actual data width
            if len(columns) > len(data.columns):
                columns = columns[:len(data.columns)]
            elif len(columns) < len(data.columns):
                # Pad with generic names if needed
                columns.extend([f'col{i}' for i in range(len(columns), len(data.columns))])

            data.columns = columns
        except ImportError:
            # Fallback to numpy
            data = np.genfromtxt(
                self.filepath,
                dtype=None,
                names=True,
                encoding='utf-8',
                skip_header=0,
            )

        self._data = data

        # Extract columns - handle both pandas DataFrame and numpy structured array
        if hasattr(data, 'columns'):
            # Pandas DataFrame
            cols = data.columns.tolist()
            self.names = data['name'].values if 'name' in cols else np.arange(len(data))
            self.z_cmb = data['zcmb'].values if 'zcmb' in cols else data['z'].values
            self.z_hel = data['zhel'].values if 'zhel' in cols else self.z_cmb
            self.z_hd = self.z_cmb  # Use zcmb as HD redshift
            self.mu = data['mb'].values if 'mb' in cols else data['mu'].values
            self.mu_err = data['dmb'].values if 'dmb' in cols else data['dmu'].values
            self.x1 = data['x1'].values if 'x1' in cols else np.zeros(len(data))
            self.x1_err = data['dx1'].values if 'dx1' in cols else np.zeros(len(data))
            self.c = data['color'].values if 'color' in cols else np.zeros(len(data))
            self.c_err = data['dcolor'].values if 'dcolor' in cols else np.zeros(len(data))
            self.ra = data['ra'].values if 'ra' in cols else np.zeros(len(data))
            self.dec = data['dec'].values if 'dec' in cols else np.zeros(len(data))
        else:
            # Numpy structured array
            names = list(data.dtype.names) if data.dtype.names else []
            self.names = data['name'] if 'name' in names else np.arange(len(data))
            self.z_cmb = data['zcmb'] if 'zcmb' in names else data['z']
            self.z_hel = data['zhel'] if 'zhel' in names else self.z_cmb
            self.z_hd = self.z_cmb
            self.mu = data['mb'] if 'mb' in names else data['mu']
            self.mu_err = data['dmb'] if 'dmb' in names else data['dmu']
            self.x1 = data['x1'] if 'x1' in names else np.zeros(len(data))
            self.x1_err = data['dx1'] if 'dx1' in names else np.zeros(len(data))
            self.c = data['color'] if 'color' in names else np.zeros(len(data))
            self.c_err = data['dcolor'] if 'dcolor' in names else np.zeros(len(data))
            self.ra = data['ra'] if 'ra' in names else np.zeros(len(data))
            self.dec = data['dec'] if 'dec' in names else np.zeros(len(data))

        self._loaded = True
        return self

    def __len__(self) -> int:
        """Return number of supernovae in dataset."""
        if not self._loaded:
            return 0
        return len(self.z_cmb)

    @property
    def z(self) -> np.ndarray:
        """Alias for z_cmb (primary redshift)."""
        return self.z_cmb

    def select(
        self,
        z_min: Optional[float] = None,
        z_max: Optional[float] = None,
        x1_min: Optional[float] = None,
        x1_max: Optional[float] = None,
        c_min: Optional[float] = None,
        c_max: Optional[float] = None
    ) -> 'PantheonSelection':
        """
        Select a subset of supernovae based on cuts.

        Parameters
        ----------
        z_min, z_max : float, optional
            Redshift range
        x1_min, x1_max : float, optional
            Stretch parameter range
        c_min, c_max : float, optional
            Color parameter range

        Returns
        -------
        PantheonSelection
            Selected subset with same interface
        """
        if not self._loaded:
            raise RuntimeError("Dataset not loaded. Call load() first.")

        mask = np.ones(len(self), dtype=bool)

        if z_min is not None:
            mask &= self.z_cmb >= z_min
        if z_max is not None:
            mask &= self.z_cmb <= z_max
        if x1_min is not None:
            mask &= self.x1 >= x1_min
        if x1_max is not None:
            mask &= self.x1 <= x1_max
        if c_min is not None:
            mask &= self.c >= c_min
        if c_max is not None:
            mask &= self.c <= c_max

        return PantheonSelection(self, mask)

    def low_d_population(self, z_max: float = 0.1) -> 'PantheonSelection':
        """
        Select LOW-D population (x1 < 0, corresponds to D < 2.18).

        Parameters
        ----------
        z_max : float
            Maximum redshift for selection

        Returns
        -------
        PantheonSelection
            LOW-D supernovae
        """
        return self.select(z_max=z_max, x1_max=0.0)

    def high_d_population(self, z_max: float = 0.1) -> 'PantheonSelection':
        """
        Select HIGH-D population (x1 > 0, corresponds to D > 2.18).

        Parameters
        ----------
        z_max : float
            Maximum redshift for selection

        Returns
        -------
        PantheonSelection
            HIGH-D supernovae
        """
        return self.select(z_max=z_max, x1_min=0.0)

    def compute_h0(self, h0_fiducial: float = 70.0) -> np.ndarray:
        """
        Compute H0 for each supernova using low-z approximation.

        H0 = c * z / d_L where d_L = 10^((mu - 25) / 5) Mpc

        Parameters
        ----------
        h0_fiducial : float
            Fiducial H0 for distance modulus (default 70)

        Returns
        -------
        np.ndarray
            H0 values in km/s/Mpc
        """
        # Distance in Mpc from distance modulus
        distance_mpc = 10 ** ((self.mu - 25) / 5)

        # H0 = cz/d (valid for low z)
        h0_values = SPEED_OF_LIGHT_KM_S * self.z_cmb / distance_mpc

        return h0_values

    def info(self) -> DatasetInfo:
        """Return dataset metadata."""
        return DatasetInfo(
            name="Pantheon+SH0ES",
            source="Brout et al. 2022, ApJ 938, 110; Scolnic et al. 2022, ApJ 938, 113",
            description="Type Ia supernovae compilation (1701 SNe, 0.001 < z < 2.26)",
            n_records=len(self) if self._loaded else 0,
            columns=['name', 'zcmb', 'zhel', 'mb', 'dmb', 'x1', 'dx1', 'color', 'dcolor', 'ra', 'dec'],
            citation="@article{Brout2022, doi={10.3847/1538-4357/ac8e04}}"
        )


class PantheonSelection:
    """
    A selected subset of the Pantheon dataset.

    Provides same interface as PantheonDataset but for a masked subset.
    """

    def __init__(self, parent: PantheonDataset, mask: np.ndarray):
        """
        Initialize selection from parent dataset and mask.

        Parameters
        ----------
        parent : PantheonDataset
            Original dataset
        mask : np.ndarray
            Boolean mask for selection
        """
        self._parent = parent
        self._mask = mask

        # Copy selected data
        self.names = parent.names[mask] if parent.names is not None else None
        self.z_cmb = parent.z_cmb[mask]
        self.z_hel = parent.z_hel[mask]
        self.z_hd = parent.z_hd[mask]
        self.mu = parent.mu[mask]
        self.mu_err = parent.mu_err[mask]
        self.x1 = parent.x1[mask]
        self.x1_err = parent.x1_err[mask]
        self.c = parent.c[mask]
        self.c_err = parent.c_err[mask]
        self.ra = parent.ra[mask]
        self.dec = parent.dec[mask]

    def __len__(self) -> int:
        return len(self.z_cmb)

    @property
    def z(self) -> np.ndarray:
        """Alias for z_cmb."""
        return self.z_cmb

    def compute_h0(self) -> np.ndarray:
        """Compute H0 for selected supernovae."""
        distance_mpc = 10 ** ((self.mu - 25) / 5)
        return SPEED_OF_LIGHT_KM_S * self.z_cmb / distance_mpc

    def mean_h0(self) -> tuple:
        """
        Compute mean H0 and standard error.

        Returns
        -------
        tuple
            (mean_h0, std_err)
        """
        h0_values = self.compute_h0()
        return np.mean(h0_values), np.std(h0_values) / np.sqrt(len(h0_values))


# Convenience function for quick access
def load_pantheon(filepath: Optional[str] = None) -> PantheonDataset:
    """
    Load Pantheon+ dataset.

    Parameters
    ----------
    filepath : Optional[str]
        Path to data file (uses default if None)

    Returns
    -------
    PantheonDataset
        Loaded dataset

    Examples
    --------
    >>> pantheon = load_pantheon()
    >>> print(f"Loaded {len(pantheon)} supernovae")
    >>> print(f"Redshift range: {pantheon.z.min():.3f} - {pantheon.z.max():.3f}")
    """
    dataset = PantheonDataset(filepath)
    dataset.load()
    return dataset
