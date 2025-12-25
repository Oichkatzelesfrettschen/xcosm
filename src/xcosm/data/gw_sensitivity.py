"""
Gravitational Wave Detector Sensitivity Data
=============================================

Provides access to sensitivity curves for gravitational wave detectors:
- LISA: Laser Interferometer Space Antenna
- ET: Einstein Telescope (ET-D configuration)

References:
- LISA: Robson et al. 2019, CQG 36, 105011
- ET: ET Consortium, ET-0000A-18
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .base import DatasetInfo, get_raw_data_path


@dataclass
class SensitivityCurve:
    """
    Gravitational wave detector sensitivity curve.

    Attributes
    ----------
    frequency : np.ndarray
        Frequency array (Hz)
    strain : np.ndarray
        Strain sensitivity (1/sqrt(Hz))
    characteristic_strain : Optional[np.ndarray]
        Characteristic strain h_c (dimensionless)
    detector : str
        Detector name
    configuration : str
        Specific configuration/mode
    """

    frequency: np.ndarray
    strain: np.ndarray
    characteristic_strain: Optional[np.ndarray]
    detector: str
    configuration: str = ""

    def __len__(self) -> int:
        return len(self.frequency)

    @property
    def f_min(self) -> float:
        """Minimum frequency (Hz)."""
        return float(self.frequency.min())

    @property
    def f_max(self) -> float:
        """Maximum frequency (Hz)."""
        return float(self.frequency.max())

    @property
    def best_sensitivity(self) -> float:
        """Best (minimum) strain sensitivity."""
        return float(self.strain.min())

    @property
    def optimal_frequency(self) -> float:
        """Frequency at best sensitivity."""
        idx = np.argmin(self.strain)
        return float(self.frequency[idx])

    def sensitivity_at(self, freq: float) -> float:
        """
        Interpolate sensitivity at given frequency.

        Parameters
        ----------
        freq : float
            Frequency (Hz)

        Returns
        -------
        float
            Strain sensitivity at that frequency
        """
        return float(np.interp(freq, self.frequency, self.strain))


class LISASensitivity:
    """
    LISA gravitational wave detector sensitivity.

    The Laser Interferometer Space Antenna is a planned ESA/NASA mission
    to detect gravitational waves in the millihertz band (0.1 mHz - 1 Hz).

    Reference: Robson et al. 2019, CQG 36, 105011
    """

    def __init__(self):
        self._curve: Optional[SensitivityCurve] = None
        self._loaded = False

    def load(self) -> "LISASensitivity":
        """Load LISA sensitivity curve."""
        filepath = get_raw_data_path() / "gw_sensitivity" / "LISA_sensitivity.dat"

        if not filepath.exists():
            raise FileNotFoundError(f"LISA sensitivity data not found: {filepath}")

        # Load data: columns are (frequency, strain, characteristic_strain)
        data = np.loadtxt(filepath)

        if data.ndim == 1:
            # Single row, reshape
            data = data.reshape(1, -1)

        frequency = data[:, 0]
        strain = data[:, 1]
        char_strain = data[:, 2] if data.shape[1] > 2 else None

        self._curve = SensitivityCurve(
            frequency=frequency,
            strain=strain,
            characteristic_strain=char_strain,
            detector="LISA",
            configuration="Science requirement",
        )

        self._loaded = True
        return self

    @property
    def curve(self) -> SensitivityCurve:
        """Get sensitivity curve (loads if needed)."""
        if not self._loaded:
            self.load()
        return self._curve

    @property
    def frequency(self) -> np.ndarray:
        """Frequency array (Hz)."""
        return self.curve.frequency

    @property
    def strain(self) -> np.ndarray:
        """Strain sensitivity (1/sqrt(Hz))."""
        return self.curve.strain

    def info(self) -> DatasetInfo:
        return DatasetInfo(
            name="LISA Sensitivity",
            source="Robson et al. 2019, CQG 36, 105011",
            description="LISA strain sensitivity curve (0.1 mHz - 1 Hz)",
            n_records=len(self.curve) if self._loaded else 0,
            columns=["frequency", "strain", "characteristic_strain"],
            citation="@article{Robson2019, doi={10.1088/1361-6382/ab1101}}",
        )


class EinsteinTelescopeSensitivity:
    """
    Einstein Telescope gravitational wave detector sensitivity.

    The Einstein Telescope is a proposed third-generation ground-based
    detector with sensitivity from ~1 Hz to ~10 kHz.

    Reference: ET Consortium, ET-0000A-18 (ET-D configuration)
    """

    def __init__(self):
        self._curve: Optional[SensitivityCurve] = None
        self._loaded = False

    def load(self) -> "EinsteinTelescopeSensitivity":
        """Load ET sensitivity curve."""
        filepath = get_raw_data_path() / "gw_sensitivity" / "ET-0000A-18.txt"

        if not filepath.exists():
            raise FileNotFoundError(f"ET sensitivity data not found: {filepath}")

        # Load data
        data = np.loadtxt(filepath)

        if data.ndim == 1:
            data = data.reshape(1, -1)

        frequency = data[:, 0]
        strain = data[:, 1]

        self._curve = SensitivityCurve(
            frequency=frequency,
            strain=strain,
            characteristic_strain=None,
            detector="Einstein Telescope",
            configuration="ET-D",
        )

        self._loaded = True
        return self

    @property
    def curve(self) -> SensitivityCurve:
        """Get sensitivity curve (loads if needed)."""
        if not self._loaded:
            self.load()
        return self._curve

    @property
    def frequency(self) -> np.ndarray:
        """Frequency array (Hz)."""
        return self.curve.frequency

    @property
    def strain(self) -> np.ndarray:
        """Strain sensitivity (1/sqrt(Hz))."""
        return self.curve.strain

    def info(self) -> DatasetInfo:
        return DatasetInfo(
            name="Einstein Telescope Sensitivity",
            source="ET Consortium, ET-0000A-18",
            description="ET-D configuration strain sensitivity (1 Hz - 10 kHz)",
            n_records=len(self.curve) if self._loaded else 0,
            columns=["frequency", "strain"],
            citation="@techreport{ET2018}",
        )


# Convenience functions
def load_lisa_sensitivity() -> LISASensitivity:
    """
    Load LISA sensitivity curve.

    Returns
    -------
    LISASensitivity
        Loaded LISA sensitivity data

    Examples
    --------
    >>> lisa = load_lisa_sensitivity()
    >>> print(f"Best sensitivity: {lisa.curve.best_sensitivity:.2e} at {lisa.curve.optimal_frequency:.4f} Hz")
    """
    sensitivity = LISASensitivity()
    return sensitivity.load()


def load_et_sensitivity() -> EinsteinTelescopeSensitivity:
    """
    Load Einstein Telescope sensitivity curve.

    Returns
    -------
    EinsteinTelescopeSensitivity
        Loaded ET sensitivity data

    Examples
    --------
    >>> et = load_et_sensitivity()
    >>> print(f"Frequency range: {et.curve.f_min:.1f} - {et.curve.f_max:.1f} Hz")
    """
    sensitivity = EinsteinTelescopeSensitivity()
    return sensitivity.load()
