"""
Observables Module
==================

Compute cosmological observables from CCF parameters.
"""

import numpy as np
from typing import Tuple, Optional

from .parameters import CCFParameters


def compute_spectral_index(
    lambda_inflation: float = 0.003
) -> Tuple[float, float]:
    """
    Compute scalar spectral index n_s.

    n_s = 1 - 2*lambda

    Parameters
    ----------
    lambda_inflation : float
        Slow-roll parameter

    Returns
    -------
    Tuple[float, float]
        (n_s, sigma_n_s) with theoretical uncertainty
    """
    n_s = 1.0 - 2.0 * lambda_inflation
    sigma = 0.004  # Theoretical uncertainty from parameter calibration
    return n_s, sigma


def compute_tensor_to_scalar(
    lambda_inflation: float = 0.003,
    cos2_theta: float = 0.10
) -> Tuple[float, float]:
    """
    Compute tensor-to-scalar ratio r.

    r = 16 * lambda * cos^2(theta)

    Parameters
    ----------
    lambda_inflation : float
        Slow-roll parameter
    cos2_theta : float
        Multi-field suppression factor

    Returns
    -------
    Tuple[float, float]
        (r, sigma_r) with theoretical uncertainty
    """
    r = 16.0 * lambda_inflation * cos2_theta
    sigma = 0.003  # From parameter uncertainty
    return r, sigma


def compute_tensor_tilt(
    lambda_inflation: float = 0.003,
    xi: float = 0.15,
    cos2_theta: float = 0.10
) -> float:
    """
    Compute tensor spectral tilt n_t.

    n_t = -2*lambda*(1 + xi*cos^2(theta))

    Parameters
    ----------
    lambda_inflation : float
        Slow-roll parameter
    xi : float
        Curvature coupling
    cos2_theta : float
        Multi-field angle

    Returns
    -------
    float
        Tensor spectral tilt
    """
    return -2.0 * lambda_inflation * (1.0 + xi * cos2_theta)


def compute_consistency_ratio(
    lambda_inflation: float = 0.003,
    xi: float = 0.15,
    cos2_theta: float = 0.10
) -> float:
    """
    Compute consistency relation ratio R = r / (-8*n_t).

    Standard inflation predicts R = 1.
    CCF predicts R ~ 0.1 due to multi-field dynamics.

    Returns
    -------
    float
        Consistency ratio R
    """
    r, _ = compute_tensor_to_scalar(lambda_inflation, cos2_theta)
    n_t = compute_tensor_tilt(lambda_inflation, xi, cos2_theta)
    return r / (-8.0 * n_t)


def compute_s8(
    alpha: float = 0.85,
    sigma_8_planck: float = 0.811
) -> Tuple[float, float]:
    """
    Compute S_8 parameter from attachment exponent.

    S_8 = sigma_8 * sqrt(Omega_m / 0.3)

    Parameters
    ----------
    alpha : float
        Preferential attachment exponent
    sigma_8_planck : float
        Planck sigma_8 value

    Returns
    -------
    Tuple[float, float]
        (S_8, sigma_S8)
    """
    s8 = sigma_8_planck * (0.85 / alpha) ** 0.5 * np.sqrt(0.315 / 0.3)
    sigma = 0.02
    return s8, sigma


def compute_dark_energy_eos(
    epsilon: float = 0.25
) -> Tuple[float, float]:
    """
    Compute dark energy equation of state w_0.

    w_0 = -1 + 2*epsilon/3

    Parameters
    ----------
    epsilon : float
        Link tension parameter

    Returns
    -------
    Tuple[float, float]
        (w_0, sigma_w0)
    """
    w0 = -1.0 + 2.0 * epsilon / 3.0
    sigma = 0.05
    return w0, sigma


def compute_dark_energy_evolution(
    epsilon: float = 0.25,
    z: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute w(z) evolution.

    w(z) = w_0 + w_a * z / (1 + z)

    Parameters
    ----------
    epsilon : float
        Link tension parameter
    z : Optional[np.ndarray]
        Redshift array (default 0 to 2)

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (z, w(z)) arrays
    """
    if z is None:
        z = np.linspace(0, 2, 100)

    w0 = -1.0 + 2.0 * epsilon / 3.0
    wa = -2.0 * epsilon / 3.0 * 1.5

    w = w0 + wa * z / (1.0 + z)

    return z, w


def compute_hubble_gradient(
    h0_cmb: float = 67.4,
    m: float = 1.15,
    k_star: float = 0.01
) -> callable:
    """
    Return H0(k) function.

    H0(k) = H0_CMB + m * log10(k/k_star)

    Parameters
    ----------
    h0_cmb : float
        CMB-scale Hubble constant
    m : float
        Gradient (km/s/Mpc per decade)
    k_star : float
        Crossover scale (Mpc^-1)

    Returns
    -------
    callable
        Function H0(k)
    """
    def h0_of_k(k: float) -> float:
        return h0_cmb + m * np.log10(k / k_star)

    return h0_of_k


def predict_cmbs4_observables(
    params: Optional[CCFParameters] = None
) -> dict:
    """
    Predict all CMB-S4 relevant observables.

    Parameters
    ----------
    params : Optional[CCFParameters]
        CCF parameters (uses defaults if None)

    Returns
    -------
    dict
        Dictionary of predictions with uncertainties
    """
    if params is None:
        params = CCFParameters()

    r, sigma_r = compute_tensor_to_scalar(params.lambda_inflation)
    n_t = compute_tensor_tilt(params.lambda_inflation)
    R = compute_consistency_ratio(params.lambda_inflation)

    return {
        "r": {
            "value": r,
            "sigma": sigma_r,
            "cmbs4_sensitivity": 0.001,
            "detection_significance": r / 0.001
        },
        "n_t": {
            "value": n_t,
            "sigma": 0.001,
            "cmbs4_sensitivity": 0.02
        },
        "R": {
            "value": R,
            "expected_standard": 1.0,
            "ccf_prediction": 0.10,
            "distinguishing_power": (1.0 - 0.1) / np.sqrt(0.04**2 + 0.15**2)
        },
        "n_s": {
            "value": params.spectral_index(),
            "sigma": 0.004
        }
    }
