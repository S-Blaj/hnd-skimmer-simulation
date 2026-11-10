"""Monte Carlo sampling and population statistics for the HND skimmer model.

All, pass, and fail refer to the SAME initial droplets at the pickup-cell
entrance. The reported initial He size is not the residual He after pickup.

The effective Kr pickup is computed by the Kuma-type mean model; individual
integer pickup events are not sampled.
"""

import numpy as np

from config import N_HE_MEAN, N_SAMPLES, SIGMA_LN
from geometry import CELL_ENTRY_RADIUS_MM
from pickup import kuma_mean_pickup
from propagation import propagate


def sample_initial_droplets(rng):
    """Sample initial He sizes and entrance positions for the HND ensemble.

    Initial sizes are lognormal, with arithmetic mean N_HE_MEAN and
    logarithmic standard deviation SIGMA_LN. Entrance positions are uniform
    in area across the circular cell entrance and independent of HND size.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random generator, normally initialized with the seed in config.py.

    Returns
    -------
    n0, x_entry, y_entry : ndarray
        Initial He atom numbers and pickup-cell entrance coordinates [m].
    """
    z = rng.normal(size=N_SAMPLES)
    n0 = np.exp(np.log(N_HE_MEAN) - SIGMA_LN**2 / 2.0 + SIGMA_LN * z)

    radius = (CELL_ENTRY_RADIUS_MM * 1e-3) * np.sqrt(rng.random(N_SAMPLES))
    phi = 2.0 * np.pi * rng.random(N_SAMPLES)
    return n0, radius * np.cos(phi), radius * np.sin(phi)


def safe_mean(values):
    """Return the mean or NaN when the selected population is empty."""
    return float(np.mean(values)) if len(values) else float("nan")


def one_pressure(p, n0, x, y, gauss_x, gauss_y, mass_choice):
    """Simulate one pressure and mass approximation; summarize selection.

    Parameters
    ----------
    p : float
        Local Kr pressure [mbar].
    n0, x, y : ndarray
        Initial He sizes and entrance positions [m].
    gauss_x, gauss_y : ndarray
        Standard-normal draws for the net transverse momentum kick.
        Reuse the same draws at every pressure and for both mass models.
    mass_choice : {'initial', 'midpoint'}
        Effective mass assumption used in the deflection calculation.

    Returns
    -------
    result : dict
        Transmission and population summary, with the same keys as the
        original single-file implementation (for plotting and CSV export).
    passed : ndarray of bool
        Objects passing BOTH cell exit and SK2.
    k : ndarray
        Continuous effective mean Kr pickup for the input HNDs.
    """
    k, invalid = kuma_mean_pickup(n0, p)
    cell_ok, passed, r_sk2, spread = propagate(
        n0, x, y, k, mass_choice, gauss_x, gauss_y
    )
    failed = ~passed  # Includes losses at cell exit as well as at SK2.

    result = {
        "pressure_mbar": p,
        "model": mass_choice,
        "N_total": len(n0),
        "N_cell": int(cell_ok.sum()),
        "N_SK2_pass": int(passed.sum()),
        "N_SK2_fail": int(failed.sum()),
        "T_cell_percent": 100 * np.mean(cell_ok),
        "T_SK2_percent": 100 * np.mean(passed),
        "Kr_mean_cell": safe_mean(k[cell_ok]),
        "Kr_mean_SK2": safe_mean(k[passed]),
        "Kr_selection_percent": (
            100 * (safe_mean(k[passed]) / safe_mean(k[cell_ok]) - 1)
            if np.any(passed) and np.any(k[cell_ok] > 0)
            else float("nan")
        ),
        "Kuma_invalid_percent": 100 * np.mean(invalid),
        "NHe_initial_mean_ALL": safe_mean(n0),
        "NHe_initial_mean_SK2_PASS": safe_mean(n0[passed]),
        "NHe_initial_mean_SK2_FAIL": safe_mean(n0[failed]),
        "r90_SK2_mm": (
            float(np.quantile(r_sk2[cell_ok], 0.90) * 1e3)
            if np.any(cell_ok) else float("nan")
        ),
        "r99_SK2_mm": (
            float(np.quantile(r_sk2[cell_ok], 0.99) * 1e3)
            if np.any(cell_ok) else float("nan")
        ),
    }
    return result, passed, k


def validate_population_summary(result):
    """Check that transmitted and lost samples reproduce the input sample.

    The weighted average of the selected and lost groups' INITIAL He sizes
    must reproduce the initial mean of the full cell-entry population.
    """
    n_total = result["N_total"]
    n_pass = result["N_SK2_pass"]
    n_fail = result["N_SK2_fail"]

    if n_pass + n_fail != n_total:
        raise AssertionError("PASS and FAIL do not sum to the initial sample")

    if n_pass and n_fail:
        combined_mean = (
            n_pass * result["NHe_initial_mean_SK2_PASS"]
            + n_fail * result["NHe_initial_mean_SK2_FAIL"]
        ) / n_total
        if not np.isclose(combined_mean, result["NHe_initial_mean_ALL"], rtol=1e-10):
            raise AssertionError("Population-weighted initial-He mean is inconsistent")
