
"""Kuma-type mean Kr pickup in shrinking helium nanodroplets."""

import numpy as np

from config import (
    C_HE_PER_KR,
    KB,
    M_KR,
    RHO_HE,
    T_KR_K,
    V_HND_M_S,
)
from geometry import CELL_LENGTH_MM


def kuma_mean_pickup(n0, pressure_mbar):
    """Return effective mean Kr pickup and a validity flag.

    n0: initial He atoms per droplet
    pressure_mbar: local Kr pressure

    The result is a continuous mean, not an integer pickup count.
    """

    n_kr = (pressure_mbar * 100.0) / (KB * T_KR_K)

    radius = (
        3.0 * n0 / (4.0 * np.pi * RHO_HE)
    ) ** (1.0 / 3.0)

    cross_section = np.pi * radius**2

    u_rms = np.sqrt(3.0 * KB * T_KR_K / M_KR)

    v_rel = np.sqrt(
        V_HND_M_S**2 + u_rms**2
    )

    lam0 = (
        n_kr
        * cross_section
        * (CELL_LENGTH_MM * 1e-3)
        * v_rel / V_HND_M_S
    )

    q = C_HE_PER_KR * lam0 / (3.0 * n0)

    k_mean = (n0 / C_HE_PER_KR) * (
        1.0 - np.maximum(0.0, 1.0 - q)**3
    )

    return k_mean, q >= 1.0
