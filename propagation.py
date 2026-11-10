
"""Transverse deflection and geometrical transmission of Kr-doped HNDs.

Simplified Gaussian momentum-kick model, not an event-by-event
pickup simulation.
"""

import numpy as np

from config import (
    C_HE_PER_KR, KB, M_HE, M_KR, T_KR_K, V_HND_M_S
)
from geometry import (
    CELL_EXIT_RADIUS_MM, CELL_LENGTH_MM,
    D_FROM_CENTER, SK2_RADIUS_MM,
    Z_ENTRY, Z_EXIT, Z_SK2,
)


def propagate(n0, x_entry, y_entry, k, mass_choice,
              gauss_x, gauss_y):
    """Calculate pickup-cell exit and SK2 transmission.

    k is the effective Kuma-type mean pickup, not an integer
    number of individual collisions.

    mass_choice:
        'initial'  -> use M0
        'midpoint' -> use (M0 + Mf) / 2

    Returns:
        cell_ok, sk2_ok, r_sk2, sigma_dx
    """

    # Approximate He evaporation and final droplet mass.
    n_he_final = np.maximum(0.0, n0 - C_HE_PER_KR * k)
    m_initial = n0 * M_HE
    m_final = n_he_final * M_HE + k * M_KR

    if mass_choice == "initial":
        m_eff = m_initial
    elif mass_choice == "midpoint":
        m_eff = 0.5 * (m_initial + m_final)
    else:
        raise ValueError("Unknown mass model: " + mass_choice)

    # One-dimensional thermal velocity dispersion of Kr.
    u_sd = np.sqrt(KB * T_KR_K / M_KR)

    # Effective transverse displacement at SK2.
    sigma_dx = (
        D_FROM_CENTER / V_HND_M_S
        * (M_KR / m_eff)
        * np.sqrt(k)
        * u_sd
    )

    dx = gauss_x * sigma_dx
    dy = gauss_y * sigma_dx

    # Propagate from cell entrance to cell exit.
    x_exit = (
        x_entry * Z_EXIT / Z_ENTRY
        + dx * (CELL_LENGTH_MM * 0.5e-3 / D_FROM_CENTER)
    )
    y_exit = (
        y_entry * Z_EXIT / Z_ENTRY
        + dy * (CELL_LENGTH_MM * 0.5e-3 / D_FROM_CENTER)
    )

    # Propagate to the second skimmer.
    x_sk2 = x_entry * Z_SK2 / Z_ENTRY + dx
    y_sk2 = y_entry * Z_SK2 / Z_ENTRY + dy

    # Aperture selection.
    cell_ok = (
        x_exit**2 + y_exit**2
        <= (CELL_EXIT_RADIUS_MM * 1e-3)**2
    )

    sk2_ok = cell_ok & (
        x_sk2**2 + y_sk2**2
        <= (SK2_RADIUS_MM * 1e-3)**2
    )

    return (
        cell_ok,
        sk2_ok,
        np.hypot(x_sk2, y_sk2),
        sigma_dx,
    )
