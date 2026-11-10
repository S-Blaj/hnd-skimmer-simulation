
"""Physical parameters and Monte Carlo settings for the HND-Kr skimmer model.

Values are carried over from the original single-file calculation.
Geometry is defined separately in geometry.py.
"""

from pathlib import Path

import numpy as np


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT = PROJECT_DIR / "output"


# ============================================================
# 2. MONTE CARLO SETTINGS
# ============================================================

PRESSURES_MBAR = np.unique(
    np.r_[
        np.linspace(0.0, 0.00130, 21),
        [0.00030, 0.00070, 0.00130],
    ]
).tolist()

N_SAMPLES = 1_000_000
SEED = 42

MASS_MODELS = ("initial", "midpoint")

SAVE_EXTRA_DIAGNOSTICS = False


# ============================================================
# 3. INITIAL HELIUM NANODROPLETS
# ============================================================

N_HE_MEAN = 9678.0
SIGMA_LN = 0.585431


# ============================================================
# 4. EXPERIMENTAL CONDITIONS
# ============================================================

T_KR_K = 300.0
V_HND_M_S = 270.9


# ============================================================
# 5. PHYSICAL CONSTANTS (SI UNITS)
# ============================================================

RHO_HE = 2.22e28
KB = 1.380649e-23
AMU = 1.66053906660e-27

M_HE = 4.002602 * AMU
M_KR = 83.798 * AMU


# ============================================================
# 6. HEDOPANT CALIBRATION
# ============================================================

# Values interpolated from neighboring HeDopant runs
# to a local Kr pressure of P = 1.3e-3 mbar.
#
# These values do NOT come from a direct HeDopant
# simulation at exactly that pressure.
#
# They refer to all simulated states, including He=0.

HE_REMAINING_REF = 998.8252487996668
KR_MEAN_REF = 30.429727488111027

# Effective number of evaporated He atoms per Kr pickup.
# This is a phenomenological model parameter.

C_HE_PER_KR = (
    N_HE_MEAN - HE_REMAINING_REF
) / KR_MEAN_REF
