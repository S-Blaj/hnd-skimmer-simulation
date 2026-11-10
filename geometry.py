
"""Experimental geometry of the HND beamline.

Input dimensions are given in millimetres.
Derived longitudinal coordinates are in metres.
The helium nozzle defines z = 0.
"""


# ============================================================
# 1. DISTANCES (mm)
# ============================================================

# Nozzle to first skimmer
NOZZLE_TO_SK1_MM = 12.0

# First skimmer to pickup-cell CENTER
SK1_TO_CELL_MM = 340.0

# Pickup-cell length
CELL_LENGTH_MM = 18.0

# Pickup-cell CENTER to second skimmer
CELL_TO_SK2_MM = 166.0


# ============================================================
# 2. APERTURE RADII (mm)
# ============================================================

# First skimmer: diameter 0.4 mm
SK1_RADIUS_MM = 0.2

# Pickup-cell entrance: diameter 3 mm
CELL_ENTRY_RADIUS_MM = 1.5

# Pickup-cell exit: diameter 4 mm
CELL_EXIT_RADIUS_MM = 2.0

# Second skimmer: diameter 3 mm
SK2_RADIUS_MM = 1.5


# ============================================================
# 3. DERIVED LONGITUDINAL POSITIONS (m)
# ============================================================

Z_SK1 = NOZZLE_TO_SK1_MM * 1e-3

Z_CELL_CENTER = (
    NOZZLE_TO_SK1_MM + SK1_TO_CELL_MM
) * 1e-3

Z_ENTRY = (
    Z_CELL_CENTER - 0.5 * CELL_LENGTH_MM * 1e-3
)

Z_EXIT = (
    Z_CELL_CENTER + 0.5 * CELL_LENGTH_MM * 1e-3
)

Z_SK2 = (
    Z_CELL_CENTER + CELL_TO_SK2_MM * 1e-3
)

# Effective kick is applied at the cell center
D_FROM_CENTER = CELL_TO_SK2_MM * 1e-3


# ============================================================
# 4. GEOMETRY VALIDATION
# ============================================================

def validate_geometry():
    """Check the point-source sampling assumption.

    Rays sampled across the pickup-cell entrance
    must also pass through the first skimmer.
    """

    max_radius_at_sk1_mm = (
        CELL_ENTRY_RADIUS_MM * Z_SK1 / Z_ENTRY
    )

    if max_radius_at_sk1_mm >= SK1_RADIUS_MM:
        raise ValueError(
            "SK1 clips rays sampled at the cell entrance; "
            "sample explicitly from SK1 or revise the geometry."
        )
