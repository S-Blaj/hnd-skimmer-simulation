"""Run the HND-Kr pickup, deflection, and skimmer-transmission model.

Usage:
    python main.py

Outputs are written to the ``output/`` directory next to this file:
    - CSV summary for all pressures and both mass assumptions
    - Short two-page PDF report
    - Two standalone scientific figures (each in PDF and PNG)
    - Optional diagnostic PNG plots (see SAVE_EXTRA_DIAGNOSTICS in config.py)

This is an exploratory Kuma-type mean-pickup + Gaussian-kick model,
not HeDopant and not a simulation of individual pickup events.
"""

import csv

import numpy as np

from analysis import (
    one_pressure,
    sample_initial_droplets,
    validate_population_summary,
)
from config import (
    MASS_MODELS,
    N_SAMPLES,
    OUTPUT,
    PRESSURES_MBAR,
    SAVE_EXTRA_DIAGNOSTICS,
    SEED,
)
from geometry import validate_geometry
from pickup import kuma_mean_pickup
from plotting import make_pdf, save_extra_diagnostic_figures


def main():
    """Run all configured pressures with common random draws across models."""
    validate_geometry()
    OUTPUT.mkdir(parents=True, exist_ok=True)

    # 1. One reproducible ensemble sampled at the pickup-cell entrance.
    rng = np.random.default_rng(SEED)
    n0, x_entry, y_entry = sample_initial_droplets(rng)

    # Reuse these transverse-kick draws at every pressure / mass assumption.
    gauss_x = rng.standard_normal(N_SAMPLES)
    gauss_y = rng.standard_normal(N_SAMPLES)

    rows = []
    selections = {}  # (pressure, mass model) -> SK2 pass mask
    kr_values = {}   # pressure -> continuous Kuma-type mean pickup per HND

    # 2. Compute pickup, propagation, and population statistics.
    for pressure in PRESSURES_MBAR:
        kr_values[pressure], _ = kuma_mean_pickup(n0, pressure)

        for mass_model in MASS_MODELS:
            result, passed, _ = one_pressure(
                pressure,
                n0,
                x_entry,
                y_entry,
                gauss_x,
                gauss_y,
                mass_model,
            )
            validate_population_summary(result)
            rows.append(result)
            selections[(pressure, mass_model)] = passed

            print(
                f"P = {pressure:.5f} mbar | {mass_model:8s} | "
                f"SK2 = {result['T_SK2_percent']:6.2f}% | "
                f"initial He after selection = "
                f"{result['NHe_initial_mean_SK2_PASS']:,.0f} | "
                f"mean Kr pickup after SK2 = {result['Kr_mean_SK2']:.2f}"
            )

    # 3. Save numerical results for further analysis / reproduction.
    csv_path = OUTPUT / "results_with_initial_size_selection.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # 4. make_pdf() also creates both standalone PDF/PNG figures.
    report_path = make_pdf(rows, n0, selections, kr_values)

    if SAVE_EXTRA_DIAGNOSTICS:
        save_extra_diagnostic_figures(rows, n0, selections, kr_values)

    # 5. Concise console summary.
    p_ref = max(PRESSURES_MBAR)
    print("\nSimulation completed")
    print(f"  Sample size:        {N_SAMPLES:,} HNDs")
    print(f"  Reference pressure: {p_ref:.5f} mbar")
    for mass_model in MASS_MODELS:
        result = next(
            row for row in rows
            if row["pressure_mbar"] == p_ref and row["model"] == mass_model
        )
        print(
            f"  SK2 {mass_model:8s}: "
            f"{result['T_SK2_percent']:.4f}% transmitted"
        )
    print(f"  CSV:                {csv_path}")
    print(f"  Report:             {report_path}")
    print(f"  Output folder:      {OUTPUT}")


if __name__ == "__main__":
    main()
