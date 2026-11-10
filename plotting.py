"""Scientific figures and concise report for the HND-Kr skimmer simulation.

This module only visualizes simulation results; it does not calculate pickup,
particle trajectories, or aperture transmission.

Outputs:
    - Two manuscript-style figures as vector PDF and high-resolution PNG
    - One two-page PDF report combining assumptions, results, and figures
    - Optional diagnostic PNG plots
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from config import (
    OUTPUT, N_SAMPLES, PRESSURES_MBAR, MASS_MODELS,
    N_HE_MEAN, SIGMA_LN, SEED,
    T_KR_K, V_HND_M_S, C_HE_PER_KR,
    KR_MEAN_REF, HE_REMAINING_REF,
)
from geometry import (
    NOZZLE_TO_SK1_MM, SK1_TO_CELL_MM, CELL_LENGTH_MM, CELL_TO_SK2_MM,
    SK1_RADIUS_MM, CELL_ENTRY_RADIUS_MM, CELL_EXIT_RADIUS_MM, SK2_RADIUS_MM,
)


def plot_overview(rows):
    fig, axs = plt.subplots(2, 1, figsize=(9, 9), layout='constrained')
    for mode in MASS_MODELS:
        rr = [r for r in rows if r['model'] == mode]
        pp = np.array([r['pressure_mbar'] for r in rr]) * 1000
        axs[0].plot(pp, [r['T_SK2_percent'] for r in rr], 'o-', label=f'{mode} mass')
        axs[1].plot(pp, [r['Kr_mean_SK2'] for r in rr], 'o-',
                    label=f'After SK2 ({mode})')
        axs[1].plot(pp, [r['Kr_mean_cell'] for r in rr], '--',
                    label=f'Cell exit ({mode})')
    axs[0].set(title='Geometrical transmission', ylabel='Objects passing SK2 (%)')
    axs[1].set(title='Mean Kuma-type Kr pickup: before and after SK2',
               xlabel=r'Local Kr pressure ($10^{-3}$ mbar)',
               ylabel='Mean effective Kr pickup per object')
    for ax in axs:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
    fig.suptitle('Kuma-type pickup + simplified transverse-kick model', fontsize=15)
    return fig


def plot_initial_he_means(rows):
    """THE USER-REQUESTED GRAPH: mean INITIAL He in ALL / PASS / FAIL."""
    fig, axs = plt.subplots(2, 1, figsize=(10, 9), layout='constrained', sharex=True)
    colors = ('#64748b', '#16826b', '#bd6932')
    keys = ('NHe_initial_mean_ALL', 'NHe_initial_mean_SK2_PASS',
            'NHe_initial_mean_SK2_FAIL')
    labels = ('All at cell entrance', 'Passed SK2', 'Did not pass SK2')
    for ax, mode in zip(axs, MASS_MODELS):
        rr = [r for r in rows if r['model'] == mode]
        pp = np.arange(len(rr))
        bar_w = 0.26
        for j, (key, label, color) in enumerate(zip(keys, labels, colors)):
            values = [r[key] for r in rr]
            bars = ax.bar(pp + (j-1)*bar_w, values, bar_w, label=label, color=color)
            for bar, val in zip(bars, values):
                if np.isfinite(val):
                    ax.annotate(f'{val:,.0f}', (bar.get_x()+bar.get_width()/2, bar.get_height()),
                                ha='center', va='bottom', fontsize=8, rotation=0,
                                xytext=(0, 3), textcoords='offset points')
        ax.set_title(f'{mode.capitalize()} mass approximation')
        ax.set_ylabel('Mean INITIAL He atoms / sampled HND')
        ax.grid(axis='y', alpha=0.2)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8, ncol=3, loc='upper left')
        # Keep zero baseline so bar height has a transparent interpretation.
        ymax = max(r[key] for r in rr for key in keys)
        ax.set_ylim(0, ymax*1.24)
    axs[-1].set_xticks(np.arange(len(PRESSURES_MBAR)))
    axs[-1].set_xticklabels([f'{p:.5f}' for p in PRESSURES_MBAR])
    axs[-1].set_xlabel('Local Kr pressure (mbar)')
    fig.suptitle('Which INITIAL droplet sizes are preferentially transmitted?', fontsize=15)
    return fig


def plot_initial_he_histograms(n0, selections, focus_pressure):
    """Density-normalized histograms; PASS and FAIL counts differ greatly."""
    fig, axs = plt.subplots(2, 1, figsize=(10, 9), layout='constrained', sharex=True)
    qlo, qhi = np.quantile(n0, [0.001, 0.999])
    bins = np.linspace(max(0.0, qlo), qhi, 65)
    for ax, mode in zip(axs, MASS_MODELS):
        passed = selections[(focus_pressure, mode)]
        failed = ~passed
        ax.hist(n0, bins=bins, density=True, histtype='step', linewidth=2,
                color='#64748b', label=f'All ({len(n0):,})')
        if passed.any():
            ax.hist(n0[passed], bins=bins, density=True, histtype='step', linewidth=2,
                    color='#16826b', label=f'Passed SK2 ({passed.sum():,})')
        if failed.any():
            ax.hist(n0[failed], bins=bins, density=True, histtype='step', linewidth=2,
                    color='#bd6932', label=f'Did not pass SK2 ({failed.sum():,})')
        ax.set_title(f'{mode.capitalize()} mass approximation')
        ax.set_ylabel('Probability density (per He atom)')
        ax.grid(alpha=0.18)
        ax.legend(fontsize=9)
    axs[-1].set_xlabel('INITIAL number of He atoms in one sampled droplet')
    fig.suptitle(f'Initial-size distribution conditioned on SK2 transmission\n'
                 f'Kr pressure = {focus_pressure:.5f} mbar; each histogram is area-normalized',
                 fontsize=14)
    return fig


def plot_kr_histograms(n0, kr_values, selections, focus_pressure):
    """Additional direct diagnostic: Kr selection; k is a MEAN, not integer."""
    fig, axs = plt.subplots(2, 1, figsize=(10, 9), layout='constrained', sharex=True)
    k = kr_values[focus_pressure]
    bins = np.linspace(0, max(2, np.quantile(k, .999)), 65)
    for ax, mode in zip(axs, MASS_MODELS):
        passed = selections[(focus_pressure, mode)]
        ax.hist(k, bins=bins, density=True, histtype='step', linewidth=2,
                color='#64748b', label='All at cell entrance')
        ax.hist(k[passed], bins=bins, density=True, histtype='step', linewidth=2,
                color='#16826b', label='Passed SK2')
        ax.hist(k[~passed], bins=bins, density=True, histtype='step', linewidth=2,
                color='#bd6932', label='Did not pass SK2')
        ax.set(title=f'{mode.capitalize()} mass approximation', ylabel='Probability density')
        ax.grid(alpha=0.18)
        ax.legend(fontsize=9)
    axs[-1].set_xlabel('Effective MEAN Kr pickup from Kuma (not integer counts)')
    fig.suptitle(f'Kr-pickup selection at {focus_pressure:.5f} mbar', fontsize=15)
    return fig



# ============================================================================
# NEW: manuscript-ready plots (TWO PDF/PNG FIGURES, SAME SIMULATION)
# ============================================================================
# Keep diagnostic figures 01-04 and the full English PDF unchanged.
# These routines only visualize the same arrays produced by main().

def sm_style():
    """Journal-style typography. Sans-serif; colors as in the main figures."""
    return {
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'mathtext.fontset': 'dejavusans',
        'font.size': 9, 'axes.labelsize': 9.0,
        'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'legend.fontsize': 7.1, 'axes.linewidth': .82,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': True, 'ytick.right': True,
        'xtick.major.size': 3.5, 'ytick.major.size': 3.5,
        'xtick.minor.size': 1.7, 'ytick.minor.size': 1.7,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'savefig.facecolor': 'white',
    }


def sm_save(fig, stem):
    """Export a journal-sized vector PDF and a high-resolution PNG."""
    fig.savefig(OUTPUT / (stem + '.pdf'), bbox_inches='tight', pad_inches=.055)
    fig.savefig(OUTPUT / (stem + '.png'), dpi=360, bbox_inches='tight', pad_inches=.055)
    plt.close(fig)


def sm_panel_letter(ax, letter):
    ax.text(-0.10, 1.07, '(' + letter + ')', transform=ax.transAxes,
            ha='left', va='top', fontsize=10, weight='bold')


def make_sm_figures(rows, n0, selections, kr_values):
    """Export supplementary figures from the SAME original Monte Carlo draws.

    Fig 1: T_SK2 and mean INITIAL He among SK2-transmitted objects vs pressure.
    Fig 2: INITIAL He size distributions at 0.0013 mbar, and effective Kuma
           mean pickup at cell exit and among transmitted objects.

    IMPORTANT: the Kr size proxy is a continuous per-N0 Kuma mean k(N0,P),
    not a simulated distribution of integer Kr pickup event counts.
    """
    OUTPUT.mkdir(parents=True, exist_ok=True)
    green, orange, gray = '#169875', '#df7822', '#5f6670'
    focus_p = PRESSURES_MBAR[-1]
    rr = {mode: sorted([r for r in rows if r['model'] == mode],
                       key=lambda r: r['pressure_mbar'])
          for mode in MASS_MODELS}
    x = np.array([r['pressure_mbar'] for r in rr['initial']]) * 1e3
    colors = {'initial': green, 'midpoint': orange}
    ls = {'initial': '-', 'midpoint': '-.'}
    marker = {'initial': 'o', 'midpoint': 's'}
    label = {'initial': 'Initial mass', 'midpoint': 'Midpoint mass'}

    with plt.rc_context(sm_style()):
        fig, (a, b) = plt.subplots(1, 2, figsize=(7.15, 3.2))
        for mode in MASS_MODELS:
            dat = rr[mode]
            a.plot(x, [r['T_SK2_percent'] for r in dat], ls[mode],
                   marker=marker[mode], markersize=3.5, lw=1.55,
                   color=colors[mode], label=label[mode])
            b.plot(x, [r['NHe_initial_mean_SK2_PASS'] for r in dat],
                   ls[mode], marker=marker[mode], markersize=3.5,
                   lw=1.55, color=colors[mode], label=label[mode])
        a.axhline(rr['initial'][0]['T_SK2_percent'],
                  ls=':', lw=1.15, color=gray, label='No Kr: geometric baseline')
        b.axhline(float(np.mean(n0)),
                  ls=':', lw=1.15, color=gray, label='All droplets at cell entry')
        a.set_ylabel('SK2 transmission (%)')
        b.set_ylabel(r'Mean initial He atoms, $\langle N_{\mathrm{He}}\rangle$')
        a.set_ylim(0, 49)
        a.legend(loc='lower left', frameon=False)
        b.legend(loc='upper left', frameon=False)
        b.ticklabel_format(axis='y', style='plain', useOffset=False)
        for ax in (a, b):
            ax.set_xlabel(r'Kr pickup pressure ($10^{-3}$ mbar)')
            ax.set_xlim(-.06, 1.37)
            ax.minorticks_on()
            sm_panel_letter(ax, 'a' if ax is a else 'b')
        fig.subplots_adjust(left=.105, right=.985, top=.89, bottom=.21, wspace=.34)
        sm_save(fig, 'SM_Fig_SK2_transmission_and_initial_size')

        fig, (a, b) = plt.subplots(1, 2, figsize=(7.15, 3.3))
        # Fixed physical bin width; normalize by TOTAL population in each group.
        # Do not normalize only the displayed part of the distribution.
        bins = np.linspace(0, 65000, 101)
        xc = .5 * (bins[:-1] + bins[1:])
        bw = np.diff(bins)[0]
        groups = [('Cell entrance', np.ones(len(n0), dtype=bool), gray, ':'),
                  ('SK2: initial mass', selections[(focus_p, 'initial')], green, '-'),
                  ('SK2: midpoint mass', selections[(focus_p, 'midpoint')], orange, '-.')]
        for lab, mask, color, style in groups:
            counts, _ = np.histogram(n0[mask], bins=bins)
            pdf = counts / (np.count_nonzero(mask) * bw)
            # y is probability density per 1,000 He atoms, x is in 1,000 He atoms
            a.step(xc/1000, pdf*1000, where='mid',
                   lw=1.55, ls=style, color=color, label=lab)
        a.set_xlabel(r'Initial He atoms, $N_{\mathrm{He},0}$ ($10^3$)')
        a.set_ylabel('Probability density (per 1000 He atoms)')
        a.set_xlim(0, 55)
        a.legend(loc='upper right', frameon=False, fontsize=6.7)
        # Only effective Kuma means, not integer pickup counts.
        for mode in MASS_MODELS:
            dat = rr[mode]
            b.plot(x, [r['Kr_mean_SK2'] for r in dat], ls[mode],
                   marker=marker[mode], markersize=3.5, lw=1.55,
                   color=colors[mode], label='SK2: ' + label[mode].lower())
        b.plot(x, [r['Kr_mean_cell'] for r in rr['initial']],
               ':', color=gray, lw=1.55, label='Cell exit')
        b.set_xlabel(r'Kr pickup pressure ($10^{-3}$ mbar)')
        b.set_ylabel(r'Mean effective Kr pickup, $\langle k\rangle$')
        b.set_xlim(-.06, 1.37)
        b.set_ylim(bottom=0)
        b.legend(loc='upper left', frameon=False, fontsize=6.8)
        for ax in (a, b):
            ax.minorticks_on()
            sm_panel_letter(ax, 'a' if ax is a else 'b')
        fig.subplots_adjust(left=.105, right=.985, top=.89, bottom=.21, wspace=.35)
        sm_save(fig, 'SM_Fig_SK2_initial_distributions_and_Kr')

    # Figure-specific numbers/caveats for manuscript drafting, no model changes.
    ref = [r for r in rows if r['pressure_mbar'] == focus_p]
    with (OUTPUT / 'SM_Fig_SK2_numerical_summary.txt').open('w', encoding='utf8') as fh:
        fh.write('Model is a Kuma mean-pickup + Monte Carlo trajectory estimate.\n')
        fh.write('HeDopant input is PCHIP-interpolated at p=1.3e-3 mbar, NOT directly simulated.\n')
        fh.write(f'Kr_ref={KR_MEAN_REF:.6f}; He_remaining_all_ref={HE_REMAINING_REF:.6f}; c_He_per_Kr={C_HE_PER_KR:.6f}.\n')
        fh.write('Transmission is relative to sampled pickup-cell entries.\n')
        fh.write('Effective Kr pickup is continuous, not integer event counts.\n')
        fh.write('Cumulative Gaussian kick is placed at pickup-cell center.\n')
        fh.write('No XPS signal weighting or individual pickup events are modeled.\n')
        fh.write('Cell-exit and SK2 results are not predictions of measured signal.\n')
        fh.write(f'Reference pressure: {focus_p:.7f} mbar.\n')
        for r in ref:
            fh.write(f"{r['model']}: transmission={r['T_SK2_percent']:.4f}%, "
                     f"NHe0_pass={r['NHe_initial_mean_SK2_PASS']:.1f}, "
                     f"Kr_cell={r['Kr_mean_cell']:.2f}, "
                     f"Kr_SK2={r['Kr_mean_SK2']:.2f}, "
                     f"q>=1 fraction={r['Kuma_invalid_percent']:.3f}%.\n")

# ============================================================================
# 7. PDF (ENGLISH) + CSV exports
# ============================================================================
def make_pdf(rows, n0, selections, kr_values):
    """Create a concise two-page A4 PDF (methods + combined figures).

    The journal-ready figure PDF/PNG files and the CSV are still exported.
    Numerical model and trajectory selections are unchanged.
    """
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pdf_path = OUTPUT / f'Kuma_SK2_short_report_{N_SAMPLES:_}.pdf'
    # Export vector figures separately, then place high-resolution previews
    # onto a single A4 results page. No additional package is required.
    make_sm_figures(rows, n0, selections, kr_values)
    focus_p = max(PRESSURES_MBAR)
    at_p = {r['model']: r for r in rows if r['pressure_mbar'] == focus_p}
    baseline = next(r['T_SK2_percent'] for r in rows
                    if r['model'] == 'initial' and r['pressure_mbar'] == 0.0)

    with PdfPages(pdf_path) as report:
        # Page 1: self-contained model explanation and key outputs.
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(.09, .948, 'Kuma pickup + SK2 size selection', fontsize=18, weight='bold')
        fig.text(.09, .918, 'Exploratory Monte Carlo model | short report',
                 fontsize=10, color='#555555')

        def heading(y, text):
            fig.text(.09, y, text, fontsize=11, weight='bold', va='top')

        def body(y, text):
            fig.text(.105, y, text, fontsize=9.2, va='top', linespacing=1.45)

        heading(.872, 'Model and input parameters')
        body(.847, f'Initial droplet population: lognormal, mean {N_HE_MEAN:,.0f} He atoms; '
             f'sigma_ln = {SIGMA_LN:.6f}; N = {N_SAMPLES:,}; seed = {SEED}.')
        body(.817, f'Kr gas: T = {T_KR_K:g} K; HND velocity = {V_HND_M_S:g} m/s; '
             f'effective He loss c = {C_HE_PER_KR:.1f} He/Kr.')
        body(.787, f'Distances (mm): nozzle-SK1 {NOZZLE_TO_SK1_MM:g}; '
             f'SK1-cell center {SK1_TO_CELL_MM:g}; cell length {CELL_LENGTH_MM:g}; '
             f'cell center-SK2 {CELL_TO_SK2_MM:g}.')
        body(.757, f'Aperture diameters (mm): SK1 {2*SK1_RADIUS_MM:g}; '
             f'cell entry {2*CELL_ENTRY_RADIUS_MM:g}; '
             f'cell exit {2*CELL_EXIT_RADIUS_MM:g}; SK2 {2*SK2_RADIUS_MM:g}.')

        heading(.710, 'Physical approximations')
        body(.682, '1. Kuma gives a continuous mean pickup for each initial HND size; '
             'individual pickup events are not sampled.')
        body(.633, '2. Apply one net 2D Gaussian momentum kick at the cell center, then test the apertures.')
        body(.584, '3. Initial mass uses M0 throughout; midpoint mass uses '
             '(M0 + Mf)/2. BOTH include Kr-induced deflection.')
        body(.535, '4. No-Kr reference has no pickup-induced deflection; '
             'it includes only geometrical losses through the apertures.')
        body(.486, '5. No photoionization or detection weighting; transmission is not experimentally calibrated.')

        heading(.431, f'Results at local Kr pressure = {focus_p:.4g} mbar')
        heads = ['Mass assumption', 'SK2 pass (%)', 'Initial He, pass', 'Kr, pass (mean)']
        data = [[mode.capitalize(), f"{at_p[mode]['T_SK2_percent']:.2f}",
                 f"{at_p[mode]['NHe_initial_mean_SK2_PASS']:,.0f}",
                 f"{at_p[mode]['Kr_mean_SK2']:.2f}"]
                for mode in MASS_MODELS]
        ax = fig.add_axes([.09, .295, .82, .110]); ax.axis('off')
        tab = ax.table(cellText=data, colLabels=heads, cellLoc='center',
                       loc='center', colWidths=[.29,.20,.27,.24])
        tab.auto_set_font_size(False)
        tab.set_fontsize(8.7)
        tab.scale(1, 1.65)
        for (row, col), cell in tab.get_celld().items():
            if row == 0:
                cell.set_facecolor('#e9edf1')
                cell.set_text_props(weight='bold')

        body(.270, f'Undoped geometrical reference: {baseline:.2f}% pass SK2 '
             '(same initial HND sampling and apertures).')
        heading(.216, 'Interpretation and limits')
        body(.189, 'The mass choices test inertial sensitivity, not statistical uncertainty.')
        body(.160, 'The evaporation coefficient comes from interpolated HeDopant data, not individual events.')
        body(.131, 'Kuma predictions beyond complete He evaporation are clipped and only illustrative.')
        fig.text(.09, .065, 'Page 2: results figures. Standalone vector PDF and PNG figures are also saved.',
                 fontsize=8.4, color='#555555')
        report.savefig(fig)
        plt.close(fig)

        # Page 2: both plots at readable size on one A4 page. The corresponding
        # journal figure PDFs remain vector outputs (see make_sm_figures).
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(.09, .953, 'Results and size-selection diagnostics',
                 fontsize=16, weight='bold', va='top')
        fig.text(.09, .914, 'A. Transmission through SK2 and selected initial He sizes',
                 fontsize=10.4, weight='bold', va='top')
        top = plt.imread(OUTPUT / 'SM_Fig_SK2_transmission_and_initial_size.png')
        ax1 = fig.add_axes([.085, .582, .83, .310]); ax1.imshow(top); ax1.axis('off')
        fig.text(.09, .551,
                 'The Kr-free baseline includes aperture losses but no pickup-induced deflection.',
                 fontsize=9.0, va='top')

        fig.text(.09, .513, 'B. Initial-size distributions and effective Kr pickup',
                 fontsize=10.4, weight='bold', va='top')
        bottom = plt.imread(OUTPUT / 'SM_Fig_SK2_initial_distributions_and_Kr.png')
        ax2 = fig.add_axes([.085, .182, .83, .310]); ax2.imshow(bottom); ax2.axis('off')
        fig.text(.09, .145,
                 'Effective pickup is the continuous Kuma mean, not a sampled integer count.',
                 fontsize=9.0, va='top')
        fig.text(.09, .091,
                 'Figures use the same HND sample and Gaussian draws across all pressures and mass models.',
                 fontsize=8.2, color='#555555')
        report.savefig(fig)
        plt.close(fig)
    return pdf_path


def save_extra_diagnostic_figures(rows, n0, selections, kr_values):
    """Optional exploratory PNGs, intentionally omitted from the short PDF."""
    diagnostics = [
        (plot_overview(rows), 'overview.png'),
        (plot_initial_he_means(rows), 'mean_INITIAL_He_all_pass_fail.png'),
        (plot_initial_he_histograms(n0, selections, max(PRESSURES_MBAR)),
         'initial_He_distributions_at_max_pressure.png'),
        (plot_kr_histograms(n0, kr_values, selections, max(PRESSURES_MBAR)),
         'effective_Kr_distributions_at_max_pressure.png'),
    ]
    for fig, filename in diagnostics:
        fig.savefig(OUTPUT / filename, dpi=190, bbox_inches='tight')
        plt.close(fig)

