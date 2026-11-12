#!/usr/bin/env python
"""
Plot the distribution of measured R65 apertures across all sources and
galaxies, per band — to visually judge whether per-source (per-band)
aperture sizing is actually buying anything, or whether a single fixed
PSF-based aperture + one correction factor per band would be an
adequate simplification.

Reads every <GALAXY>_photometry_v2.fits found under --output-dir (as
produced by run_full_pipeline.py / Photometry.save_photometry), pools
r65_native_{band} (arcsec) across all of them, and for each band keeps
only sources with aperture_source_native_{band} == 'own' and
status_native_{band} in ('ok', 'size_extrapolated') — i.e. an actual
independent size measurement for that source in that band, not a
borrowed aperture or a PSF-only fallback (those two are, by
construction, identical to a fixed reference aperture already, so
including them would understate real scatter rather than reveal it).
Each kept source's own r65_native_{band} is also compared against that
band's r65_psf_native_{band} — the fixed, point-source-only aperture a
"one size fits all" scheme would use instead.

Two panels, one figure:
  (top)    r65_native_{band} [arcsec] per band (box + per-source strip),
           with a red dashed line at each band's own PSF-only R65.
  (bottom) r65_native_{band} / r65_psf_native_{band} ratio per band —
           the same comparison, normalized: 1.0 means indistinguishable
           from a fixed point-source aperture; a wide spread/tail above
           1 means many sources are genuinely resolved and a single
           fixed aperture would under-correct their flux.

Also prints a per-band summary table (N with an own measurement vs.
N borrowed/psf_fallback — bands dominated by the latter two mean
per-source measurement isn't even being attempted/possible for most
sources there yet, regardless of how "worth it" it would be).

Only reflects whichever galaxies have already been processed under
--output-dir — for a representative answer, run this after
run_full_pipeline.py has gotten through most/all of the sample, not
just one or two galaxies.

Usage
-----
    python plot_r65_distributions.py [--output-dir output] [--out r65_distributions.png]
"""
import argparse
import glob
import os
import warnings

warnings.simplefilter('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.table import Table, vstack

# Same band order used throughout this pipeline: HST first, then JWST
# NIRCam, then JWST MIRI.
BAND_ORDER = [
    'F275W', 'F336W', 'F438W', 'F555W', 'F814W',
    'F200W', 'F300M', 'F335M', 'F360M',
    'F770W', 'F1000W', 'F1130W', 'F2100W',
]


def load_all_photometry(output_dir):
    """Read every <GALAXY>_photometry_v2.fits under output_dir and
    vstack them (outer join — galaxies don't all necessarily share the
    same band columns, e.g. differing FOV coverage)."""
    paths = sorted(glob.glob(os.path.join(output_dir, '*', '*_photometry_v2.fits')))
    if not paths:
        raise FileNotFoundError(
            f'No <GALAXY>_photometry_v2.fits files found under {output_dir} — '
            f'run run_full_pipeline.py first.'
        )
    tables = []
    for p in paths:
        t = Table.read(p)
        tables.append(t)
        print(f'  loaded {p}  ({len(t)} sources)')
    return vstack(tables, join_type='outer', metadata_conflicts='silent')


def collect_band_data(table, band):
    """Pool one band's r65_native_{band} values (arcsec) across every
    row of `table`, split by aperture_source_native_{band}. Returns
    None if the band's columns aren't present at all."""
    r65_col = f'r65_native_{band}'
    r65_psf_col = f'r65_psf_native_{band}'
    src_col = f'aperture_source_native_{band}'
    status_col = f'status_native_{band}'
    capped_col = f'size_capped_native_{band}'

    if r65_col not in table.colnames:
        return None

    r65 = np.asarray(table[r65_col], dtype=float)
    r65_psf = np.asarray(table[r65_psf_col], dtype=float)
    src = np.asarray(table[src_col]).astype(str)
    status = np.asarray(table[status_col]).astype(str)
    capped = (np.asarray(table[capped_col], dtype=bool) if capped_col in table.colnames
              else np.zeros(len(table), dtype=bool))

    valid = np.isfinite(r65) & np.isfinite(r65_psf) & (r65_psf > 0)
    n_total = int(valid.sum())
    if n_total == 0:
        return None

    is_own = valid & (src == 'own') & np.isin(status, ['ok', 'size_extrapolated'])
    is_borrowed = valid & np.char.startswith(src, 'borrowed')
    is_fallback = valid & (src == 'psf_fallback')
    n_other = n_total - int(is_own.sum()) - int(is_borrowed.sum()) - int(is_fallback.sum())

    return dict(
        r65=r65[is_own],
        ratio=r65[is_own] / r65_psf[is_own],
        capped=capped[is_own],
        n_capped=int(capped[is_own].sum()),
        r65_psf_ref=float(np.median(r65_psf[valid])),
        n_total=n_total, n_own=int(is_own.sum()), n_borrowed=int(is_borrowed.sum()),
        n_fallback=int(is_fallback.sum()), n_other=max(n_other, 0),
    )


def print_summary(bands, per_band):
    print(f"\n{'band':8s} {'N_own':>6s} {'N_borrow':>9s} {'N_fallback':>11s} {'N_capped':>9s} "
          f"{'median R65':>11s} {'median ratio':>13s} {'IQR ratio':>10s} {'%ratio>1.2':>10s}")
    for band in bands:
        d = per_band[band]
        if len(d['ratio']) > 0:
            med_r65 = np.median(d['r65'])
            med_ratio = np.median(d['ratio'])
            iqr_ratio = np.percentile(d['ratio'], 75) - np.percentile(d['ratio'], 25)
            frac_extended = 100.0 * np.mean(d['ratio'] > 1.2)
            med_r65_s, med_ratio_s = f'{med_r65:.3f}', f'{med_ratio:.3f}'
            iqr_s, frac_s = f'{iqr_ratio:.3f}', f'{frac_extended:.1f}'
        else:
            med_r65_s = med_ratio_s = iqr_s = frac_s = 'n/a'
        print(f"{band:8s} {d['n_own']:6d} {d['n_borrowed']:9d} {d['n_fallback']:11d} "
              f"{d['n_capped']:9d} {med_r65_s:>11s} {med_ratio_s:>13s} {iqr_s:>10s} {frac_s:>10s}")
    print(
        "\n(median ratio far above 1 / wide IQR -> many sources are genuinely "
        "resolved beyond the PSF, per-source sizing matters for that band.\n"
        " median ratio ~1 with a narrow IQR -> a single fixed PSF-based "
        "aperture + one correction factor would likely be an adequate "
        "simplification for that band.\n"
        " N_own ~0 -> no independent size measurement is even being made for "
        "that band yet across the processed sample; this comparison doesn't "
        "apply there regardless.\n"
        " N_capped > 0 -> that many 'own' measurements hit max_r65_growth_factor's "
        "safety cap (ratio pinned at exactly that factor) — their true scatter is "
        "larger than shown; see the orange x markers in the ratio panel.)"
    )


def make_plot(bands, per_band, out_path):
    fig, (ax_r65, ax_ratio) = plt.subplots(
        2, 1, figsize=(max(10, 0.9 * len(bands)), 9), sharex=True)
    positions = np.arange(1, len(bands) + 1)
    rng = np.random.default_rng(0)

    # ── top panel: raw R65 per band ─────────────────────────────────────
    r65_data = [per_band[b]['r65'] for b in bands]
    ax_r65.boxplot(r65_data, positions=positions, widths=0.6, showfliers=False,
                    patch_artist=True,
                    boxprops=dict(facecolor='steelblue', alpha=0.5))
    for band, pos in zip(bands, positions):
        d = per_band[band]
        if len(d['r65']):
            jitter = rng.normal(0, 0.05, size=len(d['r65']))
            ax_r65.scatter(pos + jitter, d['r65'], s=8, color='navy', alpha=0.4, zorder=3)
        ax_r65.hlines(d['r65_psf_ref'], pos - 0.35, pos + 0.35, color='crimson',
                       linestyle='--', linewidth=1.8, zorder=4,
                       label='PSF-only R65 (fixed aperture)' if pos == positions[0] else None)
    ax_r65.set_yscale('log')
    ax_r65.set_ylabel('R65 [arcsec]  (own-band measurements)')
    ax_r65.legend(loc='upper left', fontsize=9)
    ax_r65.grid(alpha=0.25, which='both')
    ax_r65.set_title(
        "Per-source R65 vs. each band's fixed PSF-only R65 — pooled across "
        "all processed galaxies\n"
        "(own/fallback source counts per band printed to the terminal)",
        fontsize=11,
    )

    # ── bottom panel: ratio to PSF-only R65 ─────────────────────────────
    # Points that hit max_r65_growth_factor's safety cap sit at exactly
    # that ratio regardless of their true (larger, unknown) size — shown
    # as orange x's rather than green dots so they read as "floor of the
    # true value," not as a genuine measurement.
    ratio_data = [per_band[b]['ratio'] for b in bands]
    ax_ratio.boxplot(ratio_data, positions=positions, widths=0.6, showfliers=False,
                      patch_artist=True,
                      boxprops=dict(facecolor='seagreen', alpha=0.5))
    for band, pos in zip(bands, positions):
        d = per_band[band]
        if len(d['ratio']):
            not_capped = ~d['capped']
            jitter = rng.normal(0, 0.05, size=len(d['ratio']))
            ax_ratio.scatter((pos + jitter)[not_capped], d['ratio'][not_capped],
                              s=8, color='darkgreen', alpha=0.4, zorder=3)
            if d['n_capped']:
                ax_ratio.scatter((pos + jitter)[d['capped']], d['ratio'][d['capped']],
                                  s=22, color='darkorange', marker='x', linewidths=1.3, zorder=5,
                                  label='hit size-growth cap (true ratio is >= this)'
                                  if pos == positions[0] else None)
    ax_ratio.axhline(1.0, color='crimson', linestyle='--', linewidth=1.8, zorder=4,
                      label='ratio = 1  (= fixed PSF-only aperture)')
    ax_ratio.set_ylabel(r'R65$_{\rm own}$ / R65$_{\rm PSF}$')
    ax_ratio.set_xticks(positions)
    ax_ratio.set_xticklabels(bands, rotation=45, ha='right')
    ax_ratio.legend(loc='upper left', fontsize=9)
    ax_ratio.grid(alpha=0.25, which='both')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--output-dir', default='output', help='root output directory (default: output)')
    ap.add_argument('--out', default='r65_distributions.png', help='output PNG path')
    args = ap.parse_args()

    print(f'Scanning {args.output_dir} for <GALAXY>_photometry_v2.fits ...')
    table = load_all_photometry(args.output_dir)
    print(f'\nTotal sources pooled across all galaxies: {len(table)}')

    per_band = {}
    for band in BAND_ORDER:
        data = collect_band_data(table, band)
        if data is not None:
            per_band[band] = data
    if not per_band:
        raise RuntimeError('No usable r65_native_{band} columns found in any loaded table.')
    bands = list(per_band.keys())

    print_summary(bands, per_band)
    make_plot(bands, per_band, args.out)
    print(f'\n✓ Plot saved → {args.out}')


if __name__ == '__main__':
    main()
