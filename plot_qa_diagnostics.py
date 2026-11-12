#!/usr/bin/env python
"""
QA diagnostic plots for the native-photometry pipeline's overall
performance: how much of the original PHANGS cluster catalog survives
isolation selection, how the native (v2, R65 aperture-corrected)
photometry compares to the convolved (v1) method and to the original
catalog fluxes, whether resulting colors look astrophysically sane, and
whether core per-source steps (emission detection/sizing, centroiding)
are behaving well across the whole processed sample.

Aperture-size distribution (a related QA plot) already exists as its
own script — see plot_r65_distributions.py.

Reads directly from already-saved per-galaxy files under --output-dir
(<GALAXY>_matched.fits, <GALAXY>_photometry_v2.fits) and the PHANGS-HST
DR5 cluster catalog under --cat-path — no live Photometry object
needed, so this can be run any time after run_full_pipeline.py has
processed some galaxies, and re-run as more galaxies finish.

Usage
-----
    python plot_qa_diagnostics.py [--output-dir output] [--cat-path <DR5 dir>]
                                   [--qa-dir output/qa_plots] [--plots all]

--plots accepts a comma-separated subset of:
    isolation, phot_native_convolved, phot_native_catalog,
    phot_convolved_catalog, ccd_optical, ccd_ir_nircam, ccd_ir_miri,
    detection, recenter,
    phot_instp_catalog, phot_instp_native, phot_instp_catalog_galaxy,
    ccd_optical_instp, reff_instp, spectra_photometry
(default: all). run_full_pipeline.py calls make_qa_plots() after a batch run
when [qa] run_qa = True.

Flux-ratio and colour plots show 1-sigma error bars per source (errors of
numerator and denominator in quadrature; the convolved v1 fluxes have no
error column, so only the other side's error is shown for them).
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
from astropy.table import Table

from plot_r65_distributions import load_all_photometry, BAND_ORDER
from analysis import CATALOG_HST_BAND_MAP, _catalog_flux_columns

CLASS_COLORS = {'YCL': 'royalblue', 'MAP': 'darkorange', 'OGCC': 'crimson'}
CLASSES = ['YCL', 'MAP', 'OGCC']

HST_BANDS = BAND_ORDER[:5]
ALL_BANDS = BAND_ORDER

_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.ini')


def _config_path(key, fallback):
    """[paths] default from the config.ini next to this script, if there is one"""
    try:
        from run_full_pipeline import load_config
        return load_config(_CONFIG)[key] or fallback
    except Exception:
        return fallback


DEFAULT_CAT_PATH = _config_path('cat_path', None)
DEFAULT_OUTPUT_DIR = _config_path('output_dir', 'output')

PLOT_NAMES = [
    'isolation', 'phot_native_convolved', 'phot_native_catalog',
    'phot_convolved_catalog', 'ccd_optical', 'ccd_ir_nircam', 'ccd_ir_miri',
    'detection', 'recenter',
    'phot_instp_catalog', 'phot_instp_native', 'phot_instp_catalog_galaxy',
    'ccd_optical_instp', 'reff_instp', 'spectra_photometry',
]
KIND_LABEL = {'native': 'native', 'convolved': 'convolved (v1)', 'catalog': 'catalog',
              'instP': 'instP', 'instG': 'instG'}
BAND_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']   # 5 HST bands, fixed order


def _rng():
    return np.random.default_rng(0)


def _class_array(table):
    return np.asarray(table['PHANGS_CC_CLASS_ML']).astype(str)


# ─────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────

def discover_galaxies(output_dir):
    """Galaxy names with at least a <GAL>_photometry_v2.fits under
    output_dir."""
    paths = sorted(glob.glob(os.path.join(output_dir, '*', '*_photometry_v2.fits')))
    return [os.path.basename(p).split('_photometry_v2.fits')[0] for p in paths]


def load_catalog_class_counts(cat_path, galaxies):
    """{galaxy: {class: count}} from the original DR5 cluster catalog,
    restricted to YCL/MAP/OGCC (the catalog's own 'outside' class is
    dropped — it isn't a cluster/association candidate at all)."""
    machine_path = os.path.join(
        cat_path, 'hlsp_phangs-cat_hst_acs-uvis_dr5-targets_v1_machine-class1+2-primary.fits')
    cat = Table.read(machine_path)
    cat_galaxy = np.asarray(cat['PHANGS_GALAXY']).astype(str)
    cat_class = np.asarray(cat['PHANGS_CC_CLASS_ML']).astype(str)
    counts = {}
    for gal in galaxies:
        gal_lower = gal.lower()
        if gal_lower == 'ngc0628':
            gal_lower = 'ngc628c'  # same special case as IsolatedSourceFinder.get_catalogs
        mask = cat_galaxy == gal_lower
        counts[gal] = {c: int(np.sum(mask & (cat_class == c))) for c in CLASSES}
    return counts


def load_stage_class_counts(output_dir, galaxies, filename_fmt):
    """{galaxy: {class: count}} from a per-galaxy FITS file (e.g.
    '<gal>_matched.fits' or '<gal>_photometry_v2.fits')."""
    counts = {}
    for gal in galaxies:
        path = os.path.join(output_dir, gal, filename_fmt.format(gal=gal))
        if not os.path.exists(path):
            continue
        t = Table.read(path)
        col = np.asarray(t['PHANGS_CC_CLASS_ML']).astype(str)
        counts[gal] = {c: int(np.sum(col == c)) for c in CLASSES}
    return counts


# ─────────────────────────────────────────────────────────────────────────
# 1. Isolation funnel: catalog -> isolated (matched) -> final
# ─────────────────────────────────────────────────────────────────────────

def plot_isolation_funnel(output_dir, cat_path, galaxies, out_path):
    catalog_counts = load_catalog_class_counts(cat_path, galaxies)
    matched_counts = load_stage_class_counts(output_dir, galaxies, '{gal}_matched.fits')
    final_counts = load_stage_class_counts(output_dir, galaxies, '{gal}_photometry_v2.fits')

    totals = {stage: {c: 0 for c in CLASSES} for stage in ('catalog', 'matched', 'final')}
    for gal in galaxies:
        for c in CLASSES:
            totals['catalog'][c] += catalog_counts.get(gal, {}).get(c, 0)
            totals['matched'][c] += matched_counts.get(gal, {}).get(c, 0)
            totals['final'][c] += final_counts.get(gal, {}).get(c, 0)

    print(f"\n{'class':6s} {'catalog':>9s} {'matched':>9s} {'final':>9s} "
          f"{'%matched':>10s} {'%final':>8s}")
    for c in CLASSES:
        cat_n, matched_n, final_n = totals['catalog'][c], totals['matched'][c], totals['final'][c]
        pct_matched = 100 * matched_n / cat_n if cat_n else float('nan')
        pct_final = 100 * final_n / cat_n if cat_n else float('nan')
        print(f"{c:6s} {cat_n:9d} {matched_n:9d} {final_n:9d} "
              f"{pct_matched:10.1f} {pct_final:8.1f}")

    x = np.arange(len(CLASSES))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for i, (stage, color) in enumerate(zip(
            ('catalog', 'matched', 'final'), ('lightgrey', 'steelblue', 'seagreen'))):
        vals = [totals[stage][c] for c in CLASSES]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=stage, color=color,
                       edgecolor='black', linewidth=0.5)
        ax.bar_label(bars, fontsize=8, padding=2)
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES)
    ax.set_ylabel('N sources (log)')
    ax.set_title(f'Catalog -> isolated (matched) -> final sample, by class\n'
                 f'({len(galaxies)} galaxies pooled)')
    ax.legend()
    ax.grid(alpha=0.25, axis='y', which='both')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────
# Shared box+strip-by-band helper (used by the 3 photometry-ratio plots
# and the recentering-shift plot)
# ─────────────────────────────────────────────────────────────────────────

def _box_strip_by_band(ax, bands, values_per_band, classes_per_band, ylabel, logy=True,
                       errors_per_band=None):
    """errors_per_band: optional band -> 1-sigma error of each value (NaN = none),
    drawn as thin vertical bars in the class colour"""
    positions = np.arange(1, len(bands) + 1)
    data = [values_per_band[b] for b in bands]
    ax.boxplot(data, positions=positions, widths=0.6, showfliers=False,
               patch_artist=True, boxprops=dict(facecolor='lightgrey', alpha=0.6))
    rng = _rng()
    seen = set()
    for band, pos in zip(bands, positions):
        vals = values_per_band[band]
        classes = classes_per_band[band]
        if len(vals) == 0:
            continue
        jitter = rng.normal(0, 0.06, size=len(vals))
        for c in CLASSES:
            m = classes == c
            if not m.any():
                continue
            label = c if c not in seen else None
            seen.add(c)
            if errors_per_band is not None and band in errors_per_band:
                e = np.asarray(errors_per_band[band], dtype=float)[m]
                v = vals[m]
                ok = np.isfinite(e) & (e > 0)
                if ok.any():
                    lo = np.minimum(e[ok], 0.95 * v[ok]) if logy else e[ok]
                    ax.errorbar((pos + jitter)[m][ok], v[ok], yerr=[lo, e[ok]], fmt='none',
                                ecolor=CLASS_COLORS[c], elinewidth=0.5, alpha=0.3, zorder=2)
            ax.scatter((pos + jitter)[m], vals[m], s=8, alpha=0.5,
                       color=CLASS_COLORS[c], label=label, zorder=3)
    if logy:
        ax.set_yscale('log')
    ax.set_xticks(positions)
    ax.set_xticklabels(bands, rotation=45, ha='right')
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, which='both')
    ax.legend(fontsize=8)


# ─────────────────────────────────────────────────────────────────────────
# 2-4. Photometry-ratio plots: native/convolved, native/catalog,
# convolved/catalog
# ─────────────────────────────────────────────────────────────────────────

def _flux_and_err(table, kind, band):
    """(flux, 1-sigma error) arrays of one method and band, or (None, None);
    error is all-NaN when the method has no error column (convolved v1)."""
    col = _resolve_flux_column(table, kind, band)
    if col is None:
        return None, None
    flux = np.asarray(table[col], dtype=float)
    if kind == 'catalog':
        err_col = _catalog_flux_columns(band, table.colnames)[1]
    else:
        err_col = f'{col}_err'
    err = (np.abs(np.asarray(table[err_col], dtype=float)) if err_col and err_col in table.colnames
           else np.full(len(table), np.nan))
    return flux, err


def _resolve_flux_column(table, kind, band):
    """kind: 'native' | 'instP' | 'instG' | 'convolved' | 'catalog' -> column
    name, or None if not present in `table`."""
    if kind in ('instP', 'instG'):
        col = f'flux_{kind}_{band}'
    elif kind == 'native':
        col = f'flux_native_{band}'
    elif kind == 'convolved':
        col = f'flux_{band}'
    elif kind == 'catalog':
        col, _ = _catalog_flux_columns(band, table.colnames)
        return col
    else:
        raise ValueError(kind)
    return col if col in table.colnames else None


def _ratio_with_err(table, band, num_kind, den_kind):
    """(ratio, 1-sigma error, valid mask) or None"""
    num, num_e = _flux_and_err(table, num_kind, band)
    den, den_e = _flux_and_err(table, den_kind, band)
    if num is None or den is None:
        return None
    valid = np.isfinite(num) & np.isfinite(den) & (num > 0) & (den > 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        r = num / den
        rel = np.sqrt(np.nan_to_num((num_e / num) ** 2) + np.nan_to_num((den_e / den) ** 2))
        err = np.where(np.isfinite(num_e) | np.isfinite(den_e), r * rel, np.nan)
    return r, err, valid


def plot_flux_ratio(table, bands, num_kind, den_kind, title, out_path):
    values_per_band, classes_per_band, errors_per_band = {}, {}, {}
    classes_all = _class_array(table)
    for band in bands:
        res = _ratio_with_err(table, band, num_kind, den_kind)
        if res is None:
            continue
        r, err, valid = res
        if valid.sum() == 0:
            continue
        values_per_band[band] = r[valid]
        errors_per_band[band] = err[valid]
        classes_per_band[band] = classes_all[valid]

    if not values_per_band:
        print(f'  [skip] {title}: no bands with usable data.')
        return

    bands_present = [b for b in bands if b in values_per_band]
    fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(bands_present)), 6))
    _box_strip_by_band(ax, bands_present, values_per_band, classes_per_band,
                        f'{KIND_LABEL[num_kind]} / {KIND_LABEL[den_kind]} flux ratio',
                        errors_per_band=errors_per_band)
    ax.axhline(1.0, color='crimson', linestyle='--', linewidth=1.5, zorder=1)
    for i, band in enumerate(bands_present, start=1):
        med = np.median(values_per_band[band])
        ax.text(i, 0.985, f'{med:.2f}', ha='center', va='top', fontsize=8,
                transform=ax.get_xaxis_transform(), bbox=dict(fc='white', ec='none', alpha=0.8, pad=1))
    no_err = [k for k in (num_kind, den_kind) if k == 'convolved']
    err_note = ('1σ errors: ' + ('both in quadrature' if not no_err else
                f'{KIND_LABEL[den_kind if num_kind == "convolved" else num_kind]} only '
                f'(convolved v1 has no error column)'))
    ax.set_title(f'{title}\n(median per band at the top; {err_note})', fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f'\n{title}')
    for band in bands_present:
        r = values_per_band[band]
        print(f'  {band:8s} N={len(r):5d}  median={np.median(r):.3f}  '
              f'IQR={np.percentile(r, 75) - np.percentile(r, 25):.3f}')


# ─────────────────────────────────────────────────────────────────────────
# 5. Color-color diagrams
# ─────────────────────────────────────────────────────────────────────────

def _native_color(table, band_blue, band_red, kind='native'):
    """-2.5*log10(f_blue/f_red) from one method's fluxes (no zero point
    needed: it cancels in the ratio); returns (color, valid)"""
    color, err, valid = _color_with_err(table, band_blue, band_red, kind)
    return color, valid


def _color_with_err(table, band_blue, band_red, kind='native'):
    """(color [mag], 1-sigma error [mag], valid) from flux_<kind>_<band>[_err]"""
    n = len(table)
    cb, cr = f'flux_{kind}_{band_blue}', f'flux_{kind}_{band_red}'
    if cb not in table.colnames or cr not in table.colnames:
        return np.full(n, np.nan), np.full(n, np.nan), np.zeros(n, bool)
    fb, fr = np.asarray(table[cb], dtype=float), np.asarray(table[cr], dtype=float)
    eb = np.abs(np.asarray(table[f'{cb}_err'], dtype=float)) if f'{cb}_err' in table.colnames else np.full(n, np.nan)
    er = np.abs(np.asarray(table[f'{cr}_err'], dtype=float)) if f'{cr}_err' in table.colnames else np.full(n, np.nan)
    valid = np.isfinite(fb) & np.isfinite(fr) & (fb > 0) & (fr > 0)
    color = np.full(n, np.nan)
    err = np.full(n, np.nan)
    color[valid] = -2.5 * np.log10(fb[valid] / fr[valid])
    err[valid] = 2.5 / np.log(10) * np.hypot(eb[valid] / fb[valid], er[valid] / fr[valid])
    return color, err, valid


def plot_color_color(table, y_bands, x_bands, title, out_path, note=None, kind='native'):
    y_color, y_err, y_valid = _color_with_err(table, *y_bands, kind=kind)
    x_color, x_err, x_valid = _color_with_err(table, *x_bands, kind=kind)
    valid = y_valid & x_valid
    classes = _class_array(table)
    n_total = int(valid.sum())

    fig, ax = plt.subplots(figsize=(7, 6.5))
    for c in CLASSES:
        m = valid & (classes == c)
        ax.errorbar(x_color[m], y_color[m], xerr=x_err[m], yerr=y_err[m], fmt='none',
                    ecolor=CLASS_COLORS[c], elinewidth=0.5, alpha=0.3, zorder=1)
        ax.scatter(x_color[m], y_color[m], s=18, alpha=0.7, color=CLASS_COLORS[c],
                   label=f'{c} (N={int(m.sum())})', edgecolors='none', zorder=2)
    ax.set_xlabel(f'{x_bands[0]} - {x_bands[1]}  [mag]')
    ax.set_ylabel(f'{y_bands[0]} - {y_bands[1]}  [mag]')
    full_title = f'{title}  (N={n_total} total; 1σ errors)'
    if note:
        full_title += f'\n{note}'
    ax.set_title(full_title)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

    per_class = ', '.join(f'{c}={int((valid & (classes == c)).sum())}' for c in CLASSES)
    print(f'\n{title}: N={n_total}  ({per_class})')


def plot_color_color_grid(table, y_bands, x_bands, title, out_path, note=None, kind='native'):
    """Same color-color diagram as plot_color_color, but as a grid of
    small subplots — one per galaxy — sharing one x/y range so galaxies
    are directly comparable at a glance."""
    y_color, y_err, y_valid = _color_with_err(table, *y_bands, kind=kind)
    x_color, x_err, x_valid = _color_with_err(table, *x_bands, kind=kind)
    valid = y_valid & x_valid
    classes = _class_array(table)
    galaxies_col = np.asarray(table['PHANGS_GALAXY']).astype(str)

    galaxies = sorted({g for g in galaxies_col[valid]})
    if not galaxies:
        print(f'  [skip] {title} (per-galaxy grid): no usable data.')
        return

    # Shared axis range across every panel, from the full pooled sample,
    # with a little padding — so panel-to-panel comparisons are honest.
    pad_x = 0.05 * (x_color[valid].max() - x_color[valid].min() or 1.0)
    pad_y = 0.05 * (y_color[valid].max() - y_color[valid].min() or 1.0)
    xlim = (x_color[valid].min() - pad_x, x_color[valid].max() + pad_x)
    ylim = (y_color[valid].min() - pad_y, y_color[valid].max() + pad_y)

    ncols = min(6, len(galaxies))
    nrows = int(np.ceil(len(galaxies) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.6 * nrows),
                              squeeze=False, sharex=True, sharey=True)

    for i, gal in enumerate(galaxies):
        ax = axes[i // ncols][i % ncols]
        gal_mask = valid & (galaxies_col == gal)
        for c in CLASSES:
            m = gal_mask & (classes == c)
            if m.any():
                ax.errorbar(x_color[m], y_color[m], xerr=x_err[m], yerr=y_err[m], fmt='none',
                            ecolor=CLASS_COLORS[c], elinewidth=0.4, alpha=0.3, zorder=1)
                ax.scatter(x_color[m], y_color[m], s=14, alpha=0.75,
                           color=CLASS_COLORS[c], edgecolors='none', zorder=2)
        ax.set_title(f'{gal}  (N={int(gal_mask.sum())})', fontsize=8.5)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2)
    for j in range(len(galaxies), nrows * ncols):
        axes[j // ncols][j % ncols].axis('off')

    axes[0][0].invert_yaxis()  # shared y-axis: inverting one inverts all

    handles = [plt.Line2D([], [], marker='o', linestyle='', color=CLASS_COLORS[c], label=c)
               for c in CLASSES]
    fig.legend(handles=handles, loc='upper right', fontsize=9, ncol=len(CLASSES))

    full_title = f'{title} — per galaxy'
    if note:
        full_title += f'\n{note}'
    fig.suptitle(full_title, fontsize=12, y=1.0)
    fig.supxlabel(f'{x_bands[0]} - {x_bands[1]}  [mag]', fontsize=10)
    fig.supylabel(f'{y_bands[0]} - {y_bands[1]}  [mag]', fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'  [{len(galaxies)} galaxy panels] {out_path}')


# ─────────────────────────────────────────────────────────────────────────
# 6. Detection-outcome breakdown per band
# ─────────────────────────────────────────────────────────────────────────

STATUS_ORDER = ['ok', 'size_extrapolated', 'borrowed', 'psf_fallback', 'no_detection']
STATUS_COLORS = {
    'ok': 'seagreen', 'size_extrapolated': 'gold', 'borrowed': 'steelblue',
    'psf_fallback': 'darkorange', 'no_detection': 'crimson', 'other': 'grey',
}


def plot_detection_breakdown(table, bands, out_path):
    bands_present = [b for b in bands if f'status_native_{b}' in table.colnames]
    if not bands_present:
        print('  [skip] detection-breakdown plot: no status_native_{band} columns found.')
        return

    fractions = {s: [] for s in STATUS_ORDER + ['other']}
    for band in bands_present:
        status = np.asarray(table[f'status_native_{band}']).astype(str)
        n = len(status)
        for s in STATUS_ORDER:
            fractions[s].append(100.0 * np.sum(status == s) / n if n else 0.0)
        known = np.isin(status, STATUS_ORDER)
        fractions['other'].append(100.0 * np.sum(~known) / n if n else 0.0)

    positions = np.arange(1, len(bands_present) + 1)
    fig, ax = plt.subplots(figsize=(max(9, 0.9 * len(bands_present)), 6))
    bottom = np.zeros(len(bands_present))
    for s in STATUS_ORDER + ['other']:
        vals = np.array(fractions[s])
        ax.bar(positions, vals, bottom=bottom, color=STATUS_COLORS[s], label=s,
               edgecolor='black', linewidth=0.3)
        bottom += vals

    ax.set_xticks(positions)
    ax.set_xticklabels(bands_present, rotation=45, ha='right')
    ax.set_ylabel('% of sources')
    ax.set_ylim(0, 100)
    ax.set_title('Native-photometry status breakdown per band')
    ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

    print('\nDetection-outcome breakdown per band (%):')
    print('band'.ljust(8) + ''.join(s.ljust(18) for s in STATUS_ORDER + ['other']))
    for i, band in enumerate(bands_present):
        print(band.ljust(8) + ''.join(f'{fractions[s][i]:6.1f}'.ljust(18)
                                       for s in STATUS_ORDER + ['other']))


# ─────────────────────────────────────────────────────────────────────────
# 7. Recentering-shift distribution per band
# ─────────────────────────────────────────────────────────────────────────

def plot_recenter_shift(table, bands, out_path):
    values_per_band, flagged_per_band = {}, {}
    for band in bands:
        shift_col = f'recenter_shift_native_{band}'
        src_col = f'aperture_source_native_{band}'
        flag_col = f'recenter_flag_native_{band}'
        if shift_col not in table.colnames:
            continue
        shift = np.asarray(table[shift_col], dtype=float)
        src = (np.asarray(table[src_col]).astype(str) if src_col in table.colnames
               else np.full(len(table), 'own'))
        flag = (np.asarray(table[flag_col], dtype=bool) if flag_col in table.colnames
                else np.zeros(len(table), dtype=bool))
        valid = np.isfinite(shift) & (src == 'own')
        if valid.sum() == 0:
            continue
        values_per_band[band] = shift[valid]
        flagged_per_band[band] = flag[valid]

    if not values_per_band:
        print('  [skip] recentering-shift plot: no usable data.')
        return

    bands_present = [b for b in bands if b in values_per_band]
    positions = np.arange(1, len(bands_present) + 1)
    fig, ax = plt.subplots(figsize=(max(9, 0.9 * len(bands_present)), 6))
    data = [values_per_band[b] for b in bands_present]
    ax.boxplot(data, positions=positions, widths=0.6, showfliers=False,
               patch_artist=True, boxprops=dict(facecolor='steelblue', alpha=0.5))
    rng = _rng()
    for band, pos in zip(bands_present, positions):
        vals = values_per_band[band]
        flagged = flagged_per_band[band]
        jitter = rng.normal(0, 0.06, size=len(vals))
        not_flagged = ~flagged
        ax.scatter((pos + jitter)[not_flagged], vals[not_flagged], s=8, color='navy',
                   alpha=0.4, zorder=3)
        if flagged.any():
            ax.scatter((pos + jitter)[flagged], vals[flagged], s=22, color='darkorange',
                       marker='x', linewidths=1.3, zorder=5,
                       label='recenter_flag=True' if pos == positions[0] else None)
    ax.set_xticks(positions)
    ax.set_xticklabels(bands_present, rotation=45, ha='right')
    ax.set_ylabel('Recentering shift [arcsec]  (own-band measurements)')
    ax.set_title('2D-Gaussian recentering shift from catalog position, per band')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

    print('\nRecentering shift per band (arcsec, own-band only):')
    for band in bands_present:
        v = values_per_band[band]
        print(f'  {band:8s} N={len(v):5d}  median={np.median(v):.4f}  '
              f'95th%={np.percentile(v, 95):.4f}  n_flagged={int(flagged_per_band[band].sum())}')


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def plot_ratio_per_galaxy(table, bands, num_kind, den_kind, title, out_path):
    """Median num/den per galaxy and band, bars = 16-84th percentile of the
    galaxy's clusters (the per-source 1-sigma errors are in the pooled plot)"""
    gal_col = np.asarray(table['PHANGS_GALAXY']).astype(str)
    galaxies = sorted(set(gal_col))
    fig, ax = plt.subplots(figsize=(max(9, 0.75 * len(galaxies) + 2), 5.6))
    width = 0.8 / len(bands)
    rows = []
    for k, (band, color) in enumerate(zip(bands, BAND_COLORS)):
        res = _ratio_with_err(table, band, num_kind, den_kind)
        if res is None:
            continue
        r, _, valid = res
        xs, med, lo, hi = [], [], [], []
        for i, g in enumerate(galaxies):
            v = r[valid & (gal_col == g)]
            if v.size == 0:
                continue
            p16, p50, p84 = np.percentile(v, [16, 50, 84])
            xs.append(i + (k - (len(bands) - 1) / 2) * width)
            med.append(p50); lo.append(p50 - p16); hi.append(p84 - p50)
            rows.append((g, band, v.size, p50))
        ax.errorbar(xs, med, yerr=[lo, hi], fmt='o', ms=5, color=color, mec='white', mew=0.6,
                    elinewidth=1.4, capsize=0, label=band, zorder=3)
    ax.axhline(1.0, color='crimson', linestyle='--', linewidth=1.3, zorder=1)
    ax.set_yscale('log')
    ax.set_xticks(np.arange(len(galaxies)))
    ax.set_xticklabels([g.upper() for g in galaxies], rotation=45, ha='right')
    ax.set_ylabel(f'{KIND_LABEL[num_kind]} / {KIND_LABEL[den_kind]} flux ratio')
    ax.set_title(f'{title}\n(median per galaxy; bars: 16–84% of its clusters)', fontsize=11)
    ax.grid(alpha=0.25, axis='y', which='both')
    ax.legend(ncol=len(bands), fontsize=9, loc='upper left')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'\n{title}: median per galaxy written ({len(rows)} galaxy-band cells)')


def plot_reff_instp(table, bands, out_path):
    """instP R_eff [pc] per band, own fits only (borrowed sizes and point
    sources left out), by class"""
    values, classes = {}, {}
    cls = _class_array(table)
    for band in bands:
        rc, sc = f'reff_pc_instP_{band}', f'size_source_instP_{band}'
        if rc not in table.colnames or sc not in table.colnames:
            continue
        r = np.asarray(table[rc], dtype=float)
        own = np.char.strip(np.asarray(table[sc]).astype(str)) == 'own'
        ok = own & np.isfinite(r) & (r > 0)
        if ok.any():
            values[band], classes[band] = r[ok], cls[ok]
    if not values:
        print('  [skip] instP R_eff: no reff_pc_instP columns.')
        return
    present = [b for b in bands if b in values]
    fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(present)), 6))
    _box_strip_by_band(ax, present, values, classes, 'instP R_eff  [pc]')
    ax.axhline(1.0, color='grey', linestyle=':', linewidth=1.2, zorder=1)
    ax.axhline(20.0, color='grey', linestyle='--', linewidth=1.0, zorder=1)
    for i, band in enumerate(present, start=1):
        ax.text(i, 0.985, f'{np.median(values[band]):.1f}', ha='center', va='top', fontsize=8,
                transform=ax.get_xaxis_transform(), bbox=dict(fc='white', ec='none', alpha=0.8, pad=1))
    ax.set_title('instP half-light radius per band (own fits; median on top)\n'
                 'dotted: 1 pc; dashed: the 20 pc fit cap', fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print('\ninstP R_eff [pc], own fits: ' + ', '.join(
        f'{b} {np.median(values[b]):.2f} (N={len(values[b])})' for b in present))


def plot_spectra_vs_photometry(output_dir, galaxies, out_path):
    """Synthetic F555W / F814W photometry from the MUSE spectra divided by the
    imaging, per cluster and galaxy: M2 (PSF fit, total light) / instP and
    M1 (annulus, corrected to total with EE_AP) / native — the same pairs as
    run_muse_spectra.plot_photometry_comparison. 1-sigma errors in quadrature.
    Skipped when no galaxy has <GAL>_muse_spectra.fits."""
    pairs = (('M2 / instP', 'C2', 'instP', '#1baf7a'), ('M1/EE$_{ap}$ / native', 'C1', 'native', '#2a78d6'))
    data = {b: {lab: [] for lab, *_ in pairs} for b in ('F555W', 'F814W')}
    used = []
    for gal in galaxies:
        sp = os.path.join(output_dir, gal, f'{gal}_muse_spectra.fits')
        v2p = os.path.join(output_dir, gal, f'{gal}_photometry_v2.fits')
        if not (os.path.exists(sp) and os.path.exists(v2p)):
            continue
        t = Table.read(sp, hdu='SOURCES')
        v2 = Table.read(v2p)
        idx = {int(v): k for k, v in enumerate(v2['ID_PHANGS_CLUSTER'])}
        rows = [(k, idx[int(v)]) for k, v in enumerate(t['ID_PHANGS_CLUSTER']) if int(v) in idx]
        if not rows:
            continue
        used.append(gal)
        ks, ps = map(np.array, zip(*rows))
        ee = np.asarray(t['EE_AP'], dtype=float)[ks]
        for b in data:
            for lab, comp, kind, _ in pairs:
                sc, fc = f'synth_{b}_{comp}_mJy', f'flux_{kind}_{b}'
                if sc not in t.colnames or fc not in v2.colnames:
                    continue
                corr = 1.0 if comp == 'C2' else 1.0 / ee
                y = np.asarray(t[sc], dtype=float)[ks] * corr
                ye = np.abs(np.asarray(t[f'{sc}_err'], dtype=float)[ks]) * corr
                x = np.asarray(v2[fc], dtype=float)[ps]
                xe = (np.abs(np.asarray(v2[f'{fc}_err'], dtype=float)[ps]) if f'{fc}_err' in v2.colnames
                      else np.full(len(ps), np.nan))
                ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
                r = y[ok] / x[ok]
                re = r * np.hypot(ye[ok] / y[ok], np.nan_to_num(xe[ok] / x[ok]))
                data[b][lab].append((gal, r, re))
    if not used:
        print('  [skip] spectra vs photometry: no <GAL>_muse_spectra.fits found.')
        return
    # bands the spectra cover (the [muse_spectra] wavelength range can miss F814W)
    bands = [b for b in data if any(r.size for lab in data[b] for _, r, _ in data[b][lab])]
    missing = [b for b in data if b not in bands]
    if not bands:
        print('  [skip] spectra vs photometry: no synthetic photometry (filters outside the spectra).')
        return
    fig, axes = plt.subplots(len(bands), 1, figsize=(max(9, 0.75 * len(used) + 2), 4.6 * len(bands) + 0.6),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]
    xpos = {g: i for i, g in enumerate(used)}
    rng = _rng()
    for ax, b in zip(axes, bands):
        for k, (lab, _, _, color) in enumerate(pairs):
            sets = [(gal, r, re) for gal, r, re in data[b][lab] if r.size]
            if not sets:
                continue
            allr = np.concatenate([r for _, r, _ in sets])
            label = f'{lab}: median {np.median(allr):.2f} (N={allr.size})'
            for j, (gal, r, re) in enumerate(sets):
                x0 = xpos[gal] + (k - 0.5) * 0.3 + rng.normal(0, 0.03, r.size)
                lo = np.minimum(np.nan_to_num(re), 0.95 * r)
                ax.errorbar(x0, r, yerr=[lo, np.nan_to_num(re)], fmt='o', ms=3, color=color, alpha=0.6,
                            elinewidth=0.6, label=label if j == 0 else None, zorder=2)
                ax.plot(xpos[gal] + (k - 0.5) * 0.3, np.median(r), '_', ms=16, mew=2.2, color='#1a1a19', zorder=3)
        ax.axhline(1.0, color='crimson', linestyle='--', linewidth=1.2, zorder=1)
        ax.set_yscale('log')
        ax.set_ylabel(f'MUSE synthetic / imaging  {b}')
        ax.grid(alpha=0.25, which='both', axis='y')
        ax.legend(fontsize=8, loc='lower left')
    axes[-1].set_xticks(range(len(used)))
    axes[-1].set_xticklabels(used, rotation=45, ha='right')
    note = (f'; {", ".join(missing)}: outside the wavelength range of the spectra' if missing else '')
    axes[0].set_title('Synthetic photometry from the MUSE spectra vs the imaging, per cluster\n'
                      f'(1σ errors in quadrature; black bars: galaxy medians{note})', fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'\nSpectra vs photometry: {len(used)} galaxies with spectra -> {out_path}')


def make_qa_plots(output_dir, cat_path=None, qa_dir=None, plots='all'):
    """All pooled QA plots of the galaxies under output_dir (those with a
    <GAL>_photometry_v2.fits) into qa_dir (default <output_dir>/qa_plots).
    plots: 'all', or a list / comma-separated string of PLOT_NAMES."""
    qa_dir = qa_dir or os.path.join(output_dir, 'qa_plots')
    os.makedirs(qa_dir, exist_ok=True)
    if isinstance(plots, str):
        selected = PLOT_NAMES if plots.strip() == 'all' else [p.strip() for p in plots.split(',') if p.strip()]
    else:
        selected = list(plots)
    bad = [p for p in selected if p not in PLOT_NAMES]
    if bad:
        raise ValueError(f'unknown QA plots {bad}; choose from {PLOT_NAMES}')

    galaxies = discover_galaxies(output_dir)
    if not galaxies:
        raise FileNotFoundError(
            f'No <GALAXY>_photometry_v2.fits files found under {output_dir} — '
            f'run run_full_pipeline.py first.'
        )
    print(f'Found {len(galaxies)} processed galaxies under {output_dir}: {galaxies}')

    if 'isolation' in selected:
        if cat_path:
            plot_isolation_funnel(output_dir, cat_path, galaxies,
                                  os.path.join(qa_dir, 'isolation_funnel.png'))
        else:
            print('  [skip] isolation funnel: no catalog path.')
    if 'spectra_photometry' in selected:
        plot_spectra_vs_photometry(output_dir, galaxies, os.path.join(qa_dir, 'spectra_vs_photometry.png'))

    remaining = [p for p in selected if p not in ('isolation', 'spectra_photometry')]
    if remaining:
        print('\nLoading pooled photometry table for the remaining plots...')
        table = load_all_photometry(output_dir)
        print(f'Total sources pooled: {len(table)}')

        if 'phot_native_convolved' in selected:
            plot_flux_ratio(table, ALL_BANDS, 'native', 'convolved',
                             'Native / Convolved flux ratio',
                             os.path.join(qa_dir, 'phot_native_vs_convolved.png'))
        if 'phot_native_catalog' in selected:
            plot_flux_ratio(table, HST_BANDS, 'native', 'catalog',
                             'Native / Catalog flux ratio',
                             os.path.join(qa_dir, 'phot_native_vs_catalog.png'))
        if 'phot_convolved_catalog' in selected:
            plot_flux_ratio(table, HST_BANDS, 'convolved', 'catalog',
                             'Convolved / Catalog flux ratio',
                             os.path.join(qa_dir, 'phot_convolved_vs_catalog.png'))
        if 'ccd_optical' in selected:
            plot_color_color(table, ('F336W', 'F438W'), ('F555W', 'F814W'),
                              'Optical color-color (native photometry)',
                              os.path.join(qa_dir, 'ccd_optical.png'))
            plot_color_color_grid(table, ('F336W', 'F438W'), ('F555W', 'F814W'),
                                   'Optical color-color (native photometry)',
                                   os.path.join(qa_dir, 'ccd_optical_per_galaxy.png'))
        if 'ccd_ir_nircam' in selected:
            plot_color_color(table, ('F300M', 'F335M'), ('F200W', 'F360M'),
                              'IR color-color, NIRCam (native photometry)',
                              os.path.join(qa_dir, 'ccd_ir_nircam.png'))
            plot_color_color_grid(table, ('F300M', 'F335M'), ('F200W', 'F360M'),
                                   'IR color-color, NIRCam (native photometry)',
                                   os.path.join(qa_dir, 'ccd_ir_nircam_per_galaxy.png'))
        if 'ccd_ir_miri' in selected:
            plot_color_color(table, ('F335M', 'F770W'), ('F1000W', 'F2100W'),
                              'IR color-color, MIRI-involving (native photometry)',
                              os.path.join(qa_dir, 'ccd_ir_miri.png'),
                              note='Low N expected: MIRI detections are rare in this sample')
            plot_color_color_grid(table, ('F335M', 'F770W'), ('F1000W', 'F2100W'),
                                   'IR color-color, MIRI-involving (native photometry)',
                                   os.path.join(qa_dir, 'ccd_ir_miri_per_galaxy.png'),
                                   note='Low N expected: MIRI detections are rare in this sample')
        if 'phot_instp_catalog' in selected:
            plot_flux_ratio(table, HST_BANDS, 'instP', 'catalog',
                             'instP / Catalog flux ratio',
                             os.path.join(qa_dir, 'phot_instP_vs_catalog.png'))
        if 'phot_instp_native' in selected:
            plot_flux_ratio(table, ALL_BANDS, 'instP', 'native',
                             'instP / Native flux ratio',
                             os.path.join(qa_dir, 'phot_instP_vs_native.png'))
        if 'phot_instp_catalog_galaxy' in selected:
            plot_ratio_per_galaxy(table, HST_BANDS, 'instP', 'catalog',
                                  'instP / Catalog flux ratio per galaxy',
                                  os.path.join(qa_dir, 'phot_instP_vs_catalog_per_galaxy.png'))
        if 'ccd_optical_instp' in selected:
            plot_color_color(table, ('F336W', 'F438W'), ('F555W', 'F814W'),
                             'Optical color-color (instP photometry)',
                             os.path.join(qa_dir, 'ccd_optical_instP.png'), kind='instP')
            plot_color_color_grid(table, ('F336W', 'F438W'), ('F555W', 'F814W'),
                                  'Optical color-color (instP photometry)',
                                  os.path.join(qa_dir, 'ccd_optical_instP_per_galaxy.png'), kind='instP')
        if 'reff_instp' in selected:
            plot_reff_instp(table, ALL_BANDS, os.path.join(qa_dir, 'reff_instP.png'))
        if 'detection' in selected:
            plot_detection_breakdown(table, ALL_BANDS, os.path.join(qa_dir, 'detection_breakdown.png'))
        if 'recenter' in selected:
            plot_recenter_shift(table, ALL_BANDS, os.path.join(qa_dir, 'recenter_shift.png'))

    print(f'\n✓ QA plots written to {qa_dir}/')
    return qa_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR,
                    help='root output directory (default: config.ini output_dir)')
    ap.add_argument('--cat-path', default=DEFAULT_CAT_PATH,
                     help='PHANGS-HST DR5 cluster-catalog directory (default: config.ini cat_path)')
    ap.add_argument('--qa-dir', default=None, help='default: <output-dir>/qa_plots')
    ap.add_argument('--plots', default='all',
                     help=f'comma-separated subset of {PLOT_NAMES}, or "all"')
    args = ap.parse_args()
    make_qa_plots(args.output_dir, args.cat_path, args.qa_dir, args.plots)


if __name__ == '__main__':
    main()
