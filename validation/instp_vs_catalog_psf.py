"""
instP flux / PHANGS-HST catalog flux, per cluster and HST band, for two runs
of the same catalog that differ only in the instrument PSF: e.g. the old run
with the galaxy-matched PSF (psf_extra_blur = galaxy) and a rerun with the
theoretical PSF (psf_extra_blur = none).

    python instp_vs_catalog_psf.py --galaxy NGC3351 \
        --old <ALL_MUSE>/output_catalog --new <ALL_MUSE>/output_catalog_theoPSF

Reads <dir>/<GAL>/<GAL>_photometry_v2.fits of both runs (catalog columns
PHANGS_<NUV|U|B|V|I>_mJy[_ERR] and flux_instP_<band>[_err]; both Milky-Way
dereddened, mJy), matched by ID_PHANGS_CLUSTER. Error bars: 1 sigma, catalog
and instP errors in quadrature. Writes <validation_output_dir>/
output_catalog_comparison/<GAL>_instP_vs_catalog_psf.png (+ .fits with the
ratios) and prints the median ratios.
"""
import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
from _paths import OUT_ROOT  # noqa: E402

from astropy.table import Table  # noqa: E402

BANDS = ['F275W', 'F336W', 'F438W', 'F555W', 'F814W']
CAT_BAND = {'F275W': 'NUV', 'F336W': 'U', 'F438W': 'B', 'F555W': 'V', 'F814W': 'I'}
OUT = os.path.join(OUT_ROOT, 'output_catalog_comparison')

INK, INK2, MUTED, GRID = '#1a1a19', '#5f5e58', '#9a9890', '#e4e3dd'
C_OLD, C_NEW = '#9a9890', '#2a78d6'      # context (old run) in gray, the new run in blue


def _id_key(v):
    try:
        f = float(v)
        return int(f) if f == int(f) else str(v).strip()
    except (TypeError, ValueError):
        return str(v).strip()


def ratios(v2, band):
    """instP / catalog and its 1-sigma error, all rows (NaN where undefined)"""
    cat = np.asarray(v2[f'PHANGS_{CAT_BAND[band]}_mJy'], float)
    cat_e = np.abs(np.asarray(v2[f'PHANGS_{CAT_BAND[band]}_mJy_ERR'], float))
    f = np.asarray(v2[f'flux_instP_{band}'], float)
    fe = np.abs(np.asarray(v2[f'flux_instP_{band}_err'], float))
    ok = np.isfinite(cat) & (cat > 0) & np.isfinite(f) & (f > 0)
    r = np.where(ok, f / np.where(ok, cat, 1.0), np.nan)
    re = r * np.hypot(fe / np.where(ok, f, 1.0), cat_e / np.where(ok, cat, 1.0))
    return r, re, cat, cat_e


def stats(r):
    r = r[np.isfinite(r) & (r > 0)]
    if not r.size:
        return np.nan, np.nan, np.nan, 0
    p16, p50, p84 = np.percentile(r, [16, 50, 84])
    return p50, p16, p84, r.size


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--galaxy', default='NGC3351')
    ap.add_argument('--old', required=True, help='output root of the reference run')
    ap.add_argument('--new', required=True, help='output root of the new run')
    ap.add_argument('--old-label', default='measured (galaxy-matched) PSF')
    ap.add_argument('--new-label', default='theoretical PSF')
    a = ap.parse_args()
    gal = a.galaxy.upper()

    old = Table.read(os.path.join(a.old, gal, f'{gal}_photometry_v2.fits'))
    new = Table.read(os.path.join(a.new, gal, f'{gal}_photometry_v2.fits'))
    pos = {_id_key(v): i for i, v in enumerate(old['ID_PHANGS_CLUSTER'])}
    idx = np.array([pos.get(_id_key(v), -1) for v in new['ID_PHANGS_CLUSTER']])
    if (idx < 0).any():
        print(f'  {int((idx < 0).sum())} clusters of the new run are not in the old one: left out')
    new = new[idx >= 0]
    old = old[idx[idx >= 0]]

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK2,
                         'xtick.color': INK2, 'ytick.color': INK2, 'axes.titlecolor': INK})

    fig, axes = plt.subplots(2, 3, figsize=(15, 9.2))
    axes = axes.ravel()
    out = Table()
    out['ID_PHANGS_CLUSTER'] = new['ID_PHANGS_CLUSTER']
    summary = {}
    for k, b in enumerate(BANDS):
        ax = axes[k]
        r_o, e_o, cat, _ = ratios(old, b)
        r_n, e_n, _, _ = ratios(new, b)
        out[f'ratio_old_{b}'], out[f'ratio_old_{b}_err'] = r_o, e_o
        out[f'ratio_new_{b}'], out[f'ratio_new_{b}_err'] = r_n, e_n
        out[f'catalog_{b}'] = cat
        summary[b] = (stats(r_o), stats(r_n))
        for r, e, c, z, ms in ((r_o, e_o, C_OLD, 2, 3.4), (r_n, e_n, C_NEW, 3, 4.0)):
            ok = np.isfinite(r) & (r > 0)
            lo = np.clip(e[ok], 0, r[ok] * 0.95)            # keep the lower bar on a log axis
            ax.errorbar(cat[ok], r[ok], yerr=[lo, e[ok]], fmt='o', ms=ms, color=c, mec='white', mew=0.5,
                        ecolor=c, elinewidth=0.6, alpha=0.75, capsize=0, zorder=z)
        ax.axhline(1.0, color=INK, lw=0.9, zorder=1)
        (m_o, *_), (m_n, *_) = summary[b]
        ax.axhline(m_o, color=C_OLD, lw=1.4, ls='--', zorder=1)
        ax.axhline(m_n, color=C_NEW, lw=1.4, ls='--', zorder=1)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(0.15, 6)
        ax.set_yticks([0.2, 0.5, 1, 2, 5])
        ax.set_yticklabels(['0.2', '0.5', '1', '2', '5'])
        ax.grid(True, which='major', color=GRID, lw=0.6, zorder=0)
        ax.set_title(f'{b}  (catalog {CAT_BAND[b]})', fontsize=11, loc='left')
        ax.set_xlabel('catalog flux [mJy]')
        if k % 3 == 0:
            ax.set_ylabel('instP flux / catalog flux')
        (m_o, p16o, p84o, n_o), (m_n, p16n, p84n, n_n) = summary[b]
        ax.text(0.98, 0.04, f'median  {m_o:.2f} → {m_n:.2f}\n'
                            f'16–84%  {p16o:.2f}–{p84o:.2f} → {p16n:.2f}–{p84n:.2f}\n'
                            f'N = {n_o} → {n_n}',
                transform=ax.transAxes, ha='right', va='bottom', fontsize=8.5, color=INK,
                bbox=dict(fc='white', ec=GRID, lw=0.6, alpha=0.9))

    # summary: median and 16-84% per band, old vs new
    ax = axes[5]
    x = np.arange(len(BANDS))
    for dx, j, c, lbl in ((-0.12, 0, C_OLD, a.old_label), (0.12, 1, C_NEW, a.new_label)):
        med = np.array([summary[b][j][0] for b in BANDS])
        lo = np.array([summary[b][j][1] for b in BANDS])
        hi = np.array([summary[b][j][2] for b in BANDS])
        ax.errorbar(x + dx, med, yerr=[med - lo, hi - med], fmt='o', ms=8, color=c, mec='white', mew=0.8,
                    ecolor=c, elinewidth=2, capsize=0, label=lbl, zorder=3)
        for xi, m in zip(x + dx, med):
            ax.text(xi + (0.06 if j else -0.06), m, f'{m:.2f}', ha='left' if j else 'right', va='center',
                    fontsize=8, color=INK)
    ax.axhline(1.0, color=INK, lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(BANDS)
    ax.set_yscale('log')
    ax.set_ylim(0.15, 6)
    ax.set_yticks([0.2, 0.5, 1, 2, 5])
    ax.set_yticklabels(['0.2', '0.5', '1', '2', '5'])
    ax.grid(True, axis='y', color=GRID, lw=0.6, zorder=0)
    ax.set_title('median ratio per band  (bars: 16–84% of clusters)', fontsize=11, loc='left')
    ax.set_xlim(-0.6, len(BANDS) - 0.4)

    handles = [Line2D([], [], marker='o', ls='', color=C_OLD, mec='white', ms=7, label=a.old_label),
               Line2D([], [], marker='o', ls='', color=C_NEW, mec='white', ms=7, label=a.new_label),
               Line2D([], [], ls='--', color=INK2, label='median of each run'),
               Line2D([], [], ls='-', color=INK, label='equal to catalog')]
    fig.legend(handles=handles, loc='upper right', ncol=4, frameon=False, fontsize=9.5, bbox_to_anchor=(0.99, 0.985))
    fig.suptitle(f'{gal}: instP flux / PHANGS-HST catalog flux  ({len(new)} clusters; 1σ errors)',
                 x=0.01, ha='left', y=0.985, fontsize=12.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(OUT, exist_ok=True)
    png = os.path.join(OUT, f'{gal}_instP_vs_catalog_psf.png')
    fig.savefig(png, dpi=130, facecolor='white')
    out.write(os.path.join(OUT, f'{gal}_instP_vs_catalog_psf.fits'), overwrite=True)

    print(f'{gal}: instP / catalog, median [16-84%] (N)')
    for b in BANDS:
        (m_o, a_o, b_o, n_o), (m_n, a_n, b_n, n_n) = summary[b]
        print(f'  {b}: {a.old_label}: {m_o:.2f} [{a_o:.2f}-{b_o:.2f}] ({n_o})   '
              f'{a.new_label}: {m_n:.2f} [{a_n:.2f}-{b_n:.2f}] ({n_n})')
    print(f'-> {png}')


if __name__ == '__main__':
    main()
