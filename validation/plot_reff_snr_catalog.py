"""
Two diagnostic plots of one galaxy's catalogue-mode (or ISF-mode) output:

  reff_vs_snr.png          instP R_eff [pc] against the instP S/N, one panel per band.
                           Own fits filled (by cluster class), sizes borrowed from
                           another band open grey, point sources (R_eff = 0) as
                           triangles on the bottom axis. Running median of own fits.
  instP_minus_catalog.png  (instP - catalogue) / catalogue against the instP S/N for
                           the five HST bands, with 1 sigma errors (instP and
                           catalogue errors in quadrature) and a histogram per band.

Both fluxes are MW-dereddened (instP in native_phot_aux_func; the PHANGS DR5
catalogue fluxes are dereddened already).

  python validation/plot_reff_snr_catalog.py \
      --dir ../data/ALL_MUSE/output_catalog_theoPSF_catalog/NGC3351

Plots go to <dir>/qa_plots/ unless --out is given.
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                     # noqa: E402
import numpy as np                                  # noqa: E402
from astropy.table import Table                     # noqa: E402

BANDS = ['F275W', 'F336W', 'F438W', 'F555W', 'F814W', 'F200W', 'F300M', 'F335M', 'F360M',
         'F770W', 'F1000W', 'F1130W', 'F2100W']
CATALOG_BANDS = {'F275W': 'NUV', 'F336W': 'U', 'F438W': 'B', 'F555W': 'V', 'F814W': 'I'}
MAX_REFF = 20.0                                     # pc, fit cap (MAXREFF in the instrument cache)
INK, MUTED, GRID = '#1f1f1d', '#6b6a63', '#e1e0d9'
BORROWED = '#a3a29b'


def class_info(t):
    """(class label per row, {label: colour}) — human class if present (DR5 human
    catalogue), else the ML class."""
    if 'PHANGS_CLUSTER_CLASS_HUMAN' in t.colnames:
        h = np.asarray(t['PHANGS_CLUSTER_CLASS_HUMAN'], float)
        if np.isfinite(h).any() and np.isin(h[np.isfinite(h)], (1, 2, 3)).all():
            lab = np.array([f'class {int(v)}' if np.isfinite(v) else '?' for v in h])
            return lab, {'class 1': '#2a78d6', 'class 2': '#eb6834', 'class 3': '#1baf7a'}
    lab = np.char.strip(np.asarray(t['PHANGS_CC_CLASS_ML']).astype(str))
    return lab, {'YCL': '#2a78d6', 'MAP': '#eb6834', 'OGCC': '#1baf7a'}


def running_median(x, y, nbin=8, min_n=5):
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0)
    if ok.sum() < 2 * min_n:
        return None
    edges = np.unique(np.quantile(np.log10(x[ok]), np.linspace(0, 1, nbin + 1)))
    xc, yc, lo, hi = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        s = ok & (np.log10(x) >= a) & (np.log10(x) <= b)
        if s.sum() >= min_n:
            xc.append(10**np.median(np.log10(x[s])))
            q = np.percentile(y[s], [16, 50, 84])
            yc.append(q[1]); lo.append(q[0]); hi.append(q[2])
    return (np.array(xc), np.array(yc), np.array(lo), np.array(hi)) if xc else None


def style(ax):
    ax.grid(color=GRID, lw=0.6, which='major')
    ax.tick_params(colors=MUTED, labelsize=8)
    for s in ax.spines.values():
        s.set_color(GRID)


def plot_reff_vs_snr(t, gal, out):
    cls, cols = class_info(t)
    bands = [b for b in BANDS if f'reff_pc_instP_{b}' in t.colnames]
    ncol = 5
    nrow = int(np.ceil(len(bands) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.0 * nrow), sharey=True,
                             constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    ymin, ymax = 0.1, 40
    stats = []
    for ax, b in zip(axes, bands):
        r = np.asarray(t[f'reff_pc_instP_{b}'], float)
        snr = np.asarray(t[f'snr_instP_{b}'], float)
        src = np.char.strip(np.asarray(t[f'size_source_instP_{b}']).astype(str))
        own = (src == 'own') & np.isfinite(r) & (r > 0) & (snr > 0)
        bor = np.char.startswith(src, 'borrowed') & np.isfinite(r) & (r > 0) & (snr > 0)
        pts = ((src == 'point_source') | (np.isfinite(r) & (r <= 0))) & (snr > 0)
        ax.scatter(snr[bor], r[bor], s=14, facecolors='none', edgecolors=BORROWED, lw=0.8, zorder=2,
                   label='size borrowed from another band')
        for c, col in cols.items():
            m = own & (cls == c)
            if m.any():
                ax.scatter(snr[m], r[m], s=14, color=col, alpha=0.75, edgecolors='white', lw=0.4,
                           zorder=3, label=f'own fit, {c}')
        ax.scatter(snr[pts], np.full(pts.sum(), ymin * 1.15), marker='v', s=16, color=MUTED,
                   alpha=0.6, zorder=2, label='point source (R$_{eff}$ = 0)')
        rm = running_median(snr[own], r[own])
        if rm is not None:
            ax.fill_between(rm[0], rm[2], rm[3], color=INK, alpha=0.08, lw=0, zorder=1)
            ax.plot(rm[0], rm[1], color=INK, lw=1.6, zorder=4, label='running median, 16-84%')
        ax.axhline(MAX_REFF, color=MUTED, ls='--', lw=0.9, zorder=1)
        ax.set(xscale='log', yscale='log', ylim=(ymin, ymax))
        med = np.median(r[own]) if own.any() else np.nan
        ax.set_title(f'{b}   own {own.sum()}, borrowed {bor.sum()}, PS {pts.sum()}\n'
                     f'median own R$_{{eff}}$ = {med:.2f} pc', fontsize=8.5, color=INK)
        style(ax)
        stats.append((b, own.sum(), bor.sum(), pts.sum(), med))
    for ax in axes[len(bands):]:
        ax.axis('off')
    for ax in axes[::ncol]:
        ax.set_ylabel('instP R$_{eff}$ [pc]', fontsize=9)
    for ax in axes[:len(bands)][-ncol:]:
        ax.set_xlabel('instP S/N', fontsize=9)
    h, l = axes[3].get_legend_handles_labels()
    by = dict(zip(l, h))
    fig.legend(by.values(), by.keys(), loc='lower right', bbox_to_anchor=(0.995, 0.02), fontsize=8,
               frameon=False, title='dashed: 20 pc fit cap', title_fontsize=8)
    fig.suptitle(f'{gal}: instP half-light radius vs S/N ({len(t)} catalogue clusters)',
                 fontsize=11, color=INK)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print('R_eff vs S/N  (band, own, borrowed, point source, median own R_eff [pc])')
    for s in stats:
        print(f'  {s[0]:<7} {s[1]:4d} {s[2]:4d} {s[3]:4d}  {s[4]:.2f}')


def plot_instp_minus_catalog(t, gal, out, ylim=(-1.0, 2.0)):
    cls, cols = class_info(t)
    bands = [b for b in CATALOG_BANDS if f'flux_instP_{b}' in t.colnames]
    fig, axes = plt.subplots(len(bands), 2, figsize=(11, 2.6 * len(bands)), sharey='row',
                             gridspec_kw={'width_ratios': [4, 1]}, constrained_layout=True)
    stats = []
    for (ax, axh), b in zip(axes, bands):
        k = CATALOG_BANDS[b]
        f = np.asarray(t[f'flux_instP_{b}'], float)
        fe = np.abs(np.asarray(t[f'flux_instP_{b}_err'], float))
        c = np.asarray(t[f'PHANGS_{k}_mJy'], float)
        ce = np.asarray(t[f'PHANGS_{k}_mJy_ERR'], float)
        ce = np.where(ce > 0, ce, np.nan)          # the catalogue error is sometimes negative
        snr = np.asarray(t[f'snr_instP_{b}'], float)
        src = np.char.strip(np.asarray(t[f'size_source_instP_{b}']).astype(str))
        ok = np.isfinite(f) & np.isfinite(c) & (c > 0) & np.isfinite(snr) & (snr > 0)
        d = (f - c) / c
        # 1 sigma of (f - c)/c = (f/c) * sqrt((fe/f)^2 + (ce/c)^2) = sqrt(fe^2 + (f/c)^2 ce^2) / c
        de = np.sqrt(np.nan_to_num(fe**2) + np.nan_to_num((f / c * ce)**2)) / c
        de = np.where(np.isfinite(fe) | np.isfinite(ce), de, np.nan)
        dc = np.clip(d, ylim[0] + 0.02, ylim[1] - 0.02)
        out_lo, out_hi = ok & (d < ylim[0]), ok & (d > ylim[1])
        for cl, col in cols.items():
            for own, mfc in ((True, col), (False, 'white')):
                m = ok & (cls == cl) & ((src == 'own') == own) & ~out_lo & ~out_hi
                if not m.any():
                    continue
                ax.errorbar(snr[m], d[m], de[m], fmt='o', ms=4, color=col, mfc=mfc, mec=col, mew=0.8,
                            elinewidth=0.6, alpha=0.75, capsize=0, zorder=3,
                            label=f'{cl}, ' + ('own size' if own else 'borrowed size / point source'))
        for m, mk in ((out_lo, 'v'), (out_hi, '^')):
            ax.scatter(snr[m], dc[m], marker=mk, s=22, color=MUTED, zorder=3)
        rm = running_median(snr[ok], d[ok])
        if rm is not None:
            ax.fill_between(rm[0], rm[2], rm[3], color=INK, alpha=0.08, lw=0, zorder=1)
            ax.plot(rm[0], rm[1], color=INK, lw=1.6, zorder=4)
        ax.axhline(0, color=MUTED, lw=1.0, zorder=2)
        ax.set(xscale='log', ylim=ylim)
        ax.set_ylabel(f'{b}\n(instP − cat) / cat', fontsize=9)
        med = np.median(d[ok])
        mad = 1.4826 * np.median(np.abs(d[ok] - med))
        ax.text(0.99, 0.95, f'N = {ok.sum()},  median = {med:+.3f},  σ$_{{MAD}}$ = {mad:.3f},  '
                            f'outside the axis: {out_lo.sum()} below, {out_hi.sum()} above',
                transform=ax.transAxes, ha='right', va='top', fontsize=8, color=INK,
                bbox=dict(fc='white', ec='none', alpha=0.8, pad=1.5))
        style(ax)
        axh.hist(dc[ok], bins=np.linspace(*ylim, 41), orientation='horizontal', color='#6250d6',
                 alpha=0.75, edgecolor='white', lw=0.4)
        axh.axhline(0, color=MUTED, lw=1.0)
        axh.axhline(med, color=INK, lw=1.4, ls='--')
        axh.set_xlabel('N', fontsize=8)
        style(axh)
        stats.append((b, ok.sum(), med, mad, np.median(d[ok & (src == 'own')])))
    axes[-1, 0].set_xlabel('instP S/N', fontsize=9)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, frameon=False, loc='outside lower center', ncol=4)
    fig.suptitle(f'{gal}: (instP − catalogue) / catalogue, 1σ errors; black: running median '
                 f'(16–84% band); triangles: off the axis', fontsize=10.5, color=INK)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print('(instP - cat)/cat  (band, N, median, sigma_MAD, median own-size only)')
    for s in stats:
        print(f'  {s[0]:<7} {s[1]:4d}  {s[2]:+.3f}  {s[3]:.3f}  {s[4]:+.3f}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', required=True, help='galaxy output folder with <GAL>_photometry_v2.fits')
    ap.add_argument('--out', default=None, help='output folder (default <dir>/qa_plots)')
    opt = ap.parse_args()
    v2 = sorted(glob.glob(os.path.join(opt.dir, '*_photometry_v2.fits')))
    if not v2:
        raise SystemExit(f'no *_photometry_v2.fits in {opt.dir}')
    t = Table.read(v2[0])
    gal = os.path.basename(v2[0]).split('_')[0]
    out = opt.out or os.path.join(opt.dir, 'qa_plots')
    os.makedirs(out, exist_ok=True)
    plot_reff_vs_snr(t, gal, os.path.join(out, 'reff_vs_snr.png'))
    plot_instp_minus_catalog(t, gal, os.path.join(out, 'instP_minus_catalog.png'))
    print(f'plots → {out}')


if __name__ == '__main__':
    main()
