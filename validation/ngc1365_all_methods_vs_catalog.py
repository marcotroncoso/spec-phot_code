"""
NGC1365, HST bands: every photometry method / PHANGS-HST catalog, per cluster.

  native      pipeline native slits (output/NGC1365/NGC1365_photometry_native.fits)
  instG       instrument PSF, Gaussian source  (..._photometry_instrument.fits)
  instP       instrument PSF, EFF3 / King1.5    (..._photometry_instrument.fits)
  2D fixed    2D model fit, centre fixed        (output_ngc1365_2d/NGC1365_2d_fit.fits)
  2D free     2D model fit, centre free         (idem)
  2D G fixed  2D model fit, Gaussian source, centre fixed (idem)
  2D G free   2D model fit, Gaussian source, centre free  (idem)
  Deger gal   4 px + sky 7-8 px x galaxy AC_V + band offsets
              (output_catalog_comparison/NGC1365_catalog_comparison.fits)
  Deger exact 4 px + sky 7-8 px x k_cluster, the per-cluster correction from the
              instP size (output_deger_ac_vs_size/NGC1365_ac_per_cluster.fits)

All Milky Way dereddened, all in mJy. Error bars 1 sigma; ratio errors add the
method and catalog errors in quadrature. Run the scripts above first; this one
only reads their tables.

Writes output_catalog_comparison/NGC1365_all_methods_vs_catalog.png (ratio per
cluster) and NGC1365_all_methods_summary.png (median ratio per band).
"""
import os

import numpy as np
from astropy.table import Table
from _paths import OUT_ROOT, OUTPUT_DIR  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
GAL = 'NGC1365'
BANDS = ['F275W', 'F336W', 'F438W', 'F555W', 'F814W']
CAT_BAND = {'F275W': 'NUV', 'F336W': 'U', 'F438W': 'B', 'F555W': 'V', 'F814W': 'I'}
OUT = os.path.join(OUT_ROOT, 'output_catalog_comparison')
CLASS_STYLE = {'MAP': ('#2a78d6', 'o'), 'OGCC': ('#eb6834', 's'), 'YCL': ('#1baf7a', 'D'),
               'outside': ('#8a8984', 'v')}
METHOD_COL = {'native': '#5f5e58', 'instG': '#9b59b6', 'instP': '#2a78d6', '2D fixed': '#1baf7a',
              '2D free': '#0b7a52', '2D G fixed': '#7fb8ec', '2D G free': '#1d5aa0', 'Deger gal': '#eb6834', 'Deger exact': '#b8401a'}
INK, INK2, GRID = '#1a1a19', '#5f5e58', '#e4e3dd'


def build_table():
    out_gal = os.path.join(OUTPUT_DIR, GAL)
    nat = Table.read(os.path.join(out_gal, f'{GAL}_photometry_native.fits'))
    ins = Table.read(os.path.join(out_gal, f'{GAL}_photometry_instrument.fits'))
    cc = Table.read(os.path.join(OUT, f'{GAL}_catalog_comparison.fits'))
    f2d = Table.read(os.path.join(OUT_ROOT, 'output_ngc1365_2d', f'{GAL}_2d_fit.fits'))
    acc = Table.read(os.path.join(OUT_ROOT, 'output_deger_ac_vs_size', f'{GAL}_ac_per_cluster.fits'))
    ids = np.asarray(cc['ID_PHANGS_CLUSTER'])
    for other in (f2d, acc):
        assert np.array_equal(np.asarray(other['ID_PHANGS_CLUSTER']), ids)
    assert len(nat) == len(ins) == len(ids)       # pipeline tables share the catalog order

    t = Table()
    t['ID_PHANGS_CLUSTER'] = ids
    t['CLASS'] = [str(c).strip() for c in cc['CLASS'].astype(str)]
    for b in BANDS:
        mw = cc.meta[f'MW_{b}']
        t[f'catalog_{b}'] = cc[f'catalog_{b}']
        t[f'catalog_{b}_err'] = np.abs(cc[f'catalog_{b}_err'])
        t[f'instP_size_{b}'] = [str(v).strip() for v in cc[f'instP_size_source_{b}'].astype(str)]
        for m, flux, err in (('native', nat[f'flux_native_{b}'], nat[f'flux_native_{b}_err']),
                             ('instG', ins[f'flux_instG_{b}'], ins[f'flux_instG_{b}_err']),
                             ('instP', ins[f'flux_instP_{b}'], ins[f'flux_instP_{b}_err']),
                             ('2D fixed', f2d[f'f2d_fixed_{b}'], f2d[f'f2d_fixed_{b}_err']),
                             ('2D free', f2d[f'f2d_free_{b}'], f2d[f'f2d_free_{b}_err']),
                             ('2D G fixed', f2d[f'f2dG_fixed_{b}'], f2d[f'f2dG_fixed_{b}_err']),
                             ('2D G free', f2d[f'f2dG_free_{b}'], f2d[f'f2dG_free_{b}_err']),
                             ('Deger gal', cc[f'deger_{b}'], cc[f'deger_{b}_err']),
                             ('Deger exact', cc[f'f4px_{b}'] * acc[f'k_cluster_{b}'] * mw,
                              cc[f'f4px_{b}_err'] * acc[f'k_cluster_{b}'] * mw)):
            t[f'{m}_{b}'] = np.asarray(flux, float)
            t[f'{m}_{b}_err'] = np.abs(np.asarray(err, float))
    return t


def _style():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                         'ytick.color': INK2, 'text.color': INK, 'axes.linewidth': 0.8})
    return plt


def _methods(t):
    return [m for m in METHOD_COL if f'{m}_{BANDS[0]}' in t.colnames]


def plot_grid(t, out_png, gal=GAL, note=''):
    plt = _style()
    methods = _methods(t)
    big = len(t) > 80                          # many clusters: thinner error bars, smaller markers
    fig, axes = plt.subplots(len(methods), len(BANDS), figsize=(3.2 * len(BANDS), 2.15 * len(methods)),
                             sharex='col', sharey=True)
    cls = np.asarray(t['CLASS'])
    for j, b in enumerate(BANDS):
        x, xe = np.asarray(t[f'catalog_{b}'], float), np.asarray(t[f'catalog_{b}_err'], float)
        own = np.asarray(t[f'instP_size_{b}']) == 'own'
        for i, m in enumerate(methods):
            ax = axes[i, j]
            y, ye = np.asarray(t[f'{m}_{b}'], float), np.asarray(t[f'{m}_{b}_err'], float)
            ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
            ax.grid(color=GRID, lw=0.6)
            ax.set_axisbelow(True)
            ax.axhline(1.0, color=INK2, lw=1, ls='--', zorder=1)
            r = y / x
            rerr = r * np.hypot(ye / y, xe / x)
            for c, (col, mk) in CLASS_STYLE.items():
                for filled in (True, False):          # filled: instP fitted its own size in this band
                    s = ok & (cls == c) & (own == filled)
                    if not s.any():
                        continue
                    ax.errorbar(x[s], r[s], yerr=rerr[s], marker=mk, ls='none', ms=3.5 if big else 5.5,
                                mfc=col if filled else 'white', mec=col, mew=0.8 if big else 1.1, ecolor=col,
                                elinewidth=0.5 if big else 0.9, capsize=0 if big else 1.5,
                                alpha=0.6 if big else 0.95, zorder=3,
                                label=f'{c}' + ('' if filled else ' (borrowed size)'))
            rr = r[ok & np.isfinite(r)]
            med = np.median(rr) if rr.size else np.nan
            ax.axhline(med, color=METHOD_COL[m], lw=1.6, zorder=2)
            ax.text(0.03, 0.95, f'{med:.2f}  [{np.percentile(rr, 16):.2f}-{np.percentile(rr, 84):.2f}]'
                    if rr.size else 'no data', transform=ax.transAxes, va='top', fontsize=8,
                    color=METHOD_COL[m], fontweight='bold')
            ax.set_xscale('log')
            ax.set_ylim(0.2, 2.0)
            if i == 0:
                ax.set_title(f'{b} ({CAT_BAND[b]})', fontsize=11)
            if j == 0:
                ax.set_ylabel(f'{m}\n/ catalog', color=METHOD_COL[m], fontsize=10)
            if i == len(methods) - 1:
                ax.set_xlabel('catalog flux [mJy]')
    hl = {}
    for ax in axes.flat:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            hl.setdefault(lab, h)
    order = [k for c in CLASS_STYLE for k in (c, f'{c} (borrowed size)') if k in hl]
    fig.legend([hl[k] for k in order], order, loc='upper right', ncol=len(order), frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.995, 0.997))
    fig.suptitle(f'{gal}: every method / PHANGS-HST catalog, HST bands, {len(t)} clusters{note}  '
                 f'(median [16-84%], 1$\\sigma$ errors)',
                 x=0.01, ha='left', fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(out_png, dpi=120, facecolor='#fcfcfb')
    plt.close(fig)


def plot_summary(t, out_png, gal=GAL, note=''):
    plt = _style()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.axhline(1.0, color=INK2, lw=1, ls='--')
    methods = _methods(t)
    xs = np.arange(len(BANDS))
    dx = np.linspace(-0.3, 0.3, len(methods))
    for k, m in enumerate(methods):
        med, lo, hi = [], [], []
        for b in BANDS:
            r = np.asarray(t[f'{m}_{b}'], float) / np.asarray(t[f'catalog_{b}'], float)
            r = r[np.isfinite(r) & (r > 0)]
            q = np.percentile(r, [16, 50, 84]) if r.size else [np.nan] * 3
            lo.append(q[0]); med.append(q[1]); hi.append(q[2])
        med, lo, hi = map(np.asarray, (med, lo, hi))
        ax.errorbar(xs + dx[k], med, yerr=[med - lo, hi - med], marker='o', ms=6, lw=1.2, capsize=2.5,
                    color=METHOD_COL[m], mec='white', mew=0.8, label=m)
    ax.set_xticks(xs, [f'{b}\n({CAT_BAND[b]})' for b in BANDS])
    ax.set_ylabel('method / catalog  (median, 16-84%)')
    ax.set_ylim(0.4, 1.2)
    ax.legend(ncol=4, frameon=False, fontsize=9, loc='lower center')
    ax.set_title(f'{gal}: median flux ratio to the PHANGS-HST catalog ({len(t)} clusters{note})', loc='left',
                 fontsize=11)
    plt.tight_layout()
    fig.savefig(out_png, dpi=130, facecolor='#fcfcfb')
    plt.close(fig)


def main():
    t = build_table()
    t.write(os.path.join(OUT, f'{GAL}_all_methods_vs_catalog.fits'), overwrite=True)
    plot_grid(t, os.path.join(OUT, f'{GAL}_all_methods_vs_catalog.png'))
    plot_summary(t, os.path.join(OUT, f'{GAL}_all_methods_summary.png'))
    print('median method / catalog [16-84%]')
    print(f'{"":12s}' + ''.join(f'{b:>20s}' for b in BANDS))
    for m in METHOD_COL:
        row = f'{m:12s}'
        for b in BANDS:
            r = np.asarray(t[f'{m}_{b}'], float) / np.asarray(t[f'catalog_{b}'], float)
            r = r[np.isfinite(r) & (r > 0)]
            row += f'{np.median(r):8.2f} [{np.percentile(r, 16):.2f}-{np.percentile(r, 84):.2f}]'
        print(row)
    print(f'written to {OUT}')


if __name__ == '__main__':
    main()
