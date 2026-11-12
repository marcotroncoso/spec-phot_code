"""
Cluster size vs signal-to-noise, per band and all bands combined.

  size  half-light radius R_eff [pc], PSF removed, from one --methods entry:
          instP    reff_pc_instP_<band> (1D slit fit). Only sizes instP fitted in that band
                   ('own') are measurements; point-source fits (R_eff = 0) sit in the bottom
                   strip; borrowed sizes are left out
          fixed, free, Gfixed, Gfree   reff_pc_f2d<mode>_<band> from <gal>_photometry_2d.fits.
                   Converged fits are measurements; non-converged ones (mostly R_eff at the
                   cap) sit in the bottom strip
  S/N   snr_instP_<band> for every method = background-subtracted aperture flux (R65) /
        (bkg_std sqrt(N_pix)), bkg_std = sigma-clipped scatter of the sky annulus
Dashed line: the 20 pc cap.

    python size_vs_snr.py --galaxy NGC3351 --catalog-mode --methods instP fixed

Per cluster: median R_eff and S/N over the bands with an own fit, error bars the 16-84%
spread between bands (all 13 bands, and HST + NIRCam only), against that median S/N or F814W S/N.

Writes <validation_output_dir>/output_catalog_comparison/<gal>[_catalog]_size_vs_snr_<method>_{bands,combined,per_cluster}.png
and <gal>[_catalog]_size_vs_snr_<method>_per_cluster.fits
"""
import argparse
import os
import sys
import warnings

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
from _paths import OUTPUT_DIR, CATALOG_OUTPUT_DIR  # noqa: E402

from astropy.table import Table  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

import ngc1365_all_methods_vs_catalog as amc  # noqa: E402

OUT, CLASS_STYLE, INK, INK2, GRID = amc.OUT, amc.CLASS_STYLE, amc.INK, amc.INK2, amc.GRID
BANDS = ['F275W', 'F336W', 'F438W', 'F555W', 'F814W', 'F200W', 'F300M', 'F335M', 'F360M',
         'F770W', 'F1000W', 'F1130W', 'F2100W']
INSTRUMENT = {'HST': (BANDS[:5], '#2a78d6'), 'NIRCam': (BANDS[5:9], '#1baf7a'), 'MIRI': (BANDS[9:], '#eb6834')}
REFF_CAP = 20.0
Y_UNRES = 0.055                    # where R_eff = 0 points are drawn
YLIM, XLIM = (0.04, 90.0), (2.0, 3000.0)
# method -> (label, bottom-strip label, what the open markers are)
METHODS = {'instP': ('instP', 'unresolved', 'fitted as point source'),
           'fixed': ('2D fixed', 'not converged', '2D fit not converged'),
           'free': ('2D free', 'not converged', '2D fit not converged'),
           'Gfixed': ('2D G fixed', 'not converged', '2D fit not converged'),
           'Gfree': ('2D G free', 'not converged', '2D fit not converged')}


def load(gal, out_dir, method):
    """per band: reff [pc], snr, src = 'own' (a measurement) / 'flagged' (bottom strip) / other"""
    d = os.path.join(out_dir, gal)
    ins = Table.read(os.path.join(d, f'{gal}_photometry_instrument.fits'))
    v2 = Table.read(os.path.join(d, f'{gal}_photometry_v2.fits'))
    assert len(ins) == len(v2)                         # row-aligned, the instrument table has no IDs
    t2d = None
    if method != 'instP':
        t2d = Table.read(os.path.join(d, f'{gal}_photometry_2d.fits'))
        assert np.array_equal(np.asarray(t2d['ID_PHANGS_CLUSTER']), np.asarray(v2['ID_PHANGS_CLUSTER']))
    cls = np.array([str(c).strip() for c in np.asarray(v2['PHANGS_CC_CLASS_ML']).astype(str)])
    data = {}
    for b in BANDS:
        if f'reff_pc_instP_{b}' not in ins.colnames:
            continue
        snr = np.asarray(ins[f'snr_instP_{b}'], float)
        if t2d is None:
            src = np.array([str(s).strip() for s in np.asarray(ins[f'size_source_instP_{b}']).astype(str)])
            src[src == 'point_source'] = 'flagged'
            reff = np.asarray(ins[f'reff_pc_instP_{b}'], float)
        else:
            if f'reff_pc_f2d{method}_{b}' not in t2d.colnames:
                continue
            reff = np.asarray(t2d[f'reff_pc_f2d{method}_{b}'], float)
            conv = np.asarray(t2d[f'converged_f2d{method}_{b}'], bool)
            src = np.where(np.isfinite(reff), np.where(conv, 'own', 'flagged'), 'none')
        data[b] = dict(reff=reff, snr=snr, src=src)
    return data, cls, np.asarray(v2['ID_PHANGS_CLUSTER'])


def running_median(x, y, nmin=8):
    edges = np.logspace(np.log10(XLIM[0]), np.log10(XLIM[1]), 13)
    xc, med = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (x >= lo) & (x < hi)
        if s.sum() >= nmin:
            xc.append(np.sqrt(lo * hi))
            med.append(np.median(y[s]))
    return np.array(xc), np.array(med)


def _frame(ax, strip):
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.axhspan(YLIM[0], 0.075, color=GRID, alpha=0.6, lw=0, zorder=0)
    ax.axhline(REFF_CAP, color=INK2, lw=1, ls='--', zorder=1)
    ax.text(0.985, Y_UNRES, strip, transform=ax.get_yaxis_transform(), ha='right', va='center',
            fontsize=7, color=INK2, zorder=5)


def _scatter(ax, snr, reff, own, unres, col, mk, label, ms):
    ax.plot(snr[own], reff[own], marker=mk, ls='none', ms=ms, mfc=col, mec='white', mew=0.4, alpha=0.75,
            zorder=3, label=label)
    ax.plot(snr[unres], np.full(unres.sum(), Y_UNRES), marker=mk, ls='none', ms=ms, mfc='white', mec=col,
            mew=0.8, alpha=0.75, zorder=3)


def _stats(ax, snr, reff, own, unres, strip):
    rho = spearmanr(snr[own], reff[own])[0] if own.sum() > 3 else np.nan
    med = np.median(reff[own]) if own.any() else np.nan
    ax.text(0.03, 0.97, f'fitted {own.sum()}, {strip} {unres.sum()}\nmedian R$_{{\\rm eff}}$ {med:.2f} pc, '
            f'Spearman $\\rho$ {rho:+.2f}', transform=ax.transAxes, va='top', fontsize=7.5, color=INK2)


def _masks(dd):
    ok = np.isfinite(dd['snr']) & (dd['snr'] > 0) & np.isfinite(dd['reff'])
    return ok & (dd['src'] == 'own') & (dd['reff'] > 0), ok & (dd['src'] == 'flagged')


def plot_bands(data, cls, out_png, gal, note, method):
    label, strip, open_desc = METHODS[method]
    plt = amc._style()
    bands = [b for b in BANDS if b in data]
    ncol = 5
    nrow = int(np.ceil(len(bands) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.0 * nrow), sharex=True, sharey=True)
    for ax, b in zip(axes.flat, bands):
        dd = data[b]
        own, unres = _masks(dd)
        _frame(ax, strip)
        for c, (col, mk) in CLASS_STYLE.items():
            m = cls == c
            _scatter(ax, dd['snr'], dd['reff'], own & m, unres & m, col, mk, c, 4)
        xc, med = running_median(dd['snr'][own], dd['reff'][own])
        ax.plot(xc, med, color=INK, lw=1.8, zorder=4, label='running median (fitted)')
        _stats(ax, dd['snr'], dd['reff'], own, unres, strip)
        ax.set_title(b, fontsize=11)
    for ax in axes.flat[len(bands):]:
        ax.axis('off')
    for ax in axes[-1]:
        ax.set_xlabel('S/N  (aperture flux / $\\sigma_{\\rm sky}\\sqrt{N}$)')
    for ax in axes[:, 0]:
        ax.set_ylabel(f'R$_{{\\rm eff}}$ {label} [pc]')
    hl = {}
    for ax in axes.flat:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            hl.setdefault(lab, h)
    axes.flat[len(bands)].legend(list(hl.values()), list(hl.keys()), loc='center', frameon=False, fontsize=10,
                                 markerscale=1.8)
    fig.suptitle(f'{gal}: {label} half-light radius vs S/N per band, {len(cls)} clusters{note}  '
                 f'(filled: R$_{{\\rm eff}}$ fitted in that band; open: {open_desc}; dashed: '
                 f'{REFF_CAP:.0f} pc cap)', x=0.01, ha='left', fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out_png, dpi=120, facecolor='#fcfcfb')
    plt.close(fig)


def plot_combined(data, cls, out_png, gal, note, method):
    label, strip, _ = METHODS[method]
    plt = amc._style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True)
    pooled = {k: [] for k in ('snr', 'reff', 'own', 'unres', 'cls', 'inst')}
    for inst, (bands, _) in INSTRUMENT.items():
        for b in bands:
            if b not in data:
                continue
            own, unres = _masks(data[b])
            pooled['snr'].append(data[b]['snr'])
            pooled['reff'].append(data[b]['reff'])
            pooled['own'].append(own)
            pooled['unres'].append(unres)
            pooled['cls'].append(cls)
            pooled['inst'].append(np.full(len(cls), inst))
    p = {k: np.concatenate(v) for k, v in pooled.items()}

    ax = axes[0]                                       # coloured by instrument
    _frame(ax, strip)
    for inst, (bands, col) in INSTRUMENT.items():
        m = p['inst'] == inst
        _scatter(ax, p['snr'], p['reff'], p['own'] & m, p['unres'] & m, col, 'o', f'{inst} ({", ".join(bands)})', 3.5)
        xc, med = running_median(p['snr'][p['own'] & m], p['reff'][p['own'] & m])
        ax.plot(xc, med, color=col, lw=2.2, zorder=4)
    ax.set_title('all bands, by instrument (lines: running median of fitted R$_{\\rm eff}$)', loc='left', fontsize=10)

    ax = axes[1]                                       # coloured by class
    _frame(ax, strip)
    for c, (col, mk) in CLASS_STYLE.items():
        m = p['cls'] == c
        _scatter(ax, p['snr'], p['reff'], p['own'] & m, p['unres'] & m, col, mk, c, 3.5)
    xc, med = running_median(p['snr'][p['own']], p['reff'][p['own']])
    ax.plot(xc, med, color=INK, lw=2.2, zorder=4, label='running median (fitted)')
    ax.set_title('all bands, by cluster class', loc='left', fontsize=10)

    for ax in axes:
        _stats(ax, p['snr'], p['reff'], p['own'], p['unres'], strip)
        ax.set_xlabel('S/N  (aperture flux / $\\sigma_{\\rm sky}\\sqrt{N}$)')
        ax.legend(loc='upper right', frameon=True, framealpha=0.9, facecolor='#fcfcfb', edgecolor=GRID, fontsize=8)
    axes[0].set_ylabel(f'R$_{{\\rm eff}}$ {label} [pc]')
    fig.suptitle(f'{gal}: {label} half-light radius vs S/N, all bands combined '
                 f'({len(cls)} clusters x {len(data)} bands{note})', x=0.01, ha='left', fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png, dpi=130, facecolor='#fcfcfb')
    plt.close(fig)


BAND_SETS = {'all': ('all bands', BANDS), 'hstnircam': ('HST + NIRCam', BANDS[:9])}


def per_cluster(data, cls, ids):
    """median and 16/84 percentiles over the bands with a size measurement ('own')"""
    t = Table()
    t['ID_PHANGS_CLUSTER'] = ids
    t['CLASS'] = cls
    for key, (_, bands) in BAND_SETS.items():
        bands = [b for b in bands if b in data]
        own = np.array([_masks(data[b])[0] for b in bands])
        for q in ('reff', 'snr'):
            v = np.where(own, np.array([data[b][q] for b in bands]), np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)          # clusters with no own fit
                lo, med, hi = np.nanpercentile(v, [16, 50, 84], axis=0)
            t[f'{q}_med_{key}'], t[f'{q}_p16_{key}'], t[f'{q}_p84_{key}'] = med, lo, hi
        t[f'n_bands_{key}'] = own.sum(0)
    t['snr_F814W'] = data['F814W']['snr']
    return t


def plot_cluster_median(t, out_png, gal, note, method):
    label = METHODS[method][0]
    plt = amc._style()
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharey=True)
    cls = np.asarray(t['CLASS'])
    for i, (key, (set_label, bands)) in enumerate(BAND_SETS.items()):
        y = np.asarray(t[f'reff_med_{key}'], float)
        yerr = np.array([y - np.asarray(t[f'reff_p16_{key}'], float), np.asarray(t[f'reff_p84_{key}'], float) - y])
        nb = np.asarray(t[f'n_bands_{key}'])
        sx = np.asarray(t[f'snr_med_{key}'], float)
        sxerr = np.array([sx - np.asarray(t[f'snr_p16_{key}'], float), np.asarray(t[f'snr_p84_{key}'], float) - sx])
        for j, (x, xerr, xlab) in enumerate([
                (sx, sxerr, f'median S/N over the same bands (16-84%)'),
                (np.asarray(t['snr_F814W'], float), None, 'S/N in F814W')]):
            ax = axes[i, j]
            ax.grid(color=GRID, lw=0.6)
            ax.set_axisbelow(True)
            ax.axhline(REFF_CAP, color=INK2, lw=1, ls='--', zorder=1)
            ok = (nb > 0) & np.isfinite(x) & (x > 0) & np.isfinite(y)
            for c, (col, mk) in CLASS_STYLE.items():
                m = ok & (cls == c)
                if m.any():
                    ax.errorbar(x[m], y[m], yerr=yerr[:, m], xerr=None if xerr is None else xerr[:, m], marker=mk,
                                ls='none', ms=4.5, mfc=col, mec='white', mew=0.4, ecolor=col, elinewidth=0.6,
                                alpha=0.7, zorder=3, label=c)
            xc, med = running_median(x[ok], y[ok])
            ax.plot(xc, med, color=INK, lw=2, zorder=4, label='running median')
            rho = spearmanr(x[ok], y[ok])[0] if ok.sum() > 3 else np.nan
            ax.text(0.03, 0.97, f'{ok.sum()} clusters, median {np.median(nb[ok]):.0f} bands per cluster\n'
                    f'median R$_{{\\rm eff}}$ {np.median(y[ok]):.2f} pc, Spearman $\\rho$ {rho:+.2f}',
                    transform=ax.transAxes, va='top', fontsize=8.5, color=INK2)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlim(*XLIM)
            ax.set_ylim(0.08, 90)
            ax.set_xlabel(xlab)
            ax.set_title(f'{set_label} ({len(bands)} bands)  vs  ' + ('median S/N' if j == 0 else 'F814W S/N'),
                         loc='left', fontsize=10)
        axes[i, 0].set_ylabel(f'median R$_{{\\rm eff}}$ {label} over bands [pc]  (16-84%)')
    axes[0, 1].legend(loc='upper right', frameon=True, framealpha=0.9, facecolor='#fcfcfb', edgecolor=GRID,
                      fontsize=8.5)
    fig.suptitle(f'{gal}: per-cluster median {label} half-light radius over the bands with a fitted size{note}\n'
                 f'error bars: 16-84% spread between bands; dashed: {REFF_CAP:.0f} pc cap',
                 x=0.01, ha='left', fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(out_png, dpi=120, facecolor='#fcfcfb')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--galaxy', default='NGC3351')
    ap.add_argument('--catalog-mode', action='store_true', help='read output_catalog/ (unfiltered catalog)')
    ap.add_argument('--methods', nargs='+', default=['instP', 'fixed'], choices=list(METHODS),
                    help='size source(s), one set of plots each')
    a = ap.parse_args()
    gal = a.galaxy.upper()
    out_dir = CATALOG_OUTPUT_DIR if a.catalog_mode else OUTPUT_DIR
    tag = f'{gal}_catalog' if a.catalog_mode else gal
    note = ', full catalog' if a.catalog_mode else ''
    os.makedirs(OUT, exist_ok=True)
    for method in a.methods:
        data, cls, ids = load(gal, out_dir, method)
        base = os.path.join(OUT, f'{tag}_size_vs_snr_{method}')
        plot_bands(data, cls, f'{base}_bands.png', gal, note, method)
        plot_combined(data, cls, f'{base}_combined.png', gal, note, method)
        tc = per_cluster(data, cls, ids)
        tc.meta['METHOD'] = method
        tc.write(f'{base}_per_cluster.fits', overwrite=True)
        plot_cluster_median(tc, f'{base}_per_cluster.png', gal, note, method)
        print(f'\n{METHODS[method][0]}')
        print(f'{"band":8s}{"fitted":>8s}{"flagged":>9s}{"med S/N":>9s}{"med Reff":>10s}{"rho":>7s}')
        for b, dd in data.items():
            own, unres = _masks(dd)
            rho = spearmanr(dd['snr'][own], dd['reff'][own])[0] if own.sum() > 3 else np.nan
            print(f'{b:8s}{own.sum():8d}{unres.sum():9d}{np.median(dd["snr"][own]):9.1f}'
                  f'{np.median(dd["reff"][own]):10.2f}{rho:7.2f}')
        for key in BAND_SETS:
            nb = np.asarray(tc[f'n_bands_{key}'])
            print(f'per cluster ({key}): {(nb > 0).sum()} clusters, median R_eff '
                  f'{np.nanmedian(tc[f"reff_med_{key}"]):.2f} pc, median {np.median(nb[nb > 0]):.0f} bands')
    print(f'written to {OUT}')


if __name__ == '__main__':
    main()
