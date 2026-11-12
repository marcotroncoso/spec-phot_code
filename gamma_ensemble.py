"""
Characteristic EFF power index gamma of a cluster sample, from a run with
[profile] eff_gamma_free = True.

Uses the instP fits of every source and band where gamma is actually
constrained: own fit (size_source_instP == 'own'), converged, R_eff >=
--min-reff-pc (unresolved sources say nothing about gamma), S/N >= --min-snr,
and gamma not at the edge of the allowed range (the number at each edge is
printed: many fits at an edge mean the range, or the model, is the limit). Prints, per
band and for all bands together, the number of fits and the 16/50/84th
percentiles of gamma, and the median per source (median over its bands). The
sample median is the value to put back as [profile] eff_gamma (with
eff_gamma_free = False) for a fixed-gamma rerun.

    python gamma_ensemble.py output_custom/NGC3351/NGC3351_photometry_instrument.fits
    python gamma_ensemble.py <table> --min-snr 20 --bands F555W F814W --plot gamma.png

The table can be <name>_photometry_instrument.fits or <name>_photometry_v2.fits.
"""
import argparse
import re

import numpy as np
from astropy.table import Table


def collect(tbl, bands=None, min_snr=10.0, min_reff_pc=0.5, edge=0.05):
    """dict band -> array of usable gamma, per-source median gamma, the fitted
    range, and dict band -> (n at lower edge, n at upper edge) among the
    otherwise usable fits"""
    lo = hi = None
    if 'profile_instP' in tbl.colnames:
        for tag in set(str(t).strip() for t in tbl['profile_instP']):
            m = re.fullmatch(r'eff([\d.]+)-([\d.]+)', tag)
            if m:
                lo, hi = float(m.group(1)), float(m.group(2))
    all_bands = [c[len('gamma_instP_'):] for c in tbl.colnames if c.startswith('gamma_instP_')]
    use = [b for b in all_bands if bands is None or b in bands]
    per_band, per_source, at_edge = {}, [[] for _ in range(len(tbl))], {}
    for b in use:
        g = np.asarray(tbl[f'gamma_instP_{b}'], float)
        ok = np.isfinite(g)
        ok &= np.char.strip(np.asarray(tbl[f'size_source_instP_{b}']).astype(str)) == 'own'
        ok &= np.asarray(tbl[f'converged_instP_{b}'], bool)
        ok &= np.asarray(tbl[f'reff_instP_{b}'], float) > 0
        if f'reff_pc_instP_{b}' in tbl.colnames:
            rpc = np.asarray(tbl[f'reff_pc_instP_{b}'], float)
            ok &= np.isfinite(rpc) & (rpc >= min_reff_pc)
        snr = np.asarray(tbl[f'snr_instP_{b}'], float)
        ok &= np.isfinite(snr) & (snr >= min_snr)
        if lo is not None:
            at_edge[b] = (int(np.sum(ok & (g <= lo + edge))), int(np.sum(ok & (g >= hi - edge))))
            ok &= (g > lo + edge) & (g < hi - edge)
        per_band[b] = g[ok]
        for i in np.flatnonzero(ok):
            per_source[i].append(g[i])
    src_med = np.array([np.median(v) for v in per_source if v])
    return per_band, src_med, (lo, hi), at_edge


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('table', help='<name>_photometry_instrument.fits or <name>_photometry_v2.fits')
    ap.add_argument('--min-snr', type=float, default=10.0, help='minimum snr_instP of a fit (default 10)')
    ap.add_argument('--min-reff-pc', type=float, default=0.5, help='minimum reff_pc_instP of a fit (default 0.5)')
    ap.add_argument('--bands', nargs='+', default=None, help='only these bands')
    ap.add_argument('--plot', default=None, help='write a histogram PNG here')
    a = ap.parse_args()

    tbl = Table.read(a.table)
    per_band, src_med, (lo, hi), at_edge = collect(tbl, a.bands, a.min_snr, a.min_reff_pc)
    if not per_band:
        raise SystemExit('no gamma_instP_<band> columns: run with [profile] eff_gamma_free = True first')
    if lo is None:
        print('NOTE: profile_instP is not a free-gamma tag (effA-B): these gammas were fixed, not fitted.')
    else:
        print(f'gamma fitted in [{lo}, {hi}]; fits within 0.05 of either edge are left out')
    print(f'{"band":8s} {"N":>4s} {"p16":>6s} {"median":>7s} {"p84":>6s}   at edges (low, high)')
    for b, g in per_band.items():
        e = f'   {at_edge[b]}' if b in at_edge else ''
        if g.size:
            p16, p50, p84 = np.percentile(g, [16, 50, 84])
            print(f'{b:8s} {g.size:4d} {p16:6.2f} {p50:7.2f} {p84:6.2f}{e}')
        else:
            print(f'{b:8s} {0:4d} {"":6s} {"":7s} {"":6s}{e}')
    allg = np.concatenate([g for g in per_band.values()]) if per_band else np.array([])
    if allg.size:
        p16, p50, p84 = np.percentile(allg, [16, 50, 84])
        print(f'{"all":8s} {allg.size:4d} {p16:6.2f} {p50:7.2f} {p84:6.2f}   (all usable fits)')
    if src_med.size:
        p16, p50, p84 = np.percentile(src_med, [16, 50, 84])
        print(f'{"sources":8s} {src_med.size:4d} {p16:6.2f} {p50:7.2f} {p84:6.2f}   (median over bands, per source)')
        print(f'\nsuggested fixed value: [profile] eff_gamma = {p50:.2f}  (eff_gamma_free = False)')

    if a.plot and allg.size:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        bins = np.arange((lo or allg.min()) - 0.125, (hi or allg.max()) + 0.25, 0.25)
        ax.hist(allg, bins=bins, color='#c9c8c2', label=f'all usable fits (N={allg.size})')
        if src_med.size:
            ax.hist(src_med, bins=bins, histtype='step', color='#4a3aa7', lw=1.8,
                    label=f'per-source median (N={src_med.size})')
            ax.axvline(np.median(src_med), color='#4a3aa7', ls='--', lw=1)
        ax.set_xlabel('EFF power index γ  (Moffat β = γ/2)')
        ax.set_ylabel('number')
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(a.plot, dpi=120)
        print(f'histogram -> {a.plot}')


if __name__ == '__main__':
    main()
