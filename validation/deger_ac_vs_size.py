"""
Deger+22 aperture correction for ONE cluster vs the galaxy-average one (NGC1365).

The catalog flux is  F_cat = F4 x k_gal,band  with F4 the 4 px (0.158") flux,
sky median at 7-8 px, and k_gal = 10**(-0.4 (AC_V + offset_band)), AC_V = -0.61
mag (one number for the whole galaxy, from the mean growth curve of bright
isolated clusters; deger_aperture_correction.py).

The correction a single cluster really needs is

    k_cl,band(R_eff) = F_total / F4(model)

with F4(model) measured exactly as on the data (4 px, sky median at 7-8 px) on a
noise-free image of the cluster: intrinsic profile (EFF gamma=3 or King c=1.5)
of size R_eff at 19.57 Mpc, convolved with the band's galaxy-matched instrument
PSF. This is the "Deger, exact correction for this cluster" of
mock_far_neighbours.py. The catalog is then off by  F_cat / F_true = k_gal / k_cl.

Writes <validation_output_dir>/output_deger_ac_vs_size/:
  NGC1365_ac_vs_size.png   k_cl(R_eff) per band vs k_gal; and k_gal / k_cl for the
                           31 real clusters at their own instP R_eff and profile
  NGC1365_ac_vs_size.fits  the curves;  NGC1365_ac_per_cluster.fits/.csv  per cluster
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)
from _paths import OUT_ROOT, OUTPUT_DIR  # noqa: E402

from astropy.table import Table  # noqa: E402

from instrument_psf import instrument_psf_for_image  # noqa: E402
from injection_test import _ap, _sky, R4, R7, R8  # noqa: E402
from deger_aperture_correction import (DEGER22_TABLE2, ac_mag_to_flux_factor, band_ac,  # noqa: E402
                                       flux_factor_to_ac_mag)
from mock_neighbours_test import load_real  # noqa: E402

GAL, DIST = 'NGC1365', 19.57
OUT = os.path.join(OUT_ROOT, 'output_deger_ac_vs_size')
HST = ['F275W', 'F336W', 'F438W', 'F555W', 'F814W']
NIRCAM = ['F200W', 'F300M', 'F335M', 'F360M']
REFF_GRID = np.array([0.0, 0.25, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 13, 16, 20])
PROFILES = {'eff3.0': dict(profile='eff', gamma=3.0, c=1.5), 'king1.5': dict(profile='king', gamma=3.0, c=1.5)}
BAND_COL = {'F275W': '#4a3aa7', 'F336W': '#2a78d6', 'F438W': '#1baf7a', 'F555W': '#eda100', 'F814W': '#eb6834',
            'F200W': '#8a8984', 'F300M': '#8a8984', 'F335M': '#8a8984', 'F360M': '#8a8984'}
INK, INK2, GRID = '#1a1a19', '#5f5e58', '#e4e3dd'


def ac_exact(model, pix, reff_pc, spec):
    """F_total / F4 (sky median 7-8 px) of a noise-free model, unit total flux"""
    hs = int(np.ceil(3.0 / pix))
    kw = dict(flux=1.0)
    if reff_pc > 0:
        kw.update(r_eff_arcsec=reff_pc * 206265.0 / (DIST * 1e6), **spec)
    img = model.render(hs, hs, (2 * hs + 1, 2 * hs + 1), **kw)
    return 1.0 / _ap(img, hs, hs, R4 / pix, _sky(img, hs, hs, R7 / pix, R8 / pix))


def main():
    os.makedirs(OUT, exist_ok=True)
    p = load_real()
    ac_v = DEGER22_TABLE2[GAL][0]
    models = {}
    for b in HST + NIRCAM:
        bl = b.lower()
        store, fw = ((p.hst_images, p.psf_hst) if bl in p.hst_images else (p.jwst_images, p.psf_jwst))
        _, header, wcs = store[bl]
        pix = wcs.proj_plane_pixel_scales()[0].to('arcsec').value
        models[b] = (instrument_psf_for_image(b, header, pix, extra_fwhm_arcsec=0.0)[0], pix)   # theoretical PSF, as the pipeline

    # curves
    rows = []
    for b, (model, pix) in models.items():
        for prof, spec in PROFILES.items():
            for r in REFF_GRID:
                rows.append(dict(band=b, profile=prof, reff_pc=r, k_cluster=ac_exact(model, pix, r, spec)))
    curves = Table(rows=rows)
    curves['ac_cluster_mag'] = flux_factor_to_ac_mag(np.asarray(curves['k_cluster']))
    k_gal = {b: float(ac_mag_to_flux_factor(band_ac(ac_v, b))) for b in HST}
    curves.meta.update(GALAXY=GAL, AC_V=ac_v, **{f'KGAL_{b}': v for b, v in k_gal.items()})
    curves.write(os.path.join(OUT, f'{GAL}_ac_vs_size.fits'), overwrite=True)

    # real clusters at their own instP size (interpolated on the curves)
    ins = Table.read(os.path.join(OUTPUT_DIR, GAL, f'{GAL}_photometry_instrument.fits'))
    nat = Table.read(os.path.join(OUTPUT_DIR, GAL, f'{GAL}_photometry_v2.fits'))
    pc = Table()
    pc['ID_PHANGS_CLUSTER'] = nat['ID_PHANGS_CLUSTER']
    pc['CLASS'] = [str(c).strip() for c in nat['PHANGS_CC_CLASS_ML']]
    pc['profile'] = [str(c).strip() for c in ins['profile_instP']]
    for b in HST + NIRCAM:
        r = np.asarray(ins[f'reff_pc_instP_{b}'], float)
        k = np.full(len(pc), np.nan)
        for prof in PROFILES:
            m = np.asarray(pc['profile']) == prof
            c = curves[(curves['band'] == b) & (curves['profile'] == prof)]
            k[m] = np.interp(r[m], np.asarray(c['reff_pc']), np.asarray(c['k_cluster']), right=np.nan)
        pc[f'reff_pc_{b}'] = r
        pc[f'size_source_{b}'] = [str(s).strip() for s in ins[f'size_source_instP_{b}']]
        pc[f'k_cluster_{b}'] = k
        if b in HST:
            pc[f'k_gal_over_k_cluster_{b}'] = k_gal[b] / k
            pc[f'catalog_over_instP_{b}'] = (np.asarray(nat[f'PHANGS_{dict(F275W="NUV", F336W="U", F438W="B", F555W="V", F814W="I")[b]}_mJy'], float)
                                             / np.asarray(ins[f'flux_instP_{b}'], float))
    for c in pc.colnames:
        if pc[c].dtype.kind == 'f':
            pc[c].format = '.4g'
    pc.write(os.path.join(OUT, f'{GAL}_ac_per_cluster.fits'), overwrite=True)
    pc.write(os.path.join(OUT, f'{GAL}_ac_per_cluster.csv'), overwrite=True)

    # ── report ──
    print(f'{GAL}: AC_V = {ac_v} mag.  Correction factor k = F_total / F4 (sky 7-8 px), EFF gamma=3')
    print('band    galaxy k (AC mag)     exact k for R_eff = ' + '  '.join(f'{r:g}pc' for r in (0, 1, 2, 3, 5, 10))
          + '     R_eff where exact = galaxy')
    for b in HST + NIRCAM:
        c = curves[(curves['band'] == b) & (curves['profile'] == 'eff3.0')]
        kk = [float(np.interp(r, c['reff_pc'], c['k_cluster'])) for r in (0, 1, 2, 3, 5, 10)]
        if b in k_gal:
            r_eq = float(np.interp(k_gal[b], c['k_cluster'], c['reff_pc']))
            print(f'{b}   x{k_gal[b]:.3f} ({band_ac(ac_v, b):+.2f})     ' + '  '.join(f'x{v:.2f}' for v in kk)
                  + f'      {r_eq:.1f} pc')
        else:
            print(f'{b}   (no catalog)          ' + '  '.join(f'x{v:.2f}' for v in kk))
    print('\nreal clusters (own instP fits only): k_gal / k_cluster = catalog / true flux predicted from size alone;'
          ' measured catalog / instP for comparison.   median [16-84%]')
    for b in HST:
        own = np.asarray(pc[f'size_source_{b}']) == 'own'
        x = np.asarray(pc[f'k_gal_over_k_cluster_{b}'], float)[own]
        y = np.asarray(pc[f'catalog_over_instP_{b}'], float)[own]
        rr = np.asarray(pc[f'reff_pc_{b}'], float)[own]
        print(f'{b}  N={own.sum():2d}  R_eff {np.median(rr):4.1f} pc [{np.percentile(rr, 16):.1f}-{np.percentile(rr, 84):.1f}]'
              f'   k_cluster x{np.nanmedian(np.asarray(pc[f"k_cluster_{b}"], float)[own]):.2f}'
              f'   k_gal/k_cluster {np.nanmedian(x):.2f} [{np.nanpercentile(x, 16):.2f}-{np.nanpercentile(x, 84):.2f}]'
              f'   catalog/instP {np.nanmedian(y):.2f} [{np.nanpercentile(y, 16):.2f}-{np.nanpercentile(y, 84):.2f}]')
    plot(curves, pc, k_gal, ac_v)


def plot(curves, pc, k_gal, ac_v):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                         'ytick.color': INK2, 'text.color': INK, 'axes.linewidth': 0.8})
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(17, 5.2), gridspec_kw=dict(width_ratios=[1.25, 1, 1.1]))
    # (a) curves
    for b in HST:
        c = curves[(curves['band'] == b) & (curves['profile'] == 'eff3.0')]
        a1.plot(c['reff_pc'], c['k_cluster'], '-', color=BAND_COL[b], lw=1.8, label=f'{b} exact (EFF3)')
        a1.axhline(k_gal[b], color=BAND_COL[b], lw=1.0, ls='--')
    c = curves[(curves['band'] == 'F555W') & (curves['profile'] == 'king1.5')]
    a1.plot(c['reff_pc'], c['k_cluster'], ':', color=BAND_COL['F555W'], lw=1.5, label='F555W exact (King c=1.5)')
    c = curves[(curves['band'] == 'F335M') & (curves['profile'] == 'eff3.0')]
    a1.plot(c['reff_pc'], c['k_cluster'], '-', color=BAND_COL['F335M'], lw=1.2, label='F335M exact (no catalog)')
    own = np.asarray(pc['size_source_F555W']) == 'own'
    r = np.asarray(pc['reff_pc_F555W'], float)[own]
    a1.plot(r, np.full(r.size, 1.04), '|', color=INK, ms=12, mew=1.2)
    a1.text(0.3, 1.07, 'real clusters, instP R_eff (F555W)', fontsize=8, color=INK)
    a1.set_xscale('symlog', linthresh=1)
    a1.set_xlim(0, 20)
    a1.set_ylim(1.0, 3.0)
    a1.set_xticks([0, 0.5, 1, 2, 3, 5, 10, 20]); a1.set_xticklabels(['0', '0.5', '1', '2', '3', '5', '10', '20'])
    a1.set_xlabel('cluster R_eff [pc]')
    a1.set_ylabel('correction factor  F_total / F_4px')
    sec = a1.secondary_yaxis('right', functions=(lambda k: -2.5 * np.log10(np.clip(k, 1e-3, None)),
                                                 lambda m: 10 ** (-0.4 * m)))
    sec.set_ylabel('AC [mag]')
    a1.set_title(f'(a) exact correction vs size (solid); NGC1365 catalog value (dashed, AC_V = {ac_v})',
                 fontsize=9.5)
    a1.legend(fontsize=7.5, loc='upper left', ncol=2)
    a1.grid(color=GRID, lw=0.6)

    # (b) per-band factors at fixed sizes vs catalog
    bands = HST
    x = np.arange(len(bands))
    for k, (rr, mk) in enumerate([(0, 'v'), (2, 'o'), (5, 's'), (10, '^')]):
        y = [float(np.interp(rr, *[np.asarray(curves[(curves['band'] == b) & (curves['profile'] == 'eff3.0')][c])
                                   for c in ('reff_pc', 'k_cluster')])) for b in bands]
        a2.plot(x, y, mk + '-', color=plt.get_cmap('viridis')(k / 3), ms=7, lw=1.2, label=f'exact, R_eff {rr} pc')
    a2.plot(x, [k_gal[b] for b in bands], 'D--', color=INK, ms=7, lw=1.5, label='NGC1365 catalog (AC_V + band offsets)')
    a2.set_xticks(x); a2.set_xticklabels(bands)
    a2.set_ylabel('correction factor')
    a2.set_title('(b) shape across bands: exact (flat) vs catalog offsets', fontsize=9.5)
    a2.set_ylim(1.15, 2.7)
    a2.legend(fontsize=7.5, loc='upper center', ncol=2)
    a2.grid(color=GRID, lw=0.6)

    # (c) real clusters: predicted catalog/true vs measured catalog/instP
    for b in HST:
        own = np.asarray(pc[f'size_source_{b}']) == 'own'
        xx = np.asarray(pc[f'k_gal_over_k_cluster_{b}'], float)[own]
        yy = np.asarray(pc[f'catalog_over_instP_{b}'], float)[own]
        a3.plot(xx, yy, 'o', ms=5, color=BAND_COL[b], alpha=0.8, mec='white', mew=0.5, label=b)
    a3.plot([0.5, 2.2], [0.5, 2.2], '--', color=INK2, lw=1)
    a3.set_xlim(0.6, 2.2); a3.set_ylim(0.6, 2.2)
    a3.set_xlabel('k_galaxy / k_cluster  (from the fitted size alone)')
    a3.set_ylabel('measured catalog / instP flux')
    a3.set_title('(c) real NGC1365 clusters: does the size explain\nthe catalog/instP offset? (1:1 dashed)',
                 fontsize=9.5)
    a3.legend(fontsize=7.5)
    a3.grid(color=GRID, lw=0.6)
    fig.suptitle(f'{GAL}: Deger+22 aperture correction, galaxy average vs each cluster '
                 f'(4 px aperture, sky 7-8 px; galaxy-matched PSFs, {DIST} Mpc)', x=0.01, ha='left', fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, f'{GAL}_ac_vs_size.png'), dpi=130, facecolor='#fcfcfb', bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
