"""
NGC1365: flux vs filter, with 1-sigma error bars, for the 3 Deger+22 clusters
that are also in our 31-source selection (Deger #450, #451, #454 = our 22, 23, 29).

Every series is measured at the Deger+22 position (Table A1) on the HST images:

  catalog        PHANGS DR5 PHANGS_<band>_mJy, error PHANGS_<band>_mJy_ERR
  Deger recipe   4 px flux (sky 7-8 px) x 10**(-0.4 (AC_V + offset_band)) x MW,
                 AC_V = -0.61 mag (deger_aperture_correction.py)
  growth20       own light within 20 px (sky 21-23 px) x MW
  native / instG / instP   pipeline functions (measure_*_one_source)

Errors (all the same definition, as in the pipeline):
  sigma_F = sigma_sky sqrt(N_ap) x (the correction applied to the flux) x MW
with sigma_sky the 3-sigma-clipped std of the sky annulus and N_ap the aperture
pixels (deger_aperture_correction.aperture_flux_err; native_phot_aux_func.
_aperture_photometry). In the ratio panel the error bar is the method's own
error divided by the catalog flux; the grey band is the catalog's own error.

Writes output_injection/NGC1365_deger_overlap_sed.png and
       output_injection/NGC1365_deger_overlap.fits/.csv
Run:   python deger_overlap_sed.py           (re-measure, ~ several min: pipeline setup)
       python deger_overlap_sed.py --plot    (re-plot from the saved table)
"""
import argparse
import logging
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
from _paths import OUT_ROOT, CAT_PATH  # noqa: E402

import astropy.units as u  # noqa: E402
from astropy.coordinates import SkyCoord  # noqa: E402
from astropy.table import Table, vstack  # noqa: E402

from deger_aperture_correction import (DEGER22_TABLE2, CATALOG_BAND, ac_mag_to_flux_factor,  # noqa: E402
                                       band_ac, aperture_flux_err)

OUT = os.path.join(OUT_ROOT, 'output_injection')
TABLE = os.path.join(OUT, 'NGC1365_deger_overlap')
BANDS = ['F275W', 'F336W', 'F438W', 'F555W', 'F814W']
WL = {'F275W': 0.2742, 'F336W': 0.3366, 'F438W': 0.4335, 'F555W': 0.5303, 'F814W': 0.7983}
DR5 = CAT_PATH
SERIES = [  # column, label, color (validated default categorical order), marker
    ('catalog', 'PHANGS DR5 catalog', '#2a78d6', 'o'),
    ('deger', 'Deger+22 recipe (re-measured: 4 px x galaxy correction)', '#eb6834', 's'),
    ('growth20', 'own light within 20 px (Deger definition, per cluster)', '#1baf7a', '^'),
    ('native', 'ours: native', '#eda100', 'D'),
    ('instG', 'ours: PSF x Gaussian', '#e87ba4', 'v'),
    ('instP', 'ours: PSF x EFF3 (instP)', '#4a3aa7', 'P'),
]


def measure():
    import injection_test as it
    from native_phot_aux_func import measure_native_photometry_one_source, measure_instrument_photometry_one_source
    from phot_aux_func import corr_flux

    p, gal = it.setup('NGC1365-1.15asec')
    logging.getLogger('analysis').setLevel(logging.ERROR)
    tbl = p.photometry_table
    bd = {b: dict(p.native_band_calib[b.lower()], psf_model=p.instrument_psf[b.lower()]) for b in BANDS}
    mw = {b: float(corr_flux(1.0, bd[b]['r_value'], p.eb_v)) for b in BANDS}
    dist = float(np.nanmedian(np.asarray(tbl['PHANGS_DMPC'], float)))
    ac_v = DEGER22_TABLE2['NGC1365'][0]

    d = Table.read(os.path.join(OUT, 'deger22_tablea1.fits'))
    d = d[d['Galaxy'] == 'NGC 1365']
    dsky = SkyCoord(np.asarray(d['RAJ2000'], float) * u.deg, np.asarray(d['DEJ2000'], float) * u.deg)
    osky = SkyCoord(np.asarray(tbl['PHANGS_RA'], float) * u.deg, np.asarray(tbl['PHANGS_DEC'], float) * u.deg)
    oidx, osep, _ = dsky.match_to_catalog_sky(osky)
    keep = np.where(osep.arcsec < 0.15)[0]

    cats = []
    for f in ['human-class1+2', 'machine-class1+2', 'human-class3', 'machine-class3']:
        c = Table.read(DR5 + f'hlsp_phangs-cat_hst_acs-uvis_dr5-targets_v1_{f}-primary.fits')
        c = c[np.char.strip(np.asarray(c['PHANGS_GALAXY']).astype(str)) == 'ngc1365']
        cols = [f'PHANGS_{x}_mJy' for x in CATALOG_BAND.values()] + [f'PHANGS_{x}_mJy_ERR' for x in CATALOG_BAND.values()]
        cats.append(c[['PHANGS_RA', 'PHANGS_DEC'] + cols])
    cat = vstack(cats)
    csky = SkyCoord(np.asarray(cat['PHANGS_RA'], float) * u.deg, np.asarray(cat['PHANGS_DEC'], float) * u.deg)
    cidx, csep, _ = dsky.match_to_catalog_sky(csky)

    rows = []
    for i in keep:
        ra, dec = float(dsky[i].ra.deg), float(dsky[i].dec.deg)
        nat = measure_native_photometry_one_source(ra, dec, {b.lower(): bd[b] for b in BANDS}, eb_v=p.eb_v)
        nr = {bl: dict(aperture_source=r.aperture_source, centroid_ra=r.centroid_ra, centroid_dec=r.centroid_dec)
              for bl, r in nat.items()}
        ins = measure_instrument_photometry_one_source(ra, dec, nr, {b.lower(): bd[b] for b in BANDS}, it.EFF3,
                                                       distance_mpc=dist, eb_v=p.eb_v)
        for b in BANDS:
            B, bl, cb = bd[b], b.lower(), CATALOG_BAND[b]
            pix = B['pixel_scale_arcsec']
            x, y = (float(v) for v in dsky[i].to_pixel(B['wcs']))
            img = B['image']
            f4 = it.to_mjy(it.catalog_style(img, x, y, pix), B)
            e4 = it.to_mjy(aperture_flux_err(img, x, y, it.R4 / pix, it.R7 / pix, it.R8 / pix)[0], B)
            g20 = it.to_mjy(it.growth20(img, x, y, pix), B)
            e20 = it.to_mjy(aperture_flux_err(img, x, y, it.R20 / pix, it.R21 / pix, it.R23 / pix)[0], B)
            k = float(ac_mag_to_flux_factor(band_ac(ac_v, b)))
            rows.append(dict(
                recno=int(d['recno'][i]), ours_idx=int(oidx[i]), band=b, wl_um=WL[b],
                catalog=float(cat[f'PHANGS_{cb}_mJy'][cidx[i]]), catalog_err=float(cat[f'PHANGS_{cb}_mJy_ERR'][cidx[i]]),
                f4px=f4 * mw[b], f4px_err=e4 * mw[b],
                deger=f4 * k * mw[b], deger_err=e4 * k * mw[b],
                growth20=g20 * mw[b], growth20_err=e20 * mw[b],
                native=nat[bl].flux_mJy, native_err=nat[bl].flux_err_mJy,
                instG=ins[bl]['instG'].flux_mJy, instG_err=ins[bl]['instG'].flux_err_mJy,
                instP=ins[bl]['instP'].flux_mJy, instP_err=ins[bl]['instP'].flux_err_mJy,
                instP_reff_pc=ins[bl]['instP'].r_eff_pc, ac_mag=band_ac(ac_v, b), mw=mw[b]))
        print(f'Deger #{int(d["recno"][i])} = our source {int(oidx[i])} done', flush=True)
    t = Table(rows=rows)
    t.meta.update(GALAXY='NGC1365', EBV=float(p.eb_v), AC_V=ac_v)
    for c in t.colnames:
        if t[c].dtype.kind == 'f':
            t[c].format = '.6g'
    t.write(TABLE + '.fits', overwrite=True)
    t.write(TABLE + '.csv', overwrite=True)
    return t


def plot(t):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    recs = list(dict.fromkeys(int(r) for r in t['recno']))
    fig, axes = plt.subplots(2, len(recs), figsize=(5 * len(recs), 7.5), sharex=True,
                             gridspec_kw=dict(height_ratios=[2.2, 1], hspace=0.06, wspace=0.25))
    x = np.array([WL[b] for b in BANDS])
    for k, rec in enumerate(recs):
        s = {r['band']: r for r in t[t['recno'] == rec]}
        ours = int(s[BANDS[0]]['ours_idx'])
        cat = np.array([s[b]['catalog'] for b in BANDS], float)
        cat_e = np.array([s[b]['catalog_err'] for b in BANDS], float)
        ax, axr = axes[0, k], axes[1, k]
        axr.fill_between(x, 1 - cat_e / cat, 1 + cat_e / cat, color='#888', alpha=0.2, lw=0,
                         label='catalog error')
        for j, (col, lbl, c, mk) in enumerate(SERIES):
            y = np.array([s[b][col] for b in BANDS], float)
            e = np.array([s[b][f'{col}_err'] for b in BANDS], float)
            dx = x * (1 + 0.012 * (j - 2.5))
            ax.plot(dx, y, '-', color=c, lw=1.2, alpha=0.6)
            ax.errorbar(dx, y, yerr=e, fmt=mk, color=c, ms=8, mec='white', mew=1.0, elinewidth=1.3,
                        capsize=2.5, label=lbl)
            axr.plot(dx, y / cat, '-', color=c, lw=1.0, alpha=0.5)
            axr.errorbar(dx, y / cat, yerr=e / cat,
                         fmt=mk, color=c, ms=8, mec='white', mew=1.0, elinewidth=1.3, capsize=2.5)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_title(f'Deger+22 #{rec} = our source {ours} (MAP)', fontsize=10)
        ax.grid(alpha=0.25, which='both')
        axr.axhline(1, color='#888', lw=1, ls='--')
        axr.set_ylim(0.4, 1.5)
        axr.grid(alpha=0.25, which='both')
        axr.set_xticks(x); axr.set_xticklabels(BANDS, rotation=30)
        axr.minorticks_off()
        if k == 0:
            ax.set_ylabel(r'$F_\nu$ [mJy]')
            axr.set_ylabel('flux / catalog')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.905), ncol=3, fontsize=8.5, frameon=False)
    fig.suptitle(r'NGC 1365: the 3 Deger+22 clusters in our selection (1$\sigma$ errors; grey band = catalog error)',
                 y=1.04, fontsize=12)
    fig.savefig(os.path.join(OUT, 'NGC1365_deger_overlap_sed.png'), dpi=130, bbox_inches='tight')
    plt.close(fig)


def report(t):
    for rec in dict.fromkeys(int(r) for r in t['recno']):
        s = t[t['recno'] == rec]
        print(f'\nDeger #{rec} = our source {int(s["ours_idx"][0])}   flux +- err [uJy] (S/N)')
        print(' ' * 10 + ''.join(f'{b:>22s}' for b in BANDS))
        for col, *_ in SERIES:
            print(f'{col:10s}' + ''.join(f'{1e3 * r[col]:7.2f}+-{1e3 * r[col + "_err"]:5.2f} ({r[col] / r[col + "_err"]:3.0f})'
                                         for r in s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plot', action='store_true', help='re-plot from the saved table')
    a = ap.parse_args()
    t = Table.read(TABLE + '.fits') if a.plot else measure()
    plot(t)
    report(t)


if __name__ == '__main__':
    main()
