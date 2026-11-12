"""
Any galaxy, HST bands: every photometry method / PHANGS-HST catalog, per
cluster. Same plots as ngc1365_all_methods_vs_catalog.py, but self-contained:
the Deger+22 fluxes are measured here on the HST images instead of read from
the NGC1365-only test tables.

  native, instG, instP   pipeline tables <out>/<gal>/<gal>_photometry_{native,instrument}.fits
  2D fixed / free        <out>/<gal>/<gal>_photometry_2d.fits (run_2d_photometry.py), if present
  2D G fixed / G free    same file, modes Gfixed / Gfree, if present
  Deger gal    4 px (0.158") at the catalog position, sky median 7-8 px,
               x 10**(-0.4 (AC_V + offset_band)) with the galaxy's AC_V (Deger+22 Table 2)
  Deger exact  the same 4 px flux x k_cluster = F_total / F_4px of a noise-free
               model of the cluster: galaxy-matched instrument PSF (x) profile_instP
               at the cluster's instP R_eff in that band
<out> = output/ (filtered catalog) or output_catalog/ (--catalog-mode). All
Milky Way dereddened (E(B-V) of the 2D file or the SFD query), all in mJy.
Error bars 1 sigma: catalog _ERR, pipeline _err, Deger sigma_sky sqrt(N_ap) x
correction x MW; ratio errors in quadrature.

    python all_methods_vs_catalog.py --galaxy NGC3351 --catalog-mode

Writes <validation_output_dir>/output_catalog_comparison/<gal>[_catalog]_all_methods_*.png/.fits
"""
import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)
from _paths import OUTPUT_DIR, CATALOG_OUTPUT_DIR  # noqa: E402

import astropy.units as u  # noqa: E402
from astropy.coordinates import SkyCoord  # noqa: E402
from astropy.io import fits  # noqa: E402
from astropy.table import Table  # noqa: E402
from astropy.wcs import WCS  # noqa: E402

import run_full_pipeline as rfp  # noqa: E402
from phot_aux_func import corr_flux  # noqa: E402
from instrument_psf import instrument_psf_for_image  # noqa: E402
from deger_aperture_correction import (DEGER22_TABLE2, HST_PIX_ARCSEC, ac_mag_to_flux_factor, band_ac,  # noqa: E402
                                       aperture_flux, aperture_flux_err, _sky)
import ngc1365_all_methods_vs_catalog as amc  # noqa: E402

BANDS, CAT_BAND, OUT = amc.BANDS, amc.CAT_BAND, amc.OUT
REFF_GRID = np.array([0.0, 0.25, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 13, 16, 20, 25, 30])
PROFILES = {'eff3.0': dict(profile='eff', gamma=3.0, c=1.5), 'king1.5': dict(profile='king', gamma=3.0, c=1.5)}
MODES_2D = {'2D fixed': 'fixed', '2D free': 'free', '2D G fixed': 'Gfixed', '2D G free': 'Gfree'}


def galaxy_ebv(gal, t2d):
    if t2d is not None and 'EBV' in t2d.meta:
        return float(t2d.meta['EBV'])
    from dustmaps.sfd import SFDQuery
    return float(SFDQuery()(SkyCoord.from_name(gal)))


def psf_fwhm(cfg, gal, band):
    psf = Table.read(cfg['psf_filepath'], format='csv')
    row = psf[np.asarray(psf['galaxy_name']).astype(str) == gal]
    return float(row[f'psf_HST_{band.lower()}'][0])


def k_exact(model, pix, reff_pc, spec, dist_mpc):
    """F_total / F_4px (sky median 7-8 px) of a noise-free model of unit total flux"""
    hs = int(np.ceil(3.0 / pix))
    kw = dict(flux=1.0)
    if reff_pc > 0:
        kw.update(r_eff_arcsec=reff_pc * 206265.0 / (dist_mpc * 1e6), **spec)
    img = model.render(hs, hs, (2 * hs + 1, 2 * hs + 1), **kw)
    k = HST_PIX_ARCSEC / pix
    return 1.0 / aperture_flux(img, hs, hs, 4 * k, _sky(img, hs, hs, 7 * k, 8 * k))


def build_table(gal, out_dir):
    d = os.path.join(out_dir, gal)
    v2 = Table.read(os.path.join(d, f'{gal}_photometry_v2.fits'))
    nat = Table.read(os.path.join(d, f'{gal}_photometry_native.fits'))
    ins = Table.read(os.path.join(d, f'{gal}_photometry_instrument.fits'))
    p2d = os.path.join(d, f'{gal}_photometry_2d.fits')
    t2d = Table.read(p2d) if os.path.exists(p2d) else None
    assert len(v2) == len(nat) == len(ins)
    if t2d is not None:
        assert np.array_equal(np.asarray(t2d['ID_PHANGS_CLUSTER']), np.asarray(v2['ID_PHANGS_CLUSTER']))
    cfg = rfp.load_config(os.path.join(_ROOT, 'config.ini'))
    paths = rfp.find_hst_band_paths(cfg['hst_root'], gal)
    ebv = galaxy_ebv(gal, t2d)
    ac_v = DEGER22_TABLE2[gal][0]
    dist = float(np.nanmedian(np.asarray(v2['PHANGS_DMPC'], float)))
    sky = SkyCoord(np.asarray(v2['PHANGS_RA'], float) * u.deg, np.asarray(v2['PHANGS_DEC'], float) * u.deg)
    prof = np.asarray(ins['profile_instP']).astype(str)

    t = Table()
    t['ID_PHANGS_CLUSTER'] = v2['ID_PHANGS_CLUSTER']
    t['CLASS'] = [str(c).strip() for c in np.asarray(v2['PHANGS_CC_CLASS_ML']).astype(str)]
    t['profile_instP'] = prof
    t.meta.update(GALAXY=gal, EBV=ebv, AC_V=ac_v, DIST=dist)
    for b in BANDS:
        with fits.open(paths[b]) as h:
            img, hdr = h[0].data, h[0].header
        w = WCS(hdr)
        pix = float(w.proj_plane_pixel_scales()[0].to('arcsec').value)
        k = HST_PIX_ARCSEC / pix
        f4, e4 = [], []
        for c in sky:
            x, y = (float(v) for v in c.to_pixel(w))
            f4.append(aperture_flux(img, x, y, 4 * k, _sky(img, x, y, 7 * k, 8 * k)) * 1e3)  # Jy -> mJy
            e4.append(aperture_flux_err(img, x, y, 4 * k, 7 * k, 8 * k)[0] * 1e3)
        f4, e4 = np.array(f4), np.array(e4)
        mw = float(corr_flux(1.0, rfp.FILTER_WAVELENGTHS[b][1], ebv))
        k_gal = float(ac_mag_to_flux_factor(band_ac(ac_v, b)))
        # exact per-cluster correction: curves on REFF_GRID, interpolated at the instP R_eff
        model = instrument_psf_for_image(b, hdr, pix, extra_fwhm_arcsec=0.0)[0]   # theoretical PSF, as the pipeline
        reff = np.asarray(ins[f'reff_pc_instP_{b}'], float)
        k_cl = np.full(len(t), np.nan)
        for tag, spec in PROFILES.items():
            m = prof == tag
            if m.any():
                curve = [k_exact(model, pix, r, spec, dist) for r in REFF_GRID]
                k_cl[m] = np.interp(reff[m], REFF_GRID, curve, right=np.nan)

        t[f'catalog_{b}'] = np.asarray(v2[f'PHANGS_{CAT_BAND[b]}_mJy'], float)
        t[f'catalog_{b}_err'] = np.abs(np.asarray(v2[f'PHANGS_{CAT_BAND[b]}_mJy_ERR'], float))
        t[f'instP_size_{b}'] = [str(s).strip() for s in np.asarray(ins[f'size_source_instP_{b}']).astype(str)]
        t[f'instP_reff_pc_{b}'] = reff
        t[f'k_cluster_{b}'] = k_cl
        t.meta[f'KGAL_{b}'] = k_gal
        t.meta[f'MW_{b}'] = mw
        cols = [('native', nat[f'flux_native_{b}'], nat[f'flux_native_{b}_err']),
                ('instG', ins[f'flux_instG_{b}'], ins[f'flux_instG_{b}_err']),
                ('instP', ins[f'flux_instP_{b}'], ins[f'flux_instP_{b}_err'])]
        if t2d is not None:
            cols += [(m, t2d[f'flux_f2d{mode}_{b}'], t2d[f'flux_f2d{mode}_{b}_err'])
                     for m, mode in MODES_2D.items() if f'flux_f2d{mode}_{b}' in t2d.colnames]
        cols += [('Deger gal', f4 * k_gal * mw, e4 * k_gal * mw),
                 ('Deger exact', f4 * k_cl * mw, e4 * k_cl * mw)]
        for m, flux, err in cols:
            t[f'{m}_{b}'] = np.asarray(flux, float)
            t[f'{m}_{b}_err'] = np.abs(np.asarray(err, float))
        print(f'  {b}: k_gal x{k_gal:.3f}, k_cluster median x{np.nanmedian(k_cl):.3f}', flush=True)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--galaxy', default='NGC3351')
    ap.add_argument('--catalog-mode', action='store_true', help='read output_catalog/ (unfiltered catalog)')
    a = ap.parse_args()
    gal = a.galaxy.upper()
    out_dir = CATALOG_OUTPUT_DIR if a.catalog_mode else OUTPUT_DIR
    tag = f'{gal}_catalog' if a.catalog_mode else gal
    note = ', full catalog' if a.catalog_mode else ''
    os.makedirs(OUT, exist_ok=True)
    t = build_table(gal, out_dir)
    t.write(os.path.join(OUT, f'{tag}_all_methods_vs_catalog.fits'), overwrite=True)
    amc.plot_grid(t, os.path.join(OUT, f'{tag}_all_methods_vs_catalog.png'), gal=gal, note=note)
    amc.plot_summary(t, os.path.join(OUT, f'{tag}_all_methods_summary.png'), gal=gal, note=note)
    print(f'{gal}: {len(t)} clusters, E(B-V) {t.meta["EBV"]:.4f}, AC_V {t.meta["AC_V"]}, D {t.meta["DIST"]} Mpc')
    print('median method / catalog [16-84%] (N)')
    print(f'{"":12s}' + ''.join(f'{b:>24s}' for b in BANDS))
    for m in amc._methods(t):
        row = f'{m:12s}'
        for b in BANDS:
            r = np.asarray(t[f'{m}_{b}'], float) / np.asarray(t[f'catalog_{b}'], float)
            r = r[np.isfinite(r) & (r > 0)]
            row += (f'{np.median(r):7.2f} [{np.percentile(r, 16):.2f}-{np.percentile(r, 84):.2f}] ({r.size:3d})'
                    if r.size else f'{"-":>24s}')
        print(row)
    print(f'written to {OUT}')


if __name__ == '__main__':
    main()
