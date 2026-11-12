#!/usr/bin/env python
"""
Flux-recovery test of fit_slit_profile_instrument_psf on the synthetic mocks.

For every FITS image in the chosen folder(s):
  1. centre : fit_2d_gaussian_center, starting at the image centre pixel
  2. size   : three ways of getting R65, all on the same averaged slit profile
              (1 px slits by default):
       gauss : plain Gaussian fit of the profile + closed-form Gaussian R65
       instG : instrument PSF (x) Gaussian source   (fit_slit_profile_instrument_psf)
       instP : instrument PSF (x) cluster profile, EFF or King
               (fit_slit_profile_instrument_psf, intrinsic_profile=...)
               — only when a non-Gaussian profile is requested / in the header
  3. flux   : aperture sum within each R65, divided by 0.65
              (true total flux = TOTFLUX in the header, 1 by default)
plus two catalog-style methods (PHANGS-HST, Deger+22), measured at the same
centre:
       deger    : 4 HST px (0.158") aperture, sky median at 7-8 px
                  (0.277-0.317"), times one aperture correction (deger_ac;
                  default x1.752 = NGC1365's catalog correction in V)
       growth20 : light within 20 HST px (0.792"), sky median at 21-23 px
                  (0.832-0.911") — Deger's definition of "total", per source
Pick any subset with methods=[...] (default gauss, instG, instP) or 'all'.
The PSF model is rebuilt from the image header, so it is the one the mock was
made with.

The noise-free mocks can be degraded on the fly (only in memory; the FITS
file is not changed):
  snr        : add Gaussian noise so the source has this S/N within 0.158"
               (the catalog's 4 px aperture), on a flat sky of bkg_sigma x sigma
  neighbours : add Gaussian neighbours seen through the same PSF, e.g.
               dict(n=3, r=(0.2, 0.5), flux=(0.1, 0.9), fwhm=(0.0, 0.15))
               (r and fwhm in arcsec, flux in units of the true total flux)
  seed       : random seed for the noise and the neighbours
With a sky (added here, or NOISESTD/BKGLEVEL in the header) the R65 fluxes
subtract a sigma-clipped annulus median (2.6-4 x R65, as the pipeline) and the
instrument fits use the pipeline's model-based correction (which also puts back
the halo light the annulus removes); background='none' / 'annulus' forces it.

--profile auto (default) uses the cluster profile written in the header of the
smooth-cluster mocks (CLPROF / EFF_GAM / KING_C) and a Gaussian otherwise;
--profile eff / king forces that profile (with --gamma / --king-c) on every
image.

Works on the mock sets:
  output_typical/  (generate_typical_resolution_test_images)
  output_smooth/   (generate_smooth_cluster_images)
  output_smooth_offset/ (same, center_offset_px='random': source off the centre pixel)
  output_hst/      (generate_all_hst_psf_test_images)
  output_jwst/     (generate_all_jwst_psf_test_images)

Usage:
    python test_instrument_fit.py                                   # output_typical
    python test_instrument_fit.py --dirs output_smooth --pattern "smooth_*"
    python test_instrument_fit.py --dirs output_smooth --pattern "smooth_eff3_F770W*" --plot
    python test_instrument_fit.py --profile king --king-c 1.5 --pattern "compactcluster_*"
    python test_instrument_fit.py --csv results.csv
    python test_instrument_fit.py --dirs output_smooth_offset --methods all --snr 30 --seed 1
"""
import argparse
import glob
import os
import sys
import warnings

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.aperture import CircularAnnulus, CircularAperture, aperture_photometry

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
from _paths import OUT_ROOT  # noqa: E402

from astropy.coordinates import SkyCoord  # noqa: E402
from astropy.wcs import WCS  # noqa: E402

from instrument_psf import InstrumentPSF, JWST_PIXEL_SCALE_ARCSEC, instrument_psf_for_image  # noqa: E402
from native_phot_aux_func import (fit_2d_gaussian_center,  # noqa: E402
                                  measure_averaged_slit_profile, gaussian_r_at_ee,
                                  fit_slit_profile_instrument_psf,
                                  instrument_fit_model_profile, _instrument_corr_factor)
from synthetic_psf_tests import instrument_psf_at_fwhm  # noqa: E402

_MODEL_CACHE = {}


def psf_model_from_header(hdr):
    """Rebuild the InstrumentPSF an image was rendered with, from its header."""
    inst = str(hdr['PSFINST']).upper()
    band = str(hdr['PSFBAND']).upper()
    pix = float(hdr['PIXSCALE'])

    if 'RESFWHM' in hdr:                                   # output_typical
        family = 'jwst' if inst.startswith('JWST') else inst.split('/')[-1].lower()
        key = ('typical', family, band, float(hdr['RESFWHM']))
        if key not in _MODEL_CACHE:
            _MODEL_CACHE[key] = instrument_psf_at_fwhm(family, band, float(hdr['RESFWHM']),
                                                       pixel_scale_arcsec=pix)[0]
    elif 'DRZFWHM' in hdr:                                 # output_hst
        key = ('hst', inst.lower(), band, float(hdr['DRZFWHM']))
        if key not in _MODEL_CACHE:
            _MODEL_CACHE[key] = InstrumentPSF.from_hst(inst.lower(), band, pixel_scale_arcsec=pix,
                                                       drizzle_fwhm_arcsec=float(hdr['DRZFWHM']),
                                                       stamp_ee=1.0)   # mocks have no wings beyond the stamp
    elif 'EXTFWHM' in hdr:                                 # output_jwst
        key = ('jwst', band, float(hdr['EXTFWHM']))
        if key not in _MODEL_CACHE:
            _MODEL_CACHE[key] = InstrumentPSF.from_jwst(band, pixel_scale_arcsec=pix,
                                                        extra_fwhm_arcsec=float(hdr['EXTFWHM']))
    else:
        raise ValueError('header has no PSF-model keywords (RESFWHM / DRZFWHM / EXTFWHM)')
    return _MODEL_CACHE[key]


# ─────────────────────────────────────────────────────────────────────────────
# real images (e.g. aux_functions.make_cutout of a PHANGS HST / JWST mosaic)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_PSF_TABLE = os.path.join(os.path.dirname(_HERE), 'data', 'psf_table.csv')


def is_mock_header(hdr):
    return 'PSFINST' in hdr


def band_from_header(hdr):
    """Filter name from FILTER, or FILTER1/FILTER2 (ACS: the one that is not CLEAR)."""
    for k in ('FILTER', 'FILTER1', 'FILTER2'):
        v = str(hdr.get(k, '')).strip().upper()
        if v.startswith('F') and not v.startswith('CLEAR'):
            return v
    return None


def is_jwst_band(band):
    return band.upper() in JWST_PIXEL_SCALE_ARCSEC


def units_per_mjy(band, hdr):
    """Image units (pixel sum) per mJy, as the pipeline converts them: JWST via
    phot_aux_func.jwst_pixel_sum_to_Jy (BUNIT MJy/sr + PIXAR_SR, or PHOTMJSR),
    HST from BUNIT=Jy (PHANGS anchor images) or PHOTFNU for electrons/s."""
    if is_jwst_band(band):
        from phot_aux_func import jwst_pixel_sum_to_Jy
        if 'PIXAR_SR' not in hdr:
            raise ValueError(f'{band}: header has no PIXAR_SR — pass the SCI-extension header '
                             f'(JWST mosaics keep BUNIT/PIXAR_SR in extension 1)')
        return 1.0 / (jwst_pixel_sum_to_Jy(1.0, hdr) * 1e3)
    bunit = str(hdr.get('BUNIT', '')).strip().upper()
    if bunit.startswith('JY'):
        return 1e-3
    if 'PHOTFNU' in hdr:                                    # electrons/s -> Jy
        return 1.0 / (float(hdr['PHOTFNU']) * 1e3)
    raise ValueError(f'{band}: cannot convert BUNIT={bunit!r} to mJy (no PHOTFNU)')


def psf_fwhm_from_table(galaxy, band, path=DEFAULT_PSF_TABLE):
    """This galaxy's measured PSF FWHM (arcsec) for `band` from psf_table.csv
    (column psf_HST_<band> / psf_JWST_<band>; F435W and F438W are the same slot)."""
    from astropy.table import Table
    t = Table.read(path, format='csv')
    rows = t[np.char.upper(np.asarray(t['galaxy_name']).astype(str)) == str(galaxy).upper()]
    if len(rows) == 0:
        raise ValueError(f'galaxy {galaxy!r} not in {path}')
    tel = 'JWST' if is_jwst_band(band) else 'HST'
    alts = {'F435W': ['f435w', 'f438w'], 'F438W': ['f438w', 'f435w']}.get(band.upper(), [band.lower()])
    for b in alts:
        col = f'psf_{tel}_{b}'
        if col in t.colnames:
            v = float(rows[0][col])
            if np.isfinite(v) and v > 0:
                return v
    return None


def real_psf_model(band, hdr, pix, fwhm=None):
    """Galaxy-matched instrument PSF of a real image, built as the pipeline
    does (instrument_psf.instrument_psf_for_image): extra blur tuned so the
    model star has `fwhm` (if given), else the sample-default blur."""
    inst = str(hdr.get('INSTRUME', '')).strip().upper()
    key = ('real', band.upper(), inst, round(pix, 6), None if fwhm is None else round(fwhm, 5))
    if key not in _MODEL_CACHE:
        model, note = instrument_psf_for_image(band, hdr, pix, target_fwhm_arcsec=fwhm)
        if model is None:
            raise ValueError(f'no instrument PSF model for {band}: {note}')
        if note:
            print(f'  PSF {band}: {note}')
        _MODEL_CACHE[key] = model
    return _MODEL_CACHE[key]


def default_slit_length(hdr, pix):
    """Half-length of the slits: ~4x the PSF FWHM and ~8x the cluster
    half-light radius (if known), at least 1"."""
    if 'RESFWHM' in hdr:
        fwhm = float(hdr['RESFWHM'])
    else:
        fwhm = 0.7 if pix > 0.1 else 0.15                  # MIRI vs HST/NIRCam
    return max(1.0, 4.0 * fwhm, 8.0 * float(hdr.get('REFF_AS', 0.0)))


def profile_from_header(hdr):
    """Cluster profile written in the header of the smooth-cluster mocks,
    as (intrinsic_profile, gamma, king_c); Gaussian if none."""
    prof = str(hdr.get('CLPROF', 'gaussian')).lower()
    if prof not in ('gaussian', 'eff', 'king'):
        prof = 'gaussian'
    return prof, float(hdr.get('EFF_GAM', 3.0)), float(hdr.get('KING_C', 1.5))


def profile_label(profile, gamma, king_c):
    return {'gaussian': 'Gauss', 'eff': f'EFF(gamma={gamma:g})',
            'king': f'King(c={king_c:g})'}[profile]


def aperture_flux(data, x0, y0, r_arcsec, pix, target_ee):
    ap = CircularAperture((x0, y0), r=r_arcsec / pix)
    return float(aperture_photometry(data, ap)['aperture_sum'][0]) / target_ee


def instrument_model_profile(model, fit, x0, y0, slit_len, n_slits, slit_width_px,
                             gamma=3.0, king_c=1.5):
    """Averaged slit profile of a fitted instrument model (flux * model + bkg),
    sampled with exactly the slits used on the data."""
    return instrument_fit_model_profile(model, x0, y0, fit.r_eff_arcsec, fit.profile, fit.flux,
                                        fit.background, slit_len, n_slits, slit_width_px,
                                        gamma=gamma, king_c=king_c)[1]


# catalog apertures (Deger+22), in arcsec: 4, 7, 8, 20, 21, 23 HST px of 0.0396"
R4, R7, R8 = 0.158, 0.277, 0.317
R20, R21, R23 = 0.792, 0.832, 0.911
DEGER_AC_NGC1365 = 1.752                 # NGC1365 catalog correction in V (MW removed)
ANNULUS_R_IN, ANNULUS_R_OUT = 2.6, 4.0   # pipeline background annulus, x R65
ALL_METHODS = ('gauss', 'instG', 'instP', 'deger', 'growth20')


def sky_median(data, x0, y0, r_in_arcsec, r_out_arcsec, pix, clip=False):
    """Median of an annulus (sigma-clipped if clip, as the pipeline does)."""
    v = CircularAnnulus((x0, y0), r_in_arcsec / pix, r_out_arcsec / pix).to_mask('center').get_values(data)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan
    return float(sigma_clipped_stats(v, sigma=3.0)[1]) if clip else float(np.median(v))


def aperture_sum(data, x0, y0, r_arcsec, pix, sky=0.0, mode='exact'):
    """Sky-subtracted sum inside a circle ('exact' overlap, or 'center' pixels
    as the pipeline's annulus-corrected photometry)."""
    ap = CircularAperture((x0, y0), r=r_arcsec / pix)
    if mode == 'exact':
        return float(aperture_photometry(np.nan_to_num(data - sky), ap)['aperture_sum'][0])
    m = ap.to_mask('center')
    return float(np.nansum(m.get_values(data)) - sky * np.sum(m.data > 0))


def degrade(data, hdr, model, snr=None, bkg_sigma=10.0, neighbours=None, seed=None,
            pix=None, xt=None, yt=None, true_flux=None):
    """Copy of `data` with neighbours and/or a flat sky + Gaussian noise added
    (see the module docstring). pix, (xt, yt) and true_flux default to the
    mock header (PIXSCALE, CLX0/CLY0, TOTFLUX). Returns (image, info dict)."""
    rng = np.random.default_rng(seed)
    pix = float(hdr['PIXSCALE']) if pix is None else pix
    img = np.array(data, dtype=float, copy=True)
    true_flux = float(hdr.get('TOTFLUX', 1.0)) if true_flux is None else true_flux
    c = (data.shape[0] - 1) // 2
    xt = float(hdr.get('CLX0', c)) if xt is None else xt
    yt = float(hdr.get('CLY0', c)) if yt is None else yt
    info = dict(sigma=0.0, sky=0.0, n_neighbours=0, neighbours=[])
    if neighbours:
        nb = dict(n=3, r=(0.2, 0.5), flux=(0.1, 0.9), fwhm=(0.0, 0.15))
        nb.update(neighbours)
        for _ in range(int(nb['n'])):
            r, th = rng.uniform(*nb['r']), rng.uniform(0, 2 * np.pi)
            f, w = rng.uniform(*nb['flux']) * true_flux, rng.uniform(*nb['fwhm'])
            xn, yn = xt + r / pix * np.cos(th), yt + r / pix * np.sin(th)
            img += model.render(xn, yn, img.shape, intrinsic_fwhm_arcsec=w, flux=f)
            info['neighbours'].append(dict(x=xn, y=yn, r_arcsec=r, flux=f / true_flux, fwhm_arcsec=w))
        info['n_neighbours'] = int(nb['n'])
    if snr:
        f4 = aperture_sum(data, xt, yt, R4, pix)          # noise-free source only
        sigma = f4 / (snr * np.sqrt(np.pi * (R4 / pix) ** 2))
        img += bkg_sigma * sigma + rng.normal(0.0, sigma, img.shape)
        info.update(sigma=float(sigma), sky=float(bkg_sigma * sigma))
    return img, info


METHOD_STYLE = {
    'gauss': dict(color='royalblue', ls='--'),
    'instG': dict(color='red', ls='-'),
    'instP': dict(color='darkorange', ls='-.'),
    'deger': dict(color='teal', ls=':'),
    'growth20': dict(color='seagreen', ls=(0, (5, 2, 1, 2))),
}


def plot_comparison(name, data, x0, y0, pix, model, results, diag, slit_len, slit_width_px,
                    gamma=3.0, king_c=1.5):
    """
    Left : image with the slits (one colour per slit) and the R65 aperture of
           every method.
    Right: per-slit profiles (same colours), the averaged profile, and the
           fitted curve + ±R65 of every method:
             gauss — plain Gaussian fit, closed-form R65
             instG — instrument PSF (x) Gaussian source
             instP — instrument PSF (x) EFF / King source (if run)
    `results` maps method -> dict(curve, r65, flux, label).
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon

    theta = diag.theta_deg
    n_slits = len(theta)
    colors = plt.cm.hsv(np.linspace(0, 1, n_slits, endpoint=False))
    half_len_px = slit_len / pix
    half_w = slit_width_px / 2.0 if slit_width_px > 1 else 0.0
    along = diag.along_arcsec

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))

    # ── image + slits + apertures ──
    finite = data[np.isfinite(data)]
    vmin, vmax = np.nanpercentile(finite, [1, 99.8])
    ax1.imshow(data, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
    for k in range(n_slits):
        t = np.deg2rad(theta[k])
        dx, dy, pdx, pdy = np.cos(t), np.sin(t), -np.sin(t), np.cos(t)
        if half_w > 0:
            corners = [(x0 + sa * half_len_px * dx + sw * half_w * pdx,
                        y0 + sa * half_len_px * dy + sw * half_w * pdy)
                       for sa, sw in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            ax1.add_patch(Polygon(corners, closed=True, facecolor=colors[k], alpha=0.3,
                                  edgecolor=colors[k], lw=1))
        else:
            ax1.plot([x0 - half_len_px * dx, x0 + half_len_px * dx],
                     [y0 - half_len_px * dy, y0 + half_len_px * dy],
                     '-', color=colors[k], lw=1, alpha=0.8)
    for meth, res in results.items():
        if np.isfinite(res['r65']):
            st = METHOD_STYLE[meth]
            rlab = 'R65' if meth in ('gauss', 'instG', 'instP') else 'r'
            ax1.add_patch(Circle((x0, y0), res['r65'] / pix, fill=False, lw=2, color=st['color'],
                                 ls=st['ls'], label=f"{rlab} {res['label']} = {res['r65']:.3f}\""))
    ax1.plot(x0, y0, '+', color='yellow', ms=12, mew=2)
    r_max_px = max([res['r65'] / pix for res in results.values() if np.isfinite(res['r65'])] + [0.0])
    pad = 1.15 * max(half_len_px, r_max_px)
    ax1.set_xlim(x0 - pad, x0 + pad)
    ax1.set_ylim(y0 - pad, y0 + pad)
    ax1.set_xlabel('x [px]')
    ax1.set_ylabel('y [px]')
    ax1.set_title(f'{name}\n{n_slits} slits, half-length {slit_len:.2f}", '
                  f'width {slit_width_px}px')
    ax1.legend(loc='upper right', fontsize=7)

    # ── profiles ──
    for k in range(n_slits):
        ax2.plot(along, diag.per_slit_profiles[k], '-', color=colors[k], lw=0.8, alpha=0.4)
    ax2.plot(along, diag.averaged_profile, 'k-', lw=2.4, label='averaged data profile')
    for meth, res in results.items():
        st = METHOD_STYLE[meth]
        if res['curve'] is not None and np.any(np.isfinite(res['curve'])):
            ax2.plot(along, res['curve'], lw=1.8, **st, label=f"fit: {res['label']}")
        if np.isfinite(res['r65']):
            rlab = 'R65' if meth in ('gauss', 'instG', 'instP') else f'{meth} r'
            ax2.axvline(res['r65'], lw=1.0, alpha=0.8, **st,
                        label=f"{rlab} {res['r65']:.3f}\" -> flux {res['flux']:.4g}")
            ax2.axvline(-res['r65'], lw=1.0, alpha=0.8, **st)
    ax2.set_xlim(-slit_len, slit_len)
    ax2.set_xlabel('Radius along slit [arcsec]')
    ax2.set_ylabel('Flux per pixel')
    ax2.set_title('Slit profiles: data vs Gaussian fit vs instrument-PSF fits\n'
                  '(deger / growth20: their aperture radius and flux)')
    ax2.legend(fontsize=7, loc='upper right')

    plt.tight_layout()
    plt.show()
    return fig


def load_image(src, header=None, ext=None):
    """
    (data, header, name) from any of
      - a FITS file path (str / os.PathLike)
      - an astropy HDUList
      - a single HDU (PrimaryHDU, ImageHDU, CompImageHDU)
      - a 2-D array, with its `header` given separately
      - a (data, header) tuple
    ext : HDU index or name for a path / HDUList; None -> the first HDU with
          2-D data. `header`, if given, replaces the HDU's own header.
    `name` labels plots and the 'file' entry of the results: the file name,
    else the header's FILENAME / EXTNAME, else 'image'.
    """
    name = None
    if isinstance(src, (str, os.PathLike)):
        name = os.path.basename(os.fspath(src))
        with fits.open(src) as hdul:
            data, hdr, _ = load_image(hdul, header=header, ext=ext)
        return data, hdr, name
    if isinstance(src, fits.HDUList):
        if ext is None:
            hdus = [h for h in src if getattr(h, 'data', None) is not None and np.ndim(h.data) == 2]
            if not hdus:
                raise ValueError('HDUList has no HDU with 2-D data')
            hdu = hdus[0]
        else:
            hdu = src[ext]
        src = hdu
    if isinstance(src, (fits.PrimaryHDU, fits.ImageHDU, fits.CompImageHDU)):
        data, hdr = src.data, src.header
    elif isinstance(src, tuple) and len(src) == 2:
        data, hdr = src
    else:
        data, hdr = src, None
    if header is not None:
        hdr = header
    if hdr is None:
        raise ValueError('an array needs its header: run_one(data, header=hdr, ...)')
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError(f'expected a 2-D image, got shape {data.shape}')
    hdr = fits.Header(hdr)                       # a copy; accepts dict-like headers too
    name = name or str(hdr.get('FILENAME', '') or hdr.get('EXTNAME', '') or 'image')
    return data.copy(), hdr, name


def run_one(path, target_ee=0.65, slit_width_px=1, slit_length_arcsec=None,
            fit_half_size_px=4.0, profile='auto', gamma=None, king_c=None, plot=False,
            search_px=None, methods=('gauss', 'instG', 'instP'), deger_ac=DEGER_AC_NGC1365,
            snr=None, bkg_sigma=10.0, neighbours=None, seed=None, background='auto',
            start=None, header=None, ext=None, name=None, wcs=None, position=None,
            band=None, galaxy=None, psf_fwhm=None, psf_table=DEFAULT_PSF_TABLE,
            distance_mpc=None, ebv=None):
    """
    Measure one image — a mock or a real cutout — with any of the methods
    (see module docstring).

    REAL IMAGES (header without the mock PSF keywords), e.g.
        cut, cut_wcs = make_cutout(data, wcs, coord, 6 * u.arcsec)
        run_one(cut, header=hdr, wcs=cut_wcs, position=coord, galaxy='NGC1365',
                methods='all', plot=True)
      header   : the image's own header (units, INSTRUME, FILTER); for JWST
                 the SCI-extension header (BUNIT / PIXAR_SR) — then give band=
      wcs      : WCS of the cutout (make_cutout returns it); default WCS(header)
      position : SkyCoord of the cluster, or (x, y) pixel position; the
                 centring starts there. Default: the image centre.
      band     : filter (default: header FILTER / FILTER1 / FILTER2)
      galaxy   : name in psf_table.csv, to match the PSF model to this galaxy's
                 measured FWHM (as the pipeline does); or give psf_fwhm=
                 (arcsec). Neither: sample-default PSF blur (printed).
      distance_mpc : to also report R_eff in pc.
      ebv      : Milky Way E(B-V); if given, fluxes are dereddened exactly as
                 the pipeline's flux_* columns (phot_aux_func.corr_flux with
                 the band's R value from run_full_pipeline.FILTER_WAVELENGTHS).
      Fluxes are then in mJy; profile='auto' means EFF gamma=3; the annulus
      background is always subtracted. Use a cutout of >= 5" so the slits,
      the 2.6-4 R65 annulus and the growth20 sky (0.91") fit inside.

    path    : the image — a FITS path, an HDUList, an HDU, a (data, header)
              tuple, or a 2-D array with header=... (see load_image). The
              header needs PIXSCALE and the PSF-model keywords of the mocks
              (PSFINST, PSFBAND + RESFWHM / DRZFWHM / EXTFWHM); TOTFLUX,
              CLX0/CLY0, CLPROF... are used when present.
    header, ext : as in load_image.
    name    : label for the plot title and the 'file' result (default: the
              file name, else the header's FILENAME / EXTNAME, else 'image').

    methods : subset of ('gauss', 'instG', 'instP', 'deger', 'growth20'), or
              'all'. Default: the three slit-profile methods.
    profile : 'auto' -> header cluster profile (CLPROF/EFF_GAM/KING_C) if any,
              else Gaussian; or 'gaussian' / 'eff' / 'king' to force one.
    gamma, king_c : EFF gamma / King concentration (None -> header value, else 3.0 / 1.5).
    deger_ac : aperture correction of the 'deger' method (4 px flux with the
              7-8 px sky -> total). Default x1.752 (NGC1365, V band).
    snr, bkg_sigma, neighbours, seed : degrade the image in memory first
              (see degrade()); snr=None and neighbours=None keep it as it is.
    background : 'auto' (annulus subtraction only if the image has a sky),
              'none' (plain aperture sum / target_ee, as before) or 'annulus'.
    search_px : centring starts at the peak of the (1 px-smoothed) image,
              searched over the whole image (None, default) or within
              ±search_px of the image centre (use that when other sources are
              in the image).
    start   : 'peak' (see search_px) or 'true': start the centring at the
              true centre (mocks: CLX0/CLY0) / at `position`. Default: 'true'
              for real images or when position is given, else 'peak'.
    The true centre (CLX0/CLY0 in the header, else the centre pixel) is
    otherwise only used to report the centring error.
    The instP method runs only when the resulting profile is not Gaussian.
    Returns a dict of results — fluxes normalized by the true total flux for
    mocks, in mJy for real images (see 'flux_unit'); NaN for methods not run.
    """
    methods = ALL_METHODS if methods == 'all' else tuple(methods)
    unknown = set(methods) - set(ALL_METHODS)
    if unknown:
        raise ValueError(f'unknown methods {sorted(unknown)}; choose from {ALL_METHODS}')
    data, hdr, auto_name = load_image(path, header=header, ext=ext)
    name = name or auto_name
    mock = is_mock_header(hdr)
    c = (data.shape[0] - 1) // 2
    used_fwhm = np.nan
    if mock:
        pix = float(hdr['PIXSCALE'])
        true_flux = float(hdr.get('TOTFLUX', 1.0))
        model = psf_model_from_header(hdr)
        slit_len = slit_length_arcsec or default_slit_length(hdr, pix)
        band = band or str(hdr.get('PSFBAND', '')).strip().upper()
        distance_mpc = distance_mpc or hdr.get('DIST_MPC')
        x_true, y_true = float(hdr.get('CLX0', c)), float(hdr.get('CLY0', c))
        flux_unit = 'total'
    else:
        wcs = wcs if wcs is not None else WCS(hdr, naxis=2)
        if not wcs.has_celestial:
            raise ValueError('real image without a celestial WCS: pass wcs= (make_cutout returns it)')
        pix = float(wcs.proj_plane_pixel_scales()[0].to('arcsec').value)
        band = (band or band_from_header(hdr) or '').upper()
        if not band:
            raise ValueError("no filter in the header: pass band='F555W' (JWST: the filter is in "
                             "the primary header, the units in the SCI header)")
        if psf_fwhm is None and galaxy is not None:
            psf_fwhm = psf_fwhm_from_table(galaxy, band, psf_table)
            if psf_fwhm is None:
                print(f'  {band}: no PSF FWHM for {galaxy} in {psf_table} — sample-default blur')
        elif psf_fwhm is None:
            print(f'  {band}: no galaxy= / psf_fwhm= given — sample-default PSF blur')
        used_fwhm = np.nan if psf_fwhm is None else float(psf_fwhm)
        model = real_psf_model(band, hdr, pix, psf_fwhm)
        true_flux = units_per_mjy(band, hdr)              # -> every flux below in mJy
        if ebv:
            from phot_aux_func import corr_flux
            from run_full_pipeline import FILTER_WAVELENGTHS
            r_value = FILTER_WAVELENGTHS.get('F438W' if band == 'F435W' else band, [None, None])[1]
            if r_value is None:
                raise ValueError(f'{band}: no R value in FILTER_WAVELENGTHS for the MW correction')
            true_flux /= float(corr_flux(1.0, r_value, ebv))
        fwhm_for_slit = psf_fwhm or (0.7 if pix > 0.1 else 0.15)
        slit_len = slit_length_arcsec or max(1.0, 4.0 * fwhm_for_slit)
        x_true, y_true = float(c), float(c)
        flux_unit = 'mJy'
    if position is not None:
        if isinstance(position, SkyCoord):
            if wcs is None:
                wcs = WCS(hdr, naxis=2)
            x_true, y_true = (float(v) for v in position.to_pixel(wcs))
        else:
            x_true, y_true = (float(v) for v in position)
    if start is None:
        start = 'true' if (position is not None or not mock) else 'peak'

    h_prof, h_gamma, h_c = profile_from_header(hdr)
    if profile == 'auto':
        prof = h_prof if mock else 'eff'                  # real clusters: EFF gamma=3 (YCL / MAP)
    else:
        prof = profile
    gamma = h_gamma if gamma is None else gamma
    king_c = h_c if king_c is None else king_c

    # 0. optional noise / sky / neighbours (in memory only)
    deg = dict(sigma=0.0, sky=0.0, n_neighbours=0, neighbours=[])
    if snr or neighbours:
        data, deg = degrade(data, hdr, model, snr=snr, bkg_sigma=bkg_sigma,
                            neighbours=neighbours, seed=seed, pix=pix, xt=x_true, yt=y_true,
                            true_flux=true_flux)
    has_sky = (not mock or bool(snr) or float(hdr.get('NOISESTD', 0) or 0) > 0
               or float(hdr.get('BKGLEVEL', 0) or 0) != 0)
    use_annulus = has_sky if background == 'auto' else (background == 'annulus')

    # 1. centre — the 2D Gaussian fit only lets the centre move
    # ±fit_half_size_px/2 from its starting guess, so the guess must already be
    # within ~2 px of the source: start at the peak of a lightly smoothed copy
    # of the image (sigma 1 px, so a single noisy pixel cannot win), searched
    # over the whole image (search_px=None) or within ±search_px of its centre.
    from scipy.ndimage import gaussian_filter
    if start in ('true', 'position'):
        gx, gy = int(round(x_true)), int(round(y_true))
    else:
        smooth = gaussian_filter(np.nan_to_num(data, nan=0.0), 1.0)
        if search_px is None:
            lo, hi = 0, data.shape[0]
        else:
            lo, hi = max(c - search_px, 0), c + search_px + 1
        py, px = np.unravel_index(np.argmax(smooth[lo:hi, lo:hi]), smooth[lo:hi, lo:hi].shape)
        gx, gy = lo + px, lo + py
        if search_px is not None and (px in (0, hi - lo - 1) or py in (0, hi - lo - 1)):
            print(f'  WARNING {name}: brightest pixel is on the edge of the '
                  f'±{search_px} px search box — the source is probably outside it')
    g = fit_2d_gaussian_center(data, gx, gy, fit_half_size_px, pix)
    if not g.converged:
        print(f'  WARNING {name}: 2D Gaussian centring did not converge, '
              f'using the starting pixel')
    x0, y0 = (g.x0_local, g.y0_local) if g.converged else (float(gx), float(gy))
    common = dict(target_ee=target_ee, slit_length_arcsec=slit_len, slit_width_px=slit_width_px)

    def r65_flux(r65, fit=None, spec=None):
        """flux within R65: plain sum / target_ee, or annulus-subtracted with
        the pipeline's correction (model-based for the instrument fits)"""
        if not np.isfinite(r65):
            return np.nan, np.nan
        if not use_annulus:
            return aperture_sum(data, x0, y0, r65, pix) / target_ee, np.nan
        sky = sky_median(data, x0, y0, ANNULUS_R_IN * r65, ANNULUS_R_OUT * r65, pix, clip=True)
        corr = (1.0 / target_ee if fit is None else
                _instrument_corr_factor(model, spec, fit.r_eff_arcsec, r65, x0, y0, target_ee,
                                        ANNULUS_R_IN, ANNULUS_R_OUT))
        return aperture_sum(data, x0, y0, r65, pix, sky, mode='center') * corr, sky

    nan = float('nan')
    out = dict(r65_gauss=nan, flux_gauss=nan, reff_instG=nan, r65_instG=nan, flux_instG=nan,
               fluxmod_instG=nan, ok_instG=False, reff_instP=nan, r65_instP=nan, flux_instP=nan,
               fluxmod_instP=nan, ok_instP=False, flux_deger=nan, flux_growth20=nan,
               sky_annulus=nan)
    results = {}                                   # for the plot

    # 2a. plain Gaussian route (its slit profile is also what the plot shows)
    sp, diag = measure_averaged_slit_profile(data, x0, y0, pix, slit_length_arcsec=slit_len,
                                             slit_width_px=slit_width_px,
                                             collect_diagnostics=True)
    if 'gauss' in methods:
        r65_gauss = gaussian_r_at_ee(sp.sigma_arcsec, target_ee) if sp.converged else np.nan
        flux_gauss, sky = r65_flux(r65_gauss)
        out.update(r65_gauss=r65_gauss, flux_gauss=flux_gauss / true_flux, sky_annulus=sky)
        results['gauss'] = dict(curve=diag.fitted_curve, r65=r65_gauss, flux=flux_gauss / true_flux,
                                label=f'Gaussian (sigma={sp.sigma_arcsec:.3f}")')

    n_slits = len(diag.theta_deg)
    fits_inst = {}
    # 2b. instrument PSF (x) Gaussian source;  2c. instrument PSF (x) cluster profile
    for meth, ip in (('instG', 'gaussian'), ('instP', prof)):
        if meth not in methods or (meth == 'instP' and prof == 'gaussian'):
            continue
        spec = dict(profile=ip, gamma=gamma, king_c=king_c)
        fit = fit_slit_profile_instrument_psf(data, x0, y0, model, intrinsic_profile=ip,
                                              gamma=gamma, king_c=king_c, **common)
        flux, sky = r65_flux(fit.r65_arcsec, fit, spec)
        fits_inst[meth] = fit
        out.update({f'reff_{meth}': fit.r_eff_arcsec, f'r65_{meth}': fit.r65_arcsec,
                    f'flux_{meth}': flux / true_flux, f'fluxmod_{meth}': fit.flux / true_flux,
                    f'ok_{meth}': bool(fit.converged)})
        if np.isfinite(sky):
            out['sky_annulus'] = sky
        lab = 'Gauss' if ip == 'gaussian' else profile_label(prof, gamma, king_c)
        results[meth] = dict(curve=instrument_model_profile(model, fit, x0, y0, slit_len, n_slits,
                                                            slit_width_px, gamma, king_c),
                             r65=fit.r65_arcsec, flux=flux / true_flux,
                             label=f'PSF (x) {lab} (R_eff={fit.r_eff_arcsec:.3f}")')

    # 2d. catalog-style apertures (Deger+22)
    if 'deger' in methods:
        f4 = aperture_sum(data, x0, y0, R4, pix, sky_median(data, x0, y0, R7, R8, pix))
        out['flux_deger'] = f4 * deger_ac / true_flux
        results['deger'] = dict(curve=None, r65=R4, flux=out['flux_deger'],
                                label=f'Deger: 4px x {deger_ac:.3f} (sky 7-8px)')
    if 'growth20' in methods:
        f20 = aperture_sum(data, x0, y0, R20, pix, sky_median(data, x0, y0, R21, R23, pix))
        out['flux_growth20'] = f20 / true_flux
        results['growth20'] = dict(curve=None, r65=R20, flux=out['flux_growth20'],
                                   label='growth20: 20px (sky 21-23px)')

    cen = wcs.pixel_to_world(x0, y0) if wcs is not None else None
    if plot:
        plot_comparison(name, data, x0, y0, pix, model, results, diag, slit_len, slit_width_px,
                        gamma, king_c)

    out.update(
        file=name, source=hdr.get('SRCTYPE', ''), psf=model.name, pix=pix,
        x0=x0, y0=y0, x_true=x_true, y_true=y_true,
        dcen_px=float(np.hypot(x0 - x_true, y0 - y_true)),
        center_ok=bool(g.converged), slit_len=slit_len,
        reff_true=float(hdr.get('REFF_AS', nan)),
        profile=prof if prof == 'gaussian' else profile_label(prof, gamma, king_c),
        band=band, flux_unit=flux_unit, psf_fwhm=used_fwhm, ebv=float(ebv) if ebv else 0.0,
        centroid_ra=float(cen.ra.deg) if cen is not None else nan,
        centroid_dec=float(cen.dec.deg) if cen is not None else nan,
        reff_pc_instG=(out['reff_instG'] * distance_mpc * 1e6 / 206265.0 if distance_mpc else nan),
        reff_pc_instP=(out['reff_instP'] * distance_mpc * 1e6 / 206265.0 if distance_mpc else nan),
        snr=float(snr) if snr else nan, noise_sigma=deg['sigma'], sky_true=deg['sky'],
        n_neighbours=deg['n_neighbours'], background='annulus' if use_annulus else 'none',
        seed=-1 if seed is None else int(seed),
    )
    return out


def _group(name):
    for g in ('pointsource', 'compactcluster', 'cluster'):
        if name.startswith(g):
            return g
    if name.startswith('smooth_'):
        return name.split('_')[1] if not name.startswith('smooth_king') else '_'.join(name.split('_')[1:3])
    return 'other'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dirs', nargs='+', default=['output_typical'],
                   help='mock folders, relative to this script (default: output_typical)')
    p.add_argument('--pattern', default='*.fits', help='glob for files inside each folder')
    p.add_argument('--profile', default='auto', choices=['auto', 'gaussian', 'eff', 'king'],
                   help='cluster profile for the instP fit (default: from the header)')
    p.add_argument('--gamma', type=float, default=None, help='EFF gamma (default: header, else 3)')
    p.add_argument('--king-c', type=float, default=None,
                   help='King concentration (default: header, else 1.5)')
    p.add_argument('--target-ee', type=float, default=0.65)
    p.add_argument('--slit-width', type=int, default=1)
    p.add_argument('--slit-length', type=float, default=None,
                   help='slit half-length in arcsec (default: max(1", 4 FWHM, 8 R_eff))')
    p.add_argument('--plot', action='store_true', help='show image + slits and the profile fits for every image')
    p.add_argument('--methods', nargs='+', default=['gauss', 'instG', 'instP'],
                   help=f"any of {' '.join(ALL_METHODS)}, or 'all' (default: gauss instG instP)")
    p.add_argument('--deger-ac', type=float, default=DEGER_AC_NGC1365,
                   help='aperture correction of the deger method (default 1.752, NGC1365 V)')
    p.add_argument('--snr', type=float, default=None,
                   help='add noise: source S/N within 0.158" (default: image as it is)')
    p.add_argument('--seed', type=int, default=None, help='random seed for --snr')
    p.add_argument('--csv', default=None, help='also write the results table to this CSV')
    args = p.parse_args()

    warnings.filterwarnings('ignore')
    paths = []
    for d in args.dirs:
        d = d if os.path.isabs(d) else os.path.join(OUT_ROOT, d)
        paths += sorted(p for p in glob.glob(os.path.join(d, args.pattern))
                        if not os.path.basename(p).startswith('true'))
    if not paths:
        sys.exit('no FITS files found')

    rows = []
    print(f"{'file':44s} {'instP profile':16s} {'dCen':>5s} | {'gR65':>6s} {'gFlux':>6s} | "
          f"{'GReff':>6s} {'GR65':>6s} {'GFlux':>6s} | {'PReff':>6s} {'PR65':>6s} {'PFlux':>6s} {'PMod':>6s}")
    for path in paths:
        try:
            r = run_one(path, target_ee=args.target_ee, slit_width_px=args.slit_width,
                        slit_length_arcsec=args.slit_length, profile=args.profile,
                        gamma=args.gamma, king_c=args.king_c, plot=args.plot,
                        methods='all' if args.methods == ['all'] else args.methods,
                        deger_ac=args.deger_ac, snr=args.snr, seed=args.seed)
        except Exception as e:
            print(f'{os.path.basename(path):44s} FAILED: {e}')
            continue
        rows.append(r)
        print(f"{r['file']:44s} {r['profile']:16s} {r['dcen_px']:5.3f} | {r['r65_gauss']:6.3f} {r['flux_gauss']:6.3f} | "
              f"{r['reff_instG']:6.3f} {r['r65_instG']:6.3f} {r['flux_instG']:6.3f} | "
              f"{r['reff_instP']:6.3f} {r['r65_instP']:6.3f} {r['flux_instP']:6.3f} "
              f"{r['fluxmod_instP']:6.3f} | deger {r['flux_deger']:6.3f} growth20 {r['flux_growth20']:6.3f}",
              flush=True)

    print('\nFlux / true flux, median [min-max]:')
    groups = {}
    for r in rows:
        groups.setdefault(_group(r['file']), []).append(r)
    for gname, sel in groups.items():
        line = f'  {gname:15s} N={len(sel):3d}'
        for key, lab in (('flux_gauss', 'Gaussian'), ('flux_instG', 'PSF(x)Gauss'),
                         ('flux_instP', 'PSF(x)profile'), ('flux_deger', 'Deger'),
                         ('flux_growth20', 'growth20')):
            v = np.array([r[key] for r in sel], dtype=float)
            if np.isfinite(v).any():
                line += (f'   {lab}: {np.nanmedian(v):.3f} '
                         f'[{np.nanmin(v):.3f}-{np.nanmax(v):.3f}]')
        print(line)

    if args.csv:
        from astropy.table import Table
        Table(rows=rows).write(args.csv, overwrite=True)
        print(f'\nWrote {args.csv}')


if __name__ == '__main__':
    main()
