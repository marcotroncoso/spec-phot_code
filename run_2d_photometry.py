"""
2D model-fit photometry on an existing Photometry object (not yet part of the
pipeline; see fit2d.py for the fit itself).

For every source and band the image pixels within r_fit of the instP centre are
fitted with

    flux * [profile(R_eff) (x) instrument PSF] + constant background

(no slits, no aperture; neighbours neither masked nor fitted). The flux is the
model's total flux. Everything else is taken from the instP run, so the two
are directly comparable:
  centre    centroid_ra/dec_instP_<band> (= native's centroid)
  profile   profile_instP (EFF gamma=3 for YCL/MAP, King c=1.5 for OGCC)
  PSF       p.instrument_psf[band] (galaxy-matched)
  fit disc  instrument_slit_length = max(1", 4 FWHM), grown to min(5 R_eff, 3")
  R_eff cap max_r_eff_pc (20 pc), distance PHANGS_DMPC
  MW        corr_flux(flux, R_band, p.eb_v), as the pipeline
Bands: by default only where native found real emission (aperture_source
'own'); the others get NaN (all_bands=True fits them anyway).

Modes (any subset):
  'fixed'   centre at the instP centre, profile_instP (EFF3 / King1.5, or EFF with
            gamma fitted when profile_instP is 'effA-B', i.e. [profile] eff_gamma_free)
  'free'    centre fitted (|dx|,|dy| <= 1.5 px), profile_instP
  'Gfixed'  as 'fixed' with a Gaussian source profile (the 2D analogue of instG)
  'Gfree'   as 'free' with a Gaussian source profile

Columns added to p.photometry_table, for each mode and each band:
  flux_f2d<mode>_<BAND>, flux_f2d<mode>_<BAND>_err   [mJy, MW-dereddened]
  reff_pc_f2d<mode>_<BAND>, chi2red_f2d<mode>_<BAND>, shift_px_f2d<mode>_<BAND>,
  converged_f2d<mode>_<BAND>, time_f2d<mode>_<BAND> [s],
  ra/dec_f2d<mode>_<BAND> (fitted centre), bkg_f2d<mode>_<BAND> (fitted constant,
  image units/px), sigma_f2d<mode>_<BAND> (pixel noise used), rfit_f2d<mode>_<BAND> ["],
  gamma_f2d<mode>_<BAND> (EFF power index of the model; NaN for Gaussian / King)
and saved to output/<gal>/<gal>_photometry_2d.fits (ID_PHANGS_CLUSTER + every
f2d column in the table, so modes run separately accumulate in the file);
load them back later with load_2d_photometry(p).

Use from a notebook, after the pipeline has run (or loaded its caches) with
p.run_native_photometry(bands, ..., instrument_fit=True):

    from run_2d_photometry import run_2d_photometry, load_2d_photometry
    t2d = run_2d_photometry(p, bands, modes=('fixed', 'free', 'Gfixed', 'Gfree'), n_workers=16)
    plot_2d_photometry(p, bands, mode='free')          # one PNG per source

Or from the shell (loads the galaxy's caches exactly as run_full_pipeline):
    python run_2d_photometry.py --galaxy NGC1365-1.15asec --workers 16 [--modes fixed free Gfixed Gfree] [--plot]
    python run_2d_photometry.py --galaxy NGC3351 --catalog-mode --modes Gfixed Gfree --save-v2
(--catalog-mode: the full-catalog run in output_catalog/, loaded as
run_full_pipeline._process_one_galaxy_catalog; modes saved earlier are kept;
--save-v2 also writes the 2D fluxes and errors into <gal>_photometry_v2.fits,
the table with all the other fluxes: merge_2d_into_v2.)
"""
import argparse
import logging
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = _HERE                      # the repository root (config.ini, the pipeline modules)
sys.path.insert(0, _ROOT)

import astropy.units as u  # noqa: E402
from astropy.coordinates import SkyCoord  # noqa: E402
from astropy.table import Table  # noqa: E402

from native_phot_aux_func import instrument_slit_length, profile_spec_from_tag  # noqa: E402
from phot_aux_func import corr_flux  # noqa: E402
from fit2d import fit_2d_instrument  # noqa: E402

logger = logging.getLogger('run_2d_photometry')
FIELDS = ('flux', 'flux_err', 'reff_pc', 'chi2red', 'shift_px', 'converged', 'time',
          'ra', 'dec', 'bkg', 'sigma', 'rfit',      # last five: what the plots need to redraw the model
          'gamma')                                  # EFF power index of the model (NaN for Gaussian / King)
MODES = ('fixed', 'free', 'Gfixed', 'Gfree')


def _col(field, mode, band):
    if field == 'flux':
        return f'flux_f2d{mode}_{band}'
    if field == 'flux_err':
        return f'flux_f2d{mode}_{band}_err'
    return f'{field}_f2d{mode}_{band}'


def _spec_from_tag(tag, gamma=None):
    """profile spec of a profile_instP tag ('eff3.0', 'eff2.0-6.0' = gamma fitted,
    'king1.5', 'gauss'); gamma (finite) fixes the EFF gamma, to redraw a fit"""
    return profile_spec_from_tag(tag, gamma=gamma)


def _mode_spec(mode, profile_tag):
    """source profile of a mode: Gaussian for 'G*' modes, else the instP profile"""
    return _spec_from_tag('gaussian' if mode.startswith('G') else profile_tag)


def _mode_free(mode):
    return mode.lower().endswith('free')


def _to_mjy(val_img, bd):
    conv = bd.get('unit_convert')
    return (conv(val_img, bd['header']) if conv is not None else val_img) * 1e3


# worker state (inherited at fork; images are not copied)
_CTX = {}


def _init(ctx):
    global _CTX
    _CTX = ctx
    for name in ('analysis', 'native_phot_aux_func', 'instrument_psf'):
        logging.getLogger(name).setLevel(logging.ERROR)


def _fit_source(task):
    """all bands and modes of one source -> dict of column -> value"""
    ctx = _CTX
    out = {}
    arcsec_per_pc = 206265.0 / (task['dist'] * 1e6) if np.isfinite(task['dist']) and task['dist'] > 0 else np.nan
    max_r_eff = ctx['max_r_eff_pc'] * arcsec_per_pc if np.isfinite(arcsec_per_pc) else 0.5
    for band, b in task['bands'].items():
        bd = ctx['band_data'][band.lower()]
        for mode in ctx['modes']:
            for f in FIELDS:
                out[_col(f, mode, band)] = False if f == 'converged' else np.nan
        if b is None:                                   # no fit in this band
            continue
        xs, ys = (float(v) for v in bd['wcs'].world_to_pixel(SkyCoord(b['ra'] * u.deg, b['dec'] * u.deg)))
        if not (np.isfinite(xs) and np.isfinite(ys)):
            continue
        mw = float(corr_flux(1.0, bd['r_value'], ctx['eb_v'])) if bd.get('r_value') is not None and ctx['eb_v'] else 1.0
        kw = dict(r_fit_arcsec=instrument_slit_length(bd, ctx['slit_length_arcsec']), max_r_eff_arcsec=max_r_eff)
        for mode in ctx['modes']:
            try:
                f = fit_2d_instrument(bd['image'], xs, ys, bd['psf_model'], _mode_spec(mode, task['profile']),
                                      free_center=_mode_free(mode), **kw)
            except Exception as e:                      # keep the batch alive
                logger.debug(f'2D fit {band} {mode} failed: {e}')
                continue
            out[_col('flux', mode, band)] = _to_mjy(f.flux, bd) * mw
            out[_col('flux_err', mode, band)] = abs(_to_mjy(f.flux_err, bd)) * mw
            out[_col('reff_pc', mode, band)] = f.r_eff_arcsec / arcsec_per_pc if np.isfinite(arcsec_per_pc) else np.nan
            out[_col('chi2red', mode, band)] = f.chi2_red
            out[_col('shift_px', mode, band)] = f.shift_px
            out[_col('converged', mode, band)] = bool(f.converged)
            out[_col('time', mode, band)] = f.time_s
            c = bd['wcs'].pixel_to_world(f.xc, f.yc)
            out[_col('ra', mode, band)] = float(c.ra.deg)
            out[_col('dec', mode, band)] = float(c.dec.deg)
            out[_col('bkg', mode, band)] = f.background
            out[_col('sigma', mode, band)] = f.sigma_pix
            out[_col('rfit', mode, band)] = f.r_fit_arcsec
            out[_col('gamma', mode, band)] = f.gamma
    return task['i'], out


def run_2d_photometry(p, filter_wavelengths, modes=('fixed', 'free'), bands=None, all_bands=False,
                      max_r_eff_pc=20.0, slit_length_arcsec=1.0, n_workers=1, idd=None,
                      save=True, out_path=None, verbose=True):
    """
    2D model-fit photometry of every source (or idd: one index / list) in the
    bands of filter_wavelengths that have an instrument PSF. Adds the columns
    listed in the module docstring to p.photometry_table, saves them (with
    ID_PHANGS_CLUSTER) to out_path (default output/<gal>/<gal>_photometry_2d.fits)
    and returns that table. Requires p.run_native_photometry(..., instrument_fit=True)
    to have run (fresh or from its caches).
    """
    t = p.photometry_table
    if t is None or not getattr(p, 'instrument_psf', None) or not getattr(p, 'native_band_calib', None):
        raise RuntimeError('run p.run_native_photometry(bands, instrument_fit=True) first.')
    modes = tuple(modes)
    bad = [m for m in modes if m not in MODES]
    if bad:
        raise ValueError(f'unknown modes {bad}; choose from {MODES}')
    use = [b for b in filter_wavelengths
           if b.lower() in p.native_band_calib and b.lower() in p.instrument_psf
           and f'centroid_ra_instP_{b}' in t.colnames and (bands is None or b in bands)]
    if not use:
        raise RuntimeError('no band with native calibration, an instrument PSF and instP results.')
    band_data = {b.lower(): dict(p.native_band_calib[b.lower()], psf_model=p.instrument_psf[b.lower()]) for b in use}
    idds = list(range(len(t))) if idd is None else ([int(idd)] if np.isscalar(idd) else [int(i) for i in idd])

    tasks = []
    for i in idds:
        row = t[i]
        tb = {}
        for b in use:
            own = str(row[f'aperture_source_native_{b}']).strip() == 'own' \
                if f'aperture_source_native_{b}' in t.colnames else False
            ra, dec = float(row[f'centroid_ra_instP_{b}']), float(row[f'centroid_dec_instP_{b}'])
            if not (np.isfinite(ra) and np.isfinite(dec)):
                ra, dec = float(row['PHANGS_RA']), float(row['PHANGS_DEC'])
            tb[b] = dict(ra=ra, dec=dec) if (own or all_bands) else None
        tasks.append(dict(i=i, bands=tb,
                          profile=str(row['profile_instP']).strip() if 'profile_instP' in t.colnames else 'eff3.0',
                          dist=float(row['PHANGS_DMPC']) if 'PHANGS_DMPC' in t.colnames else np.nan))
    ctx = dict(band_data=band_data, modes=modes, max_r_eff_pc=max_r_eff_pc,
               slit_length_arcsec=slit_length_arcsec, eb_v=float(getattr(p, 'eb_v', 0.0) or 0.0))

    t0 = time.time()
    results = {}
    if n_workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context('fork'),
                                 initializer=_init, initargs=(ctx,)) as ex:
            for k, (i, out) in enumerate(ex.map(_fit_source, tasks, chunksize=1)):
                results[i] = out
                if verbose and (k + 1) % 10 == 0:
                    print(f'  2D photometry: {k + 1}/{len(tasks)} sources, {time.time() - t0:.0f} s', flush=True)
    else:
        _init(ctx)
        for k, task in enumerate(tasks):
            i, out = _fit_source(task)
            results[i] = out
            if verbose:
                print(f'  2D photometry: {k + 1}/{len(tasks)} sources, {time.time() - t0:.0f} s', flush=True)

    cols = [_col(f, m, b) for b in use for m in modes for f in FIELDS]
    for c in cols:
        is_bool = c.startswith('converged_')
        arr = (np.asarray(t[c]).copy() if c in t.colnames
               else (np.zeros(len(t), bool) if is_bool else np.full(len(t), np.nan)))
        for i, out in results.items():
            arr[i] = out[c]
        t[c] = arr
    out_tab = Table()
    out_tab['ID_PHANGS_CLUSTER'] = t['ID_PHANGS_CLUSTER'] if 'ID_PHANGS_CLUSTER' in t.colnames else np.arange(len(t))
    for c in t.colnames:                    # every f2d column, also from modes run earlier
        if '_f2d' in c and any(c.startswith(f'{f}_f2d') for f in FIELDS):
            out_tab[c] = t[c]
    done = [m for m in MODES if _col('flux', m, use[0]) in out_tab.colnames]
    out_tab.meta.update(GALAXY=str(p.gal), MODES=','.join(done), MAXREFF=float(max_r_eff_pc),
                        SLITLEN=float(slit_length_arcsec), EBV=ctx['eb_v'], ALLBANDS=bool(all_bands),
                        NOTE='2D model fit, run_2d_photometry.py')
    if save:
        out_path = out_path or os.path.join(p.base_dir, p.gal, f'{p.gal}_photometry_2d.fits')
        out_tab.write(out_path, overwrite=True)
        if verbose:
            print(f'  2D photometry saved -> {out_path}', flush=True)
    if verbose:
        print(f'  2D photometry: {len(tasks)} sources x {len(use)} bands x {len(modes)} modes in '
              f'{time.time() - t0:.0f} s', flush=True)
    return out_tab


def _id_key(v):
    """ID_PHANGS_CLUSTER as a dict key: integers (also 384.0) as int, anything else as stripped text"""
    try:
        f = float(v)
        return int(f) if f == int(f) else str(v).strip()
    except (TypeError, ValueError):
        return v.decode().strip() if isinstance(v, bytes) else str(v).strip()


def merge_2d_into_v2(v2_path, t2d):
    """Write the 2D fluxes and errors (flux_f2d<mode>_<BAND>[_err]) of t2d into
    the main photometry table v2_path, matched by ID_PHANGS_CLUSTER; existing
    f2d columns are replaced, everything else is left as it is. Only flux and
    error: all 2D fields would pass FITS's 999-column limit (the rest stay in
    <gal>_photometry_2d.fits)."""
    from astropy.io import fits
    cols = [c for c in t2d.colnames if c.startswith('flux_f2d')]
    pos = {_id_key(v): k for k, v in enumerate(t2d['ID_PHANGS_CLUSTER'])}
    with fits.open(v2_path) as h:
        hdu0 = h[0].copy()
        hdr1 = h[1].header
        tab = Table(h[1].data)
    idx = np.array([pos.get(_id_key(v), -1) for v in tab['ID_PHANGS_CLUSTER']])
    for c in cols:
        vals = np.asarray(t2d[c], float)
        tab[c] = np.where(idx >= 0, vals[np.clip(idx, 0, None)], np.nan)
    hdu1 = fits.BinTableHDU(tab, name=hdr1.get('EXTNAME', 'PHOTOMETRY'))
    fits.HDUList([hdu0, hdu1]).writeto(v2_path, overwrite=True)
    return cols


def load_2d_photometry(p, path=None):
    """Add the saved 2D columns to p.photometry_table (matched by ID_PHANGS_CLUSTER)."""
    path = path or os.path.join(p.base_dir, p.gal, f'{p.gal}_photometry_2d.fits')
    s = Table.read(path)
    t = p.photometry_table
    pos = {_id_key(v): k for k, v in enumerate(s['ID_PHANGS_CLUSTER'])}
    idx = np.array([pos.get(_id_key(v), -1) for v in t['ID_PHANGS_CLUSTER']])
    for c in s.colnames:
        if c == 'ID_PHANGS_CLUSTER':
            continue
        vals = np.asarray(s[c])
        fill = False if vals.dtype.kind == 'b' else np.nan
        t[c] = np.where(idx >= 0, vals[np.clip(idx, 0, None)], fill)
    return s


# ─────────────────────────────────────────────────────────────────────────────
# plots
# ─────────────────────────────────────────────────────────────────────────────

def _cutout(img, x0, y0, pix, half_arcsec):
    """(sub-image, extent in arcsec relative to (x0, y0), slices)"""
    h = int(np.ceil(half_arcsec / pix))
    ix, iy = int(np.round(x0)), int(np.round(y0))
    ny, nx = img.shape
    ys, xs = slice(max(iy - h, 0), min(iy + h + 1, ny)), slice(max(ix - h, 0), min(ix + h + 1, nx))
    ext = [(xs.start - 0.5 - x0) * pix, (xs.stop - 0.5 - x0) * pix, (ys.start - 0.5 - y0) * pix,
           (ys.stop - 0.5 - y0) * pix]
    return np.asarray(img[ys, xs], float), ext, xs, ys


def _norm(sub, ext, sky, sigma, r_peak):
    """asinh stretch from sky - 1 sigma to the brightest pixel within r_peak of the centre"""
    from astropy.visualization import ImageNormalize, AsinhStretch
    finite = sub[np.isfinite(sub)]
    if finite.size == 0:
        return None
    ny, nx = sub.shape
    gx = np.linspace(ext[0], ext[1], nx + 1)[:-1] + (ext[1] - ext[0]) / nx / 2
    gy = np.linspace(ext[2], ext[3], ny + 1)[:-1] + (ext[3] - ext[2]) / ny / 2
    rr = np.hypot(*np.meshgrid(gx, gy))
    m = (rr <= max(r_peak, (ext[1] - ext[0]) / nx)) & np.isfinite(sub)
    lo = sky - sigma if np.isfinite(sky) and np.isfinite(sigma) else np.percentile(finite, 5)
    hi = float(np.max(sub[m])) if m.any() else np.percentile(finite, 99.9)
    if not hi > lo:
        hi = lo + 1.0
    return ImageNormalize(vmin=lo, vmax=hi, stretch=AsinhStretch(0.1), clip=True)


def plot_2d_photometry(p, filter_wavelengths, idd=None, bands=None, mode='free', show_instp=True,
                       out_dir=None, half_arcsec=None, dpi=100, verbose=True):
    """
    One PNG per source (output/<gal>/photometry_2d_plots/<i>_<class>_2d<mode>.png),
    one row per band:
        data | 2D model | data - 2D model | data - instP model  (last if show_instp)
    The 2D model is redrawn from the saved fit (flux, R_eff, centre, background):
    flux x [profile (x) instrument PSF] + constant, exactly what was fitted. The
    instP column is the instP result drawn in 2D (flux_instP at its R_eff and
    centre, on its annulus sky), for comparison. Images: asinh stretch from the
    cutout itself (sky - 1 sigma to the source peak); residuals: +-5 sigma of the
    pixel noise used in the fit. Dashed circle: the 2D fit disc; solid: instP R65.
    Requires run_2d_photometry (or load_2d_photometry) for this mode first.
    """
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    t = p.photometry_table
    use = [b for b in filter_wavelengths if _col('flux', mode, b) in t.colnames
           and b.lower() in p.native_band_calib and b.lower() in p.instrument_psf
           and (bands is None or b in bands)]
    if not use:
        raise RuntimeError(f"no 2D columns for mode '{mode}' — run run_2d_photometry(p, bands, modes=('{mode}',)) "
                           f"or load_2d_photometry(p) first.")
    if _col('ra', mode, use[0]) not in t.colnames:
        raise RuntimeError('these 2D results have no centre/background columns (older run) — rerun run_2d_photometry.')
    out_dir = out_dir or os.path.join(p.base_dir, p.gal, 'photometry_2d_plots')
    os.makedirs(out_dir, exist_ok=True)
    idds = list(range(len(t))) if idd is None else ([int(idd)] if np.isscalar(idd) else [int(i) for i in idd])
    ncol = 4 if show_instp else 3
    circ = np.linspace(0, 2 * np.pi, 120)
    eb_v = float(getattr(p, 'eb_v', 0.0) or 0.0)
    for i in idds:
        row = t[i]
        cls = str(row['PHANGS_CC_CLASS_ML']).strip() if 'PHANGS_CC_CLASS_ML' in t.colnames else ''
        spec = _spec_from_tag(row['profile_instP'] if 'profile_instP' in t.colnames else 'eff3.0')  # instP model
        spec2d = _mode_spec(mode, row['profile_instP'] if 'profile_instP' in t.colnames else 'eff3.0')
        dist = float(row['PHANGS_DMPC']) if 'PHANGS_DMPC' in t.colnames else np.nan
        arcsec_per_pc = 206265.0 / (dist * 1e6) if np.isfinite(dist) and dist > 0 else np.nan
        height = 3.0 * len(use) + 1.0
        fig = plt.figure(figsize=(3.4 * ncol + 0.6, height))
        gs = GridSpec(len(use), ncol, figure=fig, left=0.07, right=0.98, top=1 - 1.0 / height, bottom=0.025,
                      wspace=0.06, hspace=0.3)
        for r_i, b in enumerate(use):
            bd = p.native_band_calib[b.lower()]
            psf = p.instrument_psf[b.lower()]
            img, pix = bd['image'], bd['pixel_scale_arcsec']
            mjy_per_unit = _to_mjy(1.0, bd)
            mw = float(corr_flux(1.0, bd['r_value'], eb_v)) if bd.get('r_value') is not None and eb_v else 1.0
            g = {f: float(row[_col(f, mode, b)]) for f in ('flux', 'flux_err', 'reff_pc', 'chi2red', 'shift_px',
                                                          'ra', 'dec', 'bkg', 'sigma', 'rfit')}
            g['gamma'] = float(row[_col('gamma', mode, b)]) if _col('gamma', mode, b) in t.colnames else np.nan
            tag = row['profile_instP'] if 'profile_instP' in t.colnames else 'eff3.0'
            spec2d = _mode_spec(mode, tag) if mode.startswith('G') else _spec_from_tag(tag, g['gamma'])
            spec = _spec_from_tag(tag, float(row[f'gamma_instP_{b}']) if f'gamma_instP_{b}' in t.colnames
                                  else None)
            has_fit = np.isfinite(g['flux']) and np.isfinite(g['ra'])
            # display centre: the 2D centre, else instP's, else the catalog
            ra, dec = ((g['ra'], g['dec']) if has_fit else
                       (float(row[f'centroid_ra_instP_{b}']), float(row[f'centroid_dec_instP_{b}'])))
            if not (np.isfinite(ra) and np.isfinite(dec)):
                ra, dec = float(row['PHANGS_RA']), float(row['PHANGS_DEC'])
            x0, y0 = (float(v) for v in bd['wcs'].world_to_pixel(SkyCoord(ra * u.deg, dec * u.deg)))
            r65 = float(row[f'r65_instP_{b}']) if f'r65_instP_{b}' in t.colnames else np.nan
            half = half_arcsec or max(g['rfit'] if np.isfinite(g['rfit']) else 0.0,
                                      4.6 * r65 if np.isfinite(r65) else 0.0, 0.5)
            sub, ext, xs, ys = _cutout(img, x0, y0, pix, half)
            shape = sub.shape
            sky = g['bkg']
            if not np.isfinite(sky) and f'bkg_median_instP_{b}' in t.colnames:
                sky = float(row[f'bkg_median_instP_{b}'])
            sig = g['sigma'] if np.isfinite(g['sigma']) else float(np.nanstd(sub))
            norm = _norm(sub, ext, sky, sig, r65 if np.isfinite(r65) else 0.15)

            model = None
            if has_fit:
                reff_as = g['reff_pc'] * arcsec_per_pc if np.isfinite(arcsec_per_pc) else 0.0
                kw = (dict(profile=spec2d['profile'], r_eff_arcsec=reff_as, gamma=spec2d['gamma'],
                           c=spec2d['king_c']) if reff_as > 0 else {})
                model = psf.render(x0 - xs.start, y0 - ys.start, shape, flux=g['flux'] / mw / mjy_per_unit,
                                   **kw) + g['bkg']
            inst = None
            if show_instp and f'flux_instP_{b}' in t.colnames:
                fi, ri = float(row[f'flux_instP_{b}']), float(row[f'reff_instP_{b}'])
                ci = (float(row[f'centroid_ra_instP_{b}']), float(row[f'centroid_dec_instP_{b}']))
                bi = float(row[f'bkg_median_instP_{b}']) if f'bkg_median_instP_{b}' in t.colnames else 0.0
                if np.isfinite(fi) and np.isfinite(ri) and np.isfinite(ci[0]):
                    xi, yi = (float(v) for v in bd['wcs'].world_to_pixel(SkyCoord(ci[0] * u.deg, ci[1] * u.deg)))
                    kw = (dict(profile=spec['profile'], r_eff_arcsec=ri, gamma=spec['gamma'], c=spec['king_c'])
                          if ri > 0 else {})
                    inst = psf.render(xi - xs.start, yi - ys.start, shape, flux=fi / mw / mjy_per_unit, **kw) + bi

            panels = [(sub, 'data', 'img'), (model, f'2D model ({mode})', 'img'),
                      (None if model is None else sub - model, 'data - 2D model', 'res')]
            if show_instp:
                panels.append((None if inst is None else sub - inst, 'data - instP model', 'res'))
            for j, (im, ttl, kind) in enumerate(panels):
                ax = fig.add_subplot(gs[r_i, j])
                if im is None:
                    ax.text(0.5, 0.5, 'no 2D fit' if j < 3 else 'no instP flux', ha='center', va='center',
                            transform=ax.transAxes, fontsize=9)
                    ax.set_facecolor('#f2f1ec')
                elif kind == 'img':
                    ax.imshow(im, origin='lower', cmap='gray_r', norm=norm, extent=ext, interpolation='nearest')
                else:
                    ax.imshow(im, origin='lower', cmap='RdBu_r', vmin=-5 * sig, vmax=5 * sig, extent=ext,
                              interpolation='nearest')
                if np.isfinite(g['rfit']):
                    ax.plot(g['rfit'] * np.cos(circ), g['rfit'] * np.sin(circ), '--', color='#eda100', lw=0.9)
                if np.isfinite(r65):
                    ax.plot(r65 * np.cos(circ), r65 * np.sin(circ), '-', color='#4a3aa7', lw=0.9, alpha=0.8)
                ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
                ax.tick_params(labelsize=6.5, labelleft=(j == 0))
                if r_i == 0:
                    ax.set_title(ttl + ('  (±5σ)' if kind == 'res' else ''), fontsize=10)
                if j == 0:
                    ax.set_ylabel(f'{b}', fontsize=11, fontweight='bold')
                if j == 2 and model is not None:
                    fi = float(row[f'flux_instP_{b}']) if f'flux_instP_{b}' in t.colnames else np.nan
                    txt = (f'2D  {g["flux"]:.4g} ± {g["flux_err"]:.2g} mJy\n'
                           f'R_eff {g["reff_pc"]:.1f} pc'
                           + (f', γ {g["gamma"]:.2f}' if np.isfinite(g['gamma']) else '')
                           + f', χ²_red {g["chi2red"]:.2f}, shift {g["shift_px"]:.2f} px\n'
                           f'instP {fi:.4g} mJy  (2D / instP = {g["flux"] / fi:.3f})' if np.isfinite(fi) and fi
                           else f'2D  {g["flux"]:.4g} ± {g["flux_err"]:.2g} mJy\nR_eff {g["reff_pc"]:.1f} pc')
                    ax.text(0.02, 0.02, txt, transform=ax.transAxes, fontsize=6.8, va='bottom',
                            bbox=dict(fc='white', ec='none', alpha=0.85))
        fig.suptitle(f'{p.gal} — source {i} ({cls}; 2D {spec2d["profile"]}, instP {spec["profile"]})   '
                     f'2D model fit ({mode}) vs data; '
                     f'dashed: 2D fit disc, purple: instP R65', x=0.01, ha='left', fontsize=11, y=1 - 0.3 / height)
        fig.savefig(os.path.join(out_dir, f'{i}_{cls}_2d{mode}.png'), dpi=dpi, facecolor='white')
        plt.close(fig)
        if verbose:
            print(f'  2D photometry plot saved: source {i}', flush=True)
    if verbose:
        print(f'  2D photometry plots -> {out_dir}', flush=True)


def _load_galaxy(token):
    """Photometry object for one galaxy token (e.g. 'NGC1365-1.15asec'), loaded
    from the pipeline caches exactly as run_full_pipeline._process_one_galaxy."""
    import run_full_pipeline as rfp
    from analysis import Photometry
    gal = token.split('-')[0]
    os.chdir(_ROOT)
    cfg = rfp.load_config('config.ini')
    p = Photometry([token, gal], base_dir=cfg['output_dir'], psf_filepath=cfg['psf_filepath'],
                   psf_source=cfg['psf_source'])
    p.load_isf_results()
    p.load_images(rfp.find_hst_band_paths(cfg['hst_root'], gal), rfp.find_jwst_band_paths(cfg['jwst_root'], gal.lower()),
                  muse_path=cfg['muse_dap_root'])
    p.filter_by_common_fov(plot=False)
    p.load_or_measure_psf()
    p.convolve_images()
    bands = {b: v for b, v in rfp.FILTER_WAVELENGTHS.items() if b.lower() in p.hst_images or b.lower() in p.jwst_images}
    try:
        p._find_galaxy_eb_v()
    except Exception:
        p.eb_v = 0.0
    p.run_spectro_photometry(bands, n_workers=1, verbose=False)      # cache
    p.hst_conv.clear()
    p.jwst_conv.clear()
    p.measure_native_psf_shape()
    p.run_native_photometry(bands, n_workers=1, verbose=False, plot_diagnostics=False, instrument_fit=True,
                            instrument_slit_width_px=cfg['instrument_slit_width_px'],
                            profile_by_class=cfg['profile_by_class'], psf_blur=cfg['psf_blur'])   # caches
    return p, bands


def _load_galaxy_catalog(token, selection=None):
    """Photometry object of the full-catalog run (output_catalog/), loaded from
    its caches as run_full_pipeline._process_one_galaxy_catalog builds it."""
    import run_full_pipeline as rfp
    from analysis import Photometry
    gal = token.split('-')[0]
    os.chdir(_ROOT)
    cfg = rfp.load_config('config.ini')
    p = Photometry([token, gal], base_dir=rfp.catalog_output_dir(cfg), psf_filepath=cfg['psf_filepath'],
                   psf_source=cfg['psf_source'])
    p.load_images(rfp.find_hst_band_paths(cfg['hst_root'], gal), rfp.find_jwst_band_paths(cfg['jwst_root'], gal.lower()),
                  muse_path=None)
    p.load_catalog_sources(cfg['cat_path'], selection=selection or cfg['catalog_selection'],
                           classes=cfg['catalog_classes'],
                           drop_outside=cfg['catalog_drop_outside'])
    p.filter_by_common_fov(plot=False, include_muse=False)
    p.load_or_measure_psf()
    bands = {b: v for b, v in rfp.FILTER_WAVELENGTHS.items() if b.lower() in p.hst_images or b.lower() in p.jwst_images}
    try:
        p._find_galaxy_eb_v()
    except Exception:
        p.eb_v = 0.0
    p.init_photometry_table_from_catalog()
    p.measure_native_psf_shape()
    n_cached = len(Table.read(os.path.join(p.base_dir, gal, f'{gal}_photometry_native.fits')))
    if n_cached != len(p.photometry_table):     # never silently recompute the full-catalog caches
        raise RuntimeError(f'{len(p.photometry_table)} catalog sources but the cache has {n_cached}: '
                           f'wrong --catalog-selection (machine / human)?')
    p.run_native_photometry(bands, n_workers=1, verbose=False, plot_diagnostics=False, instrument_fit=True,
                            instrument_slit_width_px=cfg['instrument_slit_width_px'],
                            profile_by_class=cfg['profile_by_class'], psf_blur=cfg['psf_blur'])   # caches
    return p, bands


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--galaxy', required=True, help='galaxy token, as for run_full_pipeline (e.g. NGC1365-1.15asec)')
    ap.add_argument('--modes', nargs='+', default=['fixed', 'free'], choices=list(MODES))
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--all-bands', action='store_true', help='also fit bands without own native emission')
    ap.add_argument('--idd', type=int, nargs='*', default=None, help='only these source indices (test)')
    ap.add_argument('--out', default=None, help='output FITS (default output/<gal>/<gal>_photometry_2d.fits)')
    ap.add_argument('--plot', action='store_true', help='also write the per-source 2D plots')
    ap.add_argument('--plot-dir', default=None, help='plot folder (default output/<gal>/photometry_2d_plots)')
    ap.add_argument('--catalog-mode', action='store_true', help='the full-catalog run in output_catalog/')
    ap.add_argument('--catalog-selection', default=None, choices=['machine', 'human'],
                    help='catalog-mode class selection (default: config.ini catalog_selection)')
    ap.add_argument('--save-v2', action='store_true', help='also write the 2D columns into <gal>_photometry_v2.fits')
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    p, bands = _load_galaxy_catalog(a.galaxy, a.catalog_selection) if a.catalog_mode else _load_galaxy(a.galaxy)
    old = a.out or os.path.join(p.base_dir, p.gal, f'{p.gal}_photometry_2d.fits')
    if os.path.exists(old):                     # keep the modes run earlier
        load_2d_photometry(p, old)
        print(f'  earlier 2D results loaded from {old}', flush=True)
    run_2d_photometry(p, bands, modes=a.modes, all_bands=a.all_bands, n_workers=a.workers, idd=a.idd,
                      out_path=a.out)
    if a.save_v2:
        v2 = os.path.join(p.base_dir, p.gal, f'{p.gal}_photometry_v2.fits')
        n = len(merge_2d_into_v2(v2, Table.read(old)))
        print(f'  {n} 2D flux/error columns written into {v2}', flush=True)
    if a.plot:
        for mode in a.modes:
            plot_2d_photometry(p, bands, idd=a.idd, mode=mode, out_dir=a.plot_dir)


if __name__ == '__main__':
    main()
