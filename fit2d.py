"""
2D model fit of one cluster: no slits, no aperture.

    image(x, y) = flux * [source profile(R_eff) (x) instrument PSF](x - xc, y - yc) + bkg

fitted to every pixel within r_fit of the start position. Neighbours are
neither masked nor fitted: they stay in the fitted pixels, as they stay in
instP's slits and background annulus. The flux is the fitted model's TOTAL
flux (InstrumentPSF.render has unit total flux, light outside the stamp
included), so no aperture correction is involved.

Mirrors native_phot_aux_func.fit_slit_profile_instrument_psf (instP) so the
only difference is 1D averaged slits vs 2D pixels:
  - same model rendering (InstrumentPSF.render, same profile shapes)
  - flux and constant background by linear least squares for each trial
  - R_eff by a coarse grid (0 + 30 log-spaced values) + bounded refinement
  - same R_eff cap and 'converged' rule (flux > 0 and R_eff below the cap)
free_center=True then refines (R_eff, dx, dy) with Nelder-Mead from the
fixed-centre solution, |dx|, |dy| <= max_shift_px.
spec['gamma_free'] (EFF only): the EFF power index gamma is fitted jointly with
R_eff at the start centre (native_phot_aux_func.fit_reff_gamma, as instP),
within [spec['gamma_min'], spec['gamma_max']]; a free centre is then refined
at that gamma.

Flux error: linear least-squares covariance with one sigma for every pixel
(3-sigma-clipped std of the outer 20% of the fit disc), scaled by
sqrt(reduced chi2). The R_eff uncertainty is not propagated.

Run python fit2d.py for the self-test (noise-free point, 2 pc, 10 pc).
"""
import os
import sys
import time
from collections import namedtuple

import numpy as np
from astropy.stats import sigma_clipped_stats
from scipy.optimize import minimize, minimize_scalar

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

Fit2D = namedtuple('Fit2D', [
    'flux', 'flux_err',          # total model flux and its 1-sigma error (image units)
    'r_eff_arcsec', 'background',
    'xc', 'yc',                  # fitted centre (pixel coords of the input image)
    'shift_px',                  # |fitted centre - start|
    'chi2_red', 'sigma_pix', 'n_pix', 'n_eval', 'r_fit_arcsec',
    'converged', 'time_s',
    'gamma',                     # EFF power index of the model (fitted if gamma_free); NaN otherwise
], defaults=(np.nan,))
NAN_FIT = Fit2D(*([np.nan] * 12), False, np.nan)


def _fit_once(image, x0, y0, psf_model, spec, r_fit_arcsec, free_center, max_r_eff_arcsec,
              max_shift_px, n_grid):
    pix = psf_model.pixel_scale_arcsec
    half = int(np.ceil(r_fit_arcsec / pix)) + 3
    ix, iy = int(np.round(x0)), int(np.round(y0))
    ny, nx = image.shape
    ys, xs = slice(max(iy - half, 0), min(iy + half + 1, ny)), slice(max(ix - half, 0), min(ix + half + 1, nx))
    data = np.asarray(image[ys, xs], float)
    yy, xx = np.mgrid[ys, xs]
    rr = np.hypot(xx - x0, yy - y0) * pix
    use = (rr <= r_fit_arcsec) & np.isfinite(data)
    if use.sum() < 20:
        return None
    d = data[use]
    ring = use & (rr >= 0.8 * r_fit_arcsec)
    sigma = float(sigma_clipped_stats(data[ring], sigma=3.0)[2]) if ring.sum() > 10 else float(np.nanstd(d))
    shape = data.shape
    ox, oy = xs.start, ys.start
    from native_phot_aux_func import fit_reff_gamma, quantize_gamma
    is_eff = spec['profile'] == 'eff'
    gamma_free = bool(spec.get('gamma_free', False)) and is_eff
    prof_kw = dict(profile=spec['profile'], gamma=quantize_gamma(spec['gamma']) if is_eff else spec['gamma'],
                   c=spec['king_c'])
    n_eval = [0]

    def solve(r_eff, dx=0.0, dy=0.0, g=None):
        n_eval[0] += 1
        kw = prof_kw if g is None else dict(prof_kw, gamma=g)
        m = psf_model.render(x0 + dx - ox, y0 + dy - oy, shape, r_eff_arcsec=max(r_eff, 0.0), **kw)[use]
        A = np.vstack([m, np.ones_like(m)]).T
        coef, *_ = np.linalg.lstsq(A, d, rcond=None)
        resid = d - A @ coef
        return float(np.sum(resid ** 2)), coef, A

    # R_eff at the start centre: grid + bounded refinement (as instP)
    grid = np.concatenate([[0.0], np.geomspace(0.0025, max_r_eff_arcsec, n_grid)])
    if gamma_free:
        r_b, g_b, c_b = fit_reff_gamma(lambda r, g: solve(r, g=g)[0], grid,
                                       spec.get('gamma_min', 2.0), spec.get('gamma_max', 6.0))
        if np.isfinite(g_b):
            prof_kw['gamma'] = g_b
        best = dict(r=r_b, dx=0.0, dy=0.0, chi2=c_b)
    else:
        chi2 = np.array([solve(r)[0] for r in grid])
        i = int(np.argmin(chi2))
        best = dict(r=grid[i], dx=0.0, dy=0.0, chi2=chi2[i])
        lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
        if hi > lo:
            res = minimize_scalar(lambda r: solve(r)[0], bounds=(lo, hi), method='bounded', options=dict(xatol=5e-5))
            if res.success and res.fun <= best['chi2']:
                best.update(r=float(res.x), chi2=float(res.fun))

    if free_center:
        scale_r = max(best['r'], 0.01)

        def f(p):
            r, dx, dy = p[0] * scale_r, p[1], p[2]
            pen = 0.0
            if r < 0 or r > max_r_eff_arcsec:
                pen += 1e6 * (1 + abs(r))
                r = float(np.clip(r, 0, max_r_eff_arcsec))
            if abs(dx) > max_shift_px or abs(dy) > max_shift_px:
                pen += 1e6 * (1 + abs(dx) + abs(dy))
                dx, dy = np.clip([dx, dy], -max_shift_px, max_shift_px)
            return solve(r, dx, dy)[0] * (1 + pen)

        p0 = np.array([best['r'] / scale_r, 0.0, 0.0])
        simplex = np.array([p0, p0 + [0.5, 0, 0], p0 + [0, 0.3, 0], p0 + [0, 0, 0.3]])
        res = minimize(f, p0, method='Nelder-Mead',
                       options=dict(initial_simplex=simplex, xatol=1e-3, fatol=1e-6 * best['chi2'], maxfev=300))
        if res.fun < best['chi2']:
            best.update(r=float(np.clip(res.x[0] * scale_r, 0, max_r_eff_arcsec)), dx=float(res.x[1]),
                        dy=float(res.x[2]), chi2=float(res.fun))

    chi2_best, coef, A = solve(best['r'], best['dx'], best['dy'])
    n, k = len(d), 2 + 1 + (2 if free_center else 0) + (1 if gamma_free else 0)
    chi2_red = chi2_best / max(n - k, 1) / sigma ** 2
    cov = np.linalg.pinv(A.T @ A) * sigma ** 2 * chi2_red
    flux = float(coef[0])
    return Fit2D(flux=flux, flux_err=float(np.sqrt(cov[0, 0])), r_eff_arcsec=best['r'], background=float(coef[1]),
                 xc=x0 + best['dx'], yc=y0 + best['dy'], shift_px=float(np.hypot(best['dx'], best['dy'])),
                 chi2_red=float(chi2_red), sigma_pix=sigma, n_pix=int(n), n_eval=n_eval[0],
                 r_fit_arcsec=r_fit_arcsec, converged=bool(flux > 0 and best['r'] < max_r_eff_arcsec * 0.999),
                 time_s=np.nan,
                 gamma=float(prof_kw['gamma']) if is_eff and best['r'] > 0 and not (gamma_free and not np.isfinite(g_b))
                 else np.nan)


def fit_2d_instrument(image, x0, y0, psf_model, spec, r_fit_arcsec=1.0, free_center=False,
                      max_r_eff_arcsec=0.5, max_shift_px=1.5, max_r_fit_arcsec=3.0, n_grid=30):
    """2D fit of PSF (x) profile + constant within r_fit_arcsec of (x0, y0).
    spec = dict(profile='eff'|'king'|'gaussian', gamma=, king_c=[, gamma_free=,
    gamma_min=, gamma_max=]). As instP, the
    fit disc grows to min(5 R_eff, max_r_fit_arcsec) if the first fit needs it.
    Returns Fit2D (image units), NAN_FIT if there are too few pixels."""
    t0 = time.perf_counter()
    fit = _fit_once(image, x0, y0, psf_model, spec, r_fit_arcsec, free_center, max_r_eff_arcsec,
                    max_shift_px, n_grid)
    if fit is not None and fit.converged and 5 * fit.r_eff_arcsec > r_fit_arcsec and r_fit_arcsec < max_r_fit_arcsec:
        fit2 = _fit_once(image, x0, y0, psf_model, spec, min(5 * fit.r_eff_arcsec, max_r_fit_arcsec),
                         free_center, max_r_eff_arcsec, max_shift_px, n_grid)
        if fit2 is not None and fit2.converged:
            fit = fit2._replace(n_eval=fit2.n_eval + fit.n_eval)
    if fit is None:
        return NAN_FIT._replace(time_s=time.perf_counter() - t0)
    return fit._replace(time_s=time.perf_counter() - t0)


def selftest():
    """noise-free isolated EFF3 clusters with the NGC1365 F555W instrument PSF"""
    from mock_neighbours_test import load_real
    from instrument_psf import instrument_psf_for_image
    p = load_real()
    _, header, wcs = p.hst_images['f555w']
    pix = wcs.proj_plane_pixel_scales()[0].to('arcsec').value
    model, _ = instrument_psf_for_image('F555W', header, pix, extra_fwhm_arcsec=0.0)   # theoretical PSF
    spec = dict(profile='eff', gamma=3.0, king_c=1.5)
    arcsec_per_pc = 206265.0 / 19.57e6
    n = 151
    for reff_pc in (0.0, 2.0, 10.0):
        xc, yc = 75.3, 74.8
        kw = dict(r_eff_arcsec=reff_pc * arcsec_per_pc, **{'profile': 'eff', 'gamma': 3.0, 'c': 1.5}) \
            if reff_pc > 0 else {}
        img = model.render(xc, yc, (n, n), flux=1.0, **kw) + 0.01
        for free in (False, True):
            f = fit_2d_instrument(img, round(xc), round(yc) if free else yc, model, spec,
                                  r_fit_arcsec=1.0, free_center=free, max_r_eff_arcsec=20 * arcsec_per_pc)
            if not free:
                f = fit_2d_instrument(img, xc, yc, model, spec, r_fit_arcsec=1.0, free_center=False,
                                      max_r_eff_arcsec=20 * arcsec_per_pc)
            print(f'R_eff {reff_pc:4.1f} pc  {"free " if free else "fixed"}: flux {f.flux:.4f}  '
                  f'R_eff {f.r_eff_arcsec / arcsec_per_pc:5.2f} pc  bkg {f.background:.4f}  '
                  f'centre ({f.xc:.2f}, {f.yc:.2f})  evals {f.n_eval}  {f.time_s:.2f} s')


if __name__ == '__main__':
    selftest()
