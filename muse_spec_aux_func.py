"""
Cluster-only MUSE spectra for the isolated sources.

Two independent ways of separating the cluster light from the local stellar
disc and ISM inside the ISF aperture, both working directly on the MUSE
cube (DATA + STAT), never on the photometry products:

Method 1 ("annulus")
    Sum the cube inside the ISF polygon, subtract N_ap x the CONTINUUM of a
    sigma-clipped annulus spectrum. The annulus's own emission lines are
    replaced by interpolation, so only the stellar background is removed and
    the ISM lines that remain in the cluster spectrum can be measured.

Method 2 ("psf", pampelmuse-style)
    In every wavelength slice fit a linear model to a stamp around the
    cluster: PSF(target) + PSF(neighbouring DR5 clusters) + an extended
    background (2-D polynomial + grid of broad Gaussian patches, ridge
    regularised, as pampelmuse's BackgroundGrid). The PSF amplitude is the
    cluster spectrum (total flux); the extended model inside the aperture is
    the "extra" emission that does not come from the cluster.

At MUSE resolution (copt 1.15" ~ 110 pc at NGC1365) every cluster is a point
source, so "cluster" == the PSF-shaped part of the light. Gas within ~1 PSF
of the cluster (a compact HII region) is PSF-shaped too and is not
separable from the cluster in MUSE alone.

All spectra are in the cube's units, 1e-20 erg/s/cm2/A; variances 1e-40.
"""

import logging
import os

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
from regions import PixCoord, PolygonPixelRegion
from scipy import ndimage
from scipy.interpolate import LSQUnivariateSpline
from scipy.optimize import curve_fit, least_squares, minimize

logger = logging.getLogger(__name__)

C_KMS = 299792.458
MUSE_PIX_ARCSEC = 0.2

# ─────────────────────────────────────────────────────────────────────────────
# Line lists (air wavelengths, A)
# ─────────────────────────────────────────────────────────────────────────────

# Emission (and ISM absorption) lines masked at the source velocity when
# building the continuum.
EMISSION_LINES = {
    'Hb': 4861.33, 'OIII4959': 4958.91, 'OIII5007': 5006.84,
    'NI5198': 5197.90, 'NI5200': 5200.26, 'NII5755': 5754.59,
    'HeI5876': 5875.62, 'NaD1': 5889.95, 'NaD2': 5895.92,
    'OI6300': 6300.30, 'SIII6312': 6312.06, 'OI6364': 6363.78,
    'NII6548': 6548.05, 'Ha': 6562.80, 'NII6583': 6583.45,
    'HeI6678': 6678.15, 'SII6716': 6716.44, 'SII6731': 6730.82,
    'HeI7065': 7065.19, 'ArIII7136': 7135.79, 'OII7320': 7319.99,
    'OII7330': 7330.73, 'ArIII7751': 7751.06, 'Pa12': 8750.47,
    'Pa11': 8862.78, 'Pa10': 9014.91, 'SIII9069': 9068.60, 'Pa9': 9229.01,
}
# Strong sky residuals, masked in the OBSERVED frame.
SKY_LINES = [5577.34, 5889.95, 6300.30, 6363.78, 6863.96, 7276.41, 7340.89,
             7913.71, 8344.60, 8827.10, 8885.85, 8919.63, 8958.10, 9001.47]
LINE_HALF_WIDTH = 12.0     # A, mask half-width around every line
SKY_HALF_WIDTH = 4.0
TELLURIC_BANDS = [(6865., 6930.), (7590., 7700.)]

# Line groups fitted with Gaussians (common velocity and sigma per group).
LINE_GROUPS = {
    'Hb':     ['Hb'],
    'OIII':   ['OIII4959', 'OIII5007'],
    'HaNII':  ['NII6548', 'Ha', 'NII6583'],
    'SII':    ['SII6716', 'SII6731'],
}
FIT_LINES = [l for g in LINE_GROUPS.values() for l in g]

# Wavelength range used (observed A): above ~7000 A sky residuals and
# telluric bands make the cluster spectra unreliable.
WAVE_RANGE = (4850., 7000.)

# Continuum bins (observed A) for broad-band checks and the PSF fit.
CONT_BINS = [(4850., 5400.), (5400., 5950.), (5950., 6500.), (6500., 7000.)]


def muse_lsf_fwhm(wave):
    """MUSE line-spread function FWHM in A (Bacon et al. 2017, eq. 8)."""
    return 5.866e-8 * wave**2 - 9.187e-4 * wave + 6.040


def continuum_mask(wave, z):
    """True where ``wave`` (observed A) is free of emission/ISM/sky lines."""
    good = np.ones(wave.size, bool)
    for lam in EMISSION_LINES.values():
        good &= np.abs(wave - lam * (1 + z)) > LINE_HALF_WIDTH
    for lam in SKY_LINES:
        good &= np.abs(wave - lam) > SKY_HALF_WIDTH
    for lo, hi in TELLURIC_BANDS:
        good &= ~((wave > lo) & (wave < hi))
    return good


# ─────────────────────────────────────────────────────────────────────────────
# Cube access
# ─────────────────────────────────────────────────────────────────────────────

def cube_wave(header):
    n = header['NAXIS3']
    return header['CRVAL3'] + (np.arange(n) + 1 - header['CRPIX3']) * header['CD3_3']


def extract_cutouts(cube_path, centers, half, chunk=100):
    """
    Cut DATA/STAT stamps of (2*half+1)^2 spaxels around every centre in ONE
    sequential pass over the cube.

    Random access with ``section[:, y0:y1, x0:x1]`` reads ~85 s per stamp
    (strided over 3761 planes); streaming whole planes reads the full 45 GB
    cube in ~3 min and serves every source at once.

    centers : dict key -> (x, y) integer pixel centre on the cube grid (0-based)

    Returns dict key -> {'data', 'var' (nl, n, n) float32 with NaN where DQ != 0
    or off the cube, 'x0', 'y0'} plus the wavelength array.
    """
    with fits.open(cube_path, memmap=True) as hdul:
        hd = hdul['DATA'].header
        nl, ny, nx = hd['NAXIS3'], hd['NAXIS2'], hd['NAXIS1']
        wave = cube_wave(hd)
        n = 2 * half + 1
        out = {}
        for k, (xc, yc) in centers.items():
            out[k] = {'data': np.full((nl, n, n), np.nan, np.float32),
                      'var':  np.full((nl, n, n), np.nan, np.float32),
                      'x0': int(xc) - half, 'y0': int(yc) - half}

        for l0 in range(0, nl, chunk):
            l1 = min(l0 + chunk, nl)
            d = np.asarray(hdul['DATA'].section[l0:l1])
            v = np.asarray(hdul['STAT'].section[l0:l1])
            q = np.asarray(hdul['DQ'].section[l0:l1])
            bad = (q != 0) | ~np.isfinite(d) | ~np.isfinite(v) | (v <= 0)
            d = np.where(bad, np.nan, d)
            v = np.where(bad, np.nan, v)
            for c in out.values():
                ys0, xs0 = max(c['y0'], 0), max(c['x0'], 0)
                ys1, xs1 = min(c['y0'] + n, ny), min(c['x0'] + n, nx)
                if ys1 <= ys0 or xs1 <= xs0:
                    continue
                sl = (slice(l0, l1), slice(ys0 - c['y0'], ys1 - c['y0']),
                      slice(xs0 - c['x0'], xs1 - c['x0']))
                c['data'][sl] = d[:, ys0:ys1, xs0:xs1]
                c['var'][sl] = v[:, ys0:ys1, xs0:xs1]
            logger.info(f'    cube planes {l1}/{nl}')
    return out, wave


def save_cutout(path, c):
    np.savez(path, data=c['data'], var=c['var'], x0=c['x0'], y0=c['y0'])


def load_cutout(path, wsel=None):
    """Cached cube stamp; ``wsel`` (bool over the cube channels) trims it in wavelength."""
    z = np.load(path)
    sl = slice(None) if wsel is None else wsel
    return {'data': z['data'][sl], 'var': z['var'][sl],
            'x0': int(z['x0']), 'y0': int(z['y0'])}


# ─────────────────────────────────────────────────────────────────────────────
# Masks
# ─────────────────────────────────────────────────────────────────────────────

def polygon_cutout_mask(sky_region, cel_wcs, x0, y0, shape):
    """Fractional (5x5 subpixel) mask of a sky polygon on a cutout starting at (x0, y0)."""
    pix = sky_region.to_pixel(cel_wcs)
    shifted = PolygonPixelRegion(vertices=PixCoord(pix.vertices.x - x0,
                                                   pix.vertices.y - y0))
    return shifted.to_mask(mode='subpixels', subpixels=5).to_image(shape)


def annulus_mask(ap_frac, gap_px, width_px, exclude=None):
    """
    Ring around the aperture: dilate the polygon by ``gap_px`` (keeps the
    cluster's PSF wings out) and then by ``width_px`` more. ``exclude`` is
    a boolean map of spaxels to drop (other clusters / other segments).
    """
    core = ap_frac > 0.0
    yy, xx = np.mgrid[-int(np.ceil(gap_px + width_px)):int(np.ceil(gap_px + width_px)) + 1,
                      -int(np.ceil(gap_px + width_px)):int(np.ceil(gap_px + width_px)) + 1]
    rr = np.hypot(xx, yy)
    inner = ndimage.binary_dilation(core, structure=rr <= gap_px)
    outer = ndimage.binary_dilation(core, structure=rr <= gap_px + width_px)
    ring = outer & ~inner
    if exclude is not None:
        ring &= ~exclude
    return ring


def neighbour_exclusion(shape, xy_neigh, radius_px, labels=None, target_label=None):
    """
    Spaxels within ``radius_px`` of any neighbouring cluster, plus whole
    watershed segments (``labels``) that host a neighbour.
    """
    yy, xx = np.indices(shape)
    ex = np.zeros(shape, bool)
    for (x, y) in xy_neigh:
        ex |= np.hypot(xx - x, yy - y) <= radius_px
        if labels is not None:
            ix, iy = int(round(x)), int(round(y))
            if 0 <= iy < shape[0] and 0 <= ix < shape[1]:
                lab = labels[iy, ix]
                if lab > 0 and lab != target_label:
                    ex |= labels == lab
    return ex


# ─────────────────────────────────────────────────────────────────────────────
# Method 1 — continuum of a sigma-clipped annulus
# ─────────────────────────────────────────────────────────────────────────────

def interpolate_over_lines(wave, spec, good):
    """Replace masked (line) pixels by linear interpolation over continuum pixels."""
    ok = good & np.isfinite(spec)
    out = spec.copy()
    out[~ok] = np.interp(wave[~ok], wave[ok], spec[ok])
    # no extrapolation beyond the measured range (e.g. < 4750 A in the cube)
    fin = np.where(np.isfinite(spec))[0]
    out[:fin[0]] = np.nan
    out[fin[-1] + 1:] = np.nan
    return out


def empirical_error_scale(wave, flux, err, good, block=300.0, width=11, floor=1.0):
    """
    Factor k(lambda) by which the propagated errors of an extracted spectrum
    must be scaled to match its own pixel-to-pixel scatter.

    The copt cubes are convolved to a common PSF, so neighbouring spaxels are
    strongly correlated and sum(STAT) over an aperture underestimates the
    variance. In blocks of ``block`` A, k = robust std of
    (flux - running median) / err over the line-free pixels, multiplied by
    the factor that corrects a width-``width`` running median's own
    noise; k >= floor. Returns k on the wavelength grid.
    """
    from scipy.ndimage import median_filter
    f = np.where(np.isfinite(flux), flux, np.nan)
    med = median_filter(np.nan_to_num(f), size=width, mode='nearest')
    r = (f - med) / err
    # var(x - median_N(x)) ~ var(x) * (1 + pi/(2N)) for white noise
    corr = np.sqrt(1 + np.pi / (2 * width))
    edges = np.arange(wave[0], wave[-1] + block, block)
    cen, ks = [], []
    for b0, b1 in zip(edges[:-1], edges[1:]):
        s = good & (wave >= b0) & (wave < b1) & np.isfinite(r)
        if s.sum() < 30:
            continue
        mad = np.median(np.abs(r[s] - np.median(r[s])))
        cen.append(0.5 * (b0 + b1))
        ks.append(1.4826 * mad / corr)
    if not ks:
        return np.ones_like(wave)
    k = np.interp(wave, cen, ks)
    return np.maximum(k, floor)


def smooth_continuum(wave, spec, err, good, knot_step=150.0):
    """Cubic spline through the continuum pixels, knots every ``knot_step`` A."""
    ok = good & np.isfinite(spec) & np.isfinite(err) & (err > 0)
    w = wave[ok]
    knots = np.arange(w[0] + knot_step, w[-1] - knot_step / 2, knot_step)
    spl = LSQUnivariateSpline(w, spec[ok], knots, w=1.0 / err[ok], k=3)
    return spl(wave)


def annulus_background(data, var, ring, good, sigma=3.0, collapse_clip=3.0):
    """
    Per-spaxel background spectrum from the annulus spaxels.

    1. Reject whole spaxels whose continuum level is a ``collapse_clip``-sigma
       outlier (bright clumps / dust lanes).
    2. At every wavelength, sigma-clipped median over the surviving spaxels.

    Returns B (nl,), its error (nl,), the number of spaxels used and the
    final spaxel mask.
    """
    spec = data[:, ring]                                   # (nl, nsp)
    level = np.nanmedian(spec[good], axis=0)
    _, med, std = sigma_clipped_stats(level, sigma=collapse_clip)
    keep = np.isfinite(level) & (np.abs(level - med) <= collapse_clip * std)
    spec = spec[:, keep]
    ring_used = ring.copy()
    ring_used[ring] = keep

    _, B, S = sigma_clipped_stats(spec, sigma=sigma, axis=1)
    n_eff = np.sum(np.isfinite(spec), axis=1)
    B_err = 1.253 * S / np.sqrt(np.maximum(n_eff, 1))
    return np.asarray(B), np.asarray(B_err), int(keep.sum()), ring_used


def method_annulus(data, var, ap_frac, ring, wave, z, smooth=False):
    """
    Method 1. C1 = A - N_ap * B_cont.

    Returns dict with A, A_err, B (per spaxel, with lines), B_err, B_cont,
    C1, C1_err, BKG_LINES_AP (= N_ap*(B - B_cont), the annulus ISM lines
    scaled to the aperture), N_ap, N_ann.
    """
    good = continuum_mask(wave, z)
    ok = np.isfinite(data)
    A = np.nansum(data * ap_frac, axis=(1, 2))
    A_var = np.nansum(var * ap_frac**2, axis=(1, 2))
    # aperture spaxels lost to NaN at a given wavelength: rescale the sum
    cov = np.sum(ok * ap_frac, axis=(1, 2)) / ap_frac.sum()
    A = np.where(cov > 0.9, A / np.maximum(cov, 1e-3), np.nan)
    A_err = np.sqrt(A_var) / np.maximum(cov, 1e-3)

    B, B_err, n_ann, ring_used = annulus_background(data, var, ring, good)
    if smooth:
        B_cont = smooth_continuum(wave, B, B_err, good)
    else:
        B_cont = interpolate_over_lines(wave, B, good)

    n_ap = float(ap_frac.sum())
    C1 = A - n_ap * B_cont
    C1_err = np.sqrt(A_err**2 + (n_ap * B_err)**2)
    return {'A': A, 'A_err': A_err, 'B': B, 'B_err': B_err, 'B_cont': B_cont,
            'C1': C1, 'C1_err': C1_err, 'BKG_LINES_AP': n_ap * (B - B_cont),
            'BKG_LINES_AP_err': n_ap * B_err,
            'N_ap': n_ap, 'N_ann': n_ann, 'ring': ring_used}


# ─────────────────────────────────────────────────────────────────────────────
# PSF model
# ─────────────────────────────────────────────────────────────────────────────

def moffat_alpha(fwhm, beta):
    return fwhm / (2.0 * np.sqrt(2.0**(1.0 / beta) - 1.0))


def moffat_image(shape, xc, yc, fwhm, beta, oversample=5):
    """
    Pixel-integrated Moffat with unit TOTAL flux (analytic normalisation,
    so flux falling outside the stamp is accounted for).
    """
    ny, nx = shape
    off = (np.arange(oversample) + 0.5) / oversample - 0.5
    ys = (np.arange(ny)[:, None] + off[None, :]).ravel()
    xs = (np.arange(nx)[:, None] + off[None, :]).ravel()
    a = moffat_alpha(fwhm, beta)
    r2 = (xs[None, :] - xc)**2 + (ys[:, None] - yc)**2
    img = (beta - 1.0) / (np.pi * a**2) * (1.0 + r2 / a**2)**(-beta)
    img = img.reshape(ny, oversample, nx, oversample).mean(axis=(1, 3))
    return img


def fit_psf_stamp(img, err, xc0, yc0, fwhm0, beta0=2.5, half=10, mask=None):
    """
    Fit amp*Moffat + plane to a continuum image around (xc0, yc0).
    Returns dict(fwhm, beta, amp, xc, yc, snr, ok) in stamp pixel units.
    """
    ny, nx = img.shape
    ix, iy = int(round(xc0)), int(round(yc0))
    y0, y1 = max(iy - half, 0), min(iy + half + 1, ny)
    x0, x1 = max(ix - half, 0), min(ix + half + 1, nx)
    sub, sube = img[y0:y1, x0:x1], err[y0:y1, x0:x1]
    m = np.isfinite(sub) & np.isfinite(sube) & (sube > 0)
    if mask is not None:
        m &= ~mask[y0:y1, x0:x1]
    if m.sum() < 50:
        return {'ok': False}
    yy, xx = np.indices(sub.shape)

    def model(p):
        amp, xc, yc, fwhm, beta, c0, cx, cy = p
        return (amp * moffat_image(sub.shape, xc, yc, fwhm, beta)
                + c0 + cx * (xx - sub.shape[1] / 2) + cy * (yy - sub.shape[0] / 2))

    def resid(p):
        return ((sub - model(p)) / sube)[m]

    bkg0 = np.nanmedian(sub)
    amp0 = max(np.nansum(sub - bkg0), 1e-3)
    p0 = [amp0, xc0 - x0, yc0 - y0, fwhm0, beta0, bkg0, 0.0, 0.0]
    lo = [0, xc0 - x0 - 3, yc0 - y0 - 3, 2.0, 1.2, -np.inf, -np.inf, -np.inf]
    hi = [np.inf, xc0 - x0 + 3, yc0 - y0 + 3, 15.0, 50.0, np.inf, np.inf, np.inf]
    try:
        r = least_squares(resid, p0, bounds=(lo, hi), x_scale='jac')
    except Exception:
        return {'ok': False}
    J = r.jac
    try:
        cov = np.linalg.inv(J.T @ J) * max(np.sum(r.fun**2) / (m.sum() - len(p0)), 1.0)
        amp_err = np.sqrt(cov[0, 0])
        fwhm_err = np.sqrt(cov[3, 3])
    except np.linalg.LinAlgError:
        amp_err, fwhm_err = np.nan, np.nan
    amp, xc, yc, fwhm, beta = r.x[:5]
    return {'ok': bool(r.success), 'amp': amp, 'xc': xc + x0, 'yc': yc + y0,
            'fwhm': fwhm, 'fwhm_err': fwhm_err, 'beta': beta,
            'snr': amp / amp_err if amp_err > 0 else np.nan,
            'chi2nu': np.sum(r.fun**2) / (m.sum() - len(p0))}


def continuum_image(data, var, wave, good, lo, hi):
    """Mean continuum image (per A) and its error between lo and hi A."""
    sel = good & (wave >= lo) & (wave < hi)
    n = np.sum(np.isfinite(data[sel]), axis=0)
    img = np.nanmean(data[sel], axis=0)
    err = np.sqrt(np.nansum(var[sel], axis=0)) / np.maximum(n, 1)
    return img, err


class PSFModel:
    """Moffat PSF: beta constant, FWHM linear in wavelength (pixels)."""

    def __init__(self, fwhm_ref, beta, slope=0.0, wave_ref=7000.0):
        self.fwhm_ref, self.beta, self.slope, self.wave_ref = fwhm_ref, beta, slope, wave_ref

    def fwhm(self, wave):
        return self.fwhm_ref + self.slope * (np.asarray(wave) - self.wave_ref)

    def to_dict(self):
        return {'fwhm_ref_px': self.fwhm_ref, 'beta': self.beta,
                'slope_px_per_A': self.slope, 'wave_ref': self.wave_ref}


def psf_fit_usable(rows, min_snr=10.0, chi2_factor=3.0):
    """
    Fits good enough for the PSF: S/N >= min_snr and chi2nu below
    chi2_factor x the sample median. The averaged continuum images have tiny
    formal errors, so chi2nu is dominated by disc structure (~100s) and only
    a relative cut is meaningful.
    """
    ok = [r for r in rows if r.get('ok') and np.isfinite(r.get('snr', np.nan))]
    if not ok:
        return []
    c_med = np.median([r['chi2nu'] for r in ok])
    return [r for r in ok if r['snr'] >= min_snr and r['chi2nu'] < chi2_factor * c_med]


def build_psf_model(per_bin, wave_ref=7000.0, min_snr=10.0):
    """
    Combine per-source, per-bin Moffat fits (list of dicts with 'wave',
    'fwhm', 'fwhm_err', 'beta', 'snr', 'ok') into one PSFModel: median beta,
    weighted linear fit of FWHM(lambda) over the per-bin medians.
    """
    rows = psf_fit_usable(per_bin, min_snr)
    if len(rows) < 5:
        raise RuntimeError(f'only {len(rows)} usable PSF fits')
    beta = float(np.median([r['beta'] for r in rows]))
    waves = sorted({r['wave'] for r in rows})
    wm, fm, fe = [], [], []
    for w in waves:
        f = np.array([r['fwhm'] for r in rows if r['wave'] == w])
        if f.size < 3:
            continue
        wm.append(w)
        fm.append(np.median(f))
        fe.append(1.253 * np.std(f) / np.sqrt(f.size))
    wm, fm, fe = map(np.asarray, (wm, fm, fe))
    if wm.size == 0:
        raise RuntimeError(f'{len(rows)} usable PSF fits but no wavelength bin with >= 3 of them')
    if wm.size >= 2:
        coef = np.polyfit(wm - wave_ref, fm, 1, w=1 / np.maximum(fe, 1e-3))
        slope, fref = coef
    else:
        slope, fref = 0.0, fm[0]
    return PSFModel(float(fref), beta, float(slope), wave_ref), \
        {'wave': wm, 'fwhm_med': fm, 'fwhm_err': fe, 'n_used': len(rows)}


# ─────────────────────────────────────────────────────────────────────────────
# Method 2 — PSF + extended background, per wavelength slice
# ─────────────────────────────────────────────────────────────────────────────

def background_columns(shape, fwhm_px, poly_order=2, patch_spacing_fwhm=2.0,
                       patch_sigma_fwhm=1.0):
    """
    Extended background basis on the stamp: 2-D polynomial + a regular grid
    of Gaussian patches (sigma = patch_sigma_fwhm * FWHM, spacing
    patch_spacing_fwhm * FWHM), like pampelmuse's BackgroundGrid.
    Returns (columns (npix, ncol), is_patch (ncol,) bool).
    """
    ny, nx = shape
    yy, xx = np.indices(shape, dtype=float)
    u = 2 * xx / (nx - 1) - 1
    v = 2 * yy / (ny - 1) - 1
    cols = []
    for i in range(poly_order + 1):
        for j in range(poly_order + 1 - i):
            cols.append((u**i * v**j).ravel())
    n_poly = len(cols)

    spacing = patch_spacing_fwhm * fwhm_px
    sig = patch_sigma_fwhm * fwhm_px
    if spacing > 0:
        npx = max(int(round((nx - 1) / spacing)) + 1, 2)
        npy = max(int(round((ny - 1) / spacing)) + 1, 2)
        for cy in np.linspace(0, ny - 1, npy):
            for cx in np.linspace(0, nx - 1, npx):
                g = np.exp(-0.5 * ((xx - cx)**2 + (yy - cy)**2) / sig**2)
                cols.append(g.ravel())
    cols = np.array(cols).T
    is_patch = np.zeros(cols.shape[1], bool)
    is_patch[n_poly:] = True
    return cols, is_patch


def group_neighbours(xy_target, xy_neigh, fwhm_px, blend_frac=0.5):
    """
    Split neighbours into those blended with the target (closer than
    blend_frac*FWHM, folded into the target column — unresolvable) and
    separate ones.
    """
    blended, separate = [], []
    for p in xy_neigh:
        d = np.hypot(p[0] - xy_target[0], p[1] - xy_target[1])
        (blended if d < blend_frac * fwhm_px else separate).append(p)
    return blended, separate


def design_matrix(shape, xy_target, xy_neigh, fwhm_px, beta, bkg_kw=None):
    """
    Columns: [target PSF, neighbour PSFs..., background...].
    Returns A (npix, ncol), dict of column index groups.
    """
    bkg_kw = bkg_kw or {}
    cols = [moffat_image(shape, xy_target[0], xy_target[1], fwhm_px, beta).ravel()]
    for (x, y) in xy_neigh:
        cols.append(moffat_image(shape, x, y, fwhm_px, beta).ravel())
    bcols, is_patch = background_columns(shape, fwhm_px, **bkg_kw)
    A = np.column_stack(cols + [bcols])
    n_n = len(xy_neigh)
    idx = {'target': 0,
           'neigh': np.arange(1, 1 + n_n),
           'bkg': np.arange(1 + n_n, A.shape[1]),
           'patch': np.arange(1 + n_n, A.shape[1])[is_patch]}
    return A, idx


def solve_slices(A, d, w, ridge_cols, ridge_frac):
    """
    Weighted ridge least squares for many slices sharing one design matrix.

    A : (npix, ncol); d, w : (nl, npix) data and weights (1/var, 0 = masked).
    Returns p (nl, ncol), cov (nl, ncol, ncol).
    """
    AtWA = np.einsum('pi,lp,pj->lij', A, w, A, optimize=True)
    AtWd = np.einsum('pi,lp->li', A, w * d, optimize=True)
    if ridge_cols is not None and len(ridge_cols) and ridge_frac > 0:
        diag = AtWA[:, ridge_cols, ridge_cols]
        AtWA[:, ridge_cols, ridge_cols] += ridge_frac * diag
    # tiny Tikhonov on everything for numerical safety on empty slices
    tr = np.trace(AtWA, axis1=1, axis2=2) / A.shape[1]
    AtWA += (1e-10 * np.maximum(tr, 1e-30))[:, None, None] * np.eye(A.shape[1])[None]
    cov = np.linalg.inv(AtWA)
    p = np.einsum('lij,lj->li', cov, AtWd)
    return p, cov


def refine_position(img, err, xy_target, xy_neigh, fwhm_px, beta, max_shift=2.0,
                    bkg_kw=None, ridge_frac=1e-2):
    """
    Common (dx, dy) shift of the catalogue positions (HST astrometry) that
    best fits the continuum image with the full linear model.
    """
    d = np.nan_to_num(img).ravel()
    w = np.where(np.isfinite(img) & np.isfinite(err) & (err > 0), 1 / err**2, 0).ravel()

    def chi2(s):
        if np.hypot(*s) > max_shift:
            return 1e30
        xt = (xy_target[0] + s[0], xy_target[1] + s[1])
        xn = [(x + s[0], y + s[1]) for (x, y) in xy_neigh]
        A, idx = design_matrix(img.shape, xt, xn, fwhm_px, beta, bkg_kw)
        p, _ = solve_slices(A, d[None], w[None], idx['patch'], ridge_frac)
        return float(np.sum(w * (d - A @ p[0])**2))

    best = minimize(chi2, [0.0, 0.0], method='Nelder-Mead',
                    options={'xatol': 0.02, 'fatol': 1e-3, 'maxiter': 200})
    return best.x, best.fun


def method_psf(data, var, wave, ap_frac, xy_target, xy_neigh, psf, z,
               fit_half=20, block_A=100.0, ridge_frac=1e-2, clip=5.0,
               bkg_kw=None, refine=True):
    """
    Method 2. Per-slice linear fit of PSF(target) + PSF(neighbours) +
    extended background on a (2*fit_half+1)^2 stamp centred on the target.

    Returns dict with C2 (total cluster flux density), C2_err, EE_AP (PSF
    fraction inside the aperture), C2_AP (= C2*EE_AP), E_AP (+err:
    background model inside the aperture), NEIGH_AP, RES_AP (residual in
    aperture), EXTRA2 (= A - C2_AP) and its error, CHI2NU per slice,
    shift (dx, dy), and the stamp-level model for QA.
    """
    ny, nx = data.shape[1:]
    ix, iy = int(round(xy_target[0])), int(round(xy_target[1]))
    y0, y1 = max(iy - fit_half, 0), min(iy + fit_half + 1, ny)
    x0, x1 = max(ix - fit_half, 0), min(ix + fit_half + 1, nx)
    D = data[:, y0:y1, x0:x1]
    V = var[:, y0:y1, x0:x1]
    AP = ap_frac[y0:y1, x0:x1]
    shape = D.shape[1:]
    xt = (xy_target[0] - x0, xy_target[1] - y0)
    xn_all = [(x - x0, y - y0) for (x, y) in xy_neigh
              if -2 <= x - x0 <= shape[1] + 1 and -2 <= y - y0 <= shape[0] + 1]

    good = continuum_mask(wave, z)
    fwhm_mid = float(psf.fwhm(np.median(wave)))
    blended, xn = group_neighbours(xt, xn_all, fwhm_mid)

    shift = np.zeros(2)
    if refine:
        img, err = continuum_image(D, V, wave, good, wave[0], wave[-1])
        shift, _ = refine_position(img, err, xt, xn, fwhm_mid, psf.beta, bkg_kw=bkg_kw,
                                   ridge_frac=ridge_frac)
    xt = (xt[0] + shift[0], xt[1] + shift[1])
    xn = [(x + shift[0], y + shift[1]) for (x, y) in xn]

    nl = D.shape[0]
    npix = shape[0] * shape[1]
    d = np.nan_to_num(D.reshape(nl, npix)).astype(float)
    okp = np.isfinite(D.reshape(nl, npix)) & np.isfinite(V.reshape(nl, npix)) \
        & (V.reshape(nl, npix) > 0)
    w = np.where(okp, 1.0 / np.where(okp, V.reshape(nl, npix), 1.0), 0.0)

    yy, xx = np.indices(shape)
    protect = (np.hypot(xx - xt[0], yy - xt[1]) <= 1.5 * fwhm_mid).ravel()

    out = {k: np.full(nl, np.nan) for k in
           ['C2', 'C2_err', 'EE_AP', 'E_AP', 'E_AP_err', 'NEIGH_AP', 'RES_AP',
            'CHI2NU', 'A_STAMP']}
    model_stamp = np.full((nl, npix), np.nan, np.float32)
    psf_stamp = np.full((nl, npix), np.nan, np.float32)
    apv = AP.ravel()

    # blocks in wavelength sharing one design matrix (PSF changes slowly)
    edges = np.arange(wave[0], wave[-1] + block_A, block_A)
    for b0, b1 in zip(edges[:-1], edges[1:]):
        sl = np.where((wave >= b0) & (wave < b1))[0]
        if sl.size == 0:
            continue
        fw = float(psf.fwhm(np.mean(wave[sl])))
        A, idx = design_matrix(shape, xt, xn, fw, psf.beta, bkg_kw)
        ws = w[sl].copy()
        p, cov = solve_slices(A, d[sl], ws, idx['patch'], ridge_frac)
        if clip:
            r = (d[sl] - p @ A.T) * np.sqrt(ws)
            bad = (np.abs(r) > clip) & ~protect[None, :]
            if bad.any():
                ws[bad] = 0.0
                p, cov = solve_slices(A, d[sl], ws, idx['patch'], ridge_frac)

        model = p @ A.T
        npix_used = np.sum(ws > 0, axis=1)
        out['CHI2NU'][sl] = np.sum(ws * (d[sl] - model)**2, axis=1) / \
            np.maximum(npix_used - A.shape[1], 1)
        out['C2'][sl] = p[:, idx['target']]
        out['C2_err'][sl] = np.sqrt(cov[:, idx['target'], idx['target']])
        ee = float(A[:, idx['target']] @ apv)
        out['EE_AP'][sl] = ee

        g_bkg = np.zeros(A.shape[1])
        g_bkg[idx['bkg']] = A[:, idx['bkg']].T @ apv
        out['E_AP'][sl] = p @ g_bkg
        out['E_AP_err'][sl] = np.sqrt(np.einsum('i,lij,j->l', g_bkg, cov, g_bkg))
        g_n = np.zeros(A.shape[1])
        g_n[idx['neigh']] = A[:, idx['neigh']].T @ apv
        out['NEIGH_AP'][sl] = p @ g_n
        out['RES_AP'][sl] = np.sum(np.where(okp[sl], d[sl] - model, 0.0) * apv, axis=1)
        out['A_STAMP'][sl] = np.sum(np.where(okp[sl], d[sl], 0.0) * apv, axis=1)
        model_stamp[sl] = model
        psf_stamp[sl] = p[:, [idx['target']]] * A[:, idx['target']][None, :]

    # unmeasurable slices (all NaN in the stamp)
    empty = okp.sum(axis=1) < A.shape[1] + 10
    for k in out:
        out[k][empty] = np.nan

    out['C2_AP'] = out['C2'] * out['EE_AP']
    out['C2_AP_err'] = out['C2_err'] * out['EE_AP']
    out['shift'] = shift
    out['n_neigh'] = len(xn)
    out['n_blended'] = len(blended)
    out['stamp_slice'] = (slice(y0, y1), slice(x0, x1))
    out['model_stamp'] = model_stamp.reshape(nl, *shape)
    out['psf_stamp'] = psf_stamp.reshape(nl, *shape)
    out['xy_fit'] = (xt[0] + x0, xt[1] + y0)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Emission-line fluxes
# ─────────────────────────────────────────────────────────────────────────────

def fit_line_group(wave, flux, err, z, names, vmax=300.0):
    """
    Gaussians with common velocity offset and intrinsic sigma (convolved with
    the MUSE LSF) on a linear continuum. Fluxes are free (can be negative,
    so noise is unbiased) and are integrated fluxes in 1e-20 erg/s/cm2.

    Returns dict name -> (flux, err) plus 'dv', 'sig' for the group.
    """
    lam0 = np.array([EMISSION_LINES[n] for n in names])
    lo, hi = lam0.min() * (1 + z) - 30, lam0.max() * (1 + z) + 30
    sel = (wave > lo) & (wave < hi) & np.isfinite(flux) & np.isfinite(err) & (err > 0)
    res = {n: (np.nan, np.nan) for n in names}
    if sel.sum() < 10:
        return res
    x, y, e = wave[sel], flux[sel], err[sel]
    xm = x.mean()
    lsf_sig = muse_lsf_fwhm(lam0 * (1 + z)) / 2.3548

    def model(xx, c0, c1, dv, sv, *F):
        m = c0 + c1 * (xx - xm)
        for i in range(len(lam0)):
            mu = lam0[i] * (1 + z) * (1 + dv / C_KMS)
            s = np.sqrt((lam0[i] * (1 + z) * sv / C_KMS)**2 + lsf_sig[i]**2)
            m = m + F[i] / (np.sqrt(2 * np.pi) * s) * np.exp(-0.5 * ((xx - mu) / s)**2)
        return m

    side = np.ones(x.size, bool)
    for l in lam0:
        side &= np.abs(x - l * (1 + z)) > 8
    c0 = np.median(y[side]) if side.sum() > 3 else np.median(y)
    F0 = []
    for l in lam0:
        pk = np.abs(x - l * (1 + z)) < 3
        F0.append(max((y[pk].max() - c0) * 2.5 * 2.0, 1.0) if pk.any() else 1.0)
    p0 = [c0, 0.0, 0.0, 40.0] + F0
    lob = [-np.inf, -np.inf, -vmax, 1.0] + [-np.inf] * len(lam0)
    hib = [np.inf, np.inf, vmax, 300.0] + [np.inf] * len(lam0)
    try:
        p, pc = curve_fit(model, x, y, p0=p0, sigma=e, absolute_sigma=True,
                          bounds=(lob, hib), maxfev=20000)
    except Exception:
        return res
    pe = np.sqrt(np.clip(np.diag(pc), 0, None))
    for i, n in enumerate(names):
        res[n] = (p[4 + i], pe[4 + i])
    res['dv'], res['sig'] = p[2], p[3]
    return res


def fit_lines(wave, flux, err, z):
    """All LINE_GROUPS -> dict line -> (flux, err)."""
    out = {}
    for g, names in LINE_GROUPS.items():
        r = fit_line_group(wave, flux, err, z, names)
        for n in names:
            out[n] = r[n]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic photometry (comparison with the imaging only)
# ─────────────────────────────────────────────────────────────────────────────

def read_filter(path):
    t = np.loadtxt(path, comments='#')
    return t[:, 0], t[:, 1]


def synthetic_fnu_mJy(wave, flam, flam_err, fw, ft, min_cover=0.5):
    """
    Photon-counting synthetic flux density in mJy from F_lambda in
    1e-20 erg/s/cm2/A, averaged over the covered part of the filter.
    Returns (fnu, err, fraction of the filter's response-weighted
    throughput covered by the spectrum); NaN below ``min_cover``.
    """
    T = np.interp(wave, fw, ft, left=0, right=0)
    ok = np.isfinite(flam) & np.isfinite(flam_err)
    cover_tot = np.trapezoid(ft * fw, fw)
    cover = np.trapezoid(T * wave, wave) / cover_tot
    num = np.trapezoid(np.where(ok, flam, 0) * T * wave, wave)
    num_var = np.sum((np.where(ok, flam_err, 0) * T * wave * np.gradient(wave))**2)
    den = np.trapezoid(T * C_KMS * 1e13 / wave, wave)       # c in A/s
    fnu = num / den * 1e-20                                 # erg/s/cm2/Hz
    fnu_err = np.sqrt(num_var) / den * 1e-20
    if not cover >= min_cover:
        return np.nan, np.nan, cover
    return fnu / 1e-26, fnu_err / 1e-26, cover


def write_region_cutout_wcs(cel_wcs, x0, y0):
    """Celestial WCS of a cutout starting at (x0, y0)."""
    w = cel_wcs.deepcopy()
    w.wcs.crpix = w.wcs.crpix - np.array([x0, y0])
    return w
