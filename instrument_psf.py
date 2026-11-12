"""
Instrument PSF models for native photometry.

A real compact source is seen as

    source light profile  (x)  telescope PSF  (x)  extra instrumental blur

and the telescope PSF is NOT a Gaussian (HST: diffraction core + rings +
spider spikes, heavier wings). Fitting such a profile with a Gaussian and
taking the Gaussian closed-form R65 under-sizes the aperture (the true 65%
radius is larger), so flux/0.65 comes out low (~0.86 for an HST point
source). InstrumentPSF instead models the source as

    Gaussian(intrinsic FWHM)  (x)  model PSF  (x)  Gaussian(extra FWHM)

with only the intrinsic FWHM (plus amplitude/background) free, and gives
the R65 of that model rendered exactly like the data (same pixel grid, same
sub-pixel center).

PSF models:
  HST  — WFC3/UVIS and ACS/WFC 4x-oversampled effective PSFs (pickles in
         data/psf/), plus a per-band drizzle/jitter Gaussian.
  JWST — stpsf 2.2.0 NIRCam/MIRI PSFs (data/psf/jwst_psf/, made with
         the PhangsTeam/jwst_kernels settings), plus a per-band extra
         Gaussian for NIRCam (none for MIRI).
"""

import os
import pickle
from functools import lru_cache

import numpy as np
from scipy.ndimage import map_coordinates
from astropy.convolution import Gaussian2DKernel, convolve_fft

FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))

_HERE = os.path.dirname(os.path.abspath(__file__))
HST_PSF_FILES = {
    'wfc3': os.path.join(_HERE, 'data', 'psf', 'hst_wfc3_uv_psf_dict.pickle'),
    'acs': os.path.join(_HERE, 'data', 'psf', 'hst_acs_wfc_psf_dict.pickle'),
}

# Extra Gaussian blur (FWHM, arcsec) applied on top of the model HST PSF so
# the rendered point source matches real PHANGS-HST stars: the value that
# makes the rendered PSF's Moffat FWHM (psf_aux_func.fit_2d_moffat, same
# method that filled psf_table.csv) equal the median measured FWHM across
# the galaxies in psf_table.csv. Accounts for drizzling/dither combination,
# exposure registration and jitter, none of which are in a single-exposure
# model PSF. ACS F435W is based on only 2 galaxies.
HST_DRIZZLE_FWHM_ARCSEC = {
    ('wfc3', 'F275W'): 0.062,
    ('wfc3', 'F336W'): 0.056,
    ('wfc3', 'F438W'): 0.058,
    ('wfc3', 'F555W'): 0.076,
    ('wfc3', 'F814W'): 0.068,
    ('acs', 'F435W'): 0.063,
    ('acs', 'F814W'): 0.060,
}


# Fraction of a star's TOTAL light inside a circle of radius r: the HST
# model PSF stamps stop at ~0.5" (WFC3 0.49", ACS 0.62"), but the real PSF
# keeps ~7-12% of the light further out. That fraction becomes the PSF's
# total_flux (as for the stpsf JWST PSFs), so "total" means total and R65 /
# the aperture correction include the far wings.
# WFC3/UVIS: STScI synthetic EE tables (wfc3uvis{1,2}_aper_007_syn.csv,
# total = 6"), mean of UVIS1 and UVIS2. ACS/WFC: Bohlin (2016) hot-star
# table (bohlin2016_wfc_ee, total = 5.5"), radii 9, 10, 20 px of 0.05".
_HST_EE_RADII = {
    'wfc3': (0.36, 0.40, 0.44, 0.48, 0.51, 0.55, 0.59, 0.63, 0.67),
    'acs': (0.45, 0.50, 1.00),
}
_HST_EE = {
    'wfc3': {
        'F275W': (0.8580, 0.8654, 0.8718, 0.8778, 0.8822, 0.8881, 0.8940, 0.8999, 0.9055),
        'F336W': (0.8860, 0.8921, 0.8972, 0.9016, 0.9054, 0.9088, 0.9121, 0.9156, 0.9194),
        'F438W': (0.9008, 0.9074, 0.9128, 0.9172, 0.9209, 0.9244, 0.9273, 0.9298, 0.9321),
        'F555W': (0.9044, 0.9118, 0.9178, 0.9228, 0.9269, 0.9305, 0.9336, 0.9363, 0.9387),
        'F606W': (0.9033, 0.9109, 0.9174, 0.9228, 0.9273, 0.9311, 0.9344, 0.9374, 0.9399),
        'F814W': (0.8928, 0.9021, 0.9087, 0.9142, 0.9158, 0.9212, 0.9254, 0.9289, 0.9322),
    },
    'acs': {
        'F435W': (0.902, 0.907, 0.941),
        'F555W': (0.910, 0.915, 0.946),
        'F606W': (0.910, 0.916, 0.947),
        'F814W': (0.908, 0.914, 0.949),
    },
}


def hst_stamp_ee(instrument, band, radius_arcsec):
    """Fraction of the total light of an HST point source inside
    radius_arcsec (from the tables above); None if the band is not tabulated."""
    table = _HST_EE.get(instrument.lower(), {}).get(band.upper())
    if table is None:
        return None
    return float(np.interp(radius_arcsec, _HST_EE_RADII[instrument.lower()], table))


# JWST PSFs: stpsf 2.2.0 models (OVERDIST extension = optics + detector
# effects), generated with the settings of PhangsTeam/jwst_kernels
# make_psf.py (oversample 4, odd parity, FOV 10" NIRCam / 19.98" MIRI).
JWST_PSF_DIR = os.path.join(_HERE, 'data', 'psf', 'jwst_psf')

# PHANGS-JWST v4p1 mosaic pixel scales (arcsec/pix).
JWST_PIXEL_SCALE_ARCSEC = {
    'F200W': 0.0307,
    'F300M': 0.0630, 'F335M': 0.0630, 'F360M': 0.0630,
    'F770W': 0.1109, 'F1000W': 0.1109, 'F1130W': 0.1109, 'F2100W': 0.1109,
}

# Extra Gaussian blur (FWHM, arcsec) on top of the stpsf model, same role as
# HST_DRIZZLE_FWHM_ARCSEC: the value that makes the rendered PSF's Moffat
# FWHM (psf_aux_func.fit_2d_moffat on a 60x60 cutout) equal the median
# psf_table.csv FWHM (18 galaxies). F200W is calibrated with alpha fixed at
# 2.8 (the free-alpha fit of the model runs into alpha's 1.5 lower bound;
# real stars give alpha ~3.2). F335M's value is the largest and may be
# inflated by 3.3um PAH emission around the "stars" used in psf_table.
# MIRI: no extra blur — the psf_table MIRI FWHMs are not physically
# consistent (F770W broader than F1000W, F2100W sharper than the optics
# alone), so they cannot calibrate anything; pass extra_fwhm_arcsec to
# InstrumentPSF.from_jwst to override.
JWST_EXTRA_FWHM_ARCSEC = {
    'F200W': 0.040,
    'F300M': 0.067,
    'F335M': 0.097,
    'F360M': 0.085,
    'F770W': 0.0, 'F1000W': 0.0, 'F1130W': 0.0, 'F2100W': 0.0,
}


def load_jwst_psf(band, psf_dir=JWST_PSF_DIR):
    """stpsf model PSF for one PHANGS-JWST band: (camera, oversampled 2-D
    array, its pixel scale in arcsec, array sum = fraction of the total flux
    inside the PSF field of view)."""
    from astropy.io import fits
    band = band.upper()
    for camera in ('NIRCam', 'MIRI'):
        path = os.path.join(psf_dir, f'{camera}_PSF_filter_{band}.fits')
        if os.path.exists(path):
            with fits.open(path) as hdul:
                psf = np.asarray(hdul[0].data, dtype=np.float64)
                scale = float(hdul[0].header['PIXELSCL'])
            psf = np.where(np.isfinite(psf), psf, 0.0)
            return camera, psf, scale, float(psf.sum())
    raise FileNotFoundError(f'No stpsf PSF for {band} in {psf_dir}')


def load_hst_psf(instrument, band, psf_files=HST_PSF_FILES):
    """Model HST PSF for one band: (oversampled 2-D array, its pixel scale in
    arcsec). Pixels beyond the stamp's inscribed circle are zeroed: the WFC3
    arrays carry a constant non-PSF border (rows 0/100, col 100) holding
    ~19% of the array sum, which would otherwise be rendered as flux."""
    with open(psf_files[instrument], 'rb') as fh:
        entry = pickle.load(fh)[band]
    psf = np.asarray(entry['over_sampled_psf'], dtype=np.float64).copy()
    ny, nx = psf.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
    psf[np.hypot(xx - cx, yy - cy) > min(cx, cy) - 2] = 0.0
    psf[~np.isfinite(psf)] = 0.0
    return psf, float(entry['pixel_scale_psf_over_sampled'])


def _resample_psf(psf, psf_scale_arcsec, fine_scale_arcsec):
    """Resample a centered PSF onto an odd-sized grid of pitch
    fine_scale_arcsec, peak kept on the central pixel, unit sum."""
    c_in = (psf.shape[0] - 1) / 2.0
    half = int(np.floor(c_in * psf_scale_arcsec / fine_scale_arcsec))
    off = np.arange(-half, half + 1) * fine_scale_arcsec / psf_scale_arcsec
    yy, xx = np.meshgrid(c_in + off, c_in + off, indexing='ij')
    out = map_coordinates(psf, [yy, xx], order=3, prefilter=True,
                          mode='constant', cval=0.0)
    out = np.clip(out, 0.0, None)
    return out / out.sum()


def _gaussian_fine_kernel(fwhm_arcsec, fine_scale_arcsec):
    sigma = fwhm_arcsec * FWHM_TO_SIGMA / fine_scale_arcsec
    kernel = Gaussian2DKernel(sigma, mode='oversample')
    kernel.normalize()
    return kernel


def _convolve_gaussian(kernel, fwhm_arcsec, fine_scale_arcsec):
    if not fwhm_arcsec or fwhm_arcsec <= 0:
        return kernel
    return convolve_fft(kernel, _gaussian_fine_kernel(fwhm_arcsec, fine_scale_arcsec),
                        normalize_kernel=True, boundary='fill', fill_value=0.0)


def _place_sources_bilinear(image, xs, ys, fluxes):
    """Splat each source's flux onto its 4 nearest pixels by bilinear
    weight, in place. For an exact-integer (x, y) this degenerates to a
    single-pixel delta."""
    npix_y, npix_x = image.shape
    for x, y, flux in zip(xs, ys, fluxes):
        x0 = int(np.floor(x))
        y0 = int(np.floor(y))
        fx = x - x0
        fy = y - y0
        for dx, wx in ((0, 1 - fx), (1, fx)):
            for dy, wy in ((0, 1 - fy), (1, fy)):
                xi, yi = x0 + dx, y0 + dy
                if 0 <= xi < npix_x and 0 <= yi < npix_y and wx * wy > 0:
                    image[yi, xi] += flux * wx * wy


def render_sources(xs, ys, fluxes, shape, pixel_scale_arcsec, kernel_fine, n_fine):
    """Image (shape = (ny, nx), output pixels) of point sources at output-pixel
    coordinates xs, ys, convolved with `kernel_fine` (unit-sum kernel sampled
    on the grid n_fine times finer than the output pixels), then summed into
    output pixels."""
    ny, nx = shape
    fine = np.zeros((ny * n_fine, nx * n_fine), dtype=np.float64)
    shift = (n_fine - 1) / 2.0
    _place_sources_bilinear(fine, np.asarray(xs, float) * n_fine + shift,
                            np.asarray(ys, float) * n_fine + shift, fluxes)
    fine = convolve_fft(fine, kernel_fine, normalize_kernel=False, boundary='fill',
                        fill_value=0.0, allow_huge=True)
    return fine.reshape(ny, n_fine, nx, n_fine).sum(axis=(1, 3))


def render_with_hst_psf(xs, ys, fluxes, npix, pixel_scale_arcsec, psf, psf_scale_arcsec,
                        drizzle_fwhm_arcsec=0.0, intrinsic_fwhm_arcsec=0.0, n_fine=4):
    """Noiseless npix x npix image of point sources (output-pixel coordinates
    xs, ys) seen through a model HST PSF: sources (x) Gaussian(intrinsic) (x)
    PSF (x) Gaussian(drizzle), built on a grid n_fine times finer than the
    output pixels and summed into output pixels. Total flux equals
    sum(fluxes) (minus anything landing off the image)."""
    fine_scale = pixel_scale_arcsec / n_fine
    kernel = _resample_psf(psf, psf_scale_arcsec, fine_scale)
    for fwhm in (intrinsic_fwhm_arcsec, drizzle_fwhm_arcsec):
        kernel = _convolve_gaussian(kernel, fwhm, fine_scale)
    kernel = np.clip(kernel, 0.0, None)
    kernel /= kernel.sum()
    return render_sources(xs, ys, fluxes, (npix, npix), pixel_scale_arcsec, kernel, n_fine)


# ─────────────────────────────────────────────────────────────────────────────
# Intrinsic (source) light profiles, used both to build the smooth-cluster
# mocks and as the source model of the instrument fit:
#   'gaussian' : Sigma ∝ exp(-r²/2σ²)                       (σ = R_eff/√(2 ln2))
#   'eff'      : Sigma ∝ (1 + r²/a²)^(-γ/2), cut at r_max  (Elson+87; γ=4 Plummer)
#   'king'     : Sigma ∝ [1/√(1+(r/rc)²) - 1/√(1+(rt/rc)²)]², r < rt   (King 1962),
#                concentration c = log10(rt/rc)
# Every profile is parameterized by its half-light radius R_eff (within its
# truncation); the scale radius (σ, a, r_c) follows from it.
# ─────────────────────────────────────────────────────────────────────────────

def _eff_cum(r, a, gamma):
    """Unnormalized projected light inside radius r of an EFF profile (any
    gamma: the gamma = 2 limit is logarithmic; the sign is the same for all
    radii, so ratios are fine)."""
    if abs(gamma - 2.0) < 1e-9:
        return np.log1p((r / a) ** 2)
    return 1.0 - (1.0 + (r / a) ** 2) ** (1.0 - gamma / 2.0)


def eff_scale_for_reff(r_eff, gamma, r_max):
    """EFF scale radius a such that half of the light inside r_max is inside r_eff."""
    from scipy.optimize import brentq
    return brentq(lambda a: _eff_cum(r_eff, a, gamma) / _eff_cum(r_max, a, gamma) - 0.5,
                  1e-4 * r_eff, 100 * r_eff)


def smooth_profile_sb(r, profile, scale, gamma=3.0, c=1.5, r_max=np.inf):
    """Unnormalized surface brightness at radius r (same units as scale)."""
    r = np.asarray(r, dtype=float)
    if profile == 'gaussian':
        sb = np.exp(-0.5 * (r / scale) ** 2)
    elif profile == 'eff':
        sb = (1.0 + (r / scale) ** 2) ** (-gamma / 2.0)
    elif profile == 'king':
        rt = scale * 10.0 ** c
        sb = np.clip(1.0 / np.sqrt(1.0 + (r / scale) ** 2)
                     - 1.0 / np.sqrt(1.0 + (rt / scale) ** 2), 0.0, None) ** 2
        sb = np.where(r < rt, sb, 0.0)
    else:
        raise ValueError(f'unknown profile {profile!r}')
    return np.where(r <= r_max, sb, 0.0)


def smooth_profile_truncation(profile, scale, r_eff, gamma=3.0, c=1.5,
                              eff_r_max_over_reff=10.0):
    """Outer radius beyond which the profile is zero (or negligible)."""
    if profile == 'gaussian':
        return 6.0 * scale                       # encloses 1 - 1.5e-8 of the light
    if profile == 'eff':
        return eff_r_max_over_reff * r_eff
    return scale * 10.0 ** c                     # King tidal radius


def _half_light_radius(profile, scale, gamma, c, r_max, n=20000):
    r = np.linspace(0.0, r_max, n)
    cum = np.cumsum(2 * np.pi * r * smooth_profile_sb(r, profile, scale, gamma, c, r_max))
    return float(np.interp(0.5 * cum[-1], cum, r))


def smooth_profile_scale_for_reff(profile, r_eff, gamma=3.0, c=1.5, eff_r_max_over_reff=10.0):
    """Scale radius (σ, a or r_c) giving half-light radius r_eff."""
    from scipy.optimize import brentq
    if profile == 'gaussian':
        return r_eff / np.sqrt(2.0 * np.log(2.0))
    if profile == 'eff':
        return eff_scale_for_reff(r_eff, gamma, eff_r_max_over_reff * r_eff)

    def f(s):
        rmax = smooth_profile_truncation(profile, s, r_eff, gamma, c, eff_r_max_over_reff)
        return _half_light_radius(profile, s, gamma, c, rmax) - r_eff
    return brentq(f, r_eff / 1000.0, 10.0 * r_eff, xtol=1e-7 * r_eff)


def render_smooth_profile(npix, pixel_scale_arcsec, profile, scale, r_max, gamma=3.0, c=1.5,
                          total_flux=1.0, supersample=5, center_offset_arcsec=(0.0, 0.0)):
    """Image (npix x npix) of a smooth profile centered on the center pixel
    shifted by center_offset_arcsec = (dx, dy) (any real value: the profile is
    evaluated analytically around the shifted center), each pixel = mean of
    supersample² sub-pixel samples, normalized so the image sums to
    total_flux (all light lies inside r_max, which must fit)."""
    xc = (npix - 1) / 2.0
    dx, dy = center_offset_arcsec
    if (r_max + np.hypot(dx, dy)) / pixel_scale_arcsec > xc - 1:
        raise ValueError('profile truncation radius does not fit in the image')
    sub = ((np.arange(npix * supersample) + 0.5) / supersample - 0.5 - xc) * pixel_scale_arcsec
    xx, yy = np.meshgrid(sub - dx, sub - dy)
    sb = smooth_profile_sb(np.hypot(xx, yy), profile, scale, gamma, c, r_max)
    img = sb.reshape(npix, supersample, npix, supersample).mean(axis=(1, 3))
    return img * (total_flux / img.sum())


@lru_cache(maxsize=None)
def _profile_shape(profile, gamma, c, eff_r_max_over_reff):
    """Scale-free shape of a profile with R_eff = 1: (scale radius, truncation
    radius, radius grid, normalized enclosed-light curve on that grid)."""
    s = smooth_profile_scale_for_reff(profile, 1.0, gamma, c, eff_r_max_over_reff)
    rm = smooth_profile_truncation(profile, s, 1.0, gamma, c, eff_r_max_over_reff)
    r = np.linspace(0.0, rm, 40001)
    cum = np.concatenate([[0.0], np.cumsum(
        np.diff(r) * np.pi * (r[1:] * smooth_profile_sb(r[1:], profile, s, gamma, c, rm)
                              + r[:-1] * smooth_profile_sb(r[:-1], profile, s, gamma, c, rm)))])
    return s, rm, r, cum / cum[-1]


def profile_enclosed_fraction(radius, r_eff, profile, gamma=3.0, c=1.5, eff_r_max_over_reff=10.0):
    """Fraction of a profile's total light inside `radius` (same units as r_eff)."""
    _, _, r, cum = _profile_shape(profile, float(gamma), float(c), float(eff_r_max_over_reff))
    return float(np.interp(radius / r_eff, r, cum))


def render_profile_kernel(r_eff, profile, fine_scale_arcsec, half_size, gamma=3.0, c=1.5,
                          eff_r_max_over_reff=10.0, supersample=3):
    """
    Source profile of half-light radius r_eff (arcsec) on a (2*half_size+1)²
    grid of pitch fine_scale_arcsec, centered. Light beyond the inscribed
    circle of the grid is left out and NOT renormalized: the array sums to
    the fraction of the TOTAL light inside that circle. A source much smaller
    than one grid pixel is returned as a single central pixel.
    """
    n = 2 * half_size + 1
    out = np.zeros((n, n))
    if r_eff < 0.25 * fine_scale_arcsec:
        out[half_size, half_size] = 1.0
        return out
    s, rm, _, _ = _profile_shape(profile, float(gamma), float(c), float(eff_r_max_over_reff))
    scale, r_max = s * r_eff, rm * r_eff
    r_cut = min(r_max, half_size * fine_scale_arcsec)
    # pixel-center sampling everywhere ...
    ax = (np.arange(n) - half_size) * fine_scale_arcsec
    img = smooth_profile_sb(np.hypot(*np.meshgrid(ax, ax)), profile, scale, gamma, c, r_cut)
    # ... and supersampled pixel averages in the core, where the profile curves fastest
    hc = min(half_size, int(np.ceil(4 * scale / fine_scale_arcsec)) + 3)
    nc = 2 * hc + 1
    sub = ((np.arange(nc * supersample) + 0.5) / supersample - 0.5 - hc) * fine_scale_arcsec
    sb = smooth_profile_sb(np.hypot(*np.meshgrid(sub, sub)), profile, scale, gamma, c, r_cut)
    img[half_size - hc:half_size + hc + 1, half_size - hc:half_size + hc + 1] = \
        sb.reshape(nc, supersample, nc, supersample).mean(axis=(1, 3))
    return img * (profile_enclosed_fraction(r_cut, r_eff, profile, gamma, c,
                                            eff_r_max_over_reff) / img.sum())


class InstrumentPSF:
    """
    Instrument PSF on a given image pixel scale, able to render

        source profile (x) PSF (x) Gaussian(extra_fwhm)

    at any sub-pixel position, exactly the way the mocks are rendered. The
    source profile is a Gaussian of FWHM intrinsic_fwhm_arcsec (default), or
    any of 'gaussian' / 'eff' / 'king' with half-light radius r_eff_arcsec
    (pass profile=..., r_eff_arcsec=..., gamma=..., c=...).

    Build with InstrumentPSF.from_hst('wfc3', 'F555W', pixel_scale_arcsec=0.04)
    or InstrumentPSF.from_jwst('F300M', pixel_scale_arcsec=0.063).

    total_flux : sum of the PSF over its stamp, as a fraction of the source's
        TOTAL flux. 1 for the HST pickles (their normalization is arbitrary,
        so the stamp defines "total"); for stpsf PSFs it is the array sum
        (0.96-0.99), i.e. the light beyond the PSF field of view is still
        counted in the total flux and in the encircled-energy normalization.
    """

    def __init__(self, psf, psf_scale_arcsec, pixel_scale_arcsec,
                 extra_fwhm_arcsec=0.0, n_fine=4, total_flux=1.0, name=''):
        self.pixel_scale_arcsec = float(pixel_scale_arcsec)
        self.extra_fwhm_arcsec = float(extra_fwhm_arcsec)
        self.n_fine = int(n_fine)
        self.fine_scale_arcsec = self.pixel_scale_arcsec / self.n_fine
        self.total_flux = float(total_flux)
        self.name = name
        base = _resample_psf(psf, psf_scale_arcsec, self.fine_scale_arcsec)
        base = _convolve_gaussian(base, self.extra_fwhm_arcsec, self.fine_scale_arcsec)
        base = np.clip(base, 0.0, None)
        self._base_kernel = base / base.sum() * self.total_flux
        self._c = (base.shape[0] - 1) // 2

    @classmethod
    def from_hst(cls, instrument, band, pixel_scale_arcsec=0.04,
                 drizzle_fwhm_arcsec=None, n_fine=4, psf_files=HST_PSF_FILES,
                 stamp_ee=None):
        """instrument: 'wfc3' or 'acs' (case-insensitive); band e.g. 'F555W'.
        drizzle_fwhm_arcsec=None -> HST_DRIZZLE_FWHM_ARCSEC default for the band.
        stamp_ee : fraction of the total light inside the model stamp. None ->
            from the STScI encircled-energy tables at the stamp radius
            (hst_stamp_ee; 1.0 if the band is not tabulated); pass 1.0 to treat
            the stamp as the whole PSF (what the synthetic HST mocks were made with)."""
        instrument = instrument.lower()
        band = band.upper()
        psf, psf_scale = load_hst_psf(instrument, band, psf_files)
        if drizzle_fwhm_arcsec is None:
            drizzle_fwhm_arcsec = HST_DRIZZLE_FWHM_ARCSEC.get((instrument, band), 0.0)
        if stamp_ee is None:
            stamp_ee = hst_stamp_ee(instrument, band, (psf.shape[0] - 1) / 2 * psf_scale) or 1.0
        return cls(psf, psf_scale, pixel_scale_arcsec, extra_fwhm_arcsec=drizzle_fwhm_arcsec,
                   n_fine=n_fine, total_flux=stamp_ee, name=f'HST/{instrument.upper()} {band}')

    @classmethod
    def from_jwst(cls, band, pixel_scale_arcsec=None, extra_fwhm_arcsec=None, n_fine=4,
                  psf_dir=JWST_PSF_DIR):
        """band e.g. 'F200W', 'F300M', 'F770W' (case-insensitive).
        pixel_scale_arcsec=None -> PHANGS-JWST v4p1 mosaic scale for the band;
        extra_fwhm_arcsec=None -> JWST_EXTRA_FWHM_ARCSEC default for the band."""
        band = band.upper()
        camera, psf, psf_scale, arrsum = load_jwst_psf(band, psf_dir)
        if pixel_scale_arcsec is None:
            pixel_scale_arcsec = JWST_PIXEL_SCALE_ARCSEC[band]
        if extra_fwhm_arcsec is None:
            extra_fwhm_arcsec = JWST_EXTRA_FWHM_ARCSEC.get(band, 0.0)
        return cls(psf, psf_scale, pixel_scale_arcsec, extra_fwhm_arcsec=extra_fwhm_arcsec,
                   n_fine=n_fine, total_flux=arrsum, name=f'JWST/{camera} {band}')

    def kernel(self, intrinsic_fwhm_arcsec=0.0, half_size_fine=None):
        """
        Fine-grid kernel for a given intrinsic FWHM, optionally cropped to
        +-half_size_fine fine pixels around its center. Cropping is exact
        inside the crop (the Gaussian is applied to a crop padded by 5 sigma)
        and is NOT renormalized, so flux outside the crop stays missing.
        """
        c = self._c
        k = self._base_kernel
        sig_fine = (intrinsic_fwhm_arcsec or 0.0) * FWHM_TO_SIGMA / self.fine_scale_arcsec
        if half_size_fine is not None:
            pad = int(np.ceil(5 * sig_fine)) + 2
            h = min(int(half_size_fine) + pad, c)
            k = k[c - h:c + h + 1, c - h:c + h + 1]
        if sig_fine > 0.1:                       # below 0.1 fine px the Gaussian is a point (and can't be normalised)
            k = convolve_fft(k, _gaussian_fine_kernel(intrinsic_fwhm_arcsec, self.fine_scale_arcsec),
                             normalize_kernel=True, boundary='fill', fill_value=0.0)
            k = np.clip(k, 0.0, None)
        if half_size_fine is not None:
            ck = (k.shape[0] - 1) // 2
            h = min(int(half_size_fine), ck)
            k = k[ck - h:ck + h + 1, ck - h:ck + h + 1]
        return k

    def source_kernel(self, r_eff_arcsec, profile='gaussian', gamma=3.0, c=1.5,
                      half_size_fine=None, eff_r_max_over_reff=10.0):
        """
        Fine-grid kernel of PSF (x) source profile ('gaussian', 'eff' or
        'king') of half-light radius r_eff_arcsec, cropped to +-half_size_fine
        fine pixels (not renormalized: light falling outside the crop is
        missing, as in kernel()). For 'gaussian' this is kernel(2 * r_eff),
        since a 2-D Gaussian's half-light radius is FWHM / 2.
        """
        if profile == 'gaussian':
            return self.kernel(2.0 * r_eff_arcsec, half_size_fine)
        from scipy.signal import fftconvolve
        c0 = self._c
        k_half = c0 if half_size_fine is None else int(half_size_fine)
        # Light landing inside the crop comes from source light up to one crop
        # width further out (via the PSF), so the source profile is rendered
        # out to 2x the crop, independently of the (possibly smaller) PSF
        # stamp — the HST stamps are only ~0.5" in radius.
        hs = 2 * k_half
        hb = min(hs, c0)
        base = self._base_kernel[c0 - hb:c0 + hb + 1, c0 - hb:c0 + hb + 1]
        prof = render_profile_kernel(r_eff_arcsec, profile, self.fine_scale_arcsec, hs,
                                     gamma, c, eff_r_max_over_reff)
        k = fftconvolve(prof, base, mode='same')       # centered on prof's center
        k = np.clip(k, 0.0, None)
        return k[hs - k_half:hs + k_half + 1, hs - k_half:hs + k_half + 1]

    def clear_cache(self, keep=()):
        """Kept for API compatibility (kernels are no longer cached)."""

    def render(self, x0, y0, shape, intrinsic_fwhm_arcsec=0.0, flux=1.0,
               profile='gaussian', r_eff_arcsec=None, gamma=3.0, c=1.5):
        """Image of `shape` with one source of total `flux` at pixel (x0, y0).
        Source = Gaussian of FWHM intrinsic_fwhm_arcsec, or, when r_eff_arcsec
        is given, `profile` with that half-light radius. Only the part of the
        light that falls on the stamp is rendered."""
        ny, nx = shape
        reach_px = max(x0, nx - 1 - x0, y0, ny - 1 - y0) + 1
        half_fine = int(np.ceil(reach_px * self.n_fine)) + self.n_fine
        if r_eff_arcsec is None:
            k = self.kernel(intrinsic_fwhm_arcsec, half_fine)
        else:
            k = self.source_kernel(r_eff_arcsec, profile, gamma, c, half_fine)
        return render_sources([x0], [y0], [flux], shape, self.pixel_scale_arcsec, k, self.n_fine)

    def r_at_ee(self, intrinsic_fwhm_arcsec=0.0, target_ee=0.65, x0_frac=0.0, y0_frac=0.0,
                profile='gaussian', r_eff_arcsec=None, gamma=3.0, c=1.5):
        """
        Radius (arcsec) of the circular aperture that encloses target_ee of
        the model's TOTAL flux, measured the same way the data are: the model
        is rendered on the image's own pixel grid at the source's sub-pixel
        offset (x0_frac, y0_frac) and summed with exact-overlap photutils
        apertures. 1/target_ee is then the exact aperture correction for
        this model. Source as in render().
        """
        from photutils.aperture import CircularAperture, aperture_photometry

        half = 15
        while True:
            n = 2 * half + 1
            xc, yc = half + x0_frac, half + y0_frac
            img = self.render(xc, yc, (n, n), intrinsic_fwhm_arcsec, profile=profile,
                              r_eff_arcsec=r_eff_arcsec, gamma=gamma, c=c)
            radii_px = np.linspace(0.3, half - 1, 400)
            tbl = aperture_photometry(img, [CircularAperture((xc, yc), r) for r in radii_px])
            ee = np.array([tbl[f'aperture_sum_{i}'][0] for i in range(len(radii_px))], dtype=float)
            ee = np.maximum.accumulate(ee)
            if ee[-1] >= target_ee or half > 2000:
                break
            half *= 2
        if ee[-1] < target_ee:
            raise RuntimeError(f"r_at_ee: max EE {ee[-1]:.3f} < target {target_ee}")
        i = int(np.searchsorted(ee, target_ee))
        r = np.interp(target_ee, ee[i - 1:i + 1], radii_px[i - 1:i + 1]) if i > 0 else radii_px[0]
        return float(r * self.pixel_scale_arcsec)


@lru_cache(maxsize=None)
def _theoretical_psf_cached(band, instrume, pixel_scale_key):
    from psf_aux_func import fit_2d_moffat
    pix = pixel_scale_key / 1e6
    model, note = instrument_psf_for_image(band, {'INSTRUME': instrume}, pix, extra_fwhm_arcsec=0.0)
    if model is None:
        return float('nan'), float('nan'), float('nan'), note
    fwhm = model_star_fwhm(model)
    # Moffat shape with alpha free (as measure_native_psf_shape did on stars);
    # the bare PSF's strong wings can drive alpha to the fitter's 1.5 bound,
    # which fit_2d_moffat rejects -> fall back to the fixed-alpha 2.8 shape
    alphas, gammas = [], []
    half, n = 30, 60
    for dx, dy in ((0.0, 0.0), (0.37, -0.21)):
        img = model.render(half + dx, half + dy, (n, n))
        out = fit_2d_moffat(img, half, half, plot=False, fixed_alpha=False)
        if out[0] is not None and np.isfinite(out[4]):
            alphas.append(float(out[4]))
            gammas.append(float(out[5]))
    # A diffraction-limited JWST PSF (sharp core + rings) is not Moffat-shaped:
    # its free fit can give alpha >> 6, a Gaussian-like core without wings.
    # Keep a fitted alpha only in the physical range 1.5-6 (as
    # Photometry._warn_if_alpha_unphysical), else the standard 2.8.
    if alphas and 1.5 < np.median(alphas) <= 6.0:
        alpha, gamma = float(np.median(alphas)), float(np.median(gammas))
    else:
        why = 'failed' if not alphas else f'gave alpha={np.median(alphas):.1f} (not Moffat-like)'
        alpha = 2.8
        gamma = fwhm / pix / (2.0 * np.sqrt(2.0 ** (1.0 / alpha) - 1.0)) if np.isfinite(fwhm) else float('nan')
        note = (note + '; ' if note else '') + f'free-alpha Moffat fit {why}: alpha = 2.8'
    return float(fwhm), alpha, float(gamma), note


def theoretical_psf(band, header, pixel_scale_arcsec):
    """
    FWHM and Moffat shape of the THEORETICAL PSF of `band` (STScI effective
    PSF for HST, stpsf for JWST, no extra blur) on an image of this pixel
    scale, measured from a rendered model star with the same Moffat fit that
    filled psf_table.csv (FWHM: alpha fixed at 2.8, as model_star_fwhm;
    shape: alpha free). Nothing is measured on the data.

    header : the image header (INSTRUME picks WFC3 / ACS for HST) or None.
    Returns dict(fwhm_arcsec, alpha, gamma_px, note); NaN if the band has no
    PSF model. Cached per (band, camera, pixel scale).
    """
    inst = str((header or {}).get('INSTRUME', '')).strip().upper()
    fwhm, alpha, gamma, note = _theoretical_psf_cached(band.upper(), inst,
                                                       int(round(float(pixel_scale_arcsec) * 1e6)))
    return dict(fwhm_arcsec=fwhm, alpha=alpha, gamma_px=gamma, note=note)


def muse_fwhm_from_token(token):
    """MUSE PSF FWHM [arcsec] from a PHANGS-MUSE token such as
    'NGC1365-1.15asec'; None if the token carries no '<x>asec' part."""
    import re
    m = re.search(r'-([0-9]*\.?[0-9]+)asec', str(token))
    return float(m.group(1)) if m else None


def model_star_fwhm(model, half_size=30, offsets=((0.0, 0.0), (0.37, -0.21))):
    """
    FWHM (arcsec) of a point source rendered with `model`, measured exactly
    like the stars behind psf_table.csv (psf_aux_func.measure_psf_fwhm_from_image):
    2*half_size square cutout, 2-D Moffat + constant fit with alpha fixed at
    2.8. Median over a few sub-pixel positions; NaN if every fit fails.
    """
    from psf_aux_func import fit_2d_moffat
    vals = []
    n = 2 * half_size
    for dx, dy in offsets:
        img = model.render(half_size + dx, half_size + dy, (n, n))
        fitted = fit_2d_moffat(img, half_size, half_size, plot=False, fixed_alpha=True)[0]
        if fitted is not None and np.isfinite(fitted.fwhm):
            vals.append(float(fitted.fwhm) * model.pixel_scale_arcsec)
    return float(np.median(vals)) if vals else float('nan')


def match_extra_fwhm(build, target_fwhm_arcsec, max_extra_arcsec=0.3, tol_arcsec=0.002, max_iter=14):
    """
    Extra Gaussian blur (FWHM, arcsec) for which build(extra) — an
    InstrumentPSF — has model_star_fwhm == target_fwhm_arcsec, by bisection
    (the measured FWHM grows monotonically with the blur). Returns
    (extra, achieved_fwhm); extra = 0 if the bare model is already at least
    as broad as the target, max_extra_arcsec if even that is too sharp.
    """
    f_lo = model_star_fwhm(build(0.0))
    if not np.isfinite(f_lo) or f_lo >= target_fwhm_arcsec:
        return 0.0, f_lo
    f_hi = model_star_fwhm(build(max_extra_arcsec))
    if not np.isfinite(f_hi) or f_hi <= target_fwhm_arcsec:
        return max_extra_arcsec, f_hi
    lo, hi, f_mid, mid = 0.0, max_extra_arcsec, f_lo, 0.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = model_star_fwhm(build(mid))
        if not np.isfinite(f_mid):
            break
        if abs(f_mid - target_fwhm_arcsec) < tol_arcsec:
            break
        if f_mid < target_fwhm_arcsec:
            lo = mid
        else:
            hi = mid
    return mid, f_mid


def instrument_psf_for_image(band, header, pixel_scale_arcsec, n_fine=4, target_fwhm_arcsec=None,
                             extra_fwhm_arcsec=None):
    """
    InstrumentPSF for a real PHANGS image of `band`, on its own pixel scale.
    HST camera from header['INSTRUME'] (WFC3 or ACS). Band names follow the
    pipeline: an ACS 'F438W' is really F435W (renamed by
    run_full_pipeline.find_hst_band_paths); ACS F555W has no ACS model PSF
    and uses the WFC3 one.

    target_fwhm_arcsec : this galaxy's measured PSF FWHM for the band
        (psf_table.csv). If given, the extra blur on top of the model PSF is
        tuned so the model star's FWHM, measured the same way, matches it
        (match_extra_fwhm) instead of using the sample-median default
        (HST_DRIZZLE_FWHM_ARCSEC / JWST_EXTRA_FWHM_ARCSEC). Not done for
        MIRI, whose psf_table FWHMs are not physically consistent.
    extra_fwhm_arcsec : use exactly this extra blur (0 = the bare theoretical
        PSF: STScI effective PSF / stpsf model); overrides target_fwhm_arcsec.

    Returns (model, note) — model None if no PSF model exists for the band,
    note a short human-readable remark ('' if none). model.extra_fwhm_arcsec
    is the blur used; model.matched_fwhm_arcsec the model star's FWHM after
    tuning (NaN if not tuned).
    """
    band = band.upper()
    inst = str(header.get('INSTRUME', '')).strip().upper()
    notes = []
    try:
        if band in JWST_PIXEL_SCALE_ARCSEC:
            camera, psf, psf_scale, total = load_jwst_psf(band)
            default_extra = JWST_EXTRA_FWHM_ARCSEC.get(band, 0.0)
            name = f'JWST/{camera} {band}'
            tunable = camera != 'MIRI'
        else:
            camera = 'acs' if inst.startswith('ACS') else 'wfc3'
            if inst and not inst.startswith(('ACS', 'WFC3')):
                notes.append(f'unknown INSTRUME={inst!r}, assuming WFC3')
            psf_band = band
            if camera == 'acs' and band == 'F438W':
                psf_band = 'F435W'
                notes.append('ACS image: F435W PSF used for the F438W slot')
            with open(HST_PSF_FILES[camera], 'rb') as fh:
                available = pickle.load(fh).keys()
            if psf_band not in available:
                if camera == 'acs':
                    camera = 'wfc3'
                    notes.append(f'no ACS model PSF for {psf_band}: WFC3 {psf_band} PSF used')
                else:
                    return None, f'no HST model PSF for {psf_band}'
            psf, psf_scale = load_hst_psf(camera, psf_band)
            total = hst_stamp_ee(camera, psf_band, (psf.shape[0] - 1) / 2 * psf_scale)
            if total is None:
                total = 1.0
                notes.append(f'no encircled-energy table for {camera.upper()} {psf_band}: '
                             f'light beyond the PSF stamp ignored')
            default_extra = HST_DRIZZLE_FWHM_ARCSEC.get((camera, psf_band), 0.0)
            name = f'HST/{camera.upper()} {psf_band}'
            tunable = True

        def build(extra):
            return InstrumentPSF(psf, psf_scale, pixel_scale_arcsec, extra_fwhm_arcsec=extra,
                                 n_fine=n_fine, total_flux=total, name=name)

        extra, achieved = default_extra, float('nan')
        if extra_fwhm_arcsec is not None:
            extra = max(float(extra_fwhm_arcsec), 0.0)
        elif target_fwhm_arcsec is not None and np.isfinite(target_fwhm_arcsec) and target_fwhm_arcsec > 0:
            if tunable:
                extra, achieved = match_extra_fwhm(build, float(target_fwhm_arcsec))
                if not np.isfinite(achieved) or abs(achieved - target_fwhm_arcsec) > 0.01:
                    notes.append(f'PSF FWHM match imperfect: target {target_fwhm_arcsec:.3f}", '
                                 f'model {achieved:.3f}" with blur {extra:.3f}"')
            else:
                notes.append('MIRI: sample-default blur kept (psf_table FWHM not used)')
        model = build(extra)
        model.matched_fwhm_arcsec = achieved
        return model, '; '.join(notes)
    except (FileNotFoundError, KeyError) as e:
        return None, f'no model PSF for {band}: {e}'
