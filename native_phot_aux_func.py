"""
Native-resolution, aperture-corrected photometry for compact isolated sources.

v2 design (ground-up redesign, replacing the two-pass/per-slit-independent-
fit/least-squares-centroid-reconstruction approach of v1) — per source, per
band:

1. EMISSION DETECTION (detect_emission) — before anything else, test whether
   this band actually shows real emission for this source at the catalog
   position, using the band's own PSF-based R65 as a fixed-size test
   aperture. Genuinely per-source-per-band (no instrument special-casing),
   so it naturally handles both a MIRI band occasionally showing real
   emission (a YCL) and an optical band occasionally lacking it (an OGCC
   with no F275W flux) as the same code path.

2. 2-D GAUSSIAN CENTERING (fit_2d_gaussian_center) — only when emission is
   detected: fit Gaussian2D + Const2D on the raw local cutout (the Const2D
   term absorbs the sky pedestal) to find the real sub-pixel center. Only
   the fitted center is used downstream; width is diagnostic only.

3. AVERAGED-SLIT-PROFILE SIZE MEASUREMENT (measure_averaged_slit_profile) —
   sample n_slits (12) two-sided diameters through the fitted center,
   average the profiles POINTWISE across slits (so one NaN sample excludes
   only that point, never the whole slit), then fit ONE Gaussian1D+Const1D
   to the single averaged profile. This is the SAME function called both on
   synthetic calibration-model images (build_native_calibration_curve) and
   on real source cutouts, so the two code paths cannot drift apart.

4. R65 + correction factor — the measured sigma is inverted
   (estimate_extra_conv_fwhm) against the band-level calibration curve to
   estimate the source's own intrinsic size, and compute_r65 gives this
   cluster's own R65 from that estimate. A cheap re-check re-measures sigma
   with a WIDENED slit domain (matching the just-derived R65) when that R65
   implies a genuinely larger source than the small default window could
   see — catching a first-pass estimate that was constrained by too small a
   window rather than the source's real size — and refines R65 from that.
   Because R65 is, by construction, the radius enclosing exactly
   target_ee (0.65) of the flux of a model matching the estimated size, the
   correction factor is exactly 1/target_ee whenever a size estimate is
   used to derive the aperture — this is not a shortcut, it is the literal
   mathematical content of "R65 aperture". All of the real work is in
   getting the APERTURE RADIUS right, not in a variable correction factor.

5. APERTURE-BORROWING / PSF-FALLBACK for bands without detected emission
   (REFERENCE_BAND_GROUPS / get_reference_candidates) — a band with no
   emission borrows a similar-PSF band's own fitted position + R65
   (reused as-is, unscaled) if one is available for this same source;
   otherwise it falls back to its own PSF-based R65 (zero extra
   convolution) at the catalog position, still background-subtracted
   (expected to land near zero for a genuine non-detection, e.g. most MIRI
   bands). This is resolved per source in measure_native_photometry_one_source,
   which needs every band's round-1 (own-band) result before it can resolve
   round-2 (borrow/fallback) bands for that same source — bands are no
   longer independent tasks the way they were in v1.

This module is deliberately independent of `analysis.py`/`Photometry` state
(no `self`, no WCS/regions objects held as module state) so it is testable
on plain numpy arrays; `Photometry.run_native_photometry` (analysis.py) is
the only place that touches the pipeline's own image/PSF attributes and
assembles per-band setup dicts for measure_native_photometry_one_source.
"""

import logging
from collections import namedtuple

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import map_coordinates
from astropy.modeling import models, fitting
from astropy.convolution import Moffat2DKernel, convolve_fft
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats
from regions import CircleSkyRegion

from phot_aux_func import _bbox_local_mask_and_image, corr_flux

logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


# ─────────────────────────────────────────────────────────────────────────────
# Analytic Moffat curve of growth (unit-test cross-check only — not used in
# the runtime path, which renders images numerically so the *same* rendering
# machinery is exercised for both the base PSF and the Moffat-convolved
# models, for which no closed form exists).
# ─────────────────────────────────────────────────────────────────────────────

def moffat_gamma_from_fwhm(fwhm, alpha):
    """
    gamma such that astropy.modeling.models.Moffat2D(gamma=gamma, alpha=alpha)
    has the given FWHM, i.e. the inverse of Moffat2D.fwhm:
        fwhm = 2 * gamma * sqrt(2**(1/alpha) - 1)
    """
    return fwhm / (2.0 * np.sqrt(2.0 ** (1.0 / alpha) - 1.0))


def moffat_ee_analytic(r, fwhm, alpha):
    """
    Closed-form encircled energy of a single circular Moffat2D (astropy's
    parametrization: (1 + (r/gamma)**2)**(-alpha)), normalized to unit total
    flux:
        EE(r) = 1 - (1 + (r/gamma)**2)**(1 - alpha)
    Vectorized over r. Cross-checked against direct 2-D numerical
    integration in tests/test_native_photometry.py.
    """
    gamma = moffat_gamma_from_fwhm(fwhm, alpha)
    r = np.asarray(r, dtype=float)
    return 1.0 - (1.0 + (r / gamma) ** 2) ** (1.0 - alpha)


def moffat_r_at_ee_analytic(target_ee, fwhm, alpha):
    """Closed-form inverse of moffat_ee_analytic: radius at which the
    analytic single-Moffat curve of growth reaches `target_ee`."""
    gamma = moffat_gamma_from_fwhm(fwhm, alpha)
    return gamma * np.sqrt((1.0 - target_ee) ** (1.0 / (1.0 - alpha)) - 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Model-image rendering
# ─────────────────────────────────────────────────────────────────────────────

def render_moffat_model_image(fwhm_arcsec, alpha, extra_conv_fwhm_arcsec,
                                pixel_scale_arcsec, oversample=4, size_fwhm=20):
    """
    Render a small, oversampled, unit-flux-normalized image of a circular
    Moffat2D of the given FWHM/alpha, optionally convolved with a second
    Moffat2DKernel of FWHM `extra_conv_fwhm_arcsec` (same alpha) — the
    "PSF-convolved-with-Moffat-of-growing-size" model at the heart of the
    calibration curve.

    Parameters
    ----------
    fwhm_arcsec            : float — base (PSF) Moffat FWHM
    alpha                  : float — Moffat shape parameter (shared by both
                              the base and the extra-convolution kernel)
    extra_conv_fwhm_arcsec : float — extra convolution FWHM; 0 means "no
                              extra convolution" (model index 0, the PSF
                              itself)
    pixel_scale_arcsec     : float — NATIVE pixel scale of the band this
                              model represents (before oversampling)
    oversample             : int — render at pixel_scale_arcsec/oversample
    size_fwhm              : float — image half-width in units of the total
                              (base + extra, in quadrature-ish sense — see
                              below) FWHM, generous enough that the wings
                              are not truncated

    Returns
    -------
    img          : 2-D ndarray, normalized to sum() == 1
    render_scale : float — the actual pixel scale (arcsec) of `img`
    """
    render_scale = pixel_scale_arcsec / oversample

    # Generous half-size: base FWHM plus extra convolution FWHM (in arcsec),
    # converted to oversampled pixels, times a safety factor so the Moffat's
    # power-law wings aren't clipped.
    total_fwhm_arcsec = fwhm_arcsec + extra_conv_fwhm_arcsec
    half_size_px = int(np.ceil(size_fwhm * total_fwhm_arcsec / render_scale / 2.0))
    half_size_px = max(half_size_px, 20)  # floor so tiny FWHMs still render sensibly
    n = 2 * half_size_px + 1
    yy, xx = np.mgrid[0:n, 0:n]
    x0 = y0 = half_size_px

    gamma_base = moffat_gamma_from_fwhm(fwhm_arcsec, alpha) / render_scale
    base = models.Moffat2D(amplitude=1.0, x_0=x0, y_0=y0,
                            gamma=gamma_base, alpha=alpha)
    img = base(xx, yy)
    img /= img.sum()

    if extra_conv_fwhm_arcsec > 0:
        gamma_extra = moffat_gamma_from_fwhm(extra_conv_fwhm_arcsec, alpha) / render_scale
        kernel = Moffat2DKernel(gamma=gamma_extra, alpha=alpha)
        img = convolve_fft(img, kernel, normalize_kernel=True,
                            boundary='fill', fill_value=0.0)
        img[img < 0] = 0.0  # guard tiny negative ringing from FFT convolution
        img /= img.sum()

    return img, render_scale


# ─────────────────────────────────────────────────────────────────────────────
# R65: the 65%-encircled-energy radius of a Moffat model — either the bare
# PSF (extra_conv_fwhm_arcsec=0, the band-wide value) or a specific
# CONVOLVED model representing one cluster's own estimated intrinsic size.
# ─────────────────────────────────────────────────────────────────────────────

def gaussian_r_at_ee(sigma_arcsec, target_ee=0.65):
    """
    Closed-form radius enclosing `target_ee` of the flux of a CIRCULAR 2-D
    GAUSSIAN with 1-D sigma `sigma_arcsec` — i.e. taking a measured sigma at
    face value, with no PSF model, no Moffat shape, no calibration curve:

        EE(r) = 1 - exp(-r^2 / (2 sigma^2))
        r_ee  = sigma * sqrt(-2 * ln(1 - target_ee))

    This is the "no fancy steps" alternative requested alongside
    compute_r65's method='gaussian': plug the averaged-slit-profile's own
    fitted sigma straight into the standard Gaussian formula. It is exact
    ONLY if the thing being measured (PSF, or PSF+intrinsic broadening) is
    genuinely Gaussian-shaped — see compute_r65's docstring for why that is
    usually NOT a safe assumption for a real HST/JWST PSF, which this
    pipeline's own PSF fits (psf_aux_func.fit_2d_moffat) model as a Moffat,
    not a Gaussian, precisely because of its heavier-than-Gaussian wings.
    """
    return float(sigma_arcsec * np.sqrt(-2.0 * np.log(1.0 - target_ee)))


def compute_r65(fwhm_arcsec, alpha, pixel_scale_arcsec, target_ee=0.65,
                 method='numeric', oversample=4, extra_conv_fwhm_arcsec=0.0,
                 sigma_arcsec=None, psf_model=None, intrinsic_fwhm_arcsec=0.0):
    """
    R65 of a Moffat model: the bare PSF by default (extra_conv_fwhm_arcsec=0,
    matching the paper's own "R65 of the PSF" definition), or a specific
    PSF-convolved-with-Moffat model (extra_conv_fwhm_arcsec > 0) — used to
    compute a CLUSTER's own R65 from its estimated intrinsic size, rather
    than the band's fixed PSF-based one.

    method='numeric' (default, used at runtime): render the model image,
        compute its curve of growth on a fine radius grid via cumulative
        circular-mask sums, and invert with a monotonic Pchip interpolator
        to find r where EE(r) == target_ee.
    method='closed_form': moffat_r_at_ee_analytic — only valid for
        extra_conv_fwhm_arcsec=0 (a single Moffat has a closed-form curve of
        growth; a Moffat-convolved-with-Moffat does not), exposed only for
        the cross-check unit test (agreement with 'numeric' validates the
        rendering machinery reused for the convolved models, where no
        closed form exists).
    method='gaussian': gaussian_r_at_ee(sigma_arcsec, target_ee) — bypasses
        the Moffat model and the calibration curve entirely. Requires
        `sigma_arcsec` (e.g. the raw sigma_arcsec straight out of
        measure_averaged_slit_profile on the real source). fwhm_arcsec,
        alpha, pixel_scale_arcsec, oversample, and extra_conv_fwhm_arcsec
        are all ignored in this mode.

        WHY the default path does NOT just do this — two independent
        reasons, both real:

        1. Real HST/JWST PSFs are not Gaussian. This pipeline's own PSF
           characterization (psf_aux_func.fit_2d_moffat / psf_table.csv)
           fits a Moffat2D, not a Gaussian, because real diffraction PSFs
           have heavier wings than a Gaussian of "the same width" — a
           Moffat with a realistic alpha (~2-4) carries meaningfully more
           flux beyond r65 than a Gaussian does. Using the Gaussian
           closed-form on a Moffat-shaped source therefore UNDER-estimates
           the true r65 (the radius that really encloses target_ee of a
           heavier-tailed profile is larger) — the resulting aperture then
           encloses LESS than target_ee of the real flux, so dividing by
           1/target_ee systematically UNDER-corrects the total flux. This
           is the leading suspect for a flat ~20-30% flux deficit.
        2. The sigma measured by measure_averaged_slit_profile is not a
           trustworthy physical width to begin with: it comes from fitting
           a 1-D GAUSSIAN model to a profile that is actually Moffat-shaped
           (PSF, or PSF-convolved-with-the-source's-own-profile). A Gaussian
           least-squares fit to Moffat-shaped data does not recover "the"
           Moffat width in any standard closed form — it is only a
           self-consistent SHAPE STATISTIC once you know how that same
           fitting procedure behaves on models of known size. That is
           exactly what build_native_calibration_curve exists to do: it
           measures this same biased "sigma" on synthetic Moffat-PSF-
           convolved-with-Moffat models of KNOWN size, using the identical
           measure_averaged_slit_profile call, then estimate_extra_conv_fwhm
           inverts that empirical mapping for a real source. Skipping the
           calibration curve means sigma_arcsec is used as if it were a
           literal physical Gaussian sigma, which it was never calibrated
           to be.

        In short: method='gaussian' is a legitimate, much simpler
        approximation, but it reintroduces both of the systematic biases
        the Moffat+calibration-curve machinery exists to remove. It is
        provided here for exactly the kind of A/B comparison this
        docstring's caller is likely doing — not because it is expected to
        be more accurate. If it turns out to match reality better than the
        default path, that is itself strong evidence of a bug in the
        calibration curve (e.g. a wrong alpha in psf_table.csv, or a
        truncated/extrapolated calibration range) rather than evidence that
        the Moffat/calibration approach is the wrong idea in principle.

    method='instrument': R65 of Gaussian(intrinsic_fwhm_arcsec) (x) the real
        instrument PSF `psf_model` (instrument_psf.InstrumentPSF, e.g.
        InstrumentPSF.from_hst('wfc3', 'F555W', 0.04)), rendered on the image
        pixel grid — use with fit_slit_profile_instrument_psf, whose result
        already carries this as r65_arcsec. Only target_ee, psf_model and
        intrinsic_fwhm_arcsec are used.

    Note: because R65 is DEFINED as the radius enclosing exactly target_ee
    of a model's flux, the encircled energy at r65 for that same model is
    target_ee by construction — this is why measure_native_photometry_one_source
    uses corr_factor = 1/target_ee directly rather than a separate lookup
    once a source's own R65 has been derived from its own size estimate.
    (This same logic — corr_factor = 1/target_ee — still applies exactly
    under method='gaussian', since gaussian_r_at_ee's r_ee is likewise
    defined as the radius enclosing exactly target_ee of ITS model's flux.)

    Returns
    -------
    r65_arcsec : float
    """
    if method == 'gaussian':
        if sigma_arcsec is None:
            raise ValueError("method='gaussian' requires sigma_arcsec "
                              "(the measured averaged-slit-profile sigma)")
        return gaussian_r_at_ee(sigma_arcsec, target_ee)

    if method == 'instrument':
        if psf_model is None:
            raise ValueError("method='instrument' requires psf_model (an InstrumentPSF)")
        return psf_model.r_at_ee(intrinsic_fwhm_arcsec, target_ee)

    if method == 'closed_form':
        if extra_conv_fwhm_arcsec != 0.0:
            raise ValueError("method='closed_form' is only valid for extra_conv_fwhm_arcsec=0.0 "
                              "(a single Moffat has a closed-form curve of growth; a "
                              "Moffat-convolved-with-Moffat model does not)")
        return float(moffat_r_at_ee_analytic(target_ee, fwhm_arcsec, alpha))

    if method != 'numeric':
        raise ValueError(f"Unknown method {method!r} — must be 'numeric', 'closed_form', 'gaussian' or 'instrument'")

    img, render_scale = render_moffat_model_image(
        fwhm_arcsec, alpha, extra_conv_fwhm_arcsec=extra_conv_fwhm_arcsec,
        pixel_scale_arcsec=pixel_scale_arcsec, oversample=oversample,
    )
    ny, nx = img.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    r_px = np.sqrt((xx - nx // 2) ** 2 + (yy - ny // 2) ** 2)

    r_max_px = min(nx, ny) // 2 - 1
    radii_px = np.linspace(0.5, r_max_px, 200)
    ee = np.array([img[r_px <= rp].sum() for rp in radii_px])

    if ee[-1] < target_ee:
        raise RuntimeError(
            f"compute_r65: rendered image too small — max EE={ee[-1]:.3f} "
            f"never reaches target_ee={target_ee}; increase size_fwhm."
        )

    # ee is monotonically non-decreasing in radii_px BY PHYSICAL CONSTRUCTION
    # (cumulative sum of a non-negative image), but floating-point/pixelization
    # noise can produce tiny local decreases. np.maximum.accumulate enforces
    # GLOBAL monotonicity (a proper isotonic correction), unlike a local
    # pairwise "keep if greater than the previous RAW point" filter — the
    # latter can still let the retained sequence decrease overall.
    ee_monotonic = np.maximum.accumulate(ee)
    strictly_increasing = np.concatenate([[True], np.diff(ee_monotonic) > 0])
    ee_unique = ee_monotonic[strictly_increasing]
    radii_unique = radii_px[strictly_increasing]

    inv = PchipInterpolator(ee_unique, radii_unique * render_scale, extrapolate=False)
    r65 = float(inv(target_ee))
    return r65


# ─────────────────────────────────────────────────────────────────────────────
# Emission detection — genuinely per-source-per-band, runs BEFORE any
# centering/R65 exists for this source (it uses the band's fixed PSF-based
# R65 purely as a test aperture).
# ─────────────────────────────────────────────────────────────────────────────

EmissionTestResult = namedtuple('EmissionTestResult', [
    'has_emission',        # bool
    'peak_snr',            # float — (max source-aperture pixel - bkg) / bkg_std
    'aperture_snr',        # float — (bkg-subtracted aperture sum) / its propagated error
    'bkg_median_per_pix',  # float — sigma-clipped annulus background level
    'bkg_std',             # float — sigma-clipped annulus background scatter
    'status',              # str — 'ok', 'no_emission', 'no_flux', or 'no_bkg'
])


def detect_emission(image, header, wcs, center_sky, r65_psf_arcsec,
                     annulus_r_in_factor=2.6, annulus_r_out_factor=4.0,
                     peak_snr_threshold=5.0, aperture_snr_threshold=3.0):
    """
    Test whether there is real emission for this source in this band, at the
    CATALOG position (no recentering yet — that itself needs to know a peak
    exists) using the band's fixed PSF-based R65 as a test aperture and a
    sigma-clipped background annulus around it. Both criteria must pass:

    - peak_snr  = (brightest source-aperture pixel - bkg) / bkg_std >= 5.0
    - aperture_snr = (bkg-subtracted aperture sum) / its propagated error >= 3.0

    Requiring both guards against a single hot pixel (fails aperture_snr)
    and a very extended, low-contrast excess with no real peak (fails
    peak_snr). Genuinely per-source-per-band — no instrument special-casing
    — so a MIRI band with real emission (a YCL) and an optical band without
    it (an OGCC) are handled by the exact same code path.

    Returns
    -------
    EmissionTestResult
    """
    r_in_arcsec = annulus_r_in_factor * r65_psf_arcsec
    r_out_arcsec = annulus_r_out_factor * r65_psf_arcsec
    source_region = CircleSkyRegion(center=center_sky, radius=r65_psf_arcsec * u.arcsec)
    inner_region = CircleSkyRegion(center=center_sky, radius=r_in_arcsec * u.arcsec)
    outer_region = CircleSkyRegion(center=center_sky, radius=r_out_arcsec * u.arcsec)

    # img_local must come from the OUTER region's own call: _bbox_local_mask_
    # and_image only populates img_local within the PASSED region's own
    # footprint (zero elsewhere in the shared ref_bbox frame — see its
    # docstring), and ref_bbox is itself outer_region's bbox, so only the
    # outer_region call's img_local covers the full annulus. Taking it from
    # source_region (the smallest region) instead would silently zero out
    # every annulus pixel, making the background always read as exactly
    # zero — see the identical, already-correct pattern in
    # phot_aux_func.measure_photometry, which sources img_local from its own
    # (larger) annulus_region rather than the smaller source region.
    ref_bbox = outer_region.to_pixel(wcs).to_mask().bbox
    img_local, mask_outer = _bbox_local_mask_and_image(outer_region, wcs, image, ref_bbox)
    _, mask_source = _bbox_local_mask_and_image(source_region, wcs, image, ref_bbox)
    _, mask_inner = _bbox_local_mask_and_image(inner_region, wcs, image, ref_bbox)

    nan = float('nan')
    if mask_source.astype(bool).sum() == 0:
        return EmissionTestResult(False, nan, nan, nan, nan, 'no_flux')

    annulus_mask = mask_outer.astype(bool) & ~mask_inner.astype(bool)
    ann_values = np.where(annulus_mask, img_local, np.nan)
    _, bkg_median_per_pix, bkg_std = sigma_clipped_stats(ann_values, sigma=3.0)

    if not np.isfinite(bkg_std) or bkg_std <= 0:
        return EmissionTestResult(False, nan, nan,
                                   float(bkg_median_per_pix) if np.isfinite(bkg_median_per_pix) else nan,
                                   nan, 'no_bkg')

    source_mask_bool = mask_source.astype(bool)
    n_source_pix = int(source_mask_bool.sum())
    source_vals = np.where(source_mask_bool, img_local, np.nan)

    peak_val = float(np.nanmax(source_vals))
    peak_snr = (peak_val - bkg_median_per_pix) / bkg_std

    raw_sum = float(np.nansum(source_vals))
    bsub_sum = raw_sum - bkg_median_per_pix * n_source_pix
    aperture_err = bkg_std * np.sqrt(n_source_pix)
    aperture_snr = bsub_sum / aperture_err if aperture_err > 0 else nan

    has_emission = bool(
        np.isfinite(peak_snr) and peak_snr >= peak_snr_threshold
        and np.isfinite(aperture_snr) and aperture_snr >= aperture_snr_threshold
    )

    return EmissionTestResult(
        has_emission, float(peak_snr), float(aperture_snr) if np.isfinite(aperture_snr) else nan,
        float(bkg_median_per_pix), float(bkg_std), 'ok' if has_emission else 'no_emission',
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2-D Gaussian centering — replaces the old peak-argmax + local-centroid
# recenter_on_peak. Follows the same fitting idiom as
# psf_aux_func.fit_2d_moffat (peak-based initial guess, bounded center,
# TRFLSQFitter, symmetric bounds-hit convergence check).
# ─────────────────────────────────────────────────────────────────────────────

Gaussian2DCenterResult = namedtuple('Gaussian2DCenterResult', [
    'x0_local', 'y0_local',           # fitted center, image_local pixel coords
    'sigma_x_arcsec', 'sigma_y_arcsec',  # diagnostic only — never used downstream;
                                       #   size measurement lives entirely in
                                       #   measure_averaged_slit_profile
    'amplitude', 'background',
    'converged',                      # bool — False falls back to the initial guess
])


def fit_2d_gaussian_center(image_local, x0_guess, y0_guess, fit_half_size_px,
                            pixel_scale_arcsec):
    """
    Fit Gaussian2D + Const2D (the Const2D absorbs the sky pedestal, so the
    RAW — not background-subtracted — cutout is used directly, avoiding a
    chicken-and-egg center/background dependency) on a square window around
    (x0_guess, y0_guess) in image_local's own pixel frame.

    Only the fitted center is returned for downstream use; x/y widths are
    diagnostic only (this function never estimates the source's SIZE — that
    is measure_averaged_slit_profile's job, kept as the one place size is
    measured, so two independent size estimates can never disagree).

    Returns
    -------
    Gaussian2DCenterResult — converged=False (falling back to
    (x0_guess, y0_guess)) if the cutout has too little data or the fit does
    not converge / parks at a bound.
    """
    ny, nx = image_local.shape
    half = max(int(round(fit_half_size_px)), 2)
    ix0 = int(np.clip(round(x0_guess), 0, nx - 1))
    iy0 = int(np.clip(round(y0_guess), 0, ny - 1))

    x_lo, x_hi = max(ix0 - half, 0), min(ix0 + half + 1, nx)
    y_lo, y_hi = max(iy0 - half, 0), min(iy0 + half + 1, ny)
    cutout = image_local[y_lo:y_hi, x_lo:x_hi]

    nan = float('nan')
    finite = np.isfinite(cutout)
    if finite.sum() < 6:
        return Gaussian2DCenterResult(x0_guess, y0_guess, nan, nan, nan, nan, False)

    yy, xx = np.mgrid[y_lo:y_hi, x_lo:x_hi]

    peak_val = float(np.nanmax(cutout))
    bkg_guess = float(np.nanmedian(cutout))
    amp_guess = max(peak_val - bkg_guess, 1e-9)
    x_guess = float(x0_guess)
    y_guess = float(y0_guess)
    stddev_guess = max(fit_half_size_px / 3.0, 0.5)

    gauss = models.Gaussian2D(
        amplitude=amp_guess, x_mean=x_guess, y_mean=y_guess,
        x_stddev=stddev_guess, y_stddev=stddev_guess, theta=0.0,
    )
    const = models.Const2D(amplitude=bkg_guess)
    composite = gauss + const
    composite.theta_0.fixed = True  # axis-aligned only — no rotation needed

    tol = max(0.5 * fit_half_size_px, 1.0)
    composite.amplitude_0.bounds = (0.05 * amp_guess, None)
    composite.x_mean_0.bounds = (x_guess - tol, x_guess + tol)
    composite.y_mean_0.bounds = (y_guess - tol, y_guess + tol)
    composite.x_stddev_0.bounds = (0.3, fit_half_size_px)
    composite.y_stddev_0.bounds = (0.3, fit_half_size_px)

    fitter = fitting.TRFLSQFitter(calc_uncertainties=False)
    try:
        fitted = fitter(composite, xx[finite], yy[finite], cutout[finite], maxiter=1000)
    except Exception as e:
        logger.debug(f"fit_2d_gaussian_center: fit raised {e}")
        return Gaussian2DCenterResult(x0_guess, y0_guess, nan, nan, nan, nan, False)

    fit_info = fitter.fit_info
    status = fit_info.get('status', fit_info.get('ierr', 0))
    x_fitted = fitted.x_mean_0.value
    y_fitted = fitted.y_mean_0.value

    at_x_bound = (np.isclose(x_fitted, x_guess - tol, atol=1e-3)
                  or np.isclose(x_fitted, x_guess + tol, atol=1e-3))
    at_y_bound = (np.isclose(y_fitted, y_guess - tol, atol=1e-3)
                  or np.isclose(y_fitted, y_guess + tol, atol=1e-3))
    converged = (status in (1, 2, 3, 4)) and not at_x_bound and not at_y_bound

    if not converged:
        return Gaussian2DCenterResult(x0_guess, y0_guess, nan, nan, nan, nan, False)

    return Gaussian2DCenterResult(
        x0_local=float(x_fitted), y0_local=float(y_fitted),
        sigma_x_arcsec=abs(float(fitted.x_stddev_0.value)) * pixel_scale_arcsec,
        sigma_y_arcsec=abs(float(fitted.y_stddev_0.value)) * pixel_scale_arcsec,
        amplitude=float(fitted.amplitude_0.value), background=float(fitted.amplitude_1.value),
        converged=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Averaged-slit-profile size measurement — the ONE function used both to
# build the calibration curve (on synthetic models) and to measure a real
# source. Samples n_slits two-sided diameters, averages them POINTWISE
# (so one NaN sample excludes only that point, never the whole slit), then
# fits ONE Gaussian1D + Const1D to the single averaged profile.
# ─────────────────────────────────────────────────────────────────────────────

AveragedSlitProfile = namedtuple('AveragedSlitProfile', [
    'sigma_arcsec', 'sigma_err_arcsec', 'mean_arcsec',
    'amplitude', 'background',
    'n_slits_with_data',  # int — max, over along-slit positions, of how many
                          #   of the n_slits slits had a finite sample there
    'converged',
])

AveragedSlitProfileDiagnostics = namedtuple('AveragedSlitProfileDiagnostics', [
    'along_arcsec',        # ndarray (n_along,) — the shared along-slit domain
    'theta_deg',            # ndarray (n_slits,) — each slit's position angle
    'per_slit_profiles',    # ndarray (n_slits, n_along) — raw sampled flux
    'averaged_profile',      # ndarray (n_along,) — pointwise nanmean across slits
    'fitted_curve',          # ndarray (n_along,) — fitted Gaussian+Const, NaN if
                            #   the fit failed
])


# How the n_slits profiles are combined pointwise into the one profile that is
# fitted: 'median' (default; a neighbour or artefact crossing a few slits does
# not pull the profile) or 'mean'. Used identically for the data, the native
# calibration curves and the instrument-fit models, so they stay consistent.
SLIT_COMBINE = 'median'


def _sample_averaged_slit_profile(image, x0, y0, pixel_scale_arcsec,
                                  slit_length_arcsec=1.0, n_slits=12, slit_width_px=2,
                                  combine=None):
    """
    Sample n_slits diameters through (x0, y0) and combine them pointwise
    (median, or mean; combine=None -> SLIT_COMBINE) — the sampling half of
    measure_averaged_slit_profile, shared with fit_slit_profile_instrument_psf
    so a model image and the data are extracted identically. The samples
    across one slit's width are always averaged.

    Returns (along_arcsec, theta_deg, per_slit_profiles, averaged_profile,
    n_slits_with_data).
    """
    min_samples = 20
    along_step_arcsec = min(pixel_scale_arcsec / 4.0, slit_length_arcsec / min_samples)
    along_step_arcsec = max(along_step_arcsec, 1e-6)
    n_along_half = max(int(np.ceil(slit_length_arcsec / along_step_arcsec)), 8)
    along_arcsec = np.linspace(-slit_length_arcsec, slit_length_arcsec, 2 * n_along_half - 1)
    along_px = along_arcsec / pixel_scale_arcsec

    if slit_width_px <= 1:
        width_offsets_px = np.array([0.0])
    else:
        width_offsets_px = np.linspace(-slit_width_px / 2, slit_width_px / 2, slit_width_px)

    theta_deg = np.arange(n_slits) * 180.0 / n_slits
    per_slit_profiles = np.full((n_slits, len(along_arcsec)), np.nan)

    for k in range(n_slits):
        theta = np.deg2rad(theta_deg[k])
        dx, dy = np.cos(theta), np.sin(theta) #now
        pdx, pdy = -np.sin(theta), np.cos(theta)
        samples = np.zeros((len(width_offsets_px), len(along_px)))
        for wi, w_off in enumerate(width_offsets_px):
            xs = x0 + along_px * dx + w_off * pdx
            ys = y0 + along_px * dy + w_off * pdy
            coords = np.vstack([ys, xs])
            # order=1 (bilinear), not order>=2: order>=2 spline interpolation
            # requires a "prefilter" pass over the WHOLE input array (turning
            # raw pixel values into actual B-spline coefficients) to be a
            # true interpolant; skipping it (prefilter=False) — which is
            # otherwise required, since the default prefilter is a global,
            # long-range (IIR-like) filter and a single NaN anywhere in
            # `image` otherwise propagates to EVERY sampled point across ALL
            # slits, not just samples near the NaN (verified: a 3x3 NaN hole
            # turned an entire 481-point profile, including a slit 90 degrees
            # away from the hole, into all-NaN) — means the raw samples are
            # used AS IF they were already spline coefficients, which they
            # are not: this silently SMOOTHS/attenuates the reconstructed
            # profile rather than interpolating it (verified directly: at
            # the exact peak pixel of a synthetic source, order=3 with
            # prefilter=False read back ~22% below the true pixel value,
            # while order=1 read back the exact value). Since this
            # pipeline's PSFs are routinely only a few pixels wide, that
            # smoothing biased sigma/amplitude for genuinely undersampled
            # sources. order=1 has no such prefilter step at all (a no-op
            # regardless of the prefilter argument), so it reproduces the
            # data exactly at grid points AND keeps a NaN's effect confined
            # to its immediate neighboring samples only — solving both
            # problems at once, with no accuracy trade-off actually lost
            # (the "better" prefiltered cubic spline was never running).
            samples[wi] = map_coordinates(image, coords, order=1,
                                            mode='constant', cval=np.nan,
                                            prefilter=False)
        with np.errstate(all='ignore'):
            per_slit_profiles[k] = np.nanmean(samples, axis=0)

    combine = SLIT_COMBINE if combine is None else combine
    if combine not in ('median', 'mean'):
        raise ValueError(f"combine must be 'median' or 'mean', got {combine!r}")
    with np.errstate(all='ignore'):
        n_with_data = np.sum(np.isfinite(per_slit_profiles), axis=0)
        averaged_profile = (np.nanmedian(per_slit_profiles, axis=0) if combine == 'median'
                            else np.nanmean(per_slit_profiles, axis=0))
    n_slits_with_data = int(np.max(n_with_data)) if len(n_with_data) else 0
    return along_arcsec, theta_deg, per_slit_profiles, averaged_profile, n_slits_with_data


def measure_averaged_slit_profile(image, x0, y0, pixel_scale_arcsec,
                                    slit_length_arcsec=1.0, n_slits=12,
                                    slit_width_px=2, collect_diagnostics=False,
                                    plot=False):
    """
    Measure a source-size proxy sigma from n_slits diameters through
    (x0, y0), each of HALF-length slit_length_arcsec on each side (so a
    default of 1.0" gives a 2" total diameter, per the pipeline's own
    calibration convention), averaged POINTWISE across slits before fitting
    — a NaN sample at one along-slit position in one slit simply drops out
    of that position's mean, it never discards the rest of that slit.

    This directly addresses the per-slit-independent-fit design's own
    weakness (each of 12 independent background pedestals disagreeing with
    each other, visible as different background LEVELS on the same source
    in different directions): averaging first means only ONE Gaussian+Const
    fit — one background level — is ever fit for a given source/band.

    Parameters
    ----------
    image                : 2-D ndarray. Does NOT need to be background-
                            subtracted — the Const1D term absorbs the sky
                            pedestal, same reasoning as fit_2d_gaussian_center.
    x0, y0                : float — profile center, `image` PIXEL coordinates.
    pixel_scale_arcsec     : float — arcsec/pixel of `image`.
    slit_length_arcsec     : float — HALF-length of each diameter, i.e. the
                            diameter samples from -slit_length_arcsec to
                            +slit_length_arcsec. Default 1.0" (2" diameter).
    n_slits                : int — number of diameters, spaced 180/n_slits
                            degrees apart (a diameter at theta is the same
                            LINE as one at theta+180 — genuinely distinct
                            orientations only span a half-turn).
    slit_width_px           : int — parallel sample lines spanning the
                            slit's width, averaged per along-slit position.
    collect_diagnostics      : bool — if True, also return an
                            AveragedSlitProfileDiagnostics record with the
                            raw per-slit profiles, the averaged profile, and
                            the fitted curve (used by
                            Photometry.plot_native_slit_profiles for visual
                            QA). False (default) keeps the second return
                            value as None.
    plot                     : bool — if True, show two diagnostic
                            matplotlib figures (matplotlib is imported
                            lazily, only when plot=True, so the numpy-only
                            test path for this module is unaffected):
                            (1) `image` with all n_slits diameters (and
                            their width, if slit_width_px > 1) overlaid
                            through (x0, y0), each slit drawn in its own
                            color; (2) along_arcsec on the x-axis with each
                            slit's raw sampled profile in that SAME color,
                            plus the pointwise-averaged profile and the
                            fitted Gaussian+Const curve, so a given slit in
                            the image and its profile are identifiable by
                            color alone. Plotted regardless of whether the
                            fit converges — a failed/nonsensical fit is
                            exactly when this is most useful.

    Returns
    -------
    (AveragedSlitProfile, diagnostics_or_None)
    """
    along_arcsec, theta_deg, per_slit_profiles, averaged_profile, n_slits_with_data = \
        _sample_averaged_slit_profile(image, x0, y0, pixel_scale_arcsec,
                                      slit_length_arcsec, n_slits, slit_width_px)

    diag = dict(along_arcsec=along_arcsec, theta_deg=theta_deg,
                per_slit_profiles=per_slit_profiles, averaged_profile=averaged_profile,
                fitted_curve=np.full_like(along_arcsec, np.nan))

    def _plot_slits_and_profiles():
        import matplotlib.pyplot as plt

        colors = plt.cm.hsv(np.linspace(0, 1, n_slits, endpoint=False))
        half_len_px = slit_length_arcsec / pixel_scale_arcsec
        half_width_px = (slit_width_px / 2.0) if slit_width_px > 1 else 0.0

        # ── figure 1: slits (and their width) drawn over the source cutout ──
        fig1, ax1 = plt.subplots(figsize=(6, 6))
        finite_img = image[np.isfinite(image)]
        if finite_img.size:
            vmin, vmax = np.nanpercentile(finite_img, [1, 99.5])
        else:
            vmin, vmax = 0.0, 1.0
        ax1.imshow(image, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
        for k in range(n_slits):
            theta = np.deg2rad(theta_deg[k])
            dx, dy = np.cos(theta), np.sin(theta)
            pdx, pdy = -np.sin(theta), np.cos(theta)
            if half_width_px > 0:
                # Draw the slit as the rectangle it actually samples (length
                # x width), not just its centerline — its 4 corners, in
                # order, going out along +dx/+dy then across by ±half_width.
                corners_x = [
                    x0 - half_len_px * dx - half_width_px * pdx,
                    x0 + half_len_px * dx - half_width_px * pdx,
                    x0 + half_len_px * dx + half_width_px * pdx,
                    x0 - half_len_px * dx + half_width_px * pdx,
                ]
                corners_y = [
                    y0 - half_len_px * dy - half_width_px * pdy,
                    y0 + half_len_px * dy - half_width_px * pdy,
                    y0 + half_len_px * dy + half_width_px * pdy,
                    y0 - half_len_px * dy + half_width_px * pdy,
                ]
                ax1.fill(corners_x, corners_y, color=colors[k], alpha=0.35,
                          edgecolor=colors[k], linewidth=1.2, zorder=5,
                          label=f'{theta_deg[k]:.0f} deg')
            else:
                x_end = np.array([x0 - half_len_px * dx, x0 + half_len_px * dx])
                y_end = np.array([y0 - half_len_px * dy, y0 + half_len_px * dy])
                ax1.plot(x_end, y_end, '-', color=colors[k], linewidth=1.3, alpha=0.9,
                          zorder=5, label=f'{theta_deg[k]:.0f} deg')
        ax1.plot(x0, y0, '+', color='white', markersize=10, markeredgewidth=1.5, zorder=6)
        pad = half_len_px * 1.2
        ax1.set_xlim(x0 - pad, x0 + pad)
        ax1.set_ylim(y0 - pad, y0 + pad)
        ax1.set_title(f'{n_slits} slits — length={slit_length_arcsec:.2f}", width={slit_width_px}px')
        ax1.legend(fontsize=6, ncol=3, loc='upper right', framealpha=0.6)
        plt.tight_layout()
        plt.show()

        # ── figure 2: per-slit profiles (same colors), average, and fit ────
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        for k in range(n_slits):
            ax2.plot(diag['along_arcsec'], diag['per_slit_profiles'][k], '.',
                      color=colors[k], markersize=3, alpha=0.5,
                      label=f'{theta_deg[k]:.0f} deg')
        ax2.plot(diag['along_arcsec'], diag['averaged_profile'], '-', color='black',
                  linewidth=2.2, zorder=5, label='averaged profile')
        if np.any(np.isfinite(diag['fitted_curve'])):
            ax2.plot(diag['along_arcsec'], diag['fitted_curve'], 'r--', linewidth=1.8,
                      zorder=6, label='Gaussian+Const fit')
        ax2.set_xlabel('Radius [arcsec]')
        ax2.set_ylabel('Flux')
        ax2.set_title('Per-slit profiles, averaged profile, and fit')
        ax2.legend(fontsize=6, ncol=3)
        plt.tight_layout()
        plt.show()

    def _fail():
        if plot:
            _plot_slits_and_profiles()
        result = AveragedSlitProfile(np.nan, np.nan, np.nan, np.nan, np.nan,
                                      n_slits_with_data, False)
        diagnostics = AveragedSlitProfileDiagnostics(**diag) if collect_diagnostics else None
        return result, diagnostics

    valid = np.isfinite(averaged_profile)
    if valid.sum() < 5:
        return _fail()

    x_fit = along_arcsec[valid]
    y_fit = averaged_profile[valid]

    amp_guess = float(np.nanmax(y_fit) - np.nanmedian(y_fit))
    bkg_guess = float(np.nanmedian(y_fit))
    y_above_bkg = np.clip(y_fit - bkg_guess, 0.0, None)
    weight_sum = float(np.sum(y_above_bkg))
    if weight_sum > 0:
        sigma_guess = float(np.sqrt(np.sum(y_above_bkg * x_fit ** 2) / weight_sum))
        sigma_guess = max(sigma_guess, pixel_scale_arcsec)
    else:
        sigma_guess = slit_length_arcsec / 3.0

    gauss = models.Gaussian1D(amplitude=amp_guess, mean=0.0, stddev=sigma_guess)
    gauss.mean.bounds = (-slit_length_arcsec, slit_length_arcsec)
    gauss.stddev.bounds = (pixel_scale_arcsec / 2.0, 50 * slit_length_arcsec)
    const = models.Const1D(amplitude=bkg_guess)
    composite = gauss + const

    # calc_uncertainties=True: the only place in this module a formal
    # parameter-covariance uncertainty is needed — with only ONE fit per
    # source/band now (not 12 independent ones to take a dispersion across),
    # this is how sigma_err_arcsec is obtained.
    fitter = fitting.TRFLSQFitter(calc_uncertainties=True)
    try:
        fitted = fitter(composite, x_fit, y_fit, maxiter=1000)
    except Exception as e:
        logger.debug(f"measure_averaged_slit_profile: fit raised {e}")
        return _fail()

    diag['fitted_curve'] = fitted(along_arcsec)

    fit_info = fitter.fit_info
    status = fit_info.get('status', fit_info.get('ierr', 0))
    if status not in (1, 2, 3, 4):
        return _fail()

    fitted_stddev = abs(fitted.stddev_0.value)
    fitted_mean = fitted.mean_0.value
    stddev_std = getattr(fitted.stddev_0, 'std', None)
    sigma_err = float(stddev_std) if stddev_std is not None and np.isfinite(stddev_std) else np.nan

    if plot:
        _plot_slits_and_profiles()

    result = AveragedSlitProfile(
        sigma_arcsec=fitted_stddev, sigma_err_arcsec=sigma_err, mean_arcsec=fitted_mean,
        amplitude=float(fitted.amplitude_0.value),
        background=float(fitted.amplitude_1.value),
        n_slits_with_data=n_slits_with_data, converged=True,
    )
    diagnostics = AveragedSlitProfileDiagnostics(**diag) if collect_diagnostics else None
    return result, diagnostics


# ─────────────────────────────────────────────────────────────────────────────
# Instrument-PSF slit-profile fit — alternative to the Gaussian1D fit of
# measure_averaged_slit_profile for when the real PSF is known
# (instrument_psf.InstrumentPSF). The model, Gaussian(intrinsic FWHM) (x)
# instrument PSF, is rendered at the source's own sub-pixel center and run
# through the SAME slit extraction as the data, so pixelization, bilinear
# slit sampling and PSF wings are all identical between data and model.
# ─────────────────────────────────────────────────────────────────────────────

InstrumentSlitFit = namedtuple('InstrumentSlitFit', [
    'intrinsic_fwhm_arcsec',  # source FWHM for profile='gaussian' (= 2 R_eff), NaN otherwise
    'r_eff_arcsec',           # fitted half-light radius of the source itself (0 = point source)
    'profile',                # source profile used: 'gaussian', 'eff' or 'king'
    'flux',                   # fitted total flux of the model (model has unit flux)
    'background',             # fitted constant pedestal (0 if fit_background=False)
    'r65_arcsec',             # radius enclosing target_ee of the fitted model
    'target_ee',
    'rms_residual',           # rms of (data - model) over the averaged profile
    'converged',
    'gamma',                  # EFF power index used (fitted when gamma_free); NaN for other profiles
], defaults=(float('nan'),))


# ─────────────────────────────────────────────────────────────────────────────
# Free EFF power index. The EFF profile Sigma ∝ (1 + r²/a²)^(-gamma/2) is a
# Moffat with beta = gamma/2; its core radius a always follows from the fitted
# R_eff. With gamma_free the power index is fitted too, jointly with R_eff:
# a coarse (R_eff x gamma) grid, a bounded 1-D R_eff refinement at the best
# gamma and its two neighbours, a parabola through those three chi2 values for
# gamma, and a last R_eff refinement there. gamma is kept on a 0.05 grid
# (instrument_psf caches one profile shape per gamma value). EFF needs
# gamma > ~1.75 for R_eff to exist within the 10 R_eff truncation.
# ─────────────────────────────────────────────────────────────────────────────
GAMMA_STEP = 0.05
GAMMA_HARD_MIN = 1.8
GAMMA_COARSE_STEP = 0.5


def quantize_gamma(g):
    """gamma on the GAMMA_STEP grid (what is rendered and reported)."""
    return float(np.round(np.round(float(g) / GAMMA_STEP) * GAMMA_STEP, 4))


def fit_reff_gamma(chi2_of, r_grid, gamma_min, gamma_max):
    """
    Joint minimum of chi2_of(r_eff, gamma) over r_grid (r_grid[0] may be 0, a
    point source, where gamma does not matter) and gamma in [gamma_min,
    gamma_max]. Returns (r_eff, gamma, chi2); gamma is NaN when the best
    R_eff is 0 (unconstrained).
    """
    from scipy.optimize import minimize_scalar
    gmin, gmax = quantize_gamma(max(gamma_min, GAMMA_HARD_MIN)), quantize_gamma(gamma_max)
    n_g = max(int(np.round((gmax - gmin) / GAMMA_COARSE_STEP)) + 1, 2)
    gammas = np.unique([quantize_gamma(g) for g in np.linspace(gmin, gmax, n_g)])
    r_grid = np.asarray(r_grid, float)
    chi = np.empty((len(r_grid), len(gammas)))
    for j, g in enumerate(gammas):
        for i, r in enumerate(r_grid):
            if r == 0 and j > 0:
                chi[i, j] = chi[i, 0]          # point source: gamma irrelevant
            else:
                chi[i, j] = chi2_of(r, g)

    done = {}                                  # gamma -> (r, chi2) after R_eff refinement

    def refine(g, i_start=None):
        if g in done:
            return done[g]
        if i_start is None:
            col = np.array([chi2_of(r, g) for r in r_grid])
        else:
            col = chi[:, i_start]
        i = int(np.argmin(col))
        best_r, best_c = float(r_grid[i]), float(col[i])
        lo, hi = r_grid[max(i - 1, 0)], r_grid[min(i + 1, len(r_grid) - 1)]
        if hi > lo:
            res = minimize_scalar(lambda r: chi2_of(r, g), bounds=(lo, hi), method='bounded',
                                  options=dict(xatol=5e-5))
            if res.success and res.fun <= best_c:
                best_r, best_c = float(res.x), float(res.fun)
        done[g] = (best_r, best_c)
        return done[g]

    jbest = int(np.argmin(chi.min(axis=0)))
    trio = [j for j in (jbest - 1, jbest, jbest + 1) if 0 <= j < len(gammas)]
    for j in trio:
        refine(float(gammas[j]), j)
    if len(trio) == 3:
        x = np.array([gammas[j] for j in trio], float)
        y = np.array([done[float(gammas[j])][1] for j in trio])
        a, b, _ = np.polyfit(x, y, 2)
        if a > 0:
            g_star = quantize_gamma(np.clip(-b / (2 * a), x[0], x[2]))
            refine(g_star)
    g_best = min(done, key=lambda g: done[g][1])
    r_best, c_best = done[g_best]
    return r_best, (float(g_best) if r_best > 0 else float('nan')), c_best


def fit_slit_profile_instrument_psf(image, x0, y0, psf_model, target_ee=0.65,
                                    slit_length_arcsec=1.0, n_slits=12, slit_width_px=1,
                                    max_intrinsic_fwhm_arcsec=1.0, fit_background=True,
                                    intrinsic_profile='gaussian', gamma=3.0, king_c=1.5,
                                    max_r_eff_arcsec=None, plot=False,
                                    gamma_free=False, gamma_min=2.0, gamma_max=6.0):
    """
    Fit the averaged slit profile through (x0, y0) with

        flux * [source profile(R_eff) (x) instrument PSF] + background

    where the instrument PSF (psf_model, an instrument_psf.InstrumentPSF, e.g.
    InstrumentPSF.from_hst('wfc3', 'F555W', 0.04)) and the SHAPE of the
    source profile are fixed, and only the source size R_eff (its half-light
    radius), flux and background are free.

    intrinsic_profile : shape of the source before the PSF
        'gaussian' — 2-D Gaussian (FWHM = 2 R_eff); the default
        'eff'      — Elson, Fall & Freeman 1987, Sigma ∝ (1 + r²/a²)^(-gamma/2),
                     truncated at 10 R_eff; gamma=4 is a Plummer profile,
                     young clusters have gamma ~ 2.5-3 (Ryon+2017)
        'king'     — King 1962 with concentration king_c = log10(r_t/r_c)
                     (~1-1.5 for globular clusters)
    gamma, king_c are FIXED shape parameters by default: with real noise
    they are degenerate with R_eff, and they control how much light sits in
    the halo — i.e. the aperture correction — so choose them deliberately.
    gamma_free=True (EFF only) fits gamma jointly with R_eff within
    [gamma_min, gamma_max] (fit_reff_gamma); the result's .gamma is the
    fitted value (NaN if R_eff = 0, where gamma is unconstrained).

    For each trial R_eff the unit-flux model is rendered at the same sub-pixel
    center as the source and sampled with _sample_averaged_slit_profile (the
    very same slits as the data); flux and background then follow from
    linear least squares, so the only non-linear search is 1-D over R_eff
    (0 and 30 log-spaced values up to max_r_eff_arcsec, default
    max_intrinsic_fwhm_arcsec / 2, then a bounded refinement).

    r65_arcsec is the radius enclosing target_ee of the TOTAL light of the
    FITTED MODEL, on the image's own pixel grid (psf_model.r_at_ee), so
    aperture_sum(r65)/target_ee is the total flux of the model. `flux` is
    that same total, obtained directly from the profile fit — the two are an
    internal cross-check. The slits should cover the source's halo
    (slit_length_arcsec >~ 3-5 R_eff for EFF/King).

    Use slit_width_px=1: a wider slit averages off-center lines, lowering the
    profile — harmless here since the model is sampled identically, but 1 px
    keeps it the direct cut through the center.

    Returns InstrumentSlitFit.
    """
    if intrinsic_profile not in ('gaussian', 'eff', 'king'):
        raise ValueError(f"intrinsic_profile must be 'gaussian', 'eff' or 'king', "
                         f"got {intrinsic_profile!r}")
    max_r_eff = max_r_eff_arcsec if max_r_eff_arcsec is not None else max_intrinsic_fwhm_arcsec / 2.0
    gamma_free = bool(gamma_free) and intrinsic_profile == 'eff'
    if intrinsic_profile == 'eff':
        gamma = quantize_gamma(gamma)
    prof_kw = dict(profile=intrinsic_profile, gamma=gamma, c=king_c)

    pix = psf_model.pixel_scale_arcsec
    along, _, _, y_data, _ = _sample_averaged_slit_profile(
        image, x0, y0, pix, slit_length_arcsec, n_slits, slit_width_px)

    half = int(np.ceil(slit_length_arcsec / pix)) + 3
    shape = (2 * half + 1, 2 * half + 1)
    x0_frac = x0 - np.round(x0)
    y0_frac = y0 - np.round(y0)
    xl, yl = half + x0_frac, half + y0_frac

    def model_profile(r_eff, g=None):
        kw = prof_kw if g is None else dict(prof_kw, gamma=g)
        img = psf_model.render(xl, yl, shape, r_eff_arcsec=r_eff, **kw)
        return _sample_averaged_slit_profile(img, xl, yl, pix, slit_length_arcsec,
                                             n_slits, slit_width_px)[3]

    def solve(r_eff, g=None):
        m = model_profile(r_eff, g)
        ok = np.isfinite(m) & np.isfinite(y_data)
        if fit_background:
            A = np.vstack([m[ok], np.ones(ok.sum())]).T
        else:
            A = m[ok][:, None]
        coef, *_ = np.linalg.lstsq(A, y_data[ok], rcond=None)
        resid = y_data[ok] - A @ coef
        flux = float(coef[0])
        bkg = float(coef[1]) if fit_background else 0.0
        return float(np.sum(resid ** 2)), flux, bkg, m, int(ok.sum())

    nan = float('nan')
    if np.isfinite(y_data).sum() < 5:
        return InstrumentSlitFit(nan, nan, intrinsic_profile, nan, nan, nan, target_ee, nan, False)

    # coarse grid in R_eff, then bounded 1-D refinement around the minimum
    grid = np.concatenate([[0.0], np.geomspace(0.0025, max_r_eff, 30)])
    if gamma_free:
        best_r, g_fit, best_chi2 = fit_reff_gamma(lambda r, g: solve(r, g)[0], grid,
                                                  gamma_min, gamma_max)
        if np.isfinite(g_fit):
            gamma = g_fit
            prof_kw = dict(prof_kw, gamma=gamma)
    else:
        chi2 = np.array([solve(r)[0] for r in grid])
        i = int(np.argmin(chi2))
        lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
        best_r, best_chi2 = grid[i], chi2[i]
        if hi > lo:
            from scipy.optimize import minimize_scalar
            res = minimize_scalar(lambda r: solve(r)[0], bounds=(lo, hi), method='bounded',
                                  options=dict(xatol=5e-5))
            if res.success and res.fun <= best_chi2:
                best_r, best_chi2 = float(res.x), float(res.fun)
    gamma_out = (float('nan') if intrinsic_profile != 'eff'
                 or (gamma_free and not best_r > 0) else float(gamma))

    _, flux, bkg, m_best, n_ok = solve(best_r)
    r65 = psf_model.r_at_ee(target_ee=target_ee, x0_frac=x0_frac, y0_frac=y0_frac,
                            r_eff_arcsec=best_r, **prof_kw)
    converged = bool(flux > 0 and best_r < max_r_eff * 0.999)
    fwhm = 2.0 * best_r if intrinsic_profile == 'gaussian' else nan

    if plot:
        import matplotlib.pyplot as plt
        shape_lbl = {'gaussian': 'Gauss', 'eff': f'EFF(gamma={gamma:g})',
                     'king': f'King(c={king_c:g})'}[intrinsic_profile]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(along, y_data, 'k-', lw=2, label='averaged data profile')
        ax.plot(along, flux * m_best + bkg, 'r--', lw=1.8,
                label=f'{psf_model.name} (x) {shape_lbl}, R_eff={best_r:.3f}"')
        ax.axvline(-r65, color='gray', ls=':')
        ax.axvline(r65, color='gray', ls=':', label=f'R{int(round(100 * target_ee))} = {r65:.3f}"')
        ax.set_xlabel('Radius [arcsec]')
        ax.set_ylabel('Flux')
        ax.set_title(f'Instrument-PSF fit: flux={flux:.4g}, bkg={bkg:.3g}')
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.show()

    return InstrumentSlitFit(
        intrinsic_fwhm_arcsec=fwhm, r_eff_arcsec=float(best_r), profile=intrinsic_profile,
        flux=flux, background=bkg, r65_arcsec=r65, target_ee=target_ee,
        rms_residual=float(np.sqrt(best_chi2 / max(n_ok, 1))), converged=converged,
        gamma=gamma_out,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Per-band calibration curve: (extra-convolution FWHM <-> mean measured
# sigma) built from n_models Moffat-convolution model sizes, using the SAME
# measure_averaged_slit_profile function real sources are measured with.
# ee_model is kept for diagnostic/plotting purposes (plot 2) — it is NOT
# used to derive a real source's correction factor any more (see the module
# docstring: corr_factor = 1/target_ee once R65 is derived from the size
# estimate, by construction).
# ─────────────────────────────────────────────────────────────────────────────

CalibrationCurve = namedtuple('CalibrationCurve', [
    'sigma_bar_model',       # ndarray (n_models,), sorted ascending
    'ee_model',              # ndarray (n_models,) — EE at r65_arcsec, diagnostic only
    'extra_conv_fwhm_model', # ndarray (n_models,) — the extra-convolution FWHM
                             #   (arcsec) that produced each model, same order
    'r65_arcsec',            # float — the radius this curve's ee_model was built at
    'fwhm_arcsec',           # float — input base PSF FWHM (provenance)
    'alpha',                 # float — input Moffat shape parameter (provenance)
])


def build_native_calibration_curve(fwhm_arcsec, alpha, pixel_scale_arcsec,
                                    r65_arcsec, n_models=20,
                                    max_extra_conv_fwhm_over_sigma_psf=10.0,
                                    render_oversample=4,
                                    n_slits=12, slit_width_px=2,
                                    slit_length_arcsec=1.0):
    """
    Build one band's calibration curve: for n_models Moffat-convolution
    model sizes (model 0 = the bare PSF, up to
    max_extra_conv_fwhm_over_sigma_psf * sigma_PSF of extra convolution),
    render each model and measure its sigma via measure_averaged_slit_profile
    (the SAME function real sources are measured with, at the SAME
    slit_length_arcsec — this is what makes estimate_extra_conv_fwhm's
    inversion of a real source's measured sigma meaningful).

    ee_model (EE at the fixed r65_arcsec this curve was built at) is also
    computed, for the size-calibration diagnostic plot only — see module
    docstring for why it is not used to derive a real source's flux
    correction any more.

    Returns
    -------
    CalibrationCurve
    """
    sigma_psf_arcsec = fwhm_arcsec / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    max_extra_fwhm = max_extra_conv_fwhm_over_sigma_psf * sigma_psf_arcsec

    extra_conv_fwhms = np.concatenate([
        [0.0],
        np.linspace(max_extra_fwhm / n_models, max_extra_fwhm, n_models - 1),
    ])

    sigma_bar_model = np.full(n_models, np.nan)
    ee_model = np.full(n_models, np.nan)

    size_fwhm = max(20.0, 2.2 * slit_length_arcsec / fwhm_arcsec)

    for i, extra_fwhm in enumerate(extra_conv_fwhms):
        img, render_scale = render_moffat_model_image(
            fwhm_arcsec, alpha, extra_fwhm, pixel_scale_arcsec,
            oversample=render_oversample, size_fwhm=size_fwhm,
        )
        ny, nx = img.shape
        yy, xx = np.mgrid[0:ny, 0:nx]
        xc, yc = nx // 2, ny // 2
        r_px = np.sqrt((xx - xc) ** 2 + (yy - yc) ** 2)
        r65_px = r65_arcsec / render_scale

        if r65_px >= min(nx, ny) / 2:
            logger.warning(
                f"build_native_calibration_curve: model {i} (extra_fwhm="
                f"{extra_fwhm:.4f}\") rendered too small for r65={r65_arcsec:.4f}\" "
                f"— EE for this model may be unreliable."
            )
        if slit_length_arcsec / render_scale >= min(nx, ny) / 2:
            logger.warning(
                f"build_native_calibration_curve: model {i} (extra_fwhm="
                f"{extra_fwhm:.4f}\") rendered too small for "
                f"slit_length_arcsec={slit_length_arcsec:.2f}\" — "
                f"sigma for this model will be truncated/unreliable."
            )

        ee_model[i] = img[r_px <= r65_px].sum()

        slit_result, _ = measure_averaged_slit_profile(
            img, xc, yc, render_scale, slit_length_arcsec=slit_length_arcsec,
            n_slits=n_slits, slit_width_px=slit_width_px,
        )
        sigma_bar_model[i] = slit_result.sigma_arcsec if slit_result.converged else np.nan

    valid = np.isfinite(sigma_bar_model) & np.isfinite(ee_model)
    if valid.sum() < 2:
        raise RuntimeError(
            "build_native_calibration_curve: fewer than 2 valid models — "
            "cannot build a calibration curve."
        )

    order = np.argsort(sigma_bar_model[valid])
    sigma_bar_sorted = sigma_bar_model[valid][order]
    ee_sorted = np.minimum.accumulate(ee_model[valid][order])
    extra_conv_fwhm_sorted = extra_conv_fwhms[valid][order]

    return CalibrationCurve(sigma_bar_sorted, ee_sorted, extra_conv_fwhm_sorted,
                             r65_arcsec, fwhm_arcsec, alpha)


def ee_from_calibration_curve(calib_curve, sigma_bar):
    """
    Look up the encircled-energy fraction at `sigma_bar`, via monotonic
    Pchip interpolation over the calibration curve — diagnostic/plotting use
    only (see build_native_calibration_curve's docstring); not used to
    derive a real source's flux correction.

    Returns
    -------
    (ee: float, extrapolated: bool)
    """
    sb = calib_curve.sigma_bar_model
    ee = calib_curve.ee_model

    if not np.isfinite(sigma_bar):
        return np.nan, True

    extrapolated = sigma_bar < sb[0] or sigma_bar > sb[-1]
    sigma_bar_clipped = float(np.clip(sigma_bar, sb[0], sb[-1]))

    interp = PchipInterpolator(sb, ee, extrapolate=False)
    ee_value = float(interp(sigma_bar_clipped))

    return ee_value, extrapolated


def estimate_extra_conv_fwhm(calib_curve, sigma_bar):
    """
    Invert a measured sigma back to an ESTIMATE of the source's own
    intrinsic extra-convolution FWHM (how much bigger than the bare PSF
    this source appears) — fed into compute_r65(..., extra_conv_fwhm_arcsec
    =<this estimate>) to get this source's own R65.

    Same monotonic-Pchip-with-clip-and-flag approach as
    ee_from_calibration_curve, interpolating extra_conv_fwhm_model instead.

    Returns
    -------
    (extra_conv_fwhm_arcsec: float, extrapolated: bool)
    """
    sb = calib_curve.sigma_bar_model
    fwhm = calib_curve.extra_conv_fwhm_model

    if not np.isfinite(sigma_bar):
        return np.nan, True

    extrapolated = sigma_bar < sb[0] or sigma_bar > sb[-1]
    sigma_bar_clipped = float(np.clip(sigma_bar, sb[0], sb[-1]))

    interp = PchipInterpolator(sb, fwhm, extrapolate=False)
    fwhm_value = float(interp(sigma_bar_clipped))

    return fwhm_value, extrapolated


# ─────────────────────────────────────────────────────────────────────────────
# Aperture-borrowing reference-band groups — hardcoded per the pipeline
# author's own specification (not auto-computed from PSF FWHM): all HST
# bands are similar to each other, with F555W preferred first as the
# reference; F200W stands alone (never borrows, never lent); the three
# ~3-micron NIRCam medium bands are mutually similar; every MIRI band is
# different from every other MIRI band (each is its own singleton group —
# simply absent from this structure, so it never has borrowing candidates).
# ─────────────────────────────────────────────────────────────────────────────

REFERENCE_BAND_GROUPS = {
    'hst_optical': {
        'bands': ('f275w', 'f336w', 'f438w', 'f555w', 'f814w'),
        'priority': ('f555w',),
    },
    'nircam_short': {
        'bands': ('f200w',),
        'priority': (),
    },
    'nircam_medium': {
        'bands': ('f300m', 'f335m', 'f360m'),
        # F300M is consistently the best-behaved of the three (least prone
        # to size_extrapolated — see measure_native_photometry_one_source's
        # own-extrapolated-size fallback), so prefer borrowing its aperture
        # over F335M/F360M lending to each other.
        'priority': ('f300m',),
    },
    # MIRI bands (f770w, f1000w, f1130w, f2100w, ...) deliberately absent —
    # each is its own singleton group, never borrows from another MIRI band.
}


def get_reference_candidates(band_lower, all_bands_lower, wavelength_by_band=None,
                              groups=REFERENCE_BAND_GROUPS):
    """
    Ordered list of candidate reference bands for `band_lower` (excluding
    itself and restricted to bands actually present in `all_bands_lower`):
    the group's own declared priority band(s) first, then the rest of that
    same group ordered by wavelength proximity to `band_lower` (nearest
    first) if `wavelength_by_band` is given. A band not listed in any group
    (every MIRI band, by design) returns [] — no borrowing candidates at
    all, so it can only ever resolve via the PSF-fallback path.
    """
    for spec in groups.values():
        group_bands = spec['bands']
        if band_lower not in group_bands:
            continue
        priority = [b for b in spec.get('priority', ()) if b != band_lower and b in group_bands]
        rest = [b for b in group_bands if b != band_lower and b not in priority]
        if wavelength_by_band:
            target_wl = wavelength_by_band.get(band_lower)
            if target_wl is not None:
                rest = sorted(rest, key=lambda b: abs(wavelength_by_band.get(b, target_wl) - target_wl))
        candidates = priority + rest
        return [c for c in candidates if c in all_bands_lower]
    return []


# ─────────────────────────────────────────────────────────────────────────────
# Per-source measurement
# ─────────────────────────────────────────────────────────────────────────────

NativePhotResult = namedtuple('NativePhotResult', [
    'flux_mJy', 'flux_raw_mJy', 'flux_err_mJy',
    'corr_factor',              # exactly 1/target_ee whenever a valid aperture
                                # was used (see module docstring) — NaN otherwise
    'emission_detected', 'peak_snr', 'aperture_snr',   # from detect_emission
    'sigma_arcsec',             # the measured size (arcsec); NaN unless
                                # aperture_source=='own'
    'r65_arcsec',               # this source's own aperture radius actually used
    'r65_psf_arcsec',           # this band's fixed PSF-based radius, for comparison
    'aperture_source',          # 'own' | 'borrowed:{band}' | 'psf_fallback' | 'none'
    'extra_conv_fwhm_estimate_arcsec',
    'centroid_ra', 'centroid_dec',   # sky position actually used for the aperture
    'recenter_shift_arcsec', 'recenter_flagged',
    'bkg_median_per_pix', 'bkg_std', 'snr',
    'size_extrapolated',        # True if the size estimate fell outside the
                                # band-level calibration curve's own sigma range
    'size_capped',               # True if the derived r65_arcsec was clamped
                                # down to max_r65_growth_factor * r65_psf_arcsec
                                # (a corrupted/outlier fit blowing R65 up far
                                # beyond what the other bands agree on) —
                                # only ever True when aperture_source=='own'
    'status',                   # 'ok' | 'size_extrapolated' | 'no_detection' |
                                # 'borrowed' | 'psf_fallback' | 'no_flux' |
                                # 'exception:{e}'
])


def _empty_result(status, r65_arcsec=float('nan'), r65_psf_arcsec=float('nan'),
                   aperture_source='none'):
    nan = float('nan')
    return NativePhotResult(
        flux_mJy=nan, flux_raw_mJy=nan, flux_err_mJy=nan, corr_factor=nan,
        emission_detected=False, peak_snr=nan, aperture_snr=nan,
        sigma_arcsec=nan, r65_arcsec=r65_arcsec, r65_psf_arcsec=r65_psf_arcsec,
        aperture_source=aperture_source, extra_conv_fwhm_estimate_arcsec=nan,
        centroid_ra=nan, centroid_dec=nan,
        recenter_shift_arcsec=nan, recenter_flagged=False,
        bkg_median_per_pix=nan, bkg_std=nan, snr=nan,
        size_extrapolated=False, size_capped=False, status=status,
    )


def _aperture_photometry(center_sky, r65_arcsec, image, header, wcs, pixel_scale_arcsec,
                          annulus_r_in_factor, annulus_r_out_factor, unit_convert):
    """
    Plain background-subtracted circular-aperture photometry at a GIVEN
    center and radius — no recentering, no slit fit. Shared by the 'own',
    'borrowed', and 'psf_fallback' aperture-source paths, which differ only
    in how center_sky/r65_arcsec were arrived at, never in how the flux
    itself is measured once those are known.

    Returns None if the aperture has no flux at all, else a dict of
    flux_raw_mJy, flux_bsub_mJy, flux_raw_err_mJy, bkg_median_per_pix,
    bkg_std, n_source_pix.
    """
    r_in_arcsec = annulus_r_in_factor * r65_arcsec
    r_out_arcsec = annulus_r_out_factor * r65_arcsec
    source_region = CircleSkyRegion(center=center_sky, radius=r65_arcsec * u.arcsec)
    inner_region = CircleSkyRegion(center=center_sky, radius=r_in_arcsec * u.arcsec)
    outer_region = CircleSkyRegion(center=center_sky, radius=r_out_arcsec * u.arcsec)

    # img_local must come from the OUTER region's own call — see the
    # identical comment in detect_emission for why (source_region's call
    # would silently zero out the whole annulus).
    ref_bbox = outer_region.to_pixel(wcs).to_mask().bbox
    img_local, mask_outer = _bbox_local_mask_and_image(outer_region, wcs, image, ref_bbox)
    _, mask_source = _bbox_local_mask_and_image(source_region, wcs, image, ref_bbox)
    _, mask_inner = _bbox_local_mask_and_image(inner_region, wcs, image, ref_bbox)

    if mask_source.astype(bool).sum() == 0:
        return None

    annulus_mask = mask_outer.astype(bool) & ~mask_inner.astype(bool)
    ann_values = np.where(annulus_mask, img_local, np.nan)
    _, bkg_median_per_pix, bkg_std = sigma_clipped_stats(ann_values, sigma=3.0)
    n_source_pix = int(mask_source.astype(bool).sum())

    raw_sum = float(np.nansum(img_local * mask_source))
    bsub_sum = raw_sum - bkg_median_per_pix * n_source_pix

    convert = unit_convert if unit_convert is not None else (lambda v, hdr: v)
    flux_raw_mJy = (convert(raw_sum, header) * u.Jy).to(u.mJy).value
    flux_bsub_mJy = (convert(bsub_sum, header) * u.Jy).to(u.mJy).value
    flux_raw_err_perpix = bkg_std * np.sqrt(n_source_pix)
    flux_raw_err_mJy = (convert(flux_raw_err_perpix, header) * u.Jy).to(u.mJy).value

    return dict(flux_raw_mJy=flux_raw_mJy, flux_bsub_mJy=flux_bsub_mJy,
                flux_raw_err_mJy=flux_raw_err_mJy,
                bkg_median_per_pix=float(bkg_median_per_pix), bkg_std=float(bkg_std),
                n_source_pix=n_source_pix)


def _measure_own_band(center_sky, image, header, wcs, pixel_scale_arcsec,
                       calib_curve, r65_psf_arcsec,
                       recenter_flag_frac, annulus_r_in_factor, annulus_r_out_factor,
                       n_slits, slit_width_px, slit_length_arcsec,
                       target_ee, snr_threshold, unit_convert,
                       max_r65_growth_factor=2.0):
    """
    Full own-band measurement: 2-D Gaussian centering, one averaged-slit
    sigma measurement at the fixed slit_length_arcsec domain, a cheap
    re-check with a WIDENED domain if the derived R65 implies a source
    bigger than that default window could see, a safety cap on the
    resulting R65 (see below), and background-subtracted aperture
    photometry at the resulting R65. Returns a plain dict (not yet
    dereddened or wrapped in NativePhotResult — the caller,
    measure_native_photometry_one_source, does that uniformly for all three
    aperture-source cases).

    max_r65_growth_factor : clamps r65_source to at most this multiple of
        r65_psf_arcsec — a real-data incident (4/5 HST filters agreeing on
        R65~0.05", one landing at 0.2" — 4x bigger — from a corrupted
        search-pass fit) showed the widened re-check above is not itself
        enough protection against a single outlier fit producing a
        wildly-oversized aperture. size_extrapolated (unbounded, just a
        flag) and size_capped (this clamp) are deliberately independent:
        a source can be extrapolated without being capped (mildly past the
        calibration ceiling, still a reasonable radius) or capped without
        being (rare, would need extra_conv_fwhm_est inside the calibration
        range yet still imply r65 > the cap). The corr_factor is left at
        1/target_ee unchanged even when capped — this deliberately
        under-corrects a genuinely huge source rather than trusting an
        outlier fit's full (possibly wrong) size estimate, the same
        accepted trade-off already made by the widened re-check itself.
    """
    x0_cat, y0_cat = center_sky.to_pixel(wcs)

    fit_half_size_arcsec = max(3.0 * r65_psf_arcsec, 5.0 * slit_length_arcsec)
    crop_radius_arcsec = fit_half_size_arcsec * 1.3
    crop_region = CircleSkyRegion(center=center_sky, radius=crop_radius_arcsec * u.arcsec)
    ref_bbox = crop_region.to_pixel(wcs).to_mask().bbox
    img_local, mask_crop = _bbox_local_mask_and_image(crop_region, wcs, image, ref_bbox)
    if mask_crop.astype(bool).sum() == 0:
        return dict(status='no_flux')

    x0_local = x0_cat - ref_bbox.ixmin
    y0_local = y0_cat - ref_bbox.iymin

    fit_half_size_px = max(3.0 * r65_psf_arcsec / pixel_scale_arcsec, 5.0)
    center_fit = fit_2d_gaussian_center(img_local, x0_local, y0_local,
                                         fit_half_size_px, pixel_scale_arcsec)

    if center_fit.converged:
        x0_use, y0_use = center_fit.x0_local, center_fit.y0_local
    else:
        x0_use, y0_use = x0_local, y0_local

    shift_px = float(np.hypot(x0_use - x0_local, y0_use - y0_local))
    recenter_shift_arcsec = shift_px * pixel_scale_arcsec
    recenter_flagged = (not center_fit.converged) or (
        recenter_shift_arcsec > recenter_flag_frac * r65_psf_arcsec)

    x0_full = x0_use + ref_bbox.ixmin
    y0_full = y0_use + ref_bbox.iymin
    center_sky_fit = wcs.pixel_to_world(x0_full, y0_full)

    search, _ = measure_averaged_slit_profile(
        img_local, x0_use, y0_use, pixel_scale_arcsec,
        slit_length_arcsec=slit_length_arcsec, n_slits=n_slits, slit_width_px=slit_width_px,
    )
    if not search.converged:
        return dict(status='no_size_measurement',
                     recenter_shift_arcsec=recenter_shift_arcsec, recenter_flagged=recenter_flagged,
                     centroid_ra=float(center_sky_fit.ra.deg), centroid_dec=float(center_sky_fit.dec.deg))

    extra_conv_est, size_extrapolated = estimate_extra_conv_fwhm(calib_curve, search.sigma_arcsec)
    if not np.isfinite(extra_conv_est):
        extra_conv_est, size_extrapolated = 0.0, True

    r65_source = compute_r65(calib_curve.fwhm_arcsec, calib_curve.alpha, pixel_scale_arcsec,
                              target_ee=target_ee, extra_conv_fwhm_arcsec=max(extra_conv_est, 0.0))
    sigma_final = search.sigma_arcsec

    # Cheap re-check: if the derived R65 implies a genuinely bigger source
    # than the small fixed slit_length_arcsec window could see, re-measure
    # sigma with a window matching the just-derived R65 — catches a
    # first-pass estimate that was constrained by too small a window
    # rather than the source's real size — and refine from that.
    if r65_source > slit_length_arcsec:
        recheck, _ = measure_averaged_slit_profile(
            img_local, x0_use, y0_use, pixel_scale_arcsec,
            slit_length_arcsec=r65_source, n_slits=n_slits, slit_width_px=slit_width_px,
        )
        if recheck.converged:
            extra_conv_est2, size_extrapolated2 = estimate_extra_conv_fwhm(
                calib_curve, recheck.sigma_arcsec)
            if np.isfinite(extra_conv_est2):
                sigma_final = recheck.sigma_arcsec
                extra_conv_est, size_extrapolated = extra_conv_est2, size_extrapolated2
                r65_source = compute_r65(calib_curve.fwhm_arcsec, calib_curve.alpha,
                                          pixel_scale_arcsec, target_ee=target_ee,
                                          extra_conv_fwhm_arcsec=max(extra_conv_est, 0.0))

    r65_cap = max_r65_growth_factor * r65_psf_arcsec
    size_capped = r65_source > r65_cap
    if size_capped:
        r65_source = r65_cap

    phot = _aperture_photometry(center_sky_fit, r65_source, image, header, wcs,
                                 pixel_scale_arcsec, annulus_r_in_factor,
                                 annulus_r_out_factor, unit_convert)
    if phot is None:
        return dict(status='no_flux', r65_arcsec=r65_source,
                     recenter_shift_arcsec=recenter_shift_arcsec, recenter_flagged=recenter_flagged,
                     centroid_ra=float(center_sky_fit.ra.deg), centroid_dec=float(center_sky_fit.dec.deg))

    corr_factor = 1.0 / target_ee
    flux_bsub_raw_mJy = phot['flux_bsub_mJy'] * corr_factor
    flux_err_mJy = abs(phot['flux_raw_err_mJy'] * corr_factor)

    snr = (phot['flux_bsub_mJy'] / phot['flux_raw_err_mJy']
           if phot['flux_raw_err_mJy'] not in (0, None) and np.isfinite(phot['flux_raw_err_mJy'])
           else float('nan'))

    status = 'size_extrapolated' if size_extrapolated else 'ok'
    if not np.isfinite(snr) or snr < snr_threshold:
        status = 'no_detection'

    return dict(
        status=status, center_sky=center_sky_fit, r65_arcsec=r65_source,
        sigma_arcsec=sigma_final, extra_conv_fwhm_estimate_arcsec=extra_conv_est,
        size_extrapolated=size_extrapolated, size_capped=size_capped,
        flux_raw_mJy=phot['flux_raw_mJy'], flux_bsub_raw_mJy=flux_bsub_raw_mJy,
        flux_err_mJy=flux_err_mJy, corr_factor=corr_factor,
        bkg_median_per_pix=phot['bkg_median_per_pix'], bkg_std=phot['bkg_std'], snr=snr,
        recenter_shift_arcsec=recenter_shift_arcsec, recenter_flagged=recenter_flagged,
        centroid_ra=float(center_sky_fit.ra.deg), centroid_dec=float(center_sky_fit.dec.deg),
    )


def _assemble_result(own, aperture_source, r65_psf_arcsec, r_value, eb_v,
                      emission_detected, peak_snr, aperture_snr):
    """Apply Milky Way dereddening (phot_aux_func.corr_flux, reused not
    reimplemented) and wrap an internal measurement dict into the public
    NativePhotResult, uniformly for the 'own'/'borrowed'/'psf_fallback'
    paths."""
    status = own.get('status', 'exception')
    if status not in ('ok', 'size_extrapolated', 'borrowed', 'psf_fallback', 'no_detection'):
        return _empty_result(status, r65_psf_arcsec=r65_psf_arcsec, aperture_source=aperture_source)

    flux_raw_mJy = own.get('flux_raw_mJy', np.nan)
    flux_bsub_raw_mJy = own.get('flux_bsub_raw_mJy', np.nan)
    flux_err_mJy = own.get('flux_err_mJy', np.nan)

    if status == 'no_detection' or not np.isfinite(flux_bsub_raw_mJy):
        flux_mJy = np.nan
    elif r_value is not None:
        flux_mJy = corr_flux(flux_bsub_raw_mJy, r_value, eb_v)
        flux_err_mJy = corr_flux(flux_err_mJy, r_value, eb_v)
    else:
        flux_mJy = flux_bsub_raw_mJy

    return NativePhotResult(
        flux_mJy=flux_mJy, flux_raw_mJy=flux_raw_mJy, flux_err_mJy=flux_err_mJy,
        corr_factor=own.get('corr_factor', np.nan),
        emission_detected=emission_detected, peak_snr=peak_snr, aperture_snr=aperture_snr,
        sigma_arcsec=own.get('sigma_arcsec', np.nan),
        r65_arcsec=own.get('r65_arcsec', np.nan), r65_psf_arcsec=r65_psf_arcsec,
        aperture_source=aperture_source,
        extra_conv_fwhm_estimate_arcsec=own.get('extra_conv_fwhm_estimate_arcsec', np.nan),
        centroid_ra=own.get('centroid_ra', np.nan), centroid_dec=own.get('centroid_dec', np.nan),
        recenter_shift_arcsec=own.get('recenter_shift_arcsec', np.nan),
        recenter_flagged=own.get('recenter_flagged', False),
        bkg_median_per_pix=own.get('bkg_median_per_pix', np.nan), bkg_std=own.get('bkg_std', np.nan),
        snr=own.get('snr', np.nan),
        size_extrapolated=own.get('size_extrapolated', False),
        size_capped=own.get('size_capped', False),
        status=status,
    )


def measure_native_photometry_one_source(ra, dec, band_data,
                                          reference_groups=REFERENCE_BAND_GROUPS,
                                          eb_v=0.0, target_ee=0.65,
                                          recenter_flag_frac=0.5,
                                          annulus_r_in_factor=2.6, annulus_r_out_factor=4.0,
                                          n_slits=12, slit_width_px=2, slit_length_arcsec=1.0,
                                          snr_threshold=3.0,
                                          peak_snr_threshold=5.0, aperture_snr_threshold=3.0,
                                          max_r65_growth_factor=2.0):
    """
    Full native photometry for ONE source, across ALL bands in band_data —
    the per-source orchestrator that replaces the old fully-band-independent
    per-(source,band) design. Bands are no longer independent: round 1 tests
    every band for real emission and, where found, measures it
    independently (aperture_source='own'); round 2 resolves every remaining
    band by borrowing a similar-PSF band's own aperture (position + R65,
    reused as-is/unscaled) if this source has one, or falling back to this
    band's own PSF-based aperture at the catalog position otherwise (see
    REFERENCE_BAND_GROUPS / get_reference_candidates). Never raises — one
    bad band is recorded with its own status, not a whole-source failure.

    Parameters
    ----------
    ra, dec   : float, deg — catalog source position
    band_data : dict  band_lower -> dict with (at minimum): image, header,
                wcs, pixel_scale_arcsec, calib_curve (CalibrationCurve,
                band-level, built at r65_psf_arcsec), r65_psf_arcsec,
                r_value, unit_convert, central_wl_AA — exactly the per-band
                setup dict Photometry.run_native_photometry builds once per
                band (see that method).
    max_r65_growth_factor : float — caps an 'own'-band r65_arcsec to at most
                this multiple of the band's own r65_psf_arcsec; see
                _measure_own_band's docstring for the real-data incident
                this guards against.

    Returns
    -------
    dict  band_lower -> NativePhotResult
    """
    center_sky = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    all_bands = list(band_data.keys())
    wavelength_by_band = {b: band_data[b].get('central_wl_AA') for b in all_bands}

    own_results = {}

    # ── round 1: emission test + own-band measurement, independent per band ──
    for band_lower, bd in band_data.items():
        try:
            emission = detect_emission(
                bd['image'], bd['header'], bd['wcs'], center_sky, bd['r65_psf_arcsec'],
                annulus_r_in_factor=annulus_r_in_factor, annulus_r_out_factor=annulus_r_out_factor,
                peak_snr_threshold=peak_snr_threshold, aperture_snr_threshold=aperture_snr_threshold,
            )
        except Exception as e:
            logger.debug(f"detect_emission({band_lower}) failed: {e}")
            own_results[band_lower] = None
            continue

        if not emission.has_emission:
            own_results[band_lower] = None
            continue

        try:
            own = _measure_own_band(
                center_sky, bd['image'], bd['header'], bd['wcs'], bd['pixel_scale_arcsec'],
                bd['calib_curve'], bd['r65_psf_arcsec'],
                recenter_flag_frac, annulus_r_in_factor, annulus_r_out_factor,
                n_slits, slit_width_px, slit_length_arcsec, target_ee, snr_threshold,
                bd.get('unit_convert'), max_r65_growth_factor=max_r65_growth_factor,
            )
        except Exception as e:
            logger.debug(f"_measure_own_band({band_lower}) failed: {e}")
            own = dict(status=f'exception:{e}')

        own['emission_detected'] = True
        own['peak_snr'] = emission.peak_snr
        own['aperture_snr'] = emission.aperture_snr
        own_results[band_lower] = own

    # ── round 2: resolve every band that didn't get a usable own measurement ──
    results = {}
    for band_lower, bd in band_data.items():
        own = own_results.get(band_lower)
        own_status = own.get('status') if own is not None else None

        if own_status in ('ok', 'no_detection'):
            results[band_lower] = _assemble_result(
                own, aperture_source='own', r65_psf_arcsec=bd['r65_psf_arcsec'],
                r_value=bd.get('r_value'), eb_v=eb_v,
                emission_detected=own.get('emission_detected', False),
                peak_snr=own.get('peak_snr', np.nan), aperture_snr=own.get('aperture_snr', np.nan),
            )
            continue

        emission_detected = own is not None
        peak_snr = own.get('peak_snr', np.nan) if own else np.nan
        aperture_snr = own.get('aperture_snr', np.nan) if own else np.nan

        if own_status == 'size_extrapolated':
            # A size_extrapolated own measurement means this band's own
            # sigma fell outside its calibration curve's range — less
            # trustworthy than reusing a similar-PSF sibling band's own
            # IN-RANGE ('ok') aperture, when one is available for this same
            # source (e.g. F300M lending to F335M/F360M). Only fall through
            # to keeping the own extrapolated measurement if no clean
            # sibling exists — never demote it to a borrowed *extrapolated*
            # sibling (no improvement) or to psf_fallback (a regression).
            candidates = get_reference_candidates(band_lower, all_bands, wavelength_by_band,
                                                   reference_groups)
            has_clean_ref = any(
                own_results.get(c) is not None and own_results[c].get('status') == 'ok'
                for c in candidates
            )
            if has_clean_ref:
                try:
                    results[band_lower] = _resolve_band(
                        band_lower, bd, center_sky, own_results, all_bands, wavelength_by_band,
                        reference_groups, annulus_r_in_factor, annulus_r_out_factor,
                        target_ee, snr_threshold, eb_v, emission_detected, peak_snr, aperture_snr,
                        require_clean=True,
                    )
                except Exception as e:
                    logger.debug(f"_resolve_band({band_lower}, require_clean) failed: {e}")
                    results[band_lower] = _assemble_result(
                        own, aperture_source='own', r65_psf_arcsec=bd['r65_psf_arcsec'],
                        r_value=bd.get('r_value'), eb_v=eb_v,
                        emission_detected=own.get('emission_detected', False),
                        peak_snr=own.get('peak_snr', np.nan), aperture_snr=own.get('aperture_snr', np.nan),
                    )
            else:
                results[band_lower] = _assemble_result(
                    own, aperture_source='own', r65_psf_arcsec=bd['r65_psf_arcsec'],
                    r_value=bd.get('r_value'), eb_v=eb_v,
                    emission_detected=own.get('emission_detected', False),
                    peak_snr=own.get('peak_snr', np.nan), aperture_snr=own.get('aperture_snr', np.nan),
                )
            continue

        try:
            results[band_lower] = _resolve_band(
                band_lower, bd, center_sky, own_results, all_bands, wavelength_by_band,
                reference_groups, annulus_r_in_factor, annulus_r_out_factor,
                target_ee, snr_threshold, eb_v, emission_detected, peak_snr, aperture_snr,
            )
        except Exception as e:
            logger.debug(f"_resolve_band({band_lower}) failed: {e}")
            results[band_lower] = _empty_result(f'exception:{e}',
                                                  r65_psf_arcsec=bd['r65_psf_arcsec'])

    return results


def _resolve_band(band_lower, bd, center_sky, own_results, all_bands, wavelength_by_band,
                   reference_groups, annulus_r_in_factor, annulus_r_out_factor,
                   target_ee, snr_threshold, eb_v, emission_detected, peak_snr, aperture_snr,
                   require_clean=False):
    """
    Round-2 resolution for one band with no usable own-band measurement:
    borrow a reference band's own aperture (case a) or fall back to this
    band's own PSF-based aperture (case b/c) — see
    measure_native_photometry_one_source's docstring.

    require_clean : bool — when True, only an 'ok' (non-extrapolated)
        sibling qualifies as a lender. Used when this band's OWN measurement
        was 'size_extrapolated' and the caller has already confirmed a clean
        sibling exists — borrowing another extrapolated sibling would be no
        improvement over keeping the own estimate.
    """
    accepted_statuses = ('ok',) if require_clean else ('ok', 'size_extrapolated')
    candidates = get_reference_candidates(band_lower, all_bands, wavelength_by_band, reference_groups)
    ref_result, ref_band = None, None
    for cand in candidates:
        cand_own = own_results.get(cand)
        if cand_own is not None and cand_own.get('status') in accepted_statuses:
            ref_result, ref_band = cand_own, cand
            break

    if ref_result is not None:
        aperture_center = ref_result['center_sky']
        aperture_r65 = ref_result['r65_arcsec']
        aperture_source = f'borrowed:{ref_band}'
        default_status = 'borrowed'
    else:
        aperture_center = center_sky
        aperture_r65 = bd['r65_psf_arcsec']
        aperture_source = 'psf_fallback'
        default_status = 'psf_fallback'

    phot = _aperture_photometry(
        aperture_center, aperture_r65, bd['image'], bd['header'], bd['wcs'],
        bd['pixel_scale_arcsec'], annulus_r_in_factor, annulus_r_out_factor, bd.get('unit_convert'),
    )
    if phot is None:
        return _empty_result('no_flux', r65_psf_arcsec=bd['r65_psf_arcsec'], aperture_source=aperture_source)

    corr_factor = 1.0 / target_ee
    flux_bsub_raw_mJy = phot['flux_bsub_mJy'] * corr_factor
    flux_err_mJy = abs(phot['flux_raw_err_mJy'] * corr_factor)
    snr = (phot['flux_bsub_mJy'] / phot['flux_raw_err_mJy']
           if phot['flux_raw_err_mJy'] not in (0, None) and np.isfinite(phot['flux_raw_err_mJy'])
           else float('nan'))

    status = default_status
    if np.isfinite(snr) and snr < snr_threshold:
        status = 'no_detection'
    elif not np.isfinite(snr) and aperture_source == 'psf_fallback':
        # psf_fallback has no independent recentering/size — an
        # indeterminate SNR there is expected (near-zero flux, near-zero
        # error) and should not itself be treated as evidence of detection.
        status = 'no_detection'

    own_like = dict(
        status=status, center_sky=aperture_center, r65_arcsec=aperture_r65,
        sigma_arcsec=np.nan, extra_conv_fwhm_estimate_arcsec=np.nan,
        size_extrapolated=False, size_capped=False,
        flux_raw_mJy=phot['flux_raw_mJy'], flux_bsub_raw_mJy=flux_bsub_raw_mJy,
        flux_err_mJy=flux_err_mJy, corr_factor=corr_factor,
        bkg_median_per_pix=phot['bkg_median_per_pix'], bkg_std=phot['bkg_std'], snr=snr,
        recenter_shift_arcsec=np.nan, recenter_flagged=False,
        centroid_ra=float(aperture_center.ra.deg) if aperture_source != 'psf_fallback' else np.nan,
        centroid_dec=float(aperture_center.dec.deg) if aperture_source != 'psf_fallback' else np.nan,
    )
    return _assemble_result(
        own_like, aperture_source=aperture_source, r65_psf_arcsec=bd['r65_psf_arcsec'],
        r_value=bd.get('r_value'), eb_v=eb_v,
        emission_detected=emission_detected, peak_snr=peak_snr, aperture_snr=aperture_snr,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Instrument-PSF photometry — ADDITIONAL to the Gaussian-slit / Moffat-
# calibration method above, run on top of its results (it reuses that
# method's own-band centroids and aperture_source decisions). Two models per
# band: 'instG' = PSF (x) Gaussian and 'instP' = PSF (x) the cluster profile
# chosen from the cluster class (CLUSTER_PROFILE_BY_CLASS). Fitting the
# averaged slit profile with the real instrument PSF gives the aperture that
# truly encloses target_ee of the light; the Gaussian slit fit under-sizes it
# (see instrument_psf.py and validation/test_instrument_fit.py).
# ─────────────────────────────────────────────────────────────────────────────

# Source profile per PHANGS_CC_CLASS_ML. Young and intermediate-age clusters
# follow EFF profiles (Elson, Fall & Freeman 1987; gamma ~ 2.5-3 for LEGUS
# clusters, Ryon+2017), old globular clusters King profiles (c ~ 1.5 is the
# Milky Way median). gamma/king_c are fixed, not fitted (see
# fit_slit_profile_instrument_psf).
CLUSTER_PROFILE_BY_CLASS = {
    'YCL': dict(profile='eff', gamma=3.0),
    'MAP': dict(profile='eff', gamma=3.0),
    'OGCC': dict(profile='king', king_c=1.5),
}
DEFAULT_CLUSTER_PROFILE = dict(profile='eff', gamma=3.0)

INSTRUMENT_METHODS = ('instG', 'instP')


def profile_spec_for_class(cc_class, profile_by_class=CLUSTER_PROFILE_BY_CLASS):
    """Full profile spec (profile, gamma, king_c, gamma_free, gamma_min,
    gamma_max) for a cluster class. Classes not in profile_by_class use its
    'default' entry if it has one, else EFF gamma=3."""
    if isinstance(cc_class, bytes):
        cc_class = cc_class.decode()
    default = profile_by_class.get('default', DEFAULT_CLUSTER_PROFILE)
    spec = dict(profile_by_class.get(str(cc_class).strip(), default))
    spec.setdefault('gamma', 3.0)
    spec.setdefault('king_c', 1.5)
    spec.setdefault('gamma_free', False)
    spec.setdefault('gamma_min', 2.0)
    spec.setdefault('gamma_max', 6.0)
    return spec


def profile_tag(spec):
    """Short label of a profile spec: 'gauss', 'eff3.0', 'king1.5', and
    'eff2.0-6.0' for EFF with gamma fitted in [2.0, 6.0]."""
    if spec['profile'] == 'gaussian':
        return 'gauss'
    if spec['profile'] == 'eff':
        if spec.get('gamma_free'):
            return f"eff{spec.get('gamma_min', 2.0):.1f}-{spec.get('gamma_max', 6.0):.1f}"
        return f"eff{spec['gamma']:.1f}" if round(spec['gamma'], 1) == spec['gamma'] \
            else f"eff{spec['gamma']:.2f}"
    return f"king{spec['king_c']:.1f}"


def profile_spec_from_tag(tag, gamma=None):
    """Inverse of profile_tag. gamma (e.g. a stored gamma_instP_<band>) replaces
    the tag's gamma when finite, and the spec is then a fixed-gamma one — what
    is needed to redraw a fitted model."""
    if isinstance(tag, bytes):
        tag = tag.decode()
    tag = str(tag).strip()
    if tag.startswith('king'):
        return dict(profile='king', gamma=3.0, king_c=float(tag[4:]), gamma_free=False)
    if tag.startswith('gauss'):
        return dict(profile='gaussian', gamma=3.0, king_c=1.5, gamma_free=False)
    body = tag[3:] if tag.startswith('eff') else '3.0'
    if '-' in body:
        lo, hi = (float(v) for v in body.split('-'))
        spec = dict(profile='eff', gamma=0.5 * (lo + hi), king_c=1.5, gamma_free=True,
                    gamma_min=lo, gamma_max=hi)
    else:
        spec = dict(profile='eff', gamma=float(body), king_c=1.5, gamma_free=False)
    if gamma is not None and np.isfinite(gamma):
        spec.update(gamma=float(gamma), gamma_free=False)
    return spec


def profile_by_class_from_settings(eff_gamma=3.0, eff_gamma_free=False, eff_gamma_min=2.0,
                                   eff_gamma_max=6.0, ogcc_profile='king', king_c=1.5):
    """profile_by_class for run_native_photometry from the [profile] config keys:
    YCL, MAP and unclassified clusters get the EFF profile (gamma fixed at
    eff_gamma, or fitted in [eff_gamma_min, eff_gamma_max]); OGCC get King
    (concentration king_c) or, with ogcc_profile='eff', the same EFF."""
    if eff_gamma_free and not (GAMMA_HARD_MIN <= eff_gamma_min < eff_gamma_max):
        raise ValueError(f'need {GAMMA_HARD_MIN} <= eff_gamma_min < eff_gamma_max '
                         f'(got {eff_gamma_min}, {eff_gamma_max})')
    if not eff_gamma_free and eff_gamma < GAMMA_HARD_MIN:
        raise ValueError(f'eff_gamma must be >= {GAMMA_HARD_MIN} (got {eff_gamma})')
    eff = dict(profile='eff', gamma=quantize_gamma(eff_gamma), king_c=float(king_c),
               gamma_free=bool(eff_gamma_free), gamma_min=float(eff_gamma_min),
               gamma_max=float(eff_gamma_max))
    ogcc_profile = str(ogcc_profile).strip().lower()
    if ogcc_profile not in ('king', 'eff'):
        raise ValueError(f"ogcc_profile must be 'king' or 'eff' (got {ogcc_profile!r})")
    ogcc = dict(eff) if ogcc_profile == 'eff' else dict(profile='king', gamma=3.0, king_c=float(king_c))
    return {'YCL': dict(eff), 'MAP': dict(eff), 'OGCC': ogcc, 'default': dict(eff)}


GAUSSIAN_PROFILE = dict(profile='gaussian', gamma=3.0, king_c=1.5)


InstrumentPhotResult = namedtuple('InstrumentPhotResult', [
    'flux_mJy', 'flux_err_mJy',   # aperture flux * corr_factor, dereddened
    'flux_raw_mJy',               # raw (not background-subtracted) aperture sum
    'flux_model_mJy',             # total flux of the fitted model (cross-check), dereddened
    'corr_factor',                # aperture correction actually applied (see _instrument_corr_factor)
    'r65_arcsec',                 # radius enclosing target_ee of the model's total light
    'r_eff_arcsec', 'r_eff_pc',   # source half-light radius (fitted or borrowed; 0 = point source)
    'background_fit',             # pedestal fitted to the slit profile (NaN unless own fit)
    'rms_residual',               # rms of (data - model) over the averaged profile (NaN unless own fit)
    'converged',                  # own fit converged
    'size_source',                # 'own' | 'borrowed:{band}' | 'point_source'
    'centroid_ra', 'centroid_dec',
    'bkg_median_per_pix', 'bkg_std', 'snr',
    'status',                     # 'ok' | 'no_detection' | 'no_flux' | 'exception:{e}'
    'gamma',                      # EFF power index of the model (fitted, borrowed or fixed); NaN otherwise
], defaults=(float('nan'),))


def _empty_instrument_result(status, size_source=''):
    nan = float('nan')
    return InstrumentPhotResult(
        flux_mJy=nan, flux_err_mJy=nan, flux_raw_mJy=nan, flux_model_mJy=nan,
        corr_factor=nan, r65_arcsec=nan, r_eff_arcsec=nan, r_eff_pc=nan,
        background_fit=nan, rms_residual=nan, converged=False, size_source=size_source,
        centroid_ra=nan, centroid_dec=nan, bkg_median_per_pix=nan, bkg_std=nan, snr=nan,
        status=status,
    )


def _instrument_corr_factor(psf_model, spec, r_eff_arcsec, r65_arcsec, x0, y0, target_ee,
                            annulus_r_in_factor, annulus_r_out_factor):
    """
    Aperture correction for the fitted model, measured with the SAME pixel
    masks _aperture_photometry uses (regions 'center' mode) and the same
    annulus background estimate:

        corr = 1 / (EE_ap - N_ap * median_annulus)

    with EE_ap the model's light inside the aperture and median_annulus its
    per-pixel level in the background annulus (both as fractions of the
    model's TOTAL flux). A Gaussian/point source has ~no light in the annulus
    and this is ~1/target_ee; an EFF/King halo still has light at 2.6-4 R65,
    which the annulus background removes from the aperture sum — this puts
    it back.
    """
    from regions import CirclePixelRegion, CircleAnnulusPixelRegion, PixCoord

    pix = psf_model.pixel_scale_arcsec
    r_px = r65_arcsec / pix
    half = int(np.ceil(annulus_r_out_factor * r_px)) + 3
    n = 2 * half + 1
    xc, yc = half + (x0 - np.round(x0)), half + (y0 - np.round(y0))
    img = psf_model.render(xc, yc, (n, n), r_eff_arcsec=r_eff_arcsec, profile=spec['profile'],
                           gamma=spec['gamma'], c=spec['king_c'])
    center = PixCoord(xc, yc)
    ap = CirclePixelRegion(center, r_px).to_mask(mode='center').to_image((n, n)).astype(bool)
    ann = CircleAnnulusPixelRegion(center, annulus_r_in_factor * r_px,
                                   annulus_r_out_factor * r_px).to_mask(mode='center').to_image((n, n)).astype(bool)
    ee_ap = float(img[ap].sum())
    halo = float(np.median(img[ann])) * int(ap.sum()) if ann.any() else 0.0
    denom = ee_ap - halo
    if not np.isfinite(denom) or denom <= 0.05:
        return 1.0 / target_ee
    return 1.0 / denom


def _instrument_aperture_flux(center_sky, x0, y0, psf_model, spec, r_eff_arcsec, r65_arcsec,
                              bd, target_ee, annulus_r_in_factor, annulus_r_out_factor,
                              snr_threshold, eb_v):
    """Aperture photometry at the model R65 plus the model-based correction.
    Returns a dict of the flux fields of InstrumentPhotResult, or None."""
    phot = _aperture_photometry(center_sky, r65_arcsec, bd['image'], bd['header'], bd['wcs'],
                                bd['pixel_scale_arcsec'], annulus_r_in_factor,
                                annulus_r_out_factor, bd.get('unit_convert'))
    if phot is None:
        return None
    corr = _instrument_corr_factor(psf_model, spec, r_eff_arcsec, r65_arcsec, x0, y0, target_ee,
                                   annulus_r_in_factor, annulus_r_out_factor)
    flux = phot['flux_bsub_mJy'] * corr
    err = abs(phot['flux_raw_err_mJy'] * corr)
    snr = (phot['flux_bsub_mJy'] / phot['flux_raw_err_mJy']
           if phot['flux_raw_err_mJy'] not in (0, None) and np.isfinite(phot['flux_raw_err_mJy'])
           else float('nan'))
    status = 'ok'
    if not np.isfinite(snr) or snr < snr_threshold:
        status = 'no_detection'
        flux = float('nan')
    elif bd.get('r_value') is not None:
        flux = corr_flux(flux, bd['r_value'], eb_v)
        err = corr_flux(err, bd['r_value'], eb_v)
    return dict(flux_mJy=flux, flux_err_mJy=err, flux_raw_mJy=phot['flux_raw_mJy'],
                corr_factor=corr, bkg_median_per_pix=phot['bkg_median_per_pix'],
                bkg_std=phot['bkg_std'], snr=snr, status=status)


def _model_flux_mJy(fit_flux, bd, eb_v):
    """Total flux of the fitted model in mJy (dereddened like the aperture flux)."""
    if not np.isfinite(fit_flux):
        return float('nan')
    convert = bd.get('unit_convert') or (lambda v, hdr: v)
    val = (convert(fit_flux, bd['header']) * u.Jy).to(u.mJy).value
    if bd.get('r_value') is not None:
        val = corr_flux(val, bd['r_value'], eb_v)
    return float(val)


def instrument_slit_length(bd, slit_length_arcsec):
    """Slit half-length used for the instrument fit of one band: at least
    slit_length_arcsec and 4x the band's PSF FWHM."""
    fwhm = getattr(bd.get('calib_curve'), 'fwhm_arcsec', 0.0) or 0.0
    return max(slit_length_arcsec, 4.0 * fwhm)


def measure_instrument_photometry_one_source(ra, dec, native_row, band_data, profile_spec,
                                             distance_mpc=float('nan'),
                                             reference_groups=REFERENCE_BAND_GROUPS,
                                             eb_v=0.0, target_ee=0.65,
                                             annulus_r_in_factor=2.6, annulus_r_out_factor=4.0,
                                             n_slits=12, slit_width_px=1, slit_length_arcsec=1.0,
                                             max_r_eff_pc=20.0, max_slit_length_arcsec=3.0,
                                             snr_threshold=3.0):
    """
    Instrument-PSF photometry of ONE source in every band of band_data, with
    two source models: 'instG' (PSF (x) Gaussian) and 'instP' (PSF (x)
    profile_spec, the cluster-class profile).

    native_row : dict band_lower -> dict(aperture_source, centroid_ra,
                 centroid_dec) — this source's results from the existing
                 native photometry (measure_native_photometry_one_source).
    band_data  : as for measure_native_photometry_one_source, plus
                 'psf_model' (instrument_psf.InstrumentPSF) per band.
    distance_mpc : converts R_eff to pc and max_r_eff_pc to arcsec.

    Round 1 — bands whose native aperture_source is 'own' (real emission
    detected there): fit the averaged slit profile through the native
    centroid with fit_slit_profile_instrument_psf (R_eff, flux, background
    free; shape fixed), refitting once with longer slits (5 R_eff, up to
    max_slit_length_arcsec) if the source turns out bigger than the slits.
    Round 2 — every other band, or an own fit that did not converge: borrow
    the SOURCE SIZE (R_eff) from the first converged reference band
    (get_reference_candidates) and compute R65 with this band's OWN PSF, at
    the lender's position; with no lender, treat it as a point source
    (R_eff = 0) at the native centroid (catalog position if none).
    Flux: background-subtracted aperture sum at the model R65 times
    _instrument_corr_factor. Never raises.

    Returns dict band_lower -> {'instG': InstrumentPhotResult,
                                'instP': InstrumentPhotResult}.
    """
    all_bands = list(band_data.keys())
    wavelength_by_band = {b: band_data[b].get('central_wl_AA') for b in all_bands}
    cat_sky = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    arcsec_per_pc = (206265.0 / (distance_mpc * 1e6)
                     if np.isfinite(distance_mpc) and distance_mpc > 0 else float('nan'))
    max_r_eff_arcsec = (max_r_eff_pc * arcsec_per_pc if np.isfinite(arcsec_per_pc) else 0.5)
    specs = {'instG': GAUSSIAN_PROFILE, 'instP': profile_spec}

    def to_pc(r_arcsec):
        return r_arcsec / arcsec_per_pc if np.isfinite(arcsec_per_pc) else float('nan')

    def band_center(band_lower):
        nr = native_row.get(band_lower, {})
        cra, cdec = nr.get('centroid_ra', np.nan), nr.get('centroid_dec', np.nan)
        if np.isfinite(cra) and np.isfinite(cdec):
            return SkyCoord(ra=cra * u.deg, dec=cdec * u.deg)
        return cat_sky

    def model_gamma(spec, r_eff):
        return float(spec['gamma']) if spec['profile'] == 'eff' and r_eff > 0 else float('nan')

    def finish(bd, sky, spec, r_eff, r65, size_source, fit=None):
        psf_model = bd['psf_model']
        x0, y0 = sky.to_pixel(bd['wcs'])
        x0, y0 = float(x0), float(y0)
        ap = _instrument_aperture_flux(sky, x0, y0, psf_model, spec, r_eff, r65, bd, target_ee,
                                       annulus_r_in_factor, annulus_r_out_factor,
                                       snr_threshold, eb_v)
        if ap is None:
            return _empty_instrument_result('no_flux', size_source)
        return InstrumentPhotResult(
            flux_mJy=ap['flux_mJy'], flux_err_mJy=ap['flux_err_mJy'],
            flux_raw_mJy=ap['flux_raw_mJy'],
            flux_model_mJy=_model_flux_mJy(fit.flux, bd, eb_v) if fit is not None else float('nan'),
            corr_factor=ap['corr_factor'], r65_arcsec=r65,
            r_eff_arcsec=r_eff, r_eff_pc=to_pc(r_eff),
            background_fit=fit.background if fit is not None else float('nan'),
            rms_residual=fit.rms_residual if fit is not None else float('nan'),
            converged=bool(fit.converged) if fit is not None else False,
            size_source=size_source,
            centroid_ra=float(sky.ra.deg), centroid_dec=float(sky.dec.deg),
            bkg_median_per_pix=ap['bkg_median_per_pix'], bkg_std=ap['bkg_std'],
            snr=ap['snr'], status=ap['status'], gamma=model_gamma(spec, r_eff),
        )

    results = {b: {} for b in all_bands}
    own_fits = {m: {} for m in INSTRUMENT_METHODS}   # method -> band -> (fit, sky)

    # ── round 1: own fits where the native method found real emission ──────
    for band_lower, bd in band_data.items():
        if native_row.get(band_lower, {}).get('aperture_source') != 'own':
            continue
        sky = band_center(band_lower)
        x0, y0 = sky.to_pixel(bd['wcs'])
        x0, y0 = float(x0), float(y0)
        slit_len = instrument_slit_length(bd, slit_length_arcsec)
        for m, spec in specs.items():
            try:
                kw = dict(target_ee=target_ee, n_slits=n_slits, slit_width_px=slit_width_px,
                          intrinsic_profile=spec['profile'], gamma=spec['gamma'],
                          king_c=spec['king_c'], max_r_eff_arcsec=max_r_eff_arcsec,
                          gamma_free=spec.get('gamma_free', False),
                          gamma_min=spec.get('gamma_min', 2.0), gamma_max=spec.get('gamma_max', 6.0))
                fit = fit_slit_profile_instrument_psf(bd['image'], x0, y0, bd['psf_model'],
                                                      slit_length_arcsec=slit_len, **kw)
                if fit.converged and 5.0 * fit.r_eff_arcsec > slit_len and slit_len < max_slit_length_arcsec:
                    longer = min(5.0 * fit.r_eff_arcsec, max_slit_length_arcsec)
                    fit2 = fit_slit_profile_instrument_psf(bd['image'], x0, y0, bd['psf_model'],
                                                           slit_length_arcsec=longer, **kw)
                    if fit2.converged:
                        fit = fit2
                if not (fit.converged and np.isfinite(fit.r65_arcsec)):
                    continue
                # the model actually fitted: gamma as fitted (free) or as given
                spec_fit = dict(spec, gamma_free=False)
                if spec['profile'] == 'eff' and np.isfinite(fit.gamma):
                    spec_fit['gamma'] = float(fit.gamma)
                own_fits[m][band_lower] = (fit, sky, spec_fit)
                results[band_lower][m] = finish(bd, sky, spec_fit, fit.r_eff_arcsec,
                                                fit.r65_arcsec, 'own', fit)
            except Exception as e:
                logger.debug(f'instrument fit {m} ({band_lower}) failed: {e}')

    # ── round 2: borrow R_eff, or point source ─────────────────────────────
    for band_lower, bd in band_data.items():
        candidates = get_reference_candidates(band_lower, all_bands, wavelength_by_band,
                                              reference_groups)
        for m, spec in specs.items():
            if m in results[band_lower]:
                continue
            try:
                lender = next((c for c in candidates if c in own_fits[m]), None)
                if lender is not None:
                    # size AND shape (fitted gamma) from the lender band
                    fit_l, sky, spec_use = own_fits[m][lender]
                    r_eff = fit_l.r_eff_arcsec
                    size_source = f'borrowed:{lender}'
                else:
                    r_eff = 0.0
                    sky = band_center(band_lower)
                    size_source = 'point_source'
                    spec_use = dict(spec, gamma_free=False)
                x0, y0 = sky.to_pixel(bd['wcs'])
                r65 = bd['psf_model'].r_at_ee(
                    target_ee=target_ee, x0_frac=float(x0 - np.round(x0)),
                    y0_frac=float(y0 - np.round(y0)), profile=spec_use['profile'],
                    r_eff_arcsec=r_eff, gamma=spec_use['gamma'], c=spec_use['king_c'])
                results[band_lower][m] = finish(bd, sky, spec_use, r_eff, r65, size_source)
            except Exception as e:
                logger.debug(f'instrument photometry {m} ({band_lower}) failed: {e}')
                results[band_lower][m] = _empty_instrument_result(f'exception:{e}'[:32])

    return results


def instrument_fit_model_profile(psf_model, x0, y0, r_eff_arcsec, profile, flux, background,
                                 slit_length_arcsec, n_slits=12, slit_width_px=1,
                                 gamma=3.0, king_c=1.5):
    """Averaged slit profile of a fitted instrument model (flux * model +
    background), sampled with exactly the slits used on the data. Returns
    (along_arcsec, profile)."""
    pix = psf_model.pixel_scale_arcsec
    half = int(np.ceil(slit_length_arcsec / pix)) + 3
    shape = (2 * half + 1, 2 * half + 1)
    xl = half + x0 - np.round(x0)
    yl = half + y0 - np.round(y0)
    img = psf_model.render(xl, yl, shape, profile=profile, r_eff_arcsec=r_eff_arcsec,
                           gamma=gamma, c=king_c)
    along, _, _, prof, _ = _sample_averaged_slit_profile(img, xl, yl, pix, slit_length_arcsec,
                                                         n_slits, slit_width_px)
    return along, flux * prof + background


# ─────────────────────────────────────────────────────────────────────────────
# Plot-time-only slit diagnostics — never called from the production
# measurement path above. Used solely by Photometry.plot_native_slit_profiles
# to inspect the averaged-slit fit driving an 'own'-aperture-source
# measurement's size/flux. Deterministic (no RNG anywhere in this module),
# so it reproduces exactly what was measured given the same stored
# r65_native_{band}/r65_psf_native_{band}.
# ─────────────────────────────────────────────────────────────────────────────

SlitPlotData = namedtuple('SlitPlotData', [
    'x0_full', 'y0_full',   # recentered center, full-image pixel coords
    'diagnostics',           # AveragedSlitProfileDiagnostics
    'sigma_arcsec',          # float — same value the real measurement used
    'status',                # 'ok' or 'no_measurement'
])


def remeasure_averaged_slit_profile_at_aperture(ra, dec, image, header, wcs,
                                                  r65_psf_arcsec, r65_arcsec,
                                                  pixel_scale_arcsec,
                                                  n_slits=12, slit_width_px=2,
                                                  slit_length_arcsec=1.0):
    """
    Re-run the 2-D centering + averaged-slit fit for an 'own'-aperture-source
    measurement, with collect_diagnostics=True to recover the raw per-slit
    profiles/fitted curve the production path discards. The slit domain used
    is max(slit_length_arcsec, r65_arcsec) — exactly matching whether the
    stored r65_native_{band} came from the default window or the widened
    re-check (see _measure_own_band) — so this deterministically reproduces
    the real measurement given the same stored aperture.

    Only meaningful for a source/band whose aperture_source=='own' — a
    borrowed/psf_fallback band has no independent slit fit to show;
    Photometry.plot_native_slit_profiles handles those with an
    always-show-image fallback instead of calling this.

    Returns None if the crop has no data at all.
    """
    center_sky = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    x0_cat, y0_cat = center_sky.to_pixel(wcs)

    domain = max(slit_length_arcsec, r65_arcsec if np.isfinite(r65_arcsec) else 0.0)
    fit_half_size_arcsec = max(3.0 * r65_psf_arcsec, 5.0 * domain)
    crop_radius_arcsec = fit_half_size_arcsec * 1.3
    crop_region = CircleSkyRegion(center=center_sky, radius=crop_radius_arcsec * u.arcsec)
    ref_bbox = crop_region.to_pixel(wcs).to_mask().bbox
    img_local, mask_crop = _bbox_local_mask_and_image(crop_region, wcs, image, ref_bbox)
    if mask_crop.astype(bool).sum() == 0:
        return None

    x0_local = x0_cat - ref_bbox.ixmin
    y0_local = y0_cat - ref_bbox.iymin
    fit_half_size_px = max(3.0 * r65_psf_arcsec / pixel_scale_arcsec, 5.0)
    center_fit = fit_2d_gaussian_center(img_local, x0_local, y0_local,
                                         fit_half_size_px, pixel_scale_arcsec)
    x0_use, y0_use = ((center_fit.x0_local, center_fit.y0_local) if center_fit.converged
                       else (x0_local, y0_local))

    result, diagnostics = measure_averaged_slit_profile(
        img_local, x0_use, y0_use, pixel_scale_arcsec, slit_length_arcsec=domain,
        n_slits=n_slits, slit_width_px=slit_width_px, collect_diagnostics=True,
    )

    return SlitPlotData(
        x0_full=x0_use + ref_bbox.ixmin, y0_full=y0_use + ref_bbox.iymin,
        diagnostics=diagnostics, sigma_arcsec=result.sigma_arcsec,
        status='ok' if result.converged else 'no_measurement',
    )


# ─────────────────────────────────────────────────────────────────────────────
# Deger et al. 2022 (PHANGS-HST catalog) aperture correction — for comparison
# plots only, never used by the photometry itself. Details and derivation:
# validation/deger_aperture_correction.py.
#   catalog flux = F4 x 10**(-0.4 (AC_V + offset_band)),
#   F4 = flux within 4 HST px (0.158") with the sky median at 7-8 px.
# ─────────────────────────────────────────────────────────────────────────────
DEGER22_R4_ARCSEC = 0.158                       # 4 px x 0.0396"
DEGER22_SKY_ARCSEC = (0.277, 0.317)             # 7-8 px
# Table 2: AC_tot in V (mag) per galaxy
DEGER22_AC_V = {
    'NGC4826': -0.72, 'NGC3351': -0.68, 'NGC3627': -0.81, 'NGC2835': -0.81, 'NGC5248': -0.65,
    'NGC4571': -0.61, 'NGC4689': -0.63, 'NGC4569': -0.76, 'NGC4535': -0.71, 'NGC1792': -0.80,
    'NGC4548': -0.64, 'NGC1566': -0.62, 'NGC1433': -0.62, 'NGC1559': -0.67, 'NGC1365': -0.61,
    'NGC4654': -0.83, 'NGC2775': -0.45,
}
# Section 4.3: AC_band = AC_V + offset
DEGER22_BAND_OFFSET_MAG = {'F275W': -0.19, 'F336W': -0.12, 'F438W': -0.03, 'F555W': 0.0, 'F814W': -0.12}


def deger22_correction_factor(galaxy, band):
    """Multiplicative catalog correction 10**(-0.4 (AC_V + offset)) for this galaxy
    and HST band; NaN when the galaxy is not in Deger+22 Table 2 or the band has
    no offset (e.g. JWST)."""
    ac_v = DEGER22_AC_V.get(str(galaxy).upper().replace(' ', ''))
    off = DEGER22_BAND_OFFSET_MAG.get(str(band).upper())
    if ac_v is None or off is None:
        return float('nan')
    return float(10 ** (-0.4 * (ac_v + off)))


def deger22_4px_flux(image, x0, y0, pixel_scale_arcsec):
    """(F4, sigma_F4) in image units: flux within 0.158" with the sky median of the
    0.277-0.317" annulus subtracted; error sigma_sky sqrt(N_ap) (clipped std)."""
    from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
    r4 = DEGER22_R4_ARCSEC / pixel_scale_arcsec
    ann = CircularAnnulus((x0, y0), DEGER22_SKY_ARCSEC[0] / pixel_scale_arcsec,
                          DEGER22_SKY_ARCSEC[1] / pixel_scale_arcsec)
    v = ann.to_mask('center').get_values(image)
    v = v[np.isfinite(v)]
    if v.size < 3:
        return float('nan'), float('nan')
    sky = float(np.median(v))
    sig = float(sigma_clipped_stats(v, sigma=3.0)[2])
    f4 = float(aperture_photometry(np.nan_to_num(image - sky), CircularAperture((x0, y0), r4))['aperture_sum'][0])
    return f4, sig * np.sqrt(np.pi * r4 ** 2)


def exact_4px_correction(psf_model, r_eff_arcsec, profile_spec, x0_frac=0.0, y0_frac=0.0):
    """F_total / F4 for a noise-free model of this cluster (profile_spec, R_eff,
    instrument PSF), F4 measured exactly like deger22_4px_flux: the correction the
    Deger+22 recipe would need for this one cluster."""
    pix = psf_model.pixel_scale_arcsec
    hs = int(np.ceil(1.5 / pix))
    n = 2 * hs + 1
    kw = {}
    if np.isfinite(r_eff_arcsec) and r_eff_arcsec > 0:
        kw = dict(profile=profile_spec['profile'], r_eff_arcsec=float(r_eff_arcsec),
                  gamma=profile_spec.get('gamma', 3.0), c=profile_spec.get('king_c', 1.5))
    img = psf_model.render(hs + x0_frac, hs + y0_frac, (n, n), flux=1.0, **kw)
    f4, _ = deger22_4px_flux(img, hs + x0_frac, hs + y0_frac, pix)
    return float(1.0 / f4) if f4 > 0 else float('nan')
