"""
Regression tests for native_phot_aux_func.py's own functions, exercised
directly (not through the Photometry class — see
test_native_photometry_integration.py for that) — the native-resolution,
per-cluster-R65 aperture-corrected photometry method, "v2" design (see the
module's own docstring): per-source-per-band emission detection, 2-D
Gaussian centering, an averaged-slit-profile size measurement (samples
n_slits diameters, averages them POINTWISE across slits before fitting ONE
Gaussian1D+Const1D — not the earlier per-slit-independent-fit-then-
accept/reject approach), and aperture-borrowing/PSF-fallback for bands
without their own detected emission.

Several of these tests exist specifically because they caught real bugs,
not just because "more tests are good":
- test_scale_invariant (TestMeasureAveragedSlitProfile): TRFLSQFitter was
  found to silently bias the fitted sigma low by ~40% when fitting profiles
  at realistic Jy-scale flux magnitudes (~1e-7) vs. O(1) synthetic test
  data — a numerical-conditioning bug in the fitter, fixed by normalizing
  the profile to O(1) before fitting (mean/stddev are invariant under a
  multiplicative y-rescaling; only the discarded fitted amplitude is not).
- test_build_native_calibration_curve_ee_monotonic (TestCalibrationCurve):
  the Gaussian1D stddev fit had an upper bound pinned at the aperture
  radius, which is correct for real (small) sources but wrong for the
  calibration grid's most heavily-convolved (broad) models — it silently
  truncated the fitted sigma for exactly the extended-source regime the
  calibration curve most needs to cover, producing a non-monotonic
  EE-vs-sigma curve. Fixed by relaxing the bound and adding an isotonic
  (running-minimum) correction for the residual noise at the extreme end.
- test_nan_in_one_slit_only_drops_that_point_not_whole_slit
  (TestMeasureAveragedSlitProfile): a NaN sample anywhere in one slit's
  domain must only drop out of THAT along-slit position's pointwise mean,
  never discard the rest of that slit — the explicit design requirement
  behind averaging profiles pointwise rather than requiring every slit to
  be fully clean.
- test_r65_growth_is_capped / test_r65_growth_cap_is_configurable
  (TestEndToEndRecovery): regression for a real-data incident (4/5 HST
  filters agreeing on R65~0.05", one landing at 0.2" — 4x bigger — from a
  corrupted search-pass fit). max_r65_growth_factor clamps an 'own'-band
  r65_arcsec to at most this multiple of the band's own r65_psf_arcsec —
  a widened re-check of the slit domain alone is not sufficient protection
  against a single outlier fit.
- detect_emission's own background computation (exercised indirectly by
  every TestEndToEndRecovery test, since v2 gates _measure_own_band behind
  it): a prior bug in how _bbox_local_mask_and_image's return value was
  combined between a source/inner/outer region triple made the background
  annulus always read as literally zero, so detect_emission always failed
  its bkg_std>0 guard and every source fell back to psf_fallback — see the
  fix and comments in detect_emission/_aperture_photometry themselves.
"""
import numpy as np
import pytest
from astropy.wcs import WCS
from astropy.modeling.models import Moffat2D

from native_phot_aux_func import (
    moffat_gamma_from_fwhm,
    moffat_ee_analytic,
    render_moffat_model_image,
    compute_r65,
    measure_averaged_slit_profile,
    build_native_calibration_curve,
    ee_from_calibration_curve,
    estimate_extra_conv_fwhm,
    measure_native_photometry_one_source,
    remeasure_averaged_slit_profile_at_aperture,
)


# ── shared synthetic setup ──────────────────────────────────────────────────

BAND_CASES = [
    # (fwhm_arcsec, alpha, pixel_scale_arcsec, label)
    (0.15, 2.8, 0.05, "mid-HST-like"),
    (0.08, 2.5, 0.03, "sharp-HST-like"),
    (0.50, 2.2, 0.11, "coarse-JWST-like"),
    (0.20, 3.5, 0.06, "NIRCam-like"),
]


def _native_wcs(pixel_scale_arcsec, npix=300, crval=(150.0, 2.0)):
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crval = list(crval)
    w.wcs.crpix = [npix / 2, npix / 2]
    w.wcs.cdelt = [-pixel_scale_arcsec / 3600, pixel_scale_arcsec / 3600]
    w.pixel_shape = (npix, npix)
    return w


def _single_band_data(image, header, wcs, pixel_scale_arcsec, calib_curve, r65_psf_arcsec,
                       band='testband', r_value=None, unit_convert=None, central_wl_AA=5500.0):
    """Minimal single-band band_data dict — the shape
    measure_native_photometry_one_source expects (see its own docstring
    and Photometry.run_native_photometry, which builds this dict once per
    real band)."""
    return {
        band: dict(image=image, header=header, wcs=wcs, pixel_scale_arcsec=pixel_scale_arcsec,
                   calib_curve=calib_curve, r65_psf_arcsec=r65_psf_arcsec,
                   r_value=r_value, unit_convert=unit_convert, central_wl_AA=central_wl_AA),
    }


# ── analytic Moffat EE cross-check ──────────────────────────────────────────

class TestMoffatAnalytic:

    def test_ee_bounds_and_monotonic(self):
        fwhm, alpha = 0.15, 2.8
        r = np.linspace(0.001, 2.0, 200)
        ee = moffat_ee_analytic(r, fwhm, alpha)
        assert np.all(np.diff(ee) > 0)
        assert ee[0] > 0 and ee[0] < 0.1
        assert ee[-1] > 0.99

    @pytest.mark.parametrize("fwhm,alpha,pixscale", [(f, a, p) for f, a, p, _ in BAND_CASES])
    def test_r65_numeric_matches_closed_form(self, fwhm, alpha, pixscale):
        r65_numeric = compute_r65(fwhm, alpha, pixscale, method='numeric')
        r65_closed = compute_r65(fwhm, alpha, pixscale, method='closed_form')
        assert abs(r65_numeric - r65_closed) / r65_closed < 0.02

    @pytest.mark.parametrize("fwhm,alpha,pixscale", [
        # A wide sweep of real-instrument-like (fwhm, alpha, pixel_scale)
        # combinations — regression for a bug found on real HST/JWST data:
        # compute_r65's numeric method built its monotonic curve-of-growth
        # via a local pairwise filter ("keep if greater than the previous
        # RAW point") that does not guarantee the FILTERED sequence is
        # globally monotonic (e.g. ee=[0.1,0.30,0.15,0.20] kept
        # [0.1,0.30,0.20], which itself decreases) — PchipInterpolator then
        # raised "x must be strictly increasing". Fixed with
        # np.maximum.accumulate (a proper isotonic correction) instead.
        (0.03, 1.6, 0.011), (0.05, 1.8, 0.02), (0.12, 4.5, 0.031),
        (0.30, 1.5, 0.05), (0.09, 6.0, 0.013), (0.25, 2.9, 0.065),
        (0.06, 2.1, 0.008), (0.4, 3.8, 0.11), (0.02, 5.5, 0.006),
    ])
    def test_r65_numeric_does_not_crash_on_nonmonotonic_ee(self, fwhm, alpha, pixscale):
        """compute_r65(method='numeric') must never raise on any physically
        sensible (fwhm, alpha, pixel_scale) combination, regardless of
        small numerical/pixelization non-monotonicity in the raw EE curve."""
        r65 = compute_r65(fwhm, alpha, pixscale, method='numeric')
        assert np.isfinite(r65) and r65 > 0


# ── model rendering ──────────────────────────────────────────────────────────

class TestRenderModelImage:

    def test_normalized_to_unit_flux(self):
        img, scale = render_moffat_model_image(0.15, 2.8, 0.0, 0.05, oversample=4)
        assert abs(img.sum() - 1.0) < 1e-9

    def test_extra_convolution_broadens_and_lowers_peak(self):
        img_sharp, _ = render_moffat_model_image(0.15, 2.8, 0.0, 0.05, oversample=4)
        img_broad, _ = render_moffat_model_image(0.15, 2.8, 0.3, 0.05, oversample=4)
        assert img_broad.max() < img_sharp.max()
        assert abs(img_broad.sum() - 1.0) < 1e-6


# ── averaged-slit-profile size measurement ──────────────────────────────────

class TestMeasureAveragedSlitProfile:

    def test_recovers_known_gaussian_sigma(self):
        pixel_scale = 0.05
        true_sigma = 0.3
        size = 200
        yy, xx = np.mgrid[0:size, 0:size]
        x0, y0 = size / 2, size / 2
        r2 = (xx - x0) ** 2 + (yy - y0) ** 2
        img = np.exp(-r2 / (2 * (true_sigma / pixel_scale) ** 2)) * 1e-8  # realistic tiny scale

        result, _ = measure_averaged_slit_profile(
            img, x0, y0, pixel_scale, slit_length_arcsec=3.0, n_slits=12, slit_width_px=2)
        assert result.converged
        assert result.n_slits_with_data == 12
        assert abs(result.sigma_arcsec - true_sigma) / true_sigma < 0.01

    def test_collect_diagnostics_matches_result_and_reproduces_profile(self):
        """collect_diagnostics=True must not change sigma_arcsec (same
        computation, just also returning what was already computed
        internally); the returned averaged_profile must be exactly the
        pointwise nanmean of the 12 raw per-slit profiles (the whole point
        of averaging FIRST, before any fit); and the fitted curve must
        actually reproduce the averaged profile it was fit to."""
        pixel_scale = 0.05
        true_sigma = 0.3
        size = 200
        yy, xx = np.mgrid[0:size, 0:size]
        x0, y0 = size / 2, size / 2
        r2 = (xx - x0) ** 2 + (yy - y0) ** 2
        img = np.exp(-r2 / (2 * (true_sigma / pixel_scale) ** 2)) * 1e-8

        result_plain, diag_none = measure_averaged_slit_profile(
            img, x0, y0, pixel_scale, slit_length_arcsec=3.0, n_slits=12, slit_width_px=2)
        result_diag, diagnostics = measure_averaged_slit_profile(
            img, x0, y0, pixel_scale, slit_length_arcsec=3.0, n_slits=12, slit_width_px=2,
            collect_diagnostics=True)

        assert diag_none is None
        assert result_plain.sigma_arcsec == pytest.approx(result_diag.sigma_arcsec, nan_ok=True)
        assert result_plain.converged and result_diag.converged
        assert diagnostics.per_slit_profiles.shape[0] == 12
        assert diagnostics.theta_deg == pytest.approx(np.arange(12) * 180.0 / 12)

        with np.errstate(all='ignore'):
            expected_avg = np.nanmean(diagnostics.per_slit_profiles, axis=0)
        np.testing.assert_allclose(diagnostics.averaged_profile, expected_avg, equal_nan=True)

        valid = np.isfinite(diagnostics.averaged_profile) & np.isfinite(diagnostics.fitted_curve)
        assert valid.sum() > 5
        resid = np.abs(diagnostics.fitted_curve[valid] - diagnostics.averaged_profile[valid])
        assert np.max(resid) < 0.05 * np.max(np.abs(diagnostics.averaged_profile[valid]))

    def test_scale_invariant(self):
        """Regression for the TRFLSQFitter numerical-conditioning bug: the
        SAME profile shape must give the SAME sigma_arcsec whether the data
        is O(1) or realistic Jy-scale (~1e-7)."""
        pixel_scale, fwhm, alpha = 0.05, 0.10, 2.8
        r65 = compute_r65(fwhm, alpha, pixel_scale)
        gamma_px = moffat_gamma_from_fwhm(fwhm, alpha) / pixel_scale
        size = 14
        xc, yc = 5.3, 7.7
        yy, xx = np.mgrid[0:size, 0:size]
        m = Moffat2D(amplitude=1.0, x_0=xc, y_0=yc, gamma=gamma_px, alpha=alpha)
        img_unit = m(xx, yy)
        img_unit /= img_unit.sum()

        r_unit, _ = measure_averaged_slit_profile(
            img_unit * 1.0, xc, yc, pixel_scale, slit_length_arcsec=5.0,
            n_slits=12, slit_width_px=2)
        r_tiny, _ = measure_averaged_slit_profile(
            img_unit * 1e-7, xc, yc, pixel_scale, slit_length_arcsec=5.0,
            n_slits=12, slit_width_px=2)
        assert r_unit.converged and r_tiny.converged
        assert abs(r_unit.sigma_arcsec - r_tiny.sigma_arcsec) / r_unit.sigma_arcsec < 1e-3

    def test_nan_in_one_slit_only_drops_that_point_not_whole_slit(self):
        """A NaN hole affecting part of one slit's domain must only drop
        out of THAT along-slit position's pointwise mean — never discard
        the rest of that slit — per the explicit design requirement behind
        averaging profiles pointwise before fitting a single Gaussian."""
        pixel_scale = 0.05
        true_sigma = 0.3
        size = 200
        yy, xx = np.mgrid[0:size, 0:size]
        x0, y0 = size / 2.0, size / 2.0
        r2 = (xx - x0) ** 2 + (yy - y0) ** 2
        img = np.exp(-r2 / (2 * (true_sigma / pixel_scale) ** 2)) * 1e-8

        # a NaN hole in the wings of slit k=0 (theta=0, along +x), away
        # from the profile's peak so it doesn't dominate the fit
        img_nan = img.copy()
        hole_x = int(x0 + 2.0 / pixel_scale)
        img_nan[int(y0) - 1:int(y0) + 2, hole_x - 1:hole_x + 2] = np.nan

        result, diagnostics = measure_averaged_slit_profile(
            img_nan, x0, y0, pixel_scale, slit_length_arcsec=3.0,
            n_slits=12, slit_width_px=1, collect_diagnostics=True)

        assert result.converged
        # sigma is barely affected — the hole is diluted by the other 11
        # slits at that along-slit position, not by discarding slit k=0
        assert abs(result.sigma_arcsec - true_sigma) / true_sigma < 0.02
        # slit k=0 still has plenty of finite samples elsewhere along its
        # own length — it was not dropped wholesale
        assert np.isfinite(diagnostics.per_slit_profiles[0]).sum() > 10
        # the hole actually mattered: at least one along-slit position has
        # fewer than 12 finite slits contributing to its mean
        with np.errstate(all='ignore'):
            n_with_data = np.sum(np.isfinite(diagnostics.per_slit_profiles), axis=0)
        assert n_with_data.min() < 12

    def test_single_contaminated_slit_is_diluted_by_averaging(self):
        """A strong contaminant sitting on ONE diameter's axis must not
        dominate the single combined fit the way it would an independent
        per-slit fit — averaging pointwise across all 12 orientations
        BEFORE fitting dilutes one bad orientation's contribution at each
        along-slit position to 1/12, rather than requiring an explicit
        accept/reject step. The recovered sigma should stay reasonably
        close to the true value despite the contamination."""
        pixel_scale = 0.05
        true_sigma = 0.3
        size = 200
        yy, xx = np.mgrid[0:size, 0:size]
        x0, y0 = size / 2, size / 2
        r2 = (xx - x0) ** 2 + (yy - y0) ** 2
        img = np.exp(-r2 / (2 * (true_sigma / pixel_scale) ** 2))

        contam_r_px = 0.5 / pixel_scale
        cx, cy = int(x0 + contam_r_px), int(y0)
        img_contam = img.copy()
        img_contam[cy - 3:cy + 3, cx - 3:cx + 3] += 5.0 * img.max()

        result_clean, _ = measure_averaged_slit_profile(
            img, x0, y0, pixel_scale, slit_length_arcsec=3.0, n_slits=12, slit_width_px=2)
        result_contam, _ = measure_averaged_slit_profile(
            img_contam, x0, y0, pixel_scale, slit_length_arcsec=3.0, n_slits=12, slit_width_px=2)

        assert result_clean.converged and result_contam.converged
        # the contaminated fit is perturbed relative to the clean one...
        assert result_contam.sigma_arcsec != pytest.approx(result_clean.sigma_arcsec, rel=1e-6)
        # ...but averaging across 12 orientations keeps the combined sigma
        # within a modest tolerance of the true value, unlike a raw
        # single-slit fit on the contaminated orientation alone would be
        assert abs(result_contam.sigma_arcsec - true_sigma) / true_sigma < 0.5


# ── calibration curve ────────────────────────────────────────────────────────

class TestCalibrationCurve:

    @pytest.mark.parametrize("fwhm,alpha,pixscale,label", BAND_CASES)
    def test_ee_monotonic_decreasing_in_sigma(self, fwhm, alpha, pixscale, label):
        r65 = compute_r65(fwhm, alpha, pixscale)
        calib = build_native_calibration_curve(fwhm, alpha, pixscale, r65, n_models=20)
        assert np.all(np.diff(calib.ee_model) <= 1e-9), label

    def test_ee_lookup_matches_model_points(self):
        fwhm, alpha, pixscale = 0.15, 2.8, 0.05
        r65 = compute_r65(fwhm, alpha, pixscale)
        calib = build_native_calibration_curve(fwhm, alpha, pixscale, r65, n_models=20)

        ee, extrapolated = ee_from_calibration_curve(calib, calib.sigma_bar_model[10])
        assert not extrapolated
        assert abs(ee - calib.ee_model[10]) < 1e-6

    def test_ee_lookup_flags_extrapolation(self):
        fwhm, alpha, pixscale = 0.15, 2.8, 0.05
        r65 = compute_r65(fwhm, alpha, pixscale)
        calib = build_native_calibration_curve(fwhm, alpha, pixscale, r65, n_models=20)

        _, extrap_below = ee_from_calibration_curve(calib, calib.sigma_bar_model[0] - 0.01)
        _, extrap_above = ee_from_calibration_curve(calib, calib.sigma_bar_model[-1] + 0.5)
        assert extrap_below and extrap_above

    def test_estimate_extra_conv_fwhm_self_consistent(self):
        """Feeding a calibration model's OWN sigma_bar back into
        estimate_extra_conv_fwhm must recover that model's own
        extra_conv_fwhm (the size-estimate inversion step)."""
        fwhm, alpha, pixscale = 0.15, 2.8, 0.05
        r65 = compute_r65(fwhm, alpha, pixscale)
        calib = build_native_calibration_curve(fwhm, alpha, pixscale, r65, n_models=20)

        for i in range(len(calib.sigma_bar_model)):
            est, extrapolated = estimate_extra_conv_fwhm(calib, calib.sigma_bar_model[i])
            assert not extrapolated
            assert abs(est - calib.extra_conv_fwhm_model[i]) < 1e-6


class TestPerClusterR65:

    def test_compute_r65_grows_with_extra_convolution(self):
        """A broader (more convolved) model must have a LARGER R65 than the
        bare PSF — the physical basis for per-cluster R65 growing with
        cluster size."""
        fwhm, alpha, pixscale = 0.10, 2.8, 0.05
        r65_psf = compute_r65(fwhm, alpha, pixscale)
        r65_broader = compute_r65(fwhm, alpha, pixscale, extra_conv_fwhm_arcsec=0.1)
        r65_broader_still = compute_r65(fwhm, alpha, pixscale, extra_conv_fwhm_arcsec=0.3)
        assert r65_psf < r65_broader < r65_broader_still

    def test_closed_form_rejects_nonzero_extra_conv(self):
        with pytest.raises(ValueError):
            compute_r65(0.1, 2.8, 0.05, method='closed_form', extra_conv_fwhm_arcsec=0.1)


# ── end-to-end synthetic recovery ────────────────────────────────────────────

class TestEndToEndRecovery:

    def _inject_and_measure(self, true_fwhm_arcsec, psf_fwhm_arcsec=0.10, psf_alpha=2.8,
                             pixel_scale_arcsec=0.05, true_total_flux_Jy=2.0e-6, seed=42,
                             offset_px=(0.5, -0.3), noise_std=2e-9, bkg_level=1e-8,
                             max_r65_growth_factor=2.0):
        # offset_px is deliberately small by default: unlike the pre-v2
        # design, detect_emission (round 1) tests for emission at the
        # UNRECENTERED catalog position using the band's small, FIXED
        # r65_psf_arcsec test aperture — a large catalog-position offset
        # can make a real source miss that small test aperture entirely and
        # fall to psf_fallback before 2-D Gaussian recentering ever gets a
        # chance to run. test_recenter_flag_threshold below picks its own,
        # deliberately larger-but-still-detectable offsets to exercise that
        # recentering path specifically.
        rng = np.random.default_rng(seed)
        wcs = _native_wcs(pixel_scale_arcsec)
        size = 300
        yy, xx = np.mgrid[0:size, 0:size]
        xc, yc = 150.3, 149.7
        gamma_px = moffat_gamma_from_fwhm(true_fwhm_arcsec, psf_alpha) / pixel_scale_arcsec
        moffat = Moffat2D(amplitude=1.0, x_0=xc, y_0=yc, gamma=gamma_px, alpha=psf_alpha)
        source_img = moffat(xx, yy)
        source_img *= true_total_flux_Jy / source_img.sum()
        image = source_img + bkg_level + rng.normal(0, noise_std, size=(size, size))

        r65 = compute_r65(psf_fwhm_arcsec, psf_alpha, pixel_scale_arcsec)
        calib = build_native_calibration_curve(psf_fwhm_arcsec, psf_alpha, pixel_scale_arcsec,
                                                 r65, n_models=20)
        ra_cat, dec_cat = wcs.pixel_to_world_values(xc + offset_px[0], yc + offset_px[1])

        band_data = _single_band_data(image, {}, wcs, pixel_scale_arcsec, calib, r65)
        results = measure_native_photometry_one_source(
            ra_cat, dec_cat, band_data, eb_v=0.0,
            max_r65_growth_factor=max_r65_growth_factor,
        )
        return results['testband'], true_total_flux_Jy * 1e3

    @pytest.mark.parametrize("true_fwhm_arcsec", [0.10, 0.12, 0.15, 0.18])
    def test_recovers_injected_flux_within_calibration_range(self, true_fwhm_arcsec):
        """Source size within ~1.8x the reference PSF FWHM — solidly inside
        the calibration curve's range — should recover injected flux to
        within ~20% (background/noise/recentering all included, not a
        noiseless idealization)."""
        result, true_flux_mJy = self._inject_and_measure(true_fwhm_arcsec)
        assert result.status == 'ok'
        assert not result.size_extrapolated
        rel_err = abs(result.flux_mJy - true_flux_mJy) / true_flux_mJy
        assert rel_err < 0.20, f"fwhm={true_fwhm_arcsec}: rel_err={rel_err:.2%}"

    def test_flags_extrapolation_for_source_far_broader_than_calibration(self):
        """A source much broader than any calibration model must be caught
        by size_extrapolated=True, not silently returned as if trustworthy.
        Flux is boosted here: at the default flux level a source this
        diffuse (6x the reference PSF FWHM, same total flux spread over a
        much larger area) has too low a peak surface brightness to even
        clear detect_emission's own-band gate — a real, physically
        expected effect (a very extended source is intrinsically
        lower-contrast for the same integrated flux), not something this
        test is trying to probe; boosting flux isolates the extrapolation
        behavior from that separate detection-threshold effect."""
        result, _ = self._inject_and_measure(true_fwhm_arcsec=0.60, true_total_flux_Jy=1.0e-4)
        assert result.size_extrapolated

    def test_sigma_arcsec_reported_raw_not_clipped(self):
        """sigma_arcsec must be the RAW measured size proxy, never clipped
        to the calibration curve's range — unlike
        extra_conv_fwhm_estimate_arcsec (the size ESTIMATE derived from it,
        which IS clipped by estimate_extra_conv_fwhm) and unlike r65_arcsec
        (which the growth cap CAN clip). This is what lets a diagnostic
        plot show how far past the calibration ceiling a flagged source
        actually fell, not just that it was flagged."""
        psf_fwhm_arcsec, psf_alpha, pixel_scale_arcsec = 0.10, 2.8, 0.05
        r65 = compute_r65(psf_fwhm_arcsec, psf_alpha, pixel_scale_arcsec)
        calib = build_native_calibration_curve(psf_fwhm_arcsec, psf_alpha, pixel_scale_arcsec,
                                                 r65, n_models=20)

        result, _ = self._inject_and_measure(
            true_fwhm_arcsec=0.60, psf_fwhm_arcsec=psf_fwhm_arcsec,
            psf_alpha=psf_alpha, pixel_scale_arcsec=pixel_scale_arcsec,
            true_total_flux_Jy=1.0e-4,
        )

        assert result.size_extrapolated
        assert np.isfinite(result.sigma_arcsec)
        assert result.sigma_arcsec > calib.sigma_bar_model[-1]

    def test_recenter_flag_threshold(self):
        """A deliberately large-but-still-detectable catalog-position
        offset must trip recenter_flagged; a tiny one must not. The "large"
        offset here is intentionally more modest than an extreme
        mismatch — see _inject_and_measure's own note on why an offset
        much bigger than the band's r65_psf_arcsec can miss
        detect_emission's small fixed test aperture altogether."""
        result_small, _ = self._inject_and_measure(0.10, offset_px=(0.1, 0.05))
        result_large, _ = self._inject_and_measure(0.10, offset_px=(1.0, 0.8))
        assert not result_small.recenter_flagged
        assert result_large.recenter_flagged

    def test_r65_grows_with_true_source_size(self):
        """The whole point of the per-cluster redesign: a genuinely broader
        source must end up with a LARGER own aperture (r65_arcsec) than a
        compact one, both larger than the band's fixed PSF-based aperture
        (r65_psf_arcsec, which is the SAME for every source in a band).
        Uses 0.18" (1.8x the reference PSF FWHM) as the "extended" case —
        comfortably below where this synthetic setup starts tripping the
        default max_r65_growth_factor=2.0 cap (that boundary sits right
        around 0.20" here) — see TestR65GrowthCap for the (deliberately
        capped) larger sizes."""
        result_compact, _ = self._inject_and_measure(true_fwhm_arcsec=0.10)
        result_extended, _ = self._inject_and_measure(true_fwhm_arcsec=0.18)

        assert result_compact.r65_psf_arcsec == pytest.approx(result_extended.r65_psf_arcsec)
        assert result_compact.r65_arcsec < result_extended.r65_arcsec
        # the extended source's own aperture should meaningfully exceed the
        # band's naive PSF-sized one (not just marginally), and must not
        # have been clamped by the growth cap at this size
        assert not result_extended.size_capped
        assert result_extended.r65_arcsec > 1.5 * result_extended.r65_psf_arcsec

    @pytest.mark.parametrize("true_fwhm_arcsec,rel_err_tol", [
        (0.10, 0.20), (0.15, 0.20), (0.20, 0.20),
        # 0.25 and 0.35 (2.5x and 3.5x the reference PSF) both hit the
        # default max_r65_growth_factor=2.0 cap in this synthetic setup
        # (see TestR65GrowthCap below) — their uncapped size estimates
        # would put R65 well beyond the band's PSF-based R65, which the cap
        # intentionally refuses to trust. This is an accepted, deliberate
        # trade-off (avoid oversized/contaminated apertures at the cost of
        # under-correcting genuinely huge sources) rather than a
        # regression — loosened tolerances document that, don't hide it.
        (0.25, 0.30), (0.35, 0.60),
    ])
    def test_recovers_injected_flux_across_size_range(self, true_fwhm_arcsec, rel_err_tol):
        """End-to-end recovery across a wide range of source sizes relative
        to the reference PSF (1x to 3.5x)."""
        result, true_flux_mJy = self._inject_and_measure(true_fwhm_arcsec)
        assert np.isfinite(result.flux_mJy), f"fwhm={true_fwhm_arcsec}: flux was masked/NaN"
        rel_err = abs(result.flux_mJy - true_flux_mJy) / true_flux_mJy
        assert rel_err < rel_err_tol, f"fwhm={true_fwhm_arcsec}: rel_err={rel_err:.2%}"

    def test_no_detection_masking_on_pure_noise(self):
        """A band with no real source emission (pure background + noise)
        must be flagged as a non-detection and its flux masked to NaN, not
        reported as a small/noisy but ostensibly-real flux value. With no
        emission detected, this resolves via the psf_fallback path (see
        measure_native_photometry_one_source), same as a real MIRI
        non-detection would."""
        rng = np.random.default_rng(5)
        pixel_scale_arcsec = 0.05
        size = 300
        wcs = _native_wcs(pixel_scale_arcsec, size)
        image = 1e-8 + rng.normal(0, 2e-9, size=(size, size))
        ra, dec = wcs.pixel_to_world_values(size / 2, size / 2)

        psf_fwhm, psf_alpha = 0.10, 2.8
        r65 = compute_r65(psf_fwhm, psf_alpha, pixel_scale_arcsec)
        calib = build_native_calibration_curve(psf_fwhm, psf_alpha, pixel_scale_arcsec,
                                                 r65, n_models=20)

        band_data = _single_band_data(image, {}, wcs, pixel_scale_arcsec, calib, r65,
                                       band='noband')
        result = measure_native_photometry_one_source(ra, dec, band_data, eb_v=0.0)['noband']
        assert result.status == 'no_detection'
        assert np.isnan(result.flux_mJy)
        assert np.isfinite(result.flux_raw_mJy)  # still reported, just flagged unreliable

    def test_real_source_is_not_masked_as_non_detection(self):
        """Sanity counterpart to the above: a real, well-detected source
        must NOT be masked, so the S/N threshold isn't so aggressive it
        throws away genuine detections."""
        result, _ = self._inject_and_measure(true_fwhm_arcsec=0.10)
        assert result.status != 'no_detection'
        assert np.isfinite(result.flux_mJy)


class TestR65GrowthCap:

    def test_r65_growth_is_capped(self):
        """A source engineered to produce a size estimate far beyond
        max_r65_growth_factor * r65_psf_arcsec must have its final aperture
        clamped down, not trusted as-is — regression for a real-data case:
        4/5 HST filters agreeing on R65~0.05", one landing at 0.2" (4x
        bigger) from a corrupted search-pass fit."""
        result, _ = TestEndToEndRecovery()._inject_and_measure(true_fwhm_arcsec=0.35)
        assert result.size_capped
        assert result.r65_arcsec == pytest.approx(2.0 * result.r65_psf_arcsec, rel=1e-6)

    def test_r65_growth_cap_is_configurable(self):
        """max_r65_growth_factor must actually change where the cap engages
        — a tighter factor caps a source a looser one wouldn't."""
        wcs = _native_wcs(0.05)
        size = 300
        rng = np.random.default_rng(11)
        yy, xx = np.mgrid[0:size, 0:size]
        xc, yc = 150.0, 150.0
        psf_fwhm, psf_alpha, pixel_scale = 0.10, 2.8, 0.05
        true_fwhm = 0.20
        gamma_px = moffat_gamma_from_fwhm(true_fwhm, psf_alpha) / pixel_scale
        moffat = Moffat2D(amplitude=1.0, x_0=xc, y_0=yc, gamma=gamma_px, alpha=psf_alpha)
        source_img = moffat(xx, yy)
        source_img *= 2.0e-6 / source_img.sum()
        image = source_img + 1e-8 + rng.normal(0, 2e-9, size=(size, size))
        ra, dec = wcs.pixel_to_world_values(xc, yc)

        r65 = compute_r65(psf_fwhm, psf_alpha, pixel_scale)
        calib = build_native_calibration_curve(psf_fwhm, psf_alpha, pixel_scale, r65, n_models=20)
        band_data = _single_band_data(image, {}, wcs, pixel_scale, calib, r65)

        result_tight = measure_native_photometry_one_source(
            ra, dec, band_data, eb_v=0.0, max_r65_growth_factor=1.1)['testband']
        result_loose = measure_native_photometry_one_source(
            ra, dec, band_data, eb_v=0.0, max_r65_growth_factor=10.0)['testband']
        assert result_tight.size_capped
        assert result_tight.r65_arcsec == pytest.approx(1.1 * result_tight.r65_psf_arcsec, rel=1e-6)
        assert result_loose.r65_arcsec > result_tight.r65_arcsec


# ── plot-time slit-diagnostic re-run ─────────────────────────────────────────

class TestRemeasureSlitDiagnostics:

    def _scene(self, pixel_scale_arcsec=0.05, psf_fwhm=0.10, psf_alpha=2.8, true_fwhm=0.10):
        size = 300
        wcs = _native_wcs(pixel_scale_arcsec, size)
        rng = np.random.default_rng(3)
        yy, xx = np.mgrid[0:size, 0:size]
        xc, yc = 150.0, 150.0
        gamma_px = moffat_gamma_from_fwhm(true_fwhm, psf_alpha) / pixel_scale_arcsec
        moffat = Moffat2D(amplitude=1.0, x_0=xc, y_0=yc, gamma=gamma_px, alpha=psf_alpha)
        source_img = moffat(xx, yy)
        source_img *= 2.0e-6 / source_img.sum()
        image = source_img + 1e-8 + rng.normal(0, 2e-9, size=(size, size))
        ra, dec = wcs.pixel_to_world_values(xc, yc)
        r65 = compute_r65(psf_fwhm, psf_alpha, pixel_scale_arcsec)
        return ra, dec, image, wcs, pixel_scale_arcsec, r65

    def test_sigma_matches_measure_native_photometry_one_source(self):
        """remeasure_averaged_slit_profile_at_aperture, called with the same
        aperture radius, must reproduce exactly the sigma_arcsec the real
        (production) measurement obtained at that same radius — it's a
        re-run of the identical validated code path, not a separate
        reimplementation, so this should hold to machine precision."""
        ra, dec, image, wcs, pixel_scale, r65 = self._scene()
        calib = build_native_calibration_curve(0.10, 2.8, pixel_scale, r65, n_models=20)
        band_data = _single_band_data(image, {}, wcs, pixel_scale, calib, r65)

        result = measure_native_photometry_one_source(
            ra, dec, band_data, eb_v=0.0)['testband']
        plot_data = remeasure_averaged_slit_profile_at_aperture(
            ra, dec, image, {}, wcs, r65, result.r65_arcsec, pixel_scale,
        )

        assert plot_data is not None and plot_data.status == 'ok'
        assert plot_data.sigma_arcsec == pytest.approx(result.sigma_arcsec, rel=1e-9, nan_ok=True)
        assert plot_data.diagnostics.per_slit_profiles.shape[0] == 12
        assert np.isfinite(plot_data.x0_full) and np.isfinite(plot_data.y0_full)


# ── dereddening reuse ────────────────────────────────────────────────────────

def test_dereddening_matches_phot_aux_func_corr_flux():
    from phot_aux_func import corr_flux
    flux, r_value, eb_v = 1.234, 2.5, 0.08
    result, _ = TestEndToEndRecovery()._inject_and_measure(0.10)
    # can't directly re-derive result.flux_mJy's pre-dereddening value here
    # without duplicating the pipeline, so instead assert the shared
    # function itself is what's imported and used (no local reimplementation)
    import native_phot_aux_func
    assert native_phot_aux_func.corr_flux is corr_flux
    assert corr_flux(flux, r_value, eb_v) == flux * 10 ** (0.4 * r_value * eb_v)
