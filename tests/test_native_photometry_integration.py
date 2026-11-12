"""
Integration tests for native_phot_aux_func.py wired into analysis.py's
Photometry class (Photometry.measure_native_psf_shape,
Photometry.run_native_photometry) — as opposed to test_native_photometry.py,
which tests native_phot_aux_func.py's own functions directly without going
through Photometry at all.

These tests build a minimal Photometry instance via __new__ (mirroring the
existing tests/test_parallel_photometry.py pattern) rather than a full
pipeline run, so they don't need real FITS/catalog files.

A few of these caught real bugs during development, not just wiring issues:
- test_run_native_photometry_is_additive: guards the "add columns, never
  touch existing ones" contract explicitly (a broken placeholder line once
  slipped through here — `slits[...] = ... np.nan  # placeholder overwritten
  below` — that never actually populated the per-slit diagnostic array).
- test_alpha_recovery_on_clean_isolated_stars /
  test_warn_if_alpha_unphysical: regression coverage for a real robustness
  gap found while validating measure_native_psf_shape() — fit_2d_moffat's
  convergence check only rejected fits parked at alpha's LOWER bound (1.5),
  not its upper bound (20.0); combined with a very permissive default
  n_sources, spurious/low-S/N DAOStarFinder detections could silently drag
  the aggregate median alpha to ~18-19 instead of a physically sensible
  value. Fixed in psf_aux_func.fit_2d_moffat (symmetric bound check) and
  Photometry.measure_native_psf_shape (n_sources default 10000 -> 50, plus
  a sanity-check warning).
"""
import os
import shutil

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pytest
from astropy import units as u
from astropy.wcs import WCS
from astropy.table import Table
from astropy.modeling.models import Moffat2D
from astropy.coordinates import SkyCoord
from regions import PolygonSkyRegion, CircleSkyRegion

import analysis
from native_phot_aux_func import moffat_gamma_from_fwhm, remeasure_averaged_slit_profile_at_aperture


def _native_wcs(pixel_scale_arcsec, npix=300, crval=(150.0, 2.0)):
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crval = list(crval)
    w.wcs.crpix = [npix / 2, npix / 2]
    w.wcs.cdelt = [-pixel_scale_arcsec / 3600, pixel_scale_arcsec / 3600]
    w.pixel_shape = (npix, npix)
    return w


def _tiny_square_region(center_sky, half_width_arcsec=0.1):
    """Minimal PolygonSkyRegion (has .vertices.x/.y once .to_pixel()'d) —
    just enough to exercise plot_native_photometry_diagnostics' MUSE panel,
    which reprojects self.regions_HST_final the same way plot_sed does."""
    d = (half_width_arcsec * u.arcsec).to(u.deg).value
    ra, dec = center_sky.ra.deg, center_sky.dec.deg
    verts = SkyCoord(ra=[ra - d, ra + d, ra + d, ra - d] * u.deg,
                      dec=[dec - d, dec - d, dec + d, dec + d] * u.deg)
    return PolygonSkyRegion(vertices=verts)


def _build_photometry(tmp_path, n_sources=6, seed=7, with_muse=False):
    rng = np.random.default_rng(seed)
    pixel_scale = 0.05
    psf_fwhm, psf_alpha = 0.10, 2.8
    true_fwhm = 0.12
    true_flux_Jy = {'f555w': 2.0e-6, 'f200w': 1.5e-6}
    PHOTMJSR, PIXAR_SR = 1.0, 2.35e-13
    size = 300
    sources = [(rng.uniform(60, 240), rng.uniform(60, 240)) for _ in range(n_sources)]

    hst_images, jwst_images = {}, {}
    wcs_hst = _native_wcs(pixel_scale, size)
    wcs_jwst = _native_wcs(pixel_scale, size)

    for band, images, true_jy in [('f555w', hst_images, true_flux_Jy['f555w']),
                                   ('f200w', jwst_images, true_flux_Jy['f200w'])]:
        img = np.zeros((size, size))
        yy, xx = np.mgrid[0:size, 0:size]
        # JWST raw pixel values must be in DN/s (raw * PHOTMJSR * PIXAR_SR is
        # MJy, matching phot_aux_func.spectra_from_aperture's JWST branch and
        # analysis._jwst_unit_convert's Jy-normalized contract -> divide by 1e6)
        injected_total = (true_jy * 1e-6) / (PHOTMJSR * PIXAR_SR) if band == 'f200w' else true_jy
        for (xc, yc) in sources:
            gamma_px = moffat_gamma_from_fwhm(true_fwhm, psf_alpha) / pixel_scale
            m = Moffat2D(amplitude=1.0, x_0=xc, y_0=yc, gamma=gamma_px, alpha=psf_alpha)
            src = m(xx, yy)
            src *= injected_total / src.sum()
            img += src
        noise_scale = injected_total * 1e-3
        img += injected_total * 0.005 + rng.normal(0, noise_scale, size=(size, size))
        header = {'PHOTMJSR': PHOTMJSR, 'PIXAR_SR': PIXAR_SR} if band == 'f200w' else {}
        images[band] = (img, header, wcs_hst if band == 'f555w' else wcs_jwst)

    ras, decs = [], []
    for (xc, yc) in sources:
        ra, dec = wcs_hst.pixel_to_world_values(xc, yc)
        ras.append(ra)
        decs.append(dec)
    final_table = Table({'PHANGS_RA': ras, 'PHANGS_DEC': decs, 'id': np.arange(n_sources)})

    p = analysis.Photometry.__new__(analysis.Photometry)
    p.base_dir = str(tmp_path)
    p.gal = 'TESTGAL_NATIVE'
    p.hst_images = hst_images
    p.jwst_images = jwst_images
    p.psf_hst = {'f555w': psf_fwhm}
    p.psf_jwst = {'f200w': psf_fwhm}
    p.psf_hst_alpha = {'f555w': psf_alpha}
    p.psf_jwst_alpha = {'f200w': psf_alpha}
    p.final_table = final_table
    p.eb_v = 0.0
    p.photometry_table = Table(final_table)
    os.makedirs(os.path.join(str(tmp_path), p.gal), exist_ok=True)

    if with_muse:
        muse_size = 200
        muse_wcs = _native_wcs(0.2, muse_size, crval=(150.0, 2.0))
        p.muse_data = rng.normal(1e-3, 1e-4, size=(muse_size, muse_size))
        p.muse_wcs = muse_wcs
        p.regions_HST_final = [
            _tiny_square_region(SkyCoord(ra=ra * u.deg, dec=dec * u.deg))
            for ra, dec in zip(ras, decs)
        ]
        p.annulus_regions = [
            CircleSkyRegion(center=SkyCoord(ra=ra * u.deg, dec=dec * u.deg),
                             radius=0.3 * u.arcsec)
            for ra, dec in zip(ras, decs)
        ]
        n_wl = 50
        p.log_wl = np.log(np.linspace(4750.0, 9350.0, n_wl))
        p.photometry_spectra = {
            'spectra_Jy': np.array(
                [rng.normal(1e-6, 1e-7, size=n_wl) for _ in range(n_sources)],
                dtype=object,
            )
        }

    return p, true_flux_Jy


FILTER_WAVELENGTHS = {'f555w': (5500.0, 3.1), 'f200w': (20000.0, 0.5)}


class TestRunNativePhotometry:

    def test_requires_photometry_table(self, tmp_path):
        p, _ = _build_photometry(tmp_path)
        p.photometry_table = None
        with pytest.raises(RuntimeError, match='photometry_table'):
            p.run_native_photometry(FILTER_WAVELENGTHS)

    def test_requires_psf_alpha(self, tmp_path):
        p, _ = _build_photometry(tmp_path)
        p.psf_hst_alpha = {}
        p.psf_jwst_alpha = {}
        with pytest.raises(RuntimeError, match='alpha'):
            p.run_native_photometry(FILTER_WAVELENGTHS)

    def test_is_additive(self, tmp_path):
        """Existing (non-native) columns must be untouched, and new native_*
        columns added and actually populated. Per the module's own design
        (native_phot_aux_func.py docstring: "Plot-time-only slit
        diagnostics — never called from the production measurement path"),
        raw per-slit profiles are NOT cached on self — they're recomputed
        on demand via remeasure_averaged_slit_profile_at_aperture for
        plotting — so this checks that recompute path actually produces
        finite diagnostics for an 'own'-aperture source/band, rather than
        asserting on the pre-refactor self.native_photometry_slits cache."""
        p, _ = _build_photometry(tmp_path)
        p.photometry_table['flux_f555w'] = np.full(len(p.final_table), 999.0)

        p.run_native_photometry(FILTER_WAVELENGTHS, n_workers=1, verbose=False)

        assert np.all(p.photometry_table['flux_f555w'] == 999.0)
        assert 'flux_native_f555w' in p.photometry_table.colnames
        assert 'flux_native_f200w' in p.photometry_table.colnames
        assert 'sigma_native_f555w' in p.photometry_table.colnames
        assert 'size_extrapolated_native_f555w' in p.photometry_table.colnames
        assert np.any(np.isfinite(p.photometry_table['sigma_native_f555w']))

        row = p.photometry_table[0]
        assert str(row['aperture_source_native_f555w']).strip() == 'own'
        plot_data = remeasure_averaged_slit_profile_at_aperture(
            float(row['PHANGS_RA']), float(row['PHANGS_DEC']),
            *p.hst_images['f555w'],
            p.native_band_calib['f555w']['r65_psf_arcsec'],
            float(row['r65_native_f555w']),
            p.native_band_calib['f555w']['pixel_scale_arcsec'],
        )
        assert plot_data is not None and plot_data.status == 'ok'
        assert np.any(np.isfinite(plot_data.diagnostics.averaged_profile))

    def test_recovers_injected_flux(self, tmp_path):
        p, true_flux_Jy = _build_photometry(tmp_path)
        p.run_native_photometry(FILTER_WAVELENGTHS, n_workers=1, verbose=False)

        for band, true_jy in true_flux_Jy.items():
            true_mJy = true_jy * 1e3
            recovered = np.asarray(p.photometry_table[f'flux_native_{band}'], dtype=float)
            assert np.all(np.isfinite(recovered))
            rel_err = np.abs(recovered - true_mJy) / true_mJy
            assert np.all(rel_err < 0.2), f'{band}: rel_err={rel_err}'

    def test_parallel_matches_sequential(self, tmp_path):
        p_seq, _ = _build_photometry(tmp_path / 'seq')
        p_seq.run_native_photometry(FILTER_WAVELENGTHS, n_workers=1, verbose=False)

        p_par, _ = _build_photometry(tmp_path / 'par')
        p_par.run_native_photometry(FILTER_WAVELENGTHS, n_workers=3, verbose=False)

        native_cols = [c for c in p_seq.photometry_table.colnames if 'native' in c]
        for c in native_cols:
            seq_vals, par_vals = p_seq.photometry_table[c], p_par.photometry_table[c]
            if seq_vals.dtype.kind in 'US':
                assert list(seq_vals) == list(par_vals), c
            else:
                assert np.allclose(np.asarray(seq_vals, dtype=float),
                                    np.asarray(par_vals, dtype=float), equal_nan=True), c

    def test_cache_is_reused_and_force_recompute_bypasses_it(self, tmp_path):
        p1, _ = _build_photometry(tmp_path, n_sources=1)
        p1.run_native_photometry(FILTER_WAVELENGTHS, n_workers=1, verbose=False)
        flux1 = p1.photometry_table['flux_native_f555w'][0]

        # second instance with a blank (~zero-flux) image — if the cache is
        # correctly reused, this must NOT affect the result
        p2, _ = _build_photometry(tmp_path, n_sources=1)
        size = p2.hst_images['f555w'][0].shape[0]
        blank = np.full((size, size), 1e-8)
        p2.hst_images['f555w'] = (blank, {}, p2.hst_images['f555w'][2])
        p2.run_native_photometry(FILTER_WAVELENGTHS, n_workers=1, verbose=False)
        assert np.isclose(p2.photometry_table['flux_native_f555w'][0], flux1)

        # force_recompute=True must bypass the cache and use the blank image
        p3, _ = _build_photometry(tmp_path, n_sources=1)
        p3.hst_images['f555w'] = (blank, {}, p3.hst_images['f555w'][2])
        p3.run_native_photometry(FILTER_WAVELENGTHS, n_workers=1, verbose=False, force_recompute=True)
        assert not np.isclose(p3.photometry_table['flux_native_f555w'][0], flux1, equal_nan=False) \
            or not np.isfinite(p3.photometry_table['flux_native_f555w'][0])


class TestJwstUnitConvert:
    """Regression for a real bug: raw_sum * PHOTMJSR * PIXAR_SR is in MJy
    (see phot_aux_func.spectra_from_aperture's JWST branch, which explicitly
    tags it `* u.MJy` before `.to('mJy')`), but analysis._jwst_unit_convert
    once returned that bare product and native_phot_aux_func._measure_at_aperture
    treats unit_convert's return value as already being in Jy — silently
    undercounting JWST native flux by 1e6x. Caught from real NGC1365 data
    where native JWST fluxes sat several orders of magnitude below both the
    convolved-image method and the PHANGS catalog."""

    def test_matches_phot_aux_func_jwst_convention(self):
        raw_sum = 6.38e6
        header = {'PHOTMJSR': 0.7, 'PIXAR_SR': 2.35e-13}

        got_jy = analysis._jwst_unit_convert(raw_sum, header)

        expected_mjy = raw_sum * header['PHOTMJSR'] * header['PIXAR_SR']
        expected_jy = (expected_mjy * u.MJy).to(u.Jy).value

        assert np.isclose(got_jy, expected_jy)
        assert np.isclose(got_jy, expected_mjy * 1e6)


class TestPlotNativePhotometryDiagnostics:
    """Smoke tests for the aperture/annulus QA plot added alongside the
    JWST unit-conversion fix — the systematic JWST offset was only obvious
    once plotted against wavelength, so this plot is meant to catch that
    class of bug visually going forward. Only checks that files land where
    expected and nothing crashes; not a pixel-content test."""

    def test_muse_panel_and_spectrum_dont_crash(self, tmp_path):
        """With synthetic MUSE data present (self.muse_data/regions_HST_final/
        annulus_regions/log_wl/photometry_spectra), the restructured 14-panel
        grid (MUSE + HST bands on top, remaining JWST bands below, mirroring
        plot_sed) must render without error and still produce one PNG per
        source."""
        p, _ = _build_photometry(tmp_path, n_sources=2, with_muse=True)
        p.run_native_photometry(FILTER_WAVELENGTHS, n_workers=1, verbose=False)

        out_dir = os.path.join(str(tmp_path), p.gal, 'native_diagnostics')
        pngs = [f for f in os.listdir(out_dir) if f.endswith('.png')]
        assert len(pngs) == 2

    def test_missing_muse_data_falls_back_gracefully(self, tmp_path):
        """Without MUSE attributes (the default fixture), the plot must
        still work exactly as before — no MUSE panel, no crash."""
        p, _ = _build_photometry(tmp_path, n_sources=1, with_muse=False)
        p.run_native_photometry(FILTER_WAVELENGTHS, n_workers=1, verbose=False)

        out_dir = os.path.join(str(tmp_path), p.gal, 'native_diagnostics')
        assert len([f for f in os.listdir(out_dir) if f.endswith('.png')]) == 1

    def test_run_native_photometry_writes_diagnostic_plots_by_default(self, tmp_path):
        p, _ = _build_photometry(tmp_path, n_sources=2)
        p.run_native_photometry(FILTER_WAVELENGTHS, n_workers=1, verbose=False)

        out_dir = os.path.join(str(tmp_path), p.gal, 'native_diagnostics')
        assert os.path.isdir(out_dir)
        pngs = [f for f in os.listdir(out_dir) if f.endswith('.png')]
        assert len(pngs) == 2

    def test_plot_diagnostics_false_skips_plotting(self, tmp_path):
        p, _ = _build_photometry(tmp_path, n_sources=2)
        p.run_native_photometry(FILTER_WAVELENGTHS, n_workers=1, verbose=False,
                                 plot_diagnostics=False)

        out_dir = os.path.join(str(tmp_path), p.gal, 'native_diagnostics')
        assert not os.path.isdir(out_dir)

    def test_single_source_selection(self, tmp_path):
        p, _ = _build_photometry(tmp_path, n_sources=3)
        p.run_native_photometry(FILTER_WAVELENGTHS, n_workers=1, verbose=False,
                                 plot_diagnostics=False)

        p.plot_native_photometry_diagnostics(FILTER_WAVELENGTHS, idd=1, verbose=False)

        out_dir = os.path.join(str(tmp_path), p.gal, 'native_diagnostics')
        pngs = [f for f in os.listdir(out_dir) if f.endswith('.png')]
        assert len(pngs) == 1
        assert pngs[0].startswith('1_')

    def test_requires_native_columns(self, tmp_path):
        p, _ = _build_photometry(tmp_path, n_sources=1)
        with pytest.raises(RuntimeError, match='run_native_photometry'):
            p.plot_native_photometry_diagnostics(FILTER_WAVELENGTHS)

    def test_cache_hit_still_produces_plots(self, tmp_path):
        p1, _ = _build_photometry(tmp_path, n_sources=1)
        p1.run_native_photometry(FILTER_WAVELENGTHS, n_workers=1, verbose=False)

        out_dir = os.path.join(str(tmp_path), p1.gal, 'native_diagnostics')
        shutil.rmtree(out_dir)
        assert not os.path.isdir(out_dir)

        p2, _ = _build_photometry(tmp_path, n_sources=1)
        p2.run_native_photometry(FILTER_WAVELENGTHS, n_workers=1, verbose=False)

        assert os.path.isdir(out_dir)
        assert len([f for f in os.listdir(out_dir) if f.endswith('.png')]) == 1


class TestPlotNativeSizeCalibration:
    """Smoke tests for the size-calibration QA plot — shows each source's
    own search-pass sigma_bar against its band's calibration curve, to
    explain WHY a source was flagged size_extrapolated (see
    native_phot_aux_func.measure_native_photometry_one_band's docstring:
    a source can only be pushed past the curve's ceiling, essentially never
    below its floor, since model 0 is already the bare PSF)."""

    def test_run_native_photometry_writes_calibration_plots_by_default(self, tmp_path):
        p, _ = _build_photometry(tmp_path, n_sources=2)
        p.run_native_photometry(FILTER_WAVELENGTHS, n_workers=1, verbose=False)

        out_dir = os.path.join(str(tmp_path), p.gal, 'native_size_calibration')
        assert os.path.isdir(out_dir)
        pngs = [f for f in os.listdir(out_dir) if f.endswith('.png')]
        assert len(pngs) == 2

    def test_plot_diagnostics_false_skips_size_calibration_too(self, tmp_path):
        p, _ = _build_photometry(tmp_path, n_sources=2)
        p.run_native_photometry(FILTER_WAVELENGTHS, n_workers=1, verbose=False,
                                 plot_diagnostics=False)

        out_dir = os.path.join(str(tmp_path), p.gal, 'native_size_calibration')
        assert not os.path.isdir(out_dir)

    def test_requires_native_band_calib(self, tmp_path):
        p, _ = _build_photometry(tmp_path, n_sources=1)
        with pytest.raises(RuntimeError, match='native_band_calib'):
            p.plot_native_size_calibration(FILTER_WAVELENGTHS)

    def test_cache_hit_still_produces_calibration_plots(self, tmp_path):
        p1, _ = _build_photometry(tmp_path, n_sources=1)
        p1.run_native_photometry(FILTER_WAVELENGTHS, n_workers=1, verbose=False)

        out_dir = os.path.join(str(tmp_path), p1.gal, 'native_size_calibration')
        shutil.rmtree(out_dir)
        assert not os.path.isdir(out_dir)

        p2, _ = _build_photometry(tmp_path, n_sources=1)
        p2.run_native_photometry(FILTER_WAVELENGTHS, n_workers=1, verbose=False)

        assert os.path.isdir(out_dir)
        assert len([f for f in os.listdir(out_dir) if f.endswith('.png')]) == 1


class TestPlotNativeSlitProfiles:
    """Smoke tests for the slit-profile QA plot — the previously-black-box
    step (measure_slit_sigmas discarding its raw per-slit data) made
    inspectable, styled after the Appendix A paper's own figure (slits as
    diameters through the center, all profiles overlaid on one signed-radius
    log panel). One file per SOURCE with every band as a row — opt-in, not
    auto-triggered by run_native_photometry."""

    def test_writes_one_file_per_source_with_all_bands(self, tmp_path):
        p, _ = _build_photometry(tmp_path, n_sources=2)
        p.run_native_photometry(FILTER_WAVELENGTHS, n_workers=1, verbose=False,
                                 plot_diagnostics=False)

        p.plot_native_slit_profiles(FILTER_WAVELENGTHS, verbose=False)

        out_dir = os.path.join(str(tmp_path), p.gal, 'native_slit_profiles')
        assert os.path.isdir(out_dir)
        pngs = [f for f in os.listdir(out_dir) if f.endswith('.png')]
        assert len(pngs) == 2
        # no per-band subdirectories
        assert not any(os.path.isdir(os.path.join(out_dir, f)) for f in os.listdir(out_dir))

    def test_idd_and_bands_filters(self, tmp_path):
        p, _ = _build_photometry(tmp_path, n_sources=3)
        p.run_native_photometry(FILTER_WAVELENGTHS, n_workers=1, verbose=False,
                                 plot_diagnostics=False)

        p.plot_native_slit_profiles(FILTER_WAVELENGTHS, idd=1, bands=['f555w'], verbose=False)

        out_dir = os.path.join(str(tmp_path), p.gal, 'native_slit_profiles')
        pngs = [f for f in os.listdir(out_dir) if f.endswith('.png')]
        assert len(pngs) == 1
        assert pngs[0].startswith('1_')

    def test_requires_native_columns(self, tmp_path):
        p, _ = _build_photometry(tmp_path, n_sources=1)
        with pytest.raises(RuntimeError, match='run_native_photometry'):
            p.plot_native_slit_profiles(FILTER_WAVELENGTHS)


class TestMeasureNativePsfShape:

    def test_alpha_recovery_on_clean_isolated_stars(self, tmp_path):
        """Regression for the n_sources=10000 -> spurious-detection
        contamination bug: with well-separated, high-S/N stars and the new
        conservative n_sources default, alpha should recover close to truth."""
        rng = np.random.default_rng(3)
        pixel_scale = 0.03
        size = 800
        true_fwhm, true_alpha = 0.15, 2.6

        img = np.full((size, size), 50.0) + rng.normal(0, 1.0, (size, size))
        yy, xx = np.mgrid[0:size, 0:size]
        gamma_px = moffat_gamma_from_fwhm(true_fwhm, true_alpha) / pixel_scale
        positions, attempts = [], 0
        while len(positions) < 8 and attempts < 1000:
            xc, yc = rng.uniform(100, size - 100), rng.uniform(100, size - 100)
            if all(np.hypot(xc - px, yc - py) > 150 for px, py in positions):
                positions.append((xc, yc))
            attempts += 1
        for (xc, yc) in positions:
            amp = rng.uniform(800, 2000)
            m = Moffat2D(amplitude=amp, x_0=xc, y_0=yc, gamma=gamma_px, alpha=true_alpha)
            img += m(xx, yy)

        wcs_hst = _native_wcs(pixel_scale, size)
        p = analysis.Photometry.__new__(analysis.Photometry)
        p.gal = 'TESTPSFSHAPE_INT'
        p.psf_filepath = str(tmp_path / 'psf_table.csv')
        p.hst_images = {'f555w': (img, {}, wcs_hst)}
        p.jwst_images = {}

        p.measure_native_psf_shape(fwhm_guess_hst=5.0, r_limit_hst=0.4, n_sources=8)

        assert 'f555w' in p.psf_hst_alpha
        assert abs(p.psf_hst_alpha['f555w'] - true_alpha) < 0.3
        assert abs(p.psf_hst_gamma['f555w'] - gamma_px) < 0.3

    def test_does_not_touch_existing_fwhm_columns(self, tmp_path):
        """measure_native_psf_shape must only add new *_alpha/_gamma columns
        to psf_table.csv, never disturb the existing psf_HST_{band}/
        psf_JWST_{band} FWHM columns load_or_measure_psf owns."""
        from aux_functions import save_psf_to_table
        from astropy.io import ascii

        psf_filepath = str(tmp_path / 'psf_table.csv')
        save_psf_to_table('TESTGAL_TOUCH', {'psf_HST_f555w': 0.123}, filepath=psf_filepath)

        rng = np.random.default_rng(1)
        pixel_scale = 0.03
        size = 200
        img = np.full((size, size), 50.0) + rng.normal(0, 1.0, (size, size))
        yy, xx = np.mgrid[0:size, 0:size]
        gamma_px = moffat_gamma_from_fwhm(0.15, 2.6) / pixel_scale
        m = Moffat2D(amplitude=2000, x_0=100, y_0=100, gamma=gamma_px, alpha=2.6)
        img += m(xx, yy)
        wcs_hst = _native_wcs(pixel_scale, size)

        p = analysis.Photometry.__new__(analysis.Photometry)
        p.gal = 'TESTGAL_TOUCH'
        p.psf_filepath = psf_filepath
        p.hst_images = {'f555w': (img, {}, wcs_hst)}
        p.jwst_images = {}
        p.measure_native_psf_shape(fwhm_guess_hst=5.0, r_limit_hst=0.4, n_sources=5)

        tbl = ascii.read(psf_filepath, format='csv')
        idx = list(tbl['galaxy_name']).index('TESTGAL_TOUCH')
        assert float(tbl['psf_HST_f555w'][idx]) == 0.123
        assert 'psf_HST_f555w_alpha' in tbl.colnames

    def test_warn_if_alpha_unphysical(self, caplog):
        import logging
        # analysis.logger has propagate=False (see module setup), so caplog's
        # root-logger handler won't see it via the normal propagation path —
        # attach caplog's handler directly to this logger for the test.
        analysis.logger.addHandler(caplog.handler)
        try:
            with caplog.at_level(logging.WARNING, logger='analysis'):
                caplog.clear()
                analysis.Photometry._warn_if_alpha_unphysical('HST', 'f555w', 15.0)
                assert any('outside the typical physical range' in r.message
                           for r in caplog.records)

                caplog.clear()
                analysis.Photometry._warn_if_alpha_unphysical('HST', 'f555w', 2.6)
                assert not caplog.records
        finally:
            analysis.logger.removeHandler(caplog.handler)
