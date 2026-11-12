"""
Regression tests for the bbox-local aperture-mask rewrite in phot_aux_func.py
and aux_functions.py (Phase 1 of the pipeline-hardening plan).

The rewrite replaces `region.to_pixel(wcs).to_mask().to_image(full_shape)`
(which allocates a full-image-sized array for a tiny aperture) with masks
and image cutouts cropped to a small local bounding box. These tests pin the
new bbox-local code down against the old full-image-mask formulas it
replaced, across interior, corner, and image-edge-straddling apertures, so a
future change to this hot path can't silently drift the flux/spectra math.
"""
import numpy as np
from astropy.stats import sigma_clipped_stats
from astropy.coordinates import SkyCoord
from astropy.table import Table
from regions import PolygonPixelRegion, PixCoord, CircleSkyRegion
from scipy.ndimage import binary_dilation
from photutils.detection import find_peaks

from phot_aux_func import (
    _bbox_local_mask_and_image,
    corr_value_func,
    radius_circle,
    to_Jy,
    find_id,
    spectra_from_aperture,
)
from aux_functions import is_isolated_in_region


# ── helpers ──────────────────────────────────────────────────────────────

def _old_full_mask(region, wcs, shape):
    """The pre-rewrite full-image mask, kept here only as a reference to
    compare the new bbox-local code against."""
    return region.to_pixel(wcs).to_mask().to_image(shape)


REGION_CASES = {
    "interior": ([75, 90, 88, 72], [115, 118, 130, 125]),
    "near_corner": ([2, 10, 10, 2], [2, 2, 10, 10]),
    "straddles_far_edge": ([195, 199, 199, 195], [195, 195, 199, 199]),
}


# ── _bbox_local_mask_and_image vs. the old full-image mask ────────────────

class TestBboxLocalMaskAndImage:

    @staticmethod
    def _make_region_and_annulus(vx, vy, wcs):
        poly = PolygonPixelRegion(vertices=PixCoord(x=vx, y=vy))
        region = poly.to_sky(wcs)
        annulus = radius_circle(region)
        return region, annulus

    def test_matches_old_full_image_mask(self, wcs, rng):
        image = rng.normal(5.0, 1.0, size=(200, 200))
        for name, (vx, vy) in REGION_CASES.items():
            region, annulus = self._make_region_and_annulus(vx, vy, wcs)

            old_mask_src = _old_full_mask(region, wcs, image.shape)
            old_mask_ann = _old_full_mask(annulus, wcs, image.shape)
            old_flux = np.nansum(image * old_mask_src)
            old_n_source = int(old_mask_src.astype(bool).sum())
            old_annulus_mask = old_mask_ann * (~old_mask_src.astype(bool))
            old_ann_values = image * old_annulus_mask
            old_ann_values[old_ann_values == 0] = np.nan
            _, old_median, _ = sigma_clipped_stats(old_ann_values, sigma=3.0)

            ref_bbox = annulus.to_pixel(wcs).to_mask().bbox
            img_local, mask_ann_local = _bbox_local_mask_and_image(annulus, wcs, image, ref_bbox)
            _, mask_src_local = _bbox_local_mask_and_image(region, wcs, image, ref_bbox)
            new_flux = np.nansum(img_local * mask_src_local)
            new_n_source = int(mask_src_local.astype(bool).sum())

            corr_value, _ = corr_value_func(mask_src_local, mask_ann_local, img_local)

            assert new_n_source == old_n_source, name
            assert np.isclose(new_flux, old_flux), name
            assert np.isclose(corr_value, old_median * old_n_source, rtol=1e-6), name

    def test_region_entirely_outside_image_returns_zeros(self, wcs):
        image = np.ones((50, 50))
        poly = PolygonPixelRegion(vertices=PixCoord(x=[-50, -40, -40, -50], y=[-50, -40, -40, -50]))
        region = poly.to_sky(wcs)
        ref_bbox = region.to_pixel(wcs).to_mask().bbox
        img_local, mask_local = _bbox_local_mask_and_image(region, wcs, image, ref_bbox)
        assert np.all(mask_local == 0)
        assert np.all(img_local == 0)


# ── spectra_from_aperture: bbox-local + sum-then-convert refactor ─────────

class TestSpectraFromAperture:

    def test_matches_old_per_pixel_to_jy(self, wcs, rng):
        shape = (150, 150)
        original_id_map = rng.integers(0, 100, size=shape)
        original_id_map[rng.random(shape) < 0.05] = -1  # unbinned spaxels
        binned_id_map = original_id_map.copy()

        nwl = 20
        log_wl = np.linspace(np.log(4000), np.log(9000), nwl)
        spectra = Table()
        spectra['SPEC'] = [rng.normal(10, 1, nwl) for _ in range(100)]

        poly = PolygonPixelRegion(vertices=PixCoord(x=[60, 75, 73, 58], y=[70, 72, 85, 80]))
        region = poly.to_sky(wcs)

        sum_o, sum_Jy = spectra_from_aperture(
            region, shape, wcs, log_wl, spectra, (original_id_map, binned_id_map)
        )

        mask_full = _old_full_mask(region, wcs, shape)
        y_c, x_c = np.where(mask_full)
        ref_o, ref_Jy = [], []
        for x, y in zip(x_c, y_c):
            idn, _ = find_id(x, y, original_id_map, binned_id_map)
            if idn < 0 or idn >= len(spectra):
                continue
            s = spectra[idn]['SPEC']
            ref_o.append(s)
            ref_Jy.append(to_Jy(log_wl, s).value)
        ref_sum_o = np.sum(ref_o, axis=0)
        ref_sum_Jy = np.sum(ref_Jy, axis=0)

        assert np.allclose(sum_o, ref_sum_o)
        assert np.allclose(sum_Jy.value, ref_sum_Jy)

    def test_bad_spaxel_ids_are_excluded_not_fatal(self, wcs, rng):
        shape = (60, 60)
        # every id is invalid (-1) -> aperture has no usable spaxels at all
        original_id_map = np.full(shape, -1)
        binned_id_map = original_id_map.copy()
        log_wl = np.linspace(np.log(4000), np.log(9000), 10)
        spectra = Table()
        spectra['SPEC'] = [rng.normal(10, 1, 10) for _ in range(5)]

        poly = PolygonPixelRegion(vertices=PixCoord(x=[20, 30, 28, 18], y=[20, 22, 32, 28]))
        region = poly.to_sky(wcs)

        # should not raise, even though every pixel's id is invalid
        spectra_from_aperture(region, shape, wcs, log_wl, spectra, (original_id_map, binned_id_map))


# ── is_isolated_in_region: bbox-local dilation/find_peaks ──────────────────

class TestIsIsolatedInRegion:

    @staticmethod
    def _old_approach(image, wcs, sky_region, threshold_sigma=3.0, box_size=3):
        pixel_region = sky_region.to_pixel(wcs)
        region_mask = pixel_region.to_mask().to_image(image.shape).astype(bool)
        if region_mask.sum() == 0:
            return True, 0
        dilated = binary_dilation(region_mask, iterations=10)
        annulus = dilated & ~region_mask & np.isfinite(image)
        if annulus.sum() > 5:
            _, bkg_median, bkg_std = sigma_clipped_stats(image[annulus], sigma=3.0)
        else:
            _, bkg_median, bkg_std = sigma_clipped_stats(image[np.isfinite(image)], sigma=3.0)
        threshold = bkg_median + threshold_sigma * bkg_std
        exclude_mask = ~region_mask | ~np.isfinite(image)
        peaks = find_peaks(image, threshold=threshold, box_size=box_size, mask=exclude_mask)
        n_peaks = 0 if peaks is None else len(peaks)
        return (n_peaks <= 1), n_peaks

    def test_matches_old_full_image_approach(self, wcs, rng):
        cases = []

        img_a = rng.normal(50, 3, size=(400, 400))
        img_a[200, 210] += 300
        poly_a = PolygonPixelRegion(
            vertices=PixCoord(x=[195, 215, 213, 193], y=[195, 198, 215, 210])
        ).to_sky(wcs)
        cases.append((img_a, poly_a))

        img_b = rng.normal(50, 3, size=(400, 400))
        img_b[200, 205] += 300
        img_b[203, 212] += 250
        poly_b = PolygonPixelRegion(
            vertices=PixCoord(x=[195, 215, 213, 193], y=[195, 198, 215, 210])
        ).to_sky(wcs)
        cases.append((img_b, poly_b))

        img_c = rng.normal(50, 3, size=(400, 400))
        img_c[5, 8] += 300
        poly_c = PolygonPixelRegion(
            vertices=PixCoord(x=[2, 12, 10, 0], y=[2, 4, 12, 9])
        ).to_sky(wcs)
        cases.append((img_c, poly_c))

        for image, region in cases:
            old_isolated, old_n = self._old_approach(image, wcs, region)
            new_isolated, new_n, _ = is_isolated_in_region(
                image, wcs, region, 0, 0, threshold_sigma=3.0, box_size=3, verbose=False
            )
            assert new_isolated == old_isolated
            assert new_n == old_n

    def test_region_entirely_outside_image_does_not_crash(self, wcs, rng):
        image = rng.normal(50, 3, size=(400, 400))
        poly = PolygonPixelRegion(
            vertices=PixCoord(x=[-50, -40, -40, -50], y=[-50, -40, -40, -50])
        ).to_sky(wcs)
        isolated, n_peaks, peaks = is_isolated_in_region(
            image, wcs, poly, 0, 0, verbose=False
        )
        assert isolated is True
        assert n_peaks == 0
        assert peaks is None
