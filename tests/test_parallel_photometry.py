"""
Regression test for Photometry.run_spectro_photometry's opt-in
multiprocessing path (Phase 2 of the pipeline-hardening plan).

n_workers=1 (the default) runs the original sequential loop unchanged.
n_workers>1 dispatches per-source work to a ProcessPoolExecutor using the
'fork' start method, with large shared inputs (convolved images, MUSE
spectra table, region lists) passed once via `initializer`/`initargs`
rather than per task. This test builds a small fully-synthetic `Photometry`
instance (no real FITS/catalog files needed) and asserts the parallel path
reproduces the sequential path's output exactly, column for column.
"""
import os
import warnings

import numpy as np
import pytest
from astropy.table import Table
from regions import PolygonPixelRegion, PixCoord

import analysis


@pytest.fixture
def synthetic_photometry(wcs, rng, tmp_path):
    n_sources = 8

    hst_img = rng.normal(5.0, 1.0, size=(200, 200)).astype(float)
    jwst_img = rng.normal(5.0, 1.0, size=(200, 200)).astype(float)
    jwst_hdr = {'PHOTMJSR': 1.0, 'PIXAR_SR': 2.35e-13}

    filter_wavelengths = {
        'f555w': (5500.0, 3.1),
        'f606w': (6060.0, 2.9),
        'f814w': (8140.0, 1.8),
        'f125w': (12500.0, 1.2),
        'f160w': (16000.0, 0.9),
        'f200w': (20000.0, 0.5),
    }
    hst_conv_dict = {b: (hst_img, {}, wcs) for b in ['f555w', 'f606w', 'f814w', 'f125w', 'f160w']}
    jwst_conv_dict = {'f200w': (jwst_img, jwst_hdr, wcs)}

    muse_shape = (60, 60)
    original_id_map = rng.integers(0, 200, size=muse_shape)
    original_id_map[rng.random(muse_shape) < 0.05] = -1
    binned_id_map = original_id_map.copy()
    log_wl = np.linspace(np.log(4000), np.log(9000), 15)
    spectra = Table()
    spectra['SPEC'] = [rng.normal(10, 1, 15) for _ in range(200)]

    regions_hst, regions_muse = [], []
    for _ in range(n_sources):
        cx, cy = rng.uniform(10, 50), rng.uniform(10, 50)
        r = rng.uniform(1.5, 3)
        vx = cx + r * np.array([-1, 1, 1.2, -0.8])
        vy = cy + r * np.array([-1, -0.9, 1, 1.1])
        poly = PolygonPixelRegion(vertices=PixCoord(x=list(vx), y=list(vy)))
        sky = poly.to_sky(wcs)
        regions_hst.append(sky)
        regions_muse.append(sky)

    p = analysis.Photometry.__new__(analysis.Photometry)
    p.base_dir = str(tmp_path)
    p.gal = 'TESTGAL'
    p.hst_conv = hst_conv_dict
    p.jwst_conv = jwst_conv_dict
    p.regions_HST_final = regions_hst
    p.regions_MUSE_final = regions_muse
    p.final_table = Table({'id': np.arange(n_sources)})
    p.muse_data = np.zeros(muse_shape)
    p.muse_wcs = wcs
    p.log_wl = log_wl
    p.spectra = spectra
    p.muse_id_maps = (original_id_map, binned_id_map)
    p.eb_v = 0.05
    p._build_conv_dicts = lambda: (hst_conv_dict, jwst_conv_dict)
    p._find_galaxy_eb_v = lambda: None

    os.makedirs(os.path.join(str(tmp_path), 'TESTGAL'), exist_ok=True)
    return p, filter_wavelengths


def test_parallel_matches_sequential(synthetic_photometry):
    p, filter_wavelengths = synthetic_photometry

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p.run_spectro_photometry(filter_wavelengths, type_ap='HST', verbose=False, n_workers=1)
    seq_table = p.photometry_table.copy()

    p2 = analysis.Photometry.__new__(analysis.Photometry)
    p2.__dict__.update(p.__dict__)
    p2.photometry_table = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p2.run_spectro_photometry(filter_wavelengths, type_ap='HST', verbose=False, n_workers=3)
    par_table = p2.photometry_table

    assert seq_table.colnames == par_table.colnames
    for col in seq_table.colnames:
        assert np.allclose(seq_table[col], par_table[col], equal_nan=True), col
