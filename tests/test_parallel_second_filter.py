"""
Regression test for IsolatedSourceFinder.second_filter's opt-in
multiprocessing path (Phase 2 of the pipeline-hardening plan) — mirrors
test_parallel_photometry.py's approach for Photometry.run_spectro_photometry.
"""
import warnings

import numpy as np
from astropy.table import Table
from regions import PolygonPixelRegion, PixCoord

import analysis


def _make_isolated_source_finder(wcs, rng, cc_class_key='CLASS'):
    hst_data = rng.normal(50, 3, size=(400, 400))
    n_sources = 10
    ras, decs, regions = [], [], []
    for i in range(n_sources):
        cx, cy = rng.uniform(50, 350), rng.uniform(50, 350)
        r = rng.uniform(3, 6)
        hst_data[int(cy), int(cx)] += 300
        if i % 3 == 0:
            # inject a contaminating second peak for some sources so the
            # test exercises both the isolated and contaminated branches
            hst_data[int(cy) + 3, int(cx) + 3] += 250
        vx = cx + r * np.array([-1, 1, 1.2, -0.8])
        vy = cy + r * np.array([-1, -0.9, 1, 1.1])
        poly = PolygonPixelRegion(vertices=PixCoord(x=list(vx), y=list(vy)))
        regions.append(poly.to_sky(wcs))
        ras.append(10.0)
        decs.append(-20.0)

    final_table = Table({
        'PHANGS_RA': ras, 'PHANGS_DEC': decs,
        f'{cc_class_key}_ML': ['YCL'] * n_sources,
    })

    isf = analysis.IsolatedSourceFinder.__new__(analysis.IsolatedSourceFinder)
    isf.hst_data = hst_data
    isf.hst_wcs = wcs
    isf.regions_final = [regions, regions]
    isf.final_table = final_table
    isf.cc_class_key = cc_class_key
    return isf


def test_parallel_matches_sequential(wcs, rng):
    wcs.pixel_shape = (400, 400)

    isf_seq = _make_isolated_source_finder(wcs, np.random.default_rng(5))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        isf_seq.second_filter(n_workers=1)

    isf_par = _make_isolated_source_finder(wcs, np.random.default_rng(5))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        isf_par.second_filter(n_workers=3)

    assert np.array_equal(isf_seq.mask_isolated, isf_par.mask_isolated)
    # sanity: both isolated and contaminated sources should appear, so the
    # comparison isn't trivially true because every source landed the same way
    assert isf_seq.mask_isolated.any()
    assert not isf_seq.mask_isolated.all()
