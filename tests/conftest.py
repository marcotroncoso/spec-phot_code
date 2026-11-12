"""
Shared synthetic fixtures for the spec-phot pipeline test suite.

These build small, fully synthetic FITS-like WCS + image data so the tests
have no dependency on real HST/JWST/MUSE files being present, and run in
well under a second. They exist to pin down the exact numerical behavior of
the bbox-local aperture-mask rewrite (see phot_aux_func._bbox_local_mask_and_image
and aux_functions.is_isolated_in_region) against the full-image-mask approach
it replaced.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from astropy.wcs import WCS


@pytest.fixture
def wcs():
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crval = [10.0, -20.0]
    w.wcs.crpix = [50, 50]
    w.wcs.cdelt = [-0.0001, 0.0001]
    return w


@pytest.fixture
def rng():
    return np.random.default_rng(20260902)
