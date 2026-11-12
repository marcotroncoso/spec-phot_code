"""
Unit tests for cigale_export.py — the CIGALE-format flux-table exporter
built on top of the native-resolution (v2) photometry columns.

These are deliberately independent of any real photometry run: each test
builds a minimal astropy Table row (or two) with just the columns
photometry_row_to_cigale/build_cigale_table actually read, so they pin
down the export *rules* (detected vs. upper-limit vs. missing) rather
than re-testing the photometry measurement itself (covered by
test_native_photometry.py / test_native_photometry_integration.py).
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from astropy import constants as const
from astropy import units as u
from astropy.table import Table

import cigale_export as ce


class TestHubbleRedshift:

    def test_matches_direct_formula(self):
        D = 70.0  # Mpc
        H0 = 70.0
        expected = (H0 * D * (u.km / u.s) / const.c).to(u.dimensionless_unscaled).value
        assert ce.hubble_redshift(D, H0=H0) == pytest.approx(expected)

    def test_monotonic_in_distance(self):
        z_near = ce.hubble_redshift(5.0)
        z_far = ce.hubble_redshift(20.0)
        assert 0 < z_near < z_far

    def test_scales_with_H0(self):
        z_lowH0 = ce.hubble_redshift(10.0, H0=60.0)
        z_hiH0 = ce.hubble_redshift(10.0, H0=80.0)
        assert z_lowH0 < z_hiH0


def _row(**cols):
    """One-row astropy Table -> its single Row (what photometry_table[i]
    actually hands the exporter)."""
    return Table(rows=[list(cols.values())], names=list(cols.keys()))[0]


class TestPhotometryRowToCigale:

    def test_detected_band_uses_bkg_subtracted_flux(self):
        row = _row(status_native_F555W='ok',
                   flux_native_F555W=1.5e-3,
                   flux_native_F555W_raw=1.6e-3,
                   flux_native_F555W_err=2.0e-5)
        out = ce.photometry_row_to_cigale(row, bands=['F555W'])
        assert out['hst.wfc3.uvis2.F555W'] == pytest.approx(1.5e-3)
        assert out['hst.wfc3.uvis2.F555W_err'] == pytest.approx(2.0e-5)

    @pytest.mark.parametrize('status', ['ok', 'size_extrapolated', 'borrowed', 'psf_fallback'])
    def test_every_detected_status_is_treated_as_a_normal_measurement(self, status):
        row = _row(status_native_F555W=status,
                   flux_native_F555W=2.0e-3,
                   flux_native_F555W_raw=2.1e-3,
                   flux_native_F555W_err=1.0e-5)
        out = ce.photometry_row_to_cigale(row, bands=['F555W'])
        assert out['hst.wfc3.uvis2.F555W'] == pytest.approx(2.0e-3)
        assert out['hst.wfc3.uvis2.F555W_err'] == pytest.approx(1.0e-5)

    def test_no_detection_becomes_negative_upper_limit_from_raw_flux(self):
        row = _row(status_native_F275W='no_detection',
                   flux_native_F275W=np.nan,
                   flux_native_F275W_raw=4.0e-4,
                   flux_native_F275W_err=6.0e-5)
        out = ce.photometry_row_to_cigale(row, bands=['F275W'])
        assert out['hst.wfc3.uvis2.F275W'] == pytest.approx(-4.0e-4)
        assert out['hst.wfc3.uvis2.F275W_err'] == pytest.approx(6.0e-5)

    def test_no_detection_with_nonpositive_raw_flux_is_nan(self):
        row = _row(status_native_F275W='no_detection',
                   flux_native_F275W=np.nan,
                   flux_native_F275W_raw=-1.0e-5,
                   flux_native_F275W_err=6.0e-5)
        out = ce.photometry_row_to_cigale(row, bands=['F275W'])
        assert np.isnan(out['hst.wfc3.uvis2.F275W'])
        assert np.isnan(out['hst.wfc3.uvis2.F275W_err'])

    @pytest.mark.parametrize('status', ['no_flux', 'no_size_measurement', 'exception:boom'])
    def test_hard_failure_statuses_are_nan_even_with_a_positive_raw_flux(self, status):
        # A hard failure has no trustworthy aperture at all — a leftover
        # raw-flux number (if any) must not leak through as an upper
        # limit the way a genuine 'no_detection' SNR-gate failure does.
        row = _row(status_native_F1000W=status,
                   flux_native_F1000W=np.nan,
                   flux_native_F1000W_raw=3.0e-4,
                   flux_native_F1000W_err=np.nan)
        out = ce.photometry_row_to_cigale(row, bands=['F1000W'])
        assert np.isnan(out['jwst.miri.F1000W'])
        assert np.isnan(out['jwst.miri.F1000W_err'])

    def test_band_never_run_is_nan(self):
        row = _row(status_native_F555W='ok', flux_native_F555W=1.0e-3,
                   flux_native_F555W_raw=1.0e-3, flux_native_F555W_err=1.0e-5)
        out = ce.photometry_row_to_cigale(row, bands=['F555W', 'F2100W'])
        assert np.isnan(out['jwst.miri.F2100W'])
        assert np.isnan(out['jwst.miri.F2100W_err'])

    def test_masked_error_value_is_nan_not_a_warning_or_crash(self):
        t = Table(rows=[['ok', 1.0e-3, 1.0e-3, 1.0e-5]],
                  names=['status_native_F555W', 'flux_native_F555W',
                         'flux_native_F555W_raw', 'flux_native_F555W_err'],
                  masked=True)
        t['flux_native_F555W_err'].mask = [True]
        out = ce.photometry_row_to_cigale(t[0], bands=['F555W'])
        assert out['hst.wfc3.uvis2.F555W'] == pytest.approx(1.0e-3)
        assert np.isnan(out['hst.wfc3.uvis2.F555W_err'])


class TestBuildCigaleTable:

    def _galaxy_table(self, galaxy, cluster_ids, distance_mpc):
        n = len(cluster_ids)
        return Table({
            'ID_PHANGS_CLUSTER': np.array(cluster_ids, dtype=float),
            'PHANGS_GALAXY': [galaxy] * n,
            'PHANGS_DMPC': [distance_mpc] * n,
            'status_native_F555W': ['ok'] * n,
            'flux_native_F555W': np.full(n, 1.0e-3),
            'flux_native_F555W_raw': np.full(n, 1.0e-3),
            'flux_native_F555W_err': np.full(n, 1.0e-5),
        })

    def test_single_galaxy_ids_and_redshift(self):
        t = self._galaxy_table('NGC1365', [57, 60], distance_mpc=19.0)
        out = ce.build_cigale_table(t, bands=['F555W'])
        assert list(out['id']) == ['NGC1365_57', 'NGC1365_60']
        assert out['redshift'][0] == pytest.approx(ce.hubble_redshift(19.0))
        assert out['hst.wfc3.uvis2.F555W'][0] == pytest.approx(1.0e-3)

    def test_multiple_galaxies_with_different_band_coverage_are_stacked(self):
        # Second galaxy never ran F275W at all — build_cigale_table must
        # not choke on the missing column (outer-join vstack), and rows
        # from that galaxy must report NaN for it.
        t1 = self._galaxy_table('NGC1365', [1], distance_mpc=19.0)
        t1['status_native_F275W'] = ['ok']
        t1['flux_native_F275W'] = [5.0e-4]
        t1['flux_native_F275W_raw'] = [5.0e-4]
        t1['flux_native_F275W_err'] = [1.0e-5]

        t2 = self._galaxy_table('NGC4254', [2], distance_mpc=16.8)

        out = ce.build_cigale_table([t1, t2], bands=['F275W', 'F555W'])
        assert list(out['id']) == ['NGC1365_1', 'NGC4254_2']
        assert out['hst.wfc3.uvis2.F275W'][0] == pytest.approx(5.0e-4)
        assert np.isnan(out['hst.wfc3.uvis2.F275W'][1])
        assert out['hst.wfc3.uvis2.F555W'][1] == pytest.approx(1.0e-3)

    def test_column_order_is_id_redshift_then_bands_in_order(self):
        t = self._galaxy_table('NGC1365', [1], distance_mpc=19.0)
        out = ce.build_cigale_table(t, bands=['F555W'])
        assert out.colnames == ['id', 'redshift', 'hst.wfc3.uvis2.F555W',
                                'hst.wfc3.uvis2.F555W_err']


class TestWriteCigaleAscii:

    def test_writes_one_table_with_all_rows(self, tmp_path):
        t = Table({
            'ID_PHANGS_CLUSTER': np.array([1.0, 2.0]),
            'PHANGS_GALAXY': ['NGC1365', 'NGC1365'],
            'PHANGS_DMPC': [19.0, 19.0],
            'status_native_F555W': ['ok', 'ok'],
            'flux_native_F555W': [1.0e-3, 2.0e-3],
            'flux_native_F555W_raw': [1.0e-3, 2.0e-3],
            'flux_native_F555W_err': [1.0e-5, 2.0e-5],
        })
        cig = ce.build_cigale_table(t, bands=['F555W'])
        path = str(tmp_path / 'NGC1365_cigale_input.txt')
        ce.write_cigale_ascii(cig, path)

        readback = Table.read(path, format='ascii.commented_header')
        assert len(readback) == 2
        assert list(readback['id']) == ['NGC1365_1', 'NGC1365_2']
        assert readback['hst.wfc3.uvis2.F555W'][1] == pytest.approx(2.0e-3)
