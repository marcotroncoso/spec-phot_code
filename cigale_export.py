"""
Export native-resolution (v2) photometry to a CIGALE-format flux table.

One row per cluster: id, redshift (derived from the PHANGS-catalog
distance via the Hubble-law approximation, valid at these nearby
distances), then <cigale_filter> / <cigale_filter>_err pairs for every
band in CIGALE_FILTER_NAMES, in that fixed order — matching the layout
of CIGALE's own example flux tables (e.g. examples/sdss_qso/sdss_qso.txt
in the CIGALE manual).

Only the native/R65-aperture-corrected flux (flux_native_{band}) is
exported, not the convolved (v1) flux — v1 has no propagated flux error.

Non-detections (status_native_{band} == 'no_detection') are exported as
a negative flux — CIGALE's convention for an upper limit — using
flux_native_{band}_raw (the un-background-subtracted aperture flux) as
the limit value; this is the same quantity plot_native_photometry_
diagnostics' own SED panel already plots as a proxy upper-limit marker
for non-detections (see analysis.py's _draw_sed), reused here rather
than inventing a new upper-limit convention. Bands with no usable
measurement at all (band not observed for that source, or status in
{'no_flux', 'no_size_measurement', 'exception:*'}) are written as NaN
for both flux and error.
"""

import numpy as np
from astropy import constants as const
from astropy import units as u
from astropy.table import Table, vstack
from astropy.io import ascii as io_ascii

# band key (as used throughout analysis.py/native_phot_aux_func.py) ->
# CIGALE filter database name.
CIGALE_FILTER_NAMES = {
    'F275W': 'hst.wfc3.uvis2.F275W',
    'F336W': 'hst.wfc3.uvis2.F336W',
    'F438W': 'hst.wfc3.uvis2.F438W',
    'F555W': 'hst.wfc3.uvis2.F555W',
    'F814W': 'hst.wfc3.uvis2.F814W',
    'F200W': 'jwst.nircam.F200W',
    'F300M': 'jwst.nircam.F300M',
    'F335M': 'jwst.nircam.F335M',
    'F360M': 'jwst.nircam.F360M',
    'F770W': 'jwst.miri.F770W',
    'F1000W': 'jwst.miri.F1000W',
    'F1130W': 'jwst.miri.F1130W',
    'F2100W': 'jwst.miri.F2100W',
}

_DETECTED_STATUSES = ('ok', 'size_extrapolated', 'borrowed', 'psf_fallback')


def _to_float(value):
    """float(value), treating a masked table entry as NaN instead of
    letting numpy raise/warn on the conversion."""
    if value is np.ma.masked:
        return np.nan
    return float(value)


def _format_id(galaxy, cluster_id):
    """'{galaxy}_{cluster_id}', rendering a whole-valued float catalog id
    (ID_PHANGS_CLUSTER is stored as float64) as an integer string."""
    if isinstance(cluster_id, (float, np.floating)) and np.isfinite(cluster_id) \
            and cluster_id == int(cluster_id):
        cluster_id = int(cluster_id)
    return f'{galaxy}_{cluster_id}'


def hubble_redshift(distance_mpc, H0=70.0):
    """z = H0 * D / c (linear Hubble-law approximation) — adequate at
    PHANGS-sample distances (a few to ~20 Mpc, z <~ 0.01), where
    cosmological curvature terms are far below the precision distance
    itself carries.

    Parameters
    ----------
    distance_mpc : float or array — distance in Mpc.
    H0            : float — km/s/Mpc.
    """
    v = H0 * np.asarray(distance_mpc, dtype=float) * (u.km / u.s)
    return (v / const.c).to(u.dimensionless_unscaled).value


def photometry_row_to_cigale(row, bands=None):
    """
    Build the {cigale_filter: value, cigale_filter+'_err': err, ...}
    dict for one photometry_table row.

    Parameters
    ----------
    row   : astropy.table.Row (or anything supporting `col in row.colnames`
            and `row[col]`) — one row of a _photometry_v2.fits-style table.
    bands : iterable of band keys to consider; defaults to every key in
            CIGALE_FILTER_NAMES. A band absent from `row.colnames`
            (never ran native photometry for this source/band) yields
            NaN/NaN, same as a band that ran but failed outright.
    """
    if bands is None:
        bands = CIGALE_FILTER_NAMES.keys()

    out = {}
    for b in bands:
        cigale_name = CIGALE_FILTER_NAMES[b]
        flux_col = f'flux_native_{b}'
        raw_col = f'flux_native_{b}_raw'
        err_col = f'flux_native_{b}_err'
        status_col = f'status_native_{b}'

        value, err = np.nan, np.nan
        if status_col in row.colnames:
            status = str(row[status_col]).strip()
            err_val = _to_float(row[err_col]) if err_col in row.colnames else np.nan

            if status in _DETECTED_STATUSES and flux_col in row.colnames:
                flux_val = _to_float(row[flux_col])
                if np.isfinite(flux_val):
                    value, err = flux_val, err_val
            elif status == 'no_detection' and raw_col in row.colnames:
                raw_val = _to_float(row[raw_col])
                if np.isfinite(raw_val) and raw_val > 0:
                    value, err = -raw_val, err_val

        out[cigale_name] = value
        out[f'{cigale_name}_err'] = err
    return out


def build_cigale_table(tables, H0=70.0, id_col='ID_PHANGS_CLUSTER',
                        galaxy_col='PHANGS_GALAXY', distance_col='PHANGS_DMPC',
                        bands=None):
    """
    Assemble the full CIGALE input table from one or more photometry
    tables (typically one _photometry_v2.fits per galaxy).

    Parameters
    ----------
    tables       : astropy.table.Table or list of Table.
    H0           : float — km/s/Mpc, passed to hubble_redshift.
    id_col       : str — per-cluster catalog id column (combined with
                   `galaxy_col` to make a sample-wide-unique CIGALE id,
                   since cluster ids aren't guaranteed unique across
                   galaxies).
    galaxy_col   : str — per-row galaxy-name column.
    distance_col : str — per-row distance column, Mpc.
    bands        : iterable of band keys, or None for every
                   CIGALE_FILTER_NAMES key.

    Returns
    -------
    astropy.table.Table with columns id, redshift, <cigale_filter>,
    <cigale_filter>_err, ... (fixed band order).
    """
    if isinstance(tables, Table):
        tables = [tables]
    combined = vstack(list(tables), join_type='outer', metadata_conflicts='silent')

    if bands is None:
        bands = list(CIGALE_FILTER_NAMES.keys())

    ids = [_format_id(row[galaxy_col], row[id_col]) for row in combined]
    redshift = hubble_redshift(combined[distance_col], H0=H0)

    rows = [photometry_row_to_cigale(row, bands=bands) for row in combined]
    out = Table(rows=rows, names=[c for b in bands
                                   for c in (CIGALE_FILTER_NAMES[b], f'{CIGALE_FILTER_NAMES[b]}_err')])
    out.add_column(redshift, index=0, name='redshift')
    out.add_column(ids, index=0, name='id')
    return out


def write_cigale_ascii(table, path):
    """Write a CIGALE flux table to a whitespace-delimited ascii file
    with a '# id redshift ...'-style commented header, matching the
    layout of CIGALE's own example tables."""
    io_ascii.write(table, path, format='commented_header', delimiter=' ',
                    overwrite=True)
