#!/usr/bin/env python
"""
Photometry of ANY source catalog on ANY set of HST / JWST images: no
IsolatedSourceFinder, no PHANGS DR5 catalog, no MUSE. One run = one field
(galaxy), configured by a small INI file (see catalog_run.example.ini):

  [catalog]  the table (FITS / ECSV / CSV), its RA / Dec / ID / class columns,
             the distance, the field name, optional Milky Way E(B-V)
  [images]   one line per band, BAND = path[, FITS extension]; and/or
             hst_root / jwst_root to find PHANGS-layout images of the field
  [run]      output folder, workers, which photometry, 2D modes, plots

Steps, as run_full_pipeline.py --catalog-mode:
  load images -> Photometry.load_custom_catalog -> keep sources on every image
  (require_all_bands) -> PSF FWHM (data/psf_table.csv, measured if missing)
  -> E(B-V) -> native + instrument photometry (native, instG, instP)
  -> optional 2D model fits (run_2d_photometry) -> optional per-source plots
  -> optional CIGALE table

Outputs in <output_dir>/<name>/, the same files as the other modes:
  <name>_photometry_native.fits, _instrument.fits (caches), _photometry_v2.fits
  (everything), _photometry_2d.fits (2D fits), native_slit_profiles/ and
  photometry_2d_plots/ (plots), pipeline.log, run_config.ini (copy of the config)

    python run_catalog_photometry.py --config my_run.ini [--max-sources 5] [--dry-run]
"""
import argparse
import configparser
import logging
import os
import shutil
import sys
import warnings

import numpy as np

warnings.simplefilter('ignore')
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import astropy.units as u  # noqa: E402
from astropy.coordinates import SkyCoord  # noqa: E402
from astropy.io import fits  # noqa: E402
from astropy.wcs import WCS  # noqa: E402

from analysis import Photometry  # noqa: E402
import run_full_pipeline as rfp  # noqa: E402

logger = logging.getLogger('run_catalog_photometry')
HST_BANDS = list(rfp.FILTER_WAVELENGTHS)[:5]          # FILTER_WAVELENGTHS: 5 HST bands, then JWST


def load_run_config(path):
    cp = configparser.ConfigParser(inline_comment_prefixes=('#', ';'))
    cp.optionxform = str                                   # keep band names as written
    if not cp.read(path):
        raise FileNotFoundError(f'config file not found: {path}')
    if cp.has_section('paths') and not cp.has_section('catalog'):
        raise SystemExit(f'{path} is a run_full_pipeline.py config ([paths], [galaxies], ...), not a mode C '
                         'config ([catalog] + [images], see catalog_run.example.ini). Run it with\n'
                         f'    python run_full_pipeline.py --config {path} [--catalog-mode] [--dry-run]')
    cfg_dir = os.path.dirname(os.path.abspath(path))

    def _p(v):
        v = v.strip()
        return os.path.join(cfg_dir, v) if v and not os.path.isabs(v) else v

    def _get(sec, key, fallback=''):
        return cp.get(sec, key, fallback=fallback).strip()

    def _float(sec, key):
        v = _get(sec, key)
        return float(v) if v else None

    images = {}
    roots = {}
    for key, val in (cp.items('images') if cp.has_section('images') else []):
        if key.lower() in ('hst_root', 'jwst_root'):
            roots[key.lower()] = _p(val)
            continue
        parts = [s.strip() for s in val.split(',')]
        band = key.upper()
        if band not in rfp.FILTER_WAVELENGTHS:
            raise ValueError(f'[images] {key}: unsupported band (supported: {", ".join(rfp.FILTER_WAVELENGTHS)})')
        ext = parts[1] if len(parts) > 1 and parts[1] else None
        images[band] = (_p(parts[0]), int(ext) if ext and ext.isdigit() else ext)

    name = _get('catalog', 'name')
    if not name:
        raise ValueError('[catalog] name is required (output folder, PSF-table row)')
    return dict(
        catalog=_p(_get('catalog', 'file')), name=name, fmt=_get('catalog', 'format') or None,
        ra_col=_get('catalog', 'ra_col', 'RA'), dec_col=_get('catalog', 'dec_col', 'DEC'),
        id_col=_get('catalog', 'id_col') or None, class_col=_get('catalog', 'class_col') or None,
        distance_mpc=_float('catalog', 'distance_mpc'), dist_col=_get('catalog', 'dist_col') or None,
        ebv=_float('catalog', 'ebv'),
        images=images, roots=roots,
        output_dir=_p(_get('run', 'output_dir', 'output_custom')),
        psf_filepath=_p(_get('run', 'psf_filepath', os.path.join(_HERE, 'data', 'psf_table.csv'))),
        n_workers=cp.getint('run', 'n_workers', fallback=4),
        instrument_fit=cp.getboolean('run', 'instrument_fit', fallback=True),
        instrument_slit_width_px=cp.getint('run', 'instrument_slit_width_px', fallback=1),
        require_all_bands=cp.getboolean('run', 'require_all_bands', fallback=True),
        force_recompute=cp.getboolean('run', 'force_recompute', fallback=False),
        modes_2d=[m for m in _get('run', 'modes_2d').replace(',', ' ').split() if m],
        plots=_get('run', 'plots', 'none').lower(),
        cigale_export=cp.getboolean('run', 'cigale_export', fallback=False),
        profile_by_class=rfp.read_profile_section(cp),
        psf_blur=_get('run', 'psf_extra_blur', 'none') or 'none',
        psf_source=(_get('run', 'psf_source', 'theoretical') or 'theoretical').lower(),
        cigale_h0=cp.getfloat('run', 'cigale_h0', fallback=70.0),
    )


def image_paths(cfg):
    """{band: (path, ext)}: PHANGS-layout images under hst_root / jwst_root
    (as run_full_pipeline finds them), overridden by explicit [images] entries."""
    paths = {}
    if cfg['roots'].get('hst_root'):
        paths.update({b: (f, None) for b, f in rfp.find_hst_band_paths(cfg['roots']['hst_root'], cfg['name']).items()})
    if cfg['roots'].get('jwst_root'):
        paths.update({b: (f, None) for b, f in
                      rfp.find_jwst_band_paths(cfg['roots']['jwst_root'], cfg['name'].lower()).items()})
    paths.update(cfg['images'])
    return {b: v for b, v in paths.items() if b in rfp.FILTER_WAVELENGTHS}


def load_images(p, paths):
    """Fill p.hst_images / p.jwst_images ((data, header, wcs) per lower-case band)
    as Photometry.load_images does, with a selectable FITS extension per band
    (default: HST the primary HDU, JWST 'SCI' or else extension 1). HST zeros
    become NaN (no-data), as in Photometry.load_images."""
    for band, (path, ext) in paths.items():
        is_hst = band in HST_BANDS
        with fits.open(path) as h:
            if ext is None:
                ext = 0 if is_hst else ('SCI' if 'SCI' in h else 1)
            if h[ext].data is None or np.ndim(h[ext].data) != 2:
                raise ValueError(f'{band}: extension {ext!r} of {path} holds no 2-D image — '
                                 f'give the right one as "{band} = path, EXT"')
            data, hdr = np.asarray(h[ext].data, float), h[ext].header.copy()
        if is_hst:
            data = np.where(data == 0.0, np.nan, data)
            p.hst_images[band.lower()] = (data, hdr, WCS(hdr))
        else:
            p.jwst_images[band.lower()] = (data, hdr, WCS(hdr))
        logger.info(f'  {"HST" if is_hst else "JWST"} {band:7s} {os.path.basename(path)} [{ext}]  shape={data.shape}')


def galaxy_ebv(p, cfg):
    """Milky Way E(B-V): the config value, else SFD at the catalog's centre."""
    if cfg['ebv'] is not None:
        return cfg['ebv']
    try:
        from dustmaps.sfd import SFDQuery
        t = p.final_table
        c = SkyCoord(np.nanmedian(np.asarray(t['PHANGS_RA'], float)) * u.deg,
                     np.nanmedian(np.asarray(t['PHANGS_DEC'], float)) * u.deg)
        return float(SFDQuery()(c))
    except Exception as e:
        logger.warning(f'  E(B-V) from SFD failed ({e}) — using 0; set [catalog] ebv to fix')
        return 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True, help='run config (see catalog_run.example.ini)')
    ap.add_argument('--max-sources', type=int, default=None, help='only the first N sources (test run)')
    ap.add_argument('--dry-run', action='store_true',
                    help='check the catalogue, images, settings and resources; measures and writes nothing')
    a = ap.parse_args()
    cfg = load_run_config(a.config)
    if a.dry_run:
        import preflight
        sys.exit(0 if preflight.check_catalog_run(cfg, image_paths(cfg), a.max_sources) else 1)
    name = cfg['name']
    outd = os.path.join(cfg['output_dir'], name)
    os.makedirs(outd, exist_ok=True)
    shutil.copy(a.config, os.path.join(outd, 'run_config.ini'))

    fmt = logging.Formatter('%(asctime)s %(message)s')
    stream_h, file_h = logging.StreamHandler(sys.stdout), logging.FileHandler(os.path.join(outd, 'pipeline.log'), mode='w')
    for h in (stream_h, file_h):
        h.setFormatter(fmt)
    for lg in (logger, logging.getLogger('analysis')):
        lg.setLevel(logging.INFO)
        if not lg.handlers:                    # analysis.py already prints its own messages
            lg.addHandler(stream_h)
        lg.addHandler(file_h)

    paths = image_paths(cfg)
    if not paths:
        raise RuntimeError('no images: give [images] BAND = path lines or hst_root / jwst_root')
    logger.info(f'=== {name}: {len(paths)} bands, catalog {cfg["catalog"]} ===')

    p = Photometry([name, name], base_dir=cfg['output_dir'], psf_filepath=cfg['psf_filepath'],
                   psf_source=cfg['psf_source'])
    load_images(p, paths)
    p.load_custom_catalog(cfg['catalog'], cfg['ra_col'], cfg['dec_col'], id_col=cfg['id_col'],
                          class_col=cfg['class_col'], distance_mpc=cfg['distance_mpc'],
                          dist_col=cfg['dist_col'], fmt=cfg['fmt'])
    if a.max_sources:
        p.final_table = p.final_table[:a.max_sources]
    if cfg['require_all_bands']:
        p.filter_by_common_fov(plot=False, include_muse=False)
    if len(p.final_table) == 0:
        raise RuntimeError('no source falls on every image (set require_all_bands = False to keep them)')
    p.load_or_measure_psf()
    p.eb_v = galaxy_ebv(p, cfg)
    logger.info(f'  Milky Way E(B-V) = {p.eb_v:.4f}')

    bands = {b: v for b, v in rfp.FILTER_WAVELENGTHS.items() if b.lower() in p.hst_images or b.lower() in p.jwst_images}
    p.init_photometry_table_from_catalog()
    p.measure_native_psf_shape()
    logger.info(f'Native + instrument photometry of {len(p.photometry_table)} sources, {len(bands)} bands, '
                f'{cfg["n_workers"]} workers ...')
    p.run_native_photometry(bands, n_workers=cfg['n_workers'], verbose=False, force_recompute=cfg['force_recompute'],
                            plot_diagnostics=False, instrument_fit=cfg['instrument_fit'],
                            instrument_slit_width_px=cfg['instrument_slit_width_px'],
                            profile_by_class=cfg['profile_by_class'],
                            psf_blur=cfg['psf_blur'],
                            dic_passbands={b: rfp.DIC_PASSBANDS[b] for b in bands if b in rfp.DIC_PASSBANDS})
    p.save_photometry(suffix='photometry_v2')
    v2_path = os.path.join(outd, f'{name}_photometry_v2.fits')

    if cfg['modes_2d']:
        if not cfg['instrument_fit']:
            raise RuntimeError('modes_2d needs instrument_fit = True (the 2D fits start from instP)')
        import run_2d_photometry as rd
        t2d = rd.run_2d_photometry(p, bands, modes=tuple(cfg['modes_2d']), n_workers=cfg['n_workers'])
        rd.merge_2d_into_v2(v2_path, t2d)

    if cfg['plots'] in ('slits', 'all'):
        for i in range(len(p.photometry_table)):
            p.plot_native_slit_profiles(bands, idd=i, plot_2d=False, verbose=False)
        logger.info(f'  slit-profile plots -> {os.path.join(outd, "native_slit_profiles")}')
    if cfg['plots'] in ('2d', 'all') and cfg['modes_2d']:
        import run_2d_photometry as rd
        for mode in cfg['modes_2d']:
            rd.plot_2d_photometry(p, bands, mode=mode, verbose=False)
        logger.info(f'  2D plots -> {os.path.join(outd, "photometry_2d_plots")}')

    if cfg['cigale_export']:
        import cigale_export
        cig = cigale_export.build_cigale_table(p.photometry_table, H0=cfg['cigale_h0'])
        cigale_export.write_cigale_ascii(cig, os.path.join(outd, f'{name}_cigale_input.txt'))

    logger.info(f'=== {name} done: {v2_path} ===')


if __name__ == '__main__':
    main()
