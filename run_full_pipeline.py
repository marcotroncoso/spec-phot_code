#!/usr/bin/env python
"""
Batch driver: run the full IsolatedSourceFinder + Photometry (v1 + v2)
pipeline for every galaxy in the sample, config-file driven, one galaxy
per worker process. Each galaxy then (optionally) writes its own
CIGALE input table (one row per source) into its own output directory.

Usage
-----
    python run_full_pipeline.py [--config config.ini] [--n-workers N] [--galaxy TOKEN]
                                [--catalog-mode] [--qa-only] [--no-qa] [--dry-run]

--dry-run checks the inputs, settings and resources (preflight.py) and
lists what each galaxy would compute or reuse, without running anything.

Per galaxy: ISF -> v1 + native + instP photometry -> [run_2d] 2D model fits
(+ [2d] plots) -> CIGALE -> [run_spectra] MUSE M1/M2 spectra -> [run_sed]
injection test + SED plots (photometry + spectra); after a batch
run, [qa] run_qa makes the pooled QA plots (plot_qa_diagnostics). Every
switch and setting is in the config file.

--galaxy runs a single galaxy synchronously (no process pool) — useful
for testing/debugging one galaxy before launching the full batch.

Parallelization model
----------------------
Per-source parallelism already exists *inside* IsolatedSourceFinder.
second_filter / Photometry.run_spectro_photometry / run_native_photometry
(each takes its own n_workers). Nesting another process pool on top of
that would oversubscribe CPUs, so this driver instead parallelizes
*across galaxies*: `n_workers` here is the number of galaxies processed
concurrently, each running fully sequentially internally. With ~19-22
independent galaxies in this sample that is already embarrassingly
parallel, and it keeps one galaxy's failure from affecting any other.

Memory
------
Each galaxy's Photometry object holds several large image arrays at
once (13 bands, native + convolved copies, plus the MUSE cube) — easily
multiple GB for the bigger galaxies. Two things keep that from
accumulating across a batch run: the worker pool is created with
`max_tasks_per_child=1`, so every galaxy gets a *fresh* process — CPython
requires the 'forkserver' or 'spawn' start method for this (plain 'fork'
raises ValueError), so this uses 'forkserver': a small persistent helper
process is started once, and each worker is a cheap fork() *of that
helper* rather than a from-scratch interpreter spawn — that's torn
down — returning all its memory to the OS — the moment that galaxy
finishes, regardless of how long its neighbors take; and within a single
galaxy's own run, the convolved-image dicts (only needed by v1/
convolved photometry) are freed right after that step, before the
(usually longer) native-photometry + plotting stages run. Each galaxy's
peak RSS is logged (see `peak_rss_gb` in its pipeline.log and in the
per-galaxy result dict) so `n_workers` can be sized from real numbers:
worst case ≈ n_workers × the largest single-galaxy peak_rss_gb.
"""
import argparse
import configparser
import gc
import glob
import logging
import multiprocessing as mp
import os
import re
import resource
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.simplefilter('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis import IsolatedSourceFinder, Photometry
import cigale_export

# Same 13-band dict used throughout this pipeline's diagnostic scripts —
# [central_wavelength_AA, R_value] per band, first 5 entries HST then
# JWST (order matters: phot_aux_func.measure_photometry slices bands
# positionally). Edit here if the band set ever changes.
FILTER_WAVELENGTHS = {
    'F275W': [2742.5, 5.487], 'F336W': [3366.4, 4.453], 'F438W': [4335.3, 3.623],
    'F555W': [5302.8, 2.855], 'F814W': [7983.1, 1.441],
    'F200W': [19700, 0.267], 'F300M': [29800, 0.229], 'F335M': [33500, 0.178], 'F360M': [36100, 0.147],
    'F770W': [76690, 0], 'F1000W': [99620, 0], 'F1130W': [113070, 0], 'F2100W': [209070, 0],
}

# band -> (wl_min_AA, wl_max_AA), for plot_native_photometry_diagnostics'
# filter-width SED spans.
DIC_PASSBANDS = {
    'F275W': (2400, 3100), 'F336W': (3100, 3600), 'F438W': (4000, 4700),
    'F555W': (4700, 6000), 'F814W': (6900, 9600),
    'F200W': (17600, 22300), 'F300M': (28300, 31300), 'F335M': (31700, 35300),
    'F360M': (34400, 37800), 'F770W': (65200, 88000), 'F1000W': (91100, 108800),
    'F1130W': (108000, 118400), 'F2100W': (188400, 231400),
}

_GALAXY_TOKEN_RE = re.compile(r'^([A-Z0-9]+)-([0-9.]+)asec$')

logger = logging.getLogger('run_full_pipeline')
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(_h)


# ─────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────

def read_profile_section(parser):
    """profile_by_class from the optional [profile] section of a config
    (defaults: EFF gamma=3 fixed for YCL / MAP / unclassified, King c=1.5 for
    OGCC — the profiles used before the section existed)."""
    from native_phot_aux_func import profile_by_class_from_settings
    sec = 'profile'
    return profile_by_class_from_settings(
        eff_gamma=parser.getfloat(sec, 'eff_gamma', fallback=3.0),
        eff_gamma_free=parser.getboolean(sec, 'eff_gamma_free', fallback=False),
        eff_gamma_min=parser.getfloat(sec, 'eff_gamma_min', fallback=2.0),
        eff_gamma_max=parser.getfloat(sec, 'eff_gamma_max', fallback=6.0),
        ogcc_profile=parser.get(sec, 'ogcc_profile', fallback='king'),
        king_c=parser.getfloat(sec, 'king_c', fallback=1.5),
    )


MODES_2D = ('fixed', 'free', 'Gfixed', 'Gfree')
REDO_STEPS = ('isf', 'photometry', '2d', 'spectra', 'injection', 'sed', 'plots')


def _read_redo(parser):
    """[run] redo (steps to recompute even when up to date) + the old
    force_recompute = True, which means all of them"""
    raw = parser.get('run', 'redo', fallback='').replace(',', ' ').split()
    if parser.getboolean('run', 'force_recompute', fallback=False) or 'all' in raw:
        return set(REDO_STEPS)
    bad = [r for r in raw if r not in REDO_STEPS]
    if bad:
        raise ValueError(f'[run] redo: unknown {bad}; choose from {REDO_STEPS} or all')
    return set(raw)


def _read_2d_section(parser):
    """[2d] section: settings of the 2D model fits (run_2d_photometry)"""
    modes = parser.get('2d', 'modes', fallback='fixed free').replace(',', ' ').split()
    bad = [m for m in modes if m not in MODES_2D]
    if bad:
        raise ValueError(f'[2d] modes: unknown {bad}; choose from {MODES_2D}')
    return {'modes': tuple(modes),
            'n_workers': parser.getint('2d', 'n_workers', fallback=4),
            'plots': parser.getboolean('2d', 'plots', fallback=False),
            'all_bands': parser.getboolean('2d', 'all_bands', fallback=False),
            'overwrite': parser.getboolean('2d', 'overwrite', fallback=False)}


def load_config(path):
    parser = configparser.ConfigParser(inline_comment_prefixes=('#', ';'))
    if not parser.read(path):
        raise FileNotFoundError(f'config file not found: {path}')
    if parser.has_section('catalog') and not parser.has_section('paths'):
        raise SystemExit(f'{path} is a run_catalog_photometry.py config (mode C: [catalog] + [images]), not a '
                         'pipeline config ([paths], [galaxies], ...). Run it with\n'
                         f'    python run_catalog_photometry.py --config {path} [--dry-run]')
    cfg_dir = os.path.dirname(os.path.abspath(path))

    def _path(key, fallback):
        # relative [paths] entries are relative to the config file, not the working directory
        v = parser.get('paths', key, fallback=fallback).strip()
        return os.path.join(cfg_dir, v) if v and not os.path.isabs(v) else v

    return {
        'hst_root': _path('hst_root', ''),
        'jwst_root': _path('jwst_root', ''),
        'muse_dap_root': _path('muse_dap_root', ''),
        'cat_path': _path('cat_path', ''),
        'psf_filepath': _path('psf_filepath', 'data/psf_table.csv'),
        'output_dir': _path('output_dir', 'output'),
        'catalog_output_dir': _path('catalog_output_dir', ''),
        'validation_output_dir': _path('validation_output_dir', 'validation/output'),

        'galaxy_list': parser.get('galaxies', 'list', fallback='auto'),
        'sample': parser.get('galaxies', 'sample', fallback='PM'),

        'run_isf': parser.getboolean('run', 'run_isf', fallback=True),
        'force_recompute': parser.getboolean('run', 'force_recompute', fallback=False),
        'redo': _read_redo(parser),
        'plot_diagnostics': parser.getboolean('run', 'plot_diagnostics', fallback=True),
        'instrument_fit': parser.getboolean('run', 'instrument_fit', fallback=True),
        'instrument_slit_width_px': parser.getint('run', 'instrument_slit_width_px', fallback=1),
        'n_workers': parser.get('run', 'n_workers', fallback='').strip(),
        'psf_blur': parser.get('run', 'psf_extra_blur', fallback='none').strip() or 'none',
        'psf_source': parser.get('run', 'psf_source', fallback='theoretical').strip().lower() or 'theoretical',
        'source_catalog': parser.get('run', 'source_catalog', fallback='isf').strip().lower(),
        'catalog_selection': parser.get('run', 'catalog_selection', fallback='machine').strip().lower(),
        'catalog_classes': tuple(c.strip() for c in parser.get('run', 'catalog_classes', fallback='1+2').split(',')
                                 if c.strip()),
        'catalog_n_workers': parser.getint('run', 'catalog_n_workers', fallback=8),
        'catalog_drop_outside': parser.getboolean('run', 'catalog_drop_outside', fallback=False),

        'cigale_export': parser.getboolean('cigale', 'export', fallback=True),
        'cigale_ext': parser.get('cigale', 'ext', fallback='txt'),
        'cigale_h0': parser.getfloat('cigale', 'h0', fallback=70.0),

        'profile_by_class': read_profile_section(parser),

        # optional steps after the photometry, and the QA plots after a batch
        'run_2d': parser.getboolean('run', 'run_2d', fallback=False),
        'run_spectra': parser.getboolean('run', 'run_spectra', fallback=False),
        'run_sed': parser.getboolean('run', 'run_sed', fallback=False),
        'sed': {
            'injection': parser.getboolean('sed', 'injection', fallback=True),
            'n_per_env': parser.getint('sed', 'n_per_env', fallback=10),
            'n_workers': parser.getint('sed', 'n_workers', fallback=4),
            'overwrite': parser.getboolean('sed', 'overwrite', fallback=False),
        },
        '2d': _read_2d_section(parser),
        'qa': {
            'run_qa': parser.getboolean('qa', 'run_qa', fallback=True),
            'qa_dir': (lambda v: os.path.join(cfg_dir, v) if v and not os.path.isabs(v) else v)(
                parser.get('qa', 'qa_dir', fallback='').strip()),
            'plots': parser.get('qa', 'plots', fallback='all').strip() or 'all',
        },
        'config_path': os.path.abspath(path),
    }


def catalog_output_dir(cfg):
    """Output root of catalog mode: [paths] catalog_output_dir, default
    <output_dir>_catalog — kept apart so its caches never mix with the
    ISF-selection run (same file names, different source lists)."""
    return cfg['catalog_output_dir'] or (cfg['output_dir'].rstrip('/') + '_catalog')


def discover_galaxies(cfg):
    """Galaxy tokens (e.g. 'NGC1365-1.15asec') to process.

    'auto' scans muse_dap_root for '<GAL>-<X.XX>asec' subdirectories
    (that naming already exists on disk for every galaxy with MUSE DAP
    products) and keeps only those that also have an HST directory
    (hst_root/<GAL>) and a JWST directory (jwst_root/<gal_lower>).
    """
    raw = cfg['galaxy_list'].strip()
    if raw.lower() != 'auto':
        return [t.strip() for t in raw.split(',') if t.strip()]

    tokens = []
    for name in sorted(os.listdir(cfg['muse_dap_root'])):
        m = _GALAXY_TOKEN_RE.match(name)
        if not m:
            continue
        gal = m.group(1)
        gal_lower = gal.lower()
        if not os.path.isdir(os.path.join(cfg['hst_root'], gal)):
            logger.warning(f'{name}: no HST directory under {cfg["hst_root"]}/{gal} — skipping')
            continue
        if not os.path.isdir(os.path.join(cfg['jwst_root'], gal_lower)):
            logger.warning(f'{name}: no JWST directory under {cfg["jwst_root"]}/{gal_lower} — skipping')
            continue
        tokens.append(name)
    return tokens


def find_hst_band_paths(hst_root, gal):
    """{band: filepath} for one galaxy's HST anchor images.

    Globs broadly (not embedding the galaxy name in the pattern) since
    some galaxies' filenames carry a suffix on the galaxy name itself
    (e.g. NGC4254's files are named '..._ngc4254mosaic_...'); the band
    is always the second-to-last '_'-separated token regardless.

    A few early-cycle PHANGS-HST targets (e.g. NGC0628) were observed
    in F435W rather than F438W for the same "B-band" filter slot —
    close enough in wavelength to treat as interchangeable here. Left
    as F435W, it isn't a key in FILTER_WAVELENGTHS, so it silently
    drops out of that galaxy's `bands` dict entirely; since
    phot_aux_func.measure_photometry splits HST/JWST by *position*
    (bands[:5]/bands[5:]), losing one HST entry shifts every band after
    it, making the first JWST band get misread as HST and crash with a
    KeyError for every source. Normalizing to F438W here avoids that.
    """
    paths = {}
    for f in glob.glob(os.path.join(hst_root, gal, '*_anchor.fits')):
        band = os.path.basename(f).split('_')[5].upper()
        if band == 'F435W':
            band = 'F438W'
        paths[band] = f
    return paths


def find_jwst_band_paths(jwst_root, gal_lower):
    """{band: filepath} for one galaxy's JWST anchor images, under the
    PHANGS v4p1/2107 release layout:
    <jwst_root>/<gal_lower>/images/hlsp_2107_jwst_<instrument>_<gal_lower>_<band>_v4p1_anchor-at<band>.fits
    (that directory also holds a '<gal>_2107_jwst_images.tar.gz' — the
    '*.fits' glob already excludes it)."""
    paths = {}
    for f in glob.glob(os.path.join(jwst_root, gal_lower, 'images', '*.fits')):
        band = os.path.basename(f).split('_')[5].upper()
        paths[band] = f
    return paths


# ─────────────────────────────────────────────────────────────────────────
# Per-galaxy worker
# ─────────────────────────────────────────────────────────────────────────

def _redo(cfg, step):
    return step in cfg.get('redo', ())


def _isf_current(matched_path, cfg):
    """(True, '') when the ISF products exist and were made with the current
    PSF source and MUSE sample (header keys PSFSRC / SAMPLE; products written
    before those keys existed are taken as current), else (False, reason)"""
    if not os.path.exists(matched_path):
        return False, 'products missing'
    from astropy.io import fits
    h = fits.getheader(matched_path, 0)
    for key, want in (('PSFSRC', cfg['psf_source']), ('SAMPLE', cfg['sample'])):
        have = h.get(key)
        if have is not None and str(have) != str(want):
            return False, f'made with {key}={have}, config has {want}'
    return True, ''


def _plots_current(plot_dir, n_sources, refs, pattern='*.png'):
    """True when plot_dir holds at least one `pattern` file per source and
    all of them are newer than every existing file in refs (the results they
    draw)"""
    files = glob.glob(os.path.join(plot_dir, pattern))
    if n_sources == 0 or len(files) < n_sources:
        return False
    ref_time = max((os.path.getmtime(r) for r in refs if os.path.exists(r)), default=0.0)
    return min(os.path.getmtime(f) for f in files) >= ref_time


def _run_2d_step(p, bands, outd, gal, cfg, gal_logger):
    """2D model fits ([run] run_2d) on the live Photometry object, merged
    into <GAL>_photometry_v2.fits, plus the per-source 2D plots ([2d] plots).
    Reuses <GAL>_photometry_2d.fits when it already has every requested mode
    and is newer than the instrument photometry it starts from. Returns a
    short status string."""
    import run_2d_photometry as rd
    from astropy.table import Table
    d = cfg['2d']
    out_path = os.path.join(outd, f'{gal}_photometry_2d.fits')
    v2_path = os.path.join(outd, f'{gal}_photometry_v2.fits')
    inst_path = os.path.join(outd, f'{gal}_photometry_instrument.fits')
    done = []
    if os.path.exists(out_path):
        rd.load_2d_photometry(p, out_path)                 # keeps modes run earlier
        done = str(Table.read(out_path).meta.get('MODES', '')).split(',')
    up_to_date = (os.path.exists(out_path) and os.path.exists(inst_path)
                  and os.path.getmtime(out_path) >= os.path.getmtime(inst_path))
    overwrite = d['overwrite'] or _redo(cfg, '2d')
    todo = [m for m in d['modes'] if overwrite or not up_to_date or m not in done]
    if todo:
        gal_logger.info(f'Running 2D model fits, modes {todo} ({d["n_workers"]} workers)...')
        t2d = rd.run_2d_photometry(p, bands, modes=tuple(todo), all_bands=d['all_bands'],
                                   n_workers=d['n_workers'], out_path=out_path, verbose=False)
    else:
        gal_logger.info(f'2D fits {list(d["modes"])} up to date in {out_path}')
        t2d = Table.read(out_path)
    n = len(rd.merge_2d_into_v2(v2_path, t2d))
    gal_logger.info(f'  ✓ 2D fits saved → {out_path}; {n} 2D flux columns merged into {v2_path}')
    plotted = []
    if d['plots']:
        plot_dir = os.path.join(outd, 'photometry_2d_plots')
        for m in d['modes']:
            if not _redo(cfg, 'plots') and _plots_current(plot_dir, len(p.photometry_table), [out_path],
                                                          pattern=f'*_2d{m}.png'):
                gal_logger.info(f'2D plots ({m}) up to date — reused')
                continue
            gal_logger.info(f'Writing 2D plots ({m})...')
            rd.plot_2d_photometry(p, bands, mode=m, verbose=False)
            plotted.append(m)
    return (f'ok (fits: {",".join(todo) or "reused"}'
            + (f'; plots: {",".join(plotted) or "reused"}' if d['plots'] else '') + ')')


def _run_spectra_step(token, cfg, fh, gal_logger):
    """MUSE M1/M2 spectra ([run] run_spectra), settings from [muse_spectra];
    skipped when <GAL>_muse_spectra.fits is newer than the ISF products,
    unless [muse_spectra] overwrite. Returns a short status string."""
    import run_muse_spectra as rms
    settings = rms.spectra_settings(cfg['config_path'])
    if not (settings['overwrite'] or _redo(cfg, 'spectra')) and rms.spectra_up_to_date(cfg, token):
        gal_logger.info(f'MUSE spectra up to date ({rms.spectra_output_path(cfg, token)}) — reused')
        return 'ok (reused)'
    gal_logger.info(f'Running MUSE spectra (M1 annulus + M2 PSF fit, {settings["n_workers"]} workers)...')
    spec_logger = logging.getLogger('muse_spectra')
    spec_logger.setLevel(logging.INFO)
    spec_logger.addHandler(fh)
    try:
        res = rms.run_galaxy_spectra(token, cfg, settings)
    finally:
        spec_logger.removeHandler(fh)
    gal_logger.info(f'  ✓ MUSE spectra of {res["n_sources"]} sources → {res["path"]} '
                    f'(PSF FWHM {res["psf_fwhm_arcsec"]:.2f}")')
    return f'ok ({res["n_sources"]} sources)'


def _run_sed_step(token, cfg, fh, gal_logger):
    """SED plots of every cluster with photometry and MUSE spectra ([run]
    run_sed), after the galaxy's injection test ([sed] injection), which gives
    their systematic errors. Each part is reused while up to date. Returns a
    short status string."""
    import run_muse_spectra as rms
    import muse_injection as mi
    import muse_sed
    if not os.path.exists(rms.spectra_output_path(cfg, token)):
        gal_logger.info('SED plots skipped: no MUSE spectra (run_spectra)')
        return 'n/a (no spectra)'
    d = cfg['sed']
    loggers = [logging.getLogger(n) for n in ('muse_injection', 'muse_sed', 'muse_spectra')]
    for lg in loggers:
        lg.setLevel(logging.INFO)
        lg.addHandler(fh)
    parts = []
    try:
        if d['injection']:
            if _redo(cfg, 'injection') or not mi.injection_up_to_date(cfg, token):
                gal_logger.info(f'Running the MUSE injection test ({3 * d["n_per_env"]} positions, '
                                f'{d["n_workers"]} workers)...')
                settings = rms.spectra_settings(cfg['config_path'])
                res = mi.run_injection(token, cfg, settings, n_per_env=d['n_per_env'],
                                       n_workers=d['n_workers'])
                gal_logger.info(f'  ✓ injection test ({res["n_cases"]} cases) → {res["path"]}')
                parts.append('injection computed')
            else:
                gal_logger.info('Injection test up to date — reused')
                parts.append('injection reused')
        if d['overwrite'] or _redo(cfg, 'sed') or _redo(cfg, 'plots') \
                or not muse_sed.sed_up_to_date(cfg, token):
            gal_logger.info('Writing SED plots (photometry + MUSE spectra)...')
            res = muse_sed.make_galaxy_seds(token, cfg)
            gal_logger.info(f'  ✓ {res["n"]} SED plots → {res["dir"]}'
                            + ('' if res['systematics'] else ' (statistical errors only)'))
            parts.append(f'{res["n"]} plots' + ('' if res['systematics'] else ', stat. errors only'))
        else:
            gal_logger.info('SED plots up to date — reused')
            parts.append('plots reused')
    finally:
        for lg in loggers:
            lg.removeHandler(fh)
    return f'ok ({"; ".join(parts)})'


def _process_one_galaxy(token, cfg):
    """Run the full pipeline for one galaxy token. Never raises — catches
    everything and reports failure in the returned dict, so one galaxy's
    problem doesn't take down the batch."""
    gal = token.split('-')[0]
    gal_lower = gal.lower()
    outd = os.path.join(cfg['output_dir'], gal)
    os.makedirs(outd, exist_ok=True)

    gal_logger = logging.getLogger(f'run_full_pipeline.{gal}')
    gal_logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(outd, 'pipeline.log'), mode='w')
    fh.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    gal_logger.addHandler(fh)
    analysis_logger = logging.getLogger('analysis')
    analysis_logger.addHandler(fh)

    result = {'galaxy': gal, 'ok': False, 'error': None,
              'photometry_path': None, 'n_cigale': 0, 'peak_rss_gb': None,
              '2d': 'off', 'spectra': 'off', 'sed': 'off', 'memory': []}
    memory = result['memory']

    try:
        gal_logger.info(f'=== {gal} ({token}) ===')

        hst_paths = find_hst_band_paths(cfg['hst_root'], gal)
        jwst_paths = find_jwst_band_paths(cfg['jwst_root'], gal_lower)
        if not hst_paths:
            raise RuntimeError(f'no HST band images found under {cfg["hst_root"]}/{gal}')

        matched_path = os.path.join(outd, f'{gal}_matched.fits')
        isf_ok, isf_why = _isf_current(matched_path, cfg)
        isf_ran = False
        if _redo(cfg, 'isf') or not isf_ok:
            if not cfg['run_isf']:
                raise RuntimeError(
                    f'ISF products need (re)computing ({isf_why or "redo"}: {matched_path}) '
                    f'and run_isf=False in config'
                )
            gal_logger.info(f'Running IsolatedSourceFinder ({isf_why or "redo = isf"})...')
            isf = IsolatedSourceFinder(token, cfg['hst_root'], cfg['muse_dap_root'],
                                        cfg['cat_path'], cfg['sample'],
                                        psf_filepath=cfg['psf_filepath'], psf_source=cfg['psf_source'])
            isf.run_full_pipeline()
            isf.save_results(base_dir=cfg['output_dir'])
            # isf holds its own large HST/MUSE arrays (native, convolved,
            # reprojected) that Photometry below reloads independently —
            # a plain function-local variable stays alive (and its
            # memory resident) until the frame exits, not as soon as its
            # last use, so drop the reference explicitly rather than let
            # it overlap with everything Photometry loads next.
            del isf
            gc.collect()
            isf_ran = True
        else:
            gal_logger.info(f'ISF products up to date ({matched_path}) — reused')
        memory.append('ISF ' + ('computed' if isf_ran else 'reused'))

        # the photometry is redone when asked, or when the ISF products are newer than its caches
        native_cache = os.path.join(outd, f'{gal}_photometry_native.fits')
        force_phot = (_redo(cfg, 'photometry') or isf_ran
                      or (os.path.exists(native_cache)
                          and os.path.getmtime(native_cache) < os.path.getmtime(matched_path)))
        if force_phot:
            gal_logger.info('Photometry caches are recomputed ('
                            + ('redo = photometry' if _redo(cfg, 'photometry') else 'ISF products are newer')
                            + ')')

        gal_logger.info('Loading ISF results + images...')
        p = Photometry([token, gal], base_dir=cfg['output_dir'], psf_filepath=cfg['psf_filepath'],
                       psf_source=cfg['psf_source'])
        p.load_isf_results()
        p.load_images(hst_paths, jwst_paths, muse_path=cfg['muse_dap_root'])
        p.filter_by_common_fov(plot=False)
        p.load_or_measure_psf()
        p.convolve_images()

        bands = {b: v for b, v in FILTER_WAVELENGTHS.items()
                 if b.lower() in p.hst_images or b.lower() in p.jwst_images}
        if not bands:
            raise RuntimeError('no configured band matches this galaxy\'s loaded images')

        try:
            p._find_galaxy_eb_v()
        except Exception as e:
            gal_logger.warning(f'E(B-V) lookup failed ({e}) — falling back to eb_v=0.0')
            p.eb_v = 0.0

        gal_logger.info('Running v1 (convolved) photometry...')
        p.run_spectro_photometry(bands, n_workers=1, verbose=True, force_recompute=force_phot)
        # run_spectro_photometry's own cache-hit path (analysis.py's
        # run_spectro_photometry, ~line 1968) checks for <gal>_photometry
        # .fits/_photometry_spectra.npz/_annuli.reg, but only ever writes
        # _annuli.reg itself on a fresh compute — the other two only
        # exist if save_photometry(suffix='photometry') (the DEFAULT
        # suffix, distinct from the 'photometry_v2' save below) is
        # called explicitly. Without this, v1 silently recomputes from
        # scratch on every run of this driver regardless of
        # force_recompute, defeating its own resumability.
        p.save_photometry(suffix='photometry')
        # Convolved images are only used by v1/convolved photometry
        # (and by the opt-in plot_sed, never called here) — everything
        # from here on (native photometry, all diagnostic plots, save)
        # reads self.hst_images/self.jwst_images (native-resolution)
        # instead, so this drops roughly half the resident image memory
        # for the rest of this galaxy's run.
        p.hst_conv.clear()
        p.jwst_conv.clear()
        gc.collect()

        gal_logger.info('Running v2 (native, R65 aperture-corrected) photometry...')
        p.measure_native_psf_shape()
        dic_pb = {b: DIC_PASSBANDS[b] for b in bands if b in DIC_PASSBANDS}
        p.run_native_photometry(bands, n_workers=1, verbose=True,
                                 force_recompute=force_phot,
                                 plot_diagnostics=False,          # drawn below, only when out of date
                                 instrument_fit=cfg['instrument_fit'],
                                 instrument_slit_width_px=cfg['instrument_slit_width_px'],
                                 profile_by_class=cfg['profile_by_class'],
                                 psf_blur=cfg['psf_blur'],
                                 dic_passbands=dic_pb)
        memory.append('photometry ' + ('computed' if force_phot else 'cached where up to date'))

        p.save_photometry(suffix='photometry_v2')
        photometry_path = os.path.join(outd, f'{gal}_photometry_v2.fits')

        # per-source plots: drawn only when missing or older than the results they show
        n_src = len(p.photometry_table)
        refs = [native_cache, os.path.join(outd, f'{gal}_photometry_instrument.fits')]
        if cfg['plot_diagnostics']:
            if _redo(cfg, 'plots') or not all(
                    _plots_current(os.path.join(outd, d), n_src, refs)
                    for d in ('native_diagnostics', 'native_size_calibration')):
                p.plot_native_photometry_diagnostics(bands, dic_passbands=dic_pb, verbose=True)
                p.plot_native_size_calibration(bands, verbose=True)
                if cfg['instrument_fit']:
                    p.plot_instrument_flux_comparison(bands)
                memory.append('diagnostic plots drawn')
            else:
                gal_logger.info('Native diagnostic plots up to date — reused')
                memory.append('diagnostic plots reused')
        slit_dirs = ['native_slit_profiles'] + (['instrument_2d_models'] if cfg['instrument_fit'] else [])
        if _redo(cfg, 'plots') or not all(_plots_current(os.path.join(outd, d), n_src, refs)
                                          for d in slit_dirs):
            p.plot_native_slit_profiles(bands)
            memory.append('slit plots drawn')
        else:
            gal_logger.info('Slit-profile plots up to date — reused')
            memory.append('slit plots reused')

        if cfg['run_2d']:
            try:
                result['2d'] = _run_2d_step(p, bands, outd, gal, cfg, gal_logger)
            except Exception as e:                 # the photometry above stays valid
                gal_logger.exception(f'2D fits FAILED: {e}')
                result['2d'] = f'FAILED: {e}'

        n_cigale = 0
        if cfg['cigale_export']:
            gal_logger.info('Writing CIGALE input table...')
            cig = cigale_export.build_cigale_table(p.photometry_table, H0=cfg['cigale_h0'])
            cigale_path = os.path.join(outd, f"{gal}_cigale_input.{cfg['cigale_ext']}")
            cigale_export.write_cigale_ascii(cig, cigale_path)
            n_cigale = len(cig)
            gal_logger.info(f'  ✓ CIGALE table written → {cigale_path} ({n_cigale} sources)')

        result.update(ok=True, photometry_path=photometry_path, n_cigale=n_cigale)

        if cfg['run_spectra']:
            del p                                  # free the images before the cube is read
            gc.collect()
            try:
                result['spectra'] = _run_spectra_step(token, cfg, fh, gal_logger)
            except Exception as e:                 # the photometry above stays valid
                gal_logger.exception(f'MUSE spectra FAILED: {e}')
                result['spectra'] = f'FAILED: {e}'
        if cfg['run_sed']:
            if 'p' in locals():
                del p
                gc.collect()
            try:
                result['sed'] = _run_sed_step(token, cfg, fh, gal_logger)
            except Exception as e:
                gal_logger.exception(f'SED plots FAILED: {e}')
                result['sed'] = f'FAILED: {e}'
        gal_logger.info(f'=== {gal} done ===')

    except Exception as e:
        gal_logger.exception(f'FAILED: {e}')
        result['error'] = str(e)

    finally:
        # ru_maxrss is this PROCESS's peak resident memory since it
        # started — with the pool's max_tasks_per_child=1 (see main()),
        # each process handles exactly one galaxy, so this is cleanly
        # that galaxy's own peak, not a running max carried over from a
        # previous one.
        peak_rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        result['peak_rss_gb'] = peak_rss_gb
        gal_logger.info(f'Peak memory (this process): {peak_rss_gb:.2f} GB')
        gal_logger.removeHandler(fh)
        analysis_logger.removeHandler(fh)
        fh.close()

    return result


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def _process_one_galaxy_catalog(token, cfg):
    """
    Catalog mode for one galaxy: native + instrument photometry of EVERY
    cluster in the PHANGS-HST DR5 catalog (cfg catalog_selection/classes,
    default machine class 1+2 — the catalog the ISF selection starts from),
    not only the isolated ISF sources. Sources are kept if they fall on all
    HST and JWST images (MUSE is not used). No IsolatedSourceFinder, no
    convolved photometry, no MUSE spectra (those need the ISF regions), no
    per-source diagnostic PNGs (only the per-galaxy instrument comparison).
    Results go to <catalog_output_dir>/<GALAXY>/, same file names as the
    normal run. Never raises.
    """
    gal = token.split('-')[0]
    out_root = catalog_output_dir(cfg)
    outd = os.path.join(out_root, gal)
    os.makedirs(outd, exist_ok=True)

    gal_logger = logging.getLogger(f'run_full_pipeline.{gal}')
    gal_logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(outd, 'pipeline.log'), mode='w')
    fh.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    gal_logger.addHandler(fh)
    analysis_logger = logging.getLogger('analysis')
    analysis_logger.addHandler(fh)

    result = {'galaxy': gal, 'ok': False, 'error': None,
              'photometry_path': None, 'n_cigale': 0, 'peak_rss_gb': None,
              '2d': 'off', 'spectra': 'off', 'memory': []}
    try:
        gal_logger.info(f'=== {gal} ({token}) — catalog mode ===')
        if cfg['run_spectra'] or cfg['run_sed']:
            gal_logger.info('run_spectra / run_sed are ignored in catalog mode (they need the ISF regions)')
            result['spectra'] = 'n/a (catalog mode)'
        hst_paths = find_hst_band_paths(cfg['hst_root'], gal)
        jwst_paths = find_jwst_band_paths(cfg['jwst_root'], gal.lower())
        if not hst_paths:
            raise RuntimeError(f'no HST band images found under {cfg["hst_root"]}/{gal}')

        p = Photometry([token, gal], base_dir=out_root, psf_filepath=cfg['psf_filepath'],
                       psf_source=cfg['psf_source'])
        p.load_images(hst_paths, jwst_paths, muse_path=None)
        p.load_catalog_sources(cfg['cat_path'], selection=cfg['catalog_selection'],
                               classes=cfg['catalog_classes'], drop_outside=cfg['catalog_drop_outside'])
        p.filter_by_common_fov(plot=False, include_muse=False)
        p.load_or_measure_psf()

        bands = {b: v for b, v in FILTER_WAVELENGTHS.items()
                 if b.lower() in p.hst_images or b.lower() in p.jwst_images}
        if not bands:
            raise RuntimeError('no configured band matches this galaxy\'s loaded images')
        try:
            p._find_galaxy_eb_v()
        except Exception as e:
            gal_logger.warning(f'E(B-V) lookup failed ({e}) — falling back to eb_v=0.0')
            p.eb_v = 0.0

        p.init_photometry_table_from_catalog()
        p.measure_native_psf_shape()
        dic_pb = {b: DIC_PASSBANDS[b] for b in bands if b in DIC_PASSBANDS}
        gal_logger.info(f'Running native + instrument photometry on {len(p.photometry_table)} catalog sources '
                        f'({cfg["catalog_n_workers"]} workers)...')
        p.run_native_photometry(bands, n_workers=cfg['catalog_n_workers'], verbose=False,
                                force_recompute=_redo(cfg, 'photometry'),
                                plot_diagnostics=False,          # per-source PNGs: off in catalog mode
                                instrument_fit=cfg['instrument_fit'],
                                instrument_slit_width_px=cfg['instrument_slit_width_px'],
                                profile_by_class=cfg['profile_by_class'],
                                psf_blur=cfg['psf_blur'],
                                dic_passbands=dic_pb)
        if cfg['instrument_fit'] and cfg['plot_diagnostics']:
            p.plot_instrument_flux_comparison(bands)    # one per-galaxy summary PNG

        p.save_photometry(suffix='photometry_v2')
        photometry_path = os.path.join(outd, f'{gal}_photometry_v2.fits')

        if cfg['run_2d']:
            try:
                result['2d'] = _run_2d_step(p, bands, outd, gal, cfg, gal_logger)
            except Exception as e:
                gal_logger.exception(f'2D fits FAILED: {e}')
                result['2d'] = f'FAILED: {e}'

        n_cigale = 0
        if cfg['cigale_export']:
            cig = cigale_export.build_cigale_table(p.photometry_table, H0=cfg['cigale_h0'])
            cigale_path = os.path.join(outd, f"{gal}_cigale_input.{cfg['cigale_ext']}")
            cigale_export.write_cigale_ascii(cig, cigale_path)
            n_cigale = len(cig)
            gal_logger.info(f'  ✓ CIGALE table written → {cigale_path} ({n_cigale} sources)')

        gal_logger.info(f'=== {gal} done (catalog mode) ===')
        result.update(ok=True, photometry_path=photometry_path, n_cigale=n_cigale)
    except Exception as e:
        gal_logger.exception(f'FAILED: {e}')
        result['error'] = str(e)
    finally:
        peak_rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        result['peak_rss_gb'] = peak_rss_gb
        gal_logger.info(f'Peak memory (this process): {peak_rss_gb:.2f} GB')
        gal_logger.removeHandler(fh)
        analysis_logger.removeHandler(fh)
        fh.close()
    return result


def run_qa(cfg, catalog_mode=False):
    """Pooled QA plots ([qa] section) of every galaxy under the run's output
    root. Catalog mode has no ISF, v1 photometry or spectra: those plots are
    left out of plots = all."""
    from plot_qa_diagnostics import make_qa_plots, PLOT_NAMES
    root = catalog_output_dir(cfg) if catalog_mode else cfg['output_dir']
    plots = cfg['qa']['plots']
    if catalog_mode and plots == 'all':
        plots = [n for n in PLOT_NAMES if n not in ('isolation', 'phot_native_convolved',
                                                    'phot_convolved_catalog', 'spectra_photometry')]
    logger.info(f'\nQA plots of {root} ...')
    qa_dir = make_qa_plots(root, cfg['cat_path'], qa_dir=cfg['qa']['qa_dir'] or None, plots=plots)
    logger.info(f'✓ QA plots → {qa_dir}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default='config.ini', help='path to config.ini')
    ap.add_argument('--n-workers', type=int, default=None,
                     help='number of galaxies to process concurrently (skips the interactive prompt)')
    ap.add_argument('--galaxy', default=None,
                     help='run only this galaxy token synchronously (e.g. NGC1365-1.15asec), no process pool')
    ap.add_argument('--qa-only', action='store_true',
                     help='only make the pooled QA plots of the galaxies already in the output, then exit')
    ap.add_argument('--no-qa', action='store_true',
                     help='no QA plots after a batch run (overrides [qa] run_qa)')
    ap.add_argument('--dry-run', action='store_true',
                     help='check that every input, setting and resource is in place and show what the '
                          'run would compute or reuse; measures and writes nothing')
    ap.add_argument('--catalog-mode', action='store_true',
                     help='photometry of the full DR5 catalog instead of the ISF selection '
                          '(same as [run] source_catalog = catalog); output in <output_dir>_catalog/')
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.catalog_mode:
        cfg['source_catalog'] = 'catalog'
    if cfg['source_catalog'] not in ('isf', 'catalog'):
        raise ValueError(f"[run] source_catalog must be 'isf' or 'catalog', got {cfg['source_catalog']!r}")
    process = _process_one_galaxy_catalog if cfg['source_catalog'] == 'catalog' else _process_one_galaxy
    if cfg['source_catalog'] == 'catalog':
        logger.info(f'Catalog mode: {cfg["catalog_selection"]} class {"+".join(cfg["catalog_classes"])} '
                    f'DR5 sources, output in {catalog_output_dir(cfg)}')

    catalog_mode = cfg['source_catalog'] == 'catalog'
    if args.qa_only:
        run_qa(cfg, catalog_mode)
        return

    if args.dry_run:
        import preflight
        galaxies = [args.galaxy] if args.galaxy else discover_galaxies(cfg)
        n_dry = 1 if args.galaxy else (args.n_workers or (int(cfg['n_workers']) if cfg['n_workers'] else
                                                          min(4, os.cpu_count() or 1)))
        sys.exit(0 if preflight.check_pipeline(cfg, galaxies, catalog_mode, n_dry) else 1)

    if args.galaxy:
        result = process(args.galaxy, cfg)
        status = 'OK' if result['ok'] else f"FAILED: {result['error']}"
        logger.info(f"[{args.galaxy}] {status}  2D: {result.get('2d')}  spectra: {result.get('spectra')}  "
                    f"SED: {result.get('sed', 'off')}  (peak memory: {result['peak_rss_gb']:.2f} GB)")
        if result.get('memory'):
            logger.info(f"  steps: {'; '.join(result['memory'])}")
        if cfg['qa']['run_qa'] and not args.no_qa:
            logger.info('QA plots are made after batch runs only; when all galaxies are done run\n'
                        f'  python run_full_pipeline.py --config {args.config} --qa-only')
        return

    galaxies = discover_galaxies(cfg)
    if not galaxies:
        logger.info('No galaxies found/configured — check [galaxies] in the config file.')
        return
    logger.info(f'Galaxies to process ({len(galaxies)}): {", ".join(galaxies)}')

    n_workers = args.n_workers
    if n_workers is None and cfg['n_workers']:
        n_workers = int(cfg['n_workers'])
    if n_workers is None:
        n_cpu = os.cpu_count() or 1
        default_n = min(4, n_cpu, len(galaxies))
        try:
            resp = input(f'Number of CPU cores to use, one galaxy per core '
                         f'(1-{n_cpu}) [default {default_n}]: ').strip()
            n_workers = int(resp) if resp else default_n
        except EOFError:
            # No TTY (e.g. launched under nohup/a scheduler) — fall back
            # to the same default rather than crashing.
            logger.info(f'No interactive input available — defaulting to {default_n} workers.')
            n_workers = default_n
    n_workers = max(1, min(n_workers, len(galaxies)))

    logger.info(f'Processing {len(galaxies)} galaxies with {n_workers} worker process(es)...\n')

    results = []
    # 'forkserver', not 'fork': max_tasks_per_child requires it (plain
    # 'fork' raises ValueError — CPython can't guarantee a clean-slate
    # child that way). A small helper process starts once; every worker
    # is then a cheap fork() *of that helper*, not a full interpreter
    # re-spawn.
    mp_ctx = mp.get_context('forkserver')
    # max_tasks_per_child=1: a fresh worker process per galaxy rather
    # than a reused one — guarantees each galaxy's full memory footprint
    # (image arrays etc.) is returned to the OS the instant that galaxy
    # finishes, instead of lingering in a worker that's kept alive for
    # whichever galaxy gets scheduled to it next. See the module
    # docstring's "Memory" section.
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_ctx,
                              max_tasks_per_child=1) as executor:
        futures = {executor.submit(process, tok, cfg): tok for tok in galaxies}
        for fut in as_completed(futures):
            tok = futures[fut]
            try:
                res = fut.result()
            except Exception as e:
                res = {'galaxy': tok.split('-')[0], 'ok': False, 'error': str(e),
                        'photometry_path': None, 'n_cigale': 0, 'peak_rss_gb': None}
            results.append(res)
            status = 'OK' if res['ok'] else f"FAILED: {res['error']}"
            rss = res.get('peak_rss_gb')
            rss_str = f'  (peak memory: {rss:.2f} GB)' if rss is not None else ''
            steps = (f"  2D: {res.get('2d', 'off')}  spectra: {res.get('spectra', 'off')}"
                     f"  SED: {res.get('sed', 'off')}")
            logger.info(f'  [{tok}] {status}{steps}{rss_str}')
            if res.get('memory'):
                logger.info(f"      steps: {'; '.join(res['memory'])}")

    n_ok = sum(r['ok'] for r in results)
    logger.info(f'\nDone: {n_ok}/{len(results)} galaxies succeeded.')
    for r in results:
        if not r['ok']:
            logger.info(f"  ✗ {r['galaxy']}: {r['error']}")
        for step in ('2d', 'spectra', 'sed'):
            if str(r.get(step, '')).startswith('FAILED'):
                logger.info(f"  ✗ {r['galaxy']} {step}: {r[step]}")

    rss_values = sorted(r['peak_rss_gb'] for r in results if r.get('peak_rss_gb') is not None)
    if rss_values:
        median_rss = rss_values[len(rss_values) // 2]
        logger.info(
            f'\nPeak memory per galaxy: min={rss_values[0]:.2f} GB, '
            f'median={median_rss:.2f} GB, max={rss_values[-1]:.2f} GB — '
            f'worst-case total with n_workers={n_workers} ≈ {n_workers * rss_values[-1]:.1f} GB.'
        )

    if cfg['cigale_export']:
        # Each galaxy already wrote its own <GALAXY>_cigale_input.<ext>
        # table under its own output directory (inside _process_one_
        # galaxy) — no combined master table. Just report the total here.
        n_cigale_total = sum(r.get('n_cigale', 0) for r in results if r['ok'])
        logger.info(
            f'\n✓ CIGALE input tables written for {n_cigale_total} sources across '
            f'{n_ok} galaxies, as <GALAXY>_cigale_input.{cfg["cigale_ext"]} in each '
            f'galaxy\'s own output directory.'
        )

    if cfg['qa']['run_qa'] and not args.no_qa and n_ok:
        try:
            run_qa(cfg, catalog_mode)
        except Exception as e:
            logger.exception(f'QA plots FAILED: {e}')


if __name__ == '__main__':
    main()
