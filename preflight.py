"""
Dry run: check that everything a run needs is in place, without measuring
anything and without writing any file.

    python run_full_pipeline.py --config config.ini --dry-run [--catalog-mode] [--galaxy TOKEN]
    python run_catalog_photometry.py --config my_run.ini --dry-run

What is checked (FITS headers and catalogue tables only, no image data):
  - the config values, the Python packages and the PSF / filter files in data/;
  - the SFD dust map, and the name lookup used for the galaxy's E(B-V);
  - the input folders and catalogues; for every galaxy, its HST / JWST images
    (header, units, PSF model) and, in mode A, the MUSE products;
  - how many catalogue sources fall inside each image (mode B and C);
  - which steps the run would compute and which it would reuse (the same
    up-to-date tests the pipeline uses);
  - the output folder, free disk space, CPU load and memory against the
    number of processes the run will start.

Every line is ✓ (ok), ⚠ (the run works but check this) or ✗ (the run will
fail, or skip a galaxy). The exit code is 1 when there is any ✗.
"""
import glob
import importlib
import os
import shutil

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HST_BANDS = ('F275W', 'F336W', 'F438W', 'F555W', 'F814W')
JWST_BANDS = ('F200W', 'F300M', 'F335M', 'F360M', 'F770W', 'F1000W', 'F1130W', 'F2100W')
DR5_FILE = 'hlsp_phangs-cat_hst_acs-uvis_dr5-targets_v1_{selection}-class{cls}-primary.fits'
PACKAGES = ('numpy', 'scipy', 'astropy', 'photutils', 'matplotlib', 'regions', 'reproject',
            'skimage', 'dustmaps')
OK, WARN, ERR = '✓', '⚠', '✗'


class Report:
    """Check results, printed section by section"""

    def __init__(self):
        self.lines = []

    def section(self, title):
        self.lines.append(('', f'\n── {title} ' + '─' * max(0, 70 - len(title))))

    def add(self, level, msg):
        self.lines.append((level, msg))

    def ok(self, msg):
        self.add(OK, msg)

    def warn(self, msg):
        self.add(WARN, msg)

    def err(self, msg):
        self.add(ERR, msg)

    def check(self, cond, msg_ok, msg_bad, level_bad=ERR):
        self.add(OK if cond else level_bad, msg_ok if cond else msg_bad)
        return cond

    @property
    def n_err(self):
        return sum(lv == ERR for lv, _ in self.lines)

    @property
    def n_warn(self):
        return sum(lv == WARN for lv, _ in self.lines)

    def print(self):
        for lv, msg in self.lines:
            print(msg if not lv else f'  {lv} {msg}')
        print('\n' + '═' * 72)
        if self.n_err:
            print(f'  {ERR} NOT READY: {self.n_err} problem(s), {self.n_warn} warning(s). Fix the ✗ lines above.')
        elif self.n_warn:
            print(f'  {WARN} READY, with {self.n_warn} warning(s) to look at.')
        else:
            print(f'  {OK} READY: every check passed.')
        print('  (dry run: nothing was measured or written)')
        return self.n_err == 0


# ─────────────────────────────────────────────────────────────────────────
# checks shared by every mode
# ─────────────────────────────────────────────────────────────────────────

def check_settings(rep, psf_source, psf_blur, profile_by_class, modes_2d=()):
    rep.section('Settings')
    rep.check(psf_source in ('theoretical', 'table'), f'psf_source = {psf_source}',
              f"psf_source = {psf_source!r}: must be 'theoretical' or 'table'")
    blur_ok = psf_blur in ('none', 'galaxy', 'default', 'match_galaxy_psf')
    if not blur_ok:
        try:
            blur_ok = float(psf_blur) >= 0
        except ValueError:
            blur_ok = False
    rep.check(blur_ok, f'psf_extra_blur = {psf_blur}',
              f"psf_extra_blur = {psf_blur!r}: must be none, galaxy, default or a FWHM in arcsec")
    try:
        from native_phot_aux_func import profile_tag
        tags = sorted({profile_tag(s) for k, s in profile_by_class.items() if k != 'default'})
        rep.ok('cluster profiles: ' + ', '.join(tags))
    except Exception as e:                                     # noqa: BLE001
        rep.err(f'[profile] section: {e}')
    if modes_2d:
        rep.ok(f'2D fit modes: {" ".join(modes_2d)}')


def check_packages(rep, extra=()):
    rep.section('Python packages')
    missing = []
    for name in PACKAGES + tuple(extra):
        try:
            importlib.import_module(name)
        except Exception:                                      # noqa: BLE001
            missing.append(name)
    rep.check(not missing, f'all needed packages import ({len(PACKAGES) + len(extra)})',
              f'missing packages: {", ".join(missing)} (pip install -r requirements.txt)')


def check_data_files(rep, bands, need_filters=False, psf_table=None):
    """PSF models for the bands that will be measured, MUSE filter curves,
    the PSF table when psf_source = table"""
    rep.section('Repository data (data/)')
    import pickle
    from instrument_psf import HST_PSF_FILES, JWST_PSF_DIR
    hst_bands = [b for b in bands if b in HST_BANDS]
    # WFC3 must cover every HST band; ACS images use the ACS model where one
    # exists (F435W for the F438W slot) and the WFC3 model otherwise
    for inst, path in HST_PSF_FILES.items():
        if not os.path.exists(path):
            rep.err(f'HST {inst.upper()} PSF models missing: {path}')
            continue
        try:
            with open(path, 'rb') as fh:
                keys = set(pickle.load(fh))
        except Exception as e:                                 # noqa: BLE001
            rep.err(f'cannot read {path}: {e}')
            continue
        if inst == 'wfc3':
            miss = [b for b in hst_bands if b not in keys]
            rep.check(not miss, f'HST WFC3 PSF models for {len(hst_bands)} bands',
                      f'HST WFC3 PSF models missing for {", ".join(miss)} in {path}')
        else:
            rep.ok(f'HST {inst.upper()} PSF models ({", ".join(sorted(keys))}; WFC3 for the other bands)')
    jw = [b for b in bands if b in JWST_BANDS]
    miss = [b for b in jw if not glob.glob(os.path.join(JWST_PSF_DIR, f'*_PSF_filter_{b}.fits'))]
    if jw:
        rep.check(not miss, f'JWST stpsf PSFs for {len(jw)} bands',
                  f'JWST PSFs missing for {", ".join(miss)} in {JWST_PSF_DIR}')
    if need_filters:
        fdir = os.path.join(HERE, 'data', 'filters')
        miss = [f for f in ('HST_WFC3_UVIS2.F555W.dat', 'HST_WFC3_UVIS2.F814W.dat')
                if not os.path.exists(os.path.join(fdir, f))]
        rep.check(not miss, 'HST filter curves for the MUSE synthetic photometry',
                  f'filter curves missing in {fdir}: {", ".join(miss)}')
    if psf_table is not None:
        rep.check(os.path.exists(psf_table), f'PSF table {psf_table}',
                  f'psf_source = table but the PSF table is missing: {psf_table} (it is created, '
                  'and every PSF measured, on the first run)', WARN)


def check_dustmap(rep, needed=True):
    rep.section('Milky Way extinction')
    if not needed:
        rep.ok('E(B-V) given in the config: no dust map needed')
        return
    try:
        from dustmaps import std_paths
        d = os.path.join(std_paths.data_dir(), 'sfd')
        files = [os.path.join(d, f'SFD_dust_4096_{p}.fits') for p in ('ngp', 'sgp')]
        rep.check(all(os.path.exists(f) for f in files), f'SFD dust map in {d}',
                  f'SFD dust map not found in {d}: E(B-V) falls back to 0 (no MW dereddening). '
                  'Download it once (README, Installation)')
    except Exception as e:                                     # noqa: BLE001
        rep.err(f'dustmaps not usable: {e}')


def resolve_galaxy(name):
    """(SkyCoord, '') or (None, reason): the name lookup the pipeline uses for
    the E(B-V) (needs the CDS name resolver or astropy's cache)"""
    try:
        from astropy.coordinates import SkyCoord
        return SkyCoord.from_name(name), ''
    except Exception as e:                                     # noqa: BLE001
        return None, str(e).splitlines()[0][:100]


def image_info(path, band, ext=None):
    """header checks of one image: (level, message, header or None)"""
    from astropy.io import fits
    hst = band in HST_BANDS
    try:
        with fits.open(path) as h:
            if ext is None:
                ext = 0 if hst else ('SCI' if 'SCI' in h else 1)
            hdr = h[ext].header
            prim = h[0].header
        if hdr.get('NAXIS', 0) != 2:
            return ERR, f'{band}: extension {ext!r} is not a 2-D image', None
        if hst:
            inst = str(hdr.get('INSTRUME', prim.get('INSTRUME', ''))).strip().upper()
            if not inst.startswith(('WFC3', 'ACS')):
                return WARN, f'{band}: INSTRUME = {inst or "missing"} (WFC3 or ACS selects the PSF model)', hdr
        else:
            if 'PIXAR_SR' not in hdr and 'PIXAR_SR' not in prim:
                return ERR, f'{band}: no PIXAR_SR in the header (needed to convert MJy/sr to Jy)', hdr
        return OK, f'{band} {hdr["NAXIS1"]}×{hdr["NAXIS2"]}', hdr
    except Exception as e:                                     # noqa: BLE001
        return ERR, f'{band}: cannot open {path}: {str(e)[:100]}', None


def sources_on_images(ra, dec, images):
    """Sources on valid data of each image, by the rule of
    Photometry.filter_by_common_fov: the nearest pixel is inside the image,
    finite and not 0. images = {band: (path, ext, header)}; only the pixels
    at the source positions are read (memory-mapped). Returns
    ({band: n}, n on every image)."""
    import warnings
    from astropy.io import fits
    from astropy.wcs import WCS
    on_all = np.ones(len(ra), bool)
    per_band = {}
    for band, (path, ext, hdr) in images.items():
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            x, y = WCS(hdr).celestial.all_world2pix(ra, dec, 0)
        xi, yi = np.round(x).astype(int), np.round(y).astype(int)
        ok = (xi >= 0) & (yi >= 0) & (xi < hdr['NAXIS1']) & (yi < hdr['NAXIS2'])
        try:
            with fits.open(path, memmap=True) as h:
                e = ext if ext is not None else (0 if band in HST_BANDS else ('SCI' if 'SCI' in h else 1))
                v = np.asarray(h[e].data[yi[ok], xi[ok]], float)
            ok[np.flatnonzero(ok)] = np.isfinite(v) & (v != 0)
        except Exception:                                      # noqa: BLE001
            pass                                               # bounds only
        per_band[band] = int(ok.sum())
        on_all &= ok
    return per_band, int(on_all.sum())


def check_output(rep, out_root, label='output folder'):
    """writable folder (or a creatable one) and free space; nothing is created"""
    parent = out_root
    while parent and not os.path.exists(parent):
        parent = os.path.dirname(parent.rstrip('/'))
    if not parent:
        rep.err(f'{label} {out_root}: no existing parent folder')
        return
    if not os.access(parent, os.W_OK):
        rep.err(f'{label} {out_root}: {parent} is not writable')
        return
    free = shutil.disk_usage(parent).free / 1e9
    exists = 'exists' if os.path.exists(out_root) else 'will be created'
    rep.check(free > 20, f'{label} {out_root} ({exists}; {free:.0f} GB free)',
              f'{label} {out_root} ({exists}): only {free:.1f} GB free', WARN)


def check_resources(rep, n_procs, mem_note):
    rep.section('Machine')
    n_cpu = os.cpu_count() or 1
    load = os.getloadavg()[0]
    rep.ok(f'{n_cpu} CPUs, load {load:.0f} (1 min average)')
    free_cpu = n_cpu - load
    rep.check(n_procs <= max(free_cpu, 1), f'the run starts up to ~{n_procs} busy processes; ~{free_cpu:.0f} CPUs '
              'are idle now',
              f'the run starts up to ~{n_procs} busy processes but only ~{max(free_cpu, 0):.0f} CPUs are idle: '
              'lower the workers, or wait for the load to drop', WARN)
    try:
        with open('/proc/meminfo') as fh:
            mem = {ln.split(':')[0]: float(ln.split()[1]) / 1e6 for ln in fh}
        rep.ok(f'{mem["MemAvailable"]:.0f} GB of memory available ({mem_note})')
    except Exception:                                          # noqa: BLE001
        pass
    blas = {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}
    rep.check(all(v == '1' for v in blas.values()), 'BLAS threads limited to 1',
              'OMP/OPENBLAS/MKL_NUM_THREADS are not set to 1: with several workers each one may start '
              'many threads (README, Practical notes)', WARN)


MODE_2D_TEXT = {
    'fixed': 'cluster profile ⊗ PSF, centre fixed at the instP centre',
    'free': 'cluster profile ⊗ PSF, centre fitted',
    'Gfixed': 'Gaussian ⊗ PSF, centre fixed at the instP centre',
    'Gfree': 'Gaussian ⊗ PSF, centre fitted',
}


def describe_measurements(rep, instrument_fit, profile_by_class, modes_2d=(), class_note='',
                          convolved=False, extra=()):
    """'Measurements' section: which photometry the run makes, per method
    (column tag in brackets), with the cluster profiles it fits"""
    from native_phot_aux_func import profile_spec_for_class, profile_tag
    rep.section('Measurements')
    if convolved:
        rep.ok('convolved [flux_<B>]: every band convolved to the MUSE PSF, flux in the ISF '
               'segmentation polygons (matches the MUSE spectra)')
    rep.ok('native [_native_]: aperture per band from a Gaussian + constant fit to the 12-slit profile, '
           'sky annulus subtracted, aperture corrected')
    if instrument_fit:
        prof = {c: profile_tag(profile_spec_for_class(c, profile_by_class)) for c in ('YCL', 'MAP', 'OGCC')}
        prof['other'] = profile_tag(profile_spec_for_class('', profile_by_class))
        by = ', '.join(f'{c} {t}' for c, t in prof.items())
        rep.ok('instG [_instG_]: PSF ⊗ Gaussian fitted to the slit profile; flux in the model R65 aperture '
               '× model correction; R_eff')
        rep.ok(f'instP [_instP_]: PSF ⊗ cluster profile, same method (profile by class: {by})')
        if class_note:
            rep.add(WARN if class_note.startswith('!') else OK, class_note.lstrip('!'))
    else:
        rep.warn('instG / instP off (instrument_fit = False): native photometry only, no R_eff')
    for m in modes_2d:
        rep.ok(f'2D {m} [_f2d{m}_]: {MODE_2D_TEXT[m]}, fitted to every pixel within the fit radius; '
               'total model flux')
    for line in extra:
        rep.ok(line)
    rep.ok('all fluxes in mJy, Milky Way dereddened, 1σ errors; one table: <name>_photometry_v2.fits')


# ─────────────────────────────────────────────────────────────────────────
# modes A and B: run_full_pipeline.py
# ─────────────────────────────────────────────────────────────────────────

def _cache_psfsrc(path):
    from astropy.io import fits
    try:
        return fits.getheader(path, 1).get('PSFSRC')
    except Exception:                                          # noqa: BLE001
        return None


def _photometry_plan(outd, gal, cfg, ref_path=None):
    """'compute' / 'reuse' (+ why) for the native + instrument photometry"""
    if 'photometry' in cfg['redo']:
        return 'compute (redo = photometry)'
    caches = [os.path.join(outd, f'{gal}_photometry_native.fits')]
    if cfg['instrument_fit']:
        caches.append(os.path.join(outd, f'{gal}_photometry_instrument.fits'))
    if not all(os.path.exists(c) for c in caches):
        return 'compute (no cache)'
    if ref_path and os.path.exists(ref_path) and min(os.path.getmtime(c) for c in caches) < os.path.getmtime(ref_path):
        return 'compute (ISF products are newer)'
    src = {_cache_psfsrc(c) or 'table' for c in caches}
    if src != {cfg['psf_source']}:
        return f'compute (caches made with psf_source = {"/".join(sorted(src))})'
    return 'reuse (if the profile / slit settings are unchanged)'


def _2d_plan(outd, gal, cfg):
    from astropy.table import Table
    path = os.path.join(outd, f'{gal}_photometry_2d.fits')
    want = list(cfg['2d']['modes'])
    if cfg['2d']['overwrite'] or '2d' in cfg['redo'] or not os.path.exists(path):
        return f'compute {",".join(want)}'
    done = str(Table.read(path).meta.get('MODES', '')).split(',')
    inst = os.path.join(outd, f'{gal}_photometry_instrument.fits')
    if os.path.exists(inst) and os.path.getmtime(path) < os.path.getmtime(inst):
        return f'compute {",".join(want)} (older than the instrument photometry)'
    todo = [m for m in want if m not in done]
    return f'compute {",".join(todo)}, reuse {",".join(m for m in want if m in done) or "-"}' if todo else 'reuse'


def _dr5_tables(cfg, rep, files):
    """{file: Table} of the DR5 catalogues that exist (read once)"""
    from astropy.table import Table
    out = {}
    for f in files:
        path = os.path.join(cfg['cat_path'], f)
        if not os.path.exists(path):
            rep.err(f'catalogue missing: {path}')
            continue
        try:
            out[f] = Table.read(path)
            rep.ok(f'{f} ({len(out[f])} rows)')
        except Exception as e:                                 # noqa: BLE001
            rep.err(f'cannot read {path}: {e}')
    return out


def _galaxy_rows(t, gal):
    name = gal.lower()
    name = 'ngc628c' if name == 'ngc0628' else name
    names = np.char.lower(np.char.strip(np.asarray(t['PHANGS_GALAXY']).astype(str)))
    return t[names == name]


def check_pipeline(cfg, galaxies, catalog_mode, n_workers):
    """Dry run of run_full_pipeline.py (mode A, or B with catalog_mode).
    Returns True when nothing blocks the run."""
    import run_full_pipeline as rfp
    rep = Report()
    mode = 'B (every DR5 catalogue cluster)' if catalog_mode else 'A (isolated clusters, ISF)'
    out_root = rfp.catalog_output_dir(cfg) if catalog_mode else cfg['output_dir']
    rep.section(f'Run: mode {mode}')
    rep.ok(f'config {cfg["config_path"]}')
    steps = ['ISF'] if not catalog_mode else []
    steps += ['native + instP photometry' if cfg['instrument_fit'] else 'native photometry']
    if cfg['run_2d']:
        steps.append(f'2D fits ({" ".join(cfg["2d"]["modes"])})')
    if not catalog_mode:
        steps += ['MUSE spectra'] * cfg['run_spectra'] + ['injection test + SED plots'] * cfg['run_sed']
    elif cfg['run_spectra'] or cfg['run_sed']:
        rep.warn('run_spectra / run_sed are ignored in catalogue mode (they need the ISF regions)')
    if cfg['cigale_export']:
        steps.append('CIGALE table')
    rep.ok('steps: ' + ' → '.join(steps))
    if catalog_mode:
        rep.ok(f'catalogue: {cfg["catalog_selection"]} class {" + ".join(cfg["catalog_classes"])}'
               + (', dropping class "outside"' if cfg['catalog_drop_outside'] else ''))
    if cfg['redo']:
        rep.warn(f'redo = {", ".join(sorted(cfg["redo"]))}: these steps are recomputed even when up to date')
    rep.check(len(galaxies) > 0, f'{len(galaxies)} galaxies: {", ".join(galaxies)}',
              'no galaxies: check [galaxies] list (or muse_dap_root when list = auto)')

    extra = []
    if not catalog_mode and cfg['run_spectra']:
        extra.append('MUSE spectra: M1 (ISF polygon − annulus continuum) and M2 (PSF fit of cluster + '
                     'neighbours + background), with synthetic F555W / F814W fluxes')
    if not catalog_mode and cfg['run_sed']:
        extra.append('SED plots per cluster (photometry + spectra), errors ⊕ injection-test scatter')
    describe_measurements(rep, cfg['instrument_fit'], cfg['profile_by_class'],
                          cfg['2d']['modes'] if cfg['run_2d'] else (),
                          class_note=f'classes from {"PHANGS_CC_CLASS_ML" if catalog_mode and cfg["catalog_selection"] == "machine" else "the DR5 catalogue"}',
                          convolved=not catalog_mode, extra=extra)
    check_settings(rep, cfg['psf_source'], cfg['psf_blur'], cfg['profile_by_class'],
                   cfg['2d']['modes'] if cfg['run_2d'] else ())
    if catalog_mode:
        rep.check(cfg['catalog_selection'] in ('machine', 'human'),
                  f'catalog_selection = {cfg["catalog_selection"]}',
                  f'catalog_selection = {cfg["catalog_selection"]!r}: must be machine or human')
        bad = [c for c in cfg['catalog_classes'] if c not in ('1+2', '3')]
        rep.check(not bad, f'catalog_classes = {", ".join(cfg["catalog_classes"])}',
                  f'catalog_classes: unknown {bad} (use 1+2 and/or 3)')
    check_packages(rep, ('dust_extinction',) if (cfg['run_sed'] and not catalog_mode) else ())
    check_data_files(rep, HST_BANDS + JWST_BANDS,
                     need_filters=not catalog_mode and (cfg['run_spectra'] or cfg['run_sed']),
                     psf_table=cfg['psf_filepath'] if cfg['psf_source'] == 'table' else None)
    check_dustmap(rep)

    rep.section('Input folders and catalogues')
    roots = [('hst_root', cfg['hst_root']), ('jwst_root', cfg['jwst_root']), ('cat_path', cfg['cat_path'])]
    if not catalog_mode or cfg['galaxy_list'].strip().lower() == 'auto':
        roots.append(('muse_dap_root', cfg['muse_dap_root']))
    for key, path in roots:
        rep.check(bool(path) and os.path.isdir(path), f'{key} {path}', f'{key} not found: {path or "(blank)"}')
    if catalog_mode:
        files = [DR5_FILE.format(selection=cfg['catalog_selection'], cls=c) for c in cfg['catalog_classes']]
    elif 'IR5' in cfg['cat_path']:
        files = []
        rep.ok('IR5 catalogue layout: per-galaxy files are found by the ISF')
    else:
        files = [DR5_FILE.format(selection=s, cls='1+2') for s in ('human', 'machine')]
    tables = _dr5_tables(cfg, rep, files) if os.path.isdir(cfg['cat_path']) else {}
    check_output(rep, out_root)

    rep.section(f'Galaxies ({len(galaxies)})')
    for token in galaxies:
        _check_galaxy(rep, token, cfg, catalog_mode, tables, out_root)

    n_inner = cfg['catalog_n_workers'] if catalog_mode else 1
    if cfg['run_2d']:
        n_inner = max(n_inner, cfg['2d']['n_workers'])
    if not catalog_mode and cfg['run_spectra']:
        from run_muse_spectra import spectra_settings
        n_inner = max(n_inner, spectra_settings(cfg['config_path'])['n_workers'])
    if not catalog_mode and cfg['run_sed'] and cfg['sed']['injection']:
        n_inner = max(n_inner, cfg['sed']['n_workers'])
    n_gal = max(1, min(n_workers or 1, len(galaxies) or 1))
    check_resources(rep, n_gal * n_inner,
                    f'{n_gal} galaxies at once × several GB each; the largest PHANGS galaxies peak near 10 GB')
    if cfg['qa']['run_qa']:
        rep.section('After the run')
        rep.ok(f'QA plots → {cfg["qa"]["qa_dir"] or os.path.join(out_root, "qa_plots")} '
               '(batch runs only; single-galaxy runs: --qa-only afterwards)')
    return rep.print()


def _check_galaxy(rep, token, cfg, catalog_mode, tables, out_root):
    import run_full_pipeline as rfp
    from instrument_psf import muse_fwhm_from_token
    gal = token.split('-')[0]
    outd = os.path.join(out_root, gal)
    rep.add('', f'\n  {token}')
    n_err0 = rep.n_err

    hst = rfp.find_hst_band_paths(cfg['hst_root'], gal) if os.path.isdir(cfg['hst_root']) else {}
    jwst = rfp.find_jwst_band_paths(cfg['jwst_root'], gal.lower()) if os.path.isdir(cfg['jwst_root']) else {}
    hst = {b: p for b, p in hst.items() if b in rfp.FILTER_WAVELENGTHS}
    jwst = {b: p for b, p in jwst.items() if b in rfp.FILTER_WAVELENGTHS}
    headers, bad = {}, []
    for band, path in list(hst.items()) + list(jwst.items()):
        lv, msg, hdr = image_info(path, band)
        if hdr is not None:
            headers[band] = (path, None, hdr)
        if lv != OK:
            bad.append((lv, msg))
    miss_h = [b for b in HST_BANDS if b not in hst]
    miss_j = [b for b in JWST_BANDS if b not in jwst]
    if not hst:
        rep.err(f'no HST images under {cfg["hst_root"]}/{gal}')
    else:
        rep.check(not miss_h, f'HST {len(hst)}/5 bands',
                  f'HST {len(hst)}/5 bands, missing {", ".join(miss_h)}', WARN)
    rep.check(not miss_j, f'JWST {len(jwst)}/8 bands',
              f'JWST {len(jwst)}/8 bands' + (f', missing {", ".join(miss_j)}' if jwst else
                                            f' (none under {cfg["jwst_root"]}/{gal.lower()}/images)'), WARN)
    for lv, msg in bad:
        rep.add(lv, msg)

    coord, why = resolve_galaxy(gal)
    rep.check(coord is not None, 'name resolves (E(B-V) lookup)',
              f'name lookup failed ({why}): E(B-V) falls back to 0 (no MW dereddening)', WARN)

    if catalog_mode:
        rows = [_galaxy_rows(t, gal) for t in tables.values()]
        n_cat = sum(len(r) for r in rows)
        if tables and not rep.check(n_cat > 0, f'{n_cat} catalogue clusters',
                                    f'no {cfg["catalog_selection"]} class {"+".join(cfg["catalog_classes"])} '
                                    'clusters of this galaxy in the catalogue'):
            pass
        elif n_cat and headers:
            ra = np.concatenate([np.asarray(r['PHANGS_RA'], float) for r in rows])
            dec = np.concatenate([np.asarray(r['PHANGS_DEC'], float) for r in rows])
            if cfg['catalog_drop_outside']:
                key = 'PHANGS_CC_CLASS_ML' if cfg['catalog_selection'] == 'machine' else 'PHANGS_CC_CLASS_HUMAN'
                keep = np.concatenate([np.char.strip(np.asarray(r[key]).astype(str)) != 'outside' for r in rows])
                ra, dec = ra[keep], dec[keep]
            per_band, n_all = sources_on_images(ra, dec, headers)
            rep.check(n_all > 0, f'{n_all} of {len(ra)} clusters on valid data of every image '
                      '(the sources that will be measured)',
                      f'none of the {len(ra)} clusters is inside every image '
                      f'(per band: {", ".join(f"{b} {n}" for b, n in per_band.items())})')
        plan = [f'photometry {_photometry_plan(outd, gal, cfg)}']
    else:
        fwhm = muse_fwhm_from_token(token)
        if cfg['psf_source'] == 'theoretical':
            rep.check(fwhm is not None, f'MUSE PSF FWHM {fwhm}" (from the token name)',
                      f'token {token!r} has no "-<x>asec" part: the MUSE PSF FWHM cannot be set')
        dap = os.path.join(cfg['muse_dap_root'], token)
        maps = os.path.join(dap, f'{token}_DR2.2v2', f'{token}_MAPS.fits')
        allspec = os.path.join(dap, f'{token}_DR2.2v2', f'{token}_AllSpectra.fits')
        rep.check(os.path.exists(maps) and os.path.exists(allspec), 'MUSE DAP maps and spectra',
                  f'MUSE DAP products missing: {maps if not os.path.exists(maps) else allspec}')
        if cfg['run_spectra']:
            cube = os.path.join(dap, f'{token}.fits')
            rep.check(os.path.exists(cube), f'MUSE cube ({os.path.getsize(cube) / 1e9:.1f} GB)'
                      if os.path.exists(cube) else '', f'MUSE cube missing: {cube} (needed by run_spectra)')
        matched = os.path.join(outd, f'{gal}_matched.fits')
        isf_ok, isf_why = rfp._isf_current(matched, cfg)
        if 'isf' in cfg['redo'] or not isf_ok:
            why = isf_why or 'redo = isf'
            rep.check(cfg['run_isf'], f'ISF will run ({why})',
                      f'ISF products need computing ({why}) but run_isf = False')
            plan = ['ISF compute', 'photometry compute (after the ISF)']
        else:
            plan = ['ISF reuse', f'photometry {_photometry_plan(outd, gal, cfg, matched)}']
    if cfg['run_2d']:
        plan.append(f'2D {_2d_plan(outd, gal, cfg)}')
    if not catalog_mode and cfg['run_spectra']:
        from run_muse_spectra import spectra_up_to_date
        fresh = spectra_up_to_date(cfg, token) and 'spectra' not in cfg['redo']
        plan.append('spectra ' + ('reuse' if fresh else 'compute'))
    if not catalog_mode and cfg['run_sed']:
        import muse_injection as mi
        import muse_sed
        if cfg['sed']['injection']:
            plan.append('injection ' + ('reuse' if mi.injection_up_to_date(cfg, token)
                                        and 'injection' not in cfg['redo'] else 'compute'))
        plan.append('SED ' + ('reuse' if muse_sed.sed_up_to_date(cfg, token) and not cfg['sed']['overwrite']
                              and 'sed' not in cfg['redo'] else 'compute'))
    rep.ok('plan: ' + '; '.join(plan)) if rep.n_err == n_err0 else rep.add('', '    (plan skipped: fix the ✗ above)')


# ─────────────────────────────────────────────────────────────────────────
# mode C: run_catalog_photometry.py
# ─────────────────────────────────────────────────────────────────────────

def check_catalog_run(cfg, paths, max_sources=None):
    """Dry run of run_catalog_photometry.py. `paths` = image_paths(cfg)."""
    from astropy.table import Table
    rep = Report()
    rep.section(f'Run: mode C, field {cfg["name"]}')
    outd = os.path.join(cfg['output_dir'], cfg['name'])
    caches = [os.path.join(outd, f'{cfg["name"]}_photometry_native.fits')]
    if cfg['instrument_fit']:
        caches.append(os.path.join(outd, f'{cfg["name"]}_photometry_instrument.fits'))
    if cfg['force_recompute']:
        rep.ok('photometry: compute (force_recompute = True)')
    elif all(os.path.exists(c) for c in caches):
        rep.ok('photometry: reuse the caches in the output folder (if the settings are unchanged)')
    else:
        rep.ok('photometry: compute (no cache)')
    if cfg['class_col']:
        note = f'classes from column {cfg["class_col"]} (YCL / MAP / OGCC; anything else is "other")'
    else:
        note = '!no class_col: every source gets the "other" profile (OGCC are not fitted with King)'
    extra = []
    if cfg['plots'] != 'none':
        extra.append(f'plots = {cfg["plots"]}: '
                     + {'slits': 'slit-profile PNG per source', '2d': '2D data/model/residual PNG per source',
                        'all': 'slit-profile and 2D PNGs per source'}.get(cfg['plots'], cfg['plots']))
    if cfg['cigale_export']:
        extra.append(f'CIGALE input table ({cfg["name"]}_cigale_input.txt)')
    describe_measurements(rep, cfg['instrument_fit'], cfg['profile_by_class'], cfg['modes_2d'],
                          class_note=note, extra=extra)
    check_settings(rep, cfg['psf_source'], cfg['psf_blur'], cfg['profile_by_class'], cfg['modes_2d'])
    if cfg['modes_2d'] and not cfg['instrument_fit']:
        rep.err('modes_2d needs instrument_fit = True (the 2D fits start from the instP results)')
    rep.check(cfg['plots'] in ('none', 'slits', '2d', 'all'), f'plots = {cfg["plots"]}',
              f'plots = {cfg["plots"]!r}: use none, slits, 2d or all')
    check_packages(rep)
    check_data_files(rep, list(paths), psf_table=cfg['psf_filepath'] if cfg['psf_source'] == 'table' else None)
    check_dustmap(rep, needed=cfg['ebv'] is None)

    rep.section('Catalogue')
    t = None
    if not cfg['catalog'] or not os.path.exists(cfg['catalog']):
        rep.err(f'[catalog] file not found: {cfg["catalog"] or "(blank)"}')
    else:
        try:
            t = Table.read(cfg['catalog'], format=cfg['fmt']) if cfg['fmt'] else Table.read(cfg['catalog'])
            rep.ok(f'{cfg["catalog"]}: {len(t)} rows')
        except Exception as e:                                 # noqa: BLE001
            rep.err(f'cannot read {cfg["catalog"]}: {e}')
    ra = dec = None
    if t is not None:
        cols = {'ra_col': cfg['ra_col'], 'dec_col': cfg['dec_col'], 'id_col': cfg['id_col'],
                'class_col': cfg['class_col'], 'dist_col': cfg['dist_col']}
        miss = [f'{k} = {v}' for k, v in cols.items() if v and v not in t.colnames]
        rep.check(not miss, 'columns found: ' + ', '.join(f'{v}' for v in cols.values() if v),
                  f'columns not in the table: {", ".join(miss)} (it has {", ".join(t.colnames[:15])}'
                  + (' ...' if len(t.colnames) > 15 else '') + ')')
        if not miss:
            try:
                # same rule as Photometry.load_custom_catalog: strings are sexagesimal, else degrees
                from astropy.coordinates import SkyCoord
                import astropy.units as u
                r, d = np.asarray(t[cfg['ra_col']]), np.asarray(t[cfg['dec_col']])
                c = (SkyCoord(r.astype(str), d.astype(str), unit=(u.hourangle, u.deg))
                     if r.dtype.kind in 'USO' or d.dtype.kind in 'USO'
                     else SkyCoord(r.astype(float) * u.deg, d.astype(float) * u.deg))
                ra, dec = c.ra.deg, c.dec.deg
                rep.ok(f'coordinates read (centre {np.median(ra):.4f}, {np.median(dec):+.4f} deg)')
            except Exception as e:                             # noqa: BLE001
                rep.err(f'cannot read the coordinates: {e}')
            if ra is not None and max_sources:
                ra, dec = ra[:max_sources], dec[:max_sources]
            if cfg['class_col'] and cfg['class_col'] in t.colnames:
                vals = sorted({str(v).strip() for v in t[cfg['class_col']]})
                known = [v for v in vals if v in ('YCL', 'MAP', 'OGCC')]
                rep.check(bool(known), f'classes: {", ".join(vals[:8])}',
                          f'class column has no YCL / MAP / OGCC values ({", ".join(vals[:8])}): '
                          'every source gets the EFF profile', WARN)
        has_dist = cfg['distance_mpc'] is not None or (cfg['dist_col'] and cfg['dist_col'] in t.colnames)
        rep.check(has_dist, 'distance given', 'no distance: R_eff in arcsec only, cap 0.5"', WARN)

    rep.section(f'Images ({len(paths)} bands)')
    if not paths:
        rep.err('no images: give [images] BAND = path lines or hst_root / jwst_root')
    headers = {}
    for band, (path, ext) in paths.items():
        if not os.path.exists(path):
            rep.err(f'{band}: file not found: {path}')
            continue
        lv, msg, hdr = image_info(path, band, ext)
        rep.add(lv, msg + f'  {os.path.basename(path)}')
        if hdr is not None:
            headers[band] = (path, ext, hdr)
    if ra is not None and headers:
        per_band, n_all = sources_on_images(ra, dec, headers)
        for b, n in per_band.items():
            if n == 0:
                rep.err(f'{b}: no source on valid data of this image (wrong image, or coordinates?)')
        if cfg['require_all_bands']:
            rep.check(n_all > 0, f'{n_all} of {len(ra)} sources on valid data of every image (will be measured)',
                      f'no source is inside every image: set require_all_bands = False, or check the images')
        else:
            rep.ok('sources per band: ' + ', '.join(f'{b} {n}' for b, n in per_band.items()))

    rep.section('Output')
    check_output(rep, os.path.join(cfg['output_dir'], cfg['name']))
    check_resources(rep, cfg['n_workers'], 'one field, several GB')
    return rep.print()
