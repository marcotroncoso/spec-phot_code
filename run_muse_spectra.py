#!/usr/bin/env python
"""
Cluster-only MUSE spectra of the ISF isolated sources (see muse_spec_aux_func).

Inputs (nothing from the photometry is used to build the spectra):
  output/<GAL>/<GAL>_matched.fits, <GAL>_MUSE.reg, <GAL>_HST.reg, <GAL>_MUSE.fits
  the copt MUSE cube  <muse_dap_root>/<TOKEN>/<TOKEN>.fits  (DATA, STAT, DQ)
  the DAP MAPS (V_STARS, systemic velocity) and the DR5 catalogue (neighbours)

Outputs:
  output/<GAL>/<GAL>_muse_spectra.fits
  output/<GAL>/muse_spectra/   per-source figures, PSF + summary figures,
                               cutouts/ (cached cube stamps)
Per-source figures are named like the native slit profiles,
<row>_<class>_spectra.png with <row> the source's row in
<GAL>_photometry_v2.fits (the photometry is read only for this name);
sources not in the photometry table get ID<ID>_<class>_spectra.png.

Settings (aperture, wavelength range, M1/M2 parameters, workers, plots) come
from the [muse_spectra] section of the config file; command-line flags
override them. run_full_pipeline.py runs the same thing per galaxy
(run_galaxy_spectra) when [run] run_spectra = True.

Run e.g.
  nice -n 10 python run_muse_spectra.py --galaxy NGC1365-1.15asec
  nice -n 10 python run_muse_spectra.py --galaxy NGC1365-1.15asec --no-plots
"""

import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse
import configparser
import json
import logging
import multiprocessing as mp
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from regions import Regions

import muse_spec_aux_func as msa

logger = logging.getLogger('muse_spectra')

HERE = os.path.dirname(os.path.abspath(__file__))
FILTER_DIR = os.path.join(HERE, 'data', 'filters')      # HST transmission curves (from cigale-PHANGS)
FILTERS = {'F555W': 'HST_WFC3_UVIS2.F555W.dat',
           'F814W': 'HST_WFC3_UVIS2.F814W.dat'}
DR5_MACHINE = 'hlsp_phangs-cat_hst_acs-uvis_dr5-targets_v1_machine-class1+2-primary.fits'
PSF_FALLBACK_BETA = 2.5      # Moffat beta when no cluster gives a usable PSF fit (typical MUSE WFM)

# colours: categorical slots 1-4 of the dataviz reference palette + neutral ink
COL = {'A': '#6b6a64', 'C1': '#2a78d6', 'BKG': '#eb6834', 'C2': '#1baf7a',
       'EXTRA': '#eda100'}


# ─────────────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────────────

def load_inputs(token, cfg, aperture, wave_range=msa.WAVE_RANGE):
    gal = token.split('-')[0]
    outd = os.path.join(cfg['output_dir'], gal)
    tab = Table.read(os.path.join(outd, f'{gal}_matched.fits'), 'MATCHED')
    regs = list(Regions.read(os.path.join(outd, f'{gal}_{aperture}.reg'), format='ds9'))
    if len(regs) != len(tab):
        raise RuntimeError(f'{len(regs)} regions vs {len(tab)} matched rows')
    with fits.open(os.path.join(outd, f'{gal}_MUSE.fits')) as h:
        labels = h['LABELS'].data.astype(np.int32)
        lab_wcs = WCS(h['LABELS'].header)

    dap = os.path.join(cfg['muse_dap_root'], token)
    cube_path = os.path.join(dap, f'{token}.fits')
    maps_path = os.path.join(dap, f'{token}_DR2.2v2', f'{token}_MAPS.fits')
    with fits.open(cube_path, memmap=True) as h:
        cube_hdr = h['DATA'].header.copy()
    cel = WCS(cube_hdr).celestial
    if labels.shape != (cube_hdr['NAXIS2'], cube_hdr['NAXIS1']) or \
            not np.allclose(lab_wcs.wcs.crpix, cel.wcs.crpix):
        raise RuntimeError('ISF LABELS map is not on the cube grid')
    with fits.open(maps_path) as h:
        vsys = float(h[0].header['REDSHIFT'])
        vstars = h['V_STARS'].data.astype(float)

    cat = Table.read(os.path.join(cfg['cat_path'], DR5_MACHINE))
    cat = cat[cat['PHANGS_GALAXY'] == gal.lower()]

    # photometry-table row of each ID, only to name figures like native_slit_profiles
    phot_path = os.path.join(outd, f'{gal}_photometry_v2.fits')
    phot_row = {}
    if os.path.exists(phot_path):
        ids = Table.read(phot_path)['ID_PHANGS_CLUSTER']
        phot_row = {int(v): r for r, v in enumerate(ids)}

    # cutouts are cached over the full cube; wsel trims them to wave_range
    wave_full = msa.cube_wave(cube_hdr)
    wsel = (wave_full >= wave_range[0]) & (wave_full <= wave_range[1])
    return {'gal': gal, 'token': token, 'outd': outd, 'tab': tab, 'regs': regs,
            'labels': labels, 'cube_path': cube_path, 'cube_hdr': cube_hdr,
            'cel': cel, 'wave': wave_full[wsel], 'wsel': wsel, 'wave_range': wave_range,
            'vsys': vsys,
            'vstars': vstars, 'cat': cat, 'phot_row': phot_row}


def source_geometry(inp, fwhm_px, gap_fwhm, width_fwhm, fit_half):
    """Pixel centres, cutout half-size, neighbour lists."""
    cel = inp['cel']
    tab, cat = inp['tab'], inp['cat']
    sc = SkyCoord(tab['PHANGS_RA'], tab['PHANGS_DEC'], unit='deg')
    xs, ys = cel.world_to_pixel(sc)
    cx, cy = cel.world_to_pixel(SkyCoord(cat['PHANGS_RA'], cat['PHANGS_DEC'], unit='deg'))

    r_need = []
    for reg, x, y in zip(inp['regs'], xs, ys):
        v = reg.to_pixel(cel).vertices
        r_need.append(np.max(np.hypot(v.x - x, v.y - y)))
    half = int(np.ceil(max(np.max(r_need) + (gap_fwhm + width_fwhm) * fwhm_px + 3,
                           fit_half + 3)))

    geo = []
    for i, (x, y) in enumerate(zip(xs, ys)):
        ix, iy = int(round(float(x))), int(round(float(y)))
        x0, y0 = ix - half, iy - half
        d = np.hypot(cx - x, cy - y)
        same = d < 0.05 / msa.MUSE_PIX_ARCSEC        # the target itself
        near = (d < half * 1.5) & ~same
        geo.append({'xy': (float(x) - x0, float(y) - y0), 'center': (ix, iy),
                    'x0': x0, 'y0': y0,
                    'neigh': [(float(a) - x0, float(b) - y0) for a, b in zip(cx[near], cy[near])]})
    return geo, half


def get_cutouts(inp, geo, half, cut_dir):
    os.makedirs(cut_dir, exist_ok=True)
    ids = [int(i) for i in inp['tab']['ID_PHANGS_CLUSTER']]
    path = {i: os.path.join(cut_dir, f'{inp["gal"]}_{i}_h{half}.npz') for i in ids}
    missing = {i: g['center'] for i, g in zip(ids, geo) if not os.path.exists(path[i])}
    if missing:
        logger.info(f'  extracting {len(missing)} cutouts (one pass over the cube)')
        t = time.time()
        cuts, _ = msa.extract_cutouts(inp['cube_path'], missing, half)
        for i, c in cuts.items():
            msa.save_cutout(path[i], c)
        logger.info(f'  cutouts done in {time.time() - t:.0f} s')
    return path


def local_z(inp, g, ap_frac):
    x0, y0 = g['x0'], g['y0']
    ny, nx = ap_frac.shape
    sub = np.full((ny, nx), np.nan)
    Y0, X0 = max(y0, 0), max(x0, 0)
    Y1, X1 = min(y0 + ny, inp['vstars'].shape[0]), min(x0 + nx, inp['vstars'].shape[1])
    sub[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0] = inp['vstars'][Y0:Y1, X0:X1]
    v = np.nanmedian(sub[ap_frac > 0.5]) if np.isfinite(sub[ap_frac > 0.5]).any() \
        else np.nanmedian(sub)
    v = 0.0 if not np.isfinite(v) else v
    return (inp['vsys'] + v) / msa.C_KMS


# ─────────────────────────────────────────────────────────────────────────────
# PSF
# ─────────────────────────────────────────────────────────────────────────────

def measure_psf(inp, geo, paths, fwhm0_px, plot_dir, plot=True):
    rows = []
    for i, (sid, g) in enumerate(zip(inp['tab']['ID_PHANGS_CLUSTER'], geo)):
        c = msa.load_cutout(paths[int(sid)], inp['wsel'])
        z = inp['vsys'] / msa.C_KMS
        good = msa.continuum_mask(inp['wave'], z)
        shape = c['data'].shape[1:]
        excl = msa.neighbour_exclusion(shape, g['neigh'], fwhm0_px)
        for lo, hi in msa.CONT_BINS:
            img, err = msa.continuum_image(c['data'], c['var'], inp['wave'], good, lo, hi)
            r = msa.fit_psf_stamp(img, err, g['xy'][0], g['xy'][1], fwhm0_px, mask=excl)
            r.update({'id': int(sid), 'wave': 0.5 * (lo + hi)})
            rows.append(r)
    wave_ref = 0.5 * sum(inp['wave_range'])
    try:
        psf, summary = msa.build_psf_model(rows, wave_ref=wave_ref)
        summary['fallback'] = ''
    except RuntimeError as e:
        # too few clusters to measure the PSF: Moffat with the FWHM of the token
        # name, flat in wavelength, and the median beta of whatever fits exist
        ok = msa.psf_fit_usable(rows)
        beta = float(np.median([r['beta'] for r in ok])) if ok else PSF_FALLBACK_BETA
        psf = msa.PSFModel(float(fwhm0_px), beta, 0.0, wave_ref)
        summary = {'wave': np.array([]), 'fwhm_med': np.array([]), 'fwhm_err': np.array([]),
                   'n_used': len(ok), 'fallback': str(e)}
        logger.warning(f'  PSF could not be measured ({e}): using the token FWHM '
                       f'{fwhm0_px * msa.MUSE_PIX_ARCSEC:.2f}" and beta {beta:.2f}')
    if not plot:
        return psf, summary, rows

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    ok = msa.psf_fit_usable(rows)
    w = np.array([r['wave'] for r in ok])
    f = np.array([r['fwhm'] for r in ok]) * msa.MUSE_PIX_ARCSEC
    fe = np.array([r['fwhm_err'] for r in ok]) * msa.MUSE_PIX_ARCSEC
    jitter = (np.random.default_rng(0).random(w.size) - 0.5) * 120
    ax[0].errorbar(w + jitter, f, fe, fmt='o', ms=3, color='#b7b5ab', ecolor='#d6d4cb',
                   lw=1, label='single cluster')
    if len(summary['wave']):
        ax[0].errorbar(summary['wave'], summary['fwhm_med'] * msa.MUSE_PIX_ARCSEC,
                       summary['fwhm_err'] * msa.MUSE_PIX_ARCSEC, fmt='s', ms=7,
                       color=COL['C1'], lw=2, label='median ± error')
    if summary.get('fallback'):
        ax[0].text(0.02, 0.98, 'PSF not measurable: token FWHM used', transform=ax[0].transAxes,
                   va='top', fontsize=8, color='#1a1a19')
    ww = np.linspace(*inp['wave_range'], 10)
    ax[0].plot(ww, psf.fwhm(ww) * msa.MUSE_PIX_ARCSEC, color=COL['C1'], lw=2)
    nominal = float(inp['token'].split('-')[1].replace('asec', ''))
    ax[0].axhline(nominal, color=COL['BKG'], lw=1.5, ls='--', label=f'nominal {nominal}"')
    ax[0].set(xlabel='wavelength [Å]', ylabel='Moffat FWHM ["]',
              title=f'{inp["gal"]}: PSF from isolated clusters')
    ax[0].legend(frameon=False, fontsize=8)
    if ok:
        ax[1].hist([r['beta'] for r in ok], bins=20, color='#b7b5ab')
    ax[1].axvline(psf.beta, color=COL['C1'], lw=2, label=f'median β = {psf.beta:.2f}')
    ax[1].set(xlabel='Moffat β', ylabel='N fits')
    ax[1].legend(frameon=False, fontsize=8)
    fig.savefig(os.path.join(plot_dir, f'{inp["gal"]}_psf.png'), dpi=120)
    plt.close(fig)
    return psf, summary, rows


# ─────────────────────────────────────────────────────────────────────────────
# Per source
# ─────────────────────────────────────────────────────────────────────────────

_G = {}


def process_source(i):
    inp, geo, paths, psf, opt = _G['inp'], _G['geo'], _G['paths'], _G['psf'], _G['opt']
    sid = int(inp['tab']['ID_PHANGS_CLUSTER'][i])
    g = geo[i]
    c = msa.load_cutout(paths[sid], inp['wsel'])
    data, var, wave = c['data'], c['var'], inp['wave']
    shape = data.shape[1:]
    ap = msa.polygon_cutout_mask(inp['regs'][i], inp['cel'], g['x0'], g['y0'], shape)
    z = local_z(inp, g, ap)
    fw = float(psf.fwhm(np.median(wave)))

    # labels cut-out (target segment = label under the aperture)
    lab = np.zeros(shape, np.int32)
    L = inp['labels']
    Y0, X0 = max(g['y0'], 0), max(g['x0'], 0)
    Y1, X1 = min(g['y0'] + shape[0], L.shape[0]), min(g['x0'] + shape[1], L.shape[1])
    lab[Y0 - g['y0']:Y1 - g['y0'], X0 - g['x0']:X1 - g['x0']] = L[Y0:Y1, X0:X1]
    tl = np.bincount(lab[ap > 0.5].ravel()).argmax() if (ap > 0.5).any() else 0
    excl = msa.neighbour_exclusion(shape, g['neigh'], fw, labels=lab, target_label=tl)
    ring = msa.annulus_mask(ap, opt.gap_fwhm * fw, opt.width_fwhm * fw, exclude=excl)

    res = {'id': sid, 'z': z, 'ap': ap, 'ring_all': ring}
    m1 = msa.method_annulus(data, var, ap, ring, wave, z, smooth=opt.smooth)
    res['m1'] = m1
    m2 = msa.method_psf(data, var, wave, ap, g['xy'], g['neigh'], psf, z,
                        fit_half=opt.fit_half, ridge_frac=opt.ridge)
    # empirical rescaling of the propagated (correlated-noise) errors
    good = msa.continuum_mask(wave, z)
    res['err_scale'] = {}
    for d_, f_, e_ in ((m1, 'A', 'A_err'), (m1, 'C1', 'C1_err'), (m1, 'B', 'B_err'),
                       (m2, 'C2', 'C2_err'), (m2, 'E_AP', 'E_AP_err')):
        k = msa.empirical_error_scale(wave, d_[f_], d_[e_], good)
        d_[e_] = d_[e_] * k
        res['err_scale'][f_] = float(np.nanmedian(k))
    m1['BKG_LINES_AP_err'] = m1['N_ap'] * m1['B_err']
    m2['C2_AP_err'] = m2['C2_err'] * m2['EE_AP']
    m2['EXTRA2'] = m1['A'] - m2['C2_AP']
    m2['EXTRA2_err'] = np.sqrt(m1['A_err']**2 + m2['C2_AP_err']**2)
    res['m2'] = m2

    # emission lines on every component
    comps = {'A': (m1['A'], m1['A_err']), 'C1': (m1['C1'], m1['C1_err']),
             'BKGL': (m1['BKG_LINES_AP'], m1['BKG_LINES_AP_err']),
             'C2': (m2['C2'], m2['C2_err']), 'C2AP': (m2['C2_AP'], m2['C2_AP_err']),
             'EXTRA2': (m2['EXTRA2'], m2['EXTRA2_err']),
             'EAP': (m2['E_AP'], m2['E_AP_err'])}
    res['lines'] = {k: msa.fit_lines(wave, f, e, z) for k, (f, e) in comps.items()}

    # synthetic photometry (for the comparison with the imaging only)
    res['synth'] = {}
    for b, fpath in FILTERS.items():
        fwv, ft = msa.read_filter(os.path.join(FILTER_DIR, fpath))
        for k in ('C1', 'C2', 'A'):
            res['synth'][(k, b)] = msa.synthetic_fnu_mJy(wave, *comps[k], fwv, ft)

    res['f_psf'] = f_psf(res)
    if opt.plots:
        plot_source(inp, i, res, data, var, wave, psf)
    # drop the stamp cubes before returning to the parent
    for k in ('model_stamp', 'psf_stamp'):
        m2.pop(k, None)
    return res


def _narrowband(data, wave, z, lam, half=5.0, side=(15.0, 30.0)):
    lo = lam * (1 + z)
    on = np.abs(wave - lo) <= half
    off = (np.abs(wave - lo) >= side[0]) & (np.abs(wave - lo) <= side[1])
    return np.nansum(data[on], axis=0) - on.sum() * np.nanmedian(data[off], axis=0)


def plot_tag(inp, sid, cls):
    """'<row>_<class>' as in native_slit_profiles, or 'ID<ID>_<class>' without photometry."""
    r = inp['phot_row'].get(int(sid))
    return f'{r}_{cls}' if r is not None else f'ID{int(sid)}_{cls}'


def plot_source(inp, i, res, data, var, wave, psf):
    m1, m2, z, ap = res['m1'], res['m2'], res['z'], res['ap']
    sid, row = res['id'], inp['tab'][i]
    good = msa.continuum_mask(wave, z)
    fig = plt.figure(figsize=(15, 11), constrained_layout=True)
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1.1, 1.1])

    sl = m2['stamp_slice']
    img, _ = msa.continuum_image(data, var, wave, good, wave[0], wave[-1])
    ha = _narrowband(data, wave, z, msa.EMISSION_LINES['Ha'])
    on = np.abs(wave - msa.EMISSION_LINES['Ha'] * (1 + z)) <= 5.0
    ha_psf = np.nansum(m2['psf_stamp'][on], axis=0)
    ha_mod = np.nansum(m2['model_stamp'][on], axis=0)
    ha_stamp = np.nansum(data[on][:, sl[0], sl[1]], axis=0)
    panels = [(img, 'continuum (mean, per Å)', True),
              (ha, 'Hα ± 5 Å, continuum-subtracted', True),
              (ha_stamp - ha_psf, 'Hα data − cluster PSF (M2)', False),
              (ha_stamp - ha_mod, 'Hα data − full model (M2)', False)]
    for k, (im, title, full) in enumerate(panels):
        ax = fig.add_subplot(gs[0, k])
        v = np.nanpercentile(im, [2, 99.5])
        ax.imshow(im, origin='lower', cmap='Greys', vmin=v[0], vmax=v[1])
        # stamp panels (M2) are offset from the full cutout
        ox, oy = (0, 0) if full else (sl[1].start, sl[0].start)
        xg, yg = np.arange(ap.shape[1]) - ox, np.arange(ap.shape[0]) - oy
        ax.contour(xg, yg, ap, levels=[0.5], colors=COL['C1'], linewidths=1.2)
        ax.contour(xg, yg, m1['ring'].astype(float), levels=[0.5], colors=COL['BKG'],
                   linewidths=1)
        x, y = m2['xy_fit']
        ax.plot(x - ox, y - oy, '+', color=COL['C2'], ms=10, mew=1.5)
        for (nx_, ny_) in _G['geo'][i]['neigh']:
            ax.plot(nx_ - ox, ny_ - oy, 'x', color=COL['EXTRA'], ms=6)
        ax.set_xlim(-0.5, im.shape[1] - 0.5)
        ax.set_ylim(-0.5, im.shape[0] - 0.5)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]), ax.set_yticks([])

    def band(ax, y, e, col, lab, lw=1.0):
        ax.plot(wave, y, color=col, lw=lw, label=lab)
        ax.fill_between(wave, y - e, y + e, color=col, alpha=0.25, lw=0)

    ax = fig.add_subplot(gs[1, :])
    band(ax, m1['A'], m1['A_err'], COL['A'], 'aperture A(λ)')
    band(ax, m1['N_ap'] * m1['B_cont'], m1['N_ap'] * m1['B_err'], COL['BKG'],
         f'N_ap × annulus continuum (N_ann={m1["N_ann"]})')
    band(ax, m1['C1'], m1['C1_err'], COL['C1'], 'M1 cluster = A − N_ap·B_cont')
    band(ax, m2['C2_AP'], m2['C2_AP_err'], COL['C2'], 'M2 cluster PSF in aperture')
    lo, hi = np.nanpercentile(np.r_[m1['A'], m1['C1'], m2['C2_AP']], [0.5, 99.0])
    ax.set_ylim(min(lo, 0) * 1.1, hi * 1.15)
    ax.axhline(0, color='#c3c2b7', lw=0.8)
    ax.set(xlabel='observed wavelength [Å]', ylabel='F$_λ$ [10$^{-20}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]')
    ax.legend(frameon=False, fontsize=8, ncol=4, loc='upper right')
    fpsf = res['f_psf']
    r = inp['phot_row'].get(int(sid))
    ax.set_title(f'{inp["gal"]} {"#" + str(r) + " = " if r is not None else ""}ID {sid} '
                 f'({str(row["PHANGS_CC_CLASS_ML"]).strip()}), '
                 f'z={z:.5f}, shift M2=({m2["shift"][0]:.2f},{m2["shift"][1]:.2f}) px, '
                 f'f_PSF(Hα)={fpsf["Ha"][0]:.3f}±{fpsf["Ha"][1]:.3f}', fontsize=10)

    zooms = [(4840, 5030, 'Hβ, [OIII]'), (6530, 6750, 'Hα, [NII], [SII]')]
    for k, (l0, l1, t) in enumerate(zooms):
        ax = fig.add_subplot(gs[2, 2 * k:2 * k + 2])
        s = (wave > l0 * (1 + z)) & (wave < l1 * (1 + z))
        band(ax, m1['A'], m1['A_err'], COL['A'], 'aperture')
        band(ax, m1['C1'], m1['C1_err'], COL['C1'], 'M1 cluster')
        band(ax, m2['C2_AP'], m2['C2_AP_err'], COL['C2'], 'M2 cluster (in ap.)')
        band(ax, m2['EXTRA2'], m2['EXTRA2_err'], COL['EXTRA'], 'M2 extra = A − cluster')
        yy = np.r_[m1['A'][s], m1['C1'][s], m2['EXTRA2'][s], m2['C2_AP'][s]]
        lo, hi = np.nanmin(yy), np.nanmax(yy)
        ax.set_xlim(l0 * (1 + z), l1 * (1 + z))
        ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.08 * (hi - lo))
        ax.axhline(0, color='#c3c2b7', lw=0.8)
        ax.set(xlabel='observed wavelength [Å]', title=t)
        if k == 0:
            ax.legend(frameon=False, fontsize=8)
    tag = plot_tag(inp, sid, str(row['PHANGS_CC_CLASS_ML']).strip())
    fig.savefig(os.path.join(_G['plot_dir'], f'{tag}_spectra.png'), dpi=110)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────────────────────────

ARRAYS_M1 = ['A', 'A_err', 'B', 'B_err', 'B_cont', 'C1', 'C1_err',
             'BKG_LINES_AP', 'BKG_LINES_AP_err']
ARRAYS_M2 = ['C2', 'C2_err', 'EE_AP', 'C2_AP', 'C2_AP_err', 'E_AP', 'E_AP_err',
             'NEIGH_AP', 'RES_AP', 'EXTRA2', 'EXTRA2_err', 'CHI2NU']


def f_psf(res):
    """Per line: (PSF line flux in aperture) / (aperture line flux)."""
    out = {}
    for n in msa.FIT_LINES:
        fp, ep = res['lines']['C2AP'][n]
        fa, ea = res['lines']['A'][n]
        r = fp / fa if fa else np.nan
        out[n] = (r, abs(r) * np.sqrt((ep / fp)**2 + (ea / fa)**2) if fp and fa else np.nan)
    return out


def write_output(inp, results, psf, psf_summary, opt):
    tab = inp['tab']
    t = Table()
    for c in ('INDEX', 'ID_PHANGS_CLUSTER', 'PHANGS_RA', 'PHANGS_DEC',
              'PHANGS_CC_CLASS_ML', 'PHANGS_DMPC'):
        t[c] = tab[c]
    t['PHOT_ROW'] = [inp['phot_row'].get(int(v), -1) for v in tab['ID_PHANGS_CLUSTER']]
    t['z_local'] = [r['z'] for r in results]
    t['N_ap'] = [r['m1']['N_ap'] for r in results]
    t['N_ann'] = [r['m1']['N_ann'] for r in results]
    t['psf_shift_x'] = [r['m2']['shift'][0] for r in results]
    t['psf_shift_y'] = [r['m2']['shift'][1] for r in results]
    t['n_neigh_fit'] = [r['m2']['n_neigh'] for r in results]
    t['n_blended'] = [r['m2']['n_blended'] for r in results]
    t['EE_AP'] = [np.nanmedian(r['m2']['EE_AP']) for r in results]
    for k in results[0]['err_scale']:
        t[f'err_scale_{k}'] = [r['err_scale'][k] for r in results]
    for comp in ('A', 'C1', 'BKGL', 'C2', 'C2AP', 'EXTRA2', 'EAP'):
        for n in msa.FIT_LINES:
            t[f'{n}_{comp}'] = [r['lines'][comp][n][0] for r in results]
            t[f'{n}_{comp}_err'] = [r['lines'][comp][n][1] for r in results]
    for n in msa.FIT_LINES:
        t[f'fpsf_{n}'] = [r['f_psf'][n][0] for r in results]
        t[f'fpsf_{n}_err'] = [r['f_psf'][n][1] for r in results]
    for (k, b) in results[0]['synth']:
        t[f'synth_{b}_{k}_mJy'] = [r['synth'][(k, b)][0] for r in results]
        t[f'synth_{b}_{k}_mJy_err'] = [r['synth'][(k, b)][1] for r in results]
        t[f'synth_{b}_cover'] = [r['synth'][(k, b)][2] for r in results]
    for lo, hi in msa.CONT_BINS:
        s = (inp['wave'] >= lo) & (inp['wave'] < hi)
        for k, key in (('C1', 'm1'), ('C2', 'm2'), ('C2_AP', 'm2')):
            t[f'cont_{k}_{int(lo)}_{int(hi)}'] = [np.nanmedian(r[key][k][s]) for r in results]
            # error of the median over the bin's pixels
            t[f'cont_{k}_{int(lo)}_{int(hi)}_err'] = [
                1.253 * np.nanmedian(r[key][k + '_err'][s]) / np.sqrt(np.isfinite(r[key][k][s]).sum())
                for r in results]
    t['flag_shift_max'] = np.hypot(t['psf_shift_x'], t['psf_shift_y']) > 1.95

    hdr = fits.Header()
    hdr['BUNIT'] = '1e-20 erg/s/cm2/A'
    hdr['LINEUNIT'] = '1e-20 erg/s/cm2'
    hdr['APERTURE'] = opt.aperture
    hdr['GAP_FWHM'] = opt.gap_fwhm
    hdr['WID_FWHM'] = opt.width_fwhm
    hdr['FITHALF'] = opt.fit_half
    hdr['RIDGE'] = opt.ridge
    hdr['SMOOTHBG'] = opt.smooth
    hdr['PSFFALLB'] = (bool(psf_summary.get('fallback')), 'PSF not measured: token FWHM used')
    hdr['WAVE_LO'] = (inp['wave_range'][0], 'observed A, lower end used')
    hdr['WAVE_HI'] = (inp['wave_range'][1], 'observed A, upper end used')
    for k, v in psf.to_dict().items():
        hdr[('PSF_' + k.upper())[:8]] = v
    hdus = [fits.PrimaryHDU(header=hdr),
            fits.ImageHDU(inp['wave'], name='WAVE'),
            fits.BinTableHDU(t, name='SOURCES')]
    for k in ARRAYS_M1:
        hdus.append(fits.ImageHDU(np.array([r['m1'][k] for r in results], np.float32),
                                  name=f'M1_{k}'.upper()))
    for k in ARRAYS_M2:
        hdus.append(fits.ImageHDU(np.array([r['m2'][k] for r in results], np.float32),
                                  name=f'M2_{k}'.upper()))
    ps = Table({'wave': psf_summary['wave'], 'fwhm_px': psf_summary['fwhm_med'],
                'fwhm_err_px': psf_summary['fwhm_err']})
    hdus.append(fits.BinTableHDU(ps, name='PSF'))
    path = os.path.join(inp['outd'], f'{inp["gal"]}_muse_spectra.fits')
    fits.HDUList(hdus).writeto(path, overwrite=True)
    logger.info(f'  wrote {path}')
    return t


def plot_summary(inp, t, plot_dir):
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
    cls = np.char.strip(np.array(t['PHANGS_CC_CLASS_ML'], dtype=str))
    ccol = {'YCL': COL['C1'], 'MAP': COL['BKG'], 'OGCC': COL['C2']}

    # M1 / M2 continuum ratio vs aperture size
    lo, hi = msa.CONT_BINS[1]
    k = f'{int(lo)}_{int(hi)}'
    c1, e1 = np.array(t[f'cont_C1_{k}']), np.array(t[f'cont_C1_{k}_err'])
    c2, e2 = np.array(t[f'cont_C2_AP_{k}']), np.array(t[f'cont_C2_AP_{k}_err'])
    r = c1 / c2
    re = np.abs(r) * np.hypot(e1 / c1, e2 / c2)
    for cc, col in ccol.items():
        m = cls == cc
        ax[0].errorbar(np.array(t['N_ap'])[m], r[m], re[m], fmt='o', ms=5, color=col,
                       label=cc, lw=1)
    ax[0].axhline(1, color='#c3c2b7', lw=1)
    ax[0].set(xscale='log', yscale='log', xlabel='aperture size N$_{ap}$ [spaxels]',
              ylabel=f'continuum M1 / M2$_{{ap}}$ ({int(lo)}–{int(hi)} Å)',
              title='Annulus (M1) vs PSF fit (M2) continuum')
    ax[0].legend(frameon=False, fontsize=8)

    # f_PSF per line by class
    names = ['Ha', 'NII6583', 'SII6716', 'OIII5007', 'Hb']
    for kk, (cc, col) in enumerate(ccol.items()):
        m = cls == cc
        for j, n in enumerate(names):
            fa = np.array(t[f'{n}_A'])
            snr = fa / np.array(t[f'{n}_A_err'])
            s_ = m & (snr > 5)
            y = np.array(t[f'fpsf_{n}'])[s_]
            e = np.array(t[f'fpsf_{n}_err'])[s_]
            xj = j + (kk - 1) * 0.22 + np.linspace(-0.06, 0.06, max(s_.sum(), 1))[:s_.sum()]
            ax[1].errorbar(xj, y, e, fmt='o', ms=4, color=col,
                           label=cc if j == 0 else None, lw=1)
    ax[1].set_xticks(range(len(names)), ['Hα', '[NII]6583', '[SII]6716', '[OIII]5007', 'Hβ'])
    ax[1].axhline(1, color='#c3c2b7', lw=1)
    ax[1].axhline(0, color='#c3c2b7', lw=1)
    ax[1].set(ylabel='f$_{PSF}$ = cluster-PSF line flux / aperture line flux',
              title='Aperture line emission attributable to the cluster\n'
                    '(aperture line S/N > 5; < 0: cluster in absorption)',
              ylim=(-1.5, 1.6))
    ax[1].legend(frameon=False, fontsize=8)

    # [NII] kept by M1 vs assigned to the cluster by M2
    for n, mk in (('NII6583', 'o'), ('Ha', 's')):
        x, xe = np.array(t[f'{n}_C1']), np.array(t[f'{n}_C1_err'])
        y, ye = np.array(t[f'{n}_C2AP']), np.array(t[f'{n}_C2AP_err'])
        ax[2].errorbar(x, y, ye, xerr=xe, fmt=mk, ms=5, lw=1,
                       color=COL['C1'] if n == 'Ha' else COL['EXTRA'],
                       label={'Ha': 'Hα', 'NII6583': '[NII]6583'}[n])
    lim = np.nanmax(np.abs(np.r_[t['NII6583_C1'], t['Ha_C1']])) * 1.5
    ax[2].plot([-lim, lim], [-lim, lim], color='#c3c2b7', lw=1)
    ax[2].axhline(0, color='#c3c2b7', lw=0.8)
    ax[2].set(xscale='symlog', yscale='symlog', xlabel='line flux in M1 cluster spectrum',
              ylabel='line flux in M2 cluster (PSF, in aperture)',
              title='Line emission: M1 keeps the ISM, M2 removes it')
    for a_ in (ax[2],):
        a_.xaxis.set_major_locator(matplotlib.ticker.SymmetricalLogLocator(base=10, linthresh=1))
    ax[2].legend(frameon=False, fontsize=8)
    fig.savefig(os.path.join(plot_dir, f'{inp["gal"]}_summary.png'), dpi=120)
    plt.close(fig)


def plot_photometry_comparison(inp, t, plot_dir):
    """Synthetic F555W/F814W from the spectra vs the imaging (check only)."""
    outd, gal = inp['outd'], inp['gal']
    try:
        v2 = Table.read(os.path.join(outd, f'{gal}_photometry_v2.fits'))
        nat = Table.read(os.path.join(outd, f'{gal}_photometry_native.fits'))
        ins = Table.read(os.path.join(outd, f'{gal}_photometry_instrument.fits'))
    except FileNotFoundError as e:
        logger.warning(f'  no photometry for the comparison: {e}')
        return
    idx = {int(i): k for k, i in enumerate(v2['ID_PHANGS_CLUSTER'])}
    rows = [(k, idx[int(i)]) for k, i in enumerate(t['ID_PHANGS_CLUSTER']) if int(i) in idx]
    if not rows:
        return
    ks, ps = map(np.array, zip(*rows))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, b in zip(axes, FILTERS):
        cover = np.nanmedian(t[f'synth_{b}_cover'])
        if not np.isfinite(np.array(t[f'synth_{b}_C2_mJy'], float)).any():
            ax.text(0.5, 0.5, f'{b}: MUSE covers only {100 * cover:.0f}% of the filter',
                    transform=ax.transAxes, ha='center')
            ax.set_axis_off()
            continue
        for col, key, err_key, lab, src in (
                (COL['C2'], f'flux_instP_{b}', f'flux_instP_{b}_err', 'M2 vs instP', 'C2'),
                (COL['C1'], f'flux_native_{b}', f'flux_native_{b}_err', 'M1 / EE$_{ap}$ vs native', 'C1')):
            tb = ins if 'instP' in key else nat
            if key not in tb.colnames:
                continue
            x = np.array(tb[key])[ps]
            xe = np.array(tb[err_key])[ps] if err_key in tb.colnames else None
            # M1 is the light inside the aperture: correct it to total with
            # the PSF fraction in the aperture (point-source assumption)
            ap_corr = 1.0 if src == 'C2' else 1.0 / np.array(t['EE_AP'])[ks]
            y = np.array(t[f'synth_{b}_{src}_mJy'])[ks] * ap_corr
            ye = np.array(t[f'synth_{b}_{src}_mJy_err'])[ks] * ap_corr
            ax.errorbar(x, y, ye, xerr=xe, fmt='o', ms=4, color=col, label=lab, lw=1)
            r = y / x
            logger.info(f'  {b} {lab}: median ratio {np.nanmedian(r):.3f}, '
                        f'scatter (MAD) {1.4826 * np.nanmedian(np.abs(r - np.nanmedian(r))):.3f}, '
                        f'N={np.isfinite(r).sum()}')
        lim = [1e-4, 1]
        good = np.isfinite(np.array(t[f'synth_{b}_C2_mJy']))
        if good.any():
            vals = np.abs(np.array(t[f'synth_{b}_C2_mJy'])[good])
            lim = [np.nanmin(vals[vals > 0]) / 2, np.nanmax(vals) * 2]
        ax.plot(lim, lim, color='#c3c2b7', lw=1)
        ax.set(xscale='log', yscale='log', xlabel=f'imaging {b} [mJy]',
               ylabel=f'synthetic {b} from MUSE [mJy]',
               title=f'{b} (MUSE covers {100 * cover:.0f}% of the filter)')
        ax.legend(frameon=False, fontsize=8)
    fig.savefig(os.path.join(plot_dir, f'{gal}_spectra_vs_photometry.png'), dpi=120)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────

def spectra_settings(path):
    """[muse_spectra] section of the config, with the built-in defaults for missing keys."""
    p = configparser.ConfigParser(inline_comment_prefixes=('#', ';'))
    p.read(path)
    s = 'muse_spectra'
    return {'aperture': p.get(s, 'aperture', fallback='MUSE').strip().upper(),
            'wave_range': [p.getfloat(s, 'wave_min', fallback=msa.WAVE_RANGE[0]),
                           p.getfloat(s, 'wave_max', fallback=msa.WAVE_RANGE[1])],
            'gap_fwhm': p.getfloat(s, 'gap_fwhm', fallback=1.0),
            'width_fwhm': p.getfloat(s, 'width_fwhm', fallback=2.0),
            'smooth': p.getboolean(s, 'smooth', fallback=False),
            'fit_half': p.getint(s, 'fit_half', fallback=20),
            'ridge': p.getfloat(s, 'ridge', fallback=1e-2),
            'n_workers': p.getint(s, 'n_workers', fallback=4),
            'plots': p.getboolean(s, 'plots', fallback=True),
            'overwrite': p.getboolean(s, 'overwrite', fallback=False)}


def spectra_output_path(cfg, token):
    gal = token.split('-')[0]
    return os.path.join(cfg['output_dir'], gal, f'{gal}_muse_spectra.fits')


def spectra_up_to_date(cfg, token):
    """True when <GAL>_muse_spectra.fits exists and is newer than the ISF
    products it is built from (<GAL>_matched.fits)."""
    gal = token.split('-')[0]
    out = spectra_output_path(cfg, token)
    matched = os.path.join(cfg['output_dir'], gal, f'{gal}_matched.fits')
    return (os.path.exists(out) and os.path.exists(matched)
            and os.path.getmtime(out) >= os.path.getmtime(matched))


def run_galaxy_spectra(token, cfg, settings, ids=None):
    """
    M1 (annulus) and M2 (PSF fit) MUSE spectra of every ISF source of one
    galaxy token (e.g. 'NGC1365-1.15asec'); what main() does, callable from
    run_full_pipeline. cfg: run_full_pipeline.load_config(...) (paths);
    settings: spectra_settings(...) (the [muse_spectra] values, possibly
    overridden); ids: optional iterable of ID_PHANGS_CLUSTER to restrict to.
    Writes <GAL>_muse_spectra.fits and muse_spectra/ and returns
    dict(n_sources, path, psf_fwhm_arcsec).
    """
    from types import SimpleNamespace
    opt = SimpleNamespace(**settings)
    inp = load_inputs(token, cfg, opt.aperture, tuple(opt.wave_range))
    plot_dir = os.path.join(inp['outd'], 'muse_spectra')
    os.makedirs(plot_dir, exist_ok=True)
    logger.info(f'{inp["gal"]}: {len(inp["tab"])} sources, vsys={inp["vsys"]} km/s')

    fwhm0 = float(token.split('-')[1].replace('asec', '')) / msa.MUSE_PIX_ARCSEC
    geo, half = source_geometry(inp, fwhm0, opt.gap_fwhm, opt.width_fwhm, opt.fit_half)
    logger.info(f'  cutout half-size {half} px')
    paths = get_cutouts(inp, geo, half, os.path.join(plot_dir, 'cutouts'))

    psf_json = os.path.join(plot_dir, f'{inp["gal"]}_psf.json')
    psf, psf_summary, _ = measure_psf(inp, geo, paths, fwhm0, plot_dir, plot=opt.plots)
    with open(psf_json, 'w') as f:
        json.dump({**psf.to_dict(), 'fwhm_arcsec_at_ref': psf.fwhm_ref * msa.MUSE_PIX_ARCSEC,
                   'n_fits': psf_summary['n_used'],
                   'bins': {f'{w:.0f}': [float(m), float(e)] for w, m, e in
                            zip(psf_summary['wave'], psf_summary['fwhm_med'],
                                psf_summary['fwhm_err'])}}, f, indent=1)
    logger.info(f'  PSF: FWHM({psf.wave_ref:.0f} Å) = {psf.fwhm_ref * 0.2:.3f}", '
                f'slope {psf.slope * 0.2 * 1000:.4f}"/1000 Å, beta {psf.beta:.2f} '
                f'({psf_summary["n_used"]} fits)')

    sel = np.arange(len(inp['tab']))
    if ids:
        want = {int(v) for v in ids}
        sel = np.array([i for i in sel if int(inp['tab']['ID_PHANGS_CLUSTER'][i]) in want])
        inp['tab'] = inp['tab'][sel]
        inp['regs'] = [inp['regs'][i] for i in sel]
        geo = [geo[i] for i in sel]
        sel = np.arange(len(sel))

    _G.update({'inp': inp, 'geo': geo, 'paths': paths, 'psf': psf, 'opt': opt,
               'plot_dir': plot_dir})

    t0 = time.time()
    if opt.n_workers > 1:
        with mp.get_context('fork').Pool(opt.n_workers) as pool:
            results = pool.map(_worker, sel)
    else:
        results = [_worker(i) for i in sel]
    logger.info(f'  {len(results)} sources in {time.time() - t0:.0f} s')

    t = write_output(inp, results, psf, psf_summary, opt)
    if opt.plots:
        plot_summary(inp, t, plot_dir)
        plot_photometry_comparison(inp, t, plot_dir)
    return {'n_sources': len(results),
            'path': os.path.join(inp['outd'], f'{inp["gal"]}_muse_spectra.fits'),
            'psf_fwhm_arcsec': float(psf.fwhm_ref * msa.MUSE_PIX_ARCSEC)}


def main():
    # the config is read first, so that its [muse_spectra] values become the defaults
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', default=os.path.join(HERE, 'config.ini'))
    config_path = pre.parse_known_args()[0].config

    ap = argparse.ArgumentParser(description=__doc__, parents=[pre],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--galaxy', default='NGC1365-1.15asec', help='MUSE token')
    ap.add_argument('--aperture', choices=['MUSE', 'HST'])
    ap.add_argument('--gap-fwhm', type=float)
    ap.add_argument('--width-fwhm', type=float)
    ap.add_argument('--smooth', action=argparse.BooleanOptionalAction,
                    help='spline annulus continuum instead of line interpolation')
    ap.add_argument('--fit-half', type=int, help='M2 stamp half-size [px]')
    ap.add_argument('--ridge', type=float, help='M2 patch ridge fraction')
    ap.add_argument('--ids', type=str, default='', help='comma list of ID_PHANGS_CLUSTER')
    ap.add_argument('--wave-range', type=float, nargs=2,
                    help='observed wavelength range used [A]')
    ap.add_argument('--n-workers', type=int)
    ap.add_argument('--plots', action=argparse.BooleanOptionalAction,
                    help='write the figures (--no-plots: FITS output only)')
    ap.set_defaults(**spectra_settings(config_path))
    opt = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    from run_full_pipeline import load_config      # same [paths] handling as the pipeline
    cfg = load_config(opt.config)
    settings = {k: getattr(opt, k) for k in spectra_settings(config_path)}
    ids = [int(v) for v in opt.ids.split(',')] if opt.ids else None
    run_galaxy_spectra(opt.galaxy, cfg, settings, ids=ids)


def _worker(i):
    try:
        return process_source(i)
    except Exception:
        logger.exception(f'source row {i} failed')
        raise


if __name__ == '__main__':
    main()
