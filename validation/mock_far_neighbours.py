"""
Mock clusters with "far" neighbours (0.6-1.2", like most real NGC1365
clusters) over a grid of S/N: instP vs the Deger+22 recipe only.

Each mock system is rendered in its own 10" field on the real pixel grid of
every HST and NIRCam band (a WCS cutout of NGC1365's image), with the band's
galaxy-matched instrument PSF, a flat sky pedestal and Gaussian noise:

  cluster     EFF gamma=3, total 1 mJy in every band (flat SED), R_eff 2 or 10 pc
              at 19.57 Mpc (the real instP sizes are mostly 0.2-3 pc; 10 pc is
              the previous neighbour test). Random sub-pixel position.
  neighbours  Gaussians (intrinsic FWHM 0-0.15"), flat SEDs, at 0.6-1.2":
                clean        none
                far1         1 neighbour,  0.1-0.9 mJy
                far3         3 neighbours, 0.1-0.9 mJy   (the previous "far3")
                far6         6 neighbours, 0.1-0.9 mJy
                far3_bright  3 neighbours, 1-3 mJy       (brighter than the cluster)
  S/N         10, 20, 50, 100, 200: the cluster's S/N within 0.158" (4 HST px),
              equal in every band (the real F555W S/N is 30-190, 5-95%)

Methods (truth = 1 mJy, so every value is recovered / true):
  instP        pipeline: measure_native_photometry_one_source (centre, aperture
               source) then measure_instrument_photometry_one_source, EFF gamma=3
  deger_ideal  4 px flux (sky 7-8 px) x the EXACT aperture correction of this
               cluster model in this band: the catalog recipe at its best (this
               is the "deger" of mock_neighbours_summary.png)
  deger_1365   4 px flux (sky 7-8 px) x 10**(-0.4 (AC_V + offset_band)),
               AC_V = -0.61 mag: what the NGC1365 catalog really applies
               (HST bands only; there is no catalog recipe for NIRCam)
Errors: instP flux_instP_err from the pipeline; deger sigma_sky sqrt(N_ap) of
the 4 px aperture (sky std from the 7-8 px annulus) x the same factor. The
pull (flux - 1) / error tests whether the error bars are right (std ~ 1).

Usage:  python mock_far_neighbours.py [--n-real 20] [--workers 40] [--plot]
        python mock_far_neighbours.py --diagnostics   (slit profiles + SEDs of a subset)
Writes  <validation_output_dir>/output_mock_far/mock_far.fits (one row per system and
        band), mock_far_HST.png, mock_far_NIRCam.png, mock_far_pull.png,
        mock_far_examples.png (some of the mock fields); with --diagnostics
        slit_profiles/mock_far_<case>.png (per band: image + averaged slit
        profile with the instP fit and the noise-free truth, and the SED) and
        mock_far_SEDs.png (all diagnostic SEDs).
"""
import argparse
import logging
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)
from _paths import OUT_ROOT  # noqa: E402

import astropy.units as u  # noqa: E402
from astropy.coordinates import SkyCoord  # noqa: E402
from astropy.nddata import Cutout2D  # noqa: E402
from astropy.table import Table  # noqa: E402

import run_full_pipeline as rfp  # noqa: E402
from instrument_psf import instrument_psf_for_image  # noqa: E402
from native_phot_aux_func import (measure_native_photometry_one_source,  # noqa: E402
                                  measure_instrument_photometry_one_source, instrument_slit_length,
                                  instrument_fit_model_profile, _sample_averaged_slit_profile)
from injection_test import _ap, _sky, R4, R7, R8, EFF3  # noqa: E402
from deger_aperture_correction import (DEGER22_TABLE2, ac_mag_to_flux_factor, band_ac,  # noqa: E402
                                       aperture_flux_err)
from mock_neighbours_test import load_real, unit_per_mjy, p_jwst_bands  # noqa: E402

OUT = os.path.join(OUT_ROOT, 'output_mock_far')
DIST = 19.57
CUT_ARCSEC = 10.0
STAMP_ARCSEC = 3.0
SNRS = [10, 20, 50, 100, 200]
REFFS = [2.0, 10.0]
GROUPS = {'clean': (0, None), 'far1': (1, (0.1, 0.9)), 'far3': (3, (0.1, 0.9)),
          'far6': (6, (0.1, 0.9)), 'far3_bright': (3, (1.0, 3.0))}
NB_R = (0.6, 1.2)
HST = ['F275W', 'F336W', 'F438W', 'F555W', 'F814W']
METHODS = ['instP', 'deger_ideal', 'deger_1365']
STYLE = {'instP': ('#4a3aa7', 'P', 'instP (PSF x EFF3)'),
         'deger_ideal': ('#2a78d6', 'o', 'Deger, exact correction for this cluster'),
         'deger_1365': ('#eb6834', 's', 'Deger, NGC1365 catalog correction (AC_V = -0.61)')}
INK, INK2, GRID = '#1a1a19', '#5f5e58', '#e4e3dd'


# ─────────────────────────────────────────────────────────────────────────────
# setup (once, in the parent; workers inherit it at fork)
# ─────────────────────────────────────────────────────────────────────────────

def build_context():
    p = load_real()
    bands = {b: v for b, v in rfp.FILTER_WAVELENGTHS.items()
             if b.lower() in p.hst_images or (b.lower() in p.jwst_images and int(b[1:-1]) < 500)}
    p_jwst_bands.update(b for b in bands if b.lower() in p.jwst_images)
    # one-source dummy table: run_native_photometry only to build the per-band setup
    # (calibration curves etc., exactly as in the pipeline); its output goes to OUT
    _, _, w5 = p.hst_images['f555w']
    ny5, nx5 = p.hst_images['f555w'][0].shape
    center = w5.pixel_to_world(nx5 / 2, ny5 / 2)
    p.gal, p.base_dir, p.eb_v = 'MOCK_FAR_SETUP', OUT, 0.0
    p.photometry_table = Table(dict(ID=[0], PHANGS_RA=[center.ra.deg], PHANGS_DEC=[center.dec.deg],
                                    PHANGS_CC_CLASS_ML=['YCL'], PHANGS_DMPC=[DIST]))
    p.final_table = p.photometry_table.copy()
    p.run_native_photometry(bands, verbose=False, force_recompute=True, plot_diagnostics=False,
                            instrument_fit=False)
    ac_1365 = DEGER22_TABLE2['NGC1365'][0]
    ctx = dict(bands=list(bands), center=center, band_data={}, info={})
    for b in bands:
        bl = b.lower()
        bd = p.native_band_calib[bl]
        fwhm = (p.psf_hst if bd['instrument'] == 'HST' else p.psf_jwst).get(bl)
        model, _ = instrument_psf_for_image(b, bd['header'], bd['pixel_scale_arcsec'], target_fwhm_arcsec=fwhm)
        pix = bd['pixel_scale_arcsec']
        n = int(np.ceil(CUT_ARCSEC / pix)) | 1
        tmpl = Cutout2D(bd['image'], center, (n, n), wcs=bd['wcs'],
                        mode='partial', fill_value=0.0, copy=False)
        upm = unit_per_mjy(b, bd['header'])
        hs = int(np.ceil(STAMP_ARCSEC / pix))
        info = dict(pix=pix, n=n, upm=upm, hs=hs, ac_1365=(float(ac_mag_to_flux_factor(band_ac(ac_1365, b)))
                                                           if b in HST else np.nan))
        for reff in REFFS:
            reff_as = reff * 206265.0 / (DIST * 1e6)
            one = model.render(hs, hs, (2 * hs + 1, 2 * hs + 1), flux=upm, r_eff_arcsec=reff_as,
                               profile='eff', gamma=3.0)
            f4_clean = _ap(one, hs, hs, R4 / pix, _sky(one, hs, hs, R7 / pix, R8 / pix))
            info[('ac_ideal', reff)] = upm / f4_clean
            info[('f4_true', reff)] = _ap(one, hs, hs, R4 / pix, 0.0)
        ctx['band_data'][bl] = dict(bd, psf_model=model, image=None, wcs=tmpl.wcs)
        ctx['info'][b] = info
        print(f'  {b}: {pix:.4f}"/px, field {n} px, AC exact x{info[("ac_ideal", 2.0)]:.3f} (2 pc) '
              f'x{info[("ac_ideal", 10.0)]:.3f} (10 pc), AC NGC1365 x{info["ac_1365"]:.3f}', flush=True)
    return ctx


def make_tasks(n_real, seed):
    rng = np.random.default_rng(seed)
    tasks = []
    for reff in REFFS:
        for g, (n_nb, frange) in GROUPS.items():
            for snr in SNRS:
                for k in range(n_real):
                    nbs = [dict(r=rng.uniform(*NB_R), th=rng.uniform(0, 2 * np.pi),
                                flux=rng.uniform(*frange), fwhm=rng.uniform(0.0, 0.15)) for _ in range(n_nb)]
                    tasks.append(dict(reff=reff, group=g, snr=snr, k=k, nbs=nbs,
                                      dxy=rng.uniform(-0.5, 0.5, 2), seed=int(rng.integers(2 ** 31))))
    return tasks


# ─────────────────────────────────────────────────────────────────────────────
# one mock system (worker)
# ─────────────────────────────────────────────────────────────────────────────

_CTX = {}


def _init(ctx):
    global _CTX
    _CTX = ctx
    for name in ('analysis', 'native_phot_aux_func', 'instrument_psf'):
        logging.getLogger(name).setLevel(logging.ERROR)


def _add(img, model, x, y, hs, **kw):
    ix, iy = int(np.round(x)), int(np.round(y))
    stamp = model.render(hs + x - ix, hs + y - iy, (2 * hs + 1, 2 * hs + 1), **kw)
    ny, nx = img.shape
    y0, x0 = iy - hs, ix - hs
    ys = slice(max(y0, 0), min(y0 + 2 * hs + 1, ny))
    xs = slice(max(x0, 0), min(x0 + 2 * hs + 1, nx))
    img[ys, xs] += stamp[ys.start - y0:ys.stop - y0, xs.start - x0:xs.stop - x0]


def render_band(task, b, ctx, rng, layers=False):
    """noisy mock image of one band; with layers=True also the noise-free
    cluster and neighbour images and the sky pedestal"""
    bd, info = ctx['band_data'][b.lower()], ctx['info'][b]
    pix, n, upm, hs = info['pix'], info['n'], info['upm'], info['hs']
    model = bd['psf_model']
    reff_as = task['reff'] * 206265.0 / (DIST * 1e6)
    xc = (n - 1) / 2 + task['dxy'][0] * 0.04 / pix          # sub-pixel shift fixed on the sky
    yc = (n - 1) / 2 + task['dxy'][1] * 0.04 / pix
    clu, nbi = np.zeros((n, n)), np.zeros((n, n))
    _add(clu, model, xc, yc, hs, flux=upm, r_eff_arcsec=reff_as, profile='eff', gamma=3.0)
    for nb in task['nbs']:
        _add(nbi, model, xc + nb['r'] * np.cos(nb['th']) / pix, yc + nb['r'] * np.sin(nb['th']) / pix, hs,
             flux=nb['flux'] * upm,
             intrinsic_fwhm_arcsec=nb['fwhm'] if nb['fwhm'] >= 0.01 else 0.0)   # tiny Gaussian kernel: point
    sigma = info[('f4_true', task['reff'])] / (task['snr'] * np.sqrt(np.pi * (R4 / pix) ** 2))
    img = clu + nbi + 10.0 * sigma + rng.normal(0.0, sigma, (n, n))
    if layers:
        return img, xc, yc, clu, nbi, 10.0 * sigma
    return img, xc, yc


def run_system(task, ctx=None, details=False):
    """measure one mock system in every band; details=True also returns the
    images, noise-free layers and the pipeline results (for the diagnostics)"""
    ctx = ctx or _CTX
    rng = np.random.default_rng(task['seed'])
    cut_bd, geo, lay = {}, {}, {}
    for b in ctx['bands']:
        img, xc, yc, clu, nbi, ped = render_band(task, b, ctx, rng, layers=True)
        cut_bd[b.lower()] = dict(ctx['band_data'][b.lower()], image=img)
        geo[b] = (xc, yc)
        lay[b] = (clu, nbi, ped)
    w = cut_bd['f555w']['wcs']
    sky = w.pixel_to_world(*geo['F555W'])
    ra, dec = float(sky.ra.deg), float(sky.dec.deg)
    try:
        nat = measure_native_photometry_one_source(ra, dec, cut_bd, eb_v=0.0)
        native_row = {bl: dict(aperture_source=r.aperture_source, centroid_ra=r.centroid_ra,
                               centroid_dec=r.centroid_dec) for bl, r in nat.items()}
        ins = measure_instrument_photometry_one_source(ra, dec, native_row, cut_bd, EFF3,
                                                       distance_mpc=DIST, eb_v=0.0)
        err = ''
    except Exception as e:                                   # keep the batch alive
        nat, ins, err = None, None, f'{type(e).__name__}: {e}'[:80]
    rows = []
    for b in ctx['bands']:
        info, img = ctx['info'][b], cut_bd[b.lower()]['image']
        pix, upm = info['pix'], info['upm']
        x, y = geo[b]
        f4 = _ap(img, x, y, R4 / pix, _sky(img, x, y, R7 / pix, R8 / pix)) / upm
        e4 = aperture_flux_err(img, x, y, R4 / pix, R7 / pix, R8 / pix)[0] / upm
        k_id, k_cat = info[('ac_ideal', task['reff'])], info['ac_1365']
        row = dict(reff_pc=task['reff'], group=task['group'], snr=task['snr'], real=task['k'], band=b,
                   n_neighbours=len(task['nbs']),
                   nearest_nb_arcsec=min([nb['r'] for nb in task['nbs']], default=np.nan),
                   nb_flux_mJy=float(sum(nb['flux'] for nb in task['nbs'])),
                   deger_ideal=f4 * k_id, deger_ideal_err=e4 * k_id,
                   deger_1365=f4 * k_cat, deger_1365_err=e4 * k_cat,
                   instP=np.nan, instP_err=np.nan, instP_reff_pc=np.nan, instP_size_source='',
                   instP_status='', error=err)
        if ins is not None:
            r = ins[b.lower()]['instP']
            row.update(instP=r.flux_mJy, instP_err=r.flux_err_mJy, instP_reff_pc=r.r_eff_pc,
                       instP_size_source=str(r.size_source), instP_status=str(r.status))
        rows.append(row)
    if details:
        return rows, dict(cut_bd=cut_bd, geo=geo, layers=lay, nat=nat, ins=ins)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# summary
# ─────────────────────────────────────────────────────────────────────────────

def q(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return (np.median(v), *np.percentile(v, [16, 84])) if v.size else (np.nan,) * 3


def plot_set(t, bands, title, out_png):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                         'ytick.color': INK2, 'text.color': INK, 'axes.linewidth': 0.8})
    methods = [m for m in METHODS if np.isfinite(np.asarray(t[m][np.isin(t['band'], bands)], float)).any()]
    fig, axes = plt.subplots(len(REFFS), len(GROUPS), figsize=(3.3 * len(GROUPS), 3.4 * len(REFFS)),
                             sharex=True, sharey=True)
    for i, reff in enumerate(REFFS):
        for j, g in enumerate(GROUPS):
            ax = axes[i, j]
            ax.axhline(1.0, color=INK2, lw=1, ls='--', zorder=1)
            ax.grid(color=GRID, lw=0.6, which='both')
            ax.set_axisbelow(True)
            for k, m in enumerate(methods):
                col, mk, lbl = STYLE[m]
                med, lo, hi, err = [], [], [], []
                for snr in SNRS:
                    s = (t['reff_pc'] == reff) & (t['group'] == g) & (t['snr'] == snr) & np.isin(t['band'], bands)
                    a, b16, b84 = q(t[m][s])
                    med.append(a); lo.append(b16); hi.append(b84)
                    err.append(np.nanmedian(np.asarray(t[f'{m}_err'][s], float)))
                x = np.array(SNRS) * (1 + 0.07 * (k - 1))
                med, lo, hi, err = map(np.array, (med, lo, hi, err))
                ax.fill_between(x, lo, hi, color=col, alpha=0.15, lw=0)
                ax.errorbar(x, med, yerr=err, fmt=mk + '-', color=col, ms=6, mec='white', mew=0.8,
                            lw=1.3, elinewidth=1.3, capsize=2.5, label=lbl)
            ax.set_xscale('log')
            ax.set_ylim(0.4, 1.6)
            if i == 0:
                n_nb, fr = GROUPS[g]
                ax.set_title(f'{g}\n' + (f'{n_nb} nb at 0.6-1.2", {fr[0]}-{fr[1]} mJy' if n_nb else 'no neighbours'),
                             fontsize=9.5)
            if j == 0:
                ax.set_ylabel(f'R_eff = {reff:g} pc\nrecovered / true')
            if i == len(REFFS) - 1:
                ax.set_xlabel('S/N within 0.158"')
                ax.set_xticks(SNRS); ax.set_xticklabels([str(s) for s in SNRS]); ax.minorticks_off()
    leg = {}
    for ax in axes.ravel():                      # every method, even if a panel has no data
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            leg.setdefault(ll, hh)
    fig.legend(list(leg.values()), list(leg), loc='lower center', bbox_to_anchor=(0.5, 1.0),
               ncol=max(len(leg), 1), frameon=False, fontsize=9)
    fig.suptitle(f'{title}: median of all realizations and bands; error bar = median reported 1$\\sigma$ '
                 f'error; shaded = 16-84% scatter', x=0.01, ha='left', y=1.06, fontsize=11)
    plt.tight_layout()
    fig.savefig(out_png, dpi=130, facecolor='#fcfcfb', bbox_inches='tight')
    plt.close(fig)


def plot_pull(t, out_png):
    """std of (flux - 1) / error vs S/N: 1 means the error bars describe the real scatter.
    The scatter is measured around each cell's median, so a bias does not count as error."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(REFFS), figsize=(6 * len(REFFS), 4), sharey=True)
    for ax, reff in zip(axes, REFFS):
        for m in METHODS:
            col, mk, lbl = STYLE[m]
            for g, ls in zip(['clean', 'far3', 'far3_bright'], ['-', '--', ':']):
                y = []
                for snr in SNRS:
                    s = (t['reff_pc'] == reff) & (t['group'] == g) & (t['snr'] == snr)
                    f, e, bb = np.asarray(t[m][s], float), np.asarray(t[f'{m}_err'][s], float), np.asarray(t['band'][s])
                    pulls = []
                    for b in set(bb):
                        mb = (bb == b) & np.isfinite(f) & np.isfinite(e) & (e > 0)
                        if mb.sum() > 3:
                            pulls.append((f[mb] - np.median(f[mb])) / e[mb])
                    pulls = np.concatenate(pulls) if pulls else np.array([np.nan])
                    y.append(0.7413 * np.subtract(*np.nanpercentile(pulls, [75, 25])))   # robust std
                ax.plot(SNRS, y, mk + ls, color=col, ms=5, mec='white', mew=0.6,
                        label=f'{lbl.split(",")[0] if m != "instP" else "instP"}{", NGC1365 AC" if m == "deger_1365" else ""}  {g}')
        ax.axhline(1, color=INK2, lw=1)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xticks(SNRS); ax.set_xticklabels([str(s) for s in SNRS]); ax.minorticks_off()
        ax.set_xlabel('S/N within 0.158"')
        ax.set_title(f'R_eff = {reff:g} pc', fontsize=10)
        ax.grid(color=GRID, lw=0.6, which='both')
    axes[0].set_ylabel('scatter / reported error\n(robust std of the pull)')
    axes[-1].legend(fontsize=7, ncol=1, loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.suptitle('Are the error bars right? 1 = reported 1$\\sigma$ errors match the real scatter', x=0.01,
                 ha='left', fontsize=11)
    plt.tight_layout()
    fig.savefig(out_png, dpi=130, facecolor='#fcfcfb', bbox_inches='tight')
    plt.close(fig)


def plot_examples(ctx, tasks, out_png):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    picks = []
    for g in GROUPS:
        for snr in (20, 100):
            picks.append(next(tk for tk in tasks if tk['group'] == g and tk['snr'] == snr and tk['reff'] == 2.0))
    fig, axes = plt.subplots(2, len(GROUPS), figsize=(3 * len(GROUPS), 6.2))
    for ax, tk in zip(axes.T.ravel(), picks):
        img, xc, yc = render_band(tk, 'F555W', ctx, np.random.default_rng(tk['seed']))
        pix = ctx['info']['F555W']['pix']
        h = int(1.6 / pix)
        sub = img[int(yc) - h:int(yc) + h + 1, int(xc) - h:int(xc) + h + 1]
        lo, hi = np.percentile(sub, [5, 99.7])
        ax.imshow(sub, origin='lower', cmap='Greys', vmin=lo, vmax=hi,
                  extent=[-h * pix, h * pix, -h * pix, h * pix])
        for r, c, ls in [(R4, '#eb6834', '-'), (R7, '#2a78d6', '--'), (R8, '#2a78d6', '--'), (0.6, '#888', ':'),
                         (1.2, '#888', ':')]:
            ax.add_patch(plt.Circle((0, 0), r, fill=False, color=c, ls=ls, lw=1))
        ax.set_title(f'{tk["group"]}, S/N {tk["snr"]}', fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle('F555W examples (R_eff 2 pc): orange 4 px aperture, blue 7-8 px sky, grey 0.6-1.2" '
                 'neighbour zone', x=0.01, ha='left', fontsize=10)
    plt.tight_layout()
    fig.savefig(out_png, dpi=110, facecolor='#fcfcfb')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# diagnostics: slit profiles + SEDs for a subset of systems
# ─────────────────────────────────────────────────────────────────────────────
DIAG_SNRS = (20, 100)
N_SLITS, SLIT_W = 12, 1                    # as the instrument fit
ANN_IN, ANN_OUT = 2.6, 4.0                 # pipeline background annulus, x R65


def diag_tasks(tasks):
    """realization 0 of every (R_eff, group) at S/N 20 and 100"""
    return [tk for tk in tasks if tk['k'] == 0 and tk['snr'] in DIAG_SNRS]


def diag_tag(tk):
    return f'reff{tk["reff"]:g}pc_{tk["group"]}_snr{tk["snr"]}'


def _slit_panel(ax, axi, b, ctx, det, row, tk):
    """averaged slit profile through the instP centre: data, fitted instP model
    (amplitude + background by least squares at the fitted R_eff, as the fit
    itself), and the noise-free truth split into cluster and neighbours"""
    import matplotlib.pyplot as plt
    bd = det['cut_bd'][b.lower()]
    img, pix, wcs = bd['image'], bd['pixel_scale_arcsec'], bd['wcs']
    clu, nbi, ped = det['layers'][b]
    xt, yt = det['geo'][b]
    r = det['ins'][b.lower()]['instP'] if det['ins'] is not None else None
    if r is not None and np.isfinite(r.centroid_ra):
        x0, y0 = (float(v) for v in wcs.world_to_pixel(SkyCoord(r.centroid_ra * u.deg, r.centroid_dec * u.deg)))
    else:
        x0, y0 = xt, yt
    reff_as = r.r_eff_arcsec if r is not None and np.isfinite(r.r_eff_arcsec) else 0.0
    r65 = r.r65_arcsec if r is not None else np.nan
    slit_len = instrument_slit_length(bd, 1.0)
    if 5.0 * reff_as > slit_len and slit_len < 3.0:
        slit_len = min(5.0 * reff_as, 3.0)
    along, _, per_slit, y, _ = _sample_averaged_slit_profile(img, x0, y0, pix, slit_len, N_SLITS, SLIT_W)
    yc = _sample_averaged_slit_profile(clu, x0, y0, pix, slit_len, N_SLITS, SLIT_W)[3]
    yn = _sample_averaged_slit_profile(nbi, x0, y0, pix, slit_len, N_SLITS, SLIT_W)[3]
    upm = ctx['info'][b]['upm']
    sc = 1.0 / upm * 1e3                   # image units -> uJy per pixel (truth 1 mJy total)
    for k in range(per_slit.shape[0]):
        g = np.isfinite(per_slit[k])
        ax.plot(along[g], (per_slit[k][g] - ped) * sc, '-', color='#c9c8c2', lw=0.5, zorder=1)
    ok = np.isfinite(y)
    ax.plot(along[ok], (y[ok] - ped) * sc, '.', color=INK, ms=2.5, zorder=3, label='data (avg of 12 slits)')
    ax.plot(along, yc * sc, '-', color='#1baf7a', lw=1.6, zorder=4, label='true cluster')
    if np.nanmax(yn) > 0:
        ax.plot(along, yn * sc, '-', color='#8a8984', lw=1.0, zorder=2, label='true neighbours')
    if r is not None and np.isfinite(reff_as):
        _, unit = instrument_fit_model_profile(bd['psf_model'], x0, y0, reff_as, 'eff', 1.0, 0.0,
                                               slit_len, N_SLITS, SLIT_W, gamma=3.0, king_c=1.5)
        g = ok & np.isfinite(unit)
        (amp, bkg), *_ = np.linalg.lstsq(np.vstack([unit[g], np.ones(g.sum())]).T, y[g], rcond=None)
        ax.plot(along, (amp * unit + bkg - ped) * sc, '-.', color='#4a3aa7', lw=1.5, zorder=5,
                label=f'instP fit: R_eff {r.r_eff_pc:.1f} pc [{r.size_source}]')
        ax.axhline((bkg - ped) * sc, color='#4a3aa7', lw=0.8, ls=':', zorder=2)
        for sgn in (-1, 1):
            ax.axvline(sgn * r65, color='#4a3aa7', ls=':', lw=1)
            ax.axvspan(sgn * ANN_IN * r65, sgn * ANN_OUT * r65, color='#4a3aa7', alpha=0.07, lw=0)
            ax.axvspan(sgn * R7, sgn * R8, color='#2a78d6', alpha=0.15, lw=0)
            ax.axvline(sgn * R4, color='#eb6834', ls='--', lw=1)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xlim(-min(slit_len, 1.4), min(slit_len, 1.4))
    ax.tick_params(labelsize=6.5)
    ax.set_ylabel(f'{b}\nuJy / px', fontsize=7.5)
    ax.text(0.99, 0.97, f'instP {row["instP"]:.3f}$\\pm${row["instP_err"]:.3f}\n'
            f'Deger exact {row["deger_ideal"]:.3f}$\\pm${row["deger_ideal_err"]:.3f}'
            + (f'\nDeger NGC1365 {row["deger_1365"]:.3f}$\\pm${row["deger_1365_err"]:.3f}'
               if np.isfinite(row['deger_1365']) else ''),
            transform=ax.transAxes, ha='right', va='top', fontsize=6.5,
            bbox=dict(fc='white', ec='none', alpha=0.8))
    # image
    h = int(1.4 / pix)
    ix, iy = int(round(xt)), int(round(yt))
    sub = img[iy - h:iy + h + 1, ix - h:ix + h + 1] - ped
    lo, hi = np.percentile(sub, [5, 99.8])
    ext = [(ix - h - 0.5 - xt) * pix, (ix + h + 0.5 - xt) * pix, (iy - h - 0.5 - yt) * pix, (iy + h + 0.5 - yt) * pix]
    axi.imshow(sub, origin='lower', cmap='Greys', vmin=lo, vmax=hi, extent=ext)
    cx, cy = (x0 - xt) * pix, (y0 - yt) * pix
    circ = [(R4, '#eb6834', '-'), (R7, '#2a78d6', '--'), (R8, '#2a78d6', '--')]
    if np.isfinite(r65):
        circ += [(r65, '#4a3aa7', '-'), (ANN_IN * r65, '#4a3aa7', ':'), (ANN_OUT * r65, '#4a3aa7', ':')]
    for rad, c, ls in circ:
        axi.add_patch(plt.Circle((cx, cy), rad, fill=False, color=c, ls=ls, lw=0.9))
    for nb in tk['nbs']:
        axi.plot(nb['r'] * np.cos(nb['th']), nb['r'] * np.sin(nb['th']), '+', color='#c0392b', ms=6, mew=0.9)
    axi.set_xticks([]); axi.set_yticks([])
    return ax


def _sed_panel(ax, rows, ctx, title=None, legend=True):
    wl = np.array([ctx['band_data'][r['band'].lower()]['central_wl_AA'] / 1e4 for r in rows])
    ax.axhline(1.0, color=INK2, lw=1.2, ls='--', label='truth (1 mJy)')
    for k, m in enumerate(METHODS):
        col, mk, lbl = STYLE[m]
        f = np.array([r[m] for r in rows], float)
        e = np.array([r[f'{m}_err'] for r in rows], float)
        g = np.isfinite(f)
        if g.any():
            ax.errorbar(wl[g] * (1 + 0.012 * (k - 1)), f[g], yerr=e[g], fmt=mk + '-', color=col, ms=5,
                        mec='white', mew=0.6, lw=1.0, elinewidth=1.1, capsize=2, label=lbl)
    ax.set_xscale('log')
    ax.set_xticks([0.3, 0.5, 0.8, 1.5, 2, 3.5]); ax.set_xticklabels(['0.3', '0.5', '0.8', '1.5', '2', '3.5'])
    ax.minorticks_off()
    ax.set_ylim(0.3, 1.9)
    ax.grid(color=GRID, lw=0.6)
    ax.tick_params(labelsize=7)
    if title:
        ax.set_title(title, fontsize=8.5)
    if legend:
        ax.legend(fontsize=6.5, loc='upper left', ncol=2, frameon=True, framealpha=0.85)


def plot_system(tk, rows, det, ctx, out_png):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    bands = ctx['bands']
    fig = plt.figure(figsize=(9.5, 2.0 * len(bands) + 3.2))
    gs = fig.add_gridspec(len(bands) + 1, 2, width_ratios=[1, 3.2],
                          height_ratios=[1] * len(bands) + [1.9], hspace=0.35, wspace=0.08,
                          top=0.955, bottom=0.03, left=0.07, right=0.98)
    byb = {r['band']: r for r in rows}
    for i, b in enumerate(bands):
        ax = fig.add_subplot(gs[i, 1])
        _slit_panel(ax, fig.add_subplot(gs[i, 0]), b, ctx, det, byb[b], tk)
        if i == 0:
            ax.legend(fontsize=6, loc='upper left', frameon=True, framealpha=0.85)
        if i == len(bands) - 1:
            ax.set_xlabel('radius along slit [arcsec]', fontsize=7.5)
    _sed_panel(fig.add_subplot(gs[-1, :]), rows, ctx)
    fig.axes[-1].set_xlabel('wavelength [um]', fontsize=8)
    fig.axes[-1].set_ylabel('flux [mJy] (1$\\sigma$ errors)', fontsize=8)
    fig.suptitle(f'R_eff {tk["reff"]:g} pc, {tk["group"]}, S/N {tk["snr"]}.  Image: orange 4 px aperture, '
                 f'blue 7-8 px sky, purple instP R65 and its 2.6-4 R65 background annulus, red + neighbours.\n'
                 f'Profile: background-subtracted (true pedestal); purple dotted = fitted background; '
                 f'shading = the two sky annuli', x=0.01, ha='left', fontsize=8.5, y=0.995)
    fig.savefig(out_png, dpi=110, facecolor='#fcfcfb', bbox_inches='tight')
    plt.close(fig)


def run_diagnostics(ctx, tasks, t=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _init(ctx)
    sel = diag_tasks(tasks)
    d_out = os.path.join(OUT, 'slit_profiles')
    os.makedirs(d_out, exist_ok=True)
    seds = {}
    for tk in sel:
        rows, det = run_system(tk, ctx, details=True)
        if t is not None:                  # same seed -> must reproduce the main run
            m = ((t['reff_pc'] == tk['reff']) & (t['group'] == tk['group']) & (t['snr'] == tk['snr'])
                 & (t['real'] == tk['k']) & (t['band'] == 'F555W'))
            r555 = next(r for r in rows if r['band'] == 'F555W')
            if m.any() and not np.isclose(float(t['instP'][m][0]), r555['instP'], rtol=1e-6, equal_nan=True):
                print(f'  WARNING {diag_tag(tk)}: instP differs from the main run')
        plot_system(tk, rows, det, ctx, os.path.join(d_out, f'mock_far_{diag_tag(tk)}.png'))
        seds[(tk['reff'], tk['group'], tk['snr'])] = rows
        print(f'  {diag_tag(tk)} done', flush=True)
    # all SEDs on one page
    keys = [(rf, s) for rf in REFFS for s in DIAG_SNRS]
    fig, axes = plt.subplots(len(keys), len(GROUPS), figsize=(3.4 * len(GROUPS), 2.6 * len(keys)),
                             sharex=True, sharey=True)
    for i, (rf, snr) in enumerate(keys):
        for j, g in enumerate(GROUPS):
            ax = axes[i, j]
            _sed_panel(ax, seds[(rf, g, snr)], ctx, title=f'{g}' if i == 0 else None, legend=False)
            if j == 0:
                ax.set_ylabel(f'R_eff {rf:g} pc, S/N {snr}\nflux [mJy]', fontsize=8)
            if i == len(keys) - 1:
                ax.set_xlabel('wavelength [um]', fontsize=8)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=len(h), frameon=False, fontsize=9)
    fig.suptitle('Mock SEDs (one realization each, 1$\\sigma$ errors); truth is flat at 1 mJy', x=0.01,
                 ha='left', y=1.035, fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, 'mock_far_SEDs.png'), dpi=120, facecolor='#fcfcfb', bbox_inches='tight')
    plt.close(fig)
    print(f'  slit profiles -> {d_out}, SEDs -> {os.path.join(OUT, "mock_far_SEDs.png")}')


def report(t):
    print('\nrecovered / true: median [16-84%] over realizations and bands     (median reported error)')
    for inst, bands in (('HST', HST), ('NIRCam', [b for b in set(t['band']) if b not in HST])):
        print(f'\n=== {inst} ===')
        for reff in REFFS:
            for g in GROUPS:
                print(f'  R_eff {reff:g} pc, {g}')
                for m in METHODS:
                    line = f'    {m:12s}'
                    for snr in SNRS:
                        s = (t['reff_pc'] == reff) & (t['group'] == g) & (t['snr'] == snr) & np.isin(t['band'], bands)
                        a, lo, hi = q(t[m][s])
                        e = np.nanmedian(np.asarray(t[f'{m}_err'][s], float)) if np.isfinite(a) else np.nan
                        line += f'  S/N{snr:<3d} {a:5.3f} [{lo:4.2f}-{hi:4.2f}] ({e:4.2f})' if np.isfinite(a) else ''
                    if line.strip() != m:
                        print(line)
    ok = t['instP_size_source'] == 'own'
    print(f'\ninstP size source: own {np.mean(ok):.1%}, '
          f'point_source {np.mean(t["instP_size_source"] == "point_source"):.1%}; errors: {np.sum(t["error"] != "")}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n-real', type=int, default=20, help='realizations per (R_eff, group, S/N) cell')
    ap.add_argument('--workers', type=int, default=40)
    ap.add_argument('--seed', type=int, default=20261015)
    ap.add_argument('--max-tasks', type=int, default=None, help='dry run: only the first N systems')
    ap.add_argument('--plot', action='store_true', help='re-plot from the saved table')
    ap.add_argument('--diagnostics', action='store_true',
                    help='only the slit-profile + SED plots of a subset (realization 0, S/N 20 and 100)')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    out_tab = os.path.join(OUT, 'mock_far.fits')
    t0 = time.time()
    if a.diagnostics:
        ctx = build_context()
        run_diagnostics(ctx, make_tasks(a.n_real, a.seed), Table.read(out_tab) if os.path.exists(out_tab) else None)
        print(f'done in {time.time() - t0:.0f} s')
        return
    if a.plot:
        t = Table.read(out_tab)
    else:
        ctx = build_context()
        tasks = make_tasks(a.n_real, a.seed)
        if a.max_tasks:
            tasks = tasks[::max(len(tasks) // a.max_tasks, 1)][:a.max_tasks]
        print(f'{len(tasks)} mock systems x {len(ctx["bands"])} bands, {a.workers} workers', flush=True)
        plot_examples(ctx, make_tasks(1, a.seed), os.path.join(OUT, 'mock_far_examples.png'))
        rows = []
        if a.workers > 1:
            with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context('fork'),
                                     initializer=_init, initargs=(ctx,)) as ex:
                for i, r in enumerate(ex.map(run_system, tasks, chunksize=2)):
                    rows.extend(r)
                    if (i + 1) % 100 == 0:
                        print(f'  {i + 1}/{len(tasks)}  {time.time() - t0:.0f} s', flush=True)
        else:
            _init(ctx)
            for task in tasks:
                rows.extend(run_system(task))
        t = Table(rows=rows)
        for c in t.colnames:
            if t[c].dtype.kind == 'f':
                t[c].format = '.5g'
        t.meta.update(DIST=DIST, AC_V_1365=DEGER22_TABLE2['NGC1365'][0], NB_RMIN=NB_R[0], NB_RMAX=NB_R[1])
        t.write(out_tab, overwrite=True)
    nircam = sorted({str(b) for b in t['band']} - set(HST), key=lambda b: int(b[1:-1]))
    plot_set(t, HST, 'HST (5 bands)', os.path.join(OUT, 'mock_far_HST.png'))
    plot_set(t, nircam, f'NIRCam ({", ".join(nircam)})', os.path.join(OUT, 'mock_far_NIRCam.png'))
    plot_pull(t, os.path.join(OUT, 'mock_far_pull.png'))
    report(t)
    print(f'\ndone in {time.time() - t0:.0f} s -> {OUT}')


if __name__ == '__main__':
    main()
