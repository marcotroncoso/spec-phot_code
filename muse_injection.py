"""
Injection test of the cluster-only MUSE spectra (run_muse_spectra.py), for any
galaxy: fake point-source clusters are added to the galaxy's copt cube at
positions with no DR5 cluster within 3", then recovered with both methods

  M1 (annulus)  should return the injected spectrum x the PSF fraction in the
                aperture (EE_ap), plus any ISM lines inside the aperture
  M2 (PSF fit)  should return the injected total spectrum

Positions: n_per_env (default 10) in each of three environments by the DAP
Halpha surface brightness (low / mid / high terciles of the field). Injected
spectrum: the line-free continuum of the galaxy's brightest M2 cluster (from
<GAL>_muse_spectra.fits) at two levels (the sample median and 0.3x), each
with lines (EW(Halpha) = 50 A at HII-region ratios) and without. The M1/M2
settings (annulus gap and width, ridge, stamp size) are the [muse_spectra]
ones, the same as for the real spectra.

The scatter of recovered / injected continuum, per method, continuum bin and
environment, is the systematic error of the SED plots (muse_sed.py).

Outputs in <output_dir>/<GAL>/muse_spectra/injection/:
  <GAL>_injection_results.fits, <GAL>_injection_summary.json,
  <GAL>_injection.png, cutouts/

run_full_pipeline.py runs it per galaxy ([run] run_sed with [sed] injection);
validation/muse_spec_injection_test.py is the command-line wrapper.
"""

import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import json
import logging
import multiprocessing as mp
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from regions import CirclePixelRegion, PixCoord

import muse_spec_aux_func as msa
import run_muse_spectra as rms

logger = logging.getLogger('muse_injection')

# HII-region-like line ratios relative to Halpha
LINE_RATIOS = {'Ha': 1.0, 'Hb': 1 / 2.86, 'NII6583': 0.35, 'NII6548': 0.35 / 3.05,
               'SII6716': 0.15, 'SII6731': 0.11, 'OIII5007': 0.3, 'OIII4959': 0.1}
EW_HA = 50.0          # A
SIG_KMS = 40.0


def pick_positions(inp, n_per_env=10, min_sep_arcsec=3.0, seed=1):
    maps = os.path.join(os.path.dirname(inp['cube_path']), f'{inp["token"]}_DR2.2v2',
                        f'{inp["token"]}_MAPS.fits')
    with fits.open(maps) as h:
        ha = h['HA6562_FLUX'].data.astype(float)
    ny, nx = ha.shape
    cx, cy = inp['cel'].world_to_pixel(SkyCoord(inp['cat']['PHANGS_RA'],
                                                inp['cat']['PHANGS_DEC'], unit='deg'))
    yy, xx = np.mgrid[0:ny:4, 0:nx:4]
    xx, yy = xx.ravel(), yy.ravel()
    edge = 45
    ok = (xx > edge) & (xx < nx - edge) & (yy > edge) & (yy < ny - edge) & \
        np.isfinite(ha[yy, xx]) & np.isfinite(inp['vstars'][yy, xx])
    xx, yy = xx[ok], yy[ok]
    from scipy.spatial import cKDTree
    d, _ = cKDTree(np.c_[cx, cy]).query(np.c_[xx, yy])
    far = d > min_sep_arcsec / msa.MUSE_PIX_ARCSEC
    xx, yy = xx[far], yy[far]
    sb = ha[yy, xx]
    q = np.nanpercentile(sb, [33, 67])
    env = np.where(sb < q[0], 'low', np.where(sb < q[1], 'mid', 'high'))
    rng = np.random.default_rng(seed)
    pos = []
    for e in ('low', 'mid', 'high'):
        idx = rng.choice(np.where(env == e)[0], n_per_env, replace=False)
        pos += [(int(xx[i]), int(yy[i]), e, float(sb[i])) for i in idx]
    return pos


def template_spectrum(inp, spec_fits):
    with fits.open(spec_fits) as h:
        t = Table(h['SOURCES'].data)
        c2 = h['M2_C2'].data.astype(float)
    lo, hi = msa.CONT_BINS[1]
    cont = np.array(t[f'cont_C2_{int(lo)}_{int(hi)}'])
    k = int(np.nanargmax(cont))
    z = float(t['z_local'][k])
    good = msa.continuum_mask(inp['wave'], z)
    tmpl = msa.interpolate_over_lines(inp['wave'], c2[k], good)
    tmpl /= np.nanmedian(tmpl[(inp['wave'] > lo) & (inp['wave'] < hi)])
    return tmpl, float(np.nanmedian(cont)), int(t['ID_PHANGS_CLUSTER'][k])


def line_spectrum(wave, z, ha_flux):
    out = np.zeros_like(wave)
    for n, r in LINE_RATIOS.items():
        lam = msa.EMISSION_LINES[n] * (1 + z)
        s = np.hypot(lam * SIG_KMS / msa.C_KMS, msa.muse_lsf_fwhm(lam) / 2.3548)
        out += r * ha_flux / (np.sqrt(2 * np.pi) * s) * np.exp(-0.5 * ((wave - lam) / s)**2)
    return out


_G = {}


def run_case(args):
    k, scale, with_lines = args
    inp, psf, pos, opt = _G['inp'], _G['psf'], _G['pos'], _G['opt']
    x, y, env, sb = pos[k]
    c = msa.load_cutout(_G['paths'][k], inp['wsel'])
    wave = inp['wave']
    half = _G['half']
    shape = c['data'].shape[1:]
    xy = (x - c['x0'], y - c['y0'])
    # local velocity as in the pipeline
    z = (inp['vsys'] + np.nan_to_num(inp['vstars'][y, x])) / msa.C_KMS

    cont = scale * _G['cont_level'] * _G['tmpl']
    ha_cont = np.nanmedian(cont[np.abs(wave - 6563 * (1 + z)) < 40])
    lines = line_spectrum(wave, z, EW_HA * ha_cont) if with_lines else 0 * wave
    inj = cont + lines

    data = c['data'].astype(float).copy()
    for b0 in np.arange(wave[0], wave[-1] + 100, 100.0):
        sl = (wave >= b0) & (wave < b0 + 100)
        if not sl.any():
            continue
        P = msa.moffat_image(shape, xy[0], xy[1], float(psf.fwhm(wave[sl].mean())), psf.beta)
        data[sl] += inj[sl, None, None] * P[None]

    ap = CirclePixelRegion(PixCoord(*xy), opt.r_ap).to_mask(mode='exact').to_image(shape)
    neigh = [(a - c['x0'], b - c['y0']) for a, b in _G['neigh_xy']
             if abs(a - x) < half + 5 and abs(b - y) < half + 5]
    fw = float(psf.fwhm(np.median(wave)))
    excl = msa.neighbour_exclusion(shape, neigh, fw)
    ring = msa.annulus_mask(ap, opt.gap_fwhm * fw, opt.width_fwhm * fw, exclude=excl)
    m1 = msa.method_annulus(data, c['var'], ap, ring, wave, z)
    m2 = msa.method_psf(data, c['var'], wave, ap, xy, neigh, psf, z, fit_half=opt.fit_half,
                        ridge_frac=opt.ridge)
    # the ISM lines already in the aperture before injection
    m1_0 = msa.method_annulus(c['data'], c['var'], ap, ring, wave, z)

    out = {'k': k, 'env': env, 'sb': sb, 'scale': scale, 'lines': with_lines,
           'ee': float(np.nanmedian(m2['EE_AP']))}
    for lo, hi in msa.CONT_BINS:
        s = (wave >= lo) & (wave < hi) & msa.continuum_mask(wave, z)
        out[f'm1_{int(lo)}'] = np.nanmedian(m1['C1'][s]) / np.nanmedian((inj * m2['EE_AP'])[s])
        out[f'm2_{int(lo)}'] = np.nanmedian(m2['C2'][s]) / np.nanmedian(inj[s])
        out[f'm2e_{int(lo)}'] = np.nanmedian(m2['C2_err'][s]) / np.sqrt(s.sum()) \
            / np.nanmedian(inj[s])
        out[f'm1e_{int(lo)}'] = np.nanmedian(m1['C1_err'][s]) / np.sqrt(s.sum()) \
            / np.nanmedian((inj * m2['EE_AP'])[s])
    L1 = msa.fit_lines(wave, m1['C1'], m1['C1_err'], z)
    L2 = msa.fit_lines(wave, m2['C2'], m2['C2_err'], z)
    L0 = msa.fit_lines(wave, m1_0['C1'], m1_0['C1_err'], z)   # ISM already in the aperture
    ee = out['ee']
    for n in ('Ha', 'Hb', 'NII6583', 'SII6716', 'OIII5007'):
        true = LINE_RATIOS[n] * EW_HA * ha_cont if with_lines else 0.0
        out[f'{n}_true'] = true
        out[f'{n}_m1'], out[f'{n}_m1_err'] = L1[n]
        out[f'{n}_m2'], out[f'{n}_m2_err'] = L2[n]
        out[f'{n}_ism_ap'] = L0[n][0]
        out[f'{n}_true_ap'] = true * ee
    return out


def injection_paths(cfg, token):
    gal = token.split('-')[0]
    d = os.path.join(cfg['output_dir'], gal, 'muse_spectra', 'injection')
    return d, os.path.join(d, f'{gal}_injection_results.fits')


def injection_up_to_date(cfg, token):
    """True when the injection results exist and are newer than the galaxy's
    spectra (the template and PSF come from <GAL>_muse_spectra.fits)."""
    _, res = injection_paths(cfg, token)
    spec = rms.spectra_output_path(cfg, token)
    return (os.path.exists(res) and os.path.exists(spec)
            and os.path.getmtime(res) >= os.path.getmtime(spec))


def run_injection(token, cfg, settings, n_per_env=10, n_workers=4, r_ap=0.0, plot_results=True):
    """
    Injection test of one galaxy token (e.g. 'NGC3351-1.05asec'). cfg:
    run_full_pipeline.load_config(...); settings: run_muse_spectra.
    spectra_settings(...) ([muse_spectra]; gap_fwhm, width_fwhm, ridge,
    fit_half are used). Needs <GAL>_muse_spectra.fits. r_ap: aperture radius
    [px]; 0 = equivalent radius of the ISF polygons. Returns dict(path, n_cases).
    """
    opt = SimpleNamespace(gap_fwhm=settings['gap_fwhm'], width_fwhm=settings['width_fwhm'],
                          ridge=settings['ridge'], fit_half=settings['fit_half'], r_ap=r_ap)
    inp = rms.load_inputs(token, cfg, 'MUSE', tuple(settings['wave_range']))
    gal = inp['gal']
    out_dir, res_path = injection_paths(cfg, token)
    os.makedirs(out_dir, exist_ok=True)
    spec_fits = rms.spectra_output_path(cfg, token)

    with fits.open(spec_fits) as h:
        hdr = h[0].header
        t = Table(h['SOURCES'].data)
    psf = msa.PSFModel(hdr['PSF_FWHM'], hdr['PSF_BETA'], hdr['PSF_SLOP'], hdr['PSF_WAVE'])
    if opt.r_ap <= 0:
        opt.r_ap = float(np.sqrt(np.median(t['N_ap']) / np.pi))
    logger.info(f'{gal} injection: aperture radius {opt.r_ap:.2f} px, PSF FWHM {psf.fwhm_ref * 0.2:.3f}"')

    pos = pick_positions(inp, n_per_env)
    half = 30
    cut_dir = os.path.join(out_dir, 'cutouts')
    os.makedirs(cut_dir, exist_ok=True)
    paths = {k: os.path.join(cut_dir, f'inj_{k}_{x}_{y}_h{half}.npz')
             for k, (x, y, _, _) in enumerate(pos)}
    missing = {k: (x, y) for k, (x, y, _, _) in enumerate(pos) if not os.path.exists(paths[k])}
    if missing:
        cuts, _ = msa.extract_cutouts(inp['cube_path'], missing, half)
        for k, c in cuts.items():
            msa.save_cutout(paths[k], c)

    tmpl, cont_level, tid = template_spectrum(inp, spec_fits)
    logger.info(f'  template from ID {tid}; median continuum level {cont_level:.1f}')
    cx, cy = inp['cel'].world_to_pixel(SkyCoord(inp['cat']['PHANGS_RA'],
                                                inp['cat']['PHANGS_DEC'], unit='deg'))
    _G.update({'inp': inp, 'psf': psf, 'pos': pos, 'opt': opt, 'paths': paths,
               'tmpl': tmpl, 'cont_level': cont_level, 'half': half,
               'neigh_xy': list(zip(cx, cy))})

    cases = [(k, s, l) for k in range(len(pos)) for s in (1.0, 0.3) for l in (True, False)]
    if n_workers > 1:
        with mp.get_context('fork').Pool(n_workers) as pool:
            rows = pool.map(run_case, cases)
    else:
        rows = [run_case(c) for c in cases]
    res = Table(rows=[[r[c] for c in rows[0]] for r in rows], names=list(rows[0]))
    res.meta.update(GALAXY=gal, TOKEN=token, NPERENV=int(n_per_env), RAP=float(opt.r_ap))
    res.write(res_path, overwrite=True)

    summary = {}
    for scale in (1.0, 0.3):
        for env in ('low', 'mid', 'high'):
            m = (res['scale'] == scale) & (res['env'] == env)
            key = f'scale{scale}_{env}'
            summary[key] = {}
            for meth in ('m1', 'm2'):
                for lo, _ in msa.CONT_BINS:
                    v = np.array(res[f'{meth}_{int(lo)}'][m])
                    summary[key][f'{meth}_cont_{int(lo)}'] = [float(np.nanmedian(v)),
                                                             float(1.4826 * np.nanmedian(np.abs(v - np.nanmedian(v))))]
                ml = m & res['lines']
                for n in ('Ha', 'Hb', 'NII6583', 'OIII5007'):
                    true = np.array(res[f'{n}_true' if meth == 'm2' else f'{n}_true_ap'][ml])
                    v = np.array(res[f'{n}_{meth}'][ml]) / true
                    summary[key][f'{meth}_{n}'] = [float(np.nanmedian(v)),
                                                   float(1.4826 * np.nanmedian(np.abs(v - np.nanmedian(v))))]
                mc = m & ~res['lines']
                ref = np.array(res['Ha_true'][ml]) if ml.any() else np.nan
                v = np.array(res[f'Ha_{meth}'][mc]) / np.nanmedian(ref)
                summary[key][f'{meth}_falseHa'] = [float(np.nanmedian(v)),
                                                   float(1.4826 * np.nanmedian(np.abs(v - np.nanmedian(v))))]
    with open(os.path.join(out_dir, f'{gal}_injection_summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    for k, v in summary.items():
        logger.info(f'  {k}: ' + ', '.join(f'{a}={b[0]:.3f}±{b[1]:.3f}' for a, b in v.items()
                                           if a.endswith(('_5500', 'Ha', 'Hb', 'falseHa'))))
    if plot_results:
        plot(res, out_dir, gal)
    return {'path': res_path, 'n_cases': len(rows)}


def plot(res, out_dir, gal):
    envs = ('low', 'mid', 'high')
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
    cols = {'m1': rms.COL['C1'], 'm2': rms.COL['C2']}
    for meth, dx in (('m1', -0.12), ('m2', 0.12)):
        for scale, mk in ((1.0, 'o'), (0.3, 's')):
            for j, (lo, hi) in enumerate(msa.CONT_BINS):
                m = res['scale'] == scale
                v = np.array(res[f'{meth}_{int(lo)}'][m])
                med = np.nanmedian(v)
                # robust error of the median (a few high-SB positions are outliers)
                mad = 1.4826 * np.nanmedian(np.abs(v - med))
                err = 1.253 * mad / np.sqrt(np.isfinite(v).sum())
                ax[0].errorbar(j + dx + (0.05 if scale < 1 else 0), med, err, fmt=mk,
                               color=cols[meth], ms=6, lw=1.5,
                               label=f'{meth.upper()} ×{scale}' if j == 0 else None,
                               mfc=cols[meth] if scale == 1 else 'none')
    ax[0].axhline(1, color='#c3c2b7', lw=1)
    ax[0].set_xticks(range(len(msa.CONT_BINS)),
                     [f'{int(a)}–{int(b)}' for a, b in msa.CONT_BINS], fontsize=8)
    ax[0].set(ylabel='recovered / injected continuum (median ± error)',
              xlabel='wavelength bin [Å]',
              title='Continuum (M1 vs injected × EE$_{ap}$, M2 vs injected)')
    ax[0].legend(frameon=False, fontsize=8, ncol=2)

    for meth, dx in (('m1', -0.12), ('m2', 0.12)):
        for j, env in enumerate(envs):
            m = (res['env'] == env) & res['lines'] & (res['scale'] == 1.0)
            true = np.array(res['Ha_true' if meth == 'm2' else 'Ha_true_ap'][m])
            v = np.array(res[f'Ha_{meth}'][m]) / true
            e = np.array(res[f'Ha_{meth}_err'][m]) / true
            ax[1].errorbar(j + dx + np.linspace(-0.05, 0.05, v.size), v, e, fmt='o', ms=4,
                           color=cols[meth], lw=1, label=meth.upper() if j == 0 else None)
    ax[1].axhline(1, color='#c3c2b7', lw=1)
    ax[1].set_xticks(range(3), [f'{e} Hα SB' for e in envs])
    ax[1].set(ylabel='recovered / injected Hα', title='Hα of an EW=50 Å cluster', yscale='log')
    ax[1].legend(frameon=False, fontsize=8)

    for meth, dx in (('m1', -0.12), ('m2', 0.12)):
        for j, env in enumerate(envs):
            m = (res['env'] == env) & ~res['lines'] & (res['scale'] == 1.0)
            ref = np.nanmedian(res['Ha_true'][(res['env'] == env) & res['lines'] & (res['scale'] == 1.0)])
            v = np.array(res[f'Ha_{meth}'][m]) / ref
            e = np.array(res[f'Ha_{meth}_err'][m]) / ref
            ax[2].errorbar(j + dx + np.linspace(-0.05, 0.05, v.size), v, e, fmt='o', ms=4,
                           color=cols[meth], lw=1, label=meth.upper() if j == 0 else None)
    ax[2].axhline(0, color='#c3c2b7', lw=1)
    ax[2].set_xticks(range(3), [f'{e} Hα SB' for e in envs])
    ax[2].set(ylabel='false Hα / Hα(EW=50 Å)', title='ISM leaking into a line-free cluster',
              yscale='symlog')
    ax[2].legend(frameon=False, fontsize=8)
    fig.savefig(os.path.join(out_dir, f'{gal}_injection.png'), dpi=120)
    plt.close(fig)
