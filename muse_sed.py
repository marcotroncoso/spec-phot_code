"""
SEDs of the isolated clusters of any galaxy: HST + JWST photometry (native =
1D Gaussian slits, instP = 1D slit-profile fit, 2D = pixel model fit, PHANGS
DR5 catalogue) together with the two background-subtracted MUSE spectra of
run_muse_spectra.py (M1 annulus / EE_ap, M2 PSF fit, both total fluxes), all
in mJy and corrected for Milky Way extinction (the photometry already is; the
spectra get F99 x 0.86 with the SFD E(B-V) at the source, as Schlafly &
Finkbeiner 2011, which reproduces the pipeline's per-band R values to <4%).

Error bars on the MUSE continuum points and synthetic band fluxes are
statistical (+) systematic. The systematic is the robust scatter of the
recovered / injected continuum in the galaxy's own injection test
(muse_injection.py), for the same method, continuum bin and Halpha
environment (low / mid / high tercile) as the source. Without injection
results the errors are statistical only, and the figure says so. The thin
bands on the spectra are statistical only.

Inputs, in <output_dir>/<GAL>/:
  <GAL>_muse_spectra.fits, <GAL>_photometry_v2.fits (native, instP, 2D and
  catalogue fluxes), muse_spectra/injection/<GAL>_injection_results.fits,
  and the DAP MAPS (Halpha) of the token.
Outputs in <output_dir>/<GAL>/muse_spectra/sed/:
  <row>_<class>_sed.png (row = photometry-table row, as the slit profiles;
  the PHANGS ID is in the title) and <GAL>_sed_ratios.csv.

run_full_pipeline.py runs it per galaxy ([run] run_sed); the command-line
wrapper is validation/muse_sed_example.py.
"""
import glob
import logging
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

import muse_spec_aux_func as msa
import muse_injection as mi
import run_muse_spectra as rms
from run_full_pipeline import DIC_PASSBANDS, FILTER_WAVELENGTHS

logger = logging.getLogger('muse_sed')

COL = {'native': '#6250d6', 'instP': '#eb6834', 'f2d': '#e87ba4', 'f2dG': '#b08a1e', 'cat': '#3d3d3a',
       'M1': '#2a78d6', 'M2': '#1baf7a'}
# PHANGS DR5 catalogue: total (Deger+22 aperture-corrected), MW-dereddened fluxes
# under generic band names
CATALOG_BANDS = {'F275W': 'NUV', 'F336W': 'U', 'F438W': 'B', 'F555W': 'V', 'F814W': 'I'}
# continuum bin of the injection test that overlaps each HST filter MUSE covers
# (F814W is outside the 4850-7000 A range used)
FILTER_BINS = {'F555W': msa.CONT_BINS[0]}


def flam_to_mJy(wave, flam):
    """1e-20 erg/s/cm2/A -> mJy."""
    return flam * 1e-20 * wave**2 / (msa.C_KMS * 1e13) / 1e-26


def rebin(wave, f, e, width=10.0):
    """Mean in bins of ``width`` A; error = sqrt(sum e^2)/n."""
    edges = np.arange(wave[0], wave[-1] + width, width)
    idx = np.digitize(wave, edges) - 1
    wb, fb, eb = [], [], []
    for k in range(len(edges) - 1):
        s = (idx == k) & np.isfinite(f) & np.isfinite(e)
        if s.sum() < 3:
            continue
        wb.append(wave[s].mean())
        fb.append(f[s].mean())
        eb.append(np.sqrt(np.sum(e[s]**2)) / s.sum())
    return np.array(wb), np.array(fb), np.array(eb)


def mad(v):
    v = v[np.isfinite(v)]
    return 1.4826 * np.median(np.abs(v - np.median(v))) if v.size else np.nan


def ref(row, key):
    """Photometric reference flux, NaN when missing or non-positive."""
    v = float(row[key]) if key in row.colnames else np.nan
    return v if np.isfinite(v) and v > 0 else np.nan


def sed_dir(cfg, token):
    gal = token.split('-')[0]
    return os.path.join(cfg['output_dir'], gal, 'muse_spectra', 'sed')


def sed_up_to_date(cfg, token):
    """True when the galaxy has SED plots newer than everything they draw
    (spectra, photometry table, injection results)."""
    gal = token.split('-')[0]
    outd = os.path.join(cfg['output_dir'], gal)
    pngs = glob.glob(os.path.join(sed_dir(cfg, token), '*_sed.png'))
    csv = os.path.join(sed_dir(cfg, token), f'{gal}_sed_ratios.csv')
    if not pngs or not os.path.exists(csv):
        return False
    # the photometry's own caches, not <GAL>_photometry_v2.fits (rewritten by every run)
    refs = [rms.spectra_output_path(cfg, token), mi.injection_paths(cfg, token)[1]] + [
        os.path.join(outd, f'{gal}_photometry_{k}.fits') for k in ('native', 'instrument', '2d')]
    ref_time = max((os.path.getmtime(r) for r in refs if os.path.exists(r)), default=0.0)
    return min(os.path.getmtime(p) for p in pngs + [csv]) >= ref_time


def load_injection(cfg, token):
    """(injection results, DAP Halpha map, its WCS), or None without results."""
    path = mi.injection_paths(cfg, token)[1]
    if not os.path.exists(path):
        return None
    inj = Table.read(path)
    maps = os.path.join(cfg['muse_dap_root'], token, f'{token}_DR2.2v2', f'{token}_MAPS.fits')
    with fits.open(maps) as h:
        ha = h['HA6562_FLUX'].data.astype(float)
        wcs = WCS(h['HA6562_FLUX'].header).celestial
    return inj, ha, wcs


def injection_systematics(ra, dec, injection):
    """
    Fractional systematic per (method, continuum bin) from the injection test,
    for the Halpha-environment tercile of the source (same DAP HA6562_FLUX
    map and tercile edges as the test). Zero without injection results.
    """
    if injection is None:
        return {(m, lo): 0.0 for m in ('m1', 'm2') for lo, _ in msa.CONT_BINS}, 'n/a', np.nan
    inj, ha, wcs = injection
    x, y = wcs.world_to_pixel(SkyCoord(ra, dec, unit='deg'))
    sb = float(ha[int(round(float(y))), int(round(float(x)))])
    edges = [inj['sb'][inj['env'] == e].max() for e in ('low', 'mid')]
    env = 'low' if sb <= edges[0] else 'mid' if sb <= edges[1] else 'high'
    s = (inj['scale'] == 1.0) & (inj['env'] == env)
    sys_frac = {(m, lo): mad(np.asarray(inj[f'{m}_{int(lo)}'][s], float))
                for m in ('m1', 'm2') for lo, _ in msa.CONT_BINS}
    return sys_frac, env, sb


def load_context(token, cfg):
    from dustmaps.sfd import SFDQuery
    gal = token.split('-')[0]
    outd = os.path.join(cfg['output_dir'], gal)
    h = fits.open(rms.spectra_output_path(cfg, token))
    v2 = Table.read(os.path.join(outd, f'{gal}_photometry_v2.fits'))
    # 2D fluxes in v2 (run_2d), per source profile: the free-centre mode if there,
    # else the fixed one. [profile] model (EFF/King) and Gaussian (G modes) apart.
    def pick(modes):
        return next((m for m in modes if any(c.startswith(f'flux_f2d{m}_') for c in v2.colnames)), None)
    mode2d, mode2dG = pick(('free', 'fixed')), pick(('Gfree', 'Gfixed'))
    return {'gal': gal, 'token': token, 'h': h, 'src': Table(h['SOURCES'].data),
            'wave': h['WAVE'].data.astype(float), 'v2': v2, 'mode2d': mode2d, 'mode2dG': mode2dG,
            'sfd': SFDQuery(), 'inj': load_injection(cfg, token), 'out_dir': sed_dir(cfg, token)}


def plot_sed(ctx, sid):
    h, src, wave, gal = ctx['h'], ctx['src'], ctx['wave'], ctx['gal']
    k = int(np.where(src['ID_PHANGS_CLUSTER'] == sid)[0][0])
    row = src[k]
    spec = {'M2': (h['M2_C2'].data[k], h['M2_C2_ERR'].data[k]),
            'M1': (h['M1_C1'].data[k] / row['EE_AP'], h['M1_C1_ERR'].data[k] / row['EE_AP'])}

    from dust_extinction.parameter_averages import F99
    ebv = float(ctx['sfd'](SkyCoord(row['PHANGS_RA'], row['PHANGS_DEC'], unit='deg')))
    a_lam = 0.86 * 3.1 * F99(Rv=3.1)(wave * u.AA) * ebv
    mw = 10**(0.4 * a_lam)
    sys_frac, env, sb = injection_systematics(row['PHANGS_RA'], row['PHANGS_DEC'], ctx['inj'])
    has_sys = ctx['inj'] is not None
    err_lab = 'stat ⊕ injection syst.' if has_sys else 'statistical only (no injection test)'

    # spectra in mJy and continuum points per injection bin (continuum channels only)
    good = msa.continuum_mask(wave, float(row['z_local']))
    mj, cont = {}, {}
    for name, (f, e) in spec.items():
        fm, em = flam_to_mJy(wave, f * mw), flam_to_mJy(wave, e * mw)
        mj[name] = (fm, em)
        pts = []
        for lo, hi in msa.CONT_BINS:
            s = good & (wave >= lo) & (wave < hi) & np.isfinite(fm) & np.isfinite(em)
            if not s.any():
                continue
            fb = fm[s].mean()
            e_stat = np.sqrt(np.sum(em[s]**2)) / s.sum()
            e_sys = sys_frac[(name.lower(), lo)] * fb
            pts.append((0.5 * (lo + hi), fb, e_stat, np.hypot(e_stat, np.nan_to_num(e_sys)), lo, hi))
        cont[name] = np.array(pts).reshape(-1, 6)

    v2 = ctx['v2']
    p = int(np.where(v2['ID_PHANGS_CLUSTER'] == sid)[0][0])
    prow = v2[p]
    cls = str(row['PHANGS_CC_CLASS_ML']).strip()
    mode2d, mode2dG = ctx['mode2d'], ctx['mode2dG']

    phot = {}
    for b, (wl, _) in FILTER_WAVELENGTHS.items():
        keys = [('native', f'flux_native_{b}'), ('instP', f'flux_instP_{b}')]
        if mode2d:
            keys.append(('f2d', f'flux_f2d{mode2d}_{b}'))
        if mode2dG:
            keys.append(('f2dG', f'flux_f2d{mode2dG}_{b}'))
        for meth, key in keys:
            if key in prow.colnames and np.isfinite(prow[key]) and prow[key] > 0:
                e = float(prow[key + '_err']) if key + '_err' in prow.colnames else np.nan
                phot.setdefault(meth, []).append((wl, float(prow[key]), abs(e), b))
        key = f'PHANGS_{CATALOG_BANDS.get(b, "")}_mJy'
        if b in CATALOG_BANDS and key in prow.colnames and np.isfinite(prow[key]) and prow[key] > 0:
            # the catalogue error is sometimes negative (its own propagation artefact): no error then
            e = float(prow[key + '_ERR'])
            phot.setdefault('cat', []).append((wl, float(prow[key]), e if e > 0 else np.nan, b))

    # synthetic band fluxes from the spectra (stat (+) injection systematic)
    synth = {}
    for b, lo in FILTER_BINS.items():
        mwb = 10**(0.4 * FILTER_WAVELENGTHS[b][1] * ebv)
        for name, key in (('M2', 'C2'), ('M1', 'C1')):
            corr = 1.0 if name == 'M2' else 1.0 / row['EE_AP']
            y = row[f'synth_{b}_{key}_mJy'] * corr * mwb
            ye = row[f'synth_{b}_{key}_mJy_err'] * corr * mwb
            synth[(name, b)] = (y, ye, np.hypot(ye, np.nan_to_num(sys_frac[(name.lower(), lo[0])] * y)))

    fig, axes = plt.subplots(3, 1, figsize=(11, 13), constrained_layout=True,
                             gridspec_kw={'height_ratios': [1.15, 1, 1]})
    labels = {'native': 'native (1D Gaussian slits)', 'instP': 'instP (1D slit-profile fit)',
              'f2d': f'2D model fit, {mode2d} centre',
              'f2dG': f'2D Gaussian fit, {str(mode2dG)[1:]} centre', 'cat': 'PHANGS DR5 catalogue'}
    mk = {'native': ('o', 8), 'instP': ('s', 6), 'f2d': ('^', 6), 'f2dG': ('v', 6), 'cat': ('P', 7)}
    spec_lab = {'M1': 'MUSE M1: annulus continuum subtracted, / EE$_{ap}$',
                'M2': 'MUSE M2: PSF-fit cluster spectrum (total)'}

    def draw_phot(ax, xerr):
        for meth, pts in phot.items():
            wl, f, e, bb = map(np.array, zip(*pts))
            xe = np.array([[w - DIC_PASSBANDS[b][0] for w, b in zip(wl, bb)],
                           [DIC_PASSBANDS[b][1] - w for w, b in zip(wl, bb)]])
            ax.errorbar(wl, f.astype(float), e.astype(float),
                        xerr=xe if xerr else None, fmt=mk[meth][0], ms=mk[meth][1],
                        color=COL[meth], lw=1.5,
                        capsize=3, label=labels[meth], zorder=5, mec='white', mew=0.8)

    def draw_cont(ax, name, xerr):
        c = cont[name]
        if not len(c):
            return
        ax.errorbar(c[:, 0], c[:, 1], c[:, 3],
                    xerr=[c[:, 0] - c[:, 4], c[:, 5] - c[:, 0]] if xerr else None,
                    fmt='s', ms=7, color=COL[name], mfc=COL[name], mec='white', mew=0.8,
                    lw=1.8, capsize=4, zorder=6, label=f'{name} continuum per bin ({err_lab})')

    # top: full SED, log-log
    ax = axes[0]
    draw_phot(ax, True)
    for name in ('M1', 'M2'):
        wb, fb, eb = rebin(wave, *mj[name], 10.0)
        ax.plot(wb, fb, color=COL[name], lw=0.8, alpha=0.6, label=spec_lab[name], zorder=3)
        draw_cont(ax, name, True)
    yv = [float(f) for pts in phot.values() for w, f, e, b in pts]
    yv += [v for c in cont.values() for v in (c[:, 1] if len(c) else []) if v > 0]
    if yv:
        ax.set_ylim(0.3 * min(yv), 3 * max(yv))
    ax.set(xscale='log', yscale='log', xlim=(2000, 4.2e4), xlabel='wavelength [Å]',
           ylabel='F$_ν$ [mJy]  (MW-dereddened)',
           title=f'{gal} #{p} = ID {sid} ({cls}): '
                 f'HST + JWST photometry and MUSE background-subtracted spectra')
    ax.axvspan(wave[0], wave[-1], color='#e1e0d9', alpha=0.5, lw=0, zorder=0)
    ax.legend(frameon=False, fontsize=8, loc='lower left', ncol=2)

    # middle / bottom: one MUSE method per panel, linear, 5 A bins with 1 sigma band
    yphot = [float(f) for pts in phot.values() for w, f, e, b in pts if 4000 < w < wave[-1] + 500]
    for ax, name in zip(axes[1:], ('M2', 'M1')):
        wb, fb, eb = rebin(wave, *mj[name], 5.0)
        ylo, yhi = np.nanpercentile(fb, [1, 97]) if fb.size else (0.0, 1.0)
        top = [yhi] + ([np.nanmax(cont[name][:, 1] + cont[name][:, 3])] if len(cont[name]) else [])
        ymax = 1.25 * max(yphot + top)
        ymin = max(min(0.0, 1.25 * ylo), -0.3 * ymax)
        ax.plot(wb, fb, color=COL[name], lw=0.9, label=spec_lab[name] + ', 5 Å bins', zorder=3)
        ax.fill_between(wb, fb - eb, fb + eb, color=COL[name], alpha=0.35, lw=0, zorder=2,
                        label='1σ statistical')
        draw_phot(ax, False)
        draw_cont(ax, name, True)
        for b in FILTER_BINS:
            y, ye_stat, ye_tot = synth[(name, b)]
            if np.isfinite(y):
                ax.errorbar(FILTER_WAVELENGTHS[b][0], y, ye_tot, fmt='D', ms=8, mfc='white',
                            mec=COL[name], color=COL[name], mew=1.8, capsize=4, zorder=7,
                            label=f'synthetic HST band flux from {name}' if b == 'F555W' else None)
        if np.isfinite(ymax) and np.isfinite(ymin) and ymax > ymin:
            ax.set_ylim(ymin, ymax)
        ax.set(xlim=(wave[0] - 150, wave[-1] + 150), xlabel='wavelength [Å]',
               ylabel='F$_ν$ [mJy]  (MW-dereddened)')
        ax.legend(frameon=False, fontsize=8, loc='upper left', ncol=3)

    m2_555 = synth[('M2', 'F555W')][0]
    m1_555 = synth[('M1', 'F555W')][0]
    f2d_key = f'flux_f2d{mode2d}_F555W' if mode2d else ''
    f2dG_key = f'flux_f2d{mode2dG}_F555W' if mode2dG else ''
    ratio_txt = (f'F555W: M2 / instP = {m2_555 / ref(prow, "flux_instP_F555W"):.2f},  '
                 + (f'M2 / 2D = {m2_555 / ref(prow, f2d_key):.2f},  ' if mode2d else '')
                 + (f'M2 / 2D Gauss = {m2_555 / ref(prow, f2dG_key):.2f},  ' if mode2dG else '')
                 + f'(M1/EE) / native = {m1_555 / ref(prow, "flux_native_F555W"):.2f},  '
                 f'M2 / catalogue = {m2_555 / ref(prow, "PHANGS_V_mJy"):.2f}  '
                 f'(MUSE covers {row["synth_F555W_cover"]:.0%} of F555W)\n'
                 f'EE$_{{ap}}$ = {row["EE_AP"]:.2f},  N$_{{ap}}$ = {row["N_ap"]:.0f} spaxels,  '
                 f'E(B−V)$_{{MW}}$ = {ebv:.3f},  '
                 + (f'Hα environment: {env} (injection syst. from that tercile)' if has_sys
                    else 'no injection test: statistical errors only'))
    axes[2].text(0.99, 0.03, ratio_txt, transform=axes[2].transAxes, ha='right', va='bottom',
                 fontsize=8, color='#3d3d3a')

    path = os.path.join(ctx['out_dir'], f'{p}_{cls}_sed.png')
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return {'row': p, 'ID': int(sid), 'class': cls, 'env': env, 'EE_AP': float(row['EE_AP']),
            'N_ap': float(row['N_ap']),
            'M2_instP_F555W': m2_555 / ref(prow, 'flux_instP_F555W'),
            'M2_2D_F555W': m2_555 / ref(prow, f2d_key) if mode2d else np.nan,
            'M2_2DG_F555W': m2_555 / ref(prow, f2dG_key) if mode2dG else np.nan,
            'M1EE_native_F555W': m1_555 / ref(prow, 'flux_native_F555W'),
            'M2_cat_F555W': m2_555 / ref(prow, 'PHANGS_V_mJy'),
            'native_cat_F555W': ref(prow, 'flux_native_F555W') / ref(prow, 'PHANGS_V_mJy'),
            'path': path}


def make_galaxy_seds(token, cfg, ids=None):
    """
    SED plot of every cluster of one galaxy token with both a MUSE spectrum
    and photometry (or only ids), plus <GAL>_sed_ratios.csv. Returns
    dict(n, dir, systematics) — systematics False when there are no injection
    results (statistical errors only).
    """
    ctx = load_context(token, cfg)
    os.makedirs(ctx['out_dir'], exist_ok=True)
    phot_ids = {int(i) for i in ctx['v2']['ID_PHANGS_CLUSTER']}
    spec_ids = [int(i) for i in ctx['src']['ID_PHANGS_CLUSTER']]
    todo = [i for i in spec_ids if i in phot_ids and (ids is None or i in set(ids))]
    rows = []
    for sid in todo:
        try:
            rows.append(plot_sed(ctx, sid))
        except Exception as e:                       # one bad source must not stop the rest
            logger.warning(f'  SED of ID {sid} failed: {e}')
    ctx['h'].close()
    if rows and ids is None:
        t = Table(rows=[[r[c] for c in rows[0]] for r in rows], names=list(rows[0]))
        t.write(os.path.join(ctx['out_dir'], f'{ctx["gal"]}_sed_ratios.csv'), overwrite=True)
    logger.info(f'  {len(rows)} SED plots → {ctx["out_dir"]}'
                + ('' if ctx['inj'] is not None else ' (no injection test: statistical errors only)'))
    return {'n': len(rows), 'dir': ctx['out_dir'], 'systematics': ctx['inj'] is not None,
            'rows': rows}
