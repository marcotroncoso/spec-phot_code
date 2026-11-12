"""
Injection test: fake star clusters of KNOWN total flux are injected into the
real PHANGS images of one galaxy, next to its real clusters (same crowding and
background), and measured with every photometry method we have:

  native        measure_native_photometry_one_source (Gaussian slit size +
                Moffat calibration, R65 aperture)
  instG         PSF (x) Gaussian instrument fit
  instP_eff3    PSF (x) EFF gamma=3 instrument fit
  instP_king15  PSF (x) King c=1.5 instrument fit
  catalog       PHANGS-HST catalog recipe (Deger+22): 4 px (0.158") aperture,
                sky median 7-8 px, times the per-band correction the catalog
                actually applies in this galaxy (median catalog / 4 px flux of
                the real clusters, Milky Way dereddening removed). HST only.
  deger_avg     same 4 px measurement times the galaxy-average 4 px -> 20 px
                correction measured on the real clusters (all bands)
  growth20      this source's own light within 20 px (0.79"), sky 21-23 px
                (Deger's definition of "total", per source)

Truth models (source profile, rendered through the galaxy-matched instrument
PSF 'A', or a Moffat PSF 'B' with the galaxy's measured FWHM/alpha):
  point, eff3_2pc, eff3_5pc, king15_3pc, observed (EFF gamma=2.2 whose
  light(<4px)/light(<20px) equals the median of the real clusters).

Each injection goes into a 16" Cutout2D copy of the real image (one source at
a time, so nothing else changes); its brightness in each band is that of the
host cluster. A blank control measures native/catalog/growth20 at the same
position before injection.

Writes <validation_output_dir>/output_injection/<GAL>_injection.fits (+ .png) and
prints recovered / true per method, truth model and band.

Usage:
    python injection_test.py --galaxy NGC1365-1.15asec --workers 32
    python injection_test.py --max-hosts 2 --workers 1          # dry run
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
from _paths import OUT_ROOT  # noqa: E402

from astropy.coordinates import SkyCoord  # noqa: E402
from astropy.nddata import Cutout2D  # noqa: E402
from astropy.table import Table  # noqa: E402
from astropy.stats import sigma_clipped_stats  # noqa: E402
import astropy.units as u  # noqa: E402
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry  # noqa: E402

import run_full_pipeline as rfp  # noqa: E402
from analysis import Photometry, _catalog_flux_columns  # noqa: E402
from phot_aux_func import corr_flux, jwst_pixel_sum_to_Jy  # noqa: E402
from instrument_psf import InstrumentPSF  # noqa: E402
from native_phot_aux_func import (measure_native_photometry_one_source,  # noqa: E402
                                  measure_instrument_photometry_one_source,
                                  moffat_gamma_from_fwhm, moffat_ee_analytic)

BANDS = ['F275W', 'F336W', 'F438W', 'F555W', 'F814W', 'F200W', 'F300M']
R4, R7, R8 = 0.158, 0.277, 0.317           # 4, 7, 8 HST px (0.0396") in arcsec
R20, R21, R23 = 0.792, 0.832, 0.911        # 20, 21, 23 HST px
CUT_ARCSEC = 16.0
STAMP_ARCSEC = 3.0
EFF3 = dict(profile='eff', gamma=3.0, king_c=1.5)
KING15 = dict(profile='king', gamma=3.0, king_c=1.5)

TRUTHS = {      # tag -> (profile, gamma, king_c, r_eff_pc); 'observed' solved at run time
    'point': ('gaussian', 3.0, 1.5, 0.0),
    'eff3_2pc': ('eff', 3.0, 1.5, 2.0),
    'eff3_5pc': ('eff', 3.0, 1.5, 5.0),
    'king15_3pc': ('king', 3.0, 1.5, 3.0),
    'observed': ('eff', 2.2, 1.5, None),
}
PSF_B_TRUTHS = ('point', 'observed')
METHODS = ['native', 'instG', 'instP_eff3', 'instP_king15', 'catalog', 'deger_avg', 'growth20']

log = logging.getLogger('injection_test')


# ─────────────────────────────────────────────────────────────────────────────
# small photometry helpers (sky medians and exact circular apertures)
# ─────────────────────────────────────────────────────────────────────────────

def _sky(img, x, y, r1_px, r2_px):
    v = CircularAnnulus((x, y), r1_px, r2_px).to_mask('center').get_values(img)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else np.nan


def _ap(img, x, y, r_px, sky):
    ap = CircularAperture((x, y), r_px)
    return float(aperture_photometry(np.nan_to_num(img - sky, nan=0.0), ap)['aperture_sum'][0])


def to_mjy(val_img, bd):
    conv = bd.get('unit_convert')
    return (conv(val_img, bd['header']) if conv else val_img) * 1e3


def catalog_style(img, x, y, pix):
    """4 px flux with sky 7-8 px (image units)."""
    return _ap(img, x, y, R4 / pix, _sky(img, x, y, R7 / pix, R8 / pix))


def growth20(img, x, y, pix):
    """Light within 20 px, sky 21-23 px (image units)."""
    return _ap(img, x, y, R20 / pix, _sky(img, x, y, R21 / pix, R23 / pix))


def growth4_20(img, x, y, pix):
    """light(<4px) / light(<20px), both with sky 21-23 px."""
    s = _sky(img, x, y, R21 / pix, R23 / pix)
    return _ap(img, x, y, R4 / pix, s) / _ap(img, x, y, R20 / pix, s)


# ─────────────────────────────────────────────────────────────────────────────
# PSF B: circular Moffat with the galaxy's measured FWHM / alpha
# ─────────────────────────────────────────────────────────────────────────────

def moffat_psf(fwhm, alpha, pixel_scale, n_fine=4, radius_arcsec=2.5):
    fine = pixel_scale / n_fine
    h = int(np.ceil(radius_arcsec / fine))
    ax = np.arange(-h, h + 1) * fine
    rr = np.hypot(*np.meshgrid(ax, ax))
    g = moffat_gamma_from_fwhm(fwhm, alpha)
    psf = (1.0 + (rr / g) ** 2) ** (-alpha)
    psf[rr > h * fine] = 0.0
    total = float(moffat_ee_analytic(h * fine, fwhm, alpha))
    return InstrumentPSF(psf, fine, pixel_scale, extra_fwhm_arcsec=0.0, n_fine=n_fine,
                         total_flux=total, name=f'Moffat fwhm={fwhm:.3f} alpha={alpha:.2f}')


# ─────────────────────────────────────────────────────────────────────────────
# setup (main process)
# ─────────────────────────────────────────────────────────────────────────────

def setup(token):
    os.chdir(_ROOT)
    cfg = rfp.load_config('config.ini')
    gal = token.split('-')[0]
    p = Photometry([token, gal], base_dir=cfg['output_dir'], psf_filepath=cfg['psf_filepath'],
                   psf_source=cfg.get('psf_source', 'theoretical'))
    p.load_isf_results()
    p.load_images(rfp.find_hst_band_paths(cfg['hst_root'], gal),
                  rfp.find_jwst_band_paths(cfg['jwst_root'], gal.lower()),
                  muse_path=cfg['muse_dap_root'])
    p.filter_by_common_fov(plot=False)
    p.load_or_measure_psf()
    p.convolve_images()
    bands_all = {b: v for b, v in rfp.FILTER_WAVELENGTHS.items()
                 if b.lower() in p.hst_images or b.lower() in p.jwst_images}
    try:
        p._find_galaxy_eb_v()
    except Exception:
        p.eb_v = 0.0
    p.run_spectro_photometry(bands_all, n_workers=1, verbose=False)
    p.hst_conv.clear()
    p.jwst_conv.clear()
    p.measure_native_psf_shape()
    p.run_native_photometry(bands_all, verbose=False, plot_diagnostics=False, instrument_fit=True)
    return p, gal


def build_context(p, bands, max_hosts, n_pos, seed):
    tbl = p.photometry_table
    n_real = len(tbl)
    band_data = {}
    for b in bands:
        bl = b.lower()
        if bl not in p.native_band_calib or bl not in p.instrument_psf:
            log.warning(f'{b}: not available, skipped')
            continue
        bd = dict(p.native_band_calib[bl], psf_model=p.instrument_psf[bl])
        bd['mw'] = float(corr_flux(1.0, bd['r_value'], p.eb_v)) if bd.get('r_value') is not None else 1.0
        if bd['instrument'] == 'HST':
            fwhm, alpha = p.psf_hst.get(bl), p.psf_hst_alpha.get(bl)
        else:
            fwhm, alpha = p.psf_jwst.get(bl), p.psf_jwst_alpha.get(bl)
        bd['psf_B'] = moffat_psf(fwhm, alpha, bd['pixel_scale_arcsec'])
        band_data[bl] = bd
    bands = [b for b in bands if b.lower() in band_data]
    ra_r = np.asarray(tbl['PHANGS_RA'], float)
    dec_r = np.asarray(tbl['PHANGS_DEC'], float)
    dist = float(np.nanmedian(np.asarray(tbl['PHANGS_DMPC'], float)))
    real_sky = SkyCoord(ra_r * u.deg, dec_r * u.deg)

    # galaxy-level corrections measured on the real clusters
    corr_cat, corr_deger, ratio_obs = {}, {}, []
    for b in bands:
        bd = band_data[b.lower()]
        pix = bd['pixel_scale_arcsec']
        f4, f20, cat = [], [], []
        cat_col, _ = _catalog_flux_columns(b, tbl.colnames)
        for i in range(n_real):
            x, y = (float(v) for v in real_sky[i].to_pixel(bd['wcs']))
            f4.append(to_mjy(catalog_style(bd['image'], x, y, pix), bd))
            f20.append(to_mjy(growth20(bd['image'], x, y, pix), bd))
            cat.append(float(tbl[cat_col][i]) if cat_col else np.nan)
            if b == 'F555W':
                ratio_obs.append(growth4_20(bd['image'], x, y, pix))
        f4, f20, cat = map(np.asarray, (f4, f20, cat))
        ok = (f4 > 0) & np.isfinite(cat) & (cat > 0)
        corr_cat[b] = float(np.median(cat[ok] / f4[ok]) / bd['mw']) if ok.any() else np.nan
        ok = (f4 > 0) & (f20 > 0)
        corr_deger[b] = float(np.median(f20[ok] / f4[ok]))
    target_ratio = float(np.nanmedian(ratio_obs))

    # 'observed' truth: EFF gamma=2.2 radius matching the real clusters' 4px/20px ratio (F555W)
    bd5 = band_data['f555w']
    psf5, pix5 = bd5['psf_model'], bd5['pixel_scale_arcsec']
    arcsec_per_pc = 206265.0 / (dist * 1e6)

    def model_ratio(reff_pc):
        n = 2 * int(np.ceil(1.2 / pix5)) + 1
        c = (n - 1) / 2
        img = psf5.render(c, c, (n, n), profile='eff', r_eff_arcsec=reff_pc * arcsec_per_pc, gamma=2.2)
        return _ap(img, c, c, R4 / pix5, 0.0) / _ap(img, c, c, R20 / pix5, 0.0)

    lo, hi = 0.3, 60.0
    if model_ratio(hi) > target_ratio or model_ratio(lo) < target_ratio:
        log.warning('observed truth: target ratio out of reach; using bound')
    for _ in range(25):
        mid = np.sqrt(lo * hi)
        if model_ratio(mid) > target_ratio:
            lo = mid
        else:
            hi = mid
    reff_obs = float(np.sqrt(lo * hi))
    truths = dict(TRUTHS)
    truths['observed'] = ('eff', 2.2, 1.5, reff_obs)

    # injection positions: quiet, covered, separated
    rng = np.random.default_rng(seed)
    hosts = list(range(n_real))[:max_hosts] if max_hosts else list(range(n_real))
    w5, img5 = bd5['wcs'], bd5['image']
    chosen = []
    positions = []
    for h in hosts:
        n_ok, tries = 0, 0
        while n_ok < n_pos and tries < 400:
            tries += 1
            r = rng.uniform(1.5, 4.0)
            th = rng.uniform(0, 2 * np.pi)
            dra = r * np.cos(th) / 3600.0 / np.cos(np.deg2rad(dec_r[h]))
            ddec = r * np.sin(th) / 3600.0
            pos = SkyCoord((ra_r[h] + dra) * u.deg, (dec_r[h] + ddec) * u.deg)
            if np.min(pos.separation(real_sky).arcsec) < 1.5:
                continue
            if chosen and np.min(pos.separation(SkyCoord(chosen)).arcsec) < 1.5:
                continue
            ok = True
            for b in bands:
                bd = band_data[b.lower()]
                x, y = (float(v) for v in pos.to_pixel(bd['wcs']))
                v = CircularAperture((x, y), 1.0 / bd['pixel_scale_arcsec']).to_mask('center').get_values(bd['image'])
                if v.size == 0 or not np.all(np.isfinite(v)):
                    ok = False
                    break
            if not ok:
                continue
            x, y = (float(v) for v in pos.to_pixel(w5))
            inner = CircularAperture((x, y), 0.5 / pix5).to_mask('center').get_values(img5)
            ann = CircularAnnulus((x, y), 0.5 / pix5, 1.5 / pix5).to_mask('center').get_values(img5)
            _, med, std = sigma_clipped_stats(ann[np.isfinite(ann)])
            if np.nanmax(inner) - med > 5 * std:
                continue
            chosen.append(pos)
            positions.append((h, n_ok, float(pos.ra.deg), float(pos.dec.deg)))
            n_ok += 1
    log.info(f'{len(positions)} injection positions around {len(hosts)} hosts')

    # host brightness per band (mJy, Milky Way dereddening removed)
    host_flux = {}
    for h in hosts:
        for b in bands:
            f = float(tbl[f'flux_instP_{b}'][h]) if f'flux_instP_{b}' in tbl.colnames else np.nan
            if not np.isfinite(f) or f <= 0:
                f = float(tbl[f'flux_native_{b}'][h])
            host_flux[(h, b)] = f / band_data[b.lower()]['mw'] if np.isfinite(f) and f > 0 else np.nan

    tasks = []
    for (h, k, ra, dec) in positions:
        for tag in truths:
            tasks.append((h, k, ra, dec, tag, 'A'))
        for tag in PSF_B_TRUTHS:
            tasks.append((h, k, ra, dec, tag, 'B'))
    classes = [c.decode().strip() if isinstance(c, bytes) else str(c).strip()
               for c in tbl['PHANGS_CC_CLASS_ML']]
    return dict(band_data=band_data, bands=bands, truths=truths, host_flux=host_flux,
                dist=dist, corr_cat=corr_cat, corr_deger=corr_deger, classes=classes,
                target_ratio=target_ratio, reff_obs=reff_obs), tasks


# ─────────────────────────────────────────────────────────────────────────────
# one injection (worker)
# ─────────────────────────────────────────────────────────────────────────────

_CTX = {}


def _init(ctx):
    global _CTX
    _CTX = ctx
    logging.getLogger('analysis').setLevel(logging.ERROR)
    logging.getLogger('native_phot_aux_func').setLevel(logging.ERROR)


def run_injection(task):
    h, k, ra, dec, tag, psf_tag = task
    ctx = _CTX
    prof, gamma, king_c, reff_pc = ctx['truths'][tag]
    reff_as = reff_pc * 206265.0 / (ctx['dist'] * 1e6)
    pos = SkyCoord(ra * u.deg, dec * u.deg)

    cut_bd, blank_bd, truth = {}, {}, {}
    for b in ctx['bands']:
        bd = ctx['band_data'][b.lower()]
        pix = bd['pixel_scale_arcsec']
        n = int(np.ceil(CUT_ARCSEC / pix)) | 1
        cut = Cutout2D(bd['image'], pos, (n, n), wcs=bd['wcs'], mode='partial',
                       fill_value=np.nan, copy=True)
        blank = cut.data.copy()
        flux_mjy = ctx['host_flux'].get((h, b), np.nan)
        x, y = (float(v) for v in pos.to_pixel(cut.wcs))
        truth[b] = dict(total=flux_mjy, r20=np.nan, x=x, y=y)
        if np.isfinite(flux_mjy):
            unit = to_mjy(1.0, bd)                     # mJy per image unit of summed flux
            f_img = flux_mjy / unit
            psf = bd['psf_model'] if psf_tag == 'A' else bd['psf_B']
            hs = int(np.ceil(STAMP_ARCSEC / pix))
            ix, iy = int(np.round(x)), int(np.round(y))
            sx, sy = hs + (x - ix), hs + (y - iy)
            kw = dict(flux=f_img)
            if reff_as > 0:
                kw.update(profile=prof, r_eff_arcsec=reff_as, gamma=gamma, c=king_c)
            stamp = psf.render(sx, sy, (2 * hs + 1, 2 * hs + 1), **kw)
            y0, x0 = iy - hs, ix - hs
            ys = slice(max(y0, 0), min(y0 + 2 * hs + 1, n))
            xs = slice(max(x0, 0), min(x0 + 2 * hs + 1, n))
            cut.data[ys, xs] += stamp[ys.start - y0:ys.stop - y0, xs.start - x0:xs.stop - x0]
            truth[b]['r20'] = _ap(stamp, sx, sy, R20 / pix, 0.0) * unit
        cut_bd[b.lower()] = dict(bd, image=cut.data, wcs=cut.wcs)
        blank_bd[b.lower()] = dict(bd, image=blank, wcs=cut.wcs)

    rows = []
    try:
        nat = measure_native_photometry_one_source(ra, dec, cut_bd, eb_v=0.0)
        nat_blank = measure_native_photometry_one_source(ra, dec, blank_bd, eb_v=0.0)
        native_row = {bl: dict(aperture_source=r.aperture_source, centroid_ra=r.centroid_ra,
                               centroid_dec=r.centroid_dec) for bl, r in nat.items()}
        ins_e = measure_instrument_photometry_one_source(ra, dec, native_row, cut_bd, EFF3,
                                                         distance_mpc=ctx['dist'], eb_v=0.0)
        ins_k = measure_instrument_photometry_one_source(ra, dec, native_row, cut_bd, KING15,
                                                         distance_mpc=ctx['dist'], eb_v=0.0)
        err = ''
    except Exception as e:                                  # keep the batch alive
        nat = nat_blank = ins_e = ins_k = None
        err = f'{type(e).__name__}: {e}'[:80]

    for b in ctx['bands']:
        bl = b.lower()
        bd = cut_bd[bl]
        pix = bd['pixel_scale_arcsec']
        x, y = truth[b]['x'], truth[b]['y']
        f4 = to_mjy(catalog_style(bd['image'], x, y, pix), bd)
        f4_blank = to_mjy(catalog_style(blank_bd[bl]['image'], x, y, pix), bd)
        row = dict(host=h, pos=k, ra=ra, dec=dec, truth=tag, psf=psf_tag, band=b,
                   host_class=ctx['classes'][h], true_total=truth[b]['total'],
                   true_r20=truth[b]['r20'], error=err,
                   catalog=f4 * ctx['corr_cat'].get(b, np.nan),
                   deger_avg=f4 * ctx['corr_deger'][b],
                   growth20=to_mjy(growth20(bd['image'], x, y, pix), bd),
                   blank_catalog=f4_blank * ctx['corr_cat'].get(b, np.nan),
                   blank_growth20=to_mjy(growth20(blank_bd[bl]['image'], x, y, pix), bd),
                   native=np.nan, status_native='', apsrc_native='', blank_native=np.nan,
                   blank_status_native='', instG=np.nan, instP_eff3=np.nan, instP_king15=np.nan,
                   reff_pc_eff3=np.nan, size_src_inst='')
        if nat is not None:
            row.update(native=nat[bl].flux_mJy, status_native=nat[bl].status,
                       apsrc_native=nat[bl].aperture_source,
                       blank_native=nat_blank[bl].flux_mJy, blank_status_native=nat_blank[bl].status,
                       instG=ins_e[bl]['instG'].flux_mJy, instP_eff3=ins_e[bl]['instP'].flux_mJy,
                       instP_king15=ins_k[bl]['instP'].flux_mJy,
                       reff_pc_eff3=ins_e[bl]['instP'].r_eff_pc,
                       size_src_inst=ins_e[bl]['instP'].size_source)
        rows.append(row)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# summary + plot
# ─────────────────────────────────────────────────────────────────────────────

def summarize(t, bands, out_png, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    groups = [(tag, 'A') for tag in TRUTHS] + [(tag, 'B') for tag in PSF_B_TRUTHS]
    colors = dict(native='seagreen', instG='red', instP_eff3='darkorange', instP_king15='purple',
                  catalog='dimgray', deger_avg='royalblue', growth20='black')
    fig, axes = plt.subplots(len(groups), 1, figsize=(max(10, 1.6 * len(bands) + 3), 3.0 * len(groups)),
                             sharex=True)
    rng = np.random.default_rng(0)
    pos = np.arange(len(bands))
    width = 0.8 / len(METHODS)
    print(f'\nrecovered / true total flux: median [16-84%]   ({title})')
    for ax, (tag, psf) in zip(axes, groups):
        sel = (t['truth'] == tag) & (t['psf'] == psf)
        print(f'\n  truth = {tag}, PSF {psf}')
        print('  ' + 'method'.ljust(14) + ''.join(b.rjust(19) for b in bands))
        for j, m in enumerate(METHODS):
            off = (j - (len(METHODS) - 1) / 2) * width
            line = '  ' + m.ljust(14)
            data = []
            for b in bands:
                s = sel & (t['band'] == b)
                with np.errstate(all='ignore'):
                    r = np.asarray(t[m][s], float) / np.asarray(t['true_total'][s], float)
                r = r[np.isfinite(r)]
                data.append(r if r.size else np.array([np.nan]))
                line += (f'{np.median(r):7.3f} [{np.percentile(r, 16):4.2f}-{np.percentile(r, 84):4.2f}]'
                         if r.size else '                n/a').rjust(19)
            print(line)
            ax.boxplot(data, positions=pos + off, widths=width * 0.9, showfliers=False,
                       patch_artist=True, boxprops=dict(facecolor=colors[m], alpha=0.35),
                       medianprops=dict(color=colors[m]))
            for p_, d in zip(pos, data):
                ok = np.isfinite(d)
                ax.scatter(p_ + off + rng.normal(0, width * 0.1, ok.sum()), d[ok], s=4,
                           color=colors[m], alpha=0.6, zorder=3)
            ax.plot([], [], 's', color=colors[m], alpha=0.5, label=m)
        # where the 20 px "total" of this truth model lies
        r20 = [np.nanmedian(np.asarray(t['true_r20'][sel & (t['band'] == b)], float) /
                            np.asarray(t['true_total'][sel & (t['band'] == b)], float)) for b in bands]
        ax.plot(pos, r20, 'k_', ms=25, mew=1.5, label='true light < 20 px')
        ax.axhline(1.0, color='k', ls='--', lw=1)
        ax.set_ylim(0.3, 2.0)
        ax.set_ylabel(f'{tag}\nPSF {psf}\nrecovered / true')
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=7, ncol=8, loc='upper left')
    axes[-1].set_xticks(pos)
    axes[-1].set_xticklabels(bands)
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    plt.close(fig)

    # blank control: environment light at the injection positions
    print('\n  blank control (no injection), median environment flux / median injected flux:')
    sel = (t['truth'] == 'point') & (t['psf'] == 'A')
    for m in ('blank_native', 'blank_catalog', 'blank_growth20'):
        vals = []
        for b in bands:
            s = sel & (t['band'] == b)
            env = np.asarray(t[m][s], float) if m in t.colnames else np.array([np.nan])
            vals.append(np.nanmedian(env) / np.nanmedian(np.asarray(t['true_total'][s], float)))
        print('  ' + m.ljust(16) + ''.join(f'{v:9.3f}' for v in vals))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--galaxy', default='NGC1365-1.15asec')
    ap.add_argument('--workers', type=int, default=32)
    ap.add_argument('--max-hosts', type=int, default=0, help='0 = all real clusters')
    ap.add_argument('--n-pos', type=int, default=2, help='injection positions per host')
    ap.add_argument('--seed', type=int, default=20261005)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    t0 = time.time()
    p, gal = setup(args.galaxy)
    ctx, tasks = build_context(p, BANDS, args.max_hosts, args.n_pos, args.seed)
    print(f'setup {time.time() - t0:.0f} s; {len(tasks)} injections; '
          f'catalog corrections {ctx["corr_cat"]}; deger_avg corrections {ctx["corr_deger"]}; '
          f'real clusters light(<4px)/light(<20px) = {ctx["target_ratio"]:.3f} -> '
          f'observed truth R_eff = {ctx["reff_obs"]:.2f} pc (EFF gamma=2.2)')

    rows = []
    if args.workers <= 1:
        _init(ctx)
        for i, task in enumerate(tasks):
            rows += run_injection(task)
            print(f'  [{i + 1}/{len(tasks)}] {time.time() - t0:.0f} s', flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('fork'),
                                 initializer=_init, initargs=(ctx,)) as ex:
            for i, r in enumerate(ex.map(run_injection, tasks, chunksize=1)):
                rows += r
                if (i + 1) % 20 == 0 or i + 1 == len(tasks):
                    print(f'  [{i + 1}/{len(tasks)}] {time.time() - t0:.0f} s', flush=True)

    t = Table(rows=rows)
    t.meta.update(GALAXY=gal, REFFOBS=ctx['reff_obs'], RATIOOBS=ctx['target_ratio'])
    out_dir = os.path.join(OUT_ROOT, 'output_injection')
    os.makedirs(out_dir, exist_ok=True)
    out_fits = os.path.join(out_dir, f'{gal}_injection.fits')
    t.write(out_fits, overwrite=True)
    summarize(t, ctx['bands'], os.path.join(out_dir, f'{gal}_injection.png'),
              f'{gal} injection test ({len(tasks)} injections)')
    print(f'\nsaved {out_fits} ({time.time() - t0:.0f} s)')


if __name__ == '__main__':
    main()
