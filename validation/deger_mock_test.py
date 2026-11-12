"""
Deger+22 (PHANGS-HST catalog) photometry vs our methods on noisy mock clusters.

A mock population of isolated clusters at the distance of NGC1365 (19.57 Mpc),
in HST F555W at 0.0396"/px with NGC1365's PSF:
  profiles  EFF gamma=3, King c=1.5, EFF gamma=2.2 (extended halo)
  R_eff     0 (point), 1, 2, 3, 5, 8 pc
  S/N       10, 30, 100, 300 in the catalog's 4 px aperture (background-limited
            Gaussian noise on a constant sky pedestal)
  PSF       'A' = the galaxy-matched instrument PSF our fit uses,
            'B' = a circular Moffat with NGC1365's measured FWHM/alpha (so the
                  instrument fit is not tested against its own PSF model)
  n_real    noise realizations per case, random sub-pixel centres

Methods, all on the same noisy image:
  deger_pop   Deger+22 recipe done faithfully: 4 px flux with sky 7-8 px, times
              ONE average aperture correction derived (3 steps, Section 4.1)
              from the bright (S/N 300) members of this same mock population
  deger_1365  the same 4 px flux times NGC1365's real catalog correction (x1.752)
  growth20    each cluster's own light within 20 px, sky 21-23 px
  native, instG, instP_eff3, instP_king15  — our pipeline functions

Truth = the model's total flux (light beyond the PSF stamp included); the
truth within 20 px is stored too (Deger's definition of "total").

Usage:  python deger_mock_test.py --workers 32 [--n-real 20]
Writes <validation_output_dir>/output_deger_mock/deger_mock.{fits,png}.
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
sys.path.insert(0, os.path.dirname(_HERE))
from _paths import OUT_ROOT  # noqa: E402

from astropy.io import fits  # noqa: E402
from astropy.table import Table  # noqa: E402
from astropy.wcs import WCS  # noqa: E402

from instrument_psf import instrument_psf_for_image  # noqa: E402
from native_phot_aux_func import (compute_r65, build_native_calibration_curve,  # noqa: E402
                                  measure_native_photometry_one_source,
                                  measure_instrument_photometry_one_source)
from injection_test import _ap, _sky, moffat_psf  # noqa: E402

BAND = 'F555W'
PIX = 0.0396
FWHM, ALPHA = 0.125989, 2.825648          # NGC1365 F555W, psf_table.csv
DIST = 19.57
AC_1365 = 1.752                           # NGC1365 catalog correction in V (MW removed)
NPIX = 405                                # 16" image
STAMP = 76                                # +-3" model stamp
PX = dict(r4=4, r7=7, r8=8, r10=10, r20=20, r21=21, r23=23)
PROFILES = {'eff3': ('eff', 3.0, 1.5), 'king15': ('king', 3.0, 1.5), 'eff22': ('eff', 2.2, 1.5)}
REFFS = [0.0, 1.0, 2.0, 3.0, 5.0, 8.0]
SNRS = [10, 30, 100, 300]
EFF3 = dict(profile='eff', gamma=3.0, king_c=1.5)
KING15 = dict(profile='king', gamma=3.0, king_c=1.5)
METHODS = ['deger_pop', 'deger_1365', 'growth20', 'native', 'instG', 'instP_eff3', 'instP_king15']


def make_context():
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crpix = [NPIX // 2 + 1, NPIX // 2 + 1]
    w.wcs.crval = [53.4, -36.14]
    w.wcs.cdelt = [-PIX / 3600, PIX / 3600]
    hdr = w.to_header()
    hdr['BUNIT'] = 'Jy'
    hdr['INSTRUME'] = 'WFC3'
    psf_a, _ = instrument_psf_for_image(BAND, hdr, PIX, target_fwhm_arcsec=FWHM)
    psf_b = moffat_psf(FWHM, ALPHA, PIX)
    r65 = compute_r65(FWHM, ALPHA, PIX, target_ee=0.65)
    calib = build_native_calibration_curve(FWHM, ALPHA, PIX, r65, n_models=20, n_slits=12,
                                           slit_width_px=2, slit_length_arcsec=1.0)
    bd = dict(image=None, header=hdr, wcs=w, instrument='HST', calib_curve=calib,
              r65_psf_arcsec=r65, pixel_scale_arcsec=PIX, r_value=None, central_wl_AA=5302.8,
              unit_convert=None, psf_model=psf_a)
    return dict(bd=bd, psf={'A': psf_a, 'B': psf_b}, wcs=w)


_CTX = {}


def _init(ctx):
    global _CTX
    _CTX = ctx
    for name in ('analysis', 'native_phot_aux_func'):
        logging.getLogger(name).setLevel(logging.ERROR)


def render_truth(psf, prof_tag, reff_pc, rng):
    """unit-total-flux model at a random sub-pixel centre near the image centre"""
    prof, gamma, king_c = PROFILES[prof_tag]
    reff_as = reff_pc * 206265.0 / (DIST * 1e6)
    xc = NPIX // 2 + rng.uniform(-0.5, 0.5)
    yc = NPIX // 2 + rng.uniform(-0.5, 0.5)
    ix, iy = int(np.round(xc)), int(np.round(yc))
    sx, sy = STAMP + (xc - ix), STAMP + (yc - iy)
    kw = dict(flux=1.0)
    if reff_as > 0:
        kw.update(profile=prof, r_eff_arcsec=reff_as, gamma=gamma, c=king_c)
    stamp = psf.render(sx, sy, (2 * STAMP + 1, 2 * STAMP + 1), **kw)
    model = np.zeros((NPIX, NPIX))
    model[iy - STAMP:iy + STAMP + 1, ix - STAMP:ix + STAMP + 1] = stamp
    return model, xc, yc


def measure_all(img, xc, yc, ctx):
    """catalog-style apertures (centred on the true position) and our pipeline methods"""
    s78 = _sky(img, xc, yc, PX['r7'], PX['r8'])
    s2123 = _sky(img, xc, yc, PX['r21'], PX['r23'])
    row = dict(f4_s78=_ap(img, xc, yc, PX['r4'], s78),
               f4=_ap(img, xc, yc, PX['r4'], s2123), f10=_ap(img, xc, yc, PX['r10'], s2123),
               f20=_ap(img, xc, yc, PX['r20'], s2123),
               f7_s78=_ap(img, xc, yc, PX['r7'], s78), f8_s78=_ap(img, xc, yc, PX['r8'], s78))
    row['growth20'] = row['f20']

    bd = dict(ctx['bd'], image=img)
    ra, dec = (float(v) for v in ctx['wcs'].pixel_to_world_values(xc, yc))
    try:
        nat = measure_native_photometry_one_source(ra, dec, {'f555w': bd}, eb_v=0.0)['f555w']
        nr = {'f555w': dict(aperture_source=nat.aperture_source, centroid_ra=nat.centroid_ra,
                            centroid_dec=nat.centroid_dec)}
        ie = measure_instrument_photometry_one_source(ra, dec, nr, {'f555w': bd}, EFF3,
                                                      distance_mpc=DIST, eb_v=0.0)['f555w']
        ik = measure_instrument_photometry_one_source(ra, dec, nr, {'f555w': bd}, KING15,
                                                      distance_mpc=DIST, eb_v=0.0)['f555w']
        row.update(native=nat.flux_mJy / 1e3, native_status=nat.status, native_apsrc=nat.aperture_source,
                   instG=ie['instG'].flux_mJy / 1e3, instP_eff3=ie['instP'].flux_mJy / 1e3,
                   instP_king15=ik['instP'].flux_mJy / 1e3, reff_fit_pc=ie['instP'].r_eff_pc,
                   inst_src=ie['instP'].size_source, error='')
    except Exception as e:
        row.update(native=np.nan, native_status='', native_apsrc='', instG=np.nan, instP_eff3=np.nan,
                   instP_king15=np.nan, reff_fit_pc=np.nan, inst_src='', error=str(e)[:80])
    return row


def run_one(task):
    prof_tag, reff_pc, snr, psf_tag, seed = task
    ctx = _CTX
    rng = np.random.default_rng(seed)
    model, xc, yc = render_truth(ctx['psf'][psf_tag], prof_tag, reff_pc, rng)
    true_r20 = _ap(model, xc, yc, PX['r20'], 0.0)
    f4_true = _ap(model, xc, yc, PX['r4'], 0.0)
    sigma = f4_true / (snr * np.sqrt(np.pi * PX['r4'] ** 2))
    pedestal = 20.0 * sigma
    img = model + pedestal + rng.normal(0.0, sigma, model.shape)
    row = dict(profile=prof_tag, reff_pc=reff_pc, snr=snr, psf=psf_tag, seed=seed,
               true_total=1.0, true_r20=true_r20)
    row.update(measure_all(img, xc, yc, ctx))
    return row


def deger_average_correction(t):
    """Deger+22 Section 4.1 on the bright (S/N 300) members, as a flux factor:
    step 1  4 -> 10 px  and  step 2  10 -> 20 px from the MEAN normalized growth
    curves (sky 21-23 px), step 3 = the cluster light removed by the 7-8 px sky
    (mean ratio of the 4 px flux with the far sky to that with the 7-8 px sky)."""
    b = t[t['snr'] == max(SNRS)]
    s1 = 1.0 / np.mean(np.asarray(b['f4']) / np.asarray(b['f10']))
    s2 = 1.0 / np.mean(np.asarray(b['f10']) / np.asarray(b['f20']))
    s3 = np.mean(np.asarray(b['f4']) / np.asarray(b['f4_s78']))
    return s1 * s2 * s3, (s1, s2, s3)


def case(t, pt, rf, psf_tag):
    """rows of one truth case; the point source (R_eff 0) is shared by all profiles"""
    m = (t['psf'] == psf_tag) & (t['reff_pc'] == rf)
    return m if rf == 0 else m & (t['profile'] == pt)


def summarize(t, out_png):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = dict(deger_pop='#2a78d6', deger_1365='#eb6834', growth20='#1baf7a', native='#eda100',
                  instG='#e87ba4', instP_eff3='#4a3aa7', instP_king15='#008300')
    markers = dict(deger_pop='o', deger_1365='s', growth20='^', native='D', instG='v',
                   instP_eff3='P', instP_king15='X')
    for psf_tag in ('A', 'B'):
        fig, axes = plt.subplots(len(PROFILES), len(REFFS), figsize=(3.0 * len(REFFS), 2.7 * len(PROFILES)),
                                 sharex=True, sharey=True)
        for i, pt in enumerate(PROFILES):
            for j, rf in enumerate(REFFS):
                ax = axes[i, j]
                sel = case(t, pt, rf, psf_tag)
                for k, m in enumerate(METHODS):
                    med, lo, hi = [], [], []
                    for s in SNRS:
                        r = np.asarray(t[m][sel & (t['snr'] == s)], float)
                        r = r[np.isfinite(r)]
                        med.append(np.median(r) if r.size else np.nan)
                        lo.append(np.percentile(r, 16) if r.size else np.nan)
                        hi.append(np.percentile(r, 84) if r.size else np.nan)
                    xs = np.array(SNRS) * (1 + 0.06 * (k - 3))
                    med, lo, hi = map(np.array, (med, lo, hi))
                    ax.errorbar(xs, med, yerr=[med - lo, hi - med], fmt=markers[m] + '-', color=colors[m],
                                ms=5, lw=1.0, elinewidth=0.8, capsize=0, label=m)
                r20 = np.nanmedian(np.asarray(t['true_r20'][sel], float))
                ax.axhline(1.0, color='#888', ls='--', lw=1)
                ax.axhline(r20, color='#1baf7a', ls=':', lw=1)
                ax.set_xscale('log')
                ax.set_ylim(0.4, 1.8)
                ax.grid(alpha=0.25)
                if i == 0:
                    ax.set_title(f'R_eff = {rf:g} pc' if rf > 0 else 'point source', fontsize=9)
                if j == 0:
                    ax.set_ylabel(f'{pt}\nrecovered / true', fontsize=9)
                if i == len(PROFILES) - 1:
                    ax.set_xlabel('S/N (4 px)', fontsize=9)
                    ax.set_xticks(SNRS)
                    ax.set_xticklabels([str(s) for s in SNRS])
        h, lab = axes[0, 0].get_legend_handles_labels()
        fig.legend(h, lab, loc='lower center', bbox_to_anchor=(0.5, 0.93), ncol=7, fontsize=8, frameon=False)
        fig.suptitle(f'Mock clusters at 19.6 Mpc, F555W, truth PSF {psf_tag}  '
                     f'(dotted green: true light within 20 px)', y=1.0, fontsize=11)
        plt.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(out_png.replace('.png', f'_psf{psf_tag}.png'), dpi=120)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--workers', type=int, default=32)
    ap.add_argument('--n-real', type=int, default=20)
    ap.add_argument('--seed', type=int, default=20261006)
    args = ap.parse_args()
    t0 = time.time()
    ctx = make_context()
    tasks, k = [], 0
    for psf_tag in ('A', 'B'):
        for pt in PROFILES:
            for rf in REFFS:
                if rf == 0 and pt != 'eff3':
                    continue                      # one point-source case is enough
                for s in SNRS:
                    for _ in range(args.n_real):
                        tasks.append((pt, rf, s, psf_tag, args.seed + k))
                        k += 1
    print(f'{len(tasks)} mocks', flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('fork'),
                             initializer=_init, initargs=(ctx,)) as ex:
        for i, r in enumerate(ex.map(run_one, tasks, chunksize=4)):
            rows.append(r)
            if (i + 1) % 200 == 0:
                print(f'  [{i + 1}/{len(tasks)}] {time.time() - t0:.0f} s', flush=True)
    t = Table(rows=rows)
    t['deger_pop'] = np.nan
    for psf_tag in ('A', 'B'):
        m = t['psf'] == psf_tag
        ac, steps = deger_average_correction(t[m])
        t['deger_pop'][m] = np.asarray(t['f4_s78'][m]) * ac
        print(f'PSF {psf_tag}: Deger average correction from the S/N 300 population = x{ac:.3f} '
              f'(steps 4->10 px x{steps[0]:.3f}, 10->20 px x{steps[1]:.3f}, sky x{steps[2]:.3f}; '
              f'NGC1365 real: x{AC_1365})')
    t['deger_1365'] = np.asarray(t['f4_s78']) * AC_1365

    out_dir = os.path.join(OUT_ROOT, 'output_deger_mock')
    os.makedirs(out_dir, exist_ok=True)
    t.write(os.path.join(out_dir, 'deger_mock.fits'), overwrite=True)

    print('\nrecovered / true total flux, median [16-84%]')
    for psf_tag in ('A', 'B'):
        for s in (30, 300):
            print(f'\n  PSF {psf_tag}, S/N {s}')
            print('    ' + 'case'.ljust(16) + ''.join(m.rjust(19) for m in METHODS))
            for pt in PROFILES:
                for rf in REFFS:
                    if rf == 0 and pt != 'eff3':
                        continue
                    sel = case(t, pt, rf, psf_tag) & (t['snr'] == s)
                    line = '    ' + (f'{pt} {rf:g}pc' if rf else 'point').ljust(16)
                    for m in METHODS:
                        r = np.asarray(t[m][sel], float)
                        r = r[np.isfinite(r)]
                        line += (f'{np.median(r):6.3f} [{np.percentile(r, 16):4.2f}-{np.percentile(r, 84):4.2f}]'
                                 if r.size else 'n/a').rjust(19)
                    print(line)
    summarize(t, os.path.join(out_dir, 'deger_mock.png'))
    print(f'\ndone in {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
