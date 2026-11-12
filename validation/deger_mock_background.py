"""
How non-flat backgrounds bias each photometry method: the noisy mock clusters of
deger_mock_test.py (F555W, NGC1365 PSF 'A', 19.57 Mpc) placed on structured
backgrounds plus the same Gaussian noise.

Background scenarios, amplitudes in units of S = the cluster's mean surface
brightness inside the 4 px aperture (so the effect scales with the cluster):
  flat        constant sky pedestal only (= deger_mock_test.py)
  offset5     extra constant offset of 5 S
  grad0.5/2   linear gradient, random direction, changing by 0.5 / 2 S per 20 px
  clump0.1/.3 small-scale Gaussian random field (smoothing sigma 2 px ~ PSF),
              pixel rms 0.1 / 0.3 S: unresolved knots of a star-forming complex
  diff0.1/.3  large-scale Gaussian random field (sigma 8 px = 0.32"), pixel rms
              0.1 / 0.3 S: diffuse disk light varying on cluster scales
  neigh2/5    2 / 5 compact neighbours (EFF3, 1 pc) with 0.2-1.0x the cluster's
              flux at 0.3-1.5" from it

Every mock also stores bkg_f4: the 4 px flux (sky 7-8 px) of the background
alone, without cluster or noise, i.e. how much light the scenario leaks into
the catalog's aperture.

Deger's average correction is the one derived from the flat S/N 300 population
of deger_mock_test.py (read from output_deger_mock/deger_mock.fits), so any
change relative to 'flat' is the background's effect on the measurement.

Usage:  python deger_mock_background.py --workers 48 [--n-real 20]
Writes <validation_output_dir>/output_deger_mock/deger_mock_background.{fits,png}
and deger_mock_background_examples.png.
"""
import argparse
import logging
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from astropy.table import Table
from scipy.ndimage import gaussian_filter

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from _paths import OUT_ROOT  # noqa: E402

import deger_mock_test as dm  # noqa: E402
from deger_mock_test import NPIX, STAMP, PX, PIX, DIST, AC_1365  # noqa: E402
from injection_test import _ap, _sky  # noqa: E402

CASES = [('eff3', 0.0), ('eff3', 2.0), ('eff3', 5.0), ('eff22', 5.0)]
SNRS = [30, 100, 300]
SCENARIOS = ['flat', 'offset5', 'grad0.5', 'grad2', 'clump0.1', 'clump0.3',
             'diff0.1', 'diff0.3', 'neigh2', 'neigh5']
METHODS = ['deger_pop', 'deger_1365', 'growth20', 'native', 'instG', 'instP_eff3', 'instP_king15']
PLOT_METHODS = ['deger_pop', 'deger_1365', 'growth20', 'native', 'instG', 'instP_eff3']


def make_background(scenario, s_mean, rng, psf):
    """background image (no noise, no pedestal) for one scenario"""
    bkg = np.zeros((NPIX, NPIX))
    if scenario == 'flat':
        return bkg
    if scenario.startswith('offset'):
        return bkg + float(scenario[6:]) * s_mean
    if scenario.startswith('grad'):
        k = float(scenario[4:])
        th = rng.uniform(0, 2 * np.pi)
        yy, xx = np.mgrid[:NPIX, :NPIX] - NPIX // 2
        return k * s_mean / PX['r20'] * (xx * np.cos(th) + yy * np.sin(th))
    if scenario.startswith(('clump', 'diff')):
        k = float(scenario.lstrip('clumpdif'))
        f = gaussian_filter(rng.normal(size=(NPIX, NPIX)), 2.0 if scenario.startswith('clump') else 8.0)
        return k * s_mean * f / f.std()
    if scenario.startswith('neigh'):
        for _ in range(int(scenario[5:])):
            r = rng.uniform(0.3, 1.5) / PIX
            th = rng.uniform(0, 2 * np.pi)
            x, y = NPIX // 2 + r * np.cos(th), NPIX // 2 + r * np.sin(th)
            ix, iy = int(np.round(x)), int(np.round(y))
            reff_as = 1.0 * 206265.0 / (DIST * 1e6)
            bkg[iy - STAMP:iy + STAMP + 1, ix - STAMP:ix + STAMP + 1] += psf.render(
                STAMP + x - ix, STAMP + y - iy, (2 * STAMP + 1, 2 * STAMP + 1),
                flux=rng.uniform(0.2, 1.0), profile='eff', r_eff_arcsec=reff_as, gamma=3.0, c=1.5)
        return bkg
    raise ValueError(scenario)


def build_image(prof_tag, reff_pc, snr, scenario, seed, psf):
    rng = np.random.default_rng(seed)
    model, xc, yc = dm.render_truth(psf, prof_tag, reff_pc, rng)
    f4_true = _ap(model, xc, yc, PX['r4'], 0.0)
    s_mean = f4_true / (np.pi * PX['r4'] ** 2)
    sigma = f4_true / (snr * np.sqrt(np.pi * PX['r4'] ** 2))
    bkg = make_background(scenario, s_mean, rng, psf)
    img = model + bkg + 20.0 * sigma + rng.normal(0.0, sigma, model.shape)
    return img, model, bkg, xc, yc


def run_one(task):
    prof_tag, reff_pc, snr, scenario, seed = task
    ctx = dm._CTX
    img, model, bkg, xc, yc = build_image(prof_tag, reff_pc, snr, scenario, seed, ctx['psf']['A'])
    row = dict(profile=prof_tag, reff_pc=reff_pc, snr=snr, scenario=scenario, seed=seed, true_total=1.0,
               true_r20=_ap(model, xc, yc, PX['r20'], 0.0),
               bkg_f4=_ap(bkg, xc, yc, PX['r4'], _sky(bkg, xc, yc, PX['r7'], PX['r8'])),
               bkg_f20=_ap(bkg, xc, yc, PX['r20'], _sky(bkg, xc, yc, PX['r21'], PX['r23'])))
    row.update(dm.measure_all(img, xc, yc, ctx))
    return row


def case_mask(t, pt, rf):
    m = t['reff_pc'] == rf
    return m if rf == 0 else m & (t['profile'] == pt)


def case_label(pt, rf):
    return 'point' if rf == 0 else f'{pt} {rf:g} pc'


def summarize(t, out_png):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = dict(deger_pop='#2a78d6', deger_1365='#eb6834', growth20='#1baf7a', native='#eda100',
                  instG='#e87ba4', instP_eff3='#4a3aa7')
    markers = dict(deger_pop='o', deger_1365='s', growth20='^', native='D', instG='v', instP_eff3='P')
    xs = np.arange(len(SCENARIOS))
    fig, axes = plt.subplots(len(CASES), len(SNRS), figsize=(6.2 * len(SNRS), 2.9 * len(CASES)),
                             sharex=True, sharey=True)
    for i, (pt, rf) in enumerate(CASES):
        for j, s in enumerate(SNRS):
            ax = axes[i, j]
            sel = case_mask(t, pt, rf) & (t['snr'] == s)
            for k, m in enumerate(PLOT_METHODS):
                med, lo, hi = [], [], []
                for sc in SCENARIOS:
                    r = np.asarray(t[m][sel & (t['scenario'] == sc)], float)
                    r = r[np.isfinite(r)]
                    med.append(np.median(r) if r.size else np.nan)
                    lo.append(np.percentile(r, 16) if r.size else np.nan)
                    hi.append(np.percentile(r, 84) if r.size else np.nan)
                med, lo, hi = map(np.array, (med, lo, hi))
                ax.errorbar(xs + 0.12 * (k - 2.5), med, yerr=[med - lo, hi - med], fmt=markers[m],
                            color=colors[m], ms=4.5, elinewidth=0.8, capsize=0, label=m)
            ax.axhline(1.0, color='#888', ls='--', lw=1)
            ax.axhline(np.nanmedian(np.asarray(t['true_r20'][sel], float)), color='#1baf7a', ls=':', lw=1)
            for x in xs[1::2]:
                ax.axvspan(x - 0.5, x + 0.5, color='#000', alpha=0.035, lw=0)
            ax.set_ylim(0.3, 2.0)
            ax.grid(alpha=0.25, axis='y')
            if i == 0:
                ax.set_title(f'S/N (4 px) = {s}', fontsize=10)
            if j == 0:
                ax.set_ylabel(f'{case_label(pt, rf)}\nrecovered / true', fontsize=9)
            if i == len(CASES) - 1:
                ax.set_xticks(xs)
                ax.set_xticklabels(SCENARIOS, rotation=45, ha='right', fontsize=8)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc='lower center', bbox_to_anchor=(0.5, 0.955), ncol=6, fontsize=9, frameon=False)
    fig.suptitle('Mock clusters on structured backgrounds (F555W, NGC1365 PSF, 19.6 Mpc); '
                 'dotted green: true light within 20 px', y=1.0, fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def plot_examples(psf, out_png, seed):
    """one realization per scenario: eff3 2 pc at S/N 100, 2.4" across, with Deger's apertures"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    h = 30
    fig, axes = plt.subplots(2, len(SCENARIOS) // 2, figsize=(3.0 * len(SCENARIOS) // 2, 6.4))
    for ax, sc in zip(axes.ravel(), SCENARIOS):
        img, model, bkg, xc, yc = build_image('eff3', 2.0, 100, sc, seed, psf)
        c = NPIX // 2
        cut = img[c - h:c + h + 1, c - h:c + h + 1] - np.median(img)
        vmax = model.max()
        ax.imshow(np.arcsinh(cut / (0.02 * vmax)), origin='lower', cmap='magma',
                  vmin=np.arcsinh(-0.1 / 0.02), vmax=np.arcsinh(1 / 0.02))
        for r, ls, col in ((4, '-', 'cyan'), (7, ':', 'w'), (8, ':', 'w'), (20, '-', 'lime'),
                           (21, ':', 'lime'), (23, ':', 'lime')):
            ax.add_patch(Circle((xc - c + h, yc - c + h), r, fill=False, ls=ls, color=col, lw=0.8))
        ax.set_title(sc, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle('Example mocks: EFF3 2 pc at S/N 100; apertures 4 px (cyan), sky 7-8 px (white), '
                 '20 px and sky 21-23 px (green)', fontsize=10)
    plt.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--workers', type=int, default=32)
    ap.add_argument('--n-real', type=int, default=20)
    ap.add_argument('--seed', type=int, default=20261007)
    args = ap.parse_args()
    t0 = time.time()
    out_dir = os.path.join(OUT_ROOT, 'output_deger_mock')
    ref = Table.read(os.path.join(out_dir, 'deger_mock.fits'))
    ref = ref[np.char.strip(np.asarray(ref['psf']).astype(str)) == 'A']
    ac, _ = dm.deger_average_correction(ref)
    print(f'Deger average correction from the flat S/N 300 population (PSF A): x{ac:.3f}', flush=True)

    ctx = dm.make_context()
    plot_examples(ctx['psf']['A'], os.path.join(out_dir, 'deger_mock_background_examples.png'), args.seed)
    tasks, k = [], 0
    for pt, rf in CASES:
        for s in SNRS:
            for sc in SCENARIOS:
                for _ in range(args.n_real):
                    tasks.append((pt, rf, s, sc, args.seed + k))
                    k += 1
    print(f'{len(tasks)} mocks', flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('fork'),
                             initializer=dm._init, initargs=(ctx,)) as ex:
        for i, r in enumerate(ex.map(run_one, tasks, chunksize=4)):
            rows.append(r)
            if (i + 1) % 300 == 0:
                print(f'  [{i + 1}/{len(tasks)}] {time.time() - t0:.0f} s', flush=True)
    t = Table(rows=rows)
    t['deger_pop'] = np.asarray(t['f4_s78']) * ac
    t['deger_1365'] = np.asarray(t['f4_s78']) * AC_1365
    t.meta['DEGER_AC'] = ac
    t.write(os.path.join(out_dir, 'deger_mock_background.fits'), overwrite=True)

    print('\nrecovered / true total flux, median [16-84%];  bkg4 = background light in the 4 px aperture '
          '(sky 7-8 px), in units of the cluster total')
    for s in SNRS:
        for pt, rf in CASES:
            print(f'\n  {case_label(pt, rf)}, S/N {s}')
            print('    ' + 'scenario'.ljust(10) + 'bkg4'.rjust(7) + ''.join(m.rjust(19) for m in METHODS))
            for sc in SCENARIOS:
                sel = case_mask(t, pt, rf) & (t['snr'] == s) & (t['scenario'] == sc)
                line = '    ' + sc.ljust(10) + f'{np.median(t["bkg_f4"][sel]):+7.3f}'
                for m in METHODS:
                    r = np.asarray(t[m][sel], float)
                    r = r[np.isfinite(r)]
                    line += (f'{np.median(r):6.3f} [{np.percentile(r, 16):4.2f}-{np.percentile(r, 84):4.2f}]'
                             if r.size else 'n/a').rjust(19)
                print(line)
    summarize(t, os.path.join(out_dir, 'deger_mock_background.png'))
    print(f'\ndone in {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
