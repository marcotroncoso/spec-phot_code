"""
2D model fit (fit2d.py) vs instP (1D averaged slits + aperture) vs Deger, on
the far-neighbour mocks of mock_far_neighbours.py (same seeds: the systems are
pixel-identical to output_mock_far/mock_far.fits), with the time each method
takes.

Methods (truth = 1 mJy in every band):
  instP        pipeline: native (centre, aperture source) + instrument slit fit
  f2d_fixed    2D fit of PSF (x) EFF3 + constant, centre fixed at instP's centre
  f2d_free     same, centre free (|dx|, |dy| <= 1.5 px)
  deger_ideal  4 px flux x the exact correction for this cluster (reference)
Neighbours are neither masked nor fitted in the 2D fits.

Timing (time.perf_counter, per system): native and instP are single calls for
all bands together (they borrow between bands), reported per band as call
time / n_bands; deger and the 2D fits are per band. --timing runs a serial
benchmark (1 process) for clean numbers; the full grid runs in parallel and
its times are only indicative (shared CPU).

Usage:  python mock_2d_far.py [--n-real 20] [--workers 40] [--max-tasks N]
        python mock_2d_far.py --timing [--n-timing 50]
        python mock_2d_far.py --plot
Writes  <validation_output_dir>/output_mock_2d/: mock_2d_far.fits, mock_2d_timing.fits,
        mock_2d_far_HST.png, mock_2d_far_NIRCam.png, mock_2d_pull.png,
        mock_2d_timing.png, mock_2d_residuals.png
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
sys.path.insert(0, _HERE)
from _paths import OUT_ROOT  # noqa: E402

from astropy.table import Table  # noqa: E402

import mock_far_neighbours as mfn  # noqa: E402
from native_phot_aux_func import (measure_native_photometry_one_source,  # noqa: E402
                                  measure_instrument_photometry_one_source, instrument_slit_length)
from injection_test import _ap, _sky, R4, R7, R8, EFF3  # noqa: E402
from deger_aperture_correction import aperture_flux_err  # noqa: E402
from fit2d import fit_2d_instrument  # noqa: E402

OUT = os.path.join(OUT_ROOT, 'output_mock_2d')
DIST = mfn.DIST
ARCSEC_PER_PC = 206265.0 / (DIST * 1e6)
MAX_REFF_PC = 20.0
METHODS = ['instP', 'f2d_fixed', 'f2d_free', 'deger_ideal']
STYLE = dict(mfn.STYLE)
STYLE.update(f2d_fixed=('#1baf7a', 'D', '2D fit (centre fixed)'),
             f2d_free=('#eb6834', 'v', '2D fit (centre free)'),
             deger_ideal=('#2a78d6', 'o', 'Deger exact correction (reference)'))
TIME_COLS = {'native': 't_native', 'instP': 't_instP', 'deger': 't_deger', 'f2d_fixed': 't_f2d_fixed',
             'f2d_free': 't_f2d_free'}
INK, INK2, GRID = mfn.INK, mfn.INK2, mfn.GRID

_CTX = {}


def _init(ctx):
    global _CTX
    _CTX = ctx
    for name in ('analysis', 'native_phot_aux_func', 'instrument_psf'):
        logging.getLogger(name).setLevel(logging.ERROR)


def run_system_2d(task, ctx=None, keep=False):
    """all methods on one mock system, each timed; rendering identical to
    mock_far_neighbours.run_system (same rng sequence)"""
    ctx = ctx or _CTX
    rng = np.random.default_rng(task['seed'])
    bands = ctx['bands']
    cut_bd, geo = {}, {}
    for b in bands:
        img, xc, yc = mfn.render_band(task, b, ctx, rng)
        cut_bd[b.lower()] = dict(ctx['band_data'][b.lower()], image=img)
        geo[b] = (xc, yc)
    sky = cut_bd['f555w']['wcs'].pixel_to_world(*geo['F555W'])
    ra, dec = float(sky.ra.deg), float(sky.dec.deg)
    nb = len(bands)
    t0 = time.perf_counter()
    nat = measure_native_photometry_one_source(ra, dec, cut_bd, eb_v=0.0)
    t_nat = time.perf_counter() - t0
    native_row = {bl: dict(aperture_source=r.aperture_source, centroid_ra=r.centroid_ra,
                           centroid_dec=r.centroid_dec) for bl, r in nat.items()}
    t0 = time.perf_counter()
    ins = measure_instrument_photometry_one_source(ra, dec, native_row, cut_bd, EFF3, distance_mpc=DIST, eb_v=0.0)
    t_ins = time.perf_counter() - t0
    rows, fits = [], {}
    for b in bands:
        info, bd = ctx['info'][b], cut_bd[b.lower()]
        img, pix, upm = bd['image'], info['pix'], info['upm']
        x, y = geo[b]
        t0 = time.perf_counter()
        f4 = _ap(img, x, y, R4 / pix, _sky(img, x, y, R7 / pix, R8 / pix)) / upm
        e4 = aperture_flux_err(img, x, y, R4 / pix, R7 / pix, R8 / pix)[0] / upm
        t_deg = time.perf_counter() - t0
        k_id = info[('ac_ideal', task['reff'])]
        r = ins[b.lower()]['instP']
        # 2D fits start at instP's centre (= native's centroid, or the input position)
        if np.isfinite(r.centroid_ra):
            from astropy.coordinates import SkyCoord
            import astropy.units as u
            xs, ys = (float(v) for v in bd['wcs'].world_to_pixel(SkyCoord(r.centroid_ra * u.deg,
                                                                          r.centroid_dec * u.deg)))
        else:
            xs, ys = x, y
        r_fit = instrument_slit_length(bd, 1.0)
        kw = dict(r_fit_arcsec=r_fit, max_r_eff_arcsec=MAX_REFF_PC * ARCSEC_PER_PC)
        f_fix = fit_2d_instrument(img, xs, ys, bd['psf_model'], EFF3, free_center=False, **kw)
        f_free = fit_2d_instrument(img, xs, ys, bd['psf_model'], EFF3, free_center=True, **kw)
        fits[b] = (f_fix, f_free, (xs, ys))
        row = dict(reff_pc=task['reff'], group=task['group'], snr=task['snr'], real=task['k'], band=b,
                   n_neighbours=len(task['nbs']),
                   instP=r.flux_mJy, instP_err=r.flux_err_mJy, instP_reff_pc=r.r_eff_pc,
                   instP_size_source=str(r.size_source),
                   deger_ideal=f4 * k_id, deger_ideal_err=e4 * k_id,
                   t_native=t_nat / nb, t_instP=t_ins / nb, t_deger=t_deg,
                   centre_err_px=float(np.hypot(xs - x, ys - y)))
        for tag, f in (('f2d_fixed', f_fix), ('f2d_free', f_free)):
            row.update({tag: f.flux / upm, f'{tag}_err': f.flux_err / upm,
                        f'{tag}_reff_pc': f.r_eff_arcsec / ARCSEC_PER_PC, f'{tag}_converged': bool(f.converged),
                        f'{tag}_chi2red': f.chi2_red, f'{tag}_neval': f.n_eval, f'{tag}_rfit': f.r_fit_arcsec,
                        f't_{tag}': f.time_s,
                        f'{tag}_centre_err_px': float(np.hypot(f.xc - x, f.yc - y))})
        rows.append(row)
    if keep:
        return rows, dict(cut_bd=cut_bd, geo=geo, fits=fits)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# plots
# ─────────────────────────────────────────────────────────────────────────────

def use_methods():
    """point mock_far_neighbours' plot functions at this test's methods"""
    mfn.METHODS[:] = METHODS
    mfn.STYLE.update(STYLE)


def plot_timing(tt, t, out_png):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                         'ytick.color': INK2, 'text.color': INK, 'axes.linewidth': 0.8})
    labels = {'native': 'native\n(centre+size)', 'instP': 'instP slit fit\n(+ native)', 'deger': 'Deger\n4 px',
              'f2d_fixed': '2D fixed', 'f2d_free': '2D free'}
    cols = {'native': '#eda100', 'instP': '#4a3aa7', 'deger': '#2a78d6', 'f2d_fixed': '#1baf7a',
            'f2d_free': '#eb6834'}
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(17, 4.8), gridspec_kw=dict(width_ratios=[1.1, 1, 1]))
    data = []
    for m, c in TIME_COLS.items():
        v = np.asarray(tt[c], float)
        if m == 'instP':
            v = v + np.asarray(tt['t_native'], float)      # instP needs native first
        data.append(v[np.isfinite(v)])
    bp = a1.boxplot(data, patch_artist=True, widths=0.6, showfliers=False)
    for patch, m in zip(bp['boxes'], TIME_COLS):
        patch.set_facecolor(cols[m]); patch.set_alpha(0.55)
    for med in bp['medians']:
        med.set_color(INK)
    for k, v in enumerate(data):
        a1.text(k + 1, np.percentile(v, 99) * 1.25 if v.size else 1, f'{np.median(v):.2g} s', ha='center',
                fontsize=8.5)
    a1.set_xticks(range(1, len(data) + 1)); a1.set_xticklabels([labels[m] for m in TIME_COLS], fontsize=8.5)
    a1.set_yscale('log')
    a1.set_ylabel('seconds per band (one source)')
    a1.set_title(f'(a) serial benchmark, {len(set(zip(tt["reff_pc"], tt["group"], tt["snr"], tt["real"])))} '
                 f'systems x {len(set(tt["band"]))} bands\n(instP includes the native step it needs)', fontsize=9.5)
    a1.grid(color=GRID, lw=0.6, which='both', axis='y')
    # (b) time vs band, (c) vs S/N, both from the benchmark
    bands = list(dict.fromkeys(tt['band']))
    for m in ('instP', 'f2d_fixed', 'f2d_free'):
        v = np.asarray(tt[TIME_COLS[m]], float) + (np.asarray(tt['t_native'], float) if m == 'instP' else 0)
        a2.plot(range(len(bands)), [np.median(v[tt['band'] == b]) for b in bands], 'o-', color=cols[m],
                label=labels[m].replace('\n', ' '))
        snrs = sorted(set(tt['snr']))
        a3.plot(snrs, [np.median(v[tt['snr'] == s]) for s in snrs], 'o-', color=cols[m],
                label=labels[m].replace('\n', ' '))
    a2.set_xticks(range(len(bands))); a2.set_xticklabels(bands, rotation=30, fontsize=8)
    a2.set_ylabel('median seconds per band'); a2.set_title('(b) per band (pixel scale, PSF size)', fontsize=9.5)
    a3.set_xscale('log'); a3.set_xlabel('S/N within 0.158"'); a3.set_title('(c) vs S/N', fontsize=9.5)
    for a in (a2, a3):
        a.grid(color=GRID, lw=0.6); a.legend(fontsize=8)
    if t is not None:
        n2 = np.median(np.asarray(t['f2d_fixed_neval'], float)), np.median(np.asarray(t['f2d_free_neval'], float))
        fig.text(0.01, -0.03, f'Model evaluations per band (full grid, median): 2D fixed {n2[0]:.0f}, 2D free {n2[1]:.0f};'
                 ' instP does ~40 per profile (grid of 31 + refinement) on the same rendered stamps.', fontsize=9)
    fig.suptitle('Time per method', x=0.01, ha='left', fontsize=12)
    plt.tight_layout()
    fig.savefig(out_png, dpi=130, facecolor='#fcfcfb', bbox_inches='tight')
    plt.close(fig)


def plot_residuals(ctx, tasks, out_png):
    """data, 2D-free model and residual (F555W) for a few systems with neighbours"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    picks = [tk for tk in tasks if tk['k'] == 0 and tk['snr'] in (20, 100)
             and tk['group'] in ('far6', 'far3_bright')]
    fig, axes = plt.subplots(len(picks), 3, figsize=(9, 3 * len(picks)))
    for i, tk in enumerate(picks):
        rows, det = run_system_2d(tk, ctx, keep=True)
        bd = det['cut_bd']['f555w']
        img, pix = bd['image'], bd['pixel_scale_arcsec']
        f_fix, f_free, _ = det['fits']['F555W']
        x, y = det['geo']['F555W']
        h = int(1.6 / pix)
        ix, iy = int(round(x)), int(round(y))
        ys, xs = slice(iy - h, iy + h + 1), slice(ix - h, ix + h + 1)
        model = bd['psf_model'].render(f_free.xc - xs.start, f_free.yc - ys.start, (2 * h + 1, 2 * h + 1),
                                       flux=f_free.flux, profile='eff', r_eff_arcsec=f_free.r_eff_arcsec,
                                       gamma=3.0, c=1.5) + f_free.background
        sub = img[ys, xs]
        lo, hi = np.percentile(sub, [2, 99.7])
        ext = [-h * pix, h * pix, -h * pix, h * pix]
        r555 = next(r for r in rows if r['band'] == 'F555W')
        for j, (im, ttl) in enumerate([(sub, 'data'), (model, '2D-free model'), (sub - model, 'data - model')]):
            ax = axes[i, j]
            if j < 2:
                ax.imshow(im, origin='lower', cmap='Greys', vmin=lo, vmax=hi, extent=ext)
            else:
                s = f_free.sigma_pix
                ax.imshow(im, origin='lower', cmap='RdBu_r', vmin=-5 * s, vmax=5 * s, extent=ext)
            ax.add_patch(plt.Circle(((f_free.xc - x) * pix, (f_free.yc - y) * pix), f_free.r_fit_arcsec,
                                    fill=False, color='#eda100', lw=1, ls='--'))
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(ttl + ('  (+-5 sigma)' if j == 2 else ''), fontsize=9)
        axes[i, 0].set_ylabel(f'R_eff {tk["reff"]:g} pc, {tk["group"]}\nS/N {tk["snr"]}', fontsize=8.5)
        axes[i, 2].text(0.02, 0.02, f'instP {r555["instP"]:.3f}\n2D fixed {r555["f2d_fixed"]:.3f}\n'
                        f'2D free {r555["f2d_free"]:.3f}\nR_eff 2D {r555["f2d_free_reff_pc"]:.1f} pc',
                        transform=axes[i, 2].transAxes, fontsize=7.5, va='bottom',
                        bbox=dict(fc='white', ec='none', alpha=0.85))
    fig.suptitle('F555W: neighbours inside the 2D fit disc (dashed) are left in, unmasked and unfitted',
                 x=0.01, ha='left', fontsize=10)
    plt.tight_layout()
    fig.savefig(out_png, dpi=110, facecolor='#fcfcfb')
    plt.close(fig)


def report(t, tt):
    print('\nrecovered / true: median [16-84%] over realizations and bands   (median reported error)')
    hst = mfn.HST
    for inst, sel_b in (('HST', np.isin(t['band'], hst)), ('NIRCam', ~np.isin(t['band'], hst))):
        print(f'\n=== {inst} ===')
        for reff in mfn.REFFS:
            for g in mfn.GROUPS:
                print(f'  R_eff {reff:g} pc, {g}')
                for m in METHODS:
                    line = f'    {m:12s}'
                    for snr in mfn.SNRS:
                        s = (t['reff_pc'] == reff) & (t['group'] == g) & (t['snr'] == snr) & sel_b
                        a, lo, hi = mfn.q(t[m][s])
                        e = np.nanmedian(np.asarray(t[f'{m}_err'][s], float))
                        line += f'  S/N{snr:<3d} {a:5.3f} [{lo:4.2f}-{hi:4.2f}] ({e:4.2f})'
                    print(line)
    for tag in ('f2d_fixed', 'f2d_free'):
        print(f'{tag}: converged {np.mean(t[f"{tag}_converged"]):.1%}, median chi2_red {np.nanmedian(t[f"{tag}_chi2red"]):.2f}, '
              f'median evals {np.nanmedian(t[f"{tag}_neval"]):.0f}')
    if tt is not None:
        print('\ntiming (serial benchmark), seconds per band per source: median [16-84%]')
        for m, c in TIME_COLS.items():
            v = np.asarray(tt[c], float) + (np.asarray(tt['t_native'], float) if m == 'instP' else 0)
            print(f'  {m:10s} {np.median(v):7.3f} [{np.percentile(v, 16):.3f}-{np.percentile(v, 84):.3f}]'
                  + ('   (includes native)' if m == 'instP' else ''))


def read_table(path):
    """FITS round-trips text columns as bytes; np.isin against str then matches nothing"""
    if not os.path.exists(path):
        return None
    t = Table.read(path)
    for c in t.colnames:
        if t[c].dtype.kind == 'S':
            t[c] = np.char.decode(np.asarray(t[c]), 'ascii')
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n-real', type=int, default=20)
    ap.add_argument('--workers', type=int, default=40)
    ap.add_argument('--seed', type=int, default=20261015)          # = mock_far_neighbours default
    ap.add_argument('--max-tasks', type=int, default=None)
    ap.add_argument('--timing', action='store_true', help='serial timing benchmark only')
    ap.add_argument('--n-timing', type=int, default=50)
    ap.add_argument('--plot', action='store_true', help='re-plot from the saved tables')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    out_tab, out_time = os.path.join(OUT, 'mock_2d_far.fits'), os.path.join(OUT, 'mock_2d_timing.fits')
    t0 = time.time()
    mfn.OUT = OUT                                   # build_context writes its setup cache here
    if a.timing:
        ctx = mfn.build_context()
        _init(ctx)
        tasks = mfn.make_tasks(a.n_real, a.seed)
        tasks = tasks[::max(len(tasks) // a.n_timing, 1)][:a.n_timing]
        rows = []
        for i, tk in enumerate(tasks):
            rows.extend(run_system_2d(tk, ctx))
            print(f'  timing {i + 1}/{len(tasks)}', flush=True)
        Table(rows=rows).write(out_time, overwrite=True)
    elif not a.plot:
        ctx = mfn.build_context()
        tasks = mfn.make_tasks(a.n_real, a.seed)
        if a.max_tasks:
            tasks = tasks[::max(len(tasks) // a.max_tasks, 1)][:a.max_tasks]
        print(f'{len(tasks)} systems x {len(ctx["bands"])} bands, {a.workers} workers', flush=True)
        rows = []
        with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context('fork'),
                                 initializer=_init, initargs=(ctx,)) as ex:
            for i, r in enumerate(ex.map(run_system_2d, tasks, chunksize=1)):
                rows.extend(r)
                if (i + 1) % 50 == 0:
                    print(f'  {i + 1}/{len(tasks)}  {time.time() - t0:.0f} s', flush=True)
        t = Table(rows=rows)
        for c in t.colnames:
            if t[c].dtype.kind == 'f':
                t[c].format = '.5g'
        t.write(out_tab, overwrite=True)
        _init(ctx)
        plot_residuals(ctx, mfn.make_tasks(1, a.seed), os.path.join(OUT, 'mock_2d_residuals.png'))
    t = read_table(out_tab)
    tt = read_table(out_time)
    use_methods()
    if t is not None:
        nircam = sorted({str(b) for b in t['band']} - set(mfn.HST), key=lambda b: int(b[1:-1]))
        mfn.plot_set(t, mfn.HST, 'HST (5 bands)', os.path.join(OUT, 'mock_2d_far_HST.png'))
        mfn.plot_set(t, nircam, f'NIRCam ({", ".join(nircam)})', os.path.join(OUT, 'mock_2d_far_NIRCam.png'))
        mfn.plot_pull(t, os.path.join(OUT, 'mock_2d_pull.png'))
    if tt is not None:
        plot_timing(tt, t, os.path.join(OUT, 'mock_2d_timing.png'))
    if t is not None:
        report(t, tt)
    print(f'\ndone in {time.time() - t0:.0f} s -> {OUT}')


if __name__ == '__main__':
    main()
