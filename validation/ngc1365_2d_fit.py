"""
NGC1365: 2D model fit (fit2d.py) of the 31 real clusters vs instP and the
PHANGS catalog, and the time each method takes.

Per cluster and band (HST + NIRCam, only where instP has its own fit, i.e.
real emission), on the real images:
  f2d_fixed  PSF (x) profile + constant, centre fixed at instP's centre
  f2d_free   same with the centre free (|dx|, |dy| <= 1.5 px)
  f2dG_fixed, f2dG_free   the same two with a Gaussian source profile (2D instG)
profile (f2d_*) = instP's per-class choice (EFF gamma=3 for MAP/YCL, King c=1.5 for
OGCC), galaxy-matched instrument PSF, fit disc max(1", 4 FWHM) (grown to 5 R_eff,
<= 3", if needed), neighbours neither masked nor fitted, Milky Way dereddened
like the pipeline (corr_flux). Errors: least-squares covariance x sqrt(chi2_red).
instP (and native, which it needs) are re-run here with timing; their fluxes
are checked against the pipeline cache.

Writes <validation_output_dir>/output_ngc1365_2d/:
  NGC1365_2d_fit.fits/.csv           one row per cluster (all bands as columns)
  NGC1365_2d_vs_instP.png            2D fixed / free vs instP, all bands, errors on both axes
  NGC1365_f2d_fixed_vs_catalog.png, NGC1365_f2d_free_vs_catalog.png,
  NGC1365_instP_vs_catalog.png       same layout as ngc1365_catalog_comparison.py
  NGC1365_2d_residuals.png           data / model / residual where 2D and instP differ most
  NGC1365_2d_timing.png              seconds per band per method
"""
import logging
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)
from _paths import OUT_ROOT  # noqa: E402

import astropy.units as u  # noqa: E402
from astropy.coordinates import SkyCoord  # noqa: E402
from astropy.table import Table  # noqa: E402

import injection_test as it  # noqa: E402
import ngc1365_catalog_comparison as ncc  # noqa: E402
from native_phot_aux_func import (measure_native_photometry_one_source,  # noqa: E402
                                  measure_instrument_photometry_one_source, instrument_slit_length)
from phot_aux_func import corr_flux  # noqa: E402
from fit2d import fit_2d_instrument  # noqa: E402

OUT = os.path.join(OUT_ROOT, 'output_ngc1365_2d')
HST = ['F275W', 'F336W', 'F438W', 'F555W', 'F814W']
NIRCAM = ['F200W', 'F300M', 'F335M', 'F360M']
CAT = {'F275W': 'NUV', 'F336W': 'U', 'F438W': 'B', 'F555W': 'V', 'F814W': 'I'}
MAX_REFF_PC = 20.0
INK, INK2, GRID = ncc.INK, ncc.INK2, ncc.GRID
MCOL = {'instP': '#4a3aa7', 'f2d_fixed': '#1baf7a', 'f2d_free': '#eb6834', 'native': '#eda100',
        'f2dG_fixed': '#2a78d6', 'f2dG_free': '#9b59b6'}
# 2D variants: tag -> (free centre, source profile; None = instP's per-class profile)
M2D = {'f2d_fixed': (False, None), 'f2d_free': (True, None),
       'f2dG_fixed': (False, 'gaussian'), 'f2dG_free': (True, 'gaussian')}
M2D_LABEL = {'f2d_fixed': '2D, centre fixed', 'f2d_free': '2D, centre free',
             'f2dG_fixed': '2D Gauss, centre fixed', 'f2dG_free': '2D Gauss, centre free'}
GAUSS = dict(profile='gaussian', gamma=3.0, king_c=1.5)


def spec_from_tag(tag):
    tag = str(tag).strip()
    if tag.startswith('king'):
        return dict(profile='king', gamma=3.0, king_c=float(tag[4:]))
    return dict(profile='eff', gamma=float(tag[3:]) if tag.startswith('eff') else 3.0, king_c=1.5)


def measure():
    p, gal = it.setup('NGC1365-1.15asec')
    for name in ('analysis', 'native_phot_aux_func', 'instrument_psf'):
        logging.getLogger(name).setLevel(logging.ERROR)
    tbl = p.photometry_table
    bands = [b for b in HST + NIRCAM if b.lower() in p.native_band_calib and b.lower() in p.instrument_psf]
    bd = {b.lower(): dict(p.native_band_calib[b.lower()], psf_model=p.instrument_psf[b.lower()]) for b in bands}
    dist = float(np.nanmedian(np.asarray(tbl['PHANGS_DMPC'], float)))
    arcsec_per_pc = 206265.0 / (dist * 1e6)
    rows, times = [], []
    for i in range(len(tbl)):
        ra, dec = float(tbl['PHANGS_RA'][i]), float(tbl['PHANGS_DEC'][i])
        tag = str(tbl['profile_instP'][i]).strip()
        spec = spec_from_tag(tag)
        t0 = time.perf_counter()
        nat = measure_native_photometry_one_source(ra, dec, bd, eb_v=p.eb_v)
        t_nat = time.perf_counter() - t0
        nr = {bl: dict(aperture_source=r.aperture_source, centroid_ra=r.centroid_ra, centroid_dec=r.centroid_dec)
              for bl, r in nat.items()}
        t0 = time.perf_counter()
        ins = measure_instrument_photometry_one_source(ra, dec, nr, bd, spec, distance_mpc=dist, eb_v=p.eb_v)
        t_ins = time.perf_counter() - t0
        row = dict(ID_PHANGS_CLUSTER=int(tbl['ID_PHANGS_CLUSTER'][i]),
                   CLASS=str(tbl['PHANGS_CC_CLASS_ML'][i]).strip(), profile=tag, RA=ra, DEC=dec)
        for b in bands:
            bl, B = b.lower(), bd[b.lower()]
            r = ins[bl]['instP']
            mw = float(corr_flux(1.0, B['r_value'], p.eb_v))
            row.update({f'instP_{b}': r.flux_mJy, f'instP_{b}_err': r.flux_err_mJy, f'instP_reff_pc_{b}': r.r_eff_pc,
                        f'instP_size_source_{b}': str(r.size_source),
                        f'instP_cache_{b}': float(tbl[f'flux_instP_{b}'][i]),
                        f'native_{b}': nat[bl].flux_mJy, f'native_{b}_err': nat[bl].flux_err_mJy})
            if b in CAT:
                row.update({f'catalog_{b}': float(tbl[f'PHANGS_{CAT[b]}_mJy'][i]),
                            f'catalog_{b}_err': float(tbl[f'PHANGS_{CAT[b]}_mJy_ERR'][i])})
            for tagm in M2D:
                row.update({f'{tagm}_{b}': np.nan, f'{tagm}_{b}_err': np.nan, f'{tagm}_reff_pc_{b}': np.nan,
                            f'{tagm}_chi2red_{b}': np.nan, f'{tagm}_shift_px_{b}': np.nan,
                            f'{tagm}_converged_{b}': False})
            if str(r.size_source) != 'own' or not np.isfinite(r.centroid_ra):
                continue
            xs, ys = (float(v) for v in B['wcs'].world_to_pixel(SkyCoord(r.centroid_ra * u.deg, r.centroid_dec * u.deg)))
            kw = dict(r_fit_arcsec=instrument_slit_length(B, 1.0), max_r_eff_arcsec=MAX_REFF_PC * arcsec_per_pc)
            for tagm, (free, prof) in M2D.items():
                f = fit_2d_instrument(B['image'], xs, ys, B['psf_model'], GAUSS if prof == 'gaussian' else spec,
                                      free_center=free, **kw)
                row.update({f'{tagm}_{b}': it.to_mjy(f.flux, B) * mw, f'{tagm}_{b}_err': abs(it.to_mjy(f.flux_err, B)) * mw,
                            f'{tagm}_reff_pc_{b}': f.r_eff_arcsec / arcsec_per_pc, f'{tagm}_chi2red_{b}': f.chi2_red,
                            f'{tagm}_shift_px_{b}': f.shift_px, f'{tagm}_converged_{b}': bool(f.converged)})
                times.append(dict(ID=row['ID_PHANGS_CLUSTER'], band=b, method=tagm, t=f.time_s, n_eval=f.n_eval))
            times.append(dict(ID=row['ID_PHANGS_CLUSTER'], band=b, method='native', t=t_nat / len(bands), n_eval=0))
            times.append(dict(ID=row['ID_PHANGS_CLUSTER'], band=b, method='instP', t=(t_nat + t_ins) / len(bands),
                              n_eval=0))
        rows.append(row)
        print(f'  {i + 1}/{len(tbl)}  ID {row["ID_PHANGS_CLUSTER"]}  native {t_nat:.1f} s, instP {t_ins:.1f} s', flush=True)
    t = Table(rows=rows)
    for c in t.colnames:
        if t[c].dtype.kind == 'f':
            t[c].format = '.5g'
    t.meta.update(GALAXY=gal, EBV=float(p.eb_v), DIST=dist)
    tt = Table(rows=times)
    t.write(os.path.join(OUT, 'NGC1365_2d_fit.fits'), overwrite=True)
    t.write(os.path.join(OUT, 'NGC1365_2d_fit.csv'), overwrite=True)
    tt.write(os.path.join(OUT, 'NGC1365_2d_timing.fits'), overwrite=True)
    return t, tt, p, bd


def plot_vs_instP(t, out_png):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    bands = [b for b in HST + NIRCAM if f'f2d_fixed_{b}' in t.colnames]
    fig, axes = plt.subplots(2, len(bands), figsize=(2.6 * len(bands), 5.6), sharex='col',
                             gridspec_kw=dict(height_ratios=[2.2, 1.2]))
    for j, b in enumerate(bands):
        ax, axr = axes[0, j], axes[1, j]
        x, xe = np.asarray(t[f'instP_{b}'], float), np.abs(np.asarray(t[f'instP_{b}_err'], float))
        allv = [x]
        for k, m in enumerate(M2D):
            y, ye = np.asarray(t[f'{m}_{b}'], float), np.abs(np.asarray(t[f'{m}_{b}_err'], float))
            ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
            allv.append(y[ok])
            kw = dict(fmt='oDsv'[k], ms=5, color=MCOL[m], mec='white', mew=0.6, elinewidth=1, capsize=1.5,
                      alpha=0.9, label=M2D_LABEL[m])
            ax.errorbar(x[ok], y[ok], xerr=xe[ok], yerr=ye[ok], **kw)
            r = y[ok] / x[ok]
            axr.errorbar(x[ok] * (1 + 0.04 * (k - 1.5)), r, yerr=r * np.hypot(ye[ok] / y[ok], xe[ok] / x[ok]), **kw)
            if ok.any():
                axr.axhline(np.median(r), color=MCOL[m], lw=1.2)
                ax.text(0.04, 0.96 - 0.08 * k, f'{m[3:]}: {np.median(r):.3f} '
                        f'[{np.percentile(r, 16):.2f}-{np.percentile(r, 84):.2f}] N={ok.sum()}',
                        transform=ax.transAxes, va='top', fontsize=7, color=MCOL[m])
        v = np.concatenate([a[np.isfinite(a) & (a > 0)] for a in allv])
        lo, hi = v.min() / 1.5, v.max() * 1.5
        ax.plot([lo, hi], [lo, hi], '--', color=INK2, lw=1)
        axr.axhline(1, color=INK2, lw=1, ls='--')
        ax.set_xscale('log'); ax.set_yscale('log'); axr.set_xscale('log')
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); axr.set_ylim(0.5, 1.5)
        ax.set_title(b, fontsize=10)
        axr.set_xlabel('instP [mJy]', fontsize=8)
        for a in (ax, axr):
            a.grid(color=GRID, lw=0.5); a.tick_params(labelsize=7)
        if j == 0:
            ax.set_ylabel('2D fit [mJy]'); axr.set_ylabel('2D / instP')
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc='upper right', ncol=4, frameon=False, fontsize=9)
    fig.suptitle('NGC1365: 2D model fit vs instP (1D slits + aperture), same centre and PSF '
                 '(1$\\sigma$ errors; ratio errors in quadrature)', x=0.01, ha='left', fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png, dpi=130, facecolor='#fcfcfb')
    plt.close(fig)


def plot_residuals(t, bd, out_png, n_show=6):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    b = 'F555W'
    B = bd[b.lower()]
    pix = B['pixel_scale_arcsec']
    dev = np.abs(np.asarray(t[f'f2d_free_{b}'], float) / np.asarray(t[f'instP_{b}'], float) - 1)
    order = [i for i in np.argsort(-np.nan_to_num(dev, nan=-1)) if np.isfinite(dev[i])][:n_show]
    dist = t.meta['DIST']
    fig, axes = plt.subplots(len(order), 3, figsize=(9, 3 * len(order)))
    for row_i, i in enumerate(order):
        spec = spec_from_tag(t['profile'][i])
        x0, y0 = (float(v) for v in B['wcs'].world_to_pixel(SkyCoord(t['RA'][i] * u.deg, t['DEC'][i] * u.deg)))
        r_eff = t[f'f2d_free_reff_pc_{b}'][i] * 206265.0 / (dist * 1e6)
        # re-fit to get the centre/background (cheap enough for a few)
        f = fit_2d_instrument(B['image'], x0, y0, B['psf_model'], spec, free_center=True,
                              r_fit_arcsec=instrument_slit_length(B, 1.0),
                              max_r_eff_arcsec=MAX_REFF_PC * 206265.0 / (dist * 1e6))
        h = int(1.6 / pix)
        ix, iy = int(round(f.xc)), int(round(f.yc))
        sub = np.asarray(B['image'][iy - h:iy + h + 1, ix - h:ix + h + 1], float)
        model = B['psf_model'].render(f.xc - (ix - h), f.yc - (iy - h), sub.shape, flux=f.flux,
                                      profile=spec['profile'], r_eff_arcsec=f.r_eff_arcsec, gamma=spec['gamma'],
                                      c=spec['king_c']) + f.background
        lo, hi = np.nanpercentile(sub, [2, 99.7])
        ext = [-h * pix, h * pix, -h * pix, h * pix]
        for j, (im, ttl) in enumerate([(sub, 'data'), (model, '2D-free model'), (sub - model, 'data - model')]):
            ax = axes[row_i, j]
            if j < 2:
                ax.imshow(im, origin='lower', cmap='Greys', vmin=lo, vmax=hi, extent=ext)
            else:
                ax.imshow(im, origin='lower', cmap='RdBu_r', vmin=-5 * f.sigma_pix, vmax=5 * f.sigma_pix, extent=ext)
            ax.add_patch(plt.Circle((0, 0), f.r_fit_arcsec, fill=False, color='#eda100', lw=1, ls='--'))
            ax.set_xticks([]); ax.set_yticks([])
            if row_i == 0:
                ax.set_title(ttl + ('  (+-5 sigma)' if j == 2 else ''), fontsize=9)
        axes[row_i, 0].set_ylabel(f'ID {t["ID_PHANGS_CLUSTER"][i]} ({t["CLASS"][i]})', fontsize=8.5)
        axes[row_i, 2].text(0.02, 0.02, f'instP {t[f"instP_{b}"][i]:.4g}\n2D fixed {t[f"f2d_fixed_{b}"][i]:.4g}\n'
                            f'2D free {t[f"f2d_free_{b}"][i]:.4g} mJy\nR_eff instP/2D {t[f"instP_reff_pc_{b}"][i]:.1f}/'
                            f'{t[f"f2d_free_reff_pc_{b}"][i]:.1f} pc\nchi2_red {t[f"f2d_free_chi2red_{b}"][i]:.1f}',
                            transform=axes[row_i, 2].transAxes, fontsize=7, va='bottom',
                            bbox=dict(fc='white', ec='none', alpha=0.85))
    fig.suptitle(f'NGC1365 {b}: clusters where 2D-free and instP differ most (dashed = fit disc)', x=0.01,
                 ha='left', fontsize=10)
    plt.tight_layout()
    fig.savefig(out_png, dpi=110, facecolor='#fcfcfb')
    plt.close(fig)


def plot_timing(tt, out_png):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ms = ['native', 'instP'] + list(M2D)
    lbl = {'native': 'native', 'instP': 'instP\n(incl. native)', 'f2d_fixed': '2D fixed', 'f2d_free': '2D free',
           'f2dG_fixed': '2D Gauss\nfixed', 'f2dG_free': '2D Gauss\nfree'}
    fig, ax = plt.subplots(figsize=(9, 4.2))
    data = [np.asarray(tt['t'][tt['method'] == m], float) for m in ms]
    bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False)
    for patch, m in zip(bp['boxes'], ms):
        patch.set_facecolor(MCOL[m]); patch.set_alpha(0.55)
    for med in bp['medians']:
        med.set_color(INK)
    for k, v in enumerate(data):
        ax.text(k + 1, np.percentile(v, 95) * 1.2, f'{np.median(v):.2g} s', ha='center', fontsize=9)
    ax.set_xticks(range(1, len(ms) + 1)); ax.set_xticklabels([lbl[m] for m in ms])
    ax.set_yscale('log'); ax.set_ylabel('seconds per band (one cluster)')
    ax.grid(color=GRID, lw=0.6, axis='y', which='both')
    ax.set_title('NGC1365, real images, 1 process: time per method', fontsize=10)
    plt.tight_layout()
    fig.savefig(out_png, dpi=130, facecolor='#fcfcfb')
    plt.close(fig)


def report(t, tt):
    print('\nratio to instP: median [16-84%] (N), own-fit bands only')
    for b in HST + NIRCAM:
        if f'f2d_fixed_{b}' not in t.colnames:
            continue
        x = np.asarray(t[f'instP_{b}'], float)
        line = f'  {b}'
        for m in M2D:
            r = np.asarray(t[f'{m}_{b}'], float) / x
            r = r[np.isfinite(r)]
            line += f'   {m} {np.median(r):.3f} [{np.percentile(r, 16):.2f}-{np.percentile(r, 84):.2f}] ({r.size})'
        rr = np.asarray(t[f'f2d_free_reff_pc_{b}'], float) / np.asarray(t[f'instP_reff_pc_{b}'], float)
        line += f'   R_eff 2Dfree/instP {np.nanmedian(rr):.2f}'
        dc = np.nanmax(np.abs(np.asarray(t[f'instP_{b}'], float) / np.asarray(t[f'instP_cache_{b}'], float) - 1))
        line += f'   |instP rerun/cache - 1| max {dc:.1e}'
        print(line)
    print('\nratio to the catalog (HST): median [16-84%]')
    for b in HST:
        line = f'  {b}'
        for m in ('instP', *M2D):
            r = np.asarray(t[f'{m}_{b}'], float) / np.asarray(t[f'catalog_{b}'], float)
            r = r[np.isfinite(r)]
            line += f'   {m} {np.median(r):.3f} [{np.percentile(r, 16):.2f}-{np.percentile(r, 84):.2f}]'
        print(line)
    print('\nseconds per band per cluster: median [16-84%]')
    for m in ('native', 'instP', *M2D):
        v = np.asarray(tt['t'][tt['method'] == m], float)
        print(f'  {m:10s} {np.median(v):.3f} [{np.percentile(v, 16):.3f}-{np.percentile(v, 84):.3f}]')


def main():
    os.makedirs(OUT, exist_ok=True)
    t, tt, p, bd = measure()
    plot_vs_instP(t, os.path.join(OUT, 'NGC1365_2d_vs_instP.png'))
    for m, ttl in (('instP', 'instP (1D slits + aperture)'), ('f2d_fixed', '2D fit, centre fixed'),
                   ('f2d_free', '2D fit, centre free'), ('f2dG_fixed', '2D Gaussian fit, centre fixed'),
                   ('f2dG_free', '2D Gaussian fit, centre free')):
        ncc.plot(t, m, f'NGC1365: {ttl} vs PHANGS-HST catalog', os.path.join(OUT, f'NGC1365_{m}_vs_catalog.png'))
    plot_residuals(t, bd, os.path.join(OUT, 'NGC1365_2d_residuals.png'))
    plot_timing(tt, os.path.join(OUT, 'NGC1365_2d_timing.png'))
    report(t, tt)


if __name__ == '__main__':
    main()
