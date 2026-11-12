"""
Mock field with neighbours, run through the real pipeline (run_native_photometry
with the instrument fit, plus its slit-profile and SED diagnostic plots).

One mock image per HST and NIRCam band (no MIRI), on the band's real pixel grid
(a WCS cutout of NGC1365's image) and with its galaxy-matched instrument PSF:

  - a 5 x 5 grid of "systems" 6" apart; each system is one cluster with a flat
    SED of exactly 1 mJy total in every band (EFF gamma=3, R_eff 10 pc at
    19.57 Mpc), class YCL so the pipeline's instP uses EFF gamma=3
  - neighbours: Gaussians (intrinsic FWHM 0-0.15") with 0.1-0.9 mJy, also flat
    SEDs, at the same sky positions in every band. Five groups of five systems:
        clean      no neighbours (control)
        far3       3 neighbours at 0.6-1.2"  (in or near the background annuli)
        close3     3 neighbours at 0.2-0.5"  (in or near the aperture)
        mixed6     6 neighbours at 0.15-1.2"
        crowded10  10 neighbours at 0.15-1.2"
  - flat sky pedestal + Gaussian noise; S/N 50 for the cluster's flux within
    0.158" (the catalog's 4 HST px aperture), the same in every band

Methods: native, instG, instP (pipeline) and, measured here,
  deger_ideal  4 px (0.158") flux, sky 0.277-0.317", times the exact aperture
               correction of this cluster model in this band (the best case of
               the catalog recipe: a correction perfect for an isolated cluster)
  growth20     light within 0.792" (20 HST px), sky 0.832-0.911"
The deger_ideal values also go into the PHANGS_<band>_mJy "catalog" columns so
the pipeline's SED panel draws them.

Truth: 1 mJy in every band, so every SED should be a flat line at 1.

Usage:  python mock_neighbours_test.py [--snr 50] [--workers 25]
Writes <validation_output_dir>/output_mock_neighbours/mock_images/ (MOCK_NEIGH_<band>.fits:
noisy image + CLUSTERS and NEIGHBOURS noise-free layers; truth tables
MOCK_NEIGH_truth_clusters/_truth_neighbours/_bands.fits),
<validation_output_dir>/output_mock_neighbours/MOCK_NEIGH/ (pipeline caches,
native_diagnostics/, native_slit_profiles/, instrument comparison) and
mock_neighbours_summary.png, mock_neighbours_fields.png, mock_neighbours.fits.
"""
import argparse
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
from astropy.nddata import Cutout2D  # noqa: E402
from astropy.io import fits  # noqa: E402
from astropy.table import Table  # noqa: E402
from astropy.wcs import WCS  # noqa: E402

import run_full_pipeline as rfp  # noqa: E402
from analysis import Photometry  # noqa: E402
from instrument_psf import instrument_psf_for_image  # noqa: E402
from phot_aux_func import jwst_pixel_sum_to_Jy  # noqa: E402
from injection_test import _ap, _sky, R4, R7, R8, R20, R21, R23  # noqa: E402

TOKEN = 'NGC1365-1.15asec'
GAL = 'MOCK_NEIGH'
DIST = 19.57
REFF_PC = 10.0
CLUSTER = dict(profile='eff', gamma=3.0, c=1.5)
GRID, SPACING, FIELD = 5, 6.0, 36.0          # systems per side, arcsec, arcsec
STAMP_ARCSEC = 3.0
GROUPS = [('clean', 0, None), ('far3', 3, (0.6, 1.2)), ('close3', 3, (0.2, 0.5)),
          ('mixed6', 6, (0.15, 1.2)), ('crowded10', 10, (0.15, 1.2))]
METHODS = ['native', 'instG', 'instP', 'deger_ideal', 'growth20']
CAT_NAME = {'F275W': 'NUV', 'F336W': 'U', 'F438W': 'B', 'F555W': 'V', 'F814W': 'I'}


def load_real():
    os.chdir(_ROOT)
    cfg = rfp.load_config('config.ini')
    gal = TOKEN.split('-')[0]
    p = Photometry([TOKEN, gal], base_dir=cfg['output_dir'], psf_filepath=cfg['psf_filepath'],
                   psf_source=cfg.get('psf_source', 'theoretical'))
    p.load_images(rfp.find_hst_band_paths(cfg['hst_root'], gal),
                  rfp.find_jwst_band_paths(cfg['jwst_root'], gal.lower()), muse_path=None)
    p.load_or_measure_psf()
    p.measure_native_psf_shape()
    return p


def unit_per_mjy(band, header):
    """image units (summed over pixels) per mJy"""
    if band in p_jwst_bands:
        return 1.0 / (jwst_pixel_sum_to_Jy(1.0, header) * 1e3)
    return 1e-3                                       # HST images are in Jy


p_jwst_bands = set()


def add_stamp(img, model, x, y, pix, **kw):
    hs = int(np.ceil(STAMP_ARCSEC / pix))
    ix, iy = int(np.round(x)), int(np.round(y))
    stamp = model.render(hs + x - ix, hs + y - iy, (2 * hs + 1, 2 * hs + 1), **kw)
    ny, nx = img.shape
    y0, x0 = iy - hs, ix - hs
    ys = slice(max(y0, 0), min(y0 + 2 * hs + 1, ny))
    xs = slice(max(x0, 0), min(x0 + 2 * hs + 1, nx))
    img[ys, xs] += stamp[ys.start - y0:ys.stop - y0, xs.start - x0:xs.stop - x0]


def make_systems(center, rng):
    """cluster positions (grid) and their neighbours (sky offsets, fluxes, sizes)"""
    systems = []
    k = 0
    for gname, n_nb, (rlo, rhi) in [(g[0], g[1], g[2] or (0, 0)) for g in GROUPS]:
        for _ in range(GRID):
            i, j = divmod(k, GRID)
            dx, dy = (j - (GRID - 1) / 2) * SPACING, (i - (GRID - 1) / 2) * SPACING
            pos = center.spherical_offsets_by(dx * u.arcsec, dy * u.arcsec)
            nbs = []
            for _ in range(n_nb):
                r, th = rng.uniform(rlo, rhi), rng.uniform(0, 2 * np.pi)
                nbs.append(dict(pos=pos.spherical_offsets_by(r * np.cos(th) * u.arcsec,
                                                             r * np.sin(th) * u.arcsec),
                                r=r, flux=rng.uniform(0.1, 0.9), fwhm=rng.uniform(0.0, 0.15)))
            systems.append(dict(idx=k, group=gname, pos=pos, neighbours=nbs))
            k += 1
    return systems


def build_mock_band(band, image, header, wcs, fwhm, center, systems, snr, rng):
    pix = wcs.proj_plane_pixel_scales()[0].to('arcsec').value
    n = int(np.ceil(FIELD / pix)) | 1
    cut = Cutout2D(image, center, (n, n), wcs=wcs, mode='partial', fill_value=0.0, copy=False)
    model, note = instrument_psf_for_image(band, header, pix, target_fwhm_arcsec=fwhm)
    upm = unit_per_mjy(band, header)
    reff_as = REFF_PC * 206265.0 / (DIST * 1e6)

    # noise level from an isolated, noise-free cluster: S/N `snr` within R4
    hs = int(np.ceil(STAMP_ARCSEC / pix))
    one = model.render(hs, hs, (2 * hs + 1, 2 * hs + 1), flux=upm, r_eff_arcsec=reff_as, **CLUSTER)
    f4_clean = _ap(one, hs, hs, R4 / pix, _sky(one, hs, hs, R7 / pix, R8 / pix))
    f4_true = _ap(one, hs, hs, R4 / pix, 0.0)
    ac_ideal = upm / f4_clean                     # deger_ideal: f4 (sky 7-8 px) * ac -> 1 mJy
    sigma = f4_true / (snr * np.sqrt(np.pi * (R4 / pix) ** 2))

    clusters = np.zeros((n, n))
    neighbours = np.zeros((n, n))
    for s in systems:
        x, y = (float(v) for v in s['pos'].to_pixel(cut.wcs))
        add_stamp(clusters, model, x, y, pix, flux=upm, r_eff_arcsec=reff_as, **CLUSTER)
        for nb in s['neighbours']:
            xn, yn = (float(v) for v in nb['pos'].to_pixel(cut.wcs))
            add_stamp(neighbours, model, xn, yn, pix, flux=nb['flux'] * upm, intrinsic_fwhm_arcsec=nb['fwhm'])
    pedestal = 10.0 * sigma
    img = clusters + neighbours + pedestal + rng.normal(0.0, sigma, (n, n))
    info = dict(pix=pix, upm=upm, sigma=sigma, pedestal=pedestal, ac_ideal=ac_ideal, psf_note=note,
                rendered_frac=float(one.sum() / upm), psf_name=getattr(model, 'name', ''),
                psf_extra_fwhm=float(getattr(model, 'extra_fwhm_arcsec', np.nan)))
    return img, cut.wcs, info, clusters, neighbours


def save_mock_band(path, band, img, clusters, neighbours, header, cwcs, info, snr):
    """noisy mock (primary) + noise-free cluster-only and neighbour-only layers, real header
    units/instrument keywords with the cutout's WCS"""
    hdr = header.copy()
    for k in list(WCS(header).to_header(relax=True).keys()) + [
            'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2', 'PC1_1', 'PC1_2', 'PC2_1', 'PC2_2', 'NAXIS1', 'NAXIS2']:
        while k in hdr:
            del hdr[k]
    hdr.update(cwcs.to_header(relax=True))
    hdr['MOCK'] = (True, 'mock_neighbours_test.py')
    hdr['MOCKBAND'] = band
    hdr['MOCKSNR'] = (snr, 'cluster S/N within 0.158" (4 HST px)')
    hdr['MOCKSIG'] = (info['sigma'], 'Gaussian noise per pixel, image units')
    hdr['MOCKPED'] = (info['pedestal'], 'flat sky pedestal, image units')
    hdr['MOCKUPM'] = (info['upm'], 'image units (pixel sum) per mJy')
    hdr['MOCKREFF'] = (REFF_PC, 'cluster R_eff [pc], EFF gamma=3')
    hdr['MOCKDIST'] = (DIST, 'Mpc')
    hdr['MOCKPSF'] = (str(info['psf_name'])[:60], 'instrument PSF model')
    hdr['MOCKBLUR'] = (info['psf_extra_fwhm'], 'extra PSF blur FWHM [arcsec]')
    hdr['MOCKREND'] = (info['rendered_frac'], 'fraction of 1 mJy rendered within the 3" stamp')
    whdr = cwcs.to_header(relax=True)
    fits.HDUList([fits.PrimaryHDU(img.astype(np.float32), hdr),
                  fits.ImageHDU(clusters.astype(np.float32), whdr, name='CLUSTERS'),
                  fits.ImageHDU(neighbours.astype(np.float32), whdr, name='NEIGHBOURS')]
                 ).writeto(path, overwrite=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--snr', type=float, default=50.0)
    ap.add_argument('--workers', type=int, default=25)
    ap.add_argument('--seed', type=int, default=20261008)
    args = ap.parse_args()
    t0 = time.time()
    for name in ('analysis', 'native_phot_aux_func'):
        logging.getLogger(name).setLevel(logging.WARNING)

    p = load_real()
    out_base = os.path.join(OUT_ROOT, 'output_mock_neighbours')
    bands = {}
    for b, v in rfp.FILTER_WAVELENGTHS.items():
        bl = b.lower()
        if bl in p.hst_images or (bl in p.jwst_images and int(b[1:-1]) < 500):
            bands[b] = v
    p_jwst_bands.update(b for b in bands if b.lower() in p.jwst_images)
    print('bands:', list(bands), flush=True)

    _, _, w5 = p.hst_images['f555w']
    ny5, nx5 = p.hst_images['f555w'][0].shape
    center = w5.pixel_to_world(nx5 / 2, ny5 / 2)
    rng = np.random.default_rng(args.seed)
    systems = make_systems(center, rng)

    # ── mock images replace the real ones (and are saved) ────────────────
    img_dir = os.path.join(out_base, 'mock_images')
    os.makedirs(img_dir, exist_ok=True)
    infos = {}
    for b in bands:
        bl = b.lower()
        store = p.hst_images if bl in p.hst_images else p.jwst_images
        image, header, wcs = store[bl]
        fwhm = (p.psf_hst if store is p.hst_images else p.psf_jwst).get(bl)
        img, cwcs, info, clusters, neighbours = build_mock_band(b, image, header, wcs, fwhm, center,
                                                                systems, args.snr, rng)
        save_mock_band(os.path.join(img_dir, f'{GAL}_{b}.fits'), b, img, clusters, neighbours,
                       header, cwcs, info, args.snr)
        store[bl] = (img, header, cwcs)
        infos[b] = info
        print(f'  {b}: {img.shape[0]} px at {info["pix"]:.4f}"/px, sigma={info["sigma"]:.3g}, '
              f'rendered {info["rendered_frac"]:.3f} of the total, deger_ideal AC x{info["ac_ideal"]:.3f}'
              f'{"  (" + info["psf_note"] + ")" if info["psf_note"] else ""}', flush=True)
    # truth tables: clusters, neighbours, per-band noise/units
    Table(dict(idx=[s['idx'] for s in systems], group=[s['group'] for s in systems],
               ra=[s['pos'].ra.deg for s in systems], dec=[s['pos'].dec.deg for s in systems],
               flux_mJy=np.ones(len(systems)), profile=['eff'] * len(systems),
               gamma=np.full(len(systems), 3.0), r_eff_pc=np.full(len(systems), REFF_PC),
               n_neighbours=[len(s['neighbours']) for s in systems])
          ).write(os.path.join(img_dir, f'{GAL}_truth_clusters.fits'), overwrite=True)
    nb_rows = [dict(parent=s['idx'], group=s['group'], ra=nb['pos'].ra.deg, dec=nb['pos'].dec.deg,
                    dist_arcsec=nb['r'], flux_mJy=nb['flux'], intrinsic_fwhm_arcsec=nb['fwhm'])
               for s in systems for nb in s['neighbours']]
    Table(rows=nb_rows).write(os.path.join(img_dir, f'{GAL}_truth_neighbours.fits'), overwrite=True)
    Table(rows=[dict(band=b, pixel_scale_arcsec=i['pix'], units_per_mJy=i['upm'], sigma=i['sigma'],
                     pedestal=i['pedestal'], deger_ideal_ac=i['ac_ideal'], rendered_frac=i['rendered_frac'],
                     psf=str(i['psf_name']), psf_extra_fwhm_arcsec=i['psf_extra_fwhm'])
               for b, i in infos.items()]).write(os.path.join(img_dir, f'{GAL}_bands.fits'), overwrite=True)
    print(f'  mock images and truth tables saved -> {img_dir}', flush=True)

    for store in (p.hst_images, p.jwst_images):
        for bl in [k for k in store if k.upper() not in bands]:
            del store[bl]

    # ── mock source table, with the deger_ideal fluxes as "catalog" ──────
    ra = np.array([s['pos'].ra.deg for s in systems])
    dec = np.array([s['pos'].dec.deg for s in systems])
    tbl = Table(dict(ID=np.arange(len(systems)), PHANGS_RA=ra, PHANGS_DEC=dec,
                     PHANGS_CC_CLASS_ML=np.array(['YCL'] * len(systems)),
                     PHANGS_DMPC=np.full(len(systems), DIST),
                     group=np.array([s['group'] for s in systems]),
                     n_neighbours=np.array([len(s['neighbours']) for s in systems]),
                     nearest_neighbour_arcsec=np.array([min([nb['r'] for nb in s['neighbours']], default=np.nan)
                                                        for s in systems])))
    for b in bands:
        bl = b.lower()
        img, header, cwcs = (p.hst_images if bl in p.hst_images else p.jwst_images)[bl]
        pix, upm = infos[b]['pix'], infos[b]['upm']
        f4s, g20, sky78 = [], [], []
        for s in systems:
            x, y = (float(v) for v in s['pos'].to_pixel(cwcs))
            s78 = _sky(img, x, y, R7 / pix, R8 / pix)
            sky78.append((s78 - infos[b]['pedestal']) / infos[b]['sigma'])
            f4s.append(_ap(img, x, y, R4 / pix, s78) * infos[b]['ac_ideal'] / upm)
            g20.append(_ap(img, x, y, R20 / pix, _sky(img, x, y, R21 / pix, R23 / pix)) / upm)
        tbl[f'deger_ideal_{b}'] = f4s
        tbl[f'growth20_{b}'] = g20
        tbl[f'sky78_bias_sigma_{b}'] = sky78
        if b in CAT_NAME:
            tbl[f'PHANGS_{CAT_NAME[b]}_mJy'] = f4s

    p.gal = GAL
    p.base_dir = out_base
    p.eb_v = 0.0
    p.final_table = tbl
    p.photometry_table = tbl.copy()
    dic_pb = {b: rfp.DIC_PASSBANDS[b] for b in bands}
    p.run_native_photometry(bands, n_workers=args.workers, verbose=False, force_recompute=True,
                            plot_diagnostics=True, dic_passbands=dic_pb, instrument_fit=True)
    p.plot_native_slit_profiles(bands, verbose=False)

    # ── results ──────────────────────────────────────────────────────────
    t = p.photometry_table
    rows = []
    for i, s in enumerate(systems):
        for b in bands:
            r = dict(idx=i, group=s['group'], band=b,
                     n_neighbours=len(s['neighbours']),
                     native=float(t[f'flux_native_{b}'][i]), instG=float(t[f'flux_instG_{b}'][i]),
                     instP=float(t[f'flux_instP_{b}'][i]),
                     deger_ideal=float(t[f'deger_ideal_{b}'][i]), growth20=float(t[f'growth20_{b}'][i]),
                     apsrc=str(t[f'aperture_source_native_{b}'][i]).strip(),
                     reff_pc=float(t[f'reff_pc_instP_{b}'][i]),
                     size_src=str(t[f'size_source_instP_{b}'][i]).strip(),
                     bkg_native_sigma=(float(t[f'bkg_median_native_{b}'][i]) - infos[b]['pedestal'])
                     / infos[b]['sigma'],
                     bkg_instP_sigma=(float(t[f'bkg_median_instP_{b}'][i]) - infos[b]['pedestal'])
                     / infos[b]['sigma'],
                     bkg_78_sigma=float(t[f'sky78_bias_sigma_{b}'][i]))
            rows.append(r)
    res = Table(rows=rows)
    res.write(os.path.join(out_base, 'mock_neighbours.fits'), overwrite=True)

    print('\nrecovered / true (true = 1 mJy in every band), median over the 5 systems of each group '
          '[min-max]')
    for g, _, _ in GROUPS:
        print(f'\n  {g}')
        print('    ' + 'band'.ljust(7) + ''.join(m.rjust(20) for m in METHODS)
              + '   R_eff fit (pc)   bkg bias native/instP/7-8px (sigma/pix)')
        for b in bands:
            sel = (res['group'] == g) & (res['band'] == b)
            line = '    ' + b.ljust(7)
            for m in METHODS:
                v = np.asarray(res[m][sel], float)
                line += f'{np.nanmedian(v):7.3f} [{np.nanmin(v):4.2f}-{np.nanmax(v):4.2f}]'.rjust(20)
            line += f'   {np.nanmedian(res["reff_pc"][sel]):6.1f}         '
            line += ' / '.join(f'{np.nanmedian(res[c][sel]):+5.2f}'
                               for c in ('bkg_native_sigma', 'bkg_instP_sigma', 'bkg_78_sigma'))
            print(line)
    summarize(res, list(bands), os.path.join(out_base, 'mock_neighbours_summary.png'))
    plot_fields(p, bands, systems, os.path.join(out_base, 'mock_neighbours_fields.png'))
    print(f'\ndone in {time.time() - t0:.0f} s')


def summarize(res, bands, out_png):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = dict(native='#eda100', instG='#e87ba4', instP='#4a3aa7', deger_ideal='#2a78d6',
                  growth20='#1baf7a')
    markers = dict(native='D', instG='v', instP='P', deger_ideal='o', growth20='^')
    wl = np.array([rfp.FILTER_WAVELENGTHS[b][0] / 1e4 for b in bands])
    fig, axes = plt.subplots(1, len(GROUPS), figsize=(4.2 * len(GROUPS), 4.2), sharey=True)
    for ax, (g, _, _) in zip(axes, GROUPS):
        for k, m in enumerate(METHODS):
            med, lo, hi = [], [], []
            for b in bands:
                v = np.asarray(res[m][(res['group'] == g) & (res['band'] == b)], float)
                v = v[np.isfinite(v)]
                med.append(np.median(v) if v.size else np.nan)
                lo.append(v.min() if v.size else np.nan)
                hi.append(v.max() if v.size else np.nan)
            med, lo, hi = map(np.array, (med, lo, hi))
            ax.errorbar(wl * (1 + 0.025 * (k - 2)), med, yerr=[med - lo, hi - med], fmt=markers[m] + '-',
                        color=colors[m], ms=5, lw=1, elinewidth=0.8, capsize=0, label=m)
        ax.axhline(1.0, color='#888', ls='--', lw=1)
        ax.set_xscale('log')
        ax.set_xticks(wl)
        ax.set_xticklabels(bands, rotation=60, fontsize=7)
        ax.set_title(g, fontsize=10)
        ax.set_ylim(0.4, 2.2)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel('recovered / true flux (true = 1 mJy)')
    axes[0].legend(fontsize=8, loc='upper left')
    fig.suptitle(f'Mock {REFF_PC:g} pc EFF3 clusters with neighbours: median and min-max of 5 systems per group',
                 fontsize=11)
    plt.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def plot_fields(p, bands, systems, out_png):
    """one system per group in F555W and F200W, 3" across"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    show = [b for b in ('F555W', 'F200W') if b in bands]
    picks = [next(s for s in systems if s['group'] == g) for g, _, _ in GROUPS]
    fig, axes = plt.subplots(len(show), len(picks), figsize=(3.0 * len(picks), 3.1 * len(show)))
    axes = np.atleast_2d(axes)
    for r, b in enumerate(show):
        bl = b.lower()
        img, _, w = (p.hst_images if bl in p.hst_images else p.jwst_images)[bl]
        pix = w.proj_plane_pixel_scales()[0].to('arcsec').value
        h = int(1.5 / pix)
        for c, s in enumerate(picks):
            x, y = (int(round(float(v))) for v in s['pos'].to_pixel(w))
            cut = img[y - h:y + h + 1, x - h:x + h + 1]
            med = np.median(img)
            ax = axes[r, c]
            ax.imshow(np.arcsinh((cut - med) / (0.05 * (cut.max() - med))), origin='lower', cmap='magma',
                      extent=[-h * pix, h * pix, -h * pix, h * pix])
            for nb in s['neighbours']:
                xn, yn = (float(v) for v in nb['pos'].to_pixel(w))
                ax.plot((xn - x) * pix, (yn - y) * pix, 'c+', ms=6, mew=0.8)
            for rad, ls in ((R4, '-'), (R7, ':'), (R8, ':'), (R20, '--')):
                ax.add_patch(plt.Circle((0, 0), rad, fill=False, color='w', ls=ls, lw=0.7))
            ax.set_title(f'{s["group"]} (src {s["idx"]}) {b}', fontsize=9)
            ax.tick_params(labelsize=7)
    fig.suptitle('Example systems (cyan +: neighbours; white: 4 px aperture, 7-8 px sky, 20 px)', fontsize=10)
    plt.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


if __name__ == '__main__':
    main()
