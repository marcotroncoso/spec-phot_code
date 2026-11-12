"""
NGC1365: our instrument-profile photometry (instP) and the Deger+22 catalog
recipe re-measured by us, each compared with the PHANGS-HST catalog fluxes.

  instP   pipeline cache output/NGC1365/NGC1365_photometry_instrument.fits
          (PSF (x) EFF gamma=3 for YCL/MAP, King c=1.5 for OGCC; galaxy-matched
          PSF; Milky Way dereddened)
  deger   4 px (0.158") aperture at the catalog position, sky median at 7-8 px,
          x 10**(-0.4 (AC_V + offset_band)) with AC_V = -0.61 mag (Deger+22
          Table 2) and the Section 4.3 band offsets, x the Milky Way correction
          (deger_aperture_correction.py)
  catalog PHANGS_<NUV/U/B/V/I>_mJy (Milky Way dereddened)

If the recipe is right, deger / catalog = 1 in every band; instP / catalog
shows how our method differs from the catalog.

Error bars are 1 sigma: catalog PHANGS_<band>_mJy_ERR; instP the pipeline
flux_instP_<band>_err; deger sigma_sky sqrt(N_ap) of the 4 px aperture (sky
std from the 7-8 px annulus) x the same correction factor and MW factor as the
flux. Ratio errors add both in quadrature.

Writes <validation_output_dir>/output_catalog_comparison/:
  NGC1365_instP_vs_catalog.png, NGC1365_deger_vs_catalog.png,
  NGC1365_catalog_comparison.fits / .csv  (one row per cluster)
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)
from _paths import OUT_ROOT, OUTPUT_DIR  # noqa: E402

import astropy.units as u  # noqa: E402
from astropy.coordinates import SkyCoord  # noqa: E402
from astropy.io import fits  # noqa: E402
from astropy.table import Table  # noqa: E402
from astropy.wcs import WCS  # noqa: E402

import run_full_pipeline as rfp  # noqa: E402
from phot_aux_func import corr_flux  # noqa: E402
from deger_aperture_correction import (DEGER22_TABLE2, HST_PIX_ARCSEC, CATALOG_BAND,  # noqa: E402
                                       BAND_OFFSET_MAG, ac_mag_to_flux_factor, band_ac,
                                       aperture_flux, aperture_flux_err, _sky)

GAL = 'NGC1365'
BANDS = ['F275W', 'F336W', 'F438W', 'F555W', 'F814W']
OUT = os.path.join(OUT_ROOT, 'output_catalog_comparison')
# categorical slots 1-3 of the validated default palette (dataviz skill), in fixed order
CLASS_STYLE = {'MAP': ('#2a78d6', 'o'), 'OGCC': ('#eb6834', 's'), 'YCL': ('#1baf7a', 'D')}
INK, INK2, GRID = '#1a1a19', '#5f5e58', '#e4e3dd'


def galaxy_ebv():
    try:
        from dustmaps.sfd import SFDQuery
        return float(SFDQuery()(SkyCoord.from_name(GAL)))
    except Exception as e:                       # offline: the value the pipeline used
        print(f'SFD query failed ({e}); using E(B-V)=0.0205')
        return 0.0205


def build_table():
    out_gal = os.path.join(OUTPUT_DIR, GAL)
    cat = Table.read(os.path.join(out_gal, f'{GAL}_photometry_v2.fits'))
    ins = Table.read(os.path.join(out_gal, f'{GAL}_photometry_instrument.fits'))
    assert len(cat) == len(ins)
    ebv = galaxy_ebv()
    ac_v = DEGER22_TABLE2[GAL][0]
    cfg = rfp.load_config(os.path.join(_ROOT, 'config.ini'))
    paths = rfp.find_hst_band_paths(cfg['hst_root'], GAL)
    sky = SkyCoord(np.asarray(cat['PHANGS_RA'], float) * u.deg, np.asarray(cat['PHANGS_DEC'], float) * u.deg)

    t = Table()
    t['ID_PHANGS_CLUSTER'] = cat['ID_PHANGS_CLUSTER']
    t['RA'] = np.asarray(cat['PHANGS_RA'], float)
    t['DEC'] = np.asarray(cat['PHANGS_DEC'], float)
    t['CLASS'] = [str(c).strip() for c in cat['PHANGS_CC_CLASS_ML']]
    t['profile_instP'] = [str(p).strip() for p in ins['profile_instP']]
    t.meta.update(GALAXY=GAL, EBV=ebv, AC_V=ac_v)
    for b in BANDS:
        cb = CATALOG_BAND[b]
        with fits.open(paths[b]) as h:
            img, hdr = h[0].data, h[0].header
        w = WCS(hdr)
        k = HST_PIX_ARCSEC / float(w.proj_plane_pixel_scales()[0].to('arcsec').value)
        f4, e4 = [], []
        for c in sky:
            x, y = (float(v) for v in c.to_pixel(w))
            f4.append(aperture_flux(img, x, y, 4 * k, _sky(img, x, y, 7 * k, 8 * k)) * 1e3)  # Jy -> mJy
            e4.append(aperture_flux_err(img, x, y, 4 * k, 7 * k, 8 * k)[0] * 1e3)      # sigma_sky sqrt(N_ap)
        mw = float(corr_flux(1.0, rfp.FILTER_WAVELENGTHS[b][1], ebv))
        ac = band_ac(ac_v, b)
        t[f'catalog_{b}'] = np.asarray(cat[f'PHANGS_{cb}_mJy'], float)
        t[f'catalog_{b}_err'] = np.asarray(cat[f'PHANGS_{cb}_mJy_ERR'], float)
        t[f'instP_{b}'] = np.asarray(ins[f'flux_instP_{b}'], float)
        t[f'instP_{b}_err'] = np.asarray(ins[f'flux_instP_{b}_err'], float)
        t[f'instP_size_source_{b}'] = [str(s).strip() for s in ins[f'size_source_instP_{b}']]
        t[f'instP_reff_pc_{b}'] = np.asarray(ins[f'reff_pc_instP_{b}'], float)
        t[f'f4px_{b}'] = np.array(f4)                                   # no correction, no MW
        t[f'f4px_{b}_err'] = np.array(e4)
        t[f'deger_{b}'] = np.array(f4) * float(ac_mag_to_flux_factor(ac)) * mw
        t[f'deger_{b}_err'] = np.array(e4) * float(ac_mag_to_flux_factor(ac)) * mw
        t[f'instP_over_catalog_{b}'] = t[f'instP_{b}'] / t[f'catalog_{b}']
        t[f'deger_over_catalog_{b}'] = t[f'deger_{b}'] / t[f'catalog_{b}']
        t.meta[f'AC_{b}'] = ac
        t.meta[f'MW_{b}'] = mw
    t.meta['ERRDEF'] = 'aperture errors: sigma_sky*sqrt(N_ap) x correction x MW'
    for c in t.colnames:
        if t[c].dtype.kind == 'f':
            t[c].format = '.6g'
    return t


def stats(r):
    r = np.asarray(r, float)
    r = r[np.isfinite(r) & (r > 0)]
    return (np.median(r), *np.percentile(r, [16, 84]), r.size) if r.size else (np.nan,) * 3 + (0,)


def plot(t, method, title, out_png):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                         'ytick.color': INK2, 'text.color': INK, 'axes.linewidth': 0.8})
    fig, axes = plt.subplots(2, len(BANDS), figsize=(3.3 * len(BANDS), 6.6),
                             gridspec_kw=dict(height_ratios=[3, 1.6]), sharex='col')
    for j, b in enumerate(BANDS):
        ax, axr = axes[0, j], axes[1, j]
        x = np.asarray(t[f'catalog_{b}'], float)
        y = np.asarray(t[f'{method}_{b}'], float)
        xe = np.abs(np.asarray(t[f'catalog_{b}_err'], float))   # 1 catalog F275W error is stored negative
        ye = np.abs(np.asarray(t[f'{method}_{b}_err'], float))
        ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        lo, hi = np.min(np.r_[x[ok], y[ok]]) / 1.6, np.max(np.r_[x[ok], y[ok]]) * 1.6
        for a in (ax, axr):
            a.grid(color=GRID, lw=0.6)
            a.set_axisbelow(True)
        ax.plot([lo, hi], [lo, hi], color=INK2, lw=1, ls='--', zorder=1)
        axr.axhline(1.0, color=INK2, lw=1, ls='--', zorder=1)
        for cls, (col, mk) in CLASS_STYLE.items():
            m = ok & (np.asarray(t['CLASS']) == cls)
            if not m.any():
                continue
            kw = dict(marker=mk, ls='none', ms=7, mfc=col, mec='white', mew=1.0, zorder=3,
                      ecolor=col, elinewidth=1.0, capsize=2, alpha=0.95,
                      label=f'{cls} ({int(np.sum(np.asarray(t["CLASS"]) == cls))})')
            ax.errorbar(x[m], y[m], xerr=xe[m], yerr=ye[m], **kw)
            # ratio error: both errors in quadrature
            r = y[m] / x[m]
            axr.errorbar(x[m], r, yerr=r * np.hypot(ye[m] / y[m], xe[m] / x[m]), **kw)
        med, p16, p84, n = stats(y[ok] / x[ok])
        axr.axhline(med, color=INK, lw=1.2, zorder=2)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        axr.set_xscale('log')
        axr.set_ylim(0.6, 1.4) if method == 'deger' else axr.set_ylim(0.2, 1.8)
        ax.set_title(f'{b} ({CATALOG_BAND[b]})', fontsize=11)
        ax.text(0.04, 0.96, f'median ratio {med:.3f}\n16-84%: {p16:.2f}-{p84:.2f}\nN = {n}',
                transform=ax.transAxes, va='top', fontsize=8.5, color=INK)
        axr.set_xlabel('catalog flux [mJy]')
        if j == 0:
            ax.set_ylabel(f'{method} flux [mJy]')
            axr.set_ylabel(f'{method} / catalog')
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc='upper right', ncol=len(h), frameon=False, fontsize=9, bbox_to_anchor=(0.99, 1.0))
    fig.suptitle(title + r'  (1$\sigma$ errors)', x=0.01, ha='left', fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png, dpi=130, facecolor='#fcfcfb')
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    t = build_table()
    t.write(os.path.join(OUT, f'{GAL}_catalog_comparison.fits'), overwrite=True)
    t.write(os.path.join(OUT, f'{GAL}_catalog_comparison.csv'), overwrite=True)
    ac_v = t.meta['AC_V']
    plot(t, 'instP', f'{GAL}: instrument-profile photometry (instP) vs PHANGS-HST catalog',
         os.path.join(OUT, f'{GAL}_instP_vs_catalog.png'))
    plot(t, 'deger', f'{GAL}: Deger+22 recipe re-measured (4 px, sky 7-8 px, AC_V = {ac_v} mag + band offsets) '
                     f'vs PHANGS-HST catalog', os.path.join(OUT, f'{GAL}_deger_vs_catalog.png'))
    print(f'E(B-V) = {t.meta["EBV"]:.4f},  AC_V = {ac_v} mag')
    print('band    AC [mag]  factor   deger/catalog median [16-84]   instP/catalog median [16-84]')
    for b in BANDS:
        d, i = stats(t[f'deger_over_catalog_{b}']), stats(t[f'instP_over_catalog_{b}'])
        print(f'{b}   {t.meta[f"AC_{b}"]:+.2f}     x{float(ac_mag_to_flux_factor(t.meta[f"AC_{b}"])):.3f}   '
              f'{d[0]:.3f} [{d[1]:.3f}-{d[2]:.3f}] N={d[3]}       {i[0]:.3f} [{i[1]:.3f}-{i[2]:.3f}] N={i[3]}')
    print(f'written to {OUT}')


if __name__ == '__main__':
    main()
