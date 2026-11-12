"""
Catalog centre vs the centre our photometry finally uses, per cluster and band.

How the final centre is set (native_phot_aux_func.py):
  1. native (_measure_own_band): a Gaussian2D + constant fit to the raw image
     in a box of half-size max(3 R65_PSF, 5 px) around the catalog position
     (fit_2d_gaussian_center). If it converges, the aperture moves there. The
     shift is flagged when it exceeds 0.5 R65_PSF (recenter_flag_native_*).
  2. instG / instP (measure_instrument_photometry_one_source) do NOT refit the
     centre: the slit-profile fit runs through native's centroid. So the
     instP centroid is
        own            -> this band's own Gaussian centroid
        borrowed:<b>   -> the lender band's centroid (and its R_eff)
        point_source   -> native's centroid if native fitted one (instP fit failed),
                          else the catalog position (shift 0)
  This script uses centroid_ra/dec_instP_<band> as "final".

Expected noise of a centroid: for a Gaussian-like source of FWHM F detected
at aperture S/N, sigma_x = sigma_y ~ F / (2.355 S/N) per axis, so the
radial shift follows a Rayleigh distribution with median 1.18 sigma and
95% below 2.45 sigma (dashed lines in the S/N panel). Shifts well above that
are real (catalog vs image astrometry, or a different peak in that band).

Writes <validation_output_dir>/output_centroid_offsets/<GAL>_centroid_offsets.{png,fits,csv}
Run:   python centroid_offsets.py [--galaxy NGC1365]
"""
import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
from _paths import OUT_ROOT, OUTPUT_DIR  # noqa: E402

from astropy.table import Table  # noqa: E402

OUT = os.path.join(OUT_ROOT, 'output_centroid_offsets')
BANDS = ['F275W', 'F336W', 'F438W', 'F555W', 'F814W', 'F200W', 'F300M', 'F335M', 'F360M',
         'F770W', 'F1000W', 'F1130W', 'F2100W']
SRC_STYLE = {'own': ('#2a78d6', 'o', 'own centroid (Gaussian fit in this band)'),
             'borrowed': ('#eb6834', 's', "borrowed (lender band's centroid)"),
             'point_source': ('#8a8984', 'x', 'point-source fallback (native centroid if any, else catalog)')}
INK, INK2, GRID = '#1a1a19', '#5f5e58', '#e4e3dd'


def fwhm_from_r65(r65):
    """R65 of a Gaussian = sigma sqrt(-2 ln 0.35) = 1.449 sigma -> FWHM = 2.355 sigma."""
    return r65 * 2.3548 / np.sqrt(-2 * np.log(0.35))


def build_table(gal):
    out_gal = os.path.join(OUTPUT_DIR, gal)
    nat = Table.read(os.path.join(out_gal, f'{gal}_photometry_v2.fits'))
    ins = Table.read(os.path.join(out_gal, f'{gal}_photometry_instrument.fits'))
    assert len(nat) == len(ins)
    ra0 = np.asarray(nat['PHANGS_RA'], float)
    dec0 = np.asarray(nat['PHANGS_DEC'], float)
    dist = float(np.nanmedian(np.asarray(nat['PHANGS_DMPC'], float)))
    pc_per_arcsec = dist * 1e6 * np.pi / 180 / 3600
    rows = []
    for b in BANDS:
        if f'centroid_ra_instP_{b}' not in ins.colnames:
            continue
        ra = np.asarray(ins[f'centroid_ra_instP_{b}'], float)
        dec = np.asarray(ins[f'centroid_dec_instP_{b}'], float)
        src = np.asarray(ins[f'size_source_instP_{b}']).astype(str)
        dx = (ra - ra0) * np.cos(np.radians(dec0)) * 3600       # +East, arcsec
        dy = (dec - dec0) * 3600                                 # +North
        r65 = np.asarray(nat[f'r65_psf_native_{b}'], float)
        snr = np.asarray(nat[f'aperture_snr_native_{b}'], float)
        nsrc = np.asarray(nat[f'aperture_source_native_{b}']).astype(str)
        flag = np.asarray(nat[f'recenter_flag_native_{b}'], bool)
        shift_nat = np.asarray(nat[f'recenter_shift_native_{b}'], float)
        for k in range(len(nat)):
            s = src[k].strip()
            rows.append(dict(
                ID_PHANGS_CLUSTER=int(nat['ID_PHANGS_CLUSTER'][k]), CLASS=str(nat['PHANGS_CC_CLASS_ML'][k]).strip(),
                band=b, centre_source=s.split(':')[0], centre_source_full=s,
                ra_cat=ra0[k], dec_cat=dec0[k], ra_final=ra[k], dec_final=dec[k],
                dra_arcsec=dx[k], ddec_arcsec=dy[k], shift_arcsec=np.hypot(dx[k], dy[k]),
                shift_pc=np.hypot(dx[k], dy[k]) * pc_per_arcsec,
                r65_psf_arcsec=r65[k], fwhm_psf_arcsec=fwhm_from_r65(r65[k]),
                shift_over_r65psf=np.hypot(dx[k], dy[k]) / r65[k],
                aperture_snr=snr[k], native_aperture_source=nsrc[k].strip(),
                native_shift_arcsec=shift_nat[k], native_recenter_flag=bool(flag[k])))
    t = Table(rows=rows)
    t.meta.update(GALAXY=gal, DIST_MPC=dist, PCARCSEC=pc_per_arcsec,
                  NOTE='final = instP centroid (= native Gaussian centroid for own bands)')
    for c in t.colnames:
        if t[c].dtype.kind == 'f':
            t[c].format = '.6g'
    return t


def plot(t, out_png):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                         'ytick.color': INK2, 'text.color': INK, 'axes.linewidth': 0.8})
    gal, pcpa = t.meta['GALAXY'], t.meta['PCARCSEC']
    bands = list(dict.fromkeys(t['band']))
    fig = plt.figure(figsize=(16, 10.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05], hspace=0.38, wspace=0.28)
    rng = np.random.default_rng(1)

    # (a) shift per band, arcsec, with PSF scale markers
    ax = fig.add_subplot(gs[0, :])
    for j, b in enumerate(bands):
        s = t[t['band'] == b]
        r65 = np.nanmedian(s['r65_psf_arcsec'])
        ax.plot([j - 0.4, j + 0.4], [0.5 * r65] * 2, color='#c0392b', lw=1.2, ls=':', zorder=1)
        ax.plot([j - 0.4, j + 0.4], [r65] * 2, color=INK2, lw=1.0, ls='--', zorder=1)
        for key, (col, mk, _) in SRC_STYLE.items():
            m = np.asarray(s['centre_source']) == key
            if not m.any():
                continue
            y = np.asarray(s['shift_arcsec'][m], float)
            ax.plot(j + rng.uniform(-0.25, 0.25, m.sum()), np.clip(y, 3e-4, None), mk, ls='none', ms=5.5,
                    mfc=col if mk != 'x' else 'none', mec='white' if mk != 'x' else col, mew=0.8 if mk != 'x' else 1.2,
                    alpha=0.9, zorder=3)
        own = np.asarray(s['shift_arcsec'][np.asarray(s['centre_source']) == 'own'], float)
        if own.size:
            med = np.median(own)
            ax.plot([j - 0.32, j + 0.32], [med] * 2, color=INK, lw=2.2, zorder=4)
            ax.text(j, 0.97, f'{med * 1e3:.0f} mas\nN={own.size}', ha='center', va='top', fontsize=7.5,
                    color=INK, transform=ax.get_xaxis_transform())
        else:
            ax.text(j, 0.97, 'no own\ncentroid', ha='center', va='top', fontsize=7.5, color=INK2,
                    transform=ax.get_xaxis_transform())
    ax.set_yscale('log')
    ax.set_ylim(3e-4, 4.0)
    ax.set_xticks(range(len(bands)))
    ax.set_xticklabels(bands)
    ax.set_xlim(-0.6, len(bands) - 0.4)
    ax.set_ylabel('|final - catalog| [arcsec]')
    ax.grid(color=GRID, lw=0.6, axis='y', which='both')
    ax.set_axisbelow(True)
    sec = ax.secondary_yaxis('right', functions=(lambda a: a * pcpa, lambda p: p / pcpa))
    sec.set_ylabel('[pc]')
    ax.text(0.003, 1.02, '(a) centre shift per band. Thick line = median of own centroids (value at top). '
            'Dashed = R65 of the PSF; dotted red = 0.5 R65 (pipeline recenter-flag threshold). '
            'Values < 0.3 mas drawn at 0.3 mas.', transform=ax.transAxes, fontsize=9, va='bottom')
    hs = [plt.Line2D([], [], marker=mk, ls='none', mfc=col if mk != 'x' else 'none',
                     mec='white' if mk != 'x' else col, ms=7, label=lbl) for col, mk, lbl in SRC_STYLE.values()]
    ax.legend(handles=hs, loc='lower right', fontsize=8.5, frameon=True, framealpha=0.9)

    own = t[np.asarray(t['centre_source']) == 'own']
    hst = np.isin(own['band'], ['F275W', 'F336W', 'F438W', 'F555W', 'F814W'])
    # (b) vector offsets HST, (c) vector offsets JWST
    for col_i, (sel, title) in enumerate([(hst, '(b) HST bands, own centroids'),
                                          (~hst, '(c) JWST bands, own centroids')]):
        ax = fig.add_subplot(gs[1, col_i])
        s = own[sel]
        lim = 0.06 if col_i == 0 else 0.15
        cmap = plt.get_cmap('viridis')
        bl = list(dict.fromkeys(s['band']))
        for k, b in enumerate(bl):
            m = s['band'] == b
            c = cmap(k / max(len(bl) - 1, 1))
            dx, dy = np.asarray(s['dra_arcsec'][m], float), np.asarray(s['ddec_arcsec'][m], float)
            ax.plot(dx, dy, 'o', ms=4.5, color=c, alpha=0.75, mec='none', label=b)
            ax.plot(np.median(dx), np.median(dy), 'P', ms=12, color=c, mec=INK, mew=0.8, zorder=5)
        for r, ls in [(np.nanmedian(s['r65_psf_arcsec']) * 0.5, ':')]:
            ax.add_patch(plt.Circle((0, 0), r, fill=False, color='#c0392b', ls=ls, lw=1.2))
        ax.axhline(0, color=INK2, lw=0.8); ax.axvline(0, color=INK2, lw=0.8)
        ax.set_xlim(lim, -lim); ax.set_ylim(-lim, lim)      # East to the left
        ax.set_aspect('equal')
        ax.set_xlabel(r'$\Delta\alpha\cos\delta$ [arcsec]  (East $\leftarrow$)')
        ax.set_ylabel(r'$\Delta\delta$ [arcsec]')
        ax.grid(color=GRID, lw=0.6)
        ax.set_title(f'{title}\nbig + = band median; red circle = 0.5 R65 PSF', fontsize=9.5)
        ax.legend(fontsize=7.5, ncol=2, loc='upper left', frameon=True, framealpha=0.85, markerscale=1.2)
        out = np.sum((np.abs(np.asarray(s['dra_arcsec'], float)) > lim) | (np.abs(np.asarray(s['ddec_arcsec'], float)) > lim))
        if out:
            ax.text(0.98, 0.02, f'{out} outside the axes', transform=ax.transAxes, ha='right', fontsize=8, color=INK2)

    # (d) shift / FWHM vs S/N, with the expected centroid noise
    ax = fig.add_subplot(gs[1, 2])
    x = np.asarray(own['aperture_snr'], float)
    y = np.asarray(own['shift_arcsec'], float) / np.asarray(own['fwhm_psf_arcsec'], float)
    ax.plot(x[hst], y[hst], 'o', ms=4.5, color='#2a78d6', alpha=0.75, mec='none', label='HST')
    ax.plot(x[~hst], y[~hst], 'D', ms=4.5, color='#1baf7a', alpha=0.75, mec='none', label='JWST')
    sn = np.logspace(0, 3.3, 100)
    sig = 1 / (2.3548 * sn)                       # per-axis sigma in units of FWHM
    ax.plot(sn, 1.177 * sig, '--', color=INK2, lw=1.1, label='expected median (noise only)')
    ax.plot(sn, 2.448 * sig, ':', color=INK2, lw=1.1, label='expected 95%')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(3, 2e3); ax.set_ylim(1e-3, 3)
    ax.set_xlabel('native aperture S/N')
    ax.set_ylabel('shift / PSF FWHM')
    ax.grid(color=GRID, lw=0.6, which='both')
    ax.legend(fontsize=8, loc='lower left')
    frac = np.mean(y > 2.448 / (2.3548 * x))
    ax.set_title(f'(d) shift vs S/N: {100 * frac:.0f}% of own centroids lie\nabove the 95% noise line', fontsize=9.5)

    fig.suptitle(f'{gal}: catalog centre vs final (fitted) centre of our photometry, '
                 f'{len(set(t["ID_PHANGS_CLUSTER"]))} clusters', x=0.01, ha='left', fontsize=13)
    fig.savefig(out_png, dpi=130, facecolor='#fcfcfb', bbox_inches='tight')
    plt.close(fig)


def report(t):
    print(f'{t.meta["GALAXY"]}: 1" = {t.meta["PCARCSEC"]:.1f} pc')
    print('band    N own/borr/pt   shift own [mas]: median  84%   max   | median dRA, dDec [mas] | R65psf [mas]  flagged')
    for b in dict.fromkeys(t['band']):
        s = t[t['band'] == b]
        cs = np.asarray(s['centre_source'])
        o = s[cs == 'own']
        n = f'{np.sum(cs == "own"):2d}/{np.sum(cs == "borrowed"):2d}/{np.sum(cs == "point_source"):2d}'
        if len(o):
            sh = np.asarray(o['shift_arcsec'], float) * 1e3
            print(f'{b:7s} {n}          {np.median(sh):6.1f} {np.percentile(sh, 84):6.1f} {sh.max():6.1f}   | '
                  f'{np.median(o["dra_arcsec"]) * 1e3:+6.1f} {np.median(o["ddec_arcsec"]) * 1e3:+6.1f}            | '
                  f'{np.nanmedian(s["r65_psf_arcsec"]) * 1e3:6.0f}      {int(np.sum(o["native_recenter_flag"]))}')
        else:
            print(f'{b:7s} {n}          (no own centroid: catalog position)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--galaxy', default='NGC1365')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t = build_table(a.galaxy)
    base = os.path.join(OUT, f'{a.galaxy}_centroid_offsets')
    t.write(base + '.fits', overwrite=True)
    t.write(base + '.csv', overwrite=True)
    plot(t, base + '.png')
    report(t)


if __name__ == '__main__':
    main()
