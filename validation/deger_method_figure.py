"""
Explanatory figure of the Deger+22 (PHANGS-HST) aperture-correction recipe,
drawn with the NGC1365 F555W instrument PSF model (EFF gamma=3 clusters at
19.57 Mpc) so the shapes are realistic. Numbers are the published NGC1365
values (Deger+22 Table 2, Section 4.3).

Writes <validation_output_dir>/output_deger_ac_vs_size/deger_method_schematic.{png,pdf}
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
from _paths import OUT_ROOT  # noqa: E402

from astropy.table import Table  # noqa: E402
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry  # noqa: E402

from instrument_psf import instrument_psf_for_image  # noqa: E402
from mock_neighbours_test import load_real  # noqa: E402
from deger_aperture_correction import DEGER22_TABLE2, BAND_OFFSET_MAG, ac_mag_to_flux_factor  # noqa: E402

OUT = os.path.join(OUT_ROOT, 'output_deger_ac_vs_size')
PIX = 0.0396                         # PHANGS-HST pixel (")
DIST = 19.57
AS_PER_PC = 206265.0 / (DIST * 1e6)
# palette (validated default, fixed order) + text tokens
C1, C2, C3, C4, C5 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#4a3aa7'
INK, INK2, MUTED, GRID, SURF = '#1a1a19', '#5f5e58', '#8a8984', '#e4e3dd', '#fcfcfb'


def growth(img, x, y, radii_px, sky_in, sky_out):
    v = CircularAnnulus((x, y), sky_in, sky_out).to_mask('center').get_values(img)
    s = float(np.median(v))
    tb = aperture_photometry(img - s, [CircularAperture((x, y), r) for r in radii_px])
    return np.array([tb[f'aperture_sum_{i}'][0] for i in range(len(radii_px))])


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch
    plt.rcParams.update({'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                         'ytick.color': INK2, 'text.color': INK, 'axes.linewidth': 0.8,
                         'font.size': 10})

    p = load_real()
    _, header, wcs = p.hst_images['f555w']
    pix = wcs.proj_plane_pixel_scales()[0].to('arcsec').value
    model, _ = instrument_psf_for_image('F555W', header, pix, extra_fwhm_arcsec=0.0)   # theoretical PSF
    ac_tot, s1, s2, s3 = DEGER22_TABLE2['NGC1365']
    k_v = float(ac_mag_to_flux_factor(ac_tot))
    curve = Table.read(os.path.join(OUT, 'NGC1365_ac_vs_size.fits'))
    curve = curve[(curve['band'] == 'F555W') & (curve['profile'] == 'eff3.0')]

    fig = plt.figure(figsize=(17, 10.5), facecolor=SURF)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.95], hspace=0.42, wspace=0.28,
                          left=0.04, right=0.98, top=0.88, bottom=0.06)

    # ── (1) measure ───────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    n, c = 121, 60.0
    rng = np.random.default_rng(3)
    img = model.render(c, c, (n, n), flux=1.0, profile='eff', r_eff_arcsec=4 * AS_PER_PC, gamma=3.0, c=1.5)
    for dx, dy, f in [(-18, 10, 0.35), (16, 13, 0.5), (15, -5, 0.25)]:
        img += model.render(c + dx, c + dy, (n, n), flux=f)
    img += 0.0004 + rng.normal(0, 0.00025, (n, n))
    h = 26
    sub = img[int(c) - h:int(c) + h + 1, int(c) - h:int(c) + h + 1]
    ax.imshow(sub, origin='lower', cmap='Greys', vmin=0, vmax=np.percentile(sub, 99.6),
              extent=[-h - 0.5, h + 0.5, -h - 0.5, h + 0.5])
    ax.add_patch(Circle((0, 0), 4, fill=False, color=C2, lw=2.2))
    ring = plt.matplotlib.patches.Wedge((0, 0), 8, 0, 360, width=1, color=C1, alpha=0.55)
    ax.add_patch(ring)
    for r, ls in [(10, '--'), (20, '--')]:
        ax.add_patch(Circle((0, 0), r, fill=False, color=MUTED, lw=1.2, ls=ls))
    ax.add_patch(plt.matplotlib.patches.Wedge((0, 0), 23, 0, 360, width=2, color=MUTED, alpha=0.35))
    ax.annotate('4 px = 0.158"\naperture: $F_4$', xy=(2.8, 2.8), xytext=(9, -22), color=INK, fontsize=9,
                arrowprops=dict(arrowstyle='-', color=C2, lw=1.2))
    ax.annotate('local sky\n7-8 px (median)', xy=(-5.7, -5.7), xytext=(-25, -24), color=INK, fontsize=9,
                arrowprops=dict(arrowstyle='-', color=C1, lw=1.2))
    ax.text(0, 10.6, '10 px', ha='center', fontsize=8, color=INK2)
    ax.text(0, 20.6, '20 px = 0.79"', ha='center', fontsize=8, color=INK2)
    ax.text(-16.5, 16.5, 'far sky\n21-23 px', ha='center', fontsize=8, color=INK2)
    ax.set_xlim(-h, h); ax.set_ylim(-h, h)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title('1. Measure every cluster in a small aperture', loc='left', fontsize=11.5, fontweight='bold')
    ax.text(0, -0.13, 'Only the core is measured: robust in crowded fields,\nbut it misses the light outside 4 px.',
            transform=ax.transAxes, fontsize=9, color=INK2, va='top')

    # ── (2) calibrate once per galaxy: growth curves ─────────────────────
    ax = fig.add_subplot(gs[0, 1])
    radii = np.linspace(1, 20, 60)
    nn, cc = 141, 70.0
    curves = []
    for reff, col in [(1, None), (2, None), (3, None), (5, None), (7, None), (10, None)]:
        im = model.render(cc, cc, (nn, nn), flux=1.0, profile='eff', r_eff_arcsec=reff * AS_PER_PC, gamma=3.0, c=1.5)
        g = growth(im, cc, cc, radii, 21, 23)
        curves.append(g / g[-1])
        ax.plot(radii, g / g[-1], '-', color=MUTED, lw=0.9, alpha=0.8)
    mean = np.mean(curves, axis=0)
    ax.plot(radii, mean, '-', color=INK, lw=2.4, label='mean of the bright, isolated\nreference clusters')
    f4, f10 = np.interp(4, radii, mean), np.interp(10, radii, mean)
    for r in (4, 10, 20):
        ax.axvline(r, color=GRID, lw=1, zorder=0)
    ax.plot([4, 4], [f4, f10], color=C2, lw=3, solid_capstyle='butt')
    ax.plot([10, 10], [f10, 1.0], color=C4, lw=3, solid_capstyle='butt')
    ax.annotate(f'step 1: 4 -> 10 px\nAC$_1$ = {s1:+.2f} mag', xy=(4, (f4 + f10) / 2), xytext=(5.2, 0.42),
                fontsize=9, arrowprops=dict(arrowstyle='-', color=C2, lw=1))
    ax.annotate(f'step 2: 10 -> 20 px\nAC$_2$ = {s2:+.2f} mag', xy=(10, (f10 + 1) / 2), xytext=(11.5, 0.62),
                fontsize=9, arrowprops=dict(arrowstyle='-', color=C4, lw=1))
    ax.text(11.5, 0.30, f'step 3: cluster light inside the\n7-8 px sky ring  AC$_3$ = {s3:+.3f} mag',
            fontsize=9, color=INK)
    ax.text(19.6, 0.06, f'NGC1365: AC$_V$ = {ac_tot:+.2f} mag\n= x{k_v:.2f} in flux', ha='right', fontsize=11,
            fontweight='bold', color=INK,
            bbox=dict(boxstyle='round,pad=0.4', fc='white', ec=INK2, lw=0.8))
    ax.set_xlim(0, 20.5); ax.set_ylim(0, 1.05)
    ax.set_xlabel('aperture radius [HST px, 0.0396"]')
    ax.set_ylabel('light inside radius / light inside 20 px')
    ax.legend(loc='upper left', fontsize=8.5, frameon=False)
    ax.text(14, 0.94, 'thin: individual clusters', fontsize=8, color=MUTED)
    ax.grid(color=GRID, lw=0.5)
    ax.set_title('2. Calibrate ONE correction per galaxy (V band)', loc='left', fontsize=11.5, fontweight='bold')
    ax.text(0, -0.13, 'AC = sum of 3 steps, from the AVERAGE growth curve;\n"total" is defined as the light within 20 px (0.79").',
            transform=ax.transAxes, fontsize=9, color=INK2, va='top')

    # ── (3) other bands ──────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    bands = ['F275W', 'F336W', 'F438W', 'F555W', 'F814W']
    names = ['NUV', 'U', 'B', 'V', 'I']
    acs = [ac_tot + BAND_OFFSET_MAG[b] for b in bands]
    x = np.arange(len(bands))
    ax.bar(x, [-ac_tot] * 5, color=C1, width=0.62, label=f'AC$_V$ ({ac_tot:+.2f} mag)')
    ax.bar(x, [-BAND_OFFSET_MAG[b] for b in bands], bottom=[-ac_tot] * 5, color=C2, width=0.62,
           label='band offset (PSF wider/narrower than V)')
    for xi, a in zip(x, acs):
        ax.text(xi, -a + 0.02, f'{a:+.2f} mag\nx{float(ac_mag_to_flux_factor(a)):.2f}', ha='center', va='bottom',
                fontsize=9, color=INK)
    ax.set_xticks(x); ax.set_xticklabels([f'{b}\n{nm}' for b, nm in zip(bands, names)])
    ax.set_ylabel('|aperture correction| [mag]')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper right', fontsize=8.5, frameon=False)
    ax.grid(color=GRID, lw=0.5, axis='y'); ax.set_axisbelow(True)
    ax.set_title('3. Transfer it to the other bands', loc='left', fontsize=11.5, fontweight='bold')
    ax.text(0, -0.17, 'AC$_{band}$ = AC$_V$ + fixed offset (Deger+22 Sec. 4.3),\nthe same for every galaxy.',
            transform=ax.transAxes, fontsize=9, color=INK2, va='top')

    # ── (4) the formula ──────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, :2])
    ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis('off')
    ax.set_title('4. Catalog flux of every cluster', loc='left', fontsize=11.5, fontweight='bold')
    boxes = [(0.1, '$F_4$\n4 px flux,\nsky 7-8 px', C2),
             (2.6, '$\\times\\,10^{-0.4\\,AC_{band}}$\nthe same factor\nfor ALL clusters', C1),
             (5.1, '$\\times$ MW\ndereddening\n(SFD E(B-V))', MUTED),
             (7.6, '$F_{total}$\nPHANGS_<band>_mJy', INK)]
    for x0, txt, col in boxes:
        ax.add_patch(FancyBboxPatch((x0, 1.6), 2.0, 1.7, boxstyle='round,pad=0.05,rounding_size=0.15',
                                    fc='white', ec=col, lw=2))
        ax.text(x0 + 1.0, 2.45, txt, ha='center', va='center', fontsize=10.5, color=INK)
    for x0 in (2.15, 4.65, 7.15):
        ax.add_patch(FancyArrowPatch((x0, 2.45), (x0 + 0.42, 2.45), arrowstyle='-|>', mutation_scale=16,
                                     color=INK2, lw=1.5))
    ax.text(0.1, 1.05, f'NGC1365, V band:  F_total = F_4 x {k_v:.2f} x {1.055:.3f}.   '
            'Re-measured on the images, this reproduces the catalog to 0-2% in every band.',
            fontsize=9.5, color=INK)
    ax.text(0.1, 0.45, 'In magnitudes: m_total = m_4px + AC_band  (AC < 0 makes the cluster brighter).',
            fontsize=9.5, color=INK2)

    # ── (5) the catch ────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    r0 = np.asarray(curve['reff_pc'], float)
    k0 = np.asarray(curve['k_cluster'], float)
    r = np.linspace(0, 12, 241)
    k = np.interp(r, r0, k0)
    m = r <= 12
    ax.plot(r[m], k[m], '-', color=C5, lw=2.4, label='correction this cluster really needs')
    ax.axhline(k_v, color=C1, lw=2, ls='--', label=f'galaxy correction (x{k_v:.2f})')
    ax.fill_between(r[m], k[m], k_v, where=k[m] < k_v, color=C2, alpha=0.18, lw=0, interpolate=True)
    ax.fill_between(r[m], k[m], k_v, where=k[m] > k_v, color=C3, alpha=0.18, lw=0, interpolate=True)
    r_eq = float(np.interp(k_v, k0, r0))
    ax.plot(r_eq, k_v, 'o', ms=8, color=INK, mec='white', mew=1.5)
    ax.annotate(f'exact only for R$_{{eff}}$ ~ {r_eq:.0f} pc', xy=(r_eq, k_v), xytext=(r_eq - 5.2, 2.15),
                fontsize=9, arrowprops=dict(arrowstyle='-', color=INK2, lw=1))
    ax.text(0.4, 1.58, 'compact clusters\nover-corrected\n(x1.3-1.4 too bright)', fontsize=9, color=INK)
    ax.text(8.6, 1.42, 'extended clusters\nunder-corrected', fontsize=9, color=INK, ha='center')
    ax.axvspan(0, 2.7, color=GRID, alpha=0.5, lw=0)
    ax.text(1.35, 2.62, 'most NGC1365\nclusters (instP)', ha='center', fontsize=8.5, color=INK2)
    ax.set_xlim(0, 12); ax.set_ylim(1.1, 2.8)
    ax.set_xlabel('cluster R$_{eff}$ [pc]  (EFF $\\gamma$=3, 19.57 Mpc)')
    ax.set_ylabel('flux correction $F_{total}/F_4$ (V)')
    ax.legend(loc='upper left', fontsize=8.5, frameon=False, bbox_to_anchor=(0.0, 0.86))
    ax.grid(color=GRID, lw=0.5)
    ax.set_title('5. The catch: one factor for every cluster', loc='left', fontsize=11.5, fontweight='bold')

    fig.suptitle('How the PHANGS-HST cluster catalog gets its total fluxes (Deger et al. 2022)', x=0.04,
                 ha='left', fontsize=15, fontweight='bold', y=0.975)
    fig.text(0.04, 0.925, 'A small-aperture flux is corrected to "total" with one average correction per galaxy, '
             'calibrated on bright isolated clusters.  Shapes drawn with the NGC1365 F555W PSF; numbers are '
             'the published NGC1365 values.', fontsize=10, color=INK2)
    os.makedirs(OUT, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'deger_method_schematic.{ext}'), dpi=150, facecolor=SURF)
    plt.close(fig)


if __name__ == '__main__':
    main()
