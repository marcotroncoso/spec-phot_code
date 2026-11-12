"""
What Method 2 (PSF fit) does, shown on the real cube stamp of one cluster.

At every wavelength the stamp is modelled as a linear sum of fixed shapes
  data(x,y) = F_cl * PSF(cluster) + sum F_n * PSF(neighbour)
              + sum a_k * polynomial_k(x,y) + sum b_j * GaussianPatch_j(x,y)
and only the amplitudes are fitted; F_cl(lambda) is the cluster spectrum.
The figure shows the basis shapes, then the decomposition of three images
(continuum, [NII]6583 line-only, Halpha line-only), then the aperture spectra
around Halpha that come out of doing this slice by slice.

  python validation/muse_m2_diagram.py --id 1256
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import muse_spec_aux_func as msa                    # noqa: E402
import run_muse_spectra as rms                      # noqa: E402

COL = {'A': '#6b6a64', 'C2': '#1baf7a', 'BKG': '#eb6834', 'RES': '#2a78d6'}
FIT_HALF = 20


def band_image(D, V, wave, sel):
    """Mean image over channels ``sel`` and its variance."""
    n = np.sum(np.isfinite(D[sel]), axis=0)
    img = np.nanmean(D[sel], axis=0)
    var = np.nansum(V[sel], axis=0) / np.maximum(n, 1)**2
    return img, var


def line_only(D, V, wave, z, lam, half=3.0, off=((-55, -35), (35, 55))):
    """On-band mean minus the mean of two off-bands (rest-frame offsets in A)."""
    lo = lam * (1 + z)
    on = np.abs(wave - lo) <= half
    offm = np.zeros_like(on)
    for a, b in off:
        offm |= (wave >= lo + a * (1 + z)) & (wave <= lo + b * (1 + z))
    i_on, v_on = band_image(D, V, wave, on)
    i_off, v_off = band_image(D, V, wave, offm)
    return i_on - i_off, v_on + v_off


def decompose(img, var, A, idx, ridge):
    d = np.nan_to_num(img).ravel()
    w = np.where(np.isfinite(img) & np.isfinite(var) & (var > 0), 1 / var, 0).ravel()
    p, _ = msa.solve_slices(A, d[None], w[None], idx['patch'], ridge)
    p = p[0]
    shape = img.shape
    poly = np.setdiff1d(idx['bkg'], idx['patch'])
    comp = {'cluster': (A[:, [idx['target']]] @ p[[idx['target']]]).reshape(shape),
            'neigh': (A[:, idx['neigh']] @ p[idx['neigh']]).reshape(shape),
            'poly': (A[:, poly] @ p[poly]).reshape(shape),
            'patch': (A[:, idx['patch']] @ p[idx['patch']]).reshape(shape)}
    comp['res'] = img - sum(comp.values())
    return comp, p[idx['target']]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--id', type=int, default=1256)
    ap.add_argument('--ridge', type=float, default=1e-2)
    opt = ap.parse_args()

    from run_full_pipeline import load_config
    inp = rms.load_inputs('NGC1365-1.15asec', load_config(os.path.join(HERE, '..', 'config.ini')), 'MUSE')
    plot_dir = os.path.join(inp['outd'], 'muse_spectra')
    with open(os.path.join(plot_dir, 'NGC1365_psf.json')) as f:
        pj = json.load(f)
    psf = msa.PSFModel(pj['fwhm_ref_px'], pj['beta'], pj['slope_px_per_A'], pj['wave_ref'])
    geo, half = rms.source_geometry(inp, 1.15 / msa.MUSE_PIX_ARCSEC, 1.0, 2.0, FIT_HALF)
    i = int(np.where(inp['tab']['ID_PHANGS_CLUSTER'] == opt.id)[0][0])
    g = geo[i]
    c = msa.load_cutout(os.path.join(plot_dir, 'cutouts', f'NGC1365_{opt.id}_h{half}.npz'),
                        inp['wsel'])
    data, var, wave = c['data'], c['var'], inp['wave']
    apf = msa.polygon_cutout_mask(inp['regs'][i], inp['cel'], g['x0'], g['y0'], data.shape[1:])

    h = fits.open(os.path.join(inp['outd'], 'NGC1365_muse_spectra.fits'))
    src = Table(h['SOURCES'].data)
    k = int(np.where(src['ID_PHANGS_CLUSTER'] == opt.id)[0][0])
    z = float(src['z_local'][k])
    shift = (float(src['psf_shift_x'][k]), float(src['psf_shift_y'][k]))

    # the same stamp and positions as method_psf
    ix, iy = int(round(g['xy'][0])), int(round(g['xy'][1]))
    y0, x0 = iy - FIT_HALF, ix - FIT_HALF
    D = data[:, y0:y0 + 2 * FIT_HALF + 1, x0:x0 + 2 * FIT_HALF + 1]
    V = var[:, y0:y0 + 2 * FIT_HALF + 1, x0:x0 + 2 * FIT_HALF + 1]
    AP = apf[y0:y0 + 2 * FIT_HALF + 1, x0:x0 + 2 * FIT_HALF + 1]
    shape = D.shape[1:]
    xt0 = (g['xy'][0] - x0, g['xy'][1] - y0)
    xn_all = [(x - x0, y - y0) for (x, y) in g['neigh']
              if -2 <= x - x0 <= shape[1] + 1 and -2 <= y - y0 <= shape[0] + 1]
    fw = float(psf.fwhm(6600.0))
    _, xn = msa.group_neighbours(xt0, xn_all, fw)
    xt = (xt0[0] + shift[0], xt0[1] + shift[1])
    xn = [(x + shift[0], y + shift[1]) for (x, y) in xn]
    A, idx = msa.design_matrix(shape, xt, xn, fw, psf.beta)

    good = msa.continuum_mask(wave, z)
    rows = [('continuum 6100–6500 Å', band_image(D, V, wave, good & (wave > 6100) & (wave < 6500))),
            ('[NII]6583, line only', line_only(D, V, wave, z, 6583.45)),
            ('Hα, line only', line_only(D, V, wave, z, 6562.80, off=((-75, -60), (60, 75))))]

    fig = plt.figure(figsize=(15, 14))
    gs = fig.add_gridspec(5, 6, height_ratios=[0.85, 1, 1, 1, 1.25], hspace=0.3, wspace=0.08,
                          top=0.955, bottom=0.04)
    ext = [-0.5, shape[1] - 0.5, -0.5, shape[0] - 0.5]
    pix = msa.MUSE_PIX_ARCSEC

    def mark(ax, show_ap=True):
        ax.plot(*xt, '+', color='k', ms=9, mew=1.2)
        for (x, y) in xn:
            ax.plot(x, y, 'x', color='#3d3d3a', ms=7, mew=1.2)
        if show_ap:
            ax.contour(AP, levels=[0.5], colors='k', linewidths=0.9, extent=ext)
        ax.set_xticks([]); ax.set_yticks([])

    # ── row 0: the basis shapes ────────────────────────────────────────────
    poly_idx = np.setdiff1d(idx['bkg'], idx['patch'])
    basis = [('cluster PSF\n(Gaussian, FWHM 1.14")', A[:, idx['target']]),
             (f'neighbour PSF ({len(xn)})', A[:, idx['neigh']].sum(axis=1) if len(xn) else np.zeros(A.shape[0])),
             (f'smooth disc: 2-D polynomial\n({len(poly_idx)} terms; u², shown)', A[:, poly_idx[-1]]),
             (f'ISM patches ({len(idx["patch"])} Gaussians,\nσ = FWHM, every 2 FWHM)', A[:, idx['patch']].sum(axis=1)),
             ('one patch vs the PSF', None)]
    for j, (title, img) in enumerate(basis):
        ax = fig.add_subplot(gs[0, j])
        if img is None:
            yy = np.arange(shape[0]) - xt[1]
            ax.plot(yy * pix, A[:, idx['target']].reshape(shape)[:, int(round(xt[0]))] /
                    A[:, idx['target']].max(), color=COL['C2'], lw=2, label='PSF')
            pc = A[:, idx['patch']]
            jj = np.argmax(pc.reshape(shape[0], shape[1], -1)[int(round(xt[1])), int(round(xt[0]))])
            ax.plot(yy * pix, pc[:, jj].reshape(shape)[:, int(round(xt[0]))] / pc[:, jj].max(),
                    color=COL['BKG'], lw=2, label='patch')
            ax.set(xlabel='offset [arcsec]', yticks=[], xlim=(-4, 4))
            ax.legend(frameon=False, fontsize=8)
        else:
            ax.imshow(img.reshape(shape), origin='lower', cmap='Greys', extent=ext)
            mark(ax, show_ap=False)
        ax.set_title(title, fontsize=9)
    ax = fig.add_subplot(gs[0, 5]); ax.axis('off')
    ax.text(0, 1.12,
            'At EVERY wavelength slice λ:\n\n'
            'data(x,y) =  F$_{cl}$(λ)·PSF$_{cl}$\n'
            '         + Σ F$_n$(λ)·PSF$_n$\n'
            '         + Σ a$_k$(λ)·poly$_k$\n'
            '         + Σ b$_j$(λ)·patch$_j$\n\n'
            'Shapes fixed, only the amplitudes\nare fitted (weighted least squares;\n'
            'ridge penalty keeps the patches\nfrom imitating the PSF).\n\n'
            'F$_{cl}$(λ) for all 3761 slices\n= the cluster spectrum C2.',
            va='top', fontsize=8.6, family='monospace', color='#3d3d3a')

    # ── rows 1-3: decomposition of three images ────────────────────────────
    cols = [('data', None), ('cluster PSF', 'cluster'), ('neighbours', 'neigh'),
            ('smooth disc', 'poly'), ('ISM patches', 'patch'), ('residual', 'res')]
    for r, (label, (img, v)) in enumerate(rows, start=1):
        comp, fcl = decompose(img, v, A, idx, opt.ridge)
        vmax = np.nanpercentile(np.abs(img), 99.5)
        ap_sum = {kk: np.nansum(comp[kk] * AP) for kk in comp}
        tot = np.nansum(np.nan_to_num(img) * AP)
        for j, (ct, key) in enumerate(cols):
            ax = fig.add_subplot(gs[r, j])
            im = ax.imshow(img if key is None else comp[key], origin='lower', cmap='RdBu_r',
                           vmin=-vmax, vmax=vmax, extent=ext)
            mark(ax)
            frac = tot if key is None else ap_sum[key]
            ax.text(0.03, 0.03, f'in aperture: {frac:.0f}' + ('' if key is None else
                    f'\n({frac / tot:+.0%} of data)' if tot != 0 else ''),
                    transform=ax.transAxes, fontsize=7.5, color='#3d3d3a',
                    bbox=dict(fc='white', ec='none', alpha=0.75, pad=1.5))
            if r == 1:
                ax.set_title(('= ' if j == 1 else '+ ' if 1 < j < 5 else '' if j == 0 else '   ') + ct,
                             fontsize=10)
            if j == 0:
                ax.set_ylabel(label, fontsize=10)
        cb = fig.colorbar(im, cax=ax.inset_axes([1.04, 0, 0.05, 1]))
        cb.set_label('10$^{-20}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$ / spaxel', fontsize=7)
        cb.ax.tick_params(labelsize=7)

    # ── row 4: what slice-by-slice fitting gives around Halpha ─────────────
    ax = fig.add_subplot(gs[4, :])
    s = (wave > 6450 * (1 + z)) & (wave < 6800 * (1 + z))
    Ab, Ae = h['M1_A'].data[k], h['M1_A_ERR'].data[k]
    for name, f, e, col in (('aperture total (data)', Ab, Ae, COL['A']),
                            ('cluster PSF in aperture (C2 × EE$_{ap}$)',
                             h['M2_C2_AP'].data[k], h['M2_C2_AP_ERR'].data[k], COL['C2']),
                            ('background model in aperture (disc + ISM patches)',
                             h['M2_E_AP'].data[k], h['M2_E_AP_ERR'].data[k], COL['BKG'])):
        ax.step(wave[s], f[s], where='mid', color=col, lw=1.2, label=name)
        ax.fill_between(wave[s], f[s] - e[s], f[s] + e[s], step='mid', color=col, alpha=0.3, lw=0)
    ax.plot(wave[s], h['M2_RES_AP'].data[k][s] + h['M2_NEIGH_AP'].data[k][s], color=COL['RES'],
            lw=0.9, label='neighbours + residual in aperture')
    ax.axhline(0, color='#c3c2b7', lw=0.8)
    for nm, lam in (('[NII]', 6548.05), ('Hα', 6562.80), ('[NII]', 6583.45),
                    ('[SII]', 6716.44), ('[SII]', 6730.82)):
        ax.axvline(lam * (1 + z), color='#c3c2b7', lw=0.8, ls=':')
        ax.text(lam * (1 + z), 0.97, nm, transform=ax.get_xaxis_transform(), ha='center',
                va='top', fontsize=8.5, color='#3d3d3a')
    ax.set(xlabel='observed wavelength [Å]', ylabel='F$_λ$ [10$^{-20}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]',
           title='Doing that for every slice → spectra (inside the ISF aperture; 1σ bands)')
    ax.legend(frameon=False, fontsize=8.5, loc='upper left')

    fig.suptitle(f'Method 2 (PSF fit) on NGC1365 cluster {opt.id}: 8.2"×8.2" stamp, '
                 f'+ = cluster, × = DR5 neighbours, black contour = ISF aperture', fontsize=12, y=0.995)
    tag = rms.plot_tag(inp, opt.id, str(src['PHANGS_CC_CLASS_ML'][k]).strip())
    path = os.path.join(plot_dir, f'{tag}_m2_diagram.png')
    fig.savefig(path, dpi=120, bbox_inches='tight')
    print(path)


if __name__ == '__main__':
    main()
