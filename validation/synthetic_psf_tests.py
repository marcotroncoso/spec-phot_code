"""
Synthetic PSF test images: point source vs. star cluster.

Generates noiseless (by default) FITS test images of (1) a single true point
source and (2) a tight synthetic "star cluster" of several point sources,
each convolved with a circular Gaussian PSF over a grid of FWHM values. The
matched pair of sequences (same FWHM, same total flux, point vs. cluster) is
meant to be compared via 2D-Gaussian fitting to test whether an unresolved
star cluster behaves like a point source under the PSF, or shows systematic
shape/residual deviations from a Gaussian model.

generate_all_hst_psf_test_images builds the same point/cluster pair through
the model HST PSFs stored next to this file (WFC3/UVIS and ACS/WFC pickles),
plus a Gaussian for the extra blur of drizzled PHANGS-HST mosaics and an
optional Gaussian intrinsic cluster size.

Self-contained: only numpy + scipy + astropy. Does not depend on the rest of the
spec-phot pipeline (aux_functions.py, native_phot_aux_func.py), which are
built for different, more complex workflows.
"""
import os
import sys

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.convolution import Gaussian2DKernel, convolve_fft

# HST PSF model + renderer live in the pipeline module instrument_psf.py so the
# mocks and fit_slit_profile_instrument_psf use the identical PSF model.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from instrument_psf import (HST_PSF_FILES, HST_DRIZZLE_FWHM_ARCSEC,  # noqa: E402,F401
                            load_hst_psf, render_with_hst_psf, render_sources,
                            InstrumentPSF, JWST_PIXEL_SCALE_ARCSEC,
                            _eff_cum, eff_scale_for_reff, smooth_profile_sb,
                            smooth_profile_truncation, smooth_profile_scale_for_reff,
                            render_smooth_profile)

FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))

DEFAULT_FWHM_GRID_ARCSEC = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)

PHANGS_JWST_BANDS = ('F200W', 'F300M', 'F335M', 'F360M',
                     'F770W', 'F1000W', 'F1130W', 'F2100W')

PHANGS_HST_BANDS = (('wfc3', 'F275W'), ('wfc3', 'F336W'), ('wfc3', 'F438W'),
                    ('wfc3', 'F555W'), ('wfc3', 'F814W'))

DEFAULT_INTRINSIC_FWHM_GRID_ARCSEC = (0.0, 0.02, 0.05, 0.1, 0.15, 0.2)


def build_test_wcs(pixel_scale_arcsec=0.04, npix=401, crval=(150.0, 2.0)):
    """Minimal tangent-plane WCS for a synthetic square image, with the
    reference pixel placed exactly on the array's center pixel (requires
    npix odd)."""
    if npix % 2 != 1:
        raise ValueError(f"npix must be odd so a source can sit on an exact "
                          f"center pixel, got {npix}")
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crval = list(crval)
    w.wcs.crpix = [(npix + 1) / 2, (npix + 1) / 2]
    w.wcs.cdelt = [-pixel_scale_arcsec / 3600, pixel_scale_arcsec / 3600]
    w.pixel_shape = (npix, npix)
    return w


def _place_sources_bilinear(image, xs, ys, fluxes):
    """Splat each source's flux onto its 4 nearest pixels by bilinear
    weight, in place. For an exact-integer (x, y) this degenerates to a
    single-pixel delta."""
    npix_y, npix_x = image.shape
    for x, y, flux in zip(xs, ys, fluxes):
        x0 = int(np.floor(x))
        y0 = int(np.floor(y))
        fx = x - x0
        fy = y - y0
        for dx, wx in ((0, 1 - fx), (1, fx)):
            for dy, wy in ((0, 1 - fy), (1, fy)):
                xi, yi = x0 + dx, y0 + dy
                if 0 <= xi < npix_x and 0 <= yi < npix_y and wx * wy > 0:
                    image[yi, xi] += flux * wx * wy


def make_true_point_source_image(npix=401, total_flux=1.0):
    """Noiseless image of a single true point source at the exact center
    pixel."""
    image = np.zeros((npix, npix), dtype=np.float64)
    xc = yc = (npix - 1) / 2.0
    _place_sources_bilinear(image, [xc], [yc], [total_flux])
    return image


def make_true_cluster_image(npix=401, pixel_scale_arcsec=0.04, total_flux=1.0,
                             n_sources=6, radius_range_arcsec=(0.1, 0.3),
                             seed=20260911, member_fluxes=None):
    """Noiseless image of n_sources point sources at random (seeded), tight
    positions around the image center, mimicking an unresolved cluster core.

    Returns (image, xs_px, ys_px, fluxes) — the true injected positions and
    per-source fluxes, for stashing in the output FITS header.
    """
    rng = np.random.default_rng(seed)
    xc = yc = (npix - 1) / 2.0

    radii_arcsec = rng.uniform(radius_range_arcsec[0], radius_range_arcsec[1],
                                size=n_sources)
    angles = rng.uniform(0.0, 2 * np.pi, size=n_sources)
    radii_px = radii_arcsec / pixel_scale_arcsec
    xs = xc + radii_px * np.cos(angles)
    ys = yc + radii_px * np.sin(angles)

    if member_fluxes is None:
        fluxes = np.full(n_sources, total_flux / n_sources)
    else:
        fluxes = np.asarray(member_fluxes, dtype=np.float64)
        if fluxes.shape[0] != n_sources:
            raise ValueError("member_fluxes must have length n_sources")

    image = np.zeros((npix, npix), dtype=np.float64)
    _place_sources_bilinear(image, xs, ys, fluxes)
    return image, xs, ys, fluxes


def make_compact_cluster_positions(npix, pixel_scale_arcsec=0.04, total_flux=1.0,
                                   n_sources=7, max_radius_arcsec=0.04, seed=20260911):
    """Simple compact cluster: one star at the image center plus
    n_sources-1 stars placed uniformly over a disk of radius
    max_radius_arcsec (r = R*sqrt(u), so the center is filled — no ring/hole).
    Equal fluxes. Returns (xs_px, ys_px, fluxes)."""
    rng = np.random.default_rng(seed)
    xc = yc = (npix - 1) / 2.0
    n_out = n_sources - 1
    r = max_radius_arcsec * np.sqrt(rng.uniform(0.0, 1.0, n_out)) / pixel_scale_arcsec
    phi = rng.uniform(0.0, 2 * np.pi, n_out)
    xs = np.concatenate([[xc], xc + r * np.cos(phi)])
    ys = np.concatenate([[yc], yc + r * np.sin(phi)])
    fluxes = np.full(n_sources, total_flux / n_sources)
    return xs, ys, fluxes


def convolve_to_fwhm(true_image, target_fwhm_arcsec, pixel_scale_arcsec=0.04):
    """Convolve a noiseless true-source image with a circular Gaussian PSF
    of the given FWHM (arcsec). target_fwhm_arcsec=None returns the true
    (unconvolved) image unchanged."""
    if target_fwhm_arcsec is None:
        return true_image.copy()

    sigma_px = target_fwhm_arcsec / pixel_scale_arcsec * FWHM_TO_SIGMA
    kernel = Gaussian2DKernel(sigma_px)
    kernel.normalize()
    return convolve_fft(true_image, kernel, normalize_kernel=True,
                         boundary='fill', fill_value=0.0, allow_huge=True)


def add_noise(image, noise_std=0.0, bkg_level=0.0, seed=None):
    """Optionally add a constant background plus Gaussian noise. A no-op
    copy when both noise_std and bkg_level are 0 (the default)."""
    if noise_std <= 0 and bkg_level == 0:
        return image.copy()
    rng = np.random.default_rng(seed)
    noisy = image + bkg_level
    if noise_std > 0:
        noisy = noisy + rng.normal(0.0, noise_std, size=image.shape)
    return noisy


def build_fits_hdu(image, wcs, *, pixel_scale_arcsec, fwhm_arcsec, source_type,
                    n_sources, total_flux, seed=None, noise_std=0.0,
                    bkg_level=0.0, src_positions=None):
    """Assemble a PrimaryHDU with WCS header plus informative keywords
    describing how the image was generated."""
    header = wcs.to_header()
    header['PIXSCALE'] = (pixel_scale_arcsec, 'arcsec/pixel')
    header['FWHM_AS'] = (fwhm_arcsec if fwhm_arcsec is not None else 0.0,
                          'Gaussian PSF FWHM in arcsec (0 = unconvolved)')
    header['CONVOLVD'] = (fwhm_arcsec is not None, 'Convolved with Gaussian PSF')
    header['SRCTYPE'] = (source_type, 'point or cluster')
    header['NSOURCES'] = (n_sources, 'Number of injected point sources')
    header['TOTFLUX'] = (total_flux, 'Total injected flux, arbitrary units')
    header['SEED'] = (seed if seed is not None else 'NA', 'RNG seed for source positions')
    header['NOISESTD'] = (noise_std, 'Gaussian noise sigma added (0 = none)')
    header['BKGLEVEL'] = (bkg_level, 'Constant background added')

    if src_positions is not None:
        xs, ys, fluxes = src_positions
        for i, (x, y, f) in enumerate(zip(xs, ys, fluxes), start=1):
            header[f'SRC{i}X'] = (float(x), f'Source {i} x pixel position')
            header[f'SRC{i}Y'] = (float(y), f'Source {i} y pixel position')
            header[f'SRC{i}F'] = (float(f), f'Source {i} injected flux')

    header.add_history('Generated by synthetic_psf_tests.py')
    return fits.PrimaryHDU(data=image.astype(np.float32), header=header)


def write_fits(hdu, path):
    hdu.writeto(path, overwrite=True)
    print(f"Wrote {path}")


def generate_all_test_images(output_dir="output", pixel_scale_arcsec=0.04,
                              npix=401, total_flux=1.0,
                              fwhm_grid_arcsec=DEFAULT_FWHM_GRID_ARCSEC,
                              cluster_seed=20260911,
                              radius_range_arcsec=(0.1, 0.3),
                              noise_std=0.0, bkg_level=0.0):
    """Generate the full set of 16 synthetic FITS test images (8 point
    source + 8 cluster) and write them to output_dir. Returns the list of
    written file paths."""
    os.makedirs(output_dir, exist_ok=True)
    wcs = build_test_wcs(pixel_scale_arcsec=pixel_scale_arcsec, npix=npix)

    true_point = make_true_point_source_image(npix=npix, total_flux=total_flux)
    true_cluster, cxs, cys, cfluxes = make_true_cluster_image(
        npix=npix, pixel_scale_arcsec=pixel_scale_arcsec, total_flux=total_flux,
        n_sources=6, radius_range_arcsec=radius_range_arcsec, seed=cluster_seed,
    )

    sources = {
        'pointsource': dict(true_image=true_point, n_sources=1, seed=None,
                             src_positions=None),
        'cluster': dict(true_image=true_cluster, n_sources=6, seed=cluster_seed,
                         src_positions=(cxs, cys, cfluxes)),
    }

    fwhm_steps = (None,) + tuple(fwhm_grid_arcsec)
    written_paths = []

    for source_type, info in sources.items():
        for fwhm in fwhm_steps:
            image = convolve_to_fwhm(info['true_image'], fwhm,
                                      pixel_scale_arcsec=pixel_scale_arcsec)
            image = add_noise(image, noise_std=noise_std, bkg_level=bkg_level,
                               seed=info['seed'])
            hdu = build_fits_hdu(
                image, wcs, pixel_scale_arcsec=pixel_scale_arcsec,
                fwhm_arcsec=fwhm, source_type=source_type,
                n_sources=info['n_sources'], total_flux=total_flux,
                seed=info['seed'], noise_std=noise_std, bkg_level=bkg_level,
                src_positions=info['src_positions'],
            )
            suffix = 'unconvolved' if fwhm is None else f'fwhm{fwhm:.2f}'
            path = os.path.join(output_dir, f'{source_type}_{suffix}.fits')
            write_fits(hdu, path)
            written_paths.append(path)

    return written_paths


def generate_all_hst_psf_test_images(output_dir="output_hst", bands=PHANGS_HST_BANDS,
                                      pixel_scale_arcsec=0.04, npix=101, total_flux=1.0,
                                      intrinsic_fwhm_grid_arcsec=DEFAULT_INTRINSIC_FWHM_GRID_ARCSEC,
                                      drizzle_fwhm_arcsec=None,
                                      cluster_seed=20260911,
                                      radius_range_arcsec=(0.04, 0.08),
                                      noise_std=0.0, bkg_level=0.0, n_fine=4):
    """Same point-source / 6-star-cluster test set as generate_all_test_images,
    but imaged through the model HST PSF of each (instrument, band) in
    `bands` instead of a pure Gaussian.

    The image of each source is

        true sources  (x)  Gaussian(intrinsic FWHM)  (x)  HST PSF  (x)  Gaussian(drizzle FWHM)

    intrinsic_fwhm_grid_arcsec : cluster light-profile sizes to step through
        (0 = the sources themselves are unresolved).
    drizzle_fwhm_arcsec : extra instrumental blur. None -> per-band value from
        HST_DRIZZLE_FWHM_ARCSEC (matches real PHANGS-HST stars); 0 -> bare
        model PSF; a float -> that value for every band.

    Files: <output_dir>/<source>_<instrument>_<band>_intr<FWHM>.fits.
    Returns the list of written paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    wcs = build_test_wcs(pixel_scale_arcsec=pixel_scale_arcsec, npix=npix)

    xc = yc = (npix - 1) / 2.0
    _, cxs, cys, cfluxes = make_true_cluster_image(
        npix=npix, pixel_scale_arcsec=pixel_scale_arcsec, total_flux=total_flux,
        n_sources=6, radius_range_arcsec=radius_range_arcsec, seed=cluster_seed,
    )
    sources = {
        'pointsource': dict(xs=[xc], ys=[yc], fluxes=[total_flux], n_sources=1,
                            seed=None, src_positions=None),
        'cluster': dict(xs=cxs, ys=cys, fluxes=cfluxes, n_sources=6,
                        seed=cluster_seed, src_positions=(cxs, cys, cfluxes)),
    }

    written_paths = []
    for instrument, band in bands:
        psf, psf_scale = load_hst_psf(instrument, band)
        if drizzle_fwhm_arcsec is None:
            drz = HST_DRIZZLE_FWHM_ARCSEC.get((instrument, band), 0.0)
        else:
            drz = float(drizzle_fwhm_arcsec)

        for source_type, info in sources.items():
            for intr in intrinsic_fwhm_grid_arcsec:
                image = render_with_hst_psf(
                    info['xs'], info['ys'], info['fluxes'], npix, pixel_scale_arcsec,
                    psf, psf_scale, drizzle_fwhm_arcsec=drz,
                    intrinsic_fwhm_arcsec=intr, n_fine=n_fine,
                )
                image = add_noise(image, noise_std=noise_std, bkg_level=bkg_level,
                                  seed=info['seed'])
                hdu = build_fits_hdu(
                    image, wcs, pixel_scale_arcsec=pixel_scale_arcsec,
                    fwhm_arcsec=intr, source_type=source_type,
                    n_sources=info['n_sources'], total_flux=total_flux,
                    seed=info['seed'], noise_std=noise_std, bkg_level=bkg_level,
                    src_positions=info['src_positions'],
                )
                hdr = hdu.header
                hdr['FWHM_AS'] = (intr, 'Intrinsic Gaussian FWHM, arcsec (0 = none)')
                hdr['CONVOLVD'] = (True, 'Imaged through model HST PSF')
                hdr['PSFINST'] = (instrument.upper(), 'Model PSF instrument')
                hdr['PSFBAND'] = (band, 'Model PSF filter')
                hdr['DRZFWHM'] = (drz, 'Extra drizzle/jitter Gaussian FWHM, arcsec')
                hdr.add_history('Rendered with instrument_psf.render_with_hst_psf')

                name = f'{source_type}_{instrument}_{band}_intr{intr:.2f}.fits'
                path = os.path.join(output_dir, name)
                write_fits(hdu, path)
                written_paths.append(path)

    return written_paths


def generate_all_jwst_psf_test_images(output_dir="output_jwst", bands=PHANGS_JWST_BANDS,
                                       field_arcsec=None, total_flux=1.0,
                                       intrinsic_fwhm_grid_arcsec=DEFAULT_INTRINSIC_FWHM_GRID_ARCSEC,
                                       extra_fwhm_arcsec=None,
                                       cluster_seed=20260911,
                                       radius_range_arcsec=(0.04, 0.08),
                                       noise_std=0.0, bkg_level=0.0, n_fine=4):
    """Same point-source / 6-star-cluster test set as
    generate_all_hst_psf_test_images, imaged through the stpsf JWST PSF of
    each band at that band's PHANGS mosaic pixel scale
    (instrument_psf.JWST_PIXEL_SCALE_ARCSEC):

        true sources  (x)  Gaussian(intrinsic FWHM)  (x)  stpsf PSF  (x)  Gaussian(extra FWHM)

    The cluster members sit at the same physical radii (arcsec) in every
    band, so in pixels the cluster is ~3.6x smaller for MIRI than F200W.

    field_arcsec : image side in arcsec; None -> 4" for NIRCam, 10" for MIRI
        (F2100W's FWHM is ~0.7"). npix is made odd.
    extra_fwhm_arcsec : None -> instrument_psf.JWST_EXTRA_FWHM_ARCSEC per band;
        0 -> bare stpsf PSF; a float -> that value for every band.

    Light falling outside the image (and outside the stpsf field of view) is
    lost, so an image sums to slightly < total_flux: total_flux is the true
    total.

    Files: <output_dir>/<source>_<band>_intr<FWHM>.fits. Returns the paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    written_paths = []
    for band in bands:
        band = band.upper()
        pix = JWST_PIXEL_SCALE_ARCSEC[band]
        model = InstrumentPSF.from_jwst(band, pixel_scale_arcsec=pix,
                                        extra_fwhm_arcsec=extra_fwhm_arcsec, n_fine=n_fine)
        field = field_arcsec if field_arcsec is not None else (10.0 if model.name.startswith('JWST/MIRI') else 4.0)
        npix = int(round(field / pix)) // 2 * 2 + 1
        wcs = build_test_wcs(pixel_scale_arcsec=pix, npix=npix)

        xc = yc = (npix - 1) / 2.0
        _, cxs, cys, cfluxes = make_true_cluster_image(
            npix=npix, pixel_scale_arcsec=pix, total_flux=total_flux,
            n_sources=6, radius_range_arcsec=radius_range_arcsec, seed=cluster_seed,
        )
        sources = {
            'pointsource': dict(xs=[xc], ys=[yc], fluxes=[total_flux], n_sources=1,
                                seed=None, src_positions=None),
            'cluster': dict(xs=cxs, ys=cys, fluxes=cfluxes, n_sources=6,
                            seed=cluster_seed, src_positions=(cxs, cys, cfluxes)),
        }
        for source_type, info in sources.items():
            for intr in intrinsic_fwhm_grid_arcsec:
                kernel = model.kernel(intr, half_size_fine=npix * n_fine)
                image = render_sources(info['xs'], info['ys'], info['fluxes'], (npix, npix),
                                       pix, kernel, n_fine)
                image = add_noise(image, noise_std=noise_std, bkg_level=bkg_level,
                                  seed=info['seed'])
                hdu = build_fits_hdu(
                    image, wcs, pixel_scale_arcsec=pix,
                    fwhm_arcsec=intr, source_type=source_type,
                    n_sources=info['n_sources'], total_flux=total_flux,
                    seed=info['seed'], noise_std=noise_std, bkg_level=bkg_level,
                    src_positions=info['src_positions'],
                )
                hdr = hdu.header
                hdr['FWHM_AS'] = (intr, 'Intrinsic Gaussian FWHM, arcsec (0 = none)')
                hdr['CONVOLVD'] = (True, 'Imaged through stpsf JWST PSF')
                hdr['PSFINST'] = (model.name.split()[0].split('/')[1].upper(), 'Model PSF instrument')
                hdr['PSFBAND'] = (band, 'Model PSF filter')
                hdr['EXTFWHM'] = (model.extra_fwhm_arcsec, 'Extra Gaussian FWHM, arcsec')
                hdr.add_history('Rendered with instrument_psf.InstrumentPSF.from_jwst')

                name = f'{source_type}_{band}_intr{intr:.2f}.fits'
                path = os.path.join(output_dir, name)
                write_fits(hdu, path)
                written_paths.append(path)
    return written_paths


# Typical (rounded) final resolution per band, FWHM in arcsec.
TYPICAL_FWHM_ARCSEC = {
    ('wfc3', 'F275W'): 0.1, ('wfc3', 'F336W'): 0.1, ('wfc3', 'F438W'): 0.1,
    ('wfc3', 'F555W'): 0.1, ('wfc3', 'F814W'): 0.1,
    ('acs', 'F435W'): 0.1, ('acs', 'F814W'): 0.1,
    ('jwst', 'F200W'): 0.1, ('jwst', 'F300M'): 0.1, ('jwst', 'F335M'): 0.1,
    ('jwst', 'F360M'): 0.1,
    ('jwst', 'F770W'): 0.3, ('jwst', 'F1000W'): 0.3, ('jwst', 'F1130W'): 0.4,
    ('jwst', 'F2100W'): 0.7,
}


def measure_model_fwhm(model):
    """FWHM (arcsec) of an InstrumentPSF point source: twice the radius at
    which the azimuthally averaged profile of the pixel-integrated PSF
    (fine-grid kernel convolved with one output pixel) drops to half its
    peak. A direct half-maximum width, not a Moffat fit: the Moffat fit
    (psf_aux_func.fit_2d_moffat) fails for the wing-heavy F200W/MIRI models
    unless alpha is fixed, and switching between free and fixed alpha makes
    the measured FWHM jump by ~10%."""
    n = model.n_fine
    half_fine = int(np.ceil(3 * 0.7 / model.fine_scale_arcsec))
    k = model.kernel(0.0, half_size_fine=half_fine)
    k = convolve_fft(k, np.ones((n, n)) / n ** 2, normalize_kernel=False,
                     boundary='fill', fill_value=0.0)
    c = (k.shape[0] - 1) / 2.0
    yy, xx = np.indices(k.shape)
    r = np.hypot(xx - c, yy - c)
    idx = np.floor(r * 2).astype(int)          # half-fine-pixel radial bins
    sums = np.bincount(idx.ravel(), k.ravel())
    rsum = np.bincount(idx.ravel(), r.ravel())
    cnts = np.bincount(idx.ravel())
    ok = cnts > 0                              # some bins hold no pixel centers
    prof = sums[ok] / cnts[ok]
    rb = rsum[ok] / cnts[ok]                   # mean radius of each bin
    peak = k.max()
    i = int(np.argmax(prof < peak / 2))
    r_half = np.interp(peak / 2, [prof[i], prof[i - 1]], [rb[i], rb[i - 1]])
    return float(2 * r_half * model.fine_scale_arcsec)


def instrument_psf_at_fwhm(instrument, band, target_fwhm_arcsec, pixel_scale_arcsec=None,
                           n_fine=4):
    """
    InstrumentPSF for (instrument, band) whose point source has FWHM
    target_fwhm_arcsec (half-maximum width, measure_model_fwhm), keeping the
    band's PSF shape.

    instrument: 'wfc3' / 'acs' (HST pickles) or 'jwst' (stpsf).
    If the bare model is sharper than the target, an extra Gaussian is added
    (method 'blur'). If it is already broader (e.g. NIRCam F300M-F360M at
    0.1"), blurring cannot help, so the PSF is shrunk in angle instead
    (method 'shrink': the same diffraction pattern at a proportionally
    smaller scale, no extra Gaussian).

    Returns (model, info) with info = dict(method, bare_fwhm, extra_fwhm,
    shrink, fwhm).
    """
    from scipy.optimize import brentq
    from instrument_psf import load_jwst_psf

    instrument = instrument.lower()
    band = band.upper()
    if instrument == 'jwst':
        camera, psf, psf_scale, total = load_jwst_psf(band)
        pix = pixel_scale_arcsec or JWST_PIXEL_SCALE_ARCSEC[band]
        name = f'JWST/{camera} {band}'
    else:
        psf, psf_scale = load_hst_psf(instrument, band)
        total = 1.0
        pix = pixel_scale_arcsec or 0.04
        name = f'HST/{instrument.upper()} {band}'

    def make(extra, shrink):
        return InstrumentPSF(psf, psf_scale * shrink, pix, extra_fwhm_arcsec=extra,
                             n_fine=n_fine, total_flux=total, name=name)

    bare = measure_model_fwhm(make(0.0, 1.0))
    if bare <= target_fwhm_arcsec:
        extra = brentq(lambda e: measure_model_fwhm(make(e, 1.0)) - target_fwhm_arcsec,
                       0.0, 3 * target_fwhm_arcsec, xtol=1e-4)
        shrink, method = 1.0, 'blur'
    else:
        shrink = brentq(lambda k: measure_model_fwhm(make(0.0, k)) - target_fwhm_arcsec,
                        0.2, 1.0, xtol=1e-4)
        extra, method = 0.0, 'shrink'
    model = make(extra, shrink)
    return model, dict(method=method, bare_fwhm=bare, extra_fwhm=extra, shrink=shrink,
                       fwhm=measure_model_fwhm(model))


def generate_typical_resolution_test_images(output_dir="output_typical",
                                            targets=TYPICAL_FWHM_ARCSEC,
                                            total_flux=1.0,
                                            intrinsic_fwhm_grid_arcsec=(0.0,),
                                            cluster_seed=20260911,
                                            radius_range_arcsec=(0.04, 0.08),
                                            compact_n_sources=7,
                                            compact_max_radius_arcsec=0.04,
                                            noise_std=0.0, bkg_level=0.0, n_fine=4,
                                            save_true=True, true_pixel_scale_arcsec=0.01):
    """
    Point source + 6-star ring cluster ('cluster', stars at
    radius_range_arcsec) + compact cluster ('compactcluster': one star at the
    center and compact_n_sources-1 more spread uniformly over a disk of
    radius compact_max_radius_arcsec, see make_compact_cluster_positions)
    for every band in `targets`, imaged
    through that band's PSF set to its typical resolution
    (instrument_psf_at_fwhm). HST at 0.04"/pix, JWST at its PHANGS mosaic
    scale; image side 4" (10" for MIRI). intrinsic_fwhm_grid_arcsec adds a
    Gaussian source size on top (default: unresolved sources only).

    Files: <output_dir>/<source>_<band>_res<FWHM>[_intr<FWHM>].fits; headers
    carry RESFWHM (target), MEASFWHM, RESMETH ('blur'/'shrink'), EXTFWHM,
    SHRINK. Returns (paths, {band: info}).

    save_true : also write the UNCONVOLVED truth of each source,
        <output_dir>/true_<source>.fits, on a true_pixel_scale_arcsec grid
        (default 0.01", fine enough to see the individual stars; each star
        is split bilinearly over its 4 nearest pixels). Same 4" field and
        same star positions in arcsec as the band images.
    """
    os.makedirs(output_dir, exist_ok=True)
    written_paths, infos = [], {}
    if save_true:
        tp = true_pixel_scale_arcsec
        tn = int(round(4.0 / tp)) // 2 * 2 + 1
        tc = (tn - 1) / 2.0
        _, rx, ry, rf = make_true_cluster_image(
            npix=tn, pixel_scale_arcsec=tp, total_flux=total_flux, n_sources=6,
            radius_range_arcsec=radius_range_arcsec, seed=cluster_seed)
        kx, ky, kf = make_compact_cluster_positions(
            tn, pixel_scale_arcsec=tp, total_flux=total_flux, n_sources=compact_n_sources,
            max_radius_arcsec=compact_max_radius_arcsec, seed=cluster_seed)
        twcs = build_test_wcs(pixel_scale_arcsec=tp, npix=tn)
        for source_type, (xs, ys, fl) in {'pointsource': ([tc], [tc], [total_flux]),
                                          'cluster': (rx, ry, rf),
                                          'compactcluster': (kx, ky, kf)}.items():
            img = np.zeros((tn, tn))
            _place_sources_bilinear(img, xs, ys, fl)
            hdu = build_fits_hdu(img, twcs, pixel_scale_arcsec=tp, fwhm_arcsec=None,
                                 source_type=source_type, n_sources=len(fl),
                                 total_flux=total_flux, seed=cluster_seed,
                                 src_positions=None if source_type == 'pointsource' else (xs, ys, fl))
            path = os.path.join(output_dir, f'true_{source_type}.fits')
            write_fits(hdu, path)
            written_paths.append(path)
    for (instrument, band), target in targets.items():
        model, info = instrument_psf_at_fwhm(instrument, band, target, n_fine=n_fine)
        infos[band if instrument == 'jwst' else f'{instrument}_{band}'] = info
        pix = model.pixel_scale_arcsec
        field = 10.0 if 'MIRI' in model.name else 4.0
        npix = int(round(field / pix)) // 2 * 2 + 1
        wcs = build_test_wcs(pixel_scale_arcsec=pix, npix=npix)
        xc = yc = (npix - 1) / 2.0
        _, cxs, cys, cfluxes = make_true_cluster_image(
            npix=npix, pixel_scale_arcsec=pix, total_flux=total_flux,
            n_sources=6, radius_range_arcsec=radius_range_arcsec, seed=cluster_seed,
        )
        sources = {
            'pointsource': dict(xs=[xc], ys=[yc], fluxes=[total_flux], n_sources=1,
                                seed=None, src_positions=None),
            'cluster': dict(xs=cxs, ys=cys, fluxes=cfluxes, n_sources=6,
                            seed=cluster_seed, src_positions=(cxs, cys, cfluxes)),
        }
        kxs, kys, kfl = make_compact_cluster_positions(
            npix, pixel_scale_arcsec=pix, total_flux=total_flux,
            n_sources=compact_n_sources, max_radius_arcsec=compact_max_radius_arcsec,
            seed=cluster_seed,
        )
        sources['compactcluster'] = dict(xs=kxs, ys=kys, fluxes=kfl, n_sources=compact_n_sources,
                                         seed=cluster_seed, src_positions=(kxs, kys, kfl))
        tag = band if instrument == 'jwst' else f'{instrument}_{band}'
        for source_type, src in sources.items():
            for intr in intrinsic_fwhm_grid_arcsec:
                kernel = model.kernel(intr, half_size_fine=npix * n_fine)
                image = render_sources(src['xs'], src['ys'], src['fluxes'], (npix, npix),
                                       pix, kernel, n_fine)
                image = add_noise(image, noise_std=noise_std, bkg_level=bkg_level,
                                  seed=src['seed'])
                hdu = build_fits_hdu(
                    image, wcs, pixel_scale_arcsec=pix, fwhm_arcsec=intr,
                    source_type=source_type, n_sources=src['n_sources'],
                    total_flux=total_flux, seed=src['seed'], noise_std=noise_std,
                    bkg_level=bkg_level, src_positions=src['src_positions'],
                )
                hdr = hdu.header
                hdr['FWHM_AS'] = (intr, 'Intrinsic Gaussian FWHM, arcsec (0 = none)')
                hdr['CONVOLVD'] = (True, 'Imaged through model instrument PSF')
                hdr['PSFINST'] = (model.name.split()[0], 'Model PSF instrument')
                hdr['PSFBAND'] = (band, 'Model PSF filter')
                hdr['RESFWHM'] = (target, 'Target PSF FWHM, arcsec')
                hdr['MEASFWHM'] = (round(info['fwhm'], 4), 'Measured model PSF FWHM, arcsec')
                hdr['RESMETH'] = (info['method'], 'blur = extra Gaussian, shrink = PSF scaled')
                hdr['EXTFWHM'] = (round(info['extra_fwhm'], 4), 'Extra Gaussian FWHM, arcsec')
                hdr['SHRINK'] = (round(info['shrink'], 4), 'Angular scale factor of PSF')
                hdr.add_history('Rendered with synthetic_psf_tests.instrument_psf_at_fwhm')

                suffix = '' if len(intrinsic_fwhm_grid_arcsec) == 1 and intr == 0 else f'_intr{intr:.2f}'
                name = f'{source_type}_{tag}_res{target:.2f}{suffix}.fits'
                path = os.path.join(output_dir, name)
                write_fits(hdu, path)
                written_paths.append(path)
    return written_paths, infos


# ─────────────────────────────────────────────────────────────────────────────
# Realistic star clusters: centrally concentrated EFF profile (Elson, Fall &
# Freeman 1987), Sigma(r) ∝ (1 + r²/a²)^(-gamma/2), truncated at r_max. A
# fraction of the light is in individually placed bright stars drawn from the
# same profile (power-law luminosity function), the rest in a smooth
# component (the unresolved faint stars). gamma = 4 is a Plummer profile;
# LEGUS young clusters have gamma ~ 2.5-3 (Ryon et al. 2017).
# ─────────────────────────────────────────────────────────────────────────────

ARCSEC_PER_RAD = 206264.806


def pc_to_arcsec(size_pc, distance_mpc):
    """theta["] = 206265 * R[pc] / D[pc]."""
    return ARCSEC_PER_RAD * size_pc / (distance_mpc * 1e6)


def sample_eff_radii(n, a, gamma, r_max, rng):
    """Projected radii drawn from a truncated EFF profile (inverse CDF)."""
    u = rng.uniform(0, 1, n) * _eff_cum(r_max, a, gamma)
    return a * np.sqrt((1.0 - u) ** (1.0 / (1.0 - gamma / 2.0)) - 1.0)


def make_eff_cluster_fine(npix, n_fine, pixel_scale_arcsec, r_eff_arcsec, total_flux=1.0,
                          gamma=3.0, r_max_over_reff=10.0, n_bright=150,
                          bright_fraction=0.5, lum_slope=2.0, lum_range=100.0, seed=20260911):
    """
    True (unconvolved) cluster on a grid n_fine times finer than the output
    pixels, centered on the output image's center pixel.

    Light = bright_fraction in n_bright stars + (1 - bright_fraction) smooth.
    Star luminosities follow dN/dL ∝ L^-lum_slope over [L0, lum_range*L0]
    (lum_slope=2: each decade of L carries equal light, so a handful of stars
    dominate), positions follow the same EFF profile (uniform angle).
    Total flux = total_flux exactly (everything inside r_max, which must fit
    in the image).

    Returns (fine_image, star_x_px, star_y_px, star_flux, a_arcsec) with star
    positions in OUTPUT-pixel coordinates.
    """
    rng = np.random.default_rng(seed)
    r_max = r_max_over_reff * r_eff_arcsec
    a = eff_scale_for_reff(r_eff_arcsec, gamma, r_max)
    xc = yc = (npix - 1) / 2.0
    fine_scale = pixel_scale_arcsec / n_fine
    nf = npix * n_fine
    shift = (n_fine - 1) / 2.0
    if r_max / pixel_scale_arcsec > xc - 1:
        raise ValueError('r_max does not fit in the image; enlarge it or lower r_max_over_reff')

    fine = np.zeros((nf, nf))

    # smooth component: EFF surface brightness at fine-pixel centers
    if bright_fraction < 1:
        coord = (np.arange(nf) - (xc * n_fine + shift)) * fine_scale
        xx, yy = np.meshgrid(coord, coord)
        rr = np.hypot(xx, yy)
        sb = (1.0 + (rr / a) ** 2) ** (-gamma / 2.0)
        sb[rr > r_max] = 0.0
        fine += (1.0 - bright_fraction) * total_flux * sb / sb.sum()

    # bright stars
    xs = ys = fl = np.array([])
    if n_bright > 0 and bright_fraction > 0:
        if lum_slope == 1.0:
            lum = np.exp(rng.uniform(0, np.log(lum_range), n_bright))
        else:
            e = 1.0 - lum_slope
            lum = (1.0 + rng.uniform(0, 1, n_bright) * (lum_range ** e - 1.0)) ** (1.0 / e)
        fl = bright_fraction * total_flux * lum / lum.sum()
        r = sample_eff_radii(n_bright, a, gamma, r_max, rng) / pixel_scale_arcsec
        phi = rng.uniform(0, 2 * np.pi, n_bright)
        xs = xc + r * np.cos(phi)
        ys = yc + r * np.sin(phi)
        _place_sources_bilinear(fine, xs * n_fine + shift, ys * n_fine + shift, fl)

    return fine, xs, ys, fl, a


def generate_realistic_cluster_images(output_dir="output_cluster", targets=None,
                                      r_eff_pc=(10.0,), distance_mpc=10.0, gamma=3.0,
                                      r_max_over_reff=10.0, n_bright=150, bright_fraction=0.5,
                                      lum_slope=2.0, lum_range=100.0, total_flux=1.0,
                                      seed=20260911, noise_std=0.0, bkg_level=0.0, n_fine=4,
                                      save_true=True):
    """
    EFF star clusters (make_eff_cluster_fine) of half-light radius r_eff_pc
    at distance_mpc, imaged through each band's PSF at its typical resolution
    (TYPICAL_FWHM_ARCSEC / instrument_psf_at_fwhm). Same pixel scales as
    generate_typical_resolution_test_images; the image is made large enough
    to hold the whole truncated cluster plus the PSF wings.

    The same cluster (same seed -> same stars) is used in every band.
    Files: <output_dir>/cluster_<band>_res<FWHM>_reff<R>pc.fits (+ the
    unconvolved truth at the native HST scale, truecluster_reff<R>pc.fits,
    if save_true). Returns the list of written paths.
    """
    targets = targets or TYPICAL_FWHM_ARCSEC
    os.makedirs(output_dir, exist_ok=True)
    written = []
    for reff_pc in r_eff_pc:
        reff = pc_to_arcsec(reff_pc, distance_mpc)
        r_max = r_max_over_reff * reff
        common = dict(r_eff_arcsec=reff, total_flux=total_flux, gamma=gamma,
                      r_max_over_reff=r_max_over_reff, n_bright=n_bright,
                      bright_fraction=bright_fraction, lum_slope=lum_slope,
                      lum_range=lum_range, seed=seed)

        if save_true:
            pix = 0.01
            npix = int(np.ceil(2.4 * r_max / pix)) // 2 * 2 + 1
            fine, *_ = make_eff_cluster_fine(npix, 1, pix, **common)
            hdu = fits.PrimaryHDU(fine.astype(np.float32),
                                  header=build_test_wcs(pixel_scale_arcsec=pix, npix=npix).to_header())
            hdu.header['PIXSCALE'] = (pix, 'arcsec/pixel')
            hdu.header['REFF_PC'] = (reff_pc, 'Half-light radius, pc')
            path = os.path.join(output_dir, f'truecluster_reff{reff_pc:g}pc.fits')
            write_fits(hdu, path)
            written.append(path)

        for (instrument, band), target in targets.items():
            model, info = instrument_psf_at_fwhm(instrument, band, target, n_fine=n_fine)
            pix = model.pixel_scale_arcsec
            field = max(10.0 if 'MIRI' in model.name else 4.0, 2.4 * r_max)
            npix = int(round(field / pix)) // 2 * 2 + 1
            fine, xs, ys, fl, a = make_eff_cluster_fine(npix, n_fine, pix, **common)
            kernel = model.kernel(0.0, half_size_fine=npix * n_fine)
            fine = convolve_fft(fine, kernel, normalize_kernel=False, boundary='fill',
                                fill_value=0.0, allow_huge=True)
            image = fine.reshape(npix, n_fine, npix, n_fine).sum(axis=(1, 3))
            image = add_noise(image, noise_std=noise_std, bkg_level=bkg_level, seed=seed)

            wcs = build_test_wcs(pixel_scale_arcsec=pix, npix=npix)
            hdu = build_fits_hdu(image, wcs, pixel_scale_arcsec=pix, fwhm_arcsec=0.0,
                                 source_type='cluster', n_sources=n_bright,
                                 total_flux=total_flux, seed=seed, noise_std=noise_std,
                                 bkg_level=bkg_level)
            hdr = hdu.header
            hdr['CONVOLVD'] = (True, 'Imaged through model instrument PSF')
            hdr['PSFINST'] = (model.name.split()[0], 'Model PSF instrument')
            hdr['PSFBAND'] = (band, 'Model PSF filter')
            hdr['RESFWHM'] = (target, 'Target PSF FWHM, arcsec')
            hdr['MEASFWHM'] = (round(info['fwhm'], 4), 'Measured model PSF FWHM, arcsec')
            hdr['RESMETH'] = (info['method'], 'blur = extra Gaussian, shrink = PSF scaled')
            hdr['EXTFWHM'] = (round(info['extra_fwhm'], 4), 'Extra Gaussian FWHM, arcsec')
            hdr['SHRINK'] = (round(info['shrink'], 4), 'Angular scale factor of PSF')
            hdr['CLPROF'] = ('EFF', 'Cluster profile (Elson, Fall & Freeman 1987)')
            hdr['REFF_PC'] = (reff_pc, 'Half-light radius, pc')
            hdr['DIST_MPC'] = (distance_mpc, 'Distance, Mpc')
            hdr['REFF_AS'] = (round(reff, 5), 'Half-light radius, arcsec')
            hdr['EFF_A'] = (round(a, 5), 'EFF scale radius a, arcsec')
            hdr['EFF_GAM'] = (gamma, 'EFF power-law index gamma')
            hdr['RMAX_AS'] = (round(r_max, 4), 'Truncation radius, arcsec')
            hdr['NBRIGHT'] = (n_bright, 'Individually placed stars')
            hdr['FBRIGHT'] = (bright_fraction, 'Light fraction in those stars')
            hdr.add_history('Rendered with synthetic_psf_tests.generate_realistic_cluster_images')

            tag = band if instrument == 'jwst' else f'{instrument}_{band}'
            path = os.path.join(output_dir, f'cluster_{tag}_res{target:.2f}_reff{reff_pc:g}pc.fits')
            write_fits(hdu, path)
            written.append(path)
    return written


# ─────────────────────────────────────────────────────────────────────────────
# Smooth cluster profiles (no individual stars): the cluster light as a
# continuous surface-brightness profile with a given half-light (effective)
# radius, rendered on a fine grid and then convolved with the band PSF.
#   'gaussian' : Sigma ∝ exp(-r²/2σ²)                       (σ = R_eff/√(2 ln2))
#   'eff'      : Sigma ∝ (1 + r²/a²)^(-γ/2), cut at r_max  (Elson+87; γ=4 Plummer)
#   'king'     : Sigma ∝ [1/√(1+(r/rc)²) - 1/√(1+(rt/rc)²)]², r < rt   (King 1962),
#                concentration c = log10(rt/rc)
# Each profile's scale radius is solved so that exactly half the light (within
# its truncation) falls inside R_eff.
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_SMOOTH_PROFILES = {
    'gaussian': dict(profile='gaussian'),
    'plummer': dict(profile='eff', gamma=4.0),
    'eff3': dict(profile='eff', gamma=3.0),
    'king_c1.0': dict(profile='king', c=1.0),
    'king_c1.5': dict(profile='king', c=1.5),
}


def generate_smooth_cluster_images(output_dir="output_smooth", targets=None,
                                   profiles=DEFAULT_SMOOTH_PROFILES, r_eff_pc=10.0,
                                   distance_mpc=10.0, eff_r_max_over_reff=10.0,
                                   total_flux=1.0, noise_std=0.0, bkg_level=0.0, n_fine=4,
                                   save_true=True, true_pixel_scale_arcsec=0.01,
                                   center_offset_px=None, offset_seed=20261005,
                                   max_offset_px=3.0):
    """
    Smooth clusters (one per entry of `profiles`, see DEFAULT_SMOOTH_PROFILES)
    with half-light radius r_eff_pc at distance_mpc, imaged through every
    band's PSF at its typical resolution (TYPICAL_FWHM_ARCSEC /
    instrument_psf_at_fwhm). The profile is rendered (supersampled) on the
    grid n_fine times finer than the band pixels, convolved with the PSF and
    summed into pixels — no individual stars.

    Files: <output_dir>/smooth_<name>_<band>_res<FWHM>.fits, plus the
    unconvolved profile true_smooth_<name>.fits at true_pixel_scale_arcsec.
    Headers: CLPROF, CLSCALE (σ / a / r_c, arcsec), CLRMAX, EFF_GAM / KING_C,
    REFF_PC, REFF_AS, DIST_MPC + the PSF keywords of the typical set.

    center_offset_px : where the cluster center sits relative to the image
        center pixel, in that band's pixels:
          None      -> exactly on the center pixel (default)
          (dx, dy)  -> that offset in every band
          'random'  -> per (profile, band) a seeded random offset, uniform in
                       [-max_offset_px, +max_offset_px] in x and y, so whole
                       pixels plus an arbitrary fraction (offset_seed)
        The true center (0-based pixel coordinates) is written as CLX0/CLY0
        (and the offset as CLDX/CLDY).
    Returns the list of written paths.
    """
    targets = targets or TYPICAL_FWHM_ARCSEC
    os.makedirs(output_dir, exist_ok=True)
    r_eff = pc_to_arcsec(r_eff_pc, distance_mpc)
    written = []

    shapes = {}
    for name, spec in profiles.items():
        prof = spec['profile']
        gamma, c = spec.get('gamma', 3.0), spec.get('c', 1.5)
        scale = smooth_profile_scale_for_reff(prof, r_eff, gamma, c, eff_r_max_over_reff)
        r_max = smooth_profile_truncation(prof, scale, r_eff, gamma, c, eff_r_max_over_reff)
        shapes[name] = dict(profile=prof, scale=scale, r_max=r_max, gamma=gamma, c=c)

    def add_profile_keys(hdr, name, sh):
        hdr['CLNAME'] = (name, 'Smooth cluster profile name')
        hdr['CLPROF'] = (sh['profile'], 'gaussian / eff / king')
        hdr['CLSCALE'] = (round(sh['scale'], 6), 'sigma / EFF a / King r_c, arcsec')
        hdr['CLRMAX'] = (round(sh['r_max'], 5), 'Truncation radius, arcsec')
        if sh['profile'] == 'eff':
            hdr['EFF_GAM'] = (sh['gamma'], 'EFF gamma')
        if sh['profile'] == 'king':
            hdr['KING_C'] = (sh['c'], 'King concentration log10(rt/rc)')
        hdr['REFF_PC'] = (r_eff_pc, 'Half-light radius, pc')
        hdr['REFF_AS'] = (round(r_eff, 5), 'Half-light radius, arcsec')
        hdr['DIST_MPC'] = (distance_mpc, 'Distance, Mpc')

    if save_true:
        tp = true_pixel_scale_arcsec
        for name, sh in shapes.items():
            tn = int(np.ceil(max(4.0, 2.4 * sh['r_max']) / tp)) // 2 * 2 + 1
            img = render_smooth_profile(tn, tp, sh['profile'], sh['scale'], sh['r_max'],
                                        sh['gamma'], sh['c'], total_flux)
            hdu = build_fits_hdu(img, build_test_wcs(pixel_scale_arcsec=tp, npix=tn),
                                 pixel_scale_arcsec=tp, fwhm_arcsec=None,
                                 source_type='smoothcluster', n_sources=0,
                                 total_flux=total_flux)
            add_profile_keys(hdu.header, name, sh)
            path = os.path.join(output_dir, f'true_smooth_{name}.fits')
            write_fits(hdu, path)
            written.append(path)

    rng = np.random.default_rng(offset_seed)
    for (instrument, band), target in targets.items():
        model, info = instrument_psf_at_fwhm(instrument, band, target, n_fine=n_fine)
        pix = model.pixel_scale_arcsec
        tag = band if instrument == 'jwst' else f'{instrument}_{band}'
        for name, sh in shapes.items():
            if center_offset_px is None:
                dxp, dyp = 0.0, 0.0
            elif isinstance(center_offset_px, str) and center_offset_px == 'random':
                dxp, dyp = rng.uniform(-max_offset_px, max_offset_px, 2)
            else:
                dxp, dyp = (float(v) for v in center_offset_px)
            field = max(10.0 if 'MIRI' in model.name else 4.0,
                        2.4 * sh['r_max'] + 2 * np.hypot(dxp, dyp) * pix)
            npix = int(round(field / pix)) // 2 * 2 + 1
            fine = render_smooth_profile(npix * n_fine, pix / n_fine, sh['profile'], sh['scale'],
                                         sh['r_max'], sh['gamma'], sh['c'], total_flux,
                                         center_offset_arcsec=(dxp * pix, dyp * pix))
            fine = convolve_fft(fine, model.kernel(0.0, half_size_fine=npix * n_fine),
                                normalize_kernel=False, boundary='fill', fill_value=0.0,
                                allow_huge=True)
            image = fine.reshape(npix, n_fine, npix, n_fine).sum(axis=(1, 3))
            image = add_noise(image, noise_std=noise_std, bkg_level=bkg_level)

            hdu = build_fits_hdu(image, build_test_wcs(pixel_scale_arcsec=pix, npix=npix),
                                 pixel_scale_arcsec=pix, fwhm_arcsec=0.0,
                                 source_type='smoothcluster', n_sources=0,
                                 total_flux=total_flux, noise_std=noise_std,
                                 bkg_level=bkg_level)
            hdr = hdu.header
            hdr['CONVOLVD'] = (True, 'Imaged through model instrument PSF')
            hdr['PSFINST'] = (model.name.split()[0], 'Model PSF instrument')
            hdr['PSFBAND'] = (band, 'Model PSF filter')
            hdr['RESFWHM'] = (target, 'Target PSF FWHM, arcsec')
            hdr['MEASFWHM'] = (round(info['fwhm'], 4), 'Measured model PSF FWHM, arcsec')
            hdr['RESMETH'] = (info['method'], 'blur = extra Gaussian, shrink = PSF scaled')
            hdr['EXTFWHM'] = (round(info['extra_fwhm'], 4), 'Extra Gaussian FWHM, arcsec')
            hdr['SHRINK'] = (round(info['shrink'], 4), 'Angular scale factor of PSF')
            add_profile_keys(hdr, name, sh)
            xc = (npix - 1) / 2.0
            hdr['CLX0'] = (xc + dxp, 'True cluster center x (0-based pixel)')
            hdr['CLY0'] = (xc + dyp, 'True cluster center y (0-based pixel)')
            hdr['CLDX'] = (dxp, 'Center offset from image center x, pixels')
            hdr['CLDY'] = (dyp, 'Center offset from image center y, pixels')
            hdr.add_history('Rendered with synthetic_psf_tests.generate_smooth_cluster_images')
            path = os.path.join(output_dir, f'smooth_{name}_{tag}_res{target:.2f}.fits')
            write_fits(hdu, path)
            written.append(path)
    return written
