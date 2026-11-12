import numpy as np
from astropy import units as u
from astropy import constants as const

from astropy.stats import sigma_clipped_stats

from regions import CircleSkyRegion
from astropy.coordinates import SkyCoord

# ─────────────────────────────────────────────────────────────────────────────
# Helpers (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────

def to_Jy(wl, spec):
    with_units = spec * 1e-20 * u.erg / (u.AA * u.s * u.cm**2)
    wl_2 = (np.e**(wl) * u.AA)**2
    return (with_units * wl_2 / const.c).to('mJy')


def find_id(x, y, original, binned):
    """Return (original_id, binned_id) for a given pixel position."""
    return int(original[y, x]), int(binned[y, x])


def corr_flux(flux, r, eb_v):
    """Apply reddening correction."""
    return flux * 10**(0.4 * r * eb_v)


def jwst_pixel_sum_to_Jy(raw_sum, header):
    """A raw (dimensionless-in-file-units) JWST pixel-value sum -> Jy.

    JWST image products come from two different processing conventions,
    both seen across the datasets this pipeline has used:
    - Photometrically CALIBRATED products (BUNIT='MJy/sr' — the case for
      the PHANGS-JWST anchor/HLSP releases, e.g. v4p1/2107): pixel values
      already ARE MJy/sr, so raw_sum * PIXAR_SR alone gives MJy.
    - Un-calibrated Stage2/3 pipeline products still in DN/s: PHOTMJSR
      (the DN/s -> MJy/sr photometric conversion factor) must be applied
      first, i.e. raw_sum * PHOTMJSR * PIXAR_SR.

    Deciding from the header's own BUNIT rather than assuming one
    convention matters: blindly always multiplying by PHOTMJSR silently
    double-applies the calibration on already-MJy/sr data — every native
    JWST flux comes out suppressed by a spurious extra factor of
    ~PHOTMJSR (order 0.1-1, band-dependent) — which is exactly what
    happened when this pipeline's JWST source switched to the v4p1/2107
    anchor release without this check.
    """
    bunit = str(header.get('BUNIT', '')).strip().upper()
    already_mjy_sr = bunit in ('MJY/SR', 'MJY SR-1', 'MJY.SR**-1', 'MJY/SR2')
    factor = header['PIXAR_SR'] if already_mjy_sr else header['PHOTMJSR'] * header['PIXAR_SR']
    return (raw_sum * factor * u.MJy).to(u.Jy).value


# ─────────────────────────────────────────────────────────────────────────────
# Mask generation from a PolygonPixelRegion
# ─────────────────────────────────────────────────────────────────────────────

def _bbox_local_mask_and_image(region, wcs, image, ref_bbox):
    """
    Fractional pixel mask for ``region`` (on ``wcs``) and the corresponding
    image data, both cropped to ``ref_bbox`` instead of allocating a
    full-image-sized array.

    This is the bbox-local replacement for the old
    ``region.to_pixel(wcs).to_mask().to_image(image.shape)`` pattern, which
    allocated an array the size of the *entire* image for what is typically a
    source aperture only a few tens of pixels across. Numerically identical
    to the old approach (validated against it on synthetic data, including
    sources near/straddling the image edge) — the only difference is that
    pixels outside ``ref_bbox`` are simply not represented, since both inputs
    to every downstream sum/mask operation are zero there anyway.

    ``ref_bbox`` lets two different regions (e.g. a source and its larger
    background annulus) share one common local coordinate frame so their
    masks can be combined afterwards (e.g. ``annulus_mask & ~source_mask``).
    Pass the *larger* region's own ``to_pixel(wcs).to_mask().bbox`` as
    ``ref_bbox`` so the smaller region's footprint is always fully contained.

    Parameters
    ----------
    region   : SkyRegion or PixelRegion
    wcs      : astropy WCS of the target image
    image    : 2-D ndarray — the full image ``region`` is measured on
    ref_bbox : regions.RegionBoundingBox — common local frame to crop to

    Returns
    -------
    img_local  : 2-D ndarray, shape (ref_bbox.iymax-iymin, ref_bbox.ixmax-ixmin)
    mask_local : 2-D float ndarray, same shape — fractional pixel coverage
    """
    ny = ref_bbox.iymax - ref_bbox.iymin
    nx = ref_bbox.ixmax - ref_bbox.ixmin
    mask_local = np.zeros((ny, nx), dtype=float)
    img_local  = np.zeros((ny, nx), dtype=float)

    pix_mask = region.to_pixel(wcs).to_mask()
    sl_large, sl_small = pix_mask.get_overlap_slices(image.shape)
    if sl_large is None:
        return img_local, mask_local   # region entirely outside the image

    dy = pix_mask.bbox.iymin - ref_bbox.iymin
    dx = pix_mask.bbox.ixmin - ref_bbox.ixmin
    ys, xs = sl_small
    ref_ys = slice(ys.start + dy, ys.stop + dy)
    ref_xs = slice(xs.start + dx, xs.stop + dx)

    mask_local[ref_ys, ref_xs] = pix_mask.data[sl_small]
    img_local[ref_ys, ref_xs]  = image[sl_large]

    return img_local, mask_local


# ─────────────────────────────────────────────────────────────────────────────
# Photometry
# ─────────────────────────────────────────────────────────────────────────────

def measure_photometry(region, filter_wavelengths, hst_images,
                       jwst_images, eb_v):
    """
    Sum flux inside aperture masks for all HST and JWST bands.

    Parameters
    ----------
    mask_hst          : 2-D bool ndarray — aperture on the HST/convolved grid
    mask_jwst         : 2-D bool ndarray — aperture on the JWST grid
    filter_wavelengths: dict  band → (central_wl_AA, R_value)
                        First 5 entries are HST bands, remainder are JWST.
    hst_images        : dict  band → (data_array, ...)
    jwst_images       : dict  band → (data_array, header, wcs)
    eb_v              : float  E(B-V) reddening value for this source

    Returns
    -------
    central_wl : list[float]   central wavelengths in Angstrom
    Fnu_bands  : list[float]   dust-corrected fluxes in mJy
    """
    central_wl = []
    Fnu_bands  = []
    Fnu_wback_bands = []

    annulus_region = radius_circle(region)

    bands = list(filter_wavelengths.items())

    # ── HST bands (first 5 entries) ──────────────────────────────────────────
    for band, (cwl, r_value) in bands[:5]:
        conv_image = hst_images[band.lower()][0]
        conv_wcs = hst_images[band.lower()][2]

        ref_bbox = annulus_region.to_pixel(conv_wcs).to_mask().bbox
        img_local, mask_annulus = _bbox_local_mask_and_image(annulus_region, conv_wcs, conv_image, ref_bbox)
        _, mask_hst = _bbox_local_mask_and_image(region, conv_wcs, conv_image, ref_bbox)

        corr_value, bkg_std = corr_value_func(mask_hst, mask_annulus, img_local)

        flux_wback = (np.nansum(img_local * mask_hst)) * u.Jy.to('mJy')
        flux       = (np.nansum(img_local * mask_hst) - corr_value) * u.Jy.to('mJy')

        flux_wback = corr_flux(flux_wback, r_value, eb_v)
        flux       = corr_flux(flux, r_value, eb_v)

        central_wl.append(cwl)
        Fnu_bands.append(flux)
        Fnu_wback_bands.append(flux_wback)

    # ── JWST bands (remaining entries) ───────────────────────────────────────
    for band, (cwl, r_value) in bands[5:]:
        conv_image  = jwst_images[band.lower()][0]
        hdr         = jwst_images[band.lower()][1]
        conv_wcs         = jwst_images[band.lower()][2]

        ref_bbox = annulus_region.to_pixel(conv_wcs).to_mask().bbox
        img_local, mask_annulus = _bbox_local_mask_and_image(annulus_region, conv_wcs, conv_image, ref_bbox)
        _, mask_jwst = _bbox_local_mask_and_image(region, conv_wcs, conv_image, ref_bbox)

        corr_value, bkg_std = corr_value_func(mask_jwst, mask_annulus, img_local)

        flux_wback = jwst_pixel_sum_to_Jy(np.nansum(img_local * mask_jwst), hdr) * u.Jy.to('mJy')
        flux = jwst_pixel_sum_to_Jy(np.nansum(img_local * mask_jwst) - corr_value, hdr) * u.Jy.to('mJy')

        flux_wback = corr_flux(flux_wback, r_value, eb_v)
        flux       = corr_flux(flux, r_value, eb_v)

        central_wl.append(cwl)
        Fnu_bands.append(flux)
        Fnu_wback_bands.append(flux_wback)

    return central_wl, Fnu_bands, Fnu_wback_bands, annulus_region

def radius_circle(region):

    #region has to be a SkyRegion object

    vra, vdec = region.vertices.ra, region.vertices.dec
    cra, cdec = np.mean(vra), np.mean(vdec)

    enclosing_r  = np.max(np.sqrt((vra - cra)**2 + (vdec - cdec)**2))
    circle_r     = enclosing_r * 2

    annulus_sky = CircleSkyRegion(
                                 center=SkyCoord(cra, cdec, unit='deg'),
                                 radius=circle_r
                                 )

    return annulus_sky

def corr_value_func(region_mask, region_ann_mask, image):

    region_mask = region_mask.astype(bool)

    n_source_pix = int(region_mask.sum())
    
    annulus_mask = region_ann_mask*~region_mask
    ann_values        = image*annulus_mask
    ann_values[ann_values == 0] = np.nan
    mean, median, std = sigma_clipped_stats(ann_values, sigma=3.0)
    corr_value        = median * n_source_pix

    return corr_value, std


# ─────────────────────────────────────────────────────────────────────────────
# Spectra extraction
# ─────────────────────────────────────────────────────────────────────────────

def spectra_from_aperture(region, muse_shape, muse_wcs, log_wl, spectra, id_maps):
    """
    Extract and sum spectra + compute photometry for one matched source.

    Uses ``PolygonPixelRegion`` objects stored in ``value.regions_HST`` and
    ``value.regions_MUSE`` (populated by ``load_results``).  The correct region
    for source ``idd`` is looked up via the ``_row1``/``_row2`` index columns
    saved in ``value.final_table`` by ``match_catalogs``.

    Parameters
    ----------
    value   : IsolatedSourceFinder instance (after load_results)
    idd     : int   index into value.final_table (the matched catalog)
    log_wl  : array  log-wavelength axis for the MUSE cube
    spectra : dict   MUSE spectra keyed by spaxel ID
    id_maps : tuple  (original_id_map, binned_id_map) — the two ID planes
                     previously accessed as maps[1].data and maps[5].data
    type_ap : {'HST', 'MUSE', 'HST_direct'}
        'HST'        — build masks from the HST region; reproject to MUSE/JWST
        'MUSE'       — build masks from the MUSE region; reproject to HST/JWST
        'HST_direct' — use the HST mask directly on the HST grid for spectra
                       (only valid when HST and MUSE share the same pixel grid)
    eb_v    : float  E(B-V) reddening for this source

    Returns
    -------
    mask_hst           : 2-D bool ndarray on the convolved HST grid
    mask_muse          : 2-D bool ndarray on the MUSE grid
    mask_jwst          : 2-D bool ndarray on the JWST grid
    sum_aper_spectra_o : 1-D array  summed spectra in original flux units
    sum_aper_spectra_Jy: 1-D array  summed spectra in mJy
    """

    original_id_map, binned_id_map = id_maps

    # ── look up which region corresponds to this matched-catalog row ─────────
    # _row1 / _row2 were added by match_catalogs and saved with final_table,
    # so they are available after load_results without re-running the pipeline.

    # Bbox-local mask instead of a full-muse-frame allocation: get the
    # nonzero (fractional-coverage) pixel locations of the aperture within
    # its own small bounding box, then shift them back to MUSE-frame pixel
    # coordinates for the original/binned ID-map lookup below.
    pix_mask = region.to_pixel(muse_wcs).to_mask()
    sl_large, sl_small = pix_mask.get_overlap_slices(muse_shape)

    if sl_large is None:
        y_coords = np.array([], dtype=int)
        x_coords = np.array([], dtype=int)
    else:
        local_mask = pix_mask.data[sl_small]
        y_local, x_local = np.where(local_mask)
        y_coords = y_local + sl_large[0].start
        x_coords = x_local + sl_large[1].start

    # ── sum spectra over all spaxels inside the aperture ─────────────────────
    # to_Jy() is linear in the spectrum, so summing the raw spectra first and
    # converting once is exactly equivalent to converting every pixel before
    # summing, but avoids N_pixels-1 redundant unit-conversion calls.
    aper_spectra_o = []

    for x, y in zip(x_coords, y_coords):
        id_original, _ = find_id(x, y, original_id_map, binned_id_map)
        if id_original < 0 or id_original >= len(spectra):
            # Unbinned/masked MUSEDAP spaxel (commonly flagged -1) or an
            # otherwise invalid id — exclude this pixel rather than let a
            # negative index silently wrap to an unrelated row, or crash
            # and lose the whole source's spectrum.
            continue
        aper_spectra_o.append(spectra[id_original]['SPEC'])

    sum_aper_spectra_o  = np.sum(aper_spectra_o, axis=0)
    sum_aper_spectra_Jy = to_Jy(log_wl, sum_aper_spectra_o)

    return sum_aper_spectra_o, sum_aper_spectra_Jy