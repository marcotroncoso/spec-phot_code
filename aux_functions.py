import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy  as np
import astropy.units as u
from astropy.table import Table, vstack
from astropy.visualization import ImageNormalize, AsinhStretch, LogStretch, SqrtStretch, LinearStretch
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import skycoord_to_pixel
from astropy.stats import sigma_clipped_stats
from astropy.convolution import convolve_fft, Gaussian2DKernel
from astropy.nddata import Cutout2D
from photutils.segmentation import detect_sources, SourceCatalog
from photutils.detection import find_peaks
from scipy.ndimage import binary_dilation

from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage import measure
from regions import PolygonPixelRegion, PixCoord

from collections import defaultdict

from astropy.io import ascii
import os

import logging

logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


# extras

def get_norm(data, mi, ma, st):
    vmin = np.nanpercentile(data, mi)
    vmax = np.nanpercentile(data, ma)
    if st == "linear":
        stretch = LinearStretch()
    elif st == "log":
        stretch = LogStretch()
    elif st == "asinh":
        stretch = AsinhStretch()
    elif st == "sqrt":
        stretch = SqrtStretch()
    else:
        stretch = LinearStretch()
    return ImageNormalize(vmin=vmin, vmax=vmax, stretch=stretch)

def fov_and_center(wcs):
    x0_deg, y0_deg = wcs.proj_plane_pixel_scales()
    x0_arcsec, y0_arcsec = x0_deg.to('arcsec'), y0_deg.to('arcsec')
    size_x, size_y = wcs.pixel_shape

    ra_ref, dec_ref = wcs.wcs.crval
    sky_ref = SkyCoord(ra_ref * u.deg, dec_ref * u.deg)
    
    return (x0_arcsec*size_x, y0_arcsec*size_y), sky_ref

def make_cutout(data, wcs, position, size):
    """
    Create a cutout preserving WCS.
    
    Parameters
    ----------
    data : 2D ndarray
        Image data.
    wcs : astropy.wcs.WCS
        Original WCS of the image.
    position : astropy.SkyCoord
        Center coordinates as a SkyCoord.
    size_arcsec : float or tuple
        Size of cutout in arcsec (scalar or (x,y)).
        
    Returns
    -------
    cutout_data : 2D ndarray
    cutout_wcs  : astropy.wcs.WCS
    """

    # Convert size to angular quantity
    #size = size_arcsec * u.arcsec

    cutout = Cutout2D(data, position, size, wcs=wcs, mode='trim')
    cutout_data = np.where(cutout.data==0.0, np.nan, cutout.data)
    
    return cutout_data, cutout.wcs

# PSF functions
#
# The PSF-fitting implementation lives in psf_aux_func.py (Moffat-profile
# fitter, measure_psf_fwhm_from_image / fit_2d_moffat) — that is the
# canonical PSF measurement used throughout the pipeline. This module used to
# carry a second, Gaussian-only implementation (fit_2d_gaussian /
# measure_psf_fwhm_from_image) that was never imported by analysis.py; it has
# been removed to avoid the name collision. It cannot simply be replaced with
# an import from psf_aux_func here because psf_aux_func.py itself imports
# from this module (get_norm), which would create a circular import.

# convolution

def gaussian_kernel_sigma_pixels(fwhm_from_arcsec, fwhm_to_arcsec, pixel_scale_arcsec):
    """
    Sigma (in pixels) of the Gaussian kernel needed to PSF-match an image of
    FWHM ``fwhm_from_arcsec`` to a target FWHM ``fwhm_to_arcsec``, via
    quadratic differencing: kernel_FWHM = sqrt(to^2 - from^2).

    Returns None if ``fwhm_from_arcsec >= fwhm_to_arcsec`` — the image is
    already at or below the target resolution, so no (further) convolution
    is needed or possible without deconvolution. Shared by both
    ``convolution()`` here and ``Photometry._gaussian_kernel`` in
    analysis.py so the two PSF-matching code paths can't drift apart.
    """
    if fwhm_from_arcsec >= fwhm_to_arcsec:
        return None

    fwhm_kernel_arcsec = np.sqrt(fwhm_to_arcsec**2 - fwhm_from_arcsec**2)
    fwhm_kernel_pixels = fwhm_kernel_arcsec / pixel_scale_arcsec
    return fwhm_kernel_pixels / (2 * np.sqrt(2 * np.log(2)))


def convolution(data, wcs, data_MUSE, wcs_MUSE, kernel='gaussian', fwhm = None):

    size_MUSE, positions = fov_and_center(wcs_MUSE)
    cutout, wcs_cutout = make_cutout(data, wcs, positions, size_MUSE)

    if not fwhm:
        raise ValueError(
            "convolution() requires an explicit fwhm=(fwhm_native_arcsec, fwhm_muse_arcsec) "
            "tuple. Measure both PSFs first (e.g. via psf_aux_func.measure_psf_fwhm_from_image) "
            "and pass them in — this function no longer auto-measures PSFs itself."
        )
    fwhm_arcsec, fwhm_arcsec_MUSE = fwhm

    if kernel != 'gaussian':
        raise ValueError(f"Unsupported kernel {kernel!r} — only 'gaussian' is implemented")

    pixel_scale_arcsec = wcs_cutout.proj_plane_pixel_scales()[0].to(u.arcsec).value
    sigma_kernel_pixels = gaussian_kernel_sigma_pixels(fwhm_arcsec, fwhm_arcsec_MUSE, pixel_scale_arcsec)

    if sigma_kernel_pixels is None:
        # Native PSF already at/below the MUSE PSF — nothing to convolve.
        return cutout, cutout, wcs_cutout

    kernel_gauss = Gaussian2DKernel(sigma_kernel_pixels)
    kernel_gauss.normalize()

    convolved_image = convolve_fft(cutout, kernel_gauss, normalize_kernel=True, preserve_nan=True,
                                   fill_value=np.nan, allow_huge=True)

    return convolved_image, cutout, wcs_cutout


def create_aperture_from_mask(mask):
    """
    Create aperture from mask contour.
    
    Parameters:
    -----------
    mask : array
        Binary mask or segmentation map
    label_value : int, optional
        If mask is segmentation map, which label to extract
    
    Returns:
    --------
    contour_coords : array
        (N, 2) array of (x, y) coordinates defining the contour
    """
    #if label_value is not None:
    #    binary_mask = (mask == label_value)
    #else:
    #binary_mask = mask.astype(bool)
    
    # Find contours using skimage
    contours = measure.find_contours(mask)
    
    if len(contours) == 0:
        return None
    
    # Get the largest contour (main boundary)
    contour = max(contours, key=len)
    
    # Convert from (row, col) to (x, y)
    contour_xy = np.column_stack([contour[:, 1], contour[:, 0]])
    
    return contour_xy

def segm_data(data, bkg, N_tresh = 3):

    # === Detect sources ===
    threshold = bkg.background + N_tresh * bkg.background_rms #
    segm = detect_sources(data, threshold, npixels=1)

    # === Build source catalog ===
    cat = SourceCatalog(data, segm, convolved_data=None)
    
    # 1. Find peaks (intensity maxima)
    peaks = peak_local_max(data, min_distance=5, threshold_abs=threshold)#N_tresh)

    logger.info(f"Found {len(peaks)} peaks at positions:")
    #for i, (y_peak, x_peak) in enumerate(peaks):
        #logger.info(f"  Peak {i+1}: ({x_peak}, {y_peak}), intensity = {data_Ha[y_peak, x_peak]:.1f}")

    # 2. Create markers at peaks
    markers = np.zeros(data.shape, dtype=int)
    for i, (y_peak, x_peak) in enumerate(peaks, start=1):
        markers[y_peak, x_peak] = i

    # 3. Create mask for segmentation
    mask = data > threshold#N_tresh

    # 4. Apply watershed (use negative image because we want to segment from peaks)
    labels = watershed(-data, markers, mask=mask)

    return segm, labels

def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.2f} s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.2f} min"
    hours = minutes / 60
    return f"{hours:.2f} h"

def columns_with_masked_values(table, keys):
    bad_cols = []
    for key in keys:
        col = table[key]
        if hasattr(col, "mask") and np.any(col.mask):
            bad_cols.append(key)
    return bad_cols

def find_isolated(source_name, data, wcs, bkg, tables, selection, NN = 3, plot = True):

    """
    Identify segmentation regions that contain exactly one catalog source.

    Optimised version:
      - All catalog sky coordinates are converted to pixel coordinates in a
        single vectorised call, then each source is mapped to its watershed
        label via a direct array look-up (O(N_sources) instead of the original
        O(N_labels × N_sources) double loop).
      - Binary masks are replaced by PolygonPixelRegion objects, which are
        lightweight (store only contour vertices), can produce a mask on demand
        via region.to_mask().to_image(data.shape), and can be written directly
        to a DS9 .reg file.

    Parameters
    ----------
    source_name : str
        Galaxy name (used only for plot titles).
    data : 2-D ndarray
        Image to segment (convolved HST or MUSE flux map).
    wcs : astropy.wcs.WCS
        WCS of *data*.
    bkg : photutils Background2D
        Pre-computed background object for the image.
    tables : dict
        Catalog dictionary as built by IsolatedSourceFinder.get_catalogs().
        Expected layout: tables[selection] = [all, ycl, map, ogcc].
    selection : {'machine', 'human'}
        Which classification column to use.
    NN : float
        N-sigma threshold passed to segm_data.
    plot : bool
        Whether to show diagnostic plots.

    Returns
    -------
    table_filtered  : astropy Table          — sources in isolated regions
    labels          : 2-D int array          — full watershed label map
    labels_filtered : list[int]              — label IDs that passed the filter
    regions         : list[PolygonPixelRegion] — one region per kept segment
                        • plot contour  : ax.plot(r.vertices.x, r.vertices.y)
                        • get mask      : r.to_mask().to_image(data.shape)
                        • save .reg     : Regions(regions).write('out.reg')
    cl_type         : list[str]              — CC_CLASS value per kept source
    """

    pixel_scale = wcs.proj_plane_pixel_scales()[0].to(u.arcsec)   # kept for downstream use

    ycl_catalog, map_catalog, ogcc_catalog = tables[selection][1:]
    all_sources = tables[selection][0]

    selection2 = 'ML' if selection == 'machine' else selection.upper()

    logger.info(f"Originally {len(ycl_catalog)} ycl")
    logger.info(f"Originally {len(map_catalog)} map")
    logger.info(f"Originally {len(ogcc_catalog)} ogcc")

    # ── pixel coords for the three sub-catalogs (used only in the plot) ──────
    pos_ycl  = SkyCoord(ycl_catalog['PHANGS_RA'],  ycl_catalog['PHANGS_DEC'])
    x_ycl,  y_ycl  = skycoord_to_pixel(pos_ycl,  wcs)

    pos_map  = SkyCoord(map_catalog['PHANGS_RA'],  map_catalog['PHANGS_DEC'])
    x_map,  y_map  = skycoord_to_pixel(pos_map,  wcs)

    pos_ogcc = SkyCoord(ogcc_catalog['PHANGS_RA'], ogcc_catalog['PHANGS_DEC'])
    x_ogcc, y_ogcc = skycoord_to_pixel(pos_ogcc, wcs)

    # ── watershed segmentation ───────────────────────────────────────────────
    segm, labels = segm_data(data, bkg, N_tresh=NN)

    # ── vectorised sky → pixel conversion for the full catalog ───────────────
    # One call for all sources instead of one call per (label, source) pair.
    all_pos = SkyCoord(
        all_sources['PHANGS_RA'],
        all_sources['PHANGS_DEC'],
        unit='deg'
    )
    all_x, all_y = skycoord_to_pixel(all_pos, wcs)

    # Round to nearest integer pixel
    all_xi = np.round(all_x).astype(int)
    all_yi = np.round(all_y).astype(int)

    # ── bounds check (vectorised) ────────────────────────────────────────────
    h, w = data.shape
    in_bounds = (all_xi >= 0) & (all_xi < w) & (all_yi >= 0) & (all_yi < h)

    # ── label look-up: which watershed region does each source fall in? ───────
    # Sources outside the image or in the background (label 0) are ignored.
    source_labels = np.zeros(len(all_sources), dtype=int)          # default → background
    source_labels[in_bounds] = labels[all_yi[in_bounds], all_xi[in_bounds]]

    # ── group source indices by their label ──────────────────────────────────
    label_to_src_indices = defaultdict(list)
    for src_idx, lbl in enumerate(source_labels):
        if lbl > 0:
            label_to_src_indices[lbl].append(src_idx)

    # ── filter: keep labels with exactly one source that is not 'outside' ────
    # The PolygonPixelRegion is built only for labels that pass, replacing both
    # the raw contour array and the full binary mask.
    labels_filtered = []
    sources_filtered = []
    regions = []
    cl_type = []

    for lbl, src_indices in label_to_src_indices.items():

        # Isolation criterion: exactly one catalog source per segment
        if len(src_indices) != 1:
            continue

        source = all_sources[src_indices[0]]

        # Skip sources classified as 'outside'
        if source[f'PHANGS_CC_CLASS_{selection2}'] == 'outside':
            continue

        # Build contour only for survivors, then wrap in a region object
        binary_mask = (labels == lbl).astype(float)
        contour_xy = create_aperture_from_mask(binary_mask)   # (N,2) x/y array

        region = PolygonPixelRegion(
            vertices=PixCoord(x=contour_xy[:, 0], y=contour_xy[:, 1])
        )

        labels_filtered.append(lbl)
        sources_filtered.append(source)
        regions.append(region.to_sky(wcs))
        cl_type.append(source[f'PHANGS_CC_CLASS_{selection2}'])

    # ── assemble output table ────────────────────────────────────────────────
    table_filtered = vstack(sources_filtered)

    logger.info(f"Final {len(table_filtered)} sources")

    mask_ycl  = table_filtered[f'PHANGS_CC_CLASS_{selection2}'] == 'YCL'
    mask_map  = table_filtered[f'PHANGS_CC_CLASS_{selection2}'] == 'MAP'
    mask_ogcc = table_filtered[f'PHANGS_CC_CLASS_{selection2}'] == 'OGCC'

    logger.info(f"Final {mask_ycl.sum()} ycl")
    logger.info(f"Final {mask_map.sum()} map")
    logger.info(f"Final {mask_ogcc.sum()} ogcc")

    # ── optional diagnostic plot ─────────────────────────────────────────────
    if plot:
        ycl_table_filtered  = table_filtered[mask_ycl]
        map_table_filtered  = table_filtered[mask_map]
        ogcc_table_filtered = table_filtered[mask_ogcc]

        pos_ycl_f  = SkyCoord(ycl_table_filtered['PHANGS_RA'],  ycl_table_filtered['PHANGS_DEC'])
        pos_map_f  = SkyCoord(map_table_filtered['PHANGS_RA'],  map_table_filtered['PHANGS_DEC'])
        pos_ogcc_f = SkyCoord(ogcc_table_filtered['PHANGS_RA'], ogcc_table_filtered['PHANGS_DEC'])

        x_ycl_f,  y_ycl_f  = skycoord_to_pixel(pos_ycl_f,  wcs)
        x_map_f,  y_map_f  = skycoord_to_pixel(pos_map_f,  wcs)
        x_ogcc_f, y_ogcc_f = skycoord_to_pixel(pos_ogcc_f, wcs)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 8), sharex=True, sharey=True)

        ax1.imshow(data, norm=get_norm(data, 15, 99.9, 'asinh'))
        ax1.scatter(x_ycl_f,  y_ycl_f,  c='blue',   label='YCL')
        ax1.scatter(x_map_f,  y_map_f,  c='orange', label='MAP')
        ax1.scatter(x_ogcc_f, y_ogcc_f, c='red',    label='OGCC')
        for reg in regions:
            ax1.plot(reg.vertices.x, reg.vertices.y, '-', linewidth=2, color='purple')
        ax1.set_title(source_name)
        ax1.legend()

        n_labels = int(labels.max()) + 1
        np.random.seed(42)
        rand_colors = np.random.rand(n_labels, 4)
        rand_colors[:, 3] = 1.0
        rand_colors[0]    = [0, 0, 0, 1]
        custom_cmap = ListedColormap(rand_colors)

        ax2.imshow(labels, origin='lower', cmap=custom_cmap,
                   vmin=0, vmax=n_labels - 1, interpolation='nearest')
        ax2.scatter(x_ycl,  y_ycl,  c='blue')
        ax2.scatter(x_map,  y_map,  c='orange')
        ax2.scatter(x_ogcc, y_ogcc, c='red')
        ax2.set_title(f'{source_name} watershed segmentation map')

        plt.tight_layout()
        plt.show()

        labels, labels_filtered, regions, cl_type = np.array(labels), np.array(labels_filtered), np.array(regions), np.array(cl_type)

    return table_filtered, labels, labels_filtered, regions, cl_type

def is_isolated_in_region(image, wcs, sky_region, ra, dec,
                           threshold_sigma=3.0,
                           box_size=3,
                           plot=False,
                           verbose=True):
    """
    Detect peaks inside a PolygonSkyRegion and decide whether the cluster
    aperture contains exactly one source.

    Strategy
    --------
    1. Project the sky region onto the image pixel grid and build a boolean
       mask (True inside the polygon).
    2. Estimate the background and noise from the pixels in a small annulus
       just outside the region using sigma-clipped stats.
    3. Run ``find_peaks`` restricted to pixels inside the region mask.
       In photutils, the ``mask`` argument marks pixels to EXCLUDE, so we
       pass ``~region_mask`` to analyse only the interior.
    4. Count detected peaks:
         0 peaks  → True  (cluster is below threshold — likely isolated but
                            faint; flagged with a warning)
         1 peak   → True  (isolated)
         ≥2 peaks → False (contaminated)

    Parameters
    ----------
    image          : 2-D ndarray   f555w science image (native resolution)
    wcs            : astropy WCS   WCS of *image*
    sky_region     : PolygonSkyRegion   aperture region in sky coordinates
    ra, dec        : float         cluster position [deg] (from catalog)
    threshold_sigma: float         detection threshold in units of local σ
                                   (default 3.0)
    box_size       : int           minimum separation between peaks in pixels;
                                   ``find_peaks`` uses this as the footprint
                                   size (default 3)
    plot           : bool          show diagnostic cutout
    verbose        : bool          print detection summary

    Returns
    -------
    isolated : bool    True if the region contains exactly one peak
    n_peaks  : int     number of peaks found inside the region
    peaks    : astropy Table or None   full find_peaks output
    """

    # ── 1. project region → a bbox-local pixel mask ──────────────────────────
    # Rather than allocating a mask the size of the full native-resolution
    # image for what is typically a tiny aperture, work in a small window
    # around the region's own bounding box, padded enough that the dilation
    # in step 2 (and find_peaks' box_size in step 3) are completely
    # unaffected by the crop — i.e. numerically identical to running on the
    # full image, just without the full-image-sized allocation.
    dilation_iterations = 10
    pixel_region = sky_region.to_pixel(wcs)
    pix_mask = pixel_region.to_mask()
    bbox = pix_mask.bbox
    pad = dilation_iterations + box_size + 5
    iy0 = max(bbox.iymin - pad, 0)
    iy1 = min(bbox.iymax + pad, image.shape[0])
    ix0 = max(bbox.ixmin - pad, 0)
    ix1 = min(bbox.ixmax + pad, image.shape[1])

    sl_large, sl_small = pix_mask.get_overlap_slices(image.shape)

    if sl_large is None or iy1 <= iy0 or ix1 <= ix0:
        if verbose:
            logger.warning('region projects to zero pixels — outside image?')
        return True, 0, None

    image_local = image[iy0:iy1, ix0:ix1]
    region_mask = np.zeros(image_local.shape, dtype=bool)
    ys = slice(sl_large[0].start - iy0, sl_large[0].stop - iy0)
    xs = slice(sl_large[1].start - ix0, sl_large[1].stop - ix0)
    region_mask[ys, xs] = pix_mask.data[sl_small].astype(bool)

    if region_mask.sum() == 0:
        if verbose:
            logger.warning('region projects to zero pixels — outside image?')
        return True, 0, None

    # ── 2. background from a thin annulus just outside the region ────────────
    # Dilate the region mask by a few pixels to get the annulus shell

    dilated    = binary_dilation(region_mask, iterations=dilation_iterations)
    annulus    = dilated & ~region_mask & np.isfinite(image_local)

    if annulus.sum() > 5:
        _, bkg_median, bkg_std = sigma_clipped_stats(
            image_local[annulus], sigma=3.0
        )
    else:
        # Fallback: global sigma-clipped stats on finite pixels
        _, bkg_median, bkg_std = sigma_clipped_stats(
            image_local[np.isfinite(image_local)], sigma=3.0
        )

    threshold = bkg_median + threshold_sigma * bkg_std

    # ── 3. detect peaks inside the region only ───────────────────────────────
    # find_peaks mask convention: True = EXCLUDED → pass ~region_mask
    # so only pixels inside the polygon are searched.
    exclude_mask = ~region_mask | ~np.isfinite(image_local)

    peaks = find_peaks(
        image_local,
        threshold = threshold,
        box_size  = box_size,
        mask      = exclude_mask,
    )

    if peaks is not None:
        # shift local pixel coordinates back to the full image frame
        peaks['x_peak'] += ix0
        peaks['y_peak'] += iy0

    n_peaks = 0 if peaks is None else len(peaks)

    # ── 4. decision ───────────────────────────────────────────────────────────
    if n_peaks == 0:
        isolated = True
        verdict  = 'ISOLATED (faint — no peak above threshold)'
    elif n_peaks == 1:
        isolated = True
        verdict  = 'ISOLATED (single peak)'
    else:
        isolated = False
        verdict  = f'CONTAMINATED ({n_peaks} peaks)'

    if verbose:
        logger.info(f'  Threshold  : {threshold:.4e}  '
              f'(bkg={bkg_median:.4e}, σ={bkg_std:.4e}, '
              f'N={threshold_sigma}σ)')
        logger.info(f'  Peaks found: {n_peaks}')
        logger.info(f'  Result     : {verdict}')

    return isolated, n_peaks, peaks


def save_psf_to_table(galaxy_name, psf_values: dict, filepath="psf_table.csv"):
    """
    Save one or more PSF measurements for a galaxy to the shared CSV table.

    Only the columns supplied in psf_values are written/updated; every other
    band column is left at its current value (or initialised to 0.0 for a new
    galaxy / new column).  This allows each band to be measured and saved
    independently without requiring all bands to be available at once.

    Parameters
    ----------
    galaxy_name : str
        Galaxy identifier, e.g. "NGC1234".
    psf_values : dict
        Mapping of column name → measured FWHM in arcsec,
        e.g. {"psf_HST_f555w": 0.08, "psf_MUSE": 0.72}.
    filepath : str
        Path to the CSV file (created if it does not exist).
    """

    # Sentinel for "not yet measured" — distinct from a real measurement.
    # Use 0.0 so that the validity check (val > 0) is unambiguous.
    _EMPTY = 0.0

    if os.path.exists(filepath):
        table = Table.read(filepath, format="csv")

        for col in table.colnames:
            if col == "galaxy_name":
                table[col] = table["galaxy_name"].astype("U50")
            else:
                table[col] = table[col].astype(float)


        # Add any brand-new band columns with sentinel value
        for col in psf_values.keys():
            if col not in table.colnames:
                table[col] = _EMPTY

        if galaxy_name in table["galaxy_name"]:
            # Galaxy row exists → update only the supplied columns
            idx = list(table["galaxy_name"]).index(galaxy_name)
            for col, val in psf_values.items():
                table[col][idx] = val
        else:
            # New galaxy → append row; unset bands default to _EMPTY
            new_row = {col: _EMPTY for col in table.colnames}
            new_row["galaxy_name"] = galaxy_name
            new_row.update(psf_values)
            table.add_row(new_row)
    else:
        # CSV does not exist yet → build it from the canonical column list.
        # All bands not supplied in psf_values are initialised to _EMPTY.
        all_bands = [
            "psf_MUSE",
            "psf_HST_f275w",
            "psf_HST_f336w",
            "psf_HST_f435w",
            "psf_HST_f438w",
            "psf_HST_f555w",
            "psf_HST_f814w",
            "psf_JWST_f200w",
            "psf_JWST_f300m",
            "psf_JWST_f335m",
            "psf_JWST_f360m",
            "psf_JWST_f770w",
            "psf_JWST_f1000w",
            "psf_JWST_f1130w",
            "psf_JWST_f2100w"
        ]
        row = {"galaxy_name": galaxy_name}
        for band in all_bands:
            row[band] = psf_values.get(band, _EMPTY)

        table = Table(rows=[row], names=["galaxy_name"] + all_bands)

    ascii.write(table, filepath, format="csv", overwrite=True)
    logger.info(f"  PSF saved → {filepath}  (galaxy: {galaxy_name}, bands: {list(psf_values.keys())})")

def load_psf_from_table(galaxy_name, bands: list, filepath="psf_table.csv"):
    """
    Load PSF values for a galaxy and return them ready for convolution.

    Parameters:
        galaxy_name : str   → e.g. "NGC1234"
        bands       : list  → column names to retrieve, e.g. ["psf_MUSE", "psf_HST_F606W"]
        filepath    : str   → path to the CSV file

    Returns:
        dict  → {"psf_MUSE": 0.8, "psf_HST_F606W": 0.12, ...}
                Returns None for bands not found in the table.
    
    Example:
        psf = load_psf_from_table("NGC1234", ["psf_MUSE", "psf_HST_F606W"])
        fwhm = [psf["psf_MUSE"], psf["psf_HST_F606W"]]
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"PSF table not found: {filepath}")

    table = Table.read(filepath, format="csv")

    if galaxy_name not in table["galaxy_name"]:
        raise ValueError(f"Galaxy '{galaxy_name}' not found in PSF table.")

    row = table[table["galaxy_name"] == galaxy_name][0]

    result = {}
    for band in bands:
        if band in table.colnames:
            result[band] = float(row[band])
        else:
            logger.info(f"  Warning: column '{band}' not found in table → set to None")
            result[band] = None

    return result

def _galaxy_output_dir(base_dir, galaxy_name):
    """Return (and create if needed) the per-galaxy output folder."""
    path = os.path.join(base_dir, galaxy_name)
    os.makedirs(path, exist_ok=True)
    return path


















