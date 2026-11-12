from aux_functions import get_norm

import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling import models, fitting
from photutils.background import Background2D, MMMBackground
from photutils.detection import DAOStarFinder

import logging

logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False



# ─────────────────────────────────────────────────────────────────────────────
# 2D Moffat fit  (same interface as fit_2d_gaussian)
# ─────────────────────────────────────────────────────────────────────────────

def fit_2d_moffat(data0, true_x_mean, true_y_mean, plot=False, fixed_alpha = False):
    """
    Fit a circular 2-D Moffat + constant background to a cutout.

    Model
    -----
        I(x,y) = A · [1 + ((x−x₀)²+(y−y₀)²) / γ²]^(−α)  +  C

    The constant ``C`` absorbs any residual background pedestal so the Moffat
    component is not forced to fit a tilted floor — a common cause of α
    collapsing to its lower bound or the fit converging to the background level.

    Robustness improvements over a naive fit
    -----------------------------------------
    * **Peak-based centre guess** — the initial (x₀, y₀) is set from
      ``argmax`` of the cutout, not the geometric centre.  This matters when
      the DAOStarFinder centroid is slightly offset from the actual peak.
    * **Amplitude lower bound** — forces the Moffat peak to be positive and
      at least 10 % of the cutout peak, preventing collapse to zero amplitude.
    * **Centre bounds** — constrain x₀/y₀ to stay within ±30 % of the cutout
      half-width of the peak pixel, so the fitter cannot wander to a neighbour.
    * **α lower bound raised to 1.5** — eliminates the near-singular flat-wing
      regime (α → 1) that ``LevMarLSQFitter`` often parks at when confused.
    * **TRFLSQFitter** — Trust Region Reflective; handles bounds properly and
      is substantially more robust than ``LevMarLSQFitter`` when parameters
      are near their bounds.
    * **Convergence check** — if the fitter did not converge the function
      returns ``(None, None, None)`` so the caller can skip this source.

    Parameters
    ----------
    data0        : 2-D square ndarray  — cutout (background approximately
                                         subtracted but a residual pedestal
                                         is fine; the model handles it)
    true_x_mean  : float               — DAOStarFinder centroid x (plot only)
    true_y_mean  : float               — DAOStarFinder centroid y (plot only)
    plot         : bool

    Returns
    -------
    moffat_component : astropy Moffat2D   — use .fwhm for FWHM in pixels,
                                            or None if fit did not converge
    data_fitted      : 2-D ndarray        — full model (Moffat + background)
    residuals        : 2-D ndarray        — data0 − data_fitted
    """
    ny, nx = data0.shape
    if ny != nx:
        logger.info('Cutout is not square — aborting Moffat fit')
        return None, None, None, None, None, None

    y, x = np.mgrid[0:ny, 0:nx]

    # ── robust initial centre: actual peak in the cutout ─────────────────────
    #peak_idx   = np.nanargmax(data0)
    #peak_y, peak_x = np.unravel_index(peak_idx, data0.shape)
    #peak_val   = data0[peak_y, peak_x]
    #bkg_guess  = float(np.nanmedian(data0))         # initial background level
    peak_x   = float(true_x_mean)
    peak_y   = float(true_y_mean)
    peak_xi  = int(np.clip(np.round(peak_x), 0, nx - 1))
    peak_yi  = int(np.clip(np.round(peak_y), 0, ny - 1))
    peak_val = float(data0[peak_yi, peak_xi])
    bkg_guess = float(np.nanmedian(data0)) 

    # ── composite model: Moffat + constant background ────────────────────────
    moffat = models.Moffat2D(
        amplitude = peak_val - bkg_guess,            # source flux above background
        x_0       = float(peak_x),
        y_0       = float(peak_y),
        gamma     = 3.0,
        alpha     = 2.8,
    )

    moffat.alpha.fixed = fixed_alpha          # held constant during fit

    background = models.Const2D(amplitude=bkg_guess)
    composite  = moffat + background

    # ── bounds ────────────────────────────────────────────────────────────────
    half = ny / 2
    tol  = 0.3 * half                               # ±30% of half-width

    # Moffat (index 0 in the compound model)
    composite.amplitude_0.bounds = (0.1 * peak_val, None)   # must be positive
    composite.x_0_0.bounds       = (peak_x - tol, peak_x + tol)
    composite.y_0_0.bounds       = (peak_y - tol, peak_y + tol)
    composite.gamma_0.bounds     = (0.5, half)  
    if not fixed_alpha:             # physically sensible
        composite.alpha_0.bounds     = (1.5, 20.0)               # avoid near-singular regime

    # Background (index 1): allow small negative (over-subtracted sky)
    composite.amplitude_1.bounds = (-0.5 * abs(bkg_guess) - 1, None)

    # ── fit ───────────────────────────────────────────────────────────────────
    fitter = fitting.TRFLSQFitter(calc_uncertainties=False)
    fitted = fitter(composite, x, y, data0, maxiter=1000)

     # ── convergence check ─────────────────────────────────────────────────────
    # TRFLSQFitter (wraps scipy.optimize.least_squares) reports success via
    # 'status' with values 1-4; the older LevMarLSQFitter used 'ierr' instead.
    # Checking both keys keeps this robust to either fitter being used.
    alpha_fitted   = fitted.alpha_0.value
    gamma_fitted   = fitted.gamma_0.value
    fit_info       = fitter.fit_info
    status         = fit_info.get('status', fit_info.get('ierr', 0))
 
    # Both bounds indicate a degenerate/unconstrained fit, not a genuine
    # measurement — the lower bound (near-singular flat-wing regime) was
    # already guarded; the upper bound needs the same treatment. This only
    # matters when fixed_alpha=False (alpha_0.bounds is only set in that
    # case, line ~113 above) — with fixed_alpha=True alpha never moves from
    # its initial guess, so this could never trigger there. Found via
    # measure_native_psf_shape() (the first caller to actually exercise
    # fixed_alpha=False in practice): without this check, fits that ran to
    # the alpha upper bound (e.g. from low-S/N or blended cutouts) were
    # silently accepted as converged and pulled the aggregate median alpha
    # up to ~18-19 instead of a physically sensible ~2-4.
    alpha_lower_bound = 1.5
    alpha_upper_bound = 20.0
    at_lower_bound = np.isclose(alpha_fitted, alpha_lower_bound, atol=1e-3)
    at_upper_bound = np.isclose(alpha_fitted, alpha_upper_bound, atol=1e-3)

    converged = (status in (1, 2, 3, 4)) and not at_lower_bound and not at_upper_bound
 
    if not converged:
        return None, None, None, fitter, None, None

    # Extract the Moffat component alone for FWHM access
    moffat_component       = models.Moffat2D(
        amplitude = fitted.amplitude_0.value,
        x_0       = fitted.x_0_0.value,
        y_0       = fitted.y_0_0.value,
        gamma     = fitted.gamma_0.value,
        alpha     = fitted.alpha_0.value,
    )

    data_fitted = fitted(x, y)
    residuals   = data0 - data_fitted

    

    if plot:
        logger.info("\nFitted Moffat + background parameters:")
        logger.info(f"  Amplitude : {fitted.amplitude_0.value:.4f}")
        logger.info(f"  x_0       : {fitted.x_0_0.value:.2f}  (peak guess: {peak_x})")
        logger.info(f"  y_0       : {fitted.y_0_0.value:.2f}  (peak guess: {peak_y})")
        logger.info(f"  gamma (γ) : {fitted.gamma_0.value:.3f} px")
        logger.info(f"  alpha (α) : {alpha_fitted:.3f}")
        logger.info(f"  Background: {fitted.amplitude_1.value:.4f}")
        logger.info(f"  FWHM      : {moffat_component.fwhm:.3f} px")

        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

        im1 = axes[0].imshow(data0, origin='lower', cmap='viridis',
                             norm=get_norm(data0, 0, 99.9, 'asinh'))
        axes[0].plot(true_x_mean, true_y_mean, 'r+',
                     markersize=15, markeredgewidth=3, label='DAO centroid')
        axes[0].plot(peak_x, peak_y, 'w+',
                     markersize=12, markeredgewidth=2, label='Peak guess')
        axes[0].set_title('Data')
        axes[0].legend(fontsize=8)
        plt.colorbar(im1, ax=axes[0])

        im2 = axes[1].imshow(data_fitted, origin='lower', cmap='viridis',
                             norm=get_norm(data0, 0, 99.9, 'asinh'))
        axes[1].plot(fitted.x_0_0.value, fitted.y_0_0.value, 'r+',
                     markersize=15, markeredgewidth=3, label='Fitted centre')
        axes[1].set_title(f'Moffat+bkg  (α={alpha_fitted:.2f})')
        axes[1].legend(fontsize=8)
        plt.colorbar(im2, ax=axes[1])

        im3 = axes[2].imshow(residuals, origin='lower', cmap='RdBu_r',
                             vmin=-3 * np.std(residuals),
                             vmax= 3 * np.std(residuals))
        axes[2].set_title(f'Residuals  (std={np.std(residuals):.4f})')
        plt.colorbar(im3, ax=axes[2])

        plt.tight_layout()
        plt.show()
        #plt.close()

        # 1-D profile slices through the fitted centre
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        yc = int(round(fitted.y_0_0.value))
        xc = int(round(fitted.x_0_0.value))

        axes[0].plot(data0[yc, :],       'o', label='Data',        alpha=0.6)
        axes[0].plot(data_fitted[yc, :], '-', label='Moffat+bkg',  linewidth=2)
        axes[0].axvline(true_x_mean, color='red',   linestyle='--',
                        alpha=0.5, label='DAO centroid')
        axes[0].axvline(peak_x,      color='white', linestyle=':',
                        alpha=0.5, label='Peak guess')
        axes[0].set_title('X profile')
        axes[0].set_xlabel('X pixel')
        axes[0].set_ylabel('Intensity')
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(data0[:, xc],       'o', label='Data',        alpha=0.6)
        axes[1].plot(data_fitted[:, xc], '-', label='Moffat+bkg',  linewidth=2)
        axes[1].axvline(true_y_mean, color='red',   linestyle='--',
                        alpha=0.5, label='DAO centroid')
        axes[1].axvline(peak_y,      color='white', linestyle=':',
                        alpha=0.5, label='Peak guess')
        axes[1].set_title('Y profile')
        axes[1].set_xlabel('Y pixel')
        axes[1].set_ylabel('Intensity')
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
        #plt.close()

    return moffat_component, data_fitted, residuals, fitter, alpha_fitted, gamma_fitted

def measure_psf_fwhm_from_image(data, wcs, n_sources=10, plot=False,
                                 sigma=3.0, fwhm=3.0, r_limit=1.5,
                                 galaxy_name=None,
                                 mask_npixels=50, mask_detect_sigma=1.5,
                                 fixed_alpha = False, lw_lim = 0, up_lim = 10):
    """
    Measure the PSF FWHM from point sources in the image, fitting a 2-D
    Moffat profile (see fit_2d_moffat) to each candidate star.

    Parameters
    ----------
    data      : 2-D ndarray
    wcs       : astropy WCS, optional
    n_sources : int     maximum number of sources to fit
    plot      : bool
    sigma     : float   detection threshold in sigma above background
                        for the PSF *star* finder (default 3.0)
    fwhm      : float   initial FWHM guess for DAOStarFinder (pixels)
    r_limit   : float   upper FWHM sanity limit in arcsec
    galaxy_name : str or None
        If given (e.g. 'Circinus'), the galaxy body is detected via
        segmentation and masked out before PSF star detection, so the bright
        nucleus does not contaminate the background estimate or get picked up
        as a PSF source.  Requires ``wcs`` to be provided.
    mask_npixels : int
        Minimum connected pixels for the galaxy segmentation (default 50).
        Increase for noisier images; decrease if the galaxy is only barely
        resolved.
    mask_detect_sigma : float
        Detection threshold (in σ) used for the galaxy segmentation
        (default 1.5 — intentionally low to catch faint outer regions).

    Returns
    -------
    fwhm_arcsec : float
    fwhm_pixels : float
    """

    # ── pixel scale ──────────────────────────────────────────────────────────
    if wcs is not None:
        pixel_scale = wcs.proj_plane_pixel_scales()[0].to('arcsec').value
    else:
        pixel_scale = 0.2   # default MUSE pixel scale

    # ── galaxy mask ───────────────────────────────────────────────────────────
    # Built before background estimation so the bright nucleus is excluded
    # from both sigma_clipped_stats and DAOStarFinder.
    galaxy_mask = None
    gal_cx = gal_cy = None

    #if galaxy_name is not None and wcs is not None:
    #    galaxy_mask, gal_cx, gal_cy = make_galaxy_mask(
    #        galaxy_name, data, wcs,
    #        npixels=mask_npixels,
    #        detect_sigma=mask_detect_sigma,
    #    )

    # ── background + threshold (sigma-clipped, robust against bright nuclei) ─
    # Combine the galaxy mask with NaN pixels into one mask for stats.
    valid = np.isfinite(data)# & (data != 0)
    #stats_mask = nan_mask | galaxy_mask if galaxy_mask is not None else nan_mask

   
    #_, median_bkg, std_bkg = sigma_clipped_stats(
    #    data, sigma=3.0, maxiters=5, mask=stats_mask
    #)
    #data_sub  = data - median_bkg
    #threshold = sigma * std_bkg

    bkg_estimator = MMMBackground()
    bkg = Background2D(data, (50,50), filter_size=(3,3), bkg_estimator=bkg_estimator)
   
    sigma_bkg = np.nanmedian(bkg.background_rms)
    data = data - bkg.background

    threshold = sigma * sigma_bkg

    # ── source detection ─────────────────────────────────────────────────────
    daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold)
    # Pass the combined mask so DAOStarFinder ignores the galaxy region too
    sources = daofind(data)

    if sources is None or len(sources) == 0:
        logger.info("No sources found!")
        return None, None, [], [], None, None

    valid = np.isfinite(data)

    xi = np.round(sources['xcentroid']).astype(int)
    yi = np.round(sources['ycentroid']).astype(int)

    ny, nx = data.shape

    interior = (valid[yi, xi]
                & valid[np.clip(yi + 100, 0, ny-1), xi]
                & valid[np.clip(yi - 100, 0, ny-1), xi]
                & valid[yi, np.clip(xi + 100, 0, nx-1)]
                & valid[yi, np.clip(xi - 100, 0, nx-1)])

    sources = sources[interior]

    if len(sources) == 0:
        logger.info("No sources found!")
        return None, None, [], [], None, None

    sources.sort('flux', reverse=True)

    # ── profile fitting ───────────────────────────────────────────────────────
    fwhm_measurements = []
    alpha_measurements = []   # only populated for Moffat; useful to inspect
    gamma_measurements = []   # only populated for Moffat; useful to inspect
    not_used = []
    cutouts = []

    size = 30

    for i in range(min(n_sources, len(sources))):
        x_center = sources['xcentroid'][i]
        y_center = sources['ycentroid'][i]

        # Skip sources near edges
        if (x_center < size or x_center > data.shape[1] - size or
                y_center < size or y_center > data.shape[0] - size):
            continue

        x_int, y_int = int(x_center), int(y_center)
        cutout = data[y_int - size:y_int + size, x_int - size:x_int + size]

        if False in np.isfinite(cutout):
            continue

        elif cutout.shape != (2 * size, 2 * size):
            continue
        else: 
            cutouts.append(cutout)

        try:
            fitted, _, _, _, alpha_fitted, gamma_fitted = fit_2d_moffat(cutout, size, size, plot = False, fixed_alpha = fixed_alpha)
            if fitted is None:
                continue
            fwhm_px = fitted.fwhm             # Moffat2D has .fwhm

            fwhm_as = fwhm_px * pixel_scale

            if fwhm_as is not None and lw_lim < fwhm_as < up_lim:
                fwhm_measurements.append(fwhm_as)
                alpha_measurements.append(alpha_fitted)
                gamma_measurements.append(gamma_fitted)
            else:
                not_used.append(fwhm_as)

        except Exception:
        #    logger.info('fit fails')
            continue

    if len(fwhm_measurements) == 0:
        logger.info("Could not fit any sources!")
        logger.info('Setting 1.0" resolution')
        return 1.0, 5.0, not_used, cutouts, None, None

    fwhm_arcsec = np.median(fwhm_measurements)
    fwhm_pixels = fwhm_arcsec / pixel_scale

    alpha = np.median(alpha_measurements) if alpha_measurements else None
    gamma = np.median(gamma_measurements) if gamma_measurements else None

    logger.info(f"PSF measured from {len(fwhm_measurements)} sources "
          f"using moffat profile:")
    logger.info(f"  FWHM : {fwhm_pixels:.2f} px = {fwhm_arcsec:.3f} arcsec")
    logger.info(f"  Std  : {np.std(fwhm_measurements):.3f} arcsec")
    if alpha_measurements:
        logger.info(f"  α    : {alpha:.2f} "
              f"(median power-law index)")
    if gamma_measurements:
        logger.info(f"  γ    : {gamma:.2f} "
              f"(median shape parameter)")

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        axes[0].hist(fwhm_measurements, bins=15, edgecolor='black')
        axes[0].axvline(fwhm_arcsec, color='red', linestyle='--',
                        label=f'Median: {fwhm_arcsec:.3f}"')
        axes[0].set_xlabel('FWHM (arcsec)')
        axes[0].set_ylabel('Number of sources')
        axes[0].set_title('PSF FWHM distribution (moffat)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].imshow(data, origin='lower', cmap='viridis',
                       norm=get_norm(data, 15, 99.7, 'asinh'))
        axes[1].plot(sources['xcentroid'][:n_sources],
                     sources['ycentroid'][:n_sources],
                     'r+', markersize=10, markeredgewidth=2,
                     label='Detected sources')

        # Overlay galaxy mask boundary so you can check the exclusion zone
        if galaxy_mask is not None:
            axes[1].contour(galaxy_mask, levels=[0.5],
                            colors='yellow', linewidths=1.5,
                            linestyles='--')
            axes[1].plot(gal_cx, gal_cy, 'y*', markersize=12,
                         label=f'{galaxy_name} centre')

        axes[1].set_title(f'Detected sources')
        axes[1].legend(fontsize=8)
        axes[1].figure.colorbar(
            axes[1].images[0], ax=axes[1], label='Flux'
        )

        plt.tight_layout()
        plt.show()
        #plt.close()

    return fwhm_arcsec, fwhm_pixels, not_used, cutouts, alpha, gamma