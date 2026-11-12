from aux_functions import convolution, find_isolated, columns_with_masked_values, get_norm, format_time, save_psf_to_table, load_psf_from_table, _galaxy_output_dir, is_isolated_in_region, gaussian_kernel_sigma_pixels
from psf_aux_func import measure_psf_fwhm_from_image
from phot_aux_func import measure_photometry, spectra_from_aperture, jwst_pixel_sum_to_Jy, corr_flux
from native_phot_aux_func import (
    compute_r65, build_native_calibration_curve, measure_native_photometry_one_source,
    remeasure_averaged_slit_profile_at_aperture, REFERENCE_BAND_GROUPS,
    CLUSTER_PROFILE_BY_CLASS, INSTRUMENT_METHODS, profile_spec_for_class, profile_tag,
    profile_spec_from_tag,
    measure_instrument_photometry_one_source, instrument_fit_model_profile,
    instrument_slit_length, _sample_averaged_slit_profile, SLIT_COMBINE,
    deger22_correction_factor, deger22_4px_flux, exact_4px_correction,
    DEGER22_R4_ARCSEC, DEGER22_SKY_ARCSEC,
)
from instrument_psf import instrument_psf_for_image

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import numpy as np
from astropy.io import fits, ascii
from astropy import units as u
from astropy import constants as const
from astropy.io import fits as pyfits
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel
from astropy.coordinates import SkyCoord
from astropy.table import Table, join
from reproject import reproject_interp
from photutils.background import Background2D, MMMBackground
from regions import Regions, CircleSkyRegion

from dustmaps.sfd import SFDQuery
from matplotlib.gridspec import GridSpec

import os
import re
import warnings
import time
import datetime
from collections import namedtuple

import logging

logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

from astropy.convolution import Gaussian2DKernel, convolve_fft


# ─────────────────────────────────────────────────────────────────────────
# Multiprocessing worker for Photometry.run_spectro_photometry (opt-in via
# n_workers > 1). Kept at module level, not a Photometry method: a bound
# method or a closure over `self` can't be handed to ProcessPoolExecutor as
# cleanly as a plain function, and — more importantly — using the 'fork'
# start method (default on Linux) with these large, read-only inputs passed
# through `initializer`/`initargs` means each worker process inherits them
# via copy-on-write at fork time instead of being re-pickled per task; only
# the small per-source index actually crosses the task queue.
# ─────────────────────────────────────────────────────────────────────────

# PHANGS-HST cluster catalogs store fluxes under a generic band name
# (NUV/U/B/V/I) rather than the actual filter name — this maps the filter
# name (as used everywhere else in this module) to that catalog prefix, so
# the original catalog photometry can be looked up alongside our own.
CATALOG_HST_BAND_MAP = {
    'F275W': 'NUV', 'F336W': 'U', 'F438W': 'B', 'F555W': 'V', 'F814W': 'I',
}


def _catalog_flux_columns(band, colnames):
    """Return (flux_col, err_col) for a band's PHANGS catalog photometry,
    or (None, None) if not present. Tries the filter name directly first
    (IR5-style catalogs use PHANGS_F275W_mJy etc.), then the generic
    NUV/U/B/V/I name (DR5-style catalogs) via CATALOG_HST_BAND_MAP.
    """
    band_up = band.upper()
    for name in (band_up, CATALOG_HST_BAND_MAP.get(band_up)):
        if name is None:
            continue
        flux_col = f'PHANGS_{name}_mJy'
        if flux_col in colnames:
            err_col = f'{flux_col}_ERR'
            return flux_col, err_col if err_col in colnames else None
    return None, None


_worker_ctx = {}


def _init_photometry_worker(hst_conv_dict, jwst_conv_dict, muse_shape, muse_wcs,
                             log_wl, spectra, muse_id_maps, filter_wavelengths,
                             eb_v, regions_hst, regions_muse, type_ap):
    global _worker_ctx
    _worker_ctx = dict(
        hst_conv_dict=hst_conv_dict, jwst_conv_dict=jwst_conv_dict,
        muse_shape=muse_shape, muse_wcs=muse_wcs, log_wl=log_wl,
        spectra=spectra, muse_id_maps=muse_id_maps,
        filter_wavelengths=filter_wavelengths, eb_v=eb_v,
        regions_hst=regions_hst, regions_muse=regions_muse, type_ap=type_ap,
    )


def _photometry_worker(idd):
    """
    Run the per-source spectrum + photometry measurement for source `idd`,
    against the context set up once per worker by `_init_photometry_worker`.

    Never raises — a per-source failure is reported back as an error string
    (mirroring the try/except in the sequential loop) so one bad source
    can't take down the whole pool or silently drop other sources' results.
    """
    ctx = _worker_ctx

    if ctx['type_ap'] == 'HST':
        region_i = ctx['regions_hst'][idd]
    elif ctx['type_ap'] == 'MUSE':
        region_i = ctx['regions_muse'][idd]
    else:
        raise RuntimeError('Not valid region')

    spec_result, spec_error = None, None
    try:
        spec_result = spectra_from_aperture(
            region_i, ctx['muse_shape'], ctx['muse_wcs'], ctx['log_wl'],
            ctx['spectra'], ctx['muse_id_maps'],
        )
    except Exception as e:
        spec_error = str(e)

    phot_result, phot_error = None, None
    try:
        phot_result = measure_photometry(
            region_i, ctx['filter_wavelengths'],
            ctx['hst_conv_dict'], ctx['jwst_conv_dict'], ctx['eb_v'],
        )
    except Exception as e:
        phot_error = str(e)

    return idd, spec_result, spec_error, phot_result, phot_error


# ─────────────────────────────────────────────────────────────────────────
# Multiprocessing worker for Photometry.run_native_photometry (opt-in via
# n_workers > 1). Same fork + initializer/initargs pattern as the other
# workers above. Tasks are PER-SOURCE (not per (source, band) pair): since a
# band without its own detected emission can borrow another band's aperture
# for the SAME source (see native_phot_aux_func.measure_native_photometry_
# one_source), all of one source's bands must be resolved together, in one
# task, rather than as independent (source, band) units.
# ─────────────────────────────────────────────────────────────────────────

_native_photometry_ctx = {}


def _init_native_photometry_worker(band_data, radec, eb_v, reference_groups, measure_kwargs):
    global _native_photometry_ctx
    _native_photometry_ctx = dict(band_data=band_data, radec=radec, eb_v=eb_v,
                                   reference_groups=reference_groups,
                                   measure_kwargs=measure_kwargs)


def _jwst_unit_convert(raw_sum, header):
    # Single shared implementation with the v1/convolved JWST path
    # (phot_aux_func.measure_photometry) — see jwst_pixel_sum_to_Jy's own
    # docstring for why this can't just always apply PHOTMJSR.
    return jwst_pixel_sum_to_Jy(raw_sum, header)


def _native_photometry_worker(idd):
    ctx = _native_photometry_ctx
    ra, dec = ctx['radec'][idd]
    results = measure_native_photometry_one_source(
        ra, dec, ctx['band_data'], reference_groups=ctx['reference_groups'],
        eb_v=ctx['eb_v'], **ctx['measure_kwargs'],
    )
    return idd, results


# Same pattern for Photometry.run_instrument_photometry: one task per source,
# the per-band InstrumentPSF models (and their profile caches) shared with
# the workers at fork time.
_instrument_photometry_ctx = {}


def _init_instrument_photometry_worker(band_data, per_source, measure_kwargs):
    global _instrument_photometry_ctx
    _instrument_photometry_ctx = dict(band_data=band_data, per_source=per_source,
                                      measure_kwargs=measure_kwargs)


def _instrument_photometry_worker(idd):
    ctx = _instrument_photometry_ctx
    src = ctx['per_source'][idd]
    results = measure_instrument_photometry_one_source(
        src['ra'], src['dec'], src['native_row'], ctx['band_data'], src['profile_spec'],
        distance_mpc=src['distance_mpc'], eb_v=src['eb_v'], **ctx['measure_kwargs'],
    )
    return idd, results


# ─────────────────────────────────────────────────────────────────────────
# Multiprocessing worker for IsolatedSourceFinder.second_filter (opt-in via
# n_workers > 1). Same fork + initializer/initargs pattern as the
# Photometry worker above — the native-resolution HST image is shared with
# each worker process via copy-on-write at fork time instead of being
# re-pickled per source.
# ─────────────────────────────────────────────────────────────────────────

_second_filter_ctx = {}


def _init_second_filter_worker(hst_data, hst_wcs, regions_final_0):
    global _second_filter_ctx
    _second_filter_ctx = dict(hst_data=hst_data, hst_wcs=hst_wcs, regions=regions_final_0)


def _second_filter_worker(args):
    i, ra, dec = args
    ctx = _second_filter_ctx
    isolated, n_peaks, peaks = is_isolated_in_region(
        ctx['hst_data'], ctx['hst_wcs'], ctx['regions'][i], ra, dec,
        threshold_sigma=10.0, box_size=3, verbose=False, plot=False,
    )
    return i, isolated, n_peaks, peaks


# A namedtuple (not a plain list/dataclass) is deliberate: it is still a
# 3-tuple, so `source[0]`/`source[1]`/`source[2]` keep working exactly as
# before for any external code (e.g. notebooks) indexing into it, while
# internal code can use the clearer `.full`/`.gal`/`.gal_lower` names
# instead of magic indices.
Source = namedtuple('Source', ['full', 'gal', 'gal_lower'])


class IsolatedSourceFinder:
    """
    Class to handle PSF matching, reprojection, surface-brightness thresholding,
    and source identification between HST and MUSE images.
    """

    def __init__(self, src, hst_path, muse_path, cat_path, sample, psf_filepath='psf_table.csv',
                 psf_source='theoretical'):
        """
        Initialize the object with the paths to the HST and MUSE images.

        Parameters
        ----------
        source : str
            name of the source
        hst_path : str
            File path to the HST image.
        muse_path : str
            File path to the MUSE image.
        psf_filepath : str
            Path to the shared PSF CSV table (same convention as
            Photometry.__init__, so both classes read/write the same file
            for a given galaxy rather than risking two different tables).
        psf_source : 'theoretical' (default): HST F555W FWHM from the
            theoretical PSF (instrument_psf.theoretical_psf), MUSE FWHM from
            the token name; nothing measured, psf_table.csv untouched.
            'table': psf_table.csv, measuring what is missing.
        """
        if psf_source not in ('theoretical', 'table'):
            raise ValueError(f"psf_source must be 'theoretical' or 'table', got {psf_source!r}")
        self.psf_source = psf_source
        self.source = Source(full=src, gal=src.split('-')[0], gal_lower=src.split('-')[0].lower())
        self.status = None
        self.hst_path = hst_path
        self.muse_path = muse_path
        self.cat_path = cat_path
        self.sample = sample
        self.psf_filepath = psf_filepath
        self.hst_data = None
        self.hst_header = None
        self.hst_wcs = None
        self.muse_data = None
        self.muse_header = None
        self.muse_wcs = None
        self.best_files = {}
        self.catalogs = None
        self.hst_psf = None
        self.muse_psf = None
        self.hst_psf_pixels = None
        self.muse_psf_pixels = None
        self.hst_conv = None
        self.hst_wcs_conv = None
        self.hst_cut = None
        self.hst_conv_reproj = None
        self.N_muse = None
        self.hst_sources = None
        self.muse_sources = None
        self.matched_catalog = None
        self.regions_HST = None
        self.regions_MUSE = None
        self.regions_final = None

    # ---------------------------------------------------------
    # DATA LOADING
    # ---------------------------------------------------------
    def load_images(self):
        """
        Load the HST and MUSE images and their WCS solutions.
        """
        #MUSE
        source = self.source.full

        if self.sample == 'PM':
            maps = pyfits.open(f'{self.muse_path}/{source}/{source}_DR2.2v2/{source}_MAPS.fits')
        elif self.sample == 'JWST_C2':
            maps = pyfits.open(f'{self.muse_path}/{source}/{source}_v0/{source}_MAPS.fits')
        self.muse_data = maps[2].data
        self.muse_header = maps[2].header
        self.muse_wcs = WCS(self.muse_header)

        self.status = 'MUSE data loaded'
        logger.info(self.status)

        #HST
        source0 = self.source.gal

        source_hst = [i for i in os.listdir(f'{self.hst_path}/{source0}') if '555' in i][0]
        hdu = fits.open(f"{self.hst_path}/{source0}/{source_hst}")
        self.hst_data = np.where(hdu[0].data == 0.0, np.nan, hdu[0].data)
        self.hst_header = hdu[0].header
        self.hst_wcs = WCS(self.hst_header)

        self.status = 'HST data loaded'
        logger.info(self.status)

    # ---------------------------------------------------------
    # CATALOG LOADING
    # ---------------------------------------------------------

    def get_catalogs(self, split = True):

        galaxy_name = self.source.gal_lower

        if galaxy_name == 'ngc0628':
            galaxy_name = 'ngc628c'  # use the center pointing for catalogs

        if 'IR5' in self.cat_path:

            pattern = re.compile(f'{galaxy_name}.*class12')

            # Dictionary with priority: category -> (priority, filename)

            cat_files = os.listdir(f'{self.cat_path}')
            
            for file in cat_files:
                if not pattern.search(file):
                    continue
        
                is_obs = 'obs' in file
                is_human = 'human' in file
                is_ground = 'ground' in file
            
                # Determine category
                if is_obs:
                    category = 'obs_human' if is_human else 'obs_machine'
                else:
                    category = 'sed_human' if is_human else 'sed_machine'
        
                # Priority: HST = 1 (higher priority), ground = 2 (lower priority)
                priority = 1 if not is_ground else 2
            
                # Keep file with highest priority (lowest number)
                if category not in self.best_files or priority < self.best_files[category][0]:
                    self.best_files[category] = (priority, file)

            tables = {}
        
            # open cluster and association catalogs

            #try:
            #human tables
            obs_human = Table.read(f'{self.cat_path}/{self.best_files["obs_human"][1]}')
            sed_human = Table.read(f'{self.cat_path}/{self.best_files["sed_human"][1]}')

            human_final = join(obs_human, sed_human, keys = 'ID_PHANGS_CLUSTER', join_type = 'inner', metadata_conflicts='silent')

            tables['human'] = [human_final]

            #machine tables
            obs_machine = Table.read(f'{self.cat_path}/{self.best_files["obs_machine"][1]}')
            sed_machine = Table.read(f'{self.cat_path}/{self.best_files["sed_machine"][1]}')

            machine_final = join(obs_machine, sed_machine, keys = 'ID_PHANGS_CLUSTER', join_type = 'inner', metadata_conflicts='silent')

            tables['machine'] = [machine_final]

            self.cc_class_key = 'CC_CLASS'

        else:

            tables = {}

            human_table_path = 'hlsp_phangs-cat_hst_acs-uvis_dr5-targets_v1_human-class1+2-primary.fits'
            machine_table_path = 'hlsp_phangs-cat_hst_acs-uvis_dr5-targets_v1_machine-class1+2-primary.fits'

            human_final = Table.read(f'{self.cat_path}/{human_table_path}')
            human_final = human_final[human_final['PHANGS_GALAXY'] == galaxy_name]
            tables['human'] = [human_final]

            machine_final = Table.read(f'{self.cat_path}/{machine_table_path}')
            machine_final = machine_final[machine_final['PHANGS_GALAXY'] == galaxy_name]
            tables['machine'] = [machine_final]

            self.cc_class_key = 'PHANGS_CC_CLASS'
        

        #split in cluster class
        if split:
            for cc in ['YCL', 'MAP', 'OGCC']:
                mask_human = (human_final[f'{self.cc_class_key}_HUMAN'] == cc)
                tables['human'].append(human_final[mask_human])

                mask_machine = (machine_final[f'{self.cc_class_key}_ML'] == cc)
                tables['machine'].append(machine_final[mask_machine])


        self.catalogs = tables

        self.status = 'catalogs loaded'
        logger.info(self.status)

    
    # ---------------------------------------------------------
    # PSF IDENTIFICATION
    # ---------------------------------------------------------
    def measure_hst_psf(self, n_sources=10000, plot=False, sigma=10.0, fwhm=3.0, r_limit=0.5):
        """
        Identify or measure the PSF for the HST image.
        """

        if getattr(self, 'psf_source', 'table') == 'theoretical':
            from instrument_psf import theoretical_psf
            pix = self.hst_wcs.proj_plane_pixel_scales()[0].to('arcsec').value
            t = theoretical_psf('F555W', self.hst_header, pix)
            if not np.isfinite(t['fwhm_arcsec']):
                raise RuntimeError(f'no theoretical F555W PSF: {t["note"]}')
            self.hst_psf = t['fwhm_arcsec']
            self.hst_psf_pixels = self.hst_psf / pix
            self.status = f'HST F555W psf from the theoretical PSF: {self.hst_psf:.3f}"'
            logger.info(self.status)
            return

        # --- PSF: load from table or measure ---
        psf_col = "psf_HST_f555w"

        try:
            psf = load_psf_from_table(self.source.gal, [psf_col], filepath=self.psf_filepath)
            if psf[psf_col] != 0.0:
                valid = True
            else:
                valid = False
        except (FileNotFoundError, ValueError) as e:
            logger.info(f"  PSF table lookup failed ({e}) — will measure instead")
            valid = False

        if valid:
            logger.info(f" f555w PSF loaded from table → HST: {psf[psf_col]:.3f}")
            self.hst_psf = psf[psf_col]
            self.status = 'HST psf from table'

        else:
            logger.info('Measuring HST psf')
            self.hst_psf, self.hst_psf_pixels, _, _, _, _ = measure_psf_fwhm_from_image(self.hst_data, wcs=self.hst_wcs,
                                                                        n_sources=n_sources, sigma=sigma,
                                                                        fwhm=fwhm, r_limit=r_limit, plot=plot)
            save_psf_to_table(galaxy_name = self.source.gal,           # e.g. "NGC1234"value_source[0]
                              psf_values  = {psf_col      : self.hst_psf},
                              filepath    = self.psf_filepath)
            
            self.status = 'HST psf measured'

        logger.info(self.status)
        

    def measure_muse_psf(self, n_sources=10000, plot=False, sigma=3.0, fwhm=5.0, r_limit=2.0):
        """
        Identify or measure the PSF for the MUSE image.
        """
        # --- PSF: load from table or measure ---
        psf_col = "psf_MUSE"

        pixel_scale = self.muse_wcs.proj_plane_pixel_scales()[0].to('arcsec').value

        if 'asec' in self.source.full:

            self.muse_psf = float(self.source.full.split('-')[1].split('asec')[0])
            self.muse_psf_pixels = self.muse_psf / pixel_scale

            logger.info('MUSE psf from file name')

            return
        if getattr(self, 'psf_source', 'table') == 'theoretical':
            raise RuntimeError(f'MUSE PSF: token {self.source.full!r} has no "<x>asec" part '
                               f'(psf_source=theoretical takes the MUSE FWHM from the name)')

        try:
            psf = load_psf_from_table(self.source.gal, [psf_col], filepath=self.psf_filepath)
            if psf[psf_col] != 0.0:
                valid = True
            else:
                valid = False
        except (FileNotFoundError, ValueError) as e:
            logger.info(f"  PSF table lookup failed ({e}) — will measure instead")
            valid = False

        if valid:
            logger.info(f" MUSE PSF loaded from table → MUSE: {psf[psf_col]:.3f}")
            self.muse_psf = psf[psf_col]
            self.status = 'MUSE psf from table'

        else:
            logger.info('Measuring MUSE psf')
            self.muse_psf, self.muse_psf_pixels, _, _, _, _ = measure_psf_fwhm_from_image(self.muse_data, wcs=self.muse_wcs,
                                                                        n_sources=n_sources, sigma=sigma,
                                                                        fwhm=fwhm, r_limit=r_limit, plot=plot)
            save_psf_to_table(galaxy_name = self.source.gal,           # e.g. "NGC1234"value_source[0]
                              psf_values  = {psf_col      : self.muse_psf},
                              filepath    = self.psf_filepath)
            
            self.status = 'MUSE psf measured'
        
        logger.info(self.status)

    # ---------------------------------------------------------
    # CONVOLUTION / MATCHING
    # ---------------------------------------------------------
    def convolve(self):
        """
        Convolve the HST image to the MUSE PSF.
        """
        if self.hst_psf:
            logger.info('starting convolution')
            self.hst_conv, self.hst_cut, self.hst_wcs_conv = convolution(self.hst_data, self.hst_wcs, self.muse_data, 
                                                                         self.muse_wcs, fwhm = (self.hst_psf, self.muse_psf))
            self.status = 'convolution finished'
            logger.info(self.status)
        else:
            logger.info('measure the psf first')

    def reproject_hst_to_muse(self):
        """
        Reproject the convolved HST image to the MUSE WCS.
        """
        if self.hst_conv is not None:
            logger.info('starting reprojection')
            self.hst_conv_reproj, _ = reproject_interp((self.hst_conv, self.hst_wcs_conv), self.muse_wcs, 
                                                       shape_out = self.muse_data.shape)
            self.status = 'reprojection finished'
            logger.info(self.status)
        else:
            logger.info('convolve first')

    # ---------------------------------------------------------
    # SURFACE-BRIGHTNESS THRESHOLD
    # ---------------------------------------------------------
    def compute_hst_sb_threshold(self, sigma_level=3):
        """
        Compute the HST surface-brightness threshold for source detection.

        Parameters
        ----------
        sigma_level : float
            The sigma threshold (e.g., 3 sigma).
        """

        logger.info('computing SB')
        bkg_estimator = MMMBackground()
        self.hst_bkg = Background2D(self.hst_conv, (50, 50), filter_size=(3, 3), bkg_estimator=bkg_estimator)
        self.muse_bkg = Background2D(self.muse_data, (20, 20), filter_size=(3, 3), bkg_estimator=bkg_estimator)
        
        hst_threshold_counts = sigma_level * np.median(self.hst_bkg.background_rms)

        if 'PHOTPLAM' not in self.hst_header:
            raise KeyError(
                "HST header is missing 'PHOTPLAM' (pivot wavelength) — needed to convert "
                "the surface-brightness threshold from counts to physical flux units. "
                "Check that the loaded HST file is a calibrated drizzled science image."
            )

        SB_target = (hst_threshold_counts * u.Jy * const.c / ((self.hst_header['PHOTPLAM']*u.AA)**2)).to('erg s-1 cm-2 AA-1') / (self.hst_wcs.proj_plane_pixel_scales()[0].to('arcsec')**2)

        self.N_muse = SB_target * (self.muse_wcs.proj_plane_pixel_scales()[0].to('arcsec')**2) / (np.median(self.muse_bkg.background_rms) * 1e-20*u.erg/(u.s*u.cm**2*u.AA))
        logger.info(f'SB finished, N_muse = {self.N_muse}')
        self.status = 'SB finished'
        logger.info(self.status)

    # ---------------------------------------------------------
    # SOURCE FINDING
    # ---------------------------------------------------------
    def find_hst_sources(self, ml_hum, plot = False):
        """
        Identify sources in the HST image above the SB threshold.
        ml_hum: "machine" or "human"
        """
        logger.info('Identifying sources in HST')

        self.selection = ml_hum
        
        self.table_fil_HST, self.labels_HST, self.labels_fil_HST, self.regions_HST, self.cl_type_HST = find_isolated(self.source.gal, self.hst_conv, self.hst_wcs_conv, self.hst_bkg, self.catalogs, ml_hum, plot = plot)

        #self.table_fil_HST, self.labels_HST, self.labels_fil_HST, self.regions_HST, self.cl_type_HST = table_fil_HST0[mask_isolated], labels_HST0, labels_fil_HST0, regions_HST0[mask_isolated], cl_type_HST0

        self.status = 'HST sources identified'
        logger.info(self.status)
        

    def find_muse_sources(self, ml_hum, plot = False):
        """
        Identify sources in the MUSE image.
        ml_hum: "machine" or "human"
        """

        logger.info('Identifying sources in MUSE')

        self.table_fil_MUSE, self.labels_MUSE, self.labels_fil_MUSE, self.regions_MUSE, self.cl_type_MUSE = find_isolated(self.source.gal, self.muse_data, self.muse_wcs, self.muse_bkg, self.catalogs, ml_hum, NN = self.N_muse.value, plot = plot)

        self.status = 'MUSE sources identified'
        logger.info(self.status)
        

    # ---------------------------------------------------------
    # CATALOG MATCHING
    # ---------------------------------------------------------
    def match_catalogs(self):
        """
        Cross-match the HST and MUSE source tables.
        """
        tbl1 = self.table_fil_HST.copy()
        tbl2 = self.table_fil_MUSE.copy()

        tbl1['_row1'] = np.arange(len(tbl1))
        tbl2['_row2'] = np.arange(len(tbl2))

        bad1  = columns_with_masked_values(tbl1, tbl1.keys())
        bad2  = columns_with_masked_values(tbl2, tbl2.keys())

        # Replace masked values in the join key
        #for tbl in (tbl1, tbl2):
        for key in bad1:
            mask = tbl1[key].mask
            tbl1[key] = tbl1[key].filled(-9999)  # or "unknown"
        
        for key in bad2:
            mask = tbl2[key].mask
            tbl2[key] = tbl2[key].filled(-9999)  # or "unknown"

        joined = join(tbl1, tbl2, keys=tbl2.keys()[:-1], join_type = 'inner', metadata_conflicts='silent')

        for key in list(set(bad1) | set(bad2)):
            joined[key] = np.ma.masked_where(joined[key] == -9999, joined[key])

        idx1, idx2 = joined['_row1'], joined['_row2']

        self.final_table = joined
        self.labels_final =  [[self.labels_HST[i] for i in idx1], [self.labels_MUSE[i] for i in idx2]]
        self.labels_fil_final =  [[self.labels_fil_HST[i] for i in idx1], [self.labels_fil_MUSE[i] for i in idx2]]
        self.regions_final = [[self.regions_HST[i] for i in idx1], [self.regions_MUSE[i] for i in idx2]]
        self.cl_type_final = [[self.cl_type_HST[i] for i in idx1], [self.cl_type_MUSE[i] for i in idx2]]

        self.status = 'Matched Catalogs'
        logger.info(self.status)

    def second_filter(self, plot = False, n_workers = 1):
        """
        Per-source contamination check on the native-resolution HST image.

        Parameters
        ----------
        plot       : bool   (currently has no effect — is_isolated_in_region's
                             own `plot` parameter is not implemented)
        n_workers  : int    number of worker processes for the per-source
                            loop. 1 (default) runs sequentially in-process,
                            exactly as before; >1 uses a ProcessPoolExecutor
                            (fork start method).
        """

        logger.info('Filtering HST with native resolution')

        n_sources = len(self.final_table)
        mask_isolated = [None] * n_sources
        peaks_list = [None] * n_sources

        if n_workers <= 1:
            for i, cluster in enumerate(self.final_table):
                isolated, n_peaks, peaks = is_isolated_in_region(
                    self.hst_data, self.hst_wcs, self.regions_final[0][i],
                    cluster['PHANGS_RA'], cluster['PHANGS_DEC'],
                    threshold_sigma=10.0, box_size=3, verbose=False, plot=plot,
                )
                mask_isolated[i] = isolated
                peaks_list[i] = peaks
        else:
            tasks = [
                (i, cluster['PHANGS_RA'], cluster['PHANGS_DEC'])
                for i, cluster in enumerate(self.final_table)
            ]
            mp_ctx = mp.get_context('fork')
            with ProcessPoolExecutor(
                max_workers=n_workers, mp_context=mp_ctx,
                initializer=_init_second_filter_worker,
                initargs=(self.hst_data, self.hst_wcs, self.regions_final[0]),
            ) as executor:
                for i, isolated, n_peaks, peaks in executor.map(_second_filter_worker, tasks):
                    mask_isolated[i] = isolated
                    peaks_list[i] = peaks

        self.mask_isolated = np.array(mask_isolated)

        self.table_final_filtered = self.final_table[self.mask_isolated]

        mask_ycl  = self.table_final_filtered[f'{self.cc_class_key}_ML'] == 'YCL'
        mask_map  = self.table_final_filtered[f'{self.cc_class_key}_ML'] == 'MAP'
        mask_ogcc = self.table_final_filtered[f'{self.cc_class_key}_ML'] == 'OGCC'
        
        logger.info(f"Final after second filtering {mask_ycl.sum()} ycl")
        logger.info(f"Final after second filtering {mask_map.sum()} map")
        logger.info(f"Final after second filtering {mask_ogcc.sum()} ogcc")

    # ---------------------------------------------------------
    # PLOTS
    # ---------------------------------------------------------

    def plot_imag_cat(self):
        ycl_table, map_table, ogcc_table = self.catalogs['machine'][1], self.catalogs['machine'][2], self.catalogs['machine'][3]

        pos_ycl = SkyCoord(ycl_table['PHANGS_RA'], ycl_table['PHANGS_DEC'])
        pos_map = SkyCoord(map_table['PHANGS_RA'], map_table['PHANGS_DEC'])
        pos_ogcc = SkyCoord(ogcc_table['PHANGS_RA'], ogcc_table['PHANGS_DEC'])

        x_ycl_MUSE, y_ycl_MUSE = skycoord_to_pixel(pos_ycl, self.muse_wcs)
        x_map_MUSE, y_map_MUSE = skycoord_to_pixel(pos_map, self.muse_wcs)
        x_ogcc_MUSE, y_ogcc_MUSE = skycoord_to_pixel(pos_ogcc, self.muse_wcs)

        x_ycl, y_ycl = skycoord_to_pixel(pos_ycl, self.hst_wcs)
        x_map, y_map = skycoord_to_pixel(pos_map, self.hst_wcs)
        x_ogcc, y_ogcc = skycoord_to_pixel(pos_ogcc, self.hst_wcs)
    
        fig, (axs1, axs2) = plt.subplots(1, 2, figsize=(16,8))

        axs1.imshow(self.muse_data, origin='lower', norm = get_norm(self.muse_data, 0, 99.8, 'asinh'))
        axs1.scatter(x_ycl_MUSE, y_ycl_MUSE, c = 'blue', label = 'ycl')
        axs1.scatter(x_map_MUSE, y_map_MUSE, c = 'orange', label = 'map')
        axs1.scatter(x_ogcc_MUSE, y_ogcc_MUSE, c = 'red', label = 'ogcc')
        axs1.set_title(f'{self.source.gal} FLUX MUSE')
        axs1.legend()

        axs2.imshow(self.hst_data, origin='lower', norm = get_norm(self.hst_data, 15, 99.95, 'asinh'))
        axs2.scatter(x_ycl, y_ycl, c = 'blue', label = 'ycl')
        axs2.scatter(x_map, y_map, c = 'orange', label = 'map')
        axs2.scatter(x_ogcc, y_ogcc, c = 'red', label = 'ogcc')
        axs2.set_title(f'{self.source.gal} F438W HST')
        axs2.legend()

        plt.show()

    def plot_final_selection(self):

        if self.selection == 'machine':
            selection2 = 'ML'
        else:
            selection2 = self.selection.upper()

        mask_ycl = (self.final_table[f'{self.cc_class_key}_{selection2}'] == 'YCL')
        mask_map = (self.final_table[f'{self.cc_class_key}_{selection2}'] == 'MAP')
        mask_ogcc = (self.final_table[f'{self.cc_class_key}_{selection2}'] == 'OGCC')

        ycl_table_filtered = self.final_table[mask_ycl]
        map_table_filtered = self.final_table[mask_map]
        ogcc_table_filtered = self.final_table[mask_ogcc]

        pos_ycl_filtered = SkyCoord(ycl_table_filtered['PHANGS_RA'], ycl_table_filtered['PHANGS_DEC'])
        pos_map_filtered = SkyCoord(map_table_filtered['PHANGS_RA'], map_table_filtered['PHANGS_DEC'])
        pos_ogcc_filtered = SkyCoord(ogcc_table_filtered['PHANGS_RA'], ogcc_table_filtered['PHANGS_DEC'])

        logger.info('Originaly')
        for n_cat_o, cat_o in zip(['all', 'ycl', 'map', 'ogcc'], self.catalogs[self.selection]):
            logger.info(f'{n_cat_o}: {len(cat_o)}')

        logger.info('Filtered')
        for n_cat_f, cat_f in zip(['all', 'ycl', 'map', 'ogcc'], [self.final_table, ycl_table_filtered, map_table_filtered, ogcc_table_filtered]):
            logger.info(f'{n_cat_f}: {len(cat_f)}')
            

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 8))#, sharex=True, sharey=True)

        #HST
        
        x_ycl_hst, y_ycl_hst = skycoord_to_pixel(pos_ycl_filtered, self.hst_wcs_conv)        
        x_map_hst, y_map_hst = skycoord_to_pixel(pos_map_filtered, self.hst_wcs_conv)   
        x_ogcc_hst, y_ogcc_hst = skycoord_to_pixel(pos_ogcc_filtered, self.hst_wcs_conv)

        ax1.imshow(self.hst_conv, norm=get_norm(self.hst_conv, 15, 99.9, 'asinh'))
        ax1.scatter(x_ycl_hst, y_ycl_hst, c = 'blue', label = 'YCL')
        ax1.scatter(x_map_hst, y_map_hst, c = 'orange', label = 'MAP')
        ax1.scatter(x_ogcc_hst, y_ogcc_hst, c = 'red', label = 'OGCC')

        for reg in self.regions_final[0]:
            ax1.plot(reg.vertices.x, reg.vertices.y, '-', linewidth=2, color='purple')

        ax1.legend()
        ax1.set_title(f'{self.source.gal} HST F555W')

        #MUSE

        x_ycl_muse, y_ycl_muse = skycoord_to_pixel(pos_ycl_filtered, self.muse_wcs)        
        x_map_muse, y_map_muse = skycoord_to_pixel(pos_map_filtered, self.muse_wcs)   
        x_ogcc_muse, y_ogcc_muse = skycoord_to_pixel(pos_ogcc_filtered, self.muse_wcs)
        
        
        ax2.imshow(self.muse_data, norm=get_norm(self.muse_data, 0, 99.85, 'asinh'))
        ax2.scatter(x_ycl_muse, y_ycl_muse, c = 'blue', label = 'YCL')
        ax2.scatter(x_map_muse, y_map_muse, c = 'orange', label = 'MAP')
        ax2.scatter(x_ogcc_muse, y_ogcc_muse, c = 'red', label = 'OGCC')

        for reg in self.regions_final[1]:
            ax2.plot(reg.vertices.x, reg.vertices.y, '-', linewidth=2, color='purple')

        ax2.legend()
        ax2.set_title(f'{self.source.gal} MUSE FLUX')

        plt.tight_layout()
        plt.show()

    
    
    # ─────────────────────────────────────────────────────────────────────────────
    # SAVE
    # ─────────────────────────────────────────────────────────────────────────────

    def save_results(self, base_dir="output"):
        """
        Save all pipeline products for one galaxy to disk.

        File layout
        -----------
        output/
        <galaxy>/
            <galaxy>_HST.fits        HDU 0 – primary (metadata header)
                                    HDU 1 – BinTable: filtered HST source table
                                    HDU 2 – Image:    watershed label map (+ WCS)
            <galaxy>_MUSE.fits       same structure for MUSE
            <galaxy>_matched.fits    HDU 0 – primary (metadata header)
                                    HDU 1 – BinTable: matched/joined source table
            <galaxy>_HST.reg         DS9 region file for HST isolated segments
            <galaxy>_MUSE.reg        DS9 region file for MUSE isolated segments

        Notes
        -----
        * Regions can be loaded in DS9 directly or reconstructed with
        ``Regions.read(path)``.
        * A mask for any region can be recovered on demand with
        ``region.to_mask().to_image(data.shape)`` — no need to store it.
        * The WCS embedded in the label ImageHDU matches the convolved image WCS,
        so the label map is georeferenced and can be reprojected if needed.

        Parameters
        ----------
        base_dir : str
            Root output directory (default: "output/"). Created if absent.
        """

        gal   = self.source.gal            # e.g. "NGC1234"
        outd  = _galaxy_output_dir(base_dir, gal)
        stamp = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"

        # ── shared primary-HDU metadata header ───────────────────────────────────
        def _primary_hdr(instrument):
            hdr = fits.Header()
            hdr["GALAXY"]  = (gal,   "Galaxy name")
            hdr["INSTRUME"] = (instrument, "Instrument / image origin")
            hdr["CREATED"] = (stamp, "UTC creation timestamp")
            hdr["PIPELINE"] = ("IsolatedSourceFinder", "Pipeline class")
            # settings the products depend on: run_full_pipeline reruns the
            # ISF when these differ from the current config
            hdr["PSFSRC"]  = (getattr(self, 'psf_source', 'table'), "PSF source: theoretical or table")
            hdr["SAMPLE"]  = (str(self.sample), "MUSE sample")
            hdr["HSTPSF"]  = (float(self.hst_psf or 0.0), "HST F555W PSF FWHM used [arcsec]")
            hdr["MUSEPSF"] = (float(self.muse_psf or 0.0), "MUSE PSF FWHM used [arcsec]")
            return hdr

        # ── 1. HST FITS: source table + label map ────────────────────────────────
        hst_path = os.path.join(outd, f"{gal}_HST.fits")

        # HDU 0 – primary (no data, just header)
        hdu0_hst = fits.PrimaryHDU(header=_primary_hdr("HST"))

        # HDU 1 – filtered HST source table
        hdu1_hst = fits.BinTableHDU(
            data=self.table_fil_HST.as_array(),
            name="SOURCES"
        )
        # Carry column units if present in the astropy Table
        for col in self.table_fil_HST.colnames:
            if self.table_fil_HST[col].unit is not None:
                hdu1_hst.columns[col].unit = str(self.table_fil_HST[col].unit)

        # HDU 2 – watershed label map, georeferenced with the convolved WCS
        label_hdr_hst = self.hst_wcs_conv.to_header()
        label_hdr_hst["BUNIT"]   = ("label_id", "Watershed segment integer ID")
        label_hdr_hst["BZERO"]   = 0
        label_hdr_hst["BSCALE"]  = 1
        hdu2_hst = fits.ImageHDU(
            data=self.labels_HST.astype(np.int32),
            header=label_hdr_hst,
            name="LABELS"
        )

        fits.HDUList([hdu0_hst, hdu1_hst, hdu2_hst]).writeto(hst_path, overwrite=True)
        logger.info(f"  Saved HST FITS  → {hst_path}")

        # ── 2. MUSE FITS: source table + label map ───────────────────────────────
        muse_path = os.path.join(outd, f"{gal}_MUSE.fits")

        hdu0_muse = fits.PrimaryHDU(header=_primary_hdr("MUSE"))

        hdu1_muse = fits.BinTableHDU(
            data=self.table_fil_MUSE.as_array(),
            name="SOURCES"
        )
        for col in self.table_fil_MUSE.colnames:
            if self.table_fil_MUSE[col].unit is not None:
                hdu1_muse.columns[col].unit = str(self.table_fil_MUSE[col].unit)

        label_hdr_muse = self.muse_wcs.to_header()
        label_hdr_muse["BUNIT"]  = ("label_id", "Watershed segment integer ID")
        label_hdr_muse["BZERO"]  = 0
        label_hdr_muse["BSCALE"] = 1
        hdu2_muse = fits.ImageHDU(
            data=self.labels_MUSE.astype(np.int32),
            header=label_hdr_muse,
            name="LABELS"
        )

        fits.HDUList([hdu0_muse, hdu1_muse, hdu2_muse]).writeto(muse_path, overwrite=True)
        logger.info(f"  Saved MUSE FITS → {muse_path}")

        # ── 3. Matched catalog FITS ──────────────────────────────────────────────
        matched_path = os.path.join(outd, f"{gal}_matched.fits")

        hdu0_mat = fits.PrimaryHDU(header=_primary_hdr("HST+MUSE"))
        hdu1_mat = fits.BinTableHDU(
            data=self.table_final_filtered.as_array(),
            name="MATCHED"
        )
        for col in self.table_final_filtered.colnames:
            if self.table_final_filtered[col].unit is not None:
                hdu1_mat.columns[col].unit = str(self.table_final_filtered[col].unit)

        fits.HDUList([hdu0_mat, hdu1_mat]).writeto(matched_path, overwrite=True)
        logger.info(f"  Saved matched   → {matched_path}")


        # ── 4. Region files ──────────────────────────────────────────────────────
        reg_hst_path  = os.path.join(outd, f"{gal}_HST.reg")
        reg_muse_path = os.path.join(outd, f"{gal}_MUSE.reg")

        Regions(list(np.array(self.regions_final[0])[self.mask_isolated])).write(reg_hst_path,   overwrite=True)
        Regions(list(np.array(self.regions_final[1])[self.mask_isolated])).write(reg_muse_path, overwrite=True)

        logger.info(f"  Saved HST  .reg → {reg_hst_path}")
        logger.info(f"  Saved MUSE .reg → {reg_muse_path}")

        self.status = "Saved"
        logger.info(f"\n  ✓ All products saved for {gal}") 

    # ─────────────────────────────────────────────────────────────────────────────
    # LOAD
    # ─────────────────────────────────────────────────────────────────────────────

    def load_results(self, base_dir="output"):
        """
        Reload pipeline products saved by ``save_results`` back into the object.

        Reconstructed attributes
        ------------------------
        self.table_fil_HST   : astropy Table
        self.labels_HST      : 2-D int32 ndarray  (watershed label map)
        self.hst_wcs_conv    : astropy WCS         (recovered from label HDU header)
        self.regions_HST     : list[PolygonPixelRegion]

        self.table_fil_MUSE  : astropy Table
        self.labels_MUSE     : 2-D int32 ndarray
        self.muse_wcs        : astropy WCS         (recovered from label HDU header)
        self.regions_MUSE    : list[PolygonPixelRegion]

        self.final_table     : astropy Table

        Parameters
        ----------
        base_dir : str
            Root output directory (default: "output/").

        Raises
        ------
        FileNotFoundError
            If any expected output file is missing for this galaxy.
        """

        gal  = self.source.gal
        outd = os.path.join(base_dir, gal)

        hst_path     = os.path.join(outd, f"{gal}_HST.fits")
        muse_path    = os.path.join(outd, f"{gal}_MUSE.fits")
        matched_path = os.path.join(outd, f"{gal}_matched.fits")
        #matched2_path = os.path.join(outd, f"{gal}_matched2.fits")
        reg_hst_path  = os.path.join(outd, f"{gal}_HST.reg")
        reg_muse_path = os.path.join(outd, f"{gal}_MUSE.reg")

        for fpath in [hst_path, muse_path, matched_path, reg_hst_path, reg_muse_path]:
            if not os.path.exists(fpath):
                raise FileNotFoundError(
                    f"Expected output file not found: {fpath}\n"
                    f"Run the pipeline and call save_results() first."
                )

        # ── HST FITS ─────────────────────────────────────────────────────────────
        with fits.open(hst_path) as hdul:
            # HDU 1: source table
            self.table_fil_HST = Table(hdul["SOURCES"].data)

            # HDU 2: label map + WCS
            self.labels_HST    = hdul["LABELS"].data.astype(np.int32)
            self.hst_wcs_conv  = WCS(hdul["LABELS"].header)

        logger.info(f"  Loaded HST FITS  ← {hst_path}")

        # ── MUSE FITS ────────────────────────────────────────────────────────────
        with fits.open(muse_path) as hdul:
            self.table_fil_MUSE = Table(hdul["SOURCES"].data)
            self.labels_MUSE    = hdul["LABELS"].data.astype(np.int32)
            self.muse_wcs       = WCS(hdul["LABELS"].header)

        logger.info(f"  Loaded MUSE FITS ← {muse_path}")

        # ── Matched catalog FITS ─────────────────────────────────────────────────
        with fits.open(matched_path) as hdul:
            self.final_table = Table(hdul["MATCHED"].data)

        #with fits.open(matched2_path) as hdul:
        #    self.final_table_filtered = Table(hdul["MATCHED2"].data)

        logger.info(f"  Loaded matched   ← {matched_path}")

        # ── Region files ─────────────────────────────────────────────────────────
        # Regions.read() returns a Regions container; convert to plain list so
        # it matches the type produced by find_isolated.
        self.regions_HST  = list(Regions.read(reg_hst_path,  format="ds9"))
        self.regions_MUSE = list(Regions.read(reg_muse_path, format="ds9"))

        logger.info(f"  Loaded HST  .reg ← {reg_hst_path}  ({len(self.regions_HST)} regions)")
        logger.info(f"  Loaded MUSE .reg ← {reg_muse_path}  ({len(self.regions_MUSE)} regions)")

        self.status = "Loaded from disk"
        logger.info(f"\n  ✓ All products loaded for {gal}")
    
    # ---------------------------------------------------------
    # PIPELINE RUNNER
    # ---------------------------------------------------------
    def run_full_pipeline(self):
        """
        Run the entire workflow:
        - load images
        - measure PSFs
        - convolve and reproject HST
        - compute SB threshold
        - find sources
        - match catalogs
        """

        logger.info(f'{self.source.gal} starting')

        self.timing = {}

        # Turn warnings off inside the pipeline
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
    
            # ---------- 1. Load input data ----------
            t0 = time.perf_counter()
            self.load_images()
            self.get_catalogs()

            self.timing['load_images'] = time.perf_counter() - t0
            logger.info(f"load_images:                {format_time(self.timing['load_images'])}")
            logger.info("---------------------------------------")

            # ---------- 2. PSF estimation ----------
            t0 = time.perf_counter()
            self.measure_hst_psf()
            self.measure_muse_psf()
    
            self.timing['measure_psfs'] = time.perf_counter() - t0
            logger.info(f"measure_psfs:               {format_time(self.timing['measure_psfs'])}")
            logger.info("---------------------------------------")

            # ---------- 3. Convolution + Reprojection ----------
            t0 = time.perf_counter()
            self.convolve()
            self.reproject_hst_to_muse()
    
            self.timing['convolution_reprojection'] = time.perf_counter() - t0
            logger.info(f"convolution_reprojection:   {format_time(self.timing['convolution_reprojection'])}")
            logger.info("---------------------------------------")

            # ---------- 4. Surface-brightness threshold ----------
            t0 = time.perf_counter()
            self.compute_hst_sb_threshold()
    
            self.timing['sb_threshold'] = time.perf_counter() - t0
            logger.info(f"sb_threshold:               {format_time(self.timing['sb_threshold'])}")
            logger.info("---------------------------------------")

            # ---------- 5. Source detection ----------
            t0 = time.perf_counter()
            self.find_hst_sources('machine')
            self.find_muse_sources('machine')
    
            self.timing['source_detection'] = time.perf_counter() - t0
            logger.info(f"source_detection:           {format_time(self.timing['source_detection'])}")
            logger.info("---------------------------------------")

            # ---------- 6. Catalog matching ----------
            t0 = time.perf_counter()
            self.match_catalogs()
            
            self.timing['catalog_matching'] = time.perf_counter() - t0
            logger.info(f"catalog_matching:           {format_time(self.timing['catalog_matching'])}")
            logger.info("---------------------------------------")

            # ---------- 7. Catalog matching 2 ----------
            t0 = time.perf_counter()
            self.second_filter()
                        
            self.timing['second_filter'] = time.perf_counter() - t0
            logger.info(f"second_filter:           {format_time(self.timing['second_filter'])}")
            logger.info("---------------------------------------")

            # ---------- 8. Save results ----------
            t0 = time.perf_counter()
            self.save_results()

            self.timing['save_results'] = time.perf_counter() - t0
            logger.info(f"save_results:           {format_time(self.timing['save_results'])}")
            logger.info("---------------------------------------")

        # ---------- Total elapsed time ----------
        self.timing['total'] = sum(self.timing.values())
        logger.info(f"TOTAL TIME {self.source.gal}:  {format_time(self.timing['total'])}")


"""
photometry_class.py
-------------------
Photometry pipeline for PHANGS star clusters.

Loads IsolatedSourceFinder products, convolves multi-band images to a common
(MUSE) resolution, and measures aperture photometry + MUSE spectra for every
matched source.

Required imports from your existing modules
-------------------------------------------
From aux_functions.py:
    get_norm
    save_psf_to_table
    load_psf_from_table
    segm_data                    (used indirectly via IsolatedSourceFinder)

From psf_fitting_updated.py:
    measure_psf_fwhm_from_image
    build_muse_epsf
    fit_2d_moffat
    fit_2d_gaussian
    make_galaxy_mask

From photometry_functions.py  (the document you shared):
    to_Jy
    find_id
    corr_flux
    radius_circle
    measure_photometry
    spectra_from_aperture

Standard / third-party:
    numpy, astropy, photutils, scipy, regions
"""

class Photometry:
    """
    End-to-end aperture photometry pipeline for one galaxy.

    Workflow
    --------
    1. ``load_isf_results``   — read IsolatedSourceFinder products from disk
    2. ``load_images``        — load all HST and JWST FITS images
    3. ``load_or_measure_psf``— read PSF table; measure any missing bands
    4. ``convolve_images``    — convolve each band to MUSE resolution
    5. ``run_photometry``     — measure flux + spectrum for every source
    6. ``save_photometry``    — write results table to FITS

    Parameters
    ----------
    source       : sequence, indexable — source[0] is the MUSE file-name token
                  (e.g. "NGC4254-0.89asec"), source[1] is the clean galaxy
                  name (e.g. "NGC4254"). Note this is independent of
                  IsolatedSourceFinder's own `source` attribute (an
                  analysis.Source namedtuple) — Photometry accepts any
                  2-element indexable here, most commonly built ad hoc by
                  the caller rather than passed through from an
                  IsolatedSourceFinder instance.
    base_dir     : str     root output directory where ISF products live
    psf_filepath : str     path to the shared PSF CSV table (psf_source='table')
    psf_source   : 'theoretical' (default): every PSF FWHM / shape comes from
                   the theoretical PSF models (STScI effective PSFs for HST,
                   stpsf for JWST; instrument_psf.theoretical_psf) and the MUSE
                   FWHM from the token name ('NGC1365-1.15asec' -> 1.15"); no
                   PSF is measured on the images and psf_table.csv is neither
                   read nor written. 'table': the old behaviour (psf_table.csv,
                   measuring missing bands on the images).
    """

    def __init__(self, source, base_dir='output', psf_filepath='psf_table.csv',
                 psf_source='theoretical'):
        if psf_source not in ('theoretical', 'table'):
            raise ValueError(f"psf_source must be 'theoretical' or 'table', got {psf_source!r}")
        self.source       = source
        self.gal          = source[1]
        self.base_dir     = base_dir
        self.psf_filepath = psf_filepath
        self.psf_source   = psf_source

        # ── IsolatedSourceFinder products ────────────────────────────────────
        self.final_table  = None    # matched source catalog
        self.regions_HST  = None    # list[PolygonSkyRegion]
        self.regions_MUSE = None    # list[PolygonSkyRegion]
        self.hst_wcs_conv = None    # WCS of the convolved HST image
        self.muse_wcs     = None    # WCS of the MUSE cube

        # ── images  (band → (data, header, wcs)) ────────────────────────────
        self.hst_images   = {}      # original HST images
        self.jwst_images  = {}      # original JWST images
        self.muse_id_maps = None
        self.muse_data    = None    # MUSE flux map (2D, collapsed or white-light)
        self.muse_hdr = None
        self.muse_wcs = None

        # Convolved images — filled by convolve_images()
        self.hst_conv     = {}      # band → convolved data array
        self.jwst_conv    = {}      # band → convolved data array

        # ── PSF  (FWHM in arcsec) ────────────────────────────────────────────
        self.psf_muse     = None
        self.psf_hst      = {}      # band → fwhm_arcsec
        self.psf_jwst     = {}      # band → fwhm_arcsec

        # ── PSF shape (Moffat alpha/gamma) — measure_native_psf_shape() only,
        # feeds run_native_photometry(); independent of psf_hst/psf_jwst above
        self.psf_hst_alpha  = {}    # band → alpha
        self.psf_hst_gamma  = {}    # band → gamma
        self.psf_jwst_alpha = {}    # band → alpha
        self.psf_jwst_gamma = {}    # band → gamma

        # ── results ──────────────────────────────────────────────────────────
        self.photometry_table = None

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Load IsolatedSourceFinder results
    # ─────────────────────────────────────────────────────────────────────────

    def load_isf_results(self):
        """
        Read the IsolatedSourceFinder products written by ``save_results``.

        Populates
        ---------
        self.final_table, self.regions_HST, self.regions_MUSE,
        self.hst_wcs_conv, self.muse_wcs
        """
        outd = os.path.join(self.base_dir, self.gal)

        matched_path  = os.path.join(outd, f'{self.gal}_matched.fits')
        hst_fits_path = os.path.join(outd, f'{self.gal}_HST.fits')
        muse_fits_path= os.path.join(outd, f'{self.gal}_MUSE.fits')
        reg_hst_path  = os.path.join(outd, f'{self.gal}_HST.reg')
        reg_muse_path = os.path.join(outd, f'{self.gal}_MUSE.reg')

        for p in [matched_path, hst_fits_path, muse_fits_path,
                  reg_hst_path, reg_muse_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f'ISF product not found: {p}\n'
                    f'Run IsolatedSourceFinder and save_results() first.'
                )

        with fits.open(matched_path) as hdul:
            self.final_table = Table(hdul['MATCHED'].data)

        with fits.open(hst_fits_path) as hdul:
            self.hst_wcs_conv = WCS(hdul['LABELS'].header)

        with fits.open(muse_fits_path) as hdul:
            self.muse_wcs = WCS(hdul['LABELS'].header)

        self.regions_HST  = list(Regions.read(reg_hst_path,  format='ds9'))
        self.regions_MUSE = list(Regions.read(reg_muse_path, format='ds9'))

        #idx1, idx2 = self.final_table['_row1'], self.final_table['_row2']

        self.regions_HST_final =  self.regions_HST #self.regions_HST#[self.regions_HST[i] for i in idx1]
        self.regions_MUSE_final =  self.regions_MUSE#[self.regions_MUSE[i] for i in idx2]

        logger.info(f'  ✓ ISF results loaded for {self.gal}')
        logger.info(f'    {len(self.final_table)} matched sources')
        logger.info(f'    {len(self.regions_HST)} HST regions, '
              f'{len(self.regions_MUSE)} MUSE regions')
        logger.info(f'    {len(self.regions_HST_final)} final regions')

    def load_catalog_sources(self, cat_path, selection='machine', classes=('1+2',), drop_outside=False):
        """
        Use the full PHANGS-HST DR5 cluster catalog of this galaxy as the
        source list instead of the IsolatedSourceFinder selection ("catalog
        mode"): every cluster, isolated or not, with or without MUSE.

        selection : 'machine' (default — the catalog the ISF selection starts
                    from, class from PHANGS_CC_CLASS_ML) or 'human'
        classes   : which DR5 primary files, ('1+2',) or ('1+2', '3')
        drop_outside : drop clusters whose colour-colour class is 'outside' (not
                    YCL/MAP/OGCC), as the ISF selection does; kept by default
                    (they get DEFAULT_CLUSTER_PROFILE, EFF gamma=3)

        Populates self.final_table (same columns as the ISF table, so the
        native/instrument photometry runs unchanged); there are no ISF
        segmentation regions, so run_spectro_photometry (convolved
        photometry + MUSE spectra) cannot run on it — build the photometry
        table with init_photometry_table_from_catalog() instead.
        """
        galaxy_name = self.gal.lower()
        if galaxy_name == 'ngc0628':
            galaxy_name = 'ngc628c'          # same pointing as get_catalogs()
        parts = []
        for cls in classes:
            path = os.path.join(cat_path, f'hlsp_phangs-cat_hst_acs-uvis_dr5-targets_v1_{selection}-class{cls}-primary.fits')
            t = Table.read(path)
            names = np.char.lower(np.char.strip(np.asarray(t['PHANGS_GALAXY']).astype(str)))
            parts.append(t[names == galaxy_name])
            logger.info(f'  catalog {os.path.basename(path)}: {int((names == galaxy_name).sum())} sources')
        from astropy.table import vstack
        self.final_table = vstack(parts, metadata_conflicts='silent') if len(parts) > 1 else parts[0]
        key = 'PHANGS_CC_CLASS_ML' if selection == 'machine' else 'PHANGS_CC_CLASS_HUMAN'
        if key in self.final_table.colnames:
            cc = np.char.strip(np.asarray(self.final_table[key]).astype(str))
            n_out = int((cc == 'outside').sum())
            if drop_outside:
                self.final_table = self.final_table[cc != 'outside']
            logger.info(f"  {n_out} sources with {key} = 'outside' "
                        + ('dropped' if drop_outside else 'kept (drop_outside=False)'))
        if len(self.final_table) == 0:
            raise RuntimeError(f'no {selection} class {classes} clusters for {galaxy_name} in {cat_path}')
        self.source_mode = 'catalog'
        self.regions_HST_final, self.regions_MUSE_final = [], []
        logger.info(f'  ✓ Catalog mode: {len(self.final_table)} {selection} class {"+".join(classes)} '
                    f'sources for {self.gal}')

    def init_photometry_table_from_catalog(self):
        """photometry_table = the source list itself (catalog mode: no convolved
        photometry / MUSE spectra); the native and instrument photometry then
        add their columns to it as usual."""
        if self.final_table is None:
            raise RuntimeError('No final_table — run load_catalog_sources() first.')
        self.photometry_table = Table(self.final_table, copy=True)

    def load_custom_catalog(self, path, ra_col, dec_col, id_col=None, class_col=None,
                            distance_mpc=None, dist_col=None, fmt=None):
        """
        Any source list as the catalog (no PHANGS DR5, no ISF): a FITS / ECSV /
        CSV / any astropy-readable table with at least RA and Dec. Every column
        of the file is kept; the columns the photometry reads are filled under
        their internal names (a mapped column wins over a column of that name
        already in the file; an unmapped internal name already in the file is kept):

          ID_PHANGS_CLUSTER   id_col (default: row number, starting at 1)
          PHANGS_RA/DEC       ra_col, dec_col: degrees, or sexagesimal strings
                              ('hh:mm:ss', 'dd:mm:ss')
          PHANGS_DMPC         dist_col per source, else distance_mpc for all; used
                              for R_eff in pc and the 20 pc R_eff cap (without a
                              distance R_eff stays in arcsec, capped at 0.5")
          PHANGS_CC_CLASS_ML  class_col: YCL / MAP -> EFF gamma=3, OGCC -> King
                              c=1.5, anything else -> EFF gamma=3 (as 'outside')
          PHANGS_GALAXY       this object's galaxy / field name (CIGALE export)

        Like load_catalog_sources: populates self.final_table; follow with
        filter_by_common_fov(include_muse=False) and init_photometry_table_from_catalog().
        """
        t = Table.read(path, format=fmt) if fmt else Table.read(path)
        if len(t) == 0:
            raise RuntimeError(f'no sources in {path}')
        for col in (ra_col, dec_col, id_col, class_col, dist_col):
            if col and col not in t.colnames:
                raise KeyError(f'column {col!r} not in {path} (columns: {", ".join(t.colnames)})')

        ra, dec = t[ra_col], t[dec_col]
        if ra.dtype.kind in 'USO' or dec.dtype.kind in 'USO':
            c = SkyCoord(np.asarray(ra).astype(str), np.asarray(dec).astype(str), unit=(u.hourangle, u.deg))
            ra_deg, dec_deg = c.ra.deg, c.dec.deg
        else:
            ra_deg, dec_deg = np.asarray(ra, float), np.asarray(dec, float)
        t['PHANGS_RA'], t['PHANGS_DEC'] = ra_deg, dec_deg

        if id_col:
            ids = np.asarray(t[id_col])
            if ids.dtype.kind == 'f' and np.all(np.isfinite(ids)) and np.all(ids == np.round(ids)):
                ids = ids.astype(np.int64)                 # e.g. 384.0 after a table join
            t['ID_PHANGS_CLUSTER'] = ids
        elif 'ID_PHANGS_CLUSTER' not in t.colnames:
            t['ID_PHANGS_CLUSTER'] = np.arange(1, len(t) + 1)

        if class_col:
            t['PHANGS_CC_CLASS_ML'] = [str(c).strip() for c in np.asarray(t[class_col]).astype(str)]
        elif 'PHANGS_CC_CLASS_ML' not in t.colnames:
            t['PHANGS_CC_CLASS_ML'] = ['outside'] * len(t)

        if dist_col:
            t['PHANGS_DMPC'] = np.asarray(t[dist_col], float)
        elif distance_mpc is not None:
            t['PHANGS_DMPC'] = np.full(len(t), float(distance_mpc))
        elif 'PHANGS_DMPC' not in t.colnames:
            t['PHANGS_DMPC'] = np.full(len(t), np.nan)
            logger.warning('  no distance given: R_eff only in arcsec, R_eff cap 0.5"')

        if 'PHANGS_GALAXY' not in t.colnames:
            t['PHANGS_GALAXY'] = [self.gal.lower()] * len(t)

        self.final_table = t
        self.source_mode = 'catalog'
        self.regions_HST_final, self.regions_MUSE_final = [], []
        logger.info(f'  ✓ Custom catalog: {len(t)} sources from {os.path.basename(path)}')

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Load images
    # ─────────────────────────────────────────────────────────────────────────

    def load_images(self, hst_paths, jwst_paths, muse_path=None,
                    muse_ext=2, muse_wl_ext=None):
        """
        Load all HST and JWST images from FITS files.

        Parameters
        ----------
        hst_paths  : dict  band → filepath   e.g. {'f555w': 'NGC1234_f555w.fits'}
        jwst_paths : dict  band → filepath
        muse_path  : str or None   path to a MUSE white-light or collapsed image
                                   (only needed if running spectral extraction here)
        muse_ext   : int   FITS extension for the MUSE data (default 1)
        muse_wl_ext: int or None   FITS extension for MUSE wavelength axis

        Notes
        -----
        Each image is stored as ``(data, header, wcs)`` so the WCS is always
        available for region projection without re-opening the file.
        """
        for band, path in hst_paths.items():
            with fits.open(path) as hdul:
                hst_data = np.where(hdul[0].data == 0.0, np.nan, hdul[0].data)
                hst_header = hdul[0].header
                hst_wcs = WCS(hst_header)
            self.hst_images[band.lower()] = (hst_data, hst_header, hst_wcs)
            logger.info(f'  HST {band.upper():8s} loaded  shape={hst_data.shape}')

        for band, path in jwst_paths.items():
            with fits.open(path) as hdul:
                jwst_data = hdul[1].data
                jwst_header = hdul[1].header
                jwst_wcs = WCS(jwst_header)
            self.jwst_images[band.lower()] = (jwst_data, jwst_header, jwst_wcs)
            logger.info(f'  JWST {band.upper():7s} loaded  shape={jwst_data.shape}')

        if muse_path is not None:
            with fits.open(f'{muse_path}/{self.source[0]}/{self.source[0]}_DR2.2v2/{self.source[0]}_MAPS.fits') as hdul:
                self.muse_id_maps = [hdul[1].data.astype(float), hdul[5].data.astype(float)]
                self.muse_data = hdul[muse_ext].data.astype(float)
                self.muse_hdr = hdul[muse_ext].header
                self.muse_wcs = WCS(hdul[muse_ext].header)
            logger.info(f'  MUSE map loaded  shape={self.muse_data.shape}')

            self.spectra = Table.read(f'{muse_path}/{self.source[0]}/{self.source[0]}_DR2.2v2/{self.source[0]}_AllSpectra.fits', hdu=1)
            self.log_wl = Table.read(f'{muse_path}/{self.source[0]}/{self.source[0]}_DR2.2v2/{self.source[0]}_AllSpectra.fits', hdu=2)['LOGLAM']
            
            logger.info(f'  MUSE spectra loaded')

        logger.info(f'\n  ✓ Images loaded: {len(self.hst_images)} HST, '
              f'{len(self.jwst_images)} JWST bands')

    # ─────────────────────────────────────────────────────────────────────────
    # 3. PSF — load from table, measure missing bands
    # ─────────────────────────────────────────────────────────────────────────

    def load_or_measure_psf(self, hst_bands_to_measure=None,
                             jwst_bands_to_measure=None,
                             measure_muse=False,
                             profile='moffat',
                             galaxy_name=None,
                             fwhm_guess_hst=3.0,
                             fwhm_guess_jwst=3.0,
                             fwhm_guess_muse=7.0,
                             r_limit_hst=0.2,
                             r_limit_muse=2.0):
        """
        Load PSF values from the shared CSV; measure and save any that are
        missing (value == 0.0) or not yet in the table.

        Parameters
        ----------
        hst_bands_to_measure  : list[str] or None
            HST bands to (re-)measure even if already in the table.
            None → only measure bands that are missing / zero.
        jwst_bands_to_measure : list[str] or None
            Same for JWST.  JWST PSFs are stored in self.psf_jwst but NOT
            written to the shared CSV (which is HST+MUSE only).
        measure_muse  : bool   force re-measurement of MUSE PSF
        profile       : {'gaussian', 'moffat'}
        galaxy_name   : str or None   passed to make_galaxy_mask
        fwhm_guess_*  : float   initial DAOStarFinder FWHM guess per instrument
        r_limit_*     : float   FWHM upper sanity limit in arcsec per instrument

        With psf_source='theoretical' (the default) none of this happens:
        see _theoretical_psf.
        """
        if getattr(self, 'psf_source', 'table') == 'theoretical':
            self._theoretical_psf()
            return
        # ── read existing table ───────────────────────────────────────────────
        psf_row = {}
        if os.path.exists(self.psf_filepath):
            tbl = ascii.read(self.psf_filepath, format='csv')
            if self.gal in tbl['galaxy_name']:
                idx = list(tbl['galaxy_name']).index(self.gal)
                for col in tbl.colnames:
                    if col != 'galaxy_name':
                        psf_row[col] = float(tbl[col][idx])

        def _needs_measure(key):
            return psf_row.get(key, 0.0) == 0.0

        to_save = {}

        # ── MUSE PSF ─────────────────────────────────────────────────────────
        if measure_muse or _needs_measure('psf_MUSE'): 
            if self.muse_data is not None:
                logger.info('\n  Measuring MUSE PSF ...')
                fwhm_as, _, _, _, _ , _ = measure_psf_fwhm_from_image(
                    self.muse_data, wcs=self.muse_wcs, n_sources=10000,
                    lw_lim = 0.3, up_lim = 1.5,
                    fixed_alpha = True,
                    plot=False,
                )
                if fwhm_as is not None:
                    self.psf_muse = fwhm_as
                    to_save['psf_MUSE'] = fwhm_as
            else:
                logger.warning('MUSE data not loaded — cannot measure MUSE PSF')
        else:
            self.psf_muse = psf_row.get('psf_MUSE')
            logger.info(f'  psf_MUSE = {self.psf_muse:.3f}" (from table)')

        # ── HST PSFs ─────────────────────────────────────────────────────────
        for band, (data, hdr, wcs) in self.hst_images.items():
            col = f'psf_HST_{band}'
            force = hst_bands_to_measure is not None and band in hst_bands_to_measure
            if force or _needs_measure(col):
                logger.info(f'\n  Measuring HST {band.upper()} PSF ...')
                fwhm_as, _, _, _, _ , _ = measure_psf_fwhm_from_image(
                    data, wcs=wcs, n_sources = 10000,
                    lw_lim = 0.0, up_lim = 0.3,
                    fixed_alpha = True,
                    plot=False,
                )
                if fwhm_as is not None:
                    self.psf_hst[band] = fwhm_as
                    to_save[col] = fwhm_as
            else:
                self.psf_hst[band] = psf_row.get(col, 0.0)
                logger.info(f'  psf_HST_{band} = {self.psf_hst[band]:.3f}" (from table)')

        # ── JWST PSFs ─────────────────────────────────────────────────────────
        for band, (data, hdr, wcs) in self.jwst_images.items():
            col = f'psf_JWST_{band}'
            force = jwst_bands_to_measure is not None and band in jwst_bands_to_measure
            if force or _needs_measure(col):
                logger.info(f'\n  Measuring JWST {band.upper()} PSF ...')
                fwhm_as, _, _, _, _ , _ = measure_psf_fwhm_from_image(
                    data, wcs=wcs, n_sources = 10000,
                    lw_lim = 0.0, up_lim = 1.5,
                    fixed_alpha = True,
                    plot=False,
                )
                if fwhm_as is not None:
                    self.psf_jwst[band] = fwhm_as
                    to_save[col] = fwhm_as
                logger.info(f'  psf_JWST_{band} = {self.psf_jwst.get(band, "FAILED")}"')
            else:
                self.psf_jwst[band] = psf_row.get(col, 0.0)
                logger.info(f'  psf_JWST_{band} = {self.psf_jwst[band]:.3f}" (from table)')

        # ── save updated PSF values ───────────────────────────────────────────
        if to_save:
            save_psf_to_table(self.gal, to_save, filepath=self.psf_filepath)

        logger.info(f'\n  ✓ PSF summary for {self.gal}:')
        logger.info(f'    MUSE      : {self.psf_muse}"')
        for b, v in self.psf_hst.items():
            logger.info(f'    HST {b:8s}: {v}"')
        for b, v in self.psf_jwst.items():
            logger.info(f'    JWST {b:7s}: {v}"')

    def _theoretical_psf(self):
        """
        PSF FWHM and Moffat shape of every loaded band from the theoretical
        PSF models (instrument_psf.theoretical_psf: STScI effective PSF /
        stpsf, no extra blur, at each image's pixel scale), and the MUSE FWHM
        from the token name. Nothing is measured on the images and
        psf_table.csv is not touched.

        Populates self.psf_muse (None if the token has no '<x>asec'),
        self.psf_hst / self.psf_jwst (FWHM) and self.psf_hst_alpha/_gamma,
        self.psf_jwst_alpha/_gamma (Moffat shape, for run_native_photometry).
        """
        from instrument_psf import theoretical_psf, muse_fwhm_from_token
        self.psf_muse = muse_fwhm_from_token(self.source[0])
        for store, fwhm_d, alpha_d, gamma_d, inst in (
                (self.hst_images, self.psf_hst, self.psf_hst_alpha, self.psf_hst_gamma, 'HST'),
                (self.jwst_images, self.psf_jwst, self.psf_jwst_alpha, self.psf_jwst_gamma, 'JWST')):
            for band, (data, hdr, wcs) in store.items():
                pix = wcs.proj_plane_pixel_scales()[0].to('arcsec').value
                t = theoretical_psf(band, hdr, pix)
                if not np.isfinite(t['fwhm_arcsec']):
                    logger.warning(f'  {inst} {band}: no theoretical PSF ({t["note"]}) — band left without PSF')
                    continue
                fwhm_d[band] = t['fwhm_arcsec']
                alpha_d[band] = t['alpha']
                gamma_d[band] = t['gamma_px']
                if t['note']:
                    logger.info(f'  {inst} {band}: {t["note"]}')
        logger.info(f'\n  ✓ PSF summary for {self.gal} (theoretical PSFs; MUSE from the token name):')
        logger.info(f'    MUSE      : {self.psf_muse}"' if self.psf_muse else
                    f'    MUSE      : none (token {self.source[0]!r} has no "<x>asec")')
        for b in self.psf_hst:
            logger.info(f'    HST {b:8s}: FWHM {self.psf_hst[b]:.3f}"  alpha {self.psf_hst_alpha[b]:.2f}')
        for b in self.psf_jwst:
            logger.info(f'    JWST {b:7s}: FWHM {self.psf_jwst[b]:.3f}"  alpha {self.psf_jwst_alpha[b]:.2f}')

    # ─────────────────────────────────────────────────────────────────────────
    # 3b. PSF shape (alpha, gamma) — for run_native_photometry only
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _warn_if_alpha_unphysical(instrument, band, alpha, lo=1.0, hi=6.0):
        """
        A fitted Moffat alpha with fixed_alpha=False can converge to a
        numerically "successful" but physically meaningless value if the
        candidate star was actually a low-S/N/spurious DAOStarFinder
        detection rather than a real, well-resolved PSF star — the
        fit_2d_moffat convergence check only rejects fits parked exactly at
        its (1.5, 20.0) bounds, not fits that land on some other
        unconstrained value in between. Real Moffat PSFs are typically
        alpha ~1.5-4 (Elson et al. 1987; the fixed_alpha=True path
        elsewhere in this pipeline uses a literature alpha=2.8 as its
        working assumption) — this is a cheap sanity check, not a
        correctness guarantee, since a genuinely unusual PSF could fall
        outside this range too; treat a warning here as "go look at this
        band's fit before trusting it for native photometry", not as a
        guaranteed-wrong result.
        """
        if not (lo <= alpha <= hi):
            logger.warning(
                f'{instrument} {band}: measured Moffat alpha={alpha:.2f} is outside the '
                f'typical physical range [{lo}, {hi}] — likely contaminated by low-S/N or '
                f'spurious star detections feeding the fit (see measure_native_psf_shape\'s '
                f'n_sources docstring). Inspect this band before trusting run_native_photometry '
                f'results from it; consider a stricter n_sources or a hand-picked bright star.'
            )

    def measure_native_psf_shape(self, hst_bands_to_measure=None,
                                  jwst_bands_to_measure=None,
                                  fwhm_guess_hst=3.0, fwhm_guess_jwst=3.0,
                                  r_limit_hst=0.3, r_limit_jwst=1.5,
                                  n_sources=50):
        """
        Measure the per-band Moffat SHAPE parameter (alpha, with its paired
        gamma) for HST and JWST bands, with alpha genuinely free to vary.

        This is deliberately separate from load_or_measure_psf(): that
        method calls measure_psf_fwhm_from_image(..., fixed_alpha=True)
        everywhere, which holds alpha at its unmeasured initial guess of
        2.8 for the whole fit — correct and sufficient for its own purpose
        (FWHM only, feeding the PSF-CONVOLUTION-matching path, which is
        tested and working), but not a genuine measurement of the PSF's
        shape. run_native_photometry() needs an actually-measured alpha (to
        build a scientifically faithful Moffat model of the band's real
        PSF, rather than assuming a fixed literature value), so this method
        exists purely to feed that subsystem — load_or_measure_psf() and
        its FWHM columns are completely untouched by this.

        n_sources defaults to a modest 50 (vs. load_or_measure_psf's 10000)
        deliberately: with fixed_alpha free, a fit to a low-S/N or spurious
        DAOStarFinder detection can converge to a numerically "successful"
        but physically meaningless alpha anywhere in its bounds (unlike the
        FWHM-only fixed_alpha=True fits, which are far more robust to a bad
        candidate since alpha can't wander at all there) — with n_sources
        this large, a field with any non-trivial noise can pick up far more
        spurious low-flux detections than genuine stars, and those bad fits
        silently corrupt the aggregate median alpha (verified empirically:
        thousands of noise-level detections pulled the recovered alpha to
        ~18-19 instead of the true ~2.6 on a synthetic test field with only
        8 real stars). Restricting to the ~50 brightest candidates keeps
        this close to standard PSF-calibration practice (a modest number of
        bright, reliable stars) and avoids that failure mode; raise it only
        for a band/galaxy confirmed to have that many genuinely good stars.

        Reuses the same psf_table.csv file and the existing
        save_psf_to_table/load_psf_from_table caching helpers, but writes
        to NEW columns (psf_HST_{band}_alpha/_gamma,
        psf_JWST_{band}_alpha/_gamma) — save_psf_to_table only ever updates
        the columns it's given, so this cannot disturb the existing
        psf_HST_{band}/psf_JWST_{band} FWHM columns load_or_measure_psf
        reads and writes.

        Populates
        ---------
        self.psf_hst_alpha, self.psf_hst_gamma   : dict  band -> value
        self.psf_jwst_alpha, self.psf_jwst_gamma : dict  band -> value

        With psf_source='theoretical' the shape comes from the theoretical
        PSF models instead (set by load_or_measure_psf; computed here if not).
        """
        if getattr(self, 'psf_source', 'table') == 'theoretical':
            if not (self.psf_hst_alpha or self.psf_jwst_alpha):
                self._theoretical_psf()
            return
        self.psf_hst_alpha = {}
        self.psf_hst_gamma = {}
        self.psf_jwst_alpha = {}
        self.psf_jwst_gamma = {}

        psf_row = {}
        if os.path.exists(self.psf_filepath):
            tbl = ascii.read(self.psf_filepath, format='csv')
            if self.gal in tbl['galaxy_name']:
                idx = list(tbl['galaxy_name']).index(self.gal)
                for col in tbl.colnames:
                    if col != 'galaxy_name':
                        psf_row[col] = float(tbl[col][idx])

        def _needs_measure(key):
            return psf_row.get(key, 0.0) == 0.0

        to_save = {}

        for band, (data, hdr, wcs) in self.hst_images.items():
            col_a, col_g = f'psf_HST_{band}_alpha', f'psf_HST_{band}_gamma'
            force = hst_bands_to_measure is not None and band in hst_bands_to_measure
            if force or _needs_measure(col_a):
                logger.info(f'\n  Measuring HST {band.upper()} PSF shape (alpha free) ...')
                fwhm_as, _, _, _, alpha, gamma = measure_psf_fwhm_from_image(
                    data, wcs=wcs, n_sources=n_sources, fwhm=fwhm_guess_hst,
                    lw_lim=0.0, up_lim=r_limit_hst,
                    fixed_alpha=False, plot=False,
                )
                if alpha is not None:
                    self._warn_if_alpha_unphysical('HST', band, alpha)
                    self.psf_hst_alpha[band] = alpha
                    self.psf_hst_gamma[band] = gamma
                    to_save[col_a] = alpha
                    to_save[col_g] = gamma
            else:
                self.psf_hst_alpha[band] = psf_row.get(col_a, 0.0)
                self.psf_hst_gamma[band] = psf_row.get(col_g, 0.0)
                logger.info(f'  psf_HST_{band} shape: alpha={self.psf_hst_alpha[band]:.3f} '
                            f'gamma={self.psf_hst_gamma[band]:.3f} (from table)')

        for band, (data, hdr, wcs) in self.jwst_images.items():
            col_a, col_g = f'psf_JWST_{band}_alpha', f'psf_JWST_{band}_gamma'
            force = jwst_bands_to_measure is not None and band in jwst_bands_to_measure
            if force or _needs_measure(col_a):
                logger.info(f'\n  Measuring JWST {band.upper()} PSF shape (alpha free) ...')
                fwhm_as, _, _, _, alpha, gamma = measure_psf_fwhm_from_image(
                    data, wcs=wcs, n_sources=n_sources, fwhm=fwhm_guess_jwst,
                    lw_lim=0.0, up_lim=r_limit_jwst,
                    fixed_alpha=False, plot=False,
                )
                if alpha is not None:
                    self._warn_if_alpha_unphysical('JWST', band, alpha)
                    self.psf_jwst_alpha[band] = alpha
                    self.psf_jwst_gamma[band] = gamma
                    to_save[col_a] = alpha
                    to_save[col_g] = gamma
            else:
                self.psf_jwst_alpha[band] = psf_row.get(col_a, 0.0)
                self.psf_jwst_gamma[band] = psf_row.get(col_g, 0.0)
                logger.info(f'  psf_JWST_{band} shape: alpha={self.psf_jwst_alpha[band]:.3f} '
                            f'gamma={self.psf_jwst_gamma[band]:.3f} (from table)')

        if to_save:
            save_psf_to_table(self.gal, to_save, filepath=self.psf_filepath)

        logger.info(f'\n  ✓ PSF shape summary for {self.gal}:')
        for b, v in self.psf_hst_alpha.items():
            logger.info(f'    HST {b:8s}: alpha={v:.3f}')
        for b, v in self.psf_jwst_alpha.items():
            logger.info(f'    JWST {b:7s}: alpha={v:.3f}')

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Convolve images to MUSE resolution
    # ─────────────────────────────────────────────────────────────────────────

    def _gaussian_kernel(self, fwhm_from_arcsec, fwhm_to_arcsec, pixel_scale_arcsec):
        """
        Build a Gaussian convolution kernel to go from fwhm_from → fwhm_to.

        Returns None if fwhm_from >= fwhm_to (already at or below target
        resolution — no convolution needed). Delegates the actual sigma
        computation to aux_functions.gaussian_kernel_sigma_pixels, shared
        with aux_functions.convolution() so the two PSF-matching code paths
        (IsolatedSourceFinder vs. Photometry) can't drift apart.
        """
        sigma_pixels = gaussian_kernel_sigma_pixels(fwhm_from_arcsec, fwhm_to_arcsec, pixel_scale_arcsec)
        if sigma_pixels is None:
            return None

        return Gaussian2DKernel(x_stddev=sigma_pixels)

    def _conv_path(self, instrument, band):
        """Path used for both reading and writing a band's convolved-image cache."""
        return os.path.join(self.base_dir, self.gal, 'convolved',
                             f'{self.gal}_{instrument}_{band}_conv.fits')

    def _convolve_band(self, instrument, band, data, hdr, wcs, psf_native_dict):
        """
        Convolve one band's image to the MUSE PSF resolution, using the on-disk
        cache at self._conv_path(instrument, band) when available.

        Returns the convolved (or, if skipped, original) data array.
        """
        label = f'{instrument} {band:8s}' if instrument == 'HST' else f'{instrument} {band:7s}'
        path = self._conv_path(instrument, band)

        pscale = wcs.proj_plane_pixel_scales()[0].to('arcsec').value
        psf_b  = psf_native_dict.get(band, 0.0)

        if os.path.exists(path):
            with fits.open(path) as hdul:
                h0 = hdul[0].header
                same = (abs(float(h0.get('ORIGPSF', -1)) - psf_b) < 1e-4
                        and abs(float(h0.get('CONVPSF', -1)) - float(self.psf_muse)) < 1e-4)
                if same:
                    conv = hdul[0].data.astype(float)
            if same:
                logger.info(f'  {label}: convolved image loaded from disk')
                return conv
            logger.info(f'  {label}: cached convolved image made with other PSFs '
                        f'(ORIGPSF {h0.get("ORIGPSF")}, CONVPSF {h0.get("CONVPSF")}) — reconvolving')

        if psf_b == 0.0:
            logger.warning(f'{instrument} {band} PSF unknown — skipping convolution')
            return data

        kernel = self._gaussian_kernel(psf_b, self.psf_muse, pscale)
        if kernel is None:
            logger.info(f'  {label}: PSF ({psf_b:.3f}") ≥ MUSE ({self.psf_muse:.3f}") '
                  f'— no convolution')
            return data

        logger.info(f'  {label}: convolving {psf_b:.3f}" → {self.psf_muse:.3f}" ...')
        conv = convolve_fft(
            data, kernel,
            nan_treatment='interpolate',
            normalize_kernel=True,
            allow_huge=True,
        ).astype(float)

        orig_hdr = hdr.copy()
        orig_hdr['CONVPSF']  = (self.psf_muse, 'Target PSF FWHM [arcsec]')
        orig_hdr['ORIGPSF']  = (psf_b,         'Original PSF FWHM [arcsec]')
        orig_hdr['CONVBAND'] = (band.upper(),  'Band')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fits.writeto(path, conv.astype('float32'), header=orig_hdr, overwrite=True)
        logger.info(f'  Saved {path}')

        return conv

    def convolve_images(self):
        """
        Convolve all HST and JWST images to the MUSE PSF resolution.

        Uses quadratic differencing: kernel_FWHM = sqrt(MUSE_PSF² − band_PSF²).
        Bands whose PSF is already larger than the MUSE PSF are left unchanged
        (a warning is printed — this would require deconvolution).

        Populates
        ---------
        self.hst_conv  : dict  band → convolved data array
        self.jwst_conv : dict  band → convolved data array
        """
        if self.psf_muse is None:
            raise RuntimeError('MUSE PSF not set — run load_or_measure_psf() first (with the theoretical '
                               'PSFs it comes from the token name, e.g. NGC1365-1.15asec).')

        for band, (data, hdr, wcs) in self.hst_images.items():
            self.hst_conv[band] = self._convolve_band('HST', band, data, hdr, wcs, self.psf_hst)

        for band, (data, hdr, wcs) in self.jwst_images.items():
            self.jwst_conv[band] = self._convolve_band('JWST', band, data, hdr, wcs, self.psf_jwst)

        logger.info(f'\n  ✓ Convolution done: {len(self.hst_conv)} HST, '
              f'{len(self.jwst_conv)} JWST bands')

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Build convolved image dicts for the photometry functions
    # ─────────────────────────────────────────────────────────────────────────

    def _build_conv_dicts(self):
        """
        Return hst_images_conv and jwst_images_conv in the format expected by
        ``measure_photometry``:  band → (convolved_data, header, wcs).
        """
        hst_conv_dict = {
            band: (self.hst_conv[band],
                   self.hst_images[band][1],
                   self.hst_images[band][2])
            for band in self.hst_conv
        }
        jwst_conv_dict = {
            band: (self.jwst_conv[band],
                   self.jwst_images[band][1],
                   self.jwst_images[band][2])
            for band in self.jwst_conv
        }
        return hst_conv_dict, jwst_conv_dict

    def filter_by_common_fov(self, plot=True, include_muse=True):
        """
        Keep only sources whose sky coordinates fall inside the valid footprint
        of EVERY loaded image (HST + JWST + MUSE).

        This is necessary when combining instruments with different FOVs —
        sources identified from HST/MUSE may fall outside the smaller JWST
        NIRCam FOV, producing zero or NaN fluxes in those bands.

        Procedure
        ---------
        For each image in ``hst_images``, ``jwst_images``, and the MUSE WCS:

        1. Build a valid pixel mask: ``np.isfinite(data) & (data != 0)``.
        2. Project each source's (RA, Dec) onto that image's pixel grid.
        3. Check the source pixel is (a) within image bounds and
            (b) lands on a valid pixel.

        A source survives only if it passes all images.  No polygon
        intersection is computed — just direct per-source pixel look-ups,
        one array operation per image.

        Populates
        ---------
        self.final_table        : filtered in-place (original stored in
                                    ``self.final_table_unfiltered``)
        self.regions_HST_final  : list of HST regions for surviving sources
        self.regions_MUSE_final : list of MUSE regions for surviving sources
        self.fov_mask           : 1-D bool array (True = source kept)

        Parameters
        ----------
        plot : bool
            Show a diagnostic plot with the effective FOV overlap and the
            sources before/after filtering on the MUSE WCS footprint.
        include_muse : bool
            Also require the MUSE footprint (default). False in catalog mode,
            where no MUSE spectra are extracted.

        Returns
        -------
        fov_mask : 1-D bool ndarray   True for sources inside all FOVs
        """
        if self.final_table is None:
            raise RuntimeError('No final_table — run load_isf_results() first.')

        # ── source sky coordinates ────────────────────────────────────────────
        ra  = np.array(self.final_table['PHANGS_RA'],  dtype=float)
        dec = np.array(self.final_table['PHANGS_DEC'], dtype=float)
        n   = len(ra)

        # Cumulative boolean keep-mask — starts all True
        keep = np.ones(n, dtype=bool)

        # ── build image inventory ─────────────────────────────────────────────
        # Each entry: (label, data_array, wcs)
        images_to_check = []

        for band, (data, hdr, wcs) in self.hst_images.items():
            images_to_check.append((f'HST {band.upper()}', data, wcs))

        for band, (data, hdr, wcs) in self.jwst_images.items():
            images_to_check.append((f'JWST {band.upper()}', data, wcs))

        if include_muse and self.muse_data is not None and self.muse_wcs is not None:
            images_to_check.append(('MUSE', self.muse_data, self.muse_wcs))

        logger.info(f'  Checking {n} sources against {len(images_to_check)} images ...\n')

        # ── per-image footprint check ─────────────────────────────────────────
        per_image_counts = {}

        for label, data, wcs in images_to_check:
            ny, nx = data.shape

            # Valid pixel footprint
            valid = np.isfinite(data) & (data != 0)

            # Project all source sky coords onto this image pixel grid
            sky  = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
            px, py = sky.to_pixel(wcs)

            xi = np.round(px).astype(int)
            yi = np.round(py).astype(int)

            # In-bounds check (vectorised)
            in_bounds = (xi >= 0) & (xi < nx) & (yi >= 0) & (yi < ny)

            # Valid pixel check — only for in-bounds sources
            on_valid  = np.zeros(n, dtype=bool)
            on_valid[in_bounds] = valid[yi[in_bounds], xi[in_bounds]]

            image_mask = in_bounds & on_valid
            n_pass     = int(image_mask.sum())
            n_fail     = n - n_pass

            per_image_counts[label] = (n_pass, n_fail)
            logger.info(f'  {label:20s}  pass={n_pass:4d}  fail={n_fail:4d}')

            keep &= image_mask

        # ── results ───────────────────────────────────────────────────────────
        n_kept    = int(keep.sum())
        n_removed = n - n_kept
        logger.info(f'\n  Sources before FOV filter : {n}')
        logger.info(f'  Sources removed           : {n_removed}')
        logger.info(f'  Sources kept (all FOVs)   : {n_kept}')

        # Store unfiltered version before overwriting
        self.final_table_unfiltered = self.final_table.copy()
        self.fov_mask               = keep

        # Filter table and region lists
        self.final_table = self.final_table[keep]
        # ISF regions (absent in catalog mode, see load_catalog_sources)
        if len(self.regions_HST_final) == n:
            self.regions_HST_final = np.array(self.regions_HST_final)[keep]
        if len(self.regions_MUSE_final) == n:
            self.regions_MUSE_final = np.array(self.regions_MUSE_final)[keep]

        # regions_HST / regions_MUSE are indexed by _row1 / _row2 in final_table
        # so we need to carry through the correct subset
        #idx_hst  = np.array(self.final_table_unfiltered['_row1'], dtype=int)[keep]
        #idx_muse = np.array(self.final_table_unfiltered['_row2'], dtype=int)[keep]

        #self.regions_HST_final  = [self.regions_HST[i]  for i in idx_hst]
        #self.regions_MUSE_final = [self.regions_MUSE[i] for i in idx_muse]

        # ── diagnostic plot ───────────────────────────────────────────────────
        if plot:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # ── left: per-image pass/fail bar chart ──────────────────────────
            labels   = list(per_image_counts.keys())
            n_pass_v = [per_image_counts[l][0] for l in labels]
            n_fail_v = [per_image_counts[l][1] for l in labels]
            y_pos    = np.arange(len(labels))

            axes[0].barh(y_pos, n_pass_v, color='steelblue',
                            label='Pass', height=0.6)
            axes[0].barh(y_pos, n_fail_v, left=n_pass_v, color='tomato',
                            label='Fail (outside FOV)', height=0.6)
            axes[0].set_yticks(y_pos)
            axes[0].set_yticklabels(labels, fontsize=9)
            axes[0].set_xlabel('Number of sources', fontsize=11)
            axes[0].set_title('Sources per image footprint', fontsize=12)
            axes[0].axvline(n_kept, color='black', linestyle='--',
                            linewidth=1.2, label=f'Final kept = {n_kept}')
            axes[0].legend(fontsize=9)
            axes[0].grid(True, alpha=0.3, axis='x')

            # ── right: source positions on MUSE WCS ──────────────────────────
            if self.muse_data is not None:
                axes[1].imshow(self.muse_data, origin='lower',
                                norm=get_norm(self.muse_data, 15, 99.5, 'asinh'),
                                cmap='viridis')

            # Project all sources onto MUSE pixel grid
            sky_all = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
            px_all, py_all = sky_all.to_pixel(self.muse_wcs)

            axes[1].scatter(px_all[~keep], py_all[~keep],
                            c='tomato', s=18, marker='x', linewidths=1.2,
                            label=f'Removed ({n_removed})', zorder=4)
            axes[1].scatter(px_all[keep], py_all[keep],
                            c='cyan', s=18, marker='o', linewidths=0,
                            label=f'Kept ({n_kept})', zorder=5)

            # Overlay each image footprint boundary on the MUSE frame
            for label, data, wcs in images_to_check:
                if label == 'MUSE':
                    continue
                valid = np.isfinite(data) & (data != 0)
                # Project corner pixels of the valid area to MUSE pixel grid
                # by sampling the valid footprint boundary every 20 pixels
                ny_i, nx_i = data.shape
                ys_edge, xs_edge = np.where(valid)
                step = max(1, len(ys_edge) // 500)
                sky_edge = SkyCoord.from_pixel(
                    xs_edge[::step].astype(float),
                    ys_edge[::step].astype(float),
                    wcs
                )
                px_edge, py_edge = sky_edge.to_pixel(self.muse_wcs)
                in_m = (np.isfinite(px_edge) & np.isfinite(py_edge))
                if in_m.sum() > 10:
                    # Use convex hull to draw a clean boundary
                    from scipy.spatial import ConvexHull
                    pts = np.column_stack([px_edge[in_m], py_edge[in_m]])
                    try:
                        hull = ConvexHull(pts)
                        hull_pts = np.append(hull.vertices, hull.vertices[0])
                        axes[1].plot(pts[hull_pts, 0], pts[hull_pts, 1],
                                        '--', linewidth=1.2,
                                        label=label, alpha=0.8)
                    except Exception:
                        pass

            axes[1].set_title(f'{self.gal} — source positions\n'
                                f'on MUSE footprint', fontsize=12)
            axes[1].legend(fontsize=8, loc='upper right')
            axes[1].tick_params(left=False, bottom=False,
                                labelleft=False, labelbottom=False)

            plt.suptitle(f'Common FOV filter — {self.gal}', fontsize=13, y=1.01)
            plt.tight_layout()
            plt.show()

        logger.info(f'\n  ✓ final_table filtered: {n_kept} sources')
        logger.info(f'    regions_HST_final  : {len(self.regions_HST_final)} entries')
        logger.info(f'    regions_MUSE_final : {len(self.regions_MUSE_final)} entries')
        logger.info(f'    Unfiltered table   : self.final_table_unfiltered')

        #return keep

    def _find_galaxy_eb_v(self):

        pos = SkyCoord.from_name(self.gal)
        sfd = SFDQuery()
        eb_v = sfd(pos)
        self.eb_v = eb_v

    # ─────────────────────────────────────────────────────────────────────────
    # 6. Measure photometry for all sources
    # ─────────────────────────────────────────────────────────────────────────

    def run_spectro_photometry(self, filter_wavelengths, type_ap='HST', verbose=True, n_workers=1,
                               force_recompute=False):

        """
        Measure aperture photometry and extract MUSE spectra for every source
        in ``self.final_table``.

        Dereddening only corrects for Milky Way foreground extinction: a single
        E(B-V) value for the whole galaxy (from ``self._find_galaxy_eb_v()``,
        queried once per ``Photometry`` instance) is applied uniformly to every
        source. This is intentional — any host-galaxy-intrinsic reddening of a
        cluster is meant to be recovered later from the SED fit itself, not
        pre-subtracted here.

        Parameters
        ----------
        filter_wavelengths : dict  band → (central_wl_AA, R_value)
        type_ap             : {'HST', 'MUSE'}  which region set defines the
                              aperture used for both the broadband photometry
                              and the MUSE spectral extraction
        verbose             : bool
        n_workers           : int   number of worker processes for the
                              per-source loop. 1 (default) runs sequentially
                              in-process, exactly as before. >1 uses a
                              ProcessPoolExecutor (fork start method) — safe
                              to raise once the per-source apertures are the
                              bbox-local ones (see phot_aux_func), since each
                              worker then touches only small local arrays
                              rather than full-image-sized ones.

        Uses (already loaded via load_isf_results/load_images)
        ---------------------------------------------------------
        self.log_wl    : 1-D array  log-wavelength axis for the MUSE cube
        self.spectra   : astropy Table  MUSE spectra keyed by spaxel id
        self.muse_id_maps : tuple  (original_id_map, binned_id_map)
        self.eb_v      : float  galaxy-wide foreground E(B-V), set by _find_galaxy_eb_v()

        Populates
        ---------
        self.photometry_table : astropy Table with one row per source containing
                                fluxes in all bands plus the summed MUSE spectrum.
        """
        if not self.hst_conv:
            raise RuntimeError('Images not convolved — run convolve_images() first.')

        cache_paths = (f'{self.base_dir}/{self.gal}/{self.gal}_photometry.fits',
                       f'{self.base_dir}/{self.gal}/{self.gal}_photometry_spectra.npz',
                       f'{self.base_dir}/{self.gal}/{self.gal}_annuli.reg')
        if not force_recompute and all(os.path.exists(p) for p in cache_paths):
            cached = Table.read(cache_paths[0])
            cached_src = fits.getheader(cache_paths[0], 0).get('PSFSRC', 'table')
            if cached_src != getattr(self, 'psf_source', 'table'):
                logger.info(f'  Cached photometry ({cache_paths[0]}) was made with {cached_src} PSFs, '
                            f'now {getattr(self, "psf_source", "table")} — recomputing.')
            elif len(cached) != len(self.final_table):
                # Stale cache (final_table's source list changed since this
                # was written — e.g. a different filter_by_common_fov
                # result) — a mismatched-length photometry_table would
                # crash run_native_photometry's later per-band column
                # assignment (same "Inconsistent data column lengths"
                # failure mode as its own cache-hit path), so recompute
                # instead of adopting it.
                logger.warning(
                    f'  Cached photometry ({cache_paths[0]}) has {len(cached)} rows but '
                    f'final_table has {len(self.final_table)} — stale cache, recomputing.'
                )
            else:
                self.photometry_table = cached
                self.photometry_spectra = np.load(cache_paths[1], allow_pickle=True)
                self.annulus_regions = Regions.read(cache_paths[2])
                logger.info(f'  Photometry table and spectra loaded from disk')
                return

        hst_conv_dict, jwst_conv_dict = self._build_conv_dicts()

        self._find_galaxy_eb_v()

        bands        = list(filter_wavelengths.keys())
        n_sources    = len(self.final_table)
        n_bands      = len(bands)

        # Result arrays
        annulus_regions = np.full(n_sources, None)   # annulus regions
        flux_bsub    = np.full((n_sources, n_bands), np.nan)   # background sub
        flux_raw     = np.full((n_sources, n_bands), np.nan)   # no bkg sub
        spectra_Jy   = np.full((n_sources, len(self.log_wl)), np.nan)
        spectra_orig = np.full((n_sources, len(self.log_wl)), np.nan)

        logger.info(f'\n  Running photometry on {n_sources} sources ...\n')

        def _get_data_wcs(band):
            band = band.lower()
            if band in hst_conv_dict:
                return hst_conv_dict[band][0], hst_conv_dict[band][2]
            return jwst_conv_dict[band][0], jwst_conv_dict[band][2]

        def _apply_source_result(idd, spec_result, spec_error, phot_result, phot_error):
            status_parts = []

            if spec_result is not None:
                sum_o, sum_Jy = spec_result
                spectra_Jy[idd, :] = sum_Jy
                spectra_orig[idd, :] = sum_o
            else:
                status_parts.append(f'spectrum FAILED ({spec_error})')

            if phot_result is not None:
                wl, fnu, fnu_wb, ann = phot_result
                flux_bsub[idd, :] = fnu
                flux_raw[idd, :]  = fnu_wb
                annulus_regions[idd] = ann
                status_parts.append('ok')
            else:
                status_parts.append(f'photometry FAILED ({phot_error})')

            if verbose:
                logger.info(f'  [{idd + 1:>4d}/{n_sources}] ' + ' '.join(status_parts))

        if n_workers <= 1:
            for idd in range(n_sources):
                if type_ap == 'HST':
                    region_i = self.regions_HST_final[idd]
                elif type_ap == 'MUSE':
                    region_i = self.regions_MUSE_final[idd]
                else:
                    raise RuntimeError('Not valid region')

                spec_result, spec_error = None, None
                try:
                    spec_result = spectra_from_aperture(
                        region_i, self.muse_data.shape, self.muse_wcs,
                        self.log_wl, self.spectra, self.muse_id_maps,
                    )
                except Exception as e:
                    spec_error = str(e)

                phot_result, phot_error = None, None
                try:
                    phot_result = measure_photometry(
                        region_i, filter_wavelengths,
                        hst_conv_dict, jwst_conv_dict, self.eb_v
                    )
                except Exception as e:
                    phot_error = str(e)

                _apply_source_result(idd, spec_result, spec_error, phot_result, phot_error)
        else:
            init_args = (
                hst_conv_dict, jwst_conv_dict, self.muse_data.shape, self.muse_wcs,
                self.log_wl, self.spectra, self.muse_id_maps, filter_wavelengths,
                self.eb_v, self.regions_HST_final, self.regions_MUSE_final, type_ap,
            )
            mp_ctx = mp.get_context('fork')
            with ProcessPoolExecutor(
                max_workers=n_workers, mp_context=mp_ctx,
                initializer=_init_photometry_worker, initargs=init_args,
            ) as executor:
                for idd, spec_result, spec_error, phot_result, phot_error in executor.map(
                    _photometry_worker, range(n_sources)
                ):
                    _apply_source_result(idd, spec_result, spec_error, phot_result, phot_error)

        #save annulus

        self.annulus_regions = annulus_regions

        reg_path = os.path.join(self.base_dir, self.gal, f'{self.gal}_annuli.reg')
        Regions(list(self.annulus_regions)).write(reg_path, format='ds9', overwrite=True)
        logger.info(f'  Annuli saved → {reg_path}')

        # ── assemble output table ─────────────────────────────────────────────
        rows = {f'flux_{b}'     : flux_bsub[:, i]
                for i, b in enumerate(bands)}
        rows.update({f'flux_{b}_raw': flux_raw[:, i]
                     for i, b in enumerate(bands)})

        result = Table(self.final_table)   # start from the matched catalog
        for col, arr in rows.items():
            result[col] = arr

        # Attach spectra as object columns (variable-length arrays)
        result['spectrum_Jy']   = spectra_Jy
        result['spectrum_orig'] = spectra_orig

        self.photometry_table = result
        logger.info(f'\n  ✓ Photometry complete: {n_sources} sources, {n_bands} bands')
        #return result

    # ─────────────────────────────────────────────────────────────────────────
    # 6b. Native-resolution aperture-corrected photometry (PHANGS Appendix A,
    #     arXiv:2511.20775 — see native_phot_aux_func.py module docstring for
    #     the exact method and the simplifications made relative to the paper)
    # ─────────────────────────────────────────────────────────────────────────

    def run_native_photometry(self, filter_wavelengths,
                               reference_groups=REFERENCE_BAND_GROUPS,
                               target_ee=0.65,
                               recenter_flag_frac=0.5,
                               annulus_r_in_factor=2.6, annulus_r_out_factor=4.0,
                               n_models=20, n_slits=12,
                               slit_width_px=2, slit_length_arcsec=1.0,
                               snr_threshold=3.0,
                               peak_snr_threshold=5.0, aperture_snr_threshold=3.0,
                               max_r65_growth_factor=2.0,
                               n_workers=1, verbose=True,
                               force_recompute=False,
                               plot_diagnostics=True,
                               dic_passbands=None,
                               instrument_fit=True,
                               profile_by_class=CLUSTER_PROFILE_BY_CLASS,
                               instrument_slit_width_px=1,
                               instrument_slit_length_arcsec=1.0,
                               max_r_eff_pc=20.0,
                               match_galaxy_psf=None, psf_blur='none'):
        """
        Native-resolution, per-cluster-R65 aperture-corrected photometry —
        ADDITIONAL to (never replacing) the existing convolved-image method
        in run_spectro_photometry(), so both can be compared side by side.
        MUSE is untouched by this method (HST/JWST broadband only).

        Per source, per band: test for real emission (detect_emission); if
        found, measure its own center (2-D Gaussian fit) and size (averaged-
        12-slit profile fit) and derive this cluster's own R65 from that
        size (aperture_source='own'). A band with no detected emission
        borrows a similar-PSF band's own aperture (position + R65, reused
        unscaled) if this source has one (aperture_source='borrowed:{band}'
        — see reference_groups), else falls back to its own PSF-based R65
        at the catalog position (aperture_source='psf_fallback'). See
        native_phot_aux_func.measure_native_photometry_one_source's
        docstring for the full per-source algorithm. Sources whose
        background-subtracted flux S/N falls below snr_threshold are masked
        as non-detections (flux_native_{band}=NaN, status_native_{band}=
        'no_detection') rather than reporting a noisy near-zero number.

        Preconditions
        -------------
        self.photometry_table already populated (run run_spectro_photometry()
        first — this method only ADDS columns to it) and self.psf_hst_alpha /
        self.psf_jwst_alpha populated (run measure_native_psf_shape() first).

        Parameters
        ----------
        filter_wavelengths : dict  band → (central_wl_AA, R_value) — same
                             dict shape as run_spectro_photometry
        reference_groups     : dict — aperture-borrowing band groups (see
                             native_phot_aux_func.REFERENCE_BAND_GROUPS for
                             the default: all HST bands mutually similar
                             with F555W preferred, F200W standalone, the
                             three ~3um NIRCam medium bands mutually
                             similar, every MIRI band its own singleton
                             group)
        target_ee             : float — the aperture-defining encircled-
                             energy fraction (default 0.65, i.e. "R65");
                             corr_factor is exactly 1/target_ee by
                             construction whenever a size-estimate-derived
                             or borrowed R65 aperture is used
        recenter_flag_frac : float — flag (not block) sources whose 2-D
                             Gaussian centering shift exceeds this fraction
                             of the band's PSF-based R65, or that failed to
                             converge at all
        annulus_r_in_factor, annulus_r_out_factor : float — background
                             annulus radii as multiples of the aperture
                             radius actually used (own/borrowed/PSF-fallback)
        n_models              : int — number of Moffat-convolution models for
                             each band-level calibration curve (see
                             build_native_calibration_curve)
        n_slits, slit_width_px : slit-profile geometry, shared between all
                             calibration curves and every real-source
                             measurement
        slit_length_arcsec    : float — HALF-length of each of the n_slits
                             diameters (default 1.0", i.e. a 2" total
                             diameter) — the fixed sampling/fit domain for
                             the averaged-slit size measurement; widened
                             automatically per source if the derived R65
                             implies a bigger source (see
                             native_phot_aux_func._measure_own_band)
        snr_threshold          : float — background-subtracted flux / its
                             propagated error below this is masked as a
                             non-detection (flux_native_{b}=NaN,
                             status_native_{b}='no_detection')
        peak_snr_threshold,
        aperture_snr_threshold : float — the two detect_emission criteria
                             (peak-pixel and aperture S/N) a band must both
                             pass to be treated as having real emission for
                             this source
        max_r65_growth_factor : float — caps an 'own'-band r65_arcsec to at
                             most this multiple of the band's own
                             r65_psf_arcsec, so a single corrupted fit can't
                             blow one band's aperture up far beyond what the
                             other bands agree on (see
                             native_phot_aux_func._measure_own_band)
        n_workers            : int — 1 (default) runs sequentially in-process;
                             >1 uses a ProcessPoolExecutor (fork start
                             method), one task PER SOURCE (all of that
                             source's bands together — see
                             _native_photometry_worker), same fork/
                             initializer pattern as run_spectro_photometry/
                             second_filter
        verbose              : bool
        force_recompute      : bool — ignore any cached
                             `<gal>_photometry_native.fits` and recompute
        plot_diagnostics      : bool — if True (default), also call
                             plot_native_photometry_diagnostics() and
                             plot_native_size_calibration() right after this
                             method completes (cache hit or fresh compute),
                             with matching annulus factors
        dic_passbands         : dict or None — band → (wl_min_AA, wl_max_AA);
                             forwarded to plot_native_photometry_diagnostics()
                             for the SED panel's filter-width spans, same
                             dict plot_sed takes. None (default) skips them.
        instrument_fit        : bool — if True (default), also run
                             run_instrument_photometry() on top of these
                             results (PSF (x) Gaussian and PSF (x) cluster-
                             profile fits, columns *_instG_* / *_instP_*),
                             before the diagnostic plots so they show all
                             three methods. Its own cache, same
                             force_recompute.
        profile_by_class, instrument_slit_width_px,
        instrument_slit_length_arcsec, max_r_eff_pc,
        match_galaxy_psf, psf_blur : forwarded to run_instrument_photometry() (as
                             profile_by_class, slit_width_px,
                             slit_length_arcsec, max_r_eff_pc, match_galaxy_psf,
                             psf_blur)

        Populates
        ---------
        self.photometry_table : new columns added per band `b` (additive —
                                existing columns from run_spectro_photometry
                                are never touched):
                                flux_native_{b} (+_raw, +_err),
                                corr_factor_native_{b},
                                emission_detected_native_{b},
                                peak_snr_native_{b}, aperture_snr_native_{b},
                                sigma_native_{b} — the measured size (NaN
                                    unless aperture_source_native_{b}=='own'),
                                r65_native_{b} — the aperture radius ACTUALLY
                                    USED (own/borrowed/PSF-fallback),
                                r65_psf_native_{b} — this band's fixed
                                    PSF-based radius, for comparison,
                                aperture_source_native_{b} — 'own',
                                    'borrowed:{band}', or 'psf_fallback',
                                extra_conv_fwhm_est_native_{b},
                                centroid_ra_native_{b}, centroid_dec_native_{b},
                                recenter_shift_native_{b}, recenter_flag_native_{b},
                                bkg_median_native_{b}, bkg_std_native_{b},
                                snr_native_{b},
                                size_extrapolated_native_{b},
                                size_capped_native_{b},
                                status_native_{b}
        self.native_band_calib : dict  band (lowercase) → per-band setup dict
                                (calib_curve, r65_psf_arcsec,
                                pixel_scale_arcsec, instrument, ...) —
                                reused by plot_native_size_calibration() and
                                plot_native_slit_profiles() so they never
                                need to rebuild the band-level calibration
                                curves themselves
        """
        if self.photometry_table is None:
            raise RuntimeError(
                'self.photometry_table is empty — run run_spectro_photometry() first; '
                'run_native_photometry() only adds columns to an existing table.'
            )
        if not self.psf_hst_alpha and not self.psf_jwst_alpha:
            raise RuntimeError(
                'PSF shape (alpha/gamma) not measured — run measure_native_psf_shape() first.'
            )

        outd = os.path.join(self.base_dir, self.gal)
        cache_path = os.path.join(outd, f'{self.gal}_photometry_native.fits')

        # ── per-band setup: one calibration curve per band, built once ─────
        # Done unconditionally (even on a cache hit below) since it's cheap
        # relative to the per-source loop and plot_native_size_calibration()
        # / plot_native_slit_profiles() need these curves whether or not the
        # per-source measurement itself was cached.
        band_data = {}
        band_original_case = {}
        for band, (cwl, r_value) in filter_wavelengths.items():
            band_lower = band.lower()
            if band_lower in self.hst_images:
                image, header, wcs = self.hst_images[band_lower]
                instrument = 'HST'
                fwhm = self.psf_hst.get(band_lower)
                alpha = self.psf_hst_alpha.get(band_lower)
            elif band_lower in self.jwst_images:
                image, header, wcs = self.jwst_images[band_lower]
                instrument = 'JWST'
                fwhm = self.psf_jwst.get(band_lower)
                alpha = self.psf_jwst_alpha.get(band_lower)
            else:
                logger.warning(f'{band}: no native image loaded — skipping native photometry')
                continue

            if not fwhm or not alpha:
                logger.warning(f'{band}: PSF FWHM/alpha not available — skipping native photometry')
                continue

            pixel_scale_arcsec = wcs.proj_plane_pixel_scales()[0].to('arcsec').value
            r65_psf = compute_r65(fwhm, alpha, pixel_scale_arcsec, target_ee=target_ee)
            calib_curve = build_native_calibration_curve(
                fwhm, alpha, pixel_scale_arcsec, r65_psf, n_models=n_models,
                n_slits=n_slits, slit_width_px=slit_width_px,
                slit_length_arcsec=slit_length_arcsec,
            )
            logger.info(f'  {instrument} {band}: R65_psf={r65_psf:.4f}" '
                        f'(fwhm={fwhm:.3f}", alpha={alpha:.2f})')

            band_original_case[band_lower] = band
            band_data[band_lower] = dict(
                image=image, header=header, wcs=wcs, instrument=instrument,
                calib_curve=calib_curve, r65_psf_arcsec=r65_psf,
                pixel_scale_arcsec=pixel_scale_arcsec, r_value=r_value,
                central_wl_AA=cwl,
                unit_convert=_jwst_unit_convert if instrument == 'JWST' else None,
            )

        if not band_data:
            raise RuntimeError('No bands with both a native image and a measured PSF alpha — nothing to do.')

        self.native_band_calib = band_data

        # ── save the R65 aperture actually used, per source per band ───────
        # A CircleSkyRegion at the aperture's own centroid (own/borrowed) or
        # the catalog position (psf_fallback, which has no independent
        # center) and its own r65_native_{band} radius (falling back to
        # r65_psf_native_{band} for the rare 'no_flux' case, where no real
        # aperture flux exists but an aperture was still attempted) — one
        # DS9 .reg file per band, mirroring the existing regions_HST_final/
        # annulus_regions convention. Runs after either the cache-hit or the
        # fresh-compute path below, since both leave self.photometry_table
        # with the same r65_native_{band}/centroid_*_native_{band} columns.
        def _save_native_aperture_regions():
            ra_cat = np.asarray(self.final_table['PHANGS_RA'], dtype=float)
            dec_cat = np.asarray(self.final_table['PHANGS_DEC'], dtype=float)
            os.makedirs(outd, exist_ok=True)
            regions_native = {}
            for band_lower, band in band_original_case.items():
                r65_col, r65_psf_col = f'r65_native_{band}', f'r65_psf_native_{band}'
                cra_col, cdec_col = f'centroid_ra_native_{band}', f'centroid_dec_native_{band}'
                if r65_col not in self.photometry_table.colnames:
                    continue
                r65 = np.asarray(self.photometry_table[r65_col], dtype=float)
                r65_psf = np.asarray(self.photometry_table[r65_psf_col], dtype=float)
                cra = np.asarray(self.photometry_table[cra_col], dtype=float)
                cdec = np.asarray(self.photometry_table[cdec_col], dtype=float)
                regs = []
                for i in range(len(self.photometry_table)):
                    radius = r65[i] if np.isfinite(r65[i]) else r65_psf[i]
                    ra_i = cra[i] if np.isfinite(cra[i]) else ra_cat[i]
                    dec_i = cdec[i] if np.isfinite(cdec[i]) else dec_cat[i]
                    regs.append(CircleSkyRegion(
                        center=SkyCoord(ra=ra_i * u.deg, dec=dec_i * u.deg),
                        radius=float(radius) * u.arcsec,
                    ))
                regions_native[band] = Regions(regs)
                reg_path = os.path.join(outd, f'{self.gal}_native_apertures_{band}.reg')
                regions_native[band].write(reg_path, format='ds9', overwrite=True)
            self.native_aperture_regions = regions_native
            if regions_native:
                logger.info(f'  ✓ Native aperture regions saved → {outd}/{self.gal}_native_apertures_<band>.reg')

        def _run_instrument():
            self.run_instrument_photometry(
                filter_wavelengths, profile_by_class=profile_by_class, target_ee=target_ee,
                n_slits=n_slits, slit_width_px=instrument_slit_width_px,
                slit_length_arcsec=instrument_slit_length_arcsec, max_r_eff_pc=max_r_eff_pc,
                annulus_r_in_factor=annulus_r_in_factor, annulus_r_out_factor=annulus_r_out_factor,
                snr_threshold=snr_threshold, reference_groups=reference_groups,
                n_workers=n_workers, force_recompute=force_recompute,
                plot_diagnostics=plot_diagnostics, verbose=verbose,
                match_galaxy_psf=match_galaxy_psf, psf_blur=psf_blur,
            )

        if not force_recompute and os.path.exists(cache_path) \
                and Table.read(cache_path).meta.get('PSFSRC', 'table') != getattr(self, 'psf_source', 'table'):
            logger.info(f'  Cached native photometry ({cache_path}) was made with '
                        f'{Table.read(cache_path).meta.get("PSFSRC", "table")} PSFs, now {getattr(self, "psf_source", "table")} '
                        f'— recomputing.')
        elif not force_recompute and os.path.exists(cache_path):
            cached = Table.read(cache_path)
            if len(cached) != len(self.photometry_table):
                # Stale cache (source list changed since it was written —
                # e.g. a different filter_by_common_fov result, or a
                # re-run of IsolatedSourceFinder) — blindly assigning
                # mismatched-length columns raises astropy's own
                # "Inconsistent data column lengths" ValueError, so fall
                # through to a fresh computation instead of crashing.
                logger.warning(
                    f'  Cached native photometry ({cache_path}) has {len(cached)} rows '
                    f'but the current photometry_table has {len(self.photometry_table)} '
                    f'— stale cache, recomputing.'
                )
            else:
                for col in cached.colnames:
                    if col not in self.photometry_table.colnames:
                        self.photometry_table[col] = cached[col]
                logger.info(f'  Native photometry loaded from disk → {cache_path}')
                _save_native_aperture_regions()
                if instrument_fit:
                    _run_instrument()
                if plot_diagnostics:
                    self.plot_native_photometry_diagnostics(
                        filter_wavelengths, dic_passbands=dic_passbands,
                        annulus_r_in_factor=annulus_r_in_factor,
                        annulus_r_out_factor=annulus_r_out_factor,
                        verbose=verbose,
                    )
                    self.plot_native_size_calibration(filter_wavelengths, verbose=verbose)
                return

        n_sources = len(self.final_table)
        radec = list(zip(
            np.asarray(self.final_table['PHANGS_RA'], dtype=float),
            np.asarray(self.final_table['PHANGS_DEC'], dtype=float),
        ))

        result_fields = [
            'flux_mJy', 'flux_raw_mJy', 'flux_err_mJy', 'corr_factor',
            'peak_snr', 'aperture_snr', 'sigma_arcsec', 'r65_arcsec', 'r65_psf_arcsec',
            'extra_conv_fwhm_estimate_arcsec',
            'recenter_shift_arcsec', 'bkg_median_per_pix', 'bkg_std', 'snr',
            'centroid_ra', 'centroid_dec',
        ]

        # ── result storage ──────────────────────────────────────────────────
        results = {
            band_lower: {f: np.full(n_sources, np.nan) for f in result_fields}
            for band_lower in band_data
        }
        for band_lower in band_data:
            results[band_lower]['emission_detected'] = np.zeros(n_sources, dtype=bool)
            results[band_lower]['recenter_flagged'] = np.zeros(n_sources, dtype=bool)
            results[band_lower]['size_extrapolated'] = np.zeros(n_sources, dtype=bool)
            results[band_lower]['size_capped'] = np.zeros(n_sources, dtype=bool)
            results[band_lower]['aperture_source'] = np.full(n_sources, '', dtype='<U32')
            results[band_lower]['status'] = np.full(n_sources, '', dtype='<U32')

        def _store(idd, band_lower, result):
            r = results[band_lower]
            for f in result_fields:
                r[f][idd] = getattr(result, f)
            r['emission_detected'][idd] = result.emission_detected
            r['recenter_flagged'][idd] = result.recenter_flagged
            r['size_extrapolated'][idd] = result.size_extrapolated
            r['size_capped'][idd] = result.size_capped
            r['aperture_source'][idd] = result.aperture_source[:32]
            r['status'][idd] = result.status[:32]

        # Single source of truth for every measure_native_photometry_one_source
        # kwarg beyond band_data/reference_groups/eb_v, shared by both the
        # sequential loop and the multiprocessing worker below — previously
        # the worker only received reference_groups/eb_v and silently fell
        # back to measure_native_photometry_one_source's own defaults for
        # everything else, so n_workers>1 would have quietly diverged from
        # n_workers=1 the moment any of these was called with a non-default
        # value.
        measure_kwargs = dict(
            target_ee=target_ee,
            recenter_flag_frac=recenter_flag_frac,
            annulus_r_in_factor=annulus_r_in_factor,
            annulus_r_out_factor=annulus_r_out_factor,
            n_slits=n_slits, slit_width_px=slit_width_px,
            slit_length_arcsec=slit_length_arcsec,
            snr_threshold=snr_threshold,
            peak_snr_threshold=peak_snr_threshold,
            aperture_snr_threshold=aperture_snr_threshold,
            max_r65_growth_factor=max_r65_growth_factor,
        )

        # ── dispatch: one task PER SOURCE (all its bands together — a band
        # without emission can borrow another band's aperture for the SAME
        # source, so bands are no longer independent tasks) ────────────────
        logger.info(f'\n  Running native photometry on {n_sources} sources, '
                    f'{len(band_data)} bands ...\n')

        if n_workers <= 1:
            for idd, (ra, dec) in enumerate(radec):
                source_results = measure_native_photometry_one_source(
                    ra, dec, band_data, reference_groups=reference_groups,
                    eb_v=self.eb_v, **measure_kwargs,
                )
                for band_lower, result in source_results.items():
                    _store(idd, band_lower, result)
                if verbose:
                    logger.info(f'  [{idd + 1:>4d}/{n_sources}] done')
        else:
            mp_ctx = mp.get_context('fork')
            with ProcessPoolExecutor(
                max_workers=n_workers, mp_context=mp_ctx,
                initializer=_init_native_photometry_worker,
                initargs=(band_data, radec, self.eb_v, reference_groups, measure_kwargs),
            ) as executor:
                for idd, source_results in executor.map(_native_photometry_worker, range(n_sources)):
                    for band_lower, result in source_results.items():
                        _store(idd, band_lower, result)
                    if verbose:
                        logger.info(f'  [{idd + 1:>4d}/{n_sources}] done')

        # ── assemble into self.photometry_table ─────────────────────────────
        colname_map = {
            'flux_mJy': 'flux_native_{}',
            'flux_raw_mJy': 'flux_native_{}_raw',
            'flux_err_mJy': 'flux_native_{}_err',
            'corr_factor': 'corr_factor_native_{}',
            'emission_detected': 'emission_detected_native_{}',
            'peak_snr': 'peak_snr_native_{}',
            'aperture_snr': 'aperture_snr_native_{}',
            'sigma_arcsec': 'sigma_native_{}',
            'r65_arcsec': 'r65_native_{}',
            'r65_psf_arcsec': 'r65_psf_native_{}',
            'aperture_source': 'aperture_source_native_{}',
            'extra_conv_fwhm_estimate_arcsec': 'extra_conv_fwhm_est_native_{}',
            'centroid_ra': 'centroid_ra_native_{}',
            'centroid_dec': 'centroid_dec_native_{}',
            'recenter_shift_arcsec': 'recenter_shift_native_{}',
            'recenter_flagged': 'recenter_flag_native_{}',
            'bkg_median_per_pix': 'bkg_median_native_{}',
            'bkg_std': 'bkg_std_native_{}',
            'snr': 'snr_native_{}',
            'size_extrapolated': 'size_extrapolated_native_{}',
            'size_capped': 'size_capped_native_{}',
            'status': 'status_native_{}',
        }
        for band_lower, r in results.items():
            band = band_original_case[band_lower]
            for field, arr in r.items():
                self.photometry_table[colname_map[field].format(band)] = arr

        os.makedirs(outd, exist_ok=True)
        new_cols = [colname_map[f].format(band_original_case[bl])
                    for bl in results for f in results[bl]]
        native_out = Table(self.photometry_table[new_cols])
        native_out.meta['PSFSRC'] = getattr(self, 'psf_source', 'table')
        native_out.write(cache_path, overwrite=True)
        logger.info(f'  ✓ Native photometry saved → {cache_path}')
        _save_native_aperture_regions()

        # ── per-band summary ─────────────────────────────────────────────────
        logger.info('\n  Native photometry summary:')
        for band_lower, r in results.items():
            band = band_original_case[band_lower]
            n_own = int(np.sum(r['aperture_source'] == 'own'))
            n_borrowed = int(np.sum(np.char.startswith(r['aperture_source'].astype(str), 'borrowed')))
            n_fallback = int(np.sum(r['aperture_source'] == 'psf_fallback'))
            n_detected = int(np.sum(r['status'] == 'ok') + np.sum(r['status'] == 'size_extrapolated'))
            logger.info(
                f'    {band:8s}: own={n_own:4d}  borrowed={n_borrowed:4d}  '
                f'psf_fallback={n_fallback:4d}  detected(ok/extrapolated)={n_detected:4d}/{n_sources}'
            )

        logger.info(f'\n  ✓ Native photometry complete: {n_sources} sources, {len(band_data)} bands')

        if instrument_fit:
            _run_instrument()
        if plot_diagnostics:
            self.plot_native_photometry_diagnostics(
                filter_wavelengths, dic_passbands=dic_passbands,
                annulus_r_in_factor=annulus_r_in_factor,
                annulus_r_out_factor=annulus_r_out_factor,
                verbose=verbose,
            )
            self.plot_native_size_calibration(filter_wavelengths, verbose=verbose)

    # ─────────────────────────────────────────────────────────────────────────
    # 6c. Instrument-PSF photometry (PSF (x) Gaussian / PSF (x) cluster profile)
    # ─────────────────────────────────────────────────────────────────────────

    # name -> (InstrumentPhotResult field, column template)
    _INSTRUMENT_COLUMNS = {
        'flux_mJy': 'flux_{m}_{b}',
        'flux_err_mJy': 'flux_{m}_{b}_err',
        'flux_raw_mJy': 'flux_{m}_{b}_raw',
        'flux_model_mJy': 'flux_{m}_{b}_model',
        'corr_factor': 'corr_factor_{m}_{b}',
        'r65_arcsec': 'r65_{m}_{b}',
        'r_eff_arcsec': 'reff_{m}_{b}',
        'r_eff_pc': 'reff_pc_{m}_{b}',
        'background_fit': 'bkg_fit_{m}_{b}',
        'rms_residual': 'rms_{m}_{b}',
        'converged': 'converged_{m}_{b}',
        'size_source': 'size_source_{m}_{b}',
        'centroid_ra': 'centroid_ra_{m}_{b}',
        'centroid_dec': 'centroid_dec_{m}_{b}',
        'bkg_median_per_pix': 'bkg_median_{m}_{b}',
        'bkg_std': 'bkg_std_{m}_{b}',
        'snr': 'snr_{m}_{b}',
        'status': 'status_{m}_{b}',
        'gamma': 'gamma_{m}_{b}',          # instP only (EFF power index of the model)
    }

    def run_instrument_photometry(self, filter_wavelengths,
                                   profile_by_class=CLUSTER_PROFILE_BY_CLASS,
                                   target_ee=0.65, n_slits=12, slit_width_px=1,
                                   slit_length_arcsec=1.0, max_r_eff_pc=20.0,
                                   annulus_r_in_factor=2.6, annulus_r_out_factor=4.0,
                                   snr_threshold=3.0,
                                   reference_groups=REFERENCE_BAND_GROUPS,
                                   n_workers=1, force_recompute=False,
                                   plot_diagnostics=True, verbose=True,
                                   match_galaxy_psf=None, psf_blur='none'):
        """
        Instrument-PSF photometry, ADDITIONAL to run_native_photometry (whose
        results it builds on — run that first; it calls this itself when
        instrument_fit=True). For every source and band two models of the
        averaged slit profile are fitted,

            instG : flux * [Gaussian(R_eff)       (x) instrument PSF] + bkg
            instP : flux * [cluster profile(R_eff) (x) instrument PSF] + bkg

        with the real HST (WFC3/ACS ePSF + drizzle blur) or JWST (stpsf)
        PSF of the band (instrument_psf.instrument_psf_for_image). The
        cluster profile comes from PHANGS_CC_CLASS_ML via profile_by_class
        (default: YCL and MAP -> EFF gamma=3, OGCC -> King c=1.5). Each
        model's R65 is the radius enclosing target_ee of its TOTAL light on
        the image's pixel grid, and the flux is the annulus-background-
        subtracted aperture sum at that R65 times the model's own aperture
        correction (which also restores halo light the annulus removes).
        See native_phot_aux_func.measure_instrument_photometry_one_source
        for the per-source logic (own fits where the native method found
        emission, otherwise the source size borrowed from a sibling band,
        otherwise a point source).

        Cached in output/<gal>/<gal>_photometry_instrument.fits, separate
        from the native cache; the header records the settings (profiles,
        slit geometry, target_ee, max_r_eff_pc) and a different call
        recomputes rather than reusing it.

        Parameters
        ----------
        filter_wavelengths : dict band -> (central_wl_AA, R_value)
        profile_by_class   : dict class -> dict(profile='eff'|'king'|'gaussian',
                             gamma=..., king_c=...); classes not listed use
                             EFF gamma=3
        slit_width_px      : slit width of the instrument fit (default 1 px —
                             a direct cut through the centre); independent
                             of the native method's slit_width_px
        slit_length_arcsec : minimum slit half-length; each band uses
                             max(this, 4 x its PSF FWHM), lengthened to
                             5 R_eff (<= 3") for big sources
        max_r_eff_pc       : upper bound of the R_eff search, converted with
                             PHANGS_DMPC
        annulus_r_in_factor, annulus_r_out_factor, snr_threshold,
        reference_groups   : as in run_native_photometry
        n_workers          : 1 = sequential; >1 = fork pool, one task per source
        psf_blur           : extra Gaussian blur on top of the theoretical PSF
                             (STScI effective PSF / stpsf):
                             'none' (default): no blur, the theoretical PSF.
                             'galaxy': tuned so the model star has THIS
                             galaxy's measured FWHM (self.psf_hst / psf_jwst,
                             psf_table.csv; not MIRI). Caution: those FWHMs
                             come from the brightest compact sources of the
                             image, often clusters, so they over-broaden the
                             PSF and shrink the fitted R_eff.
                             'default': the sample-median blur
                             (HST_DRIZZLE_FWHM_ARCSEC / JWST_EXTRA_FWHM_ARCSEC).
                             a number: that blur FWHM in arcsec, every band.
        match_galaxy_psf   : legacy switch; True = psf_blur 'galaxy', False =
                             'default'. Overrides psf_blur when given.
        plot_diagnostics   : save the per-galaxy instrument_comparison.png

        Populates
        ---------
        self.photometry_table : per band b and method m in (instG, instP):
            flux_{m}_{b} (+_err, _raw, _model), corr_factor_{m}_{b},
            r65_{m}_{b}, reff_{m}_{b}, reff_pc_{m}_{b}, bkg_fit_{m}_{b},
            rms_{m}_{b}, converged_{m}_{b}, size_source_{m}_{b},
            centroid_ra/dec_{m}_{b}, bkg_median/bkg_std/snr_{m}_{b},
            status_{m}_{b}; gamma_instP_{b} (EFF power index of the instP
            model: fitted when the spec has gamma_free, else the fixed value;
            NaN for King / R_eff = 0); per source: profile_instP
        self.instrument_psf      : band_lower -> InstrumentPSF
        self.instrument_settings : the settings used (also in the cache header)
        """
        band_calib = getattr(self, 'native_band_calib', None)
        if self.photometry_table is None or not band_calib:
            raise RuntimeError('run_native_photometry() first — the instrument fit builds on its results.')
        if getattr(self, 'eb_v', None) is None:
            self._find_galaxy_eb_v()

        band_original_case = {}
        for band in filter_wavelengths:
            bl = band.lower()
            if bl in band_calib and f'aperture_source_native_{band}' in self.photometry_table.colnames:
                band_original_case[bl] = band

        # ── per-band PSF models (rebuilt each call: cheap, and they depend
        # on psf_blur) ─────────────────────────────────────────────────────
        if match_galaxy_psf is not None:
            psf_blur = 'galaxy' if match_galaxy_psf else 'default'
        mode = str(psf_blur).strip().lower()
        if mode not in ('none', 'galaxy', 'default'):
            try:
                fixed_blur = float(mode)
            except ValueError:
                raise ValueError(f"psf_blur must be 'none', 'galaxy', 'default' or a number, got {psf_blur!r}")
            mode = f'{fixed_blur:.4f}'
        self.instrument_psf = {}
        band_data = {}
        psf_blur_used = {}
        for bl, band in band_original_case.items():
            bd = band_calib[bl]
            target, extra = None, None
            if mode == 'galaxy':
                target = (self.psf_hst.get(bl) if bd.get('instrument') == 'HST'
                          else self.psf_jwst.get(bl))
            elif mode == 'none':
                extra = 0.0
            elif mode != 'default':
                extra = float(mode)
            model, note = instrument_psf_for_image(band, bd['header'], bd['pixel_scale_arcsec'],
                                                   target_fwhm_arcsec=target, extra_fwhm_arcsec=extra)
            if note:
                logger.warning(f'  {band}: {note}')
            if model is None:
                continue
            self.instrument_psf[bl] = model
            psf_blur_used[band] = round(float(model.extra_fwhm_arcsec), 4)
            matched = (f', model FWHM {model.matched_fwhm_arcsec:.3f}" vs measured {target:.3f}"'
                       if np.isfinite(getattr(model, 'matched_fwhm_arcsec', np.nan)) else '')
            logger.info(f'  {band}: instrument PSF {model.name} (extra blur '
                        f'{model.extra_fwhm_arcsec:.3f}"{matched}; stamp holds '
                        f'{model.total_flux:.3f} of the total light)')
            band_data[bl] = dict(bd, psf_model=model)
        if not band_data:
            raise RuntimeError('No band has an instrument PSF model — nothing to do.')

        settings = dict(TARGEE=float(target_ee), NSLITS=int(n_slits), SLITW=int(slit_width_px),
                        SLITLEN=float(slit_length_arcsec), MAXREFF=float(max_r_eff_pc),
                        ANNIN=float(annulus_r_in_factor), ANNOUT=float(annulus_r_out_factor),
                        PSFMATCH={'galaxy': True, 'default': False}.get(mode, mode),
                        PSFNORM=2, SLITCOMB=str(SLIT_COMBINE),
                        PSFSRC=str(getattr(self, 'psf_source', 'table')))
        # PSFNORM: version of the PSF normalization (2 = HST far wings included);
        # SLITCOMB: how the slits are combined (native_phot_aux_func.SLIT_COMBINE)
        # PSFSRC: where the native method's PSFs came from (its results feed this one)
        for band, blur in psf_blur_used.items():
            settings[f'XB{band}'[:8]] = blur
        # 'default' (unclassified sources) is not recorded on its own: from the
        # config it always equals YCL, and a new header key would invalidate
        # every existing cache
        for cls in sorted(set(list(profile_by_class) + ['YCL', 'MAP', 'OGCC']) - {'default'}):
            settings[f'P_{cls}'[:8]] = profile_tag(profile_spec_for_class(cls, profile_by_class))
        self.instrument_settings = settings

        outd = os.path.join(self.base_dir, self.gal)
        cache_path = os.path.join(outd, f'{self.gal}_photometry_instrument.fits')
        n_sources = len(self.photometry_table)

        # ── cache ──────────────────────────────────────────────────────────
        if not force_recompute and os.path.exists(cache_path):
            cached = Table.read(cache_path)
            cached_settings = {k: cached.meta.get(k) for k in settings}
            if len(cached) != n_sources:
                logger.warning(f'  Cached instrument photometry ({cache_path}) has {len(cached)} rows, '
                               f'photometry_table has {n_sources} — stale cache, recomputing.')
            elif cached_settings != settings:
                diff = {k: (cached_settings[k], v) for k, v in settings.items()
                        if cached_settings[k] != v}
                logger.info(f'  Cached instrument photometry was made with other settings '
                            f'(cached, now): {diff} — recomputing.')
            else:
                for col in cached.colnames:
                    self.photometry_table[col] = cached[col]
                self._fill_missing_instp_gamma(band_original_case)
                logger.info(f'  Instrument photometry loaded from disk → {cache_path}')
                self._save_instrument_aperture_regions(band_original_case)
                if plot_diagnostics:
                    self.plot_instrument_flux_comparison(filter_wavelengths)
                return

        # ── per-source inputs ──────────────────────────────────────────────
        tbl = self.photometry_table
        has_class = 'PHANGS_CC_CLASS_ML' in tbl.colnames
        has_dist = 'PHANGS_DMPC' in tbl.colnames
        per_source = []
        for i in range(n_sources):
            row = tbl[i]
            native_row = {
                bl: dict(aperture_source=str(row[f'aperture_source_native_{b}']).strip(),
                         centroid_ra=float(row[f'centroid_ra_native_{b}']),
                         centroid_dec=float(row[f'centroid_dec_native_{b}']))
                for bl, b in band_original_case.items() if bl in band_data
            }
            cc_class = row['PHANGS_CC_CLASS_ML'] if has_class else ''
            per_source.append(dict(
                ra=float(row['PHANGS_RA']), dec=float(row['PHANGS_DEC']),
                native_row=native_row,
                profile_spec=profile_spec_for_class(cc_class, profile_by_class),
                distance_mpc=float(row['PHANGS_DMPC']) if has_dist else float('nan'),
                eb_v=self.eb_v,
            ))

        measure_kwargs = dict(
            reference_groups=reference_groups, target_ee=target_ee,
            annulus_r_in_factor=annulus_r_in_factor, annulus_r_out_factor=annulus_r_out_factor,
            n_slits=n_slits, slit_width_px=slit_width_px, slit_length_arcsec=slit_length_arcsec,
            max_r_eff_pc=max_r_eff_pc, snr_threshold=snr_threshold,
        )

        fields = list(self._INSTRUMENT_COLUMNS)
        str_fields = ('size_source', 'status')
        store = {
            (bl, m): {f: (np.full(n_sources, '', dtype='<U32') if f in str_fields
                          else np.zeros(n_sources, dtype=bool) if f == 'converged'
                          else np.full(n_sources, np.nan))
                      for f in fields}
            for bl in band_data for m in INSTRUMENT_METHODS
        }

        def _store(idd, source_results):
            for bl, per_method in source_results.items():
                for m, res in per_method.items():
                    for f in fields:
                        v = getattr(res, f)
                        store[(bl, m)][f][idd] = v[:32] if f in str_fields else v

        logger.info(f'\n  Running instrument-PSF photometry on {n_sources} sources, '
                    f'{len(band_data)} bands (slit width {slit_width_px} px) ...\n')
        t0 = time.time()
        if n_workers <= 1:
            for idd, src in enumerate(per_source):
                _store(idd, measure_instrument_photometry_one_source(
                    src['ra'], src['dec'], src['native_row'], band_data, src['profile_spec'],
                    distance_mpc=src['distance_mpc'], eb_v=src['eb_v'], **measure_kwargs))
                if verbose:
                    logger.info(f'  [{idd + 1:>4d}/{n_sources}] instrument fit done '
                                f'({profile_tag(src["profile_spec"])}, {time.time() - t0:.0f} s)')
        else:
            mp_ctx = mp.get_context('fork')
            with ProcessPoolExecutor(
                max_workers=n_workers, mp_context=mp_ctx,
                initializer=_init_instrument_photometry_worker,
                initargs=(band_data, per_source, measure_kwargs),
            ) as executor:
                for idd, source_results in executor.map(_instrument_photometry_worker, range(n_sources)):
                    _store(idd, source_results)
                    if verbose:
                        logger.info(f'  [{idd + 1:>4d}/{n_sources}] instrument fit done '
                                    f'({time.time() - t0:.0f} s)')

        # ── assemble + save ────────────────────────────────────────────────
        new_cols = ['profile_instP']
        self.photometry_table['profile_instP'] = np.array(
            [profile_tag(s['profile_spec']) for s in per_source], dtype='<U16')
        for (bl, m), arrs in store.items():
            band = band_original_case[bl]
            for f, arr in arrs.items():
                if f == 'gamma' and m != 'instP':
                    continue
                col = self._INSTRUMENT_COLUMNS[f].format(m=m, b=band)
                self.photometry_table[col] = arr
                new_cols.append(col)

        os.makedirs(outd, exist_ok=True)
        out = Table(self.photometry_table[new_cols])
        out.meta.update(settings)
        out.write(cache_path, overwrite=True)
        logger.info(f'  ✓ Instrument photometry saved → {cache_path}')
        self._save_instrument_aperture_regions(band_original_case)

        # ── per-band summary ───────────────────────────────────────────────
        logger.info('\n  Instrument photometry summary (median flux ratios over sources with both):')
        for bl in band_data:
            band = band_original_case[bl]
            native = np.asarray(self.photometry_table[f'flux_native_{band}'], dtype=float)
            parts = []
            for m in INSTRUMENT_METHODS:
                src_col = np.asarray(store[(bl, m)]['size_source'])
                flux = store[(bl, m)]['flux_mJy']
                ok = np.isfinite(flux) & np.isfinite(native) & (native > 0)
                med = np.median(flux[ok] / native[ok]) if ok.any() else np.nan
                n_own = int(np.sum(src_col == 'own'))
                n_bor = int(np.sum(np.char.startswith(src_col.astype(str), 'borrowed')))
                n_pt = int(np.sum(src_col == 'point_source'))
                parts.append(f'{m}: own={n_own:3d} borrowed={n_bor:3d} point={n_pt:3d} '
                             f'{m}/native={med:5.3f}')
            logger.info(f'    {band:8s}  ' + '   '.join(parts))
        logger.info(f'\n  ✓ Instrument photometry complete ({time.time() - t0:.0f} s)')

        if plot_diagnostics:
            self.plot_instrument_flux_comparison(filter_wavelengths)

    def _fill_missing_instp_gamma(self, band_original_case):
        """gamma_instP_<band> for instrument caches written before it was
        stored: the fixed EFF gamma of profile_instP where R_eff > 0, else NaN."""
        tbl = self.photometry_table
        if 'profile_instP' not in tbl.colnames:
            return
        tags = [str(t).strip() for t in tbl['profile_instP']]
        for band in band_original_case.values():
            col, rcol = f'gamma_instP_{band}', f'reff_instP_{band}'
            if col in tbl.colnames or rcol not in tbl.colnames:
                continue
            reff = np.asarray(tbl[rcol], float)
            g = np.full(len(tbl), np.nan)
            for i, tag in enumerate(tags):
                spec = profile_spec_from_tag(tag)
                if spec['profile'] == 'eff' and not spec['gamma_free'] and reff[i] > 0:
                    g[i] = spec['gamma']
            tbl[col] = g

    def _save_instrument_aperture_regions(self, band_original_case, method='instP'):
        """DS9 circles at each source's instrument-fit (default instP)
        aperture: one <gal>_{method}_apertures_<band>.reg per band."""
        outd = os.path.join(self.base_dir, self.gal)
        os.makedirs(outd, exist_ok=True)
        tbl = self.photometry_table
        for band in band_original_case.values():
            r_col = f'r65_{method}_{band}'
            if r_col not in tbl.colnames:
                continue
            regs = []
            for i in range(len(tbl)):
                ra = float(tbl[f'centroid_ra_{method}_{band}'][i])
                dec = float(tbl[f'centroid_dec_{method}_{band}'][i])
                r = float(tbl[r_col][i])
                if not (np.isfinite(ra) and np.isfinite(dec)):
                    ra, dec = float(tbl['PHANGS_RA'][i]), float(tbl['PHANGS_DEC'][i])
                if not np.isfinite(r):
                    r = float(tbl[f'r65_psf_native_{band}'][i])
                regs.append(CircleSkyRegion(center=SkyCoord(ra=ra * u.deg, dec=dec * u.deg),
                                            radius=r * u.arcsec))
            Regions(regs).write(os.path.join(outd, f'{self.gal}_{method}_apertures_{band}.reg'),
                                format='ds9', overwrite=True)

    def plot_instrument_flux_comparison(self, filter_wavelengths, out_path=None):
        """
        One per-galaxy PNG comparing the three native-resolution methods,
        band by band (box = all sources, points coloured by cluster class):

          1. flux_instG / flux_native and flux_instP / flux_native
          2. flux_native, flux_instG, flux_instP / PHANGS catalog flux
                                      (HST bands only — the catalog has no JWST)
          3. flux_instP / flux_instG  — the systematic from the source profile
          4. rms_instP / rms_instG    — < 1: the cluster profile fits the slit
                                        profile better than a Gaussian (own fits only)

        Saved to output/<gal>/instrument_comparison.png unless out_path given.
        """
        tbl = self.photometry_table
        bands = [b for b in filter_wavelengths if f'flux_instP_{b}' in tbl.colnames]
        if not bands:
            logger.warning('No instrument-photometry columns — run run_instrument_photometry() first.')
            return
        class_colors = {'YCL': 'royalblue', 'MAP': 'darkorange', 'OGCC': 'crimson'}
        classes = (np.array([c.decode().strip() if isinstance(c, bytes) else str(c).strip()
                             for c in tbl['PHANGS_CC_CLASS_ML']])
                   if 'PHANGS_CC_CLASS_ML' in tbl.colnames else np.full(len(tbl), ''))
        rng = np.random.default_rng(0)

        def col(name):
            return np.asarray(tbl[name], dtype=float)

        def ratio(a, b, mask=None):
            with np.errstate(all='ignore'):
                r = a / b
            ok = np.isfinite(r) & (b > 0)
            if mask is not None:
                ok &= mask
            return np.where(ok, r, np.nan)

        def to_catalog(flux_name):
            def fn(b):
                cat_col, _ = _catalog_flux_columns(b, tbl.colnames)
                if cat_col is None:
                    return np.full(len(tbl), np.nan)
                return ratio(col(flux_name.format(b=b)), col(cat_col))
            return fn

        panels = [
            ('flux ratio to native', [
                ('instG / native', 'red', lambda b: ratio(col(f'flux_instG_{b}'), col(f'flux_native_{b}'))),
                ('instP / native', 'darkorange', lambda b: ratio(col(f'flux_instP_{b}'), col(f'flux_native_{b}'))),
            ]),
            ('flux ratio to catalog\n(HST only)', [
                ('native / catalog', 'seagreen', to_catalog('flux_native_{b}')),
                ('instG / catalog', 'red', to_catalog('flux_instG_{b}')),
                ('instP / catalog', 'darkorange', to_catalog('flux_instP_{b}')),
            ]),
            ('flux instP / instG', [
                ('instP / instG', 'purple', lambda b: ratio(col(f'flux_instP_{b}'), col(f'flux_instG_{b}'))),
            ]),
            ('rms instP / rms instG\n(own fits)', [
                ('rms ratio', 'seagreen', lambda b: ratio(
                    col(f'rms_instP_{b}'), col(f'rms_instG_{b}'),
                    mask=np.asarray(tbl[f'size_source_instP_{b}']).astype(str) == 'own')),
            ]),
        ]

        fig, axes = plt.subplots(len(panels), 1, figsize=(max(8, 0.9 * len(bands) + 3), 3.7 * len(panels)),
                                 sharex=True)
        positions = np.arange(len(bands))
        for ax, (ylabel, series) in zip(axes, panels):
            n_ser = len(series)
            width = 0.7 / n_ser
            for k, (label, color, fn) in enumerate(series):
                off = (k - (n_ser - 1) / 2) * width
                data = [fn(b) for b in bands]
                clean = [d[np.isfinite(d)] for d in data]
                ax.boxplot([c if len(c) else [np.nan] for c in clean], positions=positions + off,
                           widths=width * 0.9, showfliers=False, patch_artist=True,
                           boxprops=dict(facecolor=color, alpha=0.35),
                           medianprops=dict(color=color))
                for pos, d in zip(positions, data):
                    ok = np.isfinite(d)
                    jitter = rng.normal(0, width * 0.12, size=ok.sum())
                    ax.scatter(pos + off + jitter, d[ok], s=10, zorder=3, alpha=0.75,
                               c=[class_colors.get(c, 'grey') for c in classes[ok]])
                ax.plot([], [], 's', color=color, alpha=0.5, label=label)
            ax.axhline(1.0, color='k', ls='--', lw=1)
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25)
            ax.legend(loc='upper left', fontsize=8)
        for cls, c in class_colors.items():
            axes[0].scatter([], [], s=12, color=c, label=cls)
        axes[0].legend(loc='upper left', fontsize=8, ncol=5)
        axes[-1].set_xticks(positions)
        axes[-1].set_xticklabels(bands, rotation=45, ha='right')
        prof = getattr(self, 'instrument_settings', {})
        prof_txt = ', '.join(f'{k[2:]}={v}' for k, v in prof.items() if k.startswith('P_'))
        fig.suptitle(f'{self.gal} — instrument-PSF vs native photometry'
                     + (f'  (instP: {prof_txt})' if prof_txt else ''), fontsize=11)
        plt.tight_layout()
        out_path = out_path or os.path.join(self.base_dir, self.gal, 'instrument_comparison.png')
        plt.savefig(out_path, dpi=130)
        plt.close(fig)
        logger.info(f'  ✓ Instrument comparison plot → {out_path}')

    # ─────────────────────────────────────────────────────────────────────────
    # 7. Save photometry table
    # ─────────────────────────────────────────────────────────────────────────

    def save_photometry(self, suffix='photometry'):
        """
        Save ``self.photometry_table`` to a FITS BinTable in the galaxy
        output folder.

        Spectra columns (variable-length Python objects) are dropped before
        saving since FITS BinTable doesn't support ragged arrays; save them
        separately with numpy if needed.

        Parameters
        ----------
        suffix : str   appended to the galaxy name, e.g. 'NGC1234_photometry.fits'
        """
        if self.photometry_table is None:
            raise RuntimeError('No photometry yet — run run_photometry() first.')

        outd = os.path.join(self.base_dir, self.gal)
        os.makedirs(outd, exist_ok=True)
        path = os.path.join(outd, f'{self.gal}_{suffix}.fits')

        # Drop variable-length columns that FITS can't serialise
        tbl = self.photometry_table.copy()
        for col in ['spectrum_Jy', 'spectrum_orig']:
            if col in tbl.colnames:
                tbl.remove_column(col)

        hdr = fits.Header()
        hdr['GALAXY']   = self.gal
        hdr['CREATED']  = datetime.datetime.utcnow().isoformat(timespec='seconds') + 'Z'
        hdr['NOBJ']     = len(tbl)
        hdr['PSF_MUSE'] = (self.psf_muse or 0.0, 'MUSE PSF FWHM [arcsec]')
        hdr['PSFSRC']   = (getattr(self, 'psf_source', 'table'), 'PSF FWHM/shape source: theoretical or table')

        hdu0 = fits.PrimaryHDU(header=hdr)
        hdu1 = fits.BinTableHDU(data=tbl.as_array(), name='PHOTOMETRY')
        fits.HDUList([hdu0, hdu1]).writeto(path, overwrite=True)

        # Save spectra separately as a numpy archive. Two possible sources,
        # depending on whether run_spectro_photometry took its fresh-
        # compute path (spectrum_Jy/spectrum_orig land as columns on
        # self.photometry_table, dropped above before the FITS write) or
        # its cache-hit path (self.photometry_table is read straight from
        # the on-disk _photometry.fits — which never has those columns,
        # since they're stripped before every save — and the spectra
        # instead live in self.photometry_spectra, loaded separately from
        # the existing _spectra.npz).
        spec_path = os.path.join(outd, f'{self.gal}_{suffix}_spectra.npz')
        if 'spectrum_Jy' in self.photometry_table.colnames:
            spectra_Jy = np.array(self.photometry_table['spectrum_Jy'], dtype=object)
            spectra_orig = np.array(self.photometry_table['spectrum_orig'], dtype=object)
        elif getattr(self, 'photometry_spectra', None) is not None:
            spectra_Jy = np.array(self.photometry_spectra['spectra_Jy'], dtype=object)
            spectra_orig = np.array(self.photometry_spectra['spectra_orig'], dtype=object)
        else:
            spectra_Jy = None

        logger.info(f'  ✓ Photometry table saved → {path}')
        if spectra_Jy is not None:
            np.savez(spec_path, spectra_Jy=spectra_Jy, spectra_orig=spectra_orig)
            logger.info(f'  ✓ Spectra saved          → {spec_path}')
        else:
            logger.warning(
                '  No spectra available (photometry_table has no spectrum_Jy column '
                'and photometry_spectra is not set) — skipping spectra .npz save.'
            )

   # ─────────────────────────────────────────────────────────────────────────
    # SED + cutout diagnostic plot
    # ─────────────────────────────────────────────────────────────────────────
 
    def plot_sed(self, idd, dic_passbands, filter_wavelengths,
                 log_wl=None, zoom_factor=6):
        """
        Plot a full diagnostic figure for one source:
 
        Layout
        ------
        Row 0  (7 panels) : MUSE white-light | HST × 5 | JWST[0]
        Row 1  (7 panels) : JWST[1] … JWST[7]
        Row 2  (full)     : SED — MUSE spectrum + photometric points
                            with and without background subtraction,
                            filter width bands, and central wavelength ticks.
 
        Each image cutout shows:
        • The convolved image stretched with asinh normalisation.
        • The source aperture region contour coloured by cluster class
          (YCL = blue, MAP = orange, OGCC = red).
        • A star marker at the catalog sky position of the source.
 
        Parameters
        ----------
        idd               : int    index into ``self.final_table`` (after FOV filter)
        dic_passbands     : dict   band → (wl_min_AA, wl_max_AA)
                                   used to draw filter width spans on the SED.
        filter_wavelengths: dict   band → (central_wl_AA, R_value)
                                   first 5 entries = HST, remainder = JWST.
        log_wl            : 1-D array or None
                                   log-wavelength axis of the MUSE cube.
                                   If provided and the spectrum is available in
                                   ``self.photometry_table``, it is plotted.
        zoom_factor       : float  cutout half-width as multiple of aperture extent.
        """
        if self.photometry_table is None:
            raise RuntimeError(
                'No photometry table — run run_spectro_photometry() first.'
            )
 
        # ── source metadata ───────────────────────────────────────────────────
        row       = self.photometry_table[idd]
        ra        = float(row['PHANGS_RA'])
        dec       = float(row['PHANGS_DEC'])
        cc_class  = str(row['PHANGS_CC_CLASS_ML']).strip()
 
        class_colors = {'YCL': 'royalblue', 'MAP': 'darkorange', 'OGCC': 'crimson'}
        reg_color    = class_colors.get(cc_class, 'white')
 
        sky_pos = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
 
        # ── band ordering ─────────────────────────────────────────────────────
        all_bands   = list(filter_wavelengths.keys())
        hst_bands   = all_bands[:5]
        jwst_bands  = all_bands[5:]          # expected 8 JWST bands
 
        # 2×7 image grid: MUSE + 5 HST + 8 JWST
        top_row    = ['MUSE'] + hst_bands + [jwst_bands[0]]   # 7 panels
        bottom_row = jwst_bands[1:]                             # 7 panels
 
        # ── regions projected to each band WCS ───────────────────────────────
        sky_reg_hst  = self.regions_HST_final[idd]    # PolygonSkyRegion
        sky_reg_muse = self.regions_MUSE_final[idd]
        sky_ann = self.annulus_regions[idd]   # annulus flux (Jy)
 
        def _pixel_reg_and_wcs(band_key):
            """Return (pixel_region, wcs, convolved_data) for a band key."""
            band_key = band_key.lower()
            if band_key == 'muse':
                wcs  = self.muse_wcs
                data = self.muse_data
                preg = sky_reg_hst.to_pixel(wcs)
                anreg = sky_ann.to_pixel(wcs)
            elif band_key in self.hst_conv:
                wcs  = self.hst_images[band_key][2]
                data = self.hst_conv[band_key]
                preg = sky_reg_hst.to_pixel(wcs)
                anreg = sky_ann.to_pixel(wcs)
            else:
                wcs  = self.jwst_images[band_key][2]
                data = self.jwst_conv[band_key]
                preg = sky_reg_hst.to_pixel(wcs)
                anreg = sky_ann.to_pixel(wcs)
            return preg, wcs, data, anreg
 
        # ── figure layout ─────────────────────────────────────────────────────
        fig = plt.figure(figsize=(7 * 2.8, 3 * 3.0))
        gs  = GridSpec(3, 7, figure=fig,
                       height_ratios=[1, 1, 1.4],
                       hspace=0.08, wspace=0.04)
 
        ax_sed = fig.add_subplot(gs[2, :])
 
        # ── helper: draw one cutout panel ─────────────────────────────────────
        def _draw_cutout(ax, band_key, row_idx):
            preg, wcs, data, anreg = _pixel_reg_and_wcs(band_key)
            vx, vy = preg.vertices.x, preg.vertices.y

            theta  = np.linspace(0, 2 * np.pi, 200)
            vx_ann = anreg.center.x + anreg.radius * np.cos(theta)
            vy_ann = anreg.center.y + anreg.radius * np.sin(theta)

            mask_ycl  = self.final_table['CC_CLASS_ML'] == 'YCL'
            mask_map  = self.final_table['CC_CLASS_ML'] == 'MAP'
            mask_ogcc = self.final_table['CC_CLASS_ML'] == 'OGCC'
            
            pos_ycl_filtered  = SkyCoord(self.final_table[mask_ycl]['PHANGS_RA'],
                                        self.final_table[mask_ycl]['PHANGS_DEC'],  unit='deg')
            pos_map_filtered  = SkyCoord(self.final_table[mask_map]['PHANGS_RA'],
                                        self.final_table[mask_map]['PHANGS_DEC'],  unit='deg')
            pos_ogcc_filtered = SkyCoord(self.final_table[mask_ogcc]['PHANGS_RA'],
                                        self.final_table[mask_ogcc]['PHANGS_DEC'], unit='deg')
            
 
            # Zoom window
            x_w = np.max(vx) - np.min(vx)
            y_w = np.max(vy) - np.min(vy)
            cx  = np.min(vx) + x_w / 2
            cy  = np.min(vy) + y_w / 2
            pad = max(x_w, y_w) * zoom_factor
 
            # Stretch parameters differ by instrument
            lo, hi = (15, 99.5) if band_key != 'MUSE' and band_key in hst_bands \
                     else (5, 98.5) if band_key in jwst_bands \
                     else (15, 99.5)
 
            ax.imshow(data, origin='lower',
                      norm=get_norm(data, lo, hi, 'asinh'),
                      cmap='viridis')
 
            # Aperture region contour
            ax.plot(vx, vy, '-', color=reg_color, linewidth=1.8, zorder=5)
            ax.plot(vx_ann, vy_ann, '-', color='white', linewidth=1.8, zorder=5)

            for pos, color, label in [
                                     (pos_ycl_filtered,  'blue',   'YCL'),
                                     (pos_map_filtered,  'orange', 'MAP'),
                                     (pos_ogcc_filtered, 'red',    'OGCC'),
                                     ]:
                x, y = skycoord_to_pixel(pos, wcs)
                ax.scatter(x, y, c=color, s=8, label=label, zorder=3)
 
            # Cluster position marker (star symbol)
            #px, py = sky_pos.to_pixel(wcs)
            #ax.plot(px, py, '*', color=reg_color, markersize=7,
            #        markeredgecolor='white', markeredgewidth=0.5, zorder=6)
 
            ax.set_xlim(cx - pad, cx + pad)
            ax.set_ylim(cy - pad, cy + pad)
 
            # Title: instrument + band
            if band_key == 'MUSE':
                title = 'MUSE\nwhite'
            elif band_key in hst_bands:
                title = f'HST\n{band_key.upper()}'
            else:
                title = f'JWST\n{band_key.upper()}'
            ax.set_title(title, fontsize=7.5, pad=2)
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)
 
            # Thin coloured border matching cluster class
            for spine in ax.spines.values():
                spine.set_edgecolor(reg_color)
                spine.set_linewidth(2)
 
        # ── draw top row ──────────────────────────────────────────────────────
        for col, band_key in enumerate(top_row):
            ax = fig.add_subplot(gs[0, col])
            _draw_cutout(ax, band_key, 0)
 
        # ── draw bottom row ───────────────────────────────────────────────────
        for col, band_key in enumerate(bottom_row):
            ax = fig.add_subplot(gs[1, col])
            _draw_cutout(ax, band_key, 1)
 
        # ── SED panel ─────────────────────────────────────────────────────────
 
        # Filter width spans (rainbow coloured by central wavelength)
        all_cwl_um = [filter_wavelengths[b][0] * u.AA.to('um') for b in all_bands
                      if b in dic_passbands]
        if all_cwl_um:
            norm_c = Normalize(vmin=np.log10(min(all_cwl_um)),
                               vmax=np.log10(max(all_cwl_um)))
            cmap_c = cm.get_cmap('rainbow')
 
            for band_key in all_bands:
                if band_key not in dic_passbands:
                    continue
                wl_lo, wl_hi = dic_passbands[band_key]
                lo_um  = wl_lo * u.AA.to('um')
                hi_um  = wl_hi * u.AA.to('um')
                cl_um  = (lo_um + hi_um) / 2
                color  = cmap_c(norm_c(np.log10(cl_um)))
                ax_sed.axvspan(lo_um, hi_um, alpha=0.35, color=color, zorder=0)
                ax_sed.text(cl_um, 1.0, band_key.upper(),
                            transform=ax_sed.get_xaxis_transform(),
                            ha='center', va='bottom', fontsize=6.5, rotation=45)
 
        # Photometric points
        wl_um   = np.array([filter_wavelengths[b][0] * u.AA.to('um')
                            for b in all_bands])
        fnu     = np.array([float(row[f'flux_{b}'])     for b in all_bands])
        fnu_raw = np.array([float(row[f'flux_{b}_raw']) for b in all_bands])
 
        valid_b = np.isfinite(fnu)     & (fnu > 0)
        valid_r = np.isfinite(fnu_raw) & (fnu_raw > 0)
 
        ax_sed.scatter(wl_um[valid_b], fnu[valid_b],
                       c='black', s=55, zorder=4,
                       label='Photometry (bkg sub)', edgecolors='none')
        ax_sed.scatter(wl_um[valid_r], fnu_raw[valid_r],
                       c='tomato', s=55, marker='s', zorder=4,
                       label='Photometry (no bkg sub)', edgecolors='none')
 
        # Central wavelength tick marks at the top of the SED
        for wl_val in wl_um:
            ax_sed.axvline(wl_val, color='grey', linewidth=0.5,
                           alpha=0.4, zorder=1)
 
        # MUSE spectrum
        try: #'spectrum_Jy' in self.photometry_table.colnames:
            spec = self.photometry_spectra['spectra_Jy'][idd]
        except (KeyError, IndexError):
            spec = None
        if spec is not None:
            wl_muse_um = np.e**(self.log_wl) * u.AA.to('um')
            ax_sed.plot(wl_muse_um, spec,
                        color='steelblue', linewidth=1.0, alpha=0.8,
                        zorder=2, label='MUSE spectrum')
 
        ax_sed.set_xscale('log')
        ax_sed.set_yscale('log')
        ax_sed.set_xlabel(r'Wavelength [$\mu$m]', fontsize=12)
        ax_sed.set_ylabel(r'$F_\nu$ [mJy]',       fontsize=12)
        ax_sed.legend(fontsize=9, loc='lower right')
        ax_sed.grid(True, alpha=0.25, which='both')
        ax_sed.tick_params(axis='both', which='major', labelsize=10)
 
        # ── figure title ──────────────────────────────────────────────────────
        fig.suptitle(
            f'{self.gal}  —  source {idd}  '
            f'({cc_class},  RA={ra:.5f}°,  Dec={dec:.5f}°)',
            fontsize=12, y=1.01,
            color=reg_color
        )
 
        plt.tight_layout()

        out_dir = os.path.join(self.base_dir, self.gal, 'seds')
        plt.savefig(f'{out_dir}/{idd}_{cc_class}.png')
        plt.close()
        #plt.show()

    # ─────────────────────────────────────────────────────────────────────────
    # Native-photometry aperture/annulus QA plot
    # ─────────────────────────────────────────────────────────────────────────

    def plot_native_photometry_diagnostics(self, filter_wavelengths, dic_passbands=None,
                                            idd=None,
                                            annulus_r_in_factor=2.6,
                                            annulus_r_out_factor=4.0,
                                            zoom_factor=8, verbose=True):
        """
        Save one diagnostic PNG per source, showing — for every band with
        native-photometry columns — the native (unconvolved) image cutout
        with THIS source's own R65 aperture and background annulus drawn
        directly on top (annotated with R65 in both arcsec and pixels, and
        color-coded by status_native_{band} — see status_colors below;
        the title tags a borrowed/psf_fallback aperture_source_native_{band}),
        plus a SED panel comparing the native-
        method flux against the existing convolved-image flux and the MUSE
        spectrum. Meant purely for visual QA of aperture placement/sizing
        band by band (e.g. to catch a systematic offset like the JWST
        unit-conversion bug fixed alongside this method — that bug wasn't
        visible from the numbers alone until plotted against wavelength).

        Layout mirrors plot_sed's established 2-row image grid (MUSE +
        HST bands on top, remaining JWST bands on the bottom, SED spanning
        the full width below) — for the current 5 HST + 8 JWST dataset this
        comes out to the same 14 image panels plot_sed uses. The MUSE panel
        shows the region actually used for the MUSE spectrum (self.regions_
        HST_final, reprojected — same convention plot_sed's own MUSE panel
        already uses) purely for spatial QA; it carries no R65 annotation
        since native photometry doesn't measure a MUSE aperture. Skipped
        entirely (falls back to the old HST/JWST-only grid) if self.muse_data
        /self.regions_HST_final/self.annulus_regions aren't available.

        Written to output/<galaxy>/native_diagnostics/ (created if absent).

        Parameters
        ----------
        filter_wavelengths   : dict  band → (central_wl_AA, R_value) — same
                               dict passed to run_native_photometry(); only
                               bands with a r65_native_{band} column
                               (i.e. that actually ran) are plotted.
        dic_passbands        : dict or None — band → (wl_min_AA, wl_max_AA),
                               same dict plot_sed takes. When given, draws
                               the same rainbow-by-central-wavelength filter
                               width spans + rotated band-name labels along
                               the top of the SED panel that plot_sed uses;
                               omitted entirely (no spans) when None.
        idd                  : int or None — plot only this source index
                               (into self.photometry_table); None (default)
                               plots every source.
        annulus_r_in_factor,
        annulus_r_out_factor : float — must match the values passed to
                               run_native_photometry() (same defaults),
                               purely so the drawn annulus matches what was
                               actually used for background subtraction.
        zoom_factor           : float — cutout half-width as a multiple of
                               each band's own R65.
        verbose               : bool
        """
        if self.photometry_table is None:
            raise RuntimeError(
                'No photometry table — run run_spectro_photometry() first.'
            )

        bands = [b for b in filter_wavelengths
                 if f'r65_native_{b}' in self.photometry_table.colnames]
        if not bands:
            raise RuntimeError(
                'No native-photometry columns found for any band in '
                'filter_wavelengths — run run_native_photometry() first.'
            )
        hst_bands  = [b for b in bands if b.lower() in self.hst_images]
        jwst_bands = [b for b in bands if b.lower() in self.jwst_images]

        # Mirrors plot_sed's established row split (MUSE + HST bands on top,
        # remaining JWST bands on the bottom) so this plot's panel count and
        # layout matches a convention already in use, rather than a
        # dynamically-sized grid — for the current 5 HST + 8 JWST dataset
        # this naturally comes out to the same 2x7=14 panels plot_sed uses.
        have_muse = (getattr(self, 'muse_data', None) is not None
                     and getattr(self, 'regions_HST_final', None) is not None
                     and getattr(self, 'annulus_regions', None) is not None)
        top_row = (['muse'] if have_muse else []) + hst_bands + jwst_bands[:1]
        bottom_row = jwst_bands[1:]
        ncols = max(len(top_row), len(bottom_row), 1)

        status_colors = {
            'ok': 'lime', 'size_extrapolated': 'orange',
            'no_detection': 'red', 'borrowed': 'deepskyblue',
            'psf_fallback': 'gold',
        }

        def _draw_cutout(ax, band, row):
            band_lower = band.lower()
            if band_lower in self.hst_images:
                image, header, wcs = self.hst_images[band_lower]
                instrument = 'HST'
            else:
                image, header, wcs = self.jwst_images[band_lower]
                instrument = 'JWST'

            sky_pos = SkyCoord(ra=float(row['PHANGS_RA']) * u.deg,
                                dec=float(row['PHANGS_DEC']) * u.deg)
            px, py = skycoord_to_pixel(sky_pos, wcs)

            pixel_scale = wcs.proj_plane_pixel_scales()[0].to('arcsec').value
            r65_arcsec = float(row[f'r65_native_{band}'])
            status = str(row[f'status_native_{band}']).strip()
            flagged = bool(row[f'recenter_flag_native_{band}'])
            aperture_source = str(row[f'aperture_source_native_{band}']).strip() if f'aperture_source_native_{band}' in row.colnames else ''
            color = status_colors.get(status, 'white')

            if not np.isfinite(px) or not np.isfinite(py):
                # genuine WCS failure — no position at all to center a
                # cutout on, unlike a masked/failed measurement (handled
                # below, where the image is still shown).
                ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                        transform=ax.transAxes, fontsize=8, color='white')
                ax.set_facecolor('black')
                ax.set_title(f'{instrument}\n{band.upper()}', fontsize=7.5, pad=2)
                ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
                return

            lo, hi = (15, 99.5) if instrument == 'HST' else (5, 98.5)
            ax.imshow(image, origin='lower',
                      norm=get_norm(image, lo, hi, 'asinh'),
                      cmap='viridis')

            if not np.isfinite(r65_arcsec):
                # No valid aperture for this source/band (masked — e.g.
                # no_detection, or a measurement that found no flux at all)
                # — still show the actual image + catalog
                # position, just without an aperture/annulus overlay since
                # there's no radius to draw one at.
                # zoom off the band's own PSF-based R65 (always populated,
                # never masked) rather than a hardcoded value, times a
                # generous factor so there's real spatial context visible.
                r65_psf_col = f'r65_psf_native_{band}'
                r65_psf = float(row[r65_psf_col]) if r65_psf_col in row.colnames else np.nan
                zoom_arcsec = r65_psf * 15 if np.isfinite(r65_psf) else 2.5
                pad = (zoom_arcsec / pixel_scale) * 1.15
                ax.plot(px, py, 'x', color=color, markersize=8,
                        markeredgecolor='black', markeredgewidth=0.6, zorder=6)
                ax.set_xlim(px - pad, px + pad)
                ax.set_ylim(py - pad, py + pad)
                ax.set_title(f'{instrument}\n{band.upper()}\n{status}', fontsize=7, pad=2)
                ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
                for spine in ax.spines.values():
                    spine.set_edgecolor(color)
                    spine.set_linewidth(2)
                return

            r65_px = r65_arcsec / pixel_scale
            r_in_px = r65_px * annulus_r_in_factor
            r_out_px = r65_px * annulus_r_out_factor
            pad = r_out_px * 1.3

            theta = np.linspace(0, 2 * np.pi, 200)
            ax.plot(px + r65_px * np.cos(theta), py + r65_px * np.sin(theta),
                    '-', color=color, linewidth=1.8, zorder=5)
            ax.plot(px + r_in_px * np.cos(theta), py + r_in_px * np.sin(theta),
                    '--', color='white', linewidth=1.2, zorder=5)
            ax.plot(px + r_out_px * np.cos(theta), py + r_out_px * np.sin(theta),
                    '--', color='white', linewidth=1.2, zorder=5)
            marker = 'x' if flagged else '*'
            ax.plot(px, py, marker, color=color, markersize=8,
                    markeredgecolor='black', markeredgewidth=0.6, zorder=6)

            # instrument-fit apertures (run_instrument_photometry), if present
            inst_txt = ''
            for m, mcolor, ls in (('instG', 'red', ':'), ('instP', 'darkorange', '--')):
                r_col = f'r65_{m}_{band}'
                if r_col not in row.colnames or not np.isfinite(float(row[r_col])):
                    continue
                cra, cdec = float(row[f'centroid_ra_{m}_{band}']), float(row[f'centroid_dec_{m}_{band}'])
                mx, my = ((px, py) if not (np.isfinite(cra) and np.isfinite(cdec)) else
                          skycoord_to_pixel(SkyCoord(ra=cra * u.deg, dec=cdec * u.deg), wcs))
                rm_px = float(row[r_col]) / pixel_scale
                ax.plot(mx + rm_px * np.cos(theta), my + rm_px * np.sin(theta),
                        ls, color=mcolor, linewidth=1.4, zorder=5)
                inst_txt += f'\n{m}={float(row[r_col]):.3f}"'

            ax.text(0.03, 0.03, f'R65={r65_arcsec:.3f}"\n({r65_px:.1f}px){inst_txt}',
                    transform=ax.transAxes, fontsize=6, color='white',
                    ha='left', va='bottom', zorder=7,
                    bbox=dict(facecolor='black', alpha=0.4, pad=1, edgecolor='none'))

            ax.set_xlim(px - pad, px + pad)
            ax.set_ylim(py - pad, py + pad)
            title = f'{instrument}\n{band.upper()}\n{status}' + (f' [{aperture_source}]' if aperture_source and aperture_source != 'own' else '')
            ax.set_title(title, fontsize=7, pad=2)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(2)

        def _draw_muse_cutout(ax, i):
            # Spatial QA for where the MUSE spectrum plotted in the SED
            # panel actually came from — same region-reprojection pattern
            # plot_sed already uses for its own MUSE panel (the HST-derived
            # region reprojected onto the MUSE WCS, not regions_MUSE_final,
            # which plot_sed's own code never actually uses either). No R65
            # arcsec/px annotation here — native photometry doesn't measure
            # a MUSE aperture at all.
            wcs = self.muse_wcs
            data = self.muse_data
            preg = self.regions_HST_final[i].to_pixel(wcs)
            anreg = self.annulus_regions[i].to_pixel(wcs)
            vx, vy = preg.vertices.x, preg.vertices.y

            theta = np.linspace(0, 2 * np.pi, 200)
            vx_ann = anreg.center.x + anreg.radius * np.cos(theta)
            vy_ann = anreg.center.y + anreg.radius * np.sin(theta)

            x_w = np.max(vx) - np.min(vx)
            y_w = np.max(vy) - np.min(vy)
            cx = np.min(vx) + x_w / 2
            cy = np.min(vy) + y_w / 2
            pad = max(x_w, y_w) * zoom_factor

            ax.imshow(data, origin='lower', norm=get_norm(data, 15, 99.5, 'asinh'), cmap='viridis')
            ax.plot(vx, vy, '-', color='royalblue', linewidth=1.8, zorder=5)
            ax.plot(vx_ann, vy_ann, '-', color='white', linewidth=1.8, zorder=5)
            ax.set_xlim(cx - pad, cx + pad)
            ax.set_ylim(cy - pad, cy + pad)
            ax.set_title('MUSE\nwhite', fontsize=7.5, pad=2)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        def _draw_sed(ax_sed, row, ra, dec, i):
            # Filter width spans (rainbow coloured by central wavelength) —
            # same convention as plot_sed's own SED panel, restricted to
            # `bands` (the bands that actually ran through native
            # photometry) rather than plot_sed's all_bands.
            if dic_passbands:
                cwl_um = [filter_wavelengths[b][0] * u.AA.to('um') for b in bands
                          if b in dic_passbands]
                if cwl_um:
                    norm_c = Normalize(vmin=np.log10(min(cwl_um)),
                                       vmax=np.log10(max(cwl_um)))
                    cmap_c = cm.get_cmap('rainbow')
                    for band_key in bands:
                        if band_key not in dic_passbands:
                            continue
                        wl_lo, wl_hi = dic_passbands[band_key]
                        lo_um = wl_lo * u.AA.to('um')
                        hi_um = wl_hi * u.AA.to('um')
                        cl_um = (lo_um + hi_um) / 2
                        color = cmap_c(norm_c(np.log10(cl_um)))
                        ax_sed.axvspan(lo_um, hi_um, alpha=0.35, color=color, zorder=0)
                        ax_sed.text(cl_um, 1.0, band_key.upper(),
                                    transform=ax_sed.get_xaxis_transform(),
                                    ha='center', va='bottom', fontsize=6.5, rotation=45)

            wl_um = np.array([filter_wavelengths[b][0] * u.AA.to('um') for b in bands])
            conv_flux = np.array([
                float(row[f'flux_{b}']) if f'flux_{b}' in row.colnames else np.nan
                for b in bands
            ])
            native_flux = np.array([float(row[f'flux_native_{b}']) for b in bands])
            native_raw = np.array([float(row[f'flux_native_{b}_raw']) for b in bands])
            native_err = np.array([float(row[f'flux_native_{b}_err']) for b in bands])

            valid_conv = np.isfinite(conv_flux) & (conv_flux > 0)
            valid_native = np.isfinite(native_flux) & (native_flux > 0)
            # 'black' matches plot_sed's own convolved-photometry
            # ('Photometry (bkg sub)') color, so the two diagnostic tools
            # read consistently for the same source.
            ax_sed.scatter(wl_um[valid_conv], conv_flux[valid_conv],
                           c='black', s=55, zorder=3,
                           label='Photometry (convolved)', edgecolors='none')
            ax_sed.scatter(wl_um[valid_native], native_flux[valid_native],
                           c='seagreen', s=55, marker='D', zorder=4,
                           label='Photometry (native)', edgecolors='none')
            # Error bars: only the native method has a propagated flux
            # error (flux_native_{b}_err, from the aperture's background
            # std); the convolved (v1) method's measure_photometry never
            # computes/stores one, so there is nothing real to plot there.
            valid_native_err = valid_native & np.isfinite(native_err)
            ax_sed.errorbar(wl_um[valid_native_err], native_flux[valid_native_err],
                             yerr=native_err[valid_native_err], fmt='none',
                             ecolor='seagreen', elinewidth=1.0, capsize=2.5,
                             alpha=0.7, zorder=3)

            # Instrument-PSF fluxes (run_instrument_photometry), if present
            for m, mcolor, mmark, lbl in (
                    ('instG', 'red', 'x', 'PSF (x) Gaussian'),
                    ('instP', 'darkorange', 'D',
                     f'PSF (x) {str(row["profile_instP"]).strip()}' if 'profile_instP' in row.colnames
                     else 'PSF (x) profile')):
                if not any(f'flux_{m}_{b}' in row.colnames for b in bands):
                    continue
                f_m = np.array([float(row[f'flux_{m}_{b}']) if f'flux_{m}_{b}' in row.colnames
                                else np.nan for b in bands])
                e_m = np.array([float(row[f'flux_{m}_{b}_err']) if f'flux_{m}_{b}_err' in row.colnames
                                else np.nan for b in bands])
                ok = np.isfinite(f_m) & (f_m > 0)
                ax_sed.scatter(wl_um[ok] * (1.03 if m == 'instG' else 1.06), f_m[ok], c=mcolor,
                               s=40 if mmark == 'x' else 30, marker=mmark, zorder=5,
                               label=lbl, linewidths=1.5 if mmark == 'x' else None,
                               edgecolors='none' if mmark != 'x' else None)
                ok_e = ok & np.isfinite(e_m)
                ax_sed.errorbar(wl_um[ok_e] * (1.03 if m == 'instG' else 1.06), f_m[ok_e],
                                yerr=e_m[ok_e], fmt='none', ecolor=mcolor, elinewidth=0.8,
                                capsize=2, alpha=0.6, zorder=4)

            # Original PHANGS cluster-catalog photometry (HST bands only —
            # the catalogs predate JWST), for a 3-way comparison against
            # our own convolved (no aperture correction) and native
            # (R65 aperture-corrected) measurements.
            cat_flux = np.full(len(bands), np.nan)
            cat_err = np.full(len(bands), np.nan)
            for k, b in enumerate(bands):
                flux_col, err_col = _catalog_flux_columns(b, row.colnames)
                if flux_col is not None:
                    cat_flux[k] = float(row[flux_col])
                    if err_col is not None:
                        cat_err[k] = float(row[err_col])
            valid_cat = np.isfinite(cat_flux) & (cat_flux > 0)
            ax_sed.scatter(wl_um[valid_cat], cat_flux[valid_cat],
                           c='dimgray', s=55, marker='^', zorder=3,
                           label='Photometry (catalog)', edgecolors='none')
            # The PHANGS-HST catalog's own propagated flux error can come
            # out negative for some sources (a data-quality artifact of
            # its error propagation, not a real 1-sigma value) — matplotlib
            # rejects a negative yerr outright, and a negative uncertainty
            # isn't meaningful to plot anyway, so treat it as "no usable
            # error" rather than e.g. taking its absolute value.
            valid_cat_err = valid_cat & np.isfinite(cat_err) & (cat_err > 0)
            ax_sed.errorbar(wl_um[valid_cat_err], cat_flux[valid_cat_err],
                             yerr=cat_err[valid_cat_err], fmt='none',
                             ecolor='dimgray', elinewidth=1.0, capsize=2.5,
                             alpha=0.7, zorder=2)

            for k, b in enumerate(bands):
                status = str(row[f'status_native_{b}']).strip()
                if status == 'no_detection' and np.isfinite(native_raw[k]) and native_raw[k] > 0:
                    ax_sed.scatter(wl_um[k], native_raw[k], marker='v', c='grey',
                                   s=45, zorder=2, label='_nolegend_')

            # MUSE spectrum, when available — same source (self.photometry_spectra,
            # self.log_wl) and lack of background subtraction as plot_sed's
            # own MUSE-spectrum block; not a new gap, just carried over.
            if getattr(self, 'photometry_spectra', None) is not None and getattr(self, 'log_wl', None) is not None:
                try:
                    spec = self.photometry_spectra['spectra_Jy'][i]
                except (KeyError, IndexError):
                    spec = None
                if spec is not None:
                    wl_muse_um = np.e ** (self.log_wl) * u.AA.to('um')
                    ax_sed.plot(wl_muse_um, spec, color='navy', linewidth=1.0,
                                alpha=0.8, zorder=2, label='MUSE spectrum')

            for wl_val in wl_um:
                ax_sed.axvline(wl_val, color='grey', linewidth=0.5, alpha=0.4, zorder=1)

            ax_sed.set_xscale('log')
            ax_sed.set_yscale('log')
            ax_sed.set_xlabel(r'Wavelength [$\mu$m]', fontsize=12)
            ax_sed.set_ylabel(r'$F_\nu$ [mJy]', fontsize=12)
            ax_sed.legend(fontsize=9, loc='lower right')
            ax_sed.grid(True, alpha=0.25, which='both')
            ax_sed.tick_params(axis='both', which='major', labelsize=10)

        out_dir = os.path.join(self.base_dir, self.gal, 'native_diagnostics')
        os.makedirs(out_dir, exist_ok=True)

        idds = range(len(self.photometry_table)) if idd is None else [idd]
        n_total = len(self.photometry_table)
        for i in idds:
            row = self.photometry_table[i]
            ra = float(row['PHANGS_RA'])
            dec = float(row['PHANGS_DEC'])
            cc_class = str(row['PHANGS_CC_CLASS_ML']).strip() if 'PHANGS_CC_CLASS_ML' in row.colnames else ''

            fig = plt.figure(figsize=(ncols * 2.8, 2 * 3.0 + 3.2))
            gs = GridSpec(3, ncols, figure=fig, height_ratios=[1, 1, 1.4],
                          hspace=0.15, wspace=0.04)

            for col, band in enumerate(top_row):
                ax = fig.add_subplot(gs[0, col])
                if band == 'muse':
                    _draw_muse_cutout(ax, i)
                else:
                    _draw_cutout(ax, band, row)
            for col, band in enumerate(bottom_row):
                _draw_cutout(fig.add_subplot(gs[1, col]), band, row)

            ax_sed = fig.add_subplot(gs[2, :])
            _draw_sed(ax_sed, row, ra, dec, i)

            fig.suptitle(
                f'{self.gal}  —  source {i}  '
                f'({cc_class},  RA={ra:.5f}°,  Dec={dec:.5f}°)',
                fontsize=12, y=1.01,
            )
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f'{i}_{cc_class}_native.png'))
            plt.close(fig)

            if verbose:
                logger.info(f'  [{i + 1:>4d}/{n_total}] native diagnostic plot saved')

        logger.info(f'  ✓ Native photometry diagnostic plots saved → {out_dir}')

    # ─────────────────────────────────────────────────────────────────────────
    # Native-photometry SIZE-calibration QA plot
    # ─────────────────────────────────────────────────────────────────────────

    def plot_native_size_calibration(self, filter_wavelengths, idd=None, verbose=True):
        """
        Save one diagnostic PNG per source showing, for every band with
        native-photometry columns, the band-level SIZE calibration curve
        (extra-convolution FWHM vs. mean measured sigma — the same curve
        estimate_extra_conv_fwhm() inverts to size this cluster's own R65)
        with THIS source's own measured sigma marked on it — directly
        answering "why was this aperture/correction chosen?": R65 is, by
        construction, the radius enclosing exactly target_ee of a model
        matching that sigma, so the marked point IS the size estimate the
        aperture (and hence the always-1/target_ee correction factor) is
        based on. A band whose aperture_source isn't 'own' (borrowed or
        psf_fallback — no independent size measurement for this band) shows
        a text note instead of a marked point.

        The calibration sweep starts at model 0 = the bare PSF (zero extra
        convolution), so a source can only appear NARROWER than the curve's
        floor from measurement noise — it can never truly be sharper than
        the PSF itself. A source pushed past the curve's CEILING (10x the
        PSF's own sigma of extra convolution, already a generous sweep) is
        flagged size_extrapolated — a genuinely large/resolved cluster, a
        noisy fit latching onto background, or contamination from a near
        neighbour are the likely causes, worth checking against
        plot_native_photometry_diagnostics' aperture cutout for the same
        source/band.

        Requires self.native_band_calib (populated by run_native_photometry()
        itself — this cannot be usefully called on its own beforehand).

        Written to output/<galaxy>/native_size_calibration/ (created if absent).

        Parameters
        ----------
        filter_wavelengths : dict  band → (central_wl_AA, R_value) — same
                             dict passed to run_native_photometry(); only
                             bands with a sigma_native_{band} column (i.e.
                             that actually ran) are plotted.
        idd                 : int or None — plot only this source index
                             (into self.photometry_table); None (default)
                             plots every source.
        verbose             : bool
        """
        if self.photometry_table is None:
            raise RuntimeError(
                'No photometry table — run run_spectro_photometry() first.'
            )
        if not getattr(self, 'native_band_calib', None):
            raise RuntimeError(
                'self.native_band_calib not populated — run run_native_photometry() first.'
            )

        bands = [b for b in filter_wavelengths
                 if f'sigma_native_{b}' in self.photometry_table.colnames
                 and b.lower() in self.native_band_calib]
        if not bands:
            raise RuntimeError(
                'No native-photometry sigma_native_{band} columns '
                'found for any band in filter_wavelengths — run '
                'run_native_photometry() first.'
            )

        ncols = min(len(bands), 4)
        nrows = int(np.ceil(len(bands) / ncols))

        out_dir = os.path.join(self.base_dir, self.gal, 'native_size_calibration')
        os.makedirs(out_dir, exist_ok=True)

        idds = range(len(self.photometry_table)) if idd is None else [idd]
        n_total = len(self.photometry_table)
        for i in idds:
            row = self.photometry_table[i]
            ra = float(row['PHANGS_RA'])
            dec = float(row['PHANGS_DEC'])
            cc_class = str(row['PHANGS_CC_CLASS_ML']).strip() if 'PHANGS_CC_CLASS_ML' in row.colnames else ''

            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.6, nrows * 3.2), squeeze=False)

            for j, band in enumerate(bands):
                ax = axes[j // ncols][j % ncols]
                curve = self.native_band_calib[band.lower()]['calib_curve']
                sb_model = curve.sigma_bar_model
                fwhm_model = curve.extra_conv_fwhm_model

                ax.plot(sb_model, fwhm_model, '-o', color='steelblue',
                        markersize=3, linewidth=1.3, label='calibration models')

                aperture_source = (str(row[f'aperture_source_native_{band}']).strip()
                                    if f'aperture_source_native_{band}' in row.colnames else 'own')
                sb_src = float(row[f'sigma_native_{band}'])
                extrapolated = bool(row[f'size_extrapolated_native_{band}'])
                capped = (bool(row[f'size_capped_native_{band}'])
                          if f'size_capped_native_{band}' in row.colnames else False)
                fwhm_est = float(row[f'extra_conv_fwhm_est_native_{band}'])
                color = 'darkorange' if capped else ('crimson' if extrapolated else 'seagreen')

                xmax = sb_model[-1]
                if aperture_source != 'own':
                    ax.text(0.5, 0.5, aperture_source.replace('_', ' '),
                            ha='center', va='center', transform=ax.transAxes,
                            fontsize=8, color='grey')
                elif np.isfinite(sb_src):
                    xmax = max(xmax, sb_src)
                    ax.axvline(sb_src, color=color, linestyle='--', linewidth=1.2)
                    ax.scatter([sb_src], [fwhm_est], color=color, s=70, zorder=5,
                               marker='P' if capped else ('X' if extrapolated else '*'),
                               edgecolors='black', linewidths=0.5)
                    if extrapolated and sb_src > sb_model[-1]:
                        ax.plot([sb_model[-1], sb_src], [fwhm_model[-1], fwhm_model[-1]],
                                ':', color=color, linewidth=1.3)
                    if capped:
                        # The size ESTIMATE plotted above (sb_src, fwhm_est) is
                        # the raw, uncapped measurement — what was actually
                        # used for the aperture is r65_native_{band}, clamped
                        # to max_r65_growth_factor * r65_psf_native_{band}
                        # (see native_phot_aux_func._measure_own_band). Made
                        # explicit here since the marked point itself doesn't
                        # otherwise show that the aperture was pulled in.
                        r65_used = float(row[f'r65_native_{band}'])
                        r65_psf = float(row[f'r65_psf_native_{band}'])
                        ax.text(0.97, 0.03,
                                f'aperture capped:\nR65={r65_used:.3f}"\n'
                                f'({r65_used / r65_psf:.1f}× PSF R65)',
                                transform=ax.transAxes, fontsize=6.5, color=color,
                                ha='right', va='bottom',
                                bbox=dict(facecolor='white', alpha=0.7, pad=1.5, edgecolor=color))
                ax.axvspan(sb_model[-1], xmax * 1.02 if xmax > 0 else sb_model[-1] + 1e-3,
                           color='crimson', alpha=0.08, zorder=0)
                ax.set_xlim(min(0.0, sb_model[0]), xmax * 1.05 if xmax > 0 else sb_model[-1])

                ax.set_xlabel(r'$\sigma$ [arcsec]', fontsize=8)
                ax.set_ylabel('extra conv. FWHM [arcsec]', fontsize=8)
                title = band.upper() + ('  CAPPED' if capped else ('  EXTRAPOLATED' if extrapolated else ''))
                ax.set_title(title, fontsize=8, color=color if (capped or extrapolated) else 'black')
                ax.tick_params(labelsize=7)
                ax.grid(alpha=0.25)

            for j in range(len(bands), nrows * ncols):
                axes[j // ncols][j % ncols].axis('off')

            fig.suptitle(
                f'{self.gal}  —  source {i}  size-calibration QA  '
                f'({cc_class},  RA={ra:.5f}°,  Dec={dec:.5f}°)',
                fontsize=11, y=1.02,
            )
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f'{i}_{cc_class}_size_calibration.png'),
                        bbox_inches='tight')
            plt.close(fig)

            if verbose:
                logger.info(f'  [{i + 1:>4d}/{n_total}] size-calibration diagnostic plot saved')

        logger.info(f'  ✓ Native size-calibration diagnostic plots saved → {out_dir}')

    # ─────────────────────────────────────────────────────────────────────────
    # Native-photometry SLIT-PROFILE QA plot (one PNG per source, all bands)
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_instrument_slit_panel(self, ax, ax_img, row, band, bd, n_slits=12):
        """
        Third column of plot_native_slit_profiles: the averaged slit profile
        through the instrument-fit centre (slit width as in the instrument
        fit) with the PSF (x) Gaussian (red) and PSF (x) profile (orange)
        models at their fitted/borrowed R_eff, and ±R65 of the native (blue),
        instG and instP apertures; also draws the instG/instP circles on the
        image panel. Model amplitude and background are the linear least-
        squares solution at that R_eff, i.e. what the fit itself solves.
        Returns False (panel left blank) if there is nothing to show.
        """
        psf_model = getattr(self, 'instrument_psf', {}).get(band.lower())
        if psf_model is None or f'reff_instP_{band}' not in row.colnames:
            return False
        settings = getattr(self, 'instrument_settings', {})
        slit_w = int(settings.get('SLITW', 1))
        slit_min = float(settings.get('SLITLEN', 1.0))
        wcs, pix = bd['wcs'], bd['pixel_scale_arcsec']
        calib = getattr(self, 'native_band_calib', {}).get(band.lower(), {})
        slit_len = instrument_slit_length(calib, slit_min)
        reffs = {m: float(row[f'reff_{m}_{band}']) for m in INSTRUMENT_METHODS}
        big = max([r for r in reffs.values() if np.isfinite(r)] or [0.0])
        if 5.0 * big > slit_len and slit_len < 3.0:
            slit_len = min(5.0 * big, 3.0)

        cra, cdec = float(row[f'centroid_ra_instP_{band}']), float(row[f'centroid_dec_instP_{band}'])
        if not (np.isfinite(cra) and np.isfinite(cdec)):
            return False
        x0, y0 = skycoord_to_pixel(SkyCoord(ra=cra * u.deg, dec=cdec * u.deg), wcs)
        x0, y0 = float(x0), float(y0)
        along, theta_deg, per_slit, y_data, _ = _sample_averaged_slit_profile(
            bd['image'], x0, y0, pix, slit_len, n_slits, slit_w)
        ok = np.isfinite(y_data)
        if ok.sum() < 5:
            return False
        # the individual slits of THIS fit (they differ from the native ones:
        # own width/length, centred on the instrument-fit position)
        for k in range(per_slit.shape[0]):
            good_k = np.isfinite(per_slit[k])
            ax.plot(along[good_k], per_slit[k][good_k], '-', color='silver', lw=0.5, alpha=0.5,
                    zorder=1, label=f'{n_slits} slits' if k == 0 else None)
        ax.plot(along[ok], y_data[ok], 'k.', ms=2.5, zorder=3,
                label=f'averaged data ({slit_w}px slits, {slit_len:.2f}")')
        # the native method draws its own slits on the image only for 'own'
        # bands; elsewhere draw these, so the image always shows what was sampled
        if str(row[f'aperture_source_native_{band}']).strip() != 'own':
            r_px = slit_len / pix
            for th in np.deg2rad(theta_deg):
                ax_img.plot([x0 - r_px * np.cos(th), x0 + r_px * np.cos(th)],
                            [y0 - r_px * np.sin(th), y0 + r_px * np.sin(th)],
                            '-', color='orange', lw=0.6, alpha=0.5, zorder=4)

        prof_tag = str(row['profile_instP']).strip() if 'profile_instP' in row.colnames else 'profile'
        styles = {'instG': ('red', '-', 'PSF (x) Gauss'), 'instP': ('darkorange', '-.', f'PSF (x) {prof_tag}')}
        theta = np.linspace(0, 2 * np.pi, 100)
        for m, (color, ls, lbl) in styles.items():
            r_eff, r65 = reffs[m], float(row[f'r65_{m}_{band}'])
            if not (np.isfinite(r_eff) and np.isfinite(r65)):
                continue
            spec = (dict(profile='gaussian', gamma=3.0, king_c=1.5) if m == 'instG'
                    else self._profile_spec_from_tag(prof_tag, row, band))
            _, unit = instrument_fit_model_profile(psf_model, x0, y0, r_eff, spec['profile'], 1.0, 0.0,
                                                   slit_len, n_slits, slit_w,
                                                   gamma=spec['gamma'], king_c=spec['king_c'])
            good = ok & np.isfinite(unit)
            A = np.vstack([unit[good], np.ones(good.sum())]).T
            (amp, bkg), *_ = np.linalg.lstsq(A, y_data[good], rcond=None)
            src = str(row[f'size_source_{m}_{band}']).strip()
            reff_pc = float(row[f'reff_pc_{m}_{band}'])
            g_txt = f' γ={spec["gamma"]:.2f}' if spec['profile'] == 'eff' and r_eff > 0 else ''
            ax.plot(along, amp * unit + bkg, ls, color=color, lw=1.5,
                    label=f'{lbl}:{g_txt} R_eff={reff_pc:.1f}pc [{src}]')
            for s in (-1, 1):
                ax.axvline(s * r65, color=color, ls=':', lw=1)
            rx, ry = skycoord_to_pixel(SkyCoord(ra=cra * u.deg, dec=cdec * u.deg), wcs) \
                if m == 'instP' else skycoord_to_pixel(SkyCoord(
                    ra=float(row[f'centroid_ra_instG_{band}']) * u.deg,
                    dec=float(row[f'centroid_dec_instG_{band}']) * u.deg), wcs)
            ax_img.plot(rx + r65 / pix * np.cos(theta), ry + r65 / pix * np.sin(theta),
                        ls, color=color, lw=1.2, zorder=5)
        r65_nat = float(row[f'r65_native_{band}']) if f'r65_native_{band}' in row.colnames else np.nan
        if np.isfinite(r65_nat):
            for s in (-1, 1):
                ax.axvline(s * r65_nat, color='deepskyblue', ls='--', lw=1)
        ax.plot([], [], ':', color='grey', label='±R65 (blue: native)')
        flux_txt = '  '.join(
            f'{m}/nat={float(row[f"flux_{m}_{band}"]) / float(row[f"flux_native_{band}"]):.2f}'
            for m in INSTRUMENT_METHODS
            if f'flux_native_{band}' in row.colnames and np.isfinite(float(row[f'flux_{m}_{band}']))
            and np.isfinite(float(row[f'flux_native_{band}'])) and float(row[f'flux_native_{band}']) > 0)
        ax.set_title(f'instrument fit   {flux_txt}', fontsize=7.5)
        ax.set_xlabel('Radius [arcsec]', fontsize=7.5)
        ax.tick_params(labelsize=6.5)
        ax.set_xlim(-min(slit_len, 6 * max(r65_nat if np.isfinite(r65_nat) else 0, 0.1)),
                    min(slit_len, 6 * max(r65_nat if np.isfinite(r65_nat) else 0, 0.1)))
        ax.legend(fontsize=5.5, loc='upper right')
        return True

    @staticmethod
    def _profile_spec_from_tag(tag, row=None, band=None):
        """instP model spec of a source: profile_tag inverse, with the gamma
        fitted (or borrowed) in this band when row/band are given."""
        g = None
        if row is not None and band is not None and f'gamma_instP_{band}' in row.colnames:
            g = float(row[f'gamma_instP_{band}'])
        return profile_spec_from_tag(tag, gamma=g)

    # ─────────────────────────────────────────────────────────────────────
    # instP diagnostic plots: slit profile (+ Deger+22 comparison) and 2D
    # data / model / residual, one row per band
    # ─────────────────────────────────────────────────────────────────────
    def _instp_band_setup(self, row, band):
        """Everything the instP plots need for one source and band, or None.
        Centre, R_eff, R65 and profile are the stored instP results; the slits
        are rebuilt exactly as in the fit (width, length rule, SLIT_COMBINE)."""
        bl = band.lower()
        calib = getattr(self, 'native_band_calib', {}).get(bl)
        psf_model = getattr(self, 'instrument_psf', {}).get(bl)
        if calib is None or psf_model is None or f'reff_instP_{band}' not in row.colnames:
            return None
        cra, cdec = float(row[f'centroid_ra_instP_{band}']), float(row[f'centroid_dec_instP_{band}'])
        if not (np.isfinite(cra) and np.isfinite(cdec)):
            cra, cdec = float(row['PHANGS_RA']), float(row['PHANGS_DEC'])
        x0, y0 = skycoord_to_pixel(SkyCoord(ra=cra * u.deg, dec=cdec * u.deg), calib['wcs'])
        x0, y0 = float(x0), float(y0)
        if not (np.isfinite(x0) and np.isfinite(y0)):
            return None
        settings = getattr(self, 'instrument_settings', {}) or {}
        r_eff = float(row[f'reff_instP_{band}'])
        slit_len = instrument_slit_length(calib, float(settings.get('SLITLEN', 1.0)))
        if np.isfinite(r_eff) and 5.0 * r_eff > slit_len and slit_len < 3.0:
            slit_len = min(5.0 * r_eff, 3.0)
        tag = str(row['profile_instP']).strip() if 'profile_instP' in row.colnames else 'eff3.0'
        conv = calib.get('unit_convert')
        mjy_per_unit = (conv(1.0, calib['header']) if conv is not None else 1.0) * 1e3
        r_value, eb_v = calib.get('r_value'), getattr(self, 'eb_v', 0.0) or 0.0
        mw = float(corr_flux(1.0, r_value, eb_v)) if r_value is not None and eb_v else 1.0
        get = lambda c: float(row[c]) if c in row.colnames else np.nan  # noqa: E731
        return dict(
            calib=calib, psf=psf_model, image=calib['image'], pix=calib['pixel_scale_arcsec'],
            instrument=calib.get('instrument', ''), x0=x0, y0=y0, r_eff=r_eff,
            r65=get(f'r65_instP_{band}'), reff_pc=get(f'reff_pc_instP_{band}'),
            spec=self._profile_spec_from_tag(tag, row, band), tag=tag, slit_len=slit_len,
            slit_w=int(settings.get('SLITW', 1)), n_slits=int(settings.get('NSLITS', 12)),
            ann_in=float(settings.get('ANNIN', 2.6)), ann_out=float(settings.get('ANNOUT', 4.0)),
            flux=get(f'flux_instP_{band}'), flux_err=get(f'flux_instP_{band}_err'),
            corr=get(f'corr_factor_instP_{band}'), bkg=get(f'bkg_median_instP_{band}'),
            bkg_std=get(f'bkg_std_instP_{band}'),
            size_source=str(row[f'size_source_instP_{band}']).strip()
            if f'size_source_instP_{band}' in row.colnames else '',
            status=str(row[f'status_instP_{band}']).strip() if f'status_instP_{band}' in row.colnames else '',
            mjy_per_unit=mjy_per_unit, mw=mw,
        )

    @staticmethod
    def _instp_window_arcsec(g):
        """half-size of the image cutouts: the background annulus plus a margin"""
        r65 = g['r65'] if np.isfinite(g['r65']) else 0.1
        return max(1.15 * g['ann_out'] * r65, 0.45)

    @staticmethod
    def _instp_cutout(g, half_arcsec):
        """(sub-image, extent in arcsec relative to the instP centre)"""
        img, pix, x0, y0 = g['image'], g['pix'], g['x0'], g['y0']
        h = int(np.ceil(half_arcsec / pix))
        ix, iy = int(np.round(x0)), int(np.round(y0))
        ny, nx = img.shape
        ys, xs = slice(max(iy - h, 0), min(iy + h + 1, ny)), slice(max(ix - h, 0), min(ix + h + 1, nx))
        sub = np.asarray(img[ys, xs], float)
        ext = [(xs.start - 0.5 - x0) * pix, (xs.stop - 0.5 - x0) * pix,
               (ys.start - 0.5 - y0) * pix, (ys.stop - 0.5 - y0) * pix]
        return sub, ext, (xs, ys)

    @staticmethod
    def _local_norm(sub, bkg=np.nan, bkg_std=np.nan, peak=np.nan):
        """asinh stretch from the cutout itself: sky - 1 sigma up to the source's
        own peak (brightest pixel within R65), so the source fills the range"""
        from astropy.visualization import ImageNormalize, AsinhStretch
        finite = sub[np.isfinite(sub)]
        if finite.size == 0:
            return None
        lo = bkg - 1.0 * bkg_std if np.isfinite(bkg) and np.isfinite(bkg_std) else np.percentile(finite, 5)
        hi = peak if np.isfinite(peak) and peak > lo else np.percentile(finite, 99.9)
        if not hi > lo:
            hi = lo + 1.0
        return ImageNormalize(vmin=lo, vmax=hi, stretch=AsinhStretch(0.1), clip=True)

    @staticmethod
    def _source_peak(sub, ext, r_arcsec):
        """brightest pixel within r_arcsec of the centre of an _instp_cutout"""
        ny, nx = sub.shape
        xs = np.linspace(ext[0], ext[1], nx + 1)[:-1] + (ext[1] - ext[0]) / nx / 2
        ys = np.linspace(ext[2], ext[3], ny + 1)[:-1] + (ext[3] - ext[2]) / ny / 2
        rr = np.hypot(*np.meshgrid(xs, ys))
        m = (rr <= max(r_arcsec, (ext[1] - ext[0]) / nx)) & np.isfinite(sub)
        return float(np.max(sub[m])) if m.any() else np.nan

    def _draw_instp_slit_row(self, ax_img, ax_prof, row, band, g):
        """left: cutout + slits + apertures; right: combined slit profile with the
        instP model, R65, background annulus, and the Deger+22 4 px aperture with
        its galaxy correction vs the exact one for this cluster"""
        pix, x0, y0 = g['pix'], g['x0'], g['y0']
        r65 = g['r65']
        half = self._instp_window_arcsec(g)
        sub, ext, _ = self._instp_cutout(g, half)
        norm = self._local_norm(sub, g['bkg'], g['bkg_std'],
                                self._source_peak(sub, ext, g['r65'] if np.isfinite(g['r65']) else 0.1))
        ax_img.imshow(sub, origin='lower', cmap='gray_r', norm=norm, extent=ext, interpolation='nearest')
        theta_deg = np.arange(g['n_slits']) * 180.0 / g['n_slits']
        for th in np.deg2rad(theta_deg):
            ax_img.plot([-g['slit_len'] * np.cos(th), g['slit_len'] * np.cos(th)],
                        [-g['slit_len'] * np.sin(th), g['slit_len'] * np.sin(th)],
                        '-', color='#eda100', lw=0.4, alpha=0.3, zorder=3)
        circ = np.linspace(0, 2 * np.pi, 120)
        rings = [(DEGER22_R4_ARCSEC, '#eb6834', '-', 0.8), (DEGER22_SKY_ARCSEC[0], '#2a78d6', '--', 0.7),
                 (DEGER22_SKY_ARCSEC[1], '#2a78d6', '--', 0.7)]
        if np.isfinite(r65):
            rings += [(r65, '#4a3aa7', '-', 1.0), (g['ann_in'] * r65, '#4a3aa7', ':', 0.8),
                      (g['ann_out'] * r65, '#4a3aa7', ':', 0.8)]
        for rad, col, ls, lw in rings:
            ax_img.plot(rad * np.cos(circ), rad * np.sin(circ), ls, color=col, lw=lw, alpha=0.75, zorder=4)

        ax_img.set_xlim(-half, half); ax_img.set_ylim(-half, half)
        ax_img.set_ylabel(f'{g["instrument"]} {band}', fontsize=10, fontweight='bold')
        ax_img.set_title(f'{g["size_source"] or "-"}   {g["status"]}', fontsize=8)
        ax_img.tick_params(labelsize=7)
        ax_img.set_xlabel('arcsec', fontsize=7)

        # profile: median (SLIT_COMBINE) of the instP slits, in uJy / px above the annulus sky
        along, _, per_slit, y, _ = _sample_averaged_slit_profile(
            g['image'], x0, y0, pix, g['slit_len'], g['n_slits'], g['slit_w'])
        to_ujy = g['mjy_per_unit'] * 1e3
        sky = g['bkg'] if np.isfinite(g['bkg']) else np.nanmedian(y)
        for k in range(per_slit.shape[0]):
            ok_k = np.isfinite(per_slit[k])
            ax_prof.plot(along[ok_k], (per_slit[k][ok_k] - sky) * to_ujy, '-', color='#c9c8c2', lw=0.6,
                         zorder=1, label=f'{g["n_slits"]} slits' if k == 0 else None)
        ok = np.isfinite(y)
        ax_prof.plot(along[ok], (y[ok] - sky) * to_ujy, 'o', color='#1a1a19', ms=2.6, zorder=3,
                     label=f'{SLIT_COMBINE} of the slits ({g["slit_w"]} px wide)')
        if np.isfinite(g['r_eff']):
            _, unit = instrument_fit_model_profile(g['psf'], x0, y0, g['r_eff'], g['spec']['profile'], 1.0, 0.0,
                                                   g['slit_len'], g['n_slits'], g['slit_w'],
                                                   gamma=g['spec']['gamma'], king_c=g['spec']['king_c'])
            good = ok & np.isfinite(unit)
            if good.sum() > 3:
                A = np.vstack([unit[good], np.ones(good.sum())]).T
                (amp, bkg_fit), *_ = np.linalg.lstsq(A, y[good], rcond=None)
                ax_prof.plot(along, (amp * unit + bkg_fit - sky) * to_ujy, '-', color='#4a3aa7', lw=2.0,
                             zorder=4, label=f'instP: PSF (x) {g["tag"]}'
                             + (f' (γ={g["spec"]["gamma"]:.2f})'
                                if g['spec']['profile'] == 'eff' and g['r_eff'] > 0 else '')
                             + f', R_eff {g["reff_pc"]:.1f} pc')
                ax_prof.axhline((bkg_fit - sky) * to_ujy, color='#4a3aa7', lw=0.8, ls=':', zorder=2)
        ax_prof.axhline(0, color='#5f5e58', lw=0.7, zorder=2)
        for sgn in (-1, 1):
            if np.isfinite(r65):
                ax_prof.axvline(sgn * r65, color='#4a3aa7', lw=1.3, ls='--', zorder=2)
                ax_prof.axvspan(sgn * g['ann_in'] * r65, sgn * g['ann_out'] * r65, color='#4a3aa7',
                                alpha=0.08, lw=0)
            ax_prof.axvline(sgn * DEGER22_R4_ARCSEC, color='#eb6834', lw=1.1, ls='-.', zorder=2)
            ax_prof.axvspan(sgn * DEGER22_SKY_ARCSEC[0], sgn * DEGER22_SKY_ARCSEC[1], color='#2a78d6',
                            alpha=0.15, lw=0)
        ax_prof.plot([], [], '--', color='#4a3aa7', label='±R65 (instP aperture)')
        ax_prof.plot([], [], '-.', color='#eb6834', label='±4 px (Deger+22 aperture)')
        ax_prof.set_xlim(-half, half)
        ax_prof.set_ylabel('uJy / px above sky', fontsize=8)
        ax_prof.set_xlabel('radius along slit [arcsec]', fontsize=8)
        ax_prof.tick_params(labelsize=7)

        # corrections and fluxes
        f4, e4 = deger22_4px_flux(g['image'], x0, y0, pix)
        f4_mjy, e4_mjy = f4 * g['mjy_per_unit'] * g['mw'], e4 * g['mjy_per_unit'] * g['mw']
        k_gal = deger22_correction_factor(self.gal, band)
        k_ex = exact_4px_correction(g['psf'], g['r_eff'], g['spec'], x0 - np.round(x0), y0 - np.round(y0))
        lines = [f'instP  corr x{g["corr"]:.3f}  ->  F = {g["flux"]:.4g} ± {g["flux_err"]:.2g} mJy',
                 f'Deger+22 4 px: F4 = {f4_mjy:.4g} ± {e4_mjy:.2g} mJy']
        if np.isfinite(k_gal):
            lines.append(f'  galaxy corr x{k_gal:.3f} (AC {-2.5 * np.log10(k_gal):+.2f})  ->  F = {f4_mjy * k_gal:.4g}')
        else:
            lines.append('  galaxy corr: n/a (not in Deger+22 / not HST)')
        if np.isfinite(k_ex):
            lines.append(f'  exact corr x{k_ex:.3f} (R_eff {g["reff_pc"]:.1f} pc)  ->  F = {f4_mjy * k_ex:.4g}')
        ax_prof.text(0.99, 0.97, '\n'.join(lines), transform=ax_prof.transAxes, ha='right', va='top',
                     fontsize=7.2, family='monospace',
                     bbox=dict(fc='white', ec='#c9c8c2', alpha=0.9, boxstyle='round,pad=0.3'), zorder=6)
        ax_prof.legend(fontsize=6.5, loc='upper left', frameon=True, framealpha=0.85)

    def _instp_2d_model(self, g, xs, ys):
        """instP model image on the cutout grid (image units): flux_instP (back to
        image units, without the MW correction) x PSF (x) profile + annulus sky"""
        if not (np.isfinite(g['flux']) and np.isfinite(g['r_eff'])):
            return None
        f_img = g['flux'] / g['mw'] / g['mjy_per_unit']
        shape = (ys.stop - ys.start, xs.stop - xs.start)
        kw = dict(profile=g['spec']['profile'], r_eff_arcsec=g['r_eff'], gamma=g['spec']['gamma'],
                  c=g['spec']['king_c']) if g['r_eff'] > 0 else {}
        m = g['psf'].render(g['x0'] - xs.start, g['y0'] - ys.start, shape, flux=f_img, **kw)
        return m + (g['bkg'] if np.isfinite(g['bkg']) else 0.0)

    def plot_instrument_2d_models(self, filter_wavelengths, idd=None, bands=None, verbose=False):
        """
        One PNG per source, one row per band: the image, the instP model in 2D
        (the reported flux_instP x PSF (x) profile at the fitted/borrowed R_eff,
        on the annulus sky) and data - model (+-5 sigma of the annulus), with the
        R65 aperture and its background annulus. Shows how well the 1D slit fit
        describes the 2D source. Written to output/<galaxy>/instrument_2d_models/
        {idd}_{class}_2d.png.
        """
        if self.photometry_table is None or not getattr(self, 'instrument_psf', None):
            raise RuntimeError('run run_native_photometry(instrument_fit=True) first.')
        plot_bands = [b for b in filter_wavelengths if f'reff_instP_{b}' in self.photometry_table.colnames
                      and (bands is None or b in bands)]
        out_dir = os.path.join(self.base_dir, self.gal, 'instrument_2d_models')
        os.makedirs(out_dir, exist_ok=True)
        idds = range(len(self.photometry_table)) if idd is None else ([idd] if np.isscalar(idd) else idd)
        for i in idds:
            row = self.photometry_table[i]
            cc_class = str(row['PHANGS_CC_CLASS_ML']).strip() if 'PHANGS_CC_CLASS_ML' in row.colnames else ''
            setups = [(b, self._instp_band_setup(row, b)) for b in plot_bands]
            setups = [(b, g) for b, g in setups if g is not None]
            if not setups:
                continue
            fig = plt.figure(figsize=(11.5, 3.3 * len(setups) + 0.8))
            gs = GridSpec(len(setups), 3, figure=fig, left=0.06, right=0.97, top=1 - 1.1 / (3.3 * len(setups) + 0.8),
                          bottom=0.03, wspace=0.08, hspace=0.28)
            circ = np.linspace(0, 2 * np.pi, 120)
            for r_i, (band, g) in enumerate(setups):
                half = self._instp_window_arcsec(g)
                sub, ext, (xs, ys) = self._instp_cutout(g, half)
                model = self._instp_2d_model(g, xs, ys)
                norm = self._local_norm(sub, g['bkg'], g['bkg_std'],
                                        self._source_peak(sub, ext, g['r65'] if np.isfinite(g['r65']) else 0.1))
                sig = g['bkg_std'] if np.isfinite(g['bkg_std']) and g['bkg_std'] > 0 else np.nanstd(sub)
                panels = [(sub, 'data'), (model, 'instP model (2D)'),
                          (None if model is None else sub - model, 'data - model')]
                for j, (im, ttl) in enumerate(panels):
                    ax = fig.add_subplot(gs[r_i, j])
                    if im is None:
                        ax.text(0.5, 0.5, f'no instP flux\n({g["status"] or "n/a"})', ha='center', va='center',
                                transform=ax.transAxes, fontsize=9)
                    elif j < 2:
                        ax.imshow(im, origin='lower', cmap='gray_r', norm=norm, extent=ext, interpolation='nearest')
                    else:
                        ax.imshow(im, origin='lower', cmap='RdBu_r', vmin=-5 * sig, vmax=5 * sig, extent=ext,
                                  interpolation='nearest')
                    if np.isfinite(g['r65']):
                        for rad, ls in ((g['r65'], '-'), (g['ann_in'] * g['r65'], ':'), (g['ann_out'] * g['r65'], ':')):
                            ax.plot(rad * np.cos(circ), rad * np.sin(circ), ls, color='#eda100', lw=1.1)
                    ax.set_xlim(-half, half); ax.set_ylim(-half, half)
                    ax.tick_params(labelsize=6.5, labelleft=(j == 0))
                    if r_i == 0:
                        ax.set_title(ttl + ('  (±5σ sky)' if j == 2 else ''), fontsize=10)
                    if j == 0:
                        ax.set_ylabel(f'{g["instrument"]} {band}', fontsize=10, fontweight='bold')
                    if j == 2 and model is not None:
                        rr = np.hypot(*np.meshgrid(np.linspace(ext[0], ext[1], sub.shape[1] + 1)[:-1] + g['pix'] / 2,
                                                   np.linspace(ext[2], ext[3], sub.shape[0] + 1)[:-1] + g['pix'] / 2))
                        inside = (rr <= g['r65']) & np.isfinite(sub)
                        f_img = g['flux'] / g['mw'] / g['mjy_per_unit']
                        frac = np.nansum((sub - model)[inside]) / (0.65 * f_img) if f_img else np.nan
                        ax.text(0.02, 0.02, f'F = {g["flux"]:.4g} mJy, R_eff {g["reff_pc"]:.1f} pc [{g["size_source"]}]\n'
                                f'residual in R65 = {100 * frac:+.1f}% of the model light there',
                                transform=ax.transAxes, fontsize=7, va='bottom',
                                bbox=dict(fc='white', ec='none', alpha=0.85))
            fig.suptitle(f'{self.gal} — source {i} ({cc_class})   instP model vs data; orange: R65 and the '
                         f'{setups[0][1]["ann_in"]:g}-{setups[0][1]["ann_out"]:g} R65 background annulus',
                         x=0.01, ha='left', fontsize=11, y=1 - 0.25 / (3.3 * len(setups) + 0.8))
            fig.savefig(os.path.join(out_dir, f'{i}_{cc_class}_2d.png'), dpi=100, facecolor='white')
            plt.close(fig)
            if verbose:
                logger.info(f'  2D model plot saved for source {i}')

    def plot_native_slit_profiles(self, filter_wavelengths, idd=None, bands=None,
                                   n_slits=12, slit_width_px=2, slit_length_arcsec=1.0,
                                   verbose=True, plot_2d=True):
        """
        Save one diagnostic PNG per source — one row per band — showing
        (left) the native image cutout with the 12 slit diameters drawn
        through the 2-D-Gaussian-fit center, and (right) all 12 raw slit
        profiles (thin, silver — visual QA of scatter/outliers between
        directions) plus the single POINTWISE-AVERAGED profile (bold,
        crimson) and its one fitted Gaussian+Const curve (navy) — the
        measurement actually driving sigma_native_{band}/r65_native_{band}.
        Averaging first (rather than fitting each of the 12 independently,
        as an earlier design did) is what keeps this to ONE background
        level per source/band instead of 12 disagreeing ones.

        Only meaningful for aperture_source_native_{band}=='own' — a
        borrowed or PSF-fallback band has no independent slit fit by
        design (see native_phot_aux_func.measure_native_photometry_one_source).
        Those rows still always show the image (per the pipeline's own
        convention: even with no slit measurement, show what's there) with
        the aperture actually used drawn on top and a text label instead.

        Re-runs remeasure_averaged_slit_profile_at_aperture at each 'own'
        band's own stored r65_native_{band} — deterministic, so it
        reproduces exactly what run_native_photometry measured, keeping the
        raw per-slit data the production path discards by default.

        Not auto-triggered by run_native_photometry (unlike the other two
        plot_native_* methods) — opt-in; call with idd=<int> to preview a
        single source first.

        When run_instrument_photometry() has run (instG/instP columns and
        self.instrument_psf present), a third column shows the instrument-
        PSF fits for each band (see _draw_instrument_slit_panel) and their
        R65 circles are added to the image panel.

        With instP results (run_native_photometry(instrument_fit=True)) the layout
        is: (left) a cutout around the source, with a stretch from the cutout
        itself, the instP slits, the instP R65 aperture and its background
        annulus, and the Deger+22 4 px aperture + 7-8 px sky; (right) the slit
        profile (the SLIT_COMBINE of the slits) with only the instP model, ±R65,
        the instP correction factor and flux, and the Deger+22 4 px flux with the
        galaxy correction and the exact correction for this cluster's R_eff.
        plot_2d=True also writes plot_instrument_2d_models() for each source.
        Without instP results the native-only layout below is drawn.

        Written to output/<galaxy>/native_slit_profiles/{idd}_{class}_slits.png
        — one file per source with every band as a row (created if absent).

        Parameters
        ----------
        filter_wavelengths : dict  band → (central_wl_AA, R_value) — same
                             dict passed to run_native_photometry(); only
                             bands with a r65_native_{band} column are used.
        idd                 : int or None — plot only this source index
                             (into self.photometry_table); None (default)
                             plots every source.
        bands               : list[str] or None — restrict to these bands
                             (must be keys of filter_wavelengths); None
                             (default) plots every available band.
        n_slits, slit_width_px, slit_length_arcsec : must match the values
                             passed to run_native_photometry() (same
                             defaults) so the re-run reproduces the exact
                             same slit geometry that produced the stored
                             measurement.
        verbose             : bool
        """
        if self.photometry_table is None:
            raise RuntimeError(
                'No photometry table — run run_spectro_photometry() first.'
            )

        available = [b for b in filter_wavelengths
                     if f'r65_native_{b}' in self.photometry_table.colnames]
        if not available:
            raise RuntimeError(
                'No native-photometry columns found for any band in '
                'filter_wavelengths — run run_native_photometry() first.'
            )
        plot_bands = [b for b in bands if b in available] if bands is not None else available
        if not plot_bands:
            raise RuntimeError('None of the requested `bands` have native-photometry columns.')

        band_info = {}
        for band in plot_bands:
            band_lower = band.lower()
            if band_lower in self.hst_images:
                image, header, wcs = self.hst_images[band_lower]
                instrument = 'HST'
            elif band_lower in self.jwst_images:
                image, header, wcs = self.jwst_images[band_lower]
                instrument = 'JWST'
            else:
                logger.warning(f'{band}: no native image loaded — skipping slit-profile plot')
                continue
            r65_psf_arcsec = np.nan
            if getattr(self, 'native_band_calib', None) and band_lower in self.native_band_calib:
                r65_psf_arcsec = self.native_band_calib[band_lower]['r65_psf_arcsec']
            band_info[band] = dict(
                image=image, header=header, wcs=wcs, instrument=instrument,
                pixel_scale_arcsec=wcs.proj_plane_pixel_scales()[0].to('arcsec').value,
                r65_psf_arcsec=r65_psf_arcsec,
            )
        if not band_info:
            raise RuntimeError('None of the requested bands have a native image loaded.')

        def _draw_band_row(ax_img, ax_prof, bd, band, r65_arcsec, aperture_source, status,
                            plot_data, ra, dec):
            instrument = bd['instrument']
            pixel_scale_arcsec = bd['pixel_scale_arcsec']
            lo, hi = (15, 99.5) if instrument == 'HST' else (5, 98.5)

            if plot_data is None:
                # Either a borrowed/psf_fallback band (no independent slit
                # fit to show by design) or a genuine measurement failure —
                # always still show the image + whatever aperture info IS
                # available, rather than a blank box.
                px, py = skycoord_to_pixel(SkyCoord(ra=ra * u.deg, dec=dec * u.deg), bd['wcs'])
                if np.isfinite(px) and np.isfinite(py):
                    ax_img.imshow(bd['image'], origin='lower',
                                  norm=get_norm(bd['image'], lo, hi, 'asinh'), cmap='gray')
                    if np.isfinite(r65_arcsec):
                        r65_px = r65_arcsec / pixel_scale_arcsec
                        theta_full = np.linspace(0, 2 * np.pi, 100)
                        ax_img.plot(px + r65_px * np.cos(theta_full), py + r65_px * np.sin(theta_full),
                                    '-', color='deepskyblue', linewidth=1.4, zorder=5)
                        pad = r65_px * 3.0
                    else:
                        pad = max(3.0 / pixel_scale_arcsec, 20)
                    ax_img.plot(px, py, '*', color='red', markersize=8,
                                markeredgecolor='black', markeredgewidth=0.5, zorder=6)
                    ax_img.set_xlim(px - pad, px + pad)
                    ax_img.set_ylim(py - pad, py + pad)
                else:
                    ax_img.set_facecolor('black')
                ax_img.set_ylabel(f'{instrument}\n{band.upper()}', fontsize=8)
                ax_img.set_title(aperture_source or status or 'no measurement', fontsize=7.5)
                ax_img.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
                ax_prof.text(0.5, 0.5, 'no slit fit\n(' + (aperture_source or status or 'n/a') + ')',
                             ha='center', va='center', transform=ax_prof.transAxes,
                             fontsize=8, color='white')
                ax_prof.set_facecolor('black')
                ax_prof.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
                return

            diags = plot_data.diagnostics
            along = diags.along_arcsec
            slit_radius_px = float(np.max(np.abs(along))) / pixel_scale_arcsec
            pad = slit_radius_px * 1.15

            ax_img.imshow(bd['image'], origin='lower',
                          norm=get_norm(bd['image'], lo, hi, 'asinh'), cmap='gray')

            for k in range(diags.per_slit_profiles.shape[0]):
                theta = np.deg2rad(diags.theta_deg[k])
                x1 = plot_data.x0_full - slit_radius_px * np.cos(theta)
                y1 = plot_data.y0_full - slit_radius_px * np.sin(theta)
                x2 = plot_data.x0_full + slit_radius_px * np.cos(theta)
                y2 = plot_data.y0_full + slit_radius_px * np.sin(theta)
                ax_img.plot([x1, x2], [y1, y2], '-', color='silver',
                            linewidth=0.8, alpha=0.6, zorder=4)

                profile_k = diags.per_slit_profiles[k]
                valid = np.isfinite(profile_k) & (profile_k > 0)
                ax_prof.plot(along[valid], profile_k[valid], '.', color='silver',
                             markersize=2, alpha=0.4, zorder=1)

            valid_avg = np.isfinite(diags.averaged_profile) & (diags.averaged_profile > 0)
            ax_prof.plot(along[valid_avg], diags.averaged_profile[valid_avg], 'o',
                         color='crimson', markersize=3.5, alpha=0.9, zorder=3,
                         label='averaged profile')
            valid_fit = np.isfinite(diags.fitted_curve) & (diags.fitted_curve > 0)
            if np.any(valid_fit):
                ax_prof.plot(along[valid_fit], diags.fitted_curve[valid_fit], '-',
                             color='navy', linewidth=1.6, zorder=4, label='Gaussian+Const fit')

            ax_img.plot(plot_data.x0_full, plot_data.y0_full, '*', color='crimson',
                        markersize=8, markeredgecolor='black', markeredgewidth=0.5, zorder=6)
            if np.isfinite(r65_arcsec):
                r65_px = r65_arcsec / pixel_scale_arcsec
                theta_full = np.linspace(0, 2 * np.pi, 100)
                ax_img.plot(plot_data.x0_full + r65_px * np.cos(theta_full),
                            plot_data.y0_full + r65_px * np.sin(theta_full),
                            '--', color='deepskyblue', linewidth=1.2, alpha=0.9, zorder=5)
            ax_img.set_xlim(plot_data.x0_full - pad, plot_data.x0_full + pad)
            ax_img.set_ylim(plot_data.y0_full - pad, plot_data.y0_full + pad)
            ax_img.set_ylabel(f'{instrument}\n{band.upper()}', fontsize=8)
            r65_label = f'{r65_arcsec:.3f}"' if np.isfinite(r65_arcsec) else 'n/a'
            ax_img.set_title(f'R65={r65_label}   {status}', fontsize=7.5)
            ax_img.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

            ax_prof.set_yscale('log')
            ax_prof.axvline(0, color='grey', linewidth=0.6, alpha=0.5, zorder=0)
            ax_prof.set_xlabel('Radius [arcsec]', fontsize=7.5)
            ax_prof.set_ylabel('Flux', fontsize=7.5)
            sigma_label = f'{plot_data.sigma_arcsec:.3f}"' if np.isfinite(plot_data.sigma_arcsec) else 'n/a'
            ax_prof.set_title(rf'$\sigma$={sigma_label}', fontsize=7.5)
            ax_prof.tick_params(labelsize=6.5)
            ax_prof.legend(fontsize=6, loc='upper right')

        out_dir = os.path.join(self.base_dir, self.gal, 'native_slit_profiles')
        os.makedirs(out_dir, exist_ok=True)

        idds = range(len(self.photometry_table)) if idd is None else [idd]
        n_total = len(self.photometry_table)
        n_bands = len(band_info)
        # third column with the instrument-PSF fits when run_instrument_photometry ran
        with_inst = any(f'reff_instP_{b}' in self.photometry_table.colnames for b in band_info)

        if with_inst and getattr(self, 'instrument_psf', None):
            # instP layout: cutout with the slits | slit profile with the instP fit,
            # R65, corr factor and the Deger+22 4 px comparison; plus the 2D plot
            for i in idds:
                row = self.photometry_table[i]
                cc_class = (str(row['PHANGS_CC_CLASS_ML']).strip()
                            if 'PHANGS_CC_CLASS_ML' in row.colnames else '')
                setups = [(b, self._instp_band_setup(row, b)) for b in band_info]
                setups = [(b, g) for b, g in setups if g is not None]
                if not setups:
                    continue
                height = 3.2 * len(setups) + 0.8
                fig = plt.figure(figsize=(14, height))
                gs = GridSpec(len(setups), 2, figure=fig, left=0.05, right=0.985, top=1 - 1.1 / height,
                              bottom=0.035, wspace=0.12, hspace=0.42, width_ratios=[1, 2.6])
                for r_i, (band, g) in enumerate(setups):
                    ax_img, ax_prof = fig.add_subplot(gs[r_i, 0]), fig.add_subplot(gs[r_i, 1])
                    try:
                        self._draw_instp_slit_row(ax_img, ax_prof, row, band, g)
                    except Exception as e:
                        logger.debug(f'instP slit row {band} failed: {e}')
                        ax_prof.text(0.5, 0.5, f'plot failed: {e}', ha='center', va='center',
                                     transform=ax_prof.transAxes, fontsize=8)
                fig.suptitle(f'{self.gal} — source {i} ({cc_class}, RA={float(row["PHANGS_RA"]):.5f}, '
                             f'Dec={float(row["PHANGS_DEC"]):.5f})   slits: {SLIT_COMBINE} of '
                             f'{setups[0][1]["n_slits"]};  purple: instP R65 + background annulus;  orange: '
                             f'Deger+22 4 px;  blue: its 7-8 px sky', x=0.01, ha='left', fontsize=10.5,
                             y=1 - 0.25 / height)
                fig.savefig(os.path.join(out_dir, f'{i}_{cc_class}_slits.png'), dpi=100, facecolor='white')
                plt.close(fig)
                if plot_2d:
                    self.plot_instrument_2d_models(filter_wavelengths, idd=i, bands=list(band_info))
                if verbose:
                    logger.info(f'  [{i + 1:>4d}/{n_total}] slit-profile + 2D plots saved')
            logger.info(f'  ✓ instP slit-profile plots saved → {out_dir}'
                        + (f'; 2D model plots → {os.path.join(self.base_dir, self.gal, "instrument_2d_models")}'
                           if plot_2d else ''))
            return


        for i in idds:
            row = self.photometry_table[i]
            ra = float(row['PHANGS_RA'])
            dec = float(row['PHANGS_DEC'])
            cc_class = str(row['PHANGS_CC_CLASS_ML']).strip() if 'PHANGS_CC_CLASS_ML' in row.colnames else ''

            ncol = 3 if with_inst else 2
            fig = plt.figure(figsize=(9.5 + 5.5 * with_inst, max(3.0, 2.6 * n_bands)))
            gs = GridSpec(n_bands, ncol, figure=fig, left=0.06 if with_inst else 0.09, right=0.98,
                          top=0.90, bottom=0.06, wspace=0.3, hspace=0.7,
                          width_ratios=[1, 1.6, 1.6][:ncol])

            for row_idx, band in enumerate(band_info):
                bd = band_info[band]
                ax_img = fig.add_subplot(gs[row_idx, 0])
                ax_prof = fig.add_subplot(gs[row_idx, 1])
                ax_inst = fig.add_subplot(gs[row_idx, 2]) if with_inst else None
                r65_arcsec = (float(row[f'r65_native_{band}'])
                              if f'r65_native_{band}' in row.colnames else np.nan)
                aperture_source = (str(row[f'aperture_source_native_{band}']).strip()
                                    if f'aperture_source_native_{band}' in row.colnames else '')
                status = (str(row[f'status_native_{band}']).strip()
                          if f'status_native_{band}' in row.colnames else '')

                plot_data = None
                if (aperture_source == 'own' and np.isfinite(r65_arcsec)
                        and np.isfinite(bd['r65_psf_arcsec'])):
                    plot_data = remeasure_averaged_slit_profile_at_aperture(
                        ra, dec, bd['image'], bd['header'], bd['wcs'],
                        bd['r65_psf_arcsec'], r65_arcsec, bd['pixel_scale_arcsec'],
                        n_slits=n_slits, slit_width_px=slit_width_px,
                        slit_length_arcsec=slit_length_arcsec,
                    )
                _draw_band_row(ax_img, ax_prof, bd, band, r65_arcsec, aperture_source, status,
                               plot_data, ra, dec)
                if ax_inst is not None:
                    try:
                        drawn = self._draw_instrument_slit_panel(ax_inst, ax_img, row, band, bd,
                                                                 n_slits=n_slits)
                    except Exception as e:
                        logger.debug(f'instrument slit panel {band} failed: {e}')
                        drawn = False
                    if not drawn:
                        ax_inst.text(0.5, 0.5, 'no instrument fit', ha='center', va='center',
                                     transform=ax_inst.transAxes, fontsize=8)
                        ax_inst.tick_params(left=False, bottom=False, labelleft=False,
                                            labelbottom=False)

            fig.suptitle(
                f'{self.gal}  —  source {i}  '
                f'({cc_class},  RA={ra:.5f}°,  Dec={dec:.5f}°)',
                fontsize=11, y=0.995,
            )
            plt.savefig(os.path.join(out_dir, f'{i}_{cc_class}_slits.png'))
            plt.close(fig)

            if verbose:
                logger.info(f'  [{i + 1:>4d}/{n_total}] slit-profile plot saved')

        logger.info(f'  ✓ Native slit-profile diagnostic plots saved → {out_dir}')