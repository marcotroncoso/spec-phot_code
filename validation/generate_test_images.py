#!/usr/bin/env python
"""
CLI driver: generate the full set of 16 synthetic PSF test FITS images
(point source + star cluster, each at 8 PSF conditions) into an output
directory.

Usage:
    python generate_test_images.py
    python generate_test_images.py --output-dir output --pixel-scale 0.04
"""
import argparse
import os

from synthetic_psf_tests import generate_all_test_images, DEFAULT_FWHM_GRID_ARCSEC
from _paths import OUT_ROOT  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', default=os.path.join(OUT_ROOT, 'output'),
                    help='Directory to write FITS files into (default: <validation_output_dir>/output)')
    p.add_argument('--pixel-scale', type=float, default=0.04,
                    dest='pixel_scale_arcsec',
                    help='Pixel scale in arcsec/pixel (default: 0.04, WFC3/UVIS-like)')
    p.add_argument('--npix', type=int, default=401,
                    help='Image stamp size in pixels, must be odd (default: 401)')
    p.add_argument('--total-flux', type=float, default=1.0,
                    help='Total injected flux, arbitrary units (default: 1.0)')
    p.add_argument('--seed', type=int, default=20260911, dest='cluster_seed',
                    help='RNG seed for cluster source positions')
    p.add_argument('--noise-std', type=float, default=0.0,
                    help='Gaussian noise sigma to add (default: 0.0, noiseless)')
    p.add_argument('--bkg-level', type=float, default=0.0,
                    help='Constant background level to add (default: 0.0)')
    return p.parse_args()


def main():
    args = parse_args()
    paths = generate_all_test_images(
        output_dir=args.output_dir,
        pixel_scale_arcsec=args.pixel_scale_arcsec,
        npix=args.npix,
        total_flux=args.total_flux,
        fwhm_grid_arcsec=DEFAULT_FWHM_GRID_ARCSEC,
        cluster_seed=args.cluster_seed,
        noise_std=args.noise_std,
        bkg_level=args.bkg_level,
    )
    print(f"\nWrote {len(paths)} FITS files to {args.output_dir}/")


if __name__ == '__main__':
    main()
