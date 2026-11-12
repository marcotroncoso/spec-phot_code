"""
Per-source image plots for a list of clusters (e.g. the ones matched with other
photometric methods), from the full-catalog run in output_catalog/ -- nothing is
refitted, everything is read from the caches and <gal>_photometry_2d.fits:

  instP slit profiles   Photometry.plot_native_slit_profiles (plot_2d=False)
                        -> output_catalog/<gal>/native_slit_profiles/<i>_<class>_slits.png
  2D models             run_2d_photometry.plot_2d_photometry, one per --modes entry
                        -> output_catalog/<gal>/photometry_2d_plots/<i>_<class>_2d<mode>.png
<i> is the row in the photometry table; index.csv in each folder maps it to
ID_PHANGS_CLUSTER.

    python plot_matched_sources.py --galaxy NGC3351 \
        --match my_matched_list.ecsv --id-col ID_PHANGS_CLUSTER
"""
import argparse
import logging
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from astropy.table import Table  # noqa: E402

import run_2d_photometry as rd  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--galaxy', required=True)
    ap.add_argument('--match', required=True, help='table with the clusters to plot')
    ap.add_argument('--id-col', default='ID_PHANGS_CLUSTER_1_1', help='its ID_PHANGS_CLUSTER column')
    ap.add_argument('--catalog-selection', default='human', choices=['machine', 'human'])
    ap.add_argument('--modes', nargs='+', default=['free', 'Gfree'], choices=list(rd.MODES))
    ap.add_argument('--no-slits', action='store_true')
    a = ap.parse_args()
    match = Table.read(os.path.abspath(a.match))      # before _load_galaxy_catalog changes the directory
    logging.basicConfig(level=logging.WARNING)

    p, bands = rd._load_galaxy_catalog(a.galaxy, a.catalog_selection)
    rd.load_2d_photometry(p, os.path.join(p.base_dir, p.gal, f'{p.gal}_photometry_2d.fits'))
    t = p.photometry_table
    ids = np.array([rd._id_key(v) for v in t['ID_PHANGS_CLUSTER']], dtype=object)
    want = [rd._id_key(v) for v in match[a.id_col]]
    idds = [int(np.flatnonzero(ids == w)[0]) for w in want if (ids == w).any()]
    missing = [w for w in want if not (ids == w).any()]
    print(f'{len(idds)} of {len(want)} clusters found in the photometry table'
          + (f'; missing IDs {missing}' if missing else ''), flush=True)

    index = Table()
    index['row'] = idds
    index['ID_PHANGS_CLUSTER'] = list(ids[idds])
    index['CLASS'] = [str(t['PHANGS_CC_CLASS_ML'][i]).strip() for i in idds]
    dirs = []
    if not a.no_slits:
        for k, i in enumerate(idds):
            p.plot_native_slit_profiles(bands, idd=i, plot_2d=False, verbose=False)
            print(f'  slits {k + 1}/{len(idds)}', flush=True)
        dirs.append(os.path.join(p.base_dir, p.gal, 'native_slit_profiles'))
    out2d = os.path.join(p.base_dir, p.gal, 'photometry_2d_plots')
    for mode in a.modes:
        rd.plot_2d_photometry(p, bands, idd=idds, mode=mode, out_dir=out2d, verbose=False)
        print(f'  2D {mode}: {len(idds)} plots', flush=True)
    dirs.append(out2d)
    for d in dirs:
        index.write(os.path.join(d, 'index.csv'), overwrite=True)
        print(f'  -> {d}', flush=True)


if __name__ == '__main__':
    main()
