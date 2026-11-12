"""
Command-line wrapper of muse_sed.make_galaxy_seds: SED plots (HST + JWST
photometry and the MUSE M1 / M2 spectra) of the isolated clusters of one
galaxy. See muse_sed.py for what is drawn and how the errors are built.

  python validation/muse_sed_example.py --galaxy NGC1365-1.15asec --id 1256
  python validation/muse_sed_example.py --galaxy NGC3351-1.05asec --all

Plots go to <output_dir>/<GAL>/muse_spectra/sed/<row>_<class>_sed.png.
run_full_pipeline.py makes them for every galaxy with [run] run_sed = True.
"""
import argparse
import logging
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import muse_sed                                       # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--galaxy', default='NGC1365-1.15asec', help='MUSE token')
    ap.add_argument('--config', default=os.path.join(HERE, '..', 'config.ini'))
    ap.add_argument('--id', type=int, nargs='*', default=None, help='only these ID_PHANGS_CLUSTER')
    ap.add_argument('--all', action='store_true', help='every source with photometry and spectra (default '
                                                       'when no --id is given)')
    opt = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    from run_full_pipeline import load_config
    cfg = load_config(opt.config)
    res = muse_sed.make_galaxy_seds(opt.galaxy, cfg, ids=None if (opt.all or not opt.id) else opt.id)
    for r in res['rows']:
        print(f'#{r["row"]:>2} ID {r["ID"]:>5} {r["class"]:<4} {r["env"]:<4} N_ap {r["N_ap"]:5.0f}  '
              f'F555W M2/instP {r["M2_instP_F555W"]:.2f}  M2/2D {r["M2_2D_F555W"]:.2f}  '
              f'M2/2DG {r["M2_2DG_F555W"]:.2f}  '
              f'(M1/EE)/native {r["M1EE_native_F555W"]:.2f}  M2/cat {r["M2_cat_F555W"]:.2f}')


if __name__ == '__main__':
    main()
