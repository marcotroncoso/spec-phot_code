"""
Command-line wrapper of muse_injection.run_injection: injection test of the
cluster-only MUSE spectra of one galaxy (see muse_injection.py for the method).
Settings default to the [muse_spectra] section of the config; run after the
spectra exist:

  nice -n 10 python validation/muse_spec_injection_test.py --galaxy NGC1365-1.15asec

run_full_pipeline.py runs the same per galaxy with [run] run_sed = True and
[sed] injection = True.
"""
import argparse
import logging
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import muse_injection as mi                          # noqa: E402
import run_muse_spectra as rms                       # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--galaxy', default='NGC1365-1.15asec')
    ap.add_argument('--config', default=os.path.join(HERE, '..', 'config.ini'))
    ap.add_argument('--n-per-env', type=int, default=10)
    ap.add_argument('--gap-fwhm', type=float, help='default: [muse_spectra] gap_fwhm')
    ap.add_argument('--width-fwhm', type=float, help='default: [muse_spectra] width_fwhm')
    ap.add_argument('--ridge', type=float, help='default: [muse_spectra] ridge')
    ap.add_argument('--r-ap', type=float, default=0.0,
                    help='aperture radius [px]; 0 = equivalent radius of the ISF polygons')
    ap.add_argument('--n-workers', type=int, default=4)
    opt = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    from run_full_pipeline import load_config
    cfg = load_config(opt.config)
    settings = rms.spectra_settings(opt.config)
    for key in ('gap_fwhm', 'width_fwhm', 'ridge'):
        if getattr(opt, key) is not None:
            settings[key] = getattr(opt, key)
    mi.run_injection(opt.galaxy, cfg, settings, n_per_env=opt.n_per_env, n_workers=opt.n_workers,
                     r_ap=opt.r_ap)


if __name__ == '__main__':
    main()
