"""
Paths shared by the validation scripts, all from the repository's config.ini:

  ROOT                the repository (pipeline modules, config.ini), put on sys.path
  OUT_ROOT            where the validation outputs go: [paths] validation_output_dir
                      (default validation/output); each script writes OUT_ROOT/output_<name>/
  OUTPUT_DIR          the pipeline's per-galaxy outputs: [paths] output_dir
  CATALOG_OUTPUT_DIR  the catalog-mode outputs: [paths] catalog_output_dir (default <output_dir>_catalog)
  CAT_PATH            the PHANGS-HST DR5 cluster catalogues: [paths] cat_path
"""
import configparser
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CONFIG = os.path.join(ROOT, 'config.ini')
for _p in (HERE, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_cp = configparser.ConfigParser(inline_comment_prefixes=('#', ';'))
_cp.read(CONFIG)


def _path(key, fallback=''):
    # same rule as run_full_pipeline.load_config: relative entries are relative to config.ini
    v = _cp.get('paths', key, fallback=fallback).strip()
    return os.path.join(ROOT, v) if v and not os.path.isabs(v) else v


OUT_ROOT = _path('validation_output_dir', 'validation/output')
OUTPUT_DIR = _path('output_dir', 'output')
CATALOG_OUTPUT_DIR = _path('catalog_output_dir') or OUTPUT_DIR.rstrip('/') + '_catalog'
CAT_PATH = _path('cat_path')
