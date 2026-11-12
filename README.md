# spec-phot_code

Photometry and spectroscopy of star clusters in nearby galaxies from HST, JWST and MUSE images (built for PHANGS). The code has three parts:

1. **Isolated-cluster finder (ISF).** It selects PHANGS-HST clusters that are isolated at MUSE resolution and matches them to MUSE.
2. **Photometry** in 13 HST + JWST bands, by several methods:
   - native aperture photometry;
   - instrument-PSF profile fits along slits (instG, instP);
   - full 2D model fits.

   It runs on the ISF selection, on the whole PHANGS-HST DR5 catalogue, or on **any catalogue with any images**.
3. **MUSE cluster spectra** of the isolated clusters, by two methods: annulus subtraction (M1) and PSF fitting (M2).

The results can be exported as input tables for SED fitting with CIGALE.

---

## Contents

- [What the code measures](#what-the-code-measures)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running](#running)
  - [Check before running: dry run](#check-before-running-dry-run)
  - [A. PHANGS: isolated clusters + photometry (+ MUSE)](#a-phangs-isolated-clusters--photometry--muse)
  - [B. PHANGS: every cluster of the DR5 catalogue](#b-phangs-every-cluster-of-the-dr5-catalogue)
  - [C. Any catalogue on any images](#c-any-catalogue-on-any-images)
  - [D. 2D model photometry](#d-2d-model-photometry)
  - [E. MUSE cluster spectra](#e-muse-cluster-spectra)
  - [F. Diagnostic plots](#f-diagnostic-plots)
- [Use from Python / a notebook](#use-from-python--a-notebook)
- [Output files and columns](#output-files-and-columns)
- [Validation scripts and tests](#validation-scripts-and-tests)
- [Practical notes](#practical-notes)

---

## What the code measures

| Method | Column tag | What it does |
|---|---|---|
| **native** | `_native_` | Aperture photometry at native resolution. Each band gets its own aperture, set from a Gaussian + constant fit to the source's averaged slit profile (12 slits through the centroid). A band with no clear emission of its own borrows the aperture of another band, or uses a point-source aperture. A sky annulus is subtracted and an aperture correction applied. |
| **instG** | `_instG_` | Fits *instrument PSF ⊗ Gaussian* to the averaged slit profile, with R_eff free. The flux is measured inside the model's R65 aperture (the radius that holds 65 % of the model light), multiplied by the model's own aperture correction. |
| **instP** | `_instP_` | Same as instG, with a cluster profile chosen by class: EFF (a Moffat) for YCL/MAP and unclassified sources, King c = 1.5 for OGCC. The EFF power index γ is fixed (default 3) or fitted together with R_eff; see [Cluster profile](#g-cluster-profile-fixed-or-fitted-γ). This is the main size and flux measurement. |
| **2D** | `_f2d<mode>_` | Fits *flux × [profile ⊗ PSF] + constant* to every pixel within the fit radius, with no slits and no aperture. The flux is the model's total flux. There are four modes, `fixed`, `free`, `Gfixed` and `Gfree`. In the `fixed` modes the centre is fixed at the instP centre; in the `free` modes it is fitted. The `G` modes use a Gaussian source profile. |
| **convolved (v1)** | ISF mode only | Every band is convolved to the MUSE PSF, and the flux is measured in the ISF segmentation polygons, so the photometry matches the MUSE spectra. |

**PSFs.** By default (`[run] psf_source = theoretical`) every PSF comes from the theoretical models, and none is measured on the images:
- **Models:** STScI effective PSFs for HST WFC3/UVIS and ACS/WFC, and stpsf models for JWST NIRCam and MIRI, all stored in `data/psf/`.
- **Instrument fits** (instG, instP, 2D) use the full model images.
- **The convolution to MUSE, the native method and the ISF** use the model's FWHM and Moffat shape. These are measured on a rendered model star, with the same Moffat fit as before.
- **MUSE:** the FWHM comes from the token name (`NGC1365-1.15asec` → 1.15″).

PSFs measured on the images (`psf_source = table`, `psf_extra_blur = galaxy`) are 20–45% broader than the models. They come from the brightest compact sources, which are often clusters, and they shrink the fitted R_eff.

All fluxes are in mJy and corrected for Milky Way extinction with the SFD E(B-V). The 1σ errors come from the sky noise.

R_eff is the cluster's own half-light radius, with the PSF removed. It is given in arcsec (`reff_*`) and in pc (`reff_pc_*`), and is capped at 20 pc. Sources fitted as unresolved get R_eff = 0.

## Repository layout

```
run_full_pipeline.py        PHANGS batch driver: ISF + photometry (A), or the DR5 catalogue (B)
run_catalog_photometry.py   any catalogue + any images (C)
run_2d_photometry.py        2D model fits (D); fit2d.py holds the fitter
run_muse_spectra.py         MUSE cluster spectra (E); muse_spec_aux_func.py holds the methods
muse_injection.py           injection test of the MUSE spectra (systematic errors for the SEDs)
muse_sed.py                 SED plots: photometry + MUSE spectra
preflight.py                dry run (--dry-run): checks inputs, settings and resources
plot_matched_sources.py     per-source slit-profile and 2D plots for a list of clusters (F)
plot_qa_diagnostics.py      per-galaxy QA figures
plot_r65_distributions.py   aperture-size distributions
gamma_ensemble.py           sample gamma from a free-gamma run (G)
cigale_export.py            CIGALE input tables

analysis.py                 classes IsolatedSourceFinder and Photometry (the whole workflow)
native_phot_aux_func.py     native + instrument-PSF slit photometry
instrument_psf.py           HST / JWST instrument PSF models and rendering
phot_aux_func.py, psf_aux_func.py, aux_functions.py   helpers

config.example.ini          template for the PHANGS runs (copy to config.ini)
catalog_run.example.ini     template for run_catalog_photometry.py
data/psf_table.csv          measured PSF FWHM per galaxy and band (only used with psf_source = table)
data/psf/                   HST effective PSFs (pickles) and JWST stpsf PSFs (FITS)
data/filters/               HST transmission curves used by the MUSE synthetic photometry
validation/                 mocks, injection tests and method comparisons
tests/                      pytest suite
```

The modules import each other by plain name, such as `from analysis import Photometry`. Run the scripts from the repository folder, or add it to `sys.path`.

## Installation

The code was tested with Python 3.11 and the versions listed in `requirements.txt`.

```bash
conda create -n specphot python=3.11
conda activate specphot
pip install -r requirements.txt
```

The Milky Way extinction uses the SFD dust map from `dustmaps`. Download the map once:

```bash
python -c "from dustmaps.config import config; config['data_dir'] = '/path/to/dustmaps_data'; \
import dustmaps.sfd; dustmaps.sfd.fetch()"
```

Without the map, set `ebv =` in the config of a custom run (C).

## Configuration

The PHANGS runs (A, B, D, E) read `config.ini`. Make it from the template:

```bash
cp config.example.ini config.ini     # then edit the paths
```

| Section | Key | Meaning |
|---|---|---|
| `[paths]` | `hst_root` | HST anchor images, `<hst_root>/<GALAXY>/*_anchor.fits` |
| | `jwst_root` | JWST v4p1 images, `<jwst_root>/<galaxy>/images/*.fits` |
| | `muse_dap_root` | MUSE DAP products, `<root>/<TOKEN>/<TOKEN>_DR2.2v2/<TOKEN>_MAPS.fits` and the cube `<root>/<TOKEN>/<TOKEN>.fits` |
| | `cat_path` | PHANGS-HST DR5 cluster catalogues |
| | `psf_filepath` | PSF FWHM table (default `data/psf_table.csv`); only read with `psf_source = table` |
| | `output_dir` | Per-galaxy results go to `<output_dir>/<GALAXY>/` |
| | `catalog_output_dir` | Output root of mode B (blank means `<output_dir>_catalog`) |
| | `validation_output_dir` | Output root of the `validation/` scripts |
| `[galaxies]` | `list` | MUSE tokens such as `NGC1365-1.15asec, NGC4254-0.89asec`, or `auto` |
| `[run]` | `run_isf` | Run the ISF when its products are missing |
| | `redo` | Steps to recompute even when up to date: any of `isf, photometry, 2d, spectra, injection, sed, plots`, or `all`. Blank means only what is needed. |
| | `force_recompute` | Old switch; `True` is the same as `redo = all` |
| | `plot_diagnostics` | Write per-source diagnostic PNGs (slow) |
| | `instrument_fit` | Also run instG and instP |
| | `psf_source` | `theoretical` (default): every PSF from the theoretical models, and MUSE from the token name. `table`: `psf_table.csv`, measuring missing bands (the old behaviour) |
| | `psf_extra_blur` | Blur on top of the theoretical PSF: `none` (default), `galaxy` (match the measured FWHM), `default` (sample median), or a FWHM in arcsec |
| | `source_catalog` | `isf` (mode A) or `catalog` (mode B) |
| | `catalog_selection` | `machine` or `human` classification (mode B) |
| | `catalog_classes` | `1+2`, or `1+2, 3` (mode B) |
| | `catalog_n_workers` | Processes per galaxy (mode B) |
| | `n_workers` | Galaxies processed at once (mode A) |
| `[profile]` | `eff_gamma` | EFF power index γ when fixed (default 3.0) |
| | `eff_gamma_free` | `True`: fit γ for every source and band (G) |
| | `eff_gamma_min`, `eff_gamma_max` | Range of the fitted γ (default 2.0–6.0; at least 1.8) |
| | `ogcc_profile`, `king_c` | OGCC profile: `king` with concentration `king_c` (default 1.5), or `eff` |
| `[cigale]` | `export`, `ext`, `h0` | CIGALE table per galaxy |

Relative paths in `[paths]` are relative to the config file. The `[profile]` section works the same in the mode C run config.

## Running

### Check before running: dry run

Add `--dry-run` to any run. It checks that everything the run needs is in place, then stops. It measures nothing and writes nothing.

```bash
python run_full_pipeline.py --config config.ini --dry-run                    # mode A, every galaxy
python run_full_pipeline.py --config config.ini --catalog-mode --dry-run     # mode B
python run_full_pipeline.py --config config.ini --dry-run --galaxy NGC3351-1.05asec
python run_catalog_photometry.py --config my_run.ini --dry-run               # mode C
```

It takes seconds to a minute, because it reads only headers and catalogues. It checks:
- the config values, the Python packages, and the PSF models and filter curves in `data/`;
- the SFD dust map, and the galaxy-name lookup used for E(B-V);
- the input folders and catalogues;
- for every galaxy, its HST and JWST images: header, units and PSF model;
- in mode A, the MUSE products, and the cube if `run_spectra` is on;
- in modes B and C, the number of catalogue sources on valid data of every image. These are the sources that will be measured.

A **Measurements** section lists the photometry the config asks for: native, instG and instP (with the profile used for each class), each 2D mode, the convolved photometry, spectra and SEDs in mode A, plus plots and the CIGALE table. It also says where the column tag of each method is, and warns when no class column is given in mode C.

It also lists what each galaxy would **compute or reuse** (ISF, photometry, 2D modes, spectra, injection test, SEDs), using the same tests as the run itself. Finally it checks the output folder, free disk space, CPU load, memory and BLAS threads against the number of processes the run will start.

Every line is ✓ (ok), ⚠ (the run works, but check this) or ✗ (the run will fail or skip a galaxy). The exit code is 1 when there is any ✗, so a script can test it:

```bash
python run_full_pipeline.py --config config.ini --dry-run && python run_full_pipeline.py --config config.ini
```

### A. PHANGS: isolated clusters + photometry (+ MUSE)

```bash
python run_full_pipeline.py --config config.ini                       # every galaxy in [galaxies] list
python run_full_pipeline.py --config config.ini --galaxy NGC1365-1.15asec
```

For each galaxy this runs, in order:
1. the ISF (`IsolatedSourceFinder.run_full_pipeline`): HST and MUSE PSFs, convolution, isolated-source detection, match with the DR5 catalogue, second isolation filter;
2. the convolved photometry and the MUSE aperture spectra (`<GAL>_photometry.fits`);
3. the native and instrument photometry (`<GAL>_photometry_v2.fits`);
4. the slit-profile plots, and the instP model drawn in 2D (`instrument_2d_models/`; this is a drawing of the slit fit, not a 2D fit);
5. *optional,* `[run] run_2d = True`: the 2D model fits of mode D (`<GAL>_photometry_2d.fits`, merged into v2) and, with `[2d] plots = True`, their plots;
6. the CIGALE table;
7. *optional,* `[run] run_spectra = True`: the MUSE cluster spectra of mode E;
8. *optional,* `[run] run_sed = True`: the SED plots of every cluster, combining the photometry with the MUSE spectra. Before them, the galaxy's injection test runs (`[sed] injection`), whose scatter gives the systematic error bars.

After a batch run, the pooled QA plots of every galaxy in the output folder are made (`[qa] run_qa`, see [Everything in one run](#everything-in-one-run)).

#### Everything in one run

Every step and every setting comes from `config.ini`:

| Setting | Step |
|---|---|
| `[run] run_isf` | ISF when its products are missing |
| `[run] run_2d` and the `[2d]` section | 2D model fits: `modes`, `n_workers`, `plots`, `all_bands`, `overwrite` |
| `[run] run_spectra` and the `[muse_spectra]` section | MUSE M1/M2 spectra (ISF mode only); `overwrite` |
| `[run] run_sed` and the `[sed]` section | injection test (`injection`, `n_per_env`, `n_workers`) and SED plots (`overwrite`) |
| `[qa]` section | pooled QA plots after a batch run: `run_qa`, `qa_dir`, `plots` |

- **The pipeline remembers what it has done.** Rerunning the same config only redoes what is missing or out of date:

  | Step | Reused when | Recomputed when |
  |---|---|---|
  | ISF | `<GAL>_matched.fits` exists | it is missing, or it was made with another `psf_source` or MUSE `sample` (from its header) |
  | Convolved images | made with the same PSFs | the PSF FWHMs changed |
  | v1, native, instP photometry | caches exist, are newer than the ISF products, and were made with the same PSF, profile and slit settings | otherwise; also whenever the ISF was rerun |
  | Per-source plots (slits, instP 2D drawings, native diagnostics, 2D plots) | one PNG per source, newer than the results they show | the photometry they show changed, or a source's PNG is missing |
  | 2D fits | `<GAL>_photometry_2d.fits` has all the modes and is newer than the instP photometry | otherwise |
  | MUSE spectra | `<GAL>_muse_spectra.fits` is newer than `<GAL>_matched.fits` | otherwise |
  | Injection test | its results are newer than the spectra | otherwise |
  | SED plots | all of them are newer than the spectra, the native, instP and 2D results and the injection test | otherwise |
  | CIGALE table, QA plots | — | every run (they are cheap) |

  To redo something anyway, list it in `[run] redo` (`isf, photometry, 2d, spectra, injection, sed, plots`, or `all`), e.g. `redo = photometry, plots`. Each step logs whether it was reused or computed, and the final summary prints one `steps:` line per galaxy.
- **A failed optional step** (2D or spectra) is logged and reported at the end. The galaxy's photometry still counts as done.
- **QA plots:**
  - They are made only at the end of a batch run.
  - Single-galaxy runs (`--galaxy`, e.g. a Jupyter loop) skip them, so parallel runs don't overwrite each other's plots. When all galaxies are done, run `python run_full_pipeline.py --config config.ini --qa-only`.
  - `--no-qa` turns them off for one batch run.
- **Workers:**
  - `--n-workers` (or `[run] n_workers`) sets how many galaxies run at once.
  - Inside each galaxy, the 2D fits and the spectra start their own `[2d] n_workers` and `[muse_spectra] n_workers` processes. The total is about galaxies × those values, so size them together.

### B. PHANGS: every cluster of the DR5 catalogue

```bash
python run_full_pipeline.py --config config.ini --catalog-mode --galaxy NGC3351-1.05asec   # one galaxy
python run_full_pipeline.py --config config.ini --catalog-mode --n-workers 3              # every galaxy in [galaxies] list
```

In mode B each galaxy starts `catalog_n_workers` processes, plus `[2d] n_workers` when `run_2d` is on. A batch therefore uses about `--n-workers` × that many CPUs.

Mode B runs the native and instrument photometry for every catalogue cluster that falls on all HST and JWST images, whether it is isolated or not. MUSE is not used. The sources are selected with `catalog_selection`, `catalog_classes` and `catalog_drop_outside`. Results go to `<catalog_output_dir>/<GAL>/`, with the same file names as mode A.

The caches are tied to the selection. A run with a different `catalog_selection` recomputes everything and overwrites the earlier caches.

### C. Any catalogue on any images

Use this mode for your own source list: a different catalogue, a subsample, or a matched list. It also works for images that do not follow the PHANGS layout. It needs no ISF, no DR5 catalogue and no MUSE.

```bash
cp catalog_run.example.ini my_run.ini      # edit it
python run_catalog_photometry.py --config my_run.ini
python run_catalog_photometry.py --config my_run.ini --max-sources 5     # quick test
```

`my_run.ini` has three sections:

```ini
[catalog]
file = my_clusters.ecsv        # FITS / ECSV / CSV ...
name = NGC3351                 # output folder + PSF-table row
ra_col = RA                    # degrees or sexagesimal
dec_col = DEC
id_col = ID                    # optional
class_col =                    # optional: YCL/MAP -> EFF, OGCC -> King, else EFF
distance_mpc = 9.96            # or dist_col = <column>
ebv =                          # optional; blank = SFD at the catalogue centre

[images]
F555W = /data/ngc3351_f555w.fits          # BAND = path[, FITS extension]
F200W = /data/ngc3351_f200w_i2d.fits, SCI
hst_root =                                 # or: find PHANGS-layout images of <name>
jwst_root =

[run]
output_dir = output_custom
n_workers = 4
instrument_fit = True
require_all_bands = True       # keep only sources on every image
modes_2d = free Gfree          # optional 2D fits
plots = all                    # none | slits | 2d | all
```

**Image requirements.** These are the units the photometry assumes:
- **HST:** flux in **Jy per pixel**, which is the PHANGS-HST anchor-image convention. The header must have `INSTRUME` set to WFC3 or ACS, which selects the PSF.
- **JWST:** **MJy/sr** with `PIXAR_SR` in the header, or DN/s with both `PHOTMJSR` and `PIXAR_SR`.
- **Bands:** F275W, F336W, F438W, F555W, F814W, F200W, F300M, F335M, F360M, F770W, F1000W, F1130W and F2100W. These are the bands with PSF models in `data/psf/`.

The PSFs are the theoretical ones (`psf_source = theoretical`), so nothing is measured and any `name` works. With `psf_source = table`, the PSF FWHM of each band of a new `name` is measured from the image and added to `psf_table.csv`. Results go to `<output_dir>/<name>/`, with the same files and columns as modes A and B. The run config is copied there as `run_config.ini`.

### D. 2D model photometry

On a PHANGS run (A or B), after the photometry:

```bash
python run_2d_photometry.py --galaxy NGC3351-1.05asec --catalog-mode --catalog-selection human \
       --modes fixed free Gfixed Gfree --workers 16 --save-v2 [--plot]
```

- The results are saved to `<GAL>_photometry_2d.fits` with every 2D field. Modes run separately accumulate in the same file.
- `--save-v2` also copies the 2D fluxes and errors into `<GAL>_photometry_v2.fits`.
- `--plot` writes one PNG per source to `photometry_2d_plots/`.

In mode C, set `modes_2d` in the run config instead.

### E. MUSE cluster spectra

This needs the ISF products (mode A) and the MUSE cube. Run it as part of mode A with `[run] run_spectra = True`, or on its own:

```bash
nice -n 10 python run_muse_spectra.py --galaxy NGC1365-1.15asec [--ids 1256,1301] [--no-plots]
```

It reads the same `config.ini` as the pipeline:
- the paths come from `[paths]` (`output_dir`, `muse_dap_root`, `cat_path`);
- the settings come from `[muse_spectra]`.

| `[muse_spectra]` key | Meaning | Default |
|---|---|---|
| `aperture` | ISF region used as the aperture (`MUSE` or `HST`) | `MUSE` |
| `wave_min`, `wave_max` | observed wavelength range used [Å] | 4850, 7000 |
| `gap_fwhm`, `width_fwhm` | M1 annulus gap and width, in PSF FWHM | 1.0, 2.0 |
| `smooth` | M1: spline annulus continuum instead of interpolating over the lines | False |
| `fit_half` | M2 stamp half-size [MUSE px] | 20 |
| `ridge` | M2 ridge strength on the background patches | 0.01 |
| `n_workers` | parallel processes | 4 |
| `plots` | write the figures | True |
| `overwrite` | in the pipeline (`run_spectra`), recompute even when the spectra are up to date | False |

Command-line flags override these values: `--aperture`, `--wave-range MIN MAX`, `--gap-fwhm`, `--width-fwhm`, `--smooth/--no-smooth`, `--fit-half`, `--ridge`, `--n-workers` and `--plots/--no-plots`. `--galaxy` is given on the command line only.

- **M1:** sums the cube inside the ISF polygon, then subtracts the continuum of an annulus.
- **M2:** fits PSF(cluster) + PSF(neighbours) + a smooth background in every wavelength slice. This is in the style of pampelmuse.

Output: `<GAL>_muse_spectra.fits` and the figures in `muse_spectra/`.

If too few clusters give a usable PSF fit (as happened for NGC3627), the PSF is a Moffat with the FWHM from the token name, flat with wavelength. The log says so, and the FITS header has `PSFFALLB = True`.

**SED plots.** These combine the HST + JWST photometry (native, instP, 2D, catalogue) with the M1 and M2 spectra and their synthetic F555W flux. They come from `muse_sed.py`, run as part of the pipeline with `[run] run_sed`, or on their own:

```bash
python validation/muse_spec_injection_test.py --galaxy NGC3351-1.05asec     # systematic errors (muse_injection.py)
python validation/muse_sed_example.py --galaxy NGC3351-1.05asec [--id 1256 ...]
```

The error bars of the MUSE points are statistical ⊕ the scatter of that galaxy's injection test. Without an injection test they are statistical only, and the plot says so. The plots are written to `<GAL>/muse_spectra/sed/<row>_<class>_sed.png`, together with `<GAL>_sed_ratios.csv`.

### F. Diagnostic plots

- **Per-source plots for a list of clusters:** the instP slit profiles, and the 2D model and residual images.

  ```bash
  python plot_matched_sources.py --galaxy NGC3351 --match my_list.ecsv --id-col ID_PHANGS_CLUSTER --modes free Gfree
  ```

  Output: `native_slit_profiles/<row>_<class>_slits.png` and `photometry_2d_plots/<row>_<class>_2d<mode>.png`. An `index.csv` maps `<row>` to the cluster ID.
- **Pooled QA of all galaxies:**
  - `python plot_qa_diagnostics.py [--output-dir …] [--plots …]`, which is the same as `run_full_pipeline.py --qa-only`;
  - `python plot_r65_distributions.py`.

  The plots include the photometry ratios to the catalogue and between methods, colour–colour diagrams, instP sizes, detection and recentring statistics, and the MUSE synthetic photometry against the imaging (`spectra_vs_photometry.png`, if spectra exist). The flux plots have 1σ error bars.

### G. Cluster profile: fixed or fitted γ

The instP and 2D fits model a YCL/MAP cluster as an EFF profile (Elson, Fall & Freeman 1987), convolved with the PSF:

    Σ(r) ∝ (1 + r²/a²)^(−γ/2)        (a Moffat with β = γ/2)

The core radius *a* is always fitted, through R_eff. The power index γ sets how much light sits in the wings. It is fixed at `eff_gamma = 3` unless `[profile] eff_gamma_free = True`; then it is fitted together with R_eff for each source and band, within `[eff_gamma_min, eff_gamma_max]`.

- **Search:** a grid over R_eff and γ (γ steps of 0.5), a refinement of R_eff at the best γ and its two neighbours, then a parabola in γ. γ is reported on a 0.05 grid.
- **Cost:** about 8× slower than fixed γ (on the order of 13 s per source and band for HST).
- **Columns:** `gamma_instP_<B>` and `gamma_f2d<mode>_<B>`. γ is also applied to the aperture correction, and a band that borrows its size borrows the γ too. `profile_instP` reads `eff2.0-6.0` for a free-γ run.
- **When γ is unconstrained:** with R_eff = 0 it is NaN. For R_eff well below the PSF, or at low S/N, it is poorly constrained.
- **Caches:** changing `[profile]` makes the next run recompute the instrument photometry, because the cache header records the profile.

To get one γ for a whole sample, first run with γ free. Then run:

```bash
python gamma_ensemble.py output_custom/NGC3351/NGC3351_photometry_instrument.fits --min-snr 20 --plot gamma.png
```

This keeps only well-constrained fits: own, converged, R_eff > 0, S/N above the cut, and γ not at the range edge. It prints the percentiles of γ per band, over all bands, and per source. Put the median back as `eff_gamma` with `eff_gamma_free = False`, and rerun.

## Use from Python / a notebook

```python
import sys
sys.path.insert(0, '/path/to/spec-phot_code')
from analysis import Photometry
import run_full_pipeline as rfp

bands = {b: v for b, v in rfp.FILTER_WAVELENGTHS.items() if b in ('F555W', 'F814W', 'F200W')}
p = Photometry(['NGC3351', 'NGC3351'], base_dir='output_custom', psf_filepath='data/psf_table.csv')
p.load_images({'F555W': 'f555w.fits', 'F814W': 'f814w.fits'}, {'F200W': 'f200w.fits'})
p.load_custom_catalog('my_clusters.ecsv', ra_col='RA', dec_col='DEC', id_col='ID', distance_mpc=9.96)
p.filter_by_common_fov(plot=False, include_muse=False)
p.load_or_measure_psf()
p.eb_v = 0.0277
p.init_photometry_table_from_catalog()
p.measure_native_psf_shape()
p.run_native_photometry(bands, n_workers=4, instrument_fit=True, plot_diagnostics=False)
# fitted gamma instead: profile_by_class=profile_by_class_from_settings(eff_gamma_free=True)
#   (from native_phot_aux_func import profile_by_class_from_settings)
p.save_photometry(suffix='photometry_v2')
t = p.photometry_table                       # astropy Table, one row per source

p.plot_native_slit_profiles(bands, idd=0, plot_2d=False)      # slit-profile plot of source 0
```

`Photometry.load_images` reads HST images from the primary HDU and JWST images from extension 1. For other layouts, see `load_images` in `run_catalog_photometry.py`.

## Output files and columns

The files are written per galaxy or field, in `<output_dir>/<name>/`:

| File | Content |
|---|---|
| `<name>_photometry_v2.fits` | **Main table:** catalogue columns plus every native, instG, instP and 2D flux |
| `<name>_photometry_native.fits`, `_instrument.fits` | Caches of the native and instrument runs. They are reused while up to date, unless `redo = photometry`. |
| `<name>_photometry_2d.fits` | All 2D-fit fields |
| `<name>_photometry.fits` | Mode A only: convolved photometry |
| `*_apertures_<band>.reg` | DS9 regions of the apertures used |
| `native_slit_profiles/`, `photometry_2d_plots/` | Per-source plots |
| `pipeline.log` | Log of the run |

Main columns, where `<B>` is a band such as `F555W`:

| Column | Unit | Meaning |
|---|---|---|
| `ID_PHANGS_CLUSTER`, `PHANGS_RA`, `PHANGS_DEC` | , deg | Source ID and position. In mode C these are copied from your columns. |
| `PHANGS_CC_CLASS_ML`, `PHANGS_DMPC` | , Mpc | Class, which selects the profile, and distance |
| `flux_native_<B>`, `flux_instG_<B>`, `flux_instP_<B>` (+ `_err`) | mJy | Total flux, Milky Way dereddened |
| `reff_instP_<B>`, `reff_pc_instP_<B>` | arcsec, pc | Half-light radius with the PSF removed |
| `size_source_instP_<B>` | | `own` (fitted in this band), `borrowed:<band>` or `point_source` |
| `snr_instP_<B>` | | Aperture S/N = flux / (σ_sky √N_pix) |
| `r65_instP_<B>`, `corr_factor_instP_<B>` | arcsec, | Aperture radius and aperture correction |
| `gamma_instP_<B>` | | EFF power index of the instP model: fitted, borrowed, or the fixed value. NaN for King and for R_eff = 0. |
| `profile_instP` | | `eff3.0` (fixed γ), `eff2.0-6.0` (γ fitted in that range) or `king1.5` |
| `flux_f2d<mode>_<B>` (+ `_err`) | mJy | 2D-fit total flux |
| `reff_pc_f2d<mode>_<B>`, `gamma_…`, `chi2red_…`, `shift_px_…`, `converged_…` | | 2D-fit size, γ and quality (in `_photometry_2d.fits`) |

Bands where no emission was found have NaN for that method.

## Validation scripts and tests

`validation/` holds the scripts used to test the methods:
- mock clusters with neighbours (`mock_far_neighbours.py`, `mock_2d_far.py`);
- injection tests (`injection_test.py`, `muse_spec_injection_test.py`);
- aperture-correction comparisons with Deger et al. 2022 (`deger_*.py`);
- method-versus-catalogue comparisons (`all_methods_vs_catalog.py`);
- size versus S/N (`size_vs_snr.py`);
- instP R_eff against S/N and (instP − catalogue)/catalogue for one galaxy's output folder (`plot_reff_snr_catalog.py --dir <output>/<GALAXY>`).

They write to `<validation_output_dir>/output_<name>/`. Run them from the repository folder, for example `python validation/size_vs_snr.py --galaxy NGC3351 --catalog-mode`.

To run the unit and integration tests, which use synthetic data only (about 5 min):

```bash
python -m pytest -q          # pytest.ini limits the run to tests/
```

Two tests in `tests/test_native_photometry.py::TestMeasureAveragedSlitProfile` currently fail; all the others pass.
- `test_collect_diagnostics_matches_result_and_reproduces_profile` still expects the mean of the slits, but the code now takes the median (`SLIT_COMBINE = 'median'`).
- `test_scale_invariant` finds that the fitted width changes by more than 1e-3 between unit-scale and Jy-scale images. This has not been investigated yet.

## Practical notes

- **Shared machines.** Run with `nice -n 10`, and set `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` when you use several workers. Check the load (`uptime`) before a large run.
- **Memory.** A PHANGS galaxy with 13 bands plus MUSE needs several GB. `run_full_pipeline.py` runs each galaxy in its own process and logs the peak memory.
- **Distances.** R_eff in pc and the 20 pc cap need a distance. Without one, sizes are given in arcsec only, and the cap becomes 0.5″.
- **Completeness.** The photometry measures the sources you give it. It does not detect sources, so the S/N range reflects the input catalogue's selection.
