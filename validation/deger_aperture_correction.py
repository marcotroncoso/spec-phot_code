r"""
The PHANGS-HST aperture correction (Deger et al. 2022, MNRAS 510, 32;
Section 4): the math, where every equation comes from, and the functions
our tests use.

Running this file (python deger_aperture_correction.py) prints the numbers
and checks the recipe against the NGC1365 catalog and our mocks.

======================================================================
1. Magnitudes and fluxes
======================================================================
A magnitude is a logarithmic flux:

        m = -2.5 log10(F) + ZP                                        (1)

with ZP the zero point of the photometric system. For AB magnitudes
ZP = -2.5 log10(3631 Jy), i.e. m_AB = -2.5 log10(F_nu / 3631 Jy). For
Vega magnitudes, which the PHANGS catalogs list as PHANGS_<band>_VEGA,
ZP = 2.5 log10(F_nu,Vega), with F_nu,Vega the flux density of Vega in
that filter:

        F_nu = F_nu,Vega * 10**(-0.4 m_Vega)                          (2)

(vega_mag_to_mjy below; the F_nu,Vega of each band is read back from the
catalog itself, which lists both the magnitude and the flux in mJy).

A DIFFERENCE of magnitudes is a flux RATIO; the zero point cancels:

        m_a - m_b = -2.5 log10(F_a / F_b)
   <=>  F_a / F_b = 10**(-0.4 (m_a - m_b))                           (3)

That is all the "magnitudes to flux" step is: equation (3).

======================================================================
2. What an aperture correction is
======================================================================
PHANGS-HST measures every cluster in a small circular aperture of r = 4 px
(0.158" at 0.0396"/px), roughly the half-light radius of a cluster at these
distances. The sky is the median of an annulus at 7-8 px. A small aperture
keeps neighbours out but misses the outer light. The aperture correction
AC (in mag) is the magnitude difference between the "total" light and the
light in the 4 px aperture:

        AC     = m_total - m_4px          (negative: the total is brighter)
        m_total = m_4px + AC                                          (4)

and, from equation (3), as a flux factor:

        F_total = F_4px * 10**(-0.4 AC)                               (5)

Example: NGC1365 has AC_V = -0.61 mag, so F_total = F_4px * 10**(0.244)
= 1.754 * F_4px. The catalog V fluxes are 1.75x the 4 px fluxes.
(ac_mag_to_flux_factor / flux_factor_to_ac_mag below.)

"Total" here means the light within 20 px (0.79"): that is how far the
growth curves are followed (Section 4.1).

======================================================================
3. How Deger+22 measure AC (Section 4.1): three steps, eqs. (1)-(4) of the paper
======================================================================
The calibrators are bright (V < 24 mag in 4 px) and isolated class 1 and
class 2 clusters of the galaxy, whose normalized growth curves never drop
by more than 2% between consecutive radii (Section 4.2). The correction is
taken from the MEAN normalized growth curve of these clusters, so ONE value
per galaxy is applied to every cluster.

  Step 1: light between 4 and 10 px (class 1 + 2).
     For every calibrator, measure the flux in apertures of 4, 5, ..., 10 px
     with the sky at 21-23 px, and normalize by the 10 px flux. Average the
     normalized curves over the clusters. With f4 = <F(4)/F(10)>:

        AC_s1 = -2.5 log10(1 / f4)  = 2.5 log10(f4)            [paper eq. 1]

     (The minus signs are lost in the arXiv/PDF text; the Table 2 values are
     negative, which fixes them.) This is equation (3) applied to the mean
     ratio F(4)/F(10).

  Step 2: light between 10 and 20 px (class 1 only, as class 2 are crowded
     beyond 10 px). The same procedure with radii 10-20 px, sky 21-23 px,
     normalized at 20 px. With f10 = <F(10)/F(20)>:

        AC_s2 = -2.5 log10(1 / f10) = 2.5 log10(f10)           [paper eq. 2]

  Step 3: cluster light removed by the 7-8 px sky annulus. The catalog sky
     annulus is so close that it still contains cluster light, so the sky is
     over-estimated and some cluster light is subtracted. The paper measures
     the photometry with the 7-8 px sky and writes

        AC_s3 = -2.5 log10(f8 / f7)                             [paper eq. 3]

     with f8, f7 the mean normalized fluxes at 8 and 7 px. In our tests we
     measured this light DIRECTLY instead: the mean ratio of the 4 px flux
     with the far sky (21-23 px) to the 4 px flux with the 7-8 px sky,

        AC_s3 = -2.5 log10(< F4(sky 21-23) / F4(sky 7-8) >)     (our version)

     The two differ. With the sky taken AT 7-8 px, F(8) - F(7) is the annulus
     light minus its own median, which is ~0 for any smooth profile, so the
     literal eq. 3 is ~0 by construction. On our mock population it gives
     -0.000 mag, while the direct measurement gives -0.039 mag (3.7% of the
     flux). NGC1365's published value is -0.009 mag (0.8%). So at the level
     of a few percent, the paper's step 3 may under-correct the light the
     7-8 px sky removes. The other two steps dominate (x1.59 and x1.10 for
     NGC1365).

  Total (paper eq. 4): the steps multiply in flux, so they add in mag:

        AC_tot = AC_s1 + AC_s2 + AC_s3                                 (6)
        F_total / F_4px = 10**(-0.4 AC_tot)
                        = [1/f4] * [1/f10] * [10**(-0.4 AC_s3)]        (7)

     NGC1365 (Table 2): -0.505 - 0.099 - 0.009 = -0.613 ~ -0.61 mag,
     i.e. x1.591 x 1.095 x 1.008 = x1.754.

  The chain works because the steps are ratios that cancel:
  F20/F4 = (F10/F4) * (F20/F10). Step 1 can use crowded class 2 clusters,
  which are clean out to 10 px; step 2 needs isolated class 1 clusters only
  for the outer part.

======================================================================
4. Other bands (Section 4.3)
======================================================================
The correction is only MEASURED in V. Too few clusters pass the quality
cuts in the blue bands, especially NUV. The other bands get FIXED offsets
from AC_V, set from the WFC3 PSF FWHM at each wavelength:

        AC_band = AC_V + offset_band
        offsets: NUV (F275W) -0.19, U (F336W) -0.12, B (F438W) -0.03,
                 I (F814W) -0.12 mag

The text gives only the absolute values "0.19, 0.12, 0.03, 0.12"; the signs
were lost. main() recovers them from the NGC1365 catalog (catalog flux /
our 4 px flux per band, Milky Way removed): the implied offsets are -0.22,
-0.14, -0.05 and -0.12 mag, so all four are negative: these bands
need a LARGER correction than V because their PSF is broader than V's
(F555W has the sharpest WFC3/UVIS PSF).

This assumes no colour gradients: one cluster gets the same shape
correction in every band, apart from the PSF.

======================================================================
5. Why this matters for our comparison (Table 3)
======================================================================
AC is an AVERAGE over the calibrators. For NGC1365, the individual-cluster
corrections have mean -0.52, median -0.49 and std 0.24 mag (range 1.16 mag)
around the adopted -0.61 (Table 3). A single cluster's true correction can
therefore differ from the galaxy value by ~25% (0.24 mag) in flux. Also,
the adopted -0.61 is ~0.1 mag more negative than the median individual
correction (-0.49): for a typical cluster it adds ~10% more flux than that
cluster's own growth curve gives. This matches the x1.10 we found between
the catalog and the per-cluster 20 px light. The error also depends
systematically on size: compact clusters are over-corrected and extended
ones under-corrected (see deger_mock_test.py). Our instP instead fits each
cluster's own size.

Functions
---------
ac_mag_to_flux_factor(ac)          10**(-0.4 ac)                 eq. (5)
flux_factor_to_ac_mag(k)           -2.5 log10(k)                 eq. (3)
band_ac(ac_v, band)                AC_V + offset                 Section 4.3
total_flux_from_4px(f4, ac_v, band)
vega_mag_to_mjy(m, f0_mjy) / mjy_to_vega_mag(f, f0_mjy)          eq. (2)
ab_mag_to_mjy(m) / mjy_to_ab_mag(f)
growth_curve(img, x, y, radii_px, sky_in_px, sky_out_px)
aperture_flux_err(img, x, y, r_px, sky_in_px, sky_out_px)   sigma_sky sqrt(N_ap)
deger_steps(images)                the three steps on a set of clusters  eqs. (6)-(7)
"""
import os
import sys

import numpy as np
from astropy.stats import sigma_clipped_stats
from photutils.aperture import CircularAnnulus, CircularAperture, aperture_photometry
from _paths import OUTPUT_DIR  # noqa: E402

# ── published numbers ────────────────────────────────────────────────────
HST_PIX_ARCSEC = 0.0396                 # PHANGS-HST pixel scale ("), apertures are in these px
APERTURES_PX = dict(r4=4, sky_in=7, sky_out=8, r10=10, r20=20, far_sky_in=21, far_sky_out=23)

# Table 2: galaxy -> (AC_tot, AC_s1, AC_s2, AC_s3) in V mag
DEGER22_TABLE2 = {
    'NGC4826': (-0.72, -0.573, -0.127, -0.023), 'NGC3351': (-0.68, -0.532, -0.134, -0.018),
    'NGC3627': (-0.81, -0.613, -0.191, -0.004), 'NGC2835': (-0.81, -0.618, -0.178, -0.019),
    'NGC5248': (-0.65, -0.488, -0.157, -0.009), 'NGC4571': (-0.61, -0.416, -0.172, -0.017),
    'NGC4689': (-0.63, -0.497, -0.120, -0.018), 'NGC4569': (-0.76, -0.541, -0.212, -0.004),
    'NGC4535': (-0.71, -0.533, -0.166, -0.013), 'NGC1792': (-0.80, -0.614, -0.185, -0.005),
    'NGC4548': (-0.64, -0.456, -0.162, -0.025), 'NGC1566': (-0.62, -0.496, -0.114, -0.015),
    'NGC1433': (-0.62, -0.510, -0.096, -0.019), 'NGC1559': (-0.67, -0.520, -0.138, -0.011),
    'NGC1365': (-0.61, -0.505, -0.099, -0.009), 'NGC4654': (-0.83, -0.576, -0.247, -0.007),
    'NGC2775': (-0.45, -0.371, -0.071, -0.010),
}
# Table 3: galaxy -> (mean, median, std, range) of the INDIVIDUAL cluster corrections (mag)
DEGER22_TABLE3_NGC1365 = (-0.52, -0.49, 0.24, 1.16)

# Section 4.3: AC_band = AC_V + offset (signs verified on the NGC1365 catalog, see main())
BAND_OFFSET_MAG = {'F275W': -0.19, 'F336W': -0.12, 'F438W': -0.03, 'F555W': 0.0, 'F814W': -0.12}
CATALOG_BAND = {'F275W': 'NUV', 'F336W': 'U', 'F438W': 'B', 'F555W': 'V', 'F814W': 'I'}


# ── magnitudes <-> fluxes ────────────────────────────────────────────────
def ac_mag_to_flux_factor(ac_mag):
    """Aperture correction in mag (negative) -> multiplicative flux factor (> 1).
    F_total = F_4px * 10**(-0.4 AC)."""
    return 10.0 ** (-0.4 * np.asarray(ac_mag, float))


def flux_factor_to_ac_mag(factor):
    """Flux ratio F_total / F_4px -> aperture correction in mag (negative)."""
    return -2.5 * np.log10(np.asarray(factor, float))


def band_ac(ac_v, band):
    """AC in `band` from the galaxy's V-band AC plus the fixed offset (Section 4.3)."""
    return ac_v + BAND_OFFSET_MAG[band.upper()]


def total_flux_from_4px(f4, ac_v, band='F555W'):
    """Catalog-style total flux: the 4 px flux (sky 7-8 px) times the band's
    correction factor. ac_v e.g. DEGER22_TABLE2['NGC1365'][0]."""
    return np.asarray(f4, float) * ac_mag_to_flux_factor(band_ac(ac_v, band))


def ab_mag_to_mjy(m_ab):
    return 3631e3 * 10.0 ** (-0.4 * np.asarray(m_ab, float))


def mjy_to_ab_mag(f_mjy):
    return -2.5 * np.log10(np.asarray(f_mjy, float) / 3631e3)


def vega_mag_to_mjy(m_vega, f0_mjy):
    """F = F_nu,Vega * 10**(-0.4 m_Vega); f0_mjy = Vega's flux density in the
    filter (vega_zero_points() reads it from a PHANGS catalog)."""
    return f0_mjy * 10.0 ** (-0.4 * np.asarray(m_vega, float))


def mjy_to_vega_mag(f_mjy, f0_mjy):
    return -2.5 * np.log10(np.asarray(f_mjy, float) / f0_mjy)


def vega_zero_points(table):
    """F_nu,Vega (mJy) per band from a PHANGS catalog that lists both
    PHANGS_<b>_VEGA and PHANGS_<b>_mJy: median of F * 10**(0.4 m)."""
    out = {}
    for band, b in CATALOG_BAND.items():
        mc, fc = f'PHANGS_{b}_VEGA', f'PHANGS_{b}_mJy'
        if mc in table.colnames and fc in table.colnames:
            m, f = np.asarray(table[mc], float), np.asarray(table[fc], float)
            ok = np.isfinite(m) & np.isfinite(f) & (f > 0) & (m < 90)
            out[band] = float(np.median(f[ok] * 10.0 ** (0.4 * m[ok]))) if ok.any() else np.nan
    return out


# ── the three steps (Section 4.1) on images ──────────────────────────────
def _sky(img, x, y, r_in, r_out):
    v = CircularAnnulus((x, y), r_in, r_out).to_mask('center').get_values(img)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else np.nan


def aperture_flux(img, x, y, r_px, sky):
    ap = CircularAperture((x, y), r_px)
    return float(aperture_photometry(np.nan_to_num(img - sky), ap)['aperture_sum'][0])


def aperture_flux_err(img, x, y, r_px, sky_in_px, sky_out_px):
    """1-sigma error of a sky-subtracted aperture flux (image units), the
    same definition the pipeline uses (native_phot_aux_func._aperture_photometry):

        sigma_F = sigma_sky * sqrt(N_ap)

    sigma_sky = 3-sigma-clipped std of the sky-annulus pixels, N_ap = pixels
    in the aperture. Any correction factor k applied to the flux multiplies
    the error too: sigma(k F) = k sigma_F. Not included (as in the pipeline):
    the error of the sky median itself, ~ N_ap sigma_sky sqrt(pi/2 / N_sky),
    the correlated noise of drizzled pixels, and the scatter of the
    aperture correction (Table 3: 0.24 mag).
    Returns (sigma_F, sigma_sky, N_ap, N_sky)."""
    v = CircularAnnulus((x, y), sky_in_px, sky_out_px).to_mask('center').get_values(img)
    v = v[np.isfinite(v)]
    if v.size < 3:
        return np.nan, np.nan, 0, int(v.size)
    sig = float(sigma_clipped_stats(v, sigma=3.0)[2])
    n_ap = float(np.pi * r_px ** 2)
    return sig * np.sqrt(n_ap), sig, n_ap, int(v.size)


def growth_curve(img, x, y, radii_px, sky_in_px, sky_out_px):
    """Sky-subtracted flux in circular apertures of radii_px (px of THIS image)."""
    s = _sky(img, x, y, sky_in_px, sky_out_px)
    return np.array([aperture_flux(img, x, y, r, s) for r in radii_px])


def deger_steps(sources, pix_arcsec=HST_PIX_ARCSEC, step3='paper'):
    """
    The Deger+22 three-step correction from a list of calibrator clusters,
    sources = [(image, x, y), ...] in V (pixel coordinates of each image).
    Radii are the paper's HST pixels, converted to this image's pixels.
    step3 : 'paper' (eq. 3, f8/f7 with the 7-8 px sky) or 'direct'
            (<F4(sky 21-23) / F4(sky 7-8)>, as in our tests).
    Returns dict(AC_s1, AC_s2, AC_s3, AC_tot [mag], factor = 10**(-0.4 AC_tot),
                 f4, f10 [mean normalized fluxes]).
    """
    k = HST_PIX_ARCSEC / pix_arcsec                      # HST px -> this image's px
    P = {n: v * k for n, v in APERTURES_PX.items()}
    r1 = np.arange(4, 11) * k                           # step 1: 4..10 px
    r2 = np.arange(10, 21) * k                          # step 2: 10..20 px
    n1, n2, s3 = [], [], []
    for img, x, y in sources:
        g1 = growth_curve(img, x, y, r1, P['far_sky_in'], P['far_sky_out'])
        n1.append(g1 / g1[-1])                          # normalized at 10 px
        g2 = growth_curve(img, x, y, r2, P['far_sky_in'], P['far_sky_out'])
        n2.append(g2 / g2[-1])                          # normalized at 20 px
        if step3 == 'paper':
            g3 = growth_curve(img, x, y, np.arange(1, 21) * k, P['sky_in'], P['sky_out'])
            g3 = g3 / g3[-1]                            # normalized at 20 px
            s3.append(g3[7] / g3[6])                    # f8 / f7
        else:
            far = aperture_flux(img, x, y, P['r4'], _sky(img, x, y, P['far_sky_in'], P['far_sky_out']))
            near = aperture_flux(img, x, y, P['r4'], _sky(img, x, y, P['sky_in'], P['sky_out']))
            s3.append(far / near)
    f4 = float(np.mean([c[0] for c in n1]))             # <F(4)/F(10)>
    f10 = float(np.mean([c[0] for c in n2]))            # <F(10)/F(20)>
    ac1 = -2.5 * np.log10(1.0 / f4)
    ac2 = -2.5 * np.log10(1.0 / f10)
    ac3 = -2.5 * np.log10(float(np.mean(s3)))
    tot = ac1 + ac2 + ac3
    return dict(AC_s1=ac1, AC_s2=ac2, AC_s3=ac3, AC_tot=tot, factor=float(ac_mag_to_flux_factor(tot)),
                f4=f4, f10=f10)


# ── checks ───────────────────────────────────────────────────────────────
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    sys.path.insert(0, root)

    print('1. NGC1365, Deger+22 Table 2 (V band):')
    ac, s1, s2, s3 = DEGER22_TABLE2['NGC1365']
    print(f'   AC_tot = {s1} + {s2} + {s3} = {s1 + s2 + s3:.3f} mag  (table: {ac})')
    print(f'   flux factors: step1 x{ac_mag_to_flux_factor(s1):.3f}, step2 x{ac_mag_to_flux_factor(s2):.3f}, '
          f'step3 x{ac_mag_to_flux_factor(s3):.3f}  ->  total x{ac_mag_to_flux_factor(ac):.3f}')
    print('   per band (AC_V + offset):')
    for b in BAND_OFFSET_MAG:
        a = band_ac(ac, b)
        print(f'     {b} ({CATALOG_BAND[b]:>3s}): AC = {a:+.2f} mag  ->  F_total = {ac_mag_to_flux_factor(a):.3f} x F_4px')
    m, med, sd, rng = DEGER22_TABLE3_NGC1365
    print(f'   individual clusters (Table 3): median {med} mag = x{ac_mag_to_flux_factor(med):.3f}, '
          f'std {sd} mag = +-{100 * (ac_mag_to_flux_factor(-sd) - 1):.0f}% in flux')

    # 2. the catalog itself: catalog flux / our 4 px flux on the real images
    from astropy.io import fits
    from astropy.table import Table
    from astropy.wcs import WCS
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from phot_aux_func import corr_flux
    import run_full_pipeline as rfp
    cat = Table.read(os.path.join(OUTPUT_DIR, 'NGC1365', 'NGC1365_photometry_v2.fits'))
    print('\n2. NGC1365 catalog Vega zero points F_nu,Vega (mJy), from PHANGS_<b>_VEGA and _mJy:')
    zps = vega_zero_points(cat)
    print('   ' + '  '.join(f'{b}: {v:.4g}' for b, v in zps.items()))
    try:
        from dustmaps.sfd import SFDQuery
        ebv = float(SFDQuery()(SkyCoord.from_name('NGC1365')))
    except Exception:
        ebv = 0.0205
    cfg = rfp.load_config(os.path.join(root, 'config.ini'))
    paths = rfp.find_hst_band_paths(cfg['hst_root'], 'NGC1365')
    sky = SkyCoord(np.asarray(cat['PHANGS_RA'], float) * u.deg, np.asarray(cat['PHANGS_DEC'], float) * u.deg)
    print(f'\n3. catalog flux / our 4 px flux (sky 7-8 px) on the real images, {len(cat)} clusters, '
          f'Milky Way (E(B-V)={ebv:.4f}) removed:')
    print('   band   median ratio  ->  implied AC   offset vs V   (paper: AC_V + offset)')
    ac_v_meas = None
    for b in ('F555W', 'F275W', 'F336W', 'F438W', 'F814W'):
        with fits.open(paths[b]) as h:
            img, hdr = h[0].data, h[0].header
        w = WCS(hdr)
        pix = float(w.proj_plane_pixel_scales()[0].to('arcsec').value)
        k = HST_PIX_ARCSEC / pix
        f4 = []
        for c in sky:
            x, y = (float(v) for v in c.to_pixel(w))
            f4.append(aperture_flux(img, x, y, 4 * k, _sky(img, x, y, 7 * k, 8 * k)) * 1e3)   # Jy -> mJy
        f4 = np.array(f4)
        mw = float(corr_flux(1.0, rfp.FILTER_WAVELENGTHS[b][1], ebv))
        catf = np.asarray(cat[f'PHANGS_{CATALOG_BAND[b]}_mJy'], float)
        ok = (f4 > 0) & np.isfinite(catf) & (catf > 0)
        ratio = float(np.median(catf[ok] / f4[ok]) / mw)
        a = float(flux_factor_to_ac_mag(ratio))
        if b == 'F555W':
            ac_v_meas = a
        print(f'   {b}   x{ratio:.3f}          {a:+.3f} mag    {a - ac_v_meas:+.3f}        '
              f'{band_ac(ac, b):+.2f} (offset {BAND_OFFSET_MAG[b]:+.2f})')

    # 4. the three steps on the mock population, both step-3 versions
    mock = os.path.join(here, 'output_deger_mock', 'deger_mock.fits')
    if os.path.exists(mock):
        t = Table.read(mock)
        t = t[(np.char.strip(np.asarray(t['psf']).astype(str)) == 'A') & (t['snr'] == 300)]
        a1 = 2.5 * np.log10(np.mean(np.asarray(t['f4']) / np.asarray(t['f10'])))
        a2 = 2.5 * np.log10(np.mean(np.asarray(t['f10']) / np.asarray(t['f20'])))
        a3_direct = -2.5 * np.log10(np.mean(np.asarray(t['f4']) / np.asarray(t['f4_s78'])))
        a3_paper = -2.5 * np.log10(np.mean(np.asarray(t['f8_s78']) / np.asarray(t['f7_s78'])))
        print(f'\n4. three steps on the S/N 300 mock population (deger_mock_test.py, {len(t)} clusters):')
        print(f'   AC_s1 {a1:+.3f}  AC_s2 {a2:+.3f}  AC_s3 paper (f8/f7) {a3_paper:+.3f} / direct {a3_direct:+.3f} mag')
        print(f'   AC_tot = {a1 + a2 + a3_paper:+.3f} (paper step 3) / {a1 + a2 + a3_direct:+.3f} (direct) mag'
              f'  ->  x{ac_mag_to_flux_factor(a1 + a2 + a3_paper):.3f} / x{ac_mag_to_flux_factor(a1 + a2 + a3_direct):.3f}')


if __name__ == '__main__':
    main()
