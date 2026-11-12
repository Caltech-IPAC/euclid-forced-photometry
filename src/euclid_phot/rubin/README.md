# Euclid + Rubin forced photometry

Run `notebooks/04.1_rubin_forced_photometry.ipynb` in the environment used by the other
photometry notebooks. The first code cell holds the RSP token, Euclid release
(DR1 by default; Q1 selectable), EDFS coordinates, science-box size, bands and
mask policy. The current example requests u/g/r/i/z/y in a 100-arcsec box,
without Rubin image padding. Notebook Markdown focuses on Rubin-specific
differences and refers to the DR1 ground-based notebook for the shared steps.

## Authentication and images

Create a token at https://data.lsst.cloud under Settings → Access tokens, with
`read:tap`, `read:image`, and `exec:notebook`. Copy the complete token shown once
at creation (`gt-<key>.<secret>`), not the token-list key. Restore the placeholder
before sharing the notebook. Tokens are never written into data products.

`client.py` uses TAP, SIA, DataLink and SODA. Separate science+WCS and masked-image
requests avoid downloading the very large serialized exposure PSF/provenance
container. Their science pixels must agree before combination. Images, variance,
mask and WCS are cached locally; RMS remains independent of mask exclusions.

The baseline DP1 coadd cut rejects BAD, SAT, NO_DATA, invalid science or variance,
and non-positive variance. CR/INTRP are retained; CLIPPED, REJECTED, SENSOR_EDGE,
and INEXACT_PSF remain diagnostics. `MASK_BAD_PLANES` can add conditional cuts.
A fitting pixel gets `1/RMS²` or zero weight. This policy follows the coadd
section of Rubin's tutorial, not its separate single-visit recommendations.

## Exact official PSFs

`psf.py` evaluates and downloads the exact official coadd PSF kernels, checking
that each Butler dataset UUID matches the SIA-selected science image. Signed
interpolation lobes are preserved. There is no empirical or Gaussian fallback.

The small evaluation runs through Rubin's documented RSP notebook execution API
(`remote.py`), using its existing LSST software. No local Butler or LSST stack is
installed. On an uncached request, the helper reuses an existing RSP lab or starts
an available small recommended lab. It uses an isolated execution kernel and
stops only a lab it started. Temporary lab startup and model evaluation can take
a minute; future requests for the same dataset and position use local caches.
The adapter implements the HTTP protocol used by rubin-nublado-client because
that package's current releases require newer Python than this project's 3.11.

The Rubin-only `rubin.make_image` adapter delegates validation and unit conversion
to `ground_based.make_image`, then restores the signed official kernel before
fitting. The shared ground-based files and their nonnegative-PSF API are unchanged.

## One-time PSF profile verification

Regular notebook runs only display a saved verification PNG. Development work
uses:

```bash
.venv/bin/python scripts/verify_rubin_psf.py --size 300
.venv/bin/python scripts/verify_rubin_psf.py --redraw
```

The first command reads settings from the notebook, selects isolated high-S/N
Rubin point sources and matches their measured SEP centroids. It downloads a
wider image from the same science coadd patch, and evaluates the official PSF at
every selected stellar position. It reuses the shared ground-based stellar-stamp extraction, with a Rubin-only
signed-model comparison adapter. The empirical median is a diagnostic and never enters the fit.
The second command redraws the figure from saved inputs without any API access.

The verification is bundled under `src/euclid_phot/rubin/data/dp1/<RA>_<Dec>/`
and included in installed packages. Subset display PNGs are cached in the configured
output directory. The bundled data include:

- `psf_validation.png`: official kernels with colorbars and stellar/model radial profiles;
- `<band>_psf_validation.npz`: individual observed/model stamps, mean/median
  stacks, median residual data, per-star radial profiles, radius grid, sampling
  and dataset provenance;
- `<band>_stars.ecsv`: selected positions, centroids and catalog S/N;
- `psf_validation_summary.ecsv`: counts, descriptive core widths and method.

The 300-arcsec EDFS test retained 9/27/36/36/37/26 stars in u/g/r/i/z/y.
The saved figure has four rows and three columns: u/g/r images and profiles,
then i/z/y images and profiles, with the same labels and normalization.
Plot slots always follow u/g/r/i/z/y. Bands omitted from `RUBIN_BANDS` leave
blank slots in the overview, PSF and catalog-comparison figures. The notebook
uses cached subset PNGs for PSF display, rendered from archived profiles when
needed; this does not rerun verification or contact RSP. Inspection skips
omitted bands and handles an unavailable inspection selection without errors.
The profile comparison
is limited by sky subtraction, neighbors, color, spatial PSF variation and
stacking; it does not establish absolute photometric calibration. Shading shows
the 16–84% variation among individual stellar radial profiles, not an error on
the median. A figure from another field is not silently presented as verification
for a new coordinate: the notebook resolves its saved figure by field/release.
The 33/35-pixel kernels at 0.2 arcsec/pixel give complete radial annuli out to
3.1/3.3 arcsec. This is the downloaded model stamp extent, not a physical cutoff
of stellar light. PSF images show unit-total kernels; comparisons use stellar
core-flux normalization and matched model amplitudes. Both sides retain signed
pixels. Small centroid offsets can remain after recentering; the stellar/model
arrays preserve that difference without an additional shift fit.

## Shared fit and catalog comparison

The ground-based notebook is the fitting standard. The Rubin notebook shows
the same stages separately: `rubin.make_image` delegates image and weight preparation to `gb.make_image`,
`gb.prepare_euclid` prepares a padded MER/VIS scene, and `gb.fit_vis` fits the
normal VIS morphology tree once. The selected Euclid release and its caches
are shared with the other notebooks; no Q1/DR1 fallback is introduced.

The notebook exposes the selected neighboring VIS sources with
`gb.select_mer_scene`, then calls `gb.fit_ground` separately for each band.
The cell shows the supplied-PSF, sky, recentering and background options. The
ground fitter builds the Tractor image, clones the fixed VIS models, solves
fluxes with the shared `forced` functions, and
extracts formal errors and model images. `ground_fits[band]` is the standard
`gb.GroundFit`; `fit_scenes[band]` exposes the selected input sources. The
official Rubin PSF and calibrated WCS are used without recentering, an extra
constant-sky fit, or a SEP sky map. The downloaded science images are already
background subtracted. The ground fitter also supplies tiling for larger images. There is no
separate Rubin fitting implementation in the notebook.

The notebook aligns per-band measurements by source ID, applies the VIS quality
mask, and uses the shared catalog exporter. Only the requested science box is
exported. `gb.save_fit` writes the full per-band products into Rubin-specific
directories. There is no call to `rubin.run_forced_photometry` or `RubinRun.save`
in the notebook; those older convenience interfaces remain available.

`compare.py` performs unique sky matches within 0.5 arcsec to `dp1.Object`, the
appropriate coadd reference catalog. It compares PSF flux for Rubin point sources
and cModel flux for extended sources, converting nJy to µJy. A stellar control
must also have an Euclid PointSource model. Rubin-pointlike objects resolved in
VIS are labeled mixed; comparing them to PSF flux is not a stellar calibration.
Accepted plots require clean reference flags, the VIS reliability flag, and S/N
above 5 in both measurements. Signed fluxes and rejected nearby matches remain
in the saved tables. No empirical rescaling is applied.
The comparison cell's `EDGE_cut` input (default 2 arcsec) marks sources near
the actual Rubin image boundary gray with alpha 0.2 in both panels, labeled
`too close to the edg (2")` for that default. The match tables retain their edge distances
and flags; this styling does not change acceptance, fluxes or image padding.
The notebook displays fit images with the standard `gb.show_fit_cutouts`
(aligned VIS, science, Tractor model and residual), without table previews.
Section 6 follows the catalog comparison. Its first cell selects three good
examples (accepted interior matches within 0.1 mag in all selected bands);
the second selects five distinct outliers ranked by their largest offset
above 0.4 mag in any selected band. Both default to `INSPECT_BANDS = ["r", "y"]`,
show one source at a time in the selected band order (r, y, then the next
source), and expose `N_SOURCES`,
`CUTOUT_SIZE_ARCSEC`, and `INSPECT_OBJECT_IDS` in the notebook.
The science/RMS overview has two rows and columns VIS followed by the requested
Rubin bands. The baseline
mask remains BAD/SAT/NO_DATA; RMS coadd structure is retained, displayed with
a viridis color scale rather than interpreted as missing or excluded pixels.

Outputs under `examples/output/rubin/dp1/<Euclid release>/<field>/` include
`forced_photometry.ecsv`, per-band reference match tables, fit and comparison
PNGs. Each `<band>/` subdirectory contains the standard ground writer's
`forced_catalog.ecsv` and `fit_images.fits` (science, inverse variance,
background, model and residual), plus `official_psf.fits`.
Section 6 saves per-source, per-band PNGs under `inspection/good/` and
`inspection/outliers/`, named `<object_id>_<band>.png`. Source selections
remain in `inspection_good.ecsv` and `inspection_outliers.ecsv`.

These are exploratory model amplitudes on the calibrated coadd image scale.
Rubin's model-dependent aperture correction is not automatically applied to the
Tractor estimator; metadata explicitly records this. A diagnostic check of the
official PSF correction at the field center gave about 0.983/0.989/0.990 in u/g/r,
which alone cannot explain larger catalog differences. Formal errors use diagonal
variance and omit coadd covariance, calibration and fixed-morphology uncertainty.
Galaxy flux definitions differ between the fits. Catalog disagreement is retained
as evidence to investigate, not removed by tuning the PSF or forcing agreement.

## Validation

The earlier 50-arcsec authenticated test ran end to end: official PSFs were downloaded, the
300-arcsec verification was saved, and the local DR1 fit measured 165 sources in
the 50-arcsec core. Each band has 61 nearby Rubin matches before quality cuts.
Offline tests cover masks, uncertainty preservation, credentials, signed-PSF
flux recovery, cache reuse, remote cleanup and catalog units/classification.
Some older unrelated tests still expect the former nJy ground-image interface.

The earlier 200-arcsec u/g/r/i/z/y workflow also completed end to end
(that exploratory run used a fitted constant residual sky and 0.7-arcsec matching).
It prepared 3510 MER rows, retained 3499 VIS models, and exported 2673 sources
in the science box, with 957 nearby Rubin catalog matches per band. Each band
uses a 1000×1000-pixel image and an exact official PSF. Saved FITS products,
catalog columns, edge flags and image previews were checked; `ground_based/`
was unchanged. `run_summary.json` records the counts and residual diagnostics.
The larger field contains localized residual outliers: the global weighted RMS
is about 3.7–6.0, while robust pixel scatter is 1.02–1.07. The worst 0.1% of
pixels contribute 74–89% of chi-squared. These outliers remain for inspection;
the baseline mask and shared fit were not tuned to remove them.

## Sources

- https://dp1.lsst.io/tutorials/notebook/202/notebook-202-6.html — coadd mask policy
- https://dp1.lsst.io/tutorials/api/api-102-3.html — SODA data products
- https://dp1.lsst.io/tutorials/notebook/205/notebook-205-2.html — official kernels
- https://nublado.lsst.io/user-guide/lab.html — remote execution protocol
- https://dp1.lsst.io/tutorials/notebook/105/notebook-105-3.html — forced photometry
- https://dp1.lsst.io/overview/known_issues_and_subtleties.html — calibration limits
