# Euclid forced photometry

Measure fluxes on Euclid Q1 VIS and NISP images using Tractor source models,
with optional unWISE W1/W2, Spitzer/IRAC and user-supplied ground-based
photometry. VIS constrains the source profiles; the lower-resolution fits keep those profiles fixed and
solve for flux.

This is an educational analysis package. It provides explicit diagnostics
and limited validation tests; users must assess the PSF, residuals, source
list and uncertainty assumptions for their own science sample.

## Notebooks

- [00: Modeling a galaxy with the Tractor](notebooks/00_how_tractor_works.ipynb): one galaxy,
  with image construction, model parameters and optimization written out.
- [01: Forced photometry on Euclid NISP with a VIS prior](notebooks/01_nisp_forced_photometry.ipynb):
  a 200-arcsec field, source models on VIS, forced fluxes in Y, J and H,
  residuals and catalog checks.
- [02: Forced photometry on unWISE W1 and W2](notebooks/02_wise_forced_photometry.ipynb):
  the same field on unWISE W1 and W2, with residuals and a catalog cross-check.
- [03: Forced photometry on Spitzer/IRAC](notebooks/03_irac_forced_photometry.ipynb):
  a 200-arcsec field in EDF-F, IRAC channels 1-4 with a spatially varying
  effective PRF, residuals, and comparisons with the DAWN catalog and unWISE.
- [04: Forced photometry on your own ground-based image](notebooks/04_ground_based_forced_photometry.ipynb):
  a template for measuring any ground image with the VIS models as the
  prior; replace the example SDSS g image with your own science,
  uncertainty and PSF. Includes WCS and PSF checks against field stars, a
  comparison with the SDSS catalog, and an injected-sky background test.
- [05: Multi-band photometry in one call](notebooks/05_one_call.ipynb): a compact
  50-arcsec run measuring Euclid, unWISE and IRAC bands in one call, and the
  resulting catalog.
- [06: Photometry at supplied coordinates](notebooks/06_photometry_at_coordinates.ipynb):
  photometry at user positions without a MER source list.
- [Supporting: Model selection](notebooks/supporting_notebooks/model_selection.ipynb):
  the profile trials for one pair of neighbouring sources.
- [Supporting: Injection and recovery](notebooks/supporting_notebooks/injection_recovery.ipynb):
  matching-PSF point-source checks on real backgrounds.

Start with 00 to learn the fitting model, 01 to inspect the full workflow,
or 05 for a short working example.

## How the package is organised

Every band is measured with the same recipe; only its pixels and PSF differ:

1. build the band's Tractor image: data, inverse variance, PSF, WCS and the
   photometric scale (`tractor_image.make_tractor_image`);
2. copy the VIS source models into the band, positions and shapes kept
   (`sources.clone_sources`);
3. freeze everything but the fluxes and solve (`forced.flux_tractor`,
   `forced.solve_fluxes`);
4. turn the solve's inverse variances into errors, optionally scaled by the
   residual scatter (`forced.flux_inverse_variance`, `forced.chi_inflation`).

```
src/euclid_phot/        the package (import euclid_phot as ep)
  units.py  tractor_image.py  pixel_psf.py  sources.py  selection.py
  forced.py  background.py  scene.py  calibrate.py  pipeline.py  ...
  euclid/  wise/  irac/  ground_based/
```

The code is the `euclid_phot` package inside `src/` (the standard Python
"src layout"), so notebooks and scripts import it by name and cannot pick
up stray files from the working directory. The shared modules sit at the
top of the package, with `units`, `pixel_psf`, `background` and `scene`.
Each band folder holds only what is its own: `euclid/` (MER catalog, cutouts, PSF products, NISP), `wise/`
(unWISE coadds, PSF, catalog neighbours), `irac/` (Spitzer mosaics and the
effective PRF) and `ground_based/` (a user's image, its WCS and PSF checks).
Notebooks 02-04 run the recipe step by step for one band before calling the
band's one-line fit.

## Install and run

From the repository directory:

```bash
./scripts/install.sh
# For optional WISE support, use: ./scripts/install.sh --wise
# If the script created .venv:
source .venv/bin/activate
jupyter lab notebooks/
```

The install script uses an active virtualenv or conda environment, or creates
`.venv`. It installs this package plus the required Tractor and astrometry.net
components. WISE additionally requires `unwise_psf` and its dependencies;
IRAC needs nothing extra.
Use a Jupyter kernel from that environment. If the script created `.venv`,
activate it with `source .venv/bin/activate` before starting Jupyter.

The notebooks locate the repository from their working directory and work
from either `notebooks/` or `notebooks/supporting_notebooks/`. Inputs are
fetched from public archives on first use; image data are not bundled with
this source distribution, except the example SDSS image for notebook 04 in
`examples/data/ground_based/`. The notebooks, the command line and the
Python API all cache downloads under `examples/native_data/` (relative to
the working directory); set `EUCLID_PHOT_DATA_DIR` to use another cache.
Network access is required when an input is absent.

Every push runs a check on GitHub (`.github/workflows/ci.yml`): the install
script, a lint pass, the public imports and an offline Tractor fit of a
synthetic image. It does not download survey data; re-run the notebooks
before merging changes to the fitting code.

## Computing cost

Most of the memory goes to the VIS model selection that starts every
`run_forced_photometry` call with `"model_selection": "tree"`. Its memory
grows faster than the field area, and each extra worker adds to it. The
fits that follow (VIS fluxes, NISP, unWISE, IRAC) need a few GB.

Measured on the field of [notebook 03](notebooks/03_irac_forced_photometry.ipynb)
(EDF-F, VIS models fitted to unWISE and IRAC) on a Mac Studio with an
Apple M3 Ultra (28 cores, 96 GB):

| Field | Sources | `n_workers` | Run time | Peak memory |
|---|---|---|---|---|
| 200″, VIS models only | 856 | 10 | 1 min | 26 GB |
| 200″ | 856 | 10 | 6 min | 26 GB |
| 200″ | 856 | 1 | 10 min | 16 GB |
| 300″ | 1953 | 10 | 17 min | 77 GB |

Peak memory includes the worker processes. Run time also depends on the
processor. Notebooks 01 and 02 run the same model selection on 200″ fields
and were not measured separately; the 50″ examples in notebooks 05 and 06
are much smaller.

A Jupyter kernel that runs out of memory stops without a traceback. For
fields larger than about 200″:

- Run the fit as a Python script rather than in a notebook, and put it
  under `if __name__ == "__main__":`. Model selection starts worker
  processes, and on macOS each one re-imports the script.
- Or split the field into tiles of at most 200″, fit each with a margin of
  at least 20″ on every side, and keep from each tile only the sources
  inside its unpadded core. Peak memory then stays at the single-tile
  level, and the margin keeps kept sources away from tile edges.

A field that crosses a MER tile boundary needs separate tile fits in any
case.

## One-call API

```python
from pathlib import Path
import euclid_phot as ep

result = ep.run_forced_photometry(
    269.48, 67.30, 50.0,  # RA, Dec in degrees; box side in arcsec
    prior={"band": "VIS", "objects": "mer", "model_selection": "tree"},
    target_bands={"euclid": ("Y", "J", "H"), "wise": (), "irac": ()},
    psf_product="grid", persource_psf=True,
    mask_bright_stars=True, with_flag=True, calibrate_errors=True,
    data_dir=Path("examples/native_data"), n_workers=4,
)
catalog = result.to_table()
catalog.write("photometry.csv", overwrite=True)
```

`model_selection="prior"` starts from MER classifications and shapes;
`"tree"` compares profile classes on VIS and also refines positions.
The tree adapts [The Farmer](https://ui.adsabs.harvard.edu/abs/2023ApJS..269...20W) (Weaver et al. 2023), with different
selection details and an additional Sérsic trial. Its shapes are already
fitted, so `free_shapes` does not add another shape fit on that path.

`psf_product="grid"` uses PSF samples on a regular grid; `"catalog"` uses
samples at MER catalog positions. With `persource_psf=True`, each source
uses its nearest sample inside a joint fit. Images retain their delivered
MER pixels. A cutout spanning multiple MER tiles needs separate tile fits.

The package also supports supplied coordinates and selected profile classes;
see [notebook 06](notebooks/06_photometry_at_coordinates.ipynb) and
`help(ep.run_forced_photometry)`. With the tree, unlisted neighbours detected
in the same blob are added to the model; with an explicit profile class the
supplied list must include them. The command-line entry point is
`euclid-phot run --help`; `--wise` and `--irac` add those bands, and
`--ground-image` your own image (see "Ground-based images").

## IRAC

IRAC is a target like NISP and WISE. This example also fits W1 and W2,
which needs the `--wise` install:

```python
result = ep.run_forced_photometry(
    52.932, -28.088, 200.0,
    prior={"band": "VIS", "objects": "mer", "model_selection": "tree"},
    target_bands={"euclid": (), "wise": ("W1", "W2"),
                  "irac": ("IRAC1", "IRAC2", "IRAC3", "IRAC4")},
    irac_options={"mosaic_backend": "dawn"},  # or "seip" (default), "pbcd"
    psf_product="grid", mask_bright_stars=True,
    data_dir=Path("examples/native_data"), n_workers=4,
)
catalog = result.to_table()  # adds flux_IRAC1_ujy, flux_err_IRAC1_ujy, ...
```

Each IRAC channel uses a spatially varying effective PRF rebuilt from the
detector PRFs and the exposures in the mosaic. `flux_err_IRAC<n>_ujy`
includes a 2% PRF systematic. `result.image_set(band)` returns the data,
model and residual images of the prior band and of every WISE and IRAC
band. See [notebook 03](notebooks/03_irac_forced_photometry.ipynb).

## Ground-based images

`euclid_phot.ground_based` measures an image you supply with the VIS models
as the prior: a science array, its 1-sigma uncertainty, a PSF stamp, a
celestial WCS and the flux unit of one pixel value. The WCS is first refined
on matched MER stars, and fluxes are reported in microJansky like every
other band. In the one-call driver:

```python
from astropy import units as u
from astropy.wcs import WCS
from euclid_phot import ground_based as gb

g = gb.make_image(sci, rms, psf, u.nanomaggy, WCS(header), band="g")
result = ep.run_forced_photometry(
    265.8432, 64.965, 220.0,          # a field that covers the whole image
    prior={"band": "VIS", "objects": "mer", "model_selection": "tree"},
    target_bands={"euclid": ()}, ground_images={"g": g},
    ground_options={"background_method": "none"},  # or "sep"
    psf_product="grid", mask_bright_stars=True, n_workers=4,
)
catalog = result.to_table()           # adds flux_g_ujy, flux_err_g_ujy, ...
```

or from the command line:

```bash
euclid-phot run --ra 265.8432 --dec 64.965 --size 220 --bands "" \
    --ground-image sci.fits --ground-variance var.fits --ground-psf psf.fits \
    --ground-unit nanomaggy --ground-band g
```

Choose the field to cover the image; sources outside it are not modelled.
Without a PSF stamp, `gb.gaussian_psf(fwhm_arcsec, pixel_scale_arcsec)`
(`--ground-seeing` on the command line) gives a placeholder, and
`ground_options={"psf_mode": "nearest_star"}` or `"avg_star"` builds the PSF
from stars in the image. The errors are the formal ones from the supplied
uncertainty map, without empirical scaling.
[Notebook 04](notebooks/04_ground_based_forced_photometry.ipynb) runs the
same fit step by step with WCS and PSF checks. Its SDSS example files can be
rebuilt from the SDSS archive with `python scripts/make_sdss_example.py`,
which also makes the same files for another SDSS frame.

## Interpreting results

Flux densities are in microJansky. AB magnitudes are given for positive
fluxes; a flux the fit could not constrain is NaN. The output metadata
records PSF conventions, uncertainty definitions and error-scale factors.

`reliable` is False when the source is near a bright star, on masked pixels,
or at the cutout edge. `blended` marks a close neighbour.

Euclid errors are conditional template errors, optionally scaled using
empty-position point-source fits. WISE and IRAC use a residual-based scale
factor, and IRAC errors add a 2% effective-PRF systematic in quadrature.
None of these treatments captures all blending, morphology, PSF or sky
uncertainty.
The reported five-times-median-error magnitude is an error summary, not a
measured detection-completeness limit.

Shared morphology can fail for color gradients or infrared-only emission.
Structured residuals and sensitivity to neighbouring models require inspection.
The example retains a few-percent NISP/MER difference; it should not be removed
by tuning a PSF to force catalog agreement. WISE and IRAC blends also remain
sensitive to PSF shape and source-list completeness. The WISE fit adds unWISE
catalog sources outside the field; the IRAC fit models only the VIS sources
inside it, so IRAC fluxes within a few arcsec of the edge can absorb light
from outside. Request a slightly larger field when the edge matters.

The [Schlafly et al. (2019) unWISE catalog](https://ui.adsabs.harvard.edu/abs/2019ApJS..240...30S) uses
a different PSF normalization. Notebook 02 transfers our fitted amplitudes
to that convention for its comparison only; exported fluxes retain their
fitting convention. Agreement with that catalog is a cross-check, not an
absolute calibration or proof of individual blended flux accuracy. The same
holds for the DAWN catalog comparison in notebook 03. W1 and W2 differ in
bandpass from IRAC1 and IRAC2, so their ratios depend on source colour.

Matching-PSF injections test rendering and recovery consistency. They do not
validate the real PSF, galaxy morphology, blind detection completeness, or
all uncertainty contributions. The injection notebook states the tested scope.

## References

The image modelling uses [The Tractor](https://github.com/dstndstn/tractor).
The IRAC effective PRF follows [PRFMAP](https://github.com/cosmic-dawn/prfmap)
with the detector PRFs and flux corrections of the
[IRAC Instrument Handbook](https://irsa.ipac.caltech.edu/data/SPITZER/docs/irac/iracinstrumenthandbook/);
IRAC extinction coefficients are from
[Indebetouw et al. (2005)](https://ui.adsabs.harvard.edu/abs/2005ApJ...619..931I).
Euclid reference flux definitions are documented in the
[MER photometry cookbook](https://euclid.esac.esa.int/dr/q1/dpdd/merdpd/merphotometrycookbook.html).
Please cite the relevant survey and method papers when using their data or methods.
