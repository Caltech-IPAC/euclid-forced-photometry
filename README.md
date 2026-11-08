# Euclid forced photometry

This repository is provided by the Euclid NASA Science Center at IPAC
(ENSCI) as a reference implementation and tutorial for forced photometry
with Euclid data. It is intended for the community working with the Euclid
Quick Data Release
([Euclid Collaboration: Aussel et al. 2025](https://arxiv.org/abs/2503.15302))
and covers the cases most users encounter: Euclid NISP with a VIS prior,
the mid-infrared with unWISE and Spitzer/IRAC, ground-based images supplied
by the user, and photometry at user-defined coordinates.

Forced photometry measures the fluxes of known sources instead of
detecting them anew in each image. The source list and the model of each
source come from Euclid VIS at 0.1 arcsec per pixel, from the MER catalog
([Euclid Collaboration: Romelli et al. 2025](https://arxiv.org/abs/2503.15305))
or from coordinates the user supplies. On every other image those models
keep their positions and shapes, are convolved with that image's PSF, and
only their fluxes are solved for, neighbours together. Blended light is
thus divided according to the VIS shapes, a source too faint to detect in
the target band still receives a flux and an error, and every band shares
one set of sources and one definition of flux, as needed for colours, SEDs
and photometric redshifts.

The fitting engine is [The Tractor](https://github.com/dstndstn/tractor)
([Lang, Hogg & Mykytyn 2016](https://ascl.net/1604.008)), used as it was
for the WISE photometry of SDSS sources
([Lang, Hogg & Schlegel 2016](https://arxiv.org/abs/1410.7397)). The
material is meant for learning the method and for exploratory analysis;
the PSF, the residuals, the source list and the error assumptions must
still be checked on each science sample before the measurements are used.

## Notebooks

- [00: Modeling a galaxy with the Tractor](notebooks/00_how_tractor_works.ipynb): one galaxy,
  with image construction, model parameters and optimization written out on VIS, then its
  flux forced in NISP H with the VIS shape held fixed.
- [01: Forced photometry on Euclid NISP with a VIS prior](notebooks/01_nisp_forced_photometry.ipynb):
  a 200-arcsec field in EDF-N, source models on VIS, forced fluxes in Y, J and H
  (the H fit written out step by step), residuals and catalog checks.
- [02: Forced photometry on unWISE W1 and W2](notebooks/02_wise_forced_photometry.ipynb):
  the same field on unWISE W1 and W2, with residuals and a catalog cross-check.
- [03: Forced photometry on Spitzer/IRAC](notebooks/03_irac_forced_photometry.ipynb):
  a 200-arcsec field in EDF-F, IRAC channels 1-4 with a spatially varying
  effective PRF, residuals, and comparisons with the DAWN catalog and unWISE.
- [04: Forced photometry on an SDSS image](notebooks/04_ground_based_forced_photometry.ipynb):
  a 200-arcsec SDSS g image as the example ground-based image, with a
  comparison to the SDSS catalog; swap in your own image in section 1.
- [05: Multi-band photometry in one call](notebooks/05_one_call.ipynb): a compact
  50-arcsec run measuring Euclid, unWISE and IRAC bands in one call, and the
  resulting catalog.
- [06: Photometry at supplied coordinates](notebooks/06_photometry_at_coordinates.ipynb):
  photometry at user positions without a MER source list.
- [Supporting: Model selection](notebooks/supporting_notebooks/model_selection.ipynb):
  why the MER models are refitted, the profile trials for one pair of neighbouring
  sources, and the tree against the MER models over a 50-arcsec field.
- [Supporting: Injection and recovery](notebooks/supporting_notebooks/injection_recovery.ipynb):
  matching-PSF point-source checks on real backgrounds.

Start with 00 to learn the fitting model, 01 to inspect the full workflow,
or 05 for a short working example.

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
`.venv`. A fresh environment is safest, for example
`conda create -n euclid python=3.11` and `conda activate euclid` before
running the script, rather than conda `base`. It installs this package plus the required Tractor and astrometry.net
components. WISE additionally requires `unwise_psf` and its dependencies;
IRAC needs nothing extra. Start Jupyter from that environment so the
notebooks pick up its kernel.

No survey data ship with the repository. The first time a notebook or a
fit needs a Euclid, unWISE, IRAC or SDSS input it downloads it from the
archive and keeps it under `examples/data/<release>/` in the repository,
or under `EUCLID_PHOT_DATA_DIR` if that is set. Later
runs read from that cache, so only the first run on a field needs network
access. The notebooks can be started from `notebooks/` or from
`notebooks/supporting_notebooks/`.

The cache moved from `examples/native_data/` to `examples/data/q1/` or `examples/data/dr1/deep_survey/`; existing caches are not picked up automatically.

## Running a fit

One call fetches the images and the MER catalog, builds the VIS source
models and measures the requested bands:

```python
from pathlib import Path
import euclid_phot as ep

result = ep.run_forced_photometry(
    269.48, 67.30, 50.0,  # RA, Dec in degrees; box side in arcsec
    prior={"band": "VIS", "objects": "mer", "model_selection": "tree"},
    target_bands={"euclid": ("Y", "J", "H"), "wise": (), "irac": ()},
    data_dir=Path("examples/native_data"), n_workers=4,
)
catalog = result.to_table()
catalog.write("photometry.csv", overwrite=True)
```

The field comes first, as its centre and the side of the square cutout.
The `prior` argument says where the source models come from: here the MER
catalog, with the profile class of each source chosen on the VIS pixels by
the model-selection tree (`"model_selection": "prior"` keeps the MER
classification instead). The models can also be built on a NISP band,
`prior={"band": "H", "objects": "mer"}`, which refits the MER shapes on H;
the tree and the STARSIGNAL star mask need VIS, and notebook 02 compares
the two for unWISE. The `target_bands` argument lists the bands to
measure; unWISE and IRAC bands go in their own groups, and a ground-based
image is passed separately as `ground_images`. The source list can also be
your own coordinates, passed as `user_coords`. The remaining options, for
the PSF, bright-star masks and error calibration, are described in
`help(ep.run_forced_photometry)` and shown in
[notebook 05](notebooks/05_one_call.ipynb); notebooks 03, 04 and 06 cover
IRAC, your own image and supplied coordinates. The same fit is available
from the command line as `euclid-phot run --help`.

## Interpreting results

A fit returns a result object, and `result.to_table()` turns it into one
table with a row per source. Each row holds the flux density in
microJansky and its error in every measured band, with an AB magnitude
where the flux is positive; a flux the fit could not constrain is NaN.
Two flags summarise the fitting conditions: `reliable` is False when the
source is near a bright star, on masked pixels or at the cutout edge, and
`blended` marks a close neighbour. The table metadata records the PSF
conventions, the uncertainty definitions and the error-scale factors.

The errors are those of the flux solve with the shapes held fixed, so
they describe the pixel noise and not the uncertainty of the model. In the
Euclid bands, `calibrate_errors=True` scales them by point-source fits at
empty positions, which measure the real noise of the images. WISE and IRAC
errors are scaled by the scatter of the residuals instead, and IRAC adds
a 2% systematic for its reconstructed PRF in quadrature. None of these treatments captures all
of the uncertainty from blending, morphology, PSF or sky. The table
metadata also gives, per band, the AB magnitude at five times the median
error (`depth_5sigma_ab`); that is a summary of the errors, not a measured
detection or completeness limit.

The method assumes that every band shares the morphology measured in the
prior band, VIS by default or a NISP band when `prior["band"]` says so.
That assumption fails for sources with colour gradients or with emission
seen only at longer wavelengths. A NISP prior is closer in wavelength to
unWISE and IRAC, but its wider PSF measures small and blended shapes less
well; on the field of notebook 02 the choice changed most unWISE fluxes
by a few percent or less, with no systematic shift for red sources, and
for the sources that changed by more the unWISE pixels could not tell
which models were right; treat that difference as a systematic error.
Either way the residuals show where the assumption fails, so they should
be inspected; `result.image_set(band)` returns the data, model and
residual images of every band measured. Notebook 01
shows a few-percent difference between the NISP fluxes and the MER catalog
that we left in place rather than tune the PSF until the catalogs agree.
In the WISE and IRAC bands, where several prior-band sources fall inside
one PSF, the fluxes also depend on the PSF shape and on the completeness
of the source list. The WISE fit adds unWISE catalog sources outside the
field to account for their light; the IRAC fit models only the sources
of the prior image inside the field, so IRAC fluxes within a few arcsec
of the edge can absorb light from outside. Request a slightly larger field
when the edge matters.

The catalog comparisons in the notebooks are cross-checks, not
calibrations. The
[Schlafly et al. (2019) unWISE catalog](https://ui.adsabs.harvard.edu/abs/2019ApJS..240...30S)
uses a different PSF normalization, so notebook 02 converts our fitted amplitudes
to that convention for the comparison only; the exported fluxes keep the
fitting convention. Agreement with that catalog, or with the DAWN catalog
in notebook 03, does not make an individual blended flux right. Notebook
03 also compares IRAC1 and IRAC2 with W1 and W2; those bandpasses differ,
so the ratio depends on the source colour and is not expected to be one.

The injection-and-recovery notebook plants point sources rendered with the
fit's own PSF into real backgrounds and measures them again. That checks
that rendering and recovery agree with each other; it says nothing about
the real PSF, galaxy morphology, detection completeness or the uncertainty
terms listed above. The notebook states what it does and does not cover.

## Computing cost

Forced photometry on Euclid data at the scale of these notebooks does not
need a cluster. Every example runs on a single machine. A 200-arcsec field
of about 850 sources, the size used in notebooks 01 to 03, takes about ten
minutes with one worker process and 16 GB of memory, or six minutes and
26 GB with ten workers, on a workstation. The 50-arcsec examples of
notebooks 05 and 06 are far smaller. Larger areas are fitted as tiles, so
the cost of a wide run grows with the number of tiles, not with the field
size.

Memory is set by the VIS model selection that opens every
`run_forced_photometry` call with `"model_selection": "tree"`. It grows
faster than the field area, and each extra worker adds to it, while the
fits that follow (VIS fluxes, NISP, unWISE, IRAC) need a few GB and take
most of the time. Measured on the field of
[notebook 03](notebooks/03_irac_forced_photometry.ipynb) (EDF-F, VIS
models fitted to unWISE and IRAC) on a Mac Studio with an Apple M3 Ultra
(28 cores, 96 GB):

| Field | Sources | `n_workers` | Run time | Peak memory |
|---|---|---|---|---|
| 200″, VIS models only | 856 | 10 | 1 min | 26 GB |
| 200″ | 856 | 10 | 6 min | 26 GB |
| 200″ | 856 | 1 | 10 min | 16 GB |
| 300″ | 1953 | 10 | 17 min | 77 GB |

Peak memory includes the worker processes. Run time also depends on the
processor. Notebooks 01 and 02 run the same model selection on 200″ fields
and were not measured separately.

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

## How the package is organised

The source models are decided once, on the prior band (VIS by default, or
a NISP band): the profile class and shape of each source come either from
the MER catalog or, on VIS, from the model-selection tree, with an optional pass that frees the shapes and refines the positions
(`selection/`, `forced.fit_free_shapes`, `forced.refine_positions`).
Every other band then runs the same core recipe on those models:

1. build the band's Tractor image: data, inverse variance, PSF, WCS and the
   photometric scale (`tractor_image.make_tractor_image`);
2. copy the prior-band source models into the band, positions and shapes kept
   (`sources.clone_sources`);
3. freeze everything but the fluxes and solve (`forced.flux_tractor`,
   `forced.solve_fluxes`);
4. turn the solve's inverse variances into errors, optionally scaled by the
   residual scatter (`forced.flux_inverse_variance`, `forced.chi_inflation`).

Each band adds to that core what its data need:

- **NISP** runs the bare recipe; its errors are then scaled by point-source
  fits at empty positions (`calibrate.py`).
- **unWISE** fits a constant sky offset together with the fluxes and adds
  unWISE catalog sources outside the field, so that the wings of the
  6-arcsec PSF are modelled; errors are scaled by the residual scatter.
- **IRAC** uses a box-constrained optimizer and lets each source move by up
  to one native IRAC pixel before the final flux solve
  (`forced.fit_bounded_positions`); errors get the residual scaling and the
  PRF systematic.
- **A ground-based image** has its WCS refined on matched MER stars first,
  can have a background subtracted or a constant sky fitted, and takes one
  linearised recentring step (`forced.gauss_newton_recenter`); errors are
  the formal ones.

```
src/euclid_phot/        the package (import euclid_phot as ep)
  units.py  tractor_image.py  pixel_psf.py  sources.py  selection/
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
Notebooks 01-04 run the recipe step by step for one band before calling the
band's one-line fit.

Every push runs a check on GitHub (`.github/workflows/ci.yml`): the install
script, a lint pass, the public imports and an offline Tractor fit of a
synthetic image. It does not download survey data; re-run the notebooks
before merging changes to the fitting code.

## References

The image modelling uses [The Tractor](https://github.com/dstndstn/tractor)
([Lang, Hogg & Mykytyn 2016](https://ascl.net/1604.008)); forced photometry
with it is described in
[Lang, Hogg & Schlegel (2016)](https://arxiv.org/abs/1410.7397).
Euclid Q1 data are described in
[Euclid Collaboration: Aussel et al. (2025)](https://arxiv.org/abs/2503.15302)
and the MER catalog in
[Euclid Collaboration: Romelli et al. (2025)](https://arxiv.org/abs/2503.15305);
Euclid reference flux definitions are documented in the
[MER photometry cookbook](https://euclid.esac.esa.int/dr/q1/dpdd/merdpd/merphotometrycookbook.html).
The IRAC effective PRF follows [PRFMAP](https://github.com/cosmic-dawn/prfmap)
with the detector PRFs and flux corrections of the
[IRAC Instrument Handbook](https://irsa.ipac.caltech.edu/data/SPITZER/docs/irac/iracinstrumenthandbook/);
IRAC extinction coefficients are from
[Indebetouw et al. (2005)](https://ui.adsabs.harvard.edu/abs/2005ApJ...619..931I).
Please cite the relevant survey and method papers when using their data or methods.

## Authors

Nima Chartab, Shoubaneh Hemmati and Ian Lin, Euclid NASA Science Center at
IPAC (ENSCI).
