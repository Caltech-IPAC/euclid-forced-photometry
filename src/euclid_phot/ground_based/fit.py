"""Forced photometry of the VIS source models on a user's ground image.

The fit is the shared recipe of :mod:`euclid_phot.forced`. What is a ground
image's own:

- the PSF can be the supplied stamp, the nearest well-measured star in the
  image, or the average of those stars (``psf_mode``);
- an optional SEP sky map, for images whose sky was not subtracted
  (``background_method="sep"``);
- an optional single linearised recentering step for well-detected sources
  (``recenter_width_pixels``), since ground astrometry is looser than VIS;
- images larger than ``tile_arcsec`` are fitted in tiles with overlapping
  halos, each source measured in the tile core that contains it.

Fluxes and errors are in microJansky, as for every band.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table

from ..background import estimate_sep_background, sep_valid_mask
from ..forced import (
    flux_errors_ujy,
    flux_inverse_variance,
    flux_tractor,
    fluxes_ujy,
    gauss_newton_recenter,
    solve_fluxes,
)
from ..scene import footprint_mask
from ..sources import clone_sources
from ..tractor_image import make_tractor_image
from ..units import UJY_PER_NMGY, pixel_scale_arcsec
from .image import SmallImage
from .psf_validation import _radialize_noisy_wings, average_star_psf, reference_star_comparison


@dataclass(frozen=True)
class GroundFit:
    catalog: Table
    model: np.ndarray
    residual: np.ndarray
    background: np.ndarray
    image: SmallImage
    method: str


def select_mer_scene(mer: Table, image: SmallImage, *,
                     margin_arcsec: float = 5.0) -> Table:
    """MER rows with positive VIS flux whose profiles can put light on the
    image, brightest first."""
    for key in ("ra", "dec", "flux_vis_sersic"):
        if key not in mer.colnames:
            raise ValueError(f"MER catalog missing {key}")
    keep = footprint_mask(
        mer["ra"], mer["dec"], image.wcs, image.science_ujy.shape,
        margin_pix=margin_arcsec / pixel_scale_arcsec(image.wcs))
    flux = np.asarray(mer["flux_vis_sersic"], float)
    keep = np.flatnonzero(keep & np.isfinite(flux) & (flux > 0))
    keep = keep[np.argsort(flux[keep])[::-1]]
    if len(keep) == 0:
        raise ValueError("no MER VIS priors overlap the image")
    return mer[keep]


def fit_ground_image(image: SmallImage, scene: Table, vis_sources: list, *,
                     band: str, recenter_width_pixels: float = 0.0,
                     fit_constant_sky: bool = False, optimizer: str = "scaled"):
    """Fit the VIS models of ``scene`` to one ground image (or tile).

    Returns ``(catalog, model, residual)``; fluxes and formal 1-sigma errors
    are in microJansky. The image variance is treated as diagonal;
    calibration, PSF and morphology errors are excluded.

    ``optimizer="scaled"`` solves the linear flux problem exactly. Tractor's
    default LSQR update ("lsqr") stops short of convergence in a dense ground
    scene with a wide PSF: on the SDSS example its fluxes sit up to 2.6 sigma
    from the exact solution and move by up to 0.8 sigma under 1e-16 changes
    of the input.
    """
    if not np.isfinite(recenter_width_pixels) or recenter_width_pixels < 0:
        raise ValueError("recenter_width_pixels must be finite and nonnegative")
    if len(vis_sources) != len(scene):
        raise ValueError("VIS sources must align with MER rows")
    # 1. The Tractor image.
    sky_seed = (float(np.median(image.science_ujy[image.invvar_ujy > 0]))
                if fit_constant_sky else 0.0)
    tim = make_tractor_image(image.science_ujy, image.invvar_ujy, image.wcs,
                             image.psf, band=band, nmgy_to_data=UJY_PER_NMGY,
                             sky=sky_seed, name=f"user-{band}")
    # 2. VIS models, seeded with their VIS flux (MER flux if unusable).
    seeds = []
    for source, row in zip(vis_sources, scene):
        seed = float(source.getBrightness().getFlux("VIS"))
        if not np.isfinite(seed) or seed <= 0:
            seed = max(float(row["flux_vis_sersic"]) / UJY_PER_NMGY, 0.0)
        seeds.append(seed)
    sources = clone_sources(vis_sources, band, seed_nmgy=seeds)
    prior_ra = np.asarray([s.getPosition().ra for s in sources], float)
    prior_dec = np.asarray([s.getPosition().dec for s in sources], float)
    prior_x, prior_y = image.wcs.world_to_pixel_values(prior_ra, prior_dec)
    # 3. Flux-only solve; optionally one recentering step, then fluxes again.
    tractor = flux_tractor(tim, sources, optimizer=optimizer, fit_sky=fit_constant_sky)
    result = solve_fluxes(tractor, fit_sky=fit_constant_sky)
    shift_x = np.zeros(len(sources))
    shift_y = np.zeros(len(sources))
    if recenter_width_pixels > 0:
        shift_x, shift_y = gauss_newton_recenter(
            tractor, sources, flux_inverse_variance(result, tractor, len(sources)),
            band=band, wcs=image.wcs, width_pixels=recenter_width_pixels,
            psf_shape=image.psf.shape)
        if np.any(shift_x) or np.any(shift_y):
            result = solve_fluxes(tractor, fit_sky=fit_constant_sky)
    # 4. Fluxes and formal errors.
    flux = fluxes_ujy(sources, band)
    error = flux_errors_ujy(flux_inverse_variance(result, tractor, len(sources)))
    model = np.asarray(tractor.getModelImage(0), float)
    residual = image.science_ujy - model
    x, y = prior_x + shift_x, prior_y + shift_y
    fitted = image.wcs.pixel_to_world(x, y)
    catalog = Table()
    catalog["object_id"] = scene["object_id"]
    catalog["ra"] = fitted.ra.deg
    catalog["dec"] = fitted.dec.deg
    catalog["prior_ra"] = prior_ra
    catalog["prior_dec"] = prior_dec
    catalog["prior_x"] = prior_x
    catalog["prior_y"] = prior_y
    catalog["shift_x_pix"] = shift_x
    catalog["shift_y_pix"] = shift_y
    catalog["x"] = x
    catalog["y"] = y
    catalog["model_class"] = [type(s).__name__ for s in sources]
    catalog[f"flux_{band}_ujy"] = flux
    catalog[f"flux_err_{band}_ujy"] = error
    catalog["in_image"] = ((x >= 0) & (x < image.science_ujy.shape[1]) &
                           (y >= 0) & (y < image.science_ujy.shape[0]))
    catalog.meta["BAND"] = band
    catalog.meta["SKYUJY"] = float(tim.sky.getValue())
    catalog.meta["NRECENT"] = int(np.count_nonzero((shift_x != 0) | (shift_y != 0)))
    catalog.meta["ERRMODEL"] = "formal Tractor errors from supplied diagonal variance"
    return catalog, model, residual


def _background(image: SmallImage, method: str, *, mesh_arcsec: float = 20.0,
                clip_mad: float = 5.0) -> np.ndarray:
    if method == "none":
        return np.zeros(image.science_ujy.shape, dtype=np.float32)
    if method != "sep":
        raise ValueError("background_method must be 'none' or 'sep'")
    valid = (image.invvar_ujy > 0) & np.isfinite(image.science_ujy)
    sky_valid, _ = sep_valid_mask(image.science_ujy, valid, clip_mad)
    return estimate_sep_background(image.science_ujy, sky_valid,
                                   pixel_scale_arcsec(image.wcs),
                                   mesh_arcsec=mesh_arcsec, filter_size=3).image


def _psf_for_position(image, assessment, mode, x, y, excluded):
    """The supplied PSF, or the nearest usable star with cleaned wings."""
    if mode == "supplied":
        return image.psf
    excluded = set(excluded)
    candidates = [i for i in range(len(assessment.stars)) if i not in excluded]
    if not candidates:
        raise ValueError("No usable image stars remain for the empirical PSF")
    stars = assessment.stars
    candidates.sort(key=lambda i: (float(stars[i]["x"]) - x) ** 2 +
                                  (float(stars[i]["y"]) - y) ** 2)
    for i in candidates:
        pair = reference_star_comparison(
            image, float(stars[i]["x"]), float(stars[i]["y"]))
        if pair is None:
            continue
        try:
            return _radialize_noisy_wings(np.asarray(pair[0], float))
        except ValueError:
            continue
    raise ValueError("No selected star yields a usable local PSF stamp")


def fit_ground(image: SmallImage, vis_result, *, band: str | None = None,
               background_method: str = "none", tile_arcsec: float = 200.0,
               halo_arcsec: float = 5.0, recenter_width_pixels: float = 2.0,
               fit_constant_sky: bool = False, psf_mode: str = "supplied",
               psf_assessment=None, excluded_psf_stars=(),
               optimizer: str = "scaled") -> GroundFit:
    """Fit VIS shapes to ground pixels; tile only when the image exceeds 200″.

    Every tile includes halo sources; each source measurement belongs to one
    core. The VIS morphology fit itself is never tiled here.
    """
    if not 0 <= recenter_width_pixels <= 2.0:
        raise ValueError("recenter_width_pixels must be between 0 and 2 pixels")
    if tile_arcsec <= 0 or halo_arcsec < 0:
        raise ValueError("tile/halo sizes must be positive/nonnegative")
    if psf_mode not in {"supplied", "nearest_star", "avg_star"}:
        raise ValueError("psf_mode must be 'supplied', 'nearest_star', or 'avg_star'")
    if psf_mode != "supplied" and psf_assessment is None:
        raise ValueError(f"{psf_mode} PSF needs a PSF assessment")
    averaged_psf = (average_star_psf(image, psf_assessment.stars,
                                     excluded_psf_stars)
                    if psf_mode == "avg_star" else None)
    band = band or image.band or "ground"
    sky = _background(image, background_method)
    fit_image = replace(image, science_ujy=image.science_ujy - sky)
    mer = vis_result.mer_cat
    scene = select_mer_scene(mer, image, margin_arcsec=max(5.0, halo_arcsec))
    if len(vis_result.sources) != len(mer):
        raise ValueError("VIS sources and MER rows are not aligned")
    by_id = {obj_id: i for i, obj_id in enumerate(mer["object_id"])}
    sources = [vis_result.sources[by_id[obj_id]] for obj_id in scene["object_id"]]
    ny, nx = image.science_ujy.shape
    scale = pixel_scale_arcsec(image.wcs)
    tile_pixels = max(32, int(np.ceil(tile_arcsec / scale)))
    halo_pixels = int(np.ceil(halo_arcsec / scale))
    if max(nx, ny) <= tile_pixels:
        fit_psf = (averaged_psf if averaged_psf is not None else
                   _psf_for_position(image, psf_assessment, psf_mode,
                                     (nx - 1) / 2, (ny - 1) / 2,
                                     excluded_psf_stars))
        catalog, source_model, _ = fit_ground_image(
            replace(fit_image, psf=fit_psf), scene, sources, band=band,
            recenter_width_pixels=recenter_width_pixels,
            fit_constant_sky=fit_constant_sky, optimizer=optimizer)
        catalog["tile_id"] = np.zeros(len(catalog), dtype=int)
    else:
        catalog, source_model = _fit_tiles(
            fit_image, scene, sources, band, tile_pixels, halo_pixels,
            recenter_width_pixels, fit_constant_sky,
            psf_mode, psf_assessment, excluded_psf_stars, averaged_psf, optimizer)
    model = np.asarray(source_model, float) + sky
    residual = image.science_ujy - model
    catalog.meta["BAND"] = band
    catalog.meta["BKGTYPE"] = background_method
    catalog.meta["NTILES"] = int(np.ceil(nx / tile_pixels) * np.ceil(ny / tile_pixels))
    catalog.meta["RECWPIX"] = recenter_width_pixels
    catalog.meta["PSFMODE"] = psf_mode
    return GroundFit(catalog, model, residual, sky, image, background_method)


def _fit_tiles(image, scene, sources, band, tile_pixels, halo_pixels,
               recenter_width_pixels, fit_constant_sky, psf_mode,
               psf_assessment, excluded_psf_stars, averaged_psf=None,
               optimizer="scaled"):
    """Core ownership prevents duplicate rows; halos are used only for fitting."""
    ny, nx = image.science_ujy.shape
    x, y = image.wcs.world_to_pixel_values(np.asarray(scene["ra"], float),
                                            np.asarray(scene["dec"], float))
    result_rows = {}
    model = np.zeros((ny, nx), dtype=np.float32)
    ncols = int(np.ceil(nx / tile_pixels))
    tile_count = int(np.ceil(ny / tile_pixels)) * ncols
    tile_number = 0
    for y0 in range(0, ny, tile_pixels):
        for x0 in range(0, nx, tile_pixels):
            tile_number += 1
            x1, y1 = min(nx, x0 + tile_pixels), min(ny, y0 + tile_pixels)
            fx0, fy0 = max(0, x0 - halo_pixels), max(0, y0 - halo_pixels)
            fx1, fy1 = min(nx, x1 + halo_pixels), min(ny, y1 + halo_pixels)
            core = (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
            included = np.flatnonzero(
                (x > fx0 - halo_pixels) & (x < fx1 + halo_pixels) &
                (y > fy0 - halo_pixels) & (y < fy1 + halo_pixels))
            print(f"Ground tile {tile_number}/{tile_count}: {len(included)} sources", flush=True)
            if not len(included):
                continue
            tile_psf = (averaged_psf if averaged_psf is not None else
                        _psf_for_position(image, psf_assessment, psf_mode,
                                          (x0 + x1 - 1) / 2, (y0 + y1 - 1) / 2,
                                          excluded_psf_stars))
            part = SmallImage(
                image.science_ujy[fy0:fy1, fx0:fx1],
                image.invvar_ujy[fy0:fy1, fx0:fx1],
                image.wcs.slice((slice(fy0, fy1), slice(fx0, fx1))),
                tile_psf, image.band)
            local, tile_model, _ = fit_ground_image(
                part, scene[included], [sources[i] for i in included],
                band=band, recenter_width_pixels=recenter_width_pixels,
                fit_constant_sky=fit_constant_sky, optimizer=optimizer)
            model[y0:y1, x0:x1] = tile_model[y0-fy0:y1-fy0, x0-fx0:x1-fx0]
            for i, row in enumerate(included):
                if core[row]:
                    record = {name: local[name][i] for name in local.colnames}
                    record["x"] = float(record["x"]) + fx0
                    record["y"] = float(record["y"]) + fy0
                    record["prior_x"] = float(record["prior_x"]) + fx0
                    record["prior_y"] = float(record["prior_y"]) + fy0
                    record["tile_id"] = (y0 // tile_pixels) * ncols + x0 // tile_pixels
                    result_rows[row] = record
    if not result_rows:
        raise ValueError("No VIS priors are inside the ground-image tiles")
    catalog = Table(rows=[result_rows[i] for i in sorted(result_rows)])
    return catalog, model


def save_fit(fit: GroundFit, output_dir: str | Path) -> tuple[Path, Path]:
    """Save measured fluxes and the exact image, sky, model and residual."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cat_path = out / "forced_catalog.ecsv"
    image_path = out / "fit_images.fits"
    fit.catalog.write(cat_path, format="ascii.ecsv", overwrite=True)
    header = fit.image.wcs.to_header(relax=True)
    hdus = [fits.PrimaryHDU()]
    for name, array, unit in (
        ("SCIENCE", fit.image.science_ujy, "uJy/pixel"),
        ("INVVAR", fit.image.invvar_ujy, "1/uJy^2"),
        ("BACKGROUND", fit.background, "uJy/pixel"),
        ("MODEL", fit.model, "uJy/pixel"),
        ("RESIDUAL", fit.residual, "uJy/pixel"),
    ):
        hdu = fits.ImageHDU(np.asarray(array, dtype="f4"), header=header, name=name)
        hdu.header["BUNIT"] = unit
        hdus.append(hdu)
    fits.HDUList(hdus).writeto(image_path, overwrite=True)
    return cat_path, image_path
