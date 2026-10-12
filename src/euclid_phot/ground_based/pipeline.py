"""Small public workflow around the standard VIS fit and ground-image fitter."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

from euclid_phot import (
    extract_grid_psf, fetch_cutout,
    query_mer_catalog, run_forced_photometry,
)
from euclid_phot.irac.background import estimate_sep_background
from .image import SmallImage
from .psf_validation import (
    assess_supplied_psf, average_star_psf, reference_star_comparison,
)
from .astrometry import register_wcs_with_mer_stars
from .tractor_fit import (
    fit_tractor_ground_image, select_mer_scene, sep_background_valid_mask,
)


@dataclass(frozen=True)
class GroundInputs:
    image: SmallImage
    science_wcs: WCS
    pixel_scale_arcsec: float


@dataclass(frozen=True)
class EuclidInputs:
    mer: Table
    vis_cutout: object
    ra: float
    dec: float
    size_arcsec: float
    mer_path: Path


@dataclass(frozen=True)
class GroundFit:
    catalog: Table
    model: np.ndarray
    residual: np.ndarray
    background: np.ndarray
    image: SmallImage
    method: str


def read_fits_image(path: str | Path, ext: int | str = 0) -> tuple[np.ndarray, fits.Header]:
    """Read one explicitly selected FITS image plane and its header."""
    with fits.open(path, memmap=False) as hdus:
        hdu = hdus[ext]
        if hdu.data is None:
            raise ValueError(f"{path}[{ext!r}] contains no image data")
        return np.asarray(hdu.data, dtype=float).copy(), hdu.header.copy()


def make_image(science, uncertainty, psf, science_unit: u.UnitBase | u.Quantity,
               science_wcs: WCS, *, band: str = "ground") -> GroundInputs:
    """Validate user arrays and convert science and 1-sigma RMS to nJy/pixel.

    ``science_unit`` is the unit of one input pixel value, e.g.
    ``u.nanomaggy`` for SDSS or ``u.uJy`` for a microJy image. A calibrated
    DN image can instead supply a quantity such as ``4.7 * u.nJy`` per DN.
    The uncertainty array must be RMS in the same native units. ``science_wcs``
    is the celestial WCS of the science array; ``band`` is an optional label.
    """
    declared_unit = (1 * science_unit if isinstance(science_unit, u.UnitBase)
                     else u.Quantity(science_unit))
    # Astropy keeps maggy as a separate physical type, so apply its AB
    # definition explicitly: 1 nanomaggy = 3631 nJy.
    unit = (declared_unit.to_value(u.nanomaggy) * 3631.0
            if declared_unit.unit.is_equivalent(u.maggy)
            else declared_unit.to_value(u.nJy))
    if not np.isfinite(unit) or unit <= 0 or np.ndim(unit):
        raise ValueError("science_unit must be one positive flux per image value")
    data = np.asarray(science, dtype=float)
    rms = np.asarray(uncertainty, dtype=float)
    stamp = np.asarray(psf, dtype=float)
    if data.ndim != 2 or rms.shape != data.shape:
        raise ValueError("science and 1-sigma uncertainty must be aligned 2-D arrays")
    if (stamp.ndim != 2 or min(stamp.shape) < 3 or
            any(side % 2 == 0 for side in stamp.shape) or
            not np.all(np.isfinite(stamp)) or np.any(stamp < 0) or stamp.sum() <= 0):
        raise ValueError("PSF must be a finite, nonnegative, odd-sized 2-D stamp")
    if not isinstance(science_wcs, WCS) or not science_wcs.has_celestial:
        raise ValueError("science_wcs must be a celestial astropy WCS")
    wcs = science_wcs.celestial
    if not wcs.has_celestial or not np.all(np.isfinite(wcs.pixel_scale_matrix)):
        raise ValueError("science_wcs needs a finite celestial pixel scale")
    valid = np.isfinite(data) & np.isfinite(rms) & (rms > 0)
    if not np.any(valid):
        raise ValueError("no usable science pixels with positive uncertainty")
    invvar = np.zeros(data.shape, dtype=float)
    invvar[valid] = 1.0 / (rms[valid] * unit) ** 2
    image = SmallImage(np.where(np.isfinite(data), data * unit, 0.0),
                       invvar, wcs, stamp / stamp.sum(), band)
    scale = float(np.sqrt(abs(np.linalg.det(wcs.pixel_scale_matrix))) * 3600)
    return GroundInputs(image, wcs, scale)


def prepare_euclid(image: SmallImage, data_dir: str | Path,
                   *, padding_arcsec: float = 5.0) -> EuclidInputs:
    """Reuse local MER/VIS/GRID-PSF inputs and fetch only missing files."""
    data_dir = Path(data_dir)
    ny, nx = image.science_njy.shape
    center = image.wcs.pixel_to_world((nx - 1) / 2, (ny - 1) / 2)
    ra, dec = round(center.ra.deg, 4), round(center.dec.deg, 4)
    if padding_arcsec < 0:
        raise ValueError("padding_arcsec must be nonnegative")
    edge_x = np.linspace(-.5, nx-.5, 5)
    edge_y = np.linspace(-.5, ny-.5, 5)
    xx = np.r_[edge_x, edge_x, np.full(5, -.5), np.full(5, nx-.5)]
    yy = np.r_[np.full(5, -.5), np.full(5, ny-.5), edge_y, edge_y]
    edge_world = image.wcs.pixel_to_world(xx, yy)
    east, north = center.spherical_offsets_to(edge_world)
    raw_side = 2 * (max(np.max(np.abs(east.arcsec)),
                        np.max(np.abs(north.arcsec))) + padding_arcsec)
    # A square aligned with the ground image is insufficient when its WCS is
    # rotated. Use the bounding sky square, including a small source margin.
    size = float(10.0 * np.ceil(raw_side / 10))
    # Same cache name as run_forced_photometry, so either can reuse it.
    mer_path = data_dir / f"mer_catalog_{ra:.4f}_{dec:.4f}_{int(round(size))}.fits"
    if mer_path.is_file():
        mer = Table.read(mer_path)
        print("MER: local cache", mer_path)
    else:
        print("MER: querying Euclid for the image footprint", flush=True)
        mer = query_mer_catalog(ra, dec, size / 7200)
        if not len(mer):
            raise ValueError("No MER sources found in this image footprint")
        mer_path.parent.mkdir(parents=True, exist_ok=True)
        mer.write(mer_path, overwrite=True)
        print("MER: saved", mer_path)
    # Both functions read their local cache first and fetch missing data.
    vis = fetch_cutout("VIS", ra, dec, size, data_dir=data_dir / "cutouts",
                       with_flag=True)
    print("VIS science/RMS/flag: ready")
    psf = extract_grid_psf(
        "VIS", ra, dec, tile_id=vis.header.get("MERTILE"),
        radius_arcsec=max(60.0, 0.75 * size), data_dir=data_dir / "psf")
    print("VIS GRID PSF: ready")
    if not len(psf["stamps"]):
        raise ValueError("No VIS GRID-PSF stamps cover this field")
    return EuclidInputs(mer, vis, ra, dec, size, mer_path)


def fit_vis(inputs: EuclidInputs, data_dir: str | Path, *, n_workers: int = 8,
            existing=None):
    """Fit the same VIS prior as notebooks 01–03; optionally reuse a live result.

    An in-memory result from another notebook may be passed as ``existing``.
    Cached image/PSF/catalog files are reused automatically; the standard
    pipeline does not currently persist its fitted Tractor source list.
    """
    if existing is not None:
        if len(existing.sources) != len(inputs.mer):
            raise ValueError("existing VIS result has a different MER source count")
        if not np.array_equal(np.asarray(existing.mer_cat["object_id"]),
                              np.asarray(inputs.mer["object_id"])):
            raise ValueError("existing VIS result has different MER source IDs")
        print("VIS morphology: reusing supplied in-memory result")
        return existing
    return run_forced_photometry(
        inputs.ra, inputs.dec, inputs.size_arcsec,
        prior={"band": "VIS", "objects": "mer", "model_selection": "tree"},
        target_bands={"euclid": (), "wise": ()},
        psf_product="grid", persource_psf=True, mask_bright_stars=True,
        calibrate_errors=False, data_dir=Path(data_dir), n_workers=n_workers,
        mer_catalog=inputs.mer,
    )


def register_image(image: SmallImage, mer: Table, *, min_matches: int = 10):
    """Refine a supplied near-correct WCS using matched MER stars."""
    return register_wcs_with_mer_stars(image, mer, min_matches=min_matches)


def assess_psf(image: SmallImage, registration):
    """Compare the supplied PSF with isolated stars in the image."""
    return assess_supplied_psf(image, registration.matches)


def _background(image: SmallImage, method: str, *, mesh_arcsec: float = 20.0,
                clip_mad: float = 5.0) -> np.ndarray:
    if method == "none":
        return np.zeros(image.science_njy.shape, dtype=np.float32)
    if method != "sep":
        raise ValueError("background_method must be 'none' or 'sep'")
    valid = (image.invvar_njy > 0) & np.isfinite(image.science_njy)
    sky_valid, _ = sep_background_valid_mask(image.science_njy, valid, clip_mad)
    scale = float(np.mean(proj_plane_pixel_scales(image.wcs))) * 3600
    return estimate_sep_background(image.science_njy, sky_valid, scale,
                                   mesh_arcsec=mesh_arcsec, filter_size=3).image


def _psf_for_position(image, assessment, mode, x, y, excluded):
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
        stamp = np.asarray(pair[0], float).copy()
        yy, xx = np.indices(stamp.shape)
        radius = np.hypot(xx - (stamp.shape[1] - 1) / 2,
                          yy - (stamp.shape[0] - 1) / 2)
        ring = np.floor(radius).astype(int)
        for r in range(8, int(ring.max()) + 1):
            pixels = ring == r
            stamp[pixels] = np.median(stamp[pixels])
        stamp = np.maximum(stamp, 0)
        if stamp.sum() > 0:
            return stamp / stamp.sum()
    raise ValueError("No selected star yields a usable local PSF stamp")


def fit_ground(image: SmallImage, vis_result, *, band: str | None = None,
               background_method: str = "none", tile_arcsec: float = 200.0,
               halo_arcsec: float = 5.0, recenter_width_pixels: float = 2.0,
               fit_constant_sky: bool = False, psf_mode: str = "supplied",
               psf_assessment=None, excluded_psf_stars=()) -> GroundFit:
    """Fit VIS shapes to ground pixels; tile only when the image exceeds 200″.

    The legacy ground Tractor solver is shared while its other callers remain.
    Every tile includes halo sources; each source measurement belongs to one
    core.  The public VIS morphology fit is never tiled here.
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
    fit_image = replace(image, science_njy=image.science_njy - sky)
    mer = vis_result.mer_cat
    scene = select_mer_scene(mer, image, margin_arcsec=max(5.0, halo_arcsec),
                             min_vis_flux_ujy=0.0, max_vis_flux_ujy=1e9)
    if len(vis_result.sources) != len(mer):
        raise ValueError("VIS sources and MER rows are not aligned")
    by_id = {obj_id: i for i, obj_id in enumerate(mer["object_id"])}
    sources = [vis_result.sources[by_id[obj_id]] for obj_id in scene["object_id"]]
    ny, nx = image.science_njy.shape
    scale = float(np.mean(proj_plane_pixel_scales(image.wcs))) * 3600
    tile_pixels = max(32, int(np.ceil(tile_arcsec / scale)))
    halo_pixels = int(np.ceil(halo_arcsec / scale))
    if max(nx, ny) <= tile_pixels:
        fit_psf = (averaged_psf if averaged_psf is not None else
                   _psf_for_position(image, psf_assessment, psf_mode,
                                     (nx - 1) / 2, (ny - 1) / 2,
                                     excluded_psf_stars))
        one_image = replace(fit_image, psf=fit_psf)
        catalog, source_model, _ = fit_tractor_ground_image(
            one_image, scene, band=band, vis_sources=sources,
            recenter_width_pixels=recenter_width_pixels,
            fit_constant_sky=fit_constant_sky)
        catalog["tile_id"] = np.zeros(len(catalog), dtype=int)
    else:
        catalog, source_model = _fit_tiles(
            fit_image, scene, sources, band, tile_pixels, halo_pixels,
            recenter_width_pixels, fit_constant_sky,
            psf_mode, psf_assessment, excluded_psf_stars, averaged_psf)
    model = np.asarray(source_model, float) + sky
    residual = image.science_njy - model
    catalog.meta["BKGTYPE"] = background_method
    catalog.meta["NTILES"] = int(np.ceil(nx / tile_pixels) * np.ceil(ny / tile_pixels))
    catalog.meta["RECWPIX"] = recenter_width_pixels
    catalog.meta["PSFMODE"] = psf_mode
    return GroundFit(catalog, model, residual, sky, image, background_method)


def _fit_tiles(image, scene, sources, band, tile_pixels, halo_pixels,
               recenter_width_pixels, fit_constant_sky, psf_mode,
               psf_assessment, excluded_psf_stars, averaged_psf=None):
    """Core ownership prevents duplicate rows; halos are used only for fitting."""
    ny, nx = image.science_njy.shape
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
                image.science_njy[fy0:fy1, fx0:fx1],
                image.invvar_njy[fy0:fy1, fx0:fx1],
                image.wcs.slice((slice(fy0, fy1), slice(fx0, fx1))),
                tile_psf, image.band)
            local, tile_model, _ = fit_tractor_ground_image(
                part, scene[included], band=band,
                vis_sources=[sources[i] for i in included],
                recenter_width_pixels=recenter_width_pixels,
                fit_constant_sky=fit_constant_sky)
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
        ("SCIENCE", fit.image.science_njy, "nJy/pixel"),
        ("INVVAR", fit.image.invvar_njy, "1/nJy^2"),
        ("BACKGROUND", fit.background, "nJy/pixel"),
        ("MODEL", fit.model, "nJy/pixel"),
        ("RESIDUAL", fit.residual, "nJy/pixel"),
    ):
        hdu = fits.ImageHDU(np.asarray(array, dtype="f4"), header=header, name=name)
        hdu.header["BUNIT"] = unit
        hdus.append(hdu)
    fits.HDUList(hdus).writeto(image_path, overwrite=True)
    return cat_path, image_path


def compare_sdss_g(measured: Table, reference: Table,
                   *, max_sep_arcsec: float = 0.7) -> Table:
    """Independent SDSS check: PSF flux for stars, cModel flux for galaxies."""
    needed = {"ra", "dec", "objID", "type", "clean", "psfFlux_g",
              "psfFluxIvar_g", "cModelFlux_g", "cModelFluxIvar_g"}
    missing = needed - set(reference.colnames)
    if missing:
        raise ValueError(f"SDSS catalog lacks columns {sorted(missing)}")
    ref_coords = SkyCoord(np.asarray(reference["ra"], float) * u.deg,
                          np.asarray(reference["dec"], float) * u.deg)
    fit_coords = SkyCoord(np.asarray(measured["ra"], float) * u.deg,
                          np.asarray(measured["dec"], float) * u.deg)
    nearest, separation, _ = ref_coords.match_to_catalog_sky(fit_coords)
    unique = np.zeros(len(reference), bool)
    claimed = set()
    for i in np.argsort(separation.arcsec):
        j = int(nearest[i])
        if j not in claimed:
            unique[i] = True
            claimed.add(j)
    is_star = np.asarray(reference["type"], int) == 6
    native_flux = np.where(is_star, reference["psfFlux_g"], reference["cModelFlux_g"])
    native_ivar = np.where(is_star, reference["psfFluxIvar_g"], reference["cModelFluxIvar_g"])
    ref_flux = np.asarray(native_flux, float) * 3631.0
    with np.errstate(divide="ignore", invalid="ignore"):
        ref_err = 3631.0 / np.sqrt(np.asarray(native_ivar, float))
    fitted_flux = np.asarray(measured["flux_njy"][nearest], float)
    fitted_err = np.asarray(measured["fluxerr_njy"][nearest], float)
    kind = np.asarray(reference["type"], int)
    good = (unique & np.isin(kind, [3, 6]) & (separation.arcsec < max_sep_arcsec) &
            (np.asarray(reference["clean"], int) == 1) &
            np.asarray(measured["in_image"][nearest], bool) &
            np.isfinite(ref_flux) & np.isfinite(ref_err) &
            np.isfinite(fitted_flux) & np.isfinite(fitted_err) &
            (ref_flux > 0) & (fitted_flux > 0) &
            (ref_err > 0) & (fitted_err > 0) &
            (ref_flux / ref_err > 3) & (fitted_flux / fitted_err > 3))
    comparison = Table()
    comparison["reference_id"] = reference["objID"]
    comparison["euclid_object_id"] = measured["object_id"][nearest]
    comparison["source_type"] = np.where(is_star, "star", "galaxy")
    comparison["unique_match"] = unique
    comparison["reference_type"] = kind
    comparison["reference_clean"] = np.asarray(reference["clean"], int)
    comparison["in_image"] = np.asarray(measured["in_image"][nearest], bool)
    comparison["separation_arcsec"] = separation.arcsec
    comparison["reference_flux_njy"] = ref_flux
    comparison["reference_fluxerr_njy"] = ref_err
    comparison["forced_flux_njy"] = fitted_flux
    comparison["forced_fluxerr_njy"] = fitted_err
    comparison["accepted"] = good
    return comparison
