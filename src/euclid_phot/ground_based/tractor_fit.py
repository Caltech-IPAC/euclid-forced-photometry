"""Ground-image scene selection and Tractor fitting for VIS priors."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
from astropy.table import Table
from astropy.wcs.utils import proj_plane_pixel_scales
from scipy.stats import median_abs_deviation
from tractor import ConstantSky, Image, LinearPhotoCal, NanoMaggies, Tractor
from tractor.psf import PixelizedPSF
from tractor.wcs import ConstantFitsWcs

from euclid_phot.images import AstropyWCSAdapter

from .image import SmallImage


NANOMAGGY_NJY = 3631.0


def sep_background_valid_mask(
    data: np.ndarray, valid_mask: np.ndarray, clip_sigma: float,
) -> tuple[np.ndarray, float]:
    """Exclude pixels above the median plus N median absolute deviations.

    The mask is used only to estimate the SEP sky, not as a Tractor pixel veto.
    """
    data = np.asarray(data)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(data)
    if data.ndim != 2 or valid.shape != data.shape or not np.any(valid):
        raise ValueError("background image and valid mask need finite 2-D pixels")
    if not np.isfinite(clip_sigma) or clip_sigma <= 0:
        raise ValueError("background clip sigma must be finite and positive")
    values = data[valid]
    mad = median_abs_deviation(values)
    threshold = float(np.median(values) + clip_sigma * mad)
    return valid & (data <= threshold), threshold


def select_mer_scene(
    mer: Table, image: SmallImage, *, margin_arcsec: float = 5.0,
    min_vis_flux_ujy: float = 0.05, max_vis_flux_ujy: float = 20.0,
    max_sources: int | None = None,
) -> Table:
    """Keep bright VIS priors whose profiles can overlap the target image.

    Flux thresholds and any source cap are explicit example choices, not
    properties inferred from the mock truth catalog.
    """
    for key in ("ra", "dec", "flux_vis_sersic"):
        if key not in mer.colnames:
            raise ValueError(f"MER catalog missing {key}")
    if margin_arcsec < 0 or min_vis_flux_ujy < 0 or max_vis_flux_ujy <= min_vis_flux_ujy:
        raise ValueError("invalid scene selection limits")
    if max_sources is not None and max_sources < 1:
        raise ValueError("max_sources must be positive")
    x, y = image.wcs.world_to_pixel_values(np.asarray(mer["ra"], float),
                                            np.asarray(mer["dec"], float))
    flux = np.asarray(mer["flux_vis_sersic"], float)
    pixel_scale = float(np.mean(proj_plane_pixel_scales(image.wcs))) * 3600
    margin = margin_arcsec / pixel_scale
    height, width = image.science_njy.shape
    keep = np.flatnonzero(
        np.isfinite(x) & np.isfinite(y) & np.isfinite(flux) &
        (x > -margin) & (x < width + margin) &
        (y > -margin) & (y < height + margin) &
        (flux > min_vis_flux_ujy) & (flux < max_vis_flux_ujy))
    keep = keep[np.argsort(flux[keep])[::-1]]
    if max_sources is not None:
        keep = keep[:max_sources]
    if len(keep) == 0:
        raise ValueError("no MER VIS priors overlap the image under these limits")
    return mer[keep]


def fit_tractor_ground_image(
    image: SmallImage, mer_scene: Table, *, vis_sources: list,
    band: str = "ground",
    recenter_width_pixels: float = 0.0,
    fit_constant_sky: bool = True,
) -> tuple[Table, np.ndarray, np.ndarray]:
    """Fit fixed MER/VIS source fluxes, optionally with one constant sky.

    Flux and formal 1-sigma errors are in nJy. The image's variance is
    treated as diagonal; calibration, PSF, and morphology errors are excluded.
    When requested, a first flux fit estimates small centroid corrections for
    well detected sources. Each axis is bounded to half the stated total width,
    then a second forced fit measures fluxes at those corrected positions.
    """
    if not np.isfinite(recenter_width_pixels) or recenter_width_pixels < 0:
        raise ValueError("recenter_width_pixels must be finite and nonnegative")
    if len(vis_sources) != len(mer_scene):
        raise ValueError("VIS sources must align with MER rows")
    sources = [deepcopy(source) for source in vis_sources]
    for source, mer_row in zip(sources, mer_scene, strict=True):
        seed = float(source.getBrightness().getFlux("VIS"))
        if not np.isfinite(seed) or seed <= 0:
            seed = max(float(mer_row["flux_vis_sersic"]) / 3.631, 0.0)
        source.brightness = NanoMaggies(**{band: seed})
        if hasattr(source, "halfsize"):
            source.halfsize = None
    for source in sources:
        source.freezeAllBut("brightness")
    sky_seed = (float(np.median(image.science_njy[image.invvar_njy > 0]))
                if fit_constant_sky else 0.0)
    tim = Image(
        data=image.science_njy, invvar=image.invvar_njy,
        psf=PixelizedPSF(image.psf),
        wcs=ConstantFitsWcs(AstropyWCSAdapter(image.wcs)),
        photocal=LinearPhotoCal(NANOMAGGY_NJY, band=band),
        sky=ConstantSky(sky_seed), name=f"user-{band}",
    )
    tim.freezeAllParams()
    if fit_constant_sky:
        tim.thawParam("sky")
    tractor = Tractor([tim], sources)
    fit = tractor.optimize_forced_photometry(
        minsb=0.0, mindlnp=1.0, sky=fit_constant_sky, variance=True,
        shared_params=True)
    prior_ra = np.asarray([source.getPosition().ra for source in sources], float)
    prior_dec = np.asarray([source.getPosition().dec for source in sources], float)
    prior_x, prior_y = image.wcs.world_to_pixel_values(prior_ra, prior_dec)
    shift_x = np.zeros(len(sources), dtype=float)
    shift_y = np.zeros(len(sources), dtype=float)
    if recenter_width_pixels > 0:
        initial_ivar = np.asarray(fit.IV, float)
        initial_flux = np.asarray(
            [source.getBrightness().getFlux(band) for source in sources]) * NANOMAGGY_NJY
        initial_snr = np.where(
            initial_ivar > 0,
            initial_flux * np.sqrt(np.maximum(initial_ivar, 0)) / NANOMAGGY_NJY,
            0.0)
        initial_residual = image.science_njy - np.asarray(tractor.getModelImage(0), float)
        half_width = recenter_width_pixels / 2.0
        # A local Gauss-Newton centroid step uses the neighbor-subtracted
        # residual of the initial joint fit. Faint or unconstrained sources
        # retain their MER positions instead of fitting noise centroids.
        for i, source in enumerate(sources):
            if not np.isfinite(initial_snr[i]) or initial_snr[i] < 10:
                continue
            patch = source.getModelPatch(tim)
            if patch is None or patch.patch is None:
                continue
            radius = min(16, max(8, min(image.psf.shape) // 4))
            x0 = max(0, patch.x0, int(np.floor(prior_x[i])) - radius)
            x1 = min(image.science_njy.shape[1], patch.x1,
                     int(np.floor(prior_x[i])) + radius + 1)
            y0 = max(0, patch.y0, int(np.floor(prior_y[i])) - radius)
            y1 = min(image.science_njy.shape[0], patch.y1,
                     int(np.floor(prior_y[i])) + radius + 1)
            if x1 - x0 < 3 or y1 - y0 < 3:
                continue
            model_patch = np.asarray(
                patch.patch[y0 - patch.y0:y1 - patch.y0,
                            x0 - patch.x0:x1 - patch.x0], float)
            gradient_y, gradient_x = np.gradient(model_patch)
            jac_x, jac_y = -gradient_x, -gradient_y
            weight = image.invvar_njy[y0:y1, x0:x1]
            residual = initial_residual[y0:y1, x0:x1]
            if np.count_nonzero(weight > 0) < 9:
                continue
            normal = np.array([
                [np.sum(weight * jac_x**2), np.sum(weight * jac_x * jac_y)],
                [np.sum(weight * jac_x * jac_y), np.sum(weight * jac_y**2)],
            ])
            rhs = np.array([np.sum(weight * jac_x * residual),
                            np.sum(weight * jac_y * residual)])
            if not np.all(np.isfinite(normal)) or np.linalg.cond(normal) > 1e8:
                continue
            proposed = np.clip(np.linalg.solve(normal, rhs), -half_width, half_width)
            delta_chi2 = 2 * rhs @ proposed - proposed @ normal @ proposed
            if not np.isfinite(delta_chi2) or delta_chi2 < 9:
                continue
            world = image.wcs.pixel_to_world(prior_x[i] + proposed[0],
                                             prior_y[i] + proposed[1])
            source.setPosition(type(source.getPosition())(world.ra.deg, world.dec.deg))
            shift_x[i], shift_y[i] = proposed
        if np.any(shift_x) or np.any(shift_y):
            fit = tractor.optimize_forced_photometry(
                minsb=0.0, mindlnp=1.0, sky=fit_constant_sky, variance=True,
                shared_params=True)
    inverse_variance = np.asarray(fit.IV, float)
    if len(inverse_variance) != len(sources):
        raise RuntimeError("Tractor returned an unexpected number of flux errors")
    flux = np.asarray([source.getBrightness().getFlux(band) for source in sources]) * NANOMAGGY_NJY
    with np.errstate(divide="ignore", invalid="ignore"):
        error = np.where(inverse_variance > 0,
                         NANOMAGGY_NJY / np.sqrt(inverse_variance), np.nan)
    model = np.asarray(tractor.getModelImage(0), float)
    residual = image.science_njy - model
    x, y = prior_x + shift_x, prior_y + shift_y
    fitted_world = image.wcs.pixel_to_world(x, y)
    catalog = Table()
    catalog["object_id"] = mer_scene["object_id"]
    catalog["ra"] = fitted_world.ra.deg
    catalog["dec"] = fitted_world.dec.deg
    catalog["prior_ra"] = prior_ra
    catalog["prior_dec"] = prior_dec
    catalog["prior_x"] = prior_x
    catalog["prior_y"] = prior_y
    catalog["shift_x_pix"] = shift_x
    catalog["shift_y_pix"] = shift_y
    catalog["x"] = x
    catalog["y"] = y
    catalog["model_class"] = [type(source).__name__ for source in sources]
    catalog["flux_njy"] = flux
    catalog["fluxerr_njy"] = error
    catalog["in_image"] = ((x >= 0) & (x < image.science_njy.shape[1]) &
                           (y >= 0) & (y < image.science_njy.shape[0]))
    catalog.meta["SKYNJY"] = float(tim.sky.getValue())
    catalog.meta["RECWPIX"] = recenter_width_pixels
    catalog.meta["NRECENT"] = int(np.count_nonzero((shift_x != 0) | (shift_y != 0)))
    catalog.meta["ERRMODEL"] = "formal Tractor errors from supplied diagonal variance"
    return catalog, model, residual
