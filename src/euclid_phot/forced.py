"""Forced photometry with Tractor, the same for every band.

Every band runs the same recipe on its own Tractor image
(:func:`euclid_phot.tractor_image.make_tractor_image`):

1. clone the VIS source models for the band, positions and shapes fixed
   (:func:`euclid_phot.sources.clone_sources`);
2. freeze everything but the fluxes (and, optionally, a constant sky) and
   solve the linear problem (:func:`flux_tractor`, :func:`solve_fluxes`);
3. read each flux's inverse variance from the solve
   (:func:`flux_inverse_variance`) and turn it into a 1-sigma error,
   optionally scaled by the residual scatter (:func:`chi_inflation`).

Optional steps move the centroids by a bounded amount. Three variants
exist, one per use: :func:`refine_positions` (VIS prior, revert runaways),
:func:`fit_bounded_positions` (box-constrained joint fit, used for IRAC)
and :func:`gauss_newton_recenter` (one linearised step, used for ground
images).

The prior band (VIS) additionally fits free shapes (:func:`fit_free_shapes`)
after a first flux-only pass (:func:`fit_forced_photometry`).
"""
from __future__ import annotations

import copy as _copy
import warnings

import numpy as np
from tractor import PointSource, Tractor
from tractor.galaxy import FixedCompositeGalaxy

from .sources import _configure_shape_steps
from .units import UJY_PER_NMGY


# ---------------------------------------------------------------------------
# The flux-only solve
# ---------------------------------------------------------------------------

def freeze_to_brightness(sources) -> None:
    """Leave only each source's brightness free."""
    for src in sources:
        src.freezeAllBut("brightness")


def fluxes_ujy(sources, band: str) -> np.ndarray:
    """Current source brightnesses in ``band``, in microJansky."""
    return np.array([s.brightness.getFlux(band) for s in sources]) * UJY_PER_NMGY


def _scaled_flux_optimizer():
    from .flux_solver import ScaledFluxOptimizer
    return ScaledFluxOptimizer()


def _constrained_optimizer():
    from tractor.constrained_optimizer import ConstrainedOptimizer
    return ConstrainedOptimizer()


# Tractor's default is LSQR; "constrained" honours parameter bounds (needed
# by fit_bounded_positions); "scaled" column-normalises the templates and
# solves the linear problem exactly (needed in dense scenes with a wide PSF,
# unWISE and ground images, where one LSQR update stops short).
OPTIMIZERS = {
    "lsqr": None,
    "constrained": _constrained_optimizer,
    "scaled": _scaled_flux_optimizer,
}


def flux_tractor(tim, sources, *, optimizer: str = "lsqr",
                 fit_sky: bool = False) -> Tractor:
    """A one-image Tractor with only the source fluxes (and the sky) free."""
    if optimizer not in OPTIMIZERS:
        raise ValueError(f"optimizer must be one of {sorted(OPTIMIZERS)}")
    freeze_to_brightness(sources)
    tim.freezeAllParams()
    if fit_sky:
        tim.thawParam("sky")
    make = OPTIMIZERS[optimizer]
    if make is None:
        return Tractor([tim], sources)
    return Tractor([tim], sources, optimizer=make())


def solve_fluxes(tractor, *, fit_sky: bool = False, variance: bool = True,
                 wantims: bool = False):
    """Solve for the free fluxes; returns Tractor's forced-photometry result.

    ``result.IV`` holds the inverse variance of each free parameter
    (:func:`flux_inverse_variance` picks out the fluxes) and, with
    ``wantims``, ``result.ims1`` the fitted model images.
    """
    from .flux_solver import ScaledFluxOptimizer

    # The scaled solver treats every flux as its own column.
    shared_params = not isinstance(tractor.optimizer, ScaledFluxOptimizer)
    return tractor.optimize_forced_photometry(
        minsb=0.0, mindlnp=1.0, sky=fit_sky, variance=variance,
        shared_params=shared_params, fitstats=wantims, wantims=wantims)


def flux_inverse_variance(result, tractor, n_sources: int) -> np.ndarray:
    """Per-source flux inverse variance (nanomaggy^-2) from a flux solve.

    ``result.IV`` lists every free parameter; a fitted sky comes first in
    some Tractor versions. The brightness entries are selected by name.
    """
    iv = getattr(result, "IV", None)
    if iv is None:
        raise RuntimeError("the flux solve returned no variances (variance=False?)")
    iv = np.asarray(iv, dtype=float)
    names = tractor.getParamNames()
    index = [i for i, name in enumerate(names)
             if name.startswith("catalog.") and ".brightness." in name]
    if len(iv) == len(names) and len(index) == n_sources:
        return iv[index]
    if len(iv) == n_sources:
        return iv
    raise RuntimeError(
        f"unexpected Tractor variance layout: {len(iv)} entries for "
        f"{n_sources} sources and {len(names)} free parameters")


def flux_errors_ujy(inverse_variance, inflation: float = 1.0) -> np.ndarray:
    """1-sigma flux errors in microJansky; NaN where a flux is unconstrained."""
    iv = np.asarray(inverse_variance, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(iv > 0, inflation / np.sqrt(iv), np.nan) * UJY_PER_NMGY


def chi_inflation(data, model, invvar, *, sky: float = 0.0,
                  minimum_pool_pixels: int = 50):
    """Scale factor for the formal flux errors from the residual scatter.

    chi = (data - model) * sqrt(invvar) should have unit scatter where the
    noise model is right. Its robust scatter (``mad_std``) is measured on
    pixels where the source model is fainter than 1, 2, 3 or 5 sigma (the
    first with ``minimum_pool_pixels``), else on all pixels after
    sigma-clipping. Returns ``(factor >= 1, pool description, pool size)``.
    ``sky`` is subtracted from ``model`` to locate the source light.
    """
    from astropy.stats import mad_std, sigma_clip

    data = np.asarray(data, dtype=float)
    model = np.asarray(model, dtype=float)
    invvar = np.asarray(invvar, dtype=float)
    valid = np.isfinite(data) & np.isfinite(model) & (invvar > 0)
    chi = np.full(data.shape, np.nan, dtype=float)
    chi[valid] = (data[valid] - model[valid]) * np.sqrt(invvar[valid])
    source_model = model - float(sky)
    sigma = np.full(data.shape, np.inf, dtype=float)
    sigma[valid] = 1.0 / np.sqrt(invvar[valid])
    for multiple in (1.0, 2.0, 3.0, 5.0):
        pool = chi[valid & (np.abs(source_model) < multiple * sigma)]
        pool = pool[np.isfinite(pool)]
        if pool.size >= minimum_pool_pixels:
            scatter = float(mad_std(pool))
            return (max(1.0, scatter) if np.isfinite(scatter) else 1.0,
                    f"source-sparse (|source model|<{multiple:g}sigma)",
                    int(pool.size))
    pool = np.asarray(
        sigma_clip(chi[valid], sigma=3.0, maxiters=5).compressed(), dtype=float)
    scatter = float(mad_std(pool)) if pool.size else np.nan
    return (max(1.0, scatter) if np.isfinite(scatter) else 1.0,
            "all-pixels sigma-clipped fallback", int(pool.size))


# ---------------------------------------------------------------------------
# Prior-band (VIS) fit: flux pass, free shapes
# ---------------------------------------------------------------------------

def conditional_flux_errors(tim, sources) -> np.ndarray:
    """Flux uncertainties in microJy at the sources' current geometry.

    For each single-amplitude source, return ``1/sqrt(sum(t**2 * weight))``
    with ``t`` its unit-flux template including the image photocalibration.
    Positions, shapes, neighbours and sky are held fixed; blend/shape/sky
    covariance and correlated image noise are not included. No parameter
    or flux is fitted or changed. A source with no weighted template
    support receives NaN. Requires a ``LinearPhotoCal`` image, as built by
    :func:`euclid_phot.tractor_image.make_tractor_image`.
    """
    from tractor import LinearPhotoCal

    photocal = tim.getPhotoCal()
    if not isinstance(photocal, LinearPhotoCal):
        raise ValueError("conditional_flux_errors requires LinearPhotoCal")
    scale = float(photocal.getScale())
    weights = tim.getInvvar()
    errors = np.full(len(sources), np.nan)
    for i, src in enumerate(sources):
        patches = src.getUnitFluxModelPatches(tim, minval=0.)
        if len(patches) != 1:
            raise ValueError("conditional_flux_errors requires one flux amplitude per source")
        patch = patches[0]
        if patch is None or patch.patch is None:
            continue
        spatch, simage = patch.getSlices(tim.shape)
        template = np.asarray(patch.patch[spatch], dtype=float) * scale
        iv = float(np.sum(template**2 * weights[simage]))
        if np.isfinite(iv) and iv > 0:
            errors[i] = UJY_PER_NMGY / np.sqrt(iv)
    return errors


def _flux_ujy(src, band):
    return src.brightness.getFlux(band) * UJY_PER_NMGY


def _revert_railed_shapes(sources, seed_shapes, *,
                          growth_cap: float = 5.0,
                          re_floor_arcsec: float = 2.72,
                          max_re_arcsec: float = 20.0) -> int:
    """Return step-2 shapes that ran away along the radius degeneracy to
    their seed value, and freeze them.

    Uses the same per-source ceiling as the model-selection tree
    (selection.py): ``clip(growth_cap * seed_re, re_floor, max_re)``.
    Returns the number reverted.
    """
    n = 0
    for i, seed in seed_shapes.items():
        src = sources[i]
        try:
            re_seed = float(seed.re)
            shape = src.getShape()
        except (AttributeError, TypeError):
            continue
        try:
            re_fit = float(shape.re)
        except OverflowError:
            re_fit = np.inf
        cap = float(np.clip(growth_cap * re_seed,
                            re_floor_arcsec, max_re_arcsec))
        if not np.isfinite(re_fit) or re_fit > cap:
            src.shape = _copy.deepcopy(seed)
            src.freezeParam("shape")
            n += 1
    return n


def fit_forced_photometry(tim, sources, *,
                          band: str = "VIS",
                          initial_fluxes_ujy: np.ndarray | None = None,
                          flag_unphysical: bool = True,
                          return_errors: bool = False):
    """Prior-band step 1: freeze everything except brightness; solve fluxes.

    Parameters
    ----------
    return_errors : bool
        If True, also return ``flux_err_ujy``, the formal 1-sigma flux
        uncertainty (1/sqrt of the Tractor inverse-variance) from this
        flux-only fit, aligned 1:1 with ``sources``.

    Returns
    -------
    tractor : Tractor
    fit_quality : ndarray of bool, shape (len(sources),)
        True for sources that came out with a physical flux. Coarse
        catastrophe guard only: rejects a negative flux or a flux more than
        100x the MER reference, not a goodness-of-fit cut. Rejected sources
        are frozen in place so step 2 cannot touch them.
    flux_err_ujy : ndarray, optional
        Only when ``return_errors=True``.
    """
    tractor = flux_tractor(tim, sources)
    R = solve_fluxes(tractor, variance=return_errors)

    flux_err_ujy = None
    if return_errors:
        flux_err_ujy = flux_errors_ujy(
            flux_inverse_variance(R, tractor, len(sources)))

    fit_quality = np.ones(len(sources), dtype=bool)
    if flag_unphysical:
        for i, src in enumerate(sources):
            flux = _flux_ujy(src, band)
            ref = (initial_fluxes_ujy[i]
                   if initial_fluxes_ujy is not None else None)
            bad = flux < 0
            if ref is not None and ref > 0 and abs(flux) > 100 * ref:
                bad = True
            if bad:
                src.freezeAllParams()
                fit_quality[i] = False
    if return_errors:
        return tractor, fit_quality, flux_err_ujy
    return tractor, fit_quality


def fit_free_shapes(tractor, tim, sources, fit_quality, *,
                    band: str = "VIS",
                    initial_fluxes_ujy: np.ndarray | None = None,
                    margin_pix: int = 50,
                    n_iter: int = 30,
                    dlnp_min: float = 0.1):
    """Prior-band step 2: thaw shapes for interior, well-behaved sources.

    Re-uses the Tractor object from step 1 (positions frozen in step 1 stay
    frozen). Thaws the shape of every resolved galaxy at least ``margin_pix``
    from the edge; for a ``SersicGalaxy`` (the default MER class) that includes
    the Sersic index ``n``. ``PointSource``, ``SimpleGalaxy``, and
    ``FixedCompositeGalaxy`` are skipped. Returns the updated ``fit_quality``
    array and the number of sources thawed.
    """
    # SimpleGalaxy (Weaver et al. 2023 bridge tier) subclasses ExpGalaxy;
    # its defining frozen circular shape must survive step 2.
    from .selection import SimpleGalaxy

    H, W = tim.shape
    # Cap the margin so an interior region exists on small cutouts.
    margin_pix = min(margin_pix, max(1, min(H, W) // 4))
    wcs = tim.getWcs()
    n_thawed = 0
    seed_shapes: dict[int, object] = {}
    for i, src in enumerate(sources):
        if not fit_quality[i] or isinstance(src, (PointSource, SimpleGalaxy)):
            continue
        # LSQR can step a FixedCompositeGalaxy logre to where exp(logre)
        # overflows; the selection tree already fit its shape.
        if isinstance(src, FixedCompositeGalaxy):
            continue
        px, py = wcs.positionToPixel(src.getPosition())
        if margin_pix < px < W - margin_pix and margin_pix < py < H - margin_pix:
            _configure_shape_steps(src.getShape())
            src.thawAllParams()
            src.freezeParam("pos")
            seed_shapes[i] = _copy.deepcopy(src.getShape())
            n_thawed += 1

    def _optimize_loop():
        n_railed = 0
        for _ in range(n_iter):
            try:
                dlnp, _, _ = tractor.optimize()
            except OverflowError:
                # EllipseESoft.re = exp(logre); restore invalid shapes
                # even when an optimizer update fails partway through.
                warnings.warn(
                    "LSQR shape optimization overflowed; abandoning the "
                    "remaining step-2 iterations and restoring runaway shapes.",
                    stacklevel=2,
                )
                break
            finally:
                # Check every accepted update, including the last one.
                # Otherwise an unchecked recovery pass can leave a huge
                # radius that becomes an enormous FFT patch in WISE/IRAC.
                n_railed += _revert_railed_shapes(sources, seed_shapes)
            if dlnp < dlnp_min:
                break
        return n_railed

    n_railed = _optimize_loop()
    if n_railed:
        warnings.warn(
            f"step 2 reverted {n_railed} runaway shape(s) to the prior "
            "value (radius past the per-source ceiling); fluxes refit "
            "with those shapes frozen.",
            stacklevel=2,
        )
        # Recover amplitudes at the final, checked shapes. Repeating a
        # free-shape pass here would allow another source to run away.
        freeze_to_brightness(sources)
        solve_fluxes(tractor, variance=False)

    for i, src in enumerate(sources):
        if not fit_quality[i]:
            continue
        flux = _flux_ujy(src, band)
        ref = (initial_fluxes_ujy[i]
               if initial_fluxes_ujy is not None else None)
        bad = flux < 0
        if ref is not None and ref > 0 and abs(flux) > 100 * ref:
            bad = True
        if bad:
            fit_quality[i] = False

    return fit_quality, n_thawed


def measure_residual(tractor, tim, footprint=None):
    """Compute model + residual arrays.

    Returns
    -------
    dict with ``model``, ``residual``, ``data``, ``chi`` ndarrays and (if
    ``footprint`` is given) the sigma-clipped sigma outside ``footprint``.
    """
    from astropy.stats import sigma_clipped_stats
    model = tractor.getModelImage(0)
    data = tim.getImage()
    residual = data - model
    chi = tractor.getChiImage(imgi=0)
    out = {"model": model, "residual": residual, "data": data, "chi": chi}
    if footprint is not None:
        outside = ~footprint
        if outside.sum() > 100:
            _, _, sigma = sigma_clipped_stats(residual[outside], sigma=3.0)
            out["sigma_outside"] = float(sigma)
    return out


# ---------------------------------------------------------------------------
# Bounded recentering
# ---------------------------------------------------------------------------

def position_offsets_arcsec(sources, prior_ra, prior_dec):
    """(dx, dy, total) offsets in arcsec of each source from its prior."""
    ra = np.asarray([source.getPosition().ra for source in sources], dtype=float)
    dec = np.asarray([source.getPosition().dec for source in sources], dtype=float)
    prior_ra = np.asarray(prior_ra, dtype=float)
    prior_dec = np.asarray(prior_dec, dtype=float)
    dra = (ra - prior_ra + 180.0) % 360.0 - 180.0
    dx = dra * np.cos(np.radians(prior_dec)) * 3600.0
    dy = (dec - prior_dec) * 3600.0
    return dx, dy, np.hypot(dx, dy)


def refine_positions(tractor, tim, sources, fit_quality, *,
                     band: str = "VIS",
                     flux_floor_ujy: float = 10.0,
                     max_shift_arcsec: float = 0.2,
                     margin_pix: int = 50,
                     n_iter: int = 30,
                     dlnp_min: float = 0.1):
    """Prior band: let bright, interior sources recenter; revert runaways.

    Thaws the centroid of sources with ``flux > flux_floor_ujy`` at least
    ``margin_pix`` from the edge, then jointly re-optimizes flux + position.
    Moves beyond ``max_shift_arcsec`` revert to the prior position: the
    genuine Q1 catalog-to-centroid offset is sub-pixel, so a larger move is
    a runaway. Fluxes are re-settled at the final positions, which later
    target-band fits share.

    Returns ``(fit_quality, n_refined, shift_arcsec)`` where ``shift_arcsec``
    is aligned 1:1 with ``sources`` (0 where not thawed or reverted).
    """
    from tractor import RaDecPos

    H, W = tim.shape
    margin_pix = min(margin_pix, max(1, min(H, W) // 4))
    wcs = tim.getWcs()
    orig = [(float(s.getPosition().ra), float(s.getPosition().dec))
            for s in sources]

    thawed = []
    for src in sources:
        src.freezeAllParams()
        src.thawParam("brightness")
    for i, src in enumerate(sources):
        if not fit_quality[i] or _flux_ujy(src, band) < flux_floor_ujy:
            continue
        px, py = wcs.positionToPixel(src.getPosition())
        if margin_pix < px < W - margin_pix and margin_pix < py < H - margin_pix:
            src.thawParam("pos")
            thawed.append(i)

    shift_arcsec = np.zeros(len(sources))
    if not thawed:
        return fit_quality, 0, shift_arcsec

    for _ in range(n_iter):
        try:
            dlnp, _, _ = tractor.optimize()
        except OverflowError:
            warnings.warn(
                "LSQR position optimization overflowed; abandoning the "
                "remaining iterations. Step-1 fluxes are unaffected.",
                stacklevel=2)
            break
        if dlnp < dlnp_min:
            break

    orig_ra = np.array([o[0] for o in orig])
    orig_dec = np.array([o[1] for o in orig])
    _, _, shift = position_offsets_arcsec(sources, orig_ra, orig_dec)
    n_reverted = 0
    for i in thawed:
        if shift[i] > max_shift_arcsec:
            sources[i].pos = RaDecPos(orig_ra[i], orig_dec[i])
            n_reverted += 1
        else:
            shift_arcsec[i] = shift[i]

    for src in sources:
        src.freezeAllParams()
        src.thawParam("brightness")
    for _ in range(10):
        dlnp, _, _ = tractor.optimize()
        if dlnp < dlnp_min:
            break

    return fit_quality, len(thawed) - n_reverted, shift_arcsec


def _configure_position_limits(sources, prior_ra, prior_dec, *,
                               max_shift_arcsec, derivative_step_arcsec):
    cap_deg = float(max_shift_arcsec) / 3600.0
    step_deg = float(derivative_step_arcsec) / 3600.0
    for source, ra0, dec0 in zip(sources, prior_ra, prior_dec):
        source.freezeAllParams()
        source.thawParam("pos")
        source.thawParam("brightness")
        position = source.getPosition()
        cos_dec = max(abs(float(np.cos(np.radians(dec0)))), 1.0e-6)
        position.setStepSizes([step_deg / cos_dec, step_deg])
        position.lowers = [ra0 - cap_deg / cos_dec, dec0 - cap_deg]
        position.uppers = [ra0 + cap_deg / cos_dec, dec0 + cap_deg]
        position.maxstep = [0.5 * cap_deg / cos_dec, 0.5 * cap_deg]


def _project_position_shifts(sources, prior_ra, prior_dec, max_shift_arcsec):
    dx, dy, shift = position_offsets_arcsec(sources, prior_ra, prior_dec)
    outside = np.isfinite(shift) & (shift > float(max_shift_arcsec))
    for index in np.flatnonzero(outside):
        scale = float(max_shift_arcsec) / float(shift[index])
        cos_dec = max(abs(float(np.cos(np.radians(prior_dec[index])))), 1.0e-6)
        position = sources[index].getPosition()
        position.ra = float(prior_ra[index]) + (
            dx[index] * scale / (3600.0 * cos_dec))
        position.dec = float(prior_dec[index]) + dy[index] * scale / 3600.0
    return outside


def fit_bounded_positions(tractor, sources, prior_ra, prior_dec, *,
                          max_shift_arcsec: float, pixel_scale_arcsec: float,
                          max_iterations: int = 15, dlnp_min: float = 0.1):
    """Jointly fit fluxes and centroids, each centroid within a circle.

    Needs a Tractor built with ``optimizer="constrained"``: each position
    gets box bounds of ``max_shift_arcsec`` around its prior, and after
    every step a centroid outside the circle is projected back onto it.
    Leaves only the fluxes free again; re-solve them at the new positions
    with :func:`solve_fluxes`. Returns ``(iterations, ever_projected)``.
    """
    _configure_position_limits(
        sources, prior_ra, prior_dec,
        max_shift_arcsec=max_shift_arcsec,
        derivative_step_arcsec=max(0.01, 0.05 * pixel_scale_arcsec))
    projected = np.zeros(len(sources), dtype=bool)
    iterations = 0
    for iteration in range(int(max_iterations)):
        dlnp, _, _ = tractor.optimize()
        iterations = iteration + 1
        projected |= _project_position_shifts(
            sources, prior_ra, prior_dec, max_shift_arcsec)
        if not np.isfinite(dlnp) or dlnp < float(dlnp_min):
            break
    freeze_to_brightness(sources)
    return iterations, projected


def gauss_newton_recenter(tractor, sources, inverse_variance, *, band: str,
                          wcs, width_pixels: float, psf_shape,
                          min_snr: float = 10.0, min_delta_chi2: float = 9.0):
    """One linearised centroid step per well-detected source.

    Uses the residual of the current joint flux fit with the source's own
    model added back, solves the 2x2 normal equations for its shift, clips
    each axis to ``width_pixels / 2`` and keeps the step only if it lowers
    chi^2 by ``min_delta_chi2``. Faint or unconstrained sources keep their
    positions. Returns per-source ``(shift_x, shift_y)`` in pixels; re-solve
    the fluxes afterwards if any moved.
    """
    tim = tractor.getImage(0)
    data = np.asarray(tim.getImage(), float)
    invvar = np.asarray(tim.getInvvar(), float)
    ny, nx = data.shape
    ra0 = np.asarray([s.getPosition().ra for s in sources], float)
    dec0 = np.asarray([s.getPosition().dec for s in sources], float)
    prior_x, prior_y = wcs.world_to_pixel_values(ra0, dec0)
    shift_x = np.zeros(len(sources), dtype=float)
    shift_y = np.zeros(len(sources), dtype=float)
    iv = np.asarray(inverse_variance, float)
    flux = np.asarray([s.getBrightness().getFlux(band) for s in sources])
    snr = np.where(iv > 0, flux * np.sqrt(np.maximum(iv, 0)), 0.0)
    residual_all = data - np.asarray(tractor.getModelImage(0), float)
    half_width = width_pixels / 2.0
    radius = min(16, max(8, min(psf_shape) // 4))
    for i, source in enumerate(sources):
        if not np.isfinite(snr[i]) or snr[i] < min_snr:
            continue
        patch = source.getModelPatch(tim)
        if patch is None or patch.patch is None:
            continue
        x0 = max(0, patch.x0, int(np.floor(prior_x[i])) - radius)
        x1 = min(nx, patch.x1, int(np.floor(prior_x[i])) + radius + 1)
        y0 = max(0, patch.y0, int(np.floor(prior_y[i])) - radius)
        y1 = min(ny, patch.y1, int(np.floor(prior_y[i])) + radius + 1)
        if x1 - x0 < 3 or y1 - y0 < 3:
            continue
        model_patch = np.asarray(
            patch.patch[y0 - patch.y0:y1 - patch.y0,
                        x0 - patch.x0:x1 - patch.x0], float)
        gradient_y, gradient_x = np.gradient(model_patch)
        jac_x, jac_y = -gradient_x, -gradient_y
        weight = invvar[y0:y1, x0:x1]
        residual = residual_all[y0:y1, x0:x1]
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
        if not np.isfinite(delta_chi2) or delta_chi2 < min_delta_chi2:
            continue
        world = wcs.pixel_to_world(prior_x[i] + proposed[0],
                                   prior_y[i] + proposed[1])
        source.setPosition(type(source.getPosition())(world.ra.deg, world.dec.deg))
        shift_x[i], shift_y[i] = proposed
    return shift_x, shift_y
