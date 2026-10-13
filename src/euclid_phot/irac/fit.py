"""IRAC forced photometry of fixed VIS source models with an effective PRF.

The fit is the shared recipe of :mod:`euclid_phot.forced`. What is IRAC's
own: the mosaic and its effective PRF (:func:`prepare_channel`), the
exposure-coverage cut on the scene, the PRF aperture correction and the 2%
PRF systematic added to the errors (:func:`irac_catalog`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.table import Table

from ..background import SEPBackground, estimate_sep_background
from ..forced import (
    chi_inflation,
    fit_bounded_positions,
    flux_errors_ujy,
    flux_inverse_variance,
    flux_tractor,
    fluxes_ujy,
    position_offsets_arcsec,
    solve_fluxes,
)
from ..pixel_psf import GridPixelizedPSF
from ..scene import footprint_mask
from ..sources import clone_sources
from ..tractor_image import make_tractor_image
from ..units import UJY_PER_NMGY, pixel_scale_arcsec
from .config import (
    DEFAULT_DATA_ROOT,
    IRAC_NATIVE_PIXEL_SCALE_ARCSEC,
    IRAC_PRF_FLUX_CORRECTION,
    IRAC_PRF_SYSTEMATIC_FRACTION,
    normalize_channel,
)

from .data import (
    MosaicInputs,
    fetch_cosmic_dawn_mosaic,
    fetch_seip_mosaic,
    fetch_sha_mosaic,
)
from .mosaic import MosaicCutout
from .prfmap import build_prfmap_psf


@dataclass(frozen=True)
class PreparedChannel:
    """Science image and spatial PSF grid shared by fixed/free fits."""

    inputs: MosaicInputs
    prf_builder: object
    tractor_psf: GridPixelizedPSF
    mission: str
    prf_backend: str

    @property
    def cutout(self):
        return self.inputs.cutout

    @property
    def local_product(self):
        return self.inputs.local_product



_FETCHERS = {
    "dawn": (fetch_cosmic_dawn_mosaic, "cosmic_dawn"),
    "seip": (fetch_seip_mosaic, "seip"),
    "pbcd": (fetch_sha_mosaic, "pbcd"),
}


def _backend(mosaic_backend: str) -> str:
    backend = str(mosaic_backend).strip().lower()
    if backend not in _FETCHERS:
        raise ValueError("mosaic_backend must be 'dawn', 'seip', or 'pbcd'")
    return backend


def fetch_irac_mosaic(
    ra: float,
    dec: float,
    channel: str | int,
    *,
    size_arcsec: float,
    mosaic_backend: str = "seip",
    data_root: str | Path = DEFAULT_DATA_ROOT,
    mask_bad_bits: int | None = None,
    aorkey: int | None = None,
    super_mosaic_id: str | None = None,
    force_download: bool = False,
) -> MosaicInputs:
    """Fetch an IRAC mosaic cutout and its contributing exposures.

    ``mosaic_backend`` selects Cosmic Dawn (``"dawn"``), SEIP Super Mosaics
    (``"seip"``) or single-AOR SHA mosaics (``"pbcd"``), cached under
    ``data_root/<backend>``. ``aorkey`` (pbcd) and ``super_mosaic_id``
    (seip) pick a specific product.
    """
    backend = _backend(mosaic_backend)
    fetch, subdir = _FETCHERS[backend]
    options = {"mask_bad_bits": mask_bad_bits}
    if aorkey is not None:
        options["aorkey"] = aorkey
    if super_mosaic_id is not None:
        options["super_mosaic_id"] = super_mosaic_id
    return fetch(ra, dec, channel, size_arcsec=size_arcsec,
                 data_dir=Path(data_root) / subdir,
                 force_download=force_download, **options)


def prepare_channel(
    ra: float,
    dec: float,
    channel: str | int,
    *,
    size_arcsec: float,
    mosaic_backend: str = "seip",
    data_dir: str | Path = DEFAULT_DATA_ROOT,
    aorkey: int | None = None,
    super_mosaic_id: str | None = None,
    mask_bad_bits: int | None = None,
    mission: str | None = None,
    prfmap_grid_file: str | Path | None = None,
    prfmap_prf_directory: str | Path | None = None,
    prfmap_pixel_scale_arcsec: float | None = None,
    grid_spacing_arcsec: float = 28.8,
    prf_stamp_size: int = 51,
    weighting: str | None = None,
    pixel_integration_subsamples: int = 1,
    recenter_detector_prfs: bool = True,
    force_download: bool = False,
) -> PreparedChannel:
    """IRAC's own inputs for one channel: the mosaic and its effective PRF.

    The PRF is rebuilt from the detector PRFs and the exposures contributing
    to the mosaic. Defaults: SEIP mosaics use the cryogenic PRFs
    (``mission="cryo"``), others pick the mission from the exposures; Cosmic
    Dawn exposures are weighted by exposure time, others uniformly.
    """
    backend = _backend(mosaic_backend)
    if mission is None:
        mission = "cryo" if backend == "seip" else "auto"
    if weighting is None:
        weighting = "exptime" if backend == "dawn" else "uniform"
    inputs = fetch_irac_mosaic(
        ra, dec, channel, size_arcsec=size_arcsec, mosaic_backend=backend,
        data_root=data_dir, mask_bad_bits=mask_bad_bits, aorkey=aorkey,
        super_mosaic_id=super_mosaic_id, force_download=force_download)
    builder, psf, selected_mission, prf_backend = build_prfmap_psf(
        inputs.local_product,
        inputs.cutout,
        exposures=inputs.exposures,
        mission=mission,
        calibration_dir=Path(data_dir) / "calibration",
        force_download=force_download,
        grid_file=prfmap_grid_file,
        prf_directory=prfmap_prf_directory,
        prf_pixel_scale_arcsec=prfmap_pixel_scale_arcsec,
        grid_spacing_arcsec=grid_spacing_arcsec,
        stamp_size=prf_stamp_size,
        weighting=weighting,
        pixel_integration_subsamples=pixel_integration_subsamples,
        recenter_detector_prfs=recenter_detector_prfs,
    )
    return PreparedChannel(inputs, builder, psf, selected_mission, prf_backend)


def select_irac_scene(catalog, sources, cutout, prf_builder, *,
                      margin_arcsec: float = 15.0):
    """Priors within ``margin_arcsec`` of the cutout and covered by at least
    one exposure of the mosaic (outside the coverage there are no pixels)."""
    if len(catalog) != len(sources):
        raise ValueError("catalog and sources must align one-to-one")
    ra = np.asarray(catalog["ra"], dtype=float)
    dec = np.asarray(catalog["dec"], dtype=float)
    near = footprint_mask(ra, dec, cutout.wcs, cutout.shape,
                          margin_pix=margin_arcsec / pixel_scale_arcsec(cutout.wcs))
    covered = np.asarray([
        any(exposure.covers(r, d) for exposure in prf_builder.exposures)
        for r, d in zip(ra, dec)])
    keep = near & covered
    return catalog[keep], [s for s, k in zip(sources, keep) if k]


@dataclass
class MosaicPhotometryResult:
    """Catalog and image diagnostics from one IRAC mosaic fit."""

    channel: str
    catalog: Table
    cutout: MosaicCutout
    tractor_image: object
    tractor: object
    background: SEPBackground
    source_model_image: np.ndarray
    model_image: np.ndarray
    residual_image: np.ndarray
    chi_inflation: float
    chi_pool_kind: str
    n_chi_pool: int
    position_iterations: int


def irac_catalog(catalog, sources, channel: str, flux_err_stat_ujy, builder,
                 prior_ra, prior_dec, max_shift_arcsec: float,
                 pixel_scale: float) -> Table:
    """Per-source IRAC table: aperture-corrected fluxes and errors that add
    the PRF systematic in quadrature, plus centroid shifts and coverage."""
    out = Table()
    if "object_id" in catalog.colnames:
        out["object_id"] = catalog["object_id"]
    ra = np.asarray([s.getPosition().ra for s in sources], dtype=float)
    dec = np.asarray([s.getPosition().dec for s in sources], dtype=float)
    raw_flux = fluxes_ujy(sources, channel)
    correction = IRAC_PRF_FLUX_CORRECTION[channel]
    flux = raw_flux * correction
    raw_err_stat = np.asarray(flux_err_stat_ujy, dtype=float)
    err_stat = raw_err_stat * correction
    raw_err = np.hypot(raw_err_stat, IRAC_PRF_SYSTEMATIC_FRACTION * np.abs(raw_flux))
    err = np.hypot(err_stat, IRAC_PRF_SYSTEMATIC_FRACTION * np.abs(flux))
    out["ra"] = ra
    out["dec"] = dec
    out["prior_ra"] = np.asarray(prior_ra, dtype=float)
    out["prior_dec"] = np.asarray(prior_dec, dtype=float)
    dx, dy, shift = position_offsets_arcsec(sources, prior_ra, prior_dec)
    out["position_dx_arcsec"] = dx
    out["position_dy_arcsec"] = dy
    out["position_shift_arcsec"] = shift
    out["position_shift_pixels"] = shift / float(pixel_scale)
    out["position_at_limit"] = (
        (max_shift_arcsec > 0)
        & np.isclose(shift, max_shift_arcsec, rtol=0.0, atol=1.0e-6))
    out["source_model"] = [type(s).__name__ for s in sources]
    x, y = builder.output_wcs.world_to_pixel_values(ra, dec)
    shape = builder.output_wcs.array_shape
    if shape is None:
        shape = builder.output_wcs.pixel_shape
        shape = (shape[1], shape[0]) if shape is not None else (0, 0)
    out["in_mosaic_cutout"] = ((x >= -0.5) & (x < shape[1] - 0.5)
                               & (y >= -0.5) & (y < shape[0] - 0.5))
    # Contributor count of each source's nearest PRF grid stamp (cached).
    n_exposures = np.zeros(len(sources), dtype=np.int32)
    for i, (sx, sy) in enumerate(zip(x, y)):
        try:
            _, n_exposures[i] = builder.render(sx, sy)
        except ValueError:
            n_exposures[i] = 0
    out["n_exposures"] = n_exposures.astype(np.int16)
    key = channel.lower()
    out[f"flux_{key}_raw_ujy"] = raw_flux
    out[f"fluxerr_{key}_raw_stat_ujy"] = raw_err_stat
    out[f"fluxerr_{key}_raw_ujy"] = raw_err
    out[f"flux_{key}_ujy"] = flux
    out[f"fluxerr_{key}_stat_ujy"] = err_stat
    out[f"fluxerr_{key}_ujy"] = err
    with np.errstate(divide="ignore", invalid="ignore"):
        out[f"snr_{key}"] = flux / err
    return out


def fit_prepared(
    prepared: PreparedChannel,
    mer_catalog,
    prior_sources: list,
    *,
    neighbor_buffer_arcsec: float = 15.0,
    inflate_uncertainties: bool = True,
    background_mesh_arcsec: float = 60.0,
    background_filter_size: int = 3,
    max_position_shift_arcsec: float = 0.0,
    position_max_iterations: int = 15,
    position_dlnp_min: float = 0.1,
) -> MosaicPhotometryResult:
    """Fit the frozen VIS models to one prepared IRAC channel.

    The shared recipe: SEP background subtracted, VIS models cloned with a
    render patch covering the PRF, a flux-only solve, then (when
    ``max_position_shift_arcsec > 0``) a bounded joint position/flux fit and
    a final flux solve; errors scaled by the residual chi scatter.
    """
    if not np.isfinite(max_position_shift_arcsec) or max_position_shift_arcsec < 0:
        raise ValueError("max_position_shift_arcsec must be finite and >= 0")
    cutout = prepared.cutout
    builder = prepared.prf_builder
    channel = normalize_channel(cutout.channel)
    catalog, selected = select_irac_scene(
        mer_catalog, prior_sources, cutout, builder,
        margin_arcsec=neighbor_buffer_arcsec)
    if len(catalog) == 0:
        raise ValueError("no prior sources fall on the IRAC mosaic")
    scale = pixel_scale_arcsec(cutout.wcs)

    # 1. The Tractor image: background-subtracted uJy pixels, effective PRF.
    background = estimate_sep_background(
        cutout.data_ujy, cutout.fit_mask, scale,
        mesh_arcsec=background_mesh_arcsec, filter_size=background_filter_size)
    science = np.asarray(cutout.data_ujy, dtype=np.float32)
    data = np.where(np.isfinite(science), science - background.image, 0.0)
    product = cutout.product.product
    label = getattr(product, "product_root", None) or f"r{getattr(product, 'aorkey', 'unknown')}"
    image = make_tractor_image(
        np.asarray(data, dtype=np.float32), np.asarray(cutout.invvar, dtype=np.float32),
        cutout.wcs, prepared.tractor_psf, band=channel,
        nmgy_to_data=UJY_PER_NMGY, name=f"{channel}-{label}")

    # 2. VIS models with a render patch that does not crop the PRF wings.
    sources = clone_sources(selected, channel,
                            psf_halfsize_pix=builder.stamp_size // 2,
                            pixel_scale_arcsec=scale)
    prior_ra = np.asarray([s.getPosition().ra for s in sources], dtype=float)
    prior_dec = np.asarray([s.getPosition().dec for s in sources], dtype=float)

    # 3. Flux-only solve; optionally bounded recentering, then fluxes again.
    tractor = flux_tractor(image, sources, optimizer="constrained")
    result = solve_fluxes(tractor)
    iterations = 0
    if max_position_shift_arcsec > 0:
        iterations, _ = fit_bounded_positions(
            tractor, sources, prior_ra, prior_dec,
            max_shift_arcsec=max_position_shift_arcsec, pixel_scale_arcsec=scale,
            max_iterations=position_max_iterations, dlnp_min=position_dlnp_min)
        result = solve_fluxes(tractor)

    # 4. Errors from the solve, scaled by the residual chi scatter.
    source_model = np.asarray(tractor.getModelImage(0), dtype=np.float32)
    model = np.asarray(source_model + background.image, dtype=np.float32)
    residual = np.asarray(cutout.data_ujy - model, dtype=np.float32)
    if inflate_uncertainties:
        inflation, pool_kind, pool_count = chi_inflation(
            image.getImage(), source_model, image.getInvvar())
    else:
        inflation, pool_kind, pool_count = 1.0, "disabled", 0
    err_stat = flux_errors_ujy(
        flux_inverse_variance(result, tractor, len(sources)), inflation)

    table = irac_catalog(catalog, sources, channel, err_stat, builder,
                         prior_ra, prior_dec, float(max_position_shift_arcsec), scale)
    table.meta.update({
        "CHANNEL": channel,
        "PRFCORR": IRAC_PRF_FLUX_CORRECTION[channel],
        "PRFCREF": "IRAC Instrument Handbook Appendix C.1",
        "PRFSYS": IRAC_PRF_SYSTEMATIC_FRACTION,
        "ERRMODEL": "IRAC uncertainty-map covariance, residual inflation, and PRF systematic",
        "PRODUCT": str(getattr(product, "product_root", getattr(product, "aorkey", "unknown"))),
        "SCIENCE": prepared.inputs.backend,
        "PRF": prepared.prf_backend,
        "GRIDPSF": True,
        "CHIINFL": inflation,
        "CHIPOOL": pool_kind,
        "POSMAXAS": float(max_position_shift_arcsec),
        "POSITER": int(iterations),
        "BKGTYPE": "SEP mesh",
        "BKGMESH": float(background.mesh_arcsec),
        "BKGFILT": int(background.filter_size),
        "BKGLEV": float(background.global_level),
        "BKGRMS": float(background.global_rms),
        "BKGCONST": bool(background.constant),
    })
    if hasattr(product, "aorkey"):
        table.meta["AORKEY"] = product.aorkey
    if hasattr(builder, "recenter_detector_prfs"):
        table.meta["PRFCENT"] = bool(builder.recenter_detector_prfs)
        offsets = builder.detector_centroid_offsets_native.values()
        radial = np.asarray([np.hypot(*offset) for offset in offsets], dtype=float)
        table.meta["PRFCOFF"] = float(np.median(radial))
    if hasattr(builder, "pixel_integration_subsamples"):
        table.meta["PRFSUB"] = int(builder.pixel_integration_subsamples)
    return MosaicPhotometryResult(
        channel=channel, catalog=table, cutout=cutout, tractor_image=image,
        tractor=tractor, background=background, source_model_image=source_model,
        model_image=model, residual_image=residual, chi_inflation=inflation,
        chi_pool_kind=pool_kind, n_chi_pool=pool_count,
        position_iterations=iterations)


def fit_irac_forced(
    sources: list,
    ra: float,
    dec: float,
    size_arcsec: float,
    channels=("IRAC1", "IRAC2"),
    *,
    mosaic_backend: str = "seip",
    data_dir: str | Path = DEFAULT_DATA_ROOT,
    neighbor_buffer_arcsec: float = 15.0,
    max_position_shift_arcsec: float | None = None,
    force_download: bool = False,
    verbose: bool = True,
    prepare_options: dict | None = None,
    fit_options: dict | None = None,
) -> dict:
    """Fit fixed VIS source models to IRAC mosaics, one channel at a time.

    Each channel's mosaic is cut to the field, its effective PRF is rebuilt
    from the detector PRFs and the contributing exposures, a SEP background
    is subtracted, and the sources are convolved with the PRF and fitted
    jointly. Each centroid may move by up to ``max_position_shift_arcsec``
    (default one native IRAC pixel).

    Returns ``{channel: dict}`` with ``flux_ujy``, ``flux_err_ujy``
    (statistical plus the 2% PRF systematic), ``flux_err_stat_ujy`` and
    ``position_shift_arcsec``, aligned with ``sources`` (NaN where a source
    is outside the mosaic coverage), plus the ``data``, ``model``,
    ``residual`` and ``invvar`` images in microJansky per pixel, their
    ``wcs``, the fit ``catalog`` and diagnostics.
    """
    backend = _backend(mosaic_backend)
    options = {"size_arcsec": float(size_arcsec), "mosaic_backend": backend,
               "data_dir": Path(data_dir), **(prepare_options or {})}

    n = len(sources)
    scene = Table({
        "object_id": np.arange(n),
        "ra": [float(s.getPosition().ra) for s in sources],
        "dec": [float(s.getPosition().dec) for s in sources],
    })
    results = {}
    for channel in (normalize_channel(c) for c in channels):
        if verbose:
            print(f"  {channel}: mosaic and effective PRF ({backend})")
        prepared = prepare_channel(ra, dec, channel, **options,
                                   force_download=force_download)
        shift = (IRAC_NATIVE_PIXEL_SCALE_ARCSEC[channel]
                 if max_position_shift_arcsec is None
                 else float(max_position_shift_arcsec))
        fit = fit_prepared(prepared, scene, list(sources),
                           neighbor_buffer_arcsec=neighbor_buffer_arcsec,
                           max_position_shift_arcsec=shift,
                           **(fit_options or {}))
        cat = fit.catalog
        rows = np.asarray(cat["object_id"], dtype=int)
        key = channel.lower()

        def aligned(name):
            values = np.full(n, np.nan)
            values[rows] = np.asarray(cat[name], dtype=float)
            return values

        results[channel] = {
            "flux_ujy": aligned(f"flux_{key}_ujy"),
            "flux_err_ujy": aligned(f"fluxerr_{key}_ujy"),
            "flux_err_stat_ujy": aligned(f"fluxerr_{key}_stat_ujy"),
            "position_shift_arcsec": aligned("position_shift_arcsec"),
            "data": np.asarray(fit.cutout.data_ujy, dtype=np.float32),
            "model": np.asarray(fit.model_image, dtype=np.float32),
            "residual": np.asarray(fit.residual_image, dtype=np.float32),
            "invvar": np.asarray(fit.cutout.invvar, dtype=np.float32),
            "wcs": fit.cutout.wcs,
            "catalog": cat,
            "chi_inflation": float(fit.chi_inflation),
            "prf_backend": prepared.prf_backend,
            "mosaic_backend": backend,
        }
        if verbose:
            print(f"  {channel}: {len(rows)} sources fitted, "
                  f"chi inflation {fit.chi_inflation:.2f}")
    return results
