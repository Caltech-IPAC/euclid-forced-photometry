"""Forced photometry on unWISE W1 / W2 with the VIS source models.

The fit is the shared recipe of :mod:`euclid_phot.forced`. What is
unWISE's own:

- a fit region padded by 5 FWHM per edge, so the 6.9" PSF wings of edge
  sources stay on the pixels (:func:`unwise_fit_region`);
- the unWISE bitmask (spikes, halos, ghosts) and Vega photometric scale;
- unWISE-catalog point sources outside the VIS footprint whose wings reach
  the field (:func:`euclid_phot.wise.catalog.unwise_neighbours`);
- a jointly fitted constant sky (small coadd cutouts keep a pedestal),
  with the checked "scaled" linear solver for these dense scenes.

Lang, Hogg & Schlegel (2016) is the canonical reference for WISE forced
photometry from a higher-resolution prior. Here the VIS model classes and
shapes are kept, frozen, with flux the only free parameter, as for every
band (``source_models="point"`` is an optional all-point-source
approximation); the flux errors carry an empirical chi inflation, a
heuristic that does not include source-to-source covariance.
"""
from __future__ import annotations

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from tractor import NanoMaggies, PointSource, RaDecPos
from tractor.galaxy import disable_galaxy_cache

from ..forced import (
    chi_inflation,
    flux_errors_ujy,
    flux_inverse_variance,
    flux_tractor,
    fluxes_ujy,
    solve_fluxes,
)
from ..sources import clone_sources
from ..tractor_image import make_tractor_image
from .catalog import unwise_neighbours
from .config import (
    DEFAULT_WISE_CACHE_DIR,
    UNWISE_BAD_BITS,
    UNWISE_PIXEL_SCALE,
    VEGA_OFFSET,
    WISE_FWHM_ARCSEC,
)
from .psf import get_wise_psf


def unwise_fit_region(tile: dict, ra: float, dec: float, size_arcsec: float, *,
                      mask_bad_bits: int | None = UNWISE_BAD_BITS):
    """Cut ``size_arcsec`` around (ra, dec) from one band of
    ``fetch_unwise_cutouts``; returns ``(data, invvar, wcs)``.

    Pixels off the tile, with non-finite values or with ``mask_bad_bits``
    set in the unWISE bitmask get zero inverse variance.
    """
    size_pix = int(round(size_arcsec / UNWISE_PIXEL_SCALE))
    cx, cy = tile["wcs"].world_to_pixel_values(ra, dec)
    center = (float(cx), float(cy))
    co_d = Cutout2D(tile["data"], center, size_pix, wcs=tile["wcs"],
                    mode="partial", fill_value=0.0)
    co_iv = Cutout2D(tile["ivar"], center, size_pix, wcs=tile["wcs"],
                     mode="partial", fill_value=0.0)
    data = co_d.data.astype(np.float32)
    ivar = co_iv.data.astype(np.float32)
    valid = np.isfinite(data) & np.isfinite(ivar) & (ivar > 0)
    data = np.where(valid, data, 0.0).astype(np.float32)
    ivar = np.where(valid, ivar, 0.0).astype(np.float32)
    # Zero the invvar on bright-star artifact pixels; skipped when the
    # cached tarball predates the mask.
    if mask_bad_bits and tile.get("mask") is not None:
        co_m = Cutout2D(tile["mask"], center, size_pix, wcs=tile["wcs"],
                        mode="partial", fill_value=0)
        bad = (co_m.data.astype(np.int32) & int(mask_bad_bits)) != 0
        ivar = np.where(bad, 0.0, ivar).astype(np.float32)
    return data, ivar, co_d.wcs


def neighbour_point_sources(catalog, keep, band: str) -> list:
    """Point sources for the kept unWISE-catalog rows, seeded from their
    catalog flux (Vega nanomaggies converted to AB)."""
    column = "flux_1" if band == "W1" else "flux_2"
    out = []
    for k in np.flatnonzero(keep):
        val = catalog[column][k]
        if (hasattr(val, "mask") and np.ma.is_masked(val)) or not np.isfinite(val):
            seed = 1.0
        else:
            seed = float(val) * 10 ** (-VEGA_OFFSET[band] / 2.5)
        out.append(PointSource(
            RaDecPos(float(catalog["ra"][k]), float(catalog["dec"][k])),
            NanoMaggies(**{band: seed})))
    return out


def fit_wise_forced(sources, wise_cutouts: dict, *,
                    ra: float, dec: float,
                    cutout_size_arcsec: float,
                    prior_cutout=None,
                    psf_stamps: dict | None = None,
                    bands=("W1", "W2"),
                    source_models: str = "prior",
                    supplement_with_unwise_catalog: bool = True,
                    catalog_cache_dir=DEFAULT_WISE_CACHE_DIR / "catalogs",
                    inflate_uncertainties: bool = True,
                    mask_bad_bits: int | None = UNWISE_BAD_BITS,
                    return_model_state: bool = False) -> dict:
    """Forced photometry on W1/W2 at the Euclid positions, with a jointly-fit
    constant sky.

    Parameters
    ----------
    sources : list of VIS-fitted Tractor sources.
    wise_cutouts : output of ``fetch_unwise_cutouts``.
    ra, dec, cutout_size_arcsec : geometry of the Euclid field.
    prior_cutout : optional prior-band Cutout, providing ``wcs`` and ``shape``.
        Use its native footprint to select outside-prior catalogue sources,
        matching the default one-pixel catalogue trimming margin. Without
        it, the footprint is approximated by a square in sky coordinates.
    psf_stamps : optional dict[band] -> stamp; built via ``get_wise_psf`` at
        the field centre if missing.
    source_models : {'prior', 'point'}
        ``'prior'`` keeps each VIS-fitted model class and shape, frozen,
        with flux the only free parameter, the same treatment as the NISP
        bands. ``'point'`` collapses every source to a PointSource.
        This approximation can change blended fluxes even when the
        intrinsic profiles are smaller than the WISE PSF.
    supplement_with_unwise_catalog : whether to add outside-prior point sources.
        When requested, an unavailable catalogue raises an error before fitting.
        Set False only to deliberately fit without the additional neighbours.
    catalog_cache_dir : exact-query cache for the outside-prior catalogue.
        Set None to disable caching.
    inflate_uncertainties : whether to scale the formal flux errors by the
                            chi scatter on source-sparse pixels
                            (:func:`euclid_phot.forced.chi_inflation`).
    return_model_state : retain the fitted image and source models for
        component plots. Does not run any extra fit.

    Returns
    -------
    dict with per-band entries ``flux_ujy``, ``flux_err_ujy``, ``sky``,
    ``chi_inflation`` (one value per band) and the residuals for diagnostics.

    Notes
    -----
    ``R.IV`` supplies template information diagonals, not the inverse of
    the full source-and-sky covariance matrix. Thus ``flux_err_ujy`` omits
    blend covariance even after residual scaling. The scaling pool uses
    the source model after subtracting the fitted sky; it is an empirical
    diagnostic, not a validated uncertainty model for crowded WISE data.
    """
    if source_models not in ("prior", "point"):
        raise ValueError(
            f"source_models = {source_models!r}; must be 'prior' or 'point'.")
    disable_galaxy_cache()
    coadd_id = wise_cutouts["coadd_id"]
    if psf_stamps is None:
        psf_stamps = {
            "W1": get_wise_psf(1, coadd_id, ra=ra, dec=dec),
            "W2": get_wise_psf(2, coadd_id, ra=ra, dec=dec),
        }

    # Pad by 5x WISE FWHM per edge so edge sources keep their PSF wings
    # inside the fit region; below ~3x FWHM the lost wing flux is
    # mis-attributed to in-cutout neighbors.
    fit_size_arcsec = float(cutout_size_arcsec + 10 * WISE_FWHM_ARCSEC)

    neighbours = None
    if supplement_with_unwise_catalog:
        src_coord = SkyCoord(
            [s.getPosition().ra for s in sources],
            [s.getPosition().dec for s in sources], unit="deg")
        # Buffer = fit box padded by 4 FWHM for wings, minus the target box.
        try:
            neighbours = unwise_neighbours(
                ra, dec, src_coord,
                field_halfwidth_arcsec=cutout_size_arcsec / 2.0,
                buffer_halfwidth_arcsec=fit_size_arcsec / 2.0
                + 4 * WISE_FWHM_ARCSEC,
                prior_cutout=prior_cutout,
                catalog_cache_dir=catalog_cache_dir)
        except Exception as exc:
            raise RuntimeError(
                "Required unWISE neighbour catalogue unavailable. Restore its "
                "query cache or retry with archive access. No WISE fit was run. "
                "Use supplement_with_unwise_catalog=False only if omitting "
                "outside-prior sources is intentional.") from exc

    results: dict = {}
    n_prior = len(sources)
    for band in bands:
        if band not in wise_cutouts:
            continue
        # 1. The Tractor image: the unWISE fit region in Vega nanomaggies.
        data, ivar, wcs_fit = unwise_fit_region(
            wise_cutouts[band], ra, dec, fit_size_arcsec,
            mask_bad_bits=mask_bad_bits)
        stamp = psf_stamps[band]
        tim = make_tractor_image(
            data, ivar, wcs_fit, stamp, band=band,
            nmgy_to_data=10 ** (VEGA_OFFSET[band] / 2.5), name=f"unWISE-{band}")

        # 2. VIS models, with a render patch that does not crop the PSF wings,
        #    plus unWISE-catalog neighbours outside the prior footprint.
        band_sources = clone_sources(
            sources, band, point=(source_models == "point"),
            psf_halfsize_pix=int(stamp.shape[0]) // 2,
            pixel_scale_arcsec=UNWISE_PIXEL_SCALE)
        if neighbours is not None:
            band_sources += neighbour_point_sources(*neighbours, band)

        # 3. Flux-only solve with a joint constant sky.
        tractor = flux_tractor(tim, band_sources, optimizer="scaled", fit_sky=True)
        R = solve_fluxes(tractor, fit_sky=True)
        flux_iv = flux_inverse_variance(R, tractor, len(band_sources))
        sky_value = float(tim.getSky().getValue())
        model = np.asarray(tractor.getModelImage(0))

        # 4. Errors scaled by the residual chi scatter.
        if inflate_uncertainties:
            chi_infl, chi_pool_kind, n_chi_pool = chi_inflation(
                data, model, ivar, sky=sky_value)
        else:
            chi_infl, chi_pool_kind, n_chi_pool = 1.0, "disabled", 0
        # Only the first n_prior entries align with the input list;
        # neighbours are not exported.
        flux_ujy = fluxes_ujy(band_sources, band)[:n_prior]
        flux_err_ujy = flux_errors_ujy(flux_iv, chi_infl)[:n_prior]

        stamp = np.asarray(stamp, dtype=float)
        py, px = np.array(stamp.shape) // 2
        total_psf = float(stamp.sum())
        central_psf = float(stamp[max(0, py - 9):py + 10,
                                  max(0, px - 9):px + 10].sum())

        results[band] = {
            "flux_ujy": flux_ujy,
            "flux_err_ujy": flux_err_ujy,
            "sky": sky_value,
            "chi_inflation": chi_infl,
            "chi_pool_kind": chi_pool_kind,
            "n_chi_pool": n_chi_pool,
            "n_supplement_sources": len(band_sources) - n_prior,
            "supplement_status": "included" if supplement_with_unwise_catalog else "disabled",
            "solver": tractor.optimizer.diagnostics,
            "psf_normalization": {
                "stamp_shape": list(stamp.shape),
                "stamp_sum": total_psf,
                "central_19_sum": central_psf,
                "central_19_fraction": central_psf / total_psf,
            },
            "residual": data - model,
            "model": model,
            "data": data,
            "invvar": ivar,
            "wcs": wcs_fit,
        }
        if return_model_state:
            results[band]["fit_image"] = tim
            results[band]["fit_sources"] = band_sources
    return results
