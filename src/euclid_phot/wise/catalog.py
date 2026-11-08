"""The unWISE catalog (Schlafly et al. 2019): neighbours and comparison."""
from __future__ import annotations

import hashlib
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table

from ..data_access import get_public_irsa_client
from ..netutils import retry
from ..scene import select_isolated_sources
from ..units import UJY_PER_NMGY
from .config import DEFAULT_WISE_CACHE_DIR, VEGA_OFFSET, WISE_FWHM_ARCSEC


def vega_mag_to_ujy(mag_vega, band: str):
    """Convert a WISE Vega magnitude to AB microJansky.

    Accepts scalar or array input. The Vega -> AB offsets are
    ``W1 = 2.699``, ``W2 = 3.339`` (Jarrett et al. 2011, the convention
    used by unWISE and CatWISE2020).

    m_AB = m_Vega + Vega_offset[band]; f_uJy = 3.631 * 10**((22.5 - m_AB) / 2.5).
    """
    m_ab = np.asarray(mag_vega, dtype=float) + VEGA_OFFSET[band]
    return UJY_PER_NMGY * 10.0 ** ((22.5 - m_ab) / 2.5)


def _query_unwise_table(adql, *, cache_dir=DEFAULT_WISE_CACHE_DIR / "catalogs"):
    """Cache the exact public query so archive outages cannot change the scene."""
    query = " ".join(adql.split())
    path = None
    if cache_dir is not None:
        key = hashlib.sha256(query.encode()).hexdigest()
        path = Path(cache_dir) / f"unwise_2019_{key}.ecsv"
        if path.exists():
            cat = Table.read(path)
            if cat.meta.get("irsa_query") != query:
                raise ValueError(f"unWISE query cache provenance mismatch: {path}")
            return cat
    client = get_public_irsa_client()
    cat = retry(lambda: client.query_tap(query=adql).to_table(),
                what="IRSA TAP unWISE query")
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        cat.meta["irsa_query"] = query
        cat.meta["irsa_endpoint"] = client.tap_url
        cat.meta["retrieved_utc"] = datetime.now(timezone.utc).isoformat()
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".ecsv", delete=False) as f:
            temporary = Path(f.name)
        try:
            cat.write(temporary, format="ascii.ecsv", overwrite=True)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return cat


def query_unwise_2019(ra: float, dec: float, radius_arcsec: float, *,
                      cache_dir=DEFAULT_WISE_CACHE_DIR / "catalogs"):
    """Pull the unWISE catalog (Schlafly et al. 2019) inside a sky circle.

    An external comparison for ``fit_wise_forced``. Schlafly+2019 used
    simultaneous point-source fitting (crowdsource), with different
    source lists, PSF normalization, and earlier-epoch coadds than neo7.

    The IRSA TAP table is ``unwise_2019``. Fluxes are stored in Vega
    nanomaggies (``flux_1`` for W1, ``flux_2`` for W2) and converted to AB
    microJansky via the Jarrett et al. 2011 Vega offsets (W1=2.699,
    W2=3.339) and 3.631 microJy = 1 AB nanomaggy. Primary detections
    in either band are included; primary status removes tile duplicates
    and does not certify photometric quality. Apply per-band validity
    and quality cuts when comparing fluxes.
    Successful queries are cached in ``cache_dir``; use ``None`` to disable caching.

    Returns
    -------
    astropy.table.Table with columns ``ra``, ``dec``, ``flux_1``,
    ``dflux_1``, ``flux_2``, ``dflux_2``, ``primary_1``, ``primary_2``,
    ``flags_unwise_1``, ``flags_unwise_2``, plus derived ``w1_ujy``,
    ``w1_err_ujy``, ``w2_ujy``, ``w2_err_ujy`` in AB microJansky.
    """
    import math
    for name, val in (("ra", ra), ("dec", dec), ("radius_arcsec", radius_arcsec)):
        if not math.isfinite(val):
            raise ValueError(f"{name} must be finite, got {val!r}")
    if radius_arcsec <= 0:
        raise ValueError(f"radius_arcsec must be positive, got {radius_arcsec}")

    radius_deg = radius_arcsec / 3600.0
    adql = f"""
    SELECT ra, dec,
           flux_1, dflux_1, flux_2, dflux_2,
           primary_1, primary_2,
           flags_unwise_1, flags_unwise_2
    FROM unwise_2019
    WHERE CONTAINS(POINT('J2000', ra, dec),
                   CIRCLE('J2000', {float(ra)}, {float(dec)}, {radius_deg}))=1
      AND (primary_1 = 1 OR primary_2 = 1)
    """
    cat = _query_unwise_table(adql, cache_dir=cache_dir)

    def _col(col):
        return np.asarray(
            col.filled(np.nan) if hasattr(col, "filled") else col,
            dtype=float)

    f1 = _col(cat["flux_1"])
    f2 = _col(cat["flux_2"])
    df1 = _col(cat["dflux_1"])
    df2 = _col(cat["dflux_2"])
    # Vega nMgy -> AB nMgy: f_AB = f_Vega * 10^(-VegaOffset/2.5).
    # AB nMgy -> AB microJansky: 3.631.
    s1 = 10 ** (-VEGA_OFFSET["W1"] / 2.5) * UJY_PER_NMGY
    s2 = 10 ** (-VEGA_OFFSET["W2"] / 2.5) * UJY_PER_NMGY
    cat["w1_ujy"] = f1 * s1
    cat["w2_ujy"] = f2 * s2
    cat["w1_err_ujy"] = df1 * s1
    cat["w2_err_ujy"] = df2 * s2
    return cat


def unwise_neighbours(ra: float, dec: float,
                        existing_coords: SkyCoord,
                        *,
                        field_halfwidth_arcsec: float,
                        buffer_halfwidth_arcsec: float,
                        merge_sep_arcsec: float = 3.0,
                        prior_cutout=None,
                        catalog_cache_dir=DEFAULT_WISE_CACHE_DIR / "catalogs"):
    """Pull unWISE-catalog sources in the PSF-wing buffer outside the Euclid
    footprint (but inside the WISE buffer), excluding any within
    ``merge_sep_arcsec`` of an existing Euclid source.

    With ``prior_cutout``, the footprint follows its WCS and the default
    one-pixel margin used by ``trim_catalog_to_cutout``. Without it, an
    axis-aligned sky square is used. Interior candidates are excluded
    because some may be blends or substructure of existing priors.
    """
    # Circle must reach the square buffer's corners.
    query_radius_arcsec = buffer_halfwidth_arcsec * np.sqrt(2.0) + merge_sep_arcsec
    query_radius_deg = query_radius_arcsec / 3600.0
    # Admit a primary detection in either band: the same list feeds both
    # fits, and a W2-primary source sub-threshold in W1 must still absorb
    # W2 wing flux.
    adql = f"""
    SELECT ra, dec, flux_1, flux_2
    FROM unwise_2019
    WHERE CONTAINS(POINT('J2000',ra,dec),
                   CIRCLE('J2000',{ra},{dec},{query_radius_deg}))=1
      AND (primary_1 = 1 OR primary_2 = 1)
    """
    uw_cat = _query_unwise_table(adql, cache_dir=catalog_cache_dir)
    if len(uw_cat) == 0:
        return uw_cat, np.zeros(0, dtype=bool)
    # TAP row order is non-deterministic, and a near-degenerate LSQR solve
    # is sensitive to column order at the ~1% level; sort for bit-identical
    # reruns.
    uw_cat.sort(["ra", "dec"])
    uw_coord = SkyCoord(uw_cat["ra"], uw_cat["dec"], unit="deg")
    # Offsets from field center in arcsec; RA scaled by cos(dec), wrapped at 0/360.
    uw_ra = np.asarray(uw_cat["ra"], dtype=float)
    uw_dec = np.asarray(uw_cat["dec"], dtype=float)
    dra = (uw_ra - ra + 180.0) % 360.0 - 180.0
    dx = np.abs(dra) * np.cos(np.radians(dec)) * 3600.0
    dy = np.abs(uw_dec - dec) * 3600.0
    inside_buffer = (dx <= buffer_halfwidth_arcsec) & (dy <= buffer_halfwidth_arcsec)
    if prior_cutout is None:
        inside_field = (dx <= field_halfwidth_arcsec) & (dy <= field_halfwidth_arcsec)
    else:
        height, width = prior_cutout.shape
        px, py = prior_cutout.wcs.world_to_pixel_values(uw_ra, uw_dec)
        inside_field = ((px >= 1) & (px < width - 1)
                        & (py >= 1) & (py < height - 1))
    keep = inside_buffer & ~inside_field
    if len(existing_coords) > 0:
        for k in np.nonzero(keep)[0]:
            if uw_coord[k].separation(existing_coords).arcsec.min() < merge_sep_arcsec:
                keep[k] = False
    return uw_cat, keep


def compare_unwise_2019(sources, wise_results, unwise_catalog, psf_stamps, *,
                        vis_flux, flux_quality=None, match_radius_arcsec=1.0,
                        minimum_snr=5.0):
    """Match a WISE fit to the Schlafly et al. (2019) unWISE catalog.

    Schlafly et al. normalize their PSF within the central 19x19 pixels, so
    our full-stamp fluxes and errors are scaled by that fraction of the
    adopted PSF for the comparison only. Returns ``{band: Table}`` aligned
    with ``sources``: ``catalog_flux_ujy``, ``catalog_flux_err_ujy``,
    ``flux_ujy`` and ``flux_err_ujy`` (19x19 convention), ``snr``,
    ``selected`` (unique match to a primary, unflagged detection with
    positive finite fluxes), ``isolated`` (no VIS neighbour brighter than a
    third of the source within two WISE FWHM) and ``clean`` (selected,
    isolated and ``snr > minimum_snr``). ``meta`` holds the PSF factor.
    """
    ra = np.array([float(s.getPosition().ra) for s in sources])
    dec = np.array([float(s.getPosition().dec) for s in sources])
    idx, sep, _ = SkyCoord(ra, dec, unit="deg").match_to_catalog_sky(
        SkyCoord(unwise_catalog["ra"], unwise_catalog["dec"], unit="deg"))
    matched = sep.arcsec < match_radius_arcsec
    multiplicity = np.bincount(idx[matched], minlength=len(unwise_catalog))
    unique = matched & (multiplicity[idx] == 1)
    isolated = select_isolated_sources(ra, dec, vis_flux,
                                       radius_arcsec=2 * WISE_FWHM_ARCSEC)
    quality = (np.ones(len(sources), bool) if flux_quality is None
               else np.asarray(flux_quality, bool))

    out = {}
    for band, r in wise_results.items():
        psf = psf_stamps[band]
        cy, cx = np.array(psf.shape) // 2
        factor = float(psf[cy - 9:cy + 10, cx - 9:cx + 10].sum() / psf.sum())
        flux = np.asarray(r["flux_ujy"], float) * factor
        err = np.asarray(r["flux_err_ujy"], float) * factor
        ref = np.asarray(unwise_catalog[band.lower() + "_ujy"], float)[idx]
        ref_err = np.asarray(unwise_catalog[band.lower() + "_err_ujy"], float)[idx]
        primary = np.asarray(np.ma.filled(
            unwise_catalog["primary_" + band[1]], 0))[idx] == 1
        unflagged = np.asarray(np.ma.filled(
            unwise_catalog["flags_unwise_" + band[1]], -1))[idx] == 0
        snr = np.divide(flux, err, out=np.zeros_like(flux), where=err > 0)
        selected = (unique & primary & unflagged & quality
                    & np.isfinite(flux) & (flux > 0) & np.isfinite(err) & (err > 0)
                    & np.isfinite(ref) & (ref > 0)
                    & np.isfinite(ref_err) & (ref_err > 0))
        table = Table({"catalog_flux_ujy": ref, "catalog_flux_err_ujy": ref_err,
                       "flux_ujy": flux, "flux_err_ujy": err, "snr": snr,
                       "selected": selected, "isolated": isolated,
                       "clean": selected & isolated & (snr > minimum_snr)})
        table.meta.update(psf_factor=factor, n_catalog=len(unwise_catalog),
                          n_unique=int(unique.sum()), minimum_snr=minimum_snr)
        out[band] = table
    return out
