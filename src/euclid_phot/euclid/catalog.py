"""Shared MER catalog query for the release selected by set_data_access."""
from __future__ import annotations

import warnings
import numpy as np
from astropy.table import MaskedColumn

from ..data_access import get_access_settings, get_data_access, get_irsa_client
from ..netutils import retry

def _ra_where(ra: float, half_size_ra_deg: float) -> str:
    """ADQL RA box constraint, handling the RA = 0/360 wrap.

    A box that stays inside [0, 360] is a plain BETWEEN; one that straddles
    the wrap becomes the union of the two arcs (``ra >= a OR ra <= b``).
    """
    ra_min = ra - half_size_ra_deg
    ra_max = ra + half_size_ra_deg
    if ra_min >= 0.0 and ra_max <= 360.0:
        return f"m.ra BETWEEN {ra_min} AND {ra_max}"
    a = ra_min % 360.0
    b = ra_max % 360.0
    return f"(m.ra >= {a} OR m.ra <= {b})"


_SELECT = """
SELECT
    m.object_id, m.ra, m.dec,
    m.flux_vis_psf, m.fluxerr_vis_psf,
    m.flux_vis_sersic, m.fluxerr_vis_sersic,
    m.flux_h_templfit, m.fluxerr_h_templfit,
    m.flux_h_sersic, m.fluxerr_h_sersic,
    m.flux_j_templfit, m.fluxerr_j_templfit,
    m.flux_j_sersic, m.fluxerr_j_sersic,
    m.flux_y_templfit, m.fluxerr_y_templfit,
    m.flux_y_sersic, m.fluxerr_y_sersic,
    m.flux_detection_total,
    m.semimajor_axis, m.ellipticity, m.position_angle,
    m.point_like_prob, m.det_quality_flag,
    m.gal_ebv, m.gal_ebv_err
"""
_MORPHOLOGY_COLUMNS = {
    "sersic_sersic_vis_radius": "arcsec",
    "sersic_sersic_vis_index": None,
    "sersic_sersic_vis_axis_ratio": None,
    "sersic_angle": "deg",
}


def query_mer_catalog(ra: float, dec: float, half_size_deg: float, *,
                      brightness_cut_ujy: float = 0.0,
                      require_detection_band: str | None = None,
                      star_prob_threshold: float = 0.96):
    """Query the selected MER table, preserving the common downstream schema.

    The spatial and VIS flux cuts match the Q1 query. If the configured
    morphology table is null, its four columns are masked: the unchanged
    source builder then uses its isophotal/exponential fallback. Morphology
    is joined only from the selected release. Missing point-like probabilities
    follow the same conservative star selection as the Q1 query.
    """
    settings = get_access_settings()
    release = get_data_access()
    table = settings["catalog_table"]
    morphology = settings["morphology_table"]
    select = _SELECT.rstrip()
    join = ""
    if morphology:
        select += ",\n    " + ",\n    ".join(
            f"morph.{name}" for name in _MORPHOLOGY_COLUMNS)
        join = (f"LEFT JOIN {morphology} AS morph "
                "ON m.object_id = morph.object_id\n")

    extra = ""
    if require_detection_band is not None:
        band = require_detection_band.lower()
        if band not in ("vis", "y", "j", "h"):
            raise ValueError("require_detection_band must be VIS, Y, J or H")
        col = "flux_vis_psf" if band == "vis" else f"flux_{band}_templfit"
        extra = f"\n  AND m.{col} > 0"
    cos_dec = float(np.cos(np.radians(dec)))
    cos_edge = max(np.cos(np.radians(dec - half_size_deg)),
                   np.cos(np.radians(dec + half_size_deg)))
    radius_deg = float(np.hypot(
        half_size_deg, half_size_deg * cos_edge / cos_dec)) + 1e-3
    circle = (f"CONTAINS(POINT('ICRS', m.ra, m.dec), "
              f"CIRCLE('ICRS', {ra % 360.0}, {dec}, {radius_deg}))=1")
    adql = (select + f"\nFROM {table} AS m\n" + join
            + f"WHERE {circle}\n"
            + f"  AND {_ra_where(ra % 360.0, half_size_deg / cos_dec)}\n"
            + f"  AND m.dec BETWEEN {dec - half_size_deg} AND {dec + half_size_deg}\n"
            + f"  AND m.flux_vis_psf > {float(brightness_cut_ujy)}{extra}\n"
            + "ORDER BY m.flux_vis_psf DESC")
    client = get_irsa_client()
    mer_cat = retry(
        lambda: client.query_tap(adql, maxrec=settings["maxrec"]).to_table(),
        what=f"IRSA TAP {release} MER query")
    if len(mer_cat) == 0:
        warnings.warn(
            f"{release} MER query returned no rows at ({ra}, {dec}) in {table}.",
            stacklevel=2)
    elif len(mer_cat) >= settings["maxrec"]:
        warnings.warn(f"{release} MER query reached MAXREC and may be truncated.", stacklevel=2)

    if not morphology:
        for name, unit in _MORPHOLOGY_COLUMNS.items():
            mer_cat[name] = MaskedColumn(
                np.zeros(len(mer_cat)), mask=np.ones(len(mer_cat), dtype=bool),
                unit=unit)
        warnings.warn(
            f"{release} morphology_table is not configured in euclid_phot/conf.json; "
            "Sersic shape columns are masked. The existing source builder "
            "uses its isophotal/exponential fallback; tree-selected shapes "
            "are still fitted to the VIS pixels.", stacklevel=2)
    plp = mer_cat["point_like_prob"]
    plp_filled = plp.filled(0.0) if hasattr(plp, "filled") else np.asarray(plp, dtype=float)
    mer_cat["is_star"] = np.asarray(plp_filled, dtype=float) > star_prob_threshold
    mer_cat.meta["MERTABLE"] = table
    mer_cat.meta["MERMORPH"] = morphology or "unavailable; masked columns"
    mer_cat.meta["DATAREL"] = release
    return mer_cat
