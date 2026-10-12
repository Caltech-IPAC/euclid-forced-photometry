"""Check ground-image fluxes against the SDSS catalog."""

from __future__ import annotations

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

from ..units import UJY_PER_NMGY


def compare_sdss_g(measured: Table, reference: Table,
                   *, max_sep_arcsec: float = 0.7) -> Table:
    """Independent SDSS check: PSF flux for stars, cModel flux for galaxies.

    SDSS fluxes are nanomaggies; all comparison fluxes are in microJansky.
    """
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
    ref_flux = np.asarray(native_flux, float) * UJY_PER_NMGY
    with np.errstate(divide="ignore", invalid="ignore"):
        ref_err = UJY_PER_NMGY / np.sqrt(np.asarray(native_ivar, float))
    band = measured.meta["BAND"]
    fitted_flux = np.asarray(measured[f"flux_{band}_ujy"][nearest], float)
    fitted_err = np.asarray(measured[f"flux_err_{band}_ujy"][nearest], float)
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
    comparison["reference_flux_ujy"] = ref_flux
    comparison["reference_fluxerr_ujy"] = ref_err
    comparison["forced_flux_ujy"] = fitted_flux
    comparison["forced_fluxerr_ujy"] = fitted_err
    comparison["accepted"] = good
    return comparison
