"""Which prior sources belong in a band's fit.

Every band fits the VIS sources that can put light on its pixels: the
sources inside the image, plus neighbours within a margin outside it whose
wings reach in. :func:`footprint_mask` is that test for every band;
``margin_pix`` is positive to include a ring of neighbours outside the
image and negative to keep only sources a few pixels inside it.
"""
from __future__ import annotations

import numpy as np
from astropy.coordinates import SkyCoord


def footprint_mask(ra, dec, wcs, shape, *, margin_pix: float = 0.0) -> np.ndarray:
    """True where (ra, dec) projects within ``margin_pix`` of the image.

    The kept pixel range is ``-margin_pix <= x < width + margin_pix`` (and
    likewise in y), with pixel centres at integer coordinates.
    """
    ra = np.atleast_1d(np.asarray(ra, dtype=float))
    dec = np.atleast_1d(np.asarray(dec, dtype=float))
    x, y = wcs.world_to_pixel_values(ra, dec)
    x, y = np.asarray(x, float), np.asarray(y, float)
    height, width = shape
    m = float(margin_pix)
    return (np.isfinite(x) & np.isfinite(y)
            & (x >= -m) & (x < width + m)
            & (y >= -m) & (y < height + m))


def trim_catalog_to_cutout(mer_cat, wcs, shape, *, edge_margin_pix: int = 1):
    """Drop catalog rows whose pixel positions fall off the cutout.

    Keeps rows at least ``edge_margin_pix`` inside the edge. Boundary rows
    stay in the source list (their light is in the pixels); their own
    photometry is flagged downstream (:func:`euclid_phot.euclid.flags.flag_sources`,
    column ``edge``).
    """
    if len(mer_cat) == 0:
        return mer_cat
    keep = footprint_mask(mer_cat["ra"], mer_cat["dec"], wcs, shape,
                          margin_pix=-edge_margin_pix)
    return mer_cat[keep]


def select_isolated_sources(ra, dec, flux, *, radius_arcsec: float,
                            flux_fraction: float = 1.0 / 3.0):
    """Bright-neighbor isolation criterion in a supplied reference band.

    Returns a boolean mask, True where a source has no neighbor within
    ``radius_arcsec`` whose flux exceeds ``flux_fraction`` of its own.
    Comparisons with a low-resolution band use about two of its PSF FWHM.
    This is a sample-selection choice, not a guarantee of unbiased
    photometry; isolation in the reference band does not imply isolation in
    a band with a wider PSF.

    Parameters
    ----------
    ra, dec : array-like (deg)
        Source positions (e.g. the Euclid/MER prior positions).
    flux : array-like
        Per-source flux used for the brightness-ratio test (any consistent
        unit; the band defines which neighbours count as bright).
    radius_arcsec : float
        Neighbor search radius.
    flux_fraction : float
        A neighbor disqualifies the target if its flux exceeds this fraction
        of the target flux. Default 1/3.

    Returns
    -------
    ndarray of bool, shape (N,)
    """
    ra = np.asarray(ra, dtype=float)
    dec = np.asarray(dec, dtype=float)
    flux = np.asarray(flux, dtype=float)
    n = len(ra)
    coords = SkyCoord(ra, dec, unit="deg")
    isolated = np.ones(n, dtype=bool)
    for i in range(n):
        sep = coords[i].separation(coords).arcsec
        # Neighbors within the radius, excluding self (sep == 0).
        near = (sep > 0) & (sep <= radius_arcsec)
        if not near.any():
            continue
        ti = flux[i]
        if not np.isfinite(ti) or ti <= 0:
            isolated[i] = False
            continue
        if np.any(flux[near] > flux_fraction * ti):
            isolated[i] = False
    return isolated
