"""Flux units and pixel geometry shared by every band.

Tractor brightnesses are AB nanomaggies (m_AB = 22.5 - 2.5 log10 f), and
every band reports fluxes in microJansky: 1 nanomaggy = 3.631 uJy.
An image enters Tractor through ``LinearPhotoCal(nmgy_to_data)``, the
number of image units per nanomaggy; :func:`nmgy_to_data` gives it for
an image with a known AB zero point.
"""
from __future__ import annotations

import numpy as np

UJY_PER_NMGY = 3.631
# AB zero points: nanomaggies (Tractor) and microJansky (catalog columns).
AB_MAG_ZP = 22.5
AB_ZP_UJY = 23.9


def nmgy_to_data(mag_zero: float) -> float:
    """Image units per AB nanomaggy for an image with AB zero point ``mag_zero``.

    ``mag_zero = 23.9`` is an image in microJansky (3.631), and
    ``mag_zero = 31.4`` an image in nanoJansky (3631).
    """
    return 10 ** (0.4 * (float(mag_zero) - AB_MAG_ZP))


def flux_to_ab_mag(flux_ujy):
    """AB magnitude from flux in microJansky. Non-positive flux -> nan."""
    f = np.asarray(flux_ujy, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        m = AB_ZP_UJY - 2.5 * np.log10(f)
    return np.where(np.isfinite(f) & (f > 0), m, np.nan)


def flux_err_to_mag_err(flux_ujy, flux_err_ujy):
    """1-sigma magnitude error from flux and flux error (both microJansky)."""
    f = np.asarray(flux_ujy, dtype=float)
    e = np.asarray(flux_err_ujy, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        m_err = (2.5 / np.log(10.0)) * (e / f)
    return np.where(np.isfinite(f) & (f > 0) & np.isfinite(e), m_err, np.nan)


def pixel_scale_arcsec(wcs) -> float:
    """Pixel side in arcsec: square root of the projected pixel area."""
    return float(np.sqrt(abs(np.linalg.det(wcs.pixel_scale_matrix))) * 3600.0)


def pixel_area_sr(wcs) -> float:
    """Projected solid angle of one image pixel in steradians."""
    return abs(float(np.linalg.det(wcs.pixel_scale_matrix))) * (np.pi / 180.0) ** 2


def pixel_area_arcsec2(wcs) -> float:
    """Projected area of one image pixel in square arcsec."""
    return abs(float(np.linalg.det(wcs.pixel_scale_matrix))) * 3600.0 ** 2
