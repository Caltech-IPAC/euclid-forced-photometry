"""The user's ground image: science, uncertainty and PSF in microJansky.

:func:`make_image` validates the arrays and converts them to uJy/pixel, the
unit every band of this package reports. Without a measured PSF stamp,
:func:`gaussian_psf` makes a placeholder from the seeing FWHM.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS

from ..units import UJY_PER_NMGY, pixel_scale_arcsec


@dataclass(frozen=True)
class SmallImage:
    """Science and inverse variance in uJy/pixel, celestial WCS, unit-sum PSF."""
    science_ujy: np.ndarray
    invvar_ujy: np.ndarray
    wcs: WCS
    psf: np.ndarray
    band: str | None


@dataclass(frozen=True)
class GroundInputs:
    image: SmallImage
    science_wcs: WCS
    pixel_scale_arcsec: float


def gaussian_psf(fwhm_arcsec: float, pixel_scale_arcsec: float) -> np.ndarray:
    """Return a unit-sum, pixel-sampled Gaussian PSF for a supplied FWHM."""
    if not (np.isfinite(fwhm_arcsec) and fwhm_arcsec > 0):
        raise ValueError("FWHM must be finite and positive")
    if not (np.isfinite(pixel_scale_arcsec) and pixel_scale_arcsec > 0):
        raise ValueError("pixel scale must be finite and positive")
    sigma = fwhm_arcsec / (2 * np.sqrt(2 * np.log(2)) * pixel_scale_arcsec)
    radius = max(4, int(np.ceil(5 * sigma)))
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    stamp = np.exp(-(xx * xx + yy * yy) / (2 * sigma * sigma))
    return stamp / stamp.sum()


def read_fits_image(path: str | Path, ext: int | str = 0) -> tuple[np.ndarray, fits.Header]:
    """Read one explicitly selected FITS image plane and its header."""
    with fits.open(path, memmap=False) as hdus:
        hdu = hdus[ext]
        if hdu.data is None:
            raise ValueError(f"{path}[{ext!r}] contains no image data")
        return np.asarray(hdu.data, dtype=float).copy(), hdu.header.copy()


def make_image(science, uncertainty, psf, science_unit: u.UnitBase | u.Quantity,
               science_wcs: WCS, *, band: str = "ground") -> GroundInputs:
    """Validate user arrays and convert science and 1-sigma RMS to uJy/pixel.

    ``science_unit`` is the unit of one input pixel value, e.g.
    ``u.nanomaggy`` for SDSS or ``u.uJy`` for a microJy image. A calibrated
    DN image can instead supply a quantity such as ``4.7 * u.nJy`` per DN.
    The uncertainty array must be RMS in the same native units. ``science_wcs``
    is the celestial WCS of the science array; ``band`` is an optional label.
    """
    declared_unit = (1 * science_unit if isinstance(science_unit, u.UnitBase)
                     else u.Quantity(science_unit))
    # Astropy keeps maggy as a separate physical type, so apply its AB
    # definition explicitly: 1 nanomaggy = 3.631 uJy.
    unit = (declared_unit.to_value(u.nanomaggy) * UJY_PER_NMGY
            if declared_unit.unit.is_equivalent(u.maggy)
            else declared_unit.to_value(u.uJy))
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
    return GroundInputs(image, wcs, pixel_scale_arcsec(wcs))
