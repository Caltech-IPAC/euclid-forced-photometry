"""Ground-image data and a simple user-generated PSF option."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy.wcs import WCS


@dataclass(frozen=True)
class SmallImage:
    science_njy: np.ndarray
    invvar_njy: np.ndarray
    wcs: WCS
    psf: np.ndarray
    band: str | None


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
