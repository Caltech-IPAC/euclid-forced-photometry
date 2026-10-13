"""The Tractor image of a Euclid MER cutout (VIS or NISP).

MER-specific inputs for :func:`euclid_phot.tractor_image.make_tractor_image`:
the inverse variance from the RMS plane with the MER FLG bad bits removed,
the photometric scale from the header ``MAGZERO``, and either one PSF stamp
or the CATALOG/GRID-PSF samples (nearest sample per source).
"""
from __future__ import annotations

import numpy as np
from tractor import Image
from tractor.psf import PixelizedPSF

from ..pixel_psf import SpatialPixelizedPSF
from ..tractor_image import (
    AstropyWCSAdapter,
    apply_bitmask,
    invvar_from_rms,
    make_tractor_image,
)
from ..units import nmgy_to_data
from .config import mer_bad_bits_for_band

# Distinguishes the default bad-bit set from an explicit flag_bad_bits=None.
_USE_DEFAULT_BAD_BITS = object()


def build_tractor_image(cutout, psf_stamp: np.ndarray | None = None,
                        *,
                        psf_data: dict | None = None,
                        invvar: np.ndarray | None = None,
                        flag: np.ndarray | None = None,
                        flag_bad_bits=_USE_DEFAULT_BAD_BITS,
                        pixel_mask: np.ndarray | None = None,
                        name: str | None = None,
                        sky: float = 0.0,
                        mag_zero: float | None = None) -> Image:
    """Build a Tractor ``Image`` from a ``Cutout`` and a PSF stamp.

    Parameters
    ----------
    cutout : Cutout
        Output of ``fetch_cutout``; provides data, rms, wcs, header.
    psf_stamp : ndarray (H, W) float32, sum=1, optional
        The PSF used when ``psf_data`` is omitted. Not needed for local PSFs.
    psf_data : dict, optional
        CATALOG-PSF or GRID-PSF samples. When supplied, each source is
        rendered with its nearest sample, throughout fitting and residuals.
    invvar : ndarray, optional
        Override the inverse-variance map. If not given we compute it from
        ``cutout.rms`` as ``1/rms^2`` (with rms<=0 mapped to invvar=0).
    pixel_mask : ndarray of bool, optional
        Extra per-pixel veto applied on top of the FLG bits: the inverse
        variance is zeroed where True, e.g. the STARSIGNAL bright-star
        mask from :func:`euclid_phot.euclid.flags.starsignal_pixel_mask`.
    name : str, optional
        Image name string. Defaults to ``f"Euclid-{cutout.band}"``.
    sky : float
        Constant sky level for ConstantSky.
    mag_zero : float, optional
        AB magnitude zero point of the IMAGE. If None, read from
        ``cutout.header['MAGZERO']``; if that is missing, falls back to 23.9
        (the MER *catalog* flux-to-mag zero point) and emits a UserWarning.
        Pass ``mag_zero`` explicitly when the header lacks ``MAGZERO``.

    Returns
    -------
    tractor.Image
    """
    if invvar is None:
        if cutout.rms is None:
            raise ValueError("cutout.rms is None; pass an explicit invvar=")
        invvar = invvar_from_rms(cutout.rms, cutout.data)
    else:
        # Caller-supplied invvar must not pair finite weight with
        # non-finite data.
        invvar = np.where(np.isfinite(cutout.data), invvar, 0.0)

    flag = flag if flag is not None else getattr(cutout, "flag", None)
    bad_bits = (mer_bad_bits_for_band(cutout.band)
                if flag_bad_bits is _USE_DEFAULT_BAD_BITS else flag_bad_bits)
    invvar = apply_bitmask(invvar, flag, bad_bits)

    if pixel_mask is not None:
        invvar = np.where(np.asarray(pixel_mask, dtype=bool), 0.0, invvar)

    if mag_zero is None:
        if "MAGZERO" in cutout.header:
            mag_zero = cutout.header["MAGZERO"]
        else:
            # MER mosaics carry a band- and tile-dependent AB zeropoint in
            # MAGZERO (the mosaics are in ADU/s or electrons), several mag
            # from the catalog's 23.9.
            import warnings
            warnings.warn(
                f"no MAGZERO in header for band {getattr(cutout, 'band', '?')!r}; "
                "falling back to 23.9, which is almost certainly wrong for a "
                "Euclid MER mosaic. Pass mag_zero= explicitly.",
                stacklevel=2)
            mag_zero = 23.9
    if psf_data is not None:
        psf_tile = psf_data.get("tile_id")
        science_tile = cutout.header.get("MERTILE")
        if psf_tile and science_tile and str(psf_tile) != str(science_tile):
            raise ValueError("PSF samples and science cutout come from different MER tiles")
        from tractor.wcs import ConstantFitsWcs
        psf = SpatialPixelizedPSF(psf_data, ConstantFitsWcs(AstropyWCSAdapter(cutout.wcs)))
    elif psf_stamp is None:
        raise ValueError('Supply psf_data for local PSFs or a constant psf_stamp.')
    else:
        # Normalise by the full-precision sum, then store as float32.
        psf_sum = float(np.sum(psf_stamp))
        if not np.isfinite(psf_sum) or psf_sum <= 0:
            raise ValueError(
                f"psf_stamp has non-finite or non-positive sum ({psf_sum}); "
                "cannot build a PixelizedPSF from it.")
        psf = PixelizedPSF(np.asarray(psf_stamp, dtype=np.float32) / psf_sum)

    return make_tractor_image(
        np.where(np.isfinite(cutout.data), cutout.data, 0.0), invvar,
        cutout.wcs, psf, band=cutout.band, nmgy_to_data=nmgy_to_data(mag_zero),
        sky=sky, name=name or f"Euclid-{cutout.band}")
