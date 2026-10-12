"""The Tractor ``Image`` every band is fitted on.

A Tractor image needs five things, whatever the telescope:

- ``data``: the science pixels, in the image's own units;
- ``invvar``: the inverse variance of each pixel (0 = ignore the pixel);
- ``psf``: a PSF stamp, or a PSF object whose stamp varies with position;
- ``wcs``: the astropy WCS, wrapped for Tractor by ``AstropyWCSAdapter``;
- ``nmgy_to_data``: image units per AB nanomaggy, so that fitted
  brightnesses come out in nanomaggies (see :mod:`euclid_phot.units`).

Each band module only prepares these inputs (Euclid MER cutouts, unWISE
coadds, IRAC mosaics, a user's ground image) and calls
:func:`make_tractor_image`.
"""
from __future__ import annotations

import numpy as np
from tractor import ConstantSky, Image, LinearPhotoCal
from tractor.psf import PixelizedPSF
from tractor.wcs import ConstantFitsWcs


class AstropyWCSAdapter:
    """Wrap an astropy WCS so Tractor's ConstantFitsWcs can call into it.

    Uses the low-level ``*_values`` API: the high-level SkyCoord path
    silently transforms frames for headers that omit RADESYS (like unWISE),
    giving wrong pixel coordinates.
    """
    def __init__(self, astropy_wcs):
        self._wcs = astropy_wcs

    def radec2pixelxy(self, ra, dec):
        px, py = self._wcs.world_to_pixel_values(ra, dec)
        return True, float(px) + 1.0, float(py) + 1.0

    def pixelxy2radec(self, x, y):
        ra, dec = self._wcs.pixel_to_world_values(x - 1.0, y - 1.0)
        return float(ra), float(dec)

    def pixel_scale(self):
        ps = self._wcs.proj_plane_pixel_scales()[0]
        ps_deg = float(ps.value) if hasattr(ps, "value") else float(ps)
        return ps_deg * 3600.0

    def get_cd(self):
        cd = self._wcs.pixel_scale_matrix
        return [cd[0, 0], cd[0, 1], cd[1, 0], cd[1, 1]]


def invvar_from_rms(rms: np.ndarray, data: np.ndarray | None = None) -> np.ndarray:
    """Inverse variance from an RMS map.

    Pixels with non-positive or non-finite RMS get invvar=0. If ``data`` is
    given, pixels with non-finite data are also zeroed.
    """
    invvar = np.where((rms > 0) & np.isfinite(rms), 1.0 / rms ** 2, 0.0)
    if data is not None:
        invvar = np.where(np.isfinite(data), invvar, 0.0)
    return invvar


def apply_bitmask(invvar: np.ndarray, flag: np.ndarray | None,
                  bad_bits: int | None) -> np.ndarray:
    """Zero the inverse variance on flagged pixels.

    ``bad_bits`` is a bitmask: a pixel is dropped where ``flag & bad_bits``
    is non-zero. If ``bad_bits`` is None, any non-zero flag is treated as bad.
    """
    if flag is None:
        return invvar
    flag = np.asarray(flag)
    bad = (flag != 0) if bad_bits is None else ((flag & int(bad_bits)) != 0)
    return np.where(bad, 0.0, invvar)


def make_tractor_image(data, invvar, wcs, psf, *, band: str,
                       nmgy_to_data: float = 1.0, sky: float = 0.0,
                       name: str | None = None) -> Image:
    """Build the Tractor ``Image`` for one band.

    Parameters
    ----------
    data, invvar : 2-D arrays on the same pixel grid.
    wcs : astropy WCS of that grid.
    psf : 2-D stamp (normalised here to unit sum) or a Tractor PSF object,
        e.g. :class:`euclid_phot.pixel_psf.SpatialPixelizedPSF`.
    band : brightness key of the sources fitted on this image.
    nmgy_to_data : image units per AB nanomaggy.
    sky : constant sky level, frozen unless the fit thaws it.
    """
    if not hasattr(psf, "getPointSourcePatch"):
        stamp = np.asarray(psf)
        total = float(np.sum(stamp))
        if not np.isfinite(total) or total <= 0:
            raise ValueError(
                f"PSF stamp has non-finite or non-positive sum ({total})")
        psf = PixelizedPSF(stamp / total)
    return Image(
        data=data, invvar=invvar, psf=psf,
        wcs=ConstantFitsWcs(AstropyWCSAdapter(wcs)),
        photocal=LinearPhotoCal(nmgy_to_data, band=band),
        sky=ConstantSky(sky), name=name or band)
