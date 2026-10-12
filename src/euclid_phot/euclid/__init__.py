"""Euclid MER data: the prior band (VIS) and the NISP Y / J / H bands.

Only what is Euclid's own lives here: the MER catalog query (``catalog``),
MER mosaic cutouts (``cutouts``), the MER PSF products (``psf``), the FLG
bits and bright-star masks (``flags``, ``config``), the MER cutout's
Tractor image (``image``) and the NISP band loop (``nisp``). The fits use
the shared recipe of :mod:`euclid_phot.forced`.
"""
