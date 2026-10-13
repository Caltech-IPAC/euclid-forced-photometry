"""unWISE W1 / W2 forced photometry with the VIS source models.

Only what is unWISE's own lives here: coadd cutouts and bitmask
(``data``), the PSF model (``psf``), the unWISE catalog for outside
neighbours and the comparison (``catalog``), and the fit setup (``fit``).
The fit itself is the shared recipe of :mod:`euclid_phot.forced`.
Notebook 02 shows the workflow.
"""
from .catalog import compare_unwise_2019, query_unwise_2019, unwise_neighbours, vega_mag_to_ujy
from .config import (
    UNWISE_BAD_BITS,
    UNWISE_GHOST_BITS,
    UNWISE_HALO_BITS,
    UNWISE_PIXEL_SCALE,
    UNWISE_SPIKE_BITS,
    VEGA_OFFSET,
    WISE_FWHM_ARCSEC,
)
from .data import fetch_unwise_cutouts
from .fit import fit_wise_forced, neighbour_point_sources, unwise_fit_region
from .psf import get_wise_psf

__all__ = [
    "UNWISE_BAD_BITS", "UNWISE_GHOST_BITS", "UNWISE_HALO_BITS",
    "UNWISE_PIXEL_SCALE", "UNWISE_SPIKE_BITS", "VEGA_OFFSET",
    "WISE_FWHM_ARCSEC", "compare_unwise_2019", "fetch_unwise_cutouts",
    "fit_wise_forced", "get_wise_psf", "neighbour_point_sources",
    "query_unwise_2019", "unwise_fit_region", "unwise_neighbours",
    "vega_mag_to_ujy",
]
