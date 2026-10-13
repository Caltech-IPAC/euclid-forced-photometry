"""unWISE constants: pixel scale, PSF width, Vega offsets, mask bits."""
from __future__ import annotations

from ..config import DEFAULT_DATA_DIR

UNWISE_PIXEL_SCALE = 2.75
WISE_FWHM_ARCSEC = 6.94  # representative unWISE PSF FWHM
WISE_COADD_VERSION = "neo7"
DEFAULT_WISE_CACHE_DIR = DEFAULT_DATA_DIR / "wise"

# Vega -> AB magnitude offsets (Jarrett et al. 2011, as used by unWISE and
# CatWISE2020). unWISE images are in Vega nanomaggies.
VEGA_OFFSET = {"W1": 2.699, "W2": 3.339}


# unWISE bitmask groups (Meisner et al. 2019, PASP 131, 124504, Table 2).
# The default drops spike, halo, and ghost pixels; the broad near-bright-star
# bits (0-3) and latent bits (13-20) flag large, mostly recoverable areas and
# stay unset.
def _unwise_bit(*bits):
    return sum(1 << b for b in bits)


UNWISE_SPIKE_BITS = _unwise_bit(27, 28, 29, 30)
UNWISE_HALO_BITS = _unwise_bit(23, 24)
UNWISE_GHOST_BITS = _unwise_bit(11, 12, 25, 26)
UNWISE_BAD_BITS = UNWISE_SPIKE_BITS | UNWISE_HALO_BITS | UNWISE_GHOST_BITS
