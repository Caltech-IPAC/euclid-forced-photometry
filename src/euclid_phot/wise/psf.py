"""The unWISE PSF stamp at a field position (``unwise_psf`` model)."""
from __future__ import annotations

import numpy as np
from astropy.io import fits

from .config import WISE_COADD_VERSION


def get_wise_psf(band_number: int, coadd_id: str, *, sidelen: int = 151,
                 modelname: str | None = None,
                 ra: float | None = None, dec: float | None = None) -> np.ndarray:
    """Return a unit-flux-normalized unWISE PSF stamp.

    ``unwise_psf.get_unwise_psf`` has a Python-3 slice-indexing bug when
    ``sidelen`` is passed; we trim ourselves to an odd stamp.

    ``modelname`` defaults to the model named for ``WISE_COADD_VERSION``
    (``"neo7_unwisecat"`` for neo7). An unavailable named model triggers
    a warning before using the library default. Unit normalization does
    not establish agreement with an external catalogue's flux scale.

    Supply ``ra`` and ``dec`` together to evaluate the library's scan-angle
    prescription at the field position. Without them, use the coadd centre.
    A single field-centre stamp still approximates variation across the field.
    """
    if (ra is None) != (dec is None):
        raise ValueError("Supply both ra and dec, or neither.")
    if ra is not None and (not np.isfinite(ra) or not np.isfinite(dec)
                           or not -90 <= dec <= 90):
        raise ValueError("PSF coordinates must be finite with -90 <= dec <= 90.")
    try:
        from unwise_psf import unwise_psf as up
    except ImportError as exc:
        raise ImportError(
            "The unWISE W1/W2 step needs the `unwise_psf` PSF model, which the "
            "upstream repo does not package for pip. Install it manually:\n"
            "    git clone https://github.com/legacysurvey/unwise_psf\n"
            "    export PYTHONPATH=\"$PWD/unwise_psf/py:$PYTHONPATH\"\n"
            "(plus `pip install fitsio 'setuptools<81'`; unwise_psf reads its "
            "model files with fitsio and imports the deprecated pkg_resources "
            "API). VIS + NISP run without it."
        ) from exc
    if modelname is None:
        modelname = f"{WISE_COADD_VERSION}_unwisecat"

    def at_position(name):
        if ra is None:
            return up.get_unwise_psf(band_number, coadd_id, modelname=name)
        suffix = f"_{name}" if name else ""
        path = up.get_resource_filename(
            "unwise_psf", f"data/psf_model_w{band_number}{suffix}.fits")
        model = fits.getdata(path)
        model = up.average_two_scandirs(model)
        return up.rotate_using_rd(model, coadd_id, ra=float(ra), dec=float(dec))

    import warnings
    fallback = False
    # unwise_psf hits the deprecated pkg_resources API when resolving a
    # model file; suppress only that warning.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*pkg_resources.*")
        try:
            full = at_position(modelname)
        except OSError:
            # *_unwisecat models exist only for some versions/bands.
            fallback = True
            full = at_position("")
    if fallback:
        warnings.warn(
            f"unwise_psf has no '{modelname}' model for W{band_number}; "
            "falling back to the default PSF model.", stacklevel=2)
    h = full.shape[0] // 2
    half = sidelen // 2
    stamp = full[h - half:h + half + 1, h - half:h + half + 1].astype(np.float32)
    stamp /= stamp.sum()
    return stamp
