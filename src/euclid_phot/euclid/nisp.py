"""Forced photometry on NISP Y/J/H bands given VIS-fitted sources.

For each NISP band: the band's Tractor image (:func:`build_tractor_image`),
the VIS-fitted sources cloned with positions and shapes kept, and the shared
flux-only solve of :mod:`euclid_phot.forced`. NISP needs nothing else of its
own: the MER mosaics share the VIS pixel grid, and errors are calibrated
afterwards on empty positions (:mod:`euclid_phot.calibrate`). Returns fluxes
in microJanskys, aligned 1:1 with the input source list.
"""
from __future__ import annotations

import numpy as np

from ..forced import (
    flux_errors_ujy,
    flux_inverse_variance,
    flux_tractor,
    fluxes_ujy,
    solve_fluxes,
)
from ..sources import clone_sources
from .image import build_tractor_image


def _fit_one_nisp_band(args):
    """Worker for ``fit_nisp_forced``: one band, runnable in a thread."""
    sources, band, cutout, psf_stamp, pixel_mask, psf_data = args
    tim = build_tractor_image(cutout, psf_stamp, psf_data=psf_data,
                              pixel_mask=pixel_mask)
    band_sources = clone_sources(sources, band)
    tractor = flux_tractor(tim, band_sources)
    R = solve_fluxes(tractor)
    flux_err_ujy = flux_errors_ujy(
        flux_inverse_variance(R, tractor, len(band_sources)))
    return band, {"flux_ujy": fluxes_ujy(band_sources, band),
                  "flux_err_ujy": flux_err_ujy}


def fit_nisp_forced(sources, cutouts: dict, psf_stamps: dict | None = None,
                    *,
                    bands: tuple = ("Y", "J", "H"),
                    psf_data: dict | None = None,
                    pixel_mask: np.ndarray | None = None,
                    n_workers: int = 1) -> dict:
    """Run forced photometry on each NISP band.

    Parameters
    ----------
    sources : list of Tractor sources (from the VIS step)
    cutouts : dict[band] -> Cutout (output of fetch_cutout)
    psf_stamps : dict[band] -> ndarray, optional
        Constant PSFs; not needed when ``psf_data`` supplies local samples.
    psf_data : dict[band] -> PSF samples, optional
        Render every source through its nearest sample in that band.
        If omitted, use the constant ``psf_stamps`` as before.
    bands : tuple
        Subset of NISP bands to fit.
    pixel_mask : ndarray of bool, optional
        Pixels to exclude from every band fit (inverse variance zeroed
        there). All MER mosaics are resampled onto the common 0.1 arcsec
        VIS grid, so the prior-band bright-star mask (the STARSIGNAL
        footprints) applies to the NISP cutouts unchanged.
    n_workers : int
        If > 1, fit the bands in parallel threads. The speedup depends on
        your Tractor build, since its compiled FFT may or may not run on
        several threads at once (~1.4x at n_workers=4 on the reference build).

    Returns
    -------
    dict[band] -> dict with keys ``flux_ujy`` and ``flux_err_ujy``, each an
        ndarray in microJansky aligned 1:1 with ``sources``. ``flux_err_ujy``
        is the formal 1-sigma uncertainty (1/sqrt of the Tractor
        inverse-variance) from the flux-only fit.
    """
    work = [(sources, band, cutouts[band],
             psf_stamps[band] if psf_data is None and psf_stamps is not None else None, pixel_mask,
             psf_data[band] if psf_data is not None else None)
            for band in bands if band in cutouts]
    if n_workers <= 1 or len(work) <= 1:
        results = [_fit_one_nisp_band(w) for w in work]
    else:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(
                min(n_workers, len(work))) as exec_:
            results = list(exec_.map(_fit_one_nisp_band, work))
    return dict(results)
