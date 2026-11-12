"""Calibrated Rubin image inputs, kept separate from the required official PSF."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
import warnings

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.nddata.utils import NoOverlapError, PartialOverlapError
from astropy.table import Table
from astropy.wcs import WCS

from ..ground_based.image import make_image as ground_make_image


# DP1 coadd guidance (not single-visit masks): NO_DATA and SAT are exclusions;
# BAD is a defensive exclusion (not populated in DP1 coadds). INEXACT_PSF,
# CLIPPED, REJECTED and SENSOR_EDGE are conditional, not blanket exclusions.
# https://dp1.lsst.io/tutorials/notebook/202/notebook-202-6.html
DEFAULT_BAD_PLANES = ("BAD", "SAT", "NO_DATA")


def make_image(science, uncertainty, psf, science_unit, science_wcs, *, band="ground"):
    """Reuse the ground-image adapter, preserving Rubin's signed official PSF.

    The absolute stamp is used only for the shared adapter's validation and
    unit conversion. The original signed kernel is restored before fitting.
    """
    stamp = np.asarray(psf, dtype=float)
    if not np.all(np.isfinite(stamp)) or stamp.sum() <= 0:
        raise ValueError("Official PSF must be finite and have a positive sum")
    inputs = ground_make_image(science, uncertainty, np.abs(stamp), science_unit,
                               science_wcs, band=band)
    return replace(inputs, image=replace(inputs.image, psf=stamp / stamp.sum()))


def mask_pixels(mask, header, *, bad_planes=DEFAULT_BAD_PLANES):
    """Decode Rubin's dynamically assigned MP_* bits; never hard-code numbers."""
    missing = [name for name in bad_planes if f"MP_{name}" not in header]
    if missing:
        raise ValueError(f"Rubin mask header missing bit definitions: {missing}")
    bits = 0
    for name in bad_planes:
        bits |= 1 << int(header[f"MP_{name}"])
    return (np.asarray(mask, dtype=np.uint64) & bits) != 0


@dataclass(frozen=True)
class RubinCoadd:
    science: np.ndarray
    rms: np.ndarray
    mask: np.ndarray
    wcs: WCS
    unit: u.UnitBase
    band: str
    mask_fractions: dict[str, float] = field(default_factory=dict)
    excluded: np.ndarray | None = None
    bad_planes: tuple[str, ...] = DEFAULT_BAD_PLANES

    @property
    def valid(self):
        usable = np.isfinite(self.science) & np.isfinite(self.rms) & (self.rms > 0)
        if self.excluded is not None:
            usable &= ~self.excluded
        return usable

    def ground_inputs(self, official_psf):
        """Shared ground-image adapter once an official evaluated PSF is available.

        This method does not supply or estimate a PSF. The calling pipeline must
        verify that the stamp comes from Rubin's official model for this image
        and location, as done by fetch_official_psfs and the notebook.
        """
        fitting_rms = np.where(self.valid, self.rms, np.nan)
        return make_image(self.science, fitting_rms, official_psf, self.unit,
                           self.wcs, band=self.band)


def load_coadd(path, band, *, ra=None, dec=None, size_arcsec=None,
               bad_planes=DEFAULT_BAD_PLANES):
    """Read/crop science, RMS, masks and WCS together, without inventing a PSF."""
    if isinstance(bad_planes, str):
        raise TypeError("bad_planes must be a sequence of mask names")
    bad_planes = tuple(bad_planes)
    with fits.open(path, memmap=False) as hdus:
        science = np.asarray(hdus["SCIENCE"].data, float)
        variance = np.asarray(hdus["VARIANCE"].data, float)
        mask = np.array(hdus["MASK"].data, copy=True)
        bad = mask_pixels(mask, hdus["MASK"].header, bad_planes=bad_planes)
        if science.shape != variance.shape or science.shape != bad.shape:
            raise ValueError("Rubin science, variance, and mask shapes differ")
        wcs = WCS(hdus["SCIENCE"].header).celestial
        unit = u.Unit(hdus["SCIENCE"].header["BUNIT"])
        rms = np.full(science.shape, np.nan)
        # Preserve the archive uncertainty independently of the quality policy.
        # Exclusions enter only when constructing fitting weights.
        good = np.isfinite(variance) & (variance > 0)
        rms[good] = np.sqrt(variance[good])
        if size_arcsec is not None:
            center = SkyCoord(ra * u.deg, dec * u.deg)
            cut = Cutout2D(science, center, size_arcsec * u.arcsec, wcs=wcs, mode="strict")
            science, rms = cut.data, rms[cut.slices_original]
            mask, wcs = mask[cut.slices_original], cut.wcs
            bad = bad[cut.slices_original]
        # Keep every available flag in the diagnostics, including conditional
        # planes which are retained by the baseline fit mask.
        mask_header = hdus['MASK'].header
        fractions = {key[3:]: float(np.mean((mask & (1 << int(mask_header[key]))) != 0))
                     for key in mask_header if key.startswith('MP_')}
        return RubinCoadd(science, rms, mask, wcs, unit, band,
                          mask_fractions=fractions, excluded=bad, bad_planes=bad_planes)


@dataclass(frozen=True)
class RubinBand:
    band: str
    image: RubinCoadd
    catalog: Table
    image_path: Path
    tract: int
    patch: int
    services: tuple[str, ...]
    dataset_id: str = ""


def download_field(client, ra=59.10, dec=-48.73, size_arcsec=50.0, *,
                   bands=("u", "g"),
                   bad_planes=DEFAULT_BAD_PLANES):
    """Download one coadd patch and reference catalog per band, without fitting.

    Images use exactly the requested science box, with no added margin.
    Missing bands fail explicitly. Cross-patch mosaicking is not
    implemented. No PSF approximation or empirical PSF is constructed.
    """
    from .client import validate_band, validate_field
    validate_field(ra, dec, size_arcsec)
    if isinstance(bad_planes, str):
        raise TypeError("bad_planes must be a sequence of mask names")
    bad_planes = tuple(bad_planes)
    bands = tuple(bands)
    if not bands or len(set(bands)) != len(bands):
        raise ValueError("bands must be nonempty and unique")
    for band in bands:
        validate_band(band)
    downloaded = {}
    for band in bands:
        print(f"Rubin {client.config.collection}: downloading {band}", flush=True)
        rows = client.coadds(ra, dec, size_arcsec, band)
        if not len(rows):
            raise RuntimeError(f"No {band} coadd overlaps the requested field")
        centers = SkyCoord(np.asarray(rows["s_ra"], float) * u.deg,
                           np.asarray(rows["s_dec"], float) * u.deg)
        order = np.argsort(centers.separation(SkyCoord(ra * u.deg, dec * u.deg)).arcsec)
        for index in order:
            row = rows[index]
            tract, patch = int(row["lsst_tract"]), int(row["lsst_patch"])
            path = client.cutout(row, ra, dec, size_arcsec)
            try:
                image = load_coadd(path, band, ra=ra, dec=dec, size_arcsec=size_arcsec,
                                    bad_planes=bad_planes)
            except (PartialOverlapError, NoOverlapError):
                continue
            if not np.any(image.valid):
                continue
            if np.mean(image.valid) < 0.5:
                warnings.warn(
                    f"{band}: only {np.mean(image.valid):.1%} of pixels pass the selected mask. "
                    f"Coverage exists; inspect the mask before fitting. Flag fractions: {image.mask_fractions}",
                    stacklevel=2)
            catalog = client.objects(ra, dec, size_arcsec, band, tract=tract, patch=patch)
            catalog_path = path.with_name(path.stem + f"_{band}_objects.ecsv")
            catalog.write(catalog_path, overwrite=True)
            services = tuple(client.cutout_services(row))
            identity = str(row['obs_publisher_did']) if 'obs_publisher_did' in row.colnames else ''
            downloaded[band] = RubinBand(band, image, catalog, path, tract, patch, services, identity)
            print(f"  patch {tract}/{patch}; {image.science.shape}; "
                  f"{np.mean(image.valid):.1%} usable pixels; {len(catalog)} reference objects")
            break
        if band not in downloaded:
            raise RuntimeError(f"No single {band} patch covers the requested science box")
    return downloaded
