"""The Euclid inputs for a ground image: MER sources, VIS cutout and models.

:func:`prepare_euclid` reuses the cached MER catalog, VIS cutout and GRID
PSF around the ground image (downloading what is missing), and
:func:`fit_vis` fits the VIS source models with the standard model-selection
tree, exactly as for every other band.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.table import Table

from ..euclid.catalog import query_mer_catalog
from ..euclid.cutouts import fetch_cutout
from ..euclid.psf import extract_grid_psf
from ..pipeline import run_forced_photometry
from .image import SmallImage


@dataclass(frozen=True)
class EuclidInputs:
    mer: Table
    vis_cutout: object
    ra: float
    dec: float
    size_arcsec: float
    mer_path: Path


def prepare_euclid(image: SmallImage, data_dir: str | Path,
                   *, padding_arcsec: float = 5.0) -> EuclidInputs:
    """Reuse local MER/VIS/GRID-PSF inputs and fetch only missing files."""
    data_dir = Path(data_dir)
    ny, nx = image.science_ujy.shape
    center = image.wcs.pixel_to_world((nx - 1) / 2, (ny - 1) / 2)
    ra, dec = round(center.ra.deg, 4), round(center.dec.deg, 4)
    if padding_arcsec < 0:
        raise ValueError("padding_arcsec must be nonnegative")
    edge_x = np.linspace(-.5, nx-.5, 5)
    edge_y = np.linspace(-.5, ny-.5, 5)
    xx = np.r_[edge_x, edge_x, np.full(5, -.5), np.full(5, nx-.5)]
    yy = np.r_[np.full(5, -.5), np.full(5, ny-.5), edge_y, edge_y]
    edge_world = image.wcs.pixel_to_world(xx, yy)
    east, north = center.spherical_offsets_to(edge_world)
    raw_side = 2 * (max(np.max(np.abs(east.arcsec)),
                        np.max(np.abs(north.arcsec))) + padding_arcsec)
    # A square aligned with the ground image is insufficient when its WCS is
    # rotated. Use the bounding sky square, including a small source margin.
    size = float(10.0 * np.ceil(raw_side / 10))
    # Same cache name as run_forced_photometry, so either can reuse it.
    mer_path = data_dir / f"mer_catalog_{ra:.4f}_{dec:.4f}_{int(round(size))}.fits"
    if mer_path.is_file():
        mer = Table.read(mer_path)
        print("MER: local cache", mer_path.name)
    else:
        print("MER: querying Euclid for the image footprint", flush=True)
        mer = query_mer_catalog(ra, dec, size / 7200)
        if not len(mer):
            raise ValueError("No MER sources found in this image footprint")
        mer_path.parent.mkdir(parents=True, exist_ok=True)
        mer.write(mer_path, overwrite=True)
        print("MER: saved", mer_path.name)
    # Both functions read their local cache first and fetch missing data.
    vis = fetch_cutout("VIS", ra, dec, size, data_dir=data_dir / "cutouts",
                       with_flag=True)
    print("VIS science/RMS/flag: ready")
    psf = extract_grid_psf(
        "VIS", ra, dec, tile_id=vis.header.get("MERTILE"),
        radius_arcsec=max(60.0, 0.75 * size), data_dir=data_dir / "psf")
    print("VIS GRID PSF: ready")
    if not len(psf["stamps"]):
        raise ValueError("No VIS GRID-PSF stamps cover this field")
    return EuclidInputs(mer, vis, ra, dec, size, mer_path)


def fit_vis(inputs: EuclidInputs, data_dir: str | Path, *, n_workers: int = 8,
            existing=None):
    """Fit the same VIS prior as notebooks 01-03; optionally reuse a live result.

    An in-memory result from another notebook may be passed as ``existing``.
    Cached image/PSF/catalog files are reused automatically; the standard
    pipeline does not currently persist its fitted Tractor source list.
    """
    if existing is not None:
        if len(existing.sources) != len(inputs.mer):
            raise ValueError("existing VIS result has a different MER source count")
        if not np.array_equal(np.asarray(existing.mer_cat["object_id"]),
                              np.asarray(inputs.mer["object_id"])):
            raise ValueError("existing VIS result has different MER source IDs")
        print("VIS morphology: reusing supplied in-memory result")
        return existing
    return run_forced_photometry(
        inputs.ra, inputs.dec, inputs.size_arcsec,
        prior={"band": "VIS", "objects": "mer", "model_selection": "tree"},
        target_bands={"euclid": (), "wise": ()},
        psf_product="grid", persource_psf=True, mask_bright_stars=True,
        calibrate_errors=False, data_dir=Path(data_dir), n_workers=n_workers,
        mer_catalog=inputs.mer,
    )
