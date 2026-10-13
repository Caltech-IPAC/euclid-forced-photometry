"""unWISE coadd cutouts and their bitmask, from unwise.me."""
from __future__ import annotations

import sys
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from ..netutils import retry
from .config import DEFAULT_WISE_CACHE_DIR, UNWISE_PIXEL_SCALE, WISE_COADD_VERSION


def _unwise_cutout_url(ra: float, dec: float, size_pix: int,
                       version: str) -> str:
    """Build the unwise.me/cutout_fits request URL.

    Must be https: unwise.me answers plain http with a 308, which urllib
    does not follow on Python < 3.11. ``bands=12`` requests W1 + W2.
    """
    # cutout_fits serves only img/invvar/n/std; file_msk=on is silently
    # ignored. The bitmask is fetched per tile by _fetch_unwise_mask_tile.
    return (
        f"https://unwise.me/cutout_fits?version={version}"
        f"&ra={ra}&dec={dec}&size={size_pix}"
        f"&bands=12&file_img_m=on&file_invvar_m=on"
    )


def _unwise_wcs(header) -> WCS:
    """Read celestial WCS without treating unWISE's epoch index as an equinox.

    In unWISE headers ``EPOCH`` identifies the time-slice product (``-1``
    for a full-depth coadd). The FITS WCS reader otherwise treats this
    keyword as the obsolete celestial equinox keyword and can infer FK4.
    Preserve the delivered header and any explicit celestial frame cards.
    """
    celestial_header = header.copy()
    celestial_header.pop("EPOCH", None)
    return WCS(celestial_header)


def _fetch_unwise_mask_tile(coadd_id: str, version: str,
                            ref_wcs: WCS, ref_shape, cache_dir: Path):
    """Fetch the unWISE 31-bit bitmask for ``coadd_id`` and resample it onto the
    cutout grid (``ref_wcs`` / ``ref_shape``).

    Pulls the per-tile ``-msk.fits.gz`` from the coadd archive (~90 kB
    gzipped, a static product shared across NEO epochs; Meisner et al.
    2019, PASP 131, 124504) and reprojects it nearest-neighbor. Returns an
    int32 array shaped like the
    cutout, or None if the tile is unavailable (masking is then skipped).
    """
    from reproject import reproject_interp

    tile3 = coadd_id[:3]
    url = (f"https://unwise.me/data/{version}/unwise-coadds/fulldepth/"
           f"{tile3}/{coadd_id}/unwise-{coadd_id}-msk.fits.gz")
    local = Path(cache_dir) / f"unwise-{coadd_id}-msk.fits.gz"
    try:
        if not local.exists():
            tmp = local.with_suffix(local.suffix + ".part")
            retry(lambda: urllib.request.urlretrieve(url, tmp),
                  what="unwise.me mask download")
            with tmp.open("rb") as f:
                if f.read(2) != b"\x1f\x8b":   # not gzip -> an HTML error page
                    tmp.unlink(missing_ok=True)
                    return None
            tmp.replace(local)
        with fits.open(local) as hdul:
            mdata = hdul[0].data.astype(np.float64)
            mwcs = _unwise_wcs(hdul[0].header)
        arr, footprint = reproject_interp(
            (mdata, mwcs), ref_wcs, shape_out=tuple(ref_shape),
            order="nearest-neighbor")
        out = np.where((footprint > 0) & np.isfinite(arr),
                       np.rint(arr), 0).astype(np.int32)
        return out
    except Exception:
        return None


def _closest_coadd_id(member_names, ra: float, dec: float) -> str:
    """Pick the unWISE coadd tile closest to (ra, dec) from tarball members.

    A cutout_fits request near a tile boundary returns files from every
    overlapping tile; the fit runs on one tile's pixel grid, so the tile
    whose center is nearest the target is used (the name encodes the
    center: '2709p666' = RA 270.9, Dec +66.6). Pixels beyond that tile's
    edge enter the fit with zero weight.
    """
    ids: list[str] = []
    for n in member_names:
        cid = n.split("/")[-1].split("-")[1]
        if cid not in ids:
            ids.append(cid)
    if len(ids) == 1:
        return ids[0]

    cosd = float(np.cos(np.radians(dec)))

    def _dist2(cid):
        ra_c = int(cid[:4]) / 10.0
        dec_c = int(cid[5:8]) / 10.0 * (1.0 if cid[4] == "p" else -1.0)
        d_ra = ((ra_c - ra + 540.0) % 360.0) - 180.0
        return (d_ra * cosd) ** 2 + (dec_c - dec) ** 2

    ids.sort(key=_dist2)
    import warnings
    warnings.warn(
        f"unWISE cutout spans {len(ids)} coadd tiles ({', '.join(ids)}); "
        f"fitting on the closest, {ids[0]}. Sources beyond that tile's edge "
        "fall on zero-weight pixels and come back with NaN errors; recenter "
        "the target or shrink the cutout to keep the field on one tile.",
        stacklevel=3)
    return ids[0]


def fetch_unwise_cutouts(ra: float, dec: float, size_arcsec: float,
                         *,
                         data_dir: str | Path = DEFAULT_WISE_CACHE_DIR,
                         version: str = WISE_COADD_VERSION,
                         bands=("W1", "W2"),
                         force_download: bool = False) -> dict:
    """Download (or cache) unWISE coadds for a small field.

    Returns
    -------
    dict[band] -> dict with keys ``data``, ``ivar``, ``wcs``, ``header``,
    plus ``coadd_id`` and ``size_arcsec`` at the top level.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    size_pix = int(round(size_arcsec / UNWISE_PIXEL_SCALE))

    tarball = data_dir / f"unwise_{version}_{ra:.4f}_{dec:.4f}_{int(round(size_arcsec))}.tar.gz"
    legacy = data_dir / f"unwise_{ra:.4f}_{dec:.4f}_{int(round(size_arcsec))}.tar.gz"
    if (not tarball.exists() and legacy.exists()
            and version == WISE_COADD_VERSION and not force_download):
        tarball = legacy
    if not tarball.exists() or force_download:
        url = _unwise_cutout_url(ra, dec, size_pix, version)
        # unwise.me can return an HTML 200 on out-of-coverage targets;
        # download to .part and validate the gzip magic.
        tmp = tarball.with_suffix(tarball.suffix + ".part")
        retry(lambda: urllib.request.urlretrieve(url, tmp),
              what="unwise.me cutout download")
        with tmp.open("rb") as f:
            if f.read(2) != b"\x1f\x8b":
                tmp.unlink(missing_ok=True)
                raise RuntimeError(
                    f"unwise.me did not return a tarball for ra={ra}, dec={dec}, "
                    f"size={size_pix}px (likely outside coverage)")
        tmp.replace(tarball)

    extract_dir = data_dir / tarball.stem
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball) as tar:
        names = tar.getnames()
        # tarfile filter="data" requires Python 3.12.
        if sys.version_info >= (3, 12):
            tar.extractall(extract_dir, filter="data")
        else:
            tar.extractall(extract_dir)
    fits_members = [n for n in names if n.endswith((".fits", ".fits.gz"))]
    if not fits_members:
        raise RuntimeError(f"no FITS members in unwise tarball {tarball}")
    coadd_id = _closest_coadd_id(fits_members, ra, dec)

    # If the tarball already carries an aligned bitmask, use it; otherwise
    # the per-tile masks are fetched below.
    msk = None
    msk_members = [n for n in names
                   if "-msk" in n.split("/")[-1]
                   and n.endswith((".fits", ".fits.gz"))]
    if msk_members:
        with fits.open(extract_dir / msk_members[0].split("/")[-1]) as hdul:
            msk = np.nan_to_num(hdul[0].data, nan=0).astype(np.int32)

    out: dict = {"coadd_id": coadd_id, "size_arcsec": size_arcsec}
    ref_wcs = ref_shape = None
    for band_num, band in [(1, "W1"), (2, "W2")]:
        if band not in bands:
            continue
        img_path = extract_dir / f"unwise-{coadd_id}-w{band_num}-img-m.fits"
        iv_path = extract_dir / f"unwise-{coadd_id}-w{band_num}-invvar-m.fits.gz"
        with fits.open(img_path) as hdul:
            data = hdul[0].data.astype(np.float32)
            header = hdul[0].header.copy()
            wcs = _unwise_wcs(hdul[0].header)
        with fits.open(iv_path) as hdul:
            ivar = hdul[0].data.astype(np.float32)
        if ref_wcs is None:
            ref_wcs, ref_shape = wcs, data.shape
        out[band] = {"data": data, "ivar": ivar, "wcs": wcs, "header": header}

    # No bitmask in the tarball: fetch the per-tile mask; a miss leaves
    # mask=None and masking is skipped.
    if msk is None and ref_wcs is not None:
        msk = _fetch_unwise_mask_tile(coadd_id, version, ref_wcs, ref_shape, data_dir)
    for band in list(out):
        if band in ("W1", "W2"):
            out[band]["mask"] = msk
    return out
