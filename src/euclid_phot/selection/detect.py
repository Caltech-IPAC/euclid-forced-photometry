"""SEP blob detection, MER cross-matching and unlisted neighbours."""

from __future__ import annotations

import logging

import numpy as np
from tractor import RaDecPos

from .models import Blob, EUCLID_DEFAULTS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SEP-based blob detection and MER cross-matching.
# ---------------------------------------------------------------------------
# SEP's shipped default kernel (3x3 Gaussian, FWHM ~2 pix, byte-identical
# to SExtractor's default.conv); filter_kernel=None means no filtering.
SEP_DEFAULT_KERNEL = np.array([[1.0, 2.0, 1.0],
                               [2.0, 4.0, 2.0],
                               [1.0, 2.0, 1.0]])


def detect_blobs(
    vis_data: np.ndarray,
    vis_invvar: np.ndarray,
    *,
    threshold_sigma: float = EUCLID_DEFAULTS["sep_threshold_sigma"],
    minarea: int = EUCLID_DEFAULTS["sep_minarea"],
    dilate_pix: int = EUCLID_DEFAULTS["blob_dilate_pix"],
    deblend_cont: float = 0.005,
    deblend_nthresh: int = 32,
    bkg_box: int = 64,
    filter_kernel: np.ndarray | None = SEP_DEFAULT_KERNEL,
    filter_type: str = "matched",
    clean: bool = True,
) -> tuple[list[Blob], np.ndarray, np.ndarray]:
    """Run SEP on the VIS cutout and return (blobs, segmap, sep_objects).

    ``segmap`` is the raw SEP segmentation (per-source label image); ``blobs``
    contains the dilated/merged groups. ``sep_objects`` is the structured array
    SEP returns (a, b, theta, x, y, flux, ...).

    Parameters
    ----------
    threshold_sigma : float
        Detection threshold in units of the per-pixel noise (SExtractor
        ``DETECT_THRESH``). Raise it for a cleaner, higher-purity source list.
    minarea : int
        Minimum number of connected pixels above threshold (``DETECT_MINAREA``).
    dilate_pix : int
        Radius (pixels) by which SEP segments are grown before grouping into
        blobs; controls how readily neighboring sources share a joint fit.
    deblend_cont : float
        SEP deblend contrast (``DEBLEND_MINCONT``). Smaller splits blends more
        aggressively; 0.005 is the SExtractor default.
    deblend_nthresh : int
        Number of deblending sub-thresholds (``DEBLEND_NTHRESH``).
    bkg_box : int
        Background-mesh box size in pixels for ``sep.Background``.
    filter_kernel : ndarray or None
        Detection convolution kernel. Defaults to SEP's shipped 3x3
        FWHM~2 pix kernel (a good match to the ~1.6 pix VIS PSF FWHM);
        ``None`` disables filtering. The Farmer's own config uses a
        5x5 FWHM=2 Gaussian; COSMOS2020 detection used 7x7 FWHM=4 on
        seeing-limited ground-based stacks.
    filter_type : {'matched', 'conv'}
        ``'matched'`` (SEP default) weights the kernel by the local noise,
        which matters on coadds with spatially varying depth.
    clean : bool
        SExtractor-style cleaning of spurious detections near bright
        sources (SEP default True; the Farmer ships clean=False).

    Notes
    -----
    The detection threshold (1.5 sigma) and ``minarea`` (5) match Farmer's
    config. The deblend and grouping defaults are the SExtractor/SEP
    defaults plus a 4-pixel dilation, gentler than Farmer's shipped values
    (``DEBLEND_CONT=1e-10``, ``DEBLEND_NTHRESH=256``, 0.2" dilation), which
    over-split Euclid VIS blends.
    """
    import sep
    from scipy import ndimage

    # SEP wants a contiguous float32 array.
    data = np.ascontiguousarray(vis_data.astype(np.float32))
    err = np.zeros_like(data)
    # Q1 stores masked/no-coverage pixels with a huge RMS (~1e16), i.e. a
    # tiny but positive inverse-variance; a bare invvar > 0 test would pass
    # them. Floor the test relative to the median real weight.
    finite_iv = vis_invvar[np.isfinite(vis_invvar) & (vis_invvar > 0)]
    iv_floor = float(np.median(finite_iv)) * 1e-6 if finite_iv.size else 0.0
    good = np.isfinite(vis_invvar) & (vis_invvar > iv_floor)
    err[good] = 1.0 / np.sqrt(vis_invvar[good])
    err[~good] = err[good].max() if good.any() else 1.0
    mask = ~good

    try:
        bkg = sep.Background(data, mask=mask, bw=int(bkg_box), bh=int(bkg_box))
        data_sub = data - bkg.back()
    except Exception as exc:
        logger.warning("SEP background failed (%s); using raw data.", exc)
        data_sub = data

    # SEP's pixel buffer defaults to 300k active pixels; extract() aborts
    # with a pixstack overflow on a deep multi-arcmin field. Size it to the
    # image.
    sep.set_extract_pixstack(max(300_000, min(int(data.size // 5), 5_000_000)))

    objects, segmap = sep.extract(
        data_sub,
        thresh=threshold_sigma,
        err=err,
        mask=mask,
        minarea=int(minarea),
        filter_kernel=filter_kernel,
        filter_type=filter_type,
        deblend_nthresh=int(deblend_nthresh),
        deblend_cont=deblend_cont,
        clean=clean,
        segmentation_map=True,
    )

    if dilate_pix > 0:
        struct = np.ones((2 * dilate_pix + 1, 2 * dilate_pix + 1), dtype=bool)
        dilated = ndimage.binary_dilation(segmap > 0, structure=struct)
    else:
        dilated = segmap > 0

    blob_labels, n_blobs = ndimage.label(dilated)

    blobs: list[Blob] = []
    for blob_id in range(1, n_blobs + 1):
        footprint = blob_labels == blob_id
        members = np.unique(segmap[footprint])
        members = members[members > 0]
        segments = {}
        sep_rows = {}
        for sep_label in members:
            seg = segmap == sep_label
            segments[int(sep_label) - 1] = seg  # 0-based index into ``objects``
            sep_rows[int(sep_label) - 1] = {k: objects[int(sep_label) - 1][k]
                                            for k in objects.dtype.names}
        blobs.append(Blob(
            blob_id=blob_id,
            footprint=footprint,
            member_mer_indices=list(segments.keys()),  # placeholder; filled by assign
            segments=segments,
            sep_rows=sep_rows,
        ))
    return blobs, segmap, objects


def assign_mer_to_blobs(mer_cat, blobs: list[Blob], wcs) -> list[Blob]:
    """Assign each MER row to the blob whose footprint contains it.

    Returns a new list of Blob objects with ``member_mer_indices`` (and
    ``segments`` keyed by MER idx) populated. Sources not falling inside any
    blob are dropped; sources inside a blob but outside any SEP segment are
    attached to the geometrically nearest SEP segment in that blob.
    """
    projected = []
    if blobs:
        for mer_idx, row in enumerate(mer_cat):
            try:
                px, py = wcs.positionToPixel(RaDecPos(float(row["ra"]), float(row["dec"])))
            except Exception:
                continue
            projected.append((mer_idx, px, py))
    assigned: list[Blob] = []
    for blob in blobs:
        mer_indices = []
        new_segments: dict[int, np.ndarray] = {}
        new_sep_rows: dict[int, dict] = {}
        sep_labels = list(blob.segments.keys())  # original 0-based SEP indices

        for mer_idx, px, py in projected:
            xi, yi = int(round(px)), int(round(py))
            H, W = blob.footprint.shape
            if not (0 <= yi < H and 0 <= xi < W):
                continue
            if not blob.footprint[yi, xi]:
                continue
            owning_label = None
            for sl in sep_labels:
                if blob.segments[sl][yi, xi]:
                    owning_label = sl
                    break
            if owning_label is None and sep_labels:
                # Nearest SEP centroid in this blob.
                d_best = np.inf
                for sl in sep_labels:
                    seg_yx = np.argwhere(blob.segments[sl])
                    if not len(seg_yx):
                        continue
                    cy, cx = seg_yx.mean(axis=0)
                    d = (cy - yi) ** 2 + (cx - xi) ** 2
                    if d < d_best:
                        d_best = d
                        owning_label = sl
            if owning_label is None:
                continue
            mer_indices.append(mer_idx)
            new_segments[mer_idx] = blob.segments[owning_label]
            new_sep_rows[mer_idx] = blob.sep_rows[owning_label]

        if not mer_indices:
            continue
        assigned.append(Blob(
            blob_id=blob.blob_id,
            footprint=blob.footprint,
            member_mer_indices=mer_indices,
            segments=new_segments,
            sep_rows=new_sep_rows,
        ))
    return assigned


def unlisted_sep_detections(mer_cat, tim_vis, vis_data, vis_invvar, *,
                            min_separation_pix: float = 3.0,
                            min_snr: float = 5.0,
                            mask_margin_pix: int = 3,
                            **detect_kwargs) -> dict:
    """SEP detections that share a blob with a catalog row but match none.

    Runs the same detection and assignment as :func:`run_model_selection`.
    Inside every blob that contains at least one ``mer_cat`` row, the SEP
    segments no row was attached to are returned as candidate neighbours;
    blobs without a catalog row are ignored. A candidate is skipped when it
    lies closer than ``min_separation_pix`` to a catalog row, when its
    segment flux is below ``min_snr`` times the pixel noise summed in
    quadrature over the segment, or when the segment comes within
    ``mask_margin_pix`` of a zero-weight pixel (bright-star mask, flagged
    or missing data), where detections are treated as the masked object's
    light.

    Returns a dict of arrays: ``ra``, ``dec`` (deg), ``flux`` (SEP
    isophotal flux in image counts), ``snr``, ``x``, ``y`` (pixels).
    """
    from scipy import ndimage

    blobs_raw, _, _ = detect_blobs(vis_data, vis_invvar, **detect_kwargs)
    wcs = tim_vis.getWcs()
    finite_iv = vis_invvar[np.isfinite(vis_invvar) & (vis_invvar > 0)]
    iv_floor = float(np.median(finite_iv)) * 1e-6 if finite_iv.size else 0.0
    good = np.isfinite(vis_invvar) & (vis_invvar > iv_floor)
    near_mask = ~good
    if mask_margin_pix > 0 and near_mask.any():
        near_mask = ndimage.binary_dilation(
            near_mask, iterations=int(mask_margin_pix))
    assigned = {b.blob_id: b for b in assign_mer_to_blobs(mer_cat, blobs_raw, wcs)}
    listed = []
    for row in mer_cat:
        try:
            listed.append(wcs.positionToPixel(
                RaDecPos(float(row["ra"]), float(row["dec"]))))
        except Exception:
            continue
    listed = np.asarray(listed, dtype=float).reshape(-1, 2)

    out = {k: [] for k in ("ra", "dec", "flux", "snr", "x", "y")}
    for raw in blobs_raw:
        blob = assigned.get(raw.blob_id)
        if blob is None:
            continue
        # assign_mer_to_blobs re-keys the segment arrays without copying
        # them, so identity tells which SEP segments received a row.
        taken = {id(seg) for seg in blob.segments.values()}
        for label, seg in raw.segments.items():
            if id(seg) in taken:
                continue
            sep_row = raw.sep_rows[label]
            x, y = float(sep_row["x"]), float(sep_row["y"])
            if listed.size and np.hypot(listed[:, 0] - x,
                                        listed[:, 1] - y).min() < min_separation_pix:
                continue
            if np.any(seg & near_mask):
                continue
            pix = seg & good
            noise = float(np.sqrt(np.sum(1.0 / vis_invvar[pix]))) if pix.any() else np.inf
            snr = float(sep_row["flux"]) / noise if noise > 0 else 0.0
            if not snr >= min_snr:
                continue
            pos = wcs.pixelToPosition(x, y)
            out["ra"].append(float(pos.ra))
            out["dec"].append(float(pos.dec))
            out["flux"].append(float(sep_row["flux"]))
            out["snr"].append(snr)
            out["x"].append(x)
            out["y"].append(y)
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}
