"""Run the tree over every blob (threads or processes) and return the sources."""

from __future__ import annotations

import logging

import numpy as np
from tractor import Image

from .detect import assign_mer_to_blobs, detect_blobs
from .models import _classname, _make_masked_image, Blob, MODEL_NAMES
from .tree import ModelSelector

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Convenience: run the tree over every blob and return a list[Source].
# ---------------------------------------------------------------------------
def _fit_one_blob(args):
    """Worker for parallel Farmer fits: one blob, runnable in a thread."""
    selector, blob, tim_vis, mer_cat, pixscale, psf_by_member = args
    chosen, _ = selector.fit_blob(blob, tim_vis, mer_cat, pixscale,
                                  history=False, psf_by_member=psf_by_member)
    return blob, chosen


def _pack_blob(blob: Blob) -> dict:
    """Bit-pack a Blob's full-frame boolean masks to shrink what is sent
    to worker processes.

    A 2000x2000 footprint is 4 MB as bool and 0.5 MB packed; a field of a
    few hundred blobs would otherwise copy gigabytes to each worker process.
    """
    pack = lambda m: (np.packbits(m), m.shape)  # noqa: E731
    return {
        "blob_id": blob.blob_id,
        "footprint": pack(blob.footprint),
        "member_mer_indices": list(blob.member_mer_indices),
        "segments": {k: pack(v) for k, v in blob.segments.items()},
        "sep_rows": blob.sep_rows,
    }


def _unpack_blob(d: dict) -> Blob:
    def unpack(p):
        bits, shape = p
        n = int(shape[0]) * int(shape[1])
        return np.unpackbits(bits, count=n).astype(bool).reshape(shape)
    return Blob(
        blob_id=d["blob_id"],
        footprint=unpack(d["footprint"]),
        member_mer_indices=d["member_mer_indices"],
        segments={k: unpack(v) for k, v in d["segments"].items()},
        sep_rows=d["sep_rows"],
    )


def _fit_blob_chunk(args):
    """Worker for process-parallel Farmer fits: a chunk of packed blobs."""
    selector, packed, tim_vis, mer_cat, pixscale, psf_by_member = args
    out = []
    for pb in packed:
        blob = _unpack_blob(pb)
        chosen, _ = selector.fit_blob(blob, tim_vis, mer_cat, pixscale,
                                      history=False,
                                      psf_by_member=psf_by_member)
        out.append(chosen)
    return out


def run_model_selection(
    mer_cat,
    tim_vis: Image,
    vis_data: np.ndarray,
    vis_invvar: np.ndarray,
    *,
    pixscale_arcsec: float,
    selector: ModelSelector | None = None,
    psf_data: dict | None = None,
    n_workers: int = 1,
    **detect_kwargs,
):
    """Detect blobs and run :meth:`ModelSelector.fit_blob` over each.

    Parameters
    ----------
    n_workers : int, default 1
        If >1, fit blobs in a process pool (blob fits are independent, and
        separate processes scale better here than threads, since the
        optimizer runs single-threaded internally). Falls back to threads
        if a blob cannot be sent to a separate process for an unusual
        source class.

    Returns ``(sources_list, summary_dict)`` where ``sources_list`` is aligned
    1:1 with ``mer_cat`` (sources outside any blob get ``None``) and
    ``summary_dict`` reports how many MER sources landed at each model tier.
    """
    selector = selector or ModelSelector()
    blobs_raw, segmap, _ = detect_blobs(vis_data, vis_invvar, **detect_kwargs)
    blobs = assign_mer_to_blobs(mer_cat, blobs_raw, tim_vis.getWcs())

    psf_by_member = None
    if (psf_data is not None and selector.per_source_psf
            and not getattr(tim_vis.getPsf(), 'is_spatial', False)
            and psf_data.get("stamps") is not None
            and len(psf_data["stamps"]) > 0):
        from .pixel_psf import SpatialPixelizedPSF
        tim_vis = _make_masked_image(
            tim_vis, np.ones(tim_vis.shape, dtype=bool),
            psf=SpatialPixelizedPSF(psf_data, tim_vis.getWcs()))

    sources_list: list[object | None] = [None] * len(mer_cat)
    counts = {name: 0 for name in MODEL_NAMES}
    counts["Unblobbed"] = 0

    if n_workers <= 1 or len(blobs) <= 1:
        work = [(selector, b, tim_vis, mer_cat, pixscale_arcsec,
                 psf_by_member) for b in blobs]
        chosen_list = [c for _, c in (_fit_one_blob(w) for w in work)]
    else:
        import concurrent.futures
        n = min(n_workers, len(blobs))
        # Interleave so the brightest (usually first) blobs spread across
        # workers; each chunk pickles tim_vis once.
        chunks = [(selector, [_pack_blob(b) for b in blobs[i::n]],
                   tim_vis, mer_cat, pixscale_arcsec, psf_by_member)
                  for i in range(n)]
        try:
            with concurrent.futures.ProcessPoolExecutor(n) as exec_:
                chosen_list = [c for sub in exec_.map(_fit_blob_chunk, chunks)
                               for c in sub]
        except Exception as exc:
            logger.warning(
                "process-parallel blob fits failed (%r); retrying with "
                "threads.", exc)
            work = [(selector, b, tim_vis, mer_cat, pixscale_arcsec,
                     psf_by_member) for b in blobs]
            with concurrent.futures.ThreadPoolExecutor(n) as exec_:
                chosen_list = [c for _, c in exec_.map(_fit_one_blob, work)]

    for chosen in chosen_list:
        for mer_idx, src in chosen.items():
            sources_list[mer_idx] = src
            counts[_classname(src)] += 1

    counts["Unblobbed"] = sum(1 for s in sources_list if s is None)
    return sources_list, counts
