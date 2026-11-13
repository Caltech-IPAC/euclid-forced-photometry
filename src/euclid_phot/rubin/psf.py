"""Download exact evaluated official PSFs using a small RSP-side calculation."""
from __future__ import annotations

import base64
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .remote import RSPExecutor


@dataclass(frozen=True)
class OfficialPSF:
    stamps: np.ndarray
    positions: np.ndarray
    xy: np.ndarray
    dataset_id: str
    path: Path

    @property
    def center(self):
        return self.stamps[0]


def fetch_official_psfs(client, token, downloaded, positions, *, lab_size="Small"):
    """Evaluate official PSF kernels at sky positions; reuse local cached stamps.

    The first position in each band is the fitting-field center. Signed kernel
    values are preserved. Dataset UUIDs must match the SIA-selected coadds.
    Existing RSP labs are left running; a lab started by this helper is stopped.
    """
    if client.config.collection != "dp1":
        raise ValueError("The official remote PSF adapter currently supports Rubin DP1")
    requests, paths, cached = [], {}, {}
    for band, image in downloaded.items():
        if not image.dataset_id:
            raise ValueError("Image provenance is missing; rerun the download cell")
        sky = np.asarray(positions[band], dtype=float)
        if sky.ndim != 2 or sky.shape[1] != 2 or len(sky) == 0 or not np.all(np.isfinite(sky)):
            raise ValueError("PSF positions must be a finite (N,2) RA/Dec array")
        request = {"band": band, "tract": image.tract, "patch": image.patch,
                   "dataset_id": image.dataset_id, "positions": sky.tolist()}
        path = client._path("psf", "official-kernel-v1|" + json.dumps(request, sort_keys=True), ".npz")
        paths[band] = path
        if not path.exists():
            requests.append(request)
        cached[band] = request
    if requests:
        source = Path(__file__).with_name("_evaluate_psf.py").read_text()
        code = "payload = " + repr(requests) + "\n" + source
        print(f"Evaluating official PSFs for {len(requests)} bands through the RSP API", flush=True)
        with RSPExecutor(token, site=client.config.base_url, size=lab_size) as remote:
            output = remote.run_code(code, timeout=600)
        line = next((s for s in output.splitlines() if s.startswith("RUBIN_PSF_RESULT=")), None)
        if line is None:
            raise RuntimeError("RSP returned no evaluated PSF data")
        for item in json.loads(line.split("=", 1)[1]):
            path = paths[item["band"]]
            raw = base64.b64decode(item["npz"], validate=True)
            temporary = path.with_suffix(".npz.part")
            temporary.write_bytes(raw)
            temporary.replace(path)
    result = {}
    for band, request in cached.items():
        path = paths[band]
        with np.load(path, allow_pickle=False) as arrays:
            stamps = arrays["stamps"].copy()
            sky = arrays["positions"].copy()
            xy = arrays["xy"].copy()
            origins = arrays["origins"]
        if (stamps.ndim != 3 or not np.all(np.isfinite(stamps)) or
                np.any(stamps.sum(axis=(1, 2)) <= 0) or
                any(size % 2 == 0 for size in stamps.shape[1:])):
            raise ValueError("Official PSF kernels are invalid or not odd-sized")
        expected_origin = -(np.array(stamps.shape[1:][::-1]) // 2)
        if not np.all(origins == expected_origin):
            raise ValueError("Official PSF kernel origin does not match the stamp center")
        result[band] = OfficialPSF(stamps, sky, xy, request["dataset_id"], path)
        print(f"{band}: official PSF {stamps.shape}", flush=True)
    return result
