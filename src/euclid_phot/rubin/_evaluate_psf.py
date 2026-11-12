"""Executed only in the official RSP LSST kernel; payload is supplied by psf.py."""
import base64
import io
import json
import re

import numpy as np
from lsst.daf.butler import Butler
import lsst.geom

payload: list[dict]  # Injected by psf.py in the remote kernel, never a token.

butler = Butler("dp1", collections="LSSTComCam/DP1", skymap="lsst_cells_v1")
products = []
for request in payload:
    ref = butler.find_dataset("deep_coadd", tract=request["tract"],
                              patch=request["patch"], band=request["band"])
    if ref is None:
        raise RuntimeError("No matching official DP1 coadd")
    uuids = re.findall(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", request["dataset_id"])
    if not uuids or str(ref.id).lower() != uuids[-1].lower():
        raise RuntimeError("RSP coadd dataset ID does not match the downloaded API image")
    # Components preserve the exact official model while avoiding a full image read.
    psf = butler.get(ref.makeComponentRef("psf"))
    wcs = butler.get(ref.makeComponentRef("wcs"))
    stamps, xy, origins = [], [], []
    for ra, dec in request["positions"]:
        point = wcs.skyToPixel(lsst.geom.SpherePoint(ra * lsst.geom.degrees,
                                                    dec * lsst.geom.degrees))
        image = psf.computeKernelImage(point)
        stamps.append(np.array(image.array, dtype=np.float64))
        xy.append([point.x, point.y])
        origins.append([image.getX0(), image.getY0()])
    out = io.BytesIO()
    np.savez_compressed(out, stamps=np.stack(stamps), xy=np.array(xy),
                        positions=np.array(request["positions"]), origins=np.array(origins))
    products.append({"band": request["band"], "dataset_id": str(ref.id),
                      "npz": base64.b64encode(out.getvalue()).decode("ascii")})
print("RUBIN_PSF_RESULT=" + json.dumps(products))
