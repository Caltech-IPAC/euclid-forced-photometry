"""Register a user image to MER star positions before forced photometry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sep
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.wcs import WCS
from astropy.wcs.utils import fit_wcs_from_points, proj_plane_pixel_scales
from scipy.spatial import cKDTree

from .image import SmallImage


@dataclass(frozen=True)
class WCSRegistration:
    wcs: WCS
    matches: Table
    n_detected: int
    n_used: int
    median_before_arcsec: float
    median_after_arcsec: float
    median_dx_pixels: float
    median_dy_pixels: float


def register_wcs_with_mer_stars(
    image: SmallImage, mer: Table, *, min_vis_flux_ujy: float = 1.0,
    match_radius_pixels: float = 3.0, min_matches: int = 20,
) -> WCSRegistration:
    """Fit an image WCS to bright, isolated MER stars using detected centroids.

    Matching uses the supplied WCS as an initial guess. A robust, common
    translation removes false matches before Astropy refits the TAN WCS.
    Image, variance, and PSF pixels are not resampled. This is intended for
    small initial WCS errors, not blind astrometric solving.
    """
    if min_matches < 3 or match_radius_pixels <= 0 or min_vis_flux_ujy < 0:
        raise ValueError("invalid registration limits")
    for key in ("ra", "dec", "is_star", "flux_vis_sersic"):
        if key not in mer.colnames:
            raise ValueError(f"MER catalog missing {key}")
    data = np.ascontiguousarray(image.science_njy, dtype=np.float32)
    mask = np.ascontiguousarray(image.invvar_njy <= 0)
    background = sep.Background(data, mask=mask)
    detected = sep.extract(data - background.back(), 8.0,
                           err=background.rms(), mask=mask, minarea=5)
    # Reject failed detections and badly elongated objects. The MER star flag
    # provides the main point-source criterion.
    good_detection = ((detected["flag"] == 0) & (detected["a"] > 0) &
                      (detected["b"] / np.maximum(detected["a"], 1e-6) > 0.6))
    detected = detected[good_detection]
    if len(detected) < min_matches:
        raise ValueError("too few usable detections for WCS registration")
    star = (np.asarray(mer["is_star"], bool) &
            (np.asarray(mer["flux_vis_sersic"], float) > min_vis_flux_ujy))
    ra = np.asarray(mer["ra"], float)[star]
    dec = np.asarray(mer["dec"], float)[star]
    x, y = image.wcs.world_to_pixel_values(ra, dec)
    height, width = data.shape
    inside = (np.isfinite(x) & np.isfinite(y) &
              (x > 10) & (x < width - 10) &
              (y > 10) & (y < height - 10))
    ra, dec, x, y = ra[inside], dec[inside], x[inside], y[inside]
    if len(x) < min_matches:
        raise ValueError("too few MER stars inside the image")
    prior_xy = np.column_stack((x, y))
    # Isolate stars in the reference catalog, avoiding blended centroids.
    nearest_prior = cKDTree(prior_xy).query(prior_xy, k=2)[0][:, 1]
    det_xy = np.column_stack((detected["x"], detected["y"]))
    distances, indices = cKDTree(det_xy).query(prior_xy)
    candidate = ((nearest_prior > 12) & (distances < match_radius_pixels))
    selected = np.flatnonzero(candidate)
    if len(selected) < min_matches:
        raise ValueError("too few isolated MER stars match image detections")
    # Keep each image detection only once, preferring its closest MER star.
    selected = selected[np.argsort(distances[selected])]
    _, first = np.unique(indices[selected], return_index=True)
    selected = selected[np.sort(first)]
    measured_xy = det_xy[indices[selected]]
    predicted_xy = prior_xy[selected]
    delta = measured_xy - predicted_xy
    center = np.median(delta, axis=0)
    scatter = np.hypot(*(delta - center).T)
    limit = max(0.3, 4 * 1.4826 * np.median(np.abs(scatter - np.median(scatter))))
    used = scatter < limit
    if used.sum() < min_matches:
        raise ValueError("too few consistent star matches for WCS registration")
    xy = measured_xy[used]
    reference = SkyCoord(ra[selected][used] * u.deg,
                         dec[selected][used] * u.deg)
    # Use the same zero-based centroids as WCS.world_to_pixel. An added pixel
    # here would introduce a one-pixel offset in the returned WCS.
    corrected = fit_wcs_from_points((xy[:, 0], xy[:, 1]),
                                    reference, projection=image.wcs)
    new_x, new_y = corrected.world_to_pixel(reference)
    before = np.hypot(*(xy - predicted_xy[used]).T)
    after = np.hypot(xy[:, 0] - new_x, xy[:, 1] - new_y)
    pixel_scale = float(np.mean(proj_plane_pixel_scales(image.wcs))) * 3600
    matches = Table()
    matches["ra"] = reference.ra.deg
    matches["dec"] = reference.dec.deg
    matches["x_detected"] = xy[:, 0]
    matches["y_detected"] = xy[:, 1]
    matches["x_before"] = predicted_xy[used, 0]
    matches["y_before"] = predicted_xy[used, 1]
    matches["x_after"] = new_x
    matches["y_after"] = new_y
    return WCSRegistration(
        corrected, matches, len(detected), len(xy),
        float(np.median(before) * pixel_scale),
        float(np.median(after) * pixel_scale),
        float(np.median(delta[used, 0])),
        float(np.median(delta[used, 1])),
    )
