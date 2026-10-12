"""Check a supplied ground-image PSF against stars in that same image."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy.table import Table
from scipy.ndimage import shift

from .image import SmallImage


@dataclass(frozen=True)
class PSFAssessment:
    stars: Table
    empirical_psf: np.ndarray | None
    supplied_score: float
    empirical_score: float
    supplied_fwhm_pix: float
    observed_fwhm_pix: float
    field_fwhm_range: float
    recommendation: str  # supplied, empirical, unverified, or spatial


def _core_fwhm(stamp: np.ndarray) -> float:
    """Gaussian-equivalent core width, consistently measured for both PSFs."""
    height, width = stamp.shape
    yy, xx = np.mgrid[:height, :width]
    rr2 = (xx - (width - 1) / 2) ** 2 + (yy - (height - 1) / 2) ** 2
    core = (rr2 <= min(10.0, min(height, width) / 3) ** 2)
    weight = np.maximum(stamp, 0) * core
    total = weight.sum()
    if total <= 0:
        return float("nan")
    return float(2.355 * np.sqrt((weight * rr2).sum() / (2 * total)))


def _star_stamp(image: SmallImage, x: float, y: float,
                radius_y: int, radius_x: int,
                min_snr: float) -> tuple[np.ndarray, float] | None:
    ix, iy = int(round(x)), int(round(y))
    y0, y1 = iy - radius_y, iy + radius_y + 1
    x0, x1 = ix - radius_x, ix + radius_x + 1
    height, width = image.science_njy.shape
    if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
        return None
    data = np.asarray(image.science_njy[y0:y1, x0:x1], float)
    ivar = np.asarray(image.invvar_njy[y0:y1, x0:x1], float)
    yy, xx = np.mgrid[-radius_y:radius_y + 1, -radius_x:radius_x + 1]
    rr = np.hypot(xx - (x - ix), yy - (y - iy))
    radius = min(radius_x, radius_y)
    core = rr < max(4.0, min(radius / 2, 9.0))
    annulus = (rr > 0.72 * radius) & (rr < 0.95 * radius) & (ivar > 0)
    if not np.all(ivar[core] > 0) or annulus.sum() < 20:
        return None
    sky = float(np.median(data[annulus]))
    signal = data - sky
    flux = float(np.sum(signal[core]))
    noise = float(np.sqrt(np.sum(1 / ivar[core])))
    if flux <= 0 or not np.isfinite(noise) or flux / noise < min_snr:
        return None
    # Shift each measured centroid to the stamp center before stacking.
    aligned = shift(signal, (-(y - iy), -(x - ix)), order=3,
                    mode="constant", cval=0.0)
    # Preserve signed noise here. Clipping each stamp before stacking turns
    # zero-mean noise in thousands of outer pixels into positive PSF wings.
    centered_core = np.hypot(xx, yy) < max(4.0, min(radius / 2, 9.0))
    core_flux = float(np.sum(aligned[centered_core]))
    if not np.isfinite(core_flux) or core_flux <= 0:
        return None
    aligned /= core_flux
    if not np.isfinite(_core_fwhm(aligned)):
        return None
    return aligned, flux / noise


def _aligned_reference(psf: np.ndarray, x: float, y: float) -> np.ndarray:
    """Apply the same subpixel sampling and recentering as a stellar stamp."""
    dx, dy = x - round(x), y - round(y)
    sampled = shift(np.pad(psf, 3), (dy, dx), order=3,
                    mode="constant", cval=0.0)[3:-3, 3:-3]
    aligned = shift(sampled, (-dy, -dx), order=3,
                    mode="constant", cval=0.0)
    aligned = np.maximum(aligned, 0)
    return aligned / aligned.sum()


def reference_star_comparison(
    image: SmallImage, x: float, y: float, *, min_snr: float = 30.0,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return a sky-subtracted star and supplied PSF on the same core-flux scale.

    These are the same recentered stellar pixels and subpixel-sampled PSF used
    by the validation score. The PSF amplitude is fitted in the central 10
    pixels; the star's signed outer-wing noise is retained for inspection.
    """
    radius_y, radius_x = image.psf.shape[0] // 2, image.psf.shape[1] // 2
    extracted = _star_stamp(image, x, y, radius_y, radius_x, min_snr)
    if extracted is None:
        return None
    star, _ = extracted
    reference = _aligned_reference(np.asarray(image.psf, float), x, y)
    yy, xx = np.mgrid[:star.shape[0], :star.shape[1]]
    core = ((xx - radius_x) ** 2 + (yy - radius_y) ** 2 <= 10.0**2)
    amplitude = float(np.sum(star[core] * reference[core]) /
                      np.sum(reference[core] ** 2))
    return star, amplitude * reference


def _comparison_score(stamps: np.ndarray, psf: np.ndarray) -> float:
    """Median fractional core RMS after fitting each held-out star's amplitude."""
    height, width = stamps.shape[1:]
    yy, xx = np.mgrid[:height, :width]
    core = ((xx - (width - 1) / 2) ** 2 +
            (yy - (height - 1) / 2) ** 2 <= min(10, min(height, width) / 3) ** 2)
    if psf.ndim == 3:
        predicted = psf[:, core]
    else:
        predicted = np.broadcast_to(psf[core], (len(stamps), np.count_nonzero(core)))
    observed = stamps[:, core]
    amplitude = np.sum(observed * predicted, axis=1) / np.sum(predicted**2, axis=1)
    scale = np.maximum(np.abs(amplitude) * np.max(predicted, axis=1), 1e-12)
    return float(np.median(np.sqrt(np.mean(
        (observed - amplitude[:, None] * predicted) ** 2, axis=1)) / scale))


def _radialize_noisy_wings(stamp: np.ndarray) -> np.ndarray:
    """Suppress sparse neighbor/noise peaks while retaining a measured core."""
    height, width = stamp.shape
    yy, xx = np.mgrid[:height, :width]
    radius = np.hypot(xx - (width - 1) / 2, yy - (height - 1) / 2)
    ring = np.floor(radius).astype(int)
    radial_from = min(8, max(3, min(height, width) // 4))
    result = stamp.copy()
    for r in range(radial_from, int(ring.max()) + 1):
        pixels = ring == r
        result[pixels] = np.median(stamp[pixels])
    result = np.maximum(result, 0)
    total = result.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("empirical PSF has no positive flux")
    return result / total


def average_star_psf(image: SmallImage, stars: Table,
                     excluded_indices=()) -> np.ndarray:
    """Equal-weight mean of usable, aligned image stars, with cleaned wings."""
    excluded = set(excluded_indices)
    stamps = []
    for index, star in enumerate(stars):
        if index in excluded:
            continue
        pair = reference_star_comparison(
            image, float(star["x"]), float(star["y"]))
        if pair is not None:
            stamps.append(np.asarray(pair[0], dtype=float))
    if not stamps:
        raise ValueError("No selected star yields a usable average PSF stamp")
    return _radialize_noisy_wings(np.mean(stamps, axis=0))


def assess_supplied_psf(
    image: SmallImage, matched_stars: Table, *, min_stars: int = 6,
    min_snr: float = 30.0, improvement_fraction: float = 0.10,
    spatial_tolerance: float = 0.15, min_fractional_rms: float = 0.02,
    width_tolerance: float = 0.08,
) -> PSFAssessment:
    """Validate one PSF stamp with registered, isolated stellar centroids.

    ``matched_stars`` is the ``registration.matches`` table from the WCS
    step. No reference-catalog photometry enters the PSF construction.
    The empirical stamp is a median of 75% of usable stars; the remaining
    25% independently score both PSFs. A clear field-dependent width change
    flags that neither single stamp describes the full image.
    """
    if min_stars < 4 or min_snr <= 0 or not 0 < improvement_fraction < 1:
        raise ValueError("invalid PSF validation limits")
    if not {"x_detected", "y_detected"} <= set(matched_stars.colnames):
        raise ValueError("matched_stars needs x_detected and y_detected")
    psf = np.asarray(image.psf, float)
    if any(side % 2 != 1 for side in psf.shape):
        raise ValueError("PSF validation needs an odd-sized PSF stamp")
    radius_y, radius_x = psf.shape[0] // 2, psf.shape[1] // 2
    rows, stamps = [], []
    for row in matched_stars:
        x, y = float(row["x_detected"]), float(row["y_detected"])
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        extracted = _star_stamp(image, x, y, radius_y, radius_x, min_snr)
        if extracted is None:
            continue
        stamp, snr = extracted
        rows.append((x, y, snr, _core_fwhm(stamp)))
        stamps.append(stamp)
    stars = Table(rows=rows, names=("x", "y", "snr", "fwhm_pix"),
                  dtype=(float, float, float, float))
    supplied_width = _core_fwhm(psf)
    if len(stars) < min_stars:
        return PSFAssessment(stars, None, np.nan, np.nan,
                             supplied_width, np.nan, np.nan, "unverified")

    stamps = np.asarray(stamps)
    widths = np.asarray(stars["fwhm_pix"], float)
    observed_width = float(np.median(widths))
    # A difference between well-populated left/right or top/bottom regions
    # warns against using one PSF for a large detector.
    ranges = []
    for coordinate in ("x", "y"):
        values = np.asarray(stars[coordinate], float)
        left = widths[values <= np.median(values)]
        right = widths[values > np.median(values)]
        if min(len(left), len(right)) >= 3:
            ranges.append(abs(np.median(left) - np.median(right)) / observed_width)
    field_range = float(max(ranges, default=0))

    # Every fourth star is held out; sorting by x spreads the validation
    # sample across the field instead of concentrating it at one edge.
    order = np.argsort(np.asarray(stars["x"], float))
    holdout = np.zeros(len(stars), bool)
    holdout[order[::4]] = True
    empirical = _radialize_noisy_wings(np.median(stamps[~holdout], axis=0))
    held_stars = stars[holdout]
    held_supplied = np.asarray([
        _aligned_reference(psf, float(row["x"]), float(row["y"]))
        for row in held_stars])
    supplied_score = _comparison_score(stamps[holdout], held_supplied)
    empirical_score = _comparison_score(stamps[holdout], empirical)
    if field_range > spatial_tolerance:
        recommendation = "spatial"
    elif (empirical_score < (1 - improvement_fraction) * supplied_score and
          (supplied_score > min_fractional_rms or
           abs(supplied_width / observed_width - 1) > width_tolerance)):
        recommendation = "empirical"
    else:
        recommendation = "supplied"
    return PSFAssessment(stars, empirical, supplied_score, empirical_score,
                         supplied_width, observed_width, field_range,
                         recommendation)
