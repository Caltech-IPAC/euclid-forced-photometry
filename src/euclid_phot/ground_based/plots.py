"""Compact, reusable notebook figures for a user-supplied ground image."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import map_coordinates

from ..units import flux_err_to_mag_err, flux_to_ab_mag, pixel_area_arcsec2, pixel_scale_arcsec
from ..viz import ujy_per_unit
from .psf_validation import reference_star_comparison


def _display_scale(data, *, percentile=99.5):
    finite = np.asarray(data)[np.isfinite(data)]
    return max(float(np.percentile(np.abs(finite), percentile)), 1e-9) if len(finite) else 1.0


def _vis_on_ground(vis_cutout, image):
    """VIS surface brightness on the ground pixel grid, for display only."""
    vis_sb = (np.asarray(vis_cutout.data, float) * ujy_per_unit(vis_cutout) /
              pixel_area_arcsec2(vis_cutout.wcs))
    yy, xx = np.indices(image.science_ujy.shape, dtype=float)
    ra, dec = image.wcs.pixel_to_world_values(xx, yy)
    vis_x, vis_y = vis_cutout.wcs.world_to_pixel_values(ra, dec)
    return map_coordinates(vis_sb, (vis_y, vis_x), order=1,
                           mode="constant", cval=np.nan, prefilter=False)


def show_inputs(inputs):
    """Science, RMS and normalized PSF with colorbars and angular scale."""
    image = inputs.image
    valid = image.invvar_ujy > 0
    area = pixel_area_arcsec2(image.wcs)
    display = image.science_ujy / area
    rms = np.full(valid.shape, np.nan)
    rms[valid] = 1 / np.sqrt(image.invvar_ujy[valid]) / area
    scale = inputs.pixel_scale_arcsec
    ny, nx = display.shape
    extent = (0, nx * scale, 0, ny * scale)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, data, title, label, cmap, lim, im_extent in (
        (axes[0], display, "Science", "µJy/arcsec²", "gray",
         (-_display_scale(display) / 10, _display_scale(display)), extent),
        (axes[1], rms, "Uncertainty", "µJy/arcsec²", "viridis",
         (None, None), extent),
        (axes[2], image.psf, "Supplied PSF", "PSF fraction", "magma",
         (None, None), None),
    ):
        artist = ax.imshow(data, origin="lower", cmap=cmap,
                           vmin=lim[0], vmax=lim[1], extent=im_extent)
        if im_extent is not None:
            ax.set(xlabel="x (arcsec)", ylabel="y (arcsec)")
        else:
            ax.set(xlabel="PSF x (pixels)", ylabel="PSF y (pixels)")
        ax.set_title(title)
        fig.colorbar(artist, ax=ax, label=label)
    fig.tight_layout()
    return fig


def show_scene(inputs, euclid):
    """Show VIS and ground on the same WCS and surface-brightness scale."""
    image = inputs.image
    vis = euclid.vis_cutout
    scale = inputs.pixel_scale_arcsec
    ny, nx = image.science_ujy.shape
    area = pixel_area_arcsec2(image.wcs)
    ground = (image.science_ujy -
              np.median(image.science_ujy[image.invvar_ujy > 0])) / area
    vis_aligned = _vis_on_ground(vis, image)
    vis_aligned -= np.nanmedian(vis_aligned)
    vmax = max(_display_scale(vis_aligned), _display_scale(ground))
    extent = (0, nx * scale, 0, ny * scale)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    artist = axes[0].imshow(vis_aligned, origin="lower", cmap="gray",
                            vmin=-vmax / 10, vmax=vmax, extent=extent)
    axes[0].set(title="Euclid VIS (aligned)", xlabel="x (arcsec)",
                ylabel="y (arcsec)")
    fig.colorbar(artist, ax=axes[0], label="µJy/arcsec²")
    artist = axes[1].imshow(ground, origin="lower", cmap="gray",
                            vmin=-vmax / 10, vmax=vmax, extent=extent)
    x, y = image.wcs.world_to_pixel_values(np.asarray(euclid.mer["ra"], float),
                                            np.asarray(euclid.mer["dec"], float))
    inside = (x >= 0) & (x < nx) & (y >= 0) & (y < ny)
    axes[1].scatter(x[inside] * scale, y[inside] * scale, s=22,
                    facecolors="none", edgecolors="tab:cyan", linewidths=0.6)
    axes[1].set(title=f"Ground image; {inside.sum()} MER sources",
                xlabel="x (arcsec)", ylabel="y (arcsec)")
    fig.colorbar(artist, ax=axes[1], label="µJy/arcsec²")
    fig.tight_layout()
    return fig


def show_registration(registration):
    matches = registration.matches
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for prefix, label in (("before", "Supplied WCS"), ("after", "Registered WCS")):
        ax.scatter(np.asarray(matches[f"x_{prefix}"]) - matches["x_detected"],
                   np.asarray(matches[f"y_{prefix}"]) - matches["y_detected"],
                   s=18, alpha=0.6, label=label)
    ax.axhline(0, color="0.7", lw=0.8)
    ax.axvline(0, color="0.7", lw=0.8)
    ax.set(xlabel="Predicted − detected x (pixels)",
           ylabel="Predicted − detected y (pixels)",
           title="WCS residuals on matched stars")
    ax.legend()
    fig.tight_layout()
    return fig


def show_psf_stars(image, assessment, *, max_stars=None, star_index=None):
    """Show star, supplied PSF, and radial profile; optionally select one star."""
    if star_index is None:
        candidates = enumerate(assessment.stars)
    else:
        if not 0 <= star_index < len(assessment.stars):
            raise IndexError("star_index is outside the assessed star list")
        candidates = [(star_index, assessment.stars[star_index])]
    comparisons = []
    for index, star in candidates:
        pair = reference_star_comparison(image, float(star["x"]), float(star["y"]))
        if pair is not None:
            comparisons.append((index, star, pair))
    if max_stars is not None:
        comparisons = comparisons[:max_stars]
    if not comparisons:
        return None
    scale = pixel_scale_arcsec(image.wcs)
    fig, axes = plt.subplots(len(comparisons), 3,
                             figsize=(13, 3.1 * len(comparisons)), squeeze=False)
    for (number, star, (observed, template)), row in zip(comparisons, axes):
        half_y, half_x = (np.array(observed.shape) - 1) / 2
        extent = (-(half_x + .5) * scale, (half_x + .5) * scale,
                  -(half_y + .5) * scale, (half_y + .5) * scale)
        vmax = max(_display_scale(observed, percentile=99),
                   _display_scale(template, percentile=99))
        for ax, data, title in zip(row[:2], (observed, template),
                                   ("Science star − local sky", "Supplied PSF")):
            artist = ax.imshow(data, origin="lower", cmap="magma",
                               vmin=-vmax / 20, vmax=vmax, extent=extent)
            ax.set(title=title, xlabel="x offset (arcsec)",
                   ylabel="y offset (arcsec)")
            fig.colorbar(artist, ax=ax, label="scaled light")
        yy, xx = np.indices(observed.shape)
        radius = np.hypot(xx - half_x, yy - half_y) * scale
        bins = np.arange(0, np.max(radius) + scale, scale)
        centers = (bins[:-1] + bins[1:]) / 2
        for data, label in ((observed, "Star"), (template, "PSF")):
            profile = [np.nanmedian(data[(radius >= lo) & (radius < hi)])
                       for lo, hi in zip(bins[:-1], bins[1:])]
            row[2].plot(centers, profile, marker=".", ms=3, label=label)
        row[2].set(xlabel="Radius (arcsec)", ylabel="Scaled light / pixel",
                   title=f"Star {number}: x={float(star['x']):.0f}, y={float(star['y']):.0f}")
        row[2].set_xlim(0, 10)
        row[2].set_yscale("symlog", linthresh=1e-3)
        row[2].set_ylim(-1e-3, 2e-1)
        row[2].axhline(0, color="black", lw=0.8)
        row[2].legend()
    fig.tight_layout()
    return fig


def show_fit_cutouts(fit, vis_cutout, *, size_arcsec=20.0, n_sources=3,
                    object_ids=None, vmin=None, vmax=None):
    """Show aligned VIS and ground cutouts in µJy/arcsec².

    ``vmin`` and ``vmax`` set the common VIS/science/model limits; omitted
    limits are chosen separately for each source. Residuals use ±20% of the
    positive display limit. Explicit ``object_ids`` include edge sources.
    """
    if vmax is not None and (not np.isfinite(vmax) or vmax <= 0):
        raise ValueError("vmax must be positive and finite")
    if vmin is not None and not np.isfinite(vmin):
        raise ValueError("vmin must be finite")
    if vmin is not None and vmax is not None and vmin >= vmax:
        raise ValueError("vmin must be smaller than vmax")
    catalog = fit.catalog
    image = fit.image
    scale = pixel_scale_arcsec(image.wcs)
    area = pixel_area_arcsec2(image.wcs)
    vis_aligned = _vis_on_ground(vis_cutout, image)
    vis_aligned -= np.nanmedian(vis_aligned)
    half = max(4, int(np.ceil(size_arcsec / (2 * scale))))
    x, y = np.asarray(catalog["x"], float), np.asarray(catalog["y"], float)
    band = catalog.meta["BAND"]
    flux = np.asarray(catalog[f"flux_{band}_ujy"], float)
    error = np.asarray(catalog[f"flux_err_{band}_ujy"], float)
    with np.errstate(divide="ignore", invalid="ignore"):
        detected = (error > 0) & (flux / error > 5)
    if object_ids is None:
        eligible = np.flatnonzero(np.isfinite(flux) & detected &
                                 (x > half) & (x < image.science_ujy.shape[1] - half) &
                                 (y > half) & (y < image.science_ujy.shape[0] - half))
        if not len(eligible):
            return None
        rank = eligible[np.argsort(flux[eligible])]
        # Well-detected quantiles avoid extreme bright stars and noise peaks.
        quantiles = np.linspace(.9, .1, min(n_sources, len(rank)))
        chosen = list(dict.fromkeys(rank[np.clip(
            (quantiles * (len(rank)-1)).round().astype(int), 0, len(rank)-1)]))
    else:
        object_ids = list(object_ids)
        by_id = {int(obj_id): i for i, obj_id in
                 enumerate(np.asarray(catalog["object_id"].data, dtype=np.int64))}
        missing = [int(obj_id) for obj_id in object_ids if int(obj_id) not in by_id]
        if missing:
            raise ValueError(f"object IDs absent from fit catalog: {missing}")
        chosen = list(dict.fromkeys(by_id[int(obj_id)] for obj_id in object_ids))
        if not chosen:
            return None

    def cutout(array, xi, yi):
        ny, nx = array.shape
        x0, x1 = max(0, xi-half), min(nx, xi+half+1)
        y0, y1 = max(0, yi-half), min(ny, yi+half+1)
        result = np.full((2*half+1, 2*half+1), np.nan)
        result[y0-(yi-half):y1-(yi-half),
               x0-(xi-half):x1-(xi-half)] = array[y0:y1, x0:x1]
        return result

    fig, axes = plt.subplots(len(chosen), 4, figsize=(16, 3.7 * len(chosen)),
                             squeeze=False)
    for row, i in zip(axes, chosen):
        xi, yi = int(round(x[i])), int(round(y[i]))
        ground = cutout(image.science_ujy - fit.background, xi, yi) / area
        model = cutout(fit.model - fit.background, xi, yi) / area
        resid = cutout(fit.residual, xi, yi) / area
        auto_max = _display_scale(ground)
        hi = auto_max if vmax is None else float(vmax)
        lo = -hi/10 if vmin is None else float(vmin)
        if lo >= hi:
            raise ValueError("vmin must be smaller than the display vmax")
        extent = (-(half+.5)*scale, (half+.5)*scale,
                  -(half+.5)*scale, (half+.5)*scale)
        panels = ((cutout(vis_aligned, xi, yi), "VIS", lo, hi),
                  (ground, "Ground based", lo, hi),
                  (model, "Tractor", lo, hi),
                  (resid, "Residual", -.2*hi, .2*hi))
        for ax, (data, title, lo, hi) in zip(row, panels):
            artist = ax.imshow(data, origin="lower", cmap="RdBu_r" if title == "Residual" else "gray",
                               vmin=lo, vmax=hi, extent=extent)
            ax.set_title(title)
            ax.set(xlabel="x offset (arcsec)", ylabel="y offset (arcsec)")
            fig.colorbar(artist, ax=ax, label="µJy/arcsec²")
        row[0].set_ylabel(f"ID {catalog['object_id'][i]}\ny offset (arcsec)")
    fig.tight_layout()
    return fig


def show_catalog_comparison(comparison, *, highlight_object_id=None):
    """Compare SDSS matches, with rejected rows behind the accepted points."""
    good = np.asarray(comparison["accepted"], bool)
    ref = np.asarray(comparison["reference_flux_ujy"], float)
    fit = np.asarray(comparison["forced_flux_ujy"], float)
    ref_err = np.asarray(comparison["reference_fluxerr_ujy"], float)
    fit_err = np.asarray(comparison["forced_fluxerr_ujy"], float)
    plottable = np.isfinite(ref) & np.isfinite(fit) & (ref > 0) & (fit > 0)
    if not np.any(plottable):
        return None
    rejected = ~good & plottable
    ref_mag = flux_to_ab_mag(np.where(plottable, ref, np.nan))
    fit_mag = flux_to_ab_mag(np.where(plottable, fit, np.nan))
    ref_mag_err = flux_err_to_mag_err(ref[good], ref_err[good])
    fit_mag_err = flux_err_to_mag_err(fit[good], fit_err[good])
    kind = np.asarray(comparison["source_type"])
    if highlight_object_id is not None:
        object_ids = np.asarray(comparison["euclid_object_id"].data,
                                dtype=np.int64)
        highlighted = good & (object_ids == int(highlight_object_id))
        if not np.any(highlighted):
            raise ValueError(f"object ID {highlight_object_id} has no clean SDSS match")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    if np.any(rejected):
        axes[0].scatter(ref_mag[rejected], fit_mag[rejected],
                        s=30, color="gray", alpha=.3, label="Not accepted")
        axes[1].scatter(ref_mag[rejected],
                        (ref_mag - fit_mag)[rejected],
                        s=30, color="gray", alpha=.3, label="Not accepted")
    for label, color in (("star", "tab:blue"), ("galaxy", "tab:orange")):
        take = good & (kind == label)
        if not np.any(take):
            continue
        take_good = (kind[good] == label)
        axes[0].errorbar(ref_mag[take], fit_mag[take],
                         xerr=ref_mag_err[take_good], yerr=fit_mag_err[take_good],
                         fmt="o", ms=7, lw=.5, alpha=.65, color=color, label=label)
        axes[1].errorbar(ref_mag[take], (ref_mag-fit_mag)[take],
                         yerr=np.hypot(ref_mag_err[take_good], fit_mag_err[take_good]),
                         fmt="o", ms=7, lw=.5, alpha=.65, color=color, label=label)
    if highlight_object_id is not None:
        for ax, y in ((axes[0], fit_mag),
                      (axes[1], ref_mag - fit_mag)):
            ax.scatter(ref_mag[highlighted], y[highlighted], s=220,
                       facecolors="none", edgecolors="crimson", linewidths=2,
                       zorder=10, label=f"ID {highlight_object_id}")
    unplottable = np.count_nonzero(~good & ~plottable)
    if unplottable:
        axes[0].text(.02, .02, f"{unplottable} rejected: no AB magnitude",
                     transform=axes[0].transAxes, color="gray", fontsize=8,
                     ha="left", va="bottom")
    axes[0].plot((13, 26), (13, 26), "k--", lw=1)
    axes[0].set(xlim=(26, 13), ylim=(26, 13),
                xlabel="SDSS (AB Mag)", ylabel="This pipeline (AB Mag)",
                title="VIS-prior fit versus SDSS")
    axes[0].legend()
    axes[1].axhline(0, color="k", ls="--", lw=1)
    axes[1].set(xlim=(26, 13), ylim=(-2, 2), xlabel="SDSS (AB Mag)",
                ylabel="SDSS − Forced (mag)", title="Magnitude difference")
    if highlight_object_id is not None:
        axes[1].legend()
    fig.tight_layout()
    return fig


def show_background_comparison(no_sky_fit, sep_fit, *, limit_ujy_per_pixel=0.05):
    """Compare both fitted residuals using one symmetric brightness scale."""
    data = (no_sky_fit.residual, sep_fit.residual)
    limit = float(limit_ujy_per_pixel)
    if not np.isfinite(limit) or limit <= 0:
        raise ValueError("residual display limit must be positive")
    ny, nx = data[0].shape
    scale = pixel_scale_arcsec(no_sky_fit.image.wcs)
    extent = (0, nx*scale, 0, ny*scale)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, array, title in zip(axes, data,
                                ("No additional sky subtraction", "SEP 20″ sky subtraction")):
        artist = ax.imshow(array, origin="lower", cmap="RdBu_r",
                           vmin=-limit, vmax=limit, extent=extent)
        ax.set(title=title, xlabel="x (arcsec)", ylabel="y (arcsec)")
        fig.colorbar(artist, ax=ax, label="Residual (µJy/pixel)")
    fig.tight_layout()
    return fig


def show_sky_injection_test(injected_sky, baseline_fit, no_sky_fit, sep_fit,
                            *, smooth_sky=None, limit_ujy_per_pixel=0.05):
    """Compare sky recovery and injected-image fluxes with the original fit."""
    limit = float(limit_ujy_per_pixel)
    if not np.isfinite(limit) or limit <= 0:
        raise ValueError("residual display limit must be positive")
    true_sky = np.asarray(injected_sky, dtype=float)
    smooth_sky = true_sky if smooth_sky is None else np.asarray(smooth_sky, dtype=float)
    recovered_sky = np.asarray(sep_fit.background, dtype=float)
    if (true_sky.shape != no_sky_fit.residual.shape or
            true_sky.shape != smooth_sky.shape or
            true_sky.shape != sep_fit.residual.shape or
            true_sky.shape != recovered_sky.shape):
        raise ValueError("sky and residual maps must have the same shape")
    ny, nx = true_sky.shape
    scale = pixel_scale_arcsec(no_sky_fit.image.wcs)
    extent = (0, nx*scale, 0, ny*scale)
    sky_lo = min(np.nanpercentile(true_sky, 1),
                 np.nanpercentile(recovered_sky, 1))
    sky_hi = max(np.nanpercentile(true_sky, 99),
                 np.nanpercentile(recovered_sky, 99))
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    maps = (
        (true_sky, "Injected sky", "viridis", sky_lo, sky_hi),
        (recovered_sky, "SEP sky estimate", "viridis", sky_lo, sky_hi),
        (recovered_sky - smooth_sky, "SEP − smooth sky", "RdBu_r", -limit, limit),
        (no_sky_fit.residual, "Residual: no subtraction", "RdBu_r", -limit, limit),
        (sep_fit.residual, "Residual: SEP subtraction", "RdBu_r", -limit, limit),
    )
    for ax, (array, title, cmap, lo, hi) in zip(axes.flat, maps):
        artist = ax.imshow(array, origin="lower", cmap=cmap,
                           vmin=lo, vmax=hi, extent=extent)
        ax.set(title=title, xlabel="x (arcsec)", ylabel="y (arcsec)")
        fig.colorbar(artist, ax=ax, label="µJy/pixel")

    def rows_by_id(catalog):
        ids = np.asarray(catalog["object_id"].data, dtype=np.int64)
        if len(np.unique(ids)) != len(ids):
            raise ValueError("ground fit has duplicate object IDs")
        return {int(object_id): i for i, object_id in enumerate(ids)}

    catalogs = (baseline_fit.catalog, no_sky_fit.catalog, sep_fit.catalog)
    rows = [rows_by_id(catalog) for catalog in catalogs]
    common = sorted(set(rows[0]) & set(rows[1]) & set(rows[2]))
    ax = axes[1, 2]
    if common:
        indices = [[row[object_id] for object_id in common] for row in rows]
        fluxes = [np.asarray(catalog[f"flux_{catalog.meta['BAND']}_ujy"].data, float)[index]
                  for catalog, index in zip(catalogs, indices)]
        errors = [np.asarray(catalog[f"flux_err_{catalog.meta['BAND']}_ujy"].data, float)[index]
                  for catalog, index in zip(catalogs, indices)]
        inside = [np.asarray(catalog["in_image"].data, bool)[index]
                  for catalog, index in zip(catalogs, indices)]
        valid = np.logical_and.reduce(inside)
        for flux, error in zip(fluxes, errors):
            valid &= np.isfinite(flux) & np.isfinite(error) & (flux > 0) & (error > 0)
        valid &= fluxes[0] / errors[0] > 5
        baseline_flux = fluxes[0][valid]
        baseline_mag = flux_to_ab_mag(baseline_flux)
        for flux, color, label in (
            (fluxes[1][valid], "tab:orange", "Before subtraction"),
            (fluxes[2][valid], "tab:blue", "After SEP subtraction"),
        ):
            delta_mag = 2.5 * np.log10(flux / baseline_flux)
            ax.scatter(baseline_mag, delta_mag, s=22, alpha=.65,
                       color=color,
                       label=f"{label} (median {np.median(delta_mag):+.2f})"
                       if len(delta_mag) else label)
        ax.set_title(f"Same {valid.sum()} VIS-prior sources")
        ax.invert_xaxis()
        ax.legend(fontsize=8)
    else:
        ax.text(.5, .5, "No common VIS-prior sources", ha="center",
                va="center", transform=ax.transAxes)
    ax.axhline(0, color="black", ls="--", lw=1)
    ax.set(xlabel="Original forced photometry (AB mag)",
           ylabel="Original − injected fit (mag)")
    fig.tight_layout()
    return fig
