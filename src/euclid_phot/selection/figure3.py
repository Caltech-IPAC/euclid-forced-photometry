"""Reproduce Weaver et al. (2023) Figure 3 on one blob."""

from __future__ import annotations

import numpy as np

from .models import Blob, History


# ---------------------------------------------------------------------------
# Figure reproduction (Weaver+2023 Fig. 3).
# ---------------------------------------------------------------------------
def reproduce_figure3(
    blob: Blob,
    history: History,
    *,
    save_path: str | None = None,
    figsize_per_panel: float = 2.8,
    vlim_sigma: float = 5.0,
    flux_scale: float = 1.0,
    unit: str = "image units",
):
    """Plot the Weaver+2023 Fig. 3-style per-tier residual panels.

    Row 0: VIS data cutout and final residual (linear, ±vlim_sigma·σ); one
    row per (source, stage) trial with the model image (log stretch) and
    residual, annotated with the reduced chi^2 and a check on the accepted
    tier. σ is the sigma-clipped residual RMS outside the blob footprint.
    ``flux_scale`` converts image units for display, e.g.
    ``ep.viz.ujy_per_unit(cutout)`` with ``unit="µJy/pixel"``.
    """
    import matplotlib.pyplot as plt
    from astropy.stats import sigma_clipped_stats
    from matplotlib.colors import LogNorm, Normalize

    if history.final is None:
        raise ValueError("History has no final record; call fit_blob(..., history=True).")

    final = history.final
    H, W = final.data.shape

    # Sigma-clipped residual stats outside the blob footprint, so core
    # residual structure does not inflate the stretch.
    outside = ~blob.footprint
    if outside.sum() > 100:
        _, _, sigma = sigma_clipped_stats(final.residual[outside], sigma=3.0)
    else:
        sigma = float(np.std(final.residual[blob.footprint]))
    sigma = float(sigma)
    if sigma <= 0:
        sigma = 1e-3
    # The paper uses ±3σ; ±5σ here because bright Q1 galaxies carry a few
    # 3-5σ pixels (astrometry, PSF stamp imperfections, substructure) that
    # saturate a ±3σ stretch.
    vlim = float(vlim_sigma) * sigma

    ymin, ymax = np.argwhere(blob.footprint).min(0), np.argwhere(blob.footprint).max(0)
    pad = 5
    y0 = max(0, int(ymin[0]) - pad)
    y1 = min(H, int(ymax[0]) + pad + 1)
    x0 = max(0, int(ymin[1]) - pad)
    x1 = min(W, int(ymax[1]) + pad + 1)

    def crop(a):
        return a[y0:y1, x0:x1]

    footprint_crop = crop(blob.footprint)
    k = float(flux_scale)

    norm_lin = Normalize(vmin=-vlim * k, vmax=vlim * k)

    # Log stretch from the final model so all tier rows share one scale.
    model_in = crop(final.model)[footprint_crop]
    if model_in.size and np.any(model_in > 0):
        positive = model_in[model_in > 0]
        v_hi = float(np.nanpercentile(positive, 99.5))
        v_lo = float(max(sigma, 1e-6))
        if v_hi <= v_lo:
            v_hi = v_lo * 10
        norm_log = LogNorm(vmin=v_lo * k, vmax=v_hi * k)
    else:
        norm_log = LogNorm(vmin=sigma * k, vmax=max(sigma * 100, 1e-3) * k)

    cmap_img = "RdBu_r"
    cmap_mod = "Grays"
    n_rows = 1 + len(history.stages)
    fig, axes = plt.subplots(n_rows, 2,
                             figsize=(2 * figsize_per_panel, n_rows * figsize_per_panel),
                             squeeze=False)

    axes[0, 0].imshow(crop(final.data) * k,
                       origin="lower", cmap=cmap_img, norm=norm_lin)
    axes[0, 0].set_title(f"VIS data  (±{vlim_sigma:.0f}σ, σ={sigma * k:.3g})", fontsize=13)
    axes[0, 1].imshow(crop(final.residual) * k,
                       origin="lower", cmap=cmap_img, norm=norm_lin)
    axes[0, 1].set_title(f"Final residual  (±{vlim_sigma:.0f}σ)", fontsize=13)

    for r, rec in enumerate(history.stages, start=1):
        mod = crop(rec.model_image) * k
        res = crop(rec.residual_image) * k
        mod_clip = np.clip(mod, norm_log.vmin, None)
        check = "  (accepted)" if rec.accepted else ""
        title_color = "darkgreen" if rec.accepted else "black"

        axes[r, 0].imshow(mod_clip, origin="lower", cmap=cmap_mod, norm=norm_log)
        axes[r, 0].set_title(f"src#{rec.source_idx} = {rec.model_name}{check}\n"
                              f"(joint blob model)", fontsize=12,
                              color=title_color)
        axes[r, 1].imshow(res, origin="lower", cmap=cmap_img, norm=norm_lin)
        axes[r, 1].set_title(f"joint residual   χ²ν(src#{rec.source_idx}) = "
                              f"{rec.chi2_red:.2f}", fontsize=12,
                              color=title_color)

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = fig.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.03)
        if ax.images[0].norm is norm_lin:
            cbar.set_label(f"{unit} (±{vlim_sigma:.0f}σ)")
        else:
            cbar.set_label(f"model, {unit} (log)")

    fig.suptitle(f"Farmer-style decision tree on blob #{blob.blob_id} "
                 f"({len(history.members)} source{'s' if len(history.members) != 1 else ''})",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, axes
