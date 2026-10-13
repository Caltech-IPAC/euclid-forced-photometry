"""Model classes, Euclid defaults and per-blob records of the selection tree."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from tractor import Image, NanoMaggies, PointSource, RaDecPos
from tractor.ellipses import EllipseE, EllipseESoft
from tractor.galaxy import DevGalaxy, ExpGalaxy, FixedCompositeGalaxy, SoftenedFracDev
from tractor.sersic import SersicGalaxy, SersicIndex

from ..units import UJY_PER_NMGY


# ---------------------------------------------------------------------------
# Euclid-tuned defaults (see module docstring for rationale).
# ---------------------------------------------------------------------------
EUCLID_DEFAULTS = dict(
    sufficient_thresh=1.5,
    simplegalaxy_penalty=0.05,
    exp_dev_similar_thresh=0.15,
    simplegalaxy_radius=0.30,
    # Max centroid drift in the final joint re-fit, where position is thawed
    # to absorb sub-pixel MER-vs-VIS astrometry offsets. ~1 VIS pixel.
    max_pos_shift_arcsec=0.30,
    sep_threshold_sigma=1.5,
    sep_minarea=5,
    blob_dilate_pix=4,
    max_steps=50,
    dlnp_crit=1e-3,
    # A railed radius leaves a ring residual on large VIS galaxies, so use
    # the same 20 arcsec guard sources.py applies to MER-seeded shapes.
    max_re_arcsec=20.0,
    # Every tier evaluated, lowest reduced chi-squared wins; the published
    # first-sufficient shortcut biases faint fluxes low.
    full_ladder=True,
    full_ladder_min_snr=0.0,
    per_source_psf=False,
    # Guarded SersicGalaxy 6th tier (extension beyond published Farmer);
    # on by default: accepted only when it lowers the total chi-squared on
    # the wing-wide window, both models scored there, by more than
    # sersic_dchi2_accept (a likelihood-ratio test for the one extra shape
    # parameter; 25 is a ~5 sigma preference).
    include_sersic_tier=True,
    sersic_min_snr=20.0,
    sersic_dchi2_accept=25.0,
    sersic_init_n=2.5,
)

MODEL_NAMES = ("PointSource", "SimpleGalaxy", "ExpGalaxy", "DevGalaxy",
               "CompositeGalaxy", "SersicGalaxy")


# ---------------------------------------------------------------------------
# SimpleGalaxy: Farmer's bridge tier, adapted to Tractor's standard
# parameter-freezing API.
# ---------------------------------------------------------------------------
class SimpleGalaxy(ExpGalaxy):
    """Exponential profile with a frozen circular shape at a fixed radius.

    Used as the "marginally resolved" bridge tier between PointSource and the
    free Exp/Dev fits. The frozen circular radius is set per instance via the
    ``radius_arcsec`` argument (the selector passes its ``simplegalaxy_radius``);
    when omitted it falls back to the Euclid-tuned class default (0.30 arcsec).

    The shape parameter is frozen via Tractor's standard ``freezeParam`` API
    so ``numberOfParams()`` and ``getDerivs()`` stay consistent under
    ``optimize_forced_photometry`` (Farmer's original ``isParamFrozen``
    override is incompatible with that path).
    """

    DEFAULT_RADIUS = EUCLID_DEFAULTS["simplegalaxy_radius"]

    def __init__(self, pos, brightness, radius_arcsec=None):
        r = SimpleGalaxy.DEFAULT_RADIUS if radius_arcsec is None else float(radius_arcsec)
        shape = EllipseE(r, 0.0, 0.0)
        super().__init__(pos, brightness, shape)
        self.freezeParam("shape")

    def getName(self):
        return "SimpleGalaxy"


def set_simplegalaxy_radius(radius_arcsec: float) -> None:
    """Set the class-default SimpleGalaxy shape radius (arcsec).

    Only affects instances built with no explicit ``radius_arcsec``; the
    fit path passes the selector's ``simplegalaxy_radius`` explicitly.
    """
    SimpleGalaxy.DEFAULT_RADIUS = float(radius_arcsec)


# ---------------------------------------------------------------------------
# Blob and History dataclasses.
# ---------------------------------------------------------------------------
@dataclass
class Blob:
    """A connected group of pixels and the MER sources that fall inside it.

    ``footprint`` is the dilated boolean mask over the VIS cutout that defines
    where chi^2 is evaluated for this blob (the rest of the image gets zero
    invvar). ``segments`` maps each member source's MER row index to its own
    per-source segment mask, used for the reduced chi^2 statistic.
    """

    blob_id: int
    footprint: np.ndarray
    member_mer_indices: list[int]
    segments: dict[int, np.ndarray] = field(default_factory=dict)
    sep_rows: dict[int, dict] = field(default_factory=dict)


@dataclass
class StageRecord:
    """Per-tier snapshot for the demo figure (history=True only)."""

    stage: int
    model_name: str
    source_idx: int
    chi2_red: float
    model_image: np.ndarray
    residual_image: np.ndarray
    accepted: bool = False
    flux_ujy: float = float("nan")
    chi2_wide: float = float("nan")


@dataclass
class FinalRecord:
    """Final joint-fit snapshot, shown on the top row of the demo figure."""

    data: np.ndarray
    model: np.ndarray
    residual: np.ndarray
    footprint: np.ndarray


@dataclass
class History:
    blob_id: int
    members: list[int]
    stages: list[StageRecord] = field(default_factory=list)
    final: FinalRecord | None = None


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------
def _seed_shape_from_sep(sep_row: dict, pixscale_arcsec: float,
                         max_re_arcsec: float | None = None) -> EllipseESoft:
    """Build an initial Tractor shape from a SEP detection row.

    Mirrors Farmer's seeding; the radius cap defaults to
    ``EUCLID_DEFAULTS['max_re_arcsec']``.
    """
    a = max(float(sep_row["a"]), 0.5)
    b = max(float(sep_row["b"]), 0.5)
    theta = float(sep_row["theta"])

    r_eff = float(np.sqrt(a * b) * pixscale_arcsec)
    axis_ratio = float(b / a)
    pa_deg = 90.0 - float(np.rad2deg(theta))

    if max_re_arcsec is None:
        max_re_arcsec = EUCLID_DEFAULTS["max_re_arcsec"]
    # Per-source ceiling: without it, sources too faint to constrain a
    # shape rail at the global cap as razor-thin 20-arcsec needles. 5x the
    # SEP seed allows genuine growth; 2.72 arcsec stays the floor.
    per_source_cap = float(np.clip(5.0 * r_eff, 2.72, max_re_arcsec))
    shape = EllipseESoft.fromRAbPhi(r_eff, axis_ratio, pa_deg)
    shape.lowers = [-5.0, -np.inf, -np.inf]
    shape.uppers = [float(np.log(per_source_cap)), np.inf, np.inf]
    return shape


def _make_masked_image(tim: Image, footprint: np.ndarray, psf=None,
                       bbox=None) -> Image:
    """Clone a Tractor Image with invvar zeroed outside ``footprint``.

    The original ``tim`` is untouched. When ``psf`` is given, the clone uses
    that PSF instead of ``tim``'s (lets the caller swap in a per-blob /
    per-source CATALOG-PSF stamp). WCS/photocal/sky are still shared.

    ``bbox`` (y0, y1, x0, x1) crops the clone to that pixel window with a
    correspondingly shifted WCS; the likelihood is identical (zero weight
    outside the footprint) but rendering scales with the window area.
    """
    # Clone the PSF: PixelizedPSF lazily caches an FFT on the instance, and
    # two worker threads racing on a cold cache can return half-initialized
    # arrays.
    import copy as _copy
    psf_obj = psf if psf is not None else _copy.deepcopy(tim.getPsf())
    if bbox is None:
        invvar = tim.getInvvar().copy()
        invvar[~footprint] = 0.0
        data = tim.getImage()
        wcs = tim.getWcs()
    else:
        y0, y1, x0, x1 = bbox
        invvar = tim.getInvvar()[y0:y1, x0:x1].copy()
        invvar[~footprint[y0:y1, x0:x1]] = 0.0
        data = tim.getImage()[y0:y1, x0:x1]
        wcs = tim.getWcs().shifted(x0, y0)
        psf_obj = psf_obj.getShifted(x0, y0)
    new_tim = Image(
        data=data,
        invvar=invvar,
        psf=psf_obj,
        wcs=wcs,
        photocal=tim.getPhotoCal(),
        sky=tim.getSky(),
        name=getattr(tim, "name", "vis") + "[blob]",
    )
    new_tim.freezeAllParams()
    return new_tim


def _blob_bbox(footprint: np.ndarray, shape, margin: int = 20):
    """Bounding box (y0, y1, x0, x1) of ``footprint`` padded by ``margin``."""
    ys, xs = np.nonzero(footprint)
    y0 = max(0, int(ys.min()) - margin)
    y1 = min(int(shape[0]), int(ys.max()) + 1 + margin)
    x0 = max(0, int(xs.min()) - margin)
    x1 = min(int(shape[1]), int(xs.max()) + 1 + margin)
    return y0, y1, x0, x1


def _count_free_params(source) -> int:
    """Number of currently-thawed Tractor parameters on a source.

    A static per-class count would overcount the dof by 1-2 in stages where
    pos is frozen, biasing the reduced chi^2.
    """
    try:
        n = int(source.numberOfParams())
    except Exception:
        n = len(source.getParams())
    return max(n, 1)


def _build_initial_source(mer_row, model_class, sep_row, pixscale_arcsec,
                          *, simplegalaxy_radius=None):
    """Construct an initial Tractor source for a given model class.

    Seeds brightness from the MER flux; the tree decides morphology.
    """
    pos = RaDecPos(float(mer_row["ra"]), float(mer_row["dec"]))
    flux_uJy = float(mer_row["flux_vis_sersic"])
    if not np.isfinite(flux_uJy) or flux_uJy <= 0:
        flux_uJy = 1.0
    brightness = NanoMaggies(**{"VIS": flux_uJy / UJY_PER_NMGY})

    if model_class is PointSource:
        return PointSource(pos, brightness)
    if model_class is SimpleGalaxy:
        return SimpleGalaxy(pos, brightness, radius_arcsec=simplegalaxy_radius)

    if sep_row is not None:
        shape = _seed_shape_from_sep(sep_row, pixscale_arcsec)
    else:
        shape = EllipseESoft.fromRAbPhi(0.3, 1.0, 0.0)
        # Same per-source ceiling rule as the SEP-seeded path.
        shape.lowers = [-5.0, -np.inf, -np.inf]
        shape.uppers = [1.0, np.inf, np.inf]
    if model_class is ExpGalaxy:
        return ExpGalaxy(pos, brightness, shape)
    if model_class is DevGalaxy:
        return DevGalaxy(pos, brightness, shape)
    if model_class is FixedCompositeGalaxy:
        return FixedCompositeGalaxy(pos, brightness, SoftenedFracDev(0.5), shape, shape)
    if model_class is SersicGalaxy:
        return SersicGalaxy(pos, brightness, shape,
                            SersicIndex(EUCLID_DEFAULTS["sersic_init_n"]))
    raise ValueError(f"Unsupported model class: {model_class}")


def _classname(src) -> str:
    if isinstance(src, PointSource):
        return "PointSource"
    if isinstance(src, SimpleGalaxy):
        return "SimpleGalaxy"
    if isinstance(src, FixedCompositeGalaxy):
        return "CompositeGalaxy"
    if isinstance(src, SersicGalaxy):
        return "SersicGalaxy"
    if isinstance(src, DevGalaxy):
        return "DevGalaxy"
    if isinstance(src, ExpGalaxy):
        return "ExpGalaxy"
    return type(src).__name__
