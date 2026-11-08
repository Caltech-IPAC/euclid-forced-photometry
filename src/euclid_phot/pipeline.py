"""One-call driver: ``run_forced_photometry(prior=..., target_bands=...)``.

Composes the per-step functions: Euclid data (``euclid``), the prior
models (``sources``, ``selection``), the shared forced-photometry recipe
(``forced``) and the target bands (``euclid.nisp``, ``wise``, ``irac``);
the same step functions are exposed individually for stage-by-stage use.

``prior`` describes where positions and shapes come from (``band``,
``objects`` = "mer" or "coords", ``model_selection`` = "prior" / "tree" /
an explicit Tractor class, ``free_shapes``); ``target_bands`` lists the
Euclid, WISE and IRAC bands to propagate to. The prior band is removed from
``target_bands["euclid"]`` if present. WISE forced photometry freezes each
source's VIS position and shape, fitting flux only, exactly as for NISP;
``source_models="point"`` instead collapses every source to a
``PointSource`` (Lang et al. 2016, Sect. 2; Schlafly et al. 2019,
Sect. 6.8), valid because every Euclid R_eff is unresolved at the WISE PSF.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .data_access import get_data_access, get_data_dir
from .euclid.catalog import query_mer_catalog
from .euclid.cutouts import discover_mer_mosaics, fetch_cutout
from .euclid.image import build_tractor_image
from .euclid.nisp import fit_nisp_forced
from .euclid.psf import extract_catalog_psf, extract_grid_psf, psf_summary
from .forced import (
    fit_forced_photometry,
    fit_free_shapes,
    fluxes_ujy,
    refine_positions,
)
from .scene import footprint_mask, trim_catalog_to_cutout
from .selection import run_model_selection, unlisted_sep_detections
from .sources import build_sources_from_coords, build_sources_from_mer
from .wise import fetch_unwise_cutouts, fit_wise_forced, get_wise_psf

_VALID_PRIOR_BANDS = {"VIS", "Y", "J", "H"}
_VALID_OBJECTS = {"mer", "free", "coords"}
_VALID_EUCLID_TARGETS = {"VIS", "Y", "J", "H"}
_VALID_WISE_TARGETS = {"W1", "W2"}
_VALID_IRAC_TARGETS = {"IRAC1", "IRAC2", "IRAC3", "IRAC4"}
_EXPLICIT_MODELS = ("point", "sersic", "exp", "dev")

_DEFAULT_PRIOR = {"band": "VIS", "objects": "mer", "model_selection": None,
                  "free_shapes": True, "refine_positions": False,
                  "detect": None, "selector": None}
_DEFAULT_TARGET_BANDS = {"euclid": ("Y", "J", "H"), "wise": (), "irac": ()}

# MER flux column per Euclid band, used to seed Tractor brightness.
_MER_FLUX_COL = {
    "VIS": "flux_vis_sersic",
    "Y":   "flux_y_sersic",
    "J":   "flux_j_sersic",
    "H":   "flux_h_sersic",
}


def _normalize_prior(prior: dict | None, *,
                     has_user_coords: bool = False) -> dict:
    p = dict(_DEFAULT_PRIOR)
    if prior is not None:
        unknown = set(prior) - {"band", "objects", "free_shapes",
                                "refine_positions", "detect", "selector",
                                "model_selection"}
        if unknown:
            raise ValueError(
                f"prior has unknown keys: {sorted(unknown)}. "
                f"Allowed: 'band', 'objects', 'model_selection', "
                f"'free_shapes', 'refine_positions', 'detect', 'selector'.")
        p.update(prior)
    # objects="free" is shorthand for the tree path.
    ms = p.get("model_selection")
    if ms not in (None, "prior", "tree") + _EXPLICIT_MODELS:
        raise ValueError(
            f"prior['model_selection'] = {ms!r}; must be 'tree', 'prior', "
            f"or one of {_EXPLICIT_MODELS} (a fixed Tractor class for "
            f"every source).")
    if ms == "tree":
        if p["objects"] == "coords" and not has_user_coords:
            raise ValueError(
                "prior['model_selection']='tree' with objects='coords' "
                "requires user_coords (the positions to fit).")
        if p["objects"] == "mer" and has_user_coords:
            raise ValueError(
                "user_coords is ignored with objects='mer'; use "
                "objects='coords' to fit at supplied positions.")
        p["objects"] = "free"
    elif ms in _EXPLICIT_MODELS and p["objects"] == "free":
        raise ValueError(
            f"prior['model_selection'] = {ms!r} requires objects='mer' or "
            f"'coords' (objects='free' is the tree shorthand).")
    if p["objects"] == "free":
        p["model_selection"] = "tree"
    elif ms in _EXPLICIT_MODELS:
        p["model_selection"] = ms
    else:
        p["model_selection"] = "prior"
    # detect/selector apply only to the tree path.
    for key in ("detect", "selector"):
        if p.get(key) is not None and p["objects"] != "free":
            raise ValueError(
                f"prior[{key!r}] only applies with "
                f"prior['model_selection']='tree' "
                f"(got objects={p['objects']!r}).")
    if p.get("detect") is not None and not isinstance(p["detect"], dict):
        raise ValueError(
            "prior['detect'] must be a dict of detect_blobs keyword "
            "arguments (e.g. {'threshold_sigma': 2.0, 'deblend_cont': 1e-4}).")
    if p["band"] not in _VALID_PRIOR_BANDS:
        raise ValueError(
            f"prior['band'] = {p['band']!r}; must be one of "
            f"{sorted(_VALID_PRIOR_BANDS)}.")
    if p["objects"] not in _VALID_OBJECTS:
        raise ValueError(
            f"prior['objects'] = {p['objects']!r}; must be one of "
            f"{sorted(_VALID_OBJECTS)}.")
    if p["objects"] == "free":
        # The tree fits each blob's shapes with the bounded optimizer and
        # freezes them (Weaver et al. 2023, sec 3.4.1); an unbounded re-thaw
        # can push fitted radii past the bounds.
        p["free_shapes"] = False
    if p["objects"] == "free" and p["band"] != "VIS":
        # The ladder builds NanoMaggies(VIS=...) throughout (selection.py),
        # so the tree path is VIS-only.
        raise ValueError(
            f"prior['objects']='free' is only supported with "
            f"prior['band']='VIS' (got {p['band']!r}); the model-selection tree is "
            f"VIS-specific. Use objects='mer' for a NISP-prior fit.")
    return p


def _normalize_target_bands(tb: dict | None, prior_band: str) -> dict:
    out = {key: tuple(value) for key, value in _DEFAULT_TARGET_BANDS.items()}
    if tb is not None:
        unknown = set(tb) - set(out)
        if unknown:
            raise ValueError(
                f"target_bands has unknown keys: {sorted(unknown)}. "
                f"Allowed: 'euclid', 'wise', 'irac'.")
        for key in out:
            if key in tb:
                out[key] = tuple(tb[key])
    out["irac"] = tuple(str(b).upper() for b in out["irac"])
    bad_e = set(out["euclid"]) - _VALID_EUCLID_TARGETS
    if bad_e:
        raise ValueError(
            f"target_bands['euclid'] has unknown bands: {sorted(bad_e)}. "
            f"Allowed: {sorted(_VALID_EUCLID_TARGETS)}.")
    bad_w = set(out["wise"]) - _VALID_WISE_TARGETS
    if bad_w:
        raise ValueError(
            f"target_bands['wise'] has unknown bands: {sorted(bad_w)}. "
            f"Allowed: {sorted(_VALID_WISE_TARGETS)}.")
    bad_i = set(out["irac"]) - _VALID_IRAC_TARGETS
    if bad_i:
        raise ValueError(
            f"target_bands['irac'] has unknown bands: {sorted(bad_i)}. "
            f"Allowed: {sorted(_VALID_IRAC_TARGETS)}.")
    out["euclid"] = tuple(b for b in out["euclid"] if b != prior_band)
    return out


def _select_psf_product(psf_product: str, user_positions: bool) -> str:
    """Resolve the ``psf_product`` choice to ``'catalog'`` or ``'grid'``.

    ``'auto'`` follows the source-list mode: CATALOG-PSF stamps exist
    only at MER source positions, so they are the natural product for a
    MER-derived source list; for user-supplied positions the GRID-PSF (a
    regular ~12 arcsec sampling of the PSF model, covering any position) is
    correct, since the nearest *catalog* stamp could be arbitrarily far away.
    """
    if psf_product not in ("auto", "catalog", "grid"):
        raise ValueError(
            f"psf_product = {psf_product!r}; must be 'auto', 'catalog' or "
            f"'grid'.")
    if psf_product != "auto":
        return psf_product
    return "grid" if user_positions else "catalog"


def _parse_user_coords(user_coords) -> dict:
    """Normalize user-supplied source coordinates into arrays.

    Accepts an astropy Table (columns ``ra``, ``dec`` and optionally
    ``flux_ujy``/``model``/``re_arcsec``/``axis_ratio``/``position_angle_deg``/
    ``sersic_n``), a dict of the same keys, or an ``(N, 2)`` array of
    ``(ra, dec)`` in degrees. Returns a dict with at least ``ra`` and ``dec``.
    """
    from astropy.table import Table
    if user_coords is None:
        raise ValueError(
            "prior['objects']='coords' requires user_coords "
            "(an (N,2) (ra,dec) array, a dict, or an astropy Table).")
    out: dict = {}
    alias = {
        "flux_guess_ujy": ("flux_ujy", "flux_guess_ujy", "flux"),
        "re_arcsec": ("re_arcsec", "re"),
        "axis_ratio": ("axis_ratio", "ab"),
        "position_angle_deg": ("position_angle_deg", "position_angle", "pa"),
        "sersic_n": ("sersic_n", "n"),
    }
    if isinstance(user_coords, Table):
        cols = user_coords.colnames
        if "ra" not in cols or "dec" not in cols:
            raise ValueError("user_coords Table needs 'ra' and 'dec' columns.")
        out["ra"] = np.asarray(user_coords["ra"], float)
        out["dec"] = np.asarray(user_coords["dec"], float)
        for key, names in alias.items():
            for nm in names:
                if nm in cols:
                    out[key] = np.asarray(user_coords[nm], float); break
        if "model" in cols:
            out["model"] = str(user_coords["model"][0])
    elif isinstance(user_coords, dict):
        if "ra" not in user_coords or "dec" not in user_coords:
            raise ValueError("user_coords dict needs 'ra' and 'dec' keys.")
        out["ra"] = np.atleast_1d(np.asarray(user_coords["ra"], float))
        out["dec"] = np.atleast_1d(np.asarray(user_coords["dec"], float))
        for key in (*alias, "model"):
            if key in user_coords:
                out[key] = user_coords[key]
    else:
        arr = np.atleast_2d(np.asarray(user_coords, float))
        if arr.shape[1] < 2:
            raise ValueError("user_coords array must be (N,2) of (ra, dec).")
        out["ra"] = arr[:, 0]; out["dec"] = arr[:, 1]
    if len(out["ra"]) != len(out["dec"]):
        raise ValueError("user_coords ra and dec must have equal length.")
    return out


class _LazyProducts:
    """A products mapping that runs the SIA discovery query only on first use.

    Consumers check their on-disk caches before touching ``products``, so a
    fully cached run makes zero network calls; a live run runs the discovery
    query once on the first cache miss and reuses the result afterwards.
    """

    def __init__(self, discover):
        import threading
        self._discover = discover
        self._products = None
        self._lock = threading.Lock()

    def _ensure(self) -> dict:
        # Run the SIA discovery query only once even if several worker
        # threads reach this together.
        if self._products is None:
            with self._lock:
                if self._products is None:
                    self._products = self._discover()
        return self._products

    def __contains__(self, key):
        return key in self._ensure()

    def __getitem__(self, key):
        return self._ensure()[key]

    def get(self, key, default=None):
        return self._ensure().get(key, default)


def _synthetic_mer_from_coords(parsed: dict, flux_col: str):
    """A minimal MER-like Table for the user-coords path so the rest of the
    pipeline (initial flux, catalog assembly) runs unchanged."""
    from astropy.table import Table
    n = len(parsed["ra"])
    tab = Table()
    tab["object_id"] = np.arange(n, dtype=np.int64)
    tab["ra"] = np.asarray(parsed["ra"], float)
    tab["dec"] = np.asarray(parsed["dec"], float)
    guess = parsed.get("flux_guess_ujy")
    tab[flux_col] = (np.full(n, 1.0) if guess is None
                     else np.broadcast_to(np.asarray(guess, float), (n,)).copy())
    return tab


def _drop_neighbour_rows(result, n_keep: int) -> None:
    """Move rows beyond ``n_keep`` (the SEP neighbours appended on the
    user-coordinates tree path) out of every per-source array and into
    ``result.neighbour_sources``."""
    n_total = len(result.sources)
    result.neighbour_sources = list(result.sources[n_keep:])
    result.neighbour_fluxes_ujy = {
        b: np.asarray(v)[n_keep:] for b, v in result.fluxes_ujy.items()}
    result.sources = list(result.sources[:n_keep])
    result.mer_cat = result.mer_cat[:n_keep]
    for d in (result.fluxes_ujy, result.flux_errs_ujy):
        for b in list(d):
            d[b] = np.asarray(d[b])[:n_keep]
    if result.flux_quality is not None:
        result.flux_quality = np.asarray(result.flux_quality)[:n_keep]
    if result.chosen_models is not None:
        result.chosen_models = list(result.chosen_models[:n_keep])
    for r in [*(result.wise_results or {}).values(),
              *(result.irac_results or {}).values(),
              *(result.ground_results or {}).values()]:
        for key in ("flux_ujy", "flux_err_ujy", "flux_err_stat_ujy",
                    "position_shift_arcsec"):
            if key in r and len(r[key]) == n_total:
                r[key] = np.asarray(r[key])[:n_keep]


def _drop_coords_outside_cutout(parsed: dict, wcs, shape, *, margin_pix: int = 1):
    """Drop user coordinates whose pixel position falls outside the cutout.

    Returns ``(filtered_parsed, n_dropped)``. Per-source array entries are
    filtered to the survivors; scalar entries (e.g. a single ``model`` string,
    or a broadcast single flux/shape value) are preserved unchanged.
    """
    ra = np.atleast_1d(np.asarray(parsed["ra"], float))
    n = len(ra)
    keep = footprint_mask(ra, parsed["dec"], wcs, shape, margin_pix=-margin_pix)
    n_dropped = int((~keep).sum())
    if n_dropped == 0:
        return parsed, 0
    out = {}
    for k, v in parsed.items():
        if isinstance(v, str) or np.ndim(v) == 0 or len(np.atleast_1d(v)) != n:
            out[k] = v
        else:
            out[k] = np.asarray(v)[keep]
    return out, n_dropped


@dataclass
class ForcedPhotometryResult:
    target: tuple                                  # (ra, dec, size_arcsec)
    prior: dict = field(default_factory=dict)
    target_bands: dict = field(default_factory=dict)
    mer_cat: object = None
    sources: list = field(default_factory=list)
    cutouts: dict = field(default_factory=dict)        # band -> Cutout
    psf_data: dict = field(default_factory=dict)
    psf_stamps: dict = field(default_factory=dict)
    psf_mode: str = 'field-average'
    fluxes_ujy: dict = field(default_factory=dict)        # band -> ndarray
    flux_errs_ujy: dict = field(default_factory=dict)     # band -> ndarray (1-sigma)
    flux_quality: np.ndarray | None = None
    chosen_models: list | None = None
    # band -> fit_nisp_forced output: fluxes as fitted, before the NaN of a
    # failed prior fit, aligned with sources + neighbour_sources; image_set
    # renders the NISP models from them.
    nisp_results: dict | None = None
    wise_results: dict | None = None
    irac_results: dict | None = None
    ground_results: dict | None = None
    # band -> info dict from calibrate.measure_error_inflation; the
    # inflation factors are already applied to flux_errs_ujy.
    error_calibration: dict = field(default_factory=dict)
    # Cache directory of the run (used by to_table for the E(B-V) fallback).
    data_dir: object = None
    release: str = field(default_factory=get_data_access)
    # True = pixel given zero weight in the prior-band fit
    # (mask_bright_stars).
    prior_pixel_mask: np.ndarray | None = None
    # User-coordinates tree path only: fitted sources for SEP detections
    # that share a blob with a supplied position. They stay in the model
    # of every band but are not reported.
    neighbour_sources: list = field(default_factory=list)
    neighbour_fluxes_ujy: dict = field(default_factory=dict)   # band -> ndarray

    def to_table(self, **kwargs):
        """Assemble the per-object science catalog (astropy Table).

        Builds the catalog via :func:`euclid_phot.catalog_table.build_catalog`;
        see that function for the column list and units.
        """
        from .catalog_table import build_catalog
        table = build_catalog(self, **kwargs)
        table.meta['psf_mode'] = self.psf_mode
        table.meta['DATAREL'] = self.release
        return table

    def image_set(self, band: str) -> dict:
        """Data, model, residual and invvar images and WCS for one band.

        Available for the prior band and every fitted NISP band (the model
        is rendered from the fitted sources), and for every fitted WISE,
        IRAC and ground band. Images are
        in microJansky per pixel and ``invvar`` in 1/(microJansky/pixel)^2.
        Ground images are on their own pixel grid.
        """
        from .viz import UJY_PER_PIXEL, ujy_per_unit
        band = str(band).upper() if str(band).upper().startswith(
            ("W", "IRAC")) else str(band)
        for results, is_wise in ((self.wise_results, True),
                                 (self.irac_results, False),
                                 (self.ground_results, False)):
            if results and band in results:
                r = results[band]
                f = ujy_per_unit(band=band) if is_wise else 1.0
                return {"data": r["data"] * f, "model": r["model"] * f,
                        "residual": r["residual"] * f,
                        "invvar": r["invvar"] / f**2,
                        "wcs": r["wcs"], "unit": UJY_PER_PIXEL}
        sources = list(self.sources) + list(self.neighbour_sources)
        if self.nisp_results and band in self.nisp_results:
            # The prior-band models with the fluxes fitted in this band.
            from .sources import clone_sources
            from .units import UJY_PER_NMGY
            fitted = np.asarray(self.nisp_results[band]["flux_ujy"], float)
            sources = clone_sources(sources, band,
                                    seed_nmgy=np.nan_to_num(fitted) / UJY_PER_NMGY)
        elif band != self.prior.get("band"):
            raise ValueError(
                f"no images for {band!r}; available: the prior band "
                f"{self.prior.get('band')!r} and the NISP/WISE/IRAC/ground bands fitted")
        from tractor import Tractor
        cutout = self.cutouts[band]
        persource = self.psf_mode == 'nearest-source-position'
        tim = build_tractor_image(
            cutout, self.psf_stamps.get(band),
            psf_data=self.psf_data.get(band) if persource else None,
            pixel_mask=self.prior_pixel_mask)
        f = ujy_per_unit(cutout)
        model = np.asarray(Tractor([tim], sources).getModelImage(0)) * f
        data = np.asarray(tim.getImage()) * f
        return {"data": data, "model": model, "residual": data - model,
                "invvar": np.asarray(tim.getInvvar()) / f**2, "wcs": cutout.wcs,
                "unit": UJY_PER_PIXEL}


@dataclass
class _Run:
    """Settings of one run_forced_photometry call, shared by its stages."""
    target_ra: object
    target_dec: object
    cutout_size_arcsec: object
    prior: object
    target_bands: object
    prior_band: object
    flux_col: object
    euclid_bands: object
    data_dir: object
    cutout_dir: object
    psf_dir: object
    wise_dir: object
    half_size_deg: object
    verbose: object
    force_download: object
    n_workers: object
    persource_psf: object
    with_flag: object
    mask_bright_stars: object
    psf_product: object


def _load_source_list(run, result, user_coords, mer_catalog):
    """Stage 1: the source list (MER catalog, a supplied catalog, or user positions)."""
    target_ra = run.target_ra
    target_dec = run.target_dec
    cutout_size_arcsec = run.cutout_size_arcsec
    prior = run.prior
    flux_col = run.flux_col
    data_dir = run.data_dir
    half_size_deg = run.half_size_deg
    verbose = run.verbose
    force_download = run.force_download
    if mer_catalog is not None and prior["objects"] == "coords":
        raise ValueError(
            "mer_catalog is ignored by prior['objects']='coords' (positions "
            "come from user_coords); pass objects='mer' or 'free' to "
            "measure a caller-supplied catalog.")
    parsed_coords = None
    if prior["objects"] == "coords" or (
            prior["objects"] == "free" and user_coords is not None):
        # "coords" measures the supplied positions with fixed classes;
        # "free" + user_coords seeds the tree with them instead of a MER
        # query.
        parsed_coords = _parse_user_coords(user_coords)
        if verbose:
            print(f"[1/7] using {len(parsed_coords['ra'])} user-supplied source "
                  f"position(s) (no MER query)")
        result.mer_cat = _synthetic_mer_from_coords(parsed_coords, flux_col)
    elif user_coords is not None:
        raise ValueError(
            "user_coords requires prior['objects']='coords' (fixed models) "
            "or 'free' (model-selection tree at your positions); it is ignored by "
            f"objects={prior['objects']!r}, which would silently measure "
            "the MER catalog instead.")
    elif mer_catalog is not None:
        if verbose:
            print(f"[1/7] using caller-supplied MER catalog "
                  f"({len(mer_catalog)} rows; no query, no cache)")
        result.mer_cat = mer_catalog
    else:
        # Cache the MER catalog so a re-run skips the IRSA TAP query.
        mer_cache = data_dir / "catalogs" / (
            f"mer_catalog_{target_ra:.4f}_{target_dec:.4f}"
            f"_{int(round(cutout_size_arcsec))}.fits")
        if mer_cache.exists() and not force_download:
            from astropy.table import Table
            if verbose:
                print(f"[1/7] MER catalog from cache ({mer_cache.name})")
            result.mer_cat = Table.read(mer_cache)
        else:
            if verbose:
                print(f"[1/7] query MER catalog at ({target_ra:.4f}, {target_dec:.4f})")
            result.mer_cat = query_mer_catalog(target_ra, target_dec, half_size_deg)
            try:
                mer_cache.parent.mkdir(parents=True, exist_ok=True)
                tmp = mer_cache.with_name(mer_cache.stem + ".tmp.fits")
                result.mer_cat.write(tmp, overwrite=True)
                tmp.replace(mer_cache)
            except Exception as exc:
                import warnings
                warnings.warn(
                    f"could not cache the MER catalog to {mer_cache} "
                    f"({exc!r}); the run continues uncached.", stacklevel=2)
    return parsed_coords


def _load_cutouts(run, result, provided, parsed_coords):
    """Stages 2-3: MER mosaics (discovered lazily) and the band cutouts."""
    target_ra = run.target_ra
    target_dec = run.target_dec
    cutout_size_arcsec = run.cutout_size_arcsec
    target_bands = run.target_bands
    prior_band = run.prior_band
    flux_col = run.flux_col
    euclid_bands = run.euclid_bands
    cutout_dir = run.cutout_dir
    half_size_deg = run.half_size_deg
    verbose = run.verbose
    force_download = run.force_download
    n_workers = run.n_workers
    with_flag = run.with_flag
    mask_bright_stars = run.mask_bright_stars
    if verbose:
        print(f"[2/7] discover mosaics for {', '.join(euclid_bands)} "
              f"(lazy; only on a cache miss)")
    products = _LazyProducts(lambda: discover_mer_mosaics(
        target_ra, target_dec, half_size_deg, bands=tuple(set(euclid_bands))))

    provided_cutouts = dict(provided) if provided else {}
    unknown_provided = set(provided_cutouts) - set(euclid_bands)
    if unknown_provided:
        raise ValueError(
            f"cutouts= contains bands not used by this run: "
            f"{sorted(unknown_provided)} (run uses {list(euclid_bands)}).")
    if verbose:
        n_prov = len(provided_cutouts)
        print(f"[3/7] fetch cutouts: {', '.join(euclid_bands)}"
              + (f" ({n_prov} caller-supplied)" if n_prov else ""))
    cutouts = {}

    def _fetch_one(band):
        if band in provided_cutouts:
            return band, provided_cutouts[band]
        return band, fetch_cutout(
            band, target_ra, target_dec, cutout_size_arcsec,
            products=products, data_dir=cutout_dir,
            force_download=force_download,
            # mask_bright_stars needs the FLG plane (STARSIGNAL bit) on the
            # prior band; the NISP FLG planes feed the coadd-fatal bad-bit
            # veto inside build_tractor_image.
            with_flag=(with_flag and band == prior_band)
                      or (mask_bright_stars is True
                          and band in (prior_band,) + tuple(target_bands["euclid"])))

    if n_workers > 1 and len(euclid_bands) > 1:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(
                min(n_workers, len(euclid_bands))) as ex:
            for band, cut in ex.map(_fetch_one, euclid_bands):
                cutouts[band] = cut
    else:
        for band in euclid_bands:
            _, cutouts[band] = _fetch_one(band)
    result.cutouts = cutouts
    if parsed_coords is None:
        result.mer_cat = trim_catalog_to_cutout(
            result.mer_cat, cutouts[prior_band].wcs, cutouts[prior_band].shape)
    else:
        # An out-of-image position would make Tractor raise an IndexError
        # mid-optimization; drop them up front.
        parsed_coords, n_dropped = _drop_coords_outside_cutout(
            parsed_coords, cutouts[prior_band].wcs, cutouts[prior_band].shape)
        if n_dropped:
            import warnings
            warnings.warn(
                f"{n_dropped} user coordinate(s) fell outside the "
                f"{cutout_size_arcsec:g}\" cutout and were dropped; widen "
                "cutout_size_arcsec to include them.", stacklevel=2)
            result.mer_cat = _synthetic_mer_from_coords(parsed_coords, flux_col)
        if len(parsed_coords["ra"]) == 0:
            raise ValueError(
                "all user coordinates fall outside the cutout; nothing to fit. "
                "Check the positions or widen cutout_size_arcsec.")
    return products, parsed_coords


def _load_psfs(run, result, products, parsed_coords):
    """Stage 4: the MER PSF samples for every Euclid band."""
    target_ra = run.target_ra
    target_dec = run.target_dec
    cutout_size_arcsec = run.cutout_size_arcsec
    euclid_bands = run.euclid_bands
    psf_dir = run.psf_dir
    verbose = run.verbose
    force_download = run.force_download
    n_workers = run.n_workers
    persource_psf = run.persource_psf
    psf_product = run.psf_product
    cutouts = result.cutouts
    chosen_product = _select_psf_product(psf_product, parsed_coords is not None)
    psf_radius = max(60.0, float(cutout_size_arcsec) * 0.75)
    if verbose:
        print(f"[4/7] build per-band PSFs ({chosen_product.upper()}-PSF, "
              f"radius={psf_radius:.0f}\")")
    _extractors = {"catalog": extract_catalog_psf, "grid": extract_grid_psf}
    _other = {"catalog": "grid", "grid": "catalog"}

    def _extract_psf(band):
        kw = dict(products=products, radius_arcsec=psf_radius,
                  tile_id=cutouts[band].header.get("MERTILE"),
                  data_dir=psf_dir, force_download=force_download)
        try:
            pd = _extractors[chosen_product](band, target_ra, target_dec, **kw)
        except Exception as exc:           # product missing / fetch failed
            if psf_product != "auto":
                raise
            pd = None
            reason = repr(exc)
        else:
            reason = "no stamps near the target"
        if pd is None or len(pd.get("stamps", ())) == 0:
            if psf_product != "auto":
                raise ValueError(f"No {chosen_product.upper()}-PSF stamps for {band}")
            import warnings
            fallback = _other[chosen_product]
            warnings.warn(
                f"{chosen_product.upper()}-PSF unavailable for band "
                f"{band!r} ({reason}); falling back to "
                f"{fallback.upper()}-PSF.", stacklevel=2)
            pd = _extractors[fallback](band, target_ra, target_dec, **kw)
        return pd

    if n_workers > 1 and len(euclid_bands) > 1:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(
                min(n_workers, len(euclid_bands))) as ex:
            for band, pd in zip(euclid_bands,
                                ex.map(_extract_psf, euclid_bands),
                                strict=True):
                result.psf_data[band] = pd
    else:
        for band in euclid_bands:
            result.psf_data[band] = _extract_psf(band)
    if not persource_psf:
        for band in euclid_bands:
            avg, _ = psf_summary(result.psf_data[band])
            result.psf_stamps[band] = avg


def _mask_bright_stars(run, result, parsed_coords):
    """Stage 5a: the prior-band bright-star pixel veto (shared by NISP)."""
    target_bands = run.target_bands
    prior_band = run.prior_band
    flux_col = run.flux_col
    verbose = run.verbose
    mask_bright_stars = run.mask_bright_stars
    cutouts = result.cutouts
    if mask_bright_stars is True:
        flag_plane = getattr(cutouts[prior_band], "flag", None)
        if flag_plane is None:
            import warnings
            warnings.warn(
                "mask_bright_stars=True needs the MER FLG plane and none "
                "is available for the prior band; skipping the bright-star "
                "pixel mask. Pass a dict of bright_star_pixel_mask keywords "
                "to use the geometric fallback instead.", stacklevel=2)
        else:
            from .euclid.flags import starsignal_pixel_mask
            star_mask = starsignal_pixel_mask(cutouts[prior_band])
            result.prior_pixel_mask = star_mask
            if verbose:
                pct = 100.0 * star_mask.mean()
                print(f"[5a] STARSIGNAL bright-star mask: "
                      f"{int(star_mask.sum())} pixels ({pct:.2f}%) excluded "
                      f"from the {prior_band} fit")
    elif mask_bright_stars:
        # Geometric fallback; needs the MER star classification, so a
        # user_coords run cannot use it.
        if (parsed_coords is not None
                or "is_star" not in getattr(result.mer_cat, "colnames", [])):
            import warnings
            warnings.warn(
                "mask_bright_stars needs the MER star classification; "
                "skipping the bright-star pixel mask for this "
                "user-coordinates run.", stacklevel=2)
        else:
            from .euclid.flags import bright_star_pixel_mask, probable_bright_stars
            mer = result.mer_cat
            # The classifier abstains on the brightest stars;
            # probable_bright_stars recovers them via the PSF/Sersic ratio.
            if {"point_like_prob", "flux_vis_psf",
                    "flux_vis_sersic"} <= set(mer.colnames):
                star_flag = probable_bright_stars(mer)
            else:
                star_flag = np.asarray(mer["is_star"], bool)
            # Rank stars by PSF flux when available.
            star_flux_col = ("flux_vis_psf"
                             if "flux_vis_psf" in mer.colnames
                             else flux_col)
            star_flux = mer[star_flux_col]
            star_flux = np.asarray(
                star_flux.filled(np.nan) if hasattr(star_flux, "filled")
                else star_flux, dtype=float)
            mask_kwargs = dict(mask_bright_stars) if isinstance(
                mask_bright_stars, dict) else {}
            star_mask = bright_star_pixel_mask(
                cutouts[prior_band].shape, cutouts[prior_band].wcs,
                np.asarray(mer["ra"], float), np.asarray(mer["dec"], float),
                star_flux, star_flag, **mask_kwargs)
            result.prior_pixel_mask = star_mask
            if verbose:
                pct = 100.0 * star_mask.mean()
                print(f"[5a] bright-star pixel mask: {int(star_mask.sum())} "
                      f"pixels ({pct:.2f}%) excluded from the {prior_band} fit")

    if result.prior_pixel_mask is not None:
        reference = cutouts[prior_band]
        for band in target_bands["euclid"]:
            target = cutouts[band]
            if (target.shape != reference.shape or not reference.wcs.wcs.compare(
                    target.wcs.wcs, tolerance=1e-10)):
                raise ValueError(
                    "A shared bright-star mask requires aligned native band grids. "
                    "Fit these tiles separately or construct band-specific masks.")


def _build_prior_models(run, result, parsed_coords):
    """Stage 5: the prior-band Tractor image and the source models."""
    prior = run.prior
    prior_band = run.prior_band
    flux_col = run.flux_col
    verbose = run.verbose
    n_workers = run.n_workers
    persource_psf = run.persource_psf
    cutouts = result.cutouts
    psf_fwhm_prior = float(np.median(result.psf_data[prior_band]["fwhm"]))

    tim_prior = build_tractor_image(
        cutouts[prior_band], result.psf_stamps.get(prior_band),
        pixel_mask=result.prior_pixel_mask,
        psf_data=result.psf_data[prior_band] if persource_psf else None)

    n_report = None
    if prior["objects"] == "free":
        detect_kw = dict(prior.get("detect") or {})
        neighbour_kw = {k: detect_kw.pop(k) for k in
                        ("min_snr", "min_separation_pix", "mask_margin_pix")
                        if k in detect_kw}
        if parsed_coords is not None:
            n_report = len(result.mer_cat)
            extra = unlisted_sep_detections(
                result.mer_cat, tim_prior, cutouts[prior_band].data,
                tim_prior.getInvvar(), **neighbour_kw, **detect_kw)
            if len(extra["ra"]):
                from astropy.table import Table, vstack
                mag_zero = float(cutouts[prior_band].header.get("MAGZERO", 23.9))
                rows = Table()
                rows["object_id"] = n_report + np.arange(len(extra["ra"]), dtype=np.int64)
                rows["ra"] = extra["ra"]
                rows["dec"] = extra["dec"]
                rows[flux_col] = np.clip(
                    extra["flux"] / 10 ** (0.4 * (mag_zero - 23.9)), 0.01, None)
                result.mer_cat = vstack([result.mer_cat, rows],
                                        metadata_conflicts="silent")
                if verbose:
                    print(f"[5b] {len(rows)} SEP detection(s) share a blob with the "
                          f"supplied positions: modeled as neighbours, not reported")
        sources_list, _ = run_model_selection(
            result.mer_cat, tim_prior,
            cutouts[prior_band].data, tim_prior.getInvvar(),
            pixscale_arcsec=cutouts[prior_band].pixel_scale_arcsec,
            selector=prior.get("selector"),
            # psf_data enables per_source_psf selectors; the default keeps
            # the field-average stamp.
            psf_data=result.psf_data.get(prior_band),
            n_workers=n_workers,
            **detect_kw)
        # Rows the ladder could not place (no blob overlap) fall back to a
        # fixed-model source so the list stays 1:1 with mer_cat: MER-prior,
        # or a PointSource at the supplied position for user_coords.
        if parsed_coords is not None:
            fallback = build_sources_from_coords(
                parsed_coords["ra"], parsed_coords["dec"], band=prior_band,
                flux_guess_ujy=parsed_coords.get("flux_guess_ujy"))
            if len(result.mer_cat) > n_report:
                extra_rows = result.mer_cat[n_report:]
                fallback += build_sources_from_coords(
                    np.asarray(extra_rows["ra"], float),
                    np.asarray(extra_rows["dec"], float), band=prior_band,
                    flux_guess_ujy=np.asarray(extra_rows[flux_col], float))
        else:
            fallback = build_sources_from_mer(
                result.mer_cat, band=prior_band,
                psf_fwhm_arcsec=psf_fwhm_prior, flux_col=flux_col,
                pixel_scale_arcsec=cutouts[prior_band].pixel_scale_arcsec)
        result.sources = [
            s if s is not None else fb
            for s, fb in zip(sources_list, fallback, strict=False)]
    elif prior["objects"] == "coords":
        _ms = prior["model_selection"]
        if _ms in _EXPLICIT_MODELS:
            _model = _ms
            _re = parsed_coords.get("re_arcsec")
            if _model != "point" and _re is None:
                _re = 0.3   # default seed; step 2 adjusts it
        else:
            _model = parsed_coords.get("model", "point")
            _re = parsed_coords.get("re_arcsec")
        result.sources = build_sources_from_coords(
            parsed_coords["ra"], parsed_coords["dec"], band=prior_band,
            flux_guess_ujy=parsed_coords.get("flux_guess_ujy"),
            model=_model,
            re_arcsec=_re,
            axis_ratio=parsed_coords.get("axis_ratio"),
            position_angle_deg=parsed_coords.get("position_angle_deg"),
            sersic_n=parsed_coords.get("sersic_n"))
    else:
        _ms = prior["model_selection"]
        result.sources = build_sources_from_mer(
            result.mer_cat, band=prior_band,
            psf_fwhm_arcsec=psf_fwhm_prior, flux_col=flux_col,
            pixel_scale_arcsec=cutouts[prior_band].pixel_scale_arcsec,
            force_model=_ms if _ms in _EXPLICIT_MODELS else None)
    result.chosen_models = [type(s).__name__ for s in result.sources]
    return tim_prior, n_report


def _fit_prior(run, result, tim_prior, parsed_coords):
    """Stage 6: fluxes (and optionally shapes, positions) on the prior band."""
    prior = run.prior
    prior_band = run.prior_band
    flux_col = run.flux_col
    verbose = run.verbose
    persource_psf = run.persource_psf
    if verbose:
        objects = prior["objects"]
        action = ("forced-photometry" if not prior["free_shapes"]
                  else "forced photometry + free shapes")
        print(f"[6/7] {prior_band} fit ({objects}, {action})")
    initial_flux = np.asarray(result.mer_cat[flux_col], dtype=float)
    if parsed_coords is not None and (
            parsed_coords.get("flux_guess_ujy") is None):
        # No reference flux for user coords: a placeholder 1 uJy would
        # reject every source brighter than 100 uJy.
        initial_flux = np.full(len(initial_flux), np.nan)
    tractor_prior, fit_quality, prior_err_ujy = fit_forced_photometry(
        tim_prior, result.sources, band=prior_band,
        initial_fluxes_ujy=initial_flux, return_errors=True)
    if prior["free_shapes"]:
        fit_quality, _ = fit_free_shapes(
            tractor_prior, tim_prior, result.sources, fit_quality,
            band=prior_band, initial_fluxes_ujy=initial_flux)
    if prior["refine_positions"]:
        if verbose:
            print(f"[6a] refine positions on {prior_band} (bounded recenter)")
        fit_quality, _, _ = refine_positions(
            tractor_prior, tim_prior, result.sources, fit_quality,
            band=prior_band)
    if prior["free_shapes"] or prior["refine_positions"]:
        tractor_prior, final_quality, prior_err_ujy = fit_forced_photometry(
            tim_prior, result.sources, band=prior_band,
            initial_fluxes_ujy=initial_flux, return_errors=True)
        fit_quality &= final_quality
    result.flux_quality = fit_quality

    if persource_psf:
        prior_ujy = fluxes_ujy(result.sources, prior_band)
        with np.errstate(invalid="ignore"):
            bad = prior_ujy < 0
            ref_ok = np.isfinite(initial_flux) & (initial_flux > 0)
            bad |= ref_ok & (np.abs(prior_ujy) > 100.0 * initial_flux)
        result.flux_quality = result.flux_quality & ~bad

    prior_ujy = fluxes_ujy(result.sources, prior_band)
    prior_ujy = np.where(result.flux_quality, prior_ujy, np.nan)
    result.fluxes_ujy[prior_band] = prior_ujy
    result.flux_errs_ujy[prior_band] = np.where(
        result.flux_quality, prior_err_ujy, np.nan)


def _fit_nisp(run, result):
    """Stage 7: the NISP target bands with the shared forced recipe."""
    target_bands = run.target_bands
    verbose = run.verbose
    n_workers = run.n_workers
    persource_psf = run.persource_psf
    cutouts = result.cutouts
    nisp_targets = tuple(target_bands["euclid"])
    if nisp_targets:
        if verbose:
            print(f"[7/7] NISP forced phot: {', '.join(nisp_targets)}")
        nisp = fit_nisp_forced(
            result.sources,
            {b: cutouts[b] for b in nisp_targets},
            {b: result.psf_stamps[b] for b in nisp_targets} if not persource_psf else None,
            bands=nisp_targets,
            psf_data=result.psf_data if persource_psf else None,
            # The NISP mosaics share the prior-band 0.1 arcsec grid, so the
            # prior fit's bright-star pixel veto carries over unchanged.
            pixel_mask=result.prior_pixel_mask,
            n_workers=n_workers,
        )
        result.nisp_results = nisp
        # A diverged prior fit means an unreliable position/shape; NaN the
        # propagated flux.
        for b, d in nisp.items():
            result.fluxes_ujy[b] = np.where(
                result.flux_quality, d["flux_ujy"], np.nan)
            result.flux_errs_ujy[b] = np.where(
                result.flux_quality, d["flux_err_ujy"], np.nan)


def _fit_wise(run, result):
    """unWISE W1/W2 with the shared forced recipe."""
    target_ra = run.target_ra
    target_dec = run.target_dec
    cutout_size_arcsec = run.cutout_size_arcsec
    target_bands = run.target_bands
    prior_band = run.prior_band
    wise_dir = run.wise_dir
    verbose = run.verbose
    force_download = run.force_download
    wise_targets = tuple(target_bands["wise"])
    if wise_targets:
        if verbose:
            print(f"[+] WISE forced phot: {', '.join(wise_targets)}")
        wise_size = max(180.0, float(cutout_size_arcsec) + 120.0)
        wise_cutouts = fetch_unwise_cutouts(
            target_ra, target_dec, wise_size, data_dir=wise_dir,
            force_download=force_download)
        wise_psfs = {
            "W1": get_wise_psf(1, wise_cutouts["coadd_id"], ra=target_ra, dec=target_dec),
            "W2": get_wise_psf(2, wise_cutouts["coadd_id"], ra=target_ra, dec=target_dec),
        }
        result.wise_results = fit_wise_forced(
            result.sources, wise_cutouts,
            ra=target_ra, dec=target_dec,
            cutout_size_arcsec=cutout_size_arcsec,
            psf_stamps=wise_psfs, bands=wise_targets,
            prior_cutout=result.cutouts[prior_band],
            catalog_cache_dir=wise_dir / "catalogs")
        for band in wise_targets:
            if band in result.wise_results:
                wflux = np.asarray(result.wise_results[band]["flux_ujy"], dtype=float)
                werr = np.asarray(result.wise_results[band]["flux_err_ujy"], dtype=float)
                # As for NISP: NaN where the prior fit diverged.
                result.fluxes_ujy[band] = np.where(
                    result.flux_quality, wflux, np.nan)
                result.flux_errs_ujy[band] = np.where(
                    result.flux_quality, werr, np.nan)


def _fit_irac(run, result, irac_options):
    """Spitzer/IRAC with the shared forced recipe."""
    target_ra = run.target_ra
    target_dec = run.target_dec
    cutout_size_arcsec = run.cutout_size_arcsec
    target_bands = run.target_bands
    data_dir = run.data_dir
    verbose = run.verbose
    force_download = run.force_download
    irac_targets = tuple(target_bands["irac"])
    if irac_targets:
        from .irac.fit import fit_irac_forced
        if verbose:
            print(f"[+] IRAC forced phot: {', '.join(irac_targets)}")
        options = {"mosaic_backend": "seip", **(irac_options or {})}
        result.irac_results = fit_irac_forced(
            result.sources, target_ra, target_dec, cutout_size_arcsec,
            irac_targets, data_dir=data_dir / "irac",
            force_download=force_download, verbose=verbose, **options)
        for band, r in result.irac_results.items():
            # As for NISP and WISE: NaN where the prior fit diverged.
            result.fluxes_ujy[band] = np.where(
                result.flux_quality, r["flux_ujy"], np.nan)
            result.flux_errs_ujy[band] = np.where(
                result.flux_quality, r["flux_err_ujy"], np.nan)


def _fit_ground(run, result, ground_images, ground_options):
    """User ground images with the shared forced recipe (``ground_based``)."""
    import warnings
    from dataclasses import replace

    from .ground_based import assess_psf, fit_ground, register_image
    from .ground_based.image import GroundInputs

    options = dict(ground_options or {})
    register = options.pop("register_wcs", True)
    min_matches = options.pop("min_matches", 10)
    taken = {run.prior_band, *run.target_bands["euclid"], *run.target_bands["wise"],
             *run.target_bands["irac"]}
    clash = taken & set(ground_images)
    if clash:
        raise ValueError(f"ground band names {sorted(clash)} are already used; "
                         "name each ground image after its filter, e.g. 'g'")
    n = len(result.sources)
    row_of = {object_id: i for i, object_id in enumerate(result.mer_cat["object_id"])}
    prior_cut = result.cutouts[run.prior_band]
    result.ground_results = {}
    for band, inputs in ground_images.items():
        image = inputs.image if isinstance(inputs, GroundInputs) else inputs
        image = replace(image, band=band)
        if run.verbose:
            print(f"[+] ground forced phot: {band}")
        ny, nx = image.science_ujy.shape
        corner_ra, corner_dec = image.wcs.pixel_to_world_values(
            [0, nx - 1, 0, nx - 1], [0, 0, ny - 1, ny - 1])
        if not footprint_mask(corner_ra, corner_dec, prior_cut.wcs,
                              prior_cut.shape).all():
            warnings.warn(
                f"the {band} image extends beyond the {run.cutout_size_arcsec:g}\" "
                "Euclid field; sources outside the field are not modelled. "
                "Enlarge cutout_size_arcsec to cover the image.", stacklevel=3)
        registration = None
        if register:
            registration = register_image(image, result.mer_cat,
                                          min_matches=min_matches)
            image = replace(image, wcs=registration.wcs)
        fit_options = dict(options)
        if (fit_options.get("psf_mode", "supplied") != "supplied"
                and "psf_assessment" not in fit_options):
            if registration is None:
                raise ValueError("a PSF from image stars needs register_wcs=True")
            fit_options["psf_assessment"] = assess_psf(image, registration)
        fit = fit_ground(image, result, band=band, **fit_options)
        rows = np.array([row_of[i] for i in fit.catalog["object_id"]], dtype=int)
        flux = np.full(n, np.nan)
        err = np.full(n, np.nan)
        flux[rows] = np.asarray(fit.catalog[f"flux_{band}_ujy"], float)
        err[rows] = np.asarray(fit.catalog[f"flux_err_{band}_ujy"], float)
        result.ground_results[band] = {
            "flux_ujy": flux, "flux_err_ujy": err,
            "data": image.science_ujy, "model": fit.model,
            "residual": fit.residual, "invvar": image.invvar_ujy,
            "wcs": image.wcs, "catalog": fit.catalog,
            "registration": registration,
        }
        # As for every band: NaN where the prior fit diverged.
        result.fluxes_ujy[band] = np.where(result.flux_quality, flux, np.nan)
        result.flux_errs_ujy[band] = np.where(result.flux_quality, err, np.nan)


def run_forced_photometry(
    target_ra: float,
    target_dec: float,
    cutout_size_arcsec: float = 50.0,
    *,
    prior: dict | None = None,
    target_bands: dict | None = None,
    user_coords=None,
    data_dir: str | Path | None = None,
    force_download: bool = False,
    verbose: bool = True,
    n_workers: int = 1,
    persource_psf: bool = True,
    with_flag: bool = False,
    mask_bright_stars=False,
    calibrate_errors: bool = True,
    psf_product: str = "auto",
    cutouts: dict | None = None,
    mer_catalog=None,
    irac_options: dict | None = None,
    ground_images: dict | None = None,
    ground_options: dict | None = None,
) -> ForcedPhotometryResult:
    """Forced photometry across Euclid VIS / NISP and optionally unWISE, IRAC
    and your own ground-based images.

    Parameters
    ----------
    target_ra, target_dec : float
        Cutout center in degrees.
    cutout_size_arcsec : float
        Side of the square cutout, in arcsec.
    prior : dict, optional
        Source-list and prior-fit configuration. Two independent choices:

        * ``objects``: where the positions come from. ``"mer"`` is
          the MER catalog (positions and Sersic shapes); ``"coords"`` is
          the positions in ``user_coords``.
        * ``model_selection``: how each source's model class is chosen.
          ``"prior"`` (default): assert the catalog / user class;
          ``"tree"``: the chi-squared decision ladder picks the simplest
          class per blob that satisfies the fit, with the source list
          kept one-to-one with the input (positions seeded at the input
          coordinates, clamped to 0.3 arcsec; a position with no
          detectable blob falls back to its prior model, or to a fixed
          point source for user coordinates); or an explicit
          Tractor class for every source (``"point"``, ``"sersic"``,
          ``"exp"``, or ``"dev"``), seeded from the catalog shape where
          available and refined in step 2.

        ``objects="free"`` is shorthand for ``model_selection="tree"``.

        Other keys: ``band`` (one of VIS / Y / J / H), ``free_shapes``
        (bool; forced off on the tree path), ``refine_positions``
        (bool, default False; an opt-in bounded centroid recenter on
        the prior band, propagated to every target band). With
        ``model_selection="tree"`` two more keys tune the detection and
        the ladder: ``detect`` (dict of
        :func:`euclid_phot.selection.detect_blobs` keywords, e.g.
        ``{"threshold_sigma": 2.0, "minarea": 8, "deblend_cont": 1e-4,
        "bkg_box": 32}``; with user coordinates also ``min_snr``,
        ``min_separation_pix`` and ``mask_margin_pix`` of
        :func:`euclid_phot.selection.unlisted_sep_detections`) and
        ``selector`` (a configured
        :class:`euclid_phot.selection.ModelSelector`, for the
        chi-squared thresholds). Default ``{"band": "VIS",
        "objects": "mer", "model_selection": "prior",
        "free_shapes": True, "refine_positions": False}``.
    user_coords : optional
        Required when ``prior['objects']='coords'``. An ``(N, 2)`` array of
        ``(ra, dec)`` in degrees, a dict, or an astropy Table with ``ra``,
        ``dec`` columns (optionally ``flux_ujy``, ``model``, ``re_arcsec``,
        ``axis_ratio``, ``position_angle_deg``, ``sersic_n``). The ``model``
        entry is read once and applied uniformly to every position (point vs
        sersic for the whole list); per-source shape seeds vary per row. Any
        position that falls outside the fetched cutout is dropped with a
        warning rather than measured. With ``model_selection="tree"``, SEP
        detections that share a blob with a supplied position but match
        none of them are added as neighbours when they are at least 5 sigma
        in their segment and clear of the masked pixels (see
        :func:`euclid_phot.selection.unlisted_sep_detections`). They are
        modeled in every band so their light is not absorbed by the listed
        sources, then removed from the result, so the output stays one row
        per supplied position. Their fitted sources and per-band fluxes are
        kept in ``result.neighbour_sources`` and
        ``result.neighbour_fluxes_ujy``.
    target_bands : dict, optional
        Forced-photometry targets. Keys: ``euclid`` (tuple of Euclid
        band names), ``wise`` (tuple of ``"W1"``/``"W2"`` or empty),
        ``irac`` (tuple of ``"IRAC1"`` ... ``"IRAC4"`` or empty).
        Default ``{"euclid": ("Y", "J", "H"), "wise": (), "irac": ()}``.
        The prior band is automatically removed from ``euclid`` if present.
    data_dir : Path
        Local cache directory for cutouts / PSF stamps / unWISE tiles.
    force_download : bool
        Ignore the cache and re-fetch live from IRSA + S3 + unwise.me.
    verbose : bool
        Print step-by-step progress.
    n_workers : int
        Worker threads for cutout fetching, per-blob tree fits, per-source
        PSF groups, and NISP per-band fits. The fit-step speedup depends on
        your Tractor build, since its compiled FFT may or may not run on
        several threads at once.
    with_flag : bool
        Also fetch the MER FLG (quality) plane for the prior band and zero
        the inverse variance on coadd-fatal pixels (instrument-specific bits)
        before fitting. Off by default: the RMS map already zeroes most of
        these pixels on Q1.
    mask_bright_stars : bool or dict
        ``True`` (VIS prior only): zero the inverse variance on the MER STARSIGNAL pixels
        (star footprints, halos and spikes included) before the prior-band
        fit; the NISP mosaics share the prior band's 0.1 arcsec grid, so
        the same pixel veto carries over to the NISP fits, whose own FLG
        planes additionally veto coadd-fatal pixels.
        Fetches the FLG planes automatically and works for every
        source-list mode; the covered stars are themselves not measurable
        and carry the ``bright_star`` flag. A dict of
        :func:`euclid_phot.euclid.flags.bright_star_pixel_mask` keywords selects
        the geometric fallback for data without the FLG plane. The applied
        mask is stored as ``result.prior_pixel_mask``.
    persource_psf : bool
        Use the nearest local PSF sample at each source centroid throughout
        the Euclid fits, including shapes and residual rendering (default
        True). This is one consistent scene, with no repeated PSF-group
        refits. False uses a field-average PSF in each Euclid band.
    calibrate_errors : bool
        After the Euclid-band fits, measure the empirical PSF-scale error
        inflation per band (see :mod:`euclid_phot.calibrate`) and scale
        ``flux_errs_ujy`` by it (default True). Factors are recorded in
        ``result.error_calibration`` and the output table metadata.
    psf_product : {'auto', 'catalog', 'grid'}
        Which MER PSF product supplies the stamps. ``'catalog'`` uses
        CATALOG-PSF (one stamp per MER source); ``'grid'`` uses GRID-PSF
        (the PSF model on a regular ~12 arcsec grid, covering any sky
        position). ``'auto'`` (default) picks CATALOG-PSF for a MER source
        list and GRID-PSF for user-supplied positions. If the chosen
        product is unavailable for a band, the other is used with a warning.
    cutouts : dict, optional
        Caller-supplied ``{band: Cutout}`` overrides; bands present skip
        the fetch entirely. Used by :mod:`euclid_phot.injection` to run the
        full pipeline on modified pixels.
    mer_catalog : astropy.table.Table, optional
        Caller-supplied MER catalog override; skips the TAP query / cache
        read. Only meaningful for ``objects='mer'`` or ``'free'``
        (``'coords'`` raises).
    irac_options : dict, optional
        Keywords for :func:`euclid_phot.irac.fit_irac_forced`, e.g.
        ``{"mosaic_backend": "dawn"}`` (default ``"seip"``; ``"dawn"`` is
        deeper but covers only the Cosmic Dawn fields, ``"pbcd"`` uses
        single-observation mosaics). IRAC inputs are cached under
        ``data_dir / "irac"``.
    ground_images : dict, optional
        ``{band: image}`` of your own images, each the output of
        :func:`euclid_phot.ground_based.make_image` (science, uncertainty and
        PSF in microJansky). The band name labels the output columns, e.g.
        ``flux_g_ujy``. Choose the field (``target_ra``, ``target_dec``,
        ``cutout_size_arcsec``) to cover each image: sources outside it are
        not modelled. Each image's WCS is first refined on MER stars, which
        needs a MER source list.
    ground_options : dict, optional
        Keywords for :func:`euclid_phot.ground_based.fit_ground`
        (``psf_mode``, ``background_method``, ``recenter_width_pixels``, ...)
        plus ``register_wcs`` (default True) and ``min_matches`` (default 10)
        for the WCS refinement.

    Returns
    -------
    ForcedPhotometryResult
    """
    prior = _normalize_prior(prior, has_user_coords=user_coords is not None)
    target_bands = _normalize_target_bands(target_bands, prior["band"])
    if psf_product not in ("auto", "catalog", "grid"):
        raise ValueError(
            f"psf_product = {psf_product!r}; must be 'auto', 'catalog' or "
            f"'grid'.")
    prior_band = prior["band"]
    if mask_bright_stars is True and prior_band != "VIS":
        raise ValueError(
            "mask_bright_stars=True reads VIS STARSIGNAL footprints and "
            "requires a VIS prior. For a NISP prior, use False or a dict "
            "of geometric bright_star_pixel_mask options.")
    flux_col = _MER_FLUX_COL[prior_band]

    data_dir = get_data_dir() if data_dir is None else Path(data_dir)
    cutout_dir = data_dir / "cutouts"
    psf_dir = data_dir / "psf"
    wise_dir = data_dir / "wise"

    half_size_deg = cutout_size_arcsec / 2.0 / 3600.0
    euclid_bands = (prior_band,) + tuple(target_bands["euclid"])
    run = _Run(
        target_ra=target_ra, target_dec=target_dec,
        cutout_size_arcsec=cutout_size_arcsec, prior=prior,
        target_bands=target_bands, prior_band=prior_band, flux_col=flux_col,
        euclid_bands=euclid_bands, data_dir=data_dir, cutout_dir=cutout_dir,
        psf_dir=psf_dir, wise_dir=wise_dir, half_size_deg=half_size_deg,
        verbose=verbose, force_download=force_download, n_workers=n_workers,
        persource_psf=persource_psf, with_flag=with_flag,
        mask_bright_stars=mask_bright_stars, psf_product=psf_product)

    result = ForcedPhotometryResult(
        target=(target_ra, target_dec, cutout_size_arcsec),
        prior=dict(prior),
        target_bands=dict(target_bands),
        data_dir=data_dir)
    result.psf_mode = 'nearest-source-position' if persource_psf else 'field-average'

    parsed_coords = _load_source_list(run, result, user_coords, mer_catalog)
    products, parsed_coords = _load_cutouts(run, result, cutouts, parsed_coords)
    _load_psfs(run, result, products, parsed_coords)
    if verbose:
        print(f"[5/7] build Tractor image + source models on {prior_band}")
    _mask_bright_stars(run, result, parsed_coords)
    tim_prior, n_report = _build_prior_models(run, result, parsed_coords)
    _fit_prior(run, result, tim_prior, parsed_coords)
    _fit_nisp(run, result)
    if calibrate_errors:
        # Runs after NISP so one pass covers prior + target bands; WISE
        # errors already carry the chi-inflation.
        from .calibrate import calibrate_result_errors
        if verbose:
            print("[7a] calibrate flux errors (empty-position point-source fits)")
        calibrate_result_errors(result, verbose=verbose)

    _fit_wise(run, result)
    _fit_irac(run, result, irac_options)
    if ground_images:
        _fit_ground(run, result, ground_images, ground_options)

    if n_report is not None and len(result.sources) > n_report:
        _drop_neighbour_rows(result, n_report)

    return result
