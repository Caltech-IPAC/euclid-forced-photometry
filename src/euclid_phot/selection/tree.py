"""ModelSelector: the chi-squared decision tree, one blob at a time."""

from __future__ import annotations

import copy
import logging
import warnings

import numpy as np
from tractor import Image, NanoMaggies, PointSource, RaDecPos, Tractor
from tractor.constrained_optimizer import ConstrainedOptimizer
from tractor.ellipses import EllipseESoft
from tractor.galaxy import DevGalaxy, ExpGalaxy, FixedCompositeGalaxy
from tractor.sersic import SersicGalaxy, SersicIndex

from ..units import UJY_PER_NMGY
from .models import (
    _blob_bbox,
    _build_initial_source,
    _classname,
    _count_free_params,
    _make_masked_image,
    _seed_shape_from_sep,
    Blob,
    EUCLID_DEFAULTS,
    FinalRecord,
    History,
    set_simplegalaxy_radius,
    SimpleGalaxy,
    StageRecord,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ModelSelector: orchestrates the decision tree on one blob at a time.
# ---------------------------------------------------------------------------
class ModelSelector:
    """Run the chi^2 decision tree (Weaver et al. 2023) on Euclid VIS data.

    The defaults below are Euclid-tuned (see module docstring). Pass keyword
    overrides to compare against Farmer's published COSMOS-Web defaults.
    """

    def __init__(
        self,
        sufficient_thresh: float = EUCLID_DEFAULTS["sufficient_thresh"],
        simplegalaxy_penalty: float = EUCLID_DEFAULTS["simplegalaxy_penalty"],
        exp_dev_similar_thresh: float = EUCLID_DEFAULTS["exp_dev_similar_thresh"],
        simplegalaxy_radius: float = EUCLID_DEFAULTS["simplegalaxy_radius"],
        max_pos_shift_arcsec: float = EUCLID_DEFAULTS["max_pos_shift_arcsec"],
        max_steps: int = EUCLID_DEFAULTS["max_steps"],
        dlnp_crit: float = EUCLID_DEFAULTS["dlnp_crit"],
        full_ladder: bool = EUCLID_DEFAULTS["full_ladder"],
        full_ladder_min_snr: float = EUCLID_DEFAULTS["full_ladder_min_snr"],
        per_source_psf: bool = EUCLID_DEFAULTS["per_source_psf"],
        include_sersic_tier: bool = EUCLID_DEFAULTS["include_sersic_tier"],
        sersic_min_snr: float = EUCLID_DEFAULTS["sersic_min_snr"],
        sersic_dchi2_accept: float = EUCLID_DEFAULTS["sersic_dchi2_accept"],
    ):
        self.sufficient_thresh = float(sufficient_thresh)
        self.simplegalaxy_penalty = float(simplegalaxy_penalty)
        self.exp_dev_similar_thresh = float(exp_dev_similar_thresh)
        self.simplegalaxy_radius = float(simplegalaxy_radius)
        self.max_pos_shift_arcsec = float(max_pos_shift_arcsec)
        self.max_steps = int(max_steps)
        self.dlnp_crit = float(dlnp_crit)
        self.full_ladder = bool(full_ladder)
        self.full_ladder_min_snr = float(full_ladder_min_snr)
        self.per_source_psf = bool(per_source_psf)
        # 6th tier beyond the published ladder: a free-Sersic fit on bright
        # resolved sources whose accepted model leaves a core residual.
        self.include_sersic_tier = bool(include_sersic_tier)
        self.sersic_min_snr = float(sersic_min_snr)
        self.sersic_dchi2_accept = float(sersic_dchi2_accept)

        # Sync the shared SimpleGalaxy radius with this selector's value.
        set_simplegalaxy_radius(self.simplegalaxy_radius)

    # ---- inner machinery ------------------------------------------------
    def _optimize(self, tractor: Tractor) -> None:
        """Joint optimization of the active free parameters in ``tractor``.

        Failures are swallowed and logged, as in Farmer with
        IGNORE_FAILURES=True.
        """
        for _step in range(self.max_steps):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    dlnp, _, _ = tractor.optimize()
            except Exception as exc:
                logger.debug("optimize() raised %s; continuing.", exc)
                return
            if dlnp < self.dlnp_crit:
                return

    def _fit_stage(self, tim_blob, sources_by_idx, free_indices, segments_by_idx,
                   eval_segments=None):
        """Build a Tractor for the blob with the given source list and fit it.

        ``free_indices`` lists which source idx's have their morphology params
        thawed (the rest stay frozen at their locked-in model). ``segments``
        is a dict mapping source idx -> per-source segmentation mask, used for
        the per-source reduced chi^2.

        ``eval_segments`` (optional dict idx -> dilated mask) additionally
        scores those members on a wider window. Full-ladder mode uses this:
        the winning tier must be the one that damages the source's
        surroundings least, not just its own segment.

        Returns ``(chi2, chi2_wide, tractor)``; ``chi2_wide`` only holds the
        members present in ``eval_segments``.
        """
        all_sources = [sources_by_idx[i] for i in sorted(sources_by_idx)]
        idx_in_order = sorted(sources_by_idx)
        tractor = Tractor([tim_blob], all_sources, optimizer=ConstrainedOptimizer())

        for idx, src in zip(idx_in_order, all_sources, strict=False):
            if idx in free_indices:
                src.thawAllParams()
                src.freezeParam("pos")  # positions stay at MER for forced phot
                if isinstance(src, SimpleGalaxy):
                    # Preserve SimpleGalaxy's frozen circular shape.
                    src.freezeParam("shape")
            else:
                # Deviation from Farmer (which keeps locked sources' fluxes
                # variable at later tiers): freeze them entirely and recover
                # the joint solution in the final all-source re-fit.
                src.freezeAllParams()

        self._optimize(tractor)

        chi_img = tractor.getChiImage(imgi=0)

        def _rchi2(mask, n_free):
            npix = int(mask.sum())
            return float(np.sum(chi_img[mask] ** 2) / max(npix - n_free, 1))

        chi2 = {}
        chi2_wide = {}
        for idx in free_indices:
            nf = _count_free_params(sources_by_idx[idx])
            chi2[idx] = _rchi2(segments_by_idx[idx], nf)
            if eval_segments and idx in eval_segments:
                chi2_wide[idx] = _rchi2(eval_segments[idx], nf)
        return chi2, chi2_wide, tractor

    # ---- public entry point --------------------------------------------
    def fit_blob(
        self,
        blob: Blob,
        tim_vis: Image,
        mer_cat,
        pixscale_arcsec: float,
        history: bool = False,
        psf_stamp: np.ndarray | None = None,
        psf_by_member: dict | None = None,
    ):
        """Walk the decision tree for every source in ``blob``.

        Returns ``(sources_dict, history_or_None)`` where ``sources_dict`` maps
        each MER row index to its chosen, fit Tractor source. ``history`` is a
        :class:`History` instance when ``history=True``, otherwise ``None``.

        ``psf_stamp`` (HxW float array) overrides ``tim_vis``'s PSF for this
        blob. This is an explicit constant-PSF approximation for all its
        members; omit it to retain the image's spatial PSF.

        ``psf_by_member`` (dict[mer_idx -> stamp]) supplies local samples
        for a spatial PSF. Each member uses its own nearest sample while
        all members remain in the joint fit. ``psf_stamp`` explicitly
        overrides this with a constant PSF when supplied.
        """
        member_psf = None
        if psf_by_member is not None and psf_stamp is None:
            from .pixel_psf import SpatialPixelizedPSF
            members = list(blob.member_mer_indices)
            member_psf = SpatialPixelizedPSF({
                'stamps': np.stack([psf_by_member[i] for i in members]),
                'ra': np.array([float(mer_cat[i]['ra']) for i in members]),
                'dec': np.array([float(mer_cat[i]['dec']) for i in members]),
            }, tim_vis.getWcs())

        # Fit on the blob bounding box (identical likelihood, much cheaper
        # rendering); the history path keeps the full frame because
        # reproduce_figure3 draws in full-image coordinates.
        if history:
            bbox = None
            segments = blob.segments
        else:
            bbox = _blob_bbox(blob.footprint, tim_vis.shape)
            y0, y1, x0, x1 = bbox
            segments = {k: v[y0:y1, x0:x1] for k, v in blob.segments.items()}
        if psf_stamp is not None:
            from tractor.psf import PixelizedPSF
            blob_psf = PixelizedPSF(np.asarray(psf_stamp, dtype=np.float32))
            tim_blob = _make_masked_image(tim_vis, blob.footprint,
                                          psf=blob_psf, bbox=bbox)
        else:
            tim_blob = _make_masked_image(tim_vis, blob.footprint,
                                          psf=member_psf, bbox=bbox)
        idx_list = list(blob.member_mer_indices)
        hist = History(blob_id=blob.blob_id, members=idx_list) if history else None

        # Full-ladder deferral set: these members skip the early sufficiency
        # locks so every tier is evaluated.
        defer = set()
        wide_segments = {}
        if self.full_ladder:
            from scipy import ndimage as _ndi
            for _i in idx_list:
                if (_i in segments and
                        self._segment_snr(tim_blob, segments[_i])
                        >= self.full_ladder_min_snr):
                    defer.add(_i)
                    # Window = segment grown ~2x its radius (10-40 px): wing
                    # damage counts, sky does not drown the signal.
                    _npix = int(segments[_i].sum())
                    _grow = int(np.clip(2.0 * np.sqrt(_npix / np.pi), 10, 40))
                    wide_segments[_i] = (
                        _ndi.distance_transform_edt(~segments[_i]) <= _grow)

        chi2_tracker: dict[int, dict[int, float]] = {s: {} for s in range(1, 6)}
        solved: dict[int, bool] = {i: False for i in idx_list}
        locked_source: dict[int, object] = {}

        sources_by_idx: dict[int, object] = {}

        # --- Stage 1: SimpleGalaxy for all members --------------------------
        for idx in idx_list:
            row = mer_cat[idx]
            sep_row = blob.sep_rows.get(idx)
            sources_by_idx[idx] = _build_initial_source(
                row, SimpleGalaxy, sep_row, pixscale_arcsec,
                simplegalaxy_radius=self.simplegalaxy_radius)

        chi2_stage1, chi2w_stage1, t1 = self._fit_stage(
            tim_blob, sources_by_idx, idx_list, segments,
            eval_segments=wide_segments)
        chi2_tracker[1] = chi2_stage1
        chi2w_tracker = {1: chi2w_stage1}
        if history:
            mod_full = t1.getModelImage(0)
            for idx in idx_list:
                hist.stages.append(self._record(1, sources_by_idx[idx], idx,
                                                chi2_stage1[idx], tim_blob, mod_full,
                                                chi2_wide=chi2w_stage1.get(idx, float("nan"))))

        # --- Stage 2: PointSource vs SimpleGalaxy ---------------------------
        for idx in idx_list:
            row = mer_cat[idx]
            sources_by_idx[idx] = _build_initial_source(row, PointSource, None, pixscale_arcsec)
        unsolved = [i for i in idx_list if not solved[i]]
        chi2_stage2, chi2w_tracker[2], t2 = self._fit_stage(
            tim_blob, sources_by_idx, unsolved, segments,
            eval_segments=wide_segments)
        chi2_tracker[2] = chi2_stage2

        if history:
            mod_full = t2.getModelImage(0)
            for idx in unsolved:
                hist.stages.append(self._record(2, sources_by_idx[idx], idx,
                                                chi2_stage2[idx], tim_blob, mod_full,
                                                chi2_wide=chi2w_tracker[2].get(idx, float("nan"))))

        for idx in unsolved:
            # Deferral starts after this tier: a source that wins
            # PointSource here is a star and locks now; dragging it through
            # the galaxy tiers lets an extended model win the local window
            # while rendering puffy wings globally.
            ps = chi2_stage2[idx]
            sg = chi2_tracker[1][idx]
            delta = ps - (sg + self.simplegalaxy_penalty)
            if (ps > self.sufficient_thresh) and (sg > self.sufficient_thresh):
                continue  # neither wins; advance to Exp/Dev
            if (delta <= 0) and (ps <= self.sufficient_thresh):
                locked_source[idx] = sources_by_idx[idx]  # PointSource wins
                solved[idx] = True
                if history:
                    self._mark_accepted(hist, idx, 2)

        # --- Stage 3: ExpGalaxy for unsolved --------------------------------
        for idx in idx_list:
            if solved[idx]:
                sources_by_idx[idx] = locked_source[idx]
            else:
                row = mer_cat[idx]
                sep_row = blob.sep_rows.get(idx)
                sources_by_idx[idx] = _build_initial_source(row, ExpGalaxy, sep_row, pixscale_arcsec)
        unsolved = [i for i in idx_list if not solved[i]]
        chi2_stage3, chi2w_tracker[3], t3 = self._fit_stage(
            tim_blob, sources_by_idx, unsolved, segments,
            eval_segments=wide_segments)
        chi2_tracker[3] = chi2_stage3

        if history:
            mod_full = t3.getModelImage(0)
            for idx in unsolved:
                hist.stages.append(self._record(3, sources_by_idx[idx], idx,
                                                chi2_stage3[idx], tim_blob, mod_full,
                                                chi2_wide=chi2w_tracker[3].get(idx, float("nan"))))

        # --- Stage 4: DevGalaxy for unsolved + Exp/Dev decision -------------
        exp_snapshot = {i: copy.deepcopy(sources_by_idx[i]) for i in unsolved}
        for idx in idx_list:
            if solved[idx]:
                sources_by_idx[idx] = locked_source[idx]
            else:
                row = mer_cat[idx]
                sep_row = blob.sep_rows.get(idx)
                sources_by_idx[idx] = _build_initial_source(row, DevGalaxy, sep_row, pixscale_arcsec)
        chi2_stage4, chi2w_tracker[4], t4 = self._fit_stage(
            tim_blob, sources_by_idx, unsolved, segments,
            eval_segments=wide_segments)
        chi2_tracker[4] = chi2_stage4
        dev_snapshot = {i: copy.deepcopy(sources_by_idx[i]) for i in unsolved}

        if history:
            mod_full = t4.getModelImage(0)
            for idx in unsolved:
                hist.stages.append(self._record(4, sources_by_idx[idx], idx,
                                                chi2_stage4[idx], tim_blob, mod_full,
                                                chi2_wide=chi2w_tracker[4].get(idx, float("nan"))))

        still_unsolved = []
        for idx in unsolved:
            if idx in defer:
                still_unsolved.append(idx)
                continue  # full ladder: evaluate Composite too
            exp_chi2 = chi2_tracker[3][idx]
            dev_chi2 = chi2_tracker[4][idx]
            sg_chi2 = chi2_tracker[1][idx]
            if (exp_chi2 > self.sufficient_thresh) and (dev_chi2 > self.sufficient_thresh):
                still_unsolved.append(idx)
                continue
            if (sg_chi2 <= exp_chi2) and (sg_chi2 <= dev_chi2) and (sg_chi2 <= self.sufficient_thresh):
                row = mer_cat[idx]
                locked_source[idx] = _build_initial_source(
                    row, SimpleGalaxy, None, pixscale_arcsec,
                    simplegalaxy_radius=self.simplegalaxy_radius)
                solved[idx] = True
                if history:
                    self._mark_accepted(hist, idx, 1)
                continue
            if abs(exp_chi2 - dev_chi2) < self.exp_dev_similar_thresh:
                still_unsolved.append(idx)
                continue
            if (exp_chi2 < dev_chi2) and (exp_chi2 <= self.sufficient_thresh):
                locked_source[idx] = exp_snapshot[idx]
                solved[idx] = True
                if history:
                    self._mark_accepted(hist, idx, 3)
                continue
            if (dev_chi2 < exp_chi2) and (dev_chi2 <= self.sufficient_thresh):
                locked_source[idx] = dev_snapshot[idx]
                solved[idx] = True
                if history:
                    self._mark_accepted(hist, idx, 4)
                continue
            still_unsolved.append(idx)

        # --- Stage 5: CompositeGalaxy for the remainder ---------------------
        if still_unsolved:
            for idx in idx_list:
                if solved[idx]:
                    sources_by_idx[idx] = locked_source[idx]
                else:
                    row = mer_cat[idx]
                    sep_row = blob.sep_rows.get(idx)
                    sources_by_idx[idx] = _build_initial_source(
                        row, FixedCompositeGalaxy, sep_row, pixscale_arcsec)
            chi2_stage5, chi2w_tracker[5], t5 = self._fit_stage(
                tim_blob, sources_by_idx, still_unsolved, segments,
                eval_segments=wide_segments)
            chi2_tracker[5] = chi2_stage5

            if history:
                mod_full = t5.getModelImage(0)
                for idx in still_unsolved:
                    hist.stages.append(self._record(5, sources_by_idx[idx], idx,
                                                    chi2_stage5[idx], tim_blob, mod_full,
                                                    chi2_wide=chi2w_tracker[5].get(idx, float("nan"))))

            for idx in still_unsolved:
                exp_chi2 = chi2_tracker[3].get(idx, np.inf)
                dev_chi2 = chi2_tracker[4].get(idx, np.inf)
                comp_chi2 = chi2_tracker[5].get(idx, np.inf)
                # Farmer's Stage-5 selection is a preference ladder (Exp,
                # then Dev, then Composite while sufficient), with an argmin
                # over all five only when every galaxy model is bad; ties
                # break toward the simpler model. Full-ladder members take
                # the pure argmin instead.
                if (idx not in defer
                        and (exp_chi2 <= comp_chi2)
                        and (exp_chi2 <= self.sufficient_thresh)):
                    locked_source[idx] = exp_snapshot[idx]
                    accepted_stage = 3
                elif (idx not in defer and (dev_chi2 <= comp_chi2)
                        and (dev_chi2 <= self.sufficient_thresh)):
                    locked_source[idx] = dev_snapshot[idx]
                    accepted_stage = 4
                elif idx not in defer and comp_chi2 <= self.sufficient_thresh:
                    locked_source[idx] = sources_by_idx[idx]  # Composite
                    accepted_stage = 5
                else:
                    # Every galaxy model is bad: argmin over all five.
                    # Full-ladder members argmin over the wide-window chi2
                    # so wing damage outside the segment counts.
                    if idx in defer:
                        chis = {
                            1: chi2w_tracker.get(2, {}).get(idx, np.inf),  # PointSource
                            2: chi2w_tracker.get(1, {}).get(idx, np.inf),  # SimpleGalaxy
                            3: chi2w_tracker.get(3, {}).get(idx, np.inf),
                            4: chi2w_tracker.get(4, {}).get(idx, np.inf),
                            5: chi2w_tracker.get(5, {}).get(idx, np.inf),
                        }
                    else:
                        chis = {
                            1: chi2_tracker[2].get(idx, np.inf),  # PointSource
                            2: chi2_tracker[1].get(idx, np.inf),  # SimpleGalaxy
                            3: exp_chi2, 4: dev_chi2, 5: comp_chi2,
                        }
                    best_stage = min(chis, key=chis.get)
                    row = mer_cat[idx]
                    if best_stage == 1:
                        locked_source[idx] = _build_initial_source(row, PointSource, None, pixscale_arcsec)
                        accepted_stage = 2
                    elif best_stage == 2:
                        locked_source[idx] = _build_initial_source(
                            row, SimpleGalaxy, None, pixscale_arcsec,
                            simplegalaxy_radius=self.simplegalaxy_radius)
                        accepted_stage = 1
                    elif best_stage == 3:
                        locked_source[idx] = exp_snapshot[idx]
                        accepted_stage = 3
                    elif best_stage == 4:
                        locked_source[idx] = dev_snapshot[idx]
                        accepted_stage = 4
                    else:
                        locked_source[idx] = sources_by_idx[idx]
                        accepted_stage = 5
                solved[idx] = True
                if history:
                    self._mark_accepted(hist, idx, accepted_stage)

        # --- Final joint re-fit -------------------------------------------
        # Position is thawed here (as in Farmer's force_models): locking pos
        # at the MER centroid leaves a dipole residual whenever MER and VIS
        # astrometry differ by a fraction of a pixel.
        all_sources_final = [locked_source[i] for i in idx_list]
        tractor_final = Tractor([tim_blob], all_sources_final, optimizer=ConstrainedOptimizer())
        for src in all_sources_final:
            src.thawAllParams()
            if isinstance(src, SimpleGalaxy):
                src.freezeParam("shape")
        self._optimize(tractor_final)

        # --- Optional Stage 6: guarded SersicGalaxy refit -------------------
        # Free-n Sersic for Exp/Dev/Composite sources bright enough to
        # constrain n; kept only if the total chi^2 drops significantly.
        if self.include_sersic_tier:
            self._try_sersic_tier(tim_blob, blob, idx_list, locked_source,
                                  all_sources_final, mer_cat, pixscale_arcsec,
                                  hist if history else None,
                                  segments=segments)
            tractor_final = Tractor([tim_blob], all_sources_final, optimizer=ConstrainedOptimizer())
            for src in all_sources_final:
                src.thawAllParams()
                if isinstance(src, SimpleGalaxy):
                    src.freezeParam("shape")
            self._optimize(tractor_final)

        # Clamp runaway centroids back to the MER prior and re-fit
        # flux-only; runs last so no later refit reintroduces drift past
        # the cap.
        self._clamp_positions(tractor_final, all_sources_final, idx_list,
                              mer_cat)

        if history:
            model = tractor_final.getModelImage(0)
            data = tim_blob.getImage()
            hist.final = FinalRecord(
                data=data.copy(),
                model=model.copy(),
                residual=(data - model).copy(),
                footprint=blob.footprint.copy(),
            )

        result = {idx: src for idx, src in zip(idx_list, all_sources_final, strict=False)}
        return result, hist

    # ---- Position-drift clamp -----------------------------------------
    def _clamp_positions(self, tractor_final, sources, idx_list, mer_cat):
        """Reset any source that drifted past ``max_pos_shift_arcsec`` from its
        MER prior, then re-fit flux-only for the clamped ones.

        Keeps the sub-pixel astrometry refinement for well-behaved sources
        while preventing faint/blended centroids from wandering off the prior.
        """
        cap = self.max_pos_shift_arcsec
        clamped = []
        for src, idx in zip(sources, idx_list, strict=False):
            try:
                p = src.getPosition()
                ra0 = float(mer_cat[idx]["ra"])
                dec0 = float(mer_cat[idx]["dec"])
            except Exception:
                continue
            cosd = np.cos(np.radians(dec0))
            shift = np.hypot((p.ra - ra0) * cosd, p.dec - dec0) * 3600.0
            if shift > cap:
                src.setPosition(RaDecPos(ra0, dec0))
                clamped.append(src)
        if not clamped:
            return
        # Brightness only: freeze position so the reset sticks and shape so
        # the fitted morphology is preserved.
        for src in sources:
            src.freezeAllBut("brightness")
        self._optimize(tractor_final)

    # ---- Sersic-tier extension ----------------------------------------
    def _try_sersic_tier(self, tim_blob, blob, idx_list, locked_source,
                         all_sources_final, mer_cat, pixscale_arcsec,
                         hist, segments=None):
        """For each Exp/Dev/Composite source bright enough to constrain Sérsic
        index, try a free-n SersicGalaxy and keep it only if the total chi^2
        on the wing-wide window improves by more than ``sersic_dchi2_accept``.
        Updates ``locked_source``, ``all_sources_final`` (in place), and ``hist``.
        """
        eligible_classes = (ExpGalaxy, DevGalaxy, FixedCompositeGalaxy)
        # Don't catch SimpleGalaxy via ExpGalaxy isinstance:
        for pos_in_list, idx in enumerate(idx_list):
            src = locked_source[idx]
            if isinstance(src, SimpleGalaxy):
                continue
            if not isinstance(src, eligible_classes):
                continue

            # segments is bbox-cropped to tim_blob's grid; blob.segments is
            # the full-frame original (history path).
            seg = (segments if segments is not None else blob.segments)[idx]
            snr = self._segment_snr(tim_blob, seg)
            if snr < self.sersic_min_snr:
                logger.debug("Skip Sersic for src#%s: SNR=%.1f < %s",
                             idx, snr, self.sersic_min_snr)
                continue

            # Judge on the dilated segment: the failure this tier fixes is a
            # wing-mismatch ring outside the SEP segment (which hugs the
            # core and is blind to the ring). Both candidates score on the
            # same window with the same frozen neighbors.
            from scipy import ndimage as _ndi
            try:
                re_pix = float(src.shape.re) / max(pixscale_arcsec, 1e-6)
            except Exception:
                re_pix = 10.0
            grow = int(np.clip(2.0 * re_pix, 10, 40))
            eval_seg = _ndi.distance_transform_edt(~seg) <= grow

            # Score the locked model with the free-parameter count of its
            # accepted tier (pos frozen, shape+brightness free), captured
            # before the freezeAllParams below; a frozen source counts 0
            # params, making the dof inconsistent with the Sersic candidate.
            src.thawAllParams()
            src.freezeParam("pos")
            n_free_prev = _count_free_params(src)
            tractor_cur = Tractor([tim_blob], all_sources_final, optimizer=ConstrainedOptimizer())
            for s in all_sources_final:
                s.freezeAllParams()
            chi_prev = tractor_cur.getChiImage(imgi=0)
            npix_eval = int(eval_seg.sum())
            prev_tot = float(np.sum(chi_prev[eval_seg] ** 2))
            prev_chi2 = prev_tot / max(npix_eval - n_free_prev, 1)

            # Two starts, n=1 and n=4: the Sersic likelihood is bimodal for
            # a disk+core galaxy. Each start gets a fresh seed shape since
            # the first fit mutates it.
            saved = all_sources_final[pos_in_list]
            best = None   # (total chi2, reduced chi2, segment chi2, source, n)

            def _fresh_seed(idx=idx):
                sep_row = blob.sep_rows.get(idx)
                if sep_row is not None:
                    return _seed_shape_from_sep(sep_row, pixscale_arcsec)
                sh = EllipseESoft.fromRAbPhi(0.3, 1.0, 0.0)
                sh.lowers = [-5.0, -np.inf, -np.inf]
                sh.uppers = [1.0, np.inf, np.inf]
                return sh

            for n_init in (1.0, 4.0):
                new_src = SersicGalaxy(
                    src.getPosition(),
                    NanoMaggies(VIS=float(src.brightness.getFlux("VIS"))),
                    _fresh_seed(), SersicIndex(n_init))
                all_sources_final[pos_in_list] = new_src
                try:
                    tractor_try = Tractor([tim_blob], all_sources_final,
                                          optimizer=ConstrainedOptimizer())
                    for s in all_sources_final:
                        if s is new_src:
                            s.thawAllParams()
                            s.freezeParam("pos")
                        else:
                            s.freezeAllParams()
                    self._optimize(tractor_try)
                    chi_try = tractor_try.getChiImage(imgi=0)
                    n_free_new = _count_free_params(new_src)
                    cand_tot = float(np.sum(chi_try[eval_seg] ** 2))
                    cand_chi2 = cand_tot / max(npix_eval - n_free_new, 1)
                    # Segment-window chi^2 for the figure, so the stage-6
                    # row is comparable with the earlier tiers.
                    cand_seg = (float(np.sum(chi_try[seg] ** 2))
                                / max(int(seg.sum()) - n_free_new, 1))
                except Exception as exc:
                    logger.debug("Sersic refit (n_init=%s) raised %s for "
                                 "src#%s.", n_init, exc, idx)
                    continue
                if best is None or cand_tot < best[0]:
                    best = (cand_tot, cand_chi2, cand_seg, new_src,
                            float(new_src.sersicindex.val))

            if best is None:
                all_sources_final[pos_in_list] = saved
                continue
            new_tot, new_chi2, new_seg_chi2, new_src, n_fit = best
            all_sources_final[pos_in_list] = new_src

            improved = (prev_tot - new_tot) > self.sersic_dchi2_accept

            if hist is not None:
                # Render the best candidate, not whichever start ran last.
                tractor_best = Tractor([tim_blob], all_sources_final,
                                       optimizer=ConstrainedOptimizer())
                rec = StageRecord(
                    stage=6,
                    model_name=f"SersicGalaxy(n={n_fit:.2f})",
                    source_idx=idx,
                    chi2_red=new_seg_chi2,
                    chi2_wide=new_chi2,
                    flux_ujy=float(new_src.brightness.getFlux("VIS")) * UJY_PER_NMGY,
                    model_image=tractor_best.getModelImage(0).copy(),
                    residual_image=(tim_blob.getImage()
                                    - tractor_best.getModelImage(0)).copy(),
                    accepted=bool(improved),
                )
                hist.stages.append(rec)
                # Unmark the previously accepted tier so the figure shows
                # stage 6 winning.
                if improved:
                    for r in hist.stages:
                        if (r.source_idx == idx and r.stage != 6 and r.accepted):
                            r.accepted = False

            if improved:
                locked_source[idx] = new_src
                logger.debug("Sersic accepted for src#%s: dchi2=%.1f "
                             "(chi2red %.3f -> %.3f), n=%.2f",
                             idx, prev_tot - new_tot, prev_chi2, new_chi2, n_fit)
            else:
                logger.debug("Sersic rejected for src#%s: dchi2=%.1f "
                             "(needs > %.1f), n=%.2f",
                             idx, prev_tot - new_tot,
                             self.sersic_dchi2_accept, n_fit)
                all_sources_final[pos_in_list] = saved

    @staticmethod
    def _segment_snr(tim_blob: Image, segment: np.ndarray) -> float:
        """Approximate segment S/N: matched-filter value for a model that
        is constant inside the segment."""
        data = tim_blob.getImage()
        invvar = tim_blob.getInvvar()
        mask = segment & (invvar > 0)
        if not mask.any():
            return 0.0
        signal = float(np.sum(data[mask]))
        noise = float(np.sqrt(np.sum(1.0 / invvar[mask])))
        return signal / max(noise, 1e-12)

    # ---- history helpers -----------------------------------------------
    def _record(self, stage, src, idx, chi2_red, tim_blob, mod_full,
                chi2_wide=float("nan")):
        try:
            flux_ujy = float(src.brightness.getFlux("VIS")) * UJY_PER_NMGY
        except Exception:
            flux_ujy = float("nan")
        return StageRecord(
            stage=stage,
            model_name=_classname(src),
            source_idx=idx,
            chi2_red=chi2_red,
            model_image=mod_full.copy(),
            residual_image=(tim_blob.getImage() - mod_full).copy(),
            accepted=False,
            flux_ujy=flux_ujy,
            chi2_wide=chi2_wide,
        )

    @staticmethod
    def _mark_accepted(hist: History, idx: int, stage: int) -> None:
        for rec in hist.stages:
            if rec.source_idx == idx and rec.stage == stage:
                rec.accepted = True
