"""Chi^2 decision-tree model selection for Euclid VIS forced photometry.

An opt-in alternative to the MER-column-driven model construction in
``sources.py``, porting the decision tree of The Farmer (Weaver et al. 2023,
ApJS 269, 20; Figures 3-4) onto the Tractor machinery used here.

``sufficient_thresh`` = 1.5 is the paper's chi^2_N bad-fit criterion;
``simplegalaxy_radius`` (0.30 arcsec), ``simplegalaxy_penalty`` (0.05) and
``exp_dev_similar_thresh`` (0.15) are Euclid tunings of the Farmer values
(0.45, 0.1, 0.2).

Deviations from Farmer: positions are frozen during the tree, with a hard
``max_pos_shift_arcsec`` clamp at the final joint re-fit; every source takes
the full ladder with the lowest chi-squared winning (``full_ladder``; the
published first-sufficient shortcut biases faint fluxes low); and a guarded
SersicGalaxy 6th tier (``include_sersic_tier``) extends the published
five-model ladder.

Public API: ``ModelSelector``, ``detect_blobs``, ``assign_mer_to_blobs``,
``run_model_selection``, ``reproduce_figure3``.

Modules: ``models`` (SimpleGalaxy, Euclid defaults, blob records),
``tree`` (ModelSelector), ``detect`` (SEP blobs, MER matching),
``run`` (run_model_selection over all blobs), ``figure3``.
"""
from .detect import (
    SEP_DEFAULT_KERNEL,
    assign_mer_to_blobs,
    detect_blobs,
    unlisted_sep_detections,
)
from .figure3 import reproduce_figure3
from .models import (
    EUCLID_DEFAULTS,
    MODEL_NAMES,
    Blob,
    FinalRecord,
    History,
    SimpleGalaxy,
    StageRecord,
    set_simplegalaxy_radius,
)
from .run import run_model_selection
from .tree import ModelSelector

__all__ = [
    "Blob", "EUCLID_DEFAULTS", "FinalRecord", "History", "MODEL_NAMES",
    "ModelSelector", "SEP_DEFAULT_KERNEL", "SimpleGalaxy", "StageRecord",
    "assign_mer_to_blobs", "detect_blobs", "reproduce_figure3",
    "run_model_selection", "set_simplegalaxy_radius", "unlisted_sep_detections",
]
