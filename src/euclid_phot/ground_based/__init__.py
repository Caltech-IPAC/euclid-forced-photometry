"""Forced photometry on a user-supplied ground image using Euclid VIS priors.

Ground-image modeling and checks live in this package. VIS morphology uses
the standard ``euclid_phot`` pipeline. Notebook 04 shows the workflow.
"""

from .image import gaussian_psf

from .pipeline import (
    GroundFit,
    GroundInputs,
    EuclidInputs,
    compare_sdss_g,
    fit_ground,
    fit_vis,
    make_image,
    prepare_euclid,
    register_image, assess_psf,
    read_fits_image,
    save_fit,
)
from .plots import (
    show_inputs,
    show_scene,
    show_registration,
    show_psf_stars,
    show_fit_cutouts,
    show_catalog_comparison,
    show_background_comparison,
    show_sky_injection_test,
)

__all__ = [
    "GroundFit", "GroundInputs", "EuclidInputs", "compare_sdss_g", "fit_ground",
    "fit_vis", "gaussian_psf", "make_image", "prepare_euclid", "register_image",
    "assess_psf", "read_fits_image", "save_fit",
    "show_inputs", "show_scene", "show_registration", "show_psf_stars",
    "show_fit_cutouts", "show_catalog_comparison", "show_background_comparison",
    "show_sky_injection_test",
]
