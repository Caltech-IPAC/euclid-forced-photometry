"""Forced photometry on a user-supplied ground image with the VIS source models.

Only what is a ground image's own lives here: reading and converting the
user's arrays (``image``), the Euclid inputs around it (``prior``), checking
its WCS and PSF against field stars (``astrometry``, ``psf_validation``),
the fit setup with tiling, PSF choice and optional sky (``fit``), the SDSS
example inputs (``sdss``), the SDSS comparison (``compare``) and figures
(``plots``). The fit itself is the
shared recipe of :mod:`euclid_phot.forced`. Notebook 04 shows the workflow.
"""

from .astrometry import WCSRegistration, register_image
from .compare import compare_sdss_g
from .fit import GroundFit, fit_ground, fit_ground_image, save_fit, select_mer_scene
from .image import GroundInputs, SmallImage, gaussian_psf, make_image, read_fits_image
from .plots import (
    show_background_comparison,
    show_catalog_comparison,
    show_fit_cutouts,
    show_inputs,
    show_psf_stars,
    show_registration,
    show_scene,
    show_sky_injection_test,
)
from .prior import EuclidInputs, fit_vis, prepare_euclid
from .psf_validation import PSFAssessment, assess_psf, average_star_psf
from .sdss import SDSSExample, SDSSFrame, fetch_sdss_example, nearby_sdss_frames

__all__ = [
    "EuclidInputs", "GroundFit", "GroundInputs", "PSFAssessment", "SDSSExample",
    "SDSSFrame", "SmallImage",
    "WCSRegistration", "assess_psf", "average_star_psf", "compare_sdss_g",
    "fetch_sdss_example", "fit_ground", "fit_ground_image", "fit_vis", "gaussian_psf", "make_image",
    "nearby_sdss_frames", "prepare_euclid", "read_fits_image", "register_image", "save_fit",
    "select_mer_scene", "show_background_comparison", "show_catalog_comparison",
    "show_fit_cutouts", "show_inputs", "show_psf_stars", "show_registration",
    "show_scene", "show_sky_injection_test",
]
