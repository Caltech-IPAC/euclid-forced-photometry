"""euclid_phot: forced photometry on the selected Euclid release with Tractor.

Every band is measured with the same recipe; only its data and PSF differ.

Shared, for every band:

    units           flux units (uJy, nanomaggies, AB) and pixel geometry
    tractor_image   the Tractor image: data, invvar, PSF, WCS, photometric scale
    pixel_psf       PSFs that vary across the image (nearest stamp per source)
    sources         prior source models (MER / coordinates) and clone_sources
    selection       model-selection tree on the prior band (VIS)
    forced          freeze to fluxes, solve, flux errors, chi inflation,
                    bounded recentering
    background      SEP sky maps;  scene: which sources enter a band's fit
    calibrate       empirical error calibration on empty positions
    pipeline        run_forced_photometry, the one-call driver

Band-specific (data access, PSF, corrections, catalog checks):

    euclid/         MER catalog, cutouts, PSF products, flags; NISP Y/J/H
    wise/           unWISE coadds, PSF, unWISE catalog; W1/W2
    irac/           Spitzer mosaics, effective PRF, DAWN catalog; IRAC1-4
    ground_based/   a user's ground image: WCS and PSF checks, SDSS comparison

The package imports Tractor and matplotlib only when first used, so
importing euclid_phot does not require them. The notebooks download their
inputs from IRSA/S3 on first run and cache them under
``examples/data/q1/`` or ``examples/data/dr1/deep_survey/``.
"""
from __future__ import annotations

__version__ = "0.5.0"

# Names that resolve via __getattr__ (PEP 562). Listed here so IDEs and
# `from euclid_phot import *` find them.
_LAZY = {
    "set_data_access": ("data_access", "set_data_access"),
    "get_data_access": ("data_access", "get_data_access"),
    "get_data_dir": ("data_access", "get_data_dir"),
    # shared units and Tractor image
    "UJY_PER_NMGY": ("units", "UJY_PER_NMGY"),
    "flux_to_ab_mag": ("units", "flux_to_ab_mag"),
    "flux_err_to_mag_err": ("units", "flux_err_to_mag_err"),
    "AstropyWCSAdapter": ("tractor_image", "AstropyWCSAdapter"),
    "make_tractor_image": ("tractor_image", "make_tractor_image"),
    "invvar_from_rms": ("tractor_image", "invvar_from_rms"),
    # prior source models
    "build_sources_from_mer": ("sources", "build_sources_from_mer"),
    "build_sources_from_coords": ("sources", "build_sources_from_coords"),
    "clone_sources": ("sources", "clone_sources"),
    # scene and background
    "footprint_mask": ("scene", "footprint_mask"),
    "trim_catalog_to_cutout": ("scene", "trim_catalog_to_cutout"),
    "select_isolated_sources": ("scene", "select_isolated_sources"),
    "estimate_sep_background": ("background", "estimate_sep_background"),
    # forced photometry (shared recipe)
    "flux_tractor": ("forced", "flux_tractor"),
    "solve_fluxes": ("forced", "solve_fluxes"),
    "flux_inverse_variance": ("forced", "flux_inverse_variance"),
    "flux_errors_ujy": ("forced", "flux_errors_ujy"),
    "fluxes_ujy": ("forced", "fluxes_ujy"),
    "chi_inflation": ("forced", "chi_inflation"),
    "fit_forced_photometry": ("forced", "fit_forced_photometry"),
    "conditional_flux_errors": ("forced", "conditional_flux_errors"),
    "fit_free_shapes": ("forced", "fit_free_shapes"),
    "refine_positions": ("forced", "refine_positions"),
    "fit_bounded_positions": ("forced", "fit_bounded_positions"),
    "gauss_newton_recenter": ("forced", "gauss_newton_recenter"),
    "measure_residual": ("forced", "measure_residual"),
    # Euclid MER data
    "query_mer_catalog": ("euclid.catalog", "query_mer_catalog"),
    "discover_mer_mosaics": ("euclid.cutouts", "discover_mer_mosaics"),
    "fetch_cutout": ("euclid.cutouts", "fetch_cutout"),
    "extract_catalog_psf": ("euclid.psf", "extract_catalog_psf"),
    "extract_grid_psf": ("euclid.psf", "extract_grid_psf"),
    "get_psf_for_source": ("euclid.psf", "get_psf_for_source"),
    "psf_summary": ("euclid.psf", "psf_summary"),
    "build_tractor_image": ("euclid.image", "build_tractor_image"),
    "fit_nisp_forced": ("euclid.nisp", "fit_nisp_forced"),
    "flag_sources": ("euclid.flags", "flag_sources"),
    "blend_flags": ("euclid.flags", "blend_flags"),
    "bright_star_pixel_mask": ("euclid.flags", "bright_star_pixel_mask"),
    "starsignal_pixel_mask": ("euclid.flags", "starsignal_pixel_mask"),
    "probable_bright_stars": ("euclid.flags", "probable_bright_stars"),
    "MER_VIS_BAD_BITS": ("euclid.config", "MER_VIS_BAD_BITS"),
    "MER_NISP_BAD_BITS": ("euclid.config", "MER_NISP_BAD_BITS"),
    "MER_VIS_FLAG_BITS": ("euclid.config", "MER_VIS_FLAG_BITS"),
    # extinction
    "query_ebv": ("extinction", "query_ebv"),
    "add_extinction_columns": ("extinction", "add_extinction_columns"),
    "EXTINCTION_COEFF": ("config", "EXTINCTION_COEFF"),
    # model selection (prior band)
    "SimpleGalaxy": ("selection", "SimpleGalaxy"),
    "ModelSelector": ("selection", "ModelSelector"),
    "detect_blobs": ("selection", "detect_blobs"),
    "assign_mer_to_blobs": ("selection", "assign_mer_to_blobs"),
    "run_model_selection": ("selection", "run_model_selection"),
    "reproduce_figure3": ("selection", "reproduce_figure3"),
    # injection-recovery validation
    "make_truth_table": ("injection", "make_truth_table"),
    "inject_sources": ("injection", "inject_sources"),
    "run_injection_recovery": ("injection", "run_injection_recovery"),
    "summarize_recovery": ("injection", "summarize_recovery"),
    # error calibration
    "draw_empty_positions": ("calibrate", "draw_empty_positions"),
    "measure_error_inflation": ("calibrate", "measure_error_inflation"),
    "calibrate_result_errors": ("calibrate", "calibrate_result_errors"),
    # unWISE
    "fetch_unwise_cutouts": ("wise.data", "fetch_unwise_cutouts"),
    "get_wise_psf": ("wise.psf", "get_wise_psf"),
    "fit_wise_forced": ("wise.fit", "fit_wise_forced"),
    "query_unwise_2019": ("wise.catalog", "query_unwise_2019"),
    "vega_mag_to_ujy": ("wise.catalog", "vega_mag_to_ujy"),
    "compare_unwise_2019": ("wise.catalog", "compare_unwise_2019"),
    # IRAC
    "fit_irac_forced": ("irac.fit", "fit_irac_forced"),
    # pipeline
    "run_forced_photometry": ("pipeline", "run_forced_photometry"),
    "ForcedPhotometryResult": ("pipeline", "ForcedPhotometryResult"),
    # catalog assembly
    "build_catalog": ("catalog_table", "build_catalog"),
    # submodules
    "config": ("config", None),
    "units": ("units", None),
    "forced": ("forced", None),
    "viz": ("viz", None),
    "euclid": ("euclid", None),
    "wise": ("wise", None),
    "irac": ("irac", None),
    "ground_based": ("ground_based", None),
}

__all__ = list(_LAZY.keys()) + ["__version__"]


def __getattr__(name):
    if name not in _LAZY:
        raise AttributeError(f"module 'euclid_phot' has no attribute {name!r}")
    import importlib
    module_name, attr_name = _LAZY[name]
    module = importlib.import_module(f".{module_name}", __name__)
    if attr_name is None:
        return module
    return getattr(module, attr_name)


def __dir__():
    return sorted(__all__)
