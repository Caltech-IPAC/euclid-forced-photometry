"""Spitzer/IRAC forced photometry with fixed VIS source models.

``run_forced_photometry(..., target_bands={"irac": ("IRAC1", ...)})`` calls
:func:`fit_irac_forced`, which cuts each IRAC mosaic to the field, rebuilds
its spatially varying effective PRF from the detector PRFs and the
contributing exposures, and fits the VIS models to the IRAC pixels with
the shared recipe of :mod:`euclid_phot.forced`. This subpackage holds only
what is IRAC's own: mosaic access (``archive``, ``data``, ``mosaic``), the
effective PRF (``prf``, ``prfmap``), the PRF corrections (``config``,
``fit``) and the DAWN catalog checks (``catalogs``). Notebook 03 shows the
workflow.
"""

from .archive import PBCDLocalSet, PBCDProductSet, discover_pbcd, fetch_pbcd
from .catalogs import (
    DAWN_PL_CATALOG_URLS,
    WISE_IRAC_PAIRS,
    compare_dawn,
    compare_wise_irac,
    dawn_comparison_defaults,
    fetch_dawn_pl_catalog,
    match_dawn_irac,
    read_dawn_region,
    select_dawn_disagreements,
)
from .config import (
    IRAC_CHANNELS,
    IRAC_NATIVE_PIXEL_SCALE_ARCSEC,
    IRAC_PRF_FLUX_CORRECTION,
    IRAC_PRF_SYSTEMATIC_FRACTION,
    IRAC_PSF_FWHM_ARCSEC,
    normalize_channel,
)
from .data import (
    CosmicDawnLocalSet,
    CosmicDawnProductSet,
    MosaicInputs,
    SEIPLocalSet,
    SEIPProductSet,
    fetch_cosmic_dawn_mosaic,
    fetch_seip_mosaic,
    fetch_sha_mosaic,
    parse_cosmic_dawn_info,
    parse_seip_dcelist,
)
from .mosaic import MosaicCutout, load_mosaic_cutout
from .fit import (
    MosaicPhotometryResult,
    PreparedChannel,
    fetch_irac_mosaic,
    fit_irac_forced,
    fit_prepared,
    irac_catalog,
    prepare_channel,
    select_irac_scene,
)
from .prf import DetectorPRFLibrary, ExposureGeometry, parse_refptg
from .prfmap import (
    PRFMapFileGrid,
    PRFMapGeneratedGrid,
    build_prfmap_psf,
)

__all__ = [
    "CosmicDawnLocalSet", "CosmicDawnProductSet", "DAWN_PL_CATALOG_URLS",
    "DetectorPRFLibrary", "ExposureGeometry", "IRAC_CHANNELS",
    "IRAC_NATIVE_PIXEL_SCALE_ARCSEC", "IRAC_PRF_FLUX_CORRECTION",
    "IRAC_PRF_SYSTEMATIC_FRACTION", "IRAC_PSF_FWHM_ARCSEC", "MosaicCutout",
    "MosaicInputs", "MosaicPhotometryResult", "PBCDLocalSet", "PBCDProductSet",
    "PRFMapFileGrid", "PRFMapGeneratedGrid",
    "PreparedChannel", "SEIPLocalSet", "SEIPProductSet",
    "WISE_IRAC_PAIRS", "build_prfmap_psf",
    "compare_dawn", "compare_wise_irac", "dawn_comparison_defaults",
    "discover_pbcd", "fetch_cosmic_dawn_mosaic",
    "fetch_dawn_pl_catalog", "fetch_irac_mosaic", "fetch_pbcd",
    "fetch_seip_mosaic", "fetch_sha_mosaic", "fit_irac_forced",
    "fit_prepared", "irac_catalog", "load_mosaic_cutout",
    "match_dawn_irac", "normalize_channel",
    "parse_cosmic_dawn_info", "parse_refptg", "parse_seip_dcelist",
    "prepare_channel", "read_dawn_region", "select_dawn_disagreements",
    "select_irac_scene",
]
