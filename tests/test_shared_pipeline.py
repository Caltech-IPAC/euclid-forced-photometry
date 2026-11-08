"""One shared reader and fitting pipeline works for both releases."""
from pathlib import Path
import json

from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
import numpy as np
import pytest

import euclid_phot as ep
import euclid_phot.pipeline as pipeline
from euclid_phot.euclid import cutouts, psf


@pytest.fixture(autouse=True, params=["Q1", "DR1"])
def selected_release(request):
    previous = ep.get_data_access()
    ep.set_data_access(request.param)
    yield
    ep.set_data_access(previous)


def make_wcs(n=25):
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [269.48, 67.3]
    wcs.wcs.crpix = [(n+1)/2, (n+1)/2]
    wcs.wcs.cdelt = [-0.1/3600, 0.1/3600]
    return wcs


def test_cutout_cold_cache_warm_cache_and_force(tmp_path, monkeypatch):
    wcs = make_wcs(9)
    header = wcs.to_header()
    header["MAGZERO"] = 24.5
    products = {"VIS": {kind: {"tile_id": "123", "s3": kind}
                         for kind in ["science", "rms", "flag"]}}
    for kind, value in [("science", 1.0), ("rms", 2.0), ("flag", 0)]:
        fits.PrimaryHDU(np.full((9, 9), value), header).writeto(tmp_path / (kind + ".fits"))
    calls = []

    def discover(*args, **kwargs):
        calls.append("discover")
        return products

    def open_mosaic(path, cache_dir):
        calls.append(path)
        return fits.open(tmp_path / (path + ".fits")), "local"

    monkeypatch.setattr(cutouts, "discover_mer_mosaics", discover)
    monkeypatch.setattr(cutouts, "_open_mosaic", open_mosaic)
    kwargs = dict(data_dir=tmp_path / "dr1", with_flag=True)
    c = ep.fetch_cutout("VIS", 269.48, 67.3, 0.5, **kwargs)
    assert c.shape == (5, 5)
    np.testing.assert_array_equal(c.data, 1)
    np.testing.assert_array_equal(c.rms, 2)
    np.testing.assert_array_equal(c.flag, 0)
    assert calls == ["discover", "science", "rms", "flag"]
    calls.clear()
    ep.fetch_cutout("VIS", 269.48, 67.3, 0.5, **kwargs)
    assert calls == []
    ep.fetch_cutout("VIS", 269.48, 67.3, 0.5, force_download=True, **kwargs)
    assert calls == ["discover", "science", "rms", "flag"]


def test_shared_catalog_and_grid_psf_readers(tmp_path, monkeypatch):
    from euclid_phot import netutils
    wcs = make_wcs(5)
    header = wcs.to_header()
    header["STMPSIZE"] = 3
    stamps = np.zeros((5, 5), dtype=np.float32)
    stamps[1:4, 1:4] = 1/9
    table = Table(dict(x_center=[3], y_center=[3], x=[3.0], y=[3.0],
                       RA=[269.48], Dec=[67.3], FWHM=[0.2]))
    path = tmp_path / "psf.fits"
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(stamps, header), fits.BinTableHDU(table)]).writeto(path)
    real_open = fits.open
    monkeypatch.setattr(psf.fits, "open", lambda *args, **kwargs: real_open(path))
    monkeypatch.setattr(netutils, "read_s3_image_section", lambda hdu, slices, source: np.asarray(hdu.section[slices]))
    for reader in [ep.extract_catalog_psf, ep.extract_grid_psf]:
        result = reader("VIS", 269.48, 67.3, s3_path="dr1/PSF_TILE123.fits", tile_id="123", data_dir=tmp_path / "cache")
        assert result["stamps"].shape == (1, 3, 3)
        np.testing.assert_allclose(result["stamps"].sum(), 1.0)


def test_pipeline_uses_shared_archive_hooks(tmp_path, monkeypatch):
    wcs = make_wcs()
    header = wcs.to_header()
    header["MAGZERO"] = 23.9
    header["MERTILE"] = "123"
    data = np.zeros((25, 25))
    data[12, 12] = 2.0
    cut = cutouts.Cutout("VIS", data, np.ones_like(data)*0.1, wcs, header)
    mer = Table(dict(object_id=np.array([2694800000673000001], dtype=np.int64),
                     ra=[269.48], dec=[67.3], flux_vis_psf=[2.0], flux_vis_sersic=[2.0],
                     point_like_prob=[0.99], is_star=[True], semimajor_axis=[1.0],
                     ellipticity=[0.0], position_angle=[0.0]))
    for name in ["sersic_sersic_vis_radius", "sersic_sersic_vis_index", "sersic_sersic_vis_axis_ratio", "sersic_angle"]:
        mer[name] = [np.nan]
    mer.meta["DR1TABLE"] = "test_dr1"
    psf = dict(stamps=np.ones((1, 1, 1)), ra=np.array([269.48]), dec=np.array([67.3]),
               x=np.array([12.0]), y=np.array([12.0]), fwhm=np.array([0.2]), stmpsize=1, tile_id="123")
    calls = []

    def query(*args, **kwargs):
        calls.append("catalog")
        return mer.copy()

    def fetch(*args, **kwargs):
        calls.append("cutout")
        return cut

    def extract(*args, **kwargs):
        calls.append("psf")
        assert kwargs["tile_id"] == "123"
        return psf

    monkeypatch.setattr(pipeline, "query_mer_catalog", query)
    monkeypatch.setattr(pipeline, "fetch_cutout", fetch)
    monkeypatch.setattr(pipeline, "extract_grid_psf", extract)
    result = ep.run_forced_photometry(
        269.48, 67.3, 2.5, prior={"model_selection": "point", "free_shapes": False},
        target_bands={"euclid": ()}, psf_product="grid", persource_psf=False,
        calibrate_errors=False, data_dir=tmp_path, verbose=False)
    assert calls == ["catalog", "cutout", "psf"]
    assert result.release == ep.get_data_access()
    assert type(result).__module__ == "euclid_phot.pipeline"
    assert len(result.sources) == 1
    assert np.isfinite(result.fluxes_ujy["VIS"]).all()

    # Notebook 01 must find the MER cache produced by the shared driver.
    notebook = json.loads((Path(__file__).resolve().parents[1] /
                           "notebooks/01_nisp_forced_photometry.ipynb").read_text())
    code = next("".join(c["source"]) for c in notebook["cells"]
                if c["cell_type"] == "code" and "mer_path =" in "".join(c["source"]))
    def unexpected_query(*args, **kwargs):
        raise AssertionError("Notebook 01 missed the pipeline's MER cache")
    monkeypatch.setattr(ep, "query_mer_catalog", unexpected_query)
    scope = dict(DATA_DIR=tmp_path, TARGET_RA=269.48, TARGET_DEC=67.3,
                 CUTOUT_SIZE_ARCSEC=2.5, Table=Table, ep=ep)
    exec(compile(code, "notebook01_mer_cache", "exec"), scope)
    assert scope["mer_path"].parent == tmp_path / "catalogs"
    np.testing.assert_array_equal(scope["mer_cat"]["object_id"], mer["object_id"])


def test_ground_prior_uses_shared_hooks_and_reuses_its_catalog(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from euclid_phot.ground_based import prior
    from euclid_phot.ground_based.image import SmallImage

    image = SmallImage(np.zeros((25, 25)), np.ones((25, 25)), make_wcs(),
                       np.ones((3, 3)) / 9, "r")
    calls = []
    mer = Table(dict(object_id=[123], ra=[269.48], dec=[67.3]))

    def query(*args, **kwargs):
        calls.append("catalog")
        return mer.copy()

    def fetch(*args, **kwargs):
        calls.append("cutout")
        assert kwargs["with_flag"]
        return SimpleNamespace(header={"MERTILE": "123"})

    def extract(*args, **kwargs):
        calls.append("psf")
        assert kwargs["tile_id"] == "123"
        return {"stamps": np.ones((1, 3, 3)) / 9}

    def run(*args, **kwargs):
        calls.append("pipeline")
        assert kwargs["mer_catalog"] is inputs.mer
        return "DR1 fitted result"

    monkeypatch.setattr(prior, "query_mer_catalog", query)
    monkeypatch.setattr(prior, "fetch_cutout", fetch)
    monkeypatch.setattr(prior, "extract_grid_psf", extract)
    monkeypatch.setattr(prior, "run_forced_photometry", run)
    inputs = ep.ground_based.prepare_euclid(image, tmp_path)
    assert calls == ["catalog", "cutout", "psf"]
    assert inputs.mer_path.is_file()
    calls.clear()
    cached = ep.ground_based.prepare_euclid(image, tmp_path)
    assert calls == ["cutout", "psf"]
    np.testing.assert_array_equal(cached.mer["object_id"], mer["object_id"])
    assert ep.ground_based.fit_vis(inputs, tmp_path) == "DR1 fitted result"
    assert calls[-1] == "pipeline"
