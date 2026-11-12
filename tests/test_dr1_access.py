"""DR1 access stays isolated while retaining the existing notebook workflow."""
import json
from types import SimpleNamespace

from astropy.io import fits
from astropy.table import MaskedColumn, Table
from astropy.wcs import WCS
from astroquery.ipac.irsa import Irsa
import numpy as np
import pytest

import euclid_phot
import euclid_phot as ep
import euclid_phot.data_access as access
from euclid_phot.euclid import catalog, cutouts, psf


@pytest.fixture(autouse=True)
def selected_release():
    previous = ep.get_data_access()
    ep.set_data_access("DR1")
    yield
    ep.set_data_access(previous)


def test_client_and_shared_functions_leave_q1_unchanged():
    before = (Irsa.tap_url, Irsa.sia_url, Irsa.ssa_url, Irsa._tap, Irsa._sia)
    client = access.get_irsa_client()
    settings = access.load_access_config()["releases"]["DR1"]["irsa"]
    assert client is not Irsa
    assert client._session is not Irsa._session
    assert client.tap_url == settings["tap_url"]
    assert client.sia_url == settings["sia_url"]
    assert (Irsa.tap_url, Irsa.sia_url, Irsa.ssa_url, Irsa._tap, Irsa._sia) == before
    assert ep.build_sources_from_mer is euclid_phot.build_sources_from_mer
    assert ep.fit_nisp_forced is euclid_phot.fit_nisp_forced
    assert ep.get_data_dir() != euclid_phot.config.DEFAULT_DATA_DIR


def test_catalog_missing_morphology_uses_existing_source_fallback(monkeypatch):
    table = Table(dict(
        object_id=np.array([2695000000000000001, 2695000000000000002], dtype=np.int64),
        ra=[269.5, 269.5001], dec=[67.3, 67.3001],
        flux_vis_sersic=[2.0, 3.0], semimajor_axis=[3.0, 4.0],
        ellipticity=[0.2, 0.3], position_angle=[10.0, 20.0]))
    table["point_like_prob"] = MaskedColumn([0.99, 0.99], mask=[False, True])
    calls = []

    def query(adql, *, maxrec):
        calls.append((adql, maxrec))
        return SimpleNamespace(to_table=lambda: table)

    monkeypatch.setattr(catalog, "get_irsa_client", lambda: SimpleNamespace(query_tap=query))
    with pytest.warns(UserWarning, match="Sersic shape columns are masked"):
        result = ep.query_mer_catalog(0.001, 10.0, 0.01, require_detection_band="H")
    adql, maxrec = calls[0]
    assert "euclid_dr1_mer_catalogue_deep_survey" in adql
    assert "euclid_q1" not in adql
    assert "JOIN" not in adql
    assert "m.flux_h_templfit > 0" in adql
    assert "OR m.ra <=" in adql
    assert maxrec == access.load_access_config()["releases"]["DR1"]["maxrec"]
    assert result["object_id"].dtype == np.dtype("int64")
    np.testing.assert_array_equal(result["is_star"], [True, False])
    for name in catalog._MORPHOLOGY_COLUMNS:
        assert result[name].mask.all()
    sources = ep.build_sources_from_mer(result)
    assert [type(source).__name__ for source in sources] == ["PointSource", "ExpGalaxy"]


def test_discovery_selects_dr1_collection_and_same_tile_products(monkeypatch):
    rows = []
    for tile, ra in [(123, 269.48), (124, 269.58)]:
        for kind in ["BGSUB-MOSAIC", "MOSAIC-RMS", "CATALOG-PSF"]:
            filename = f"EUC_MER_{kind}-VIS_TILE{tile}-TEST.fits"
            rows.append(dict(
                energy_bandpassname="VIS", access_url="https://example.org/" + filename,
                cloud_access=json.dumps({"aws": {"bucket_name": "dr1-test", "key": filename}}),
                s_ra=ra, s_dec=67.3))
    table = Table(rows=rows + rows[:1])
    calls = []

    def query_sia(**kwargs):
        calls.append(kwargs)
        return table

    monkeypatch.setattr(cutouts, "get_irsa_client", lambda: SimpleNamespace(query_sia=query_sia))
    products = ep.discover_mer_mosaics(269.48, 67.3, 0.01, bands=("VIS",))
    assert calls[0]["collection"] == access.load_access_config()["releases"]["DR1"]["image_collection"]
    for product in products["VIS"].values():
        assert product["tile_id"] == "123"
        assert len(product["tiles"]) == 2
        assert product["s3"].startswith("dr1-test/")


def test_configured_morphology_is_used_without_masking(tmp_path, monkeypatch):
    settings = access.load_access_config()
    settings["releases"]["DR1"]["catalog_table"] = "dr1_test_mer"
    settings["releases"]["DR1"]["morphology_table"] = "dr1_test_morphology"
    config = tmp_path / "conf.json"
    config.write_text(json.dumps(settings))
    monkeypatch.setattr(access, "CONFIG_PATH", config)
    table = Table({"point_like_prob": [0.5]})
    for name in catalog._MORPHOLOGY_COLUMNS:
        table[name] = [0.75]
    queries = []

    def query(adql, **kwargs):
        queries.append(adql)
        return SimpleNamespace(to_table=lambda: table)

    monkeypatch.setattr(catalog, "get_irsa_client", lambda: SimpleNamespace(query_tap=query))
    result = ep.query_mer_catalog(269.48, 67.3, 0.01)
    assert "FROM dr1_test_mer AS m" in queries[0]
    assert "LEFT JOIN dr1_test_morphology AS morph" in queries[0]
    assert result["sersic_sersic_vis_radius"][0] == 0.75
    assert result.meta["MERMORPH"] == "dr1_test_morphology"


def test_cached_cutouts_and_psfs_do_not_discover_q1_or_dr1(tmp_path, monkeypatch):
    def forbid(*args, **kwargs):
        raise AssertionError("Cached input must not access either archive")

    monkeypatch.setattr(cutouts, "discover_mer_mosaics", forbid)
    from euclid_phot.euclid import cutouts as q1_cutouts
    monkeypatch.setattr(q1_cutouts, "discover_mer_mosaics", forbid)
    ra, dec = 269.48, 67.3
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [ra, dec]
    wcs.wcs.crpix = [3, 3]
    wcs.wcs.cdelt = [-0.1 / 3600, 0.1 / 3600]
    header = fits.Header({"MERTILE": "123", "MAGZERO": 24.5})
    for kind in ("science", "rms", "flag"):
        path = cutouts._cutout_cache_path("VIS", ra, dec, 0.5, tmp_path, kind)
        if kind == "flag":
            cutouts._write_mask_fits(path, np.zeros((5, 5), dtype=np.int32), wcs, header)
        else:
            cutouts._write_cutout_fits(path, np.ones((5, 5)), wcs, header)
    cutout = ep.fetch_cutout(
        "VIS", ra, dec, 0.5, data_dir=tmp_path, with_flag=True)
    assert cutout.shape == (5, 5)
    assert cutout.header["MERTILE"] == "123"

    cache = tmp_path / (
        f"psf_grid_stamps_vis_{ra:.4f}_{dec:.4f}_r60_{psf._PSF_CACHE_VERSION}.npz")
    cache = psf._requested_cache(cache, "123", None)
    np.savez(cache, stamps=np.ones((1, 3, 3)), fwhm=[0.2], tile_id="123")
    result = ep.extract_grid_psf(
        "VIS", ra, dec, tile_id="123", data_dir=tmp_path)
    assert result["stamps"].shape == (1, 3, 3)
