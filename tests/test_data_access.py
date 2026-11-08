"""Release selection changes archive access and cache roots without copying readers."""
import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astroquery.ipac.irsa import Irsa

import euclid_phot as ep
from euclid_phot import data_access as access
from euclid_phot.euclid import catalog, cutouts, psf


@pytest.fixture(autouse=True)
def selected_release():
    previous = ep.get_data_access()
    ep.set_data_access("Q1")
    yield
    ep.set_data_access(previous)


def test_switch_validation_and_existing_imports(monkeypatch, tmp_path):
    monkeypatch.setenv("EUCLID_PHOT_DATA_DIR", str(tmp_path))
    query = ep.query_mer_catalog
    fetch = ep.fetch_cutout
    ep.set_data_access("Q1")
    q1 = ep.get_data_dir()
    ep.set_data_access("DR1")
    dr1 = ep.get_data_dir()
    assert query is ep.query_mer_catalog and fetch is ep.fetch_cutout
    assert q1 == tmp_path / "q1"
    assert dr1 == tmp_path / "dr1/deep_survey"
    with ThreadPoolExecutor(1) as executor:
        assert executor.submit(ep.get_data_access).result() == "DR1"
    for bad in ("dr1", "Q2", None, []):
        with pytest.raises(ValueError, match="DATA_ACCESS"):
            ep.set_data_access(bad)
        assert ep.get_data_access() == "DR1"


@pytest.mark.parametrize("release", ["Q1", "DR1"])
def test_client_endpoints_and_global_irsa_unchanged(release):
    before = (Irsa.tap_url, Irsa.sia_url, Irsa.ssa_url, Irsa._tap, Irsa._sia)
    ep.set_data_access(release)
    client = access.get_irsa_client()
    settings = access.get_access_settings()["irsa"]
    assert (client.tap_url, client.sia_url, client.ssa_url) == tuple(settings[k] for k in ("tap_url", "sia_url", "ssa_url"))
    assert client._session is not Irsa._session
    assert (Irsa.tap_url, Irsa.sia_url, Irsa.ssa_url, Irsa._tap, Irsa._sia) == before


def test_same_catalog_query_routes_q1_and_dr1(monkeypatch):
    statements = []
    def query(adql, **kwargs):
        assert "\nFROM " in adql
        assert "sersic_angleFROM" not in adql
        statements.append(adql)
        t = Table(dict(object_id=np.array([123], dtype=np.int64), ra=[269.48], dec=[67.3], point_like_prob=[0.99]))
        if ep.get_data_access() == "Q1":
            for name in catalog._MORPHOLOGY_COLUMNS:
                t[name] = [0.75]
        return SimpleNamespace(to_table=lambda: t)
    monkeypatch.setattr(catalog, "get_irsa_client", lambda: SimpleNamespace(query_tap=query))
    ep.set_data_access("Q1")
    q1 = ep.query_mer_catalog(269.48, 67.3, 0.001)
    ep.set_data_access("DR1")
    with pytest.warns(UserWarning, match="shape columns are masked"):
        dr1 = ep.query_mer_catalog(269.48, 67.3, 0.001)
    assert "FROM euclid_q1_mer_catalogue AS m" in statements[0]
    assert "LEFT JOIN euclid_q1_mer_morphology" in statements[0]
    assert "FROM euclid_dr1_mer_catalogue_deep_survey AS m" in statements[1]
    assert "JOIN" not in statements[1]
    assert q1.meta["DATAREL"] == "Q1" and dr1.meta["DATAREL"] == "DR1"
    assert q1["sersic_sersic_vis_radius"][0] == 0.75
    assert dr1["sersic_sersic_vis_radius"].mask.all()


def test_same_discovery_routes_release_in_worker_threads(monkeypatch):
    queries = []
    def query_sia(**kwargs):
        queries.append(kwargs["collection"])
        return Table(names=("energy_bandpassname", "access_url", "cloud_access", "s_ra", "s_dec"), dtype=("U4", "U80", "U80", float, float))
    monkeypatch.setattr(cutouts, "get_irsa_client", lambda: SimpleNamespace(query_sia=query_sia))
    for release in ("Q1", "DR1"):
        ep.set_data_access(release)
        with ThreadPoolExecutor(1) as executor:
            assert executor.submit(ep.discover_mer_mosaics, 269.48, 67.3, 0.001).result() == {}
    assert queries == ["euclid_DpdMerBksMosaic", "euclid_dr1_deep_DpdMerBksMosaic"]


def test_default_cutout_cache_is_separate_after_repeated_switches(tmp_path, monkeypatch):
    monkeypatch.setenv("EUCLID_PHOT_DATA_DIR", str(tmp_path))
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [269.48, 67.3]
    wcs.wcs.crpix = [3, 3]
    wcs.wcs.cdelt = [-0.1/3600, 0.1/3600]
    calls = []
    def discover(*args, **kwargs):
        calls.append(ep.get_data_access())
        return {"VIS": {kind: {"tile_id": "123", "kind": kind} for kind in ("science", "rms", "flag")}}
    def read(tile, *args):
        value = 1 if ep.get_data_access() == "Q1" else 2
        return np.full((5, 5), value if tile["kind"] == "science" else 1), wcs, fits.Header({"MAGZERO":24.5, "MERTILE":"123"})
    monkeypatch.setattr(cutouts, "discover_mer_mosaics", discover)
    monkeypatch.setattr(cutouts, "_single_tile_cutout", read)
    for release, expected in [("Q1",1), ("DR1",2), ("Q1",1), ("DR1",2)]:
        ep.set_data_access(release)
        c = ep.fetch_cutout("VIS",269.48,67.3,0.5,with_flag=True)
        np.testing.assert_array_equal(c.data,expected)
    assert calls == ["Q1", "DR1"]


@pytest.mark.parametrize("reader", [ep.extract_catalog_psf, ep.extract_grid_psf])
def test_default_psf_cache_is_separate_after_repeated_switches(reader, tmp_path, monkeypatch):
    from euclid_phot import netutils
    monkeypatch.setenv("EUCLID_PHOT_DATA_DIR", str(tmp_path / "cache"))
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [269.48, 67.3]
    wcs.wcs.crpix = [3, 3]
    wcs.wcs.cdelt = [-0.1/3600, 0.1/3600]
    header = wcs.to_header()
    header["STMPSIZE"] = 3
    for release, fwhm in [("Q1", 0.2), ("DR1", 0.3)]:
        table = Table(dict(x_center=[3], y_center=[3], x=[3.0], y=[3.0],
                           RA=[269.48], Dec=[67.3], FWHM=[fwhm]))
        fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(np.ones((5,5)), header),
                      fits.BinTableHDU(table)]).writeto(tmp_path / f"{release}.fits")
    opens = []
    real_open = fits.open
    def open_psf(*args, **kwargs):
        opens.append(ep.get_data_access())
        return real_open(tmp_path / f"{ep.get_data_access()}.fits")
    monkeypatch.setattr(psf.fits, "open", open_psf)
    monkeypatch.setattr(netutils, "read_s3_image_section",
                        lambda hdu, slices, source: np.asarray(hdu.section[slices]))
    for release, expected in [("Q1",0.2), ("DR1",0.3), ("Q1",0.2), ("DR1",0.3)]:
        ep.set_data_access(release)
        result = reader("VIS",269.48,67.3,s3_path="test/PSF_TILE123.fits",tile_id="123")
        np.testing.assert_allclose(result["fwhm"],expected)
    assert opens == ["Q1", "DR1"]


# Official Caltech-IPAC Markdown at commit d1745a43c88d213805098db1a873862aec94be4c.
MARKDOWN_SHA256 = {'06_photometry_at_coordinates.ipynb': 'a93f234f3fdb5d03f36d8a4bf5645b48499a8a07396593d96aaa429525972fa4', '02_wise_forced_photometry.ipynb': 'b789dc576d9a9c8cf74d89f6bc40dd5318a28fcf42130638feb3a0d3d475bd29', '03_irac_forced_photometry.ipynb': '5ac423011527bcb2be80eda82bd4d664bfc91fa0a7e278edb984c70688587f6a', '01_nisp_forced_photometry.ipynb': 'b4b486a80e34308eec3f58060e2c500d8fdc13505051cb6e3906bd6de813812b', '05_one_call.ipynb': '346e62c7cf2d633f901f8ac73d5b1f403bd51537c4c6de7d305c9246c047df2e', '00_how_tractor_works.ipynb': '0ab946df1a20d7c3ff52338ab342cc915fae94d4a2937dc95fb6d134d6a3081b', '04_ground_based_forced_photometry.ipynb': '615f9370c9c3dde2096710898a6a9f7ae01d675f83d5a441ab375158bd01c477', 'supporting_notebooks/injection_recovery.ipynb': '411e385c1dee2838abda37dfee2f69108f3e2765b4777e042cdf63799816411b', 'supporting_notebooks/model_selection.ipynb': '5a1b9327e5381c7646d80f57293b7b4013e78958b628ac3f29dfe60eef1ec783'}


@pytest.mark.parametrize("name", sorted(MARKDOWN_SHA256))
def test_notebook_selector_markdown_and_shared_calls(name):
    root = Path(__file__).resolve().parents[1]
    nb = json.loads((root/"notebooks"/name).read_text())
    first_cell = next(c for c in nb["cells"] if c["cell_type"] == "code")
    first = "".join(first_cell["source"])
    tree = ast.parse(first)
    selector_nodes = [n for n in tree.body if
                      (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "DATA_ACCESS" for t in n.targets))
                      or (isinstance(n, ast.Import) and any(a.name == "euclid_phot" for a in n.names))
                      or (isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == "ep.set_data_access")]
    selector = ast.unparse(ast.Module(body=selector_nodes, type_ignores=[]))
    selected = next(n.value.value for n in tree.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "DATA_ACCESS" for t in n.targets))
    assert selected in {"Q1", "DR1"}
    assert "ep.set_data_access(DATA_ACCESS)" in first
    markdown = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown"]
    assert hashlib.sha256(json.dumps(markdown,ensure_ascii=False).encode()).hexdigest() == MARKDOWN_SHA256[name]
    for release in ("Q1", "DR1"):
        exec(compile(selector.replace(f"DATA_ACCESS = {selected!r}", f"DATA_ACCESS = {release!r}"), name, "exec"), {"__name__":"__main__"})
        assert ep.get_data_access() == release
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":continue
        tree = ast.parse("".join(cell["source"]))
        for n in ast.walk(tree):
            if isinstance(n, ast.Call):
                label = ast.unparse(n.func)
                assert not label.endswith("_DR1")
                for kw in n.keywords:
                    if kw.arg in {"force", "force_download", "force_fit"}:
                        assert not (isinstance(kw.value, ast.Constant) and kw.value.value is True)
            if isinstance(n, ast.Import):
                assert all(alias.name != "euclid_phot.dr1" for alias in n.names)


def test_configuration_ships_inside_package():
    assert access.CONFIG_PATH.parent == Path(access.__file__).resolve().parent
    assert access.CONFIG_PATH.is_file()
    assert set(access.load_access_config()["releases"]) == {"Q1", "DR1"}


@pytest.mark.parametrize("release", ["Q1", "DR1"])
def test_notebook01_reads_existing_mer_cache_without_query(release, tmp_path, monkeypatch):
    ep.set_data_access(release)
    monkeypatch.setenv("EUCLID_PHOT_DATA_DIR", str(tmp_path))
    calls = []
    def query(*args, **kwargs):
        calls.append(args)
        return Table({"object_id": [123], "ra": [269.48], "dec": [67.3]})
    monkeypatch.setattr(ep, "query_mer_catalog", query)
    path = Path(__file__).resolve().parents[1] / "notebooks/01_nisp_forced_photometry.ipynb"
    notebook = json.loads(path.read_text())
    code = next("".join(c["source"]) for c in notebook["cells"]
                if c["cell_type"] == "code" and "mer_path =" in "".join(c["source"]))
    body = []
    for node in ast.parse(code).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "cutouts" for t in node.targets):
            break
        body.append(node)
    query_cell = compile(ast.Module(body=body, type_ignores=[]), str(path), "exec")
    scope = dict(DATA_DIR=ep.get_data_dir(), TARGET_RA=269.48, TARGET_DEC=67.3,
                 CUTOUT_SIZE_ARCSEC=200., Table=Table, ep=ep)
    exec(query_cell, scope)
    cached = scope["mer_path"]
    assert not cached.exists() and len(calls) == 1
    # A cache supplied by another workflow is still reused by notebook 01.
    cached.parent.mkdir(parents=True, exist_ok=True)
    scope["mer_cat"].write(cached)
    # Simulate a new kernel: no MER table survives, but the disk cache does.
    del scope["mer_cat"]
    exec(query_cell, scope)
    assert len(calls) == 1
    np.testing.assert_array_equal(scope["mer_cat"]["object_id"], [123])


@pytest.mark.parametrize("release", ["Q1", "DR1", "public"])
def test_irsa_clients_apply_the_longer_request_timeout(release, monkeypatch):
    from requests.sessions import Session
    requests = []
    def request(self, method, url, **kwargs):
        requests.append(kwargs)
        return object()
    monkeypatch.setattr(Session, "request", request)
    client = access.get_public_irsa_client() if release == "public" else access.get_irsa_client(release)
    client._session.request("GET", "https://example.invalid")
    assert requests[0]["timeout"] == 600


def test_notebook04_output_location_is_independent_of_release(tmp_path, monkeypatch):
    base = tmp_path / "data"
    monkeypatch.setenv("EUCLID_PHOT_DATA_DIR", str(base))
    monkeypatch.setattr(ep.config, "DEFAULT_DATA_DIR", base)
    path = Path(__file__).resolve().parents[1] / "notebooks/04_ground_based_forced_photometry.ipynb"
    notebook = json.loads(path.read_text())
    first = next("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    assignments = [n for n in ast.parse(first).body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id in {"DATA_DIR", "GROUND_DIR", "OUTPUT_DIR"}
                           for t in n.targets)]
    code = compile(ast.Module(body=assignments, type_ignores=[]), str(path), "exec")
    for release in ("Q1", "DR1"):
        ep.set_data_access(release)
        scope = dict(ep=ep)
        exec(code, scope)
        expected_data = base / "q1" if release == "Q1" else base / "dr1/deep_survey"
        assert scope["DATA_DIR"] == expected_data
        assert scope["GROUND_DIR"] == expected_data / "ground_based"
        assert scope["OUTPUT_DIR"] == tmp_path / "output/ground_based"
