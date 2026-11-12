"""unWISE stays public even if Astroquery user defaults or Euclid select DR1."""
import numpy as np
import pytest
from types import SimpleNamespace
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astroquery.ipac.irsa import Irsa
from astroquery.ipac.irsa.core import IrsaClass

import euclid_phot as ep
from euclid_phot import data_access
from euclid_phot.wise import catalog


@pytest.fixture(autouse=True)
def restore_release():
    release = ep.get_data_access()
    yield
    ep.set_data_access(release)


@pytest.mark.parametrize("release", ["Q1", "DR1"])
def test_comparison_and_neighbour_queries_ignore_internal_global_client(release, monkeypatch, tmp_path):
    ep.set_data_access(release)
    internal = "https://irsadev.ipac.caltech.edu/TAP"
    monkeypatch.setattr(Irsa, "tap_url", internal)
    # A previously initialized internal TAP object must also be ignored.
    stale_tap = object()
    monkeypatch.setattr(Irsa, "_tap", stale_tap)
    endpoints = []
    sessions = []
    def query(self, query):
        endpoints.append(self.tap_url)
        sessions.append(self._session)
        table = Table(dict(ra=[269.50], dec=[67.3], flux_1=[10.], dflux_1=[1.],
                           flux_2=[20.], dflux_2=[2.], primary_1=[1], primary_2=[1],
                           flags_unwise_1=[0], flags_unwise_2=[0]))
        return SimpleNamespace(to_table=lambda: table)
    monkeypatch.setattr(IrsaClass, "query_tap", query)
    ref = catalog.query_unwise_2019(269.48,67.3,120,cache_dir=tmp_path)
    assert ref["w1_ujy"][0] > 0 and ref["w1_err_ujy"][0] > 0
    assert ref.meta["irsa_endpoint"] == "https://irsa.ipac.caltech.edu/TAP"
    # The exact-query cache must still avoid an additional archive request.
    catalog.query_unwise_2019(269.48,67.3,120,cache_dir=tmp_path)
    neighbours, keep = catalog.unwise_neighbours(
        269.48,67.3,SkyCoord([269.48],[67.3],unit="deg"),
        field_halfwidth_arcsec=1,buffer_halfwidth_arcsec=100,catalog_cache_dir=None)
    assert len(neighbours) == 1 and keep[0]
    assert endpoints == ["https://irsa.ipac.caltech.edu/TAP"]*2
    assert all(session is not Irsa._session for session in sessions)
    assert Irsa.tap_url == internal and Irsa._tap is stale_tap
    assert ep.get_data_access() == release
    # The Euclid client must retain its independently selected endpoint.
    assert data_access.get_irsa_client().tap_url == data_access.get_access_settings()["irsa"]["tap_url"]


def test_public_endpoint_comes_from_configuration(tmp_path, monkeypatch):
    import json
    config=data_access.load_access_config()
    config["public_irsa"]["tap_url"]="https://example.invalid/TAP"
    path=tmp_path/"conf.json"
    path.write_text(json.dumps(config))
    monkeypatch.setattr(data_access,"CONFIG_PATH",path)
    client=data_access.get_public_irsa_client()
    assert client.tap_url == "https://example.invalid/TAP"
