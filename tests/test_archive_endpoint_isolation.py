"""Public IRAC discovery and internal inspection must not leak across releases."""

import pytest
from astropy.table import Table
from astroquery.ipac.irsa import Irsa
from astroquery.ipac.irsa.core import IrsaClass

import euclid_phot as ep
from euclid_phot import data_access
from euclid_phot.irac import archive, data


@pytest.fixture(autouse=True)
def restore_release():
    previous=ep.get_data_access()
    yield
    ep.set_data_access(previous)


@pytest.mark.parametrize("release", ["Q1", "DR1"])
def test_irac_sha_and_seip_remain_public_with_internal_user_defaults(release, monkeypatch):
    ep.set_data_access(release)
    monkeypatch.setattr(Irsa,"sia_url","https://irsadev.ipac.caltech.edu/SIA")
    stale=object()
    monkeypatch.setattr(Irsa,"_sia",stale)
    requests=[]
    def query(self,**kwargs):
        requests.append((self.sia_url,kwargs["collection"],self._session))
        if kwargs["collection"]=="spitzer_sha":
            base="https://example/r15562496/ch1/pbcd/"
            root="SPITZER_I1_15562496_0000_5"
            return Table({"access_url":[base+root+suffix for suffix in
                ["_E8607838_maic.fits","_A43829484_munc.fits",
                 "_A43829485_mcov.fits","_A43535696_refptg.tbl"]]})
        return Table(dict(energy_bandpassname=["IRAC1"],access_url=[
            "https://example/123.4-567.IRAC.1.mosaic.fits"]))
    monkeypatch.setattr(IrsaClass,"query_sia",query)
    products=archive.discover_pbcd(10.,20.,"IRAC1",aorkey=15562496)
    assert len(products)==1 and products[0].aorkey==15562496
    assert products[0].refptg_url.endswith("_refptg.tbl")
    assert data._seip_science_products(10.,20.,"IRAC1")=={
        "123.4-567":"https://example/123.4-567.IRAC.1.mosaic.fits"}
    assert [(url,collection) for url,collection,_ in requests]==[
        ("https://irsa.ipac.caltech.edu/SIA","spitzer_sha"),
        ("https://irsa.ipac.caltech.edu/SIA","spitzer_seip")]
    assert all(session is not Irsa._session for _,_,session in requests)
    assert Irsa._sia is stale and "irsadev" in Irsa.sia_url
    assert ep.get_data_access()==release


def test_explicit_dr1_client_does_not_switch_the_active_q1_release():
    ep.set_data_access("Q1")
    before=(Irsa.tap_url,Irsa.sia_url,Irsa.ssa_url,Irsa._tap,Irsa._sia)
    dr1=data_access.get_irsa_client("DR1")
    assert dr1.tap_url==data_access.get_access_settings("DR1")["irsa"]["tap_url"]
    assert ep.get_data_access()=="Q1"
    assert (Irsa.tap_url,Irsa.sia_url,Irsa.ssa_url,Irsa._tap,Irsa._sia)==before
