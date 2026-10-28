"""Runaway prior shapes must never reach the next band's FFT renderer."""
import copy
import warnings

import numpy as np
import pytest
from astropy.wcs import WCS
from tractor import ExpGalaxy, GalaxyShape, NanoMaggies, RaDecPos, Tractor
from tractor.ellipses import EllipseESoft

from euclid_phot import forced
from euclid_phot.sources import clone_sources
from euclid_phot.tractor_image import make_tractor_image


def scene():
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [269.48, 67.3]
    wcs.wcs.crpix = [21, 21]
    wcs.wcs.cdelt = [-0.3/3600, 0.3/3600]
    yy, xx = np.mgrid[-3:4, -3:4]
    stamp = np.exp(-(xx**2 + yy**2)/2)
    tim = make_tractor_image(np.zeros((41,41)), np.ones((41,41))*100,
                             wcs, stamp, band="H")
    sources = [ExpGalaxy(RaDecPos(269.48 + offset, 67.3),
                         NanoMaggies(H=flux), GalaxyShape(radius, 0.8, 10))
               for offset, flux, radius in [(0.,3.,0.4), (0.001,2.,0.5)]]
    tractor = forced.flux_tractor(tim, sources)
    tim.getImage()[:] = tractor.getModelImage(0)
    return tim, sources, tractor


def assert_safe_next_band(tim, sources):
    clones = clone_sources(sources, "W1", psf_halfsize_pix=3,
                            pixel_scale_arcsec=2.75)
    assert all(0 < s.shape.re < 2.72 for s in clones)
    assert all(s.halfsize < 20 for s in clones)
    wise_tim = make_tractor_image(np.zeros(tim.shape), np.ones(tim.shape),
                                  tim.getWcs().wcs._wcs, np.ones((7,7)), band="W1")
    assert np.isfinite(Tractor([wise_tim], clones).getModelImage(0)).all()


def test_recovery_fits_fluxes_without_another_unchecked_shape_pass(monkeypatch):
    tim, sources, tractor = scene()
    calls = []
    def accepted_update():
        # A second free-shape recovery pass would ruin the other galaxy.
        index = min(len(calls), 1)
        if sources[index].isParamThawed("shape"):
            sources[index].shape.re = 1.e6
        calls.append(index)
        return 1., None, None
    monkeypatch.setattr(tractor, "optimize", accepted_update)
    with pytest.warns(UserWarning, match="reverted 1 runaway"):
        quality, count = forced.fit_free_shapes(
            tractor, tim, sources, np.ones(2, dtype=bool), band="H", n_iter=1)
    assert count == 2
    assert calls == [0]
    assert quality.all()
    np.testing.assert_allclose([s.shape.re for s in sources], [0.4,0.5])
    assert all(s.isParamFrozen("shape") for s in sources)
    np.testing.assert_allclose(forced.fluxes_ujy(sources,"H"),np.array([3.,2.])*3.631,rtol=1.e-5)
    assert_safe_next_band(tim, sources)


def test_each_iteration_checks_sizes_including_the_last_update(monkeypatch):
    tim, sources, tractor = scene()
    calls = []
    def accepted_update():
        index = len(calls)
        sources[index].shape.re = 1.e6
        calls.append(index)
        return 1., None, None
    monkeypatch.setattr(tractor, "optimize", accepted_update)
    with pytest.warns(UserWarning, match="reverted 2 runaway"):
        forced.fit_free_shapes(tractor,tim,sources,np.ones(2,dtype=bool),band="H",n_iter=2)
    assert calls == [0,1]
    np.testing.assert_allclose([s.shape.re for s in sources], [0.4,0.5])
    assert_safe_next_band(tim, sources)


def test_overflowing_log_radius_is_restored_before_flux_recovery(monkeypatch):
    tim, sources, tractor = scene()
    sources[0].shape = EllipseESoft.fromRAbPhi(0.4,0.8,10)
    def overflow_update():
        sources[0].shape.logre = 1000.
        raise OverflowError("simulated log-radius overflow")
    monkeypatch.setattr(tractor, "optimize", overflow_update)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        forced.fit_free_shapes(tractor,tim,sources,np.ones(2,dtype=bool),band="H",n_iter=1)
    assert any("overflowed" in str(w.message) for w in caught)
    assert any("reverted 1 runaway" in str(w.message) for w in caught)
    assert sources[0].shape.re == pytest.approx(0.4)
    assert_safe_next_band(tim, sources)


def test_small_real_shape_fit_still_improves_the_likelihood():
    tim, sources, tractor = scene()
    truth = copy.deepcopy(sources)
    truth[0].shape.re = 0.6
    tim.getImage()[:] = Tractor([tim], truth).getModelImage(0)
    before = tractor.getLogProb()
    quality, count = forced.fit_free_shapes(
        tractor,tim,sources,np.ones(2,dtype=bool),band="H",n_iter=8)
    assert count == 2 and quality.all()
    assert tractor.getLogProb() > before
    assert abs(sources[0].shape.re - 0.6) < 0.03
    assert_safe_next_band(tim,sources)
