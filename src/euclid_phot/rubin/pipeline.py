"""Local Euclid-prior Rubin fits using the shared ground-based pipeline."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits


class OfficialPSFUnavailable(RuntimeError):
    """A serialized model or catalog approximation is not an evaluated PSF."""


def require_official_psf(downloaded):
    """Compatibility guard: fetch_official_psfs now retrieves exact kernels."""
    if not downloaded:
        raise ValueError("Run the download-data stage first")
    raise OfficialPSFUnavailable(
        "No forced photometry has run; call fetch_official_psfs to retrieve official kernels")


def science_box_mask(ra, dec, center_ra, center_dec, size_arcsec):
    center = SkyCoord(center_ra * u.deg, center_dec * u.deg)
    east, north = center.spherical_offsets_to(SkyCoord(np.asarray(ra) * u.deg, np.asarray(dec) * u.deg))
    return (np.abs(east.arcsec) <= size_arcsec / 2) & (np.abs(north.arcsec) <= size_arcsec / 2)


@dataclass
class RubinRun:
    photometry: object
    downloaded: dict
    psfs: dict
    catalog: object
    comparisons: dict

    def save(self, output_dir):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.catalog.write(out / 'forced_photometry.ecsv', overwrite=True)
        for band, product in self.photometry.ground_results.items():
            self.comparisons[band].write(out / f'{band}_catalog_comparison.ecsv', overwrite=True)
            header = product['wcs'].to_header(relax=True)
            hdus = [fits.PrimaryHDU()]
            for key, unit in [('data','uJy'), ('model','uJy'), ('residual','uJy'), ('invvar','uJy-2')]:
                hdu = fits.ImageHDU(np.asarray(product[key],dtype='f4'), header, name=key.upper())
                hdu.header['BUNIT'] = unit
                hdus.append(hdu)
            fits.HDUList(hdus).writeto(out / f'{band}_fit.fits', overwrite=True)
            psf_header = fits.Header()
            psf_header['PSFMODE'] = 'official RSP evaluated kernel'
            psf_header['DATASET'] = self.psfs[band].dataset_id
            fits.PrimaryHDU(self.psfs[band].center, psf_header).writeto(out / f'{band}_official_psf.fits',overwrite=True)
        return out / 'forced_photometry.ecsv'


def run_forced_photometry(downloaded, psfs, *, ra=59.1, dec=-48.73, size_arcsec=50.,
                          euclid_release='DR1', data_dir=None, n_workers=1):
    """Use the public shared fitter; official PSFs and fixed VIS priors.

    Errors use diagonal variance without empirical correction for correlated
    coadd noise, fixed morphology, PSF mismatch or calibration uncertainty.
    """
    from .. import ground_based as gb
    from ..data_access import get_data_dir, set_data_access
    from ..pipeline import run_forced_photometry as shared_fit
    from .compare import compare_catalog
    from .psf import OfficialPSF
    set_data_access(euclid_release)
    data_dir = Path(data_dir) if data_dir is not None else get_data_dir()
    if set(downloaded) != set(psfs) or not downloaded:
        raise ValueError('Every downloaded band needs an official evaluated PSF')
    inputs = {}
    for band, data in downloaded.items():
        if not isinstance(psfs[band], OfficialPSF) or psfs[band].dataset_id != data.dataset_id:
            raise ValueError('Official PSF provenance does not match its science image')
        if SkyCoord(*psfs[band].positions[0], unit='deg').separation(SkyCoord(ra,dec,unit='deg')).arcsec > .05:
            raise ValueError('Official center PSF was evaluated at a different sky position')
        inputs[band] = data.image.ground_inputs(psfs[band].center)
    euclid = gb.prepare_euclid(next(iter(inputs.values())).image, data_dir, padding_arcsec=10)
    result = shared_fit(
        euclid.ra, euclid.dec, euclid.size_arcsec,
        prior={'band':'VIS','objects':'mer','model_selection':'tree'},
        target_bands={'euclid':(), 'wise':(), 'irac':()}, mer_catalog=euclid.mer,
        ground_images=inputs,
        ground_options={'register_wcs':False, 'psf_mode':'supplied',
                        'recenter_width_pixels':0., 'fit_constant_sky':False},
        psf_product='grid', persource_psf=True, with_flag=True,
        mask_bright_stars=True, calibrate_errors=False,
        data_dir=data_dir, n_workers=n_workers, verbose=True)
    catalog = result.to_table()
    catalog = catalog[science_box_mask(catalog['ra'],catalog['dec'],ra,dec,size_arcsec)]
    catalog.meta.update({'RUBINREL':'dp1','DATAREL':euclid_release,'SIZEAS':size_arcsec,
                         'PSFMODE':'exact official kernel evaluated in RSP; signed pixels preserved',
                         'RUBIN_APCORR_APPLIED':False,
                         'RUBIN_SKY_FIT':False,
                         'RUBIN_FLUX_CONVENTION':'Tractor model amplitudes on calibrated coadd image scale; no model-specific Rubin aperture correction',
                         'RUBIN_ERRORS':'formal diagonal variance; excludes coadd covariance and PSF/model/calibration systematics'})
    comparisons = {}
    for band, data in downloaded.items():
        catalog.meta[f'RUBIN_{band}'] = {'dataset_id':data.dataset_id,
                                        'bad_planes':list(data.image.bad_planes),
                                        'psf_cache':str(psfs[band].path)}
        comparisons[band] = compare_catalog(catalog, data.catalog, band)
    return RubinRun(result, downloaded, psfs, catalog, comparisons)
