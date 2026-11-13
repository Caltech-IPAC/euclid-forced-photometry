"""Cross-match local Euclid-prior fluxes with the Rubin coadd Object catalog."""
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from ..units import pixel_scale_arcsec


def numbers(table, key):
    return np.asarray(np.ma.filled(np.ma.asarray(table[key], dtype=float),np.nan))


def good_flag(table,key):
    return ~np.asarray(np.ma.filled(np.ma.asarray(table[key],dtype=bool),True))


def compare_catalog(measured, reference, band, *, max_sep_arcsec=.5,
                    image_wcs=None, image_shape=None, edge_margin_arcsec=3.):
    """Unique sky matches; PSF flux for stars, cModel for galaxies; nJy→µJy.

    Galaxy models differ, so agreement is a diagnostic, not a calibration.
    Acceptance uses positive S/N>5 in both measurements. Signed forced fluxes
    and rejected matches are retained in the returned comparison table.
    Optional image geometry marks proximity to its pixel boundary without
    changing acceptance or fitted measurements. Positions are those of the
    matched forced-photometry sources, not the reference centroids.
    """
    if (image_wcs is None) != (image_shape is None):
        raise ValueError('Provide both image_wcs and image_shape for edge marking')
    if not np.isfinite(edge_margin_arcsec) or edge_margin_arcsec < 0:
        raise ValueError('edge_margin_arcsec must be finite and nonnegative')
    names=('object_id','rubin_object_id','source_type','separation_arcsec',
           'reference_flux_ujy','reference_error_ujy','forced_flux_ujy','forced_error_ujy',
           'reference_clean','unique_match','accepted')
    if not len(measured) or not len(reference):
        out=Table(names=names,dtype=(np.int64,np.int64,'U7',float,float,float,float,float,bool,bool,bool))
        if image_wcs is not None:
            out['edge_distance_arcsec']=np.array([],dtype=float)
            out['near_image_edge']=np.array([],dtype=bool)
            out.meta['EDGEAS']=edge_margin_arcsec
        return out
    ref = SkyCoord(numbers(reference,'coord_ra')*u.deg,numbers(reference,'coord_dec')*u.deg)
    coords=SkyCoord(numbers(measured,'ra')*u.deg,numbers(measured,'dec')*u.deg)
    indices,separation,_=ref.match_to_catalog_sky(coords)
    unique=np.zeros(len(reference),dtype=bool)
    used=set()
    for i in np.argsort(separation.arcsec):
        if int(indices[i]) not in used:
            unique[i]=True;used.add(int(indices[i]))
    star=numbers(reference,'refExtendedness')==0
    classified=np.isin(numbers(reference,'refExtendedness'),[0,1])
    flux=np.where(star,numbers(reference,f'{band}_psfFlux'),numbers(reference,f'{band}_cModelFlux'))/1000
    err=np.where(star,numbers(reference,f'{band}_psfFluxErr'),numbers(reference,f'{band}_cModelFluxErr'))/1000
    clean=np.where(star,good_flag(reference,f'{band}_psfFlux_flag'),good_flag(reference,f'{band}_cModel_flag'))
    clean &= good_flag(reference,f'{band}_pixelFlags_saturatedCenter') & good_flag(reference,f'{band}_pixelFlags_nodata') & classified
    fitted=numbers(measured,f'flux_{band}_ujy')[indices]
    errors=numbers(measured,f'flux_err_{band}_ujy')[indices]
    with np.errstate(divide='ignore',invalid='ignore'):
        accepted=(unique & clean & (separation.arcsec<max_sep_arcsec) &
                  np.isfinite(flux) & np.isfinite(err) & (err>0) & (flux/err>5) &
                  np.isfinite(fitted) & np.isfinite(errors) & (errors>0) & (fitted/errors>5))
    if 'reliable' in measured.colnames:
        accepted &= np.asarray(measured['reliable'][indices],bool)
    out=Table([measured['object_id'][indices],reference['objectId'],np.where(star,'star','galaxy'),
               separation.arcsec,flux,err,fitted,errors,clean,unique,accepted],names=names)
    if 'model' in measured.colnames:
        out['euclid_model']=np.asarray(measured['model'][indices],str)
        out['both_pointlike']=star & (out['euclid_model']=='PointSource')
        # Rubin-seeing point sources can be resolved galaxies in VIS. Keep
        # their comparison, but do not treat them as stellar flux controls.
        out['source_type'][star & ~np.asarray(out['both_pointlike'],bool)]='mixed'
    if image_wcs is not None:
        ny,nx=image_shape
        x,y=image_wcs.world_to_pixel_values(numbers(measured,'ra')[indices],
                                            numbers(measured,'dec')[indices])
        # Outer pixel boundaries are -0.5 and N-0.5; outside sources are marked.
        distance=np.minimum.reduce((x+.5,nx-.5-x,y+.5,ny-.5-y))*pixel_scale_arcsec(image_wcs)
        out['edge_distance_arcsec']=distance
        out['near_image_edge']=distance<edge_margin_arcsec
    out=out[separation.arcsec<max_sep_arcsec]
    out.meta={'BAND':band,'REFERENCE':'Rubin DP1 Object: PSF for stars, cModel for galaxies',
               'MAXSEP':max_sep_arcsec,'WARNING':'Different galaxy models; no flux rescaling applied',
               'MIXED':'Rubin point-like classification but extended Euclid model; not a stellar calibration control'}
    if image_wcs is not None:
        out.meta['EDGEAS']=edge_margin_arcsec
        out.meta['EDGEDEF']='Forced-source distance to nearest Rubin image pixel boundary; negative outside'
    return out
