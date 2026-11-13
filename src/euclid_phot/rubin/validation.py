"""One-time official/empirical PSF diagnostics; never substitutes a fitting PSF."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import sep
from astropy.table import Table
from scipy.spatial import cKDTree

from ..ground_based.psf_validation import _star_stamp, _core_fwhm
from scipy.ndimage import shift
from ..units import pixel_scale_arcsec
from .compare import numbers, good_flag
from .image import load_coadd
from .psf import fetch_official_psfs
from .plots import PLOT_BANDS


def _aligned_reference(psf, x, y):
    """Apply the shared sampling convention without clipping signed lobes."""
    dx, dy = x - round(x), y - round(y)
    sampled = shift(np.pad(psf, 3), (dy, dx), order=3,
                    mode="constant", cval=0.)[3:-3, 3:-3]
    aligned = shift(sampled, (-dy, -dx), order=3, mode="constant", cval=0.)
    return aligned / aligned.sum()


def _reference_star_comparison(image, x, y, *, min_snr=30.):
    """Use shared observed-star extraction and Rubin's signed model kernel."""
    ry, rx = np.array(image.psf.shape) // 2
    pair = _star_stamp(image, x, y, ry, rx, min_snr)
    if pair is None:
        return None
    star, _ = pair
    reference = _aligned_reference(image.psf, x, y)
    yy, xx = np.indices(star.shape)
    core = (xx-rx)**2 + (yy-ry)**2 <= 10.**2
    amplitude = np.sum(star[core]*reference[core]) / np.sum(reference[core]**2)
    return star, amplitude*reference


def select_validation_stars(image, catalog, band, *, min_snr=30, isolation_arcsec=3.5):
    """Match clean, isolated Rubin point sources to measured image centroids."""
    names=('object_id','ra','dec','x_detected','y_detected','catalog_snr')
    rows=Table(names=names,dtype=(np.int64,float,float,float,float,float))
    valid=image.valid
    data=np.ascontiguousarray(np.where(np.isfinite(image.science),image.science,0),dtype=float)
    rms=np.ascontiguousarray(np.where(valid,image.rms,np.nanmedian(image.rms[valid])),dtype=float)
    background=sep.Background(data,mask=~valid)
    det=sep.extract(data-background.back(),8,err=rms,mask=~valid,minarea=5)
    det=det[(det['flag']==0)&(det['a']>0)&(det['b']/np.maximum(det['a'],1e-6)>.6)]
    if not len(det) or not len(catalog):return rows
    x,y=image.wcs.world_to_pixel_values(numbers(catalog,'coord_ra'),numbers(catalog,'coord_dec'))
    finite=np.isfinite(x)&np.isfinite(y)
    xy=np.column_stack((x[finite],y[finite]))
    neighbors=np.full(len(catalog),np.inf)
    if len(xy)>1:neighbors[finite]=cKDTree(xy).query(xy,k=2)[0][:,1]
    with np.errstate(divide='ignore',invalid='ignore'):
        snr=numbers(catalog,f'{band}_psfFlux')/numbers(catalog,f'{band}_psfFluxErr')
    good=(finite&(numbers(catalog,'refExtendedness')==0)&np.isfinite(snr)&(snr>min_snr)&
          (neighbors>isolation_arcsec/pixel_scale_arcsec(image.wcs)))
    for key in (f'{band}_psfFlux_flag',f'{band}_pixelFlags_nodata',f'{band}_pixelFlags_saturatedCenter'):
        good &= good_flag(catalog,key)
    indices=np.flatnonzero(good)
    if not len(indices):return rows
    measured=np.column_stack((det['x'],det['y']))
    distances,nearest=cKDTree(measured).query(np.column_stack((x[indices],y[indices])))
    used=set()
    for index,distance,match in sorted(zip(indices,distances,nearest),key=lambda a:a[1]):
        if distance>3 or match in used:continue
        px,py=measured[match]
        if min(px,py,image.science.shape[1]-px,image.science.shape[0]-py)<20:continue
        used.add(match)
        sky=image.wcs.pixel_to_world(px,py)
        rows.add_row((int(catalog['objectId'][index]),sky.ra.deg,sky.dec.deg,px,py,snr[index]))
    return rows


def radial_profile(stamp):
    yy,xx=np.indices(stamp.shape)
    r=np.hypot(xx-(stamp.shape[1]-1)/2,yy-(stamp.shape[0]-1)/2)
    bins=np.floor(r).astype(int)
    n=min(stamp.shape)//2
    return np.array([np.mean(stamp[bins==i]) for i in range(n)])


def verify_psfs(client, token, downloaded, *, ra, dec, size_arcsec=300., output_dir):
    """Save raw per-star data, radial profiles, official kernels and a PNG.

    The larger region is only for validation. Use the exact same coadd dataset
    as the small science fit; stars outside its available pixels are rejected.
    Kernels are evaluated at each stellar position to distinguish PSF spatial
    variation from disagreement between the model and observed stellar shape.
    """
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    images,catalogs,stars,positions={},{},{},{}
    for band,science in downloaded.items():
        rows=client.coadds(ra,dec,size_arcsec,band)
        selected=[row for row in rows if int(row['lsst_tract'])==science.tract and int(row['lsst_patch'])==science.patch]
        if not selected:raise RuntimeError('Validation field has no matching science coadd patch')
        row=selected[0]
        if str(row['obs_publisher_did'])!=science.dataset_id:
            raise RuntimeError('Validation image and science coadd dataset IDs differ')
        path=client.cutout(row,ra,dec,size_arcsec)
        images[band]=load_coadd(path,band,bad_planes=science.image.bad_planes)
        catalogs[band]=client.objects(ra,dec,size_arcsec,band,tract=science.tract,patch=science.patch)
        stars[band]=select_validation_stars(images[band],catalogs[band],band)
        positions[band]=[[ra,dec]]+[[float(s['ra']),float(s['dec'])] for s in stars[band]]
        print(f'{band}: {len(stars[band])} candidate stars in the {size_arcsec:g} arcsec validation region',flush=True)
    psfs=fetch_official_psfs(client,token,downloaded,positions)
    summaries=[]
    for band,data in downloaded.items():
        ground=images[band].ground_inputs(psfs[band].center).image
        observed,modeled,accepted=[],[],[]
        for i,star in enumerate(stars[band]):
            local=replace(ground,psf=psfs[band].stamps[i+1])
            pair=_reference_star_comparison(local,float(star['x_detected']),float(star['y_detected']))
            if pair is None:continue
            observed.append(pair[0]);modeled.append(pair[1]);accepted.append(i)
        count=len(observed)
        used=stars[band][accepted]
        used.write(out/f'{band}_stars.ecsv',overwrite=True)
        shape=psfs[band].center.shape
        observed=np.array(observed) if count else np.empty((0,*shape))
        modeled=np.array(modeled) if count else np.empty((0,*shape))
        empirical=np.median(observed,axis=0) if count else np.full(shape,np.nan)
        official=np.median(modeled,axis=0) if count else np.full(shape,np.nan)
        scale=pixel_scale_arcsec(ground.wcs)
        profiles=np.array([radial_profile(a) for a in observed])
        model_profiles=np.array([radial_profile(a) for a in modeled])
        radius=(np.arange(min(shape)//2)+.5)*scale
        np.savez_compressed(out/f'{band}_psf_validation.npz',
            observed_stamps=observed,official_stamps=modeled,
            empirical_median=empirical,official_median=official,
            residual_median=empirical-official,model_signed_pixels_preserved=True,
            empirical_mean=np.mean(observed,axis=0) if count else np.full(shape,np.nan),
            official_mean=np.mean(modeled,axis=0) if count else np.full(shape,np.nan),
            official_center=psfs[band].center,
            radius_arcsec=radius,observed_profiles=profiles,official_profiles=model_profiles,
            pixel_scale_arcsec=scale,field_size_arcsec=size_arcsec,
            dataset_id=data.dataset_id,ra=ra,dec=dec)
        summaries.append((band,count,_core_fwhm(official)*scale if count else np.nan,
                          _core_fwhm(empirical)*scale if count else np.nan,
                          _core_fwhm(psfs[band].center)*scale,
                          'limited' if count<4 else 'profile comparison available'))
    summary=Table(rows=summaries,names=('band','stars','official_core_fwhm_arcsec','empirical_core_fwhm_arcsec',
                                       'official_center_core_fwhm_arcsec','status'))
    summary.meta={'RA':ra,'DEC':dec,'VALIDATION_SIZE_ARCSEC':size_arcsec,
                  'METHOD':'Median recentered, sky-subtracted stellar stamps; each official kernel evaluated at that star',
                  'NORMALIZATION':'Observed stamps normalized by core flux; official amplitudes fitted in the core',
                  'LIMIT':'Empirical PSF is diagnostic only; few stars and morphology/PSF spatial variations limit the comparison'}
    summary.write(out/'psf_validation_summary.ecsv',overwrite=True)
    plot_psf_validation(out)
    return summary


def plot_psf_validation(output_dir, *, bands=None, save_dir=None):
    """Redraw archived profiles in fixed slots; omitted bands stay blank.

    Subsets get their own PNG, preserving the full saved verification. No API
    access, star selection, PSF evaluation or photometry is performed here.
    """
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from astropy.visualization import simple_norm
    out=Path(output_dir);summary=Table.read(out/'psf_validation_summary.ecsv')
    records={str(record['band']):record for record in summary}
    active=set(records) if bands is None else set(bands)
    if active-set(PLOT_BANDS):
        raise ValueError('Unknown Rubin bands in PSF display')
    fig=Figure(figsize=(15,13),layout='constrained')
    FigureCanvasAgg(fig)
    axes=fig.subplots(4,3,squeeze=False)
    for index,band in enumerate(PLOT_BANDS):
        row,col=2*(index//3),index%3
        image_ax,profile_ax=axes[row,col],axes[row+1,col]
        if band not in active or band not in records:
            image_ax.set_axis_off()
            profile_ax.set_axis_off()
            continue
        record=records[band];n=int(record['stars'])
        with np.load(out/f'{band}_psf_validation.npz',allow_pickle=False) as data:
            center=data['official_center']
            emp=data['empirical_median'] if 'empirical_median' in data else data['empirical_mean']
            official=data['official_median'] if 'official_median' in data else data['official_mean']
            scale=float(data['pixel_scale_arcsec'])
            half_y,half_x=np.array(center.shape)*scale/2
            extent=(-half_x,half_x,-half_y,half_y)
            im=image_ax.imshow(center,origin='lower',extent=extent,cmap='viridis',
                                  norm=simple_norm(center,stretch='asinh'))
            fig.colorbar(im,ax=image_ax,shrink=.8,label='Fraction of kernel flux / pixel')
            image_ax.set_title(f'{band}: official center PSF')
            image_ax.set(xlabel='Δx (arcsec)',ylabel='Δy (arcsec)')
            if n:
                radius=data['radius_arcsec'];profiles=data['observed_profiles']
                ax=profile_ax
                ax.plot(radius,radial_profile(official),label='Official, at star positions')
                ax.plot(radius,radial_profile(emp),'o-',ms=3,label='Empirical stellar median')
                if n>1:
                    lower,upper=np.percentile(profiles,[16,84],axis=0)
                    ax.fill_between(radius,lower,upper,alpha=.2,label='16–84% among stars')
                ax.set_yscale('symlog',linthresh=1e-4)
                ax.set(xlabel='Radius (arcsec)',ylabel='Core-normalized mean per pixel')
                ax.legend(fontsize=8)
            else:
                profile_ax.text(.5,.5,'No usable stars',ha='center',transform=profile_ax.transAxes)
            profile_ax.set_title(f'{band}: N={n} stars'+(' (limited)' if n<4 else ''))
    fig.suptitle('PSF vs stellar profile')
    selected=tuple(b for b in PLOT_BANDS if b in active)
    suffix='' if bands is None or selected==PLOT_BANDS else '_'+(''.join(selected) or 'none')
    path=(Path(save_dir) if save_dir is not None else out)/f'psf_validation{suffix}.png'
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=160)
    return path
