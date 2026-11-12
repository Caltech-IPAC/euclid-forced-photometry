"""Figures from saved or live local Rubin fitting results."""
from pathlib import Path
import numpy as np

PLOT_BANDS = ('u', 'g', 'r', 'i', 'z', 'y')


def show_catalog_comparisons(comparisons):
    """Keep six fixed band rows; omitted bands leave blank panels."""
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(6,2,figsize=(10,21),squeeze=False,constrained_layout=True)
    for row,band in enumerate(PLOT_BANDS):
        if band not in comparisons:
            for ax in axes[row]:
                ax.set_axis_off()
            continue
        table=comparisons[band]
        near_edge=(np.asarray(table['near_image_edge'],bool) if 'near_image_edge' in table.colnames
                   else np.zeros(len(table),dtype=bool))
        accepted=np.asarray(table['accepted'],bool)
        margin=table.meta.get('EDGEAS',3.)
        groups=[(accepted&near_edge,'gray',.2,f'too close to the edg ({margin:g}")')]
        groups += [(accepted&~near_edge&(np.asarray(table['source_type'])==kind),color,1.,kind)
                   for kind,color in [('star','tab:blue'),('galaxy','tab:orange'),('mixed','gray')]]
        for selected,color,alpha,name in groups:
            data=table[selected]
            if not len(data):continue
            ref=np.asarray(data['reference_flux_ujy'],float);fit=np.asarray(data['forced_flux_ujy'],float)
            label=f'{name} (N={len(data)})'
            axes[row,0].scatter(ref,fit,s=16,label=label,color=color,alpha=alpha)
            mag=23.9-2.5*np.log10(ref)
            axes[row,1].scatter(mag,2.5*np.log10(fit/ref),s=16,label=label,color=color,alpha=alpha)
        ax=axes[row,0]
        ax.set(xscale='log',yscale='log',xlabel='Rubin catalog flux (µJy)',ylabel='Euclid-prior forced flux (µJy)',title=f'{band}: catalog comparison')
        if ax.collections:
            lo=max(min(ax.get_xlim()[0],ax.get_ylim()[0]),1e-4);hi=max(ax.get_xlim()[1],ax.get_ylim()[1])
            ax.plot([lo,hi],[lo,hi],'k--',lw=1);ax.legend(fontsize=8)
        axes[row,1].axhline(0,color='k',lw=1)
        axes[row,1].set(xlabel='Rubin reference AB magnitude',ylabel='Rubin − forced magnitude',title=f'{band}: signed photometric offset')
    return fig


def show_fit_images(fits):
    """Show per-band GroundFit objects, or a legacy RubinRun."""
    import matplotlib.pyplot as plt
    if hasattr(fits,'photometry'):
        products=fits.photometry.ground_results
    else:
        products={band:{'data':fit.image.science_ujy,'model':fit.model,
                        'residual':fit.residual,'invvar':fit.image.invvar_ujy}
                  for band,fit in fits.items()}
    fig,axes=plt.subplots(6,3,figsize=(11,21),squeeze=False,constrained_layout=True)
    for row,band in enumerate(PLOT_BANDS):
        if band not in products:
            for ax in axes[row]:
                ax.set_axis_off()
            continue
        data=products[band]
        valid=data['invvar']>0
        lo,hi=np.percentile(data['data'][valid],[5,99])
        limit=max(np.percentile(np.abs(data['residual'][valid]),98),1e-9)
        for ax,key in zip(axes[row],('data','model','residual')):
            opts=dict(cmap='RdBu_r',vmin=-limit,vmax=limit) if key=='residual' else dict(cmap='gray',vmin=lo,vmax=hi)
            ax.imshow(np.where(valid,data[key],np.nan),origin='lower',**opts)
            ax.set_title(f'{band}: {key} (µJy/pixel)')
    return fig
