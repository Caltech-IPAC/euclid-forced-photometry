"""Compare the old and new step-2 fits on cached Q1 H images, without downloads.

Run from a checkout of this branch with euclid_phot and Tractor installed:
    PYTHONPATH=src python scripts/compare_shape_recovery.py --cache-dir /path/to/q1
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import re
import subprocess
import time
import types
import warnings

import numpy as np
from astropy.table import Table

from euclid_phot import forced
from euclid_phot.euclid.cutouts import Cutout, _cutout_cache_path, _read_cutout_fits
from euclid_phot.euclid.image import build_tractor_image
from euclid_phot.scene import trim_catalog_to_cutout
from euclid_phot.sources import build_sources_from_mer, clone_sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('examples/output/shape_recovery'))
    parser.add_argument('--base', default='d1745a43c88d213805098db1a873862aec94be4c')
    parser.add_argument('--size', type=float, default=100.)
    parser.add_argument('--iterations', type=int, default=30)
    args = parser.parse_args()
    ra, dec, band = 269.48, 67.30, 'H'
    data, wcs, header = _read_cutout_fits(_cutout_cache_path(band, ra, dec, args.size, args.cache_dir/'cutouts', 'science'))
    rms = _read_cutout_fits(_cutout_cache_path(band, ra, dec, args.size, args.cache_dir/'cutouts', 'rms'))[0]
    cutout = Cutout(band, data, rms, wcs, header)
    mer = Table.read(args.cache_dir/'catalogs'/f'mer_catalog_{ra:.4f}_{dec:.4f}_{int(args.size)}.fits')
    mer = trim_catalog_to_cutout(mer, wcs, cutout.shape)
    psf_paths = sorted((args.cache_dir/'psf').glob(f'psf_stamps_h_{ra:.4f}_{dec:.4f}_r*_fits-origin-v2_tile*.npz'))
    if not psf_paths:
        raise FileNotFoundError('A cached CATALOG-PSF for the field is required')
    with np.load(psf_paths[0]) as loaded:
        psf = {k: (loaded[k].item() if loaded[k].ndim == 0 else loaded[k]) for k in loaded.files}
    seeds = build_sources_from_mer(mer, band=band, flux_col='flux_h_sersic',
                                   psf_fwhm_arcsec=float(np.median(psf['fwhm'])),
                                   pixel_scale_arcsec=cutout.pixel_scale_arcsec)
    code = subprocess.check_output(['git', 'show', f'{args.base}:src/euclid_phot/forced.py'], text=True)
    old = types.ModuleType('euclid_phot._shape_recovery_before')
    old.__package__ = 'euclid_phot'
    exec(compile(code, 'upstream/forced.py', 'exec'), old.__dict__)
    report = dict(release='Q1', ra=ra, dec=dec, size_arcsec=args.size, band=band,
                  iterations=args.iterations, sources=len(mer), base=args.base,
                  psf_cache=psf_paths[0].name, models=dict(Counter(type(s).__name__ for s in seeds)))
    rows = Table({'object_id': mer['object_id']})
    for label, implementation in [('before', old), ('after', forced)]:
        sources = copy.deepcopy(seeds)
        tim = build_tractor_image(cutout, psf_data=psf)
        start = time.monotonic()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            tractor, quality = implementation.fit_forced_photometry(
                tim, sources, band=band, initial_fluxes_ujy=np.asarray(mer['flux_h_sersic']))
            print(f'{label}: step 1 complete; {quality.sum()}/{len(quality)} good fluxes', flush=True)
            quality, thawed = implementation.fit_free_shapes(
                tractor, tim, sources, quality, band=band,
                initial_fluxes_ujy=np.asarray(mer['flux_h_sersic']), n_iter=args.iterations)
        radii = np.array([s.getShape().re if hasattr(s, 'getShape') else 0. for s in sources])
        flux = implementation.fluxes_ujy(sources, band)
        # Inspect patch sizes only: do not render a potentially runaway WISE FFT.
        clones = clone_sources(sources, 'W1', psf_halfsize_pix=3, pixel_scale_arcsec=2.75)
        halfsize = [s.halfsize for s in clones if getattr(s, 'halfsize', None) is not None]
        messages = [str(w.message) for w in caught]
        report[label] = dict(seconds=time.monotonic()-start, good_fluxes=int(quality.sum()),
                             thawed=int(thawed), max_radius_arcsec=float(radii.max()),
                             max_wise_halfsize_pix=max(halfsize, default=0),
                             reverted=sum(int(m.group(1)) for msg in messages
                                          if (m := re.search(r'step 2 reverted (\d+)', msg))),
                             warnings=[msg for msg in messages if 'shape' in msg or 'overflow' in msg])
        rows[f'{label}_flux_ujy'] = flux
        rows[f'{label}_radius_arcsec'] = radii
        rows[f'{label}_good'] = quality
        print(label, json.dumps(report[label]), flush=True)
    valid = (rows['before_good'] & rows['after_good'] & np.isfinite(rows['before_flux_ujy'])
             & np.isfinite(rows['after_flux_ujy']) & (np.abs(rows['before_flux_ujy']) > 1e-8))
    delta = np.asarray((rows['after_flux_ujy'][valid]-rows['before_flux_ujy'][valid])/rows['before_flux_ujy'][valid])
    report['flux_comparison'] = dict(common_good=int(valid.sum()), median_fractional_change=float(np.median(delta)),
                                     p95_absolute_fractional_change=float(np.percentile(np.abs(delta), 95)),
                                     count_above_five_percent=int((np.abs(delta) > .05).sum()))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows.write(args.output_dir/'q1_shape_comparison.ecsv', overwrite=True)
    (args.output_dir/'q1_shape_summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['flux_comparison']), flush=True)


if __name__ == '__main__':
    main()
