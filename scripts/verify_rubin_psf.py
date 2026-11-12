#!/usr/bin/env python3
"""Development-only PSF verification. Notebook users only view its saved PNG."""
import argparse
import ast
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLBACKEND','Agg')
sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--notebook',type=Path,default=ROOT/'notebooks/04.1_rubin_forced_photometry.ipynb')
    parser.add_argument('--size',type=float,default=300.)
    parser.add_argument('--redraw',action='store_true',help='Redraw from saved inputs without API access')
    args=parser.parse_args()
    settings={}
    for cell in json.loads(args.notebook.read_text())['cells']:
        if cell['cell_type']!='code':continue
        for node in ast.parse(''.join(cell['source'])).body:
            if not isinstance(node,ast.Assign):continue
            try:value=ast.literal_eval(node.value)
            except (ValueError,TypeError):continue
            for target in node.targets:
                if isinstance(target,ast.Name):settings[target.id]=value
                elif isinstance(target,ast.Tuple) and isinstance(value,(tuple,list)):
                    for name,item in zip(target.elts,value):
                        if isinstance(name,ast.Name):settings[name.id]=item
    from euclid_phot.rubin import RubinClient,RubinConfig,download_field,saved_psf_validation_dir
    from euclid_phot.rubin.validation import verify_psfs,plot_psf_validation
    ra,dec=settings['RA'],settings['DEC']
    out=saved_psf_validation_dir(settings['RUBIN_COLLECTION'],ra,dec)
    if args.redraw:
        print(plot_psf_validation(out));return
    token=settings['RSP_TOKEN']
    client=RubinClient(token,cache_dir=ROOT/'examples/data/rubin',config=RubinConfig(
        base_url=settings['RSP_BASE_URL'],collection=settings['RUBIN_COLLECTION'],object_table=settings['RUBIN_OBJECT_TABLE']))
    try:
        downloaded=download_field(client,ra,dec,settings['SIZE_ARCSEC'],bands=settings['RUBIN_BANDS'],
                                   bad_planes=settings['MASK_BAD_PLANES'])
        summary=verify_psfs(client,token,downloaded,ra=ra,dec=dec,size_arcsec=args.size,output_dir=out)
        summary.pprint_all();print('Saved PSF figure and numerical inputs:',out)
    except Exception as exc:
        print(type(exc).__name__+': '+str(exc).replace(token,'[REDACTED]'))
        raise SystemExit(1)
    finally:client.close()


if __name__=='__main__':main()
