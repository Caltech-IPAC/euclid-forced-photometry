#!/usr/bin/env python3
"""One-time PSF verification; embed the figure in the notebook itself."""
import argparse
import ast
import base64
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLBACKEND','Agg')
sys.path.insert(0,str(ROOT/'src'))


def embed_psf_comparison(notebook, png):
    """Store a PNG as a Markdown attachment, independent of external files."""
    notebook=Path(notebook)
    nb=json.loads(notebook.read_text())
    for cell in nb['cells']:
        source=''.join(cell['source'])
        if ('validation_png' in source or
                'attachment:psf_validation.png' in source):
            cell_id=cell['id']
            cell.clear()
            cell.update(cell_type='markdown',id=cell_id,metadata={},
                        source=['![PSF vs stellar profile](attachment:psf_validation.png)'],
                        attachments={'psf_validation.png':{'image/png':base64.b64encode(png).decode('ascii')}})
            notebook.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')
            return
    raise ValueError('No PSF comparison cell found in notebook')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    default_notebook=ROOT/'notebooks/04.1_rubin_forced_photometry.ipynb'
    if not default_notebook.is_file():
        default_notebook=ROOT/'notebooks/rubin_forced_photometry.ipynb'
    parser.add_argument('--notebook',type=Path,default=default_notebook)
    parser.add_argument('--size',type=float,default=300.)
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
    import euclid_phot as ep
    from euclid_phot.rubin import RubinClient,RubinConfig,download_field,get_cache_dir
    from euclid_phot.rubin.validation import verify_psfs
    ra,dec=settings['RA'],settings['DEC']
    ep.set_data_access(settings['DATA_ACCESS'])
    cache_dir=get_cache_dir()
    token=settings['RSP_TOKEN']
    client=RubinClient(token,cache_dir=cache_dir,config=RubinConfig(
        base_url=settings['RSP_BASE_URL'],collection=settings['RUBIN_COLLECTION'],object_table=settings['RUBIN_OBJECT_TABLE']))
    try:
        downloaded=download_field(client,ra,dec,settings['SIZE_ARCSEC'],bands=settings['RUBIN_BANDS'],
                                   bad_planes=settings['MASK_BAD_PLANES'])
        cache_dir.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.psf-verification-',dir=cache_dir) as temporary:
            out=Path(temporary)
            verify_psfs(client,token,downloaded,ra=ra,dec=dec,size_arcsec=args.size,output_dir=out)
            embed_psf_comparison(args.notebook,(out/'psf_validation.png').read_bytes())
        print('Updated embedded PSF comparison:',args.notebook)
    except Exception as exc:
        print(type(exc).__name__+': '+str(exc).replace(token,'[REDACTED]'))
        raise SystemExit(1)
    finally:client.close()


if __name__=='__main__':main()
