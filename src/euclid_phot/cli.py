"""Command-line interface: ``euclid-phot run``.

    euclid-phot run --ra 269.48 --dec 67.30 --size 50 --out catalog.ecsv

Measures a single cutout with the default MER-prior workflow. Add your own
ground-based image with ``--ground-image``:

    euclid-phot run --ra 265.8432 --dec 64.965 --size 220 --bands "" \
        --ground-image sci.fits --ground-variance var.fits \
        --ground-psf psf.fits --ground-unit nanomaggy --ground-band g
"""
from __future__ import annotations

import argparse
import sys

from .data_access import get_data_access, set_data_access


def _add_common(p: argparse.ArgumentParser):
    p.add_argument("--data-access", choices=("Q1", "DR1"), default=get_data_access(),
                   help="Euclid release (default: %(default)s)")
    p.add_argument("--data-dir", default=None,
                   help="cache directory (default: the selected release's cache)")
    p.add_argument("--prior-band", default="VIS",
                   choices=("VIS", "Y", "J", "H"))
    p.add_argument("--objects", default="mer", choices=("mer", "free"),
                   help="source-list mode (user coords are API-only)")
    p.add_argument("--bands", default="Y,J,H",
                   help="comma-separated Euclid target bands "
                        "(default: %(default)s; '' for prior band only)")
    p.add_argument("--wise", default="",
                   help="comma-separated WISE bands (W1,W2); default none")
    p.add_argument("--irac", default="",
                   help="comma-separated IRAC channels (IRAC1,...,IRAC4); "
                        "default none")
    p.add_argument("--irac-backend", default="seip",
                   choices=("seip", "dawn", "pbcd"),
                   help="IRAC mosaics (default: %(default)s)")
    p.add_argument("--no-calibrate-errors", action="store_true",
                   help="skip the empirical error calibration")
    p.add_argument("--psf-product", default="auto",
                   choices=("auto", "catalog", "grid"))
    p.add_argument("--workers", type=int, default=1,
                   help="threads inside one cutout fit (default 1)")
    g = p.add_argument_group("ground-based image (optional)")
    g.add_argument("--ground-image", help="science FITS (first image HDU)")
    noise = g.add_mutually_exclusive_group()
    noise.add_argument("--ground-rms", help="1-sigma uncertainty FITS")
    noise.add_argument("--ground-variance", help="variance FITS")
    g.add_argument("--ground-psf", help="PSF stamp FITS (odd-sized)")
    g.add_argument("--ground-seeing", type=float,
                   help="FWHM in arcsec for a Gaussian PSF, if no --ground-psf")
    g.add_argument("--ground-unit", default="uJy",
                   help="flux of one pixel value, e.g. nanomaggy, uJy, "
                        "'4.7 nJy' (default %(default)s)")
    g.add_argument("--ground-band", default="ground",
                   help="band label for the output columns (default %(default)s)")
    g.add_argument("--ground-background", default="none", choices=("none", "sep"),
                   help="subtract a SEP sky map first (default %(default)s)")


def _parse_bands(arg: str) -> tuple:
    return tuple(b.strip() for b in arg.split(",") if b.strip())


def _read_image(path):
    """The first 2-D image HDU of a FITS file and its header."""
    import numpy as np
    from astropy.io import fits
    with fits.open(path) as hdus:
        for hdu in hdus:
            if hdu.data is not None and np.ndim(hdu.data) == 2:
                return np.asarray(hdu.data, dtype=float), hdu.header.copy()
    raise ValueError(f"{path} has no 2-D image HDU")


def _ground_images(args) -> dict | None:
    if not args.ground_image:
        return None
    import numpy as np
    from astropy import units as u
    from astropy.wcs import WCS

    from .ground_based import gaussian_psf, make_image
    from .units import pixel_scale_arcsec

    if not (args.ground_rms or args.ground_variance):
        raise SystemExit("--ground-image needs --ground-rms or --ground-variance")
    science, header = _read_image(args.ground_image)
    if args.ground_rms:
        rms = _read_image(args.ground_rms)[0]
    else:
        rms = np.sqrt(np.clip(_read_image(args.ground_variance)[0], 0, None))
    wcs = WCS(header).celestial
    if args.ground_psf:
        psf = _read_image(args.ground_psf)[0]
    elif args.ground_seeing:
        psf = gaussian_psf(args.ground_seeing, pixel_scale_arcsec(wcs))
    else:
        raise SystemExit("--ground-image needs --ground-psf or --ground-seeing")
    unit = (u.Quantity(args.ground_unit) if any(c.isdigit() for c in args.ground_unit)
            else u.Unit(args.ground_unit))
    inputs = make_image(science, rms, psf, unit, wcs, band=args.ground_band)
    return {args.ground_band: inputs}


def _build_kwargs(args) -> dict:
    return dict(
        prior={"band": args.prior_band, "objects": args.objects},
        target_bands={"euclid": _parse_bands(args.bands),
                      "wise": _parse_bands(args.wise),
                      "irac": _parse_bands(args.irac)},
        irac_options={"mosaic_backend": args.irac_backend},
        data_dir=args.data_dir,
        calibrate_errors=not args.no_calibrate_errors,
        psf_product=args.psf_product,
        n_workers=args.workers,
        ground_images=_ground_images(args),
        ground_options={"background_method": args.ground_background},
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="euclid-phot",
        description="Forced photometry on Euclid Q1/DR1 (+unWISE, IRAC, your own "
                    "ground image) with Tractor.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="measure a single cutout")
    p_run.add_argument("--ra", type=float, required=True)
    p_run.add_argument("--dec", type=float, required=True)
    p_run.add_argument("--size", type=float, default=50.0,
                       help="cutout side, arcsec (default %(default)s)")
    p_run.add_argument("--out", default="catalog.ecsv",
                       help="output table path; format from the extension "
                            "(.ecsv/.fits/.parquet)")
    _add_common(p_run)

    args = parser.parse_args(argv)

    if args.command == "run":
        set_data_access(args.data_access)
        from .pipeline import run_forced_photometry
        result = run_forced_photometry(args.ra, args.dec, args.size,
                                       **_build_kwargs(args))
        tab = result.to_table()
        if args.out.endswith(".fits"):
            # FITS headers cannot carry the nested metadata; sanitize.
            tab = tab.copy()
            tab.meta = {}
        tab.write(args.out, overwrite=True)
        print(f"wrote {len(tab)} sources to {args.out}")
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
