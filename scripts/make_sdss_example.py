"""Rebuild the SDSS example inputs of notebook 04 from the public SDSS archive.

    python scripts/make_sdss_example.py --out sdss_example

Downloads one SDSS DR17 frame with its PSF model (psField), its field
calibration (photoField) and the PhotoObj catalog, and writes the four
files notebook 04 reads, for a square cutout centred on (--ra, --dec):

- ``*_sci.fits``: the frame pixels (nanomaggies, sky already subtracted);
- ``*_var.fits``: their variance, from the SDSS recipe
  var = ((data / calib + sky) / gain + dark_variance) * calib**2;
- ``*_psf.fits``: the SDSS PSF model at the cutout centre (51 x 51 pixels),
  negative values clipped and normalised to unit sum;
- ``*_cat.ecsv``: SDSS PhotoObj rows (primary mode) inside the cutout.

The defaults reproduce the example in ``examples/data/ground_based``: the
science, variance and catalog files match exactly. The bundled PSF stamp
was trimmed further by its author (its outer pixels are zero); this script
writes the standard SDSS reconstruction instead, which differs from the
bundled stamp by up to ~6% of the peak value.

Use other --run/--camcol/--field/--band values to make the same files for
another SDSS frame; the frame must contain the requested cutout.
"""
from __future__ import annotations

import argparse
import bz2
import io
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.table import Table
from astropy.wcs import WCS
from scipy.interpolate import RectBivariateSpline

SAS = "https://data.sdss.org/sas/dr17/eboss"
SKYSERVER = "https://skyserver.sdss.org/dr17/SkyServerWS/SearchTools/SqlSearch"
BANDS = "ugriz"
PIXEL_SCALE = 0.396  # arcsec, SDSS imaging


def read_url(url: str, attempts: int = 4) -> bytes:
    """GET with a few retries (SkyServer and the SAS sometimes answer 503)."""
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=300) as response:
                return response.read()
        except OSError:
            if attempt == attempts - 1:
                raise
            time.sleep(5 * (attempt + 1))


def fetch(url: str, path: Path) -> Path:
    """Download once into ``path``."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(read_url(url))
    return path


def sdss_psf(psfield: Path, band_index: int, row: float, col: float):
    """The SDSS KL-basis PSF model at frame pixel (row, col), unit sum, and
    the fraction of negative flux clipped from it.

    Follows the SDSS recipe for psField files: each principal component's
    amplitude is a polynomial in (row, col) / 1000.
    """
    table = fits.getdata(psfield, ext=band_index + 1)
    psf = 0.0
    for k in range(len(table)):
        nrow, ncol = int(table["NROW_B"][k]), int(table["NCOL_B"][k])
        rnrow, rncol = int(table["RNROW"][k]), int(table["RNCOL"][k])
        c = table["C"][k]
        amplitude = sum(c[i % nrow, i // nrow]
                        * (row * 1e-3) ** (i % nrow) * (col * 1e-3) ** (i // nrow)
                        for i in range(nrow * ncol))
        psf = psf + amplitude * np.asarray(table["RROWS"][k], float).reshape(rnrow, rncol)
    clipped = np.clip(psf, 0.0, None)
    return clipped / clipped.sum(), float(-psf[psf < 0].sum() / psf.sum())


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ra", type=float, default=265.8432)
    parser.add_argument("--dec", type=float, default=64.9650)
    parser.add_argument("--size", type=float, default=200.0, help="cutout side, arcsec")
    parser.add_argument("--band", default="g", choices=list(BANDS))
    parser.add_argument("--run", type=int, default=6122)
    parser.add_argument("--rerun", type=int, default=301)
    parser.add_argument("--camcol", type=int, default=5)
    parser.add_argument("--field", type=int, default=153)
    parser.add_argument("--out", type=Path, default=Path("sdss_example"),
                        help="output folder (default: %(default)s)")
    parser.add_argument("--cache", type=Path, default=None,
                        help="download folder (default: <out>/downloads)")
    args = parser.parse_args(argv)

    b = BANDS.index(args.band)
    run, rerun, camcol, field = args.run, args.rerun, args.camcol, args.field
    cache = args.cache or args.out / "downloads"
    frame_name = f"frame-{args.band}-{run:06d}-{camcol}-{field:04d}.fits.bz2"
    frame_path = fetch(f"{SAS}/photoObj/frames/{rerun}/{run}/{camcol}/{frame_name}",
                       cache / frame_name)
    psfield = fetch(f"{SAS}/photo/redux/{rerun}/{run}/objcs/{camcol}/"
                    f"psField-{run:06d}-{camcol}-{field:04d}.fit",
                    cache / f"psField-{run:06d}-{camcol}-{field:04d}.fit")
    photofield = fetch(f"{SAS}/photoObj/{rerun}/{run}/photoField-{run:06d}-{camcol}.fits",
                       cache / f"photoField-{run:06d}-{camcol}.fits")

    with fits.open(io.BytesIO(bz2.decompress(frame_path.read_bytes()))) as frame:
        data = frame[0].data.astype(float)
        header = frame[0].header
        calib = frame[1].data
        sky = frame[2].data
    wcs = WCS(header)

    # Variance in nanomaggies^2: frame pixels back to counts (with the sky),
    # Poisson + dark noise, then calibrated again.
    allsky = sky["ALLSKY"][0]
    spline = RectBivariateSpline(np.arange(allsky.shape[0]), np.arange(allsky.shape[1]),
                                 allsky, kx=1, ky=1)
    sky_counts = spline(sky["YINTERP"][0], sky["XINTERP"][0])
    calib_image = np.tile(calib, (data.shape[0], 1))
    fields = fits.getdata(photofield)
    row = fields[fields["FIELD"] == field][0]
    gain, dark_variance = float(row["GAIN"][b]), float(row["DARK_VARIANCE"][b])
    counts = data / calib_image + sky_counts
    variance = (counts / gain + dark_variance) * calib_image ** 2

    npix = int(round(args.size / PIXEL_SCALE))
    centre = SkyCoord(args.ra, args.dec, unit="deg")
    cut = Cutout2D(data, centre, (npix, npix), wcs=wcs, mode="strict")
    var_cut = Cutout2D(variance, centre, (npix, npix), wcs=wcs, mode="strict")
    cut_header = cut.wcs.to_header()
    for key in ("DATE-OBS", "MJD-OBS"):
        if key in header:
            cut_header[key] = header[key]

    stem = f"SDSS_{args.ra:.4f}_{args.dec:.4f}_{args.band}_{int(round(args.size))}"
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    sci_header = cut_header.copy()
    sci_header["BUNIT"] = "nanomaggy/pixel"
    centre_ra, centre_dec = cut.wcs.pixel_to_world_values((npix - 1) / 2, (npix - 1) / 2)
    sci_header.update({
        "NJYPDN": 3631.0, "SURVEY": "SDSS DR17", "FILTER": args.band,
        "RUN": run, "RERUN": rerun, "CAMCOL": camcol, "FIELD": field,
        "SKYSUB": True, "PARENT": frame_name.removesuffix(".bz2"),
        "CUTSIDE": (float(args.size), "Requested square side, arcsec"),
        "CUTRA": float(centre_ra), "CUTDEC": float(centre_dec)})
    fits.HDUList([fits.PrimaryHDU(),
                  fits.ImageHDU(cut.data.astype(np.float32), sci_header, name="SCIENCE")]
                 ).writeto(out / f"{stem}_sci.fits", overwrite=True)
    var_header = cut_header.copy()
    var_header["BUNIT"] = "nanomaggy^2/pixel^2"
    var_header["NOISE"] = "(counts+sky)/gain+darkvar"
    fits.PrimaryHDU(var_cut.data.astype(np.float32), var_header).writeto(
        out / f"{stem}_var.fits", overwrite=True)

    # PSF at the cutout centre, in frame pixel coordinates.
    (y0, y1), (x0, x1) = [(s.start, s.stop) for s in cut.slices_original]
    psf_col, psf_row = (x0 + x1 - 1) / 2, (y0 + y1 - 1) / 2
    psf, negative = sdss_psf(psfield, b, psf_row, psf_col)
    psf_header = fits.Header()
    psf_header["BUNIT"] = "1"
    psf_header["PSFSRC"] = f"SDSS psField {args.band}"
    psf_header["PSFX"] = psf_col
    psf_header["PSFY"] = psf_row
    psf_header["NEGCLIP"] = (negative, "negative flux fraction clipped")
    fits.PrimaryHDU(psf.astype(np.float32), psf_header).writeto(
        out / f"{stem}_psf.fits", overwrite=True)

    # The SDSS catalog of the frame, cut to the cutout.
    sql = ("SELECT objID, ra, dec, run, rerun, camcol, field, type, clean, mode, "
           f"psfFlux_{args.band}, psfFluxIvar_{args.band}, "
           f"cModelFlux_{args.band}, cModelFluxIvar_{args.band} FROM PhotoObj "
           f"WHERE run={run} AND rerun={rerun} AND camcol={camcol} "
           f"AND field={field} AND mode=1")
    query = urllib.parse.urlencode({"cmd": sql, "format": "csv"})
    catalog = Table.read(read_url(f"{SKYSERVER}?{query}").decode(),
                         format="ascii.csv", comment="#")
    x, y = cut.wcs.world_to_pixel_values(np.asarray(catalog["ra"], float),
                                         np.asarray(catalog["dec"], float))
    catalog = catalog[(x >= 0) & (x < npix) & (y >= 0) & (y < npix)]
    catalog.meta = {"SOURCE": "SDSS DR17 SkyServer PhotoObj",
                    "FRAME": f"{run}-{camcol}-{field}", "CUTSIDE": float(args.size)}
    catalog.write(out / f"{stem}_cat.ecsv", overwrite=True)
    print(f"wrote {stem}_{{sci,var,psf}}.fits and {stem}_cat.ecsv "
          f"({len(catalog)} objects) to {out}")


if __name__ == "__main__":
    main()
