"""SDSS DR17 cutouts as ground-image inputs, built from the public archive.

Nothing ships with the repository: ``fetch_sdss_example`` takes a position,
a cutout size and a band, asks SkyServer which SDSS frames cover the
position (``nearby_sdss_frames``), downloads the nearest frame with its
PSF model (psField), field calibration (photoField) and PhotoObj catalog,
and writes the four files notebook 04 reads:

- ``*_sci.fits``: the frame pixels (nanomaggies, sky already subtracted);
- ``*_var.fits``: their variance, from the SDSS recipe
  var = ((data / calib + sky) / gain + dark_variance) * calib**2;
- ``*_psf.fits``: the SDSS PSF model at the cutout centre (51 x 51 pixels),
  negative values clipped and normalised to unit sum;
- ``*_cat.ecsv``: SDSS PhotoObj rows (primary mode) inside the cutout.

The cutout is taken from a single frame, so it must fit inside one: frames
are about 13.5 by 9.8 arcmin, and candidates are tried in order of distance
until one contains the whole cutout. The defaults give the notebook's
example, a 200-arcsec g-band cutout at (265.8432, 64.9650), which lands on
run-camcol-field 6122-5-153. Everything is cached, so only the first call
downloads.
"""
from __future__ import annotations

import bz2
import io
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D, NoOverlapError, PartialOverlapError
from astropy.table import Table
from astropy.wcs import WCS
from scipy.interpolate import RectBivariateSpline

from ..netutils import download_file, retry

SAS = "https://data.sdss.org/sas/dr17/eboss"
SKYSERVER = "https://skyserver.sdss.org/dr17/SkyServerWS/SearchTools/SqlSearch"
BANDS = "ugriz"
PIXEL_SCALE = 0.396  # arcsec, SDSS imaging
FRAME_SHAPE = (1489, 2048)  # pixels, every SDSS frame
FRAME_HALF_DIAGONAL_ARCMIN = 8.4  # corner to centre


@dataclass(frozen=True)
class SDSSFrame:
    """One SDSS imaging field: run-camcol-field within a rerun."""
    run: int
    camcol: int
    field: int
    rerun: int = 301

    def __str__(self) -> str:
        return f"{self.run}-{self.camcol}-{self.field}"

    @classmethod
    def from_field_id(cls, field_id: int) -> SDSSFrame:
        """Decode a SkyServer ``fieldID`` (bits: rerun 48-58, run 32-47,
        camcol 29-31, field 16-27)."""
        return cls(run=(field_id >> 32) & 0xFFFF, camcol=(field_id >> 29) & 0x7,
                   field=(field_id >> 16) & 0xFFF, rerun=(field_id >> 48) & 0x7FF)


@dataclass(frozen=True)
class SDSSExample:
    """Paths of the four files ``fetch_sdss_example`` writes, and the frame
    they were cut from."""
    science: Path
    variance: Path
    psf: Path
    catalog: Path
    frame: SDSSFrame

    @property
    def paths(self) -> tuple[Path, Path, Path, Path]:
        return self.science, self.variance, self.psf, self.catalog


def _skyserver_csv(sql: str) -> Table:
    query = urllib.parse.urlencode({"cmd": sql, "format": "csv"})

    def read():
        with urllib.request.urlopen(f"{SKYSERVER}?{query}", timeout=300) as response:
            return response.read().decode()

    text = retry(read, attempts=4, base_delay=5.0, what="SkyServer query")
    return Table.read(text, format="ascii.csv", comment="#")


def nearby_sdss_frames(ra: float, dec: float, radius_arcmin: float = 10.0
                       ) -> list[tuple[SDSSFrame, float]]:
    """SDSS frames whose centre lies within ``radius_arcmin`` of (ra, dec),
    nearest first, each with its centre distance in arcmin.

    Uses SkyServer's ``fGetNearbyFrameEq`` at zoom 0 (native resolution).
    """
    table = _skyserver_csv(
        f"SELECT fieldID, distance FROM dbo.fGetNearbyFrameEq({ra}, {dec}, "
        f"{radius_arcmin}, 0) ORDER BY distance")
    return [(SDSSFrame.from_field_id(int(row["fieldID"])), float(row["distance"]))
            for row in table]


def _sdss_psf(psfield: Path, band_index: int, row: float, col: float):
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


def _download_frame(frame: SDSSFrame, band: str, raw: Path) -> tuple[Path, Path, Path]:
    """The frame image, psField and photoField files, cached under ``raw``."""
    run, rerun, camcol, field = frame.run, frame.rerun, frame.camcol, frame.field
    names = (f"frame-{band}-{run:06d}-{camcol}-{field:04d}.fits.bz2",
             f"psField-{run:06d}-{camcol}-{field:04d}.fit",
             f"photoField-{run:06d}-{camcol}.fits")
    urls = (f"{SAS}/photoObj/frames/{rerun}/{run}/{camcol}/{names[0]}",
            f"{SAS}/photo/redux/{rerun}/{run}/objcs/{camcol}/{names[1]}",
            f"{SAS}/photoObj/{rerun}/{run}/{names[2]}")
    return tuple(retry(lambda u=url, n=name: download_file(u, raw / n, timeout=300),
                       what=name) for url, name in zip(urls, names))


def _frame_cutouts(frame_path: Path, photofield: Path, frame: SDSSFrame, band_index: int,
                   centre: SkyCoord, npix: int):
    """Science and variance cutouts of one frame, plus the frame header.

    Raises ``NoOverlapError``/``PartialOverlapError`` when the cutout does
    not fit inside the frame.
    """
    with fits.open(io.BytesIO(bz2.decompress(frame_path.read_bytes()))) as hdul:
        data = hdul[0].data.astype(float)
        header = hdul[0].header
        calib = hdul[1].data
        sky = hdul[2].data
    wcs = WCS(header)
    cut = Cutout2D(data, centre, (npix, npix), wcs=wcs, mode="strict")

    # Variance in nanomaggies^2: frame pixels back to counts (with the sky),
    # Poisson + dark noise, then calibrated again.
    allsky = sky["ALLSKY"][0]
    spline = RectBivariateSpline(np.arange(allsky.shape[0]), np.arange(allsky.shape[1]),
                                 allsky, kx=1, ky=1)
    sky_counts = spline(sky["YINTERP"][0], sky["XINTERP"][0])
    calib_image = np.tile(calib, (data.shape[0], 1))
    fields = fits.getdata(photofield)
    row = fields[fields["FIELD"] == frame.field][0]
    gain = float(row["GAIN"][band_index])
    dark_variance = float(row["DARK_VARIANCE"][band_index])
    counts = data / calib_image + sky_counts
    variance = (counts / gain + dark_variance) * calib_image ** 2
    var_cut = Cutout2D(variance, centre, (npix, npix), wcs=wcs, mode="strict")
    return cut, var_cut, header


def fetch_sdss_example(data_dir: str | Path, *,
                       ra: float = 265.8432, dec: float = 64.9650,
                       size_arcsec: float = 200.0, band: str = "g",
                       frame: SDSSFrame | tuple[int, int, int] | None = None,
                       force: bool = False) -> SDSSExample:
    """Write SDSS inputs for a square cutout under ``data_dir``; return their paths.

    With ``frame`` unset, the SDSS frames around (ra, dec) are looked up on
    SkyServer and tried nearest first until one contains the whole cutout.
    ``frame`` as ``(run, camcol, field)`` or an :class:`SDSSFrame` skips the
    lookup. The four files are reused when they already exist (unless
    ``force``); the raw SDSS products are kept in ``data_dir/sdss``.
    """
    if band not in BANDS:
        raise ValueError(f"band must be one of {BANDS!r}, got {band!r}")
    largest = min(FRAME_SHAPE) * PIXEL_SCALE
    if size_arcsec > largest:
        raise ValueError(f"size_arcsec={size_arcsec:g} exceeds the {largest:.0f}-arcsec "
                         f"short side of an SDSS frame; the cutout must fit in one frame")
    data_dir = Path(data_dir)
    stem = f"SDSS_{ra:.4f}_{dec:.4f}_{band}_{int(round(size_arcsec))}"
    paths = (data_dir / f"{stem}_sci.fits", data_dir / f"{stem}_var.fits",
             data_dir / f"{stem}_psf.fits", data_dir / f"{stem}_cat.ecsv")
    if not force and all(p.exists() for p in paths):
        header = fits.getheader(paths[0], ext=1)
        return SDSSExample(*paths, frame=SDSSFrame(
            header["RUN"], header["CAMCOL"], header["FIELD"], header["RERUN"]))
    data_dir.mkdir(parents=True, exist_ok=True)

    b = BANDS.index(band)
    raw = data_dir / "sdss"
    npix = int(round(size_arcsec / PIXEL_SCALE))
    centre = SkyCoord(ra, dec, unit="deg")
    if frame is None:
        radius = FRAME_HALF_DIAGONAL_ARCMIN + size_arcsec / 120.0
        candidates = [f for f, _ in nearby_sdss_frames(ra, dec, radius)]
        if not candidates:
            raise ValueError(f"no SDSS DR17 frame within {radius:.1f} arcmin of "
                             f"({ra}, {dec}); is it in the SDSS footprint?")
    else:
        candidates = [frame if isinstance(frame, SDSSFrame) else SDSSFrame(*frame)]

    for frame in candidates:
        print(f"SDSS DR17 frame {frame} ({band}): downloading what is missing")
        frame_path, psfield, photofield = _download_frame(frame, band, raw)
        try:
            cut, var_cut, header = _frame_cutouts(frame_path, photofield, frame, b,
                                                  centre, npix)
            break
        except (NoOverlapError, PartialOverlapError):
            print(f"  cutout does not fit inside frame {frame}; trying the next one")
    else:
        raise ValueError(f"no single SDSS frame contains the {size_arcsec:.0f}-arcsec "
                         f"cutout at ({ra}, {dec}); try a smaller size or a "
                         f"nearby position")

    cut_header = cut.wcs.to_header()
    for key in ("DATE-OBS", "MJD-OBS"):
        if key in header:
            cut_header[key] = header[key]

    sci_header = cut_header.copy()
    sci_header["BUNIT"] = "nanomaggy/pixel"
    centre_ra, centre_dec = cut.wcs.pixel_to_world_values((npix - 1) / 2, (npix - 1) / 2)
    sci_header.update({
        "NJYPDN": 3631.0, "SURVEY": "SDSS DR17", "FILTER": band,
        "RUN": frame.run, "RERUN": frame.rerun, "CAMCOL": frame.camcol,
        "FIELD": frame.field, "SKYSUB": True,
        "PARENT": frame_path.name.removesuffix(".bz2"),
        "CUTSIDE": (float(size_arcsec), "Requested square side, arcsec"),
        "CUTRA": float(centre_ra), "CUTDEC": float(centre_dec)})
    fits.HDUList([fits.PrimaryHDU(),
                  fits.ImageHDU(cut.data.astype(np.float32), sci_header, name="SCIENCE")]
                 ).writeto(paths[0], overwrite=True)
    var_header = cut_header.copy()
    var_header["BUNIT"] = "nanomaggy^2/pixel^2"
    var_header["NOISE"] = "(counts+sky)/gain+darkvar"
    fits.PrimaryHDU(var_cut.data.astype(np.float32), var_header).writeto(
        paths[1], overwrite=True)

    # PSF at the cutout centre, in frame pixel coordinates.
    (y0, y1), (x0, x1) = [(s.start, s.stop) for s in cut.slices_original]
    psf_col, psf_row = (x0 + x1 - 1) / 2, (y0 + y1 - 1) / 2
    psf, negative = _sdss_psf(psfield, b, psf_row, psf_col)
    psf_header = fits.Header()
    psf_header["BUNIT"] = "1"
    psf_header["PSFSRC"] = f"SDSS psField {band}"
    psf_header["PSFX"] = psf_col
    psf_header["PSFY"] = psf_row
    psf_header["NEGCLIP"] = (negative, "negative flux fraction clipped")
    fits.PrimaryHDU(psf.astype(np.float32), psf_header).writeto(paths[2], overwrite=True)

    # The SDSS catalog of the frame, cut to the cutout.
    catalog = _skyserver_csv(
        "SELECT objID, ra, dec, run, rerun, camcol, field, type, clean, mode, "
        f"psfFlux_{band}, psfFluxIvar_{band}, cModelFlux_{band}, cModelFluxIvar_{band} "
        f"FROM PhotoObj WHERE run={frame.run} AND rerun={frame.rerun} "
        f"AND camcol={frame.camcol} AND field={frame.field} AND mode=1")
    x, y = cut.wcs.world_to_pixel_values(np.asarray(catalog["ra"], float),
                                         np.asarray(catalog["dec"], float))
    catalog = catalog[(x >= 0) & (x < npix) & (y >= 0) & (y < npix)]
    catalog.meta = {"SOURCE": "SDSS DR17 SkyServer PhotoObj",
                    "FRAME": str(frame), "CUTSIDE": float(size_arcsec)}
    catalog.write(paths[3], overwrite=True)
    print(f"wrote {stem}_{{sci,var,psf}}.fits and {stem}_cat.ecsv "
          f"({len(catalog)} objects) to {data_dir}")
    return SDSSExample(*paths, frame=frame)
