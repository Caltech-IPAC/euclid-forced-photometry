"""External Rubin DP1 access through TAP, SIA and SODA; no Rubin stack."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import re
from urllib.parse import urlsplit

import numpy as np
import pyvo
import requests
from astropy import units as u
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from pyvo.dal.adhoc import DatalinkResults, SodaQuery

from ..netutils import ensure_https_ca_bundle
from .paths import get_cache_dir


@dataclass(frozen=True)
class RubinConfig:
    base_url: str = "https://data.lsst.cloud"
    collection: str = "dp1"
    object_table: str = "dp1.Object"

    def __post_init__(self):
        import re
        if urlsplit(self.base_url).scheme != "https":
            raise ValueError("Rubin base_url must use HTTPS")
        if not re.fullmatch(r"[A-Za-z0-9_]+", self.collection):
            raise ValueError("invalid Rubin collection")
        if not re.fullmatch(r"[A-Za-z0-9_]+\.[A-Za-z0-9_]+", self.object_table):
            raise ValueError("invalid Rubin object_table")


class _TokenAuth(requests.auth.AuthBase):
    """Attach the token only to its RSP origin, including on DataLink requests."""
    def __init__(self, token, base_url):
        self._token = token
        self._origin = urlsplit(base_url)[:2]

    def __call__(self, request):
        if urlsplit(request.url)[:2] == self._origin:
            request.headers["Authorization"] = f"Bearer {self._token}"
        return request


class _Session(requests.Session):
    def request(self, method, url, **kwargs):
        kwargs.setdefault("timeout", (30, 300))
        response = super().request(method, url, **kwargs)
        if response.status_code == 401:
            raise requests.HTTPError(
                "RSP rejected the token (401). Use the complete token shown once "
                "after creation, starting with gt-, rather than the token-list key. "
                "Also check that it has not expired or been revoked, and that it "
                "was created on the configured RSP site.", response=response)
        if response.status_code == 403:
            raise requests.HTTPError(
                "RSP denied access (403). Check the token's read:tap and read:image "
                "scopes and your access to the selected data collection.", response=response)
        return response


def validate_token(token):
    """Reject an incomplete Gafaelfawr token locally, without echoing its value."""
    if not isinstance(token, str) or not token.strip() or token.strip() in {
            "USER INPUT THEIR RSP TOKEN", "YOUR_TOKEN"}:
        raise ValueError("Paste your complete RSP API token into the first code cell")
    token = token.strip()
    if not re.fullmatch(r"gt-[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+", token):
        raise ValueError(
            "RSP_TOKEN is not a complete Rubin API token. Copy the full token "
            "shown immediately after creating it: it starts with gt- and includes "
            "a key and secret separated by a period. The key displayed in the "
            "token list cannot authenticate. Create a new token with read:tap "
            "and read:image if you did not save the full value.")
    return token


def validate_field(ra, dec, size_arcsec):
    if (not np.all(np.isfinite([ra, dec, size_arcsec])) or
            not 0 <= ra < 360 or not -90 < dec < 90 or size_arcsec <= 0):
        raise ValueError("need finite RA [0,360), Dec (-90,90), and positive box size")


def validate_band(band):
    if band not in tuple("ugrizy"):
        raise ValueError("Rubin band must be one of u, g, r, i, z, y")


class RubinClient:
    """Token-authenticated VO client with local, token-independent data caches."""
    def __init__(self, token: str, *, cache_dir: str | Path | None = None,
                 config: RubinConfig | None = None):
        token = validate_token(token)
        ensure_https_ca_bundle()
        self.config = config or RubinConfig()
        self.cache_dir = Path(cache_dir) if cache_dir is not None else get_cache_dir()
        self.session = _Session()
        self.session.auth = _TokenAuth(token.strip(), self.config.base_url)
        base = self.config.base_url.rstrip("/")
        self.tap = pyvo.dal.TAPService(f"{base}/api/tap", session=self.session)
        self.sia = pyvo.dal.SIA2Service(
            f"{base}/api/sia/{self.config.collection}/query",
            session=self.session, check_baseurl=False)
        self._columns = None

    def close(self):
        self.session.close()

    def check_connection(self, *, required_scopes=("read:tap", "read:image")):
        """Validate the current token and its scopes against RSP, never a cache."""
        base = self.config.base_url.rstrip("/")
        response = self.session.get(f"{base}/auth/api/v1/token-info")
        response.raise_for_status()
        info = response.json()
        scopes = set(info.get("scopes", []))
        missing = set(required_scopes) - scopes
        if missing:
            raise ValueError(f"RSP token lacks required scopes: {', '.join(sorted(missing))}. "
                             "Create a new token with read:tap and read:image.")
        return {"scopes": sorted(scopes)}

    def _path(self, kind, key, suffix):
        context = f"{self.config.base_url}|{self.config.collection}|{key}"
        digest = sha256(context.encode()).hexdigest()[:24]
        folder = self.cache_dir / self.config.collection / kind
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"{digest}{suffix}"

    def query(self, adql, *, maxrec=20000):
        """Run an asynchronous TAP query, reject truncation, cache successful results."""
        path = self._path("tap", f"{maxrec}|{adql}", ".ecsv")
        if path.exists():
            return Table.read(path)
        result = self.tap.run_async(adql, maxrec=maxrec)
        if getattr(result, "query_status", "OK") == "OVERFLOW":
            raise RuntimeError("Rubin TAP result truncated; reduce the search area")
        table = result.to_table()
        if len(table) >= maxrec:
            raise RuntimeError("Rubin TAP reached MAXREC; reduce the search area")
        table.meta = {}  # Avoid serializing service internals or temporary URLs.
        temporary = path.with_suffix(".ecsv.part")
        table.write(temporary, format="ascii.ecsv", overwrite=True)
        temporary.replace(path)
        return table

    def object_columns(self):
        if self._columns is None:
            rows = self.query(
                "SELECT column_name FROM TAP_SCHEMA.columns "
                f"WHERE table_name = '{self.config.object_table}'")
            self._columns = {str(c) for c in rows["column_name"]}
            if not self._columns:
                self._columns = None
                raise ValueError(f"TAP exposes no columns for {self.config.object_table}; "
                                 "check RUBIN_OBJECT_TABLE and collection access")
        return self._columns

    def objects(self, ra, dec, size_arcsec, band, *, tract=None, patch=None):
        """Reference catalog measurements belonging to the selected coadd patch."""
        validate_field(ra, dec, size_arcsec)
        validate_band(band)
        required = ["objectId", "coord_ra", "coord_dec", "tract", "patch",
                    "refExtendedness", f"{band}_psfFlux", f"{band}_psfFluxErr",
                    f"{band}_psfFlux_flag", f"{band}_cModelFlux",
                    f"{band}_cModelFluxErr", f"{band}_cModel_flag",
                    f"{band}_pixelFlags_saturatedCenter",
                    f"{band}_pixelFlags_nodata"]
        missing = set(required) - self.object_columns()
        if missing:
            raise RuntimeError(
                "This Rubin table does not expose the required catalog columns: "
                f"{sorted(missing)}.")
        radius = size_arcsec / 3600 / np.sqrt(2)
        where = ("CONTAINS(POINT('ICRS', coord_ra, coord_dec), "
                 f"CIRCLE('ICRS', {ra:.12f}, {dec:.12f}, {radius:.12f})) = 1")
        for name, value in (("tract", tract), ("patch", patch)):
            if value is not None:
                where += f" AND {name} = {int(value)}"
        return self.query(f"SELECT {', '.join(required)} "
                          f"FROM {self.config.object_table} WHERE {where}")

    def coadds(self, ra, dec, size_arcsec, band):
        validate_field(ra, dec, size_arcsec)
        validate_band(band)
        # SIA BAND is wavelength, not a filter name; filter its Rubin band column.
        rows = self.sia.search(
            pos=(ra, dec, size_arcsec / 3600 / np.sqrt(2)),
            calib_level=3, dpsubtype="lsst.deep_coadd", maxrec=1000).to_table()
        if len(rows) >= 1000:
            raise RuntimeError("SIA search may be truncated; reduce the search area")
        if "lsst_band" not in rows.colnames:
            raise RuntimeError("SIA response missing lsst_band")
        return rows[np.array([str(v) == band for v in rows["lsst_band"]])]

    def _soda(self, datalink, mode, ra, dec, size_arcsec):
        query = SodaQuery.from_resource(
            datalink, datalink.get_adhocservice_by_id(mode), session=self.session)
        query.circle = (ra * u.deg, dec * u.deg, size_arcsec / 7200 * u.deg)
        stream = query.execute_stream()
        try:
            data = stream.read()
            query.raise_if_error()
        finally:
            stream.close()
        return fits.open(BytesIO(data), memmap=False)

    def cutout_services(self, row):
        """Inspect the service IDs actually advertised for this image."""
        links = DatalinkResults.from_result_url(str(row["access_url"]), session=self.session)
        return [str(service.ID) for service in links.iter_adhocservices()]

    def cutout(self, row, ra, dec, size_arcsec):
        """Return cached FITS with science WCS, variance, and dynamic mask bit names.

        Fetch the lightweight science+WCS and masked-image modes separately.
        MaskedImageF alone need not carry WCS; full ExposureF can carry enormous
        serialized coadd PSF/provenance tables which this client does not need.
        """
        validate_field(ra, dec, size_arcsec)
        identity = str(row["obs_publisher_did"] if "obs_publisher_did" in row.colnames
                       else row["access_url"])
        path = self._path("cutouts", f"{identity}|{ra:.12f}|{dec:.12f}|{size_arcsec:.8f}", ".fits")
        if path.exists():
            return path
        links = DatalinkResults.from_result_url(str(row["access_url"]), session=self.session)
        with self._soda(links, "cutout-sync", ra, dec, size_arcsec) as sci, \
                self._soda(links, "cutout-sync-maskedimage", ra, dec, size_arcsec) as masked:
            science = sci[1]
            mask_hdu = masked["MASK"] if "MASK" in masked else masked[2]
            var_hdu = masked["VARIANCE"] if "VARIANCE" in masked else masked[3]
            pixels = masked["IMAGE"] if "IMAGE" in masked else masked[1]
            if (science.data.shape != pixels.data.shape or
                    not np.allclose(science.data, pixels.data, rtol=0, atol=0, equal_nan=True)):
                raise RuntimeError("SODA science and masked-image cutouts are not aligned")
            if not WCS(science.header).has_celestial:
                raise RuntimeError("Science cutout has no FITS celestial WCS")
            unit = science.header.get("BUNIT", sci[0].header.get("BUNIT", ""))
            if not unit or not u.Unit(unit).is_equivalent(u.nJy):
                raise RuntimeError(f"Expected calibrated nJy image, got BUNIT={unit!r}")
            header = science.header.copy()
            header["BUNIT"] = str(unit)
            header["SURVEY"] = f"Rubin {self.config.collection}"
            temporary = path.with_suffix(".fits.part")
            fits.HDUList([
                fits.PrimaryHDU(), fits.ImageHDU(science.data, header, name="SCIENCE"),
                fits.ImageHDU(mask_hdu.data, mask_hdu.header, name="MASK"),
                fits.ImageHDU(var_hdu.data, var_hdu.header, name="VARIANCE"),
            ]).writeto(temporary, overwrite=True)
            temporary.replace(path)
        return path
