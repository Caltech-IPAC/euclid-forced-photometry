"""Network helpers shared by every band's data access.

IRSA TAP/SIA, S3 byte-range reads, unwise.me and the Spitzer archives
sometimes fail transiently, so every network call goes through ``retry``:
a few tries with a short wait in between. ``download_file`` writes a file
atomically over verified HTTPS.
"""
from __future__ import annotations

import os
import shutil
import ssl
import time
import urllib.request
from pathlib import Path

# OSError covers socket/timeout errors (s3fs, urllib); pyvo/astroquery raise
# requests exceptions, which subclass OSError in requests>=2.
_DEFAULT_EXCEPTIONS = (OSError, TimeoutError)

# anon=True is required to read the public IRSA S3 bucket.
S3_FSSPEC_KWARGS = {"anon": True}


def retry(fn, *, attempts=3, base_delay=2.0,
          exceptions=_DEFAULT_EXCEPTIONS, what=""):
    """Call ``fn()`` up to ``attempts`` times, waiting between tries.

    ``fn`` is a zero-argument callable (close over its arguments). Only
    ``exceptions`` are retried; a ``ValueError`` (bad request, missing
    product) fails immediately unless listed. The last failure re-raises.
    """
    for k in range(attempts):
        try:
            return fn()
        except exceptions as exc:
            if k == attempts - 1:
                raise
            print(f"  {what or 'network call'} failed ({exc!r}); "
                  f"retry {k + 1}/{attempts - 1} in {base_delay:.0f}s")
            time.sleep(base_delay)


def ensure_https_ca_bundle() -> None:
    """Use certifi only when Python has no usable configured CA store."""
    configured = os.environ.get("SSL_CERT_FILE")
    if configured and Path(configured).is_file():
        return
    paths = ssl.get_default_verify_paths()
    if ((paths.cafile and Path(paths.cafile).is_file())
            or (paths.capath and Path(paths.capath).is_dir())):
        return
    try:
        import certifi
    except ImportError:
        return
    bundle = certifi.where()
    if Path(bundle).is_file():
        os.environ["SSL_CERT_FILE"] = bundle


def https_ssl_context() -> ssl.SSLContext:
    """Return a verified context that also works with aiohttp/fsspec.

    Some Python installations do not have a usable OpenSSL CA path.  Setting
    ``SSL_CERT_FILE`` is enough for urllib, but aiohttp may create its default
    context before that environment variable is set.  Supplying this context
    explicitly keeps remote FITS range reads verified and deterministic.
    """
    ensure_https_ca_bundle()
    configured = os.environ.get("SSL_CERT_FILE")
    if configured and Path(configured).is_file():
        return ssl.create_default_context(cafile=configured)
    return ssl.create_default_context()


def download_file(url: str, path: str | Path, *, force: bool = False,
                  timeout: float = 120.0) -> Path:
    """Download a file atomically over verified HTTPS."""
    path = Path(path)
    if path.exists() and not force:
        return path
    ensure_https_ca_bundle()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response, \
                temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path
