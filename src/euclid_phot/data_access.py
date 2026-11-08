"""Release selection for the shared Euclid readers and fitting pipeline.

Call set_data_access("Q1") or set_data_access("DR1") before a run. This setting
belongs to the Python process, so archive worker threads use the same release.
Do not change it during a running fit. Endpoints and release identifiers live
in euclid_phot/conf.json; no date triggers a release or endpoint change automatically.
"""
from __future__ import annotations

from functools import partial
import json
import os
from pathlib import Path

from .config import DEFAULT_DATA_DIR
from .netutils import ensure_https_ca_bundle

CONFIG_PATH = Path(__file__).resolve().with_name("conf.json")
_SELECTED_RELEASE = None


def load_access_config():
    with CONFIG_PATH.open() as stream:
        return json.load(stream)


def get_data_access():
    """Return the selected release, or the configured default before selection."""
    return _SELECTED_RELEASE or load_access_config()["default_data_access"]


def get_access_settings(data_access=None):
    """Read the selected release's endpoints, catalog tables and image collection."""
    release = get_data_access() if data_access is None else data_access
    config = load_access_config()
    if release not in config["releases"]:
        raise ValueError(f"DATA_ACCESS must be one of {tuple(config['releases'])}, got {release!r}")
    return config["releases"][release]


def set_data_access(data_access):
    """Select the release for subsequent calls, including already-imported readers."""
    releases = load_access_config()["releases"]
    if not isinstance(data_access, str) or data_access not in releases:
        raise ValueError(f"DATA_ACCESS must be one of {tuple(releases)}, got {data_access!r}")
    ensure_https_ca_bundle()
    global _SELECTED_RELEASE
    _SELECTED_RELEASE = data_access
    return data_access


def get_data_dir():
    """Resolve a separate cache root for the current release at call time."""
    base = Path(os.environ.get("EUCLID_PHOT_DATA_DIR",
                              DEFAULT_DATA_DIR))
    return base / get_access_settings()["cache_subdir"]


def _make_irsa_client(settings):
    """Build an isolated client from explicit endpoints, ignoring user defaults."""
    from astroquery.ipac.irsa.core import IrsaClass
    ensure_https_ca_bundle()
    client = IrsaClass()
    client.tap_url = settings["tap_url"]
    client.sia_url = settings["sia_url"]
    client.ssa_url = settings["ssa_url"]
    client._session.request = partial(client._session.request,
                                      timeout=settings["timeout_seconds"])
    return client


def get_irsa_client(data_access=None):
    """Create an isolated client for the selected or explicitly requested release.

    An explicit release is useful for archive inspection and does not change
    the release used by subsequent photometry calls.
    """
    return _make_irsa_client(get_access_settings(data_access)["irsa"])


def get_public_irsa_client():
    """Use public IRSA for external catalogs, independent of the Euclid release."""
    return _make_irsa_client(load_access_config()["public_irsa"])
