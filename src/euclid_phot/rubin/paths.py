"""Writable Rubin caches, separated by the selected Euclid release."""
from __future__ import annotations

import os
from pathlib import Path

from ..config import DEFAULT_DATA_DIR
from ..data_access import get_access_settings, get_data_access


def get_cache_dir(data_access=None):
    """Return <base>/<q1 or dr1>/rubin without changing the Euclid cache.

    The base follows EUCLID_PHOT_DATA_DIR and otherwise uses the shared
    package default. Rubin does not use Euclid's DR1 deep_survey subdirectory.
    """
    release = get_data_access() if data_access is None else data_access
    settings = get_access_settings(release)
    release = settings.get("data_release", release)
    base = Path(os.environ.get("EUCLID_PHOT_DATA_DIR", DEFAULT_DATA_DIR))
    return base / release.lower() / "rubin"
