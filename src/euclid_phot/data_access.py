"""Release selection for the shared Euclid readers and fitting pipeline.

Call set_data_access("Q1"), set_data_access("DR1"), or
set_data_access("DR1_WIDE") before a run. This setting
belongs to the Python process, so archive worker threads use the same release.
Do not change it during a running fit. Endpoints and release identifiers live
in src/euclid_phot/conf.json; no date triggers a release or endpoint change automatically.
"""
from __future__ import annotations

from functools import lru_cache, partial
import hashlib
import json
import os
from pathlib import Path
import tempfile
import warnings

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


def get_access_settings(data_access=None, *, ra=None, dec=None, survey=None, cache_only=False):
    """Read release settings, selecting a DR1 survey when coordinates are given.

    Deep is preferred where its VIS images cover the position; otherwise
    Wide is used. Coverage lookup failures are propagated, never interpreted
    as empty coverage. Successful choices are cached for offline reuse.
    DR1_WIDE always selects the DR1 Wide profile without a Deep coverage probe.
    """
    release = get_data_access() if data_access is None else data_access
    config = load_access_config()
    if release not in config["releases"]:
        raise ValueError(f"DATA_ACCESS must be one of {tuple(config['releases'])}, got {release!r}")
    settings = config["releases"][release]
    if "base_release" in settings:
        forced_survey = settings["survey"]
        if survey is not None and survey != forced_survey:
            raise ValueError(f"{release} requires survey {forced_survey!r}, got {survey!r}")
        survey = forced_survey
        base_release = settings["base_release"]
        settings = {**config["releases"][base_release], "data_release": base_release}
    if survey is not None:
        if survey not in settings.get("surveys", {}):
            raise ValueError(f"Survey {survey!r} is not configured for {release}")
        return {**settings, **settings["surveys"][survey], "survey": survey}
    if (ra is None) != (dec is None):
        raise ValueError("Provide both ra and dec for survey selection")
    if ra is None or not settings.get("surveys"):
        return settings
    selection = _survey_for_position(release, float(ra) % 360, float(dec),
                                     settings, cache_only=cache_only)
    return {**settings, **settings["surveys"][selection], "survey": selection}


def _cache_base():
    return Path(os.environ.get("EUCLID_PHOT_DATA_DIR", DEFAULT_DATA_DIR))


@lru_cache(maxsize=128)
def _probe_survey(release, ra, dec, settings_json, client_factory):
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    settings = json.loads(settings_json)
    target = SkyCoord(ra, dec, unit="deg")
    client = client_factory(release)
    try:
        for name in settings["survey_priority"]:
            survey = settings["surveys"][name]
            rows = client.query_sia(pos=(target, 1 * u.mas),
                                    collection=survey["image_collection"])
            if any(str(row["energy_bandpassname"]) == "VIS"
                   and "BGSUB-MOSAIC" in str(row["access_url"]) for row in rows):
                return name
    finally:
        session = getattr(client, "_session", None)
        if session is not None:
            session.close()
    raise ValueError(f"{release}: no Deep or Wide VIS coverage at ({ra}, {dec})")


def _survey_for_position(release, ra, dec, settings, *, cache_only=False):
    relevant = {key: settings[key] for key in ("irsa", "surveys", "survey_priority")}
    settings_json = json.dumps(relevant, sort_keys=True)
    key = hashlib.sha256(f"{release}|{ra!r}|{dec!r}|{settings_json}".encode()).hexdigest()
    path = _cache_base() / settings["cache_subdir"] / "survey_routes" / f"{key}.json"
    try:
        choice = json.loads(path.read_text())["survey"]
        if choice in settings["surveys"]:
            return choice
    except (OSError, ValueError, KeyError, TypeError):
        pass
    if cache_only:
        raise FileNotFoundError(f"Survey coverage cache is missing: {path}")
    choice = _probe_survey(release, ra, dec, settings_json, get_irsa_client)
    temporary = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump({"survey": choice, "ra": ra, "dec": dec,
                       "image_collection": settings["surveys"][choice]["image_collection"]}, stream)
        temporary.replace(path)
    except OSError as exc:
        warnings.warn(f"Could not cache survey coverage: {exc}", stacklevel=2)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return choice


def set_data_access(data_access):
    """Select the release for subsequent calls, including already-imported readers."""
    releases = load_access_config()["releases"]
    if not isinstance(data_access, str) or data_access not in releases:
        raise ValueError(f"DATA_ACCESS must be one of {tuple(releases)}, got {data_access!r}")
    ensure_https_ca_bundle()
    global _SELECTED_RELEASE
    _SELECTED_RELEASE = data_access
    return data_access


def get_data_dir(*, ra=None, dec=None, survey=None, cache_only=False):
    """Resolve a release cache, or a survey-specific cache for a position."""
    settings = get_access_settings(ra=ra, dec=dec, survey=survey, cache_only=cache_only)
    return _cache_base() / settings["cache_subdir"]


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
