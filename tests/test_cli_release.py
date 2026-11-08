"""CLI release selection reaches the same shared pipeline as the notebooks."""
from types import SimpleNamespace

import pytest
from astropy.table import Table

import euclid_phot as ep
from euclid_phot import cli, pipeline


@pytest.mark.parametrize("release", ["Q1", "DR1"])
@pytest.mark.parametrize("explicit_cache", [False, True])
def test_cli_selects_release_and_preserves_cache_override(release, explicit_cache, tmp_path, monkeypatch):
    previous = ep.get_data_access()
    ep.set_data_access("Q1")
    monkeypatch.setenv("EUCLID_PHOT_DATA_DIR", str(tmp_path))
    calls = []
    output = tmp_path / "catalog.ecsv"

    def run(*args, **kwargs):
        calls.append((ep.get_data_access(), ep.get_data_dir(), kwargs["data_dir"]))
        table = Table({"object_id": [1]}, meta={"DATAREL": ep.get_data_access()})
        return SimpleNamespace(to_table=lambda: table)

    monkeypatch.setattr(pipeline, "run_forced_photometry", run)
    args = ["run", "--ra", "269.48", "--dec", "67.3", "--data-access", release, "--out", str(output)]
    if explicit_cache:
        args += ["--data-dir", str(tmp_path / "custom")]
    try:
        assert cli.main(args) == 0
        subdir = "q1" if release == "Q1" else "dr1/deep_survey"
        assert calls == [(release, tmp_path / subdir,
                          str(tmp_path / "custom") if explicit_cache else None)]
        assert Table.read(output).meta["DATAREL"] == release
    finally:
        ep.set_data_access(previous)
