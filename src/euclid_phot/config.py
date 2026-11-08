"""Settings shared by every band.

Band constants live with their band: ``euclid.config`` (MER), ``wise.config``
(unWISE), ``irac.config`` (Spitzer/IRAC); flux units in :mod:`euclid_phot.units`.
"""
from __future__ import annotations

import os
from pathlib import Path

# Base cache directory; get_data_dir() adds the selected release subdirectory.
# Editable checkouts use examples/data; wheels use a writable user cache.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CACHE = (_REPO_ROOT / "examples/data"
                  if (_REPO_ROOT / "pyproject.toml").is_file()
                  and Path(__file__).resolve().parent.parent.name == "src"
                  else Path.home() / ".cache/euclid_phot")
DEFAULT_DATA_DIR = Path(os.environ.get("EUCLID_PHOT_DATA_DIR", _DEFAULT_CACHE))

# R_lambda = A_lambda / E(B-V). Euclid: Gordon et al. (2023) MW curve,
# A_lambda/A_V = 0.678/0.366/0.261/0.160 (Hunt et al. 2025, A&A 697, A9)
EXTINCTION_COEFF = {
    "VIS": 3.1 * 0.678,   # 2.102
    "Y":   3.1 * 0.366,   # 1.135
    "J":   3.1 * 0.261,   # 0.809
    "H":   3.1 * 0.160,   # 0.496
    "W1":  0.19,          # Yuan et al. 2013, MNRAS 430, 2188
    "W2":  0.15,
    # Indebetouw et al. 2005, ApJ 619, 931: A_lambda/A_K = 0.56, 0.43,
    # 0.43, 0.43 with A_K/E(B-V) = 0.353 for R_V = 3.1.
    "IRAC1": 0.198,
    "IRAC2": 0.152,
    "IRAC3": 0.152,
    "IRAC4": 0.152,
}
