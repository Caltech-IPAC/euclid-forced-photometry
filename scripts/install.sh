#!/usr/bin/env bash
#
# install.sh — set up the euclid_phot environment in one command.
#
# `pip install -e .` already pulls in every PyPI dependency (they are declared
# in pyproject.toml). This script additionally installs the three packages the
# README installs by hand, because none of them is a plain `pip install`:
#
#   1. Tractor          — needs `pip install --no-build-isolation` with numpy +
#                         cython already present (its setup.py imports numpy at
#                         build time and builds the _mp_fourier C extension).
#   2. astrometry.net   — not pip-installable (its setup.py defers to `make`);
#                         the pure-Python modules Tractor needs are exposed via
#                         a source clone placed on a .pth path.
#   3. unwise_psf        — not packaged; same .pth trick. Optional (--wise),
#                         only needed for the unWISE W1/W2 leg, with fitsio and
#                         setuptools<81 alongside it.
#
# Versions are pinned to a set known to work: PyPI packages through
# scripts/constraints.txt, the three GitHub packages to the commits set
# below. --unpinned takes the latest of each instead.
#
# Safe to re-run: every step checks whether it is already done.
#
# Usage:
#   ./scripts/install.sh            # VIS + NISP
#   ./scripts/install.sh --wise     # also the optional unWISE W1/W2 leg
#   ./scripts/install.sh --python /path/to/python   # install into a given env
#   ./scripts/install.sh --unpinned # latest versions instead of the pinned ones
#   ./scripts/install.sh --help
#
set -euo pipefail

WANT_WISE=0
PY_OVERRIDE=""
PINNED=1

# Commits known to work with scripts/constraints.txt (last checked 2026-11-07).
TRACTOR_COMMIT=3fd2e80eafb9cc092e203ba50a95557eb8543878      # 2026-04-22, Tractor 11.1
ASTROMETRY_COMMIT=42b12ec1d50cbfd7fe90f9e645e84643bf018029   # 2026-10-11
UNWISE_PSF_COMMIT=43d17fc66784f1fce741f46c36f3780a2a5736f5   # 2026-03-07

while [[ $# -gt 0 ]]; do
    case "$1" in
        --wise) WANT_WISE=1; shift ;;
        --unpinned) PINNED=0; shift ;;
        --python) PY_OVERRIDE="${2:-}"; shift 2 ;;
        -h|--help)
            awk 'NR>1 && /^#/{sub(/^# ?/,""); print; next} NR>1{exit}' "$0"
            exit 0 ;;
        *) echo "unknown option: $1 (try --help)" >&2; exit 2 ;;
    esac
done

log() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

# --- Locate the repo root (this script lives in scripts/) --------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/pyproject.toml" ]]; then
    REPO_ROOT="$SCRIPT_DIR"
else
    REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
if [[ ! -f "$REPO_ROOT/pyproject.toml" ]]; then
    echo "could not find pyproject.toml near $SCRIPT_DIR" >&2
    exit 1
fi
cd "$REPO_ROOT"

# pip constraints: the exact PyPI versions in scripts/constraints.txt (unless --unpinned).
PIP_C=()
if [[ "$PINNED" -eq 1 ]]; then
    PIP_C=(-c "$REPO_ROOT/scripts/constraints.txt")
fi

# --- Pick the Python interpreter to install into -----------------------------
# Priority: explicit --python > active venv > active conda env > a fresh .venv.
if [[ -n "$PY_OVERRIDE" ]]; then
    PY="$PY_OVERRIDE"
    log "Using interpreter from --python: $PY"
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
    PY="$VIRTUAL_ENV/bin/python"
    log "Using the active virtualenv: $VIRTUAL_ENV"
elif [[ -n "${CONDA_PREFIX:-}" ]]; then
    PY="$CONDA_PREFIX/bin/python"
    log "Using the active conda env: ${CONDA_DEFAULT_ENV:-$CONDA_PREFIX}"
    if [[ "${CONDA_DEFAULT_ENV:-}" == "base" ]]; then
        echo "  note: installing into conda 'base'. Consider 'conda create -n euclid python=3.12' first."
    fi
else
    if [[ ! -d "$REPO_ROOT/.venv" ]]; then
        command -v python3 >/dev/null || { echo "python3 not found on PATH" >&2; exit 1; }
        log "No environment active; creating .venv"
        python3 -m venv "$REPO_ROOT/.venv"
    else
        log "No environment active; reusing existing .venv"
    fi
    PY="$REPO_ROOT/.venv/bin/python"
    CREATED_VENV=1
fi

"$PY" -c 'import sys; print("    python", sys.version.split()[0], "->", sys.executable)'
if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3, 10) else 1)'; then
    echo "euclid_phot needs Python >= 3.10, but the selected interpreter is $("$PY" -V 2>&1)." >&2
    echo "Create a newer environment first, e.g.:  conda create -n euclid python=3.12" >&2
    echo "then re-run, or pass --python /path/to/a/3.10+/python." >&2
    exit 1
fi
if [[ "$PINNED" -eq 1 ]] && ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3, 11) else 1)'; then
    echo "The pinned versions in scripts/constraints.txt need Python >= 3.11 (selected: $("$PY" -V 2>&1))." >&2
    echo "Create a newer environment (conda create -n euclid python=3.12) or pass --unpinned." >&2
    exit 1
fi

# Write a .pth file into the env's site-packages so a source clone is importable.
# $1 = .pth filename, $2 = directory to add to the import path.
write_pth() {
    "$PY" - "$1" "$2" <<'PYEOF'
import sys, sysconfig, pathlib
name, target = sys.argv[1], sys.argv[2]
sp = pathlib.Path(sysconfig.get_path("purelib"))
sp.mkdir(parents=True, exist_ok=True)
(sp / name).write_text(str(pathlib.Path(target).resolve()) + "\n")
print(f"    {sp / name} -> {pathlib.Path(target).resolve()}")
PYEOF
}

has_module() { "$PY" -c "import $1" >/dev/null 2>&1; }

# Keep a vendored clone pristine: importing it generates __pycache__, which
# would show as changes in IDEs that auto-detect nested git repos. The
# clone's .git/info/exclude is a local-only ignore, so nothing upstream is
# touched. No-op if the pattern is already there.
exclude_pycache() {
    local exclude="$1/.git/info/exclude"
    [[ -d "$1/.git" ]] || return 0
    grep -qxF "__pycache__/" "$exclude" 2>/dev/null \
        || echo "__pycache__/" >> "$exclude"
}

# Shallow clone of one commit ($3) when pinned, of the branch head otherwise.
clone_at() {
    local url="$1" dir="$2" sha="$3"
    if [[ "$PINNED" -eq 1 ]]; then
        git init -q "$dir"
        git -C "$dir" remote add origin "$url"
        git -C "$dir" fetch -q --depth 1 origin "$sha"
        git -C "$dir" checkout -q --detach FETCH_HEAD
    else
        git clone --depth 1 "$url" "$dir"
    fi
}

# Note when an existing clone is not at the pinned commit; delete it to re-clone.
check_pin() {
    local dir="$1" sha="$2" head
    [[ "$PINNED" -eq 1 && -d "$dir/.git" ]] || return 0
    head="$(git -C "$dir" rev-parse HEAD 2>/dev/null || true)"
    if [[ -n "$head" && "$head" != "$sha" ]]; then
        echo "    note: $dir is at ${head:0:12}, not the pinned ${sha:0:12}; delete it and re-run to re-clone"
    fi
}

# Tractor's build runs SWIG to generate its C-extension wrappers (the
# _mp_fourier FFT module). SWIG is a build tool, not a pip dependency of
# Tractor, so ensure it is present before the build. The official `swig`
# wheel on PyPI works in any environment (venv/conda/system) with no system
# package manager, so try that first; fall back to conda/brew/apt.
ensure_swig() {
    if command -v swig >/dev/null 2>&1; then
        return
    fi
    log "SWIG (needed to build Tractor) was not found; installing it"
    # 1. Universal path: the `swig` wheel drops a binary in the env's bin dir.
    if "$PY" -m pip install ${PIP_C[@]+"${PIP_C[@]}"} swig >/dev/null 2>&1; then
        local scripts_dir
        scripts_dir="$("$PY" -c "import sysconfig; print(sysconfig.get_path('scripts'))")"
        export PATH="$scripts_dir:$PATH"   # so the Tractor build sees it
    fi
    if command -v swig >/dev/null 2>&1; then
        echo "    installed swig via pip -> $(command -v swig)"
        return
    fi
    # 2. Fallbacks if no wheel is available for this platform.
    local env_prefix
    env_prefix="$(cd "$(dirname "$PY")/.." && pwd)"
    if [[ -d "$env_prefix/conda-meta" ]] && command -v conda >/dev/null 2>&1; then
        conda install -y -p "$env_prefix" -c conda-forge swig
        export PATH="$env_prefix/bin:$PATH"
    elif command -v brew >/dev/null 2>&1; then
        brew install swig
    elif command -v apt-get >/dev/null 2>&1; then
        sudo apt-get install -y swig
    else
        cat >&2 <<'EOF'
    SWIG was not found and could not be installed automatically. Install it,
    then re-run this script:
        pip install swig                     (usually works anywhere)
        conda install -c conda-forge swig
        brew install swig
        sudo apt-get install swig
EOF
        exit 1
    fi
    command -v swig >/dev/null 2>&1 \
        || { echo "swig still not on PATH after install" >&2; exit 1; }
}

# --- 1. The package + its PyPI dependencies (incl. numpy, jupyter) -----------
log "Installing euclid_phot and its PyPI dependencies"
"$PY" -m pip install ${PIP_C[@]+"${PIP_C[@]}"} -e ".[dev]"

# --- 2. Tractor (needs cython, then --no-build-isolation) --------------------
if has_module tractor; then
    log "Tractor already installed; skipping"
else
    log "Installing Tractor (with cython, --no-build-isolation)"
    ensure_swig
    # --no-build-isolation uses the env's own build backend, so setuptools and
    # wheel must be present (Python 3.12+ venvs no longer ship setuptools).
    # cython + numpy (a euclid_phot dep, already installed) build _mp_fourier.
    "$PY" -m pip install ${PIP_C[@]+"${PIP_C[@]}"} setuptools wheel cython
    # Tractor's SWIG wrapper for _mp_fourier still names two Python 2 C-API
    # functions; recent clang rejects the implicit declarations and the
    # module would fail to load anyway. Map them to their Python 3 names.
    # SWIG >= 4.4 also generates an int-returning module init, in which
    # numpy < 2.3.5's import_array() macro returns NULL; recent clang/gcc
    # treat that as an error, so downgrade it to a warning.
    TRACTOR_SRC="git+https://github.com/dstndstn/tractor.git"
    if [[ "$PINNED" -eq 1 ]]; then
        TRACTOR_SRC="$TRACTOR_SRC@$TRACTOR_COMMIT"
    fi
    CFLAGS="${CFLAGS:-} -DPyString_Check=PyUnicode_Check -DPyInt_Check=PyLong_Check -Wno-int-conversion" \
        "$PY" -m pip install --no-build-isolation "$TRACTOR_SRC"
fi

# --- 3. astrometry.net pure-Python utilities (source clone on a .pth) --------
if has_module astrometry.util.fits; then
    log "astrometry.net utilities already importable; skipping"
else
    log "Setting up astrometry.net utilities"
    if [[ ! -d "$REPO_ROOT/.anet/astrometry" ]]; then
        mkdir -p "$REPO_ROOT/.anet"
        clone_at https://github.com/dstndstn/astrometry.net.git \
            "$REPO_ROOT/.anet/astrometry" "$ASTROMETRY_COMMIT"
    fi
    check_pin "$REPO_ROOT/.anet/astrometry" "$ASTROMETRY_COMMIT"
    exclude_pycache "$REPO_ROOT/.anet/astrometry"
    write_pth astrometry_net.pth "$REPO_ROOT/.anet"
    has_module astrometry.util.fits \
        || { echo "astrometry.util.fits still not importable after setup" >&2; exit 1; }
fi

# --- 4. unWISE PSF leg (optional) --------------------------------------------
if [[ "$WANT_WISE" -eq 1 ]]; then
    # The top-level unwise_psf package can import without its rendering
    # dependencies. Check the actual implementation after installing them.
    "$PY" -m pip install ${PIP_C[@]+"${PIP_C[@]}"} fitsio "setuptools<81"
    if has_module unwise_psf.unwise_psf; then
        log "unwise_psf rendering module already importable"
    else
        log "Setting up the unWISE W1/W2 leg (unwise_psf, fitsio, setuptools<81)"
        if [[ ! -d "$REPO_ROOT/.unwise_psf/py" ]]; then
            clone_at https://github.com/legacysurvey/unwise_psf \
                "$REPO_ROOT/.unwise_psf" "$UNWISE_PSF_COMMIT"
        fi
        check_pin "$REPO_ROOT/.unwise_psf" "$UNWISE_PSF_COMMIT"
        exclude_pycache "$REPO_ROOT/.unwise_psf"
        write_pth unwise_psf.pth "$REPO_ROOT/.unwise_psf/py"
        has_module unwise_psf.unwise_psf \
            || { echo "unwise_psf still not importable after setup" >&2; exit 1; }
    fi
else
    log "Skipping the unWISE leg (pass --wise to include it)"
fi

# --- Done --------------------------------------------------------------------
log "Done."
if [[ "${CREATED_VENV:-0}" -eq 1 ]]; then
    echo "Created $REPO_ROOT/.venv — activate it before launching Jupyter:"
    echo "    source .venv/bin/activate"
fi
echo "Launch the notebooks with:"
echo "    jupyter lab notebooks/"
