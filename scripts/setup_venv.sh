#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.11}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "error: $PYTHON_BIN not found. Install Python 3.11 and retry." >&2
  exit 1
fi

PYTHON_VERSION="$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "$PYTHON_VERSION" != "3.11" ]]; then
  echo "error: expected Python 3.11, got $PYTHON_VERSION from $PYTHON_BIN." >&2
  exit 1
fi

rm -rf .venv
"$PYTHON_BIN" -m venv .venv
chmod u+w .venv/bin/activate .venv/bin/activate.csh .venv/bin/activate.fish 2>/dev/null || true
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pip uninstall -y coqpit >/dev/null 2>&1 || true
python - <<'PY'
import shutil
import site
from pathlib import Path

for base in site.getsitepackages():
    stale_pkg = Path(base) / "coqpit"
    if stale_pkg.exists():
        shutil.rmtree(stale_pkg, ignore_errors=True)
PY
python -m pip install --force-reinstall --no-deps coqpit-config==0.2.4

echo "venv ready with Python $PYTHON_VERSION: source .venv/bin/activate"
