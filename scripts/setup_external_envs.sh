#!/usr/bin/env bash
set -euo pipefail

# Optional dependencies for external probes. This does not change the MinAtar
# training code path; it only installs packages needed by fluid.train_external_discrete.

CONDA_ENV="${CONDA_ENV:-BLCR}"
PYTHON_BIN="${PYTHON_BIN:-}"

resolve_conda() {
  if command -v conda >/dev/null 2>&1; then
    command -v conda
    return 0
  fi

  local candidate
  for candidate in \
    "$HOME/miniforge3/bin/conda" \
    "$HOME/miniconda3/bin/conda" \
    "$HOME/anaconda3/bin/conda" \
    "/opt/conda/bin/conda"; do
    if [[ -x "$candidate" ]]; then
      echo "$candidate"
      return 0
    fi
  done

  return 1
}

if [[ -z "$PYTHON_BIN" ]]; then
  CONDA_BIN="$(resolve_conda || true)"
  if [[ -z "$CONDA_BIN" ]]; then
    echo "ERROR: neither PYTHON_BIN nor conda is available." >&2
    echo "Set PYTHON_BIN=/path/to/python or install/activate conda." >&2
    exit 127
  fi
  PYTHON_BIN="$("$CONDA_BIN" run -n "$CONDA_ENV" python - <<'PY'
import sys
print(sys.executable)
PY
)"
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: python is not executable: ${PYTHON_BIN}" >&2
  exit 127
fi

echo "Python: ${PYTHON_BIN}"
"$PYTHON_BIN" -m pip install \
  "gymnasium[atari]>=1.0,<2.0" \
  "gymnasium[box2d]>=1.0,<2.0" \
  "ale-py>=0.10" \
  "autorom[accept-rom-license]>=0.6.1" \
  "swig>=4.2" \
  "dm_control>=1.0.20"

AUTOROM_BIN="$(dirname "$PYTHON_BIN")/AutoROM"
if [[ -x "$AUTOROM_BIN" ]]; then
  "$AUTOROM_BIN" --accept-license || true
fi

echo "External dependencies installed. Verify with:"
echo "  ${PYTHON_BIN} scripts/check_external_env_support.py"
