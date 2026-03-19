#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${VISO_ENV_NAME:-visomaster}"

source_conda() {
  if command -v conda >/dev/null 2>&1; then
    local conda_base
    conda_base="$(conda info --base)"
    # shellcheck source=/dev/null
    source "$conda_base/etc/profile.d/conda.sh"
    return
  fi
  for candidate in "$HOME/miniconda3" "$HOME/anaconda3" "/opt/conda"; do
    if [[ -f "$candidate/etc/profile.d/conda.sh" ]]; then
      # shellcheck source=/dev/null
      source "$candidate/etc/profile.d/conda.sh"
      return
    fi
  done
  echo "[ERROR] Conda not found." >&2
  exit 1
}

source_conda
conda activate "$ENV_NAME"

export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
export QT_XCB_GL_INTEGRATION="${QT_XCB_GL_INTEGRATION:-xcb_egl}"

PREFIX_LIB="$(python -c 'import sys; print(sys.prefix + "/lib")')"
export LD_LIBRARY_PATH="$PREFIX_LIB:${LD_LIBRARY_PATH:-}"

if python - <<'PY'
import ctypes.util
import importlib
ok = importlib.util.find_spec("tensorrt") is not None and bool(ctypes.util.find_library("nvinfer")) and bool(ctypes.util.find_library("nvinfer_plugin"))
raise SystemExit(0 if ok else 1)
PY
then
  echo "[VisoMaster] Runtime mode: TensorRT enabled"
else
  echo "[VisoMaster] Runtime mode: CUDA fallback (TensorRT unavailable)"
fi

cd "$ROOT_DIR"
exec python main.py
