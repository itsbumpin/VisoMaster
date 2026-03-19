#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${VISO_ENV_NAME:-visomaster}"
PYTHON_VERSION="3.11"

fail() {
  echo "[ERROR] $*" >&2
  exit 1
}

step() {
  echo
  echo "========== $* =========="
}

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

  fail "Conda not found. Install Miniconda first: https://www.anaconda.com/download"
}

step "System detection"
OS_NAME="$(uname -s)"
[[ "$OS_NAME" == "Linux" ]] || fail "This setup script supports Linux only."
echo "OS: $OS_NAME"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  fail "nvidia-smi not found. NVIDIA driver / GPU runtime is missing."
fi

GPU_INFO="$(nvidia-smi --query-gpu=name,driver_version,cuda_version --format=csv,noheader | head -n1 || true)"
[[ -n "$GPU_INFO" ]] || fail "Unable to query GPU details via nvidia-smi."
IFS=',' read -r GPU_NAME GPU_DRIVER GPU_CUDA <<<"$GPU_INFO"
GPU_NAME="${GPU_NAME// / }"; GPU_DRIVER="${GPU_DRIVER// /}"; GPU_CUDA="${GPU_CUDA// /}"
echo "GPU: $GPU_NAME"
echo "Driver: $GPU_DRIVER"
echo "CUDA: $GPU_CUDA"

DRIVER_MAJOR="${GPU_DRIVER%%.*}"
if [[ "${DRIVER_MAJOR:-0}" -lt 550 ]]; then
  fail "Driver $GPU_DRIVER is too old. Need >= 550.xx for CUDA 12.x ecosystem."
fi

CUDA_MAJOR="${GPU_CUDA%%.*}"
if [[ "${CUDA_MAJOR:-0}" -lt 12 ]]; then
  fail "CUDA $GPU_CUDA is unsupported. Need CUDA 12.x capable host driver/runtime."
fi

step "Checking package indexes"
for url in \
  "https://download.pytorch.org/whl/cu129" \
  "https://pypi.nvidia.com/simple" \
  "https://pypi.org/simple"; do
  if ! curl -fsI --max-time 10 "$url" >/dev/null; then
    fail "Cannot reach required package index: $url"
  fi
  echo "OK: $url"
done

step "Conda environment"
source_conda
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Reusing conda env: $ENV_NAME"
else
  echo "Creating conda env: $ENV_NAME (python=$PYTHON_VERSION)"
  conda create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y
fi
conda activate "$ENV_NAME"

PY_VER="$(python -c 'import platform; print(platform.python_version())')"
[[ "$PY_VER" == 3.11.* ]] || fail "Active env has Python $PY_VER, expected 3.11.x"
echo "Python: $PY_VER"

step "Installing Linux runtime OS packages (Qt/OpenGL)"
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update -y
  sudo apt-get install -y libgl1 libglib2.0-0 libxkbcommon0 libxcb1 libx11-6 libxext6 libxrender1 libsm6 libice6 ffmpeg
else
  echo "apt-get not available: skipping OS package auto-install."
fi

step "Pip bootstrap"
python -m pip install --upgrade pip setuptools wheel

step "Fast preflight (before full install)"
if ! python "$ROOT_DIR/scripts/preflight_linux.py"; then
  fail "Preflight failed early. See scripts/preflight_report.json"
fi

step "Installing base/runtime dependencies"
python -m pip install -r "$ROOT_DIR/requirements-base.txt"

step "Installing CUDA/Torch/TensorRT dependencies"
set +e
python -m pip install -r "$ROOT_DIR/requirements-cuda-cu129.txt"
CUDA_INSTALL_RC=$?
set -e
if [[ $CUDA_INSTALL_RC -ne 0 ]]; then
  echo "Initial CUDA/TRT install failed. Attempting cuDNN pin patch + retry..."
  python -m pip install "nvidia-cudnn-cu12>=9.3,<10" --extra-index-url https://pypi.nvidia.com
  python -m pip install -r "$ROOT_DIR/requirements-cuda-cu129.txt" || \
    fail "CUDA/TRT install failed after cuDNN patch. You can still inspect preflight output for root cause."
fi

step "Downloading models"
python "$ROOT_DIR/download_models.py"

step "Running import smoke tests"
python "$ROOT_DIR/scripts/import_smoke_test.py" || fail "Import smoke test failed."

step "Final summary"
TRT_STATE="unavailable"
if python - <<'PY'
import ctypes.util
import importlib
ok = importlib.util.find_spec("tensorrt") is not None and bool(ctypes.util.find_library("nvinfer")) and bool(ctypes.util.find_library("nvinfer_plugin"))
raise SystemExit(0 if ok else 1)
PY
then
  TRT_STATE="available"
fi

echo "Environment: $ENV_NAME"
echo "Python: $(python -V 2>&1)"
echo "GPU: $GPU_NAME"
echo "Driver/CUDA: $GPU_DRIVER / $GPU_CUDA"
echo "TensorRT: $TRT_STATE"
if [[ "$TRT_STATE" != "available" ]]; then
  echo "CUDA fallback will be used. Reason: TensorRT Python/libnvinfer plugin is incomplete on this node."
fi

echo
echo "Setup complete. Launch with: bash scripts/run_linux.sh"
