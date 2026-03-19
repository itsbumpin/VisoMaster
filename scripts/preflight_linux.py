#!/usr/bin/env python3
"""Fast Linux/Runpod preflight checks for VisoMaster."""

from __future__ import annotations

import ctypes.util
import importlib
import json
import platform
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

MIN_DRIVER = 550
SUPPORTED_PY = {(3, 11)}
INDEX_URLS = [
    "https://download.pytorch.org/whl/cu129",
    "https://pypi.nvidia.com/simple",
    "https://pypi.org/simple",
]
QT_LIBS = ["libGL.so.1", "libxkbcommon.so.0", "libxcb.so.1"]
CHECK_IMPORTS = [
    "PySide6",
    "qdarktheme",
    "pyqttoast",
    "send2trash",
    "pyvirtualcam",
    "tqdm",
]


def run(cmd: list[str]) -> tuple[int, str]:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return 0, out.strip()
    except Exception as exc:  # noqa: BLE001
        return 1, str(exc)


def check_url(url: str) -> bool:
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=8):
            return True
    except Exception:  # noqa: BLE001
        return False


def parse_driver(ver: str) -> int:
    match = re.search(r"(\d+)\.", ver)
    return int(match.group(1)) if match else 0


def collect_gpu_info() -> dict[str, str]:
    code, out = run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,cuda_version",
            "--format=csv,noheader",
        ]
    )
    if code != 0:
        return {"error": out}
    first = out.splitlines()[0]
    parts = [p.strip() for p in first.split(",")]
    if len(parts) < 3:
        return {"error": out}
    return {"name": parts[0], "driver": parts[1], "cuda": parts[2]}


def find_trt_libs() -> dict[str, str | None]:
    search_roots = [
        Path("/usr/lib/x86_64-linux-gnu"),
        Path("/usr/local/lib"),
        Path(sys.prefix) / "lib",
        Path.home() / "miniconda3" / "envs" / "visomaster" / "lib",
    ]
    names = {"nvinfer": "libnvinfer.so", "nvinfer_plugin": "libnvinfer_plugin.so"}
    found: dict[str, str | None] = {}

    for key, soname in names.items():
        path = ctypes.util.find_library(key)
        if path:
            found[key] = path
            continue

        hit = None
        for root in search_roots:
            if root.exists():
                for candidate in root.glob(f"{soname}*"):
                    hit = str(candidate)
                    break
            if hit:
                break
        found[key] = hit
    return found


def check_qt_runtime_libs() -> dict[str, bool]:
    code, out = run(["ldconfig", "-p"])
    if code != 0:
        return {lib: False for lib in QT_LIBS}
    return {lib: (lib in out) for lib in QT_LIBS}


def check_imports() -> dict[str, bool]:
    status = {}
    for name in CHECK_IMPORTS:
        try:
            importlib.import_module(name)
            status[name] = True
        except Exception:  # noqa: BLE001
            status[name] = False
    return status


def main() -> int:
    print("== VisoMaster Linux preflight ==")
    failures = []

    if platform.system().lower() != "linux":
        failures.append("Only Linux is supported by this preflight script.")

    py_tuple = (sys.version_info.major, sys.version_info.minor)
    py_ok = py_tuple in SUPPORTED_PY
    if not py_ok:
        failures.append(f"Unsupported Python version: {platform.python_version()} (expected 3.11.x)")

    gpu = collect_gpu_info()
    print(f"GPU: {gpu.get('name', 'not found')}")
    print(f"Driver: {gpu.get('driver', 'not found')}")
    print(f"CUDA: {gpu.get('cuda', 'not found')}")

    if "error" in gpu:
        failures.append("nvidia-smi unavailable or no NVIDIA GPU detected.")
    else:
        if parse_driver(gpu["driver"]) < MIN_DRIVER:
            failures.append(f"Driver {gpu['driver']} too old. Need >= {MIN_DRIVER}.xx for modern CUDA 12.x wheels.")
        cuda_major = int(gpu["cuda"].split(".")[0]) if gpu["cuda"].split(".")[0].isdigit() else 0
        if cuda_major < 12:
            failures.append(f"CUDA runtime {gpu['cuda']} is unsupported. Need CUDA 12.x capable driver/runtime.")

    index_results = {u: check_url(u) for u in INDEX_URLS}
    for url, ok in index_results.items():
        print(f"Index reachable [{url}]: {'yes' if ok else 'no'}")
    if not all(index_results.values()):
        failures.append("One or more required package indexes are not reachable.")

    trt_libs = find_trt_libs()
    print(f"nvinfer found: {trt_libs.get('nvinfer') or 'no'}")
    print(f"nvinfer_plugin found: {trt_libs.get('nvinfer_plugin') or 'no'}")
    if not trt_libs.get("nvinfer") or not trt_libs.get("nvinfer_plugin"):
        failures.append("TensorRT shared libraries are not discoverable (nvinfer / nvinfer_plugin).")

    qt_libs = check_qt_runtime_libs()
    for lib, ok in qt_libs.items():
        print(f"Qt runtime dep {lib}: {'ok' if ok else 'missing'}")
    if not all(qt_libs.values()):
        failures.append("Missing one or more Qt runtime shared libraries.")

    import_results = check_imports()
    for name, ok in import_results.items():
        print(f"Import {name}: {'ok' if ok else 'missing'}")

    summary = {
        "python": platform.python_version(),
        "python_supported": py_ok,
        "gpu": gpu,
        "indexes": index_results,
        "tensorrt_libs": trt_libs,
        "qt_runtime": qt_libs,
        "imports": import_results,
    }
    Path("scripts").mkdir(exist_ok=True)
    Path("scripts/preflight_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if failures:
        print("\nPRECHECK FAILED:")
        for item in failures:
            print(f" - {item}")
        print("\nFix these items first, then rerun setup_linux.sh.")
        return 1

    print("\nPreflight checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
