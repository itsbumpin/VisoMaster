#!/usr/bin/env python3
"""Import smoke tests for core VisoMaster runtime dependencies."""

from __future__ import annotations

import importlib
import sys

REQUIRED_MODULES = [
    "cv2",
    "numpy",
    "onnx",
    "onnxruntime",
    "torch",
    "torchvision",
    "PySide6",
    "qdarktheme",
    "qdarkstyle",
    "pyqttoast",
    "send2trash",
    "pyvirtualcam",
    "requests",
    "tqdm",
    "ftfy",
    "regex",
]

OPTIONAL_MODULES = [
    "tensorrt",
]


def check(modules: list[str], optional: bool = False) -> list[str]:
    failed: list[str] = []
    for name in modules:
        try:
            importlib.import_module(name)
            print(f"[OK] import {name}")
        except Exception as exc:  # noqa: BLE001
            level = "WARN" if optional else "FAIL"
            print(f"[{level}] import {name}: {exc}")
            failed.append(name)
    return failed


def main() -> int:
    missing_required = check(REQUIRED_MODULES)
    missing_optional = check(OPTIONAL_MODULES, optional=True)

    if missing_optional:
        print("\nTensorRT import is unavailable: CUDA fallback mode can still run.")

    if missing_required:
        print("\nMissing required imports:", ", ".join(missing_required))
        return 1

    print("\nAll required imports are available.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
