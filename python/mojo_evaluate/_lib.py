from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src", "kernels.mojo")
LIB = os.environ.get("MOJO_EVALUATE_LIB") or os.path.join(
    ROOT, "dist", "libmojo-evaluate.so"
)

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "me_accuracy": ([I, I, I, I, I], F),
    "me_subset_accuracy": ([I, I, I, I, I, I], F),
    "me_confusion_matrix": ([I, I, I, I, I, I, I], None),
    "me_multilabel_stats": ([I, I, I, I, I, I, I], None),
    "me_multilabel_samples": ([I, I, I, I, I, I, I], None),
    "me_regression_reductions": ([I, I, I, I, I, I, I], None),
    "me_naive_mae": ([I, I, I, I, I], None),
}


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    if os.environ.get("MOJO_EVALUATE_LIB"):
        if os.path.exists(LIB):
            return LIB
        raise BuildError(f"MOJO_EVALUATE_LIB does not exist: {LIB}")
    stale = (
        force
        or not os.path.exists(LIB)
        or (os.path.exists(SRC) and os.path.getmtime(SRC) > os.path.getmtime(LIB))
    )
    if stale:
        script = os.path.join(ROOT, "build", "build.sh")
        proc = subprocess.run(
            ["bash", script],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if proc.returncode != 0 or not os.path.exists(LIB):
            output = "\n".join(
                part.strip() for part in (proc.stdout, proc.stderr) if part.strip()
            )
            raise BuildError(output[:4000] or "Mojo build failed without output")
    return LIB


_LIBRARY: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _LIBRARY
    if _LIBRARY is None:
        _LIBRARY = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_LIBRARY, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _LIBRARY


def addr(array: np.ndarray | None) -> int:
    if array is None:
        return 0
    if not isinstance(array, np.ndarray):
        raise TypeError("FFI buffers must be NumPy arrays")
    if array.size == 0:
        raise ValueError("FFI buffers must not be empty")
    if array.dtype not in (np.dtype(np.int64), np.dtype(np.float64)):
        raise TypeError(f"unsupported FFI buffer dtype: {array.dtype}")
    if not array.flags.c_contiguous or not array.flags.aligned:
        raise ValueError("FFI buffers must be aligned and C-contiguous")
    address = int(array.ctypes.data)
    if address == 0:
        raise ValueError("FFI buffers must have a non-null address")
    return address
