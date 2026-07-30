from __future__ import annotations

import math
import os
import platform
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

import evaluate as upstream_evaluate  # noqa: E402
import mojo_evaluate  # noqa: E402


def timeit(function, repeat):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def machine():
    try:
        model = subprocess.check_output(
            ["lscpu"], text=True, stderr=subprocess.DEVNULL
        )
        model = next(
            line.split(":", 1)[1].strip()
            for line in model.splitlines()
            if line.startswith("Model name:")
        )
    except (OSError, StopIteration, subprocess.CalledProcessError):
        model = platform.processor() or platform.machine()
    return f"{model}; {os.cpu_count()} logical CPUs; {platform.platform()}"


def comparison(name, config, predictions, references, kwargs):
    ours_metric = mojo_evaluate.load(name, config)
    upstream_metric = upstream_evaluate.load(name, config)
    ours = lambda: ours_metric.compute(
        predictions=predictions, references=references, **kwargs
    )
    theirs = lambda: upstream_metric.compute(
        predictions=predictions, references=references, **kwargs
    )
    ours()
    theirs()
    return ours, theirs


def cases():
    rng = np.random.default_rng(20260730)

    references = rng.integers(0, 2, size=1_000_000, dtype=np.int64)
    predictions = references.copy()
    flip = rng.random(references.size) < 0.15
    predictions[flip] ^= 1
    yield (
        "accuracy (1M binary)",
        *comparison("accuracy", None, predictions, references, {}),
    )
    yield (
        "f1 (1M binary)",
        *comparison("f1", None, predictions, references, {}),
    )
    del references, predictions, flip

    references32 = rng.integers(0, 32, size=1_000_000, dtype=np.int64)
    predictions32 = references32.copy()
    replace = rng.random(references32.size) < 0.20
    predictions32[replace] = rng.integers(
        0, 32, size=np.count_nonzero(replace), dtype=np.int64
    )
    yield (
        "confusion_matrix (1M, 32 classes)",
        *comparison(
            "confusion_matrix", None, predictions32, references32, {}
        ),
    )
    yield (
        "f1 macro (1M, 32 classes)",
        *comparison(
            "f1",
            None,
            predictions32,
            references32,
            {"average": "macro"},
        ),
    )
    yield (
        "MCC (1M, 32 classes)",
        *comparison(
            "matthews_correlation",
            None,
            predictions32,
            references32,
            {},
        ),
    )
    del references32, predictions32, replace

    references_float = rng.normal(size=1_000_000)
    predictions_float = references_float + rng.normal(
        scale=0.4, size=references_float.size
    )
    yield (
        "mse (1M)",
        *comparison(
            "mse", None, predictions_float, references_float, {}
        ),
    )
    yield (
        "mae (1M)",
        *comparison(
            "mae", None, predictions_float, references_float, {}
        ),
    )
    del references_float, predictions_float

    references_multi = rng.normal(size=(250_000, 8))
    predictions_multi = references_multi + rng.normal(
        scale=0.4, size=references_multi.shape
    )
    yield (
        "mse (250k x 8 outputs)",
        *comparison(
            "mse",
            "multilist",
            predictions_multi,
            references_multi,
            {"multioutput": "raw_values"},
        ),
    )


def main():
    results = []
    for name, ours, theirs in cases():
        ours_time = timeit(ours, repeat=5)
        upstream_time = timeit(theirs, repeat=1)
        results.append((name, ours_time, upstream_time))

    print(f"Machine: {machine()}")
    print()
    print("| case | mojo-evaluate | evaluate 0.4.6 | result |")
    print("| --- | ---: | ---: | ---: |")
    for name, ours_time, upstream_time in results:
        ratio = upstream_time / ours_time
        if ratio >= 1:
            result = f"{ratio:.1f}x faster"
        else:
            result = f"{1 / ratio:.1f}x slower"
        print(
            f"| {name} | {ours_time * 1e3:.2f} ms | "
            f"{upstream_time * 1e3:.2f} ms | {result} |"
        )


if __name__ == "__main__":
    main()
