# mojo-evaluate

`mojo-evaluate` is a standalone Mojo port of the compute-heavy classification
and regression kernels used by Hugging Face
[`evaluate`](https://github.com/huggingface/evaluate). Its Python API keeps the
covered part of Evaluate's loader-based interface:

```python
import mojo_evaluate as evaluate

metric = evaluate.load("f1")
result = metric.compute(
    predictions=[0, 1, 1, 0],
    references=[0, 1, 0, 0],
)
print(result)
# {'f1': 0.6666666666666666}
```

The implementation is local and deterministic. Covered metrics do not
download executable metric scripts from the Hugging Face Hub.

## Coverage

| metric | covered arguments |
| --- | --- |
| `accuracy` | `normalize`, `sample_weight`, multilabel subset accuracy |
| `precision`, `recall`, `f1` | binary, multiclass, and multilabel; `labels`, `pos_label`, `average`, `sample_weight`; `zero_division` for precision and recall |
| `confusion_matrix` | `labels`, `sample_weight`, `normalize` |
| `matthews_correlation` | binary, multiclass, and multilabel; `sample_weight`, multilabel `average` |
| `mse` | `sample_weight`, multioutput modes and weights, `squared` |
| `mae` | `sample_weight`, multioutput modes and weights |
| `brier_score` | `sample_weight`, `pos_label` |
| `mase` | `training`, `periodicity`, `sample_weight`, multioutput modes and weights |
| `smape` | `sample_weight`, multioutput modes and weights |

`load`, keyword-only `compute`, streaming `add` / `add_batch`, and `combine`
are implemented. The `multilabel` and `multilist` configuration names match
upstream.

This is intentionally a metric-kernel subset, not a port of Evaluate's Hub
client and orchestration system. BLEU, ROUGE, WER, BERTScore, perplexity,
SQuAD, comparisons, measurements, evaluators, distributed accumulation,
bootstrap confidence intervals, and loading arbitrary Hub modules are not
covered. Calling `load` for an uncovered metric raises `FileNotFoundError`
instead of silently falling back to Python.

## Install and run

The repository pins its own Mojo nightly:

```bash
pixi install
pixi run build
```

The build produces `dist/libmojo-evaluate.so`. Run the example through the
managed environment:

```bash
pixi run python - <<'PY'
import mojo_evaluate as evaluate

metrics = evaluate.combine(["accuracy", "f1"])
print(metrics.compute(
    predictions=[0, 1, 1, 0],
    references=[0, 1, 0, 0],
))
PY
```

Run the parity suite and benchmarks with:

```bash
pixi run test
pixi run bench
```

The benchmark task owns a machine-wide `flock`; invoking the script directly
does not provide that protection.

## Correctness

The test suite exercises every metric and every argument listed above,
including weighted, multiclass, multilabel, multioutput, streaming, and
invalid-input cases. Results are compared with the installed Hugging Face
Evaluate modules where those modules run. The sMAPE cases use a direct NumPy
implementation of the published formula because Evaluate 0.4.6 calls a
removed private scikit-learn interface with the current environment.

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz, 72
logical CPUs, Linux 6.8.0-136-generic, on 2026-08-24:

| case | mojo-evaluate | evaluate 0.4.6 | result |
| --- | ---: | ---: | ---: |
| accuracy (1M binary) | 2.74 ms | 2976.69 ms | 1084.5x faster |
| f1 (1M binary) | 8.70 ms | 3587.88 ms | 412.6x faster |
| confusion_matrix (1M, 32 classes) | 8.27 ms | 2841.66 ms | 343.7x faster |
| f1 macro (1M, 32 classes) | 7.65 ms | 3554.52 ms | 464.7x faster |
| MCC (1M, 32 classes) | 8.48 ms | 3412.49 ms | 402.4x faster |
| mse (1M) | 4.38 ms | 2947.32 ms | 672.4x faster |
| mae (1M) | 4.51 ms | 2951.45 ms | 653.7x faster |
| mse (250k x 8 outputs) | 7.16 ms | 6141.79 ms | 857.9x faster |

These are end-to-end calls to each package's public
`load(...).compute(...)` API on the same in-memory NumPy arrays. They do not
claim that a scalar Mojo loop is thousands of times faster than scikit-learn's
Cython. Most of the difference is architectural: Evaluate validates and
serializes examples through Datasets/Arrow and its cache before calling
scikit-learn, while mojo-evaluate validates a contiguous array and invokes one
native kernel. The benchmark includes that wrapper work on both sides because
it is the cost applications pay through the mirrored public API. Each metric
is warmed once. The table reports the best of five mojo-evaluate calls and one
Evaluate call; the latter is not repeated because each call takes several
seconds.

No GPU path is provided. These kernels are streaming reductions or scatter
updates; this project makes no unmeasured GPU performance claim.

## How it works

All kernels live in one Mojo compilation unit and are exported with a C ABI.
The Python package loads the shared library with `ctypes`. Arrays cross the
boundary as integer addresses plus explicit dimensions; the exported
functions reconstruct `UnsafePointer` values with
`AnyOrigin[mut=True]`.

Classification labels use contiguous `int64` row-major buffers. Regression
values, sample weights, contingency tables, and reduction buffers use
contiguous `float64`. Mojo performs one-pass weighted accuracy, contingency,
multilabel, error, sMAPE, and seasonal-naive reductions without allocating.
Python handles label discovery and the small final operations such as
normalization and macro/weighted averaging.

The FFI makes one native call per metric computation. Python owns every input,
output, and scratch buffer, so pointer lifetimes remain tied to NumPy and the
Mojo library has nothing to free.

## License

MIT
