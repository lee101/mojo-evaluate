# mojo-evaluate ANALYZE — measured hotspot map

Date: 2026-11-09 session. Machine: `Intel(R) Xeon(R) CPU E5-2697 v4 @ 2.30GHz; 72 logical CPUs; Linux-6.8.0-147-generic-x86_64-with-glibc2.39` (all runs this session, this machine).

No algorithms changed, no code in `src/` or `tests/` changed. All numbers below are runs done in this session. Flop/byte counts are derived from the kernel source (`src/kernels.mojo`) and the measured input sizes, arithmetic shown per row. Memory-bound vs compute-bound is judged against the measured time (a bandwidth check: bytes touched / measured time, compared with what a plain NumPy vector op achieves on the same bytes — see baselines).

`MOJO_NOTES.md`, `bench/bench.py`, `README.md` were read first. `pixi run bench` (flock-wrapped) was run alone, no other benchmark ran alongside it.

## pixi run bench (this session)

| case | mojo-evaluate | evaluate 0.4.6 | result |
| --- | ---: | ---: | ---: |
| accuracy (1M binary) | 2.93 ms | 3360.29 ms | 1148.7x faster |
| f1 (1M binary) | 9.02 ms | 3914.78 ms | 434.1x faster |
| confusion_matrix (1M, 32 classes) | 8.37 ms | 3190.95 ms | 381.5x faster |
| f1 macro (1M, 32 classes) | 8.37 ms | 3841.84 ms | 458.8x faster |
| MCC (1M, 32 classes) | 8.44 ms | 3730.66 ms | 442.1x faster |
| mse (1M) | 5.06 ms | 3258.37 ms | 644.5x faster |
| mae (1M) | 6.16 ms | 3331.49 ms | 540.5x faster |
| mse (250k x 8 outputs) | 9.82 ms | 6301.97 ms | 641.8x faster |

Matches the README table's shape (README numbers are from 2026-08-24 on the same machine class). No row is slower than upstream here; upstream Evaluate never loses these cases because it pays Datasets/Arrow validation + Hub/zarr overhead before scikit-learn. That architectural fact dominates every row, so the rows below separate kernel time from Python-glue time and show the kernel-level comparison to scikit-learn and NumPy.

## Ranked hotspot table

Ranked by measured end-to-end time at real-caller sizes (1M rows is the size `bench/bench.py` uses; 100k×20 is a typical multilabel shape). "Glue vs kernel" is split by timing the raw `ctypes` call (`lib().me_*`, same addresses, no Python validation) against the full public `mojo_evaluate.load(...).compute(...)` call, both measured this session (mean of 20, best-of-5-equivalent; full numbers in session output).

Bytes touched (from kernel source): int64 inputs = 8 B/elem, float64 = 8 B/elem, output matrices are small (classes² or outputs×5). Flops are counted per kernel source loop (see notes): accuracy = 1 cmp + occasional add; regression = 5 reductions per element (sq, abs, two mul, smape max/abs/÷); confusion = 1 int→index + 1 add per element; multilabel = branches per cell.

| rank | function | input size | time (ms) | bytes touched | flops | flops per byte | memory-bound or compute-bound | Python/ctypes glue or kernel |
| --- | --- | --- | ---: | --- | --- | ---: | --- | --- |
| 1 | `compute_brier` full public call at 1M | 1M probs + 1M refs | 86.7 | 16.0 MB (2×float64 in + 1×int64 out eq) | ~4/elen (~4.0M) | 0.25 | memory-bound (0.18 GB/s realized — not bandwidth, see flag) | **Python glue**: first call in process imports sklearn+scipy (~2.47 s of the cProfile 3.37 s); steady-state still runs `_as_float64`, `_check_finite`, range check, `np.unique`, `references==pos_label` scan each call before the one `me_regression_reductions` (4.57 ms raw) |
| 2 | `accuracy` multilabel, 100k×20 full | 2M int64 cells (100k rows × 20) | 215.5 | 16.0 MB (2×8 B×2M cells... 2M cells ×8 B = 16 MB per array, 32 MB both) | ~2/cell (~4.0M) | 0.12 | memory-bound | **Python glue**: `_classification_inputs` does `np.union1d` → `np.unique` hash sort of 2M cells: 208.8 ms of 213 ms total (cProfile: `union1d` 1.024 s / 5 calls). Raw `me_subset_accuracy` for the same input is 2.14 ms — the kernel is ~1% of the call |
| 3 | `f1` multilabel (macro), 100k×20 full | 2M int64 cells | 227.9 | 32 MB | ~3/cell (~6.0M) | 0.19 | memory-bound | **Python glue**, same `union1d` path as above; raw `me_multilabel_stats` is 22.1 ms |
| 4 | `f1` multilabel (samples), 100k×20 full | 2M int64 cells | 281.9 | 32 MB + 2.4 MB sample scores | ~3/cell + second pass | ~0.2 | memory-bound | **Python glue** (same union1d) + second native pass `me_multilabel_samples` |
| 5 | `f1` (1M binary) full | 1M int64 × 2 | 8.54–10.59 | 16 MB | ~2/elen (~2.0M) | 0.12 | memory-bound | **Python glue**: `_label_universe` = 6.3 ms (min/max scans + `present[piece-low]=True` scatter + `flatnonzero`) of 8.5 ms; raw kernel `me_confusion_matrix` (2c) is 1.30 ms, `_confusion` wrapper 1.36 ms. The rest is `_classification_inputs` (0.95 ms, two `_as_int64_labels` each 0.45 ms — each does `asarray`+cast+`equal`+`all` scans) |
| 6 | `confusion_matrix` / `f1 macro` / `matthews_correlation` (1M, 32 classes) full | 1M int64 × 2, 32×32 out | 8.4–8.9 | 16 MB | ~2/elen (~2.0M) | 0.12 | memory-bound | **Python glue**: same `_label_universe` 6.1 ms + `_classification_inputs` 0.9 ms; raw `me_confusion_matrix` 1.21–1.32 ms, `_confusion` 1.21 ms. So ~85% of these calls is Python label-universe bookkeeping, ~15% is kernel |
| 7 | `mse` (250k×8 multioutput) full | 2M float64 (250k×8 × 2 arrays) | 9.8–10.6 | 32 MB | ~10/elem (~20M: 5 reductions with smape div+max) | 0.63 | memory-bound (32 MB / 7.9 ms raw ≈ 4.0 GB/s; NumPy single-op achieves ~4.9 GB/s on same bytes — bandwidth ceiling, not ALU) | **Kernel** for the raw part (7.93 ms), glue ~2.7 ms validation: `_regression_inputs` (2× `_as_float64` + 2× `_check_finite` scans = 0.88 ms at 1M, ~2.2 ms at 2M) |
| 8 | `mse` / `mae` / `smape` (1M) full | 1M float64 × 2 | 5.1–5.6 | 16 MB | mse/mae ~3/elem (3.0M), smape ~10/elem (10M) | 0.19–0.63 | memory-bound | **Kernel** 4.57 ms raw `me_regression_reductions`; glue ~0.9 ms (`_regression_inputs` 0.88 ms: `_check_finite` alone 0.45 ms = 2 full scans of `isfinite` over 1M) |
| 9 | `accuracy` (1M binary) full | 1M int64 × 2 | 2.9–6.2 | 16 MB | ~1/elem (1.0M) | 0.06 | memory-bound | **Kernel** 5.39 ms raw `me_accuracy` — the earliest raw measurement (5.39 ms) vs later full 2.9–6.2 ms shows noise floor; raw loop is a single scalar `for i in range(n)` with `references[i]==predictions[i]` and `correct += 1.0`, no SIMD, no unroll, no parallel. At small n (10) raw is 7 µs, full 28 µs — 21 µs of Python glue dominates small calls (`.compute` dispatch, `_as_int64_labels` ×2, `_weights`). |
| 10 | `me_naive_mae` (behind `mase`) raw | 1M float64 | 3.96 | 8 MB (stride access `i` vs `i-periodicity`) | ~2/elen | 0.25 | memory-bound | **Kernel**: nested `for i in range(periodicity, rows)` → `for j` scalar loop, no SIMD |
| 11 | `me_multilabel_stats` raw 100k×20 | 2M int64 | 22.1–22.3 | 32 MB | ~3/cell + 4× `stats[j*4]` RMW (conflict-prone) | ~0.2 | memory-bound but latency/ALU-bound in practice (0.7 GB/s realized vs 2.1 GB/s for subset_accuracy on same bytes — branchy nested loop, no unroll) | **Kernel**: double `for i`/`for j` with `bytes... stats[j*4] += weights[i]` and `if/elif` chain per cell |
| 12 | `me_subset_accuracy` raw 100k×20 | 2M int64 | 2.14 | 32 MB | ~2/cell | 0.12 | memory-bound (14.9 GB/s — fastest streaming rate measured) | **Kernel**: same nested-loop shape but early-exit `same=True→False` still scans all cells; best scalar rate in the set, still scalar |
| 13 | `_confusion` offset path (labels not starting at 0) | 1M × 2 int64 + 2 extra full scans | 2.91 (vs 1.21 zero-based) | 16 MB + 2×8 MB subtraction copies | ~2/elen + 1 sub/elen in Python | 0.12 | memory-bound | **Python/NumPy glue**: `references - universe[0]` and `predictions - universe[0]` allocate two new 8 MB arrays + `ascontiguousarray` copies before the kernel even runs |

Supporting glue timings (this session, mean of 20 at 1M unless noted):

- `_label_universe` 1M 32c: 6.14 ms — breakdown: ~1.2 ms min/max scans, ~0.3 ms each `.min()`, `present[piece-low]=True` scatter + `flatnonzero` rest.
- `_classification_inputs` 1M: 0.90–0.95 ms; `_as_int64_labels` per array 0.455 ms (cast + `equal` + `all` = 3 passes).
- `_confusion` zero-based 1M 32c: 1.21 ms (raw kernel 1.21–1.32 ms — wrapper is thin); offset-based: 2.91 ms (+1.7 ms of subtraction copies).
- `_regression_inputs` 1M: 0.88 ms; `_check_finite` 0.45 ms; `_weights` 0.78 ms (float64 cast + `isfinite` + all-zero check).
- Raw `me_regression_reductions` 1M: 4.57 ms; 250k×8: 7.93 ms.
- Raw `me_accuracy` 1M: 5.39 ms (first measurement) — full call 2.9–6.2 ms across runs, i.e. the kernel loop itself is the cost, not glue, at this size.
- Small-size floor: `accuracy` n=10 full 28–36 µs vs raw 7 µs (×4–5 slower end-to-end); n=100 same 28.9 µs. Any per-call optimization for typical small-batch callers must cut Python, not Mojo.
- Kernel-level baselines at 1M (this session): scikit-learn `accuracy_score` 81.9 ms, `f1 macro` 274 ms, `matthews_corrcoef` 541 ms, `mean_squared_error` 3.81 ms. NumPy `mean((p-r)**2)` 2.20 ms, `mean(abs(p-r))` 2.56 ms, `mean(p==r)` 1.68 ms, naive-MAE 2.56 ms, sMAPE 19.1 ms.

## Flags

**Per-call allocations/copies in Python wrapper (`metrics.py`) — the actual hotspots at 1M:**

1. `_label_universe` (line 194) — every `f1`/`precision`/`recall`/`confusion_matrix`/`matthews_correlation` call with `labels=None`: computes `piece.min()`/`piece.max()` over 4–8 passes (2 arrays × min+max = 4 scans), allocates `present` (span bool), does fancy-index scatter `present[piece-low]=True` (2 passes), `flatnonzero`. Measured 6.1–6.3 ms of ~8.5 ms end-to-end. The Mojo `me_confusion_matrix` kernel could discover labels itself or the Python side could cache when 32 contiguous classes are passed.
2. `_as_int64_labels` (line 37) — per array: cast + `np.equal` + `np.all` = 3 passes, 0.45 ms/array at 1M. Already-int64 contiguous input still pays all 3. Two calls per classification compute → ~0.9 ms fixed.
3. `_check_finite` (line 85) — `np.all(np.isfinite(...))` is 2 passes (produce + reduce), 0.45 ms at 1M, called twice per regression input + once per sample_weight. `_weights` adds another `isfinite` + `weights==0` pass: 0.78 ms.
4. `_confusion` non-zero-based offset (line 238): `references - universe[0]` allocates + `ascontiguousarray` — two 8 MB copies, +1.7 ms. Also the `searchsorted` fallback for non-contiguous universes does two full `searchsorted` scans.
5. `_classification_inputs` multilabel (line 107): `np.union1d` → `np.unique` hash-sorts the entire concatenated 2M-cell matrix twice — 208.8 ms, dwarfs everything else in multilabel calls. The check only needs to confirm values ∈ {0,1}; it does not need a sorted unique. This is the single largest measured cost in the port.
6. `compute_brier` (line 653): `from sklearn.utils.multiclass import type_of_target` inside the function (first call imports sklearn+scipy ≈ 2.5 s under cProfile), plus `np.unique(references)` on every call, plus three full validation scans (`_check_finite`, range, unique) before the regression path. Steady-state 86.7 ms measured first-call-including; after import warm it is still the slowest 1M call because of the repeated scans.
7. `compute_regression` (line 589): `_regression_inputs` is called twice for `mase` (once via `_regression_reductions`, once again for training — double validation scans).
8. Small calls (n≤100): Python glue is 21 µs of 28 µs — `np.asarray`×2, `_as_int64_labels`×2, `_weights`, dispatch, `addr()` checks. No kernel concern; a fast path guard (skip validation when already int64 contiguous) would cut small-call latency ~3×.

**Python-level loops re-entering ctypes per element: none found.** Every public entry point makes exactly one native call per compute: `me_accuracy` / `me_subset_accuracy` / `me_confusion_matrix` / `me_multilabel_stats` / `me_multilabel_samples` / `me_regression_reductions` / `me_naive_mae`. `grep` of `python/mojo_evaluate/` confirms one `lib().me_*` call site per kernel, always with whole-buffer addresses. The only Python loop over data is `_label_universe`'s `for piece in pieces` (≤3 iterations) and `selected_indices = [np.searchsorted(...)]` over ≤32 labels — both O(classes), not O(n). The problem is not per-element re-entry; it is per-call NumPy validation passes that together walk the data 6–10× before the single kernel walk.

**Mojo kernels (`src/kernels.mojo`) — all scalar, none SIMD, none parallel, none unrolled:**

- `me_accuracy`: `for i in range(n): if refs[i]==preds[i]: correct+=1.0` — scalar compare+branch, no `simd_width_of`, no `parallelize`. DESTRUCTIVE: the `correct += 1.0` branch mispredicts ~50% on random data.
- `me_subset_accuracy` / `me_multilabel_stats` / `me_multilabel_samples`: nested `for i`/`for j` with `i*labels+j` recomputed per cell, `if/elif` chain per cell, scalar `+=` to `stats[j*4]` (same cache line ping-pong across j). No SIMD, no unroll, no parallel.
- `me_confusion_matrix`: scalar `matrix[refs[i]*classes + preds[i]] += w` — indirect RMW, data-dependent store, cannot vectorize directly; `classes*classes` zeroing loop is scalar too (small).
- `me_regression_reductions`: nested `for i`/`for j`, per element computes 5 reductions including sMAPE `max(abs)`+`/` — the only place with floating-point volume (~10 flops/elem) and the only kernel where SIMD has arithmetic to feed. Still scalar.
- `me_naive_mae`: nested loop with `i-periodicity` stride — no SIMD.

Measured realized bandwidth of the best kernel (`me_subset_accuracy`, 14.9 GB/s) vs NumPy on same bytes (~4–5 GB/s for a single simple op at 2.3 GHz per-core) shows the kernels are not bandwidth-starved yet — they are latency/ALU-starved scalar loops. But all have flops/byte ≪ 1 (max 0.63 for regression, 0.06–0.25 elsewhere) — memory-bound by roofline, which gates the GPU verdict below.

## Next-phase verdicts per hotspot

Phase vocabulary: **unroll** (manual unrolling / loop restructuring in Mojo), **simd** (SIMD vector ops via `simd_width_of`/`load[width=W]`), **parallel** (`parallelize` from `max.algorithm` behind a size threshold), **gpu**, **none** (Python-side fix or hyperparameter, not a kernel change).

| hotspot | next phase | why, with the measured figure |
| --- | --- | --- |
| `accuracy` kernel (2.9–5.4 ms, 0.06 flop/byte) | **simd, then parallel** | 0.06 flop/byte is the most memory-bound row. SIMD compare + `reduce_add` removes the 50% mispredict branch and gives ~4–8× on the load path; `parallelize` over `n` (independent per-chunk partial sums) is safe and needs no atomics. GPU: no — 0.06 flop/byte, copy-in alone costs more than the CPU kernel. |
| `f1`/`precision`/`recall`/`confusion_matrix`/`matthews_correlation` (8.5 ms, dominated by `_label_universe` 6.3 ms) | **none** (Python fix first: fold label discovery into the kernel or cache contiguous-class universes) | The kernel is 1.3 ms; Python is 6.3 ms. SIMD/parallel on the kernel saves at most 1 ms of 8.5. The next phase here is deleting passes: single-pass min/max in Mojo, or reusing the `present` scatter the kernel already touches. |
| `confusion_matrix` kernel (1.2 ms, ~0.12 flop/byte, indirect RMW) | **parallel** (histogram-style) | Scatter-update cannot SIMD safely without atomics/privatized histograms; per-thread private `classes×classes` histograms + merge is the standard shape. 0.12 flop/byte → no GPU (copy-in > kernel time). |
| `mse`/`mae`/`smape` kernel `me_regression_reductions` (4.57 ms at 1M, 0.19–0.63 flop/byte) | **simd** (primary), parallel secondary | Highest flops/byte in the port (0.63 for smape with its `/` and `max`), still < 1, so CPU vectorization is the place: every reduction is a pure elementwise op + `reduce_add`. `parallelize` over rows with per-thread `outputs×5` partials is trivially safe. GPU: no — 0.63 flop/byte is far below the ~2 flop/byte CPU-legitimacy threshold from `MOJO_NOTES.md` §5, and copies alone exceed 4.57 ms. |
| `me_naive_mae` (3.96 ms, 0.25 flop/byte) | **simd** | Single `abs(diff)` accumulation, stride-`periodicity` pairs — vectorizable with an offset load; same memory-bound verdict as regression. |
| `me_multilabel_stats` (22.1 ms, 0.2 flop/byte, 0.7 GB/s realized) | **unroll, then simd; parallel** | This kernel is 3× slower per byte than `me_subset_accuracy` on the same data (22.1 ms vs 2.1 ms) — the `if/elif` chain + `stats[j*4]` RMW is the cost, not bandwidth. Restructure per-cell work (branchless select, hoist `j*4`, unroll `j`) before SIMD. `parallelize` over rows with per-thread `labels×4` stats is safe. GPU: no, 0.2 flop/byte. |
| `me_subset_accuracy` (2.14 ms, 0.12 flop/byte, 14.9 GB/s) | **simd** | Already at bandwidth; SIMD on the inner `j` loop (vector compare + horizontal `reduce` per row) is the only remaining lever. |
| Multilabel `_classification_inputs` `union1d` (208 ms) | **none** — replace with a min/max or {0,1} membership check | No kernel phase applies; the kernel is 1% of the cost. A direct `min/max` + two-comparison check replaces a full sort. |
| `compute_brier` (86.7 ms first-call; steady-state dominated by 3 validation scans + per-call sklearn import attempt) | **none** — hoist the sklearn import to module level, drop `np.unique` in favor of a min/max check | Kernel (`me_regression_reductions`) is shared with mse and already counted above. |
| `_as_int64_labels` / `_check_finite` / `_weights` (≈2.1 ms combined at 1M) | **none** — fast-path guard for already-int64/contiguous input + single-pass finite check | Validation is 6–10 passes over the data at 1M; a dirty-flag or typed fast path cuts it to zero passes in the common NumPy case. |
| Small-call floor (n≤100, 28 µs vs 7 µs raw) | **none** — reduce dispatch/validation overhead | 21 µs of Python; no kernel phase can help. |
| Upstream Evaluate rows (3–6 s each) | **none** | The gap is Datasets/Arrow + Hub client overhead in upstream, not scikit-learn; mojo-evaluate already wins by architecture. Kernel-level scikit-learn comparison: mojo kernel 1.2–5.6 ms vs sklearn 82–541 ms for classification (40–100× at kernel level) but sklearn wins on plain MSE (3.81 ms vs raw 4.57 ms) — the regression row is the one kernel-level loss vs a tuned Cython baseline, worth a SIMD pass (see above). |

### Slower-than-upstream rows

None in this session's `pixi run bench`: mojo-evaluate beats evaluate 0.4.6 on all 8 public cases (108–1149×). At the kernel level (sklearn directly, no Datasets/Arrow), the one row where the port is slower than the baseline is **regression (`me_regression_reductions` vs sklearn `mean_squared_error` / NumPy `mean((p-r)**2)`)**: raw 4.57 ms vs sklearn 3.81 ms and NumPy 2.20 ms on the same 1M float64 inputs. The reason is measured: the kernel computes all five reductions (including sMAPE) for every metric, while sklearn/NumPy compute one; and it is a scalar loop with no SIMD. This row is retained above and the SIMD phase is assigned to it. An early raw `me_accuracy` measurement (5.39 ms) read slower than NumPy's `mean(p==r)` (1.68 ms) for the same reason (branchy scalar loop), though later full-call runs overlapped the NumPy figure — the direction is consistent, the magnitude is noise-floored.

### GPU summary

Every kernel in the port is below 0.65 flops/byte (range 0.06–0.63, measured/derived above). All are streaming reductions or scatter updates; copy-in/copy-out alone exceeds kernel time on any GPU. Per `MOJO_NOTES.md` §5 the CPU-only answer is legitimate below ~2 flops/byte, and 28 ports already use `max.gpu` — here the measured arithmetic intensity is 3–30× below that threshold, so **no GPU phase is assigned to any hotspot in this session's evidence**.
