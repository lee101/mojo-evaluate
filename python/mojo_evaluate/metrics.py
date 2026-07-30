from __future__ import annotations

from collections.abc import Sequence
import math
import warnings

import numpy as np

from ._lib import addr, lib


SUPPORTED_METRICS = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "confusion_matrix",
    "matthews_correlation",
    "mse",
    "mae",
    "brier_score",
    "mase",
    "smape",
)

_CLASSIFICATION = {
    "accuracy",
    "precision",
    "recall",
    "f1",
    "confusion_matrix",
    "matthews_correlation",
}
_REGRESSION = {"mse", "mae", "mase", "smape"}


def _as_int64_labels(values, name: str) -> np.ndarray:
    original = np.asarray(values)
    if original.dtype.kind not in "biuf":
        raise ValueError(f"{name} must contain integer labels")
    try:
        with np.errstate(invalid="ignore", over="ignore"):
            converted = np.asarray(original, dtype=np.int64)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain integer labels") from error
    # Casting floats, uint64, or wider numeric types can otherwise truncate,
    # wrap, or map non-finite values to an arbitrary class silently.
    with np.errstate(invalid="ignore"):
        exact = np.equal(original, converted)
    if not bool(np.all(exact)):
        raise ValueError(f"{name} must contain exactly representable int64 labels")
    return np.ascontiguousarray(converted)


def _as_float64(values, name: str) -> np.ndarray:
    original = np.asarray(values)
    if original.dtype.kind == "c":
        raise ValueError(f"{name} must be real-valued")
    try:
        converted = np.asarray(original, dtype=np.float64)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be real-valued") from error
    return np.ascontiguousarray(converted)


def _weights(sample_weight, rows: int) -> np.ndarray | None:
    if sample_weight is None:
        return None
    weights = _as_float64(sample_weight, "sample_weight").reshape(-1)
    if weights.size != rows:
        raise ValueError(
            f"sample_weight has {weights.size} values, expected {rows}"
        )
    return weights


def _classification_inputs(predictions, references, multilabel: bool):
    predictions = np.asarray(predictions)
    references = np.asarray(references)
    if predictions.shape != references.shape:
        raise ValueError(
            f"predictions and references have different shapes: "
            f"{predictions.shape} != {references.shape}"
        )
    expected_ndim = 2 if multilabel else 1
    if predictions.ndim != expected_ndim:
        kind = "two-dimensional" if multilabel else "one-dimensional"
        raise ValueError(f"predictions and references must be {kind}")
    if predictions.size == 0:
        raise ValueError("predictions and references must not be empty")
    if multilabel:
        values = np.union1d(predictions, references)
        if not np.all(np.isin(values, (0, 1))):
            raise ValueError("multilabel indicators must contain only 0 and 1")
    predictions = _as_int64_labels(predictions, "predictions")
    references = _as_int64_labels(references, "references")
    return predictions, references


def _regression_inputs(predictions, references):
    predictions = _as_float64(predictions, "predictions")
    references = _as_float64(references, "references")
    if predictions.shape != references.shape:
        raise ValueError(
            f"predictions and references have different shapes: "
            f"{predictions.shape} != {references.shape}"
        )
    if predictions.ndim == 1:
        predictions = predictions[:, None]
        references = references[:, None]
    elif predictions.ndim != 2:
        raise ValueError("regression inputs must be one- or two-dimensional")
    if predictions.shape[0] == 0:
        raise ValueError("predictions and references must not be empty")
    return (
        predictions,
        references,
    )


def _aggregate(values: np.ndarray, multioutput):
    if isinstance(multioutput, str):
        if multioutput == "raw_values":
            return values
        if multioutput != "uniform_average":
            raise ValueError(
                "multioutput must be 'raw_values', 'uniform_average', "
                "or output weights"
            )
        result = np.average(values)
    else:
        output_weights = np.asarray(multioutput, dtype=np.float64)
        if output_weights.shape != values.shape:
            raise ValueError(
                f"multioutput has shape {output_weights.shape}, "
                f"expected {values.shape}"
            )
        result = np.average(values, weights=output_weights)
    return float(result)


def _regression_reductions(predictions, references, sample_weight):
    predictions, references = _regression_inputs(predictions, references)
    rows, outputs = references.shape
    weights = _weights(sample_weight, rows)
    reductions = np.empty((outputs, 5), dtype=np.float64)
    lib().me_regression_reductions(
        addr(references),
        addr(predictions),
        addr(weights),
        addr(reductions),
        rows,
        outputs,
        weights is not None,
    )
    denominator = weights.sum() if weights is not None else float(rows)
    if denominator == 0:
        raise ZeroDivisionError("Weights sum to zero, can't be normalized")
    return reductions, denominator


def _label_universe(predictions, references, labels=None, pos_label=None):
    selected = (
        None
        if labels is None
        else _as_int64_labels(labels, "labels").reshape(-1)
    )
    if selected is not None:
        if selected.size == 0:
            raise ValueError("labels must contain at least one label")
        if np.unique(selected).size != selected.size:
            raise ValueError("labels must not contain duplicates")
    pieces = [references, predictions]
    if selected is not None:
        pieces.append(selected)
    if pos_label is not None:
        pieces.append(_as_int64_labels([pos_label], "pos_label"))
    low = min(int(piece.min()) for piece in pieces if piece.size)
    high = max(int(piece.max()) for piece in pieces if piece.size)
    span = high - low + 1
    if span <= 1_000_000:
        present = np.zeros(span, dtype=bool)
        for piece in pieces:
            present[piece - low] = True
        universe = np.flatnonzero(present).astype(np.int64) + low
    else:
        universe = np.unique(np.concatenate(pieces)).astype(
            np.int64, copy=False
        )
    if selected is None:
        selected = universe
    return universe, selected.astype(np.int64, copy=False)


def _confusion(predictions, references, sample_weight, universe):
    contiguous_labels = (
        universe[-1] - universe[0] + 1 == universe.size
        and np.array_equal(
            universe, np.arange(universe[0], universe[-1] + 1, dtype=np.int64)
        )
    )
    if contiguous_labels and universe[0] == 0:
        reference_codes = references
        prediction_codes = predictions
    elif contiguous_labels:
        reference_codes = np.ascontiguousarray(references - universe[0])
        prediction_codes = np.ascontiguousarray(predictions - universe[0])
    else:
        reference_codes = np.ascontiguousarray(
            np.searchsorted(universe, references), dtype=np.int64
        )
        prediction_codes = np.ascontiguousarray(
            np.searchsorted(universe, predictions), dtype=np.int64
        )
    weights = _weights(sample_weight, references.size)
    matrix = np.empty((universe.size, universe.size), dtype=np.float64)
    lib().me_confusion_matrix(
        addr(reference_codes),
        addr(prediction_codes),
        addr(weights),
        addr(matrix),
        references.size,
        universe.size,
        weights is not None,
    )
    return matrix


def _zero_value(zero_division):
    if zero_division == "warn":
        warnings.warn(
            "Metric is ill-defined and being set to 0.0 due to zero division",
            RuntimeWarning,
            stacklevel=3,
        )
        return 0.0
    if isinstance(zero_division, str) or zero_division not in (0, 1):
        try:
            if math.isnan(float(zero_division)):
                return np.nan
        except (TypeError, ValueError):
            pass
        raise ValueError("zero_division must be 'warn', 0, 1, or np.nan")
    return float(zero_division)


def _divide(numerator, denominator, zero_division):
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    result = np.empty_like(numerator)
    valid = denominator != 0
    np.divide(numerator, denominator, out=result, where=valid)
    if np.any(~valid):
        result[~valid] = _zero_value(zero_division)
    return result


def _multilabel_counts(predictions, references, sample_weight):
    rows, labels = references.shape
    weights = _weights(sample_weight, rows)
    stats = np.empty((labels, 4), dtype=np.float64)
    lib().me_multilabel_stats(
        addr(references),
        addr(predictions),
        addr(weights),
        addr(stats),
        rows,
        labels,
        weights is not None,
    )
    return stats, weights


def _metric_from_counts(
    metric,
    tp,
    fp,
    fn,
    support,
    average,
    zero_division,
    sample_scores=None,
    sample_weight=None,
):
    if metric == "precision":
        scores = _divide(tp, tp + fp, zero_division)
    elif metric == "recall":
        scores = _divide(tp, tp + fn, zero_division)
    else:
        scores = _divide(2.0 * tp, 2.0 * tp + fp + fn, zero_division)

    if average is None:
        return scores if scores.size > 1 else float(scores[0])
    if average == "micro":
        if metric == "precision":
            return float(_divide(tp.sum(), (tp + fp).sum(), zero_division))
        if metric == "recall":
            return float(_divide(tp.sum(), (tp + fn).sum(), zero_division))
        return float(
            _divide(
                2.0 * tp.sum(),
                (2.0 * tp + fp + fn).sum(),
                zero_division,
            )
        )
    if average == "macro":
        return float(np.mean(scores))
    if average == "weighted":
        if support.sum() == 0:
            return 0.0
        return float(np.average(scores, weights=support))
    if average == "samples":
        if sample_scores is None:
            raise ValueError("average='samples' is only valid for multilabel inputs")
        values = _metric_from_counts(
            metric,
            sample_scores[:, 0],
            sample_scores[:, 1],
            sample_scores[:, 2],
            sample_scores[:, 0] + sample_scores[:, 2],
            None,
            zero_division,
        )
        values = np.atleast_1d(values)
        return float(np.average(values, weights=sample_weight))
    raise ValueError(
        "average must be one of 'binary', 'micro', 'macro', 'weighted', "
        "'samples', or None"
    )


def compute_classification(
    metric,
    predictions,
    references,
    *,
    config_name=None,
    normalize=True,
    labels=None,
    pos_label=1,
    average="binary",
    sample_weight=None,
    zero_division="warn",
):
    multilabel = config_name == "multilabel"
    predictions, references = _classification_inputs(
        predictions, references, multilabel
    )
    rows = references.shape[0]
    weights = _weights(sample_weight, rows)

    if metric == "accuracy":
        if multilabel:
            correct = lib().me_subset_accuracy(
                addr(references),
                addr(predictions),
                addr(weights),
                rows,
                references.shape[1],
                weights is not None,
            )
        else:
            correct = lib().me_accuracy(
                addr(references),
                addr(predictions),
                addr(weights),
                rows,
                weights is not None,
            )
        if not normalize:
            return {"accuracy": float(correct) if weights is not None else int(correct)}
        denominator = weights.sum() if weights is not None else rows
        if denominator == 0:
            raise ValueError("sample_weight must have a non-zero sum")
        return {"accuracy": float(correct / denominator)}

    if metric == "confusion_matrix":
        if multilabel:
            raise ValueError(
                "upstream confusion_matrix does not accept multilabel indicators"
            )
        universe, selected = _label_universe(
            predictions, references, labels=labels
        )
        matrix = _confusion(predictions, references, sample_weight, universe)
        selected_indices = [
            int(np.searchsorted(universe, label)) for label in selected
        ]
        matrix = matrix[np.ix_(selected_indices, selected_indices)]
        if normalize is not None:
            if normalize == "true":
                denominator = matrix.sum(axis=1, keepdims=True)
            elif normalize == "pred":
                denominator = matrix.sum(axis=0, keepdims=True)
            elif normalize == "all":
                denominator = matrix.sum()
            else:
                raise ValueError("normalize must be one of 'true', 'pred', 'all', or None")
            matrix = np.divide(
                matrix,
                denominator,
                out=np.zeros_like(matrix),
                where=denominator != 0,
            )
        elif weights is None:
            matrix = matrix.astype(np.int64)
        return {"confusion_matrix": matrix}

    if metric == "matthews_correlation":
        if multilabel:
            stats, _ = _multilabel_counts(
                predictions, references, sample_weight
            )
            tp, fp, fn, tn = stats.T
            denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            values = np.divide(
                tp * tn - fp * fn,
                denominator,
                out=np.zeros_like(tp),
                where=denominator != 0,
            )
            if average == "macro":
                result = float(values.mean())
            elif average is None:
                result = values.tolist()
            else:
                raise ValueError("average must be 'macro' or None")
            return {"matthews_correlation": result}

        universe, _ = _label_universe(predictions, references)
        matrix = _confusion(predictions, references, sample_weight, universe)
        total = matrix.sum()
        true_sum = matrix.sum(axis=1)
        pred_sum = matrix.sum(axis=0)
        numerator = np.trace(matrix) * total - np.dot(true_sum, pred_sum)
        denominator = math.sqrt(
            max(0.0, total * total - np.dot(pred_sum, pred_sum))
            * max(0.0, total * total - np.dot(true_sum, true_sum))
        )
        result = numerator / denominator if denominator else 0.0
        return {"matthews_correlation": float(result)}

    if multilabel:
        stats, weights = _multilabel_counts(
            predictions, references, sample_weight
        )
        selected = (
            np.arange(references.shape[1])
            if labels is None
            else _as_int64_labels(labels, "labels")
        )
        if np.any((selected < 0) | (selected >= references.shape[1])):
            raise ValueError("labels contains an invalid column index")
        tp, fp, fn, _ = stats[selected].T
        sample_scores = None
        if average == "samples":
            selected_references = np.ascontiguousarray(
                references[:, selected], dtype=np.int64
            )
            selected_predictions = np.ascontiguousarray(
                predictions[:, selected], dtype=np.int64
            )
            sample_scores = np.empty((rows, 3), dtype=np.float64)
            lib().me_multilabel_samples(
                addr(selected_references),
                addr(selected_predictions),
                0,
                addr(sample_scores),
                rows,
                selected.size,
                0,
            )
        result = _metric_from_counts(
            metric,
            tp,
            fp,
            fn,
            tp + fn,
            average,
            zero_division,
            sample_scores,
            weights,
        )
        return {metric: result}

    if average == "binary":
        selected_labels = np.array([pos_label], dtype=np.int64)
    else:
        selected_labels = labels
    universe, selected = _label_universe(
        predictions,
        references,
        labels=selected_labels,
        pos_label=pos_label if average == "binary" else None,
    )
    if average == "binary" and universe.size > 2:
        raise ValueError(
            "Target is multiclass but average='binary'. Choose another average."
        )
    matrix = _confusion(predictions, references, sample_weight, universe)
    selected_indices = np.array(
        [np.searchsorted(universe, label) for label in selected], dtype=np.int64
    )
    diagonal = np.diag(matrix)
    tp = diagonal[selected_indices]
    fp = matrix[:, selected_indices].sum(axis=0) - tp
    fn = matrix[selected_indices, :].sum(axis=1) - tp
    result = _metric_from_counts(
        metric,
        tp,
        fp,
        fn,
        tp + fn,
        None if average == "binary" else average,
        zero_division,
    )
    return {metric: result}


def compute_regression(
    metric,
    predictions,
    references,
    *,
    sample_weight=None,
    multioutput="uniform_average",
    squared=True,
    training=None,
    periodicity=1,
):
    reductions, denominator = _regression_reductions(
        predictions, references, sample_weight
    )
    if metric == "mse":
        values = reductions[:, 0] / denominator
        if not squared:
            values = np.sqrt(values)
        return {"mse": _aggregate(values, multioutput)}
    if metric == "mae":
        values = reductions[:, 1] / denominator
        return {"mae": _aggregate(values, multioutput)}
    if metric == "smape":
        values = reductions[:, 4] / denominator
        return {"smape": _aggregate(values, multioutput)}
    if training is None:
        raise TypeError("mase.compute() missing required keyword argument: 'training'")
    predictions, references = _regression_inputs(predictions, references)
    training = _as_float64(training, "training")
    if training.ndim == 1:
        training = training[:, None]
    if training.ndim != 2 or training.shape[1] != references.shape[1]:
        raise ValueError(
            "training must have the same number of outputs as references"
        )
    if not isinstance(periodicity, (int, np.integer)) or periodicity < 1:
        raise ValueError("periodicity must be a positive integer")
    if periodicity >= training.shape[0]:
        raise ValueError("periodicity must be smaller than the training length")
    naive = np.empty(training.shape[1], dtype=np.float64)
    lib().me_naive_mae(
        addr(training),
        addr(naive),
        training.shape[0],
        training.shape[1],
        periodicity,
    )
    naive /= training.shape[0] - periodicity
    actual = reductions[:, 1] / denominator
    epsilon = np.finfo(np.float64).eps
    if isinstance(multioutput, str) and multioutput == "raw_values":
        result = actual / np.maximum(naive, epsilon)
    else:
        actual_agg = _aggregate(actual, multioutput)
        naive_agg = _aggregate(naive, multioutput)
        result = actual_agg / max(naive_agg, epsilon)
    return {"mase": result}


def compute_brier(
    predictions,
    references,
    *,
    sample_weight=None,
    pos_label=1,
):
    predictions = _as_float64(predictions, "predictions").reshape(-1)
    references = np.asarray(references).reshape(-1)
    if predictions.shape != references.shape:
        raise ValueError("predictions and references have different lengths")
    if np.any((predictions < 0.0) | (predictions > 1.0)):
        raise ValueError("predictions must be probabilities in [0, 1]")
    binary_references = np.ascontiguousarray(
        references == pos_label, dtype=np.float64
    )
    reductions, denominator = _regression_reductions(
        predictions, binary_references, sample_weight
    )
    return {"brier_score": float(reductions[0, 0] / denominator)}
