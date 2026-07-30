from __future__ import annotations

from functools import lru_cache

import numpy as np
import pytest

upstream_evaluate = pytest.importorskip("evaluate")

import mojo_evaluate


@lru_cache
def upstream(name, config=None):
    return upstream_evaluate.load(name, config)


def assert_result_equal(ours, theirs, *, rtol=2e-6, atol=2e-7):
    assert ours.keys() == theirs.keys()
    for key in ours:
        ours_value = ours[key]
        theirs_value = theirs[key]
        if isinstance(ours_value, (list, np.ndarray)) or isinstance(
            theirs_value, (list, np.ndarray)
        ):
            assert np.allclose(
                ours_value, theirs_value, rtol=rtol, atol=atol, equal_nan=True
            )
        else:
            assert ours_value == pytest.approx(
                theirs_value, rel=rtol, abs=atol, nan_ok=True
            )


def compare(name, predictions, references, config=None, **kwargs):
    ours = mojo_evaluate.load(name, config).compute(
        predictions=predictions, references=references, **kwargs
    )
    theirs = upstream(name, config).compute(
        predictions=predictions, references=references, **kwargs
    )
    assert_result_equal(ours, theirs)


def smape_reference(
    predictions,
    references,
    *,
    sample_weight=None,
    multioutput="uniform_average",
):
    predictions = np.asarray(predictions, dtype=np.float64)
    references = np.asarray(references, dtype=np.float64)
    if predictions.ndim == 1:
        predictions = predictions[:, None]
        references = references[:, None]
    epsilon = np.finfo(np.float64).eps
    errors = 2 * np.abs(predictions - references) / (
        np.maximum(np.abs(references), epsilon)
        + np.maximum(np.abs(predictions), epsilon)
    )
    output_errors = np.average(errors, weights=sample_weight, axis=0)
    if isinstance(multioutput, str):
        if multioutput == "raw_values":
            return output_errors
        return float(np.average(output_errors))
    return float(np.average(output_errors, weights=multioutput))


@pytest.mark.parametrize("normalize", [True, False])
def test_accuracy(normalize):
    compare(
        "accuracy",
        [0, 2, 1, 3, 1, 2],
        [0, 1, 1, 3, 2, 2],
        normalize=normalize,
    )


def test_accuracy_weighted():
    compare(
        "accuracy",
        [0, 0, 2, 1, 1],
        [0, 1, 2, 2, 1],
        sample_weight=[0.5, 2.0, 1.5, 0.25, 3.0],
    )


def test_accuracy_multilabel():
    compare(
        "accuracy",
        [[1, 0, 1], [0, 1, 1], [1, 1, 0], [0, 0, 0]],
        [[1, 0, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]],
        config="multilabel",
        sample_weight=[1.0, 2.0, 0.5, 4.0],
    )


def test_accuracy_multilabel_count():
    compare(
        "accuracy",
        [[1, 0], [0, 1], [1, 1]],
        [[1, 0], [1, 0], [1, 1]],
        config="multilabel",
        normalize=False,
    )


@pytest.mark.parametrize("name", ["precision", "recall", "f1"])
def test_binary_scores(name):
    compare(
        name,
        [0, 1, 1, 1, 0, 0, 1],
        [0, 1, 0, 1, 1, 0, 1],
        sample_weight=[1.0, 0.5, 2.0, 1.0, 3.0, 0.25, 1.5],
    )


@pytest.mark.parametrize("name", ["precision", "recall"])
@pytest.mark.parametrize("zero_division", [0, 1, np.nan])
def test_zero_division(name, zero_division):
    compare(
        name,
        [0, 0, 0],
        [0, 0, 0],
        zero_division=zero_division,
    )


@pytest.mark.parametrize("name", ["precision", "recall", "f1"])
@pytest.mark.parametrize("average", ["micro", "macro", "weighted", None])
def test_multiclass_scores(name, average):
    compare(
        name,
        [0, 2, 1, 2, 0, 3, 3, 1, 0],
        [0, 1, 1, 2, 3, 3, 2, 1, 0],
        average=average,
        labels=[3, 1, 0],
        sample_weight=[1.0, 2.0, 0.5, 1.5, 1.0, 0.25, 3.0, 1.0, 2.0],
    )


@pytest.mark.parametrize("name", ["precision", "recall", "f1"])
@pytest.mark.parametrize("average", ["micro", "macro", "weighted", "samples", None])
def test_multilabel_scores(name, average):
    predictions = [
        [1, 0, 1, 0],
        [0, 1, 1, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 1],
    ]
    references = [
        [1, 0, 0, 0],
        [0, 1, 1, 0],
        [1, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 1, 0, 1],
    ]
    compare(
        name,
        predictions,
        references,
        config="multilabel",
        average=average,
        labels=[3, 1, 0],
        sample_weight=[1.0, 2.0, 0.5, 3.0, 1.5],
    )


@pytest.mark.parametrize("normalize", [None, "true", "pred", "all"])
def test_confusion_matrix(normalize):
    compare(
        "confusion_matrix",
        [0, 2, 1, 2, 0, 3, 3, 1, 0],
        [0, 1, 1, 2, 3, 3, 2, 1, 0],
        labels=[3, 1, 0],
        sample_weight=[1.0, 2.0, 0.5, 1.5, 1.0, 0.25, 3.0, 1.0, 2.0],
        normalize=normalize,
    )


@pytest.mark.parametrize("weighted", [False, True])
def test_matthews_correlation_multiclass(weighted):
    kwargs = {}
    if weighted:
        kwargs["sample_weight"] = [1.0, 2.0, 0.5, 1.5, 3.0, 0.25, 2.0]
    compare(
        "matthews_correlation",
        [0, 2, 1, 2, 0, 3, 3],
        [0, 1, 1, 2, 3, 3, 2],
        **kwargs,
    )


@pytest.mark.parametrize("average", [None, "macro"])
def test_matthews_correlation_multilabel(average):
    compare(
        "matthews_correlation",
        [[1, 0, 1], [0, 1, 1], [1, 1, 0], [0, 0, 0]],
        [[1, 0, 0], [0, 1, 1], [1, 0, 0], [1, 0, 0]],
        config="multilabel",
        average=average,
        sample_weight=[1.0, 2.0, 0.5, 3.0],
    )


@pytest.mark.parametrize("name", ["mse", "mae"])
def test_regression_scalar(name):
    compare(
        name,
        [1.25, -2.0, 4.5, 0.0, 3.25],
        [1.0, -1.5, 2.0, 0.5, 3.0],
        sample_weight=[1.0, 0.25, 2.0, 3.0, 0.5],
    )


@pytest.mark.parametrize("name", ["mse", "mae"])
@pytest.mark.parametrize(
    "multioutput",
    ["raw_values", "uniform_average", np.array([0.25, 0.75])],
)
def test_regression_multioutput(name, multioutput):
    compare(
        name,
        [[1.25, 4.0], [-2.0, 1.5], [4.5, -1.0], [0.0, 2.0]],
        [[1.0, 3.0], [-1.5, 2.0], [2.0, -2.0], [0.5, 2.5]],
        config="multilist",
        sample_weight=[1.0, 0.25, 2.0, 3.0],
        multioutput=multioutput,
    )


def test_root_mean_squared_error():
    compare(
        "mse",
        [[1.0, 4.0], [2.0, 5.0], [6.0, -1.0]],
        [[0.0, 2.0], [3.0, 5.0], [4.0, 1.0]],
        config="multilist",
        squared=False,
        multioutput="raw_values",
    )


def test_smape_scalar_reference():
    predictions = [1.25, -2.0, 4.5, 0.0, 3.25]
    references = [1.0, -1.5, 2.0, 0.5, 3.0]
    sample_weight = [1.0, 0.25, 2.0, 3.0, 0.5]
    ours = mojo_evaluate.load("smape").compute(
        predictions=predictions,
        references=references,
        sample_weight=sample_weight,
    )
    expected = smape_reference(
        predictions, references, sample_weight=sample_weight
    )
    assert ours["smape"] == pytest.approx(expected)


@pytest.mark.parametrize(
    "multioutput",
    ["raw_values", "uniform_average", np.array([0.25, 0.75])],
)
def test_smape_multioutput_reference(multioutput):
    predictions = [[1.25, 4.0], [-2.0, 1.5], [4.5, -1.0], [0.0, 2.0]]
    references = [[1.0, 3.0], [-1.5, 2.0], [2.0, -2.0], [0.5, 2.5]]
    sample_weight = [1.0, 0.25, 2.0, 3.0]
    ours = mojo_evaluate.load("smape", "multilist").compute(
        predictions=predictions,
        references=references,
        sample_weight=sample_weight,
        multioutput=multioutput,
    )
    expected = smape_reference(
        predictions,
        references,
        sample_weight=sample_weight,
        multioutput=multioutput,
    )
    assert np.allclose(ours["smape"], expected)


def test_brier_score():
    compare(
        "brier_score",
        [0.05, 0.8, 0.35, 0.9, 0.1],
        [0, 1, 1, 1, 0],
        sample_weight=[1.0, 2.0, 0.5, 3.0, 0.25],
        pos_label=1,
    )


def test_brier_score_alternate_positive_label():
    compare(
        "brier_score",
        [0.05, 0.8, 0.35, 0.9, 0.1],
        [-1, 2, -1, 2, -1],
        pos_label=2,
    )


def test_mase_scalar():
    compare(
        "mase",
        [11.0, 12.5, 10.0, 14.0],
        [10.0, 12.0, 11.0, 13.5],
        training=[5.0, 7.0, 8.0, 11.0, 10.0, 13.0],
        periodicity=2,
        sample_weight=[1.0, 2.0, 0.5, 3.0],
    )


@pytest.mark.parametrize(
    "multioutput",
    ["raw_values", "uniform_average", np.array([0.25, 0.75])],
)
def test_mase_multioutput(multioutput):
    compare(
        "mase",
        [[11.0, 2.0], [12.5, 3.0], [10.0, 2.5]],
        [[10.0, 2.5], [12.0, 2.0], [11.0, 3.0]],
        config="multilist",
        training=[
            [5.0, 1.0],
            [7.0, 2.0],
            [8.0, 1.5],
            [11.0, 3.0],
            [10.0, 2.0],
        ],
        periodicity=2,
        multioutput=multioutput,
    )


def test_streaming_add_and_add_batch():
    metric = mojo_evaluate.load("accuracy")
    metric.add(prediction=0, reference=0)
    metric.add_batch(predictions=[1, 0, 2], references=[0, 0, 2])
    assert metric.compute() == {"accuracy": 0.75}


def test_streaming_multilist():
    metric = mojo_evaluate.load("mae", "multilist")
    metric.add(prediction=[1.0, 2.0], reference=[0.0, 2.0])
    metric.add(prediction=[3.0, 4.0], reference=[2.0, 6.0])
    result = metric.compute(multioutput="raw_values")
    assert np.allclose(result["mae"], [1.0, 1.0])


def test_combine():
    metric = mojo_evaluate.combine(["accuracy", "f1"])
    result = metric.compute(
        predictions=[0, 1, 1, 0], references=[0, 1, 0, 0]
    )
    assert result == pytest.approx({"accuracy": 0.75, "f1": 2 / 3})


def test_loader_rejects_uncovered_metric():
    with pytest.raises(FileNotFoundError, match="not covered"):
        mojo_evaluate.load("bleu")


@pytest.mark.parametrize(
    "values",
    [
        [0.0, 1.5],
        [0, np.iinfo(np.uint64).max],
        [0, np.nan],
        [0, np.inf],
        [0, 1 + 0j],
    ],
)
def test_classification_rejects_lossy_label_conversion(values):
    with pytest.raises(ValueError, match="integer|int64"):
        mojo_evaluate.load("accuracy").compute(
            predictions=values, references=values
        )


def test_classification_accepts_exact_integer_floats_and_strided_inputs():
    predictions = np.arange(12, dtype=np.float64)[::2]
    references = predictions.copy()
    assert mojo_evaluate.load("accuracy").compute(
        predictions=predictions, references=references
    ) == {"accuracy": 1.0}


def test_regression_rejects_complex_values():
    with pytest.raises(ValueError, match="real-valued"):
        mojo_evaluate.load("mae").compute(
            predictions=[1 + 2j], references=[1 + 0j]
        )


def test_zero_sum_weights_are_rejected():
    with pytest.raises(ValueError, match="non-zero sum"):
        mojo_evaluate.load("accuracy").compute(
            predictions=[0, 1],
            references=[0, 1],
            sample_weight=[0.0, 0.0],
        )


def test_empty_labels_are_rejected():
    with pytest.raises(ValueError, match="at least one"):
        mojo_evaluate.load("confusion_matrix").compute(
            predictions=[0], references=[0], labels=[]
        )
