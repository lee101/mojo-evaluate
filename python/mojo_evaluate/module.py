from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .metrics import (
    SUPPORTED_METRICS,
    _CLASSIFICATION,
    _REGRESSION,
    compute_brier,
    compute_classification,
    compute_regression,
)


class Metric:
    def __init__(self, name: str, config_name: str | None = None, **kwargs):
        self.name = name
        self.config_name = config_name
        self._predictions: list[np.ndarray] = []
        self._references: list[np.ndarray] = []

    def __repr__(self) -> str:
        config = f", config_name={self.config_name!r}" if self.config_name else ""
        return f"Metric(name={self.name!r}{config})"

    def add(self, *, prediction=None, reference=None, **kwargs):
        if prediction is None or reference is None:
            raise ValueError("prediction and reference are required")
        vector_row = self.config_name in {"multilabel", "multilist"}
        prediction = np.asarray(prediction)
        reference = np.asarray(reference)
        if vector_row:
            prediction = np.atleast_2d(prediction)
            reference = np.atleast_2d(reference)
        else:
            prediction = np.atleast_1d(prediction)
            reference = np.atleast_1d(reference)
        self._predictions.append(prediction)
        self._references.append(reference)

    def add_batch(self, *, predictions=None, references=None, **kwargs):
        if predictions is None or references is None:
            raise ValueError("predictions and references are required")
        self._predictions.append(np.asarray(predictions))
        self._references.append(np.asarray(references))

    def _stored(self):
        if not self._predictions:
            raise ValueError(
                "No examples were provided. Pass predictions and references "
                "or call add/add_batch first."
            )
        predictions = np.concatenate(
            [np.atleast_1d(value) for value in self._predictions], axis=0
        )
        references = np.concatenate(
            [np.atleast_1d(value) for value in self._references], axis=0
        )
        self._predictions.clear()
        self._references.clear()
        return predictions, references

    def compute(self, *, predictions=None, references=None, **kwargs):
        if (predictions is None) != (references is None):
            raise ValueError("predictions and references must be provided together")
        if predictions is None:
            predictions, references = self._stored()
        if self.name in _CLASSIFICATION:
            if self.name == "accuracy":
                kwargs.setdefault("normalize", True)
            elif self.name == "confusion_matrix":
                kwargs.setdefault("normalize", None)
            elif self.name == "matthews_correlation":
                kwargs.setdefault("average", None)
            return compute_classification(
                self.name,
                predictions,
                references,
                config_name=self.config_name,
                **kwargs,
            )
        if self.name in _REGRESSION:
            return compute_regression(
                self.name, predictions, references, **kwargs
            )
        return compute_brier(predictions, references, **kwargs)


class CombinedEvaluations:
    def __init__(self, evaluation_modules, force_prefix: bool = False):
        self.evaluation_modules = evaluation_modules
        self.force_prefix = force_prefix

    def add(self, *, prediction=None, reference=None, **kwargs):
        for metric in self.evaluation_modules.values():
            metric.add(prediction=prediction, reference=reference, **kwargs)

    def add_batch(self, *, predictions=None, references=None, **kwargs):
        for metric in self.evaluation_modules.values():
            metric.add_batch(
                predictions=predictions, references=references, **kwargs
            )

    def compute(self, *, predictions=None, references=None, **kwargs):
        result = {}
        for prefix, metric in self.evaluation_modules.items():
            values = metric.compute(
                predictions=predictions, references=references, **kwargs
            )
            for key, value in values.items():
                output_key = (
                    f"{prefix}_{key}"
                    if self.force_prefix or key in result
                    else key
                )
                result[output_key] = value
        return result


def load(
    path: str,
    config_name: str | None = None,
    module_type: str | None = None,
    process_id: int = 0,
    num_process: int = 1,
    cache_dir: str | None = None,
    experiment_id: str | None = None,
    keep_in_memory: bool = False,
    download_config=None,
    download_mode=None,
    revision=None,
    **init_kwargs,
) -> Metric:
    name = path.rstrip("/").rsplit("/", 1)[-1]
    if name not in SUPPORTED_METRICS:
        supported = ", ".join(SUPPORTED_METRICS)
        raise FileNotFoundError(
            f"metric {path!r} is not covered by mojo-evaluate; "
            f"supported metrics: {supported}"
        )
    if module_type not in (None, "metric"):
        raise ValueError("mojo-evaluate only provides metric modules")
    if process_id != 0 or num_process != 1:
        raise NotImplementedError("distributed metric accumulation is not supported")
    valid_configs = {None, "default"}
    if name in _CLASSIFICATION:
        valid_configs.add("multilabel")
    if name in _REGRESSION:
        valid_configs.add("multilist")
    if config_name not in valid_configs:
        raise ValueError(
            f"config {config_name!r} is not supported for metric {name!r}"
        )
    normalized_config = None if config_name == "default" else config_name
    return Metric(name, normalized_config, **init_kwargs)


def combine(evaluations, force_prefix: bool = False) -> CombinedEvaluations:
    if isinstance(evaluations, Mapping):
        modules = {
            name: load(value) if isinstance(value, str) else value
            for name, value in evaluations.items()
        }
    else:
        modules = {}
        occurrences = {}
        for value in evaluations:
            metric = load(value) if isinstance(value, str) else value
            name = metric.name
            count = occurrences.get(name, 0)
            occurrences[name] = count + 1
            key = name if count == 0 else f"{name}_{count}"
            modules[key] = metric
    return CombinedEvaluations(modules, force_prefix=force_prefix)
