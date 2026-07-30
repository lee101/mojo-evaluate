from std.memory import UnsafePointer


comptime I64Ptr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime F64Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]


def i64p(addr: Int) -> I64Ptr:
    return I64Ptr(unsafe_from_address=addr)


def f64p(addr: Int) -> F64Ptr:
    return F64Ptr(unsafe_from_address=addr)


@export("me_accuracy")
def me_accuracy(
    references_addr: Int,
    predictions_addr: Int,
    weights_addr: Int,
    n: Int,
    weighted: Int,
) abi("C") -> Float64:
    var references = i64p(references_addr)
    var predictions = i64p(predictions_addr)
    var correct = 0.0
    if weighted != 0:
        var weights = f64p(weights_addr)
        for i in range(n):
            if references[i] == predictions[i]:
                correct += weights[i]
    else:
        for i in range(n):
            if references[i] == predictions[i]:
                correct += 1.0
    return correct


@export("me_subset_accuracy")
def me_subset_accuracy(
    references_addr: Int,
    predictions_addr: Int,
    weights_addr: Int,
    rows: Int,
    labels: Int,
    weighted: Int,
) abi("C") -> Float64:
    var references = i64p(references_addr)
    var predictions = i64p(predictions_addr)
    var correct = 0.0
    if weighted != 0:
        var weights = f64p(weights_addr)
        for i in range(rows):
            var same = True
            for j in range(labels):
                if references[i * labels + j] != predictions[i * labels + j]:
                    same = False
            if same:
                correct += weights[i]
    else:
        for i in range(rows):
            var same = True
            for j in range(labels):
                if references[i * labels + j] != predictions[i * labels + j]:
                    same = False
            if same:
                correct += 1.0
    return correct


@export("me_confusion_matrix")
def me_confusion_matrix(
    references_addr: Int,
    predictions_addr: Int,
    weights_addr: Int,
    matrix_addr: Int,
    n: Int,
    classes: Int,
    weighted: Int,
) abi("C"):
    var references = i64p(references_addr)
    var predictions = i64p(predictions_addr)
    var matrix = f64p(matrix_addr)
    for i in range(classes * classes):
        matrix[i] = 0.0
    if weighted != 0:
        var weights = f64p(weights_addr)
        for i in range(n):
            matrix[Int(references[i]) * classes + Int(predictions[i])] += weights[i]
    else:
        for i in range(n):
            matrix[Int(references[i]) * classes + Int(predictions[i])] += 1.0


@export("me_multilabel_stats")
def me_multilabel_stats(
    references_addr: Int,
    predictions_addr: Int,
    weights_addr: Int,
    stats_addr: Int,
    rows: Int,
    labels: Int,
    weighted: Int,
) abi("C"):
    var references = i64p(references_addr)
    var predictions = i64p(predictions_addr)
    var stats = f64p(stats_addr)
    for i in range(labels * 4):
        stats[i] = 0.0
    if weighted != 0:
        var weights = f64p(weights_addr)
        for i in range(rows):
            for j in range(labels):
                var reference = references[i * labels + j]
                var prediction = predictions[i * labels + j]
                if reference == 1 and prediction == 1:
                    stats[j * 4] += weights[i]
                elif reference == 0 and prediction == 1:
                    stats[j * 4 + 1] += weights[i]
                elif reference == 1 and prediction == 0:
                    stats[j * 4 + 2] += weights[i]
                else:
                    stats[j * 4 + 3] += weights[i]
    else:
        for i in range(rows):
            for j in range(labels):
                var reference = references[i * labels + j]
                var prediction = predictions[i * labels + j]
                if reference == 1 and prediction == 1:
                    stats[j * 4] += 1.0
                elif reference == 0 and prediction == 1:
                    stats[j * 4 + 1] += 1.0
                elif reference == 1 and prediction == 0:
                    stats[j * 4 + 2] += 1.0
                else:
                    stats[j * 4 + 3] += 1.0


@export("me_multilabel_samples")
def me_multilabel_samples(
    references_addr: Int,
    predictions_addr: Int,
    weights_addr: Int,
    scores_addr: Int,
    rows: Int,
    labels: Int,
    weighted: Int,
) abi("C"):
    var references = i64p(references_addr)
    var predictions = i64p(predictions_addr)
    var scores = f64p(scores_addr)
    for i in range(rows * 3):
        scores[i] = 0.0
    if weighted != 0:
        var weights = f64p(weights_addr)
        for i in range(rows):
            for j in range(labels):
                var reference = references[i * labels + j]
                var prediction = predictions[i * labels + j]
                if reference == 1 and prediction == 1:
                    scores[i * 3] += weights[i]
                elif reference == 0 and prediction == 1:
                    scores[i * 3 + 1] += weights[i]
                elif reference == 1 and prediction == 0:
                    scores[i * 3 + 2] += weights[i]
    else:
        for i in range(rows):
            for j in range(labels):
                var reference = references[i * labels + j]
                var prediction = predictions[i * labels + j]
                if reference == 1 and prediction == 1:
                    scores[i * 3] += 1.0
                elif reference == 0 and prediction == 1:
                    scores[i * 3 + 1] += 1.0
                elif reference == 1 and prediction == 0:
                    scores[i * 3 + 2] += 1.0


@export("me_regression_reductions")
def me_regression_reductions(
    references_addr: Int,
    predictions_addr: Int,
    weights_addr: Int,
    reductions_addr: Int,
    rows: Int,
    outputs: Int,
    weighted: Int,
) abi("C"):
    var references = f64p(references_addr)
    var predictions = f64p(predictions_addr)
    var reductions = f64p(reductions_addr)
    for i in range(outputs * 5):
        reductions[i] = 0.0
    if weighted != 0:
        var weights = f64p(weights_addr)
        for i in range(rows):
            for j in range(outputs):
                var idx = i * outputs + j
                var delta = predictions[idx] - references[idx]
                var weight = weights[i]
                reductions[j * 5] += delta * delta * weight
                reductions[j * 5 + 1] += abs(delta) * weight
                reductions[j * 5 + 2] += references[idx] * weight
                reductions[j * 5 + 3] += references[idx] * references[idx] * weight
                var denom = max(abs(references[idx]), 2.220446049250313e-16)
                denom += max(abs(predictions[idx]), 2.220446049250313e-16)
                reductions[j * 5 + 4] += 2.0 * abs(delta) / denom * weight
    else:
        for i in range(rows):
            for j in range(outputs):
                var idx = i * outputs + j
                var delta = predictions[idx] - references[idx]
                reductions[j * 5] += delta * delta
                reductions[j * 5 + 1] += abs(delta)
                reductions[j * 5 + 2] += references[idx]
                reductions[j * 5 + 3] += references[idx] * references[idx]
                var denom = max(abs(references[idx]), 2.220446049250313e-16)
                denom += max(abs(predictions[idx]), 2.220446049250313e-16)
                reductions[j * 5 + 4] += 2.0 * abs(delta) / denom


@export("me_naive_mae")
def me_naive_mae(
    training_addr: Int,
    reductions_addr: Int,
    rows: Int,
    outputs: Int,
    periodicity: Int,
) abi("C"):
    var training = f64p(training_addr)
    var reductions = f64p(reductions_addr)
    for j in range(outputs):
        reductions[j] = 0.0
    for i in range(periodicity, rows):
        for j in range(outputs):
            reductions[j] += abs(
                training[i * outputs + j]
                - training[(i - periodicity) * outputs + j]
            )
