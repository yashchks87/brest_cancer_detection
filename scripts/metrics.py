import argparse
import csv
import json
import math
from collections.abc import Iterable
from pathlib import Path


def _validate_inputs(
    labels: Iterable[float], predictions: Iterable[float]
) -> tuple[list[float], list[float]]:
    labels = [float(value) for value in labels]
    predictions = [float(value) for value in predictions]
    if not labels or len(labels) != len(predictions):
        raise ValueError('Labels and predictions must be nonempty and have equal lengths.')
    if any(value not in (0.0, 1.0) for value in labels):
        raise ValueError('Labels must be binary: 0 or 1.')
    if not all(math.isfinite(value) for value in predictions):
        raise ValueError('Predictions must be finite; NaN and infinity are not allowed.')
    return labels, predictions


def pfbeta(
    labels: Iterable[float], predictions: Iterable[float], beta: float = 1.0
) -> float:
    labels, predictions = _validate_inputs(labels, predictions)
    beta = float(beta)
    beta_squared = beta * beta
    if beta <= 0 or not math.isfinite(beta_squared) or beta_squared == 0:
        raise ValueError('Beta must be positive with a finite, nonzero square.')
    predictions = [min(max(value, 0.0), 1.0) for value in predictions]
    positive_count = sum(labels)
    true_positive = math.fsum(p for y, p in zip(labels, predictions) if y == 1)
    predicted_positive = math.fsum(predictions)
    if true_positive == 0 or positive_count == 0 or predicted_positive == 0:
        return 0.0
    precision = true_positive / predicted_positive
    recall = true_positive / positive_count
    return (1.0 + beta_squared) * (precision * recall) / (beta_squared * precision + recall)


def pf1(labels: Iterable[float], predictions: Iterable[float]) -> float:
    return pfbeta(labels, predictions, beta=1.0)


def _validate_id(prediction_id: str) -> None:
    if not isinstance(prediction_id, str) or not prediction_id.strip():
        raise ValueError('Each prediction_id must be a nonempty string.')


def aggregate_predictions(
    prediction_ids: Iterable[str],
    labels: Iterable[float],
    predictions: Iterable[float],
    reduction: str = 'mean',
) -> tuple[list[str], list[int], list[float]]:
    labels, predictions = _validate_inputs(labels, predictions)
    prediction_ids = list(prediction_ids)
    if len(prediction_ids) != len(labels):
        raise ValueError('Prediction IDs, labels and predictions must have equal lengths.')
    if reduction not in ('mean', 'max'):
        raise ValueError("Reduction must be 'mean' or 'max'.")
    if any(value < 0 or value > 1 for value in predictions):
        raise ValueError('Image predictions must be probabilities in [0, 1], not logits.')
    grouped: dict[str, tuple[int, list[float]]] = {}
    for prediction_id, label, prediction in zip(prediction_ids, labels, predictions):
        _validate_id(prediction_id)
        if prediction_id not in grouped:
            grouped[prediction_id] = (int(label), [])
        group_label, group_predictions = grouped[prediction_id]
        if group_label != label:
            raise ValueError('Images sharing a prediction_id must have the same cancer label.')
        group_predictions.append(prediction)
    return (
        list(grouped),
        [label for label, _ in grouped.values()],
        [math.fsum(values) / len(values) if reduction == 'mean' else max(values)
         for _, values in grouped.values()],
    )


def _read_csv(path: str | Path) -> dict[str, float]:
    with Path(path).open(newline='', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)):
            raise ValueError('CSV column names must be unique.')
        if not {'prediction_id', 'cancer'}.issubset(fields):
            raise ValueError('Each CSV must contain prediction_id and cancer columns.')
        values = {}
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError('CSV rows must have the same number of fields as the header.')
            prediction_id = row['prediction_id']
            _validate_id(prediction_id)
            if prediction_id in values:
                raise ValueError('Each CSV must have exactly one row per prediction_id.')
            values[prediction_id] = float(row['cancer'])
    if not values:
        raise ValueError('CSV files must contain at least one data row.')
    return values


def score_submission(solution_path: str | Path, submission_path: str | Path) -> float:
    solution = _read_csv(solution_path)
    submission = _read_csv(submission_path)
    if solution.keys() != submission.keys():
        missing = len(solution.keys() - submission.keys())
        extra = len(submission.keys() - solution.keys())
        raise ValueError(f'Submission IDs do not match solution: {missing} missing, {extra} extra.')
    labels, predictions = _validate_inputs(
        solution.values(), (submission[prediction_id] for prediction_id in solution)
    )
    if any(value < 0 or value > 1 for value in predictions):
        raise ValueError('Submission cancer values must be probabilities in [0, 1].')
    return pf1(labels, predictions)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Score breast-level RSNA predictions using probabilistic F1 (beta=1).'
    )
    parser.add_argument('--solution', required=True, type=Path,
                        help='Breast-level validation CSV with prediction_id,cancer (binary truth).')
    parser.add_argument('--submission', required=True, type=Path,
                        help='Prediction CSV with prediction_id,cancer (probability).')
    args = parser.parse_args()
    try:
        score = score_submission(args.solution, args.submission)
    except (OSError, ValueError, csv.Error) as error:
        parser.error(str(error))
    print(json.dumps({'pf1': score}, allow_nan=False))


if __name__ == '__main__':
    main()
