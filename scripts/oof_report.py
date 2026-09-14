"""Pool per-fold runs into one out-of-fold report with cross-fitted calibration.

Each run must come from the same dataset, fold count, and seed, and must validate
on a different fold, so the pooled predictions cover every patient exactly once.
Calibration is fitted leave-one-fold-out: the transform applied to a fold never
saw that fold, so the reported calibrated pF1 is not a model-selection score.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.training_common import validation_metrics


COMPARABLE = ('csv_sha256', 'index_sha256')
COMPARABLE_ARGUMENTS = ('folds', 'seed', 'image_size', 'encoder', 'augment', 'select_metric')


def read_predictions(path: Path) -> dict[str, float]:
    with path.open(newline='', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        if (reader.fieldnames or []) != ['prediction_id', 'cancer']:
            raise ValueError(f'Expected prediction_id,cancer columns in {path}.')
        values = {}
        for row in reader:
            if row['prediction_id'] in values:
                raise ValueError(f'Duplicate prediction_id in {path}.')
            values[row['prediction_id']] = float(row['cancer'])
    if not values:
        raise ValueError(f'No rows in {path}.')
    return values


def load_runs(directories):
    runs = []
    for directory in directories:
        directory = directory.resolve()
        if not (directory / '_TRAINING_SUCCESS').is_file():
            raise ValueError(f'Unfinished run without _TRAINING_SUCCESS: {directory}')
        config = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
        predictions = read_predictions(directory / 'best_predictions.csv')
        labels = read_predictions(directory / 'validation_labels.csv')
        if predictions.keys() != labels.keys():
            raise ValueError(f'Best predictions do not cover the validation breasts: {directory}')
        if any(value not in (0.0, 1.0) for value in labels.values()):
            raise ValueError(f'Validation labels must be binary: {directory}')
        if any(not 0.0 <= value <= 1.0 for value in predictions.values()):
            raise ValueError(f'Predictions must be probabilities in [0, 1]: {directory}')
        runs.append({'run': str(directory), 'config': config, 'fold': config['arguments']['fold'],
                     'labels': labels, 'predictions': predictions})
    reference = runs[0]
    for run in runs[1:]:
        for key in COMPARABLE:
            if run['config']['dataset'][key] != reference['config']['dataset'][key]:
                raise ValueError(f'Runs disagree on dataset {key}; they are not poolable.')
        for key in COMPARABLE_ARGUMENTS:
            if run['config']['arguments'].get(key) != reference['config']['arguments'].get(key):
                raise ValueError(f'Runs disagree on --{key.replace("_", "-")}; they are not poolable.')
    folds = [run['fold'] for run in runs]
    if len(set(folds)) != len(folds):
        raise ValueError(f'Runs must validate on distinct folds, got {sorted(folds)}.')
    seen = set()
    for run in runs:
        overlap = seen & run['predictions'].keys()
        if overlap:
            raise ValueError(f'Breast {sorted(overlap)[0]} appears in two folds; check folds.csv.')
        seen |= run['predictions'].keys()
    return sorted(runs, key=lambda run: run['fold'])


def logit(p, epsilon=1e-6):
    p = np.clip(p, epsilon, 1.0 - epsilon)
    return np.log(p / (1.0 - p))


def fit_platt(scores, labels, iterations=100, tolerance=1e-10):
    """Two-parameter logistic recalibration by iteratively reweighted least squares."""
    design = np.column_stack([scores, np.ones_like(scores)])
    weights = np.zeros(2)
    for _ in range(iterations):
        probabilities = 1.0 / (1.0 + np.exp(-design @ weights))
        variance = np.clip(probabilities * (1.0 - probabilities), 1e-9, None)
        gradient = design.T @ (probabilities - labels)
        hessian = design.T @ (design * variance[:, None]) + 1e-6 * np.eye(2)
        step = np.linalg.solve(hessian, gradient)
        weights = weights - step
        if np.abs(step).max() < tolerance:
            break
    return weights


def apply_platt(scores, weights):
    return 1.0 / (1.0 + np.exp(-(weights[0] * scores + weights[1])))


def best_binary_fraction(labels, probabilities):
    """Fraction of breasts to call positive that maximises binary F1 on this data."""
    order = np.argsort(-probabilities)
    positives = labels.sum()
    true_positive = np.cumsum(labels[order])
    counts = np.arange(1, len(labels) + 1)
    f1 = 2 * true_positive / (counts + positives)
    index = int(np.argmax(f1))
    return (index + 1) / len(labels), float(f1[index])


def score(labels, probabilities):
    ids = [str(index) for index in range(len(labels))]
    metrics, _ = validation_metrics(ids, labels.tolist(), probabilities.tolist())
    return {key: value for key, value in metrics.items()
            if key in ('pf1', 'roc_auc', 'average_precision', 'brier', 'f1_at_0_5',
                       'precision_at_0_5', 'recall_at_0_5', 'breasts', 'positive_breasts')}


def bootstrap_interval(labels, probabilities, statistic, resamples, seed):
    generator = np.random.default_rng(seed)
    values = []
    for _ in range(resamples):
        index = generator.integers(0, len(labels), len(labels))
        if labels[index].sum() < 2:
            continue
        values.append(statistic(labels[index], probabilities[index]))
    if not values:
        return None
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def roc_auc(labels, probabilities):
    order = np.argsort(probabilities, kind='mergesort')
    ranks = np.empty(len(probabilities), dtype=np.float64)
    ranks[order] = np.arange(1, len(probabilities) + 1)
    sorted_scores = probabilities[order]
    start = 0
    while start < len(sorted_scores):
        stop = start
        while stop + 1 < len(sorted_scores) and sorted_scores[stop + 1] == sorted_scores[start]:
            stop += 1
        if stop > start:
            ranks[order[start:stop + 1]] = (start + stop + 2) / 2
        start = stop + 1
    positives = labels.sum()
    negatives = len(labels) - positives
    if not positives or not negatives:
        return float('nan')
    return float((ranks[labels == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('runs', type=Path, nargs='+', help='Per-fold run directories to pool.')
    parser.add_argument('--out', type=Path, help='Optional JSON report destination (never overwritten).')
    parser.add_argument('--predictions-out', type=Path,
                        help='Optional pooled OOF prediction CSV, raw and calibrated.')
    parser.add_argument('--bootstrap', type=int, default=2000, help='Bootstrap resamples; 0 disables.')
    parser.add_argument('--seed', type=int, default=0)
    return parser


def main():
    args = build_parser().parse_args()
    for path in (args.out, args.predictions_out):
        if path is not None and path.exists():
            raise SystemExit(f'Refusing to overwrite an existing file: {path}')
    runs = load_runs(args.runs)
    ids, labels, raw, fold_of = [], [], [], []
    for run in runs:
        for key in sorted(run['labels']):
            ids.append(key)
            labels.append(run['labels'][key])
            raw.append(run['predictions'][key])
            fold_of.append(run['fold'])
    labels = np.array(labels, dtype=np.float64)
    raw = np.array(raw, dtype=np.float64)
    fold_of = np.array(fold_of)

    calibrated = np.empty_like(raw)
    binary = np.zeros_like(raw)
    transforms = {}
    for fold in sorted(set(fold_of.tolist())):
        held = fold_of == fold
        others = ~held
        if labels[others].sum() < 2:
            raise ValueError(f'Fold {fold} has too few out-of-fold positives to calibrate against.')
        weights = fit_platt(logit(raw[others]), labels[others])
        calibrated[held] = apply_platt(logit(raw[held]), weights)
        fraction, _ = best_binary_fraction(labels[others], raw[others])
        count = max(1, round(fraction * held.sum()))
        cutoff = np.sort(raw[held])[::-1][count - 1]
        binary[held] = (raw[held] >= cutoff).astype(np.float64)
        transforms[str(fold)] = {'platt_slope': float(weights[0]), 'platt_intercept': float(weights[1]),
                                 'positive_fraction': float(fraction), 'threshold': float(cutoff)}

    report = {
        'runs': [{'run': run['run'], 'fold': run['fold'],
                  'breasts': len(run['labels']), 'positive_breasts': int(sum(run['labels'].values())),
                  'fold_pf1': score(np.array([run['labels'][k] for k in sorted(run['labels'])]),
                                    np.array([run['predictions'][k] for k in sorted(run['labels'])]))}
                 for run in runs],
        'folds_covered': sorted(set(fold_of.tolist())),
        'config': {key: runs[0]['config']['arguments'].get(key) for key in COMPARABLE_ARGUMENTS},
        'pooled_raw': score(labels, raw),
        'pooled_calibrated_crossfitted': score(labels, calibrated),
        'pooled_binary_crossfitted': score(labels, binary),
        'calibration_transforms': transforms,
    }
    if args.bootstrap:
        report['bootstrap_95ci'] = {
            'roc_auc': bootstrap_interval(labels, raw, roc_auc, args.bootstrap, args.seed),
            'pf1_raw': bootstrap_interval(labels, raw,
                                          lambda y, p: 2 * (y * p).sum() / (y.sum() + p.sum()),
                                          args.bootstrap, args.seed),
            'pf1_calibrated': bootstrap_interval(labels, calibrated,
                                                 lambda y, p: 2 * (y * p).sum() / (y.sum() + p.sum()),
                                                 args.bootstrap, args.seed),
        }
    if len(report['folds_covered']) != runs[0]['config']['arguments']['folds']:
        report['warning'] = ('Partial coverage: pooled scores describe the supplied folds only, '
                             'not the whole dataset.')
    if args.predictions_out is not None:
        with args.predictions_out.open('x', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['prediction_id', 'fold', 'cancer_label', 'cancer', 'cancer_calibrated'])
            writer.writerows(zip(ids, fold_of.tolist(), labels.astype(int).tolist(),
                                 raw.tolist(), calibrated.tolist()))
    if args.out is not None:
        with args.out.open('x', encoding='utf-8') as file:
            json.dump(report, file, indent=2, allow_nan=False)
            file.write('\n')
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
