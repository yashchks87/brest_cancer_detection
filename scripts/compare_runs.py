import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.metrics import pf1


COMPARABLE = ('csv_sha256', 'index_sha256')
COMPARABLE_ARGUMENTS = ('folds', 'fold', 'seed')


def load_run(directory: Path) -> dict:
    directory = directory.resolve()
    if not (directory / '_TRAINING_SUCCESS').is_file():
        raise ValueError(f'Unfinished run without _TRAINING_SUCCESS: {directory}')
    config = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
    epochs = [json.loads(line) for line in
              (directory / 'metrics.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    if not epochs or config.get('approach') not in ('simple', 'advanced', 'vit'):
        raise ValueError(f'Run has no epoch metrics or an unsupported approach: {directory}')
    if [row['epoch'] for row in epochs] != list(range(1, len(epochs) + 1)):
        raise ValueError(f'Run epoch metrics are not contiguous: {directory}')
    best = max(epochs, key=lambda row: row['validation']['pf1'])
    predictions = read_predictions(directory / 'best_predictions.csv')
    labels = read_predictions(directory / 'validation_labels.csv')
    if predictions.keys() != labels.keys():
        raise ValueError(f'Best predictions do not cover the validation breasts: {directory}')
    if any(value not in (0.0, 1.0) for value in labels.values()):
        raise ValueError(f'Validation labels must be binary: {directory}')
    score = pf1(list(labels.values()), [predictions[key] for key in labels])
    if not math.isclose(score, best['validation']['pf1'], rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f'Recomputed pF1 {score} differs from the reported best in {directory}.')
    return {'run': str(directory), 'approach': config['approach'], 'config': config,
            'epochs': epochs, 'best': best, 'labels': labels, 'predictions': predictions}


def read_predictions(path: Path) -> dict[str, float]:
    with path.open(newline='', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        if (reader.fieldnames or []) != ['prediction_id', 'cancer']:
            raise ValueError(f'Expected prediction_id,cancer columns in {path}.')
        values = {}
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f'Malformed row in {path}.')
            if row['prediction_id'] in values:
                raise ValueError(f'Duplicate prediction_id in {path}.')
            values[row['prediction_id']] = float(row['cancer'])
    if not values:
        raise ValueError(f'No rows in {path}.')
    return values


def check_comparable(runs: list[dict]) -> list[str]:
    warnings = []
    for key in COMPARABLE:
        if len({run['config']['dataset'][key] for run in runs}) > 1:
            warnings.append(f'Runs used different dataset {key}; scores are not comparable.')
    for key in COMPARABLE_ARGUMENTS:
        if len({run['config']['arguments'][key] for run in runs}) > 1:
            warnings.append(f'Runs used a different --{key}; validation patients differ.')
    if len({tuple(sorted(run['labels'].items())) for run in runs}) > 1:
        warnings.append('Runs validated on different breasts or labels; scores are not comparable.')
    for key in ('image_size', 'epochs', 'pretrained', 'pos_weight'):
        if len({run['config']['arguments'][key] for run in runs}) > 1:
            warnings.append(f'Runs differ in --{key}; a difference may come from that setting alone.')
    if any(run['config']['arguments']['max_train_batches'] is not None for run in runs):
        warnings.append('At least one run used --max-train-batches and is not fully trained.')
    if len({run['approach'] for run in runs}) < len(runs):
        warnings.append('Repeated approaches are treated as separate runs, not averaged.')
    return warnings


def bootstrap_difference(baseline: dict, candidate: dict, samples: int, seed: int) -> dict:
    keys = sorted(baseline['labels'])
    patients = {}
    for key in keys:
        patients.setdefault(key.rsplit('_', 1)[0], []).append(key)
    groups = sorted(patients.values())
    rng = random.Random(seed)
    wins = 0
    differences = []
    for _ in range(samples):
        selected = [key for _ in groups for key in rng.choice(groups)]
        labels = [baseline['labels'][key] for key in selected]
        if not 0 < sum(labels) < len(labels):
            continue
        difference = (pf1(labels, [candidate['predictions'][key] for key in selected])
                      - pf1(labels, [baseline['predictions'][key] for key in selected]))
        differences.append(difference)
        wins += difference > 0
    if not differences:
        return {'samples': 0, 'note': 'Bootstrap needs resamples containing both classes.'}
    differences.sort()
    low = differences[max(0, int(0.025 * len(differences)) - 1)]
    high = differences[min(len(differences) - 1, int(0.975 * len(differences)))]
    return {'samples': len(differences), 'mean_difference': sum(differences) / len(differences),
            'ci_low': low, 'ci_high': high, 'candidate_better_fraction': wins / len(differences),
            'crosses_zero': low <= 0 <= high}


def compare(args: argparse.Namespace) -> dict:
    if args.bootstrap < 0:
        raise ValueError('--bootstrap must be nonnegative.')
    if len(args.runs) < 1:
        raise ValueError('Pass at least one run directory.')
    runs = [load_run(directory) for directory in args.runs]
    warnings = check_comparable(runs)
    ranked = sorted(runs, key=lambda run: run['best']['validation']['pf1'], reverse=True)
    report = {
        'validation_breasts': ranked[0]['best']['validation']['breasts'],
        'validation_positive_breasts': ranked[0]['best']['validation']['positive_breasts'],
        'constant_prevalence_pf1': ranked[0]['config']['constant_prevalence_pf1'],
        'runs': [{
            'approach': run['approach'], 'run': run['run'],
            'best_epoch': run['best']['epoch'], 'epochs_completed': len(run['epochs']),
            'image_size': run['config']['arguments']['image_size'],
            'world_size': run['config'].get('world_size', 1),
            'validation_weights': run['config']['validation_weights'],
            'train_minutes': sum(row['elapsed_seconds'] for row in run['epochs']) / 60,
            **{key: run['best']['validation'][key] for key in
               ('pf1', 'roc_auc', 'average_precision', 'brier', 'f1_at_0_5')},
        } for run in ranked],
        'warnings': warnings,
    }
    if len(ranked) > 1 and args.bootstrap:
        baseline = next((run for run in ranked if run['approach'] == args.baseline), ranked[-1])
        report['bootstrap'] = {
            'baseline': baseline['approach'], 'patient_resampled': True,
            'comparisons': [{'candidate': run['approach'],
                             **bootstrap_difference(baseline, run, args.bootstrap, args.seed)}
                            for run in ranked if run is not baseline],
        }
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Compare completed training runs on the same validation fold, with patient bootstrap.'
    )
    parser.add_argument('runs', nargs='+', type=Path, help='Completed run directories.')
    parser.add_argument('--baseline', default='simple',
                        help='Approach treated as the baseline for differences (default: simple).')
    parser.add_argument('--bootstrap', type=int, default=1000,
                        help='Patient-level bootstrap resamples; 0 disables interval estimation.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', type=Path, help='Optional new JSON report path.')
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        report = compare(args)
        if args.out is not None:
            with args.out.open('x', encoding='utf-8') as file:
                json.dump(report, file, indent=2, allow_nan=False)
                file.write('\n')
    except (OSError, ValueError, KeyError, json.JSONDecodeError, csv.Error) as error:
        parser.exit(1, f'Comparison failed: {error}\n')
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
