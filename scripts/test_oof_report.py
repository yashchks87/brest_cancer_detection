import json
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.oof_report import apply_platt, fit_platt, load_runs, logit, main, roc_auc


REPOSITORY = Path(__file__).resolve().parents[1]


def write_run(directory: Path, fold: int, labels, predictions, *, csv_hash='abc', folds=5,
              augment='strong', select_metric='average_precision', finished=True):
    directory.mkdir(parents=True)
    config = {
        'approach': 'advanced',
        'arguments': {'fold': fold, 'folds': folds, 'seed': 42, 'image_size': 512,
                      'encoder': 'convnext_tiny', 'augment': augment,
                      'select_metric': select_metric},
        'dataset': {'csv_sha256': csv_hash, 'index_sha256': 'def'},
        'selection_metric': select_metric,
    }
    (directory / 'config.json').write_text(json.dumps(config))
    rows = ['prediction_id,cancer'] + [f'{key},{value}' for key, value in labels.items()]
    (directory / 'validation_labels.csv').write_text('\n'.join(rows) + '\n')
    rows = ['prediction_id,cancer'] + [f'{key},{value}' for key, value in predictions.items()]
    (directory / 'best_predictions.csv').write_text('\n'.join(rows) + '\n')
    if finished:
        (directory / '_TRAINING_SUCCESS').write_text(json.dumps({'epochs': 8}))
    return directory


def synthetic_fold(fold, count=200, positives=20, seed=0):
    generator = np.random.default_rng(seed)
    labels, predictions = {}, {}
    for index in range(count):
        key = f'f{fold}p{index}_L'
        label = 1 if index < positives else 0
        # A ranked but badly scaled score: positives are higher, everything is small.
        score = generator.beta(2.0, 5.0) if label else generator.beta(1.0, 12.0)
        labels[key] = label
        predictions[key] = round(min(max(score, 1e-6), 1 - 1e-6), 8)
    return labels, predictions


class PlattTests(unittest.TestCase):
    def test_recalibration_recovers_a_known_distortion(self):
        generator = np.random.default_rng(1)
        truth = generator.uniform(0.01, 0.99, 4000)
        labels = (generator.uniform(size=4000) < truth).astype(float)
        distorted = 1.0 / (1.0 + np.exp(-(0.4 * logit(truth) - 1.5)))
        weights = fit_platt(logit(distorted), labels)
        self.assertAlmostEqual(weights[0], 1 / 0.4, delta=0.35)
        self.assertAlmostEqual(weights[1], 1.5 / 0.4, delta=1.0)
        recovered = apply_platt(logit(distorted), weights)
        self.assertLess(np.abs(recovered.mean() - labels.mean()), 0.02)

    def test_calibration_preserves_ranking(self):
        scores = np.array([0.01, 0.2, 0.5, 0.9])
        weights = np.array([1.7, -0.3])
        calibrated = apply_platt(logit(scores), weights)
        self.assertTrue(np.all(np.diff(calibrated) > 0))

    def test_roc_auc_handles_ties_and_perfect_separation(self):
        labels = np.array([0.0, 0.0, 1.0, 1.0])
        self.assertEqual(roc_auc(labels, np.array([0.1, 0.2, 0.8, 0.9])), 1.0)
        self.assertEqual(roc_auc(labels, np.array([0.5, 0.5, 0.5, 0.5])), 0.5)


class PoolingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def make_runs(self, folds=(0, 1, 2), **overrides):
        directories = []
        for fold in folds:
            labels, predictions = synthetic_fold(fold, seed=fold)
            directories.append(write_run(self.root / f'run{fold}', fold, labels, predictions,
                                         **overrides))
        return directories

    def test_pooling_orders_folds_and_keeps_every_breast(self):
        runs = load_runs(self.make_runs())
        self.assertEqual([run['fold'] for run in runs], [0, 1, 2])
        self.assertEqual(sum(len(run['labels']) for run in runs), 600)

    def test_duplicate_folds_and_overlapping_breasts_are_refused(self):
        labels, predictions = synthetic_fold(0)
        first = write_run(self.root / 'a', 0, labels, predictions)
        second = write_run(self.root / 'b', 0, labels, predictions)
        with self.assertRaisesRegex(ValueError, 'distinct folds'):
            load_runs([first, second])
        shifted = write_run(self.root / 'c', 1, labels, predictions)
        with self.assertRaisesRegex(ValueError, 'two folds'):
            load_runs([first, shifted])

    def test_incomparable_runs_and_unfinished_runs_are_refused(self):
        labels, predictions = synthetic_fold(0)
        first = write_run(self.root / 'a', 0, labels, predictions)
        other_labels, other_predictions = synthetic_fold(1, seed=1)
        mismatched = write_run(self.root / 'b', 1, other_labels, other_predictions, csv_hash='zzz')
        with self.assertRaisesRegex(ValueError, 'csv_sha256'):
            load_runs([first, mismatched])
        different = write_run(self.root / 'c', 1, other_labels, other_predictions, augment='light')
        with self.assertRaisesRegex(ValueError, 'augment'):
            load_runs([first, different])
        unfinished = write_run(self.root / 'd', 1, other_labels, other_predictions, finished=False)
        with self.assertRaisesRegex(ValueError, '_TRAINING_SUCCESS'):
            load_runs([first, unfinished])

    def test_command_line_report_is_cross_fitted_and_written_once(self):
        runs = self.make_runs()
        report_path = self.root / 'report.json'
        predictions_path = self.root / 'oof.csv'
        arguments = [str(run) for run in runs] + ['--out', str(report_path),
                                                  '--predictions-out', str(predictions_path),
                                                  '--bootstrap', '50']
        with unittest.mock.patch.object(sys, 'argv', ['oof_report.py', *arguments]):
            main()
        report = json.loads(report_path.read_text())
        self.assertEqual(report['folds_covered'], [0, 1, 2])
        self.assertEqual(report['pooled_raw']['breasts'], 600)
        self.assertEqual(report['pooled_raw']['positive_breasts'], 60)
        self.assertEqual(sorted(report['calibration_transforms']), ['0', '1', '2'])
        # Recalibration is monotone inside a fold, so per-fold ranking is untouched.
        # Each fold gets its own transform, so pooled ranking can shift slightly.
        self.assertAlmostEqual(report['pooled_calibrated_crossfitted']['roc_auc'],
                               report['pooled_raw']['roc_auc'], delta=0.02)
        self.assertGreater(report['pooled_calibrated_crossfitted']['pf1'],
                           report['pooled_raw']['pf1'])
        self.assertIn('warning', report)
        lines = predictions_path.read_text().splitlines()
        self.assertEqual(len(lines), 601)
        per_fold = {}
        for line in lines[1:]:
            _, fold, _, raw, calibrated = line.split(',')
            per_fold.setdefault(fold, []).append((float(raw), float(calibrated)))
        for fold, pairs in per_fold.items():
            with self.subTest(fold=fold):
                ordered = [calibrated for _, calibrated in sorted(pairs)]
                self.assertEqual(ordered, sorted(ordered))
        with self.assertRaises(SystemExit), unittest.mock.patch.object(
                sys, 'argv', ['oof_report.py', *arguments]):
            main()

    def test_help_runs_without_optional_dependencies(self):
        result = subprocess.run([sys.executable, '-B', 'scripts/oof_report.py', '--help'],
                                cwd=REPOSITORY, capture_output=True, text=True, check=True)
        self.assertIn('--predictions-out', result.stdout)


if __name__ == '__main__':
    unittest.main()
