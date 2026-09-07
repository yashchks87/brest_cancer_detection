import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.compare_runs import build_parser, compare, load_run
from scripts.metrics import pf1


class CompareTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.labels = {'1_L': 1, '1_R': 0, '2_L': 0, '2_R': 1, '3_L': 0, '3_R': 0}

    def write_run(self, name, approach, predictions, *, labels=None, epochs=2, success=True,
                  csv_sha256='csv-hash', fold=0, image_size=512, max_train_batches=None):
        labels = self.labels if labels is None else labels
        run = self.root / name
        run.mkdir()
        score = pf1(list(labels.values()), [predictions[key] for key in labels])
        config = {
            'approach': approach,
            'arguments': {'folds': 5, 'fold': fold, 'seed': 42, 'image_size': image_size,
                          'epochs': epochs, 'pretrained': True, 'pos_weight': 'auto',
                          'max_train_batches': max_train_batches},
            'dataset': {'csv_sha256': csv_sha256, 'index_sha256': 'index-hash', 'samples': 100},
            'constant_prevalence_pf1': 0.2, 'validation_weights': 'model', 'world_size': 2,
        }
        (run / 'config.json').write_text(json.dumps(config))
        rows = []
        for epoch in range(1, epochs + 1):
            value = score if epoch == epochs else score / 2
            rows.append({'epoch': epoch, 'elapsed_seconds': 60.0, 'validation': {
                'pf1': value, 'roc_auc': 0.7, 'average_precision': 0.4, 'brier': 0.1,
                'f1_at_0_5': 0.3, 'breasts': len(labels), 'positive_breasts': sum(labels.values())}})
        (run / 'metrics.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
        for filename, values in (('validation_labels.csv', labels),
                                 ('best_predictions.csv', predictions)):
            with (run / filename).open('w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['prediction_id', 'cancer'])
                writer.writerows(values.items())
        if success:
            (run / '_TRAINING_SUCCESS').write_text(json.dumps({'epochs': epochs}))
        return run

    def strong(self):
        return {'1_L': 0.9, '1_R': 0.1, '2_L': 0.2, '2_R': 0.8, '3_L': 0.05, '3_R': 0.1}

    def weak(self):
        return {'1_L': 0.4, '1_R': 0.5, '2_L': 0.5, '2_R': 0.3, '3_L': 0.5, '3_R': 0.5}

    def report(self, *runs, **options):
        arguments = [str(run) for run in runs]
        for key, value in options.items():
            arguments.extend([f'--{key.replace("_", "-")}', str(value)])
        return compare(build_parser().parse_args(arguments))

    def test_ranks_runs_and_recomputes_reported_score(self):
        baseline = self.write_run('simple', 'simple', self.weak())
        candidate = self.write_run('vit', 'vit', self.strong(), image_size=512)
        report = self.report(baseline, candidate, bootstrap=0)
        self.assertEqual([run['approach'] for run in report['runs']], ['vit', 'simple'])
        self.assertGreater(report['runs'][0]['pf1'], report['runs'][1]['pf1'])
        self.assertEqual(report['runs'][0]['best_epoch'], 2)
        self.assertEqual(report['runs'][0]['world_size'], 2)
        self.assertAlmostEqual(report['runs'][0]['train_minutes'], 2.0)
        self.assertEqual(report['validation_breasts'], 6)
        self.assertEqual(report['validation_positive_breasts'], 2)
        self.assertEqual(report['warnings'], [])

    def test_bootstrap_reports_uncertainty_over_patients(self):
        baseline = self.write_run('simple', 'simple', self.weak())
        candidate = self.write_run('vit', 'vit', self.strong())
        report = self.report(baseline, candidate, bootstrap=200)
        comparison = report['bootstrap']['comparisons'][0]
        self.assertEqual(report['bootstrap']['baseline'], 'simple')
        self.assertEqual(comparison['candidate'], 'vit')
        self.assertGreater(comparison['samples'], 0)
        self.assertGreater(comparison['mean_difference'], 0)
        self.assertLessEqual(comparison['ci_low'], comparison['mean_difference'])
        self.assertLessEqual(comparison['mean_difference'], comparison['ci_high'])
        self.assertIn('crosses_zero', comparison)
        identical = self.report(baseline, self.write_run('vit2', 'vit', self.weak()), bootstrap=200)
        self.assertAlmostEqual(identical['bootstrap']['comparisons'][0]['mean_difference'], 0)
        self.assertTrue(identical['bootstrap']['comparisons'][0]['crosses_zero'])

    def test_incomparable_runs_are_flagged_not_silently_ranked(self):
        baseline = self.write_run('simple', 'simple', self.weak())
        for name, options in [('other-data', {'csv_sha256': 'different'}),
                              ('other-fold', {'fold': 3}),
                              ('other-size', {'image_size': 1024}),
                              ('debug', {'max_train_batches': 5})]:
            with self.subTest(name=name):
                report = self.report(baseline, self.write_run(name, 'vit', self.strong(), **options))
                self.assertTrue(report['warnings'])
        different = {'1_L': 1, '1_R': 0, '9_L': 1}
        report = self.report(baseline, self.write_run('other-breasts', 'vit',
                                                     {'1_L': 0.9, '1_R': 0.1, '9_L': 0.8},
                                                     labels=different), bootstrap=0)
        self.assertTrue(any('different breasts' in warning for warning in report['warnings']))

    def test_unfinished_or_inconsistent_runs_rejected(self):
        with self.assertRaisesRegex(ValueError, '_TRAINING_SUCCESS'):
            load_run(self.write_run('unfinished', 'vit', self.strong(), success=False))
        tampered = self.write_run('tampered', 'vit', self.strong())
        (tampered / 'best_predictions.csv').write_text('prediction_id,cancer\n1_L,0.1\n')
        with self.assertRaisesRegex(ValueError, 'do not cover'):
            load_run(tampered)
        mismatched = self.write_run('mismatched', 'vit', self.strong())
        rows = [json.loads(line) for line in (mismatched / 'metrics.jsonl').read_text().splitlines()]
        rows[-1]['validation']['pf1'] = 0.99
        (mismatched / 'metrics.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
        with self.assertRaisesRegex(ValueError, 'Recomputed pF1'):
            load_run(mismatched)

    def test_report_file_is_never_overwritten(self):
        run = self.write_run('simple', 'simple', self.weak())
        destination = self.root / 'report.json'
        destination.write_text('keep')
        with self.assertRaises(FileExistsError):
            with destination.open('x'):
                pass
        self.assertEqual(destination.read_text(), 'keep')
        report = self.report(run, bootstrap=0)
        self.assertEqual(len(report['runs']), 1)
        self.assertNotIn('bootstrap', report)


if __name__ == '__main__':
    unittest.main()
