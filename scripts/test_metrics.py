import csv
import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.metrics import aggregate_predictions, pf1, pfbeta, score_submission


def reference_pfbeta(labels, predictions, beta):
    positive_count = 0
    true_positive = 0.0
    false_positive = 0.0
    for label, prediction in zip(labels, predictions):
        prediction = min(max(prediction, 0), 1)
        if label:
            positive_count += 1
            true_positive += prediction
        else:
            false_positive += prediction
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / positive_count
    if precision > 0 and recall > 0:
        return (1 + beta * beta) * precision * recall / (beta * beta * precision + recall)
    return 0.0


class MetricTests(unittest.TestCase):
    def test_hand_calculated_soft_score(self):
        self.assertAlmostEqual(pf1([1, 0, 1, 0], [0.9, 0.2, 0.8, 0.1]), 0.85)

    def test_perfect_predictions(self):
        self.assertEqual(pf1([0, 1, 1], [0, 1, 1]), 1.0)

    def test_zero_cases(self):
        for labels, predictions in [([0, 0], [0.2, 0.8]), ([1, 0], [0, 0]), ([1, 0], [0, 1])]:
            with self.subTest(labels=labels, predictions=predictions):
                self.assertEqual(pf1(labels, predictions), 0.0)

    def test_clipping_without_mutation(self):
        predictions = [-2, 3]
        self.assertEqual(pf1([0, 1], predictions), 1.0)
        self.assertEqual(predictions, [-2, 3])

    def test_binary_predictions_match_f1(self):
        self.assertAlmostEqual(pf1([1, 1, 0, 0], [1, 0, 1, 0]), 0.5)

    def test_no_automatic_thresholding(self):
        self.assertNotEqual(pf1([1, 0], [0.6, 0.4]), pf1([1, 0], [1, 0]))

    def test_reference_parity(self):
        rng = random.Random(42)
        for beta in [0.5, 1.0, 2.0]:
            for _ in range(100):
                labels = [1, 0] + [rng.randrange(2) for _ in range(100)]
                predictions = [rng.uniform(-0.2, 1.2) for _ in labels]
                self.assertAlmostEqual(
                    pfbeta(labels, predictions, beta),
                    reference_pfbeta(labels, predictions, beta),
                    places=12,
                )

    def test_generators(self):
        self.assertEqual(pf1(iter([0, 1]), iter([0, 1])), 1.0)

    def test_invalid_inputs(self):
        for labels, predictions in [([], []), ([1], []), ([2], [0.5]), ([0.5], [0.5]),
                                    ([float('nan')], [0.5]), ([1], [float('nan')]),
                                    ([1], [float('inf')]), ([1], [float('-inf')])]:
            with self.subTest(labels=labels, predictions=predictions):
                with self.assertRaises(ValueError):
                    pf1(labels, predictions)

    def test_invalid_beta(self):
        for beta in [0, -1, float('nan'), float('inf'), 1e308, 1e-308]:
            with self.subTest(beta=beta):
                with self.assertRaises(ValueError):
                    pfbeta([1], [0.5], beta)


class AggregationTests(unittest.TestCase):
    def test_mean_preserves_first_seen_order(self):
        ids, labels, predictions = aggregate_predictions(
            ['2_R', '1_L', '2_R'], [0, 1, 0], [0.1, 0.8, 0.3]
        )
        self.assertEqual(ids, ['2_R', '1_L'])
        self.assertEqual(labels, [0, 1])
        self.assertEqual(predictions, [0.2, 0.8])

    def test_max(self):
        self.assertEqual(
            aggregate_predictions(['1_L', '1_L'], [1, 1], [0.2, 0.8], reduction='max'),
            (['1_L'], [1], [0.8]),
        )

    def test_grouping_changes_image_weighting(self):
        ids = ['1_L', '1_L', '2_R']
        labels = [1, 1, 0]
        predictions = [0.8, 0.8, 0.4]
        _, breast_labels, breast_predictions = aggregate_predictions(ids, labels, predictions)
        self.assertAlmostEqual(pf1(breast_labels, breast_predictions), 1.6 / 2.2)
        self.assertNotEqual(pf1(labels, predictions), pf1(breast_labels, breast_predictions))

    def test_invalid_groups(self):
        cases = [(['1_L', '1_L'], [1, 0], [0.2, 0.8]),
                 (['1_L'], [1, 1], [0.2, 0.8]),
                 ([''], [1], [0.5]), ([None], [1], [0.5]),
                 (['1_L'], [1], [2])]
        for ids, labels, predictions in cases:
            with self.subTest(ids=ids):
                with self.assertRaises(ValueError):
                    aggregate_predictions(ids, labels, predictions)
        with self.assertRaises(ValueError):
            aggregate_predictions(['1_L'], [1], [0.5], reduction='median')


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.solution = Path(self.directory.name) / 'solution.csv'
        self.submission = Path(self.directory.name) / 'submission.csv'
        self.write_csv(self.solution, [('1_L', 1), ('2_R', 0)])
        self.write_csv(self.submission, [('2_R', 0.2), ('1_L', 0.8)])

    def write_csv(self, path, rows, header=('prediction_id', 'cancer')):
        with path.open('w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerows(rows)

    def test_aligns_ids_not_row_order(self):
        self.assertAlmostEqual(score_submission(self.solution, self.submission), 0.8)

    def test_invalid_submission(self):
        for rows in [[('1_L', 0.8)], [('1_L', 0.8), ('3_R', 0.2)],
                     [('1_L', 0.8), ('1_L', 0.2)], [('', 0.8), ('2_R', 0.2)],
                     [('1_L', float('nan')), ('2_R', 0.2)],
                     [('1_L', 2), ('2_R', 0.2)], []]:
            with self.subTest(rows=rows):
                self.write_csv(self.submission, rows)
                with self.assertRaises(ValueError):
                    score_submission(self.solution, self.submission)

    def test_invalid_solution(self):
        self.write_csv(self.solution, [('1_L', 0.5), ('2_R', 0)])
        with self.assertRaises(ValueError):
            score_submission(self.solution, self.submission)

    def test_missing_columns(self):
        self.write_csv(self.submission, [('1_L', 0.8)], header=('id', 'probability'))
        with self.assertRaises(ValueError):
            score_submission(self.solution, self.submission)

    def test_duplicate_columns(self):
        self.write_csv(self.submission, [('1_L', 0.8)], header=('cancer', 'cancer'))
        with self.assertRaises(ValueError):
            score_submission(self.solution, self.submission)

    def test_cli(self):
        result = subprocess.run(
            [sys.executable, str(Path(__file__).with_name('metrics.py')),
             '--solution', str(self.solution), '--submission', str(self.submission)],
            check=True, capture_output=True, text=True,
        )
        self.assertAlmostEqual(json.loads(result.stdout)['pf1'], 0.8)

    def test_cli_rejects_invalid_csv(self):
        self.write_csv(self.submission, [])
        result = subprocess.run(
            [sys.executable, str(Path(__file__).with_name('metrics.py')),
             '--solution', str(self.solution), '--submission', str(self.submission)],
            capture_output=True, text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('error:', result.stderr)


if __name__ == '__main__':
    unittest.main()
