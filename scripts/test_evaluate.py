import csv
import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

if any(importlib.util.find_spec(name) is None for name in
       ('torch', 'torchvision', 'numpy', 'PIL', 'streaming', 'tqdm')):
    raise unittest.SkipTest('Evaluation tests require the ML dependencies.')

import numpy as np
import torch
from PIL import Image

from scripts.convert_to_mds import build_parser as conversion_parser, convert
from scripts.evaluate import build_parser, evaluate, select_weights
from scripts.metrics import score_submission
from scripts.training_common import build_parser as training_parser, train


class EvaluateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.addClassCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(2)
        directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(directory.cleanup)
        cls.root = Path(directory.name)
        images = cls.root / 'images'
        images.mkdir()
        rows = []
        for patient in range(1, 13):
            for side in ['L', 'R']:
                for index, view in enumerate(['CC', 'MLO']):
                    image_id = patient * 100 + (0 if side == 'L' else 10) + index
                    pixels = np.arange(16 * 24, dtype=np.uint16).reshape(16, 24) * 100
                    Image.fromarray(pixels).save(images / f'{patient}_{image_id}.png')
                    rows.append([patient, image_id, side, view, int(patient <= 6 and side == 'L')])
        cls.csv_path = cls.root / 'train.csv'
        with cls.csv_path.open('w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['patient_id', 'image_id', 'laterality', 'view', 'cancer'])
            writer.writerows(rows)
        cls.remote = cls.root / 'mds'
        convert(conversion_parser().parse_args([
            '--csv', str(cls.csv_path), '--images-dir', str(images), '--out', str(cls.remote),
            '--workers', '1', '--prefetch', '1', '--no-progress',
        ]))
        cls.run_dir = cls.root / 'run'
        arguments = training_parser('simple').parse_args([
            '--mds', str(cls.remote), '--cache', str(cls.root / 'cache-train'),
            '--out', str(cls.run_dir), '--device', 'cpu', '--no-pretrained', '--image-size', '64',
            '--epochs', '1', '--folds', '3', '--fold', '0', '--exclude-folds', '2',
            '--batch-size', '2', '--num-workers', '0', '--no-progress',
        ])
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            cls.history = train(arguments, 'simple')
        cls.config = json.loads((cls.run_dir / 'config.json').read_text())

    def args(self, name, *options):
        return build_parser().parse_args([
            '--checkpoint', str(self.run_dir / 'best.pt'), '--mds', str(self.remote),
            '--cache', str(self.root / f'cache-{name}'), '--out', str(self.root / f'eval-{name}'),
            '--device', 'cpu', '--num-workers', '0', '--no-progress', *options,
        ])

    def run_evaluation(self, name, *options):
        with redirect_stderr(io.StringIO()):
            return evaluate(self.args(name, *options))

    def test_excluded_fold_is_absent_from_training_and_validation(self):
        self.assertEqual(self.config['excluded_folds'], [2])
        self.assertEqual(self.config['excluded_patients'], 4)
        self.assertEqual(self.config['excluded_images'], 16)
        self.assertEqual(self.config['train_images'] + self.config['validation_images']
                         + self.config['excluded_images'], 48)
        folds = {row['patient_id']: int(row['fold'])
                 for row in csv.DictReader((self.run_dir / 'folds.csv').read_text().splitlines())}
        held_out = {patient for patient, fold in folds.items() if fold == 2}
        validation = {row.split(',')[0].rsplit('_', 1)[0]
                      for row in (self.run_dir / 'validation_labels.csv').read_text().splitlines()[1:]}
        self.assertTrue(held_out)
        self.assertFalse(held_out & validation)

    def test_scores_held_out_fold_and_writes_report(self):
        report = self.run_evaluation('test', '--fold', '2')
        self.assertTrue(report['unbiased_test_estimate'])
        self.assertEqual(report['fold_role'], 'held-out test fold')
        self.assertEqual(report['images'], 16)
        self.assertEqual(report['metrics']['breasts'], 8)
        self.assertEqual(report['training_run_excluded_folds'], [2])
        self.assertEqual(report['weights'], 'model')
        self.assertEqual(len(report['checkpoint_sha256']), 64)
        output = self.root / 'eval-test'
        self.assertAlmostEqual(score_submission(output / 'labels.csv', output / 'predictions.csv'),
                               report['metrics']['pf1'])
        self.assertEqual(json.loads((output / 'evaluation.json').read_text()), report)
        with self.assertRaises(FileExistsError):
            self.run_evaluation('test', '--fold', '2')

    def test_training_folds_are_refused_and_selection_fold_is_gated(self):
        with self.assertRaisesRegex(ValueError, 'training data'):
            self.run_evaluation('train-fold', '--fold', '1')
        with self.assertRaisesRegex(ValueError, 'model selection'):
            self.run_evaluation('val-fold', '--fold', '0')
        report = self.run_evaluation('val-allowed', '--fold', '0', '--allow-selection-fold')
        self.assertFalse(report['unbiased_test_estimate'])
        self.assertIn('not an unbiased estimate', report['fold_role'])
        self.assertAlmostEqual(report['metrics']['pf1'],
                               self.history[0]['validation']['pf1'])

    def test_epoch_checkpoint_and_weight_selection(self):
        report = self.run_evaluation('epoch', '--fold', '2',
                                     '--checkpoint', str(self.run_dir / 'epoch_001.pt'))
        self.assertEqual(report['checkpoint_epoch'], 1)
        best = self.run_evaluation('best-again', '--fold', '2',
                                   '--checkpoint', str(self.run_dir / 'last.pt'))
        self.assertEqual(best['metrics']['pf1'], report['metrics']['pf1'])
        with self.assertRaisesRegex(ValueError, 'no EMA weights'):
            self.run_evaluation('ema-missing', '--fold', '2', '--weights', 'ema')
        checkpoint = torch.load(self.run_dir / 'best.pt', map_location='cpu', weights_only=True)
        self.assertEqual(select_weights({**checkpoint, 'ema_state': {'a': torch.zeros(1)}},
                                        'auto')[1], 'model')
        self.assertEqual(select_weights({**checkpoint, 'ema_state': {'a': torch.zeros(1)},
                                         'config': {'validation_weights': 'ema'}}, 'auto')[1], 'ema')

    def test_dataset_and_checkpoint_mismatches_are_rejected(self):
        other = self.root / 'other-mds'
        convert(conversion_parser().parse_args([
            '--csv', str(self.csv_path), '--images-dir', str(self.root / 'images'),
            '--out', str(other), '--workers', '1', '--prefetch', '1', '--limit', '40',
            '--no-progress',
        ]))
        with self.assertRaisesRegex(ValueError, 'differs from the trained checkpoint'):
            self.run_evaluation('other-data', '--fold', '2', '--mds', str(other))
        plain = self.root / 'plain.pt'
        torch.save({'model_state': {}}, plain)
        with self.assertRaisesRegex(ValueError, 'training config'):
            self.run_evaluation('plain', '--fold', '2', '--checkpoint', str(plain))


if __name__ == '__main__':
    unittest.main()
