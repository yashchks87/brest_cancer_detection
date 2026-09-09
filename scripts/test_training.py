import copy
import csv
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

if any(importlib.util.find_spec(name) is None for name in
       ('torch', 'torchvision', 'numpy', 'PIL', 'streaming', 'tqdm')):
    raise unittest.SkipTest('Training tests require the ML dependencies.')

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from scripts.benchmark_mds import RSNAStreamingDataset
from scripts.convert_to_mds import build_parser as conversion_parser, convert
from scripts.metrics import pf1, score_submission
from scripts.training_common import (
    Cluster,
    _split_logits,
    build_parser,
    positive_weight,
    run_epoch,
    save_checkpoint,
    train,
    update_ema,
    validate_args,
    validation_metrics,
)
from scripts.training_data import ImageDataset, ImageTransform, collate_samples, load_training_records


class ScalarDataset(Dataset):
    def __len__(self):
        return 5

    def __getitem__(self, index):
        return {'images': torch.full((1, 3, 1, 1), index / 5, dtype=torch.float32),
                'target': float(index % 2), 'prediction_id': str(index), 'sample_ids': [str(index)]}


class ScalarModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, images, view_mask):
        return self.linear(images[:, 0, 0, 0, 0].unsqueeze(1)).squeeze(1)


class TrainingLogicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.addClassCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(2)

    def test_metrics_handle_ties_and_breast_aggregation(self):
        labels, scores = [0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]
        metrics, rows = validation_metrics(['a', 'b', 'c', 'd'], labels, scores)
        self.assertAlmostEqual(metrics['roc_auc'], 0.75)
        self.assertAlmostEqual(metrics['average_precision'], 5 / 6)
        self.assertAlmostEqual(metrics['pf1'], pf1(labels, scores))
        self.assertEqual(len(rows), 4)
        tied, _ = validation_metrics(['a', 'b'], [0, 1], [0.5, 0.5])
        self.assertEqual(tied['roc_auc'], 0.5)
        self.assertEqual(tied['average_precision'], 0.5)
        grouped, rows = validation_metrics(['a', 'a', 'b'], [1, 1, 0], [1, 0, 0.1])
        self.assertEqual(grouped['breasts'], 2)
        self.assertAlmostEqual(grouped['pf1'], pf1([1, 0], [0.5, 0.1]))
        self.assertEqual(rows, [('a', 1, 0.5), ('b', 0, 0.1)])

    def test_one_class_metrics_are_explicit(self):
        metrics, _ = validation_metrics(['a', 'b'], [0, 0], [0.1, 0.2])
        self.assertIsNone(metrics['roc_auc'])
        self.assertIsNone(metrics['average_precision'])
        self.assertEqual(metrics['pf1'], 0)

    def test_positive_weight_uses_training_targets(self):
        self.assertEqual(positive_weight([1] + [0] * 9, 'auto'), 3)
        self.assertEqual(positive_weight([1] + [0] * 10000, 'auto'), 20)
        self.assertEqual(positive_weight([1, 0], '1'), 1)
        for value in ['0', '-1', 'nan', 'inf']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                positive_weight([1, 0], value)
        with self.assertRaises(ValueError):
            positive_weight([0, 0], 'auto')

    def test_accumulation_matches_full_batch_including_short_final_batch(self):
        torch.manual_seed(7)
        full = ScalarModel()
        accumulated = copy.deepcopy(full)
        results = []
        for model, batch_size, accumulation in [(full, 5, 1), (accumulated, 2, 3)]:
            loader = DataLoader(ScalarDataset(), batch_size=batch_size, collate_fn=collate_samples)
            result = run_epoch(model, loader, torch.device('cpu'),
                               optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
                               scaler=torch.amp.GradScaler('cuda', enabled=False),
                               grad_accum=accumulation, pos_weight=2, clip_grad=100, no_progress=True)
            self.assertEqual(result['optimizer_steps'], 1)
            self.assertEqual(result['samples'], 5)
            results.append(result)
        for expected, actual in zip(full.parameters(), accumulated.parameters()):
            torch.testing.assert_close(expected, actual)
        self.assertAlmostEqual(results[0]['loss'], results[1]['loss'], places=6)

    def test_ema_updates_parameters(self):
        model, ema = ScalarModel(), ScalarModel()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(2)
            for parameter in ema.parameters():
                parameter.zero_()
        update_ema(ema, model, 0.9)
        for parameter in ema.parameters():
            torch.testing.assert_close(parameter, torch.full_like(parameter, 0.2))

    def test_checkpoint_serializes_locally_before_copying(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / 'model.pt'
        original_save = torch.save
        files = []

        def staged_save(checkpoint, file):
            files.append(file)
            self.assertNotEqual(getattr(file, 'name', None), str(path))
            self.assertTrue(file.seekable())
            original_save(checkpoint, file)

        with patch('scripts.training_common.torch.save', side_effect=staged_save):
            save_checkpoint(path, {'model_state': {'weight': torch.tensor([1.0])}})
        self.assertEqual(len(files), 1)
        loaded = torch.load(path, map_location='cpu', weights_only=True)
        torch.testing.assert_close(loaded['model_state']['weight'], torch.tensor([1.0]))

    def test_auxiliary_logits_are_split_and_weighted(self):
        targets = torch.tensor([1.0, 0.0])
        logits = torch.tensor([[2.0, -1.0], [-2.0, 3.0]])
        main, aux = _split_logits(logits, targets, torch.tensor([[1.0], [0.0]]))
        self.assertEqual(main.tolist(), [2.0, -2.0])
        self.assertEqual(aux.tolist(), [[-1.0], [3.0]])
        plain, none = _split_logits(torch.tensor([2.0, -2.0]), targets, torch.empty(2, 0))
        self.assertEqual(plain.tolist(), [2.0, -2.0])
        self.assertIsNone(none)
        with self.assertRaisesRegex(ValueError, 'one value per target'):
            _split_logits(logits, targets, None)
        with self.assertRaisesRegex(ValueError, '3 logits per sample'):
            _split_logits(logits, targets, torch.zeros(2, 2))

    def test_auxiliary_loss_changes_gradients_only_when_weighted(self):
        class TwoHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(1, 2)

            def forward(self, images, view_mask):
                return self.linear(images[:, 0, 0, 0, 0].unsqueeze(1))

        def gradient(aux_weight):
            torch.manual_seed(0)
            model = TwoHead()
            captured = []
            model.linear.weight.register_hook(lambda grad: captured.append(grad.clone()))
            batch = {'images': torch.full((2, 1, 3, 1, 1), 0.5),
                     'view_mask': torch.ones(2, 1, dtype=torch.bool),
                     'targets': torch.tensor([1.0, 0.0]),
                     'aux_targets': torch.tensor([[1.0], [1.0]]),
                     'prediction_ids': ['a', 'b'], 'sample_ids': [['a'], ['b']]}
            optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
            run_epoch(model, [batch], torch.device('cpu'), optimizer=optimizer,
                      scaler=torch.amp.GradScaler('cpu', enabled=False), no_progress=True,
                      aux_weight=aux_weight)
            return captured[-1]

        torch.testing.assert_close(gradient(0.0)[1], torch.zeros(1))
        self.assertGreater(gradient(0.5)[1].abs().item(), 0.0)
        torch.testing.assert_close(gradient(0.0)[0], gradient(0.5)[0])

    def test_invalid_arguments_are_rejected(self):
        parser = build_parser('simple')
        for options in [('--epochs', '0'), ('--image-size', '32'), ('--grad-accum', '0'),
                        ('--lr', 'nan'), ('--weight-decay', '-1'), ('--fold', '5'),
                        ('--ema-decay', '1'), ('--intensity-max', '0')]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                validate_args(parser.parse_args(['--cache', '/tmp/cache', '--out', '/tmp/run', *options]))
        with self.assertRaisesRegex(ValueError, 'multiple of 16'):
            validate_args(parser.parse_args(['--cache', '/tmp/cache', '--out', '/tmp/run',
                                             '--image-size', '100']), 'vit')
        for approach in ('unknown', 'Simple', None):
            with self.subTest(approach=approach), self.assertRaises(ValueError):
                build_parser(approach)

    def test_distributed_environment_validation(self):
        base = ['--cache', '/tmp/cache', '--out', '/tmp/run']
        parser = build_parser('simple')
        with patch.dict('os.environ', {'WORLD_SIZE': '2', 'RANK': '0', 'LOCAL_RANK': '0'}, clear=False):
            for variable in ('MASTER_ADDR', 'MASTER_PORT'):
                os.environ.pop(variable, None)
            with self.assertRaisesRegex(ValueError, 'torchrun'):
                validate_args(parser.parse_args(base))
            os.environ.update({'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': '29500'})
            with self.assertRaisesRegex(ValueError, 'LOCAL_RANK'):
                validate_args(parser.parse_args([*base, '--device', 'cuda:1']))
            cluster = validate_args(parser.parse_args(base))
            self.assertTrue(cluster.distributed and cluster.primary)
            self.assertEqual(cluster.world_size, 2)
        with patch.dict('os.environ', {'WORLD_SIZE': '2', 'RANK': '5', 'LOCAL_RANK': '0'}):
            with self.assertRaisesRegex(ValueError, 'WORLD_SIZE'):
                validate_args(parser.parse_args(base))
        self.assertFalse(validate_args(parser.parse_args(base)).distributed)

    def test_gathered_validation_deduplicates_sampler_padding(self):
        rows = [(('1_L', ('1_10',)), 1.0, 0.8), (('1_L', ('1_11',)), 1.0, 0.6),
                (('2_R', ('2_20',)), 0.0, 0.2)]
        cluster = Cluster(rank=0, world_size=2)
        with patch.object(Cluster, 'gather', lambda self, local: rows + [rows[0]]):
            unique = {}
            for key, label, probability in sorted(cluster.gather(rows)):
                unique.setdefault(key, (label, probability))
        self.assertEqual(len(unique), 3)
        metrics, aggregated = validation_metrics([key[0] for key in unique],
                                                 [label for label, _ in unique.values()],
                                                 [value for _, value in unique.values()])
        self.assertEqual(metrics['breasts'], 2)
        self.assertEqual(dict((key, value) for key, _, value in aggregated),
                         {'1_L': 0.7, '2_R': 0.2})


class TrainingSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.addClassCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(2)
        directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(directory.cleanup)
        cls.root = Path(directory.name)
        images = cls.root / 'images'
        images.mkdir()
        cls.csv_path = cls.root / 'train.csv'
        rows = []
        cls.boxes = {}
        for patient in range(1, 5):
            for side in ['L', 'R']:
                for view_index, view in enumerate(['CC', 'MLO']):
                    image_id = patient * 100 + (0 if side == 'L' else 10) + view_index
                    cancer = int(patient <= 2 and side == 'L')
                    # A bright blob on a dark frame, on the side the vendor happens to use,
                    # so ROI cropping and canonical flipping have something real to do.
                    pixels = np.zeros((16, 24), dtype=np.uint16)
                    box = (0, 3, 9, 13) if side == 'L' else (15, 3, 24, 13)
                    pixels[box[1]:box[3], box[0]:box[2]] = 40000 + 1000 * view_index
                    Image.fromarray(pixels).save(images / f'{patient}_{image_id}.png')
                    cls.boxes[f'{patient}_{image_id}'] = box
                    rows.append([patient, image_id, side, view, cancer,
                                 max(cancer, int(patient == 3))])
        with cls.csv_path.open('w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['patient_id', 'image_id', 'laterality', 'view', 'cancer', 'biopsy'])
            writer.writerows(rows)
        cls.roi_csv = cls.root / 'roi_boxes.csv'
        with cls.roi_csv.open('w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['sample_id', 'x0', 'y0', 'x1', 'y1'])
            writer.writerows((sample_id, *box) for sample_id, box in cls.boxes.items())
        cls.remote = cls.root / 'mds'
        convert(conversion_parser().parse_args([
            '--csv', str(cls.csv_path), '--images-dir', str(images), '--out', str(cls.remote),
            '--workers', '1', '--prefetch', '1', '--no-progress',
        ]))

    def args(self, approach, name, extra=()):
        return build_parser(approach).parse_args([
            '--mds', str(self.remote), '--cache', str(self.root / f'cache-{name}'),
            '--out', str(self.root / f'run-{name}'), '--device', 'cpu', '--no-pretrained',
            '--image-size', '64', '--epochs', '1', '--folds', '2', '--batch-size', '2',
            '--grad-accum', '2', '--max-train-batches', '1', '--num-workers', '0', '--no-progress',
            *extra,
        ])

    def test_roi_oversampling_and_aux_heads_train_end_to_end(self):
        variants = {
            'cached-boxes': ['--roi-crop', '--roi-boxes', str(self.roi_csv),
                             '--canonical-side', 'left', '--positive-fraction', '0.5',
                             '--pos-weight', '1', '--aux-targets', 'biopsy', '--aux-weight', '0.3'],
            'detected-boxes': ['--roi-crop', '--roi-margin', '0.1'],
        }
        for approach in ('advanced', 'vit'):
            for name, extra in variants.items():
                with self.subTest(approach=approach, variant=name):
                    args = self.args(approach, f'{approach}-{name}', extra)
                    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), patch(
                        'torchvision.models._api.load_state_dict_from_url',
                        side_effect=AssertionError('Unexpected weights download'),
                    ):
                        history = train(args, approach)
                    self.assertEqual(len(history), 1)
                    self.assertEqual(history[0]['validation']['breasts'], 4)
                    config = json.loads((args.out / 'config.json').read_text())
                    self.assertTrue(config['roi_crop'])
                    self.assertEqual(config['roi_boxes_cached'], name == 'cached-boxes')
                    if name == 'cached-boxes':
                        self.assertEqual(config['aux_targets'], ['biopsy'])
                        self.assertEqual(config['model_outputs'], 2)
                        self.assertEqual(config['positive_sampling'], 'balanced')
                        self.assertEqual(config['canonical_side'], 'left')
                    else:
                        self.assertEqual(config['model_outputs'], 1)
                        self.assertEqual(config['positive_sampling'], 'natural')
                    score = score_submission(args.out / 'validation_labels.csv',
                                             args.out / 'best_predictions.csv')
                    self.assertAlmostEqual(score, history[0]['validation']['pf1'])

    def test_new_features_are_off_by_default(self):
        args = self.args('advanced', 'defaults-probe')
        self.assertFalse(args.roi_crop)
        self.assertIsNone(args.roi_boxes)
        self.assertEqual(args.canonical_side, 'none')
        self.assertIsNone(args.positive_fraction)
        self.assertEqual(args.aux_targets, [])
        for extra, message in (
            (['--positive-fraction', '0'], 'positive-fraction'),
            (['--positive-fraction', '1.5'], 'positive-fraction'),
            (['--aux-targets', 'biopsy', 'biopsy'], 'repeat a column'),
            (['--aux-targets', 'cancer'], 'cancer target itself'),
            (['--aux-weight', '-1'], 'aux-weight'),
            (['--roi-margin', '-0.1'], 'roi-margin'),
            (['--roi-boxes', str(self.roi_csv)], 'no effect without'),
            (['--roi-boxes', str(self.root / 'absent.csv'), '--roi-crop'], 'existing CSV'),
        ):
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, message):
                validate_args(self.args('advanced', 'reject-probe', extra), 'advanced')

    def test_real_simple_and_advanced_cpu_training(self):
        fold_files = []
        for approach in ['simple', 'advanced']:
            with self.subTest(approach=approach):
                args = self.args(approach, approach)
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), patch(
                    'torchvision.models._api.load_state_dict_from_url',
                    side_effect=AssertionError('Unexpected weights download'),
                ):
                    history = train(args, approach)
                self.assertEqual(len(history), 1)
                self.assertEqual(history[0]['optimizer_steps'], 1)
                self.assertEqual(history[0]['validation']['breasts'], 4)
                self.assertEqual(history[0]['validation']['positive_breasts'], 1)
                self.assertTrue((args.out / '_TRAINING_SUCCESS').is_file())
                self.assertTrue((args.out / 'last.pt').is_file())
                epoch_checkpoint = torch.load(args.out / 'epoch_001.pt', map_location='cpu',
                                              weights_only=True)
                self.assertEqual(epoch_checkpoint['epoch'], 1)
                self.assertEqual(epoch_checkpoint['metrics'], history[0]['validation'])
                self.assertEqual(epoch_checkpoint['model_state'].keys(),
                                 torch.load(args.out / 'last.pt', map_location='cpu',
                                            weights_only=True)['model_state'].keys())
                self.assertEqual(bool(epoch_checkpoint['ema_state']), approach != 'simple')
                best = torch.load(args.out / 'best.pt', map_location='cpu', weights_only=True)
                self.assertEqual(best['config']['approach'], approach)
                self.assertIn('model_state', best)
                score = score_submission(args.out / 'validation_labels.csv', args.out / 'best_predictions.csv')
                self.assertAlmostEqual(score, best['metrics']['pf1'])
                fold_files.append((args.out / 'folds.csv').read_bytes())
                with self.assertRaises(FileExistsError):
                    train(args, approach)
        self.assertEqual(fold_files[0], fold_files[1])

    def test_spawn_workers_read_map_style_mds_without_duplicates(self):
        _, records = load_training_records(self.remote)
        backend = RSNAStreamingDataset(remote=str(self.remote), local=str(self.root / 'spawn-cache'),
                                       decode_images=True, shuffle=False, batch_size=1)
        dataset = ImageDataset(backend, records, range(len(records)), ImageTransform(64))
        loader = DataLoader(dataset, batch_size=2, num_workers=2, multiprocessing_context='spawn',
                            collate_fn=collate_samples)
        observed = [sample_id for batch in loader for ids in batch['sample_ids'] for sample_id in ids]
        self.assertEqual(observed, [record.sample_id for record in records])

    def test_torchrun_two_processes_match_single_process_split(self):
        single = self.args('simple', 'ddp-single')
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            history = train(single, 'simple')
        output = self.root / 'run-ddp'
        result = subprocess.run(
            [sys.executable, '-B', '-m', 'torch.distributed.run', '--nnodes', '1',
             '--nproc-per-node', '2', '--master-port', '29517', 'scripts/train_simple.py',
             '--mds', str(self.remote), '--cache', str(self.root / 'cache-ddp'),
             '--out', str(output), '--device', 'cpu', '--no-pretrained', '--image-size', '64',
             '--epochs', '1', '--folds', '2', '--batch-size', '2', '--num-workers', '0',
             '--no-progress'],
            cwd=str(Path(__file__).resolve().parents[1]), text=True, capture_output=True,
            env={**os.environ, 'OMP_NUM_THREADS': '1'}, timeout=900,
        )
        self.assertEqual(result.returncode, 0, result.stderr[-4000:])
        setup = json.loads(result.stdout.splitlines()[0])
        self.assertEqual(setup['world_size'], 2)
        self.assertEqual(setup['validation_breasts'], 4)
        self.assertTrue((output / '_TRAINING_SUCCESS').is_file())
        self.assertEqual(json.loads((output / '_TRAINING_SUCCESS').read_text())['world_size'], 2)
        epochs = [json.loads(line) for line in (output / 'metrics.jsonl').read_text().splitlines()]
        self.assertEqual(len(epochs), 1)
        self.assertEqual(epochs[0]['validation']['breasts'], 4)
        self.assertEqual(epochs[0]['validation']['positive_breasts'], 1)
        self.assertEqual((output / 'folds.csv').read_bytes(), (single.out / 'folds.csv').read_bytes())
        self.assertEqual((output / 'validation_labels.csv').read_bytes(),
                         (single.out / 'validation_labels.csv').read_bytes())
        distributed_rows = (output / 'validation_predictions_epoch_001.csv').read_text().splitlines()
        single_rows = (single.out / 'validation_predictions_epoch_001.csv').read_text().splitlines()
        self.assertEqual(len(distributed_rows), len(single_rows))
        self.assertEqual({row.split(',')[0] for row in distributed_rows[1:]},
                         {row.split(',')[0] for row in single_rows[1:]})
        self.assertEqual(history[0]['validation']['breasts'], epochs[0]['validation']['breasts'])

    def test_epoch_checkpoints_are_kept_per_epoch_and_can_be_disabled(self):
        args = self.args('simple', 'epochs')
        args.epochs = 3
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            history = train(args, 'simple')
        saved = sorted(path.name for path in args.out.glob('epoch_*.pt'))
        self.assertEqual(saved, ['epoch_001.pt', 'epoch_002.pt', 'epoch_003.pt'])
        for index, name in enumerate(saved):
            checkpoint = torch.load(args.out / name, map_location='cpu', weights_only=True)
            self.assertEqual(checkpoint['epoch'], index + 1)
            self.assertEqual(checkpoint['metrics']['pf1'], history[index]['validation']['pf1'])
        best = torch.load(args.out / 'best.pt', map_location='cpu', weights_only=True)
        matching = torch.load(args.out / f'epoch_{best["epoch"]:03d}.pt', map_location='cpu',
                              weights_only=True)
        for key, value in best['model_state'].items():
            torch.testing.assert_close(value, matching['model_state'][key])
        disabled = self.args('simple', 'no-epochs')
        disabled.save_epochs = False
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            train(disabled, 'simple')
        self.assertEqual(list(disabled.out.glob('epoch_*.pt')), [])
        self.assertTrue((disabled.out / 'best.pt').is_file())
        self.assertTrue((disabled.out / 'last.pt').is_file())

    def test_output_cannot_be_nested_in_dataset_or_cache(self):
        args = self.args('simple', 'unsafe')
        args.out = self.remote / 'run'
        with self.assertRaisesRegex(ValueError, 'separate'):
            train(args, 'simple')
        self.assertFalse(args.out.exists())


if __name__ == '__main__':
    unittest.main()
