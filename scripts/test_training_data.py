import copy
import csv
import hashlib
import importlib.util
import io
import json
import random
import tempfile
import unittest
from contextlib import redirect_stderr
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from unittest.mock import Mock, patch

if any(importlib.util.find_spec(name) is None for name in
       ('torch', 'torchvision', 'numpy', 'PIL', 'streaming')):
    raise unittest.SkipTest('Training data tests require the ML dependencies.')

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from scripts.benchmark_mds import RSNAStreamingDataset
from scripts.convert_to_mds import build_parser, convert
from scripts.training_data import (
    BreastDataset,
    ImageDataset,
    ImageTransform,
    TrainingRecord,
    collate_samples,
    load_training_records,
    patient_fold_assignments,
)


class MetadataTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.csv_path = self.root / 'train.csv'
        self.images = self.root / 'unavailable_images'
        self.remote = self.root / 'mds'
        self.remote.mkdir()
        self.rows = [
            ['1', '10', 'L', 'CC', '1'], ['1', '11', 'L', 'MLO', '1'],
            ['1', '12', 'R', 'CC', '0'], ['2', '20', 'L', 'CC', '0'],
            ['2', '21', 'L', 'MLO', '0'], ['3', '30', 'R', 'CC', '1'],
            ['4', '40', 'L', 'CC', '0'], ['5', '50', 'R', 'CC', '0'],
        ]
        with self.csv_path.open('w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['patient_id', 'image_id', 'laterality', 'view', 'cancer'])
            writer.writerows(self.rows)
        index = json.dumps({'shards': [{'samples': 5}]}).encode()
        (self.remote / 'index.json').write_bytes(index)
        (self.remote / '_SUCCESS').write_text(json.dumps({'samples': 5}))
        self.manifest = {
            'schema_version': 1, 'image_storage': 'bytes', 'csv': str(self.csv_path),
            'csv_sha256': hashlib.sha256(self.csv_path.read_bytes()).hexdigest(),
            'images_dir': str(self.images), 'image_pattern': '{patient_id}_{image_id}.png',
            'csv_rows': 8, 'selected_samples': 6, 'source_files_checked': 6,
            'samples': 5, 'patients': 3, 'breasts': 4, 'positive_images': 2,
            'limit': 6, 'shuffle_seed': 23, 'missing_policy': 'skip',
            'skipped_missing_images': 1, 'skipped_images': [{
                'sample_id': '1_11', 'patient_id': '1', 'image_id': '11',
                'prediction_id': '1_L', 'cancer': 1, 'path': str(self.images / '1_11.png'),
            }],
            'index_sha256': hashlib.sha256(index).hexdigest(),
        }
        self.save_manifest(self.manifest)

    def save_manifest(self, manifest):
        (self.remote / 'conversion.json').write_text(json.dumps(manifest))

    def test_order_counts_and_no_source_image_access(self):
        original_stat, original_open = Path.stat, Path.open

        def guarded_stat(path, *args, **kwargs):
            if path == self.images or self.images in path.parents:
                self.fail(f'Training inspected a source image path: {path}')
            return original_stat(path, *args, **kwargs)

        def guarded_open(path, *args, **kwargs):
            if path == self.images or self.images in path.parents or path.suffix == '.mds':
                self.fail(f'Metadata loading opened image data: {path}')
            return original_open(path, *args, **kwargs)

        with patch.object(Path, 'stat', guarded_stat), patch.object(Path, 'open', guarded_open):
            manifest, records = load_training_records(self.remote)
        expected = self.rows[:6]
        random.Random(23).shuffle(expected)
        expected = [row for row in expected if row[1] != '11']
        self.assertEqual(manifest, self.manifest)
        self.assertEqual(records, [TrainingRecord(i, f'{row[0]}_{row[1]}', row[0],
                                                  f'{row[0]}_{row[2]}', int(row[4]))
                                   for i, row in enumerate(expected)])
        with self.assertRaises(FrozenInstanceError):
            records[0].index = 10

    def test_unshuffled_order(self):
        self.manifest['shuffle_seed'] = None
        self.save_manifest(self.manifest)
        _, records = load_training_records(self.remote)
        self.assertEqual([record.sample_id for record in records],
                         ['1_10', '1_12', '2_20', '2_21', '3_30'])

    def test_no_limit_or_skips(self):
        index = json.dumps({'shards': [{'samples': 8}]}).encode()
        (self.remote / 'index.json').write_bytes(index)
        (self.remote / '_SUCCESS').write_text(json.dumps({'samples': 8}))
        self.manifest.update(limit=None, selected_samples=8, source_files_checked=8, samples=8,
                             patients=5, breasts=6, positive_images=3, skipped_missing_images=0,
                             skipped_images=[], missing_policy='error',
                             index_sha256=hashlib.sha256(index).hexdigest())
        self.save_manifest(self.manifest)
        _, records = load_training_records(self.remote)
        expected = list(self.rows)
        random.Random(23).shuffle(expected)
        self.assertEqual([record.sample_id for record in records],
                         [f'{row[0]}_{row[1]}' for row in expected])

    def test_identical_csv_override_at_different_path(self):
        override = self.root / 'relocated.csv'
        override.write_bytes(self.csv_path.read_bytes())
        self.csv_path.unlink()
        self.assertEqual(len(load_training_records(self.remote, override)[1]), 5)
        override.write_bytes(override.read_bytes() + b'\n')
        with self.assertRaisesRegex(ValueError, 'CSV SHA256'):
            load_training_records(self.remote, override)

    def test_manifest_count_mismatches(self):
        for key in ('csv_rows', 'selected_samples', 'source_files_checked', 'samples',
                    'patients', 'breasts', 'positive_images', 'skipped_missing_images'):
            with self.subTest(key=key):
                manifest = {**self.manifest, key: self.manifest[key] + 1}
                self.save_manifest(manifest)
                with self.assertRaises(ValueError):
                    load_training_records(self.remote)

    def test_inconsistent_skipped_image_metadata(self):
        for key, value in (('sample_id', '4_40'), ('patient_id', '9'), ('image_id', '10'),
                           ('prediction_id', '1_R'), ('cancer', 0), ('cancer', True),
                           ('path', '/wrong/path.png')):
            with self.subTest(key=key, value=value):
                manifest = copy.deepcopy(self.manifest)
                manifest['skipped_images'][0][key] = value
                self.save_manifest(manifest)
                with self.assertRaisesRegex(ValueError, 'skipped|Skipped'):
                    load_training_records(self.remote)

    def test_duplicate_skips_and_unreported_exclusions(self):
        manifest = copy.deepcopy(self.manifest)
        manifest['skipped_images'] *= 2
        manifest['skipped_missing_images'] = 2
        self.save_manifest(manifest)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            load_training_records(self.remote)
        manifest = {**self.manifest, 'skipped_images': [], 'skipped_missing_images': 0}
        self.save_manifest(manifest)
        with self.assertRaisesRegex(ValueError, 'samples'):
            load_training_records(self.remote)

    def test_unsupported_metadata(self):
        for key, value in (('limit', 0), ('shuffle_seed', '23'), ('skipped_images', {}),
                           ('missing_policy', 'error'), ('missing_policy', 'unknown'),
                           ('schema_version', 2), ('image_storage', 'jpeg'), ('csv', None)):
            with self.subTest(key=key, value=value):
                self.save_manifest({**self.manifest, key: value})
                with self.assertRaises(ValueError):
                    load_training_records(self.remote)
        manifest = dict(self.manifest)
        del manifest['shuffle_seed']
        self.save_manifest(manifest)
        with self.assertRaisesRegex(ValueError, 'missing shuffle_seed'):
            load_training_records(self.remote)

    def test_completion_and_index_checks(self):
        (self.remote / 'index.json').write_bytes((self.remote / 'index.json').read_bytes() + b' ')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            load_training_records(self.remote)
        (self.remote / '_SUCCESS').unlink()
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            load_training_records(self.remote)


class FoldTests(unittest.TestCase):
    def setUp(self):
        self.records = []
        for patient in range(12):
            for side in ('L', 'R'):
                for view in range(2):
                    self.records.append(TrainingRecord(len(self.records), f'{patient}_{side}_{view}',
                                                       str(patient), f'{patient}_{side}',
                                                       int(patient >= 6 and side == 'L')))

    def test_deterministic_patient_disjoint_stratification(self):
        folds = patient_fold_assignments(self.records, 3, 42)
        self.assertEqual(folds, patient_fold_assignments(self.records, 3, 42))
        self.assertEqual(folds, patient_fold_assignments(reversed(self.records), 3, 42))
        self.assertNotEqual(folds, patient_fold_assignments(self.records, 3, 43))
        self.assertEqual(set(folds), {str(patient) for patient in range(12)})
        for fold in range(3):
            train = {record.patient_id for record in self.records if folds[record.patient_id] != fold}
            valid = {record.patient_id for record in self.records if folds[record.patient_id] == fold}
            self.assertFalse(train & valid)
            self.assertEqual(sum(int(patient) >= 6 for patient in valid), 2)
            self.assertEqual(sum(int(patient) < 6 for patient in valid), 2)
            self.assertEqual(sum(folds[record.patient_id] == fold for record in self.records), 16)

    def test_fold_errors(self):
        for count in (0, 1, 2.5):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, 'at least 2'):
                patient_fold_assignments(self.records, count, 42)
        with self.assertRaisesRegex(ValueError, 'negative patients'):
            patient_fold_assignments(self.records, 7, 42)
        with self.assertRaisesRegex(ValueError, 'positive patients'):
            patient_fold_assignments([record for record in self.records if int(record.patient_id) < 7], 2, 42)
        with self.assertRaisesRegex(ValueError, 'binary'):
            patient_fold_assignments([replace(self.records[0], label=2)], 2, 42)


class TransformTests(unittest.TestCase):
    def unnormalize(self, tensor):
        mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        return tensor * std + mean

    def test_true_16_bit_scaling(self):
        pixels = np.array([[0, 256], [4096, 65535]], dtype=np.uint16)
        expected = torch.tensor(pixels.astype(np.float32) / 65535).expand(3, -1, -1)
        for dtype in (np.uint16, np.int32, np.dtype('>u2')):
            with self.subTest(dtype=dtype):
                output = ImageTransform(2)(pixels.astype(dtype))
                self.assertEqual(output.dtype, torch.float32)
                torch.testing.assert_close(self.unnormalize(output), expected)
        low = np.full((2, 2), 4095, dtype=np.uint16)
        torch.testing.assert_close(self.unnormalize(ImageTransform(2)(low)),
                                   torch.full((3, 2, 2), 4095 / 65535))
        torch.testing.assert_close(self.unnormalize(ImageTransform(2, intensity_max=4095)(low)),
                                   torch.ones(3, 2, 2))

    def test_uint8_rgb_and_float_scaling(self):
        gray = np.array([[0, 64], [128, 255]], dtype=np.uint8)
        for pixels, maximum in ((gray, None), (gray[..., None], None),
                                 (gray.astype(np.float32) / 255, None),
                                 (gray.astype(np.float64), 255)):
            with self.subTest(dtype=pixels.dtype, maximum=maximum):
                output = ImageTransform(2, intensity_max=maximum)(pixels)
                torch.testing.assert_close(self.unnormalize(output),
                                           torch.tensor(gray.astype(np.float32) / 255).expand(3, -1, -1))
        rgb = np.stack([gray, 255 - gray, gray], axis=-1)
        before = rgb.copy()
        output = ImageTransform(2)(rgb)
        torch.testing.assert_close(self.unnormalize(output),
                                   torch.tensor(rgb.astype(np.float32) / 255).permute(2, 0, 1))
        np.testing.assert_array_equal(rgb, before)
        large = np.full((2, 2), 2e38, dtype=np.float32)
        output = ImageTransform(2, intensity_max=4e38)(large)
        torch.testing.assert_close(self.unnormalize(output), torch.full((3, 2, 2), 0.5))

    def test_aspect_preserving_padding_without_crop(self):
        wide = self.unnormalize(ImageTransform(4)(np.full((2, 4), 255, dtype=np.uint8)))
        expected = torch.zeros(3, 4, 4)
        expected[:, 1:3] = 1
        torch.testing.assert_close(wide, expected)
        tall = self.unnormalize(ImageTransform(4)(np.full((4, 2), 255, dtype=np.uint8)))
        torch.testing.assert_close(tall, expected.transpose(1, 2))
        narrow = ImageTransform(4)(np.ones((1, 100), dtype=np.uint8))
        self.assertEqual(narrow.shape, (3, 4, 4))

    def test_validation_determinism_and_seeded_training(self):
        pixels = np.arange(128, dtype=np.uint8).reshape(8, 16)
        transform = ImageTransform(12)
        torch.manual_seed(1)
        first = transform(pixels)
        torch.manual_seed(2)
        self.assertTrue(torch.equal(first, transform(pixels)))
        train_transform = ImageTransform(12, training=True)
        torch.manual_seed(8)
        first = train_transform(pixels)
        torch.manual_seed(8)
        self.assertTrue(torch.equal(first, train_transform(pixels)))
        self.assertFalse(torch.equal(first, transform(pixels)))
        self.assertEqual(first.shape, (3, 12, 12))
        self.assertTrue(torch.isfinite(first).all())

    def test_invalid_ranges_shapes_and_dtypes(self):
        arrays = [np.array([0, 1]), np.empty((0, 2), dtype=np.uint8),
                  np.zeros((2, 2, 4), dtype=np.uint8), np.zeros((2, 2), dtype=bool),
                  np.zeros((2, 2), dtype=np.int16), np.zeros((2, 2), dtype=np.uint32),
                  np.zeros((2, 2), dtype=np.complex64), np.zeros((2, 2, 3), dtype=np.int32),
                  np.array([[-1]], dtype=np.int32), np.array([[65536]], dtype=np.int32),
                  np.array([[1.01]]), np.array([[np.nan]]), np.array([[np.inf]]),
                  np.array([[-0.01]])]
        for pixels in arrays:
            with self.subTest(dtype=pixels.dtype, shape=pixels.shape), self.assertRaises(ValueError):
                ImageTransform(4)(pixels)
        with self.assertRaises(ValueError):
            ImageTransform(2, intensity_max=4095)(np.array([[4096]], dtype=np.uint16))
        for maximum in (0, -1, float('nan'), float('inf')):
            with self.subTest(maximum=maximum), self.assertRaises(ValueError):
                ImageTransform(2, intensity_max=maximum)
        for size in (0, -1, 2.5):
            with self.subTest(size=size), self.assertRaises(ValueError):
                ImageTransform(size)


class FakeBackend:
    def __init__(self, records):
        self.calls = []
        self.samples = {
            record.index: {'sample_id': record.sample_id, 'patient_id': record.patient_id,
                           'prediction_id': record.prediction_id, 'cancer': record.label,
                           'image': np.full((2, 4), 127, dtype=np.uint8),
                           'metadata': {'cancer': record.label, 'age': '60', 'biopsy': '1'}}
            for record in records
        }

    def __getitem__(self, index):
        self.calls.append(index)
        return self.samples[index]


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.records = [TrainingRecord(i, f'1_{i}', '1', '1_L', 1) for i in range(5)]
        self.records += [TrainingRecord(5, '1_5', '1', '1_R', 0)]
        self.backend = FakeBackend(self.records)
        self.transform = ImageTransform(4)

    def test_image_schema_global_indices_and_shared_backend(self):
        records = [replace(self.records[0], index=9), replace(self.records[1], index=2)]
        backend = FakeBackend(records)
        train = ImageDataset(backend, records, [1], self.transform)
        valid = ImageDataset(backend, records, [0], self.transform)
        self.assertIs(train.backend, valid.backend)
        self.assertEqual(backend.calls, [])
        sample = train[0]
        valid[0]
        self.assertEqual(backend.calls, [2, 9])
        self.assertEqual(train.targets, [1])
        self.assertEqual(set(sample), {'images', 'target', 'prediction_id', 'sample_ids'})
        self.assertEqual(sample['images'].shape, (1, 3, 4, 4))
        self.assertEqual(sample['sample_ids'], ['1_1'])
        self.assertEqual(sample['target'], 1.0)

    def test_labels_and_metadata_are_not_features(self):
        transform = Mock(side_effect=self.transform)
        dataset = ImageDataset(self.backend, self.records, [0, 5], transform)
        positive, negative = dataset[0], dataset[1]
        self.assertNotEqual(positive['target'], negative['target'])
        self.assertTrue(torch.equal(positive['images'], negative['images']))
        for call in transform.call_args_list:
            self.assertEqual(len(call.args), 1)
            self.assertIsInstance(call.args[0], np.ndarray)
        self.assertEqual(dataset.targets, [1, 0])

    def test_every_metadata_field_verified_before_transform(self):
        for key, value in (('sample_id', 'wrong'), ('patient_id', '2'),
                           ('prediction_id', '1_R'), ('cancer', 0), ('cancer', '1')):
            for dataset_type in (ImageDataset, BreastDataset):
                with self.subTest(key=key, dataset=dataset_type.__name__):
                    backend = FakeBackend(self.records)
                    backend.samples[0][key] = value
                    transform = Mock(side_effect=self.transform)
                    dataset = dataset_type(backend, self.records, [0], transform)
                    with self.assertRaisesRegex(ValueError, key):
                        dataset[0]
                    transform.assert_not_called()
        del self.backend.samples[0]['patient_id']
        with self.assertRaisesRegex(ValueError, 'patient_id'):
            ImageDataset(self.backend, self.records, [0], self.transform)[0]

    def test_breast_grouping_cap_and_all_validation_views(self):
        train = BreastDataset(self.backend, self.records, range(6), self.transform,
                              max_views=2, training=True)
        valid = BreastDataset(self.backend, self.records, range(6), self.transform,
                              max_views=2, training=False)
        self.assertEqual(self.backend.calls, [])
        self.assertEqual(train.targets, [1, 0])
        self.assertEqual(len(train), 2)
        torch.manual_seed(7)
        capped = train[0]
        self.assertEqual(len(self.backend.calls), 2)
        torch.manual_seed(7)
        self.assertEqual(capped['sample_ids'], train[0]['sample_ids'])
        subsets = set()
        for seed in range(5):
            torch.manual_seed(seed)
            subsets.add(tuple(sorted(train[0]['sample_ids'])))
        self.assertGreater(len(subsets), 1)
        self.assertEqual(capped['images'].shape, (2, 3, 4, 4))
        all_views = valid[0]
        self.assertEqual(all_views['sample_ids'], [f'1_{i}' for i in range(5)])
        self.assertEqual(all_views['images'].shape, (5, 3, 4, 4))
        self.assertEqual(valid[1]['images'].shape, (1, 3, 4, 4))
        loader = DataLoader(valid, batch_size=2, collate_fn=collate_samples)
        self.assertEqual(next(iter(loader))['view_mask'].sum(1).tolist(), [5, 1])
        uncapped = BreastDataset(self.backend, self.records, range(5), self.transform, training=True)
        self.assertEqual(len(uncapped[0]['sample_ids']), 5)

    def test_subset_grouping_and_inconsistent_breast_rejected(self):
        subset = BreastDataset(self.backend, self.records, [1, 4, 5], self.transform)
        self.assertEqual(subset[0]['sample_ids'], ['1_1', '1_4'])
        for change in ({'label': 0}, {'patient_id': '2'}):
            records = [self.records[0], replace(self.records[1], **change)]
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'one patient and cancer'):
                BreastDataset(self.backend, records, [0, 1], self.transform)
        for cap in (0, -1, 1.5):
            with self.subTest(cap=cap), self.assertRaises(ValueError):
                BreastDataset(self.backend, self.records, [0], self.transform, max_views=cap)
        for indices in ([-1], [6], [0, 0], [1.5]):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                ImageDataset(self.backend, self.records, indices, self.transform)

    def test_variable_bag_collate_and_mask(self):
        dataset = BreastDataset(self.backend, self.records, range(6), self.transform)
        items = [dataset[1], dataset[0]]
        batch = collate_samples(items)
        self.assertEqual(set(batch), {'images', 'view_mask', 'targets', 'prediction_ids', 'sample_ids'})
        self.assertEqual(batch['images'].shape, (2, 5, 3, 4, 4))
        self.assertEqual(batch['view_mask'].dtype, torch.bool)
        self.assertEqual(batch['view_mask'].tolist(), [[True, False, False, False, False], [True] * 5])
        self.assertEqual(batch['targets'].dtype, torch.float32)
        self.assertEqual(batch['targets'].tolist(), [0, 1])
        self.assertEqual(batch['prediction_ids'], ['1_R', '1_L'])
        self.assertEqual(batch['sample_ids'], [item['sample_ids'] for item in items])
        self.assertTrue(torch.equal(batch['images'][0, 0], items[0]['images'][0]))
        self.assertEqual(batch['images'][0, 1:].count_nonzero(), 0)
        means = batch['images'].mean(dim=(2, 3, 4))
        pooled = (means * batch['view_mask']).sum(1) / batch['view_mask'].sum(1)
        torch.testing.assert_close(pooled, torch.stack([item['images'].mean() for item in items]))
        with self.assertRaises(ValueError):
            collate_samples([])
        with self.assertRaisesRegex(ValueError, 'at least one view'):
            collate_samples([{**items[0], 'images': torch.empty(0, 3, 4, 4), 'sample_ids': []}])
        with self.assertRaises(ValueError):
            collate_samples([items[0], {**items[1], 'images': torch.zeros(5, 3, 5, 4)}])


class ConverterRoundtripTests(unittest.TestCase):
    def test_real_converter_order_with_skips_and_no_source_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / 'images'
            images.mkdir()
            csv_path = root / 'source.csv'
            rows = [['1', str(i), 'L', 'CC' if i % 2 else 'MLO', '1'] for i in range(6)]
            with csv_path.open('w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['patient_id', 'image_id', 'laterality', 'view', 'cancer'])
                writer.writerows(rows)
            pixels = np.array([[0, 256], [4096, 65535]], dtype=np.uint16)
            for i in (0, 1, 3, 4):
                Image.fromarray(pixels).save(images / f'1_{i}.png')
            expected = rows[:5]
            random.Random(19).shuffle(expected)
            expected = [f'{row[0]}_{row[1]}' for row in expected if row[1] != '2']
            for storage in ('bytes', 'ndarray'):
                remote = root / storage
                args = build_parser().parse_args([
                    '--csv', str(csv_path), '--images-dir', str(images), '--out', str(remote),
                    '--workers', '1', '--prefetch', '1', '--limit', '5', '--seed', '19',
                    '--skip-missing', '--no-progress', '--image-storage', storage,
                ])
                with redirect_stderr(io.StringIO()):
                    convert(args)
            for path in images.iterdir():
                path.unlink()
            images.rmdir()
            for storage in ('bytes', 'ndarray'):
                with self.subTest(storage=storage):
                    remote = root / storage
                    manifest, records = load_training_records(remote)
                    self.assertEqual(manifest['skipped_missing_images'], 1)
                    self.assertEqual([record.sample_id for record in records], expected)
                    self.assertEqual([record.index for record in records], list(range(4)))
                    backend = RSNAStreamingDataset(remote=str(remote), local=str(root / f'cache_{storage}'),
                                                   decode_images=True, shuffle=False, batch_size=2)
                    dataset = ImageDataset(backend, records, range(4), ImageTransform(2))
                    for index, sample_id in enumerate(expected):
                        sample = dataset[index]
                        self.assertEqual(sample['sample_ids'], [sample_id])
                        torch.testing.assert_close(sample['images'][0], ImageTransform(2)(pixels))
                    bags = BreastDataset(backend, records, range(4), ImageTransform(2), max_views=1)
                    self.assertEqual(bags[0]['images'].shape, (4, 3, 2, 2))


if __name__ == '__main__':
    unittest.main()
