import csv
import importlib.util
import json
import tempfile
import time
import unittest
from pathlib import Path

from scripts.convert_to_mds import (
    build_parser,
    convert,
    load_records,
    load_sample,
    ordered_parallel_map,
    validate_output,
)


class CSVTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.csv_path = self.root / 'train.csv'
        self.header = ['patient_id', 'image_id', 'laterality', 'view', 'cancer', 'age']
        self.rows = [['1', '10', 'L', 'CC', '1', ''], ['1', '11', 'L', 'MLO', '1', '60'],
                     ['1', '12', 'R', 'CC', '0', '60']]
        self.write_csv(self.rows)

    def write_csv(self, rows, header=None):
        with self.csv_path.open('w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header or self.header)
            writer.writerows(rows)

    def load(self, pattern='{patient_id}_{image_id}.png'):
        return load_records(self.csv_path, self.root, pattern)

    def test_paths_labels_and_missing_metadata_preserved(self):
        records, checksum = self.load()
        self.assertEqual(len(records), 3)
        self.assertEqual(len(checksum), 64)
        self.assertEqual(records[0].path, self.root / '1_10.png')
        self.assertEqual(records[0].metadata['age'], '')
        self.assertEqual(records[0].prediction_id, records[1].prediction_id)
        self.assertNotEqual(records[0].prediction_id, records[2].prediction_id)

    def test_duplicate_rows(self):
        self.write_csv([self.rows[0], self.rows[0]])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.load()

    def test_conflicting_breast_labels(self):
        self.rows[1][4] = '0'
        self.write_csv(self.rows)
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            self.load()

    def test_bad_metadata(self):
        for column, value in [(0, '../1'), (1, '-1'), (2, 'X'), (3, ''), (4, '0.5')]:
            with self.subTest(column=column):
                row = list(self.rows[0])
                row[column] = value
                self.write_csv([row])
                with self.assertRaises(ValueError):
                    self.load()

    def test_bad_headers_empty_and_malformed_rows(self):
        for rows, header in [([], self.header), (self.rows, ['x'] * 6),
                             ([self.rows[0][:-1]], self.header),
                             ([self.rows[0] + ['extra']], self.header)]:
            with self.subTest(rows=rows):
                self.write_csv(rows, header)
                with self.assertRaises(ValueError):
                    self.load()

    def test_bad_patterns(self):
        for pattern in ['../{image_id}.png', '/{image_id}.png', '{missing}.png',
                        '{image_id.__class__}.png', '{image_id!r}.png', 'same.png']:
            with self.subTest(pattern=pattern):
                with self.assertRaises(ValueError):
                    self.load(pattern)

    def test_existing_prediction_ids_preserved(self):
        self.write_csv([self.rows[0] + ['1-L'], self.rows[1] + ['1-L']],
                       self.header + ['prediction_id'])
        self.assertEqual(self.load()[0][0].prediction_id, '1-L')

    def test_prediction_ids_cannot_merge_breasts(self):
        self.write_csv([self.rows[0] + ['same'], self.rows[2] + ['same']],
                       self.header + ['prediction_id'])
        with self.assertRaisesRegex(ValueError, 'multiple breasts'):
            self.load()

    def test_output_never_overwrites(self):
        with self.assertRaises(FileExistsError):
            validate_output(self.root, self.root / 'images')
        with self.assertRaises(ValueError):
            validate_output(self.root / 'missing' / 'output', self.root / 'images')
        images = self.root / 'images'
        images.mkdir()
        with self.assertRaisesRegex(ValueError, 'separate'):
            validate_output(images / 'output', images)

    def test_parallel_map_is_ordered_and_propagates_failure(self):
        def delayed(value):
            time.sleep((4 - value) * 0.001)
            return value

        self.assertEqual(list(ordered_parallel_map(delayed, range(5), 2, 3)), list(range(5)))
        with self.assertRaises(ZeroDivisionError):
            list(ordered_parallel_map(lambda value: 1 / value, [1, 0, 2], 2, 3))
        with self.assertRaises(ValueError):
            list(ordered_parallel_map(delayed, range(5), 2, 1))


@unittest.skipUnless(importlib.util.find_spec('streaming') and importlib.util.find_spec('PIL'),
                     'MDS integration tests require mosaicml-streaming and Pillow')
class ConversionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        from PIL import Image
        from streaming import LocalDataset

        cls.np = np
        cls.Image = Image
        cls.LocalDataset = LocalDataset

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.images = self.root / 'images'
        self.images.mkdir()
        self.csv_path = self.root / 'train.csv'
        self.pixels = self.np.array([[0, 256], [1024, 65535]], dtype=self.np.uint16)
        self.Image.fromarray(self.pixels).save(self.images / '1_10.png')
        self.Image.fromarray(self.pixels).save(self.images / '1_11.png')
        with self.csv_path.open('w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['patient_id', 'image_id', 'laterality', 'view', 'cancer', 'age'])
            writer.writerows([['1', '10', 'L', 'CC', '1', ''], ['1', '11', 'L', 'MLO', '1', '60']])

    def args(self, name='output', *options):
        return build_parser().parse_args([
            '--csv', str(self.csv_path), '--images-dir', str(self.images),
            '--out', str(self.root / name), '--workers', '2', '--prefetch', '2',
            '--shard-size-mb', '1', *options,
        ])

    def test_bytes_roundtrip_and_success_manifest(self):
        summary = convert(self.args())
        output = self.root / 'output'
        dataset = self.LocalDataset(local=str(output))
        self.assertEqual(len(dataset), 2)
        for index in range(2):
            sample = dataset[index]
            self.assertEqual(sample['image'], (self.images / f"{sample['sample_id']}.png").read_bytes())
            self.assertEqual(sample['cancer'], 1)
            self.assertEqual(sample['prediction_id'], '1_L')
        manifest = json.loads((output / 'conversion.json').read_text())
        self.assertEqual(manifest, summary)
        self.assertEqual(json.loads((output / '_SUCCESS').read_text()), {'samples': 2})
        self.assertEqual(summary['breasts'], 1)
        self.assertEqual(summary['patients'], 1)
        self.assertIn('sha256', dataset.shards[0].raw_data.hashes)

    def test_uint16_array_roundtrip_with_compression(self):
        summary = convert(self.args('array', '--image-storage', 'ndarray', '--compression', 'zstd:1'))
        from streaming import StreamingDataset

        dataset = StreamingDataset(remote=str(self.root / 'array'), local=str(self.root / 'cache'),
                                   shuffle=False, batch_size=1, validate_hash='sha256')
        try:
            pixels = dataset[0]['image']
            self.np.testing.assert_array_equal(pixels, self.pixels)
            self.assertEqual(pixels.dtype, self.pixels.dtype)
            self.assertEqual(summary['compression'], 'zstd:1')
        finally:
            del dataset

    def test_big_endian_pixels_are_not_byte_swapped_by_mds(self):
        pixels = self.pixels.astype('>u2')
        for image_id in ('10', '11'):
            self.Image.fromarray(pixels).save(self.images / f'1_{image_id}.tiff')
        convert(self.args('endian', '--image-pattern', '{patient_id}_{image_id}.tiff',
                          '--image-storage', 'ndarray'))
        dataset = self.LocalDataset(local=str(self.root / 'endian'))
        self.np.testing.assert_array_equal(dataset[0]['image'], self.pixels)

    def test_dry_run_and_limit(self):
        summary = convert(self.args('preview', '--dry-run', '--limit', '1'))
        self.assertEqual(summary['samples'], 1)
        self.assertEqual(summary['csv_rows'], 2)
        self.assertEqual(summary['images_checked'], 1)
        self.assertFalse((self.root / 'preview').exists())

    def test_existing_output_untouched(self):
        output = self.root / 'output'
        output.mkdir()
        sentinel = output / 'keep.txt'
        sentinel.write_text('untouched')
        with self.assertRaises(FileExistsError):
            convert(self.args())
        self.assertEqual(sentinel.read_text(), 'untouched')

    def test_missing_and_corrupt_inputs_leave_no_success(self):
        (self.images / '1_10.png').unlink()
        with self.assertRaises(FileNotFoundError):
            convert(self.args())
        self.assertFalse((self.root / 'output' / '_SUCCESS').exists())
        (self.images / '1_10.png').write_bytes(b'not an image')
        with self.assertRaises(OSError):
            convert(self.args('corrupt'))
        self.assertFalse((self.root / 'corrupt' / '_SUCCESS').exists())

    def test_deterministic_shard_bytes(self):
        convert(self.args('first'))
        convert(self.args('second'))
        for name in ['index.json', 'shard.00000.mds']:
            self.assertEqual((self.root / 'first' / name).read_bytes(),
                             (self.root / 'second' / name).read_bytes())

    def test_multiple_shards_keep_all_rows(self):
        rng = self.np.random.default_rng(42)
        rows = []
        for image_id in range(20, 28):
            pixels = rng.integers(0, 256, (512, 512), dtype=self.np.uint8)
            self.Image.fromarray(pixels).save(self.images / f'2_{image_id}.png')
            rows.append(['2', str(image_id), 'R', 'CC', '0', '55'])
        with self.csv_path.open('w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['patient_id', 'image_id', 'laterality', 'view', 'cancer', 'age'])
            writer.writerows(rows)
        summary = convert(self.args())
        dataset = self.LocalDataset(local=str(self.root / 'output'))
        self.assertGreater(summary['shards'], 1)
        self.assertEqual(len(dataset), 8)
        self.assertEqual({dataset[i]['image_id'] for i in range(8)}, {str(i) for i in range(20, 28)})

    def test_size_limit_and_invalid_arguments(self):
        records, _ = load_records(self.csv_path, self.images, '{patient_id}_{image_id}.png')
        with self.assertRaisesRegex(ValueError, 'oversized'):
            load_sample(records[0], 'bytes', 1)
        for options in [('--workers', '0'), ('--prefetch', '1'), ('--limit', '0'),
                        ('--shard-size-mb', '4096'), ('--max-sample-mb', '0')]:
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    convert(self.args('invalid', *options))
        self.assertFalse((self.root / 'invalid').exists())


if __name__ == '__main__':
    unittest.main()
