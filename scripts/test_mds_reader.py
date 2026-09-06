import contextlib
import csv
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

from scripts.benchmark_mds import validate_paths


class CachePathTests(unittest.TestCase):
    def test_rejects_shared_and_overlapping_cache(self):
        for remote, local in [('/Volumes/c/s/v/data', '/Volumes/c/s/v/cache'),
                              ('/data', '/dbfs/cache'), ('/data', '/data'),
                              ('/data', '/data/cache'), ('/data/remote', '/data'),
                              ('relative', '/tmp/cache'), ('/data', 'relative')]:
            with self.subTest(remote=remote, local=local):
                with self.assertRaises(ValueError):
                    validate_paths(remote, local)

    def test_accepts_separate_local_cache(self):
        remote, local = validate_paths('/Volumes/c/s/v/data', '/local_disk0/rsna_cache')
        self.assertEqual(remote, Path('/Volumes/c/s/v/data'))
        self.assertEqual(local, Path('/local_disk0/rsna_cache'))


@unittest.skipUnless(importlib.util.find_spec('streaming') and importlib.util.find_spec('PIL'),
                     'MDS reader tests require mosaicml-streaming and Pillow')
class ReaderTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        from PIL import Image
        from scripts.convert_to_mds import build_parser, convert

        self.np = np
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.images = self.root / 'images'
        self.images.mkdir()
        self.pixels = np.array([[0, 256], [4096, 65535]], dtype=np.uint16)
        for image_id in ('10', '11'):
            Image.fromarray(self.pixels).save(self.images / f'1_{image_id}.png')
        csv_path = self.root / 'train.csv'
        with csv_path.open('w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['patient_id', 'image_id', 'laterality', 'view', 'cancer'])
            writer.writerows([['1', '10', 'L', 'CC', '1'], ['1', '11', 'L', 'MLO', '1']])
        self.convert_args = ['--csv', str(csv_path), '--images-dir', str(self.images),
                             '--workers', '1', '--prefetch', '1']
        self.remote = self.root / 'mds'
        convert(build_parser().parse_args([*self.convert_args, '--out', str(self.remote)]))

    def dataset(self, name='cache', **kwargs):
        from scripts.benchmark_mds import RSNAStreamingDataset

        return RSNAStreamingDataset(remote=str(self.remote), local=str(self.root / name),
                                    batch_size=1, shuffle=False, **kwargs)

    def test_encoded_bytes_and_decoded_uint16(self):
        encoded = self.dataset(decode_images=False)
        sample = encoded[0]
        self.assertIsInstance(sample['image'], bytes)
        self.assertEqual(sample['cancer'], 1)
        self.assertEqual(sample['prediction_id'], '1_L')
        decoded = self.dataset(name='decoded')
        pixels = decoded[0]['image']
        self.np.testing.assert_array_equal(pixels, self.pixels)
        self.assertEqual(pixels.dtype, self.pixels.dtype)
        self.assertTrue((self.root / 'decoded' / 'shard.00000.mds').exists())

    def test_transform(self):
        dataset = self.dataset(transform=lambda pixels: pixels.astype('float32'))
        self.assertEqual(dataset[0]['image'].dtype, self.np.dtype('float32'))
        with self.assertRaises(ValueError):
            self.dataset(name='invalid', transform=lambda pixels: pixels, decode_images=False)

    def test_array_storage(self):
        from scripts.convert_to_mds import build_parser, convert

        self.remote = self.root / 'arrays'
        convert(build_parser().parse_args([*self.convert_args, '--out', str(self.remote),
                                           '--image-storage', 'ndarray']))
        dataset = self.dataset(decode_images=False)
        self.np.testing.assert_array_equal(dataset[0]['image'], self.pixels)

    def test_missing_success(self):
        (self.remote / '_SUCCESS').unlink()
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            self.dataset()

    def test_bad_index_checksum(self):
        index = self.remote / 'index.json'
        index.write_bytes(index.read_bytes() + b' ')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            self.dataset()

    def test_bad_completion_count(self):
        (self.remote / '_SUCCESS').write_text(json.dumps({'samples': 100}))
        with self.assertRaisesRegex(ValueError, 'counts'):
            self.dataset()

    def test_remote_hash_validation(self):
        shard = self.remote / 'shard.00000.mds'
        data = shard.read_bytes()
        shard.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
        dataset = self.dataset()
        with self.assertRaises(ValueError):
            dataset[0]

    def test_benchmark_single_and_multiworker(self):
        from scripts.benchmark_mds import benchmark, build_parser

        for workers in (0, 2):
            with self.subTest(workers=workers):
                args = build_parser().parse_args([
                    '--remote', str(self.remote), '--local', str(self.root / f'bench{workers}'),
                    '--num-workers', str(workers), '--batch-size', '1', '--epochs', '2',
                    '--max-batches', '2', '--cache-limit', '64mb', '--predownload', '2',
                    '--decode-images',
                ])
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    results = benchmark(args)
                self.assertEqual(len(results), 2)
                self.assertTrue(all(result['samples_observed'] == 2 for result in results))
                self.assertTrue(all(result['images_per_second'] > 0 for result in results))
                events = [json.loads(line) for line in output.getvalue().splitlines()]
                self.assertEqual(events[0]['event'], 'init')
                self.assertEqual(events[-1]['event'], 'epoch')


if __name__ == '__main__':
    unittest.main()
