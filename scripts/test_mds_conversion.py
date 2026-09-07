import csv
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from threading import Event
from unittest.mock import patch

from scripts.convert_to_mds import (
    ProgressTracker,
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


class ProgressTests(unittest.TestCase):
    def test_percentage_rate_eta_payload_and_idle_time(self):
        stream = io.StringIO()
        clock = unittest.mock.Mock(return_value=10.0)
        with ProgressTracker(interval=60, stream=stream, clock=clock) as progress:
            progress.set_stage('Converting images', total=4)
            self.assertIn('image ETA --', stream.getvalue())
            clock.return_value = 12.0
            progress.advance(1024 * 1024)
            progress.refresh()
            line = stream.getvalue().splitlines()[-1]
            for value in ['1/4 images', '25.0%', '0.5 images/s', 'image ETA 00:00:06',
                          'elapsed 00:00:02', 'payload 1.0 MiB', '0.5 MiB/s']:
                self.assertIn(value, line)
            clock.return_value = 16.0
            progress.refresh()
            line = stream.getvalue().splitlines()[-1]
            self.assertIn('image ETA 00:00:18', line)
            self.assertIn('idle 00:00:04', line)
            progress.set_stage('Finalizing shards')
            self.assertIn('image ETA --', stream.getvalue().splitlines()[-1])
        self.assertFalse(progress.thread.is_alive())

    def test_initial_small_run_and_final_updates(self):
        stream = io.StringIO()
        with ProgressTracker(interval=60, stream=stream) as progress:
            progress.set_stage('Converting images', total=1)
            progress.advance(10)
        text = stream.getvalue()
        self.assertIn('Validating inputs', text)
        self.assertIn('0/1 images', text)
        self.assertIn('1/1 images (100.0%)', text)
        self.assertIn('Complete', text.splitlines()[-1])
        self.assertNotIn('\r', text)
        self.assertNotIn('\x1b', text)

    def test_heartbeat_updates_without_samples_or_stage_changes(self):
        updated = Event()

        class Stream(io.StringIO):
            def write(self, text):
                result = super().write(text)
                if self.getvalue().count('Validating inputs') >= 2:
                    updated.set()
                return result

        with ProgressTracker(interval=0.01, stream=Stream()) as progress:
            self.assertTrue(updated.wait(2), 'No heartbeat while conversion is blocked')
            self.assertEqual(progress.count, 0)
        self.assertFalse(progress.thread.is_alive())

    def test_failure_and_interrupt_do_not_report_success(self):
        for error, label in [(ValueError('bad image'), 'Failed'),
                             (KeyboardInterrupt(), 'Interrupted')]:
            with self.subTest(label=label):
                stream = io.StringIO()
                with self.assertRaises(type(error)):
                    with ProgressTracker(interval=60, stream=stream) as progress:
                        progress.set_stage('Converting images', total=2)
                        progress.advance(10)
                        raise error
                self.assertIn(f'{label} during Converting images', stream.getvalue())
                self.assertIn('1/2 images', stream.getvalue())
                self.assertNotIn('Complete', stream.getvalue())
                self.assertFalse(progress.thread.is_alive())

    def test_auto_bar_for_tty_and_explicit_log_mode(self):
        for mode, bar in [('auto', True), ('log', False)]:
            with self.subTest(mode=mode):
                stream = io.StringIO()
                stream.isatty = lambda: True
                with ProgressTracker(interval=60, mode=mode, stream=stream) as progress:
                    progress.set_stage('Converting images', total=4)
                    progress.advance(10)
                    progress.refresh()
                self.assertEqual('\r' in stream.getvalue(), bar)
                self.assertTrue(stream.getvalue().endswith('\n'))

    def test_narrow_terminal_wraps_without_losing_metrics(self):
        stream = io.StringIO()
        with patch('scripts.convert_to_mds.shutil.get_terminal_size',
                   return_value=os.terminal_size((60, 24))):
            with ProgressTracker(interval=60, mode='bar', stream=stream) as progress:
                progress.set_stage('Converting images', total=4)
                stream.seek(0)
                stream.truncate()
                progress.advance(1024 * 1024)
                progress.refresh()
                frame = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', stream.getvalue())
                self.assertTrue(all(len(line) < 60 for line in frame.splitlines()))
                for value in ['1/4 images', 'image ETA', 'images/s', 'elapsed', 'payload', 'idle']:
                    self.assertIn(value, frame)

    def test_disabled_progress_has_no_output_or_thread(self):
        stream = io.StringIO()
        with ProgressTracker(mode='none', stream=stream) as progress:
            progress.set_stage('Converting images', total=2)
            progress.advance(10)
        self.assertEqual(stream.getvalue(), '')
        self.assertIsNone(progress.thread)

    def test_sample_milestones_remain_supported(self):
        stream = io.StringIO()
        with ProgressTracker(interval=60, every=2, stream=stream) as progress:
            progress.set_stage('Converting images', total=4)
            before = stream.getvalue()
            progress.advance(10)
            self.assertEqual(stream.getvalue(), before)
            progress.advance(10)
            self.assertIn('2/4 images', stream.getvalue())

    def test_broken_progress_stream_does_not_fail_work(self):
        stream = unittest.mock.Mock()
        stream.isatty.return_value = False
        stream.write.side_effect = BrokenPipeError('closed')
        with ProgressTracker(stream=stream) as progress:
            progress.set_stage('Converting images', total=1)
            progress.advance(10)

    def test_invalid_intervals_and_sample_milestones(self):
        for interval in [0, -1, float('nan'), float('inf')]:
            with self.subTest(interval=interval):
                with self.assertRaisesRegex(ValueError, 'progress-interval'):
                    ProgressTracker(interval=interval)
        with self.assertRaisesRegex(ValueError, 'progress-every'):
            ProgressTracker(every=0)

    def test_cli_progress_options(self):
        parser = build_parser()
        args = parser.parse_args(['--out', '/unused'])
        self.assertEqual(args.progress, 'auto')
        self.assertEqual(args.progress_interval, 1.0)
        args = parser.parse_args(['--out', '/unused', '--progress', 'log',
                                  '--progress-interval', '5', '--progress-every', '100'])
        self.assertEqual((args.progress, args.progress_interval, args.progress_every),
                         ('log', 5.0, 100))
        self.assertEqual(parser.parse_args(['--out', '/unused', '--no-progress']).progress, 'none')


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
        stream = io.StringIO()
        output = self.root / 'output'
        stages = []
        set_stage = ProgressTracker.set_stage

        def record_stage(progress, stage, **kwargs):
            stages.append(stage)
            if stage == 'Finalizing shards':
                self.assertFalse((output / '_SUCCESS').exists())
            return set_stage(progress, stage, **kwargs)

        with redirect_stderr(stream), patch.object(ProgressTracker, 'set_stage', record_stage):
            summary = convert(self.args())
        self.assertIn('Validating CSV', stages)
        self.assertIn('Finalizing shards', stages)
        self.assertIn('Verifying output', stages)
        self.assertIn('2/2 images (100.0%)', stream.getvalue())
        self.assertIn('Complete', stream.getvalue().splitlines()[-1])
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
        stream = io.StringIO()
        with redirect_stderr(stream):
            summary = convert(self.args('preview', '--dry-run', '--limit', '1'))
        self.assertEqual(summary['samples'], 1)
        self.assertEqual(summary['csv_rows'], 2)
        self.assertEqual(summary['images_checked'], 1)
        self.assertFalse((self.root / 'preview').exists())
        self.assertIn('1/1 images', stream.getvalue())
        self.assertIn('Dry run complete (no output written)', stream.getvalue())
        self.assertNotIn('Finalizing shards', stream.getvalue())

    def test_cli_keeps_stdout_json_and_can_disable_progress(self):
        for mode in ['log', 'none']:
            with self.subTest(mode=mode):
                args = self.args(f'cli-{mode}', '--dry-run')
                result = subprocess.run([
                    sys.executable, '-B', '-m', 'scripts.convert_to_mds',
                    '--csv', str(args.csv), '--images-dir', str(args.images_dir),
                    '--out', str(args.out), '--dry-run', '--progress', mode,
                ], text=True, capture_output=True, check=True)
                self.assertTrue(json.loads(result.stdout)['dry_run'])
                if mode == 'none':
                    self.assertEqual(result.stderr, '')
                else:
                    self.assertIn('Dry run complete', result.stderr)
                    self.assertNotIn('\r', result.stderr)

    def test_finalization_failure_never_reports_complete(self):
        from streaming import MDSWriter

        stream = io.StringIO()
        with redirect_stderr(stream), patch.object(MDSWriter, 'finish',
                                                   side_effect=OSError('flush failed')):
            with self.assertRaisesRegex(OSError, 'flush failed'):
                convert(self.args())
        self.assertIn('Failed during Finalizing shards', stream.getvalue())
        self.assertNotIn('Complete', stream.getvalue())
        self.assertFalse((self.root / 'output' / '_SUCCESS').exists())

    def test_heartbeat_during_writer_finalization(self):
        from streaming import MDSWriter

        heartbeat = Event()
        finish = MDSWriter.finish

        class Stream(io.StringIO):
            def write(self, text):
                result = super().write(text)
                if self.getvalue().count('Finalizing shards') >= 2:
                    heartbeat.set()
                return result

        def slow_finish(writer):
            self.assertTrue(heartbeat.wait(2), 'No heartbeat while the writer is finalizing')
            return finish(writer)

        with redirect_stderr(Stream()), patch.object(MDSWriter, 'finish', slow_finish):
            convert(self.args('output', '--progress-interval', '0.01'))
        self.assertTrue((self.root / 'output' / '_SUCCESS').exists())

    def test_failed_write_is_not_counted_as_processed(self):
        from streaming import MDSWriter

        stream = io.StringIO()
        with redirect_stderr(stream), patch.object(MDSWriter, 'write',
                                                   side_effect=OSError('write failed')):
            with self.assertRaisesRegex(OSError, 'write failed'):
                convert(self.args())
        self.assertIn('Failed during Converting images | 0/2 images', stream.getvalue())
        self.assertNotIn('Complete', stream.getvalue())
        self.assertFalse((self.root / 'output' / '_SUCCESS').exists())

    def test_verification_failure_never_reports_complete(self):
        stream = io.StringIO()
        with redirect_stderr(stream), patch('scripts.convert_to_mds.json.loads',
                                            return_value={'shards': [{'samples': 0}]}):
            with self.assertRaisesRegex(RuntimeError, 'sample count'):
                convert(self.args())
        self.assertIn('Failed during Verifying output', stream.getvalue())
        self.assertNotIn('Complete', stream.getvalue())
        self.assertFalse((self.root / 'output' / '_SUCCESS').exists())

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
                        ('--shard-size-mb', '4096'), ('--max-sample-mb', '0'),
                        ('--progress-every', '0'), ('--progress-interval', '0'),
                        ('--progress-interval', 'nan'), ('--progress-interval', 'inf'),
                        ('--progress-interval', '1e30')]:
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    convert(self.args('invalid', *options))
        self.assertFalse((self.root / 'invalid').exists())


if __name__ == '__main__':
    unittest.main()
