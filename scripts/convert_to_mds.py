import argparse
import csv
import hashlib
import io
import json
import random
import string
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from itertools import islice
from pathlib import Path


DEFAULT_SOURCE = Path('/Volumes/daai_ke_team/default/images/cancer_dataset')
COLUMNS = {
    'image': 'bytes',
    'sample_id': 'str',
    'patient_id': 'str',
    'image_id': 'str',
    'prediction_id': 'str',
    'laterality': 'str',
    'view': 'str',
    'cancer': 'int',
    'metadata': 'json',
}


@dataclass(frozen=True)
class Record:
    path: Path
    metadata: dict[str, str]
    prediction_id: str


def load_records(csv_path: Path, images_dir: Path, image_pattern: str) -> tuple[list[Record], str]:
    content = csv_path.read_bytes()
    reader = csv.DictReader(io.StringIO(content.decode('utf-8-sig'), newline=''))
    fields = reader.fieldnames or []
    required = {'patient_id', 'image_id', 'laterality', 'view', 'cancer'}
    if len(fields) != len(set(fields)) or not required.issubset(fields):
        raise ValueError(f'CSV needs unique columns including {sorted(required)}.')
    for _, field, format_spec, conversion in string.Formatter().parse(image_pattern):
        if field is not None and (field not in fields or format_spec or conversion):
            raise ValueError('Image pattern may only contain plain CSV column placeholders.')
    records = []
    image_keys = set()
    image_paths = set()
    breast_labels = {}
    breast_ids = {}
    id_breasts = {}
    for row_number, row in enumerate(reader, start=2):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f'Malformed CSV row {row_number}.')
        for key in ('patient_id', 'image_id'):
            if not row[key].isascii() or not row[key].isdecimal():
                raise ValueError(f'Row {row_number}: {key} must be a nonnegative integer string.')
        if row['cancer'] not in ('0', '1') or row['laterality'] not in ('L', 'R'):
            raise ValueError(f'Row {row_number}: cancer must be 0/1 and laterality must be L/R.')
        if not row['view'].strip():
            raise ValueError(f'Row {row_number}: view must not be empty.')
        image_key = (row['patient_id'], row['image_id'])
        breast = (row['patient_id'], row['laterality'])
        prediction_id = row.get('prediction_id', f'{breast[0]}_{breast[1]}')
        if not prediction_id.strip():
            raise ValueError(f'Row {row_number}: prediction_id must not be empty.')
        if breast in breast_labels and breast_labels[breast] != row['cancer']:
            raise ValueError(f'Row {row_number}: conflicting labels for one breast.')
        if breast in breast_ids and breast_ids[breast] != prediction_id:
            raise ValueError(f'Row {row_number}: a breast maps to multiple prediction IDs.')
        if prediction_id in id_breasts and id_breasts[prediction_id] != breast:
            raise ValueError(f'Row {row_number}: a prediction ID maps to multiple breasts.')
        breast_labels[breast] = row['cancer']
        breast_ids[breast] = prediction_id
        id_breasts[prediction_id] = breast
        relative = Path(image_pattern.format_map(row))
        if relative.is_absolute() or '..' in relative.parts or relative == Path('.'):
            raise ValueError(f'Row {row_number}: image pattern must stay inside the images directory.')
        path = images_dir / relative
        if image_key in image_keys or path in image_paths:
            raise ValueError(f'Row {row_number}: duplicate image ID or image path.')
        image_keys.add(image_key)
        image_paths.add(path)
        records.append(Record(path, row, prediction_id))
    if not records:
        raise ValueError('CSV contains no image rows.')
    return records, hashlib.sha256(content).hexdigest()


def load_sample(record: Record, image_storage: str, max_sample_bytes: int) -> dict:
    from PIL import Image

    with record.path.open('rb') as file:
        data = file.read(max_sample_bytes + 1)
    if not data or len(data) > max_sample_bytes:
        raise ValueError(f'Empty or oversized image: {record.path}')
    with Image.open(io.BytesIO(data)) as image:
        if image.format not in ('PNG', 'JPEG', 'TIFF') or getattr(image, 'n_frames', 1) != 1:
            raise ValueError(f'Expected a single-frame PNG, JPEG, or TIFF: {record.path}')
        if image_storage == 'bytes':
            image.verify()
            payload = data
        elif image_storage == 'ndarray':
            import numpy as np

            if image.mode not in ('L', 'RGB', 'RGBA', 'I', 'I;16', 'I;16B', 'I;16L'):
                raise ValueError(f'Unsupported array image mode {image.mode}: {record.path}')
            if image.width * image.height * max(len(image.getbands()), 4) > max_sample_bytes:
                raise ValueError(f'Decoded image may exceed --max-sample-mb: {record.path}')
            payload = np.array(image)
            payload = payload.astype(payload.dtype.newbyteorder('='), copy=False)
            if payload.nbytes > max_sample_bytes:
                raise ValueError(f'Decoded image exceeds --max-sample-mb: {record.path}')
        else:
            raise ValueError("Image storage must be 'bytes' or 'ndarray'.")
    row = record.metadata
    return {
        'image': payload,
        'sample_id': f"{row['patient_id']}_{row['image_id']}",
        'patient_id': row['patient_id'],
        'image_id': row['image_id'],
        'prediction_id': record.prediction_id,
        'laterality': row['laterality'],
        'view': row['view'],
        'cancer': int(row['cancer']),
        'metadata': dict(row),
    }


def ordered_parallel_map(function, items, workers: int, prefetch: int):
    if workers < 1 or prefetch < workers:
        raise ValueError('Workers must be positive and prefetch must be at least workers.')
    iterator = iter(items)
    executor = ThreadPoolExecutor(max_workers=workers)
    pending = deque()
    try:
        pending.extend(executor.submit(function, item) for item in islice(iterator, prefetch))
        while pending:
            yield pending.popleft().result()
            next_item = next(iterator, None)
            if next_item is not None:
                pending.append(executor.submit(function, next_item))
    finally:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)


def validate_output(output: Path, images_dir: Path) -> None:
    if output.exists():
        raise FileExistsError(f'Output already exists; choose a new version directory: {output}')
    if not output.parent.is_dir():
        raise ValueError(f'Output parent must already exist: {output.parent}')
    if output == images_dir or output in images_dir.parents or images_dir in output.parents:
        raise ValueError('Output must be separate from the source images directory.')


def convert(args: argparse.Namespace) -> dict:
    if args.workers < 1 or args.prefetch < args.workers:
        raise ValueError('--workers must be positive and --prefetch must be at least --workers.')
    if not 1 <= args.shard_size_mb < 4096 or args.max_sample_mb < 1:
        raise ValueError('--shard-size-mb must be 1..4095 and --max-sample-mb must be positive.')
    if args.limit is not None and args.limit < 1:
        raise ValueError('--limit must be positive.')
    if args.progress_every < 1:
        raise ValueError('--progress-every must be positive.')
    images_dir = args.images_dir.resolve()
    csv_path = args.csv.resolve()
    output = args.out.resolve()
    if not images_dir.is_dir():
        raise ValueError(f'Images directory not found: {images_dir}')
    validate_output(output, images_dir)
    records, csv_sha256 = load_records(csv_path, images_dir, args.image_pattern)
    csv_rows = len(records)
    records = records[:args.limit]
    if not args.no_shuffle:
        random.Random(args.seed).shuffle(records)
    summary = {
        'schema_version': 1,
        'csv': str(csv_path),
        'csv_sha256': csv_sha256,
        'images_dir': str(images_dir),
        'image_pattern': args.image_pattern,
        'image_storage': args.image_storage,
        'csv_rows': csv_rows,
        'samples': len(records),
        'patients': len({record.metadata['patient_id'] for record in records}),
        'breasts': len({record.prediction_id for record in records}),
        'positive_images': sum(record.metadata['cancer'] == '1' for record in records),
        'shard_size_mb': args.shard_size_mb,
        'compression': None if args.compression == 'none' else args.compression,
        'shuffle_seed': None if args.no_shuffle else args.seed,
        'limit': args.limit,
        'output': str(output),
    }
    max_sample_bytes = args.max_sample_mb * 1024 * 1024
    if args.dry_run:
        for record in records[:8]:
            load_sample(record, args.image_storage, max_sample_bytes)
        return {**summary, 'dry_run': True, 'images_checked': min(8, len(records))}

    from streaming import MDSWriter

    output.mkdir(exist_ok=False)
    columns = {**COLUMNS, 'image': args.image_storage}
    start = time.perf_counter()
    count = 0
    payload_bytes = 0
    samples = ordered_parallel_map(
        lambda record: load_sample(record, args.image_storage, max_sample_bytes),
        records, args.workers, args.prefetch,
    )
    try:
        with MDSWriter(out=str(output), columns=columns,
                       size_limit=args.shard_size_mb * 1024 * 1024,
                       compression=summary['compression'], hashes=['sha256']) as writer:
            for sample in samples:
                writer.write(sample)
                payload = sample['image']
                payload_bytes += len(payload) if isinstance(payload, bytes) else payload.nbytes
                count += 1
                if count % args.progress_every == 0:
                    elapsed = time.perf_counter() - start
                    print(f'{count}/{len(records)} images; {count / elapsed:.1f} images/s',
                          file=sys.stderr, flush=True)
    finally:
        samples.close()
    index_path = output / 'index.json'
    index = json.loads(index_path.read_text(encoding='utf-8'))
    indexed_samples = sum(shard['samples'] for shard in index['shards'])
    if count != len(records) or indexed_samples != count:
        raise RuntimeError('Written/indexed sample count does not match the selected CSV rows.')
    elapsed = time.perf_counter() - start
    summary.update({
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'mosaicml_streaming_version': version('mosaicml-streaming'),
        'pillow_version': version('Pillow'),
        'shards': len(index['shards']),
        'image_payload_bytes': payload_bytes,
        'elapsed_seconds': elapsed,
        'images_per_second': count / elapsed,
        'index_sha256': hashlib.sha256(index_path.read_bytes()).hexdigest(),
    })
    with (output / 'conversion.json').open('x', encoding='utf-8') as file:
        json.dump(summary, file, indent=2)
        file.write('\n')
    with (output / '_SUCCESS').open('x', encoding='utf-8') as file:
        json.dump({'samples': count}, file)
        file.write('\n')
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Pack CSV-indexed mammogram images into immutable MosaicML MDS shards.'
    )
    parser.add_argument('--csv', type=Path, default=DEFAULT_SOURCE / 'train 3.csv')
    parser.add_argument('--images-dir', type=Path, default=DEFAULT_SOURCE / 'images')
    parser.add_argument('--image-pattern', default='{patient_id}_{image_id}.png')
    parser.add_argument('--out', type=Path, required=True,
                        help='New output directory under an existing mounted Volume/local parent.')
    parser.add_argument('--image-storage', choices=('bytes', 'ndarray'), default='bytes',
                        help='Original encoded bytes, or decoded arrays with dtype/shape preserved.')
    parser.add_argument('--shard-size-mb', type=int, default=256)
    parser.add_argument('--compression', choices=('none', 'zstd:1', 'zstd:3'), default='none')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--prefetch', type=int, default=16,
                        help='Maximum queued/in-flight image reads, at least the worker count.')
    parser.add_argument('--max-sample-mb', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-shuffle', action='store_true')
    parser.add_argument('--limit', type=int, help='Convert only the first N CSV rows, for smoke tests.')
    parser.add_argument('--progress-every', type=int, default=1000)
    parser.add_argument('--dry-run', action='store_true',
                        help='Validate CSV and up to eight selected images; write nothing.')
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        summary = convert(args)
    except ImportError as error:
        parser.exit(1, f'Missing/incompatible dependency: {error}. Use the Databricks ML Python '
                       'environment with mosaicml-streaming, Pillow and NumPy installed.\n')
    except (OSError, ValueError, csv.Error, RuntimeError) as error:
        parser.exit(1, f'Conversion failed: {error}\nAny partial output is incomplete unless '
                       '_SUCCESS exists. Nothing is overwritten; retry with a new output path.\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
