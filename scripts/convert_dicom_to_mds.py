"""Convert the original competition DICOMs into MDS shards without losing detail.

The existing PNG cache was exported at 1024x1024 8-bit, which downsamples the long
side by a median of 3.25x, squashes a portrait image into a square, and collapses
~2500 native grey levels into 256. This converter goes back to the DICOMs and does
the lossy steps in the right order: window and invert from the DICOM tags, detect
the breast at native resolution, crop, and only then resize.

Pixels are stored in their native 12-bit range inside uint16 PNG, not rescaled to
65535: identical information, but the constant high byte compresses ~25% better.
Train with --intensity-max 4095 so the scale is interpreted correctly.

Reading happens directly from the competition zip, so the 54,706 DICOMs are never
unpacked onto the Volume.
"""

import argparse
import hashlib
import json
import multiprocessing as mp
import random
import sys
import zipfile
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.convert_to_mds import COLUMNS, ProgressTracker, load_records
from scripts.training_data import expand_box, roi_bounding_box

DEFAULT_SOURCE = Path('/Volumes/daai_ke_team/default/images/cancer_dataset')
STORED_MAX = 4095  # 12-bit range; pass --intensity-max 4095 when training.
_ARCHIVE = {}


def _archive(path: str) -> zipfile.ZipFile:
    """One ZipFile handle per worker process; reopening per image is far slower."""
    if path not in _ARCHIVE:
        _ARCHIVE[path] = zipfile.ZipFile(path)
    return _ARCHIVE[path]


def decode_dicom(data: bytes) -> tuple[np.ndarray, dict]:
    """Decode to a float array in [0, 1] with the breast bright, plus provenance."""
    import pydicom

    dataset = pydicom.dcmread(zipfile.io.BytesIO(data))
    pixels = dataset.pixel_array
    if pixels.ndim != 2:
        raise ValueError(f'Expected a single-frame grayscale image, got shape {pixels.shape}.')
    image = pixels.astype(np.float32)
    photometric = str(getattr(dataset, 'PhotometricInterpretation', 'MONOCHROME2'))
    if photometric == 'MONOCHROME1':
        image = float(image.max()) - image
    return image, {
        'photometric': photometric,
        'bits_stored': int(getattr(dataset, 'BitsStored', 0)),
        'native_rows': int(dataset.Rows),
        'native_columns': int(dataset.Columns),
        'transfer_syntax': str(getattr(dataset.file_meta, 'TransferSyntaxUID', '')),
    }


def window(image: np.ndarray, low: float, high: float) -> np.ndarray:
    """Percentile min-max scaling.

    Preferred over the DICOM WindowCenter/Width tags because those are frequently
    multi-valued or defaulted to the full range here, and percentile scaling is
    robust to the very bright labels and burned-in markers some vendors include.
    """
    lower, upper = np.percentile(image, [low, high])
    if not np.isfinite([lower, upper]).all() or upper <= lower:
        lower, upper = float(image.min()), float(image.max())
    if upper <= lower:
        raise ValueError('Image has no intensity range after windowing.')
    return np.clip((image - lower) / (upper - lower), 0.0, 1.0)


def preprocess(image: np.ndarray, target: int, margin: float, canonical_side: str | None,
               compress_level: int) -> tuple[bytes, dict]:
    from PIL import Image

    height, width = image.shape
    box = roi_bounding_box((image * STORED_MAX).astype(np.uint16))
    # Detect the chest wall on the full frame: after cropping, the box fills the
    # frame and the side is no longer recoverable.
    flipped = False
    if canonical_side is not None:
        chest = 'left' if box[0] <= width - box[2] else 'right'
        flipped = chest != canonical_side
    x0, y0, x1, y1 = expand_box(box, width, height, margin)
    crop = image[y0:y1, x0:x1]
    if flipped:
        crop = crop[:, ::-1]
    scale = target / max(crop.shape)
    size = (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale)))
    stored = Image.fromarray((crop * STORED_MAX).astype(np.uint16))
    if scale < 1.0:
        stored = stored.resize(size, Image.BILINEAR)
    buffer = zipfile.io.BytesIO()
    stored.save(buffer, format='PNG', compress_level=compress_level)
    return buffer.getvalue(), {
        'roi_box': [int(v) for v in box], 'crop_box': [int(x0), int(y0), int(x1), int(y1)],
        'canonical_flip': flipped, 'stored_size': list(stored.size),
        'roi_area_fraction': round((box[2] - box[0]) * (box[3] - box[1]) / (width * height), 4),
    }


def _convert_one(job):
    archive_path, member, row, prediction_id, options = job
    try:
        with _archive(archive_path).open(member) as handle:
            data = handle.read()
        image, provenance = decode_dicom(data)
        payload, geometry = preprocess(window(image, *options['percentiles']), options['target'],
                                       options['margin'], options['canonical_side'],
                                       options['compress_level'])
    except Exception as error:  # noqa: BLE001 - reported per image, never silently dropped
        return {'error': f'{type(error).__name__}: {error}', 'member': member}
    return {
        'image': payload,
        'sample_id': f"{row['patient_id']}_{row['image_id']}",
        'patient_id': row['patient_id'],
        'image_id': row['image_id'],
        'prediction_id': prediction_id,
        'laterality': row['laterality'],
        'view': row['view'],
        'cancer': int(row['cancer']),
        'metadata': {**row, **provenance, **geometry},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--zip', type=Path, required=True, help='Competition archive.')
    parser.add_argument('--csv', type=Path, default=DEFAULT_SOURCE / 'train 3.csv')
    parser.add_argument('--out', type=Path, required=True, help='New MDS directory.')
    parser.add_argument('--member-pattern', default='train_images/{patient_id}/{image_id}.dcm')
    parser.add_argument('--target-long-side', type=int, default=1536)
    parser.add_argument('--roi-margin', type=float, default=0.05)
    parser.add_argument('--canonical-side', choices=('none', 'left', 'right'), default='left',
                        help='Flip so the chest wall lands on this side, decided at native '
                             'resolution. Train with --canonical-side none afterwards.')
    parser.add_argument('--percentiles', type=float, nargs=2, default=(1.0, 99.9),
                        metavar=('LOW', 'HIGH'))
    parser.add_argument('--compress-level', type=int, default=6, choices=range(10))
    parser.add_argument('--shard-size-mb', type=int, default=256)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-shuffle', action='store_true')
    parser.add_argument('--limit', type=int, help='Convert only the first N CSV rows.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Validate inputs and convert eight images in memory; write nothing.')
    parser.add_argument('--progress', choices=('auto', 'bar', 'log', 'none'), default='auto')
    parser.add_argument('--no-progress', action='store_const', const='none', dest='progress')
    return parser


def convert(args, progress) -> dict:
    output = args.out.resolve()
    if not args.dry_run:
        if output.exists():
            raise FileExistsError(f'Output already exists; choose a new directory: {output}')
        if not output.parent.is_dir():
            raise ValueError(f'Output parent must already exist: {output.parent}')
    if args.target_long_side < 64:
        raise ValueError('--target-long-side must be at least 64.')
    if args.workers < 1:
        raise ValueError('--workers must be positive.')
    if not 0 <= args.percentiles[0] < args.percentiles[1] <= 100:
        raise ValueError('--percentiles must satisfy 0 <= LOW < HIGH <= 100.')

    progress.set_stage('Validating CSV')
    # images_dir/image_pattern are recorded so load_training_records can rebuild the
    # same records; the DICOMs live in the archive and are never written to disk.
    records, checksum = load_records(args.csv, args.zip.resolve().parent,
                                     args.member_pattern.split('/', 1)[1])
    csv_rows = len(records)
    records = records[:args.limit] if args.limit else records
    if not args.no_shuffle:
        random.Random(args.seed).shuffle(records)

    progress.set_stage('Checking archive members', total=len(records))
    with zipfile.ZipFile(args.zip) as archive:
        available = set(archive.namelist())
    members, missing = [], []
    for record in records:
        member = args.member_pattern.format_map(record.metadata)
        (members if member in available else missing).append(member)
        progress.advance(0)
    if missing:
        raise FileNotFoundError(f'Missing {len(missing)} of {len(records)} archive members, '
                                f'starting with {missing[0]}.')

    canonical = None if args.canonical_side == 'none' else args.canonical_side
    options = {'target': args.target_long_side, 'margin': args.roi_margin,
               'canonical_side': canonical, 'percentiles': tuple(args.percentiles),
               'compress_level': args.compress_level}
    jobs = [(str(args.zip), member, record.metadata, record.prediction_id, options)
            for member, record in zip(members, records)]

    summary = {
        'schema_version': 1,
        'image_storage': 'bytes',
        'source': 'dicom',
        'archive': str(args.zip),
        'csv': str(args.csv),
        'csv_sha256': checksum,
        'images_dir': str(args.zip.resolve().parent),
        'image_pattern': args.member_pattern.split('/', 1)[1],
        'member_pattern': args.member_pattern,
        'csv_rows': csv_rows,
        'limit': args.limit,
        'shuffle_seed': None if args.no_shuffle else args.seed,
        'selected_samples': len(records),
        'source_files_checked': len(records),
        'missing_policy': 'error',
        'skipped_missing_images': 0,
        'skipped_images': [],
        'samples': len(records),
        'patients': len({r.metadata['patient_id'] for r in records}),
        'breasts': len({r.prediction_id for r in records}),
        'positive_images': sum(int(r.metadata['cancer']) for r in records),
        'compression': 'none',
        'target_long_side': args.target_long_side,
        'stored_max': STORED_MAX,
        'stored_dtype': 'uint16',
        'intensity_max_for_training': STORED_MAX,
        'roi_margin': args.roi_margin,
        'canonical_side': canonical,
        'window_percentiles': list(args.percentiles),
        'pydicom_version': version('pydicom'),
    }

    if args.dry_run:
        progress.completion = 'Dry run complete (no output written)'
        progress.set_stage('Checking dry-run images', total=min(8, len(jobs)))
        for job in jobs[:8]:
            sample = _convert_one(job)
            if 'error' in sample:
                raise ValueError(f"{sample['member']}: {sample['error']}")
            progress.advance(len(sample['image']))
        return {**summary, 'dry_run': True, 'images_checked': min(8, len(jobs))}

    progress.set_stage('Loading MDS dependencies')
    from streaming import MDSWriter

    progress.set_stage('Initializing MDS writer')
    output.mkdir(exist_ok=False)
    count = payload_bytes = 0
    started = datetime.now(timezone.utc)
    # imap, not imap_unordered: load_training_records rebuilds records in CSV order
    # after the same shuffle, and matches them to shard positions by index.
    with mp.get_context('spawn').Pool(args.workers) as pool, \
            MDSWriter(out=str(output), columns=COLUMNS,
                      size_limit=args.shard_size_mb * 1024 * 1024,
                      compression=None, hashes=['sha256']) as writer:
        progress.set_stage('Converting images', total=len(jobs))
        for sample in pool.imap(_convert_one, jobs, chunksize=4):
            if 'error' in sample:
                raise ValueError(f"{sample['member']}: {sample['error']}")
            writer.write(sample)
            payload_bytes += len(sample['image'])
            count += 1
            progress.advance(len(sample['image']))
        progress.set_stage('Finalizing shards')

    progress.set_stage('Verifying output')
    index_path = output / 'index.json'
    index = json.loads(index_path.read_text(encoding='utf-8'))
    if count != len(jobs) or sum(s['samples'] for s in index['shards']) != count:
        raise RuntimeError('Written/indexed sample count does not match the selected CSV rows.')
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    summary.update({
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'mosaicml_streaming_version': version('mosaicml-streaming'),
        'pillow_version': version('Pillow'),
        'shards': len(index['shards']),
        'image_payload_bytes': payload_bytes,
        'elapsed_seconds': elapsed,
        'images_per_second': count / elapsed if elapsed else 0.0,
        'index_sha256': hashlib.sha256(index_path.read_bytes()).hexdigest(),
    })
    progress.set_stage(f'Publishing manifest ({len(index["shards"])} shards)')
    with (output / 'conversion.json').open('x', encoding='utf-8') as file:
        json.dump(summary, file, indent=2)
        file.write('\n')
    with (output / '_SUCCESS').open('x', encoding='utf-8') as file:
        json.dump({'samples': count}, file)
        file.write('\n')
    return summary


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        with ProgressTracker(mode=args.progress) as progress:
            summary = convert(args, progress)
    except KeyboardInterrupt:
        parser.exit(130, 'Conversion interrupted. Partial output without _SUCCESS is incomplete.\n')
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        parser.exit(1, f'Conversion failed: {error}\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
