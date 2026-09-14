"""Measure original DICOM geometry and bit depth against the existing PNG cache.

Run this on a small sample before committing to a full re-export. It reports the
native resolution, stored bit depth, photometric interpretation, transfer syntax,
and windowing tags, and quantifies what the 1024x1024 8-bit cache gave up.
"""

import argparse
import csv
import json
import random
import sys
import zipfile
from pathlib import Path

import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


DEFAULT_IMAGES = Path('/Volumes/daai_ke_team/default/images/cancer_dataset/images')
DEFAULT_CSV = Path('/Volumes/daai_ke_team/default/images/cancer_dataset/train 3.csv')


def stratified_members(csv_path: Path, limit: int, seed: int) -> list[str]:
    """Pick a sample spread across machines and balanced on the cancer label."""
    rows = list(csv.DictReader(csv_path.open(newline='', encoding='utf-8-sig')))
    generator = random.Random(seed)
    positives = [r for r in rows if r['cancer'] == '1']
    by_machine: dict[str, list] = {}
    for row in rows:
        by_machine.setdefault(row['machine_id'], []).append(row)
    chosen, seen = [], set()
    # At least one image per machine, then top up with positives and random negatives.
    for machine in sorted(by_machine, key=lambda m: -len(by_machine[m])):
        chosen.append(generator.choice(by_machine[machine]))
    for pool in (positives, rows):
        while len(chosen) < limit and pool:
            chosen.append(generator.choice(pool))
        if len(chosen) >= limit:
            break
    members = []
    for row in chosen[:limit]:
        name = f"train_images/{row['patient_id']}/{row['image_id']}.dcm"
        if name not in seen:
            seen.add(name)
            members.append(name)
    return members


def describe(path: Path, png_dir: Path, decode: bool) -> dict:
    import pydicom

    dataset = pydicom.dcmread(str(path), stop_before_pixels=not decode)
    rows, columns = int(dataset.Rows), int(dataset.Columns)
    spacing = (getattr(dataset, 'ImagerPixelSpacing', None)
               or getattr(dataset, 'PixelSpacing', None))
    spacing = float(spacing[0]) if spacing else None
    record = {
        'file': path.name,
        'rows': rows,
        'columns': columns,
        'aspect_h_over_w': round(rows / columns, 3),
        'bits_stored': int(getattr(dataset, 'BitsStored', 0)),
        'photometric': str(getattr(dataset, 'PhotometricInterpretation', '')),
        'transfer_syntax': str(getattr(dataset.file_meta, 'TransferSyntaxUID', '')),
        'transfer_syntax_name': str(getattr(dataset.file_meta.TransferSyntaxUID, 'name', '')),
        'pixel_spacing_mm': spacing,
        'has_voi_lut': 'VOILUTSequence' in dataset,
        'window_center': str(getattr(dataset, 'WindowCenter', '')),
        'window_width': str(getattr(dataset, 'WindowWidth', '')),
    }
    if decode:
        pixels = dataset.pixel_array
        record.update(native_levels=int(np.unique(pixels).size),
                      native_min=int(pixels.min()), native_max=int(pixels.max()))

    # The cache is named {patient_id}_{image_id}.png; DICOMs are {patient}/{image}.dcm.
    cached = png_dir / f'{path.parent.name}_{path.stem}.png'
    if cached.is_file():
        from PIL import Image

        cache = np.array(Image.open(cached))
        record['cache_shape'] = list(cache.shape)
        record['cache_levels'] = int(np.unique(cache).size)
        record['downsample_long_side'] = round(max(rows, columns) / max(cache.shape), 2)
        # The cache is square, so the horizontal stretch equals the native aspect ratio.
        record['aspect_stretch_applied'] = round(rows / columns, 3)
        if spacing:
            record['native_mm_per_pixel'] = round(spacing, 4)
            record['cache_mm_per_pixel'] = round(spacing * max(rows, columns) / max(cache.shape), 4)
    return record


def summarise(records: list[dict]) -> dict:
    def column(key):
        return [r[key] for r in records if r.get(key) is not None]

    summary = {'sampled': len(records)}
    for key in ('rows', 'columns', 'bits_stored', 'downsample_long_side',
                'aspect_stretch_applied', 'native_mm_per_pixel', 'cache_mm_per_pixel',
                'native_levels', 'cache_levels'):
        values = column(key)
        if values:
            summary[key] = {'min': min(values), 'median': float(np.median(values)),
                            'max': max(values)}
    for key in ('photometric', 'transfer_syntax_name'):
        counts = {}
        for record in records:
            counts[record.get(key, '')] = counts.get(record.get(key, ''), 0) + 1
        summary[key] = counts
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('dicom', type=Path, nargs='?',
                        help='A .dcm file or a directory to scan recursively.')
    parser.add_argument('--zip', type=Path,
                        help='Sample members straight out of the competition archive instead, '
                             'without unpacking it.')
    parser.add_argument('--extract-to', type=Path, default=Path('/root/dicom_probe'),
                        help='Where --zip members are written before inspection.')
    parser.add_argument('--csv', type=Path, default=DEFAULT_CSV,
                        help='Metadata CSV used to spread the --zip sample across machines.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--png-dir', type=Path, default=DEFAULT_IMAGES,
                        help='Existing PNG cache for the side-by-side comparison.')
    parser.add_argument('--limit', type=int, default=20, help='Maximum files to inspect.')
    parser.add_argument('--decode', action=argparse.BooleanOptionalAction, default=True,
                        help='Decode pixels to count native grey levels; slower.')
    parser.add_argument('--out', type=Path, help='Optional JSON report (never overwritten).')
    return parser


def main():
    args = build_parser().parse_args()
    if args.out is not None and args.out.exists():
        raise SystemExit(f'Refusing to overwrite an existing report: {args.out}')
    if (args.dicom is None) == (args.zip is None):
        raise SystemExit('Pass either a DICOM path or --zip, not both.')
    if args.zip is not None:
        if not args.zip.is_file():
            raise SystemExit(f'No such archive: {args.zip}')
        args.extract_to.mkdir(parents=True, exist_ok=True)
        files = []
        with zipfile.ZipFile(args.zip) as archive:
            available = set(archive.namelist())
            wanted = [m for m in stratified_members(args.csv, args.limit, args.seed)
                      if m in available]
            if not wanted:
                raise SystemExit('None of the sampled members exist in the archive; '
                                 'check the archive contents and the CSV.')
            for member in wanted:
                target = args.extract_to / Path(member).parent.name / Path(member).name
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, target.open('wb') as destination:
                    destination.write(source.read())
                files.append(target)
    elif args.dicom.is_file():
        files = [args.dicom]
    elif args.dicom.is_dir():
        files = sorted(args.dicom.rglob('*.dcm'))[:args.limit]
    else:
        raise SystemExit(f'No such DICOM file or directory: {args.dicom}')
    if not files:
        raise SystemExit(f'No .dcm files found under {args.dicom}')
    records = []
    for path in files:
        try:
            records.append(describe(path, args.png_dir, args.decode))
        except Exception as error:  # one unreadable file must not abort the probe
            records.append({'file': path.name, 'error': f'{type(error).__name__}: {error}'})
    report = {'files': records, 'summary': summarise([r for r in records if 'error' not in r])}
    if args.out is not None:
        args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
