"""Precompute breast ROI bounding boxes once so training pays no per-epoch cost.

The boxes are tight around the largest bright connected component; the training
margin stays a separate, tunable knob so one cache serves every experiment.
"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.benchmark_mds import RSNAStreamingDataset, validate_paths
from scripts.training_data import (
    ROI_ANALYSIS_SIZE,
    ROI_MIN_AREA_FRACTION,
    ROI_THRESHOLD,
    load_training_records,
    roi_bounding_box,
)


class BoxDataset:
    def __init__(self, backend, records, threshold, min_area_fraction, analysis_size):
        self.backend = backend
        self.records = records
        self.threshold = threshold
        self.min_area_fraction = min_area_fraction
        self.analysis_size = analysis_size

    def __len__(self):
        return len(self.records)

    def __getitem__(self, position):
        record = self.records[position]
        sample = self.backend[record.index]
        if sample.get('sample_id') != record.sample_id:
            raise ValueError(f'MDS sample mismatch at index {record.index}: sample_id expected '
                             f'{record.sample_id!r}, got {sample.get("sample_id")!r}.')
        pixels = sample['image']
        height, width = pixels.shape[:2]
        box = roi_bounding_box(pixels, self.threshold, self.min_area_fraction, self.analysis_size)
        metadata = sample.get('metadata') or {}
        return (record.sample_id, *box, width, height,
                str(metadata.get('site_id', '')), str(metadata.get('laterality', '')))


def summarise(rows):
    fractions, fallbacks, sides, by_site = [], 0, Counter(), {}
    for _, x0, y0, x1, y1, width, height, site, _ in rows:
        fraction = (x1 - x0) * (y1 - y0) / (width * height)
        fractions.append(fraction)
        fallbacks += int(x0 == 0 and y0 == 0 and x1 == width and y1 == height)
        sides[('left' if x0 <= width - x1 else 'right')] += 1
        by_site.setdefault(site, []).append(fraction)
    fractions.sort()
    middle = fractions[len(fractions) // 2] if fractions else float('nan')
    return {
        'images': len(rows),
        'mean_area_fraction': (sum(fractions) / len(fractions)) if fractions else float('nan'),
        'median_area_fraction': middle,
        'smallest_area_fraction': fractions[0] if fractions else float('nan'),
        'largest_area_fraction': fractions[-1] if fractions else float('nan'),
        'mean_linear_magnification': (sum(f ** -0.5 for f in fractions) / len(fractions)
                                      if fractions else float('nan')),
        'full_frame_fallbacks': fallbacks,
        'chest_wall_side': dict(sides),
        'mean_area_fraction_by_site': {site: sum(values) / len(values)
                                       for site, values in sorted(by_site.items())},
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--mds', type=Path, required=True)
    parser.add_argument('--csv', type=Path)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True, help='Destination ROI box CSV.')
    parser.add_argument('--threshold', type=float, default=ROI_THRESHOLD)
    parser.add_argument('--min-area-fraction', type=float, default=ROI_MIN_AREA_FRACTION)
    parser.add_argument('--analysis-size', type=int, default=ROI_ANALYSIS_SIZE)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--cache-limit', default='20gb')
    parser.add_argument('--limit', type=int, help='Debug-only image cap.')
    parser.add_argument('--clean-shared-memory', action=argparse.BooleanOptionalAction, default=True,
                        help='Clear stale StreamingDataset shared memory left by a crashed run. '
                             'Disable this if a training job is using the same cache directory.')
    return parser


def main():
    args = build_parser().parse_args()
    if args.out.exists():
        raise SystemExit(f'Refusing to overwrite an existing ROI cache: {args.out}')
    if not args.out.parent.is_dir():
        raise SystemExit(f'Output parent must already exist: {args.out.parent}')
    from torch.utils.data import DataLoader

    if args.clean_shared_memory:
        from streaming.base.util import clean_stale_shared_memory

        clean_stale_shared_memory()
    remote, local = validate_paths(str(args.mds), str(args.cache))
    _, records = load_training_records(remote, args.csv)
    records = records[:args.limit] if args.limit else records
    backend = RSNAStreamingDataset(remote=str(remote), local=str(local), decode_images=True,
                                   shuffle=False, batch_size=1, cache_limit=args.cache_limit)
    dataset = BoxDataset(backend, records, args.threshold, args.min_area_fraction,
                         args.analysis_size)
    loader = DataLoader(dataset, batch_size=None, num_workers=args.num_workers,
                        **({'multiprocessing_context': 'spawn'} if args.num_workers else {}))
    rows = []
    for position, row in enumerate(loader, start=1):
        rows.append(tuple(row))
        if position % 5000 == 0:
            print(f'{position}/{len(dataset)} images', file=sys.stderr, flush=True)
    with args.out.open('x', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['sample_id', 'x0', 'y0', 'x1', 'y1', 'width', 'height', 'site_id',
                         'laterality'])
        writer.writerows(rows)
    summary = {'roi_boxes': str(args.out), 'threshold': args.threshold,
               'min_area_fraction': args.min_area_fraction, 'analysis_size': args.analysis_size,
               **summarise(rows)}
    args.out.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
