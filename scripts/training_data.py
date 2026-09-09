import csv
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from scripts.benchmark_mds import read_manifest
from scripts.convert_to_mds import load_records


AUX_TARGET_VALUES = {'0': 0.0, '1': 1.0, 'False': 0.0, 'True': 1.0}
ROI_ANALYSIS_SIZE = 256
ROI_THRESHOLD = 0.05
ROI_MIN_AREA_FRACTION = 0.01


@dataclass(frozen=True)
class TrainingRecord:
    index: int
    sample_id: str
    patient_id: str
    prediction_id: str
    label: int
    laterality: str = ''
    aux: tuple[float, ...] = field(default_factory=tuple)


def _check_count(manifest, key, expected):
    if type(manifest.get(key)) is not int or manifest[key] != expected:
        raise ValueError(f'Manifest {key} does not match reconstructed CSV metadata ({expected}).')


def _aux_values(metadata: dict, aux_targets, sample_id: str) -> tuple[float, ...]:
    values = []
    for column in aux_targets:
        if column not in metadata:
            raise ValueError(f'Auxiliary target {column!r} is not a column in the source CSV.')
        raw = metadata[column]
        if raw not in AUX_TARGET_VALUES:
            raise ValueError(f'Auxiliary target {column!r} must be binary for every row; '
                             f'sample {sample_id} has {raw!r}. Columns with blanks are unsupported.')
        values.append(AUX_TARGET_VALUES[raw])
    return tuple(values)


def load_training_records(remote: Path, csv_path: Path | None = None,
                          aux_targets=()) -> tuple[dict, list[TrainingRecord]]:
    try:
        manifest = read_manifest(Path(remote))
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError(f'Corrupt MDS manifest: {error}') from error
    for key in ('csv', 'csv_sha256', 'images_dir', 'image_pattern'):
        if not isinstance(manifest.get(key), str) or not manifest[key].strip():
            raise ValueError(f'Manifest requires a nonempty {key} string.')
    for key in ('limit', 'shuffle_seed', 'skipped_images', 'skipped_missing_images', 'missing_policy'):
        if key not in manifest:
            raise ValueError(f'Unsupported conversion metadata: missing {key}.')
    limit, seed = manifest['limit'], manifest['shuffle_seed']
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError('Manifest limit must be a positive integer or null.')
    if seed is not None and type(seed) is not int:
        raise ValueError('Manifest shuffle_seed must be an integer or null.')
    if manifest['missing_policy'] not in ('skip', 'error'):
        raise ValueError('Unsupported manifest missing_policy.')
    source, checksum = load_records(Path(csv_path) if csv_path is not None else Path(manifest['csv']),
                                    Path(manifest['images_dir']), manifest['image_pattern'])
    if checksum != manifest['csv_sha256']:
        raise ValueError('Source CSV SHA256 differs from conversion manifest; use the original CSV contents.')
    _check_count(manifest, 'csv_rows', len(source))
    source = source[:limit]
    if seed is not None:
        random.Random(seed).shuffle(source)
    _check_count(manifest, 'selected_samples', len(source))
    _check_count(manifest, 'source_files_checked', len(source))
    selected = {f"{record.metadata['patient_id']}_{record.metadata['image_id']}": record
                for record in source}
    skipped = manifest['skipped_images']
    if not isinstance(skipped, list):
        raise ValueError('Manifest skipped_images must be a list.')
    _check_count(manifest, 'skipped_missing_images', len(skipped))
    if skipped and manifest['missing_policy'] != 'skip':
        raise ValueError('Manifest excludes images without the skip missing_policy.')
    skipped_ids = set()
    for entry in skipped:
        if not isinstance(entry, dict) or not isinstance(entry.get('sample_id'), str):
            raise ValueError('Invalid skipped image metadata.')
        sample_id = entry['sample_id']
        if sample_id in skipped_ids or sample_id not in selected:
            raise ValueError(f'Duplicate or unselected skipped sample_id: {sample_id}.')
        record = selected[sample_id]
        expected = {'patient_id': record.metadata['patient_id'], 'image_id': record.metadata['image_id'],
                    'prediction_id': record.prediction_id, 'cancer': int(record.metadata['cancer']),
                    'path': str(record.path)}
        if type(entry.get('cancer')) is not int or any(entry.get(key) != value for key, value in expected.items()):
            raise ValueError(f'Skipped image metadata differs from CSV for {sample_id}.')
        skipped_ids.add(sample_id)
    aux_targets = tuple(aux_targets)
    records = []
    for sample_id, record in selected.items():
        if sample_id not in skipped_ids:
            records.append(TrainingRecord(len(records), sample_id, record.metadata['patient_id'],
                                          record.prediction_id, int(record.metadata['cancer']),
                                          record.metadata.get('laterality', ''),
                                          _aux_values(record.metadata, aux_targets, sample_id)))
    for key, count in (
        ('samples', len(records)),
        ('patients', len({record.patient_id for record in records})),
        ('breasts', len({record.prediction_id for record in records})),
        ('positive_images', sum(record.label for record in records)),
    ):
        _check_count(manifest, key, count)
    return manifest, records


def patient_fold_assignments(records, n_folds: int, seed: int) -> dict[str, int]:
    if type(n_folds) is not int or n_folds < 2:
        raise ValueError('n_folds must be an integer of at least 2.')
    labels = {}
    for record in records:
        if record.label not in (0, 1):
            raise ValueError('Patient fold labels must be binary.')
        labels[record.patient_id] = max(labels.get(record.patient_id, 0), record.label)
    rng = random.Random(seed)
    assignments = {}
    for label, name in ((0, 'negative'), (1, 'positive')):
        patients = sorted(patient for patient, value in labels.items() if value == label)
        if len(patients) < n_folds:
            raise ValueError(f'Need at least {n_folds} {name} patients for stratified folds; found {len(patients)}.')
        rng.shuffle(patients)
        assignments.update((patient, index % n_folds) for index, patient in enumerate(patients))
    return assignments


def _grayscale_preview(pixels: np.ndarray, analysis_size: int) -> np.ndarray:
    image = pixels if pixels.ndim == 2 else pixels.mean(axis=2)
    step = max(1, math.ceil(max(image.shape) / analysis_size))
    preview = np.asarray(image[::step, ::step], dtype=np.float32)
    peak = float(preview.max()) if preview.size else 0.0
    return preview / peak if peak > 0 else preview


def _component_boxes(mask: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    """Label 4-connected runs with union-find; return (area, x0, y0, x1, y1) per component."""
    parent: list[int] = []

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parent[max(left, right)] = min(left, right)

    runs: list[tuple[int, int, int]] = []
    previous: list[tuple[int, int, int]] = []
    for row, line in enumerate(mask):
        current = []
        edges = np.flatnonzero(np.diff(np.concatenate(([0], line.astype(np.int8), [0]))))
        for start, end in zip(edges[0::2].tolist(), edges[1::2].tolist()):
            label = len(runs)
            parent.append(label)
            runs.append((row, start, end))
            for previous_start, previous_end, previous_label in previous:
                if previous_start < end and start < previous_end:
                    union(label, previous_label)
            current.append((start, end, label))
        previous = current
    components: dict[int, list[int]] = {}
    for label, (row, start, end) in enumerate(runs):
        root = find(label)
        box = components.get(root)
        if box is None:
            components[root] = [end - start, start, row, end, row + 1]
        else:
            box[0] += end - start
            box[1], box[2] = min(box[1], start), min(box[2], row)
            box[3], box[4] = max(box[3], end), max(box[4], row + 1)
    return [tuple(box) for box in components.values()]


def roi_bounding_box(pixels: np.ndarray, threshold: float = ROI_THRESHOLD,
                     min_area_fraction: float = ROI_MIN_AREA_FRACTION,
                     analysis_size: int = ROI_ANALYSIS_SIZE) -> tuple[int, int, int, int]:
    """Tight half-open (x0, y0, x1, y1) box around the largest bright component.

    Falls back to the full frame whenever detection is unconvincing, so a bad
    threshold can never silently delete anatomy.
    """
    if not isinstance(pixels, np.ndarray) or pixels.ndim not in (2, 3):
        raise ValueError('Pixels must be a grayscale or RGB NumPy array.')
    if not 0 < threshold < 1 or not 0 <= min_area_fraction < 1:
        raise ValueError('threshold must be in (0, 1) and min_area_fraction in [0, 1).')
    if type(analysis_size) is not int or analysis_size < 16:
        raise ValueError('analysis_size must be an integer of at least 16.')
    height, width = pixels.shape[:2]
    full = (0, 0, width, height)
    preview = _grayscale_preview(pixels, analysis_size)
    if not preview.size or not np.isfinite(preview).all():
        return full
    boxes = _component_boxes(preview > threshold)
    if not boxes:
        return full
    area, x0, y0, x1, y1 = max(boxes)
    if area < min_area_fraction * preview.size:
        return full
    scale_x, scale_y = width / preview.shape[1], height / preview.shape[0]
    box = (max(0, int(math.floor(x0 * scale_x))), max(0, int(math.floor(y0 * scale_y))),
           min(width, int(math.ceil(x1 * scale_x))), min(height, int(math.ceil(y1 * scale_y))))
    return full if box[2] - box[0] < 1 or box[3] - box[1] < 1 else box


def expand_box(box, width: int, height: int, margin: float) -> tuple[int, int, int, int]:
    if not math.isfinite(margin) or margin < 0:
        raise ValueError('ROI margin must be finite and nonnegative.')
    x0, y0, x1, y1 = box
    pad_x, pad_y = margin * (x1 - x0), margin * (y1 - y0)
    return (max(0, int(math.floor(x0 - pad_x))), max(0, int(math.floor(y0 - pad_y))),
            min(width, int(math.ceil(x1 + pad_x))), min(height, int(math.ceil(y1 + pad_y))))


def read_roi_boxes(path: Path) -> dict[str, tuple[int, int, int, int]]:
    with Path(path).open(newline='', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        required = {'sample_id', 'x0', 'y0', 'x1', 'y1'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'ROI box CSV must contain the columns {sorted(required)}.')
        boxes = {}
        for row in reader:
            sample_id = row['sample_id']
            box = tuple(int(row[key]) for key in ('x0', 'y0', 'x1', 'y1'))
            if sample_id in boxes:
                raise ValueError(f'Duplicate ROI box for sample {sample_id}.')
            if box[0] < 0 or box[1] < 0 or box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError(f'ROI box for sample {sample_id} is empty or negative.')
            boxes[sample_id] = box
    if not boxes:
        raise ValueError('ROI box CSV contains no rows.')
    return boxes


class ImageTransform:
    def __init__(self, size: int, training: bool = False, intensity_max: float | None = None,
                 roi_crop: bool = False, roi_margin: float = 0.05,
                 canonical_side: str | None = None):
        if type(size) is not int or size < 1:
            raise ValueError('Image size must be a positive integer.')
        if intensity_max is not None and (not math.isfinite(intensity_max) or intensity_max <= 0):
            raise ValueError('intensity_max must be finite and positive.')
        if not isinstance(roi_crop, bool):
            raise ValueError('roi_crop must be a boolean.')
        if not math.isfinite(roi_margin) or roi_margin < 0:
            raise ValueError('roi_margin must be finite and nonnegative.')
        if canonical_side not in (None, 'left', 'right'):
            raise ValueError("canonical_side must be None, 'left', or 'right'.")
        self.size = size
        self.training = training
        self.intensity_max = intensity_max
        self.roi_crop = roi_crop
        self.roi_margin = roi_margin
        self.canonical_side = canonical_side

    def _needs_canonical_flip(self, box, frame_width: int) -> bool:
        if self.canonical_side is None:
            return False
        chest_side = 'left' if box[0] <= frame_width - box[2] else 'right'
        return chest_side != self.canonical_side

    def __call__(self, pixels: np.ndarray, box=None) -> torch.Tensor:
        if not isinstance(pixels, np.ndarray) or pixels.ndim not in (2, 3):
            raise ValueError('Pixels must be a grayscale or RGB NumPy array.')
        if not pixels.size or (pixels.ndim == 3 and pixels.shape[2] not in (1, 3)):
            raise ValueError('Pixels must have nonempty spatial dimensions and one or three channels.')
        canonical_flip = False
        if self.roi_crop or self.canonical_side is not None:
            frame_height, frame_width = pixels.shape[:2]
            box = roi_bounding_box(pixels) if box is None else tuple(box)
            if box[0] < 0 or box[1] < 0 or box[2] > frame_width or box[3] > frame_height:
                raise ValueError(f'ROI box {box} falls outside the {frame_width}x{frame_height} frame.')
            if box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError(f'ROI box {box} is empty.')
            canonical_flip = self._needs_canonical_flip(box, frame_width)
            if self.roi_crop:
                x0, y0, x1, y1 = expand_box(box, frame_width, frame_height, self.roi_margin)
                pixels = pixels[y0:y1, x0:x1]
        kind, width = pixels.dtype.kind, pixels.dtype.itemsize
        if kind == 'u' and width in (1, 2):
            scale = 255.0 if width == 1 else 65535.0
        elif kind == 'i' and width == 4 and (pixels.ndim == 2 or pixels.shape[2] == 1):
            scale = 65535.0
        elif kind == 'f':
            scale = 1.0
        else:
            raise ValueError(f'Unsupported pixel dtype or color mode: {pixels.dtype}.')
        scale = scale if self.intensity_max is None else self.intensity_max
        if not np.isfinite(pixels).all() or float(pixels.min()) < 0 or float(pixels.max()) > scale:
            raise ValueError(f'Pixels must be finite and within [0, {scale:g}]; set intensity_max for a known range.')
        dtype = np.float32 if kind in ('u', 'i') and 1 <= scale <= 65535 else np.float64
        image = torch.from_numpy(np.divide(pixels, scale, dtype=dtype).astype(np.float32, copy=False))
        image = image.unsqueeze(-1) if image.ndim == 2 else image
        image = image.permute(2, 0, 1)
        if image.shape[0] == 1:
            image = image.expand(3, -1, -1)
        if canonical_flip:
            image = TF.hflip(image)
        height, width = image.shape[-2:]
        ratio = self.size / max(height, width)
        height, width = max(1, round(height * ratio)), max(1, round(width * ratio))
        image = TF.resize(image, [height, width], interpolation=InterpolationMode.BILINEAR, antialias=True)
        left, top = (self.size - width) // 2, (self.size - height) // 2
        image = TF.pad(image, [left, top, self.size - width - left, self.size - height - top])
        if self.training:
            if torch.rand(()).item() < 0.5:
                image = TF.hflip(image)
            angle = torch.empty(()).uniform_(-5, 5).item()
            image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR, expand=True)
            image = TF.resize(image, [self.size, self.size], antialias=True)
        return TF.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).contiguous()


class ImageDataset(Dataset):
    def __init__(self, backend, records, indices, transform, roi_boxes=None):
        indices = list(indices)
        if any(not isinstance(index, (int, np.integer)) or index < 0 or index >= len(records)
               for index in indices) or len(set(indices)) != len(indices):
            raise ValueError('Dataset indices must be unique valid record indices.')
        self.backend = backend
        self.records = [records[index] for index in indices]
        self.transform = transform
        self.roi_boxes = roi_boxes
        self.targets = [record.label for record in self.records]
        if roi_boxes is not None:
            missing = [record.sample_id for record in self.records if record.sample_id not in roi_boxes]
            if missing:
                raise ValueError(f'ROI box cache is missing {len(missing)} samples, '
                                 f'starting with {missing[0]}. Recompute it for this dataset.')

    def __len__(self):
        return len(self.targets)

    def _make_sample(self, records):
        images = []
        for record in records:
            sample = self.backend[record.index]
            expected = {'sample_id': record.sample_id, 'patient_id': record.patient_id,
                        'prediction_id': record.prediction_id, 'cancer': record.label}
            for key, value in expected.items():
                if sample.get(key) != value:
                    raise ValueError(f'MDS sample mismatch at index {record.index}: {key} '
                                     f'expected {value!r}, got {sample.get(key)!r}.')
            box = None if self.roi_boxes is None else self.roi_boxes[record.sample_id]
            images.append(self.transform(sample['image'], box))
        return {'images': torch.stack(images), 'target': float(records[0].label),
                'aux_targets': list(records[0].aux),
                'prediction_id': records[0].prediction_id,
                'sample_ids': [record.sample_id for record in records]}

    def __getitem__(self, index):
        return self._make_sample([self.records[index]])


class BreastDataset(ImageDataset):
    def __init__(self, backend, records, indices, transform, max_views: int | None = None,
                 training: bool = False, roi_boxes=None):
        if max_views is not None and (type(max_views) is not int or max_views < 1):
            raise ValueError('max_views must be a positive integer or None.')
        super().__init__(backend, records, indices, transform, roi_boxes)
        groups = {}
        for record in self.records:
            group = groups.setdefault(record.prediction_id, [])
            if group and (group[0].patient_id != record.patient_id or group[0].label != record.label
                          or group[0].aux != record.aux):
                raise ValueError(f'Breast {record.prediction_id} must have one patient, cancer '
                                 f'label, and auxiliary target vector.')
            group.append(record)
        self.groups = list(groups.values())
        self.targets = [group[0].label for group in self.groups]
        self.max_views = max_views
        self.training = training

    def __getitem__(self, index):
        records = self.groups[index]
        if self.training and self.max_views is not None and len(records) > self.max_views:
            records = [records[index] for index in torch.randperm(len(records))[:self.max_views].tolist()]
        return self._make_sample(records)


class BalancedDistributedSampler(Sampler):
    """Rebalance each epoch to a fixed positive fraction, identically on every rank.

    Positives are drawn with replacement (there are far too few to fill the quota
    otherwise) and negatives without, so the epoch keeps its original length and
    the optimizer-step count stays comparable to an unbalanced run. Every rank
    derives the same permutation from ``seed + epoch``, so no communication is
    needed and resuming stays deterministic.
    """

    def __init__(self, targets, positive_fraction: float, num_replicas: int = 1, rank: int = 0,
                 seed: int = 0):
        if not math.isfinite(positive_fraction) or not 0 < positive_fraction < 1:
            raise ValueError('positive_fraction must be a finite number strictly between 0 and 1.')
        if type(num_replicas) is not int or num_replicas < 1:
            raise ValueError('num_replicas must be a positive integer.')
        if type(rank) is not int or not 0 <= rank < num_replicas:
            raise ValueError('rank must be an integer in 0..num_replicas-1.')
        targets = [int(target) for target in targets]
        if any(target not in (0, 1) for target in targets):
            raise ValueError('Sampler targets must be binary.')
        self.positives = np.flatnonzero(np.asarray(targets) == 1)
        self.negatives = np.flatnonzero(np.asarray(targets) == 0)
        if not len(self.positives) or not len(self.negatives):
            raise ValueError('Balanced sampling needs both positive and negative samples.')
        self.total = len(targets)
        self.positive_fraction = float(positive_fraction)
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        self.samples_per_replica = math.ceil(self.total / num_replicas)
        self.positive_quota = min(max(1, round(self.total * self.positive_fraction)), self.total - 1)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _epoch_indices(self) -> np.ndarray:
        generator = np.random.default_rng((self.seed * 1_000_003 + self.epoch) % (2 ** 32))
        negative_quota = self.total - self.positive_quota
        positives = generator.choice(self.positives, size=self.positive_quota, replace=True)
        negatives = (generator.permutation(self.negatives)[:negative_quota]
                     if negative_quota <= len(self.negatives)
                     else generator.choice(self.negatives, size=negative_quota, replace=True))
        indices = np.concatenate([positives, negatives])
        generator.shuffle(indices)
        padding = self.samples_per_replica * self.num_replicas - len(indices)
        return np.concatenate([indices, indices[:padding]]) if padding else indices

    def __iter__(self):
        return iter(self._epoch_indices()[self.rank::self.num_replicas].tolist())

    def __len__(self):
        return self.samples_per_replica


def collate_samples(items):
    if not items:
        raise ValueError('Cannot collate an empty batch.')
    shape = items[0]['images'].shape[1:]
    for item in items:
        images = item['images']
        if images.ndim != 4 or images.shape[0] < 1 or images.shape[1:] != shape or shape[0] != 3:
            raise ValueError('Each sample needs at least one view with matching [3, H, W] dimensions.')
        if images.dtype != torch.float32 or min(shape) < 1:
            raise ValueError('Images must be nonempty float32 tensors.')
        if len(item['sample_ids']) != len(images) or item['target'] not in (0, 1):
            raise ValueError('Sample IDs must match the views and targets must be binary.')
    aux_width = len(items[0].get('aux_targets', ()))
    if any(len(item.get('aux_targets', ())) != aux_width for item in items):
        raise ValueError('Every sample must carry the same number of auxiliary targets.')
    max_views = max(len(item['images']) for item in items)
    images = items[0]['images'].new_zeros((len(items), max_views, *shape))
    view_mask = torch.zeros((len(items), max_views), dtype=torch.bool)
    for index, item in enumerate(items):
        count = len(item['images'])
        images[index, :count] = item['images']
        view_mask[index, :count] = True
    return {'images': images, 'view_mask': view_mask,
            'targets': torch.tensor([item['target'] for item in items], dtype=torch.float32),
            'aux_targets': torch.tensor([list(item.get('aux_targets', ())) for item in items],
                                        dtype=torch.float32).reshape(len(items), aux_width),
            'prediction_ids': [item['prediction_id'] for item in items],
            'sample_ids': [item['sample_ids'] for item in items]}
