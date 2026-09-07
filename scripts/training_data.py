import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from scripts.benchmark_mds import read_manifest
from scripts.convert_to_mds import load_records


@dataclass(frozen=True)
class TrainingRecord:
    index: int
    sample_id: str
    patient_id: str
    prediction_id: str
    label: int


def _check_count(manifest, key, expected):
    if type(manifest.get(key)) is not int or manifest[key] != expected:
        raise ValueError(f'Manifest {key} does not match reconstructed CSV metadata ({expected}).')


def load_training_records(remote: Path, csv_path: Path | None = None) -> tuple[dict, list[TrainingRecord]]:
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
    records = []
    for sample_id, record in selected.items():
        if sample_id not in skipped_ids:
            records.append(TrainingRecord(len(records), sample_id, record.metadata['patient_id'],
                                          record.prediction_id, int(record.metadata['cancer'])))
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


class ImageTransform:
    def __init__(self, size: int, training: bool = False, intensity_max: float | None = None):
        if type(size) is not int or size < 1:
            raise ValueError('Image size must be a positive integer.')
        if intensity_max is not None and (not math.isfinite(intensity_max) or intensity_max <= 0):
            raise ValueError('intensity_max must be finite and positive.')
        self.size = size
        self.training = training
        self.intensity_max = intensity_max

    def __call__(self, pixels: np.ndarray) -> torch.Tensor:
        if not isinstance(pixels, np.ndarray) or pixels.ndim not in (2, 3):
            raise ValueError('Pixels must be a grayscale or RGB NumPy array.')
        if not pixels.size or (pixels.ndim == 3 and pixels.shape[2] not in (1, 3)):
            raise ValueError('Pixels must have nonempty spatial dimensions and one or three channels.')
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
    def __init__(self, backend, records, indices, transform):
        indices = list(indices)
        if any(not isinstance(index, (int, np.integer)) or index < 0 or index >= len(records)
               for index in indices) or len(set(indices)) != len(indices):
            raise ValueError('Dataset indices must be unique valid record indices.')
        self.backend = backend
        self.records = [records[index] for index in indices]
        self.transform = transform
        self.targets = [record.label for record in self.records]

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
            images.append(self.transform(sample['image']))
        return {'images': torch.stack(images), 'target': float(records[0].label),
                'prediction_id': records[0].prediction_id,
                'sample_ids': [record.sample_id for record in records]}

    def __getitem__(self, index):
        return self._make_sample([self.records[index]])


class BreastDataset(ImageDataset):
    def __init__(self, backend, records, indices, transform, max_views: int | None = None,
                 training: bool = False):
        if max_views is not None and (type(max_views) is not int or max_views < 1):
            raise ValueError('max_views must be a positive integer or None.')
        super().__init__(backend, records, indices, transform)
        groups = {}
        for record in self.records:
            group = groups.setdefault(record.prediction_id, [])
            if group and (group[0].patient_id != record.patient_id or group[0].label != record.label):
                raise ValueError(f'Breast {record.prediction_id} must have one patient and cancer label.')
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
    max_views = max(len(item['images']) for item in items)
    images = items[0]['images'].new_zeros((len(items), max_views, *shape))
    view_mask = torch.zeros((len(items), max_views), dtype=torch.bool)
    for index, item in enumerate(items):
        count = len(item['images'])
        images[index, :count] = item['images']
        view_mask[index, :count] = True
    return {'images': images, 'view_mask': view_mask,
            'targets': torch.tensor([item['target'] for item in items], dtype=torch.float32),
            'prediction_ids': [item['prediction_id'] for item in items],
            'sample_ids': [item['sample_ids'] for item in items]}
