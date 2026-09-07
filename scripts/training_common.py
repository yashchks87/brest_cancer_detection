import argparse
import contextlib
import copy
import csv
import json
import math
import os
import random
import shutil
import sys
import tempfile
import time
from itertools import groupby, islice
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from scripts.benchmark_mds import RSNAStreamingDataset, validate_paths
from scripts.metrics import aggregate_predictions, pf1
from scripts.training_data import (
    BreastDataset,
    ImageDataset,
    ImageTransform,
    collate_samples,
    load_training_records,
    patient_fold_assignments,
)
from scripts.training_models import build_model


DEFAULT_MDS = Path('/Volumes/daai_ke_team/default/images/cancer_dataset/shrads/cancer_dataset_mds_v3')
BREAST_APPROACHES = ('advanced', 'vit')
DESCRIPTIONS = {
    'simple': 'Train a ResNet-18 image baseline with breast-level validation.',
    'advanced': 'Train ConvNeXt-Tiny multi-view breast attention pooling.',
    'vit': 'Train a ViT-B/16 multi-view breast attention experiment for comparison.',
}


class Cluster:
    def __init__(self, rank=0, world_size=1, local_rank=0):
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank

    @property
    def distributed(self):
        return self.world_size > 1

    @property
    def primary(self):
        return self.rank == 0

    def barrier(self):
        if self.distributed:
            dist.barrier()

    def sum(self, values, device):
        if not self.distributed:
            return list(values)
        tensor = torch.tensor(list(values), dtype=torch.float64, device=device)
        dist.all_reduce(tensor)
        return tensor.tolist()

    def gather(self, rows):
        if not self.distributed:
            return list(rows)
        gathered = [None] * self.world_size
        dist.all_gather_object(gathered, list(rows))
        return [row for part in gathered for row in part]


def cluster_from_environment():
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if world_size < 1 or not 0 <= rank < world_size or not 0 <= local_rank < world_size:
        raise ValueError('Inconsistent WORLD_SIZE/RANK/LOCAL_RANK; launch one process or use torchrun.')
    if world_size > 1 and not {'MASTER_ADDR', 'MASTER_PORT'} <= set(os.environ):
        raise ValueError('Distributed training needs MASTER_ADDR/MASTER_PORT; launch with torchrun.')
    return Cluster(rank, world_size, local_rank)


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def positive_weight(targets, setting):
    positives = sum(targets)
    negatives = len(targets) - positives
    if not positives or not negatives:
        raise ValueError('The training split must contain both positive and negative targets.')
    weight = min(20.0, max(1.0, math.sqrt(negatives / positives))) if setting == 'auto' else float(setting)
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError('--pos-weight must be auto or a finite positive number.')
    return weight


@torch.no_grad()
def update_ema(ema_model, model, decay):
    for target, source in zip(ema_model.parameters(), model.parameters(), strict=True):
        target.lerp_(source.detach(), 1.0 - decay)
    for target, source in zip(ema_model.buffers(), model.buffers(), strict=True):
        target.copy_(source)


def run_epoch(model, loader, device, *, optimizer=None, scaler=None, amp=False,
              amp_dtype=torch.float16, grad_accum=1, pos_weight=1.0, clip_grad=1.0,
              max_batches=None, ema_model=None, ema_decay=0.0, description='',
              no_progress=False, cluster=None):
    training = optimizer is not None
    cluster = Cluster() if cluster is None else cluster
    model.train(training)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight if training else 1.0, device=device), reduction='sum'
    )
    batches = len(loader) if max_batches is None else min(len(loader), max_batches)
    if batches < 1:
        raise ValueError('Cannot train or validate with an empty loader.')
    total_loss = 0.0
    observed = group_samples = updates = 0
    rows = []
    if training:
        optimizer.zero_grad(set_to_none=True)
    with tqdm(islice(loader, batches), total=batches, desc=description, unit='batch', file=sys.stderr,
              mininterval=1.0, dynamic_ncols=True, disable=no_progress or not cluster.primary) as progress:
        for step, batch in enumerate(progress):
            images = batch['images'].to(device, non_blocking=True)
            mask = batch['view_mask'].to(device, non_blocking=True)
            targets = batch['targets'].to(device, non_blocking=True)
            boundary = (step + 1) % grad_accum == 0 or step + 1 == batches
            accumulating = (training and not boundary and isinstance(model, DistributedDataParallel))
            with model.no_sync() if accumulating else contextlib.nullcontext():
                with torch.set_grad_enabled(training):
                    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
                        logits = model(images, mask)
                        if logits.shape != targets.shape:
                            raise ValueError('Model logits must have one value per target.')
                        loss = criterion(logits, targets)
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite loss; inspect image intensities, learning rate, and weights.')
                    if training:
                        scaler.scale(loss).backward()
                        group_samples += len(targets)
            if training and boundary:
                scaler.unscale_(optimizer)
                denominator = cluster.sum([group_samples], device)[0] / cluster.world_size
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(denominator)
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad,
                                         error_if_nonfinite=not scaler.is_enabled())
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= previous_scale:
                    updates += 1
                    if ema_model is not None:
                        update_ema(ema_model, model, ema_decay)
                optimizer.zero_grad(set_to_none=True)
                group_samples = 0
            total_loss += loss.detach().item()
            observed += len(targets)
            if not training:
                probabilities = logits.detach().float().sigmoid().cpu().tolist()
                labels = targets.detach().cpu().tolist()
                rows.extend(((prediction_id, tuple(sample_ids)), label, probability)
                            for prediction_id, sample_ids, label, probability
                            in zip(batch['prediction_ids'], batch['sample_ids'], labels, probabilities))
            progress.set_postfix(loss=f'{total_loss / observed:.4f}', refresh=False)
    total_loss, observed = cluster.sum([total_loss, observed], device)
    unique = {}
    for key, label, probability in sorted(cluster.gather(rows)):
        unique.setdefault(key, (label, probability))
    return {'loss': total_loss / observed, 'samples': int(observed), 'unique_samples': len(unique),
            'optimizer_steps': updates,
            'prediction_ids': [key[0] for key in unique],
            'labels': [label for label, _ in unique.values()],
            'probabilities': [probability for _, probability in unique.values()]}


def validation_metrics(prediction_ids, labels, probabilities, reduction='mean'):
    ids, targets, scores = aggregate_predictions(prediction_ids, labels, probabilities, reduction)
    positives, negatives = sum(targets), len(targets) - sum(targets)
    correct_pairs = negative_seen = 0.0
    groups = [(score, [label for _, label in group])
              for score, group in groupby(sorted(zip(scores, targets)), key=lambda item: item[0])]
    for _, group in groups:
        group_positive = sum(group)
        group_negative = len(group) - group_positive
        correct_pairs += group_positive * (negative_seen + 0.5 * group_negative)
        negative_seen += group_negative
    true_positive = predicted_count = 0
    average_precision = 0.0
    for _, group in reversed(groups):
        increment = sum(group)
        true_positive += increment
        predicted_count += len(group)
        if positives:
            average_precision += increment / positives * true_positive / predicted_count
    true_positive = sum(label == 1 and score >= 0.5 for label, score in zip(targets, scores))
    predicted_positive = sum(score >= 0.5 for score in scores)
    precision = true_positive / predicted_positive if predicted_positive else 0.0
    recall = true_positive / positives if positives else 0.0
    metrics = {
        'pf1': pf1(targets, scores),
        'roc_auc': correct_pairs / (positives * negatives) if positives and negatives else None,
        'average_precision': average_precision if positives else None,
        'f1_at_0_5': 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        'precision_at_0_5': precision,
        'recall_at_0_5': recall,
        'brier': sum((score - label) ** 2 for label, score in zip(targets, scores)) / len(targets),
        'breasts': len(targets),
        'positive_breasts': positives,
    }
    return metrics, list(zip(ids, targets, scores))


def save_checkpoint(path, checkpoint):
    with tempfile.TemporaryFile() as staging:
        torch.save(checkpoint, staging)
        staging.seek(0)
        with path.open('wb') as destination:
            shutil.copyfileobj(staging, destination, length=8 * 1024 * 1024)


def write_predictions(path, rows):
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['prediction_id', 'cancer'])
        writer.writerows((prediction_id, probability) for prediction_id, _, probability in rows)


def build_parser(approach):
    if approach not in DESCRIPTIONS:
        raise ValueError('Approach must be simple, advanced, or vit.')
    breast = approach in BREAST_APPROACHES
    parser = argparse.ArgumentParser(description=DESCRIPTIONS[approach])
    parser.add_argument('--mds', type=Path, default=DEFAULT_MDS)
    parser.add_argument('--csv', type=Path, help='Original conversion CSV; hash must match the manifest.')
    parser.add_argument('--cache', type=Path, required=True,
                        help='Dedicated node-local cache, separate from MDS/output; never on /Volumes.')
    parser.add_argument('--out', type=Path, required=True,
                        help='New run directory under an existing parent; use /Volumes for persistence.')
    parser.add_argument('--epochs', type=int, default=5 if approach == 'simple' else 10)
    parser.add_argument('--image-size', type=int,
                        default={'simple': 512, 'advanced': 1024, 'vit': 384}[approach],
                        help='Square canvas after aspect-preserving padding; ViT requires a multiple of 16.')
    parser.add_argument('--batch-size', type=int, default=16 if approach == 'simple' else 1)
    parser.add_argument('--grad-accum', type=int, default=1 if approach == 'simple' else 8)
    parser.add_argument('--lr', type=float, default=3e-5 if approach == 'vit' else 1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4 if approach == 'simple' else 0.05)
    parser.add_argument('--dropout', type=float, default=0.0 if approach == 'simple' else 0.2)
    parser.add_argument('--pos-weight', default='auto',
                        help='BCE positive weight: auto uses clipped sqrt(negative/positive) from training only; '
                             'set 1 for unweighted BCE. No oversampling is applied.')
    parser.add_argument('--clip-grad', type=float, default=1.0)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--exclude-folds', type=int, nargs='*', default=[], metavar='FOLD',
                        help='Folds withheld from training and validation, for a locked test set '
                             'scored later by scripts/evaluate.py.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-workers', type=int, default=4 if approach == 'simple' else 2,
                        help='Loader workers per process; total workers scale with the number of GPUs.')
    parser.add_argument('--cache-limit', default='20gb',
                        help='Shared node-local cache budget; ranks on one node share this cache directory.')
    parser.add_argument('--max-views', type=int, default=2,
                        help='Breast-level training view cap; validation always uses all views.')
    parser.add_argument('--view-chunk-size', type=int, default=2,
                        help='Breast-level evaluation encoder chunk size; does not truncate views.')
    parser.add_argument('--intensity-max', type=float,
                        help='Explicit input intensity scale, e.g. 4095 for 12-bit pixels in uint16.')
    parser.add_argument('--pooling', choices=('mean', 'max'), default='mean',
                        help='Image-to-breast validation pooling; breast-level models emit one score already.')
    parser.add_argument('--pretrained', action=argparse.BooleanOptionalAction, default=True,
                        help='Use torchvision ImageNet weights (may download); --no-pretrained is offline.')
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=breast)
    parser.add_argument('--ema-decay', type=float, default=0.0 if approach == 'simple' else 0.999,
                        help='EMA decay; zero disables EMA. Validation/checkpoint selection use EMA when enabled.')
    parser.add_argument('--device', default='auto',
                        help='auto, cpu, cuda, or cuda:N. Distributed runs take the GPU from LOCAL_RANK.')
    parser.add_argument('--save-epochs', action=argparse.BooleanOptionalAction, default=True,
                        help='Keep a weights checkpoint for every epoch beside best.pt/last.pt; '
                             '--no-save-epochs keeps only best/last to save disk.')
    parser.add_argument('--max-train-batches', type=int,
                        help='Debug-only training batch limit per epoch; validation still covers the full fold.')
    parser.add_argument('--no-progress', action='store_true')
    return parser


def validate_args(args, approach='simple'):
    for name in ('epochs', 'batch_size', 'grad_accum', 'max_views', 'view_chunk_size'):
        if getattr(args, name) < 1:
            raise ValueError(f'--{name.replace("_", "-")} must be positive.')
    if args.image_size < 64 or args.num_workers < 0:
        raise ValueError('--image-size must be at least 64 and --num-workers must be nonnegative.')
    if approach == 'vit' and args.image_size % 16:
        raise ValueError('--image-size must be a multiple of 16 for the ViT patch grid.')
    if args.folds < 2 or not 0 <= args.fold < args.folds:
        raise ValueError('--folds must be at least 2 and --fold must be in 0..folds-1.')
    excluded = sorted(set(args.exclude_folds))
    if len(excluded) != len(args.exclude_folds) or any(not 0 <= fold < args.folds for fold in excluded):
        raise ValueError('--exclude-folds must list distinct folds in 0..folds-1.')
    if args.fold in excluded:
        raise ValueError('--exclude-folds must not contain the validation --fold.')
    if len(excluded) >= args.folds - 1:
        raise ValueError('--exclude-folds leaves no training folds.')
    if args.max_train_batches is not None and args.max_train_batches < 1:
        raise ValueError('--max-train-batches must be positive.')
    for name in ('lr', 'clip_grad'):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f'--{name.replace("_", "-")} must be finite and positive.')
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError('--weight-decay must be finite and nonnegative.')
    if not 0 <= args.dropout < 1 or not 0 <= args.ema_decay < 1:
        raise ValueError('--dropout and --ema-decay must be in [0, 1).')
    if args.intensity_max is not None and (not math.isfinite(args.intensity_max) or args.intensity_max <= 0):
        raise ValueError('--intensity-max must be finite and positive.')
    cluster = cluster_from_environment()
    if cluster.distributed and args.device not in ('auto', 'cpu', 'cuda'):
        raise ValueError('Distributed runs select the GPU from LOCAL_RANK; pass --device auto, cuda, or cpu.')
    return cluster


def resolve_device(args, cluster):
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto'
                          else args.device)
    if device.type not in ('cuda', 'cpu'):
        raise ValueError('--device must select cpu or cuda.')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('CUDA is unavailable; choose --device cpu or use a GPU-enabled environment.')
        index = cluster.local_rank if cluster.distributed else (device.index or 0)
        if index >= torch.cuda.device_count():
            raise ValueError(f'CUDA device {index} is unavailable; this host has '
                             f'{torch.cuda.device_count()} visible GPUs.')
        device = torch.device('cuda', index)
        torch.cuda.set_device(device)
    return device


def train(args, approach):
    cluster = validate_args(args, approach)
    device = resolve_device(args, cluster)
    if cluster.distributed:
        dist.init_process_group(backend='nccl' if device.type == 'cuda' else 'gloo',
                                device_id=device if device.type == 'cuda' else None)
    try:
        return _train(args, approach, cluster, device)
    finally:
        if cluster.distributed and dist.is_initialized():
            dist.destroy_process_group()


def _train(args, approach, cluster, device):
    remote, local = validate_paths(str(args.mds), str(args.cache))
    output = args.out.resolve()
    if output.exists():
        raise FileExistsError(f'Run output already exists; choose a new directory: {output}')
    if not output.parent.is_dir():
        raise ValueError(f'Run output parent must already exist: {output.parent}')
    for path in (remote, local):
        if output == path or output in path.parents or path in output.parents:
            raise ValueError('Run output, dataset, and cache must be separate, non-nested directories.')
    random.seed(args.seed)
    np.random.seed(args.seed % (2 ** 32))
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    manifest, records = load_training_records(remote, args.csv)
    source_images = Path(manifest['images_dir']).resolve()
    if output == source_images or output in source_images.parents or source_images in output.parents:
        raise ValueError('Run output must be separate from the original source images directory.')
    folds = patient_fold_assignments(records, args.folds, args.seed)
    withheld = set(args.exclude_folds)
    train_indices = [i for i, record in enumerate(records)
                     if folds[record.patient_id] != args.fold and folds[record.patient_id] not in withheld]
    val_indices = [i for i, record in enumerate(records) if folds[record.patient_id] == args.fold]
    if not train_indices:
        raise ValueError('No training images remain after applying --fold and --exclude-folds.')
    backend = RSNAStreamingDataset(remote=str(remote), local=str(local), decode_images=True,
                                   shuffle=False, batch_size=1, cache_limit=args.cache_limit)
    train_transform = ImageTransform(args.image_size, training=True, intensity_max=args.intensity_max)
    val_transform = ImageTransform(args.image_size, training=False, intensity_max=args.intensity_max)
    if approach in BREAST_APPROACHES:
        train_dataset = BreastDataset(backend, records, train_indices, train_transform,
                                      max_views=args.max_views, training=True)
        val_dataset = BreastDataset(backend, records, val_indices, val_transform, training=False)
    else:
        train_dataset = ImageDataset(backend, records, train_indices, train_transform)
        val_dataset = ImageDataset(backend, records, val_indices, val_transform)
    weight = positive_weight(train_dataset.targets, args.pos_weight)
    raw_model = build_model(approach, pretrained=args.pretrained, dropout=args.dropout,
                            image_size=args.image_size,
                            view_chunk_size=(args.view_chunk_size if approach in BREAST_APPROACHES
                                             else args.batch_size)).to(device)
    ema_model = copy.deepcopy(raw_model).eval().requires_grad_(False) if args.ema_decay else None
    model = (DistributedDataParallel(raw_model, device_ids=[device.index] if device.type == 'cuda' else None)
             if cluster.distributed else raw_model)
    optimizer = torch.optim.AdamW(raw_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
                 if approach != 'simple' else None)
    amp = args.amp and device.type == 'cuda'
    amp_dtype = torch.bfloat16 if amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=amp and amp_dtype == torch.float16)
    samplers = [DistributedSampler(dataset, num_replicas=cluster.world_size, rank=cluster.rank,
                                   shuffle=shuffle, seed=args.seed, drop_last=False)
                if cluster.distributed else None
                for dataset, shuffle in ((train_dataset, True), (val_dataset, False))]
    worker_kwargs = ({'multiprocessing_context': 'spawn', 'persistent_workers': True,
                      'prefetch_factor': 2} if args.num_workers else {})
    loader_kwargs = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                         pin_memory=device.type == 'cuda', collate_fn=collate_samples,
                         worker_init_fn=seed_worker, **worker_kwargs)
    train_loader = DataLoader(train_dataset, sampler=samplers[0], shuffle=samplers[0] is None,
                              generator=torch.Generator().manual_seed(args.seed), **loader_kwargs)
    val_loader = DataLoader(val_dataset, sampler=samplers[1], shuffle=False,
                            generator=torch.Generator().manual_seed(args.seed + 1), **loader_kwargs)
    val_truth = {records[i].prediction_id: records[i].label for i in val_indices}
    train_truth = {records[i].prediction_id: records[i].label for i in train_indices}
    prevalence = sum(train_truth.values()) / len(train_truth)
    config = {
        'approach': approach,
        'arguments': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        'dataset': {'path': str(remote), 'index_sha256': manifest['index_sha256'],
                    'csv_sha256': manifest['csv_sha256'], 'samples': manifest['samples'],
                    'skipped_missing_images': manifest.get('skipped_missing_images', 0)},
        'torch_version': str(torch.__version__),
        'positive_loss_weight': weight,
        'device': str(device),
        'world_size': cluster.world_size,
        'effective_batch_breasts_or_images': args.batch_size * args.grad_accum * cluster.world_size,
        'amp_dtype': str(amp_dtype) if amp else None,
        'excluded_folds': sorted(withheld),
        'excluded_patients': len({patient for patient, fold in folds.items() if fold in withheld}),
        'excluded_images': sum(folds[record.patient_id] in withheld for record in records),
        'train_images': len(train_indices), 'validation_images': len(val_indices),
        'train_breasts': len(train_truth), 'validation_breasts': len(val_truth),
        'train_positive_breasts': sum(train_truth.values()),
        'validation_positive_breasts': sum(val_truth.values()),
        'constant_prevalence_pf1': pf1(val_truth.values(), [prevalence] * len(val_truth)),
        'validation_weights': 'ema' if ema_model is not None else 'model',
    }
    cluster.barrier()
    if cluster.primary:
        output.mkdir(exist_ok=False)
        with (output / 'config.json').open('x', encoding='utf-8') as file:
            json.dump(config, file, indent=2, allow_nan=False)
            file.write('\n')
        with (output / 'folds.csv').open('x', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['patient_id', 'fold'])
            writer.writerows(sorted(folds.items()))
        with (output / 'validation_labels.csv').open('x', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['prediction_id', 'cancer'])
            writer.writerows(val_truth.items())
        print(json.dumps({'event': 'setup', **config}, allow_nan=False), flush=True)
    best_score = -math.inf
    history = []
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        learning_rate = optimizer.param_groups[0]['lr']
        if samplers[0] is not None:
            samplers[0].set_epoch(epoch)
        training = run_epoch(
            model, train_loader, device, optimizer=optimizer, scaler=scaler, amp=amp,
            amp_dtype=amp_dtype, grad_accum=args.grad_accum, pos_weight=weight,
            clip_grad=args.clip_grad, max_batches=args.max_train_batches,
            ema_model=ema_model, ema_decay=args.ema_decay, cluster=cluster,
            description=f'Epoch {epoch}/{args.epochs} train', no_progress=args.no_progress,
        )
        evaluation_model = ema_model if ema_model is not None else raw_model
        validation = run_epoch(evaluation_model, val_loader, device, amp=amp, amp_dtype=amp_dtype,
                               cluster=cluster, description=f'Epoch {epoch}/{args.epochs} validation',
                               no_progress=args.no_progress)
        if validation['unique_samples'] != len(val_dataset):
            raise RuntimeError(f'Validation covered {validation["unique_samples"]} of '
                               f'{len(val_dataset)} held-out samples.')
        metrics, rows = validation_metrics(validation['prediction_ids'], validation['labels'],
                                           validation['probabilities'], args.pooling)
        if {prediction_id: label for prediction_id, label, _ in rows} != val_truth:
            raise RuntimeError('Validation predictions do not match the held-out breast targets.')
        result = {'epoch': epoch, 'lr': learning_rate, 'train_loss': training['loss'],
                  'train_samples': training['samples'], 'optimizer_steps': training['optimizer_steps'],
                  'validation_loss': validation['loss'], 'validation': metrics,
                  'elapsed_seconds': time.perf_counter() - start}
        history.append(result)
        if scheduler is not None:
            scheduler.step()
        improved = metrics['pf1'] > best_score
        best_score = max(best_score, metrics['pf1'])
        if cluster.primary:
            write_predictions(output / f'validation_predictions_epoch_{epoch:03d}.csv', rows)
            if args.save_epochs:
                save_checkpoint(output / f'epoch_{epoch:03d}.pt', {
                    'model_state': raw_model.state_dict(),
                    'ema_state': ema_model.state_dict() if ema_model is not None else None,
                    'epoch': epoch, 'metrics': metrics, 'config': config,
                    'validation_weights': config['validation_weights'],
                })
            if improved:
                save_checkpoint(output / 'best.pt', {'model_state': evaluation_model.state_dict(),
                                                     'epoch': epoch, 'metrics': metrics, 'config': config})
                write_predictions(output / 'best_predictions.csv', rows)
            save_checkpoint(output / 'last.pt', {
                'model_state': raw_model.state_dict(),
                'ema_state': ema_model.state_dict() if ema_model is not None else None,
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict() if scheduler is not None else None,
                'scaler_state': scaler.state_dict(), 'epoch': epoch, 'metrics': metrics,
                'best_pf1': best_score, 'config': config,
            })
            with (output / f'metrics_epoch_{epoch:03d}.json').open('x', encoding='utf-8') as file:
                json.dump(result, file, indent=2, allow_nan=False)
                file.write('\n')
            with (output / 'metrics.jsonl').open('w', encoding='utf-8') as file:
                for row in history:
                    file.write(json.dumps(row, allow_nan=False) + '\n')
            print(json.dumps({'event': 'epoch', **result}, allow_nan=False), flush=True)
        cluster.barrier()
    if cluster.primary:
        with (output / '_TRAINING_SUCCESS').open('x', encoding='utf-8') as file:
            json.dump({'epochs': args.epochs, 'best_pf1': best_score, 'world_size': cluster.world_size},
                      file, allow_nan=False)
            file.write('\n')
    return history


def main(approach):
    parser = build_parser(approach)
    args = parser.parse_args()
    try:
        train(args, approach)
    except KeyboardInterrupt:
        parser.exit(130, 'Training interrupted. Completed epoch checkpoints remain in the run directory.\n')
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        parser.exit(1, f'Training failed: {error}\nUse a new output directory for another run.\n')
