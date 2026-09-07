import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from scripts.benchmark_mds import RSNAStreamingDataset, validate_paths
from scripts.metrics import pf1
from scripts.training_common import (
    BREAST_APPROACHES,
    DEFAULT_MDS,
    run_epoch,
    validation_metrics,
    write_predictions,
)
from scripts.training_data import (
    BreastDataset,
    ImageDataset,
    ImageTransform,
    collate_samples,
    load_training_records,
    patient_fold_assignments,
)
from scripts.training_models import build_model


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def select_weights(checkpoint: dict, preference: str) -> tuple[dict, str]:
    ema = checkpoint.get('ema_state')
    if preference == 'ema' and not ema:
        raise ValueError('This checkpoint has no EMA weights; use --weights model.')
    if preference == 'model' or (preference == 'auto' and not ema):
        return checkpoint['model_state'], 'model'
    if preference == 'auto' and checkpoint.get('config', {}).get('validation_weights') != 'ema':
        return checkpoint['model_state'], 'model'
    return ema, 'ema'


def evaluate(args: argparse.Namespace) -> dict:
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('Evaluation is single-process; run it without torchrun.')
    output = args.out.resolve()
    if output.exists():
        raise FileExistsError(f'Evaluation output already exists; choose a new directory: {output}')
    if not output.parent.is_dir():
        raise ValueError(f'Evaluation output parent must already exist: {output.parent}')
    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    config = checkpoint.get('config')
    if not isinstance(config, dict) or 'approach' not in config:
        raise ValueError('Checkpoint does not contain a training config; it may not be from these scripts.')
    settings = config['arguments']
    approach = config['approach']
    excluded = set(config.get('excluded_folds') or [])
    if args.fold in excluded:
        selection = 'held-out test fold'
    elif args.fold == settings['fold']:
        if not args.allow_selection_fold:
            raise ValueError(f'Fold {args.fold} was used for validation and model selection in this run. '
                             'Its score is optimistic; pass --allow-selection-fold to compute it anyway.')
        selection = 'validation fold used for model selection (not an unbiased estimate)'
    else:
        raise ValueError(f'Fold {args.fold} was part of this run\'s training data; scoring it is meaningless. '
                         f'Train with --exclude-folds {args.fold} to keep it held out.')
    remote, local = validate_paths(str(args.mds), str(args.cache))
    for path in (remote, local):
        if output == path or output in path.parents or path in output.parents:
            raise ValueError('Evaluation output, dataset, and cache must be separate directories.')
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto'
                          else args.device)
    if device.type not in ('cuda', 'cpu'):
        raise ValueError('--device must select cpu or cuda.')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable; choose --device cpu.')
    manifest, records = load_training_records(remote, args.csv)
    for key in ('csv_sha256', 'index_sha256'):
        if manifest[key] != config['dataset'][key]:
            raise ValueError(f'Dataset {key} differs from the trained checkpoint; scores would not be valid.')
    folds = patient_fold_assignments(records, settings['folds'], settings['seed'])
    indices = [i for i, record in enumerate(records) if folds[record.patient_id] == args.fold]
    if not indices:
        raise ValueError(f'Fold {args.fold} contains no images.')
    torch.manual_seed(settings['seed'])
    backend = RSNAStreamingDataset(remote=str(remote), local=str(local), decode_images=True,
                                   shuffle=False, batch_size=1, cache_limit=args.cache_limit)
    transform = ImageTransform(settings['image_size'], training=False,
                               intensity_max=settings['intensity_max'])
    dataset = (BreastDataset(backend, records, indices, transform, training=False)
               if approach in BREAST_APPROACHES
               else ImageDataset(backend, records, indices, transform))
    model = build_model(approach, pretrained=False, dropout=settings['dropout'],
                        image_size=settings['image_size'],
                        view_chunk_size=args.view_chunk_size or settings['view_chunk_size'])
    state, weights = select_weights(checkpoint, args.weights)
    model.load_state_dict(state)
    model = model.to(device).eval()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == 'cuda',
                        collate_fn=collate_samples,
                        **({'multiprocessing_context': 'spawn'} if args.num_workers else {}))
    results = run_epoch(model, loader, device, amp=args.amp and device.type == 'cuda',
                        description=f'Scoring fold {args.fold}', no_progress=args.no_progress)
    if results['unique_samples'] != len(dataset):
        raise RuntimeError(f'Scored {results["unique_samples"]} of {len(dataset)} samples in the fold.')
    metrics, rows = validation_metrics(results['prediction_ids'], results['labels'],
                                       results['probabilities'], args.pooling or settings['pooling'])
    truth = {records[i].prediction_id: records[i].label for i in indices}
    if {prediction_id: label for prediction_id, label, _ in rows} != truth:
        raise RuntimeError('Scored breasts do not match the fold targets.')
    prevalence = sum(truth.values()) / len(truth)
    report = {
        'checkpoint': str(checkpoint_path), 'checkpoint_sha256': file_sha256(checkpoint_path),
        'checkpoint_epoch': checkpoint.get('epoch'), 'approach': approach, 'weights': weights,
        'fold': args.fold, 'fold_role': selection, 'unbiased_test_estimate': args.fold in excluded,
        'dataset': config['dataset'], 'image_size': settings['image_size'],
        'pooling': args.pooling or settings['pooling'],
        'images': len(indices), 'metrics': metrics,
        'constant_prevalence_pf1': pf1(list(truth.values()), [prevalence] * len(truth)),
        'training_run_validation_fold': settings['fold'], 'training_run_excluded_folds': sorted(excluded),
    }
    output.mkdir(exist_ok=False)
    write_predictions(output / 'predictions.csv', rows)
    with (output / 'labels.csv').open('x', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['prediction_id', 'cancer'])
        writer.writerows(truth.items())
    with (output / 'evaluation.json').open('x', encoding='utf-8') as file:
        json.dump(report, file, indent=2, allow_nan=False)
        file.write('\n')
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Score a trained checkpoint on one held-out fold. Preprocessing and folds come '
                    'from the checkpoint; training folds are refused.'
    )
    parser.add_argument('--checkpoint', type=Path, required=True, help='best.pt, last.pt, or epoch_NNN.pt.')
    parser.add_argument('--fold', type=int, required=True, help='Fold to score; must be a withheld fold.')
    parser.add_argument('--mds', type=Path, default=DEFAULT_MDS)
    parser.add_argument('--csv', type=Path, help='Original conversion CSV; hash must match the manifest.')
    parser.add_argument('--cache', type=Path, required=True, help='Node-local cache directory.')
    parser.add_argument('--out', type=Path, required=True, help='New directory for the report and CSVs.')
    parser.add_argument('--weights', choices=('auto', 'model', 'ema'), default='auto',
                        help='Which stored weights to score; auto follows the training run.')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--view-chunk-size', type=int, help='Override the encoder chunk size for memory.')
    parser.add_argument('--pooling', choices=('mean', 'max'), help='Override image-to-breast pooling.')
    parser.add_argument('--cache-limit', default='20gb')
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--allow-selection-fold', action='store_true',
                        help='Permit scoring the run\'s own validation fold; the result is optimistic.')
    parser.add_argument('--no-progress', action='store_true')
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0 or (args.view_chunk_size or 1) < 1:
        parser.error('--batch-size and --view-chunk-size must be positive; --num-workers nonnegative.')
    try:
        report = evaluate(args)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.exit(1, f'Evaluation failed: {error}\n')
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
