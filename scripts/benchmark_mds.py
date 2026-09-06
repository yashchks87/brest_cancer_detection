import argparse
import hashlib
import io
import json
import time
from itertools import islice
from pathlib import Path

try:
    from streaming import StreamingDataset
except ModuleNotFoundError as error:
    if error.name != 'streaming':
        raise
    StreamingDataset = object


def validate_paths(remote: str, local: str) -> tuple[Path, Path]:
    remote_path, local_path = Path(remote), Path(local)
    if not remote_path.is_absolute() or not local_path.is_absolute():
        raise ValueError('Remote and local must be absolute mounted filesystem paths.')
    remote_path, local_path = remote_path.resolve(), local_path.resolve()
    if remote_path == local_path or remote_path in local_path.parents or local_path in remote_path.parents:
        raise ValueError('Remote dataset and local cache paths must be separate, non-nested directories.')
    if any(root == local_path or root in local_path.parents for root in (Path('/Volumes'), Path('/dbfs'))):
        raise ValueError('Cache must be on node-local disk, not /Volumes or /dbfs.')
    return remote_path, local_path


def read_manifest(remote: Path) -> dict:
    for name in ('_SUCCESS', 'conversion.json', 'index.json'):
        if not (remote / name).is_file():
            raise ValueError(f'Incomplete dataset: missing {name} in {remote}.')
    manifest = json.loads((remote / 'conversion.json').read_text(encoding='utf-8'))
    success = json.loads((remote / '_SUCCESS').read_text(encoding='utf-8'))
    index_bytes = (remote / 'index.json').read_bytes()
    index = json.loads(index_bytes)
    if not isinstance(manifest, dict) or not isinstance(success, dict) or not isinstance(index, dict):
        raise ValueError('Dataset manifests must be JSON objects.')
    if manifest.get('schema_version') != 1 or manifest.get('image_storage') not in ('bytes', 'ndarray'):
        raise ValueError('Unsupported conversion schema or image storage type.')
    count = manifest.get('samples')
    if type(count) is not int or count < 1 or success.get('samples') != count:
        raise ValueError('Invalid or inconsistent completion sample counts.')
    if sum(shard['samples'] for shard in index['shards']) != count:
        raise ValueError('Index sample count differs from conversion manifest.')
    if manifest.get('index_sha256') != hashlib.sha256(index_bytes).hexdigest():
        raise ValueError('Index checksum differs from conversion manifest.')
    return manifest


class RSNAStreamingDataset(StreamingDataset):
    def __init__(self, remote: str, local: str, decode_images: bool = True,
                 transform=None, **kwargs):
        if StreamingDataset is object:
            raise ImportError('mosaicml-streaming is missing; use the Databricks ML Python environment.')
        if transform is not None and not decode_images:
            raise ValueError('Transforms require decode_images=True.')
        if kwargs.get('streams') is not None or kwargs.get('split') is not None:
            raise ValueError('Pass the completed dataset directory directly; streams/split are unsupported.')
        remote_path, local_path = validate_paths(remote, local)
        self.manifest = read_manifest(remote_path)
        self.decode_images = decode_images
        self.transform = transform
        kwargs.setdefault('validate_hash', 'sha256')
        kwargs.setdefault('download_timeout', 120)
        super().__init__(remote=str(remote_path), local=str(local_path), **kwargs)

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        if self.decode_images:
            if isinstance(sample['image'], bytes):
                import numpy as np
                from PIL import Image

                with Image.open(io.BytesIO(sample['image'])) as image:
                    sample['image'] = np.array(image)
            if self.transform is not None:
                sample['image'] = self.transform(sample['image'])
        return sample


def count_batch(samples: list[dict]) -> dict[str, list[int]]:
    return {'payload_bytes': [len(sample['image']) if isinstance(sample['image'], bytes)
                              else sample['image'].nbytes for sample in samples]}


def positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError('Must be positive.')
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Benchmark MDS storage/image decoding through a node-local StreamingDataset cache.'
    )
    parser.add_argument('--remote', required=True, help='Completed mounted Volume dataset directory.')
    parser.add_argument('--local', required=True, help='Separate node-local cache directory.')
    parser.add_argument('--batch-size', type=positive_int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--epochs', type=positive_int, default=2)
    parser.add_argument('--max-batches', type=positive_int)
    parser.add_argument('--cache-limit', default='20gb')
    parser.add_argument('--predownload', type=positive_int, default=256)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--decode-images', action='store_true')
    return parser


def benchmark(args: argparse.Namespace) -> list[dict]:
    if args.num_workers < 0:
        raise ValueError('--num-workers must be nonnegative.')
    from streaming import StreamingDataLoader

    start = time.perf_counter()
    dataset = RSNAStreamingDataset(
        remote=args.remote, local=args.local, decode_images=args.decode_images,
        batch_size=args.batch_size, shuffle=args.shuffle, shuffle_seed=args.seed,
        cache_limit=args.cache_limit, predownload=args.predownload,
    )
    loader_kwargs = ({'persistent_workers': True, 'prefetch_factor': 2,
                      'multiprocessing_context': 'spawn'} if args.num_workers else {})
    loader = StreamingDataLoader(dataset, batch_size=args.batch_size,
                                 num_workers=args.num_workers, collate_fn=count_batch,
                                 drop_last=False, **loader_kwargs)
    print(json.dumps({'event': 'init', 'seconds': time.perf_counter() - start,
                      'dataset_samples': dataset.manifest['samples'],
                      'image_storage': dataset.manifest['image_storage']}), flush=True)
    results = []
    for epoch in range(args.epochs):
        start = time.perf_counter()
        count = payload_bytes = 0
        first_batch_seconds = None
        for batch in islice(loader, args.max_batches):
            if first_batch_seconds is None:
                first_batch_seconds = time.perf_counter() - start
            count += len(batch['payload_bytes'])
            payload_bytes += sum(batch['payload_bytes'])
        elapsed = time.perf_counter() - start
        result = {
            'event': 'epoch', 'epoch': epoch + 1, 'samples_observed': count,
            'elapsed_seconds': elapsed, 'images_per_second': count / elapsed,
            'logical_payload_mib_per_second': payload_bytes / (1024 * 1024) / elapsed,
            'first_batch_seconds': first_batch_seconds, 'decode_images': args.decode_images,
        }
        print(json.dumps(result), flush=True)
        results.append(result)
    return results


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        benchmark(args)
    except (ImportError, OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(1, f'Benchmark failed: {error}\nCheck the ML Python dependencies, completed '
                       'remote dataset, Volume permissions, and node-local cache capacity.\n')


if __name__ == '__main__':
    main()
