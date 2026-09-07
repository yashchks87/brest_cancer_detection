# RSNA breast cancer detection: evaluation and modeling

## Project goal and current status

This repository aims to reproduce and improve breast-level cancer prediction from screening mammograms for the [RSNA Screening Mammography Breast Cancer Detection competition](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/overview).

The original competition ended on February 27, 2023. This is a research/reproduction project, not an active prize entry. Check Kaggle directly for any current late-submission availability.

**Implemented:**

- `scripts/metrics.py`: dependency-free probabilistic F1 scoring, optional image-to-breast aggregation, and validation CSV scoring aligned by ID.
- `scripts/test_metrics.py`: unit tests for reference-formula parity, edge cases, aggregation, CSV validation, and the command-line interface.
- `scripts/convert_to_mds.py`: CSV-driven, bounded-parallel conversion of existing images into MosaicML Streaming MDS shards on a mounted Unity Catalog Volume.
- `scripts/benchmark_mds.py`: reusable cached streaming reader and storage/image-decoding benchmark.
- `scripts/test_mds_conversion.py` and `scripts/test_mds_reader.py`: synthetic MDS conversion and reader tests.
- `scripts/train_simple.py`: ResNet-18 image training with breast-level validation.
- `scripts/train_advanced.py`: ConvNeXt-Tiny multi-view breast attention training, AMP, gradient accumulation, cosine scheduling, and EMA.
- `scripts/train_vit.py`: ViT-B/16 multi-view breast experiment sharing the advanced training protocol for a controlled comparison.
- `scripts/compare_runs.py`: same-fold run comparison with recomputed pF1, comparability warnings, and patient-level bootstrap intervals.
- `scripts/training_data.py`, `scripts/training_models.py`, and `scripts/training_common.py`: shared MDS metadata reconstruction, patient folds, image transforms, models, training, metrics, and checkpoints.
- `scripts/test_training*.py`: metadata/fold/transform/model tests and one-epoch synthetic CPU training tests.

**Not implemented yet:** DICOM preprocessing, an inference/submission CLI, automatic checkpoint resume, multi-node training, threshold optimization, calibration, or ensembling. No real-data-trained model or measured model-quality result is included. The modeling options below remain experiment recommendations, not claims of completed experiments.

## Competition target and data

Predict `cancer` once per breast, identified by `(patient_id, laterality)`. Several images/views can correspond to the same breast. Do not score every image as an independent competition example: breasts with more images would receive extra weight.

The original dataset provides:

- `train_images/<patient_id>/<image_id>.dcm` and corresponding test images.
- `train.csv` / `test.csv` with image and patient metadata.
- `sample_submission.csv` with `prediction_id,cancer` columns.

Use the actual `prediction_id` supplied in test metadata and preserve the IDs in `sample_submission.csv`; do not guess their separator or format. The underscore-separated IDs in the local examples below are illustrative validation keys only.

`cancer`, `biopsy`, `invasive`, `BIRADS`, `density`, and `difficult_negative_case` are training-only columns in the archived competition schema. Never use them as inference inputs. Some can be investigated as auxiliary training targets, with appropriate missing-label masks. Verify the actual train/test columns before selecting any metadata inputs.

## Exact competition scoring rule

The competition metric is **probabilistic F1 (pF1), with beta = 1**, and higher is better. For binary labels `y_i` and submitted probabilities `p_i`, the published reference clips predictions to `[0, 1]` and uses:

```text
pTP = sum(y_i * p_i)
pFP = sum((1 - y_i) * p_i)
P   = sum(y_i)

pPrecision = pTP / (pTP + pFP)
pRecall    = pTP / P
pF1        = 2 * pPrecision * pRecall / (pPrecision + pRecall)
           = 2 * pTP / (P + sum(p_i))
```

There is no automatic threshold search, rounding, class weighting, or sigmoid inside this metric. If you submit binary predictions, it reduces to ordinary F1. A locally optimized binary F1 is a separate diagnostic unless those exact binary predictions are what you submit.

`pfbeta` implements the same published precision/recall formula for a configurable beta; only `beta=1` is the competition metric. Floating-point summation can differ from the loop reference at roundoff level.

### Explicit robustness policies

These are local safeguards, not claims about Kaggle's handling of invalid submissions:

- The low-level `pf1` / `pfbeta` functions clip finite out-of-range predictions like the reference, without mutating the input.
- Empty or unequal-length inputs, nonbinary labels, and NaN/infinite predictions are rejected.
- No positive labels, no predicted-positive mass, or zero true-positive mass returns `0.0`, avoiding division by zero in degenerate validation sets.
- CSV scoring and aggregation require probabilities already in `[0, 1]`, catching accidental logits instead of silently clipping them.
- CSV scoring rejects duplicate IDs, missing/extra IDs, malformed rows, duplicate headers, and missing required columns. It aligns by ID, not row order.
- Aggregation rejects conflicting labels within a breast. Mean and max pooling are optional modeling choices, **not part of the official pF1 formula**.

pF1 is not a proper probability scoring rule: maximizing it does not guarantee calibrated cancer probabilities. Separate leaderboard-oriented postprocessing from probability calibration and clinical operating-point evaluation.

## Usage

Requires Python 3.10 or newer. The scorer and tests use only the standard library; no package installation is needed. Run these commands from the repository root, `/root/brest_cancer_detection`.

### Score already aggregated breast predictions

```python
from scripts.metrics import pf1

labels = [1, 0, 1, 0]
probabilities = [0.9, 0.2, 0.8, 0.1]
score = pf1(labels, probabilities)
print(score)
```

Expected score: approximately `0.85`.

Inputs are one-dimensional numeric iterables, such as lists, NumPy arrays, or pandas Series. Index labels are ignored: ensure that labels and probabilities have matching positional order. For PyTorch, first apply sigmoid to binary logits, detach, transfer to CPU, and flatten; for example, `logits.sigmoid().detach().cpu().reshape(-1).tolist()`. The metric is for evaluation, not a differentiable training loss.

### Aggregate validation images to breasts

```python
from scripts.metrics import aggregate_predictions, pf1

prediction_ids = ['10_L', '10_L', '10_R', '10_R']
image_labels = [1, 1, 0, 0]
image_probabilities = [0.9, 0.7, 0.2, 0.1]

breast_ids, breast_labels, breast_probabilities = aggregate_predictions(
    prediction_ids, image_labels, image_probabilities, reduction='mean'
)
score = pf1(breast_labels, breast_probabilities)
```

The result preserves first-seen ID order. Use `reduction='max'` to compare max pooling on validation data. This helper requires labels and is intended for validation; an unlabeled inference/submission builder has not been implemented. Local validation IDs can be constructed from `patient_id` and `laterality`, but never combine both breasts into one target.

### Score validation CSV files

Create a breast-level `validation_labels.csv` containing binary ground truth:

```csv
prediction_id,cancer
10_L,1
10_R,0
```

Create `validation_predictions.csv` containing probabilities for the same IDs:

```csv
prediction_id,cancer
10_R,0.15
10_L,0.8
```

Then run:

```bash
python scripts/metrics.py --solution validation_labels.csv --submission validation_predictions.csv
```

The command prints JSON such as `{"pf1": 0.8205128205128206}` (the last digits may vary with floating-point arithmetic). These are example filenames, not included datasets. The ground-truth CSV must already contain one row per breast; raw image-level `train.csv` is not directly suitable. The actual hidden test labels are unavailable locally.

### Run verification

```bash
python -B -m unittest discover -s scripts -p 'test_*.py' -v
python -B scripts/metrics.py --help
```

The suite includes 300 seeded comparisons with the published reference formula across beta values 0.5, 1, and 2, along with explicit soft-score, clipping, zero-case, grouping, and CLI tests. These tests verify the metric implementation, not model quality.

## MDS shards on Unity Catalog Volumes

### Verified source layout

```text
/Volumes/daai_ke_team/default/images/cancer_dataset/
    train 3.csv
    images/
        10006_462822612.png
        ...
```

The converter uses `{patient_id}_{image_id}.png` from CSV rows, avoiding an expensive recursive listing of the Volume. A read-only dry run validated the CSV's **54,706 image rows, 11,913 patients, 23,826 breasts, and 1,158 positive image rows**, and checked eight selected image files. It did not verify every source image or create the full shard dataset.

A full source-path audit on September 7, 2026 found **19 missing PNGs out of 54,706 expected files**, including `51695_1368841260.png`. The complete restoration checklist was saved to `/Volumes/daai_ke_team/default/images/cancer_dataset/shrads/missing_images_v2.json`. No source images or CSV rows were changed, and no new shards were created. A complete-source conversion requires restoring those files; `--skip-missing` can explicitly produce a reduced dataset instead. This path audit did not validate every present image's contents.

This is **Mosaic Data Shard (MDS)** storage, not a mosaic image augmentation. It packs many images into sequentially readable shard files to reduce small-file overhead. Actual throughput depends on storage/network bandwidth, CPU decoding, cache capacity, worker count, and training transforms; MDS is not a guarantee of maximum cluster speed.

### Environment

The converter/reader were exercised with the cluster's existing `mosaicml-streaming==0.12.0`, `torch==2.7.0`, `torchvision==0.22.0`, `Pillow==11.1.0`, and `numpy==2.1.3`. No packages were installed or upgraded for this task. Databricks Runtime ML 15.2+ includes Mosaic Streaming; use the ML runtime's Python environment on all participating nodes. The newly created project `venv` does not automatically inherit cluster packages.

Before conversion, check the interpreter you intend to use:

```bash
python -c "import sys, streaming, PIL, numpy; print(sys.executable); print(streaming.__version__)"
```

If this fails inside an empty virtualenv, switch back to the Databricks ML Python environment, or provision a compatible environment deliberately. Do not blindly upgrade the cluster's CUDA/PyTorch stack. The metric-only code still requires no external packages.

### Dry run, smoke conversion, and full conversion

Run from `/root/brest_cancer_detection`. The default CSV/images paths match the source above. Output must be a **new** directory whose parent already exists; the following proposed output locations are siblings of the source dataset in the same Volume.

Read-only preflight:

```bash
python scripts/convert_to_mds.py \
  --out /Volumes/daai_ke_team/default/images/cancer_dataset_mds_v1 \
  --dry-run
```

This validates all CSV rows, checks that every selected image path exists and is a regular file, and loads at most eight selected images. The full path check also runs before every real conversion, using bounded parallel metadata requests without recursively listing the Volume. It does not verify the contents/readability of every image or test write permissions; files can still disappear or become unreadable after preflight. `--limit` selects the first N CSV rows before deterministic shuffling, restricts the path check to those rows, and is only a smoke-test feature, not a representative training split.

Small real conversion, to a separate destination:

```bash
python scripts/convert_to_mds.py \
  --out /Volumes/daai_ke_team/default/images/cancer_dataset_mds_smoke_v1 \
  --limit 128
```

Full conversion, preserving original image bytes:

```bash
python scripts/convert_to_mds.py \
  --csv "/Volumes/daai_ke_team/default/images/cancer_dataset/train 3.csv" \
  --images-dir /Volumes/daai_ke_team/default/images/cancer_dataset/images \
  --out /Volumes/daai_ke_team/default/images/cancer_dataset_mds_v1 \
  --workers 8 --prefetch 16 --shard-size-mb 256
```

`--workers` controls concurrent image reads on **one conversion process/node**. This is not Spark-distributed conversion: run it once, not independently on every executor against the same destination. `--prefetch` bounds outstanding image reads and must be at least `--workers`; unlike unbounded executor submission, it does not queue all image payloads in RAM. Memory still includes prefetched images, the writer's shard buffer, temporary copies, and decode/compression buffers. The default per-image safety limit is 64 MiB; tune `--max-sample-mb` deliberately for larger data.

Use `--image-pattern '{patient_id}/{image_id}.png'` for a different nested PNG layout. Single-frame PNG, JPEG, and TIFF are accepted; original DICOM conversion is not implemented. Images are never resized, cropped, normalized, or converted to lossy JPEG by this script. Train-only metadata is preserved as metadata, not automatically fed into a model.

### Missing source images

By default, missing CSV-listed images cause an error **before any MDS output directory is created**. The converter checks every selected source path, reports the total missing count and up to 20 paths in the error, and never silently excludes rows or substitutes other images. Permission errors and non-file paths fail separately rather than being classified as missing images.

To save a complete restoration checklist without creating shards:

```bash
python scripts/convert_to_mds.py \
  --out /Volumes/daai_ke_team/default/images/cancer_dataset/shrads/cancer_dataset_mds_v3 \
  --dry-run \
  --missing-report /Volumes/daai_ke_team/default/images/cancer_dataset/shrads/missing_images_v2.json
```

`--missing-report` is optional. It writes a new JSON file **only if images are missing**, containing the CSV path/hash, selected count, missing count, and every missing patient ID, image ID, and expected path. The report's parent directory must exist, and the report must be outside the source image directory and MDS output directory. Existing report files are never overwritten: choose a new report name on a later audit. A dry run without this option still writes nothing; with this option, only the diagnostic report may be written.

Restore the exact missing images from the original source, then repeat preflight using a new report filename. Do not replace a missing image with another view or a blank placeholder. If an older conversion left partial output, preserve it and choose a new output version for the retry; the converter does not resume partial datasets. Successful summaries include `source_files_checked`; it counts path checks, not full image-content validation.

To explicitly convert only available images, add `--skip-missing`:

```bash
python scripts/convert_to_mds.py \
  --out /Volumes/daai_ke_team/default/images/cancer_dataset/shrads/cancer_dataset_mds_v3 \
  --skip-missing --workers 8 --prefetch 16 --shard-size-mb 256
```

This excludes only paths found missing during preflight. It preserves all remaining images and leaves the source CSV and files unchanged. A warning is printed even with `--no-progress`, and the final status includes the skipped count. Corrupt, oversized, non-file, or unreadable inputs still fail; a file that disappears after preflight also fails rather than being silently dropped. If no images remain, no output is created. `--limit` is applied before checking/skipping and does not backfill skipped rows from later CSV entries.

The final JSON summary and `conversion.json` always include `missing_policy`, `selected_samples` (before skipping), `skipped_missing_images`, and `skipped_images` (sample/patient/image/breast IDs, label, and expected path for every exclusion). `samples`, patient/breast/positive-image counts, the progress denominator, the shard index, and `_SUCCESS` all describe the **retained subset**. The optional separate `--missing-report` also works with skipping; it is not required for the manifest audit. `--dry-run --skip-missing` checks the paths and loads up to eight retained images without creating shards.

With the 19 missing paths found in the audit above, the expected retained count is **54,687**, provided the source remains unchanged and no other read/content errors occur. Missing views can leave incomplete breasts or remove entire patient/breast groups; review `skipped_images` when evaluating coverage or comparing results with the full dataset. `_SUCCESS` in skip mode confirms the retained subset was written completely, not that all original CSV images were available.

### Live conversion progress

Progress is enabled by default on **stderr**, leaving the final JSON summary on stdout unchanged. `--progress auto` selects an updating terminal bar when stderr is a terminal, or timestamped, newline-delimited logs in Databricks notebooks, redirected output, and job logs. Terminal output wraps to keep the metrics visible on narrow screens; use `--progress log` if your console does not support cursor controls.

The tracker shows the current stage, processed/selected images, percentage, average images/sec, elapsed time, estimated remaining image-processing time, logical image-payload MiB and MiB/sec, and time since the last processed image or stage change (`idle`). A background heartbeat updates even while the main thread waits on image reads or shard writes. An increasing idle time signals no newly completed sample/stage, not proof of a deadlock; it does not cancel slow storage operations.

- `--progress-interval 1`: seconds between heartbeat updates (default: 1; must be finite and positive).
- `--progress-every 1000`: additional updates at image-count milestones, retained for compatibility. Startup, stage transitions, the final image, and success/failure are always reported when progress is enabled, including runs smaller than this count.
- `--progress log --progress-interval 5`: less frequent, notebook/job-friendly heartbeat logs.
- `--progress bar`: explicitly select the live terminal display.
- `--no-progress` or `--progress none`: suppress progress only; errors and the final JSON summary remain enabled.

**ETA is an estimate for the remaining images, not a completion guarantee.** It uses average processing throughput and is unknown (`--`) until there is a measured rate. During final shard flushing, index verification, and manifest publication, ETA returns to unknown while the heartbeat and elapsed timer continue. Payload throughput is not physical disk/network bandwidth, and processed images may still be buffered by the MDS writer: **100% images does not mean the dataset is ready**. `Complete` is reported only after verification and creation of `_SUCCESS`. Dry runs first track all selected source paths, then the up-to-eight loaded images. Dry runs report that no shards were written and distinguish an explicitly saved missing-image report from a run that wrote nothing. Failures/interruption report their stage and processed count without declaring completion; Ctrl-C exits with status 130 after cleanup.

### Storage choices

| Setting | Stored image representation | When to try it |
| --- | --- | --- |
| Default `--image-storage bytes --compression none` | Original encoded image bytes, byte-for-byte | Recommended first: PNG is already compressed, so it avoids redundant shard decompression |
| `--image-storage ndarray --compression none` | Decoded pixel arrays with shape and numeric precision preserved | Avoid per-image PNG decoding on warm-cache training; requires much more disk/cache/network capacity |
| `--image-storage ndarray --compression zstd:1` | Losslessly compressed array shards | Trade shard-level decompression and expanded cache space against network/storage savings |

For example, create a separate array-backed version to benchmark:

```bash
python scripts/convert_to_mds.py \
  --out /Volumes/daai_ke_team/default/images/cancer_dataset_mds_arrays_v1 \
  --image-storage ndarray --compression zstd:1
```

Array conversion preserves 16-bit pixel values and normalizes byte order to native endian for MDS encoding. It deliberately rejects unsupported modes such as palette images rather than discard palette semantics. Compressed shards expand in the node-local cache, so size the cache for expanded data and concurrent downloads, not only the compressed Volume size.

### Output integrity and schema

Each completed output contains:

```text
shard.00000.mds          (or shard.00000.mds.zstd)
shard.00001.mds
...
index.json
conversion.json
_SUCCESS
```

`conversion.json` records counts, source CSV hash, image pattern/storage mode, shuffle seed, shard settings, package versions, index hash, and conversion timing. Each MDS shard carries a SHA-256 hash. `_SUCCESS` is written last after writer completion and index/sample-count checks. The reader below requires the completion marker and manifest. These markers are local application safeguards, not standard MDS requirements or a transactional publication protocol.

The converter refuses **all existing output directories**, including empty ones. It never overwrites or deletes source data/output, never silently skips missing/corrupt images, and has no resume mode. If interrupted, partial shards or even an index may exist without `_SUCCESS`; do not train on them. Retry into a new version directory. Existing cached datasets must be treated as immutable: changing shard contents behind an existing cache can serve stale data.

Each sample contains `image`, `sample_id`, `patient_id`, `image_id`, `prediction_id`, `laterality`, `view`, `cancer`, and `metadata`. IDs are strings; `cancer` is an integer. `metadata` preserves all CSV column values as strings, including empty values and training-only columns. An existing CSV `prediction_id` is preserved and checked for consistency; otherwise a local validation key `patient_id_laterality` is generated. This does not replace the official test submission IDs.

### Read from the Volume, cache on each node's local disk

Use the Volume as `remote` and a distinct **node-local** directory such as `/local_disk0/rsna_cache/v1` as `local`. Do not put the cache on `/Volumes`, `/dbfs`, or shared network storage. The scripts use mounted absolute paths, not `dbfs:/...` URIs. The same Volume must be accessible from every training process/node under the cluster's Unity Catalog permissions; no credentials are embedded in code.

Storage-only benchmark:

```bash
python scripts/benchmark_mds.py \
  --remote /Volumes/daai_ke_team/default/images/cancer_dataset_mds_v1 \
  --local /local_disk0/rsna_cache/v1 \
  --cache-limit 20gb --predownload 256 \
  --batch-size 32 --num-workers 4 --epochs 2
```

Add `--decode-images` to measure PNG decoding as well, and `--shuffle` to approximate shuffled training access. A limited check can use `--max-batches 20`. Reported payload MiB/s is logical sample payload throughput, **not physical network bandwidth**; decoded arrays and encoded bytes have different sizes. The benchmark collates counts/byte sizes rather than shipping full image tensors to the main process, so it is not end-to-end model-training throughput.

An already populated cache means the first measured epoch is not necessarily cold. Use a new local cache path when intentionally testing cold-cache behavior. Later epochs are only mostly warm if the working set fits in cache; the scripts never automatically clear caches or shared memory belonging to other jobs.

The reader is reusable from training code:

```python
from scripts.benchmark_mds import RSNAStreamingDataset
from streaming import StreamingDataLoader

dataset = RSNAStreamingDataset(
    remote='/Volumes/daai_ke_team/default/images/cancer_dataset_mds_v1',
    local='/local_disk0/rsna_cache/training_v1',
    decode_images=True,
    shuffle=True,
    batch_size=32,
    cache_limit='20gb',
    predownload=256,
    validate_hash='sha256',
)
sample = dataset[0]
print(sample['image'].shape, sample['image'].dtype, sample['cancer'])
```

`decode_images=True` returns NumPy pixel arrays without implicit RGB conversion, resizing, or normalization; `transform=` may supply a model-specific array-to-tensor transform. Before constructing a training `StreamingDataLoader`, provide a transform/collate function that handles variable image sizes, preserves the intended intensity scale, and selects only appropriate metadata. Raw CSV metadata should not be passed wholesale into the model. Sixteen-bit images must not be blindly divided by 255.

For distributed training, construct the dataset inside each training process after distributed setup; use the same dataset version and matching per-device `batch_size` in the dataset and loader. Streaming handles rank/worker partitioning: do not add `DistributedSampler` or independent DataLoader shuffling. Share one cache per dataset among ranks on the same node, but use job/version-specific cache directories for unrelated runs. Make these Python modules available on every node, not just the driver's `/root` directory.

Create patient-disjoint train/validation CSVs **before** converting training splits, then convert each into a separate versioned output. The full-source conversion command is an archival/benchmark dataset, not an automatic train/validation split. Preserve all views and both breasts of a patient in one fold. Distributed iteration can pad samples; deduplicate by `sample_id` before breast aggregation when evaluating, and do not average batch pF1.

### Tuning and tests

Start with 256 MiB shards, 8 conversion read threads, and 4 loader workers. Compare 64/128/256 MiB shard sizes and 2/4/8 loader workers empirically. Increase predownload only when needed to hide remote latency; excessive prefetch wastes RAM/cache. Keep enough local disk free for uncompressed shards and the number of concurrently active shards. More workers or bigger shards are not always faster.

```bash
python -B -m unittest discover -s scripts -p 'test_*.py' -v
python scripts/convert_to_mds.py --help
python scripts/benchmark_mds.py --help
```

MDS tests use temporary synthetic images, including 16-bit and big-endian pixels; they check byte/pixel round trips, shard rollover, metadata preservation, overwrite protection, incomplete output, cache safety, and reader behavior. Run them with the ML dependencies installed for full coverage; integration tests may be skipped without those packages. The completed real-data `cancer_dataset_mds_v3` manifest records 54 shards and 54,687 retained images, with 19 missing inputs audited. Training metadata reconstruction has been checked against that manifest, but no real-data training score or cluster-wide training throughput has been measured.

References: [Databricks Mosaic Streaming guide](https://docs.databricks.com/aws/en/machine-learning/load-data/streaming), [MDSWriter](https://docs.mosaicml.com/projects/streaming/en/stable/api_reference/generated/streaming.MDSWriter.html), and [StreamingDataset](https://docs.mosaicml.com/projects/streaming/en/stable/api_reference/generated/streaming.StreamingDataset.html).

## Runnable training: simple and advanced

Both entry points use the existing `cancer_dataset_mds_v3` by default and require the same ML environment as the MDS reader, plus PyTorch, torchvision, NumPy, Pillow, and tqdm. No dependencies are installed by the scripts. `--pretrained` is enabled by default and may download official torchvision ImageNet weights; use `--no-pretrained` for offline/synthetic verification, not as an equivalent pretrained baseline.

| Entry point | Training target and model | Starting defaults |
| --- | --- | --- |
| `scripts/train_simple.py` | One image at a time, ResNet-18; mean probability pooling per breast for validation | 512-pixel square canvas, batch 16, 5 epochs, AdamW, fixed learning rate |
| `scripts/train_advanced.py` | One breast at a time, shared ConvNeXt-Tiny encoder and gated attention over its views | 1024-pixel canvas, batch 1 breast, up to 2 randomly selected training views, accumulation 8, 10 epochs, AMP, cosine LR, EMA |
| `scripts/train_vit.py` | Same breast-level attention protocol with a ViT-B/16 encoder, for a CNN-vs-transformer comparison | 384-pixel canvas (multiple of 16), learning rate 3e-5, otherwise identical to the advanced defaults |

Run the simple baseline first, from the repository root:

```bash
python -u scripts/train_simple.py \
  --cache /local_disk0/rsna_training/simple_fold0 \
  --out /Volumes/daai_ke_team/default/images/cancer_dataset/train_simple_v1 \
  --device cuda:0 --epochs 5
```

Then compare the advanced model on the same validation patients:

```bash
python -u scripts/train_advanced.py \
  --cache /local_disk0/rsna_training/advanced_fold0 \
  --out /Volumes/daai_ke_team/default/images/cancer_dataset/train_advanced_v1 \
  --device cuda:0 --image-size 1024 --batch-size 1 --grad-accum 8 --epochs 10
```

Output must be a **new** directory whose parent already exists. Dataset, run output, original source images, and cache must not overlap. Put run outputs on a persistent Volume and cache on node-local storage. Use a distinct cache/output for unrelated concurrent runs. The default `--cache-limit 20gb` accommodates the current roughly 14 GB shard dataset, subject to other disk usage; cache warmup and PNG decoding still consume I/O/CPU. Keep extra local space for temporary checkpoint serialization.

If GPU memory is insufficient, reduce image size, batch size, or the breast-level training view cap; increasing accumulation preserves an approximate effective batch size without increasing simultaneous activations. The advanced single-GPU default effective batch is 8 breasts, except for a smaller final accumulation group. Validation encoder calls are chunked with `--view-chunk-size 2`; no validation views are truncated.

### Multiple GPUs with torchrun

Each script runs single-process on one GPU by default and becomes data-parallel (DDP) under `torchrun`, with no code changes:

```bash
python -m torch.distributed.run --nnodes 1 --nproc-per-node 4 \
  scripts/train_advanced.py \
  --cache /local_disk0/rsna_training/advanced_fold0 \
  --out /Volumes/daai_ke_team/default/images/cancer_dataset/train_advanced_v1 \
  --epochs 10
```

- One process per GPU; each rank takes its GPU from `LOCAL_RANK`, so **do not** pass `--device cuda:N` under `torchrun` (it is rejected). `--device cpu` forces the gloo backend, which is only useful for debugging.
- The **effective batch multiplies by the number of GPUs**: `batch-size × grad-accum × world size`. The four-GPU advanced default is 32 breasts per step, and `config.json` records it as `effective_batch_breasts_or_images`. Retune `--lr` when you change the effective batch; the scripts do not scale the learning rate automatically.
- `--num-workers` is **per process**, so total loader processes and cache pressure scale with GPU count. Ranks on one node share the `--cache` directory, as Mosaic Streaming expects; do not give ranks different cache paths, and do not share a cache between unrelated concurrent runs.
- Gradients are averaged over the global sample count, and gradient accumulation avoids inter-GPU syncing until each accumulation boundary. Training loss is all-reduced across ranks.
- **Validation still covers the whole fold exactly once.** Each rank scores a shard of the held-out fold; predictions are gathered, and the duplicates that distributed sampling pads with are removed by `(prediction_id, sample_ids)` before breast aggregation. The run fails rather than reporting a score if the deduplicated count does not equal the held-out sample count. Reported validation loss still includes padded duplicates, so treat it as approximate and prefer the metrics.
- Only rank 0 writes `config.json`, fold/label CSVs, predictions, metrics, and checkpoints, so Volume outputs are written once. Checkpoints contain unwrapped weights and load without DDP.
- Multi-GPU numerics are not bit-identical to a single-GPU run; batch composition, reduction order, and cuDNN kernels differ. Compare runs at matching effective batch sizes rather than assuming equality. At shutdown, Mosaic Streaming may print harmless leaked-shared-memory warnings.
- Verified here: 2-process CPU/gloo and 2- and 4-GPU NCCL runs on a synthetic dataset produced full-fold validation with identical folds/labels to the single-process run. No multi-GPU throughput or accuracy result on the real dataset is claimed.

### Vision Transformer experiment

`scripts/train_vit.py` shares the breast-level dataset, attention pooling head, folds, loss, metrics, and checkpoint logic with `train_advanced.py`, changing only the encoder. That is what makes it an encoder comparison rather than a pipeline comparison:

```bash
python -m torch.distributed.run --nnodes 1 --nproc-per-node 4 \
  scripts/train_vit.py \
  --cache /local_disk0/rsna_training/vit_fold0 \
  --out /Volumes/daai_ke_team/default/images/cancer_dataset/train_vit_v1 \
  --image-size 384 --epochs 10
```

`--image-size` must be a multiple of the 16-pixel patch size and is rejected otherwise. Official ViT-B/16 weights are trained at 224, so any other size loads those weights with **bicubic-interpolated position embeddings** (`torchvision.models.vision_transformer.interpolate_embeddings`) and a freshly initialized head; the interpolation is a standard adaptation, not an equivalent pretrained model, and mismatched weights fail loudly instead of loading partially.

Expect ViTs to be more sensitive than the CNNs here: attention cost grows quadratically with token count, so 384 pixels already means 576 patches, and higher resolutions increase memory quickly. The default learning rate is lower (3e-5) because ViTs typically need it, and transformers usually need more data or stronger augmentation than this ImageNet-pretrained fine-tuning recipe provides. **Whether ViT beats ConvNeXt on this dataset is an open question that only your measured results can answer**; a fair comparison needs the same fold, seed, epochs, and comparable effective batch, and ideally several folds and seeds.

### Comparing runs

```bash
python scripts/compare_runs.py \
  /Volumes/.../train_simple_v1 /Volumes/.../train_advanced_v1 /Volumes/.../train_vit_v1 \
  --baseline advanced --bootstrap 1000
```

The report ranks completed runs by best-epoch breast-level pF1 alongside ROC-AUC, AP, Brier score, F1 at 0.5, best epoch, and training minutes, and it recomputes pF1 from each run's saved prediction CSV, failing if that disagrees with the reported metric. It refuses unfinished runs (no `_TRAINING_SUCCESS`) and warns instead of silently ranking when runs used a different dataset hash, fold, seed, image size, epoch count, pretraining flag, positive weight, debug batch cap, or validation breast set.

`--bootstrap` resamples **patients** (both breasts kept together) to report the mean pF1 difference against the baseline, a 95% interval, and whether that interval crosses zero. An interval crossing zero means the data do not establish a difference; it does not prove the models are equivalent. This single-fold, single-seed comparison cannot separate encoder quality from run-to-run variance, and pF1 differences on roughly 100 positive breasts per fold are noisy, so repeat across folds/seeds before concluding that one encoder is better.

### Leakage controls, preprocessing, and metrics

- Both scripts use deterministic, patient-disjoint folds (`--folds 5 --fold 0 --seed 42`), stratified by whether a patient has any positive breast. Both breasts and all views stay in one fold. Run one fold per invocation; change `--fold` to 1..4 for the remaining folds. The verified default fold 0 has 2,384 validation patients, 11,014 images, 4,768 breasts, and 101 positive breasts in the current retained dataset.
- No re-sharding is required for these single-node trainers. They reconstruct exact global MDS indices from the original CSV, conversion seed, limit, and skipped-image audit, verify the CSV hash and manifest counts, then apply disjoint map-style subsets over one cached MDS reader. Every fetched sample's ID, patient, breast ID, and label is checked against the reconstructed metadata. `--csv` may point to an identical copy of the original CSV; original PNG files are not needed for training. Use datasets created by the current converter with its full audit metadata.
- Only pixels and a valid-view mask reach the model. Cancer labels are targets; train-only CSV metadata is not used as an input feature.
- Images are resized **in memory** with aspect-preserving padding; the source PNG/MDS files remain unchanged. Grayscale is repeated into three channels, then ImageNet mean/std normalization is applied. Unsigned 8-bit and 16-bit pixels use 255 and 65535 respectively. Set `--intensity-max 4095` only when the source really contains 12-bit intensities; inspect your preprocessing rather than assuming this. Float inputs must already be in [0,1] unless an explicit maximum is provided. Unsupported modes/ranges fail instead of silently coercing them.
- Training uses modest flips/rotations without random crops. Validation transforms are deterministic. The advanced training cap samples views randomly each epoch; validation always pools every available view from the breast. The 19 previously skipped images cannot be recovered by training and remain absent.
- Both use BCE-with-logits. The default `--pos-weight auto` is `min(20, max(1, sqrt(negative/positive)))`, calculated from training targets only: image targets for the simple model, breast targets for the advanced model. There is no oversampling. Use `--pos-weight 1` to compare unweighted BCE. Reweighting changes the effective prior; outputs are not guaranteed calibrated.
- Validation runs over the **entire held-out fold**, without balancing or a batch cap. Checkpoint selection uses raw breast-level pF1; the scripts also report tie-aware ROC-AUC, average precision, Brier score, and fixed-0.5 precision/recall/F1. Mean breast pooling is the simple default; `--pooling max` is an explicit ablation. Advanced attention emits one breast probability. The constant training-breast-prevalence predictor is recorded for comparison. Thresholds are not optimized on validation results automatically.
- Advanced validation and best-checkpoint selection use EMA weights by default (`--ema-decay 0.999`); zero disables EMA. AMP prefers bfloat16 on supported CUDA hardware and otherwise uses scaled float16. CPU runs use float32. Seeded runs are reproducible in their folds/sampling setup, not guaranteed bit-identical across hardware or library versions.

### Training outputs and verification

Each run saves `config.json` (dataset hashes, settings, split counts, and effective loss weight), `folds.csv`, `validation_labels.csv`, per-epoch validation prediction CSVs and metrics, `metrics.jsonl`, `best_predictions.csv`, `epoch_NNN.pt`, `best.pt`, and `last.pt`. The prediction CSVs work with the existing metric CLI. `_TRAINING_SUCCESS` is written only after all requested epochs finish.

Every epoch is kept as its own `epoch_NNN.pt` containing that epoch's raw weights, EMA weights when enabled, epoch number, validation metrics, and config, so any epoch can be reloaded or re-evaluated later rather than only the best and final ones. `best.pt` holds the validation-selected weights (EMA when enabled) and `last.pt` additionally holds optimizer, scheduler, scaler, and EMA state for the latest epoch. Per-epoch files therefore deliberately omit optimizer state; sizes are roughly 45 MB per epoch for ResNet-18, 110 MB for ConvNeXt-Tiny, and 330 MB for ViT-B/16, doubled when EMA is on, so a 10-epoch ViT run needs several gigabytes on the Volume. Use `--no-save-epochs` to keep only `best.pt`/`last.pt`.

These are **epoch-boundary checkpoints**; automatic resume, exact RNG/sampler restoration, and an inference CLI are not implemented. An interrupted current epoch must be rerun in any future continuation implementation. Existing run directories are never reused by the training CLI.

To support Unity Catalog Volume limitations, checkpoint ZIP serialization happens in a temporary local file followed by a sequential copy; metric logs are rewritten, not appended. Uploads are not an atomic transaction on cloud storage. Keep both best/last checkpoints and verify a checkpoint loads before relying on it after an interruption. For unattended execution, use a Databricks Job rather than relying on a laptop-connected foreground SSH terminal.

```bash
python -B scripts/train_simple.py --help
python -B scripts/train_advanced.py --help
python -B scripts/train_vit.py --help
python -B scripts/compare_runs.py --help
python -B -m unittest discover -s scripts -p 'test_*.py' -v
```

Training tests include real ResNet-18, ConvNeXt-Tiny, and ViT-B/16 CPU forward/backward passes and one synthetic MDS training epoch per approach, with `--no-pretrained`, 64-pixel inputs, and no external downloads. They also exercise a real two-process `torchrun` run that must reproduce the single-process folds and full-fold validation, spawned MDS loader workers, fold isolation, skipped-image ordering, 16-bit scaling, attention masks, ViT position-embedding interpolation, gradient accumulation, EMA, metrics, checkpoint reloads, and the comparison report's comparability guards. `--max-train-batches` is a debug-only cap on training, not validation; do not compare such runs as fully trained models. Passing these tests demonstrates pipeline behavior, not breast-cancer detection performance.

## Evaluation protocol for trustworthy experiments

1. **Split by patient, never by image or breast alone.** Both breasts and all views of a patient must remain in one fold. Start with a fixed patient-held-out split for debugging; use 5 patient-grouped folds for reliable comparison if compute permits. Stratify patient records by whether either breast is positive, then map folds back to images. Alternatively, use `StratifiedGroupKFold` with `patient_id` groups and inspect the resulting breast-level prevalence.
2. Inspect positives/negatives, age, site, and machine distributions per fold. Keep validation at natural prevalence; do not oversample or balance it.
3. Accumulate all validation predictions, aggregate by breast, then score. Do not average minibatch pF1 values: pF1 is non-additive. For distributed evaluation, gather all rows and remove sampler-padding duplicates before aggregation.
4. Save out-of-fold (OOF) predictions and report both pooled breast-level OOF pF1 and each fold's score. Fold-average pF1 and pooled pF1 need not match.
5. Track raw pF1, average precision (AP), ROC-AUC, and binary precision/recall/F1 at a fixed operating threshold. AP and trapezoidal PR-AUC are not interchangeable; label the chosen definition explicitly. The training scripts implement breast-level pF1, ROC-AUC, AP, Brier score, and precision/recall/F1 at 0.5; add further prespecified clinical operating-point analysis separately.
6. Compare mean/max view pooling and any threshold or probability transformation on OOF predictions. Freeze the chosen policy before final evaluation. The maximum threshold-tuned F1 on the same OOF data is a model-selection result, not an unbiased final score; use a separate patient-held-out set or nested/cross-fitted tuning for that claim. Do not assume 0.5 is optimal.
7. Fit calibration, if needed, on held-out predictions using a proper loss such as log loss; assess log loss/Brier score and reliability separately from pF1. Avoid using the public leaderboard as a repeated tuning set.
8. For uncertainty, bootstrap patients rather than images, retaining both breasts per sampled patient. Compare subgroup performance across sites, machines, ages, and implants, with sample counts and uncertainty.
9. Record fold assignments, seeds, dataset/preprocessing version, checkpoint, input size, pooling rule, postprocessing, runtime, and memory usage. Change one major component at a time.

## Modeling options: simple to advanced

These are candidate experiments; none guarantees an improvement. Input resolutions are starting points, not fixed best settings. Larger inputs preserve subtle findings but increase compute and can overfit.

| Level | Approach | How to use it effectively | Main trade-off |
| --- | --- | --- | --- |
| Sanity check | Constant prevalence predictor; optionally logistic regression on test-available metadata | Verify the pipeline, leakage checks, and score calculation before image training | Not a serious image-detection solution; metadata can exploit site shortcuts |
| Simple image baseline | ImageNet-pretrained ResNet-18 or EfficientNet-B0, one mammogram at a time | Breast foreground crop, aspect-preserving resize/pad around 512 pixels on the long side, BCE-with-logits, mean breast pooling | Inexpensive iteration; small calcifications may disappear |
| Strong single model | ConvNeXt-Tiny/Small or EfficientNetV2-S at higher resolution | Compare roughly 1024-2048 pixels on the long side, progressive resizing, mixed precision, controlled imbalance handling, grouped CV | More GPU memory and slower inference; crop quality matters |
| Multi-view breast model | Shared encoder for CC/MLO views with concatenation or attention pooling | Predict one breast-level output, use masks for missing/extra views, compare against simple probability pooling | Better use of view context is plausible, but batching and missing-view handling are harder |
| Global + local model | Whole-breast encoder plus high-resolution patch multiple-instance learning (MIL) | Combine global context with local detail; attention-pool patches without assuming every patch in a positive breast contains cancer | Higher engineering/compute cost; patch selection can miss lesions |
| Domain-pretrained model | Mammography-pretrained encoder followed by target-data fine-tuning | Audit checkpoint provenance, license, dataset overlap, and label compatibility; compare against ImageNet on identical folds | Potential transfer benefit, but contaminated pretraining invalidates validation |
| Advanced ensemble | Diverse high-resolution/multi-view/MIL models across folds | Average aligned breast probabilities, learn only modest blend/postprocessing choices on OOF data, retain models with complementary errors | Expensive; additional models may add little and may exceed inference limits |

A learned breast-region detector is another preprocessing option after a simple foreground crop works. It needs annotations or appropriately licensed pretrained weights; the cancer label alone is not a lesion box. Use a full-image fallback for failed/unsafe crops.

### Training and preprocessing priorities

- Inspect representative DICOMs across sites and machines before committing to conversion. Support required transfer syntaxes, including JPEG 2000; handle photometric interpretation (`MONOCHROME1` inversion), pixel padding, and appropriate LUT/windowing where applicable. Check final image appearance rather than blindly applying every transform.
- Retain a lossless/high-bit-depth source cache where feasible. Crop background conservatively, preserve aspect ratio, and check that breast edges and subtle lesions are not removed. Use identical preprocessing for validation and inference.
- Start with BCE-with-logits and a simple pretrained encoder. Compare moderate positive oversampling **or** a positive loss weight; do not automatically combine aggressive versions of both. These methods alter the effective training prior, so validate output postprocessing at natural prevalence.
- Consider focal loss only as an ablation after the baseline. A more complicated loss is not automatically better for pF1.
- Use modest, anatomy-preserving augmentation. Avoid crops/downsampling that erase tiny findings. If using left/right context, keep orientation and laterality metadata consistent under transforms.
- Use mixed precision and gradient accumulation if supported by the chosen GPU. Tune effective batch size and learning rate together; select checkpoints using held-out breast-level results, with AP as a useful secondary stability signal.
- Add age/implant or other test-available metadata only after establishing an image baseline. Treat site/machine IDs cautiously because they can encourage non-generalizing shortcuts.
- External mammography data and pretrained weights require provenance, licensing, overlap checks, and breast-level label harmonization. For historical competition reproduction, also restrict resources to what was eligible at the original deadline; newer foundation models belong to a separate modern research track.

### Recommended experiment sequence

1. Build verified DICOM conversion, patient folds, and a 512-pixel EfficientNet-B0 baseline with mean breast pooling.
2. Improve crop/normalization quality, compare pooling and imbalance strategies, and collect reproducible OOF predictions.
3. Compare a higher-resolution ConvNeXt or EfficientNetV2 on the same folds. Test resolution increases before simply increasing parameter count.
4. Add multi-view fusion or global/local MIL only if error analysis shows a need for complementary views or lost fine detail.
5. Ensemble complementary models, select postprocessing on OOF data, and perform one final untouched evaluation.

The published first-place solution is a useful reference for ROI preprocessing, ConvNeXt models, external data, and efficient inference. It is evidence for a strong engineering path, not proof that every component will improve this repository's eventual model.

## Submission and research scope

The archived original code-competition requirements specified notebook submissions, internet disabled, a maximum nine-hour CPU or GPU notebook runtime, and output named `submission.csv`. Public external data/pretrained models were allowed under the rules. Verify current platform behavior and the applicable rules rather than assuming historical limits are unchanged.

The inference notebook must process the actual hidden test metadata, not hard-code the small visible test sample. Emit exactly one row per required `prediction_id`, ensure finite probabilities in `[0, 1]`, and align to the supplied sample submission. Benchmark preprocessing plus inference plus ensembling end to end; offline scoring here does not reproduce Kaggle's hidden test environment.

This repository supports competition research, not clinical diagnosis. A high pF1 does not establish clinical safety or transportability; independent external validation and clinically appropriate operating-point assessment are separate work.

## Sources and verification scope

- [Competition overview and evaluation](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/overview)
- [Organizer-linked probabilistic F-score implementation](https://www.kaggle.com/code/sohier/probabilistic-f-score)
- [Archived competition description, evaluation, data schema, and original code requirements](https://raw.githubusercontent.com/openai/mle-bench/main/mlebench/competitions/rsna-breast-cancer-detection/description.md)
- [First-place solution write-up](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/writeups/mr-robot-1st-place-solution)
- [First-place implementation repository](https://github.com/dangnh0611/kaggle_rsna_breast_cancer/)

Kaggle's dynamic pages did not expose readable content in the setup environment. The metric and submission requirements were checked against the archived description and corroborating public implementations. The local implementation matches the published formula for valid finite binary-label inputs; it has not been executed against Kaggle's private scoring service.
