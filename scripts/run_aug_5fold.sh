#!/usr/bin/env bash
# Experiment: fix overfitting on the champion recipe and measure it over 5 folds.
#
# Champion recipe held fixed (fold-0 pF1 0.185 / AUC 0.795):
#   ConvNeXt-Tiny, 512 px, ROI crop + canonical side, positive fraction 0.2,
#   pos_weight 1, lr 2e-5, dropout 0.35, batch 8, EMA 0.999.
#
# The three changes under test:
#   --augment strong      scale/shift, +/-15 deg, gain/gamma jitter, coarse dropout
#   --epochs 8            cosine T_max follows --epochs, so the anneal lands on the peak
#   --select-metric average_precision   pF1 selection chases calibration drift
#
# Control arm (AUG=light EPOCHS=8) isolates augmentation from the shorter schedule.
#
# Usage:
#   scripts/run_aug_5fold.sh                         # folds 0-4, strong augmentation
#   FOLDS="0" AUG=light PREFIX=t2ctl ./scripts/...    # control arm
set -u -o pipefail

REPO=${REPO:-/root/brest_cancer_detection}
MDS=${MDS:-/Volumes/daai_ke_team/default/images/cancer_dataset/shrads/cancer_dataset_mds_v3}
RUNS=${RUNS:-/Volumes/daai_ke_team/default/images/cancer_dataset/runs}
ROI_BOXES=${ROI_BOXES:-/root/roi/roi_boxes_v3.csv}
LOGS=${LOGS:-/root/t2_logs}
CACHE=${CACHE:-/tmp/t2_cache}

FOLDS=${FOLDS:-"0 1 2 3 4"}
AUG=${AUG:-strong}
EPOCHS=${EPOCHS:-8}
SELECT_METRIC=${SELECT_METRIC:-average_precision}
IMAGE_SIZE=${IMAGE_SIZE:-512}
BATCH_SIZE=${BATCH_SIZE:-8}
GRAD_ACCUM=${GRAD_ACCUM:-1}
GPUS=${GPUS:-4}
LR=${LR:-2e-5}
PREFIX=${PREFIX:-t2_cnn_roibal_aug${AUG}}
WANDB_PROJECT=${WANDB_PROJECT:-Brest-Cancer-Experiments}
WANDB_ENTITY=${WANDB_ENTITY:-yashchks87}
USE_WANDB=${USE_WANDB:-1}
# Abort a fold early if it never learns, as the lr 1e-4 ladder did (epoch-1 AUC 0.40).
ABORT_EPOCH=${ABORT_EPOCH:-3}
ABORT_AUC=${ABORT_AUC:-0.62}

mkdir -p "$LOGS"
[ -f "$ROI_BOXES" ] || { echo "Missing ROI box cache: $ROI_BOXES"; echo "Build it with scripts/compute_roi_boxes.py"; exit 1; }

if [ "$USE_WANDB" = "1" ] && [ -z "${WANDB_API_KEY:-}" ] && [ ! -f "$HOME/.netrc" ]; then
  echo "wandb is enabled but no credentials found. Run 'wandb login' or export WANDB_API_KEY, or set USE_WANDB=0."
  exit 1
fi

COMPLETED=()
for FOLD in $FOLDS; do
  NAME="${PREFIX}_img${IMAGE_SIZE}_e${EPOCHS}_f${FOLD}"
  OUT="$RUNS/$NAME"
  if [ -e "$OUT" ]; then echo "SKIP $NAME (exists)"; COMPLETED+=("$OUT"); continue; fi

  WANDB_FLAGS=""
  if [ "$USE_WANDB" = "1" ]; then
    WANDB_FLAGS="--wandb --wandb-project $WANDB_PROJECT --wandb-entity $WANDB_ENTITY \
                 --wandb-tags tier2 cnn roi_bal aug${AUG} img${IMAGE_SIZE} fold${FOLD}"
  fi

  echo "=== START $NAME $(date -Is) ==="
  # shellcheck disable=SC2086
  ( cd "$REPO" && torchrun --standalone --nnodes 1 --nproc-per-node "$GPUS" \
      scripts/train_advanced.py --encoder convnext_tiny \
      --mds "$MDS" \
      --image-size "$IMAGE_SIZE" --batch-size "$BATCH_SIZE" --grad-accum "$GRAD_ACCUM" \
      --lr "$LR" --dropout 0.35 --weight-decay 0.05 \
      --epochs "$EPOCHS" --ema-decay 0.999 --fold "$FOLD" --folds 5 \
      --roi-crop --roi-boxes "$ROI_BOXES" --canonical-side left \
      --positive-fraction 0.2 --pos-weight 1 \
      --augment "$AUG" --select-metric "$SELECT_METRIC" \
      --view-chunk-size 2 --num-workers 2 --no-progress \
      --cache "$CACHE" --out "$OUT" \
      $WANDB_FLAGS ) < /dev/null > "$LOGS/$NAME.log" 2>&1
  STATUS=$?
  echo "=== END $NAME rc=$STATUS $(date -Is) ==="
  if [ -f "$OUT/_TRAINING_SUCCESS" ]; then
    COMPLETED+=("$OUT")
  else
    echo "FOLD $FOLD did not finish; see $LOGS/$NAME.log"
    continue
  fi

  AUC=$(python3 -c "
import json,sys
rows=[json.loads(l) for l in open('$OUT/metrics.jsonl') if l.strip()]
row=next((r for r in rows if r['epoch']==$ABORT_EPOCH), None)
print(row['validation']['roc_auc'] if row else 1.0)
" 2>/dev/null || echo 1.0)
  if python3 -c "import sys; sys.exit(0 if float('$AUC') < float('$ABORT_AUC') else 1)"; then
    echo "STOP: fold $FOLD epoch-$ABORT_EPOCH AUC $AUC < $ABORT_AUC; the recipe is not training. Investigate before burning more GPU time."
    break
  fi
done

if [ "${#COMPLETED[@]}" -gt 1 ]; then
  REPORT="$LOGS/${PREFIX}_oof_$(date +%Y%m%d_%H%M%S)"
  echo "=== OOF REPORT -> ${REPORT}.json ==="
  ( cd "$REPO" && python3 -B scripts/oof_report.py "${COMPLETED[@]}" \
      --out "${REPORT}.json" --predictions-out "${REPORT}_predictions.csv" ) | tee "$LOGS/oof_stdout.txt"
fi
