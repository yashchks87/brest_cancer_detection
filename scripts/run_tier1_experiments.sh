#!/usr/bin/env bash
# Tier-1 A/B: ROI cropping, positive oversampling and auxiliary heads, for both
# ViT-B/16 and ConvNeXt-Tiny.
#
# The variants form a staircase so each feature can be attributed separately
# instead of only learning whether the bundle helps:
#
#   baseline      current best recipe (lr 2e-5, pos_weight 3, dropout 0.35)
#   roi           + --roi-crop --canonical-side left
#   roi_bal       + --positive-fraction 0.2 --pos-weight 1
#   roi_bal_aux   + --aux-targets biopsy invasive
#
# Both architectures run at lr 2e-5. The previous CNN ladder used 1e-4, which
# never trained (epoch-1 AUC 0.40), so its results are not a usable baseline.
#
# Usage:
#   scripts/run_tier1_experiments.sh                      # both arches, all variants, fold 0
#   ARCHES=vit VARIANTS="baseline roi" ./scripts/...      # subset
#   FOLDS="0 1 2" ARCHES=vit VARIANTS=roi_bal_aux ./...   # multi-fold confirmation
set -u -o pipefail

REPO=${REPO:-/root/brest_cancer_detection}
MDS=${MDS:-/Volumes/daai_ke_team/default/images/cancer_dataset/shrads/cancer_dataset_mds_v3}
RUNS=${RUNS:-/Volumes/daai_ke_team/default/images/cancer_dataset/runs}
ROI_BOXES=${ROI_BOXES:-/root/roi/roi_boxes_v3.csv}
LOGS=${LOGS:-/root/tier1_logs}
CACHE=${CACHE:-/tmp/tier1_cache}

ARCHES=${ARCHES:-"vit cnn"}
VARIANTS=${VARIANTS:-"baseline roi roi_bal roi_bal_aux"}
FOLDS=${FOLDS:-"0"}
EPOCHS=${EPOCHS:-14}
IMAGE_SIZE=${IMAGE_SIZE:-512}
BATCH_SIZE=${BATCH_SIZE:-8}
GRAD_ACCUM=${GRAD_ACCUM:-1}
GPUS=${GPUS:-4}
LR=${LR:-2e-5}
PREFIX=${PREFIX:-t1}
WANDB_PROJECT=${WANDB_PROJECT:-Brest-Cancer-Experiments}
WANDB_ENTITY=${WANDB_ENTITY:-yashchks87}
USE_WANDB=${USE_WANDB:-1}

mkdir -p "$LOGS"
[ -f "$ROI_BOXES" ] || { echo "Missing ROI box cache: $ROI_BOXES"; echo "Build it with scripts/compute_roi_boxes.py"; exit 1; }

variant_flags() {
  case "$1" in
    baseline)    echo "--pos-weight 3" ;;
    roi)         echo "--pos-weight 3 --roi-crop --roi-boxes $ROI_BOXES --canonical-side left" ;;
    roi_bal)     echo "--pos-weight 1 --roi-crop --roi-boxes $ROI_BOXES --canonical-side left --positive-fraction 0.2" ;;
    roi_bal_aux) echo "--pos-weight 1 --roi-crop --roi-boxes $ROI_BOXES --canonical-side left --positive-fraction 0.2 --aux-targets biopsy invasive --aux-weight 0.3" ;;
    *) echo "UNKNOWN" ;;
  esac
}

for FOLD in $FOLDS; do
for ARCH in $ARCHES; do
for VARIANT in $VARIANTS; do
  FLAGS=$(variant_flags "$VARIANT")
  [ "$FLAGS" = "UNKNOWN" ] && { echo "Unknown variant: $VARIANT"; exit 1; }

  if [ "$ARCH" = "vit" ]; then
    SCRIPT=scripts/train_vit.py; ENCODER_FLAG=""
  else
    SCRIPT=scripts/train_advanced.py; ENCODER_FLAG="--encoder convnext_tiny"
  fi

  NAME="${PREFIX}_${ARCH}_${VARIANT}_img${IMAGE_SIZE}_f${FOLD}"
  OUT="$RUNS/$NAME"
  if [ -e "$OUT" ]; then echo "SKIP $NAME (exists)"; continue; fi

  WANDB_FLAGS=""
  if [ "$USE_WANDB" = "1" ]; then
    WANDB_FLAGS="--wandb --wandb-project $WANDB_PROJECT --wandb-entity $WANDB_ENTITY \
                 --wandb-tags tier1 $ARCH $VARIANT img${IMAGE_SIZE} fold${FOLD}"
  fi

  echo "=== START $NAME $(date -Is) ==="
  # shellcheck disable=SC2086
  ( cd "$REPO" && torchrun --standalone --nnodes 1 --nproc-per-node "$GPUS" \
      "$SCRIPT" $ENCODER_FLAG \
      --mds "$MDS" \
      --image-size "$IMAGE_SIZE" --batch-size "$BATCH_SIZE" --grad-accum "$GRAD_ACCUM" \
      --lr "$LR" --dropout 0.35 --weight-decay 0.05 \
      --epochs "$EPOCHS" --ema-decay 0.999 --fold "$FOLD" --folds 5 \
      --view-chunk-size 2 --num-workers 2 --no-progress \
      --cache "$CACHE" --out "$OUT" \
      $FLAGS $WANDB_FLAGS ) < /dev/null > "$LOGS/$NAME.log" 2>&1
  echo "=== END $NAME rc=$? $(date -Is) ==="
done
done
done
