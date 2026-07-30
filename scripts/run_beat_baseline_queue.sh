#!/bin/bash
# Goal: get a SINGLE model below the baseline (0.771 cm) via stronger recipes.
# Validation-based selection: each run saves history (val MAE per epoch) + test metrics.
# We pick the winner by lowest validation MAE, then report its test MAE (no test-tuning).
# Order: strongest candidates first (EfficientNet 0.781 @ weak recipe, then ConvNeXt),
# then the multi-seed reliability runs.
cd /home/sb2597/autofish_baseline_repro
source .venv/bin/activate
mkdir -p queue_logs
LOG=queue_logs/beat_baseline_queue.log

run () {
  NAME=$1; CFG=$2
  if [ -f "runs/$NAME/test_metrics.json" ]; then echo "[$(date)] SKIP $NAME (exists)" >> $LOG; return; fi
  echo "[$(date)] START $NAME" >> $LOG
  python -m src.autofish_vfm.train_baseline \
    --config configs/$CFG --index data/processed/index.csv \
    --crops-dir data/processed/crops --out-dir runs/$NAME > queue_logs/$NAME.train.log 2>&1
  if [ $? -ne 0 ]; then echo "[$(date)] TRAIN FAILED $NAME" >> $LOG; return; fi
  python -m src.autofish_vfm.evaluate --checkpoint runs/$NAME/best.pt --config configs/$CFG \
    --index data/processed/index.csv --crops-dir data/processed/crops \
    --out runs/$NAME/test_metrics.json > queue_logs/$NAME.eval.log 2>&1
  python -m src.autofish_vfm.evaluate --checkpoint runs/$NAME/best.pt --config configs/$CFG \
    --index data/processed/index.csv --crops-dir data/processed/crops \
    --split val --out runs/$NAME/val_metrics.json >> queue_logs/$NAME.eval.log 2>&1
  echo "[$(date)] DONE $NAME" >> $LOG
}

# --- beat-baseline attempts (strongest candidates first) ---
run efficientnet_b0_strong_a  efficientnet_b0_strong_a.json
run efficientnet_b0_strong_b  efficientnet_b0_strong_b.json
run efficientnet_b0_strong_c  efficientnet_b0_strong_c.json
run convnext_tiny_strong_a    convnext_tiny_strong_a.json
run convnext_tiny_strong_b    convnext_tiny_strong_b.json

# --- reliability (multi-seed) after the priority attempts ---
run convnext_tiny_s1       convnext_tiny_s1.json
run convnext_tiny_s2       convnext_tiny_s2.json
run baseline_official_s1   baseline_official_s1.json
run baseline_official_s2   baseline_official_s2.json

echo "[$(date)] BEAT-BASELINE QUEUE COMPLETE" >> $LOG
