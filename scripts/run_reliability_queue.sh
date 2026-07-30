#!/bin/bash
# NEW work only (no repeats of seed-42 runs):
#  - multi-seed reliability for the two top single models (baseline, ConvNeXt)
#  - EfficientNet-B0 (new supervised encoder)
# Launch:  tmux new -d -s relq 'bash run_reliability_queue.sh'
cd /home/sb2597/autofish_baseline_repro
source .venv/bin/activate
mkdir -p queue_logs
LOG=queue_logs/reliability_queue.log

run () {
  NAME=$1; CFG=$2
  if [ -f "runs/$NAME/test_metrics.json" ]; then echo "[$(date)] SKIP $NAME (exists)" >> $LOG; return; fi
  echo "[$(date)] START $NAME" >> $LOG
  /usr/bin/time -v python -m src.autofish_vfm.train_baseline \
    --config configs/$CFG --index data/processed/index.csv \
    --crops-dir data/processed/crops --out-dir runs/$NAME > queue_logs/$NAME.train.log 2>&1
  if [ $? -ne 0 ]; then echo "[$(date)] TRAIN FAILED $NAME" >> $LOG; return; fi
  # test metrics
  python -m src.autofish_vfm.evaluate --checkpoint runs/$NAME/best.pt --config configs/$CFG \
    --index data/processed/index.csv --crops-dir data/processed/crops \
    --out runs/$NAME/test_metrics.json > queue_logs/$NAME.eval.log 2>&1
  # validation predictions (for ensembling)
  python -m src.autofish_vfm.evaluate --checkpoint runs/$NAME/best.pt --config configs/$CFG \
    --index data/processed/index.csv --crops-dir data/processed/crops \
    --split val --out runs/$NAME/val_metrics.json >> queue_logs/$NAME.eval.log 2>&1
  echo "[$(date)] DONE $NAME" >> $LOG
}

run efficientnet_b0        efficientnet_b0.json
run convnext_tiny_s1       convnext_tiny_s1.json
run convnext_tiny_s2       convnext_tiny_s2.json
run baseline_official_s1   baseline_official_s1.json
run baseline_official_s2   baseline_official_s2.json

echo "[$(date)] RELIABILITY QUEUE COMPLETE" >> $LOG
