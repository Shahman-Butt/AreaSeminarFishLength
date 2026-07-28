#!/bin/bash
# Persistent classification queue. Waits for the current patch-token regression
# run to finish, then trains/evaluates species classifiers one by one.
cd /home/sb2597/autofish_baseline_repro
source .venv/bin/activate
mkdir -p queue_logs

echo "[$(date)] CLASSIFICATION QUEUE WAITING FOR PATCH-TOKEN RUN" >> queue_logs/classification_queue.log
while pgrep -af "dinov2_vits14_patchtokens_frozen" >/dev/null; do
  sleep 120
done
echo "[$(date)] CLASSIFICATION QUEUE START" >> queue_logs/classification_queue.log

python - <<'PY'
import json
c = json.load(open("configs/cls_mobilenet_v2.json"))
c["epochs"] = 1
c["max_train_batches"] = 3
c["max_val_batches"] = 3
c["batch_size"] = 4
c["num_workers"] = 2
json.dump(c, open("configs/_cls_smoke.json", "w"), indent=2)
PY

python -m src.autofish_vfm.train_classifier \
  --config configs/_cls_smoke.json \
  --index data/processed/index.csv \
  --crops-dir data/processed/crops \
  --out-dir runs/_cls_smoke \
  > queue_logs/_cls_smoke.train.log 2>&1
if [ $? -ne 0 ]; then
  echo "[$(date)] CLASSIFICATION SMOKE FAILED" >> queue_logs/classification_queue.log
  exit 1
fi
echo "[$(date)] CLASSIFICATION SMOKE PASSED" >> queue_logs/classification_queue.log

run_cls () {
  NAME=$1
  CFG=$2
  echo "[$(date)] START $NAME" >> queue_logs/classification_queue.log
  python -m src.autofish_vfm.train_classifier \
    --config configs/$CFG \
    --index data/processed/index.csv \
    --crops-dir data/processed/crops \
    --out-dir runs/$NAME \
    > queue_logs/$NAME.train.log 2>&1
  if [ $? -ne 0 ]; then
    echo "[$(date)] TRAIN FAILED $NAME" >> queue_logs/classification_queue.log
    exit 1
  fi
  python -m src.autofish_vfm.evaluate_classifier \
    --checkpoint runs/$NAME/best.pt \
    --config configs/$CFG \
    --index data/processed/index.csv \
    --crops-dir data/processed/crops \
    --out runs/$NAME/test_metrics.json \
    > queue_logs/$NAME.eval.log 2>&1
  if [ $? -ne 0 ]; then
    echo "[$(date)] EVAL FAILED $NAME" >> queue_logs/classification_queue.log
    exit 1
  fi
  echo "[$(date)] DONE $NAME" >> queue_logs/classification_queue.log
}

run_cls cls_mobilenet_v2 cls_mobilenet_v2.json
run_cls cls_convnext_tiny cls_convnext_tiny.json
run_cls cls_clip_frozen cls_clip_frozen.json
run_cls cls_dinov2_frozen cls_dinov2_frozen.json

echo "[$(date)] CLASSIFICATION QUEUE COMPLETE" >> queue_logs/classification_queue.log
