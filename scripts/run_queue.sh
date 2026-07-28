#!/bin/bash
# Persistent experiment queue. Runs train + evaluate for each config, logs everything.
# Launch inside tmux so it survives disconnects:  tmux new -d -s fish 'bash run_queue.sh'
cd /home/sb2597/autofish_baseline_repro
source .venv/bin/activate
mkdir -p queue_logs

run_reg () {
  NAME=$1; CFG=$2
  echo "[$(date)] START $NAME" >> queue_logs/queue.log
  python -m src.autofish_vfm.train_baseline \
    --config configs/$CFG --index data/processed/index.csv \
    --crops-dir data/processed/crops --out-dir runs/$NAME \
    > queue_logs/$NAME.train.log 2>&1
  python -m src.autofish_vfm.evaluate \
    --checkpoint runs/$NAME/best.pt --config configs/$CFG \
    --index data/processed/index.csv --crops-dir data/processed/crops \
    --out runs/$NAME/test_metrics.json \
    > queue_logs/$NAME.eval.log 2>&1
  echo "[$(date)] DONE $NAME" >> queue_logs/queue.log
}

# --- experiment queue (add lines here to extend) ---
run_reg dinov2_vits14_patchtokens_frozen dinov2_vits14_patchtokens_frozen.json

echo "[$(date)] QUEUE COMPLETE" >> queue_logs/queue.log
