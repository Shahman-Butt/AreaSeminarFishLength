#!/bin/bash
# NEW experiments only (no repeats of completed seed-42 runs).
# Controlled, multi-seed patch-vs-CLS comparison + patch-token last-block improvement attempt.
# Launch inside tmux:  tmux new -d -s newexp 'bash run_new_experiments.sh'
cd /home/sb2597/autofish_baseline_repro
source .venv/bin/activate
mkdir -p queue_logs
LOG=queue_logs/new_experiments.log

run_reg () {
  NAME=$1; CFG=$2
  if [ -f "runs/$NAME/test_metrics.json" ]; then
    echo "[$(date)] SKIP $NAME (results exist)" >> $LOG; return
  fi
  echo "[$(date)] START $NAME" >> $LOG
  /usr/bin/time -v python -m src.autofish_vfm.train_baseline \
    --config configs/$CFG --index data/processed/index.csv \
    --crops-dir data/processed/crops --out-dir runs/$NAME \
    > queue_logs/$NAME.train.log 2>&1
  if [ $? -ne 0 ]; then echo "[$(date)] TRAIN FAILED $NAME" >> $LOG; return; fi
  python -m src.autofish_vfm.evaluate \
    --checkpoint runs/$NAME/best.pt --config configs/$CFG \
    --index data/processed/index.csv --crops-dir data/processed/crops \
    --out runs/$NAME/test_metrics.json > queue_logs/$NAME.eval.log 2>&1
  echo "[$(date)] DONE $NAME" >> $LOG
}

# --- Controlled CLS-vs-patch comparison (identical HPs, only token type differs) ---
run_reg dinov2_vits14_clstoken_ctrl_s42   dinov2_vits14_clstoken_ctrl_s42.json
run_reg dinov2_vits14_clstoken_ctrl_s1    dinov2_vits14_clstoken_ctrl_s1.json
run_reg dinov2_vits14_clstoken_ctrl_s2    dinov2_vits14_clstoken_ctrl_s2.json
run_reg dinov2_vits14_patchtokens_frozen_s1  dinov2_vits14_patchtokens_frozen_s1.json
run_reg dinov2_vits14_patchtokens_frozen_s2  dinov2_vits14_patchtokens_frozen_s2.json

# --- Improvement attempt: patch tokens + last-block fine-tune (warm-start) ---
run_reg dinov2_vits14_patchtokens_lastblock  dinov2_vits14_patchtokens_lastblock.json

echo "[$(date)] NEW EXPERIMENTS COMPLETE" >> $LOG
