#!/bin/bash
# Validation-split inference for ensemble model selection (inference only, no training).
cd /home/sb2597/autofish_baseline_repro
source .venv/bin/activate

evalval () {
  NAME=$1; CFG=$2
  python -m src.autofish_vfm.evaluate \
    --checkpoint runs/$NAME/best.pt --config configs/$CFG \
    --index data/processed/index.csv --crops-dir data/processed/crops \
    --split val --out runs/$NAME/val_metrics.json > queue_logs/$NAME.valeval.log 2>&1
  echo "done $NAME"
}

evalval baseline_official                    baseline_official.json
evalval convnext_tiny_official               convnext_tiny_official.json
evalval clip_vitb32_lastblock_from_frozen    clip_vitb32_lastblock_from_frozen.json
evalval clip_vitb32_frozen                   clip_vitb32_frozen.json
evalval dinov2_vits14_patchtokens_frozen     dinov2_vits14_patchtokens_frozen.json
echo "VAL EVAL COMPLETE"
