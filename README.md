# Fish Length Estimation Baseline Reproduction

This repository is set up for the first research question in the seminar:
reproduce the AutoFish deep-learning length-estimation baseline.

For the complete professor-facing write-up of the baseline reproduction,
foundation-model experiments, results, interpretation, limitations, and next
steps, see:

```text
AREA_SEMINAR_FULL_REPORT.md
```

The paper is **Bengtson et al., "AutoFish: Dataset and Benchmark for
Fine-grained Analysis of Fish", WACVW 2025 / arXiv:2501.03767**. The baseline
we reproduce is the paper's **REG** method: masked fish crops, MobileNetV2
ImageNet encoder, bbox coordinates as extra input, and a regression head that
predicts fish length in centimeters. The paper reports **0.62 cm MAE** for
non-occluded images (Set1+Set2) and **1.38 cm MAE** for occluded images (All).

## What is already here

- Official AutoFish training release cloned to `external/autofish_training_release`.
- AutoFish dataset downloaded to `data/raw/autofish`.
- Reproduction split fixed to the authors' release split:
  - train: `2,3,4,5,7,8,9,12,13,15,16,18,19,23,24`
  - val: `1,6,11,17,25`
  - test: `10,14,20,21,22`

## Environment

Use Python 3.11 on the server/GPU machine.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For CUDA PyTorch, install the CUDA wheel first if needed:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Data checks and crop generation

```bash
python scripts/build_autofish_index.py --raw-dir data/raw/autofish --out data/processed/index.csv --splits-out data/processed/splits.json
python scripts/make_crops.py --raw-dir data/raw/autofish --index data/processed/index.csv --out-dir data/processed/crops
python scripts/check_processed.py --index data/processed/index.csv --crops-dir data/processed/crops
```

The Set1/Set2/All mapping is inferred from each group's image numbers:
`00001-00020 = Set1`, `00021-00040 = Set2`, and `00041-00060 = All`.

The downloaded annotations contain one cross-split duplicate fish ID:
`fish_id=113` appears once in group 5 and repeatedly in group 22. The indexer
drops the singleton annotation by default and writes the exact exclusion to
`data/processed/exclusions.json`; use `--keep-cross-split-duplicates` only when
you intentionally want byte-for-byte behavior closer to the authors' raw split.

## Train and evaluate the baseline

```bash
python -m src.autofish_vfm.train_baseline --config configs/baseline_official.json --index data/processed/index.csv --crops-dir data/processed/crops --out-dir results/baseline_mobilenetv2
python -m src.autofish_vfm.evaluate --checkpoint results/baseline_mobilenetv2/best.pt --config configs/baseline_official.json --index data/processed/index.csv --crops-dir data/processed/crops --out results/baseline_mobilenetv2/test_metrics.json
```

On Linux/server, the full Q1 baseline pipeline is:

```bash
bash scripts/run_baseline_server.sh
```

For a quick environment smoke test before the full 200-epoch run:

```bash
python -m src.autofish_vfm.train_baseline --config configs/baseline_smoke.json --index data/processed/index.csv --crops-dir data/processed/crops --out-dir results/smoke
```

The main table to compare against:

| Model | Paper non-occluded MAE | Paper occluded MAE | Our non-occluded MAE | Our occluded MAE | Our all-test MAE |
|---|---:|---:|---:|---:|---:|
| REG MobileNetV2 | 0.62 cm | 1.38 cm | 0.633 cm | 0.909 cm | 0.771 cm |

Project progress table:

| Stage | Model / experiment | Paper non-occluded MAE | Paper occluded MAE | Our non-occluded MAE | Our occluded MAE | Our all-test MAE | Status |
|---|---|---:|---:|---:|---:|---:|---|
| Q1 baseline | REG MobileNetV2 | 0.62 cm | 1.38 cm | 0.633 cm | 0.909 cm | 0.771 cm | Complete |
| Q2 VFM | DINOv2 ViT-S/14 frozen encoder + regression head | Not reported | Not reported | 1.690 cm | 1.786 cm | 1.738 cm | Complete |
| Q2 VFM | DINOv2 ViT-S/14 fine-tuned, encoder LR 1e-5 | Not reported | Not reported | 1.636 cm | 1.919 cm | 1.778 cm | Complete |
| Q2 VFM | DINOv2 ViT-S/14 fine-tuned, encoder LR 1e-6 | Not reported | Not reported | 2.075 cm | 2.189 cm | 2.132 cm | Complete |
| Q2 VFM | DINOv2 ViT-S/14 frozen head then last block | Not reported | Not reported | 1.340 cm | 1.537 cm | 1.439 cm | Complete |
| Q2 VFM | CLIP ViT-B/32 frozen encoder + regression head | Not reported | Not reported | 0.898 cm | 1.106 cm | 1.002 cm | Complete |
| Q2 VFM | CLIP ViT-B/32 frozen head then last visual block | Not reported | Not reported | 0.842 cm | 1.074 cm | 0.958 cm | Complete |
| Q2 VFM | ConvNeXt-Tiny ImageNet encoder | Not reported | Not reported | 0.814 cm | 1.014 cm | 0.914 cm | Complete |

Completed official server run:

- Server folder: `/home/sb2597/autofish_baseline_repro`
- Run folder: `/home/sb2597/autofish_baseline_repro/runs/baseline_official`
- Local copied metrics: `runs/baseline_official/test_metrics.json`
- Local copied predictions: `runs/baseline_official/test_metrics.predictions.csv`
- Training: 200 epochs, best validation checkpoint selected from epoch 153.
- Best validation MAE: 0.805 cm.

The non-occluded test count is 1,879 instead of 1,880 because the preprocessing
removes one singleton cross-split duplicate annotation (`fish_id=113`) to avoid
train/test fish leakage.

## Current VFM experiment result

The first VFM experiment kept the same data, crops, splits, metrics, and regression
head idea, but replaces the MobileNetV2 image encoder with a frozen DINOv2
ViT-S/14 encoder:

```bash
python -m src.autofish_vfm.train_baseline --config configs/dinov2_vits14_frozen.json --index data/processed/index.csv --crops-dir data/processed/crops --out-dir runs/dinov2_vits14_frozen
python -m src.autofish_vfm.evaluate --checkpoint runs/dinov2_vits14_frozen/best.pt --config configs/dinov2_vits14_frozen.json --index data/processed/index.csv --crops-dir data/processed/crops --out runs/dinov2_vits14_frozen/test_metrics.json
```

Server run folder:

```text
/home/sb2597/autofish_baseline_repro/runs/dinov2_vits14_frozen
```

Local copied metrics:

```text
runs/dinov2_vits14_frozen/test_metrics.json
```

Result: the frozen DINOv2 encoder is weaker than the fully trained MobileNetV2
baseline on this length-regression task. This suggests that simply freezing a
general vision foundation model is not enough for precise fish-length estimation;
the next fair experiment should fine-tune at least part of the encoder or cache
DINOv2 features and test a stronger regression head.

Fine-tuned DINOv2 result:

```text
runs/dinov2_vits14_finetune_lr1e5/test_metrics.json
```

The small-learning-rate fine-tuning run improved the non-occluded DINOv2 score
slightly (`1.690 cm -> 1.636 cm`) but worsened the occluded score
(`1.786 cm -> 1.919 cm`). It remained behind the MobileNetV2 baseline, and the
validation curve was unstable, so the next attempt uses a lower encoder learning
rate (`1e-6`) while keeping the head learning rate at `1e-4`.

The `1e-6` encoder-learning-rate run was more conservative but performed worse
than the frozen and `1e-5` DINOv2 runs:

```text
runs/dinov2_vits14_finetune_lr1e6/test_metrics.json
```

Result: non-occluded MAE `2.075 cm`, occluded MAE `2.189 cm`, all-test MAE
`2.132 cm`. This suggests that lowering the whole-encoder learning rate alone
does not solve the problem; the better next direction is controlled partial
fine-tuning, for example training the head first and then unfreezing only the
last DINOv2 block.

Last-block DINOv2 result:

```text
runs/dinov2_vits14_lastblock_from_frozen/test_metrics.json
```

This was the best DINOv2 variant so far: non-occluded MAE `1.340 cm`, occluded
MAE `1.537 cm`, all-test MAE `1.439 cm`. It improved clearly over frozen
DINOv2 and full-encoder fine-tuning, but it still did not beat the reproduced
MobileNetV2 baseline.

CLIP frozen encoder result:

```text
runs/clip_vitb32_frozen/test_metrics.json
```

The frozen CLIP ViT-B/32 encoder is the best vision-foundation-model result so
far: non-occluded MAE `0.898 cm`, occluded MAE `1.106 cm`, all-test MAE
`1.002 cm`. It is still behind our reproduced MobileNetV2 baseline
(`0.633 cm` non-occluded, `0.771 cm` all-test), but it is much stronger than the
DINOv2 variants. The next sensible experiment is controlled CLIP partial
fine-tuning, for example training only the regression head plus the last visual
transformer block.

CLIP last-block fine-tuning result:

```text
runs/clip_vitb32_lastblock_from_frozen
```

This run starts from the frozen CLIP checkpoint and unfreezes only the final
visual transformer block plus the regression head. It improved frozen CLIP:
non-occluded MAE `0.898 cm -> 0.842 cm`, occluded MAE `1.106 cm -> 1.074 cm`,
and all-test MAE `1.002 cm -> 0.958 cm`. It is now the best foundation-model
result in this project so far, but it still remains behind the reproduced
MobileNetV2 baseline.

ConvNeXt-Tiny result:

```text
runs/convnext_tiny_official/test_metrics.json
```

ConvNeXt-Tiny is now the strongest non-baseline encoder result so far:
non-occluded MAE `0.814 cm`, occluded MAE `1.014 cm`, all-test MAE `0.914 cm`.
It improves over CLIP last-block fine-tuning (`0.958 cm` all-test MAE), but it
still does not beat the reproduced MobileNetV2 baseline (`0.771 cm` all-test
MAE).
