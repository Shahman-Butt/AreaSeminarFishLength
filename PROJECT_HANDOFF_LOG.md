# Project Handoff Log: AutoFish Fish Length Estimation

## Current Completion Status

The project is **partly complete, not fully finished**.

Completed:

- Q1 baseline reproduction is complete.
- The reproduced AutoFish REG MobileNetV2 baseline closely matches the paper.
- Several Q2 encoder/foundation-model experiments are complete.
- Results have been copied from the server to the local project.
- Main documentation files have been updated.
- A full report has been created.

Not fully complete:

- No multi-seed repeat experiments have been done yet.
- EfficientNet has not yet been run.
- No final seminar paper/slides have been produced.
- No formal error analysis by species, size range, or occlusion difficulty has been done.
- Results are mostly single-run comparisons, so they should be presented as first experimental findings, not final statistically validated claims.

Recommended next step:

> Run EfficientNet-B0 as the next supervised ImageNet encoder experiment, because MobileNetV2 and ConvNeXt suggest supervised ImageNet encoders are more promising than frozen/general foundation models for this precise regression task.

## User and Project Context

The user is working on an Area Seminar project about automated fish length estimation. The project is local in VS Code, but training runs are executed on a university GPU server through SSH.

Local Windows project:

```text
C:\Users\Shahman\Desktop\SEM3\AreaSeminar
```

Server:

```text
sb2597@baltic.informatik.uni-rostock.de
```

Server project folder:

```text
/home/sb2597/autofish_baseline_repro
```

SSH key used by Codex:

```text
C:\Users\Shahman\.ssh\baltic_codex_ed25519
```

Important note:

> The current VS Code/Codex session is local, not attached to the server filesystem through VS Code Remote SSH. Server actions are done through SSH/SCP commands.

## Original Task

The professor's main task was:

1. Reproduce the AutoFish paper baseline first.
2. Then replace/test different image encoders and vision foundation models.
3. Compare whether newer encoders can improve fish length estimation.
4. Document everything clearly for seminar/report/professor discussion.

## Target Paper

Paper:

```text
Bengtson et al.,
"AutoFish: Dataset and Benchmark for Fine-grained Analysis of Fish",
WACVW 2025 / arXiv:2501.03767
```

Reproduced method:

```text
REG MobileNetV2 baseline
```

Paper baseline numbers:

| Model | Paper non-occluded MAE | Paper occluded MAE |
|---|---:|---:|
| REG MobileNetV2 | 0.62 cm | 1.38 cm |

## Dataset and Preprocessing

Dataset:

```text
Hugging Face: vapaau/autofish
```

Server raw data:

```text
/home/sb2597/autofish_baseline_repro/data/raw/autofish
```

Processed data:

```text
/home/sb2597/autofish_baseline_repro/data/processed/index.csv
/home/sb2597/autofish_baseline_repro/data/processed/crops
```

Processed dataset summary:

| Item | Count |
|---|---:|
| Rows / annotations | 18,157 |
| Images | 1,500 |
| Unique fish | 454 |
| Groups | 25 |
| Missing crops | 0 |
| Fish leakage after cleanup | 0 |

Official split:

| Split | Groups |
|---|---|
| Train | 2, 3, 4, 5, 7, 8, 9, 12, 13, 15, 16, 18, 19, 23, 24 |
| Validation | 1, 6, 11, 17, 25 |
| Test | 10, 14, 20, 21, 22 |

Set mapping:

| Image number range | Set name | Meaning |
|---|---|---|
| `00001-00020` | Set1 | non-occluded |
| `00021-00040` | Set2 | non-occluded |
| `00041-00060` | All | occluded/crowded |

Leakage cleanup:

- Found one cross-split duplicate fish ID: `fish_id=113`.
- Removed a singleton duplicate annotation to prevent leakage.
- This is why non-occluded test count is `1,879` instead of `1,880`.

## Server Setup Summary

Server GPU:

```text
NVIDIA RTX 5000 Ada Generation, about 32 GB VRAM
```

Server environment:

```text
/home/sb2597/autofish_baseline_repro/.venv
```

Important packages:

- PyTorch
- Torchvision
- OpenCLIP
- NumPy
- Pandas
- Pillow
- scikit-learn
- tqdm
- Hugging Face tools
- pycocotools

The server default shell is `tcsh`, so use `bash -lc` or pipe a bash script into SSH for complex commands.

## Main Code Files

| File | Purpose |
|---|---|
| `src/autofish_vfm/data.py` | Dataset loading, crops, transforms, normalization |
| `src/autofish_vfm/models.py` | MobileNetV2, DINOv2, CLIP, ConvNeXt model definitions |
| `src/autofish_vfm/train_baseline.py` | Training loop, checkpoint saving |
| `src/autofish_vfm/evaluate.py` | Final test evaluation and prediction CSV export |
| `src/autofish_vfm/metrics.py` | MAE, RMSE, MAPE, bias, R2 |
| `configs/` | Experiment configs |
| `runs/` | Local copied results |

## Important Documentation Files

| File | Purpose |
|---|---|
| `README.md` | Short project guide and main results table |
| `PROJECT_SUMMARY_GUIDE.md` | Layman + technical explanations and professor Q&A |
| `AREA_SEMINAR_FULL_REPORT.md` | Full report-style document |
| `PROJECT_HANDOFF_LOG.md` | This handoff log |

## Completed Results

### 1. REG MobileNetV2 Baseline

Local folder:

```text
runs/baseline_official
```

Result:

| Subset | MAE cm | RMSE cm | MAPE % | Bias cm | R2 | n |
|---|---:|---:|---:|---:|---:|---:|
| Full test | 0.771 | 1.268 | 2.411 | 0.035 | 0.947 | 3,759 |
| Non-occluded Set1+Set2 | 0.633 | 1.027 | 1.960 | 0.080 | 0.965 | 1,879 |
| Occluded All | 0.909 | 1.470 | 2.862 | -0.009 | 0.929 | 1,880 |

Interpretation:

> Baseline reproduction is successful. Paper non-occluded result is `0.62 cm`; ours is `0.633 cm`.

### 2. DINOv2 ViT-S/14 Frozen

Local folder:

```text
runs/dinov2_vits14_frozen
```

Result:

| Subset | MAE cm |
|---|---:|
| Non-occluded | 1.690 |
| Occluded | 1.786 |
| Full test | 1.738 |

Interpretation:

> Frozen DINOv2 did not transfer well for precise fish length regression.

### 3. DINOv2 Full Fine-Tune, Encoder LR 1e-5

Local folder:

```text
runs/dinov2_vits14_finetune_lr1e5
```

Result:

| Subset | MAE cm |
|---|---:|
| Non-occluded | 1.636 |
| Occluded | 1.919 |
| Full test | 1.778 |

Interpretation:

> Slight non-occluded improvement over frozen DINOv2, but occluded became worse.

### 4. DINOv2 Full Fine-Tune, Encoder LR 1e-6

Local folder:

```text
runs/dinov2_vits14_finetune_lr1e6
```

Result:

| Subset | MAE cm |
|---|---:|
| Non-occluded | 2.075 |
| Occluded | 2.189 |
| Full test | 2.132 |

Interpretation:

> Too conservative or poorly adapted; worse than frozen DINOv2.

### 5. DINOv2 Last Block Fine-Tuned

Local folder:

```text
runs/dinov2_vits14_lastblock_from_frozen
```

Result:

| Subset | MAE cm |
|---|---:|
| Non-occluded | 1.340 |
| Occluded | 1.537 |
| Full test | 1.439 |

Interpretation:

> Best DINOv2 variant, but still far behind MobileNetV2 and later CLIP/ConvNeXt.

### 6. CLIP ViT-B/32 Frozen

Local folder:

```text
runs/clip_vitb32_frozen
```

Result:

| Subset | MAE cm |
|---|---:|
| Non-occluded | 0.898 |
| Occluded | 1.106 |
| Full test | 1.002 |

Interpretation:

> CLIP transferred much better than DINOv2 and became the first strong non-baseline result.

### 7. CLIP ViT-B/32 Last Visual Block Fine-Tuned

Local folder:

```text
runs/clip_vitb32_lastblock_from_frozen
```

Result:

| Subset | MAE cm |
|---|---:|
| Non-occluded | 0.842 |
| Occluded | 1.074 |
| Full test | 0.958 |

Best validation:

```text
epoch 11, validation MAE 1.003 cm
```

Interpretation:

> Partial CLIP fine-tuning improved frozen CLIP, but still did not beat MobileNetV2.

### 8. ConvNeXt-Tiny ImageNet Encoder

Local folder:

```text
runs/convnext_tiny_official
```

Result:

| Subset | MAE cm | RMSE cm | MAPE % | Bias cm | R2 | n |
|---|---:|---:|---:|---:|---:|---:|
| Full test | 0.914 | 1.338 | 2.822 | 0.131 | 0.941 | 3,759 |
| Non-occluded Set1+Set2 | 0.814 | 1.181 | 2.496 | 0.136 | 0.954 | 1,879 |
| Occluded All | 1.014 | 1.479 | 3.148 | 0.127 | 0.928 | 1,880 |

Best validation:

```text
epoch 83, validation MAE 0.922 cm
```

Interpretation:

> ConvNeXt-Tiny is currently the best non-baseline encoder. It beats CLIP last-block but still does not beat MobileNetV2.

## Current Ranking by Full Test MAE

| Rank | Model | Full test MAE |
|---:|---|---:|
| 1 | REG MobileNetV2 baseline | 0.771 cm |
| 2 | ConvNeXt-Tiny | 0.914 cm |
| 3 | CLIP ViT-B/32 last visual block fine-tuned | 0.958 cm |
| 4 | CLIP ViT-B/32 frozen | 1.002 cm |
| 5 | DINOv2 last block fine-tuned | 1.439 cm |
| 6 | DINOv2 frozen | 1.738 cm |
| 7 | DINOv2 full fine-tune LR 1e-5 | 1.778 cm |
| 8 | DINOv2 full fine-tune LR 1e-6 | 2.132 cm |

## Scientific Interpretation So Far

Main conclusion:

> The AutoFish REG MobileNetV2 baseline was successfully reproduced and remains the best model so far.

Foundation model conclusion:

> Foundation models are not automatically better for this precise regression task. CLIP transfers better than DINOv2, but still does not beat MobileNetV2.

Supervised encoder conclusion:

> ConvNeXt-Tiny is the closest challenger and suggests that supervised ImageNet encoders may be more promising than general foundation features for fish length estimation.

## Recommended Next Actions

### Immediate Next Experiment

Run EfficientNet-B0:

```text
EfficientNet-B0 ImageNet encoder + bbox input + regression head
```

Why:

- MobileNetV2 is still best.
- ConvNeXt-Tiny is now second best.
- Both are supervised ImageNet-style encoders.
- EfficientNet is the next natural supervised encoder family.

Suggested first config:

- model: `efficientnet_b0`
- pretrained: ImageNet
- image size: 224
- bbox input: true
- head: `[512, 128, 1]`
- learning rate: start around `1e-4`
- batch size: 16 or 32 depending GPU memory
- epochs: 100
- same split and metrics

### After EfficientNet

1. If EfficientNet-B0 is strong, test EfficientNet-B2.
2. Repeat MobileNetV2, ConvNeXt-Tiny, and best EfficientNet with 3 seeds.
3. Add error analysis:
   - species-level error,
   - length-range error,
   - occluded vs non-occluded error,
   - worst prediction examples.
4. Prepare seminar slides/report.

## Useful Server Commands

Check server project:

```powershell
ssh -i C:\Users\Shahman\.ssh\baltic_codex_ed25519 sb2597@baltic.informatik.uni-rostock.de "bash -lc 'cd /home/sb2597/autofish_baseline_repro && ls'"
```

Run training on server:

```bash
cd /home/sb2597/autofish_baseline_repro
. .venv/bin/activate
python -m src.autofish_vfm.train_baseline --config configs/<CONFIG>.json --index data/processed/index.csv --crops-dir data/processed/crops --out-dir runs/<RUN_NAME>
```

Run evaluation:

```bash
python -m src.autofish_vfm.evaluate --checkpoint runs/<RUN_NAME>/best.pt --config configs/<CONFIG>.json --index data/processed/index.csv --crops-dir data/processed/crops --out runs/<RUN_NAME>/test_metrics.json
```

Copy result files back:

```powershell
scp -i C:\Users\Shahman\.ssh\baltic_codex_ed25519 sb2597@baltic.informatik.uni-rostock.de:/home/sb2597/autofish_baseline_repro/runs/<RUN_NAME>/test_metrics.json runs\<RUN_NAME>\
scp -i C:\Users\Shahman\.ssh\baltic_codex_ed25519 sb2597@baltic.informatik.uni-rostock.de:/home/sb2597/autofish_baseline_repro/runs/<RUN_NAME>/history.csv runs\<RUN_NAME>\
scp -i C:\Users\Shahman\.ssh\baltic_codex_ed25519 sb2597@baltic.informatik.uni-rostock.de:/home/sb2597/autofish_baseline_repro/runs/<RUN_NAME>/test_metrics.predictions.csv runs\<RUN_NAME>\
```

## How to Explain to Professor

Short version:

> We successfully reproduced the AutoFish REG MobileNetV2 baseline. The paper reports `0.62 cm` MAE on non-occluded fish, and our reproduction achieved `0.633 cm`, so the baseline is reliable. Then we tested DINOv2, CLIP, and ConvNeXt-Tiny using the same data split and metrics. DINOv2 did not perform well, CLIP transferred better, and ConvNeXt-Tiny became the closest challenger. However, MobileNetV2 still remains the best overall model with `0.771 cm` full-test MAE.

Technical version:

> The current evidence suggests that for this precise fish length regression task, task-adapted supervised ImageNet encoders are more effective than frozen or lightly fine-tuned general foundation features. ConvNeXt-Tiny improved over CLIP but did not surpass MobileNetV2, so the next fair direction is EfficientNet and multi-seed validation of the strongest models.

