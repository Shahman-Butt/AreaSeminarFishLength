# ODE Report — Automated Fish Length Estimation with Vision Foundation Models

**ODE = Overview · Data · Execution** — a structured, reproducible record of the machine-learning
workflow, following the seminar's Scientific-Writing protocol.

Authors: Abu Bakar, Laksh Jiwani, Shahman Butt · Supervisor: Bohan Zhuang, M.Sc. · Professor:
Stefan Oehmcke · University of Rostock (VACOT).
Repository: https://github.com/Shahman-Butt/AreaSeminarFishLength

---

## O — OVERVIEW

**Research question.** Can a modern image encoder or vision foundation model beat the reproduced
AutoFish MobileNetV2 baseline at estimating fish length from a photo, under identical data, split,
and metrics?

**Target.** Reproduce the paper's baseline (0.62 cm non-occluded MAE), then run a controlled
encoder-swap comparison. Secondary tasks: species classification; a reliable improvement inside a
foundation model.

**Scope & assumptions.**
- Task is **regression** (predict a continuous length in cm), not detection/classification/segmentation.
- Single fish per crop; bounding box provided by the dataset (we do not train a detector).
- Only the **encoder** changes between length experiments; everything else is held fixed.

**Why this setup.** Reproducing the baseline first calibrates the whole pipeline, so any later
difference is attributable to the encoder rather than to implementation error. A group-level split
prevents the same fish leaking across train/test.

**Contributions (current state).**
1. Baseline reproduced to within 0.013 cm (non-occluded).
2. Controlled single-model comparison of 5 encoders (9 length runs).
3. A reliable, multi-seed improvement *within* DINOv2: patch-token pooling beats the CLS token.
4. Species classification across 4 encoders.
5. **EfficientNet-B0 reaches 0.781 cm — within 0.010 cm of the baseline** at a basic recipe; a
   validation-based recipe search is under way to try to cross it with a single model.

---

## D — DATA

**Dataset.** AutoFish (Bengtson et al., WACV Workshops 2025), Hugging Face `vapaau/autofish`:
1,500 top-view RGB images, 454 unique fish, 25 groups, **18,157** instance annotations. Each
annotation has a species label, a `fish_id`, a hand-measured `length` (cm), a bounding box, and a
polygon segmentation.

**Labels.** Length in cm (regression target); species (7 classes: cod, haddock, hake,
horse_mackerel, other, saithe, whiting) for the classification task.

**Preprocessing** (`scripts/`):
1. `build_autofish_index.py` → single `index.csv` (18,157 rows) joining annotations, images, species.
2. `make_crops.py` → per-annotation **masked, square** crops resized to 224×224 (background outside
   the fish polygon is blacked out; square crop preserves aspect ratio). 0 missing crops.
3. ImageNet normalization; ColorJitter augmentation on the training split only.

**Splits (official, group-level).** Train 15 groups / Validation 5 / Test 5.
Test = **3,759** annotations (1,879 non-occluded [Set1+Set2] + 1,880 occluded [All]).

**Leakage control.** An audit found one fish (`fish_id 113`) crossing splits via a singleton
duplicate annotation (id 3759, in a train group while its 40 other images are in a test group). It
was removed (`exclusions.json`); the audit then confirms **zero** fish cross any split. This is why
the non-occluded test count is 1,879, not 1,880.

---

## E — EXECUTION

**Model.** Encoder → feature vector, concatenated with 4 normalized bbox values → MLP regression
head (Linear+BatchNorm+ReLU blocks → 1 output). Only the encoder differs between length experiments.

**Models & configurations tried** (full-test MAE, cm):

| Encoder | Type | Adaptation | Recipe (Adam) | Full-test MAE |
|---|---|---|---|---:|
| MobileNetV2 (baseline) | CNN | full FT | 1e-3, 200 ep, batch 32 | **0.771** |
| EfficientNet-B0 | CNN | full FT | 1e-4, 100 ep, batch 16 | **0.781** |
| ConvNeXt-Tiny | CNN | full FT | 1e-4, 100 ep, batch 16 | 0.914 |
| CLIP ViT-B/32 | Transformer | last-block FT | 1e-4, 100 ep | 0.958 |
| CLIP ViT-B/32 | Transformer | frozen | 1e-4, 100 ep | 1.002 |
| DINOv2 ViT-S/14 (patch) | Transformer | frozen, patch pooling | 1e-4, 100 ep | 1.261 ± 0.054 (3 seeds) |
| DINOv2 ViT-S/14 (CLS) | Transformer | last-block FT | 1e-4, 100 ep | 1.439 |
| DINOv2 ViT-S/14 (CLS) | Transformer | frozen | 1e-3, 100 ep | 1.738 |
| DINOv2 ViT-S/14 (CLS) | Transformer | full FT | enc 1e-5 / 1e-6 | 1.778 / 2.132 |

**Training protocol.** L1 loss, Adam; best checkpoint selected on **validation** MAE; test set used
once. Fixed seed (42) per run; one JSON config per experiment; hardware NVIDIA RTX 5000 Ada (32 GB),
Python 3.11. New trainer options added this round: **cosine LR schedule + weight decay**.

**Metrics.** MAE (primary, cm), plus RMSE, MAPE, bias, R²; reported on full test, non-occluded, and
occluded subsets. Baseline full-test: RMSE 1.268, MAPE 2.41%, bias +0.035, R² 0.947.

**Key results.**
- **Baseline reproduction:** 0.633 cm non-occluded vs paper 0.62 (Δ 0.013) → pipeline validated.
- **Single-model ranking:** MobileNetV2 best; **EfficientNet-B0 within 0.010 cm**; supervised CNNs
  top the ranking; foundation models trail.
- **Reliable DINOv2 improvement (3 seeds, identical settings):** patch-token pooling 1.261 ± 0.054
  vs CLS token 1.843 ± 0.023 — a 0.58 cm gain with non-overlapping ranges. Last-block fine-tuning did
  not help (1.345). Best DINOv2 configuration = frozen patch pooling.
- **Species classification (accuracy):** ConvNeXt 99.6%, MobileNetV2 99.1%, DINOv2 98.2%, CLIP 95.1%
  — all strong; DINOv2 rises from last (length) to near-top (species), evidence that foundation
  features suit semantics over precise geometry (hypothesis).

**Validation-based recipe search (in progress).** Because EfficientNet-B0 used a weaker recipe than
the baseline, we run stronger recipes (cosine LR + weight decay + tuned LR + 200 epochs) for
EfficientNet-B0 and ConvNeXt-Tiny, select the winner on **validation**, and report once on **test**.
Persistent execution (`scripts/run_beat_baseline_queue.sh`) on the GPU server; results committed as
they land.

**Findings & interpretation.**
- Supervised CNNs (MobileNetV2, EfficientNet-B0, ConvNeXt) fill the top of the single-model ranking;
  general foundation features transfer only partially (CLIP) or poorly (DINOv2) for precise metric
  regression. Stated as an observation; mechanistic reasons are hypotheses.
- EfficientNet-B0's near-baseline result at a basic recipe indicates the baseline is likely beatable
  by a single model with proper tuning.

**Limitations.**
- Single training run for most single models (multi-seed done for the DINOv2 patch-vs-CLS study).
- Foundation models tested at small scale (ViT-S/14, ViT-B/32) with limited fine-tuning budgets.
- No single model beats the baseline yet; mask segmentation not yet evaluated.
- The occluded-set reproduction difference vs the paper (0.909 vs 1.38) is unexplained.

**Future work.** Finish the recipe search (likely to cross the baseline); EfficientNet-B2; patch
pooling for CLIP; larger ViT variants; multi-seed the top single models; mask segmentation as a
third task; DINOv3 (pending weight access).

**Reproducibility.** Fixed seeds; official group split hard-coded; versioned configs; requirements.txt;
per-run config/history/metrics/predictions; checkpoints on server; persistent queue scripts. Generative
AI assisted with code scaffolding and drafting; all experiments and conclusions were verified by the
authors.
