# Experiment Audit — AutoFish Area Seminar

**Purpose.** A complete inventory of what is already computed (must not be repeated), what is
reusable, what is genuinely new, and the computational priority of remaining work. Prepared under a
strict **no-repeat rule**: no completed training/evaluation with usable saved results is rerun.

Generated: 2026-07-28. Repo: https://github.com/Shahman-Butt/AreaSeminarFishLength

---

## 1. COMPLETED work — DO NOT REPEAT (usable saved results exist)

All of the following have saved `config.json`, `history.csv`, `test_metrics.json`, per-fish
`test_metrics.predictions.csv`, and (on the server) `best.pt` checkpoints.

### Length regression (task = predict length in cm), full-test MAE

| Run | Encoder | Adaptation | Full-test MAE (cm) | Status |
|---|---|---|---:|---|
| baseline_official | MobileNetV2 | full FT (paper recipe) | **0.771** | ✅ done |
| convnext_tiny_official | ConvNeXt-Tiny | full FT | 0.914 | ✅ done |
| clip_vitb32_lastblock_from_frozen | CLIP ViT-B/32 | last block | 0.958 | ✅ done |
| clip_vitb32_frozen | CLIP ViT-B/32 | frozen | 1.002 | ✅ done |
| **dinov2_vits14_patchtokens_frozen** | DINOv2 ViT-S/14 | frozen, **patch tokens** | **1.199** | ✅ done (NEW this round) |
| dinov2_vits14_lastblock_from_frozen | DINOv2 ViT-S/14 | last block, CLS | 1.439 | ✅ done |
| dinov2_vits14_frozen | DINOv2 ViT-S/14 | frozen, CLS (LR1e-3/b32) | 1.738 | ✅ done |
| dinov2_vits14_finetune_lr1e5 | DINOv2 ViT-S/14 | full FT, enc 1e-5 | 1.778 | ✅ done |
| dinov2_vits14_finetune_lr1e6 | DINOv2 ViT-S/14 | full FT, enc 1e-6 | 2.132 | ✅ done |

### Species classification (task = identify species), full-test accuracy / macro-F1

| Run | Encoder | Accuracy | Macro-F1 | Status |
|---|---|---:|---:|---|
| cls_convnext_tiny | ConvNeXt-Tiny | 0.9957 | 0.9921 | ✅ done |
| cls_mobilenet_v2 | MobileNetV2 | 0.9912 | 0.9895 | ✅ done |
| cls_dinov2_frozen | DINOv2 ViT-S/14 frozen | 0.9819 | 0.9775 | ✅ done |
| cls_clip_frozen | CLIP ViT-B/32 frozen | 0.9513 | 0.9545 | ✅ done |

### Non-training artifacts (reusable, do not regenerate)
- Preprocessed `data/processed/index.csv` (18,157 rows), `crops/` (18,157 PNGs), `splits.json`,
  `exclusions.json` (fish-113 leak fix). ✅
- Error analysis tables + per-fish head-to-head (`results/error_analysis/`). ✅
- Qualitative figures (`results/qualitative/`). ✅
- Documentation set (`docs/`, `poster/`). ✅

---

## 2. REUSABLE artifacts (inputs for new work, not recomputed)

| Artifact | Location | Reused for |
|---|---|---|
| `dinov2_vits14_patchtokens_frozen/best.pt` | server | warm-start of patch-token last-block run |
| `dinov2_vits14_frozen/best.pt` | server | already used to warm-start CLS last-block |
| All prediction CSVs | `runs/*/test_metrics.predictions.csv` | error analysis, figures (no retrain) |
| index.csv / crops / splits | `data/processed/` | every experiment (fixed protocol) |
| Trained classifiers `best.pt` | server `runs/cls_*` | any further classification analysis |

---

## 3. NEW experiments — necessary and NOT previously done

### 3a. Controlled patch-vs-CLS comparison (scientific necessity) — RUNNING
**Reason it is new/necessary:** the existing CLS-frozen result (1.738) used LR 1e-3 / batch 32, but
the patch-token result (1.199) used LR 1e-4 / batch 16. The 0.54 cm gap is therefore **confounded**
by hyperparameters. A hyperparameter-matched CLS baseline was never run, so the improvement is not
yet cleanly attributable to patch tokens.

Launched (persistent tmux `newexp`), identical HPs (LR 1e-4, batch 16, 100 ep), only token differs:
- `dinov2_vits14_clstoken_ctrl_s42/s1/s2` — HP-matched CLS baseline, 3 seeds (NEW).
- `dinov2_vits14_patchtokens_frozen_s1/s2` — patch tokens, seeds 1,2 (seed 42 REUSED).
→ yields **mean ± std over 3 seeds** for both, a reliable controlled comparison.

### 3b. Improvement attempt: patch tokens + last-block fine-tune — RUNNING
`dinov2_vits14_patchtokens_lastblock` — warm-started from the existing patch-token frozen
checkpoint; tests whether partial fine-tuning pushes patch-token DINOv2 further. (NEW.)

### 3c. Not yet done / attempt pending
- **DINOv3 backbone** — hub repo present on server but import needs `torchmetrics`; weights may be
  license-gated. Priority: medium; attempt then document outcome honestly.
- **Mask segmentation** — the paper's third task uses Mask2Former (heavy detection/segmentation
  stack). Priority: GPU-intensive, multi-day; scoped as future work, not an overnight run.

---

## 4. Computational priority of remaining work

| Priority | Item | Rough cost | Rerun? |
|---|---|---|---|
| **Low** | Error analysis / figures / doc updates from saved predictions | minutes, CPU | reuse only |
| **Low** | Multi-seed patch + CLS frozen (per run ~40 min) | ~3.5 h total, running | NEW |
| **Medium** | Patch-token last-block improvement attempt | ~30 min, running | NEW |
| **Medium** | DINOv3 frozen + patch tokens (if weights load) | ~1 h + setup | NEW, uncertain |
| **GPU-intensive** | Mask segmentation (Mask2Former-style) | multi-day | future work |
| **GPU-intensive** | Full multi-seed of MobileNetV2 baseline (200 ep ×3) | several h | optional rigor |

---

## 5. What "improvement over baseline" means here (honest framing)

The MobileNetV2 length baseline (0.771 cm) is strong and hard to beat with small foundation-model
variants. The **credible, reliable improvement** we are establishing is **within the DINOv2 family**:
patch-token pooling over the CLS token. The multi-seed controlled runs (3a) will state whether that
improvement is real (mean ± std, non-overlapping) rather than a single lucky result. Beating the
MobileNetV2 baseline outright is **not yet demonstrated** and is reported as such.

---

*Live status of the new runs is in the server file `queue_logs/new_experiments.log`. Results are
pulled to `runs/` and committed as each completes; final compiled outputs follow completion.*
