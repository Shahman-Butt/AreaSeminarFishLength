# Professor Question & Recommendation Checklist

Every point the supervisor raised (two emails), mapped to what we did, the evidence, and the exact
repository files. ✅ = addressed with results · 🟡 = partially addressed · ⛔ = blocked/scoped.

Repo: https://github.com/Shahman-Butt/AreaSeminarFishLength

---

## Email 1 — direction & poster

| # | His point | Status | Evidence / result | Files |
|---|---|---|---|---|
| 1 | Did you test **species / class identification**? | ✅ | 4 encoders trained; accuracy 95.1–99.6% (ConvNeXt 0.9957, MobileNetV2 0.9912, DINOv2 0.9819, CLIP 0.9513) | `runs/cls_*`, `src/autofish_vfm/train_classifier.py`, `evaluate_classifier.py`, `configs/cls_*.json` |
| 2 | Did you test **mask segmentation**? | ⛔ | Not done; needs a Mask2Former-style stack (multi-day). Scoped as future work. | `docs/EXPERIMENT_AUDIT.md` §3d |
| 3 | Do these tasks show the **same trend** (CNN > foundation models)? | 🟡 | Directionally yes on classification (CNNs on top), but gaps are tiny (95–99.6%) and **DINOv2 jumps from worst-at-length to near-top-at-species** — evidence that foundation features suit semantics over precise geometry. | `runs/cls_*/test_metrics.json`, `docs/AutoFish_Final_Report.docx` §4.3 |
| 4 | Add **example dataset images** to the poster | ✅ | Example masked crops (non-occluded + occluded) generated | `results/qualitative/dataset_examples.png`, `scripts/make_qualitative_figures.py` |
| 5 | Add **visualized results** — where DINOv2 and MobileNet differ | ✅ | Figure of the largest MobileNet-right / DINOv2-wrong cases (all small flatfish DINOv2 over-predicts to 34–38 cm) | `results/qualitative/mobilenet_vs_dino.png` |
| 6 | Analyze **when CLIP works better vs MobileNet** | ✅ | CLIP beats MobileNet on 38.6% of individual fish (40.5% on occluded); per-species/size/occlusion tables | `results/error_analysis/`, `results/qualitative/clip_wins.png`, `scripts/error_analysis.py` |

---

## Email 2 — code verification

| # | His question | Status | Answer & evidence | Files |
|---|---|---|---|---|
| 1 | Did you try **DINOv3** as backbone? | ⛔ | Architecture loads, but Meta's pretrained weights are license-gated (official download → HTTP 403 Forbidden). Needs access-approved weights. | `docs/EXPERIMENT_AUDIT.md` §3c |
| 2 | **CLS token or patch tokens** for DINOv2? | ✅ | Originally CLS only. We added patch-token mean-pooling and ran a controlled 3-seed test: **patch reliably beats CLS by 0.58 cm** (1.261 ± 0.054 vs 1.843 ± 0.023, non-overlapping ranges). Best DINOv2 config. | `src/autofish_vfm/models.py` (`use_patch_tokens`), `runs/dinov2_vits14_patchtokens_frozen*`, `runs/dinov2_vits14_clstoken_ctrl_s*` |
| 3 | Poster says **"same parameters as paper"** but GitHub shows **different learning rates** | ✅ | Clarified on poster: the baseline reproduces the paper (Adam 1e-3); foundation models use lower LRs (1e-4; 1e-5/1e-6 for DINOv2 full FT) as pretrained encoders require. Identical across models: data, split, input, head, loss, metrics. | `poster/AutoFish_A3_poster.html` (Method block) |

---

## Bonus: things done beyond what was asked

| Item | Evidence | Files |
|---|---|---|
| Controlled multi-seed reliability (patch vs CLS, HP-matched) | 3 seeds each, non-overlapping ranges | `runs/dinov2_vits14_*_s{42,1,2}` |
| Ablation: patch + last-block fine-tune | 1.345 cm — does not beat frozen patch pooling | `runs/dinov2_vits14_patchtokens_lastblock` |
| No-repeat audit of all prior computation | Full inventory | `docs/EXPERIMENT_AUDIT.md` |
| Hyperparameter-tuning guide (layman) | Why/what/how per model | `docs/PROJECT_COMPLETE_GUIDE.md` §G |

---

## One-line answers to have ready

- **Species?** "Yes — 95 to 99.6% accuracy; CNNs top but foundation models very close, and DINOv2 jumps up, which supports our semantics-vs-geometry story."
- **Segmentation?** "Not yet — it needs a heavy Mask2Former setup; it's scoped as future work."
- **DINOv3?** "The code loads it but Meta gates the weights (403 Forbidden); we'd need access approval."
- **CLS or patch tokens?** "We used CLS, then tested patch tokens — they reliably win by 0.58 cm over three seeds."
- **Same parameters as the paper?** "Baseline yes; foundation models use lower learning rates by necessity; everything else identical. The poster now says this."
