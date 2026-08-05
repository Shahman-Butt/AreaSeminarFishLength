# Viva Prep — Part 2: Our Project, End to End

Builds on Part 1 (`VIVA_PREP_PART1_PAPER.md`). Every fact below is grounded in this project's actual
code, configs, and saved run results — verified against `runs/*/test_metrics.json` and
`configs/*.json`, not recalled from memory.

---

## 1. Research question, motivation, objectives

**Research question (this is OUR question, not the paper's):**
> Can a modern image encoder or vision foundation model beat the reproduced AutoFish MobileNetV2
> baseline at fish length regression, under identical data, split, and metrics?

**Motivation (layman):** The paper proved a small CNN can measure fish automatically. Since then,
much bigger "foundation models" (DINOv2, CLIP) have become popular and are often assumed to be
better at everything. We wanted to actually test that assumption on this precise, narrow task,
rather than take it on faith.

**Motivation (technical):** Foundation models are typically benchmarked on broad semantic tasks
(classification, retrieval, captioning). Precise **metric regression** (predicting a continuous
physical quantity to sub-centimetre accuracy) is a qualitatively different demand — it needs fine
spatial/geometric information, not just "what object is this." This project is an empirical test of
whether that semantic strength transfers to a geometric regression task.

**Objectives, in order:**
1. Reproduce the paper's `REG` baseline faithfully (calibrate the whole pipeline).
2. Run a controlled encoder-swap comparison (CNNs vs. foundation models) on length regression.
3. Investigate *why* any encoder underperforms, not just report that it does (patch-token study).
4. Test whether the CNN-vs-foundation-model pattern generalizes to a second task (species
   classification).
5. Push the closest single model (EfficientNet-B0) as far as possible with a controlled recipe search.

---

## 2. What came from the paper vs. what WE did — the honest accounting

| Item | Source |
|---|---|
| Dataset (images, masks, lengths, IDs, groups) | **100% the paper's** (Bengtson et al.) |
| Official train/val/test group split | **100% the paper's** (confirmed in Part 1 §6.1 — not independently invented) |
| REG architecture (MobileNetV2 + bbox + 2-layer head) | **100% the paper's** (their Fig. 5) |
| REG training recipe (L1, Adam, LR 1e-3, batch 32, 200 epochs) | **100% the paper's** (their §4.2.2) |
| Reproducing REG and checking it matches | **Our work** — validates our pipeline is faithful |
| Leakage audit finding/fixing fish 113 | **Our work** — an extra verification step not described as a specific finding in the paper |
| Testing DINOv2, CLIP, ConvNeXt, EfficientNet-B0 | **Our novel contribution** — not in the paper at all |
| Patch-token vs. CLS-token study for DINOv2 (3-seed controlled) | **Our novel contribution** |
| Species classification task | **Our novel contribution** — not in the paper |
| Stronger-recipe search for EfficientNet-B0 | **Our novel contribution** |
| Mask2Former segmentation, SAM annotation, skeletonization (SKL) | **Not touched by us at all** — paper-only, explicitly out of scope |

**One-sentence viva answer to "what's novel about your work?":**
> "We reproduced the paper's regression baseline faithfully, then extended it with a research
> question the paper never asks: whether newer supervised CNNs and vision foundation models can
> beat that baseline, including a controlled study of *why* one of them (DINOv2) initially failed
> and how to fix it."

---

## 3. The preprocessing pipeline, step by step (raw image → tensor)

```
Raw photo (2464×2056 px, from the paper's camera)
      │
      ▼
Polygon outline (COCO-format, human-corrected via SAM — from the paper's annotation)
      │  scripts/make_crops.py: mask_from_polygons()
      ▼
Black/white mask (0 = background, 255 = this one fish)
      │  scripts/make_crops.py: Image.paste(img, mask=mask)
      ▼
Masked image (everything except this fish blacked out — other/overlapping fish disappear)
      │  scripts/make_crops.py: square_bbox_from_mask()
      ▼
Square window found (centred on the fish, side = max(width, height) of its mask)
      │  scripts/make_crops.py: crop_with_padding()
      ▼
Cropped + padded (black-padded if the square ran off the original photo's edge)
      │  scripts/make_crops.py: .resize((224,224))
      ▼
224×224 PNG saved to disk (data/processed/crops/, ONE per fish, cached — done once, reused forever)
      │  src/autofish_vfm/data.py: CropDataset.__getitem__
      ▼
transforms.Resize → transforms.ToTensor()  (pixels 0–255 → 0.0–1.0, HWC → CHW)
      │
      ▼
[transforms.ColorJitter]  (TRAINING split only: brightness/contrast/saturation/hue jitter)
      │
      ▼
transforms.Normalize(mean, std)  (ImageNet statistics — matches what the pretrained encoder expects)
      │
      ▼
Final tensor: shape [3, 224, 224], plus a separate bbox tensor [4] (from the ORIGINAL,
un-cropped photo's coordinates, normalized to 0..1)
      │  torch.utils.data.DataLoader
      ▼
Batched tensors → GPU → model
```

**Why each step exists (the "why," not just the "what"):**
- **Masking** exists because photos in the "All" (occluded) set contain multiple overlapping fish;
  without masking, the model would have no way to know *which* fish's length to predict.
- **Square crop** (not a plain rectangle) exists because the final resize to a fixed 224×224 would
  otherwise stretch a non-square rectangle, distorting the fish's true proportions — directly
  corrupting the length signal.
- **Caching to PNG once** exists purely for speed: masking/cropping is done once for all 18,157
  fish, so every one of the ~15 experiments this project ran doesn't redo that work.
- **ColorJitter, training only** exists to prevent the model from memorizing exact lighting/color
  instead of learning general size cues; it's turned OFF for val/test so evaluation is always on
  the real, unperturbed image.
- **Normalize with ImageNet stats** exists because every pretrained encoder (MobileNetV2, DINOv2,
  etc.) was itself trained on images normalized this way — skipping it would feed the encoder
  inputs far outside the numeric range its pretrained weights expect, badly hurting transfer.
- **The bbox tensor, from the ORIGINAL photo** exists because resizing every crop to the same
  224×224 destroys the fish's real scale (a tiny fish and a huge fish look the same size in the
  crop); these 4 numbers are how the model recovers "how big was this fish, really?"

---

## 4. The dataset, precisely

| Fact | Value | Source |
|---|---:|---|
| Images | 1,500 | Paper + our `index.csv` |
| Unique fish (specimens) | 454 | Paper + our `index.csv` |
| Instance annotations | Paper: 18,160 · **Ours (after leakage fix): 18,157** | See note below |
| Groups | 25 | Paper + ours |
| Train / Val / Test groups | 15 / 5 / 5 | **Confirmed identical to paper**, see Part 1 §6.1 |
| Test set size | 3,759 (1,879 non-occluded + 1,880 occluded) | Our `index.csv`, post-leakage-fix |

**⚠️ Honest, unresolved discrepancy to know about:** the paper states 18,160 total instance masks;
our processed index has 18,157 after removing 1 leaked annotation (which implies our *raw*,
pre-cleanup count was 18,158 — 2 short of the paper's stated 18,160). This small gap (2–3
annotations) is not fully explained and has not been investigated further. **If asked in the viva,
say exactly this** — a small, acknowledged, unexplained count discrepancy — rather than claiming an
exact match that isn't quite there.

**The leakage audit (our own verification work, described fully in earlier project docs):**
`fish_id 113` had 41 annotations: 40 in test group 22, and 1 stray annotation in train group 5 (a
duplicate). We removed the single stray annotation, re-verified zero fish now cross any split, and
logged the fix in `data/processed/exclusions.json`. This is why the non-occluded test count is
1,879, not 1,880.

---

## 5. What we compared — the controlled encoder-swap design

**The controlled variable:** only the encoder (the "eye") changes between experiments. Held fixed
across every length-regression run: data, split, crop pipeline, bbox input, regression-head *shape*
family, L1 loss, and evaluation protocol.

**What legitimately differs by necessity (and why that's not a flaw):** learning rate and epoch
count differ between the baseline (paper's own recipe: 1e-3, 200 epochs) and the encoder-swap
experiments (1e-4, 100 epochs) — because pretrained foundation-model encoders can be destabilized by
an aggressive learning rate, so a gentler recipe was used for all the swap experiments uniformly.
This is disclosed explicitly, not hidden.

**The full set of things tested (9 length-regression experiments + 6 more from the reliability/recipe round):**
MobileNetV2 (baseline) · ConvNeXt-Tiny · EfficientNet-B0 (basic + 3 stronger recipes) · CLIP
(frozen + last-block) · DINOv2 (CLS frozen, CLS last-block, CLS full-FT ×2 learning rates, patch
frozen ×3 seeds, patch + last-block) · plus 4 species-classification runs.

---

## 6. Results — the overview (full detail deferred to Part 6)

| Rank | Model | Full-test MAE | Status |
|---:|---|---:|---|
| 1 | MobileNetV2 (baseline) | 0.771 cm | Best single model |
| 2 | EfficientNet-B0 (basic recipe) | 0.781 cm | Closest challenger — single run, unconfirmed |
| 3 | ConvNeXt-Tiny | 0.914 cm | |
| 4–5 | CLIP (last-block / frozen) | 0.958 / 1.002 cm | |
| 6 | DINOv2 (patch-token, 3-seed) | 1.261 ± 0.054 cm | Reliable, confirmed improvement over CLS |
| 7–10 | DINOv2 (CLS variants) | 1.439 – 2.132 cm | |

**Species classification (all encoders, second task):** 95.1–99.6% accuracy across the board —
ConvNeXt best (99.57%), DINOv2 surprisingly near the top (98.19%) despite being worst at length.

---

## 7. Conclusions (as currently stated)

1. No single model has beaten the baseline yet; EfficientNet-B0 is closest but unconfirmed
   (single-seed).
2. A reliable, statistically-confirmed improvement exists *within* DINOv2 (patch pooling vs. CLS).
3. The CNN-vs-foundation-model pattern seen in length regression does **not** fully hold for
   classification — DINOv2's relative ranking changes dramatically by task, suggesting foundation
   models suit semantic tasks better than fine geometric ones. (💡 hypothesis, not proven.)
4. A stronger training recipe made EfficientNet-B0 *worse* (0.934 cm), not better — evidence the
   basic recipe may already be near this architecture's ceiling on this dataset size.

## 8. Future work (as currently stated)

Multi-seed the top single models (only DINOv2 has this so far); finish the stronger-recipe search;
try patch pooling for CLIP; test larger model variants (EfficientNet-B2, bigger ViTs); evaluate mask
segmentation as a third task (never attempted); pursue DINOv3 (blocked — license-gated weights,
confirmed HTTP 403 on the official download).

---

## 9. Quick-fire viva Q&A — Part 2 (project-specific)

**Q1: What's the difference between your research question and the paper's?**
The paper asks "can we build a working automatic fish-measuring system?" We ask "given that working
system's baseline, can newer image encoders beat it?" — a comparison study the paper never runs.

**Q2: Why keep the paper's exact split instead of making your own?**
To make results directly comparable to the paper's numbers, and because a different split would
answer a different, incomparable question — plus the split is specifically designed (per Part 1) to
avoid fish-identity leakage across train/test.

**Q3: What's the ONE thing you changed from the paper's dataset processing?**
We found and removed a single leaked fish (ID 113) that the paper's own release did not flag; this
is a defensive quality check on top of the paper's design, not a change to their split rule itself.

**Q4: Why does the bbox come from the original photo, not the crop?**
The crop is always resized to 224×224, destroying absolute scale; the bbox (from the un-cropped
photo, normalized 0–1) is how the model recovers the fish's true size in the original scene.

**Q5: Is your project a full reproduction of the AutoFish paper?**
No — only the REG (CNN regression) baseline is reproduced. Segmentation (Mask2Former) and
skeletonization (SKL) were never touched; that's explicitly out of scope.

**Q6: What's genuinely novel in your project versus just "running more models"?**
The controlled 3-seed patch-token study for DINOv2 is a genuine, statistically-grounded finding (not
just a benchmark run), and the species-classification cross-check tests whether a length-regression
conclusion generalizes — that's a deliberate scientific design choice, not incidental extra work.

**Q7: Why keep learning rate different between the baseline and the swap experiments — isn't that unfair?**
It's a necessary, disclosed compromise: the baseline uses the paper's own recipe (for faithful
reproduction); the swap experiments use a gentler, uniform recipe across all encoders (because
foundation models need gentler fine-tuning) — the comparison among the *swap* encoders is fully
controlled; only the baseline vs. swaps comparison carries this caveat, and it's stated openly.

---

*Part 2 of the staged viva-prep series. Part 3 (encoder deep-dives) continues next.*
