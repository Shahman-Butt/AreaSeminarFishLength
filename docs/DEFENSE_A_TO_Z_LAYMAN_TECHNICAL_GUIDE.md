# AutoFish Defense Guide: A to Z (Layman + Technical)

A complete, sequential, guided story of the project — written so you can defend it confidently in
front of your professor. Every important concept is explained twice: **Layman** (simple, intuitive)
and **Technical** (precise ML/CV terminology). All numbers below are read from the project's saved
run files (`runs/*/test_metrics.json`, `configs/*.json`, `results/error_analysis/`) — nothing is
invented. Where a number could not be found locally, it is marked "not found locally."

Team: Abu Bakar, Laksh Jiwani, Shahman Butt · Supervisor: Bohan Zhuang, M.Sc. · Professor: Stefan
Oehmcke · University of Rostock (VACOT)
Repository: https://github.com/Shahman-Butt/AreaSeminarFishLength

---

## 1. What problem are we solving?

**Layman:** We want a computer to look at a photo of a fish and tell us how long it is, in
centimetres — without a human touching a ruler.

**Technical:** This is a supervised **regression** problem: given an image (plus a bounding box), predict
a single continuous scalar, the fish's length in cm.

## 2. Why does fish length estimation matter?

**Layman:** Fisheries scientists need fish lengths to judge how healthy a fish population is and
whether catches respect legal size limits. Measuring every fish by hand is slow, and it stresses the
fish. A camera-based system could measure automatically.

**Technical:** Length-frequency data feeds stock-assessment models (growth curves, biomass estimates,
catch-quota compliance). Automating measurement from imagery removes a manual bottleneck and reduces
handling stress on live specimens.

## 3. What paper are we reproducing?

**Layman:** A recent research paper called AutoFish built a camera-based fish-measuring system and
published their results. Our first job was to rebuild their system and check we get the same numbers.

**Technical:** Bengtson et al., *"AutoFish: Dataset and Benchmark for Fine-grained Analysis of Fish"*,
WACV Workshops 2025 (arXiv:2501.03767). We reproduce their **REG MobileNetV2** length-regression
baseline.

## 4. What dataset is used?

**Layman:** A collection of 1,500 photographs of real fish laid on a table, with every fish's true
length written down by hand.

**Technical:** Hugging Face dataset `vapaau/autofish`. After preprocessing: **18,157** crops/annotations,
**1,500** images, **454** unique fish, **25** groups. Each annotation carries a species label, a
`fish_id`, a hand-measured `length` (cm), a bounding box, and a polygon segmentation mask.

## 5. How is the data split?

**Layman:** We divide the fish into three buckets — practice (train), a check-in set (validation), and
a final exam (test) — and we make sure a fish is never in two buckets at once, by splitting on whole
groups of fish rather than individual photos.

**Technical:** Official **group-level** split:
- Train groups: 2, 3, 4, 5, 7, 8, 9, 12, 13, 15, 16, 18, 19, 23, 24
- Validation groups: 1, 6, 11, 17, 25
- Test groups: 10, 14, 20, 21, 22

Test set = 3,759 annotations (1,879 non-occluded + 1,880 occluded).

**Set meanings — Layman:** Set1+Set2 are photos of fish laid out separately (easy). "All" is photos of
fish piled up and overlapping (hard, realistic).
**Technical:** Set1+Set2 = non-occluded regime; "All" = occluded regime. Both are reported separately
because occlusion materially changes difficulty (see §16).

## 6. What are preprocessing, crops, masks, and bounding boxes?

**Layman:** For each fish we cut out a small, square picture of just that fish (a "crop"), we black out
everything that isn't that fish (a "mask", useful when fish overlap), and we remember the rectangle
("bounding box") that originally contained the fish in the full photo, because resizing the crop hides
how big the fish really was.

**Technical:**
- **Crop** — a square region cut from the source image, aspect-ratio preserving, centered on the
  fish's mask, resized to 224×224.
- **Mask** — a binary/instance segmentation used to zero out background and other fish, isolating the
  target instance even under occlusion.
- **Bounding box (bbox)** — 4 values (x, y, w, h), normalized by image width/height, concatenated to
  the encoder's feature vector before the regression head; restores absolute scale information lost by
  resizing.

## 7. What is leakage, and what does the fish-113 exclusion mean?

**Layman:** "Leakage" is when the same fish accidentally ends up on both the practice side and the
exam side. That's cheating — the model could just recognise that individual fish instead of learning
to measure fish in general. We found exactly one case (fish ID 113) where this happened through a
stray duplicate photo record, and we removed it.

**Technical:** An automated audit groups all annotations by `fish_id` and counts distinct splits per
fish; any count > 1 is a leak. `fish_id=113` had 41 annotations: 40 in a test group and 1 in a train
group (via a singleton duplicate). The train-side annotation (id 3759) was removed and logged in
`exclusions.json`; the audit then re-verifies **zero** fish cross any split. This is why the
non-occluded test count is 1,879, not 1,880.

## 8. What baseline did we reproduce?

**Layman:** We rebuilt the paper's original small measuring model and tested it on our exam fish. Our
result landed extremely close to what the paper reported — proof that our whole pipeline works
correctly.

**Technical:**

| | Paper (Bengtson et al.) | Our reproduction |
|---|---:|---:|
| Non-occluded MAE | 0.62 cm | **0.633 cm** |
| Occluded MAE | 1.38 cm | **0.909 cm** |
| Full-test MAE | not reported this way | **0.771 cm** |

Δ non-occluded = 0.013 cm (≈2% relative) → reproduction validated. The occluded result is *better*
than the paper's; the exact reason is unknown (see §16 and the Q&A).

## 9. What models did we try, and why?

**Layman:** We tested several different "eyes" for the same measuring machine: the original small
network, two newer/bigger networks trained the classic way, and two "foundation models" — giant
AI models trained on huge amounts of general images.

**Technical:** Encoders tested (all with the same bbox-concatenation + MLP regression head):
MobileNetV2 (baseline), EfficientNet-B0, ConvNeXt-Tiny (supervised-ImageNet CNNs), CLIP ViT-B/32,
DINOv2 ViT-S/14 (vision foundation models). Only the encoder differs between length experiments —
split, input, head, loss, and metrics are held fixed.

## 10. How does each model work (layman + technical)?

### MobileNetV2
**Layman:** A compact, efficient visual brain, originally designed to run on phones. It's the
"apprentice" trained specifically for this exact job.
**Technical:** A CNN built from **inverted residual blocks** and **depthwise-separable convolutions**,
~3.4M parameters, ImageNet-pretrained, fully fine-tuned end-to-end for this task.

### EfficientNet-B0
**Layman:** A more modern, carefully "balanced" compact network — scaled up in width, depth, and
resolution together rather than by guesswork.
**Technical:** CNN using **MBConv blocks** with **compound scaling**, ~5.3M parameters, ImageNet
V1-pretrained, fully fine-tuned.

### ConvNeXt-Tiny
**Layman:** A modern CNN (2022) that borrowed good ideas from Transformers while staying a classic
convolutional design.
**Technical:** ~28M-parameter CNN using large-kernel depthwise convolutions, LayerNorm, and GELU,
ImageNet-pretrained, fully fine-tuned.

### CLIP ViT-B/32
**Layman:** A model that learned by matching millions of internet pictures to their captions, so it
understands images in relation to language.
**Technical:** OpenAI/OpenCLIP ViT-B/32, contrastively pretrained on ~400M image–text pairs. Tested
**frozen** (only the head trains) and **last-block fine-tuned** (final visual transformer block +
final norm/projection unfrozen, warm-started from the frozen checkpoint).

### DINOv2 ViT-S/14
**Layman:** A model that taught itself to understand images with no labels and no captions at all —
just by comparing different views of the same picture.
**Technical:** Self-supervised Vision Transformer (DINO/iBOT objectives), ~21M parameters. Tested
**frozen** (CLS token and patch-token variants), **last-block fine-tuned**, and **full fine-tune** at
two encoder learning rates (1e-5, 1e-6).

## 11. How is the code structured?

**Layman:** The project is organized like a small factory: one part loads and prepares the fish
photos, one part defines each "brain" design, one part runs the training, one part checks the exam
score, and separate parts do the same for the species-guessing task.

**Technical:**
- `src/autofish_vfm/data.py` — dataset loading, crop/transform pipeline, bbox features, regression
  and classification label handling.
- `src/autofish_vfm/models.py` — model definitions (MobileNetV2, EfficientNet, ConvNeXt, CLIP, DINOv2)
  and the shared MLP regression head.
- `src/autofish_vfm/train_baseline.py` — length-regression training loop.
- `src/autofish_vfm/evaluate.py` — regression evaluation + per-fish prediction CSV export.
- `src/autofish_vfm/train_classifier.py` — species-classification training loop.
- `src/autofish_vfm/evaluate_classifier.py` — classification evaluation.
- `src/autofish_vfm/metrics.py` — MAE/RMSE/MAPE/bias/R² and classification metrics (accuracy, macro-F1).
- `configs/` — one JSON per experiment (all hyperparameters, versioned).
- `runs/` — saved outputs per experiment: `config.json`, `history.csv`, `test_metrics.json`,
  `test_metrics.predictions.csv`.
- `scripts/` — persistent training queues, chart generation, error analysis, report builders.
- `results/error_analysis/` — CSV breakdowns by occlusion/species/length-range, head-to-head tables.
- `results/qualitative/` — example figures (dataset crops, model-disagreement cases).
- `poster/` — poster HTML/PDF.

## 12. How does training work?

**Layman:** Show the model a batch of fish crops, let it guess the length, compare to the true length,
nudge the model slightly to be less wrong, and repeat this thousands of times. Periodically, check
performance on a separate "referee" set (validation) and keep the version that did best there.

**Technical pipeline:**
1. Load `index.csv` and crop images.
2. Filter by official group IDs → train/val/test split.
3. Apply transforms (resize 224², ImageNet normalization; ColorJitter augmentation on train only).
4. Feed the image crop through the encoder → feature vector.
5. Concatenate the 4 normalized bbox values.
6. Feed into the MLP regression head.
7. Predict `length_cm`.
8. Compute L1 loss against ground truth.
9. Backpropagate; update weights via Adam for the configured number of epochs.
10. After each epoch, evaluate on validation; save `best.pt` whenever validation MAE improves.
11. Evaluate the best checkpoint once on the held-out test set.
12. Save `test_metrics.json` and a per-fish `test_metrics.predictions.csv`.

## 13. What configurations / hyperparameters were used?

**Layman:** Each model has a recipe: how big a step to adjust by (learning rate), how many fish to
look at before adjusting (batch size), how many times to go through all the training fish (epochs),
and how much of the model is allowed to change (frozen vs. fine-tuned).

**Technical** (verified from `configs/*.json`):

| Model | Head | Loss | Optimizer | LR | Batch | Epochs | Encoder frozen? |
|---|---|---|---|---:|---:|---:|---|
| MobileNetV2 (baseline) | [1000, 500, 1] | L1 | Adam | 1e-3 | 32 | 200 | No (full FT) |
| EfficientNet-B0 | [512, 128, 1] | L1 | Adam | 1e-4 | 16 | 100 | No (full FT) |
| ConvNeXt-Tiny | [512, 128, 1] | L1 | Adam | 1e-4 | 16 | 100 | No (full FT) |
| CLIP frozen | [512, 128, 1] | L1 | Adam | 1e-4 | 16 | 100 | Yes |
| CLIP last-block | [512, 128, 1] | L1 | Adam | enc 1e-4 (block only) | 16 | 100 | Partial |
| DINOv2 CLS frozen | [512, 128, 1] | L1 | Adam | 1e-3 | 32 | 100 | Yes |
| DINOv2 patch frozen | [512, 128, 1] | L1 | Adam | 1e-4 | 16 | 100 | Yes |
| DINOv2 last-block | [512, 128, 1] | L1 | Adam | enc/head split | 8 | 50 | Partial |
| DINOv2 full FT | [512, 128, 1] | L1 | Adam | enc 1e-5 / 1e-6 | varies | 100 | No |

All configs: `image_size=224`, `pretrained=true`, `bbox_input=true`, `normalize_bbox=true`, seed 42
(patch-vs-CLS controlled study additionally uses seeds 1, 2).

Species-classification configs (verified, all identical apart from encoder): head `[512, 128, 7]`,
cross-entropy loss, Adam, LR 1e-4, batch 16, 60 epochs; MobileNetV2/ConvNeXt fully fine-tuned; CLIP and
DINOv2 frozen.

**Why different learning rates?** Pretrained foundation-model encoders can lose useful pretrained
structure if fine-tuned too aggressively, so they use gentler (lower) learning rates than the
baseline, which reproduces the paper's own recipe (Adam 1e-3).

## 14. What do the metrics mean?

**Layman examples**, using a fish that is truly 31.0 cm and predicted at 30.5 cm (error = -0.5 cm):

- **MAE** (Mean Absolute Error): average |error| across all fish — "on average, how many cm off are
  we?" Our best model: 0.771 cm.
- **RMSE** (Root Mean Squared Error): like MAE but squares errors first, so big mistakes count extra —
  reveals whether some predictions are very far off.
- **MAPE** (Mean Absolute Percentage Error): the error as a % of the true length — "off by about 2.4%
  of the fish's size."
- **Bias**: average signed error — positive means the model tends to over-predict, negative means
  under-predict, near-zero means balanced.
- **R²**: how much of the natural variation in length the model explains (1.0 = perfect).
- **Accuracy** (classification): fraction of fish whose species was correctly guessed.
- **Macro-F1** (classification): balances precision and recall per species and averages across
  species equally, so it doesn't get inflated by common species alone.

**Technical formulas:** MAE = mean(|ŷ−y|); RMSE = √mean((ŷ−y)²); MAPE = mean(|ŷ−y|/y)×100; Bias =
mean(ŷ−y); R² = 1 − SS_res/SS_tot; Accuracy = correct/total; Macro-F1 = mean over classes of
2·precision·recall/(precision+recall).

## 15. What results did we get?

**Length regression — full-test MAE (cm), lower is better:**

| Rank | Model | Full | Non-occluded | Occluded |
|---:|---|---:|---:|---:|
| 1 | **MobileNetV2 (baseline)** | **0.771** | 0.633 | 0.909 |
| 2 | **EfficientNet-B0** | **0.781** | 0.670 | **0.893** |
| 3 | ConvNeXt-Tiny | 0.914 | 0.814 | 1.014 |
| 4 | CLIP last-block | 0.958 | 0.842 | 1.074 |
| 5 | CLIP frozen | 1.002 | 0.898 | 1.106 |
| 6 | DINOv2 patch-token frozen | 1.199–1.261* | ~1.11–1.34 | ~1.29–1.54 |
| 7 | DINOv2 last-block (CLS) | 1.439 | 1.340 | 1.537 |
| 8 | DINOv2 frozen (CLS) | 1.738 | 1.690 | 1.786 |
| 9 | DINOv2 full FT (enc LR 1e-5) | 1.778 | 1.636 | 1.919 |
| 10 | DINOv2 full FT (enc LR 1e-6) | 2.132 | 2.075 | 2.189 |

*DINOv2 patch-token frozen: single seed (42) = 1.199; across 3 seeds (42, 1, 2) mean ± sd =
1.261 ± 0.054 — see §17.

**Key interpretation: no single model beats the MobileNetV2 baseline yet.** But **EfficientNet-B0 is
remarkably close** (0.781 vs 0.771, only 0.010 cm behind) — and interestingly, **EfficientNet-B0 is
actually *better* than the baseline on occluded fish** (0.893 vs 0.909 cm) while being *worse* on
non-occluded fish (0.670 vs 0.633 cm), so the full-test average lands just behind.

## 16. What does error analysis show?

**By occlusion:** every model degrades on occluded fish (the "occlusion penalty" — the gap between
occluded and non-occluded MAE — ranges from 0.096 cm for frozen DINOv2 to 0.282 cm for its full
fine-tuned variant).

**By species:** haddock is consistently easiest (MobileNetV2: 0.587 cm); hake and the "other" category
are hardest for most models.

**By length range:** a U-shape — the shortest (22.5–28cm) and longest (37.5–50.5cm) fish are hardest
for every model; middle-length fish are easiest.

**CLIP vs MobileNetV2, per fish:** although MobileNetV2 wins on *average*, **CLIP (last-block) beats
MobileNetV2 on 38.6% of individual test fish** — 36.7% on non-occluded fish, 40.5% on occluded fish.
This shows the "MobileNetV2 is best" story is true on average but not universally true fish-by-fish.

**DINOv2 disagreements:** the largest MobileNetV2-right/DINOv2-wrong cases are all small, wide-bodied
flatfish (~22.5 cm) that DINOv2 over-predicts to 34–38 cm — visualized in
`results/qualitative/mobilenet_vs_dino.png`.

## 17. What does "patch tokens vs CLS token" mean, and why did patch tokens help?

**Layman:** A Vision Transformer can describe an image either as one overall summary ("CLS token") or
as a grid of many local descriptions ("patch tokens"), one per small region. The CLS summary is great
for "what is this?" but throws away exactly the spatial/size detail that measuring length needs.
Patch tokens keep that detail.

**Technical:** We ran a controlled, matched-hyperparameter, 3-seed comparison (seeds 42, 1, 2), only
the token type differing:

| DINOv2 frozen | Mean ± SD (full-test MAE) |
|---|---|
| CLS token | 1.843 ± 0.023 |
| Patch token (mean-pooled) | 1.261 ± 0.054 |

Patch pooling gives a **0.58 cm reliable improvement**, with non-overlapping seed ranges — a controlled,
statistically credible result, and the best DINOv2 configuration found. Adding last-block fine-tuning
on top of patch pooling did **not** help further (0.771→ actually 1.345 cm, worse than frozen patch
pooling), suggesting the frozen patch representation is already close to what this small dataset can
support.

## 18. What does "ensemble" mean, and what did ours achieve?

**Layman:** Instead of trusting one model, ask several models for their guess and average the answers.
If MobileNetV2 says 30.2 cm, ConvNeXt says 30.8 cm, and CLIP says 30.5 cm, the ensemble's answer is the
average, about 30.5 cm. Averaging often smooths out each individual model's mistakes.

**Technical:** A **validation-selected weighted ensemble** — weights chosen by grid search on the
**validation** set only, then applied once to the **test** set (no test-tuning). Result: **MobileNetV2
× 3 + ConvNeXt × 1 + CLIP-frozen × 1** (normalized weights), validation MAE 0.744 cm →

| | Baseline | Ensemble | Improvement |
|---|---:|---:|---:|
| Full-test | 0.771 | **0.711** | +0.060 cm (+7.8%) |
| Non-occluded | 0.633 | 0.600 | beats baseline |
| Occluded | 0.909 | 0.822 | beats baseline |

**Important caveat, stated honestly:** this is **not a new architecture** — it is a combined-prediction
system that runs three trained models and averages their outputs. It is scientifically valid *because*
the weights were chosen on validation and reported once on test (avoiding test-set tuning), but it costs
**more inference compute** (three forward passes instead of one), and the underlying single models are
each single-seed runs. It should not be presented as "a model that beats the baseline" — it should be
presented as "a combination strategy that beats the baseline."

## 19. What does the poster show?

**Layman:** The current poster tells the honest, single-model story: MobileNetV2 is still the best
individual model, EfficientNet-B0 is remarkably close, DINOv2's patch tokens are a reliable
improvement within that model family, and species classification works very well across all encoders.
It also shows where we plan to keep pushing (stronger training recipes for EfficientNet/ConvNeXt).

**Technical:** `poster/AutoFish_A3_poster.html` / `.pdf` — single-model focus (the ensemble is
intentionally *not* the headline, per project decision), with: the full ranking chart, an
"EfficientNet-B0 closest to baseline" highlight chart, the patch-vs-CLS controlled-comparison chart,
the species-classification accuracy chart, a configurations table, and a "future work" panel describing
the ongoing stronger-recipe search.

## 20. What limitations remain?

- No single model has yet beaten the MobileNetV2 baseline on full-test MAE (EfficientNet-B0 is
  closest, single-run).
- Most single-model results are **single-run** (fixed seed 42); only the DINOv2 patch-vs-CLS
  comparison has multi-seed statistics (n=3).
- Foundation models were tested at small scale (ViT-S/14, ViT-B/32); larger variants untested.
- DINOv3 was investigated but is blocked by license-gated pretrained weights (HTTP 403 on official
  download) — not found to be freely accessible locally.
- Mask segmentation (a task the original paper also studies) has not been evaluated in our project.
- The occluded-set reproduction difference vs. the paper (0.909 vs. 1.38 cm) remains unexplained.
- The ensemble result, while valid, uses more inference compute and is not itself a new architecture.

---

# Professor Defense Q&A

**1. Did you reproduce the baseline?**
Yes. Non-occluded MAE 0.633 cm vs. the paper's 0.62 cm (Δ 0.013 cm).

**2. Why is your baseline valid?**
Same official group split, same architecture (MobileNetV2 + bbox + regression head), same loss (L1),
matching the paper's reported non-occluded result within 2%.

**3. Why does your occluded result differ from the paper (0.909 vs. 1.38 cm)?**
Unknown with certainty — our result is *better* than the paper's, and we do not have their exact
training/augmentation details to explain the gap. Flagged honestly as an open question, not hidden.

**4. Why use a group split instead of a random per-image split?**
Because the same physical fish appears in many photos; a random split would let a fish's identity leak
across train/test, letting the model "recognize" a fish instead of measuring it — invalidating the test
score.

**5. What is leakage?**
Test-set information (here: fish identity) unintentionally present in training data, producing an
optimistic, invalid evaluation.

**6. Why was fish_id=113 removed?**
It had annotations split across train and test groups (40 in test, 1 in train via a duplicate record).
The single train-side annotation was removed to eliminate identity leakage; verified by re-running the
leakage audit (result: zero leaks).

**7. Why use bbox input?**
Because resizing every crop to 224×224 destroys absolute scale information; the bounding box restores
it as 4 extra numeric features. The original paper uses the same design, which keeps our comparison
consistent.

**8. Why crop to the bounding box?**
To give the network a tight, size-normalized view of a single fish rather than a whole cluttered scene.

**9. Why use masks?**
To isolate the target fish's pixels from background and from other overlapping fish, which matters
most for the occluded ("All") image regime.

**10. Why resize to 224×224?**
Standard input resolution matching the pretrained weights of all tested encoders (ImageNet-style CNNs
and ViT-based models), enabling transfer learning.

**11. What is MobileNetV2?**
A lightweight CNN using inverted residual blocks and depthwise-separable convolutions; the paper's
original baseline encoder.

**12. What is EfficientNet?**
A CNN family scaled by a compound coefficient across depth/width/resolution; we use EfficientNet-B0,
the smallest variant.

**13. What is ConvNeXt?**
A modern (2022) CNN redesigned with Transformer-inspired training recipes and large-kernel depthwise
convolutions, while remaining a pure convolutional architecture.

**14. What is CLIP?**
A vision-language model contrastively pretrained on ~400M image-caption pairs; we use only its image
encoder (ViT-B/32).

**15. What is DINOv2?**
A self-supervised Vision Transformer trained with no labels, using student-teacher self-distillation
across augmented views.

**16. Why did DINOv2 perform weakly (as CLS-token, frozen)?**
Hypothesis: its CLS token is optimized for semantic/global image understanding, not for fine spatial
size cues, which precise length regression needs.

**17. Why did patch tokens help?**
Because they preserve per-location spatial information the CLS token discards; length is fundamentally
a spatial/geometric property, and patch pooling keeps that signal. Verified over 3 seeds: 1.261±0.054
vs. 1.843±0.023 cm (non-overlapping ranges).

**18. Why did EfficientNet-B0 nearly match the baseline?**
It's a well-designed, ImageNet-pretrained supervised CNN, similar in spirit to MobileNetV2, even though
it was trained with a comparatively modest recipe (fewer epochs, lower LR than the baseline's own
recipe) — suggesting a stronger recipe could close the remaining 0.010 cm gap.

**19. Did any single model beat the baseline?**
No — not yet. EfficientNet-B0 (0.781 cm) is the closest, and is actually better than the baseline on
occluded fish specifically (0.893 vs 0.909 cm).

**20. Did the ensemble beat the baseline?**
Yes: 0.711 cm vs. 0.771 cm (validation-selected weights, reported once on test), beating the baseline
on both non-occluded and occluded subsets.

**21. Is the ensemble a new model/architecture?**
No. It is a combined-prediction system (a weighted average of three already-trained models' outputs),
not a new architecture or a single trained network.

**22. Can we "publish" the ensemble result as our headline finding?**
It can be reported as a valid, honestly-obtained result (selected on validation, evaluated once on
test), but it should be clearly labeled as an ensembling strategy, not a single-model win, and it comes
with the added cost of three forward passes at inference.

**23. What about species classification?**
A separate task using the same encoders and dataset. All encoders perform strongly: ConvNeXt-Tiny
99.57% accuracy / 99.21% macro-F1, MobileNetV2 99.12% / 98.95%, DINOv2 frozen 98.19% / 97.75%, CLIP
frozen 95.13% / 95.45%. ConvNeXt is marginally the best classifier.

**24. What about segmentation?**
Not evaluated in this project; noted as a limitation and possible future work (the original AutoFish
paper also studies segmentation).

**25. Did we try DINOv3?**
Investigated; blocked by license-gated pretrained weights (the official download returns HTTP 403
Forbidden). No DINOv3 results exist locally.

**26. Why wasn't EfficientNet in the old poster?**
It was added in this round of experiments once we identified supervised-CNN encoders as the most
promising direction and wanted the next natural candidate in that family.

**27. Why EfficientNet now?**
Because it reached 0.781 cm — within 0.010 cm of the baseline — at only a basic training recipe,
making it the strongest single-model challenger and a natural focus for further tuning.

**28. Why different learning rates across models?**
Foundation-model encoders (CLIP, DINOv2) can lose useful pretrained structure if fine-tuned too
aggressively, so they use lower learning rates than the baseline, which reproduces the paper's own
recipe (1e-3). This is disclosed explicitly on the poster and in the configs table (§13).

**29. Are these results statistically final?**
No. Most single-model comparisons are single-seed. Only the DINOv2 patch-vs-CLS study has 3-seed
statistics. EfficientNet-B0's near-baseline result is a valid single-run finding, not yet a
statistically confirmed win — multi-seed repeats are needed before stronger claims are justified.

**30. What are the main limitations?**
Single-run results for most models; small-scale foundation-model variants only; DINOv3 blocked;
segmentation not evaluated; the occluded reproduction gap vs. the paper is unexplained; the ensemble
adds inference cost and isn't a new architecture.

**31. What are the next steps?**
Complete the validation-based stronger-recipe search (cosine LR schedule + weight decay + tuned LR +
more epochs) for EfficientNet-B0/ConvNeXt to test whether a single model can cross the baseline;
multi-seed the top single models; try patch pooling for CLIP; test EfficientNet-B2 and larger ViT
variants; evaluate mask segmentation; pursue DINOv3 access.

---

*Generative AI was used for code scaffolding and text drafting in this document. All experiments,
metrics, and configurations were run and verified by the authors against the saved files listed at the
top of this document.*
