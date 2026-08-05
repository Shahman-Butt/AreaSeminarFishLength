# Viva Prep — Part 1: The AutoFish Paper, Explained From Scratch

**Source used:** the actual paper PDF, downloaded from https://arxiv.org/pdf/2501.03767 and read
in full (all 10 pages, real extracted text — not from memory or general knowledge). Every fact,
number, and quote below is traceable to that document. Where the extracted text was ambiguous
(e.g. exact species-count ordering in a figure), this is flagged explicitly rather than guessed.

**Paper:** Bengtson, Lehotský, Ismiroglou, Madsen, Moeslund, Pedersen — *"AutoFish: Dataset and
Benchmark for Fine-grained Analysis of Fish"*, arXiv:2501.03767v1 [cs.CV], 7 Jan 2025, Aalborg
University, Denmark. ⚠️ Note: our own project docs cite this as "WACVW 2025" — the arXiv PDF
itself (as downloaded) does not display a venue tag on any of its 10 pages. This may be accurate
publication info added after arXiv posting, but it is **not confirmed by the PDF text itself**, so
in a viva, cite it as "arXiv:2501.03767" (guaranteed correct) rather than asserting the venue
unless you have separately confirmed it.

---

## 0. The one-paragraph version (say this first if asked "what is the paper about?")

**Layman:** Fish counting and measuring on fishing boats is currently done by hand. The authors
built a public photo dataset of real fish on a conveyor belt, with every fish's outline and true
length recorded, specifically so that researchers (including us) can train and test AI systems that
automate this. They also built and tested two example AI systems themselves — one to find fish in a
photo (segmentation), and two competing ways to measure a fish once it's found (skeletonization vs.
a small neural network).

**Technical:** The paper contributes (1) a novel dataset (1,500 images, 454 fish, 18,160 instance
segmentation masks, group-based train/test splitting to avoid identity leakage), and (2) baseline
results for two downstream tasks: instance segmentation (Mask2Former, ResNet-50 vs. Swin-B
backbones, best mAP 89.15%) and length estimation (two independent methods — SKL:
skeletonization + homography, and REG: MobileNetV2 regression — each evaluated on both
ground-truth and model-predicted masks).

---

## 1. Abstract — line by line

> *"Automated fish documentation processes are in the near future expected to play an essential
> role in sustainable fisheries management and for addressing challenges of overfishing."*

**Layman:** The motivation is overfishing — better automatic fish-counting tech should help fix it.
**Technical:** This frames the work as applied/deployment-oriented research (fisheries management),
not pure methodology research — relevant when a professor asks "why does this matter beyond being
an interesting ML problem?"

> *"...a novel and publicly available dataset named AutoFish designed for fine-grained fish
> analysis. The dataset comprises 1,500 images of 454 specimens of visually similar fish placed in
> various constellations on a white conveyor belt..."*

**Layman:** 1,500 photos, 454 individual real fish, deliberately similar-looking species (harder
task) placed in different arrangements.
**Technical:** "Fine-grained" here means distinguishing visually similar species/individuals — the
paper deliberately chose taxonomically close species (the cod family) to make the benchmark
harder, not easier.

> *"...annotated with instance segmentation masks, IDs, and length measurements... manual point
> annotations, initial segmentation masks proposed by the Segment Anything Model (SAM), and
> subsequent manual correction of the masks."*

**Layman:** Every fish's outline was drawn by AI first (SAM, Meta's segmentation foundation model)
and then corrected by hand.
**Technical:** This is a **human-in-the-loop annotation pipeline** — SAM proposes masks from point
prompts, humans verify/correct. This is *not* the same as the Mask2Former model discussed later
(SAM is only an annotation-time tool; Mask2Former is the trained deployable model evaluated in the
paper's experiments).

> *"We establish baseline instance segmentation results using two variations of the Mask2Former
> architecture, with the best performing model reaching an mAP of 89.15%."*

**Technical:** mAP = mean Average Precision, the standard segmentation/detection quality metric
(explained fully in §5 below). 89.15% is the Swin-B backbone's combined result.

> *"...two baseline length estimation methods, the best performing being a custom MobileNetV2-based
> regression model reaching an MAE of 0.62cm in images with no occlusion and 1.38cm in images with
> occlusion."*

**⚠️ This is the sentence our project has quoted throughout as "the paper's baseline." It is true,
but incomplete** — see Part 6 (GT vs PD) below for the full picture. This abstract number is
specifically **REGpd** (regression on *predicted*, not ground-truth, masks).

---

## 2. Introduction & Related Work — the short version

**Motivation (Layman):** Overfishing damages ocean ecosystems and coastal economies. Today,
fisheries are monitored by catch limits and occasional in-person boat inspections — slow,
incomplete, easy to under-report. Cameras + AI could give continuous, honest, automatic records.

**Related work (Technical):** The paper positions itself against:
- Older **handcrafted-feature** methods (pre-deep-learning computer vision) — brittle, need
  re-engineering for each new species.
- **Detection-based** approaches (YOLO + bounding boxes) — can't capture a fish's actual bent,
  overlapping shape.
- **Segmentation-based** approaches (Mask R-CNN and similar) — the paper's own approach follows this
  line, because masks handle non-rigid, occluding fish much better than boxes.
- Existing fish datasets (FishNet, FD, FDWE, DeepFish) — Table 1 in the paper shows AutoFish is the
  only one combining: instance segmentation + fish IDs (repeat individuals) + length + both-sides
  imaging, from a conveyor-belt environment.

**Viva-relevant nuance:** the paper explicitly credits **Ovalle et al.** as the inspiration for the
REG method (a small MobileNetV1 regressor for conveyor-belt fish length) — AutoFish's contribution
there is applying/scaling this idea with MobileNetV2 and a proper dataset+split, not inventing the
underlying CNN-regression concept from scratch.

---

## 3. The Dataset — in full detail

### 3.1 Fish composition
- Species: cod, haddock, whiting (all *Gadidae*/cod family — visually similar, hence "fine-grained"),
  plus hake and horse mackerel, with rarer species grouped as "other."
- ⚠️ The exact per-species counts and average lengths are given in Figure 2 of the paper, but the
  raw text-extraction interleaves numbers and percentages in a way that cannot be reliably
  re-paired without seeing the actual figure image (the numbers "102, 119, 103, 52, 49, 29" and
  percentages "26.2%, 22.5%, 6.4%, 10.8%, 11.5%, 22.7%" and average lengths "35.6±2.1cm,
  41.1±3.9cm, 37.4±10.8cm, 28.7±3.1cm, 31.5±2.3cm, 35.8±2.7cm" are all present, six values each,
  presumably one per species/category — but confidently assigning which number belongs to which
  named species would be guessing from text alone). **If asked exact per-species counts in the
  viva, say: "the paper reports this in Figure 2; I'd want to check the figure directly for exact
  pairing rather than guess."** That is a completely acceptable, honest answer.
- Fish were caught by a real Danish commercial trawler in the North Sea/Skagerrak/Kattegat.

### 3.2 Camera setup
- A **static** 100×100 cm white conveyor belt (i.e. this is a controlled lab replica of a moving
  belt, not a live vessel), camera mounted 1.5 m above (Jai GO-5100C-USB + KOWA lens), f/11,
  1 m focus distance, images at 2464×2056 px.
- **Camera calibration**: 20 checkerboard images per group (each square 20×20 mm), used later for
  lens-distortion correction and the pixel→cm homography (this calibration step is what the SKL
  method needs and REG does not).

### 3.3 Image collection — this is where your project's split comes from
- Length measured by hand by a marine biologist first, rounded to nearest 5 mm.
- Fish partitioned into **25 groups** of 14–24 fish, **pseudo-randomly** mixed (not sorted by
  species) to mimic a real catch.
- **Each group → 60 images, split into 3 subsets of 20 images each:**
  - **Set1** (images 1–10 initial side, 11–20 flipped side): half the group's fish, laid separately
  - **Set2** (images 21–30, 31–40): the other half, laid separately
  - **All** (images 41–50, 51–60): the *whole* group, deliberately placed touching/overlapping
- Every fish appears **exactly 40 times** total (20 times per side).

**✅ This exactly matches your project's `set_name_from_file()` function** (`scripts/build_autofish_index.py`):
image numbers 1–20 → Set1, 21–40 → Set2, 41–60 → All. This is not a coincidence or an assumption
your project made — it is precisely what the paper itself describes in Figure 3. You can say with
full confidence in the viva: *"our Set1/Set2/All mapping is not an assumption, it is directly
specified by the paper's own dataset construction, Figure 3."*

### 3.4 Annotation procedure
- Point-annotate every fish's ID in every image during recording (so IDs can be tracked across
  reshuffles) → SAM proposes initial masks from those points → manual correction → occlusion cases
  (one fish split into multiple mask pieces) merged under the same ID → everything compiled in
  **MS COCO format**.

**✅ This also matches your project**: your `index.csv` stores `segmentation` as COCO-style polygon
JSON, consistent with this described format.

---

## 4. Methods — the two-stage pipeline

The paper frames its own baseline pipeline exactly as:

```
Raw image → [Instance Segmentation: Mask2Former] → Masks → [Length Estimation: SKL or REG] → Length (cm)
```

### 4.1 Instance segmentation — Mask2Former
- **Two backbone variants** tested: ResNet-50 (CNN) and Swin-B (Swin Transformer, "larger
  alternative").
- Both pretrained on MS COCO, then fine-tuned **1000 steps**, batch size 8, **no validation set or
  early stopping used during this fine-tuning**, optimizer **AdamW**, learning rate multistep
  schedule from 0.1 down to 0.0001.
- Augmentations (Table 2, segmentation row): horizontal & vertical flip each with 50% probability,
  plus contrast/brightness/saturation jitter in range [0.75, 1.25] each.

**Why segmentation exists at all (layman):** before you can measure a fish's length from a photo,
you first need to know *exactly which pixels are the fish* — segmentation is that "find and outline
every fish" step. **Why it's outside our project's scope:** our project always starts from the
dataset's own ground-truth masks; we never train or run a segmentation model ourselves. This is a
legitimate, explicitly scoped-down piece of the paper's full system, not a gap we're hiding.

### 4.2 Length estimation — two competing methods

#### 4.2.1 SKL (skeletonization)
**Layman:** Take the fish's mask outline, find its "skeleton" (a thin center-line, like the spine),
smooth that into a curve, then use a pre-measured calibration grid to convert that curve's pixel
length into real centimetres.

**Technical, step by step:**
1. Apply the **Zhang–Suen thinning algorithm** (a classic 1984 image-processing method) to the mask
   to get a 1-pixel-wide skeleton.
2. Fit a **4th-degree polynomial** to that skeleton — a smooth curve approximating the fish's
   central line.
3. To handle **forked tails** or **occlusion** (mask split into pieces), evaluate the polynomial
   against the mask's **convex hull** instead of the raw skeleton directly.
4. Compute a **homography** (pixel→real-world mapping) per group from the 20 calibration
   checkerboard images; this also corrects lens distortion.
5. Map the polynomial's points onto the real conveyor-belt plane and sum the distances between them
   → final length in cm.

**Why this matters for comparison:** SKL needs **no training data at all** — just calibration
images. It is a classic computer-vision pipeline, not deep learning, for the length-estimation step
itself (though it depends on a mask, which *could* come from a deep model).

#### 4.2.2 REG (CNN regression) — this is what your project reproduces
**Architecture (paper's Figure 5, and this is an exact match to your baseline config):**
- ImageNet-pretrained **MobileNetV2** → 1280-dim feature vector
- Input image: the RGB mask, **cropped to a black square bounding box** around the fish
- The **normalized bounding box** (4 values, e.g. `[0.23, 0.89, 0.11, 0.22]`) is concatenated to
  the 1280 image features
- Two fully-connected layers: **FC1 → 1000, FC2 → 500**, then a final output = length

**✅ This is a byte-for-byte match to your baseline config** (`configs/baseline_official.json`
uses `head: [1000, 500, 1]`, `bbox_input: true`) — your reproduction did not guess this
architecture, it copied it directly from the paper's own Figure 5.

**Training recipe (paper, §4.2.2):** batch size 32, **200 epochs**, **L1 loss**, **Adam**, fixed
learning rate **0.001**, no learning-rate schedule mentioned. **No geometric augmentation** ("as
they may affect the pixel to centimeter mapping") — only photometric jitter (contrast [0.50,1.50],
brightness [0.80,1.20], saturation [0.60,1.40], per Table 2's length-estimation row, and
**no flips** for this task, unlike segmentation).

**✅ Also an exact match to your `baseline_official.json`**: batch 32, epochs 200, L1 loss, Adam,
LR 0.001. This is why your reproduction landing at 0.633 cm vs. the paper's related number is a
genuinely faithful reproduction, not a loose approximation — nearly every training hyperparameter
is directly copied from the paper text, not inferred or guessed.

**One real difference worth knowing:** the paper's photometric augmentation is described with
specific numeric *ranges* per parameter (their own custom augmentation scheme); your project uses
torchvision's `ColorJitter(brightness=0.2, contrast=0.5, saturation=0.4, hue=0.3)`, which is
*similar in spirit* (randomized color jitter) but not necessarily numerically identical in how the
ranges are parameterized (torchvision's ColorJitter samples multiplicatively around 1.0 using a
single symmetric range per parameter, whereas the paper reports asymmetric min/max ranges directly).
This is a legitimate, minor implementation difference to acknowledge if pressed, not something to
claim as an exact match.

---

## 5. Results — Instance Segmentation (§5.1)

**Metric: mAP = AP@[IoU=0.5:0.95]** — Average Precision computed by thresholding the overlap
(Intersection-over-Union) between predicted and true masks at every step from 0.5 to 0.95 (in 0.05
increments), then averaging. This is the standard COCO-style segmentation/detection metric.

**Layman translation:** for every possible "how overlapping counts as correct" strictness level
(from loose to very strict), check what fraction of predictions are right, then average across all
those strictness levels. A single number that rewards both *finding* the fish and outlining it
*precisely*.

**Headline result (Table 3):** Swin-B backbone, Combined set: **89.15% mAP** (best). ResNet-50
Combined: 88.31%. Swin-B beats ResNet-50 in every species and every occlusion regime, but the gap
is small (<1–2 points generally). Performance drops from "Separated" (~93%) to "Touching" (~85%) —
**occlusion hurts segmentation quality too**, not just length estimation.

**Training-data scaling (Fig. 6):** performance saturates at **9 training groups** (~180 fish,
~540 images) — more data beyond that gives little further improvement. Useful if asked "how much
data did they actually need."

---

## 6. Results — Length Estimation, and the full GT vs. PD explanation (§5.2)

### 6.1 The evaluation setup
- Length-estimation experiments use their own **train/val/test split within the training groups**:
  15 train groups, **5 validation groups = [1, 6, 11, 17, 25]**, evaluated on the same
  **5 test groups = [10, 14, 20, 21, 22]** reserved earlier for everything.

**✅ These are EXACTLY your project's official split group lists**, digit for digit. This is the
single most important confirmation in this whole document: **your project's split was not invented,
assumed, or taken from the "official training release" as an independent source — it is the split
described directly in the paper's own text**, and the official code release simply implements what
the paper specifies.

- Metrics: **MAE (cm)** and **MAPE (%)**.
- Each method evaluated on **THREE occlusion regimes**: Separated, Touching, Combined (Separated+Touching).
- Each method evaluated using **TWO kinds of input mask**:
  - **`gt`** = the human-annotated ground-truth mask
  - **`pd`** = the mask *predicted* by the Swin-B Mask2Former segmentation model (only predictions
    above 90% confidence are used)

### 6.2 GT vs PD — explained in full, with the exact numbers

**Layman analogy:** Imagine two exam conditions for the "measure this fish" test.
- **GT condition** = you're handed a perfect, hand-traced outline of the fish (no ambiguity about
  which pixels are fish).
- **PD condition** = you're handed an outline that was *itself guessed* by another AI model, which
  is usually good but sometimes slightly wrong or missing (especially when fish overlap).

Testing on `pd` measures "how well does the *whole automatic pipeline* work, end to end, including
the segmentation model's own mistakes?" Testing on `gt` measures "how well does *just the length
estimator* work, assuming perfect input?" These answer different questions, and the paper reports
both on purpose — it's not an inconsistency, it's a deliberate ablation.

**The full Table 4, exactly as printed in the paper:**

| Method | Separated | Touching | Combined |
|---|---:|---:|---:|
| **SKLgt** | 0.59 cm (1.79%) | 1.43 cm (4.51%) | 1.01 cm (3.15%) |
| **REGgt** | 0.67 cm (2.10%) | 0.96 cm (3.08%) | 0.82 cm (2.59%) |
| **SKLpd** | 0.62 cm (1.87%) | 2.43 cm (7.42%) | 1.51 cm (4.59%) |
| **REGpd** | **0.62 cm** (1.92%) | **1.38 cm** (4.32%) | 0.99 cm (3.10%) |

**Why the abstract's "0.62 cm / 1.38 cm" comes from REGpd, specifically:** the abstract reports the
headline, real-world, fully-automatic-pipeline numbers — i.e. what you'd actually get if you ran the
whole system on a boat with no human drawing masks by hand. That's `REGpd`: Separated = 0.62 cm,
Touching = 1.38 cm. It is the *deployment-realistic* number, which is why the authors chose it for
the abstract over the (slightly better-looking, in most cells) `gt` numbers.

**Why this is *not* contradictory (a likely viva question):** it looks odd at first that `pd`
(predicted, imperfect masks) sometimes performs *similarly to or even better than* `gt` on the
Separated set specifically (SKLpd 0.62 ≈ SKLgt 0.59; REGpd 0.62 vs REGgt 0.67, `pd` even slightly
*better*). This is explainable: on Separated (non-occluded, easy) images, the segmentation model is
extremely accurate (mAP ~93%+, IoU ~0.94 for confident predictions per §5 discussion) — so `pd`
masks are nearly as good as `gt` there, and small model-specific quirks in exactly how the mask
boundary is drawn can occasionally align slightly better with how the regression head was
calibrated. **But on Touching (occluded) images, the gap is dramatic**: REGpd 1.38 cm vs. REGgt
0.96 cm, and even more extreme for SKL (2.43 vs 1.43) — because occlusion is exactly where the
segmentation model itself starts making real mistakes (missing fish, merging fish, wrong
boundaries), and those mistakes get inherited by the length estimator. **This is expected, not a
bug or an inconsistency**: predicted-mask evaluation should degrade more than ground-truth
evaluation specifically where the upstream model (segmentation) is weakest, which is occlusion.

### 6.3 What this means for YOUR project's numbers — the correction

Your project's results:

| Your subset | Your MAE | Paper equivalent regime | Correct comparison (GT, since you use GT masks) | What you've been comparing to (PD) |
|---|---:|---|---:|---:|
| Non-occluded (Set1+Set2) | **0.633 cm** | Separated | REGgt = 0.67 cm | REGpd = 0.62 cm |
| Occluded (All) | **0.909 cm** | Touching | REGgt = 0.96 cm | REGpd = 1.38 cm |
| Full test | **0.771 cm** | Combined | REGgt = 0.82 cm | REGpd = 0.99 cm |

**Say this in the viva:** *"Because our pipeline always uses ground-truth masks (we never run a
segmentation model), the fair comparison target is REGgt, not the abstract's REGpd headline number.
Against REGgt, our reproduction is close and consistent across all three regimes — 0.633 vs 0.67,
0.909 vs 0.96, 0.771 vs 0.82 — with no unexplained anomaly. The abstract's well-known '0.62 / 1.38'
numbers are the fully-automatic, predicted-mask pipeline, which is a strictly harder and different
setting than ours."*

This reframes your entire "why is our occluded number mysteriously better" open question — it was
never mysteriously better; it was being compared to the wrong baseline number all along.

---

## 7. Discussion & Error Analysis (§6)

- SKL and REG perform similarly on **Separated** images (~0.62 cm) → skeletonization is a *viable,
  training-free* option when occlusion risk is low (e.g. small/low-power vessels).
- REG clearly wins on **Touching/Combined** → skeletonization structurally cannot handle occlusion:
  when part of a fish's mask is missing (e.g. head hidden behind another fish), the skeleton is
  incomplete, so SKL **systematically underestimates** length. This is directly visible in Fig. 8's
  error-distribution plots: SKL's mean bias is consistently **negative** (e.g. −1.78 cm on Combined
  PD), while REG's bias stays **near zero** (e.g. +0.07 cm on Combined PD) — REG can infer missing
  length from other visible cues, learned during training; SKL cannot.
- **Practical tradeoff** (a good viva answer for "which method would you actually deploy?"): SKL
  adapts to a new boat/setup with just a few calibration photos, no training; REG needs a full new
  labeled training dataset and training run to adapt to a new environment — more accurate under
  occlusion, but far more resource-intensive to redeploy.

## 8. Future work proposed BY THE PAPER (not something we did)

Section 6.1 proposes using the dataset's unique fish **IDs** to average multiple measurements of the
*same* fish across its 40 appearances (via the median), showing MAE could drop from ~0.99 cm (1
sample) to ~0.4 cm (40 samples), with diminishing returns after ~5 samples (~0.5 cm). **This is the
paper's own idea for future work (via re-identification), not something implemented in the official
code release or in your project** — worth knowing so you don't accidentally claim it as something
"we tried."

## 9. Ethics & Funding
Fish were already dead at landing (normal commercial catch), collection cleared with the Danish
Ministry of Food, Agriculture and Fisheries, compliant with EU/Danish animal-experimentation law.
Funded by the EU's European Maritime and Fisheries Fund + the Danish Agricultural and Fisheries
Agency.

---

## 10. Quick-fire viva Q&A — Part 1 (paper-specific)

**Q1: What two things does the paper actually contribute?**
A dataset (AutoFish) and baseline results for two tasks on it: instance segmentation (Mask2Former)
and length estimation (SKL and REG).

**Q2: How many images, fish, and masks?**
1,500 images, 454 unique fish, 18,160 instance segmentation masks (paper's own stated count).

**Q3: Why 25 groups instead of one giant pool of fish?**
To support a train/test split with **zero fish-identity leakage** — each fish belongs to exactly
one group, and groups (not individual photos) are assigned to train/val/test, so the same physical
fish never appears on both sides.

**Q4: What are Set1, Set2, and All?**
Per group: Set1 and Set2 each show half the group's fish, laid out separately (non-occluded); All
shows the whole group deliberately overlapping (occluded). 20 images each, 60 per group total.

**Q5: Why does the paper test with both ground-truth and predicted masks?**
To separate two questions: "how good is the length estimator alone, given perfect input?" (`gt`) vs.
"how good is the whole real-world automatic pipeline, including the segmentation model's own
errors?" (`pd`). Both are useful, different, valid numbers.

**Q6: Where does 0.62 cm / 1.38 cm come from, exactly?**
REGpd — the CNN regression method, evaluated on masks *predicted* by the Swin-B segmentation model,
on Separated (0.62) and Touching (1.38) images respectively.

**Q7: Is it a contradiction that predicted masks sometimes score as well as ground truth?**
No — on easy (Separated) images the segmentation model is nearly perfect (~93%+ mAP), so predicted
masks are almost as good as ground truth there. The gap only opens up on occluded (Touching) images,
where the segmentation model itself struggles — exactly where you'd expect predicted-mask
performance to degrade relative to ground truth.

**Q8: Which paper baseline should OUR project be compared against, and why?**
REGgt, not REGpd — because our project always uses ground-truth masks (we never train or run a
segmentation model), so the fair, like-for-like comparison is the paper's ground-truth-mask numbers:
0.67 / 0.96 / 0.82 cm for Separated/Touching/Combined.

**Q9: Why does the paper use MobileNetV2 specifically?**
It's inspired by prior work (Ovalle et al.) showing a small MobileNet-family regressor was
sufficient for this task; MobileNetV2 is lightweight, ImageNet-pretrained, and fast — appropriate
for a baseline meant to be simple and reproducible, not maximally powerful.

**Q10: Why no geometric augmentation (flips/rotation) for the length-estimation training?**
Because the model's job includes mapping pixel positions/sizes to real-world centimetres via the
bounding box input; geometric transforms would distort that pixel-to-cm relationship, unlike
photometric jitter (brightness/contrast/saturation), which doesn't affect geometry.

**Q11: What is skeletonization, in one sentence?**
Thinning a fish's mask down to a 1-pixel-wide center-line (via the Zhang–Suen algorithm), fitting a
smooth curve to it, and converting that curve's length from pixels to centimetres using a
camera-calibrated homography.

**Q12: Why does skeletonization underestimate length under occlusion?**
Because if part of the fish's mask (e.g. the head) is missing due to occlusion, the skeleton is
incomplete, so the fitted curve is shorter than the real fish — a systematic negative bias, visible
in the paper's own error-distribution plots (Fig. 8).

**Q13: What does mAP measure, and how is it computed here?**
Mean Average Precision, computed as AP averaged over IoU overlap thresholds from 0.5 to 0.95 in
steps of 0.05 — a strict, standard COCO-style metric rewarding both correct detection and precise
mask boundaries.

**Q14: Which backbone won for segmentation, and by how much?**
Swin-B (transformer) beat ResNet-50 (CNN) in every category, but only by roughly 1–2 mAP points —
Combined: 89.15% vs. 88.31%.

**Q15: What did the paper find about training-data size for segmentation?**
Performance saturates around 9 training groups (~180 fish, ~540 images) — more data past that point
gave little further benefit.

**Q16: Is the "future work" idea about fish-ID re-identification something your project implemented?**
No — that's the paper's own proposed future direction (median length over multiple samples of the
same fish ID), not implemented in the official code release or in our project.

**Q17: Did the paper report a venue like WACVW 2025 directly in the PDF?**
Not visible in the extracted text of the arXiv PDF used for this review — the paper is confirmed as
arXiv:2501.03767v1, dated 7 Jan 2025. Cite the arXiv ID as the guaranteed-correct reference.

---

*This document is Part 1 of a staged viva-preparation series. Part 2 (our project end-to-end),
Part 3 (encoder deep-dives), and the rest of the plan continue in subsequent documents/turns.*
