# Fish Length Estimation with Vision Foundation Models
## Full research plan for the Deep Learning for Maritime Vision Applications seminar

**Supervisor:** Bohan Zhuang (bohan.zhuang@uni-rostock.de)
**Primary dataset:** AutoFish — Bengtson et al., WACVW 2025 (1,500 images, 454 unique fish, instance segmentation masks + manual length measurements)
**Optional dataset:** A North Sea fish dataset, if it becomes available in time

---

## 1. Reading of the task — what the professor is actually asking

The PDF defines one core research question and one optional extension:

> **Core:** *How well does an established deep learning baseline perform for fish length estimation, and does replacing its encoder with a Vision Foundation Model (VFM) improve performance, in particular when labeled data are limited?*
>
> **Optional:** *To what extent does the resulting approach transfer to a second fish dataset?*

This is a **comparative empirical study**, not a brand-new method paper. The professor will judge you on:

1. **Rigor of the comparison** — same data splits, same target definition, same evaluation metrics, same training budget for both models. Anything else makes the comparison meaningless.
2. **Honesty about the baseline** — you must "reproduce or approximate the baseline approach as closely as feasible." The AutoFish paper's own best baseline (MobileNetV2 regression on cropped fish, 0.62 cm MAE no-occlusion / 1.38 cm with occlusion) is your reference point.
3. **The data-efficiency story** — the *"in particular when labeled data are limited"* clause is doing a lot of work. The professor specifically wants to see whether a VFM-pretrained encoder helps more when you have, say, 10% or 25% of labels. This is the most interesting result you can produce and you should design the experiments around it.
4. **Failure analysis** — error stratified by occlusion level, species, fish size, and where the model is biased (e.g. systematic underestimation of long fish).

What the professor is **not** asking you to do:
- Invent a new architecture from scratch
- Beat state of the art
- Solve the optional second-dataset task (only if the data arrives in time)

---

## 2. Understanding the AutoFish dataset

This is the foundation of everything. From the paper and the HuggingFace card:

| Property | Value |
|---|---|
| Total images | 1,500 (RGB, 2464 × 2056) |
| Unique fish specimens | 454 |
| Instance segmentation masks | 18,160 |
| Groups | 25 (each group has 14–24 fish, none overlap between groups) |
| Per group | 60 images (20 in Set1, 20 in Set2, 20 in All) |
| Length annotation | Manually measured by a marine biologist, rounded to the nearest 5 mm |
| Camera | Jai GO-5100C-USB + KOWA LM12HC lens, 1.5 m above a white conveyor belt |
| Calibration | 20 checkerboard images per group (20 mm squares) |
| Species | Cod, haddock, whiting, hake, horse mackerel (+ a small "other" bin) |
| Sets | **Set1**: half of group's fish, no overlap. **Set2**: other half, no overlap. **All**: every fish in the group, intentionally with high occlusion. |

### Crucial property: the group structure prevents leakage

Each fish appears in exactly one group, and each group is split internally into Set1 / Set2 / All. **You must split on groups, not on images**, otherwise the same physical fish appears in both train and test and the model just memorizes individuals. This is why the dataset authors organized it this way and you should explicitly state this design choice in your report and slides.

### Target definition (the very first thing the professor will ask)

You must commit to one prediction target up front. The natural choice — and what the AutoFish authors do — is:

> **Predict the real-world length in centimeters of one fish, given a tight image crop of that fish.**

Crops are obtained by taking the bounding box of the ground-truth instance segmentation mask, then resizing to a fixed input size (e.g. 224×224 for ImageNet-style encoders, or the native size of the VFM). At training and test time you use ground-truth masks; in a deployed system these would come from an instance segmentation model (Mask2Former, in the AutoFish paper), but the length-estimation task is intentionally decoupled from segmentation so the two error sources don't get conflated.

The length values are in cm, rounded to 5 mm. Important consequences:

- The label resolution is 0.5 cm — any MAE below ~0.25 cm is essentially noise-floor.
- The lengths are real-world cm, not pixel counts. The CNN must *implicitly* learn the pixel-to-cm mapping. Since the camera height (1.5 m) and lens are fixed for the whole dataset, this mapping is constant per group up to calibration distortion. This is what makes the regression learnable from a single crop.

---

## 3. What "the baseline" actually is

From Section 4.2 of the AutoFish paper there are two baselines:

**SKL (Mask Skeletonization):** classical pipeline. Take the instance mask → skeletonize → measure pixel arc length along the skeleton → multiply by a calibrated pixel-to-cm factor. **No learning.** Reported as the simpler of the two baselines.

**REG (CNN regression):** the deep-learning baseline you are expected to reproduce. The architecture is a **custom MobileNetV2-based regressor**:
- Input: a fish crop (from the instance mask bounding box)
- Encoder: MobileNetV2 backbone (often ImageNet-pretrained)
- Head: global average pool → fully connected → single scalar output (length in cm)
- Loss: typically L1 (MAE) or smooth-L1
- Reported numbers: **MAE 0.62 cm on no-occlusion (Set1/Set2), 1.38 cm on occluded (All) — this is the bar you must hit or come close to.**

Your job is to reproduce REG faithfully enough that you can claim "this is the published baseline" and then ablate the encoder.

**Important nuance:** the paper does not publicly release the REG training code in a turnkey way. You will need to follow the paper description and implement it from torchvision / timm primitives. Your write-up should say "we approximate the baseline based on the architecture and hyperparameters reported in Bengtson et al., 2025" — this is honest and accurate.

---

## 4. What a Vision Foundation Model (VFM) is, and which to pick

A Vision Foundation Model is a large encoder pretrained on huge amounts of image data using self-supervised or weakly-supervised objectives, producing features that transfer well to many downstream tasks with little or no fine-tuning. The standard families relevant to this project:

| VFM | Pretraining | Why it might help here |
|---|---|---|
| **DINOv2** (Meta, 2023) | Self-supervised on 142M curated images | Features known to be strong for fine-grained and dense prediction tasks; works well frozen |
| **DINOv3** (Meta, 2025) | Successor to DINOv2, larger and better | The current best general-purpose SSL encoder |
| **SAM / SAM2 image encoder** | Trained for segmentation on 1B+ masks | Strong shape-aware features, well-suited to measuring extent of objects |
| **CLIP** (OpenAI) | Image-text contrastive on 400M pairs | Strong semantic features but less geometric; usually weaker for measurement tasks |
| **MAE** (He et al.) | Masked autoencoding | Solid but generally beaten by DINOv2/3 on dense tasks |

**My recommendation for the seminar:** **DINOv2 (ViT-B/14)** as the primary VFM, with optional second comparison to **SAM2's image encoder** if time allows. DINOv2 is the safest choice because:
- It's well-documented, easy to load (`torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')`)
- It has strong published results on fine-grained tasks
- Its features are spatial-aware (good for measuring fish extent)
- ViT-B/14 is small enough to fine-tune on a single GPU

State this choice in the meeting and justify it on those grounds. The professor will not penalize you for not trying every VFM — they will penalize you for a sloppy comparison.

### How to plug DINOv2 into the baseline pipeline

Replace MobileNetV2 with DINOv2 ViT-B/14:
- Input: 224×224 or 518×518 crop (DINOv2's native patch is 14, so input must be a multiple of 14)
- Take the CLS token output (768-d for ViT-B) → small MLP head (e.g. 768 → 256 → 1)
- Train head only ("linear probe" or "MLP probe") **as the first variant**
- Optionally also try full fine-tuning with a low learning rate on the encoder

The reason for the "head-only" variant is exactly the data-efficiency question: a frozen VFM with a small head is the standard low-data setup. If DINOv2-frozen beats fully-trained MobileNetV2 at 10% data, that's your headline result.

---

## 5. Concrete experimental design

This is what you will run, in order. Treat this as a checklist.

### 5.1 Data preparation

1. Download the dataset from `vapaau/autofish` on HuggingFace (it ships with a loader script: `datasets.load_dataset('vapaau/autofish', revision='script', trust_remote_code=True)`).
2. Parse the per-fish annotations: for every fish instance, you need (image_path, segmentation_mask, bbox, length_cm, species, fish_id, group_id, set_name ∈ {Set1, Set2, All}).
3. **Group-level split.** Hold out **5 groups** as test, **4 groups** as validation, **16 groups** as train. This gives roughly 64/16/20 fish split, with no specimen ever appearing across splits. Document this split deterministically (fix the random seed, list the group IDs in your report).
4. For every fish instance, generate a tight RGB crop by taking the bounding box of the mask, padding by 10% on each side (so the model sees a little context), then resizing to 224×224 with bilinear interpolation. Save these crops to disk as your processed dataset — much faster to train on.
5. **Two evaluation regimes** (matching the paper):
   - **No-occlusion**: only crops from Set1 and Set2 (single fish, isolated)
   - **With-occlusion**: only crops from All (densely packed, fish touching/overlapping)
   These give two MAE numbers per model, like the paper reports.

### 5.2 Models to train

Train all four of these on the **same** train split with the **same** input pipeline:

| Name | Encoder | Encoder weights | Head | Encoder trained? |
|---|---|---|---|---|
| **B-Scratch** (sanity) | MobileNetV2 | random init | GAP → FC → 1 | yes |
| **B-Baseline** (the paper's REG) | MobileNetV2 | ImageNet | GAP → FC → 1 | yes |
| **V-Frozen** | DINOv2 ViT-B/14 | DINOv2 pretrained | MLP(768→256→1) on CLS token | **no** (head only) |
| **V-FT** | DINOv2 ViT-B/14 | DINOv2 pretrained | MLP(768→256→1) on CLS token | yes (small LR) |

B-Scratch shows pretraining matters even for the baseline (cheap sanity check). B-Baseline is the bar. V-Frozen tests "do raw VFM features already encode length?". V-FT tests "does light fine-tuning unlock more?"

### 5.3 Training recipe (same for all four)

- Optimizer: AdamW
- Loss: Smooth-L1 (Huber, β=1.0). Less sensitive to outliers than MSE.
- Batch size: 32 (adjust if GPU memory tight)
- Epochs: 50 with early stopping on validation MAE, patience 10
- LR: 1e-3 for head, 1e-5 for the DINOv2 backbone when fine-tuning; 1e-3 for MobileNetV2 backbone
- LR schedule: cosine decay
- Augmentations: random horizontal flip (fish are left/right symmetric), small ±10° rotation, ±10% scale jitter, color jitter (brightness/contrast/saturation 0.2). **Do NOT use random crop that changes aspect ratio aggressively** — length is encoded in geometry.
- Seed: fix one seed, then re-run with 3 seeds for the headline numbers and report mean ± std. This is non-negotiable for a credible comparison.

### 5.4 Data-efficiency experiment (the headline)

Train each of the four models on subsampled fractions of the training data:
- **100% (full), 50%, 25%, 10%, 5%** of training fish (subsampled at the **fish level**, not crop level, to avoid leakage)
- 3 seeds at each fraction
- Plot MAE vs. training fraction, one line per model

**What you expect to see (and what makes the result publishable as a seminar):** the two MobileNetV2 lines should degrade sharply as data drops; the DINOv2-frozen line should stay much flatter; DINOv2-fine-tuned should be best at 100% and competitive with frozen at 5%. If you see roughly this, the story writes itself.

### 5.5 Failure / bias analysis

For your best two models (B-Baseline and the best V-* variant), produce on the test set:

1. **Error vs. occlusion**: MAE on Set1/Set2 vs. on All (mirroring the paper's table).
2. **Error vs. species**: MAE bar chart per species, ordered by sample count.
3. **Error vs. length**: scatter of predicted vs. true length, plus residuals binned by true length (5-cm bins). This reveals systematic bias — e.g. compressing the dynamic range (predicting near the mean for hard cases).
4. **Worst-case crops**: pull the 20 highest-error predictions for each model and inspect them visually. Note the pattern (heavy occlusion? unusual pose? specific species? cut off at edge of belt?).

### 5.6 Optional: second-dataset transfer

If the North Sea dataset arrives in time:
- Apply the best AutoFish-trained model **zero-shot** → report MAE
- Then fine-tune on a small amount of the new dataset (e.g. 20–50 fish) → report MAE
- The VFM-based model should transfer much better than the MobileNetV2 baseline, which is the whole point of the optional extension

If the dataset does not arrive, simply state this in your final report. Nobody will penalize you for a dataset that didn't materialize.

---

## 6. Metrics — exactly what to compute

Use **all** of these, not just one:

- **MAE** (cm) — primary metric, matches the paper. Easy to interpret: "off by 1.4 cm on average."
- **RMSE** (cm) — penalizes large errors more. Pair with MAE to see if errors are heavy-tailed.
- **MAPE** (%) — mean absolute percentage error. Useful because a 1 cm error on a 10 cm whiting matters more than on a 50 cm cod.
- **Bias** (mean signed error, cm) — tells you if the model systematically over- or under-predicts.
- **R²** — how much variance in length the model captures.

Report all five, both for no-occlusion and with-occlusion regimes.

---

## 7. Software stack and setup

You can run everything on a single GPU (an RTX 3090, 4090, or A6000 is plenty; even a 16 GB card works if you reduce batch size). My recommendation:

```
Python 3.11
PyTorch 2.x with CUDA
torchvision, timm (for MobileNetV2 and many backbones)
datasets, huggingface_hub (for AutoFish)
torch.hub for DINOv2
albumentations (augmentations) or torchvision.transforms.v2
numpy, pandas, scikit-learn, matplotlib
weights & biases or tensorboard for logging — pick one
```

Minimal one-time setup script:

```bash
conda create -n autofish python=3.11 -y
conda activate autofish
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install timm datasets huggingface_hub albumentations scikit-learn pandas matplotlib tqdm wandb
```

For DINOv2:
```python
import torch
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
```
That single line gets you the encoder.

---

## 8. Suggested timeline (rough — adjust to your actual deadlines)

| Week | Milestone |
|---|---|
| 1 | Download AutoFish, parse annotations, build group-level splits, generate cropped dataset on disk. **Deliverable:** a Jupyter notebook that loads N random crops and shows them with their ground-truth lengths. |
| 2 | Implement and train B-Baseline (MobileNetV2). **Deliverable:** MAE within ~0.2 cm of the paper's number on no-occlusion. If you're way off, debug before moving on. |
| 3 | Implement V-Frozen and V-FT (DINOv2). **Deliverable:** all four models trained on full data, single seed, with logged metrics. |
| 4 | Run the data-efficiency sweep (5 fractions × 4 models × 3 seeds = 60 training runs; each is small, so 1–2 days on one GPU). **Deliverable:** the headline plot. |
| 5 | Failure analysis, write report, prepare slides. |
| 6 | Buffer + (optional) second dataset if available. |

---

## 9. Pre-emptive answers to questions the professor will ask

These are worth rehearsing before the meeting:

**Q: Why DINOv2 and not SAM or CLIP?**
A: DINOv2's self-supervised features have been shown to transfer well to dense and fine-grained tasks without fine-tuning, which directly matches the "limited labels" question in the task. SAM's encoder is also a reasonable choice and I plan to compare to it if time allows, but DINOv2 has stronger evidence on small-data regression-style tasks. CLIP features are more semantic than geometric and would be a poor first choice for a measurement task.

**Q: Why not also replace the segmentation step with a VFM?**
A: The task explicitly asks about replacing the encoder of the length-estimation baseline. Segmentation is a separate problem that the AutoFish paper addresses with Mask2Former; conflating both would muddy the comparison. I use ground-truth masks at evaluation time, which is also what the paper's length-estimation experiments do.

**Q: How do you know your baseline reproduction is faithful?**
A: I match the architecture (MobileNetV2 + regression head), the loss family (L1/Smooth-L1), and report the same metric (MAE in cm) on the same two regimes (no-occlusion / with-occlusion). I will report my B-Baseline number side-by-side with the paper's published 0.62 / 1.38 cm. If my number is within ~0.2 cm of theirs the reproduction is credible; if it's significantly worse I'll investigate before proceeding.

**Q: What if DINOv2 doesn't beat MobileNetV2 at full data?**
A: That's a valid scientific outcome and I would report it honestly. The interesting question is the small-data regime — even if DINOv2 ties at 100% data, beating MobileNetV2 at 10% data is the result the task is asking about.

**Q: Are 1,500 images and 454 fish enough for a regression task?**
A: They're tight, which is exactly why this is an interesting problem and why VFMs may help. The data-efficiency experiment makes this an explicit variable rather than a hidden confounder.

**Q: Why crops and not whole images?**
A: Whole-image regression conflates "which fish?" with "how long is it?". Working per-crop isolates the length-estimation problem, matches the AutoFish paper's setup, and gives clean per-instance error analysis.

**Q: Do you need to handle the pixel-to-cm calibration explicitly?**
A: The CNN approach learns it implicitly because the camera setup is fixed across the dataset. If we extended to a new camera (second dataset) we would need either retraining or an explicit calibration input (e.g. an ArUco marker in frame). I'll note this as a limitation.

---

## 10. Deliverables for the meeting

Bring:
- This plan (or a 1-page summary of it)
- A clear statement of the four models you will train and why
- The split strategy (group-level, deterministic seed, listed group IDs)
- The data-efficiency experiment design
- The metrics table you intend to fill in
- A timeline with checkpoints
- Pre-emptive answers to the questions above

You don't need code or results yet — the professor will want to see that you understand the problem, have a coherent plan, and have already thought about pitfalls.

---

## 11. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Can't reproduce baseline MAE | Try multiple LR / augmentation combos; if still off, document the gap honestly and proceed with whatever number you do get |
| DINOv2 features turn out not to help | Still a valid result — write it up; also try SAM2 encoder as a second VFM |
| Group-level split gives high test variance | Average over 3 different group-level splits, not just 3 seeds |
| Compute budget too small | Drop the 3-seeds requirement for the data-efficiency sweep, run 1 seed × 5 fractions × 4 models = 20 runs; keep 3 seeds only for the headline 100% and 10% rows |
| North Sea dataset arrives at the last minute | Limit transfer experiment to zero-shot + tiny-fine-tune; do not try to compete with the main study |
| Length labels are noisy (rounded to 5 mm) | Floor MAE expectations at ~0.25 cm; do not chase below that |

---

## 12. References

- Bengtson, Lehotský, Ismiroglou, Madsen, Moeslund, Pedersen. *AutoFish: Dataset and Benchmark for Fine-grained Analysis of Fish.* WACVW 2025. arXiv:2501.03767.
- Herrmann, Øye, Dyrstad, Alvestad. *Can automatic measuring replace humans when evaluating a shrimp fishery?* Reg. Studies in Marine Science, 2024.
- Oquab et al. *DINOv2: Learning Robust Visual Features without Supervision.* TMLR 2024.
- Kirillov et al. *Segment Anything.* ICCV 2023.
- Ravi et al. *SAM 2: Segment Anything in Images and Videos.* 2024.
