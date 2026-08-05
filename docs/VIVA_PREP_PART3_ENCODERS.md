# Viva Prep — Part 3: Every Encoder, Deeply

Builds on Parts 1–2. All architecture facts are standard, well-established ML knowledge (cited to
their original papers where relevant); all performance numbers are from this project's own
`runs/*/test_metrics.json`.

---

## 0. The shared skeleton every encoder plugs into

Every model in this project = **Encoder → concat 4 bbox numbers → small MLP head → 1 number**.
Only the encoder box changes. This matters for the viva: any question about "how is the comparison
fair" traces back to this shared skeleton (see `src/autofish_vfm/models.py`, already fully commented
in the repo).

---

## 1. MobileNetV2 — the baseline

**History (technical):** Sandler et al., CVPR 2018, "MobileNetV2: Inverted Residuals and Linear
Bottlenecks." Designed for on-device (phone) inference — small, fast, still accurate.

**Architecture, internal working:**
- Built from stacked **inverted residual blocks**. A normal residual block (ResNet-style) goes
  wide→narrow→wide; MobileNetV2 inverts this: **narrow→wide→narrow**.
- Each block: (1) a 1×1 **"expand"** convolution increases channel count, (2) a 3×3 **depthwise**
  convolution — one filter *per channel* (far cheaper than a normal convolution, which mixes all
  channels together), (3) a 1×1 **"project"** convolution shrinks channels back down, with **no
  activation** on this last step (the "linear bottleneck" in the paper's title — keeping this layer
  linear was shown empirically to preserve more useful information than adding a ReLU here).
- Stacking many such blocks, with occasional stride-2 blocks to shrink spatial resolution, ends in a
  single pooled feature vector: **1280 numbers** for MobileNetV2.
- Params: ~3.4M — very small compared to the other encoders here.

**Layman:** Think of each block as: blow the picture's description up bigger (expand), scan it
cheaply piece by piece (depthwise), then compress the useful bits back down (project). Repeat many
times, each time the picture gets physically smaller but the description gets richer.

**Why chosen (by the paper, inherited by us):** it's small, fast, ImageNet-pretrained, and prior work
(Ovalle et al., cited in the paper) showed a MobileNet-family regressor was sufficient for
conveyor-belt fish length. We use it as our reproduction target for exactly this reason — matching
the paper's own choice.

**In our project — training:** full fine-tune (nothing frozen), head `[1000, 500, 1]`, Adam LR 1e-3,
batch 32, 200 epochs — copied directly from the paper's Figure 5 and §4.2.2.

**Advantages:** small, fast, easy to train from scratch even on ~11K crops; strong supervised
ImageNet prior for object shape/size cues.
**Disadvantages:** limited representational capacity vs. larger models; may plateau on harder tasks;
purely local receptive field growth (like all CNNs) means very long-range spatial relationships need
many layers to connect.

**Result:** **0.771 cm** full-test MAE — still the best single model in the whole project.

---

## 2. EfficientNet-B0 — the closest challenger

**History:** Tan & Le, ICML 2019, "EfficientNet: Rethinking Model Scaling for CNNs." Introduced
**compound scaling**: instead of arbitrarily making a network deeper, or wider, or fed bigger
images, scale all three together by one formula, found via a small architecture search (the "B0"
baseline network) then scaled up (B1, B2, ... B7).

**Compound scaling, explained:**
- **Depth** = how many layers (more layers = can learn more complex, hierarchical features, but
  harder to train and diminishing returns).
- **Width** = how many channels per layer (more channels = more feature "vocabulary" at each stage,
  but quadratic cost growth).
- **Resolution** = the input image size (bigger images = more fine detail visible, but more compute).
- **Compound scaling insight:** scaling only one of these (as older CNN designs often did somewhat
  ad hoc) gives diminishing returns; scaling all three together, in a fixed ratio found by search, is
  more parameter-efficient. **B0 is the smallest, unscaled base network** — the variant we use.

**MBConv block (the building block):** essentially the same inverted-residual idea as MobileNetV2
(expand → depthwise conv → project), **plus a Squeeze-and-Excitation (SE) block**.

**Squeeze-and-Excitation, explained:**
- **Squeeze:** average-pool each channel down to a single number — "how active is this channel,
  overall, across the whole image?"
- **Excitation:** a tiny 2-layer network turns those per-channel numbers into per-channel *weights*
  (0 to 1) — "how important is this channel right now?"
- Multiply the original feature map by those weights — channels the network decides are more useful
  for this particular image get amplified; less useful ones get suppressed.
- **Layman:** it's a lightweight internal "attention" mechanism, but over *channels* (feature types),
  not spatial locations — the network can dynamically emphasize different visual cues per image.

**Params:** ~5.3M (bigger than MobileNetV2's 3.4M, smaller than ConvNeXt's 28M).

**In our project:** basic recipe — Adam LR 1e-4, batch 16, 100 epochs, full fine-tune, head
`[512, 128, 1]` — same recipe family as the other encoder-swap experiments.

**Result and the honest reliability story:**
- Basic recipe: **0.781 cm** — only 0.010 cm behind the baseline. **Single run, single seed (42) —
  not statistically confirmed.**
- Stronger recipe attempt (cosine LR schedule, LR 1e-3, 200 epochs, weight decay): **0.934 cm —
  worse**, not better.

**Why it came so close (💡 hypothesis, not proven):** EfficientNet-B0 and MobileNetV2 are
architectural cousins — both compact, fully-supervised, ImageNet-pretrained CNNs, both fully
fine-tuned end-to-end. It's plausible this whole family of efficient supervised CNNs suits a masked,
single-object regression task well.

**Why it did NOT beat MobileNetV2:** unknown for certain — the gap (0.010 cm) is well within
plausible single-seed noise; it may or may not be a genuine architectural difference.

**Why the stronger recipe made it worse (💡 hypothesis):** with only ~11,000 training crops, a 10×
higher learning rate plus double the epochs likely pushed the model past its best point into
overfitting or optimization instability, rather than finding a better solution.

---

## 3. ConvNeXt-Tiny

**History:** Liu et al., CVPR 2022, "A ConvNet for the 2020s." A deliberate exercise: take a standard
ResNet and gradually "modernize" it with design choices popularized by Vision Transformers (larger
kernels, fewer activation functions, LayerNorm instead of BatchNorm, GELU instead of ReLU) — while
staying a **pure CNN** throughout, to test how much of a ViT's advantage is really architectural vs.
just training-recipe/scale.

**Architecture:** large-kernel (7×7) depthwise convolutions (much bigger receptive field per layer
than MobileNetV2's 3×3), LayerNorm, GELU activations, an inverted-bottleneck-like block structure.
Params: ~28M — the largest CNN tested here.

**In our project:** full fine-tune, head `[512, 128, 1]`, Adam LR 1e-4, batch 16, 100 epochs.

**Result:** **0.914 cm** — third place among single models, clearly behind both MobileNetV2 and
EfficientNet-B0 despite being the largest/most modern CNN tested.

**Why it didn't do better despite being "more modern" (💡 hypothesis):** more parameters and a
larger receptive field don't automatically help on a small (~11K crops), narrow-domain task; the
smaller MobileNetV2/EfficientNet-B0 may simply be better matched to this dataset's size and the
recipe used (also possibly under-tuned — see Part 2's LR-recipe caveat).

---

## 4. CLIP ViT-B/32 — vision-language

**History:** Radford et al. (OpenAI), 2021, "Learning Transferable Visual Models From Natural
Language Supervision." Trained on ~400 million (image, caption) pairs scraped from the internet,
using **contrastive learning**.

**Contrastive learning, explained:**
- For a batch of (image, caption) pairs, encode every image and every caption into vectors.
- Train so that a **matching** image-caption pair's vectors are close together (high similarity),
  and every **non-matching** pair (this image with every *other* caption in the batch) is pushed
  apart.
- **Layman:** the model is playing a giant matching game — "which caption out of these 400 belongs
  to this photo?" — and gets better at both seeing and reading by being forced to solve it.
- Result: an **embedding space** where semantically similar images and texts land near each other,
  even though the model was never given explicit class labels.

**Image encoder used here:** ViT-B/32 — a Vision Transformer (patch size 32×32, ~88M parameters for
the image tower). **We only ever call `encode_image()`** — the text tower exists in the checkpoint
but this project's regression task never uses it.

**In our project:** tested **frozen** (only the head trains) and with the **last visual
transformer block fine-tuned** (warm-started from the frozen checkpoint).

**Results:** frozen **1.002 cm**, last-block fine-tuned **0.958 cm** — partial fine-tuning helped.

**Why CLIP is excellent for semantics but only middling here (💡 hypothesis):** CLIP's training
signal (matching images to captions) rewards the model for capturing *what things are and how they
relate conceptually* — very transferable to classification-like tasks (confirmed by our species
task: CLIP got 95.13% accuracy). But nothing in that training objective explicitly rewards precise
geometric/metric information like "how many centimetres long is this specific object" — so it
transfers only partially to fine-grained length regression, better than DINOv2's self-supervised
features but still behind task-adapted supervised CNNs.

---

## 5. DINOv2 ViT-S/14 — self-supervised, and the project's central technical finding

**History:** Oquab et al. (Meta), 2023, "DINOv2: Learning Robust Visual Features without
Supervision." Self-supervised — trained with **no labels and no captions at all**, purely by making
the model agree with itself across different augmented views of the same image (a
student-teacher self-distillation setup, building on the original DINO and iBOT methods).

**Vision Transformer mechanics, explained from scratch:**
- **Patch embedding:** the image is chopped into a grid of small patches (14×14 pixels each, for
  this "/14" variant) — e.g. a 224×224 image becomes a 16×16 grid = 256 patches. Each patch is
  flattened and linearly projected into a "token" vector.
- **[CLS] token:** one extra, *learned* token is prepended to that sequence of patch tokens — it has
  no fixed meaning at the start, but through training it learns to become a **global summary** of
  the whole image.
- **Self-attention:** in every Transformer block, **every token can directly look at and weight
  information from every other token** — including patches on the opposite side of the image, in a
  single layer. This is fundamentally different from a CNN, where information only reaches distant
  pixels after passing through many stacked local layers.
- **Params:** ~21M for ViT-S/14 ("S" = small — the smallest DINOv2 variant, chosen for this project).

**CLS token vs. patch tokens — the key distinction, and our project's finding:**
- `x_norm_clstoken` = the single learned summary vector — good for "what is this image, overall?"
  but, being one compressed vector, it discards a lot of fine spatial/positional detail.
- `x_norm_patchtokens` = the full grid of 256 individual patch vectors — each one still "knows"
  roughly where in the image it came from.
- **Our project's code (`models.py`) offers both**, controlled by `use_patch_tokens`: default uses
  CLS; setting it true instead **mean-pools all patch tokens into one vector**, keeping more spatial
  information than the CLS token alone.

**Why DINOv2 struggled at regression, and why patch pooling fixed it (💡 hypothesis, but backed by
a controlled experiment):** length is fundamentally a spatial/geometric quantity. The CLS token,
optimized during pretraining for global semantic summarization, likely discards exactly the
fine-grained size/shape cues length regression needs. Patch tokens retain that spatial detail.

**The controlled proof (not just a hypothesis about the code — an actual confirmed experiment):**
identical hyperparameters, only the token type differs, **3 seeds each**:

| DINOv2 frozen | Mean ± SD (full-test MAE) |
|---|---|
| CLS token | 1.843 ± 0.023 cm |
| Patch token | **1.261 ± 0.054 cm** |

**Non-overlapping seed ranges** — this is the one result in the whole project that is statistically
reliable, not a single lucky run. Adding last-block fine-tuning on top of patch pooling did **not**
help further (1.345 cm, worse than frozen patch pooling) — the frozen patch representation already
seems to be near what this small dataset supports.

**Adaptation levels tested for DINOv2 (all four):**
| Variant | Full-test MAE |
|---|---:|
| Frozen, CLS token (matched HP) | 1.843 ± 0.023 |
| Frozen, patch tokens | 1.261 ± 0.054 |
| Last-block fine-tuned, CLS | 1.439 |
| Full fine-tune, CLS, encoder LR 1e-5 | 1.778 |
| Full fine-tune, CLS, encoder LR 1e-6 | 2.132 (worst result in the whole project) |

**Why full fine-tuning made things worse, not better:** with only ~11,000 training crops, fully
retraining a 21M-parameter Transformer risks catastrophic forgetting of its useful pretrained
knowledge, or unstable optimization — both plausible here given the pattern (gentler LR = worse,
i.e. moving *toward* frozen wasn't the fix either; the fix was changing *which tokens* are read,
not how much of the network trains).

---

## 6. Cross-encoder comparison table (the one-page cheat sheet)

| Encoder | Type | Params | Pretraining | Best config | Full-test MAE |
|---|---|---:|---|---|---:|
| MobileNetV2 | CNN | ~3.4M | ImageNet, supervised | full FT | **0.771** |
| EfficientNet-B0 | CNN | ~5.3M | ImageNet, supervised | full FT (basic recipe) | 0.781 (unconfirmed) |
| ConvNeXt-Tiny | CNN | ~28M | ImageNet, supervised | full FT | 0.914 |
| CLIP ViT-B/32 | ViT | ~88M (image tower) | 400M image-text pairs, contrastive | last-block FT | 0.958 |
| DINOv2 ViT-S/14 | ViT | ~21M | self-supervised, no labels | frozen, patch tokens (3-seed) | 1.261 ± 0.054 |

**The pattern to be ready to explain:** supervised CNNs (top 3) beat both foundation models on this
precise regression task; among foundation models, CLIP (language-shaped) transfers better than
DINOv2 (purely self-supervised); DINOv2's biggest single fix wasn't more training, it was reading
different tokens.

---

## 7. Quick-fire viva Q&A — Part 3 (encoders)

**Q1: What's the one architectural idea MobileNetV2, EfficientNet-B0, and ConvNeXt all share?**
All are CNNs using some form of inverted bottleneck / depthwise-separable convolution to keep
compute down while still growing feature depth.

**Q2: What does compound scaling actually mean?**
Scaling a network's depth, width, and input resolution together in a fixed ratio, rather than
scaling just one dimension arbitrarily — found to be more parameter-efficient (EfficientNet, Tan & Le 2019).

**Q3: What does Squeeze-and-Excitation do, in one sentence?**
It lets the network dynamically re-weight how important each feature channel is for the current
image, via a small "squeeze the whole channel to one number, then predict a per-channel weight" step.

**Q4: What's the fundamental architectural difference between a CNN and a ViT?**
A CNN builds up understanding through many layers of small, local filters (limited receptive field
early on); a ViT lets every patch attend to every other patch in a single self-attention layer
(global receptive field from layer 1), at the cost of needing more data/pretraining to learn good
inductive biases a CNN gets "for free."

**Q5: What is the [CLS] token, technically?**
A learned token prepended to the patch-token sequence in a ViT; through training it aggregates
information via self-attention into a single global summary vector, typically used for
image-level tasks like classification.

**Q6: Why do patch tokens outperform the CLS token for THIS task specifically?**
Length is a spatial/geometric property; the CLS token is optimized for semantic summarization and
discards fine positional detail, while patch tokens retain per-location information — confirmed by
a controlled, 3-seed experiment (1.261 vs 1.843 cm, non-overlapping ranges).

**Q7: Is the patch-token result definitely reliable? How do you know?**
Yes, more reliable than any other single-model result in the project — it used identical
hyperparameters (isolating the token-type variable), 3 different seeds, and the resulting ranges
did not overlap, unlike the EfficientNet-B0 result, which is single-seed.

**Q8: What is contrastive learning, and how does CLIP use it?**
Training a model to pull matching pairs (here: an image and its real caption) close together in
embedding space, while pushing non-matching pairs apart — CLIP does this across ~400M internet
image-caption pairs, with no explicit class labels.

**Q9: Why does CLIP transfer better than DINOv2 here, if neither was designed for length regression?**
💡 Hypothesis: CLIP's language-grounded training may retain more object-level shape/size cues
useful for regression than DINOv2's purely self-supervised objective, which is more oriented toward
general visual similarity/clustering. Not proven, stated as a hypothesis.

**Q10: Why does the project only ever call CLIP's `encode_image()`, never the text side?**
Because the task (predict a number from an image) has no text component — the text tower exists in
the pretrained checkpoint but is simply unused code-path in this project.

**Q11: Why did full fine-tuning of DINOv2 perform WORSE than freezing it?**
Likely overfitting / optimization instability on a small (~11K crop) dataset when retraining a full
21M-parameter Transformer — the pattern (lower encoder LR made results worse, not better) suggests
the issue wasn't "too aggressive a step size" alone, since even the gentlest LR (1e-6) still
underperformed frozen.

**Q12: If DINOv2 patch pooling worked, why not try it for CLIP too?**
It wasn't tried in this project round — explicitly listed as future work, since CLIP's frozen/
last-block results were already reasonably competitive without it.

**Q13: Why is ConvNeXt-Tiny, the largest and most modern CNN tested, not the best?**
Unknown for certain; a plausible explanation is that more parameters and larger receptive fields
don't automatically help on a relatively small, narrow-domain dataset (~11K training crops), and/or
the shared swap-recipe wasn't specifically tuned per encoder.

**Q14: What happens if you increase EfficientNet-B0's learning rate and training length?**
Counter-intuitively, it got worse (0.934 cm vs. the basic recipe's 0.781 cm) — evidence the basic
recipe may already be close to this architecture's ceiling on this dataset size, not that it was
badly under-trained.

**Q15: Rank the five encoders and explain the pattern in one sentence.**
MobileNetV2 (0.771) > EfficientNet-B0 (0.781, unconfirmed) > ConvNeXt (0.914) > CLIP (0.958) >
DINOv2 patch (1.261) — supervised CNNs lead, CLIP's language-shaped features transfer better than
DINOv2's purely self-supervised ones, and DINOv2 needed a specific fix (patch tokens) to become
competitive with CLIP at all.

---

*Part 3 of the staged viva-prep series (39 Q&A total so far across Parts 1–3). Part 4
(hyperparameters) continues next.*
