# Viva Prep — Part 5: The Poster, Box by Box

Builds on Parts 1–4. Every quote below is the **exact, current text** of `poster/AutoFish_A3_poster.html`
(extracted directly from the file, not paraphrased from memory).

---

## Header

> *"Fish Length Estimation with Vision Foundation Models: Which Single Encoder Comes Closest to the
> AutoFish Baseline?"*

**Why this exact title:** it commits to two things at once — the topic (comparing encoders on fish
length) and the honest scope (a *single-encoder* comparison, not an ensemble, and framed as "comes
closest to," not "beats"). If asked "why not just say 'improving on AutoFish'?" — because that would
overclaim; no single model has beaten the baseline yet, and the title reflects that truthfully.

---

## Box: Motivation & Question

> *"Fish length drives fisheries stock assessment; manual measuring is slow and invasive."*
> *"AutoFish (Bengtson et al., WACVW 2025) gives images + a CNN length-regression baseline."*
> **Question:** *"can a modern encoder or vision foundation model beat the reproduced MobileNetV2
> baseline at fish length regression, under identical data, split, and metrics?"*

**Why it exists:** the 30-second test (per the seminar's own scientific-writing lecture) — a reader
standing 2 metres away should immediately grasp the problem and the question. **How to explain to a
professor:** *"We open with why this task matters practically, cite the exact baseline we're
measuring against, then state our research question precisely — including the fairness conditions
(identical data/split/metrics) that make the later comparison valid."*

⚠️ **Known imprecision to be ready to address (see Part 1):** "WACVW 2025" is not confirmed directly
from the arXiv PDF text itself — if pressed, say you would cite arXiv:2501.03767 as the
guaranteed-correct reference.

---

## Box: Data & Setup

> *"1,500 images · 454 fish · 18,157 annotations · 25 groups."*
> *"Group-level split 15/5/5; no fish in two splits (leak fish-113 removed)."*
> *"Test = 3,759 (1,879 non-occluded + 1,880 occluded)."*
> *"Metric: mean absolute error (MAE, cm), lower is better."*

**Why it exists:** establishes the exact scale and structure of the experiment before any result is
shown — a reader needs to know "how much data, how was it split, what's being measured" to interpret
everything that follows.

**Note the small stated discrepancy (18,157 vs. the paper's 18,160)** — this poster states *our*
processed count honestly; see Part 2 §4 for the full explanation of that gap.

**How to explain "leak fish-113 removed" in one breath:** *"One fish accidentally had annotations in
both a training and a test group; we found and removed the stray one, then re-verified zero fish
now cross any split."*

---

## Box: Method / Workflow

> Diagram: *Fish crop (masked) → Resize 224² → Image encoder → + box (4 val.) → Head [512,128,1] → Length (cm)*
> *"Only the encoder is swapped; split, input, head, L1 loss, metrics identical."*
> *"Learning rate set per encoder type (baseline reproduces paper: Adam 1e-3)."*

**Why it exists:** this is the poster's core "how" — a visual proof that the comparison is
controlled. **The second bullet is the poster's own honest disclosure** of the one hyperparameter
that legitimately differs across experiments (see Part 4 §2), pre-empting the "is this really a fair
comparison?" question before it's even asked.

**How to walk a professor through this diagram, pointing at each box in turn:** *"Every fish crop is
masked and resized identically, whichever encoder we're testing goes here [point], we always add the
same 4 bounding-box numbers, feed into the same head shape, and get one number out — length."*

---

## Box: Single-model comparison (the main chart)

> **Takeaway:** *"among single models, MobileNetV2 (0.771 cm) is best and every foundation model
> trails it; supervised CNNs fill the top. EfficientNet-B0 (0.781 cm) comes within 0.010 cm of the
> baseline."*

**What the chart shows:** the full ranking bar chart (`results/figures/length_ranking.png`) — every
tested model's full-test MAE, lower bars = better, colour-coded by encoder family.

**Why "single-model" is emphasized:** a deliberate, explicit scope decision — the project
investigated and then removed an ensemble-based result (a validation-selected weighted combination
of multiple models, which *did* beat the baseline) because an ensemble is a combined-prediction
strategy, not a single trained model, and the poster's stated question is specifically about single
encoders.

---

## Box: Baseline reproduction

> *"Non-occluded test MAE, our run vs. the AutoFish paper: Paper 0.62 cm · Ours 0.633 cm · Δ 0.013
> cm → pipeline validated."*

**Why this box exists, and why it comes early:** this is the poster's credibility anchor — before
trusting any comparison between encoders, the reader needs proof the whole pipeline is faithful to
the paper. A 0.013 cm gap (about a tenth of a millimetre) is presented as validation.

**The refinement from Part 1 you can add verbally (not on the poster itself):** *"0.62 cm is
specifically the paper's REGpd number — predicted-mask evaluation. Since our pipeline always uses
ground-truth masks, the more precise comparison target is REGgt at 0.67 cm, which our 0.633 cm is
actually slightly better than."* This is a stronger, more technically precise answer than what's
printed, and demonstrating it shows depth beyond the poster's simplified framing.

---

## Box: Key findings

> - *"MobileNetV2 is the best single model (0.771 cm)."*
> - *"EfficientNet-B0 nearly matches it (0.781) at a basic recipe."*
> - *"Patch tokens reliably beat CLS for DINOv2 (3 seeds)."*
> - *"Occlusion hurts every model (baseline 0.633 → 0.909)."*

**Why exactly these four, and in this order:** each is a distinct, independently defensible claim —
(1) the headline ranking result, (2) the most exciting *unconfirmed* result (flagged honestly with
"nearly," not "beats"), (3) the one *statistically confirmed* result in the whole project, (4) a
robustness/limitation observation that applies universally, not just to one model. **A professor
question to expect:** "which of these four are you most confident in, and why?" → Answer: #3 (patch
tokens), because it's the only one backed by a controlled multi-seed comparison; #2 is explicitly the
least confident, being single-seed.

---

## Box: Closest to the baseline (chart)

Shows `results/figures/closest_single_models.png` — a focused 3-bar comparison (MobileNetV2 vs.
EfficientNet-B0 vs. ConvNeXt-Tiny) visually highlighting the 0.010 cm gap.

**Why a separate chart from the main ranking:** the main ranking chart has 7+ bars at very different
scales; this chart zooms in specifically on the interesting near-miss, which would be visually lost
in the full ranking.

---

## Box: Patch tokens beat CLS (DINOv2) (chart)

Shows `results/figures/patch_vs_cls.png` — the 3-seed mean±SD bar comparison (1.843±0.023 vs.
1.261±0.054).

**Why this gets its own dedicated box:** it's the project's one statistically rigorous finding and
deserves visual emphasis proportional to its evidentiary strength, not buried in a bullet point.

---

## Box: Species classification (chart)

Shows `results/figures/species_accuracy.png` — accuracy bars for all 4 tested encoders on the second
task.

**Why include a second task on a poster about length regression:** it's a generalization check —
"does our length-regression conclusion (CNNs > foundation models) hold on a different kind of
problem?" The answer, visually obvious from this chart, is "not entirely" — DINOv2 is competitive
here despite being worst at length, which is itself an interesting, poster-worthy finding.

---

## Box: Models & configurations tried (the table)

Exact table on the poster:

| Encoder | Type | Adaptation | Recipe (Adam) | MAE |
|---|---|---|---|---:|
| MobileNetV2 (baseline) | CNN | full FT | 1e-3, 200 ep, b32 | 0.771 |
| EfficientNet-B0 | CNN | full FT | 1e-4, 100 ep, b16 | 0.781 |
| ConvNeXt-Tiny | CNN | full FT | 1e-4, 100 ep, b16 | 0.914 |
| CLIP ViT-B/32 | Transf. | frozen / last-block | 1e-4, 100 ep | 1.002 / 0.958 |
| DINOv2 ViT-S/14 (patch) | Transf. | frozen, patch pool | 1e-4, 100 ep | 1.261 |
| DINOv2 ViT-S/14 (CLS) | Transf. | frozen / last-blk / full | 1e-4 … 1e-6 | 1.44–2.13 |

Plus a caption line: *"Species classification (same encoders, cross-entropy): ConvNeXt 99.6% ·
MobileNetV2 99.1% · DINOv2 98.2% · CLIP 95.1% accuracy."*

**Why a raw table, in addition to charts:** charts show the *pattern*; the table shows the *exact,
reproducible settings* — this is the poster's nod to reproducibility, letting a technically literate
reader see precisely what recipe produced each number without needing the underlying config files.

---

## Box: Future work — path to beat the baseline

> - *"EfficientNet-B0 is only 0.010 cm behind at a basic recipe — a stronger recipe should cross it."*
> - *"Now testing (validation-selected): EfficientNet-B0 & ConvNeXt-Tiny with cosine LR schedule +
>   weight decay, tuned LR (5e-4/1e-3), 200 epochs."*
> - *"Then: EfficientNet-B2, patch pooling for CLIP, larger ViT variants."*
> - *"Mask segmentation as a third task; DINOv3 (pending weight access)."*

**Why this box is framed as "path to beat," not a claim already achieved:** honesty about status —
at the time the poster reflects, the stronger-recipe attempt had actually made EfficientNet-B0
*worse* (0.934 cm, see Part 3 §2), which is not yet reflected in this box's optimistic framing.
**If asked "did the stronger recipe work?" — say directly:** *"Not yet — the first stronger-recipe
attempt actually made results worse, which is itself an informative finding about this
architecture's training sensitivity on a small dataset; the poster's future-work framing predates
that specific result."*

**DINOv3 "pending weight access":** confirmed, investigated directly — the architecture loads via
PyTorch Hub, but Meta's official pretrained weights download returns HTTP 403 Forbidden
(license-gated). Not a code problem; a genuine access blocker.

---

## Box: Limitations

> - *"Single training run for most single models (multi-seed done for the DINOv2 patch-vs-CLS study)."*
> - *"Foundation models tested at small scale (ViT-S/14, ViT-B/32) with limited fine-tuning budgets."*
> - *"No single model beats the baseline yet; mask segmentation not yet evaluated."*

**Why this box exists at all (and why it's good practice, not weakness):** per the seminar's own
scientific-writing guidance, an honest limitations section is a *credibility* signal, not an
admission of failure — it shows awareness of exactly what would need to happen for stronger claims
to be justified.

---

## Footer

> *"Reference: S. H. Bengtson et al., "AutoFish", WACV Workshops 2025 (arXiv:2501.03767). Dataset:
> Hugging Face vapaau/autofish."*
> *"Reproducibility: fixed seeds, official group split, versioned configs, requirements.txt, saved
> checkpoints & per-fish predictions. NVIDIA RTX 5000 Ada (32 GB)."*
> *"AI statement: generative AI assisted with code scaffolding and drafting; all experiments,
> results, and conclusions are the authors' own and were verified by the authors."*

**Why the AI statement is on the poster at all:** required by the course's explicit AI-disclosure
policy — non-negotiable, and its absence would itself be a viva/grading risk regardless of how the
AI was actually used.

---

## Quick-fire viva Q&A — Part 5 (poster)

**Q1: Why does the poster title say "closest to," not "beats"?**
Because no single model has actually beaten the baseline — the title is deliberately scoped to what
was actually found, not what was hoped for.

**Q2: Why is the poster's Motivation box so short?**
Poster-writing convention (per the course's own scientific-writing guidance): short text blocks that
support the visual story, not long paragraphs — the 30-second/2-metre readability tests.

**Q3: Which claim on the poster are you LEAST confident in, and why?**
"EfficientNet-B0 nearly matches [the baseline]" — it's a single run, single seed, and the gap
(0.010 cm) is well within plausible seed-to-seed noise.

**Q4: Which claim on the poster are you MOST confident in, and why?**
"Patch tokens reliably beat CLS for DINOv2" — the only claim backed by a controlled, 3-seed
comparison with non-overlapping result ranges.

**Q5: The poster doesn't mention an ensemble result — was one tried?**
Yes — a validation-selected weighted ensemble of the top models did beat the baseline (0.711 cm), but
it was deliberately excluded from the poster because it's a combined-prediction strategy across
multiple models, not a single trained encoder, which is outside the poster's stated question.

**Q6: Why include species classification on a poster about length regression?**
As a generalization check on the main finding — and it produced an interesting result on its own
(DINOv2 competitive at classification despite being worst at length), worth showing.

**Q7: What does the "Future work" box's cosine-schedule experiment actually show so far?**
The first completed attempt made EfficientNet-B0 worse (0.934 cm), not better — an informative,
if counter-intuitive, negative result not yet reflected in the poster's optimistic framing.

**Q8: Why does the footer specify the exact GPU used?**
Reproducibility disclosure — lets someone attempting to replicate the work know the compute
environment, and is standard practice per the course's reproducibility requirements.

---

*Part 5 of the staged viva-prep series (62 Q&A total so far across Parts 1–5). Part 6 (every
experimental result explained) continues next.*
