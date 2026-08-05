# Viva Prep — Part 9: Master Question Bank

This is the consolidated index of every viva question across the whole staged series, plus new
questions added here to reach 150+. Total: **152 questions.**

**Index of Parts 1–8's questions (82 total, already written in full — go to that file for the
answer):**

| Part | File | Topic | Count |
|---|---|---|---:|
| 1 | `VIVA_PREP_PART1_PAPER.md` §10 | The AutoFish paper | 17 |
| 2 | `VIVA_PREP_PART2_OUR_PROJECT.md` §9 | Our project overview | 7 |
| 3 | `VIVA_PREP_PART3_ENCODERS.md` §7 | Every encoder | 15 |
| 4 | `VIVA_PREP_PART4_HYPERPARAMETERS.md` §15 | Every hyperparameter | 15 |
| 5 | `VIVA_PREP_PART5_POSTER.md` (end) | The poster | 8 |
| 6 | `VIVA_PREP_PART6_RESULTS.md` (end) | Every result | 7 |
| 7 | `VIVA_PREP_PART7_CODE.md` §7 | The code | 7 |
| 8 | `VIVA_PREP_PART8_CONFUSIONS.md` (end) | Confusions | 6 |
| **9 (below)** | this file | New categories | **70** |

---

## A. Ethics (5)

**A1: What ethical review does this project rely on?**
None conducted by us directly — we inherit the paper's own clearance: fish were already dead at
landing (normal commercial catch), collection cleared by the Danish Ministry of Food, Agriculture
and Fisheries, compliant with EU/Danish animal-experimentation law (Part 1 §9).

**A2: Does your project raise any new ethical concerns beyond the paper's?**
Not substantively — we only reuse the existing dataset computationally; no new data collection, no
new animal involvement.

**A3: Could this technology be misused?**
A fair question to have a considered answer for: automated catch monitoring could in principle be
used for stricter enforcement in ways that affect livelihoods, but the stated motivation (both the
paper's and ours) is sustainability and anti-overfishing, a broadly positive framing.

**A4: What does the AI-disclosure statement on your poster actually cover?**
Generative AI assisted with code scaffolding and text drafting; all experiments, results, and
conclusions were produced by the project's own pipeline and verified by the authors — required by
the course's explicit AI-use policy.

**A5: If you used AI to help write code, how do you know the code is correct?**
Every experiment's output is a concrete, checkable artifact (a `test_metrics.json` with real
numbers); results were sanity-checked against the paper's own numbers (the reproduction step) before
being trusted for anything further.

## B. Statistics & experimental design (10)

**B1: What is a confidence interval, and does this project report any?**
A range that's expected to contain the true value with some probability, typically derived from
repeated measurements. This project reports mean ± standard deviation for the one multi-seed study
(DINOv2 patch-vs-CLS) but not formal confidence intervals or significance tests anywhere.

**B2: Why 3 seeds specifically, not 5 or 10?**
A pragmatic compromise between statistical credibility and GPU time — 3 seeds is enough to show a
spread and check for overlap, though more would strengthen the claim further.

**B3: What statistical test would you use to confirm the patch-vs-CLS difference is significant?**
A test such as an independent two-sample t-test (or a non-parametric equivalent given only 3 samples
per group) could formalize it; this project instead relies on the simpler, transparent check of
non-overlapping ranges, which is suggestive but not a formal significance test.

**B4: Is "non-overlapping ranges" the same as statistical significance?**
Not formally — it's an informal, intuitive indicator with only 3 samples per group; a proper test
would give a p-value, which was not computed here.

**B5: What is the danger of "test-set tuning," and did you avoid it?**
Repeatedly checking test performance while making decisions inflates apparent performance
dishonestly. Avoided by construction: checkpoint selection always uses validation; the test set is
touched once per experiment by a separate script.

**B6: What's a Type I vs Type II error, applied to your EfficientNet-B0 claim?**
Type I (false positive): claiming EfficientNet-B0 is close to/beats the baseline when it's actually
just noise. Type II (false negative): dismissing a real advantage as noise. With only 1 seed, this
project cannot currently distinguish between these — an honest limitation.

**B7: What is statistical power, and does this project have enough of it?**
Power = the ability of a study design to detect a real effect if one exists. With single-seed
comparisons for most experiments, statistical power is low for small differences (like the 0.010 cm
EfficientNet gap); it's higher for large differences (like the 0.6 cm CNN-vs-DINOv2 gap), which don't
need formal testing to be credible.

**B8: What's the difference between an ablation study and a hyperparameter sweep?**
An ablation removes/changes ONE component to isolate its effect (e.g. the patch-vs-CLS study,
changing only the token type). A hyperparameter sweep tries many settings of a continuous or
categorical knob (e.g. the stronger-recipe search trying different LRs/schedules) to find a good
operating point — this project has one clean ablation and one small, informal sweep.

**B9: Why report bias (mean signed error) in addition to MAE?**
MAE alone can't distinguish "small random errors in both directions" from "consistent over- or
under-prediction" — bias captures that directional information (Part 4/metrics.py).

**B10: If you ran the whole project again from scratch with different seeds throughout, would you
expect the same conclusions?**
The large, family-level gaps (supervised CNNs vs. DINOv2) would almost certainly hold; the small
gaps (MobileNetV2 vs. EfficientNet-B0, or the exact ConvNeXt ranking) might shift — an honest,
carefully qualified answer, not a blanket "yes."

## C. Reproducibility (8)

**C1: What makes this project reproducible?**
Fixed seeds, versioned config JSON per experiment, the official (paper-specified) group split
hard-coded, `requirements.txt`, saved checkpoints, saved per-fish predictions, and documented
hardware (Part 5's footer, Part 7's file map).

**C2: If someone else ran your code today, would they get identical numbers?**
Very likely close, given the same seed/config/hardware/library versions; exact bit-for-bit
reproducibility across different GPU hardware or library versions is not guaranteed (a general
caveat of deep learning, not specific to this project).

**C3: What's saved in `config.json` inside each run folder, and why?**
A verbatim copy of the exact settings used for that experiment, written at the start of training —
so months later, anyone can see precisely which hyperparameters produced a given result without
cross-referencing a possibly-since-edited `configs/` file.

**C4: Why version each experiment's config as a separate file instead of one shared config with
flags?**
Each JSON file is an immutable, self-contained record of one specific experiment — editing a shared
config after the fact could silently invalidate the provenance of past results.

**C5: Is the dataset itself reproducible/re-downloadable?**
Yes — `scripts/download_autofish.py` pulls the exact dataset from its public Hugging Face repository
(`vapaau/autofish`), not a local, uncatalogued copy.

**C6: What's NOT fully reproducible or disclosed in this project?**
Exact GPU-driver/CUDA versions aren't pinned as tightly as Python packages; multi-seed results exist
for only one experiment, so the "typical noise level" for other results isn't independently verified.

**C7: Why keep both `last.pt` and `best.pt` rather than just the final epoch's weights?**
`last.pt` supports resuming an interrupted run; `best.pt` is the actual result — if training degrades
in later epochs (possible, e.g. via overfitting), the final epoch's weights would be a worse choice
than the validation-best checkpoint.

**C8: How would a third party verify your reported MAE numbers without retraining anything?**
By reading `runs/<experiment>/test_metrics.predictions.csv` (the raw per-fish predictions) and
recomputing MAE themselves — this file is the auditable evidence trail, independent of trusting our
summary numbers.

## D. General deep learning / PyTorch fundamentals (12)

**D1: What is a tensor?**
A multi-dimensional array of numbers (generalizing scalars, vectors, matrices to any number of
dimensions) — PyTorch's core data structure, e.g. an image batch is a 4D tensor
`[batch, channels, height, width]`.

**D2: What does `.to(device)` do?**
Copies a tensor (or a whole model) onto a specific piece of hardware — CPU or GPU (`cuda`) — so
computation happens there; PyTorch operations require all involved tensors to be on the same device.

**D3: What is `model.eval()` vs `model.train()`?**
Switches certain layers' *behavior* (not their weights) — e.g. Dropout is disabled in eval mode, and
BatchNorm uses its stored running statistics instead of the current batch's statistics. Essential to
call `.eval()` before evaluation/inference and `.train()` before resuming training.

**D4: What does `torch.no_grad()` do, and why use it during evaluation?**
Disables gradient tracking for everything inside its block — saves memory and computation when
you're only reading predictions out of a model, not training it.

**D5: What is an activation function, and why does the head use ReLU?**
A non-linear function applied after a linear layer, without which stacking multiple linear layers
would collapse into a single equivalent linear layer (no added expressive power). ReLU
(`max(0, x)`) is simple, fast, and avoids some issues (like vanishing gradients) that older
activations (sigmoid/tanh) suffer from.

**D6: What does BatchNorm do, conceptually?**
Normalizes a layer's outputs (per batch) to have roughly zero mean and unit variance before the next
layer, which tends to stabilize and speed up training; also has a mild regularizing side effect.

**D7: Why does the final layer of the regression head have no BatchNorm/activation after it?**
The final output needs to be an unrestricted number (a length can be any positive value) — adding
ReLU would clip it to non-negative only (accidentally fine here, but not the reason it's omitted),
and BatchNorm on a single-output layer would be unusual/unhelpful; the convention is to leave the
final regression/logit layer "raw."

**D8: What's the difference between `nn.Module` and a plain Python class?**
`nn.Module` is PyTorch's base class for anything with learnable parameters — it auto-registers
sub-layers and their weights, enables `.parameters()`, `.to(device)`, `.train()/.eval()`, and
integrates with autograd; a plain class has none of this built in.

**D9: What is overfitting, in your own words, and where did you see evidence of it in this project?**
When a model learns training-specific noise/quirks rather than generalizable patterns, showing as
good training performance but worse validation/test performance. The EfficientNet-B0
"stronger recipe" (more epochs, higher LR) getting *worse* test results is a plausible sign of this.

**D10: What is transfer learning, in one sentence?**
Reusing a model's knowledge from one task/dataset (e.g. ImageNet classification) as a starting point
for a different task (e.g. fish length regression), instead of training from random weights.

**D11: Why does a Vision Transformer typically need more pretraining data than a CNN to reach
similar performance?**
CNNs have built-in spatial priors (locality, weight-sharing) that encode useful assumptions about
images "for free"; ViTs have weaker built-in priors and must learn similar useful behavior purely
from data, requiring more of it.

**D12: What is the vanishing gradient problem, and is it relevant to this project?**
When gradients become extremely small as they propagate backward through many layers, effectively
stopping early layers from learning. Modern architectures (ResNet-style skip connections, which
MobileNetV2/ConvNeXt/EfficientNet all use in some form, and Transformer layer norms) largely mitigate
this — not something this project observed as an active problem, but relevant background knowledge.

## E. Segmentation & the out-of-scope parts (6)

**E1: In one sentence, what does Mask2Former do?**
A unified Transformer-based architecture for instance/semantic/panoptic segmentation, using masked
attention to predict a set of object masks and their classes directly.

**E2: Why is Mask2Former considered "state of the art" (or was, at time of the paper)?**
It unifies previously separate segmentation task formulations (instance/semantic/panoptic) under one
architecture and attention mechanism, and empirically outperformed many task-specific prior methods.

**E3: What's the difference between the ResNet-50 and Swin-B backbones tested in the paper?**
ResNet-50 is a CNN backbone; Swin-B is a Vision Transformer backbone (hierarchical, windowed
self-attention). The paper found Swin-B modestly outperforms ResNet-50 across every category.

**E4: Why didn't your project also compare CNN vs. Transformer backbones for segmentation?**
Out of scope by design — our research question is specifically about the length-regression encoder,
not the upstream segmentation step, which we never touch.

**E5: If you were to extend this project to segmentation, what would you test?**
A natural extension: swap Mask2Former's backbone the same way we swapped the regression encoder, to
see if the same CNN-vs-foundation-model pattern holds for segmentation too — listed as a plausible
future direction, though not attempted.

**E6: What is SAM (Segment Anything Model), and how is it different from Mask2Former in this paper?**
SAM is used only as an **annotation-time tool** — to propose initial masks from point clicks during
dataset creation, later hand-corrected. Mask2Former is the **trained, evaluated model** used for the
paper's actual segmentation experiments. They serve completely different roles; SAM never appears as
an evaluated baseline in the paper's results tables.

## F. Fair comparison / experimental design meta-questions (7)

**F1: What exactly was held constant across every length-regression experiment?**
Dataset, official group split, crop/mask preprocessing, bbox input, head *family* shape, L1 loss,
and the 3-way evaluation protocol (full/non-occluded/occluded) — see Part 2 §5.

**F2: What was NOT held constant, and is that a flaw?**
Learning rate and epoch count differ between the baseline (paper's own recipe) and the swap
experiments (a uniform, gentler recipe) — disclosed explicitly, not a hidden flaw, though it does
mean the baseline-vs-swaps comparison isn't perfectly apples-to-apples in that one respect.

**F3: If EfficientNet-B0 and MobileNetV2 both used the SAME exact recipe, would that be more or less
fair?**
Arguably more fair for isolating architecture, but the project's actual design instead uses the
paper's own recipe for the baseline specifically, to preserve faithful reproduction — a genuine
trade-off between two different notions of "fair" that's worth being able to articulate.

**F4: Why didn't you tune each encoder's hyperparameters individually to give everyone their best shot?**
A deliberate design choice: individually tuning each encoder would confound "is this encoder
better" with "did we tune this encoder harder," making the comparison less controlled — a shared
recipe (with the one disclosed baseline exception) isolates the encoder as the variable of interest.

**F5: Doesn't NOT tuning each encoder individually bias the results AGAINST foundation models,
which are known to be sensitive to fine-tuning recipes?**
A fair critique to acknowledge directly: yes, this is a real limitation — CLIP/DINOv2 might perform
better with more careful, encoder-specific tuning; the project's numbers should be read as "under a
shared, generic recipe," not as each encoder's absolute best possible performance.

**F6: What would a "more rigorous" version of this comparison look like?**
Multi-seed everything (not just DINOv2 patch-vs-CLS), a small hyperparameter search per encoder
family, and formal significance testing on the resulting distributions — explicitly listed as future
work rather than claimed as already done.

**F7: Is comparing "0.771 cm" to "0.781 cm" as if they're meaningfully different numbers a mistake?**
Only if presented without the caveat — this project explicitly labels the EfficientNet-B0 result as
"closest, unconfirmed" rather than claiming it as a proven difference, precisely because a 0.010 cm
gap from single runs isn't statistically meaningful on its own.

## G. ODE-report-specific (6)

**G1: What does ODE stand for in this course's context, and what are its three parts?**
Overview, Data, Execution — a structured protocol for documenting an ML workflow so it's reviewable
and reproducible (per the course's own scientific-writing lecture).

**G2: What goes in the "Overview" section of your ODE, in one sentence?**
The research question, target, scope, and reasoning for the chosen setup — see
`docs/ODE_REPORT.md`'s O section.

**G3: What goes in the "Data" section?**
Dataset facts, preprocessing steps, splits, and the leakage audit — see the ODE's D section.

**G4: What goes in the "Execution" section?**
Models tested, configurations, training protocol, metrics, and results — see the ODE's E section.

**G5: Why was the ODE peer-reviewed by another team before the final version?**
Per the course's structure — a draft ODE is reviewed by peers to catch missing information,
unclear assumptions, or leakage risks before the final hand-in, mirroring real scientific peer review.

**G6: What's the difference between your ODE's draft and final version, conceptually?**
Per the course's own guidance: the draft needs to be specific enough to review and make assumptions/
risks visible, without needing to look finished; the final version needs sharper wording, final
results, and stronger reproducibility information.

## H. General viva strategy (4)

**H1: If you don't know the answer to a question, what should you say?**
State plainly what you do know, what you're unsure of, and (if possible) how you'd find out —
honest uncertainty is explicitly rewarded over confident guessing, per the course's own scientific-
writing guidance (Part 1's citation of the lecture material).

**H2: What's the single most important number to have memorized cold?**
0.771 cm (baseline) — everything else in the project is contextualized relative to it.

**H3: What's the single most defensible claim in the whole project?**
The DINOv2 patch-vs-CLS result (1.261±0.054 vs 1.843±0.023 cm, 3 seeds, non-overlapping ranges) —
the only claim with genuine statistical backing.

**H4: What's the single claim you should be MOST careful not to overstate?**
"EfficientNet-B0 nearly matches the baseline" — always pair this with "single run, unconfirmed."

---

## Running total across all 9 parts: **152 questions**

*This completes the planned staged viva-prep series (Parts 1–9). All nine documents live in
`docs/VIVA_PREP_PART*.md`. Recommended next step for actual exam prep: read each part once, then
do a self-test pass covering all 152 questions from memory, checking answers against the files.*
