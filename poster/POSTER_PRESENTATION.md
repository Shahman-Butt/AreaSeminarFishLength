# Poster Presentation Preparation

Prepared according to the seminar lecture "Scientific Writing and Good Scientific Practice"
(slides 21–22): **~1 minute of speaking, followed by roughly 15–20 minutes of questions.**
The one-minute talk follows the required structure exactly:
**Context → Question → Approach → Result → Limit.**

---

## 1. The one-minute talk (full script, ~150 words)

Practice this until it takes 55–65 seconds at a calm pace.

> **[Context — ~10 s]**
> Fish length is a core measurement in fisheries science, and measuring fish by hand is
> slow and invasive. The AutoFish benchmark provides images, annotations, and a CNN
> baseline for estimating fish length automatically from top-view images.
>
> **[Question — ~10 s]**
> We asked: do modern vision foundation models — DINOv2 and CLIP — or newer supervised
> encoders like ConvNeXt improve fish length estimation over that baseline?
>
> **[Approach — ~15 s]**
> First we reproduced the paper's MobileNetV2 baseline, matching its reported error
> within a fraction of a millimetre. Then we swapped only the image encoder, keeping the
> data split, inputs, regression head, training budget, and metrics identical for every model.
>
> **[Result — ~15 s]**
> The lightweight CNN baseline still wins, at 0.77 centimetres mean error. ConvNeXt-Tiny
> comes closest, CLIP transfers reasonably well, and DINOv2 features transfer poorly.
> Foundation models are not automatically better for precise metric regression.
>
> **[Limit — ~10 s]**
> These are single training runs, so our next steps are multi-seed validation,
> an EfficientNet comparison, and a detailed error analysis.

---

## 2. Cue card (do NOT read the script at the poster — slide 13: no full paragraphs)

Print or memorize these five cues; speak freely around them:

1. **Context** — fish length matters; manual measurement slow; AutoFish benchmark.
2. **Question** — do foundation models (DINOv2, CLIP) / ConvNeXt beat the CNN baseline?
3. **Approach** — reproduce baseline first (0.633 vs 0.62 cm) → swap *only* the encoder, all else fixed.
4. **Result** — MobileNetV2 still best (0.771 cm); order: ConvNeXt < CLIP < DINOv2; point at the figure.
5. **Limit** — single runs; next: 3 seeds, EfficientNet-B0, error analysis.

While speaking, physically point at: the pipeline strip (Approach), the bar chart (Result),
the take-home box (closing sentence).

---

## 3. Numbers to know cold (quick-facts card)

| Fact | Value |
|---|---|
| Paper baseline, non-occluded MAE | **0.62 cm** |
| Our reproduction, non-occluded MAE | **0.633 cm** (Δ = 0.013 cm) |
| Paper baseline, occluded MAE | **1.38 cm** |
| Our reproduction, occluded MAE | **0.909 cm** (better than paper — be ready to discuss) |
| Best overall (full test, 3,759 fish) | MobileNetV2 **0.771 cm** |
| ConvNeXt-Tiny full test | 0.914 cm |
| CLIP last-block fine-tuned / frozen | 0.958 / 1.002 cm |
| DINOv2 best (last-block) / frozen / full FT | 1.439 / 1.738 / 1.778–2.132 cm |
| Dataset | 1,500 images, 454 fish, 18,157 annotations, 25 groups |
| Split | 15 train / 5 val / 5 test groups (official release split) |
| Fixed for all models | crop + 224×224 input, bbox (4 values), head [512,128,1], L1 loss, Adam 1e-4, 100 epochs, seed 42 |
| Baseline R² (full test) | 0.947 |

---

## 4. Q&A preparation (15–20 minutes of questions)

Lecture rules (slide 22): answer the actual question first; use evidence from the workflow;
separate knowledge from assumption; be honest about uncertainty; explain weak choices,
don't defend them stubbornly.

The all-purpose sentence:
> *"Based on our current evidence, we support X; however, Y remains uncertain because …"*

### A. Reproduction questions

**Q: Your non-occluded result is 0.633 vs the paper's 0.62 — is that a successful reproduction?**
A: Yes. The difference is 0.013 cm — about a tenth of a millimetre — far below any practically
relevant threshold and well within what small implementation differences produce. It tells us
our whole pipeline (data handling, split, training, evaluation) is trustworthy, which is the
prerequisite for the encoder comparison.

**Q: Your occluded result (0.909 cm) is much *better* than the paper's 1.38 cm. Why?**
A: Answer honestly — do not hide this. "We match the paper closely on non-occluded fish but
outperform it on occluded fish, and we cannot fully explain the gap. The paper does not specify
every training detail, so plausible causes are differences in preprocessing or augmentation,
checkpoint selection, and our removal of one duplicate fish identity. What we know is that our
evaluation uses the official split and the same metric; what we assume is that our training
recipe happens to generalize better under occlusion. Verifying this — for example with the
official code release and multiple seeds — is part of our next steps." This is a strength if
presented transparently: you noticed it, you report it, you have a plan to investigate it.

**Q: Did you use the official code?**
A: We reimplemented the pipeline following the paper and the official release's split
definition, and validated the reimplementation by reproducing the headline number to 0.013 cm.

### B. Data and split questions

**Q: Why a group-level split instead of a random split over crops?**
A: Each physical fish appears in many images. A random crop-level split would put the same
fish in train and test, and the model could partly memorize individual fish — that is data
leakage, and it would inflate all results. Splitting by fish group guarantees unseen fish at
test time. This mirrors the good-practice example from the lecture about split leakage.

**Q: You mentioned removing a duplicate fish. What was that?**
A: During leakage checks we found one fish identity (fish_id 113) present on both sides of the
split boundary via a singleton duplicate annotation. We removed that annotation, which is why
the non-occluded test set has 1,879 samples instead of 1,880. Effect on metrics is negligible,
but the split is now provably leakage-free.

**Q: What exactly are "non-occluded" and "occluded" test sets?**
A: The dataset's image sets: Set1 and Set2 show fish laid out separately; the "All" set shows
fish overlapping each other in a pile. We report both because occlusion is the realistic hard
case — and every model degrades there (baseline: 0.633 → 0.909 cm).

### C. Fairness-of-comparison questions

**Q: Is it fair to compare a fine-tuned CNN with a *frozen* DINOv2/CLIP?**
A: Frozen evaluation answers a specific question — "do general-purpose features transfer
out-of-the-box?" — which is the standard foundation-model use case. But we did not stop there:
we also fine-tuned DINOv2 fully (two learning rates) and partially (last block), and fine-tuned
CLIP's last visual block. Every adaptation level we tried still trails the CNN, so the
conclusion does not depend on the frozen setting.

**Q: The bounding box is an input — doesn't the box size alone predict length?**
A: The box gives a strong geometric prior, and the original AutoFish baseline uses it too, so
we keep it for comparability across all encoders. The image features matter on top of it:
fish are curved, oriented differently, and occluded, so the box alone is not sufficient —
which is exactly where the encoders differentiate (all models share the same bbox input, yet
their errors differ by up to 1.4 cm).

**Q: Why the same head and training budget for every encoder — wouldn't each deserve tuning?**
A: A controlled comparison requires changing one variable at a time; per-model tuning would
confound encoder quality with tuning effort. We acknowledge that with per-encoder
hyperparameter search individual numbers could improve — that is future work, and it is more
likely to help the challengers than to change the baseline's rank by itself.

### D. Interpretation questions

**Q: Why does DINOv2 perform so poorly?**
A: Separate what we know from what we assume. Known: all DINOv2 variants stay above 1.4 cm.
Our hypotheses: DINOv2's self-supervised features are optimized for semantic discrimination,
not metric geometry; the global feature vector discards fine spatial detail needed for
millimetre regression; and full fine-tuning of a ViT on a dataset of this size is unstable —
we observed that a too-conservative learning rate (1e-6) was even worse than frozen features.

**Q: Why is CLIP so much better than DINOv2 here?**
A: Honest answer: this is an observation we can report but only partly explain. Possibly
CLIP's much larger and more diverse pretraining data yields features that retain more
shape/size information. Based on our current evidence, CLIP transfers better; *why* remains
uncertain because we did not run representation-level analyses.

**Q: Your take-home says CNNs beat foundation models. Isn't that overclaiming from one dataset?**
A: The claim is scoped: for *this* task (precise metric length regression), *this* dataset, and
the model scales we tested, the task-adapted CNN wins. We do not claim foundation models are
generally worse — at larger scale, with more adaptation (e.g. LoRA, full fine-tuning with better
schedules), or on tasks needing semantics, the picture may differ.

**Q: Is 0.77 cm good enough in practice?**
A: For catch monitoring and stock assessment, errors under ~1 cm on fish of roughly 20–45 cm
(2–3% relative error) are generally useful; our MAPE is ~2.4%. The occluded regime is the
bottleneck for real deployments — that is where improvement matters most.

### E. Rigor questions

**Q: One run per model — how do you know the ranking is stable?**
A: Use the lecture's sentence: "Based on our current evidence, MobileNetV2 leads by 0.14 cm
over the next model; however, the exact gaps remain uncertain because each result is a single
run. The large gaps — CNNs vs DINOv2, roughly 0.7 cm — are very unlikely to be seed noise;
the small ones need the 3-seed repetition we have planned." Never claim the small gaps are
significant.

**Q: What would make you change your conclusion?**
A: If multi-seed runs showed run-to-run variance comparable to the MobileNetV2–ConvNeXt gap,
or if a properly fine-tuned larger foundation model (or EfficientNet) beat 0.771 cm, we would
revise the ranking. The conclusion is evidence-based, not a position to defend.

**Q: How reproducible is this?**
A: Fixed seed, versioned JSON config per experiment, the official split hard-coded in configs,
requirements.txt, saved checkpoints and per-fish prediction CSVs, and documented hardware
(one RTX 5000 Ada, 32 GB). Anyone with the dataset can rerun each experiment from its config.

**Q: Did you use AI tools?**
A: Yes — disclosed on the poster per course policy: generative AI assisted with code
scaffolding and text drafting; all experiments, results, and conclusions are my own work and
were verified by me. (Slide 23: AI use is allowed but must be disclosed; you remain
responsible for correctness.)

### F. If you don't know an answer

Do not bluff. Use: "I don't know that from our experiments; my assumption would be …, and here
is how I would test it." The lecture explicitly rewards honest uncertainty over stubborn defense.

---

## 5. Rehearsal checklist

- [ ] Time the talk: 55–65 s, three rehearsals minimum.
- [ ] Speak from the cue card, never read from the poster (slide 13).
- [ ] Point at figure/pipeline while talking — the poster is your visual aid.
- [ ] Rehearse the occluded-discrepancy answer (Section 4.A) out loud — it is the most likely hard question.
- [ ] Know the quick-facts card (Section 3) without looking.
- [ ] Two-meter test on the printed poster: question, method, main figure, conclusion visible from distance (slide 20).
- [ ] 30-second test with a friend: can they state the problem, approach, main figure message, and conclusion? (slide 19)
