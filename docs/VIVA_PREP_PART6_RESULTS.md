# Viva Prep — Part 6: Every Experimental Result, Explained

Builds on Parts 1–5. Every number below is from `runs/*/test_metrics.json`, cross-checked against
its `config.json`. Metrics themselves (MAE, RMSE, MAPE, bias, R², accuracy, macro-F1) are explained
conceptually in Part 4's neighbor topics and `src/autofish_vfm/metrics.py`'s code comments — this
part focuses on *what each experiment's config was, what result it got, and why*.

---

## 1. Baseline — MobileNetV2 (`baseline_official`)

**Config:** full fine-tune, head `[1000,500,1]`, Adam LR 1e-3, batch 32, 200 epochs, seed 42 —
copied directly from the paper's own recipe.

**Result:**
| Subset | MAE | RMSE | MAPE | Bias | R² |
|---|---:|---:|---:|---:|---:|
| Full test | 0.771 | — | — | — | — |
| Non-occluded | 0.633 | — | — | — | — |
| Occluded | 0.909 | — | — | — | — |

**Why this result happened:** faithful reproduction of a recipe already validated by the paper
itself — expected to land close to the paper's numbers, and it did (see Part 1 §6.3 for the
corrected REGgt comparison). No surprises here; this experiment's *purpose* was calibration, not discovery.

---

## 2. EfficientNet-B0, basic recipe (`efficientnet_b0`)

**Config:** full fine-tune, head `[512,128,1]`, Adam LR 1e-4, batch 16, 100 epochs, seed 42 — the
same "swap recipe" family as ConvNeXt/CLIP/DINOv2.

**Result:** full-test 0.781, non-occluded 0.670, **occluded 0.893** (notably *better* than the
baseline's 0.909 on this specific subset).

**Why this result happened (💡 hypothesis):** architectural similarity to MobileNetV2 (both compact
supervised CNNs) plausibly transfers well; the occluded-subset strength specifically is unexplained —
worth noting as a genuinely interesting sub-finding: EfficientNet-B0 beats the baseline on occluded
fish specifically, even though it loses on non-occluded fish and therefore on the full-test average.
**A good viva answer if asked "does EfficientNet-B0 ever actually beat the baseline?":** *"Yes — on
the occluded subset specifically (0.893 vs 0.909 cm), though this is drowned out by the non-occluded
subset in the full-test average, and remains a single, unconfirmed run."*

## 3. EfficientNet-B0, "strong A" recipe (`efficientnet_b0_strong_a`)

**Config:** full fine-tune, Adam LR 1e-3 (10× higher), cosine schedule, weight decay 1e-4, batch 32,
200 epochs (double), seed 42.

**Result:** 0.934 — **worse** than both the baseline and the basic-recipe EfficientNet-B0.

**Why this result happened (💡 hypothesis):** with only ~11,000 training crops, a 10× higher learning
rate combined with double the training length likely pushed the model past its optimal point,
either overfitting to training-specific noise or landing in an unstable region of the loss surface
that the cosine schedule's late-training small steps couldn't recover from.

## 4. ConvNeXt-Tiny (`convnext_tiny_official`)

**Config:** identical "swap recipe" to EfficientNet-B0's basic run (LR 1e-4, batch 16, 100 epochs).

**Result:** 0.914 full-test — third among single models, clearly behind both smaller CNNs.

**Why this result happened (💡 hypothesis):** more parameters (~28M vs. ~3.4–5.3M) don't
automatically help on a relatively small, narrow-domain dataset; may also be under-tuned relative to
its potential, since it received the same generic recipe as every other swap experiment rather than
architecture-specific tuning.

## 5. CLIP ViT-B/32, frozen (`clip_vitb32_frozen`)

**Config:** encoder fully frozen, only head trains, LR 1e-4, batch 16, 100 epochs.

**Result:** 1.002 full-test.

**Why:** tests the raw transferability of CLIP's pretrained features with zero task adaptation —
establishes a "floor" for what CLIP alone (unadapted) can do.

## 6. CLIP ViT-B/32, last-block fine-tuned (`clip_vitb32_lastblock`)

**Config:** last visual transformer block + final norm/projection unfrozen, warm-started from the
frozen run's checkpoint, same LR/batch/epochs family.

**Result:** 0.958 — an improvement over frozen (1.002).

**Why:** a small amount of task-specific adaptation lets CLIP's features shift slightly toward what
this task needs, without destabilizing the bulk of its pretrained knowledge — the "sweet spot"
partial-adaptation pattern also seen with DINOv2 (see below).

## 7. DINOv2, CLS token, frozen — ORIGINAL uncontrolled run (`dinov2_vits14_frozen`)

**Config:** LR **1e-3** (not 1e-4 — this run predates the controlled study and used a different LR
than the later matched comparison), batch 32, 100 epochs.

**Result:** 1.738.

**Why flagged specially:** this is the number that appears in the *original* project results, but it
is **not directly comparable** to the patch-token result, because the hyperparameters differ (see
next entries) — this is exactly why the controlled re-run (#10 below) was necessary.

## 8. DINOv2, CLS token, last-block fine-tuned (`dinov2_vits14_lastblock`)

**Config:** last transformer block unfrozen, warm-started from the frozen run, batch 8 (smaller,
likely memory-driven), 50 epochs, split encoder/head learning rates.

**Result:** 1.439 — the best of the *original* (non-patch-token) DINOv2 variants.

## 9. DINOv2, CLS token, full fine-tune, two learning rates (`finetune_lr1e5`, `finetune_lr1e6`)

**Results:** encoder LR 1e-5 → 1.778; encoder LR 1e-6 → **2.132** (the worst result in the entire
project).

**Why the pattern is backwards from intuition (💡 hypothesis):** normally a gentler learning rate is
"safer." Here, the *gentlest* setting (1e-6) performed worst — suggesting the problem with full
fine-tuning here isn't simply "too large a step," since making steps even smaller didn't help.
Plausible explanation: full fine-tuning of a 21M-parameter Transformer on ~11K crops is fundamentally
poorly matched to this data regime regardless of step size, and 1e-6 may simply be too slow to make
any beneficial adaptation within the epoch budget while still slowly drifting away from the useful
frozen starting point.

## 10. DINOv2, patch tokens, frozen — the controlled study (3 seeds)

**Configs:** `dinov2_vits14_patchtokens_frozen` (seed 42), `_s1`, `_s2` — LR **1e-4** (matched to the
CLS-token control below), batch 16, 100 epochs.

**Result:** 1.199, 1.330, 1.253 → **mean 1.261 ± SD 0.054**.

## 11. DINOv2, CLS token, frozen — the matched CONTROL (3 seeds)

**Configs:** `dinov2_vits14_clstoken_ctrl_s42/s1/s2` — **identical** LR 1e-4, batch 16, 100 epochs to
#10, differing *only* in `use_patch_tokens`.

**Result:** 1.811, 1.863, 1.855 → **mean 1.843 ± SD 0.023**.

**Why #10 vs #11 is the project's gold-standard comparison:** every hyperparameter matches exactly;
the only variable is which tokens are read out. Non-overlapping ranges (patch max 1.330 < CLS min
1.811) make this a statistically credible, not merely suggestive, result.

**Note the LR correction this required:** the *original* frozen DINOv2 result (#7, 1.738 cm) used LR
1e-3, not 1e-4 — meaning it is NOT the correct baseline for judging the patch-token improvement.
The properly controlled comparison uses #11 (1.843, at LR 1e-4) as the CLS baseline, not #7. **If
asked "so which is the 'real' frozen-DINOv2-CLS number, 1.738 or 1.843?"** — answer: *"Both are real
results, at different hyperparameters; 1.843 is the one directly comparable to the patch-token result
because every other setting is matched; 1.738 was an earlier, uncontrolled run at a different
learning rate."*

## 12. DINOv2, patch tokens + last-block fine-tune (`patchtokens_lastblock`)

**Config:** starts from the frozen patch-token checkpoint, last block unfrozen.

**Result:** 1.345 — **worse** than frozen patch pooling (1.261 mean).

**Why:** the frozen patch-token representation already appears to be close to what this dataset size
can support; further fine-tuning on top of it doesn't help and mildly hurts, likely for the same
small-dataset instability reasons as DINOv2's full fine-tuning experiments above.

## 13. Species classification (4 runs)

**Configs:** same 4 encoders (MobileNetV2, ConvNeXt-Tiny, CLIP frozen, DINOv2 frozen), head width
changed to 7 (one per species), cross-entropy loss instead of L1, otherwise matched training settings.

**Results:** ConvNeXt 99.57% / macro-F1 99.21%; MobileNetV2 99.12% / 98.95%; DINOv2 98.19% / 97.75%;
CLIP 95.13% / 95.45%.

**Why the ranking differs from length regression (💡 hypothesis, the project's key generalization
finding):** species classification is a *semantic* task (what kind of object is this?), where
self-supervised/language-pretrained features are known to excel; length regression is a *geometric*
task (how big, precisely?), which these same features handle worse. DINOv2 rising from last place
(length) to near the top (species) is the clearest single piece of evidence for this hypothesis in
the whole project — though it remains a hypothesis, not a proven causal mechanism.

---

## Cross-experiment patterns worth stating proactively

1. **Every model gets worse on occluded fish** (except EfficientNet-B0's specific subset anomaly,
   #2 above) — occlusion is a genuine, universal difficulty driver, not encoder-specific.
2. **Partial (last-block) fine-tuning outperformed both frozen and full fine-tuning** for both CLIP
   and DINOv2's CLS-token variants — a consistent "sweet spot" pattern across two different
   foundation models.
3. **Full fine-tuning of DINOv2 was uniformly bad**, and got worse with gentler learning rates —
   the clearest sign that full fine-tuning itself, not just the learning rate choice, is poorly
   suited to this data regime for this model.
4. **The only truly apples-to-apples DINOv2 comparison is #10 vs. #11** (both at LR 1e-4); any other
   pairing (e.g. citing the original 1.738 cm number) involves an uncontrolled hyperparameter
   difference and should be flagged as such if used.

---

## Quick-fire viva Q&A — Part 6 (results)

**Q1: Which experiment result are you personally most surprised by?**
EfficientNet-B0 beating the baseline specifically on the occluded subset (0.893 vs. 0.909 cm), even
though it loses overall — an unexplained but genuine sub-finding.

**Q2: Why does the original frozen-DINOv2-CLS result (1.738 cm) differ from the controlled study's
CLS baseline (1.843 cm)?**
They used different learning rates (1e-3 vs. 1e-4) — the 1.843 cm number is the one properly matched
to the patch-token comparison; 1.738 cm was an earlier, differently-configured run.

**Q3: Why was full fine-tuning bad for DINOv2 at every learning rate tried, including very gentle ones?**
It suggests the problem isn't simply "too large a step size" — full retraining of a 21M-parameter
Transformer on ~11K crops may be fundamentally poorly matched to this data regime regardless of LR.

**Q4: What's the strongest evidence in this project that foundation models suit semantic tasks
better than geometric ones?**
DINOv2 going from worst-at-length (1.261–2.132 cm depending on variant) to near-top-at-species
(98.19% accuracy) with the same encoder, same dataset, only the task changing.

**Q5: Did partial fine-tuning help both foundation models, or just one?**
Both — CLIP improved from 1.002 (frozen) to 0.958 (last-block); DINOv2's CLS-token improved from
1.843 (frozen, controlled) to 1.439 (last-block, though at different uncontrolled hyperparameters).

**Q6: Is the ConvNeXt result (0.914 cm) fully explained?**
No — it's a single run at the same generic swap recipe as every other model; it's unknown whether
architecture-specific tuning would change its ranking relative to EfficientNet-B0.

**Q7: What would you need to do to make the EfficientNet-B0 "closest" result publication-worthy?**
Repeat it with multiple seeds (like the DINOv2 patch-token study) to establish a mean and spread,
and ideally repeat the baseline itself with multiple seeds too, so the 0.010 cm gap can be judged
against a known noise floor rather than compared as two single points.

---

*Part 6 of the staged viva-prep series (69 Q&A total so far across Parts 1–6). Part 7 (code
walkthrough) continues next.*
