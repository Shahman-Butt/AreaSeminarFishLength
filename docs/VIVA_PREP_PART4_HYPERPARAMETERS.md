# Viva Prep — Part 4: Every Hyperparameter, Individually

Builds on Parts 1–3. Grounded in `src/autofish_vfm/train_baseline.py` and `configs/*.json` (both
already heavily commented in the repo — this document explains the *concepts* in depth; the code
comments explain the *exact lines*).

---

## 1. Parameters vs. hyperparameters — the foundational distinction

**Layman:** Parameters are what the model *learns* (the millions of numbers inside its layers).
Hyperparameters are what *we* choose before training even starts, and the model never changes them
on its own.

**Technical:** Parameters (weights, biases) are updated by gradient descent during training.
Hyperparameters (learning rate, batch size, epochs, architecture choices like head shape) are fixed
externally, typically in a `config.json` file in this project, and control *how* training happens
rather than being learned by it.

---

## 2. Learning rate

**What it is (layman):** how big a step the model takes each time it corrects itself. Analogy:
adjusting a shower's temperature — a huge turn overshoots hot/cold repeatedly; a tiny turn takes
forever to reach the right temperature.

**What it is (technical):** the scalar multiplier on the gradient in the weight-update rule
(`weight = weight - learning_rate × gradient`, in its simplest form; Adam modifies this with
momentum and per-parameter adaptive scaling — see §9).

**Why it exists:** without it, every weight update would be exactly the size of the raw gradient,
which is usually far too large and causes wild, unstable training.

**Values actually used in this project:**
| Context | LR | Why |
|---|---:|---|
| Baseline (MobileNetV2), matching the paper | 1e-3 | Paper's own recipe (§4.2.2), full network trainable from a reasonable starting point |
| Encoder-swap experiments (ConvNeXt, EfficientNet, CLIP frozen/last-block, DINOv2 frozen/last-block) | 1e-4 | Gentler — protects pretrained foundation-model weights from being destabilized |
| DINOv2 full fine-tune | 1e-5, then 1e-6 | Even gentler for the *encoder* specifically, since the whole 21M-parameter network is unlocked |
| Stronger EfficientNet-B0 recipe attempt | 1e-3, cosine-annealed | Higher LR + a decay schedule — tested to see if it could beat the baseline; it made results worse (0.934 cm) |

**What happens if it changes (both directions):** too high → loss oscillates or diverges, weights
can blow up (seen in the failed "strong" EfficientNet recipe); too low → training crawls, may not
finish improving within the epoch budget, or can get stuck in a poor local region (a plausible
explanation for DINOv2's LR-1e-6 full fine-tune being the worst result in the whole project).

---

## 3. Epoch — walked through in full detail (as requested)

**What it is:** one complete pass through every example in the training set.

**Exactly what happens during ONE epoch (the real loop, from `train_baseline.py`):**
1. **Shuffle** the training data (a fresh random order each epoch, so the model doesn't learn any
   accidental pattern from data ordering).
2. Split into **mini-batches** (e.g. 32 fish per batch for the baseline).
3. For each mini-batch:
   a. **Forward pass:** feed the batch (images + bboxes) through the model → get predictions.
   b. **Compute the loss:** compare predictions to the true lengths (L1 loss — see §11).
   c. **Backward pass (backpropagation):** compute how much each of the model's millions of weights
      contributed to that error (the gradient of the loss with respect to every weight).
   d. **Optimizer step:** Adam uses those gradients to actually update every trainable weight a
      small amount (scaled by the learning rate).
4. After ALL mini-batches are done (one full sweep through training data), **switch to
   evaluation mode** and run the model over the **validation set** (no weight updates here — just
   scoring).
5. **Checkpoint saving:** always overwrite `last.pt`; overwrite `best.pt` only if this epoch's
   validation score is the best seen so far.
6. **Repeat** from step 1 for the next epoch, up to the configured total (`epochs` in the config).

**Values used:** baseline = 200 epochs; encoder-swap experiments = 100 epochs; DINOv2 last-block
fine-tune = 50 epochs (a shorter, targeted adaptation, since it starts from an already-good frozen
checkpoint); stronger EfficientNet-B0 recipes = 150–200 epochs.

**Why more epochs isn't automatically better:** each epoch is another chance to improve, but also
another chance to overfit (memorize training-specific quirks instead of general patterns) — this is
exactly what likely happened when the stronger EfficientNet-B0 recipe used 200 epochs instead of 100
and got a worse result.

---

## 4. Batch size

**What it is (layman):** how many fish the model looks at before making one correction, rather than
correcting after every single fish.

**What it is (technical):** the number of samples averaged into a single gradient estimate before
one optimizer step. Larger batches give a smoother, less noisy gradient estimate (more fish
"voting" on the correction) but need more GPU memory and take fewer, larger steps per epoch;
smaller batches are noisier but update more frequently.

**Values used:** baseline = 32; encoder-swap experiments = 16; DINOv2 last-block = 8 (smaller batch,
likely due to memory constraints when part of a large ViT is unfrozen); stronger recipes tried both
16 and 32.

---

## 5. Weight decay

**What it is (layman):** a gentle, constant pull on every weight toward zero, applied every single
update — like a mild "spring" resisting weights from growing too large.

**What it is (technical):** L2 regularization added directly into the optimizer's update rule
(`torch.optim.Adam(..., weight_decay=wd)`); it penalizes large weight magnitudes, which tends to
reduce overfitting by discouraging the model from relying too heavily on any single weight.

**Values used:** 0 (off) for every original experiment (baseline, encoder swaps); only the
**stronger-recipe search** configs introduced non-zero weight decay (1e-4 or 5e-5), alongside the
cosine schedule — an attempt at more disciplined training, which for EfficientNet-B0 did not improve
results.

---

## 6. Cosine learning-rate schedule

**What it is (layman):** instead of using the same step size for every epoch, gradually shrink the
step size following a smooth curve, so training takes big confident steps early and small careful
steps near the end.

**What it is (technical):** `torch.optim.lr_scheduler.CosineAnnealingLR` — the learning rate follows
one half-cycle of a cosine curve, from the starting LR down to (near) zero over `T_max` epochs.

**Why it exists:** the idea (well-established in deep learning practice) is that early training
benefits from larger updates to move quickly toward a good region, while late training benefits from
smaller updates to settle precisely rather than bouncing around a good solution.

**Used where:** only in the "stronger recipe" experiments for EfficientNet-B0/ConvNeXt (not in any
of the original baseline or encoder-swap runs, which used a constant LR throughout).

---

## 7. Gradient & backpropagation

**Gradient (layman):** a measure of "if I nudge this one weight slightly, how much does the error
change, and in which direction?" — computed for every weight in the network.

**Gradient (technical):** the partial derivative of the loss with respect to each parameter.

**Backpropagation (layman):** the algorithm that efficiently computes ALL of those millions of
gradients in one backward sweep through the network, by applying the chain rule layer by layer, from
the output back to the input.

**Backpropagation (technical):** `loss.backward()` in PyTorch triggers automatic differentiation
through the computational graph built during the forward pass, accumulating `.grad` on every
trainable tensor.

**Why it exists:** without an efficient method to compute gradients, training a network with millions
of parameters would be computationally infeasible (you'd need to test each weight individually).

---

## 8. Checkpoint, `best.pt`, `last.pt`

**What a checkpoint is:** a saved snapshot of every one of the model's current weight values,
written to disk (`torch.save(model.state_dict(), path)`).

**`last.pt`:** overwritten after *every single epoch* — always the most recent state, useful for
resuming an interrupted run.

**`best.pt`:** overwritten *only* when the current epoch's validation score beats every previous
epoch's validation score.

**Why this distinction matters — the core scientific-integrity mechanism of the whole project:**
the checkpoint that ultimately gets scored on the TEST set (by `evaluate.py`, run once, separately)
is always `best.pt` — chosen purely by looking at VALIDATION performance during training. The test
set is never consulted while deciding which checkpoint to keep. This is what makes "0.771 cm" a
legitimate, un-cherry-picked number rather than the best of many peeks at the test set.

---

## 9. Adam optimizer

**What it is (layman):** the specific rulebook the model uses to turn "how wrong was I, and in what
direction" (the gradient) into an actual weight update — smarter than just "subtract the gradient,"
because it remembers useful history from previous steps.

**What it is (technical):** Adaptive Moment Estimation. Maintains a running average of past
gradients (momentum, like a ball rolling downhill retaining some speed) **and** a running average of
past *squared* gradients (used to adaptively scale the step size per-parameter — parameters with
consistently large gradients get smaller effective steps, and vice versa). This combination usually
converges faster and more robustly than plain gradient descent, which is why it's the default choice
across essentially this entire project (and the paper's own REG recipe).

**AdamW** (used by the paper for Mask2Former, not used anywhere in our project): a variant that
decouples weight decay from the gradient-based update, generally considered a cleaner implementation
of weight decay than plain Adam's — worth knowing exists, since the paper itself uses it for a
different sub-task (segmentation) than the one we reproduce (regression, which uses plain Adam).

---

## 10. L1 loss (and why it's not the same as MAE, technically, though numerically related)

**What it is (layman):** during training, "how wrong was this one guess?" measured as a plain
absolute difference — no squaring, no fancy weighting.

**What it is (technical):** `L1Loss = mean(|prediction - target|)` averaged over a batch — this is
what the `criterion` computes and what `.backward()` differentiates.

**L1 loss vs. MAE — the important distinction for a viva:** they are the **same formula**, but
different *roles*. L1 loss is used **during training**, per batch, to drive gradient updates. MAE
(computed by `metrics.py`'s `regression_metrics()`) is used **during evaluation**, over an entire
split, purely to *report* a number — it never produces a gradient or updates any weight. It's
correct to say "we train with L1 loss and report MAE," even though the arithmetic is identical,
because the *purpose and context* differ.

**Why L1 over alternatives (e.g. MSE/L2 loss):** L1 loss penalizes errors linearly (a 4 cm error
counts exactly 4× as much as a 1 cm error), whereas L2/MSE loss squares errors (a 4 cm error counts
16× as much as a 1 cm error) — L1 is more robust to occasional large outlier errors dominating
training, which matters here since occluded fish can have larger, noisier errors.

---

## 11. Seed, reproducibility, and why only DINOv2 got multiple seeds

**What a seed is (layman):** a fixed starting point for all the "randomness" in training (data
shuffle order, initial random weights, augmentation randomness) — using the same seed twice
reproduces the exact same run.

**What it is (technical):** `seed_everything(seed)` fixes Python's `random`, NumPy's RNG, and both
PyTorch's CPU and CUDA random number generators to the same state.

**Why 42 specifically:** no technical significance — it's simply this project's chosen fixed
default (a very common convention in ML code, referencing the "Hitchhiker's Guide to the Galaxy"
joke; not chosen for any mathematical property).

**Why multiple seeds matter:** neural network training has genuine run-to-run randomness even with
identical hyperparameters (different random weight initialization, different data shuffle order).
A single run's result could be somewhat lucky or unlucky. Running the *same* config with different
seeds (e.g. 42, 1, 2) and looking at the **spread** of results tells you whether a finding is
reliable (small spread, or non-overlapping vs. a competitor) or noise (large spread).

**Why ONLY the DINOv2 patch-vs-CLS study used 3 seeds, and not everything else:** an honest,
disclosed limitation of scope/time, not a deliberate choice to hide uncertainty elsewhere. The
patch-vs-CLS question was identified as the project's most important controlled finding and given
the multi-seed treatment it needed to be credible; other comparisons (the baseline vs. EfficientNet
gap, for instance) remain single-seed and are explicitly flagged as unconfirmed throughout this
project's documentation.

---

## 12. Validation vs. test — precisely

**Validation set:** held-out data used **during** training, checked after every epoch, to decide
which checkpoint (`best.pt`) to keep. The model's *weights* are never directly trained on it, but it
*does* influence which checkpoint gets chosen — so it's not perfectly "unseen" in a strict sense.

**Test set:** held-out data used **exactly once**, at the very end, by a separate script
(`evaluate.py`), after all training and checkpoint-selection decisions are already locked in. This
is the number that gets reported as the project's actual result.

**Why the distinction matters:** if you tuned anything (hyperparameters, checkpoint choice, even
which model architecture "won") by repeatedly looking at test performance, your test number would no
longer be an honest estimate of real-world performance — it would be *fit* to the test set, a problem
called "test-set leakage through repeated peeking." This project's discipline (validation for all
decisions, test touched once) is a direct defense against that.

---

## 13. Frozen encoder vs. fine-tuning vs. full fine-tuning — the adaptation spectrum

| Level | What's trainable | Used for |
|---|---|---|
| **Frozen** | Only the small head | Testing "how good are the pretrained features, as-is?" — DINOv2/CLIP frozen runs |
| **Last-block fine-tuning** | Head + only the final transformer block/layer of the encoder | A middle ground — DINOv2/CLIP last-block runs |
| **Full fine-tuning** | Every weight in the whole model | Baseline, ConvNeXt, EfficientNet-B0 (all fully trainable from the start), and DINOv2's full-FT experiments |

**Why frozen exists as an option at all:** it's the fairest way to ask "is this pretrained
representation, by itself, useful for my task?" — full fine-tuning can mask a weak pretrained
representation by simply retraining around it, conflating "how good was the pretraining" with "how
much adaptation capacity did we allow."

---

## 14. Normalization (ImageNet statistics) and ImageNet-pretrained weights

**Normalization (layman):** rescaling pixel colour values into a standard numeric range the
pretrained network expects, rather than the raw 0–255 (or 0.0–1.0) range a fresh image loads with.

**Normalization (technical):** `transforms.Normalize(mean, std)` subtracts a fixed per-channel mean
and divides by a fixed per-channel standard deviation — the *same* mean/std originally used when
each pretrained encoder was trained on ImageNet. Skipping this, or using different values, would feed
the pretrained weights input statistics they've never seen, badly damaging transfer learning.

**ImageNet-pretrained weights:** every CNN in this project (MobileNetV2, EfficientNet-B0, ConvNeXt)
starts from weights already trained on ImageNet (1.2M labeled images, 1000 object classes) rather
than random initialization. This gives the network a huge head start — general edge/texture/shape
detectors are already present before any fish-specific training begins. DINOv2 and CLIP are
*not* ImageNet-pretrained specifically — they use their own, much larger and differently-sourced
pretraining data (see Part 3).

---

## 15. Quick-fire viva Q&A — Part 4 (hyperparameters)

**Q1: What's the difference between a parameter and a hyperparameter, with an example from this project?**
A parameter is a learned weight inside MobileNetV2 (there are millions); a hyperparameter is
something we set beforehand, like the learning rate 1e-3 — the model never changes it during training.

**Q2: Why does the baseline use LR 1e-3 but the encoder-swap experiments use 1e-4?**
The baseline matches the paper's own recipe exactly; the swap experiments use a gentler, uniform
rate because pretrained foundation-model encoders can be destabilized by an aggressive learning rate.

**Q3: Walk me through exactly what happens in one training epoch.**
Shuffle data → split into mini-batches → for each batch: forward pass, compute L1 loss, backward
pass (compute gradients), Adam updates weights → after all batches, evaluate on validation (no
weight updates) → save `last.pt` always, `best.pt` only if validation improved → repeat.

**Q4: What's the difference between `last.pt` and `best.pt`, and why keep both?**
`last.pt` is always the most recent state (useful to resume a crashed run); `best.pt` is the
checkpoint with the best validation score seen so far — this is the ONE we ever evaluate on test.

**Q5: Why is checkpoint selection based on validation, never test?**
To guarantee the reported test number isn't the result of repeatedly peeking at test performance
and cherry-picking — that would invalidate it as an honest estimate.

**Q6: What is weight decay, and where was it used?**
A constant pull on every weight toward zero each update, discouraging overly large weights; used
only in the "stronger recipe" search for EfficientNet-B0/ConvNeXt (1e-4 or 5e-5), not in any
original experiment.

**Q7: Explain the cosine learning-rate schedule.**
The learning rate follows a cosine curve from its starting value down toward zero over training —
big steps early, small careful steps late. Used only in the stronger-recipe experiments.

**Q8: What is Adam, and why is it preferred over plain gradient descent here?**
An optimizer combining momentum (remembering past gradient direction) with per-parameter adaptive
step sizes (based on past squared gradients) — generally converges faster and more robustly than
vanilla gradient descent, which is why it's used throughout this project and by the paper's own recipe.

**Q9: Is L1 loss the same thing as MAE?**
Same formula (mean absolute difference), different role: L1 loss is used during training to produce
gradients; MAE is used during evaluation purely to report a number, with no gradient involved.

**Q10: Why does this project use seed 42, and does that number mean anything mathematically?**
It's simply this project's fixed default value for reproducibility — no special mathematical
property, a common convention in ML code.

**Q11: Why does only the DINOv2 patch-vs-CLS study use multiple seeds?**
An honest, disclosed limitation of time/scope — that comparison was judged the project's most
important controlled claim and given the rigor it needed; other comparisons remain single-seed and
are explicitly flagged as unconfirmed.

**Q12: What's the difference between validation and test, precisely?**
Validation is checked repeatedly during training and influences which checkpoint gets kept; test is
touched exactly once, after every decision is already locked in, and is the number actually reported.

**Q13: What does "frozen encoder" mean, and why test it at all instead of always fine-tuning?**
The encoder's weights are locked; only the head trains. It's the fairest test of "how good are the
pretrained features alone?" — full fine-tuning can mask weak pretrained features by retraining
around them.

**Q14: Why must normalization use ImageNet's specific mean/std, not just any values?**
Because the pretrained encoders were themselves trained on images normalized that exact way —
mismatched statistics would feed the pretrained weights inputs far outside what they were calibrated
for, badly damaging transfer learning.

**Q15: If you doubled the number of epochs for every experiment, what would you expect to happen?**
Not automatically better results — more epochs mean more chances to overfit as well as more chances
to improve; this project directly observed this with EfficientNet-B0 (200 epochs performed worse
than 100).

---

*Part 4 of the staged viva-prep series (54 Q&A total so far across Parts 1–4). Part 5 (poster
walkthrough) continues next.*
