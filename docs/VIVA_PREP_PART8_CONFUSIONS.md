# Viva Prep — Part 8: Common Confusions, Resolved

Builds on Parts 1–7. Each pair below is a distinction that's easy to blur under viva pressure. Each
gets a **one-line rule** you can say instantly, plus the fuller reasoning.

---

## 1. GT vs. PD

**One-line rule:** *GT = human-drawn perfect mask; PD = a model's own guessed mask. Our project
always uses GT.*

The paper evaluates both because they answer different questions: GT isolates "how good is the
length estimator alone?"; PD measures "how good is the whole real deployed pipeline, errors and
all?" Our project never runs a segmentation model, so every one of our numbers is implicitly a "GT"
setting — the correct paper baseline to cite against ours is REGgt (0.67/0.96/0.82 cm), not the
abstract's REGpd (0.62/1.38/0.99 cm). Full detail in Part 1 §6.

## 2. Bounding box vs. segmentation mask

**One-line rule:** *A box is a rectangle; a mask is the exact pixel-by-pixel outline.*

A bounding box (4 numbers: x, y, width, height) is a coarse rectangle around an object. A
segmentation mask is precise, pixel-level — it can represent a curved, bent, or partially-occluded
fish shape that no rectangle could. Our project **uses both, for different purposes**: the mask is
used to *cut out and black-background* the fish (`make_crops.py`); the bounding box is a *separate,
4-number numeric input* concatenated to the model's features, restoring scale information the crop
loses. They are not alternatives to each other in this project — they're used together.

## 3. Regression vs. classification

**One-line rule:** *Regression predicts a number; classification predicts a category.*

Length estimation = regression (continuous, e.g. 30.4 cm — could be any value). Species
identification = classification (discrete, one of 7 fixed categories). Same encoders, same
pipeline shape, different loss function (L1 vs. cross-entropy) and different head width (1 output
vs. 7 outputs) — see Part 3 §0 and Part 7 §6.

## 4. Encoder vs. head

**One-line rule:** *The encoder "sees"; the head "decides."*

The encoder (MobileNetV2, DINOv2, etc.) turns an image into a feature vector — general-purpose
"understanding." The head is a small, task-specific network (2–3 layers) that turns that feature
vector into the actual final answer. Only the encoder is swapped between experiments; the head
family (shape, activation pattern) stays structurally consistent.

## 5. Feature vector

**One-line rule:** *A feature vector is the encoder's numeric "description" of an image — not
human-readable, but structured so similar images produce similar vectors.*

E.g. MobileNetV2 produces a 1280-number vector per image; DINOv2 produces a 384-number vector. These
numbers don't individually mean anything interpretable (no single number is "length" or "color") —
the *pattern* across all of them is what the head learns to decode.

## 6. Loss vs. metric

**One-line rule:** *Loss drives training (needs a gradient); a metric is just for reporting (doesn't
need a gradient).*

L1 loss is used during training because PyTorch can differentiate it to get gradients. MAE
(computed after training, on held-out data, purely to report a number) uses the *exact same
formula* as L1 loss, but is never differentiated or used to update weights — see Part 4 §10 for the
full distinction.

## 7. Training vs. inference

**One-line rule:** *Training changes weights; inference (evaluation) only uses them.*

During training, every batch triggers `.backward()` and `optimizer.step()` (weights change). During
evaluation (`evaluate.py`, or the validation check inside training), the model is wrapped in
`@torch.no_grad()` and `model.eval()` — no weights are touched, only predictions are read out.

## 8. Parameters vs. hyperparameters

**One-line rule:** *Parameters are learned; hyperparameters are chosen.*

Covered fully in Part 4 §1 — a parameter is one of the millions of numbers inside the network's
layers, updated by gradient descent; a hyperparameter (learning rate, batch size, epochs) is set in
a config file before training starts and never changes on its own.

## 9. MAE vs. L1 loss

**One-line rule:** *Same formula, different job — L1 loss trains, MAE reports.*

See Part 4 §10 and confusion #6 above — this is one of the most likely "gotcha" viva questions
precisely because they're numerically identical but conceptually distinct in role.

## 10. Validation vs. test

**One-line rule:** *Validation is checked repeatedly during training; test is touched exactly once,
at the very end.*

Validation influences which checkpoint gets kept (an indirect form of "seeing" the data during
development). Test is only ever scored once, after every decision (architecture, hyperparameters,
checkpoint) is already locked in — see Part 4 §12.

## 11. Checkpoint vs. final model

**One-line rule:** *"Checkpoint" is the general term; `best.pt` is THE specific checkpoint that
counts as "the model."*

Every epoch produces a `last.pt` checkpoint (always overwritten); only the epoch with the best
validation score becomes `best.pt`, which is the only checkpoint ever evaluated on test or reported
as a result. When someone says "the trained MobileNetV2 model," they mean `runs/baseline_official/best.pt`.

## 12. Pretrained vs. fine-tuned

**One-line rule:** *Pretrained = knowledge from a different, much bigger dataset; fine-tuned =
adapted afterward on OUR fish data.*

Every encoder in this project starts pretrained (ImageNet for the CNNs, 400M image-text pairs for
CLIP, self-supervised web images for DINOv2) — none are trained from random weights. "Fine-tuned"
describes what happens *next*: further training on the AutoFish crops, at varying intensity (frozen
/ last-block / full — see confusion #13).

## 13. Frozen vs. trainable

**One-line rule:** *Frozen = locked, doesn't learn from our data at all; trainable = updates during
our training.*

A frozen encoder's `requires_grad=False` means it never changes during our training — the model only
learns via the small head on top. A trainable (fine-tuned) encoder does update. See Part 4 §13 for
the full frozen / last-block / full-fine-tune spectrum.

## 14. CLS token vs. patch tokens

**One-line rule:** *CLS = one global summary vector; patch tokens = many local vectors, one per
image region.*

This is THE central technical finding of the whole project (Part 3 §5) — patch tokens, averaged
together, preserve more spatial/size information than the single CLS summary, and this measurably
improved DINOv2's length-regression accuracy in a controlled, 3-seed experiment.

## 15. Mask vs. crop

**One-line rule:** *A mask says WHICH pixels are the fish; a crop is the smaller rectangular IMAGE
cut out afterward.*

The mask is used first (to blacken non-fish pixels); the crop is the physical cutting-out step that
happens after masking, based on a square window computed from the mask's extent. See Part 2 §3 for
the full pipeline order.

## 16. Bounding box normalization

**One-line rule:** *Dividing by the original photo's width/height turns pixel coordinates into a
universal 0–1 scale.*

`bbox_x / width` etc. — without this, a bbox from a 2464-pixel-wide photo and a bbox from a
differently-sized photo wouldn't be comparable numbers to the network; normalizing to 0–1 makes them
consistent regardless of the source image's exact resolution.

## 17. ImageNet normalization

**One-line rule:** *Rescaling pixel colors to match what the pretrained network was originally
trained on — using the SAME mean/std the original ImageNet training used.*

Not a general-purpose "make pixels nicer" step — the specific numbers (`normalize_mean`,
`normalize_std` in configs) must match what each pretrained encoder actually saw during its own
pretraining, or transfer learning suffers. See Part 4 §14.

## 18. ColorJitter

**One-line rule:** *Randomly wobbles brightness/contrast/saturation/hue, training-only, so the model
doesn't memorize exact lighting.*

Applied only when `augment=True` (training split); never for validation/test, so evaluation is
always on the real, unperturbed image (Part 2 §3, Part 4).

## 19. Cosine scheduler

**One-line rule:** *Shrinks the learning rate smoothly over training — big steps early, tiny steps
late.*

Only used in the "stronger recipe" experiments for EfficientNet-B0/ConvNeXt; every original
experiment used a constant learning rate throughout. See Part 4 §6.

## 20. Weight decay

**One-line rule:** *A constant, gentle pull on every weight toward zero, each update — discourages
overly large weights.*

Zero (off) in every original experiment; only introduced in the stronger-recipe search, alongside
the cosine schedule. See Part 4 §5.

## 21. Why 224×224?

**One-line rule:** *It's the standard input size every pretrained encoder in this project expects,
inherited from ImageNet-era convention.*

Not an arbitrary choice — MobileNetV2/EfficientNet/ConvNeXt's ImageNet-pretrained weights, and
DINOv2/CLIP's own pretraining, were built around this (or a compatible) input resolution; using a
different size would either be rejected by the architecture or require resizing internally anyway.

## 22. Why concatenate bbox instead of, say, adding it or using it separately?

**One-line rule:** *Concatenation keeps the bbox as independent, uncorrupted information alongside
the image features, rather than blending them.*

`torch.cat([features, bbox], dim=1)` places the 4 bbox numbers as additional entries in the same
vector, which the head's first linear layer then learns to weigh appropriately — this is simpler and
more standard than, e.g., trying to inject the bbox earlier inside the encoder's convolutional
layers (which weren't designed to accept extra non-image inputs).

## 23. Why EfficientNet-B0 nearly matched MobileNetV2

**One-line rule:** *Architectural family similarity — both are compact, fully-supervised,
ImageNet-pretrained CNNs — but the reason is a hypothesis, not proven, and the result is
single-seed/unconfirmed.*

See Part 3 §2 and Part 6 §2 — always pair this claim with the reliability caveat.

## 24. Why foundation models underperformed here specifically

**One-line rule:** *Their pretraining objectives (semantic matching / self-supervised similarity)
don't explicitly reward precise geometric/metric information the way supervised ImageNet
classification incidentally does.*

Also a hypothesis — supported indirectly by the species-classification cross-check (Part 6 §13),
where the same foundation models perform much better, consistent with a "good at semantics, weaker
at precise geometry" story.

## 25. Why compare against the paper's PUBLISHED baseline rather than retraining segmentation ourselves

**One-line rule:** *Scope control — our research question is about encoder comparison for the
length-regression step specifically, not about rebuilding the whole pipeline.*

Retraining Mask2Former would be a substantial separate project; using the paper's own published
numbers (and specifically REGgt, since we match their evaluation conditions) is the standard,
accepted way to build on prior published work without redoing everything from scratch.

## 26. Why this project is "reproduction PLUS extension," not a full AutoFish reproduction

**One-line rule:** *We reproduce exactly one of the paper's two baseline methods (REG), and add
research questions the paper never asks.*

Full accounting in Part 2 §2 — segmentation and skeletonization were never touched; the encoder
comparison, patch-token study, and species-classification task are entirely our own additions.

---

## Quick-fire viva Q&A — Part 8 (confusions)

**Q1: In one sentence, what's the difference between GT and PD?**
GT is a human-verified mask; PD is a mask guessed by a trained segmentation model — our project
always uses GT.

**Q2: Is a bounding box the same as a segmentation mask?**
No — a box is a coarse rectangle; a mask is the pixel-precise outline. This project uses both
together, for different purposes.

**Q3: Are MAE and L1 loss mathematically different?**
No, identical formula — the difference is purely in role: L1 loss drives training gradients, MAE is
a post-hoc reporting metric with no gradient involved.

**Q4: What's the difference between a checkpoint and "the model"?**
Every epoch saves a checkpoint (`last.pt`, always overwritten); only the single best-on-validation
epoch becomes `best.pt`, which is what anyone means by "the trained model" in this project.

**Q5: Is "pretrained" the same as "fine-tuned"?**
No — pretrained describes where an encoder's starting knowledge came from (a different, larger
dataset); fine-tuned describes what happened afterward, on our specific fish data, at some level of
intensity (frozen/partial/full).

**Q6: Why does the CLS-vs-patch-token distinction matter so much for this specific project?**
Because it's the one place where a single, well-understood architectural choice (which tokens to
read from a ViT) produced a large, statistically confirmed change in results — a clean, teachable
example of "how a model reads an image" mattering as much as "which model" is used.

---

*Part 8 of the staged viva-prep series (82 Q&A total so far across Parts 1–8). Part 9 (the
consolidated growing question bank) is next — I'll compile all Q&A from Parts 1–8 into one master
list and continue adding until we reach the 150+ target.*
