# First Meeting Briefing
## Seminar: Fish Length Estimation with Vision Foundation Models
### Deep Learning for Maritime Vision Applications — Supervisor: Bohan Zhuang

---

## How to use this document

This is the script for your **first supervisor meeting**. The professor will essentially ask: *"You've had one to two weeks — what have you done? What do you understand? Where are you going?"*

The goal of this meeting is **not** to commit to a final method. The goal is to show that you have read the task carefully, that you understand the domain, the dataset, and the methods involved, and that you have a sensible direction in mind. You want to leave the meeting with a few clarification answers from the professor that will shape the next two weeks.

Structure the meeting like this:

1. Restate the task in your own words (shows you understood)
2. Walk through the domain and key concepts (shows you did the reading)
3. Explain the dataset (shows you looked at the data, not just the task PDF)
4. Sketch a possible high-level direction (without locking in details)
5. Ask clarifying questions
6. Propose what you'll do in weeks 3–4 and when to meet next

Plan for 20–30 minutes. Let the professor steer; don't monologue.

---

## 1. The task in plain language

Start with the layman version, then layer in the technical detail. The professor knows the task already — this is for *you* to demonstrate understanding, not to inform them.

**Layman version:**

> Fishermen and marine biologists today measure fish by hand — pulling each fish out of a catch, laying it on a ruler, and writing down the length. This is slow, repetitive, and expensive. If we could take a photo of fish on a conveyor belt and have a computer estimate each fish's length automatically, fisheries could be monitored at a much larger scale, which matters for fighting overfishing and managing fish stocks sustainably.
>
> The question this seminar asks is: there's an existing computer-vision method that does this reasonably well — a small neural network trained to predict fish length from cropped images. Can we make it better by replacing the "eye" of that network with a much more powerful, modern image-understanding model? And does that help especially when we don't have many labeled examples to train on?

**Technical version:**

> The task is a regression problem: given a cropped RGB image of a single fish, predict a single scalar — its real-world length in centimeters. The published baseline on the AutoFish dataset is a MobileNetV2-based regressor reaching 0.62 cm mean absolute error on non-occluded fish and 1.38 cm on occluded ones. We are asked to (a) reproduce this baseline, (b) build a variant where the MobileNetV2 encoder is replaced by a Vision Foundation Model — a large pretrained encoder such as DINOv2 or SAM — keeping everything else comparable, (c) evaluate both under reduced-data conditions, and (d) analyze failure modes. An optional extension tests transfer to a second fish dataset if it becomes available.

---

## 2. Key concepts and terms you should be comfortable with

Before the meeting, make sure you can explain each of these in two sentences. The professor will probably probe one or two.

### Regression vs. classification
A neural network usually predicts a category (cat / dog / fish). **Regression** predicts a continuous number instead — here, a length in cm. The network architecture is mostly the same; only the last layer and the loss function change.

### Encoder + head
Modern vision networks have two parts. The **encoder** (or backbone) turns an image into a compact feature vector — a list of numbers summarizing what's in the image. The **head** is a small network on top that turns those features into the final prediction. The whole project is about swapping the encoder while leaving the head approximately the same.

### Pretraining and transfer learning
Training a network from scratch needs huge amounts of data. Instead, we usually start from a network that was already trained on a large general-purpose dataset (like ImageNet) and adapt it to our task. This is **transfer learning**. The starting network is **pretrained**.

### Vision Foundation Model (VFM)
A relatively new term for a particular kind of pretrained encoder: very large, trained on huge amounts of image data using **self-supervised** objectives (i.e. without any human labels), and producing features that transfer well to many downstream tasks. Examples: DINOv2 (Meta, 2023), DINOv3 (2025), SAM and SAM 2 (the encoders inside the Segment Anything models), MAE, CLIP. The key empirical claim about VFMs is that their features are so rich that you can often get strong results by **freezing** the encoder entirely and training only a small head on top.

### Why VFMs might help when labels are scarce
Because the encoder already encodes a lot about images from its massive pretraining, you don't need to teach it from scratch — you only need to teach the small head how to map features to length. With less data, this is much easier than fitting a large model end-to-end. That is precisely why the task asks about reduced-data settings.

### Instance segmentation mask
A pixel-level outline of one specific object. In AutoFish, every fish in every image has a mask telling you exactly which pixels belong to that fish. We use these masks to **crop** each fish out of the full image before feeding it to the length-estimation network.

### Occlusion
When one fish partially covers another. AutoFish includes images where fish are isolated (easy) and images where they're piled up and overlapping (hard). Length estimation is much harder under occlusion because parts of the fish are hidden.

### MAE (mean absolute error)
The natural metric here: the average of |predicted length − true length| across all fish. Reported in centimeters. Lower is better. Easy to explain in a paragraph: "on average we're off by 1.2 cm."

---

## 3. What I've read and understood (talking points)

This is the bulk of the meeting. Walk the professor through what you've absorbed. Be specific — vague claims like "I read the paper" mean nothing; specific facts demonstrate that you actually engaged with the material.

### About the AutoFish dataset (Bengtson et al., WACVW 2025)

- It contains **1,500 RGB images** of fish on a white conveyor belt, captured in a controlled lab setting (camera 1.5 m above the belt).
- **454 unique fish specimens**, organized into **25 groups**. Each group contains 14–24 different fish. Crucially, **each fish appears in only one group**.
- Each group is captured in three subsets:
  - **Set1**: half the fish in the group, isolated (no occlusion), 20 images.
  - **Set2**: the other half, also isolated, 20 images.
  - **All**: every fish in the group placed together with deliberate occlusion, 20 images.
- This gives ~60 images × 25 groups = 1,500 total.
- Species: mainly **cod, haddock, whiting, hake, and horse mackerel** — all visually similar, all economically important.
- **Length labels** were measured manually by a marine biologist and **rounded to the nearest 5 mm**. That rounding is important: no model can be more accurate than the labels.
- Every fish instance has a **pixel-level segmentation mask** in addition to the length label.

**Why the group structure matters:** if we just split images randomly into train/test, the same physical fish would appear on both sides. The model could memorize individual fish instead of learning to measure them. The group structure is the dataset authors' deliberate way of preventing this — we must split at the **group** level, not the image level. This is the single most important point about the dataset and worth raising with the professor explicitly.

### About the baseline method

The AutoFish paper reports two length-estimation baselines:

1. **Skeletonization (classical):** take the fish's segmentation mask, reduce it to a 1-pixel-wide skeleton, measure its pixel length, multiply by a known pixel-to-cm conversion. No learning involved. Works because the camera setup is calibrated and fixed.
2. **CNN regression (the one we'll focus on):** crop the fish from the bounding box of its segmentation mask, feed it to a MobileNetV2 backbone, append a small regression head, train it to output length in cm. Reports MAE of **0.62 cm on non-occluded fish, 1.38 cm on occluded fish**. This is "the baseline" the task description refers to.

Note that the CNN learns the pixel-to-cm conversion **implicitly** — it never sees the calibration directly. This works because the camera geometry is the same across the whole dataset.

### About Vision Foundation Models

The task says to replace the encoder with a VFM. The main candidates, roughly in order of how well-suited they seem for this problem:

- **DINOv2 / DINOv3** (Meta, 2023/2025). Self-supervised ViTs trained on 142M curated images. Features known to transfer well to fine-grained tasks and dense prediction. The community's current default "first choice" for a VFM encoder.
- **SAM / SAM 2 image encoder** (Meta, 2023/2024). The encoder inside the Segment Anything Model, trained on more than a billion masks. Features are very shape-aware, which intuitively suits a measurement task.
- **MAE, CLIP, others.** MAE is solid but generally beaten by DINOv2 on dense tasks. CLIP features are more semantic than geometric and probably less well-suited to a measurement problem.

You don't need to commit to one in the first meeting. Just demonstrate you understand the landscape. The professor may have a specific preference and this is a good thing to ask about (see Section 6).

### What the research question is really asking

Re-read the wording carefully:

> *"How well does an established deep learning baseline perform for fish length estimation, and does replacing its encoder with a Vision Foundation Model (VFM) improve performance, in particular when labeled data are limited?"*

There are really three sub-questions:

1. Can I reproduce the published baseline? (Sanity check.)
2. Does swapping the encoder for a VFM help on the full dataset? (Main comparison.)
3. Does the VFM help **more** in low-data settings? (The "in particular" clause — this is the most interesting and most publishable result.)

The optional extension question asks whether the VFM-based approach also generalizes to a second, different fish dataset, but that depends on data availability.

---

## 4. A sketch of where this is heading (without locking in details)

Be deliberate here. You want to show you can imagine the project end-to-end, but you do **not** want to commit to a specific architecture, specific hyperparameters, or a specific train/test split until you have agreement from the professor.

Phrase this section as "what I think a reasonable plan looks like, subject to your input." The plan has roughly four phases:

1. **Data exploration and infrastructure.** Get the AutoFish dataset, look at examples, understand the annotation format, decide on a deterministic group-level split for train/validation/test, and write the code to extract per-fish crops. End state: a clean dataset of fish crops with length labels, ready to feed any model.

2. **Baseline reproduction.** Re-implement the MobileNetV2 regression model as described in the paper. Verify the MAE I get is close to the published numbers. This is non-negotiable — without a credible baseline, the comparison in step 3 means nothing.

3. **VFM variant.** Take one VFM (probably DINOv2 to start, depending on the professor's input), plug it in as the encoder, keep everything else comparable, train, evaluate. Then run both models under reduced-data conditions to answer the "in particular when labels are limited" part.

4. **Analysis and reporting.** Look at where each model fails — by species, by occlusion level, by fish size — and write up the comparison.

Optionally, if the North Sea dataset arrives, test transfer.

I have **not** committed to: input resolution, augmentation policy, exact split ratios, whether to fine-tune the VFM or keep it frozen, whether to also try SAM, how many seeds to run. Those are decisions for after the next 1–2 weeks of getting the basic infrastructure working.

---

## 5. Questions for the professor (write these on paper, bring them)

Asking good questions in the first meeting is worth as much as showing what you've read. It signals that you're thinking ahead and that you respect the professor's time. Pick maybe four to six of these — don't ask all of them. Pick the ones most relevant to the points above.

**About scope**

1. The task mentions "an established deep learning baseline." Do you specifically mean the MobileNetV2 regression model from the AutoFish paper, or is there flexibility there?
2. For the VFM, do you have a preference — DINOv2, SAM, something else? Or is the choice mine to justify?
3. How important is the optional second-dataset extension for the final grade? Should I plan around it from the start, or treat it as a bonus if time allows?

**About methodology**

4. For the reduced-data experiments, what scale do you have in mind — fractions like 50%, 25%, 10% of training data? Or specific absolute numbers of training examples?
5. Should I use ground-truth segmentation masks for cropping at evaluation time, or run a separate segmentation model in the pipeline? The paper does the former; I'd default to that unless you'd prefer end-to-end.
6. For the failure analysis, do you want it split by species, by occlusion, by length range, or by something else?

**About logistics**

7. What compute resources do you expect me to have access to? Is there a group GPU, or am I working off my own machine?
8. Are there code or model conventions you'd like me to follow (a specific framework, a specific logging tool, code style)?
9. What's the expected length and format of the final report and presentation?
10. How frequently would you like to meet? Every two weeks? Monthly?

**About the data**

11. Is there any update on the North Sea dataset? Roughly when might it become available?
12. Are there students who have worked on related projects whose code or notes I could see?

---

## 6. Closing the meeting — what to commit to for next time

End with a concrete plan for the next 1–2 weeks. Something like:

> *"For our next meeting, I'd like to come back having: (1) downloaded and explored the AutoFish dataset, (2) implemented a fixed group-level train/val/test split, (3) generated per-fish crops, and (4) started training a first version of the MobileNetV2 baseline so we can compare against the published numbers. Does that sound like a reasonable scope for two weeks?"*

This is a small, achievable, **verifiable** chunk of work. Verifiable matters — at the next meeting you can point at a number and a plot, not just at code that runs.

Suggest a next-meeting date before leaving.

---

## 7. Quick-reference cheatsheet — bring this on paper

| Term | One-line answer |
|---|---|
| Task | Predict fish length in cm from a cropped RGB image; regression problem. |
| Dataset | AutoFish: 1,500 images, 454 fish, 25 groups, conveyor-belt setup, lengths to 5 mm. |
| Baseline | MobileNetV2 encoder + small regression head; published MAE 0.62 / 1.38 cm. |
| Proposed change | Replace MobileNetV2 with a Vision Foundation Model (e.g. DINOv2). |
| Why a VFM might help | Pretrained on huge data; features transfer well; less reliance on local labels. |
| Key experiment | Compare both under reduced training-data conditions. |
| Critical pitfall | Group-level split, never image-level — otherwise the same fish leaks across train/test. |
| Metric | MAE in cm, reported separately for non-occluded and occluded fish. |
| Honest unknowns | Which VFM, whether to fine-tune it, exact data fractions — to be decided with supervisor input. |

---

## 8. If the professor asks something you don't know

Three rules:

1. **Don't bluff.** Say "I don't know yet, that's something I want to think about / read about before next time." This is far better than guessing wrong and being caught.
2. **Don't apologize excessively.** One acknowledgment is enough; then move on.
3. **Write the question down.** Visibly. The professor notices, and you can come back to it at the next meeting having actually thought about it.

Good luck. The biggest thing the professor wants to see in a first meeting is that you take the work seriously and you've engaged with the material — both of which you can demonstrate easily with the points above.
