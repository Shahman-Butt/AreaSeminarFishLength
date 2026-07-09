# Understanding Your Seminar Task — From the Ground Up

A complete, plain-language walkthrough of the fish length estimation project. Read this before anything else. Once you've gone through it, the technical plan in the other documents will make much more sense.

---

## Part 1: The Real-World Problem

### Why do fish need to be measured at all?

Imagine a commercial fishing boat coming back to port with a giant catch — thousands of fish in crates. Before any of those fish can be sold, somebody has to record what's there: how many fish, of what species, and how big each one is.

Length matters more than you'd think. Many fish species have legal minimum sizes — if a cod is below, say, 35 cm, you're not allowed to keep it because it's too young to have reproduced. Scientists also use the length distribution of a catch to estimate the health of fish populations: if fishermen are bringing in lots of small fish and almost no big ones, that's a warning sign that the population is being overfished.

Right now, measuring fish is mostly done by hand. A person picks up each fish, places it on a measuring board, reads off the length, writes it down. For a single boat with a few tonnes of catch, this can take hours. For an entire industry, it's enormous amounts of human labor. And in scientific surveys — where biologists go out specifically to sample fish populations — it's the same story.

So: **if a computer could look at a photo of fish on a conveyor belt and automatically tell you how long each one is, you could save huge amounts of time and money, and you could monitor fishing at much larger scales than is currently possible.** That's the practical motivation behind this whole project.

### Why is this hard for a computer?

You might think this is easy — surely you just count pixels along the fish? In a perfect world with a fixed camera, yes. But the real problems are:

- Fish overlap each other on a conveyor belt. Parts of a fish are hidden behind other fish.
- Fish bend and curl, so a straight pixel measurement underestimates their true length.
- A fish photographed from above looks different depending on whether it's lying flat or slightly turned.
- Different species have different body shapes, so "the long axis" isn't always obvious.
- You need to convert pixels (in the image) to centimeters (in the real world), which depends on the camera setup.

Computer vision researchers have been working on this for years. The newest approach — which is what your project is about — uses **deep learning**.

---

## Part 2: What Deep Learning Means Here

### What's a neural network, really?

A neural network is a big mathematical function. You feed it some input (in our case, an image), it does many, many simple arithmetic operations, and it produces some output (in our case, a single number: the length of the fish).

The crucial property is that the function has **parameters** — millions of them — that start out as random numbers. By showing the network many examples of (image, correct length) pairs and slightly adjusting the parameters each time so the prediction gets a bit closer to the correct answer, eventually the network learns to make good predictions on new images it has never seen.

This adjustment process is called **training**. The examples used for training are called the **training set**. The (image, correct length) pairs are called **labels** — somebody had to manually measure those fish so the computer can learn from them.

### Classification vs. regression — which one is this?

Most neural networks you've heard of do **classification**: given an image, pick one of a fixed set of categories (cat, dog, fish). Output is a category.

This project is different. The output is a **continuous number** — a length in centimeters, anywhere from maybe 10 cm to 60 cm. That's called **regression**. The architecture of the network is mostly the same as a classifier; only the very last layer and the way you score errors are different.

In a classifier, you score errors by checking "did the network pick the right category?" In a regressor, you score them by checking "how close was the predicted number to the true number?" That distance — the average error in cm — is the main thing we'll be measuring throughout this project.

### Encoder and head — the two halves of a vision network

Almost every modern vision neural network is structured the same way: two parts stacked together.

**Part 1: the encoder (also called the backbone).** This is the big, heavy part of the network. It takes the raw image — a grid of red, green, and blue pixel values — and crunches it down into a much smaller list of numbers that summarizes what's in the image. Think of it as the network's "eye" plus "brain that recognizes shapes and textures." If the image contains a fish, the encoder will produce a list of numbers that captures things like *the body is elongated, the texture is silvery, there's a dorsal fin here, the tail is forked,* and so on. You and I can't read those numbers directly, but downstream parts of the network can.

**Part 2: the head.** This is a small, lightweight part of the network — usually just one or two layers. It takes the summary numbers from the encoder and turns them into the final answer. For a classifier head, the output is a probability for each category. For a regression head, the output is a single number — in our case, the length in cm.

**This split matters enormously for your project, because your entire task is essentially: keep the head the same, but try a different encoder.** The published baseline uses a small, older encoder called MobileNetV2. Your job is to replace it with a much more powerful, modern encoder — a "Vision Foundation Model" — and see if that helps.

### Pretraining and transfer learning — the most important idea in modern AI

Here's the most important practical concept in modern deep learning. If you understand only one technical idea from this document, make it this one.

Training a big network from scratch needs huge amounts of data. Often millions of labeled examples. We don't have anywhere near that many labeled fish images — the AutoFish dataset has only 1,500 photos. If we trained a network from scratch on 1,500 images, it would be terrible.

**The trick is to start with a network that has already been trained on a different, much larger dataset, and then adapt it to our task.** This is called **transfer learning**. The pre-existing network is called **pretrained**.

The classic example: there's a famous dataset called **ImageNet** with about 1.3 million images of everyday objects (dogs, cars, mushrooms, lampposts, and so on). For decades, researchers have trained networks on ImageNet and shared those trained networks publicly. If you want to build any new image-related system, you typically start from one of those pretrained ImageNet networks and **fine-tune** it on your specific task.

Why does this work? Because the lower layers of a vision network learn very general things — how to detect edges, corners, textures, simple shapes. Those skills are useful for almost any image task. The upper layers learn more task-specific things, like "this looks like a Labrador" — those need to be replaced or adjusted for your task. By starting from ImageNet, the network already knows about edges, textures, and shapes, and you just need to teach it the fish-specific parts.

**This is why the baseline in the AutoFish paper works at all with only 1,500 images.** The MobileNetV2 encoder was pretrained on ImageNet, and then fine-tuned on the AutoFish data.

### So what is a "Vision Foundation Model"?

This is a more recent term. In the last few years, researchers realized that ImageNet pretraining — while great — was limited because ImageNet itself is only 1.3 million images and the labels were human-assigned categories. A new generation of pretrained models has been trained on **far larger datasets** (100 million, even 1 billion images) and using **self-supervised** objectives.

"Self-supervised" means the model learns from images alone, without any human labels. The typical trick is to give the model a corrupted version of an image — say, with parts masked out or blurred — and ask it to predict the original. Or to show it two different crops of the same image and ask it to recognize that they came from the same source. By doing this kind of task on hundreds of millions of images, the model accidentally learns extraordinarily rich, general-purpose features about the visual world.

These massive, self-supervised, general-purpose pretrained encoders are what we now call **Vision Foundation Models**, or VFMs. The most prominent ones today:

- **DINOv2** (released by Meta in 2023): trained on 142 million curated images using a self-supervised method called DINO. Its features are widely reported to transfer extraordinarily well to many downstream tasks. There's also a newer **DINOv3** from 2025.
- **SAM** (Segment Anything Model, Meta, 2023): trained on over a billion segmentation masks. SAM is technically a segmentation tool, but its internal encoder produces very shape-aware features that are useful far beyond segmentation. **SAM 2** is the 2024 successor.
- **CLIP** (OpenAI, 2021): trained on 400 million image-text pairs from the web. Its features are very strong for tasks that involve language or semantics, but a bit weaker for purely geometric tasks like measuring things.
- A few others (MAE, BLIP, EVA, etc.) — you don't need to memorize them.

**The big empirical claim about VFMs is this:** their features are so rich that for many tasks, you can **freeze** the entire encoder (don't update its parameters at all), train only a tiny head on top, and still get great results. This is incredibly useful when you have little training data, because there's much less to learn.

That is exactly why the task PDF mentions "in particular when labeled data are limited." VFMs are *supposed* to shine in low-data settings. Your project is asking: does this claim hold for fish length estimation?

---

## Part 3: The Dataset (AutoFish)

Now let's talk about the actual data you'll be working with. The dataset is called **AutoFish**, published in early 2025 by a research group at Aalborg University in Denmark. It's freely available on Hugging Face.

### How the data was collected

The researchers set up a laboratory rig: a 1-metre-by-1-metre section of a white conveyor belt with a high-quality industrial camera mounted 1.5 metres directly above it. The camera is fixed in place, the lighting is controlled, and the focus is set sharply. This is deliberately a clean, controlled setting — much easier than a real fishing boat, but a sensible starting point.

The fish themselves were caught by a real Danish fishing vessel in the North Sea. Five species dominate the dataset: cod, haddock, whiting, hake, and horse mackerel. The first three are closely related (all in the cod family) and look fairly similar — small differences in coloration, fin shape, and body proportions. This makes the dataset interesting: it's not "fish vs. shark vs. eel," it's distinguishing between similar-looking commercial species.

Before photographing, a marine biologist manually measured every fish with a ruler, **rounding each length to the nearest 5 millimetres** (0.5 cm). This is standard practice in fisheries science — measuring to 1 mm doesn't make a meaningful difference for population studies.

### The numbers

- **1,500 images** in total.
- **454 unique fish** — the same physical fish appears in multiple images (in different positions and orientations).
- **18,160 fish instances** across all images, each with its own segmentation mask, ID, and length label.
- The fish were divided into **25 groups**, with 14 to 24 fish per group.

### The Set1 / Set2 / All structure (read this carefully — it matters a lot)

Within each group, the same fish were photographed in three different ways:

- **Set1**: half the fish in the group are arranged on the belt so that **none of them touch or overlap**. Photographed in 20 different arrangements (varying positions and orientations). Easy case: every fish is fully visible.
- **Set2**: the other half of the fish in the group, also arranged with no overlap. Same setup as Set1, 20 images. Also an easy case.
- **All**: every fish in the group is placed on the belt at once, **deliberately overlapping and touching each other**. 20 images. Hard case: many fish are partially hidden behind others.

So each group contributes 60 images: 20 Set1 + 20 Set2 + 20 All.

**Why the researchers did this:** Set1 and Set2 give the model clean, easy examples to learn from. The "All" set tests whether the model can still measure fish accurately when they're piled up (which is what real fishing-boat conditions look like). The published baseline reports two MAE numbers — one for the easy regime, one for the hard regime — and you'll want to do the same.

### The critical thing about groups — please understand this

This is the most important practical detail in the entire dataset, and it's the thing your professor will almost certainly probe.

**Each individual fish appears in exactly one group.** Fish #173 lives entirely in group 7. It never appears in group 8 or any other group.

Why does this matter? Because when we split the dataset for training and testing, we **must** split at the group level — for example, "groups 1 through 16 are for training, groups 17–20 for validation, groups 21–25 for testing." If we instead split at the image level (taking random photos from across all groups), the same physical fish would appear in both the training set and the test set. The network could memorize the appearance of specific individual fish and look perfect on the test set, even though it hadn't really learned to measure fish in general.

The dataset authors specifically organized the data into groups to make a clean, leakage-free split possible. Honoring this structure is non-negotiable. Mention this in your meeting — it shows you understood the data.

### What annotations come with each fish

For every one of the 18,160 fish instances, you get:

- A **segmentation mask**: a pixel-by-pixel outline of exactly which pixels in the image belong to that fish. This is more precise than just a rectangular bounding box.
- A **fish ID**: a unique number identifying this specific physical fish. (So you can track "fish #173 appears in these 60 images.")
- A **species label**: cod, haddock, whiting, etc.
- A **length** in centimetres, rounded to 5 mm.

The masks are very useful: they let you crop out exactly one fish from a crowded image, with no other fish or background distracting the network.

---

## Part 4: What the Existing Baseline Does

Now let's look at the existing method we're going to compare against. The AutoFish paper proposes **two** baseline methods for length estimation. You should know what both do, but only one is what your project really cares about.

### Baseline 1 (Skeletonization, classical approach)

This one doesn't use deep learning at all. The recipe is:

1. Take the segmentation mask of one fish (a pixel outline of the fish).
2. Apply a classical image-processing operation called **skeletonization**, which shrinks the mask down to a 1-pixel-wide curve running along the centerline of the fish, head to tail. Imagine pressing a fish flat until you're left with just its backbone-trace.
3. Measure the length of that curve in pixels.
4. Multiply by a known **pixel-to-centimeter conversion factor** (which the researchers calibrated using checkerboard patterns at known sizes — that's why the AutoFish paper mentions 20 calibration images per group).

That's it. No training. It works because the camera setup is fixed, so the pixel-to-cm conversion is reliable. It's a clever, simple approach and a useful sanity check, but it's not what we're focused on.

### Baseline 2 (CNN regression — this is the one)

This is **the baseline** the task PDF means when it says "an established deep learning baseline."

The recipe:

1. For each fish in an image, use its segmentation mask to find the bounding box (the smallest rectangle that contains the fish).
2. **Crop** that rectangle out of the original image, with a little extra padding around the edges. Now you have a small image containing exactly one fish.
3. Resize the crop to a fixed size (e.g. 224×224 pixels) so the network always sees the same shape.
4. Feed the crop into a neural network with:
   - **Encoder:** MobileNetV2. This is a small, efficient convolutional neural network originally designed for mobile phones. It was pretrained on ImageNet.
   - **Head:** a tiny addition on top — basically one linear layer — that outputs a single number.
5. Train the whole network on the (crop, length) pairs from the training set, telling it to make its output match the true length as closely as possible.

After training, when you give the network a new fish crop, it outputs an estimated length in cm.

**The reported results:** mean absolute error of **0.62 cm on non-occluded fish** (the Set1/Set2 case) and **1.38 cm on occluded fish** (the All case). Those are the numbers your reproduction needs to come close to.

### One subtle but important point

Notice that the network is **not** told the pixel-to-cm conversion. It just sees a 224×224 crop and outputs a number. Yet somehow it learns to produce length in centimetres rather than in pixels.

How? Because all 1,500 images come from the same fixed camera at the same fixed height. The relationship between pixels-in-the-crop and real-world-cm is constant across the dataset (up to small distortions). The network implicitly learns this conversion during training. The downside is that if you took the trained model and pointed a different camera at fish, it would fail — the learned conversion would no longer apply. This is a limitation worth knowing about and noting in your meeting.

---

## Part 5: The Research Question, Decoded

The task PDF says:

> *"How well does an established deep learning baseline perform for fish length estimation, and does replacing its encoder with a Vision Foundation Model (VFM) improve performance, in particular when labeled data are limited?"*

Let's unpack this one phrase at a time.

**"How well does an established deep learning baseline perform"** — first, you need to reproduce the baseline (the MobileNetV2 regressor) yourself and verify your reproduced version gets numbers close to the published ones (0.62 / 1.38 cm MAE). This is your sanity check that the rest of your work is built on solid ground.

**"and does replacing its encoder with a Vision Foundation Model (VFM) improve performance"** — then, you build a second version where you swap MobileNetV2 out for a VFM like DINOv2 or SAM, keeping the head and the training procedure as similar as possible. Train it the same way. Measure the MAE again. Compare.

**"in particular when labeled data are limited"** — this is the most important part. Even if VFMs aren't dramatically better when you have all the training data, the bet is that they should be much better when you have very little. So you'd repeat the comparison while pretending you only have, say, 50% of the training data. Then 25%. Then 10%. Then 5%. Plot the results. If the VFM line stays high (good) while the MobileNetV2 line collapses (bad) as you reduce data, you've demonstrated something interesting.

**"To what extent does the resulting approach transfer to a second fish dataset?"** — this is the optional extension. If a different fish dataset becomes available during the project, you'd test whether your trained model still works on it. (Probably not very well, since the camera setup would differ — but VFMs might handle the change better than MobileNetV2.)

So the three concrete things your project is asking are:

1. **Reproduce the baseline.**
2. **Build a VFM-based version and compare them at full data.**
3. **Compare them again at reduced amounts of training data — this is the headline experiment.**

Plus optionally:

4. **Test on a second dataset, if one is available.**

---

## Part 6: A Glossary of Everything You Just Learned

For quick reference. Print this if you like.

| Term | What it means in plain language |
|---|---|
| **Regression** | A neural network task where the output is a continuous number (e.g. length in cm), not a category. |
| **Encoder / backbone** | The big part of a vision neural network that turns an image into a summary of numbers describing what's in it. |
| **Head** | The small part on top of the encoder that turns the summary into the final answer (a length, a category, etc.). |
| **Training** | The process of slowly adjusting the network's millions of internal numbers so that it makes accurate predictions on the training examples. |
| **Training set / validation set / test set** | The labelled examples are split into three groups: training (used to teach the network), validation (used to check progress while training and to tune choices), and test (touched only at the very end to report final performance). |
| **Label** | The correct answer for an example. In our case, the true length of a fish, measured by a marine biologist. |
| **Pretraining** | Training a network on a large, general dataset before adapting it to your specific task. |
| **Transfer learning** | Taking a pretrained network and fine-tuning or adapting it to a new task. |
| **Fine-tuning** | Continuing to train a pretrained network on your task's data, usually with a smaller learning rate to avoid destroying what it already knows. |
| **Freezing** | Choosing not to update some parts of the network during training — typically the pretrained encoder, so only the head is learned. Useful when data is limited. |
| **ImageNet** | The classic large image dataset (~1.3M images, 1000 categories). The default source of pretrained encoders for many years. |
| **Vision Foundation Model (VFM)** | A more recent, much larger pretrained image encoder, usually trained with self-supervised objectives on hundreds of millions of unlabelled images. Examples: DINOv2, SAM, CLIP. |
| **Self-supervised learning** | Training a model on images alone, without human labels, by giving it puzzle tasks like "fill in the masked part of this image." |
| **MobileNetV2** | A small, efficient convolutional neural network from 2018. The encoder in the AutoFish published baseline. |
| **DINOv2** | A prominent VFM by Meta from 2023, trained self-supervised on 142M images. Excellent for general transfer. |
| **SAM (Segment Anything)** | A model by Meta that segments any object in any image. Its encoder is sometimes used as a VFM in its own right. |
| **Segmentation mask** | A pixel-precise outline of one object in an image. AutoFish provides one for every fish. |
| **Bounding box** | A simpler kind of outline — just the smallest rectangle that contains the object. |
| **Crop** | A small image extracted from a larger one. Here, the rectangular region around one fish. |
| **Occlusion** | When one object partially blocks another. The "All" subsets of AutoFish are deliberately full of occlusion. |
| **MAE (mean absolute error)** | The average of \|predicted − actual\| across all test examples. Our primary metric. Reported in cm. Lower is better. |
| **MSE / RMSE** | Other ways to measure error. RMSE (root mean squared error) penalizes large mistakes more than small ones. |
| **Group-level split** | The thing your professor will care about: dividing AutoFish for train/val/test by group, not by image, so the same fish never appears in two splits. |

---

## Part 7: How This Sets Up the First Meeting

After reading all of the above, you can now genuinely say: *"I understand what this project is about. I understand the dataset. I understand what the published baseline does. I understand what a VFM is and why it might help. I have a sensible idea of what the main experiment will look like."*

That's exactly what the first meeting should be. You're not committing to specific architectures, splits, or hyperparameters yet — you're showing your supervisor that the foundation is solid before you start building on it.

The next document (`first_meeting_briefing.md`) gives you the actual meeting script: what to say, what order to say it in, what questions to ask. The document after that (`research_plan.md`) is the deeper plan for what you'll do over the following weeks, and you don't need most of it for the first meeting — it's there for later.

When you read those documents, you'll notice they assume you already know what's in *this* one. That's deliberate. This is the foundation.
