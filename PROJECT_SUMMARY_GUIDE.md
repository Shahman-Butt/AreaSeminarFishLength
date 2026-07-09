# AutoFish Baseline Reproduction Summary Guide

For the full formal Area Seminar report, use:

```text
AREA_SEMINAR_FULL_REPORT.md
```

## 1. Where the project is

You are currently looking at the **local Windows copy** of the project in VS Code:

```text
C:\Users\Shahman\Desktop\SEM3\AreaSeminar
```

The actual long GPU training was run on the university server:

```text
sb2597@baltic.informatik.uni-rostock.de
/home/sb2597/autofish_baseline_repro
```

So there are two copies:

| Place | What it means | Main use |
|---|---|---|
| Local Windows folder | Your editable project copy in VS Code | Reading code, editing docs, keeping copied results |
| Server folder | The Linux/GPU execution copy | Downloading dataset, generating crops, training model |

We are **not currently inside the server shell** in this VS Code window. Codex connects to the server through SSH commands when needed. The completed result files were copied back from the server into this local folder under:

```text
runs/baseline_official
```

## 2. What paper we reproduced

The target paper is:

```text
Bengtson et al.,
"AutoFish: Dataset and Benchmark for Fine-grained Analysis of Fish",
WACVW 2025 / arXiv:2501.03767
```

The paper introduces the **AutoFish** dataset and benchmark. The part we reproduced is the fish length estimation baseline.

### Layman explanation

The paper asks:

> If we have underwater images of fish, can a computer estimate the real fish length in centimeters?

The authors provide images, fish masks, bounding boxes, fish identities, and measured lengths. They also provide baseline methods so future work can compare against them.

### Technical explanation

The reproduced baseline is the paper's **REG** model:

- Input: cropped fish image.
- Preprocessing: use fish segmentation/mask so the model focuses on fish pixels.
- Encoder: **MobileNetV2** pretrained on ImageNet.
- Extra numerical input: bounding-box coordinates.
- Output: one continuous value, fish length in centimeters.
- Loss: L1 loss / mean absolute error style regression.
- Evaluation metric: MAE in centimeters.

The paper reports the main REG MobileNetV2 results as:

| Evaluation subset | Paper MAE |
|---|---:|
| Non-occluded Set1+Set2 | 0.62 cm |
| Occluded All | 1.38 cm |

## 3. What we built

We created a clean reproduction project around the first research question from the roadmap:

> Reproduce the baseline fish-length estimation result before moving to foundation-model comparisons.

Important files:

| File/folder | Purpose |
|---|---|
| `configs/baseline_official.json` | Official-style 200 epoch MobileNetV2 baseline config |
| `configs/baseline_smoke.json` | Tiny smoke-test config to check the pipeline quickly |
| `scripts/download_autofish.py` | Downloads the AutoFish dataset from Hugging Face |
| `scripts/build_autofish_index.py` | Builds a clean CSV index from raw annotations |
| `scripts/make_crops.py` | Creates cropped fish images for training |
| `scripts/check_processed.py` | Verifies processed data, crop count, and split leakage |
| `src/autofish_vfm/data.py` | PyTorch dataset code |
| `src/autofish_vfm/models.py` | MobileNetV2 regression model |
| `src/autofish_vfm/train_baseline.py` | Training loop |
| `src/autofish_vfm/evaluate.py` | Test evaluation and metrics export |
| `runs/baseline_official` | Final copied result files from server |

## 4. Dataset and split

Dataset:

```text
Hugging Face: vapaau/autofish
```

Server dataset location:

```text
/home/sb2597/autofish_baseline_repro/data/raw/autofish
```

Processed server data:

```text
/home/sb2597/autofish_baseline_repro/data/processed/index.csv
/home/sb2597/autofish_baseline_repro/data/processed/crops
```

The official group split used:

| Split | Groups |
|---|---|
| Train | 2, 3, 4, 5, 7, 8, 9, 12, 13, 15, 16, 18, 19, 23, 24 |
| Validation | 1, 6, 11, 17, 25 |
| Test | 10, 14, 20, 21, 22 |

Set mapping used:

| Image number range | Set name | Meaning |
|---|---|---|
| 00001-00020 | Set1 | non-occluded |
| 00021-00040 | Set2 | non-occluded |
| 00041-00060 | All | occluded/crowded setting |

During indexing, we found one annotation issue:

```text
fish_id=113 appeared across train and test.
```

To avoid fish leakage, the script removed one singleton duplicate annotation and wrote the exclusion into:

```text
data/processed/exclusions.json
```

This is why our non-occluded test count is **1,879** instead of exactly **1,880**.

## 5. Server setup

Server:

```text
baltic.informatik.uni-rostock.de
```

Username:

```text
sb2597
```

Project folder created on server:

```text
/home/sb2597/autofish_baseline_repro
```

Environment:

- Python 3.10 on server.
- `python3 -m venv` was not available because server lacked `ensurepip`.
- We installed user-local `virtualenv`.
- Created isolated environment:

```text
/home/sb2597/autofish_baseline_repro/.venv
```

Important installed packages:

- PyTorch
- Torchvision
- CUDA-enabled PyTorch wheel
- NumPy
- Pandas
- Pillow
- scikit-learn
- tqdm
- Hugging Face Hub / datasets
- pycocotools

GPU used:

```text
NVIDIA RTX 5000 Ada Generation
```

CUDA was available and detected by PyTorch.

## 6. What we ran

### Step 1: Download dataset

Downloaded AutoFish from Hugging Face onto the server.

### Step 2: Build clean index

Built a processed annotation table:

```text
data/processed/index.csv
```

Final processed count:

| Item | Count |
|---|---:|
| Rows / annotations | 18,157 |
| Images | 1,500 |
| Unique fish | 454 |
| Groups | 25 |
| Missing crops | 0 |
| Fish leakage across splits | 0 |

### Step 3: Generate fish crops

Generated one crop per annotation:

```text
data/processed/crops
```

All 18,157 crops were created successfully.

### Step 4: Smoke test

Before the full training, we ran a tiny smoke test with only a few batches.

Purpose:

- Confirm dataset loading works.
- Confirm crop paths are correct.
- Confirm MobileNetV2 builds.
- Confirm CUDA training works.
- Confirm checkpoint writing works.

The smoke metric was intentionally bad because it only trained on a tiny subset. That was expected.

### Step 5: Full official baseline

Ran the full 200 epoch REG MobileNetV2 baseline on the server.

Server output folder:

```text
/home/sb2597/autofish_baseline_repro/runs/baseline_official
```

Training completed successfully:

| Training detail | Value |
|---|---:|
| Epochs | 200 |
| Best validation epoch | 153 |
| Best validation MAE | 0.805 cm |

After training, we evaluated the best checkpoint on the test split.

## 7. Final reproduction results

Copied local result files:

```text
runs/baseline_official/history.csv
runs/baseline_official/test_metrics.json
runs/baseline_official/test_metrics.predictions.csv
```

Main result table:

| Model | Paper non-occluded MAE | Our non-occluded MAE | Paper occluded MAE | Our occluded MAE | Our full test MAE |
|---|---:|---:|---:|---:|---:|
| REG MobileNetV2 | 0.62 cm | 0.633 cm | 1.38 cm | 0.909 cm | 0.771 cm |

## 7.1 So-far comparison table

This table shows what is finished now and what comes next.

| Stage | Model / experiment | Paper non-occluded MAE | Paper occluded MAE | Our non-occluded MAE | Our occluded MAE | Our full test MAE | Status |
|---|---|---:|---:|---:|---:|---:|---|
| Q1 baseline | REG MobileNetV2 | 0.62 cm | 1.38 cm | 0.633 cm | 0.909 cm | 0.771 cm | Complete |
| Q2 VFM | DINOv2 ViT-S/14 frozen encoder + regression head | Not reported | Not reported | 1.690 cm | 1.786 cm | 1.738 cm | Complete |
| Q2 VFM | DINOv2 ViT-S/14 fine-tuned, encoder LR 1e-5 | Not reported | Not reported | 1.636 cm | 1.919 cm | 1.778 cm | Complete |
| Q2 VFM | DINOv2 ViT-S/14 fine-tuned, encoder LR 1e-6 | Not reported | Not reported | 2.075 cm | 2.189 cm | 2.132 cm | Complete |
| Q2 VFM | DINOv2 ViT-S/14 frozen head then last block | Not reported | Not reported | 1.340 cm | 1.537 cm | 1.439 cm | Complete |
| Q2 VFM | CLIP ViT-B/32 frozen encoder + regression head | Not reported | Not reported | 0.898 cm | 1.106 cm | 1.002 cm | Complete |
| Q2 VFM | CLIP ViT-B/32 frozen head then last visual block | Not reported | Not reported | 0.842 cm | 1.074 cm | 0.958 cm | Complete |
| Q2 VFM | ConvNeXt-Tiny ImageNet encoder | Not reported | Not reported | 0.814 cm | 1.014 cm | 0.914 cm | Complete |

Layman reading:

> The first row is our completed reproduction. The rows marked TBD are the next experiments. Their paper columns say Not reported because the AutoFish paper's main table is for its own baseline methods; our next work is to test new foundation-model encoders against the reproduced baseline.

Full metrics:

| Subset | MAE cm | RMSE cm | MAPE % | Bias cm | R2 | n |
|---|---:|---:|---:|---:|---:|---:|
| Full test | 0.771 | 1.268 | 2.411 | 0.035 | 0.947 | 3,759 |
| Non-occluded Set1+Set2 | 0.633 | 1.027 | 1.960 | 0.080 | 0.965 | 1,879 |
| Occluded All | 0.909 | 1.470 | 2.862 | -0.009 | 0.929 | 1,880 |

## 8. Did our results match the paper?

### Short answer

Yes, the baseline reproduction is successful.

The most important paper number is:

```text
Paper non-occluded MAE: 0.62 cm
Our non-occluded MAE:   0.633 cm
```

That is extremely close. The difference is only:

```text
0.013 cm
```

That is about one tenth of a millimeter, which is very small.

For occluded fish:

```text
Paper occluded MAE: 1.38 cm
Our occluded MAE:   0.909 cm
```

Our result is better than the reported number.

### Layman explanation

Imagine the model is measuring fish with a ruler from images. On normal clear fish images, the original paper was wrong by about **0.62 cm** on average. Our reproduced model was wrong by about **0.63 cm**, basically the same.

For harder crowded/occluded fish images, the paper was wrong by about **1.38 cm**. Our reproduced model was wrong by about **0.91 cm**, which is better.

So the project successfully recreated the baseline and reached the expected level of performance.

### Technical explanation

The non-occluded benchmark closely matches the paper, meaning the training pipeline, group split, crop generation, MobileNetV2 architecture, bbox feature input, and regression objective are all behaving correctly.

The occluded result is better than the paper. Possible reasons:

- Our leakage-prevention cleanup removed one problematic cross-split duplicate annotation.
- Our crop/mask preprocessing is close to the paper but not byte-for-byte identical.
- Random initialization and GPU nondeterminism can change regression performance.
- The best checkpoint was selected using validation MAE across 200 epochs.
- Library versions differ from the original paper environment.

This difference is not a failure. The baseline is reproduced very closely where the paper's key headline value is most direct, and the occluded score is stronger rather than worse.

## 9. How to explain this in a report

A good report sentence:

```text
We reproduced the AutoFish REG MobileNetV2 length-estimation baseline using the official group-level split. The reproduced non-occluded MAE was 0.633 cm, closely matching the paper's reported 0.62 cm. On the occluded subset, our model achieved 0.909 cm MAE compared with the reported 1.38 cm, likely due to minor preprocessing and environment differences plus removal of one cross-split duplicate annotation.
```

## 10. What is complete and what is next

Complete:

- Server SSH access configured.
- Separate server project folder created.
- Isolated Python environment created.
- AutoFish dataset downloaded.
- Official split reproduced.
- Crop preprocessing completed.
- Dataset integrity checked.
- MobileNetV2 REG baseline trained for 200 epochs.
- Best checkpoint evaluated.
- Results copied back locally.
- README and project notes updated.

Not yet done:

- More foundation-model encoder experiments beyond the first DINOv2 and CLIP tests.
- Multi-seed statistical repeats.
- Ablation studies.
- Final seminar report writing.

This means **Q1 baseline reproduction is complete** and ready to be used as the reference point for the next research questions.

## 10.1 Q2 experiment completed: DINOv2 frozen encoder

The first next experiment is now separated from the baseline:

```text
/home/sb2597/autofish_baseline_repro/runs/dinov2_vits14_frozen
```

What stays the same:

- same AutoFish dataset,
- same train/validation/test split,
- same fish crops,
- same bounding-box input,
- same length target in centimeters,
- same final test metrics.

What changes:

```text
MobileNetV2 image encoder -> DINOv2 ViT-S/14 image encoder
```

Layman explanation:

> We keep the exam exactly the same and only change the image-understanding brain. This lets us ask whether a modern vision foundation model understands fish shape better than the older MobileNetV2 baseline.

Technical explanation:

- DINOv2 is loaded through PyTorch Hub from `facebookresearch/dinov2`.
- Current model: `dinov2_vits14`.
- The DINOv2 encoder is frozen for the first experiment.
- Only the small regression head is trained.
- Frozen encoder keeps GPU memory low and makes the first comparison safer on the shared server.

Final result:

```text
Full test MAE: 1.738 cm
Non-occluded Set1+Set2 MAE: 1.690 cm
Occluded All MAE: 1.786 cm
```

Interpretation:

> Frozen DINOv2 did not beat the reproduced MobileNetV2 baseline. This does not mean DINOv2 is bad; it means that using DINOv2 only as a frozen feature extractor is probably not enough for this precise length-estimation task. Fish length requires fine scale and geometry information, and the baseline MobileNetV2 was fully trained on the task.

Next technical step:

```text
Fine-tune DINOv2 partially, or cache DINOv2 features and train a stronger regression model.
```

The fine-tuning run used:

```text
DINOv2 ViT-S/14 encoder learning rate: 0.00001
Regression head learning rate: 0.0001
Batch size: 8
Epochs: 50
```

Layman explanation:

> Instead of keeping DINOv2 completely fixed, we now let it adjust very gently to the fish-length task. The small learning rate means the model changes slowly, reducing the risk of destroying its pretrained visual knowledge.

Final fine-tuned DINOv2 result:

```text
Full test MAE: 1.778 cm
Non-occluded Set1+Set2 MAE: 1.636 cm
Occluded All MAE: 1.919 cm
```

Interpretation:

> Fine-tuning improved the non-occluded DINOv2 score slightly compared with frozen DINOv2, but it made the occluded score worse. The validation curve was unstable, so this setup is not yet a strong replacement for the MobileNetV2 baseline.

Technical reading:

- Frozen DINOv2 non-occluded MAE: `1.690 cm`
- Fine-tuned DINOv2 non-occluded MAE: `1.636 cm`
- Frozen DINOv2 occluded MAE: `1.786 cm`
- Fine-tuned DINOv2 occluded MAE: `1.919 cm`

Next better-controlled fine-tuning options:

- lower encoder learning rate, for example `1e-6` (already completed),
- freeze most DINOv2 blocks and train only the last block,
- add learning-rate warmup,
- train the head first, then unfreeze encoder,
- cache DINOv2 features and test stronger regressors before expensive fine-tuning.

Current next run:

```text
DINOv2 ViT-S/14 encoder learning rate: 0.000001
Regression head learning rate: 0.0001
Batch size: 8
Epochs: 50
```

Layman expectation:

> This run lets DINOv2 adapt much more slowly. If the previous run was unstable because the pretrained model changed too quickly, this lower learning rate should be steadier.

Final `1e-6` result:

```text
Full test MAE: 2.132 cm
Non-occluded Set1+Set2 MAE: 2.075 cm
Occluded All MAE: 2.189 cm
```

Interpretation:

> The lower learning rate made the run more conservative, but it did not improve the final result. It performed worse than frozen DINOv2 and worse than the `1e-5` fine-tuning run. So the issue is probably not only "learning rate too high"; the model likely needs a more structured adaptation strategy.

Best next direction:

```text
Head-first training, then unfreeze only the last DINOv2 block.
```

Why:

- the head can first learn the fish-length mapping without disturbing DINOv2,
- only the last visual block adapts afterward,
- this is less aggressive than full-encoder fine-tuning,
- it may preserve general DINOv2 features while adapting high-level geometry cues.

Final last-block result:

```text
Full test MAE: 1.439 cm
Non-occluded Set1+Set2 MAE: 1.340 cm
Occluded All MAE: 1.537 cm
```

Interpretation:

> This is the best DINOv2 experiment so far. It confirms that controlled partial fine-tuning is better than both frozen DINOv2 and full-encoder fine-tuning for this task. However, it still does not beat the reproduced MobileNetV2 baseline, which remains the strongest model in our experiments.

Comparison to other DINOv2 runs:

| DINOv2 variant | Non-occluded MAE | Occluded MAE | Full test MAE |
|---|---:|---:|---:|
| Frozen encoder | 1.690 cm | 1.786 cm | 1.738 cm |
| Full fine-tune, encoder LR 1e-5 | 1.636 cm | 1.919 cm | 1.778 cm |
| Full fine-tune, encoder LR 1e-6 | 2.075 cm | 2.189 cm | 2.132 cm |
| Frozen head then last block | 1.340 cm | 1.537 cm | 1.439 cm |

## 10.2 Q2 experiment completed: CLIP frozen encoder

The next foundation-model experiment used:

```text
CLIP ViT-B/32 frozen image encoder + regression head
```

Local copied result files:

```text
runs/clip_vitb32_frozen/history.csv
runs/clip_vitb32_frozen/test_metrics.json
runs/clip_vitb32_frozen/test_metrics.predictions.csv
```

Best validation epoch:

```text
epoch 82
validation MAE 1.016 cm
```

Final CLIP frozen result:

| Subset | MAE cm | RMSE cm | MAPE % | Bias cm | R2 | n |
|---|---:|---:|---:|---:|---:|---:|
| Full test | 1.002 | 1.367 | 3.148 | 0.248 | 0.938 | 3,759 |
| Non-occluded Set1+Set2 | 0.898 | 1.174 | 2.840 | 0.307 | 0.954 | 1,879 |
| Occluded All | 1.106 | 1.536 | 3.457 | 0.189 | 0.922 | 1,880 |

Layman explanation:

> CLIP was trained on many image-text pairs, so it has broad visual knowledge. We froze that visual brain and trained only a small final ruler-like layer for fish length. It worked much better than frozen DINOv2 in this project, but it still did not beat the task-trained MobileNetV2 baseline.

Technical interpretation:

- CLIP frozen is the strongest VFM experiment so far.
- It improves a lot over all DINOv2 variants.
- It still trails MobileNetV2 on the main non-occluded baseline comparison:

```text
MobileNetV2 non-occluded MAE: 0.633 cm
CLIP frozen non-occluded MAE: 0.898 cm
```

- On occluded images, CLIP frozen is worse than our MobileNetV2 run but still better than the paper's reported occluded REG number:

```text
Paper REG occluded MAE:       1.38 cm
Our MobileNetV2 occluded MAE: 0.909 cm
Our CLIP frozen occluded MAE: 1.106 cm
```

What this means:

> CLIP is promising enough to continue. The next test should not just freeze CLIP; it should fine-tune a small part of CLIP carefully, such as the last visual transformer block plus the regression head.

Completed follow-up run:

```text
runs/clip_vitb32_lastblock_from_frozen
```

Layman explanation:

> We are keeping most of CLIP's visual knowledge fixed and only letting the final visual block plus the small length-prediction head adjust to the fish dataset. This is a careful middle path between "freeze everything" and "change the whole large model."

Technical setup:

- starts from `runs/clip_vitb32_frozen/best.pt`,
- unfreezes `1` CLIP visual transformer block,
- encoder learning rate: `1e-6`,
- head learning rate: `5e-5`,
- batch size: `8`,
- epochs: `40`,
- output folder: `runs/clip_vitb32_lastblock_from_frozen`.

Best validation epoch:

```text
epoch 11
validation MAE 1.003 cm
```

Final CLIP last-block test result:

| Subset | MAE cm | RMSE cm | MAPE % | Bias cm | R2 | n |
|---|---:|---:|---:|---:|---:|---:|
| Full test | 0.958 | 1.323 | 2.991 | 0.138 | 0.942 | 3,759 |
| Non-occluded Set1+Set2 | 0.842 | 1.117 | 2.632 | 0.144 | 0.959 | 1,879 |
| Occluded All | 1.074 | 1.501 | 3.351 | 0.133 | 0.925 | 1,880 |

Comparison to frozen CLIP:

| CLIP variant | Non-occluded MAE | Occluded MAE | Full test MAE |
|---|---:|---:|---:|
| Frozen encoder | 0.898 cm | 1.106 cm | 1.002 cm |
| Last visual block fine-tuned | 0.842 cm | 1.074 cm | 0.958 cm |

Interpretation:

> Partial CLIP fine-tuning helped. The improvement is not huge, but it is consistent across non-occluded, occluded, and full-test metrics. This makes CLIP the strongest foundation-model direction so far. However, MobileNetV2 still remains the best overall model for the reproduced baseline task.

## 10.3 Q2 experiment completed: ConvNeXt-Tiny encoder

The optional supervised ImageNet encoder experiment is now complete:

```text
runs/convnext_tiny_official
```

ConvNeXt-Tiny was chosen because MobileNetV2 is also a supervised ImageNet-style encoder and remains the best model. Testing ConvNeXt asks a clean question:

> If a lightweight supervised CNN-style baseline works well, does a stronger modern supervised ImageNet encoder improve the result?

Final ConvNeXt-Tiny result:

| Subset | MAE cm | RMSE cm | MAPE % | Bias cm | R2 | n |
|---|---:|---:|---:|---:|---:|---:|
| Full test | 0.914 | 1.338 | 2.822 | 0.131 | 0.941 | 3,759 |
| Non-occluded Set1+Set2 | 0.814 | 1.181 | 2.496 | 0.136 | 0.954 | 1,879 |
| Occluded All | 1.014 | 1.479 | 3.148 | 0.127 | 0.928 | 1,880 |

Best validation epoch:

```text
epoch 83
validation MAE 0.922 cm
```

Comparison with strongest previous non-baseline models:

| Model | Non-occluded MAE | Occluded MAE | Full test MAE |
|---|---:|---:|---:|
| CLIP last visual block fine-tuned | 0.842 cm | 1.074 cm | 0.958 cm |
| ConvNeXt-Tiny | 0.814 cm | 1.014 cm | 0.914 cm |
| MobileNetV2 baseline | 0.633 cm | 0.909 cm | 0.771 cm |

Interpretation:

> ConvNeXt-Tiny is the best non-baseline encoder result so far. It beats CLIP last-block fine-tuning on all three main test metrics. However, the reproduced MobileNetV2 baseline still remains the best overall result.

## 11. Deep layman glossary of important terms

### Baseline

A **baseline** is the first standard result we try to reproduce before doing anything new.

Layman version:

> Before claiming "my new method is better", we must first show that we can reproduce the old method fairly.

In this project, the baseline is the paper's **REG MobileNetV2** model for fish length prediction.

Why it matters:

- It proves our dataset pipeline is correct.
- It gives us a fair comparison point.
- Later, if we use Vision Foundation Models, we compare them against this reproduced baseline.

### Reproduction

Reproduction means we try to repeat the paper's experiment and get similar results.

Layman version:

> We are checking whether the paper's method works when we run it ourselves.

Good reproduction does not always mean every number is identical. Small differences can happen because of:

- different GPU,
- different PyTorch version,
- random initialization,
- slightly different image preprocessing,
- data-cleaning decisions.

Our key non-occluded result matched very closely:

```text
Paper: 0.62 cm MAE
Ours:  0.633 cm MAE
```

### REG

**REG** means **regression**.

Layman version:

> Regression is when the model predicts a number.

Here the number is fish length in centimeters.

Examples:

| Task | Type |
|---|---|
| Predict "fish" or "not fish" | classification |
| Predict species name | classification |
| Predict fish length as 34.7 cm | regression |

So REG means the model is not choosing a category. It is estimating a continuous numeric value.

### MobileNetV2

**MobileNetV2** is a convolutional neural network architecture.

Layman version:

> It is a lightweight image-understanding model that looks at an image and extracts useful visual patterns.

It was originally designed to be efficient, so it can run faster and with fewer parameters than very large models.

In our project:

- MobileNetV2 looks at the fish crop.
- It extracts visual features like shape, body outline, size cues, texture, and proportions.
- A regression head converts those features into a length prediction.

### ImageNet pretrained

ImageNet is a huge dataset of everyday images.

Layman version:

> Instead of training the model from zero, we start from a model that already learned general image patterns.

Even though ImageNet is not a fish-length dataset, pretrained models already know useful low-level visual ideas:

- edges,
- curves,
- textures,
- object shapes,
- contrast,
- parts of objects.

This helps the fish model learn faster and better.

### Encoder

The **encoder** is the feature extractor part of the model.

Layman version:

> It turns an image into a compact summary of important visual information.

For example, instead of seeing raw pixels, the encoder produces a feature vector that says, in a learned way:

```text
long body, thin shape, visible tail, certain proportions, certain texture...
```

The regression head then uses this summary to predict fish length.

### Regression head

The **head** is the final prediction part of the model.

Layman version:

> The encoder understands the image; the head gives the final answer.

In our model:

```text
fish crop image -> MobileNetV2 encoder -> feature vector
bbox numbers ---------------------------> joined with feature vector
joined features -> regression head -> predicted length in cm
```

### Bounding box / bbox

A **bounding box** is a rectangle around the fish.

Layman version:

> It tells the computer where the fish is in the image.

The bbox usually has four numbers:

```text
x position, y position, width, height
```

Why bbox helps:

- Fish size in the image is related to real length.
- The model can use both visual appearance and geometric position/size cues.
- The paper's REG baseline includes bbox coordinates, so we included them too.

### Segmentation mask

A **segmentation mask** marks the exact fish pixels.

Layman version:

> It is like coloring only the fish and ignoring the background.

Why it helps:

- Underwater images have clutter.
- Other fish or background can confuse the model.
- Masking focuses the model on the target fish.

### Crop

A **crop** is a smaller image cut out from the full image.

Layman version:

> Instead of giving the whole underwater photo to the model, we cut out the fish area.

Why crop:

- The fish becomes the main subject.
- Training becomes simpler.
- The model sees less irrelevant background.
- It follows the baseline method.

In this project, we created one crop for every fish annotation:

```text
18,157 crops
```

### Data split

A **data split** divides data into separate parts:

| Split | Purpose |
|---|---|
| Train | Model learns from this |
| Validation | Used during training to choose the best checkpoint |
| Test | Final unbiased evaluation |

Layman version:

> Train is like practice questions, validation is like mock exams, and test is the final exam.

Why we must keep them separate:

- If the model sees test examples during training, the final score is unfair.
- We want to know if it works on fish it has not learned from directly.

### Group-level split

The AutoFish data is organized into groups. We split by group, not random individual images.

Layman version:

> We keep whole recording groups separate so the model cannot accidentally see nearly identical fish during training and testing.

This is stricter and fairer than randomly mixing images.

### Leakage

**Leakage** means information from the test set accidentally gets into training.

Layman version:

> It is like seeing final exam answers while studying.

In machine learning, leakage makes results look better than they really are.

In this dataset, we found one fish ID appeared across train and test:

```text
fish_id=113
```

We removed one singleton duplicate annotation to prevent leakage.

Why professor may care:

> If leakage exists, the model may memorize a fish instead of truly learning length estimation.

Our processed data check showed:

```text
fish leakage across splits: 0
```

### Epoch

An **epoch** means one full pass through the training data.

Layman version:

> If the training set is a textbook, one epoch means the model read the whole textbook once.

We trained for:

```text
200 epochs
```

Why many epochs:

- Early epochs learn rough patterns.
- Later epochs refine predictions.
- The model keeps improving until validation performance stops improving.

Best validation checkpoint happened at:

```text
epoch 153
```

This means epoch 153 had the best validation MAE, so that checkpoint was used for test evaluation.

### Checkpoint

A **checkpoint** is a saved model state.

Layman version:

> It is like saving the model's brain at a certain training moment.

We saved:

| File | Meaning |
|---|---|
| `last.pt` | model after final epoch |
| `best.pt` | model with best validation result |

The test results use `best.pt`.

### Validation

Validation is the model check during training.

Layman version:

> We do not use the final exam every time. We use a mock exam to decide which model version is best.

Why:

- Avoid choosing based on test set.
- Avoid overfitting.
- Select a fair best checkpoint.

### Overfitting

Overfitting means the model learns training examples too specifically.

Layman version:

> It memorizes practice questions instead of learning the real concept.

Signs:

- training loss keeps improving,
- validation performance stops improving or gets worse.

Using validation best checkpoint helps reduce this risk.

## 12. Metrics explained in layman terms

### Error

For one fish:

```text
error = predicted length - real length
```

Example:

```text
real length = 30 cm
predicted length = 31.5 cm
error = +1.5 cm
```

### MAE

MAE means **Mean Absolute Error**.

Layman version:

> On average, how many centimeters wrong is the model?

Formula idea:

```text
absolute error = ignore plus/minus sign
MAE = average of all absolute errors
```

Example:

| Real | Predicted | Absolute error |
|---:|---:|---:|
| 30 cm | 31 cm | 1 cm |
| 20 cm | 18 cm | 2 cm |
| 40 cm | 41 cm | 1 cm |

MAE:

```text
(1 + 2 + 1) / 3 = 1.33 cm
```

Why MAE is important here:

- It is easy to explain.
- It is in centimeters.
- The paper uses it as the main result.

Our key MAE:

```text
0.633 cm on non-occluded fish
```

Meaning:

> On clear fish images, the model is wrong by about 0.63 cm on average.

### RMSE

RMSE means **Root Mean Squared Error**.

Layman version:

> Similar to MAE, but it punishes big mistakes more strongly.

Why:

- A few very bad predictions increase RMSE a lot.
- If RMSE is much larger than MAE, it means there are some large outlier errors.

Our non-occluded:

```text
MAE  = 0.633 cm
RMSE = 1.027 cm
```

This means most errors are small, but some examples are harder and create larger mistakes.

### MAPE

MAPE means **Mean Absolute Percentage Error**.

Layman version:

> On average, what percent wrong is the model compared with the true fish length?

Example:

```text
real fish = 50 cm
error = 1 cm
percentage error = 2%
```

Our non-occluded MAPE:

```text
1.96%
```

Meaning:

> On clear images, predictions are about 2% wrong on average.

### Bias

Bias tells whether the model tends to overpredict or underpredict.

Layman version:

> Does the model usually guess too big or too small?

If bias is:

| Bias | Meaning |
|---:|---|
| positive | model predicts too large on average |
| negative | model predicts too small on average |
| near zero | no strong direction |

Our full test bias:

```text
0.035 cm
```

Meaning:

> The model is almost not biased overall.

### R2

R2 is a statistical score showing how much variation the model explains.

Layman version:

> How well do the predictions follow the real length pattern?

R2 is usually:

| R2 | Meaning |
|---:|---|
| 1.0 | perfect predictions |
| 0.0 | no better than predicting average |
| negative | worse than predicting average |

Our full test R2:

```text
0.947
```

Meaning:

> The model explains most of the variation in fish length.

## 13. Model logic in very plain words

The model works like this:

1. Take a full fish image.
2. Use annotation information to identify one fish.
3. Use the mask and bounding box to crop/focus on that fish.
4. Resize the crop to a fixed size.
5. Give the crop to MobileNetV2.
6. MobileNetV2 extracts visual features.
7. Add bbox information as extra numerical clues.
8. The regression head predicts length in centimeters.
9. During training, compare prediction with real measured length.
10. Update model weights to reduce the error.

Deep learning intuition:

> The model is not explicitly measuring pixels with a ruler. It learns a mapping from fish appearance and crop geometry to real length by seeing many examples with known measured lengths.

## 14. Technical stack and specs

### Local machine

Used for:

- editing files,
- writing project code,
- keeping copied results,
- documentation.

Local path:

```text
C:\Users\Shahman\Desktop\SEM3\AreaSeminar
```

### Server

Used for:

- downloading dataset,
- preprocessing,
- crop generation,
- GPU training,
- final evaluation.

Server:

```text
baltic.informatik.uni-rostock.de
```

User:

```text
sb2597
```

Server project:

```text
/home/sb2597/autofish_baseline_repro
```

### GPU

```text
NVIDIA RTX 5000 Ada Generation
```

Why GPU matters:

> Training deep neural networks on images is much faster on a GPU than on a CPU.

### Programming tools

| Tool | Used for |
|---|---|
| Python | Main programming language |
| PyTorch | Model training and GPU computation |
| Torchvision | MobileNetV2 and image transforms |
| Pandas | CSV tables and result summaries |
| NumPy | numerical operations |
| Pillow | image loading/cropping |
| pycocotools | COCO-style annotation/mask handling |
| Hugging Face tools | dataset download |
| SSH/SCP | server connection and file transfer |

## 15. Why our result can be different from paper

Even with the same method, exact equality is not guaranteed.

Reasons:

| Reason | Layman explanation |
|---|---|
| Random seed / initialization | The model starts from slightly different random values |
| GPU nondeterminism | Some GPU operations are not exactly repeatable |
| Library versions | PyTorch/torchvision versions can behave slightly differently |
| Crop implementation | Our crop logic is close, but may not be byte-identical to paper code |
| Leakage cleanup | We removed one cross-split duplicate annotation |
| Best checkpoint selection | Best epoch can vary |

Important point:

> Our non-occluded score is extremely close to the paper, so the reproduction is valid.

## 16. Professor-style questions and answers

### Q: What exactly did you reproduce?

We reproduced the AutoFish paper's REG MobileNetV2 baseline for fish length estimation. The model predicts fish length in centimeters from masked fish crops plus bounding-box information.

### Q: Why reproduce the baseline first?

Because later improvements must be compared against a trusted reference. If the baseline pipeline is wrong, any later comparison with foundation models would be unreliable.

### Q: What is the input to the model?

The model receives a cropped fish image and normalized bounding-box information. The crop focuses on one annotated fish.

### Q: What is the output?

One number: predicted fish length in centimeters.

### Q: Is this classification or regression?

Regression, because the output is a continuous numeric value, not a class label.

### Q: Why use MobileNetV2?

Because it is the architecture used by the paper's REG baseline. It is efficient and can extract useful visual features from fish crops.

### Q: Why use pretrained ImageNet weights?

Pretrained weights give the model general visual knowledge before fish-specific training. This usually improves performance and reduces training time.

### Q: What is MAE and why is it the main metric?

MAE is mean absolute error. It tells the average centimeter mistake. It is easy to interpret and directly matches the paper's reported metric.

### Q: Did your result match the paper?

Yes. The paper reports 0.62 cm MAE on non-occluded Set1+Set2. Our reproduction achieved 0.633 cm, which is extremely close.

### Q: Why is your occluded result better than the paper?

Possible reasons include preprocessing differences, random training variation, library version differences, and our removal of one cross-split duplicate annotation. Since the main non-occluded result matches closely, the baseline reproduction is still successful.

### Q: What is data leakage?

Data leakage happens when test information accidentally appears in training. It makes results unfairly good. We checked for fish ID leakage and removed one duplicate issue.

### Q: Why split by groups?

Group-level splitting prevents very similar fish observations from being mixed between train and test. This makes evaluation more realistic.

### Q: Why create crops instead of using full images?

Crops reduce background noise and focus the model on the target fish. This follows the paper baseline.

### Q: What does best validation epoch mean?

During training, we save the model that performs best on validation data. The best validation model was from epoch 153, and we used it for final test evaluation.

### Q: What files prove the run completed?

On the server:

```text
/home/sb2597/autofish_baseline_repro/runs/baseline_official/best.pt
/home/sb2597/autofish_baseline_repro/runs/baseline_official/history.csv
/home/sb2597/autofish_baseline_repro/runs/baseline_official/test_metrics.json
/home/sb2597/autofish_baseline_repro/runs/baseline_official/test_metrics.predictions.csv
```

Locally copied:

```text
runs/baseline_official/history.csv
runs/baseline_official/test_metrics.json
runs/baseline_official/test_metrics.predictions.csv
```

### Q: What is the final conclusion?

The Q1 baseline reproduction is complete. The model reproduced the paper's main non-occluded result very closely and achieved strong performance on the occluded subset.

### Q: Why did frozen DINOv2 perform worse than MobileNetV2?

Because DINOv2 was used as a frozen general feature extractor, while MobileNetV2 was fully trained for this exact fish-length task. DINOv2 has strong general image understanding, but fish length estimation needs precise scale, body geometry, and crop-specific cues. If the encoder is frozen, it cannot adapt its visual features to this exact measurement task.

This is a useful result:

```text
Foundation model features alone are not automatically better for every scientific measurement task.
```

The next fairer test is to fine-tune part of DINOv2 or train a stronger task-specific regressor on top of cached DINOv2 features.

### Q: Why did CLIP perform better than DINOv2 here?

CLIP's visual features transferred better to this crop-based fish-length task. A likely reason is that CLIP's image encoder has learned broad object-level visual representations from image-text training, and those features were more useful for fish crops than the frozen DINOv2 features in this setup. This does not prove CLIP is always better; it means CLIP was better under our same split, same crops, same head, and same metrics.

### Q: Did CLIP beat the reproduced MobileNetV2 baseline?

No. The best CLIP result was the last-block fine-tuned version with `0.842 cm` non-occluded MAE and `0.958 cm` full-test MAE. The MobileNetV2 baseline still achieved `0.633 cm` non-occluded MAE and `0.771 cm` full-test MAE. So CLIP is the best foundation-model experiment so far, but not the best overall model.

## 17. One-minute explanation for presentation

We reproduced the AutoFish paper's fish length estimation baseline. The task is to predict the real length of a fish in centimeters from underwater images. We used the official AutoFish dataset, created fish crops using the annotations, and trained the paper's REG MobileNetV2 regression model on the university GPU server. The data was split by official groups into train, validation, and test sets to avoid unfair overlap. We also checked for fish identity leakage and removed one duplicate issue. The paper reports 0.62 cm MAE on non-occluded fish, and our reproduction achieved 0.633 cm, which is very close. On occluded fish, the paper reports 1.38 cm MAE and our run achieved 0.909 cm. This means the baseline reproduction is successful and can now be used as the reference for later foundation-model experiments.
