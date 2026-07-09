# Area Seminar Full Report: AutoFish Fish Length Estimation

## Abstract

This report documents the current state of the Area Seminar project on automatic fish length estimation using the AutoFish dataset. The first objective was to reproduce the baseline from Bengtson et al., "AutoFish: Dataset and Benchmark for Fine-grained Analysis of Fish", WACVW 2025 / arXiv:2501.03767. The reproduced baseline is the paper's REG MobileNetV2 model, which predicts fish length in centimeters from cropped fish images and bounding-box information.

The baseline reproduction was successful. The paper reports `0.62 cm` MAE on the non-occluded Set1+Set2 subset, and our reproduction achieved `0.633 cm`, which is extremely close. The paper reports `1.38 cm` MAE on the occluded All subset, and our reproduction achieved `0.909 cm`.

After reproducing the baseline, we tested modern vision foundation model encoders under the same dataset split, crop pipeline, regression target, and evaluation metrics. The best foundation-model result so far is CLIP ViT-B/32 with the last visual transformer block fine-tuned, achieving `0.842 cm` non-occluded MAE, `1.074 cm` occluded MAE, and `0.958 cm` full-test MAE. This improves over frozen CLIP, but it still does not beat the reproduced MobileNetV2 baseline.

## 1. Project Motivation

The practical problem is simple to explain: given underwater images of fish, can a machine learning model estimate the real fish length in centimeters?

This matters because manual fish measurement is slow, labor-intensive, and not scalable. If fish length can be estimated reliably from images, it can support ecological monitoring, aquaculture, marine biology, and automated population analysis.

The seminar task has two stages:

1. Reproduce the paper baseline correctly.
2. Replace or adapt the image encoder with different modern encoders or vision foundation models and test whether they improve the fish length estimation result.

The first stage is essential. If the baseline pipeline is not trustworthy, then any later comparison against foundation models would be unfair.

## 2. Target Paper

The target paper is:

```text
Bengtson et al.,
"AutoFish: Dataset and Benchmark for Fine-grained Analysis of Fish",
WACVW 2025 / arXiv:2501.03767
```

The paper introduces the AutoFish dataset and several benchmark tasks. The part reproduced in this project is the fish length estimation baseline.

The baseline we reproduced is called REG, meaning regression. Regression means the model predicts a continuous numeric value. In this project, the output is fish length in centimeters.

## 3. Research Objective

The professor's main instruction was:

```text
First reproduce the baseline result, then test different encoders or foundation model variants and compare their results.
```

In more formal terms, the research question is:

> Can modern vision foundation models improve fish length estimation compared with the reproduced AutoFish REG MobileNetV2 baseline?

The project therefore needs both engineering correctness and scientific comparison. We must keep the data split, preprocessing, target variable, and metrics consistent, otherwise the model comparison would not be meaningful.

## 4. Local and Server Setup

There are two project locations:

| Place | Path | Purpose |
|---|---|---|
| Local Windows project | `C:\Users\Shahman\Desktop\SEM3\AreaSeminar` | Code editing, documentation, copied result artifacts |
| University server project | `/home/sb2597/autofish_baseline_repro` | Dataset processing, GPU training, evaluation |

Server access:

```text
sb2597@baltic.informatik.uni-rostock.de
```

GPU used:

```text
NVIDIA RTX 5000 Ada Generation, about 32 GB VRAM
```

The server project was kept separate from other work. A dedicated project folder and isolated Python environment were created so the AutoFish work would not modify unrelated server files.

## 5. Technical Stack

Main tools:

| Tool | Purpose |
|---|---|
| Python | Main programming language |
| PyTorch | Neural network training and GPU computation |
| Torchvision | MobileNetV2 model and image transforms |
| OpenCLIP | CLIP ViT-B/32 encoder |
| PyTorch Hub | DINOv2 loading |
| Pandas | CSV processing and result tables |
| NumPy | Numerical operations |
| Pillow | Image loading and crop handling |
| pycocotools | Mask/annotation processing |
| Hugging Face tools | Dataset download |
| SSH/SCP | Server connection and copying results |

Important local files:

| File/folder | Purpose |
|---|---|
| `src/autofish_vfm/data.py` | Dataset loading and image transform pipeline |
| `src/autofish_vfm/models.py` | MobileNetV2, DINOv2, and CLIP regression models |
| `src/autofish_vfm/train_baseline.py` | Training loop |
| `src/autofish_vfm/evaluate.py` | Final test evaluation |
| `configs/` | Separate config for each experiment |
| `runs/` | Copied result artifacts from server |
| `README.md` | Short project guide |
| `PROJECT_SUMMARY_GUIDE.md` | Layman and technical summary guide |
| `AREA_SEMINAR_FULL_REPORT.md` | This full report |

## 6. Dataset

Dataset:

```text
Hugging Face: vapaau/autofish
```

Server raw dataset location:

```text
/home/sb2597/autofish_baseline_repro/data/raw/autofish
```

Processed data:

```text
/home/sb2597/autofish_baseline_repro/data/processed/index.csv
/home/sb2597/autofish_baseline_repro/data/processed/crops
```

Processed dataset summary:

| Item | Count |
|---|---:|
| Rows / annotations | 18,157 |
| Images | 1,500 |
| Unique fish | 454 |
| Groups | 25 |
| Missing crops | 0 |
| Fish leakage across splits | 0 |

Layman explanation:

> The dataset contains underwater fish images plus labels that tell us where the fish is and how long it really is. We use those labels to train a model to predict fish length from images.

## 7. Split Strategy

The official group split was used:

| Split | Groups |
|---|---|
| Train | 2, 3, 4, 5, 7, 8, 9, 12, 13, 15, 16, 18, 19, 23, 24 |
| Validation | 1, 6, 11, 17, 25 |
| Test | 10, 14, 20, 21, 22 |

Set mapping:

| Image number range | Set name | Meaning |
|---|---|---|
| `00001-00020` | Set1 | non-occluded |
| `00021-00040` | Set2 | non-occluded |
| `00041-00060` | All | occluded/crowded |

Layman explanation:

> Train data is like practice questions, validation data is like a mock exam, and test data is the final exam. The model learns from train, we choose the best checkpoint using validation, and we report final results on test.

Why group-level splitting matters:

> If very similar images or the same fish appear in both train and test, the model may look better than it really is. Group-level splitting helps keep the final exam fair.

## 8. Leakage Handling

Data leakage means test information accidentally appears during training. This is a serious issue because it can make results unfairly strong.

During preprocessing, one cross-split duplicate fish ID was found:

```text
fish_id=113
```

It appeared once in group 5 and repeatedly in group 22. To avoid fish identity leakage, the singleton duplicate annotation was removed by default. This is why our non-occluded test count is `1,879` instead of exactly `1,880`.

Technical meaning:

> We prioritized a clean evaluation split over blindly keeping a duplicated annotation that could contaminate train/test separation.

## 9. Preprocessing and Crop Generation

The model does not train directly on the whole underwater image. Instead, one crop is created for each annotated fish.

Main preprocessing:

- build a clean annotation index,
- infer Set1/Set2/All from image number,
- use bounding boxes and masks,
- generate fish crops,
- resize crops to the configured image size,
- normalize images according to the encoder requirements,
- optionally include normalized bounding-box features as numerical input.

Layman explanation:

> Instead of asking the model to search the whole picture, we cut out the fish area and give that focused crop to the model. This reduces background noise and follows the paper baseline.

## 10. Baseline Model: REG MobileNetV2

The reproduced paper baseline is:

```text
REG MobileNetV2
```

Architecture:

```text
fish crop image -> MobileNetV2 encoder -> image feature vector
bounding-box numbers -> joined with image features
joined features -> regression head -> predicted length in cm
```

Important details:

- MobileNetV2 is pretrained on ImageNet.
- The model predicts one continuous value.
- The target is fish length in centimeters.
- The paper's main metric is MAE.
- Training used the official split and 200 epochs.
- Best checkpoint was selected using validation MAE.

Layman explanation:

> MobileNetV2 works like the visual brain. It looks at the fish crop and extracts shape and visual clues. The regression head works like the final ruler. It turns those clues into one number: the predicted length.

## 11. Metrics

### MAE

MAE means Mean Absolute Error.

```text
MAE = average absolute centimeter mistake
```

If the true length is `30 cm` and the model predicts `31 cm`, the absolute error is `1 cm`.

MAE is the main metric because it is easy to interpret and directly matches the paper.

### RMSE

RMSE means Root Mean Squared Error. It punishes large mistakes more strongly than MAE.

If RMSE is much larger than MAE, it usually means a few examples have large errors.

### MAPE

MAPE means Mean Absolute Percentage Error.

It describes the average error as a percentage of the true fish length.

### Bias

Bias tells whether the model tends to predict too large or too small.

| Bias | Meaning |
|---:|---|
| Positive | overpredicts on average |
| Negative | underpredicts on average |
| Near zero | no strong direction |

### R2

R2 measures how well predictions follow the variation in true lengths.

| R2 | Meaning |
|---:|---|
| 1.0 | perfect |
| 0.0 | no better than predicting the average |
| negative | worse than predicting the average |

## 12. Completed Baseline Result

Paper target:

| Model | Paper non-occluded MAE | Paper occluded MAE |
|---|---:|---:|
| REG MobileNetV2 | 0.62 cm | 1.38 cm |

Our reproduced baseline:

| Subset | MAE cm | RMSE cm | MAPE % | Bias cm | R2 | n |
|---|---:|---:|---:|---:|---:|---:|
| Full test | 0.771 | 1.268 | 2.411 | 0.035 | 0.947 | 3,759 |
| Non-occluded Set1+Set2 | 0.633 | 1.027 | 1.960 | 0.080 | 0.965 | 1,879 |
| Occluded All | 0.909 | 1.470 | 2.862 | -0.009 | 0.929 | 1,880 |

Comparison:

| Model | Paper non-occluded MAE | Our non-occluded MAE | Paper occluded MAE | Our occluded MAE | Our full test MAE |
|---|---:|---:|---:|---:|---:|
| REG MobileNetV2 | 0.62 cm | 0.633 cm | 1.38 cm | 0.909 cm | 0.771 cm |

Conclusion:

> The baseline reproduction is successful. The non-occluded paper result was `0.62 cm`, and our result was `0.633 cm`, a very close match.

## 13. Foundation Model Experiments

After reproducing the baseline, we tested DINOv2 and CLIP encoders.

The comparison was designed to keep everything else the same:

- same dataset,
- same train/val/test split,
- same crop pipeline,
- same length target,
- same MAE/RMSE/MAPE/bias/R2 evaluation,
- same bounding-box feature idea,
- only the image encoder/fine-tuning strategy changes.

This makes the comparison scientifically cleaner.

## 14. Main Experiment Table

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

The paper columns say "Not reported" for DINOv2 and CLIP because those are our new experiments, not direct paper baseline rows.

## 15. DINOv2 Results and Interpretation

### Frozen DINOv2

Result:

| Subset | MAE cm |
|---|---:|
| Non-occluded | 1.690 |
| Occluded | 1.786 |
| Full test | 1.738 |

Interpretation:

> Frozen DINOv2 did not beat MobileNetV2. A frozen foundation model can provide general visual features, but fish length estimation needs precise task-specific geometry. Because the encoder was frozen, DINOv2 could not adapt its features to this measurement task.

### Full DINOv2 fine-tuning, encoder LR 1e-5

Result:

| Subset | MAE cm |
|---|---:|
| Non-occluded | 1.636 |
| Occluded | 1.919 |
| Full test | 1.778 |

Interpretation:

> Non-occluded performance improved slightly compared with frozen DINOv2, but occluded performance became worse. This suggests unstable adaptation.

### Full DINOv2 fine-tuning, encoder LR 1e-6

Result:

| Subset | MAE cm |
|---|---:|
| Non-occluded | 2.075 |
| Occluded | 2.189 |
| Full test | 2.132 |

Interpretation:

> Lowering the whole-encoder learning rate did not solve the problem. It made the model more conservative, but performance worsened.

### DINOv2 last-block fine-tuning

Result:

| Subset | MAE cm |
|---|---:|
| Non-occluded | 1.340 |
| Occluded | 1.537 |
| Full test | 1.439 |

Interpretation:

> Controlled partial fine-tuning was clearly better than frozen DINOv2 and full DINOv2 fine-tuning. However, it still did not beat MobileNetV2.

## 16. CLIP Results and Interpretation

### Frozen CLIP ViT-B/32

Result:

| Subset | MAE cm | RMSE cm | MAPE % | Bias cm | R2 | n |
|---|---:|---:|---:|---:|---:|---:|
| Full test | 1.002 | 1.367 | 3.148 | 0.248 | 0.938 | 3,759 |
| Non-occluded Set1+Set2 | 0.898 | 1.174 | 2.840 | 0.307 | 0.954 | 1,879 |
| Occluded All | 1.106 | 1.536 | 3.457 | 0.189 | 0.922 | 1,880 |

Interpretation:

> Frozen CLIP was much stronger than frozen DINOv2 in this project. It still did not beat MobileNetV2, but it became the most promising foundation-model direction.

### CLIP last visual block fine-tuning

Result:

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

> Partial CLIP fine-tuning helped consistently. The improvement is modest, but it appears across all subsets. CLIP was the strongest foundation-model result, but it was later surpassed by ConvNeXt-Tiny among the non-baseline encoders.

### ConvNeXt-Tiny

Result:

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

Interpretation:

> ConvNeXt-Tiny is the best non-baseline encoder result so far. It improves over CLIP last-block fine-tuning on non-occluded, occluded, and full-test MAE. However, it still does not beat the reproduced MobileNetV2 baseline.

## 17. Why MobileNetV2 Still Wins

The strongest model is still the reproduced MobileNetV2 baseline:

| Model | Non-occluded MAE | Occluded MAE | Full test MAE |
|---|---:|---:|---:|
| REG MobileNetV2 | 0.633 cm | 0.909 cm | 0.771 cm |
| Best non-baseline encoder: ConvNeXt-Tiny | 0.814 cm | 1.014 cm | 0.914 cm |

Possible reasons:

- MobileNetV2 was fully trained for this exact fish length task.
- The AutoFish REG baseline may be well matched to the crop and bbox setup.
- Foundation model features are general, not automatically optimized for precise metric regression.
- Fish length depends on fine geometry and scale, not only semantic object understanding.
- Stronger non-baseline encoders helped, but still did not surpass the reproduced MobileNetV2 baseline.
- More careful learning-rate schedules, augmentation choices, or multi-stage training may be needed.

Layman explanation:

> A large general model may know a lot about images, but this task is like measuring with a ruler. The older MobileNetV2 baseline was trained directly for that ruler task, so it still performs best.

## 18. Why CLIP Beat DINOv2 in Our Experiments

CLIP transferred better than DINOv2 under our exact setup.

Possible reasons:

- CLIP's image encoder may provide more useful object-level crop features for this dataset.
- CLIP ViT-B/32 has a 512-dimensional image feature representation that worked well with the regression head.
- CLIP normalization and preprocessing may have aligned well with the fish crop images.
- DINOv2 may need a different feature pooling strategy, larger model, stronger head, or more careful fine-tuning.
- DINOv2 full fine-tuning was unstable, while CLIP partial fine-tuning gave a consistent improvement.

Important caution:

> This does not prove CLIP is universally better than DINOv2. It only means CLIP performed better in our controlled experiments so far.

## 19. Limitations

Current limitations:

- Experiments are mostly single-seed runs.
- We have not yet performed statistical repeats.
- We did not tune every hyperparameter extensively.
- Foundation models were tested in a small number of practical variants.
- Some preprocessing may differ slightly from the paper implementation.
- Occluded subset differences from the paper may be affected by preprocessing, leakage cleanup, random seed, or library version differences.
- We have not yet performed detailed error analysis by species, length range, or occlusion severity.

## 20. Future Work

Good next steps:

| Direction | Why it matters |
|---|---|
| Multi-seed repeats | Check whether results are stable or random |
| CLIP learning-rate sweep | CLIP is promising, so tune it more carefully |
| CLIP two-stage training | Train head first, then last block, maybe with warmup |
| ConvNeXt / EfficientNet | Strong supervised encoders may beat MobileNetV2 |
| Stronger regression head | The encoder may be good but the head may be too simple |
| Error analysis | Identify where models fail: species, length ranges, occlusion |
| Crop/bbox ablation | Test how much bbox and masks contribute |
| Ensemble or prediction averaging | May reduce regression error |
| Multi-seed final table | More defensible seminar/report result |

Recommended immediate next experiment:

```text
ConvNeXt-Tiny or EfficientNet with a more stable learning-rate schedule, because ConvNeXt is now the strongest non-baseline encoder.
```

Alternative strong next experiment:

```text
EfficientNet-B0/B2 pretrained on ImageNet, fully fine-tuned with bbox input.
```

Why EfficientNet is still attractive:

> MobileNetV2 is a supervised ImageNet CNN and works best so far. ConvNeXt improved over CLIP, so another supervised ImageNet family such as EfficientNet may be worth testing.

## 21. Professor Questions and Answers

### Q: What exactly did you reproduce?

We reproduced the AutoFish paper's REG MobileNetV2 fish length estimation baseline using fish crops, bounding-box features, and a regression head predicting length in centimeters.

### Q: Did the reproduction match the paper?

Yes. The paper reports `0.62 cm` MAE on non-occluded Set1+Set2, and our run achieved `0.633 cm`.

### Q: Why is the occluded result better than the paper?

Possible reasons include preprocessing differences, random seed, library versions, checkpoint selection, and our explicit removal of one cross-split duplicate annotation. Since the key non-occluded result closely matches the paper, the reproduction is still valid.

### Q: What is leakage?

Leakage means information from the test set accidentally enters training. It is like seeing final exam answers before the exam. We checked fish identity leakage and removed one duplicate issue.

### Q: Why include bounding-box coordinates?

The paper baseline includes bbox information. Bbox values provide geometric cues about where and how large the fish appears in the image.

### Q: What is a foundation model?

A foundation model is a large pretrained model trained on broad data, intended to transfer to many downstream tasks. DINOv2 and CLIP are examples.

### Q: Why did frozen DINOv2 perform poorly?

Frozen DINOv2 could not adapt to the precise fish-length regression task. General visual features are not automatically enough for centimeter-level measurement.

### Q: Why did partial fine-tuning help?

Partial fine-tuning allows the model to adapt some high-level visual features while preserving most pretrained knowledge. It was more stable than changing the entire encoder.

### Q: Did any non-baseline encoder beat MobileNetV2?

No. ConvNeXt-Tiny is the best non-baseline encoder result so far, but MobileNetV2 still has lower MAE.

### Q: What is the final conclusion so far?

The baseline reproduction is complete and successful. ConvNeXt-Tiny is the strongest non-baseline encoder so far, CLIP is the strongest foundation-model direction, and the reproduced MobileNetV2 baseline remains the best overall model.

## 22. Appendix: Important Commands

Environment setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Data preprocessing:

```bash
python scripts/build_autofish_index.py --raw-dir data/raw/autofish --out data/processed/index.csv --splits-out data/processed/splits.json
python scripts/make_crops.py --raw-dir data/raw/autofish --index data/processed/index.csv --out-dir data/processed/crops
python scripts/check_processed.py --index data/processed/index.csv --crops-dir data/processed/crops
```

Baseline training:

```bash
python -m src.autofish_vfm.train_baseline --config configs/baseline_official.json --index data/processed/index.csv --crops-dir data/processed/crops --out-dir runs/baseline_official
```

Evaluation:

```bash
python -m src.autofish_vfm.evaluate --checkpoint runs/baseline_official/best.pt --config configs/baseline_official.json --index data/processed/index.csv --crops-dir data/processed/crops --out runs/baseline_official/test_metrics.json
```

CLIP last-block experiment:

```bash
python -m src.autofish_vfm.train_baseline --config configs/clip_vitb32_lastblock_from_frozen.json --index data/processed/index.csv --crops-dir data/processed/crops --out-dir runs/clip_vitb32_lastblock_from_frozen
python -m src.autofish_vfm.evaluate --checkpoint runs/clip_vitb32_lastblock_from_frozen/best.pt --config configs/clip_vitb32_lastblock_from_frozen.json --index data/processed/index.csv --crops-dir data/processed/crops --out runs/clip_vitb32_lastblock_from_frozen/test_metrics.json
```

## 23. Appendix: Important Result Folders

| Folder | Meaning |
|---|---|
| `runs/baseline_official` | Reproduced REG MobileNetV2 baseline |
| `runs/dinov2_vits14_frozen` | Frozen DINOv2 experiment |
| `runs/dinov2_vits14_finetune_lr1e5` | Full DINOv2 fine-tune with encoder LR 1e-5 |
| `runs/dinov2_vits14_finetune_lr1e6` | Full DINOv2 fine-tune with encoder LR 1e-6 |
| `runs/dinov2_vits14_lastblock_from_frozen` | Partial DINOv2 last-block fine-tune |
| `runs/clip_vitb32_frozen` | Frozen CLIP experiment |
| `runs/clip_vitb32_lastblock_from_frozen` | Best foundation-model experiment |
| `runs/convnext_tiny_official` | Best non-baseline encoder experiment so far |

## 24. Final Conclusion

The Area Seminar project has completed its first major objective: reproducing the AutoFish REG MobileNetV2 baseline. The reproduced non-occluded result, `0.633 cm` MAE, closely matches the paper's `0.62 cm` MAE and validates the dataset, preprocessing, training, and evaluation pipeline.

The second objective has also started meaningfully. Several foundation-model and supervised encoder variants were tested. DINOv2 did not beat the baseline, although partial DINOv2 fine-tuning improved over frozen DINOv2. CLIP performed much better than DINOv2. ConvNeXt-Tiny became the best non-baseline encoder result so far with `0.814 cm` non-occluded MAE and `0.914 cm` full-test MAE.

However, the reproduced MobileNetV2 baseline remains the best overall model. The current scientific conclusion is:

```text
Foundation models are promising, especially CLIP, but they do not automatically outperform a well-trained task-specific baseline for precise fish length regression.
```
