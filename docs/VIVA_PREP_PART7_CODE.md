# Viva Prep — Part 7: Code Walkthrough

Builds on Parts 1–6. **Every source file in this project already has extensive inline comments**
explaining exactly what each function does, line by line, with worked examples — see
`src/autofish_vfm/*.py` and `scripts/{build_autofish_index,make_crops,download_autofish,check_processed}.py`
directly for that level of detail. This document is the **higher-level map**: where data enters,
where it moves, where tensors change shape, where gradients flow, where results get saved — tying
the files together rather than repeating their internal comments.

---

## 1. The complete file map, in execution order

```
STAGE 0 — Get the data (run once)
  scripts/download_autofish.py
      → data/raw/autofish/{images/, annotations.json}

STAGE 1 — Build the index (run once)
  scripts/build_autofish_index.py
      reads:  data/raw/autofish/annotations.json
      writes: data/processed/index.csv        (18,157 rows, one per fish-in-photo)
              data/processed/splits.json       (official group lists)
              data/processed/exclusions.json   (the fish-113 leakage fix, logged)

STAGE 2 — Build the crops (run once, shared by every experiment)
  scripts/make_crops.py
      reads:  data/processed/index.csv
      writes: data/processed/crops/*.png       (18,157 masked, square, 224x224 PNGs)

STAGE 2.5 — Sanity check (optional, run anytime)
  scripts/check_processed.py
      reads:  index.csv + crops/ folder
      verifies: no missing crops, zero fish leakage

STAGE 3 — Train ONE experiment
  src/autofish_vfm/train_baseline.py  (regression)  OR  train_classifier.py  (species)
      reads:  configs/<experiment>.json, data/processed/{index.csv, crops/}
      calls:  data.py's CropDataset, models.py's build_model(), metrics.py's regression_metrics()
      writes: runs/<experiment>/{config.json, history.csv, last.pt, best.pt}

STAGE 4 — Evaluate ONE experiment, once
  src/autofish_vfm/evaluate.py  (regression)  OR  evaluate_classifier.py  (species)
      reads:  runs/<experiment>/best.pt, the same config.json
      writes: runs/<experiment>/test_metrics.json
              runs/<experiment>/test_metrics.predictions.csv   (per-fish predictions)

STAGE 5 — Analysis (no GPU, no retraining — reads ONLY the files STAGE 4 produced)
  scripts/error_analysis.py          → results/error_analysis/*.csv
  scripts/make_qualitative_figures.py → results/qualitative/*.png
  scripts/make_result_charts.py       → results/figures/*.png

STAGE 6 — Documents (reads STAGE 4/5 outputs, produces human-facing deliverables)
  scripts/build_poster.py       → poster/AutoFish_A3_poster.html
  scripts/build_word_report.py  → docs/AutoFish_Final_Report.docx
  scripts/build_ode_docx.py     → docs/ODE_REPORT.docx
```

**Why this strict stage ordering matters for the viva:** it's the mechanism that made the project's
"no-repeat" rule possible — every GPU-expensive step (Stage 3) happens exactly once per experiment,
and every downstream document (Stage 5–6) is built purely from cheap CSV/JSON files already on disk,
never by re-touching the model or the GPU.

---

## 2. Where a tensor's SHAPE changes, end to end (for one fish, batch size 1 for clarity)

| Stage | Shape / form | File |
|---|---|---|
| Raw photo | `2464 × 2056 × 3` (H×W×C, uint8 pixels 0–255) | disk, from the paper's camera |
| Polygon → mask | `2464 × 2056` (single channel, 0 or 255) | `make_crops.py: mask_from_polygons()` |
| Masked photo | `2464 × 2056 × 3` (background blacked out) | `make_crops.py: main()` |
| Square crop | `side × side × 3` (side = fish's longer mask dimension) | `make_crops.py: crop_with_padding()` |
| Resized PNG (saved to disk) | `224 × 224 × 3` (uint8) | `make_crops.py`, final `.resize()` |
| Loaded + `ToTensor()` | `3 × 224 × 224` (float32, 0.0–1.0, **channels-first now**) | `data.py: CropDataset.__getitem__` |
| + Normalize | `3 × 224 × 224` (float32, roughly zero-mean) | `data.py`, same method |
| Batched (DataLoader, batch=32) | `32 × 3 × 224 × 224` | `train_baseline.py`, the `for batch in train_loader` loop |
| bbox tensor | `32 × 4` | `data.py`, built alongside the image tensor |
| Through the encoder | `32 × N` (N = 1280 for MobileNetV2, 384 for DINOv2, 512 for CLIP, 768 for ConvNeXt, 1280 for EfficientNet-B0) | `models.py`, each `<Encoder>Regressor.forward()` |
| Concat bbox | `32 × (N+4)` | `models.py`, `torch.cat([features, bbox], dim=1)` |
| Through the head | `32 × 1` (regression) or `32 × 7` (classification) | `models.py`, `make_regression_head()` output |
| Loss | scalar (single number, the batch's average error) | `train_baseline.py`, `criterion(pred, target)` |

**The one line that is the entire "why does the bbox get added here and not earlier" answer:**
`features = torch.cat([features, bbox], dim=1)` — this happens **after** the encoder, not before,
because the encoder is a pretrained network that expects a pure image input; the bbox is injected
only at the point where the model transitions from "generic image understanding" (encoder) to
"task-specific decision" (head).

---

## 3. Where gradients flow, and where the optimizer touches the model

```
loss = criterion(pred, target)              # a single scalar number
loss.backward()                              # PyTorch walks BACKWARD through every operation
                                              # that produced `pred`, computing d(loss)/d(weight)
                                              # for every trainable weight along the way —
                                              # this reaches: head weights, AND encoder weights
                                              # IF the encoder is not frozen (requires_grad=True)
optimizer.step()                             # Adam reads every weight's .grad (just computed)
                                              # and updates that weight — but ONLY for weights
                                              # that were included in the optimizer's parameter
                                              # list (see build_optimizer() in train_baseline.py,
                                              # which explicitly filters to `requires_grad=True`
                                              # params only)
```

**Why a frozen encoder's weights never change, mechanically:** `param.requires_grad = False` (set in
`models.py`'s encoder classes) tells PyTorch's autograd engine to **skip** computing gradients for
those specific tensors entirely during `.backward()` — so even though `loss.backward()` still runs
through the frozen encoder's operations (its outputs are still needed to compute the loss), no
`.grad` is ever populated on its weights, and `optimizer.step()` has nothing to update there even if
it tried.

---

## 4. Where results get saved, and by which exact call

| File written | Written by | Contains |
|---|---|---|
| `runs/<exp>/config.json` | `train_baseline.py`, at the very start | A copy of the exact settings used — provenance |
| `runs/<exp>/last.pt` | `train_baseline.py`, end of every epoch | Most recent weights (resume safety net) |
| `runs/<exp>/best.pt` | `train_baseline.py`, only when validation improves | The ONE checkpoint ever evaluated on test |
| `runs/<exp>/history.csv` | `train_baseline.py`, once, at the very end | Every epoch's train loss + validation metrics |
| `runs/<exp>/test_metrics.json` | `evaluate.py`, run separately, once | The headline MAE/RMSE/MAPE/bias/R², 3 subsets |
| `runs/<exp>/test_metrics.predictions.csv` | `evaluate.py`, same run | Per-fish predicted vs. true length — raw material for all later analysis |

**The critical design point to be ready to explain:** `train_baseline.py` and `evaluate.py` are
**two separate scripts**, run separately, and `evaluate.py` is only ever pointed at `best.pt` (a
checkpoint chosen purely by validation performance). This physical separation is what makes it
structurally impossible for test-set performance to have influenced which checkpoint was kept.

---

## 5. `build_model(config)` — the one function that ties encoder choice to everything else

```python
# models.py
def build_model(config):
    model_name = config["model"]
    if model_name == "mobilenet_v2": return MobileNetV2Regressor(...)
    if model_name == "dinov2":       return DINOv2Regressor(...)
    if model_name == "convnext":     return ConvNeXtRegressor(...)
    if model_name == "efficientnet": return EfficientNetRegressor(...)
    if model_name == "clip":         return CLIPRegressor(...)
    raise ValueError(f"Unsupported model: {model_name}")
```

**Why this single function matters for the "fair comparison" argument:** every one of
`train_baseline.py`, `evaluate.py`, `train_classifier.py`, `evaluate_classifier.py` calls this exact
same function with a config file's settings — there is no separate code path per encoder anywhere
else in the training/evaluation loop. Swapping MobileNetV2 for EfficientNet-B0 is **one changed
string** (`"model": "mobilenet_v2"` → `"model": "efficientnet"`) in a JSON file; every other line of
training/evaluation code executes identically regardless of which branch fires here.

---

## 6. `CropDataset` — the one class that both training scripts share

`train_baseline.py` and `train_classifier.py` both build their datasets from the **same**
`CropDataset` class (`data.py`) — the only difference is whether `label_map` is passed:

```python
CropDataset(..., label_map=None)          # -> target = length_cm  (regression, train_baseline.py)
CropDataset(..., label_map=LABEL_MAP)     # -> target = species index  (classification, train_classifier.py)
```

This is a deliberate reuse decision: the masking, cropping, bbox computation, and augmentation logic
is identical for both tasks — only the "answer key" differs, and that's controlled by a single
constructor argument rather than duplicated code.

---

## 7. Quick-fire viva Q&A — Part 7 (code)

**Q1: If I wanted to add a 6th encoder, what exactly would I need to change?**
Write a new `<Something>Regressor` class in `models.py` implementing the same
`forward((image, bbox)) -> length` contract, add one `if model_name == "...":` branch to
`build_model()`, and write a new config JSON — no changes needed anywhere else (data loading,
training loop, evaluation).

**Q2: Where exactly does the bounding box get combined with the image features?**
Inside each encoder class's `forward()` method, via `torch.cat([features, bbox], dim=1)` — after the
encoder produces its feature vector, before the regression/classification head.

**Q3: How does the code guarantee the test set never influences checkpoint selection?**
`train_baseline.py` only ever compares validation MAE to decide whether to overwrite `best.pt`; the
test set is never even loaded during training — it's only touched later, once, by the separate
`evaluate.py` script.

**Q4: Mechanically, why does a frozen encoder's weights never update?**
`requires_grad = False` is set on those parameters, so PyTorch's autograd never computes gradients
for them during `.backward()`, and the optimizer (which is built only from `requires_grad=True`
parameters) never touches them.

**Q5: What's the difference between how train_baseline.py and train_classifier.py compute their loss?**
train_baseline.py uses `nn.L1Loss()` (regression, comparing a predicted number to a true number);
train_classifier.py uses `nn.CrossEntropyLoss()` (classification, comparing 7 predicted class scores
to a true class index) — everything else about the training loop (batching, forward pass,
backward pass, optimizer step, checkpoint saving) is structurally identical.

**Q6: Where do the "no-repeat rule" analysis scripts (error_analysis.py etc.) get their data from —
do they ever touch the GPU?**
No — they read only the `test_metrics.predictions.csv` files already saved by `evaluate.py`; they
run on CPU, doing plain pandas/matplotlib work, and never reload a model.

**Q7: What would happen if you accidentally set augment=True for the validation dataset?**
Validation scores would become noisy and inconsistent between epochs (since ColorJitter randomizes
colors differently each time), making it unreliable for deciding which checkpoint is genuinely
"best" — this is exactly why `train_baseline.py` hard-codes `augment=False` for `val_ds`.

---

*Part 7 of the staged viva-prep series (76 Q&A total so far across Parts 1–7). Part 8 (common
confusions) continues next.*
