# ============================================================================
# train_classifier.py — the species-CLASSIFICATION training script.
#
# This is the "sister" of train_baseline.py: same dataset, same encoders,
# same overall training-loop shape — but instead of predicting a CONTINUOUS
# number (length in cm), it predicts a DISCRETE category (which of 7 species
# this fish is). Comparing this file to train_baseline.py side by side is a
# good way to see exactly what changes between a regression task and a
# classification task, and (just as informatively) what STAYS THE SAME.
#
# WHAT'S DIFFERENT FROM train_baseline.py:
#   - CropDataset is given a `label_map` (see data.py), which switches its
#     target from "true length in cm" to "true species index".
#   - The loss function is CrossEntropyLoss (classification) instead of
#     L1Loss (regression) — see the loss-function comment below for why.
#   - The best checkpoint is chosen by highest VALIDATION ACCURACY, instead
#     of lowest validation MAE.
#   - build_model(config) is called with a config whose "head" ends in 7
#     (one score per species) instead of 1 (one length number) — the model
#     architecture code in models.py doesn't need to know or care about this
#     difference; only the final head width changes.
#
# WHAT'S THE SAME (reused directly, not reimplemented):
#   seed_everything, build_optimizer, move_batch — all imported straight from
#   train_baseline.py, because "fix randomness", "build the optimizer", and
#   "move a batch onto the GPU" work identically regardless of the task.
#
# WHERE THIS FITS: configs/cls_*.json -> this file -> runs/cls_*/{best.pt, ...}
#                  -> evaluate_classifier.py -> runs/cls_*/test_metrics.json
# ============================================================================

"""Species classification training: same encoders/pipeline as length regression,
but the head outputs class logits and the loss is cross-entropy.

Reuses build_model (head last value = num_classes) so the encoder-swap comparison
is identical in spirit to the regression study, answering: does the CNN-beats-
foundation-models trend also hold for species identification?
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import CropDataset
from .metrics import classification_metrics
from .models import build_model
from .train_baseline import seed_everything, build_optimizer, move_batch

# Fixed, alphabetically-sorted species -> integer index mapping.
# Example: SPECIES[0] == "cod", so LABEL_MAP == {"cod": 0, "haddock": 1, ...}.
# This exact mapping is reused by evaluate_classifier.py too (imported from
# there), which matters: if training used cod=0 but evaluation assumed
# cod=3, every prediction would silently be scored against the wrong
# species — sorting alphabetically and defining it ONCE in this one place
# guarantees both scripts always agree.
SPECIES = ["cod", "haddock", "hake", "horse_mackerel", "other", "saithe", "whiting"]
LABEL_MAP = {s: i for i, s in enumerate(SPECIES)}


@torch.no_grad()  # scoring only, no training happening in this function
def evaluate_split(model, loader, device, max_batches=None):
    """The classification equivalent of train_baseline.py's `predict()`
    function. Key difference: a classifier outputs 7 raw scores ("logits"),
    one per species, e.g. [2.1, -0.3, 0.8, ...] — `argmax(1)` picks out the
    INDEX of the highest score as the model's actual guess (e.g. index 0 if
    the first number was the biggest, meaning "I think this is a cod").
    """
    model.eval()
    y_true, y_pred = [], []
    for batch_idx, batch in enumerate(loader, start=1):
        if max_batches is not None and batch_idx > max_batches:
            break
        x, target, _ = move_batch(batch, device)
        logits = model(x)                                   # shape [batch_size, 7] — 7 scores per fish
        y_pred.extend(logits.argmax(1).cpu().numpy().tolist())  # pick the species with the highest score
        y_true.extend(target.cpu().numpy().tolist())
    return y_true, y_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--crops-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    seed_everything(config["seed"])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Same dataset settings as train_baseline.py, PLUS `label_map=LABEL_MAP`
    # — this one extra key is what tells CropDataset (see data.py's
    # __getitem__) to hand back a species index instead of a length.
    dk = {
        "image_size": config.get("image_size"),
        "normalize_mean": config.get("normalize_mean"),
        "normalize_std": config.get("normalize_std"),
        "label_map": LABEL_MAP,
    }
    train_ds = CropDataset(args.index, args.crops_dir, split="train", augment=True, **dk)
    val_ds = CropDataset(args.index, args.crops_dir, split="val", augment=False, **dk)
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True,
                              num_workers=config["num_workers"], pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False,
                            num_workers=config["num_workers"], pin_memory=device.type == "cuda")

    # build_model(config) is the EXACT SAME function used for length
    # regression (see models.py) — the only difference is the config file's
    # "head" setting ends in 7 instead of 1, so the same MobileNetV2/DINOv2/
    # etc. classes here produce 7 species-scores instead of 1 length-number.
    model = build_model(config).to(device)

    # CrossEntropyLoss: the standard loss for "pick one out of N categories".
    # Internally it turns the model's 7 raw scores into probabilities that
    # sum to 1 (via softmax) and penalises the model for putting low
    # probability on the correct species — unlike L1Loss (train_baseline.py),
    # which measures a numeric distance, this measures "how confidently and
    # correctly did you classify this?".
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, config)   # identical helper, reused from train_baseline.py

    best_acc = -1.0   # start below any possible real accuracy (accuracy is always >= 0), so
                       # the very first epoch's result always counts as an improvement
    history = []
    max_train_batches = config.get("max_train_batches")
    max_val_batches = config.get("max_val_batches")

    for epoch in range(1, config["epochs"] + 1):
        # ---- one full pass over the training data (identical shape to train_baseline.py) ----
        model.train()
        losses = []
        for batch_idx, batch in enumerate(train_loader, start=1):
            if max_train_batches is not None and batch_idx > max_train_batches:
                break
            x, target, _ = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), target)   # forward pass + loss, in one line here
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # ---- score on validation, then decide whether to save a new best checkpoint ----
        yt, yp = evaluate_split(model, val_loader, device, max_val_batches)
        m = classification_metrics(yt, yp, num_classes=len(SPECIES))   # see metrics.py
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)),
               "val_accuracy": m["accuracy"], "val_macro_f1": m["macro_f1"]}
        history.append(row)
        print(json.dumps(row), flush=True)
        torch.save(model.state_dict(), out_dir / "last.pt")
        # Checkpoint-selection rule for classification: HIGHEST validation
        # accuracy wins (the mirror image of train_baseline.py, where LOWEST
        # validation MAE wins — "higher is better" vs "lower is better").
        if m["accuracy"] > best_acc:
            best_acc = m["accuracy"]
            torch.save(model.state_dict(), out_dir / "best.pt")

    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    # Just like train_baseline.py, the actual reported test-set score is
    # computed separately, once, by evaluate_classifier.py — never here.


if __name__ == "__main__":
    main()
