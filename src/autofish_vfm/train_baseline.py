# ============================================================================
# train_baseline.py — the length-REGRESSION training script (run from a terminal).
#
# WHERE THIS FITS IN THE PROJECT:
#   configs/<experiment>.json  (every hyperparameter for one experiment)
#           |
#           v
#   THIS FILE reads that config, builds a model via models.py's build_model(),
#   loads data via data.py's CropDataset, and runs the training loop.
#           |
#           v
#   runs/<experiment>/{config.json, history.csv, last.pt, best.pt}
#           |
#           v
#   evaluate.py loads runs/<experiment>/best.pt and scores it ONCE on the
#   held-out test set, producing runs/<experiment>/test_metrics.json
#
# HOW TO RUN THIS FILE (exactly as the project's queue scripts do):
#   python -m src.autofish_vfm.train_baseline \
#       --config configs/baseline_official.json \
#       --index data/processed/index.csv \
#       --crops-dir data/processed/crops \
#       --out-dir runs/baseline_official
#
# For the CLASSIFICATION equivalent of this file (predict species instead of
# length), see train_classifier.py — same overall shape, different loss and
# a different final head width.
# ============================================================================

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import CropDataset
from .metrics import regression_metrics
from .models import build_model


def build_optimizer(model, config):
    """Creates the Adam optimizer — the algorithm that actually updates the
    model's weights after each batch, based on the gradients computed by
    loss.backward() (see the training loop below).

    Most experiments use ONE learning rate for the whole model. But for
    partial fine-tuning of a foundation model (DINOv2/CLIP with
    trainable_blocks > 0), we want the ALREADY-TRAINED encoder to change only
    a tiny bit (a small "encoder_learning_rate"), while the BRAND NEW head —
    which starts from random weights and has to learn everything from
    scratch — can take much bigger steps (a larger "head_learning_rate").
    That's what the "two parameter groups" branch below sets up.

    Example: config has "encoder_learning_rate": 0.00001 and
    "head_learning_rate": 0.0001 -> the encoder's few trainable params nudge
    10x slower than the head's params, every single optimizer step.
    """
    wd = config.get("weight_decay", 0.0)  # weight decay = a gentle pull-towards-zero on every
    # weight each step, which helps prevent overfitting (0.0 = off, the default for most runs)

    if config.get("encoder_learning_rate") and hasattr(model, "encoder"):
        # ---- Two-speed training: encoder learns slowly, head learns faster ----
        # Only collect the encoder params that are ACTUALLY trainable
        # (frozen ones were already set to requires_grad=False in models.py).
        encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
        # Everything else in the model (the head, i.e. self.classifier) that
        # is trainable — found by name so it works regardless of which
        # encoder class we're using.
        head_params = [
            p
            for name, p in model.named_parameters()
            if p.requires_grad and not name.startswith("encoder.")
        ]
        param_groups = []
        if encoder_params:
            param_groups.append(
                {"params": encoder_params, "lr": config["encoder_learning_rate"]}
            )
        if head_params:
            # If no separate head_learning_rate is given, fall back to the
            # ordinary "learning_rate" setting for the head.
            param_groups.append(
                {"params": head_params, "lr": config.get("head_learning_rate", config["learning_rate"])}
            )
        return torch.optim.Adam(param_groups, weight_decay=wd)

    # ---- The normal case: everything trainable shares ONE learning rate ----
    # This is what the MobileNetV2/EfficientNet/ConvNeXt baseline experiments
    # use (their configs only ever set "learning_rate", never
    # "encoder_learning_rate").
    return torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=config["learning_rate"],
        weight_decay=wd,
    )


def build_scheduler(optimizer, config):
    """Optional learning-rate SCHEDULE: instead of using the same learning
    rate for every single epoch, gradually shrink it over the course of
    training. "cosine" annealing follows a smooth curve from the starting
    learning rate down to (near) zero by the final epoch — the idea is to
    take big confident steps early on and small careful steps near the end,
    to settle into a better final answer. Only used by the "stronger recipe"
    experiments (see configs/*_strong_*.json); most runs set no schedule at
    all, in which case this returns None and the learning rate stays constant.
    """
    if config.get("lr_schedule") == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    return None


def seed_everything(seed):
    """Fixes every source of randomness (Python's random module, NumPy, and
    PyTorch's CPU + GPU random number generators) to the SAME starting point,
    given the same seed number. Without this, two runs of the "same"
    experiment would shuffle the data differently, initialise weights
    differently, etc., making results impossible to reproduce exactly.
    Every config.json in this project sets "seed": 42 (or 1 / 2 for the
    multi-seed reliability experiments) — this function is what actually
    applies that number.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch(batch, device):
    """A DataLoader batch arrives as CPU tensors (that's where data.py's
    CropDataset puts them). Before the model can use them, image/bbox/target
    tensors need to be copied onto the GPU (device="cuda") if one is
    available — everything else (the `meta` dict) is left as plain Python
    values since the model never touches it.

    `batch` here is EXACTLY the ((image, bbox), target, meta) tuple that
    CropDataset.__getitem__ returns in data.py — DataLoader has just stacked
    many of those individual fish into one batched tensor per field.
    """
    (image, bbox), target, meta = batch
    return (image.to(device), bbox.to(device)), target.to(device), meta


@torch.no_grad()  # disables gradient tracking for everything in this function — we are only
# reading predictions out of the model here, never training it, so this saves memory/compute.
def predict(model, loader, device, max_batches=None):
    """Runs the model over an entire DataLoader (e.g. the validation set) and
    collects every prediction. Used both (a) after every training epoch, to
    check validation MAE and decide whether to save a new "best" checkpoint,
    and (b) by evaluate.py for the final one-time test-set scoring.

    max_batches is only used by "smoke test" configs (tiny configs that
    process just a couple of batches, to sanity-check that an experiment's
    code runs correctly before committing GPU-hours to the full run).
    """
    model.eval()  # switches the model to "evaluation mode" (turns off dropout, freezes
    # BatchNorm's running statistics, etc.) — the counterpart to model.train() below.
    y_true, y_pred, set_names = [], [], []
    for batch_idx, batch in enumerate(loader, start=1):
        if max_batches is not None and batch_idx > max_batches:
            break
        x, target, meta = move_batch(batch, device)
        pred = model(x)
        # .squeeze(1) removes the "1" dimension from a [batch_size, 1] tensor,
        # turning it into a flat [batch_size] list of numbers, which is what
        # regression_metrics() (in metrics.py) expects.
        y_true.extend(target.squeeze(1).cpu().numpy().tolist())
        y_pred.extend(pred.squeeze(1).cpu().numpy().tolist())
        set_names.extend(meta["set_name"])  # "Set1"/"Set2"/"All" — kept for later per-subset analysis
    return y_true, y_pred, set_names


def main():
    # ---- Step 1: read the command-line arguments ----
    # Example invocation:
    #   python -m src.autofish_vfm.train_baseline --config configs/baseline_official.json
    #          --index data/processed/index.csv --crops-dir data/processed/crops
    #          --out-dir runs/baseline_official
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)      # which experiment's JSON settings to use
    parser.add_argument("--index", required=True)        # data/processed/index.csv
    parser.add_argument("--crops-dir", required=True)    # data/processed/crops/
    parser.add_argument("--out-dir", required=True)      # where to write runs/<experiment>/
    args = parser.parse_args()

    # ---- Step 2: load the experiment's settings and lock in randomness ----
    config = json.loads(Path(args.config).read_text())
    seed_everything(config["seed"])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Copy the exact config used into the output folder too, so that months
    # later anyone (including us) can see precisely which settings produced
    # this run's results, without having to guess or cross-reference configs/.
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    # Use the GPU if one is available (it will be, on the training server);
    # fall back to CPU automatically otherwise (e.g. for quick local smoke tests).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Step 3: build the training and validation datasets/loaders ----
    # These three settings are shared by both splits, so we build the dict
    # once and reuse it (**dataset_kwargs unpacks it as keyword arguments).
    dataset_kwargs = {
        "image_size": config.get("image_size"),
        "normalize_mean": config.get("normalize_mean"),
        "normalize_std": config.get("normalize_std"),
    }
    # augment=True ONLY for training (random colour jitter, see data.py) —
    # validation must always be evaluated on the same, unperturbed images
    # every single time, so its score is directly comparable epoch to epoch.
    train_ds = CropDataset(args.index, args.crops_dir, split="train", augment=True, **dataset_kwargs)
    val_ds = CropDataset(args.index, args.crops_dir, split="val", augment=False, **dataset_kwargs)

    # DataLoader is PyTorch's batching machinery: it calls CropDataset.__getitem__
    # many times, stacks the results into batches of size config["batch_size"],
    # and (with num_workers>0) does this loading in background processes so the
    # GPU is never left waiting for the next batch to be ready.
    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,   # scramble the fish order every epoch, so the model doesn't
                         # learn any accidental pattern from the order they're stored in
        num_workers=config["num_workers"],
        pin_memory=device.type == "cuda",  # speeds up the CPU->GPU data transfer
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["batch_size"],
        shuffle=False,  # order doesn't matter for evaluation, and keeping it fixed
                         # makes debugging/inspecting predictions easier
        num_workers=config["num_workers"],
        pin_memory=device.type == "cuda",
    )

    # ---- Step 4: build the model, optionally warm-start it, pick the loss ----
    model = build_model(config).to(device)  # see models.py — this reads config["model"]
                                              # and constructs the right encoder+head
    if config.get("resume_checkpoint"):
        # Used by "last-block fine-tuning" experiments: start from an already
        # -trained frozen model's weights (e.g. runs/dinov2_vits14_frozen/best.pt)
        # instead of from the encoder's original pretrained-only weights.
        model.load_state_dict(torch.load(config["resume_checkpoint"], map_location=device))

    # L1 loss = mean absolute error between prediction and truth (the loss
    # used by every length-regression experiment in this project). SmoothL1
    # (Huber loss) is supported as an alternative but not used by any of our
    # final configs.
    criterion = nn.L1Loss() if config["loss"] == "l1" else nn.SmoothL1Loss()
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    # ---- Step 5: the training loop itself ----
    best_val = float("inf")   # start "best" at infinitely bad, so ANY real result improves on it
    history = []               # one row per epoch, saved to history.csv at the end
    max_train_batches = config.get("max_train_batches")  # only set for tiny smoke-test configs
    max_val_batches = config.get("max_val_batches")

    for epoch in range(1, config["epochs"] + 1):
        # ---- (a) one full pass over the TRAINING data ----
        model.train()  # switch back to "training mode" (re-enables dropout/BatchNorm updates)
        train_losses = []
        for batch_idx, batch in enumerate(train_loader, start=1):
            if max_train_batches is not None and batch_idx > max_train_batches:
                break
            x, target, _ = move_batch(batch, device)   # x = (image, bbox); we don't need meta here

            optimizer.zero_grad(set_to_none=True)  # clear out gradients left over from the
                                                     # previous batch (PyTorch accumulates them
                                                     # by default, so this reset is required)
            pred = model(x)                         # FORWARD PASS: crop+bbox -> predicted length
            loss = criterion(pred, target)          # how wrong was that guess? (mean |pred-truth|)
            loss.backward()                         # BACKPROPAGATION: compute how much each
                                                       # weight in the whole network contributed
                                                       # to that error (the gradient)
            optimizer.step()                        # actually nudge every trainable weight a
                                                       # little bit in the direction that reduces
                                                       # the loss (this is where "learning" happens)
            train_losses.append(loss.item())         # .item() pulls the number out of the tensor

        # ---- (b) score the model on VALIDATION data (no training happens here) ----
        y_true, y_pred, _ = predict(model, val_loader, device, max_val_batches)
        val_metrics = regression_metrics(y_true, y_pred)  # MAE, RMSE, MAPE, bias, R² (see metrics.py)

        # Record this epoch's numbers for the history.csv log.
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            **{f"val_{k}": v for k, v in val_metrics.items()},  # e.g. val_mae_cm, val_rmse_cm, ...
        }
        history.append(row)
        print(json.dumps(row), flush=True)  # printed live so a human watching the training
                                              # log (or a monitoring script) can see progress

        # ---- (c) checkpoint saving: this is how we avoid "test-tuning" ----
        # ALWAYS overwrite last.pt (handy if a run crashes and needs resuming).
        torch.save(model.state_dict(), out_dir / "last.pt")
        # ONLY overwrite best.pt when validation MAE improved. This is the
        # single most important line for scientific honesty in this project:
        # the checkpoint we ultimately evaluate on the TEST set (in
        # evaluate.py) is chosen purely by looking at VALIDATION performance,
        # never by peeking at the test set during training.
        if val_metrics["mae_cm"] < best_val:
            best_val = val_metrics["mae_cm"]
            torch.save(model.state_dict(), out_dir / "best.pt")

        if scheduler is not None:
            scheduler.step()  # advance the cosine learning-rate schedule by one epoch, if enabled

    # ---- Step 6: save the full per-epoch training history to disk ----
    import pandas as pd  # imported here rather than at the top since it's only needed for this
                          # one line, keeping the "core" imports at the top minimal

    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    # After this function returns, a completely separate script (evaluate.py)
    # is responsible for loading runs/<experiment>/best.pt and scoring it
    # once on the TEST set — that final, most important number is
    # deliberately NOT computed anywhere in this file.


if __name__ == "__main__":
    # This guard means "only run main() if this file was executed directly
    # (e.g. `python -m src.autofish_vfm.train_baseline ...`), not if some
    # other file merely imports functions from this one." evaluate.py, for
    # example, imports `move_batch` from this file without triggering a
    # second training run.
    main()
