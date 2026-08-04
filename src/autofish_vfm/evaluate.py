# ============================================================================
# evaluate.py — scores ONE already-trained checkpoint, ONCE, on a chosen split.
#
# WHERE THIS FITS IN THE PROJECT:
#   runs/<experiment>/best.pt   (produced by train_baseline.py)
#           |
#           v
#   THIS FILE loads that checkpoint (frozen — no more training happens here)
#   and runs it forward over every fish in the requested split, comparing
#   each prediction to the true length.
#           |
#           v
#   runs/<experiment>/test_metrics.json               (headline MAE/RMSE/etc numbers)
#   runs/<experiment>/test_metrics.predictions.csv     (the per-fish predictions — this
#                                                        is the file error_analysis.py and
#                                                        make_qualitative_figures.py read
#                                                        later, with no need to retrain)
#
# WHY THIS IS A SEPARATE FILE FROM train_baseline.py:
#   Keeping "training" and "final scoring" in two different scripts makes it
#   structurally impossible to accidentally let the test set influence which
#   checkpoint gets kept (see train_baseline.py's checkpoint-saving logic,
#   which only ever looks at VALIDATION performance). This file is only ever
#   run AFTER training has completely finished, and only ever on a single,
#   already-decided checkpoint (best.pt).
#
# HOW TO RUN THIS FILE:
#   python -m src.autofish_vfm.evaluate \
#       --checkpoint runs/baseline_official/best.pt \
#       --config configs/baseline_official.json \
#       --index data/processed/index.csv --crops-dir data/processed/crops \
#       --out runs/baseline_official/test_metrics.json
#   (add --split val instead of the default "test" to score on validation
#   data instead — used e.g. when picking recipe/ensemble settings without
#   touching the test set at all)
# ============================================================================

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data import CropDataset
from .metrics import regression_metrics
from .models import build_model
from .train_baseline import move_batch  # reuse the exact same CPU->GPU batch-moving helper


@torch.no_grad()  # we are only ever reading predictions out of a frozen model here —
# never training — so gradient tracking would just waste memory and time.
def main():
    # ---- Step 1: read command-line arguments ----
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)  # e.g. runs/baseline_official/best.pt
    parser.add_argument("--config", required=True)      # the SAME config.json used to train it
                                                           # (needed so we build the identical
                                                           # architecture before loading weights into it)
    parser.add_argument("--index", required=True)
    parser.add_argument("--crops-dir", required=True)
    parser.add_argument("--out", required=True)          # where to write test_metrics.json
    parser.add_argument("--split", default="test")        # which split to score: "test" (the
                                                           # default, used for the headline numbers)
                                                           # or "val" (used only for recipe/ensemble
                                                           # selection, never for a reported result)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build the dataset for whichever split was requested. augment=False
    # always — evaluation must see the real, unperturbed images (ColorJitter
    # augmentation is a training-only trick, see data.py).
    ds = CropDataset(
        args.index,
        args.crops_dir,
        split=args.split,
        augment=False,
        image_size=config.get("image_size"),
        normalize_mean=config.get("normalize_mean"),
        normalize_std=config.get("normalize_std"),
    )
    loader = DataLoader(
        ds,
        batch_size=config["batch_size"],
        shuffle=False,  # no need to shuffle for evaluation, and keeping order fixed makes
                         # it easier to line predictions back up with specific fish afterwards
        num_workers=config["num_workers"],
    )

    # ---- Step 2: rebuild the exact model architecture, then load the trained weights into it ----
    # build_model(config) constructs a FRESH model with random/pretrained
    # starting weights (same as at the start of training) — load_state_dict
    # then overwrites every weight with the ones actually learned during training.
    model = build_model(config).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()  # turn off dropout / freeze BatchNorm stats, same as during validation in training

    # ---- Step 3: run every fish in this split through the model once ----
    rows = []
    for batch in loader:
        x, target, meta = move_batch(batch, device)
        pred = model(x).squeeze(1).cpu().numpy()     # model's guesses, back to plain numpy numbers
        truth = target.squeeze(1).cpu().numpy()       # the real, true lengths for comparison
        # Build one dictionary PER FISH (not per batch) so the final CSV has
        # exactly one row per fish, with everything needed for later analysis:
        # which fish it was, which occlusion set, which species, and both the
        # true and predicted length.
        for i in range(len(pred)):
            rows.append(
                {
                    "annotation_id": int(meta["annotation_id"][i]),
                    "fish_id": int(meta["fish_id"][i]),
                    "group": int(meta["group"][i]),
                    "set_name": meta["set_name"][i],   # "Set1"/"Set2" (non-occluded) or "All" (occluded)
                    "species": meta["species"][i],
                    "length_cm": float(truth[i]),
                    "pred_cm": float(pred[i]),
                }
            )

    # ---- Step 4: compute the headline metrics, split three ways ----
    # This is exactly why every result table in this project reports THREE
    # numbers per model: the overall test score, plus a breakdown by how hard
    # the fish were to see (occlusion). regression_metrics() (metrics.py)
    # computes MAE/RMSE/MAPE/bias/R² for whichever rows are passed to it.
    df = pd.DataFrame(rows)
    metrics = {
        "test_all": regression_metrics(df["length_cm"], df["pred_cm"]),
        "non_occluded_set1_set2": regression_metrics(
            df[df["set_name"].isin(["Set1", "Set2"])]["length_cm"],
            df[df["set_name"].isin(["Set1", "Set2"])]["pred_cm"],
        ),
        "occluded_all": regression_metrics(
            df[df["set_name"] == "All"]["length_cm"],
            df[df["set_name"] == "All"]["pred_cm"],
        ),
    }

    # ---- Step 5: save both the summary numbers AND the raw per-fish predictions ----
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))                       # e.g. test_metrics.json
    df.to_csv(out.with_suffix(".predictions.csv"), index=False)         # e.g. test_metrics.predictions.csv
    # ^ This predictions CSV is the "raw material" that later, completely
    # separate scripts (scripts/error_analysis.py, scripts/make_qualitative_figures.py,
    # scripts/make_result_charts.py) read to build every table/figure/chart in
    # the project — WITHOUT ever needing to reload the model or rerun anything
    # on the GPU. This is the mechanism that made the project's "no-repeat"
    # rule possible: expensive GPU work happens exactly once per experiment,
    # here, and everything downstream is cheap CSV/plotting work.
    print(json.dumps(metrics, indent=2))  # also print to the terminal/log for a quick human check


if __name__ == "__main__":
    main()
