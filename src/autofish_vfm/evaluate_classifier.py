# ============================================================================
# evaluate_classifier.py — scores a trained species classifier ONCE on test data.
#
# This is the classification twin of evaluate.py: same overall shape (load a
# frozen checkpoint, run it over a split, save metrics + a per-fish CSV), but
# working with species PREDICTIONS (an index 0-6) instead of length
# predictions (a number in cm). Compare this file to evaluate.py side by side
# to see exactly what a classification evaluation script needs that a
# regression one doesn't (turning logits into a predicted class via argmax,
# and converting indices back to human-readable species names for the CSV).
#
# WHERE THIS FITS: runs/cls_<encoder>/best.pt (from train_classifier.py)
#                  -> THIS FILE -> runs/cls_<encoder>/test_metrics.json
#                                  runs/cls_<encoder>/test_metrics.predictions.csv
# ============================================================================

"""Evaluate a trained species classifier on the test set.
Writes accuracy + macro-F1 (overall and by occlusion) and a per-fish predictions CSV.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data import CropDataset
from .metrics import classification_metrics
from .models import build_model
from .train_baseline import move_batch
from .train_classifier import LABEL_MAP, SPECIES  # reuse the SAME species<->index mapping
                                                     # that was used during training — critical,
                                                     # otherwise index 0 might mean "cod" during
                                                     # training but get mislabeled here.


@torch.no_grad()  # evaluation only — no weights change in this file
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)  # e.g. runs/cls_mobilenet_v2/best.pt
    parser.add_argument("--config", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--crops-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # split="test" is hard-coded here (unlike evaluate.py, which accepts
    # --split) because every species-classification experiment in this
    # project only ever needed a final test-set number, never a validation
    # re-scoring pass.
    ds = CropDataset(args.index, args.crops_dir, split="test", augment=False,
                     image_size=config.get("image_size"),
                     normalize_mean=config.get("normalize_mean"),
                     normalize_std=config.get("normalize_std"),
                     label_map=LABEL_MAP)   # <- this is what makes CropDataset hand back
                                             # species indices instead of lengths (see data.py)
    loader = DataLoader(ds, batch_size=config["batch_size"], shuffle=False,
                        num_workers=config["num_workers"])

    # Rebuild the exact architecture used in training, then load its learned weights.
    model = build_model(config).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    rows = []
    for batch in loader:
        x, target, meta = move_batch(batch, device)
        # model(x) gives 7 raw scores per fish; .argmax(1) picks the index of
        # the highest one — "which species did the model bet on?"
        pred = model(x).argmax(1).cpu().numpy()
        truth = target.cpu().numpy()
        for i in range(len(pred)):
            rows.append({
                "annotation_id": int(meta["annotation_id"][i]),
                "group": int(meta["group"][i]),
                "set_name": meta["set_name"][i],
                # Store BOTH the human-readable species name (SPECIES[idx])
                # AND the raw numeric index — the name is what a person
                # reading the CSV wants to see; the index is what
                # classification_metrics() below needs to compute scores.
                "true_species": SPECIES[int(truth[i])],
                "pred_species": SPECIES[int(pred[i])],
                "true_idx": int(truth[i]),
                "pred_idx": int(pred[i]),
            })
    df = pd.DataFrame(rows)

    # Same three-way split as the regression evaluate.py: overall, then
    # non-occluded fish only, then occluded fish only — so we can see whether
    # species classification is ALSO harder on overlapping fish (spoiler: yes,
    # but far less dramatically than length regression is).
    non_occ = df[df.set_name.isin(["Set1", "Set2"])]
    occ = df[df.set_name == "All"]
    metrics = {
        "test_all": classification_metrics(df.true_idx, df.pred_idx, len(SPECIES)),
        "non_occluded_set1_set2": classification_metrics(non_occ.true_idx, non_occ.pred_idx, len(SPECIES)),
        "occluded_all": classification_metrics(occ.true_idx, occ.pred_idx, len(SPECIES)),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))
    # Per-fish predictions CSV, exactly like evaluate.py's — this is what
    # later analysis scripts read to build confusion matrices / per-species
    # breakdowns, without ever touching the GPU again.
    df.to_csv(out.with_suffix(".predictions.csv"), index=False)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
