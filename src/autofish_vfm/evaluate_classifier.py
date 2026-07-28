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
from .train_classifier import LABEL_MAP, SPECIES


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--crops-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = CropDataset(args.index, args.crops_dir, split="test", augment=False,
                     image_size=config.get("image_size"),
                     normalize_mean=config.get("normalize_mean"),
                     normalize_std=config.get("normalize_std"),
                     label_map=LABEL_MAP)
    loader = DataLoader(ds, batch_size=config["batch_size"], shuffle=False,
                        num_workers=config["num_workers"])

    model = build_model(config).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    rows = []
    for batch in loader:
        x, target, meta = move_batch(batch, device)
        pred = model(x).argmax(1).cpu().numpy()
        truth = target.cpu().numpy()
        for i in range(len(pred)):
            rows.append({
                "annotation_id": int(meta["annotation_id"][i]),
                "group": int(meta["group"][i]),
                "set_name": meta["set_name"][i],
                "true_species": SPECIES[int(truth[i])],
                "pred_species": SPECIES[int(pred[i])],
                "true_idx": int(truth[i]),
                "pred_idx": int(pred[i]),
            })
    df = pd.DataFrame(rows)
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
    df.to_csv(out.with_suffix(".predictions.csv"), index=False)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
