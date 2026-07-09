import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data import CropDataset
from .metrics import regression_metrics
from .models import build_model
from .train_baseline import move_batch


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
    ds = CropDataset(
        args.index,
        args.crops_dir,
        split="test",
        augment=False,
        image_size=config.get("image_size"),
        normalize_mean=config.get("normalize_mean"),
        normalize_std=config.get("normalize_std"),
    )
    loader = DataLoader(
        ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
    )

    model = build_model(config).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    rows = []
    for batch in loader:
        x, target, meta = move_batch(batch, device)
        pred = model(x).squeeze(1).cpu().numpy()
        truth = target.squeeze(1).cpu().numpy()
        for i in range(len(pred)):
            rows.append(
                {
                    "annotation_id": int(meta["annotation_id"][i]),
                    "fish_id": int(meta["fish_id"][i]),
                    "group": int(meta["group"][i]),
                    "set_name": meta["set_name"][i],
                    "species": meta["species"][i],
                    "length_cm": float(truth[i]),
                    "pred_cm": float(pred[i]),
                }
            )

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

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))
    df.to_csv(out.with_suffix(".predictions.csv"), index=False)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
