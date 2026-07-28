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

# Deterministic species -> index map (sorted for reproducibility).
SPECIES = ["cod", "haddock", "hake", "horse_mackerel", "other", "saithe", "whiting"]
LABEL_MAP = {s: i for i, s in enumerate(SPECIES)}


@torch.no_grad()
def evaluate_split(model, loader, device, max_batches=None):
    model.eval()
    y_true, y_pred = [], []
    for batch_idx, batch in enumerate(loader, start=1):
        if max_batches is not None and batch_idx > max_batches:
            break
        x, target, _ = move_batch(batch, device)
        logits = model(x)
        y_pred.extend(logits.argmax(1).cpu().numpy().tolist())
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

    model = build_model(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, config)

    best_acc = -1.0
    history = []
    max_train_batches = config.get("max_train_batches")
    max_val_batches = config.get("max_val_batches")
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        losses = []
        for batch_idx, batch in enumerate(train_loader, start=1):
            if max_train_batches is not None and batch_idx > max_train_batches:
                break
            x, target, _ = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        yt, yp = evaluate_split(model, val_loader, device, max_val_batches)
        m = classification_metrics(yt, yp, num_classes=len(SPECIES))
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)),
               "val_accuracy": m["accuracy"], "val_macro_f1": m["macro_f1"]}
        history.append(row)
        print(json.dumps(row), flush=True)
        torch.save(model.state_dict(), out_dir / "last.pt")
        if m["accuracy"] > best_acc:
            best_acc = m["accuracy"]
            torch.save(model.state_dict(), out_dir / "best.pt")

    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)


if __name__ == "__main__":
    main()
