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
    if config.get("encoder_learning_rate") and hasattr(model, "encoder"):
        encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
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
            param_groups.append(
                {"params": head_params, "lr": config.get("head_learning_rate", config["learning_rate"])}
            )
        return torch.optim.Adam(param_groups)
    return torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=config["learning_rate"],
    )


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch(batch, device):
    (image, bbox), target, meta = batch
    return (image.to(device), bbox.to(device)), target.to(device), meta


@torch.no_grad()
def predict(model, loader, device, max_batches=None):
    model.eval()
    y_true, y_pred, set_names = [], [], []
    for batch_idx, batch in enumerate(loader, start=1):
        if max_batches is not None and batch_idx > max_batches:
            break
        x, target, meta = move_batch(batch, device)
        pred = model(x)
        y_true.extend(target.squeeze(1).cpu().numpy().tolist())
        y_pred.extend(pred.squeeze(1).cpu().numpy().tolist())
        set_names.extend(meta["set_name"])
    return y_true, y_pred, set_names


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
    dataset_kwargs = {
        "image_size": config.get("image_size"),
        "normalize_mean": config.get("normalize_mean"),
        "normalize_std": config.get("normalize_std"),
    }
    train_ds = CropDataset(args.index, args.crops_dir, split="train", augment=True, **dataset_kwargs)
    val_ds = CropDataset(args.index, args.crops_dir, split="val", augment=False, **dataset_kwargs)
    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=device.type == "cuda",
    )

    model = build_model(config).to(device)
    if config.get("resume_checkpoint"):
        model.load_state_dict(torch.load(config["resume_checkpoint"], map_location=device))
    criterion = nn.L1Loss() if config["loss"] == "l1" else nn.SmoothL1Loss()
    optimizer = build_optimizer(model, config)

    best_val = float("inf")
    history = []
    max_train_batches = config.get("max_train_batches")
    max_val_batches = config.get("max_val_batches")
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        train_losses = []
        for batch_idx, batch in enumerate(train_loader, start=1):
            if max_train_batches is not None and batch_idx > max_train_batches:
                break
            x, target, _ = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        y_true, y_pred, _ = predict(model, val_loader, device, max_val_batches)
        val_metrics = regression_metrics(y_true, y_pred)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        print(json.dumps(row), flush=True)

        torch.save(model.state_dict(), out_dir / "last.pt")
        if val_metrics["mae_cm"] < best_val:
            best_val = val_metrics["mae_cm"]
            torch.save(model.state_dict(), out_dir / "best.pt")

    import pandas as pd

    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)


if __name__ == "__main__":
    main()
