from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CropDataset(Dataset):
    def __init__(
        self,
        index_csv,
        crops_dir,
        split,
        groups=None,
        augment=False,
        image_size=None,
        normalize_mean=None,
        normalize_std=None,
    ):
        self.df = pd.read_csv(index_csv)
        if split != "all":
            self.df = self.df[self.df["split"] == split]
        if groups:
            self.df = self.df[self.df["group"].isin(groups)]
        self.df = self.df.reset_index(drop=True)
        self.crops_dir = Path(crops_dir)
        t = []
        if image_size:
            t.append(transforms.Resize((image_size, image_size)))
        t.append(transforms.ToTensor())
        if augment:
            t.append(
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.5,
                    saturation=0.4,
                    hue=0.3,
                )
            )
        if normalize_mean is not None and normalize_std is not None:
            t.append(transforms.Normalize(mean=normalize_mean, std=normalize_std))
        self.transform = transforms.Compose(t)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(self.crops_dir / f"{int(row.annotation_id):06d}.png").convert("RGB")
        image = self.transform(image)
        bbox = torch.tensor(
            [
                row.bbox_x / row.width,
                row.bbox_y / row.height,
                row.bbox_w / row.width,
                row.bbox_h / row.height,
            ],
            dtype=torch.float32,
        )
        target = torch.tensor([row.length_cm], dtype=torch.float32)
        meta = {
            "annotation_id": int(row.annotation_id),
            "fish_id": int(row.fish_id),
            "group": int(row.group),
            "set_name": row.set_name,
            "species": row.species,
            "length_cm": float(row.length_cm),
        }
        return (image, bbox), target, meta
