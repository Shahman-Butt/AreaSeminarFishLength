import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from tqdm import tqdm


def mask_from_polygons(segmentation: str, size: tuple[int, int]) -> Image.Image:
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    for polygon in json.loads(segmentation):
        points = list(zip(polygon[0::2], polygon[1::2]))
        draw.polygon(points, fill=255)
    return mask


def square_bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    side = max(x1 - x0 + 1, y1 - y0 + 1)
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    x0 = int(round(cx - side / 2.0))
    y0 = int(round(cy - side / 2.0))
    return x0, y0, side, side


def crop_with_padding(img: Image.Image, bbox: tuple[int, int, int, int]) -> Image.Image:
    x, y, w, h = bbox
    canvas = Image.new("RGB", (w, h), (0, 0, 0))
    src = (max(0, x), max(0, y), min(img.width, x + w), min(img.height, y + h))
    dst = (src[0] - x, src[1] - y)
    canvas.paste(img.crop(src), dst)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw/autofish")
    parser.add_argument("--index", default="data/processed/index.csv")
    parser.add_argument("--out-dir", default="data/processed/crops")
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.index)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = list(df.groupby("image_path", sort=False))
    pbar = tqdm(total=len(df), desc="crops")
    for image_path, image_rows in groups:
        pending = [
            row
            for _, row in image_rows.iterrows()
            if args.overwrite or not (out_dir / f"{int(row.annotation_id):06d}.png").exists()
        ]
        if not pending:
            pbar.update(len(image_rows))
            continue

        img = Image.open(image_path).convert("RGB")
        for row in pending:
            out_path = out_dir / f"{int(row.annotation_id):06d}.png"
            mask = mask_from_polygons(row.segmentation, img.size)

            masked = Image.new("RGB", img.size, (0, 0, 0))
            masked.paste(img, mask=mask)

            bbox = square_bbox_from_mask(np.array(mask))
            crop = crop_with_padding(masked, bbox)
            crop = crop.resize((args.size, args.size), Image.Resampling.BILINEAR)
            crop.save(out_path)
        pbar.update(len(image_rows))
    pbar.close()

    print(f"wrote crops to {out_dir}")


if __name__ == "__main__":
    main()
