# ============================================================================
# make_crops.py — the masking + cropping pipeline: turns full table photos
# into one clean, masked, square, 224x224 PNG per fish.
#
# WHERE THIS FITS IN THE PROJECT:
#   data/processed/index.csv        (produced by build_autofish_index.py —
#                                     has one row per fish, with its polygon
#                                     outline and bounding box)
#                    |
#                    v
#   THIS FILE reads every row and produces:
#                    |
#                    v
#   data/processed/crops/000001.png, 000002.png, ...   (one PNG per fish,
#                                                          named by annotation_id)
#                    |
#                    v
#   src/autofish_vfm/data.py's CropDataset loads these PNGs directly —
#   it never looks at the original full photos or does any masking itself.
#
# WHY THIS STEP EXISTS AS ITS OWN SCRIPT (run once, ahead of time):
#   Masking + cropping every fish out of its original photo is somewhat
#   expensive (image processing per fish). Doing it ONCE here and caching the
#   result as PNG files means every training run — and there were many, for
#   every encoder and every hyperparameter recipe — just opens a small
#   pre-made PNG instead of redoing this work from scratch every single time.
#
# HOW TO RUN THIS FILE:
#   python scripts/make_crops.py --raw-dir data/raw/autofish
#                                 --index data/processed/index.csv
#                                 --out-dir data/processed/crops
# ============================================================================

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from tqdm import tqdm  # a progress bar — purely cosmetic, so a human watching a
                        # terminal can see how far through the 18,157 fish we are


def mask_from_polygons(segmentation: str, size: tuple[int, int]) -> Image.Image:
    """STEP 1 of the masking pipeline: turn a fish's stored outline into an
    actual black-and-white stencil image.

    `segmentation` is a JSON string (from index.csv) describing one or more
    polygons — each polygon is a flat list [x1, y1, x2, y2, x3, y3, ...] of
    the (x, y) points that trace the fish's outline in the ORIGINAL photo.

    WORKED EXAMPLE: a tiny triangular "fish" outline might be stored as
    [10, 10, 50, 10, 30, 40] -> after zip(polygon[0::2], polygon[1::2]) that
    becomes points [(10,10), (50,10), (30,40)] -> draw.polygon(...) fills in
    the triangle between those three points with white (255), leaving
    everything else in the image black (0).

    Returns a single-channel ("L" = greyscale/luminance) image the same size
    as the original photo: 0 = "not the fish", 255 = "this pixel is the fish".
    """
    mask = Image.new("L", size, 0)   # start with an all-black canvas the size of the original photo
    draw = ImageDraw.Draw(mask)
    for polygon in json.loads(segmentation):   # usually one polygon, but a fish's outline
                                                 # can sometimes be stored as several pieces
        # polygon[0::2] = every x-coordinate (indices 0, 2, 4, ...)
        # polygon[1::2] = every y-coordinate (indices 1, 3, 5, ...)
        # zip(...) pairs them back up into (x, y) points.
        points = list(zip(polygon[0::2], polygon[1::2]))
        draw.polygon(points, fill=255)   # paint this shape white = "this is fish"
    return mask


def square_bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    """STEP 2: find a SQUARE window, centred on the fish, tight enough to
    contain the whole mask.

    WHY SQUARE MATTERS: every crop eventually gets resized to a fixed
    224x224 square (see main() below). If we cropped a non-square
    rectangle and then force-resized it to a square, the fish's proportions
    would get STRETCHED — a fish that was really long and thin might end up
    looking short and fat, actively corrupting the length signal we are
    trying to measure. Using a square from the very start avoids this
    entirely, no matter the resize.

    WORKED EXAMPLE: suppose the white mask pixels span x from 100 to 180
    (width 81) and y from 50 to 90 (height 41). The longer side is 81, so we
    use side=81 for BOTH width and height, centred on the fish's midpoint —
    giving a square window a little "wider" than the fish actually needed
    vertically, rather than a tight-but-non-square rectangle.
    """
    ys, xs = np.where(mask > 0)             # coordinates of every "this is fish" pixel
    x0, x1 = int(xs.min()), int(xs.max())    # leftmost / rightmost fish pixel
    y0, y1 = int(ys.min()), int(ys.max())    # topmost / bottommost fish pixel
    side = max(x1 - x0 + 1, y1 - y0 + 1)      # the SQUARE side length = the longer of width/height
    cx = (x0 + x1) / 2.0                      # the fish's horizontal centre
    cy = (y0 + y1) / 2.0                      # the fish's vertical centre
    # Re-centre the square window on that midpoint.
    x0 = int(round(cx - side / 2.0))
    y0 = int(round(cy - side / 2.0))
    return x0, y0, side, side


def crop_with_padding(img: Image.Image, bbox: tuple[int, int, int, int]) -> Image.Image:
    """STEP 3: actually cut out that square window from the (already masked)
    photo — safely, even if the square runs off the edge of the original photo.

    Example: a fish near the very edge of the table might need a square
    window that would technically extend a few pixels past the photo's
    border. Instead of crashing or wrapping around, we paste only the part
    that actually overlaps the real photo onto a black canvas — so the
    missing sliver just stays black, exactly like the rest of the masked
    background.
    """
    x, y, w, h = bbox
    canvas = Image.new("RGB", (w, h), (0, 0, 0))   # blank black canvas of the target crop size
    # Clip the requested window to the photo's real boundaries...
    src = (max(0, x), max(0, y), min(img.width, x + w), min(img.height, y + h))
    # ...and work out where that clipped region should land on our canvas
    # (it won't be at (0,0) if the window ran off the LEFT/TOP edge).
    dst = (src[0] - x, src[1] - y)
    canvas.paste(img.crop(src), dst)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw/autofish")
    parser.add_argument("--index", default="data/processed/index.csv")
    parser.add_argument("--out-dir", default="data/processed/crops")
    parser.add_argument("--size", type=int, default=224)   # final crop size — 224x224 matches what
                                                              # every encoder in models.py expects
    parser.add_argument("--overwrite", action="store_true")  # by default, skip fish whose crop
    # PNG already exists — makes it cheap to re-run this script after adding
    # a few new annotations, without redoing all 18,157 fish from scratch
    args = parser.parse_args()

    df = pd.read_csv(args.index)   # the master spreadsheet from build_autofish_index.py
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Group all fish annotations by which PHOTO they came from. Many fish
    # share the same photo (a table full of fish = one photo, many
    # annotations), so this lets us open each original photo file only ONCE
    # and cut out every fish in it, instead of re-opening the same photo file
    # over and over for each fish — much faster.
    groups = list(df.groupby("image_path", sort=False))
    pbar = tqdm(total=len(df), desc="crops")
    for image_path, image_rows in groups:
        # Skip any fish whose crop PNG is already sitting on disk (unless
        # --overwrite was passed) — lets this script be safely re-run/resumed.
        pending = [
            row
            for _, row in image_rows.iterrows()
            if args.overwrite or not (out_dir / f"{int(row.annotation_id):06d}.png").exists()
        ]
        if not pending:
            pbar.update(len(image_rows))
            continue

        img = Image.open(image_path).convert("RGB")   # open the ONE original photo,
                                                         # shared by every fish in `pending`
        for row in pending:
            out_path = out_dir / f"{int(row.annotation_id):06d}.png"   # e.g. "data/processed/crops/006721.png"

            # ---- The full masking pipeline for ONE fish, in order ----
            # 1) build the black/white stencil from this fish's stored outline
            mask = mask_from_polygons(row.segmentation, img.size)

            # 2) blacken out everything that ISN'T this fish. Image.paste's
            #    `mask=` argument means: copy pixels from `img` wherever the
            #    mask is white (255); leave the black canvas showing through
            #    wherever the mask is black (0). This is the step that makes
            #    overlapping/occluding fish disappear from the final crop —
            #    only THIS fish's own polygon is white.
            masked = Image.new("RGB", img.size, (0, 0, 0))
            masked.paste(img, mask=mask)

            # 3) find a tight SQUARE window around the fish, and cut it out
            bbox = square_bbox_from_mask(np.array(mask))
            crop = crop_with_padding(masked, bbox)

            # 4) resize to the network's expected fixed input size (224x224),
            #    using bilinear interpolation (a smooth, standard resize —
            #    good quality without being unnecessarily slow)
            crop = crop.resize((args.size, args.size), Image.Resampling.BILINEAR)
            crop.save(out_path)   # write the final PNG — this is the exact file that
                                    # data.py's CropDataset will load, unmodified, for every
                                    # future training run using this fish
        pbar.update(len(image_rows))
    pbar.close()

    print(f"wrote crops to {out_dir}")


if __name__ == "__main__":
    main()
