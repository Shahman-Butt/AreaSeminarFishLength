import argparse
import json
from pathlib import Path

import pandas as pd


OFFICIAL_SPLIT = {
    "train": [2, 3, 4, 5, 7, 8, 9, 12, 13, 15, 16, 18, 19, 23, 24],
    "val": [1, 6, 11, 17, 25],
    "test": [10, 14, 20, 21, 22],
}


def set_name_from_file(file_name: str) -> str:
    image_no = int(Path(file_name).stem)
    if 1 <= image_no <= 20:
        return "Set1"
    if 21 <= image_no <= 40:
        return "Set2"
    if 41 <= image_no <= 60:
        return "All"
    raise ValueError(f"Unexpected image number in {file_name}")


def split_from_group(group: int) -> str:
    for split, groups in OFFICIAL_SPLIT.items():
        if group in groups:
            return split
    raise ValueError(f"Group {group} is not in the official split")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw/autofish")
    parser.add_argument("--out", default="data/processed/index.csv")
    parser.add_argument("--splits-out", default="data/processed/splits.json")
    parser.add_argument("--exclusions-out", default="data/processed/exclusions.json")
    parser.add_argument("--keep-cross-split-duplicates", action="store_true")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    ann_path = raw_dir / "annotations.json"
    data = json.loads(ann_path.read_text())

    images = {img["id"]: img for img in data["images"]}
    categories = {cat["id"]: cat["name"] for cat in data["categories"]}

    rows = []
    for ann in data["annotations"]:
        img = images[ann["image_id"]]
        group = int(img["group"])
        rows.append(
            {
                "annotation_id": ann["id"],
                "image_id": ann["image_id"],
                "file_name": img["file_name"],
                "image_path": str(raw_dir / img["file_name"]),
                "group": group,
                "split": split_from_group(group),
                "set_name": set_name_from_file(img["file_name"]),
                "category_id": ann["category_id"],
                "species": categories[ann["category_id"]],
                "fish_id": ann["fish_id"],
                "length_cm": ann["length"],
                "bbox_x": ann["bbox"][0],
                "bbox_y": ann["bbox"][1],
                "bbox_w": ann["bbox"][2],
                "bbox_h": ann["bbox"][3],
                "segmentation": json.dumps(ann["segmentation"]),
                "width": img["width"],
                "height": img["height"],
                "side_up": ann.get("side_up", ""),
            }
        )

    df = pd.DataFrame(rows)
    exclusions = []
    fish_split_counts = df.groupby("fish_id")["split"].nunique()
    leaks = fish_split_counts[fish_split_counts > 1]
    if not leaks.empty and not args.keep_cross_split_duplicates:
        for fish_id in leaks.index:
            fish_rows = df[df["fish_id"] == fish_id]
            group_counts = fish_rows.groupby("group").size().sort_values(ascending=False)
            keep_groups = set(group_counts[group_counts == group_counts.iloc[0]].index.tolist())
            drop_rows = fish_rows[~fish_rows["group"].isin(keep_groups)]
            exclusions.extend(drop_rows["annotation_id"].astype(int).tolist())
        df = df[~df["annotation_id"].isin(exclusions)].reset_index(drop=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    split_out = Path(args.splits_out)
    split_out.write_text(json.dumps(OFFICIAL_SPLIT, indent=2))

    exclusions_out = Path(args.exclusions_out)
    exclusions_out.write_text(
        json.dumps(
            {
                "reason": "Dropped singleton/low-count fish_id duplicates that cross train/val/test splits.",
                "annotation_ids": exclusions,
            },
            indent=2,
        )
    )

    fish_split_counts = df.groupby("fish_id")["split"].nunique()
    leaks = fish_split_counts[fish_split_counts > 1]
    if not leaks.empty:
        raise SystemExit(f"Fish leakage across splits: {leaks.to_dict()}")

    print(f"wrote {out}")
    print(f"images: {len(images)}")
    print(f"annotations: {len(df)}")
    print(f"unique fish: {df['fish_id'].nunique()}")
    print(f"groups: {df['group'].nunique()}")
    print(df.groupby(["split", "set_name"]).size().unstack(fill_value=0))
    print("fish leakage across splits: 0")


if __name__ == "__main__":
    main()
