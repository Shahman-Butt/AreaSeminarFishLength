import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="data/processed/index.csv")
    parser.add_argument("--crops-dir", default="data/processed/crops")
    args = parser.parse_args()

    df = pd.read_csv(args.index)
    crops_dir = Path(args.crops_dir)
    missing = [
        ann_id
        for ann_id in df["annotation_id"]
        if not (crops_dir / f"{int(ann_id):06d}.png").exists()
    ]

    leaks = df.groupby("fish_id")["split"].nunique()
    leaks = leaks[leaks > 1]

    print(f"rows: {len(df)}")
    print(f"images: {df['image_id'].nunique()}")
    print(f"unique fish: {df['fish_id'].nunique()}")
    print(f"groups: {df['group'].nunique()}")
    print(f"missing crops: {len(missing)}")
    print(f"fish leakage across splits: {len(leaks)}")
    print(df.groupby(["split", "set_name"]).size().unstack(fill_value=0))

    if missing:
        raise SystemExit(f"Missing first crop ids: {missing[:10]}")
    if not leaks.empty:
        raise SystemExit(f"Leakage fish ids: {leaks.index.tolist()[:10]}")


if __name__ == "__main__":
    main()
