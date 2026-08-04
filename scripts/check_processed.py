# ============================================================================
# check_processed.py — a final "did the data pipeline actually work?" sanity check.
#
# WHERE THIS FITS: run AFTER build_autofish_index.py and make_crops.py, as a
# quick double-check before trusting the processed data for training. It
# doesn't produce anything new — it just READS index.csv and the crops
# folder and verifies two things stayed true:
#   1. every fish listed in index.csv actually has a matching crop PNG on disk
#   2. no fish leaks across train/val/test splits (a second, independent
#      check of the same leakage rule build_autofish_index.py already enforces)
#
# HOW TO RUN THIS FILE:
#   python scripts/check_processed.py
# If everything is fine it just prints a summary and exits normally; if
# anything is wrong it raises SystemExit with the specific problem IDs,
# stopping any pipeline script that runs it as a "gate" before training.
# ============================================================================

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

    # ---- Check 1: does every fish in the spreadsheet have a real crop file? ----
    # If make_crops.py was interrupted partway through, or run against a
    # different index.csv than the current one, some fish could be "listed"
    # in the CSV but have no matching PNG — this would silently crash
    # training later (FileNotFoundError deep inside a DataLoader worker) if
    # not caught here first, with a much less helpful error message.
    missing = [
        ann_id
        for ann_id in df["annotation_id"]
        if not (crops_dir / f"{int(ann_id):06d}.png").exists()
    ]

    # ---- Check 2: independent re-verification of the leakage fix ----
    # Same logic as the check inside build_autofish_index.py — group by
    # fish_id, see how many DIFFERENT splits each fish appears in. Anything
    # >1 means a fish is still leaking across train/val/test. Running this
    # check again here, as a totally separate script, means a bug
    # accidentally introduced later wouldn't go unnoticed just because it
    # happened to sidestep the check inside build_autofish_index.py.
    leaks = df.groupby("fish_id")["split"].nunique()
    leaks = leaks[leaks > 1]

    # ---- Print a compact summary a human can eyeball at a glance ----
    print(f"rows: {len(df)}")
    print(f"images: {df['image_id'].nunique()}")
    print(f"unique fish: {df['fish_id'].nunique()}")
    print(f"groups: {df['group'].nunique()}")
    print(f"missing crops: {len(missing)}")
    print(f"fish leakage across splits: {len(leaks)}")
    print(df.groupby(["split", "set_name"]).size().unstack(fill_value=0))  # how many fish per
    # split x occlusion-set combination — should match the numbers documented in the README

    # ---- Fail loudly (non-zero exit code) if either check found a problem ----
    # This makes it safe to chain this script into an automated pipeline:
    # "build index -> make crops -> check_processed.py -> (only if this
    # passes) start training" — a broken data pipeline stops here rather
    # than wasting hours of GPU time training on bad/incomplete data.
    if missing:
        raise SystemExit(f"Missing first crop ids: {missing[:10]}")  # show only the first
        # 10 problem IDs, enough to start debugging without flooding the terminal
    if not leaks.empty:
        raise SystemExit(f"Leakage fish ids: {leaks.index.tolist()[:10]}")


if __name__ == "__main__":
    main()
