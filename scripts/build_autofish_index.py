# ============================================================================
# build_autofish_index.py — turns the raw, nested Hugging Face download into
# one clean spreadsheet, AND catches/fixes the fish-113 data-leakage problem.
#
# WHERE THIS FITS IN THE PROJECT (the very first step of the whole pipeline):
#   scripts/download_autofish.py   -->  data/raw/autofish/{images/, annotations.json}
#                                              |
#                                              v
#   THIS FILE reads that raw download and produces:
#                                              |
#                                              v
#   data/processed/index.csv        -- one row per fish-in-a-photo (18,157 rows)
#   data/processed/splits.json      -- which group numbers are train/val/test
#   data/processed/exclusions.json  -- which rows were removed, and why
#                                              |
#                                              v
#   scripts/make_crops.py reads index.csv next, to know what to crop
#   src/autofish_vfm/data.py's CropDataset reads index.csv every time
#   training or evaluation runs
#
# HOW TO RUN THIS FILE:
#   python scripts/build_autofish_index.py --raw-dir data/raw/autofish
# ============================================================================

import argparse
import json
from pathlib import Path

import pandas as pd


# The OFFICIAL group -> split assignment, taken directly from the AutoFish
# paper's own training-release code (not something we invented ourselves).
# Every fish belongs to exactly one of these 25 numbered "groups" (a batch of
# fish photographed together); this dict decides which whole groups go into
# training vs. validation vs. testing. Splitting by whole GROUP (rather than
# by individual photo) is what prevents the same physical fish from ending up
# on both sides of the train/test boundary — see split_from_group() below.
OFFICIAL_SPLIT = {
    "train": [2, 3, 4, 5, 7, 8, 9, 12, 13, 15, 16, 18, 19, 23, 24],
    "val": [1, 6, 11, 17, 25],
    "test": [10, 14, 20, 21, 22],
}


def set_name_from_file(file_name: str) -> str:
    """Figures out whether a photo shows easy (separated) or hard (piled up,
    overlapping) fish, purely from its filename number.

    The dataset's own naming convention (discovered from the raw files, not
    documented anywhere else) is:
        images numbered 00001-00020 -> "Set1"  (fish laid out separately)
        images numbered 00021-00040 -> "Set2"  (fish laid out separately)
        images numbered 00041-00060 -> "All"   (fish piled/overlapping)
    Example: file_name="00007.png" -> image_no=7 -> falls in 1..20 -> "Set1".
    Set1 and Set2 together are what the project calls "non-occluded"; "All"
    is what the project calls "occluded" — this one function is the single
    source of truth for that whole distinction used throughout the project.
    """
    image_no = int(Path(file_name).stem)  # ".stem" strips the folder + ".png", leaving just "00007" -> 7
    if 1 <= image_no <= 20:
        return "Set1"
    if 21 <= image_no <= 40:
        return "Set2"
    if 41 <= image_no <= 60:
        return "All"
    raise ValueError(f"Unexpected image number in {file_name}")  # fail loudly rather than silently
    # mis-labelling a photo — a wrong occlusion label would quietly corrupt every result table


def split_from_group(group: int) -> str:
    """Looks up which split ("train"/"val"/"test") a given group number
    belongs to, using the OFFICIAL_SPLIT dict above.
    Example: split_from_group(10) -> "test" (group 10 is in OFFICIAL_SPLIT["test"]).
    """
    for split, groups in OFFICIAL_SPLIT.items():
        if group in groups:
            return split
    raise ValueError(f"Group {group} is not in the official split")  # catches typos/unexpected
    # group numbers immediately, instead of silently dropping fish from every split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw/autofish")
    parser.add_argument("--out", default="data/processed/index.csv")
    parser.add_argument("--splits-out", default="data/processed/splits.json")
    parser.add_argument("--exclusions-out", default="data/processed/exclusions.json")
    # Escape hatch for debugging/comparison only: normally we ALWAYS remove
    # leaking fish (see below); this flag lets you rebuild the index WITHOUT
    # that safety fix, to see what the raw numbers would have looked like.
    # None of this project's actual experiments use this flag.
    parser.add_argument("--keep-cross-split-duplicates", action="store_true")
    args = parser.parse_args()

    # ---- Step 1: load the raw COCO-style annotations.json ----
    raw_dir = Path(args.raw_dir)
    ann_path = raw_dir / "annotations.json"
    data = json.loads(ann_path.read_text())

    # The raw file stores images and categories as flat lists; turn them into
    # dictionaries keyed by their id, so we can look any of them up instantly
    # while looping over annotations below (instead of re-scanning a list
    # every time, which would be extremely slow for 18,000+ annotations).
    images = {img["id"]: img for img in data["images"]}
    categories = {cat["id"]: cat["name"] for cat in data["categories"]}

    # ---- Step 2: flatten everything into ONE row per fish-in-a-photo ----
    rows = []
    for ann in data["annotations"]:
        img = images[ann["image_id"]]        # look up which photo this fish annotation belongs to
        group = int(img["group"])
        rows.append(
            {
                "annotation_id": ann["id"],           # unique ID for this ONE fish-in-a-photo
                "image_id": ann["image_id"],
                "file_name": img["file_name"],
                "image_path": str(raw_dir / img["file_name"]),
                "group": group,
                "split": split_from_group(group),      # "train"/"val"/"test", decided above
                "set_name": set_name_from_file(img["file_name"]),  # "Set1"/"Set2"/"All"
                "category_id": ann["category_id"],
                "species": categories[ann["category_id"]],  # e.g. "cod" — human-readable species name
                "fish_id": ann["fish_id"],             # identifies the PHYSICAL fish (can repeat
                                                         # across many photos/annotations)
                "length_cm": ann["length"],             # the hand-measured true length — our regression target
                "bbox_x": ann["bbox"][0],               # the rectangle around this fish IN THE ORIGINAL
                "bbox_y": ann["bbox"][1],               # (un-cropped) photo — used later by data.py to
                "bbox_w": ann["bbox"][2],               # build the "scale hint" fed to every model
                "bbox_h": ann["bbox"][3],
                "segmentation": json.dumps(ann["segmentation"]),  # the fish's exact outline (as a
                # JSON string of polygon points) — used later by make_crops.py to build the black mask
                "width": img["width"],                  # original photo's full width/height, needed
                "height": img["height"],                # to normalise the bbox numbers above into 0..1
                "side_up": ann.get("side_up", ""),
            }
        )

    df = pd.DataFrame(rows)  # now we have one big table: 18,157 rows, one per fish-in-a-photo

    # ---- Step 3: THE LEAKAGE AUDIT — find and remove any fish that crosses splits ----
    # In plain words: even though we split by whole GROUP (not by individual
    # photo), it's still possible for the SAME PHYSICAL FISH (same fish_id)
    # to appear in more than one group by mistake — which would still leak
    # its identity across the train/test boundary. This block checks for
    # exactly that and fixes it automatically.
    exclusions = []

    # For every unique fish_id, count how many DIFFERENT splits it appears
    # in. A healthy fish_id should have exactly 1 (all its photos are in the
    # same split, because they're all in the same group). More than 1 means trouble.
    fish_split_counts = df.groupby("fish_id")["split"].nunique()
    leaks = fish_split_counts[fish_split_counts > 1]

    if not leaks.empty and not args.keep_cross_split_duplicates:
        for fish_id in leaks.index:
            # Look at every annotation of this ONE problem fish...
            fish_rows = df[df["fish_id"] == fish_id]
            # ...count how many times it appears in each of its groups...
            group_counts = fish_rows.groupby("group").size().sort_values(ascending=False)
            # ...and decide the fish "really belongs" to whichever group(s)
            # it appears in MOST — e.g. if a fish has 40 annotations in group
            # 22 and only 1 annotation in group 5, group 22 is clearly its
            # real home and the single annotation in group 5 is a stray duplicate.
            keep_groups = set(group_counts[group_counts == group_counts.iloc[0]].index.tolist())
            # Everything NOT in the "real home" group(s) gets marked for removal.
            drop_rows = fish_rows[~fish_rows["group"].isin(keep_groups)]
            exclusions.extend(drop_rows["annotation_id"].astype(int).tolist())
        # Actually remove those rows from the table.
        df = df[~df["annotation_id"].isin(exclusions)].reset_index(drop=True)
        # (In this project's real dataset, this logic finds exactly ONE
        # leaking fish — fish_id 113 — and removes its single stray
        # annotation (id 3759), which is why the final non-occluded test
        # count is 1,879 instead of 1,880. See docs/DEFENSE_A_TO_Z...md §7
        # for the full worked-out example with real numbers.)

    # ---- Step 4: write everything out to disk ----
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)   # data/processed/index.csv — read by every other script from here on

    split_out = Path(args.splits_out)
    split_out.write_text(json.dumps(OFFICIAL_SPLIT, indent=2))   # a record of exactly which
    # group numbers went into which split — kept as its own file for transparency/reproducibility

    exclusions_out = Path(args.exclusions_out)
    exclusions_out.write_text(
        json.dumps(
            {
                "reason": "Dropped singleton/low-count fish_id duplicates that cross train/val/test splits.",
                "annotation_ids": exclusions,   # exactly which rows were removed, and why — this
                # is the paper trail a reviewer (or professor) can check to verify the leakage fix
            },
            indent=2,
        )
    )

    # ---- Step 5: RE-CHECK for leakage after the fix, and refuse to continue if any remains ----
    # This is a genuine safety net, not just a print statement: if the
    # cleanup logic above somehow failed to fully separate a fish (a bug, or
    # a fish with a genuine 50/50 split across two groups with equal counts),
    # this line stops the whole script with an error rather than silently
    # shipping a leaky dataset. Every time this script has actually been run
    # for this project, this check has passed (0 leaks remaining).
    fish_split_counts = df.groupby("fish_id")["split"].nunique()
    leaks = fish_split_counts[fish_split_counts > 1]
    if not leaks.empty:
        raise SystemExit(f"Fish leakage across splits: {leaks.to_dict()}")

    # ---- Step 6: print a human-readable summary so you can eyeball the result ----
    print(f"wrote {out}")
    print(f"images: {len(images)}")
    print(f"annotations: {len(df)}")
    print(f"unique fish: {df['fish_id'].nunique()}")
    print(f"groups: {df['group'].nunique()}")
    print(df.groupby(["split", "set_name"]).size().unstack(fill_value=0))  # a small table showing
    # exactly how many fish landed in each split x occlusion-set combination — a quick sanity check
    # that the numbers look like what's documented in the project's README (e.g. 1,879 non-occluded test)
    print("fish leakage across splits: 0")


if __name__ == "__main__":
    main()
