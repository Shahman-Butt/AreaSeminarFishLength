# ============================================================================
# download_autofish.py — step 0 of the whole project: fetch the dataset.
#
# WHERE THIS FITS: this is the VERY FIRST script in the pipeline. It has no
# input from anywhere else in the project — it just pulls the AutoFish
# dataset down from Hugging Face's public dataset hub.
#
#   (nothing, this is the start)
#         |
#         v
#   THIS FILE  -->  data/raw/autofish/{images/..., annotations.json}
#         |
#         v
#   scripts/build_autofish_index.py reads that raw download next
#
# HOW TO RUN THIS FILE:
#   python scripts/download_autofish.py
#
# Using a SCRIPT for this (instead of manually downloading files by hand) is
# part of what makes the whole project reproducible: anyone with this repo
# can fetch the EXACT SAME dataset with one command, rather than us having to
# distribute 16 GB of images ourselves alongside the code.
# ============================================================================

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download  # the official Hugging Face helper for
                                                  # downloading an entire dataset repository


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/raw/autofish")
    # "vapaau/autofish" is the dataset's address on huggingface.co — the
    # AutoFish paper's authors published their data there.
    parser.add_argument("--repo-id", default="vapaau/autofish")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    # snapshot_download quietly does a LOT of work for us: it downloads
    # every file in the dataset repository (1,500 images + the
    # annotations.json file), skips re-downloading anything already present
    # (safe to re-run if interrupted), and verifies file integrity — all in
    # one call.
    path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",       # tells Hugging Face this is a DATASET repo, not a model repo
        local_dir=str(out_dir),    # where to save everything: data/raw/autofish/
    )
    print(path)   # prints the final local folder path, as a quick confirmation it worked


if __name__ == "__main__":
    main()
