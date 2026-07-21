"""Qualitative result figures for the poster (uses local crops + saved predictions).

Produces PNGs in results/qualitative/:
  1. dataset_examples.png     - example fish crops (non-occluded + occluded) to show the task
  2. mobilenet_vs_dino.png    - cases where MobileNetV2 is right but DINOv2 is far off
  3. clip_wins.png            - cases where CLIP beats MobileNetV2
Each crop is labelled with true length and each model's prediction.
No GPU / no retraining: reads runs/<run>/test_metrics.predictions.csv and data/processed/crops.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CROPS = ROOT / "data" / "processed" / "crops"
OUT = ROOT / "results" / "qualitative"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, RED, GREEN, INK = "#184f95", "#c0392b", "#1baf7a", "#0b0b0b"


def load(run):
    df = pd.read_csv(ROOT / "runs" / run / "test_metrics.predictions.csv")
    df["abs_err"] = (df["pred_cm"] - df["length_cm"]).abs()
    return df


def crop_path(aid):
    return CROPS / f"{int(aid):06d}.png"


def panel(ax, aid, title_lines, colors):
    img = mpimg.imread(crop_path(aid))
    ax.imshow(img)
    ax.set_xticks([]); ax.set_yticks([])
    y = 1.02
    for line, c in zip(title_lines, colors):
        ax.text(0.5, y, line, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=8.5, color=c, fontweight="bold" if y == 1.02 else "normal")
        y += 0.075


def fig_dataset_examples():
    mob = load("baseline_official")
    non = mob[mob.set_name.isin(["Set1", "Set2"])].sort_values("abs_err").head(200)
    occ = mob[mob.set_name == "All"].sort_values("abs_err").head(200)
    picks = list(non.sample(3, random_state=1)["annotation_id"]) + \
            list(occ.sample(3, random_state=1)["annotation_id"])
    labels = ["non-occluded"] * 3 + ["occluded"] * 3
    fig, axes = plt.subplots(1, 6, figsize=(13, 2.6))
    for ax, aid, lab in zip(axes, picks, labels):
        row = mob[mob.annotation_id == aid].iloc[0]
        panel(ax, aid, [f"{lab}", f"{row.species}, {row.length_cm:.1f} cm"], [INK, "#52514e"])
    fig.suptitle("AutoFish example crops: the model sees one masked fish at a time",
                 fontsize=11, fontweight="bold", y=1.12)
    fig.tight_layout()
    fig.savefig(OUT / "dataset_examples.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_mobilenet_vs_dino():
    mob = load("baseline_official").set_index("annotation_id")
    dino = load("dinov2_vits14_lastblock_from_frozen").set_index("annotation_id")
    common = mob.join(dino[["pred_cm", "abs_err"]], rsuffix="_dino")
    common["gap"] = common["abs_err_dino"] - common["abs_err"]
    top = common.sort_values("gap", ascending=False).head(6).reset_index()
    fig, axes = plt.subplots(1, 6, figsize=(13, 3.0))
    for ax, (_, r) in zip(axes, top.iterrows()):
        panel(ax, r.annotation_id,
              [f"true {r.length_cm:.1f} cm",
               f"MobileNet {r.pred_cm:.1f}",
               f"DINOv2 {r.pred_cm_dino:.1f}"],
              [INK, BLUE, RED])
    fig.suptitle("Where MobileNetV2 is accurate but DINOv2 fails (largest error gap)",
                 fontsize=11, fontweight="bold", y=1.14)
    fig.tight_layout()
    fig.savefig(OUT / "mobilenet_vs_dino.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_clip_wins():
    mob = load("baseline_official").set_index("annotation_id")
    clip = load("clip_vitb32_lastblock_from_frozen").set_index("annotation_id")
    common = mob.join(clip[["pred_cm", "abs_err"]], rsuffix="_clip")
    common["gain"] = common["abs_err"] - common["abs_err_clip"]  # positive = CLIP better
    # require CLIP clearly good and MobileNet clearly off, among occluded
    cand = common[(common.set_name == "All") & (common.abs_err_clip < 0.5)]
    top = cand.sort_values("gain", ascending=False).head(6).reset_index()
    fig, axes = plt.subplots(1, 6, figsize=(13, 3.0))
    for ax, (_, r) in zip(axes, top.iterrows()):
        panel(ax, r.annotation_id,
              [f"true {r.length_cm:.1f} cm",
               f"CLIP {r.pred_cm_clip:.1f}",
               f"MobileNet {r.pred_cm:.1f}"],
              [INK, GREEN, BLUE])
    fig.suptitle("Where CLIP beats MobileNetV2 (occluded fish, CLIP near-perfect)",
                 fontsize=11, fontweight="bold", y=1.14)
    fig.tight_layout()
    fig.savefig(OUT / "clip_wins.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_dataset_examples()
    fig_mobilenet_vs_dino()
    fig_clip_wins()
    print("Wrote figures to", OUT)
    for p in sorted(OUT.glob("*.png")):
        print(" -", p.name)
