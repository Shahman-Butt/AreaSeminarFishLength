"""Error analysis across all trained models, from saved per-fish prediction files.

Answers the supervisor's questions:
  - In which situations does each model win (by species, fish size, occlusion)?
  - Where do MobileNetV2 and DINOv2 disagree most?
  - When does CLIP beat MobileNetV2 and vice versa?

Reads runs/<run>/test_metrics.predictions.csv (no GPU, no retraining needed).
Writes CSV tables + a markdown summary into results/error_analysis/.
"""
import pandas as pd
import numpy as np
from pathlib import Path

RUNS = {
    "MobileNetV2": "baseline_official",
    "ConvNeXt-Tiny": "convnext_tiny_official",
    "CLIP-frozen": "clip_vitb32_frozen",
    "CLIP-lastblock": "clip_vitb32_lastblock_from_frozen",
    "DINOv2-frozen": "dinov2_vits14_frozen",
    "DINOv2-lastblock": "dinov2_vits14_lastblock_from_frozen",
    "DINOv2-full-1e5": "dinov2_vits14_finetune_lr1e5",
    "DINOv2-full-1e6": "dinov2_vits14_finetune_lr1e6",
}

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "error_analysis"
OUT.mkdir(parents=True, exist_ok=True)


def load():
    frames = {}
    for name, run in RUNS.items():
        p = ROOT / "runs" / run / "test_metrics.predictions.csv"
        df = pd.read_csv(p)
        df["abs_err"] = (df["pred_cm"] - df["length_cm"]).abs()
        df["occ"] = np.where(df["set_name"] == "All", "occluded", "non-occluded")
        frames[name] = df
    return frames


def mae(df):
    return df["abs_err"].mean()


def main():
    frames = load()
    md = ["# Error Analysis (from saved predictions, no retraining)\n"]
    md.append(f"Models: {', '.join(frames)}\n")

    # 1) Overall + occlusion
    rows = []
    for name, df in frames.items():
        rows.append({
            "model": name,
            "MAE_full": round(mae(df), 3),
            "MAE_non_occluded": round(mae(df[df.occ == "non-occluded"]), 3),
            "MAE_occluded": round(mae(df[df.occ == "occluded"]), 3),
            "occlusion_penalty": round(mae(df[df.occ == "occluded"]) - mae(df[df.occ == "non-occluded"]), 3),
        })
    occ = pd.DataFrame(rows).sort_values("MAE_full")
    occ.to_csv(OUT / "by_occlusion.csv", index=False)
    md.append("## 1. Overall and by occlusion (MAE cm)\n")
    md.append(occ.to_markdown(index=False))
    md.append("")

    # 2) By species
    species = sorted(frames["MobileNetV2"]["species"].unique())
    sp_rows = []
    for name, df in frames.items():
        r = {"model": name}
        for s in species:
            sub = df[df.species == s]
            r[s] = round(mae(sub), 3) if len(sub) else np.nan
        sp_rows.append(r)
    sp = pd.DataFrame(sp_rows)
    sp.to_csv(OUT / "by_species.csv", index=False)
    md.append("## 2. By species (MAE cm)\n")
    md.append(sp.to_markdown(index=False))
    counts = frames["MobileNetV2"].groupby("species").size()
    md.append("\nTest sample count per species: " +
              ", ".join(f"{s}={int(counts[s])}" for s in species))
    md.append("")

    # 3) By length range (deciles based on true length)
    base = frames["MobileNetV2"]
    edges = np.quantile(base["length_cm"], np.linspace(0, 1, 6))
    edges[0] -= 1e-6
    labels = [f"{edges[i]:.1f}-{edges[i+1]:.1f}cm" for i in range(len(edges) - 1)]
    len_rows = []
    for name, df in frames.items():
        binned = pd.cut(df["length_cm"], bins=edges, labels=labels)
        r = {"model": name}
        for lab in labels:
            sub = df[binned == lab]
            r[lab] = round(mae(sub), 3) if len(sub) else np.nan
        len_rows.append(r)
    lr = pd.DataFrame(len_rows)
    lr.to_csv(OUT / "by_length_range.csv", index=False)
    md.append("## 3. By fish length range (MAE cm, quintiles)\n")
    md.append(lr.to_markdown(index=False))
    md.append("")

    # 4) Head-to-head: CLIP (best) vs MobileNetV2, per fish
    a = frames["MobileNetV2"][["annotation_id", "species", "occ", "length_cm", "abs_err"]].rename(columns={"abs_err": "mobilenet_err"})
    b = frames["CLIP-lastblock"][["annotation_id", "abs_err"]].rename(columns={"abs_err": "clip_err"})
    h = a.merge(b, on="annotation_id")
    h["clip_minus_mobilenet"] = h["clip_err"] - h["mobilenet_err"]
    clip_wins = (h["clip_err"] < h["mobilenet_err"]).mean() * 100
    md.append("## 4. Head-to-head: CLIP(last-block) vs MobileNetV2 (per fish)\n")
    md.append(f"- CLIP beats MobileNetV2 on **{clip_wins:.1f}%** of individual test fish "
              f"(even though MobileNetV2 has the lower average).")
    # where CLIP wins by occlusion
    for cond in ["non-occluded", "occluded"]:
        sub = h[h.occ == cond]
        w = (sub["clip_err"] < sub["mobilenet_err"]).mean() * 100
        md.append(f"- On {cond} fish, CLIP wins {w:.1f}% of the time.")
    # where CLIP wins by species
    md.append("- CLIP win-rate by species: " + ", ".join(
        f"{s}={(h[h.species==s]['clip_err']<h[h.species==s]['mobilenet_err']).mean()*100:.0f}%"
        for s in species))
    h.to_csv(OUT / "clip_vs_mobilenet_per_fish.csv", index=False)
    md.append("")

    # 5) MobileNetV2 vs DINOv2: biggest disagreements (for qualitative figure)
    d = frames["DINOv2-lastblock"][["annotation_id", "abs_err", "pred_cm"]].rename(columns={"abs_err": "dino_err", "pred_cm": "dino_pred"})
    m = frames["MobileNetV2"][["annotation_id", "species", "occ", "set_name", "group", "length_cm", "abs_err", "pred_cm"]].rename(columns={"abs_err": "mobilenet_err", "pred_cm": "mobilenet_pred"})
    disagree = m.merge(d, on="annotation_id")
    disagree["dino_minus_mobilenet_err"] = disagree["dino_err"] - disagree["mobilenet_err"]
    disagree = disagree.sort_values("dino_minus_mobilenet_err", ascending=False)
    disagree.to_csv(OUT / "mobilenet_vs_dino_disagreement.csv", index=False)
    md.append("## 5. Biggest MobileNetV2-vs-DINOv2 disagreements (top = MobileNet right, DINOv2 wrong)\n")
    top = disagree.head(10)[["annotation_id", "species", "occ", "length_cm", "mobilenet_pred", "dino_pred", "mobilenet_err", "dino_err"]].round(2)
    md.append(top.to_markdown(index=False))
    md.append("")

    (OUT / "SUMMARY.md").write_text("\n".join(md), encoding="utf-8")
    print("Wrote analysis to", OUT)
    print("\n".join(md[:40]))


if __name__ == "__main__":
    main()
