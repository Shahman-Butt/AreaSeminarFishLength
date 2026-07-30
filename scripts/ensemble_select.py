"""Rigorous ensemble selection: choose the ensemble on the VALIDATION set,
then report it once on the TEST set (never tuned on test).

Reads val + test per-fish predictions for each candidate model.
"""
import pandas as pd
import numpy as np
import itertools
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "MobileNetV2": "baseline_official",
    "ConvNeXt": "convnext_tiny_official",
    "CLIP_lb": "clip_vitb32_lastblock_from_frozen",
    "CLIP_fr": "clip_vitb32_frozen",
    "DINOpatch": "dinov2_vits14_patchtokens_frozen",
}


def load(split):
    fname = "val_metrics.predictions.csv" if split == "val" else "test_metrics.predictions.csv"
    frames = {}
    for k, r in MODELS.items():
        d = pd.read_csv(ROOT / "runs" / r / fname)[["annotation_id", "set_name", "length_cm", "pred_cm"]]
        frames[k] = d.rename(columns={"pred_cm": k}).set_index("annotation_id")
    base = frames["MobileNetV2"][["set_name", "length_cm"]].copy()
    for k in MODELS:
        base = base.join(frames[k][[k]])
    return base


def mae(df, cols, mask=None):
    pred = df[list(cols)].mean(axis=1).values
    e = np.abs(pred - df["length_cm"].values)
    return e[mask].mean() if mask is not None else e.mean()


val = load("val")
test = load("test")

# --- individual model val/test MAE (for reference) ---
print("Individual models (unweighted, full split MAE):")
for k in MODELS:
    print(f"  {k:12s} val={mae(val,[k]):.3f}  test={mae(test,[k]):.3f}")

# --- select best unweighted ensemble on VALIDATION ---
best = None
for n in range(1, len(MODELS) + 1):
    for combo in itertools.combinations(MODELS.keys(), n):
        v = mae(val, combo)
        if best is None or v < best[1]:
            best = (combo, v)

combo, val_mae = best
tmask_no = test["set_name"].isin(["Set1", "Set2"]).values
tmask_oc = (test["set_name"] == "All").values
print("\n=== Ensemble selected ON VALIDATION (unweighted mean) ===")
print("  members :", " + ".join(combo))
print(f"  val MAE : {val_mae:.3f} cm")
print("\n=== Reported ONCE on TEST ===")
print(f"  test full     : {mae(test, combo):.3f} cm   (baseline MobileNetV2 = {mae(test,['MobileNetV2']):.3f})")
print(f"  test non-occ  : {mae(test, combo, tmask_no):.3f} cm   (baseline = {mae(test,['MobileNetV2'],tmask_no):.3f})")
print(f"  test occluded : {mae(test, combo, tmask_oc):.3f} cm   (baseline = {mae(test,['MobileNetV2'],tmask_oc):.3f})")
delta = mae(test, ['MobileNetV2']) - mae(test, combo)
print(f"\n  Improvement over baseline: {delta:+.3f} cm ({100*delta/mae(test,['MobileNetV2']):+.1f}%)")
print("  ", "BEATS BASELINE" if delta > 0 else "does not beat baseline")
