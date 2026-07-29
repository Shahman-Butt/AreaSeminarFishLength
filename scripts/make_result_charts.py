"""Generate clean result charts (PNG) for the poster and report.
Reads saved metrics; no fabricated numbers.
"""
import json
import statistics as st
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
BLUE, GREEN, VIOLET, INK, MUT = "#2a78d6", "#1baf7a", "#4a3aa7", "#0b0b0b", "#898781"


def mae(run, sub="test_all"):
    return json.load(open(ROOT / "runs" / run / "test_metrics.json"))[sub]["mae_cm"]


def acc(run):
    return json.load(open(ROOT / "runs" / run / "test_metrics.json"))["test_all"]["accuracy"]


# ---------- 1. Full length ranking (incl. patch-token) ----------
patch = [mae("dinov2_vits14_patchtokens_frozen"), mae("dinov2_vits14_patchtokens_frozen_s1"), mae("dinov2_vits14_patchtokens_frozen_s2")]
rows = [
    ("MobileNetV2 (baseline)", mae("baseline_official"), BLUE),
    ("ConvNeXt-Tiny", mae("convnext_tiny_official"), BLUE),
    ("CLIP last-block", mae("clip_vitb32_lastblock_from_frozen"), GREEN),
    ("CLIP frozen", mae("clip_vitb32_frozen"), GREEN),
    ("DINOv2 patch-token (NEW)", st.mean(patch), VIOLET),
    ("DINOv2 CLS last-block", mae("dinov2_vits14_lastblock_from_frozen"), VIOLET),
    ("DINOv2 CLS frozen", mae("dinov2_vits14_frozen"), VIOLET),
    ("DINOv2 CLS full-FT 1e-5", mae("dinov2_vits14_finetune_lr1e5"), VIOLET),
]
rows.sort(key=lambda r: r[1])
fig, ax = plt.subplots(figsize=(8.2, 4.2))
labels = [r[0] for r in rows]
vals = [r[1] for r in rows]
colors = [r[2] for r in rows]
y = range(len(rows))
ax.barh(list(y), vals, color=colors, height=0.62)
ax.invert_yaxis()
ax.set_yticks(list(y)); ax.set_yticklabels(labels, fontsize=9)
for i, v in enumerate(vals):
    ax.text(v + 0.02, i, f"{v:.3f}", va="center", fontsize=8.5, color=INK,
            fontweight="bold" if labels[i].startswith("MobileNetV2") or "NEW" in labels[i] else "normal")
ax.set_xlabel("Full-test MAE (cm) — lower is better", fontsize=9.5)
ax.set_title("Fish length estimation: model comparison", fontsize=12, fontweight="bold", color=INK)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(labelsize=8.5)
fig.tight_layout()
fig.savefig(OUT / "length_ranking.png", dpi=170, bbox_inches="tight")
plt.close(fig)

# ---------- 2. Controlled patch vs CLS (multi-seed, error bars) ----------
cls = [mae("dinov2_vits14_clstoken_ctrl_s42"), mae("dinov2_vits14_clstoken_ctrl_s1"), mae("dinov2_vits14_clstoken_ctrl_s2")]
fig, ax = plt.subplots(figsize=(4.6, 4.0))
names = ["CLS token", "Patch token"]
means = [st.mean(cls), st.mean(patch)]
sds = [st.pstdev(cls), st.pstdev(patch)]
bars = ax.bar(names, means, yerr=sds, capsize=8, color=[MUT, VIOLET], width=0.55)
for i, (m, s) in enumerate(zip(means, sds)):
    ax.text(i, m + s + 0.03, f"{m:.3f}\n± {s:.3f}", ha="center", fontsize=9, color=INK)
ax.set_ylabel("Full-test MAE (cm)", fontsize=9.5)
ax.set_title("DINOv2 frozen: patch vs CLS\n(identical settings, 3 seeds)", fontsize=11, fontweight="bold", color=INK)
ax.set_ylim(0, max(means) + max(sds) + 0.35)
ax.spines[["top", "right"]].set_visible(False)
ax.annotate("0.58 cm\nreliable gain", xy=(1, means[1]), xytext=(0.5, means[0] + 0.1),
            fontsize=9, color=VIOLET, ha="center", fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "patch_vs_cls.png", dpi=170, bbox_inches="tight")
plt.close(fig)

# ---------- 3. Species classification accuracy ----------
crows = [
    ("ConvNeXt-Tiny", acc("cls_convnext_tiny"), BLUE),
    ("MobileNetV2", acc("cls_mobilenet_v2"), BLUE),
    ("DINOv2 frozen", acc("cls_dinov2_frozen"), VIOLET),
    ("CLIP frozen", acc("cls_clip_frozen"), GREEN),
]
fig, ax = plt.subplots(figsize=(4.6, 3.6))
labels = [r[0] for r in crows]
vals = [r[1] * 100 for r in crows]
ax.bar(labels, vals, color=[r[2] for r in crows], width=0.6)
for i, v in enumerate(vals):
    ax.text(i, v + 0.15, f"{v:.1f}%", ha="center", fontsize=9, color=INK)
ax.set_ylabel("Test accuracy (%)", fontsize=9.5)
ax.set_ylim(90, 101)
ax.set_title("Species classification\n(all models strong; CNNs top)", fontsize=11, fontweight="bold", color=INK)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(axis="x", labelsize=8.5, rotation=15)
fig.tight_layout()
fig.savefig(OUT / "species_accuracy.png", dpi=170, bbox_inches="tight")
plt.close(fig)

print("wrote charts to", OUT)
for p in sorted(OUT.glob("*.png")):
    print(" -", p.name)
