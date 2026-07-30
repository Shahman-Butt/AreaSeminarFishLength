"""Build the A3 research poster (single-model focus, graph-heavy, minimal text).
Embeds result charts as base64 so the poster is self-contained.
No ensemble (per requirement: single-model comparison). EfficientNet-B0 highlighted as
closest to baseline; stronger-recipe path shown as future work.
"""
import base64
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"


def b64(name):
    return base64.b64encode((FIG / f"{name}.png").read_bytes()).decode()


CSS = """
  @page { size: A3 portrait; margin: 0; }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { background: #e8e8e6; }
  body { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; color: #0b0b0b; }
  .poster { width: 297mm; height: 420mm; margin: 0 auto; background: #fcfcfb;
    padding: 8mm 10mm 7mm 10mm; display: flex; flex-direction: column; gap: 4mm; }
  @media print { html, body { background: #fcfcfb; } .poster { margin: 0; } }
  :root { --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --grid:#e1e0d9;
    --card:#f5f4f1; --accent:#2a78d6; --deep:#184f95; --green:#0a7a4a; --gold:#b06f00; }
  header { border-bottom: 0.8mm solid var(--accent); padding-bottom: 3mm; }
  header h1 { font-size: 24pt; line-height: 1.12; letter-spacing: -0.01em; }
  header h1 .q { color: var(--deep); }
  header .authors { margin-top: 2mm; font-size: 11pt; }
  header .affil { margin-top: 0.6mm; font-size: 9pt; color: var(--ink2); }
  .row { display: flex; gap: 4mm; }
  .block { background: var(--card); border-radius: 2.5mm; padding: 3.5mm 4mm; flex: 1; }
  .block h2 { font-size: 12pt; color: var(--deep); margin-bottom: 1.8mm; }
  .block p, .block li { font-size: 9.4pt; line-height: 1.34; }
  .block ul { list-style: none; }
  .block li { margin-bottom: 1.2mm; padding-left: 3.4mm; position: relative; }
  .block li::before { content: "▸"; position: absolute; left: 0; color: var(--accent); font-size: 8pt; top: 0.2mm; }
  .block .small { font-size: 8.4pt; color: var(--ink2); }
  strong { font-weight: 650; }
  .qbox { margin-top: 1.6mm; background: #fcfcfb; border-left: 1.1mm solid var(--accent);
    border-radius: 1.4mm; padding: 2mm 2.8mm; font-size: 9.4pt; line-height: 1.32; }
  .pipe { display: flex; align-items: stretch; gap: 1.2mm; margin: 1.5mm 0; }
  .pipe .step { flex: 1; background: #fcfcfb; border: 0.3mm solid var(--grid); border-radius: 1.3mm;
    padding: 1.4mm 0.8mm; text-align: center; font-size: 7.6pt; line-height: 1.2;
    display: flex; align-items: center; justify-content: center; }
  .pipe .arrow { align-self: center; color: var(--muted); font-size: 9pt; }
  img.chart { width: 100%; height: auto; border-radius: 1.5mm; }
  .takeaway { margin-top: 1.6mm; background: #fcfcfb; border-radius: 1.4mm; padding: 2mm 2.8mm;
    font-size: 9.3pt; line-height: 1.32; border-left: 1.1mm solid var(--accent); }
  .tiles { display: flex; gap: 2.5mm; margin-top: 1.5mm; }
  .tile { flex: 1; background: #fcfcfb; border-radius: 1.6mm; padding: 2mm 2.6mm; border: 0.3mm solid var(--grid); }
  .tile .label { font-size: 7.8pt; color: var(--ink2); }
  .tile .value { font-size: 17pt; font-weight: 650; margin-top: 0.4mm; }
  .tile .unit { font-size: 9pt; font-weight: 400; color: var(--ink2); }
  table { border-collapse: collapse; width: 100%; margin-top: 1.5mm; font-size: 8.1pt; }
  th, td { border: 0.25mm solid var(--grid); padding: 1mm 1.6mm; text-align: left; }
  th { background: #eef3fb; color: var(--deep); }
  td.n, th.n { text-align: right; font-variant-numeric: tabular-nums; }
  .future { background: #fdf6e8; border-left: 1.2mm solid var(--gold); border-radius: 2mm; padding: 3.5mm 4mm; flex: 1.4; }
  .future h2 { font-size: 12pt; color: var(--gold); margin-bottom: 1.8mm; }
  .future li::before { color: var(--gold); }
  footer { border-top: 0.3mm solid var(--grid); padding-top: 1.6mm; display: flex; gap: 5mm;
    font-size: 7.2pt; color: var(--ink2); line-height: 1.3; margin-top: auto; }
  footer div { flex: 1; } footer strong { color: var(--ink); }
"""

HTML = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>AutoFish A3 Poster: Fish Length Estimation with Vision Foundation Models</title>
<style>__CSS__</style></head><body>
<div class="poster">

  <header>
    <h1>Fish Length Estimation with Vision Foundation Models:<br>
      <span class="q">Which Single Encoder Comes Closest to the AutoFish Baseline?</span></h1>
    <div class="authors">Abu Bakar &nbsp;·&nbsp; Laksh Jiwani &nbsp;·&nbsp; Shahman Butt</div>
    <div class="affil">Area Seminar: Deep Learning for Maritime Vision Applications, VACOT, University of Rostock
      &nbsp;·&nbsp; Supervisor: Bohan Zhuang, M.Sc. &nbsp;·&nbsp; Professor: Stefan Oehmcke</div>
  </header>

  <!-- ROW 1 -->
  <div class="row">
    <div class="block">
      <h2>Motivation &amp; Question</h2>
      <ul>
        <li>Fish length drives fisheries stock assessment; manual measuring is slow and invasive.</li>
        <li>AutoFish (Bengtson et al., WACVW 2025) gives images + a CNN length-regression baseline.</li>
      </ul>
      <div class="qbox"><strong>Question:</strong> can a modern encoder or vision foundation model
        beat the reproduced MobileNetV2 baseline at fish length regression, under identical data,
        split, and metrics?</div>
    </div>
    <div class="block">
      <h2>Data &amp; Setup</h2>
      <ul>
        <li><strong>1,500</strong> images · <strong>454</strong> fish · <strong>18,157</strong> annotations · <strong>25</strong> groups.</li>
        <li><strong>Group-level split</strong> 15/5/5; no fish in two splits (leak fish-113 removed).</li>
        <li>Test = 3,759 (1,879 non-occluded + 1,880 occluded).</li>
        <li><strong>Metric:</strong> mean absolute error (MAE, cm), lower is better.</li>
      </ul>
    </div>
    <div class="block">
      <h2>Method / Workflow</h2>
      <div class="pipe">
        <div class="step">Fish crop<br>(masked)</div><div class="arrow">→</div>
        <div class="step">Resize<br>224²</div><div class="arrow">→</div>
        <div class="step">Image<br>encoder</div><div class="arrow">→</div>
        <div class="step">+ box<br>(4 val.)</div><div class="arrow">→</div>
        <div class="step">Head<br>[512,128,1]</div><div class="arrow">→</div>
        <div class="step"><strong>Length</strong><br>(cm)</div>
      </div>
      <ul>
        <li><strong>Only the encoder is swapped;</strong> split, input, head, L1 loss, metrics identical.</li>
        <li>Learning rate set per encoder type (baseline reproduces paper: Adam 1e-3).</li>
      </ul>
    </div>
  </div>

  <!-- ROW 2: main ranking + reproduction/findings -->
  <div class="row">
    <div class="block" style="flex:1.5; border:0.5mm solid var(--accent);">
      <h2>Single-model comparison (full-test MAE)</h2>
      <img class="chart" src="data:image/png;base64,__RANK__">
      <div class="takeaway"><strong>Takeaway:</strong> among single models, MobileNetV2 (0.771 cm) is
        best and every foundation model trails it; supervised CNNs fill the top. <strong>EfficientNet-B0
        (0.781 cm) comes within 0.010 cm</strong> of the baseline.</div>
    </div>
    <div class="block" style="flex:1;">
      <h2>Baseline reproduction</h2>
      <p class="small">Non-occluded test MAE, our run vs. the AutoFish paper:</p>
      <div class="tiles">
        <div class="tile"><div class="label">Paper</div><div class="value">0.62<span class="unit"> cm</span></div></div>
        <div class="tile" style="border-color:var(--accent);"><div class="label">Ours</div><div class="value">0.633<span class="unit"> cm</span></div></div>
      </div>
      <p class="small" style="margin-top:1.5mm;">Δ 0.013 cm → pipeline validated.</p>
      <h2 style="margin-top:3mm;">Key findings</h2>
      <ul>
        <li><strong>MobileNetV2 is the best single model</strong> (0.771 cm).</li>
        <li><strong>EfficientNet-B0 nearly matches it</strong> (0.781) at a basic recipe.</li>
        <li><strong>Patch tokens reliably beat CLS</strong> for DINOv2 (3 seeds).</li>
        <li>Occlusion hurts every model (baseline 0.633 → 0.909).</li>
      </ul>
    </div>
  </div>

  <!-- ROW 3: three result charts -->
  <div class="row">
    <div class="block">
      <h2>Closest to the baseline</h2>
      <img class="chart" src="data:image/png;base64,__CLOSE__">
    </div>
    <div class="block">
      <h2>Patch tokens beat CLS (DINOv2)</h2>
      <img class="chart" src="data:image/png;base64,__PATCH__">
    </div>
    <div class="block">
      <h2>Species classification</h2>
      <img class="chart" src="data:image/png;base64,__SPEC__">
    </div>
  </div>

  <!-- ROW 4: configurations table + future work -->
  <div class="row">
    <div class="block" style="flex:1.6;">
      <h2>Models &amp; configurations tried</h2>
      <table>
        <tr><th>Encoder</th><th>Type</th><th>Adaptation</th><th>Recipe (Adam)</th><th class="n">MAE</th></tr>
        <tr><td>MobileNetV2 <em>(baseline)</em></td><td>CNN</td><td>full FT</td><td>1e-3, 200 ep, b32</td><td class="n">0.771</td></tr>
        <tr><td>EfficientNet-B0</td><td>CNN</td><td>full FT</td><td>1e-4, 100 ep, b16</td><td class="n">0.781</td></tr>
        <tr><td>ConvNeXt-Tiny</td><td>CNN</td><td>full FT</td><td>1e-4, 100 ep, b16</td><td class="n">0.914</td></tr>
        <tr><td>CLIP ViT-B/32</td><td>Transf.</td><td>frozen / last-block</td><td>1e-4, 100 ep</td><td class="n">1.002 / 0.958</td></tr>
        <tr><td>DINOv2 ViT-S/14 (patch)</td><td>Transf.</td><td>frozen, patch pool</td><td>1e-4, 100 ep</td><td class="n">1.261</td></tr>
        <tr><td>DINOv2 ViT-S/14 (CLS)</td><td>Transf.</td><td>frozen / last-blk / full</td><td>1e-4 … 1e-6</td><td class="n">1.44–2.13</td></tr>
      </table>
      <p class="small" style="margin-top:1.2mm;">Species classification (same encoders, cross-entropy):
        ConvNeXt 99.6% · MobileNetV2 99.1% · DINOv2 98.2% · CLIP 95.1% accuracy.</p>
    </div>
    <div class="future">
      <h2>Future work: path to beat the baseline</h2>
      <ul>
        <li><strong>EfficientNet-B0 is only 0.010 cm behind</strong> at a basic recipe — a stronger recipe
          should cross it.</li>
        <li><strong>Now testing</strong> (validation-selected): EfficientNet-B0 &amp; ConvNeXt-Tiny with
          <strong>cosine LR schedule + weight decay</strong>, tuned LR (5e-4/1e-3), 200 epochs.</li>
        <li>Then: EfficientNet-B2, patch pooling for CLIP, larger ViT variants.</li>
        <li>Mask segmentation as a third task; DINOv3 (pending weight access).</li>
      </ul>
    </div>
  </div>

  <!-- ROW 5: limitations -->
  <div class="row">
    <div class="block">
      <h2>Limitations</h2>
      <ul>
        <li>Single training run for most single models (multi-seed done for the DINOv2 patch-vs-CLS study).</li>
        <li>Foundation models tested at small scale (ViT-S/14, ViT-B/32) with limited fine-tuning budgets.</li>
        <li>No single model beats the baseline yet; mask segmentation not yet evaluated.</li>
      </ul>
    </div>
  </div>

  <footer>
    <div><strong>Reference:</strong> S. H. Bengtson et al., "AutoFish", WACV Workshops 2025 (arXiv:2501.03767).
      Dataset: Hugging Face <em>vapaau/autofish</em>.</div>
    <div><strong>Reproducibility:</strong> fixed seeds, official group split, versioned configs, requirements.txt,
      saved checkpoints &amp; per-fish predictions. NVIDIA RTX 5000 Ada (32 GB).</div>
    <div><strong>AI statement:</strong> generative AI assisted with code scaffolding and drafting; all
      experiments, results, and conclusions are the authors' own and were verified by the authors.</div>
  </footer>

</div></body></html>"""

out = (HTML
       .replace("__CSS__", CSS)
       .replace("__RANK__", b64("length_ranking"))
       .replace("__CLOSE__", b64("closest_single_models"))
       .replace("__PATCH__", b64("patch_vs_cls"))
       .replace("__SPEC__", b64("species_accuracy")))
(ROOT / "poster" / "AutoFish_A3_poster.html").write_text(out, encoding="utf-8")
print("poster written,", len(out), "chars")
