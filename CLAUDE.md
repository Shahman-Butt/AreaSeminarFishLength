# Fish Length Estimation with Vision Foundation Models

Project goal: reproduce the AutoFish CNN regression baseline from Bengtson et al.,
AutoFish: Dataset and Benchmark for Fine-grained Analysis of Fish, then compare
encoder swaps to Vision Foundation Models in later work.

Current milestone: Q1 only, baseline reproduction.

Fixed decisions:
- Dataset: Hugging Face `vapaau/autofish`, local snapshot in `data/raw/autofish`.
- Source paper/code: AutoFish WACVW 2025 and official Bitbucket training release.
- Baseline target: REG MobileNetV2 length regressor, reported MAE 0.62 cm on
  Set1+Set2 and 1.38 cm on All.
- Split: use the official release split for reproduction:
  train `[2,3,4,5,7,8,9,12,13,15,16,18,19,23,24]`,
  val `[1,6,11,17,25]`,
  test `[10,14,20,21,22]`.
- Evaluation regimes: non-occluded = Set1 + Set2, occluded = All.
- Set mapping assumption from file order: image numbers 1-20 are Set1, 21-40
  are Set2, 41-60 are All.
- Completed Q1 server run: `/home/sb2597/autofish_baseline_repro/runs/baseline_official`.
- Completed Q1 local result copies:
  `runs/baseline_official/test_metrics.json`,
  `runs/baseline_official/test_metrics.predictions.csv`,
  `runs/baseline_official/history.csv`.
- Reproduced REG MobileNetV2 result: 0.633 cm MAE on Set1+Set2 and 0.909 cm
  MAE on All, compared with paper targets 0.62 cm and 1.38 cm.

Conventions:
- Python 3.11 on the server/GPU environment.
- Keep generated data under `data/processed`.
- Keep trained weights and metrics under `results`.
- Do not split by image or crop; group-level split is mandatory.
