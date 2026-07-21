# Error Analysis (from saved predictions, no retraining)

Models: MobileNetV2, ConvNeXt-Tiny, CLIP-frozen, CLIP-lastblock, DINOv2-frozen, DINOv2-lastblock, DINOv2-full-1e5, DINOv2-full-1e6

## 1. Overall and by occlusion (MAE cm)

| model            |   MAE_full |   MAE_non_occluded |   MAE_occluded |   occlusion_penalty |
|:-----------------|-----------:|-------------------:|---------------:|--------------------:|
| MobileNetV2      |      0.771 |              0.633 |          0.909 |               0.277 |
| ConvNeXt-Tiny    |      0.914 |              0.814 |          1.014 |               0.2   |
| CLIP-lastblock   |      0.958 |              0.842 |          1.074 |               0.232 |
| CLIP-frozen      |      1.002 |              0.898 |          1.106 |               0.208 |
| DINOv2-lastblock |      1.439 |              1.34  |          1.537 |               0.196 |
| DINOv2-frozen    |      1.738 |              1.69  |          1.786 |               0.096 |
| DINOv2-full-1e5  |      1.778 |              1.636 |          1.919 |               0.282 |
| DINOv2-full-1e6  |      2.132 |              2.075 |          2.189 |               0.114 |

## 2. By species (MAE cm)

| model            |   cod |   haddock |   hake |   horse_mackerel |   other |   saithe |   whiting |
|:-----------------|------:|----------:|-------:|-----------------:|--------:|---------:|----------:|
| MobileNetV2      | 0.833 |     0.587 |  1.194 |            0.736 |   1.297 |    0.723 |     0.649 |
| ConvNeXt-Tiny    | 0.919 |     0.713 |  1.505 |            0.983 |   1.276 |    0.825 |     0.77  |
| CLIP-frozen      | 1.109 |     0.88  |  1.45  |            0.88  |   1.156 |    0.911 |     0.869 |
| CLIP-lastblock   | 1.044 |     0.793 |  1.392 |            0.852 |   1.208 |    0.806 |     0.884 |
| DINOv2-frozen    | 1.742 |     1.489 |  2.786 |            1.356 |   1.475 |    2.309 |     1.728 |
| DINOv2-lastblock | 1.062 |     0.862 |  2.522 |            0.905 |   5.76  |    1.363 |     1.45  |
| DINOv2-full-1e5  | 2.042 |     1.446 |  1.818 |            1.874 |   4.055 |    1.652 |     1.417 |
| DINOv2-full-1e6  | 1.629 |     1.663 |  4.385 |            1.362 |   1.62  |    3.748 |     2.468 |

Test sample count per species: cod=840, haddock=1000, hake=400, horse_mackerel=440, other=160, saithe=80, whiting=839

## 3. By fish length range (MAE cm, quintiles)

| model            |   22.5-28.0cm |   28.0-31.0cm |   31.0-34.0cm |   34.0-37.5cm |   37.5-50.5cm |
|:-----------------|--------------:|--------------:|--------------:|--------------:|--------------:|
| MobileNetV2      |         0.935 |         0.608 |         0.677 |         0.603 |         1.093 |
| ConvNeXt-Tiny    |         0.925 |         0.857 |         0.864 |         0.75  |         1.235 |
| CLIP-frozen      |         1.2   |         0.944 |         0.8   |         0.839 |         1.304 |
| CLIP-lastblock   |         1.088 |         0.848 |         0.821 |         0.832 |         1.275 |
| DINOv2-frozen    |         1.725 |         1.542 |         1.355 |         1.633 |         2.709 |
| DINOv2-lastblock |         1.867 |         0.838 |         0.928 |         1.304 |         2.564 |
| DINOv2-full-1e5  |         2.513 |         1.475 |         1.548 |         1.587 |         1.749 |
| DINOv2-full-1e6  |         1.63  |         1.819 |         1.28  |         2.094 |         4.547 |

## 4. Head-to-head: CLIP(last-block) vs MobileNetV2 (per fish)

- CLIP beats MobileNetV2 on **38.6%** of individual test fish (even though MobileNetV2 has the lower average).
- On non-occluded fish, CLIP wins 36.7% of the time.
- On occluded fish, CLIP wins 40.5% of the time.
- CLIP win-rate by species: cod=37%, haddock=39%, hake=40%, horse_mackerel=40%, other=42%, saithe=46%, whiting=37%

## 5. Biggest MobileNetV2-vs-DINOv2 disagreements (top = MobileNet right, DINOv2 wrong)

|   annotation_id | species   | occ          |   length_cm |   mobilenet_pred |   dino_pred |   mobilenet_err |   dino_err |
|----------------:|:----------|:-------------|------------:|-----------------:|------------:|----------------:|-----------:|
|           10585 | other     | occluded     |        22.5 |            23.2  |       38.66 |            0.7  |      16.16 |
|            7583 | other     | occluded     |        25   |            24.95 |       38.19 |            0.05 |      13.19 |
|           10062 | other     | non-occluded |        22.5 |            23.62 |       35.99 |            1.12 |      13.49 |
|           10212 | other     | non-occluded |        22.5 |            23.07 |       34.62 |            0.57 |      12.12 |
|           10151 | other     | non-occluded |        22.5 |            23.11 |       34.61 |            0.61 |      12.11 |
|           10092 | other     | non-occluded |        22.5 |            23.18 |       34.48 |            0.68 |      11.98 |
|           10503 | other     | occluded     |        22.5 |            23.33 |       34.35 |            0.83 |      11.85 |
|           10356 | other     | occluded     |        22.5 |            23.23 |       33.92 |            0.73 |      11.42 |
|           14473 | other     | occluded     |        26   |            27.7  |       38.35 |            1.7  |      12.35 |
|           10132 | other     | non-occluded |        22.5 |            24.42 |       34.96 |            1.92 |      12.46 |
