# ============================================================================
# metrics.py — turns raw predictions into the numbers we actually report.
#
# WHERE THIS FITS IN THE PROJECT:
#   train_baseline.py calls regression_metrics() after every epoch (to decide
#   whether to save a new "best.pt" checkpoint).
#   evaluate.py calls regression_metrics() once at the very end, on the test
#   set, to produce the headline numbers everyone sees in the poster/report.
#   train_classifier.py / evaluate_classifier.py call classification_metrics()
#   the same way, but for the species-guessing task instead of length.
#
# Nothing in this file ever touches a model or the GPU — it's pure arithmetic
# on two lists of numbers: what the model guessed, and what was actually true.
# ============================================================================

import math

import numpy as np


def classification_metrics(y_true, y_pred, num_classes=None):
    """Accuracy and macro-F1 for species classification.

    y_true, y_pred are integer class indices (e.g. 0=cod, 1=haddock, ...) —
    see train_classifier.py's SPECIES/LABEL_MAP for exactly which number
    means which species.

    WORKED EXAMPLE with 3 fish:
        y_true = [0, 0, 1]   (truth: cod, cod, haddock)
        y_pred = [0, 1, 1]   (guess: cod, haddock, haddock)
        -> accuracy = 2/3 = 0.667  (2 out of 3 guesses were exactly right)
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    # ---- Accuracy: the simplest possible score ----
    # "What fraction of fish did we guess the species of correctly?"
    # (y_true == y_pred) makes an array of True/False; np.mean() over
    # True/False treats True as 1 and False as 0, so the mean IS the fraction correct.
    acc = float(np.mean(y_true == y_pred))

    # Figure out which class indices actually exist, so we compute one F1
    # score per species even if a species never appears in y_pred by chance.
    classes = range(num_classes) if num_classes else sorted(set(y_true.tolist()) | set(y_pred.tolist()))

    f1s = []
    for c in classes:
        # For THIS species `c` only, treat it as a "is it this species, yes
        # or no?" problem and count the four possible outcomes:
        tp = int(np.sum((y_pred == c) & (y_true == c)))  # true positive:  said c, WAS c
        fp = int(np.sum((y_pred == c) & (y_true != c)))  # false positive: said c, was NOT c
        fn = int(np.sum((y_pred != c) & (y_true == c)))  # false negative: said NOT c, WAS c
        # Precision: "of the times we said species c, how often were we right?"
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        # Recall: "of all the fish that really were species c, how many did we catch?"
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        # F1 = the harmonic mean of precision and recall — a single number
        # that is only high when BOTH precision and recall are high (unlike a
        # plain average, which can be fooled by one very high, one very low).
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)

    return {
        "accuracy": acc,
        # "Macro" F1 = average the per-species F1 scores with EQUAL weight
        # per species, regardless of how many fish of each species exist.
        # This matters because our dataset is imbalanced (e.g. 1,000 haddock
        # test fish vs. only 80 saithe — see docs/DEFENSE guide §16); a plain
        # accuracy score could look great just by nailing the common species
        # while quietly failing the rare ones. Macro-F1 does not let that hide.
        "macro_f1": float(np.mean(f1s)),
        "n": int(len(y_true)),   # how many fish this score was computed over
    }


def regression_metrics(y_true, y_pred):
    """MAE / RMSE / MAPE / bias / R² for length regression — the main
    scoreboard of the whole project.

    WORKED EXAMPLE with 3 fish, true lengths 30, 25, 40 cm, predicted 31, 24, 38 cm:
        errors = pred - true = [+1, -1, -2]     (cm)
        MAE  = mean(|+1|, |-1|, |-2|) = mean(1, 1, 2) = 1.33 cm
               -> "on average we're off by 1.33 cm"
        bias = mean(+1, -1, -2) = -0.67 cm
               -> "on average we slightly UNDER-predict" (negative = too short)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true   # positive error = over-predicted (guessed too long);
                             # negative error = under-predicted (guessed too short)

    # MAE (Mean Absolute Error): average size of the mistake, ignoring
    # direction. THE headline number quoted everywhere in this project
    # (e.g. "MobileNetV2 = 0.771 cm").
    mae = np.mean(np.abs(err))

    # RMSE (Root Mean Squared Error): squares the errors before averaging,
    # then square-roots the result. Squaring makes big mistakes count MUCH
    # more than small ones (a 4 cm error contributes 16x more than a 1 cm
    # error, before the final square root) — so RMSE > MAE always, and a
    # large gap between the two is a sign that a few predictions were very
    # far off, even if most were fine.
    rmse = math.sqrt(np.mean(err**2))

    # MAPE (Mean Absolute Percentage Error): the same idea as MAE, but scaled
    # by each fish's own true length, then turned into a percentage.
    # np.maximum(..., 1e-8) is just a safety net against dividing by zero for
    # a (never-actually-occurring) fish with length 0.
    mape = np.mean(np.abs(err) / np.maximum(np.abs(y_true), 1e-8)) * 100.0

    # Bias: the SIGNED average error (no absolute value). Near zero means
    # the model's mistakes cancel out (sometimes too long, sometimes too
    # short, roughly equally); consistently positive/negative means the
    # model has a systematic tendency to over- or under-predict.
    bias = np.mean(err)

    # R² (coefficient of determination): "what fraction of the natural
    # variation in fish length does the model explain?" 1.0 = perfect
    # predictions; 0.0 = no better than always guessing the average length.
    ss_res = np.sum(err**2)                              # leftover ("residual") error the model didn't explain
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)       # total variation in the true lengths
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "mae_cm": float(mae),
        "rmse_cm": float(rmse),
        "mape_percent": float(mape),
        "bias_cm": float(bias),
        "r2": float(r2),
        "n": int(len(y_true)),   # how many fish this score was computed over — always reported
                                   # alongside the metrics so a reader can judge how much data
                                   # backs up the number (e.g. 3,759 for the full test set).
    }
