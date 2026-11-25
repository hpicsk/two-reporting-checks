"""
Kim & Lee's published model and a table lookup, on Kim & Lee's published split.

npj Comput. Mater. 11, 300 (2025) reports R^2 = 0.99 and RMSE 5.86 uOhm.cm on a
validation set built by: "The dataset was randomly divided, with 10% designated
as a validation set." The rows are positions on libraries, 43 alloy systems
supplying 70k+ rows, so a random row split can put positions adjacent to a
validation row, on the same library, in training. This scores their own network
and a lookup that fits nothing to the target on that same split, with the same
statistic.

THEIR MODEL is run as published, not reconstructed: `model/ANN.h5` is Dense and
ReLU only, so the forward pass is four matrix products in numpy and needs no
TensorFlow. Its inputs are normalised exactly as their pre-trained notebook
normalises them, min-max over train and validation together, because that is
their pipeline and the number being checked is theirs.

THE LOOKUP (src/metal/lookup.py) fits its range on the training rows alone, so
it is the weaker of the two in the one respect the two differ. That asymmetry
runs against the finding here rather than toward it.

CRITERIA, fixed before any number below is read:
  N1  lookup R^2 on their validation set.
      >= 0.95        -> their reported R^2 is dominated by interpolation within
                        sampled libraries; the critique lands, on a
                        named paper, using its own published data.
      0.50 - 0.95    -> partial. The model adds real value and the split still
                        flatters it; ships as a bound, not a headline.
      < 0.50         -> the account FAILS on this dataset. Their 0.99 is not a
                        lookup artifact.

Degrees of freedom:
  SPLIT        MEASURED -- theirs, shipped as two files; nothing is re-partitioned.
  FEATURES     MEASURED -- the nine their pre-trained notebook lists, in its order.
  Nothing is FITTED here and nothing is trained.

BASELINE: the authors' published ANN, at their published weights | obvious
alternative: none a referee can name -- the comparison is against the paper's
own model on the paper's own rows | tuning: theirs, as published (3x512, 1,926
epochs); the lookup has no tuning surface | discharge: run

    python reproduce/npj_split.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from metal.lookup import one_nn  # noqa: E402

BASE = ROOT / "data" / "npj"
OUT = ROOT / "outputs" / "npj" / "split.json"

# Their pre-trained notebook's norm_list, in its order: the model's input layer
# is bound to it, so this list is read from their code and not chosen here.
FEATURES = ["t", "depo", "r", "del_r", "del_EN", "S", "H", "VEC", "Th"]
# The six that are functions of composition alone. Thickness, deposition rate and
# homologous temperature are processing conditions, so a lookup restricted to
# these six is the one that answers "how close is this composition to a
# composition already measured".
COMPOSITION = ["r", "del_r", "del_EN", "S", "H", "VEC"]
TARGET = "resistivity"

N1_LANDS, N1_BOUND = 0.95, 0.50

# Their published numbers, from the "Machine learning modeling" section of the
# article. Asserted rather than quoted: if the forward pass below stops
# reproducing them, every npj number in the manuscript is resting on a
# reconstruction rather than on their model.
PAPER_TRAIN_RMSE = 3.94
PAPER_VAL_RMSE = 5.86
# The article prints the validation RMSE twice for the selected architecture,
# 5.86 in Results and 5.87 in Methods, so its own spread is 0.01. The tolerance
# sits above that and far below the whole-unit error a wrong normalisation or
# layer order would give, which is the failure it guards.
RMSE_TOLERANCE = 0.05


def forward(weights, x):
    """ANN.h5 is Dense+ReLU with a linear output. Four matrix products."""
    for kernel, bias in weights[:-1]:
        x = np.maximum(x @ kernel + bias, 0.0)
    kernel, bias = weights[-1]
    return (x @ kernel + bias).ravel()


def load_weights(path):
    order = ["layer0", "layer1", "layer2", "r"]
    with h5py.File(path, "r") as fh:
        group = fh["model_weights"]
        return [(np.asarray(group[n][n]["kernel:0"], dtype=float),
                 np.asarray(group[n][n]["bias:0"], dtype=float)) for n in order]


def scores(y, pred):
    residual = float(((y - pred) ** 2).sum())
    return {"rmse": float(np.sqrt(residual / len(y))),
            "r2": float(1.0 - residual / float(((y - y.mean()) ** 2).sum()))}


def main() -> None:
    if not (BASE / "csv" / "Train.csv").exists():
        raise FileNotFoundError(
            f"{BASE} is missing. Run reproduce/fetch_npj.py, which clones the "
            f"authors' repository at the pinned commit and verifies it.")
    train = pd.read_csv(BASE / "csv" / "Train.csv")
    val = pd.read_csv(BASE / "csv" / "Val.csv")
    y_train = train[TARGET].to_numpy(dtype=float)
    y_val = val[TARGET].to_numpy(dtype=float)

    print(f"train {len(train):,}  val {len(val):,}  total {len(train)+len(val):,}  "
          f"val share {len(val)/(len(train)+len(val)):.1%}")
    print(f"val resistivity: mean {y_val.mean():.1f}  sd {y_val.std(ddof=1):.1f}  "
          f"range {y_val.min():.1f}-{y_val.max():.1f} uOhm.cm\n")

    # --- their model, their normalisation, their weights ---------------------
    joint = pd.concat([train, val], axis=0).reset_index(drop=True)[FEATURES]
    lo, hi = joint.min(), joint.max()
    span = (hi - lo).replace(0, 1.0)
    weights = load_weights(BASE / "model" / "ANN.h5")
    model_train = forward(weights, ((train[FEATURES] - lo) / span).to_numpy(float))
    model_val = forward(weights, ((val[FEATURES] - lo) / span).to_numpy(float))
    model = {"train": scores(y_train, model_train), "val": scores(y_val, model_val),
             "hidden_layers": len(weights) - 1,
             "hidden_units": int(weights[0][0].shape[1])}

    print(f"their ANN, their weights, run in numpy "
          f"({model['hidden_layers']} hidden layers of {model['hidden_units']}):")
    print(f"  train RMSE {model['train']['rmse']:.2f} (paper {PAPER_TRAIN_RMSE})   "
          f"val RMSE {model['val']['rmse']:.2f} (paper {PAPER_VAL_RMSE})   "
          f"val R^2 {model['val']['r2']:.4f} (paper 0.99)\n")
    for split, published in (("train", PAPER_TRAIN_RMSE), ("val", PAPER_VAL_RMSE)):
        assert abs(model[split][
            "rmse"] - published) <= RMSE_TOLERANCE, (
            f"npj Comput. Mater. 11, 300 (2025) reports {split} RMSE "
            f"{published} uOhm.cm; this forward pass gives "
            f"{model[split]['rmse']:.3f}. The reproduction that licenses every "
            f"npj number in the manuscript has broken.")

    # --- the lookup, on their split ------------------------------------------
    lookup = {}
    for name, columns in (("all_features", FEATURES), ("composition_only", COMPOSITION)):
        predicted, _ = one_nn(train[columns].to_numpy(float), y_train,
                              val[columns].to_numpy(float))
        lookup[name] = scores(y_val, predicted) | {"n_features": len(columns)}
        record = lookup[name]
        print(f"  lookup, {name:17s} ({len(columns)} features)  "
              f"R^2 {record['r2']:.4f}   RMSE {record['rmse']:.2f} uOhm.cm\n")

    best = max(lookup["all_features"]["r2"], lookup["composition_only"]["r2"])
    verdict = ("LANDS" if best >= N1_LANDS else
               "BOUND" if best >= N1_BOUND else "FAILS")
    print(f"N1 VERDICT: {verdict}   best lookup R^2 = {best:.4f}   "
          f"against their model's {model['val']['r2']:.4f}\n")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "n_train": len(train), "n_val": len(val),
        "val_share": len(val) / (len(train) + len(val)),
        "val_target_sd": float(y_val.std(ddof=1)),
        "features": FEATURES, "composition_features": COMPOSITION,
        # Every column their repository ships, not only the ones consumed.
        # sec:datasets claims their release carries no system identifier, and a
        # list of what we selected cannot support that.
        "released_columns": list(train.columns),
        "paper": {"train_rmse": PAPER_TRAIN_RMSE, "val_rmse": PAPER_VAL_RMSE,
                  "val_r2_printed": "0.99"},
        "model": model, "lookup": lookup,
        "criteria": {"N1": {"best_lookup_r2": best, "lands_at": N1_LANDS,
                            "bound_at": N1_BOUND, "verdict": verdict}},
    }, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
