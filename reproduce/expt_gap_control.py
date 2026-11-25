"""
The control: experimental band gaps of distinct compounds, with property and
model held at HTEM-DB band gap's.

A control dataset without libraries can differ from the combinatorial datasets
on three axes at once: its rows are distinct compounds rather than positions on
libraries, its property can be computed rather than measured, and the property
itself can differ. This control holds the last two axes fixed.
Matbench's `matbench_expt_gap` is experimentally measured band gap, the
property HTEM-DB's second target measures, on 4,604 compositions (Zhuo,
Mansouri Tehrani & Brgoch, J. Phys. Chem. Lett. 9, 1668, 2018; Dunn et al., npj
Comput. Mater. 6, 138, 2020). Its rows sit a median of one per chemical system,
where HTEM-DB band gap's sit a median of 43 per library (median_rows_per_system
in outputs/expt_gap/control.json, median_rows_per_library in
outputs/htem/split.json). The kind of property, the model and the partition
are HTEM-DB band gap's (sec:protocol); the sampling structure is not.

The protocol IS reproduce/htem_split.py's: this script imports that module and
calls its out-of-fold and bootstrap functions with its constants, so the match
is a fact about the import graph rather than about two files agreeing. Five
random folds drawn from each of HTEM-DB's ten seeds, pooled out-of-fold
predictions under each, scores averaged over the ten partitions, the same lookup
and the same forest of 200 trees with every other setting at its library
default. The cluster bootstrap resamples compounds, this dataset's sampling
unit, so its endpoints are estimated over evaluation units and comparable in
kind to HTEM-DB's library-resampled intervals. There is no blocked protocol: the
dataset has no libraries, which is what makes it a control.

FEATURES are elemental atomic fractions parsed from the formula. HTEM-DB's feature
set adds thickness and deposition temperature because HTEM-DB ships them; this
dataset ships no processing variable, so the feature set is the composition
every dataset in this study shares.

CRITERIA, fixed before any number below is read. Both compare this dataset's
random-split interval against the combinatorial pair that bounds the quantity,
which at commit time is HTEM-DB band gap on both: random-split lookup R^2 0.658
[0.555, 0.741] and model margin +0.128 [+0.093, +0.172]. The thresholds are
read from outputs/htem/split.json at run time rather than typed here, so a
re-run of HTEM-DB moves them.
  E1  THE LOOKUP SITS BELOW THE COMBINATORIAL RANGE -> the upper endpoint of
      this dataset's lookup R^2 interval is below the lower endpoint of the
      smallest combinatorial lookup interval. Missed if the intervals touch.
  E2  THE MODEL'S MARGIN SITS ABOVE THE COMBINATORIAL RANGE -> the lower
      endpoint of this dataset's margin interval is above the upper endpoint of
      the largest combinatorial margin interval. Missed if the intervals touch.
Both are reported as they fall. Both run twice: on the mean over the ten random
partitions, and on the first partition alone, the one partition that HTEM-DB
had when these criteria were fixed, whose numbers are the ones quoted above.
sec:esi-draw-spread prints both comparisons, because they disagree on E2.

WHAT EACH OUTCOME MEANS, written before the run:
  E1 and E2 both met  -> the DFT objection is discharged: with property and
      model fixed, the lookup's reach and the model's gain both move with
      the sampling structure. Destination: the control paragraph of
      sec:methods, one row of tab:main panel a, and one sentence beside the
      JARVIS sentence in par:margin. Nothing else in the manuscript changes.
  E1 met, E2 missed   -> the lookup's dependence on library structure holds;
      the model's larger gain off-library is a JARVIS result, not a general
      one. Destination: par:margin, where the sentence crediting the model's
      gain is scoped to JARVIS.
  E1 missed           -> the lookup reaches the combinatorial range on distinct
      experimental compounds, so par:margin's attribution of the lookup result
      to the libraries is false and the claim is about composition-only
      regression, which Meredig et al. established. Stop before editing the
      manuscript; the result goes to the author first.

THE ROWS WITH A NONZERO GAP, a criterion fixed on 2026-11-21 before any number
it reads. About half of this dataset's targets are metals at exactly zero,
whereas every HTEM-DB band gap exceeds zero, so E1 alone cannot tell the
absence of libraries from a target that is zero on half the rows. The same
protocol therefore runs again on the rows whose gap exceeds zero, under the
same seeds, with the same compound bootstrap:
  E3  THE LOOKUP SITS BELOW THE COMBINATORIAL RANGE ON THE NONZERO ROWS -> the
      upper endpoint of the lookup's R^2 interval on those rows is below the
      lower endpoint of the smallest combinatorial lookup interval, E1's
      threshold. Missed if the intervals touch.
The margin on those rows is computed and printed as measured, under no
criterion. What each outcome changes, written before the run:
  E3 met    -> par:margin keeps its reading that the libraries rather than the
      kind of property give the lookup its R^2 on HTEM-DB, and states the zero
      share and the lookup's score on the nonzero rows beside it; the Matbench
      paragraph of sec:datasets states the zero share and the nonzero rows;
      sec:esi-draw-spread prints the margin on those rows.
  E3 missed -> the zero-gap rows, not only the absence of libraries, can give
      the Matbench lookup its low R^2, so par:margin's reading is cut and the
      comparison is printed as measured. The result goes to the author before
      the manuscript is edited.
Confounds that E3 does not remove, named before the run: the nonzero rows are
fewer than HTEM-DB band gap's, their gaps span a wider range, and they are
distinct compounds rather than alloys. par:margin's reading stays a suggestion
for these reasons whichever way E3 falls.

Degrees of freedom:
  TARGET       MEASURED -- experimental band gap in eV, untransformed, as HTEM-DB's
               band gap target is; about half the rows are metals at exactly
               zero, and the share is recorded in the output.
  FEATURES     CHOSEN -- composition fractions, see above.
  N_FOLDS, seeds, forest size, bootstrap size  CHOSEN -- imported from
               reproduce/htem_split.py; nothing is set here.
  Nothing is FITTED: no criterion here is matched to a target.

BASELINE: the same default 200-tree random forest as HTEM-DB's | obvious
alternative: a referee says the forest is weak on 81 sparse columns, so the
margin understates what a model can gain | tuning: none, on either arm, as on
HTEM-DB -- and an understated margin runs against E2, not toward it | discharge:
par:margin. The two margin intervals separate over the ten partitions and
overlap on the first, and the sentence declines the general claim rather than
asserting its negation, so a margin the forest understates has nothing there to
overturn

    python reproduce/expt_gap_control.py
"""
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Composition

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "expt_gap" / "matbench_expt_gap.json.gz"
HTEM_RESULTS = ROOT / "outputs" / "htem" / "split.json"
OUT = ROOT / "outputs" / "expt_gap" / "control.json"

# The protocol is HTEM-DB's, imported rather than copied: constants, the
# out-of-fold loop with its lookup and forest, and the cluster bootstrap.
_spec = importlib.util.spec_from_file_location(
    "htem_split", ROOT / "reproduce" / "htem_split.py")
htem_split = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(htem_split)


def features(formulas):
    fractions = [Composition(f).fractional_composition.as_dict() for f in formulas]
    elements = sorted({e for d in fractions for e in d})
    X = np.array([[d.get(e, 0.0) for e in elements] for d in fractions], dtype=float)
    systems = np.array([" ".join(sorted(d)) for d in fractions])
    return X, elements, systems


def bounds(htem, lookup_key="random_lookup_r2", margin_key="model_margin"):
    """The combinatorial pair bounding each quantity, read from HTEM-DB's file.

    The default keys are the means over the ten random partitions. The
    first-partition keys give the same two bounds on the first of those
    partitions alone, which sec:esi-draw-spread compares this control against
    as well."""
    targets = htem["targets"].values()
    lookup_lo = min((t["ci"][lookup_key]["lo"], t["label"]) for t in targets)
    margin_hi = max((t["ci"][margin_key]["hi"], t["label"]) for t in targets)
    return lookup_lo, margin_hi


def main() -> None:
    if not CACHE.exists():
        raise FileNotFoundError(
            f"{CACHE} is missing. Run reproduce/fetch_expt_gap.py to rebuild it.")
    if not HTEM_RESULTS.exists():
        raise FileNotFoundError(
            f"{HTEM_RESULTS} is missing. The criteria are read from it; run "
            f"reproduce/htem_split.py first.")
    htem = json.loads(HTEM_RESULTS.read_text())
    (lookup_lo, lookup_bound_by), (margin_hi, margin_bound_by) = bounds(htem)

    df = pd.read_json(CACHE, orient="split", compression="gzip")
    X, elements, systems = features(df["composition"])
    y = df["gap expt"].to_numpy(float)
    duplicates = len(X) - len(np.unique(X.round(6), axis=0))
    rows_per_system = list(Counter(systems).values())

    print(f"rows {len(y):,}  chemical systems {len(set(systems)):,}  "
          f"elements {len(elements)}  features {X.shape[1]}")
    print(f"target: mean {y.mean():.2f}  sd {y.std():.2f}  "
          f"range {y.min():.2f}..{y.max():.2f}   zero-gap share {(y == 0).mean():.1%}   "
          f"median rows/system {np.median(rows_per_system):.0f}   "
          f"rows sharing a composition {duplicates}\n")

    preds, partitions = {}, []
    for seed in htem_split.FOLD_SEEDS:
        lookup, model, distance = htem_split.out_of_fold(
            X, y, htem_split.random_partition(seed), None)
        preds[f"random_lookup_{seed}"], preds[f"random_model_{seed}"] = lookup, model
        entry = {"seed": seed, "lookup_r2": htem_split.r2(y, lookup),
                 "model_r2": htem_split.r2(y, model),
                 "median_nn_distance": float(np.median(distance))}
        entry["margin"] = entry["model_r2"] - entry["lookup_r2"]
        partitions.append(entry)
        print(f"  RANDOM {seed:<5d} lookup R^2 {entry['lookup_r2']:+.4f}   "
              f"random-forest R^2 {entry['model_r2']:+.4f}   "
              f"margin {entry['margin']:+.4f}")
    record = {key: float(np.mean([p[key] for p in partitions]))
              for key in ("lookup_r2", "model_r2", "median_nn_distance")}
    margin = record["model_r2"] - record["lookup_r2"]
    print(f"  RANDOM   lookup R^2 {record['lookup_r2']:+.4f}   "
          f"random-forest R^2 {record['model_r2']:+.4f}   "
          f"median NN distance {record['median_nn_distance']:.4f}")

    # Every compound is its own cluster: the dataset's sampling unit is the row.
    first = htem_split.FOLD_SEEDS[0]
    mean = htem_split.mean_over_partitions
    intervals = htem_split.cluster_bootstrap(y, preds, np.arange(len(y)), {
        "random_lookup_r2": lambda s: mean(s, "lookup"),
        "random_model_r2": lambda s: mean(s, "model"),
        "model_margin": lambda s: mean(s, "model") - mean(s, "lookup"),
        # Both criteria on the first random partition alone, the comparison
        # sec:esi-draw-spread prints beside the one over the ten partitions.
        "first_partition_lookup_r2": lambda s: s[f"random_lookup_{first}"],
        "first_partition_margin": lambda s: (s[f"random_model_{first}"]
                                             - s[f"random_lookup_{first}"]),
    })

    def verdicts(lookup_key, margin_key, lookup_bound, margin_bound):
        return (("BELOW THE COMBINATORIAL RANGE"
                 if intervals[lookup_key]["hi"] < lookup_bound
                 else "REACHES THE COMBINATORIAL RANGE"),
                ("ABOVE THE COMBINATORIAL RANGE"
                 if intervals[margin_key]["lo"] > margin_bound
                 else "INSIDE THE COMBINATORIAL RANGE"))

    e1, e2 = verdicts("random_lookup_r2", "model_margin", lookup_lo, margin_hi)
    (first_lookup_lo, first_lookup_bound_by), (first_margin_hi, first_margin_bound_by) = (
        bounds(htem, "first_partition_lookup_r2", "first_partition_margin"))
    e1_first, e2_first = verdicts("first_partition_lookup_r2",
                                  "first_partition_margin",
                                  first_lookup_lo, first_margin_hi)
    print(f"\n  E1 lookup R^2 {record['lookup_r2']:+.4f} "
          f"[{intervals['random_lookup_r2']['lo']:+.4f}, {intervals['random_lookup_r2']['hi']:+.4f}]"
          f"  against smallest combinatorial lookup lo {lookup_lo:+.4f} ({lookup_bound_by})"
          f"  -> {e1}")
    print(f"  E2 model minus lookup {margin:+.4f} "
          f"[{intervals['model_margin']['lo']:+.4f}, {intervals['model_margin']['hi']:+.4f}]"
          f"  against largest combinatorial margin hi {margin_hi:+.4f} ({margin_bound_by})"
          f"  -> {e2}")
    print(f"  on the first partition alone: E1 -> {e1_first}, E2 -> {e2_first} "
          f"(margin {partitions[0]['margin']:+.4f} "
          f"[{intervals['first_partition_margin']['lo']:+.4f}, "
          f"{intervals['first_partition_margin']['hi']:+.4f}] against "
          f"{first_margin_hi:+.4f} on {first_margin_bound_by})\n")

    # E3: the same protocol on the rows whose gap exceeds zero.
    nonzero = y > 0
    X_nz, y_nz = X[nonzero], y[nonzero]
    preds_nz, partitions_nz = {}, []
    for seed in htem_split.FOLD_SEEDS:
        lookup, model, _ = htem_split.out_of_fold(
            X_nz, y_nz, htem_split.random_partition(seed), None)
        preds_nz[f"random_lookup_{seed}"], preds_nz[f"random_model_{seed}"] = lookup, model
        entry = {"seed": seed, "lookup_r2": htem_split.r2(y_nz, lookup),
                 "model_r2": htem_split.r2(y_nz, model)}
        entry["margin"] = entry["model_r2"] - entry["lookup_r2"]
        partitions_nz.append(entry)
    record_nz = {key: float(np.mean([p[key] for p in partitions_nz]))
                 for key in ("lookup_r2", "model_r2")}
    intervals_nz = htem_split.cluster_bootstrap(y_nz, preds_nz, np.arange(len(y_nz)), {
        "random_lookup_r2": lambda s: mean(s, "lookup"),
        "random_model_r2": lambda s: mean(s, "model"),
        "model_margin": lambda s: mean(s, "model") - mean(s, "lookup"),
    })
    e3 = ("BELOW THE COMBINATORIAL RANGE"
          if intervals_nz["random_lookup_r2"]["hi"] < lookup_lo
          else "REACHES THE COMBINATORIAL RANGE")
    print(f"  E3 on the {nonzero.sum():,} rows with a nonzero gap: lookup R^2 "
          f"{record_nz['lookup_r2']:+.4f} "
          f"[{intervals_nz['random_lookup_r2']['lo']:+.4f}, "
          f"{intervals_nz['random_lookup_r2']['hi']:+.4f}]  against "
          f"{lookup_lo:+.4f} ({lookup_bound_by})  -> {e3};  margin "
          f"{record_nz['model_r2'] - record_nz['lookup_r2']:+.4f} "
          f"[{intervals_nz['model_margin']['lo']:+.4f}, "
          f"{intervals_nz['model_margin']['hi']:+.4f}]\n")

    result = {
        "label": "experimental band gap",
        "source": "matbench_expt_gap",
        "rows": int(len(y)), "elements": len(elements), "n_features": int(X.shape[1]),
        "chemical_systems": int(len(set(systems))),
        "median_rows_per_system": float(np.median(rows_per_system)),
        "duplicate_composition_rows": int(duplicates),
        "zero_gap_share": float((y == 0).mean()),
        "target_sd": float(y.std()),
        "protocol": {
            "from": "reproduce/htem_split.py",
            "n_folds": htem_split.N_FOLDS, "fold_seeds": htem_split.FOLD_SEEDS,
            "rf_trees": htem_split.RF_TREES, "rf_seed": htem_split.RF_SEED,
            "bootstrap": {"seed": htem_split.BOOTSTRAP_SEED,
                          "replicates": htem_split.BOOTSTRAP_REPLICATES,
                          "cluster": "composition", "alpha": htem_split.ALPHA},
        },
        "random": record,
        "random_partitions": partitions,
        "model_margin": margin,
        "ci": intervals,
        "criteria": {
            "E1": {"lookup_hi": intervals["random_lookup_r2"]["hi"],
                   "threshold": lookup_lo, "bound_by": lookup_bound_by, "verdict": e1,
                   "on_first_partition": {
                       "lookup_hi": intervals["first_partition_lookup_r2"]["hi"],
                       "threshold": first_lookup_lo,
                       "bound_by": first_lookup_bound_by, "verdict": e1_first}},
            "E2": {"margin_lo": intervals["model_margin"]["lo"],
                   "threshold": margin_hi, "bound_by": margin_bound_by, "verdict": e2,
                   "on_first_partition": {
                       "margin_lo": intervals["first_partition_margin"]["lo"],
                       "threshold": first_margin_hi,
                       "bound_by": first_margin_bound_by, "verdict": e2_first}},
            "E3": {"lookup_hi": intervals_nz["random_lookup_r2"]["hi"],
                   "threshold": lookup_lo, "bound_by": lookup_bound_by, "verdict": e3},
        },
        "nonzero_gap": {
            "rows": int(nonzero.sum()),
            "chemical_systems": int(len(set(systems[nonzero]))),
            "target_sd": float(y_nz.std()),
            "random": record_nz,
            "random_partitions": partitions_nz,
            "model_margin": record_nz["model_r2"] - record_nz["lookup_r2"],
            "ci": intervals_nz,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
