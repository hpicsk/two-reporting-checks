"""
HTEM-DB (NREL): the same lookup, on a dataset whose release records each row's library.

Kim & Lee's dataset does not ship an alloy-system label, so their random split can
be shown to be dense but not re-run blocked. HTEM-DB ships `sample_library_id` on
every sample, so both partitions run on one dataset with one model and only
the grouping changed. That is what turns two instances into a claim about a type
of dataset.

Dataset: every HTEM-DB library carrying both four-point-probe electrical data and XRF
composition. Two targets, each on the rows that carry it: log10 resistivity
(logged because it spans orders of magnitude) and direct band gap (from optical
transmission, untransformed). Features: elemental atomic fractions from XRF, plus
thickness and deposition temperature (NO_TEMPERATURE on a row that records none),
the composition-plus-processing feature set of the other combinatorial datasets.

Both targets run the whole protocol. They are two properties of one dataset, not
two datasets, and the nesting is measured rather than asserted: `nesting()` below
records that every band-gap row is also an electrical row, on a subset of the
same libraries. Two pairs here are two properties of one set of libraries, and
the manuscript's blocked-pair count spans fewer datasets than pairs.

PARTITIONS. The random partition is drawn once from each of FOLD_SEEDS, and a
random-partition score is the mean over those partitions. The blocked partition
is GroupKFold's one assignment of libraries to folds, which no seed moves.

PREDICTORS, both scored on pooled out-of-fold predictions so every row is
predicted exactly once under each partition, with sklearn's R^2 convention (the
scored set's own mean as reference, so the constant predictor is 0):
  - 1-NN lookup (src/metal/lookup.py): copy the nearest training row's target.
  - Random forest: the model a practitioner reaches for on tabular data this size.

INTERVALS are a cluster bootstrap over libraries, not over folds. Folds share
training rows, so fold-to-fold spread says whether a number is stable across the
partition and not how precisely it is estimated; libraries are the dataset's
sampling unit, so an interval resampling them rests on the dataset's own
libraries, over a hundred per property, rather than on a few replicates,
because a percentile bootstrap over a few replicates covers well below its
stated level.
The same rows are scored under both protocols, so every contrast below is paired
row by row and the bootstrap resamples libraries once per replicate for all of
them. Each resample recomputes every random-partition statistic as the mean
over the random partitions. The spread across those partitions is reported as a
range and not as an interval: the partitions share rows, so the range says how
far one partition's number can sit from the mean.

CRITERIA, fixed before any number below is read:
  H1  RANDOM ROW SPLIT reproduces the npj pattern -> lookup R^2 within 0.05 of
      the trained model's. If the model clears the lookup by more than 0.05 on a
      random split, this dataset does NOT show the pattern and is reported as a
      negative instance.
  H2  LIBRARY-BLOCKED SPLIT costs more than the model's own margin -> the drop in
      model R^2 from random to blocked exceeds (model - lookup) on the random
      split. This is the claim: the partition protocol moves the number further
      than the choice of model does.
  H3  DIRECTION. Blocked model R^2 is reported as it falls, including if it stays
      high, which would refute the account on this dataset.
Reported either way, and per property: a criterion met on one target and missed
on the other is reported as split, not as a dataset-level verdict.

TEN RANDOM PARTITIONS. Every random-partition number is the mean over the
partitions that FOLD_SEEDS draw, because a margin measured on one random
partition can lie far from the mean over many (sec:esi-draw-spread). Each
property's margins are recorded per partition, with their range and their
mean. The second control (reproduce/expt_gap_control.py) and the tuned models
(reproduce/htem_ablation.py) take the seeds from this module. The random
partition was first drawn once, from seed 42; the move to ten was fixed on
2026-09-15, after that partition's numbers were known and before any
ten-partition number was read, and sec:esi-draw-spread prints the second
control's comparison on that first partition beside the one over the ten.

THE LOOKUP'S FALL, criteria fixed on 2026-11-21 before any number they read.
par:partition states that the lookup's R^2 on HTEM-DB falls by more than half
and further than the model's, and sec:features states the second comparison
under every alternative temperature fill. Both had been read off point
estimates. Each is now decided per property on the same cluster bootstrap over
libraries that gives every other interval here:
  C1  MORE THAN HALF -> the interval of half the lookup's random-partition R^2
      minus its blocked-partition R^2 lies above zero.
  C2  FURTHER THAN THE MODEL'S -> the interval of the lookup's fall minus the
      model's fall, each from the random-partition mean to the blocked
      partition, lies above zero.
  C3  UNDER EVERY FILL -> C2 holds with each of NO_TEMPERATURE_ALTERNATIVES in
      place of NO_TEMPERATURE.
What each outcome changes, written before the run. A criterion met keeps its
clause for that property, and the interval that decided it is printed beside
the verdict in the sentence that states it. A criterion missed removes its
clause for that property from every sentence that states it, and the property's
values are printed as measured; C1 missed on a property also removes, for that
property, the conclusion that most of the lookup's random-partition R^2 depends
on rows of the test row's own library. The identifiability question names no
confound for C1 and C2 beyond the first limit that par:partition already states:
for the lookup, which fits nothing, the blocked partition changes only which
training rows lie nearest.

Degrees of freedom:
  TARGETS      MEASURED -- the two properties HTEM-DB ships on enough rows to fold
               five ways; the transform on each is fixed by its own range.
  FEATURES     CHOSEN -- composition plus the two processing columns, thickness
               (on every eligible row) and deposition temperature (NO_TEMPERATURE
               where a row records none), matching the composition-plus-processing
               features of the other combinatorial datasets.
  N_FOLDS      CHOSEN -- five, the same count under both protocols so the
               train/test row shares are comparable and only the grouping differs.
  FOLD_SEEDS   CHOSEN -- EMPD's ten draw seeds (reproduce/empd_draws.py), so the
               random partitions of every dataset come from one seed set.
  RF_TREES     CHOSEN -- 200 trees, twice scikit-learn 1.6.1's default of 100,
               to lower the variance that the trees' random draws add to the
               forest's predictions while keeping the forests of the ten random
               partitions and of the blocked partition, five folds each, quick
               to fit. The count was neither derived nor
               tuned, and it is identical under both protocols. The excess stays
               positive on both properties under reproduce/htem_ablation.py's
               tuned forest of 400 trees and its gradient-boosting machine
               (tab:ablation), so the excess does not rest on this count. Every
               other forest setting is at its library default. See the BASELINE
               tag below.
  RF_SEED      CHOSEN -- one forest seed for every partition: the random
               partitions vary the split seed (FOLD_SEEDS), and the forest seed
               is held fixed so that the two protocols differ only in grouping.
  NO_TEMPERATURE  CHOSEN -- see the comment at the constant.
  NO_TEMPERATURE_ALTERNATIVES  CHOSEN -- three other fills below every recorded
               temperature, re-scored in place of NO_TEMPERATURE; sec:features
               prints the largest change they cause.
  BOOTSTRAP_SEED, BOOTSTRAP_REPLICATES, ALPHA  CHOSEN -- one generator seed, so
               that a re-run reproduces every interval; alpha 0.05, the 95%
               percentile interval that sec:statistics names; 2000 resamples,
               so that the interval's endpoints, which are its 2.5% and 97.5%
               percentiles, change little with the generator seed. Where the
               bootstrap distribution is normal with standard deviation s, the
               Monte Carlo standard error of either endpoint at 2000 resamples
               is sqrt(0.025 * 0.975 / 2000) / phi(1.96) * s, about 0.06 s, or
               3% of the interval's half-width of 1.96 s. The rounding step that
               sec:statistics sets is more than 5% of the half-width, so a
               printed endpoint moves by at most about one step with the seed.
  Nothing is FITTED: no criterion here is matched to a target.

BASELINE: untuned random forest of 200 trees, every other setting at its
library default | obvious alternative: a referee names gradient boosting or a
tuned forest, and says the blocked column is a weak model rather than a hard
question | tuning: none, under either protocol -- the forest's settings are the
same under both, so the untuned penalty is paid symmetrically | discharge: run
-- reproduce/htem_ablation.py re-runs both protocols with a forest and a
gradient-boosting machine tuned inside each training block, and par:partition
reports what the excess does under all three

    python reproduce/htem_split.py
"""
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, KFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from metal.lookup import one_nn  # noqa: E402

# The fetch is cached in the repository so this runs offline; reproduce/fetch_htem.py
# rebuilds it from the public API.
CACHE = ROOT / "data" / "htem" / "htem_rows.ndjson.gz"
OUT = ROOT / "outputs" / "htem" / "split.json"

N_FOLDS = 5
FOLD_SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144]
RF_TREES = 200
RF_SEED = 42
# The deposition temperature of a row that records none (sec:features).
# CHOSEN: below every recorded temperature. Any fill below them all leaves the
# order of the training rows by temperature unchanged, but scikit-learn places a
# split's threshold midway between the two values that the split separates, so
# the fill moves the threshold of a split between a filled row and a recorded
# temperature, a test row can fall on the other side of it, and the forest's
# predictions can change. The lookup's rescaled distances change with it too.
# The value is justified by measurement: analyse() re-scores both partitions
# with each of NO_TEMPERATURE_ALTERNATIVES in its place, and sec:features
# prints the largest change in the margins, the excesses and the
# blocked-partition lookup score.
NO_TEMPERATURE = -1.0
NO_TEMPERATURE_ALTERNATIVES = (0.0, -100.0, -1000.0)
BOOTSTRAP_SEED = 42
BOOTSTRAP_REPLICATES = 2000
ALPHA = 0.05
H1_TOLERANCE = 0.05          # "within 0.05 R^2", fixed before the run

# Each target, with the rows that carry it and the transform its range needs.
# Filters are predicates rather than a column list because band gap is present on
# a subset of the electrical rows, so the feature space is rebuilt per target.
TARGETS = (
    ("log10_resistivity", "log10 resistivity",
     lambda r: r.get("fpm_resistivity") and r["fpm_resistivity"] > 0,
     lambda r: np.log10(r["fpm_resistivity"])),
    ("direct_band_gap", "direct band gap",
     lambda r: r.get("opt_direct_bandgap") is not None,
     lambda r: r["opt_direct_bandgap"]),
)


def r2(y, pred, idx=None):
    if idx is not None:
        y, pred = y[idx], pred[idx]
    return 1.0 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def random_partition(seed):
    """The random partition one of FOLD_SEEDS draws: folds over rows."""
    return KFold(N_FOLDS, shuffle=True, random_state=seed)


def mean_over_partitions(scores, predictor):
    """A predictor's mean R^2 over the random partitions, from scores keyed
    random_<predictor>_<seed>."""
    return float(np.mean([scores[f"random_{predictor}_{seed}"]
                          for seed in FOLD_SEEDS]))


def lookup_fall_beyond_half(scores):
    """Half the lookup's random-partition R^2 minus its blocked-partition R^2,
    positive where the lookup keeps less than half under the blocked partition."""
    return 0.5 * mean_over_partitions(scores, "lookup") - scores["blocked_lookup"]


def lookup_fall_minus_model_fall(scores):
    """The lookup's fall from the random-partition mean to the blocked partition,
    minus the model's."""
    return ((mean_over_partitions(scores, "lookup") - scores["blocked_lookup"])
            - (mean_over_partitions(scores, "model") - scores["blocked_model"]))


def out_of_fold(X, y, splitter, groups):
    """Predict every row exactly once, by the model trained without its fold."""
    lookup = np.empty(len(y))
    model = np.empty(len(y))
    distance = np.empty(len(y))
    for train, test in splitter.split(X, y, groups):
        lookup[test], distance[test] = one_nn(X[train], y[train], X[test])
        # n_jobs=-1 leaves the order of the parallel reduction unfixed, so a
        # re-run can differ in the last bits of a float. The observed drift is at
        # the sixteenth decimal, against numbers printed to three or four, so no
        # digit the manuscript prints depends on it.
        forest = RandomForestRegressor(n_estimators=RF_TREES, random_state=RF_SEED,
                                       n_jobs=-1).fit(X[train], y[train])
        model[test] = forest.predict(X[test])
    return lookup, model, distance


def cluster_bootstrap(y, preds, groups, statistics):
    """Percentile intervals for statistics of pooled predictions, resampling
    whole libraries. One resample serves every statistic, so contrasts between
    them stay paired."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    members = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    replicates = {name: np.empty(BOOTSTRAP_REPLICATES) for name in statistics}
    for b in range(BOOTSTRAP_REPLICATES):
        picked = rng.integers(0, len(members), len(members))
        idx = np.concatenate([members[i] for i in picked])
        scores = {key: r2(y, p, idx) for key, p in preds.items()}
        for name, fn in statistics.items():
            replicates[name][b] = fn(scores)
    return {name: {"lo": float(np.percentile(v, 100 * ALPHA / 2)),
                   "hi": float(np.percentile(v, 100 * (1 - ALPHA / 2)))}
            for name, v in replicates.items()}


def eligible(all_rows, key):
    """The rows one target is measured on: it carries the target, and it carries
    the features that design_matrix() builds."""
    _, _, keep, _ = next(t for t in TARGETS if t[0] == key)
    return [r for r in all_rows if keep(r) and r.get("xrf_elements")
            and r.get("xrf_concentration") and r.get("thickness") is not None]


def design_matrix(all_rows, key):
    """Features, target and grouping for one target. reproduce/htem_ablation.py
    imports this, so its arms and these run on one feature construction rather
    than on two that agree."""
    _, _, _, target = next(t for t in TARGETS if t[0] == key)
    rows = eligible(all_rows, key)
    elements = sorted({e for r in rows for e in r["xrf_elements"]})

    def features(r):
        fractions = dict(zip(r["xrf_elements"], r["xrf_concentration"]))
        total = sum(fractions.values()) or 1.0
        temperature = r.get("absolute_temp_c")
        return ([fractions.get(e, 0.0) / total for e in elements]
                + [r["thickness"],
                   NO_TEMPERATURE if temperature is None else temperature])

    X = np.array([features(r) for r in rows], dtype=float)
    y = np.array([target(r) for r in rows], dtype=float)
    libraries = np.array([r["sample_library_id"] for r in rows])
    return X, y, libraries, elements


def nesting(all_rows):
    """How far the two targets' rows and libraries overlap, counted from the
    rows rather than restated."""
    sets = {key: eligible(all_rows, key) for key, _, _, _ in TARGETS}
    (wide_key, wide), (narrow_key, narrow) = sorted(
        sets.items(), key=lambda kv: -len(kv[1]))
    row_id = lambda rows: {r["id"] for r in rows}
    lib_id = lambda rows: {r["sample_library_id"] for r in rows}
    return {
        "wider": wide_key, "narrower": narrow_key,
        "wider_rows": len(wide), "narrower_rows": len(narrow),
        "narrower_rows_inside_wider": len(row_id(narrow) & row_id(wide)),
        "wider_libraries": len(lib_id(wide)),
        "narrower_libraries": len(lib_id(narrow)),
        "narrower_libraries_inside_wider": len(lib_id(narrow) & lib_id(wide)),
    }


def analyse(all_rows, key, label):
    rows = eligible(all_rows, key)
    X, y, libraries, elements = design_matrix(all_rows, key)

    print(f"== {label}")
    print(f"rows {len(rows):,}  libraries {len(set(libraries))}  "
          f"elements {len(elements)}  features {X.shape[1]}")
    print(f"target: mean {y.mean():.2f}  sd {y.std():.2f}  "
          f"range {y.min():.2f}..{y.max():.2f}   "
          f"median rows/library {np.median(list(Counter(libraries).values())):.0f}\n")

    def predict(X_):
        """Out-of-fold predictions under every random partition and under the
        blocked one, keyed as mean_over_partitions reads their scores."""
        preds, distances = {}, {}
        for seed in FOLD_SEEDS:
            lookup, model, distances[seed] = out_of_fold(
                X_, y, random_partition(seed), None)
            preds[f"random_lookup_{seed}"] = lookup
            preds[f"random_model_{seed}"] = model
        lookup, model, distances["blocked"] = out_of_fold(
            X_, y, GroupKFold(N_FOLDS), libraries)
        preds["blocked_lookup"], preds["blocked_model"] = lookup, model
        return preds, distances

    preds, distances = predict(X)
    scores = {name: r2(y, p) for name, p in preds.items()}
    partitions = []
    for seed in FOLD_SEEDS:
        entry = {"seed": seed, "lookup_r2": scores[f"random_lookup_{seed}"],
                 "model_r2": scores[f"random_model_{seed}"],
                 "median_nn_distance": float(np.median(distances[seed]))}
        entry["margin"] = entry["model_r2"] - entry["lookup_r2"]
        partitions.append(entry)
        print(f"  RANDOM {seed:<5d} lookup R^2 {entry['lookup_r2']:+.4f}   "
              f"random-forest R^2 {entry['model_r2']:+.4f}   "
              f"margin {entry['margin']:+.4f}")
    record = {
        "random": {"lookup_r2": mean_over_partitions(scores, "lookup"),
                   "model_r2": mean_over_partitions(scores, "model"),
                   "median_nn_distance": float(np.mean(
                       [p["median_nn_distance"] for p in partitions]))},
        "blocked": {"lookup_r2": scores["blocked_lookup"],
                    "model_r2": scores["blocked_model"],
                    "median_nn_distance": float(np.median(distances["blocked"]))},
    }
    for name in ("random", "blocked"):
        print(f"  {name.upper():8s} lookup R^2 {record[name]['lookup_r2']:+.4f}   "
              f"random-forest R^2 {record[name]['model_r2']:+.4f}   "
              f"median NN distance {record[name]['median_nn_distance']:.4f}")

    margin = record["random"]["model_r2"] - record["random"]["lookup_r2"]
    drop = record["random"]["model_r2"] - record["blocked"]["model_r2"]

    # The same partitions with each alternative in place of NO_TEMPERATURE, on
    # the rows that record no temperature and on no other row.
    missing = np.array([r.get("absolute_temp_c") is None for r in rows])
    fills = {}
    for fill in NO_TEMPERATURE_ALTERNATIVES:
        X_fill = X.copy()
        X_fill[missing, -1] = fill
        fill_preds, _ = predict(X_fill)
        s = {name: r2(y, p) for name, p in fill_preds.items()}
        fills[str(fill)] = {"random_lookup_r2": mean_over_partitions(s, "lookup"),
                            "random_model_r2": mean_over_partitions(s, "model"),
                            "blocked_lookup_r2": s["blocked_lookup"],
                            "blocked_model_r2": s["blocked_model"],
                            "lookup_fall_minus_model_fall":
                                lookup_fall_minus_model_fall(s),
                            "ci": cluster_bootstrap(y, fill_preds, libraries, {
                                "lookup_fall_minus_model_fall":
                                    lookup_fall_minus_model_fall})}
    shifts = [(abs((s["random_model_r2"] - s["random_lookup_r2"]) - margin),
               abs((s["random_lookup_r2"] - s["blocked_model_r2"]) - (drop - margin)),
               abs(s["blocked_lookup_r2"] - record["blocked"]["lookup_r2"]))
              for s in fills.values()]
    sensitivity = {
        "rows_without_temperature": int(missing.sum()),
        "lowest_recorded_temperature": float(X[~missing, -1].min()),
        "alternatives": fills,
        "max_margin_or_excess_shift": max(max(m, e) for m, e, _ in shifts),
        "max_blocked_lookup_shift": max(b for _, _, b in shifts),
    }
    print(f"  {sensitivity['rows_without_temperature']} rows record no temperature; "
          f"the alternatives move margin or excess by at most "
          f"{sensitivity['max_margin_or_excess_shift']:.4f} and the blocked lookup "
          f"by at most {sensitivity['max_blocked_lookup_shift']:.4f}")

    intervals = cluster_bootstrap(y, preds, libraries, {
        "random_lookup_r2": lambda s: mean_over_partitions(s, "lookup"),
        "random_model_r2": lambda s: mean_over_partitions(s, "model"),
        "blocked_lookup_r2": lambda s: s["blocked_lookup"],
        "blocked_model_r2": lambda s: s["blocked_model"],
        "model_margin": lambda s: (mean_over_partitions(s, "model")
                                   - mean_over_partitions(s, "lookup")),
        "split_drop": lambda s: mean_over_partitions(s, "model") - s["blocked_model"],
        # (random_model - blocked_model) - (random_model - random_lookup)
        # collapses to random_lookup - blocked_model: the claim is that a
        # lookup that fits nothing to the target, under a random split,
        # outscores a trained forest under a blocked one.
        "drop_minus_margin": lambda s: (mean_over_partitions(s, "lookup")
                                        - s["blocked_model"]),
        # sec:esi-draw-spread sets the second control against this dataset on
        # the first random partition as well as on the mean over the ten, so
        # that partition's own interval is estimated here.
        "first_partition_lookup_r2": lambda s: s[f"random_lookup_{FOLD_SEEDS[0]}"],
        "first_partition_margin": lambda s: (s[f"random_model_{FOLD_SEEDS[0]}"]
                                             - s[f"random_lookup_{FOLD_SEEDS[0]}"]),
        # C1 and C2 of the docstring.
        "lookup_fall_beyond_half": lookup_fall_beyond_half,
        "lookup_fall_minus_model_fall": lookup_fall_minus_model_fall,
    })
    c1 = intervals["lookup_fall_beyond_half"]["lo"] > 0
    c2 = intervals["lookup_fall_minus_model_fall"]["lo"] > 0
    c3 = all(f["ci"]["lookup_fall_minus_model_fall"]["lo"] > 0 for f in fills.values())
    print(f"  C1 half the random lookup minus the blocked lookup : "
          f"{lookup_fall_beyond_half(scores):+.4f} "
          f"[{intervals['lookup_fall_beyond_half']['lo']:+.4f}, "
          f"{intervals['lookup_fall_beyond_half']['hi']:+.4f}]  -> "
          f"{'MORE THAN HALF' if c1 else 'NOT ESTABLISHED'}")
    print(f"  C2 lookup fall minus model fall                    : "
          f"{lookup_fall_minus_model_fall(scores):+.4f} "
          f"[{intervals['lookup_fall_minus_model_fall']['lo']:+.4f}, "
          f"{intervals['lookup_fall_minus_model_fall']['hi']:+.4f}]  -> "
          f"{'FURTHER THAN THE MODEL' if c2 else 'NOT ESTABLISHED'}")
    for fill, f in fills.items():
        print(f"  C3 with {fill:>7s} in place of {NO_TEMPERATURE}: "
              f"{f['lookup_fall_minus_model_fall']:+.4f} "
              f"[{f['ci']['lookup_fall_minus_model_fall']['lo']:+.4f}, "
              f"{f['ci']['lookup_fall_minus_model_fall']['hi']:+.4f}]")

    h1 = "PATTERN SHOWN" if margin <= H1_TOLERANCE else "NEGATIVE INSTANCE"
    h2 = "SPLIT MOVES IT MORE" if drop > margin else "MODEL CHOICE MOVES IT MORE"
    h3 = ("below the constant predictor" if record["blocked"]["model_r2"] < 0
          else "above the constant predictor")
    print(f"\n  H1 model minus lookup, random split : {margin:+.4f} "
          f"[{intervals['model_margin']['lo']:+.4f}, {intervals['model_margin']['hi']:+.4f}]"
          f"  -> {h1} (tolerance {H1_TOLERANCE})")
    print(f"  H2 model R^2 drop, random -> blocked : {drop:+.4f} "
          f"[{intervals['split_drop']['lo']:+.4f}, {intervals['split_drop']['hi']:+.4f}]"
          f"  -> {h2}")
    print(f"     drop minus margin                 : {drop - margin:+.4f} "
          f"[{intervals['drop_minus_margin']['lo']:+.4f}, "
          f"{intervals['drop_minus_margin']['hi']:+.4f}]")
    print(f"  H3 blocked model R^2                 : "
          f"{record['blocked']['model_r2']:+.4f} "
          f"[{intervals['blocked_model_r2']['lo']:+.4f}, "
          f"{intervals['blocked_model_r2']['hi']:+.4f}]  ({h3})\n")

    return {
        "label": label, "rows": len(rows), "libraries": int(len(set(libraries))),
        "elements": len(elements), "n_features": int(X.shape[1]),
        "median_rows_per_library": float(np.median(list(Counter(libraries).values()))),
        "target_sd": float(y.std()),
        "random": record["random"], "blocked": record["blocked"],
        "random_partitions": partitions,
        "margin_min": min(p["margin"] for p in partitions),
        "margin_max": max(p["margin"] for p in partitions),
        "model_margin": margin, "split_drop": drop,
        "drop_minus_margin": drop - margin,
        "no_temperature_sensitivity": sensitivity,
        "ci": intervals,
        "criteria": {"H1": {"margin": margin, "tolerance": H1_TOLERANCE,
                            "verdict": h1},
                     "H2": {"drop": drop, "margin": margin, "verdict": h2},
                     "H3": {"blocked_model_r2": record["blocked"]["model_r2"],
                            "verdict": h3},
                     "C1": {"value": lookup_fall_beyond_half(scores),
                            "ci": intervals["lookup_fall_beyond_half"], "met": c1},
                     "C2": {"value": lookup_fall_minus_model_fall(scores),
                            "ci": intervals["lookup_fall_minus_model_fall"], "met": c2},
                     "C3": {"met": c3}},
    }


def main() -> None:
    if not CACHE.exists():
        raise FileNotFoundError(
            f"{CACHE} is missing. Run reproduce/fetch_htem.py to rebuild it from "
            f"https://htem-api.nlr.gov/api")
    with gzip.open(CACHE, "rt") as fh:
        rows = [json.loads(line) for line in fh]

    nested = nesting(rows)
    print(f"nesting: {nested['narrower_rows_inside_wider']:,} of "
          f"{nested['narrower_rows']:,} {nested['narrower']} rows are also "
          f"{nested['wider']} rows, on "
          f"{nested['narrower_libraries_inside_wider']} of "
          f"{nested['narrower_libraries']} shared libraries\n")

    result = {
        "n_folds": N_FOLDS, "fold_seeds": FOLD_SEEDS, "rf_trees": RF_TREES,
        "no_temperature": NO_TEMPERATURE,
        "no_temperature_alternatives": list(NO_TEMPERATURE_ALTERNATIVES),
        "nesting": nested,
        "bootstrap": {"seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES,
                      "cluster": "sample_library_id", "alpha": ALPHA},
        "targets": {key: analyse(rows, key, label) for key, label, _, _ in TARGETS},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
