"""Tuned models on the HTEM-DB blocked partition, run after the result they test.

The run answers an alternative that a section of the manuscript leaves open,
and its destination and both of its outcomes were written here before it ran.
It is not a new result reported in its own right.

MODELS -> par:partition, discharging the BASELINE tag in
reproduce/htem_split.py.
    The alternative: par:partition said the comparison holds the model fixed
    and does not survey models, which leaves open the possibility that the
    blocked partition's low score reflects a weak model rather than a hard
    prediction task. The alternative is run rather than argued, because a
    sentence that names the possibility does not rule it out.
    Both outcomes. If the excess stays positive under every arm, par:partition
    reports the arms and its scope sentence cites them instead of conceding the
    point. If any arm lifts the blocked model above the random-partition lookup,
    that arm is the finding: the excess would be a fact about an untuned forest,
    and the claim in par:partition is what changes, in the same commit.

THE SECOND VERSION, written on 2026-11-21 before it ran. Two faults of the
first version were found after its excess under the tuned models was known.
Under the blocked partition both tuned models chose the most regularized corner
of their grids, the forest a feature share of 0.3 in every fold and the
gradient-boosting machine 15 leaves in nine folds of ten, so the search did not
show where their best setting lies. The gradient-boosting machine also chose
its number of rounds by scikit-learn's early stopping, on a random tenth of its
training rows that no library groups, so under the blocked partition that
choice saw positions adjacent to the rows it scored, the leak that the blocked
partition removes. This version widens every axis past the corner that the
blocked partition chose and past the opposite edge wherever the axis allows,
at a budget still matched across the two arms, and chooses the number of
rounds on the same validation slice that chooses the configuration, which the
blocked partition groups by library.
CRITERIA, fixed before this version ran:
  T1  For each property and each tuned model, the interval of the excess under
      the blocked partition lies above zero. Met on every pair: the tuned-model
      section of the Supplementary Information keeps that the excess is
      positive under either tuned model, par:partition's second limit keeps its
      claim, sec:esi-posthoc describes the widened grids and the choice of rounds,
      and the Supplementary Information names the two faults in one sentence
      that says they were found after the first version's excess was known.
      Missed on any pair: that pair's excess is printed as measured, and par:partition's
      claim that the positive excess is established on HTEM-DB is narrowed to
      the models under which it holds. The result goes to the author before the
      manuscript is edited.
  T2  THE EDGE, reported either way. Each fold's winner is checked against the
      first and last value of every axis of its grid, and each chosen number of
      rounds against MAX_ROUNDS. Where a winner under the blocked partition
      still sits on an edge, sec:esi-posthoc says that the search reached the edge
      of its grid on that model.
Under either outcome, tab:ablation prints each tuned model's random-partition
score, which the first version computed and the manuscript never printed.

The arms share one out-of-fold pass per partition, so every number below is
scored on the rows and partitions reproduce/htem_split.py scores, from that
module's feature construction rather than from a second one that agrees with it.
The excess takes the random-partition lookup as that module does, as the mean
over its random partitions. The untuned forest is re-run here and asserted
against the committed split.json on every partition: if the two disagree, this
file is measuring a different dataset and says so rather than publishing an
ablation of something else.

Degrees of freedom:
  ARMS         CHOSEN -- a tuned forest and a tuned gradient-boosting machine,
               the two models that the obvious alternative in the BASELINE tag
               of reproduce/htem_split.py names, plus the manuscript's own
               forest as the reference arm.
  GRIDS        CHOSEN -- the axes each library's own documentation puts first,
               at a budget matched across the two arms (SEARCH_BUDGET below),
               each axis spanning both sides of the first version's winners.
  INNER_FOLDS  CHOSEN -- the validation slice is one fold of four, a quarter
               of the training block.
  TUNED_TREES  CHOSEN -- a budget, fixed on the tuned forest, not a tuned axis.
  MAX_ROUNDS   CHOSEN -- the most rounds the gradient-boosting machine may
               take; T2 reports whether a winner reaches it.
  SELECT_JOBS, THREADS_PER_JOB  CHOSEN -- how the configurations are spread
               over the processors; no score depends on them.
  Winners      FITTED -- to inner-validation R^2 inside each training block,
               never to a test block and never to any number the paper prints.
               The number of rounds is FITTED in the same way.
  Nothing here is FITTED to a manuscript claim.

BASELINE: forest and gradient-boosting machine, each tuned inside the training
block | obvious alternative: a referee names a tuned model and says the blocked
column is a weak model rather than a hard question -- this file is that
alternative | tuning: identical budget on both arms and identical under both
protocols, SEARCH_BUDGET configurations and, for the gradient-boosting machine,
its number of rounds, selected on a validation slice of each training block,
grouped by library wherever the protocol is | discharge: run, reported in
par:partition under all three arms

    python reproduce/htem_ablation.py
"""
import gzip
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GroupKFold, KFold
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "reproduce"))

from htem_split import (CACHE, FOLD_SEEDS,  # noqa: E402
                        N_FOLDS, RF_SEED, RF_TREES, TARGETS, cluster_bootstrap,
                        design_matrix, mean_over_partitions, r2,
                        random_partition)
from metal.lookup import one_nn  # noqa: E402

OUT = ROOT / "outputs" / "htem" / "ablation.json"
COMMITTED = ROOT / "outputs" / "htem" / "split.json"

# The inner split that selects a configuration: one slice of each training
# block, grouped by library wherever the outer protocol is grouped, so tuning
# cannot see across a library the protocol holds out.
INNER_FOLDS = 4
TUNED_TREES = 400          # a budget, not a tuned axis: fixed on the tuned forest
MAX_ROUNDS = 1000
SELECT_JOBS = 25
THREADS_PER_JOB = 3

# One grid per arm, at a matched size. The axes are the ones each model's
# own documentation puts first; what matters for the claim is that neither arm
# is given a search the other is not, because two models tuned to different
# depths differ in search before they differ in anything else. A feature share
# of 1.0 and one row per leaf are the ends that their axes allow.
AXES = {
    "tuned_forest": {"max_features": (1.0, 0.5, 0.3, 0.15, 0.05),
                     "min_samples_leaf": (1, 3, 10, 30, 100)},
    "boosted": {"learning_rate": (0.01, 0.03, 0.06, 0.12, 0.25),
                "max_leaf_nodes": (3, 7, 15, 63, 255)},
}
BOOST_L2 = 1.0
GRIDS = {
    "tuned_forest": [
        {"max_features": mf, "min_samples_leaf": leaf}
        for mf, leaf in product(*AXES["tuned_forest"].values())
    ],
    "boosted": [
        {"learning_rate": lr, "max_leaf_nodes": leaves, "l2_regularization": BOOST_L2}
        for lr, leaves in product(*AXES["boosted"].values())
    ],
}
# Matched by construction rather than by truncation: two axes of five on each
# arm. Trimming a longer grid to a shorter one would cut it in whatever order it
# was built, which is a choice about the search and not a budget.
SEARCH_BUDGET = len(GRIDS["tuned_forest"])
assert all(len(g) == SEARCH_BUDGET for g in GRIDS.values()), (
    "the two arms are given different search budgets: "
    + ", ".join(f"{a} {len(g)}" for a, g in GRIDS.items()))

def build(arm, config, rounds=MAX_ROUNDS, n_jobs=-1):
    """The arm's model. The gradient-boosting machine takes no early stopping
    of its own: its number of rounds comes from select(), on the slice that
    chooses its configuration."""
    if arm == "default_forest":
        return RandomForestRegressor(n_estimators=RF_TREES, random_state=RF_SEED,
                                     n_jobs=n_jobs)
    if arm == "tuned_forest":
        return RandomForestRegressor(n_estimators=TUNED_TREES, random_state=RF_SEED,
                                     n_jobs=n_jobs, **config)
    return HistGradientBoostingRegressor(max_iter=rounds, early_stopping=False,
                                         random_state=RF_SEED, **config)


def score(arm, config, X_fit, y_fit, X_val, y_val):
    """One configuration's R^2 on the validation slice and, for the
    gradient-boosting machine, the number of rounds at which that R^2 peaks."""
    with threadpool_limits(THREADS_PER_JOB):
        model = build(arm, config, n_jobs=THREADS_PER_JOB).fit(X_fit, y_fit)
        if arm == "boosted":
            curve = [r2(y_val, p) for p in model.staged_predict(X_val)]
            best = int(np.argmax(curve))
            return curve[best], best + 1
        return r2(y_val, model.predict(X_val)), None


def on_edge(arm, config, rounds):
    """The axes on whose first or last value a winner sits, and "rounds" where
    the number of rounds reaches MAX_ROUNDS (T2)."""
    edges = [axis for axis, values in AXES[arm].items()
             if config[axis] in (values[0], values[-1])]
    return edges + (["rounds"] if rounds == MAX_ROUNDS else [])


def inner_split(X, y, groups, seed):
    """One validation slice of a training block, grouped when the outer protocol
    is, and otherwise shuffled by the seed of the random partition it tunes
    inside. Returns the first split, so selection is deterministic."""
    if groups is None:
        splitter = KFold(INNER_FOLDS, shuffle=True, random_state=seed)
    else:
        splitter = GroupKFold(INNER_FOLDS)
    return next(iter(splitter.split(X, y, groups)))


def select(arm, X, y, groups, seed):
    """The configuration that scores best on a validation slice of this training
    block, with its number of rounds for the gradient-boosting machine. Never
    sees the fold's test rows. Ties go to the configuration listed later, as
    max() over (score, index) gives them."""
    fit_idx, val_idx = inner_split(X, y, groups, seed)
    results = Parallel(n_jobs=SELECT_JOBS)(
        delayed(score)(arm, c, X[fit_idx], y[fit_idx], X[val_idx], y[val_idx])
        for c in GRIDS[arm])
    won = max(range(len(results)), key=lambda i: (results[i][0], i))
    return won, results[won][1]


def out_of_fold(X, y, splitter, groups, arms, seed=None):
    """Every row predicted once by every arm, by the model trained without its fold.
    One pass, so the arms are scored on identical folds. seed is the random
    partition's, which the inner split of a tuned arm reuses."""
    lookup = np.empty(len(y))
    model = {arm: np.empty(len(y)) for arm in arms}
    winners = {arm: [] for arm in arms if arm in GRIDS}
    for train, test in splitter.split(X, y, groups):
        lookup[test], _ = one_nn(X[train], y[train], X[test])
        inner_groups = None if groups is None else groups[train]
        for arm in arms:
            rounds = MAX_ROUNDS
            if arm in GRIDS:
                won, chosen = select(arm, X[train], y[train], inner_groups, seed)
                config = GRIDS[arm][won]
                rounds = chosen or MAX_ROUNDS
                winners[arm].append({**config,
                                     **({"rounds": chosen} if chosen else {}),
                                     "on_edge": on_edge(arm, config, chosen)})
            else:
                config = {}
            model[arm][test] = (build(arm, config, rounds)
                                .fit(X[train], y[train]).predict(X[test]))
    return lookup, model, winners


def analyse(all_rows, key, label):
    X, y, libraries, _ = design_matrix(all_rows, key)
    arms = ("default_forest", "tuned_forest", "boosted")
    # Each pass as (name, splitter, groups, seed): the random partitions under
    # their seeds, then the one blocked partition. Predictions are keyed
    # <protocol>_<arm>[_<seed>], as mean_over_partitions reads their scores.
    passes = [(f"random_{seed}", random_partition(seed), None, seed)
              for seed in FOLD_SEEDS]
    passes.append(("blocked", GroupKFold(N_FOLDS), libraries, None))

    print(f"== {label}  ({len(y):,} rows, {len(set(libraries))} libraries)", flush=True)
    preds, winners = {}, {}
    for name, splitter, groups, seed in passes:
        lookup, model, won = out_of_fold(X, y, splitter, groups, arms, seed)
        key_of = (lambda arm: f"random_{arm}_{seed}") if seed is not None else (
            lambda arm: f"blocked_{arm}")
        preds[key_of("lookup")] = lookup
        for arm in arms:
            preds[key_of(arm)] = model[arm]
        winners["blocked" if seed is None else seed] = won
        print(f"  {name.upper():12s} lookup {r2(y, lookup):+.4f}   "
              + "   ".join(f"{a} {r2(y, model[a]):+.4f}" for a in arms),
              flush=True)
    scores = {name: r2(y, p) for name, p in preds.items()}

    # The reference arm is the manuscript's own. If it has moved, this file is
    # not an ablation of the published result.
    committed = json.loads(COMMITTED.read_text())["targets"][key]
    published = [(f"random_{{arm}}_{p['seed']}", p)
                 for p in committed["random_partitions"]]
    published.append(("blocked_{arm}", committed["blocked"]))
    for pattern, record in published:
        for arm_key, field in (("default_forest", "model_r2"), ("lookup", "lookup_r2")):
            name = pattern.format(arm=arm_key)
            got, want = scores[name], record[field]
            assert abs(got - want) < 1e-9, (
                f"{label} {name} is {got:.6f} here against {want:.6f} in "
                f"outputs/htem/split.json. The ablation and the published result "
                f"are not scoring the same dataset.")
    print("  reference arm reproduces outputs/htem/split.json exactly.")

    random_lookup = mean_over_partitions(scores, "lookup")
    statistics = {}
    for arm in arms:
        statistics[f"excess_{arm}"] = (lambda s, a=arm: mean_over_partitions(s, "lookup")
                                       - s[f"blocked_{a}"])
        statistics[f"random_{arm}"] = lambda s, a=arm: mean_over_partitions(s, a)
        statistics[f"blocked_{arm}"] = lambda s, a=arm: s[f"blocked_{a}"]
    intervals = cluster_bootstrap(y, preds, libraries, statistics)

    per_arm = {}
    for arm in arms:
        excess = random_lookup - scores[f"blocked_{arm}"]
        blocked_winners = winners["blocked"].get(arm)
        per_arm[arm] = {
            "random_model_r2": mean_over_partitions(scores, arm),
            "blocked_model_r2": scores[f"blocked_{arm}"],
            "excess": excess, "ci": intervals[f"excess_{arm}"],
            "ci_random_model_r2": intervals[f"random_{arm}"],
            "ci_blocked_model_r2": intervals[f"blocked_{arm}"],
            "configurations_tried": len(GRIDS.get(arm, [{}])),
            "selected_per_fold": {
                "random": ({str(seed): winners[seed][arm] for seed in FOLD_SEEDS}
                           if arm in GRIDS else None),
                "blocked": blocked_winners},
        }
        if arm in GRIDS:
            per_arm[arm]["T1"] = {"met": intervals[f"excess_{arm}"]["lo"] > 0}
            per_arm[arm]["T2"] = {"blocked_folds_on_edge": sum(
                1 for w in blocked_winners if w["on_edge"])}
        print(f"  excess, {arm:14s} {excess:+.4f} "
              f"[{intervals[f'excess_{arm}']['lo']:+.4f}, "
              f"{intervals[f'excess_{arm}']['hi']:+.4f}]"
              + (f"   T1 {'MET' if per_arm[arm]['T1']['met'] else 'MISSED'}, "
                 f"blocked winners on an edge: "
                 f"{[w['on_edge'] for w in blocked_winners]}" if arm in GRIDS else ""))

    return {"label": label, "rows": int(len(y)),
            "lookup": {"random": random_lookup, "blocked": scores["blocked_lookup"]},
            "arms": per_arm}


def main() -> None:
    if not COMMITTED.exists():
        raise FileNotFoundError(
            f"{COMMITTED} is missing. Run reproduce/htem_split.py first: this "
            f"file is an ablation of that result and checks itself against it.")
    with gzip.open(CACHE, "rt") as fh:
        rows = [json.loads(line) for line in fh]

    result = {
        "n_folds": N_FOLDS, "fold_seeds": FOLD_SEEDS, "inner_folds": INNER_FOLDS,
        "search_budget": SEARCH_BUDGET, "tuned_trees": TUNED_TREES,
        "max_rounds": MAX_ROUNDS, "grids": GRIDS,
        "targets": {key: analyse(rows, key, label) for key, label, _, _ in TARGETS},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
