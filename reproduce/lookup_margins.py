"""What a trained network buys over the lookup, on EMPD, the draw-replicated dataset.

Every property is scored the same way: one lookup that fits nothing to the
target and one network per partition, paired partition by partition over the
same ten seeds, the network averaged over its three initializations before the
pairing so each partition contributes one number. The per-partition scores are
reproduce/empd_draws.py's.

Each interval is a percentile bootstrap over the ten partitions, and tab:main
prints the endpoints of those for the margins, the blocked-partition scores and
the excesses. The ten partitions are few replicates, and a percentile bootstrap
over so few covers well below its stated level, so the sentences state neither
an interval's endpoints nor its sign: they count the partitions on each side of
zero instead.

Degrees of freedom:
  ARM          CHOSEN -- the network of src/metal/network.py, the one EMPD is
               scored on. Its BASELINE tag is in
               reproduce/empd_draws.py.
  PAIRING      CHOSEN -- partition by partition, because the lookup and the
               network are scored on the same partitions, so every contrast
               here is within-partition.
  BOOTSTRAP    CHOSEN -- numpy default_rng(42), 2000 resamples, alpha 0.05;
               reproduce/htem_split.py's tag derives the endpoints' Monte Carlo
               error at 2000 resamples.
  Nothing is FITTED: no quantity here is matched to a target.

    python reproduce/lookup_margins.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "margins" / "results.json"

BOOTSTRAP_SEED = 42
BOOTSTRAP_REPLICATES = 2000
ALPHA = 0.05

DATASETS = (
    ("aihub", "EMPD", ("resistivity", "hardness")),
)


def mean_ci(values) -> dict:
    x = np.asarray(values, dtype=float).ravel()
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = [x[rng.integers(0, x.size, x.size)].mean()
             for _ in range(BOOTSTRAP_REPLICATES)]
    lo, hi = (float(np.percentile(draws, 100 * ALPHA / 2)),
              float(np.percentile(draws, 100 * (1 - ALPHA / 2))))
    return {"mean": float(x.mean()), "sd": float(x.std(ddof=1)),
            "ci_lo": lo, "ci_hi": hi, "n": int(x.size)}


def analyse(dataset: str, label: str, tasks) -> dict:
    path = ROOT / "outputs" / dataset / "draws.json"
    record = json.loads(path.read_text())
    seeds = record["draw_seeds"]

    print(f"{label}, from {record['script']}")
    print(f"{len(seeds)} draws x {len(record['init_seeds'])} inits")

    out = {"dataset": dataset, "label": label, "source": record["script"],
           "draw_seeds": seeds, "n_inits": len(record["init_seeds"]),
           "rows": record["rows"], "systems": record["systems"], "tasks": {}}

    for task in tasks:
        per = [record["draws"][str(s)][task] for s in seeds]
        lookup = np.array([d["lookup_r2"] for d in per])
        model = np.array([np.mean(d["model_r2_per_init"]) for d in per])
        margin = model - lookup
        summary = mean_ci(margin)

        entry = {
            "rows": record["rows"][task],
            "systems": None if record["systems"] is None else record["systems"][task],
            "lookup_r2": {"mean": float(lookup.mean()),
                          "sd": float(lookup.std(ddof=1))},
            "model_r2": {"mean": float(model.mean()),
                         "sd": float(model.std(ddof=1))},
            "margin": summary,
            "margin_per_draw": [float(v) for v in margin],
            "margin_min": float(margin.min()), "margin_max": float(margin.max()),
            "margin_outside_ci": int(((margin < summary["ci_lo"])
                                      | (margin > summary["ci_hi"])).sum()),
        }
        print(f"\n  {task}  ({entry['rows']:,} rows"
              + (f" over {entry['systems']} systems)" if entry["systems"] else ")"))
        print(f"    lookup      {lookup.mean():+.4f} +- {lookup.std(ddof=1):.4f}")
        print(f"    model       {model.mean():+.4f} +- {model.std(ddof=1):.4f}")
        print(f"    margin      {summary['mean']:+.4f}  95% CI "
              f"[{summary['ci_lo']:+.4f}, {summary['ci_hi']:+.4f}]"
              f"   sd {summary['sd']:.4f}  n {summary['n']}")

        # The blocked partition, where the dataset has one. Paired to the random
        # partition draw by draw: the two stages ran the same committed seed
        # set, so the drop is a within-draw contrast rather than a difference of
        # two independently drawn means.
        if "blocked_model_r2_per_init" in per[0]:
            blocked_lookup = np.array([d["blocked_lookup_r2"] for d in per])
            blocked = np.array([np.mean(d["blocked_model_r2_per_init"]) for d in per])
            entry["blocked_lookup_r2"] = mean_ci(blocked_lookup)
            entry["blocked_model_r2"] = mean_ci(blocked)
            entry["split_drop"] = mean_ci(model - blocked)
            entry["drop_minus_margin"] = mean_ci(lookup - blocked)
            entry["blocked_n_test"] = [d["blocked_n_test"] for d in per]
            # Each draw puts a different set of systems in the test set, so these
            # ten numbers are R^2 over ten different row sets against ten
            # different target variances. Their mean is reported and so is their
            # shape: over a distribution this wide the mean describes no typical
            # partition, and sec:esi-draw-spread says so from the range and the
            # median recorded here rather than from the interval, because a
            # percentile bootstrap over ten partitions covers well below its
            # stated level.
            entry["blocked_per_draw"] = [float(v) for v in blocked]
            entry["blocked_median"] = float(np.median(blocked))
            entry["blocked_min"] = float(blocked.min())
            entry["blocked_max"] = float(blocked.max())
            excess_per_draw = lookup - blocked
            entry["excess_per_draw"] = [float(v) for v in excess_per_draw]
            entry["excess_median"] = float(np.median(excess_per_draw))
            entry["excess_min"] = float(excess_per_draw.min())
            entry["excess_max"] = float(excess_per_draw.max())
            entry["excess_negative_draws"] = int((excess_per_draw < 0).sum())
            drop, excess = entry["split_drop"], entry["drop_minus_margin"]
            print(f"    blocked lookup {blocked_lookup.mean():+.4f} +- "
                  f"{blocked_lookup.std(ddof=1):.4f}"
                  f"   95% CI [{entry['blocked_lookup_r2']['ci_lo']:+.4f}, "
                  f"{entry['blocked_lookup_r2']['ci_hi']:+.4f}]")
            print(f"    blocked     {blocked.mean():+.4f} +- {blocked.std(ddof=1):.4f}"
                  f"   95% CI [{entry['blocked_model_r2']['ci_lo']:+.4f}, "
                  f"{entry['blocked_model_r2']['ci_hi']:+.4f}]"
                  + ("  spans the constant predictor"
                     if entry["blocked_model_r2"]["ci_hi"] >= 0 >
                     entry["blocked_model_r2"]["mean"] else ""))
            print(f"    drop        {drop['mean']:+.4f}  95% CI "
                  f"[{drop['ci_lo']:+.4f}, {drop['ci_hi']:+.4f}]")
            print(f"    drop - margin {excess['mean']:+.4f}  95% CI "
                  f"[{excess['ci_lo']:+.4f}, {excess['ci_hi']:+.4f}]"
                  f"   median {entry['excess_median']:+.4f}"
                  f"   range [{entry['excess_min']:+.4f}, "
                  f"{entry['excess_max']:+.4f}]"
                  f"   negative on {entry['excess_negative_draws']}"
                  f"/{len(seeds)} draws")
            print(f"    blocked rows per draw "
                  f"[{min(entry['blocked_n_test']):,}, "
                  f"{max(entry['blocked_n_test']):,}] of {entry['rows']:,}")

        print(f"    range       [{margin.min():+.4f}, {margin.max():+.4f}]"
              f"   outside the interval over all draws "
              f"{entry['margin_outside_ci']}/{len(seeds)}")

        out["tasks"][task] = entry
    print()
    return out


def main() -> None:
    results = {"bootstrap": {"seed": BOOTSTRAP_SEED,
                             "replicates": BOOTSTRAP_REPLICATES, "alpha": ALPHA},
               "datasets": {name: analyse(name, label, tasks)
                           for name, label, tasks in DATASETS}}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
