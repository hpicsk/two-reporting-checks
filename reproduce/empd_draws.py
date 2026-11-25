"""The lookup and the network on EMPD's ten random and ten blocked partitions.

Each seed draws one random and one blocked partition of the union of EMPD's
three tables (src/metal/aihub.py). On both, a network is trained per property
from each of three initializations (src/metal/network.py). The lookup
(src/metal/lookup.py) is scored on both, on the network's features. Writes
outputs/aihub/draws.json, which
reproduce/lookup_margins.py reads.

Devices. The random-partition networks ran on CUDA and the blocked-partition
networks on the CPU at CPU_THREADS threads, and each cell reproduces its value
only on its own device. The same seeds on another device follow a different
trajectory in every cell, so a run elsewhere is a fresh draw of the same
experiment rather than a copy of this one. The cells run in parallel
processes; each seeds itself, so no value depends on WORKERS.

Degrees of freedom:
  DRAW_SEEDS   CHOSEN -- ten seeds, each drawing one random and one blocked
               partition (sec:protocol).
  INIT_SEEDS   CHOSEN -- three initializations per partition; a partition's
               score is their mean (sec:predictors).
  DEVICES, CPU_THREADS  CHOSEN -- CUDA for the random partitions and the CPU
               at CPU_THREADS threads for the blocked ones, the devices on which
               the committed values were produced and on which alone they
               reproduce (Devices, above).
  FEATURES     CHOSEN -- the fifteen element percentages, the six descriptors
               the tables ship, and the film thickness every row records
               (src/metal/aihub.py), as sec:features states.
  TARGET_SCALERS CHOSEN -- median and interquartile range for resistivity,
               standardization for hardness, each fitted on the training rows.
               Both are affine, so neither moves an R^2.
  MLP_CONFIG   CHOSEN -- src/metal/network.py. See the BASELINE tag.
  Nothing is FITTED to a target statistic.

BASELINE: a two-hidden-layer perceptron at a fixed configuration | obvious
alternative: a referee names a gradient-boosted tree ensemble, the usual
stronger model on tabular data this size | tuning: none; the configuration is
fixed before any partition here and is the same under both partitions and all
ten seeds | discharge: claim reduced -- sec:esi-draw-spread places each EMPD
blocked mean of the model against the constant predictor by its interval, and
par:partition rests the positive excess on HTEM-DB alone, where
reproduce/htem_ablation.py repeats it under two tuned models

    METAL_AIHUB_PATH=... python reproduce/empd_draws.py
"""

from __future__ import annotations

import json
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import RobustScaler, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from metal import aihub  # noqa: E402
from metal.lookup import one_nn  # noqa: E402
from metal.network import MLP_CONFIG, fit_and_score, resolve_device  # noqa: E402

OUT = ROOT / "outputs" / "aihub" / "draws.json"

DRAW_SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144]
INIT_SEEDS = [42, 123, 456]
TARGET_SCALERS = {"resistivity": RobustScaler, "hardness": StandardScaler}
PARTITIONS = {"random": aihub.random_partition, "blocked": aihub.blocked_partition}
DEVICES = {"random": "cuda", "blocked": "cpu"}
CPU_THREADS = 4
WORKERS = 12

_rows = None


def _cell(kind: str, seed: int, task: str, init: int) -> float:
    """One network: one partition, one property, one initialization."""
    global _rows
    if _rows is None:
        _rows = aihub.union()
    torch.set_num_threads(CPU_THREADS)
    parts = PARTITIONS[kind](_rows, seed)
    return fit_and_score(parts, aihub.FEATURE_COLS, aihub.TARGET_COLS[task],
                         f"has_{task}", TARGET_SCALERS[task], init,
                         resolve_device(DEVICES[kind]))


def _lookup(parts, task: str) -> tuple[float, int]:
    """The lookup's R^2 over the test rows carrying ``task``, copying from
    the training rows that carry it."""
    train, _stop, test = parts
    mask, column = f"has_{task}", aihub.TARGET_COLS[task]
    train, test = train[train[mask]], test[test[mask]]
    predicted, _ = one_nn(train[aihub.FEATURE_COLS].to_numpy(float),
                          train[column].to_numpy(float),
                          test[aihub.FEATURE_COLS].to_numpy(float))
    y = test[column].to_numpy(float)
    r2 = 1.0 - ((y - predicted) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return float(r2), len(test)


def main() -> None:
    for kind in PARTITIONS:
        resolve_device(DEVICES[kind])
    rows = aihub.union()
    cells = [(kind, seed, task, init) for kind in PARTITIONS for seed in DRAW_SEEDS
             for task in TARGET_SCALERS for init in INIT_SEEDS]
    with ProcessPoolExecutor(WORKERS, multiprocessing.get_context("spawn")) as pool:
        futures = {cell: pool.submit(_cell, *cell) for cell in cells}

        draws = {}
        for seed in DRAW_SEEDS:
            random_parts = aihub.random_partition(rows, seed)
            blocked_parts = aihub.blocked_partition(rows, seed)
            draws[str(seed)] = {}
            for task in TARGET_SCALERS:
                lookup_r2, n_test = _lookup(random_parts, task)
                blocked_lookup_r2, blocked_n_test = _lookup(blocked_parts, task)
                draws[str(seed)][task] = {
                    "lookup_r2": lookup_r2,
                    "model_r2_per_init": [futures["random", seed, task, i].result()
                                          for i in INIT_SEEDS],
                    "n_test": n_test,
                    "blocked_lookup_r2": blocked_lookup_r2,
                    "blocked_model_r2_per_init": [
                        futures["blocked", seed, task, i].result() for i in INIT_SEEDS],
                    "blocked_n_test": blocked_n_test,
                }
                d = draws[str(seed)][task]
                print(f"draw {seed:5d} {task:12s} lookup {lookup_r2:.4f}  "
                      f"model {np.mean(d['model_r2_per_init']):.4f}  "
                      f"blocked lookup {blocked_lookup_r2:+.4f}  "
                      f"blocked {np.mean(d['blocked_model_r2_per_init']):+.4f}",
                      flush=True)

    record = {
        "script": "reproduce/empd_draws.py",
        "draw_seeds": DRAW_SEEDS,
        "init_seeds": INIT_SEEDS,
        "devices": {**DEVICES, "cpu_threads": CPU_THREADS},
        "model_config": MLP_CONFIG,
        "features": {"elements": aihub.ELEMENT_COLS,
                     "descriptors": aihub.DESCRIPTOR_COLS,
                     "processing": aihub.PROCESSING_COLS},
        "rows": {task: int(rows[f"has_{task}"].sum()) for task in aihub.CSV_FILES},
        "systems": {task: int(rows.loc[rows[f"has_{task}"], aihub.SYSTEM_COL].nunique())
                    for task in aihub.CSV_FILES},
        "blocked_target_share": aihub.BLOCKED_SHARE,
        "draws": draws,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(record, indent=2) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
