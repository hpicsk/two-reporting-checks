# Two reporting checks for machine learning accuracy on combinatorial materials libraries

The code and the result files of the article of the same title.

A combinatorial library is a film whose composition varies continuously across
its substrate. An instrument measures a property at many positions on the
library. A random partition of a combinatorial dataset's rows tests a model on
positions adjacent to training positions on the same library, so the model's R²
can overstate its accuracy on a library excluded from training. The article
proposes two checks of a model's random-partition R².

- The first check scores a lookup on the model's test rows. The lookup
  (`src/metal/lookup.py`) copies the target of the training row nearest to each
  test row in the model's features. It fits no parameters to the target, so its
  R² comes only from training rows that resemble the test rows. The model's
  margin is the model's R² minus the lookup's.
- The second check reports the drop in the model's R² from the random
  partition to a blocked partition, which excludes whole libraries from
  training on HTEM-DB and whole alloy systems on EMPD. The excess is the drop
  minus the margin, which equals the lookup's random-partition R² minus the
  model's blocked-partition R².

## Results

The two tables are Table 1 of the article, rounded as the article's Methods
state.

**Under a random partition of the rows.**

| dataset and property | rows | group | lookup R² | model R² | model − lookup |
|---|---|---|---|---|---|
| Kim & Lee 2025, resistivity | 71,633 | 43 systems, not recorded | 0.9924 | 0.9946 | +0.0022 |
| EMPD, resistivity | 58,937 | 25 systems | 0.887 | 0.946 | +0.059 [+0.057, +0.061] |
| EMPD, hardness | 901 | 22 systems | 0.873 | 0.921 | +0.048 [+0.035, +0.063] |
| HTEM-DB, resistivity | 8,628 | 257 libraries | 0.84 | 0.90 | +0.057 [+0.042, +0.075] |
| HTEM-DB, band gap | 5,264 | 156 libraries | 0.66 | 0.78 | +0.12 [+0.09, +0.16] |
| Matbench, band gap | 4,604 | no libraries | 0.26 | 0.48 | +0.23 [+0.17, +0.29] |

**Under a blocked partition.**

| dataset and property | lookup R² | model R² | excess |
|---|---|---|---|
| EMPD, resistivity | −0.1 [−0.6, 0.4] | 0.1 [−0.2, 0.4] | +0.8 [+0.4, +1.1] |
| EMPD, hardness | −4 [−10, 0] | −6 [−12, −1] | +7 [+2, +13] |
| HTEM-DB, resistivity | 0.14 [−0.06, 0.31] | 0.36 [0.20, 0.48] | +0.48 [+0.37, +0.62] |
| HTEM-DB, band gap | 0.2 [−0.1, 0.4] | 0.47 [0.33, 0.58] | +0.19 [+0.10, +0.29] |

HTEM-DB resistivity is modeled as its base-10
logarithm. Kim and Lee's scores are on the one partition that their files
define. Every other score is the mean over the dataset's partitions, which are
ten random partitions on each dataset, ten blocked partitions on EMPD and one
blocked partition on HTEM-DB. Bracketed ranges are 95% percentile bootstrap
intervals over 2,000 resamples, of libraries for HTEM-DB, of compounds for
Matbench, and of the ten partitions for EMPD. The EMPD intervals
therefore omit how a score would vary over other alloy systems, so the positive
excess is established on HTEM-DB alone. Kim and Lee's row has no interval,
because their files define a single partition and record no library.

On EMPD hardness, the ten random partitions give margins from +0.017 to +0.103,
so a margin measured on one partition can be more than twice the mean, +0.048.
The margin varies less on EMPD resistivity, HTEM-DB resistivity, HTEM-DB band
gap and Matbench, whose ranges are +0.053 to +0.064, +0.051 to +0.064, +0.114 to
+0.130 and +0.193 to +0.267.

Matbench, the control dataset, has no libraries. On it, the lookup's R² is
lower than on any of the five pairs of a combinatorial dataset and a property,
and the margin is wider than on any of them. Matbench matches HTEM-DB band gap in the kind of property, model and partition,
but 53% of its compounds have a band gap of zero, whereas every HTEM-DB band gap
exceeds zero. Its lookup scores 0.26 [0.19, 0.32], below the 0.66 [0.56, 0.74] on HTEM-DB band
gap. On the 2,154 Matbench compounds whose gap exceeds zero, however, the lookup
scores 0.51 [0.45, 0.56], an interval whose upper end, 0.562, exceeds the lower end of HTEM-DB band gap's interval, 0.559,
so the zero gaps, and
not only the absence of libraries, can lower the lookup's score on Matbench. On
HTEM-DB itself, the lookup's R² falls further than the model's under the blocked
partition on both properties, and on resistivity it falls by more than half. The
wider Matbench margin is not established, because the comparison
moved to the means over the ten partitions only after the intervals of the
Matbench margin and the HTEM-DB band gap margin overlapped on the first
partition.

Three limits bound what the results show.

- A drop has two sources, the adjacent training positions that a blocked
  partition removes and the difficulty of predicting a library excluded from
  training from composition and a few processing variables, so neither the drop
  nor the excess measures the contribution of adjacent positions alone.
- The positive excess on HTEM-DB is shown only for an untuned random forest and
  for two tuned models, a forest and a gradient-boosting machine
  (`reproduce/htem_ablation.py`), so other models could give a smaller excess.
- EMPD and Kim and Lee's dataset share measurements, so the five pairs come from
  two sources of libraries and are not five independent tests.

## Reproduce

The scripts regenerate the committed result files under the versions that
`requirements.txt` pins, on Python 3.12:

    pip install -r requirements.txt

Everything except EMPD runs from a clone. `data/` holds the HTEM-DB rows, the
Matbench task. `reproduce/fetch_npj.py` downloads Kim and
Lee's files from their repository at a pinned commit and checks each file
against a recorded checksum.

    python reproduce/fetch_npj.py        # Kim and Lee's files, then:
    python reproduce/npj_split.py        # Kim and Lee's row
    python reproduce/htem_split.py       # both HTEM-DB rows (reproduce/fetch_htem.py refreshes data/htem)
    python reproduce/expt_gap_control.py # Matbench row, on HTEM-DB's ten seeds (reproduce/fetch_expt_gap.py refreshes data/expt_gap)
    python reproduce/htem_ablation.py    # the tuned models on both HTEM-DB partitions
    python reproduce/lookup_margins.py   # EMPD rows

EMPD cannot be redistributed, so its per-partition scores are committed in
`outputs/aihub/draws.json`, and the counts that the Methods print about its
release are committed in `outputs/aihub/structure.json`. Given a local copy of
the dataset, both files regenerate:

    METAL_AIHUB_PATH=/path/to/01-1.정식개방데이터 python reproduce/empd_draws.py
    METAL_AIHUB_PATH=/path/to/01-1.정식개방데이터 python reproduce/aihub_comeasurement.py

The two scripts read the dataset's location from `METAL_AIHUB_PATH` with no
default, so a missing variable stops the run with an error that names the
variable. `reproduce/aihub_comeasurement.py` also asserts that the row and system counts
in `draws.json` are the ones that it reads from the dataset, and
`reproduce/lookup_margins.py` runs again after the two EMPD scripts.

A network's score reproduces to the last digit only on the device that computed
it, which is CUDA for the EMPD random partitions and the CPU at four threads for
the EMPD blocked partitions.

Then the paper:

    python reproduce/make_parameters.py  # paper/parameters.tex + paper/tables/
    python reproduce/make_figure.py      # paper/figures/
    python -m pytest test_assertions.py  # every claim against its result file
    cd paper && latexmk -pdf main && latexmk -pdf esi

`paper/parameters.tex` declares, per block, the artifact over which each number
is measured. The suite checks that every declared artifact exists.
