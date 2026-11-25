"""Turn the committed result files into the manuscript's numbers.

Two outputs, one source of truth. `paper/parameters.tex` holds every value the
prose prints, as semantic macros; `paper/tables/main.tex` holds the table body.
Both are regenerated from outputs/ and neither is hand-edited.

POPULATIONS. Two row sets of one size read as one population in a sentence
that prints both counts. Every macro therefore ships under a population marker naming the
artifact it was measured on
and the unit noun a count of it takes, and test_assertions.py fails when two
population names resolve to one artifact. The unit noun is the grep handle: a
count printed beside the wrong noun is the failure no binding can see.

PRECISION. A value that has a bootstrap interval, printed or not, is rounded
to the digit that the interval resolves: the first significant digit of its
half-width, or the second where the first is 1 (`resolved`). Every other R^2 is
printed to three decimals, except on Kim and Lee's row, which has no interval and
is printed to four, because at three its margin, the row's finding, would keep
only one significant digit (sec:statistics). Each value is rounded separately.
RMSE to two decimals, distances to five, counts with a thousands separator.

    python reproduce/make_parameters.py
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from metal import aihub as empd_module  # noqa: E402
OUTPUTS = ROOT / "outputs"
PARAMETERS = ROOT / "paper" / "parameters.tex"
TABLE = ROOT / "paper" / "tables" / "main.tex"
TABLE_ABLATION = ROOT / "paper" / "tables" / "ablation.tex"

BEGIN = "% ---- BEGIN GENERATED ----"
END = "% ---- END GENERATED ----"

# artifact and unit noun per population, in the order the manuscript meets them
POPULATIONS = {
    "npjDataset": ("data/npj/csv/Train.csv+data/npj/csv/Val.csv", "rows"),
    "npjTrain": ("data/npj/csv/Train.csv", "rows"),
    "npjVal": ("data/npj/csv/Val.csv", "rows"),
    "empdRes": ("outputs/aihub/draws.json:resistivity", "rows"),
    "empdHard": ("outputs/aihub/draws.json:hardness", "rows"),
    "empdAmorph": ("outputs/aihub/draws.json:amorphous", "rows"),
    "npjShared": ("outputs/aihub/structure.json:npj_shared_measurements", "rows"),
    "htemRes": ("outputs/htem/split.json:log10_resistivity", "rows"),
    "htemGap": ("outputs/htem/split.json:direct_band_gap", "rows"),
    "htemResAblation": ("outputs/htem/ablation.json:log10_resistivity", "rows"),
    "htemGapAblation": ("outputs/htem/ablation.json:direct_band_gap", "rows"),
    "exptGap": ("outputs/expt_gap/control.json", "rows"),
    "design": ("reproduce/", "settings"),
    "software": ("requirements.txt", "versions"),
    "programs": ("reproduce/make_parameters.py", "sources"),
}

# A macro whose unit or artifact differs from its block's, declared on its own
# line, because a block names one population and one artifact can be counted
# by several units: a table by its rows, by the alloy systems in it, by its
# features. The noun that the manuscript prints after each count is checked
# against the unit declared here (test_assertions.py).
UNIT_OF = {
    "npjFeatureCount": "features", "npjCompositionFeatureCount": "descriptors",
    "empdResSystems": "systems", "empdHardSystems": "systems",
    "empdResValidationOnlySystems": "systems", "empdHardValidationOnlySystems": "systems",
    "empdResBlockedNegative": "partitions", "empdResBlockedPositive": "partitions",
    "empdHardBlockedNegative": "partitions", "npjSharedMinDecimals": "decimals",
    "htemResLibraries": "libraries", "htemGapLibraries": "libraries",
    "exptGapNonzeroRows": "compounds",
    "mlpFirstLayer": "units", "mlpSecondLayer": "units", "mlpEpochs": "epochs",
    "mlpPatience": "epochs", "mlpLrPatience": "epochs", "mlpBatchSize": "rows",
    "randomPairs": "pairs", "blockedPairs": "pairs", "libraryDatasets": "datasets",
    "blockedDatasets": "datasets", "splitDraws": "partitions", "splitInits": "times",
    "htemFolds": "folds", "htemInnerFolds": "folds", "htemTrees": "trees",
    "htemTunedTrees": "trees", "empdElementCount": "elements",
    "empdDescriptorCount": "descriptors", "htemSearchBudget": "configurations",
    "htemBoostRounds": "rounds", "bootstrapReplicates": "resamples",
}
# Macros of the design block that are measured, and those read from another
# artifact than their block's.
ARTIFACT_OF = {
    "randomPairs": "outputs/ (the pairs that the result files score)",
    "blockedPairs": "outputs/ (the pairs that the result files score)",
    "libraryDatasets": "outputs/ (the pairs that the result files score)",
    "blockedDatasets": "outputs/ (the pairs that the result files score)",
    "largestRandomMargin": "outputs/ (the pairs that the result files score)",
    "largestRandomMarginPair": "outputs/ (the pairs that the result files score)",
    "htemNoTemperatureShift": "outputs/htem/split.json",
    "htemNoTemperatureLookupShift": "outputs/htem/split.json",
}


def load(name):
    return json.loads((OUTPUTS / name).read_text())


def pinned():
    """{package: version} from requirements.txt, the environment the Methods name."""
    out = {}
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        line = line.split("#")[0].strip()
        if line:
            name, version = line.split("==")
            out[name] = version
    return out


def count(n):
    return f"{n:,}".replace(",", "{,}")


def r2(x, places=3):
    # A score below zero prints with a typeset minus, as signed() prints one.
    return unsigned_zero(f"{x:.{places}f}").replace("-", "$-$")


def signed(x, places=3):
    return unsigned_zero(f"{x:+.{places}f}").replace("-", "$-$")


def unsigned_zero(text):
    """A value that rounds to zero prints without a sign, because "-0" states a
    sign that the printed digits do not carry."""
    return text.lstrip("+-") if float(text) == 0 else text


def endpoints(d):
    # The two bootstraps name their endpoints differently: the draw-paired one
    # in reproduce/lookup_margins.py writes ci_lo/ci_hi, the cluster bootstrap in
    # reproduce/htem_split.py writes lo/hi.
    lo = d["ci_lo"] if "ci_lo" in d else d["lo"]
    hi = d["ci_hi"] if "ci_hi" in d else d["hi"]
    return lo, hi


def _leading(x):
    """The first significant digit of x > 0 and the power of ten it stands at."""
    mantissa, exponent = f"{x:.12e}".split("e")
    return int(mantissa[0]), int(exponent)


def resolved(d):
    """Decimal places that an interval resolves (PRECISION): the half-width
    rounded to one significant digit, or two where the first is 1, read again
    after the rounding, since 0.0965 rounds to 0.10."""
    lo, hi = endpoints(d)
    half = (hi - lo) / 2
    digit, power = _leading(half)
    half = round(half, -power + (1 if digit == 1 else 0))
    digit, power = _leading(half)
    return max(0, -power + (1 if digit == 1 else 0))


def interval(d, places=None, fmt=None):
    # Margins print signed; an R^2 interval prints as its point value does.
    fmt = fmt or signed
    places = resolved(d) if places is None else places
    lo, hi = endpoints(d)
    return f"[{fmt(lo, places)}, {fmt(hi, places)}]"


def with_interval(name, value, d, fmt=None):
    """A value and its interval, both to the places the interval resolves."""
    fmt = fmt or signed
    places = resolved(d)
    return [(name, fmt(value, places)), (f"{name}CI", interval(d, places, fmt))]


def at_resolution(value, d, fmt=None):
    """A value whose interval exists but is not printed beside it."""
    return (fmt or r2)(value, resolved(d))


SMALL = ("zero", "one", "two", "three", "four", "five", "six", "seven",
         "eight", "nine", "ten")


def power_of_ten(x):
    """A setting that is an exact power of ten, as $10^{k}$."""
    k = round(math.log10(x))
    assert math.isclose(x, 10.0 ** k), f"{x} is not a power of ten"
    return f"$10^{{{k}}}$"


def words(n):
    """Counts at or below ten are spelled, which is what the prose needs; the
    table's counts are all larger and print as digits."""
    return SMALL[n] if 0 <= n <= 10 else count(n)


def options(values):
    """A swept setting's values as the caption lists them: "5, 10 or 20"."""
    values = sorted(values)
    return ", ".join(count(v) for v in values[:-1]) + " or " + count(values[-1])


def pct(x, places=1):
    return f"{100 * x:.{places}f}\\%"


def at_most(x, places):
    """A bound, rounded up, so that "at most" holds at the printed precision."""
    return f"{math.ceil(x * 10 ** places) / 10 ** places:.{places}f}"


def ablation_macros(prefix, target):
    """One target's tuned arms.

    The untuned arm is not repeated here: it is the manuscript's own forest and
    already prints from the split.json blocks, so a second macro for it would be
    two names for one number.
    """
    out = []
    for arm, suffix in (("tuned_forest", "Tuned"), ("boosted", "Boosted")):
        entry = target["arms"][arm]
        out += [(f"{prefix}Random{suffix}",
                 at_resolution(entry["random_model_r2"], entry["ci_random_model_r2"])),
                (f"{prefix}Blocked{suffix}",
                 at_resolution(entry["blocked_model_r2"], entry["ci_blocked_model_r2"])),
                *with_interval(f"{prefix}Excess{suffix}", entry["excess"], entry["ci"])]
    return out


def build():
    npj = load("npj/split.json")
    htem = load("htem/split.json")
    ablation = load("htem/ablation.json")
    margins = load("margins/results.json")
    aihub = margins["datasets"]["aihub"]["tasks"]
    expt = load("expt_gap/control.json")

    res, hard = aihub["resistivity"], aihub["hardness"]
    config = load("aihub/draws.json")["model_config"]
    # The share of rows each random partition of EMPD scores, the dataset whose
    # partitions are drawn per seed. sec:protocol prints one share, so it must
    # hold for every draw of every property, at the precision printed.
    held_out = set()
    for dataset in ("aihub",):
        record = load(f"{dataset}/draws.json")
        for draw in record["draws"].values():
            for task, entry in draw.items():
                held_out.add(round(100 * entry["n_test"] / record["rows"][task]))
    if len(held_out) != 1:
        raise ValueError(f"random partitions score different shares of rows: "
                         f"{sorted(held_out)}%")
    # The share of rows each EMPD blocked partition holds out is decided by
    # the alloy systems drawn, so it varies; sec:protocol prints its
    # range per property, and prints the random partitions' share as the
    # blocked test set's target size, so the two must be one value.
    empd = load("aihub/draws.json")
    blocked_target = round(100 * empd["blocked_target_share"]["test"])
    if {blocked_target} != held_out:
        raise ValueError(f"the blocked target share {blocked_target}% is not the "
                         f"random partitions' {sorted(held_out)}%")
    blocked_share = {task: [draw[task]["blocked_n_test"] / empd["rows"][task]
                            for draw in empd["draws"].values()]
                     for task in ("resistivity", "hardness")}
    # What the release holds beyond its tables, and the rows it shares with Kim
    # and Lee's files, as reproduce/aihub_comeasurement.py read them.
    structure = load("aihub/structure.json")
    release, shared = structure["release"], structure["npj_shared_measurements"]
    order = release["split_order"]
    grid_low, grid_high, grid_step = shared["factor_grid"]
    hres = htem["targets"]["log10_resistivity"]
    hgap = htem["targets"]["direct_band_gap"]
    ares = ablation["targets"]["log10_resistivity"]
    agap = ablation["targets"]["direct_band_gap"]
    lookup_all = npj["lookup"]["all_features"]
    lookup_comp = npj["lookup"]["composition_only"]

    # The five combinatorial dataset/property pairs measured under a random
    # partition, and the four of them whose group a blocked partition can
    # exclude from training. The
    # controls are excluded from both: they are what the claim is contrasted
    # against.
    random_margins = {
        "Kim and Lee's dataset": npj["model"]["val"]["r2"] - lookup_all["r2"],
        "EMPD resistivity": res["margin"]["mean"],
        "EMPD hardness": hard["margin"]["mean"],
        "HTEM-DB resistivity": hres["model_margin"],
        "HTEM-DB band gap": hgap["model_margin"],
    }
    widest = max(random_margins, key=random_margins.get)
    # The places each pair's margin prints at in tab:main (PRECISION), so the
    # sentence naming the widest margin prints the value its row prints.
    margin_places = {
        "Kim and Lee's dataset": 4,
        "EMPD resistivity": resolved(res["margin"]),
        "EMPD hardness": resolved(hard["margin"]),
        "HTEM-DB resistivity": resolved(hres["ci"]["model_margin"]),
        "HTEM-DB band gap": resolved(hgap["ci"]["model_margin"]),
    }
    # The shares of a random partition of EMPD, as the module that draws them
    # fixes them.
    test_share, stop_share = empd_module.HELD_OUT_SHARE, empd_module.VALIDATION_SHARE
    # The commit of Kim and Lee's repository that reproduce/fetch_npj.py checks out.
    spec = importlib.util.spec_from_file_location("fetch_npj", ROOT / "reproduce" / "fetch_npj.py")
    fetch_npj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fetch_npj)
    # Which source of libraries each random-partition pair comes from. The pair
    # count above is derived from the results; this maps that same set onto
    # sources, so a pair added there and not here raises instead of leaving
    # \\programCount stale. EMPD's
    # release and Kim and Lee's dataset are counted as one source because they
    # share measurements, as sec:datasets shows.
    program_of = {
        "Kim and Lee's dataset": "Kairos Lab, SKKU and Yonsei",
        "EMPD resistivity": "Kairos Lab, SKKU and Yonsei",
        "EMPD hardness": "Kairos Lab, SKKU and Yonsei",
        "HTEM-DB resistivity": "NREL",
        "HTEM-DB band gap": "NREL",
    }
    programs = {program_of[pair] for pair in random_margins}
    # The dataset each random-partition pair comes from, over the same derived
    # pair set, so the dataset count that the Abstract and par:intro print
    # moves with the pairs.
    dataset_of = {
        "Kim and Lee's dataset": "Kim and Lee's dataset",
        "EMPD resistivity": "EMPD",
        "EMPD hardness": "EMPD",
        "HTEM-DB resistivity": "HTEM-DB",
        "HTEM-DB band gap": "HTEM-DB",
    }
    library_datasets = {dataset_of[pair] for pair in random_margins}
    # Keyed by the dataset each pair belongs to, so the pair count and the dataset
    # count are the same derivation: four pairs, and fewer datasets than pairs.
    blocked_excess = {
        ("EMPD", "resistivity"): res["drop_minus_margin"]["mean"],
        ("EMPD", "hardness"): hard["drop_minus_margin"]["mean"],
        ("HTEM-DB", "log10 resistivity"): hres["drop_minus_margin"],
        ("HTEM-DB", "direct band gap"): hgap["drop_minus_margin"],
    }
    blocked_datasets = {dataset for dataset, _ in blocked_excess}
    versions = pinned()

    blocks = [
        ("npjDataset", [
            ("npjDatasetRows", count(npj["n_train"] + npj["n_val"])),
        ]),
        ("npjTrain", [
            ("npjModelTrainRmse", f"{npj['model']['train']['rmse']:.2f}"),
            ("npjPaperTrainRmse", f"{npj['paper']['train_rmse']:.2f}"),
        ]),
        ("npjVal", [
            ("npjModelRSq", r2(npj["model"]["val"]["r2"], 4)),
            ("npjModelRmse", f"{npj['model']['val']['rmse']:.2f}"),
            ("npjPaperRmse", f"{npj['paper']['val_rmse']:.2f}"),
            ("npjFeatureCount", words(lookup_all["n_features"])),
            ("npjCompositionFeatureCount", words(lookup_comp["n_features"])),
            ("npjLookupRSq", r2(lookup_all["r2"], 4)),
            ("npjMargin", signed(npj["model"]["val"]["r2"] - lookup_all["r2"], 4)),
            ("npjLookupRmse", f"{lookup_all['rmse']:.2f}"),
        ]),
        ("empdRes", [
            ("empdResBlockedHeldOutMin", pct(min(blocked_share["resistivity"]), 0)),
            ("empdResBlockedHeldOutMax", pct(max(blocked_share["resistivity"]), 0)),
            ("empdResRows", count(res["rows"])),
            ("empdResSystems", str(res["systems"])),
            ("empdResLookup", r2(res["lookup_r2"]["mean"])),
            ("empdResModel", r2(res["model_r2"]["mean"])),
            *with_interval("empdResMargin", res["margin"]["mean"], res["margin"]),
            *with_interval("empdResBlockedLookup", res["blocked_lookup_r2"]["mean"],
                           res["blocked_lookup_r2"], r2),
            *with_interval("empdResBlocked", res["blocked_model_r2"]["mean"],
                           res["blocked_model_r2"], r2),
            *with_interval("empdResExcess", res["drop_minus_margin"]["mean"],
                           res["drop_minus_margin"]),
            ("empdResBlockedMin", signed(res["blocked_min"])),
            ("empdResBlockedMax", signed(res["blocked_max"])),
            ("empdResMarginMin", signed(res["margin_min"])),
            ("empdResMarginMax", signed(res["margin_max"])),
            # How many blocked partitions put the model below the constant
            # predictor, which sec:esi-draw-spread counts rather than reading
            # the sign of an interval over ten partitions.
            ("empdResBlockedNegative", words(sum(v < 0 for v in res["blocked_per_draw"]))),
            ("empdResBlockedPositive", words(sum(v > 0 for v in res["blocked_per_draw"]))),
            ("empdResValidationOnlySystems",
             words(order["resistivity"]["systems_in_one_split"]["Validation"])),
        ]),
        ("empdHard", [
            ("empdHardBlockedHeldOutMin", pct(min(blocked_share["hardness"]), 0)),
            ("empdHardBlockedHeldOutMax", pct(max(blocked_share["hardness"]), 0)),
            ("empdHardRows", count(hard["rows"])),
            ("empdHardSystems", str(hard["systems"])),
            ("empdHardLookup", r2(hard["lookup_r2"]["mean"])),
            ("empdHardModel", r2(hard["model_r2"]["mean"])),
            *with_interval("empdHardMargin", hard["margin"]["mean"], hard["margin"]),
            ("empdHardMarginMin", signed(hard["margin_min"])),
            *with_interval("empdHardBlockedLookup", hard["blocked_lookup_r2"]["mean"],
                           hard["blocked_lookup_r2"], r2),
            *with_interval("empdHardBlocked", hard["blocked_model_r2"]["mean"],
                           hard["blocked_model_r2"], r2),
            *with_interval("empdHardExcess", hard["drop_minus_margin"]["mean"],
                           hard["drop_minus_margin"]),
            ("empdHardBlockedMin", signed(hard["blocked_min"])),
            ("empdHardBlockedMax", signed(hard["blocked_max"])),
            ("empdHardBlockedMedian", signed(hard["blocked_median"])),
            ("empdHardBlockedNegative", words(sum(v < 0 for v in hard["blocked_per_draw"]))),
            ("empdHardMarginMax", signed(hard["margin_max"])),
            ("empdHardValidationOnlySystems",
             words(order["hardness"]["systems_in_one_split"]["Validation"])),
        ]),
        ("empdAmorph", [
            ("empdAmorphousRows", count(margins["datasets"]["aihub"]["rows"]["amorphous"])),
        ]),
        ("npjShared", [
            ("npjSharedRows", count(shared["rows_at_best_factor"])),
            ("npjSharedOtherRows", count(shared["most_rows_at_another_factor"])),
            ("npjSharedFactor", f"{shared['best_factor']:.5f}"),
            ("npjFactorGridLow", f"{grid_low:.2f}"),
            ("npjFactorGridHigh", f"{grid_high:.2f}"),
            ("npjFactorGridStep", f"{grid_step:.5f}"),
            ("npjSharedMinDecimals", words(shared["min_decimals"])),
            ("npjSharedThicknessRatio",
             f"{shared['thickness_ratio_npj_to_empd']['median']:.3f}"),
            ("npjSharedThicknessRows", count(shared["rows_with_an_empd_thickness"])),
        ]),
        ("htemRes", [
            ("htemResRows", count(hres["rows"])),
            ("htemResLibraries", str(hres["libraries"])),
            ("htemResRandomLookup", at_resolution(hres["random"]["lookup_r2"],
                                               hres["ci"]["random_lookup_r2"])),
            ("htemResRandomModel", at_resolution(hres["random"]["model_r2"],
                                              hres["ci"]["random_model_r2"])),
            *with_interval("htemResBlockedLookup", hres["blocked"]["lookup_r2"],
                           hres["ci"]["blocked_lookup_r2"], r2),
            *with_interval("htemResBlockedModel", hres["blocked"]["model_r2"],
                           hres["ci"]["blocked_model_r2"], r2),
            *with_interval("htemResMargin", hres["model_margin"], hres["ci"]["model_margin"]),
            *with_interval("htemResExcess", hres["drop_minus_margin"],
                           hres["ci"]["drop_minus_margin"]),
            # C1 and C2 of reproduce/htem_split.py: the lookup's blocked R^2
            # below half its random-partition R^2, and the lookup's fall
            # beyond the model's.
            *with_interval("htemResBeyondHalf", hres["criteria"]["C1"]["value"],
                           hres["criteria"]["C1"]["ci"]),
            *with_interval("htemResFallExcess", hres["criteria"]["C2"]["value"],
                           hres["criteria"]["C2"]["ci"]),
            # C3: the lowest lower endpoint of C2's interval over the fills.
            ("htemResFillFallLo", signed(min(
                f["ci"]["lookup_fall_minus_model_fall"]["lo"]
                for f in hres["no_temperature_sensitivity"]["alternatives"].values()))),
            ("htemResMarginMin", signed(hres["margin_min"])),
            ("htemResMarginMax", signed(hres["margin_max"])),
        ]),
        ("htemGap", [
            ("htemGapRows", count(hgap["rows"])),
            ("htemGapLibraries", str(hgap["libraries"])),
            *with_interval("htemGapRandomLookup", hgap["random"]["lookup_r2"],
                           hgap["ci"]["random_lookup_r2"], r2),
            # E3's comparison at the precision that decides it: the interval's
            # printed lower end meets exptGapNonzeroLookupHi at two decimals.
            ("htemGapRandomLookupLo", r2(endpoints(hgap["ci"]["random_lookup_r2"])[0])),
            ("htemGapRandomModel", at_resolution(hgap["random"]["model_r2"],
                                              hgap["ci"]["random_model_r2"])),
            *with_interval("htemGapBlockedLookup", hgap["blocked"]["lookup_r2"],
                           hgap["ci"]["blocked_lookup_r2"], r2),
            *with_interval("htemGapBlockedModel", hgap["blocked"]["model_r2"],
                           hgap["ci"]["blocked_model_r2"], r2),
            *with_interval("htemGapMargin", hgap["model_margin"], hgap["ci"]["model_margin"]),
            *with_interval("htemGapExcess", hgap["drop_minus_margin"],
                           hgap["ci"]["drop_minus_margin"]),
            # C1 and C2 of reproduce/htem_split.py: the lookup's blocked R^2
            # below half its random-partition R^2, and the lookup's fall
            # beyond the model's.
            *with_interval("htemGapBeyondHalf", hgap["criteria"]["C1"]["value"],
                           hgap["criteria"]["C1"]["ci"]),
            *with_interval("htemGapFallExcess", hgap["criteria"]["C2"]["value"],
                           hgap["criteria"]["C2"]["ci"]),
            # C3: the lowest lower endpoint of C2's interval over the fills.
            ("htemGapFillFallLo", signed(min(
                f["ci"]["lookup_fall_minus_model_fall"]["lo"]
                for f in hgap["no_temperature_sensitivity"]["alternatives"].values()))),
            ("htemGapMarginMin", signed(hgap["margin_min"])),
            ("htemGapMarginMax", signed(hgap["margin_max"])),
            # The margin on the first random partition alone, which
            # sec:esi-draw-spread sets against the second control's own.
            *with_interval("htemGapFirstMargin", hgap["random_partitions"][0]["margin"],
                           hgap["ci"]["first_partition_margin"]),
        ]),
        # The tuned arms. Same rows and folds as the two blocks above, a
        # different artifact: reproduce/htem_ablation.py, which asserts its
        # reference arm against outputs/htem/split.json before writing.
        ("htemResAblation", ablation_macros("htemRes", ares)),
        ("htemGapAblation", ablation_macros("htemGap", agap)),
        ("exptGap", [
            ("exptGapRows", count(expt["rows"])),
            *with_interval("exptGapLookup", expt["random"]["lookup_r2"],
                           expt["ci"]["random_lookup_r2"], r2),
            ("exptGapModel", at_resolution(expt["random"]["model_r2"],
                                           expt["ci"]["random_model_r2"])),
            *with_interval("exptGapMargin", expt["model_margin"], expt["ci"]["model_margin"]),
            ("exptGapMarginMin",
             signed(min(p["margin"] for p in expt["random_partitions"]))),
            ("exptGapMarginMax",
             signed(max(p["margin"] for p in expt["random_partitions"]))),
            *with_interval("exptGapFirstMargin", expt["random_partitions"][0]["margin"],
                           expt["ci"]["first_partition_margin"]),
            # E3 of reproduce/expt_gap_control.py: the rows whose gap exceeds zero.
            ("exptGapZeroShare", pct(expt["zero_gap_share"], 0)),
            ("exptGapNonzeroRows", count(expt["nonzero_gap"]["rows"])),
            *with_interval("exptGapNonzeroLookup", expt["nonzero_gap"]["random"]["lookup_r2"],
                           expt["nonzero_gap"]["ci"]["random_lookup_r2"], r2),
            # The upper end that htemGapRandomLookupLo is compared with.
            ("exptGapNonzeroLookupHi", r2(endpoints(expt["nonzero_gap"]["ci"]["random_lookup_r2"])[1])),
            *with_interval("exptGapNonzeroMargin", expt["nonzero_gap"]["model_margin"],
                           expt["nonzero_gap"]["ci"]["model_margin"]),
        ]),
        ("programs", [
            ("programCount", words(len(programs))),
        ]),
        ("design", [
            ("mlpFirstLayer", str(config["hidden_dim"])),
            ("mlpSecondLayer", str(config["task_hidden_dim"])),
            ("mlpDropout", str(config["dropout"])),
            ("mlpEpochs", str(config["epochs"])),
            ("mlpPatience", str(config["patience"])),
            ("mlpBatchSize", str(config["batch_size"])),
            ("mlpLearningRate", power_of_ten(config["learning_rate"])),
            ("mlpWeightDecay", power_of_ten(config["weight_decay"])),
            ("mlpLrFactor", str(config["lr_factor"])),
            ("mlpLrPatience", str(config["lr_patience"])),
            ("mlpMaxGradNorm", str(config["max_grad_norm"])),
            # Two quantifiers the manuscript ranges over its own table with. Both
            # are computed from the rows themselves rather than from the members a
            # sentence happens to name, so adding a dataset moves them.
            ("randomPairs", words(len(random_margins))),
            ("libraryDatasets", words(len(library_datasets))),
            ("largestRandomMargin", signed(random_margins[widest],
                                           margin_places[widest])),
            # The member the superlative selects, named by the argmax rather
            # than by hand, so a changed maximum moves the sentence too.
            ("largestRandomMarginPair", widest),
            ("blockedPairs", words(len(blocked_excess))),
            ("blockedDatasets", words(len(blocked_datasets))),
            ("splitDraws", words(len(margins["datasets"]["aihub"]["draw_seeds"]))),
            ("splitInits", words(margins["datasets"]["aihub"]["n_inits"])),
            ("heldOutShare", f"{held_out.pop()}\\%"),
            ("empdTrainShare", pct(1 - test_share - stop_share, 0)),
            ("empdStopShare", pct(stop_share, 0)),
            ("npjCommit", fetch_npj.COMMIT[:7]),
            ("htemFolds", words(htem["n_folds"])),
            ("htemTrees", str(htem["rf_trees"])),
            ("htemNoTemperature", signed(htem["no_temperature"], 0)),
            # The values the fill was replaced by, and how far the replacement
            # moved the scores, over both HTEM-DB properties. The two shifts
            # are bounds on a measurement, so the generator writes them, rounded
            # up so that "at most" holds at the printed precision.
            ("htemNoTemperatureAlternatives",
             ", ".join(r2(v, 0) for v in htem["no_temperature_alternatives"][:-1])
             + " or " + r2(htem["no_temperature_alternatives"][-1], 0)),
            ("htemNoTemperatureShift", at_most(max(
                t["no_temperature_sensitivity"]["max_margin_or_excess_shift"]
                for t in htem["targets"].values()), 4)),
            ("htemNoTemperatureLookupShift", at_most(max(
                t["no_temperature_sensitivity"]["max_blocked_lookup_shift"]
                for t in htem["targets"].values()), 3)),
            # The network's features on EMPD, as the script that trained it
            # recorded them (sec:predictors).
            ("empdElementCount", count(len(empd["features"]["elements"]))),
            ("empdDescriptorCount", words(len(empd["features"]["descriptors"]))),
            ("htemTunedTrees", str(ablation["tuned_trees"])),
            ("htemSearchBudget", words(ablation["search_budget"])),
            ("htemInnerFolds", words(ablation["inner_folds"])),
            ("htemBoostRounds", count(ablation["max_rounds"])),
            # The two grids that the budget counts, read from the run so that
            # sec:esi-posthoc lists the configurations the tuned models chose among.
            ("htemForestFeatureShares", options(
                {c["max_features"] for c in ablation["grids"]["tuned_forest"]})),
            ("htemForestMinLeaf", options(
                {c["min_samples_leaf"] for c in ablation["grids"]["tuned_forest"]})),
            ("htemBoostRates", options(
                {c["learning_rate"] for c in ablation["grids"]["boosted"]})),
            ("htemBoostLeaves", options(
                {c["max_leaf_nodes"] for c in ablation["grids"]["boosted"]})),
            # One value on the whole grid; a second would fail the unpacking.
            ("htemBoostPenalty", count(
                *{c["l2_regularization"] for c in ablation["grids"]["boosted"]})),
            ("bootstrapReplicates", count(htem["bootstrap"]["replicates"])),
            # The interval level is fixed in the bootstrap scripts, so the prose
            # follows the constant rather than restating it.
            ("bootstrapLevel", f"{100 * (1 - htem['bootstrap']['alpha']):g}"),
        ]),
        # The libraries that run the lookup, Kim and Lee's network, the
        # perceptron and the forests (sec:predictors), at the versions pinned.
        ("software", [
            ("numpyVersion", versions["numpy"]),
            ("scipyVersion", versions["scipy"]),
            ("sklearnVersion", versions["scikit-learn"]),
            ("torchVersion", versions["torch"]),
        ]),
    ]
    return blocks, (npj, htem, margins)


MANUAL_SEED = """% Numbers the manuscript prints.
%
% Two regions. Everything between the markers below is regenerated from the
% committed result files and is never hand-edited. Anything outside them is a
% fixed convention this repository did not measure -- a value quoted from a
% cited paper -- and is hand-edited and reviewed like any other source.
%
% Each generated block declares the population its macros are measured over: the
% artifact the number came from, and the unit noun a count of that population
% takes. Two population names resolving to one artifact is a test failure.

%% POPULATION cited | artifact: npj Comput. Mater. 11, 300 (2025) | unit: rows
% Quoted from Kim & Lee's article, not measured here.
\\newcommand{\\npjSystems}{43}

%% POPULATION citedRelease | artifact: AI Hub dataset 71339 and Matbench v0.1 | unit: releases
% EMPD's release number, as its data description prints it, and the date on
% which the local copy was downloaded, which the archive's directory records.
% Matbench's version, as Dunn et al. name the suite that holds the task.
\\newcommand{\\empdRelease}{1.0}
\\newcommand{\\empdRetrieved}{23 December 2025}
\\newcommand{\\exptGapVersion}{0.1}

"""


def render_parameters(blocks) -> str:
    lines = [BEGIN]
    for population, macros in blocks:
        artifact, unit = POPULATIONS[population]
        lines.append("")
        lines.append(f"%% POPULATION {population} | artifact: {artifact} | unit: {unit}")
        for name, value in macros:
            line = f"\\newcommand{{\\{name}}}{{{value}}}"
            if name in ARTIFACT_OF or name in UNIT_OF:
                line += (f" % artifact: {ARTIFACT_OF.get(name, artifact)}"
                         f" | unit: {UNIT_OF.get(name, unit)}")
            lines.append(line)
    lines += ["", END]
    return "\n".join(lines)


def render_table(npj, htem, margins) -> str:
    """The table body, as two panels.

    Panel (b) prints no bare random-to-blocked drop. That quantity is the
    difference of two cells the table already carries, and at three decimals the
    band gap row's printed cells subtract to a different number than the drop
    itself does, so printing both would hand a reader two answers to one
    question. What the argument consumes is the drop minus the model's own
    margin, which is exactly (random lookup) minus (blocked model): every panel
    reproduces it from cells that are printed.
    """
    # Panel (a): every dataset under the partition its own source used. The rows
    # column is each pair's size, the whole release on every row, so one column
    # counts one population. The validation rows Kim &
    # Lee's score is taken over are printed where the prose uses them.
    random_rows = [
        ("Kim \\& Lee 2025, resistivity", "\\npjDatasetRows",
         "\\npjSystems\\ systems, not recorded",
         "\\npjLookupRSq", "\\npjModelRSq", "\\npjMargin", ""),
        ("EMPD, resistivity", "\\empdResRows", "\\empdResSystems\\ systems",
         "\\empdResLookup", "\\empdResModel", "\\empdResMargin", "\\empdResMarginCI"),
        ("EMPD, hardness", "\\empdHardRows", "\\empdHardSystems\\ systems",
         "\\empdHardLookup", "\\empdHardModel", "\\empdHardMargin", "\\empdHardMarginCI"),
        ("HTEM-DB, resistivity", "\\htemResRows", "\\htemResLibraries\\ libraries",
         "\\htemResRandomLookup", "\\htemResRandomModel", "\\htemResMargin", "\\htemResMarginCI"),
        ("HTEM-DB, band gap", "\\htemGapRows", "\\htemGapLibraries\\ libraries",
         "\\htemGapRandomLookup", "\\htemGapRandomModel", "\\htemGapMargin", "\\htemGapMarginCI"),
        ("Matbench, band gap", "\\exptGapRows",
         "no libraries",
         "\\exptGapLookup", "\\exptGapModel", "\\exptGapMargin", "\\exptGapMarginCI"),
    ]
    # Panel (b): the datasets whose group is recoverable.
    blocked_rows = [
        ("EMPD, resistivity", "\\empdResBlockedLookup~\\empdResBlockedLookupCI",
         "\\empdResBlocked~\\empdResBlockedCI",
         "\\empdResExcess~\\empdResExcessCI"),
        ("EMPD, hardness", "\\empdHardBlockedLookup~\\empdHardBlockedLookupCI",
         "\\empdHardBlocked~\\empdHardBlockedCI",
         "\\empdHardExcess~\\empdHardExcessCI"),
        ("HTEM-DB, resistivity",
         "\\htemResBlockedLookup~\\htemResBlockedLookupCI",
         "\\htemResBlockedModel~\\htemResBlockedModelCI",
         "\\htemResExcess~\\htemResExcessCI"),
        ("HTEM-DB, band gap",
         "\\htemGapBlockedLookup~\\htemGapBlockedLookupCI",
         "\\htemGapBlockedModel~\\htemGapBlockedModelCI",
         "\\htemGapExcess~\\htemGapExcessCI"),
    ]

    out = ["% Generated from the committed result files; not hand-edited.",
           "\\textbf{(a) Under a random partition of the rows.}\\\\[3pt]",
           "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lrlrrl}",
           "\\hline",
           "dataset and property & rows & group & lookup $R^2$ & "
           "model $R^2$ & model $-$ lookup \\\\",
           "\\hline"]
    for label, rows, unit, lookup, model, margin, ci in random_rows:
        gap = f"{margin}~{ci}" if ci else margin
        out.append(f"{label} & {rows} & {unit} & {lookup} & {model} & {gap} \\\\")
    out += ["\\hline", "\\end{tabular*}",
            "",
            "\\vspace{10pt}",
            "",
            "\\textbf{(b) Under a blocked partition.}"
            "\\\\[3pt]",
            "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lrll}",
            "\\hline",
            "dataset and property & lookup $R^2$ & model $R^2$ & excess \\\\",
            "\\hline"]
    for label, lookup, model, excess in blocked_rows:
        out.append(f"{label} & {lookup} & {model} & {excess} \\\\")
    out += ["\\hline", "\\end{tabular*}", ""]
    return "\n".join(out)


def render_ablation_table() -> str:
    """The tuned models' table, tab:ablation.

    It takes one row per property and model, the row shape of tab:main, rather
    than one column pair per property: an interval beside every cell makes the
    wide form overrun the text block.
    """
    properties = (("Res", "resistivity"),
                  ("Gap", "band gap"))
    arms = [("\\htemTrees-tree forest, untuned",
             "RandomModel", "BlockedModel", "Excess", "ExcessCI"),
            ("\\htemTunedTrees-tree forest, tuned",
             "RandomTuned", "BlockedTuned", "ExcessTuned", "ExcessTunedCI"),
            ("gradient boosting, tuned",
             "RandomBoosted", "BlockedBoosted", "ExcessBoosted", "ExcessBoostedCI")]

    out = ["% Generated from the committed result files; not hand-edited.",
           "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}llrrl}", "\\hline",
           "property & model & random $R^2$ & blocked $R^2$ & excess \\\\", "\\hline"]
    for i, (prefix, label) in enumerate(properties):
        if i:
            out.append("\\hline")
        for j, (arm, random_, blocked, excess, ci) in enumerate(arms):
            out.append(f"{label if j == 0 else ''} & {arm} & "
                       f"\\htem{prefix}{random_} & \\htem{prefix}{blocked} & "
                       f"\\htem{prefix}{excess}~\\htem{prefix}{ci} \\\\")
    out += ["\\hline", "\\end{tabular*}", ""]
    return "\n".join(out)


def main() -> None:
    blocks, (npj, htem, margins) = build()
    PARAMETERS.parent.mkdir(parents=True, exist_ok=True)
    TABLE.parent.mkdir(parents=True, exist_ok=True)

    # Only the generated region is rewritten. The manual region is seeded once
    # and thereafter belongs to whoever edits it.
    existing = PARAMETERS.read_text() if PARAMETERS.exists() else MANUAL_SEED
    if BEGIN in existing and END in existing:
        head = existing[:existing.index(BEGIN)]
        tail = existing[existing.index(END) + len(END):]
    else:
        head, tail = existing, "\n"
    PARAMETERS.write_text(head + render_parameters(blocks) + tail)
    TABLE.write_text(render_table(npj, htem, margins))
    TABLE_ABLATION.write_text(render_ablation_table())
    total = sum(len(macros) for _, macros in blocks)
    print(f"wrote {PARAMETERS.relative_to(ROOT)} ({total} macros over "
          f"{len(blocks)} populations)")
    for path in (TABLE, TABLE_ABLATION):
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
