"""Every claim the manuscript makes, checked against the files it was measured on.

Run with `pytest test_assertions.py`. Nothing here loads synthetic data: each
assertion reads outputs/ or data/ and compares against the sentence in
paper/main.tex that the failure message quotes, by \\label slug, because nothing
dereferences a failure message.

Four axes, and only the first is the one a green suite usually covers:

  1. BINDINGS  -- each macro in paper/parameters.tex holds the number its
     artifact holds.
  2. QUANTIFIERS -- the words around those macros ("all five", "the widest",
     "larger than on any") range over sets derived from the artifacts, never
     from the members a sentence happens to name.
  3. JOINS AND POPULATIONS -- two macros printed in one sentence are measured
     over declared populations, and two declarations never resolve to one file.
  4. VERDICTS -- every criterion the runs committed reports its outcome,
     including the ones that missed.

Steps of the paper's arguments that no computation decides, each with the
reason:

  - par:margin, "near an R^2 of one, RMSE distinguishes the network from the
    lookup more clearly than R^2 does": a reading of two printed differences,
    a 15% fall in RMSE against a margin of +0.0022 in R^2; no statistic
    measures "more clearly".
  - sec:predictors, "this leakage can raise the network's score and not the
    lookup's": follows from which rows each rescaling reads, which the code
    shows and no run measures.
  - sec:predictors, "a margin narrower than the one in Table 1a would only
    strengthen the finding": an implication between two printed quantities,
    not a number.
  - sec:features, "a row's composition and thickness vary smoothly between
    adjacent positions of a library": EMPD's construction guideline states the
    interpolation that implies it; the release ships no measurement before
    interpolation to test it on.
  - par:partition, "a drop comes both from": a property of the design, which
    no run separates.
  - par:conclusion, "the first needs no training and the second needs only a
    model's predictions on groups excluded from its training": properties of
    the two checks' definitions.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
PAPER = ROOT / "paper"
PARAMETERS = PAPER / "parameters.tex"
MAIN = PAPER / "main.tex"
TABLE = PAPER / "tables" / "main.tex"


def documents() -> list:
    """Every document the submission ships: the article and its supplementary
    information. Derived from the presence of a preamble rather than listed, so
    a document added to paper/ is covered the day it is written. Both reach
    the reader, so both discharge a macro."""
    return sorted(p for p in PAPER.glob("*.tex")
                  if "\\documentclass" in p.read_text())


def document_text() -> str:
    return "".join(p.read_text() for p in documents())


def table_bodies() -> str:
    """Every table body those documents input, concatenated. Derived from the
    \\input lines rather than listed, so a table added to a document is covered
    the day it is included."""
    return "".join((PAPER / f"{rel}.tex").read_text() for rel in
                   re.findall(r"\\input\{(tables/[\w/-]+)\}", document_text()))

# Half a unit in the last printed digit: the tolerance a rounded macro owes.


def half_ulp(places: int) -> float:
    return 0.5 * 10 ** (-places)


R2_TOL = half_ulp(3)      # an R^2 with no interval, printed to three decimals
R2_TOL_FINE = half_ulp(4)  # ...except the npj row, printed to four


def printed_tol(printed: str) -> float:
    """Half a unit in the last place that a printed number shows. A value that
    has an interval prints to the places the interval resolves, so its
    tolerance is read from what is printed rather than fixed
    (sec:statistics)."""
    raw = printed.replace("{,}", "").replace(r"$-$", "-").replace(r"\%", "")
    return half_ulp(len(raw.split(".")[1]) if "." in raw else 0)


def interval_tol(printed: str) -> float:
    """printed_tol for an interval's endpoints, which print to one precision."""
    return printed_tol(printed.strip("[]").split(", ")[0])
RMSE_TOL = half_ulp(2)


@pytest.fixture(scope="session")


def npj():
    return json.loads((ROOT / "outputs" / "npj" / "split.json").read_text())


@pytest.fixture(scope="session")


def htem():
    return json.loads((ROOT / "outputs" / "htem" / "split.json").read_text())


@pytest.fixture(scope="session")


def ablation():
    return json.loads((ROOT / "outputs" / "htem" / "ablation.json").read_text())


@pytest.fixture(scope="session")


def margins():
    return json.loads((ROOT / "outputs" / "margins" / "results.json").read_text())


@pytest.fixture(scope="session")


def expt_gap():
    return json.loads((ROOT / "outputs" / "expt_gap" / "control.json").read_text())


@pytest.fixture(scope="session")


def structure():
    return json.loads((ROOT / "outputs" / "aihub" / "structure.json").read_text())


@pytest.fixture(scope="session")


def macros():
    """Every \\newcommand in parameters.tex, as name -> printed string."""
    text = PARAMETERS.read_text()
    out = {}
    for name, value in re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", text):
        out[name] = value
    return out


def number(macros, name: str) -> float:
    """The numeric value of a macro, undoing LaTeX's typography."""
    raw = macros[name]
    raw = raw.replace("{,}", "").replace(r"$-$", "-").replace(r"\%", "")
    return float(raw)


def endpoints(macros, name: str) -> tuple[float, float]:
    lo, hi = macros[name].strip("[]").split(", ")
    clean = lambda s: float(s.replace(r"$-$", "-"))
    return clean(lo), clean(hi)


# --------------------------------------------------------------------------
# 1. Bindings
# --------------------------------------------------------------------------


def test_npj_reproduces_the_published_network(npj, macros):
    """sec:predictors: 'We nevertheless keep this rescaling, because a margin
    narrower than the one in Table 1a would only strengthen the finding that
    the lookup reaches most of the network's R^2, and because with this rescaling our evaluation
    reproduces the RMSE that the article reports on each of the two files to
    within one unit of its last reported digit. The network's RMSE is 3.95 uOhm.cm on the training file and
    5.87 uOhm.cm on the validation file, compared with the article's 3.94 and
    5.86.'"""
    for macro, measured, cited, published in [
        ("npjModelTrainRmse", npj["model"]["train"]["rmse"],
         "npjPaperTrainRmse", npj["paper"]["train_rmse"]),
        ("npjModelRmse", npj["model"]["val"]["rmse"],
         "npjPaperRmse", npj["paper"]["val_rmse"]),
    ]:
        assert math.isclose(number(macros, macro), measured, abs_tol=RMSE_TOL), (
            f"Manuscript prints {macro}={macros[macro]} (sec:methods) "
            f"but outputs/npj/split.json holds {measured:.4f}"
        )
        assert number(macros, cited) == published
        # 'to within one unit of its last reported digit': the two reported
        # values are one step of their last decimal apart at most.
        assert abs(number(macros, macro) - published) <= 0.01 + 1e-9, (
            f"sec:predictors says our evaluation reproduces the article's RMSE "
            f"to within one unit of its last printed digit; {macros[macro]} against "
            f"{published}")
        # The claim is a reproduction, so the two must agree at the precision
        # the article states its own numbers to.
        assert abs(measured - published) < 0.05, (
            f"Manuscript claims the released weights reproduce Kim and Lee's "
            f"published error (sec:methods) but {measured:.4f} vs {published}"
        )


def test_npj_lookup_and_margin(npj, macros):
    """par:margin: 'The margin ranges from +0.0022 on Kim and Lee's dataset',
    and 'The margin on Kim and Lee's dataset separates the lookup's R^2 of
    0.9924 from their network's 0.9946 on their validation set, which is their
    test set.'
    sec:predictors gives the article's own RMSE beside the recomputed one, and
    both scores also round to the 0.99 that the article prints."""
    lookup = npj["lookup"]["all_features"]["r2"]
    model = npj["model"]["val"]["r2"]
    printed = npj["paper"]["val_r2_printed"]
    places = len(printed.split(".")[1])
    for name, value in (("lookup", lookup), ("network", model)):
        assert round(value, places) == float(printed), (
            f"par:margin says both scores round to the article's {printed}; "
            f"the {name} scores {value:.4f}")
    assert math.isclose(number(macros, "npjLookupRSq"), lookup, abs_tol=R2_TOL_FINE)
    assert math.isclose(number(macros, "npjModelRSq"), model, abs_tol=R2_TOL_FINE)
    assert math.isclose(number(macros, "npjMargin"), model - lookup,
                        abs_tol=R2_TOL_FINE), (
        "Manuscript claims a margin of +0.0022 in R^2 on Kim and Lee's "
        f"validation set (par:margin) but the data gives {model - lookup:.4f}"
    )


def test_npj_rmse_values_are_the_ones_the_run_produced(npj, macros):
    """sec:esi-settings: 'On those rows, the network's RMSE is 5.87 uOhm.cm and
    the lookup's 6.94 uOhm.cm, so, near an R^2 of one, RMSE distinguishes the
    network from the lookup more clearly than R^2 does', and par:margin: 'On
    those rows, RMSE separates the two more clearly'. Both values are bound, and the
    network's must be the lower one for "distinguishes" to favor it."""
    lookup = npj["lookup"]["all_features"]["rmse"]
    model = npj["model"]["val"]["rmse"]
    for macro, value in (("npjLookupRmse", lookup), ("npjModelRmse", model)):
        assert math.isclose(number(macros, macro), value, abs_tol=half_ulp(2)), (
            f"sec:esi-settings prints \\{macro}={macros[macro]}; the run gives {value:.4f}")
    assert model < lookup, (
        f"sec:esi-settings sets the network's RMSE below the lookup's; {model:.3f} "
        f"against {lookup:.3f}")


def test_the_control_scores_below_every_combinatorial_lookup(
        expt_gap, npj, htem, margins):
    """par:margin: 'On Matbench, the lookup's mean R^2 is lower and the mean
    margin wider than on any of the five pairs in Table 1a.' The interval form
    of the lookup's half is pinned in test_e1 below."""
    score = expt_gap["random"]["lookup_r2"]
    for name, value in random_lookups(npj, htem, margins).items():
        assert score < value, (
            f"par:margin claims the lookup's R^2 is lower on Matbench than on "
            f"any of the five pairs, but it is {score:.4f} against {name}'s "
            f"{value:.4f}"
        )


@pytest.mark.parametrize("prefix,target", [("htemRes", "log10_resistivity"),
                                           ("htemGap", "direct_band_gap")])


def test_htem_cells(htem, macros, prefix, target):
    """tab:main, both panels, the HTEM-DB rows."""
    t = htem["targets"][target]
    pairs = [
        (f"{prefix}Rows", t["rows"]),
        (f"{prefix}Libraries", t["libraries"]),
        (f"{prefix}RandomLookup", t["random"]["lookup_r2"]),
        (f"{prefix}RandomModel", t["random"]["model_r2"]),
        (f"{prefix}BlockedLookup", t["blocked"]["lookup_r2"]),
        (f"{prefix}BlockedModel", t["blocked"]["model_r2"]),
        (f"{prefix}Margin", t["model_margin"]),
        (f"{prefix}Excess", t["drop_minus_margin"]),
    ]
    for macro, value in pairs:
        tol = 0 if macro.endswith(("Rows", "Libraries")) else printed_tol(macros[macro])
        assert math.isclose(number(macros, macro), value, abs_tol=tol), (
            f"Manuscript prints {macro}={macros[macro]} (tab:main) "
            f"but outputs/htem/split.json holds {value}"
        )
    for macro, key in [(f"{prefix}ExcessCI", "drop_minus_margin"),
                       (f"{prefix}MarginCI", "model_margin")]:
        lo, hi = endpoints(macros, macro)
        tol = interval_tol(macros[macro])
        assert math.isclose(lo, t["ci"][key]["lo"], abs_tol=tol)
        assert math.isclose(hi, t["ci"][key]["hi"], abs_tol=tol)


def test_the_margin_ranges_are_the_partitions_own(htem, expt_gap, margins, macros):
    """sec:esi-draw-spread: 'The margin varies less across the random
    partitions on the other pairs of EMPD and HTEM-DB and on Matbench, from +0.053 to +0.064 on
    EMPD resistivity, from +0.051 to +0.064 on HTEM-DB resistivity, from +0.114
    to +0.130 on HTEM-DB band gap and from +0.193 to +0.267 on Matbench', less
    than on EMPD hardness, whose partitions give margins from +0.017 to +0.103.
    Each endpoint is derived from the per-partition margins the run recorded,
    never from the two a sentence names, and so is the width that "less"
    compares."""
    aihub = margins["datasets"]["aihub"]["tasks"]
    per_partition = {
        "empdRes": aihub["resistivity"]["margin_per_draw"],
        "htemRes": [p["margin"] for p in
                    htem["targets"]["log10_resistivity"]["random_partitions"]],
        "htemGap": [p["margin"] for p in
                    htem["targets"]["direct_band_gap"]["random_partitions"]],
        "exptGap": [p["margin"] for p in expt_gap["random_partitions"]],
    }
    for prefix, drawn in per_partition.items():
        for macro, value in ((f"{prefix}MarginMin", min(drawn)),
                             (f"{prefix}MarginMax", max(drawn))):
            assert math.isclose(number(macros, macro), value, abs_tol=R2_TOL), (
                f"sec:esi-draw-spread prints \\{macro}={macros[macro]}; the "
                f"partitions give {value:+.4f}"
            )
    hardness = aihub["hardness"]["margin_per_draw"]
    for prefix, drawn in per_partition.items():
        assert max(drawn) - min(drawn) < max(hardness) - min(hardness), (
            f"sec:esi-draw-spread says the margin varies less on {prefix} than "
            f"on EMPD hardness, but its partitions span {max(drawn) - min(drawn):.4f} "
            f"against {max(hardness) - min(hardness):.4f}"
        )


def test_the_first_partition_is_where_those_two_margins_are_closest(
        htem, expt_gap, macros):
    """sec:esi-draw-spread: 'The first partition gives Matbench its smallest
    margin and HTEM-DB band gap one of its two largest, which explains why the two
    intervals overlap on it but not over the means.' Both ranks range over the
    per-partition margins, so a reordering of the seeds moves them. The
    sentence explains why par:margin's two comparisons disagree, which test_e2
    pins."""
    control = [p["margin"] for p in expt_gap["random_partitions"]]
    gap = [p["margin"] for p in htem["targets"]["direct_band_gap"]["random_partitions"]]
    assert control[0] == min(control), (
        f"sec:esi-draw-spread calls the first partition Matbench's smallest "
        f"margin; it is {control[0]:+.4f} against a smallest of {min(control):+.4f}"
    )
    assert sorted(gap, reverse=True).index(gap[0]) <= 1, (
        f"sec:esi-draw-spread calls the first partition's margin one of HTEM-DB "
        f"band gap's two largest; it is {gap[0]:+.4f} in "
        f"{[round(m, 4) for m in sorted(gap, reverse=True)]}"
    )
    for macro, value, interval in (
            ("exptGapFirstMargin", control[0], expt_gap["ci"]["first_partition_margin"]),
            ("htemGapFirstMargin", gap[0],
             htem["targets"]["direct_band_gap"]["ci"]["first_partition_margin"])):
        assert math.isclose(number(macros, macro), value,
                            abs_tol=printed_tol(macros[macro])), (
            f"sec:esi-draw-spread prints \\{macro}={macros[macro]}; the first "
            f"partition gives {value:+.4f}"
        )
        lo, hi = endpoints(macros, f"{macro}CI")
        tol = interval_tol(macros[f"{macro}CI"])
        assert math.isclose(lo, interval["lo"], abs_tol=tol)
        assert math.isclose(hi, interval["hi"], abs_tol=tol)


@pytest.mark.parametrize("prefix,task", [("empdRes", "resistivity"),
                                         ("empdHard", "hardness")])


def test_aihub_cells(margins, macros, prefix, task):
    """tab:main, both panels, the EMPD rows."""
    t = margins["datasets"]["aihub"]["tasks"][task]
    for macro, value in [
        (f"{prefix}Rows", t["rows"]),
        (f"{prefix}Systems", t["systems"]),
        (f"{prefix}Lookup", t["lookup_r2"]["mean"]),
        (f"{prefix}Model", t["model_r2"]["mean"]),
        (f"{prefix}Margin", t["margin"]["mean"]),
        (f"{prefix}BlockedLookup", t["blocked_lookup_r2"]["mean"]),
        (f"{prefix}Blocked", t["blocked_model_r2"]["mean"]),
        (f"{prefix}Excess", t["drop_minus_margin"]["mean"]),
        (f"{prefix}MarginMax", t["margin_max"]),
    ]:
        tol = 0 if macro.endswith(("Rows", "Systems")) else printed_tol(macros[macro])
        assert math.isclose(number(macros, macro), value, abs_tol=tol), (
            f"Manuscript prints {macro}={macros[macro]} (tab:main) "
            f"but outputs/margins/results.json holds {value}"
        )
    for macro, key in [(f"{prefix}MarginCI", "margin"),
                       (f"{prefix}BlockedLookupCI", "blocked_lookup_r2"),
                       (f"{prefix}BlockedCI", "blocked_model_r2"),
                       (f"{prefix}ExcessCI", "drop_minus_margin")]:
        lo, hi = endpoints(macros, macro)
        tol = interval_tol(macros[macro])
        assert math.isclose(lo, t[key]["ci_lo"], abs_tol=tol)
        assert math.isclose(hi, t[key]["ci_hi"], abs_tol=tol)


def test_expt_gap_control_cells(expt_gap, htem, macros):
    """tab:main, the control row, and the sec:esi-draw-spread sentences
    that set its lookup interval beside HTEM-DB band gap's."""
    t = expt_gap
    for macro, value in [
        ("exptGapRows", t["rows"]),
        ("exptGapLookup", t["random"]["lookup_r2"]),
        ("exptGapModel", t["random"]["model_r2"]),
        ("exptGapMargin", t["model_margin"]),
        
    ]:
        tol = 0 if macro.endswith(("Rows", "Systems")) else printed_tol(macros[macro])
        assert math.isclose(number(macros, macro), value, abs_tol=tol), (
            f"Manuscript prints {macro}={macros[macro]} (tab:main, par:margin) "
            f"but outputs/expt_gap/control.json holds {value}"
        )
    gap = htem["targets"]["direct_band_gap"]
    res = htem["targets"]["log10_resistivity"]
    for macro, d in [("exptGapLookupCI", t["ci"]["random_lookup_r2"]),
                     ("exptGapMarginCI", t["ci"]["model_margin"]),
                     ("htemGapRandomLookupCI", gap["ci"]["random_lookup_r2"]),
                     ("htemGapBlockedLookupCI", gap["ci"]["blocked_lookup_r2"]),
                     ("htemResBlockedLookupCI", res["ci"]["blocked_lookup_r2"]),
                     ("htemGapBlockedModelCI", gap["ci"]["blocked_model_r2"]),
                     ("htemResBlockedModelCI", res["ci"]["blocked_model_r2"])]:
        lo, hi = endpoints(macros, macro)
        tol = interval_tol(macros[macro])
        assert math.isclose(lo, d["lo"], abs_tol=tol)
        assert math.isclose(hi, d["hi"], abs_tol=tol)


def test_expt_gap_rows_are_compounds_not_positions(expt_gap):
    """sec:datasets: the Matbench task's rows 'are compounds with a measured
    band gap', and 'The two control datasets are not sets of libraries.' The
    structure is what makes it a control, so the
    structure
    is asserted: the median chemical system holds one row, where a library
    holds tens of positions."""
    assert expt_gap["median_rows_per_system"] == 1, (
        "sec:datasets describes the control's rows as compounds rather "
        f"than positions on a library; the median system holds "
        f"{expt_gap['median_rows_per_system']} rows"
    )


def test_expt_gap_protocol_is_htem_protocol(expt_gap, htem):
    """sec:protocol: 'Matbench matches HTEM-DB band gap in property, model
    and partition.' The sentence rests on an import in
    reproduce/expt_gap_control.py; the constants it inherited are checked
    against HTEM-DB's own file."""
    p = expt_gap["protocol"]
    assert p["from"] == "reproduce/htem_split.py"
    for key, value in [("n_folds", htem["n_folds"]),
                       ("fold_seeds", htem["fold_seeds"]),
                       ("rf_trees", htem["rf_trees"])]:
        assert p[key] == value, (
            f"sec:protocol says the control matches HTEM-DB's model and "
            f"partition, "
            f"but {key} is {p[key]} against HTEM-DB's {value}"
        )
    for key in ("replicates", "alpha"):
        assert p["bootstrap"][key] == htem["bootstrap"][key]


def test_a_single_partition_margin_can_exceed_twice_the_mean(
        margins, htem, expt_gap, macros):
    """sec:esi-draw-spread: 'On EMPD hardness, that mean is +0.048 and the
    partitions give margins from +0.017 to +0.103, so one partition's margin
    can be more than twice the mean.' The same section prints both properties' ranges."""
    seeds = {"EMPD": margins["datasets"]["aihub"]["draw_seeds"],
             "HTEM-DB": htem["fold_seeds"],
             "Matbench": expt_gap["protocol"]["fold_seeds"]}
    for name, drawn in seeds.items():
        assert WORDS[len(drawn)] == macros["splitDraws"], (
            f"sec:esi-draw-spread says the main text reports {name}'s mean over "
            f"{macros['splitDraws']} partitions; its record carries {len(drawn)}")
    for prefix, task in [("empdRes", "resistivity"), ("empdHard", "hardness")]:
        t = margins["datasets"]["aihub"]["tasks"][task]
        for macro, value in [(f"{prefix}MarginMin", t["margin_min"]),
                             (f"{prefix}MarginMax", t["margin_max"])]:
            assert math.isclose(number(macros, macro), value, abs_tol=R2_TOL), (
                f"sec:esi-draw-spread prints {macro}={macros[macro]} but the "
                f"per-draw record holds {value:+.4f}"
            )
    hard = margins["datasets"]["aihub"]["tasks"]["hardness"]
    assert math.isclose(number(macros, "empdHardMargin"), hard["margin"]["mean"],
                        abs_tol=printed_tol(macros["empdHardMargin"])), (
        "sec:esi-draw-spread calls the hardness mean over the partitions the value in "
        f"Table 1a, \\empdHardMargin={macros['empdHardMargin']}, but the "
        f"per-draw record averages {hard['margin']['mean']:+.4f}"
    )
    mean = hard["margin"]["mean"]
    assert mean > 0 and hard["margin_max"] > 2 * mean, (
        "sec:esi-draw-spread says a margin measured on one random partition can be more "
        f"than twice the mean over ten partitions, but the hardness mean is "
        f"{mean:+.4f} and the largest partition margin {hard['margin_max']:+.4f}"
    )


def test_aihub_structure_matches_the_archive(structure, margins, macros):
    """tab:main and sec:datasets print the alloy-system counts. They are recovered from the
    composition column by reproduce/aihub_comeasurement.py, which reads the
    archive, and must agree with the counts reproduce/empd_draws.py recorded
    in outputs/aihub/draws.json, which a clone reads instead."""
    extract = margins["datasets"]["aihub"]
    for key in ("rows", "systems"):
        assert structure[key] == extract[key], (
            f"aihub_comeasurement.py measured {key}={structure[key]} on the "
            f"archive; outputs/aihub/draws.json holds {extract[key]}"
        )
    assert number(macros, "empdAmorphousRows") == structure["rows"]["amorphous"]


def test_empd_counts_cover_every_split_of_the_archive(margins, structure):
    """sec:datasets prints EMPD's counts as what the release records: 'EMPD
    records resistivity at 58,937 positions on libraries of 25 alloy systems',
    and
    tab:main's rows column prints them as each dataset's size.
    reproduce/aihub_comeasurement.py counts rows and alloy systems over every
    split the archive ships, apart from the loader, so a loader that reads one
    split, or counts a misspelt element symbol as a system of its own, fails
    here."""
    census = structure["archive"]
    extract = margins["datasets"]["aihub"]
    for key in ("rows", "systems"):
        assert extract[key] == census[key], (
            f"the archive's {' and '.join(census['splits'])} splits hold {key} "
            f"{census[key]}; outputs/aihub/draws.json, which the manuscript's EMPD row "
            f"and system counts are read from, carries {extract[key]}")


WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
         "nine", "ten")


def test_empd_tables_name_no_library_but_its_raw_files_hold_one_each(structure):
    """par:partition: 'EMPD's tables record no library, so its blocked
    partitions exclude whole alloy systems from training'. sec:datasets:
    'Unlike the tables, these raw measurements are filed by library. Each
    library has a file of its measured resistances.' The raw files are the
    library record that the tables drop, so no sentence may say that EMPD
    records no library."""
    files = structure["release"]["library_files"]
    assert "EMPD records no library" not in document_text(), (
        "the manuscript says EMPD records no library; its release keeps "
        f"{files['resistance_files']} raw resistance files, one per library")
    assert files["resistance_files"] == files["resistance_libraries"], (
        "sec:datasets says the release keeps each library's resistance in "
        f"a file of its own; {files['resistance_files']} files carry "
        f"{files['resistance_libraries']} sample numbers")
    assert files["resistance_libraries_with_eds_and_thickness_files"] == files[
        "resistance_libraries"], (
        "sec:datasets says each library's resistance, composition and "
        f"thickness are in separate files; of {files['resistance_libraries']} "
        "libraries with a resistance file, "
        f"{files['resistance_libraries_with_eds_and_thickness_files']} also "
        "keep a composition and a thickness file")


def test_empd_release_splits_each_table_in_row_order(structure, macros):
    """sec:datasets: 'The release splits each table in row order into a
    training part and a validation part, which together contain the rows counted
    above, and withholds a third part for testing.' sec:esi-settings: 'EMPD's
    release splits each table in row order, so a model trained on the training
    part is tested partly on alloy systems excluded from that training, because
    the validation part alone contains one of the resistivity alloy systems and
    four of the hardness systems', and sec:protocol: 'so we pool the training
    and validation parts that the release publishes and draw EMPD's random
    partitions ourselves too.' Table 1 counts the pooled rows. 'Each table' ranges over the three tables
    the archive ships, so the order is asserted on all three."""
    assert "neither dataset publishes one" not in document_text(), (
        "sec:protocol says EMPD publishes no partition; its release ships a "
        "training part and a validation part")
    order = structure["release"]["split_order"]
    for task, entry in order.items():
        parts = entry["splits"]
        assert set(parts) == {"Training", "Validation"}, (
            f"sec:datasets names a training part and a validation part; EMPD's "
            f"{task} table ships {sorted(parts)}")
        assert parts["Training"]["numbers"][1] < parts["Validation"]["numbers"][0], (
            f"sec:datasets says EMPD's {task} table is split in row order; the "
            f"training part's numbers run {parts['Training']['numbers']} and "
            f"the validation part's {parts['Validation']['numbers']}")
    for prefix, task in [("empdRes", "resistivity"), ("empdHard", "hardness")]:
        parts = order[task]["splits"]
        pooled = parts["Training"]["rows"] + parts["Validation"]["rows"]
        assert number(macros, f"{prefix}Rows") == pooled, (
            f"sec:datasets partitions the training and validation parts, which "
            f"hold {pooled:,} {task} rows, but Table 1 prints \\{prefix}Rows="
            f"{macros[f'{prefix}Rows']}")
        only = order[task]["systems_in_one_split"]["Validation"]
        assert macros[f"{prefix}ValidationOnlySystems"] == WORDS[only], (
            f"sec:esi-settings says the validation part alone contains "
            f"{macros[f'{prefix}ValidationOnlySystems']} {task} systems; the "
            f"archive has {only}")
        # 'so a model trained on the training part is tested partly on alloy
        # systems excluded from its training'.
        assert only > 0, (
            f"sec:datasets says the validation part tests a model partly on "
            f"{task} systems excluded from its training; it holds none alone")


# The English name EMPD's own AI Hub description gives the release, which the
# abbreviation abbreviates.
EMPD_NAME = "Experimental Materials Property Database"


def test_empd_is_named_as_its_release_names_itself():
    """par:intro names EMPD by the English name that its AI
    Hub description gives, and the bibliography entry carries it as the title.
    A translation of the Korean title is a second name for one object."""
    text = document_text()
    assert EMPD_NAME in text, (
        f"the manuscript never names EMPD as {EMPD_NAME!r}, the name the "
        "abbreviation stands for")
    assert "experiment-based material property data" not in text.lower(), (
        "the manuscript still names EMPD by a translation of its Korean title")
    entry = re.search(r"@misc\{aihubMaterials,.*?\n\}",
                      (PAPER / "refs.bib").read_text(), re.S).group(0)
    title = re.search(r"\btitle\s*=\s*\{(.*?)\},\n", entry, re.S).group(1)
    assert title == EMPD_NAME, (
        f"refs.bib titles aihubMaterials {title!r}; the release calls itself "
        f"{EMPD_NAME!r}")


def test_empd_and_kim_and_lee_share_measurements(structure, macros):
    """sec:datasets: 'EMPD and Kim and Lee's dataset share measurements. ...
    We compared the quotient with the raw EMPD resistances that are recorded with six or
    more decimals and that occur at only one position in the release ... Kim
    and Lee's resistivity, divided by their thickness and by 0.45326,
    reproduces such a resistance to its last recorded decimal on 2,405 of their
    rows, and no other divisor near it reproduces as many ... The 2,405 rows
    therefore lie on
    libraries that both datasets contain, so we count the two datasets as one
    source of libraries.' sec:esi-settings: 'No divisor in a range around
    0.45326, from 0.38 to 0.52 in steps of 0.00001, reproduces more than 157 of
    Kim and Lee's rows, apart from 0.45326 itself ... EMPD's raw files record a
    thickness at the positions of 2,289 of the 2,405 rows that it reproduces,
    where the median ratio of Kim and Lee's thickness to EMPD's is 1.000.' The
    2,289 are counted from the rows whose ratio the
    scan computed, so the median and its population come from one set."""
    text = document_text()
    for stale in ("may share measurements", "share measurements is not established"):
        assert stale not in text, (
            f"the manuscript says EMPD and Kim and Lee's dataset {stale}; "
            "outputs/aihub/structure.json records the rows they share")
    shared = structure["npj_shared_measurements"]
    lo, hi, step = shared["factor_grid"]
    for macro, value, tol in [
            ("npjSharedRows", shared["rows_at_best_factor"], 0),
            ("npjSharedOtherRows", shared["most_rows_at_another_factor"], 0),
            ("npjSharedFactor", shared["best_factor"], half_ulp(5)),
            ("npjSharedThicknessRatio",
             shared["thickness_ratio_npj_to_empd"]["median"], half_ulp(3)),
            ("npjSharedThicknessRows", shared["rows_with_an_empd_thickness"], 0),
            ("npjFactorGridLow", lo, half_ulp(2)),
            ("npjFactorGridHigh", hi, half_ulp(2)),
            ("npjFactorGridStep", step, half_ulp(5) / 10)]:
        assert math.isclose(number(macros, macro), value, abs_tol=tol), (
            f"sec:datasets prints \\{macro}={macros[macro]}; "
            f"outputs/aihub/structure.json records {value}")
    assert macros["npjSharedMinDecimals"] == WORDS[shared["min_decimals"]], (
        f"sec:datasets says raw values with {macros['npjSharedMinDecimals']} "
        f"or more decimals are compared; the scan used {shared['min_decimals']}")
    assert 0 < shared["rows_with_an_empd_thickness"] <= shared["rows_at_best_factor"], (
        "sec:datasets takes the thickness ratio over part of the reproduced rows")
    assert shared["rows_at_best_factor"] > shared["most_rows_at_another_factor"], (
        "sec:datasets sets the matches at one factor against the most at "
        "any other, and the one factor must reproduce more rows")


def test_empd_columns_beside_the_targets_are_not_features(structure):
    """sec:features: 'On EMPD, the features are the atomic percent of each of
    15 elements, the film thickness and the six descriptors that EMPD's tables
    contain', so the resistance and the elastic modulus that the tables also
    record are no feature. sec:datasets: 'We therefore compared the
    two datasets' resistivity values through the measured resistance, which EMPD multiplies
    by the film's thickness and by a conversion factor to give each resistivity
    value.'"""
    features = json.loads((ROOT / "outputs" / "aihub" / "draws.json").read_text())["features"]
    used = {name for group in features.values() for name in group}
    for task, names in {"resistivity": ["Resistance"],
                        "hardness": ["Modulus"]}.items():
        for name in names:
            assert name not in used, (
                f"sec:features lists EMPD's features without the {task} "
                f"table's {name}; outputs/aihub/draws.json lists it among them")
    factor = structure["release"]["resistance_factor"]
    # "A conversion factor": one factor for every row, the largest deviation
    # over the rows held to one part in a million, far inside the digits the
    # table prints for each column.
    assert factor["max_relative_deviation"] < 1e-6, (
        "sec:datasets says one conversion factor and the film's thickness "
        "turn the measured resistance into resistivity; one row departs by "
        f"{factor['max_relative_deviation']:.2e}")


def test_empd_interpolation_is_cited_to_its_guideline():
    """sec:features: 'EMPD's construction guideline states that the
    composition and the thickness at each measured position are interpolated
    linearly from the library's own composition and thickness measurements, so
    a row's composition and thickness vary smoothly between adjacent positions
    of a library. A test row's nearest training row in the features that we
    use on EMPD can therefore be an adjacent position of its own library, whose
    target the lookup copies.' The features are therefore not measured at every position, and the
    sentence says whose statement that is, naming the guideline and citing
    it."""
    text = " ".join(MAIN.read_text().split())
    assert re.search(r"construction guideline[^.]*interpolated[^.]*"
                     r"\\cite\{aihubGuideline\}", text), (
        "sec:features does not state, naming and citing EMPD's construction "
        "guideline, that composition and thickness are interpolated")


def population_of(macro: str) -> str | None:
    """The population block a macro is declared under, or None."""
    current = None
    for line in PARAMETERS.read_text().splitlines():
        marker = re.match(r"%% POPULATION (\w+) \|", line)
        if marker:
            current = marker.group(1)
        elif re.match(rf"\\newcommand\{{\\{macro}\}}", line):
            return current
    return None


def test_the_npj_release_carries_no_system_identifier(npj, structure, macros):
    """sec:datasets: 'Their article groups the rows into alloy systems, which
    neither file records', and
    tab:main's '43 systems, not recorded'. sec:features: 'Of Kim and Lee's nine
    features, six are descriptors computed from composition alone.' Three facts carry
    those sentences. Their release ships the features their network consumes and the target, and
    nothing else; its composition columns are descriptors, transcribed here from
    their notebook's feature list; and the system count is declared over their
    article rather than over any artifact here. EMPD is the contrast, and its
    systems come from a column that is there."""
    spare = [c for c in npj["released_columns"] if c not in npj["features"]]
    assert len(spare) == 1, (
        f"par:intro says Kim and Lee's released files record neither a row's "
        f"library nor its alloy system: beyond "
        f"the {len(npj['features'])} features their network consumes their "
        f"release should hold only the target, and it holds {spare}"
    )
    descriptors = {"r", "del_r", "del_EN", "S", "H", "VEC"}
    assert set(npj["composition_features"]) == descriptors, (
        "sec:features says Kim and Lee's composition features are "
        f"descriptors computed from composition; they are "
        f"{npj['composition_features']}"
    )
    artifact, _unit = declared_populations()[population_of("npjSystems")]
    assert artifact.startswith("npj Comput"), (
        f"sec:datasets credits the alloy-system grouping to Kim and Lee's "
        f"article, so \\npjSystems is a quotation from it; it is declared "
        f"over {artifact!r}"
    )
    assert min(structure["systems"].values()) > 0, (
        "par:intro contrasts that release with EMPD's, whose composition "
        "column gives each row's alloy system; outputs/aihub/structure.json "
        "now carries no systems, so the contrast has only one side"
    )


def test_each_random_partition_is_attributed_to_whoever_drew_it(
        npj, htem, margins, macros):
    """sec:protocol: 'On Kim and Lee's dataset we use the random partition that
    their training and validation files define. HTEM-DB publishes no
    partition, so we draw its random partitions ourselves.' The artifacts
    separate
    the two: a partition drawn here
    leaves behind the seeds or folds that drew it, and a published one leaves
    the share its own article states."""
    words = {5: "five", 10: "ten"}
    assert not {"draw_seeds", "n_folds"} & set(npj), (
        "sec:methods attributes Kim and Lee's random partition to their "
        "repository; outputs/npj/split.json now carries a draw of our own"
    )
    assert 0.099 < npj["val_share"] < 0.101, (
        f"sec:protocol calls Kim and Lee's partition the one their repository "
        f"publishes, which their article describes as 10% held out; the "
        f"files hold {npj['val_share']:.4f}"
    )
    seeds = margins["datasets"]["aihub"]["draw_seeds"]
    assert macros["splitDraws"] == words[len(seeds)], (
        f"sec:methods says the EMPD random partition is drawn here over "
        f"{macros['splitDraws']} draws; the artifact carries {len(seeds)}"
    )
    assert macros["htemFolds"] == words[htem["n_folds"]], (
        f"sec:methods says the HTEM-DB random partition is drawn here over "
        f"{macros['htemFolds']} folds; the artifact carries {htem['n_folds']}"
    )
    assert htem["fold_seeds"] == seeds, (
        f"sec:protocol says HTEM-DB's random partitions are drawn from the same "
        f"seeds as EMPD's, which is what puts both under \\splitDraws; HTEM-DB "
        f"carries {htem['fold_seeds']} against EMPD's {seeds}"
    )


# --------------------------------------------------------------------------
# 2. Quantifiers, over sets derived from the artifacts
# --------------------------------------------------------------------------


def random_margins(npj, htem, margins):
    """The set every 'all five' in the manuscript ranges over, derived from the
    artifacts rather than from the pairs a sentence names."""
    aihub = margins["datasets"]["aihub"]["tasks"]
    return {
        "Kim and Lee's dataset": (npj["model"]["val"]["r2"]
                                 - npj["lookup"]["all_features"]["r2"]),
        "EMPD resistivity": aihub["resistivity"]["margin"]["mean"],
        "EMPD hardness": aihub["hardness"]["margin"]["mean"],
        "HTEM-DB resistivity": htem["targets"]["log10_resistivity"]["model_margin"],
        "HTEM-DB band gap": htem["targets"]["direct_band_gap"]["model_margin"],
    }


def random_lookups(npj, htem, margins):
    """The random-partition lookup score of every combinatorial pair: the set
    'lower than on any of the five pairs' in par:margin ranges over."""
    aihub = margins["datasets"]["aihub"]["tasks"]
    return {
        "Kim and Lee's dataset": npj["lookup"]["all_features"]["r2"],
        "EMPD resistivity": aihub["resistivity"]["lookup_r2"]["mean"],
        "EMPD hardness": aihub["hardness"]["lookup_r2"]["mean"],
        "HTEM-DB resistivity": htem["targets"]["log10_resistivity"]["random"]["lookup_r2"],
        "HTEM-DB band gap": htem["targets"]["direct_band_gap"]["random"]["lookup_r2"],
    }


def test_each_table_one_score_is_a_mean_over_its_partitions(npj, htem, margins,
                                                           expt_gap, macros):
    """tab:main: 'Each $R^2$ is a mean over ten partitions, except that Kim and
    Lee's files define one partition and we use one blocked partition of
    HTEM-DB.' Each score of Table 1 is read off its per-partition records, so a
    score that a single partition gives, or a partition count other than
    \\splitDraws, fails here."""
    words = {1: "one", 5: "five", 10: "ten"}
    draws = {v: k for k, v in words.items()}[macros["splitDraws"]]
    extract = lambda name: json.loads(
        (ROOT / "outputs" / name / "draws.json").read_text())
    aihub = extract("aihub")
    # A partition's score on EMPD is the mean over its inits.
    mean = lambda values: sum(values) / len(values)
    scores = {}
    for task in ("resistivity", "hardness"):
        summary = margins["datasets"]["aihub"]["tasks"][task]
        per_draw = [draw[task] for draw in aihub["draws"].values()]
        for name, key, per_init in (("lookup", "lookup_r2", None),
                                    ("model", "model_r2", "model_r2_per_init")):
            for partition, prefix in (("random", ""), ("blocked", "blocked_")):
                scores[f"EMPD {task}, {partition} {name}"] = (
                    summary[prefix + key]["mean"],
                    [mean(d[prefix + per_init]) if per_init else d[prefix + key]
                     for d in per_draw])
    for name, record in (("HTEM-DB resistivity", htem["targets"]["log10_resistivity"]),
                         ("HTEM-DB band gap", htem["targets"]["direct_band_gap"]),
                         ("Matbench", expt_gap)):
        for key in ("lookup_r2", "model_r2"):
            scores[f"{name}, random {key[:-3]}"] = (
                record["random"][key],
                [p[key] for p in record["random_partitions"]])
    for score, (value, per_partition) in scores.items():
        assert len(per_partition) == draws, (
            f"tab:main says each R^2 is a mean over {macros['splitDraws']} "
            f"partitions; {score} has {len(per_partition)}"
        )
        assert math.isclose(value, mean(per_partition), abs_tol=1e-12), (
            f"tab:main says each R^2 is a mean over its partitions; {score} is "
            f"{value}, against a mean of {mean(per_partition)}"
        )
    assert not {"draw_seeds", "n_folds", "random_partitions"} & set(npj), (
        "tab:main says Kim and Lee's files define one partition; "
        "outputs/npj/split.json carries more than one"
    )
    for name in ("log10_resistivity", "direct_band_gap"):
        target = htem["targets"][name]
        assert all(isinstance(v, float) for v in target["blocked"].values()) and not (
                {k for k in target if k.startswith("blocked")} - {"blocked"}), (
            f"tab:main says we use one blocked partition of HTEM-DB; {name} "
            f"carries {target['blocked']}"
        )


def test_random_pair_count(npj, htem, margins, macros):
    """par:margin: 'Each check is computed per pair of a dataset and a measured
    property. The pairs are resistivity on each dataset, hardness on EMPD and
    band gap on HTEM-DB, five pairs in all' and 'on all five pairs'."""
    words = {4: "four", 5: "five", 6: "six"}
    pairs = random_margins(npj, htem, margins)
    assert macros["randomPairs"] == words[len(pairs)], (
        f"Manuscript says {macros['randomPairs']} random-partition pairs "
        f"(par:intro) but the artifacts carry {len(pairs)}"
    )


def test_library_dataset_count(npj, htem, margins, macros):
    """abstract: 'on three datasets'; par:margin: 'the dataset that Kim and Lee
    released, three combinatorial datasets'.

    The dataset count is derived over the same pair set as the pair count, so
    a dataset added to the results is a failure here rather than a stale word
    in the abstract."""
    dataset_of = {
        "Kim and Lee's dataset": "Kim and Lee's dataset",
        "EMPD resistivity": "EMPD",
        "EMPD hardness": "EMPD",
        "HTEM-DB resistivity": "HTEM-DB",
        "HTEM-DB band gap": "HTEM-DB",
    }
    pairs = set(random_margins(npj, htem, margins))
    assert set(dataset_of) == pairs, (
        "the abstract and par:intro count the datasets that the pairs come "
        f"from; the pairs the artifacts carry and the pairs mapped to a dataset "
        f"differ by {pairs ^ set(dataset_of)}"
    )
    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
    datasets = {dataset_of[pair] for pair in pairs}
    assert macros["libraryDatasets"] == words[len(datasets)], (
        f"Manuscript says the lookup reaches most of the model's R^2 on "
        f"{macros['libraryDatasets']} combinatorial datasets (abstract) but the "
        f"pairs come from {len(datasets)}"
    )


def test_every_random_pair_has_a_program_and_they_number_two(npj, htem, margins,
                                                             macros):
    """par:partition: 'The five pairs come from only two sources of libraries,
    because EMPD and Kim and Lee's dataset share measurements (see Methods
    section).'

    The pair count is derived from the results and the program count from the
    map below, so the two are printed in one sentence and only one of them
    moves with the artifacts unless this check ties them. Requiring the map to
    cover the derived pair set exactly is what makes a sixth pair a failure
    here rather than a stale word in the abstract.

    The attribution itself is not in any result file. EMPD's release and Kim
    and Lee's dataset are counted as one source of libraries because they share
    measurements (sec:datasets), which the resistances that Kim and Lee's rows
    reproduce show.
    """
    program_of = {
        "Kim and Lee's dataset": "Kairos Lab, SKKU and Yonsei",
        "EMPD resistivity": "Kairos Lab, SKKU and Yonsei",
        "EMPD hardness": "Kairos Lab, SKKU and Yonsei",
        "HTEM-DB resistivity": "NREL",
        "HTEM-DB band gap": "NREL",
    }
    pairs = set(random_margins(npj, htem, margins))
    assert set(program_of) == pairs, (
        "Manuscript prints a pair count and a program count in one sentence "
        "(par:partition); the pairs the artifacts carry "
        f"and the pairs attributed to a program differ by "
        f"{pairs ^ set(program_of)}"
    )
    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
    programs = {program_of[pair] for pair in pairs}
    assert macros["programCount"] == words[len(programs)], (
        f"Manuscript says those pairs come from {macros['programCount']} "
        f"sources of libraries (par:partition) but the attribution carries "
        f"{len(programs)}"
    )
    assert len(programs) < len(pairs), (
        "par:partition offers the program count as the reason the pairs are not "
        f"independent sources, but {len(programs)} programs over {len(pairs)} "
        "pairs states no such limit"
    )


def test_htem_db_is_the_only_source_recording_a_library(npj, htem, margins):
    """abstract: 'On the one dataset that records each position's library, ...';
    par:intro: 'HTEM-DB records each row's library', with the other two
    datasets' tables recording no library.

    The claim ranges over the sources of the derived pair set, so each pair is
    read for the library grouping its own artifact carries: HTEM-DB's results
    count the libraries of every target, EMPD's count alloy systems and no
    library, and Kim and Lee's release ships no library column. The two EMPD
    pairs and Kim and Lee's dataset are the one other source
    (test_every_random_pair_has_a_program_and_they_number_two), so a library
    grouping on the HTEM-DB pairs alone is the exclusivity the abstract states.
    """
    aihub = margins["datasets"]["aihub"]["tasks"]
    libraries_of = {
        "Kim and Lee's dataset": sum("librar" in c.lower()
                                     for c in npj["released_columns"]),
        "EMPD resistivity": aihub["resistivity"].get("libraries", 0),
        "EMPD hardness": aihub["hardness"].get("libraries", 0),
        "HTEM-DB resistivity": htem["targets"]["log10_resistivity"].get(
            "libraries", 0),
        "HTEM-DB band gap": htem["targets"]["direct_band_gap"].get(
            "libraries", 0),
    }
    pairs = set(random_margins(npj, htem, margins))
    assert set(libraries_of) == pairs, (
        "the abstract calls HTEM-DB the one dataset that records each position's "
        "library, a claim over every pair the artifacts carry; the pairs read "
        f"for a library grouping and the pairs derived differ by "
        f"{pairs ^ set(libraries_of)}"
    )
    recording = {pair for pair, n in libraries_of.items() if n}
    assert recording == {"HTEM-DB resistivity", "HTEM-DB band gap"}, (
        "abstract calls HTEM-DB 'the one dataset that records each position's "
        f"library'; the artifacts record a library grouping on "
        f"{sorted(recording)}"
    )


def test_largest_random_margin_bounds_every_pair(npj, htem, margins, macros):
    """par:margin: 'The margin ranges from +0.0022 on Kim and Lee's dataset to
    +0.12 on HTEM-DB band gap'. Three claims: the
    members that the manuscript names are the smallest and the largest over the
    whole set, and the value printed is the largest member's, at the precision
    its row in tab:main prints."""
    pairs = random_margins(npj, htem, margins)
    widest = max(pairs, key=pairs.get)
    assert macros["largestRandomMarginPair"] == widest, (
        f"Manuscript names {macros['largestRandomMarginPair']} as the widest "
        f"margin (par:margin); the artifacts make it {widest}"
    )
    narrowest = min(pairs, key=pairs.get)
    assert narrowest == "Kim and Lee's dataset", (
        f"par:margin says the margins run from Kim and Lee's dataset; the "
        f"artifacts make {narrowest} the narrowest"
    )
    printed = macros["largestRandomMargin"]
    assert math.isclose(number(macros, "largestRandomMargin"), pairs[widest],
                        abs_tol=printed_tol(printed)), (
        f"par:margin prints the widest margin as {printed}; {widest} is "
        f"{pairs[widest]:+.4f}"
    )
    assert printed == macros["htemGapMargin"] or widest != "HTEM-DB band gap", (
        f"par:margin prints {printed} where tab:main prints "
        f"{macros['htemGapMargin']} for the same margin"
    )




def test_control_margin_exceeds_every_combinatorial_pair(npj, htem, margins,
                                                       expt_gap):
    """par:margin: 'On Matbench, the lookup's mean R^2 is lower and the mean
    margin wider than on any of the five pairs in Table 1a'. The comparison is
    made from the control's interval."""
    controls = {"Matbench": expt_gap["ci"]["model_margin"]["lo"]}
    pairs = random_margins(npj, htem, margins)
    for (control_name, lower), (name, value) in (
            (c, p) for c in controls.items() for p in pairs.items()):
        control = {"ci_lo": lower}
        assert control["ci_lo"] > value, (
            "Manuscript claims the control's margin is wider than on any of "
            f"the five pairs (par:margin) but {name} is {value:+.4f} "
            f"against the lower endpoint {control['ci_lo']:+.4f} of the "
            "control's interval"
        )


def test_empd_random_partitions_score_the_printed_share(macros):
    """sec:protocol: 'On EMPD, a random partition assigns 70% of each
    property's rows to training, 15% to the early-stopping rows and 15% to the
    test set'. Read off EMPD's per-partition record, draw by draw, so a draw
    scored on a different share of its rows fails here."""
    aihub = json.loads((ROOT / "outputs" / "aihub" / "draws.json").read_text())
    share = number(macros, "heldOutShare")
    for draw in aihub["draws"].values():
        for task, entry in draw.items():
            got = 100 * entry["n_test"] / aihub["rows"][task]
            assert round(got) == share, (
                f"sec:protocol prints a test share of {share}%, but "
                f"a {task} draw scores {got:.2f}% of its rows"
            )


def test_the_blocked_held_out_share_range_is_read_off_the_draws(macros):
    """sec:protocol: a blocked partition of EMPD has parts that 'have the same
    intended shares of each property's rows as those of a random partition', and
    'Because the alloy systems enter the test set whole, the test set contains
    between 9% and 20% of the resistivity rows and between 6% and 20% of the
    hardness rows, depending on the partition'. The range is over the draws in
    outputs/aihub/draws.json, at the precision printed, and the target is the
    one the draws record."""
    aihub = json.loads((ROOT / "outputs" / "aihub" / "draws.json").read_text())
    for task, prefix in (("resistivity", "empdRes"), ("hardness", "empdHard")):
        shares = [100 * d[task]["blocked_n_test"] / aihub["rows"][task]
                  for d in aihub["draws"].values()]
        lo, hi = number(macros, f"{prefix}BlockedHeldOutMin"), number(macros, f"{prefix}BlockedHeldOutMax")
        assert (round(min(shares)), round(max(shares))) == (lo, hi), (
            f"sec:protocol prints the {task} blocked test share as "
            f"{lo}% to {hi}%, but the draws hold out {min(shares):.1f}% to {max(shares):.1f}%"
        )
    target = 100 * aihub["blocked_target_share"]["test"]
    assert round(target) == number(macros, "heldOutShare"), (
        "sec:protocol prints the random partitions' test share as the blocked "
        f"test set's target size, but the draws record a target of {target:.1f}%"
    )


def test_every_blocked_pair_has_a_positive_excess(htem, margins, macros):
    """par:partition: 'The excess, a pair's drop minus its margin, is
    positive on all four pairs of HTEM-DB and EMPD', reported
    in the final column of Table 1b, and 'Every interval for the excess
    excludes zero.' The lower endpoint is checked on every row. The quantifier
    ranges over the rows of panel b, derived here from the artifacts that fill
    them."""
    blocked = {}
    for name, t in [("HTEM-DB resistivity", htem["targets"]["log10_resistivity"]),
                    ("HTEM-DB band gap", htem["targets"]["direct_band_gap"])]:
        blocked[name] = (t["drop_minus_margin"],
                         t["ci"]["drop_minus_margin"]["lo"])
    for name, task in [("EMPD resistivity", "resistivity"),
                       ("EMPD hardness", "hardness")]:
        t = margins["datasets"]["aihub"]["tasks"][task]["drop_minus_margin"]
        blocked[name] = (t["mean"], t["ci_lo"])

    words = {3: "three", 4: "four", 5: "five"}
    assert macros["blockedPairs"] == words[len(blocked)], (
        f"Manuscript says {macros['blockedPairs']} blocked pairs (par:partition) "
        f"but the artifacts carry {len(blocked)}"
    )
    for name, (point, lo) in blocked.items():
        assert point > 0, (
            "Manuscript claims blocking costs the model more than its margin "
            f"relative to the lookup on every pair (par:partition) but {name} "
            f"is {point:+.4f}"
        )
        assert lo > 0, (
            f"Table 1b prints {name}'s excess interval beside a positive "
            f"excess (par:partition), but its lower endpoint is {lo:+.4f}"
        )


def test_the_ablation_reference_arm_is_the_published_forest(ablation, htem):
    """par:partition reads the three arms as one contrast under three
    estimators, which holds only if the ablation scores the dataset the
    manuscript publishes. The reference arm is the published forest, so it is
    the handle: if it has moved, the other two arms are an ablation of
    something else."""
    for key, target in ablation["targets"].items():
        published = htem["targets"][key]
        got = target["arms"]["default_forest"]
        assert math.isclose(got["blocked_model_r2"],
                            published["blocked"]["model_r2"], abs_tol=1e-9), (
            f"outputs/htem/ablation.json scores the {key} blocked forest at "
            f"{got['blocked_model_r2']:.6f} against "
            f"{published['blocked']['model_r2']:.6f} in split.json"
        )
        assert math.isclose(got["excess"], published["drop_minus_margin"],
                            abs_tol=1e-9), (
            f"the {key} excess is {got['excess']:.6f} in the ablation and "
            f"{published['drop_minus_margin']:.6f} in split.json"
        )


def test_the_excess_survives_every_estimator_the_ablation_ran(ablation, macros):
    """par:partition: 'the positive excess on HTEM-DB is shown only for an
    untuned random forest and for two tuned models, a forest and a
    gradient-boosting machine' and sec:esi-posthoc: 'Under either
    tuned model, the HTEM-DB excess is positive on both properties. Every
    interval for that excess excludes zero.' The quantifier ranges over the arms
    the artifact carries, so an arm added to reproduce/htem_ablation.py is
    covered without touching this test."""
    for key, target in ablation["targets"].items():
        assert len(target["arms"]) >= 3, (
            f"par:partition reports three models on {key}; the ablation "
            f"carries {len(target['arms'])}"
        )
        for arm, entry in target["arms"].items():
            assert entry["excess"] > 0, (
                "par:partition claims blocking costs the model more than its "
                f"margin relative to the lookup under every model, but on "
                f"{key} the {arm} arm gives {entry['excess']:+.4f}"
            )
            assert entry["ci"]["lo"] > 0, (
                f"sec:esi-posthoc claims every interval excludes zero; on {key} "
                f"the {arm} arm's lower endpoint is {entry['ci']['lo']:+.4f}"
            )


def test_the_ablation_arms_were_given_one_search_budget(ablation, macros):
    """sec:esi-posthoc: 'Each tuned model chooses among 25 configurations by their
    scores on one of four folds into which it splits the rows of its training
    folds', so both tuned models searched the same number of configurations. Two
    models tuned to different depths differ in search before they differ in
    anything the paper proposes."""
    budget = ablation["search_budget"]
    assert {len(g) for g in ablation["grids"].values()} == {budget}, (
        f"the ablation's grids are {[len(g) for g in ablation['grids'].values()]}"
        f", not one budget of {budget}"
    )
    assert number(macros, "htemSearchBudget") == budget, (
        f"sec:esi-posthoc says {macros['htemSearchBudget']} configurations; the "
        f"run tried {budget}"
    )
    for key, target in ablation["targets"].items():
        for arm, entry in target["arms"].items():
            if entry["selected_per_fold"]["random"] is None:
                continue
            assert entry["configurations_tried"] == budget, (
                f"the {key} {arm} arm searched "
                f"{entry['configurations_tried']} configurations, not {budget}"
            )


def test_the_printed_grids_are_the_grids_searched(ablation, macros):
    """sec:esi-posthoc: 'Its configurations combine a share of 0.3, 0.5 or 1.0 of
    the features considered at each split with a minimum of 1, 2 or 5 rows per
    leaf. The configurations of the gradient-boosting machine combine a
    learning rate of 0.03, 0.06 or 0.12 with at most 15, 31 or 63 leaves per
    tree. Every configuration of the gradient-boosting machine sets the L2
    regularization to 1.0.' The sentence
    says the grid is every combination
    of the listed values, so the grid is asserted to be that product."""
    printed = {
        "tuned_forest": {"max_features": "htemForestFeatureShares",
                         "min_samples_leaf": "htemForestMinLeaf"},
        "boosted": {"learning_rate": "htemBoostRates",
                    "max_leaf_nodes": "htemBoostLeaves",
                    "l2_regularization": "htemBoostPenalty"},
    }
    for arm, axes in printed.items():
        grid = ablation["grids"][arm]
        values = {axis: {c[axis] for c in grid} for axis in axes}
        for axis, name in axes.items():
            listed = {float(v) for v in re.split(r", | or ", macros[name])}
            assert listed == {float(v) for v in values[axis]}, (
                f"sec:esi-posthoc prints {macros[name]} for the {arm} {axis}; the "
                f"run searched {sorted(values[axis])}")
        combinations = math.prod(len(v) for v in values.values())
        assert len(grid) == combinations == len({tuple(sorted(c.items())) for c in grid}), (
            f"sec:esi-posthoc says the {arm} configurations combine the listed "
            f"values; the run's grid holds {len(grid)} of {combinations} combinations")
    assert number(macros, "htemTunedTrees") == ablation["tuned_trees"], (
        f"sec:esi-posthoc says 'The tuned forest averages {macros['htemTunedTrees']} "
        f"decision trees'; reproduce/htem_ablation.py grew {ablation['tuned_trees']}")


def test_the_ablation_cells_are_the_numbers_the_run_produced(ablation, macros):
    """tab:ablation prints two tuned models per property, under the random and
    the blocked partitions. Every cell is checked against
    outputs/htem/ablation.json."""
    for key, prefix in (("log10_resistivity", "htemRes"),
                        ("direct_band_gap", "htemGap")):
        target = ablation["targets"][key]
        for arm, suffix in (("tuned_forest", "Tuned"), ("boosted", "Boosted")):
            entry = target["arms"][arm]
            for kind, field in (("Blocked", "blocked_model_r2"), ("Random", "random_model_r2")):
                name = f"{prefix}{kind}{suffix}"
                assert math.isclose(number(macros, name), entry[field],
                                    abs_tol=printed_tol(macros[name])), (
                    f"tab:ablation prints \\{name}={macros[name]}; the run gives "
                    f"{entry[field]:+.4f}")
            assert math.isclose(number(macros, f"{prefix}Excess{suffix}"),
                                entry["excess"],
                                abs_tol=printed_tol(macros[f"{prefix}Excess{suffix}"])), (
                f"tab:ablation prints \\{prefix}Excess{suffix}="
                f"{macros[f'{prefix}Excess{suffix}']}; the run gives "
                f"{entry['excess']:+.4f}"
            )
            lo, hi = endpoints(macros, f"{prefix}Excess{suffix}CI")
            tol = interval_tol(macros[f"{prefix}Excess{suffix}CI"])
            assert math.isclose(lo, entry["ci"]["lo"], abs_tol=tol)
            assert math.isclose(hi, entry["ci"]["hi"], abs_tol=tol)


def test_tuning_narrows_the_excess_most_on_band_gap(ablation):
    """sec:esi-posthoc: 'The excess changes most under the tuned forest on
    band gap, from +0.19 to +0.10'. A superlative ranges over a set the paper
    enumerates, so it is checked over every model of every target rather
    than over the one the sentence names."""
    narrowing = {}
    for key, target in ablation["targets"].items():
        reference = target["arms"]["default_forest"]["excess"]
        for arm, entry in target["arms"].items():
            if arm != "default_forest":
                narrowing[(key, arm)] = reference - entry["excess"]
    widest = max(narrowing, key=narrowing.get)
    assert widest == ("direct_band_gap", "tuned_forest"), (
        "sec:esi-posthoc says the tuned forest on band gap changes the excess "
        "most; "
        f"the largest narrowing is {widest} at {narrowing[widest]:+.4f}"
    )


def test_the_two_htem_pairs_are_nested(htem, macros):
    """sec:datasets: 'HTEM-DB records resistivity and, at some of the same
    positions, an optical direct band gap', and 'Of those rows, 5,264 over 156
    libraries record a band gap.' The
    quantifier ranges over the row sets themselves, which
    reproduce/htem_split.py derives from the dataset."""
    n = htem["nesting"]
    assert n["narrower_rows_inside_wider"] == n["narrower_rows"], (
        f"sec:methods claims every {n['narrower']} row is also a "
        f"{n['wider']} row, but {n['narrower_rows'] - n['narrower_rows_inside_wider']} "
        f"of {n['narrower_rows']} are not"
    )
    assert n["narrower_libraries_inside_wider"] == n["narrower_libraries"], (
        f"sec:methods puts the {n['narrower']} libraries inside the "
        f"{n['wider']} libraries; "
        f"{n['narrower_libraries'] - n['narrower_libraries_inside_wider']} "
        f"are outside"
    )
    for macro, value in (("htemGapRows", n["narrower_rows"]),
                         ("htemResRows", n["wider_rows"]),
                         ("htemGapLibraries", n["narrower_libraries"]),
                         ("htemResLibraries", n["wider_libraries"])):
        assert number(macros, macro) == value, (
            f"sec:methods prints \\{macro} beside the nesting claim; the "
            f"dataset carries {value}"
        )


def test_the_network_features_are_the_ones_the_esi_counts(macros):
    """sec:features: 'On EMPD, the features are the atomic percent of each of
    15 elements, the film thickness and the six descriptors that EMPD's tables
    contain. EMPD's descriptors are the mean atomic radius, the atomic size
    mismatch, the mixing enthalpy, the mean electronegativity, the
    electronegativity difference and the number of elements.'
    Read off the record that the training script wrote, with EMPD's
    descriptor names bound to its columns in the record's order. A column
    named in two feature groups is one column of the table, so one group
    overwrites the other's values; the groups must be disjoint for every named
    feature to be one."""
    empd_names = {"ravg": "mean atomic radius", "delta": "atomic size mismatch",
                  "dHmix": "mixing enthalpy", "ENavg": "mean electronegativity",
                  "dEN": "electronegativity difference", "N": "number of elements"}
    empd = json.loads((ROOT / "outputs" / "aihub" / "draws.json").read_text())
    names = [f"the {empd_names[c]}" for c in empd["features"]["descriptors"]]
    printed = ", ".join(names[:-1]) + " and " + names[-1]
    assert printed in " ".join(document_text().split()), (
        f"sec:features names EMPD's descriptors, but the record's "
        f"{empd['features']['descriptors']} would print as '{printed}'")
    for name, prefix, processing in (("aihub", "empd", ["Thickness"]),):
        features = json.loads(
            (ROOT / "outputs" / name / "draws.json").read_text())["features"]
        names = [column for group in features.values() for column in group]
        shared = sorted({column for column in names if names.count(column) > 1})
        assert not shared, (
            f"{name}: {shared} name more than one feature, so one overwrites the other")
        assert number(macros, f"{prefix}ElementCount") == len(features["elements"])
        assert macros[f"{prefix}DescriptorCount"] == {6: "six"}.get(
            len(features["descriptors"])), (
            f"sec:features prints {macros[f'{prefix}DescriptorCount']} "
            f"descriptors for {name}; the record lists {len(features['descriptors'])}")
        assert features.get("processing", []) == processing, (
            f"sec:features names the film thickness among EMPD's features; "
            f"the {name} record lists "
            f"{features.get('processing', [])}")


def test_the_temperature_fill_value_is_justified_by_its_sensitivity(htem, macros):
    """sec:features: 'A row that records no deposition temperature takes -1
    instead, a value below every recorded temperature', and sec:esi-settings:
    'any value below every recorded deposition temperature would serve as the
    value of a row that records none, because replacing -1 by 0, -100
    or -1000 changes each HTEM-DB margin and excess by at most 0.0049 in R^2 and
    each blocked-partition lookup score by at most 0.069 in R^2. On both
    properties the
    lookup's R^2 still falls further than the model's under the blocked
    partition, whichever of these values replaces -1.' Both bounds and the
    comparison range over both HTEM-DB properties and every value the fill was
    replaced by, -1 included."""
    fills = [htem["no_temperature"]] + htem["no_temperature_alternatives"]
    assert number(macros, "htemNoTemperature") == htem["no_temperature"]
    for key, target in htem["targets"].items():
        s = target["no_temperature_sensitivity"]
        assert all(f < s["lowest_recorded_temperature"] for f in fills), (
            f"sec:features says the fill lies below every recorded "
            f"temperature, but {key} records {s['lowest_recorded_temperature']}")
        assert sorted(s["alternatives"]) == sorted(
            str(f) for f in htem["no_temperature_alternatives"])
        assert s["max_margin_or_excess_shift"] <= number(macros, "htemNoTemperatureShift"), (
            f"sec:esi-settings bounds the change in each margin and excess by "
            f"{macros['htemNoTemperatureShift']}; {key} changes by "
            f"{s['max_margin_or_excess_shift']:.4f}")
        assert s["max_blocked_lookup_shift"] <= number(macros, "htemNoTemperatureLookupShift"), (
            f"sec:esi-settings bounds the change in the blocked-partition lookup "
            f"score by {macros['htemNoTemperatureLookupShift']}; {key} changes by "
            f"{s['max_blocked_lookup_shift']:.4f}")
        scores = {str(htem["no_temperature"]): {
            f"{kind}_{arm}_r2": target[kind][f"{arm}_r2"]
            for kind in ("random", "blocked") for arm in ("lookup", "model")}}
        scores.update(s["alternatives"])
        for fill, a in scores.items():
            lookup_drop = a["random_lookup_r2"] - a["blocked_lookup_r2"]
            model_drop = a["random_model_r2"] - a["blocked_model_r2"]
            assert lookup_drop > model_drop, (
                f"sec:features says the lookup's R^2 still falls further than "
                f"the model's under the blocked partition for every fill value; "
                f"{key} with {fill} lowers the lookup by {lookup_drop:.4f} and the "
                f"model by {model_drop:.4f}")


def test_the_blocked_pairs_span_fewer_datasets_than_pairs(htem, margins, macros):
    """par:partition: 'The excess, a pair's drop minus its margin, is
    positive on all four pairs of HTEM-DB and EMPD', the pairs of Table 1b,
    and par:partition: 'The second check runs on HTEM-DB and EMPD, the two
    datasets whose releases record each row's library or its alloy system'. Both counts are derived
    from the artifacts filling panel b, so a dataset added to the blocked
    column moves them together."""
    pairs = {("HTEM-DB", key) for key in htem["targets"]}
    pairs |= {("EMPD", task) for task, entry
              in margins["datasets"]["aihub"]["tasks"].items()
              if "blocked_model_r2" in entry}
    datasets = {dataset for dataset, _ in pairs}
    words = {2: "two", 3: "three", 4: "four", 5: "five"}
    assert macros["blockedPairs"] == words[len(pairs)], (
        f"manuscript says {macros['blockedPairs']} blocked pairs; the "
        f"artifacts carry {len(pairs)}"
    )
    assert macros["blockedDatasets"] == words[len(datasets)], (
        f"par:intro applies the second check to {macros['blockedDatasets']} "
        f"datasets; the artifacts carry {len(datasets)}"
    )


def test_the_aihub_blocked_column_is_reported_by_its_shape(margins, macros):
    """par:partition: 'On EMPD, the excess is also positive for both properties
    under each of ten blocked partitions', and sec:esi-draw-spread: 'On EMPD, the model's blocked R^2
    varies so widely across the ten blocked partitions that its mean describes
    no typical partition', 'The blocked R^2 ranges from -0.936 to +0.839 on
    resistivity and from -23.541 to +0.730 on hardness' and 'Because of this
    spread, the model's mean on hardness in Table 1b of the main text, -6,
    lies far below its median of -1.354, so the mean depends mostly on the
    partitions with the lowest R^2'. Every figure in those paragraphs is
    checked against the per-draw record, because the paragraph
    exists to stop the mean being read alone."""
    aihub = margins["datasets"]["aihub"]["tasks"]
    printed = {
        "resistivity": {"empdResBlockedMin": "blocked_min",
                        "empdResBlockedMax": "blocked_max"},
        "hardness": {"empdHardBlockedMin": "blocked_min",
                     "empdHardBlockedMax": "blocked_max",
                     "empdHardBlockedMedian": "blocked_median"},
    }
    for task, macro_map in printed.items():
        for macro, field in macro_map.items():
            assert math.isclose(number(macros, macro), aihub[task][field],
                                abs_tol=R2_TOL), (
                f"the paper prints \\{macro}; the per-draw record gives "
                f"{aihub[task][field]:+.4f}"
            )

    hard = aihub["hardness"]
    assert hard["blocked_model_r2"]["mean"] < hard["blocked_median"], (
        "sec:esi-draw-spread says EMPD hardness's blocked mean sits far below its "
        f"median; the mean is {hard['blocked_model_r2']['mean']:+.4f} against "
        f"a median of {hard['blocked_median']:+.4f}"
    )
    # 'so the mean depends mostly on the partitions with the lowest R^2': the
    # partitions below the median carry more than half of the sum that the
    # mean divides.
    draws = sorted(hard["blocked_per_draw"])
    low = draws[:len(draws) // 2]
    assert sum(low) / sum(draws) > 0.5, (
        f"sec:esi-draw-spread says the hardness mean depends mostly on the "
        f"partitions with the lowest R^2; the lower half carries "
        f"{sum(low) / sum(draws):.0%} of the sum")
    # 'lies below zero under nine of the ten partitions' and, on resistivity,
    # 'below zero under three partitions and above zero under the other seven'.
    for task, macro in (("hardness", "empdHardBlockedNegative"),
                        ("resistivity", "empdResBlockedNegative")):
        per = margins["datasets"]["aihub"]["tasks"][task]["blocked_per_draw"]
        assert macros[macro] == WORDS[sum(v < 0 for v in per)], (
            f"sec:esi-draw-spread prints \\{macro}={macros[macro]}; the "
            f"{task} partitions put {sum(v < 0 for v in per)} below zero")
    per = margins["datasets"]["aihub"]["tasks"]["resistivity"]["blocked_per_draw"]
    assert macros["empdResBlockedPositive"] == WORDS[sum(v > 0 for v in per)]
    assert sum(v < 0 for v in per) + sum(v > 0 for v in per) == len(per)

    # "Every EMPD partition of both properties" ranges over the per-draw
    # record, so the record supplies the set rather than the sentence.
    for task, entry in aihub.items():
        negative = sum(e <= 0 for e in entry["excess_per_draw"])
        assert negative == 0, (
            "par:partition says the per-partition excess is positive on every "
            f"EMPD partition of both properties; on {task} {negative} of "
            f"{len(entry['excess_per_draw'])} are not"
        )


def test_each_aihub_blocked_draw_scores_a_different_row_set(margins):
    """sec:esi-draw-spread: 'The blocked R^2 ranges from -0.936 to +0.839 on
    resistivity and from -23.541 to +0.730 on hardness, partly because each
    partition's R^2 is computed over a different set of test rows, relative to
    the variance of their targets'. The test-set sizes are
    what shows it, and they are not printed, so nothing else would catch them
    becoming equal."""
    for task, entry in margins["datasets"]["aihub"]["tasks"].items():
        sizes = entry.get("blocked_n_test")
        if sizes is None:
            continue
        assert len(set(sizes)) > 1, (
            f"sec:esi-draw-spread says each EMPD {task} blocked partition scores a "
            f"different set of rows; every partition holds out {sizes[0]} rows"
        )


def test_excess_equals_random_lookup_minus_blocked_model(htem, margins):
    """sec:protocol: 'The drop is the model's random-partition R^2 minus its
    blocked-partition R^2. The excess is the drop minus the margin. The excess
    equals the lookup's random-partition R^2 minus the model's blocked-partition
    R^2.' The identity is the sentence's
    whole argument, so it is checked rather than trusted."""
    for name, t in htem["targets"].items():
        identity = t["random"]["lookup_r2"] - t["blocked"]["model_r2"]
        assert math.isclose(t["drop_minus_margin"], identity, rel_tol=1e-9), (
            f"sec:protocol claims drop - margin = random lookup - blocked "
            f"model; on {name} the data gives {t['drop_minus_margin']:.6f} "
            f"against {identity:.6f}"
        )


def test_lookup_reaches_most_of_the_model_on_every_pair(npj, htem, margins):
    """par:margin: 'Under a random partition, the lookup reaches most of the
    model's R^2 on all five pairs.' abstract: 'A nearest-neighbor lookup
    reaches most of the model's R^2 on three datasets.' fig:split's caption:
    'Under a random partition, the lookup reaches most of the model's R^2 on
    each property of each combinatorial dataset.' 'Most' is
    decidable: the lookup
    holds more than half of what the model scores."""
    aihub = margins["datasets"]["aihub"]["tasks"]
    pairs = {
        "Kim and Lee's dataset": (npj["lookup"]["all_features"]["r2"],
                                 npj["model"]["val"]["r2"]),
        "EMPD resistivity": (aihub["resistivity"]["lookup_r2"]["mean"],
                               aihub["resistivity"]["model_r2"]["mean"]),
        "EMPD hardness": (aihub["hardness"]["lookup_r2"]["mean"],
                            aihub["hardness"]["model_r2"]["mean"]),
        "HTEM-DB resistivity": (htem["targets"]["log10_resistivity"]["random"]["lookup_r2"],
                             htem["targets"]["log10_resistivity"]["random"]["model_r2"]),
        "HTEM-DB band gap": (htem["targets"]["direct_band_gap"]["random"]["lookup_r2"],
                          htem["targets"]["direct_band_gap"]["random"]["model_r2"]),
    }
    for name, (lookup, model) in pairs.items():
        assert lookup > 0.5 * model, (
            "Manuscript claims the lookup reaches most of the model's accuracy "
            f"on all five pairs (par:margin) but on {name} it is {lookup:.4f} "
            f"against {model:.4f}"
        )


def test_the_lookup_falls_further_than_the_model_under_blocking(htem):
    """par:partition: 'The lookup's R^2 on HTEM-DB falls by more than half and further than
    the model's', from its random-partition score to its blocked one. Over HTEM-DB's targets,
    derived from the file rather than named."""
    for name, t in htem["targets"].items():
        lookup_drop = t["random"]["lookup_r2"] - t["blocked"]["lookup_r2"]
        model_drop = t["random"]["model_r2"] - t["blocked"]["model_r2"]
        assert lookup_drop > model_drop, (
            f"par:partition says the lookup falls further than the model on "
            f"{name}; it drops {lookup_drop:.4f} against {model_drop:.4f}"
        )
        # par:partition: 'so most of the lookup's random-partition R^2 depends
        # on training rows from the test row's own library'.
        assert lookup_drop > 0.5 * t["random"]["lookup_r2"], (
            f"par:partition says most of the lookup's random-partition R^2 is "
            f"lost under blocking; on {name} it loses {lookup_drop:.4f} of "
            f"{t['random']['lookup_r2']:.4f}"
        )
        # sec:protocol: 'The excess equals the lookup's random-partition R^2
        # minus the model's blocked-partition R^2, so a positive excess means
        # that the model scores lower on groups excluded from training than the
        # lookup scores under the random partition.' The abstract states a
        # positive excess as an overstatement larger than a gain
        # (test_the_blocked_loss_exceeds_its_margin).
        assert t["blocked"]["model_r2"] < t["random"]["lookup_r2"], (
            f"sec:protocol says a positive excess puts the blocked model below "
            f"the random-partition lookup; on {name} it scores "
            f"{t['blocked']['model_r2']:.4f} against {t['random']['lookup_r2']:.4f}"
        )


def test_the_descriptor_quantifier_holds_over_its_own_table(macros):
    """par:intro: HTEM-DB's authors 'find that excluding whole libraries rather
    than random positions from training raises the root mean squared error
    (RMSE) of their model of conductivity'. The descriptor's Table 1 scores the model under four sets of input
    variables, and the claim holds under each.

    A claim about Table 1 of the HTEM-DB descriptor (Sci. Data 5, 180053),
    quoted and not measured here, so the table is transcribed with it. RMSE/SD
    in log10(S/m), position-level and library-level resampling, in the column
    order the table prints.
    """
    table = {"compositional": ((1.35, 0.02), (1.68, 0.13)),
             "structural": ((2.15, 0.01), (2.25, 0.12)),
             "synthesis": ((0.976, 0.03), (1.93, 0.11)),
             "combined": ((0.67, 0.03), (1.61, 0.15))}

    assert len(table) == 4, (
        "The descriptor's Table 1 scores four sets of input variables; "
        f"the transcription carries {len(table)}")
    losing = [name for name, ((pos, _), (lib, _)) in table.items() if lib <= pos]
    assert not losing, (
        "Manuscript claims excluding whole libraries raises the error of their "
        f"model (par:intro) but it does not under {losing}"
    )


def test_thelen_reports_two_statistics_and_no_baseline():
    """par:intro: 'We found three reports that test models on combinatorial
    datasets under a blocked partition and score the groups excluded from
    training by the root mean squared error (RMSE) or the mean absolute error'
    and 'None of the three reports gives the coefficient of determination R^2
    under its blocked partition'. Thelen et al. are the report that gives a
    mean absolute error for an excluded library and R^2 only for random
    partitions.

    Quoted from Thelen et al. (Adv. Sci. 12, e07302), not measured here, so
    what each of their two figures reports is transcribed with the claim. The
    manuscript rests on the two carrying different statistics, which is a
    property of that transcription rather than of any magnitude, so the check
    reads the statistic each figure carries and not its values.
    """
    reported = {"training fraction of every library (Fig. 9)": "R2",
                "each library excluded in turn (Fig. 8a)": "MAE"}
    assert reported["each library excluded in turn (Fig. 8a)"] != "R2", (
        "Manuscript says Thelen et al. report no R^2 for an excluded library "
        f"(par:intro) but the transcription carries {reported}"
    )
    assert len(set(reported.values())) == len(reported)


# --------------------------------------------------------------------------
# 3. Joins and populations
# --------------------------------------------------------------------------


def declared_populations() -> dict[str, tuple[str, str]]:
    """Population name -> (artifact, unit noun), read from parameters.tex."""
    out = {}
    for line in PARAMETERS.read_text().splitlines():
        m = re.match(r"%% POPULATION (\w+) \| artifact: (.+?) \| unit: (\w+)$", line)
        if m:
            out[m.group(1)] = (m.group(2).strip(), m.group(3))
    return out


def test_populations_resolve_apart():
    """Two declared populations resolving to one artifact are one population,
    and the second name is a stale alias."""
    declared = declared_populations()
    assert declared, "parameters.tex declares no populations"
    seen = {}
    for name, (artifact, _unit) in declared.items():
        assert artifact not in seen, (
            f"populations {seen[artifact]!r} and {name!r} both resolve to "
            f"{artifact!r}: one of the two names is stale"
        )
        seen[artifact] = name


def test_the_manual_region_is_what_a_fresh_bootstrap_would_write():
    """The manual region is hand-edited in paper/parameters.tex, and
    make_parameters.py carries a second copy of it as the seed a run writes
    when that file is absent. Nothing dereferences the seed while the file
    exists, so a block added to one and not the other leaves a clone building
    against undefined macros with this suite green. Compare the two rather
    than listing the blocks each should hold.
    """
    generator = (ROOT / "reproduce" / "make_parameters.py").read_text()
    seed = generator.split('MANUAL_SEED = """', 1)[1].split('"""', 1)[0]
    seed = seed.replace("\\\\", "\\")
    written = PARAMETERS.read_text()
    written = written[:written.index("% ---- BEGIN GENERATED ----")]
    assert seed == written, (
        "make_parameters.py's MANUAL_SEED and paper/parameters.tex's manual "
        "region have diverged, so regenerating from an absent parameters.tex "
        "would not reproduce the committed file"
    )


def test_every_population_artifact_exists():
    """A declaration naming a file nobody writes is a pointer to nothing."""
    for name, (artifact, _unit) in declared_populations().items():
        # Membership is derived from the declaration, not from a list of
        # journal names that would need a line per citation and would go stale
        # silently on the one it lacks.
        if name.startswith("cited") or artifact.endswith(".md"):
            continue  # quoted from a paper, or documentary, not a written path
        for part in artifact.split(":")[0].split("+"):
            path = ROOT / part.strip()
            assert path.exists(), (
                f"population {name!r} declares artifact {part!r}, which does "
                "not exist. Upstream datasets are fetched, not vendored: run "
                "reproduce/fetch_npj.py and reproduce/fetch_htem.py first"
            )


def test_size_collisions_name_both_artifacts():
    """Equal sizes are not proof of a crossing, so a collision is a warning
    discharged by naming both artifacts -- which the generator's header does."""
    macros_text = PARAMETERS.read_text()
    counts = {}
    for name, value in re.findall(r"\\newcommand\{\\(\w+Rows)\}\{([\d{,}]+)\}",
                                  macros_text):
        counts.setdefault(value.replace("{,}", ""), []).append(name)
    collisions = {v: names for v, names in counts.items() if len(names) > 1}
    header = (ROOT / "reproduce" / "make_parameters.py").read_text()
    for value, names in collisions.items():
        assert value in header, (
            f"macros {names} all print {value} rows over different populations "
            "and make_parameters.py's header does not name the collision"
        )


def test_no_macro_is_defined_and_unprinted():
    """A generated number no document prints is a measurement with no reader.
    Scanning the article alone would miss the \\input'd table bodies and the supplementary
    information, both of which the reader receives."""
    text = document_text() + table_bodies()
    defined = set(re.findall(r"\\newcommand\{\\(\w+)\}", PARAMETERS.read_text()))
    unused = sorted(m for m in defined
                    if not re.search(rf"\\{m}(?![a-zA-Z])", text))
    assert not unused, f"parameters.tex defines macros nothing prints: {unused}"


def test_no_macro_is_printed_and_undefined():
    text = document_text() + table_bodies()
    defined = set(re.findall(r"\\newcommand\{\\(\w+)\}", PARAMETERS.read_text()))
    latex = {"documentclass", "usepackage", "input", "title", "author", "date",
             "begin", "end", "maketitle", "section", "subsection", "label",
             "citep", "textwidth", "includegraphics", "caption", "centering",
             "small", "textbf", "textit", "emph", "ref", "bibliographystyle",
             "bibliography", "toprule", "midrule", "bottomrule", "log", "mu",
             "Omega", "cdot", "newcommand", "vspace", "hline", "quad"}
    used = set(re.findall(r"\\([a-zA-Z]+)", text))
    dangling = sorted(m for m in used
                      if m not in defined and m not in latex
                      and any(m.startswith(p) for p in
                              ("npj", "aihub", "htem", "expt", "mlp",
                               "split", "random", "blocked", "largest",
                               "bootstrap", "margin", "program", "thelen")))
    assert not dangling, f"main.tex prints undefined macros: {dangling}"


def test_each_value_is_printed_to_the_digit_its_interval_resolves(
        htem, margins, expt_gap, ablation, macros):
    """sec:statistics: 'Table 1 and Table S1 round each value that has a bootstrap interval to the decimal place of the first significant
    digit of the interval's half-width, whether or not the table shows the
    interval. They keep one more decimal place if that digit is 1. They round each value that
    has no interval to three decimals, except Kim and Lee's scores and margin,
    which Table 1 rounds to four'. The rule is checked
    against the artifacts' intervals, whether or not the table prints the
    interval beside the value, so a regenerated interval that resolves another
    digit fails here until the generator prints it."""
    def resolved(lo, hi):
        half = (hi - lo) / 2
        mantissa, power = f"{half:.12e}".split("e")
        half = round(half, -int(power) + (1 if mantissa[0] == "1" else 0))
        mantissa, power = f"{half:.12e}".split("e")
        return max(0, -int(power) + (1 if mantissa[0] == "1" else 0))

    def places(name):
        raw = macros[name].replace(r"$-$", "-")
        return len(raw.split(".")[1]) if "." in raw else 0

    aihub = margins["datasets"]["aihub"]["tasks"]
    with_ci = []
    for prefix, task in (("empdRes", "resistivity"), ("empdHard", "hardness")):
        t = aihub[task]
        for name, key in (("Margin", "margin"), ("BlockedLookup", "blocked_lookup_r2"),
                          ("Blocked", "blocked_model_r2"), ("Excess", "drop_minus_margin")):
            with_ci.append((prefix + name, t[key]["ci_lo"], t[key]["ci_hi"]))
    for prefix, key in (("htemRes", "log10_resistivity"), ("htemGap", "direct_band_gap")):
        ci = htem["targets"][key]["ci"]
        for name, field in (("RandomLookup", "random_lookup_r2"),
                            ("RandomModel", "random_model_r2"),
                            ("BlockedLookup", "blocked_lookup_r2"),
                            ("BlockedModel", "blocked_model_r2"),
                            ("Margin", "model_margin"), ("Excess", "drop_minus_margin")):
            with_ci.append((prefix + name, ci[field]["lo"], ci[field]["hi"]))
        for arm, suffix in (("tuned_forest", "Tuned"), ("boosted", "Boosted")):
            entry = ablation["targets"][key]["arms"][arm]["ci"]
            with_ci.append((f"{prefix}Excess{suffix}", entry["lo"], entry["hi"]))
    first = htem["targets"]["direct_band_gap"]["ci"]["first_partition_margin"]
    with_ci.append(("htemGapFirstMargin", first["lo"], first["hi"]))
    for name, field in (("exptGapLookup", "random_lookup_r2"),
                        ("exptGapModel", "random_model_r2"),
                        ("exptGapMargin", "model_margin"),
                        ("exptGapFirstMargin", "first_partition_margin")):
        with_ci.append((name, expt_gap["ci"][field]["lo"], expt_gap["ci"][field]["hi"]))
    for name, field in (("exptGapNonzeroLookup", "random_lookup_r2"),
                        ("exptGapNonzeroMargin", "model_margin")):
        d = expt_gap["nonzero_gap"]["ci"][field]
        with_ci.append((name, d["lo"], d["hi"]))
    for key, prefix in (("log10_resistivity", "htemRes"), ("direct_band_gap", "htemGap")):
        criteria = htem["targets"][key]["criteria"]
        with_ci.append((f"{prefix}BeyondHalf", criteria["C1"]["ci"]["lo"], criteria["C1"]["ci"]["hi"]))
        with_ci.append((f"{prefix}FallExcess", criteria["C2"]["ci"]["lo"], criteria["C2"]["ci"]["hi"]))
    ablation = json.loads((ROOT / "outputs" / "htem" / "ablation.json").read_text())
    for key, prefix in (("log10_resistivity", "htemRes"), ("direct_band_gap", "htemGap")):
        for arm, suffix in (("tuned_forest", "Tuned"), ("boosted", "Boosted")):
            entry = ablation["targets"][key]["arms"][arm]
            for kind, field in (("Random", "ci_random_model_r2"), ("Blocked", "ci_blocked_model_r2")):
                with_ci.append((f"{prefix}{kind}{suffix}", entry[field]["lo"], entry[field]["hi"]))
    for name, lo, hi in with_ci:
        assert places(name) == resolved(lo, hi), (
            f"\\{name}={macros[name]} prints {places(name)} decimals; its interval "
            f"[{lo:+.4f}, {hi:+.4f}] resolves {resolved(lo, hi)} (sec:statistics)"
        )
        if f"{name}CI" in macros:
            ci = macros[f"{name}CI"].strip("[]").split(", ")[0].replace(r"$-$", "-")
            assert (len(ci.split(".")[1]) if "." in ci else 0) == places(name), (
                f"\\{name}CI prints its endpoints to another precision than \\{name}")
    without_ci = ["empdResLookup", "empdResModel", "empdHardLookup", "empdHardModel"]
    for name in without_ci:
        assert places(name) == 3, (
            f"\\{name}={macros[name]} has no interval and prints {places(name)} "
            f"decimals; sec:statistics says three")
    for name in ("npjLookupRSq", "npjModelRSq", "npjMargin"):
        assert places(name) == 4, (
            f"\\{name}={macros[name]}: sec:statistics gives the Kim and Lee row four decimals")


def test_the_kim_and_lee_row_needs_its_fourth_decimal(macros):
    """sec:statistics gives the Kim and Lee row four decimals, and the reason:
    'They print each value that has no interval to three decimals, except Kim
    and Lee's scores and margin, which Table 1 prints to four, because at three
    decimals the margin, +0.0022, would keep only one significant digit.' The reason is about what three
    decimals would print, so it is checked against the rounding rather than
    against the four-decimal row the table carries."""
    def significant(value, places):
        digits = f"{abs(round(value, places)):.{places}f}".replace(".", "").lstrip("0")
        return len(digits)

    margin = number(macros, "npjMargin")
    assert significant(margin, 3) == 1 and significant(margin, 4) >= 2, (
        f"sec:statistics says three decimals would leave the Kim and Lee margin "
        f"{margin:+.4f} one significant digit, but three print {margin:+.3f}: the "
        f"row could print three like every other"
    )


def test_a_difference_of_printed_values_differs_at_most_in_its_last_digit(macros):
    """sec:statistics: 'A difference computed from rounded values can differ
    in its last digit from the rounded difference.'
    tab:ablation prints its excesses by the same rounding rule, so its rows are
    checked too. Each difference that either table prints is set
    against the printed values that it subtracts: model - lookup against its
    row's two scores, panel b's excess against the row's random-partition
    lookup in panel a minus its blocked model score, and tab:ablation's
    excess against the HTEM-DB lookup score in tab:main's panel a minus the
    row's blocked score. The two may differ by one unit in the last digit that the
    coarser of them prints. The rows are read off the generated tables, so a
    row added to either is covered."""
    def value(name):
        return float(macros[name].replace("{,}", "").replace(r"$-$", "-"))

    def places(name):
        raw = macros[name].replace(r"$-$", "-")
        return len(raw.split(".")[1]) if "." in raw else 0

    def rows(body, width):
        """Each data row of a generated table body: its label and the macro
        names in each of its cells. A header row prints no macro in its last
        cell."""
        out = []
        for line in body.splitlines():
            columns = line.rstrip(" \\").split(" & ")
            if len(columns) == width and re.search(r"\\\w+", columns[-1]):
                out.append((columns[0], [re.findall(r"\\(\w+)", c) for c in columns]))
        return out

    panel_a, panel_b = (PAPER / "tables" / "main.tex").read_text().split("\\textbf{(b)")
    tables = {"tab:main panel a": rows(panel_a, 6), "tab:main panel b": rows(panel_b, 4),
              "tab:ablation": rows((PAPER / "tables" / "ablation.tex").read_text(), 5)}
    for name, found in tables.items():
        assert found, f"read no rows off {name}; its generated layout changed"
    lookup_of, subtractions = {}, []    # (difference, minuend, subtrahend)
    for label, (_, _, _, lookup, model, margin) in tables["tab:main panel a"]:
        lookup_of[label] = lookup[0]
        subtractions.append((margin[0], model[0], lookup[0]))
    for label, (_, _, model, excess) in tables["tab:main panel b"]:
        subtractions.append((excess[0], lookup_of[label], model[0]))
    for _, (_, _, _, blocked, excess) in tables["tab:ablation"]:
        prefix = re.match(r"htem(?:Res|Gap)", blocked[0]).group(0)
        subtractions.append((excess[0], f"{prefix}RandomLookup", blocked[0]))
    for difference, minuend, subtrahend in subtractions:
        unit = 10.0 ** -min(places(difference), places(minuend), places(subtrahend))
        gap = abs(value(difference) - (value(minuend) - value(subtrahend)))
        assert gap <= unit + 1e-12, (
            f"sec:statistics allows a difference of printed "
            f"values to differ in its last digit, but \\{difference}={macros[difference]} "
            f"against \\{minuend}={macros[minuend]} minus "
            f"\\{subtrahend}={macros[subtrahend]} differs by {gap:.4f}")


def test_the_interval_level_the_prose_prints_is_the_one_every_script_used(
        htem, margins, expt_gap, macros):
    """sec:methods and both float captions print the bootstrap level. It is a
    design constant fixed in the scripts, so it is read back from every artifact
    that carries one rather than typed. Two artifacts
    disagreeing would put one interval at a level the sentence does not name."""
    alphas = {"htem": htem["bootstrap"]["alpha"],
              "margins": margins["bootstrap"]["alpha"],
              "expt_gap": expt_gap["protocol"]["bootstrap"]["alpha"]}
    assert len(set(alphas.values())) == 1, (
        f"sec:methods prints one interval level; the artifacts carry {alphas}"
    )
    printed = number(macros, "bootstrapLevel")
    assert math.isclose(printed, 100 * (1 - alphas["htem"]), abs_tol=1e-9), (
        f"the manuscript prints a {printed:g}% interval; the scripts used "
        f"{100 * (1 - alphas['htem']):g}%"
    )


def test_the_manuscript_prints_no_ratio_of_two_measurements():
    """The defect this repository corrected once already: a multiple printed
    beside operands a reader can recombine on a different basis. The paper
    prints both quantities and no multiple, and this keeps it that way."""
    text = document_text()
    banned = re.findall(
        r"\b(\d+(?:\.\d+)?\s*(?:times|fold|x)\b|orders? of magnitude|"
        r"twice as|half as|factor of)", text, flags=re.IGNORECASE)
    assert not banned, (
        f"main.tex states a multiple: {banned}. Both quantities are printed "
        "instead, because the denominators here are intervals"
    )


# --------------------------------------------------------------------------
# 4. Verdicts -- every criterion the runs committed, including the misses
# --------------------------------------------------------------------------

# Where each committed criterion reaches the reader. The membership is derived
# from the artifacts below, never from this map; the map only says which
# sentence carries the comparison each one tests. The verdict strings are repo
# register and never cross into the manuscript, and
# neither does a pass or a fail: the sentence states the measured comparison.
# A criterion whose sentence is reworded fails here and is re-pointed in the
# same commit.
CRITERION_REPORTS = {
    # the lookup's own R^2 on their validation set, against N1's 0.95
    "N1": "separates the lookup's $R^2$ of \\npjLookupRSq\\ from their network's \\npjModelRSq",
    "H2": "is positive on all \\blockedPairs\\ pairs of HTEM-DB and EMPD",
    # H3 commits to reporting the blocked score as it falls, whatever it is
    "H3": "model's $R^2$ falls from \\htemResRandomModel\\ to "
          "\\htemResBlockedModel",
    "E1": "bootstrap intervals), below the \\htemGapRandomLookup",
    # E3 missed: the comparison on the rows whose gap exceeds zero is printed
    # as measured, and par:margin's reading is gone
    "E3": "exceeds the lower end of HTEM-DB band gap's interval",
    "C1": "also lies below half of its random-partition $R^2$, by \\htemResBeyondHalf",
    "C2": "so the lookup depends more than the model on training rows from the "
          "test row's own library",
    "C3": "the interval of the lookup's fall minus the model's has a lower "
          "endpoint of at least \\htemResFillFallLo",
    "T1": "Every interval for that excess excludes zero, including that of the "
          "band-gap excess",
    "T2": "each tuned model chooses in some folds a configuration on an edge of "
          "its grid",
    "E2": "The Matbench margin's interval lies above that of HTEM-DB band gap",
}

# Committed criteria whose threshold no claim in the manuscript compares
# against, each with the macros that print the values it was applied to. The
# verdict is not printed; every one of those values is, which the test below
# asserts. H1 fixed a tolerance on the model's margin relative to the lookup, on Kim
# and Lee's dataset and both HTEM-DB properties. The manuscript states those
# margins and does not claim the lookup matches the model within any
# tolerance, so the threshold would be printed only to be met or missed.
CRITERIA_REPORTED_AS_VALUES = {
    "H1": ("npjMargin", "htemResMargin", "htemGapMargin"),
}


def committed_criteria(*artifacts) -> dict:
    """{criterion: [verdict, ...]} over every criteria block the artifacts hold.

    Walked rather than read from known paths, so a criterion added under a new
    target, arm or dataset is found the day it is written.
    """
    found = {}

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "criteria" and isinstance(value, dict):
                    for name, c in value.items():
                        found.setdefault(name, []).append(
                            c.get("verdict") if isinstance(c, dict) else c)
                elif re.fullmatch(r"[A-Z]\d", key) and isinstance(value, dict):
                    # A criterion recorded beside the arm it judges, as
                    # reproduce/htem_ablation.py records T1 and T2.
                    found.setdefault(key, []).append(value)
                elif isinstance(value, (dict, list)):
                    walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    for artifact in artifacts:
        walk(artifact)
    return found


def test_every_committed_criterion_reaches_the_reader(npj, htem, margins,
                                                      expt_gap, ablation):
    """Every committed criterion's outcome reaches the reader, including the
    ones the results do not flatter. A criterion committed and
    never mentioned again is how a gate stops binding where no assertion can
    see it, and nothing else here reaches that failure: every other test asks
    whether a number the manuscript prints matches its artifact, which says
    nothing about a criterion the manuscript stopped printing."""
    committed = committed_criteria(npj, htem, margins, expt_gap, ablation)
    homes = set(CRITERION_REPORTS) | set(CRITERIA_REPORTED_AS_VALUES)
    twice = sorted(set(CRITERION_REPORTS) & set(CRITERIA_REPORTED_AS_VALUES))
    assert not twice, (
        f"criteria given both a sentence and a values-only home: {twice}"
    )
    orphaned = sorted(set(committed) - homes)
    invented = sorted(homes - set(committed))
    assert not orphaned and not invented, (
        f"criteria the runs committed with no sentence naming where they "
        f"reach the reader: {orphaned or 'none'}; sentences claiming a "
        f"criterion no artifact carries: {invented or 'none'}"
    )
    prose = " ".join(document_text().split())
    for name, sentence in CRITERION_REPORTS.items():
        assert " ".join(sentence.split()) in prose, (
            f"criterion {name} was committed and the artifacts report it as "
            f"{committed[name]}, but the manuscript no longer carries the "
            f"sentence that told the reader: {sentence!r}"
        )
    for name, names in CRITERIA_REPORTED_AS_VALUES.items():
        for macro in names:
            assert re.search(rf"\\{macro}(?![A-Za-z])", prose + table_bodies()), (
                f"criterion {name} reaches the reader only through the values "
                f"it was applied to, and \\{macro} is no longer printed"
            )


def test_h2_and_h3_verdicts(htem):
    """par:partition rests on H2; H3 is the criterion that keeps the blocked
    HTEM-DB column from being read as a collapse."""
    for name, t in htem["targets"].items():
        assert t["criteria"]["H2"]["verdict"] == "SPLIT MOVES IT MORE", (
            f"par:partition claims the partition moves the number further than "
            f"the model does; {name} reads {t['criteria']['H2']['verdict']!r}"
        )
        assert t["criteria"]["H3"]["verdict"] == "above the constant predictor", (
            f"par:partition reports HTEM-DB's blocked model above the constant "
            f"predictor; {name} reads {t['criteria']['H3']['verdict']!r}"
        )


def test_the_blocked_loss_exceeds_its_margin(htem):
    """abstract: 'On the one dataset that records each position's library,
    excluding libraries from training lowers the model's R^2 by more than the
    model gains over the lookup'. It states
    that the drop exceeds the margin, which H2 pins; the check reads it through
    sec:protocol's identity, under which the excess is the lookup's
    random-partition R^2 minus the model's blocked R^2."""
    for name, t in htem["targets"].items():
        blocked, lookup = t["blocked"]["model_r2"], t["random"]["lookup_r2"]
        assert blocked < lookup, (
            f"the abstract says excluding {name}'s libraries from training lowers "
            f"the model's R^2 by more than it gains over the lookup, so its blocked R^2 lies "
            f"below the lookup's random-partition R^2; it is "
            f"{blocked:+.4f} against {lookup:+.4f}"
        )


def test_n1_lands(npj):
    """The npj criterion: a lookup with no parameters fitted to the target lands
    at or above 0.95 in R^2 on the dataset's own published validation set."""
    n1 = npj["criteria"]["N1"]
    assert n1["verdict"] == "LANDS", f"N1 reads {n1['verdict']!r}"
    assert n1["best_lookup_r2"] >= n1["lands_at"]


def test_e1_met_the_lookup_falls_below_every_combinatorial_pair(
        expt_gap, npj, htem, margins):
    """sec:esi-draw-spread: 'On Matbench, the lookup scores 0.26 [0.19, 0.32]
    ..., below the 0.66 [0.56, 0.74] on HTEM-DB band gap.' The test compares
    intervals and ranges over HTEM-DB's two targets; the quantifier in
    par:margin's 'the lookup's mean R^2 is lower ... than on any of the five
    pairs' ranges over all five pairs' point scores."""
    e1 = expt_gap["criteria"]["E1"]
    assert e1["verdict"] == "BELOW THE COMBINATORIAL RANGE", (
        f"sec:esi-draw-spread reports the Matbench lookup interval below both HTEM-DB "
        f"lookup intervals; E1 reads {e1['verdict']!r}"
    )
    lows = {t["label"]: t["ci"]["random_lookup_r2"]["lo"]
            for t in htem["targets"].values()}
    lowest = min(lows, key=lows.get)
    assert e1["bound_by"] == lowest == "direct band gap", (
        "sec:esi-draw-spread sets the control against HTEM-DB band gap, the "
        f"property where the lookup scores lower on HTEM-DB; the artifacts make the "
        f"lower interval {lowest!r} and the test was scored against "
        f"{e1['bound_by']!r}"
    )
    control_hi = expt_gap["ci"]["random_lookup_r2"]["hi"]
    for label, lo in lows.items():
        assert control_hi < lo, (
            "sec:esi-draw-spread says the Matbench lookup interval lies below both "
            f"HTEM-DB lookup intervals; its upper endpoint {control_hi:.4f} "
            f"reaches {label}'s lower endpoint {lo:.4f}"
        )
    assert e1["threshold"] == lows[lowest], (
        "E1's threshold no longer matches outputs/htem/split.json: the control "
        "was scored against an earlier HTEM-DB run"
    )
    hi = expt_gap["ci"]["random_lookup_r2"]["hi"]
    assert hi < lows[lowest], (
        "sec:esi-draw-spread claims the control's lookup interval and HTEM-DB band "
        f"gap's do not overlap; the endpoints are {hi:.4f} and {lows[lowest]:.4f}"
    )
    for name, value in random_lookups(npj, htem, margins).items():
        assert hi < value, (
            "par:margin claims the control's lookup R^2 is lower than on "
            f"any of the five pairs, but {name}'s lookup is {value:.4f} against the "
            f"control's upper endpoint {hi:.4f}"
        )


def test_e2_separates_over_the_partitions_and_overlaps_on_the_first(
        expt_gap, npj, htem, margins):
    """sec:esi-draw-spread: 'The Matbench margin's interval lies above that of
    HTEM-DB band gap, the pair that Matbench matches (Table 1a of the main
    text). However, we first compared the two margins on a single random
    partition, the first of the ten. We moved the comparison to the means over
    the ten partitions only after the intervals of the two margins overlapped
    on that partition, so on Matbench the wider margin is not established.'
    The verdict rests on the two comparisons disagreeing, so both are pinned: a run that makes them agree fails here
    instead of quietly settling the question one way."""
    e2 = expt_gap["criteria"]["E2"]
    first = e2["on_first_partition"]
    assert e2["verdict"] == "ABOVE THE COMBINATORIAL RANGE", (
        f"sec:esi-draw-spread says the Matbench margin interval sits above HTEM-DB band "
        f"gap's over the ten partitions; E2 reads {e2['verdict']!r}"
    )
    assert first["verdict"] == "INSIDE THE COMBINATORIAL RANGE", (
        f"sec:esi-draw-spread says the two intervals overlap on the first partition; "
        f"there E2 reads {first['verdict']!r}, so that half of the sentence is "
        "the one to revisit"
    )
    for key, entry, source in ((("model_margin"), e2, "the ten partitions"),
                               (("first_partition_margin"), first,
                                "the first partition")):
        highs = {t["label"]: t["ci"][key]["hi"] for t in htem["targets"].values()}
        widest = max(highs, key=highs.get)
        assert entry["bound_by"] == widest == "direct band gap", (
            f"par:margin sets the control against HTEM-DB band gap; over "
            f"{source} the widest combinatorial margin interval is {widest!r} "
            f"and the control was scored against {entry['bound_by']!r}"
        )
        assert entry["threshold"] == highs[widest], (
            "E2's threshold no longer matches outputs/htem/split.json: the "
            "control was scored against an earlier HTEM-DB run"
        )
    lo = expt_gap["ci"]["model_margin"]["lo"]
    hi = htem["targets"]["direct_band_gap"]["ci"]["model_margin"]["hi"]
    assert lo > hi, (
        "par:margin says the two margin intervals over the ten partitions do "
        f"not overlap; the control's lower endpoint is {lo:.4f} against "
        f"{hi:.4f}"
    )
    first_lo = expt_gap["ci"]["first_partition_margin"]["lo"]
    first_hi = htem["targets"]["direct_band_gap"]["ci"]["first_partition_margin"]["hi"]
    assert first_lo <= first_hi, (
        "par:margin says the two intervals overlap on the first partition; "
        f"the control's lower endpoint is {first_lo:.4f} against {first_hi:.4f}"
    )
    # The pair the sentence names is the pair the criterion was scored
    # against, and it is also the widest of the five.
    pairs = random_margins(npj, htem, margins)
    assert max(pairs, key=pairs.get) == "HTEM-DB band gap"


def test_each_aihub_blocked_mean_is_placed_against_the_constant_predictor(margins):
    """sec:esi-draw-spread: 'The model's mean blocked R^2 and its interval both
    lie below zero on hardness, so, on average over the ten partitions, the
    model predicts the hardness of the alloy systems excluded from training
    less accurately than a constant prediction of the test rows' mean hardness.
    On resistivity, the model's mean blocked
    R^2 lies above zero, but its interval spans zero'."""
    tasks = margins["datasets"]["aihub"]["tasks"]
    hard = tasks["hardness"]["blocked_model_r2"]
    assert hard["mean"] < 0 and hard["ci_hi"] < 0, (
        "sec:esi-draw-spread says EMPD hardness's blocked mean and its interval both "
        f"lie below the constant predictor; the mean is {hard['mean']:+.4f} and "
        f"the interval [{hard['ci_lo']:+.4f}, {hard['ci_hi']:+.4f}]"
    )
    res = tasks["resistivity"]["blocked_model_r2"]
    assert res["mean"] > 0 and res["ci_lo"] < 0 < res["ci_hi"], (
        "sec:esi-draw-spread says EMPD resistivity's blocked mean lies above the "
        "constant predictor and its interval spans it; the mean is "
        f"{res['mean']:+.4f} and the interval [{res['ci_lo']:+.4f}, "
        f"{res['ci_hi']:+.4f}]"
    )


def test_replicate_interval_endpoints_stay_out_of_the_prose():
    """A percentile bootstrap over ten partitions covers well below its
    stated level, so its endpoints stay in the float, and the prose states the
    EMPD verdicts from the per-partition values instead. HTEM-DB intervals
    resample libraries and the control's resample compounds
    -- evaluation units -- and may be quoted."""
    body = re.sub(r"\\begin\{(table|figure)\*?\}.*?\\end\{\1\*?\}", "",
                  document_text(), flags=re.S)
    replicate_cis = ["empdResMarginCI", "empdHardMarginCI", "empdResBlockedCI",
                     "empdHardBlockedCI", "empdResExcessCI", "empdHardExcessCI",
                     "empdResBlockedLookupCI", "empdHardBlockedLookupCI"]
    leaked = [m for m in replicate_cis if re.search(rf"\\{m}(?![a-zA-Z])", body)]
    assert not leaked, (
        f"prose quotes endpoints estimated over ten partition draws: {leaked}. "
        "They belong in Table 1"
    )


def test_no_timing_claim_stands_in_for_evidence():
    """The commit order behind these criteria is not inspectable once the
    upstream repository is deleted, so the manuscript states each criterion and
    both verdicts and never claims when it was fixed."""
    text = document_text().lower()
    banned = ["pre-registered", "preregistered", "pre-specified", "a priori",
              "fixed before the run", "committed before", "in advance of the run"]
    found = [phrase for phrase in banned if phrase in text]
    assert not found, (
        f"main.tex claims a decision's timing as evidence: {found}. "
        "A term of art names an institution, not a practice"
    )


# --------------------------------------------------------------------------
# 5. Names that only get cited -- nothing dereferences these at run time
# --------------------------------------------------------------------------


def cited_text() -> str:
    """Every repo file that can cite a name, taken from version control:
    tracked files and untracked files git does not ignore. An ignored
    directory holds dated records of the tree as it stood, which cite what
    existed then."""
    listed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT, check=True, capture_output=True, text=True).stdout.split("\n")
    parts = []
    for name in listed:
        path = ROOT / name
        if path.is_file() and (path.suffix in {".py", ".md", ".tex"}
                               or path.name == ".gitignore"):
            parts.append(path.read_text(errors="ignore"))
    return "\n".join(parts)


def test_every_cited_slug_resolves():
    """A \\label slug cited in a docstring or a failure message is a pointer no
    program dereferences: the typesetter reports a dangling \\ref, nothing
    reports a dangling quote."""
    defined = set(re.findall(r"\\label\{([^}]+)\}", document_text()))
    cited = set(re.findall(r"\b(?:sec|tab|fig|par|eq):[a-zA-Z_-]+", cited_text()))
    dangling = sorted(cited - defined)
    assert not dangling, (
        f"repo files cite labels the manuscript does not define: {dangling}"
    )


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()


def test_the_release_holds_the_uploaded_tree_without_the_working_history():
    """Data availability and Code availability name one repository, the
    release. Its latest commit carries the branch's tree at the last upload to
    the journal, the commit that HEAD shares with the latest tag under upload/,
    or HEAD's tree before the first upload, so a referee finds there what the
    statements describe and none of the files that only the upload sent, such
    as the cover letter. No commit of the release is a commit of the working
    repository, and the tree holds nothing under _internal/, because a release
    that shares the working repository's history publishes every commit
    message and every file that the working repository ever tracked. The test
    fetches the release, so it fails until the release exists and wherever the
    network cannot reach it."""
    urls = set(re.findall(r"\\url\{(https://github\.com/[^}]+)\}",
                          MAIN.read_text()))
    assert len(urls) == 1, f"the availability statements name {sorted(urls)}"
    url = urls.pop()
    head = subprocess.run(["git", "ls-remote", url, "HEAD"], cwd=ROOT,
                          capture_output=True, text=True, timeout=60,
                          env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
    assert head.returncode == 0 and head.stdout.strip(), (
        f"the availability statements name {url}, which holds no commit: "
        f"{head.stderr.strip()}")
    release = head.stdout.split()[0]
    git("fetch", "--quiet", url, release)
    uploads = git("tag", "--list", "upload/*", "--sort=-creatordate").split()
    # The tag names the commit of the upload, which can hold the uploaded
    # files on a commit off the branch; the release carries the branch's tree
    # where the upload left it.
    uploaded = git("merge-base", uploads[0], "HEAD") if uploads else "HEAD"
    assert git("rev-parse", f"{release}^{{tree}}") == git(
        "rev-parse", f"{uploaded}^{{tree}}"), (
        f"{url} does not hold the tree of {uploaded}")
    shared = set(git("rev-list", release).split()) & set(
        git("rev-list", "HEAD").split())
    assert not shared, (
        f"{url} carries commits of the working repository: {sorted(shared)}")
    internal = [p for p in git("ls-tree", "-r", "--name-only", release).split()
                if p.startswith("_internal/")]
    assert not internal, f"{url} publishes {internal}"


def test_every_pointer_between_the_two_documents_resolves():
    """The article and its supplementary information cannot \\ref across the
    document boundary, so every pointer between them is a hand-typed name and
    nothing dereferences it. Both sides are derived: the supplementary information's numbering
    from the order its own sectioning commands appear, the article's float
    count from its floats."""
    esi = next(p for p in documents() if p.name == "esi.tex")
    body = esi.read_text()
    defined = {f"Section~S{i}" for i, _ in
               enumerate(re.findall(r"\\section\{", body), start=1)}
    for env, word in (("figure", "Fig.~S"), ("table", "Table~S")):
        n = len(re.findall(rf"\\begin\{{{env}\}}", body))
        defined |= {f"{word}{i}" for i in range(1, n + 1)}
    main = " ".join(MAIN.read_text().split())
    cited = (set(re.findall(r"Supplementary Information (Section~S\d+)", main))
             | set(re.findall(r"\b((?:Fig\.|Table)~S\d+)", main)))
    assert cited, "main.tex points at no supplementary material"
    dangling = sorted(cited - defined)
    assert not dangling, (
        f"main.tex points at supplementary material that does not exist: "
        f"{dangling}; the supplementary information defines {sorted(defined)}"
    )

    floats = len(re.findall(r"\\begin\{table\*?\}", MAIN.read_text()))
    back = sorted(set(re.findall(r"Table~(\d+)[a-z]? of the "
                                 r"(?:article|main text)", body)))
    unresolved = [n for n in back if int(n) > floats]
    assert not unresolved, (
        f"esi.tex points at Table {unresolved} of an article carrying "
        f"{floats} table(s)"
    )


def test_every_cited_script_path_exists():
    """The same rule for paths. Names carrying a retired tree's own prefix are
    citations into that tree, not into this one, and are excluded by that
    prefix rather than by a hand-kept list."""
    text = cited_text()
    cited = set(re.findall(r"\b(?:reproduce|src/metal)/[a-z_]+\.py", text))
    missing = sorted(p for p in cited if not (ROOT / p).exists())
    assert not missing, f"repo files cite scripts that do not exist: {missing}"


def test_every_table_the_document_inputs_is_one_the_generator_writes():
    """The same rule one directory over: a table body committed where the
    document reads it, and written by nobody, publishes an older run with every
    check green."""
    inputs = re.findall(r"\\input\{(tables/[\w/-]+)\}", document_text())
    generator = (ROOT / "reproduce" / "make_parameters.py").read_text()
    assert inputs, "main.tex inputs no table body"
    for rel in inputs:
        path = PAPER / f"{rel}.tex"
        assert path.exists(), f"main.tex inputs {rel}, which does not exist"
        assert f'"{path.name}"' in generator, (
            f"main.tex inputs {path.name}, which "
            "reproduce/make_parameters.py does not write: the document is "
            "reading a copy"
        )


def test_the_figure_the_document_includes_is_the_one_the_generator_writes():
    """A second copy of a generated artifact publishes an older run with every
    check green. The document reads the generator's own
    destination, so there is nothing to sync. Scoped to figures/, the directory
    the generator writes, the way the table check above is scoped to tables/."""
    included = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{(figures/[^}]+)\}",
                          document_text())
    generator = (ROOT / "reproduce" / "make_figure.py").read_text()
    assert included, "main.tex includes no figure"
    for name in included:
        path = (PAPER / name).with_suffix(".pdf")
        assert path.exists(), f"main.tex includes {name}, which does not exist"
        assert path.name in generator, (
            f"main.tex includes {path.name}, which reproduce/make_figure.py "
            "does not write: the document is reading a copy"
        )


def test_every_number_the_readme_restates_is_a_number_the_paper_prints(macros):
    """README.md restates Table 1, the margin ranges and the control comparison
    by hand. No generator writes it, so a regenerated result would leave the
    README printing the old run with every other check green. Each decimal
    the README prints is compared with the values that
    paper/parameters.tex defines; the README's other numbers are counts and
    versions that the list below names, each with the reason it is not a
    macro."""
    def numbers(text):
        text = (text.replace("{,}", ",").replace("$-$", "-").replace("−", "-")
                .replace("\\%", "%"))
        return {m.group(0).lstrip("+-").replace(",", "")
                for m in re.finditer(r"[+-]?\d[\d,]*\.\d+|[+-]?\d{1,3}(?:,\d{3})+", text)}
    printed = set()
    for value in macros.values():
        printed |= numbers(value)
    not_results = {
        "3.12",   # the Python version that the README's install line names
    }
    readme = (ROOT / "README.md").read_text()
    stale = sorted(numbers(readme) - printed - not_results)
    assert not stale, (
        f"README.md prints numbers that no macro of paper/parameters.tex holds: "
        f"{stale}. Copy them from the regenerated parameters, or name them here "
        f"with the reason they are not results")


def test_each_clause_of_the_later_runs_states_what_its_interval_decides(htem, expt_gap, ablation, macros):
    """par:partition: 'The lookup's R^2 on HTEM-DB falls further than the
    model's on both properties' and 'The lookup's blocked-partition R^2 on
    resistivity also lies below half of its random-partition R^2 ... On band gap, the lookup's blocked-partition R^2 lies below half of its
    random-partition R^2 by +0.16 [-0.01, +0.37], an interval that includes
    zero'; par:partition: 'so the lookup depends more than the model on
    training rows from the test row's own library'; sec:features: 'Under each of
    these values, the interval of the lookup's fall minus the model's has a
    lower endpoint of at least +0.037 on resistivity and +0.018 on band gap';
    sec:esi-draw-spread: 'The upper end of that interval, 0.562, exceeds the
    lower end of HTEM-DB band gap's interval, 0.559'; and
    sec:esi-posthoc: 'Every interval for that excess excludes zero'. Each
    clause is the verdict of a criterion committed before its run (C1, C2, C3
    in reproduce/htem_split.py, E3 in reproduce/expt_gap_control.py, T1 in
    reproduce/htem_ablation.py), asserted here on the interval that decided it,
    so a re-run that moves an endpoint across zero fails the sentence."""
    for key, t in htem["targets"].items():
        c = t["criteria"]
        assert c["C2"]["ci"]["lo"] > 0, (
            f"par:partition says the lookup falls further than the model on {key}; "
            f"the interval of the difference is {c['C2']['ci']}")
        for fill, f in t["no_temperature_sensitivity"]["alternatives"].items():
            assert f["ci"]["lookup_fall_minus_model_fall"]["lo"] > 0, (
                f"sec:features says the comparison holds under every fill; with "
                f"{fill} on {key} the interval is {f['ci']['lookup_fall_minus_model_fall']}")
    res, gap = (htem["targets"][k]["criteria"]["C1"]["ci"]
                for k in ("log10_resistivity", "direct_band_gap"))
    assert res["lo"] > 0, f"par:partition says more than half on resistivity; {res}"
    assert gap["lo"] <= 0 <= gap["hi"], (
        f"par:partition says the band-gap interval includes zero; it is {gap}")
    nonzero = expt_gap["nonzero_gap"]["ci"]["random_lookup_r2"]
    band_gap = htem["targets"]["direct_band_gap"]["ci"]["random_lookup_r2"]
    assert nonzero["hi"] > band_gap["lo"], (
        f"sec:esi-draw-spread says the upper end of the nonzero-gap Matbench interval "
        f"{nonzero} exceeds the lower end of HTEM-DB band gap's {band_gap}")
    # The two endpoints meet at the intervals' two printed decimals, so the
    # sentence prints them at three, where the comparison it states shows.
    hi, lo = number(macros, "exptGapNonzeroLookupHi"), number(macros, "htemGapRandomLookupLo")
    assert math.isclose(hi, nonzero["hi"], abs_tol=0.0005) and math.isclose(lo, band_gap["lo"], abs_tol=0.0005), (
        f"sec:esi-draw-spread prints the endpoints {hi} and {lo}; the intervals are {nonzero} and {band_gap}")
    assert hi > lo, (
        f"sec:esi-draw-spread says the printed upper end {hi} exceeds the printed lower end {lo}")
    for key, t in ablation["targets"].items():
        for arm, entry in t["arms"].items():
            assert entry["ci"]["lo"] > 0, (
                f"sec:esi-posthoc says every interval for the excess excludes zero; "
                f"{arm} on {key} gives {entry['ci']}")


def test_each_count_prints_its_own_unit_noun():
    """A count prints its own population's unit noun, so a sentence that prints
    a count beside another population's noun is a crossing that a reader would
    otherwise have to catch. Each count macro's unit is read from its
    declaration in paper/parameters.tex, the line's own where it gives one and
    its block's otherwise, and set against the noun that the manuscript prints
    after the macro, past the modifiers that name which members ("random
    partitions", "alloy systems"). The row counts of Kim and Lee's dataset and of
    EMPD print "positions" where the sentence also says "one row per position",
    which makes the two nouns one unit there."""
    declared, block = {}, None
    for line in PARAMETERS.read_text().splitlines():
        m = re.match(r"%% POPULATION (\w+) \| artifact: (.+?) \| unit: (\w+)$", line)
        if m:
            block = m.group(3)
            continue
        m = re.match(r"\\newcommand\{\\(\w+)\}\{(.*?)\}(?: % artifact: .+? \| unit: (\w+))?$", line)
        if m and block:
            declared[m.group(1)] = (m.group(2), m.group(3) or block)
    spelled = set(WORDS)
    counts = {name: unit for name, (value, unit) in declared.items()
              if re.fullmatch(r"\d+", value.replace("{,}", "")) or value in spelled}
    modifiers = {"random", "blocked", "combinatorial", "validation", "alloy",
                 "crystal", "cross-validation", "decision", "inner", "hardness",
                 "resistivity", "Matbench", "control"}
    nouns = {"rows", "positions", "systems", "libraries", "features", "descriptors",
             "elements", "partitions", "pairs", "datasets", "structures", "compounds",
             "sources", "epochs", "units", "trees", "folds", "configurations",
             "resamples", "times", "decimals", "rounds", "files", "models", "tests"}
    text = " ".join(re.sub(r"(?<!\\)%.*", "", (PAPER / name).read_text())
                    for name in ("main.tex", "esi.tex"))
    text = " ".join(text.split())
    crossings = []
    for name, unit in counts.items():
        for m in re.finditer(rf"\\{name}(?![A-Za-z])\\? ?((?:[A-Za-z-]+ ?){{1,4}})", text):
            words = m.group(1).split()
            while words and words[0] in modifiers:
                words.pop(0)
            if not words or words[0] not in nouns:
                continue
            head = words[0]
            same = head == unit or {head, unit} == {"rows", "positions"}
            if not same and not (head == "tests" and unit == "pairs"):
                crossings.append(f"\\{name} ({unit}) printed before '{head}'")
    assert not crossings, (
        "counts printed beside another population's noun: " + "; ".join(crossings))
