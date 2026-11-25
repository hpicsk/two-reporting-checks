"""
What EMPD's release holds beyond the counts its tables print, read from the
archive.

Co-measurement. EMPD's construction guideline says that measurements of
different kinds taken on one film at one composition are written into one label
(its section on the labelling procedure). This checks that against the
delivered archive by matching on the position the archive itself ships
(Compo system name + X + Y).

Note on the key: resistivity carries ~1.7 rows per (system, X, Y), so several
libraries of one system exist and a position match alone could be a position on
another library of that system rather than the same position. For amorphous
that ambiguity is resolved -- the amorphous table also ships a Resistivity
column, so matched rows are compared on the measured VALUE. For hardness no shared measured column exists, so its overlap
is reported as a coordinate match and labelled as the weaker evidence it is.

The co-measurement counts enter no score: every lookup and every network is
fitted and scored on one property's own labelled rows, so the dataset's label
structure never enters.

The release beyond its tables (sec:datasets). The raw files keep each
library's resistance, composition and thickness in files of their own, which
library_files() counts. split_order() gives the
order the shipped training and validation parts follow; resistance_factor()
and columns() what the tables carry beside their targets.
npj_shared_measurements() compares Kim and Lee's rows with the raw
resistances. LONG_DECIMALS and FACTOR_GRID are CHOSEN, each with its reason
beside it. Nothing here is FITTED.
"""
from __future__ import annotations

import json
import re
import sys
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from metal.aihub import (CSV_FILES, ELEMENT_COLS, LABELLED_ZIPS, _table_key,  # noqa: E402
                         archive_path, load_all, system_name)

OUT = ROOT / "outputs" / "aihub" / "structure.json"
# A clone has no archive, so every EMPD number reaches the manuscript through
# the committed outputs/aihub/draws.json. This asserts that the row and system
# counts reproduce/empd_draws.py recorded there are the ones this script
# recovers from the archive's composition column, so the two cannot drift apart
# unnoticed.
EXTRACT = ROOT / "outputs" / "aihub" / "draws.json"


def census() -> dict:
    """Rows and alloy systems per property, over every split the archive ships.

    Counted apart from src/metal/aihub.py's loader, so test_assertions.py can
    check the loader's counts against a second reading of the archive. The
    splits are found by their labelled-data archives rather than listed. A
    table is matched by its file name with whitespace removed, because one
    split names a table with a space the other split's name lacks. A row whose
    every cell is empty is not a measurement. A composition token that is not
    an element symbol is resolved from its own rows' element columns: the one
    element carried by every such row that the entry's other tokens do not name.
    """
    squeeze = lambda name: re.sub(r"\s+", "", name)
    wanted = {task: squeeze(name) for task, name in CSV_FILES.items()}
    splits, resolved = [], {}
    rows = {task: 0 for task in CSV_FILES}
    found = {task: set() for task in CSV_FILES}
    for archive in sorted(archive_path().glob("*/02.라벨링데이터/*.zip")):
        splits.append(archive.parent.parent.name)
        with zipfile.ZipFile(archive) as zf:
            for task, key in wanted.items():
                members = [m for m in zf.namelist() if squeeze(Path(m).name) == key]
                if len(members) != 1:
                    raise FileNotFoundError(
                        f"{archive} holds {len(members)} tables named like "
                        f"{CSV_FILES[task]!r}")
                df = pd.read_csv(BytesIO(zf.read(members[0]))).dropna(how="all")
                if df["Compo"].isna().any():
                    raise ValueError(f"{archive}: a {task} row with data has no Compo")
                rows[task] += len(df)
                columns = [c for c in ELEMENT_COLS if c in df.columns]
                for compo, group in df.groupby(df["Compo"].astype(str).str.strip()):
                    tokens = compo.split("/")
                    for token in [t for t in tokens if t not in ELEMENT_COLS]:
                        carried = {c for c in columns if (group[c].fillna(0) > 0).all()}
                        unnamed = carried - set(tokens)
                        if len(unnamed) != 1:
                            raise ValueError(
                                f"{archive}: {task} Compo {compo!r} carries the "
                                f"non-element {token!r}, and its rows' element "
                                f"columns name {sorted(unnamed)} beside the rest")
                        element = unnamed.pop()
                        if resolved.setdefault(token, element) != element:
                            raise ValueError(
                                f"{token!r} resolves to {resolved[token]} in one "
                                f"table and to {element} in another")
                        tokens = [element if t == token else t for t in tokens]
                    found[task].add("/".join(sorted(tokens)))
    return {"splits": splits, "rows": rows,
            "systems": {task: len(s) for task, s in found.items()},
            "resolved_tokens": resolved}


# The release beyond its labelled tables: the raw files each library's
# measurements sit in, the order its shipped split follows, and what its tables
# carry without stating. Every function reads the archive afresh.
SOURCE_ZIPS = "*/01.원천데이터/*.zip"
# Raw-file names: a resistance file is <sample>-<n>-Resistance-<label>.csv,
# an EDS or thickness file <label>_<sample>.csv. Each label names a library's
# elements.
RESISTANCE_FILE = re.compile(r"(\d+)-(\d+)-Resistance-(.+)\.csv")
SAMPLE_FILE = re.compile(r"(.+)_(\d+)\.csv")
# Kim and Lee's two released files, the rows compared with the raw
# resistance files.
NPJ_FILES = [ROOT / "data" / "npj" / "csv" / name for name in ("Train.csv", "Val.csv")]
# CHOSEN: only raw values recorded with at least this many decimals, at one
# position in the whole release, are compared. Most raw values carry three
# decimals, and with four decimals allowed every factor on the grid already
# matches thousands of Kim and Lee's rows, so a short value recurs by chance.
LONG_DECIMALS = 6
# CHOSEN: the conversion factors scanned, on the scale at which resistivity in
# micro-ohm cm is factor x resistance in ohms x thickness in nm. The range
# brackets pi/ln 2 / 10, the four-point probe's factor, by about 15% each way.
FACTOR_GRID = (0.38, 0.52, 1e-5)
def _half_unit(text: str) -> float:
    """Half a unit in the last decimal a table prints for a value."""
    decimals = len(text.split(".")[1]) if "." in text else 0
    return 0.5 * 10.0 ** -decimals


def _sources(kind: str):
    """(split, file name, table) for every raw file of one measurement kind,
    read as printed."""
    for archive in sorted(archive_path().glob(SOURCE_ZIPS)):
        if kind not in archive.name:
            continue
        with zipfile.ZipFile(archive) as zf:
            for member in zf.namelist():
                table = pd.read_csv(BytesIO(zf.read(member)), dtype=str)
                table.columns = [c.strip() for c in table.columns]
                yield archive.parent.parent.name, member.strip("/"), table


def _labelled(task: str) -> pd.DataFrame:
    """One property table over every split, as printed, with its split."""
    parts = []
    for archive in sorted(archive_path().glob(LABELLED_ZIPS)):
        with zipfile.ZipFile(archive) as zf:
            member = [m for m in zf.namelist() if _table_key(m) == _table_key(CSV_FILES[task])][0]
            part = pd.read_csv(BytesIO(zf.read(member)), dtype=str).dropna(how="all")
        parts.append(part.assign(split=archive.parent.parent.name))
    return pd.concat(parts, ignore_index=True)


def library_files() -> dict:
    """The raw resistance files, one per library, and how many of those
    libraries also keep a composition and a thickness file."""
    resistance = list(_sources("Resistance"))
    sample_of = {}
    for _, name, _ in resistance:
        m = RESISTANCE_FILE.fullmatch(name)
        if m is None:
            raise ValueError(f"{name!r} is not <sample>-<n>-Resistance-<label>.csv")
        sample_of[name] = int(m[1])
    samples = set(sample_of.values())
    samples_with = {}
    for kind in ("EDS", "Thickness"):
        for _, name, _ in _sources(kind):
            m = SAMPLE_FILE.fullmatch(name)
            if m is None:
                raise ValueError(f"{name!r} is not <label>_<sample>.csv")
            samples_with.setdefault(kind, set()).add(int(m[2]))
    return {
        "resistance_files": len(sample_of),
        "resistance_libraries": len(samples),
        "resistance_libraries_with_eds_and_thickness_files":
            len(samples & samples_with["EDS"] & samples_with["Thickness"]),
    }


def split_order() -> dict:
    """Per table, the rows and the Number span of each shipped split, and how
    the alloy systems fall across the splits."""
    out = {}
    for task in CSV_FILES:
        t = _labelled(task)
        t = t.assign(Number=t.Number.astype(float), system=t.Compo.map(system_name))
        parts = {s: {"rows": len(g), "numbers": [int(g.Number.min()), int(g.Number.max())]}
                 for s, g in t.groupby("split")}
        where = t.groupby("system")["split"].agg(frozenset)
        out[task] = {"splits": parts,
                     "systems_in_one_split": {s: int((where == frozenset({s})).sum())
                                              for s in parts},
                     "systems_in_every_split": int((where.map(len) == len(parts)).sum())}
    return out


def resistance_factor() -> dict:
    """Resistivity over resistance times thickness, on every resistivity
    row."""
    t = _labelled("resistivity")
    ratio = t.Resistivity.astype(float) / (t.Resistance.astype(float) * t.Thickness.astype(float))
    return {"median": float(ratio.median()),
            "max_relative_deviation": float((ratio / ratio.median() - 1).abs().max())}


def columns() -> dict:
    """The columns each labelled table carries."""
    return {task: [c for c in _labelled(task).columns if c != "split"] for task in CSV_FILES}


def npj_shared_measurements() -> dict:
    """Kim and Lee's rows that reproduce a raw EMPD resistance.

    Kim and Lee's files give resistivity and thickness, and resistivity is a
    conversion factor times resistance times thickness. For each factor on
    FACTOR_GRID, an npj row's implied resistance is compared with every
    raw EMPD value recorded with LONG_DECIMALS or more decimals at one position
    only, and matches where it lies within half a unit of that value's last
    recorded decimal. A factor the two datasets share matches many rows; any
    other factor matches what chance leaves. The rows at the best factor are
    traced to their raw file, and Kim and Lee's thickness is compared with
    EMPD's at that position.
    """
    npj = pd.concat([pd.read_csv(f) for f in NPJ_FILES], ignore_index=True)
    raw, thickness = [], {}
    for _, name, table in _sources("Resistance"):
        sample = int(RESISTANCE_FILE.fullmatch(name)[1])
        raw.append(table.assign(file=name, sample=sample))
    for _, name, table in _sources("Thickness"):
        sample = int(SAMPLE_FILE.fullmatch(name)[2])
        for x, y, t in zip(table.X.astype(float), table.Y.astype(float),
                           table.Thickness.astype(float)):
            thickness[(sample, x, y)] = t
    raw = pd.concat(raw, ignore_index=True)
    raw = raw.assign(R=raw.Resistance.astype(float), X=raw.X.astype(float),
                     Y=raw.Y.astype(float), tol=raw.Resistance.map(_half_unit),
                     decimals=raw.Resistance.map(lambda s: len(s.split(".")[1]) if "." in s else 0))
    long = raw[(raw.decimals >= LONG_DECIMALS) & (raw.R.map(raw.R.value_counts()) == 1)]
    long = long.sort_values("R").reset_index(drop=True)
    values, tolerance = long.R.to_numpy(), long.tol.to_numpy()
    per_factor = (npj.resistivity / npj.t).to_numpy()

    def match(factor):
        implied = per_factor / factor
        i = np.clip(np.searchsorted(values, implied), 1, len(values) - 1)
        nearest = np.where(abs(values[i] - implied) < abs(values[i - 1] - implied), i, i - 1)
        return nearest, abs(values[nearest] - implied) <= tolerance[nearest] + 1e-15

    lo, hi, step = FACTOR_GRID
    factors = np.round(np.arange(lo, hi + step / 2, step), 5)
    counts = np.array([match(f)[1].sum() for f in factors])
    best = int(counts.argmax())
    others = np.delete(counts, best)
    nearest, ok = match(factors[best])
    hits = long.iloc[nearest[ok]]
    ratio = np.array([t / thickness.get((s, x, y), np.nan) for t, s, x, y in
                      zip(npj.t.to_numpy()[ok], hits["sample"], hits.X, hits.Y)])
    found = ratio[np.isfinite(ratio)]
    return {"npj_rows": len(npj),
            "raw_values_compared": len(long),
            "raw_libraries_compared": int(long["sample"].nunique()),
            "min_decimals": LONG_DECIMALS,
            "factor_grid": list(FACTOR_GRID),
            "best_factor": float(factors[best]),
            "rows_at_best_factor": int(counts[best]),
            "most_rows_at_another_factor": int(others.max()),
            "factor_of_most_rows_at_another": float(factors[np.delete(np.arange(len(factors)), best)][others.argmax()]),
            "median_rows_over_the_grid": float(np.median(counts)),
            "libraries_reached": {name: int(n) for name, n in hits.file.value_counts().items()},
            "rows_with_an_empd_thickness": len(found),
            "thickness_ratio_npj_to_empd": {"median": float(np.median(found)),
                                            "p5": float(np.percentile(found, 5)),
                                            "p95": float(np.percentile(found, 95))}}


d = load_all()
r, a, h = d["resistivity"], d["amorphous"], d["hardness"]


def coords(df):
    """The archive's own position key: system name plus stage position."""
    return list(zip(df.Compo.astype(str).str.strip(),
                    np.round(df.X.astype(float), 4),
                    np.round(df.Y.astype(float), 4)))


def systems(df):
    """The alloy systems a table spans, counted as element sets."""
    return set(df.Compo.map(system_name))


KR = set(coords(r))
KA = set(coords(a))
rmap: dict = {}
for k, v in zip(coords(r), r.Resistivity.astype(float)):
    rmap.setdefault(k, []).append(v)

sys_r, sys_a, sys_h = systems(r), systems(a), systems(h)

print(f"rows: resistivity {len(r):,}  amorphous {len(a)}  hardness {len(h)}")
print(f"systems: resistivity {len(sys_r)} / amorphous {len(sys_a)} / "
      f"hardness {len(sys_h)}, union {len(sys_r | sys_a | sys_h)}")
print(f"  hardness systems also in resistivity: {len(sys_h & sys_r)}   "
      f"hardness-only: {len(sys_h - sys_r - sys_a)}\n")

print("STRONG evidence -- shared coordinate AND identical measured value:")
hits = [(av, rmap[k]) for k, av in zip(coords(a), a.Resistivity.astype(float))
        if k in rmap]
exact = sum(min(abs(av - x) for x in rv) < 1e-6 for av, rv in hits)
print(f"  amorphous rows at a resistivity-measured coordinate : {len(hits)}/{len(a)}")
print(f"  ...of which the resistivity value is bit-identical  : {exact}/{len(hits)}")
print(f"  every amorphous row ships a Resistivity value       : "
      f"{a.Resistivity.notna().sum()}/{len(a)}\n")

print("WEAKER evidence -- shared coordinate only (no shared measured column):")
print(f"  hardness rows at a resistivity-measured coordinate  : "
      f"{sum(k in KR for k in coords(h))}/{len(h)}")
print(f"  hardness rows at an amorphous-measured coordinate   : "
      f"{sum(k in KA for k in coords(h))}/{len(h)}")

el = [c for c in ELEMENT_COLS if c in r.columns and c in h.columns and c in a.columns]
cr = set(map(tuple, r[el].round(6).to_numpy()))
ca = set(map(tuple, a[el].round(6).to_numpy()))
ch = list(map(tuple, h[el].round(6).to_numpy()))
print(f"  hardness rows whose exact composition is in resistivity: "
      f"{sum(c in cr for c in ch)}/{len(ch)}")
print(f"  hardness rows whose exact composition is in amorphous  : "
      f"{sum(c in ca for c in ch)}/{len(ch)}")
print(f"  (matched on {len(el)} shared element columns)")

print("\nVERDICT: the tables are substantially CO-MEASURED, as the construction "
      "guideline's labelling procedure says.")

structure = {
    "rows": {"resistivity": len(r), "amorphous": len(a), "hardness": len(h)},
    "systems": {"resistivity": len(sys_r), "amorphous": len(sys_a),
                "hardness": len(sys_h)},
    "systems_union": len(sys_r | sys_a | sys_h),
    "comeasurement": {
        "amorphous_rows_at_resistivity_coordinate": len(hits),
        "amorphous_rows_with_identical_resistivity": exact,
        "hardness_rows_at_resistivity_coordinate": sum(k in KR for k in coords(h)),
        "hardness_rows_sharing_a_resistivity_composition": sum(c in cr for c in ch),
    },
    "archive": census(),
    "release": {"library_files": library_files(), "split_order": split_order(),
                "resistance_factor": resistance_factor(),
                "columns": columns()},
    "npj_shared_measurements": npj_shared_measurements(),
}
print("\nRELEASE:")
print(json.dumps({key: structure[key] for key in
                  ("release", "npj_shared_measurements")}, indent=1))
carried = json.loads(EXTRACT.read_text())
for field in ("rows", "systems"):
    for task in ("resistivity", "hardness"):
        assert structure[field][task] == carried[field][task], (
            f"the archive has {structure[field][task]} {field} for {task}; "
            f"{EXTRACT.name}, which every EMPD number in the manuscript is "
            f"read from, carries {carried[field][task]}. Regenerate it with "
            f"reproduce/empd_draws.py.")
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(structure, indent=1) + "\n")
print(f"wrote {OUT.relative_to(ROOT)}; row and system counts agree with "
      f"{EXTRACT.relative_to(ROOT)}")
