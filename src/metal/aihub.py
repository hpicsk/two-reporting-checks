"""EMPD -- the Experimental Materials Property Database: tables and partitions.

The archive lookup, the three property tables over every split the archive
ships, their union, and the random and blocked partitions of that union
(sec:protocol). Every lookup and network score
runs within one property, but both partitions are drawn over all three tables
together, so the amorphous rows help decide which resistivity and hardness rows
go to the test set, although their label is out of scope.
"""

from __future__ import annotations

import os
import re
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# The archive is not redistributable, so this repository ships no copy and has no
# path worth guessing. A hard lookup: a machine without the variable gets the
# variable's name, not a stranger's home directory that happens to be absent.
# Resolved at call time, so importing this module for one pure helper does not
# require the dataset.
_ARCHIVE_VAR = "METAL_AIHUB_PATH"

# The labelled-data ZIP of each split the archive ships, found rather than
# listed. The Number column runs on from one split into the next, so the splits
# are parts of one release, and each property table is its rows in every split.
LABELLED_ZIPS = "*/02.라벨링데이터/*.zip"

# The three property tables the archive actually ships, by their filename inside
# each labelled-data ZIP.
CSV_FILES = {
    "resistivity": "Resistivity data set.csv",
    "amorphous": "Amorphous data set.csv",
    "hardness": "Hardness data set.csv",
}

TARGET_COLS = {
    "resistivity": "Resistivity",
    "amorphous": "Amorphous",
    "hardness": "Hardness",
}

ELEMENT_COLS = ["Al", "Ti", "Cr", "Fe", "Co", "Ni", "Cu", "Zr",
                "Mo", "W", "Mn", "Si", "Mg", "Re", "Ta"]

# A "Compo" token that names no element, and the element its rows carry. The
# Training amorphous table writes "Al/Cu/Ni/Zi" on rows whose Zr column is
# nonzero; reproduce/aihub_comeasurement.py resolves the token from those
# columns independently. Any other token outside ELEMENT_COLS raises.
COMPO_TOKEN_FIXES = {"Zi": "Zr"}

# The six descriptors every table ships beside the element percentages, each
# computed from composition. A table that records no percentage for an element
# carries 0 for it in the union.
DESCRIPTOR_COLS = ["ravg", "delta", "dHmix", "ENavg", "dEN", "N"]
# The processing variable every table records on every row: film thickness.
PROCESSING_COLS = ["Thickness"]
# The features of both predictors (sec:features).
FEATURE_COLS = ELEMENT_COLS + DESCRIPTOR_COLS + PROCESSING_COLS

# A row's alloy system, as system_name() gives it. EMPD's tables name no
# library, so the blocked partition keeps each system whole in one part, and
# with each system every library deposited from it (sec:protocol).
SYSTEM_COL = "system"

# The three properties in the order the stratification key reads them.
STRATA = ("resistivity", "hardness", "amorphous")

# Random partition: CHOSEN, 70/15/15 training/early-stopping/test, stratified
# by table so each part holds each property in proportion (sec:protocol).
# HELD_OUT_SHARE is the test set's share and VALIDATION_SHARE the early-stopping
# rows' share; reproduce/make_parameters.py reads both names.
HELD_OUT_SHARE = 0.15
VALIDATION_SHARE = 0.15

# Blocked partition: CHOSEN, the share of each property's rows each part aims
# at, the key "val" naming the early-stopping rows. Whole systems are placed one
# at a time, so the realized shares vary with the systems a seed draws
# (sec:protocol prints the range of test shares).
BLOCKED_SHARE = {"train": 0.70, "val": 0.15, "test": 0.15}

# R^2 is undefined on fewer than two rows. A blocked part carrying fewer rows of
# a property is a fault in the assignment, not a partition to drop.
MIN_ROWS = 2

# EMPD hardness is nanoindentation hardness in GPa: metal-alloy nanoindentation
# values are O(10) GPa, whereas a Vickers HV scale would be O(10^2-10^3). A loaded
# maximum at or above this ceiling signals a units regression, not real hardness.
HARDNESS_GPA_CEILING = 100.0


def archive_path() -> Path:
    """The local AI Hub archive, named by the environment."""
    configured = os.environ.get(_ARCHIVE_VAR)
    if not configured:
        raise RuntimeError(
            f"{_ARCHIVE_VAR} is not set. Point it at your local copy of AI "
            f"Hub's 'Experimental Materials Property Database', at its "
            f"delivered open-data directory "
            f"(.../01-1.정식개방데이터; the numeric folder prefix AI Hub "
            f"assigns is a catalogue position, not a stable identifier). The "
            f"archive is not redistributable, so this repository ships no "
            f"copy of it."
        )
    path = Path(configured)
    if not path.exists():
        raise FileNotFoundError(
            f"{_ARCHIVE_VAR} names {path}, which does not exist."
        )
    return path


def _table_key(name: str) -> str:
    """A table's file name with whitespace removed. The Validation ZIP names the
    resistivity table "Resistivity data set .csv", with a space before the
    extension that the Training ZIP's name lacks."""
    return re.sub(r"\s+", "", name.rsplit("/", 1)[-1])


def load(property_name: str) -> pd.DataFrame:
    """One property table, its rows in every split stacked in split order.

    Raises if a split does not hold the table, if the splits' columns differ,
    or if a row carrying data has no "Compo". A row whose every cell is empty is
    not a measurement and is dropped; the Validation hardness table carries such
    rows.
    """
    if property_name not in CSV_FILES:
        raise KeyError(
            f"{property_name!r} is not one of the three properties this archive "
            f"ships: {sorted(CSV_FILES)}"
        )
    zips = sorted(archive_path().glob(LABELLED_ZIPS))
    if not zips:
        raise FileNotFoundError(f"{archive_path()} holds no {LABELLED_ZIPS}")
    wanted = _table_key(CSV_FILES[property_name])

    parts = []
    for zip_path in zips:
        with zipfile.ZipFile(zip_path, "r") as zf:
            matching = [f for f in zf.namelist() if _table_key(f) == wanted]
            if len(matching) != 1:
                raise FileNotFoundError(
                    f"{zip_path} holds {len(matching)} tables named "
                    f"{CSV_FILES[property_name]!r}")
            with zf.open(matching[0]) as fh:
                part = pd.read_csv(BytesIO(fh.read())).dropna(how="all")
        if parts and list(part.columns) != list(parts[0].columns):
            raise ValueError(f"{zip_path}'s {property_name} table has other "
                             f"columns than {zips[0]}'s")
        if part["Compo"].isna().any():
            raise ValueError(f"{zip_path}: {int(part['Compo'].isna().sum())} "
                             f"{property_name} rows carry data but no Compo")
        parts.append(part)
    df = pd.concat(parts, ignore_index=True)

    if property_name == "hardness":
        h_max = float(df[TARGET_COLS["hardness"]].max())
        if h_max >= HARDNESS_GPA_CEILING:
            raise ValueError(
                f"EMPD hardness loaded with maximum {h_max:.1f}, at or above "
                f"the {HARDNESS_GPA_CEILING:.0f} GPa sanity ceiling. This target "
                f"is nanoindentation hardness in GPa; a value this large "
                f"indicates an HV/Vickers scale."
            )
    return df


def load_all() -> dict[str, pd.DataFrame]:
    """All three property tables. Any failure raises rather than warning."""
    return {name: load(name) for name in CSV_FILES}


def system_name(compo) -> str:
    """A "Compo" entry as its alloy system: the elements in alphabetical order.
    Two resistivity libraries are named "Zr/Cu/Al" and "Al/Cu/Zr", and two
    "Cu/Zr/Ti" and "Zr/Cu/Ti"; each pair is one system. A token is first
    corrected by COMPO_TOKEN_FIXES, and one that still names no element raises."""
    if pd.isna(compo) or compo == "":
        return ""
    tokens = [COMPO_TOKEN_FIXES.get(t, t) for t in str(compo).split("/")]
    unknown = [t for t in tokens if t not in ELEMENT_COLS]
    if unknown:
        raise ValueError(f"Compo {compo!r} names {unknown}, which are not "
                         f"element symbols")
    return "/".join(sorted(tokens))


def union() -> pd.DataFrame:
    """The three tables stacked in CSV_FILES order, one property per row.

    ``has_<property>`` marks the property a row carries. A feature or target a
    table does not record is 0 on its rows; the mask keeps a missing target out
    of every fit and score.
    """
    parts = []
    for name, df in load_all().items():
        df = df.copy()
        df[SYSTEM_COL] = df["Compo"].apply(system_name)
        for task in CSV_FILES:
            df[f"has_{task}"] = task == name
        for column in TARGET_COLS.values():
            if column not in df.columns:
                df[column] = 0.0
        parts.append(df)
    rows = pd.concat(parts, ignore_index=True)
    for column in FEATURE_COLS + list(TARGET_COLS.values()):
        rows[column] = rows[column].fillna(0)
    return rows


def random_partition(rows: pd.DataFrame, seed: int):
    """(training, early-stopping, test) parts of one random partition."""
    key = rows[f"has_{STRATA[0]}"].astype(str)
    for task in STRATA[1:]:
        key = key + "_" + rows[f"has_{task}"].astype(str)
    train_stop, test = train_test_split(rows, test_size=HELD_OUT_SHARE,
                                        stratify=key, random_state=seed)
    train, stop = train_test_split(
        train_stop, test_size=VALIDATION_SHARE / (1 - HELD_OUT_SHARE),
        stratify=key.loc[train_stop.index], random_state=seed)
    return tuple(part.reset_index(drop=True) for part in (train, stop, test))


def blocked_partition(rows: pd.DataFrame, seed: int):
    """(training, early-stopping, test) parts that each hold whole systems.

    The systems are shuffled by ``seed``, then placed one at a time into the
    part furthest below its share. A part's shortfall is that of its
    worst-served property, as a fraction of the property's rows, so a system
    carrying resistivity rows alone cannot fill a part while hardness is short.
    """
    rng = np.random.default_rng(seed)
    systems = np.array(sorted(rows[SYSTEM_COL].astype(str).unique()))
    rng.shuffle(systems)
    masks = [f"has_{task}" for task in STRATA]
    totals = {m: float(rows[m].sum()) for m in masks}
    have = {part: {m: 0.0 for m in masks} for part in BLOCKED_SHARE}
    assign = {}
    for system in systems:
        in_system = rows[rows[SYSTEM_COL].astype(str) == system]

        def shortfall(part):
            return max((BLOCKED_SHARE[part] * totals[m] - have[part][m]) / totals[m]
                       for m in masks)

        best = max(BLOCKED_SHARE, key=shortfall)
        assign[system] = best
        for m in masks:
            have[best][m] += float(in_system[m].sum())

    part_of = rows[SYSTEM_COL].astype(str).map(assign)
    parts = tuple(rows[part_of == part].reset_index(drop=True)
                  for part in BLOCKED_SHARE)
    for part, df in zip(BLOCKED_SHARE, parts):
        for m in masks:
            if int(df[m].sum()) < MIN_ROWS:
                raise ValueError(
                    f"blocked partition {seed}: the {part} part carries "
                    f"{int(df[m].sum())} rows of {m[4:]}, fewer than the "
                    f"{MIN_ROWS} an R^2 needs")
    return parts
