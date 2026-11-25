"""Fetch Kim & Lee's published code, weights and splits at the pinned commit.

`npj Comput. Mater.` 11, 300 (2025). The authors publish the trained model, the
training and validation splits, and the code that computes the network's
predictions, which is what makes the lookup comparison in npj_split.py possible.

Their files are NOT redistributed here: this fetches them, then verifies each
against the checksum recorded when the measurements were made, so a clone
reconstructs exactly the bytes that were measured rather than inheriting a copy
whose provenance has gone stale. A checksum mismatch stops the run.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/TaeyeopK/ML-for-Electrical-Resistivity-of-Alloys.git"
COMMIT = "8c2f80812e2d8a77a62e7da42fdce6ef2d588f85"

# Verified 2026-09-04 against the bytes every npj number in this repository was
# measured on. Pinned: the three files that reproduce/npj_split.py reads, and
# model/ANN.py, which defines the network's layers that npj_split.py's forward
# pass re-implements. No script here reads ANN.py.
CHECKSUMS = {
    "csv/Train.csv": "864b734376ff15e27cfe319f34d8bc70",
    "csv/Val.csv": "50d2982c5bae49721d7acc2c13f49a90",
    "model/ANN.h5": "74c3caab0dd12e63172e9dd0fce9f218",
    "model/ANN.py": "90416c1b570db084ae395ca679303a10",
}

DEST = Path(__file__).resolve().parent.parent / "data" / "npj"


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    if DEST.exists():
        shutil.rmtree(DEST)
    DEST.parent.mkdir(parents=True, exist_ok=True)

    subprocess.run(["git", "clone", "--quiet", REPO, str(DEST)], check=True)
    subprocess.run(["git", "-C", str(DEST), "checkout", "--quiet", COMMIT],
                   check=True)

    bad = []
    for rel, expected in CHECKSUMS.items():
        path = DEST / rel
        if not path.exists():
            bad.append(f"{rel}: missing at {COMMIT[:8]}")
            continue
        got = md5(path)
        if got != expected:
            bad.append(f"{rel}: md5 {got}, recorded {expected}")
    if bad:
        raise SystemExit(
            "Upstream files do not match what was measured:\n  "
            + "\n  ".join(bad)
            + f"\nThe pin is {COMMIT}. Every npj number in this repository was "
              f"measured on the recorded bytes; re-measure before trusting a "
              f"changed upstream."
        )

    print(f"fetched {REPO} at {COMMIT[:8]} -> {DEST}")
    print(f"verified {len(CHECKSUMS)} files against their recorded checksums")


if __name__ == "__main__":
    sys.exit(main())
