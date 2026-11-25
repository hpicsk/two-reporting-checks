"""Fetch Matbench's experimental band gap set and verify it against the
checksum Matminer records for it.

`matbench_expt_gap`: 4,604 compositions with experimentally measured band gaps
(Zhuo, Mansouri Tehrani & Brgoch, J. Phys. Chem. Lett. 9, 1668, 2018), as
curated for Matbench v0.1 (Dunn et al., npj Comput. Mater. 6, 138, 2020) and
deduplicated by composition. It is served from Materials Project's Matbench
host. The SHA-256 below is the one Matminer's dataset_metadata.json records for
the file (hackingmaterials/matminer, read 2026-09-05), so a clone reconstructs
the bytes that were measured rather than whatever the host serves later. A
mismatch stops the run.

The 37 kB file is cached in the repository, as the HTEM-DB rows are, so the
control runs offline.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

URL = "https://ml.materialsproject.org/projects/matbench_expt_gap.json.gz"
SHA256 = "783e7d1461eb83b00b2f2942da4b95fda5e58a0d1ae26b581c24cf8a82ca75b2"
DEST = Path(__file__).resolve().parent.parent / "data" / "expt_gap" / "matbench_expt_gap.json.gz"


def main() -> None:
    # The host answers 403 to urllib's default User-Agent and 200 to a browser's.
    request = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
    got = hashlib.sha256(payload).hexdigest()
    if got != SHA256:
        raise SystemExit(
            f"{URL} does not match what was measured:\n  sha256 {got}\n  recorded {SHA256}\n"
            f"Every expt_gap number in this repository was measured on the recorded "
            f"bytes; re-measure before trusting a changed upstream.")
    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.write_bytes(payload)
    print(f"fetched {URL} -> {DEST}")
    print(f"verified sha256 {SHA256[:12]}... ({len(payload):,} bytes)")


if __name__ == "__main__":
    sys.exit(main())
