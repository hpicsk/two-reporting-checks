"""Fetch the HTEM-DB rows reproduce/htem_split.py reads, from the public API.

Every sample of every library carrying both electrical and X-ray fluorescence
data, kept if it records a four-point-probe resistivity (Zakutayev et al., Sci.
Data 5, 180053, 2018). NREL, which runs the database, is now the National
Laboratory of the Rockies, and the API is served from its nlr.gov domain. The
API is live, so no checksum pins it: a fetch that differs from the cached rows
is a changed upstream, and every HTEM-DB number would have to be re-measured.

The rows are cached in the repository, so the analysis runs offline.
"""
import concurrent.futures as cf
import gzip
import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

API = "https://htem-api.nlr.gov/api"
DEST = Path(__file__).resolve().parent.parent / "data" / "htem" / "htem_rows.ndjson.gz"
KEEP = ['id', 'sample_library_id', 'position', 'thickness', 'absolute_temp_c', 'xyz_mm',
        'fpm_resistivity', 'fpm_conductivity', 'fpm_sheet_resistance', 'fpm_standard_deviation',
        'opt_direct_bandgap', 'opt_average_vis_trans', 'xrf_elements', 'xrf_concentration',
        'xrf_compounds']


def get(url, tries=3):
    """The decoded response, or None when the API refuses the request (HTTP
    400), which it does for some sample ids a library lists. Any other error is
    retried, then raised."""
    for t in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 400:
                return None
            if t == tries - 1:
                raise
        except Exception:
            if t == tries - 1:
                raise


def main():
    libs = [lib for lib in get(f"{API}/sample_library")
            if (lib.get('has_ele') or 0) > 0 and (lib.get('has_xrf') or 0) > 0]
    lock, done, refused = threading.Lock(), [0], []

    def do_lib(lib):
        meta = get(f"{API}/sample_library/{lib['id']}")
        out = []
        for sid in (meta.get('sample_ids') or []):
            s = get(f"{API}/sample/{sid}")
            if s is None:
                with lock:
                    refused.append(sid)
                continue
            if s.get('fpm_resistivity') is not None:
                r = {k: s.get(k) for k in KEEP}
                r['lib_elements'] = lib.get('elements')
                r['lib_quality'] = lib.get('quality')
                r['depo_compounds'] = lib.get('deposition_compounds')
                out.append(r)
        with lock:
            done[0] += 1
            if done[0] % 25 == 0:
                print(f"  {done[0]}/{len(libs)} libraries", flush=True)
        return out

    rows = []
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        for res in ex.map(do_lib, libs):
            rows.extend(res)
    DEST.parent.mkdir(parents=True, exist_ok=True)
    # No name or time in the gzip header, so an unchanged upstream refetches
    # the cached bytes exactly.
    with open(DEST, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb",
                                                filename="", mtime=0) as f:
        f.write("".join(json.dumps(r) + "\n" for r in rows).encode())
    print(f"DONE rows={len(rows)} "
          f"libraries={len({r['sample_library_id'] for r in rows})} -> {DEST}")
    print(f"the API refused {len(refused)} listed sample ids, "
          f"from {min(refused, default=None)} to {max(refused, default=None)}")


if __name__ == "__main__":
    main()
