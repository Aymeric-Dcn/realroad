#!/usr/bin/env python3
"""
Run the whole pipeline, from open data to an Assetto Corsa track.

    python realroad.py --install "C:\\Program Files (x86)\\Steam\\steamapps\\common\\assettocorsa"
    python realroad.py --from build_track --install "..."     # only the last steps
    python realroad.py --list

Downloads are resumable and skipped when already there; the other steps are
re-run (a few minutes in total). Stops at the first error.
"""
import argparse
import subprocess
import sys
import time

STEPS = [
    ("osm", ["fetch.py", "osm"], "road centreline"),
    ("mnt", ["fetch.py", "mnt"], "terrain (LiDAR HD DTM)"),
    ("ortho", ["fetch.py", "ortho"], "aerial imagery"),
    ("lidar", ["fetch.py", "lidar"], "corridor point cloud"),
    ("far", ["fetch.py", "far"], "distant relief for the horizon (optional)"),
    ("road_profile", ["road_profile.py"], "elevation, crossfall, bridges"),
    ("road_edges", ["road_edges.py"], "real carriageway edges"),
    ("road_markings", ["road_markings.py"], "real centre and edge lines"),
    ("build_track", ["build_track.py"], "track geometry"),
    ("kn5_export", ["kn5_export.py"], "Assetto Corsa track folder"),
]

OPTIONAL = {"far"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", default="config.yaml")
    ap.add_argument("--from", dest="start", choices=[s[0] for s in STEPS], help="start at this step")
    ap.add_argument("--to", dest="stop", choices=[s[0] for s in STEPS], help="stop after this step")
    ap.add_argument("--install", metavar="AC_ROOT", help="copy the track into <AC_ROOT>/content/tracks")
    ap.add_argument("--list", action="store_true", help="list the steps and exit")
    a = ap.parse_args()
    names = [s[0] for s in STEPS]
    if a.list:
        for n, cmd, what in STEPS:
            print(f"  {n:14s} python {' '.join(cmd):24s} {what}")
        return
    i0 = names.index(a.start) if a.start else 0
    i1 = names.index(a.stop) + 1 if a.stop else len(STEPS)
    t_all = time.time()
    for n, cmd, what in STEPS[i0:i1]:
        args = [sys.executable, *cmd, "-c", a.config]
        if n == "kn5_export" and a.install:
            args += ["--install", a.install]
        print(f"\n=== {n}: {what}", flush=True)
        t = time.time()
        r = subprocess.run(args)
        if r.returncode != 0 and n in OPTIONAL:
            print(f"    ! optional step '{n}' failed: going on without it")
            continue
        if r.returncode != 0:
            raise SystemExit(f"\nStep '{n}' failed (exit code {r.returncode}). Fix it, then resume with "
                             f"`python realroad.py --from {n}`.")
        print(f"    ({time.time() - t:.0f} s)")
    print(f"\nDone in {(time.time() - t_all) / 60:.1f} min.")


if __name__ == "__main__":
    main()
