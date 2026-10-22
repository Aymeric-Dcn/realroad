#!/usr/bin/env python3
"""
Run the whole pipeline, from open data to an Assetto Corsa track.

A new road, in one command (writes roads/<name>.yaml, then builds it):

    python realroad.py --name gourdon-caussols --start 43.72130,6.97700 --end 43.74209,6.90447 --install --clean

An existing road file (config.yaml by default, or one of roads/):

    python realroad.py -c roads/gourdon-caussols.yaml --install
    python realroad.py -c roads/gourdon-caussols.yaml --from build_track --install   # only the last steps
    python realroad.py -c roads/gourdon-caussols.yaml --clean-only   # free the space later
    python realroad.py --list

--install alone finds Assetto Corsa in the Steam libraries; give the folder
if it is elsewhere. Before downloading, the disk space and time needed are
estimated and you are asked to confirm (--yes to skip). --clean deletes the
downloads and intermediate files once the track is installed: only the track
(in Assetto Corsa) and a few small files stay.

Downloads are resumable and skipped when already there; the other steps are
re-run. Stops at the first error.
"""
import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

STEPS = [
    ("osm", ["fetch.py", "osm"], "road centreline"),
    ("mnt", ["fetch.py", "mnt"], "terrain (LiDAR HD DTM)"),
    ("ortho", ["fetch.py", "ortho"], "aerial imagery"),
    ("lidar", ["fetch.py", "lidar"], "corridor point cloud"),
    ("far", ["fetch.py", "far"], "distant relief for the horizon (optional)"),
    ("mnh", ["fetch.py", "mnh"], "canopy height for the trees (optional)"),
    ("buildings", ["fetch.py", "buildings"], "BD TOPO buildings (optional)"),
    ("streetview", ["fetch.py", "streetview"], "Panoramax street-level photos (optional)"),
    ("roadside", ["fetch.py", "roadside"], "guard rails, walls, signs: OSM + LiDAR (optional)"),
    ("assets", ["assets.py"], "free CC0 textures and tree sprites, shared by all roads (optional)"),
    ("road_profile", ["road_profile.py"], "elevation, crossfall, bridges"),
    ("road_edges", ["road_edges.py"], "real carriageway edges"),
    ("road_markings", ["road_markings.py"], "real centre and edge lines"),
    ("trees", ["trees.py"], "tree positions, heights, kinds (optional)"),
    ("trees3d", ["trees3d.py"], "3D tree models for CSP, sources (optional; then Blender + CM)"),
    ("build_track", ["build_track.py"], "track geometry"),
    ("kn5_export", ["kn5_export.py"], "Assetto Corsa track folder"),
    ("vao", ["vao.py"], "baked ambient occlusion for CSP (optional; needs tools/CustomTracksBakery.exe)"),
    ("streetview_sheet", ["streetview_sheet.py"], "street-level comparison sheet (optional)"),
]

OPTIONAL = {"far", "mnh", "buildings", "streetview", "roadside", "assets", "trees3d", "trees", "vao",
            "streetview_sheet"}
DOWNLOADS = {"osm", "mnt", "ortho", "lidar", "far", "mnh", "buildings", "streetview", "roadside", "assets"}
# Street-level photos only serve the comparison sheet: skipped unless asked for
SKIP_BY_DEFAULT = {"streetview", "streetview_sheet"}

# Measured on the two first roads (Route Napoléon 17.4 km, Gourdon-Caussols 10.2 km)
MB_PER_TILE = 31.0          # DTM + canopy height, per 1 km² IGN tile
MB_PER_KM = 64.0            # orthophotos, LiDAR corridor, textures, track build
MB_FIXED = 30.0             # horizon relief, tree models
MB_RAW_LIDAR = 400.0        # one raw LiDAR tile at a time while cropping
MB_TRACK_PER_KM = 24.0      # the Assetto Corsa track folder
MIN_PER_KM = 2.8            # first build, downloads included (Gourdon-Caussols: 28.7 min)

# Heavy files deleted by --clean (everything else is a few MB)
CLEAN = ["mnt", "mnh", "lidar", "lidar_raw", "lidar_roadside", "ortho", "backdrop", "texture_dds",
         "track_build", "streetview", "cross_bins.npz", "roadside_points.npz", "screens"]


# ------------------------------------------------------------ Assetto Corsa
def steam_libraries():
    roots = []
    for base in (os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
                 os.environ.get("ProgramFiles", r"C:\Program Files")):
        roots.append(Path(base) / "Steam")
    roots += [Path.home() / ".steam" / "steam", Path.home() / ".local" / "share" / "Steam"]
    libs = []
    for r in roots:
        vdf = r / "steamapps" / "libraryfolders.vdf"
        if vdf.exists():
            for m in re.finditer(r'"path"\s+"([^"]+)"', vdf.read_text(encoding="utf-8", errors="ignore")):
                libs.append(Path(m.group(1).replace("\\\\", "\\")))
        libs.append(r)
    return libs


def find_ac():
    for lib in steam_libraries():
        ac = lib / "steamapps" / "common" / "assettocorsa"
        if (ac / "content" / "tracks").is_dir():
            return ac
    return None


# --------------------------------------------------------------- road files
def parse_point(txt):
    m = re.match(r"^\s*(-?\d+(?:\.\d+)?)\s*[,; ]\s*(-?\d+(?:\.\d+)?)\s*$", txt)
    if not m:
        raise SystemExit(f"'{txt}': expected LAT,LON (e.g. 43.72130,6.97700, as Google Maps copies it)")
    lat, lon = float(m.group(1)), float(m.group(2))
    if not (41 < lat < 51.5 and -5.5 < lon < 10):
        raise SystemExit(f"{lat}, {lon} is not in metropolitan France (IGN LiDAR HD coverage).")
    return lat, lon


def write_road_file(a):
    name = re.sub(r"[^a-z0-9-]+", "-", a.name.lower()).strip("-")
    path = Path("roads") / f"{name}.yaml"
    (s_lat, s_lon), (e_lat, e_lon) = parse_point(a.start), parse_point(a.end)
    if path.exists():
        print(f"{path} already exists: using it (delete it to start over).")
        return str(path)
    path.parent.mkdir(exist_ok=True)
    small = {"de", "du", "des", "la", "le", "les", "et", "a", "au", "aux", "sur", "en", "d", "l"}
    words = name.split("-")
    auto = " ".join(w if (w in small and i) else w.capitalize() for i, w in enumerate(words))
    title = a.title or (a.name if a.name != a.name.lower() or " " in a.name else auto)
    lines = [
        "# Road section to generate (written by realroad.py; everything not set here has a default)",
        f"name: {name}",
        f"start: {{lat: {s_lat:.5f}, lon: {s_lon:.5f}}}",
        f"end:   {{lat: {e_lat:.5f}, lon: {e_lon:.5f}}}",
    ]
    if a.ref:
        lines.append(f'osm_ref_regex: "^{a.ref.replace(" ", " ?")}$"   # stick to this road number')
    else:
        lines.append("# no road number: shortest drive between the points, preferring the bigger roads")
    lines += [
        "track:",
        f"  id: realroad_{name.replace('-', '_')}",
        f"  title: {json.dumps(title, ensure_ascii=False)}",
        f"  barrier_type: {a.barriers}   # valley-side barriers: guardrail, parapet (stone walls) or auto",
    ]
    if a.author:
        lines.append(f"  author: {json.dumps(a.author, ensure_ascii=False)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"→ {path}")
    return str(path)


def load_yaml(path):
    import yaml

    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_dir"] = Path(cfg.get("data_dir", "data")) / cfg["name"]
    return cfg


# ------------------------------------------------------------------ estimate
def dir_mb(p):
    p = Path(p)
    if p.is_file():
        return p.stat().st_size / 1e6
    if not p.exists():
        return 0.0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6


def route_xy(d):
    fc = json.loads((d / "route_l93.geojson").read_text(encoding="utf-8"))
    g = fc["features"][0]["geometry"]
    c = g["coordinates"] if g["type"] == "LineString" else [p for part in g["coordinates"] for p in part]
    return [(float(x), float(y)) for x, y, *_ in c]


def estimate(cfg, ac):
    d = cfg["_dir"]
    pts = route_xy(d)
    length = sum(math.dist(p, q) for p, q in zip(pts, pts[1:]))
    margin = cfg.get("terrain_margin_m", 300)
    tiles = set()
    for p, q in zip(pts, pts[1:]):
        n = max(1, int(math.dist(p, q) / 50))
        for i in range(n + 1):
            x = p[0] + (q[0] - p[0]) * i / n
            y = p[1] + (q[1] - p[1]) * i / n
            for dx in (-margin, 0, margin):
                for dy in (-margin, 0, margin):
                    tiles.add((int((x + dx) // 1000), int((y + dy) // 1000)))
    km = length / 1000
    total = MB_PER_TILE * len(tiles) + MB_PER_KM * km + MB_FIXED
    have = dir_mb(d)
    temp = max(0.0, total - have) + MB_RAW_LIDAR
    track = MB_TRACK_PER_KM * km
    done = sum((d / f).exists() for f in ("mnt", "ortho", "lidar"))
    minutes = MIN_PER_KM * km * (0.35 if done == 3 else 1.0)
    return {"km": km, "tiles": len(tiles), "temp_mb": temp, "data_mb": total, "track_mb": track,
            "min": minutes, "have_mb": have}


def gb(mb):
    return f"{mb / 1000:.1f} GB" if mb >= 950 else f"{mb:.0f} MB"


def free_mb(p):
    p = Path(p)
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free / 1e6


def show_estimate(cfg, ac, clean, ask):
    e = estimate(cfg, ac)
    d = cfg["_dir"]
    print(f"\nRoad: {e['km']:.1f} km, {e['tiles']} IGN tiles of 1 km².")
    print(f"  downloads and work files: ~{gb(e['data_mb'])} in {d}"
          + (f" ({gb(e['have_mb'])} already there)" if e["have_mb"] > 50 else ""))
    print(f"  space needed now: ~{gb(e['temp_mb'] + (e['track_mb'] if ac else 0))}"
          f" (free: {gb(free_mb(d))})")
    if ac:
        print(f"  track left in Assetto Corsa: ~{gb(e['track_mb'])} in {Path(ac) / 'content' / 'tracks'}")
    if clean:
        print(f"  --clean: the ~{gb(e['data_mb'])} of work files are deleted at the end")
    else:
        print("  (add --clean to delete the work files at the end and keep only the track)")
    print(f"  time: ~{e['min']:.0f} min, mostly downloads (depends on your connection)")
    need = e["temp_mb"] + (e["track_mb"] if ac else 0)
    if free_mb(d) < need * 1.1:
        raise SystemExit(f"\nNot enough free disk space ({gb(free_mb(d))} free, ~{gb(need)} needed).")
    if ask and sys.stdin.isatty():
        r = input("Go on? [Y/n] ").strip().lower()
        if r not in ("", "y", "yes", "o", "oui"):
            raise SystemExit("Stopped. Nothing else was downloaded.")


# --------------------------------------------------------------------- clean
def clean(cfg):
    d = cfg["_dir"]
    # the CSP tree models are shared by all roads: keep a copy
    bins = sorted((d / "csp_trees").glob("*.bin"))
    if bins:
        shared = d.parent / "_shared" / "csp_trees"
        shared.mkdir(parents=True, exist_ok=True)
        for b in bins:
            if not (shared / b.name).exists():
                shutil.copy2(b, shared / b.name)
    freed = 0.0
    for name in CLEAN:
        p = d / name
        if not p.exists():
            continue
        freed += dir_mb(p)
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        else:
            p.unlink()
    left = dir_mb(d)
    print(f"\nCleaned: {gb(freed)} freed, {gb(left)} of small files kept in {d}"
          f" (a rebuild downloads the data again).")


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", default=None, help="road file (default config.yaml)")
    ap.add_argument("--name", help="new road: name (e.g. gourdon-caussols)")
    ap.add_argument("--start", metavar="LAT,LON", help="new road: start point")
    ap.add_argument("--end", metavar="LAT,LON", help="new road: end point")
    ap.add_argument("--title", help="new road: name shown in Assetto Corsa")
    ap.add_argument("--ref", help="new road: stick to this road number (e.g. D12)")
    ap.add_argument("--barriers", default="auto", choices=["auto", "guardrail", "parapet"],
                    help="new road: valley-side barriers")
    ap.add_argument("--author", help="new road: author shown in Assetto Corsa")
    ap.add_argument("--from", dest="start_step", choices=[s[0] for s in STEPS], help="start at this step")
    ap.add_argument("--to", dest="stop", choices=[s[0] for s in STEPS], help="stop after this step")
    ap.add_argument("--install", nargs="?", const="auto", metavar="AC_ROOT",
                    help="copy the track into <AC_ROOT>/content/tracks (alone: find Assetto Corsa)")
    ap.add_argument("--clean", action="store_true", help="delete downloads and work files once installed")
    ap.add_argument("--clean-only", action="store_true", help="just delete the work files of this road")
    ap.add_argument("--photos", action="store_true", help="also fetch Panoramax street photos (comparison sheet)")
    ap.add_argument("-y", "--yes", action="store_true", help="do not ask before downloading")
    ap.add_argument("--list", action="store_true", help="list the steps and exit")
    a = ap.parse_args()
    repo = Path(__file__).resolve().parent
    if a.config and Path(a.config).exists() and Path.cwd().resolve() != repo:
        a.config = str(Path(a.config).resolve())
    os.chdir(repo)          # the steps are run as scripts of this folder
    names = [s[0] for s in STEPS]
    if a.list:
        for n, cmd, what in STEPS:
            print(f"  {n:16s} python {' '.join(cmd):24s} {what}")
        return
    if a.start or a.end or a.name:
        if not (a.start and a.end and a.name):
            raise SystemExit("A new road needs --name, --start and --end.")
        if a.config:
            raise SystemExit("Give either -c <road file> or --name/--start/--end, not both.")
        a.config = write_road_file(a)
    a.config = a.config or "config.yaml"
    if not Path(a.config).exists():
        raise SystemExit(f"No road file {a.config}.")
    ac = None
    if a.install:
        ac = find_ac() if a.install == "auto" else Path(a.install)
        if ac is None:
            raise SystemExit("Assetto Corsa not found in the Steam libraries: give its folder, "
                             "--install \"D:\\SteamLibrary\\steamapps\\common\\assettocorsa\"")
        if not (Path(ac) / "content" / "tracks").is_dir():
            raise SystemExit(f"{ac} is not an Assetto Corsa folder (no content\\tracks).")
        print(f"Assetto Corsa: {ac}")
    if a.clean and not ac and not a.clean_only:
        raise SystemExit("--clean needs --install: the track would be deleted with the work files.")
    cfg = load_yaml(a.config)
    if a.clean_only:
        clean(cfg)
        return

    i0 = names.index(a.start_step) if a.start_step else 0
    i1 = names.index(a.stop) + 1 if a.stop else len(STEPS)
    run = [s for s in STEPS[i0:i1] if a.photos or s[0] not in SKIP_BY_DEFAULT or a.start_step == s[0]]
    downloads = any(n in DOWNLOADS - {"osm"} for n, _, _ in run)
    resume = f"python realroad.py -c {a.config}" + (f' --install "{ac}"' if ac else "") + (" --clean" if a.clean else "")
    t_all = time.time()
    estimated = False
    for n, cmd, what in run:
        if downloads and not estimated and n != "osm":
            estimated = True
            if (cfg["_dir"] / "route_l93.geojson").exists():
                show_estimate(cfg, ac, a.clean, not a.yes)
        args = [sys.executable, *cmd, "-c", a.config]
        if n in ("kn5_export", "vao") and ac:
            args += ["--install", str(ac)]
        if n == "vao" and not ac:
            continue
        print(f"\n=== {n}: {what}", flush=True)
        t = time.time()
        r = subprocess.run(args)
        if r.returncode != 0 and n in OPTIONAL:
            print(f"    ! optional step '{n}' failed: going on without it")
            continue
        if r.returncode != 0:
            raise SystemExit(f"\nStep '{n}' failed (exit code {r.returncode}). Fix it, then resume with:\n"
                             f"  {resume} --from {n}")
        print(f"    ({time.time() - t:.0f} s)")
    print(f"\nDone in {(time.time() - t_all) / 60:.1f} min.")
    if ac:
        tid = (cfg.get("track") or {}).get("id") or \
            "realroad_" + re.sub(r"[^a-z0-9]+", "_", cfg["name"].lower()).strip("_")   # as fetch.load_cfg
        dest = Path(ac) / "content" / "tracks" / tid
        if dest.exists():
            print(f"Track installed: {dest} ({gb(dir_mb(dest))})")
        if a.clean:
            if not dest.exists():
                print("! the track is not in Assetto Corsa: nothing cleaned")
            elif names.index("kn5_export") >= i1 or "kn5_export" not in [s[0] for s in run]:
                print("! kn5_export did not run: nothing cleaned")
            else:
                clean(cfg)


if __name__ == "__main__":
    main()
