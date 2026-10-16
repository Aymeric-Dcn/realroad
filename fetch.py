#!/usr/bin/env python3
"""
Pipeline stage 1: fetch the open data for one road section.

    python fetch.py osm      # road centreline (OpenStreetMap, IGN BD TOPO fallback)
    python fetch.py mnt      # LiDAR HD DTM tiles (terrain, 50 cm)
    python fetch.py ortho    # IGN 20 cm orthophotos around the road
    python fetch.py lidar    # LiDAR HD point cloud of the corridor (--remote: partial COPC reads)
    python fetch.py all      # osm + mnt + ortho (LiDAR is heavy, run it separately)

All geometries are in Lambert-93 (EPSG:2154), in metres.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import requests
import shapely
import yaml
from pyproj import CRS, Transformer
from shapely.geometry import LineString, box, mapping, shape
from shapely.ops import transform as shp_transform

UA = {"User-Agent": "realroad-pipeline/0.1 (projet perso open data)"}
OVERPASS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
WFS = "https://data.geopf.fr/wfs/ows"
WMS = "https://data.geopf.fr/wms-r/wms"
# One feature per 1 km² tile, holding the URLs of the DTM/DSM/CHM (50 cm GeoTIFF)
# and of the point cloud (COPC .laz). Checked 2026-10-16; IGN renames layers now and then.
LAYER_LIDAR_META = "IGNF_LIDAR-HD_METADONNEE:metadata"

TO_L93 = Transformer.from_crs(4326, 2154, always_xy=True)
TO_WGS = Transformer.from_crs(2154, 4326, always_xy=True)


# ------------------------------------------------------------------ helpers
def load_cfg(path):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_dir"] = Path(cfg.get("data_dir", "data")) / cfg["name"]
    cfg["_dir"].mkdir(parents=True, exist_ok=True)
    return cfg


def http_get(url, params=None, stream=False, retries=4):
    msg = ""
    for i in range(retries):
        try:
            r = requests.get(url, params=params, headers=UA, stream=stream, timeout=180)
            if r.status_code == 200:
                return r
            msg = f"HTTP {r.status_code} : {r.text[:1500]}"
            if 400 <= r.status_code < 500 and r.status_code != 429:
                break
        except requests.RequestException as e:
            msg = str(e)
        wait = 3 * 2 ** i
        if i == retries - 1:
            break
        print(f"  ! {msg} — retrying in {wait}s")
        time.sleep(wait)
    raise RuntimeError(f"Failed: {url}\n  {msg}")


def download(url, dest):
    """Resumable download (via a .part file); skipped if the file already exists."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    r = http_get(url, stream=True)
    ctype = r.headers.get("Content-Type", "")
    if "xml" in ctype or "html" in ctype:
        raise RuntimeError(f"The server returned an error instead of the file:\n  {r.text[:800]}")
    total = int(r.headers.get("Content-Length", 0))
    done = 0
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r    {done / 1e6:7.0f} / {total / 1e6:.0f} MB", end="", flush=True)
    if total:
        print()
    tmp.replace(dest)
    return True


def save_geojson(path, geom, crs_epsg, props=None):
    fc = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": f"urn:ogc:def:crs:EPSG::{crs_epsg}"}},
        "features": [{"type": "Feature", "properties": props or {}, "geometry": mapping(geom)}],
    }
    Path(path).write_text(json.dumps(fc), encoding="utf-8")


def load_route(cfg):
    p = cfg["_dir"] / "route_l93.geojson"
    if not p.exists():
        sys.exit("No centreline yet: run `python fetch.py osm` first.")
    fc = json.loads(p.read_text(encoding="utf-8"))
    return shape(fc["features"][0]["geometry"])


def terrain_area(cfg, route):
    return route.buffer(cfg["terrain_margin_m"])


def corridor_area(cfg, route):
    return route.buffer(cfg["corridor_m"], cap_style="flat")


# ---------------------------------------------------------- road centreline
def route_from_lines(lines, start_xy, end_xy, label="the segments", gap_m=60):
    """Shortest path over a set of polylines (Lambert-93) between the vertices
    closest to the start and end points. Polylines are joined wherever their
    vertices coincide (to within 10 cm)."""
    if not lines:
        raise RuntimeError(f"No road segment found in {label}: check osm_ref_regex and the start/end points.")
    key = lambda c: (round(c[0], 1), round(c[1], 1))
    G = nx.Graph()
    for coords in lines:
        ks = [key(c) for c in coords]
        for a, b in zip(ks, ks[1:]):
            if a != b:
                G.add_edge(a, b, length=math.dist(a, b))

    # Roundabouts and junctions often lack the road number: stitch together
    # pieces whose end vertices are less than gap_m apart.
    comps = list(nx.connected_components(G))
    if len(comps) > 1:
        ends = [[n for n in c if G.degree(n) == 1] or list(c) for c in comps]
        bridges = []
        for i in range(len(comps)):
            for j in range(i + 1, len(comps)):
                d, a, b = min((math.dist(a, b), a, b) for a in ends[i] for b in ends[j])
                if d <= gap_m:
                    bridges.append((d, a, b))
        for d, a, b in sorted(bridges):
            if not nx.has_path(G, a, b):
                G.add_edge(a, b, length=d)
                print(f"  · stitched a {d:.0f} m gap (unnumbered roundabout / junction)")

    def nearest(p):
        n = min(G.nodes, key=lambda k: math.dist(k, p))
        return n, math.dist(n, p)

    s, ds = nearest(start_xy)
    e, de = nearest(end_xy)
    for name, d in (("start", ds), ("end", de)):
        if d > 250:
            print(f"  ! The {name} point is {d:.0f} m from the road: check it.")
    try:
        path = nx.shortest_path(G, s, e, weight="length")
    except nx.NetworkXNoPath:
        raise RuntimeError(
            f"In {label}, the segments do not connect start and end "
            "(gap in the line, or a different road number on some stretch). "
            "Widen osm_ref_regex or move the points."
        )
    return LineString(path)


def overpass(query, rounds=2):
    last = ""
    for rnd in range(rounds):
        for url in OVERPASS:
            try:
                r = requests.post(url, data={"data": query}, headers=UA, timeout=120)
                if r.status_code == 200:
                    return r.json()
                last = f"HTTP {r.status_code}"
            except requests.RequestException as e:
                last = type(e).__name__
            print(f"  ! {url.split('/')[2]} : {last}")
        if rnd < rounds - 1:
            print("  … all busy, retrying in 20 s")
            time.sleep(20)
    raise RuntimeError(f"Overpass unavailable ({last})")


def lines_from_osm(cfg, start_ll, end_ll):
    lats, lons = [start_ll[1], end_ll[1]], [start_ll[0], end_ll[0]]
    m = 0.02  # ~2 km margin
    bbox = f"{min(lats) - m:.5f},{min(lons) - m:.5f},{max(lats) + m:.5f},{max(lons) + m:.5f}"
    q = f"""[out:json][timeout:90];
way["highway"]["ref"~"{cfg['osm_ref_regex']}"]({bbox});
(._;>;); out body;"""
    osm = overpass(q)
    (cfg["_dir"] / "route_osm_raw.json").write_text(json.dumps(osm), encoding="utf-8")
    nodes = {e["id"]: TO_L93.transform(e["lon"], e["lat"])
             for e in osm["elements"] if e["type"] == "node"}
    return [[nodes[n] for n in w["nodes"] if n in nodes]
            for w in osm["elements"] if w["type"] == "way"]


def lines_from_ign(cfg, start_ll, end_ll):
    """IGN BD TOPO road segments with an attribute (the road number, field
    `cpx_numero`) matching osm_ref_regex. Same open licence as the LiDAR."""
    import re

    rx = re.compile(cfg["osm_ref_regex"])
    a, b = TO_L93.transform(*start_ll), TO_L93.transform(*end_ll)
    area = box(min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1])).buffer(2000)
    feats = wfs_features(cfg.get("ign_road_layer", "BDTOPO_V3:troncon_de_route"), area)
    (cfg["_dir"] / "route_ign_raw.json").write_text(json.dumps(feats), encoding="utf-8")
    lines = []
    for f in feats:
        props = f.get("properties") or {}
        if not any(isinstance(v, str) and rx.match(v.strip()) for v in props.values()):
            continue
        g = _to_l93_if_needed(shape(f["geometry"]))
        parts = g.geoms if g.geom_type == "MultiLineString" else [g]
        lines += [[(c[0], c[1]) for c in p.coords] for p in parts]
    print(f"  {len(feats)} BD TOPO segments in the area, {len(lines)} carry the road number.")
    return lines


def cmd_osm(cfg):
    s, e = cfg["start"], cfg["end"]
    start_ll, end_ll = (s["lon"], s["lat"]), (e["lon"], e["lat"])
    start_xy, end_xy = TO_L93.transform(*start_ll), TO_L93.transform(*end_ll)
    source = cfg.get("road_source", "auto")
    route = None
    if source in ("auto", "osm"):
        print("Querying OpenStreetMap…")
        try:
            route = route_from_lines(lines_from_osm(cfg, start_ll, end_ll), start_xy, end_xy, "OSM")
            src = "OpenStreetMap"
        except RuntimeError as err:
            if source == "osm":
                raise
            print(f"  ! OSM: {err}\n  → falling back to IGN BD TOPO")
    if route is None:
        print("Querying BD TOPO (IGN)…")
        route = route_from_lines(lines_from_ign(cfg, start_ll, end_ll), start_xy, end_xy, "BD TOPO")
        src = "BD TOPO IGN"
    save_geojson(cfg["_dir"] / "route_l93.geojson", route, 2154, {"source": src})
    save_geojson(cfg["_dir"] / "route_wgs84.geojson", shp_transform(TO_WGS.transform, route), 4326, {"source": src})
    print(f"Centreline ({src}): {route.length / 1000:.2f} km, {len(route.coords)} points.")
    print(f"→ Check it by dropping {cfg['_dir'] / 'route_wgs84.geojson'} on https://geojson.io")


# ------------------------------------------------------------------ IGN tiles
def _to_l93_if_needed(g):
    x, y = g.centroid.x, g.centroid.y
    if abs(x) <= 180 and abs(y) <= 90:  # the server answered in lon/lat
        return shp_transform(TO_L93.transform, g)
    return g


def _tile_name(url):
    """File name: the FILENAME= parameter (DTM served via WMS) or the URL path tail (LAZ)."""
    from urllib.parse import parse_qs, urlparse

    q = parse_qs(urlparse(url).query)
    for k, v in q.items():
        if k.upper() == "FILENAME":
            return v[0]
    return Path(urlparse(url).path).name


def select_tiles(features, area, url_key):
    """1 km² tiles from the LiDAR HD metadata layer that intersect `area`."""
    out, seen = [], set()
    for f in features:
        g = _to_l93_if_needed(shape(f["geometry"]))
        if not g.intersects(area):
            continue
        url = (f.get("properties") or {}).get(url_key)
        if not url:
            continue
        name = _tile_name(url)
        if name in seen:
            continue
        seen.add(name)
        out.append({"name": name, "url": url, "bounds": list(g.bounds)})
    return out


def wfs_features(layer, area):
    minx, miny, maxx, maxy = area.bounds
    feats, start, page_size = [], 0, 1000
    while True:
        r = http_get(WFS, params={
            "SERVICE": "WFS", "VERSION": "2.0.0", "REQUEST": "GetFeature",
            "TYPENAMES": layer, "OUTPUTFORMAT": "application/json",
            "SRSNAME": "EPSG:2154",
            "BBOX": f"{minx:.2f},{miny:.2f},{maxx:.2f},{maxy:.2f},urn:ogc:def:crs:EPSG::2154",
            "COUNT": page_size, "STARTINDEX": start,
        })
        try:
            page = r.json().get("features", [])
        except ValueError:
            raise RuntimeError(f"Unexpected WFS response for {layer}: {r.text[:1500]}")
        feats += page
        if len(page) < page_size:
            break
        start += page_size
    return feats


def lidar_tiles(area, url_key):
    """url_key: 'url_mnt' (50 cm DTM GeoTIFF), 'url_npl' (COPC .laz point cloud),
    'url_mns' (DSM), 'url_mnh' (canopy height)."""
    return select_tiles(wfs_features(LAYER_LIDAR_META, area), area, url_key)


def cmd_mnt(cfg):
    route = load_route(cfg)
    tiles = lidar_tiles(terrain_area(cfg, route), "url_mnt")
    if not tiles:
        sys.exit("No DTM tile found: area not published yet? Check macarte.ign.fr.")
    (cfg["_dir"] / "mnt_tiles.json").write_text(json.dumps(tiles, indent=1), encoding="utf-8")
    print(f"{len(tiles)} DTM tiles to fetch.")
    for i, t in enumerate(tiles, 1):
        dest = cfg["_dir"] / "mnt" / t["name"]
        print(f"  [{i}/{len(tiles)}] {t['name']}")
        download(t["url"], dest)


# -------------------------------------------------------------------- LiDAR
def _new_las_header(src_header):
    """Fresh LAS 1.4 header (without the COPC-specific VLRs) keeping the same
    point format, scales and offsets."""
    import laspy

    h = laspy.LasHeader(point_format=src_header.point_format, version="1.4")
    h.scales = src_header.scales
    h.offsets = src_header.offsets
    return h


def _filter(pts, area, classes):
    pts = pts[np.isin(np.asarray(pts.classification), classes)]
    if len(pts):
        pts = pts[shapely.contains_xy(area, np.asarray(pts.x), np.asarray(pts.y))]
    return pts


def corridor_cells(area, tile_bounds, cell_m):
    """Non-overlapping square cells of the tile that touch the corridor."""
    minx, miny, maxx, maxy = tile_bounds
    cells = []
    y = miny
    while y < maxy - 1e-6:
        x = minx
        while x < maxx - 1e-6:
            c = (x, y, min(x + cell_m, maxx), min(y + cell_m, maxy))
            if box(*c).intersects(area):
                cells.append(c)
            x += cell_m
        y += cell_m
    return cells


def crop_copc_remote(url, dst, area, classes, tile_bounds, cell_m=150, threads=2):
    """Remotely read (HTTP Range requests) only the parts of the COPC cloud that
    touch the corridor. Less data, but slower than a plain download on a good
    connection, and IGN answers 429 (rate limit) if too many requests run in
    parallel, hence the low thread count and the retries."""
    import laspy
    from laspy.copc import Bounds

    shapely.prepare(area)
    cells = corridor_cells(area, tile_bounds, cell_m)
    kept = 0
    with laspy.CopcReader.open(url, http_num_threads=threads) as rdr:
        with laspy.open(dst, mode="w", header=_new_las_header(rdr.header)) as w:
            for k, (x0, y0, x1, y1) in enumerate(cells, 1):
                for attempt in range(5):
                    try:
                        pts = rdr.query(Bounds(mins=np.array([x0, y0]), maxs=np.array([x1, y1])))
                        break
                    except Exception as e:
                        if "429" not in str(e) or attempt == 4:
                            raise
                        time.sleep(5 * 2 ** attempt)  # rate limited: back off
                if len(pts):
                    # COPC query bounds are inclusive: avoid duplicates between cells
                    x, y = np.asarray(pts.x), np.asarray(pts.y)
                    pts = pts[(x >= x0) & (x < x1) & (y >= y0) & (y < y1)]
                pts = _filter(pts, area, classes)
                if len(pts):
                    w.write_points(pts)
                    kept += len(pts)
                print(f"\r    cell {k}/{len(cells)} — {kept:,} points", end="", flush=True)
    print()
    return kept


def crop_laz_local(src, dst, area, classes):
    """Non-COPC fallback: download the whole tile, then crop it chunk by chunk."""
    import laspy

    shapely.prepare(area)
    kept = 0
    with laspy.open(src) as reader:
        with laspy.open(dst, mode="w", header=_new_las_header(reader.header)) as writer:
            for pts in reader.chunk_iterator(3_000_000):
                pts = _filter(pts, area, classes)
                if len(pts):
                    writer.write_points(pts)
                    kept += len(pts)
    return kept


def cmd_lidar(cfg, remote=False):
    route = load_route(cfg)
    area = corridor_area(cfg, route)
    tiles = lidar_tiles(area, "url_npl")
    if not tiles:
        sys.exit("No LiDAR tile found: area not published yet? Check macarte.ign.fr.")
    (cfg["_dir"] / "lidar_tiles.json").write_text(json.dumps(tiles, indent=1), encoding="utf-8")
    crop_dir = cfg["_dir"] / "lidar"
    crop_dir.mkdir(parents=True, exist_ok=True)
    out_name = lambda t: t["name"].replace(".copc.laz", ".laz")
    todo = [t for t in tiles if not (crop_dir / out_name(t)).exists()
            and not (crop_dir / (out_name(t) + ".empty")).exists()]
    print(f"{len(tiles)} LiDAR tiles touch the corridor (±{cfg['corridor_m']} m), "
          f"{len(todo)} left to process.")
    if todo and not remote:
        print("Downloading whole tiles (~150-400 MB each, deleted after cropping).")
    classes = cfg["lidar_classes"]
    for i, t in enumerate(todo, 1):
        dst = crop_dir / out_name(t)
        tmp = dst.with_name(dst.name + ".part.laz")
        print(f"  [{i}/{len(todo)}] {t['name']}")
        t0 = time.time()
        if remote:
            try:
                n = crop_copc_remote(t["url"], tmp, area, classes, t["bounds"])
            except Exception as e:  # rate limit, no Range support, old laspy…
                tmp.unlink(missing_ok=True)
                sys.exit(f"\n    ! Remote COPC read failed ({type(e).__name__}: {e})\n"
                         "    Run `python fetch.py lidar` without --remote to download whole tiles.")
        else:
            raw = cfg["_dir"] / "lidar_raw" / t["name"]
            download(t["url"], raw)
            n = crop_laz_local(raw, tmp, area, classes)
            if not cfg.get("keep_raw_laz", False):
                raw.unlink(missing_ok=True)
        if n == 0:
            tmp.unlink(missing_ok=True)
            (crop_dir / (out_name(t) + ".empty")).touch()
        else:
            tmp.replace(dst)
        mb = dst.stat().st_size / 1e6 if dst.exists() else 0
        print(f"    → {n:,} points kept, {mb:.1f} MB, {time.time() - t0:.0f} s")


# ------------------------------------------------------------------- ortho
def ortho_grid(area, res, px=2000):
    """Square cells of `px` pixels that touch the area."""
    step = res * px
    minx, miny, maxx, maxy = area.bounds
    x0, y0 = math.floor(minx / step) * step, math.floor(miny / step) * step
    cells = []
    y = y0
    while y < maxy:
        x = x0
        while x < maxx:
            if box(x, y, x + step, y + step).intersects(area):
                cells.append((x, y, x + step, y + step))
            x += step
        y += step
    return cells


def world_file(cell, px):
    minx, miny, maxx, maxy = cell
    res = (maxx - minx) / px
    return "\n".join(str(v) for v in (res, 0.0, 0.0, -res, minx + res / 2, maxy - res / 2)) + "\n"


def cmd_ortho(cfg, px=2000):
    route = load_route(cfg)
    res = cfg["ortho_res_m"]
    cells = ortho_grid(terrain_area(cfg, route), res, px)
    out = cfg["_dir"] / "ortho"
    out.mkdir(exist_ok=True)
    prj = CRS.from_epsg(2154).to_wkt("WKT1_ESRI")
    print(f"{len(cells)} ortho images ({px} px, {res * px:.0f} m per side).")
    for i, c in enumerate(cells, 1):
        name = f"ortho_{int(c[0])}_{int(c[1])}"
        jpg = out / f"{name}.jpg"
        if jpg.exists():
            continue
        r = http_get(WMS, params={
            "SERVICE": "WMS", "VERSION": "1.3.0", "REQUEST": "GetMap",
            "LAYERS": cfg["ortho_layer"], "STYLES": "", "CRS": "EPSG:2154",
            "BBOX": ",".join(f"{v:.2f}" for v in c), "WIDTH": px, "HEIGHT": px,
            "FORMAT": "image/jpeg",
        })
        if not r.headers.get("Content-Type", "").startswith("image"):
            raise RuntimeError(f"Unexpected WMS response: {r.text[:300]}")
        jpg.write_bytes(r.content)
        (out / f"{name}.jgw").write_text(world_file(c, px))
        (out / f"{name}.prj").write_text(prj)
        print(f"  [{i}/{len(cells)}] {name}")


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["osm", "mnt", "ortho", "lidar", "all"])
    ap.add_argument("-c", "--config", default="config.yaml")
    ap.add_argument("--remote", action="store_true",
                    help="lidar: partial remote COPC reads instead of downloading whole tiles")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    if a.step in ("osm", "all"):
        cmd_osm(cfg)
    if a.step in ("mnt", "all"):
        cmd_mnt(cfg)
    if a.step in ("ortho", "all"):
        cmd_ortho(cfg)
    if a.step == "lidar":
        cmd_lidar(cfg, a.remote)


if __name__ == "__main__":
    main()
