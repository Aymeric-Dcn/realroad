#!/usr/bin/env python3
"""
Pipeline stage 1: fetch the open data for one road section.

    python fetch.py osm      # road centreline (OpenStreetMap, IGN BD TOPO fallback)
    python fetch.py mnt      # LiDAR HD DTM tiles (terrain, 50 cm)
    python fetch.py ortho    # IGN 20 cm orthophotos around the road
    python fetch.py lidar    # LiDAR HD point cloud of the corridor (--remote: partial COPC reads)
    python fetch.py far      # low-resolution relief + imagery for the horizon (~8 km around)
    python fetch.py mnh      # LiDAR HD canopy height (trees), 50 cm
    python fetch.py buildings  # IGN BD TOPO buildings (footprint, heights) near the road
    python fetch.py streetview # Panoramax street-level photos along the road (open licence)
    python fetch.py roadside   # objects along the road: OSM barriers / signs + LiDAR above-ground points
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
# Elevation for the distant relief (RGE ALTI, then BD ALTI): first one that answers is used
BACKDROP_DEM_LAYERS = ["ELEVATION.ELEVATIONGRIDCOVERAGE.HIGHRES", "ELEVATION.ELEVATIONGRIDCOVERAGE"]

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
            if total > 5e6:  # progress only for big files
                print(f"\r    {done / 1e6:7.0f} / {total / 1e6:.0f} MB", end="", flush=True)
    if total > 5e6:
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
    vertices coincide (to within 10 cm). A line may come as (coords, cost
    factor): the path then prefers the bigger roads (any road network, no
    road number needed)."""
    if not lines:
        raise RuntimeError(f"No road segment found in {label}: check osm_ref_regex and the start/end points.")
    key = lambda c: (round(c[0], 1), round(c[1], 1))
    G = nx.Graph()
    weighted = any(isinstance(ln, tuple) for ln in lines)
    for ln in lines:
        coords, factor = ln if isinstance(ln, tuple) else (ln, 1.0)
        ks = [key(c) for c in coords]
        for a, b in zip(ks, ks[1:]):
            if a != b:
                G.add_edge(a, b, length=math.dist(a, b) * factor)

    # Roundabouts and junctions often lack the road number: stitch together
    # pieces whose end vertices are less than gap_m apart (numbered road only:
    # on a full network it would invent shortcuts).
    comps = list(nx.connected_components(G))
    if len(comps) > 1 and not weighted:
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


# Without a road number, the route is the cheapest path over all the roads a
# car can use: length × this factor (numbered roads × 0.8 on top).
OSM_ROAD_COST = {"motorway": 1.0, "trunk": 1.0, "primary": 1.0, "secondary": 1.0, "tertiary": 1.1,
                 "unclassified": 1.4, "residential": 1.6, "living_street": 2.5, "service": 4.0}


def lines_from_osm(cfg, start_ll, end_ll):
    lats, lons = [start_ll[1], end_ll[1]], [start_ll[0], end_ll[0]]
    m = 0.02  # ~2 km margin
    bbox = f"{min(lats) - m:.5f},{min(lons) - m:.5f},{max(lats) + m:.5f},{max(lons) + m:.5f}"
    rx = cfg.get("osm_ref_regex")
    sel = (f'way["highway"]["ref"~"{rx}"]({bbox});' if rx else
           f'way["highway"~"^({"|".join(OSM_ROAD_COST)})(_link)?$"]({bbox});')
    q = f"""[out:json][timeout:90];
{sel}
(._;>;); out body;"""
    osm = overpass(q)
    (cfg["_dir"] / "route_osm_raw.json").write_text(json.dumps(osm), encoding="utf-8")
    nodes = {e["id"]: TO_L93.transform(e["lon"], e["lat"])
             for e in osm["elements"] if e["type"] == "node"}
    out = []
    for w in osm["elements"]:
        if w["type"] != "way":
            continue
        coords = [nodes[n] for n in w["nodes"] if n in nodes]
        if rx:
            out.append(coords)
        else:
            t = w.get("tags") or {}
            hw = t.get("highway", "").replace("_link", "")
            if t.get("access") in ("no", "private") or t.get("motor_vehicle") == "no":
                continue
            out.append((coords, OSM_ROAD_COST.get(hw, 3.0) * (0.8 if t.get("ref") else 1.0)))
    return out


def lines_from_ign(cfg, start_ll, end_ll):
    """IGN BD TOPO road segments with an attribute (the road number, field
    `cpx_numero`) matching osm_ref_regex. Same open licence as the LiDAR."""
    import re

    rx = re.compile(cfg["osm_ref_regex"]) if cfg.get("osm_ref_regex") else None
    a, b = TO_L93.transform(*start_ll), TO_L93.transform(*end_ll)
    area = box(min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1])).buffer(2000)
    feats = wfs_features(cfg.get("ign_road_layer", "BDTOPO_V3:troncon_de_route"), area)
    (cfg["_dir"] / "route_ign_raw.json").write_text(json.dumps(feats), encoding="utf-8")
    lines = []
    for f in feats:
        props = f.get("properties") or {}
        factor = 1.0
        if rx is not None:
            if not any(isinstance(v, str) and rx.match(v.strip()) for v in props.values()):
                continue
        else:   # any road a car can use; importance 1 (main) … 6 (minor)
            if props.get("nature") in ("Chemin", "Sentier", "Escalier", "Piste cyclable") or \
                    props.get("acces_vehicule_leger") in ("Physiquement impossible", "Restreint aux ayants droit"):
                continue
            try:
                factor = 1.0 + 0.15 * (int(props.get("importance") or 5) - 1)
            except ValueError:
                factor = 1.6
            factor *= 0.8 if props.get("cpx_numero") else 1.0
        g = _to_l93_if_needed(shape(f["geometry"]))
        parts = g.geoms if g.geom_type == "MultiLineString" else [g]
        for p in parts:
            coords = [(c[0], c[1]) for c in p.coords]
            lines.append(coords if rx is not None else (coords, factor))
    print(f"  {len(feats)} BD TOPO segments in the area, {len(lines)} used"
          f"{' (carrying the road number)' if rx is not None else ''}.")
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


def cmd_mnh(cfg):
    """Canopy height model (LiDAR HD MNH): height of trees above the ground."""
    route = load_route(cfg)
    area = route.buffer(cfg.get("trees_margin_m", 250))
    tiles = lidar_tiles(area, "url_mnh")
    if not tiles:
        sys.exit("No canopy height tile found for this area.")
    (cfg["_dir"] / "mnh_tiles.json").write_text(json.dumps(tiles, indent=1), encoding="utf-8")
    print(f"{len(tiles)} canopy height tiles to fetch.")
    for i, t in enumerate(tiles, 1):
        print(f"  [{i}/{len(tiles)}] {t['name']}")
        download(t["url"], cfg["_dir"] / "mnh" / t["name"])


def cmd_buildings(cfg):
    """BD TOPO buildings near the road: footprint + hauteur, altitude_minimale_sol,
    altitude_minimale_toit (eaves), altitude_maximale_toit (ridge)."""
    route = load_route(cfg)
    margin = cfg.get("buildings_margin_m", 250)
    area = route.buffer(margin)
    feats = wfs_features(cfg.get("ign_buildings_layer", "BDTOPO_V3:batiment"), area)
    keep = []
    for f in feats:
        try:
            g = _to_l93_if_needed(shape(f["geometry"]))
        except Exception:
            continue
        if g.is_empty or not g.intersects(area):
            continue
        props = {k: (f.get("properties") or {}).get(k) for k in (
            "cleabs", "hauteur", "altitude_minimale_sol", "altitude_maximale_sol",
            "altitude_minimale_toit", "altitude_maximale_toit", "nature", "usage_1",
            "construction_legere", "nombre_d_etages")}
        keep.append({"type": "Feature", "properties": props, "geometry": mapping(g)})
    out = cfg["_dir"] / "buildings_l93.geojson"
    out.write_text(json.dumps({"type": "FeatureCollection", "features": keep}), encoding="utf-8")
    print(f"{len(keep)} buildings within {margin} m of the road → {out}")


PANORAMAX = "https://api.panoramax.xyz/api"


def cmd_streetview(cfg):
    """Panoramax (open street-level imagery, IGN and contributors): pictures along
    the road, one every streetview_step_m, standard definition (2048 px)."""
    route = load_route(cfg)
    step = cfg.get("streetview_step_m", 20)
    out = cfg["_dir"] / "streetview"
    out.mkdir(exist_ok=True)
    # 1. index of the pictures near the road (search boxes every 150 m)
    items = {}
    n_box = max(1, int(route.length / 150))
    for i in range(n_box + 1):
        pt = route.interpolate(i * route.length / n_box)
        lon0, lat0 = TO_WGS.transform(pt.x - 110, pt.y - 110)
        lon1, lat1 = TO_WGS.transform(pt.x + 110, pt.y + 110)
        try:
            r = http_get(f"{PANORAMAX}/search", params={
                "bbox": f"{lon0:.6f},{lat0:.6f},{lon1:.6f},{lat1:.6f}", "limit": 500}, retries=3)
            feats = r.json().get("features", [])
        except (RuntimeError, ValueError) as e:
            print(f"  ! search {i}: {e}")
            continue
        for f in feats:
            pr = f.get("properties") or {}
            sd = ((f.get("assets") or {}).get("sd") or {}).get("href")
            if not sd:
                continue
            lon, lat = f["geometry"]["coordinates"][:2]
            x, y = TO_L93.transform(lon, lat)
            items[f["id"]] = {
                "id": f["id"], "x": x, "y": y, "lon": lon, "lat": lat, "sd": sd,
                "azimuth": pr.get("view:azimuth"), "datetime": pr.get("datetime"),
                "fov": (pr.get("pers:interior_orientation") or {}).get("field_of_view"),
                "license": pr.get("license"), "producer": ((f.get("providers") or [{}])[0]).get("name"),
                "collection": f.get("collection"),
            }
        if i % 20 == 0:
            print(f"  search {i}/{n_box}: {len(items)} pictures so far")
    if not items:
        sys.exit("No Panoramax picture found along the road.")
    # 2. one picture per step along the road (nearest to the station, within 15 m)
    pts = list(items.values())
    xy = np.array([[p["x"], p["y"]] for p in pts])
    chosen, covered = [], 0
    stations = np.arange(0, route.length, step)
    for sv in stations:
        q = route.interpolate(sv)
        d = np.hypot(xy[:, 0] - q.x, xy[:, 1] - q.y)
        k = int(d.argmin())
        if d[k] <= 15:
            covered += 1
            if not chosen or chosen[-1]["id"] != pts[k]["id"]:
                chosen.append({**pts[k], "s_m": float(sv), "dist_m": float(d[k])})
    print(f"Coverage: {100 * covered / len(stations):.0f} % of the road within 15 m of a picture "
          f"({len(items)} pictures found, {len(chosen)} kept, one per {step} m).")
    (out / "index.json").write_text(json.dumps(chosen, indent=1), encoding="utf-8")
    # 3. download (resumable)
    for i, p in enumerate(chosen, 1):
        dest = out / f"{int(p['s_m']):06d}_{p['id']}.jpg"
        p["file"] = dest.name
        if not dest.exists():
            try:
                download(p["sd"], dest)
            except Exception as e:  # one missing picture is not fatal
                print(f"  ! {p['id']}: {e}")
        if i % 50 == 0:
            print(f"  [{i}/{len(chosen)}] pictures")
    (out / "index.json").write_text(json.dumps(chosen, indent=1), encoding="utf-8")
    print(f"→ {out} (licence: {', '.join(sorted({str(p['license']) for p in chosen}))})")


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


# ---------------------------------------------------------------- roadside
# LiDAR HD classes of things standing beside the road that are not vegetation:
# 1 unclassified (guard rails, posts, fences often end up here), 6 building,
# 64 permanent above-ground (walls), 67 miscellaneous built structures.
ROADSIDE_CLASSES = [1, 6, 64, 67]


def cmd_roadside(cfg, remote=True):
    """What stands along the road: OSM barriers (guard rails, walls), signs and
    milestones, plus the LiDAR above-ground points of a narrow band (walls,
    guard rails, posts) used to find them where OSM has nothing."""
    route = load_route(cfg)
    band = cfg.get("roadside_band_m", 18)
    out_dir = cfg["_dir"]
    # 1. OpenStreetMap
    lon0, lat0 = TO_WGS.transform(*route.bounds[:2])
    lon1, lat1 = TO_WGS.transform(*route.bounds[2:])
    m = 0.003
    bbox = f"{lat0 - m:.5f},{lon0 - m:.5f},{lat1 + m:.5f},{lon1 + m:.5f}"
    q = f"""[out:json][timeout:120];
node["highway"~"^(stop|give_way)$"]({bbox})->.sg;
(
  way(bn.sg)["highway"];
  way["barrier"]({bbox});
  node["barrier"]({bbox});
  node["traffic_sign"]({bbox});
  node["highway"~"^(milestone|traffic_signals|stop|give_way|street_lamp|bus_stop|speed_camera)$"]({bbox});
  node["railway"="milestone"]({bbox});
  way["man_made"~"^(embankment|retaining_wall)$"]({bbox});
  node["power"~"^(pole|tower)$"]({bbox});
  node["man_made"="utility_pole"]({bbox});
  {f'way["highway"]["ref"~"{cfg["osm_ref_regex"]}"]["maxspeed"]({bbox});' if cfg.get("osm_ref_regex") else f'way["highway"]["maxspeed"]({bbox});'}
  node["place"~"^(village|town|hamlet)$"]({bbox});
);
(._;>;); out body;"""
    try:
        osm = overpass(q)
        (out_dir / "osm_roadside.json").write_text(json.dumps(osm), encoding="utf-8")
        kinds = {}
        for e in osm["elements"]:
            t = e.get("tags") or {}
            k = t.get("barrier") or t.get("traffic_sign") and "traffic_sign" or t.get("highway") \
                or t.get("power") or t.get("man_made") or t.get("place")
            if k:
                kinds[k] = kinds.get(k, 0) + 1
        print(f"  OSM: {', '.join(f'{v} {k}' for k, v in sorted(kinds.items(), key=lambda kv: -kv[1]))}")
    except RuntimeError as e:
        print(f"  ! OSM objects skipped: {e}")
    # 2. LiDAR above-ground points of a narrow band (non-vegetation classes)
    area = route.buffer(band, cap_style="flat")
    tiles = lidar_tiles(area, "url_npl")
    crop_dir = out_dir / "lidar_roadside"
    crop_dir.mkdir(parents=True, exist_ok=True)
    out_name = lambda t: t["name"].replace(".copc.laz", ".laz")
    todo = [t for t in tiles if not (crop_dir / out_name(t)).exists()
            and not (crop_dir / (out_name(t) + ".empty")).exists()]
    print(f"  LiDAR: {len(tiles)} tiles touch the ±{band} m band, {len(todo)} left "
          f"(classes {ROADSIDE_CLASSES}, {'remote reads' if remote else 'whole tiles'}).")
    for i, t in enumerate(todo, 1):
        dst = crop_dir / out_name(t)
        tmp = dst.with_name(dst.name + ".part.laz")
        print(f"  [{i}/{len(todo)}] {t['name']}")
        t0 = time.time()
        n = None
        if remote:
            try:
                n = crop_copc_remote(t["url"], tmp, area, ROADSIDE_CLASSES, t["bounds"], cell_m=100)
            except Exception as e:  # rate limit, no Range support…: whole tile instead
                tmp.unlink(missing_ok=True)
                print(f"    ! remote read failed ({type(e).__name__}), downloading the whole tile")
        if n is None:
            raw = out_dir / "lidar_raw" / t["name"]
            download(t["url"], raw)
            n = crop_laz_local(raw, tmp, area, ROADSIDE_CLASSES)
            if not cfg.get("keep_raw_laz", False):
                raw.unlink(missing_ok=True)
        if n == 0:
            tmp.unlink(missing_ok=True)
            (crop_dir / (out_name(t) + ".empty")).touch()
        else:
            tmp.replace(dst)
        mb = dst.stat().st_size / 1e6 if dst.exists() else 0
        print(f"    → {n:,} points kept, {mb:.1f} MB, {time.time() - t0:.0f} s")
    # one small array file for roadside.py / build_track.py (no laspy needed there)
    import laspy

    parts = []
    for f in sorted(crop_dir.glob("*.laz")):
        if f.name.endswith(".part.laz"):
            continue
        las = laspy.read(f)
        parts.append(np.column_stack([np.asarray(las.x), np.asarray(las.y), np.asarray(las.z),
                                      np.asarray(las.classification), np.asarray(las.intensity),
                                      np.asarray(las.return_number), np.asarray(las.number_of_returns)]))
    pts = np.concatenate(parts) if parts else np.zeros((0, 7))
    np.savez_compressed(out_dir / "roadside_points.npz", xyz=pts[:, :3].astype(np.float64),
                        cls=pts[:, 3].astype(np.uint8), intensity=pts[:, 4].astype(np.uint16),
                        ret=pts[:, 5].astype(np.uint8), nret=pts[:, 6].astype(np.uint8))
    cls, cnt = np.unique(pts[:, 3].astype(int), return_counts=True)
    print(f"→ {crop_dir}\n→ {out_dir / 'roadside_points.npz'}: {len(pts):,} points "
          f"({', '.join(f'class {c}: {n:,}' for c, n in zip(cls, cnt))})")


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
        jpg.write_bytes(ortho_image(cfg, c, px))
        (out / f"{name}.jgw").write_text(world_file(c, px))
        (out / f"{name}.prj").write_text(prj)
        print(f"  [{i}/{len(cells)}] {name}")


# IGN renames / withdraws orthophoto layers now and then (HR.ORTHOIMAGERY.
# ORTHOPHOTOS answered "unknown layer" on 2026-10-20 while still listed):
# the first layer that answers is kept for the rest of the run.
ORTHO_LAYERS = ["HR.ORTHOIMAGERY.ORTHOPHOTOS", "ORTHOIMAGERY.ORTHOPHOTOS", "ORTHOIMAGERY.ORTHOPHOTOS.BDORTHO"]
_ortho_ok = {}


def ortho_image(cfg, bbox, px):
    """JPEG bytes of the orthophoto over bbox (Lambert-93), px x px."""
    layers = [cfg.get("ortho_layer")] + [l for l in ORTHO_LAYERS if l != cfg.get("ortho_layer")]
    if "layer" in _ortho_ok:
        layers = [_ortho_ok["layer"]] + [l for l in layers if l != _ortho_ok["layer"]]
    last = ""
    for layer in [l for l in layers if l]:
        try:
            r = http_get(WMS, params={
                "SERVICE": "WMS", "VERSION": "1.3.0", "REQUEST": "GetMap",
                "LAYERS": layer, "STYLES": "", "CRS": "EPSG:2154",
                "BBOX": ",".join(f"{v:.2f}" for v in bbox), "WIDTH": px, "HEIGHT": px,
                "FORMAT": "image/jpeg",
            })
        except RuntimeError as e:
            last = str(e)
            continue
        if not r.headers.get("Content-Type", "").startswith("image"):
            last = r.text[:300]
            continue
        if _ortho_ok.get("layer") != layer:
            if "layer" in _ortho_ok or layer != layers[0]:
                print(f"  (orthophoto layer: {layer})")
            _ortho_ok["layer"] = layer
        return r.content
    raise RuntimeError(f"No orthophoto layer answered: {last[:400]}")


# ---------------------------------------------------------------- horizon
def _is_tiff(b):
    return b[:4] in (b"II*\x00", b"MM\x00*")


def cmd_far(cfg):
    """Low-resolution relief and imagery around the road, for the horizon:
    one GeoTIFF DTM (25 m) and 4 km orthophoto tiles (4 m/px) out to backdrop_m."""
    route = load_route(cfg)
    margin = cfg.get("backdrop_m", 8000)
    res_dem = cfg.get("backdrop_dem_res_m", 25)
    out = cfg["_dir"] / "backdrop"
    out.mkdir(exist_ok=True)
    cells = ortho_grid(route.buffer(margin), 4.0, 1000)
    minx = min(c[0] for c in cells); miny = min(c[1] for c in cells)
    maxx = max(c[2] for c in cells); maxy = max(c[3] for c in cells)
    dem = out / "backdrop_dem.tif"
    if not dem.exists():
        w, h = int((maxx - minx) / res_dem), int((maxy - miny) / res_dem)
        layers = [cfg["backdrop_dem_layer"]] if cfg.get("backdrop_dem_layer") else BACKDROP_DEM_LAYERS
        errors = []
        for layer in layers:
            try:
                r = http_get(WMS, params={
                    "SERVICE": "WMS", "VERSION": "1.3.0", "REQUEST": "GetMap", "LAYERS": layer,
                    "STYLES": "", "CRS": "EPSG:2154", "BBOX": f"{minx:.2f},{miny:.2f},{maxx:.2f},{maxy:.2f}",
                    "WIDTH": w, "HEIGHT": h, "FORMAT": "image/geotiff",
                }, retries=2)
            except RuntimeError as e:
                errors.append(f"{layer}: {str(e)[-300:]}")
                continue
            if not _is_tiff(r.content):
                errors.append(f"{layer}: not a GeoTIFF ({r.headers.get('Content-Type')}: {r.content[:200]!r})")
                continue
            dem.with_suffix(".part").write_bytes(r.content)
            dem.with_suffix(".part").replace(dem)
            # world file next to it, in case the GeoTIFF has no georeferencing
            rx, ry = (maxx - minx) / w, (maxy - miny) / h
            (out / "backdrop_dem.tfw").write_text(
                "\n".join(str(v) for v in (rx, 0.0, 0.0, -ry, minx + rx / 2, maxy - ry / 2)) + "\n")
            print(f"Distant relief: {layer}, {w}x{h} px at {res_dem} m.")
            break
        else:
            raise SystemExit("No elevation layer answered for the distant relief:\n  " + "\n  ".join(errors)
                             + "\nSet backdrop_dem_layer in config.yaml (see the WMS-R GetCapabilities).")
    print(f"{len(cells)} distant orthophoto tiles (4 km, 4 m/px).")
    for i, c in enumerate(cells, 1):
        name = f"backdrop_{int(c[0])}_{int(c[1])}"
        jpg = out / f"{name}.jpg"
        if jpg.exists():
            continue
        jpg.write_bytes(ortho_image(cfg, c, 1000))
        (out / f"{name}.jgw").write_text(world_file(c, 1000))
        print(f"  [{i}/{len(cells)}] {name}")


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["osm", "mnt", "ortho", "lidar", "far", "mnh", "buildings", "streetview", "roadside", "all"])
    ap.add_argument("-c", "--config", default="config.yaml")
    ap.add_argument("--remote", action="store_true",
                    help="lidar: partial remote COPC reads instead of downloading whole tiles")
    ap.add_argument("--whole-tiles", action="store_true",
                    help="roadside: download whole LiDAR tiles instead of remote COPC reads")
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
    if a.step == "far":
        cmd_far(cfg)
    if a.step == "mnh":
        cmd_mnh(cfg)
    if a.step == "buildings":
        cmd_buildings(cfg)
    if a.step == "streetview":
        cmd_streetview(cfg)
    if a.step == "roadside":
        cmd_roadside(cfg, remote=not a.whole_tiles)


if __name__ == "__main__":
    main()
