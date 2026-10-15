#!/usr/bin/env python3
"""
Étape 1 du pipeline : récupérer les données ouvertes d'un tronçon de route.

    python fetch.py osm      # tracé de la route (OpenStreetMap, sinon BD TOPO IGN)
    python fetch.py mnt      # dalles MNT LiDAR HD (relief, 50 cm)
    python fetch.py ortho    # orthophotos IGN 20 cm autour de la route
    python fetch.py lidar    # nuages de points LiDAR HD, découpés au couloir de la route
    python fetch.py all      # osm + mnt + ortho (le LiDAR, lourd, se lance à part)

Toutes les géométries sont en Lambert-93 (EPSG:2154), en mètres.
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
# Une entité par dalle de 1 km² avec les URLs du MNT/MNS/MNH (GeoTIFF 50 cm) et du
# nuage de points (COPC .laz). Vérifié le 16/10/2026 ; l'IGN renomme parfois ses couches.
LAYER_LIDAR_META = "IGNF_LIDAR-HD_METADONNEE:metadata"

TO_L93 = Transformer.from_crs(4326, 2154, always_xy=True)
TO_WGS = Transformer.from_crs(2154, 4326, always_xy=True)


# ---------------------------------------------------------------- utilitaires
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
        print(f"  ! {msg} — nouvel essai dans {wait}s")
        time.sleep(wait)
    raise RuntimeError(f"Échec : {url}\n  {msg}")


def download(url, dest):
    """Téléchargement reprenable (fichier .part), saute si déjà présent."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    r = http_get(url, stream=True)
    ctype = r.headers.get("Content-Type", "")
    if "xml" in ctype or "html" in ctype:
        raise RuntimeError(f"Le serveur a renvoyé une erreur au lieu du fichier :\n  {r.text[:800]}")
    total = int(r.headers.get("Content-Length", 0))
    done = 0
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r    {done / 1e6:7.0f} / {total / 1e6:.0f} Mo", end="", flush=True)
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
        sys.exit("Pas de tracé : lance d'abord `python fetch.py osm`.")
    fc = json.loads(p.read_text(encoding="utf-8"))
    return shape(fc["features"][0]["geometry"])


def terrain_area(cfg, route):
    return route.buffer(cfg["terrain_margin_m"])


def corridor_area(cfg, route):
    return route.buffer(cfg["corridor_m"], cap_style="flat")


# ------------------------------------------------------------- tracé route
def route_from_lines(lines, start_xy, end_xy, label="les tronçons", gap_m=60):
    """Plus court chemin sur un ensemble de polylignes (Lambert-93) entre les
    sommets les plus proches du départ et de l'arrivée. Les polylignes sont
    reliées là où leurs sommets coïncident (à 10 cm près)."""
    if not lines:
        raise RuntimeError(f"Aucun tronçon trouvé dans {label} : vérifie osm_ref_regex et les points.")
    key = lambda c: (round(c[0], 1), round(c[1], 1))
    G = nx.Graph()
    for coords in lines:
        ks = [key(c) for c in coords]
        for a, b in zip(ks, ks[1:]):
            if a != b:
                G.add_edge(a, b, length=math.dist(a, b))

    # Les ronds-points et carrefours n'ont souvent pas le numéro de route : on
    # recoud les bouts de tracé dont les extrémités sont à moins de gap_m.
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
                print(f"  · trou de {d:.0f} m recousu (rond-point / carrefour sans numéro)")

    def nearest(p):
        n = min(G.nodes, key=lambda k: math.dist(k, p))
        return n, math.dist(n, p)

    s, ds = nearest(start_xy)
    e, de = nearest(end_xy)
    for name, d in (("départ", ds), ("arrivée", de)):
        if d > 250:
            print(f"  ! Le point d'{name} est à {d:.0f} m de la route : vérifie-le.")
    try:
        path = nx.shortest_path(G, s, e, weight="length")
    except nx.NetworkXNoPath:
        raise RuntimeError(
            f"Dans {label}, les tronçons ne sont pas reliés entre départ et arrivée "
            "(trou dans le tracé ou numéro de route différent sur une portion). "
            "Élargis osm_ref_regex ou déplace les points."
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
            print("  … tous occupés, nouvel essai dans 20 s")
            time.sleep(20)
    raise RuntimeError(f"Overpass indisponible ({last})")


def lines_from_osm(cfg, start_ll, end_ll):
    lats, lons = [start_ll[1], end_ll[1]], [start_ll[0], end_ll[0]]
    m = 0.02  # ~2 km de marge
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
    """Tronçons de la BD TOPO (IGN) dont un attribut (numéro de route) matche
    osm_ref_regex. Même licence ouverte que le LiDAR."""
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
    print(f"  {len(feats)} tronçons BD TOPO dans la zone, {len(lines)} portent le bon numéro.")
    return lines


def cmd_osm(cfg):
    s, e = cfg["start"], cfg["end"]
    start_ll, end_ll = (s["lon"], s["lat"]), (e["lon"], e["lat"])
    start_xy, end_xy = TO_L93.transform(*start_ll), TO_L93.transform(*end_ll)
    source = cfg.get("road_source", "auto")
    route = None
    if source in ("auto", "osm"):
        print("Requête OpenStreetMap…")
        try:
            route = route_from_lines(lines_from_osm(cfg, start_ll, end_ll), start_xy, end_xy, "OSM")
            src = "OpenStreetMap"
        except RuntimeError as err:
            if source == "osm":
                raise
            print(f"  ! OSM : {err}\n  → je passe sur la BD TOPO de l'IGN")
    if route is None:
        print("Requête BD TOPO (IGN)…")
        route = route_from_lines(lines_from_ign(cfg, start_ll, end_ll), start_xy, end_xy, "la BD TOPO")
        src = "BD TOPO IGN"
    save_geojson(cfg["_dir"] / "route_l93.geojson", route, 2154, {"source": src})
    save_geojson(cfg["_dir"] / "route_wgs84.geojson", shp_transform(TO_WGS.transform, route), 4326, {"source": src})
    print(f"Tracé ({src}) : {route.length / 1000:.2f} km, {len(route.coords)} points.")
    print(f"→ Vérifie-le en glissant {cfg['_dir'] / 'route_wgs84.geojson'} sur https://geojson.io")


# ----------------------------------------------------------------- dalles IGN
def _to_l93_if_needed(g):
    x, y = g.centroid.x, g.centroid.y
    if abs(x) <= 180 and abs(y) <= 90:  # le serveur a répondu en lon/lat
        return shp_transform(TO_L93.transform, g)
    return g


def _tile_name(url):
    """Nom de fichier : paramètre FILENAME= (MNT servi en WMS) ou fin de l'URL (LAZ)."""
    from urllib.parse import parse_qs, urlparse

    q = parse_qs(urlparse(url).query)
    for k, v in q.items():
        if k.upper() == "FILENAME":
            return v[0]
    return Path(urlparse(url).path).name


def select_tiles(features, area, url_key):
    """Dalles 1 km² de la couche métadonnées LiDAR HD qui touchent `area`."""
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
            raise RuntimeError(f"Réponse WFS inattendue pour {layer} : {r.text[:1500]}")
        feats += page
        if len(page) < page_size:
            break
        start += page_size
    return feats


def lidar_tiles(area, url_key):
    """url_key : 'url_mnt' (MNT GeoTIFF 50 cm), 'url_npl' (nuage COPC .laz),
    'url_mns', 'url_mnh'."""
    return select_tiles(wfs_features(LAYER_LIDAR_META, area), area, url_key)


def cmd_mnt(cfg):
    route = load_route(cfg)
    tiles = lidar_tiles(terrain_area(cfg, route), "url_mnt")
    if not tiles:
        sys.exit("Aucune dalle MNT : zone pas encore diffusée ? Vérifie sur macarte.ign.fr.")
    (cfg["_dir"] / "mnt_tiles.json").write_text(json.dumps(tiles, indent=1), encoding="utf-8")
    print(f"{len(tiles)} dalles MNT à récupérer.")
    for i, t in enumerate(tiles, 1):
        dest = cfg["_dir"] / "mnt" / t["name"]
        print(f"  [{i}/{len(tiles)}] {t['name']}")
        download(t["url"], dest)


# -------------------------------------------------------------------- LiDAR
def crop_laz(src, dst, area, classes):
    """Garde les points des classes voulues à l'intérieur de `area`. Lecture par
    blocs pour ne pas charger une dalle de 2 Go en mémoire."""
    import laspy

    shapely.prepare(area)
    kept = 0
    with laspy.open(src) as reader:
        with laspy.open(dst, mode="w", header=reader.header) as writer:
            for pts in reader.chunk_iterator(3_000_000):
                sub = pts[np.isin(np.asarray(pts.classification), classes)]
                if len(sub) == 0:
                    continue
                sub = sub[shapely.contains_xy(area, np.asarray(sub.x), np.asarray(sub.y))]
                if len(sub):
                    writer.write_points(sub)
                    kept += len(sub)
    if kept == 0:
        Path(dst).unlink(missing_ok=True)
    return kept


def cmd_lidar(cfg, yes=False):
    route = load_route(cfg)
    area = corridor_area(cfg, route)
    tiles = lidar_tiles(area, "url_npl")
    if not tiles:
        sys.exit("Aucune dalle LiDAR : zone pas encore diffusée ? Vérifie sur macarte.ign.fr.")
    (cfg["_dir"] / "lidar_tiles.json").write_text(json.dumps(tiles, indent=1), encoding="utf-8")
    crop_dir = cfg["_dir"] / "lidar"
    crop_dir.mkdir(parents=True, exist_ok=True)
    todo = [t for t in tiles if not (crop_dir / t["name"]).exists()
            and not (crop_dir / (t["name"] + ".empty")).exists()]
    print(f"{len(tiles)} dalles LiDAR touchent le couloir, {len(todo)} restent à traiter.")
    print(f"Ordre de grandeur : ~{len(todo)} Go à télécharger (0,5 à 2 Go par dalle).")
    if todo and not yes and input("Continuer ? [o/N] ").strip().lower() not in ("o", "oui", "y"):
        return
    for i, t in enumerate(todo, 1):
        raw = cfg["_dir"] / "lidar_raw" / t["name"]
        print(f"  [{i}/{len(todo)}] {t['name']}")
        download(t["url"], raw)
        n = crop_laz(raw, crop_dir / t["name"], area, cfg["lidar_classes"])
        print(f"    → {n:,} points gardés".replace(",", " "))
        if n == 0:
            (crop_dir / (t["name"] + ".empty")).touch()
        if not cfg.get("keep_raw_laz", False):
            raw.unlink(missing_ok=True)


# ------------------------------------------------------------------- ortho
def ortho_grid(area, res, px=2000):
    """Cellules carrées de `px` pixels qui touchent la zone."""
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
    print(f"{len(cells)} images d'ortho ({px} px, {res * px:.0f} m de côté).")
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
            raise RuntimeError(f"Réponse WMS inattendue : {r.text[:300]}")
        jpg.write_bytes(r.content)
        (out / f"{name}.jgw").write_text(world_file(c, px))
        (out / f"{name}.prj").write_text(prj)
        print(f"  [{i}/{len(cells)}] {name}")


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["osm", "mnt", "ortho", "lidar", "all"])
    ap.add_argument("-c", "--config", default="config.yaml")
    ap.add_argument("-y", "--yes", action="store_true", help="ne pas demander confirmation")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    if a.step in ("osm", "all"):
        cmd_osm(cfg)
    if a.step in ("mnt", "all"):
        cmd_mnt(cfg)
    if a.step in ("ortho", "all"):
        cmd_ortho(cfg)
    if a.step == "lidar":
        cmd_lidar(cfg, a.yes)


if __name__ == "__main__":
    main()
