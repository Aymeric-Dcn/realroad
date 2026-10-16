#!/usr/bin/env python3
"""
Pipeline stage 3: build the track geometry for Assetto Corsa.

    python build_track.py      → data/<name>/track_build/  (meshes.npz, manifest.json, texture/)

then, with Blender (no UI needed):

    blender --factory-startup -b -P blender_export.py -- data/<name>/track_build

Geometry, all in metres:
  - road: ribbon (width per segment from BD TOPO) built from road_profile.csv (LiDAR elevation
    and crossfall), physical surface ROAD
  - verges: strips from the road edge down/up to the DTM, surface GRASS
  - terrain: DTM grid around the road, textured with the IGN orthophotos
    (one mesh per 400 m ortho tile), surface GRASS; pushed under the road
  - AC dummies: start, pit, hotlap start and A-to-B timing gates

Coordinates are shifted to a local origin (stored in manifest.json) so that
vertex values stay small; X = east, Y = north, Z = up (Blender convention).
"""
import argparse
import json
import math
import re
import shutil
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.spatial import cKDTree

from fetch import load_cfg
from preview import smooth

AC_MAX_VERTS = 60_000  # KN5 meshes use 16-bit indices (65 535), keep a margin

DEFAULTS = {
    "id": "realroad_track",
    "title": "realroad track",
    "road_width_m": "auto",      # "auto" = LiDAR edges (road_edges.py) else BD TOPO, "bdtopo", or a number
    "default_width_m": 7.0,      # used where BD TOPO has no width
    "min_width_m": 4.0,
    "width_field": "largeur_de_chaussee",
    "road_cross_vertices": 7,
    "verge_m": 2.5,
    "terrain_half_width_m": 120.0,
    "terrain_grid_m": 2.5,
    "z_sigma_m": 6.0,
    "crossfall_sigma_m": 10.0,
    "despike_window_m": 15.0,
    "z_despike_m": 0.06,
    "crossfall_despike_pct": 3.0,
    "crossfall_clip_pct": 10.0,
    "road_chunk_m": 200.0,
    "texture_repeat_m": 13.0,
    "start_offset_m": 15.0,
}


# ------------------------------------------------------------------ helpers
class FastDTM:
    """Bilinear DTM sampling, one tile in memory at a time (fast for millions
    of points, unlike rasterio's per-point sample())."""

    def __init__(self, files):
        import rasterio

        self.tiles = []
        for f in files:
            with rasterio.open(f) as d:
                self.tiles.append((f, d.bounds, d.transform, d.nodata))

    def sample(self, x, y):
        import rasterio

        z = np.full(len(x), np.nan)
        for f, b, tr, nodata in self.tiles:
            m = np.isnan(z) & (x >= b.left) & (x < b.right) & (y > b.bottom) & (y <= b.top)
            if not m.any():
                continue
            with rasterio.open(f) as d:
                a = d.read(1).astype(np.float64)
            if nodata is not None:
                a[a == nodata] = np.nan
            a[a < -1000] = np.nan
            # pixel-centre coordinates → fractional row/col
            col = (x[m] - tr.c) / tr.a - 0.5
            row = (y[m] - tr.f) / tr.e - 0.5
            c0 = np.clip(np.floor(col).astype(int), 0, a.shape[1] - 2)
            r0 = np.clip(np.floor(row).astype(int), 0, a.shape[0] - 2)
            fc = np.clip(col - c0, 0, 1)
            fr = np.clip(row - r0, 0, 1)
            z[m] = (a[r0, c0] * (1 - fc) * (1 - fr) + a[r0, c0 + 1] * fc * (1 - fr)
                    + a[r0 + 1, c0] * (1 - fc) * fr + a[r0 + 1, c0 + 1] * fc * fr)
        return z


def load_profile(path):
    d = np.genfromtxt(path, delimiter=",", names=True)
    cf = d["crossfall_pct"].copy()
    bad = ~np.isfinite(cf) | (d["axis_suspect"] > 0)
    ok = ~bad
    cf = np.interp(d["s_m"], d["s_m"][ok], cf[ok]) if ok.sum() > 1 else np.zeros_like(cf)
    return d["s_m"], d["x"], d["y"], d["z_filled_m"], cf


def despike(s, v, window, tol):
    """Replace values further than `tol` from the local median (window samples)
    by linear interpolation: removes isolated holes/spikes, keeps real shape."""
    med = median_filter(v, size=max(3, window | 1), mode="nearest")
    bad = np.abs(v - med) > tol
    if bad.any() and (~bad).sum() > 1:
        v = v.copy()
        v[bad] = np.interp(s[bad], s[~bad], v[~bad])
    return v, int(bad.sum())


def smooth_profile(s, z, cf_pct, p):
    """Elevation and crossfall as the car will feel them: despike, then a
    Gaussian low-pass (a moving average leaves small steps a car feels)."""
    step = float(np.median(np.diff(s)))
    win = int(p["despike_window_m"] / step)
    zd, nz = despike(s, z, win, p["z_despike_m"])
    cd, nc = despike(s, cf_pct, win, p["crossfall_despike_pct"])
    zs = gaussian_filter1d(zd, p["z_sigma_m"] / step, mode="nearest")
    cs = gaussian_filter1d(cd, p["crossfall_sigma_m"] / step, mode="nearest")
    cs = np.clip(cs, -p["crossfall_clip_pct"], p["crossfall_clip_pct"]) / 100
    curv = np.abs(np.gradient(np.gradient(zs, step), step))
    print(f"  profile: {nz} elevation and {nc} crossfall spikes removed; "
          f"min vertical radius {1 / max(curv.max(), 1e-9):.0f} m, "
          f"max deviation from LiDAR {100 * np.abs(zs - z).max():.1f} cm")
    return zs, cs


def frames(x, y, step, window_m=10.0):
    n = max(1, int(window_m / step))
    tx, ty = np.gradient(smooth(x, n)), np.gradient(smooth(y, n))
    norm = np.hypot(tx, ty)
    norm[norm == 0] = 1
    tx, ty = tx / norm, ty / norm
    return tx, ty, -ty, tx  # tangent, left normal


def grid_faces(n_rows, n_cols):
    """Two triangles per quad of a row-major (n_rows × n_cols) vertex grid,
    counter-clockwise seen from above when rows go forward and columns go
    from right to left."""
    r, c = np.meshgrid(np.arange(n_rows - 1), np.arange(n_cols - 1), indexing="ij")
    a = (r * n_cols + c).ravel()
    b, d_, e = a + 1, a + n_cols, a + n_cols + 1
    return np.concatenate([np.column_stack([a, d_, b]), np.column_stack([b, d_, e])]).astype(np.int32)


def fix_winding_up(verts, faces):
    """Make every triangle face up (+Z), so normals and AC physics are right."""
    v = verts[faces]
    nz = (v[:, 1, 0] - v[:, 0, 0]) * (v[:, 2, 1] - v[:, 0, 1]) - (v[:, 1, 1] - v[:, 0, 1]) * (v[:, 2, 0] - v[:, 0, 0])
    flip = nz < 0
    faces[flip] = faces[flip][:, [0, 2, 1]]
    return faces


# ------------------------------------------------------------ procedural textures
def write_png(path, rgb):
    import matplotlib.image as mpimg

    mpimg.imsave(path, np.clip(rgb, 0, 1))


def asphalt_texture(path, width_m, length_m, px_per_m=80, seed=1):
    """Asphalt with French markings: dashed centre line (3 m line / 10 m gap,
    T1 type) and continuous edge lines, 12 cm wide."""
    rng = np.random.default_rng(seed)
    w, h = int(width_m * px_per_m), int(length_m * px_per_m)
    base = 0.24 + 0.05 * rng.standard_normal((h, w))
    blur = smooth_2d(rng.standard_normal((h // 8 + 1, w // 8 + 1)), 1)
    base += 0.03 * np.kron(blur, np.ones((8, 8)))[:h, :w]
    img = np.repeat(base[..., None], 3, axis=2) * np.array([1.0, 1.0, 1.02])
    line = int(0.12 * px_per_m)
    edge_in = int(0.25 * px_per_m)
    paint = np.array([0.86, 0.86, 0.84])
    img[:, edge_in:edge_in + line] = paint
    img[:, w - edge_in - line:w - edge_in] = paint
    c0 = w // 2 - line // 2
    dash = int(3.0 * px_per_m)
    img[:dash, c0:c0 + line] = paint
    img += 0.02 * rng.standard_normal(img.shape)
    write_png(path, img)


def verge_texture(path, size=512, seed=2):
    rng = np.random.default_rng(seed)
    n = smooth_2d(rng.standard_normal((size, size)), 2)
    n = (n - n.min()) / (np.ptp(n) + 1e-9)
    grass = np.array([0.33, 0.36, 0.20])
    dirt = np.array([0.45, 0.39, 0.29])
    img = grass * n[..., None] + dirt * (1 - n[..., None])
    img += 0.04 * rng.standard_normal(img.shape)
    write_png(path, img)


def smooth_2d(a, r):
    for _ in range(r):
        a = (a + np.roll(a, 1, 0) + np.roll(a, -1, 0) + np.roll(a, 1, 1) + np.roll(a, -1, 1)) / 5
    return a


# ------------------------------------------------------------------- widths
def road_widths(s, x, y, cfg, p):
    """Carriageway width at every station. "auto": from the IGN BD TOPO
    `largeur_de_chaussee` of the nearest segment carrying the road number,
    with smooth transitions between segments."""
    if p["road_width_m"] != "auto":
        return np.full(len(s), float(p["road_width_m"]))
    import re

    raw = cfg["_dir"] / "route_ign_raw.json"
    if raw.exists():
        feats = json.loads(raw.read_text(encoding="utf-8"))
    else:
        from shapely.geometry import box

        from fetch import wfs_features

        print("  fetching BD TOPO road segments for the widths…")
        area = box(x.min(), y.min(), x.max(), y.max()).buffer(500)
        feats = wfs_features(cfg.get("ign_road_layer", "BDTOPO_V3:troncon_de_route"), area)
        raw.write_text(json.dumps(feats), encoding="utf-8")
    from fetch import _to_l93_if_needed
    from shapely.geometry import shape

    rx = re.compile(cfg["osm_ref_regex"])
    pts, wid = [], []
    for f in feats:
        props = f.get("properties") or {}
        w = props.get(p["width_field"])
        if not isinstance(w, (int, float)) or w <= 0:
            continue
        if not any(isinstance(v, str) and rx.match(v.strip()) for v in props.values()):
            continue
        g = _to_l93_if_needed(shape(f["geometry"]))
        for part in (g.geoms if g.geom_type == "MultiLineString" else [g]):
            n = max(2, int(part.length / 2) + 1)  # a point every ~2 m
            for v in np.linspace(0, part.length, n):
                q = part.interpolate(v)
                pts.append((q.x, q.y))
                wid.append(float(w))
    w = np.full(len(s), np.nan)
    if pts:
        dist, k = cKDTree(np.array(pts)).query(np.column_stack([x, y]), distance_upper_bound=20)
        ok = np.isfinite(dist)
        w[ok] = np.array(wid)[k[ok]]
    known = np.isfinite(w).mean()
    w = np.where(np.isfinite(w), w, p["default_width_m"])
    step = float(np.median(np.diff(s)))
    w = median_filter(w, size=int(40 / step) | 1, mode="nearest")   # ignore tiny segments
    w = gaussian_filter1d(w, 10 / step, mode="nearest")             # ~30 m transitions
    # BD TOPO one-way roundabout legs are 3 m: too narrow to drive, the real
    # ones have extra width in the curve
    w = np.clip(w, p["min_width_m"], 14.0)
    vals, counts = np.unique(np.round(w), return_counts=True)
    print(f"  width from BD TOPO on {100 * known:.0f} % of the road: "
          + ", ".join(f"{v:.0f} m ({100 * c / len(w):.0f} %)" for v, c in zip(vals, counts)))
    return w


def road_extent(s, x, y, cfg, p):
    """Right (negative) and left (positive) carriageway edge offsets from the
    axis at every station. "auto": measured LiDAR edges (road_edges.csv) where
    available, BD TOPO width centred on the measured road centre elsewhere."""
    mode = p["road_width_m"]
    if mode not in ("auto", "bdtopo"):
        w = float(mode)
        return np.full(len(s), -w / 2), np.full(len(s), w / 2)
    w_bd = road_widths(s, x, y, cfg, {**p, "road_width_m": "auto"})
    edges = cfg["_dir"] / "road_edges.csv"
    if mode == "bdtopo" or not edges.exists():
        if mode == "auto":
            print("  (no road_edges.csv: run `python road_edges.py` to use the measured width)")
        return -w_bd / 2, w_bd / 2
    e = np.genfromtxt(edges, delimiter=",", names=True)
    if len(e) != len(s):
        raise SystemExit("road_edges.csv is out of date: run `python road_edges.py` again.")
    step = float(np.median(np.diff(s)))
    measured = gaussian_filter1d(e["measured"].astype(float), 15 / step, mode="nearest") > 0.5
    centre = np.where(measured, e["centre_offset_m"], np.nan)
    ok = np.isfinite(centre)
    centre = np.interp(s, s[ok], centre[ok]) if ok.sum() > 1 else np.zeros(len(s))
    right = np.where(measured, e["right_m"], centre - w_bd / 2)
    left = np.where(measured, e["left_m"], centre + w_bd / 2)
    right = gaussian_filter1d(right, 5 / step, mode="nearest")  # soften the joins
    left = gaussian_filter1d(left, 5 / step, mode="nearest")
    width = left - right
    lo = p["min_width_m"]
    if (width < lo).any():  # keep it drivable, around the same centre
        c_ = (left + right) / 2
        width = np.maximum(width, lo)
        right, left = c_ - width / 2, c_ + width / 2
    print(f"  width from LiDAR edges on {100 * measured.mean():.0f} % of the road "
          f"(BD TOPO elsewhere): median {np.median(width):.2f} m, "
          f"5-95 % {np.percentile(width, 5):.2f}-{np.percentile(width, 95):.2f} m, "
          f"road centre up to {np.percentile(np.abs((left + right) / 2), 95):.2f} m off the OSM axis")
    return right, left


# ---------------------------------------------------------------- geometry
def build_road(s, x, y, z, cf_pct, right, left, p):
    step = float(np.median(np.diff(s)))
    tx, ty, nx, ny = frames(x, y, step)
    zs, c = smooth_profile(s, z, cf_pct, p)
    # columns go from the right edge to the left edge
    t = right[:, None] + (left - right)[:, None] * np.linspace(0, 1, p["road_cross_vertices"])[None, :]
    X = x[:, None] + t * nx[:, None]
    Y = y[:, None] + t * ny[:, None]
    Z = zs[:, None] + c[:, None] * t
    U = np.broadcast_to(np.linspace(0, 1, t.shape[1]), X.shape)
    V = np.broadcast_to((s / p["texture_repeat_m"])[:, None], X.shape)
    edges = {"right": (X[:, 0], Y[:, 0], Z[:, 0]), "left": (X[:, -1], Y[:, -1], Z[:, -1])}
    frame = {"tx": tx, "ty": ty, "nx": nx, "ny": ny, "z": zs, "c": c, "right": right, "left": left}
    return (X, Y, Z, U, V), edges, frame


def chunk_rows(n_rows, rows_per_chunk):
    starts = list(range(0, n_rows - 1, rows_per_chunk))
    return [(a, min(a + rows_per_chunk, n_rows - 1)) for a in starts]  # inclusive end row


def build_verges(x, y, edges, frame, dtm, p):
    """Right and left strips: road edge → DTM at verge_m further out."""
    out = {}
    for side, sign in (("right", -1), ("left", 1)):
        ex, ey, ez = edges[side]
        ox = ex + sign * p["verge_m"] * frame["nx"]
        oy = ey + sign * p["verge_m"] * frame["ny"]
        oz = dtm.sample(ox, oy)
        oz = np.where(np.isfinite(oz), oz, ez - 0.3)
        mx, my = (ex + ox) / 2, (ey + oy) / 2
        mz = np.minimum((ez + oz) / 2, np.maximum(ez, oz) - 0.05)  # slight ditch shape
        cols = [(ex, ey, ez), (mx, my, mz), (ox, oy, oz)]
        if side == "right":  # keep columns ordered right → left, like the road
            cols = cols[::-1]
        X = np.column_stack([c_[0] for c_ in cols])
        Y = np.column_stack([c_[1] for c_ in cols])
        Z = np.column_stack([c_[2] for c_ in cols])
        out[side] = (X, Y, Z)
    return out


def build_terrain(st_x, st_y, frame, dtm, ortho_tiles, p):
    """One grid mesh per ortho tile; vertices kept within terrain_half_width_m
    of the road; vertices under the road/verges pushed below the road."""
    tree = cKDTree(np.column_stack([st_x, st_y]))
    g = p["terrain_grid_m"]
    meshes = []
    for tile in ortho_tiles:
        minx, miny, maxx, maxy = tile["bounds"]
        xs = np.arange(minx, maxx + g / 2, g)
        ys = np.arange(miny, maxy + g / 2, g)
        GX, GY = np.meshgrid(xs, ys[::-1])  # rows north → south
        gx, gy = GX.ravel(), GY.ravel()
        dist, k = tree.query(np.column_stack([gx, gy]), distance_upper_bound=p["terrain_half_width_m"] + 2 * g)
        near = np.isfinite(dist)
        if not near.any():
            continue
        k = np.where(near, k, 0)
        z = np.full(len(gx), np.nan)
        z[near] = dtm.sample(gx[near], gy[near])
        # lateral offset to the axis → road height there, push terrain under it
        tt = (gx - st_x[k]) * frame["nx"][k] + (gy - st_y[k]) * frame["ny"][k]
        r_, l_ = frame["right"][k], frame["left"][k]
        z_road = frame["z"][k] + frame["c"][k] * np.clip(tt, r_, l_)
        v_ = p["verge_m"] - 0.1
        under = near & (tt >= r_ - v_) & (tt <= l_ + v_) & (dist <= np.maximum(-r_, l_) + v_)
        z[under] = np.minimum(np.nan_to_num(z[under], nan=z_road[under]), z_road[under]) - 0.5
        keep = near & np.isfinite(z)
        n_rows, n_cols = GX.shape
        faces = grid_faces(n_rows, n_cols)
        faces = faces[keep[faces].all(axis=1)]
        if len(faces) == 0:
            continue
        used = np.unique(faces)
        remap = -np.ones(len(gx), np.int64)
        remap[used] = np.arange(len(used))
        verts = np.column_stack([gx[used], gy[used], z[used]])
        uvs = np.column_stack([(gx[used] - minx) / (maxx - minx), (gy[used] - miny) / (maxy - miny)])
        f = remap[faces].astype(np.int32)
        # split if a tile ever exceeds the KN5 vertex limit (not with default settings)
        meshes.append((tile, verts, f, uvs))
    return meshes


def ortho_tile_list(ortho_dir):
    tiles = []
    for jpg in sorted(Path(ortho_dir).glob("ortho_*.jpg")):
        m = re.match(r"ortho_(-?\d+)_(-?\d+)$", jpg.stem)
        jgw = jpg.with_suffix(".jgw")
        if not m or not jgw.exists():
            continue
        a, _, _, e, cx, cy = [float(v) for v in jgw.read_text().split()]
        from PIL import Image

        w, h = Image.open(jpg).size
        minx, maxy = cx - a / 2, cy - e / 2
        tiles.append({"file": jpg, "bounds": (minx, maxy + e * h, minx + a * w, maxy)})
    return tiles


def dummies(s, x, y, frame, p):
    """AC empties. Forward = direction of travel, cars drive on the right."""
    L = s[-1]

    def k_at(sv):
        return int(np.clip(np.searchsorted(s, sv), 0, len(s) - 1))

    def at(sv, t, name):
        k = k_at(sv)
        px = x[k] + t * frame["nx"][k]
        py = y[k] + t * frame["ny"][k]
        pz = frame["z"][k] + frame["c"][k] * t + 1.0  # 1-2 m above the road (AC convention)
        return {"name": name, "pos": [px, py, pz], "fwd": [float(frame["tx"][k]), float(frame["ty"][k]), 0.0]}

    s0 = p["start_offset_m"]
    r = lambda sv: float(frame["right"][k_at(sv)])
    l = lambda sv: float(frame["left"][k_at(sv)])
    lane = lambda sv: (3 * r(sv) + l(sv)) / 4  # centre of the right-hand lane
    return [
        at(s0, lane(s0), "AC_START_0"),
        at(s0, lane(s0), "AC_HOTLAP_START_0"),
        at(s0 + 10, lane(s0 + 10), "AC_PIT_0"),
        at(s0 + 20, r(s0 + 20) - 0.5, "AC_AB_START_R"),
        at(s0 + 20, l(s0 + 20) + 0.5, "AC_AB_START_L"),
        at(L - 20, r(L - 20) - 0.5, "AC_AB_FINISH_R"),
        at(L - 20, l(L - 20) + 0.5, "AC_AB_FINISH_L"),
    ]


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="config.yaml")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    p = {**DEFAULTS, **(cfg.get("track") or {})}
    d = cfg["_dir"]
    prof = d / "road_profile.csv"
    if not prof.exists():
        raise SystemExit("No road profile: run `python road_profile.py` first.")
    mnt = sorted((d / "mnt").glob("*.tif"))
    if not mnt:
        raise SystemExit("No DTM: run `python fetch.py mnt` first.")
    ortho = ortho_tile_list(d / "ortho")
    if not ortho:
        raise SystemExit("No orthophotos: run `python fetch.py ortho` first.")
    _, px_, py_, _, _ = load_profile(prof)
    covered = np.zeros(len(px_), bool)
    for t in ortho:
        x0_, y0_, x1_, y1_ = t["bounds"]
        covered |= (px_ >= x0_) & (px_ < x1_) & (py_ >= y0_) & (py_ < y1_)
    if covered.mean() < 0.999:
        raise SystemExit(f"Orthophotos only cover {100 * covered.mean():.0f} % of the road "
                         f"({len(ortho)} images): the download was interrupted. Run "
                         "`python fetch.py ortho` again (it resumes where it stopped).")

    out = d / "track_build"
    tex = out / "texture"
    if out.exists():
        shutil.rmtree(out)
    tex.mkdir(parents=True)

    s, x, y, z, cf = load_profile(prof)
    dtm = FastDTM(mnt)
    print(f"Road: {s[-1] / 1000:.2f} km, {len(s)} stations")
    right, left = road_extent(s, x, y, cfg, p)
    widths = left - right
    (X, Y, Z, U, V), edges, frame = build_road(s, x, y, z, cf, right, left, p)
    verges = build_verges(x, y, edges, frame, dtm, p)
    print("Terrain…")
    terr = build_terrain(x, y, frame, dtm, ortho, p)

    # local origin (kept in the manifest to go back to Lambert-93)
    x0 = round(float(np.mean(x)), -1)
    y0 = round(float(np.mean(y)), -1)
    z0 = math.floor(float(np.nanmin(z)) / 10) * 10 - 50

    def local(v):
        return np.column_stack([v[:, 0] - x0, v[:, 1] - y0, v[:, 2] - z0]).astype(np.float32)

    arrays, objects = {}, []

    def add(name, material, verts, faces, uvs, physical_surface=None, smooth_shading=True):
        assert len(verts) <= AC_MAX_VERTS, (name, len(verts))
        assert np.isfinite(verts).all(), name
        faces = fix_winding_up(verts, faces.copy())
        i = len(objects)
        arrays[f"v{i}"], arrays[f"f{i}"], arrays[f"uv{i}"] = local(verts), faces, uvs.astype(np.float32)
        objects.append({"name": name, "material": material, "key": i,
                        "surface": physical_surface, "smooth": smooth_shading,
                        "verts": int(len(verts)), "tris": int(len(faces))})

    # AC physics: mesh names "<n><SURFACE>" where SURFACE is a key of surfaces.ini
    phys = {"ROAD": 0, "GRASS": 0}

    def phys_name(surface, tag):
        phys[surface] += 1
        return f"{phys[surface]}{surface}_{tag}"

    n_cols = X.shape[1]
    rows_per_chunk = int(p["road_chunk_m"] / np.median(np.diff(s)))
    for ci, (r0, r1) in enumerate(chunk_rows(X.shape[0], rows_per_chunk)):
        sl = slice(r0, r1 + 1)
        verts = np.column_stack([X[sl].ravel(), Y[sl].ravel(), Z[sl].ravel()])
        uvs = np.column_stack([U[sl].ravel(), V[sl].ravel()])
        add(phys_name("ROAD", f"{ci:03d}"), "road", verts, grid_faces(r1 - r0 + 1, n_cols), uvs, "ROAD")
        for side in ("right", "left"):
            VX, VY, VZ = (arr[sl] for arr in verges[side])
            vv = np.column_stack([VX.ravel(), VY.ravel(), VZ.ravel()])
            vuv = np.column_stack([np.tile(np.linspace(0, 1, VX.shape[1]), VX.shape[0]),
                                   np.repeat(s[sl] / 4.0, VX.shape[1])])
            add(phys_name("GRASS", f"verge_{side[0]}{ci:03d}"), "verge", vv,
                grid_faces(r1 - r0 + 1, VX.shape[1]), vuv, "GRASS")

    materials = {"road": "asphalt.png", "verge": "verge.png"}
    for tile, verts, faces, uvs in terr:
        mname = f"ortho_{int(tile['bounds'][0])}_{int(tile['bounds'][1])}"
        materials[mname] = tile["file"].name
        shutil.copy(tile["file"], tex / tile["file"].name)
        add(phys_name("GRASS", f"terrain_{mname[6:]}"), mname, verts, faces, uvs, "GRASS")

    asphalt_texture(tex / "asphalt.png", float(np.median(widths)), p["texture_repeat_m"])
    verge_texture(tex / "verge.png")

    dm = dummies(s, x, y, frame, p)
    for dd in dm:
        dd["pos"] = [dd["pos"][0] - x0, dd["pos"][1] - y0, dd["pos"][2] - z0]

    np.savez_compressed(out / "meshes.npz", **arrays)
    manifest = {
        "track_id": p["id"], "title": p["title"],
        "origin_lambert93": [x0, y0, z0],
        "length_m": float(s[-1]),
        "objects": objects, "materials": materials, "dummies": dm,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    # AC track folder skeleton: the kn5 saved by ksEditor goes next to ui/
    from fetch import TO_WGS

    lon, lat = TO_WGS.transform(float(np.mean(x)), float(np.mean(y)))
    ui = out / "ac_track" / p["id"] / "ui"
    ui.mkdir(parents=True)
    (ui / "ui_track.json").write_text(json.dumps({
        "name": p["title"],
        "description": "Generated by realroad from open data (IGN LiDAR HD, BD TOPO, "
                       "BD ORTHO © IGN, Etalab 2.0; OpenStreetMap © contributors, ODbL).",
        "tags": ["point to point", "mountain", "public road", "france"],
        "geotags": [f"{lat:.5f}", f"{lon:.5f}"],
        "country": "France",
        "city": p.get("city", ""),
        "length": f"{s[-1]:.0f}",
        "width": f"{np.median(widths):.1f}",
        "pitboxes": "1",
        "run": "A2B",
        "author": p.get("author", ""),
        "version": "0.1",
    }, indent=1, ensure_ascii=False), encoding="utf-8")
    nv = sum(o["verts"] for o in objects)
    nt = sum(o["tris"] for o in objects)
    print(f"{len(objects)} meshes, {nv:,} vertices, {nt:,} triangles, "
          f"{len(materials)} materials, {len(dm)} AC dummies")
    print(f"→ {out}")


if __name__ == "__main__":
    main()
