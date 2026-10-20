#!/usr/bin/env python3
"""
Pipeline stage 3: build the track geometry for Assetto Corsa.

    python build_track.py      → data/<name>/track_build/  (meshes.npz, manifest.json, texture/)

then write the AC track (stage 4):

    python kn5_export.py --install "<assettocorsa folder>"

(the older route, still working: Blender → FBX → ksEditor, see blender_export.py)

Geometry, all in metres:
  - road: ribbon built from road_profile.csv (LiDAR elevation and crossfall),
    edges measured by road_edges.py (BD TOPO width where not measured),
    physical surface ROAD
  - markings: the real centre/edge lines from road_markings.py, thin strips
    2 cm above the asphalt (visual only); generic ones in the texture otherwise
  - verges: orthophoto strips blending from the road edge into the DTM, surface GRASS
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
    "verge_m": 2.5,              # blend from the road edge to the terrain over this width
    "verge_outer_m": 7.0,        # the verge strip continues (just under the terrain) to here
    "verge_texture": "ortho",    # "ortho" = the orthophoto (blends with the terrain), "procedural"
    "roadside_photos": True,     # bake the street-level photos (fetch.py streetview) on the verges
    "roadside_band_m": 12.0,     # verge width when photos are used
    "roadside_piece_m": 50.0,    # one baked texture per piece of verge
    "roadside_texture_px": [1024, 256],
    "terrain_half_width_m": 120.0,
    "terrain_grid_m": 2.5,
    "far_terrain_m": 290.0,      # coarse terrain ring out to this distance (0 = off); data reach ~300 m
    "far_terrain_grid_m": 10.0,
    "backdrop_grid_m": 50.0,     # distant relief (fetch.py far), visual only
    "trees": True,               # billboards from trees.csv (trees.py) when present
    "buildings": True,           # from buildings_l93.geojson (fetch.py buildings) when present
    "buildings_max_m": 200.0,
    "buildings_min_area_m2": 15.0,
    "buildings_skip_light": True,    # BD TOPO construction_legere (sheds, greenhouses)
    "shaders": "advanced",       # "advanced": Kunos detail / normal-mapped shaders (road, rock, terrain);
                                 # "basic": plain ksPerPixel everywhere (v0.5 look)
    "rock_slope_deg": 38.0,      # bare (not green on the photo) triangles steeper than this, and all
                                 # faces steeper than 62°, get the rock material
    "rock_repeat_m": 4.0,
    "terrain_mask_px": 256,      # resolution of the terrain shader masks (per 400 m orthophoto tile)
    "csp_trees": "auto",         # CSP 3D trees (trees3d.py): auto = when data/<name>/csp_trees/*.bin exist
    "furniture": True,           # guard rails, walls, edge posts, signs, poles (furniture.py)
    "furniture_posts": True,     # J6 edge posts (laid out by the rules, not surveyed)
    "furniture_lidar": False,    # guard rails / walls from roadside_points.npz: too few points on this
                                 # road (parapets not classified), rock faces taken for walls
    "furniture_parapets": True,  # low walls on the valley side, laid out by the terrain drop
    "parapet_drop_m": 1.0,       # terrain this much below the road edge within 2.5 m
    "parapet_min_run_m": 20.0,
    "parapet_height_m": 0.65,
    "furniture_lidar_classes": [1, 6, 64, 67],
    "furniture_min_points": 3,   # LiDAR points per metre of road to count as a barrier
    "furniture_min_run_m": 10.0,
    "furniture_power_max_m": 120.0,
    "z_sigma_m": 6.0,
    "crossfall_sigma_m": 10.0,
    "despike_window_m": 15.0,
    "z_despike_m": 0.06,
    "crossfall_despike_pct": 3.0,
    "crossfall_clip_pct": 10.0,
    "road_chunk_m": 200.0,
    "texture_repeat_m": 13.0,
    "markings": "auto",          # "auto" = real markings from road_markings.csv when present, "texture", "none"
    "marking_width_m": {"centre": 0.12, "left": 0.15, "right": 0.15},
    "marking_lift_m": 0.02,      # above the asphalt (visual only, no physics)
    "marking_inset_m": 0.3,      # edge line centre, inside the carriageway edge
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


def asphalt_texture(path, width_m, length_m, px_per_m=80, seed=1, paint_lines=True):
    """Asphalt, with generic French markings painted in (dashed T1 centre line,
    continuous edge lines, 12 cm) unless the real markings are built as meshes."""
    rng = np.random.default_rng(seed)
    w, h = int(width_m * px_per_m), int(length_m * px_per_m)
    base = 0.24 + 0.05 * rng.standard_normal((h, w))
    blur = smooth_2d(rng.standard_normal((h // 8 + 1, w // 8 + 1)), 1)
    base += 0.03 * np.kron(blur, np.ones((8, 8)))[:h, :w]
    img = np.repeat(base[..., None], 3, axis=2) * np.array([1.0, 1.0, 1.02])
    line = int(0.12 * px_per_m)
    edge_in = int(0.25 * px_per_m)
    paint = np.array([0.86, 0.86, 0.84])
    if not paint_lines:
        img += 0.02 * rng.standard_normal(img.shape)
        write_png(path, img)
        return
    img[:, edge_in:edge_in + line] = paint
    img[:, w - edge_in - line:w - edge_in] = paint
    c0 = w // 2 - line // 2
    dash = int(3.0 * px_per_m)
    img[:dash, c0:c0 + line] = paint
    img += 0.02 * rng.standard_normal(img.shape)
    write_png(path, img)


def paint_texture(path, size=64, seed=3):
    rng = np.random.default_rng(seed)
    img = np.array([0.84, 0.84, 0.81]) + 0.03 * rng.standard_normal((size, size, 1))
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

    rx = re.compile(cfg["osm_ref_regex"]) if cfg.get("osm_ref_regex") else None
    pts, wid = [], []
    for f in feats:
        props = f.get("properties") or {}
        w = props.get(p["width_field"])
        if not isinstance(w, (int, float)) or w <= 0:
            continue
        if rx is not None and not any(isinstance(v, str) and rx.match(v.strip()) for v in props.values()):
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
        dist, k = cKDTree(np.array(pts)).query(np.column_stack([x, y]),
                                               distance_upper_bound=20 if rx is not None else 8)
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


def load_markings(cfg, s, p):
    """Painted metres per line from road_markings.csv, or None."""
    if p["markings"] != "auto":
        return None
    f = cfg["_dir"] / "road_markings.csv"
    if not f.exists():
        print("  (no road_markings.csv: generic markings in the asphalt texture; "
              "run `python road_markings.py` for the real ones)")
        return None
    m = np.genfromtxt(f, delimiter=",", names=True)
    if len(m) != len(s):
        print("  ! road_markings.csv is out of date (run `python road_markings.py` again): generic markings")
        return None
    return {k: m[k] > 0 for k in ("right", "centre", "left")}


def build_markings(x, y, frame, painted, p, chunk_rows=500):
    """Thin strips 2 cm above the road for every painted run: list of
    (line, verts, faces, uvs). Rows follow the road stations; one mesh per
    ~500 stations so AC can cull them like the road chunks."""
    out = []
    r, l = frame["right"], frame["left"]
    inset = p["marking_inset_m"]
    offsets = {"right": r + inset, "left": l - inset, "centre": (r + l) / 2}
    n = len(x)
    for line, on in painted.items():
        t = offsets[line]
        hw = p["marking_width_m"][line] / 2
        cols = np.stack([t - hw, t + hw], 1)                         # right → left
        X = x[:, None] + cols * frame["nx"][:, None]
        Y = y[:, None] + cols * frame["ny"][:, None]
        Z = frame["z"][:, None] + frame["c"][:, None] * cols + p["marking_lift_m"]
        V, F, UV = [], [], []
        base = 0
        m = np.r_[False, on, False].astype(np.int8)
        starts, ends = np.flatnonzero(np.diff(m) == 1), np.flatnonzero(np.diff(m) == -1)
        for a, b in zip(starts, ends):
            b = min(b, n - 1)                                        # run covers stations a..b
            if b <= a:
                continue
            k = b - a + 1
            V.append(np.column_stack([X[a:b + 1].ravel(), Y[a:b + 1].ravel(), Z[a:b + 1].ravel()]))
            UV.append(np.column_stack([np.tile([0.0, 1.0], k), np.repeat(np.arange(k, dtype=float), 2)]))
            F.append(grid_faces(k, 2) + base)
            base += 2 * k
            nxt = starts[starts > a]
            if len(nxt) == 0 or nxt[0] // chunk_rows != a // chunk_rows:
                out.append((line, np.concatenate(V), np.concatenate(F), np.concatenate(UV)))
                V, F, UV, base = [], [], [], 0
        if V:
            out.append((line, np.concatenate(V), np.concatenate(F), np.concatenate(UV)))
    return out


def chunk_rows(n_rows, rows_per_chunk):
    starts = list(range(0, n_rows - 1, rows_per_chunk))
    return [(a, min(a + rows_per_chunk, n_rows - 1)) for a in starts]  # inclusive end row


def build_verges(x, y, edges, frame, dtm, p):
    """Right and left strips from the road edge to the terrain.

    Columns at 0, 0.4 m (shoulder, 3 cm under the edge), then a smooth blend
    to the DTM at verge_m, then on to verge_outer_m 20 cm under the DTM: the
    terrain is cut out around the road (build_terrain) and overlaps the outer
    part, so there is neither a gap nor triangles poking through."""
    blend, outer = p["verge_m"], p["verge_outer_m"]
    follow = max(blend, p.get("verge_follow_m") or blend)   # follows the DTM exactly up to here
    offs = [0.0, 0.4, 0.4 + (blend - 0.4) / 3, 0.4 + 2 * (blend - 0.4) / 3, blend]
    if follow > blend:
        k = max(1, int(round((follow - blend) / 1.25)))
        offs += list(blend + (follow - blend) * np.arange(1, k + 1) / k)
    offs += [(follow + outer) / 2, outer]
    offs = np.array(offs)
    out = {"offs": offs}
    for side, sign in (("right", -1), ("left", 1)):
        ex, ey, ez = edges[side]
        cols = []
        for o in offs:
            ox = ex + sign * o * frame["nx"]
            oy = ey + sign * o * frame["ny"]
            if o == 0:
                oz = ez
            elif o <= 0.4 + 1e-9:
                oz = ez - 0.03
            else:
                d_ = dtm.sample(ox, oy)
                d_ = np.where(np.isfinite(d_), d_, ez - 0.3)
                if o <= blend + 1e-9:
                    w = (o - 0.4) / (blend - 0.4)
                    w = w * w * (3 - 2 * w)                       # smoothstep
                    oz = (ez - 0.03) * (1 - w) + d_ * w
                elif o <= follow + 1e-9:
                    oz = d_
                else:
                    oz = d_ - 0.2
            cols.append((ox, oy, oz))
        if side == "right":  # keep columns ordered right → left, like the road
            cols = cols[::-1]
        out[side] = tuple(np.column_stack([c_[k] for c_ in cols]) for k in range(3))
    return out


def tile_runs(px, py, tiles):
    """Split rows into runs lying in the same ortho tile (by the given point);
    consecutive runs share their boundary row so the strip stays closed."""
    idx = np.full(len(px), -1)
    for i, t in enumerate(tiles):
        x0, y0, x1, y1 = t["bounds"]
        idx[(idx < 0) & (px >= x0) & (px < x1) & (py >= y0) & (py < y1)] = i
    runs_, a = [], 0
    for k in range(1, len(px) + 1):
        if k == len(px) or idx[k] != idx[a]:
            runs_.append((a, min(k, len(px) - 1), idx[a]))
            a = k
    return runs_


def build_terrain(st_x, st_y, frame, dtm, ortho_tiles, p):
    """One grid mesh per ortho tile; vertices kept within terrain_half_width_m
    of the road; cut out under the road and the verge blend."""
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
        # cut the terrain out under the road and the verge blend: the verge
        # strip (down to verge_outer_m, just under the DTM) fills the hole
        follow = max(p["verge_m"], p.get("verge_follow_m") or p["verge_m"])
        cut = p["verge_outer_m"] >= follow + 1.5 * g
        v_ = follow if cut else p["verge_m"] - 0.1
        under = near & (tt >= r_ - v_) & (tt <= l_ + v_) & (dist <= np.maximum(-r_, l_) + v_)
        if cut:
            z[under] = np.nan
        else:  # old behaviour: push it under the road
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


def build_far_terrain(st_x, st_y, dtm, ortho_tiles, p):
    """Coarse ring from just inside the detailed terrain out to far_terrain_m,
    0.3 m lower so the detailed mesh always wins where they overlap. Distances
    are to the nearest road point anywhere, so it never covers another part
    of the road (hairpins)."""
    if not p["far_terrain_m"] or p["far_terrain_m"] <= p["terrain_half_width_m"]:
        return []
    tree = cKDTree(np.column_stack([st_x, st_y]))
    g = p["far_terrain_grid_m"]
    inner = p["terrain_half_width_m"] - 2 * g
    meshes = []
    for tile in ortho_tiles:
        minx, miny, maxx, maxy = tile["bounds"]
        xs = np.arange(minx, maxx + g / 2, g)
        ys = np.arange(miny, maxy + g / 2, g)
        GX, GY = np.meshgrid(xs, ys[::-1])
        gx, gy = GX.ravel(), GY.ravel()
        dist, _ = tree.query(np.column_stack([gx, gy]), distance_upper_bound=p["far_terrain_m"] + g)
        ring = np.isfinite(dist) & (dist >= inner)
        if not ring.any():
            continue
        z = np.full(len(gx), np.nan)
        z[ring] = dtm.sample(gx[ring], gy[ring]) - 0.3
        keep = ring & np.isfinite(z)
        faces = grid_faces(*GX.shape)
        faces = faces[keep[faces].all(axis=1)]
        if len(faces) == 0:
            continue
        used = np.unique(faces)
        remap = -np.ones(len(gx), np.int64)
        remap[used] = np.arange(len(used))
        verts = np.column_stack([gx[used], gy[used], z[used]])
        uvs = np.column_stack([(gx[used] - minx) / (maxx - minx), (gy[used] - miny) / (maxy - miny)])
        meshes.append((tile, verts, remap[faces].astype(np.int32), uvs))
    return meshes


def build_backdrop(st_x, st_y, d, p):
    """Distant relief for the horizon from fetch.py far: 50 m grid per 4 km
    tile, from just inside the far terrain ring outwards. Lowered 3 m near the
    ring so the finer terrain always wins. Visual only (no physics)."""
    folder = d / "backdrop"
    dem_f = folder / "backdrop_dem.tif"
    if not dem_f.exists():
        print("  (no distant relief: run `python fetch.py far` for the horizon)")
        return []
    tiles = ortho_tile_list(folder, "backdrop_*.jpg")
    dem = FastDTM([dem_f])
    tree = cKDTree(np.column_stack([st_x, st_y]))
    g = p["backdrop_grid_m"]
    start = max(p["far_terrain_m"] or 0, p["terrain_half_width_m"])
    meshes = []
    for tile in tiles:
        minx, miny, maxx, maxy = tile["bounds"]
        xs = np.arange(minx, maxx + g / 2, g)
        ys = np.arange(miny, maxy + g / 2, g)
        GX, GY = np.meshgrid(xs, ys[::-1])
        gx, gy = GX.ravel(), GY.ravel()
        dist, _ = tree.query(np.column_stack([gx, gy]))
        keep = dist >= start - 1.5 * g
        if not keep.any():
            continue
        z = np.full(len(gx), np.nan)
        z[keep] = dem.sample(gx[keep], gy[keep]) - np.where(dist[keep] < start + 2 * g, 3.0, 0.0)
        keep &= np.isfinite(z)
        faces = grid_faces(*GX.shape)
        faces = faces[keep[faces].all(axis=1)]
        if len(faces) == 0:
            continue
        used = np.unique(faces)
        remap = -np.ones(len(gx), np.int64)
        remap[used] = np.arange(len(used))
        verts = np.column_stack([gx[used], gy[used], z[used]])
        uvs = np.column_stack([(gx[used] - minx) / (maxx - minx), (gy[used] - miny) / (maxy - miny)])
        meshes.append((tile, verts, remap[faces].astype(np.int32), uvs))
    return meshes


def apply_advanced_shaders(materials, near_mats, far_only, ortho, dtm, tex, p, n_rock):
    """Kunos shaders with detail and normal maps (setups copied from Kunos
    tracks): the road gets asphalt grain, relief and specular maps
    (ksPerPixelMultiMap_NMDetail), steep faces a limestone rock texture with
    relief (ksPerPixelNM), the terrain near the road the orthophoto plus
    rock / grass / gravel detail layers chosen by a mask made from the photo
    colours and the slope (ksMultilayer_fresnel_nm)."""
    from PIL import Image

    from textures import road_textures, rock_textures, terrain_detail_textures, terrain_mask

    # road: diffuse with a (nearly) black alpha = the detail texture shows everywhere
    road_textures(tex, tex / "asphalt.png")
    a = Image.open(tex / "asphalt.png").convert("RGB")
    a.putalpha(Image.new("L", a.size, 5))
    a.save(tex / "asphalt_a.png")
    materials["road"] = {"texture": "asphalt_a.png", "shader": "ksPerPixelMultiMap_NMDetail", "kind": "road_adv",
                         "dds": "dxt5", "maps": [["txNormal", 1, "road_nm.png", "dxt1"],
                                                 ["txMaps", 2, "road_maps.png", "dxt1"],
                                                 ["txDetail", 3, "road_detail.png", "dxt5"],
                                                 ["txNormalDetail", 4, "road_detail_nm.png", "dxt1"]]}
    if n_rock:
        rock_textures(tex)
        materials["rock"] = {"texture": "rock.png", "shader": "ksPerPixelNM", "kind": "rock",
                             "maps": [["txNormal", 1, "rock_nm.png", "dxt1"]]}
    terrain_detail_textures(tex)
    n = p["terrain_mask_px"]
    n_ml = 0
    for t_ in ortho:
        mname = f"ortho_{int(t_['bounds'][0])}_{int(t_['bounds'][1])}"
        if mname not in near_mats or mname in far_only or mname not in materials:
            continue
        x0_, y0_, x1_, y1_ = t_["bounds"]
        img = np.asarray(Image.open(t_["file"]).convert("RGB").resize((n, n), Image.BILINEAR), np.float32) / 255
        px_ = (x1_ - x0_) / n
        gx = x0_ + (np.arange(n) + 0.5) * px_
        gy = y1_ - (np.arange(n) + 0.5) * px_
        GX, GY = np.meshgrid(gx, gy)
        z = dtm.sample(GX.ravel(), GY.ravel()).reshape(n, n)
        z = np.where(np.isfinite(z), z, np.nanmean(z) if np.isfinite(z).any() else 0.0)
        dzy, dzx = np.gradient(z, px_)
        slope = np.degrees(np.arctan(np.hypot(dzx, dzy)))
        m = terrain_mask(img, slope)
        Image.fromarray((m * 255 + 0.5).clip(0, 255).astype(np.uint8), "RGBA").save(tex / f"mask_{mname}.png")
        materials[mname] = {"texture": t_["file"].name, "shader": "ksMultilayer_fresnel_nm", "kind": "terrain_ml",
                            "maps": [["txMask", 1, f"mask_{mname}.png", "dxt5"],
                                     ["txDetailR", 2, "det_rock.png", "dxt5"],
                                     ["txDetailG", 3, "det_grass.png", "dxt5"],
                                     ["txDetailB", 4, "det_soil.png", "dxt5"],
                                     ["txDetailA", 5, "det_grass.png", "dxt5"],
                                     ["txDetailNM", 21, "det_flat_nm.png", "dxt1"]]}
        n_ml += 1
    print(f"  shaders: road with detail and relief, {n_rock:,} rock triangles, {n_ml} terrain tiles "
          f"with detail layers")


def ortho_tile_list(ortho_dir, pattern="ortho_*.jpg"):
    tiles = []
    for jpg in sorted(Path(ortho_dir).glob(pattern)):
        m = re.match(r"[a-z]+_(-?\d+)_(-?\d+)$", jpg.stem)
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
    photos = None
    if p["roadside_photos"] and (d / "streetview" / "index.json").exists():
        from roadside import Photos

        photos = Photos(d / "streetview", s, x, y, frame["z"], (frame["right"] + frame["left"]) / 2)
        if len(photos.items) < 2:
            photos = None
        else:
            p["verge_outer_m"] = max(p["verge_outer_m"], p["roadside_band_m"])
            p["verge_follow_m"] = p["verge_outer_m"] - 3.0
            print(f"  roadside from {len(photos.items)} street-level photos (band {p['verge_outer_m']:.0f} m)")
    verges = build_verges(x, y, edges, frame, dtm, p)
    print("Terrain…")
    terr = build_terrain(x, y, frame, dtm, ortho, p)
    far = build_far_terrain(x, y, dtm, ortho, p)
    backdrop = build_backdrop(x, y, d, p)

    # local origin (kept in the manifest to go back to Lambert-93)
    x0 = round(float(np.mean(x)), -1)
    y0 = round(float(np.mean(y)), -1)
    z0 = math.floor(float(np.nanmin(z)) / 10) * 10 - 50

    def local(v):
        return np.column_stack([v[:, 0] - x0, v[:, 1] - y0, v[:, 2] - z0]).astype(np.float32)

    arrays, objects = {}, []
    half = np.maximum(-frame["right"], frame["left"])
    axis_tree = cKDTree(np.column_stack([x, y]))

    def add(name, material, verts, faces, uvs, physical_surface=None, smooth_shading=True, keep_winding=False,
            **extra):
        assert len(verts) <= AC_MAX_VERTS, (name, len(verts))
        assert np.isfinite(verts).all(), name
        faces = faces.copy() if keep_winding else fix_winding_up(verts, faces.copy())
        i = len(objects)
        arrays[f"v{i}"], arrays[f"f{i}"], arrays[f"uv{i}"] = local(verts), faces, uvs.astype(np.float32)
        objects.append({"name": name, "material": material, "key": i,
                        "surface": physical_surface, "smooth": smooth_shading,
                        "verts": int(len(verts)), "tris": int(len(faces)), **extra})

    advanced = p["shaders"] == "advanced"
    cos_rock = math.cos(math.radians(p["rock_slope_deg"]))
    cos_cliff = math.cos(math.radians(62.0))
    from roadside import OrthoSampler

    ortho_sampler_any = OrthoSampler(ortho)
    n_rock = [0]

    def add_split(name, material, verts, faces, uvs, surface):
        """Steep triangles (rock faces, road cuts) get the rock material, mapped
        along the road (u = station, v = height) so the texture is not stretched
        down the face like the orthophoto is; the rest keeps its material."""
        if not advanced:
            add(name, material, verts, faces, uvs, surface)
            return
        a_, b_, c_ = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
        nrm = np.cross(b_ - a_, c_ - a_)
        nz = np.abs(nrm[:, 2]) / np.maximum(np.linalg.norm(nrm, axis=1), 1e-9)
        steep = nz < cos_rock
        if steep.any():
            # bare rock only: on the photo, not green and not dark (steep wooded
            # slopes are common); near-vertical faces always
            cen = (a_[steep] + b_[steep] + c_[steep]) / 3
            col = ortho_sampler_any.colours(cen[:, 0], cen[:, 1]) / 255.0
            bare = (col.mean(1) > 0.36) & (col[:, 1] - (col[:, 0] + col[:, 2]) / 2 < 0.03)
            idx = np.flatnonzero(steep)
            steep[idx[~(bare | (nz[steep] < cos_cliff))]] = False
        if steep.any():
            fs = faces[steep]
            used = np.unique(fs)
            remap = -np.ones(len(verts), np.int64)
            remap[used] = np.arange(len(used))
            rv = verts[used]
            ks_ = axis_tree.query(rv[:, :2])[1]
            ruv = np.column_stack([s[ks_] / p["rock_repeat_m"], rv[:, 2] / p["rock_repeat_m"]])
            add(name + "_rock", "rock", rv, remap[fs].astype(np.int32), ruv, surface)
            n_rock[0] += int(steep.sum())
            faces = faces[~steep]
        if len(faces):
            used = np.unique(faces)
            remap = -np.ones(len(verts), np.int64)
            remap[used] = np.arange(len(used))
            add(name, material, verts[used], remap[faces].astype(np.int32), uvs[used], surface)

    # AC physics: mesh names "<n><SURFACE>" where SURFACE is a key of surfaces.ini
    phys = {"ROAD": 0, "GRASS": 0, "WALL": 0}

    def phys_name(surface, tag):
        phys[surface] += 1
        return f"{phys[surface]}{surface}_{tag}"

    materials = {"road": "asphalt.png"}
    if p["verge_texture"] != "ortho":
        materials["verge"] = "verge.png"

    def use_tile(tile):
        mname = f"ortho_{int(tile['bounds'][0])}_{int(tile['bounds'][1])}"
        if mname not in materials:
            materials[mname] = tile["file"].name
            shutil.copy(tile["file"], tex / tile["file"].name)
        return mname

    n_roadside, cov_all = [0], []
    if photos is not None:
        from roadside import OrthoSampler, bake_strip

        ortho_sampler = OrthoSampler(ortho)
    n_cols = X.shape[1]
    rows_per_chunk = int(p["road_chunk_m"] / np.median(np.diff(s)))
    for ci, (r0, r1) in enumerate(chunk_rows(X.shape[0], rows_per_chunk)):
        sl = slice(r0, r1 + 1)
        verts = np.column_stack([X[sl].ravel(), Y[sl].ravel(), Z[sl].ravel()])
        uvs = np.column_stack([U[sl].ravel(), V[sl].ravel()])
        add(phys_name("ROAD", f"{ci:03d}"), "road", verts, grid_faces(r1 - r0 + 1, n_cols), uvs, "ROAD")
        for side in ("right", "left"):
            VX, VY, VZ = (arr[sl] for arr in verges[side])
            if photos is not None:
                # columns from the road edge outwards for the baking
                if side == "right":
                    VX, VY, VZ = VX[:, ::-1], VY[:, ::-1], VZ[:, ::-1]
                step_ = float(np.median(np.diff(s)))
                n_piece = max(2, int(p["roadside_piece_m"] / step_))
                for a_ in range(0, VX.shape[0] - 1, n_piece):
                    b_ = min(a_ + n_piece, VX.shape[0] - 1)
                    rs = slice(a_, b_ + 1)
                    img, puv, cov = bake_strip(VX[rs], VY[rs], VZ[rs], verges["offs"], s[sl][rs], photos,
                                               ortho_sampler, size=tuple(p["roadside_texture_px"]))
                    mname = f"roadside_{side[0]}{n_roadside[0]:04d}"
                    n_roadside[0] += 1
                    img.save(tex / f"{mname}.jpg", quality=90)
                    materials[mname] = {"texture": f"{mname}.jpg", "kind": "terrain"}
                    vv = np.column_stack([VX[rs].ravel(), VY[rs].ravel(), VZ[rs].ravel()])
                    add_split(phys_name("GRASS", f"verge_{side[0]}{ci:03d}_{a_}"), mname, vv,
                              grid_faces(b_ - a_ + 1, VX.shape[1]), puv, "GRASS")
                    cov_all.append(cov)
                    if n_roadside[0] % 50 == 0:
                        print(f"    {n_roadside[0]} roadside textures", flush=True)
                continue
            if p["verge_texture"] == "ortho":
                # orthophoto, mapped like the terrain: the verge blends into it
                mid = VX.shape[1] // 2
                for ri, (a_, b_, ti) in enumerate(tile_runs(VX[:, mid], VY[:, mid], ortho)):
                    if b_ <= a_ or ti < 0:
                        continue
                    t_ = ortho[ti]
                    bx0, by0, bx1, by1 = t_["bounds"]
                    gx_, gy_, gz_ = VX[a_:b_ + 1], VY[a_:b_ + 1], VZ[a_:b_ + 1]
                    vv = np.column_stack([gx_.ravel(), gy_.ravel(), gz_.ravel()])
                    vuv = np.column_stack([np.clip((gx_.ravel() - bx0) / (bx1 - bx0), 0, 1),
                                           np.clip((gy_.ravel() - by0) / (by1 - by0), 0, 1)])
                    mname = use_tile(t_)
                    add_split(phys_name("GRASS", f"verge_{side[0]}{ci:03d}_{ri}"), mname, vv,
                        grid_faces(b_ - a_ + 1, VX.shape[1]), vuv, "GRASS")
            else:
                vv = np.column_stack([VX.ravel(), VY.ravel(), VZ.ravel()])
                vuv = np.column_stack([np.tile(np.linspace(0, 1, VX.shape[1]), VX.shape[0]),
                                       np.repeat(s[sl] / 4.0, VX.shape[1])])
                add(phys_name("GRASS", f"verge_{side[0]}{ci:03d}"), "verge", vv,
                    grid_faces(r1 - r0 + 1, VX.shape[1]), vuv, "GRASS")

    if photos is not None:
        print(f"  roadside: {n_roadside[0]} baked textures, street photos used on "
              f"{100 * float(np.mean(cov_all)) if cov_all else 0:.0f} % of the verge texels")
    painted = load_markings(cfg, s, p)
    if painted is not None:
        marks = build_markings(x, y, frame, painted, p)
        for mi, (line, mv, mf, muv) in enumerate(marks):
            add(f"marking_{line}_{mi:02d}", "marking", mv, mf, muv)   # visual only: no surface prefix
        materials["marking"] = "paint.png"
        paint_texture(tex / "paint.png")
        print(f"  real markings: {len(marks)} meshes, painted "
              + ", ".join(f"{k} {100 * v.mean():.0f} %" for k, v in painted.items()))
    for tile, verts, faces, uvs in terr:
        mname = f"ortho_{int(tile['bounds'][0])}_{int(tile['bounds'][1])}"
        materials[mname] = tile["file"].name
        shutil.copy(tile["file"], tex / tile["file"].name)
        add_split(phys_name("GRASS", f"terrain_{mname[6:]}"), mname, verts, faces, uvs, "GRASS")
    near_mats = set(materials)
    for tile, verts, faces, uvs in far:
        mname = f"ortho_{int(tile['bounds'][0])}_{int(tile['bounds'][1])}"
        if mname not in materials:
            materials[mname] = tile["file"].name
            shutil.copy(tile["file"], tex / tile["file"].name)
        add(phys_name("GRASS", f"far_{mname[6:]}"), mname, verts, faces, uvs, "GRASS")
    for tile, verts, faces, uvs in backdrop:
        mname = f"backdrop_{int(tile['bounds'][0])}_{int(tile['bounds'][1])}"
        materials[mname] = tile["file"].name
        shutil.copy(tile["file"], tex / tile["file"].name)
        add(mname, mname, verts, faces, uvs)                              # visual only
    if backdrop:
        print(f"  distant relief: {len(backdrop)} tiles, {sum(len(b[1]) for b in backdrop):,} vertices")

    # 3D scenery (visual only): trees (trees.py) and buildings (fetch.py buildings)
    from scenery import build_buildings, build_trees
    from textures import facade_texture, foliage_atlas, tree_atlas

    scen = []
    half = np.maximum(-frame["right"], frame["left"])
    axis_tree = cKDTree(np.column_stack([x, y]))
    from trees3d import KIND_SPECIES, MODEL_CROWN_R, MODEL_HEIGHT, SPECIES

    # the tree models are the same for every road: use this road's, or those
    # already converted for another road of the data folder
    csp_dir = d / "csp_trees"
    for cand in [csp_dir] + sorted(d.parent.glob("*/csp_trees")):
        if all((cand / f"{sp}.bin").exists() for sp in SPECIES):
            csp_dir = cand
            break
    csp_bins = [csp_dir / f"{sp}.bin" for sp in SPECIES]
    use_csp = p["csp_trees"] is True or (p["csp_trees"] == "auto" and all(b.exists() for b in csp_bins))
    if use_csp and not all(b.exists() for b in csp_bins):
        print("  ! csp_trees: true but the .bin models are missing (see trees3d.py): flat trees instead")
        use_csp = False
    csp_list = None
    if p["trees"] and (d / "trees.csv").exists():
        tm, ts = build_trees(d / "trees.csv", dtm, ortho, road=(axis_tree, half), tall=not use_csp)
        if use_csp:
            # CSP 3D trees: list in track coordinates (east, up, -north from the local origin)
            rng_ = np.random.default_rng(21)
            csp_list = []
            for tx_, ty_, tz_, th_, tr_, tk_ in ts.pop("tall"):
                opts = KIND_SPECIES.get(int(tk_), ("oak",))
                sp = opts[int(rng_.integers(0, len(opts)))]
                size = th_ / MODEL_HEIGHT[sp]
                crown = tr_
                d_, k_ = axis_tree.query([tx_, ty_])
                edge_ = d_ - half[k_]
                if edge_ < crown:          # do not let the model hang far over the road
                    crown = max(edge_ + 1.0, 0.6 * tr_)
                width = float(np.clip((crown / MODEL_CROWN_R[sp]) / max(size, 1e-3), 0.5, 1.6))
                csp_list.append([sp, round(tx_ - x0, 2), round(tz_ - z0 - 0.1, 2), round(-(ty_ - y0), 2),
                                 round(float(rng_.uniform(0, 360)), 0), round(float(size), 3), round(width, 3)])
            (out / "csp_trees").mkdir(exist_ok=True)
            from trees3d import fix_bin

            for b in csp_bins:
                filled = fix_bin(b, csp_dir if (csp_dir / "leaves.png").exists() else csp_dir / "src",
                                 out / "csp_trees" / b.name)
                if filled:
                    print(f"  {b.name}: textures added ({', '.join(filled)})")
        pal_f = d / "trees_palette.json"
        pal = json.loads(pal_f.read_text()) if pal_f.exists() else {
            "pine": [62, 78, 52], "broadleaf": [88, 100, 66], "shrub": [104, 104, 74]}
        tree_atlas(tex / "trees.png", pal)
        materials["trees"] = {"texture": "trees.png", "shader": "ksTree", "alpha": True, "kind": "trees"}
        if ts.get("cards"):
            foliage_atlas(tex / "foliage.png", pal)
            materials["foliage"] = {"texture": "foliage.png", "shader": "ksTree", "alpha": True, "kind": "trees"}
        scen += tm
        print(f"  trees: {ts['trees']:,} trees{' (CSP 3D models)' if use_csp else ''}, {ts['bushes']:,} bushes, "
              f"{ts['cards']:,} foliage cards, {ts['trunks']:,} solid trunks, {len(tm)} meshes")
    elif p["trees"]:
        print("  (no trees.csv: run `python fetch.py mnh` then `python trees.py` for the trees)")
    if p["buildings"] and (d / "buildings_l93.geojson").exists():
        bm, n_b, n_g = build_buildings(d / "buildings_l93.geojson", dtm, ortho, axis_tree, half, p)
        for m_ in bm:
            if m_[1].startswith("ortho_"):
                use_tile(next(t_ for t_ in ortho if f"ortho_{int(t_['bounds'][0])}_{int(t_['bounds'][1])}" == m_[1]))
        if bm:
            facade_texture(tex / "facade.png")
            materials["facade"] = {"texture": "facade.png", "kind": "facade"}
        scen += bm
        print(f"  buildings: {n_b} ({n_g} with a gable roof)")
    elif p["buildings"]:
        print("  (no buildings_l93.geojson: run `python fetch.py buildings` for the buildings)")
    if p["furniture"]:
        from furniture import Road, build_furniture

        road = Road(s, x, y, frame, dtm, p["verge_m"])
        fm, fmat, fst, runs = build_furniture(d, road, p, cfg.get("osm_ref_regex"), tex)
        (out / "furniture_runs.json").write_text(json.dumps(
            [{k: v for k, v in r_.items() if not k.startswith("off_")} for r_ in runs], indent=1))
        materials.update(fmat)
        scen += fm
        print("  road furniture: " + ", ".join(f"{k.replace('_', ' ')} {v:,}" for k, v in fst.items())
              + ("" if (d / "roadside_points.npz").exists() else
                 "  (no LiDAR roadside points: run `python fetch.py roadside` for the guard rails and walls)"))
    for name, mat, sv, sf, suv, keep, *opt in scen:
        opt = dict(opt[0]) if opt else {}
        surface = opt.pop("surface", None)    # "WALL": cars collide with it
        if surface:
            name = phys_name(surface, name)
        add(name, mat, sv, sf, suv, surface, keep_winding=keep, **opt)
    far_only = sorted(m_ for m_ in set(materials) - near_mats if m_.startswith(("ortho_", "backdrop_")))

    asphalt_texture(tex / "asphalt.png", float(np.median(widths)), p["texture_repeat_m"],
                    paint_lines=painted is None and p["markings"] != "none")
    if "verge" in materials:
        verge_texture(tex / "verge.png")
    if advanced:
        apply_advanced_shaders(materials, near_mats, far_only, ortho, dtm, tex, p, n_rock[0])

    dm = dummies(s, x, y, frame, p)
    for dd in dm:
        dd["pos"] = [dd["pos"][0] - x0, dd["pos"][1] - y0, dd["pos"][2] - z0]

    np.savez_compressed(out / "meshes.npz", **arrays)
    manifest = {
        "track_id": p["id"], "title": p["title"],
        "origin_lambert93": [x0, y0, z0],
        "length_m": float(s[-1]),
        "objects": objects, "materials": materials, "dummies": dm,
        "far_only_materials": far_only,
        "csp_trees": csp_list,            # CSP 3D trees: [species, x, y, z, angle, size, width] (kn5_export.py)   # textures only seen from afar: kn5_export uses a lower resolution
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    # AC track folder skeleton (kn5_export.py fills it; or the kn5 saved by ksEditor)
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
