#!/usr/bin/env python3
"""
Pipeline stage 3: build the track geometry for Assetto Corsa.

    python build_track.py      → data/<name>/track_build/  (meshes.npz, manifest.json, texture/)

then, with Blender (no UI needed):

    blender --factory-startup -b -P blender_export.py -- data/<name>/track_build

Geometry, all in metres:
  - road: ribbon of `road_width_m` built from road_profile.csv (LiDAR elevation
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
from scipy.spatial import cKDTree

from fetch import load_cfg
from preview import smooth

AC_MAX_VERTS = 60_000  # KN5 meshes use 16-bit indices (65 535), keep a margin

DEFAULTS = {
    "id": "realroad_track",
    "title": "realroad track",
    "road_width_m": 6.5,
    "road_cross_vertices": 7,
    "verge_m": 2.5,
    "terrain_half_width_m": 120.0,
    "terrain_grid_m": 2.5,
    "z_smooth_m": 5.0,
    "crossfall_smooth_m": 15.0,
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


# ---------------------------------------------------------------- geometry
def build_road(s, x, y, z, cf_pct, p):
    step = float(np.median(np.diff(s)))
    tx, ty, nx, ny = frames(x, y, step)
    zs = smooth(z, max(1, int(p["z_smooth_m"] / step)))
    c = np.clip(smooth(cf_pct, max(1, int(p["crossfall_smooth_m"] / step))),
                -p["crossfall_clip_pct"], p["crossfall_clip_pct"]) / 100
    half = p["road_width_m"] / 2
    # columns go from right (t = -half) to left (t = +half)
    t = np.linspace(-half, half, p["road_cross_vertices"])
    X = x[:, None] + t[None, :] * nx[:, None]
    Y = y[:, None] + t[None, :] * ny[:, None]
    Z = zs[:, None] + c[:, None] * t[None, :]
    U = np.broadcast_to(np.linspace(0, 1, len(t)), X.shape)
    V = np.broadcast_to((s / p["texture_repeat_m"])[:, None], X.shape)
    edges = {"right": (X[:, 0], Y[:, 0], Z[:, 0]), "left": (X[:, -1], Y[:, -1], Z[:, -1])}
    frame = {"tx": tx, "ty": ty, "nx": nx, "ny": ny, "z": zs, "c": c}
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
    half = p["road_width_m"] / 2
    inner = half + p["verge_m"]
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
        z_road = frame["z"][k] + frame["c"][k] * np.clip(tt, -half, half)
        under = near & (dist <= inner - 0.1)
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
    half = p["road_width_m"] / 2

    def at(sv, t, name):
        k = int(np.clip(np.searchsorted(s, sv), 0, len(s) - 1))
        px = x[k] + t * frame["nx"][k]
        py = y[k] + t * frame["ny"][k]
        pz = frame["z"][k] + frame["c"][k] * t + 0.3
        return {"name": name, "pos": [px, py, pz], "fwd": [float(frame["tx"][k]), float(frame["ty"][k]), 0.0]}

    lane = -half / 2  # centre of the right-hand lane
    s0 = p["start_offset_m"]
    return [
        at(s0, lane, "AC_START_0"),
        at(s0, lane, "AC_HOTLAP_START_0"),
        at(s0 + 10, lane, "AC_PIT_0"),
        at(s0 + 20, -half - 0.5, "AC_AB_START_R"),
        at(s0 + 20, half + 0.5, "AC_AB_START_L"),
        at(L - 20, -half - 0.5, "AC_AB_FINISH_R"),
        at(L - 20, half + 0.5, "AC_AB_FINISH_L"),
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

    out = d / "track_build"
    tex = out / "texture"
    if out.exists():
        shutil.rmtree(out)
    tex.mkdir(parents=True)

    s, x, y, z, cf = load_profile(prof)
    dtm = FastDTM(mnt)
    print(f"Road: {s[-1] / 1000:.2f} km, {len(s)} stations")
    (X, Y, Z, U, V), edges, frame = build_road(s, x, y, z, cf, p)
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

    asphalt_texture(tex / "asphalt.png", p["road_width_m"], p["texture_repeat_m"])
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
        "width": f"{p['road_width_m']:.1f}",
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
