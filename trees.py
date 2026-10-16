#!/usr/bin/env python3
"""
Pipeline stage 2d: the real trees, from the LiDAR HD canopy height model.

    python fetch.py mnh   # canopy height tiles (50 cm GeoTIFF)
    python trees.py       → data/<name>/trees.csv, trees_palette.json, trees.png

The canopy height model (MNH) is the height of the vegetation above the
ground. Every tree top is a local maximum of it:

1. smooth the canopy height a little (σ 0.75 m) so one crown gives one peak,
2. keep the peaks higher than min_height_m that are the maximum within a
   window growing with the height (bigger trees are further apart),
3. drop those on the road (trunk at least clearance_m outside the edge) or
   further than max_distance_m from it,
4. crown radius from the canopy itself: distance where it falls under half
   the tree height, read along 8 directions (bounded by an allometric guess),
5. kind (pine / broadleaf / shrub) from the height and the orthophoto colour
   of the crown; the median crown colours give the palette of the tree
   textures, so the billboards match the photo.

build_track.py turns trees.csv into crossed billboards (shader ksTree).
"""
import argparse
import json

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.spatial import cKDTree

from fetch import load_cfg

DEFAULTS = {
    "min_height_m": 3.0,       # lower vegetation stays in the orthophoto
    "max_distance_m": 140.0,   # from the road axis (detailed terrain: 120 m)
    "clearance_m": 1.0,        # trunk outside the carriageway edge by at least this
    "smooth_m": 0.75,
    "window_base_m": 2.5,      # peak window: base + per_m × height
    "window_per_m": 0.18,
    "max_trees": 60000,        # nearest to the road first
    "pine_min_height_m": 7.0,
}


def read_tiles(files):
    import rasterio

    for f in files:
        with rasterio.open(f) as d:
            a = d.read(1).astype(np.float32)
            nod = d.nodata
            tr = d.transform
        if nod is not None:
            a[a == nod] = np.nan
        a[(a < -1) | (a > 80)] = np.nan
        yield f, a, tr


def crown_radius(chm, r0, c0, h, px, max_r):
    """Mean distance (m) along 8 directions where the canopy drops under h/2."""
    out = []
    n_r, n_c = chm.shape
    steps = int(max_r / px)
    for dr, dc in ((0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)):
        k = np.arange(1, steps + 1)
        rr, cc = r0 + dr * k, c0 + dc * k
        ok = (rr >= 0) & (rr < n_r) & (cc >= 0) & (cc < n_c)
        prof = np.full(steps, 0.0)
        prof[ok] = np.nan_to_num(chm[rr[ok], cc[ok]])
        below = np.flatnonzero(prof < 0.5 * h)
        k_end = below[0] if len(below) else steps
        out.append(k_end * px * (1.414 if dr and dc else 1.0))
    return float(np.median(out))


def detect(cfg, q):
    d = cfg["_dir"]
    files = sorted((d / "mnh").glob("*.tif"))
    if not files:
        raise SystemExit("No canopy height tiles: run `python fetch.py mnh` first.")
    prof = np.genfromtxt(d / "road_profile.csv", delimiter=",", names=True)
    sx, sy = prof["x"], prof["y"]
    half = np.full(len(sx), 3.5)
    if (d / "road_edges.csv").exists():
        e = np.genfromtxt(d / "road_edges.csv", delimiter=",", names=True)
        if len(e) == len(sx):
            half = np.maximum(np.abs(e["right_m"]), np.abs(e["left_m"]))
    tree = cKDTree(np.column_stack([sx, sy]))
    x0, y0 = sx.min() - q["max_distance_m"], sy.min() - q["max_distance_m"]
    x1, y1 = sx.max() + q["max_distance_m"], sy.max() + q["max_distance_m"]
    found = []
    for f, a, tr in read_tiles(files):
        px = tr.a
        bx0, by1 = tr.c, tr.f
        bx1, by0 = bx0 + px * a.shape[1], by1 - px * a.shape[0]
        if bx1 < x0 or bx0 > x1 or by1 < y0 or by0 > y1:
            continue
        sm = gaussian_filter(np.nan_to_num(a), q["smooth_m"] / px)
        # window grows with height: run the filter for a few height classes
        peaks = np.zeros(a.shape, bool)
        for hmin, hmax in ((q["min_height_m"], 8), (8, 15), (15, 25), (25, 80)):
            w = q["window_base_m"] + q["window_per_m"] * (hmin + min(hmax, 30)) / 2
            size = max(3, int(round(w / px)) | 1)
            mx = maximum_filter(sm, size=size, mode="nearest")
            peaks |= (sm == mx) & (sm >= hmin) & (sm < hmax)
        r, c = np.nonzero(peaks)
        if not len(r):
            continue
        x = bx0 + (c + 0.5) * px
        y = by1 - (r + 0.5) * px
        dist, k = tree.query(np.column_stack([x, y]), distance_upper_bound=q["max_distance_m"])
        ok = np.isfinite(dist)
        ok[ok] &= dist[ok] >= half[k[ok]] + q["clearance_m"]
        for ri, ci, xi, yi, di in zip(r[ok], c[ok], x[ok], y[ok], dist[ok]):
            h = float(sm[ri, ci])
            guess = float(np.clip(0.8 + 0.2 * h, 1.2, 7.0))
            cr = crown_radius(sm, ri, ci, h, px, 1.6 * guess)
            found.append((xi, yi, h, float(np.clip(cr, 0.6 * guess, 1.4 * guess)), di))
        print(f"  {f.name}: {ok.sum()} trees", flush=True)
    if not found:
        raise SystemExit("No tree found near the road.")
    t = np.array(found)
    # the same crown can peak in two neighbouring tiles: keep one per 1.5 m
    keep = np.ones(len(t), bool)
    pairs = cKDTree(t[:, :2]).query_pairs(1.5, output_type="ndarray")
    if len(pairs):
        lower = np.where(t[pairs[:, 0], 2] < t[pairs[:, 1], 2], pairs[:, 0], pairs[:, 1])
        keep[lower] = False
    t = t[keep]
    if len(t) > q["max_trees"]:
        t = t[np.argsort(t[:, 4])[:q["max_trees"]]]
        print(f"  capped to the {q['max_trees']} trees nearest to the road")
    return t


def ortho_colours(d, xs, ys, r):
    """Mean orthophoto colour inside each crown (NaN where no image)."""
    import re

    from PIL import Image

    rgb = np.full((len(xs), 3), np.nan)
    for jpg in sorted((d / "ortho").glob("ortho_*.jpg")):
        jgw = jpg.with_suffix(".jgw")
        if not jgw.exists() or not re.match(r"ortho_-?\d+_-?\d+$", jpg.stem):
            continue
        a_, _, _, e_, cx, cy = [float(v) for v in jgw.read_text().split()]
        with Image.open(jpg) as im:
            w, h = im.size
            x0, y1 = cx - a_ / 2, cy - e_ / 2
            x1, y0 = x0 + a_ * w, y1 + e_ * h
            m = (xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)
            if not m.any():
                continue
            img = np.asarray(im.convert("RGB"), dtype=np.float32)
        for i in np.flatnonzero(m):
            cc = int((xs[i] - x0) / a_)
            rr = int((y1 - ys[i]) / -e_)
            k = max(1, int(0.5 * r[i] / a_))
            patch = img[max(0, rr - k):rr + k + 1, max(0, cc - k):cc + k + 1]
            if patch.size:
                rgb[i] = patch.reshape(-1, 3).mean(0)
    return rgb


def classify(t, rgb, q):
    """0 = pine, 1 = broadleaf, 2 = shrub. Aleppo/maritime pines are the
    tall, dark crowns; oaks and others lighter and rounder."""
    h = t[:, 2]
    bright = np.nanmean(rgb, axis=1)
    med = np.nanmedian(bright) if np.isfinite(bright).any() else 100.0
    kind = np.where(h < 4.5, 2, 1)
    dark = np.nan_to_num(bright, nan=med) <= med
    kind[(h >= q["pine_min_height_m"]) & dark] = 0
    return kind


def palette(rgb, kind):
    out = {}
    defaults = {0: [62, 78, 52], 1: [88, 100, 66], 2: [104, 104, 74]}
    for k, name in ((0, "pine"), (1, "broadleaf"), (2, "shrub")):
        sel = rgb[(kind == k) & np.isfinite(rgb).all(1)]
        out[name] = [int(v) for v in (np.median(sel, axis=0) if len(sel) > 20 else defaults[k])]
    return out


def plot(path, t, kind, sx, sy):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2, figsize=(13, 5.5))
    col = np.array(["#2f5d34", "#7aa35a", "#c2b26b"])[kind]
    axs[0].plot(sx, sy, "k-", lw=0.6)
    axs[0].scatter(t[:, 0], t[:, 1], s=0.3, c=col)
    axs[0].set_aspect("equal")
    axs[0].set_title(f"{len(t)} trees (dark: pine, light: broadleaf, beige: shrub)")
    axs[1].hist([t[kind == k, 2] for k in range(3)], bins=40, stacked=True,
                color=["#2f5d34", "#7aa35a", "#c2b26b"], label=["pine", "broadleaf", "shrub"])
    axs[1].set_xlabel("height (m)")
    axs[1].legend()
    plt.tight_layout()
    plt.savefig(path, dpi=110)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="config.yaml")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    q = {**DEFAULTS, **(cfg.get("trees") or {})}
    d = cfg["_dir"]
    t = detect(cfg, q)
    rgb = ortho_colours(d, t[:, 0], t[:, 1], t[:, 3])
    kind = classify(t, rgb, q)
    pal = palette(rgb, kind)
    np.savetxt(d / "trees.csv", np.column_stack([t[:, :4], kind]), delimiter=",",
               fmt=["%.2f", "%.2f", "%.2f", "%.2f", "%d"], header="x,y,height_m,crown_r_m,kind", comments="")
    (d / "trees_palette.json").write_text(json.dumps(pal, indent=1), encoding="utf-8")
    prof = np.genfromtxt(d / "road_profile.csv", delimiter=",", names=True)
    plot(d / "trees.png", t, kind, prof["x"], prof["y"])
    n = np.bincount(kind, minlength=3)
    print(f"{len(t)} trees: {n[0]} pines, {n[1]} broadleaf, {n[2]} shrubs; height median "
          f"{np.median(t[:, 2]):.1f} m, max {t[:, 2].max():.1f} m; palette {pal}")
    print(f"→ {d / 'trees.csv'}\n→ {d / 'trees.png'}")


if __name__ == "__main__":
    main()
