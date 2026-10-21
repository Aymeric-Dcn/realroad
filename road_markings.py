#!/usr/bin/env python3
"""
Pipeline stage 2c: the real road markings (centre line and edge lines).

    python road_markings.py   → data/<name>/road_markings.csv, road_markings.png

Uses what road_edges.py already measured (cross_bins.npz: LiDAR intensity and
orthophoto colour across the road every metre, and road_edges.csv):

1. Line score across the road: a paint line is a narrow bright peak, both in
   the orthophoto (whiter than the asphalt ±0.75 m around) and in the LiDAR
   intensity (paint reflects more). The two are normalised and added, so the
   LiDAR still sees lines in tree shadows.
2. Each line (centre, left edge, right edge) is tracked from its expected
   position (road_edges.csv), allowing ±0.6 m, then read every metre:
   paint / no paint.
3. Each line is split into stretches: none, dashed, continuous (share of
   painted metres over 31 m, stretches shorter than 25 m merged).
4. Painted metres are rebuilt from that: continuous stretches fully painted;
   dashed stretches keep the dashes where they were seen (3 m long, missing
   dashes added at the local spacing, e.g. 3 m / 10 m for a French T1 centre
   line, 3 m / 3.5 m for T'3 edge lines).

build_track.py turns road_markings.csv into thin white meshes on the road
(2 cm above the asphalt, visual only) and drops the generic markings painted
in the asphalt texture.
"""
import argparse
import json

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

from fetch import load_cfg

DEFAULTS = {
    "peak_window_m": 0.75,     # line = brighter than the median over ± this
    "track_tol_m": 0.6,        # line search around its expected position
    "track_pool_m": 15,        # along-road pooling to find the line position
    "paint_threshold": 0.3,    # combined score above which a metre is painted
    "state_window_m": 31,
    "none_below": 0.12,        # share of painted metres
    "continuous_above": 0.8,
    "min_stretch_m": 25,
    "dash_m": 3.0,
    "edge_line_inset_m": 0.3,  # road_edges.py: edge = edge line + 0.3 m
    # centre line not seen (trees, shadows, worn paint) on a road that has one
    # elsewhere: drawn dashed (French T1, 3 m dash / 13 m) where wide enough
    "centre_fill": True,
    "centre_fill_min_width_m": 5.2,
    "centre_fill_min_share": 0.3,   # the road must show a centre line on this share at least
}
LINES = ("right", "centre", "left")


def line_score(cb, q):
    """Combined paint score per station and lateral bin (≈1 on a clear line)."""
    c = cb["centres"]
    bin_m = c[1] - c[0]
    half = int(round(q["peak_window_m"] / bin_m))
    W = cb["rgb"].min(axis=2).astype(np.float32)
    pw = W - median_filter(W, size=(1, 2 * half + 1), mode="nearest")
    I = np.where(cb["cnt"] > 0, cb["inten"].astype(np.float64), np.nan)
    I = np.where(np.isfinite(I), I, np.nanmedian(I))
    bg = median_filter(I, size=(1, 2 * half + 1), mode="nearest")
    pi = (I - bg) / (bg + 1.0)
    core = np.abs(c) < 6
    have = cb["have"].any(axis=1)
    sw = np.percentile(pw[have][:, core], 99.5) if have.any() else 1.0
    si = np.percentile(pi[:, core], 99.5)
    score = np.clip(pi / max(si, 1e-9), 0, None)
    if have.any():
        score += np.clip(pw / max(sw, 1e-9), 0, None) * cb["have"]
    return score


def track_line(score, centres, expected, q):
    """Follow one line near its expected offset; return offset and paint flag."""
    n_st, n_b = score.shape
    bin_m = centres[1] - centres[0]
    rows = np.arange(n_st)
    pool = uniform_filter1d(score, q["track_pool_m"], axis=0, mode="nearest")
    j0 = np.round((expected - centres[0]) / bin_m - 0.5).astype(int)
    span = int(round(q["track_tol_m"] / bin_m))
    js = np.clip(j0[:, None] + np.arange(-span, span + 1)[None, :], 0, n_b - 1)
    jb = js[rows, pool[rows[:, None], js].argmax(1)]
    jb = np.round(median_filter(jb.astype(float), 9, mode="nearest")).astype(int)
    js2 = np.clip(jb[:, None] + np.arange(-1, 2)[None, :], 0, n_b - 1)
    v = score[rows[:, None], js2].max(1)
    return centres[jb], v > q["paint_threshold"]


def runs(mask):
    """(start, end) index pairs of True runs, end exclusive."""
    m = np.r_[False, mask, False].astype(np.int8)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def stretches(on, step, q):
    """0 none / 1 dashed / 2 continuous for every station."""
    f = uniform_filter1d(on.astype(float), max(3, int(q["state_window_m"] / step)), mode="nearest")
    st = np.where(f < q["none_below"], 0, np.where(f > q["continuous_above"], 2, 1))
    st = median_filter(st, size=max(3, int(q["state_window_m"] / step)) | 1, mode="nearest")
    # merge short stretches into the previous one
    min_n = int(q["min_stretch_m"] / step)
    changed = True
    while changed:
        changed = False
        bounds = np.r_[0, np.flatnonzero(np.diff(st)) + 1, len(st)]
        for a, b in zip(bounds[:-1], bounds[1:]):
            if b - a < min_n and len(bounds) > 2:
                st[a:b] = st[a - 1] if a > 0 else st[b]
                changed = True
                break
    return st


def rebuild(on, state, s, q):
    """Painted metres from the stretches and the dashes actually seen."""
    step = float(np.median(np.diff(s)))
    paint = state == 2
    half = q["dash_m"] / 2
    for a, b in runs(state == 1):
        seen = [(s[a + i] + s[a + j - 1]) / 2 for i, j in runs(on[a:b])]
        if not seen:                       # filled stretch: nothing seen, standard spacing
            seen = [s[a] + q["dash_m"]]
        seen = np.array(seen)
        gaps = np.diff(seen)
        spacing = float(np.clip(np.median(gaps), 5.0, 16.0)) if len(gaps) else 13.0
        dashes = [seen[0]]
        for g, x in zip(gaps, seen[1:]):
            if g > 1.6 * spacing:  # dashes hidden by shadows, cars, worn paint
                k = int(round(g / spacing)) - 1
                dashes += list(dashes[-1] + g * np.arange(1, k + 1) / (k + 1))
            if g > 0.6 * spacing:
                dashes.append(x)
        # extend to the stretch ends at the same spacing
        while dashes[0] - spacing > s[a] + half:
            dashes.insert(0, dashes[0] - spacing)
        while dashes[-1] + spacing < s[b - 1] - half:
            dashes.append(dashes[-1] + spacing)
        hl = min(half, 0.45 * spacing)
        for x in dashes:
            i0 = int(np.searchsorted(s, x - hl - step / 2))
            i1 = int(np.searchsorted(s, x + hl - step / 2))
            paint[max(a, i0):min(b, max(i1, i0 + 1))] = True
    return paint


def plot(path, s, cb, track, painted, state, segments=(1000, 3050, 5500, 10500, 14500), L=150):
    from PIL import Image

    c = cb["centres"]
    bin_m = c[1] - c[0]
    cols = []
    colours = {"right": (255, 40, 40), "centre": (40, 230, 40), "left": (40, 140, 255)}
    for s0 in segments:
        if s0 + L > len(s):
            continue
        big = np.repeat(np.repeat(cb["rgb"][s0:s0 + L], 4, 0), 2, 1).astype(np.uint8)
        for name, col in colours.items():
            pos = track[name][s0:s0 + L]
            for i in range(L):
                j = int(round((pos[i] - c[0]) / bin_m - 0.5)) * 2
                if painted[name][s0 + i]:
                    big[i * 4:(i + 1) * 4, min(j + 5, big.shape[1] - 4):min(j + 9, big.shape[1])] = col
        lo, hi = int((-6 - c[0]) / bin_m) * 2, int((6 - c[0]) / bin_m) * 2
        cols += [big[:, lo:hi], np.full((L * 4, 4, 3), 255, np.uint8)]
    # stretch overview: one row per line, 10 m per pixel
    n = len(s) // 10
    ov = np.zeros((3 * 14, n, 3), np.uint8)
    pal = {0: (40, 40, 40), 1: (240, 200, 40), 2: (250, 250, 250)}
    for r, name in enumerate(LINES):
        st = state[name][::10][:n]
        for k, v in pal.items():
            ov[r * 14:r * 14 + 12][:, st == k] = v
    top = np.concatenate(cols, 1) if cols else np.zeros((10, 10, 3), np.uint8)
    w = max(top.shape[1], 600)
    ov_img = np.asarray(Image.fromarray(ov).resize((w, ov.shape[0] * 2), Image.NEAREST))
    top = np.pad(top, ((0, 0), (0, w - top.shape[1]), (0, 0)), constant_values=255)
    Image.fromarray(np.concatenate([ov_img, np.full((8, w, 3), 255, np.uint8), top], 0)).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="config.yaml")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    q = {**DEFAULTS, **(cfg.get("markings") or {})}
    d = cfg["_dir"]
    if not (d / "cross_bins.npz").exists() or not (d / "road_edges.csv").exists():
        raise SystemExit("No cross_bins.npz / road_edges.csv: run `python road_edges.py` first.")
    cb = np.load(d / "cross_bins.npz")
    e = np.genfromtxt(d / "road_edges.csv", delimiter=",", names=True)
    s = e["s_m"]
    step = float(np.median(np.diff(s)))
    score = line_score(cb, q)
    expected = {"centre": e["centre_offset_m"],
                "left": e["left_m"] - q["edge_line_inset_m"],
                "right": e["right_m"] + q["edge_line_inset_m"]}
    track, seen, state, painted = {}, {}, {}, {}
    for name in LINES:
        track[name], seen[name] = track_line(score, cb["centres"], expected[name], q)
        state[name] = stretches(seen[name], step, q)
        if name == "centre" and q["centre_fill"] and np.mean(state[name] > 0) >= q["centre_fill_min_share"]:
            gap = (state[name] == 0) & (e["width_m"] >= q["centre_fill_min_width_m"])
            if gap.any():
                st = state[name].copy()
                for a_, b_ in runs(gap):
                    before = st[a_ - 1] if a_ > 0 else 0
                    after = st[b_] if b_ < len(st) else 0
                    # continuous on both sides (bend, crest): continuous, otherwise dashed
                    st[a_:b_] = 2 if (before == 2 and after == 2) else 1
                state[name] = st
                print(f"  centre line not seen on {gap.sum() * step:.0f} m of a wide enough road: drawn dashed")
        painted[name] = rebuild(seen[name], state[name], s, q)
    out = d / "road_markings.csv"
    cols = [s] + [painted[n].astype(int) for n in LINES] + [state[n] for n in LINES]
    np.savetxt(out, np.column_stack(cols), delimiter=",", fmt=["%.1f"] + ["%d"] * 6,
               header="s_m,right,centre,left,right_state,centre_state,left_state", comments="")
    plot(d / "road_markings.png", s, cb, track, painted, state)
    stats = {}
    for n in LINES:
        share = {k: round(100 * float(np.mean(state[n] == v)), 1) for k, v in (("none", 0), ("dashed", 1), ("continuous", 2))}
        stats[n] = {**share, "painted_pct": round(100 * float(painted[n].mean()), 1)}
        print(f"  {n:6s}: continuous {share['continuous']:4.1f} %, dashed {share['dashed']:4.1f} %, "
              f"none {share['none']:4.1f} %  (painted {stats[n]['painted_pct']:.0f} % of the metres)")
    (d / "road_markings_stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(f"→ {out}\n→ {d / 'road_markings.png'}")


if __name__ == "__main__":
    main()
