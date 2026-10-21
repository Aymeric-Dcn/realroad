#!/usr/bin/env python3
"""
Pipeline stage 2b: measure the real carriageway edges.

    python road_edges.py      → data/<name>/road_edges.csv, road_edges.png, cross_bins.npz

For every 1 m station, ground points within ±10 m across the road are put into
12.5 cm bins (mean height and laser intensity) and the orthophoto is sampled
at the same bins. Three cues, from the most to the least trusted:

1. Edge lines. White markings show up both as LiDAR intensity peaks and as
   white pixels in the orthophoto. The carriageway edge is taken just outside
   the edge line found next to the LiDAR edge (below). Only these detections
   are trusted; detections far from the 100 m trend (driveways, lay-bys) are
   dropped and the gaps are interpolated. Where one edge line is hidden (trees,
   shadows) it is mirrored about the centre marking (both lanes equally wide).
2. LiDAR intensity step. Asphalt returns less light at 1064 nm than gravel,
   grass or soil (only ~15 % less here, so a step detector is used rather
   than a threshold), searched inside the geometric limit where the surface
   leaves the road plane (ditch, bank, wall). Used inside long unmarked
   stretches only.
3. BD TOPO width, as a guard rail: half-widths are measured from the real
   road centre and kept within BD TOPO -1/+0.75 m where nothing was observed,
   -1.5/+2.5 m where an edge line was seen (hairpin widening, paved shoulders).

Tuned on the Route Napoleon with the orthophoto as a visual reference
(road_edges.png shows the road "unrolled" with the detected edges).
"""
import argparse
import json

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter, uniform_filter1d
from scipy.spatial import cKDTree

from fetch import load_cfg, load_route
from road_profile import make_stations

DEFAULTS = {
    "half_range_m": 10.0,      # how far across the road we look, each side
    "bin_m": 0.125,
    # LiDAR intensity step
    "step_pool": 9,            # stations pooled along the road
    "step_window_m": 1.25,     # mean intensity compared on each side of a candidate edge
    "step_min_contrast": 0.04,
    "geom_tol_m": 0.15,        # surface departure that marks the geometric limit
    "line_clip": 1.35,         # damp white lines in the step detector
    # edge lines
    "line_pool": 15,
    "line_threshold": 0.6,
    "line_search_m": 1.2,      # edge line looked for within this distance of the step edge
    "line_offset_m": 0.3,      # carriageway edge = edge line + this
    # fusion
    "reject_window_m": 101.0,
    "reject_tol_m": 0.8,
    "max_gap_m": 60.0,         # longer unmarked gaps use the step edge (clamped to BD TOPO)
    "bdtopo_tol_m": 1.0,
    "median_m": 31.0,
    "sigma_m": 5.0,
    "tmin_m": 1.5,
    "tmax_m": 7.0,
    # final half-width limits around the road centre, vs the BD TOPO half-width
    "direct_tol_m": (1.5, 2.5),  # (narrower, wider) where an edge line was seen within ~10 m
    "guess_tol_m": (1.0, 0.75),  # elsewhere
    "final_sigma_m": 6.0,
    # centre marking off the road axis (centre_shift)
    "narrow_suspect_m": 4.5,     # narrower than this: check from the centre marking
    "centre_shift_max_m": 2.6,
    "centre_shift_intensity_tol": 0.25,
}


# ------------------------------------------------------------- binning
def laz_chunks_with_intensity(files, chunk=2_000_000):
    import laspy

    for f in files:
        with laspy.open(f) as r:
            for p in r.chunk_iterator(chunk):
                yield (np.asarray(p.x), np.asarray(p.y), np.asarray(p.z),
                       np.asarray(p.classification), np.asarray(p.intensity, dtype=np.float32))


def bin_cross_sections(st, chunks, z_ref, c_ref, q):
    """Accumulate per (station, bin): count, Σ residual, Σ intensity, where the
    residual is relative to the road-centre plane z_ref + c_ref·t."""
    n_st = len(st["s"])
    nb = int(round(2 * q["half_range_m"] / q["bin_m"]))
    tree = cKDTree(np.column_stack([st["x"], st["y"]]))
    cnt = np.zeros(n_st * nb, np.float64)
    sres = np.zeros(n_st * nb, np.float64)
    sint = np.zeros(n_st * nb, np.float64)
    for x, y, z, cls, inten in chunks:
        d, idx = tree.query(np.column_stack([x, y]), distance_upper_bound=q["half_range_m"] + st["step"])
        ok = np.isfinite(d)
        idx, x, y, z, ii = idx[ok], x[ok], y[ok], z[ok], inten[ok]
        dx, dy = x - st["x"][idx], y - st["y"][idx]
        along = dx * st["tx"][idx] + dy * st["ty"][idx]
        t = dx * st["nx"][idx] + dy * st["ny"][idx]
        keep = (np.abs(along) <= st["step"] / 2 + 0.25) & (np.abs(t) <= q["half_range_m"])
        idx, t, zz, ii = idx[keep], t[keep], z[keep], ii[keep]
        if not len(idx):
            continue
        b = np.clip(((t + q["half_range_m"]) / q["bin_m"]).astype(int), 0, nb - 1)
        flat = idx.astype(np.int64) * nb + b
        res = zz - (z_ref[idx] + c_ref[idx] * t)
        cnt += np.bincount(flat, minlength=n_st * nb)
        sres += np.bincount(flat, weights=res, minlength=n_st * nb)
        sint += np.bincount(flat, weights=ii, minlength=n_st * nb)
    with np.errstate(invalid="ignore", divide="ignore"):
        res = (sres / cnt).reshape(n_st, nb)
        inten = (sint / cnt).reshape(n_st, nb)
    centres = -q["half_range_m"] + q["bin_m"] * (np.arange(nb) + 0.5)
    return cnt.reshape(n_st, nb).astype(np.uint16), res.astype(np.float32), inten.astype(np.float32), centres


def ortho_cross(st, centres, ortho_dir):
    """Orthophoto RGB at every (station, bin)."""
    import re
    from pathlib import Path

    from PIL import Image

    n, nb = len(st["s"]), len(centres)
    X = st["x"][:, None] + centres[None, :] * st["nx"][:, None]
    Y = st["y"][:, None] + centres[None, :] * st["ny"][:, None]
    rgb = np.zeros((n, nb, 3), np.uint8)
    have = np.zeros((n, nb), bool)
    for jpg in sorted(Path(ortho_dir).glob("ortho_*.jpg")):
        jgw = jpg.with_suffix(".jgw")
        if not re.match(r"ortho_-?\d+_-?\d+$", jpg.stem) or not jgw.exists():
            continue
        a, _, _, e, cx, cy = [float(v) for v in jgw.read_text().split()]
        img = Image.open(jpg)
        w, h = img.size
        x0, y1 = cx - a / 2, cy - e / 2
        x1, y0 = x0 + a * w, y1 + e * h
        m = (X >= x0) & (X < x1) & (Y > y0) & (Y <= y1) & ~have
        if not m.any():
            continue
        arr = np.asarray(img.convert("RGB"))
        col = np.clip(((X[m] - x0) / a).astype(int), 0, w - 1)
        row = np.clip(((Y[m] - y1) / e).astype(int), 0, h - 1)
        rgb[m] = arr[row, col]
        have[m] = True
    return rgb, have


# ------------------------------------------------------------- detection
def _sides(centres):
    for name, sg in (("left", 1), ("right", -1)):
        idx = np.where(sg * centres > 0)[0]
        idx = idx[np.argsort(sg * centres[idx])]
        yield name, sg, idx, sg * centres[idx]


def step_edges(cnt, res, inten, centres, q):
    """LiDAR intensity step edge, within the geometric limit of the road plane."""
    w = cnt.astype(float)
    P = lambda a: uniform_filter1d(a, q["step_pool"], axis=0, mode="nearest")
    cw = P(w)
    I = np.where(np.isfinite(inten), inten, 0.0).astype(float)
    R = np.nan_to_num(res.astype(float))
    with np.errstate(invalid="ignore", divide="ignore"):
        Im = P(I * w) / cw
        Rm = P(R * w) / cw
    core = np.abs(centres) < 1.5
    ref = np.nanmedian(Im[:, core], axis=1)
    ref = np.where(np.isfinite(ref) & (ref > 0), ref, np.nanmedian(ref))
    Ic = np.minimum(np.nan_to_num(Im), q["line_clip"] * ref[:, None])
    bin_m = centres[1] - centres[0]
    k = int(round(q["step_window_m"] / bin_m))
    out = {}
    for name, sg, idx, tt in _sides(centres):
        X, Rr = Ic[:, idx], np.nan_to_num(Rm[:, idx])
        fit = (tt >= 0.75) & (tt <= 2.75)
        A = np.vstack([np.ones(fit.sum()), tt[fit]]).T
        coef = np.linalg.lstsq(A, Rr[:, fit].T, rcond=None)[0]
        dev = np.abs(Rr - (coef[0][:, None] + coef[1][:, None] * tt[None, :]))
        run = dev > q["geom_tol_m"]
        for r in range(1, 4):
            run[:, :-r] &= dev[:, r:] > q["geom_tol_m"]
        glim = np.where(run.any(1), tt[np.argmax(run, 1)], q["tmax_m"])
        cs = np.hstack([np.zeros((len(X), 1)), np.cumsum(X, 1)])
        J = np.arange(k, len(tt) - k)
        contrast = ((cs[:, J + k] - cs[:, J]) - (cs[:, J] - cs[:, J - k])) / k / ref[:, None]
        valid = (tt[J][None, :] >= q["tmin_m"] + 0.25) & (tt[J][None, :] <= np.minimum(glim, q["tmax_m"])[:, None] + 0.25)
        contrast = np.where(valid, contrast, -np.inf)
        j = np.argmax(contrast, 1)
        cmax = contrast[np.arange(len(X)), j]
        e = np.where(cmax >= q["step_min_contrast"], tt[J][j] - bin_m / 2, np.nan)
        out[name] = sg * e
    return out


def marking_score(cnt, inten, rgb, have, centres, q):
    """Evidence of a white marking per (station, bin): LiDAR intensity peak +
    orthophoto whiteness peak, each relative to its ±0.75 m surroundings."""
    w = cnt.astype(float)
    P = lambda a: uniform_filter1d(a, q["line_pool"], axis=0, mode="nearest")
    I = np.where(np.isfinite(inten), inten, 0.0).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        Ip = np.nan_to_num(P(I * w) / P(w))
    white = P(rgb.min(axis=2).astype(float) * have) / np.maximum(P(have.astype(float)), 1e-6) / 255.0
    half = int(round(0.75 / (centres[1] - centres[0])))
    peak = lambda a: a - median_filter(a, size=(1, 2 * half + 1), mode="nearest")
    core = np.abs(centres) < 1.5
    li = peak(Ip) / (np.median(Ip[:, core], axis=1)[:, None] + 1e-6)
    ph = peak(white)
    li = np.clip(li / (np.percentile(li, 99) + 1e-9), 0, None)
    ph = np.clip(ph / (np.percentile(ph, 99) + 1e-9), 0, None) if have.any() else 0 * li
    return li + ph


def line_edges(score, centres, step, q):
    """Outer side of the edge line closest to the step edge (positive t = left)."""
    out = {}
    for name, sg, idx, tt in _sides(centres):
        S = score[:, idx]
        st = np.nan_to_num(sg * step[name], nan=3.25)
        pk = (S > q["line_threshold"]) & (S >= np.roll(S, 1, 1)) & (S >= np.roll(S, -1, 1))
        pk &= ((tt >= q["tmin_m"]) & (tt <= q["tmax_m"] - 0.5))[None, :]
        d = np.abs(tt[None, :] - st[:, None])
        d = np.where(pk & (d <= q["line_search_m"]), d, np.inf)
        j = np.argmin(d, 1)
        found = np.isfinite(d[np.arange(len(S)), j])
        out[name] = (sg * (tt[j] + q["line_offset_m"]), found)
    return out


def fuse(s, line_e, found, step_e, bd_half, sign, q):
    """Trusted line edges, outliers dropped, gaps interpolated; long unmarked
    gaps use the step edge clamped around the BD TOPO half-width."""
    st = float(np.median(np.diff(s)))
    e = np.where(found, line_e, np.nan)
    ok = np.isfinite(e)
    if ok.sum() > 2:
        trend = median_filter(np.interp(s, s[ok], e[ok]), size=int(q["reject_window_m"] / st) | 1, mode="nearest")
        e = np.where(np.abs(e - trend) <= q["reject_tol_m"], e, np.nan)
        ok = np.isfinite(e)
    guard = np.clip(np.nan_to_num(sign * step_e, nan=bd_half), bd_half - q["bdtopo_tol_m"], bd_half + q["bdtopo_tol_m"])
    if ok.sum() < 2:
        filled = sign * guard
    else:
        filled = np.interp(s, s[ok], e[ok])
        idx = np.where(ok)[0]
        gap = np.zeros(len(s))
        for a, b in zip(idx[:-1], idx[1:]):
            if b - a > 1:
                gap[a + 1:b] = (b - a) * st
        gap[:idx[0]] = idx[0] * st + q["max_gap_m"] + 1
        gap[idx[-1] + 1:] = (len(s) - 1 - idx[-1]) * st + q["max_gap_m"] + 1
        filled = np.where(gap > q["max_gap_m"], sign * guard, filled)
    filled = median_filter(filled, size=int(q["median_m"] / st) | 1, mode="nearest")
    return gaussian_filter1d(filled, q["sigma_m"] / st, mode="nearest"), ok


def centre_line(score, centres, left_line, right_line, found_l, found_r, step, q):
    """Centre marking: strongest marking peak within ±2 m of the expected centre
    (mid of the edge lines when both are known, else of the step edges)."""
    mid = np.where(found_l & found_r, (left_line + right_line) / 2,
                   np.nan_to_num((step["left"] + step["right"]) / 2, nan=0.0))
    pk = (score > q["line_threshold"]) & (score >= np.roll(score, 1, 1)) & (score >= np.roll(score, -1, 1))
    cand = np.where(pk & (np.abs(centres[None, :] - mid[:, None]) <= 2.0), score, -np.inf)
    j = np.argmax(cand, 1)
    return centres[j], np.isfinite(cand[np.arange(len(score)), j])


def centre_shift(cnt, inten, score, centres, s, q):
    """Offset of the centre marking from the road axis (OSM / BD TOPO), smoothed
    along the road; 0 where there is no centre marking. The axis can run 1-2 m
    off the real centre (D45 to Dabo: on the right lane); the edges are then
    looked for from the marking, so that the centre line is not taken for an
    edge line. A peak counts as the centre marking only with asphalt on both
    sides (same LiDAR intensity at ±1.5 m): an edge line has the verge outside."""
    st = float(np.median(np.diff(s)))
    bin_m = centres[1] - centres[0]
    w = cnt.astype(float)
    P = lambda a: uniform_filter1d(a, q["line_pool"], axis=0, mode="nearest")
    I = np.where(np.isfinite(inten), inten, 0.0).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        Ip = P(I * w) / P(w)
    lim = q["centre_shift_max_m"]
    near = np.abs(centres) <= lim
    pk = (score > q["line_threshold"]) & (score >= np.roll(score, 1, 1)) & (score >= np.roll(score, -1, 1)) & near[None, :]
    k15 = int(round(1.5 / bin_m))
    j_all = np.arange(len(centres))
    jl, jr = np.clip(j_all + k15, 0, len(centres) - 1), np.clip(j_all - k15, 0, len(centres) - 1)
    il, ir = Ip[:, jl], Ip[:, jr]
    ref = np.nanmedian(np.where(near[None, :], Ip, np.nan), axis=1)[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        same = np.abs(il - ir) / np.maximum(ref, 1e-6) < q["centre_shift_intensity_tol"]
    cand = np.where(pk & same, score, -np.inf)
    j = np.argmax(cand, 1)
    found = np.isfinite(cand[np.arange(len(s)), j])
    c = centres[j] + bin_m / 2
    if found.sum() < 2:
        return np.zeros(len(s))
    win = int(51 / st) | 1
    cm = median_filter(np.interp(s, s[found], c[found]), size=win, mode="nearest")
    share = uniform_filter1d(found.astype(float), win, mode="nearest")
    c0 = np.where(share > 0.3, cm, 0.0)
    return np.clip(gaussian_filter1d(c0, 10 / st, mode="nearest"), -lim, lim)


def shift_bins(a, c0, bin_m, fill):
    """Re-centre each cross-section on t = c0 (a: stations x bins [x ...])."""
    n, nb = a.shape[:2]
    k = np.round(c0 / bin_m).astype(int)
    j = np.arange(nb)[None, :] + k[:, None]
    ok = (j >= 0) & (j < nb)
    out = a[np.arange(n)[:, None], np.clip(j, 0, nb - 1)]
    out[~ok] = fill
    return out, k * bin_m


def mirror(left, right, found_l, found_r, c, found_c, off):
    """Rebuild an edge whose line is hidden (trees, shadows) by symmetry of the
    other edge about the centre marking: both lanes have the same width."""
    L = np.where(found_l, left, np.where(found_c & found_r, 2 * c - (right + off) + off, np.nan))
    R = np.where(found_r, right, np.where(found_c & found_l, 2 * c - (left - off) - off, np.nan))
    return L, np.isfinite(L), R, np.isfinite(R)


def regularize(s, L, R, direct_l, direct_r, c, found_c, bd, q):
    """Half-widths measured from the road centre (smoothed centre marking, or the
    mid of the edges). Sides observed nearby may widen a lot (hairpins, paved
    shoulders); interpolated / guessed sides stay close to the BD TOPO half-width."""
    st = float(np.median(np.diff(s)))
    mid = (L + R) / 2
    cc = np.where(found_c, c, np.nan)
    ok = np.isfinite(cc)
    cc = np.interp(s, s[ok], cc[ok]) if ok.sum() > 1 else mid
    cc = median_filter(cc, size=int(51 / st) | 1, mode="nearest")
    cc = np.where(np.abs(cc - mid) > 1.5, mid, cc)
    cc = gaussian_filter1d(cc, 10 / st, mode="nearest")
    h = bd / 2
    out = []
    for e, direct, sg in ((L, direct_l, 1), (R, direct_r, -1)):
        lo = np.where(direct, h - q["direct_tol_m"][0], h - q["guess_tol_m"][0])
        hi = np.where(direct, h + q["direct_tol_m"][1], h + q["guess_tol_m"][1])
        hh = gaussian_filter1d(np.clip(sg * (e - cc), lo, hi), q["final_sigma_m"] / st, mode="nearest")
        out.append(cc + sg * hh)
    return out[0], out[1], cc


def measure(s, cnt, res, inten, have, score, centres, bd, q):
    """Edges and centre (offsets from the road axis) from the binned cross-sections."""
    step = step_edges(cnt, res, inten, centres, q)
    lines = line_edges(score, centres, step, q)
    (ll, fl), (rl, fr) = lines["left"], lines["right"]
    c_line, fc = centre_line(score, centres, ll, rl, fl, fr, step, q)
    ll, fl, rl, fr = mirror(ll, rl, fl, fr, c_line, fc, q["line_offset_m"])
    L, okL = fuse(s, ll, fl, step["left"], bd / 2, 1, q)
    R, okR = fuse(s, rl, fr, step["right"], bd / 2, -1, q)
    near = lambda ok_: gaussian_filter1d(ok_.astype(float), 5.0) > 0.2
    L, R, centre = regularize(s, L, R, near(okL), near(okR), c_line, fc, bd, q)
    return L, R, centre, okL, okR, fc


def edges_from_bins(s, cnt, res, inten, rgb, have, centres, bd, q):
    """Carriageway edges and centre, as offsets from the road axis."""
    score = marking_score(cnt, inten, rgb, have, centres, q)
    L, R, centre, okL, okR, fc = measure(s, cnt, res, inten, have, score, centres, bd, q)
    # Where that gives an implausibly narrow road, the axis may run along one
    # lane and the centre line be taken for an edge: measure again from the
    # centre marking there (only there: on 3-lane roads the shift picks a lane line)
    width = L - R
    narrow = width < q["narrow_suspect_m"]
    if narrow.any():
        bin_m = centres[1] - centres[0]
        c0 = centre_shift(cnt, inten, score, centres, s, q)
        if np.abs(c0[narrow]).max() >= 0.5:
            sh = [shift_bins(a, c0, bin_m, fv)[0] for a, fv in ((cnt, 0), (res, np.nan), (inten, np.nan),
                                                                    (have, False), (score, 0))]
            c0q = shift_bins(cnt, c0, bin_m, 0)[1]
            L2, R2, C2, okL2, okR2, fc2 = measure(s, *sh, centres, bd, q)
            L2, R2, C2 = L2 + c0q, R2 + c0q, C2 + c0q
            st = float(np.median(np.diff(s)))
            better = narrow & (L2 - R2 > width + 1.0)
            wgt = np.clip(gaussian_filter1d(uniform_filter1d(better.astype(float), int(31 / st) | 1), 5 / st), 0, 1)
            wgt = np.clip(wgt * 3, 0, 1)
            if better.any():
                print(f"  {better.sum() * st:.0f} m measured too narrow from the road axis (centre line taken "
                      f"for an edge): measured again from the centre marking")
                L, R, centre = L + wgt * (L2 - L), R + wgt * (R2 - R), centre + wgt * (C2 - centre)
                okL, okR = np.where(wgt > 0.5, okL2, okL), np.where(wgt > 0.5, okR2, okR)
    return L, R, centre, okL, okR, fc


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="config.yaml")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    q = {**DEFAULTS, **(cfg.get("edges") or {})}
    d = cfg["_dir"]
    prof = np.genfromtxt(d / "road_profile.csv", delimiter=",", names=True)
    files = sorted((d / "lidar").glob("*.laz"))
    if not files:
        raise SystemExit("No LiDAR files: run `python fetch.py lidar` first.")
    route = load_route(cfg)
    st = make_stations(route, float(np.median(np.diff(prof["s_m"]))))
    if len(st["s"]) != len(prof):
        raise SystemExit("road_profile.csv is out of date: run `python road_profile.py` again.")
    s = prof["s_m"]
    cf = prof["crossfall_pct"].copy()
    ok = np.isfinite(cf)
    cf = np.interp(s, s[ok], cf[ok]) / 100
    print(f"{len(s)} stations, binning {len(files)} LiDAR files (±{q['half_range_m']} m, {q['bin_m']} m bins)…")
    cnt, res, inten, centres = bin_cross_sections(st, laz_chunks_with_intensity(files), prof["z_filled_m"], cf, q)
    print("Sampling the orthophotos…")
    rgb, have = ortho_cross(st, centres, d / "ortho")
    if not have.any():
        print("  ! no orthophoto: edge lines from LiDAR intensity only (run `python fetch.py ortho`)")
    np.savez_compressed(d / "cross_bins.npz", cnt=cnt, res=res, inten=inten, centres=centres,
                        s=s, rgb=rgb, have=have)

    from build_track import DEFAULTS as TRACK_DEFAULTS, road_widths

    bd = road_widths(s, st["x"], st["y"], cfg, {**TRACK_DEFAULTS, **(cfg.get("track") or {}), "road_width_m": "auto"})
    L, R, centre, okL, okR, fc = edges_from_bins(s, cnt, res, inten, rgb, have, centres, bd, q)
    width = L - R
    stats = {
        "edge_lines_found_pct": {"left": round(100 * float(okL.mean()), 1), "right": round(100 * float(okR.mean()), 1),
                                 "centre": round(100 * float(fc.mean()), 1)},
        "width_median_m": round(float(np.median(width)), 2),
        "width_p5_p95_m": [round(float(np.percentile(width, v)), 2) for v in (5, 95)],
        "centre_offset_p95_abs_m": round(float(np.percentile(np.abs(centre), 95)), 2),
    }
    print(json.dumps(stats, indent=1))
    (d / "road_edges_stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    np.savetxt(d / "road_edges.csv", np.column_stack([s, L, R, width, centre, okL & okR]),
               delimiter=",", header="s_m,left_m,right_m,width_m,centre_offset_m,measured",
               comments="", fmt=["%.1f", "%.3f", "%.3f", "%.3f", "%.3f", "%d"])

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 15))
    ax = fig.add_subplot(5, 1, 1)
    ax.plot(s / 1000, width, lw=1, color="tab:blue", label="measured width")
    ax.plot(s / 1000, bd, lw=1, color="0.5", ls="--", label="BD TOPO")
    ax.set_ylim(3, 12)
    ax.set_ylabel("width (m)")
    ax.set_xlabel("distance (km)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"{cfg['name']} — carriageway edges (unrolled orthophoto below: green = detected edges)")
    L_ = len(s)
    for i, frac in enumerate((0.12, 0.35, 0.6, 0.85)):
        a0 = int(frac * L_)
        sl = slice(a0, min(L_, a0 + 300))
        axs = fig.add_subplot(5, 1, i + 2)
        axs.imshow(rgb[sl].transpose(1, 0, 2), aspect="auto", origin="lower",
                   extent=[s[sl][0], s[sl][-1], centres[0], centres[-1]])
        axs.plot(s[sl], L[sl], color="lime", lw=1.5)
        axs.plot(s[sl], R[sl], color="lime", lw=1.5)
        axs.plot(s[sl], centre[sl], color="red", lw=0.8)
        axs.axhline(0, color="yellow", ls="--", lw=0.7)
        axs.set_ylim(-9, 9)
        axs.set_ylabel("t (m)")
    fig.tight_layout()
    fig.savefig(d / "road_edges.png", dpi=90)
    print(f"→ {d / 'road_edges.csv'}\n→ {d / 'road_edges.png'}\n→ {d / 'cross_bins.npz'} (diagnostics)")


if __name__ == "__main__":
    main()
