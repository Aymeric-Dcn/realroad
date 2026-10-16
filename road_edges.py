#!/usr/bin/env python3
"""
Pipeline stage 2b: measure the real carriageway edges from the LiDAR.

    python road_edges.py      → data/<name>/road_edges.csv, road_edges.png, cross_bins.npz

For every 1 m station, ground points within ±10 m across the road are put into
25 cm bins. Each bin gets its mean height and mean laser intensity. Walking
outwards from the centre on each side, the tarmac ends where most bins stop
looking like road:
  - geometry: the surface leaves the line fitted on that side of the road
    (ditch, bank, kerb), or tilts too much
  - intensity: asphalt is dark at 1064 nm; grass, gravel and soil return more
    light (white lines do too, but they are only one bin wide, so a run of
    several "non-road" bins is required)
Both edges give the width and the real road centre (offset from the OSM axis).
cross_bins.npz keeps the binned data so the detector can be tuned offline.
"""
import argparse
import json

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter

from fetch import load_cfg, load_route
from scipy.spatial import cKDTree

from road_profile import make_stations

DEFAULTS = {
    "half_range_m": 10.0,     # how far across the road we look, each side
    "bin_m": 0.125,
    "inner_fit_m": (0.75, 2.75),  # bins used to fit each side's road surface line
    "geom_tol_m": 0.08,       # max height deviation from that line for "road"
    "slope_max": 0.12,        # max local cross slope between bins for "road"
    "int_ratio_max": 1.6,     # max intensity / asphalt reference for "road"
    "int_ratio_min": 0.45,
    "window_m": 1.0,          # edge = start of a 1 m window…
    "window_bad_frac": 0.7,   # … that is mostly non-road…
    "run_m": 0.375,           # … and opens a solid non-road run (longer than a white line + blur)
    "pool_stations": 5,       # bins are pooled over this many stations (≈ 5 m) before deciding
    "min_half_m": 2.0,        # sanity limits on each half-width
    "max_half_m": 7.5,
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


# ------------------------------------------------------------- detection
def detect_edges(cnt, res, inten, centres, q):
    """Return left (t > 0) and right (t < 0) edge offsets per station (NaN when
    not found) and the per-bin road mask, for diagnostics."""
    n_st, nb = res.shape
    # pool neighbouring stations: ~2.5 points per bin and per metre is too few
    k = max(1, int(q["pool_stations"]))
    if k > 1:
        c = cnt.astype(np.float64)
        sr = np.nan_to_num(res.astype(np.float64)) * c
        si = np.nan_to_num(inten.astype(np.float64)) * c
        box = lambda a_: np.apply_along_axis(lambda v: np.convolve(v, np.ones(k), mode="same"), 0, a_)
        cnt_p = box(c)
        with np.errstate(invalid="ignore", divide="ignore"):
            res = box(sr) / cnt_p
            inten = box(si) / cnt_p
        cnt = cnt_p
    known = cnt > 0
    # asphalt intensity reference: centre bins, smoothed along the road
    core = np.abs(centres) < 1.5
    ref = np.nanmedian(np.where(known[:, core], inten[:, core], np.nan), axis=1)
    ref = np.where(np.isfinite(ref), ref, np.nanmedian(ref))
    ref = median_filter(ref, size=31, mode="nearest")
    ratio = inten / ref[:, None]
    int_ok = (ratio <= q["int_ratio_max"]) & (ratio >= q["int_ratio_min"])

    left = np.full(n_st, np.nan)
    right = np.full(n_st, np.nan)
    road = np.zeros_like(known)
    a, b = q["inner_fit_m"]
    for side, sign in (("left", 1), ("right", -1)):
        order = np.where(sign * centres > 0)[0]
        order = order[np.argsort(sign * centres[order])]  # outward
        tt = sign * centres[order]                         # positive distances
        R = res[:, order]
        K = known[:, order]
        fit = (tt >= a) & (tt <= b)
        # per-station line through the inner bins of this side (handles crown)
        w = K[:, fit].astype(float)
        X = tt[fit][None, :]
        Y = np.nan_to_num(R[:, fit])
        sw = w.sum(1)
        mx = (w * X).sum(1) / np.maximum(sw, 1)
        my = (w * Y).sum(1) / np.maximum(sw, 1)
        slope = (w * (X - mx[:, None]) * (Y - my[:, None])).sum(1) / np.maximum((w * (X - mx[:, None]) ** 2).sum(1), 1e-6)
        line = my[:, None] + slope[:, None] * (tt[None, :] - mx[:, None])
        dev = np.abs(R - line)
        local_slope = np.abs(np.diff(np.nan_to_num(R), axis=1, prepend=0)) / q["bin_m"]
        geom_ok = (dev <= q["geom_tol_m"]) & (local_slope <= q["slope_max"] + 0.04 + np.abs(slope)[:, None])
        ok = geom_ok & int_ok[:, order]
        bad = K & ~ok
        road[:, order] = K & ok
        # edge: first bin (beyond 1 m) starting a window with ≥ window_bad bad bins…
        wb = max(2, int(round(q["window_m"] / q["bin_m"])))
        nbad = int(np.ceil(q["window_bad_frac"] * wb))
        run = max(2, int(round(q["run_m"] / q["bin_m"])))
        csum = np.concatenate([np.zeros((n_st, 1)), np.cumsum(bad, axis=1)], axis=1)
        win = csum[:, wb:] - csum[:, :-wb]                # bad count in [j, j+wb)
        start = np.searchsorted(tt, 1.0)
        # …and the edge bin must open a run of `run_m` of bad bins, so a
        # single bright white line just inside the edge does not trigger it
        two = bad.copy()
        for r_ in range(1, run):
            two[:, :-r_] &= bad[:, r_:]
            two[:, -r_:] = False
        hit = (win[:, start:] >= nbad) & two[:, start:start + win.shape[1] - start]
        has = hit.any(axis=1)
        j = start + np.argmax(hit, axis=1)
        edge = tt[j] - q["bin_m"] / 2                      # inner boundary of bin j
        edge = np.where(has, edge, np.nan)
        edge = np.where((edge >= q["min_half_m"]) & (edge <= q["max_half_m"]), edge, np.nan)
        (left if side == "left" else right)[:] = sign * edge
    return left, right, road, ratio


def smooth_edges(s, e, sigma_m=5.0, med_m=15.0):
    step = float(np.median(np.diff(s)))
    ok = np.isfinite(e)
    if ok.sum() < 2:
        return e, ok
    filled = np.interp(s, s[ok], e[ok])
    filled = median_filter(filled, size=int(med_m / step) | 1, mode="nearest")
    return gaussian_filter1d(filled, sigma_m / step, mode="nearest"), ok


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
    cf = prof["crossfall_pct"].copy()
    ok = np.isfinite(cf)
    cf = np.interp(prof["s_m"], prof["s_m"][ok], cf[ok]) / 100
    print(f"{len(st['s'])} stations, binning {len(files)} LiDAR files (±{q['half_range_m']} m, {q['bin_m']} m bins)…")
    cnt, res, inten, centres = bin_cross_sections(st, laz_chunks_with_intensity(files), prof["z_filled_m"], cf, q)
    np.savez_compressed(d / "cross_bins.npz", cnt=cnt, res=res.astype(np.float16),
                        inten=inten.astype(np.float16), centres=centres, s=prof["s_m"])

    left, right, road, _ = detect_edges(cnt, res, inten, centres, q)
    s = prof["s_m"]
    L, okL = smooth_edges(s, left)
    R, okR = smooth_edges(s, right)
    width = L - R
    centre = (L + R) / 2
    found = okL & okR
    stats = {
        "edges_found_pct": round(100 * float(found.mean()), 1),
        "width_median_m": round(float(np.median(width)), 2),
        "width_p5_p95_m": [round(float(np.percentile(width, v)), 2) for v in (5, 95)],
        "centre_offset_median_m": round(float(np.median(centre)), 2),
        "centre_offset_p95_abs_m": round(float(np.percentile(np.abs(centre), 95)), 2),
    }
    print(json.dumps(stats, indent=1))
    (d / "road_edges_stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    np.savetxt(d / "road_edges.csv", np.column_stack([s, L, R, width, centre, found]),
               delimiter=",", header="s_m,left_m,right_m,width_m,centre_offset_m,measured",
               comments="", fmt=["%.1f", "%.3f", "%.3f", "%.3f", "%.3f", "%d"])

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    km = s / 1000
    fig = plt.figure(figsize=(14, 10))
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(km, width, lw=1, color="tab:blue", label="measured width")
    ax1.scatter(km[::10], (left - right)[::10], s=1, color="0.6", label="raw (every 10 m)")
    ax1.set_ylabel("width (m)")
    ax1.set_ylim(3, 12)
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_title(f"{cfg['name']} — carriageway edges from LiDAR")
    ax2 = fig.add_subplot(3, 1, 2, sharex=ax1)
    ax2.plot(km, centre, lw=1, color="tab:purple")
    ax2.axhline(0, color="k", lw=0.5)
    ax2.set_ylabel("road centre − OSM axis (m)\n+ = left")
    ax2.set_xlabel("distance (km)")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
    # a strip of cross-sections: road mask over 300 m
    ax3 = fig.add_subplot(3, 1, 3)
    k0 = int(np.argmin(np.abs(s - 2000)))
    sl = slice(k0, k0 + 300)
    ax3.imshow(np.where(cnt[sl] > 0, road[sl], np.nan).T, aspect="auto", origin="lower",
               extent=[s[sl][0], s[sl][-1], centres[0], centres[-1]], cmap="RdYlGn", vmin=0, vmax=1)
    ax3.plot(s[sl], L[sl], color="k", lw=1)
    ax3.plot(s[sl], R[sl], color="k", lw=1)
    ax3.set_ylabel("t (m)")
    ax3.set_xlabel("s (m) — green: road-like bins, red: not road, black: detected edges")
    fig.tight_layout()
    fig.savefig(d / "road_edges.png", dpi=120)
    print(f"→ {d / 'road_edges.csv'}\n→ {d / 'road_edges.png'}\n→ {d / 'cross_bins.npz'} (diagnostics)")


if __name__ == "__main__":
    main()
