#!/usr/bin/env python3
"""
Pipeline stage 2: the real road surface profile from the LiDAR point cloud.

    python road_profile.py        → data/<name>/road_profile.csv + road_profile.png

The road is cut into 1 m stations. For each station the ground (class 2) and
bridge-deck (class 17) points lying across the road are projected onto the
station's cross-section, and a robust line z = z0 + c·t is fitted over the
carriageway (t = signed lateral offset, left positive):
  - z0  → elevation of the road centre (bridge decks preferred when present)
  - c   → crossfall / camber (%), positive = road rises to the left
  - rms → fit residual: rough surface, or the axis is not on the road
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from fetch import load_cfg, load_route
from preview import MntSampler, fix_bridges, smooth

GROUND, BRIDGE = 2, 17


# ------------------------------------------------------------------ stations
def make_stations(route, step=1.0, tangent_window_m=10.0):
    """Points every `step` metres along the centreline, with unit tangents and
    left normals. Tangents come from a smoothed copy of the line so that the
    angular OSM vertices don't tilt the cross-sections."""
    s = np.arange(0.0, route.length, step)
    pts = [route.interpolate(v) for v in s]
    x = np.array([p.x for p in pts])
    y = np.array([p.y for p in pts])
    n = max(1, int(tangent_window_m / step))
    xs, ys = smooth(x, n), smooth(y, n)
    tx, ty = np.gradient(xs), np.gradient(ys)
    norm = np.hypot(tx, ty)
    norm[norm == 0] = 1.0
    tx, ty = tx / norm, ty / norm
    return {"s": s, "x": x, "y": y, "tx": tx, "ty": ty, "nx": -ty, "ny": tx, "step": step}


# ---------------------------------------------------------- point collection
def project_chunk(st, tree, x, y, z, cls, max_offset):
    """Assign each point to its nearest station and express it as (along, t).
    Keeps points within `max_offset` across and within half a step along, which
    also rejects points belonging to the other leg of a hairpin."""
    d, idx = tree.query(np.column_stack([x, y]), distance_upper_bound=max_offset + st["step"])
    ok = np.isfinite(d)
    idx, x, y, z, cls = idx[ok], x[ok], y[ok], z[ok], cls[ok]
    dx, dy = x - st["x"][idx], y - st["y"][idx]
    along = dx * st["tx"][idx] + dy * st["ty"][idx]
    t = dx * st["nx"][idx] + dy * st["ny"][idx]
    keep = (np.abs(along) <= st["step"] / 2 + 0.25) & (np.abs(t) <= max_offset)
    return (idx[keep].astype(np.int32), t[keep].astype(np.float32),
            z[keep].astype(np.float32), cls[keep].astype(np.uint8))


def collect(st, chunks, max_offset):
    """chunks: iterable of (x, y, z, classification) arrays."""
    tree = cKDTree(np.column_stack([st["x"], st["y"]]))
    parts = [project_chunk(st, tree, *c, max_offset) for c in chunks]
    parts = [p for p in parts if len(p[0])]
    if not parts:
        raise SystemExit("No LiDAR point near the road: run `python fetch.py lidar` first.")
    return tuple(np.concatenate(a) for a in zip(*parts))


def laz_chunks(files, chunk=2_000_000):
    import laspy

    for f in files:
        with laspy.open(f) as r:
            for p in r.chunk_iterator(chunk):
                yield (np.asarray(p.x), np.asarray(p.y), np.asarray(p.z),
                       np.asarray(p.classification))


# ------------------------------------------------------------------- fitting
def robust_line(t, z, iters=3, clip=0.15):
    """Least-squares z = z0 + c·t, re-fitted after dropping residuals > clip
    (kerbs, parked cars misclassified as ground, noise)."""
    keep = np.ones(len(t), bool)
    for _ in range(iters):
        if keep.sum() < 4:
            break
        A = np.column_stack([np.ones(keep.sum()), t[keep]])
        (z0, c), *_ = np.linalg.lstsq(A, z[keep], rcond=None)
        res = z - (z0 + c * t)
        new = np.abs(res) <= max(clip, 2.5 * np.std(res[keep]))
        if (new == keep).all():
            break
        keep = new
    rms = float(np.sqrt(np.mean(res[keep] ** 2))) if keep.sum() else np.nan
    return z0, c, rms, int(keep.sum())


def fit_stations(n_st, idx, t, z, cls, half_width=2.5, min_pts=8):
    out = {k: np.full(n_st, np.nan) for k in ("z", "crossfall", "rms")}
    out["n"] = np.zeros(n_st, int)
    out["bridge"] = np.zeros(n_st, bool)
    sel = np.abs(t) <= half_width
    idx, t, z, cls = idx[sel], t[sel], z[sel], cls[sel]
    order = np.argsort(idx, kind="stable")
    idx, t, z, cls = idx[order], t[order], z[order], cls[order]
    starts = np.searchsorted(idx, np.arange(n_st + 1))
    for k in range(n_st):
        a, b = starts[k], starts[k + 1]
        if b - a < min_pts:
            continue
        tk, zk, ck = t[a:b], z[a:b], cls[a:b]
        deck = ck == BRIDGE
        if deck.sum() >= min_pts:  # on a bridge: the deck is the road
            tk, zk = tk[deck], zk[deck]
            out["bridge"][k] = True
        z0, c, rms, n = robust_line(tk.astype(float), zk.astype(float))
        if n >= min_pts:
            out["z"][k], out["crossfall"][k], out["rms"][k], out["n"][k] = z0, 100 * c, rms, n
    return out


def fill_gaps(s, v):
    ok = np.isfinite(v)
    if ok.sum() < 2:
        return v
    return np.interp(s, s[ok], v[ok])


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="config.yaml")
    ap.add_argument("--step", type=float, default=1.0, help="station spacing (m)")
    ap.add_argument("--half-width", type=float, default=2.5,
                    help="half-width of the strip used for the fit (m)")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    route = load_route(cfg)
    files = sorted((cfg["_dir"] / "lidar").glob("*.laz"))
    if not files:
        raise SystemExit("No LiDAR files: run `python fetch.py lidar` first.")

    st = make_stations(route, a.step)
    n_st = len(st["s"])
    print(f"{n_st} stations, reading {len(files)} LiDAR files…")
    idx, t, z, cls = collect(st, laz_chunks(files), max_offset=cfg["corridor_m"])
    print(f"  {len(idx):,} points within ±{cfg['corridor_m']} m of the axis")
    fit = fit_stations(n_st, idx, t, z, cls, half_width=a.half_width)

    s = st["s"]
    covered = np.isfinite(fit["z"]).mean()
    # The axis is probably off the carriageway where the "road" is very steep
    # sideways or rough: embankment, ditch or wall instead of tarmac.
    suspect = (np.abs(fit["crossfall"]) > 12) | (fit["rms"] > 0.12)
    suspect_smooth = smooth(suspect.astype(float), int(15 / a.step)) > 0.5
    # Elevations there are not the road's: interpolate across those stretches
    zf = fill_gaps(s, np.where(suspect_smooth, np.nan, fit["z"]))
    grade = np.gradient(smooth(zf, int(10 / a.step)), a.step) * 100
    cross = fill_gaps(s, np.where(suspect_smooth, np.nan, fit["crossfall"]))

    # DTM along the same stations, for comparison
    z_dtm = None
    mnt = sorted((cfg["_dir"] / "mnt").glob("*.tif"))
    if mnt:
        z_dtm = MntSampler(mnt).sample(st["x"], st["y"])

    def runs(mask):
        r, i = [], 0
        while i < len(mask):
            if mask[i]:
                j = i
                while j + 1 < len(mask) and mask[j + 1]:
                    j += 1
                r.append((s[i], s[j]))
                i = j + 1
            else:
                i += 1
        return r

    bridges = [(a0, b0) for a0, b0 in runs(fit["bridge"]) if b0 - a0 >= 3]
    suspects = [(a0, b0) for a0, b0 in runs(suspect_smooth) if b0 - a0 >= 10]
    for a0, b0 in bridges:
        print(f"  · bridge deck (LiDAR class 17) km {a0 / 1000:.3f}–{b0 / 1000:.3f} ({b0 - a0 + a.step:.0f} m)")
    for a0, b0 in suspects:
        print(f"  ! axis probably off the road km {a0 / 1000:.3f}–{b0 / 1000:.3f} ({b0 - a0 + a.step:.0f} m)")

    stats = {
        "stations": n_st,
        "coverage_pct": round(100 * covered, 1),
        "points_used_median": int(np.median(fit["n"][fit["n"] > 0])) if (fit["n"] > 0).any() else 0,
        "elev_min_m": round(float(np.nanmin(zf)), 2),
        "elev_max_m": round(float(np.nanmax(zf)), 2),
        "max_uphill_grade_pct": round(float(grade.max()), 1),
        "max_downhill_grade_pct": round(float(grade.min()), 1),
        "crossfall_p5_p95_pct": [round(float(np.nanpercentile(cross, q)), 1) for q in (5, 95)],
        "surface_rms_median_cm": round(100 * float(np.nanmedian(fit["rms"])), 1),
        "bridges": len(bridges),
        "axis_suspect_m": int(sum(b0 - a0 + a.step for a0, b0 in suspects)),
    }
    if z_dtm is not None:
        ok = np.isfinite(z_dtm) & np.isfinite(fit["z"]) & ~fit["bridge"]
        diff = fit["z"][ok] - z_dtm[ok]
        stats["lidar_minus_dtm_median_cm"] = round(100 * float(np.median(diff)), 1)
        stats["lidar_minus_dtm_p95_abs_cm"] = round(100 * float(np.percentile(np.abs(diff), 95)), 1)
    print(json.dumps(stats, indent=1))
    out_dir = cfg["_dir"]
    (out_dir / "road_profile_stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")

    header = "s_m,x,y,z_m,z_filled_m,grade_pct,crossfall_pct,rms_m,n_pts,bridge,axis_suspect"
    rows = np.column_stack([s, st["x"], st["y"], fit["z"], zf, grade, fit["crossfall"],
                            fit["rms"], fit["n"], fit["bridge"], suspect_smooth])
    np.savetxt(out_dir / "road_profile.csv", rows, delimiter=",", header=header, comments="",
               fmt=["%.1f", "%.2f", "%.2f", "%.3f", "%.3f", "%.2f", "%.2f", "%.3f", "%d", "%d", "%d"])

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    km = s / 1000
    fig, ax = plt.subplots(4, 1, figsize=(14, 11), sharex=True,
                           gridspec_kw={"height_ratios": [3, 2, 2, 1.2]})
    if z_dtm is not None:
        z_dtm_fixed, _ = fix_bridges(s, fill_gaps(s, z_dtm), a.step)
        ax[0].plot(km, z_dtm_fixed, lw=0.8, color="0.6", label="DTM (bridges interpolated)")
    ax[0].plot(km, zf, lw=1.2, color="tab:blue", label="LiDAR road surface")
    for a0, b0 in bridges:
        ax[0].axvspan(a0 / 1000, b0 / 1000, color="tab:orange", alpha=0.4)
    for a0, b0 in suspects:
        for axx in ax:
            axx.axvspan(a0 / 1000, b0 / 1000, color="tab:red", alpha=0.12)
    ax[0].set_ylabel("elevation (m)")
    ax[0].set_title(f"{cfg['name']} — LiDAR road profile (orange: bridges, red: axis suspect)")
    ax[0].legend(loc="lower right", fontsize=8)
    ax[1].fill_between(km, grade, 0, where=grade >= 0, color="tab:red", alpha=0.6)
    ax[1].fill_between(km, grade, 0, where=grade < 0, color="tab:blue", alpha=0.6)
    ax[1].set_ylabel("grade (%)")
    ax[2].plot(km, fit["crossfall"], lw=0.5, color="0.6")
    ax[2].plot(km, smooth(cross, int(10 / a.step)), lw=1.2, color="tab:green")
    ax[2].axhline(0, color="k", lw=0.5)
    ax[2].set_ylim(-15, 15)
    ax[2].set_ylabel("crossfall (%)\n+ = rises to the left")
    ax[3].plot(km, 100 * fit["rms"], lw=0.6, color="tab:purple")
    ax[3].set_ylim(0, 20)
    ax[3].set_ylabel("fit rms (cm)")
    ax[3].set_xlabel("distance (km)")
    for axx in ax:
        axx.grid(alpha=0.3)
    fig.tight_layout()
    out = out_dir / "road_profile.png"
    fig.savefig(out, dpi=130)
    print(f"→ {out}\n→ {out_dir / 'road_profile.csv'}")


if __name__ == "__main__":
    main()
