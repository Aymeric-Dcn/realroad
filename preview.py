#!/usr/bin/env python3
"""
Section preview: elevation profile, grade and curves, from the DTM.

    python preview.py            → data/<name>/preview.png + stats in the terminal
"""
import argparse
import json
from pathlib import Path

import numpy as np

from fetch import load_cfg, load_route


class MntSampler:
    """Read elevation at Lambert-93 (x, y) from a set of GeoTIFF tiles."""

    def __init__(self, files):
        import rasterio

        self.ds = [rasterio.open(f) for f in files]

    def sample(self, xs, ys):
        z = np.full(len(xs), np.nan)
        for d in self.ds:
            b = d.bounds
            m = np.isnan(z) & (xs >= b.left) & (xs < b.right) & (ys > b.bottom) & (ys <= b.top)
            if not m.any():
                continue
            vals = np.array([v[0] for v in d.sample(zip(xs[m], ys[m]))], dtype=float)
            if d.nodata is not None:
                vals[vals == d.nodata] = np.nan
            vals[vals < -1000] = np.nan
            z[m] = vals
        return z


def resample(route, step):
    d = np.arange(0, route.length, step)
    pts = [route.interpolate(s) for s in d]
    return d, np.array([p.x for p in pts]), np.array([p.y for p in pts])


def smooth(a, n):
    if n <= 1:
        return a
    k = np.ones(n) / n
    pad = np.pad(a, n // 2, mode="edge")
    return np.convolve(pad, k, mode="valid")[: len(a)]


def rolling_median(a, n):
    n = max(3, n | 1)  # odd window
    pad = np.pad(a, n // 2, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(pad, n), axis=1)


def fix_bridges(d, z, step, window_m=30.0, tol_m=1.5):
    """The DTM is bare earth: under a bridge it dives into the ravine. Detect
    narrow dips (below the local median) and replace them by interpolating
    between the two abutments. Returns the corrected z + the list of zones."""
    med = rolling_median(z, int(window_m / step))
    bad = (med - z) > tol_m
    zones, i = [], 0
    while i < len(bad):
        if bad[i]:
            j = i
            while j + 1 < len(bad) and bad[j + 1]:
                j += 1
            zones.append((d[i], d[j], float((med - z)[i:j + 1].max())))
            i = j + 1
        else:
            i += 1
    zf = z.copy()
    zf[bad] = np.interp(d[bad], d[~bad], z[~bad])
    return zf, zones


def heading_curvature(xs, ys, step, window_m=24.0):
    n = max(1, int(window_m / step))
    xs, ys = smooth(xs, n), smooth(ys, n)  # smooth the line (OSM vertices are angular)
    h = np.unwrap(np.arctan2(np.gradient(ys), np.gradient(xs)))
    return smooth(np.gradient(h) / step, n)  # 1/m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="config.yaml")
    ap.add_argument("--step", type=float, default=2.0, help="pas d'échantillonnage (m)")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    route = load_route(cfg)

    files = sorted((cfg["_dir"] / "mnt").glob("*.tif"))
    if not files:
        raise SystemExit("Pas de MNT : lance `python fetch.py mnt`.")
    d, xs, ys = resample(route, a.step)
    z = MntSampler(files).sample(xs, ys)
    missing = np.isnan(z).mean()
    if missing > 0:
        print(f"! {missing:.1%} du tracé sans altitude (dalle manquante ?), interpolé.")
        ok = ~np.isnan(z)
        z = np.interp(d, d[ok], z[ok])

    z_raw = z
    z, bridges = fix_bridges(d, z, a.step)
    for s0, s1, depth in bridges:
        print(f"  · pont / ouvrage probable km {s0 / 1000:.2f} "
              f"({s1 - s0 + a.step:.0f} m, creux de {depth:.1f} m dans le MNT) → corrigé")
    zs = smooth(z, int(20 / a.step))  # ~20 m smoothing for the grade
    slope = np.gradient(zs, a.step) * 100
    k = np.abs(heading_curvature(xs, ys, a.step))
    radius = np.full_like(k, np.inf)
    np.divide(1.0, k, out=radius, where=k > 1e-4)

    stats = {
        "longueur_km": round(route.length / 1000, 2),
        "alt_min_m": round(float(z.min()), 1),
        "alt_max_m": round(float(z.max()), 1),
        "denivele_positif_m": round(float(np.clip(np.diff(zs), 0, None).sum()), 0),
        "pente_max_montee_pct": round(float(slope.max()), 1),
        "pente_max_descente_pct": round(float(slope.min()), 1),
        "virages_rayon_<_30m": int(np.sum(np.diff((radius < 30).astype(int)) == 1)),
        "virages_rayon_<_60m": int(np.sum(np.diff((radius < 60).astype(int)) == 1)),
        "ponts_corriges": len(bridges),
    }
    print(json.dumps(stats, indent=1, ensure_ascii=False))
    (cfg["_dir"] / "stats.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False), encoding="utf-8")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14, 9))
    ax_map = fig.add_subplot(1, 3, 1)
    sc = ax_map.scatter(xs, ys, c=z, s=2, cmap="terrain")
    ax_map.plot(xs[0], ys[0], "go", label="départ")
    ax_map.plot(xs[-1], ys[-1], "ro", label="arrivée")
    ax_map.set_aspect("equal")
    ax_map.set_title("Tracé (couleur = altitude)")
    ax_map.legend(loc="best")
    ax_map.tick_params(labelsize=7)
    fig.colorbar(sc, ax=ax_map, shrink=0.6, label="m")

    km = d / 1000
    ax1 = fig.add_subplot(3, 3, (2, 3))
    ax1.plot(km, z_raw, lw=0.8, color="0.7", label="MNT brut")
    ax1.plot(km, z, lw=1.2, color="tab:blue", label="corrigé (ponts)")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.set_ylabel("altitude (m)")
    ax1.set_title(f"{cfg['name']} — {stats['longueur_km']} km")
    ax2 = fig.add_subplot(3, 3, (5, 6), sharex=ax1)
    ax2.fill_between(km, slope, 0, where=slope >= 0, color="tab:red", alpha=0.6)
    ax2.fill_between(km, slope, 0, where=slope < 0, color="tab:blue", alpha=0.6)
    ax2.set_ylabel("pente (%)")
    ax3 = fig.add_subplot(3, 3, (8, 9), sharex=ax1)
    ax3.plot(km, np.clip(radius, 0, 300), lw=0.8, color="tab:purple")
    ax3.set_ylabel("rayon virage (m)")
    ax3.set_ylim(0, 300)
    ax3.set_xlabel("distance (km)")
    for ax in (ax1, ax2, ax3):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    out = cfg["_dir"] / "preview.png"
    fig.savefig(out, dpi=130)
    print(f"→ {out}")


if __name__ == "__main__":
    main()
