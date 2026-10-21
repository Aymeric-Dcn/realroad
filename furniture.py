"""
Road furniture for build_track.py: what stands along the road.

    guard rails   LiDAR above-ground points (fetch.py roadside → roadside_points.npz)
                  + OpenStreetMap barrier=guard_rail
    walls         same sources (barrier=wall / retaining_wall, LiDAR)
    edge posts    French J6 delineators, white with a black band, every 15-50 m
                  outside villages, where there is no guard rail. Not surveyed:
                  laid out like the road rules say (closer in bends)
    signs         speed limits where the OSM maxspeed changes, village entry /
                  exit (EB10 / EB20 with the village name), STOP and give way
                  at the side roads (OSM)
    power         wooden poles and lattice pylons (OSM power=pole / tower)

Everything is returned as scenery meshes (see scenery.py) in Lambert-93 +
metres; guard rails and walls are solid (cars collide with them).
"""
import json
import math

import numpy as np

MAX_VERTS = 60_000


# ------------------------------------------------------------- projection
def lambert93(lon, lat):
    """WGS84 → Lambert-93 (EPSG:2154), closed formulas (no pyproj needed)."""
    a, f = 6378137.0, 1 / 298.257222101
    e = math.sqrt(2 * f - f * f)
    lon0, lat0, lat1, lat2 = map(math.radians, (3.0, 46.5, 44.0, 49.0))
    x0, y0 = 700000.0, 6600000.0

    def m(p):
        return np.cos(p) / np.sqrt(1 - e * e * np.sin(p) ** 2)

    def t(p):
        return np.tan(np.pi / 4 - p / 2) / ((1 - e * np.sin(p)) / (1 + e * np.sin(p))) ** (e / 2)

    n = (math.log(m(lat1)) - math.log(m(lat2))) / (math.log(t(lat1)) - math.log(t(lat2)))
    F = m(lat1) / (n * t(lat1) ** n)
    rho0 = a * F * t(lat0) ** n
    lam, phi = np.radians(np.asarray(lon, float)), np.radians(np.asarray(lat, float))
    rho = a * F * t(phi) ** n
    th = n * (lam - lon0)
    return x0 + rho * np.sin(th), y0 + rho0 - rho * np.cos(th)


# --------------------------------------------------------------- the road
class Road:
    """Stations of the road with their frame; ground height beside it."""

    def __init__(self, s, x, y, frame, dtm, verge_m):
        self.s, self.x, self.y = s, x, y
        self.f = frame
        self.dtm = dtm
        self.verge_m = verge_m
        self.step = float(np.median(np.diff(s)))
        from scipy.spatial import cKDTree

        self.tree = cKDTree(np.column_stack([x, y]))
        # curvature radius (signed: > 0 turning left), smoothed over ~30 m
        h = np.unwrap(np.arctan2(frame["ty"], frame["tx"]))
        k = np.gradient(h) / self.step
        n = max(1, int(15 / self.step))
        k = np.convolve(np.pad(k, n, mode="edge"), np.ones(2 * n + 1) / (2 * n + 1), mode="valid")
        self.radius = 1.0 / np.where(np.abs(k) < 1e-5, 1e-5, k)

    def k(self, s):
        return np.clip(np.searchsorted(self.s, s), 0, len(self.s) - 1)

    def edge(self, k, side):
        """Offset of the carriageway edge (side +1 left, -1 right), positive left."""
        return self.f["left"][k] if side > 0 else self.f["right"][k]

    def xy(self, k, t):
        return self.x[k] + t * self.f["nx"][k], self.y[k] + t * self.f["ny"][k]

    def road_z(self, k, t):
        tc = np.clip(t, self.f["right"][k], self.f["left"][k])
        return self.f["z"][k] + self.f["c"][k] * tc

    def ground(self, k, t):
        """Height of the verge at offset t (positive left): the road edge
        blending into the terrain over verge_m, like build_verges."""
        k = np.asarray(k)
        t = np.asarray(t, float)
        px, py = self.xy(k, t)
        zt = self.dtm.sample(np.atleast_1d(px), np.atleast_1d(py)).reshape(np.shape(px))
        ze = self.road_z(k, t)
        e = np.where(t > 0, self.f["left"][k], self.f["right"][k])
        out = np.abs(t) - np.abs(e)
        w = np.clip(out / self.verge_m, 0, 1)
        w = w * w * (3 - 2 * w)
        zg = np.where(np.isfinite(zt), ze - 0.03 + (zt - ze + 0.03) * w, ze - 0.03)
        return np.where(out <= 0, ze, zg)

    def heading(self, k):
        return np.arctan2(self.f["ty"][k], self.f["tx"][k])


# ------------------------------------------------------- mesh accumulator
class Acc:
    """Meshes grouped by (material, options); split under the vertex limit."""

    def __init__(self):
        self.g = {}

    def add(self, key, material, V, F, UV, opts=None):
        opts = opts or {}
        k = (key, material, json.dumps(opts, sort_keys=True))
        parts = self.g.setdefault(k, [[], [], [], 0])
        if parts[3] + len(V) > MAX_VERTS:
            self.flush_one(k)
            parts = self.g.setdefault(k, [[], [], [], 0])
        parts[0].append(np.asarray(V, float))
        parts[1].append(np.asarray(F, np.int64) + parts[3])
        parts[2].append(np.asarray(UV, float))
        parts[3] += len(V)

    def flush_one(self, k):
        parts = self.g.pop(k)
        self.done = getattr(self, "done", [])
        if parts[0]:
            n = sum(1 for d in self.done if d[0].startswith(k[0] + "_"))
            self.done.append((f"{k[0]}_{n}", k[1], np.concatenate(parts[0]), np.concatenate(parts[1]).astype(np.int32),
                              np.concatenate(parts[2]), True, json.loads(k[2])))

    def meshes(self):
        for k in list(self.g):
            self.flush_one(k)
        return getattr(self, "done", [])


def _box(c, yaw, w, d, h, z0=0.0):
    """Box standing on c (x, y, z): width w along yaw, depth d, height h.
    8 corners + 5 faces (no bottom), outward, uv in metres."""
    ca, sa = math.cos(yaw), math.sin(yaw)
    u = np.array([ca, sa, 0.0]) * w / 2
    v = np.array([-sa, ca, 0.0]) * d / 2
    up = np.array([0.0, 0.0, h])
    b = np.asarray(c, float) + [0, 0, z0]
    V, F, UV = [], [], []
    sides = [(-u - v, u - v, w), (u - v, u + v, d), (u + v, -u + v, w), (-u + v, -u - v, d)]
    for p0, p1, L in sides:
        n = len(V)
        V += [b + p0, b + p1, b + p1 + up, b + p0 + up]
        UV += [[0, 0], [L, 0], [L, h], [0, h]]
        F += [[n, n + 1, n + 2], [n, n + 2, n + 3]]
    n = len(V)
    V += [b - u - v + up, b + u - v + up, b + u + v + up, b - u + v + up]
    UV += [[0, 0], [w, 0], [w, d], [0, d]]
    F += [[n, n + 1, n + 2], [n, n + 2, n + 3]]
    return np.array(V), np.array(F), np.array(UV)


def _double(F):
    return np.concatenate([F, F[:, ::-1]])


# ---------------------------------------------------------- OpenStreetMap
def load_osm(path):
    if not path.exists():
        return {"nodes": {}, "ways": [], "node_ways": []}
    osm = json.loads(path.read_text(encoding="utf-8"))
    nodes, ways, node_ways = {}, [], []
    for e in osm.get("elements", []):
        if e["type"] == "node":
            x, y = lambert93(e["lon"], e["lat"])
            nodes[e["id"]] = (float(x), float(y), e.get("tags") or {})
    for e in osm.get("elements", []):
        if e["type"] == "way":
            ids = [n for n in e.get("nodes", []) if n in nodes]
            if len(ids) >= 2:
                ways.append((np.array([nodes[n][:2] for n in ids]), e.get("tags") or {}))
                node_ways.append((ids, e.get("tags") or {}))
    return {"nodes": nodes, "ways": ways, "node_ways": node_ways}


def _speed(v):
    try:
        return int(str(v).split()[0])
    except (ValueError, IndexError):
        return {"FR:urban": 50, "FR:rural": 80}.get(str(v))


def maxspeed_profile(road, osm, ref_regex):
    """Speed limit at every station, from the OSM ways of the road (0 = unknown)."""
    import re

    rx = re.compile(ref_regex) if ref_regex else None
    best = np.full(len(road.s), np.inf)
    vmax = np.zeros(len(road.s), int)
    for pts, tags in osm["ways"]:
        if "highway" not in tags or "maxspeed" not in tags:
            continue
        if rx and not rx.match(tags.get("ref", "")):
            continue
        v = _speed(tags["maxspeed"])
        if not v:
            continue
        # densify the way and mark the stations it runs along
        seg = []
        for a, b in zip(pts[:-1], pts[1:]):
            n = max(2, int(np.hypot(*(b - a)) / 2))
            seg.append(a + (b - a) * np.linspace(0, 1, n)[:, None])
        seg = np.concatenate(seg)
        d, k = road.tree.query(seg, distance_upper_bound=12)
        ok = np.isfinite(d)
        for kk, dd in zip(k[ok], d[ok]):
            if dd < best[kk]:
                best[kk], vmax[kk] = dd, v
    # fill short gaps (junctions, < 60 m) with the neighbours' value; longer
    # stretches without a tagged limit stay unknown (0 = no village, no sign):
    # a 50 zone at one end must not spread over the whole road
    known = np.flatnonzero(vmax > 0)
    if len(known):
        idx = np.arange(len(vmax))
        nxt = known[np.clip(np.searchsorted(known, idx), 0, len(known) - 1)]
        prv = known[np.clip(np.searchsorted(known, idx, side="right") - 1, 0, len(known) - 1)]
        gap = np.abs(road.s[nxt] - road.s[prv])
        fill = (vmax == 0) & (gap < 60) & (vmax[prv] == vmax[nxt]) & (prv <= idx) & (nxt >= idx)
        vmax = np.where(fill, vmax[prv], vmax)
    return vmax


# -------------------------------------------------------- LiDAR detection
def detect_barriers(road, pts_npz, osm, p):
    """Guard rails and walls along each side, as runs of stations with an
    offset and a top height. LiDAR: non-vegetation points 0.35-2.5 m above
    the ground, 0.2-5 m outside the carriageway edge, binned per metre of
    road; a bin with enough points at a steady offset is a barrier, rails
    when the top is at rail height (0.6-0.95 m) and thin, walls otherwise.
    OSM ways add what the LiDAR misses."""
    n = len(road.s)
    runs = []
    hit = {}
    if pts_npz is not None and pts_npz.exists():
        z = np.load(pts_npz)
        xyz, cls = z["xyz"], z["cls"]
        sel = np.isin(cls, p["furniture_lidar_classes"])
        xyz = xyz[sel]
        if len(xyz):
            d, k = road.tree.query(xyz[:, :2], distance_upper_bound=25)
            ok = np.isfinite(d)
            xyz, k = xyz[ok], k[ok]
            dx, dy = xyz[:, 0] - road.x[k], xyz[:, 1] - road.y[k]
            t = dx * road.f["nx"][k] + dy * road.f["ny"][k]
            side = np.where(t > 0, 1, -1)
            e = np.where(side > 0, road.f["left"][k], road.f["right"][k])
            out = np.abs(t) - np.abs(e)
            zg = road.dtm.sample(xyz[:, 0], xyz[:, 1])
            hgt = xyz[:, 2] - zg
            ok = (out > 0.15) & (out < 5.0) & (hgt > 0.35) & (hgt < 2.6) & np.isfinite(hgt)
            for sd in (-1, 1):
                m = ok & (side == sd)
                hit[sd] = (k[m], out[m], hgt[m])
    for sd in (-1, 1):
        if sd not in hit:
            continue
        kk, oo, hh = hit[sd]
        bins = (road.s[kk] / 1.0).astype(int)        # 1 m bins along the road
        nb = int(road.s[-1]) + 1
        cnt = np.bincount(bins, minlength=nb)
        med_o = np.full(nb, np.nan)
        top = np.full(nb, np.nan)
        spread = np.full(nb, np.nan)
        order = np.argsort(bins)
        b_sorted = bins[order]
        starts = np.searchsorted(b_sorted, np.arange(nb))
        ends = np.searchsorted(b_sorted, np.arange(nb), side="right")
        for b in np.flatnonzero(cnt >= p["furniture_min_points"]):
            ii = order[starts[b]:ends[b]]
            o = oo[ii]
            mo = np.median(o)
            close = np.abs(o - mo) < 0.5
            if close.sum() < p["furniture_min_points"]:
                continue
            med_o[b] = np.median(o[close])
            top[b] = np.percentile(hh[ii][close], 85)
            spread[b] = np.percentile(o[close], 90) - np.percentile(o[close], 10)
        good = np.isfinite(med_o)
        # runs: gaps up to 4 m bridged, at least 10 m long
        idx = np.flatnonzero(good)
        if not len(idx):
            continue
        groups = np.split(idx, np.flatnonzero(np.diff(idx) > 4) + 1)
        for g in groups:
            if g[-1] - g[0] < p["furniture_min_run_m"]:
                continue
            o_run = np.nanmedian(med_o[g])
            if np.nanstd(med_o[g]) > 0.8:     # scattered: parked cars, rocks…
                continue
            t_run = float(np.nanmedian(top[g]))
            kind = "rail" if 0.5 <= t_run <= 1.0 and np.nanmedian(spread[g]) < 0.45 else "wall"
            s0, s1 = float(g[0]), float(g[-1] + 1)
            runs.append({"side": sd, "s0": s0, "s1": s1, "kind": kind, "src": "lidar",
                         "off": float(np.clip(o_run, 0.25, 4.5)), "height": float(np.clip(t_run, 0.6, 2.5)),
                         "off_s": np.arange(g[0], g[-1] + 1), "off_v": np.interp(np.arange(g[0], g[-1] + 1), g, np.nan_to_num(med_o[g], nan=o_run))})
    # OpenStreetMap barriers along the road
    for pts, tags in osm["ways"]:
        b = tags.get("barrier")
        if b not in ("guard_rail", "wall", "retaining_wall", "jersey_barrier"):
            continue
        dense = []
        for a_, b_ in zip(pts[:-1], pts[1:]):
            m = max(2, int(np.hypot(*(b_ - a_))))
            dense.append(a_ + (b_ - a_) * np.linspace(0, 1, m)[:, None])
        dense = np.concatenate(dense)
        d, k = road.tree.query(dense, distance_upper_bound=12)
        ok = np.isfinite(d)
        if ok.sum() < 5:
            continue
        k = k[ok]
        t = (dense[ok, 0] - road.x[k]) * road.f["nx"][k] + (dense[ok, 1] - road.y[k]) * road.f["ny"][k]
        sd = 1 if np.median(t) > 0 else -1
        e = np.where(sd > 0, road.f["left"][k], road.f["right"][k])
        out = np.abs(t) - np.abs(e)
        if np.median(out) > 6 or np.median(out) < -0.5:
            continue
        s0, s1 = float(road.s[k].min()), float(road.s[k].max())
        if s1 - s0 < 5:
            continue
        # skip if the LiDAR already found it
        if any(r_["side"] == sd and r_["s0"] < s1 and r_["s1"] > s0 for r_ in runs):
            continue
        kind = "rail" if b in ("guard_rail",) else "wall"
        h = 0.75 if kind == "rail" else float(_speed(tags.get("height", "")) or 1.0)
        runs.append({"side": sd, "s0": s0, "s1": s1, "kind": kind, "src": "osm",
                     "off": float(np.clip(np.median(out), 0.3, 4.5)), "height": h})
    return runs


def detect_parapets(road, built_up, taken, p):
    """Low parapet walls on the valley side. On the Route Napoléon the
    street photos show them wherever the ground drops away beside the road
    (stone or concrete, ~0.65 m). The LiDAR HD barely sees them (a handful of
    points, not in the DTM), so they are laid out by that rule: terrain
    more than `parapet_drop_m` below the road edge within 2.5 m, over at
    least 20 m, outside villages. Not surveyed one by one."""
    n = len(road.s)
    step = 2
    ks = np.arange(0, n, step)
    offs = np.arange(0.25, 4.01, 0.25)
    runs = []
    for sd in (-1, 1):
        e = np.where(sd > 0, road.f["left"][ks], road.f["right"][ks])
        T = e[:, None] + sd * offs[None, :]
        px = road.x[ks][:, None] + T * road.f["nx"][ks][:, None]
        py = road.y[ks][:, None] + T * road.f["ny"][ks][:, None]
        zt = road.dtm.sample(px.ravel(), py.ravel()).reshape(px.shape)
        ze = road.road_z(ks, e)
        drop = ze[:, None] - zt
        near = drop[:, offs <= 2.5]
        steep = np.nanmax(np.nan_to_num(near, nan=0), axis=1) >= p["parapet_drop_m"]
        steep &= np.nanmax(np.nan_to_num(drop, nan=0), axis=1) >= p["parapet_drop_m"] + 0.5
        steep &= ~built_up[ks]
        steep &= ~taken[sd][np.clip(road.s[ks].astype(int), 0, len(taken[sd]) - 1)]
        # where the drop starts: the wall stands just before it
        first = np.argmax(np.nan_to_num(drop, nan=0) > 0.3, axis=1)
        off = np.clip(offs[first] - 0.25, 0.3, 2.0)
        idx = np.flatnonzero(steep)
        if not len(idx):
            continue
        groups = np.split(idx, np.flatnonzero(np.diff(idx) > 4) + 1)   # gaps up to ~8 m bridged
        for g in groups:
            s0, s1 = road.s[ks[g[0]]], road.s[ks[g[-1]]]
            if s1 - s0 < p["parapet_min_run_m"]:
                continue
            ss = road.s[ks[g[0]:g[-1] + 1]]
            oo = np.convolve(np.pad(off[g[0]:g[-1] + 1], 5, mode="edge"), np.ones(11) / 11, mode="valid")
            runs.append({"side": sd, "s0": float(s0), "s1": float(s1), "kind": "wall", "src": "rule",
                         "off": float(np.median(oo)), "height": p["parapet_height_m"],
                         "off_s": ss, "off_v": oo, "thickness": 0.35})
    return runs


# ---------------------------------------------------------------- builders
RAIL_LOW, RAIL_HIGH = 0.45, 0.76      # W-beam rail band above the ground (m)


def build_rail(acc, road, run):
    """Galvanised W-beam guard rail: a band 0.31 m high facing the road (both
    sides drawn), posts every 2 m, ends sloping into the ground over 4 m."""
    sd = run["side"]
    s_ = np.arange(run["s0"], run["s1"] + 0.01, 1.0)
    if len(s_) < 3:
        return 0.0
    k = road.k(s_)
    off = np.interp(s_, run["off_s"], run["off_v"]) if "off_s" in run else np.full(len(s_), run["off"])
    off = np.convolve(np.pad(off, 3, mode="edge"), np.ones(7) / 7, mode="valid")
    t = road.edge(k, sd) + sd * off
    px, py = road.xy(k, t)
    zg = road.ground(k, t)
    # sloped ends
    L = s_ - s_[0]
    R = s_[-1] - s_
    ramp = np.clip(np.minimum(L, R) / 4.0, 0, 1)
    lo = zg + RAIL_LOW * ramp - 0.05 * (1 - ramp)
    hi = zg + RAIL_HIGH * ramp + 0.12 * (1 - ramp)
    n = len(s_)
    V = np.concatenate([np.column_stack([px, py, lo]), np.column_stack([px, py, hi])])
    u = L / 4.0
    UV = np.concatenate([np.column_stack([u, np.zeros(n)]), np.column_stack([u, np.ones(n)])])
    F = []
    for i in range(n - 1):
        F += [[i, i + 1, n + i + 1], [i, n + i + 1, n + i]]
    F = _double(np.array(F))
    acc.add("guardrail", "guardrail", V, F, UV, {"surface": "WALL"})
    # posts (C profile ~ 0.1 m), behind the rail
    for i in range(0, n, 2):
        if ramp[i] <= 0:
            continue
        yaw = road.heading(k[i])
        back = 0.08 * sd
        c = (px[i] + back * road.f["nx"][k[i]], py[i] + back * road.f["ny"][k[i]], zg[i] - 0.3)
        bv, bf, buv = _box(c, yaw, 0.1, 0.06, RAIL_HIGH + 0.3 - 0.02)
        acc.add("guardrail_posts", "post_dark", bv, bf, buv * 0.2, {"lod_out": 300.0})
    return s_[-1] - s_[0]


def build_wall(acc, road, run):
    """Dry-stone / concrete wall (muret) of the measured height, 0.45 m thick."""
    sd = run["side"]
    s_ = np.arange(run["s0"], run["s1"] + 0.01, 1.0)
    if len(s_) < 3:
        return 0.0
    k = road.k(s_)
    off = np.interp(s_, run["off_s"], run["off_v"]) if "off_s" in run else np.full(len(s_), run["off"])
    off = np.convolve(np.pad(off, 3, mode="edge"), np.ones(7) / 7, mode="valid")
    th = run.get("thickness", 0.45)
    t_in = road.edge(k, sd) + sd * off
    t_out = t_in + sd * th
    zg = road.ground(k, t_in)
    z0 = zg - 0.4
    z1 = zg + run["height"]
    n = len(s_)
    ax, ay = road.xy(k, t_in)
    bx, by = road.xy(k, t_out)
    L = s_ - s_[0]
    V = np.concatenate([np.column_stack([ax, ay, z0]), np.column_stack([ax, ay, z1]),
                        np.column_stack([bx, by, z1]), np.column_stack([bx, by, z0])])
    UV = np.concatenate([np.column_stack([L, z0 - zg]), np.column_stack([L, z1 - zg]),
                         np.column_stack([L, z1 - zg + th]), np.column_stack([L, z0 - zg + 2 * th])]) / 1.0
    F = []
    for a_, b_ in ((0, 1), (1, 2), (2, 3)):
        for i in range(n - 1):
            F += [[a_ * n + i, a_ * n + i + 1, b_ * n + i + 1], [a_ * n + i, b_ * n + i + 1, b_ * n + i]]
    F = np.array(F)
    # counter-clockwise seen from outside the wall: depends on the side
    F = F[:, ::-1] if sd < 0 else F
    acc.add("walls", "stonewall", V, F, UV, {"surface": "WALL"})
    return s_[-1] - s_[0]


def build_posts(acc, road, blocked, built_up, p):
    """J6 delineators: every 50 m in straight lines, closer in bends, both
    sides at the same place, not where there is a guard rail or a wall,
    not in villages. White post, black band with the reflector."""
    s_end = road.s[-1]
    s = p["start_offset_m"] if "start_offset_m" in p else 10.0
    n = 0
    while s < s_end - 5:
        k = int(road.k(s))
        R = abs(road.radius[k])
        step = float(np.clip(0.25 * R, 16.0, 50.0))
        if not built_up[k]:
            for sd in (-1, 1):
                if blocked[sd][min(int(s), len(blocked[sd]) - 1)]:
                    continue
                t = road.edge(k, sd) + sd * 0.8
                px, py = road.xy(k, t)
                zg = float(road.ground(k, t))
                yaw = road.heading(k)
                # trapezoid post ~0.12 m, 1.0 m high; the reflector faces the traffic
                V, F, UV = _box((px, py, zg - 0.2), yaw, 0.12, 0.1, 1.2)
                UV = np.clip(UV[:, [0, 1]] / [0.12, 1.2], 0, 1)
                # columns: left half of the texture = right side (rectangle reflector),
                # right half = left side (two dots)
                UV[:, 0] = UV[:, 0] * 0.5 + (0.5 if sd > 0 else 0.0)
                acc.add("edge_posts", "j6", V, F, UV, {"lod_out": 220.0, "shadows": False})
                n += 1
        s += step
    return n


def build_power(acc, osm, road, p):
    """Wooden poles and lattice pylons (OSM)."""
    n_pole = n_tower = 0
    for nid, (x, y, tags) in osm["nodes"].items():
        pw = tags.get("power") or ("pole" if tags.get("man_made") == "utility_pole" else None)
        if pw not in ("pole", "tower"):
            continue
        d, k = road.tree.query([x, y])
        if d > (p["furniture_power_max_m"] if pw == "pole" else 2500):
            continue
        if d < np.abs(road.edge(k, 1)) + 1.5:
            continue
        zg = road.dtm.sample(np.array([x]), np.array([y]))[0]
        if not np.isfinite(zg):
            zg = road.f["z"][k]
        if pw == "pole":
            h = float(_speed(tags.get("height", "")) or 8.5)
            V, F, UV = _box((x, y, zg - 0.5), 0.0, 0.24, 0.24, h + 0.5)
            acc.add("power_poles", "wood", V, F, UV * [1, 0.25], {"lod_out": 600.0})
            n_pole += 1
        else:
            h = float(_speed(tags.get("height", "")) or 32.0)
            yaw = road.heading(k) + np.pi / 2       # unknown: across the road
            V, F, UV = _pylon(x, y, zg, h, yaw)
            acc.add("pylons", "pylon", V, _double(F), UV, {"lod_out": 3000.0, "shadows": False})
            n_tower += 1
    return n_pole, n_tower


def _pylon(x, y, zg, h, yaw):
    """Lattice tower as alpha-tested textured faces: tapered shaft and two
    cross-arms. UV v in metres / 4 (the lattice texture repeats every 4 m)."""
    ca, sa = math.cos(yaw), math.sin(yaw)
    U = np.array([ca, sa, 0.0])
    W = np.array([-sa, ca, 0.0])
    c = np.array([x, y, zg])
    V, F, UV = [], [], []

    def frustum(z0, z1, w0, w1):
        for (a, b) in (((-1, -1), (1, -1)), ((1, -1), (1, 1)), ((1, 1), (-1, 1)), ((-1, 1), (-1, -1))):
            n = len(V)
            p0 = c + (a[0] * U + a[1] * W) * w0 / 2 + [0, 0, z0]
            p1 = c + (b[0] * U + b[1] * W) * w0 / 2 + [0, 0, z0]
            p2 = c + (b[0] * U + b[1] * W) * w1 / 2 + [0, 0, z1]
            p3 = c + (a[0] * U + a[1] * W) * w1 / 2 + [0, 0, z1]
            V.extend([p0, p1, p2, p3])
            UV.extend([[0, z0 / 4], [w0 / 4, z0 / 4], [w1 / 4, z1 / 4], [0, z1 / 4]])
            F.extend([[n, n + 1, n + 2], [n, n + 2, n + 3]])

    frustum(-0.5, 0.62 * h, 0.2 * h, 0.07 * h)
    frustum(0.62 * h, h, 0.07 * h, 0.035 * h)
    for za, span in ((0.66 * h, 0.55 * h), (0.84 * h, 0.42 * h)):   # cross-arms
        n = len(V)
        hw = 0.035 * h
        for sgn in (-1, 1):
            n = len(V)
            p0 = c + U * sgn * hw + [0, 0, za]
            p1 = c + U * sgn * span / 2 + [0, 0, za + 0.4]
            p2 = c + U * sgn * span / 2 + [0, 0, za + 1.0]
            p3 = c + U * sgn * hw + [0, 0, za + 2.2]
            V.extend([p0, p1, p2, p3])
            UV.extend([[0, 0], [span / 8, 0.1], [span / 8, 0.25], [0, 0.55]])
            F.extend([[n, n + 1, n + 2], [n, n + 2, n + 3]])
    return np.array(V), np.array(F), np.array(UV)


# ------------------------------------------------------------------ signs
def _sign_plan(road, vmax, osm, p):
    """[(s, side, facing_dir(+1 travels with s, -1 against), kind, text)]"""
    plan = []
    n = len(road.s)
    if vmax.any():
        # villages: stretches limited to 50 or less; the village sign itself
        # sets the limit, so no separate speed sign at its ends
        bounds = []
        urban = vmax <= 50
        places = [(x, y, t["name"]) for (x, y, t) in osm["nodes"].values()
                  if t.get("place") in ("village", "town", "hamlet") and t.get("name")]
        if urban.any():
            idx = np.flatnonzero(urban)
            groups = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
            for g in groups:
                if road.s[g[-1]] - road.s[g[0]] < 80:
                    continue
                bounds += [road.s[g[0]], road.s[g[-1]]]
                mid = g[len(g) // 2]
                name = ""
                if places:
                    dd = [math.hypot(px - road.x[mid], py - road.y[mid]) for px, py, _ in places]
                    j = int(np.argmin(dd))
                    if dd[j] < 1500:
                        name = places[j][2]
                for kk, sd, fd, kind in ((g[0], -1, +1, "town"), (g[-1], -1, +1, "town_end"),
                                         (g[-1], +1, -1, "town"), (g[0], +1, -1, "town_end")):
                    if 0 < kk < n - 1:
                        plan.append((road.s[kk], sd, fd, kind, name))
        bounds = np.array(bounds)
        # speed changes, both directions
        for i in np.flatnonzero(np.diff(vmax) != 0):
            a, b = vmax[i], vmax[i + 1]
            if not (a and b):
                continue
            if len(bounds) and np.abs(bounds - road.s[i]).min() < 40 and min(a, b) <= 50:
                continue
            plan.append((road.s[i + 1], -1, +1, f"speed{b}", ""))      # driving with s: right side
            plan.append((road.s[i], +1, -1, f"speed{a}", ""))          # driving against s: left side
    for nid, (x, y, tags) in osm["nodes"].items():
        hw = tags.get("highway")
        if hw not in ("stop", "give_way"):
            continue
        d, k = road.tree.query([x, y])
        if d > 25:
            continue
        app = _approach(osm, nid, tags, road)
        if app is None:
            continue
        plan.append(("node", app[0], app[1], app[2], "stop" if hw == "stop" else "giveway"))
    return plan


def _approach(osm, nid, tags, road):
    """Where a STOP / give-way sign stands and which way it faces: on the
    right of the road it controls, 2 m before the node, facing the drivers
    who come along that road (OSM direction=forward/backward along the way).
    Returns (x, y, face_angle) or None when the road is unknown."""
    x, y = osm["nodes"][nid][:2]
    best = None
    for ids, wt in osm.get("node_ways", []):
        if nid not in ids or "highway" not in wt:
            continue
        i = ids.index(nid)
        d = tags.get("direction", "")
        if d == "forward" and i > 0:
            prev = ids[i - 1]
        elif d == "backward" and i < len(ids) - 1:
            prev = ids[i + 1]
        elif d not in ("forward", "backward"):
            # no direction: traffic comes from the end farther from the main road
            cand = [ids[j] for j in (i - 1, i + 1) if 0 <= j < len(ids)]
            if not cand:
                continue
            prev = max(cand, key=lambda n: road.tree.query(osm["nodes"][n][:2])[0])
        else:
            continue
        px, py = osm["nodes"][prev][:2]
        L = math.hypot(x - px, y - py)
        if L < 0.5:
            continue
        ax, ay = (x - px) / L, (y - py) / L          # direction of travel towards the node
        main = wt.get("ref", "") and road.tree.query([x, y])[0] < 6
        score = (0 if main else 1, L)
        if best is None or score > best[0]:
            back = min(2.0, 0.8 * L)
            sx, sy = x - ax * back + ay * 3.2, y - ay * back - ax * 3.2   # right of the road
            best = (score, (sx, sy, math.atan2(-ay, -ax)))
    return best[1] if best else None


SIGN_KINDS = ["speed30", "speed50", "speed70", "speed80", "speed90", "stop", "giveway"]


def build_signs(acc, road, plan, town_slots):
    """Sign plates (double-sided quad, back grey) on a galvanised pole."""
    from textures import sign_uv

    n = 0
    for item in plan:
        if item[0] == "node":
            _, px, py, face, kind = item          # face: the plate normal looks at the drivers
            zg = road.dtm.sample(np.array([px]), np.array([py]))[0]
            if not np.isfinite(zg):
                continue
            text = ""
        else:
            s, sd, fd, kind, text = item
            k = int(road.k(s))
            t = road.edge(k, sd) + sd * 1.2
            px, py = road.xy(k, t)
            zg = float(road.ground(k, t))
            face = road.heading(k) + (np.pi if fd > 0 else 0.0)   # facing the oncoming drivers
        if kind.startswith("speed") and kind not in SIGN_KINDS:
            continue
        town = kind in ("town", "town_end")
        if town:
            key = (kind, text)
            if key not in town_slots:
                continue
            uv_rect = town_slots[key]
            w, h = 1.3, 0.65
        else:
            uv_rect = sign_uv(kind)
            w = h = 0.75 if kind != "giveway" else 0.9
        bottom = zg + (1.5 if town else 1.6)
        # pole
        V, F, UV = _box((px, py, zg - 0.3), face, 0.07, 0.07, bottom - zg + h + 0.3 - 0.05)
        acc.add("sign_poles", "post_metal", V, F, UV * 0.3, {"lod_out": 350.0})
        # plate: normal = (cos face, sin face)
        nx_, ny_ = math.cos(face), math.sin(face)
        ux, uy = -ny_, nx_                       # plate width direction (left → right seen from the front)
        cx, cy = px + nx_ * 0.05, py + ny_ * 0.05
        V = np.array([[cx - ux * w / 2, cy - uy * w / 2, bottom], [cx + ux * w / 2, cy + uy * w / 2, bottom],
                      [cx + ux * w / 2, cy + uy * w / 2, bottom + h], [cx - ux * w / 2, cy - uy * w / 2, bottom + h]])
        u0, v0, u1, v1 = uv_rect
        UV = np.array([[u0, v0], [u1, v0], [u1, v1], [u0, v1]])
        Fq = np.array([[0, 1, 2], [0, 2, 3]])
        if np.dot(np.cross(V[1] - V[0], V[2] - V[0])[:2], [nx_, ny_]) < 0:
            Fq = Fq[:, ::-1]
        acc.add("signs", "signs", V, Fq, UV, {"lod_out": 350.0})
        # back of the plate (grey, same outline)
        bu0, bv0, bu1, bv1 = sign_uv("back_rect" if town else "back_tri" if kind == "giveway" else "back_disc")
        Vb = V - [nx_ * 0.01, ny_ * 0.01, 0]
        acc.add("signs", "signs", Vb, Fq[:, ::-1], np.array([[bu0, bv0], [bu1, bv0], [bu1, bv1], [bu0, bv1]]),
                {"lod_out": 350.0})
        n += 1
    return n


# ------------------------------------------------------------------- main
def build_furniture(d, road, p, ref_regex, tex_dir):
    """All road furniture: returns (meshes, materials, stats)."""
    from textures import furniture_textures

    osm = load_osm(d / "osm_roadside.json")
    pts = d / "roadside_points.npz"
    acc = Acc()
    stats = {}
    runs = detect_barriers(road, pts if (pts.exists() and p["furniture_lidar"]) else None, osm, p)
    nb = int(road.s[-1]) + 2
    blocked = {-1: np.zeros(nb, bool), 1: np.zeros(nb, bool)}
    for r_ in runs:
        blocked[r_["side"]][int(max(0, r_["s0"] - 5)):int(r_["s1"] + 5)] = True
    vmax = maxspeed_profile(road, osm, ref_regex)
    built_up = (vmax > 0) & (vmax <= 50)
    if p["furniture_parapets"]:
        runs += detect_parapets(road, built_up, blocked, p)
    L = {"rail": 0.0, "wall": 0.0}
    for r_ in runs:
        if r_["kind"] == "rail":
            L["rail"] += build_rail(acc, road, r_)
        else:
            L["wall"] += build_wall(acc, road, r_)
        blocked[r_["side"]][int(max(0, r_["s0"] - 10)):int(r_["s1"] + 10)] = True
    stats["guard_rails_m"], stats["walls_m"] = round(L["rail"]), round(L["wall"])
    stats["edge_posts"] = build_posts(acc, road, blocked, built_up, p) if p["furniture_posts"] else 0
    plan = _sign_plan(road, vmax, osm, p)
    towns = sorted({(it[3], it[4]) for it in plan if it[0] != "node" and it[3] in ("town", "town_end")})
    slots = furniture_textures(tex_dir, towns)
    stats["signs"] = build_signs(acc, road, plan, slots)
    stats["power_poles"], stats["pylons"] = build_power(acc, osm, road, p)
    materials = {
        "guardrail": {"texture": "guardrail.png", "kind": "metal"},
        "post_dark": {"texture": "post_dark.png", "kind": "metal"},
        "post_metal": {"texture": "post_metal.png", "kind": "metal"},
        "stonewall": {"texture": "stonewall.png", "kind": "facade"},
        "j6": {"texture": "j6.png", "kind": "sign"},
        "signs": {"texture": "signs.png", "shader": "ksPerPixelAT", "alpha": True, "kind": "sign"},
        "wood": {"texture": "wood.png", "kind": "facade"},
        "pylon": {"texture": "pylon.png", "shader": "ksPerPixelAT", "alpha": True, "kind": "metal"},
    }
    meshes = acc.meshes()
    used = {m_[1] for m_ in meshes}
    return meshes, {k: v for k, v in materials.items() if k in used}, stats, runs
