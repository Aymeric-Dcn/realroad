"""
3D scenery for build_track.py: tree billboards (trees.py) and buildings
(BD TOPO, fetch.py buildings). Everything here is visual only (no physics).

Meshes are returned as (name, material, verts, faces, uvs, keep_winding) in
Lambert-93 + metres; build_track.py shifts them to its local origin.
"""
import json

import numpy as np

from textures import FACADE_BAY_M, TREE_CELL, TREE_CELLS, TREE_WIDTH_FACTOR

MAX_VERTS = 60_000
PLANES = 3          # vertical billboard planes per tree


def _tile_index(x, y, tiles):
    idx = np.full(len(x), -1)
    for i, t in enumerate(tiles):
        x0, y0, x1, y1 = t["bounds"]
        idx[(idx < 0) & (x >= x0) & (x < x1) & (y >= y0) & (y < y1)] = i
    return idx


def _tile_name(t):
    return f"{int(t['bounds'][0])}_{int(t['bounds'][1])}"


# ------------------------------------------------------------------- trees
def build_trees(trees_csv, dtm, tiles, seed=3):
    """Two crossed vertical quads per tree, both sides (normals end up
    pointing up in kn5_export, as Kunos does for tree billboards)."""
    t = np.genfromtxt(trees_csv, delimiter=",", names=True)
    if t.size == 0:
        return []
    t = np.atleast_1d(t)
    x, y, h, r, kind = t["x"], t["y"], t["height_m"], t["crown_r_m"], t["kind"].astype(int)
    zg = dtm.sample(x, y)
    ok = np.isfinite(zg) & (h > 0.5)
    x, y, h, r, kind, zg = x[ok], y[ok], h[ok], r[ok], kind[ok], zg[ok]
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, np.pi, len(x))
    pick = rng.integers(0, 2, len(x))
    tidx = _tile_index(x, y, tiles)
    out = []
    margin = 2.0 / (512 * TREE_CELLS)
    for ti in np.unique(tidx):
        sel = np.flatnonzero(tidx == ti)
        per = 4 * PLANES
        for c0 in range(0, len(sel), MAX_VERTS // per):
            s_ = sel[c0:c0 + MAX_VERTS // per]
            n = len(s_)
            V = np.zeros((n, per, 3))
            UV = np.zeros((n, per, 2))
            cell = np.array([TREE_CELL[k][p % len(TREE_CELL[k])] for k, p in zip(kind[s_], pick[s_])])
            u0, u1 = cell / TREE_CELLS + margin, (cell + 1) / TREE_CELLS - margin
            wf = np.array([TREE_WIDTH_FACTOR.get(k, 1.2) for k in kind[s_]])
            hw = r[s_] * wf
            zb, zt = zg[s_] - 0.15, zg[s_] - 0.15 + h[s_] * 1.03
            for q in range(PLANES):   # planes 180/PLANES degrees apart, like a star seen from above
                ang = theta[s_] + q * np.pi / PLANES
                ux, uy = np.cos(ang), np.sin(ang)
                for j, (sx, zz) in enumerate(((-1, zb), (1, zb), (1, zt), (-1, zt))):
                    V[:, 4 * q + j, 0] = x[s_] + sx * hw * ux
                    V[:, 4 * q + j, 1] = y[s_] + sx * hw * uy
                    V[:, 4 * q + j, 2] = zz
                UV[:, 4 * q + 0] = np.column_stack([u0, np.zeros(n)])
                UV[:, 4 * q + 1] = np.column_stack([u1, np.zeros(n)])
                UV[:, 4 * q + 2] = np.column_stack([u1, np.ones(n)])
                UV[:, 4 * q + 3] = np.column_stack([u0, np.ones(n)])
            base = (np.arange(n) * per)[:, None]
            quad = np.array([[0, 1, 2], [0, 2, 3], [0, 2, 1], [0, 3, 2]])
            F = np.concatenate([base[:, :, None] + quad[None] + 4 * q for q in range(PLANES)], axis=1).reshape(-1, 3)
            name = f"trees_{_tile_name(tiles[ti]) if ti >= 0 else 'out'}_{c0 // (MAX_VERTS // per)}"
            out.append((name, "trees", V.reshape(-1, 3), F.astype(np.int32), UV.reshape(-1, 2), True))
    return out


# --------------------------------------------------------------- buildings
def _signed_area(pts):
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _clean_ring(pts, tol=0.2):
    """Drop repeated / almost collinear vertices (simplification)."""
    pts = np.asarray(pts, float)
    if len(pts) > 1 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    changed = True
    while changed and len(pts) > 3:
        changed = False
        for i in range(len(pts)):
            a_, b_, c_ = pts[i - 1], pts[i], pts[(i + 1) % len(pts)]
            ab, ac = b_ - a_, c_ - a_
            L = np.hypot(*ac)
            dev = abs(ab[0] * ac[1] - ab[1] * ac[0]) / L if L > 1e-9 else np.hypot(*ab)
            if dev < tol or np.hypot(*ab) < 0.05:
                pts = np.delete(pts, i, axis=0)
                changed = True
                break
    return pts


def _min_rect(pts):
    """Minimum-area rectangle (4 corners) of a point set, by testing each
    convex-hull edge direction."""
    from scipy.spatial import ConvexHull

    try:
        hull = pts[ConvexHull(pts).vertices]
    except Exception:
        hull = pts
    best = None
    for i in range(len(hull)):
        e = hull[(i + 1) % len(hull)] - hull[i]
        L = np.hypot(*e)
        if L < 1e-9:
            continue
        u = e / L
        v = np.array([-u[1], u[0]])
        pu, pv = hull @ u, hull @ v
        area = np.ptp(pu) * np.ptp(pv)
        if best is None or area < best[0]:
            best = (area, u, v, pu.min(), pu.max(), pv.min(), pv.max())
    _, u, v, a0, a1, b0, b1 = best
    return np.array([a0 * u + b0 * v, a1 * u + b0 * v, a1 * u + b1 * v, a0 * u + b1 * v]), best[0]


def _ear_clip(pts):
    """Triangles (index triplets) of a simple CCW polygon."""
    idx = list(range(len(pts)))
    tris = []

    def inside(p_, a_, b_, c_):
        d1 = (p_[0] - b_[0]) * (a_[1] - b_[1]) - (a_[0] - b_[0]) * (p_[1] - b_[1])
        d2 = (p_[0] - c_[0]) * (b_[1] - c_[1]) - (b_[0] - c_[0]) * (p_[1] - c_[1])
        d3 = (p_[0] - a_[0]) * (c_[1] - a_[1]) - (c_[0] - a_[0]) * (p_[1] - a_[1])
        return not ((d1 < 0 or d2 < 0 or d3 < 0) and (d1 > 0 or d2 > 0 or d3 > 0))

    guard = 0
    while len(idx) > 3 and guard < 10000:
        guard += 1
        for k in range(len(idx)):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            a_, b_, c_ = pts[i0], pts[i1], pts[i2]
            if (b_[0] - a_[0]) * (c_[1] - a_[1]) - (b_[1] - a_[1]) * (c_[0] - a_[0]) <= 1e-12:
                continue  # reflex
            if any(inside(pts[j], a_, b_, c_) for j in idx if j not in (i0, i1, i2)):
                continue
            tris.append((i0, i1, i2))
            idx.pop(k)
            break
        else:
            break  # degenerate polygon: give up on the rest
    if len(idx) == 3:
        tris.append(tuple(idx))
    return tris


def _rings(geom):
    """Exterior rings of a GeoJSON Polygon / MultiPolygon (holes ignored)."""
    t, c = geom.get("type"), geom.get("coordinates") or []
    if t == "Polygon":
        return [np.array(c[0])[:, :2]] if c else []
    if t == "MultiPolygon":
        return [np.array(pc[0])[:, :2] for pc in c if pc]
    return []


def _num(v, default=None):
    try:
        v = float(v)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def building_heights(props):
    """(eaves, roof height) from the BD TOPO attributes. BD TOPO `hauteur` is
    measured from the ground to the gutter (eaves); the roof height is the
    difference between the highest and lowest roof altitudes."""
    sol = _num(props.get("altitude_minimale_sol"))
    toit_min = _num(props.get("altitude_minimale_toit"))
    toit_max = _num(props.get("altitude_maximale_toit"))
    eave = _num(props.get("hauteur"))
    if (eave is None or eave < 1.5) and sol is not None and toit_min is not None:
        eave = toit_min - sol
    if eave is None or eave < 1.5:
        eave = 5.5
    eave = float(np.clip(eave, 2.3, 25.0))
    roof = toit_max - toit_min if (toit_max is not None and toit_min is not None) else None
    if roof is None or not (0.0 <= roof <= 8.0):
        roof = 1.8
    return eave, float(roof)


def build_buildings(geojson, dtm, tiles, axis_tree, half_width, p):
    """Walls (façade texture) and roofs (orthophoto: the real roof colours),
    gable roofs on near-rectangular footprints, flat ones otherwise."""
    fc = json.loads(open(geojson, encoding="utf-8").read())
    walls = {}   # tile -> lists
    roofs = {}
    n_b = n_gable = 0
    bay_u, bay_v = FACADE_BAY_M
    for f in fc.get("features", []):
        props_ = f.get("properties") or {}
        if p.get("buildings_skip_light", True) and props_.get("construction_legere"):
            continue   # sheds, greenhouses, carports: they all looked like houses
        for ring in _rings(f.get("geometry") or {}):
            ex = _clean_ring(ring)
            if len(ex) < 3:
                continue
            area = _signed_area(ex)
            if area < 0:            # make it counter-clockwise
                ex, area = ex[::-1].copy(), -area
            if area < p["buildings_min_area_m2"]:
                continue
            dist, k = axis_tree.query(ex)
            if dist.min() > p["buildings_max_m"] or (dist < half_width[k] + 1.0).any():
                continue   # too far, or on the road (footprint error)
            zg = dtm.sample(ex[:, 0], ex[:, 1])
            if not np.isfinite(zg).any():
                continue
            zg = np.where(np.isfinite(zg), zg, np.nanmin(zg))
            ground, top_ground = float(zg.min()), float(zg.max())
            eave, roof_h = building_heights(f.get("properties") or {})
            z_eave = max(float(zg.mean()) + eave, top_ground + 2.2)
            z_ridge = z_eave + roof_h
            z_base = ground - 0.5
            cxy = ex.mean(0)
            ti = int(_tile_index(np.array([cxy[0]]), np.array([cxy[1]]), tiles)[0])
            if ti < 0:
                continue
            W = walls.setdefault(ti, ([], [], []))
            R = roofs.setdefault(ti, ([], [], []))
            # walls
            cum = 0.0
            for i in range(len(ex)):
                a_, b_ = ex[i], ex[(i + 1) % len(ex)]
                L = float(np.hypot(*(b_ - a_)))
                if L < 0.05:
                    continue
                v = np.array([[*a_, z_base], [*b_, z_base], [*b_, z_eave], [*a_, z_eave]])
                uv = np.array([[cum / bay_u, (z_base - ground) / bay_v], [(cum + L) / bay_u, (z_base - ground) / bay_v],
                               [(cum + L) / bay_u, (z_eave - ground) / bay_v], [cum / bay_u, (z_eave - ground) / bay_v]])
                _append(W, v, np.array([[0, 1, 2], [0, 2, 3]]), uv)
                cum += L
            # roof
            q, rect_area = _min_rect(ex)
            rect = area / max(rect_area, 1e-6)
            tb = tiles[ti]["bounds"]
            short = min(np.hypot(*(q[1] - q[0])), np.hypot(*(q[2] - q[1])))
            z_ridge = z_eave + min(z_ridge - z_eave, 0.42 * short)
            if z_ridge - z_eave >= 0.8 and rect >= 0.85 and len(ex) <= 12:
                n_gable += 1
                e0, e1 = q[1] - q[0], q[2] - q[1]
                if np.hypot(*e0) < np.hypot(*e1):  # make q0→q1 a long side
                    q = np.roll(q, -1, axis=0)
                # ridge between the middles of the two short sides
                m1, m2 = (q[1] + q[2]) / 2, (q[3] + q[0]) / 2
                ov = 0.3  # eaves overhang
                cen = q.mean(0)
                qo = cen + (q - cen) * (1 + ov / max(np.hypot(*(q[0] - cen)), 1.0))
                for (pa, pb, ra, rb) in ((qo[0], qo[1], m2, m1), (qo[2], qo[3], m1, m2)):
                    v = np.array([[*pa, z_eave - 0.15], [*pb, z_eave - 0.15], [*ra, z_ridge], [*rb, z_ridge]])
                    _append(R, v, np.array([[0, 1, 3], [0, 3, 2]]), _ortho_uv(v, tb))
                # gable triangles (façade texture), on the rectangle's short sides
                for (pa, pb, rm) in ((q[1], q[2], m1), (q[3], q[0], m2)):
                    v = np.array([[*pa, z_eave], [*pb, z_eave], [*rm, z_ridge]])
                    L = float(np.hypot(*(pb - pa)))
                    uv = np.array([[0, (z_eave - ground) / bay_v], [L / bay_u, (z_eave - ground) / bay_v],
                                   [L / 2 / bay_u, (z_ridge - ground) / bay_v]])
                    nrm = np.cross(v[1] - v[0], v[2] - v[0])[:2]
                    tri = [[0, 1, 2]] if np.dot(nrm, (pa + pb) / 2 - cen) >= 0 else [[0, 2, 1]]  # face outwards
                    _append(W, v, np.array(tri), uv)
            else:
                tris = _ear_clip(ex)
                if tris:
                    v = np.column_stack([ex, np.full(len(ex), z_eave + 0.05)])
                    _append(R, v, np.array(tris), _ortho_uv(v, tb))
            n_b += 1
    out = []
    for ti, (V, F, UV) in walls.items():
        out += _chunks(f"buildings_walls_{_tile_name(tiles[ti])}", "facade", V, F, UV, keep_winding=True)
    for ti, (V, F, UV) in roofs.items():
        mname = f"ortho_{_tile_name(tiles[ti])}"
        out += _chunks(f"buildings_roofs_{_tile_name(tiles[ti])}", mname, V, F, UV, keep_winding=False)
    return out, n_b, n_gable


def _ortho_uv(v, b):
    x0, y0, x1, y1 = b
    return np.column_stack([np.clip((v[:, 0] - x0) / (x1 - x0), 0, 1), np.clip((v[:, 1] - y0) / (y1 - y0), 0, 1)])


def _append(acc, v, f, uv):
    V, F, UV = acc
    base = sum(len(a) for a in V)
    V.append(v)
    F.append(f + base)
    UV.append(uv)


def _chunks(name, material, V, F, UV, keep_winding):
    """Split the accumulated pieces into meshes under the vertex limit."""
    out, cur = [], ([], [], [])
    n = 0
    k = 0
    offs = np.cumsum([0] + [len(v) for v in V])
    for i in range(len(V)):
        if n + len(V[i]) > MAX_VERTS and cur[0]:
            out.append(_pack(f"{name}_{k}", material, cur, keep_winding))
            cur, n, k = ([], [], []), 0, k + 1
        cur[0].append(V[i])
        cur[1].append(F[i] - offs[i] + n)
        cur[2].append(UV[i])
        n += len(V[i])
    if cur[0]:
        out.append(_pack(f"{name}_{k}", material, cur, keep_winding))
    return out


def _pack(name, material, cur, keep_winding):
    return (name, material, np.concatenate(cur[0]), np.concatenate(cur[1]).astype(np.int32),
            np.concatenate(cur[2]), keep_winding)
