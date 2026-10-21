"""
3D scenery for build_track.py: trees and bushes (trees.py) and buildings
(BD TOPO, fetch.py buildings).

Meshes are returned as (name, material, verts, faces, uvs, keep_winding[, opts])
in Lambert-93 + metres; build_track.py shifts them to its local origin. opts:
surface ("WALL": cars collide with it), hidden (collision only), lod_out (m),
shadows.
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
TREE_LOD_M = 0.0         # tall trees: always drawn (they make the landscape)
BUSH_LOD_M = 220.0       # low bushes: not drawn beyond this distance
CARD_LOD_M = 260.0       # foliage cards inside the near crowns
CARD_MAX_ROAD_M = 45.0   # trees closer than this to the road get foliage cards
TRUNK_MAX_EDGE_M = 7.0
UNDERSTORY_MAX_ROAD_M = 40.0   # bushes added under the trees closer than this   # trunks closer than this to the road edge are solid (collisions)


def _quads_mesh(name, material, V, UV, opts):
    """(n, 4k, 3) quads → double-sided mesh chunks under the vertex limit."""
    n, per = V.shape[0], V.shape[1]
    out = []
    step = max(1, MAX_VERTS // per)
    quad = np.array([[0, 1, 2], [0, 2, 3], [0, 2, 1], [0, 3, 2]])
    for c0 in range(0, n, step):
        v = V[c0:c0 + step]
        m = len(v)
        base = (np.arange(m) * per)[:, None, None]
        F = np.concatenate([base + quad[None] + 4 * q for q in range(per // 4)], axis=1).reshape(-1, 3)
        out.append((f"{name}_{c0 // step}", material, v.reshape(-1, 3), F.astype(np.int32),
                    UV[c0:c0 + step].reshape(-1, 2), True, dict(opts)))
    return out


def _billboards(x, y, zg, h, r, kind, theta, pick, planes):
    """Star of `planes` vertical quads per tree, textured with the atlas cell."""
    n = len(x)
    per = 4 * planes
    V = np.zeros((n, per, 3))
    UV = np.zeros((n, per, 2))
    margin = 2.0 / (512 * TREE_CELLS)
    cell = np.array([TREE_CELL[k][p % len(TREE_CELL[k])] for k, p in zip(kind, pick)])
    u0, u1 = cell / TREE_CELLS + margin, (cell + 1) / TREE_CELLS - margin
    wf = np.array([TREE_WIDTH_FACTOR.get(k, 1.2) for k in kind])
    hw = r * wf
    zb, zt = zg - 0.15, zg - 0.15 + h * 1.03
    for q in range(planes):   # planes 180/planes degrees apart, like a star seen from above
        ang = theta + q * np.pi / planes
        ux, uy = np.cos(ang), np.sin(ang)
        for j, (sx, zz) in enumerate(((-1, zb), (1, zb), (1, zt), (-1, zt))):
            V[:, 4 * q + j, 0] = x + sx * hw * ux
            V[:, 4 * q + j, 1] = y + sx * hw * uy
            V[:, 4 * q + j, 2] = zz
        UV[:, 4 * q + 0] = np.column_stack([u0, np.zeros(n)])
        UV[:, 4 * q + 1] = np.column_stack([u1, np.zeros(n)])
        UV[:, 4 * q + 2] = np.column_stack([u1, np.ones(n)])
        UV[:, 4 * q + 3] = np.column_stack([u0, np.ones(n)])
    return V, UV


# crown depth (share of the height) per kind: pines have a crown on top of a
# bare trunk, oaks a deep round one, small trees down to the ground
CROWN_DEPTH = {0: 0.38, 1: 0.6, 2: 0.75}


def _cards(rng, x, y, zg, h, r, kind):
    """Foliage cards: tilted square quads spread inside the crown volume, so
    the crown looks full from every side (billboards alone look flat or see-
    through when seen at an angle)."""
    from textures import FOLIAGE_CELL, FOLIAGE_CELLS

    n_card = np.clip((2.5 + 1.1 * r).astype(int), 3, 9)
    tot = int(n_card.sum())
    owner = np.repeat(np.arange(len(x)), n_card)
    depth = np.array([CROWN_DEPTH.get(k, 0.6) for k in kind])[owner] * h[owner]
    zc = zg[owner] + h[owner] - 0.5 * depth
    # random point in the crown ellipsoid (denser towards the outside)
    d = rng.normal(size=(tot, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    rad = rng.uniform(0.35, 0.8, tot)
    cx = x[owner] + d[:, 0] * rad * r[owner]
    cy = y[owner] + d[:, 1] * rad * r[owner]
    cz = zc + d[:, 2] * rad * 0.42 * depth
    size = np.clip(0.95 * r[owner], 1.0, 3.6) * rng.uniform(0.8, 1.15, tot)
    phi = rng.uniform(0, np.pi, tot)
    tau = rng.uniform(-0.6, 0.6, tot)                   # tilt from vertical
    u = np.column_stack([np.cos(phi), np.sin(phi), np.zeros(tot)])
    n0 = np.column_stack([-np.sin(phi), np.cos(phi), np.zeros(tot)])
    v = np.column_stack([np.zeros(tot), np.zeros(tot), np.ones(tot)]) * np.cos(tau)[:, None] + n0 * np.sin(tau)[:, None]
    c = np.column_stack([cx, cy, cz])
    hs = (size / 2)[:, None]
    V = np.stack([c - u * hs - v * hs, c + u * hs - v * hs, c + u * hs + v * hs, c - u * hs + v * hs], axis=1)
    cell = np.array([FOLIAGE_CELL.get(k, 2) for k in kind])[owner]
    cell = np.where((cell == 0) & (rng.random(tot) < 0.5), 1, cell)
    m = 2.0 / (512 * FOLIAGE_CELLS)
    u0, u1 = cell / FOLIAGE_CELLS + m, (cell + 1) / FOLIAGE_CELLS - m
    UV = np.stack([np.column_stack([u0, np.zeros(tot) + m]), np.column_stack([u1, np.zeros(tot) + m]),
                   np.column_stack([u1, np.ones(tot) - m]), np.column_stack([u0, np.ones(tot) - m])], axis=1)
    return V, UV


def _trunks(x, y, zg, h):
    """Octagonal prisms around the trunks (collision only)."""
    k = np.arange(8) * np.pi / 4
    rr = (0.18 + 0.012 * h)[:, None]
    ring = np.stack([x[:, None] + rr * np.cos(k), y[:, None] + rr * np.sin(k)], axis=2)   # (n, 8, 2)
    lo = np.repeat((zg - 0.6)[:, None], 8, 1)
    hi = np.repeat((zg + 3.0)[:, None], 8, 1)
    V = np.concatenate([np.dstack([ring, lo]), np.dstack([ring, hi])], axis=1)            # (n, 16, 3)
    F = []
    for i in range(8):
        j = (i + 1) % 8
        F += [[i, j, 8 + j], [i, 8 + j, 8 + i]]
    return V, np.array(F)


def build_trees(trees_csv, dtm, tiles, road=None, seed=3, tall=True):
    """Tall trees: 3 crossed vertical quads (shader ksTree, normals up as
    Kunos does), plus foliage cards inside the crowns near the road; low
    bushes (kind 3): 2 quads, hidden beyond BUSH_LOD_M; solid trunks next to
    the road. road = (cKDTree of the axis, half width per station).
    tall=False: no billboards for the tall trees (CSP 3D trees are drawn
    instead); stats["tall"] then lists them (x, y, z, height, crown r, kind)."""
    t = np.genfromtxt(trees_csv, delimiter=",", names=True)
    if t.size == 0:
        return [], {}
    t = np.atleast_1d(t)
    x, y, h, r, kind = t["x"], t["y"], t["height_m"], t["crown_r_m"], t["kind"].astype(int)
    zg = dtm.sample(x, y)
    ok = np.isfinite(zg) & (h > 0.3)
    x, y, h, r, kind, zg = x[ok], y[ok], h[ok], r[ok], kind[ok], zg[ok]
    rng = np.random.default_rng(seed)
    if road is not None:
        dist, k = road[0].query(np.column_stack([x, y]))
        edge = dist - road[1][k]
    else:
        dist = edge = np.full(len(x), np.inf)
    # understory: the canopy model cannot see under the crowns, but woods
    # along the road are never bare between the trunks
    under = np.flatnonzero((kind < 3) & (h >= 6.0) & (dist < UNDERSTORY_MAX_ROAD_M))
    if len(under):
        n_u = rng.integers(1, 3, len(under))
        own = np.repeat(under, n_u)
        ang = rng.uniform(0, 2 * np.pi, len(own))
        rad = r[own] * np.sqrt(rng.uniform(0.05, 0.8, len(own)))
        ux, uy = x[own] + rad * np.cos(ang), y[own] + rad * np.sin(ang)
        uz = dtm.sample(ux, uy)
        ud = road[0].query(np.column_stack([ux, uy]))[0] if road is not None else np.full(len(own), np.inf)
        ue = ud - (road[1][road[0].query(np.column_stack([ux, uy]))[1]] if road is not None else 0)
        okk = np.isfinite(uz) & (ue > 1.5)
        x = np.concatenate([x, ux[okk]]); y = np.concatenate([y, uy[okk]]); zg = np.concatenate([zg, uz[okk]])
        h = np.concatenate([h, rng.uniform(1.1, 2.3, okk.sum())])
        r = np.concatenate([r, rng.uniform(0.8, 1.4, okk.sum())])
        kind = np.concatenate([kind, np.full(okk.sum(), 3)])
        dist = np.concatenate([dist, ud[okk]]); edge = np.concatenate([edge, ue[okk]])
    theta = rng.uniform(0, np.pi, len(x))
    pick = rng.integers(0, 2, len(x))
    tidx = _tile_index(x, y, tiles)
    out = []
    stats = {"trees": int((kind < 3).sum()), "bushes": int((kind == 3).sum()), "cards": 0, "trunks": 0}
    if not tall:
        sel = kind < 3
        stats["tall"] = np.column_stack([x[sel], y[sel], zg[sel], h[sel], r[sel], kind[sel]])
    for ti in np.unique(tidx):
        tn = _tile_name(tiles[ti]) if ti >= 0 else "out"
        sel_t = np.flatnonzero((tidx == ti) & (kind < 3))
        sel_b = np.flatnonzero((tidx == ti) & (kind == 3))
        if len(sel_t) and tall:
            V, UV = _billboards(x[sel_t], y[sel_t], zg[sel_t], h[sel_t], r[sel_t], kind[sel_t],
                                theta[sel_t], pick[sel_t], PLANES)
            out += _quads_mesh(f"trees_{tn}", "trees", V, UV, {"lod_out": TREE_LOD_M})
            near = sel_t[dist[sel_t] < CARD_MAX_ROAD_M]
            if len(near):
                V, UV = _cards(rng, x[near], y[near], zg[near], h[near], r[near], kind[near])
                out += _quads_mesh(f"foliage_{tn}", "foliage", V, UV, {"lod_out": CARD_LOD_M})
                stats["cards"] += len(V)
        if len(sel_t):
            solid = sel_t[(edge[sel_t] < TRUNK_MAX_EDGE_M) & (h[sel_t] >= 4.0)]
            if len(solid):
                TV, TF = _trunks(x[solid], y[solid], zg[solid], h[solid])
                per = TV.shape[1]
                step = MAX_VERTS // per
                for c0 in range(0, len(solid), step):
                    v = TV[c0:c0 + step]
                    F = (np.arange(len(v)) * per)[:, None, None] + TF[None]
                    out.append((f"trunks_{tn}_{c0 // step}", "trees", v.reshape(-1, 3), F.reshape(-1, 3).astype(np.int32),
                                np.zeros((v.shape[0] * per, 2)), True, {"surface": "WALL", "hidden": True}))
                stats["trunks"] += len(solid)
        if len(sel_b):
            V, UV = _billboards(x[sel_b], y[sel_b], zg[sel_b], h[sel_b], r[sel_b], kind[sel_b],
                                theta[sel_b], pick[sel_b], 2)
            out += _quads_mesh(f"bushes_{tn}", "trees", V, UV, {"lod_out": BUSH_LOD_M, "shadows": False})
    return out, stats


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


def build_buildings(geojson, dtm, tiles, axis_tree, half_width, p, climate=None, roof_mats=None):
    """Walls and roofs, gable roofs on near-rectangular footprints, flat ones
    otherwise. climate: façade variants of village.py per building (else one
    "facade" material); roof_mats: photographed tiles chosen from the roof
    colour on the orthophoto (else the orthophoto itself on the roofs)."""
    fc = json.loads(open(geojson, encoding="utf-8").read())
    walls = {}   # material -> lists
    roofs = {}   # material (+ tile for orthophoto roofs) -> lists
    sampler = None
    if roof_mats:
        from roadside import OrthoSampler

        sampler = OrthoSampler(tiles)
    if climate:
        from village import ROOF_REPEAT_M, annex_material, wall_material

        # houses first (centroid, style): annexes take the style of the nearest one
        hc, hm = [], []
        for f in fc.get("features", []):
            pr = f.get("properties") or {}
            for ring in _rings(f.get("geometry") or {}):
                m_ = wall_material(pr, climate, abs(_signed_area(np.asarray(ring, float))))
                if m_ and m_.startswith("facade_") and m_[7:].isdigit():
                    hc.append(np.asarray(ring, float).mean(0))
                    hm.append(m_)
        from scipy.spatial import cKDTree as _KD

        htree = _KD(np.array(hc)) if hc else None

        def _nearest_house(c):
            if htree is None:
                return None
            dd, kk = htree.query(c)
            return hm[kk] if dd < 40 else None
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
            wmat = wall_material(props_, climate, area) if climate else "facade"
            if climate and wmat is None:     # annex: plain wall of the nearest house
                wmat = annex_material(_nearest_house(ex.mean(0)))
            W = walls.setdefault(wmat, ([], [], []))
            rkind = None
            if sampler is not None:
                from village import roof_kind

                pts_ = np.vstack([ex, cxy[None, :]])
                pts_ = cxy + 0.6 * (pts_ - cxy)               # inside the footprint
                rkind = roof_kind(sampler.colours(pts_[:, 0], pts_[:, 1]).mean(0) / 255.0)
                if rkind not in roof_mats:
                    rkind = None
            rkey = f"roof_{rkind}" if rkind else (ti,)
            R = roofs.setdefault(rkey, ([], [], []))
            tb = tiles[ti]["bounds"]
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
                    if rkind is None:
                        ruv = _ortho_uv(v, tb)
                    else:   # tiles: u along the eaves, v up the slope
                        e_ = (pb - pa) / max(np.hypot(*(pb - pa)), 1e-6)
                        up_ = v - v[0]
                        uu = up_[:, :2] @ e_
                        vv = np.hypot(np.linalg.norm(up_[:, :2] - uu[:, None] * e_, axis=1), up_[:, 2])
                        ruv = np.column_stack([uu, vv]) / ROOF_REPEAT_M
                    _append(R, v, np.array([[0, 1, 3], [0, 3, 2]]), ruv)
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
                    ruv = _ortho_uv(v, tb) if rkind is None else v[:, :2] / ROOF_REPEAT_M
                    _append(R, v, np.array(tris), ruv)
            n_b += 1
    out = []
    for wm, (V, F, UV) in walls.items():
        out += [m_ + ({"surface": "WALL"},)   # solid: cars hit the walls
                for m_ in _chunks(f"buildings_walls_{wm}", wm, V, F, UV, keep_winding=True)]
    for key, (V, F, UV) in roofs.items():
        if isinstance(key, tuple):
            name = _tile_name(tiles[key[0]])
            out += _chunks(f"buildings_roofs_{name}", f"ortho_{name}", V, F, UV, keep_winding=False)
        else:
            out += _chunks(f"buildings_{key}", key, V, F, UV, keep_winding=False)
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
