"""
Villages: house façades, roofs and pavements.

Façades are one bay of 4 m x 3 m (FACADE_BAY_M) that tiles along the walls and
per floor: a photographed render or stone (ambientCG, CC0, assets.py) in the
colours of the region, with a window, its frame, sill and shutters drawn on
it. Several variants per road, picked per building (BD TOPO id), so a street
is not one colour. Annexes get wood or plain render without windows, farm
and industrial buildings corrugated metal, churches stone. Roofs get
photographed tiles, the kind chosen from the roof colour on the orthophoto
(clay red, grey, dark). Without the assets: the same, drawn on a plain render.
"""
import hashlib

import numpy as np
from PIL import Image, ImageDraw

from textures import FACADE_BAY_M

PX_PER_M = 128            # façade texture: 512 x 384 for a 4 m x 3 m bay

# (wall photo, render colour, window frame, shutters or None)
STYLES = {
    "mediterranean": [
        ("Plaster002", (0.86, 0.74, 0.56), (0.93, 0.90, 0.82), (0.47, 0.60, 0.55)),   # ochre, sage shutters
        ("Plaster003", (0.90, 0.80, 0.68), (0.95, 0.93, 0.88), (0.42, 0.52, 0.62)),   # cream, blue-grey
        ("Plaster002", (0.86, 0.66, 0.55), (0.94, 0.90, 0.84), (0.55, 0.44, 0.33)),   # rose, brown
        ("PaintedPlaster017", (0.93, 0.90, 0.84), (0.97, 0.96, 0.92), (0.38, 0.50, 0.40)),
        ("Bricks066", (0.78, 0.72, 0.62), (0.90, 0.86, 0.78), (0.52, 0.56, 0.60)),    # stone house
        ("Plaster003", (0.82, 0.70, 0.52), (0.92, 0.88, 0.80), (0.60, 0.40, 0.32)),   # dark ochre, red-brown
    ],
    "temperate": [
        ("PaintedPlaster017", (0.92, 0.91, 0.87), (0.72, 0.45, 0.40), None),         # white, sandstone frames
        ("Plaster003", (0.89, 0.85, 0.76), (0.72, 0.47, 0.42), (0.45, 0.33, 0.25)),   # cream, wood shutters
        ("Plaster002", (0.80, 0.80, 0.77), (0.85, 0.83, 0.78), None),                 # grey render
        ("Plaster003", (0.90, 0.82, 0.70), (0.70, 0.46, 0.40), (0.36, 0.42, 0.34)),
        ("Bricks099", (0.82, 0.62, 0.55), (0.78, 0.58, 0.52), None),                  # sandstone house
        ("PaintedPlaster017", (0.88, 0.86, 0.80), (0.86, 0.84, 0.80), (0.50, 0.36, 0.28)),
    ],
}
ROOF_ASSETS = {"clay": "RoofingTiles013A", "grey": "RoofingTiles010", "dark": "RoofingTiles009"}
ROOF_REPEAT_M = 2.5


def _photo(assets, aid, size, kind="Color"):
    if assets:
        f = sorted((assets / aid).glob(f"*_{kind}.jpg"))
        if f:
            im = Image.open(f[0]).convert("RGB").resize(size, Image.LANCZOS)
            return np.asarray(im, np.float32) / 255.0
    return None


def _render(rng, size):
    """Plain render (no assets): smooth blotches and fine grain, mean 1."""
    h, w = size[1], size[0]
    small = rng.normal(0, 1, (h // 16 + 1, w // 16 + 1))
    big = np.asarray(Image.fromarray(((small - small.min()) / np.ptp(small) * 255).astype(np.uint8))
                     .resize((w, h), Image.BICUBIC), np.float32) / 255
    return (0.9 + 0.2 * big)[..., None] + 0.03 * rng.normal(0, 1, (h, w, 1))


def _wall(assets, aid, colour, rng, size):
    """Wall texture: the photo's detail (luminance, mean 1) times the colour."""
    ph = _photo(assets, aid, size)
    if ph is None:
        det = _render(rng, size)
    else:
        lum = ph.mean(2, keepdims=True)
        det = lum / max(float(lum.mean()), 1e-3)
        if aid.startswith("Bricks"):          # stone keeps some of its own colour
            det = 0.5 * det + 0.5 * ph / np.maximum(ph.mean((0, 1)), 1e-3)
        det = 1 + 0.8 * (det - 1)
    return np.clip(det * np.asarray(colour, np.float32), 0, 1)


def _window(img, frame, shutters, rng):
    """Window 1.0 x 1.35 m in the middle of the bay, 0.85 m above the floor:
    stone or painted frame, sill, glass with mullions, shutters."""
    d = ImageDraw.Draw(img)
    m = PX_PER_M
    w, h = img.size
    x0, x1 = int(w / 2 - 0.5 * m), int(w / 2 + 0.5 * m)
    y1 = int(h - 0.85 * m)                   # image y down: floor at the bottom
    y0 = int(y1 - 1.35 * m)
    fr = tuple(int(255 * c) for c in frame)
    b = int(0.1 * m)
    d.rectangle([x0 - b, y0 - b, x1 + b, y1 + b], fill=fr)
    d.rectangle([x0 - b - 6, y1 + b - 4, x1 + b + 6, y1 + b + 8], fill=tuple(int(0.92 * c) for c in fr))  # sill
    # glass: dark, sky reflection at the top
    for yy in range(y0, y1):
        t = (yy - y0) / max(1, y1 - y0)
        g = int(40 + 50 * (1 - t) ** 2)
        d.line([(x0, yy), (x1, yy)], fill=(g - 6, g, g + 10))
    wood = (230, 228, 220)
    d.line([((x0 + x1) // 2, y0), ((x0 + x1) // 2, y1)], fill=wood, width=5)
    for k in (1, 2):
        yy = y0 + k * (y1 - y0) // 3
        d.line([(x0, yy), (x1, yy)], fill=wood, width=4)
    d.rectangle([x0, y0, x1, y1], outline=wood, width=5)
    if shutters is not None:
        sc = np.asarray(shutters) * rng.uniform(0.9, 1.08)
        col = tuple(int(255 * np.clip(c, 0, 1)) for c in sc)
        dark = tuple(int(0.8 * c) for c in col)
        sw = (x1 - x0) // 2 + 4
        for sx0 in (x0 - b - sw - 4, x1 + b + 4):
            d.rectangle([sx0, y0 - 2, sx0 + sw, y1 + 2], fill=col)
            for yy in range(y0 + 6, y1, 11):     # louvres
                d.line([(sx0 + 4, yy), (sx0 + sw - 4, yy)], fill=dark, width=3)
            d.rectangle([sx0, y0 - 2, sx0 + sw, y1 + 2], outline=dark, width=3)


def facade_textures(assets, climate, tex, seed=11):
    """All façade materials of the road: {material: spec}. Grime at the foot of
    the walls, a darker plinth band, then the window."""
    size = (int(FACADE_BAY_M[0] * PX_PER_M), int(FACADE_BAY_M[1] * PX_PER_M))
    rng = np.random.default_rng(seed)
    mats = {}
    for i, (aid, col, frame, shutters) in enumerate(STYLES[climate]):
        arr = _wall(assets, aid, col, rng, size)
        img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
        _window(img, frame, shutters, rng)
        name = f"facade_{i}"
        Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).save(tex / f"{name}_plain.png")
        mats[f"{name}_plain"] = {"texture": f"{name}_plain.png", "kind": "facade"}
        img.save(tex / f"{name}.png")
        mats[name] = {"texture": f"{name}.png", "kind": "facade"}
    # annexes / barns: wood or plain render, no window
    wood = _photo(assets, "Planks021", size)
    annex = (wood * np.array([1.0, 0.95, 0.9])) if wood is not None else _wall(None, "", (0.72, 0.62, 0.5), rng, size)
    Image.fromarray((np.clip(annex, 0, 1) * 255).astype(np.uint8)).save(tex / "facade_annex.png")
    mats["facade_annex"] = {"texture": "facade_annex.png", "kind": "facade"}
    metal = _photo(assets, "CorrugatedSteel005", size)
    ind = metal if metal is not None else _wall(None, "", (0.7, 0.72, 0.72), rng, size)
    Image.fromarray((np.clip(ind, 0, 1) * 255).astype(np.uint8)).save(tex / "facade_shed.png")
    mats["facade_shed"] = {"texture": "facade_shed.png", "kind": "facade"}
    stone = _wall(assets, "Bricks099" if climate == "temperate" else "Bricks066",
                  (0.86, 0.72, 0.64) if climate == "temperate" else (0.84, 0.78, 0.68), rng, size)
    Image.fromarray((np.clip(stone, 0, 1) * 255).astype(np.uint8)).save(tex / "facade_church.png")
    mats["facade_church"] = {"texture": "facade_church.png", "kind": "facade"}
    return mats


def roof_textures(assets, tex):
    """Photographed roof tiles: {kind: material} for the kinds found."""
    mats = {}
    for kind, aid in ROOF_ASSETS.items():
        ph = _photo(assets, aid, (512, 512))
        if ph is None:
            continue
        Image.fromarray((np.clip(ph, 0, 1) * 255).astype(np.uint8)).save(tex / f"roof_{kind}.png")
        mats[kind] = {"texture": f"roof_{kind}.png", "kind": "facade"}
    return mats


def roof_kind(rgb):
    """Kind of roof from its colour seen from above (0-1 RGB)."""
    r, g, b = rgb
    lum = (r + g + b) / 3
    if r > 1.12 * g and r > 1.2 * b:
        return "clay"
    return "dark" if lum < 0.3 else "grey"


def wall_material(props, climate, area=0.0):
    """Façade material of a BD TOPO building; None for an annex (it takes the
    plain wall of the nearest house, see annex_material)."""
    usage = str(props.get("usage_1") or "")
    nature = str(props.get("nature") or "")
    if nature in ("Eglise", "Chapelle") or usage == "Religieux":
        return "facade_church"
    # sheds and barns: farm / industrial use, or a big "industrial, farm or
    # commercial" building of unknown use (shops and services stay houses)
    if usage in ("Industriel", "Agricole") or ("Industriel" in nature and usage in ("", "Indifférencié", "None")
                                               and area > 200):
        return "facade_shed"
    if usage == "Annexe":
        return None
    key = str(props.get("cleabs") or props.get("id") or "")
    h = int(hashlib.md5(key.encode()).hexdigest()[:6], 16)
    return f"facade_{h % len(STYLES[climate])}"


def annex_material(house_material):
    """Annexes (garages, extensions, BD TOPO splits of one house): the render
    of the house next to them, without windows; wood only far from any house."""
    if house_material and house_material.startswith("facade_") and house_material[7:].isdigit():
        return house_material + "_plain"
    return "facade_annex"


# ------------------------------------------------------------- pavements
SIDEWALK = {
    "sidewalks": True,
    "sidewalk_building_m": 12.0,   # a house within this distance of the road edge: village street
    "sidewalk_share": 0.35,        # ... on this share of the 100 m around
    "sidewalk_width_m": 1.6,       # (narrower when the house is closer)
    "sidewalk_min_m": 0.8,
    "sidewalk_kerb_m": 0.12,
    "sidewalk_min_run_m": 20.0,
    "sidewalk_join_m": 25.0,       # gaps shorter than this (driveways) are bridged
    "sidewalk_junction_m": 5.0,    # cut this far on each side of a side road (+ its half width)
}


def _footprint_points(geojson, step=1.0):
    import json

    fc = json.loads(open(geojson, encoding="utf-8").read())
    pts = []
    for f in fc.get("features", []):
        g = f.get("geometry") or {}
        rings = [g["coordinates"][0]] if g.get("type") == "Polygon" else \
            [c[0] for c in g.get("coordinates", [])] if g.get("type") == "MultiPolygon" else []
        for r in rings:
            r = np.asarray(r, float)[:, :2]
            for a, b in zip(r[:-1], r[1:]):
                n = max(1, int(np.hypot(*(b - a)) / step))
                pts.append(a + (b - a) * (np.arange(n) / n)[:, None])
    return np.vstack(pts) if pts else np.zeros((0, 2))


def _junctions(ign_raw, road, max_d=15.0):
    """(station, side, half width) of the side roads joining the road (BD TOPO
    segments whose end is next to the carriageway but which are not the road)."""
    import json

    if not ign_raw.exists():
        return []
    out = []
    for f in json.loads(ign_raw.read_text(encoding="utf-8")):
        g = f.get("geometry") or {}
        lines = [g["coordinates"]] if g.get("type") == "LineString" else g.get("coordinates", []) \
            if g.get("type") == "MultiLineString" else []
        w = (f.get("properties") or {}).get("largeur_de_chaussee") or 4.0
        for ln in lines:
            c = np.asarray(ln, float)[:, :2]
            if len(c) < 2:
                continue
            d_all, _ = road.tree.query(c)
            if np.median(d_all) < 4.0:          # the road itself
                continue
            for end, nxt in ((c[0], c[min(1, len(c) - 1)]), (c[-1], c[max(-2, -len(c))])):
                d, k = road.tree.query(end)
                if d > max_d:
                    continue
                t = (nxt[0] - road.x[k]) * road.f["nx"][k] + (nxt[1] - road.y[k]) * road.f["ny"][k]
                out.append((float(road.s[k]), 1 if t > 0 else -1, float(w) / 2))
    return out


def _runs(mask):
    m = np.r_[False, mask, False].astype(np.int8)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def plan_sidewalks(road, buildings_geojson, ign_raw, q):
    """Width of the pavement (0 = none) per station on each side {+1 left, -1 right}."""
    pts = _footprint_points(buildings_geojson)
    n = len(road.s)
    st = road.step
    out = {}
    if not len(pts):
        return {-1: np.zeros(n), 1: np.zeros(n)}
    from scipy.spatial import cKDTree
    from scipy.ndimage import median_filter, uniform_filter1d

    tree = cKDTree(pts)
    junc = _junctions(ign_raw, road)
    for sd in (-1, 1):
        e = road.f["left"] if sd > 0 else road.f["right"]
        ex = road.x + e * road.f["nx"]
        ey = road.y + e * road.f["ny"]
        gap = np.full(n, np.inf)
        for k, nb in enumerate(tree.query_ball_point(np.column_stack([ex, ey]), q["sidewalk_building_m"] + 3)):
            if not nb:
                continue
            P = pts[nb] - [ex[k], ey[k]]
            t = sd * (P[:, 0] * road.f["nx"][k] + P[:, 1] * road.f["ny"][k])
            a = P[:, 0] * road.f["tx"][k] + P[:, 1] * road.f["ty"][k]
            ok = (t > 0.3) & (np.abs(a) < 3.0)
            if ok.any():
                gap[k] = t[ok].min()
        near = gap < q["sidewalk_building_m"]
        share = uniform_filter1d(near.astype(float), max(3, int(100 / st)), mode="nearest")
        on = near | (share > q["sidewalk_share"])
        on &= share > q["sidewalk_share"]
        w = np.where(gap < q["sidewalk_width_m"] + 0.5, np.clip(gap - 0.25, q["sidewalk_min_m"], None),
                     q["sidewalk_width_m"])
        w = np.where(on, w, 0.0)
        # bridge driveways, drop short pieces
        for a_, b_ in _runs(w == 0):
            if 0 < a_ and b_ < n and (b_ - a_) * st < q["sidewalk_join_m"]:
                w[a_:b_] = min(w[a_ - 1], w[b_])
        for a_, b_ in _runs(w > 0):
            if (b_ - a_) * st < q["sidewalk_min_run_m"]:
                w[a_:b_] = 0
        w = np.where(w > 0, median_filter(np.where(w > 0, w, np.nan), size=int(9 / st) | 1, mode="nearest"), 0)
        w = np.nan_to_num(w)
        # open at the side roads
        for s0, side, hw in junc:
            if side == sd:
                w[np.abs(road.s - s0) < q["sidewalk_junction_m"] + hw] = 0
        out[sd] = w
    return out


def sidewalk_top(road, ks, sd, ww, h):
    """Height of the pavement at the kerb: a kerb above the road edge, or the
    real raised pavement of the LiDAR terrain when it is higher (town
    centres), smoothed along the road and capped at 45 cm."""
    from scipy.ndimage import uniform_filter1d

    e = road.edge(ks, sd)
    ze = road.road_z(ks, e)
    px, py = road.xy(ks, e + sd * np.maximum(ww * 0.6, 0.5))
    zg = road.dtm.sample(np.atleast_1d(px), np.atleast_1d(py))
    zg = np.where(np.isfinite(zg), zg, ze)
    top = np.clip(np.maximum(ze + h, zg + 0.03), ze + h, ze + 0.45)
    return uniform_filter1d(top, max(1, min(len(top), int(7 / road.step))), mode="nearest")


def clamp_verges(verges, road, plan, q):
    """Keep the verge strip under the pavement: where it rose above it (ground
    beside the road higher than the kerb) it covered the pavement with the
    orthophoto (earth-looking pavements)."""
    offs = verges["offs"]
    for side, sd in (("right", -1), ("left", 1)):
        w = plan[sd]
        ks = np.flatnonzero(w > 0)
        if not len(ks):
            continue
        top = np.full(len(w), np.nan)
        for a_, b_ in _runs(w > 0):
            kk = np.arange(a_, b_)
            top[kk] = sidewalk_top(road, kk, sd, w[kk], q["sidewalk_kerb_m"])
        VZ = verges[side][2]
        cols = np.arange(len(offs))[::-1] if side == "right" else np.arange(len(offs))
        for ci, o in zip(cols, offs):
            m = (w > 0) & (o <= w + 0.3)
            if m.any():
                VZ[m, ci] = np.minimum(VZ[m, ci], top[m] - 0.06)


def build_sidewalks(road, plan, q, acc):
    """Pavement meshes: top at kerb height above the road edge (cross slope 2 %
    towards the road), the kerb face, the outer face down into the ground.
    Drivable (ROAD surface): the kerb is a 12 cm step. Returns metres built."""
    h = q["sidewalk_kerb_m"]
    total = 0.0
    for sd, w in plan.items():
        for a_, b_ in _runs(w > 0):
            ks = np.arange(a_, b_)
            if len(ks) < 2:
                continue
            e = road.edge(ks, sd)
            ww = w[ks]
            ze = road.road_z(ks, e)
            pin = np.column_stack(road.xy(ks, e))
            pout = np.column_stack(road.xy(ks, e + sd * ww))
            zt_in = sidewalk_top(road, ks, sd, ww, h)
            zt_out = zt_in + 0.02 * ww
            ground = road.ground(ks, e + sd * (ww + 0.3)) - 0.3
            n = len(ks)
            V_top = np.vstack([np.column_stack([pin, zt_in]), np.column_stack([pout, zt_out])])
            F = [[i, i + 1, n + i + 1] for i in range(n - 1)] + [[i, n + i + 1, n + i] for i in range(n - 1)]
            sv = road.s[ks]
            UV = np.vstack([np.column_stack([sv / 2.0, np.zeros(n)]), np.column_stack([sv / 2.0, ww / 2.0])])
            acc.add("sidewalk", "sidewalk", V_top, np.array(F), UV, {"surface": "ROAD"})
            # kerb face (road side) and outer face, both windings (seen from both sides)
            for P, zlo, zhi in ((pin, ze - 0.02, zt_in), (pout, np.minimum(ground, zt_out - 0.2), zt_out)):
                V = np.vstack([np.column_stack([P, zlo]), np.column_stack([P, zhi])])
                Fk = [[i, i + 1, n + i + 1] for i in range(n - 1)] + [[i, n + i + 1, n + i] for i in range(n - 1)]
                Fk += [[a, c, b] for a, b, c in Fk]
                UVk = np.vstack([np.column_stack([sv / 2.0, np.zeros(n)]),
                                 np.column_stack([sv / 2.0, (zhi - zlo) / 2.0])])
                acc.add("kerb", "kerb", V, np.array(Fk), UVk, {"surface": "ROAD"})
            total += float(road.s[ks[-1]] - road.s[ks[0]])
    return total


def sidewalk_textures(assets, tex):
    """Pavement: fine grey asphalt (most village pavements in France), smooth
    and even, so it does not read as ground; kerb: light concrete."""
    pav = _photo(assets, "Asphalt031", (512, 512))
    rng = np.random.default_rng(5)
    if pav is None:
        pav = np.clip(0.42 * _render(rng, (512, 512)), 0, 1).repeat(3, 2)
    else:
        lum = pav.mean(2, keepdims=True)
        pav = np.clip(0.42 + 0.5 * (lum - lum.mean()), 0, 1).repeat(3, 2) * np.array([1.0, 1.0, 1.02])
    Image.fromarray((np.clip(pav, 0, 1) * 255).astype(np.uint8)).save(tex / "sidewalk.png")
    kerb = _photo(assets, "Concrete034", (256, 256))
    kerb = np.clip(kerb * (0.68 / max(float(kerb.mean()), 1e-3)), 0, 1) if kerb is not None else \
        np.clip(0.68 * _render(rng, (256, 256)), 0, 1).repeat(3, 2)
    Image.fromarray((kerb * 255).astype(np.uint8)).save(tex / "kerb.png")
    return {"sidewalk": {"texture": "sidewalk.png", "kind": "facade"},
            "kerb": {"texture": "kerb.png", "kind": "facade"}}
