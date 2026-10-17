#!/usr/bin/env python3
"""
3D tree models for Custom Shaders Patch (CSP) track trees.

CSP draws real 3D trees (volume, wind, shadows, LODs, instancing) from small
.bin models placed by a text list. This script builds the models from
numbers only (no third-party assets): trunk and branches as tapered tubes,
foliage as leaf-cluster cards around the branch tips, three levels of detail,
generated bark and foliage textures (diffuse + normal) in the orthophoto
colours (trees_palette.json).

    python trees3d.py                → data/<name>/csp_trees/src/*.obj + textures
    blender -b -P blender_trees.py -- data/<name>/csp_trees/src
                                     → data/<name>/csp_trees/<species>.fbx

Then, once, in Content Manager → Content → Tools → Creator tools → Tree
converter: each .fbx → data/<name>/csp_trees/<species>.bin. build_track.py
and kn5_export.py use the .bin models instead of the flat tree billboards as
soon as all of them are there.

Species (kind in trees.csv): 0 Aleppo pine (two shapes), 1 holm oak,
2 small tree / tall shrub. Model heights in MODEL_HEIGHT; every tree of the
list is scaled to its LiDAR height and crown width.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SPECIES = ["pine_a", "pine_b", "oak", "shrub"]
KIND_SPECIES = {0: ("pine_a", "pine_b"), 1: ("oak",), 2: ("shrub",)}
MODEL_HEIGHT = {"pine_a": 14.0, "pine_b": 12.0, "oak": 9.0, "shrub": 4.0}
MODEL_CROWN_R = {"pine_a": 4.5, "pine_b": 5.0, "oak": 4.0, "shrub": 2.0}
# (radial segments of the trunk, of the branches, share of leaf cards kept)
LODS = {"A": (10, 6, 1.0), "B": (7, 4, 0.55), "C": (5, 3, 0.28)}


# ------------------------------------------------------------- geometry
class Mesh:
    def __init__(self):
        self.v, self.n, self.uv, self.f = [], [], [], {}   # f: material -> faces (v index triples)

    def add(self, mat, V, N, UV, F):
        base = len(self.v)
        self.v += list(V)
        self.n += list(N)
        self.uv += list(UV)
        self.f.setdefault(mat, []).extend((np.asarray(F) + base).tolist())

    def tris(self):
        return sum(len(f) for f in self.f.values())


def _frame(d):
    d = d / np.linalg.norm(d)
    a = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(d, a)
    u /= np.linalg.norm(u)
    return u, np.cross(d, u)


def tube(mesh, path, radii, seg, mat="bark", v_scale=1.0):
    """Tapered tube along a polyline (path: (n, 3), radii: (n,)), with
    parallel-transported rings; uv u around, v along (metres / v_scale)."""
    path = np.asarray(path, float)
    n = len(path)
    t = np.gradient(path, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    u, w = _frame(t[0])
    rings_v, rings_n, rings_uv = [], [], []
    vlen = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    for i in range(n):
        if i:
            # parallel transport of the frame
            b = np.cross(t[i - 1], t[i])
            if np.linalg.norm(b) > 1e-6:
                b /= np.linalg.norm(b)
                ang = math.acos(np.clip(np.dot(t[i - 1], t[i]), -1, 1))
                u = _rot(u, b, ang)
            u -= t[i] * np.dot(u, t[i])
            u /= np.linalg.norm(u)
            w = np.cross(t[i], u)
        for k in range(seg + 1):
            a = 2 * math.pi * k / seg
            nn = math.cos(a) * u + math.sin(a) * w
            rings_v.append(path[i] + radii[i] * nn)
            rings_n.append(nn)
            rings_uv.append([k / seg, vlen[i] / v_scale])
    F = []
    for i in range(n - 1):
        for k in range(seg):
            a0 = i * (seg + 1) + k
            b0 = a0 + seg + 1
            F += [[a0, a0 + 1, b0 + 1], [a0, b0 + 1, b0]]
    mesh.add(mat, rings_v, rings_n, rings_uv, F)


def _rot(v, axis, ang):
    c, s = math.cos(ang), math.sin(ang)
    return v * c + np.cross(axis, v) * s + axis * np.dot(axis, v) * (1 - c)


def card(mesh, c, size, normal_from, rng, cell, cells=4, tilt=0.7):
    """One leaf-cluster card (double-sided quad), random orientation, normals
    pointing away from the crown centre (round, soft shading like real crowns)."""
    phi = rng.uniform(0, math.pi)
    tau = rng.uniform(-tilt, tilt)
    u = np.array([math.cos(phi), math.sin(phi), 0.0])
    n0 = np.array([-math.sin(phi), math.cos(phi), 0.0])
    v = np.array([0, 0, 1.0]) * math.cos(tau) + n0 * math.sin(tau)
    hs = size / 2
    P = [c - u * hs - v * hs, c + u * hs - v * hs, c + u * hs + v * hs, c - u * hs + v * hs]
    nrm = c - normal_from
    nrm = nrm / (np.linalg.norm(nrm) + 1e-9)
    nrm = 0.65 * nrm + 0.35 * np.array([0, 0, 1.0])
    nrm /= np.linalg.norm(nrm)
    m = 2.0 / (512 * cells)
    u0, u1 = cell / cells + m, (cell + 1) / cells - m
    UV = [[u0, m], [u1, m], [u1, 1 - m], [u0, 1 - m]]
    mesh.add("leaves", P + P, [nrm] * 8, UV + UV, [[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6]])


def _branch_path(start, direction, length, rng, droop=0.15, wiggle=0.08, n=5):
    d = direction / np.linalg.norm(direction)
    pts = [np.asarray(start, float)]
    for i in range(1, n):
        d = d + rng.normal(0, wiggle, 3)
        d[2] -= droop / n
        d /= np.linalg.norm(d)
        pts.append(pts[-1] + d * length / (n - 1))
    return np.array(pts)


def build_species(name, lod, seed=1):
    """One tree model (z up, base at the origin) for a level of detail."""
    rng = np.random.default_rng(seed)
    seg_t, seg_b, keep = LODS[lod]
    H = MODEL_HEIGHT[name]
    R = MODEL_CROWN_R[name]
    mesh = Mesh()
    tips = []      # (point, crown centre, card size)
    if name.startswith("pine"):
        # Aleppo pine: leaning, slightly sinuous bare trunk, branches in the top
        # third forking into flat needle clouds
        lean = np.array([rng.normal(0, 0.12), rng.normal(0, 0.12), 1.0])
        top = 0.62 if name == "pine_a" else 0.55
        trunk = _branch_path([0, 0, -0.3], lean, H * top + 0.3, rng, droop=-0.05, wiggle=0.05, n=7)
        tube(mesh, trunk, np.linspace(0.11 * H / 14 * 2.2, 0.08 * H / 14 * 2.2 * 0.5, len(trunk)), seg_t)
        crown_c = trunk[-1] + np.array([0, 0, H * (1 - top) * 0.45])
        n_main = 7 if name == "pine_a" else 9
        for i in range(n_main):
            f = i / n_main
            start = trunk[int(4 + f * (len(trunk) - 5))]
            az = 2 * math.pi * (i * 0.382 + rng.uniform(-0.05, 0.05))
            up = 0.35 + 0.45 * f if name == "pine_a" else 0.15 + 0.3 * f
            d = np.array([math.cos(az), math.sin(az), up])
            L = R * rng.uniform(0.75, 1.05) * (1.0 - 0.3 * f)
            br = _branch_path(start, d, L, rng, droop=0.0, wiggle=0.1)
            tube(mesh, br, np.linspace(0.09, 0.02, len(br)), seg_b)
            for j in (2, 3, 4):
                for _ in range(5):
                    tips.append((br[j] + rng.normal(0, 0.45, 3) * [1, 1, 0.35], crown_c, rng.uniform(1.1, 1.8)))
        for _ in range(24):   # top cloud, flat (umbrella-like crowns)
            p = crown_c + rng.normal(0, 1, 3) * [R * 0.45, R * 0.45, 0.45]
            tips.append((p, crown_c, rng.uniform(1.6, 2.4)))
        cell = 0 if name == "pine_a" else 1
    elif name == "oak":
        # holm oak: short trunk splitting low, dense round crown
        trunk = _branch_path([0, 0, -0.3], [rng.normal(0, 0.05), rng.normal(0, 0.05), 1], 0.35 * H, rng, droop=0, wiggle=0.06, n=5)
        tube(mesh, trunk, np.linspace(0.26, 0.18, len(trunk)), seg_t)
        crown_c = np.array([0, 0, 0.62 * H])
        for i in range(6):
            az = 2 * math.pi * (i / 6 + rng.uniform(-0.05, 0.05))
            d = np.array([math.cos(az), math.sin(az), 1.3])
            br = _branch_path(trunk[-1], d, R * 1.1, rng, droop=0.25, wiggle=0.12)
            tube(mesh, br, np.linspace(0.13, 0.03, len(br)), seg_b)
            for j in (2, 3, 4):
                for _ in range(6):
                    tips.append((br[j] + rng.normal(0, 0.5, 3), crown_c, rng.uniform(1.1, 1.7)))
        for _ in range(70):   # fill the round crown
            d = rng.normal(0, 1, 3)
            d /= np.linalg.norm(d)
            p = crown_c + d * rng.uniform(0.5, 1.0) * [R, R, 0.55 * H * 0.5]
            tips.append((p, crown_c, rng.uniform(1.4, 2.1)))
        cell = 2
    else:
        # small tree / tall shrub (kermes oak, juniper…): several stems, crown to the ground
        crown_c = np.array([0, 0, 0.5 * H])
        for i in range(4):
            az = 2 * math.pi * i / 4 + rng.uniform(-0.3, 0.3)
            st = _branch_path([0, 0, -0.2], [0.35 * math.cos(az), 0.35 * math.sin(az), 1], 0.8 * H, rng, droop=0, wiggle=0.1, n=4)
            tube(mesh, st, np.linspace(0.07, 0.02, len(st)), seg_b)
        for _ in range(70):
            d = rng.normal(0, 1, 3)
            d /= np.linalg.norm(d)
            p = crown_c + d * rng.uniform(0.3, 1.0) * [R, R, 0.5 * H]
            p[2] = max(p[2], 0.35)
            tips.append((p, crown_c, rng.uniform(1.0, 1.6)))
        cell = 3
    # leaf cards: a share of them for the lower LODs, made bigger to keep the volume
    order = rng.permutation(len(tips))
    n_keep = max(4, int(round(len(tips) * keep)))
    grow = math.sqrt(len(tips) / n_keep)
    for i in order[:n_keep]:
        p, cc, size = tips[i]
        for _ in range(4 if lod == "A" else 2):
            card(mesh, p + rng.normal(0, 0.25, 3), size * grow, cc, rng, cell)
    return mesh


def write_obj(path, mesh, mtl_name):
    lines = [f"mtllib {mtl_name}"]
    lines += [f"v {x:.4f} {y:.4f} {z:.4f}" for x, y, z in mesh.v]
    lines += [f"vt {u:.5f} {v:.5f}" for u, v in mesh.uv]
    lines += [f"vn {x:.4f} {y:.4f} {z:.4f}" for x, y, z in mesh.n]
    for mat, faces in mesh.f.items():
        lines.append(f"usemtl {mat}")
        lines += ["f " + " ".join(f"{i + 1}/{i + 1}/{i + 1}" for i in tri) for tri in faces]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


# ------------------------------------------------------------- textures
def _normal_from_height(hgt, strength=2.0):
    gy, gx = np.gradient(hgt)
    n = np.dstack([-gx * strength, -gy * strength, np.ones_like(hgt)])
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    return ((n * 0.5 + 0.5) * 255).astype(np.uint8)


def bark_textures(out, seed=4, n=512):
    """Pine / oak bark: vertical plates and cracks (diffuse + normal)."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, n, endpoint=False)
    plates = np.zeros((n, n))
    for _ in range(40):
        cx, cy = rng.uniform(0, 1, 2)
        w, h = rng.uniform(0.04, 0.09), rng.uniform(0.08, 0.2)
        dx = np.minimum(np.abs(x[None, :] - cx), 1 - np.abs(x[None, :] - cx)) / w
        dy = np.minimum(np.abs(x[:, None] - cy), 1 - np.abs(x[:, None] - cy)) / h
        plates = np.maximum(plates, np.clip(1 - (dx ** 2 + dy ** 2), 0, 1))
    noise = np.asarray(Image.fromarray((rng.random((n // 8, n // 8)) * 255).astype(np.uint8))
                       .resize((n, n), Image.BICUBIC), np.float32) / 255
    hgt = 0.7 * np.sqrt(plates) + 0.3 * noise
    col = np.array([92, 74, 60])[None, None] * (0.55 + 0.6 * hgt[..., None])
    Image.fromarray(np.clip(col, 0, 255).astype(np.uint8)).save(out / "bark.png")
    Image.fromarray(_normal_from_height(hgt * 6)).save(out / "bark_normal.png")


def leaves_textures(out, palette, seed=9):
    """Leaf-cluster atlas, 4 cells of 512 x 512: pine needles A / B (tufts of
    thin strokes), holm oak (small round leaves), shrub. RGBA diffuse
    (alpha = cut-out) and a normal map that makes each tuft rounded."""
    from textures import foliage_palette

    pal = foliage_palette(palette)
    rng = np.random.default_rng(seed)
    W = 512
    atlas = np.zeros((W, 4 * W, 4), np.uint8)
    hgt_all = np.zeros((W, 4 * W), np.float32)
    for cell, kind in enumerate(("pine", "pine", "broadleaf", "shrub")):
        img = Image.new("RGBA", (W, W), (0, 0, 0, 0))
        hgt = Image.new("L", (W, W), 0)
        d, dh = ImageDraw.Draw(img), ImageDraw.Draw(hgt)
        base = np.array(pal["pine" if kind == "pine" else kind], float)
        n_tufts = 26 if kind == "pine" else 60
        for _ in range(n_tufts):
            r = W * 0.45 * math.sqrt(rng.uniform(0, 1))
            a = rng.uniform(0, 2 * math.pi)
            cx, cy = W / 2 + r * math.cos(a), W / 2 + r * math.sin(a) * 0.85
            if kind == "pine":
                for _ in range(70):     # needles radiating from the tuft centre
                    b = rng.uniform(0, 2 * math.pi)
                    L = rng.uniform(18, 46)
                    c = base * rng.uniform(0.7, 1.3)
                    x1, y1 = cx + L * math.cos(b), cy + L * math.sin(b)
                    d.line([(cx, cy), (x1, y1)], fill=tuple(int(v) for v in np.clip(c, 0, 255)) + (255,), width=2)
                    dh.line([(cx, cy), (x1, y1)], fill=int(rng.uniform(120, 255)), width=2)
            else:
                k = 9 if kind == "broadleaf" else 7
                for _ in range(22):
                    ox, oy = rng.normal(0, 16, 2)
                    c = base * rng.uniform(0.7, 1.3)
                    e = [cx + ox - k, cy + oy - k * 0.7, cx + ox + k, cy + oy + k * 0.7]
                    d.ellipse(e, fill=tuple(int(v) for v in np.clip(c, 0, 255)) + (255,))
                    dh.ellipse(e, fill=int(rng.uniform(140, 255)))
        a_ = np.asarray(img).copy()
        a_[..., 3] = np.where(a_[..., 3] > 0, 255, 0)
        atlas[:, cell * W:(cell + 1) * W] = a_
        hgt_all[:, cell * W:(cell + 1) * W] = np.asarray(hgt.filter(ImageFilter.GaussianBlur(2)), np.float32) / 255
    from textures import bleed

    rgba = bleed(atlas.astype(np.float32) / 255.0)
    Image.fromarray((np.clip(rgba, 0, 1) * 255).astype(np.uint8), "RGBA").save(out / "leaves.png")
    Image.fromarray(_normal_from_height(hgt_all * 4)).save(out / "leaves_normal.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="config.yaml")
    ap.add_argument("--out", help="output folder (default data/<name>/csp_trees/src)")
    a = ap.parse_args()
    palette = {"pine": [62, 78, 52], "broadleaf": [88, 100, 66], "shrub": [104, 104, 74]}
    out = Path(a.out) if a.out else None
    try:
        from fetch import load_cfg

        cfg = load_cfg(a.config)
        pf = cfg["_dir"] / "trees_palette.json"
        if pf.exists():
            palette = json.loads(pf.read_text())
        out = out or cfg["_dir"] / "csp_trees" / "src"
    except Exception as e:     # standalone use
        print(f"(config not used: {e})")
        out = out or Path("csp_trees_src")
    out.mkdir(parents=True, exist_ok=True)
    bark_textures(out)
    leaves_textures(out, palette)
    (out / "trees.mtl").write_text(
        "newmtl bark\nKd 1 1 1\nmap_Kd bark.png\nmap_Bump bark_normal.png\n\n"
        "newmtl leaves\nKd 1 1 1\nmap_Kd leaves.png\nmap_d leaves.png\nmap_Bump leaves_normal.png\n",
        encoding="utf-8")
    for i, sp in enumerate(SPECIES):
        for lod in LODS:
            m = build_species(sp, lod, seed=11 + i)
            write_obj(out / f"{sp}_LOD_{lod}.obj", m, "trees.mtl")
            print(f"  {sp} LOD {lod}: {m.tris():,} triangles")
    meta = {"model_height_m": MODEL_HEIGHT, "model_crown_r_m": MODEL_CROWN_R}
    (out / "models.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"→ {out}\nNext: blender -b -P blender_trees.py -- \"{out}\"")


if __name__ == "__main__":
    main()
