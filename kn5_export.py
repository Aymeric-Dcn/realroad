#!/usr/bin/env python3
"""
Pipeline stage 4: write the Assetto Corsa track directly — no Blender, no ksEditor.

    python kn5_export.py                      → data/<name>/track_build/ac_track/<id>/
    python kn5_export.py --install "C:\\Program Files (x86)\\Steam\\steamapps\\common\\assettocorsa"

Reads track_build/ (meshes.npz, manifest.json, texture/) made by build_track.py
and writes a complete track folder:

  <id>.kn5            geometry, materials and textures (DXT1 DDS with mipmaps)
  ai/fast_lane.ai     AI spline along the road centre (also used for replays/minimap)
  data/map.ini, map.png, data/cameras.ini
  ui/ui_track.json, ui/outline.png, ui/preview.png

Only needs numpy and Pillow (scipy is used when available, for speed).

The KN5 layout follows AcTools (github.com/gro-ove/actools, Kn5Writer/Kn5Basic)
and was checked against Kunos tracks: version 6, textures, materials, then a
node tree. Coordinates: KN5 is Y-up, (x, y, z)_kn5 = (east, up, -north) from our
local (east, north, up) frame — the same conversion the FBX route used.
"""
import argparse
import io
import json
import math
import shutil
import struct
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

KN5_MAX_VERTS = 65_535

MATERIALS = {
    # ksPerPixel values in the range Kunos uses for grass/asphalt. The orthophoto
    # already contains light and shade, so mostly ambient and no specular:
    # ksEditor's defaults (ambient 0, diffuse 0.8, specular 0.8) washed it out.
    # First drive of v0.2: terrain at 0.45/0.35 still looked overexposed next to the
    # distant relief (the photo is already "lit"), so lower.
    "terrain": {"ksAmbient": 0.28, "ksDiffuse": 0.22, "ksSpecular": 0.0, "ksSpecularEXP": 1.0},
    "road": {"ksAmbient": 0.30, "ksDiffuse": 0.35, "ksSpecular": 0.12, "ksSpecularEXP": 12.0},
    "verge": {"ksAmbient": 0.40, "ksDiffuse": 0.40, "ksSpecular": 0.0, "ksSpecularEXP": 1.0},
    "marking": {"ksAmbient": 0.40, "ksDiffuse": 0.45, "ksSpecular": 0.15, "ksSpecularEXP": 20.0},
    # Kunos tree materials: shader ksTree, alpha tested, mostly ambient (0.35-0.55 / 0.1-0.2);
    # scaled down like the terrain
    "trees": {"ksAmbient": 0.38, "ksDiffuse": 0.18, "ksSpecular": 0.0, "ksSpecularEXP": 1.0, "ksAlphaRef": 0.3},
    "facade": {"ksAmbient": 0.30, "ksDiffuse": 0.30, "ksSpecular": 0.05, "ksSpecularEXP": 10.0},
    # advanced shaders (track.shaders: advanced), parameters as on Kunos tracks
    # (black_cat_county), brightness matched to the basic look above
    "road_adv": {"ksAmbient": 0.37, "ksDiffuse": 0.44, "ksSpecular": 0.6, "ksSpecularEXP": 20.0,
                 "fresnelC": 0.015, "fresnelEXP": 15.0, "fresnelMaxLevel": 0.15, "isAdditive": 0.0,
                 "useDetail": 1.0, "detailUVMultiplier": 12.0, "shadowBiasMult": 0.0, "detailNormalBlend": 0.75},
    "rock": {"ksAmbient": 0.30, "ksDiffuse": 0.28, "ksSpecular": 0.08, "ksSpecularEXP": 6.0,
             "fresnelC": 0.0, "fresnelEXP": 0.0, "nmObjectSpace": 0.0, "isAdditive": 0.0, "fresnelMaxLevel": 0.0},
    # multR/G/B: detail repeats per metre (rock 16 m, grass 4 m, gravel 5 m);
    # magicMult ≈ 1 / mean of the detail textures (0.62) so the photo keeps its brightness
    "terrain_ml": {"ksAmbient": 0.28, "ksDiffuse": 0.22, "ksSpecular": 0.04, "ksSpecularEXP": 10.0,
                   "multR": 0.0625, "multG": 0.25, "multB": 0.2, "multA": 0.0, "magicMult": 1.6,
                   "fresnelC": 0.0, "fresnelEXP": 3.0, "fresnelMaxLevel": 0.0,
                   "tarmacSpecularMultiplier": 0.0, "detailNMMult": 0.0},
    # road furniture: galvanised steel (guard rails, poles), road signs
    "metal": {"ksAmbient": 0.32, "ksDiffuse": 0.40, "ksSpecular": 0.35, "ksSpecularEXP": 25.0},
    "sign": {"ksAmbient": 0.42, "ksDiffuse": 0.42, "ksSpecular": 0.10, "ksSpecularEXP": 12.0},
}
DEFAULTS = {
    "ortho_texture_px": 2048,    # orthophoto tiles are resampled to this (power of two)
    "ortho_far_texture_px": 1024,  # tiles only used by the far terrain ring
    "max_texture_px": 2048,      # other textures
    "ai_step_m": 1.5,            # spacing of the AI spline points (Kunos tracks: 1.4-1.5 m)
    "ai_lat_accel": 7.0,         # m/s², corner speed hint for the AI
    "ai_brake": 7.0,
    "ai_accel": 3.5,
    "ai_max_speed": 40.0,        # m/s
    "camera_every_m": 350.0,
    "map_margin_m": 20.0,
    "map_max_px": 1600,
    "csp_config": True,          # extension/ext_config.ini for Custom Shaders Patch
}


def csp_trees_list(trees):
    """CSP tree list: one line per tree, track coordinates, scaled to the
    LiDAR height and crown width."""
    lines = ["configure: color variance = 0.06, 0.06, 0.06",
             "configure: brightness variance = 0.9, 1.1"]
    for sp, x, y, z, ang, size, width in trees:
        lines.append(f"tree: {sp}.bin; pos={x}, {y}, {z}; angle={ang:.0f}; size={size}; width={width}")
    return "\n".join(lines) + "\n"


def csp_ext_config(materials, trees_list=None):
    """Custom Shaders Patch settings for the track (ignored without CSP):
    3D grass on the green parts of the verges and terrain, wet road in the
    rain. CSP material filters: `?` matches any end of the name."""
    has = lambda prefix: any(m.startswith(prefix) for m in materials)
    grass = [f for f, ok in (("roadside_?", has("roadside_")), ("ortho_?", has("ortho_")),
                             ("verge", "verge" in materials)) if ok]
    occl = [m for m in ("road", "marking") if m in materials]
    lines = [
        "; Generated by realroad (kn5_export.py) for Custom Shaders Patch.",
        "; Delete a section to switch that effect off.",
        "",
        "[GRASS_FX]",
        f"GRASS_MATERIALS = {', '.join(grass)}",
        f"OCCLUDING_MATERIALS = {', '.join(occl)}",
        "ORIGINAL_GRASS_MATERIALS =",
        "; the orthophoto greens are dull and dry (Provence): low saturation threshold,",
        "; yellowish tones allowed, bright rock and gravel excluded",
        "MASK_MAIN_THRESHOLD = 0.08",
        "MASK_RED_THRESHOLD = 0.35",
        "MASK_MIN_LUMINANCE = 0.03",
        "MASK_MAX_LUMINANCE = 0.42",
        "MASK_BLUR = 1",
        "SHAPE_SIZE = 0.9",
        "SHAPE_TIDY = 0.0",
        "SHAPE_CUT = 0.0",
        "SHAPE_WIDTH = 1.1",
        "",
        "[RAIN_FX]",
        f"PUDDLES_MATERIALS = {'road' if 'road' in materials else ''}",
        f"SOAKING_MATERIALS = {', '.join(occl)}",
        f"ROUGH_MATERIALS = {', '.join(g for g in grass)}",
        f"LINES_MATERIALS = {'marking' if 'marking' in materials else ''}",
        "",
    ]
    if trees_list:
        lines += [
            "[TREES]",
            f"LIST_0 = {trees_list}",
            f"SURFACE_MATERIALS = {', '.join(grass)}",
            "",
        ]
    return "\n".join(lines)


# --------------------------------------------------------------- DDS (DXT1)
def _pack565(c):
    c = np.clip(np.rint(c), 0, 255).astype(np.int32)
    return ((c[:, 0] * 31 + 127) // 255 << 11) | ((c[:, 1] * 63 + 127) // 255 << 5) | ((c[:, 2] * 31 + 127) // 255)


def _unpack565(v):
    r, g, b = (v >> 11) & 31, (v >> 5) & 63, v & 31
    return np.column_stack([(r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2)]).astype(np.float32)


def dxt1_blocks(rgb):
    """Encode an (H, W, 3) uint8 image (H, W multiples of 4) to DXT1 blocks:
    endpoints on the principal colour axis of each 4x4 block, 4-colour mode."""
    h, w, _ = rgb.shape
    b = rgb.reshape(h // 4, 4, w // 4, 4, 3).transpose(0, 2, 1, 3, 4).reshape(-1, 16, 3).astype(np.float32)
    mean = b.mean(1)
    c = b - mean[:, None]
    cov = np.einsum("bki,bkj->bij", c, c)
    v = np.ones((len(b), 3), np.float32)
    for _ in range(6):  # power iteration → principal axis
        v = np.einsum("bij,bj->bi", cov, v)
        n = np.linalg.norm(v, axis=1, keepdims=True)
        v = np.where(n > 1e-6, v / np.maximum(n, 1e-12), 0.57735)
    t = np.einsum("bki,bi->bk", c, v)
    hi = mean + t.max(1)[:, None] * v
    lo = mean + t.min(1)[:, None] * v
    c0, c1 = _pack565(hi), _pack565(lo)
    swap = c0 < c1
    c0, c1 = np.where(swap, c1, c0), np.where(swap, c0, c1)
    e0, e1 = _unpack565(c0), _unpack565(c1)
    pal = np.stack([e0, e1, (2 * e0 + e1) / 3, (e0 + 2 * e1) / 3], 1)  # (n, 4, 3)
    d = ((b[:, :, None, :] - pal[:, None, :, :]) ** 2).sum(-1)       # (n, 16, 4)
    idx = d.argmin(-1).astype(np.uint32)
    idx[c0 == c1] = 0  # 3-colour mode would make index 3 black
    bits = (idx << (2 * np.arange(16, dtype=np.uint32))).sum(1, dtype=np.uint64).astype(np.uint32)
    out = np.empty(len(b), dtype=[("c0", "<u2"), ("c1", "<u2"), ("i", "<u4")])
    out["c0"], out["c1"], out["i"] = c0, c1, bits
    return out.tobytes()


def dds_dxt1(img, size=None):
    """PIL image → DXT1 .dds bytes with a full mipmap chain (power-of-two size)."""
    img = img.convert("RGB")
    if size is None:
        size = tuple(min(4096, max(4, 1 << round(math.log2(max(1, s))))) for s in img.size)
    if img.size != size:
        img = img.resize(size, Image.LANCZOS)
    w, h = size
    levels, data = 0, []
    cur = img
    while True:
        cw, ch = cur.size
        a = np.asarray(cur if (cw >= 4 and ch >= 4) else cur.resize((max(4, cw), max(4, ch)), Image.NEAREST))
        data.append(dxt1_blocks(np.ascontiguousarray(a)))
        levels += 1
        if cw == 1 and ch == 1:
            break
        cur = cur.resize((max(1, cw // 2), max(1, ch // 2)), Image.BOX)
    return _dds_header(w, h, len(data[0]), levels) + b"".join(data)


def _alpha_blocks(a):
    """DXT5 alpha blocks of an (H, W) uint8 alpha (H, W multiples of 4):
    min/max endpoints, 8-level mode."""
    h, w = a.shape
    b = a.reshape(h // 4, 4, w // 4, 4).transpose(0, 2, 1, 3).reshape(-1, 16).astype(np.float32)
    a0, a1 = b.max(1), b.min(1)
    span = np.maximum(a0 - a1, 1e-6)
    # palette of the a0 > a1 mode: index 0 = a0, 1 = a1, 2..7 interpolated
    t = np.rint((a0[:, None] - b) / span[:, None] * 7).astype(np.int32)       # 0 = a0 … 7 = a1
    code = np.choose(np.clip(t, 0, 7), [0, 2, 3, 4, 5, 6, 7, 1]).astype(np.uint64)
    code[a0 == a1] = 0
    bits = (code << (3 * np.arange(16, dtype=np.uint64))).sum(1, dtype=np.uint64)
    out = np.zeros((len(b), 8), np.uint8)
    out[:, 0] = a0.astype(np.uint8)
    out[:, 1] = a1.astype(np.uint8)
    for k in range(6):
        out[:, 2 + k] = ((bits >> np.uint64(8 * k)) & np.uint64(255)).astype(np.uint8)
    return out


def dds_dxt5(img, size=None):
    """PIL image → DXT5 .dds (colour + interpolated alpha) with mipmaps: for
    textures whose alpha the shader reads (detail blend, specular, masks)."""
    img = img.convert("RGBA")
    if size is None:
        size = tuple(min(4096, max(4, 1 << round(math.log2(max(1, s))))) for s in img.size)
    if img.size != size:
        img = img.resize(size, Image.LANCZOS)
    w, h = size
    levels, data = 0, []
    cur = img
    while True:
        cw, ch = cur.size
        a = np.asarray(cur if (cw >= 4 and ch >= 4) else cur.resize((max(4, cw), max(4, ch)), Image.NEAREST))
        col = np.frombuffer(dxt1_blocks(np.ascontiguousarray(a[..., :3])), np.uint8).reshape(-1, 8)
        alp = _alpha_blocks(np.ascontiguousarray(a[..., 3]))
        data.append(np.concatenate([alp, col], axis=1).tobytes())
        levels += 1
        if cw == 1 and ch == 1:
            break
        cur = cur.resize((max(1, cw // 2), max(1, ch // 2)), Image.BOX)
    hdr = struct.pack("<4sIIIIIII", b"DDS ", 124, 0xA1007, h, w, len(data[0]), 0, levels)
    hdr += b"\0" * 44
    hdr += struct.pack("<II4s5I", 32, 4, b"DXT5", 0, 0, 0, 0, 0)
    hdr += struct.pack("<5I", 0x401008, 0, 0, 0, 0)
    return hdr + b"".join(data)


def dds_rgba(img, size, alpha_ref=0.3):
    """PIL image → uncompressed 32-bit BGRA .dds with mipmaps. The alpha of each
    mip level is rescaled so that the share of pixels above alpha_ref stays the
    same: alpha-tested foliage does not thin out with distance."""
    img = img.convert("RGBA")
    if img.size != size:
        img = img.resize(size, Image.LANCZOS)
    base = np.asarray(img, np.float32) / 255.0
    cover = float((base[..., 3] > alpha_ref).mean())
    w, h = size
    data, cur, levels = [], img, 0
    while True:
        a = np.asarray(cur, np.float32) / 255.0
        if levels > 0 and cover > 0:
            lo, hi = 0.5, 6.0
            for _ in range(20):
                mid = (lo + hi) / 2
                if (np.clip(a[..., 3] * mid, 0, 1) > alpha_ref).mean() < cover:
                    lo = mid
                else:
                    hi = mid
            a[..., 3] = np.clip(a[..., 3] * (lo + hi) / 2, 0, 1)
        # sharpen around the cut: opaque texels well above any alpha reference
        # (0.75+), clear ones well below (0.1-), so the coverage is the same
        # whatever threshold the shader really uses (ksTree may ignore ksAlphaRef)
        al = a[..., 3]
        a[..., 3] = np.where(al > alpha_ref, 0.75 + 0.25 * (al - alpha_ref) / (1 - alpha_ref), 0.1 * al / alpha_ref)
        bgra = (a[..., [2, 1, 0, 3]] * 255 + 0.5).astype(np.uint8)
        data.append(bgra.tobytes())
        levels += 1
        cw, ch = cur.size
        if cw == 1 and ch == 1:
            break
        cur = cur.resize((max(1, cw // 2), max(1, ch // 2)), Image.BOX)
    hdr = struct.pack("<4sIIIIIII", b"DDS ", 124, 0x2100F, h, w, w * 4, 0, levels)
    hdr += b"\0" * 44
    hdr += struct.pack("<II4s5I", 32, 0x41, b"\0\0\0\0", 32, 0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000)
    hdr += struct.pack("<5I", 0x401008, 0, 0, 0, 0)
    assert len(hdr) == 128
    return hdr + b"".join(data)


def _cached_rgba(dds_cache, src, size):
    cached = dds_cache / f"{src.stem}_{src.stat().st_size}_{size[0]}x{size[1]}_rgba2.dds"
    if cached.exists():
        return cached.read_bytes()
    dds = dds_rgba(Image.open(src), size)
    part = cached.with_suffix(".part")
    part.write_bytes(dds)
    part.replace(cached)
    return dds


def _dds_header(w, h, top_size, levels):
    hdr = struct.pack("<4sIIIIIII", b"DDS ", 124, 0xA1007, h, w, top_size, 0, levels)
    hdr += b"\0" * 44                                     # reserved1[11]
    hdr += struct.pack("<II4s5I", 32, 4, b"DXT1", 0, 0, 0, 0, 0)  # pixel format
    hdr += struct.pack("<5I", 0x401008, 0, 0, 0, 0)        # caps, caps2-4, reserved2
    assert len(hdr) == 128
    return hdr


# ---------------------------------------------------------------- KN5 writer
class Kn5Writer:
    def __init__(self, f):
        self.f = f

    def raw(self, b):
        self.f.write(b)

    def i32(self, v):
        self.f.write(struct.pack("<i", v))

    def u32(self, v):
        self.f.write(struct.pack("<I", v))

    def u8(self, v):
        self.f.write(struct.pack("<B", v))

    def f32(self, *v):
        self.f.write(struct.pack("<%df" % len(v), *v))

    def s(self, text):
        b = text.encode("utf-8")
        self.i32(len(b))
        self.f.write(b)

    def header(self):
        self.raw(b"sc6969")
        self.i32(6)
        self.i32(0)

    def texture(self, name, data):
        self.i32(1)
        self.s(name)
        self.u32(len(data))
        self.raw(data)

    def material(self, name, props, textures, shader="ksPerPixel", alpha_tested=False):
        """textures: kn5 texture name (txDiffuse) or [(slot name, slot index, kn5 texture name)]."""
        if isinstance(textures, str):
            textures = [("txDiffuse", 0, textures)]
        self.s(name)
        self.s(shader)
        self.u8(0)    # blend mode: opaque
        self.u8(1 if alpha_tested else 0)
        self.i32(0)   # depth mode: normal
        values = {"ksAmbient": 0.4, "ksDiffuse": 0.4, "ksSpecular": 0.0, "ksSpecularEXP": 1.0,
                  "ksEmissive": 0.0, "ksAlphaRef": 0.0, **props}
        self.i32(len(values))
        for k, v in values.items():
            self.s(k)
            self.f32(v if k != "ksEmissive" else 0.0)   # value A
            self.f32(0.0, 0.0)                           # value B
            self.f32(*((v, v, v) if k == "ksEmissive" else (0.0, 0.0, 0.0)))  # value C
            self.f32(0.0, 0.0, 0.0, 0.0)                 # value D
        self.i32(len(textures))
        for slot, idx, tex in textures:
            self.s(slot)
            self.i32(idx)
            self.s(tex)

    def dummy(self, name, matrix, n_children):
        self.i32(1)
        self.s(name)
        self.i32(n_children)
        self.u8(1)
        self.f32(*np.asarray(matrix, np.float32).ravel())

    def mesh(self, name, vertices, indices, material_id, centre, radius, lod_out=0.0, hidden=False,
             shadows=True):
        """hidden: physics-only mesh (collision walls), not drawn.
        lod_out: not drawn beyond this distance in metres (0 = always)."""
        self.i32(2)
        self.s(name)
        self.i32(0)
        self.u8(1)                           # active
        self.u8(int(shadows and not hidden)); self.u8(int(not hidden)); self.u8(0)  # cast shadows, visible, transparent
        self.u32(len(vertices))
        self.raw(np.ascontiguousarray(vertices, np.float32).tobytes())
        self.u32(indices.size)
        self.raw(np.ascontiguousarray(indices, "<u2").tobytes())
        self.u32(material_id)
        self.u32(0)                          # layer
        self.f32(0.0, float(lod_out))        # LOD in/out (0 = always)
        self.f32(*centre)
        self.f32(radius)
        self.u8(int(not hidden))             # renderable


def to_kn5(v):
    """Local (east, north, up) → KN5 (east, up, -north)."""
    v = np.asarray(v, np.float64)
    return np.column_stack([v[:, 0], v[:, 2], -v[:, 1]])


def kn5_vertices(verts, faces, uvs, inset=None):
    """Vertex array (n, 11): position, normal, uv, tangent — KN5 layout."""
    p = to_kn5(verts)
    a, b, c = p[faces[:, 0]], p[faces[:, 1]], p[faces[:, 2]]
    fn = np.cross(b - a, c - a)                 # area-weighted, faces are CCW from above
    n = np.zeros_like(p)
    for k in range(3):
        np.add.at(n, faces[:, k], fn)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.where(ln > 1e-12, n / np.maximum(ln, 1e-12), [0.0, 1.0, 0.0])
    uv = np.asarray(uvs, np.float64).copy()
    if inset is not None:                        # keep bilinear filtering inside the tile
        w, h = inset
        uv[:, 0] = (uv[:, 0] * (w - 1) + 0.5) / w
        uv[:, 1] = (uv[:, 1] * (h - 1) + 0.5) / h
    uv[:, 1] = 1.0 - uv[:, 1]                    # DirectX: v goes down
    # tangent = dP/du
    du1 = uv[faces[:, 1], 0] - uv[faces[:, 0], 0]
    du2 = uv[faces[:, 2], 0] - uv[faces[:, 0], 0]
    dv1 = uv[faces[:, 1], 1] - uv[faces[:, 0], 1]
    dv2 = uv[faces[:, 2], 1] - uv[faces[:, 0], 1]
    r = du1 * dv2 - du2 * dv1
    r = np.where(np.abs(r) > 1e-12, r, 1e-12)
    ft = ((b - a) * dv2[:, None] - (c - a) * dv1[:, None]) / r[:, None]
    t = np.zeros_like(p)
    for k in range(3):
        np.add.at(t, faces[:, k], ft)
    t -= n * np.sum(t * n, axis=1, keepdims=True)
    lt = np.linalg.norm(t, axis=1, keepdims=True)
    fallback = np.cross(n, [0.0, 0.0, 1.0])
    t = np.where(lt > 1e-9, t / np.maximum(lt, 1e-12), fallback / np.maximum(np.linalg.norm(fallback, axis=1, keepdims=True), 1e-12))
    out = np.column_stack([p, n, uv, t]).astype(np.float32)
    lo, hi = p.min(0), p.max(0)
    centre = (lo + hi) / 2
    radius = float(np.linalg.norm(p - centre, axis=1).max())
    return out, centre, radius


def dummy_matrix(pos_local, fwd_local, flip=False):
    """Row-major DirectX matrix (rows = local X, Y, Z axes, then translation).
    AC puts cars along the dummy's +Z; X = Y × Z (checked on ksEditor output)."""
    f = to_kn5(np.array([[fwd_local[0], fwd_local[1], 0.0]]))[0]
    f[1] = 0.0
    f /= np.linalg.norm(f)
    if flip:
        f = -f
    y = np.array([0.0, 1.0, 0.0])
    x = np.cross(y, f)
    p = to_kn5(np.array([pos_local]))[0]
    return np.array([[*x, 0], [*y, 0], [*f, 0], [*p, 1]], np.float64)


# ---------------------------------------------------------- road centreline
def road_line(man, data):
    """Road edges and centre (local frame) from the road meshes: rows along the
    road, columns from the right edge to the left edge."""
    rows_r, rows_l = [], []
    for o in man["objects"]:
        if o["material"] != "road":
            continue
        k = o["key"]
        v = data[f"v{k}"].astype(np.float64)
        u = data[f"uv{k}"][:, 0]
        n_cols = int(np.argmax(u[1:] <= u[0]) + 1) if (u[1:] <= u[0]).any() else len(u)
        g = v.reshape(-1, n_cols, 3)
        if rows_r:
            g = g[1:]  # chunks share their boundary row
        rows_r.append(g[:, 0])
        rows_l.append(g[:, -1])
    R, L = np.concatenate(rows_r), np.concatenate(rows_l)
    return R, L, (R + L) / 2


def gaussian_smooth(a, sigma):
    """Gaussian filter along axis 0 (samples ~1 m apart), edges held."""
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.concatenate([np.repeat(a[:1], r, 0), a, np.repeat(a[-1:], r, 0)])
    return np.column_stack([np.convolve(pad[:, i], k, mode="valid") for i in range(a.shape[1])])


def resample(points, step):
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    s = np.arange(0, d[-1] + 1e-6, step)
    if d[-1] - s[-1] > 0.3 * step:
        s = np.r_[s, d[-1]]
    return s, np.column_stack([np.interp(s, d, points[:, i]) for i in range(points.shape[1])])


# -------------------------------------------------------------- AI spline
def knn_cells(px, pz, cx, cz, k=10):
    """For every cell centre, the k nearest spline points (xz distance, nearest first)."""
    try:
        from scipy.spatial import cKDTree

        _, idx = cKDTree(np.column_stack([px, pz])).query(np.column_stack([cx, cz]), k=k)
        return idx.astype(np.int32)
    except ImportError:
        pass
    # numpy fallback: coarse nearest neighbours, then exact search in windows around them
    stride = 8
    cidx = np.arange(0, len(px), stride)
    out = np.empty((len(cx), k), np.int32)
    win = np.arange(-(stride + k + 4), stride + k + 5)
    for a in range(0, len(cx), 4000):
        qx, qz = cx[a:a + 4000, None], cz[a:a + 4000, None]
        dc = (px[cidx][None] - qx) ** 2 + (pz[cidx][None] - qz) ** 2
        top = np.argpartition(dc, min(8, len(cidx) - 1), axis=1)[:, :8]
        cand = (cidx[top][:, :, None] + win[None, None, :]).reshape(len(qx), -1)
        cand = np.clip(cand, 0, len(px) - 1)
        cand.sort(axis=1)
        d = (px[cand] - qx) ** 2 + (pz[cand] - qz) ** 2
        d[:, 1:][cand[:, 1:] == cand[:, :-1]] = np.inf  # duplicates
        order = np.argsort(d, axis=1, kind="stable")[:, :k]  # ties → lower index (cand is sorted)
        out[a:a + len(qx)] = np.take_along_axis(cand, order, 1)
    return out


def ai_spline(R, L, C, p):
    """fast_lane.ai, version 7 (layout from AcTools AiSpline; field meanings
    checked against Kunos files: forward/length from the previous point,
    grade = forward.y, camber = normal·right, radius from the heading change,
    grid = 10 nearest points to the centre of each 10 m cell, 350 m margin)."""
    # the mesh centre wobbles by a few cm where the measured width changes:
    # smooth it (σ 2 m) so heading, radius and grade are clean
    C = gaussian_smooth(C, 2.0)
    s_c = np.r_[0, np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))]
    s, Ck = resample(C, p["ai_step_m"])
    Rk = np.column_stack([np.interp(s, s_c, R[:, i]) for i in range(3)])
    Lk = np.column_stack([np.interp(s, s_c, L[:, i]) for i in range(3)])
    P, PR, PL = to_kn5(Ck), to_kn5(Rk), to_kn5(Lk)
    n = len(P)
    step = np.linalg.norm(np.diff(P, axis=0), axis=1)
    length = np.r_[0.0, step]                       # distance from the previous point
    cum = np.cumsum(length)
    fwd = np.diff(P, axis=0) / step[:, None]
    fwd = np.vstack([fwd[:1], fwd])                 # from the previous point
    lat = PL - PR
    normal = np.cross(fwd, lat)
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    right = np.cross(fwd, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right, axis=1, keepdims=True)
    side_l = np.linalg.norm((PL - P)[:, [0, 2]], axis=1)
    side_r = np.linalg.norm((P - PR)[:, [0, 2]], axis=1)
    # horizontal heading change over ±2 points → radius and turn direction
    h = fwd[:, [0, 2]] / np.linalg.norm(fwd[:, [0, 2]], axis=1, keepdims=True)
    k2 = 2
    hp, hn = h[np.clip(np.arange(n) - k2, 0, n - 1)], h[np.clip(np.arange(n) + k2, 0, n - 1)]
    cross = hp[:, 0] * hn[:, 1] - hp[:, 1] * hn[:, 0]
    ang = np.arccos(np.clip(np.sum(hp * hn, 1), -1, 1))
    span = cum[np.clip(np.arange(n) + k2, 0, n - 1)] - cum[np.clip(np.arange(n) - k2, 0, n - 1)]
    curv = np.convolve(ang / np.maximum(span, 1e-3), np.ones(3) / 3, mode="same")
    radius = np.clip(1.0 / np.maximum(curv, 1e-4), 1.0, 10000.0)
    direction = np.where(cross > 0, -1.0, 1.0)
    # speed hint: corner limit, then braking/acceleration passes
    v = np.minimum(np.sqrt(p["ai_lat_accel"] * radius), p["ai_max_speed"])
    for i in range(n - 2, -1, -1):
        v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2 * p["ai_brake"] * length[i + 1]))
    v[0] = min(v[0], 5.0)
    for i in range(1, n):
        v[i] = min(v[i], math.sqrt(v[i - 1] ** 2 + 2 * p["ai_accel"] * length[i]))
    nxt = np.r_[v[1:], v[-1]]
    dl = np.r_[length[1:], length[-1]]
    decel = (v ** 2 - nxt ** 2) / (2 * np.maximum(dl, 1e-3))
    brake = np.clip(decel / p["ai_brake"], 0, 1)
    gas = np.where(nxt >= v - 1e-3, 1.0, 0.0)

    buf = io.BytesIO()
    buf.write(struct.pack("<4i", 7, n, 0, 0))
    pts = np.zeros(n, dtype=[("p", "<f4", 3), ("len", "<f4"), ("id", "<i4")])
    pts["p"], pts["len"], pts["id"] = P, cum, np.arange(n)
    buf.write(pts.tobytes())
    buf.write(struct.pack("<i", n))
    extra = np.column_stack([
        v, gas, brake, np.zeros(n), radius, side_l, side_r, np.sum(normal * right, 1), direction,
        normal, length, fwd, np.zeros(n), fwd[:, 1],
    ]).astype("<f4")
    buf.write(extra.tobytes())
    # spatial grid
    dens, margin, k = 10.0, 350.0, 10
    mn = np.array([P[:, 0].min() - margin, 0.0, P[:, 2].min() - margin])
    mx = np.array([P[:, 0].max() + margin, 0.0, P[:, 2].max() + margin])
    nx, nz = int((mx[0] - mn[0]) / dens), int((mx[2] - mn[2]) / dens)
    ii, jj = np.meshgrid(np.arange(nx), np.arange(nz), indexing="ij")
    cx = mn[0] + (ii.ravel() + 0.5) * dens
    cz = mn[2] + (jj.ravel() + 0.5) * dens
    nn = knn_cells(P[:, 0], P[:, 2], cx, cz, k).reshape(nx, nz, k)
    buf.write(struct.pack("<i", 1))
    buf.write(struct.pack("<3f3fifi", *mx, *mn, k, dens, nx))
    row = np.empty((nz, 1 + k), "<i4")
    row[:, 0] = k
    for i in range(nx):
        buf.write(struct.pack("<i", nz))
        row[:, 1:] = nn[i]
        buf.write(row.tobytes())
    info = {"points": n, "length_m": float(cum[-1]), "grid": (nx, nz),
            "speed_kmh": (float(3.6 * v.min()), float(3.6 * np.median(v)), float(3.6 * v.max()))}
    return buf.getvalue(), P, cum, info


# -------------------------------------------------------- map, ui, cameras
def map_files(P, p):
    margin = p["map_margin_m"]
    minx, maxx = P[:, 0].min(), P[:, 0].max()
    minz, maxz = P[:, 2].min(), P[:, 2].max()
    ext = max(maxx - minx, maxz - minz) + 2 * margin
    scale = max(1.0, math.ceil(2 * ext / p["map_max_px"]) / 2)
    xo, zo = -minx + margin, -minz + margin
    w, h = (maxx - minx + 2 * margin) / scale, (maxz - minz + 2 * margin) / scale
    ini = ("[PARAMETERS]\nWIDTH=%.2f\nHEIGHT=%.2f\nMARGIN=%g\nSCALE_FACTOR=%g\nMAX_SIZE=%d\n"
           "X_OFFSET=%.2f\nZ_OFFSET=%.2f\nDRAWING_SIZE=10\n") % (w, h, margin, scale, p["map_max_px"], xo, zo)
    ss = 2
    img = Image.new("RGBA", (int(w) * ss, int(h) * ss), (0, 0, 0, 0))
    xy = [((x + xo) / scale * ss, (z + zo) / scale * ss) for x, z in P[::2, [0, 2]]]
    ImageDraw.Draw(img).line(xy, fill=(235, 235, 235, 255), width=10 * ss, joint="curve")
    return ini, img.resize((int(w), int(h)), Image.LANCZOS)


def outline_png(P, size=(365, 192), pad=12):
    x, z = P[:, 0], P[:, 2]
    sc = min((size[0] - 2 * pad) / np.ptp(x), (size[1] - 2 * pad) / np.ptp(z))
    ox = (size[0] - sc * np.ptp(x)) / 2 - sc * x.min()
    oz = (size[1] - sc * np.ptp(z)) / 2 - sc * z.min()
    ss = 3
    img = Image.new("RGBA", (size[0] * ss, size[1] * ss), (0, 0, 0, 0))
    xy = [((sc * a + ox) * ss, (sc * b + oz) * ss) for a, b in zip(x[::3], z[::3])]
    ImageDraw.Draw(img).line(xy, fill=(255, 255, 255, 255), width=3 * ss, joint="curve")
    return img.resize(size, Image.LANCZOS)


def preview_png(man, P, tex_dir, size=(355, 200)):
    """Orthophoto mosaic of the whole route, with the road drawn on top."""
    x0, y0, _ = man["origin_lambert93"]
    tiles = []
    for name, f in man["materials"].items():
        if name.startswith("ortho_"):
            _, ex, ny = name.split("_")
            tiles.append((float(ex) - x0, float(ny) - y0, tex_dir / (f["texture"] if isinstance(f, dict) else f)))
    if not tiles:
        return None
    ex, nz = P[:, 0], -P[:, 2]  # back to east/north
    pad = 150.0
    w_m = np.ptp(ex) + 2 * pad
    h_m = np.ptp(nz) + 2 * pad
    aspect = size[0] / size[1]
    if w_m / h_m < aspect:
        w_m = h_m * aspect
    else:
        h_m = w_m / aspect
    cx, cy = (ex.min() + ex.max()) / 2, (nz.min() + nz.max()) / 2
    left, top = cx - w_m / 2, cy + h_m / 2
    ss = 3
    W, H = size[0] * ss, size[1] * ss
    mpp = w_m / W
    canvas = Image.new("RGB", (W, H), (96, 104, 84))
    for tx, ty, f in tiles:  # tile name = lower-left corner, 400 m tiles
        if not Path(f).exists():
            continue
        im = Image.open(f).convert("RGB")
        tw = int(round(400 / mpp))
        if tw < 1:
            continue
        im = im.resize((tw, tw), Image.BILINEAR)
        canvas.paste(im, (int(round((tx - left) / mpp)), int(round((top - (ty + 400)) / mpp))))
    xy = [((a - left) / mpp, (top - b) / mpp) for a, b in zip(ex[::3], nz[::3])]
    ImageDraw.Draw(canvas).line(xy, fill=(255, 210, 40), width=2 * ss, joint="curve")
    return canvas.resize(size, Image.LANCZOS)


class HeightField:
    """Max terrain/road height per 5 m cell, from the mesh vertices (local frame)."""

    def __init__(self, man, data, cell=5.0):
        pts = np.concatenate([data[f"v{o['key']}"] for o in man["objects"]]).astype(np.float64)
        self.cell = cell
        self.x0, self.y0 = pts[:, 0].min(), pts[:, 1].min()
        i = ((pts[:, 0] - self.x0) // cell).astype(np.int64)
        j = ((pts[:, 1] - self.y0) // cell).astype(np.int64)
        self.nj = int(j.max()) + 1
        key = i * self.nj + j
        order = np.argsort(key)
        k, z = key[order], pts[order, 2]
        starts = np.r_[0, np.flatnonzero(np.diff(k)) + 1]
        self.keys, self.zmax = k[starts], np.maximum.reduceat(z, starts)

    def at(self, x, y):
        i = int((x - self.x0) // self.cell)
        j = int((y - self.y0) // self.cell)
        best = -np.inf
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                kk = (i + di) * self.nj + (j + dj)
                pos = np.searchsorted(self.keys, kk)
                if pos < len(self.keys) and self.keys[pos] == kk:
                    best = max(best, self.zmax[pos])
        return best


def cameras_ini(C, R, L, hf, every, title):
    """TV cameras beside the road, each covering the stretch until the next."""
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(C[:, :2], axis=0), axis=1))]
    total = d[-1]
    marks = np.arange(0, total, every)
    cams = []
    for n_, s0 in enumerate(marks):
        s_cam = min(s0 + 0.55 * every, total - 1)       # stand mid-way, facing back
        k = int(np.searchsorted(d, s_cam))
        side = R if n_ % 2 == 0 else L
        out = side[k] - C[k]
        out[2] = 0
        out /= max(np.linalg.norm(out), 1e-6)
        pos = side[k] + out * 6.0
        ground = hf.at(pos[0], pos[1])
        pos[2] = max(ground if np.isfinite(ground) else -1e9, C[k, 2]) + 3.0
        target = C[int(np.searchsorted(d, max(s0, s_cam - 0.4 * every)))] + [0, 0, 1.0]
        f = target - pos
        f /= np.linalg.norm(f)
        up = np.array([0.0, 0.0, 1.0]) - f * f[2]
        up /= np.linalg.norm(up)
        cams.append((to_kn5(pos[None])[0], to_kn5(f[None])[0], to_kn5(up[None])[0],
                     s0 / total, min(1.0, (s0 + every) / total)))
    out = ["[HEADER]", "VERSION=3", f"CAMERA_COUNT={len(cams)}", f"SET_NAME={title}", ""]
    for i, (pos, f, up, a, b) in enumerate(cams):
        out += [f"[CAMERA_{i}]", f"NAME={i + 1}",
                "POSITION=%.3f, %.3f, %.3f" % tuple(pos), "FORWARD=%.5f, %.5f, %.5f" % tuple(f),
                "UP=%.5f, %.5f, %.5f" % tuple(up), "MIN_FOV=10", "MAX_FOV=45",
                "IN_POINT=%.6f" % a, "OUT_POINT=%.6f" % b,
                "SHADOW_SPLIT0=10", "SHADOW_SPLIT1=40", "SHADOW_SPLIT2=180", "NEAR_PLANE=0.5",
                "FAR_PLANE=10000", "MIN_EXPOSURE=0.45", "MAX_EXPOSURE=0.55", "DOF_FACTOR=1",
                "DOF_RANGE=10000", "DOF_FOCUS=0", "DOF_MANUAL=0", "SPLINE=", "SPLINE_ROTATION=0",
                "FOV_GAMMA=1", "SPLINE_ANIMATION_LENGTH=15", "IS_FIXED=0", ""]
    return "\n".join(out), len(cams)


# ---------------------------------------------------------------------- main
def load_settings(config):
    p = dict(DEFAULTS)
    mats = {k: dict(v) for k, v in MATERIALS.items()}
    build = None
    try:
        from fetch import load_cfg

        cfg = load_cfg(config)
        t = cfg.get("track") or {}
        p.update({k: v for k, v in t.items() if k in DEFAULTS})
        for k, v in (t.get("materials") or {}).items():
            mats.setdefault(k, {}).update(v)
        build = cfg["_dir"] / "track_build"
    except (ImportError, FileNotFoundError, KeyError) as e:
        print(f"(config not used: {e})")
    return p, mats, build


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", default="config.yaml")
    ap.add_argument("--build", help="track_build folder (default: from config.yaml)")
    ap.add_argument("--id", help="track id to write (default: the one in config.yaml)")
    ap.add_argument("--install", metavar="AC_ROOT", help="also copy the track into <AC_ROOT>/content/tracks")
    ap.add_argument("--no-ai", action="store_true", help="skip ai/fast_lane.ai (and replay cameras)")
    ap.add_argument("--spawn-flip", action="store_true", help="cars spawn facing the other way")
    a = ap.parse_args()
    p, mats, build = load_settings(a.config)
    build = Path(a.build) if a.build else build
    if build is None or not (build / "manifest.json").exists():
        raise SystemExit("No track_build/manifest.json: run `python build_track.py` first.")
    man = json.loads((build / "manifest.json").read_text(encoding="utf-8"))
    data = np.load(build / "meshes.npz")
    tid = (a.id or man["track_id"]).lower()
    out = build / "ac_track" / tid
    out.mkdir(parents=True, exist_ok=True)
    for sub in ("ai", "data", "ui"):
        (out / sub).mkdir(exist_ok=True)
    tex_dir = build / "texture"
    dds_cache = build.parent / "texture_dds"  # outside track_build: survives rebuilds
    dds_cache.mkdir(exist_ok=True)

    # textures (one per material)
    mat_names = list(man["materials"])
    far_only = set(man.get("far_only_materials", []))
    kn5_path = out / f"{tid}.kn5"
    tmp = kn5_path.with_suffix(".kn5.part")
    with open(tmp, "wb") as f:
        w = Kn5Writer(f)
        w.header()
        tex_size = {}
        specs = {m: (v if isinstance(v, dict) else {"texture": v}) for m, v in man["materials"].items()}

        def encode(src, size, enc):
            """DDS bytes of a texture, cached (re-exports and interrupted runs are fast)."""
            tag = "" if enc == "dxt1" else f"_{enc}"            # DXT1 names as before: old caches stay valid
            cached = dds_cache / f"{src.stem}_{src.stat().st_size}_{size[0]}x{size[1]}{tag}.dds"
            if cached.exists():
                return cached.read_bytes()
            img = Image.open(src)
            dds = {"rgba": lambda: dds_rgba(img, size), "dxt5": lambda: dds_dxt5(img, size)}.get(
                enc, lambda: dds_dxt1(img, size))()
            part = cached.with_suffix(".part")
            part.write_bytes(dds)
            part.replace(cached)
            return dds

        def pow2(src, cap):
            with Image.open(src) as im:
                return tuple(min(cap, max(4, 1 << round(math.log2(s_)))) for s_ in im.size)

        # texture table: each material's diffuse (named after the material, as
        # before) + the extra maps of the advanced shaders (shared by file)
        tex_table = []           # (kn5 name, source, size, encoding)
        extra_names = {}
        for m in mat_names:
            spec = specs[m]
            src = tex_dir / spec["texture"]
            if m in far_only:
                size = (p["ortho_far_texture_px"],) * 2
            elif m.startswith("ortho_"):
                size = (p["ortho_texture_px"],) * 2
            else:
                size = pow2(src, p["max_texture_px"])
            enc = spec.get("dds") or ("rgba2" if spec.get("alpha") else "dxt1")
            enc = "rgba" if enc == "rgba2" else enc
            tex_size[m] = size
            tex_table.append((f"{m}.dds", src, size, enc))
            for slot, idx, fname, fenc in spec.get("maps", []):
                if fname not in extra_names:
                    fsrc = tex_dir / fname
                    extra_names[fname] = f"x_{Path(fname).stem}.dds"
                    tex_table.append((extra_names[fname], fsrc, pow2(fsrc, p["max_texture_px"]), fenc))
        w.i32(len(tex_table))
        for i, (tname, src, size, enc) in enumerate(tex_table):
            w.texture(tname, _cached_rgba(dds_cache, src, size) if enc == "rgba" else encode(src, size, enc))
            if (i + 1) % 10 == 0 or i + 1 == len(tex_table):
                print(f"  textures {i + 1}/{len(tex_table)}", flush=True)
        w.i32(len(mat_names))
        for m in mat_names:
            spec = specs[m]
            kind = spec.get("kind") or ("terrain" if m.startswith(("ortho_", "backdrop_")) else m)
            props = dict(mats.get(kind, mats["terrain"]))
            props.update(spec.get("props") or {})
            if spec.get("alpha") and not props.get("ksAlphaRef"):
                props["ksAlphaRef"] = 0.5          # alpha-tested: cut-out below half opacity
            texs = [("txDiffuse", 0, f"{m}.dds")] + [(slot, idx, extra_names[fname])
                                                     for slot, idx, fname, _ in spec.get("maps", [])]
            w.material(m, props, texs, shader=spec.get("shader", "ksPerPixel"),
                       alpha_tested=bool(spec.get("alpha")))
        # node tree: root → meshes + dummies
        objs = man["objects"]
        w.dummy("realroad", np.eye(4), len(objs) + len(man["dummies"]))
        nv = nt = 0
        for o in objs:
            k = o["key"]
            verts, faces, uvs = data[f"v{k}"], data[f"f{k}"], data[f"uv{k}"]
            assert len(verts) <= KN5_MAX_VERTS, o["name"]
            inset = tex_size[o["material"]] if o["material"].startswith(("ortho_", "backdrop_", "roadside_")) else None
            vb, centre, radius = kn5_vertices(verts, faces.astype(np.int64), uvs, inset)
            w.mesh(o["name"], vb, faces.astype("<u2"), mat_names.index(o["material"]), centre, radius,
                   lod_out=o.get("lod_out", 0.0), hidden=o.get("hidden", False), shadows=o.get("shadows", True))
            nv += len(verts)
            nt += len(faces)
        for d in man["dummies"]:
            w.dummy(d["name"], dummy_matrix(d["pos"], d["fwd"], a.spawn_flip), 0)
    tmp.replace(kn5_path)
    trees_list = None
    if man.get("csp_trees"):
        tdir = out / "extension" / "trees"
        tdir.mkdir(parents=True, exist_ok=True)
        for b in (build / "csp_trees").glob("*.bin"):
            shutil.copy(b, tdir / b.name)
        (tdir / "realroad_trees.txt").write_text(csp_trees_list(man["csp_trees"]), encoding="utf-8")
        trees_list = "trees/realroad_trees.txt"
        print(f"  CSP 3D trees: {len(man['csp_trees']):,} in extension/trees/realroad_trees.txt")
    if p["csp_config"] or trees_list:
        (out / "extension").mkdir(exist_ok=True)
        (out / "extension" / "ext_config.ini").write_text(csp_ext_config(mat_names, trees_list), encoding="utf-8")
    print(f"  {kn5_path.name}: {len(objs)} meshes, {nv:,} vertices, {nt:,} triangles, "
          f"{len(tex_table)} textures, {kn5_path.stat().st_size / 1e6:.0f} MB")

    # AI spline, map, cameras
    R, L, C = road_line(man, data)
    ai, P, cum, info = ai_spline(R, L, C, p)
    if a.no_ai:
        (out / "ai" / "fast_lane.ai").unlink(missing_ok=True)
        (out / "data" / "cameras.ini").unlink(missing_ok=True)
    else:
        (out / "ai" / "fast_lane.ai").write_bytes(ai)
        hf = HeightField(man, data)
        cam, ncam = cameras_ini(C, R, L, hf, p["camera_every_m"], "realroad TV")
        (out / "data" / "cameras.ini").write_text(cam, encoding="utf-8")
        print(f"  ai/fast_lane.ai: {info['points']} points, {info['length_m'] / 1000:.2f} km, "
              f"speed hint {info['speed_kmh'][0]:.0f}-{info['speed_kmh'][2]:.0f} km/h; {ncam} replay cameras")
    ini, mimg = map_files(P, p)
    (out / "data" / "map.ini").write_text(ini, encoding="utf-8")
    mimg.save(out / "map.png")
    outline_png(P).save(out / "ui" / "outline.png")
    prev = preview_png(man, P, tex_dir)
    if prev is not None:
        prev.save(out / "ui" / "preview.png")

    # ui_track.json (from build_track.py, or a minimal one)
    src_ui = build / "ac_track" / man["track_id"] / "ui" / "ui_track.json"
    ui = json.loads(src_ui.read_text(encoding="utf-8")) if src_ui.exists() else {
        "name": man.get("title", tid), "run": "A2B", "length": f"{man['length_m']:.0f}"}
    if tid != man["track_id"]:
        ui["name"] = f"{ui.get('name', tid)} ({tid})"
    ui["version"] = "0.2"
    (out / "ui" / "ui_track.json").write_text(json.dumps(ui, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"→ {out}")

    if a.install:
        dest = Path(a.install) / "content" / "tracks" / tid
        if not (Path(a.install) / "content" / "tracks").is_dir():
            raise SystemExit(f"{a.install} does not look like an Assetto Corsa folder (no content/tracks)")
        shutil.copytree(out, dest, dirs_exist_ok=True)
        if a.no_ai:
            (dest / "ai" / "fast_lane.ai").unlink(missing_ok=True)
        print(f"→ installed in {dest}")


if __name__ == "__main__":
    main()
