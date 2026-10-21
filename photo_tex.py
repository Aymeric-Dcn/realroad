"""
Textures made from the free CC0 photo assets (assets.py → data/_shared/assets):
real twigs, leaves and bark for the trees, real asphalt, rock, grass and
gravel for the ground. Every function returns None (or False) when the
assets are missing, and the caller keeps its procedural texture.
"""
import math
from pathlib import Path

import numpy as np
from PIL import Image

# conifer sprites by climate: Aleppo / maritime pines in the Mediterranean,
# firs and spruces elsewhere (Vosges, Alps above the pines, Massif central)
CONIFER = {"mediterranean": "pine_tree_01", "temperate": "fir_tree_01"}


def assets_dir(data_dir):
    p = Path(data_dir) / "_shared" / "assets"
    return p if p.is_dir() else None


def _rgb(path, size=None):
    im = Image.open(path).convert("RGB")
    if size:
        im = im.resize(size, Image.LANCZOS)
    return np.asarray(im, np.float32) / 255.0


def _l(path, size=None):
    im = Image.open(path).convert("L")
    if size:
        im = im.resize(size, Image.LANCZOS)
    return np.asarray(im, np.float32) / 255.0


# --------------------------------------------------------------- sprites
def sprites(diff_path, alpha_path, min_area=0.002, max_fill=0.6, min_fill=0.08, opening=0, min_dim=24,
            green=0.02):
    """Cut the separate twigs / leaves out of an atlas: connected parts of the
    alpha mask that are big enough, neither solid (bark strips and trunk
    pieces of the same atlas) nor nearly empty (bare branches), not the whole
    sheet, and green. opening (pixels) first removes the thin stems that join
    the leaves of a spray, to get single leaves. Returns RGBA float crops."""
    from scipy.ndimage import binary_closing, binary_opening, find_objects, label

    rgb = _rgb(diff_path)
    a = _l(alpha_path)
    m = a > 0.5
    core = binary_opening(m, iterations=opening) if opening else m
    lab, n = label(binary_closing(core, iterations=2))
    out = []
    H, W = m.shape
    for i, sl in enumerate(find_objects(lab), 1):
        if sl is None:
            continue
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        part = lab[sl] == i
        if opening:          # grow back the leaf edges eaten by the opening (not the stems)
            from scipy.ndimage import binary_dilation

            part = binary_dilation(part, iterations=max(1, opening // 2)) & m[sl]
        px = m[sl] & part
        area = px.sum()
        fill = area / (h * w)
        if area < min_area * H * W or not (min_fill <= fill <= max_fill) or min(h, w) < min_dim \
                or h > 0.9 * H or w > 0.9 * W:
            continue
        c = rgb[sl][px].mean(0)
        if c[1] - max(c[0], c[2]) < green * c.mean():      # not foliage (bark, junk geometry)
            continue
        out.append(np.dstack([rgb[sl], a[sl] * part]))
    return out


def _paste(canvas, spr, cx, cy, scale, angle):
    """Alpha-over of a sprite (RGBA float) rotated and scaled, centred at (cx, cy)."""
    im = Image.fromarray((np.clip(spr, 0, 1) * 255).astype(np.uint8), "RGBA")
    w, h = im.size
    im = im.resize((max(2, int(w * scale)), max(2, int(h * scale))), Image.LANCZOS)
    im = im.rotate(angle, resample=Image.BICUBIC, expand=True)
    s = np.asarray(im, np.float32) / 255.0
    h, w = s.shape[:2]
    x0, y0 = int(cx - w / 2), int(cy - h / 2)
    H, W = canvas.shape[:2]
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
    if xa >= xb or ya >= yb:
        return
    s = s[ya - y0:yb - y0, xa - x0:xb - x0]
    dst = canvas[ya:yb, xa:xb]
    al = s[..., 3:4]
    dst[..., :3] = s[..., :3] * al + dst[..., :3] * (1 - al)
    dst[..., 3:4] = al + dst[..., 3:4] * (1 - al)


def _tint(rgba, target, strength=0.75):
    """Move the mean colour of the opaque texels towards the tree colour seen in
    the orthophotos (per channel gain), keeping the photo detail."""
    m = rgba[..., 3] > 0.5
    if not m.any():
        return rgba
    mean = rgba[..., :3][m].mean(0)
    gain = (np.asarray(target, np.float32) / 255.0) / np.maximum(mean, 1e-3)
    gain = 1 + strength * (gain - 1)
    out = rgba.copy()
    out[..., :3] = np.clip(rgba[..., :3] * gain, 0, 1)
    return out


def cluster(sprs, size, rng, n, scale=(0.45, 0.75), spread=0.28, upright=False):
    """One leaf-cluster image: n sprites around the centre, random sizes and
    angles (upright: pointing away from the centre, like twigs on a branch)."""
    canvas = np.zeros((size, size, 4), np.float32)
    for k in range(n):
        spr = sprs[int(rng.integers(0, len(sprs)))]
        h, w = spr.shape[:2]
        sc = rng.uniform(*scale) * size / max(h, w)
        r = min(spread * size * abs(rng.normal(0, 0.6)), 0.42 * size)
        a = rng.uniform(0, 2 * math.pi)
        cx, cy = size / 2 + r * math.cos(a), size / 2 + r * math.sin(a)
        ang = (-math.degrees(a) - 90 + rng.normal(0, 25)) if upright else rng.uniform(0, 360)
        _paste(canvas, spr, cx, cy, sc, ang)
    return canvas


def foliage_fill(sprs, w, h, rng, size_px, coverage=2.2):
    """A sheet of foliage: sprites of about size_px pasted all over (coverage =
    sprite area pasted / sheet area), random angles. RGBA float, opaque where
    covered. Its colour is normalised (mean 1) to be multiplied by the crown
    colour and shading of the billboards."""
    canvas = np.zeros((h, w, 4), np.float32)
    n = int(coverage * w * h / (size_px * size_px * 0.35))
    for _ in range(n):
        spr = sprs[int(rng.integers(0, len(sprs)))]
        sc = size_px * rng.uniform(0.7, 1.3) / max(spr.shape[:2])
        _paste(canvas, spr, rng.uniform(0, w), rng.uniform(0, h), sc, rng.uniform(0, 360))
    m = canvas[..., 3] > 0.5
    if m.any():
        canvas[..., :3] /= np.maximum(canvas[..., :3][m].mean(0), 1e-3)
    return canvas


def _normal_from_height(hgt, strength=2.0):
    gy, gx = np.gradient(hgt)
    n = np.dstack([-gx * strength, -gy * strength, np.ones_like(hgt)])
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    return ((n * 0.5 + 0.5) * 255).astype(np.uint8)


def _finish(atlas, path_rgba, path_nm):
    from textures import bleed

    atlas[..., 3] = (atlas[..., 3] > 0.5).astype(np.float32)        # hard cut-out
    atlas = bleed(atlas, iterations=8)
    Image.fromarray((np.clip(atlas, 0, 1) * 255).astype(np.uint8), "RGBA").save(path_rgba)
    lum = atlas[..., :3].mean(2) * atlas[..., 3]
    from scipy.ndimage import gaussian_filter

    Image.fromarray(_normal_from_height(gaussian_filter(lum, 1.2) * 6)).save(path_nm)


def foliage_sets(assets, climate):
    """Sprites per foliage kind: conifer, broadleaf, shrub (None if missing)."""
    con = assets / CONIFER[climate]
    leaf = assets / "tree_small_02"
    need = [con / "twig_diff.jpg", con / "twig_alpha.jpg", leaf / "leaves_diff.jpg", leaf / "leaves_alpha.jpg"]
    if not all(p.exists() for p in need):
        return None
    # conifer twigs: big sprites only (the small parts of the sheets are bits of
    # needle geometry); pine needles are olive-brown: hardly greener than red
    sets = {"conifer": sprites(*need[:2], min_dim=200, green=-0.08),
            # clean single leaves only (the sprays of the sheet keep thick stems)
            "broadleaf": sprites(*need[2:], opening=10, min_fill=0.6, max_fill=0.85, green=0.05)}
    sets["shrub"] = sets["broadleaf"]       # (shrub_02's long leaves look dead once tinted)
    if not sets["conifer"] or not sets["broadleaf"]:
        return None
    return sets


def climate_of(lat, lon):
    """Mediterranean (Aleppo pines, holm oaks) or temperate (firs, beeches)."""
    return "mediterranean" if (lat < 44.6 and lon > 2.5) or lat < 43.6 else "temperate"


def csp_leaves(assets, palette, out, climate, seed=9, cell=512):
    """Leaf atlas of the CSP tree models (4 cells: conifer A, conifer B,
    broadleaf, shrub; same layout as trees3d.leaves_textures) from the photo
    sprites, tinted to the road's tree colours. Writes leaves.png and
    leaves_normal.png; False when the assets are missing."""
    from textures import foliage_palette

    sets = foliage_sets(assets, climate) if assets else None
    if not sets:
        return False
    pal = foliage_palette(palette)
    rng = np.random.default_rng(seed)
    cells = []
    for kind, col, n, sc, up in (("conifer", pal["pine"], 9, (0.42, 0.68), True),
                                 ("conifer", pal["pine"], 11, (0.36, 0.6), True),
                                 ("broadleaf", pal["broadleaf"], 170, (0.06, 0.11), False),
                                 ("shrub", pal["shrub"], 150, (0.05, 0.09), False)):
        c = cluster(sets[kind], cell, rng, n, scale=sc, upright=up, spread=0.28 if up else 0.36)
        cells.append(_tint(c, col))
    _finish(np.concatenate(cells, axis=1), out / "leaves.png", out / "leaves_normal.png")
    return True


def csp_bark(assets, out, size=512):
    """Bark of the tree models: a photographed broadleaf bark (diffuse + normal)."""
    src = assets / "tree_small_02" if assets else None
    if not src or not (src / "Diffuse.jpg").exists():
        return False
    Image.open(src / "Diffuse.jpg").convert("RGB").resize((size, size), Image.LANCZOS).save(out / "bark.png")
    if (src / "nor_gl.jpg").exists():
        Image.open(src / "nor_gl.jpg").convert("RGB").resize((size, size), Image.LANCZOS).save(out / "bark_normal.png")
    return True


# ---------------------------------------------------------------- ground
def _asset(assets, aid, kind):
    if not assets:
        return None
    f = sorted((assets / aid).glob(f"*_{kind}.jpg"))
    return f[0] if f else None


def _detail(rgb, mean, desat, contrast=1.0):
    """Detail layer for a Kunos multiply shader: photo detail, colour mostly
    removed (the orthophoto or the diffuse gives the hue), mean brought to
    `mean` so that it neither darkens nor lightens on average."""
    lum = rgb.mean(2, keepdims=True)
    c = lum + (1 - desat) * (rgb - lum)
    m = float(c.mean())
    c = m + contrast * (c - m)
    return np.clip(c * (mean / max(m, 1e-3)), 0, 1)


def _save_rgba(path, rgb, a):
    Image.fromarray((np.dstack([np.clip(rgb, 0, 1), np.clip(a, 0, 1)]) * 255).astype(np.uint8), "RGBA").save(path)


def road_photo(assets, out, asphalt_png):
    """Road shader maps from a photographed asphalt (ambientCG Asphalt031):
    the detail grain (txDetail, mean 0.8, specular from the roughness) and its
    normal map, a fine relief (txNormal) and a light grain in the diffuse."""
    col, nrm, rough = (_asset(assets, "Asphalt031", k) for k in ("Color", "NormalGL", "Roughness"))
    if not col:
        return False
    n = 1024
    rgb = _rgb(col, (n, n))
    spec = 0.06 + 0.3 * (1 - _l(rough, (n, n))) if rough else np.full((n, n), 0.1)
    _save_rgba(out / "road_detail.png", _detail(rgb, 0.8, 1.0, 0.8), spec)
    if nrm:
        Image.open(nrm).convert("RGB").resize((n, n), Image.LANCZOS).save(out / "road_detail_nm.png")
    # diffuse: the procedural base (tone, wheel tracks) times a light photo grain
    base = np.asarray(Image.open(asphalt_png).convert("RGB"), np.float32) / 255
    h, w = base.shape[:2]
    tile = _detail(_rgb(col, (256, 256)), 1.0, 1.0)[..., :1]
    reps = (h // 256 + 1, w // 256 + 1, 1)
    grain = np.tile(tile, reps)[:h, :w]
    Image.fromarray((np.clip(base * (0.75 + 0.25 * grain), 0, 1) * 255).astype(np.uint8)).save(asphalt_png)
    return True


def terrain_photo(assets, out, climate):
    """Detail layers of the terrain shader from photos: R rock face, G grass
    (lawn green in the north, the same photo dried to straw in the
    Mediterranean), B gravel / soil. Mean ~0.6, colour mostly removed."""
    rock = _asset(assets, "Rock051", "Color")
    grass = _asset(assets, "Grass004", "Color")
    soil = _asset(assets, "Ground110", "Color")
    if not (rock and grass and soil):
        return False
    n = 1024
    _save_rgba(out / "det_rock.png", _detail(_rgb(rock, (n, n)), 0.6, 0.7, 0.8), np.full((n, n), 0.12))
    g = _rgb(grass, (n, n))
    if climate == "mediterranean":
        g = g * np.array([1.15, 1.0, 0.95]) + np.array([0.05, 0.03, 0.0])   # drier, straw tones
    _save_rgba(out / "det_grass.png", _detail(g, 0.6, 0.6), np.full((n, n), 0.06))
    _save_rgba(out / "det_soil.png", _detail(_rgb(soil, (n, n)), 0.6, 0.6, 0.6), np.full((n, n), 0.1))
    return True


def rock_photo(assets, out, tint=None):
    """Rock faces of the road cuts (ksPerPixelNM): photographed cliff
    (ambientCG Rock051) and its normal map, coloured like the bare rock of
    this road on the orthophotos (limestone beige, sandstone pink…) and kept
    mid-dark (mean 0.42): the faces catch the sun, a light texture burns out."""
    col, nrm = _asset(assets, "Rock051", "Color"), _asset(assets, "Rock051", "NormalGL")
    if not col:
        return False
    n = 1024
    rgb = _detail(_rgb(col, (n, n)), 1.0, 0.85)                 # grey photo, mean 1
    target = np.array([0.62, 0.58, 0.52]) if tint is None else np.clip(np.asarray(tint, float), 0.2, 0.9)
    target = target * (0.42 / max(float(target.mean()), 1e-3))
    Image.fromarray((np.clip(rgb * target, 0, 1) * 255).astype(np.uint8)).save(out / "rock.png")
    if nrm:
        # softened relief: the photo's normal map is made for close-up renders
        nm = _rgb(nrm, (n, n)) * 2 - 1
        nm[..., :2] *= 0.5
        nm /= np.linalg.norm(nm, axis=2, keepdims=True)
        Image.fromarray(((nm * 0.5 + 0.5) * 255).astype(np.uint8)).save(out / "rock_nm.png")
    return True
