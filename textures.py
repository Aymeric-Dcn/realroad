"""
Procedural textures for the 3D scenery: tree billboards and building façades.

Kept simple and generated from numbers only (no third-party images, so no
licence question). Colours come from the orthophoto where possible
(trees_palette.json), so what is drawn matches the photo seen from above.

    tree_atlas(path, palette)   → RGBA PNG, 4 cells side by side (512 x 1024 each):
                                  0 pine A, 1 pine B, 2 broadleaf, 3 shrub
    facade_texture(path)        → RGB PNG, one 4 m x 3 m bay (wall + window), tiling
"""
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

TREE_CELLS = 4
CELL_W, CELL_H = 512, 1024


def _blobs(rng, h, w, n, r_min, r_max):
    """Sum of random soft dots: leafy noise in [0, 1]."""
    img = Image.new("L", (w, h), 0)
    dr = ImageDraw.Draw(img)
    for _ in range(n):
        x, y = rng.uniform(0, w), rng.uniform(0, h)
        r = rng.uniform(r_min, r_max)
        dr.ellipse([x - r, y - r, x + r, y + r], fill=int(rng.uniform(150, 255)))
    return np.asarray(img.filter(ImageFilter.GaussianBlur(1.2)), np.float32) / 255.0


def _fbm(rng, h, w, scales=(64, 24, 9)):
    """Smooth fractal noise in [0, 1]."""
    out = np.zeros((h, w), np.float32)
    amp = 1.0
    for sc in scales:
        small = rng.random((max(2, h // sc), max(2, w // sc))).astype(np.float32)
        img = Image.fromarray((small * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)
        out += amp * np.asarray(img, np.float32) / 255.0
        amp *= 0.6
    out -= out.min()
    return out / max(out.max(), 1e-6)


def _crown(rng, clumps, base_rgb, w=CELL_W, h=CELL_H, density=0.55):
    """Alpha and colour of a crown made of clumps (cx, cy, rx, ry) in pixels:
    mostly filled inside, ragged leafy edges, light from the upper left."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    env = np.zeros((h, w), np.float32)
    shade = np.zeros((h, w), np.float32)
    for cx, cy, rx, ry in clumps:
        rx = min(rx, w / 2 - 12)                    # stay inside the atlas cell
        cx = float(np.clip(cx, rx + 10, w - rx - 10))
        ry = min(ry, h / 2 - 12)
        cy = float(np.clip(cy, ry + 10, h - ry - 10))
        d = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
        env = np.maximum(env, np.clip(1.0 - d, 0, 1))
        sh = np.clip(0.6 - 0.4 * (yy - cy) / ry - 0.25 * (xx - cx) / rx, 0, 1) * (d < 1.3)
        shade = np.maximum(shade, sh)
    leaves = _blobs(rng, h, w, int(0.02 * w * h * density), 1.2, 3.2)
    fbm = _fbm(rng, h, w)
    field = np.sqrt(env) + 0.35 * (leaves - 0.5) + 0.45 * (fbm - 0.5) * (1 - env)
    alpha = (field > 0.42).astype(np.float32)
    # small gaps inside the crown (sky seen through it)
    alpha *= (fbm + 0.3 * leaves > 0.18 + 0.25 * (1 - density)).astype(np.float32)
    base = np.array(base_rgb, np.float32) / 255.0
    var = 0.78 + 0.35 * leaves[..., None] + 0.2 * (fbm[..., None] - 0.5)
    col = np.clip(base * var * (0.72 + 0.55 * shade[..., None]), 0, 1)
    return alpha, col


def _trunk(draw, x0, y0, x1, y1, w0, w1, col):
    draw.polygon([(x0 - w0, y0), (x0 + w0, y0), (x1 + w1, y1), (x1 - w1, y1)], fill=col)


def _cell(kind, variant, palette, seed):
    rng = np.random.default_rng(seed)
    w, h = CELL_W, CELL_H
    rgba = np.zeros((h, w, 4), np.float32)
    trunk_img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    td = ImageDraw.Draw(trunk_img)
    bark = (88, 72, 58, 255)
    if kind == "pine":
        # Aleppo / maritime pine: tall bare trunk, irregular flattened crown on top
        lean = rng.uniform(-40, 40)
        top_y = rng.uniform(0.38, 0.48) * h
        _trunk(td, w / 2, h, w / 2 + lean, top_y + 60, 14, 7, bark)
        for _ in range(3):  # a few branches into the crown
            bx = w / 2 + lean * 0.8
            _trunk(td, bx, top_y + 80, bx + rng.uniform(-150, 150), top_y + rng.uniform(-40, 40), 5, 2, bark)
        clumps = []
        n = 10 if variant == 0 else 8
        for i in range(n):  # layered, flattened plates around the trunk top
            cx = w / 2 + lean + rng.normal(0, 0.16 * w)
            cy = (0.08 + (top_y / h - 0.02) * (i + rng.random()) / n) * h
            clumps.append((cx, cy, rng.uniform(0.18, 0.28) * w, rng.uniform(0.05, 0.08) * h))
        alpha, col = _crown(rng, clumps, palette["pine"], density=0.9)
    elif kind == "broadleaf":
        # holm / downy oak: short trunk, dense round crown
        _trunk(td, w / 2, h, w / 2 + rng.uniform(-15, 15), 0.6 * h, 16, 10, bark)
        clumps = [(w / 2 + rng.normal(0, 0.12 * w), rng.uniform(0.22, 0.6) * h,
                   rng.uniform(0.2, 0.3) * w, rng.uniform(0.12, 0.17) * h) for _ in range(8)]
        alpha, col = _crown(rng, clumps, palette["broadleaf"], density=0.95)
    else:
        # garrigue shrub / young tree: crown down to the ground
        clumps = [(w / 2 + rng.normal(0, 0.15 * w), rng.uniform(0.35, 0.85) * h,
                   rng.uniform(0.22, 0.32) * w, rng.uniform(0.15, 0.22) * h) for _ in range(9)]
        alpha, col = _crown(rng, clumps, palette["shrub"], density=0.95)
    t = np.asarray(trunk_img, np.float32) / 255.0
    rgba[..., :3] = t[..., :3] * t[..., 3:] * (1 - alpha[..., None]) + col * alpha[..., None]
    rgba[..., 3] = np.maximum(alpha, t[..., 3])
    return rgba


def bleed(rgba, iterations=6):
    """Give transparent pixels the colour of their opaque neighbours, so that
    filtering/mipmaps do not make dark halos around the leaves."""
    a = (rgba[..., 3] > 0.5).astype(np.float32)
    col = rgba[..., :3] * a[..., None]
    wsum = a.copy()
    acc, wacc = col.copy(), wsum.copy()
    for _ in range(iterations):
        acc = (acc + np.roll(acc, 1, 0) + np.roll(acc, -1, 0) + np.roll(acc, 1, 1) + np.roll(acc, -1, 1))
        wacc = (wacc + np.roll(wacc, 1, 0) + np.roll(wacc, -1, 0) + np.roll(wacc, 1, 1) + np.roll(wacc, -1, 1))
    fill = acc / np.maximum(wacc, 1e-6)[..., None]
    mean = col.sum((0, 1)) / max(a.sum(), 1)
    fill = np.where(wacc[..., None] > 1e-6, fill, mean)
    out = rgba.copy()
    out[..., :3] = np.where(a[..., None] > 0, rgba[..., :3], fill)
    return out


def foliage_palette(palette):
    """The orthophoto crown colours are dark and bluish (shade, haze): lift
    them to a plausible sunlit foliage colour while keeping their hue order."""
    out = {}
    for k, c in palette.items():
        c = np.array(c, np.float32)
        lum = c.mean()
        c = c * max(1.0, 88.0 / max(lum, 1.0))                 # at least mid-dark
        g = max(c[1], 1.12 * max(c[0], c[2]))                  # green dominant
        # Mediterranean foliage is olive rather than teal: red not too low, blue low
        c = np.array([np.clip(c[0], 0.68 * g, g / 1.1), g, min(c[2], 0.62 * g)])
        out[k] = [int(v) for v in np.clip(c, 0, 230)]
    return out


def tree_atlas(path, palette, seed=7):
    palette = foliage_palette(palette)
    cells = [("pine", 0), ("pine", 1), ("broadleaf", 0), ("shrub", 0)]
    atlas = np.concatenate([_cell(k, v, palette, seed + i) for i, (k, v) in enumerate(cells)], axis=1)
    atlas = bleed(atlas)
    Image.fromarray((np.clip(atlas, 0, 1) * 255).astype(np.uint8), "RGBA").save(path)
    return path


# cell index of each kind in the atlas (pine has two variants)
TREE_CELL = {0: (0, 1), 1: (2,), 2: (3,), 3: (3,)}
# share of the billboard width taken by the crown in each cell (3: low bush)
TREE_WIDTH_FACTOR = {0: 1.25, 1: 1.15, 2: 1.15, 3: 1.3}

FOLIAGE_CELLS = 4
FOLIAGE_CELL = {0: 0, 1: 2, 2: 3}   # pine (0 or 1), broadleaf, shrub


def foliage_atlas(path, palette, seed=17):
    """Square leafy clumps (512 x 512 each) for the foliage cards placed
    inside the crowns: 0, 1 pine needles (darker, ragged), 2 broadleaf, 3 shrub."""
    palette = foliage_palette(palette)
    cells = []
    for i, name in enumerate(("pine", "pine", "broadleaf", "shrub")):
        rng = np.random.default_rng(seed + i)
        w = h = 512
        n = 9 if name == "pine" else 7
        clumps = [(w / 2 + rng.normal(0, 0.13 * w), h / 2 + rng.normal(0, 0.13 * h),
                   rng.uniform(0.16, 0.26) * w, rng.uniform(0.12, 0.2 if name == "pine" else 0.24) * h)
                  for _ in range(n)]
        alpha, col = _crown(rng, clumps, palette[name], w=w, h=h, density=0.95)
        rgba = np.zeros((h, w, 4), np.float32)
        rgba[..., :3] = col
        rgba[..., 3] = alpha
        cells.append(rgba)
    atlas = bleed(np.concatenate(cells, axis=1))
    Image.fromarray((np.clip(atlas, 0, 1) * 255).astype(np.uint8), "RGBA").save(path)
    return path


def facade_texture(path, seed=11, w=512, h=384):
    """One bay of a Provençal façade, 4 m wide x 3 m high: warm render,
    a window with a stone frame and shutters, tiles horizontally and per floor."""
    rng = np.random.default_rng(seed)
    base = np.array([0.80, 0.70, 0.55])
    noise = Image.fromarray((rng.normal(0.5, 0.15, (h // 4, w // 4)).clip(0, 1) * 255).astype(np.uint8))
    noise = np.asarray(noise.resize((w, h), Image.BICUBIC).filter(ImageFilter.GaussianBlur(3)), np.float32) / 255
    img = base[None, None, :] * (0.88 + 0.24 * noise[..., None])
    img += 0.02 * rng.standard_normal((h, w, 1))
    im = Image.fromarray((img.clip(0, 1) * 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    # window 1.0 x 1.4 m centred, frame, shutters
    ppm = w / 4.0
    wx0, wx1 = int(1.5 * ppm), int(2.5 * ppm)
    wy0, wy1 = int(0.7 * ppm), int(2.1 * ppm)
    d.rectangle([wx0 - 8, wy0 - 8, wx1 + 8, wy1 + 10], fill=(222, 210, 188))
    d.rectangle([wx0, wy0, wx1, wy1], fill=(38, 42, 48))
    d.line([((wx0 + wx1) // 2, wy0), ((wx0 + wx1) // 2, wy1)], fill=(200, 196, 186), width=4)
    d.line([(wx0, (wy0 + wy1) // 2), (wx1, (wy0 + wy1) // 2)], fill=(200, 196, 186), width=4)
    shutter = (96, 128, 118)
    sw = (wx1 - wx0) // 2
    d.rectangle([wx0 - 8 - sw, wy0 - 4, wx0 - 10, wy1 + 4], fill=shutter)
    d.rectangle([wx1 + 10, wy0 - 4, wx1 + 8 + sw, wy1 + 4], fill=shutter)
    for yy in range(wy0, wy1, 14):  # louvres
        d.line([(wx0 - 8 - sw, yy), (wx0 - 10, yy)], fill=(80, 108, 98), width=2)
        d.line([(wx1 + 10, yy), (wx1 + 8 + sw, yy)], fill=(80, 108, 98), width=2)
    # base of the wall slightly darker (a bit of dirt)
    arr = np.asarray(im, np.float32)
    grad = np.linspace(1.0, 0.9, h)[:, None, None] ** 2
    im = Image.fromarray(np.clip(arr * grad, 0, 255).astype(np.uint8))
    im.save(path)
    return path


FACADE_BAY_M = (4.0, 3.0)   # metres covered by one texture repeat (u, v)


# ------------------------------------------------------------ road furniture
SIGN_W, SIGN_H = 2048, 1024
SIGN_CELL = 256
_SIGN_SMALL = ["speed30", "speed50", "speed70", "speed80", "speed90", "stop", "giveway", "back_disc",
               "back_tri", "back_rect"]
# town signs (512 x 256): the rest of row 1, then rows 2 and 3
_TOWN_SLOTS = [(2 * SIGN_CELL + i * 512, SIGN_CELL) for i in range(3)] + \
              [(i * 512, r * SIGN_CELL) for r in (2, 3) for i in range(4)]


def _px_to_uv(x0, y0, w, h):
    m = 1.5
    return ((x0 + m) / SIGN_W, 1 - (y0 + h - m) / SIGN_H, (x0 + w - m) / SIGN_W, 1 - (y0 + m) / SIGN_H)


def sign_uv(kind):
    i = _SIGN_SMALL.index(kind)
    return _px_to_uv((i % 8) * SIGN_CELL, (i // 8) * SIGN_CELL, SIGN_CELL, SIGN_CELL)


def _font(size):
    from PIL import ImageFont

    for name in ("DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf", "LiberationSans-Bold.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "C:/Windows/Fonts/arialbd.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _fit_text(d, text, box_w, size):
    while size > 10:
        f = _font(size)
        w = d.textlength(text, font=f)
        if w <= box_w:
            return f
        size -= 2
    return _font(size)


RED, WHITE, BLACK, GREY = (200, 30, 35, 255), (245, 245, 240, 255), (20, 20, 22, 255), (150, 152, 150, 255)


def signs_atlas(path, towns):
    """French road signs (shapes and colours of the rules), transparent
    around them (alpha tested). towns = [(kind, name)], kind town / town_end."""
    img = Image.new("RGBA", (SIGN_W, SIGN_H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    C = SIGN_CELL
    for i, k in enumerate(_SIGN_SMALL):
        x0, y0 = (i % 8) * C, (i // 8) * C
        cx, cy, r = x0 + C / 2, y0 + C / 2, C / 2 - 6
        if k.startswith("speed"):
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=RED)
            ri = r * 0.8
            d.ellipse([cx - ri, cy - ri, cx + ri, cy + ri], fill=WHITE)
            f = _fit_text(d, k[5:], ri * 1.45, int(r * 1.05))
            d.text((cx, cy + 4), k[5:], font=f, fill=BLACK, anchor="mm")
        elif k == "stop":
            pts = [(cx + r * math.cos(math.pi / 8 + j * math.pi / 4), cy + r * math.sin(math.pi / 8 + j * math.pi / 4)) for j in range(8)]
            d.polygon(pts, fill=WHITE)
            pts = [(cx + 0.92 * r * math.cos(math.pi / 8 + j * math.pi / 4), cy + 0.92 * r * math.sin(math.pi / 8 + j * math.pi / 4)) for j in range(8)]
            d.polygon(pts, fill=RED)
            d.text((cx, cy + 3), "STOP", font=_fit_text(d, "STOP", 1.5 * r, int(0.62 * r)), fill=WHITE, anchor="mm")
        elif k == "giveway":
            tri = [(x0 + 6, y0 + 14), (x0 + C - 6, y0 + 14), (cx, y0 + C - 8)]
            d.polygon(tri, fill=RED)
            sh = 0.2
            inner = [(tri[0][0] + sh * (cx - tri[0][0]) * 1.0, tri[0][1] + 20), (tri[1][0] - sh * (tri[1][0] - cx), tri[1][1] + 20),
                     (cx, tri[2][1] - 38)]
            d.polygon(inner, fill=WHITE)
        elif k == "back_disc":
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=GREY)
        elif k == "back_tri":
            d.polygon([(x0 + 6, y0 + 14), (x0 + C - 6, y0 + 14), (cx, y0 + C - 8)], fill=GREY)
        elif k == "back_rect":
            d.rectangle([x0 + 4, y0 + 4 + C / 4, x0 + C - 4, y0 + 3 * C / 4 - 4], fill=GREY)
    slots = {}
    for (kind, name), (x0, y0) in zip(towns, _TOWN_SLOTS):
        w, h = 512, 256
        d.rectangle([x0 + 4, y0 + 4, x0 + w - 4, y0 + h - 4], fill=RED)
        d.rectangle([x0 + 22, y0 + 22, x0 + w - 22, y0 + h - 22], fill=WHITE)
        label = (name or "").upper()
        if label:
            words = label.replace("-", "- ").split()
            lines = [label] if len(label) <= 14 else [" ".join(words[:max(1, len(words) // 2)]).replace("- ", "-"),
                                                      " ".join(words[max(1, len(words) // 2):]).replace("- ", "-")]
            fs = 70 if len(lines) == 1 else 52
            for j, ln in enumerate(lines):
                f = _fit_text(d, ln, w - 70, fs)
                yy = y0 + h / 2 + (j - (len(lines) - 1) / 2) * fs * 1.1
                d.text((x0 + w / 2, yy), ln, font=f, fill=BLACK, anchor="mm")
        if kind == "town_end":
            d.line([(x0 + 30, y0 + h - 30), (x0 + w - 30, y0 + 30)], fill=RED, width=22)
        slots[(kind, name)] = _px_to_uv(x0, y0, w, h)
    arr = bleed(np.asarray(img, np.float32) / 255.0, iterations=4)   # no dark fringes in the mipmaps
    Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), "RGBA").save(path)
    return slots


def furniture_textures(tex_dir, towns):
    """All the furniture textures; returns the town sign slots."""
    rng = np.random.default_rng(23)
    # galvanised W-beam: bright / dark bands across the height, a little dirt
    v = np.linspace(0, 1, 64)
    prof = 0.78 + 0.16 * np.cos(2 * np.pi * (v * 2 - 0.25)) + 0.05 * np.cos(2 * np.pi * v * 6)
    a = np.repeat(prof[:, None], 256, 1) * (0.95 + 0.05 * rng.random((64, 256)))
    a = np.clip(a[..., None] * np.array([182, 188, 190]), 0, 255)
    Image.fromarray(a.astype(np.uint8)).save(tex_dir / "guardrail.png")
    for name, col in (("post_dark", (92, 96, 98)), ("post_metal", (160, 165, 168))):
        b = np.clip(np.array(col)[None, None] * (0.9 + 0.1 * rng.random((32, 32, 1))), 0, 255)
        Image.fromarray(b.astype(np.uint8)).save(tex_dir / f"{name}.png")
    # dry-stone wall: irregular stones, dark joints (1.5 m per repeat)
    n = 256
    pts = rng.uniform(0, n, (90, 2))
    yy, xx = np.mgrid[0:n, 0:n]
    best = np.full((n, n), 1e9)
    second = np.full((n, n), 1e9)
    lab = np.zeros((n, n), int)
    for i, (px, py) in enumerate(pts):
        for ox in (-n, 0, n):
            for oy in (-n, 0, n):
                dd = ((xx - px - ox) * 0.8) ** 2 + (yy - py - oy) ** 2
                nb = dd < best
                second = np.where(nb, best, np.minimum(second, dd))
                lab = np.where(nb, i, lab)
                best = np.where(nb, dd, best)
    edge = np.sqrt(second) - np.sqrt(best)
    tone = rng.uniform(0.75, 1.1, len(pts))[lab]
    base = np.array([168, 156, 136]) * tone[..., None]
    base *= (0.45 + 0.55 * np.clip(edge / 4.0, 0, 1))[..., None]
    base *= 0.92 + 0.08 * rng.random((n, n, 1))
    Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).save(tex_dir / "stonewall.png")
    # J6 delineator: left half = right-side post (yellow-orange rectangle),
    # right half = left-side post (two white dots); v: 0 = 0.2 m under the ground
    j6 = np.zeros((256, 64, 3), np.uint8)
    j6[:] = (236, 236, 230)
    band0, band1 = int(256 * (1 - 0.98)), int(256 * (1 - 0.80))
    j6[band0:band1] = (25, 25, 25)
    j6[int(256 * (1 - 0.2)):] = (110, 100, 85)          # buried part
    jm = Image.fromarray(j6)
    dj = ImageDraw.Draw(jm)
    dj.rectangle([10, band0 + 8, 22, band1 - 8], fill=(255, 150, 20))
    for cy in (band0 + 14, band1 - 14):
        dj.ellipse([42, cy - 5, 52, cy + 5], fill=(250, 250, 250))
    jm.save(tex_dir / "j6.png")
    # wooden pole
    w_ = np.array([112, 86, 62]) * (0.8 + 0.2 * rng.random((1, 64, 1))) * (0.92 + 0.08 * rng.random((256, 1, 1)))
    Image.fromarray(np.clip(np.broadcast_to(w_, (256, 64, 3)), 0, 255).astype(np.uint8)).save(tex_dir / "wood.png")
    # pylon lattice (alpha): edge bars, X bracing, one horizontal per repeat
    L = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
    dl = ImageDraw.Draw(L)
    col = (120, 128, 124, 255)
    dl.rectangle([0, 0, 10, 255], fill=col)
    dl.rectangle([245, 0, 255, 255], fill=col)
    dl.line([(0, 0), (255, 255)], fill=col, width=6)
    dl.line([(255, 0), (0, 255)], fill=col, width=6)
    dl.rectangle([0, 0, 255, 6], fill=col)
    L.save(tex_dir / "pylon.png")
    return signs_atlas(tex_dir / "signs.png", towns)
