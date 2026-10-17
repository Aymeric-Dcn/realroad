"""
Procedural textures for the 3D scenery: tree billboards and building façades.

Kept simple and generated from numbers only (no third-party images, so no
licence question). Colours come from the orthophoto where possible
(trees_palette.json), so what is drawn matches the photo seen from above.

    tree_atlas(path, palette)   → RGBA PNG, 4 cells side by side (512 x 1024 each):
                                  0 pine A, 1 pine B, 2 broadleaf, 3 shrub
    facade_texture(path)        → RGB PNG, one 4 m x 3 m bay (wall + window), tiling
"""
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
TREE_CELL = {0: (0, 1), 1: (2,), 2: (3,)}
# share of the billboard width taken by the crown in each cell
TREE_WIDTH_FACTOR = {0: 1.25, 1: 1.15, 2: 1.15}


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
