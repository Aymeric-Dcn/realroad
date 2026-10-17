"""
The real roadside, from street-level 360° photos (Panoramax, fetch.py streetview).

The verge strips along the road get their own texture, "baked" by projecting
the photos onto them, the way the orthophoto is draped over the terrain:
every texel is a 3D point of the strip; it takes its colour from the photos
that see it well, then fades into the orthophoto colour at the outer edge so
it blends with the terrain.

Camera poses: the GPS of the photos is only good to a few metres, so each
photo is tied to the road instead:
  - position along the road: GPS projected on the axis, smoothed against time
    within its sequence (the camera car drives smoothly),
  - across: centre of the lane it drives in (opposite lane when driving the
    other way), height 2.3 m (camera on the roof),
  - heading: the road's (the panorama centre is the direction of travel).
Checked by projecting the measured road edges into the photos: they fall on
the real edges within about a metre.

A point is seen well from 6.5-40 m away (best around 12 m) and above -20° of
pitch (lower is the camera car's roof). The two best photos are blended.
"""
import datetime as dt
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image

CAM_HEIGHT = 2.3
LANE_OFFSET = 1.75


class Photos:
    def __init__(self, folder, s, x, y, z, centre_offset, cache=24):
        folder = Path(folder)
        idx = [p for p in json.loads((folder / "index.json").read_text(encoding="utf-8"))
               if p.get("file") and (folder / p["file"]).exists()]
        self.folder = folder
        self.items = idx
        self.cache = OrderedDict()
        self.cache_n = cache
        if not idx:
            self.pos = np.zeros((0, 3))
            return
        tx, ty = np.gradient(x), np.gradient(y)
        n = np.hypot(tx, ty)
        tx, ty = tx / n, ty / n
        heading = np.degrees(np.arctan2(tx, ty)) % 360
        t = np.array([dt.datetime.fromisoformat(p["datetime"].replace("Z", "+00:00")).timestamp()
                      if p.get("datetime") else np.nan for p in idx])
        sk = np.array([s[np.argmin(np.hypot(x - p["x"], y - p["y"]))] for p in idx])
        coll = np.array([str(p.get("collection")) for p in idx])
        pos, yaw, s_cam = [], [], []
        for i, p in enumerate(idx):
            sc = sk[i]
            m = (coll == coll[i]) & (np.abs(np.arange(len(idx)) - i) <= 6) & np.isfinite(t)
            if np.isfinite(t[i]) and m.sum() >= 4:
                A = np.column_stack([t[m] - t[i], np.ones(m.sum())])
                sc = float(np.linalg.lstsq(A, sk[m], rcond=None)[0][1])
            k = int(np.clip(np.searchsorted(s, sc), 0, len(s) - 1))
            az = p.get("azimuth")
            opposite = az is not None and abs((float(az) - heading[k] + 540) % 360 - 180) > 90
            lat = centre_offset[k] + (LANE_OFFSET if opposite else -LANE_OFFSET)
            pos.append((x[k] - lat * ty[k], y[k] + lat * tx[k], z[k] + CAM_HEIGHT))
            yaw.append((heading[k] + 180) % 360 if opposite else heading[k])
            s_cam.append(sc)
        self.pos = np.array(pos)
        self.yaw = np.array(yaw)
        self.s = np.array(s_cam)

    def image(self, i):
        if i in self.cache:
            self.cache.move_to_end(i)
            return self.cache[i]
        with Image.open(self.folder / self.items[i]["file"]) as im:
            a = np.asarray(im.convert("RGB"), np.float32)
        self.cache[i] = a
        if len(self.cache) > self.cache_n:
            self.cache.popitem(last=False)
        return a

    def colours(self, P, best_m=12.0):
        """Colours of 3D points (N, 3) from the two photos that see them best:
        (c1, w1, c2, w2), weights 0 where no photo sees the point."""
        n = len(P)
        out = [np.zeros((n, 3), np.float32), np.zeros(n, np.float32),
               np.zeros((n, 3), np.float32), np.zeros(n, np.float32)]
        if not len(self.pos) or not n:
            return out
        c = P[:, :2].mean(0)
        reach = np.hypot(*(P[:, :2].max(0) - P[:, :2].min(0))) / 2 + 45
        cand = np.flatnonzero(np.hypot(self.pos[:, 0] - c[0], self.pos[:, 1] - c[1]) < reach)
        if not len(cand):
            return out
        W = np.zeros((len(cand), n), np.float32)
        for j, i in enumerate(cand):
            d = P - self.pos[i]
            dh = np.hypot(d[:, 0], d[:, 1])
            pitch = np.degrees(np.arctan2(d[:, 2], dh))
            w = np.exp(-((dh - best_m) / 8.0) ** 2) * (dh > 6.5) * (dh < 40)
            W[j] = w * np.clip((pitch + 20) / 4, 0, 1)
        order = np.argsort(-W, axis=0)[:2]           # best and second best photo per point
        for rank in range(min(2, len(cand))):
            jr = order[rank]
            wr = W[jr, np.arange(n)]
            for j, i in enumerate(cand):
                m = (jr == j) & (wr > 1e-3)
                if not m.any():
                    continue
                out[2 * rank][m] = self._sample(i, P[m])
                out[2 * rank + 1][m] = wr[m]
        return out

    def _sample(self, i, P):
        img = self.image(i)
        H, Wd = img.shape[:2]
        d = P - self.pos[i]
        bearing = np.degrees(np.arctan2(d[:, 0], d[:, 1]))
        rel = (bearing - self.yaw[i] + 540) % 360 - 180
        pitch = np.degrees(np.arctan2(d[:, 2], np.hypot(d[:, 0], d[:, 1])))
        u = (rel / 360 + 0.5) * Wd - 0.5
        v = np.clip((0.5 - pitch / 180) * H - 0.5, 0, H - 1.001)
        u0 = np.floor(u).astype(int)
        v0 = np.floor(v).astype(int)
        fu, fv = (u - u0)[:, None], (v - v0)[:, None]
        u1 = (u0 + 1) % Wd
        u0 %= Wd
        return (img[v0, u0] * (1 - fu) * (1 - fv) + img[v0, u1] * fu * (1 - fv)
                + img[v0 + 1, u0] * (1 - fu) * fv + img[v0 + 1, u1] * fu * fv)


class OrthoSampler:
    def __init__(self, tiles, cache=6):
        self.tiles = tiles
        self.cache = OrderedDict()
        self.cache_n = cache

    def _img(self, i):
        if i in self.cache:
            self.cache.move_to_end(i)
            return self.cache[i]
        with Image.open(self.tiles[i]["file"]) as im:
            a = np.asarray(im.convert("RGB"), np.float32)
        self.cache[i] = a
        if len(self.cache) > self.cache_n:
            self.cache.popitem(last=False)
        return a

    def colours(self, x, y):
        out = np.full((len(x), 3), 90.0, np.float32)
        for i, t in enumerate(self.tiles):
            x0, y0, x1, y1 = t["bounds"]
            m = (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
            if not m.any():
                continue
            a = self._img(i)
            h, w = a.shape[:2]
            cf = np.clip((x[m] - x0) / (x1 - x0) * w - 0.5, 0, w - 1.001)
            rf = np.clip((y1 - y[m]) / (y1 - y0) * h - 0.5, 0, h - 1.001)
            c0, r0 = cf.astype(int), rf.astype(int)
            fc, fr = (cf - c0)[:, None], (rf - r0)[:, None]
            c1, r1 = np.minimum(c0 + 1, w - 1), np.minimum(r0 + 1, h - 1)
            out[m] = (a[r0, c0] * (1 - fc) * (1 - fr) + a[r0, c1] * fc * (1 - fr)
                      + a[r1, c0] * (1 - fc) * fr + a[r1, c1] * fc * fr)
        return out


def _blur(a, r):
    """Box blur (separable, edge-clamped) of a (w, h, c) array."""
    out = a.astype(np.float32)
    for axis in (0, 1):
        pad = [(0, 0)] * out.ndim
        pad[axis] = (r, r)
        p_ = np.pad(out, pad, mode="edge")
        c = np.cumsum(p_, axis=axis, dtype=np.float64)
        c = np.concatenate([np.zeros_like(np.take(c, [0], axis=axis)), c], axis=axis)
        n = out.shape[axis]
        out = ((np.take(c, np.arange(2 * r + 1, n + 2 * r + 1), axis=axis)
                - np.take(c, np.arange(0, n), axis=axis)) / (2 * r + 1)).astype(np.float32)
    return out


def bake_strip(X, Y, Z, offs, s_rows, photos, ortho, size=(1024, 256), fade=(0.55, 0.8), consistency=18.0):
    """Texture of a verge strip given as a (rows, cols) grid of points.

    u runs along the rows (by distance s), v across by the 3D distance from
    the road edge (cumulative, normalised by the widest row). offs: offset of
    each column from the road edge (m), used to fade into the orthophoto
    between fade[0] and fade[1] of the strip width. Returns (image, uvs)."""
    R, C = X.shape
    seg = np.sqrt(np.diff(X, axis=1) ** 2 + np.diff(Y, axis=1) ** 2 + np.diff(Z, axis=1) ** 2)
    cum = np.concatenate([np.zeros((R, 1)), np.cumsum(seg, axis=1)], axis=1)
    vmax = max(float(cum[:, -1].max()), 1e-3)
    su = (s_rows - s_rows[0]) / max(s_rows[-1] - s_rows[0], 1e-3)
    uvs = np.column_stack([np.repeat(su, C), (cum / vmax).ravel()])
    w, h = size
    tu = (np.arange(w) + 0.5) / w
    tv = (np.arange(h) + 0.5) / h * vmax
    rowf = np.interp(tu, su, np.arange(R))
    r0 = np.clip(np.floor(rowf).astype(int), 0, R - 2)
    fr = (rowf - r0)[:, None]
    P = np.zeros((w, h, 3))
    off_t = np.zeros((w, h))
    valid = np.zeros((w, h), bool)
    for i in range(w):
        cr = cum[r0[i]] * (1 - fr[i]) + cum[r0[i] + 1] * fr[i]
        for k, A in enumerate((X, Y, Z)):
            row = A[r0[i]] * (1 - fr[i]) + A[r0[i] + 1] * fr[i]
            P[i, :, k] = np.interp(tv, cr, row)
        off_t[i] = np.interp(tv, cr, offs)
        valid[i] = tv <= cr[-1] + 1e-6
    pts = P.reshape(-1, 3)
    c1, w1, c2, w2 = photos.colours(pts)
    base = ortho.colours(pts[:, 0], pts[:, 1])
    # photo consistency: real ground looks the same from two camera positions;
    # walls, cars, trees projected on the ground do not (they smear differently)
    both = (w1 > 0) & (w2 > 0)
    g1, g2 = _blur(c1.reshape(w, h, 3), 4), _blur(c2.reshape(w, h, 3), 4)
    diff = np.abs(g1 - g2).mean(-1).ravel()
    consistent = both & (diff < consistency)
    # streaks (tall things smeared over the ground) are strongly oriented
    # texture: structure-tensor coherence on the photo colour
    lum = (c1 * w1[:, None] + c2 * w2[:, None]).sum(-1) / np.maximum(w1 + w2, 1e-6) / 3
    L = lum.reshape(w, h)
    gx, gy = np.gradient(L)
    jxx, jyy, jxy = (_blur(a_[..., None], 3)[..., 0] for a_ in (gx * gx, gy * gy, gx * gy))
    tr = jxx + jyy
    coh = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / np.maximum(tr, 1e-6)
    streak = ((coh > 0.75) & (tr > 40)).ravel()
    consistent &= ~streak
    consistent = _blur(consistent.reshape(w, h, 1).astype(np.float32), 3).ravel() > 0.7
    col = (c1 * w1[:, None] + c2 * w2[:, None]) / np.maximum(w1 + w2, 1e-6)[:, None]
    wsum = np.where(consistent, w1 + w2, 0)
    rel = off_t.ravel() / max(offs[-1], 1e-3)
    fade_w = np.clip((fade[1] - rel) / (fade[1] - fade[0]), 0, 1)
    wp = _blur((np.clip(wsum * 3, 0, 1) * fade_w).reshape(w, h, 1), 3).ravel()
    # match the photo brightness to the orthophoto (their light differs)
    m = consistent & (rel > 0.2) & (rel < 0.8)
    if m.sum() > 200:
        gain = np.clip(base[m].mean(0) / np.maximum(col[m].mean(0), 1), 0.55, 1.25)
        col *= gain[None, :]
    out = col * wp[:, None] + base * (1 - wp[:, None])
    img = np.clip(out, 0, 255).astype(np.uint8).reshape(w, h, 3).transpose(1, 0, 2)
    # texture rows: v = 0 (road edge) at the bottom of the image (Blender UV convention)
    img = img[::-1]
    return Image.fromarray(np.ascontiguousarray(img)), uvs, float(consistent.mean())
