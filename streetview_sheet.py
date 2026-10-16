#!/usr/bin/env python3
"""
Street-level comparison: what the road really looks like, every 250 m.

    python fetch.py streetview     # Panoramax 360° photos along the road
    python streetview_sheet.py     → data/<name>/streetview/front/*.jpg, streetview_sheet.jpg

Each 360° photo (equirectangular, centre column = direction of travel of the
camera car, view:azimuth) is turned into the view a driver has,
looking along the road (90° field of view, direction of travel of the track),
so it can be compared with the same place in the game. The sheet puts four
views per kilometre, labelled with the distance from the start.

Pictures: Panoramax (IGN and contributors), open licence (Etalab 2.0 for the
IGN ones). They stay in data/ (not in Git).
"""
import argparse
import json

import numpy as np
from PIL import Image, ImageDraw

from fetch import load_cfg


def front_view(pano, yaw_deg, hfov_deg=90.0, size=(960, 420), pitch_deg=2.0):
    """Perspective view out of an equirectangular panorama. yaw_deg is measured
    from the centre column of the panorama, positive to the right."""
    W, H = pano.size
    src = np.asarray(pano.convert("RGB"), np.float32)
    w, h = size
    f = (w / 2) / np.tan(np.radians(hfov_deg) / 2)
    xs, ys = np.meshgrid(np.arange(w) - w / 2 + 0.5, h / 2 - np.arange(h) - 0.5)
    # camera frame: x right, y up, z forward; pitch, then yaw
    p = np.radians(pitch_deg)
    y2 = ys * np.cos(p) + f * np.sin(p)
    z2 = -ys * np.sin(p) + f * np.cos(p)
    yaw = np.arctan2(xs, z2) + np.radians(yaw_deg)
    pitch = np.arctan2(y2, np.hypot(xs, z2))
    u = ((yaw / (2 * np.pi) + 0.5) % 1.0) * W - 0.5
    v = (0.5 - pitch / np.pi) * H - 0.5
    u0, v0 = np.floor(u).astype(int), np.clip(np.floor(v).astype(int), 0, H - 2)
    fu, fv = (u - u0)[..., None], np.clip(v - v0, 0, 1)[..., None]
    u1 = (u0 + 1) % W
    u0 %= W
    out = (src[v0, u0] * (1 - fu) * (1 - fv) + src[v0, u1] * fu * (1 - fv)
           + src[v0 + 1, u0] * (1 - fu) * fv + src[v0 + 1, u1] * fu * fv)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def road_heading(s_m, prof):
    """Heading (degrees from north, clockwise) of the road at distance s."""
    k = int(np.clip(np.searchsorted(prof["s_m"], s_m), 2, len(prof) - 3))
    k0, k1 = max(0, k - 10), min(len(prof) - 1, k + 10)
    dx, dy = prof["x"][k1] - prof["x"][k0], prof["y"][k1] - prof["y"][k0]
    return float(np.degrees(np.arctan2(dx, dy)) % 360)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="config.yaml")
    ap.add_argument("--every", type=float, default=250.0, help="metres between views on the sheet")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    d = cfg["_dir"]
    folder = d / "streetview"
    idx_f = folder / "index.json"
    if not idx_f.exists():
        raise SystemExit("No street-level photos: run `python fetch.py streetview` first.")
    idx = [p for p in json.loads(idx_f.read_text(encoding="utf-8")) if p.get("file") and (folder / p["file"]).exists()]
    if not idx:
        raise SystemExit("The street-level photos are not downloaded yet: run `python fetch.py streetview`.")
    prof = np.genfromtxt(d / "road_profile.csv", delimiter=",", names=True)
    front = folder / "front"
    front.mkdir(exist_ok=True)
    s_all = np.array([p["s_m"] for p in idx])
    wanted = np.arange(0, prof["s_m"][-1], a.every)
    tiles = []
    for sv in wanted:
        k = int(np.argmin(np.abs(s_all - sv)))
        if abs(s_all[k] - sv) > a.every / 2:
            continue
        p = idx[k]
        out = front / f"{int(p['s_m']):06d}.jpg"
        if not out.exists():
            az = p.get("azimuth")
            az = float(az) if az is not None else 0.0
            yaw = (road_heading(p["s_m"], prof) - az + 540) % 360 - 180
            with Image.open(folder / p["file"]) as pano:
                if pano.size[0] < 2 * pano.size[1] * 0.9:   # not a 360° picture: use as is
                    view = pano.convert("RGB").resize((960, 420))
                else:
                    view = front_view(pano, yaw)
            view.save(out, quality=88)
        tiles.append((p["s_m"], out, (p.get("datetime") or "")[:10]))
    # contact sheet: 4 views per row
    cw, ch, cols = 480, 210, 4
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cw, rows * (ch + 22)), (20, 20, 20))
    dr = ImageDraw.Draw(sheet)
    for i, (sv, f, date) in enumerate(tiles):
        x, y = (i % cols) * cw, (i // cols) * (ch + 22)
        with Image.open(f) as im:
            sheet.paste(im.resize((cw, ch)), (x, y + 22))
        dr.text((x + 6, y + 4), f"km {sv / 1000:.2f}   {date}", fill=(235, 235, 235))
    sheet.save(d / "streetview_sheet.jpg", quality=85)
    print(f"{len(tiles)} driver views (one every {a.every:.0f} m) → {front}\n→ {d / 'streetview_sheet.jpg'}")


if __name__ == "__main__":
    main()
