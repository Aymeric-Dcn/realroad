#!/usr/bin/env python3
"""
Free (CC0) textures for the graphics, downloaded once and shared by all roads.

    python assets.py          → data/_shared/assets/<id>/...  (~60 MB, skipped when there)

- ambientCG (https://ambientcg.com, CC0): asphalt, rock faces, grass, gravel,
  forest floor. Photographed, tileable, with normal and roughness maps. Used
  by textures.py for the road surface and the detail layers of the terrain
  and rock shaders.
- Poly Haven (https://polyhaven.com, CC0): bark and twig / leaf sprites of
  photographed trees (fir, pine, a broadleaf, a shrub). Used by trees3d.py
  for the CSP 3D trees and by textures.py for the flat tree billboards.

CC0: no attribution required; credited anyway in data/_shared/assets/CREDITS.txt
and in the README. Everything still works without them (procedural textures).
"""
import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import requests

UA = {"User-Agent": "realroad (open-data track generator)"}

# ambientCG material id → role. 2K for what is seen up close, 1K elsewhere.
AMBIENTCG = {
    "Road012A": ("road", "2K"),          # worn grey asphalt, no markings
    "Asphalt031": ("road_alt", "2K"),    # lighter, fine grain (dry Provence roads)
    "Rock051": ("rock", "2K"),           # cliff / road cut
    "Rock058": ("rock_grey", "1K"),
    "Grass004": ("grass", "1K"),
    "Ground037": ("grass_damp", "1K"),   # damp forest grass (Vosges)
    "Ground110": ("gravel", "1K"),       # dry grey-brown gravel
    "Ground078": ("forest_floor", "1K"),
    # villages: house renders, stone, barns and sheds, roofs, pavements
    "Plaster002": ("render_rough", "1K"),
    "Plaster003": ("render", "1K"),
    "PaintedPlaster017": ("render_white", "1K"),
    "Bricks066": ("stone_wall", "1K"),
    "Bricks099": ("sandstone_wall", "1K"),   # churches, Vosges sandstone
    "Planks021": ("barn_wood", "1K"),
    "CorrugatedSteel005": ("shed_metal", "1K"),
    "RoofingTiles013A": ("roof_clay", "1K"),
    "RoofingTiles010": ("roof_grey", "1K"),
    "RoofingTiles009": ("roof_dark", "1K"),
    "Concrete034": ("pavement", "1K"),
    "PavingStones142": ("paving", "1K"),
}
# Poly Haven model id → texture maps to fetch (2K JPG)
POLYHAVEN = {
    "fir_tree_01": ["twig_diff", "twig_alpha", "twig_nor_gl", "bark_diff", "bark_nor_gl"],
    "pine_tree_01": ["twig_diff", "twig_alpha", "twig_nor_gl", "bark_diff", "bark_nor_gl"],
    "tree_small_02": ["leaves_diff", "leaves_alpha", "leaves_nor_gl", "Diffuse", "nor_gl"],
    "shrub_02": ["Diffuse", "Alpha", "nor_gl"],
}
KEEP = ("_Color.", "_NormalGL.", "_Roughness.", "_AmbientOcclusion.", "_Opacity.")


def shared_dir(data_dir="data"):
    return Path(data_dir) / "_shared" / "assets"


def get(url, **kw):
    r = requests.get(url, headers=UA, timeout=120, **kw)
    r.raise_for_status()
    return r


def fetch_ambientcg(dest, aid, res):
    out = dest / aid
    if out.exists() and any(out.glob("*_Color.jpg")):
        return False
    z = zipfile.ZipFile(io.BytesIO(get(f"https://ambientcg.com/get?file={aid}_{res}-JPG.zip").content))
    out.mkdir(parents=True, exist_ok=True)
    for n in z.namelist():
        if any(k in n for k in KEEP) and n.lower().endswith((".jpg", ".png")):
            (out / Path(n).name).write_bytes(z.read(n))
    return True


def fetch_polyhaven(dest, pid, maps):
    out = dest / pid
    files = get(f"https://api.polyhaven.com/files/{pid}").json()
    new = False
    for m in maps:
        f = out / f"{m}.jpg"
        if f.exists() and f.stat().st_size:
            continue
        entry = files.get(m, {}).get("2k", {}).get("jpg")
        if not entry:
            print(f"  ! {pid}: no 2K JPG for {m}")
            continue
        out.mkdir(parents=True, exist_ok=True)
        f.write_bytes(get(entry["url"]).content)
        new = True
    return new


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", default=None, help="(ignored: the assets are shared by all roads)")
    ap.add_argument("--data-dir", default="data")
    a = ap.parse_args()
    dest = shared_dir(a.data_dir)
    dest.mkdir(parents=True, exist_ok=True)
    failed = []
    for aid, (role, res) in AMBIENTCG.items():
        try:
            print(f"  ambientCG {aid} ({role}): {'downloaded' if fetch_ambientcg(dest, aid, res) else 'already there'}")
        except (requests.RequestException, zipfile.BadZipFile) as e:
            failed.append(aid)
            print(f"  ! ambientCG {aid}: {e}")
    for pid, maps in POLYHAVEN.items():
        try:
            print(f"  Poly Haven {pid}: {'downloaded' if fetch_polyhaven(dest, pid, maps) else 'already there'}")
        except requests.RequestException as e:
            failed.append(pid)
            print(f"  ! Poly Haven {pid}: {e}")
    (dest / "CREDITS.txt").write_text(
        "CC0 assets used by realroad (no attribution required, credited with thanks):\n"
        + "".join(f"- {aid}: ambientCG, https://ambientcg.com/view?id={aid}\n" for aid in AMBIENTCG)
        + "".join(f"- {pid}: Poly Haven, https://polyhaven.com/a/{pid}\n" for pid in POLYHAVEN),
        encoding="utf-8")
    (dest / "index.json").write_text(json.dumps({"ambientcg": {k: v[0] for k, v in AMBIENTCG.items()},
                                                  "polyhaven": POLYHAVEN}, indent=1), encoding="utf-8")
    size = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file()) / 1e6
    print(f"→ {dest} ({size:.0f} MB)")
    if failed:
        sys.exit(f"{len(failed)} asset(s) failed: {', '.join(failed)} (procedural textures used instead)")


if __name__ == "__main__":
    main()
