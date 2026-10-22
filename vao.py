#!/usr/bin/env python3
"""
Baked ambient occlusion for Custom Shaders Patch (VAO patch): darker at the
foot of the walls, under the trees, in the ditches and against the banks.

    python vao.py -c roads/<road>.yaml --install "<assettocorsa folder>"

Uses CustomTracksBakery (by the author of CSP, free; Windows), which is not
redistributed here: download it once and put CustomTracksBakery.exe and
SlimDX.dll in realroad/tools/ (see the README). The baker runs in the
installed track folder and the patch goes to
<assettocorsa>/extension/vao-patches/<track id>.vao-patch, where CSP loads
it. Without the baker this step says so and does nothing.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from fetch import load_cfg

# meshes that must not cast AO (glass, sky, distant relief) and the shaders
# whose normals are kept (trees: their own soft shading)
ARGS = [
    "--opacity=0.7",
    "--saturation-gain=1",
    "--multiplier=1.0",
    "--extra-pass",
    "--extra-pass-brightness-mult=1.6",
    "--include-kn5s=*",
    "--occluders-split-threshold=250",
    "--skip=backdrop_*, mid_*, far_*, marking_*",
    "--sync-normals=shader:ksTree, shader:ksGrass",
    "--queue-size=8000",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", default="config.yaml")
    ap.add_argument("--install", required=True, metavar="AC_ROOT")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    tid = (cfg.get("track") or {}).get("id")
    tools = Path(__file__).resolve().parent / "tools"
    exe = tools / "CustomTracksBakery.exe"
    if not exe.exists():
        print("No tools/CustomTracksBakery.exe: VAO patch skipped (see the README to get the baker).")
        return
    if sys.platform != "win32":
        print("CustomTracksBakery runs on Windows only: VAO patch skipped.")
        return
    track = Path(a.install) / "content" / "tracks" / tid
    if not (track / f"{tid}.kn5").exists():
        sys.exit(f"{track / (tid + '.kn5')} not found: install the track first.")
    for f in (exe, tools / "SlimDX.dll"):
        if f.exists():
            shutil.copy(f, track / f.name)
    (track / "Arguments.txt").write_text("\n".join(ARGS) + "\n", encoding="utf-8")
    before = set(track.glob("*.vao-patch"))
    print(f"Baking ambient occlusion for {tid} (a few minutes)…", flush=True)
    r = subprocess.run([str(track / exe.name), f"{tid}.kn5"], cwd=track)
    new = sorted(set(track.glob("*.vao-patch")) - before, key=lambda f: f.stat().st_mtime) or \
        sorted(track.glob("*.vao-patch"), key=lambda f: f.stat().st_mtime)
    for f in (track / exe.name, track / "SlimDX.dll", track / "Arguments.txt"):
        f.unlink(missing_ok=True)
    if r.returncode != 0 or not new:
        sys.exit(f"The baker failed (exit code {r.returncode}).")
    dest = Path(a.install) / "extension" / "vao-patches" / f"{tid}.vao-patch"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(new[-1]), dest)
    print(f"→ {dest}")


if __name__ == "__main__":
    main()
