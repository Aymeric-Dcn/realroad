"""
Blender side of the pipeline: meshes.npz + manifest.json → FBX for ksEditor.

Run headless (no Blender UI needed):

    blender --factory-startup -b -P blender_export.py -- data/<name>/track_build

Writes <track_id>.fbx (+ textures copied next to it) and track.blend (to look
at the result in Blender) into the same folder. Uses only Blender's bundled
Python and numpy.
"""
import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector


def args():
    if "--" not in sys.argv:
        raise SystemExit("usage: blender -b -P blender_export.py -- <track_build folder> [--spawn-flip]")
    rest = sys.argv[sys.argv.index("--") + 1:]
    return Path(rest[0]).resolve(), "--spawn-flip" in rest


def make_material(name, texture_path):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    img = bpy.data.images.load(str(texture_path))
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = img
    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    if "Roughness" in bsdf.inputs:
        bsdf.inputs["Roughness"].default_value = 0.9
    return mat


def make_mesh(name, verts, faces, uvs, material, smooth):
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts.tolist(), [], faces.tolist())
    uv = me.uv_layers.new(name="UVMap")
    # from_pydata creates loops in face order: loop i ↔ faces.ravel()[i]
    uv.data.foreach_set("uv", uvs[faces.ravel()].ravel())
    me.polygons.foreach_set("use_smooth", [bool(smooth)] * len(me.polygons))
    me.materials.append(material)
    me.validate()
    me.update()
    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)
    if obj.name != name:
        raise RuntimeError(f"Blender renamed {name} to {obj.name}: names must be unique")
    return obj


def make_dummy(d, flip):
    """AC places the car along the dummy's +Z axis (FBX/AC space). With the
    FBX export below (Blender -Y → FBX +Z, Blender Z → FBX Y), that is the
    empty's local -Y in Blender. --spawn-flip uses +Y instead."""
    e = bpy.data.objects.new(d["name"], None)
    e.empty_display_type = "ARROWS"
    e.empty_display_size = 2.0
    fwd = Vector(d["fwd"]).normalized()
    up = Vector((0.0, 0.0, 1.0))
    y_axis = fwd if flip else -fwd
    x_axis = y_axis.cross(up).normalized()
    rot = Matrix((x_axis, y_axis, up)).transposed()  # columns = local axes
    e.matrix_world = Matrix.Translation(Vector(d["pos"])) @ rot.to_4x4()
    bpy.context.scene.collection.objects.link(e)
    if e.name != d["name"]:
        raise RuntimeError(f"Blender renamed {d['name']} to {e.name}")
    return e


def main():
    build, flip = args()
    man = json.loads((build / "manifest.json").read_text(encoding="utf-8"))
    data = np.load(build / "meshes.npz")
    bpy.ops.wm.read_factory_settings(use_empty=True)

    mats = {m: make_material(m, build / "texture" / (f["texture"] if isinstance(f, dict) else f))
            for m, f in man["materials"].items()}
    for i, o in enumerate(man["objects"], 1):
        k = o["key"]
        make_mesh(o["name"], data[f"v{k}"], data[f"f{k}"], data[f"uv{k}"], mats[o["material"]], o["smooth"])
        if i % 50 == 0:
            print(f"  {i}/{len(man['objects'])} meshes")
    for d in man["dummies"]:
        make_dummy(d, flip)

    fbx = build / f"{man['track_id']}.fbx"
    bpy.ops.export_scene.fbx(
        filepath=str(fbx),
        use_selection=False,
        object_types={"MESH", "EMPTY"},
        apply_unit_scale=True,
        apply_scale_options="FBX_SCALE_ALL",
        axis_forward="-Z",
        axis_up="Y",
        bake_space_transform=True,
        use_mesh_modifiers=False,
        mesh_smooth_type="FACE",
        add_leaf_bones=False,
        path_mode="COPY",
        embed_textures=False,
    )
    bpy.ops.wm.save_as_mainfile(filepath=str(build / "track.blend"))
    print(f"→ {fbx}")
    print(f"→ {build / 'track.blend'}")


main()
