"""
Blender script: turn the tree models of trees3d.py into one FBX per species,
ready for Content Manager's CSP tree converter.

    blender -b -P blender_trees.py -- data/<name>/csp_trees/src

For each species: the three OBJ levels of detail (LOD_A, LOD_B, LOD_C) in one
scene, pivot at the middle of the base, materials with diffuse (alpha =
cut-out for the leaves) and normal textures, textures embedded in the FBX.
Written next to the src folder: data/<name>/csp_trees/<species>.fbx
"""
import os
import sys

import bpy

SPECIES = ["pine_a", "pine_b", "oak", "shrub"]
LODS = ["A", "B", "C"]


def material(name, src, diffuse, normal, alpha):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(os.path.join(src, diffuse), check_existing=True)
    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    if alpha:
        nt.links.new(tex.outputs["Alpha"], bsdf.inputs["Alpha"])
        for attr, val in (("blend_method", "CLIP"), ("surface_render_method", "DITHERED")):
            try:
                setattr(m, attr, val)
            except (AttributeError, TypeError):
                pass
    ntex = nt.nodes.new("ShaderNodeTexImage")
    ntex.image = bpy.data.images.load(os.path.join(src, normal), check_existing=True)
    try:
        ntex.image.colorspace_settings.name = "Non-Color"
    except TypeError:
        pass
    nmap = nt.nodes.new("ShaderNodeNormalMap")
    nt.links.new(ntex.outputs["Color"], nmap.inputs["Color"])
    nt.links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])
    return m


def import_obj(path):
    before = set(bpy.data.objects)
    if hasattr(bpy.ops.wm, "obj_import"):
        bpy.ops.wm.obj_import(filepath=path, forward_axis="Y", up_axis="Z")
    else:                                   # Blender < 3.2
        bpy.ops.import_scene.obj(filepath=path, axis_forward="Y", axis_up="Z")
    return [o for o in bpy.data.objects if o not in before]


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if not argv:
        raise SystemExit("usage: blender -b -P blender_trees.py -- <csp_trees/src folder>")
    src = os.path.abspath(argv[0])
    dst = os.path.dirname(src)
    for sp in SPECIES:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bark = material("bark", src, "bark.png", "bark_normal.png", alpha=False)
        leaves = material("leaves", src, "leaves.png", "leaves_normal.png", alpha=True)
        objs = []
        for lod in LODS:
            path = os.path.join(src, f"{sp}_LOD_{lod}.obj")
            if not os.path.exists(path):
                continue
            new = [o for o in import_obj(path) if o.type == "MESH"]
            if not new:
                continue
            o = new[0]
            o.name = o.data.name = f"{sp}_LOD_{lod}"
            for slot in o.material_slots:
                nm = (slot.material.name if slot.material else "").lower()
                slot.material = leaves if nm.startswith("leaves") else bark
            o.location = (0.0, 0.0, 0.0)
            objs.append(o)
        if not objs:
            print(f"! {sp}: no OBJ found in {src}")
            continue
        bpy.ops.object.select_all(action="DESELECT")
        for o in objs:
            o.select_set(True)
        bpy.context.view_layer.objects.active = objs[0]
        out = os.path.join(dst, f"{sp}.fbx")
        bpy.ops.export_scene.fbx(filepath=out, use_selection=True, path_mode="COPY", embed_textures=True,
                                 axis_forward="-Z", axis_up="Y", apply_scale_options="FBX_SCALE_UNITS",
                                 mesh_smooth_type="OFF", object_types={"MESH"})
        print(f"→ {out} ({len(objs)} LODs)")


main()
