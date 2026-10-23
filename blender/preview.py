"""Renders a posed horse with its mane pieces, the way the game assembles it.

    blender --background --factory-startup --python blender/preview.py -- \
        --models <models_X/HorseMod folder> --out <image.png> [--body Mare] [--breed AmericanQuarterPalomino]
        [--anim Horse_Idle.glb --time 0] [--views side_r,side_l,rear,iso,neck,tail]
"""
import argparse
import os
import sys

import bmesh
import bpy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import glb  # noqa: E402
from common import (Body, Pose, HAIR_TEXTURE, HORSE_MANE_FOR_BONE, HORSE_MOD, HORSE_TEXTURES, MANE_BONES,  # noqa: E402
                    MANE_FOR_BONE, MANE_HEX)

# glTF (+Y up, +Z forward) -> Blender (+Z up, -Y forward)
G2B = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float)
# mane attachment offsets from HorseMod's mare.txt / stallion.txt (bone-local)
ATTACH_OFFSET = {"DEF_ManeEnd": (0.0, -0.0141, -0.0007)}

VIEWS = {
    # name: (camera location, look-at target, orthographic scale or None for perspective lens 50mm)
    "side_r": ((-3.2, -0.05, 0.75), (0, -0.05, 0.62), 2.05),
    "side_l": ((3.2, -0.05, 0.75), (0, -0.05, 0.62), 2.05),
    "rear": ((-1.5, 2.2, 1.2), (0, 0.15, 0.55), 1.9),
    "iso": ((-2.2, -2.2, 2.0), (0, -0.05, 0.5), 2.0),
    "neck": ((-1.6, -0.9, 1.25), (0, -0.42, 0.95), 0.85),
    "tail": ((-1.3, 1.6, 0.9), (0, 0.5, 0.58), 0.95),
    "back": ((0, 2.6, 0.75), (0, 0.45, 0.6), 1.1),
    "tail_side": ((-2.4, 0.45, 0.6), (0, 0.45, 0.58), 0.85),
    "neck_l": ((1.6, -0.9, 1.25), (0, -0.42, 0.95), 0.85),
    "face": ((-0.5, -2.4, 1.25), (0, -0.7, 1.0), 0.7),
    "head_low": ((-1.3, -1.2, 0.75), (0, -0.62, 0.38), 0.75),  # the head down while eating
    "poll": ((-1.4, -0.75, 1.2), (0, -0.62, 1.03), 0.38),
}


def hex_to_linear(h):
    c = np.array([int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)])
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def material(name, image, tint=None):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 0.85
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(image, check_existing=True)
    if tint is None:
        nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    else:
        mix = nt.nodes.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        mix.blend_type = "MULTIPLY"
        mix.inputs[0].default_value = 1.0
        mix.inputs[7].default_value = (*tint, 1.0)
        nt.links.new(tex.outputs["Color"], mix.inputs[6])
        nt.links.new(mix.outputs[2], bsdf.inputs["Base Color"])
    mat.use_backface_culling = True
    return mat


def add_mesh(name, positions, tris, uvs, mat):
    me = bpy.data.meshes.new(name)
    pb = (np.c_[positions, np.ones(len(positions))] @ G2B.T)[:, :3]
    me.from_pydata(pb.tolist(), [], tris.tolist())
    uvl = me.uv_layers.new(name="UV")
    loop_vi = np.zeros(len(me.loops), int)
    me.loops.foreach_get("vertex_index", loop_vi)
    uvb = np.c_[uvs[:, 0], 1.0 - uvs[:, 1]]
    uvl.data.foreach_set("uv", uvb[loop_vi].reshape(-1))
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
    bm.to_mesh(me)
    bm.free()
    me.shade_smooth()
    me.materials.append(mat)
    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def build_scene(models, body_name, breed, anim, time):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    foal = body_name.startswith("Foal")
    body_dir = os.path.join(HORSE_MOD, "common", "media", "models_X", "HorseMod", "Foal") if foal else os.path.join(models, "Horse")
    body = Body(os.path.join(body_dir, body_name + ".glb"))
    pose = Pose(body, anim, time) if anim else Pose(body)
    a = body.attrs
    posed = pose.skin(a["POSITION"].astype(float), a["JOINTS_0"], a["WEIGHTS_0"])
    body_mat = material("body", os.path.join(HORSE_TEXTURES, f"Horse_{breed}.png"))
    add_mesh("body", posed, body.indices.reshape(-1, 3), a["TEXCOORD_0"], body_mat)

    # this mod keeps one mane mesh per bone and its texture next to the HorseMod folder;
    # the Horse mod shares one mesh between the middle bones and uses the vanilla hair texture
    our_manes = os.path.abspath(os.path.join(models, "..", "HorseManeTailImproved"))
    ours = os.path.isdir(our_manes) and not foal
    if ours:
        mane_dir, mane_names = our_manes, MANE_FOR_BONE
        mane_tex = os.path.abspath(os.path.join(models, "..", "..", "textures", "HorseManeTailImproved", "ManeStrands.png"))
    else:
        mane_dir, mane_names, mane_tex = models, HORSE_MANE_FOR_BONE, HAIR_TEXTURE
        if foal:  # foals always get the Horse mod's manes
            mane_dir = os.path.join(HORSE_MOD, "common", "media", "models_X", "HorseMod")
    mane_mat = material("mane", mane_tex, tint=hex_to_linear(MANE_HEX[breed]))
    for bone in MANE_BONES:
        g, b = glb.read(os.path.join(mane_dir, mane_names[bone] + ".glb"))
        pa, idx = glb.primitive_arrays(g, b)
        local = np.eye(4)
        local[:3, 3] = ATTACH_OFFSET.get(bone, (0, 0, 0))
        M = pose.mats[bone] @ local
        world = (np.c_[pa["POSITION"], np.ones(len(pa["POSITION"]))] @ M.T)[:, :3]
        add_mesh(bone, world, idx.reshape(-1, 3), pa["TEXCOORD_0"], mane_mat)
    # in-game root motion can move the horse; centre it under the cameras
    lowest = posed[:, 1].min()
    for obj in bpy.context.scene.objects:
        obj.location.z -= lowest


def setup_render(res):
    scene = bpy.context.scene
    for engine in ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT", "BLENDER_WORKBENCH"):
        try:
            scene.render.engine = engine
            break
        except TypeError:
            continue
    scene.render.resolution_x, scene.render.resolution_y = res
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Standard"
    world = bpy.data.worlds.new("w")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (0.55, 0.58, 0.62, 1)
    world.node_tree.nodes["Background"].inputs[1].default_value = 0.9
    scene.world = world
    sun = bpy.data.lights.new("sun", "SUN")
    sun.energy = 3.2
    sun_obj = bpy.data.objects.new("sun", sun)
    sun_obj.rotation_euler = (np.radians(50), np.radians(10), np.radians(-35))
    scene.collection.objects.link(sun_obj)
    ground = bpy.data.meshes.new("ground")
    ground.from_pydata([(-6, -6, 0), (6, -6, 0), (6, 6, 0), (-6, 6, 0)], [], [(0, 1, 2, 3)])
    gmat = bpy.data.materials.new("ground")
    gmat.use_nodes = True
    gmat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.32, 0.36, 0.28, 1)
    ground.materials.append(gmat)
    scene.collection.objects.link(bpy.data.objects.new("ground", ground))


def render_view(name, path):
    scene = bpy.context.scene
    loc, target, ortho = VIEWS[name]
    cam = bpy.data.cameras.new(name)
    cam.type = "ORTHO"
    cam.ortho_scale = ortho
    cam.clip_end = 50
    obj = bpy.data.objects.new(name, cam)
    scene.collection.objects.link(obj)
    obj.location = loc
    direction = np.subtract(target, loc)
    obj.rotation_euler = _look_rotation(direction)
    scene.camera = obj
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)


def _look_rotation(direction):
    from mathutils import Vector
    return Vector(direction).to_track_quat("-Z", "Y").to_euler()


def compose(paths, cols, out):
    imgs = [bpy.data.images.load(p) for p in paths]
    w, h = imgs[0].size
    rows = (len(imgs) + cols - 1) // cols
    sheet = np.ones((rows * h, cols * w, 4), np.float32)
    for i, img in enumerate(imgs):
        px = np.array(img.pixels[:], np.float32).reshape(h, w, 4)
        r, c = divmod(i, cols)
        sheet[(rows - 1 - r) * h:(rows - r) * h, c * w:(c + 1) * w] = px
    out_img = bpy.data.images.new("sheet", cols * w, rows * h, alpha=True)
    out_img.pixels[:] = sheet.reshape(-1)
    out_img.filepath_raw = out
    out_img.file_format = "PNG"
    out_img.save()


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--body", default="Mare")
    ap.add_argument("--breed", default="AmericanQuarterPalomino")
    ap.add_argument("--anim", default="Horse_Idle.glb")
    ap.add_argument("--time", type=float, default=0.0)
    ap.add_argument("--views", default="side_r,side_l,rear,iso,neck,tail")
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--res", default="640x480")
    ap.add_argument("--compare", help="models folder to render as a 'before' row above --models")
    args = ap.parse_args(argv)
    res = tuple(int(v) for v in args.res.split("x"))
    anim = args.anim if args.anim != "none" else None
    views = args.views.split(",")
    tmp = os.path.splitext(args.out)[0] + "_views"
    os.makedirs(tmp, exist_ok=True)
    paths = []
    for tag, models in ([("before", args.compare)] if args.compare else []) + [("after", args.models)]:
        build_scene(models, args.body, args.breed, anim, args.time)
        setup_render(res)
        for v in views:
            p = os.path.join(tmp, f"{tag}_{v}.png")
            render_view(v, p)
            paths.append(p)
    compose(paths, len(views) if args.compare else args.cols, args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
