"""Renders the Workshop poster (512x512), preview and icon (64x64) from the built models.

    blender --background --factory-startup --python blender/poster.py
"""
import os
import sys

import bpy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import OUT_MODELS, PROJECT  # noqa: E402
import preview  # noqa: E402

MOD_DIR = os.path.join(PROJECT, "Contents", "mods", "PZHorseManeAndTailImproved", "42")
WORK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_work")
FONT = r"C:\Windows\Fonts\arialbd.ttf"
TITLE = ["HORSE MANE & TAIL", "REWORK"]

FRAME = (0.69, 0.19, 0.16)
EDGE = (0.94, 0.88, 0.78)
PANEL = (0.16, 0.145, 0.135)
GOLD = (0.97, 0.76, 0.19)


def srgb_to_linear(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def render_rgba(path, size):
    scene = bpy.context.scene
    scene.render.resolution_x = scene.render.resolution_y = size
    scene.render.film_transparent = True
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    img = bpy.data.images.load(path, check_existing=False)
    px = np.array(img.pixels[:], np.float32).reshape(size, size, 4)
    bpy.data.images.remove(img)
    return px


def camera(loc, target, ortho):
    cam = bpy.data.cameras.new("cam")
    cam.type = "ORTHO"
    cam.ortho_scale = ortho
    obj = bpy.data.objects.new("cam", cam)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = loc
    obj.rotation_euler = preview._look_rotation(np.subtract(target, loc))
    bpy.context.scene.camera = obj
    return obj


def horse_layer(size, breed, loc, target, ortho):
    preview.build_scene(OUT_MODELS, "Mare", breed, "Horse_Idle.glb", 0.0)
    preview.setup_render((size, size))
    bpy.data.objects.remove(bpy.data.objects["ground"])
    camera(loc, target, ortho)
    return render_rgba(os.path.join(WORK, "poster_horse.png"), size)


def text_layer(size):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.view_settings.view_transform = "Standard"
    font = bpy.data.fonts.load(FONT)
    mat = bpy.data.materials.new("gold")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.remove(nt.nodes["Principled BSDF"])
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs[0].default_value = (*srgb_to_linear(GOLD), 1)
    nt.links.new(em.outputs[0], nt.nodes["Material Output"].inputs[0])
    for i, line in enumerate(TITLE):
        cu = bpy.data.curves.new(f"t{i}", "FONT")
        cu.body = line
        cu.font = font
        cu.align_x = "CENTER"
        cu.size = 0.082
        cu.offset = 0.0
        cu.materials.append(mat)
        obj = bpy.data.objects.new(f"t{i}", cu)
        obj.location = (0, -0.305 - 0.095 * i, 0)
        scene.collection.objects.link(obj)
    camera((0, 0, 5), (0, 0, 0), 1.0)
    return render_rgba(os.path.join(WORK, "poster_text.png"), size)


def over(dst, src):
    a = src[..., 3:4]
    dst[..., :3] = src[..., :3] * a + dst[..., :3] * (1 - a)
    return dst


def save(px, path):
    h, w = px.shape[:2]
    img = bpy.data.images.new(os.path.basename(path), w, h, alpha=True)
    img.pixels[:] = px.reshape(-1)
    img.filepath_raw = path
    img.file_format = "PNG"
    img.save()


def main():
    os.makedirs(WORK, exist_ok=True)
    size = 512
    breed = "AmericanQuarterPalomino"
    horse = horse_layer(size, breed, (-2.6, -1.5, 1.35), (0, 0.02, 0.66), 2.3)
    # icon: the head and mane up close, rendered at 4x and box-filtered down to 64
    head = horse_layer(256, breed, (-1.5, -1.3, 1.25), (0, -0.55, 1.0), 0.75)
    text = text_layer(size)

    # pixel values below are sRGB, like the rendered layers; rows are bottom-up in Blender images
    y, x = np.mgrid[0:size, 0:size]
    poster = np.ones((size, size, 4), np.float32)
    poster[..., :3] = FRAME
    inner = (x >= 22) & (x < size - 22) & (y >= 22) & (y < size - 22)
    poster[inner, :3] = EDGE
    panel = (x >= 28) & (x < size - 28) & (y >= 28) & (y < size - 28)
    poster[panel, :3] = PANEL
    # lift the horse into the upper part of the panel, clear of the title
    shifted = np.zeros_like(horse)
    shifted[70:, :] = horse[:-70, :]
    shifted[~panel] = 0
    poster = over(poster, shifted)
    poster = over(poster, text)
    poster[..., 3] = 1

    for path in (os.path.join(MOD_DIR, "poster.png"), os.path.join(PROJECT, "preview.png")):
        save(poster, path)
        print("wrote", path)

    icon = np.ones((256, 256, 4), np.float32)
    icon[..., :3] = PANEL
    icon = over(icon, head)
    icon = icon.reshape(64, 4, 64, 4, 4).mean(axis=(1, 3))
    icon[..., 3] = 1
    save(icon, os.path.join(MOD_DIR, "icon.png"))
    print("wrote", os.path.join(MOD_DIR, "icon.png"))


main()
