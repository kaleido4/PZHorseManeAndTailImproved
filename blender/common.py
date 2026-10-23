"""Shared paths, constants and helpers for the build and preview scripts.

All geometry is in glTF space: +Y up, +Z towards the head, +X the horse's left.
"""
import os

import numpy as np

import glb
from anim import Animated

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
REFERENCE = os.path.join(HERE, "reference")
MOD_MEDIA = os.path.join(PROJECT, "Contents", "mods", "PZHorseManeAndTailImproved", "common", "media")
OUT_MODELS = os.path.join(MOD_MEDIA, "models_X", "HorseMod")

# The Horse Mod install is only read from, never written to.
HORSE_MOD = os.environ.get("HORSE_MOD_DIR", r"D:\Games\Steam\steamapps\workshop\content\108600\3661336777\mods\HorseMod")
PZ_DIR = os.environ.get("PZ_DIR", r"D:\Games\Steam\steamapps\common\ProjectZomboid")
HORSE_ANIMS = os.path.join(HORSE_MOD, "common", "media", "anims_X", "Horse")
HORSE_TEXTURES = os.path.join(HORSE_MOD, "common", "media", "textures", "Body", "HorseMod")
HAIR_TEXTURE = os.path.join(PZ_DIR, "media", "textures", "F_Hair_White.png")

BODIES = ["Mare", "Mare_NoHead", "Stallion", "Stallion_NoHead"]
TAIL_BONES = ["DEF_Tail1", "DEF_Tail2", "DEF_Tail3"]
MANE_BONES = ["DEF_ManeStart", "DEF_ManeMid1", "DEF_ManeMid2", "DEF_ManeMid3", "DEF_ManeMid4", "DEF_ManeMid5", "DEF_ManeEnd"]
# the Horse mod's mane meshes (models_X/HorseMod), one shared by the five middle bones
HORSE_MANE_FOR_BONE = {
    "DEF_ManeStart": "SimpleManeStart",
    "DEF_ManeMid1": "SimpleManeMid",
    "DEF_ManeMid2": "SimpleManeMid",
    "DEF_ManeMid3": "SimpleManeMid",
    "DEF_ManeMid4": "SimpleManeMid",
    "DEF_ManeMid5": "SimpleManeMid",
    "DEF_ManeEnd": "SimpleManeEnd",
}
# this mod's mane meshes (models_X/HorseManeTailImproved), one per bone; named after the
# Horse mod's attachment slots, which lua/client/HorseManeTailImproved/ManeModels.lua maps
MANE_FOR_BONE = {bone: bone[len("DEF_"):] for bone in MANE_BONES}
OUT_MANES = os.path.join(MOD_MEDIA, "models_X", "HorseManeTailImproved")
# Mane colors from HorseMod/attachments/AttachmentData.lua, used for previews only.
MANE_HEX = {
    "AmericanQuarterPalomino": "#EADAB6",
    "AmericanQuarterBlueRoan": "#19191C",
    "AmericanQuarterStrawberryRoan": "#78382B",
    "AmericanQuarterSealBay": "#1F1819",
    "AmericanPaintTobiano": "#FBDEA7",
    "AmericanPaintOvero": "#292524",
    "AppaloosaGrullaBlanket": "#24201D",
    "AppaloosaLeopard": "#FFF7E4",
    "ThoroughbredBay": "#140C08",
    "ThoroughbredFleaBittenGrey": "#FCECC5",
    "ThoroughbredCricket": "#3C1C16",
    "ThoroughbredReverseBrindle": "#1F1819",
    "MustangBuckskinDun": "#33271A",
    "MustangGrullaDun": "#1A1916",
    "CamarilloWhite": "#FFF4CC",
    "RockyMountainLiverChestnut": "#A89F82",
}


def ensure_reference():
    """Copy the original body meshes out of the Horse mod install if they are not here yet."""
    import shutil
    src = os.path.join(HORSE_MOD, "common", "media", "models_X", "HorseMod", "Horse")
    os.makedirs(REFERENCE, exist_ok=True)
    for name in BODIES:
        dst = os.path.join(REFERENCE, name + ".glb")
        if not os.path.exists(dst):
            shutil.copyfile(os.path.join(src, name + ".glb"), dst)


class Body:
    """A skinned horse body mesh read from a GLB."""

    def __init__(self, path):
        self.path = path
        self.gltf, self.bin = glb.read(path)
        self.attrs, self.indices = glb.primitive_arrays(self.gltf, self.bin)
        skin = self.gltf["skins"][0]
        self.joint_nodes = skin["joints"]
        self.joint_names = [self.gltf["nodes"][j]["name"] for j in self.joint_nodes]
        ibm = glb.accessor(self.gltf, self.bin, skin["inverseBindMatrices"])
        self.ibm = ibm.reshape(-1, 4, 4).transpose(0, 2, 1).astype(np.float64)

    def joint(self, name):
        return self.joint_names.index(name)

    def bind_matrix(self, name):
        return np.linalg.inv(self.ibm[self.joint(name)])

    def weight_on(self, names):
        J = self.attrs["JOINTS_0"].astype(int)
        W = self.attrs["WEIGHTS_0"]
        return sum((W * (J == self.joint(n))).sum(axis=1) for n in names)


class Pose:
    """Joint world matrices for one animation frame (or the bind pose when anim is None)."""

    def __init__(self, body, anim_file=None, time=0.0):
        self.body = body
        if anim_file is None:
            self.mats = {n: body.bind_matrix(n) for n in body.joint_names}
            return
        an = Animated(os.path.join(HORSE_ANIMS, anim_file))
        _, chans, end = an.anims[0]
        t = min(time, end)
        # 'Bip01' appears twice; the joint is the one parenting DEF_Spine1.
        by_name = {}
        for i, n in enumerate(an.g["nodes"]):
            nm = n.get("name")
            if nm == "Bip01" and an.names["DEF_Spine1"] not in n.get("children", []):
                continue
            by_name.setdefault(nm, i)
        cache = {}
        self.mats = {n: an.world(by_name[n], chans, t, cache) for n in body.joint_names}

    def skin(self, positions, joints, weights):
        """Linear blend skinning of mesh-space positions."""
        skin_mats = np.stack([self.mats[n] @ self.body.ibm[i] for i, n in enumerate(self.body.joint_names)])
        hom = np.c_[positions, np.ones(len(positions))]
        out = np.zeros((len(positions), 3))
        for k in range(joints.shape[1]):
            m = skin_mats[joints[:, k].astype(int)]
            out += weights[:, k:k + 1] * np.einsum("nij,nj->ni", m, hom)[:, :3]
        return out


def unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n == 0, 1, n)


def smooth_normals(positions, tris):
    """Area-weighted vertex normals; vertices at the same position share a normal."""
    p = positions[tris]
    fn = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    key = np.round(positions, 5)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    acc = np.zeros((inv.max() + 1, 3))
    for c in range(3):
        np.add.at(acc, inv[tris[:, c]], fn)
    return unit(acc[inv]).astype(np.float32)
