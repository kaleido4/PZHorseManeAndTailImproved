"""Sanity checks for the built models. Exits non-zero on any failure.

    "<Blender>/5.2/python/bin/python.exe" blender/verify.py
"""
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import glb  # noqa: E402
from common import BODIES, MANE_FOR_BONE, MOD_MEDIA, OUT_MANES, OUT_MODELS, REFERENCE, TAIL_BONES, Body  # noqa: E402
from tail import TailIsland, UV_V  # noqa: E402

failures = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)


def open_edge_count(positions, tris):
    _, weld = np.unique(np.round(positions, 5), axis=0, return_inverse=True)
    weld = weld.reshape(-1)
    count = Counter()
    for t in tris:
        for i in range(3):
            a, b = weld[t[i]], weld[t[(i + 1) % 3]]
            count[(min(a, b), max(a, b))] += 1
    return sum(1 for n in count.values() if n == 1)


def verify_body(name):
    print(name)
    ref = Body(os.path.join(REFERENCE, name + ".glb"))
    new = Body(os.path.join(OUT_MODELS, "Horse", name + ".glb"))
    a, tris = new.attrs, new.indices.reshape(-1, 3)
    n = len(a["POSITION"])
    check(all(len(v) == n for v in a.values()), "all vertex attributes have the same count")
    check(tris.max() < n and tris.min() >= 0, "indices in range")
    check(new.joint_names == ref.joint_names, "joint list unchanged")
    check(np.allclose(new.ibm, ref.ibm, atol=1e-6), "inverse bind matrices unchanged")
    check([nd.get("name") for nd in new.gltf["nodes"]] == [nd.get("name") for nd in ref.gltf["nodes"]], "node hierarchy unchanged")
    w = a["WEIGHTS_0"]
    check(np.allclose(w.sum(axis=1), 1, atol=1e-4) and (w >= 0).all(), "skin weights are normalised")
    check(a["JOINTS_0"].max() < len(new.joint_names), "joint indices in range")
    check(not np.isnan(a["POSITION"]).any() and not np.isnan(a["NORMAL"]).any(), "no NaN positions or normals")

    # new vertices (tail and forelock): everything that was not in the reference
    ref_keys = {tuple(p) for p in np.round(ref.attrs["POSITION"], 5)}
    is_new = np.array([tuple(p) not in ref_keys for p in np.round(a["POSITION"], 5)])
    island = TailIsland(ref)
    uv = a["TEXCOORD_0"][is_new]
    inside = []
    for u, v in uv:
        row = int(v * island.R)
        inside.append(UV_V[0] - 1e-6 <= v <= UV_V[1] + 1e-6 and any(l <= u <= r for l, r in island.edges.get(row, [])))
    check(all(inside), f"all {len(uv)} new hair UVs land inside the tail strips")
    head_hair = is_new & (new.weight_on(TAIL_BONES) == 0)
    if "NoHead" in name:
        check(not head_hair.any(), "no forelock on a body without a head")
    else:
        on_head = new.weight_on(["DEF_Head"])[head_hair]
        check(head_hair.sum() > 0 and (on_head > 0.99).mean() > 0.6
              and np.allclose(new.weight_on(["DEF_Head", "DEF_Neck1", "DEF_Neck2"])[head_hair], 1),
              f"forelock ({head_hair.sum()} vertices) rides on the head; only its root blends into the neck, no ears or mouth")

    ref_open = open_edge_count(ref.attrs["POSITION"], ref.indices.reshape(-1, 3))
    body_tris = tris[~is_new[tris].any(axis=1)]
    check(open_edge_count(a["POSITION"], body_tris) <= ref_open + 0, f"no new holes in the body ({ref_open} open edges in the original)")


def verify_mane(name):
    print(name)
    g, b = glb.read(os.path.join(OUT_MANES, name + ".glb"))
    a, idx = glb.primitive_arrays(g, b)
    n = len(a["POSITION"])
    check(idx.max() < n, "indices in range")
    check(((a["TEXCOORD_0"] >= 0) & (a["TEXCOORD_0"] <= 1)).all(), "UVs inside the texture")
    check(np.allclose(np.linalg.norm(a["NORMAL"], axis=1), 1, atol=1e-3), "unit normals")
    # thin sheets are two layers facing opposite ways, so their area-weighted normals cancel out
    p = a["POSITION"][idx.reshape(-1, 3)]
    fn = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    area = np.linalg.norm(fn, axis=1)
    check(area.min() > 1e-10, "no degenerate triangles")
    check(np.linalg.norm(fn.sum(axis=0)) < 0.15 * area.sum(), "two-sided (visible with back-face culling)")


def verify_mod_files():
    print("mod files")
    shared = [n for n in ("SimpleManeStart", "SimpleManeMid", "SimpleManeEnd") if os.path.exists(os.path.join(OUT_MODELS, n + ".glb"))]
    check(not shared, "the Horse mod's shared mane meshes are not replaced (foals use them)")
    script = os.path.join(MOD_MEDIA, "scripts", "HorseManeTailImproved", "mane_models.txt")
    with open(script, encoding="utf-8") as f:
        text = f.read()
    check(all(f"model {n} {{" in text and f"mesh = HorseManeTailImproved/{n}," in text for n in MANE_FOR_BONE.values()),
          "every mane mesh has a model in mane_models.txt")
    lua = os.path.join(MOD_MEDIA, "lua", "client", "HorseManeTailImproved", "ManeModels.lua")
    with open(lua, encoding="utf-8") as f:
        text = f.read()
    check(all(f'{n} = "HorseManeTailImproved.{n}"' in text for n in MANE_FOR_BONE.values()), "ManeModels.lua maps every mane slot")
    check(os.path.exists(os.path.join(MOD_MEDIA, "textures", "HorseManeTailImproved", "ManeStrands.png")), "mane texture exists")


for body in BODIES:
    verify_body(body)
for piece in MANE_FOR_BONE.values():
    verify_mane(piece)
verify_mod_files()

print()
print("FAILED: " + "; ".join(failures) if failures else "All checks passed.")
sys.exit(1 if failures else 0)
