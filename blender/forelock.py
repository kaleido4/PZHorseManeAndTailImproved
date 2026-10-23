"""Puts the forelock into the body mesh, skinned like the forehead under it.

As a mane attachment it would ride on DEF_ManeEnd, which turns up to 18 degrees
against the head while grazing and lifts the forelock off the face. In the body
mesh it moves exactly with the skin it lies on. It uses the tail strips of the
breed texture, the same hair colour as the tail.
"""
import numpy as np

import mane
from common import Pose


def forelock_part(body, island):
    """(positions, uvs, tris, normals, joints, weights) of the forelock in the body's bind space."""
    frames = mane.bone_frames()
    sections, _, M_end = frames["DEF_ManeEnd"]
    local, uvs, tris, local_normals = mane.forelock([sections], island.mapper(2))

    # where it sits on the idle-posed horse, and the posed skin under it
    pose = Pose(body, *mane.PROFILE_POSE)
    a = body.attrs
    world = (np.c_[local, np.ones(len(local))] @ M_end.T)[:, :3]
    world_normals = local_normals @ M_end[:3, :3].T
    posed = pose.skin(a["POSITION"].astype(float), a["JOINTS_0"], a["WEIGHTS_0"])
    nearest = np.argmin(np.linalg.norm(posed[None, :, :] - world[:, None, :], axis=2), axis=1)

    # The root at the poll blends neck and head like the skin there; the rest rides on
    # the head alone (nearest-skin weights would pull in the ears and the mouth).
    head = body.joint("DEF_Head")
    joints = np.zeros((len(world), 4), np.uint8)
    weights = np.zeros((len(world), 4), np.float32)
    joints[:, 0], weights[:, 0] = head, 1.0
    root = local[:, 2] < 0.0
    allowed = {body.joint(n) for n in ("DEF_Head", "DEF_Neck1", "DEF_Neck2")}
    for v in np.nonzero(root)[0]:
        j, w = a["JOINTS_0"][nearest[v]].astype(int), a["WEIGHTS_0"][nearest[v]].copy()
        w[[k not in allowed for k in j]] = 0
        if w.sum() > 0:
            joints[v], weights[v] = j, w / w.sum()

    # undo the skinning with those same weights to get bind-space positions
    skin = np.stack([pose.mats[n] @ body.ibm[i] for i, n in enumerate(body.joint_names)])
    bind = np.empty_like(world)
    normals = np.empty_like(world)
    for v in range(len(world)):
        S = np.linalg.inv(sum(weights[v, k] * skin[joints[v, k]] for k in range(4)))
        bind[v] = (S @ np.r_[world[v], 1.0])[:3]
        n = S[:3, :3] @ world_normals[v]
        normals[v] = n / np.linalg.norm(n)
    return bind, uvs, tris, normals.astype(np.float32), joints, weights
