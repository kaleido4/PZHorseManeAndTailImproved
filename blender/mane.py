"""Builds the three mane pieces (start, mid, end) as static meshes.

Each piece is drawn in its mane bone's local frame, in mesh units (the bone
matrices carry the skeleton's 0.6 scale, so 1 mesh unit = 0.6 world units):
+Z runs along the crest towards the head, +Y points up out of the crest and
+X is the horse's left. The neck surface is sampled from the body in the idle
animation, because that is how the bones sit in game.
"""
import os

import numpy as np

import glb
from common import Body, Pose, REFERENCE, MANE_BONES, unit
from hair import Clump, catmull_rom, merge

SIDE = -1             # hair falls to -X, the horse's right (off) side
RINGS = 6             # path points per clump, tip included
PROFILE_POSE = ("Horse_Idle.glb", 0.0)

# right half of F_Hair_White.png holds long vertical strands
HAIR_U = (0.45, 0.95)
HAIR_V = (0.37, 0.98)


class NeckSections:
    """Cross-sections of the posed body in one bone's frame."""

    Z_PLANES = np.round(np.arange(-0.24, 0.3401, 0.02), 3)

    def __init__(self, local_positions, tris):
        self.planes = {}
        for z in self.Z_PLANES:
            p = local_positions[tris]
            d = p[:, :, 2] - z
            segs = []
            for a, c in ((0, 1), (1, 2), (2, 0)):
                cross = (d[:, a] < 0) != (d[:, c] < 0)
                s = d[cross, a] / (d[cross, a] - d[cross, c])
                pts = p[cross, a, :2] + s[:, None] * (p[cross, c, :2] - p[cross, a, :2])
                segs.append((np.nonzero(cross)[0], pts))
            # pair up the two crossing points of every triangle cut by the plane
            idx = np.concatenate([i for i, _ in segs])
            pts = np.concatenate([q for _, q in segs])
            if len(idx) == 0:
                self.planes[z] = np.zeros((0, 2, 2))
                continue
            order = np.argsort(idx, kind="stable")
            idx, pts = idx[order], pts[order]
            keep = np.r_[idx[1:] == idx[:-1], False]
            self.planes[z] = np.stack([pts[keep], pts[np.r_[False, keep[:-1]]]], axis=1)

    def _plane(self, z):
        return self.planes[self.Z_PLANES[np.argmin(np.abs(self.Z_PLANES - z))]]

    def lateral(self, z, y):
        """Furthest surface point towards SIDE at height y (as a positive distance), or nan."""
        segs = self._plane(z)
        (x1, y1), (x2, y2) = segs[:, 0].T, segs[:, 1].T
        hit = ((y1 - y) * (y2 - y) <= 0) & (y1 != y2)
        if not hit.any():
            return np.nan
        x = x1[hit] + (y - y1[hit]) / (y2[hit] - y1[hit]) * (x2[hit] - x1[hit])
        return (SIDE * x).max()

    def top(self, z, half_width=None):
        """Highest surface point in the plane, optionally only within |x| < half_width."""
        pts = self._plane(z).reshape(-1, 2)
        if half_width is not None:
            pts = pts[np.abs(pts[:, 0]) < half_width]
        return pts[:, 1].max() if len(pts) else np.nan


def bone_frames(pose_anim=PROFILE_POSE):
    body = Body(os.path.join(REFERENCE, "Mare.glb"))
    pose = Pose(body, *pose_anim)
    posed = pose.skin(body.attrs["POSITION"].astype(float), body.attrs["JOINTS_0"], body.attrs["WEIGHTS_0"])
    tris = body.indices.reshape(-1, 3)
    frames = {}
    for name in MANE_BONES:
        M = pose.mats[name]
        local = (np.c_[posed, np.ones(len(posed))] @ np.linalg.inv(M).T)[:, :3]
        down = np.linalg.inv(M[:3, :3]) @ np.array([0.0, -1.0, 0.0])
        frames[name] = (NeckSections(local, tris), unit(down))
    return frames


def lateral_any(sections, z, y, pct):
    vals = np.array([s.lateral(z, y) for s in sections])
    vals = vals[~np.isnan(vals)]
    return np.percentile(vals, pct) if len(vals) else np.nan


def hair_uv(index):
    width = 0.17
    u0 = HAIR_U[0] + (index * 0.11) % (HAIR_U[1] - HAIR_U[0] - width)

    def uv(s, t):
        return (u0 + t * width, HAIR_V[0] + s * (HAIR_V[1] - HAIR_V[0]))
    return uv


def side_clump(sections, down, z_c, length, width, index, layer=0.0, pct=60, drift_scale=0.7, apex=0.016):
    """A lock rooted on the crest at z_c that arcs over to SIDE and hangs down.

    Neighbouring locks overlap; `layer` pushes a lock further out so overlapping
    locks stack instead of fighting for the same surface.
    """
    crest = np.nanmax([s.top(z_c, 0.03) for s in sections])
    # z change per unit of depth so the hair follows gravity; never forwards, which
    # near the poll would lay the hair along the face
    drift = min(0.0, drift_scale * down[2] / max(-down[1], 0.2))
    pts = [
        (-SIDE * 0.002, crest - 0.012, z_c),
        (SIDE * 0.016, crest + apex + 0.5 * layer, z_c),
    ]
    for depth in np.linspace(0.025, length, 5):
        z = z_c + drift * depth
        y = crest - depth
        d = lateral_any(sections, z, y, pct)
        if np.isnan(d):
            d = pts[-1][0] * SIDE
        clearance = 0.012 + layer + 0.05 * depth
        pts.append((SIDE * (d + clearance), y, z))
    path = catmull_rom(pts, RINGS)
    s = np.linspace(0, 1, RINGS)
    w = np.interp(s, [0, 0.3, 0.8, 1], [0.8, 1.0, 0.8, 0.3]) * width
    h = np.interp(s, [0, 0.3, 0.8, 1], [0.008, 0.018, 0.012, 0.004])
    centre_y = crest - 0.15

    def outward(p, t):
        return unit(np.array([p[0], p[1] - centre_y, 0.0]))
    return Clump(path, w, h, (0, 0, 1), outward, hair_uv(index)).build()


def forelock(sections, length=0.12, width=0.075, index=3):
    """A tuft that rises between the ears and falls forward onto the forehead."""
    sec = sections[0]
    pts = [(0.0, sec.top(-0.02, 0.025) - 0.01, -0.02)]
    for i, z in enumerate(np.linspace(0.02, length, 4)):
        lift = 0.03 * (1 - i / 3) + 0.012
        pts.append((0.0, sec.top(z, 0.025) + lift, z))
    path = catmull_rom(pts, RINGS)
    s = np.linspace(0, 1, RINGS)
    w = np.interp(s, [0, 0.4, 0.8, 1], [0.7, 1.0, 0.7, 0.3]) * width
    h = np.interp(s, [0, 0.4, 0.8, 1], [0.024, 0.024, 0.012, 0.004])

    def outward(p, t):
        return np.array([0.0, 1.0, 0.0])
    return Clump(path, w, h, (1, 0, 0), outward, hair_uv(index)).build()


def build_pieces():
    frames = bone_frames()
    mids = [frames[n][0] for n in MANE_BONES[1:6]]
    mid_down = unit(np.mean([frames[n][1] for n in MANE_BONES[1:6]], axis=0))
    start, start_down = frames["DEF_ManeStart"]
    end, end_down = frames["DEF_ManeEnd"]
    # (z along the crest, hanging length, width, outward layer)
    start_locks = [(+0.075, 0.16, 0.10, 0.000), (+0.025, 0.13, 0.10, 0.008), (-0.030, 0.10, 0.09, 0.000), (-0.080, 0.06, 0.08, 0.006)]
    mid_locks = [(+0.075, 0.17, 0.10, 0.000), (+0.025, 0.20, 0.10, 0.008), (-0.025, 0.16, 0.10, 0.000), (-0.075, 0.19, 0.10, 0.008)]
    end_locks = [(-0.080, 0.17, 0.10, 0.000), (-0.030, 0.13, 0.09, 0.008)]
    pieces = {
        "SimpleManeStart": merge([side_clump([start], start_down, z, L, w, i, layer=ly) for i, (z, L, w, ly) in enumerate(start_locks)]),
        "SimpleManeMid": merge([side_clump(mids, mid_down, z, L, w, i, layer=ly) for i, (z, L, w, ly) in enumerate(mid_locks)]),
        "SimpleManeEnd": merge([side_clump([end], end_down, z, L, w, i, layer=ly) for i, (z, L, w, ly) in enumerate(end_locks)]
                               + [forelock([end])]),
    }
    return pieces


def write_piece(path, name, positions, uvs, tris, normals):
    gltf = {
        "asset": {"generator": "PZHorseManeAndTailImproved build.py", "version": "2.0"},
        "scene": 0,
        "scenes": [{"name": "Scene", "nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "OBJ-" + name}],
        "meshes": [{"name": "GEO-" + name, "primitives": [{"attributes": {}, "indices": None}]}],
    }
    builder = glb.Builder(gltf)
    prim = gltf["meshes"][0]["primitives"][0]
    prim["attributes"]["POSITION"] = builder.add(positions.astype(np.float32), target=34962, minmax=True)
    prim["attributes"]["NORMAL"] = builder.add(normals.astype(np.float32), target=34962)
    prim["attributes"]["TEXCOORD_0"] = builder.add(uvs.astype(np.float32), target=34962)
    prim["indices"] = builder.add(tris.astype(np.uint16).reshape(-1), target=34963)
    glb.write(path, gltf, builder.finish())
