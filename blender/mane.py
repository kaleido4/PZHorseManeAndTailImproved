"""Builds the mane: one static mesh per mane bone of an adult horse.

Each piece is drawn in its mane bone's local frame, in mesh units (the bone
matrices carry the skeleton's 0.6 scale, so 1 mesh unit = 0.6 world units):
+Z runs along the crest towards the head, +Y points up out of the crest and
+X is the horse's left. The neck surface is sampled from the body in the idle
animation, because that is how the bones sit in game.

Every piece is a few flat locks that rise over the crest, fall down one side of
the neck sweeping back along it in a bow, flick at the tip and end in a sharp point. How far each lock hangs is solved
from a target height for its tip, so the bottom of the mane is cut in a V: short
at the poll, longest at the base of the neck, short again at the withers.
"""
import os

import numpy as np

import glb
from common import Body, Pose, REFERENCE, MANE_BONES, MANE_FOR_BONE, unit
from hair import Clump, catmull_rom

SIDE = -1             # hair falls to -X, the horse's right (off) side
RINGS = 9             # points along a lock, tip included
CLEARANCE = 0.016     # gap between the neck and the inner face of a lock
APEX = 0.038          # how high the hair rises over the crest
LOCK_LAYER = 0.012    # every other lock sits this much further out, so overlapping locks stack
PROFILE_POSE = ("Horse_Idle.glb", 0.0)

# the whole of ManeStrands.png (see mane_texture.py): strands run along v, roots at v=0,
# lock edges at u=0 and u=1
HAIR_U = (0.0, 1.0)
HAIR_V = (0.0, 1.0)

# Bottom edge of the mane: (position along the crest from poll 0 to withers 1, how far
# the lock's tip hangs below where it roots on the crest, in world units). Between the
# points it is interpolated. In the idle pose the crest runs from about 1.06 at the poll
# to 0.90 at the withers (the ground is 0), so this cuts the bottom edge in a V.
V_CUT = [(0.0, 0.10), (0.30, 0.17), (0.76, 0.33), (1.0, 0.09)]
# where the locks of each bone root along its crest (bone-local z), back to front
LOCKS = {"DEF_ManeStart": [-0.11, -0.05, 0.0, 0.06], "DEF_ManeEnd": [-0.13, -0.08, -0.035]}
MID_LOCKS = [-0.06, -0.005, 0.05]
# Every lock gets its own shape from these ranges (seeded, so builds are repeatable):
LOCK_WIDTH = (0.105, 0.145)   # widest point
LOCK_SWEEP = (0.38, 0.50)     # how far back along the neck it sweeps, per unit of length
LOCK_FLICK = (0.06, 0.12)     # how far its tip flicks forwards again, per unit of length
LOCK_JITTER = (-0.05, 0.04)   # added to its hanging length, so the tips are uneven
SEED = 42
# the filler under the bend of the locks
FILLER_LENGTH = 0.15          # how far down the neck it reaches (shorter by the poll)
FILLER_LAYER = -0.009         # sits this much closer to the neck than the locks
FILLER_OVERLAP = 0.07         # wider than its locks' spread, to meet the next piece's filler


class NeckSections:
    """Cross-sections of the posed body in one bone's frame."""

    Z_PLANES = np.round(np.arange(-0.60, 0.3401, 0.02), 3)

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

    def centre(self, z):
        """A point on the midline just under the top surface. Near the poll the plane also
        cuts the neck and chest far below, so the middle of the whole section is no good."""
        return np.array([0.0, self.top(z, 0.045) - 0.08])

    def wrap(self, z, s, lift):
        """Point on the outline of the cross-section at arc length s from the top of the
        midline (s > 0 towards +X, s < 0 towards -X), lifted off it along its normal.
        Unlike a height lookup this follows the surface round onto the sides.
        Returns (point, outward normal, tangent along the outline), all in x and y."""
        segs = self._plane(z)
        c = self.centre(z)
        side = 1.0 if s >= 0 else -1.0
        thetas = np.radians(np.arange(0, 151, 2))
        outline = []
        a, b = segs[:, 0], segs[:, 1] - segs[:, 0]
        for th in thetas:
            d = np.array([side * np.sin(th), np.cos(th)])
            # first exit of the ray from the centre through the outline
            den = b[:, 0] * d[1] - b[:, 1] * d[0]
            ok = np.abs(den) > 1e-12
            r = a[ok] - c
            t = (b[ok, 0] * r[:, 1] - b[ok, 1] * r[:, 0]) / den[ok]
            u = (d[0] * r[:, 1] - d[1] * r[:, 0]) / den[ok]
            hit = (t > 0) & (u >= 0) & (u <= 1)
            outline.append(c + d * t[hit].min() if hit.any() else outline[-1])
        outline = np.array(outline)
        arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(outline, axis=0), axis=1))]
        i = np.clip(np.searchsorted(arc, abs(s)), 1, len(arc) - 1)
        f = (abs(s) - arc[i - 1]) / max(arc[i] - arc[i - 1], 1e-9)
        p = outline[i - 1] + np.clip(f, 0, 1) * (outline[i] - outline[i - 1])
        tangent = unit(outline[i] - outline[i - 1])
        normal = np.array([tangent[1], -tangent[0]])
        if np.dot(normal, p - c) < 0:
            normal = -normal
        return p + lift * normal, normal, tangent


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
        frames[name] = (NeckSections(local, tris), unit(down), M)
    return frames


def hair_uv(s, t):
    return (HAIR_U[0] + t * (HAIR_U[1] - HAIR_U[0]), HAIR_V[0] + s * (HAIR_V[1] - HAIR_V[0]))


def lock_path(sections, z_c, length, layer, sweep, flick, rings=RINGS):
    """Centre line of a lock rooted on the crest at z_c. It rises over the crest, falls
    down SIDE of the neck sweeping back along it in a bow, and its tip flicks forwards
    again. It keeps clear of the neck while there is neck beside it."""
    crest = sections.top(z_c, 0.03)
    # short locks by the poll and the withers rise less, or they curl into a knot
    apex = APEX * min(1.0, length / 0.18)
    pts = [
        (-SIDE * 0.012, crest - 0.016, z_c),
        (SIDE * 0.028, crest + apex + 0.5 * layer, z_c),
    ]
    prev, d_prev = 0.0, 0.028
    for f in (0.12, 0.3, 0.5, 0.7, 0.86, 1.0):
        depth = f * length
        # the back-sweep eases out along the lock (a bow), then the tip turns forwards
        z = z_c - sweep * length * np.sin(0.5 * np.pi * f) + flick * length * f ** 3
        # stand off the widest bit of neck between the previous point and this one,
        # so the lock between them stays outside
        span = [sections.lateral(z, crest - q) for q in np.linspace(prev, depth, 5)]
        span = [v for v in span if not np.isnan(v)]
        d = max(span) if span else d_prev
        d = max(d, d_prev - CLEARANCE)  # once past the neck, keep hanging straight
        pts.append((SIDE * (d + CLEARANCE + layer), crest - depth, z))
        prev, d_prev = depth, d + CLEARANCE
    return catmull_rom(pts, rings), crest


# (fraction along the lock, width, ridge height): a lock tapering to a sharp point,
# and the filler under the bend of the locks, low and blunt
LOCK_SHAPE = ([0, 0.15, 0.35, 0.6, 0.85, 1], [0.95, 1.0, 1.0, 0.7, 0.3, 0.0], [0.003, 0.007, 0.016, 0.012, 0.005, 0.002])
FILLER_SHAPE = ([0, 0.2, 0.5, 0.8, 1], [1.0, 1.0, 0.95, 0.6, 0.25], [0.002, 0.005, 0.007, 0.005, 0.002])


def lock(sections, z_c, length, layer, sweep, flick, width, shape=LOCK_SHAPE, rings=RINGS):
    """A flat lock that tapers to a sharp point: flat towards the neck, a low ridge outside."""
    path, crest = lock_path(sections, z_c, length, layer, sweep, flick, rings)
    s = np.linspace(0, 1, rings)
    w = np.interp(s, shape[0], shape[1]) * width
    h = np.interp(s, shape[0], shape[2])
    centre_y = crest - 0.15
    T = np.gradient(path, axis=0)
    # lay the width along the neck surface, at right angles to where the lock runs. The
    # surface faces up on top of the crest and sideways down the neck; a sideways tilt
    # over the crest would twist the swept-back lock and stick a corner up.
    k = np.clip((crest - path[:, 1]) / 0.06, 0, 1)
    out = np.stack([SIDE * k, 1 - k, np.zeros_like(k)], axis=1)
    across = np.cross(out, T)

    def outward(p, t):
        return unit(np.array([p[0], 0.5 * (p[1] - centre_y), 0.0]))
    pos, uv, tris = Clump(path, w, h, across, outward, hair_uv).build()
    return pos, uv, tris, mass_normals(pos, crest)


def mass_normals(pos, crest):
    """Normals of the hair mass rather than of each lock: up on top of the crest, turning
    to face SIDE (and a little up) down the neck. With every lock's own rounded normals
    the game's light shades each lock like a shiny tube; with these the locks light
    together as one soft clump."""
    k = np.clip((crest - pos[:, 1]) / 0.07, 0, 1)[:, None]
    top = unit(np.stack([pos[:, 0] / 0.05, np.ones(len(pos)), np.zeros(len(pos))], axis=1))
    side = unit(np.array([SIDE, 0.25, 0.0]))
    return unit((1 - k) * top + k * side).astype(np.float32)


def merge_with_normals(parts):
    """Concatenate (positions, uvs, tris, normals) parts into one mesh."""
    pos, uv, tri, nrm = [], [], [], []
    base = 0
    for p, u, t, n in parts:
        pos.append(p)
        uv.append(u)
        tri.append(t + base)
        nrm.append(n)
        base += len(p)
    return np.vstack(pos), np.vstack(uv), np.vstack(tri), np.vstack(nrm)


# Forelock locks: (start across the forehead, length down the face, how far round
# towards SIDE it flows by its tip, measured along the skin, width, ridge height).
FORELOCK = [
    (0.0, 0.25, 0.06, 0.070, 0.016),
    (SIDE * 0.022, 0.21, 0.10, 0.060, 0.015),
    (-SIDE * 0.020, 0.17, 0.03, 0.052, 0.013),
]


def forelock(sections, uv):
    """Pointed locks that rise between the ears and fall down the forehead, flowing
    round onto the side of the face towards SIDE over the eye. Every point is wrapped
    onto the skin under it, so the hair lies on the face wherever it flows. Drawn in the
    end bone's frame against the idle pose; forelock.py moves it into the body mesh,
    because the end bone turns up to 18 degrees against the head while grazing.
    """
    sec = sections[0]
    locks = []
    for x0, length, bend, width, ridge in FORELOCK:
        pts = [(x0 * 0.5, sec.top(-0.02, 0.025) - 0.012, -0.02)]
        across = [(1.0, 0.0, 0.0)]
        for f, lift in ((0.12, 0.016), (0.4, 0.009), (0.7, 0.006), (1.0, 0.005)):
            z = f * length
            p, _, tangent = sec.wrap(z, x0 + SIDE * bend * f * f, lift)
            pts.append((p[0], p[1], z))
            across.append((tangent[0], tangent[1], 0.0))
        path = catmull_rom(pts, 6)
        across = np.array([np.interp(np.linspace(0, len(across) - 1, 6), np.arange(len(across)), np.array(across)[:, k]) for k in range(3)]).T
        s = np.linspace(0, 1, 6)
        w = np.interp(s, [0, 0.25, 0.6, 1], [0.7, 1.0, 0.65, 0.0]) * width
        h = np.interp(s, [0, 0.25, 0.6, 1], [0.6, 1.0, 0.7, 0.15]) * ridge

        def outward(p, t):
            return unit(np.r_[p[:2] - sec.centre(p[2]), 0.0])
        pos, uvs, tris = Clump(path, w, h, across, outward, uv).build()
        # shaded like the face under it (out from the head), not like a tube
        normals = np.array([outward(p, None) for p in pos], np.float32)
        locks.append((pos, uvs, tris, normals))
    return merge_with_normals(locks)


def solve_length(sections, M, z, target_y, layer, sweep, flick):
    """Hanging length that puts the lock's tip at world height target_y."""
    def tip_y(length):
        path, _ = lock_path(sections, z, length, layer, sweep, flick)
        return (M @ np.r_[path[-1], 1.0])[1]
    lo, hi = 0.03, 1.2
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        if tip_y(mid) > target_y:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def build_pieces():
    frames = bone_frames()
    rng = np.random.default_rng(SEED)
    # where each lock roots on the crest, in the idle-posed world (the horse faces +z)
    roots = {}
    for bone in MANE_BONES:
        sections, _, M = frames[bone]
        roots[bone] = [M @ np.array([0.0, sections.top(z, 0.03), z, 1.0]) for z in LOCKS.get(bone, MID_LOCKS)]
    poll = max(r[2] for rs in roots.values() for r in rs)
    withers = min(r[2] for rs in roots.values() for r in rs)

    pieces = {}
    n = 0  # locks so far along the neck, to alternate their layer
    for bone in MANE_BONES:
        sections, _, M = frames[bone]
        locks, lengths = [], []
        zs = LOCKS.get(bone, MID_LOCKS)
        for z, root in zip(zs, roots[bone]):
            t = (poll - root[2]) / (poll - withers)
            target = root[1] - np.interp(t, [p for p, _ in V_CUT], [d for _, d in V_CUT])
            layer = LOCK_LAYER * (n % 2)
            sweep, flick = rng.uniform(*LOCK_SWEEP), rng.uniform(*LOCK_FLICK)
            width, jitter = rng.uniform(*LOCK_WIDTH), rng.uniform(*LOCK_JITTER)
            length = max(0.05, solve_length(sections, M, z, target, layer, sweep, flick) + jitter)
            locks.append(lock(sections, z, length, layer, sweep, flick, width))
            lengths.append(length)
            n += 1
        # Filler under the bend: one low, wide lock across the whole piece, closer to the
        # neck than the locks, so no neck shows where they turn over the crest and fall.
        span = max(zs) - min(zs)
        filler_len = min(FILLER_LENGTH, 0.7 * min(lengths))
        locks.append(lock(sections, 0.5 * (min(zs) + max(zs)), filler_len, FILLER_LAYER, 0.15, 0.0,
                          span + FILLER_OVERLAP, FILLER_SHAPE, rings=6))
        pieces[MANE_FOR_BONE[bone]] = merge_with_normals(locks)
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
