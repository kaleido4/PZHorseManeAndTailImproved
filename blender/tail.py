"""Replaces the tail inside the mare/stallion body meshes.

The old tail triangles (all three corners weighted to DEF_Tail1-3) are removed
and a new tail is added, skinned to the same tail bones: one bundle of hair that
grows out of the dock, hangs straight down close behind the hindquarters and is
cut straight across at the bottom. Grooves around the bundle give the strands.
UVs land inside the original tail UV island, which every breed texture paints
in its tail colour.
"""
from collections import Counter, defaultdict

import numpy as np

import glb
from common import Body, TAIL_BONES, smooth_normals, unit
from forelock import forelock_part
from hair import merge

UV_V = (0.318, 0.462)  # v range that stays clear of the pointed strip ends
UV_MARGIN = 2.5 / 1024
UV_STRIP = 1           # which of the four tail strips the bundle is mapped onto

TAIL_BOTTOM = 0.18     # height of the straight cut (the hock is at ~0.35, the ground at 0)
SIDES = 24             # around the bundle; every other one is a groove between strands
# how deep each groove cuts in, per strand, so they do not look machined
STRAND_DEPTH = [0.34, 0.24, 0.38, 0.28, 0.20, 0.36, 0.30, 0.22, 0.35, 0.26, 0.32, 0.21]
STRAND_BULGE = [0.04, -0.03, 0.06, 0.00, -0.04, 0.05, 0.02, -0.02, 0.05, -0.01, 0.03, -0.05]
# Below the last tail bone the hair would swing as one rigid plank when the tail
# lifts. Part of its weight stays with the body so it trails and droops instead;
# the extra rings down there let that droop bend smoothly.
TRAIL_START = 0.48     # height where trailing starts (end of DEF_Tail3)
TRAIL_MAX = 0.30       # share of body weight at the cut
# (height, centre z, half width, half depth, groove scale) from the dock down to the cut.
# Below DEF_Tail3 (~0.48) the hair is rigid, so few rings are needed there.
PROFILE = [
    (0.814, -0.428, 0.028, 0.028, 0.0),
    (0.795, -0.456, 0.035, 0.033, 0.3),
    (0.760, -0.489, 0.048, 0.040, 0.7),
    (0.705, -0.511, 0.058, 0.045, 1.0),
    (0.620, -0.522, 0.062, 0.047, 1.0),
    (0.500, -0.526, 0.065, 0.048, 1.0),
    (0.420, -0.526, 0.067, 0.048, 1.0),
    (0.340, -0.526, 0.069, 0.049, 1.0),
    (0.260, -0.525, 0.070, 0.049, 1.0),
    (TAIL_BOTTOM, -0.524, 0.072, 0.050, 1.0),
]


class TailIsland:
    """The original tail UV island, rasterised, split into its vertical strips."""

    R = 1024

    def __init__(self, body):
        uv = body.attrs["TEXCOORD_0"]
        tailw = body.weight_on(TAIL_BONES)
        tris = body.indices.reshape(-1, 3)
        pure = tris[(tailw[tris] > 0.01).all(axis=1)]
        R = self.R
        ys, xs = np.mgrid[0:R, 0:R]
        P = np.stack([(xs + 0.5) / R, (ys + 0.5) / R], -1)
        mask = np.zeros((R, R), bool)
        for t in pure:
            a, b, c = uv[t]
            v0, v1 = b - a, c - a
            d = v0[0] * v1[1] - v1[0] * v0[1]
            if abs(d) < 1e-12:
                continue
            p = P - a
            l1 = (p[..., 0] * v1[1] - v1[0] * p[..., 1]) / d
            l2 = (v0[0] * p[..., 1] - p[..., 0] * v0[1]) / d
            mask |= (l1 >= 0) & (l2 >= 0) & (l1 + l2 <= 1)
        # row -> [(left_u, right_u)] for the 4 strips, left to right. Strips touch in a few
        # rows; those rows are interpolated from the rows around them.
        rows = np.arange(int(UV_V[0] * R) - 1, int(UV_V[1] * R) + 2)
        found = {}
        for row in rows:
            cols = np.nonzero(mask[row])[0]
            runs = np.split(cols, np.nonzero(np.diff(cols) > 1)[0] + 1)
            runs = [r for r in runs if len(r) >= 6]
            if len(runs) == 4:
                found[row] = [(r[0] / R, (r[-1] + 1) / R) for r in runs]
        if len(found) < 0.8 * len(rows):
            raise RuntimeError(f"tail strips found in only {len(found)} of {len(rows)} rows")
        good = np.array(sorted(found))
        table = np.array([found[r] for r in good])  # (rows, 4 strips, 2)
        self.edges = {
            row: [tuple(np.interp(row, good, table[:, k, e]) for e in range(2)) for k in range(4)]
            for row in rows
        }

    def mapper(self, strip):
        def uv(s, t):
            v = UV_V[0] + s * (UV_V[1] - UV_V[0])
            left, right = self.edges[int(v * self.R)][strip]
            return (left + UV_MARGIN + t * (right - left - 2 * UV_MARGIN), v)
        return uv


def bundle(island):
    """The tail as one tube from the dock to the cut, plus caps at both ends.

    Returns three parts (tube, cut, root cap) so each gets its own normals: the
    cut keeps a crisp edge instead of being rounded into the sides.
    """
    prof = np.array(PROFILE)
    centres = np.c_[np.zeros(len(prof)), prof[:, 0], prof[:, 1]]
    T = unit(np.gradient(centres, axis=0))
    lat = np.array([1.0, 0.0, 0.0])
    uvmap = island.mapper(UV_STRIP)
    n_rings = len(prof)
    rings = []
    pos, uvs = [], []
    for k, (c, (_, _, w, d, groove)) in enumerate(zip(centres, prof)):
        depth_axis = unit(np.cross(T[k], lat))
        ring = []
        for j in range(SIDES + 1):
            a = 2 * np.pi * j / SIDES
            # odd vertices sit in the grooves, even ones on the strand ridges
            strand = (j // 2) % len(STRAND_DEPTH)
            r = 1 - groove * STRAND_DEPTH[strand] if j % 2 else 1 + groove * STRAND_BULGE[strand]
            ring.append(c + r * (w * np.cos(a) * lat + d * np.sin(a) * depth_axis))
            uvs.append(uvmap(k / (n_rings - 1), j / SIDES))
        rings.append(ring)
        pos += ring
    pos = np.array(pos)
    per = SIDES + 1
    tris = []
    for k in range(n_rings - 1):
        for j in range(SIDES):
            a0, b0 = k * per + j, k * per + j + 1
            a1, b1 = a0 + per, b0 + per
            tris += [(a0, a1, b0), (b0, a1, b1)]
    tris = np.array(tris)
    ring_of = np.arange(len(pos)) // per
    p = pos[tris]
    fn = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    away = p.mean(axis=1) - centres[ring_of[tris]].mean(axis=1)
    flip = (fn * away).sum(axis=1) < 0
    tris[flip] = tris[flip][:, ::-1]
    tube = (pos, np.array(uvs), tris)

    def cap(k, facing):
        ring = np.array(rings[k][:SIDES])
        cpos = np.vstack([ring, centres[k]])
        cuv = [uvmap(k / (n_rings - 1), j / SIDES) for j in range(SIDES)] + [uvmap(k / (n_rings - 1), 0.5)]
        ctris = []
        for j in range(SIDES):
            tri = (j, (j + 1) % SIDES, SIDES)
            q = cpos[list(tri)]
            if np.dot(np.cross(q[1] - q[0], q[2] - q[0]), facing) < 0:
                tri = tri[::-1]
            ctris.append(tri)
        return cpos, np.array(cuv), np.array(ctris)

    return [tube, cap(n_rings - 1, T[-1]), cap(0, -T[0])]


def chain_param(body, positions):
    """Position along the tail bone chain: 0..1 Tail1, 1..2 Tail2, 2..3 Tail3, >3 past the tip."""
    heads = [body.bind_matrix(n)[:3, 3] for n in TAIL_BONES]
    tip = heads[2] + (heads[2] - heads[1])
    chain = np.array(heads + [tip, tip + 10 * (tip - heads[2])])
    best = np.full(len(positions), np.inf)
    param = np.zeros(len(positions))
    for k in range(len(chain) - 1):
        a, b = chain[k], chain[k + 1]
        seg = b - a
        f = np.clip(((positions - a) @ seg) / (seg @ seg), 0, 1)
        dist = np.linalg.norm(positions - (a + f[:, None] * seg), axis=1)
        better = dist < best
        best[better] = dist[better]
        param[better] = k + (f[better] if k < 3 else 1 + f[better] * 10)
    return param


def skin_weights(body, positions):
    s = chain_param(body, positions)
    sc = np.clip(s, 0.5, 2.5)
    w = np.stack([np.maximum(0, 1 - np.abs(sc - (k + 0.5))) for k in range(3)], axis=1)
    spine = np.clip((0.22 - s) / 0.22, 0, 1) * 0.85
    trail = np.clip((TRAIL_START - positions[:, 1]) / (TRAIL_START - TAIL_BOTTOM), 0, 1) * TRAIL_MAX
    spine = np.maximum(spine, trail)
    w = w * (1 - spine)[:, None]
    joints = np.zeros((len(positions), 4), np.uint8)
    weights = np.zeros((len(positions), 4), np.float32)
    joints[:, 0] = body.joint("DEF_Spine1")
    weights[:, 0] = spine
    for k, name in enumerate(TAIL_BONES):
        joints[:, k + 1] = body.joint(name)
        weights[:, k + 1] = w[:, k]
    weights /= weights.sum(axis=1, keepdims=True)
    # glTF wants unused slots to point at joint 0 with weight 0; keep the strongest first
    order = np.argsort(-weights, axis=1)
    joints = np.take_along_axis(joints, order, 1)
    weights = np.take_along_axis(weights, order, 1)
    joints[weights == 0] = 0
    return joints, weights


def tail_geometry(body):
    """Tail parts as [(positions, uvs, tris)]; normals are computed per part."""
    return bundle(TailIsland(body))


def hole_caps(body, all_tris, keep_tris):
    """Close the openings left in the rump where the old tail was stitched on.

    Returns vertex indices into the body (one per cap corner, copied so the cap
    deforms with the rump) and triangles over those corners.
    """
    P = body.attrs["POSITION"]
    _, weld = np.unique(np.round(P, 5), axis=0, return_inverse=True)
    weld = weld.reshape(-1)

    def open_edges(ts):
        count = Counter()
        for t in ts:
            for i in range(3):
                a, b = weld[t[i]], weld[t[(i + 1) % 3]]
                count[(min(a, b), max(a, b))] += 1
        return {e for e, n in count.items() if n == 1}

    new_edges = open_edges(keep_tris) - open_edges(all_tris)
    rep = {}
    for t in keep_tris:
        for v in t:
            rep.setdefault(weld[v], v)
    adj = defaultdict(list)
    for a, b in new_edges:
        adj[a].append(b)
        adj[b].append(a)
    corners, tris, seen = [], [], set()
    for start in adj:
        if start in seen:
            continue
        loop, prev, cur = [start], None, start
        seen.add(start)
        while True:
            nxt = [n for n in adj[cur] if n != prev and n not in seen]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            loop.append(cur)
            seen.add(cur)
        verts = [rep[w] for w in loop]
        base = len(corners)
        corners += verts
        outward = body.attrs["NORMAL"][verts].mean(axis=0)
        for i in range(1, len(verts) - 1):
            tri = [base, base + i, base + i + 1]
            p = P[[verts[0], verts[i], verts[i + 1]]]
            if np.dot(np.cross(p[1] - p[0], p[2] - p[0]), outward) < 0:
                tri = tri[::-1]
            tris.append(tri)
    return np.array(corners, int), np.array(tris, int).reshape(-1, 3)


def graft(src_path, dst_path, with_forelock=True):
    """Write src_path's body to dst_path with the new tail, and the forelock unless
    the body has no head."""
    body = Body(src_path)
    a = body.attrs
    tris = body.indices.reshape(-1, 3)
    tailw = body.weight_on(TAIL_BONES)
    keep_tris = tris[~(tailw[tris] > 0.01).all(axis=1)]
    corners, cap_tris = hole_caps(body, tris, keep_tris)
    used = np.unique(keep_tris)
    remap = np.full(len(a["POSITION"]), -1)
    remap[used] = np.arange(len(used))
    keep_tris = remap[keep_tris]

    # cap corners copy the rump vertices but take one flat coat texel from the first corner
    cap_uv = np.repeat(a["TEXCOORD_0"][corners[:1]], len(corners), axis=0)
    cap_nrm = np.repeat(unit(a["NORMAL"][corners].mean(axis=0, keepdims=True)), len(corners), axis=0)

    parts = tail_geometry(body)
    tpos, tuv, ttris = merge(parts)
    tnorm = np.vstack([smooth_normals(p, t) for p, _, t in parts])
    tjoints, tweights = skin_weights(body, tpos)
    if with_forelock:
        fpos, fuv, ftris, fnorm, fjoints, fweights = forelock_part(body, TailIsland(body))
        ttris = np.vstack([ttris, ftris + len(tpos)])
        tnorm = np.vstack([tnorm, fnorm])
        tpos, tuv = np.vstack([tpos, fpos]), np.vstack([tuv, fuv])
        tjoints, tweights = np.vstack([tjoints, fjoints]), np.vstack([tweights, fweights])
    n_tail = len(tpos)

    pos = np.vstack([a["POSITION"][used], a["POSITION"][corners], tpos]).astype(np.float32)
    nrm = np.vstack([a["NORMAL"][used], cap_nrm, tnorm]).astype(np.float32)
    uv = np.vstack([a["TEXCOORD_0"][used], cap_uv, tuv]).astype(np.float32)
    col = np.vstack([a["COLOR_0"][used], a["COLOR_0"][corners], np.ones((n_tail, 4))])
    col = np.round(np.clip(col, 0, 1) * 65535).astype(np.uint16)
    jnt = np.vstack([a["JOINTS_0"][used], a["JOINTS_0"][corners], tjoints]).astype(np.uint8)
    wgt = np.vstack([a["WEIGHTS_0"][used], a["WEIGHTS_0"][corners], tweights]).astype(np.float32)
    idx = np.vstack([keep_tris, cap_tris + len(used), ttris + len(used) + len(corners)]).astype(np.uint16).reshape(-1)

    gltf = body.gltf
    ibm = body.ibm.transpose(0, 2, 1).reshape(-1, 16).astype(np.float32)
    builder = glb.Builder(gltf)
    prim = gltf["meshes"][0]["primitives"][0]
    prim["attributes"] = {
        "POSITION": builder.add(pos, target=34962, minmax=True),
        "NORMAL": builder.add(nrm, target=34962),
        "TEXCOORD_0": builder.add(uv, target=34962),
        "COLOR_0": builder.add(col, target=34962, normalized=True),
        "JOINTS_0": builder.add(jnt, target=34962),
        "WEIGHTS_0": builder.add(wgt, target=34962),
    }
    prim["indices"] = builder.add(idx, target=34963)
    gltf["skins"][0]["inverseBindMatrices"] = builder.add(ibm)
    gltf["asset"]["generator"] = "PZHorseManeAndTailImproved build.py (from " + gltf["asset"].get("generator", "?") + ")"
    glb.write(dst_path, gltf, builder.finish())
    return {
        "removed_tris": int(len(tris) - len(keep_tris)),
        "hole_cap_tris": int(len(cap_tris)),
        "added_tris": int(len(ttris)),
        "total_tris": int(len(idx) // 3),
        "verts": int(len(pos)),
    }
