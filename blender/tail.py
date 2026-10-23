"""Replaces the tail inside the mare/stallion body meshes.

The old tail triangles (all three corners weighted to DEF_Tail1-3) are removed
and a new dock plus hair clumps are added, skinned to the same tail bones.
UVs land inside the original tail UV island, which every breed texture paints
in its tail colour.
"""
from collections import Counter, defaultdict

import numpy as np

import glb
from common import Body, TAIL_BONES, smooth_normals, unit
from hair import Clump, catmull_rom, merge

RINGS = 6
TAIL_CENTRE_Z = -0.522
DEPTH_SCALE = 0.5      # bundle is flatter front-to-back than side-to-side
UV_V = (0.318, 0.462)  # v range that stays clear of the pointed strip ends
UV_MARGIN = 2.5 / 1024

# (angle from straight back in degrees, root position along the dock 0..1, tip height, outward layer)
CLUMPS = [
    (0, 0.08, 0.300, 0.008),
    (20, 0.14, 0.312, 0.000),
    (-20, 0.14, 0.306, 0.000),
    (42, 0.24, 0.322, 0.008),
    (-42, 0.24, 0.326, 0.008),
    (65, 0.34, 0.338, 0.000),
    (-65, 0.34, 0.334, 0.000),
    (90, 0.46, 0.350, 0.008),
    (-90, 0.46, 0.354, 0.008),
    (115, 0.60, 0.372, 0.000),
    (-115, 0.60, 0.368, 0.000),
]
DOCK_PATH = [(0, 0.812, -0.430), (0, 0.792, -0.462), (0, 0.750, -0.488), (0, 0.700, -0.505), (0, 0.655, -0.512)]
DOCK_RADIUS = [0.034, 0.034, 0.032, 0.028, 0.024, 0.018]
DOCK_SIDES = 6


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


def radial(theta_deg):
    th = np.radians(theta_deg)
    return np.array([np.sin(th), 0.0, -np.cos(th) * DEPTH_SCALE])


def dock(island):
    path = catmull_rom(DOCK_PATH, len(DOCK_RADIUS))
    T = unit(np.gradient(path, axis=0))
    uvmap = island.mapper(0)
    pos, uvs = [], []
    for k, (p, r) in enumerate(zip(path, DOCK_RADIUS)):
        side = unit(np.cross(T[k], [1.0, 0, 0]))
        lat = np.array([1.0, 0, 0])
        for j in range(DOCK_SIDES + 1):
            a = 2 * np.pi * j / DOCK_SIDES
            pos.append(p + r * (np.cos(a) * lat + np.sin(a) * side))
            uvs.append(uvmap(k / (len(path) - 1), j / DOCK_SIDES))
    tris = []
    ring = DOCK_SIDES + 1
    for k in range(len(path) - 1):
        for j in range(DOCK_SIDES):
            a0, b0 = k * ring + j, k * ring + j + 1
            a1, b1 = a0 + ring, b0 + ring
            tris += [(a0, a1, b0), (b0, a1, b1)]
    end = len(pos)
    pos.append(path[-1] + T[-1] * 0.01)
    uvs.append(uvmap(1.0, 0.5))
    last = (len(path) - 1) * ring
    tris += [(last + j, end, last + j + 1) for j in range(DOCK_SIDES)]
    pos, tris = np.array(pos), np.array(tris)
    # wind outwards from the dock axis
    centre = np.vstack([path, path[-1:]])[np.minimum(np.arange(len(pos)) // ring, len(path))]
    p = pos[tris]
    fn = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    flip = (fn * (p.mean(axis=1) - centre[tris].mean(axis=1))).sum(axis=1) < 0
    tris[flip] = tris[flip][:, ::-1]
    return pos, np.array(uvs), tris, path


def hair_clump(island, dock_path, index, theta, root_at, tip_y, layer):
    rad = radial(theta)
    k = root_at * (len(dock_path) - 1)
    i = min(int(k), len(dock_path) - 2)
    d = dock_path[i] + (k - i) * (dock_path[i + 1] - dock_path[i])
    r = np.interp(root_at, np.linspace(0, 1, len(DOCK_RADIUS)), DOCK_RADIUS)

    def centre(y, z=TAIL_CENTRE_Z):
        return np.array([0.0, y, z])
    pts = [
        d + rad * r * 0.5,
        d + rad * (r + 0.010 + layer) + np.array([0, -0.030, -0.004]),
        centre(0.62) + rad * (0.042 + layer),
        centre(0.48) + rad * (0.052 + layer),
        centre(tip_y + 0.065, TAIL_CENTRE_Z + 0.004) + rad * (0.044 + layer),
        centre(tip_y, TAIL_CENTRE_Z + 0.006) + rad * 0.034,
    ]
    path = catmull_rom(pts, RINGS + 1)
    s = np.linspace(0, 1, RINGS + 1)
    w = np.interp(s, [0, 0.3, 0.8, 1], [0.050, 0.072, 0.060, 0.015])
    h = np.interp(s, [0, 0.3, 0.8, 1], [0.016, 0.018, 0.014, 0.004])
    across = unit(np.cross([0.0, 1.0, 0.0], rad))

    def outward(p, t):
        return unit(np.array([p[0], 0.0, (p[2] - TAIL_CENTRE_Z) / DEPTH_SCALE]) + 1e-6)
    return Clump(path, w, h, across, outward, island.mapper(index % 4)).build()


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
    island = TailIsland(body)
    dpos, duv, dtris, dpath = dock(island)
    parts = [(dpos, duv, dtris)]
    for i, (theta, root_at, tip_y, layer) in enumerate(CLUMPS):
        parts.append(hair_clump(island, dpath, i, theta, root_at, tip_y, layer))
    return merge(parts)


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


def graft(src_path, dst_path):
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

    tpos, tuv, ttris = tail_geometry(body)
    tnorm = smooth_normals(tpos, ttris)
    tjoints, tweights = skin_weights(body, tpos)
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
