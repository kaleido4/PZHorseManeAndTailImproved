"""Hair clump geometry shared by the mane and tail builders.

A clump is a closed strand with a triangular cross-section that follows a path
and ends in a point: a flat inner face (towards the neck or tail core) and a
ridge on the outside. Closed geometry keeps it visible with back-face culling.
"""
import numpy as np

from common import unit


def catmull_rom(points, count):
    """Resample a polyline into `count` points spaced evenly by arc length."""
    pts = np.asarray(points, float)
    ext = np.vstack([2 * pts[0] - pts[1], pts, 2 * pts[-1] - pts[-2]])
    dense = []
    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
        for t in np.linspace(0, 1, 16, endpoint=False):
            t2, t3 = t * t, t * t * t
            dense.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    dense.append(pts[-1])
    dense = np.array(dense)
    seg = np.linalg.norm(np.diff(dense, axis=0), axis=1)
    arc = np.r_[0, np.cumsum(seg)]
    targets = np.linspace(0, arc[-1], count)
    return np.stack([np.interp(targets, arc, dense[:, k]) for k in range(3)], axis=1)


class Clump:
    """One hair clump.

    path:      (n, 3) points from root to tip (tip is the last point).
    width:     (n,) clump width along `across` at each point (tip value ignored).
    thickness: (n,) ridge height along the outward direction.
    across:    (n, 3) or (3,) preferred width direction.
    outward:   function(point, tangent) -> rough outward direction (away from the body).
    uv:        function(s, t) -> (u, v); s runs 0..1 root to tip, t runs 0..1 across.
    """

    def __init__(self, path, width, thickness, across, outward, uv):
        self.path = np.asarray(path, float)
        n = len(self.path)
        self.width = np.broadcast_to(np.asarray(width, float), (n,))
        self.thickness = np.broadcast_to(np.asarray(thickness, float), (n,))
        self.across = np.broadcast_to(np.asarray(across, float), (n, 3))
        self.outward = outward
        self.uv = uv

    def build(self):
        P = self.path
        n = len(P)
        T = unit(np.gradient(P, axis=0))
        positions, uvs, tris, ring_of, centres = [], [], [], [], []
        seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
        s_of = np.r_[0, np.cumsum(seg)] / seg.sum()
        for k in range(n - 1):
            a = self.across[k] - np.dot(self.across[k], T[k]) * T[k]
            a = unit(a)
            o = np.cross(T[k], a)
            if np.dot(o, self.outward(P[k], T[k])) < 0:
                o, a = -o, -a
            half = 0.5 * self.width[k]
            inner_a = P[k] - a * half
            inner_b = P[k] + a * half
            ridge = P[k] + o * self.thickness[k]
            s = s_of[k]
            centres.append((inner_a + inner_b + ridge) / 3)
            # ring layout: inner_a(t=0), ridge(t=0.5), inner_b(t=1), inner_a again for the inner face seam
            for p, t in ((inner_a, 0.0), (ridge, 0.5), (inner_b, 1.0), (inner_a, 0.0)):
                positions.append(p)
                uvs.append(self.uv(s, t))
                ring_of.append(k)
        tip = len(positions)
        positions.append(P[-1])
        uvs.append(self.uv(1.0, 0.5))
        ring_of.append(n - 1)
        centres.append(P[-1])
        for k in range(n - 2):
            r0, r1 = 4 * k, 4 * (k + 1)
            for j in range(3):
                a0, b0, a1, b1 = r0 + j, r0 + j + 1, r1 + j, r1 + j + 1
                tris += [(a0, a1, b0), (b0, a1, b1)]
        last = 4 * (n - 2)
        for j in range(3):
            tris.append((last + j, tip, last + j + 1))
        positions = np.array(positions)
        tris = np.array(tris)
        # wind side faces outwards: away from the centre of the cross-sections they span
        centres = np.array(centres)[np.array(ring_of)]
        p = positions[tris]
        fn = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        away = p.mean(axis=1) - centres[tris].mean(axis=1)
        flip = (fn * away).sum(axis=1) < 0
        tris[flip] = tris[flip][:, ::-1]
        # root cap faces back along the path
        cap = np.array([0, 1, 2])
        if np.dot(np.cross(positions[1] - positions[0], positions[2] - positions[0]), T[0]) > 0:
            cap = cap[::-1]
        tris = np.vstack([tris, cap])
        return positions, np.array(uvs), tris


class Sheet:
    """A closed slab of hair spanned by several paths ("lines") side by side.

    lines:     (C+1, R, 3) points; every line runs root to tip with R points. Lines
               of different lengths give the bottom edge its points and notches.
    thickness: (C+1, R) slab thickness at each point; keep it above 0 so the inner
               and outer surfaces never share a vertex (that would break smooth normals).
    outward:   function(point) -> rough direction away from the body.
    uv:        function(s, t) -> (u, v); s runs root to tip, t across the lines.
    """

    def __init__(self, lines, thickness, outward, uv, closed=True):
        self.lines = np.asarray(lines, float)
        self.thickness = np.asarray(thickness, float)
        self.outward = outward
        self.uv = uv
        # a sheet only a few millimetres thick does not need its edges closed;
        # the two faces alone keep it visible from both sides
        self.closed = closed

    def build(self):
        L = self.lines
        ncol, nrow = L.shape[:2]
        d_s = np.gradient(L, axis=1)
        d_t = np.gradient(L, axis=0)
        n = unit(np.cross(d_s, d_t))
        ref = np.array([[self.outward(p) for p in line] for line in L])
        n *= np.sign((n * ref).sum(axis=2, keepdims=True) + 1e-12)
        inner = L
        outer = L + n * self.thickness[..., None]
        grid_uv = np.array([[self.uv(r / (nrow - 1), c / (ncol - 1)) for r in range(nrow)] for c in range(ncol)])

        positions = np.vstack([inner.reshape(-1, 3), outer.reshape(-1, 3)])
        uvs = np.vstack([grid_uv.reshape(-1, 2), grid_uv.reshape(-1, 2)])
        off = ncol * nrow

        def vid(layer, c, r):
            return layer * off + c * nrow + r

        quads = []  # (a, b, c, d, wanted normal)
        for c in range(ncol - 1):
            for r in range(nrow - 1):
                for layer, sign in ((0, -1), (1, 1)):
                    q = (vid(layer, c, r), vid(layer, c + 1, r), vid(layer, c + 1, r + 1), vid(layer, c, r + 1))
                    quads.append((q, sign * n[c, r]))
        for c, sign in ((0, -1), (ncol - 1, 1)) if self.closed else ():  # the two side edges
            for r in range(nrow - 1):
                q = (vid(0, c, r), vid(0, c, r + 1), vid(1, c, r + 1), vid(1, c, r))
                quads.append((q, sign * unit(d_t[c, r])))
        for r, sign in ((0, -1), (nrow - 1, 1)) if self.closed else ():  # root and tip edges
            for c in range(ncol - 1):
                q = (vid(0, c, r), vid(0, c + 1, r), vid(1, c + 1, r), vid(1, c, r))
                quads.append((q, sign * unit(d_s[c, r])))
        tris = []
        for (a, b, c, d), want in quads:
            for tri in ((a, b, c), (a, c, d)):
                p = positions[list(tri)]
                fn = np.cross(p[1] - p[0], p[2] - p[0])
                tris.append(tri if np.dot(fn, want) >= 0 else tri[::-1])
        return positions, uvs, np.array(tris)


def merge(parts):
    """Concatenate (positions, uvs, tris, extra...) tuples into one mesh."""
    pos, uv, tri = [], [], []
    base = 0
    for p, u, t in parts:
        pos.append(p)
        uv.append(u)
        tri.append(t + base)
        base += len(p)
    return np.vstack(pos), np.vstack(uv), np.vstack(tri)
