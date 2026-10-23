"""Evaluate glTF node animations (linear/step TRS) into world matrices with numpy."""
import numpy as np

import glb


def quat_to_mat(q):
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def trs_matrix(t, r, s):
    m = np.eye(4)
    m[:3, :3] = quat_to_mat(np.asarray(r, float)) * np.asarray(s, float)
    m[:3, 3] = t
    return m


def slerp(a, b, f):
    d = np.dot(a, b)
    if d < 0:
        b, d = -b, -d
    if d > 0.9995:
        q = a + f * (b - a)
        return q / np.linalg.norm(q)
    th = np.arccos(d)
    return (np.sin((1 - f) * th) * a + np.sin(f * th) * b) / np.sin(th)


class Animated:
    def __init__(self, path):
        self.g, self.b = glb.read(path)
        g = self.g
        self.parent = {}
        for i, n in enumerate(g["nodes"]):
            for c in n.get("children", []):
                self.parent[c] = i
        self.names = {n.get("name"): i for i, n in reversed(list(enumerate(g["nodes"])))}
        self.anims = []
        for a in g.get("animations", []):
            chans = {}
            for ch in a["channels"]:
                smp = a["samplers"][ch["sampler"]]
                times = glb.accessor(g, self.b, smp["input"]).reshape(-1)
                vals = glb.accessor(g, self.b, smp["output"]).astype(float)
                chans[(ch["target"]["node"], ch["target"]["path"])] = (times, vals, smp.get("interpolation", "LINEAR"))
            end = max(t[-1] for t, _, _ in chans.values())
            self.anims.append((a.get("name"), chans, end))

    def local(self, node, chans, time):
        n = self.g["nodes"][node]
        t = np.array(n.get("translation", [0, 0, 0]), float)
        r = np.array(n.get("rotation", [0, 0, 0, 1]), float)
        s = np.array(n.get("scale", [1, 1, 1]), float)
        out = {"translation": t, "rotation": r, "scale": s}
        for path in out:
            key = (node, path)
            if chans is None or key not in chans:
                continue
            times, vals, interp = chans[key]
            if time <= times[0]:
                out[path] = vals[0]
            elif time >= times[-1]:
                out[path] = vals[-1]
            else:
                i = np.searchsorted(times, time) - 1
                f = (time - times[i]) / (times[i + 1] - times[i])
                if interp == "STEP":
                    out[path] = vals[i]
                elif path == "rotation":
                    out[path] = slerp(vals[i], vals[i + 1], f)
                else:
                    out[path] = vals[i] * (1 - f) + vals[i + 1] * f
        return trs_matrix(out["translation"], out["rotation"], out["scale"])

    def world(self, node, chans=None, time=0.0, cache=None):
        cache = {} if cache is None else cache
        if node in cache:
            return cache[node]
        m = self.local(node, chans, time)
        if node in self.parent:
            m = self.world(self.parent[node], chans, time, cache) @ m
        cache[node] = m
        return m
