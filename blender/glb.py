"""Minimal GLB (binary glTF 2.0) reader/writer used by the build scripts.

Only covers what the Horse Mod meshes use: one binary buffer, plain
(non-sparse) accessors, triangle lists.
"""
import json
import struct

import numpy as np

COMPONENT_DTYPES = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}
DTYPE_COMPONENTS = {np.dtype(v): k for k, v in COMPONENT_DTYPES.items()}
TYPE_SIZES = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
SIZE_TYPES = {1: "SCALAR", 2: "VEC2", 3: "VEC3", 4: "VEC4", 16: "MAT4"}


def read(path):
    """Return (gltf_json, bin_bytes) for a .glb file."""
    with open(path, "rb") as f:
        data = f.read()
    magic, version, _length = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF" or version != 2:
        raise ValueError(f"{path}: not a glTF 2.0 binary")
    offset = 12
    gltf, binary = None, b""
    while offset < len(data):
        chunk_len, chunk_type = struct.unpack_from("<I4s", data, offset)
        chunk = data[offset + 8: offset + 8 + chunk_len]
        if chunk_type == b"JSON":
            gltf = json.loads(chunk.decode("utf-8"))
        elif chunk_type == b"BIN\x00":
            binary = chunk
        offset += 8 + chunk_len
    return gltf, binary


def accessor(gltf, binary, index):
    """Read accessor `index` as a numpy array shaped (count, components)."""
    acc = gltf["accessors"][index]
    view = gltf["bufferViews"][acc["bufferView"]]
    dtype = np.dtype(COMPONENT_DTYPES[acc["componentType"]])
    ncomp = TYPE_SIZES[acc["type"]]
    start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    stride = view.get("byteStride", 0) or dtype.itemsize * ncomp
    count = acc["count"]
    raw = np.frombuffer(binary, dtype=np.uint8, count=stride * (count - 1) + dtype.itemsize * ncomp, offset=start)
    rows = np.lib.stride_tricks.as_strided(raw, shape=(count, dtype.itemsize * ncomp), strides=(stride, 1))
    arr = np.ascontiguousarray(rows).view(dtype).reshape(count, ncomp)
    if acc.get("normalized"):
        arr = arr.astype(np.float32) / np.iinfo(dtype).max
    return arr


def primitive_arrays(gltf, binary, mesh_index=0, prim_index=0):
    """Return ({attribute_name: array}, indices) for one primitive."""
    prim = gltf["meshes"][mesh_index]["primitives"][prim_index]
    attrs = {name: accessor(gltf, binary, idx) for name, idx in prim["attributes"].items()}
    indices = accessor(gltf, binary, prim["indices"]).reshape(-1) if "indices" in prim else None
    return attrs, indices


class Builder:
    """Collects arrays into a single binary buffer and appends accessors for them."""

    def __init__(self, gltf):
        self.gltf = gltf
        self.blob = bytearray()
        gltf["bufferViews"] = []
        gltf["accessors"] = []

    def add(self, array, target=None, normalized=False, minmax=False):
        array = np.ascontiguousarray(array)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        while len(self.blob) % 4:
            self.blob.append(0)
        view = {"buffer": 0, "byteOffset": len(self.blob), "byteLength": array.nbytes}
        if target is not None:
            view["target"] = target
        self.blob += array.tobytes()
        self.gltf["bufferViews"].append(view)
        acc = {
            "bufferView": len(self.gltf["bufferViews"]) - 1,
            "componentType": DTYPE_COMPONENTS[array.dtype],
            "count": array.shape[0],
            "type": SIZE_TYPES[array.shape[1]],
        }
        if normalized:
            acc["normalized"] = True
        if minmax:
            acc["min"] = array.min(axis=0).tolist()
            acc["max"] = array.max(axis=0).tolist()
        self.gltf["accessors"].append(acc)
        return len(self.gltf["accessors"]) - 1

    def finish(self):
        while len(self.blob) % 4:
            self.blob.append(0)
        self.gltf["buffers"] = [{"byteLength": len(self.blob)}]
        return bytes(self.blob)


def write(path, gltf, binary):
    js = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    js += b" " * (-len(js) % 4)
    binary = bytes(binary) + b"\x00" * (-len(binary) % 4)
    total = 12 + 8 + len(js) + 8 + len(binary)
    with open(path, "wb") as f:
        f.write(struct.pack("<4sII", b"glTF", 2, total))
        f.write(struct.pack("<I4s", len(js), b"JSON"))
        f.write(js)
        f.write(struct.pack("<I4s", len(binary), b"BIN\x00"))
        f.write(binary)
