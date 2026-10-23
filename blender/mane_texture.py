"""Generates the mane's strand texture: light grey straight strands, tinted in game
by each breed's mane colour. Strands run along v, with the roots at v=0 shaded
where they lie under the hair above them.
"""
import os
import struct
import zlib

import numpy as np

from common import MOD_MEDIA
from mane import HAIR_V

SIZE = 256
MANE_TEXTURE = os.path.join(MOD_MEDIA, "textures", "HorseManeTailImproved", "ManeStrands.png")


def write_png(path, rgba):
    h, w, _ = rgba.shape
    raw = b"".join(b"\x00" + rgba[y].tobytes() for y in range(h))

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)))
        f.write(chunk(b"IDAT", zlib.compress(raw, 9)))
        f.write(chunk(b"IEND", b""))


def strands():
    rng = np.random.default_rng(1337)
    x = np.arange(SIZE * 4) / 4.0  # supersampled across
    # a few fine strand lines over a smooth lock; too many lines look like striped cloth
    fine = np.zeros_like(x)
    for _ in range(45):
        c, w, a = rng.uniform(0, SIZE), rng.uniform(0.6, 1.3), rng.uniform(0.3, 1.0)
        d = np.abs((x - c + SIZE / 2) % SIZE - SIZE / 2)
        fine = np.maximum(fine, a * np.exp(-0.5 * (d / w) ** 2))
    # each lock is darker towards its edges (u = 0 and 1), so neighbouring locks separate
    u = x / SIZE
    edge = 0.70 + 0.30 * np.sin(np.pi * u) ** 0.7
    across = (0.88 + 0.12 * fine) * edge

    v = (np.arange(SIZE) + 0.5) / SIZE
    # strands drift a little sideways along their length so they do not look ruled
    shift = 1.4 * np.sin(2 * np.pi * v * 1.3) + 0.7 * np.sin(2 * np.pi * v * 3.7 + 1.0)
    img = np.empty((SIZE, SIZE))
    for row in range(SIZE):
        xs = (np.arange(SIZE) + 0.5 + shift[row]) % SIZE
        img[row] = np.interp(xs, x, across, period=SIZE)
    # shade the roots, where hair lies under the hair above it
    root = np.clip((v - HAIR_V[0]) / 0.08, 0, 1)
    img *= (0.78 + 0.22 * root)[:, None]
    gray = np.clip(img * 255, 0, 255).astype(np.uint8)
    return np.dstack([gray, gray, gray, np.full_like(gray, 255)])


def build():
    os.makedirs(os.path.dirname(MANE_TEXTURE), exist_ok=True)
    write_png(MANE_TEXTURE, strands())
    return MANE_TEXTURE
