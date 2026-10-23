"""Builds every model and texture the mod ships.

Run with Blender's bundled Python (it has numpy):
    "<Blender>/5.2/python/bin/python.exe" blender/build.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import BODIES, MOD_MEDIA, OUT_MANES, OUT_MODELS, REFERENCE, ensure_reference, smooth_normals  # noqa: E402
import mane  # noqa: E402
import mane_texture  # noqa: E402
import tail  # noqa: E402


def main():
    ensure_reference()
    os.makedirs(os.path.join(OUT_MODELS, "Horse"), exist_ok=True)
    os.makedirs(OUT_MANES, exist_ok=True)
    print("texture:", os.path.relpath(mane_texture.build(), MOD_MEDIA))

    for name, (pos, uv, tris) in mane.build_pieces().items():
        path = os.path.join(OUT_MANES, name + ".glb")
        mane.write_piece(path, name, pos, uv, tris, smooth_normals(pos, tris))
        print(f"{name}: {len(tris)} tris, bounds {np.round(pos.min(0), 3)} .. {np.round(pos.max(0), 3)}")

    for body in BODIES:
        stats = tail.graft(os.path.join(REFERENCE, body + ".glb"), os.path.join(OUT_MODELS, "Horse", body + ".glb"),
                           with_forelock="NoHead" not in body)
        print(f"{body}: {stats}")


if __name__ == "__main__":
    main()
