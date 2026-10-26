"""Convert KeyShot r8.obj → static/models/r8.json for the WebGL viewer.

    python tools/preprocess_r8.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OBJ = ROOT / "3d-objects" / "r8.obj"
OUT = ROOT / "static" / "models" / "r8.json"
SKIP = {"interior 1", "interior 2", "interior roof"}
TARGET_LENGTH_M = 4.5


def resolve(i: str, n: int) -> int:
    v = int(i)
    return (v - 1) if v > 0 else (n + v)


def main():
    verts: list[tuple[float, float, float]] = []
    tris: list[tuple[int, int, int]] = []
    cur_ok = True

    with OBJ.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                verts.append(tuple(map(float, line.split()[1:4])))
            elif line.startswith("g "):
                cur_ok = line[2:].strip() not in SKIP
            elif line.startswith("f ") and cur_ok:
                n = len(verts)
                idx = [resolve(p.split("/")[0], n) for p in line.split()[1:]]
                for i in range(1, len(idx) - 1):
                    tris.append((idx[0], idx[i], idx[i + 1]))

    used = sorted({i for t in tris for i in t})
    cx = sum(verts[i][0] for i in used) / len(used)
    cy = min(verts[i][1] for i in used)
    cz = sum(verts[i][2] for i in used) / len(used)
    zspan = max(verts[i][2] for i in used) - min(verts[i][2] for i in used)
    scale = TARGET_LENGTH_M / zspan if zspan else 0.001

    remap = {old: i for i, old in enumerate(used)}
    pos: list[float] = []
    for old in used:
        x, y, z = verts[old]
        x, y, z = (x - cx) * scale, (y - cy) * scale, (z - cz) * scale
        # KeyShot X=width Y=up Z=length → viewer +X forward, +Y up
        pos.extend([z, y, -x])

    ind: list[int] = []
    for a, b, c in tris:
        ind.extend([remap[a], remap[b], remap[c]])

    xs, ys, zs = pos[0::3], pos[1::3], pos[2::3]
    out = {
        "forward": "+x",
        "up": "+y",
        "length": max(xs) - min(xs),
        "width": max(zs) - min(zs),
        "height": max(ys) - min(ys),
        "positions": [round(v, 5) for v in pos],
        "indices": ind,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print(
        f"Wrote {OUT}  tris={len(ind)//3}  "
        f"L={out['length']:.2f} W={out['width']:.2f} H={out['height']:.2f} m  "
        f"({OUT.stat().st_size/1e6:.1f} MB)"
    )


if __name__ == "__main__":
    main()
