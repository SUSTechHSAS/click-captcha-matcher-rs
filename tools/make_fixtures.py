#!/usr/bin/env python3
"""Regenerate tests/fixtures: synthetic captchas with known answers + JPEG variants.

    python tools/make_fixtures.py --ref ../click-captcha-matcher --fonts /dir/with/fonts

Captchas are drawn by the reference repo's synth.py using freely licensed fonts
only (Droid Sans Fallback, Noto Sans CJK); no field captcha is committed.
expected.txt: one line per file,

    <file> <status> <fnv1a64 of Pillow's convert("L")> [<x-y,x-y,x-y,x-y> <margin>]

status 0 = decodes, otherwise the libccm error code the file must produce. The
answer column is the reference solver's (and the drawn ground truth's) answer.
"""

from __future__ import annotations

import argparse
import io
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

FREE_FONTS = ("DroidSansFallbackFull.ttf", "NotoSansCJK-Regular.ttc")
OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


def fnv1a(data: bytes) -> str:
    h = 0xCBF29CE484222325
    for b in data:
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def jpeg(im: Image.Image, **kw) -> bytes:
    buf = io.BytesIO()
    im.save(buf, "JPEG", **kw)
    return buf.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ref", required=True, help="checkout of the Python click-captcha-matcher")
    ap.add_argument("--fonts", required=True, help=f"directory containing {' / '.join(FREE_FONTS)}")
    ap.add_argument("--count", type=int, default=4)
    args = ap.parse_args()

    sys.path.insert(0, args.ref)
    import geometry as G
    import synth
    from solver import CaptchaSolver

    with tempfile.TemporaryDirectory() as tmp:
        for f in FREE_FONTS:
            (Path(tmp) / f).symlink_to(Path(args.fonts, f).resolve())
        synth.FONT_DIR = tmp
        gen = synth.Generator(seed=2026)
    ref = CaptchaSolver(Path(args.ref) / "runs" / "w16" / "matcher.onnx")
    rng = np.random.default_rng(7)

    OUT.mkdir(parents=True, exist_ok=True)
    lines, first = [], None
    while len([l for l in lines if l.startswith("synth")]) < args.count:
        s = gen.sample()
        # Real captchas are YCbCr 4:2:0 with a little chroma noise; so are these.
        rgb = np.repeat(s.gray[:, :, None], 3, axis=2).astype(np.int16)
        rgb += rng.integers(-6, 7, rgb.shape, dtype=np.int16)
        im = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
        data = jpeg(im, quality=80)
        points, margin = ref.solve(data)
        centers = G.candidate_centers(np.asarray(Image.open(io.BytesIO(data)).convert("L")))
        truth = [[int(round(centers[t, 0])), int(round(centers[t, 1]))] for t in s.target]
        if points != truth or margin < 0.1:  # keep clear-cut cases: robust to int8 noise
            continue
        name = f"synth{sum(l.startswith('synth') for l in lines)}.jpg"
        first = first or im
        (OUT / name).write_bytes(data)
        gray = np.asarray(Image.open(io.BytesIO(data)).convert("L")).tobytes()
        answer = ",".join(f"{x}-{y}" for x, y in points)
        lines.append(f"{name} 0 {fnv1a(gray)} {answer} {margin:.6f}")

    variants = {
        "v444.jpg": jpeg(first, quality=85, subsampling=0),
        "v422.jpg": jpeg(first, quality=85, subsampling=1),
        "vgray.jpg": jpeg(first.convert("L"), quality=85),
        "vrestart.jpg": jpeg(first, quality=75, restart_marker_blocks=3),
        "vq98.jpg": jpeg(first, quality=98),
        "vprogressive.jpg": jpeg(first, quality=80, progressive=True),
        "vsmall.jpg": jpeg(first.resize((125, 40)), quality=80),
    }
    for name, data in variants.items():
        (OUT / name).write_bytes(data)
        im = Image.open(io.BytesIO(data)).convert("L")
        if im.size != (G.WIDTH, G.HEIGHT):
            lines.append(f"{name} -4 -")
        elif name == "vprogressive.jpg":
            lines.append(f"{name} -3 {fnv1a(np.asarray(im).tobytes())}")
        else:
            lines.append(f"{name} 0 {fnv1a(np.asarray(im).tobytes())}")
    (OUT / "expected.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
