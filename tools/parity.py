#!/usr/bin/env python3
"""Check libccm against the Python/onnxruntime solver it replaces.

    python tools/parity.py --ref ../click-captcha-matcher samples/ oracle/w16 oracle/s1

1. decoder: libccm's gray image == Pillow's Image.open(f).convert("L"), pixel for pixel
2. network: an fp32 export vs onnxruntime (tests the kernels), and the shipped
   quantized model vs onnxruntime: same click points? how far does the margin move?
3. oracle: directories with a results.jsonl from the live server ("passed": true
   rows are confirmed answers): does libccm give the same answer?

Needs numpy, pillow and onnxruntime, like the reference solver.
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("images", nargs="+", help="JPEG files or directories")
    ap.add_argument("--ref", required=True, help="checkout of the Python click-captcha-matcher")
    ap.add_argument("--run", default="w16", help="model under <ref>/runs/ (default w16)")
    ap.add_argument("--ccm", default=None, help=".ccm to test instead of the embedded model")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    sys.path.insert(0, args.ref)
    from assign import assign  # noqa: E402  (reference implementation)
    ref_solver = load_module("ref_solver", Path(args.ref) / "solver.py")
    native = load_module("ccm_solver", HERE.parent / "python" / "solver.py")
    export = load_module("export_ccm", HERE / "export_ccm.py")

    onnx = Path(args.ref) / "runs" / args.run / "matcher.onnx"
    ref = ref_solver.CaptchaSolver(onnx, threads=1)
    ours = native.CaptchaSolver(args.ccm)
    with tempfile.TemporaryDirectory() as tmp:
        f32 = Path(tmp) / "f32.ccm"
        f32.write_bytes(export.encode(export.load_onnx(str(onnx)), "f32"))
        ours_f32 = native.CaptchaSolver(f32)
    print(ours.info, "| model:", args.ccm or "embedded w16", "| reference:", onnx)

    files, oracle = [], {}
    for item in args.images:
        p = Path(item)
        if p.is_dir():
            files += sorted(p.glob("*.jpg"))
            if (p / "results.jsonl").is_file():
                for line in (p / "results.jsonl").read_text().splitlines():
                    rec = json.loads(line)
                    oracle[str(p / rec["file"])] = rec
        else:
            files.append(p)
    if args.limit:
        files = files[:args.limit]

    n = len(files)
    dec_bad = centers_bad = f32_flip = q_flip = q_pts = 0
    f32_dsim = q_dsim = 0.0
    q_dmargin, flips = [], []
    orc_total = orc_ok = orc_ref_ok = 0
    for f in files:
        data = f.read_bytes()
        gray = np.asarray(Image.open(io.BytesIO(data)).convert("L"))
        mine = np.frombuffer(ours.decode(data), np.uint8).reshape(gray.shape)
        dec_bad += not np.array_equal(gray, mine)

        rsim, rcenters = ref.similarity(gray)
        rslots, _, rmargin = assign(rsim)
        rpoints = [[int(round(rcenters[s, 0])), int(round(rcenters[s, 1]))] for s in rslots]

        fsim, fcenters = ours_f32.similarity(mine.tobytes())
        centers_bad += not np.array_equal(np.float32(fcenters), rcenters)
        f32_dsim = max(f32_dsim, float(np.abs(np.float32(fsim) - rsim).max()))
        f32_flip += not np.array_equal(assign(np.float32(fsim))[0], rslots)

        qsim, _ = ours.similarity(mine.tobytes())
        q_dsim = max(q_dsim, float(np.abs(np.float32(qsim) - rsim).max()))
        points, margin = ours.solve(data)
        q_dmargin.append(margin - rmargin)
        if points != rpoints:
            q_pts += 1
            q_flip += not np.array_equal(assign(np.float32(qsim))[0], rslots)
            flips.append((f.name, round(rmargin, 4)))

        rec = oracle.get(str(f))
        if rec and rec.get("passed"):
            orc_total += 1
            orc_ok += points == rec["points"]
            orc_ref_ok += rpoints == rec["points"]

    dm = np.abs(np.array(q_dmargin))
    print(f"images                       {n}")
    print(f"decoded gray != Pillow       {dec_bad}")
    print(f"click centers != reference   {centers_bad}")
    print(f"fp32 export: max |dsim|      {f32_dsim:.2e}   assignment changes {f32_flip}")
    print(f"shipped model: max |dsim|    {q_dsim:.4f}   |dmargin| p50 {np.median(dm):.4f} "
          f"p99 {np.percentile(dm, 99):.4f} max {dm.max():.4f}")
    print(f"shipped model: points differ {q_pts} ({q_flip} assignment changes)"
          + (f"; reference margins there: {[m for _, m in flips][:10]}" if flips else ""))
    if orc_total:
        print(f"server-verified answers      {orc_ok}/{orc_total} reproduced "
              f"(Python reference: {orc_ref_ok}/{orc_total})")
    return 0 if dec_bad == 0 and centers_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
