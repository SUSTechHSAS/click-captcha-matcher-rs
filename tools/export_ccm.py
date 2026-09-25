#!/usr/bin/env python3
"""Convert a trained matcher (matcher.onnx or model.pt) into a .ccm weight file.

    python tools/export_ccm.py runs/w16/model.pt models/w16.ccm --calib real.npz   # shipped model
    python tools/export_ccm.py runs/w16/matcher.onnx w16.ccm                      # no calibration data
    python tools/export_ccm.py runs/s1/matcher.onnx s1.ccm --bits 8
    python tools/export_ccm.py runs/w16/matcher.onnx w16-f32.ccm --dtype f32

BatchNorm is folded into the convolutions (the ONNX export already has it folded).
Integer weights are symmetric per output channel with a per-layer bit width; the
prompt fc holds 40% of all weights and tolerates the fewest bits.

With --calib (crops from prepare_real.py, needs torch) the integer codes are
chosen by GPTQ (Frantar et al., 2022): layer by layer in forward order, each
rounding error is compensated on the not-yet-rounded weights so that the layer's
output on real captchas moves as little as possible. Default then: prompt fc
4 bits, last candidate conv 6, the rest 8 -- on 17k held-out field captchas this
changes fewer answers than plain int8 rounding. Without --calib the codes are
rounded to nearest (clipping range searched per channel below 8 bits) and the
default is the safer 5 / 6 bits. Inference always runs in float32; the format
only shrinks the file. Reading .onnx needs only numpy; .pt and --calib need torch.

File layout (little-endian):
    b"CCM1"  u8 width  u8 dtype (0 f32, 1 f16, 2 int)  u16 dim
    12 layers in forward order: p0..p3 (prompt convs), pfc, c0..c5 (cand convs), cfc
    per layer, f32/f16:  bias f32[cout], weights[cout, ...] in PyTorch order
    per layer, int:      u8 bits, scale f32[cout], bias f32[cout],
                         weights as `bits`-bit two's complement, LSB-first bitstream
"""

from __future__ import annotations

import argparse
import struct
import sys

import numpy as np

DTYPES = {"f32": 0, "f16": 1, "int": 2}
LAYERS = ("p0", "p1", "p2", "p3", "pfc", "c0", "c1", "c2", "c3", "c4", "c5", "cfc")
DEFAULT_BITS = {True: "pfc=4,c5=6", False: "pfc=5,c5=6"}  # with / without --calib
BN_EPS = 1e-5


def expected_shapes(width: int, dim: int) -> list[tuple[int, ...]]:
    w = width
    conv = lambda cin, cout: (cout, cin, 3, 3)  # noqa: E731
    return [conv(1, w), conv(w, w), conv(w, 2 * w), conv(2 * w, 2 * w), (dim, 2 * w * 16),
            conv(1, w), conv(w, w), conv(w, 2 * w), conv(2 * w, 2 * w), conv(2 * w, 4 * w),
            conv(4 * w, 4 * w), (dim, 4 * w)]


# ---------------------------------------------------------------- .onnx (numpy only)
def _varint(buf: bytes, i: int) -> tuple[int, int]:
    r = s = 0
    while True:
        b = buf[i]
        i += 1
        r |= (b & 0x7F) << s
        s += 7
        if b < 0x80:
            return r, i


def _fields(buf: bytes):
    """Yield (field number, wire type, value) of one protobuf message."""
    i = 0
    while i < len(buf):
        key, i = _varint(buf, i)
        f, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(buf, i)
        elif wt == 1:
            v, i = buf[i:i + 8], i + 8
        elif wt == 2:
            n, i = _varint(buf, i)
            v, i = buf[i:i + n], i + n
        elif wt == 5:
            v, i = buf[i:i + 4], i + 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wt}")
        yield f, wt, v


def _tensor(buf: bytes) -> tuple[str, np.ndarray]:
    dims, dtype, name, raw, floats = [], 0, "", None, []
    for f, wt, v in _fields(buf):
        if f == 1:  # dims (packed or not)
            if wt == 2:
                j = 0
                while j < len(v):
                    d, j = _varint(v, j)
                    dims.append(d)
            else:
                dims.append(v)
        elif f == 2:
            dtype = v
        elif f == 4:  # float_data
            floats += list(struct.unpack(f"<{len(v) // 4}f", v)) if wt == 2 else [struct.unpack("<f", v)[0]]
        elif f == 8:
            name = v.decode()
        elif f == 9:
            raw = v
    if dtype != 1:
        raise ValueError(f"initializer {name}: only float32 is supported (got type {dtype})")
    arr = np.frombuffer(raw, "<f4") if raw is not None else np.array(floats, np.float32)
    return name, arr.reshape(dims).astype(np.float32)


def load_onnx(path: str) -> list[tuple[np.ndarray, np.ndarray]]:
    model = open(path, "rb").read()
    graph = next(v for f, _, v in _fields(model) if f == 7)
    inits, nodes = {}, []
    for f, _, v in _fields(graph):
        if f == 5:
            name, arr = _tensor(v)
            inits[name] = arr
        elif f == 1:
            node = {"in": [], "out": [], "op": "", "attr": {}}
            for g, _, u in _fields(v):
                if g == 1:
                    node["in"].append(u.decode())
                elif g == 2:
                    node["out"].append(u.decode())
                elif g == 4:
                    node["op"] = u.decode()
                elif g == 5:
                    aname, ival, fval = "", None, None
                    for h, _, a in _fields(u):
                        if h == 1:
                            aname = a.decode()
                        elif h == 2:
                            fval = struct.unpack("<f", a)[0]
                        elif h == 3:
                            ival = a
                    node["attr"][aname] = ival if ival is not None else fval
            nodes.append(node)

    layers: list[list] = []
    producer: dict[str, int] = {}  # tensor name -> index into layers of the conv producing it
    for n in nodes:
        op, ins = n["op"], n["in"]
        if op == "Conv":
            w = inits[ins[1]]
            b = inits[ins[2]] if len(ins) > 2 and ins[2] else np.zeros(w.shape[0], np.float32)
            producer[n["out"][0]] = len(layers)
            layers.append([w, b])
        elif op == "BatchNormalization":  # unfolded export: fold into the producing conv
            k = producer[ins[0]]
            g, beta, mean, var = (inits[x].astype(np.float64) for x in ins[1:5])
            s = g / np.sqrt(var + float(n["attr"].get("epsilon", BN_EPS)))
            w, b = layers[k]
            layers[k] = [(w * s[:, None, None, None]).astype(np.float32),
                         ((b - mean) * s + beta).astype(np.float32)]
            producer[n["out"][0]] = k
        elif op == "Gemm":
            w = inits[ins[1]]
            if not n["attr"].get("transB", 0):
                w = w.T
            b = inits[ins[2]] if len(ins) > 2 and ins[2] else np.zeros(w.shape[0], np.float32)
            layers.append([np.ascontiguousarray(w), b])
    return [(w, b) for w, b in layers]


# ---------------------------------------------------------------- .pt (torch)
def load_pt(path: str) -> list[tuple[np.ndarray, np.ndarray]]:
    import torch

    sd = torch.load(path, map_location="cpu")
    out = []
    for tower, convs in (("prompt", (0, 1, 3, 4)), ("cand", (0, 1, 3, 4, 6, 7))):
        for i in convs:
            p = f"{tower}.body.{i}"
            w = sd[f"{p}.0.weight"].double().numpy()
            g, beta = sd[f"{p}.1.weight"].double().numpy(), sd[f"{p}.1.bias"].double().numpy()
            mean, var = sd[f"{p}.1.running_mean"].double().numpy(), sd[f"{p}.1.running_var"].double().numpy()
            s = g / np.sqrt(var + BN_EPS)
            out.append(((w * s[:, None, None, None]).astype(np.float32), (beta - mean * s).astype(np.float32)))
        out.append((sd[f"{tower}.head.weight"].numpy().astype(np.float32),
                    sd[f"{tower}.head.bias"].numpy().astype(np.float32)))
    return out


# ---------------------------------------------------------------- writer
def parse_bits(spec: str) -> list[int]:
    """Bits per layer from "8" or e.g. "pfc=5,c5=6" (unlisted layers: 8)."""
    bits = [8] * len(LAYERS)
    for item in filter(None, spec.split(",")):
        name, _, b = item.rpartition("=")
        for i, layer in enumerate(LAYERS):
            if not name or name == layer:
                bits[i] = int(b)
        if name and name not in LAYERS:
            raise ValueError(f"unknown layer {name!r}; layers are {', '.join(LAYERS)}")
    if not all(2 <= b <= 8 for b in bits):
        raise ValueError("bits must be between 2 and 8")
    return bits


def quantize(flat: np.ndarray, bits: int, search: bool | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric per-row round-to-nearest -> (int codes, f32 scales). With `search`
    (default: below 8 bits) each row's clipping range is the one, among 50-100%
    of max|w|, with the least squared error."""
    qmax = 2 ** (bits - 1) - 1
    amax = np.abs(flat).max(1).astype(np.float32)
    amax[amax == 0] = 1.0
    best_q = best_s = best_err = None
    search = bits < 8 if search is None else search
    for a in (np.linspace(0.5, 1.0, 51, dtype=np.float32) if search else [np.float32(1.0)]):
        s = (amax * a / qmax).astype(np.float32)
        q = np.clip(np.round(flat / s[:, None]), -qmax, qmax)
        err = ((q * s[:, None] - flat) ** 2).sum(1)
        if best_err is None:
            best_q, best_s, best_err = q, s, err
        else:
            better = err < best_err
            best_q[better], best_s[better], best_err = q[better], s[better], np.minimum(err, best_err)
    return best_q.astype(np.int64), best_s


def pack(q: np.ndarray, bits: int) -> bytes:
    """Two's complement `bits`-bit values, LSB-first bitstream, padded to a byte."""
    u = q.ravel() & ((1 << bits) - 1)
    stream = ((u[:, None] >> np.arange(bits)) & 1).astype(np.uint8).ravel()
    return np.packbits(stream, bitorder="little").tobytes()


def gptq(layers: list[tuple[np.ndarray, np.ndarray]], bits: list[int], calib: str, n: int,
         damp: float = 0.01) -> list[tuple[np.ndarray, np.ndarray]]:
    """(codes, scales) per layer by sequential GPTQ on `n` calibration captchas."""
    import torch
    import torch.nn.functional as F

    z = np.load(calib)
    prompts = torch.from_numpy(np.asarray(z["prompts"][:n], np.float32)).reshape(-1, 1, 16, 16)
    cands = torch.from_numpy(np.asarray(z["cands"][:n], np.float32)).reshape(-1, 1, 40, 40)
    ws = [torch.from_numpy(w.copy()) for w, _ in layers]  # quantized in place, layer by layer
    bs = [torch.from_numpy(b) for _, b in layers]

    def inputs(li: int, x: torch.Tensor) -> torch.Tensor:
        """Rows of what layer li sees (im2col for convs), through the layers already done."""
        first, pools = (0, (1, 3)) if li <= 4 else (5, (6, 8))
        for k in range(first, li):
            stride = 2 if k == 5 else 1
            x = torch.relu(F.conv2d(x, ws[k], bs[k], stride, 1))
            if k in pools:
                x = F.max_pool2d(x, 2)
        if li == 4:
            return x.flatten(1)
        if li == 11:
            return x.mean((2, 3))
        return F.unfold(x, 3, padding=1, stride=2 if li == 5 else 1).transpose(1, 2).reshape(-1, x.shape[1] * 9)

    out = []
    for li, ((w, _), nbits) in enumerate(zip(layers, bits)):
        W = torch.from_numpy(w.reshape(w.shape[0], -1)).double()
        H = torch.zeros(W.shape[1], W.shape[1], dtype=torch.float64)
        with torch.no_grad():
            for x in (prompts if li <= 4 else cands).split(256):
                X = inputs(li, x).double()
                H += X.T @ X
        _, scale = quantize(w.reshape(w.shape[0], -1).astype(np.float32), nbits, search=True)
        s = torch.from_numpy(scale).double()
        qmax = 2 ** (nbits - 1) - 1
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0
        H += damp * torch.diag(H).mean() * torch.eye(H.shape[0], dtype=H.dtype)
        perm = torch.argsort(torch.diag(H), descending=True)  # most active inputs first
        W, H = W[:, perm], H[perm][:, perm]
        U = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
        Q = torch.zeros_like(W)
        for i in range(W.shape[1]):
            q = torch.clamp(torch.round(W[:, i] / s), -qmax, qmax)
            Q[:, i] = q
            err = (W[:, i] - q * s) / U[i, i]
            W[:, i + 1:] -= err[:, None] * U[i, i + 1:][None, :]
        codes = Q[:, torch.argsort(perm)]
        ws[li] = (codes * s[:, None]).float().reshape(w.shape)
        out.append((codes.numpy().astype(np.int64), scale))
    return out


def encode(layers: list[tuple[np.ndarray, np.ndarray]], dtype: str, bits: str = "8",
           calib: str | None = None, calib_n: int = 2048) -> bytes:
    if len(layers) != 12:
        raise ValueError(f"expected 12 conv/fc layers (4+1 prompt, 6+1 cand), found {len(layers)}")
    width, dim = layers[0][0].shape[0], layers[4][0].shape[0]
    shapes = expected_shapes(width, dim)
    for k, ((w, b), shape) in enumerate(zip(layers, shapes)):
        if tuple(w.shape) != shape or b.shape != (shape[0],):
            raise ValueError(f"layer {LAYERS[k]}: weight {w.shape} / bias {b.shape}, expected {shape}")
        if not (np.isfinite(w).all() and np.isfinite(b).all()):
            raise ValueError(f"layer {LAYERS[k]}: non-finite weights")
    if width % 16 or dim % 16 or width > 64 or dim > 512:
        raise ValueError(f"width {width} / dim {dim}: both must be multiples of 16 (width <= 64, dim <= 512)")

    nbits = parse_bits(bits)
    if dtype == "int":
        codes = (gptq(layers, nbits, calib, calib_n) if calib else
                 [quantize(w.reshape(w.shape[0], -1).astype(np.float32), b) for (w, _), b in zip(layers, nbits)])
    parts = [b"CCM1", struct.pack("<BBH", width, DTYPES[dtype], dim)]
    for k, ((w, b), nb) in enumerate(zip(layers, nbits)):
        bias = b.astype("<f4").tobytes()
        if dtype == "int":
            q, scale = codes[k]
            parts += [bytes([nb]), scale.astype("<f4").tobytes(), bias, pack(q, nb)]
        else:
            parts += [bias, w.reshape(w.shape[0], -1).astype("<f2" if dtype == "f16" else "<f4").tobytes()]
    return b"".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src", help="matcher.onnx or model.pt")
    ap.add_argument("dst", help="output .ccm")
    ap.add_argument("--dtype", choices=sorted(DTYPES), default="int")
    ap.add_argument("--bits", default=None,
                    help=f'int bits: "8" for all layers, or per layer like "pfc=4,c5=6" (others 8); '
                         f'layers: {" ".join(LAYERS)}. Default: "{DEFAULT_BITS[True]}" with --calib, '
                         f'"{DEFAULT_BITS[False]}" without')
    ap.add_argument("--calib", default=None, help="npz with prompts/cands crops (prepare_real.py) for GPTQ")
    ap.add_argument("--calib-n", type=int, default=2048, help="calibration captchas to use (default 2048)")
    args = ap.parse_args()
    args.bits = args.bits or DEFAULT_BITS[bool(args.calib)]

    layers = load_pt(args.src) if args.src.endswith((".pt", ".pth")) else load_onnx(args.src)
    blob = encode(layers, args.dtype, args.bits, args.calib, args.calib_n)
    with open(args.dst, "wb") as fh:
        fh.write(blob)
    n = sum(w.size for w, _ in layers)
    kind = (f"int ({args.bits} bits, {'GPTQ' if args.calib else 'round to nearest'})"
            if args.dtype == "int" else args.dtype)
    print(f"{args.src} -> {args.dst}: width {layers[0][0].shape[0]}, {n:,} weights, "
          f"{kind}, {len(blob):,} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
