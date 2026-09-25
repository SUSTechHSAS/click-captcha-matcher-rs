"""click-captcha-matcher (native): captcha JPEG -> 4 click points, via libccm + ctypes.

Drop-in replacement for the Python solver.py; needs no numpy, PIL or onnxruntime:

    from solver import CaptchaSolver

    solver = CaptchaSolver()                    # embedded w16 model; or CaptchaSolver("s1.ccm")
    points, margin = solver.solve(jpeg_bytes)   # points: [[x, y]] * 4, in prompt order
    if margin < solver.min_margin:              # low confidence
        ...                                     # fetch a fresh captcha instead of submitting

The shared library is looked up in $CCM_LIB, next to this file, in PyInstaller's
bundle directory, and in ../target/release (a source checkout).
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

WIDTH, HEIGHT = 250, 80
LIB_NAME = {"win32": "ccm.dll", "darwin": "libccm.dylib"}.get(sys.platform, "libccm.so")
ERRORS = {-1: "bad argument", -2: "not a valid JPEG", -3: "unsupported JPEG variant",
          -4: f"not a {WIDTH}x{HEIGHT} image", -5: "bad model file"}


class CaptchaError(ValueError):
    """The image could not be solved (bad JPEG, wrong size, ...)."""

    def __init__(self, code: int):
        super().__init__(ERRORS.get(code, f"error {code}"))
        self.code = code


def find_library() -> str:
    here = Path(__file__).resolve().parent
    candidates = [os.environ.get("CCM_LIB"), here / LIB_NAME,
                  Path(getattr(sys, "_MEIPASS", here)) / LIB_NAME,
                  here.parent / "target" / "release" / LIB_NAME]
    for c in candidates:
        if c and Path(c).is_file():
            return str(c)
    raise OSError(f"{LIB_NAME} not found (build it with `cargo build --release`, "
                  f"copy it next to {Path(__file__).name}, or set CCM_LIB)")


def load_library(path: str | None = None) -> ctypes.CDLL:
    lib = ctypes.CDLL(path or find_library())
    u8p, i32p, f32p = ctypes.c_char_p, ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_float)
    lib.ccm_new.argtypes = [u8p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_int)]
    lib.ccm_new.restype = ctypes.c_void_p
    lib.ccm_free.argtypes = [ctypes.c_void_p]
    lib.ccm_free.restype = None
    lib.ccm_solve.argtypes = [ctypes.c_void_p, u8p, ctypes.c_size_t, i32p, f32p]
    lib.ccm_solve_gray.argtypes = [ctypes.c_void_p, u8p, i32p, f32p]
    lib.ccm_decode.argtypes = [u8p, ctypes.c_size_t, ctypes.c_char_p]
    lib.ccm_similarity.argtypes = [ctypes.c_void_p, u8p, f32p, f32p]
    lib.ccm_version.restype = lib.ccm_isa.restype = ctypes.c_char_p
    lib.ccm_version.argtypes = lib.ccm_isa.argtypes = []
    return lib


class CaptchaSolver:
    def __init__(self, model: str | os.PathLike | None = None, min_margin: float = 0.0,
                 threads: int = 1, lib: str | None = None):
        """`model`: a .ccm file (tools/export_ccm.py), or None for the embedded w16.
        `threads` is accepted for compatibility; solving is single-threaded."""
        self._lib = load_library(lib)
        data = None
        if model is not None:
            path = Path(model)
            if path.suffix == ".onnx":
                raise ValueError(f"{path}: convert it first: python tools/export_ccm.py {path} model.ccm")
            data = path.read_bytes()
        err = ctypes.c_int(0)
        self._handle = self._lib.ccm_new(data, len(data) if data else 0, ctypes.byref(err))
        if not self._handle:
            raise CaptchaError(err.value or -5)
        self.min_margin = min_margin

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self._lib.ccm_free(self._handle)
            self._handle = None

    __del__ = close

    @staticmethod
    def to_gray(image) -> bytes:
        """numpy (80, 250) gray or (80, 250, 3) color array -> 20000 bytes, as the Python solver."""
        import numpy as np

        arr = np.asarray(image)
        if arr.ndim == 3:  # channels are ~equal in this captcha
            arr = arr.mean(axis=2)
        if arr.shape != (HEIGHT, WIDTH):
            raise CaptchaError(-4)
        return np.ascontiguousarray(arr.astype(np.uint8)).tobytes()

    def solve(self, image) -> tuple[list[list[int]], float]:
        """JPEG bytes, a file path, or a numpy image -> ([[x, y]] * 4, margin)."""
        if isinstance(image, (str, os.PathLike)):
            image = Path(image).read_bytes()
        pts, margin = (ctypes.c_int32 * 8)(), ctypes.c_float()
        if isinstance(image, (bytes, bytearray, memoryview)):
            data = bytes(image)
            rc = self._lib.ccm_solve(self._handle, data, len(data), pts, ctypes.byref(margin))
        else:
            rc = self._lib.ccm_solve_gray(self._handle, self.to_gray(image), pts, ctypes.byref(margin))
        if rc != 0:
            raise CaptchaError(rc)
        return [[pts[2 * i], pts[2 * i + 1]] for i in range(4)], margin.value

    def decode(self, jpeg: bytes) -> bytes:
        """JPEG -> 20000 bytes of gray, identical to Pillow's Image.open(f).convert("L")."""
        gray = ctypes.create_string_buffer(WIDTH * HEIGHT)
        rc = self._lib.ccm_decode(bytes(jpeg), len(jpeg), gray)
        if rc != 0:
            raise CaptchaError(rc)
        return gray.raw

    def similarity(self, gray: bytes) -> tuple[list[list[float]], list[list[float]]]:
        """Gray image (20000 bytes) -> 4x6 cosine similarity and the 6 click centers."""
        sim, centers = (ctypes.c_float * 24)(), (ctypes.c_float * 12)()
        rc = self._lib.ccm_similarity(self._handle, bytes(gray), sim, centers)
        if rc != 0:
            raise CaptchaError(rc)
        return ([list(sim[6 * i:6 * i + 6]) for i in range(4)],
                [list(centers[2 * i:2 * i + 2]) for i in range(6)])

    @property
    def info(self) -> str:
        return f"libccm {self._lib.ccm_version().decode()} ({self._lib.ccm_isa().decode()} kernel)"


if __name__ == "__main__":
    solver = CaptchaSolver()
    print(solver.info)
    for arg in sys.argv[1:]:
        pts, m = solver.solve(arg)
        print(arg, ",".join(f"{x}-{y}" for x, y in pts), f"{m:.6f}")
