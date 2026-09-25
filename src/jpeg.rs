//! Baseline JPEG -> 250x80 gray, bit-exact with Pillow's `Image.open(f).convert("L")`.
//!
//! Pillow decodes through libjpeg-turbo with its defaults (accurate integer IDCT,
//! "fancy" triangle-filter chroma upsampling, table-free integer YCbCr->RGB) and
//! then applies ITU-R 601 luma with rounding. Every step below reproduces that
//! integer arithmetic, so the network sees exactly the pixels the Python solver
//! feeds it. Only what a captcha needs is supported: 8-bit baseline / extended
//! sequential Huffman, 1 or 3 components, 1x1 / 2x1 / 2x2 sampling ratios.

use crate::geometry::{H, W};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// Not a JPEG, or a corrupt one.
    Format,
    /// A valid JPEG using something left out here (progressive, 12-bit, CMYK, ...).
    Unsupported,
    /// Decodable, but not 250x80.
    Size,
}

/// Component planes are stored MCU-padded: at most 256x80 samples each.
pub const PLANE_W: usize = 256;
pub const PLANE: usize = PLANE_W * H;
/// Scratch needed by [`decode`]: three planes.
pub const SCRATCH: usize = 3 * PLANE;

/// jpeg_natural_order: zigzag index -> row-major index.
const ZIGZAG: [u8; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27,
    20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58,
    59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

type Result<T> = core::result::Result<T, Error>;

fn be16(d: &[u8], i: usize) -> Result<usize> {
    match d.get(i..i + 2) {
        Some(b) => Ok((b[0] as usize) << 8 | b[1] as usize),
        None => Err(Error::Format),
    }
}

// ------------------------------------------------------------------ Huffman

const LOOK: usize = 9;

#[derive(Clone)]
struct Huff {
    /// First LOOK bits -> (code length << 8) | symbol, 0 when the code is longer.
    look: [u16; 1 << LOOK],
    maxcode: [i32; 18],
    valoff: [i32; 17],
    vals: [u8; 256],
    ok: bool,
}

impl Huff {
    const EMPTY: Huff =
        Huff { look: [0; 1 << LOOK], maxcode: [-1; 18], valoff: [0; 17], vals: [0; 256], ok: false };

    /// jpeg_make_d_derived_tbl, including its validity checks.
    #[inline(never)]
    fn build(&mut self, bits: &[u8], vals: &[u8], dc: bool) -> Result<()> {
        if dc && vals.iter().any(|&v| v > 15) {
            return Err(Error::Format);
        }
        *self = Huff::EMPTY;
        self.vals[..vals.len()].copy_from_slice(vals);
        let (mut code, mut p) = (0i32, 0usize);
        for l in 1..=16 {
            let n = bits[l - 1] as usize;
            self.valoff[l] = p as i32 - code;
            for _ in 0..n {
                // No code may be all ones: every code of length l stays below 2^l - 1.
                if code + 1 >= 1 << l {
                    return Err(Error::Format);
                }
                if l <= LOOK {
                    let shift = LOOK - l;
                    let start = (code as usize) << shift;
                    let entry = (l as u16) << 8 | vals[p] as u16;
                    self.look[start..start + (1 << shift)].fill(entry);
                }
                p += 1;
                code += 1;
            }
            if n > 0 {
                self.maxcode[l] = code - 1;
            }
            code <<= 1;
        }
        self.maxcode[17] = i32::MAX;
        self.ok = true;
        Ok(())
    }
}

// ------------------------------------------------------------------ bit reader

struct Bits<'a> {
    d: &'a [u8],
    pos: usize,
    /// Next bits, MSB first; `n` of them are valid.
    acc: u64,
    n: u32,
    /// Hit a marker (or the end): feed zeros from now on, as libjpeg does.
    stop: bool,
}

impl<'a> Bits<'a> {
    /// Keep at least 16 bits buffered: the longest code, or the most extra bits.
    #[inline(always)]
    fn need16(&mut self) {
        if self.n < 16 {
            self.refill();
        }
    }

    #[inline(never)]
    fn refill(&mut self) {
        while self.n <= 56 {
            let mut b = 0u8;
            if !self.stop {
                match self.d.get(self.pos) {
                    Some(&0xFF) => {
                        // FF (FF..) 00 is a stuffed FF; FF followed by anything else is a marker.
                        let mut q = self.pos + 1;
                        while self.d.get(q) == Some(&0xFF) {
                            q += 1;
                        }
                        if self.d.get(q) == Some(&0) {
                            b = 0xFF;
                            self.pos = q + 1;
                        } else {
                            self.stop = true;
                        }
                    }
                    Some(&v) => {
                        b = v;
                        self.pos += 1;
                    }
                    None => self.stop = true,
                }
            }
            self.acc |= (b as u64) << (56 - self.n);
            self.n += 8;
        }
    }

    #[inline]
    fn take(&mut self, k: u32) -> i32 {
        // k <= 16 <= n (need16)
        let v = (self.acc >> (64 - k)) as i32;
        self.acc <<= k;
        self.n -= k;
        v
    }

    #[inline]
    fn symbol(&mut self, h: &Huff) -> Result<u8> {
        self.need16();
        let e = h.look[(self.acc >> (64 - LOOK)) as usize];
        if e != 0 {
            self.take((e >> 8) as u32);
            return Ok(e as u8);
        }
        let mut l = LOOK + 1;
        while l <= 16 {
            let code = (self.acc >> (64 - l)) as i32;
            if code <= h.maxcode[l] {
                self.take(l as u32);
                return Ok(h.vals[(h.valoff[l] + code) as usize & 0xFF]);
            }
            l += 1;
        }
        Err(Error::Format)
    }

    /// `s` more bits, sign-extended the JPEG way (HUFF_EXTEND).
    #[inline]
    fn extend(&mut self, s: u8) -> i32 {
        if s == 0 {
            return 0;
        }
        self.need16();
        let v = self.take(s as u32);
        if v < 1 << (s - 1) {
            v - (1 << s) + 1
        } else {
            v
        }
    }
}

/// Position just past the next marker code at or after `i`, and the code.
fn next_marker(d: &[u8], mut i: usize) -> Result<(u8, usize)> {
    loop {
        while i < d.len() && d[i] != 0xFF {
            i += 1;
        }
        while i < d.len() && d[i] == 0xFF {
            i += 1;
        }
        let m = *d.get(i).ok_or(Error::Format)?;
        i += 1;
        if m != 0 {
            return Ok((m, i));
        }
    }
}

// ------------------------------------------------------------------ IDCT (jidctint.c)

const CONST_BITS: u32 = 13;
const PASS1_BITS: u32 = 2;
const F0_298: i64 = 2446;
const F0_390: i64 = 3196;
const F0_541: i64 = 4433;
const F0_765: i64 = 6270;
const F0_899: i64 = 7373;
const F1_175: i64 = 9633;
const F1_501: i64 = 12299;
const F1_847: i64 = 15137;
const F1_961: i64 = 16069;
const F2_053: i64 = 16819;
const F2_562: i64 = 20995;
const F3_072: i64 = 25172;

#[inline]
fn descale(x: i64, n: u32) -> i64 {
    (x + (1 << (n - 1))) >> n
}

/// IDCT_range_limit[x & RANGE_MASK]: +128 and clamp, with libjpeg's wraparound.
#[inline]
fn range_limit(x: i64) -> u8 {
    let v = (x as i32) & 1023;
    if v < 128 {
        (v + 128) as u8
    } else if v < 512 {
        255
    } else if v < 896 {
        0
    } else {
        (v - 896) as u8
    }
}

/// One 1-D pass: even/odd parts of the LL&M IDCT on 8 dequantized inputs.
#[inline(always)]
fn idct_1d(x: [i64; 8]) -> [i64; 8] {
    let z1 = (x[2] + x[6]) * F0_541;
    let tmp2 = z1 - x[6] * F1_847;
    let tmp3 = z1 + x[2] * F0_765;
    let tmp0 = (x[0] + x[4]) << CONST_BITS;
    let tmp1 = (x[0] - x[4]) << CONST_BITS;
    let (tmp10, tmp13, tmp11, tmp12) = (tmp0 + tmp3, tmp0 - tmp3, tmp1 + tmp2, tmp1 - tmp2);

    let (t0, t1, t2, t3) = (x[7], x[5], x[3], x[1]);
    let (z1, z2, z3, z4) = (t0 + t3, t1 + t2, t0 + t2, t1 + t3);
    let z5 = (z3 + z4) * F1_175;
    let z1 = -z1 * F0_899;
    let z2 = -z2 * F2_562;
    let z3 = -z3 * F1_961 + z5;
    let z4 = -z4 * F0_390 + z5;
    let t0 = t0 * F0_298 + z1 + z3;
    let t1 = t1 * F2_053 + z2 + z4;
    let t2 = t2 * F3_072 + z2 + z3;
    let t3 = t3 * F1_501 + z1 + z4;
    [tmp10 + t3, tmp11 + t2, tmp12 + t1, tmp13 + t0, tmp13 - t0, tmp12 - t1, tmp11 - t2, tmp10 - t3]
}

/// jpeg_idct_islow: `coef` in natural order (not dequantized), 8x8 samples into `out`.
#[inline(never)]
fn idct(coef: &[i32; 64], q: &[u16; 64], out: &mut [u8], stride: usize) {
    let mut ws = [0i32; 64];
    for c in 0..8 {
        if (1..8).all(|r| coef[r * 8 + c] == 0) {
            let dc = ((coef[c] as i64 * q[c] as i64) << PASS1_BITS) as i32;
            for r in 0..8 {
                ws[r * 8 + c] = dc;
            }
            continue;
        }
        let x: [i64; 8] = core::array::from_fn(|r| coef[r * 8 + c] as i64 * q[r * 8 + c] as i64);
        for (r, v) in idct_1d(x).into_iter().enumerate() {
            ws[r * 8 + c] = descale(v, CONST_BITS - PASS1_BITS) as i32;
        }
    }
    for r in 0..8 {
        let w = &ws[r * 8..r * 8 + 8];
        let o = &mut out[r * stride..r * stride + 8];
        if w[1..].iter().all(|&v| v == 0) {
            o.fill(range_limit(descale(w[0] as i64, PASS1_BITS + 3)));
            continue;
        }
        let y = idct_1d(core::array::from_fn(|k| w[k] as i64));
        for k in 0..8 {
            o[k] = range_limit(descale(y[k], CONST_BITS + PASS1_BITS + 3));
        }
    }
}

// ------------------------------------------------------------------ decoder

#[derive(Clone, Copy)]
struct Comp {
    id: u8,
    h: usize,
    v: usize,
    tq: usize,
    td: usize,
    ta: usize,
    pred: i32,
    /// Quantization table latched at the component's first scan, like libjpeg.
    q: [u16; 64],
    seen: bool,
}

struct Decoder {
    qt: [[u16; 64]; 4],
    qt_ok: [bool; 4],
    dc: [Huff; 4],
    ac: [Huff; 4],
    comps: [Comp; 3],
    ncomp: usize,
    hmax: usize,
    vmax: usize,
    restart: usize,
    jfif: bool,
    adobe: Option<u8>,
}

/// Decode `data` into `gray` (250x80, row-major). `scratch` must hold [`SCRATCH`] bytes.
pub fn decode(data: &[u8], scratch: &mut [u8], gray: &mut [u8]) -> Result<()> {
    assert!(scratch.len() >= SCRATCH && gray.len() >= W * H);
    if data.len() < 4 || data[0] != 0xFF || data[1] != 0xD8 {
        return Err(Error::Format);
    }
    let comp0 =
        Comp { id: 0, h: 1, v: 1, tq: 0, td: 0, ta: 0, pred: 0, q: [0; 64], seen: false };
    let mut d = Decoder {
        qt: [[0; 64]; 4],
        qt_ok: [false; 4],
        dc: [Huff::EMPTY, Huff::EMPTY, Huff::EMPTY, Huff::EMPTY],
        ac: [Huff::EMPTY, Huff::EMPTY, Huff::EMPTY, Huff::EMPTY],
        comps: [comp0; 3],
        ncomp: 0,
        hmax: 1,
        vmax: 1,
        restart: 0,
        jfif: false,
        adobe: None,
    };
    let mut pos = 2;
    loop {
        let (m, p) = match next_marker(data, pos) {
            Ok(x) => x,
            // Missing EOI after complete scans: libjpeg only warns.
            Err(_) if d.complete() => break,
            Err(e) => return Err(e),
        };
        pos = p;
        match m {
            0xD9 => break,
            0xD8 => return Err(Error::Format),
            0x01 | 0xD0..=0xD7 => continue,
            _ => {}
        }
        let len = be16(data, pos)?;
        if len < 2 || pos + len > data.len() {
            return Err(Error::Format);
        }
        let seg = &data[pos + 2..pos + len];
        pos += len;
        match m {
            0xC0 | 0xC1 => d.sof(seg)?,
            0xC2 | 0xC3 | 0xC5..=0xC7 | 0xC9..=0xCB | 0xCD..=0xCF => return Err(Error::Unsupported),
            0xC4 => d.dht(seg)?,
            0xDB => d.dqt(seg)?,
            0xDD => d.restart = be16(seg, 0)?,
            0xDA => pos = d.sos(seg, data, pos, scratch)?,
            0xE0 if seg.starts_with(b"JFIF\0") => d.jfif = true,
            0xEE if seg.len() >= 12 && seg.starts_with(b"Adobe") => d.adobe = Some(seg[11]),
            _ => {}
        }
    }
    if !d.complete() {
        return Err(Error::Format);
    }
    d.to_gray(scratch, gray);
    Ok(())
}

impl Decoder {
    fn complete(&self) -> bool {
        self.ncomp > 0 && self.comps[..self.ncomp].iter().all(|c| c.seen)
    }

    #[inline(never)]
    fn sof(&mut self, s: &[u8]) -> Result<()> {
        if self.ncomp != 0 || s.len() < 6 {
            return Err(Error::Format);
        }
        if s[0] != 8 {
            return Err(Error::Unsupported);
        }
        let (h, w, n) = (be16(s, 1)?, be16(s, 3)?, s[5] as usize);
        if s.len() < 6 + 3 * n || n == 0 {
            return Err(Error::Format);
        }
        if n != 1 && n != 3 {
            return Err(Error::Unsupported);
        }
        if w != W || h != H {
            return Err(Error::Size);
        }
        for i in 0..n {
            let c = &mut self.comps[i];
            let (hv, tq) = (s[7 + 3 * i], s[8 + 3 * i] as usize);
            c.id = s[6 + 3 * i];
            c.h = (hv >> 4) as usize;
            c.v = (hv & 15) as usize;
            if c.h == 0 || c.v == 0 || tq > 3 {
                return Err(Error::Format);
            }
            if c.h > 2 || c.v > 2 {
                return Err(Error::Unsupported);
            }
            c.tq = tq;
        }
        self.ncomp = n;
        self.hmax = self.comps[..n].iter().map(|c| c.h).max().unwrap_or(1);
        self.vmax = self.comps[..n].iter().map(|c| c.v).max().unwrap_or(1);
        for c in &self.comps[..n] {
            match (self.hmax / c.h, self.vmax / c.v) {
                (1, 1) | (2, 1) | (2, 2) => {}
                _ => return Err(Error::Unsupported),
            }
        }
        Ok(())
    }

    #[inline(never)]
    fn dqt(&mut self, mut s: &[u8]) -> Result<()> {
        while let Some(&pt) = s.first() {
            let (p, t) = ((pt >> 4) as usize, (pt & 15) as usize);
            let n = 64 << p;
            if p > 1 || t > 3 || s.len() < 1 + n {
                return Err(Error::Format);
            }
            for k in 0..64 {
                let v = if p == 0 { s[1 + k] as u16 } else { be16(s, 1 + 2 * k)? as u16 };
                self.qt[t][ZIGZAG[k] as usize] = v;
            }
            self.qt_ok[t] = true;
            s = &s[1 + n..];
        }
        Ok(())
    }

    #[inline(never)]
    fn dht(&mut self, mut s: &[u8]) -> Result<()> {
        while let Some(&ct) = s.first() {
            let (class, t) = ((ct >> 4) as usize, (ct & 15) as usize);
            if class > 1 || t > 3 || s.len() < 17 {
                return Err(Error::Format);
            }
            let count: usize = s[1..17].iter().map(|&b| b as usize).sum();
            if count > 256 || s.len() < 17 + count {
                return Err(Error::Format);
            }
            let table = if class == 0 { &mut self.dc[t] } else { &mut self.ac[t] };
            table.build(&s[1..17], &s[17..17 + count], class == 0)?;
            s = &s[17 + count..];
        }
        Ok(())
    }

    /// Decode one scan; returns the position where the entropy-coded data ended.
    fn sos(&mut self, s: &[u8], data: &[u8], pos: usize, planes: &mut [u8]) -> Result<usize> {
        let ns = *s.first().ok_or(Error::Format)? as usize;
        if self.ncomp == 0 || ns == 0 || ns > self.ncomp || s.len() < 4 + 2 * ns {
            return Err(Error::Format);
        }
        let mut idx = [0usize; 3];
        for i in 0..ns {
            let ci = self.comps[..self.ncomp]
                .iter()
                .position(|c| c.id == s[1 + 2 * i])
                .ok_or(Error::Format)?;
            let c = &mut self.comps[ci];
            c.td = (s[2 + 2 * i] >> 4) as usize;
            c.ta = (s[2 + 2 * i] & 15) as usize;
            if c.td > 3 || c.ta > 3 || !self.dc[c.td].ok || !self.ac[c.ta].ok || !self.qt_ok[c.tq] {
                return Err(Error::Format);
            }
            if !c.seen {
                c.q = self.qt[c.tq];
                c.seen = true;
            }
            c.pred = 0;
            idx[i] = ci;
        }

        // Non-interleaved scans walk the component's own block grid, one block per MCU.
        let (mcux, mcuy) = if ns == 1 {
            let c = &self.comps[idx[0]];
            let cw = (W * c.h).div_ceil(self.hmax);
            let ch = (H * c.v).div_ceil(self.vmax);
            (cw.div_ceil(8), ch.div_ceil(8))
        } else {
            (W.div_ceil(8 * self.hmax), H.div_ceil(8 * self.vmax))
        };

        let mut bits = Bits { d: data, pos, acc: 0, n: 0, stop: false };
        let mut coef = [0i32; 64];
        let mut todo = self.restart;
        for m in 0..mcux * mcuy {
            if self.restart > 0 {
                if todo == 0 {
                    // Drop the byte-alignment padding and step over RSTn.
                    let (mk, p) = next_marker(data, bits.pos)?;
                    if !(0xD0..=0xD7).contains(&mk) {
                        return Err(Error::Format);
                    }
                    bits = Bits { d: data, pos: p, acc: 0, n: 0, stop: false };
                    for &ci in &idx[..ns] {
                        self.comps[ci].pred = 0;
                    }
                    todo = self.restart;
                }
                todo -= 1;
            }
            let (mx, my) = (m % mcux, m / mcux);
            for &ci in &idx[..ns] {
                let c = &mut self.comps[ci];
                let (bw, bh) = if ns == 1 { (1, 1) } else { (c.h, c.v) };
                for v in 0..bh {
                    for h in 0..bw {
                        let ac = block(&mut bits, &self.dc[c.td], &self.ac[c.ta], &mut c.pred, &mut coef)?;
                        let (bx, by) = (mx * bw + h, my * bh + v);
                        let out = &mut planes[ci * PLANE + by * 8 * PLANE_W + bx * 8..];
                        if ac {
                            idct(&coef, &c.q, out, PLANE_W);
                        } else {
                            // What idct() computes for a DC-only block (over half of
                            // a captcha): its zero-column and zero-row shortcuts.
                            let dc = ((coef[0] as i64 * c.q[0] as i64) << PASS1_BITS) as i32;
                            let v = range_limit(descale(dc as i64, PASS1_BITS + 3));
                            for r in 0..8 {
                                out[r * PLANE_W..r * PLANE_W + 8].fill(v);
                            }
                        }
                    }
                }
            }
        }
        Ok(bits.pos)
    }

    /// Upsample to full resolution, convert to RGB and then to Pillow's "L".
    fn to_gray(&self, planes: &[u8], gray: &mut [u8]) {
        if self.ncomp == 1 {
            for (y, out) in gray.chunks_exact_mut(W).take(H).enumerate() {
                out.copy_from_slice(&planes[y * PLANE_W..y * PLANE_W + W]);
            }
            return;
        }
        let rgb = match self.adobe {
            _ if self.jfif => false,
            Some(t) => t == 0,
            None => self.comps[..3].iter().map(|c| c.id).eq([b'R', b'G', b'B']),
        };
        let up: [Up; 3] = core::array::from_fn(|k| {
            let c = &self.comps[k];
            let (dw, dh) = ((W * c.h).div_ceil(self.hmax), (H * c.v).div_ceil(self.vmax));
            Up { h2: self.hmax > c.h, v2: self.vmax > c.v, dw, dh }
        });
        #[cfg(target_arch = "x86_64")]
        if crate::kernel::has_avx2() {
            // SAFETY: the CPU supports AVX2.
            let up_row = |u: &Up, p: &[u8], y, o: &mut _| unsafe { avx2::upsample(u, p, y, o) };
            let conv = |r: &_, rgb, o: &mut _| unsafe { avx2::convert(r, rgb, o) };
            return rows_to_gray(&up, planes, rgb, gray, up_row, conv);
        }
        rows_to_gray(&up, planes, rgb, gray, upsample, convert)
    }
}

/// How one component reaches full resolution: `h2` = 2x wide, `v2` = 2x tall.
struct Up {
    h2: bool,
    v2: bool,
    /// Samples per row / rows in the component (jpeg downsampled_width/height).
    dw: usize,
    dh: usize,
}

type Rows = [[u8; 2 * PLANE_W]; 3];

#[inline(always)]
fn rows_to_gray(
    up: &[Up; 3],
    planes: &[u8],
    rgb: bool,
    gray: &mut [u8],
    up_row: impl Fn(&Up, &[u8], usize, &mut [u8; 2 * PLANE_W]),
    conv: impl Fn(&Rows, bool, &mut [u8]),
) {
    let mut rows = [[0u8; 2 * PLANE_W]; 3];
    for (y, out) in gray.chunks_exact_mut(W).take(H).enumerate() {
        for (k, row) in rows.iter_mut().enumerate() {
            up_row(&up[k], &planes[k * PLANE..(k + 1) * PLANE], y, row);
        }
        conv(&rows, rgb, out);
    }
}

#[inline(never)]
fn upsample(u: &Up, plane: &[u8], y: usize, out: &mut [u8; 2 * PLANE_W]) {
    upsample_impl(u, plane, y, out)
}

#[inline(never)]
fn convert(rows: &Rows, rgb: bool, out: &mut [u8]) {
    convert_impl(rows, rgb, out)
}

/// The same two functions compiled with AVX2: their integer loops vectorize 8-16 wide.
#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::{Rows, Up, PLANE_W};

    #[target_feature(enable = "avx2")]
    pub unsafe fn upsample(u: &Up, plane: &[u8], y: usize, out: &mut [u8; 2 * PLANE_W]) {
        super::upsample_impl(u, plane, y, out)
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn convert(rows: &Rows, rgb: bool, out: &mut [u8]) {
        super::convert_impl(rows, rgb, out)
    }
}

/// One output row: RGB -> Pillow's "L", from YCbCr per jdcolor.c unless `rgb`.
#[inline(always)]
fn convert_impl(rows: &Rows, rgb: bool, out: &mut [u8]) {
    let [c0, c1, c2] = rows;
    let px = out.iter_mut().zip(&c0[..W]).zip(&c1[..W]).zip(&c2[..W]);
    if rgb {
        for (((o, &r), &g), &b) in px {
            *o = luma(r as i32, g as i32, b as i32);
        }
    } else {
        for (((o, &y), &cb), &cr) in px {
            // build_ycc_rgb_table / ycc_rgb_convert, SCALEBITS = 16
            let (y, cb, cr) = (y as i32, cb as i32 - 128, cr as i32 - 128);
            let r = y + ((91881 * cr + 32768) >> 16);
            let g = y + ((-22554 * cb + 32768 - 46802 * cr) >> 16);
            let b = y + ((116130 * cb + 32768) >> 16);
            *o = luma(r.clamp(0, 255), g.clamp(0, 255), b.clamp(0, 255));
        }
    }
}

/// Full-resolution row `y` of one component (jdsample.c fancy upsampling),
/// written as independent per-pixel expressions so it vectorizes.
#[inline(always)]
#[allow(clippy::needless_range_loop)] // x-1 / x / x+1 stencils read best as indices
fn upsample_impl(u: &Up, plane: &[u8], y: usize, out: &mut [u8; 2 * PLANE_W]) {
    let dw = u.dw;
    let row = |r: usize| &plane[r * PLANE_W..r * PLANE_W + dw];
    let mut pairs = [0u16; PLANE_W];
    match (u.h2, u.v2) {
        (false, _) => return out[..W].copy_from_slice(&row(y)[..W]),
        (true, false) => {
            // h2v1_fancy_upsample: 3/4 nearer + 1/4 further sample
            let i = row(y);
            let px = |x: usize| i[x] as u16;
            let pair = |x: usize, l: u16, r: u16| ((px(x) * 3 + l + 1) >> 2) | ((px(x) * 3 + r + 2) >> 2) << 8;
            for x in 1..dw - 1 {
                pairs[x] = pair(x, px(x - 1), px(x + 1));
            }
            // Edge outputs are the edge sample itself: (4i + 1) >> 2 == (4i + 2) >> 2 == i.
            pairs[0] = pair(0, px(0), px(1));
            pairs[dw - 1] = pair(dw - 1, px(dw - 2), px(dw - 1));
        }
        (true, true) => {
            // h2v2_fancy_upsample: the same in both directions (9/16, 3/16, 3/16, 1/16);
            // the context row past either edge repeats the edge row.
            let cy = y / 2;
            let other = if y % 2 == 0 { cy.saturating_sub(1) } else { (cy + 1).min(u.dh - 1) };
            let (i0, i1) = (row(cy), row(other));
            let mut cs = [0u16; PLANE_W];
            let cs = &mut cs[..dw];
            for ((c, &a), &b) in cs.iter_mut().zip(i0).zip(i1) {
                *c = a as u16 * 3 + b as u16;
            }
            let pair = |x: usize, l: u16, r: u16| ((cs[x] * 3 + l + 8) >> 4) | ((cs[x] * 3 + r + 7) >> 4) << 8;
            for x in 1..dw - 1 {
                pairs[x] = pair(x, cs[x - 1], cs[x + 1]);
            }
            pairs[0] = pair(0, cs[0], cs[1]); // edge columns count double: 4/16 of themselves
            pairs[dw - 1] = pair(dw - 1, cs[dw - 2], cs[dw - 1]);
        }
    }
    interleave(&pairs[..dw], &mut out[..2 * dw]);
}

/// Unpack (even | odd << 8) pairs into consecutive output bytes.
#[inline(always)]
fn interleave(pairs: &[u16], out: &mut [u8]) {
    for (o, p) in out.chunks_exact_mut(2).zip(pairs) {
        o.copy_from_slice(&p.to_le_bytes());
    }
}

/// Pillow Convert.c rgb2l: L24(rgb) >> 16.
#[inline]
fn luma(r: i32, g: i32, b: i32) -> u8 {
    ((r * 19595 + g * 38470 + b * 7471 + 0x8000) >> 16) as u8
}

/// decode_mcu for one block: DC difference + AC run-lengths, natural order.
/// Returns whether any AC coefficient is nonzero.
#[inline]
fn block(bits: &mut Bits, dc: &Huff, ac: &Huff, pred: &mut i32, coef: &mut [i32; 64]) -> Result<bool> {
    *coef = [0; 64];
    let mut any = false;
    let t = bits.symbol(dc)?;
    *pred = pred.wrapping_add(bits.extend(t));
    coef[0] = *pred as i16 as i32; // JCOEF
    let mut k = 1;
    while k < 64 {
        let rs = bits.symbol(ac)?;
        let (run, s) = ((rs >> 4) as usize, rs & 15);
        if s == 0 {
            if run != 15 {
                break;
            }
            k += 16;
            continue;
        }
        k += run;
        if k > 63 {
            return Err(Error::Format);
        }
        let v = bits.extend(s) as i16 as i32;
        coef[ZIGZAG[k] as usize] = v;
        any |= v != 0;
        k += 1;
    }
    Ok(any)
}

