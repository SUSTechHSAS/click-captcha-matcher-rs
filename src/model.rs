//! The two-tower matcher (model.py) on HWC activations, and its .ccm weight file.
//!
//! ```text
//! prompt 16x16 -> conv w, conv w, pool, conv 2w, conv 2w, pool, flatten, fc dim -> L2 norm
//! cand   40x40 -> conv w /2, conv w, pool, conv 2w, conv 2w, pool, conv 4w, conv 4w, mean, fc dim -> L2 norm
//! ```
//!
//! Every conv is 3x3 + folded BatchNorm + ReLU. Weights are dequantized once at
//! load time; inference is float32 throughout.

use crate::geometry::{CAND_PIXELS, CAND_SIZE, PROMPT_PIXELS, PROMPT_SIZE};
use crate::kernel::{self, Gemm};
use alloc::vec;
use alloc::vec::Vec;
use core::ops::{Deref, DerefMut};

/// The weight file is malformed or describes an unsupported shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModelError;

const P_CONV: [usize; 4] = [0, 1, 2, 3];
const P_FC: usize = 4;
const C_CONV: [usize; 6] = [5, 6, 7, 8, 9, 10];
const C_FC: usize = 11;

/// Most output pixels of any conv (the 20x20 map after the stride-2 conv), and
/// the size of the zero-padded 40x40 candidate input.
const MAX_PIXELS: usize = (CAND_SIZE / 2) * (CAND_SIZE / 2);
const PAD_IN: usize = (CAND_SIZE + 2) * (CAND_SIZE + 2);

#[derive(Clone, Copy, Default)]
struct Layer {
    cin: usize,
    cout: usize,
    /// Inputs per output channel: 9*cin for a conv, fan_in for an fc.
    k: usize,
    /// Offsets of the packed weights and of the bias in `Model::buf`.
    w: usize,
    b: usize,
}

pub struct Model {
    pub width: usize,
    pub dim: usize,
    layers: [Layer; 12],
    buf: Aligned,
}

/// Zeroed f32 buffer starting on a 64-byte boundary (malloc promises only 16),
/// so the kernel's 32-byte loads and stores never straddle cache lines.
struct Aligned {
    v: Vec<f32>,
    off: usize,
    len: usize,
}

impl Aligned {
    fn new(len: usize) -> Aligned {
        let v = vec![0f32; len + 15];
        let off = v.as_ptr().align_offset(64);
        Aligned { v, off, len }
    }
}

impl Deref for Aligned {
    type Target = [f32];
    fn deref(&self) -> &[f32] {
        &self.v[self.off..self.off + self.len]
    }
}

impl DerefMut for Aligned {
    fn deref_mut(&mut self) -> &mut [f32] {
        &mut self.v[self.off..self.off + self.len]
    }
}

/// Activation buffers for one forward pass.
pub struct Scratch {
    pad: Vec<f32>,
    a: Aligned,
    b: Aligned,
    feat_p: Vec<f32>,
    feat_c: Vec<f32>,
    emb: Vec<f32>,
    ia: [usize; MAX_PIXELS],
    io: [usize; MAX_PIXELS],
}

impl Scratch {
    pub fn new(m: &Model) -> Scratch {
        let act = (CAND_SIZE / 2 + 2) * (CAND_SIZE / 2 + 2) * m.width; // 22x22 x w is the largest map
        Scratch {
            pad: vec![0.0; PAD_IN],
            a: Aligned::new(act),
            b: Aligned::new(act),
            feat_p: vec![0.0; 4 * 32 * m.width],
            feat_c: vec![0.0; 6 * 4 * m.width],
            emb: vec![0.0; 10 * m.dim],
            ia: [0; MAX_PIXELS],
            io: [0; MAX_PIXELS],
        }
    }
}

struct Reader<'a> {
    b: &'a [u8],
    i: usize,
}

impl Reader<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8], ModelError> {
        let s = self.b.get(self.i..self.i + n).ok_or(ModelError)?;
        self.i += n;
        Ok(s)
    }

    fn f32s(&mut self, n: usize) -> Result<impl Iterator<Item = f32> + '_, ModelError> {
        Ok(self.take(4 * n)?.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])))
    }
}

/// IEEE half -> float for the finite values the exporter writes (normals and subnormals).
fn f16_to_f32(h: u16) -> f32 {
    let v = f32::from_bits(((h & 0x7fff) as u32) << 13) * f32::from_bits(0x7780_0000); // * 2^112
    if h & 0x8000 != 0 {
        -v
    } else {
        v
    }
}

impl Model {
    /// Parse a .ccm file (see tools/export_ccm.py for the layout).
    pub fn load(bytes: &[u8]) -> Result<Model, ModelError> {
        let hdr = bytes.get(..8).ok_or(ModelError)?;
        let (width, dtype, dim) = (hdr[4] as usize, hdr[5], u16::from_le_bytes([hdr[6], hdr[7]]) as usize);
        if &hdr[..4] != b"CCM1"
            || dtype > 2
            || width == 0
            || width % 16 != 0
            || width > 64
            || dim == 0
            || dim % 16 != 0
            || dim > 512
        {
            return Err(ModelError);
        }
        let w = width;
        // (cin, cout, is_conv); the prompt fc sees 2w channels x 4x4
        let spec = [
            (1, w, true),
            (w, w, true),
            (w, 2 * w, true),
            (2 * w, 2 * w, true),
            (32 * w, dim, false),
            (1, w, true),
            (w, w, true),
            (w, 2 * w, true),
            (2 * w, 2 * w, true),
            (2 * w, 4 * w, true),
            (4 * w, 4 * w, true),
            (4 * w, dim, false),
        ];
        let mut layers = [Layer::default(); 12];
        let mut total = 0;
        for (l, &(cin, cout, conv)) in layers.iter_mut().zip(&spec) {
            let k = if conv { 9 * cin } else { cin };
            *l = Layer { cin, cout, k, w: total, b: total + cout * k };
            total += cout * k + cout;
        }

        let mut buf = Aligned::new(total);
        let mut rd = Reader { b: bytes, i: 8 };
        let mut scale = [0f32; 512];
        for (li, l) in layers.iter().enumerate() {
            let bits = if dtype == 2 { rd.take(1)?[0] as u32 } else { 0 };
            if dtype == 2 {
                if !(2..=8).contains(&bits) {
                    return Err(ModelError);
                }
                for (s, v) in scale.iter_mut().zip(rd.f32s(l.cout)?) {
                    *s = v;
                }
            }
            for (d, v) in buf[l.b..l.b + l.cout].iter_mut().zip(rd.f32s(l.cout)?) {
                *d = v;
            }
            let n = l.cout * l.k;
            let raw = rd.take(match dtype {
                0 => 4 * n,
                1 => 2 * n,
                _ => (n * bits as usize).div_ceil(8),
            })?;
            let (mut acc, mut have, mut pos) = (0u32, 0u32, 0usize);
            for co in 0..l.cout {
                for e in 0..l.k {
                    let i = co * l.k + e;
                    let v = match dtype {
                        0 => f32::from_le_bytes([raw[4 * i], raw[4 * i + 1], raw[4 * i + 2], raw[4 * i + 3]]),
                        1 => f16_to_f32(u16::from_le_bytes([raw[2 * i], raw[2 * i + 1]])),
                        _ => {
                            // next `bits`-bit two's complement code from the LSB-first stream
                            while have < bits {
                                acc |= (raw[pos] as u32) << have;
                                pos += 1;
                                have += 8;
                            }
                            let q = ((acc << (32 - bits)) as i32) >> (32 - bits);
                            acc >>= bits;
                            have -= bits;
                            q as f32 * scale[co]
                        }
                    };
                    // PyTorch order -> the kernel's input order within one output channel
                    let kidx = if li == P_FC {
                        let c = 2 * w; // flatten of (c, y, x) -> HWC (y, x, c)
                        (e % 16) * c + e / 16
                    } else if li == C_FC {
                        e
                    } else {
                        (e % 9) * l.cin + e / 9 // (ci, ky, kx) -> (ky, kx, ci)
                    };
                    buf[l.w + ((co / 16) * l.k + kidx) * 16 + co % 16] = v;
                }
            }
        }
        if rd.i != bytes.len() {
            return Err(ModelError);
        }
        Ok(Model { width, dim, layers, buf })
    }

    /// 3x3 conv + ReLU on a zero-padded (n+2)^2 x cin map -> zero-padded (n/stride+2)^2 x cout.
    #[inline(never)]
    fn conv(&self, li: usize, inp: &[f32], n: usize, stride: usize, out: &mut [f32], s: &mut Ixs) {
        let l = &self.layers[li];
        let m = n / stride;
        zero_border(out, m, l.cout);
        let mut q = 0;
        for y in 0..m {
            for x in 0..m {
                s.0[q] = (y * stride * (n + 2) + x * stride) * l.cin;
                s.1[q] = ((y + 1) * (m + 2) + x + 1) * l.cout;
                q += 1;
            }
        }
        let g = Gemm {
            inp,
            a: &s.0[..q],
            o: &s.1[..q],
            rows: 3,
            stride: (n + 2) * l.cin,
            len: 3 * l.cin,
            w: &self.buf[l.w..l.b],
            b: &self.buf[l.b..l.b + l.cout],
            cout: l.cout,
            relu: true,
        };
        kernel::run(&g, out);
    }

    /// Fully connected layer over `n` rows of `inp`.
    #[inline(never)]
    fn fc(&self, li: usize, inp: &[f32], n: usize, out: &mut [f32], s: &mut Ixs) {
        let l = &self.layers[li];
        for i in 0..n {
            s.0[i] = i * l.k;
            s.1[i] = i * l.cout;
        }
        let g = Gemm {
            inp,
            a: &s.0[..n],
            o: &s.1[..n],
            rows: 1,
            stride: 0,
            len: l.k,
            w: &self.buf[l.w..l.b],
            b: &self.buf[l.b..l.b + l.cout],
            cout: l.cout,
            relu: false,
        };
        kernel::run(&g, out);
    }

    /// (4, 16x16) prompt crops and (6, 40x40) candidate crops -> (4, 6) cosine similarity.
    pub fn similarity(&self, prompts: &[f32], cands: &[f32], s: &mut Scratch) -> [[f32; 6]; 4] {
        let (w, dim) = (self.width, self.dim);
        let Scratch { pad, a, b, feat_p, feat_c, emb, ia, io } = s;
        let ix = &mut Ixs(ia, io);

        for (k, crop) in prompts.chunks_exact(PROMPT_PIXELS).take(4).enumerate() {
            pad_crop(crop, PROMPT_SIZE, pad);
            self.conv(P_CONV[0], pad, 16, 1, a, ix);
            self.conv(P_CONV[1], a, 16, 1, b, ix);
            maxpool(b, 16, w, a, true);
            self.conv(P_CONV[2], a, 8, 1, b, ix);
            self.conv(P_CONV[3], b, 8, 1, a, ix);
            maxpool(a, 8, 2 * w, &mut feat_p[k * 32 * w..(k + 1) * 32 * w], false);
        }
        for (k, crop) in cands.chunks_exact(CAND_PIXELS).take(6).enumerate() {
            pad_crop(crop, CAND_SIZE, pad);
            self.conv(C_CONV[0], pad, 40, 2, a, ix);
            self.conv(C_CONV[1], a, 20, 1, b, ix);
            maxpool(b, 20, w, a, true);
            self.conv(C_CONV[2], a, 10, 1, b, ix);
            self.conv(C_CONV[3], b, 10, 1, a, ix);
            maxpool(a, 10, 2 * w, b, true);
            self.conv(C_CONV[4], b, 5, 1, a, ix);
            self.conv(C_CONV[5], a, 5, 1, b, ix);
            mean_pool(b, 5, 4 * w, &mut feat_c[k * 4 * w..(k + 1) * 4 * w]);
        }
        let (ep, ec) = emb.split_at_mut(4 * dim);
        self.fc(P_FC, feat_p, 4, ep, ix);
        self.fc(C_FC, feat_c, 6, ec, ix);
        for e in emb.chunks_exact_mut(dim) {
            normalize(e);
        }

        let mut sim = [[0f32; 6]; 4];
        for (i, p) in emb[..4 * dim].chunks_exact(dim).enumerate() {
            for (j, c) in emb[4 * dim..].chunks_exact(dim).enumerate() {
                sim[i][j] = p.iter().zip(c).map(|(x, y)| x * y).sum();
            }
        }
        sim
    }
}

/// Per-pixel input / output offsets handed to the kernel.
struct Ixs<'a>(&'a mut [usize; MAX_PIXELS], &'a mut [usize; MAX_PIXELS]);

#[inline(never)]
fn zero_border(buf: &mut [f32], m: usize, c: usize) {
    let row = (m + 2) * c;
    buf[..row].fill(0.0);
    buf[(m + 1) * row..(m + 2) * row].fill(0.0);
    for y in 1..=m {
        buf[y * row..y * row + c].fill(0.0);
        buf[(y + 1) * row - c..(y + 1) * row].fill(0.0);
    }
}

fn pad_crop(crop: &[f32], n: usize, out: &mut [f32]) {
    out[..(n + 2) * (n + 2)].fill(0.0);
    for (y, row) in crop.chunks_exact(n).enumerate() {
        out[(y + 1) * (n + 2) + 1..][..n].copy_from_slice(row);
    }
}

/// 2x2 max pool of a padded n x n x c map into an m x m map, zero-padded or packed.
#[inline(never)]
fn maxpool(inp: &[f32], n: usize, c: usize, out: &mut [f32], padded: bool) {
    let m = n / 2;
    let (po, off) = if padded { (m + 2, 1) } else { (m, 0) };
    if padded {
        zero_border(out, m, c);
    }
    for y in 0..m {
        for x in 0..m {
            let i = ((2 * y + 1) * (n + 2) + 2 * x + 1) * c;
            let (r0, r1) = (&inp[i..i + 2 * c], &inp[i + (n + 2) * c..i + (n + 2) * c + 2 * c]);
            let o = &mut out[((y + off) * po + x + off) * c..][..c];
            for k in 0..c {
                o[k] = r0[k].max(r0[c + k]).max(r1[k].max(r1[c + k]));
            }
        }
    }
}

/// Global average pool of a padded n x n x c map.
#[inline(never)]
fn mean_pool(inp: &[f32], n: usize, c: usize, out: &mut [f32]) {
    out.fill(0.0);
    for y in 0..n {
        for x in 0..n {
            let px = &inp[((y + 1) * (n + 2) + x + 1) * c..][..c];
            for (o, v) in out.iter_mut().zip(px) {
                *o += v;
            }
        }
    }
    for o in out {
        *o /= (n * n) as f32;
    }
}

/// F.normalize: x / max(||x||, 1e-12).
fn normalize(v: &mut [f32]) {
    let norm = sqrt(v.iter().map(|x| x * x).sum()).max(1e-12);
    for x in v {
        *x /= norm;
    }
}

/// Correctly rounded sqrt (core has none): Newton in f64, rounded once to f32.
fn sqrt(x: f32) -> f32 {
    if x <= 0.0 || x.is_nan() {
        return 0.0;
    }
    let x = x as f64;
    let mut y = f64::from_bits((x.to_bits() >> 1) + (1023u64 << 51));
    for _ in 0..6 {
        y = 0.5 * (y + x / y);
    }
    y as f32
}
