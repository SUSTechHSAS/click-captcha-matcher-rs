//! The one compute kernel behind every conv and FC layer:
//!
//! ```text
//! out[o[p] + co] = act(b[co] + sum_{r<rows} sum_{j<len} inp[a[p] + r*stride + j] * w[co/16][r*len + j][co%16])
//! ```
//!
//! A 3x3 conv over zero-padded HWC activations is 3 rows of 3*cin contiguous
//! samples per output pixel; an FC layer is 1 row of fan_in. Output channels are
//! processed 16 at a time, pixels 4-6 at a time. x86_64 picks an AVX2+FMA version
//! at runtime and aarch64 always has NEON; anything else uses the portable version.

use core::sync::atomic::{AtomicU8, Ordering};

pub struct Gemm<'a> {
    pub inp: &'a [f32],
    /// Per output pixel: offset of its first input row, and of its output channels.
    pub a: &'a [usize],
    pub o: &'a [usize],
    pub rows: usize,
    pub stride: usize,
    pub len: usize,
    /// Weights packed as [cout/16][rows*len][16].
    pub w: &'a [f32],
    pub b: &'a [f32],
    pub cout: usize,
    pub relu: bool,
}

const UNKNOWN: u8 = 0;
const GENERIC: u8 = 1;
const AVX2: u8 = 2;
const NEON: u8 = 3;
static ISA: AtomicU8 = AtomicU8::new(UNKNOWN);

fn isa() -> u8 {
    match ISA.load(Ordering::Relaxed) {
        UNKNOWN => {
            let v = if cfg!(target_arch = "aarch64") {
                NEON
            } else if avx2_fma() {
                AVX2
            } else {
                GENERIC
            };
            ISA.store(v, Ordering::Relaxed);
            v
        }
        v => v,
    }
}

/// Name of the kernel in use: "avx2", "neon" or "generic".
pub fn isa_name() -> &'static str {
    match isa() {
        AVX2 => "avx2",
        NEON => "neon",
        _ => "generic",
    }
}

/// AVX2 + FMA available (and not overridden by `force_generic`).
#[cfg(target_arch = "x86_64")]
pub(crate) fn has_avx2() -> bool {
    isa() == AVX2
}

/// Use the portable kernel from now on (for testing and benchmarking).
pub fn force_generic() {
    ISA.store(GENERIC, Ordering::Relaxed);
}

#[cfg(target_arch = "x86_64")]
#[allow(unused_unsafe)]
fn avx2_fma() -> bool {
    use core::arch::x86_64::{__cpuid, __cpuid_count, _xgetbv};
    #[target_feature(enable = "xsave")]
    unsafe fn xcr0() -> u64 {
        _xgetbv(0)
    }
    // SAFETY: CPUID exists on every x86_64 CPU; XGETBV is only used once OSXSAVE says it exists.
    unsafe {
        let l1 = __cpuid(1);
        let (fma, osxsave, avx) = (l1.ecx >> 12 & 1, l1.ecx >> 27 & 1, l1.ecx >> 28 & 1);
        fma == 1
            && osxsave == 1
            && avx == 1
            && xcr0() & 6 == 6 // the OS saves XMM and YMM state
            && __cpuid(0).eax >= 7
            && __cpuid_count(7, 0).ebx >> 5 & 1 == 1
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn avx2_fma() -> bool {
    false
}

pub fn run(g: &Gemm, out: &mut [f32]) {
    // Everything the unchecked kernels touch is validated here, once per layer.
    assert!(g.cout % 16 == 0 && g.rows > 0 && g.len > 0 && g.a.len() == g.o.len());
    assert!(g.w.len() >= g.cout * g.rows * g.len && g.b.len() >= g.cout);
    let span = (g.rows - 1) * g.stride + g.len;
    for (&a, &o) in g.a.iter().zip(g.o) {
        assert!(a + span <= g.inp.len() && o + g.cout <= out.len());
    }
    if g.a.is_empty() {
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if isa() == AVX2 {
        // SAFETY: bounds checked above, and the CPU has AVX2 + FMA.
        return unsafe { avx2::run(g, out) };
    }
    #[cfg(target_arch = "aarch64")]
    if isa() == NEON {
        // SAFETY: bounds checked above; NEON is part of aarch64.
        return unsafe { neon::run(g, out) };
    }
    generic(g, out)
}

fn generic(g: &Gemm, out: &mut [f32]) {
    let (n, k) = (g.a.len(), g.rows * g.len);
    for cb in 0..g.cout / 16 {
        let wb = &g.w[cb * k * 16..(cb + 1) * k * 16];
        let bias: [f32; 16] = g.b[cb * 16..cb * 16 + 16].try_into().unwrap();
        for p in (0..n).step_by(4) {
            let a: [usize; 4] = core::array::from_fn(|i| g.a[(p + i).min(n - 1)]);
            let mut acc = [bias; 4];
            for r in 0..g.rows {
                let x: [&[f32]; 4] = core::array::from_fn(|i| &g.inp[a[i] + r * g.stride..][..g.len]);
                let wr = &wb[r * g.len * 16..(r + 1) * g.len * 16];
                for ((((wv, &x0), &x1), &x2), &x3) in
                    wr.chunks_exact(16).zip(x[0]).zip(x[1]).zip(x[2]).zip(x[3])
                {
                    for (acc, xv) in acc.iter_mut().zip([x0, x1, x2, x3]) {
                        for c in 0..16 {
                            acc[c] += xv * wv[c];
                        }
                    }
                }
            }
            for (i, acc) in acc.iter().enumerate().take(n - p) {
                let dst = &mut out[g.o[p + i] + cb * 16..][..16];
                for c in 0..16 {
                    dst[c] = if g.relu { acc[c].max(0.0) } else { acc[c] };
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::Gemm;
    use core::arch::x86_64::*;

    /// Pixels per block: 6 x 16 channels = 12 accumulators, enough independent
    /// FMA chains to cover the FMA latency with 15 of the 16 YMM registers.
    const P: usize = 6;

    /// SAFETY: caller checks bounds (see `run`) and AVX2 + FMA support.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn run(g: &Gemm, out: &mut [f32]) {
        let (n, k) = (g.a.len(), g.rows * g.len);
        let (ip, op) = (g.inp.as_ptr(), out.as_mut_ptr());
        let zero = _mm256_setzero_ps();
        for cb in 0..g.cout / 16 {
            let wb = g.w.as_ptr().add(cb * k * 16);
            let bp = g.b.as_ptr().add(cb * 16);
            let (b0, b1) = (_mm256_loadu_ps(bp), _mm256_loadu_ps(bp.add(8)));
            let mut p = 0;
            while p < n {
                let a: [usize; P] = core::array::from_fn(|i| g.a[(p + i).min(n - 1)]);
                let mut acc = [b0, b1, b0, b1, b0, b1, b0, b1, b0, b1, b0, b1];
                for r in 0..g.rows {
                    let w = wb.add(r * g.len * 16);
                    let x: [*const f32; P] = core::array::from_fn(|i| ip.add(a[i] + r * g.stride));
                    for j in 0..g.len {
                        let w0 = _mm256_loadu_ps(w.add(j * 16));
                        let w1 = _mm256_loadu_ps(w.add(j * 16 + 8));
                        for i in 0..P {
                            let xv = _mm256_broadcast_ss(&*x[i].add(j));
                            acc[2 * i] = _mm256_fmadd_ps(xv, w0, acc[2 * i]);
                            acc[2 * i + 1] = _mm256_fmadd_ps(xv, w1, acc[2 * i + 1]);
                        }
                    }
                }
                for i in 0..P.min(n - p) {
                    let dst = op.add(g.o[p + i] + cb * 16);
                    let (mut lo, mut hi) = (acc[2 * i], acc[2 * i + 1]);
                    if g.relu {
                        lo = _mm256_max_ps(lo, zero);
                        hi = _mm256_max_ps(hi, zero);
                    }
                    _mm256_storeu_ps(dst, lo);
                    _mm256_storeu_ps(dst.add(8), hi);
                }
                p += P;
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::Gemm;
    use core::arch::aarch64::*;

    /// 5 pixels x 16 channels = 20 accumulators: more independent FMA chains
    /// than four 4-cycle FMA pipes need, plus 4 weight and 5 input vectors
    /// (29 of the 32 registers). 5x5, 10x10 and 20x20 maps divide evenly.
    const P: usize = 5;

    /// SAFETY: caller checks bounds (see `run`).
    #[allow(clippy::needless_range_loop)] // accumulators are indexed like registers
    pub unsafe fn run(g: &Gemm, out: &mut [f32]) {
        let (n, k) = (g.a.len(), g.rows * g.len);
        let (ip, op) = (g.inp.as_ptr(), out.as_mut_ptr());
        let zero = vdupq_n_f32(0.0);
        for cb in 0..g.cout / 16 {
            let wb = g.w.as_ptr().add(cb * k * 16);
            let bp = g.b.as_ptr().add(cb * 16);
            let bias = [vld1q_f32(bp), vld1q_f32(bp.add(4)), vld1q_f32(bp.add(8)), vld1q_f32(bp.add(12))];
            let mut p = 0;
            while p < n {
                let a: [usize; P] = core::array::from_fn(|i| g.a[(p + i).min(n - 1)]);
                let mut acc = [bias; P];
                for r in 0..g.rows {
                    let w = wb.add(r * g.len * 16);
                    let x: [*const f32; P] = core::array::from_fn(|i| ip.add(a[i] + r * g.stride));
                    let mut j = 0;
                    // 4 inputs per pixel per load, consumed lane by lane
                    while j + 4 <= g.len {
                        let xv: [float32x4_t; P] = core::array::from_fn(|i| vld1q_f32(x[i].add(j)));
                        macro_rules! step {
                            ($l:literal) => {
                                let wj = w.add((j + $l) * 16);
                                let wv = [vld1q_f32(wj), vld1q_f32(wj.add(4)), vld1q_f32(wj.add(8)), vld1q_f32(wj.add(12))];
                                for i in 0..P {
                                    for c in 0..4 {
                                        acc[i][c] = vfmaq_laneq_f32::<$l>(acc[i][c], wv[c], xv[i]);
                                    }
                                }
                            };
                        }
                        step!(0);
                        step!(1);
                        step!(2);
                        step!(3);
                        j += 4;
                    }
                    while j < g.len {
                        let wj = w.add(j * 16);
                        let wv = [vld1q_f32(wj), vld1q_f32(wj.add(4)), vld1q_f32(wj.add(8)), vld1q_f32(wj.add(12))];
                        for i in 0..P {
                            let xs = *x[i].add(j);
                            for c in 0..4 {
                                acc[i][c] = vfmaq_n_f32(acc[i][c], wv[c], xs);
                            }
                        }
                        j += 1;
                    }
                }
                for i in 0..P.min(n - p) {
                    let dst = op.add(g.o[p + i] + cb * 16);
                    for c in 0..4 {
                        let v = if g.relu { vmaxq_f32(acc[i][c], zero) } else { acc[i][c] };
                        vst1q_f32(dst.add(4 * c), v);
                    }
                }
                p += P;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate std;
    use std::vec::Vec;

    #[test]
    fn simd_matches_generic() {
        #[cfg(target_arch = "x86_64")]
        let simd = avx2_fma().then_some(avx2::run as unsafe fn(&Gemm, &mut [f32]));
        #[cfg(target_arch = "aarch64")]
        let simd = Some(neon::run as unsafe fn(&Gemm, &mut [f32]));
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        let simd: Option<unsafe fn(&Gemm, &mut [f32])> = None;
        if let Some(simd) = simd {
            let mut seed = 1u32;
            let mut rnd = || {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                (seed >> 8) as f32 / (1 << 24) as f32 - 0.5
            };
            // (pixels, rows, len, cout, relu): conv-like and fc-like shapes, odd pixel counts
            for &(n, rows, len, cout, relu) in &[(25, 3, 96, 32, true), (7, 3, 3, 16, true), (4, 1, 512, 128, false), (6, 1, 64, 16, false), (1, 3, 48, 48, true)] {
                let stride = len + 5;
                let inp: Vec<f32> = (0..n * stride + rows * stride + len).map(|_| rnd()).collect();
                let w: Vec<f32> = (0..cout * rows * len).map(|_| rnd()).collect();
                let b: Vec<f32> = (0..cout).map(|_| rnd()).collect();
                let a: Vec<usize> = (0..n).map(|p| p * stride).collect();
                let o: Vec<usize> = (0..n).map(|p| p * cout).collect();
                let g = Gemm { inp: &inp, a: &a, o: &o, rows, stride, len, w: &w, b: &b, cout, relu };
                let (mut x, mut y) = (std::vec![0.0; n * cout], std::vec![0.0; n * cout]);
                generic(&g, &mut x);
                unsafe { simd(&g, &mut y) };
                for (p, q) in x.iter().zip(&y) {
                    assert!((p - q).abs() <= 1e-4 * (1.0 + p.abs()), "{p} vs {q}");
                }
            }
        }
    }
}
