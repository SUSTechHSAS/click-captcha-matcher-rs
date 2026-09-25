//! click-captcha-matcher: 250x80 click captcha JPEG -> 4 click points, in prompt order.
//!
//! JPEG decoding, fixed-layout cropping, the two-tower CNN and the 360-permutation
//! assignment, all in `no_std` Rust with libc as the only dependency. The C ABI
//! is in `include/ccm.h`; `python/solver.py` wraps it with ctypes.

#![cfg_attr(not(test), no_std)]

extern crate alloc;

pub mod assign;
mod capi;
pub mod geometry;
pub mod jpeg;
pub mod kernel;
pub mod model;
#[cfg(not(test))]
mod rt;
#[cfg(test)]
mod tests;

use alloc::vec;
use alloc::vec::Vec;
use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicBool, Ordering};

use geometry::{round_half_even, CAND_PIXELS, H, PROMPT_PIXELS, W};
pub use model::Model;

/// models/w16.ccm, the recommended model: GPTQ-quantized, 4-bit prompt fc,
/// 6-bit last candidate conv, 8-bit elsewhere (124 KB; see tools/export_ccm.py).
#[cfg(feature = "embed-model")]
pub static EMBEDDED_MODEL: &[u8] = include_bytes!("../models/w16.ccm");
#[cfg(not(feature = "embed-model"))]
pub static EMBEDDED_MODEL: &[u8] = &[];

/// Error codes shared with the C ABI.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum Error {
    Arg = -1,
    Jpeg = -2,
    Unsupported = -3,
    Size = -4,
    Model = -5,
}

impl From<jpeg::Error> for Error {
    fn from(e: jpeg::Error) -> Error {
        match e {
            jpeg::Error::Format => Error::Jpeg,
            jpeg::Error::Unsupported => Error::Unsupported,
            jpeg::Error::Size => Error::Size,
        }
    }
}

pub struct Solution {
    /// Click points (x, y), in prompt order.
    pub points: [[i32; 2]; 4],
    pub slots: [usize; 4],
    /// Confidence: best minus runner-up assignment score.
    pub margin: f32,
    pub score: f32,
}

/// Buffers for one solve (~215 KB with w16); reuse it across calls.
pub struct Work {
    /// The decoded 250x80 gray image, row-major.
    pub gray: Vec<u8>,
    planes: Vec<u8>,
    prompts: Vec<f32>,
    cands: Vec<f32>,
    net: model::Scratch,
}

impl Work {
    pub fn new(m: &Model) -> Work {
        Work {
            gray: vec![0; W * H],
            planes: vec![0; jpeg::SCRATCH],
            prompts: vec![0.0; 4 * PROMPT_PIXELS],
            cands: vec![0.0; 6 * CAND_PIXELS],
            net: model::Scratch::new(m),
        }
    }

    /// Decode a JPEG into `self.gray` exactly like Pillow's `convert("L")`.
    pub fn decode(&mut self, jpeg: &[u8]) -> Result<(), Error> {
        Ok(jpeg::decode(jpeg, &mut self.planes, &mut self.gray)?)
    }

    /// (4, 6) cosine similarity and the 6 click centers for the image in `self.gray`.
    pub fn similarity(&mut self, m: &Model) -> ([[f32; 6]; 4], [[f32; 2]; 6]) {
        let centers = geometry::candidate_centers(&self.gray);
        geometry::prompt_crops(&self.gray, &mut self.prompts);
        geometry::candidate_crops(&self.gray, &centers, &mut self.cands);
        (m.similarity(&self.prompts, &self.cands, &mut self.net), centers)
    }

    pub fn solve_gray(&mut self, m: &Model) -> Solution {
        let (sim, centers) = self.similarity(m);
        let a = assign::assign(&sim);
        let points = a.slots.map(|s| [round_half_even(centers[s][0]), round_half_even(centers[s][1])]);
        Solution { points, slots: a.slots, margin: a.margin, score: a.score }
    }

    pub fn solve(&mut self, m: &Model, jpeg: &[u8]) -> Result<Solution, Error> {
        self.decode(jpeg)?;
        Ok(self.solve_gray(m))
    }
}

/// A model plus cached work buffers. `&Solver` can be shared between threads:
/// concurrent calls beyond the first get temporary buffers.
pub struct Solver {
    pub model: Model,
    work: UnsafeCell<Work>,
    busy: AtomicBool,
}

// SAFETY: `work` is only touched while `busy` is held (see `with_work`).
unsafe impl Sync for Solver {}

impl Solver {
    pub fn new(model: Model) -> Solver {
        let work = UnsafeCell::new(Work::new(&model));
        Solver { model, work, busy: AtomicBool::new(false) }
    }

    pub fn with_work<R>(&self, f: impl FnOnce(&Model, &mut Work) -> R) -> R {
        if self.busy.swap(true, Ordering::Acquire) {
            return f(&self.model, &mut Work::new(&self.model));
        }
        // SAFETY: we own `busy`, so no one else holds a reference to `work`.
        let r = f(&self.model, unsafe { &mut *self.work.get() });
        self.busy.store(false, Ordering::Release);
        r
    }

    pub fn solve(&self, jpeg: &[u8]) -> Result<Solution, Error> {
        self.with_work(|m, w| w.solve(m, jpeg))
    }
}
