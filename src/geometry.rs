//! Fixed-layout cropping for the 250x80 click captcha: a line-by-line port of
//! geometry.py, down to numpy's float64 arithmetic and round-half-to-even, so
//! the crops and click points are identical to the Python solver's.
//!
//! Layout (measured on 1229 field samples):
//!   * 4 prompt glyphs in a fixed 13px grid: x = 81 + 13k .. +12, rows 2..12.
//!   * 6 candidate glyphs in fixed slots separated by columns that never carry
//!     ink. Glyphs are ~24px, rotated, rows ~30..65.
//!   * Interference lines are light gray; glyph ink is dark (< 110).

pub const W: usize = 250;
pub const H: usize = 80;
pub const PROMPT_X0: usize = 81;
pub const PROMPT_PITCH: usize = 13;
pub const PROMPT_ROWS: (usize, usize) = (2, 14);
pub const PROMPT_SIZE: usize = 16;
pub const SLOTS: [(usize, usize); 6] =
    [(0, 36), (37, 76), (77, 116), (117, 156), (157, 196), (198, 237)];
pub const CAND_ROWS: (usize, usize) = (25, 78);
pub const CAND_SIZE: usize = 40;
pub const INK_THRESHOLD: u8 = 110;

pub const PROMPT_PIXELS: usize = PROMPT_SIZE * PROMPT_SIZE;
pub const CAND_PIXELS: usize = CAND_SIZE * CAND_SIZE;

/// uint8 gray (white background) -> float32 ink in [0, 1], same rounding as numpy.
#[inline]
fn ink(g: u8) -> f32 {
    (255.0 - g as f32) / 255.0
}

/// Python's round() on a float32: nearest integer, ties to even.
pub fn round_half_even(x: f32) -> i32 {
    let t = x as i32;
    let floor = if (t as f32) > x { t - 1 } else { t };
    let frac = x - floor as f32; // exact for |x| < 2^23
    if frac > 0.5 || (frac == 0.5 && floor % 2 != 0) {
        floor + 1
    } else {
        floor
    }
}

/// 4 prompt cells of 16x16 (row-major, concatenated); glyph at a fixed offset.
pub fn prompt_crops(gray: &[u8], out: &mut [f32]) {
    let (y0, y1) = PROMPT_ROWS;
    for (k, cell) in out.chunks_exact_mut(PROMPT_PIXELS).take(4).enumerate() {
        cell.fill(0.0);
        let x0 = PROMPT_X0 + PROMPT_PITCH * k - 1;
        for y in y0..y1 {
            let src = &gray[y * W + x0..y * W + x0 + 14];
            let dst = &mut cell[(2 + y - y0) * PROMPT_SIZE + 1..][..14];
            for (d, &g) in dst.iter_mut().zip(src) {
                *d = ink(g);
            }
        }
    }
}

/// (x, y) click centers: dark-ink centroid inside each slot.
pub fn candidate_centers(gray: &[u8]) -> [[f32; 2]; 6] {
    let (y0, y1) = CAND_ROWS;
    let mut centers = [[0f32; 2]; 6];
    for (k, &(a, b)) in SLOTS.iter().enumerate() {
        let mut rows = [0i64; CAND_ROWS.1 - CAND_ROWS.0];
        let mut cols = [0i64; 40];
        for y in y0..y1 {
            for x in a..b {
                if gray[y * W + x] < INK_THRESHOLD {
                    rows[y - y0] += 1;
                    cols[x - a] += 1;
                }
            }
        }
        centers[k] = if rows.iter().sum::<i64>() < 5 {
            // nothing dark: fall back to the slot center
            [((a + b) as f64 / 2.0) as f32, ((y0 + y1) as f64 / 2.0) as f32]
        } else {
            // Winsorized centroid (5%/95%) of the projections, so stray dark
            // line pixels don't drag the center.
            [(a as f64 + clipped_mean(&cols[..b - a])) as f32, (y0 as f64 + clipped_mean(&rows)) as f32]
        };
    }
    centers
}

/// Mean of positions weighted by `hist` after clipping them to its 5%/95% quantiles.
fn clipped_mean(hist: &[i64]) -> f64 {
    let q = 0.05f64;
    let total: i64 = hist.iter().sum();
    let (lo_v, hi_v) = (q * total as f64, (1.0 - q) * total as f64);
    // np.searchsorted(cum, lo_v, "right") and np.searchsorted(cum, hi_v, "left")
    let (mut lo, mut hi, mut cum) = (hist.len(), hist.len(), 0i64);
    for (i, &h) in hist.iter().enumerate() {
        cum += h;
        if lo == hist.len() && cum as f64 > lo_v {
            lo = i;
        }
        if hi == hist.len() && cum as f64 >= hi_v {
            hi = i;
        }
    }
    let s: i64 = hist.iter().enumerate().map(|(i, &h)| i.max(lo).min(hi) as i64 * h).sum();
    s as f64 / total as f64
}

/// 6 crops of 40x40 centered on each candidate glyph (clipped to its slot).
pub fn candidate_crops(gray: &[u8], centers: &[[f32; 2]; 6], out: &mut [f32]) {
    const HALF: i32 = CAND_SIZE as i32 / 2;
    for (k, crop) in out.chunks_exact_mut(CAND_PIXELS).take(6).enumerate() {
        crop.fill(0.0);
        let (a, b) = (SLOTS[k].0 as i32, SLOTS[k].1 as i32);
        let (cx, cy) = (round_half_even(centers[k][0]), round_half_even(centers[k][1]));
        let (ya, xa) = (cy - HALF, cx - HALF);
        let (sy0, sy1) = (ya.max(CAND_ROWS.0 as i32), (ya + 40).min(CAND_ROWS.1 as i32));
        let (sx0, sx1) = (xa.max(a), (xa + 40).min(b));
        for y in sy0..sy1 {
            for x in sx0..sx1 {
                crop[((y - ya) * 40 + (x - xa)) as usize] = ink(gray[y as usize * W + x as usize]);
            }
        }
    }
}
