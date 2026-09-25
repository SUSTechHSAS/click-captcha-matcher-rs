//! C ABI, see include/ccm.h. Every function is thread-safe; a solver may be shared.

use crate::geometry::{H, W};
use crate::{jpeg, kernel, Error, Model, Solution, Solver, EMBEDDED_MODEL};
use alloc::boxed::Box;
use alloc::vec;
use core::ffi::c_char;
use core::{ptr, slice};

/// Load a .ccm model (`model` = NULL: the embedded one). NULL on failure, with
/// the reason in `*err` when `err` is not NULL.
#[no_mangle]
pub unsafe extern "C" fn ccm_new(model: *const u8, len: usize, err: *mut i32) -> *mut Solver {
    let bytes = if model.is_null() { EMBEDDED_MODEL } else { slice::from_raw_parts(model, len) };
    match Model::load(bytes) {
        Ok(m) => Box::into_raw(Box::new(Solver::new(m))),
        Err(_) => {
            if !err.is_null() {
                *err = Error::Model as i32;
            }
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn ccm_free(s: *mut Solver) {
    if !s.is_null() {
        drop(Box::from_raw(s));
    }
}

unsafe fn write_solution(sol: &Solution, points: *mut i32, margin: *mut f32) {
    let out = slice::from_raw_parts_mut(points, 8);
    for (i, p) in sol.points.iter().enumerate() {
        out[2 * i] = p[0];
        out[2 * i + 1] = p[1];
    }
    if !margin.is_null() {
        *margin = sol.margin;
    }
}

/// JPEG bytes -> points[8] = x0,y0,..,x3,y3 in prompt order, *margin = confidence.
#[no_mangle]
pub unsafe extern "C" fn ccm_solve(
    s: *const Solver,
    jpeg: *const u8,
    len: usize,
    points: *mut i32,
    margin: *mut f32,
) -> i32 {
    if s.is_null() || jpeg.is_null() || points.is_null() {
        return Error::Arg as i32;
    }
    match (*s).solve(slice::from_raw_parts(jpeg, len)) {
        Ok(sol) => {
            write_solution(&sol, points, margin);
            0
        }
        Err(e) => e as i32,
    }
}

/// Like `ccm_solve`, for an already decoded 250x80 gray image (row-major).
#[no_mangle]
pub unsafe extern "C" fn ccm_solve_gray(
    s: *const Solver,
    gray: *const u8,
    points: *mut i32,
    margin: *mut f32,
) -> i32 {
    if s.is_null() || gray.is_null() || points.is_null() {
        return Error::Arg as i32;
    }
    let sol = (*s).with_work(|m, w| {
        w.gray.copy_from_slice(slice::from_raw_parts(gray, W * H));
        w.solve_gray(m)
    });
    write_solution(&sol, points, margin);
    0
}

/// JPEG bytes -> 250x80 gray, identical to Pillow's `Image.open(f).convert("L")`.
#[no_mangle]
pub unsafe extern "C" fn ccm_decode(jpeg: *const u8, len: usize, gray: *mut u8) -> i32 {
    if jpeg.is_null() || gray.is_null() {
        return Error::Arg as i32;
    }
    let mut planes = vec![0u8; jpeg::SCRATCH];
    let out = slice::from_raw_parts_mut(gray, W * H);
    match jpeg::decode(slice::from_raw_parts(jpeg, len), &mut planes, out) {
        Ok(()) => 0,
        Err(e) => Error::from(e) as i32,
    }
}

/// Gray image -> sim[24] (4x6 cosine similarity, row-major) and centers[12] (x, y per slot).
#[no_mangle]
pub unsafe extern "C" fn ccm_similarity(
    s: *const Solver,
    gray: *const u8,
    sim: *mut f32,
    centers: *mut f32,
) -> i32 {
    if s.is_null() || gray.is_null() || sim.is_null() {
        return Error::Arg as i32;
    }
    let (m, c) = (*s).with_work(|model, w| {
        w.gray.copy_from_slice(slice::from_raw_parts(gray, W * H));
        w.similarity(model)
    });
    slice::from_raw_parts_mut(sim, 24).copy_from_slice(m.as_flattened());
    if !centers.is_null() {
        slice::from_raw_parts_mut(centers, 12).copy_from_slice(c.as_flattened());
    }
    0
}

#[no_mangle]
pub extern "C" fn ccm_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr().cast()
}

/// Kernel in use: "avx2", "neon" or "generic".
#[no_mangle]
pub extern "C" fn ccm_isa() -> *const c_char {
    match kernel::isa_name() {
        "avx2" => c"avx2".as_ptr(),
        "neon" => c"neon".as_ptr(),
        _ => c"generic".as_ptr(),
    }
}

/// Switch to the portable kernel (testing / benchmarking).
#[no_mangle]
pub extern "C" fn ccm_force_generic() {
    kernel::force_generic();
}
