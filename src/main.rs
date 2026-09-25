//! ccm-cli: solve click captchas from the command line.
//!
//! ```text
//! ccm-cli captcha.jpg              101-45,179-48,14-51,51-49 0.225842
//! ccm-cli -j a.jpg b.jpg           one JSON object per line
//! ccm-cli -b 20 samples/*.jpg      timing: decode + crop + model + assign, 1 thread
//! ccm-cli -m models/s1.ccm x.jpg   another model file ("-" reads the image from stdin)
//! ```

#![no_std]
#![no_main]

extern crate alloc;

use alloc::vec::Vec;
use ccm::{kernel, Error, Model, Solution, Solver, EMBEDDED_MODEL};
use core::ffi::{c_char, c_int, CStr};

const USAGE: &[u8] = b"usage: ccm-cli [-j] [-m model.ccm] [-b N] [-g] image.jpg|- ...\n\
  prints `x0-y0,x1-y1,x2-y2,x3-y3 margin` per image, points in prompt order\n\
  -j    JSON lines\n\
  -m    .ccm model file (default: the embedded w16)\n\
  -b N  benchmark: solve every image N times, print ms/captcha\n\
  -g    use the portable kernel instead of AVX2 / NEON\n";

/// Seconds on a monotonic clock.
#[cfg(unix)]
fn now() -> f64 {
    let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    // SAFETY: writes into a live timespec.
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
    ts.tv_sec as f64 + ts.tv_nsec as f64 * 1e-9
}

#[cfg(windows)]
fn now() -> f64 {
    // SAFETY: clock() has no preconditions; CLOCKS_PER_SEC is 1000 on Windows.
    unsafe { libc::clock() as f64 / 1e3 }
}

/// Buffered output with just the formatting this tool needs (no core::fmt, no printf).
struct Out(Vec<u8>);

impl Out {
    fn s(&mut self, s: &[u8]) -> &mut Self {
        self.0.extend_from_slice(s);
        self
    }

    fn int(&mut self, v: i64) -> &mut Self {
        if v < 0 {
            self.0.push(b'-');
        }
        let (mut u, mut d, mut n) = (v.unsigned_abs(), [0u8; 20], 0);
        loop {
            d[n] = b'0' + (u % 10) as u8;
            n += 1;
            u /= 10;
            if u == 0 {
                break;
            }
        }
        self.0.extend(d[..n].iter().rev());
        self
    }

    fn fixed(&mut self, v: f64, digits: u32) -> &mut Self {
        let scale = 10u64.pow(digits);
        let u = (if v < 0.0 { -v } else { v } * scale as f64 + 0.5) as u64;
        if v < 0.0 && u != 0 {
            self.0.push(b'-');
        }
        self.int((u / scale) as i64).s(b".");
        let frac = u % scale;
        for k in (0..digits).rev() {
            self.0.push(b'0' + (frac / 10u64.pow(k) % 10) as u8);
        }
        self
    }

    fn json_str(&mut self, s: &[u8]) -> &mut Self {
        self.0.push(b'"');
        for &c in s {
            match c {
                b'"' | b'\\' => self.s(&[b'\\', c]),
                0..=0x1f => self.s(b"\\u00").s(&[b"0123456789abcdef"[(c >> 4) as usize], b"0123456789abcdef"[(c & 15) as usize]]),
                _ => self.s(&[c]),
            };
        }
        self.s(b"\"")
    }

    fn flush(&mut self, fd: c_int) {
        let mut rest = &self.0[..];
        while !rest.is_empty() {
            // SAFETY: writing from a live buffer.
            let n = unsafe { libc::write(fd, rest.as_ptr().cast(), rest.len().min(1 << 30) as _) };
            if n <= 0 {
                break;
            }
            rest = &rest[n as usize..];
        }
        self.0.clear();
    }
}

fn reason(e: Error) -> &'static [u8] {
    match e {
        Error::Arg => b"bad argument",
        Error::Jpeg => b"not a valid JPEG",
        Error::Unsupported => b"unsupported JPEG variant",
        Error::Size => b"not a 250x80 image",
        Error::Model => b"bad model file",
    }
}

#[cfg(windows)]
extern "C" {
    fn _setmode(fd: c_int, mode: c_int) -> c_int;
}

fn read_all(path: &CStr) -> Option<Vec<u8>> {
    let stdin = path.to_bytes() == b"-";
    // SAFETY: plain stdio calls on a file we open and close here.
    unsafe {
        #[cfg(windows)]
        if stdin {
            _setmode(0, 0x8000); // _O_BINARY
        }
        let f = if stdin {
            libc::fdopen(0, c"rb".as_ptr())
        } else {
            libc::fopen(path.as_ptr(), c"rb".as_ptr())
        };
        if f.is_null() {
            return None;
        }
        let (mut data, mut buf) = (Vec::new(), [0u8; 16384]);
        loop {
            let n = libc::fread(buf.as_mut_ptr().cast(), 1, buf.len(), f);
            data.extend_from_slice(&buf[..n]);
            if n < buf.len() {
                break;
            }
        }
        if !stdin {
            libc::fclose(f);
        }
        Some(data)
    }
}

fn fail(msg: &[&[u8]], code: c_int) -> c_int {
    let mut err = Out(Vec::new());
    err.s(b"ccm-cli: ");
    for m in msg {
        err.s(m);
    }
    err.s(b"\n").flush(2);
    code
}

fn print_solution(out: &mut Out, name: &[u8], sol: &Solution, json: bool, prefix: bool) {
    if json {
        out.s(b"{\"file\": ").json_str(name).s(b", \"points\": [");
        for (i, p) in sol.points.iter().enumerate() {
            out.s(if i > 0 { b", [" } else { b"[" }).int(p[0] as i64).s(b", ").int(p[1] as i64).s(b"]");
        }
        out.s(b"], \"margin\": ").fixed(sol.margin as f64, 6).s(b"}\n");
    } else {
        if prefix {
            out.s(name).s(b" ");
        }
        for (i, p) in sol.points.iter().enumerate() {
            out.s(if i > 0 { b"," } else { b"" }).int(p[0] as i64).s(b"-").int(p[1] as i64);
        }
        out.s(b" ").fixed(sol.margin as f64, 6).s(b"\n");
    }
}

/// # Safety
/// Called by the C runtime, with `argv` holding `argc` valid C strings.
#[no_mangle]
pub unsafe extern "C" fn main(argc: c_int, argv: *const *const c_char) -> c_int {
    let args: Vec<&CStr> = core::slice::from_raw_parts(argv, argc as usize).iter().map(|&p| CStr::from_ptr(p)).collect();

    let (mut json, mut rounds, mut model_path, mut files) = (false, 0usize, None, Vec::new());
    let mut i = 1;
    while i < args.len() {
        match args[i].to_bytes() {
            b"-j" => json = true,
            b"-g" => kernel::force_generic(),
            b"-h" | b"--help" => {
                Out(Vec::new()).s(USAGE).flush(1);
                return 0;
            }
            b"-m" | b"-b" if i + 1 < args.len() => {
                if args[i].to_bytes() == b"-m" {
                    model_path = Some(args[i + 1]);
                } else {
                    rounds = args[i + 1].to_bytes().iter().try_fold(0usize, |n, &c| {
                        c.is_ascii_digit().then(|| n.saturating_mul(10).saturating_add((c - b'0') as usize))
                    }).unwrap_or(0);
                    if rounds == 0 {
                        return fail(&[b"-b needs a positive count"], 2);
                    }
                }
                i += 1;
            }
            a if a.len() > 1 && a[0] == b'-' => return fail(&[b"unknown option ", a, b"\n", USAGE], 2),
            _ => files.push(args[i]),
        }
        i += 1;
    }
    if files.is_empty() {
        return fail(&[USAGE], 2);
    }

    let model_bytes = match model_path {
        Some(p) => match read_all(p) {
            Some(b) => b,
            None => return fail(&[b"cannot read ", p.to_bytes()], 2),
        },
        None => EMBEDDED_MODEL.to_vec(),
    };
    let solver = match Model::load(&model_bytes) {
        Ok(m) => Solver::new(m),
        Err(_) => return fail(&[reason(Error::Model)], 2),
    };

    let mut out = Out(Vec::new());
    let mut status = 0;
    let mut blobs = Vec::new();
    for &f in &files {
        let name = f.to_bytes();
        let Some(data) = read_all(f) else {
            status = fail(&[b"cannot read ", name], 1);
            continue;
        };
        match solver.solve(&data) {
            Ok(sol) if rounds == 0 => print_solution(&mut out, name, &sol, json, files.len() > 1),
            Ok(_) => blobs.push(data),
            Err(e) if json => {
                out.s(b"{\"file\": ").json_str(name).s(b", \"error\": \"").s(reason(e)).s(b"\"}\n");
                status = 1;
            }
            Err(e) => status = fail(&[name, b": ", reason(e)], 1),
        }
        out.flush(1);
    }

    if rounds > 0 && !blobs.is_empty() {
        let mut check = 0f64;
        let t0 = now();
        for _ in 0..rounds {
            for b in &blobs {
                check += solver.solve(b).map_or(0.0, |s| s.margin as f64);
            }
        }
        let secs = now() - t0;
        let n = rounds * blobs.len();
        out.int(blobs.len() as i64).s(b" images x ").int(rounds as i64).s(b" rounds: ")
            .fixed(secs * 1e3 / n as f64, 3).s(b" ms/captcha end-to-end, 1 thread, kernel ")
            .s(kernel::isa_name().as_bytes()).s(b" (checksum ").fixed(check / n as f64, 4).s(b")\n");
        out.flush(1);
    }
    status
}
