//! Tests against tests/fixtures (regenerate with tools/make_fixtures.py).

extern crate std;

use crate::geometry::{round_half_even, H, W};
use crate::{jpeg, Error, Model, Solver, Work, EMBEDDED_MODEL};
use std::string::String;
use std::vec::Vec;

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/");

struct Fixture {
    name: String,
    data: Vec<u8>,
    status: i32,
    hash: String,
    answer: Option<(String, f32)>,
}

fn fixtures() -> Vec<Fixture> {
    let manifest = std::fs::read_to_string(std::format!("{FIXTURES}expected.txt")).unwrap();
    manifest
        .lines()
        .map(|line| {
            let f: Vec<&str> = line.split_whitespace().collect();
            Fixture {
                name: f[0].into(),
                data: std::fs::read(std::format!("{FIXTURES}{}", f[0])).unwrap(),
                status: f[1].parse().unwrap(),
                hash: f[2].into(),
                answer: (f.len() == 5).then(|| (f[3].into(), f[4].parse().unwrap())),
            }
        })
        .collect()
}

fn fnv1a(data: &[u8]) -> String {
    let h = data.iter().fold(0xcbf29ce484222325u64, |h, &b| (h ^ b as u64).wrapping_mul(0x100000001b3));
    std::format!("{h:016x}")
}

fn decode(data: &[u8]) -> Result<Vec<u8>, Error> {
    let (mut planes, mut gray) = (std::vec![0u8; jpeg::SCRATCH], std::vec![0u8; W * H]);
    jpeg::decode(data, &mut planes, &mut gray)?;
    Ok(gray)
}

#[test]
fn decoder_matches_pillow() {
    for f in fixtures() {
        match decode(&f.data) {
            Ok(gray) => {
                assert_eq!(f.status, 0, "{} should fail", f.name);
                assert_eq!(fnv1a(&gray), f.hash, "{}: pixels differ from Pillow", f.name);
            }
            Err(e) => assert_eq!(e as i32, f.status, "{}", f.name),
        }
    }
}

#[test]
fn solves_synthetic_captchas() {
    let solver = Solver::new(Model::load(EMBEDDED_MODEL).unwrap());
    let mut n = 0;
    for f in fixtures() {
        let Some((answer, margin)) = &f.answer else { continue };
        let sol = solver.solve(&f.data).unwrap();
        let got: Vec<String> = sol.points.iter().map(|p| std::format!("{}-{}", p[0], p[1])).collect();
        assert_eq!(&got.join(","), answer, "{}", f.name);
        assert!((sol.margin - margin).abs() < 0.02, "{}: margin {} vs {}", f.name, sol.margin, margin);
        n += 1;
    }
    assert!(n >= 4);
}

#[test]
fn work_buffers_are_reusable() {
    let model = Model::load(EMBEDDED_MODEL).unwrap();
    let mut work = Work::new(&model);
    let files = fixtures();
    let first = work.solve(&model, &files[0].data).unwrap().points;
    for f in &files {
        let _ = work.solve(&model, &f.data);
    }
    assert_eq!(work.solve(&model, &files[0].data).unwrap().points, first);
}

#[test]
fn corrupt_input_never_panics() {
    // The JPEG comes from a remote server: any byte string must give Ok or Err.
    let base = &fixtures()[0].data;
    let mut seed = 0x2545f4914f6cdd1du64;
    let mut rand = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    for cut in (0..base.len()).step_by(97) {
        let _ = decode(&base[..cut]);
    }
    for _ in 0..3000 {
        let mut d = base.clone();
        for _ in 0..1 + rand() % 8 {
            let i = rand() as usize % d.len();
            d[i] = rand() as u8;
        }
        let _ = decode(&d);
    }
    let _ = decode(&[0xFF, 0xD8, 0xFF]);
    let _ = decode(&[]);
}

#[test]
fn model_file_is_validated() {
    assert!(Model::load(EMBEDDED_MODEL).is_ok());
    assert!(Model::load(&EMBEDDED_MODEL[..EMBEDDED_MODEL.len() - 1]).is_err());
    let mut longer = EMBEDDED_MODEL.to_vec();
    longer.push(0);
    assert!(Model::load(&longer).is_err());
    let mut magic = EMBEDDED_MODEL.to_vec();
    magic[0] = b'X';
    assert!(Model::load(&magic).is_err());
    assert!(Model::load(&[]).is_err());
}

#[test]
fn rounds_like_python() {
    for (x, want) in [(0.5, 0), (1.5, 2), (2.5, 2), (17.5, 18), (18.5, 18), (2.4999, 2), (2.5001, 3), (51.5, 52), (0.0, 0)] {
        assert_eq!(round_half_even(x), want, "round({x})");
    }
}

#[test]
fn assignment_is_injective_and_scored() {
    let mut sim = [[0.0f32; 6]; 4];
    for (i, s) in [5, 2, 0, 3].into_iter().enumerate() {
        sim[i][s] = 1.0;
    }
    sim[0][2] = 0.9; // tempting, but slot 2 belongs to prompt 1: [2, _, 0, 3] only scores 2.9
    sim[3][1] = 0.8; // runner-up: [5, 2, 0, 1] = 3.8
    let a = crate::assign::assign(&sim);
    assert_eq!(a.slots, [5, 2, 0, 3]);
    assert!((a.score - 4.0).abs() < 1e-6 && (a.margin - 0.2).abs() < 1e-6, "{} {}", a.score, a.margin);
}
