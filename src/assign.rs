//! 360-permutation assignment over the 4x6 similarity matrix (port of assign.py).

pub struct Assignment {
    /// Candidate slot of each prompt glyph, in prompt order.
    pub slots: [usize; 4],
    pub score: f32,
    /// Best minus runner-up total score: low margin => refetch instead of submit.
    pub margin: f32,
}

/// Best injective prompt -> candidate mapping over all 6P4 = 360 permutations,
/// visited in itertools.permutations order and summed left to right like numpy.
pub fn assign(sim: &[[f32; 6]; 4]) -> Assignment {
    let (mut best, mut second, mut slots) = (f32::NEG_INFINITY, f32::NEG_INFINITY, [0; 4]);
    for a in 0..6 {
        for b in (0..6).filter(|&b| b != a) {
            for c in (0..6).filter(|&c| c != a && c != b) {
                for d in (0..6).filter(|&d| d != a && d != b && d != c) {
                    let s = sim[0][a] + sim[1][b] + sim[2][c] + sim[3][d];
                    if s > best {
                        (second, best, slots) = (best, s, [a, b, c, d]);
                    } else if s > second {
                        second = s;
                    }
                }
            }
        }
    }
    Assignment { slots, score: best, margin: best - second }
}
