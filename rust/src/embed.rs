//! Bit-exact port of dsearch/embed.py `hashed` backend:
//! feature-hashed 1/2/3-grams of instruction tokens, sublinear TF, L2 norm.

use blake2::digest::{Update, VariableOutput};
use blake2::Blake2bVar;
use std::collections::HashMap;

pub const HASHED_DIM: usize = 512;

fn hash_idx(feature: &str, dim: usize) -> (usize, f64) {
    let mut h = Blake2bVar::new(8).unwrap();
    h.update(feature.as_bytes());
    let mut out = [0u8; 8];
    h.finalize_variable(&mut out).unwrap();
    let idx = u32::from_le_bytes([out[0], out[1], out[2], out[3]]) as usize % dim;
    let sign = if out[4] & 1 == 1 { 1.0 } else { -1.0 };
    (idx, sign)
}

/// Embed one token doc (the `tokens` string: "ppc N\n<tok> <tok> ...").
/// Accumulates in f64 to match Python float semantics, returns f32 unit vector.
pub fn embed_hashed_doc(doc: &str, dim: usize) -> Vec<f32> {
    let toks: Vec<&str> = doc.lines().last().unwrap_or("").split_whitespace().collect();
    let mut vec = vec![0.0f64; dim];
    let mut counts: HashMap<String, u32> = HashMap::new();
    for n in 1..=3usize {
        if toks.len() + 1 > n {
            for j in 0..=(toks.len() - n) {
                let g = toks[j..j + n].join(" ");
                *counts.entry(g).or_insert(0) += 1;
            }
        }
    }
    for (g, c) in &counts {
        let (idx, sign) = hash_idx(g, dim);
        vec[idx] += sign * (1.0 + (*c as f64).ln());
    }
    let norm = vec.iter().map(|v| v * v).sum::<f64>().sqrt();
    let norm = if norm == 0.0 { 1.0 } else { norm };
    vec.iter().map(|v| (v / norm) as f32).collect()
}

pub fn embed_hashed(docs: &[&str], dim: usize) -> Vec<Vec<f32>> {
    use rayon::prelude::*;
    docs.par_iter().map(|d| embed_hashed_doc(d, dim)).collect()
}
