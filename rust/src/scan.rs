//! Exact brute-force cosine top-k over the flat index.
//! Vectors are unit-normalized at build time, so similarity = dot product.

use crate::index::Index;
use rayon::prelude::*;

#[derive(Clone, Copy, Debug)]
pub struct Hit {
    pub row: u32,
    pub sim: f32,
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    // 8 independent accumulators -> LLVM autovectorizes to AVX-512 FMA
    // with target-cpu=native (dim is a multiple of 8 in practice; handle tail anyway).
    let n = a.len().min(b.len());
    let chunks = n / 8;
    let mut acc = [0.0f32; 8];
    for c in 0..chunks {
        let i = c * 8;
        for l in 0..8 {
            acc[l] += a[i + l] * b[i + l];
        }
    }
    let mut s = acc.iter().sum::<f32>();
    for i in chunks * 8..n {
        s += a[i] * b[i];
    }
    s
}

/// Fixed-size min-heap keeping the k largest (sim, row) pairs.
struct TopK {
    k: usize,
    heap: Vec<Hit>, // min-heap by sim, ties broken by row (larger row = smaller)
}

impl TopK {
    fn new(k: usize) -> Self {
        TopK { k, heap: Vec::with_capacity(k + 1) }
    }
    #[inline]
    fn less(a: Hit, b: Hit) -> bool {
        // "smaller" = worse: lower sim, or equal sim with larger row (keep smaller rows)
        a.sim < b.sim || (a.sim == b.sim && a.row > b.row)
    }
    #[inline]
    fn push(&mut self, h: Hit) {
        if self.heap.len() < self.k {
            self.heap.push(h);
            let mut i = self.heap.len() - 1;
            while i > 0 {
                let p = (i - 1) / 2;
                if Self::less(self.heap[i], self.heap[p]) {
                    self.heap.swap(i, p);
                    i = p;
                } else {
                    break;
                }
            }
        } else if !self.heap.is_empty() && Self::less(self.heap[0], h) {
            self.heap[0] = h;
            let mut i = 0;
            loop {
                let (l, r) = (2 * i + 1, 2 * i + 2);
                let mut m = i;
                if l < self.heap.len() && Self::less(self.heap[l], self.heap[m]) {
                    m = l;
                }
                if r < self.heap.len() && Self::less(self.heap[r], self.heap[m]) {
                    m = r;
                }
                if m == i {
                    break;
                }
                self.heap.swap(i, m);
                i = m;
            }
        }
    }
    #[inline]
    fn min_sim(&self) -> f32 {
        if self.heap.len() < self.k {
            f32::NEG_INFINITY
        } else {
            self.heap[0].sim
        }
    }
    fn into_sorted(self) -> Vec<Hit> {
        let mut v = self.heap;
        v.sort_by(|a, b| b.sim.partial_cmp(&a.sim).unwrap().then(a.row.cmp(&b.row)));
        v
    }
    fn merge(mut self, other: TopK) -> TopK {
        for h in other.heap {
            self.push(h);
        }
        self
    }
}

/// Single-query top-k with a row predicate (evaluated before the dot product).
pub fn topk_scan<F>(idx: &Index, query: &[f32], k: usize, pred: F) -> Vec<Hit>
where
    F: Fn(usize) -> bool + Sync,
{
    let dim = idx.dim();
    let n = idx.len();
    let vectors = idx.vectors();
    let chunk = 8192usize;
    (0..n.div_ceil(chunk))
        .into_par_iter()
        .map(|ci| {
            let start = ci * chunk;
            let end = (start + chunk).min(n);
            let mut tk = TopK::new(k);
            for row in start..end {
                if !pred(row) {
                    continue;
                }
                let v = &vectors[row * dim..(row + 1) * dim];
                let s = dot(query, v);
                if s > tk.min_sim() {
                    tk.push(Hit { row: row as u32, sim: s });
                }
            }
            tk
        })
        .reduce(|| TopK::new(k), TopK::merge)
        .into_sorted()
}

/// Serial single-query top-k — for use inside an outer rayon loop (sweep).
pub fn topk_scan_serial<F>(idx: &Index, query: &[f32], k: usize, pred: F) -> Vec<Hit>
where
    F: Fn(usize) -> bool,
{
    let dim = idx.dim();
    let vectors = idx.vectors();
    let mut tk = TopK::new(k);
    for row in 0..idx.len() {
        if !pred(row) {
            continue;
        }
        let v = &vectors[row * dim..(row + 1) * dim];
        let s = dot(query, v);
        if s > tk.min_sim() {
            tk.push(Hit { row: row as u32, sim: s });
        }
    }
    tk.into_sorted()
}

/// Multi-query scan: for each query, an independent top-k (single pass over
/// the corpus — memory-bandwidth bound, so extra queries are nearly free).
pub fn multi_topk_scan<F>(idx: &Index, queries: &[&[f32]], k: usize, pred: F) -> Vec<Vec<Hit>>
where
    F: Fn(usize) -> bool + Sync,
{
    let dim = idx.dim();
    let n = idx.len();
    let vectors = idx.vectors();
    let nq = queries.len();
    let chunk = 8192usize;
    let parts: Vec<Vec<TopK>> = (0..n.div_ceil(chunk))
        .into_par_iter()
        .map(|ci| {
            let start = ci * chunk;
            let end = (start + chunk).min(n);
            let mut tks: Vec<TopK> = (0..nq).map(|_| TopK::new(k)).collect();
            for row in start..end {
                if !pred(row) {
                    continue;
                }
                let v = &vectors[row * dim..(row + 1) * dim];
                for (qi, q) in queries.iter().enumerate() {
                    let s = dot(q, v);
                    if s > tks[qi].min_sim() {
                        tks[qi].push(Hit { row: row as u32, sim: s });
                    }
                }
            }
            tks
        })
        .collect();
    let mut merged: Vec<TopK> = (0..nq).map(|_| TopK::new(k)).collect();
    for part in parts {
        for (qi, tk) in part.into_iter().enumerate() {
            let m = std::mem::replace(&mut merged[qi], TopK::new(0));
            merged[qi] = m.merge(tk);
        }
    }
    merged.into_iter().map(|tk| tk.into_sorted()).collect()
}
