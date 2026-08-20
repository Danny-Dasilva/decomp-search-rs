mod embed;
mod index;
mod ingest;
mod normalize;
mod scan;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use index::Index;
use scan::Hit;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(name = "dsearch", about = "Fast similarity search over decomp-project functions")]
struct Cli {
    /// Directory holding .dsi index files
    #[arg(long, global = true, env = "DSEARCH_INDEX_DIR", default_value = "data/dsi")]
    index_dir: PathBuf,
    /// Embedding backend (selects which index file is searched)
    #[arg(long, global = true, env = "DSEARCH_BACKEND", default_value = "local")]
    backend: String,
    /// Emit machine-readable JSON on stdout
    #[arg(long, global = true)]
    json: bool,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Whole-function similarity: nearest neighbors of a stored function
    Find {
        function: String,
        #[arg(long)]
        project: Option<String>,
        /// Only hits with match_pct >= this (default 99.5)
        #[arg(long, default_value_t = 99.5)]
        min_match: f32,
        /// Search everything (disables --min-match filter)
        #[arg(long)]
        all: bool,
        #[arg(short, default_value_t = 15)]
        k: usize,
        /// Drop hits from the query's own translation unit
        #[arg(long)]
        exclude_self_unit: bool,
        /// On zero hits, relax --min-match down the 99.5/99/95/90 ladder
        #[arg(long)]
        ladder: bool,
    },
    /// Construct-level (32-insn window) similarity search
    Findw {
        function: String,
        #[arg(long)]
        project: Option<String>,
        #[arg(long, default_value_t = 99.5)]
        min_match: f32,
        #[arg(long)]
        all: bool,
        #[arg(short, default_value_t = 15)]
        k: usize,
        #[arg(long)]
        exclude_self_unit: bool,
        /// On zero hits, relax --min-match down the 99.5/99/95/90 ladder
        #[arg(long)]
        ladder: bool,
    },
    /// One-call donor lookup for agents: whole-function twins + window
    /// (construct) twins, JSON, with the min-match fallback ladder built in
    Donors {
        function: String,
        #[arg(long)]
        project: Option<String>,
        #[arg(long, default_value_t = 99.5)]
        min_match: f32,
        /// k for whole-function twins
        #[arg(short, default_value_t = 10)]
        k: usize,
        /// k for window twins
        #[arg(long, default_value_t = 6)]
        wk: usize,
        /// backend for the window search (hashed catches literal constructs)
        #[arg(long, default_value = "hashed")]
        wbackend: String,
    },
    /// Freeform text query (hashed backend only; embeds at query time)
    Search {
        text: Vec<String>,
        #[arg(long)]
        project: Option<String>,
        #[arg(long, default_value_t = 99.5)]
        min_match: f32,
        #[arg(long)]
        all: bool,
        #[arg(short, default_value_t = 15)]
        k: usize,
    },
    /// Solvability sweep: every sub-max-match function ranked by its best
    /// matched neighbor (JSON output, jsonq-compatible)
    Sweep {
        #[arg(long)]
        project: Option<String>,
        #[arg(long, default_value_t = 99.999)]
        max_match: f32,
        #[arg(long, default_value_t = 8)]
        min_insns: u32,
        #[arg(long, default_value_t = 99.5)]
        donor_min: f32,
        #[arg(long, default_value_t = 0.85)]
        min_sim: f32,
        #[arg(short, default_value_t = 20)]
        k: usize,
    },
    /// Row counts for the backend's tables
    Stats,
    /// Recall benchmark over known twin pairs
    Eval {
        #[arg(long, default_value = "eval/known_pairs.json")]
        pairs: PathBuf,
        #[arg(short, default_value_t = 20)]
        k: usize,
    },
    /// Build a .dsi index from exported metadata (JSONL) + raw f32 vectors
    BuildIndex {
        #[arg(long)]
        meta: PathBuf,
        #[arg(long)]
        vectors: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 512)]
        dim: usize,
    },
    /// Ingest a dtk-based project's target objects into the index
    IngestDtk {
        root: PathBuf,
        #[arg(long)]
        project: String,
        #[arg(long, default_value = "GALE01")]
        version: String,
        /// decomp.dev report JSON (path or URL) for match percentages
        #[arg(long)]
        report: Option<String>,
        #[arg(long, default_value_t = 8)]
        min_insns: usize,
        /// Re-embed everything (ignore stored vectors)
        #[arg(long)]
        full: bool,
        /// Also (re)build the 32-insn sliding-window table
        #[arg(long)]
        windows: bool,
        /// Python interpreter with the dsearch package, used to embed
        /// new/changed docs for model backends (local/voyage)
        #[arg(long, env = "DSEARCH_PY")]
        py: Option<String>,
        /// Path to tools/embed_docs.py (defaults next to this repo)
        #[arg(long)]
        embed_script: Option<PathBuf>,
    },
    /// Micro-benchmark: repeated find queries, reports latency percentiles
    Bench {
        #[arg(long, default_value_t = 100)]
        iters: usize,
        /// Also benchmark window search
        #[arg(long)]
        windows: bool,
    },
}

fn table_path(dir: &Path, kind: &str, backend: &str) -> PathBuf {
    dir.join(format!("{kind}_{backend}.dsi"))
}

fn open_table(dir: &Path, kind: &str, backend: &str) -> Result<Index> {
    let p = table_path(dir, kind, backend);
    Index::open(&p).with_context(|| {
        format!(
            "no index at {} — build one with `dsearch build-index` or the exporter",
            p.display()
        )
    })
}

struct OutHit {
    sim: f32,
    row: usize,
    q_start: i32,
    t_start: i32,
}

/// One hit in dsearch.jsonq's exact `_slim` schema (drop-in compatible),
/// plus `same_unit` so agents can spot same-TU donors without re-deriving it.
fn slim(idx: &Index, h: &OutHit, windows: bool, q_unit: Option<(u32, u32)>) -> serde_json::Value {
    let r = &idx.rows()[h.row];
    let mut o = serde_json::json!({
        "name": idx.name(h.row),
        "unit": idx.unit(h.row),
        "src_path": idx.src_path(h.row),
        "match_pct": (r.match_pct as f64 * 1000.0).round() / 1000.0,
        "n_insns": r.n_insns,
        "sim": (h.sim as f64 * 10000.0).round() / 10000.0,
    });
    if let Some((pidx, uidx)) = q_unit {
        o["same_unit"] = serde_json::json!(r.project_idx == pidx && r.unit_idx == uidx);
    }
    if windows {
        o["q_at"] = serde_json::json!(h.q_start);
        o["t_at"] = serde_json::json!(h.t_start);
    }
    o
}

/// The observed agent fallback ladder: 99.5 -> 99 -> 95 -> 90.
const LADDER: [f32; 4] = [99.5, 99.0, 95.0, 90.0];

/// Run `run(min_match)` at the given threshold; on zero hits (and ladder
/// enabled) relax down the standard ladder. Returns (hits, effective threshold).
fn with_ladder<T>(min_match: Option<f32>, ladder: bool, mut run: impl FnMut(Option<f32>) -> Vec<T>) -> (Vec<T>, Option<f32>) {
    let first = run(min_match);
    if !first.is_empty() || !ladder || min_match.is_none() {
        return (first, min_match);
    }
    let start = min_match.unwrap();
    for m in LADDER.iter().copied().filter(|&m| m < start) {
        let hits = run(Some(m));
        if !hits.is_empty() {
            return (hits, Some(m));
        }
    }
    (vec![], min_match)
}

fn print_hits(idx: &Index, hits: &[OutHit], json: Option<serde_json::Value>, windows: bool, title: &str, q_unit: Option<(u32, u32)>) {
    if let Some(query) = json {
        let arr: Vec<serde_json::Value> = hits.iter().map(|h| slim(idx, h, windows, q_unit)).collect();
        println!("{}", serde_json::json!({"query": query, "hits": arr}));
        return;
    }
    println!("{title}");
    if windows {
        println!("{:>6}  {:>6}  {:>5}  {:>5}  {:<40}  unit", "sim", "match%", "q@", "t@", "function");
    } else {
        println!("{:>6}  {:>6}  {:>5}  {:<40}  unit", "sim", "match%", "insns", "function");
    }
    for h in hits {
        let r = &idx.rows()[h.row];
        let mp = if r.match_pct < 0.0 { "?".to_string() } else { format!("{:.1}", r.match_pct) };
        if windows {
            println!(
                "{:>6.3}  {:>6}  {:>5}  {:>5}  {:<40}  {}",
                h.sim, mp, h.q_start, h.t_start, idx.name(h.row), idx.unit(h.row)
            );
        } else {
            println!(
                "{:>6.3}  {:>6}  {:>5}  {:<40}  {}",
                h.sim, mp, r.n_insns, idx.name(h.row), idx.unit(h.row)
            );
        }
    }
}

/// Python cmd_find parity: match-filtered top-(k*10+50), then drop self /
/// same-unit client-side, truncate to k.
fn do_find(idx: &Index, qrow: usize, k: usize, min_match: Option<f32>, exclude_self_unit: bool) -> Vec<OutHit> {
    let rows = idx.rows();
    let query = idx.vector(qrow);
    let fetch = k * 10 + 50;
    let hits = scan::topk_scan(idx, query, fetch, |row| {
        min_match.is_none_or(|m| rows[row].match_pct >= m)
    });
    let (q_pidx, q_uidx) = (rows[qrow].project_idx, rows[qrow].unit_idx);
    hits.into_iter()
        .filter(|h| h.row as usize != qrow)
        .filter(|h| {
            !exclude_self_unit
                || !(rows[h.row as usize].project_idx == q_pidx && rows[h.row as usize].unit_idx == q_uidx)
        })
        .take(k)
        .map(|h| OutHit { sim: h.sim, row: h.row as usize, q_start: -1, t_start: -1 })
        .collect()
}

fn json_err(cli: &Cli, msg: &str) -> Result<()> {
    if cli.json {
        println!("{}", serde_json::json!({"error": msg}));
        std::process::exit(1);
    }
    bail!("{msg}");
}

#[allow(clippy::too_many_arguments)]
fn cmd_find(cli: &Cli, function: &str, project: Option<&str>, min_match: f32, all: bool, k: usize, exclude_self_unit: bool, ladder: bool) -> Result<()> {
    let idx = open_table(&cli.index_dir, "functions", &cli.backend)?;
    let cands = idx.rows_by_name(function, project);
    let Some(&qrow) = cands.first() else {
        return json_err(cli, &format!("function '{function}' not in index (backend {})", cli.backend));
    };
    let mm = if all { None } else { Some(min_match) };
    let (out, eff) = with_ladder(mm, ladder, |m| do_find(&idx, qrow, k, m, exclude_self_unit));
    let qr = &idx.rows()[qrow];
    let qjson = cli.json.then(|| {
        let mut q = serde_json::json!({
            "name": idx.name(qrow), "unit": idx.unit(qrow),
            "n_insns": qr.n_insns,
            "match_pct": (qr.match_pct as f64 * 1000.0).round() / 1000.0,
        });
        if eff != mm {
            q["min_match_used"] = serde_json::json!(eff);
        }
        q
    });
    print_hits(&idx, &out, qjson, false, &format!("similar to {function} ({} insns)", qr.n_insns), Some((qr.project_idx, qr.unit_idx)));
    Ok(())
}

/// Window search core: one batched multi-query scan, then per-candidate-
/// function best aggregation (Python cmd_findw parity).
fn do_findw(idx: &Index, qrows: &[usize], qname: &str, k: usize, min_match: Option<f32>, exclude_self_unit: bool) -> Vec<OutHit> {
    let rows = idx.rows();
    let queries: Vec<&[f32]> = qrows.iter().map(|&r| idx.vector(r)).collect();
    let per_q = scan::multi_topk_scan(idx, &queries, 80, |row| {
        min_match.is_none_or(|m| rows[row].match_pct >= m)
    });
    let q_pidx = rows[qrows[0]].project_idx;
    let q_uidx = rows[qrows[0]].unit_idx;
    use std::collections::HashMap;
    let mut best: HashMap<(u32, u32, &str), (f32, u32, i32, i32)> = HashMap::new();
    for (qi, hits) in per_q.iter().enumerate() {
        let q_start = rows[qrows[qi]].wstart;
        for h in hits {
            let hr = h.row as usize;
            if idx.name(hr) == qname && rows[hr].project_idx == q_pidx {
                continue; // self-function windows
            }
            if exclude_self_unit && rows[hr].project_idx == q_pidx && rows[hr].unit_idx == q_uidx {
                continue;
            }
            let key = (rows[hr].project_idx, rows[hr].unit_idx, idx.name(hr));
            let e = best.entry(key).or_insert((f32::NEG_INFINITY, h.row, -1, -1));
            if h.sim > e.0 {
                *e = (h.sim, h.row, q_start, rows[hr].wstart);
            }
        }
    }
    let mut agg: Vec<OutHit> = best
        .into_values()
        .map(|(sim, row, qs, ts)| OutHit { sim, row: row as usize, q_start: qs, t_start: ts })
        .collect();
    agg.sort_by(|a, b| b.sim.partial_cmp(&a.sim).unwrap().then(a.row.cmp(&b.row)));
    agg.truncate(k);
    agg
}

#[allow(clippy::too_many_arguments)]
fn cmd_findw(cli: &Cli, function: &str, project: Option<&str>, min_match: f32, all: bool, k: usize, exclude_self_unit: bool, ladder: bool) -> Result<()> {
    let idx = open_table(&cli.index_dir, "windows", &cli.backend)?;
    let mut qrows = idx.rows_by_name(function, project);
    qrows.truncate(1000);
    if qrows.is_empty() {
        return json_err(cli, &format!("no windows for '{function}' (backend {}) — ingest with --windows", cli.backend));
    }
    let mm = if all { None } else { Some(min_match) };
    let (agg, eff) = with_ladder(mm, ladder, |m| do_findw(&idx, &qrows, function, k, m, exclude_self_unit));
    let qr = &idx.rows()[qrows[0]];
    let qjson = cli.json.then(|| {
        let mut q = serde_json::json!({"name": function, "windows": qrows.len()});
        if eff != mm {
            q["min_match_used"] = serde_json::json!(eff);
        }
        q
    });
    print_hits(&idx, &agg, qjson, true, &format!("construct matches for {function} ({} windows)", qrows.len()), Some((qr.project_idx, qr.unit_idx)));
    Ok(())
}

/// One-call donor lookup: whole-function twins on the primary backend plus
/// window twins on --wbackend, ladder always on. Replaces the harness's
/// two-subprocess `donors()` round trip.
fn cmd_donors(cli: &Cli, function: &str, project: Option<&str>, min_match: f32, k: usize, wk: usize, wbackend: &str) -> Result<()> {
    let idx = open_table(&cli.index_dir, "functions", &cli.backend)?;
    let cands = idx.rows_by_name(function, project);
    let Some(&qrow) = cands.first() else {
        return json_err(cli, &format!("function '{function}' not in index (backend {})", cli.backend));
    };
    let qr = &idx.rows()[qrow];
    let q_unit = (qr.project_idx, qr.unit_idx);
    let (twins, eff) = with_ladder(Some(min_match), true, |m| do_find(&idx, qrow, k, m, false));
    let twins_json: Vec<serde_json::Value> = twins.iter().map(|h| slim(&idx, h, false, Some(q_unit))).collect();

    let (windows_json, wcount, weff) = match open_table(&cli.index_dir, "windows", wbackend) {
        Ok(widx) => {
            let mut qrows = widx.rows_by_name(function, project);
            qrows.truncate(1000);
            if qrows.is_empty() {
                (vec![], 0, None)
            } else {
                let wq = (widx.rows()[qrows[0]].project_idx, widx.rows()[qrows[0]].unit_idx);
                let (whits, weff) = with_ladder(Some(min_match), true, |m| do_findw(&widx, &qrows, function, wk, m, false));
                let js = whits.iter().map(|h| slim(&widx, h, true, Some(wq))).collect();
                (js, qrows.len(), weff)
            }
        }
        Err(_) => (vec![], 0, None),
    };

    println!(
        "{}",
        serde_json::json!({
            "query": {
                "name": idx.name(qrow), "unit": idx.unit(qrow), "src_path": idx.src_path(qrow),
                "n_insns": qr.n_insns,
                "match_pct": (qr.match_pct as f64 * 1000.0).round() / 1000.0,
                "min_match_used": eff, "windows": wcount, "windows_min_match_used": weff,
            },
            "twins": twins_json,
            "windows": windows_json,
        })
    );
    Ok(())
}

fn cmd_search(cli: &Cli, text: &str, project: Option<&str>, min_match: f32, all: bool, k: usize) -> Result<()> {
    if cli.backend != "hashed" {
        bail!("query-time embedding is only supported for --backend hashed (got '{}')", cli.backend);
    }
    let idx = open_table(&cli.index_dir, "functions", &cli.backend)?;
    let rows = idx.rows();
    let qv = embed::embed_hashed_doc(text, idx.dim());
    let mm = if all { None } else { Some(min_match) };
    let pidx = match project {
        Some(p) => match idx.project_idx_of(p) {
            Some(i) => Some(i),
            None => bail!("project '{p}' not in index"),
        },
        None => None,
    };
    let hits = scan::topk_scan(&idx, &qv, k, |row| {
        mm.is_none_or(|m| rows[row].match_pct >= m)
            && pidx.is_none_or(|pi| rows[row].project_idx == pi)
    });
    let out: Vec<OutHit> = hits
        .into_iter()
        .map(|h: Hit| OutHit { sim: h.sim, row: h.row as usize, q_start: -1, t_start: -1 })
        .collect();
    let qjson = cli.json.then(|| serde_json::json!({"text": text}));
    print_hits(&idx, &out, qjson, false, &format!("results for: {text}"), None);
    Ok(())
}

/// Solvability sweep (jsonq `sweep` parity): every sub-max-match function
/// ranked by its best matched (>= donor-min) neighbor. One batched scan.
#[allow(clippy::too_many_arguments)]
fn cmd_sweep(cli: &Cli, project: Option<&str>, max_match: f32, min_insns: u32, donor_min: f32, min_sim: f32, k: usize) -> Result<()> {
    let idx = open_table(&cli.index_dir, "functions", &cli.backend)?;
    let rows = idx.rows();
    let pidx = project.and_then(|p| idx.project_idx_of(p));
    if project.is_some() && pidx.is_none() {
        return json_err(cli, &format!("project '{}' not in index", project.unwrap()));
    }
    let qrows: Vec<usize> = (0..idx.len())
        .filter(|&r| {
            let row = &rows[r];
            row.match_pct >= 0.0
                && row.match_pct < max_match
                && row.n_insns >= min_insns
                && pidx.is_none_or(|pi| row.project_idx == pi)
        })
        .collect();
    use rayon::prelude::*;
    let results: Vec<(usize, Hit)> = qrows
        .par_iter()
        .filter_map(|&qr| {
            let hits = scan::topk_scan_serial(&idx, idx.vector(qr), k, |row| {
                rows[row].match_pct >= donor_min
            });
            let best = hits
                .into_iter()
                .find(|h| h.row as usize != qr && idx.name(h.row as usize) != idx.name(qr))?;
            (best.sim >= min_sim).then_some((qr, best))
        })
        .collect();
    let mut out = serde_json::Map::new();
    for (qr, best) in results {
        let br = best.row as usize;
        out.insert(
            idx.name(qr).to_string(),
            serde_json::json!({
                "sim": (best.sim as f64 * 10000.0).round() / 10000.0,
                "pct": (rows[qr].match_pct as f64 * 1000.0).round() / 1000.0,
                "n_insns": rows[qr].n_insns,
                "donor": idx.name(br),
                "donor_pct": (rows[br].match_pct as f64 * 1000.0).round() / 1000.0,
                "donor_unit": idx.unit(br),
                "donor_src": idx.src_path(br),
                "donor_insns": rows[br].n_insns,
                "same_unit": idx.unit(br) == idx.unit(qr) && rows[br].project_idx == rows[qr].project_idx,
            }),
        );
    }
    println!("{}", serde_json::Value::Object(out));
    Ok(())
}

fn cmd_eval(cli: &Cli, pairs: &Path, k: usize) -> Result<()> {
    let idx = open_table(&cli.index_dir, "functions", &cli.backend)?;
    let data: Vec<serde_json::Value> =
        serde_json::from_str(&std::fs::read_to_string(pairs).with_context(|| format!("read {}", pairs.display()))?)?;
    let mut ok = 0;
    for case in &data {
        let query = case["query"].as_str().unwrap_or("");
        let project = case["project"].as_str();
        let expect: Vec<&str> = case["expect"].as_array().map(|a| a.iter().filter_map(|v| v.as_str()).collect()).unwrap_or_default();
        let note = case["note"].as_str().unwrap_or("");
        let cands = idx.rows_by_name(query, project);
        let Some(&qrow) = cands.first() else {
            println!("MISS  {query}: not in index ({note})");
            continue;
        };
        let out = do_find(&idx, qrow, k, None, false);
        let rank = out.iter().position(|h| expect.contains(&idx.name(h.row)));
        match rank {
            Some(r) => {
                ok += 1;
                println!("ok    {query}: {} at rank {} ({note})", idx.name(out[r].row), r + 1);
            }
            None => {
                let top: Vec<&str> = out.iter().take(5).map(|h| idx.name(h.row)).collect();
                println!("MISS  {query}: expected {:?}, top5 {:?} ({note})", expect, top);
            }
        }
    }
    println!("{ok}/{} recovered in top {k}", data.len());
    Ok(())
}

fn cmd_stats(cli: &Cli) -> Result<()> {
    for kind in ["functions", "windows"] {
        match open_table(&cli.index_dir, kind, &cli.backend) {
            Ok(idx) => println!("{kind}_{}: {} rows, dim {}", cli.backend, idx.len(), idx.dim()),
            Err(_) => println!("{kind}_{}: (absent)", cli.backend),
        }
    }
    Ok(())
}

fn cmd_build_index(meta: &Path, vectors: &Path, out: &Path, dim: usize) -> Result<()> {
    use std::io::{BufRead, BufReader, Read};
    let mf = BufReader::new(std::fs::File::open(meta).with_context(|| format!("open {}", meta.display()))?);
    let mut vf = BufReader::with_capacity(1 << 20, std::fs::File::open(vectors)?);
    let mut rows = vec![];
    let mut buf = vec![0u8; dim * 4];
    for line in mf.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(&line)?;
        vf.read_exact(&mut buf).context("vector file shorter than metadata")?;
        let vector: Vec<f32> = buf.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        rows.push(index::BuildRow {
            name: v["name"].as_str().unwrap_or("").to_string(),
            project: v["project"].as_str().unwrap_or("").to_string(),
            unit: v["unit"].as_str().unwrap_or("").to_string(),
            src_path: v["src_path"].as_str().unwrap_or("").to_string(),
            n_insns: v["n_insns"].as_u64().unwrap_or(0) as u32,
            match_pct: v["match_pct"].as_f64().unwrap_or(-1.0) as f32,
            wstart: v["wstart"].as_i64().map(|w| w as i32).unwrap_or(-1),
            tokens: v["tokens"].as_str().unwrap_or("").to_string(),
            vector,
        });
    }
    let n = rows.len();
    index::write_index(out, dim, rows)?;
    println!("wrote {} rows to {}", n, out.display());
    Ok(())
}

/// Incremental table rebuild: reuse stored vectors for rows whose token doc
/// is unchanged (sync.py parity: tokens is the change-detection key); embed
/// only new/changed docs — natively for hashed, via the Python helper for
/// model backends. Rows from other projects are carried over untouched.
#[allow(clippy::too_many_arguments)]
fn sync_table(
    cli: &Cli,
    kind: &str,
    project: &str,
    desired: Vec<ingest::Desired>,
    full: bool,
    py: Option<&str>,
    embed_script: Option<&Path>,
) -> Result<()> {
    let path = table_path(&cli.index_dir, kind, &cli.backend);
    let existing = Index::open(&path).ok();
    // id -> (tokens, vector) for the ingested project; keep other projects verbatim
    let mut keep: Vec<index::BuildRow> = vec![];
    let mut prev: HashMap<String, (String, Vec<f32>)> = HashMap::new();
    if let Some(ex) = &existing {
        for row in 0..ex.len() {
            if ex.project(row) != project {
                keep.push(index::BuildRow {
                    name: ex.name(row).to_string(),
                    project: ex.project(row).to_string(),
                    unit: ex.unit(row).to_string(),
                    src_path: ex.src_path(row).to_string(),
                    n_insns: ex.rows()[row].n_insns,
                    match_pct: ex.rows()[row].match_pct,
                    wstart: ex.rows()[row].wstart,
                    tokens: ex.tokens(row).to_string(),
                    vector: ex.vector(row).to_vec(),
                });
            } else if !full {
                prev.insert(ex.id(row), (ex.tokens(row).to_string(), ex.vector(row).to_vec()));
            }
        }
    }
    // dedup desired by id, first occurrence wins (sync.py parity)
    let mut seen = std::collections::HashSet::new();
    let mut unchanged = 0usize;
    let mut to_embed: Vec<ingest::Desired> = vec![];
    for d in desired {
        let id = if d.wstart >= 0 {
            format!("{project}:{}:{}:w{}", d.unit, d.name, d.wstart)
        } else {
            format!("{project}:{}:{}", d.unit, d.name)
        };
        if !seen.insert(id.clone()) {
            continue;
        }
        match prev.get(&id) {
            Some((tokens, vector)) if *tokens == d.tokens => {
                unchanged += 1;
                keep.push(index::BuildRow {
                    name: d.name,
                    project: project.to_string(),
                    unit: d.unit,
                    src_path: d.src_path,
                    n_insns: d.n_insns,
                    match_pct: d.match_pct,
                    wstart: d.wstart,
                    tokens: d.tokens,
                    vector: vector.clone(),
                });
            }
            _ => to_embed.push(d),
        }
    }
    eprintln!(
        "{kind}: {} unchanged (vector reused), {} to embed, {} rows from other projects",
        unchanged,
        to_embed.len(),
        keep.iter().filter(|r| r.project != project).count()
    );
    if !to_embed.is_empty() {
        let vectors: Vec<Vec<f32>> = if cli.backend == "hashed" {
            let docs: Vec<&str> = to_embed.iter().map(|d| d.tokens.as_str()).collect();
            embed::embed_hashed(&docs, 512)
        } else {
            let Some(py) = py else {
                bail!(
                    "{} docs need embedding with backend '{}' — pass --py <python-with-dsearch> \
                     (or DSEARCH_PY) so ingest can run the model, or use --backend hashed",
                    to_embed.len(),
                    cli.backend
                );
            };
            let script = embed_script
                .map(|p| p.to_path_buf())
                .unwrap_or_else(|| {
                    std::env::current_exe()
                        .ok()
                        .and_then(|e| e.parent().map(|p| p.join("../../../tools/embed_docs.py")))
                        .unwrap_or_else(|| PathBuf::from("tools/embed_docs.py"))
                });
            use std::io::Write as _;
            use std::process::Stdio;
            let mut child = std::process::Command::new(py)
                .arg(&script)
                .arg("--backend")
                .arg(&cli.backend)
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::inherit())
                .spawn()
                .with_context(|| format!("spawn {py} {}", script.display()))?;
            {
                let mut stdin = child.stdin.take().unwrap();
                for d in &to_embed {
                    let line = serde_json::json!({"tokens": d.tokens});
                    writeln!(stdin, "{line}")?;
                }
            }
            let out = child.wait_with_output()?;
            if !out.status.success() {
                bail!("embed helper failed");
            }
            let dim = 512usize;
            if out.stdout.len() != to_embed.len() * dim * 4 {
                bail!(
                    "embed helper returned {} bytes, expected {}",
                    out.stdout.len(),
                    to_embed.len() * dim * 4
                );
            }
            out.stdout
                .chunks_exact(dim * 4)
                .map(|c| c.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect())
                .collect()
        };
        for (d, v) in to_embed.into_iter().zip(vectors) {
            keep.push(index::BuildRow {
                name: d.name,
                project: project.to_string(),
                unit: d.unit,
                src_path: d.src_path,
                n_insns: d.n_insns,
                match_pct: d.match_pct,
                wstart: d.wstart,
                tokens: d.tokens,
                vector: v,
            });
        }
    }
    let n = keep.len();
    index::write_index(&path, 512, keep)?;
    eprintln!("{kind}_{}: wrote {} rows to {}", cli.backend, n, path.display());
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn cmd_ingest_dtk(
    cli: &Cli,
    root: &Path,
    project: &str,
    version: &str,
    report: Option<&str>,
    min_insns: usize,
    full: bool,
    windows: bool,
    py: Option<&str>,
    embed_script: Option<&Path>,
) -> Result<()> {
    let rep = match report {
        Some(r) => ingest::load_report(r)?,
        None => HashMap::new(),
    };
    let (fns, wins) = ingest::collect_desired(root, version, min_insns, &rep, windows)?;
    eprintln!("{} functions{}", fns.len(), if windows { format!(", {} windows", wins.len()) } else { String::new() });
    sync_table(cli, "functions", project, fns, full, py, embed_script)?;
    if windows {
        sync_table(cli, "windows", project, wins, full, py, embed_script)?;
    }
    Ok(())
}

fn cmd_bench(cli: &Cli, iters: usize, windows: bool) -> Result<()> {
    use std::time::Instant;
    let idx = open_table(&cli.index_dir, "functions", &cli.backend)?;
    let n = idx.len();
    // deterministic pseudo-random query rows
    let mut qrows = vec![];
    let mut x = 0x9e3779b97f4a7c15u64;
    for _ in 0..iters {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        qrows.push((x % n as u64) as usize);
    }
    // warmup: touch all vector pages
    let t0 = Instant::now();
    let _ = scan::topk_scan(&idx, idx.vector(qrows[0]), 15, |_| true);
    let warm = t0.elapsed();
    let mut lat = vec![];
    for &q in &qrows {
        let t = Instant::now();
        let _ = do_find(&idx, q, 15, Some(99.5), false);
        lat.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pct = |p: f64| lat[((lat.len() as f64 * p) as usize).min(lat.len() - 1)];
    println!(
        "functions_{}: {} rows dim {} | first(cold-ish) {:.2}ms | find p50 {:.3}ms p90 {:.3}ms p99 {:.3}ms",
        cli.backend, n, idx.dim(), warm.as_secs_f64() * 1000.0, pct(0.5), pct(0.9), pct(0.99)
    );
    if windows {
        let widx = open_table(&cli.index_dir, "windows", &cli.backend)?;
        let wrows = widx.rows();
        let mut lat = vec![];
        for &q in qrows.iter().take(iters.min(30)) {
            let name = idx.name(q).to_string();
            let qws = widx.rows_by_name(&name, None);
            if qws.is_empty() {
                continue;
            }
            let t = Instant::now();
            let queries: Vec<&[f32]> = qws.iter().map(|&r| widx.vector(r)).collect();
            let _ = scan::multi_topk_scan(&widx, &queries, 80, |row| wrows[row].match_pct >= 99.5);
            lat.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        if !lat.is_empty() {
            lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let pct = |p: f64| lat[((lat.len() as f64 * p) as usize).min(lat.len() - 1)];
            println!(
                "windows_{}: {} rows | findw p50 {:.3}ms p90 {:.3}ms max {:.3}ms (n={})",
                cli.backend, widx.len(), pct(0.5), pct(0.9), lat[lat.len() - 1], lat.len()
            );
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match &cli.cmd {
        Cmd::Find { function, project, min_match, all, k, exclude_self_unit, ladder } => {
            cmd_find(&cli, function, project.as_deref(), *min_match, *all, *k, *exclude_self_unit, *ladder)
        }
        Cmd::Findw { function, project, min_match, all, k, exclude_self_unit, ladder } => {
            cmd_findw(&cli, function, project.as_deref(), *min_match, *all, *k, *exclude_self_unit, *ladder)
        }
        Cmd::Donors { function, project, min_match, k, wk, wbackend } => {
            cmd_donors(&cli, function, project.as_deref(), *min_match, *k, *wk, wbackend)
        }
        Cmd::Search { text, project, min_match, all, k } => {
            let t = text.join(" ");
            cmd_search(&cli, &t, project.as_deref(), *min_match, *all, *k)
        }
        Cmd::Sweep { project, max_match, min_insns, donor_min, min_sim, k } => {
            cmd_sweep(&cli, project.as_deref(), *max_match, *min_insns, *donor_min, *min_sim, *k)
        }
        Cmd::Stats => cmd_stats(&cli),
        Cmd::Eval { pairs, k } => cmd_eval(&cli, pairs, *k),
        Cmd::BuildIndex { meta, vectors, out, dim } => cmd_build_index(meta, vectors, out, *dim),
        Cmd::IngestDtk { root, project, version, report, min_insns, full, windows, py, embed_script } => {
            cmd_ingest_dtk(&cli, root, project, version, report.as_deref(), *min_insns, *full, *windows, py.as_deref(), embed_script.as_deref())
        }
        Cmd::Bench { iters, windows } => cmd_bench(&cli, *iters, *windows),
    }
}
