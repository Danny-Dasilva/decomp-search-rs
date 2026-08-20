//! dtk-project ingest: parse target objects via powerpc-eabi-objdump,
//! normalize, hashed-embed, and (re)build the .dsi index.
//!
//! Port of dsearch/ingest_dtk.py. The `hashed` backend is fully native.
//! For `local`/`voyage` (model embeddings) the ingest computes token docs
//! and reuses stored vectors for unchanged tokens; NEW/changed docs cannot
//! be embedded natively — those are reported so the Python embedder can fill
//! them in (or run ingest with --backend hashed).

use crate::normalize::{token_text, window_texts, Function, Insn};
use anyhow::{bail, Context, Result};
use regex::Regex;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::LazyLock;

static SYM: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^([0-9a-f]+) <(.+)>:").unwrap());
static INSN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^\s*([0-9a-f]+):\s+(?:[0-9a-f]{2} ){4}\s*([a-z0-9_.+-]+)\s*(.*?)\s*$").unwrap()
});
static RELOC: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\s*[0-9a-f]+:\s+(R_PPC_\S+)\s+(\S+)").unwrap());

pub fn find_objdump(project_root: &Path) -> String {
    if let Ok(env) = std::env::var("DSEARCH_OBJDUMP") {
        if !env.is_empty() {
            return env;
        }
    }
    let cand = project_root.join("build/binutils/powerpc-eabi-objdump");
    if cand.exists() {
        return cand.to_string_lossy().into_owned();
    }
    if let Some(parent) = project_root.parent() {
        if let Ok(sibs) = std::fs::read_dir(parent) {
            for sib in sibs.flatten() {
                let c = sib.path().join("build/binutils/powerpc-eabi-objdump");
                if c.exists() {
                    return c.to_string_lossy().into_owned();
                }
            }
        }
    }
    "powerpc-eabi-objdump".to_string()
}

/// -> {fn_name: fuzzy_match_percent}
pub fn load_report(source: &str) -> Result<HashMap<String, f32>> {
    let text = if source.starts_with("http") {
        let out = Command::new("curl")
            .args(["-fsSL", source])
            .output()
            .context("curl for report URL")?;
        if !out.status.success() {
            bail!("failed to fetch report from {source}");
        }
        String::from_utf8(out.stdout)?
    } else {
        std::fs::read_to_string(source).with_context(|| format!("read report {source}"))?
    };
    let rep: serde_json::Value = serde_json::from_str(&text)?;
    let mut out = HashMap::new();
    if let Some(units) = rep["units"].as_array() {
        for u in units {
            if let Some(fns) = u["functions"].as_array() {
                for f in fns {
                    if let Some(name) = f["name"].as_str() {
                        out.insert(
                            name.to_string(),
                            f["fuzzy_match_percent"].as_f64().unwrap_or(0.0) as f32,
                        );
                    }
                }
            }
        }
    }
    Ok(out)
}

pub fn parse_object(objdump: &str, obj_path: &Path, unit: &str) -> Result<Vec<Function>> {
    let out = Command::new(objdump)
        .arg("-dr")
        .arg(obj_path)
        .output()
        .with_context(|| format!("run {objdump}"))?;
    if !out.status.success() {
        return Ok(vec![]); // parity: CalledProcessError -> skip unit
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let mut fns: Vec<Function> = vec![];
    let mut cur: Option<Function> = None;
    for line in text.lines() {
        if let Some(m) = SYM.captures(line) {
            if let Some(f) = cur.take() {
                if !f.insns.is_empty() {
                    fns.push(f);
                }
            }
            cur = Some(Function {
                name: m[2].to_string(),
                unit: unit.to_string(),
                insns: vec![],
            });
            continue;
        }
        let Some(f) = cur.as_mut() else { continue };
        if let Some(mr) = RELOC.captures(line) {
            if let Some(last) = f.insns.last_mut() {
                last.reloc = Some(mr[1].strip_prefix("R_PPC_").unwrap_or(&mr[1]).to_string());
            }
            continue;
        }
        if let Some(mi) = INSN.captures(line) {
            f.insns.push(Insn {
                addr: u64::from_str_radix(&mi[1], 16).unwrap_or(0),
                mnemonic: mi[2].to_string(),
                operands: mi[3].to_string(),
                reloc: None,
            });
        }
    }
    if let Some(f) = cur.take() {
        if !f.insns.is_empty() {
            fns.push(f);
        }
    }
    Ok(fns)
}

pub fn iter_units(project_root: &Path, version: &str) -> Result<Vec<(PathBuf, String)>> {
    let build_root = project_root.join("build").join(version);
    if !build_root.is_dir() {
        bail!("no build dir at {}", build_root.display());
    }
    let mut out = vec![];
    fn walk(dir: &Path, build_root: &Path, out: &mut Vec<(PathBuf, String)>) {
        let Ok(entries) = std::fs::read_dir(dir) else { return };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                walk(&p, build_root, out);
            } else if p.extension().is_some_and(|x| x == "o") {
                let rel = p.strip_prefix(build_root).unwrap().to_path_buf();
                if rel.components().any(|c| c.as_os_str() == "obj") {
                    out.push((p.clone(), rel.to_string_lossy().into_owned()));
                }
            }
        }
    }
    walk(&build_root, &build_root, &mut out);
    out.sort();
    if out.is_empty() {
        bail!("no target objects under {}", build_root.display());
    }
    Ok(out)
}

pub fn find_source_file(project_root: &Path, unit: &str) -> Option<String> {
    let stem = unit.strip_suffix(".o").unwrap_or(unit);
    for base in ["src", "source", ""] {
        for ext in [".c", ".cpp"] {
            let p = if base.is_empty() {
                project_root.join(format!("{stem}{ext}"))
            } else {
                project_root.join(base).join(format!("{stem}{ext}"))
            };
            if p.exists() {
                return Some(p.strip_prefix(project_root).unwrap().to_string_lossy().into_owned());
            }
        }
    }
    None
}

/// A desired record before embedding.
pub struct Desired {
    pub name: String,
    pub unit: String,
    pub src_path: String,
    pub n_insns: u32,
    pub match_pct: f32,
    pub wstart: i32,
    pub tokens: String,
}

/// Enumerate desired rows (functions and, optionally, windows) for a project.
pub fn collect_desired(
    root: &Path,
    version: &str,
    min_insns: usize,
    report: &HashMap<String, f32>,
    windows: bool,
) -> Result<(Vec<Desired>, Vec<Desired>)> {
    use rayon::prelude::*;
    let objdump = find_objdump(root);
    let units = iter_units(root, version)?;
    eprintln!("{} units", units.len());
    let results: Vec<(Vec<Desired>, Vec<Desired>)> = units
        .par_iter()
        .map(|(obj, unit)| {
            let src = find_source_file(root, unit).unwrap_or_default();
            let fns = parse_object(&objdump, obj, unit).unwrap_or_default();
            let mut d = vec![];
            let mut dw = vec![];
            for f in fns {
                if f.insns.len() < min_insns {
                    continue;
                }
                let pct = report.get(&f.name).copied().unwrap_or(-1.0);
                d.push(Desired {
                    name: f.name.clone(),
                    unit: unit.clone(),
                    src_path: src.clone(),
                    n_insns: f.insns.len() as u32,
                    match_pct: pct,
                    wstart: -1,
                    tokens: token_text(&f),
                });
                if windows {
                    for (start, doc) in window_texts(&f, 32, 16) {
                        let n = doc.lines().last().unwrap_or("").split_whitespace().count();
                        dw.push(Desired {
                            name: f.name.clone(),
                            unit: unit.clone(),
                            src_path: src.clone(),
                            n_insns: n as u32,
                            match_pct: pct,
                            wstart: start as i32,
                            tokens: doc,
                        });
                    }
                }
            }
            (d, dw)
        })
        .collect();
    let mut fns = vec![];
    let mut wins = vec![];
    for (d, dw) in results {
        fns.extend(d);
        wins.extend(dw);
    }
    Ok((fns, wins))
}
