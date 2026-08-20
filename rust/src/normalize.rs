//! Bit-exact port of dsearch/normalize.py: disassembled functions -> token streams.
//!
//! The token stream preserves structural signal (mnemonic skeleton, operand
//! shapes, branch direction) and discards register numbers, addresses,
//! symbol names, literal values.

use regex::Regex;
use std::sync::LazyLock;

#[derive(Debug, Clone)]
pub struct Insn {
    pub addr: u64,
    pub mnemonic: String,
    pub operands: String,
    pub reloc: Option<String>,
}

#[derive(Debug, Clone)]
pub struct Function {
    pub name: String,
    pub unit: String,
    pub insns: Vec<Insn>,
}

static BRANCH_MNEMONICS: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^b(l|c|ne|eq|lt|gt|le|ge|so|ns|dnz|dz|ctr|lr)?[+-]?$|^b(ne|eq|lt|gt|le|ge)(lr|ctr)?[+-]?$")
        .unwrap()
});
static TARGET: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^(0x)?[0-9a-f]+$").unwrap());
static DISP: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^(-?(?:0x)?[0-9a-fA-F]+)\((r\d{1,2}|sp|rtoc)\)$").unwrap());
static GPR: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^(r\d{1,2}|sp|rtoc)$").unwrap());
static FPR: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^f\d{1,2}$").unwrap());
static CRF: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^cr\d$").unwrap());
static QR: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^qr\d$").unwrap());
// Python: re.match(r"(?:0x)?([0-9a-f]+)\b", p) — match at start, \b after hex run.
static TGT_PREFIX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^(?:0x)?([0-9a-f]+)\b").unwrap());

fn operand_shape(op: &str) -> &'static str {
    let op = op.trim();
    if op.is_empty() {
        return "";
    }
    if DISP.is_match(op) {
        return "#(r)";
    }
    if GPR.is_match(op) {
        return "r";
    }
    if FPR.is_match(op) {
        return "f";
    }
    if CRF.is_match(op) {
        return "cr";
    }
    if QR.is_match(op) {
        return "q";
    }
    if TARGET.is_match(op) {
        return "#";
    }
    "@"
}

pub fn insn_tokens(insn: &Insn) -> String {
    let mnem = insn.mnemonic.as_str();
    let ops = insn.operands.as_str();

    if mnem == "b" || (mnem.starts_with('b') && BRANCH_MNEMONICS.is_match(mnem)) {
        if matches!(mnem, "bl" | "blr" | "bctr" | "bctrl" | "blrl") {
            return mnem.to_string();
        }
        let parts: Vec<&str> = ops.split(',').map(str::trim).collect();
        let mut direction = "";
        for p in &parts {
            if let Some(m) = TGT_PREFIX.captures(p) {
                // Python guards p.split()[0] behind `m and ...`; p non-empty when m matched.
                let first_ws = p.split_whitespace().next().unwrap_or("");
                if TARGET.is_match(first_ws) {
                    if let Ok(tgt) = u64::from_str_radix(&m[1], 16) {
                        direction = if tgt <= insn.addr { "back" } else { "fwd" };
                        break;
                    }
                }
            }
        }
        return if direction.is_empty() {
            mnem.to_string()
        } else {
            format!("{mnem}({direction})")
        };
    }

    if ops.is_empty() {
        return mnem.to_string();
    }
    let shapes: Vec<&str> = ops
        .split(',')
        .map(operand_shape)
        .filter(|s| !s.is_empty())
        .collect();
    let mut tok = format!("{mnem}({})", shapes.join(","));
    if let Some(reloc) = &insn.reloc {
        tok.push('[');
        tok.push_str(reloc);
        tok.push(']');
    }
    tok
}

pub fn function_tokens(fn_: &Function) -> Vec<String> {
    fn_.insns.iter().map(insn_tokens).collect()
}

pub fn token_text(fn_: &Function) -> String {
    let toks = function_tokens(fn_);
    format!("ppc {}\n{}", toks.len(), toks.join(" "))
}

/// Sliding windows over the instruction stream: (start_insn, doc) pairs.
pub fn window_texts(fn_: &Function, size: usize, stride: usize) -> Vec<(usize, String)> {
    let toks = function_tokens(fn_);
    if toks.is_empty() {
        return vec![];
    }
    if toks.len() <= size {
        return vec![(0, format!("ppc {}\n{}", toks.len(), toks.join(" ")))];
    }
    let mut out = vec![];
    let last = toks.len() - size;
    let mut start = 0;
    while start <= last {
        let w = &toks[start..start + size];
        out.push((start, format!("ppc {}\n{}", w.len(), w.join(" "))));
        start += stride;
    }
    if last % stride != 0 {
        let w = &toks[last..];
        out.push((last, format!("ppc {}\n{}", w.len(), w.join(" "))));
    }
    out
}
