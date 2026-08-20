//! DSIDX: flat mmap'd index file. Zero-copy load; vectors contiguous f32.
//!
//! Layout (all little-endian, offsets from file start):
//!   [Header][projects dict][units dict][rows][name perm][string blob][pad][vectors]
//!
//! - vectors: count * dim f32, row-major, 64-byte aligned, L2-normalized at build.
//! - rows: fixed 40-byte records referencing the blob / dicts.
//! - name perm: row indices sorted by (name, wstart, row) for binary search.
//! - tokens stored in blob (query never touches those pages; used by ingest diff).

use anyhow::{bail, Context, Result};
use bytemuck::{Pod, Zeroable};
use memmap2::Mmap;
use std::fs::File;
use std::io::{BufWriter, Seek, Write};
use std::path::Path;

pub const MAGIC: &[u8; 8] = b"DSIDX001";

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct Header {
    pub magic: [u8; 8],
    pub dim: u32,
    pub flags: u32,
    pub count: u64,
    pub projects_off: u64,
    pub projects_count: u64,
    pub units_off: u64,
    pub units_count: u64,
    pub rows_off: u64,
    pub perm_off: u64,
    pub blob_off: u64,
    pub blob_len: u64,
    pub vec_off: u64,
    pub reserved: [u64; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct Row {
    pub name_off: u32,
    pub name_len: u32,
    pub tok_off: u32,
    pub tok_len: u32,
    pub unit_idx: u32,
    pub n_insns: u32,
    pub match_pct: f32,
    /// window start insn; -1 for whole-function rows
    pub wstart: i32,
    pub project_idx: u32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct StrRef {
    pub off: u32,
    pub len: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct UnitEntry {
    pub unit: StrRef,
    pub src_path: StrRef,
}

pub struct Index {
    _mmap: Mmap,
    pub header: Header,
    // raw pointers into the mmap, lifetimes tied to _mmap
    projects: &'static [StrRef],
    units: &'static [UnitEntry],
    rows: &'static [Row],
    perm: &'static [u32],
    blob: &'static [u8],
    vectors: &'static [f32],
}

impl Index {
    pub fn open(path: &Path) -> Result<Index> {
        let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
        let mmap = unsafe { Mmap::map(&file)? };
        if mmap.len() < std::mem::size_of::<Header>() {
            bail!("index file too small: {}", path.display());
        }
        let header: Header = *bytemuck::from_bytes(&mmap[..std::mem::size_of::<Header>()]);
        if &header.magic != MAGIC {
            bail!("bad magic in {}", path.display());
        }
        let count = header.count as usize;
        // SAFETY: mmap lives as long as Index (never exposed beyond &self).
        let buf: &'static [u8] = unsafe { std::mem::transmute::<&[u8], &'static [u8]>(&mmap[..]) };
        let sect = |off: u64, len_bytes: usize| -> Result<&'static [u8]> {
            let off = off as usize;
            if off + len_bytes > buf.len() {
                bail!("section out of bounds");
            }
            Ok(&buf[off..off + len_bytes])
        };
        let projects: &[StrRef] = bytemuck::cast_slice(sect(
            header.projects_off,
            header.projects_count as usize * std::mem::size_of::<StrRef>(),
        )?);
        let units: &[UnitEntry] = bytemuck::cast_slice(sect(
            header.units_off,
            header.units_count as usize * std::mem::size_of::<UnitEntry>(),
        )?);
        let rows: &[Row] =
            bytemuck::cast_slice(sect(header.rows_off, count * std::mem::size_of::<Row>())?);
        let perm: &[u32] = bytemuck::cast_slice(sect(header.perm_off, count * 4)?);
        let blob = sect(header.blob_off, header.blob_len as usize)?;
        let vec_bytes = count * header.dim as usize * 4;
        let vraw = sect(header.vec_off, vec_bytes)?;
        if header.vec_off % 64 != 0 {
            bail!("vectors not 64-byte aligned");
        }
        let vectors: &[f32] = bytemuck::cast_slice(vraw);
        Ok(Index {
            header,
            projects,
            units,
            rows,
            perm,
            blob,
            vectors,
            _mmap: mmap,
        })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.header.count as usize
    }
    #[inline]
    pub fn dim(&self) -> usize {
        self.header.dim as usize
    }
    #[inline]
    pub fn rows(&self) -> &[Row] {
        self.rows
    }
    #[inline]
    pub fn vectors(&self) -> &[f32] {
        self.vectors
    }
    #[inline]
    pub fn vector(&self, row: usize) -> &[f32] {
        let d = self.dim();
        &self.vectors[row * d..(row + 1) * d]
    }
    #[inline]
    fn s(&self, off: u32, len: u32) -> &str {
        std::str::from_utf8(&self.blob[off as usize..(off + len) as usize]).unwrap_or("")
    }
    #[inline]
    pub fn name(&self, row: usize) -> &str {
        let r = &self.rows[row];
        self.s(r.name_off, r.name_len)
    }
    #[inline]
    pub fn tokens(&self, row: usize) -> &str {
        let r = &self.rows[row];
        self.s(r.tok_off, r.tok_len)
    }
    #[inline]
    pub fn project(&self, row: usize) -> &str {
        let p = self.projects[self.rows[row].project_idx as usize];
        self.s(p.off, p.len)
    }
    #[inline]
    pub fn unit(&self, row: usize) -> &str {
        let u = self.units[self.rows[row].unit_idx as usize];
        self.s(u.unit.off, u.unit.len)
    }
    #[inline]
    pub fn src_path(&self, row: usize) -> &str {
        let u = self.units[self.rows[row].unit_idx as usize];
        self.s(u.src_path.off, u.src_path.len)
    }
    pub fn project_idx_of(&self, name: &str) -> Option<u32> {
        (0..self.projects.len()).find(|&i| {
            let p = self.projects[i];
            self.s(p.off, p.len) == name
        }).map(|i| i as u32)
    }
    pub fn id(&self, row: usize) -> String {
        let r = &self.rows[row];
        if r.wstart >= 0 {
            format!(
                "{}:{}:{}:w{}",
                self.project(row),
                self.unit(row),
                self.name(row),
                r.wstart
            )
        } else {
            format!("{}:{}:{}", self.project(row), self.unit(row), self.name(row))
        }
    }

    /// All rows with exactly this name (optionally restricted to project),
    /// in (wstart, row) order. Binary search over the name-sorted perm.
    pub fn rows_by_name(&self, name: &str, project: Option<&str>) -> Vec<usize> {
        let perm = self.perm;
        let mut lo = 0usize;
        let mut hi = perm.len();
        while lo < hi {
            let mid = (lo + hi) / 2;
            if self.name(perm[mid] as usize) < name {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        let mut out = vec![];
        while lo < perm.len() {
            let row = perm[lo] as usize;
            if self.name(row) != name {
                break;
            }
            if project.is_none_or(|p| self.project(row) == p) {
                out.push(row);
            }
            lo += 1;
        }
        out
    }
}

/// One row of input to the builder.
pub struct BuildRow {
    pub name: String,
    pub project: String,
    pub unit: String,
    pub src_path: String,
    pub n_insns: u32,
    pub match_pct: f32,
    pub wstart: i32,
    pub tokens: String,
    pub vector: Vec<f32>,
}

pub fn write_index(path: &Path, dim: usize, mut rows_in: Vec<BuildRow>) -> Result<()> {
    use std::collections::HashMap;
    // stable order: sort by (project, unit, name, wstart) for locality/determinism
    rows_in.sort_by(|a, b| {
        (&a.project, &a.unit, &a.name, a.wstart).cmp(&(&b.project, &b.unit, &b.name, b.wstart))
    });

    let mut blob: Vec<u8> = vec![];
    let mut intern = |blob: &mut Vec<u8>, s: &str| -> StrRef {
        let off = blob.len() as u32;
        blob.extend_from_slice(s.as_bytes());
        StrRef { off, len: s.len() as u32 }
    };

    let mut proj_map: HashMap<String, u32> = HashMap::new();
    let mut projects: Vec<StrRef> = vec![];
    let mut unit_map: HashMap<(String, String), u32> = HashMap::new();
    let mut units: Vec<UnitEntry> = vec![];
    let mut rows: Vec<Row> = Vec::with_capacity(rows_in.len());

    for r in &rows_in {
        let pidx = *proj_map.entry(r.project.clone()).or_insert_with(|| {
            let sr = intern(&mut blob, &r.project);
            projects.push(sr);
            (projects.len() - 1) as u32
        });
        let ukey = (r.project.clone(), r.unit.clone());
        let uidx = *unit_map.entry(ukey).or_insert_with(|| {
            let u = intern(&mut blob, &r.unit);
            let s = intern(&mut blob, &r.src_path);
            units.push(UnitEntry { unit: u, src_path: s });
            (units.len() - 1) as u32
        });
        let name = intern(&mut blob, &r.name);
        let tok = intern(&mut blob, &r.tokens);
        rows.push(Row {
            name_off: name.off,
            name_len: name.len,
            tok_off: tok.off,
            tok_len: tok.len,
            unit_idx: uidx,
            n_insns: r.n_insns,
            match_pct: r.match_pct,
            wstart: r.wstart,
            project_idx: pidx,
            _pad: 0,
        });
        if blob.len() > u32::MAX as usize - (1 << 20) {
            bail!("string blob exceeds u32 offset space");
        }
    }

    // name-sorted permutation
    let mut perm: Vec<u32> = (0..rows.len() as u32).collect();
    perm.sort_by(|&a, &b| {
        let (ra, rb) = (&rows_in[a as usize], &rows_in[b as usize]);
        (&ra.name, ra.wstart, a).cmp(&(&rb.name, rb.wstart, b))
    });

    let hsize = std::mem::size_of::<Header>() as u64;
    let projects_off = hsize;
    let units_off = projects_off + (projects.len() * std::mem::size_of::<StrRef>()) as u64;
    let rows_off = units_off + (units.len() * std::mem::size_of::<UnitEntry>()) as u64;
    let perm_off = rows_off + (rows.len() * std::mem::size_of::<Row>()) as u64;
    let blob_off = perm_off + (perm.len() * 4) as u64;
    let vec_off = (blob_off + blob.len() as u64).div_ceil(64) * 64;

    let header = Header {
        magic: *MAGIC,
        dim: dim as u32,
        flags: 0,
        count: rows.len() as u64,
        projects_off,
        projects_count: projects.len() as u64,
        units_off,
        units_count: units.len() as u64,
        rows_off,
        perm_off,
        blob_off,
        blob_len: blob.len() as u64,
        vec_off,
        reserved: [0; 3],
    };

    let tmp = path.with_extension("dsi.tmp");
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut w = BufWriter::new(File::create(&tmp)?);
    w.write_all(bytemuck::bytes_of(&header))?;
    w.write_all(bytemuck::cast_slice(&projects))?;
    w.write_all(bytemuck::cast_slice(&units))?;
    w.write_all(bytemuck::cast_slice(&rows))?;
    w.write_all(bytemuck::cast_slice(&perm))?;
    w.write_all(&blob)?;
    let pos = w.stream_position()?;
    let pad = vec_off - pos;
    w.write_all(&vec![0u8; pad as usize])?;
    // L2-renormalize on write so query-time dot == cosine exactly
    for (i, r) in rows_in.iter().enumerate() {
        if r.vector.len() != dim {
            bail!("row {i} has dim {} != {dim}", r.vector.len());
        }
        let norm = r.vector.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>().sqrt();
        if norm > 0.0 && (norm - 1.0).abs() > 1e-3 {
            let v: Vec<f32> = r.vector.iter().map(|v| (*v as f64 / norm) as f32).collect();
            w.write_all(bytemuck::cast_slice(&v))?;
        } else {
            w.write_all(bytemuck::cast_slice(&r.vector))?;
        }
    }
    w.flush()?;
    drop(w);
    std::fs::rename(&tmp, path)?;
    Ok(())
}
