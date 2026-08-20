# decomp-search

Fast local similarity search over decompilation-project functions. Rust
engine (`rust/`) over mmap'd `.dsi` flat indexes; Python package
(`dsearch/`) survives as the model-embedding sidecar + type-layout index.

## Commands

```sh
cargo build --release                  # binary: target/release/dsearch
cargo install --path rust              # or install to ~/.cargo/bin
dsearch --backend local stats          # sanity: row counts per table
dsearch --backend local eval --pairs <pairs.json>   # recall on known twin pairs
dsearch bench --iters 200 --windows    # internal latency percentiles
python bench/run_bench.py --help       # head-to-head vs Python (bench/RESULTS.md)
```

There is no test suite yet; `eval` + `bench` + the parity checks below are
the verification story. Run them after touching `normalize.rs`, `embed.rs`,
`scan.rs`, or `index.rs`.

## Architecture (read in this order)

- `rust/src/index.rs` — the `.dsi` format: header, project/unit dicts,
  40-byte row records, name-sorted permutation (binary-search lookup),
  string blob, 64-byte-aligned L2-normalized f32 vectors. Zero-copy mmap.
- `rust/src/scan.rs` — exact brute-force cosine top-k (vectors are unit
  norm, so sim = dot). `multi_topk_scan` answers ALL query windows in one
  pass over the corpus — that's why findw is fast; never turn it back into
  per-window scans.
- `rust/src/normalize.rs` + `rust/src/embed.rs` — **bit-exact ports** of
  `dsearch/normalize.py` and `dsearch/embed.py::embed_hashed` (BLAKE2b
  feature-hashed 1/2/3-grams, 1+ln(c) TF, L2 norm, dim 512). Any change
  here breaks compatibility with stored vectors; if you must change them,
  bump the index and re-embed everything.
- `rust/src/ingest.rs` — objdump parse + incremental sync (token text is
  the change key; unchanged rows reuse stored vectors).
- `rust/src/main.rs` — CLI. `--json` output MUST stay schema-compatible
  with `dsearch/jsonq.py` (`{"query":…, "hits":[{name, unit, src_path,
  match_pct, n_insns, sim, …}]}`) — agent harnesses parse it.

## Invariants

- Search results must stay rank-identical to the Python/LanceDB
  implementation (ties may permute). Check with the parity method in
  `bench/RESULTS.md` before changing scan/filter semantics.
- `find` replicates the legacy overfetch (`k*10+50` then client-side self/
  unit filtering) deliberately — same results, not an accident to "fix".
- Vectors in `.dsi` are unit-normalized at build time; query code assumes
  dot == cosine.

## Data & deployment

- Index files: `data/dsi/*.dsi` (`--index-dir` / `DSEARCH_INDEX_DIR`);
  `{functions|windows}_{hashed|local|voyage}.dsi`. Never committed.
- Migration from a LanceDB index: `tools/export_index.py` →
  `dsearch build-index` per table.
- Typical deployment: a `dsearch` shim on PATH exporting
  `DSEARCH_INDEX_DIR`, indexes refreshed after each ingest (or nightly:
  Python model-embedding ingest → export → `build-index` → Rust `sweep`).

## Agent-facing skill

`.claude/skills/decomp-search/SKILL.md` is the canonical usage doc for
agents (donor-transplant workflow, when to use find vs findw vs donors).
Symlink it into `~/.claude/skills/decomp-search` to enable it. Keep it in
sync with CLI changes.

## Gotchas

- The hashed backend embeds natively; `local`/`voyage` need the Python
  sidecar (`tools/embed_docs.py`, `--py`/`DSEARCH_PY`) for NEW docs only.
- Duplicate function names exist across projects/units (REL `fn_1_*`);
  lookups are deterministic (name-sorted) but pass `--project` to pin one.
- `.cargo/config.toml` sets `target-cpu=native` — binaries are not
  portable across CPU generations; build on the machine that serves.
- `search` (freeform text) is hashed-only in Rust by design.
