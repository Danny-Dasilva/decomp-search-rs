# decomp-search

[![License: MIT/Apache-2.0](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE-MIT)
[![Rust](https://img.shields.io/badge/rust-2021-orange.svg)](rust/Cargo.toml)

**Find the matched twin of any unmatched function in 10 milliseconds.**

decomp-search is a local similarity-search engine for matching
decompilation projects (dtk-based GameCube/Wii and friends). Ingest a
project's target assembly, then ask: *"this function won't match; show me
the most similar functions that already DO match, so I can steal their
source recipe."* Donor transplanting is the highest-yield technique in
agent-driven decomp campaigns, and this tool makes the donor lookup
effectively free.

The corpus is embedded PowerPC instruction streams. The query is a whole
function, or any 32-instruction window of one.

```console
$ dsearch find getDoorGlobalPosition__Q34Game4Cave7MapNodeFi --all -k 5
similar to getDoorGlobalPosition__Q34Game4Cave7MapNodeFi (151 insns)
   sim  match%  insns  function                                     unit
 0.977   100.0    115  createWayPoints__Q34Game10ItemBridge4ItemFv  obj/plugProjectKandoU/itemBridge.o
 0.974   100.0    131  moveNoTarget__Q34Game4Baby3ObjFv             obj/plugProjectNishimuraU/Baby.o
 0.973   100.0    314  createCrashEnemy__Q34Game10DangoMushi3ObjFv  obj/plugProjectNishimuraU/DangoMushi.o
 0.973   100.0    154  onSetPosition__Q34Game8ItemWeed4ItemFv       obj/plugProjectKandoU/itemWeed.o
 0.973   100.0    135  doSimulation__Q34Game13PanModokiBase3ObjFf   obj/plugProjectMorimuraU/panModoki.o

real    0m0.011s
```

## Why decomp-search?

- **Fast.** A cold process answers whole-function queries in ~10 ms and
  window (construct) queries in 30-90 ms, which is 80-141x faster than the
  previous Python/LanceDB implementation measured end-to-end (table below).
  There is no server or warmup step; mmap and the OS page cache do the work.
- **Exact.** Every query is a full SIMD cosine scan over the corpus, with
  no ANN index or recall knobs. Results are rank-identical to the reference
  implementation (verified over 160 sampled queries).
- **Built for agents.** `--json` everywhere with a stable schema, a
  one-call `donors` command that returns whole-function and construct twins
  together, a built-in `--ladder` that relaxes the match-percentage filter
  (99.5, 99, 95, 90) so the caller never has to retry, `same_unit` flags on
  every hit, and output that never truncates a symbol name.
- **Instant ingest.** A full 16k-function project (objdump, normalize,
  embed, index) takes ~0.5 s on the hashed backend. Re-ingest is
  incremental: unchanged functions reuse their stored vectors.

## Performance

End-to-end CLI, full process cost per call, same index and same results as
the Python/LanceDB reference (44.6k functions, 209k windows, dim 512):

| command | python (LanceDB) | rust | speedup |
|---|---|---|---|
| `find` | 853 ms | **9.6 ms** | 89x |
| `findw`, 5-window fn | 2.4 s | **30 ms** | 80x |
| `findw`, 10-window fn | 3.8 s | **40 ms** | 97x |
| `findw`, 42-window fn (p99) | 12.2 s | **86 ms** | 141x |
| `donors` (find + findw combined) | 4.5 s | **47 ms** | 96x |
| `ingest-dtk`, full project | minutes | **0.56 s** | n/a |
| `sweep` (rank all open fns by best donor) | up to 1 h | **0.2 s** | n/a |

Warm-process floor (embedding the library in a server): `find` p50 1.5 ms,
`findw` p50 4.9 ms. Methodology and verification in
[`bench/RESULTS.md`](bench/RESULTS.md); reproduce with
`bench/run_bench.py` and `dsearch bench`.

Where the old stack spent its time: 89% of every call was Python imports
(445 ms of it an unused REST client), searches were unindexed flat scans
through Arrow with a 10-50x overfetch, and window search ran one full scan
per query window. The Rust engine mmaps vectors zero-copy, pushes filters
into the scan, and answers all query windows in one memory pass.

## Install

```sh
# from git, one line (binary lands in ~/.cargo/bin):
cargo install --git https://github.com/Danny-Dasilva/decomp-search-rs dsearch

# or from a clone:
cargo build --release          # -> target/release/dsearch
```

Then get an index (either path):

```sh
# A) build one from your project's target objects (hashed backend, ~1 s):
dsearch --backend hashed ingest-dtk <repo_root> --project <name> \
    --version <VER> --windows \
    --report 'https://decomp.dev/<org>/<repo>/<VER>.json?mode=report'

# B) migrate an existing LanceDB index built by the Python implementation:
python tools/export_index.py --db data/index.lancedb --out /tmp/export
dsearch build-index --meta /tmp/export/functions_local.meta.jsonl \
    --vectors /tmp/export/functions_local.vec.f32 \
    --out data/dsi/functions_local.dsi        # repeat per table
```

Indexes live in `data/dsi/` (`--index-dir` / `DSEARCH_INDEX_DIR`). Sanity
check: `dsearch --backend hashed stats`.

## Usage

```sh
# the twin-finder: matched donors for an unmatched function
dsearch find <fn> --min-match 99.5 -k 10

# one-call donor lookup for agents (whole-fn + window twins, JSON, ladder):
dsearch donors <fn> --project <name> -k 10 --wk 6

# construct-level: match ANY 32-insn window of <fn> against the corpus
dsearch --backend hashed findw <fn> -k 10

# whole neighborhood, no match% filter / cross-TU only
dsearch find <fn> --all
dsearch find <fn> --exclude-self-unit

# rank every open function by its best matched donor ("what's solvable?")
dsearch --backend local sweep --project <name> --min-sim 0.85 > solvable.json

# machine-readable anything
dsearch --json find <fn> -k 10
```

JSON hits carry `name, unit, src_path, match_pct, n_insns, sim, same_unit`
(plus `q_at`/`t_at` window offsets). `--ladder` relaxes `--min-match` down
the 99.5/99/95/90 ladder on zero hits and reports `min_match_used`. Agents
used to hand-roll that retry loop.

## How it works

- **Normalize**: each function's disassembly becomes a token stream that
  keeps structural signal (mnemonic skeleton, operand shapes, branch
  direction: `b(back)` is a backedge) and drops what varies between twins
  (registers, addresses, symbol names, literals).
- **Embed**: `hashed` = BLAKE2b feature-hashed 1/2/3-grams, sublinear TF,
  512-dim, fully deterministic, embedded natively in Rust.
  `local`/`voyage` = [voyage-4-nano](https://huggingface.co/voyageai/voyage-4-nano)
  document embeddings. The model runs only in the Python sidecar at ingest
  time; `find`/`findw` search with stored vectors.
- **Index** (`.dsi`): one mmap'd file per table: header, project/unit
  dictionaries, fixed 40-byte rows, a name-sorted permutation for
  binary-search lookup, string blob, then 64-byte-aligned L2-normalized
  f32 vectors. Zero-copy load; queries never touch token pages.
- **Scan**: unit vectors make cosine a dot product. Chunked scans
  autovectorize (AVX-512 where available) with per-thread top-k heaps via
  rayon. `findw` computes every query window against the corpus in a
  single pass; the scan is memory-bandwidth bound, so extra windows are
  nearly free.

## Using it from Claude / agent harnesses

- [`CLAUDE.md`](CLAUDE.md) gives coding agents the project map,
  invariants, and verification commands.
- [`.claude/skills/decomp-search/SKILL.md`](.claude/skills/decomp-search/SKILL.md)
  is the agent-facing usage skill (donor-transplant workflow, find vs
  findw vs donors). Symlink it into `~/.claude/skills/decomp-search` to
  enable it in Claude Code.
- `--json` output is schema-compatible with the legacy `dsearch.jsonq`
  module, so existing harness wrappers can swap the subprocess target and
  keep their parsers.

## When NOT to use it

- You want semantic free-text search over the corpus: `search` is
  hashed-backend-only in Rust; model-backend text queries still go through
  the Python CLI.
- Your corpus is millions of functions: the exact-scan design is sized for
  the ~10^5 range, where it beats an ANN index on speed and is exact.
  Past that you'd want quantization or ANN, and neither is here yet.
- You need the struct-layout tooling (`types-*`): that subsystem is
  unchanged Python (`dsearch/typeidx.py`).

## License

MIT or Apache-2.0, at your option.
