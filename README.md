# decomp-search

Local similarity search over decompilation-project functions. Ingest a
project's target assembly (and match metadata), embed every function, and
query for structurally similar functions — e.g. "given this unmatched
function, show me the most similar **matched** functions so I can steal
their source recipe."

**The search engine is a Rust binary** (`rust/`) over a flat mmap'd index
(`.dsi` files): zero-copy load, exact brute-force cosine over contiguous
f32 vectors with a batched multi-query scan for window search. No server,
no daemon — a cold process answers in ~10ms. The Python package (`dsearch/`)
remains for the model-embedding sidecar (`local`/`voyage` backends), the
type-layout index, and legacy compatibility.

## Performance

End-to-end CLI (full process cost, as agents pay it), same index, same
results (rank-parity verified over 160 sampled queries; ties may permute):

| command | python (LanceDB) | rust | speedup |
|---|---|---|---|
| `find` (44.6k functions) | 853 ms | **9.6 ms** | 89x |
| `findw`, 5-window fn (209k windows) | 2.4 s | **30 ms** | 80x |
| `findw`, 10-window fn | 3.8 s | **40 ms** | 97x |
| `findw`, 42-window fn (p99) | 12.2 s | **86 ms** | 141x |
| `donors` (find + findw combined) | 4.5 s | **47 ms** | 96x |
| `ingest-dtk` melee, full (16.3k fns + 56k windows, hashed) | minutes | **0.56 s** | — |

Why: the Python CLI spent 89% of its time importing (757 ms of which 445 ms
is an unused REST client), searched with no ANN index (flat scan through
Arrow), overfetched 10–50x for client-side filtering, and ran one scan *per
query window*. The Rust engine mmaps vectors, pushes filters into the scan,
and answers all query windows in one memory pass. In the melee agent logs,
`findw` was called 18,913 times at median 7.5 s — ~50 h of cumulative agent
wait this replaces with ~0.2 h.

## Setup

```sh
cd rust && cargo build --release        # binary: rust/target/release/dsearch
# optional: cargo install --path rust   # puts `dsearch` on PATH
```

Index files live in `data/dsi/` (`--index-dir` / `DSEARCH_INDEX_DIR`).
Backend selection: `--backend hashed|local|voyage` (default `local`,
`DSEARCH_BACKEND` overrides) — each backend is its own index file, so they
coexist for A/B comparison.

- `hashed`: deterministic feature-hashed n-grams over a normalized
  instruction-token stream (BLAKE2b, 512-dim). Embedded natively in Rust —
  bit-exact with the Python implementation. No API, no model.
- `local` (default): [voyage-4-nano](https://huggingface.co/voyageai/voyage-4-nano)
  via sentence-transformers, MRL-truncated to 512 dims. Vectors are produced
  by the Python sidecar at ingest time; queries never run the model.
- `voyage`: same model via the Voyage API (`VOYAGE_API_KEY`).

Normalization keeps the structural signal (mnemonic skeleton, operand
shapes, **branch direction** — `b(back)` is a backedge) and discards what
varies between twins (register numbers, addresses, symbol names).

## Migrating an existing LanceDB index

```sh
.venv/bin/python tools/export_index.py --db data/index.lancedb --out /tmp/export
dsearch build-index --meta /tmp/export/functions_local.meta.jsonl \
    --vectors /tmp/export/functions_local.vec.f32 --out data/dsi/functions_local.dsi
# ... repeat per table (functions/windows × hashed/local)
```

## Ingest a dtk-based project

Needs the project's built target objects (`build/<VERSION>/obj/**/*.o`) and
optionally a decomp.dev progress report:

```sh
dsearch --backend hashed ingest-dtk ~/etc/melee \
    --project melee --version GALE01 --windows \
    --report 'https://decomp.dev/doldecomp/melee/GALE01.json?mode=report'
```

Ingest is **incremental**: each function's token text is diffed against the
stored row, so re-running only re-embeds new/changed functions
(metadata-only changes like a moved match % reuse the stored vector).
Hashed ingest of a full project is sub-second. For the `local` backend,
new/changed docs are embedded through the Python sidecar: pass
`--py .venv/bin/python` (or set `DSEARCH_PY`); unchanged rows never touch
the model. Multiple games coexist in one index.

## Query

```sh
# one-call donor lookup for agents: whole-fn twins + window twins, JSON,
# min-match fallback ladder (99.5→99→95→90) built in:
dsearch donors lbHeap_80015900 --project melee -k 10 --wk 6

# top matched functions similar to an unmatched one (the twin-finder):
dsearch find lbHeap_80015900 --min-match 99.5

# unfiltered similarity (see the whole neighborhood):
dsearch find mpRightWallGetTop --all

# cross-TU only (drop trivial same-file siblings):
dsearch find mpRightWallGetTop --exclude-self-unit

# machine-readable (same schema as the old dsearch.jsonq):
dsearch --json find lbHeap_80015900 -k 10
```

JSON hits carry `name, unit, src_path, match_pct, n_insns, sim, same_unit`
(+ `q_at`/`t_at` for windows). `--ladder` relaxes `--min-match` down the
99.5/99/95/90 ladder on zero hits and reports `min_match_used`.

## Construct-level (window) search

`ingest-dtk --windows` also indexes sliding 32-insn windows (stride 16) of
every function. `findw <fn>` then matches *any part* of the query function
against *any part* of the corpus — this finds construct twins (a loop shape
buried inside a larger matched function) that whole-function vectors
provably miss:

```sh
dsearch --backend hashed findw lbHeap_80015900 -k 10
# -> MakeColorGenTExp t@416: the 2x-unroll construct, invisible to `find`
```

All query windows are answered in **one** batched scan over the corpus
(memory-bandwidth bound, so extra windows are nearly free — 42 windows cost
86 ms, not 42 × 30 ms).

## Solvability sweep

```sh
dsearch --backend local sweep --project melee --min-sim 0.85 > solvable.json
```

Every sub-100% function ranked by its best matched (≥ `--donor-min`)
neighbor — the "probably solvable by donor" list. Same JSON as the old
nightly `jsonq sweep`, but fast enough to re-run on demand.

## Benchmarks & eval

```sh
dsearch bench --windows                      # internal latency percentiles
.venv/bin/python bench/run_bench.py ...      # head-to-head vs Python (bench/RESULTS.md)
dsearch --backend local eval                 # recall on eval/known_pairs.json
```

## Type-layout index

The struct-layout subsystem (redundant structs, union views, cast scans) is
unchanged Python — see `dsearch/typeidx.py` and the skill doc
(`.claude/skills/decomp-search/SKILL.md`).

## Index format (`.dsi`)

One mmap'd file per table: header, project/unit dictionaries, fixed 40-byte
row records, a name-sorted permutation (binary-search lookup), string blob
(names + token docs — token pages are never touched by queries), then
64-byte-aligned L2-normalized f32 vectors. Load cost is a mmap + header
parse; the OS page cache keeps vectors hot between calls.
