---
name: decomp-search
description: Local similarity search over decomp-project functions — find matched twins and recipe donors for an unmatched function's asm, whole-function or construct-level (windows). Use when stuck on a match residual, hunting shape twins, or scoping whether a function is unique.
---

# decomp-search

Rust binary, mmap'd flat index, exact brute-force cosine. Every query is
fast enough to call freely in a loop: `find` ~10ms, `findw` ~30-90ms,
`donors` ~50ms — full process cost, no server needed.

Build once: `cargo build --release` in `rust/` (or `cargo install --path rust`).
Binary: `rust/target/release/dsearch`. Index dir: `data/dsi/*.dsi`
(`--index-dir` or `DSEARCH_INDEX_DIR`). Backend: `--backend hashed|local`
(default `local`; `DSEARCH_BACKEND` overrides).

## One-call donor lookup (preferred for agents)

```sh
dsearch --backend local donors <fn> --project melee -k 10 --wk 6
```

JSON only. Returns whole-function twins (`twins`, local backend) AND
construct/window twins (`windows`, hashed backend via `--wbackend`) in one
call, with the min-match fallback ladder (99.5 -> 99 -> 95 -> 90) built in —
`min_match_used` in `query` tells you if it relaxed. Every hit carries
`name, unit, src_path, match_pct, n_insns, sim, same_unit` (+ `q_at`/`t_at`
window insn offsets). `same_unit: true` hits are the highest-yield donors.

## Core query — matched donors for a function

```sh
dsearch find <fn> --min-match 99.5 --exclude-self-unit -k 10 [--json] [--ladder]
```

- `--all` = no match% filter (see the whole neighborhood)
- `--ladder` = on zero hits, auto-relax min-match down 99/95/90
- Drop `--exclude-self-unit` to include same-TU siblings
- `find` never runs a model at query time — it searches with the function's
  stored vector, so a backend only covers projects ingested with it
  (check `stats`).
- Plain output is aligned text (never truncated); `--json` gives the
  machine schema (same shape as the old `dsearch.jsonq`).

## Construct-level query — window twins

A loop/construct buried inside a larger matched function won't surface in
`find` (whole-function vectors). `findw` searches every 32-insn sliding
window of the query fn against all indexed windows in ONE batched scan and
aggregates the best hit per candidate function (`q@`/`t@` = window insn
offsets):

```sh
dsearch --backend hashed findw <fn> --min-match 99.5 -k 10 [--json] [--ladder]
```

Validated: found lbHeap_80015900 -> MakeColorGenTExp (2x-unroll construct
at t@416) which whole-function search misses. findw for recall, twinscan
(`melee/build/twinscan_*.py`) for proof.

## Donor transplant (the highest-yield workflow)

When `find` returns a matched function at sim >0.98 with the same shape,
objdump BOTH and compare: if the instruction streams are identical modulo
displacements/trip counts, transplant the donor's SOURCE STRUCTURE wholesale
(wrapper+inline split, accessor macros, alias pairs, decl order, PAD values)
instead of tuning the current code. Validated: fn_802523D8 (mninfo) hit 100%
in one step from the mncount donor fn_802514D8 (sim 0.991).

## Picking solve targets

```sh
dsearch --backend local sweep --project melee --min-sim 0.85 > solvable.json
```

Ranks every sub-100% function by its best matched neighbor (same JSON as the
old nightly `jsonq sweep`, but runs in seconds — re-sweep on demand instead
of waiting for the cron).

## Ingest a project (dtk-based)

Needs target objects (`build/<VER>/obj/**/*.o`) + a decomp.dev report:

```sh
dsearch --backend hashed ingest-dtk <repo_root> \
    --project <name> --version <VER> --windows \
    --report 'https://decomp.dev/<org>/<repo>/<VER>.json?mode=report'
```

Sub-second for a full project (hashed). Incremental: unchanged token docs
reuse their stored vectors, so re-ingest after a build is instant and
metadata-only changes (match% moves) never re-embed. For the `local`
(voyage-4-nano) backend, new/changed docs are embedded via the Python
helper: add `--py <python-with-dsearch-pkg>` (or `DSEARCH_PY`); everything
else stays native. Migrating an existing LanceDB index instead:
`tools/export_index.py` + `dsearch build-index`.

Project names in the index: melee, mp4, pikmin2 — reuse these exact names
on re-ingest or you'll create a duplicate project.

## Type-layout index (redundant structs / casts / union views)

Still the Python subsystem (`dsearch/typeidx.py`, stdlib-only JSON at
`data/types-<project>.json`) — no embeddings, unchanged interface:

```sh
.venv/bin/python -m dsearch.cli types-ingest ~/etc/melee --project melee
# types-dups / types-near / types-unions / types-casts as before
```

## Maintain the eval

When you find a true twin pair manually, add it to `eval/known_pairs.json`,
then check recall: `dsearch --backend local eval` (7/8 baseline; the miss is
the documented findw-only construct case).

## Gotchas

- A backend's index only covers projects ingested with that backend —
  `stats` shows row counts per table.
- `search` (freeform text query) is hashed-backend-only in the Rust binary;
  model-backend text queries still need the Python CLI.
- Duplicate function names across projects/units exist (e.g. REL `fn_1_*`);
  pass `--project` to disambiguate which one you mean.
- This file is canonical in the repo
  (`.claude/skills/decomp-search/SKILL.md`); `~/.claude/skills/decomp-search/SKILL.md`
  is a symlink to it.
