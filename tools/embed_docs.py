#!/usr/bin/env python3
"""Embed token docs read from stdin (JSONL: {"tokens": ...} or raw strings),
write raw little-endian f32 vectors (dim 512) to stdout, in input order.

Used by `dsearch ingest-dtk` for the model backends (local/voyage) — the
hashed backend is embedded natively in Rust. Needs the dsearch package
importable (run from the repo root or pip-install it).

Usage: embed_docs.py --backend local
"""

import argparse
import json
import struct
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="local")
    args = ap.parse_args()

    docs = []
    for line in sys.stdin:
        line = line.rstrip("\n")
        if not line:
            continue
        try:
            obj = json.loads(line)
            docs.append(obj["tokens"] if isinstance(obj, dict) else obj)
        except json.JSONDecodeError:
            docs.append(line)

    from dsearch.embed import embed

    def progress(done: int, total: int) -> None:
        print(f"embedded {done}/{total}", file=sys.stderr, flush=True)

    vecs = embed(docs, backend=args.backend, progress=progress)
    out = sys.stdout.buffer
    for v in vecs:
        out.write(struct.pack(f"<{len(v)}f", *v))
    out.flush()


if __name__ == "__main__":
    main()
