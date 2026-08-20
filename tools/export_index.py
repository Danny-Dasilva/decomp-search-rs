#!/usr/bin/env python3
"""Export LanceDB tables to the .dsi intermediate format.

For each table (functions_*/windows_*): writes <out>/<table>.meta.jsonl
(one JSON object per row) + <out>/<table>.vec.f32 (raw little-endian f32,
same row order). Feed both to `dsearch build-index`.

Usage: python export_index.py --db data/index.lancedb --out /tmp/export
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import lancedb
import numpy as np


def wstart_of(row_id: str) -> int | None:
    if ":w" in row_id:
        tail = row_id.rsplit(":w", 1)[1]
        if tail.isdigit():
            return int(tail)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tables", nargs="*", help="default: all real tables")
    args = ap.parse_args()

    db = lancedb.connect(args.db)
    names = args.tables or [
        t for t in db.table_names() if not t.startswith("._")
    ]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for name in names:
        table = db.open_table(name)
        n = table.count_rows()
        print(f"{name}: {n} rows", file=sys.stderr, flush=True)
        meta_f = (out / f"{name}.meta.jsonl").open("w")
        vec_f = (out / f"{name}.vec.f32").open("wb")
        written = 0
        for batch in table.to_arrow().to_batches(max_chunksize=8192):
            d = batch.to_pydict()
            vecs = np.asarray(d["vector"], dtype=np.float32)
            vec_f.write(vecs.tobytes())
            for i in range(len(d["id"])):
                rec = {
                    "name": d["name"][i],
                    "project": d["project"][i],
                    "unit": d["unit"][i],
                    "src_path": d["src_path"][i] or "",
                    "n_insns": int(d["n_insns"][i]),
                    "match_pct": float(d["match_pct"][i]),
                    "tokens": d["tokens"][i],
                }
                ws = wstart_of(d["id"][i])
                if ws is not None:
                    rec["wstart"] = ws
                meta_f.write(json.dumps(rec) + "\n")
            written += len(d["id"])
            print(f"  {written}/{n}", file=sys.stderr, flush=True)
        meta_f.close()
        vec_f.close()


if __name__ == "__main__":
    main()
