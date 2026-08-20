#!/usr/bin/env python3
"""Head-to-head benchmark: Python (LanceDB) dsearch vs Rust dsearch.

Runs matched end-to-end CLI invocations (full process cost, as agents pay it)
over the same query set and index data, reports median/p90 per command.

Usage: python bench/run_bench.py [--iters 10] [--out bench/RESULTS.md]
Requires: the patched Python copy (see --pydir), the .dsi indexes, and the
LanceDB index.
"""

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_RUST = HERE.parent / "rust/target/release/dsearch"
DEFAULT_DSI = HERE.parent / "data/dsi"

# override with --queries; pick names spanning small/median/p99 window counts
DEFAULT_QUERIES = ["lbHeap_80015900", "mpRightWallGetTop", "OSInit"]


def timeit(cmd, iters, warmup=2, cwd=None):
    for _ in range(warmup):
        subprocess.run(cmd, capture_output=True, cwd=cwd)
    times = []
    for _ in range(iters):
        t = time.perf_counter()
        r = subprocess.run(cmd, capture_output=True, cwd=cwd)
        times.append((time.perf_counter() - t) * 1000)
        if r.returncode != 0:
            return None, (r.stderr or r.stdout)[-300:].decode(errors="replace")
    return times, None


def fmt(times):
    if not times:
        return "FAIL"
    med = statistics.median(times)
    p90 = sorted(times)[min(len(times) - 1, int(len(times) * 0.9))]
    return f"{med:.1f} / {p90:.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--rust", default=str(DEFAULT_RUST))
    ap.add_argument("--dsi", default=str(DEFAULT_DSI))
    ap.add_argument("--py", default=".venv/bin/python",
                    help="python with the dsearch package + lancedb")
    ap.add_argument("--queries", nargs="*", default=DEFAULT_QUERIES,
                    help="function names to benchmark")
    ap.add_argument("--pydir", required=True, help="dir containing a working dsearch package")
    ap.add_argument("--db", required=True, help="LanceDB index path")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    Q = args.queries
    rows = []

    def bench(label, pycmd, rscmd):
        pt, perr = timeit(pycmd, args.iters, cwd=args.pydir) if pycmd else (None, "n/a")
        rt, rerr = timeit(rscmd, args.iters)
        pm = statistics.median(pt) if pt else None
        rm = statistics.median(rt) if rt else None
        speedup = f"{pm / rm:.0f}x" if pm and rm else "-"
        rows.append((label, fmt(pt) if pt else (perr or "n/a"), fmt(rt) if rt else rerr, speedup))
        print(f"{label:45s} py {fmt(pt) if pt else perr}  rs {fmt(rt) if rt else rerr}  {speedup}")

    py_base = [args.py, "-m", "dsearch.jsonq", "--db", args.db]
    rs_base = [args.rust, "--index-dir", args.dsi, "--json"]

    for backend in ("hashed", "local"):
        bench(
            f"find {Q[0]} ({backend})",
            py_base + ["--backend", backend, "find", Q[0], "--project", "", "-k", "10"],
            rs_base + ["--backend", backend, "find", Q[0], "-k", "10"],
        )
    for q in Q:
        bench(
            f"findw {q} (local)",
            py_base + ["--backend", "local", "findw", q, "--project", "", "-k", "10"],
            rs_base + ["--backend", "local", "findw", q, "-k", "10"],
        )
    # donors: python = two subprocess calls (find local + findw hashed), as
    # agent harnesses issue them; rust = one process
    label = f"donors {Q[0]} (find local + findw hashed)"
    pt1, e1 = timeit(py_base + ["--backend", "local", "find", Q[0], "--project", "", "-k", "10"], args.iters, cwd=args.pydir)
    pt2, e2 = timeit(py_base + ["--backend", "hashed", "findw", Q[0], "--project", "", "-k", "6"], args.iters, cwd=args.pydir)
    rt, rerr = timeit([args.rust, "--index-dir", args.dsi, "--backend", "local", "donors", Q[0], "-k", "10", "--wk", "6"], args.iters)
    if pt1 and pt2 and rt:
        combined = [a + b for a, b in zip(pt1, pt2)]
        pm, rm = statistics.median(combined), statistics.median(rt)
        rows.append((label, fmt(combined), fmt(rt), f"{pm / rm:.0f}x"))
        print(f"{label:45s} py {fmt(combined)}  rs {fmt(rt)}  {pm / rm:.0f}x")

    if args.out:
        with open(args.out, "w") as f:
            f.write("# Benchmark: Python (LanceDB) vs Rust — end-to-end CLI, ms (median / p90)\n\n")
            f.write(f"iters={args.iters}, warm page cache, full process cost per call\n\n")
            f.write("| command | python | rust | speedup |\n|---|---|---|---|\n")
            for r in rows:
                f.write(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} |\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
