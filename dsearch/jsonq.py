"""JSON query interface for agent-harness integration.

Prints machine-readable results so callers don't parse rich tables.

  python -m dsearch.jsonq find FN [--project NAME] [--backend local]
         [--min-match 99.5] [-k 10] [--all] [--exclude-self-unit]
  python -m dsearch.jsonq findw FN [...same...]
  python -m dsearch.jsonq sweep [--project NAME] [--backend local]
         [--max-match 99.999] [--min-insns 8] [--donor-min 99.5] [-k 20]

`sweep` is the solvability sweep: every sub-`max-match` function ranked by its
best matched (>= donor-min) neighbor. Output: {fn: {sim, donor, donor_pct,
donor_unit, donor_src, donor_insns, n_insns, same_unit, pct}}.
"""

from __future__ import annotations

import argparse
import json
import sys

from . import db as dbmod


def _open(args, kind="functions"):
    conn = dbmod.connect(args.db)
    name = dbmod.table_name(args.backend, kind)
    if not dbmod.has_table(conn, name):
        print(json.dumps({"error": f"no {kind} index for backend "
                                   f"{args.backend!r}"}))
        sys.exit(1)
    return conn.open_table(name)


def _lookup(table, name, project):
    q = table.search().where(
        f"name = '{name}'" + (f" AND project = '{project}'" if project else ""),
        prefilter=True).limit(2).to_list()
    return q[0] if q else None


def _slim(r, sim):
    return {"name": r["name"], "unit": r["unit"], "src_path": r["src_path"],
            "match_pct": round(float(r["match_pct"]), 3),
            "n_insns": int(r["n_insns"]), "sim": round(sim, 4)}


def cmd_find(args):
    table = _open(args)
    row = _lookup(table, args.function, args.project)
    if row is None:
        print(json.dumps({"error": f"function {args.function!r} not in index"}))
        return
    where = None if args.all else f"match_pct >= {args.min_match}"
    res = (table.search(row["vector"]).metric("cosine")
           .where(where, prefilter=True).limit(args.k * 10 + 50).to_list())
    out = []
    for r in res:
        if r["id"] == row["id"]:
            continue
        if args.exclude_self_unit and r["unit"] == row["unit"] \
                and r["project"] == row["project"]:
            continue
        out.append(_slim(r, 1.0 - r["_distance"]))
        if len(out) >= args.k:
            break
    print(json.dumps({"query": {"name": row["name"], "unit": row["unit"],
                                "n_insns": int(row["n_insns"]),
                                "match_pct": round(float(row["match_pct"]), 3)},
                      "hits": out}))


def cmd_findw(args):
    wt = _open(args, kind="windows")
    where = f"name = '{args.function}'"
    if args.project:
        where += f" AND project = '{args.project}'"
    qrows = wt.search().where(where, prefilter=True).limit(1000).to_list()
    if not qrows:
        print(json.dumps({"error": f"no windows for {args.function!r}"}))
        return
    flt = None if args.all else f"match_pct >= {args.min_match}"
    best = {}
    for q in qrows:
        res = (wt.search(q["vector"]).metric("cosine")
               .where(flt, prefilter=True).limit(80).to_list())
        for h in res:
            if h["name"] == q["name"] and h["project"] == q["project"]:
                continue
            if args.exclude_self_unit and h["unit"] == q["unit"] \
                    and h["project"] == q["project"]:
                continue
            key = (h["project"], h["unit"], h["name"])
            sim = 1.0 - h["_distance"]
            if key not in best or sim > best[key]["sim"]:
                e = _slim(h, sim)
                e["q_at"] = int(q["id"].rsplit(":w", 1)[1])
                e["t_at"] = int(h["id"].rsplit(":w", 1)[1])
                best[key] = e
    ranked = sorted(best.values(), key=lambda e: -e["sim"])[: args.k]
    print(json.dumps({"query": {"name": args.function,
                                "windows": len(qrows)}, "hits": ranked}))


def cmd_sweep(args):
    table = _open(args)
    where = f"match_pct >= 0 AND match_pct < {args.max_match}"
    if args.project:
        where = f"project = '{args.project}' AND " + where
    rows = table.search().where(where, prefilter=True).limit(50000).to_list()
    out = {}
    for r in rows:
        if r["n_insns"] < args.min_insns:
            continue
        hits = (table.search(r["vector"]).metric("cosine")
                .where(f"match_pct >= {args.donor_min}", prefilter=True)
                .limit(args.k).to_list())
        best = None
        for h in hits:
            if h["id"] == r["id"] or h["name"] == r["name"]:
                continue
            best = h
            break
        if best is None:
            continue
        sim = 1.0 - best["_distance"]
        if sim < args.min_sim:
            continue
        out[r["name"]] = {
            "sim": round(sim, 4), "pct": round(float(r["match_pct"]), 3),
            "n_insns": int(r["n_insns"]),
            "donor": best["name"], "donor_pct": round(float(best["match_pct"]), 3),
            "donor_unit": best["unit"], "donor_src": best["src_path"],
            "donor_insns": int(best["n_insns"]),
            "same_unit": best["unit"] == r["unit"],
        }
        print(f"{len(out)} swept", file=sys.stderr, flush=True) \
            if len(out) % 500 == 0 else None
    print(json.dumps(out))


def main():
    p = argparse.ArgumentParser(prog="dsearch.jsonq")
    p.add_argument("--db", default=str(dbmod.DEFAULT_DB))
    p.add_argument("--backend", default="local",
                   choices=["hashed", "local", "voyage"])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("find", cmd_find), ("findw", cmd_findw)):
        sp = sub.add_parser(name)
        sp.add_argument("function")
        sp.add_argument("--project", default=None)
        sp.add_argument("--min-match", type=float, default=99.5)
        sp.add_argument("--all", action="store_true")
        sp.add_argument("-k", type=int, default=10)
        sp.add_argument("--exclude-self-unit", action="store_true")
        sp.set_defaults(func=fn)
    ss = sub.add_parser("sweep")
    ss.add_argument("--project", default=None)
    ss.add_argument("--max-match", type=float, default=99.999)
    ss.add_argument("--min-insns", type=int, default=8)
    ss.add_argument("--donor-min", type=float, default=99.5)
    ss.add_argument("--min-sim", type=float, default=0.85)
    ss.add_argument("-k", type=int, default=20)
    ss.set_defaults(func=cmd_sweep)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
