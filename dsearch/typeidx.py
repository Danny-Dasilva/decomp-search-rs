"""Structural type-layout index: redundant structs, union views, cast scans.

Completely separate from the embedding pipeline: the input is a clang
record-layout dump of a project's m2ctx context file (self-contained C),
laid out for PPC EABI (matches MWCC -align powerpc), stored as plain JSON
under data/types-<project>.json.

Every record is flattened to layout *leaves*: (bit_offset, kind, count)
where kind normalizes away names, typedefs, and signedness (i8/i16/i32/
i64/f32/f64/ptr/bf). Two records with the same leaf signature are the
same decode; a union member whose subtree signature matches a sibling's
is a redundant view; a record that only ever appears in cast expressions
and is mostly pad fields is an overlay view.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

_BUILTIN = {
    "char": ("i8", 1), "signed char": ("i8", 1), "unsigned char": ("i8", 1),
    "_Bool": ("i8", 1), "bool": ("i8", 1),
    "short": ("i16", 2), "unsigned short": ("i16", 2),
    "int": ("i32", 4), "unsigned int": ("i32", 4),
    "long": ("i32", 4), "unsigned long": ("i32", 4),
    "long long": ("i64", 8), "unsigned long long": ("i64", 8),
    "float": ("f32", 4), "double": ("f64", 8), "long double": ("f64", 8),
    "void": ("void", 0),
    "signed": ("i32", 4), "unsigned": ("i32", 4),
    "signed short": ("i16", 2), "signed int": ("i32", 4),
    "signed long": ("i32", 4), "signed long long": ("i64", 8),
    "short int": ("i16", 2), "unsigned short int": ("i16", 2),
    "long int": ("i32", 4), "unsigned long int": ("i32", 4),
    "long long int": ("i64", 8), "unsigned long long int": ("i64", 8),
}
_KIND_SIZE = {"i8": 1, "i16": 2, "i32": 4, "i64": 8, "f32": 4, "f64": 8,
              "ptr": 4, "void": 0}

_PAD_NAME = re.compile(r"^_?(pad|filler|unused|unk_?pad|dummy)", re.I)


# ---------------------------------------------------------------- ctx parse

def parse_typedefs(ctx_text: str) -> dict[str, str]:
    """Map typedef aliases to their underlying type spelling.

    Handles `typedef T Name;`, `typedef T Name[N];`, function pointers
    (-> "*"), and `typedef struct Tag {...} Name;` via brace matching.
    Anonymous-tag typedefs map Name -> "@anon:<lineno>" so dump names like
    `struct (unnamed at ctx.c:91:9)` can be joined back to their alias.
    """
    out: dict[str, str] = {}
    for m in re.finditer(r"typedef\s+([^;{}()]+?)\s*;", ctx_text):
        body = m.group(1)
        dm = re.search(r"([A-Za-z_]\w*)\s*((?:\[[^\]]*\]\s*)*)$", body)
        if not dm:
            continue
        name, dims = dm.group(1), dm.group(2)
        base = re.sub(r"\bconst\b|\bvolatile\b", "",
                      body[: dm.start()]).strip()
        if not base:
            continue
        out[name] = ("*" if base.endswith("*")
                     else base + dims.replace(" ", ""))
    for m in re.finditer(r"typedef\s+[\w \*]+\(\s*\*\s*(\w+)\s*\)\s*\(",
                         ctx_text):
        out[m.group(1)] = "*"
    # typedef struct Tag? { ... } Name;  (brace-matched)
    for m in re.finditer(r"typedef\s+(struct|union|enum)\s*(\w+)?\s*\{",
                         ctx_text):
        kw, tag = m.group(1), m.group(2)
        depth, i = 1, m.end()
        while depth and i < len(ctx_text):
            c = ctx_text[i]
            depth += (c == "{") - (c == "}")
            i += 1
        tail = ctx_text[i:ctx_text.index(";", i)]
        lineno = ctx_text.count("\n", 0, m.end()) + 1
        for name in re.findall(r"\*?\s*(\w+)", tail):
            out[name] = f"{kw} {tag}" if tag else f"@anon:{lineno}"
    for m in re.finditer(r"typedef\s+(struct|union|enum)\s+(\w+)\s+(\w+)\s*;",
                         ctx_text):
        out[m.group(3)] = f"{m.group(1)} {m.group(2)}"
    return out


# --------------------------------------------------------------- dump parse

@dataclass
class Field:
    bitoff: int          # absolute offset in bits from record start
    depth: int           # 1 = direct member
    type: str
    name: str            # "" for unnamed members
    is_bf: bool = False


@dataclass
class Record:
    name: str
    size: int = 0
    align: int = 0
    fields: list[Field] = field(default_factory=list)
    leaves: list[tuple] = field(default_factory=list)  # (bitoff, kind, count)


_LINE = re.compile(r"^\s*([0-9]+)(?::([0-9]+)-?[0-9]*)?\s*\|(\s+)(.*)$")
_TAIL = re.compile(r"\[sizeof=(\d+),\s*(?:dsize=\d+,\s*)?align=(\d+)")


def parse_dump(text: str) -> dict[str, Record]:
    records: dict[str, Record] = {}
    cur: Record | None = None
    for line in text.splitlines():
        if "Dumping AST Record Layout" in line:
            cur = None
            continue
        tm = _TAIL.search(line)
        if tm and cur is not None:
            cur.size, cur.align = int(tm.group(1)), int(tm.group(2))
            if cur.name not in records:      # keep first definition
                records[cur.name] = cur
            cur = None
            continue
        m = _LINE.match(line)
        if not m:
            continue
        off, bit, indent, rest = m.groups()
        depth = (len(indent) - 1) // 2
        if depth == 0:
            cur = Record(name=rest.strip())
            continue
        if cur is None:
            continue
        parts = rest.rstrip().rsplit(" ", 1)
        if len(parts) == 2 and re.fullmatch(r"[A-Za-z_]\w*", parts[1]):
            ftype, fname = parts
        else:
            ftype, fname = rest.strip(), ""
        bitoff = int(off) * 8 + (int(bit) if bit else 0)
        cur.fields.append(Field(bitoff, depth, ftype.strip(), fname,
                                is_bf=bit is not None))
    return records


# ------------------------------------------------------------ canonicalize

def resolve(type_str: str, typedefs: dict[str, str],
            depth: int = 0) -> tuple[str, int, int]:
    """-> (kind, elem_size_bytes, count). kind 'rec:<name>' for records."""
    t = re.sub(r"\s+", " ", type_str.replace("const ", "")).strip()
    if depth > 20:
        return ("unk", 0, 1)
    if "(*)" in t:
        return ("ptr", 4, 1)
    count = 1
    for d in re.findall(r"\[(\d+)\]", t):
        count *= int(d)
    t = re.sub(r"(\[\d*\])+$", "", t).strip()
    if t.endswith("*"):
        return ("ptr", 4, count)
    if t.startswith("enum "):
        return ("i32", 4, count)
    if t in _BUILTIN:
        k, s = _BUILTIN[t]
        return (k, s, count)
    if t.startswith(("struct ", "union ")):
        return (f"rec:{t}", 0, count)
    if t in typedefs:
        k, s, c = resolve(typedefs[t], typedefs, depth + 1)
        return (k, s, c * count)
    return ("unk", 0, count)


def build_leaves(rec: Record, records: dict[str, Record],
                 typedefs: dict[str, str]) -> list[tuple]:
    """Flatten to (bitoff, kind, count) leaves, expanding arrays of records
    via the record index. Nested single records are already expanded inline
    by clang, so record-typed fields that have children are skipped."""
    leaves: list[tuple] = []
    fields = rec.fields
    for i, f in enumerate(fields):
        has_children = i + 1 < len(fields) and fields[i + 1].depth > f.depth
        if has_children:
            continue                     # expanded inline; take the leaves
        if f.is_bf:
            leaves.append((f.bitoff, "bf", 1))
            continue
        kind, esize, count = resolve(f.type, typedefs)
        if kind.startswith("rec:"):
            sub = records.get(kind[4:])
            if sub is not None and count <= 256:
                subleaves = build_leaves(sub, records, typedefs)
                for k in range(count):
                    base = f.bitoff + k * sub.size * 8
                    leaves.extend((base + o, sk, sc)
                                  for (o, sk, sc) in subleaves)
                continue
        leaves.append((f.bitoff, kind, count))
    leaves.sort()
    return leaves


def sig_of(leaves: list[tuple]) -> str:
    return hashlib.md5(repr(leaves).encode()).hexdigest()[:16]


def pad_fraction(rec: Record, typedefs: dict[str, str]) -> float:
    if not rec.size:
        return 0.0
    pad = 0
    for f in rec.fields:
        if f.depth == 1 and _PAD_NAME.match(f.name or ""):
            _, esize, count = resolve(f.type, typedefs)
            pad += esize * count
    return pad / rec.size


# ------------------------------------------------------------------ ingest

def make_dump(ctx: Path, clang: str = "clang") -> str:
    r = subprocess.run(
        [clang, "-target", "powerpc-unknown-eabi", "-std=gnu99",
         "-fsyntax-only", "-Wno-everything",
         "-Xclang", "-fdump-record-layouts-complete", str(ctx)],
        capture_output=True, text=True)
    if not r.stdout.strip():
        raise RuntimeError(f"clang produced no layouts: {r.stderr[:500]}")
    return r.stdout


def resolve_anon_aliases(records: dict[str, Record],
                         typedefs: dict[str, str]) -> dict[str, str]:
    """Join `struct (unnamed at ctx.c:N:C)` dump names to typedef aliases
    recorded as @anon:<lineno>."""
    by_line = {v[6:]: k for k, v in typedefs.items()
               if v.startswith("@anon:")}
    out = {}
    for name in records:
        m = re.search(r"\(unnamed at [^)]*:(\d+):\d+\)", name)
        if m and m.group(1) in by_line:
            out[name] = by_line[m.group(1)]
    return out


def index_path(project: str) -> Path:
    return DATA_DIR / f"types-{project}.json"


def ingest(project: str, ctx: Path, clang: str = "clang") -> dict:
    ctx_text = ctx.read_text(errors="replace")
    typedefs = parse_typedefs(ctx_text)
    records = parse_dump(make_dump(ctx, clang))
    aliases = resolve_anon_aliases(records, typedefs)
    out = {"project": project, "ctx": str(ctx), "typedefs": typedefs,
           "aliases": aliases, "records": {}}
    for name, rec in records.items():
        leaves = build_leaves(rec, records, typedefs)
        out["records"][name] = {
            "size": rec.size, "align": rec.align,
            "n_fields": sum(1 for f in rec.fields if f.depth == 1),
            "sig": sig_of(leaves), "leaves": leaves,
            "pad_frac": round(pad_fraction(rec, typedefs), 3),
            "fields": [[f.bitoff, f.depth, f.type, f.name] for f in
                       rec.fields],
        }
    index_path(project).parent.mkdir(parents=True, exist_ok=True)
    index_path(project).write_text(json.dumps(out))
    return out


def load(project: str) -> dict:
    p = index_path(project)
    if not p.exists():
        raise SystemExit(f"no type index for '{project}' — run types-ingest")
    return json.loads(p.read_text())


# ----------------------------------------------------------------- queries

def display_name(idx: dict, name: str) -> str:
    alias = idx["aliases"].get(name)
    return f"{alias} ({name.split('(')[0].strip()} anon)" if alias else name


def dup_groups(idx: dict, min_leaves: int = 3) -> list[list[str]]:
    by_sig: dict[str, list[str]] = {}
    for name, r in idx["records"].items():
        if len(r["leaves"]) >= min_leaves and r["size"] > 0:
            by_sig.setdefault(f'{r["size"]}:{r["sig"]}', []).append(name)
    return sorted((g for g in by_sig.values() if len(g) > 1),
                  key=lambda g: -idx["records"][g[0]]["size"])


def prefix_pairs(idx: dict, min_leaves: int = 4) -> list[tuple[str, str]]:
    """A's full leaf list == the first leaves of B (A a truncated decode
    of B, or B an extended decode of A)."""
    recs = [(n, tuple(map(tuple, r["leaves"])), r["size"])
            for n, r in idx["records"].items()
            if len(r["leaves"]) >= min_leaves]
    by_head: dict[tuple, list[int]] = {}
    for i, (_, lv, _) in enumerate(recs):
        by_head.setdefault(lv[:min_leaves], []).append(i)
    out = []
    for i, (na, la, sa) in enumerate(recs):
        for j in by_head.get(la[:min_leaves], []):
            nb, lb, sb = recs[j]
            if i != j and len(lb) > len(la) and lb[:len(la)] == la \
                    and sb > sa:
                out.append((na, nb))
    return sorted(out, key=lambda p: -idx["records"][p[0]]["size"])


def near(idx: dict, name: str, k: int = 15) -> list[tuple[float, str]]:
    import difflib
    recs = idx["records"]
    target = _find_record(idx, name)
    ta = [f"{o}:{kd}x{c}" for o, kd, c in recs[target]["leaves"]]
    scored = []
    for n, r in recs.items():
        if n == target or not r["leaves"]:
            continue
        tb = [f"{o}:{kd}x{c}" for o, kd, c in r["leaves"]]
        ratio = difflib.SequenceMatcher(None, ta, tb, autojunk=False).ratio()
        scored.append((ratio, n))
    return sorted(scored, reverse=True)[:k]


def _find_record(idx: dict, name: str) -> str:
    recs = idx["records"]
    for cand in (name, f"struct {name}", f"union {name}"):
        if cand in recs:
            return cand
    td = idx["typedefs"].get(name)
    if td and td in recs:
        return td
    if td and td.startswith("@anon:"):
        for anon, alias in idx["aliases"].items():
            if alias == name:
                return anon
    hits = [n for n in recs if name in n]
    if len(hits) == 1:
        return hits[0]
    raise SystemExit(f"unknown record '{name}'"
                     + (f" (candidates: {hits[:8]})" if hits else ""))


def union_views(idx: dict, name: str) -> list[list[str]]:
    """Groups of direct members whose relative leaf subtrees are identical."""
    rec = idx["records"][_find_record(idx, name)]
    fields = rec["fields"]
    typedefs = idx["typedefs"]
    groups: dict[str, list[str]] = {}
    for i, (off, depth, ftype, fname) in enumerate(fields):
        if depth != 1:
            continue
        sub = [(o - off, k, c) for o, d, k, c in _subtree_leaves(idx, rec, i)]
        key = sig_of(sorted(sub))
        groups.setdefault(key, []).append(fname or ftype)
    return [g for g in groups.values() if len(g) > 1]


def _fsize(idx: dict, f: list) -> int:
    kind, esize, count = resolve(f[2], idx["typedefs"])
    if kind.startswith("rec:") and kind[4:] in idx["records"]:
        esize = idx["records"][kind[4:]]["size"]
    return esize * count


def _subtree_leaves(idx: dict, rec: dict, i: int):
    """Leaves belonging to direct member i: its own inline-expanded children
    from the dump, or itself resolved."""
    fields = rec["fields"]
    off, depth, ftype, fname = fields[i]
    j = i + 1
    subs = []
    while j < len(fields) and fields[j][1] > depth:
        subs.append(fields[j])
        j += 1
    if subs:
        leaf_rows = []
        for k, f in enumerate(subs):
            deeper = k + 1 < len(subs) and subs[k + 1][1] > f[1]
            if not deeper:
                kind, esz, cnt = resolve(f[2], idx["typedefs"])
                leaf_rows.append((f[0], f[1], kind, cnt))
        return leaf_rows
    kind, esz, cnt = resolve(ftype, idx["typedefs"])
    return [(off, depth, kind, cnt)]


def members_at(idx: dict, name: str, byte_off: int) -> list[tuple[str, int]]:
    """Every field path in the record covering byte offset `byte_off`."""
    rec = idx["records"][_find_record(idx, name)]
    fields = rec["fields"]
    stack: list[tuple[int, str]] = []
    hits = []
    for i, (off, depth, ftype, fname) in enumerate(fields):
        while stack and stack[-1][0] >= depth:
            stack.pop()
        label = fname or f"<{ftype}>"
        stack.append((depth, label))
        size = _fsize(idx, fields[i])
        has_children = (i + 1 < len(fields) and fields[i + 1][1] > depth)
        if has_children:
            continue
        start = off // 8
        if size and start <= byte_off < start + size:
            path = ".".join(s for _, s in stack)
            hits.append((f"{path}  [{ftype}]", start))
    return hits


# -------------------------------------------------------------- cast scan

_CAST = re.compile(r"\(\s*(?:const\s+)?(?:struct\s+|union\s+)?"
                   r"([A-Za-z_]\w*)\s*(?:const\s+)?\*+\s*\)\s*[&\w(]")


def scan_casts(idx: dict, src_root: Path):
    """Per record type: cast sites, non-cast uses, pad fraction, dup group.
    A type with 0 non-cast uses and high pad fraction is an overlay view."""
    recs = idx["records"]
    known: dict[str, str] = {}
    for n in recs:
        m = re.match(r"(?:struct|union) (\w+)$", n)
        if m:
            known[m.group(1)] = n
    for alias, target in idx["typedefs"].items():
        if target in recs:
            known.setdefault(alias, target)
    for anon, alias in idx["aliases"].items():
        known.setdefault(alias, anon)

    from collections import Counter
    sites: dict[str, list[str]] = {}
    cast_counts: Counter = Counter()
    word_counts: Counter = Counter()
    ident = re.compile(r"[A-Za-z_]\w*")
    files = [p for p in src_root.rglob("*") if p.suffix in (".c", ".h")]
    for p in files:
        text = p.read_text(errors="replace")
        word_counts.update(w for w in ident.findall(text) if w in known)
        for m in _CAST.finditer(text):
            nm = m.group(1)
            if nm in known:
                ln = text.count("\n", 0, m.start()) + 1
                sites.setdefault(nm, []).append(f"{p.relative_to(src_root)}"
                                                f":{ln}")
                cast_counts[nm] += 1
    uses = {nm: word_counts[nm] - cast_counts[nm] for nm in sites}
    sig_groups: dict[str, list[str]] = {}
    for n, r in recs.items():
        sig_groups.setdefault(f'{r["size"]}:{r["sig"]}', []).append(n)
    rows = []
    for nm, locs in sorted(sites.items(), key=lambda kv: -len(kv[1])):
        rec = recs[known[nm]]
        dups = ([x for x in
                 sig_groups.get(f'{rec["size"]}:{rec["sig"]}', [])
                 if x != known[nm]]
                if rec["size"] >= 8 and len(rec["leaves"]) >= 4 else [])
        rows.append({"type": nm, "record": known[nm], "sites": locs,
                     "non_cast_uses": uses.get(nm, 0),
                     "pad_frac": rec["pad_frac"], "size": rec["size"],
                     "layout_dups": dups})
    return rows
