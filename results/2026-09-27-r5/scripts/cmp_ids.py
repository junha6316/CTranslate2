"""Exact token + score identity between two matrix_ids/check_windows JSON files.
Usage: cmp_ids.py A.json B.json [--scores]"""
import json, sys

KEYS = ("model", "ctype", "timestamps", "beam", "batch", "window", "ts", "workload")


def key(r):
    return tuple(r.get(k) for k in KEYS)


a = {key(r): r for r in json.load(open(sys.argv[1]))}
b = {key(r): r for r in json.load(open(sys.argv[2]))}
same_tok = same_all = 0
common = [k for k in a if k in b]
max_ds = 0.0
for k in common:
    ta, tb = a[k]["tok_ids"], b[k]["tok_ids"]
    sa = a[k].get("scores", a[k].get("score"))
    sb = b[k].get("scores", b[k].get("score"))
    tok_eq = ta == tb
    sc_eq = sa == sb
    if isinstance(sa, list) and isinstance(sb, list) and len(sa) == len(sb):
        ds = max([abs(x - y) for x, y in zip(sa, sb)] or [0.0])
    elif isinstance(sa, float) and isinstance(sb, float):
        ds = abs(sa - sb)
    else:
        ds = 0.0
    max_ds = max(max_ds, ds)
    same_tok += tok_eq
    same_all += tok_eq and sc_eq
    if not (tok_eq and sc_eq):
        print("  DIFF", k, "tokens" if not tok_eq else "", "scores |d|=%.3g" % ds if not sc_eq else "")
tag = "BITIDENTICAL" if same_all == len(common) == len(a) == len(b) else "DIFF"
print(f"{sys.argv[1]} vs {sys.argv[2]}: configs={len(common)} (A={len(a)} B={len(b)}) "
      f"tokens_identical={same_tok} tokens+scores_identical={same_all} max|dscore|={max_ds:.3g} -> {tag}")
