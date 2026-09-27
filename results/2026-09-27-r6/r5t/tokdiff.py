"""Token parity of A vs B rows (same workload/beam): exact, timestamp-only diff (the
text tokens, id < TS_BEGIN, identical), or text diff (+ word error rate on the decoded
text tokens); |dscore|. Usage: tokdiff.py A.json B.json [TS_BEGIN=50364]"""
import json, sys

ts_begin = int(sys.argv[3]) if len(sys.argv) > 3 else 50364


def key(r):
    return (r.get("workload"), r.get("beam"), r.get("model"), r.get("ctype"),
            r.get("timestamps"), r.get("batch"))


def load(p):
    out = {}
    for r in json.load(open(p)):
        out.setdefault(key(r), r)
    return out


def wer(a, b):
    # token-level edit distance / len(a)
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
            prev, d[j] = d[j], cur
    return d[len(b)] / max(1, len(a))


A, B = load(sys.argv[1]), load(sys.argv[2])
n = exact = tsonly = 0
max_ds = 0.0
for k in A:
    if k not in B:
        continue
    n += 1
    ta, tb = A[k]["tok_ids"], B[k]["tok_ids"]
    sa, sb = A[k].get("score"), B[k].get("score")
    ds = abs(sa - sb) if isinstance(sa, float) and isinstance(sb, float) else 0.0
    max_ds = max(max_ds, ds)
    if ta == tb:
        exact += 1
        continue
    xa = [t for t in ta if t < ts_begin]
    xb = [t for t in tb if t < ts_begin]
    if xa == xb:
        tsonly += 1
        print(f"  TS-ONLY {k[:2]} ntok {len(ta)}/{len(tb)} |dscore|={ds:.3g}")
    else:
        print(f"  TEXT-DIFF {k[:2]} ntok {len(ta)}/{len(tb)} text-token WER={wer(xa, xb):.4f} |dscore|={ds:.3g}")
print(f"{sys.argv[1]} vs {sys.argv[2]}: rows={n} exact={exact} timestamp-only={tsonly} "
      f"text-diff={n - exact - tsonly} max|dscore|={max_ds:.3g}")
