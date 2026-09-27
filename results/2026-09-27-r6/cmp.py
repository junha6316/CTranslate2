import json, sys, statistics
def load(p):
    return {(r['model'], r['ctype'], r['timestamps'], r['beam'], r['batch']): r for r in json.load(open(p))}
a, b = load(sys.argv[1]), load(sys.argv[2])
ds, same = [], 0
rows = []
for k in a:
    if k not in b: continue
    d = (b[k]['median_ms'] - a[k]['median_ms']) / a[k]['median_ms'] * 100
    ds.append(d)
    eq = a[k]['tok_ids'] == b[k]['tok_ids']
    same += eq
    rows.append((d, k, a[k]['median_ms'], b[k]['median_ms'], eq))
print(f"{sys.argv[1]} -> {sys.argv[2]}: {len(ds)} cfgs, tokens identical {same}/{len(ds)}, median {statistics.median(ds):+.1f}%, range {min(ds):+.1f}..{max(ds):+.1f}%")
for d, k, x, y, eq in sorted(rows):
    if abs(d) > 3 or not eq:
        print(f"  {d:+6.1f}%  {k}  {x:.1f} -> {y:.1f}{'' if eq else '  TOK-DIFF'}")
