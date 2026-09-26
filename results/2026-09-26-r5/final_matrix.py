import json, statistics
def load(p):
    return {(r['model'], r['ctype'], r['timestamps'], r['beam'], r['batch']): r for r in json.load(open(p))}
R = "/opt/real/r6/"
for fl in ("off", "on"):
    A = [load(f"{R}final_r4base_{i}_{fl}.json") for i in (1, 2)]
    B = [load(f"{R}final_r5_{i}_{fl}.json") for i in (1, 2)]
    keys = list(A[0])
    ds, same, rows = [], 0, []
    for k in keys:
        a = statistics.mean(x[k]['median_ms'] for x in A)
        b = statistics.mean(x[k]['median_ms'] for x in B)
        d = (b - a) / a * 100
        eq = all(x[k]['tok_ids'] == A[0][k]['tok_ids'] for x in A + B)
        p1 = (B[0][k]['median_ms'] - A[0][k]['median_ms']) / A[0][k]['median_ms'] * 100
        p2 = (B[1][k]['median_ms'] - A[1][k]['median_ms']) / A[1][k]['median_ms'] * 100
        ds.append(d); same += eq; rows.append((d, k, a, b, eq, p1, p2))
    print(f"flash-{fl}: {len(ds)} cfgs, tokens identical (4 runs) {same}/{len(ds)}, median {statistics.median(ds):+.2f}%, range {min(ds):+.1f}..{max(ds):+.1f}%")
    for d, k, a, b, eq, p1, p2 in sorted(rows):
        corner = fl == "on" and k[3] == 5 and not k[2]
        if abs(d) > 3 or not eq or corner:
            print(f"  {d:+6.1f}% (pass1 {p1:+.1f}, pass2 {p2:+.1f})  {k}  {a:.1f} -> {b:.1f}{'' if eq else '  TOK-DIFF'}{'  [corner]' if corner else ''}")
