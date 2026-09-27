"""G-long report: per (config, workload, beam) median over rounds of the per-run medians;
deltas vs single-cap graphs@128 (positive = faster than sc128) and vs eager; the G-long
pass criteria. Usage: glong_report.py PFX [MODEL]"""
import glob, json, statistics, sys
from collections import defaultdict

PFX = sys.argv[1]
rows = defaultdict(list)
for f in sorted(glob.glob(f"{PFX}_*_[0-9].json")):
    for r in json.load(open(f)):
        cfg = r["label"][len(PFX) + 1:]
        rows[(cfg, r["workload"], r["beam"])].append(r)
cfgs = sorted({k[0] for k in rows}, key=lambda c: ["eager", "sc128", "t64", "t64w", "lad", "sc256", "sc448", "t64p", "sc128p"].index(c) if c in ["eager", "sc128", "t64", "t64w", "lad", "sc256", "sc448", "t64p", "sc128p"] else 99)
wls = sorted({k[1] for k in rows}, key=lambda w: (w.split(":")[0], int(w.split(":")[1])))
beams = sorted({k[2] for k in rows}, reverse=True)
med = {k: statistics.median(r["median_ms"] for r in v) for k, v in rows.items()}


def summ(cfg, wl, beam):
    rr = rows.get((cfg, wl, beam), [])
    ss = [s for r in rr for s in r.get("summaries", [])]
    if not ss:
        return ""
    tr = sorted({s["transitions"] for s in ss})
    cap = sorted({s["captures"] for s in ss})
    dis = sorted({s["disabled"] for s in ss})
    seg = ss[-1]["segment_replays"]
    return f"tr={tr} cap={cap} seg={seg} tiers=[{ss[-1]['tiers']}]" + ("" if dis == ["none"] else f" dis={dis}")


for beam in beams:
    print(f"\n### beam {beam}: median ms (delta vs sc128, + = faster)")
    print("workload      ntok(e/sc/t64) P   " + "".join(f"{c:>15s}" for c in cfgs))
    for wl in wls:
        if ("sc128", wl, beam) not in med:
            continue
        base = med[("sc128", wl, beam)]
        nt = [rows[(c, wl, beam)][0]["ntok"] if (c, wl, beam) in rows else -1 for c in ("eager", "sc128", "t64")]
        P = rows[("sc128", wl, beam)][0].get("prompt_len")
        line = f"{wl:12s} {nt[0]:3d}/{nt[1]:3d}/{nt[2]:3d} {P:4d} "
        for c in cfgs:
            if (c, wl, beam) in med:
                m = med[(c, wl, beam)]
                line += f"{m:8.1f}({base - m:+5.1f})"
            else:
                line += " " * 15
        print(line)
    for c in cfgs:
        for wl in wls:
            s = summ(c, wl, beam)
            if s and c in ("t64", "t64w", "lad", "t64p"):
                print(f"   {c:6s} {wl:10s} b{beam}: {s}")


def gains(cfg, ref, kind, beam):
    out = {}
    for wl in wls:
        if wl.startswith(kind) and (cfg, wl, beam) in med and (ref, wl, beam) in med:
            out[wl] = med[(ref, wl, beam)] - med[(cfg, wl, beam)]
    return out


def mean(d):
    return sum(d.values()) / len(d) if d else float("nan")


verdict = {}
print("\n### criteria")
for cand, ref, tag in (("t64", "sc128", "pool unset"), ("t64p", "sc128p", "pool 512M")):
    if (cand, wls[0], beams[0]) not in med or (ref, wls[0], beams[0]) not in med:
        continue
    d5 = gains(cand, ref, "dense", 5)
    p5 = gains(cand, ref, "prev", 5)
    d1 = gains(cand, ref, "dense", 1)
    ntok = {wl: rows[("eager", wl, 5)][0]["ntok"] if ("eager", wl, 5) in rows else rows[(ref, wl, 5)][0]["ntok"] for wl in d5}
    c1 = mean(d5) >= 5 and all(g >= 3 for wl, g in d5.items() if ntok[wl] >= 160)
    c2 = mean(p5) >= 5 and all(g >= 1.5 for g in p5.values())
    c3 = mean(d1) >= 4 if d1 else None
    allg = [med[(ref, wl, b)] - med[(cand, wl, b)] for wl in wls for b in beams if (cand, wl, b) in med and (ref, wl, b) in med]
    c4 = min(allg) >= -3
    print(f"[{tag}] 1. beam5 W_dense mean {mean(d5):+.2f} ms, per-window {({k: round(v, 2) for k, v in d5.items()})} -> {'PASS' if c1 else 'FAIL'}")
    print(f"[{tag}] 2. beam5 W_prev mean {mean(p5):+.2f} ms, per-k {({k: round(v, 2) for k, v in p5.items()})} -> {'PASS' if c2 else 'FAIL'}")
    print(f"[{tag}] 3. greedy W_dense mean {mean(d1):+.2f} ms, per-window {({k: round(v, 2) for k, v in d1.items()})} -> {'PASS' if c3 else 'FAIL'}")
    print(f"[{tag}] 4. worst row {min(allg):+.2f} ms (need >= -3) -> {'PASS' if c4 else 'FAIL'}")
    verdict[tag] = (c1, c2, c3, c4)
if ("sc256", wls[0], 5) in med and ("sc448", wls[0], 5) in med:
    for beam in beams:
        dm = {c: mean({wl: med[(c, wl, beam)] for wl in wls if wl.startswith("dense") and (c, wl, beam) in med}) for c in ("t64", "sc256", "sc448", "sc128", "lad", "eager")}
        c5 = dm["t64"] <= dm["sc256"] and dm["t64"] <= dm["sc448"]
        print(f"5. beam{beam} W_dense mean ms: " + ", ".join(f"{c} {v:.2f}" for c, v in dm.items()) + f" -> {'PASS' if c5 else 'FAIL'}")
if ("t64w", wls[0], 5) in med:
    for beam in beams:
        fw = [med[("t64w", wl, beam)] - med[("t64", wl, beam)] for wl in wls if ("t64w", wl, beam) in med]
        print(f"7. beam{beam} fast vs warmup re-entry: mean(warmup - fast) {sum(fw) / len(fw):+.2f} ms, min {min(fw):+.2f} -> {'PASS' if sum(fw) / len(fw) >= 0 else 'FAIL'}")
print("\n### vs eager (mean ms delta, + = faster than eager)")
for c in cfgs:
    if c == "eager":
        continue
    for beam in beams:
        for kind in ("dense", "prev"):
            g = gains(c, "eager", kind, beam)
            if g:
                print(f"  {c:6s} b{beam} {kind:5s}: mean {mean(g):+7.2f}  min {min(g.values()):+7.2f}  max {max(g.values()):+7.2f}")
print("\n### token identity vs sc128 and vs eager (per row)")
for c in cfgs:
    same_sc = same_e = n = 0
    for wl in wls:
        for beam in beams:
            if (c, wl, beam) in rows and ("sc128", wl, beam) in rows:
                n += 1
                t = rows[(c, wl, beam)][0]["tok_ids"]
                same_sc += t == rows[("sc128", wl, beam)][0]["tok_ids"]
                same_e += ("eager", wl, beam) in rows and t == rows[("eager", wl, beam)][0]["tok_ids"]
    print(f"  {c:6s}: rows {n}, tokens == sc128 {same_sc}, == eager {same_e}")
