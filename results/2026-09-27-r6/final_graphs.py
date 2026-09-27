import json, statistics, glob
cfgs = ["r4base", "r5sc128", "r5tiers", "r5eager"]
D = {}
for c in cfgs:
    for rd in (1, 2):
        for r in json.load(open(f"/opt/real/r6/final_g_{c}_{rd}.json")):
            D.setdefault((c, r["workload"], r["beam"]), []).append(r)
specs = []
for (c, w, b) in D:
    if (w, b) not in specs: specs.append((w, b))
for beam in (5, 1):
    print(f"### beam {beam}: median-of-2-rounds ms (delta vs r4base, - = faster); tok same as r4base?")
    for (w, b) in specs:
        if b != beam: continue
        base = D[("r4base", w, b)]
        bm = statistics.median(x["median_ms"] for x in base)
        line = f"{w:9s} ntok={base[0]['ntok']:3d} r4base {bm:7.1f}"
        for c in cfgs[1:]:
            rows = D[(c, w, b)]
            m = statistics.median(x["median_ms"] for x in rows)
            tok = all(x["tok_ids"] == base[0]["tok_ids"] for x in rows)
            line += f" | {c} {m:7.1f} ({(m-bm)/bm*100:+5.1f}%){'' if tok else ' TOKDIFF'}"
        s = D[("r5tiers", w, b)][0]["summaries"]
        s = s[-1] if s else {}
        line += f" | tiers: tr={s.get('transitions')} cap={s.get('captures')} rep={s.get('replays')} seg={s.get('segment_replays')} [{s.get('tiers')}]"
        s2 = D[("r5sc128", w, b)][0]["summaries"]; s2 = s2[-1] if s2 else {}
        line += f" | sc128: cap={s2.get('captures')} rep={s2.get('replays')} steps={s2.get('steps')} dis={s2.get('disabled')}"
        print(line)
