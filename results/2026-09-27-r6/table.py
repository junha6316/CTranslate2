import json
rows = [json.loads(l) for l in open("/opt/real/r6/probes.jsonl")]
ref = {}
for d in rows:
    key = (d["model"], d["ctype"], d.get("flash", 1))
    if d["label"].startswith("base_") and key not in ref:
        ref[key] = d
for d in rows:
    if d["label"] == "smoke":
        continue
    key = (d["model"], d["ctype"], d.get("flash", 1))
    r = ref.get(key)
    out = []
    for s in ("cold", "poisoned"):
        if s not in d:
            continue
        x = d[s]
        eq0 = r is not None and x["tok_ids"] == r["cold"]["tok_ids"]
        eqw = None
        if "per_window_tok_ids" in x and r is not None and "per_window_tok_ids" in r["cold"]:
            eqw = x["per_window_tok_ids"] == r["cold"]["per_window_tok_ids"]
        wt = x.get("per_window_total_ms")
        out.append("%s %.2f IQR %.2f ntok %d w0==base %s allwin==base %s wtotal %s"
                   % (s[:4], x["median_ms"], x["iqr_ms"], x["ntok"], eq0, eqw, wt))
    thr = d.get("pool_threshold_env")
    print("%-22s %-28s thr=%s | %s" % (d["label"], "/".join(map(str, key)),
                                       "max" if thr and len(thr) > 10 else thr, " | ".join(out)))
