"""Compare two G2 tags: token+score identity and wall delta per row. Usage: g2cmp.py TAGA TAGB"""
import json, sys, glob, os
A, B = sys.argv[1], sys.argv[2]
D = "/opt/real/r6/g2"
bad = 0
for mode in ["prealloc", "padkv", "g128", "g448", "tiers"]:
    for suf in ["", "_b4"]:
        pa, pb = f"{D}/{A}_{mode}{suf}.json", f"{D}/{B}_{mode}{suf}.json"
        if not (os.path.exists(pa) and os.path.exists(pb)):
            print(mode + suf, "missing"); continue
        ra, rb = json.load(open(pa)), json.load(open(pb))
        for x, y in zip(ra, rb):
            k = (x.get("workload", "b4"), x.get("beam", 5))
            same = x["tok_ids"] == y["tok_ids"] and (x.get("score"), x.get("scores")) == (y.get("score"), y.get("scores"))
            bad += not same
            d = (y["median_ms"] - x["median_ms"]) / x["median_ms"] * 100
            print(f"{mode+suf:12s} {str(k):22s} {x['median_ms']:8.1f} -> {y['median_ms']:8.1f} {d:+6.2f}% {'SAME' if same else 'DIFF'}")
print("G2 diffs:", bad)
