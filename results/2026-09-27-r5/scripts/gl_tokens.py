"""Token parity of every G-long config vs eager (round 1 rows): exact / timestamp-only /
text diff per (workload, beam), plus |dscore|. Usage: gl_tokens.py PFX [TS_BEGIN]"""
import json, sys, glob

PFX = sys.argv[1]
TS = int(sys.argv[2]) if len(sys.argv) > 2 else 50364


def load(tag):
    f = f"{PFX}_{tag}_1.json"
    try:
        return {(r["workload"], r["beam"]): r for r in json.load(open(f))}
    except FileNotFoundError:
        return None


eager = load("eager")
cfgs = ["sc128", "t64", "t64w", "lad", "sc256", "sc448", "t64p", "sc128p"]
data = {c: load(c) for c in cfgs}
keys = sorted(eager, key=lambda k: (k[1] * -1, k[0]))
print("workload     beam  " + " ".join(f"{c:>8s}" for c in cfgs if data[c]))
for k in keys:
    line = f"{k[0]:12s} b{k[1]}  "
    for c in cfgs:
        d = data[c]
        if not d or k not in d:
            continue
        a, b = eager[k]["tok_ids"], d[k]["tok_ids"]
        if a == b:
            tag = "="
        elif [t for t in a if t < TS] == [t for t in b if t < TS]:
            tag = "ts"
        else:
            tag = "TEXT"
        ds = abs(eager[k]["score"] - d[k]["score"])
        line += f"{tag:>4s}{ds:4.0e}".replace("e-0", "e-") + " "
    print(line)
print("(= identical ids; ts = timestamp-only diff; TEXT = text-token diff; number = |dscore| vs eager)")
