#!/bin/bash
# G-fallback: fault injection, 2 decodes per process (std:0 beam5, VERBOSE=2).
cd /opt/real/r6/r5t
run() {
  local tree=$1; shift
  ( source /opt/real/r6/r5t/env.sh "$tree"
    while [ "$1" != "--" ]; do export "$1"; shift; done; shift
    "$@" )
}
G="CT2_CUDA_GRAPHS=1 CT2_VERBOSE=2"
T64="CT2_CUDA_GRAPHS_RESERVE=64 CT2_CUDA_GRAPHS_TIERS=+64"
echo "=== G-fallback references (no fault) $(date +%T)"
run r3 $G $T64 -- python onedecode.py std:0 --beam 5 --n 2 --json gf_ref_t64.json > gf_ref_t64.log 2>&1
run r3 $G CT2_CUDA_GRAPHS_RESERVE=128 -- python onedecode.py std:0 --beam 5 --n 2 --json gf_ref_r128.json > gf_ref_r128.log 2>&1
run r3 $G CT2_CUDA_GRAPHS_RESERVE=64 -- python onedecode.py std:0 --beam 5 --n 2 --json gf_ref_sc64.json > gf_ref_sc64.log 2>&1
run r3 -- python onedecode.py std:0 --beam 5 --json gf_ref_eager.json > gf_ref_eager.log 2>&1
for F in tier_capture tier_instantiate tier_replay tier_alloc; do
  run r3 $G $T64 CT2_CUDA_GRAPHS_FAULT=$F -- python onedecode.py std:0 --beam 5 --n 2 --json gf_${F}_t64.json > gf_${F}_t64.log 2>&1
done
for F in capture instantiate replay fingerprint alloc; do
  run r3 $G CT2_CUDA_GRAPHS_RESERVE=128 CT2_CUDA_GRAPHS_FAULT=$F -- python onedecode.py std:0 --beam 5 --n 2 --json gf_${F}_r128.json > gf_${F}_r128.log 2>&1
  run r3 $G $T64 CT2_CUDA_GRAPHS_FAULT=$F -- python onedecode.py std:0 --beam 5 --n 2 --json gf_${F}_t64.json > gf_${F}_t64.log 2>&1
done
python3 - <<'EOF'
import json, re
def segments(log):
    segs, cur = [], None
    for line in open(log):
        if line.startswith("BEGIN "):
            cur = []
            segs.append(cur)
        elif cur is not None:
            cur.append(line)
    return segs
def counts(seg):
    return dict(tier=sum("CUDA graphs: tier" in l for l in seg),
                fb=sum("falling back" in l for l in seg),
                reason=[re.sub(r".*falling back to eager for this decode ", "", l).strip() for l in seg if "falling back" in l],
                order=[("T" if "CUDA graphs: tier" in l else "F") for l in seg if "CUDA graphs: tier" in l or "falling back" in l])
ref = {k: json.load(open(f"gf_ref_{k}.json")) for k in ["t64", "r128", "sc64", "eager"]}
eager_toks = ref["eager"][0]["tok_ids"]
allok = True
for F in ["tier_capture", "tier_instantiate", "tier_replay", "tier_alloc"]:
    rows = json.load(open(f"gf_{F}_t64.json"))
    s = [counts(x) for x in segments(f"gf_{F}_t64.log")]
    ok = (s[0]["tier"] == 1 and s[0]["fb"] == 1 and s[0]["order"] == ["T", "F"]
          and s[1]["tier"] == 1 and s[1]["fb"] == 0
          and rows[0]["tok_ids"] == ref["t64"][0]["tok_ids"] and rows[0]["score"] == ref["t64"][0]["score"]
          and rows[1]["tok_ids"] == ref["t64"][1]["tok_ids"] and rows[1]["score"] == ref["t64"][1]["score"])
    allok &= ok
    print(f"{F:17s} R=64 +64: decode0 tier={s[0]['tier']} fb={s[0]['fb']} order={s[0]['order']} reason={s[0]['reason']} | decode1 tier={s[1]['tier']} fb={s[1]['fb']} | "
          f"NTOK {rows[0]['ntok']}/{rows[1]['ntok']} score {rows[0]['score']:.8g} (no-fault {ref['t64'][0]['score']:.8g}) -> {'PASS' if ok else 'FAIL'}")
for F in ["capture", "instantiate", "replay", "fingerprint", "alloc"]:
    for cfg in ["r128", "t64"]:
        rows = json.load(open(f"gf_{F}_{cfg}.json"))
        s = [counts(x) for x in segments(f"gf_{F}_{cfg}.log")]
        ok = rows[0]["ntok"] == 84 and s[0]["fb"] >= 1 and rows[0]["tok_ids"] == eager_toks
        if F == "fingerprint" and cfg == "t64":
            ok &= s[0]["tier"] == 0 and rows[0]["tok_ids"] == ref["sc64"][0]["tok_ids"] and rows[0]["score"] == ref["sc64"][0]["score"]
        allok &= ok
        print(f"{F:12s} {cfg:4s}: decode0 tier={s[0]['tier']} fb={s[0]['fb']} reason={s[0]['reason'][:1]} | decode1 tier={s[1]['tier']} fb={s[1]['fb']} | NTOK {rows[0]['ntok']} tok==eager {rows[0]['tok_ids']==eager_toks} score {rows[0]['score']:.8g} -> {'PASS' if ok else 'FAIL'}")
print("single-cap@64 score", ref["sc64"][0]["score"], "tiers score", ref["t64"][0]["score"], "eager score", ref["eager"][0]["score"], "tiers tok==eager", ref["t64"][0]["tok_ids"] == eager_toks)
print("G-fallback verdict:", "PASS" if allok else "FAIL")
EOF
grep -h "does not fit" gf_fingerprint_t64.log | sed -E 's/^\[[^]]*\] \[ctranslate2\] \[thread [0-9]+\] \[debug\] //'
echo "=== GFALLBACKDONE $(date +%T) ==="
