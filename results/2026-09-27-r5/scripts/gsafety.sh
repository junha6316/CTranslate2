#!/bin/bash
# G-safety: sanitizer, CHECK dual-run, phase coverage, batch-2 early EOS.
cd /opt/real/r6/r5t
run() {
  local tree=$1; shift
  ( source /opt/real/r6/r5t/env.sh "$tree"
    while [ "$1" != "--" ]; do export "$1"; shift; done; shift
    "$@" )
}
G="CT2_CUDA_GRAPHS=1"
SAN="/usr/local/cuda-12.8/bin/compute-sanitizer --tool memcheck --track-stream-ordered-races all"
PP="pp:20 pp:21 pp:22 pp:23 pp:24 pp:25 pp:26 pp:27 pp:28 pp:29 pp:30 pp:31"
echo "=== G-safety.1 CHECK dual-run (0 mismatches, >=1 replay after every transition) $(date +%T)"
run r3 $G CT2_CUDA_GRAPHS_RESERVE=32 CT2_CUDA_GRAPHS_TIERS=+32 CT2_CUDA_GRAPHS_CHECK=1 -- python check_windows.py gsc_R32 gsc_R32.json std:0 std:1 std:2 > gsc_R32.log 2>&1
run r3 $G CT2_CUDA_GRAPHS_RESERVE=128 CT2_CUDA_GRAPHS_TIERS=+64 CT2_CUDA_GRAPHS_CHECK=1 -- python check_windows.py gsc_R128 gsc_R128.json dense:1 dense:4 dense:5 > gsc_R128.log 2>&1
for f in gsc_R32 gsc_R128; do
  echo "--- $f: mismatch lines in log: $(grep -c 'logits mismatch' $f.log)"; grep -E "^gsc" $f.log
done
python3 - <<'EOF'
import json
for f in ["gsc_R32.json", "gsc_R128.json"]:
    rows = json.load(open(f))
    bad = []
    for r in rows:
        for s in r["summaries"]:
            if s["probe_mismatch_lines"] or any(x < 1 for x in s["segment_replays"][1:]) or s["disabled"] != "none":
                bad.append((r["workload"], r["beam"], s))
    print(f, "rows", len(rows), "transitions", sorted({s['transitions'] for r in rows for s in r['summaries']}),
          "probe mismatches", sum(s["probe_mismatch_lines"] for r in rows for s in r["summaries"][:1]),
          "->", "PASS" if not bad else f"FAIL {bad}")
EOF
echo "=== G-safety.2 phase coverage P=20..31 at R=32 +32 beam5 (CHECK=1, VERBOSE=2) $(date +%T)"
run r3 $G CT2_CUDA_GRAPHS_RESERVE=32 CT2_CUDA_GRAPHS_TIERS=+32 CT2_CUDA_GRAPHS_CHECK=1 CT2_VERBOSE=2 -- python onedecode.py $PP --beam 5 --json gsp_check.json > gsp_check.log 2>&1
echo "mismatch lines: $(grep -c 'logits mismatch' gsp_check.log)  falling-back: $(grep -c 'falling back' gsp_check.log)"
grep -E "BEGIN|CUDA graphs: tier|falling back|NTOK" gsp_check.log | sed -E 's/^\[[^]]*\] \[ctranslate2\] \[thread [0-9]+\] \[debug\] //' | cut -c1-160
echo "phases at the transitions: $(grep -o 'at step 32 (from [A-Za-z]*' gsp_check.log | sort | uniq -c | tr '\n' ' ')"
run r3 $G CT2_CUDA_GRAPHS_RESERVE=32 CT2_CUDA_GRAPHS_TIERS=+32 -- python onedecode.py $PP --beam 5 --json gsp_nocheck.json > gsp_nocheck.log 2>&1
python3 cmp_ids.py gsp_check.json gsp_nocheck.json
echo "=== G-safety.3 batch-2 early EOS (std:0 + std:7, R=64 +64, beam5) $(date +%T)"
run r3 $G CT2_CUDA_GRAPHS_RESERVE=64 CT2_CUDA_GRAPHS_TIERS=+64 CT2_VERBOSE=2 -- python onedecode.py std:0 --batch2 std:7 --beam 5 --json gsb_tiers.json > gsb_tiers.log 2>&1
run r3 -- python onedecode.py std:0 --batch2 std:7 --beam 5 --json gsb_eager.json > gsb_eager.log 2>&1
grep -E "CUDA graphs: tier|falling back|does not fit|decode summary|NTOK" gsb_tiers.log | sed -E 's/^\[[^]]*\] \[ctranslate2\] \[thread [0-9]+\] \[debug\] //' | cut -c1-200
grep NTOK gsb_eager.log
python3 -c "
import json
a=json.load(open('gsb_tiers.json')); b=json.load(open('gsb_eager.json'))
print('batch2 tiers vs eager tokens identical:', [x['tok_ids']==y['tok_ids'] for x,y in zip(a,b)], 'scores', [(x['score'],y['score']) for x,y in zip(a,b)])
"
echo "=== G-safety.4 compute-sanitizer memcheck + stream-ordered races $(date +%T)"
run r3 $G CT2_CUDA_GRAPHS_RESERVE=32 CT2_CUDA_GRAPHS_TIERS=+32 CT2_VERBOSE=2 -- $SAN python onedecode.py std:0 --beam 5 > gss_R32_b5.log 2>&1
run r3 $G CT2_CUDA_GRAPHS_RESERVE=32 CT2_CUDA_GRAPHS_TIERS=+32 CT2_VERBOSE=2 -- $SAN python onedecode.py std:0 --beam 1 > gss_R32_b1.log 2>&1
run r3 $G CT2_CUDA_GRAPHS_RESERVE=128 CT2_CUDA_GRAPHS_TIERS=+64 CT2_VERBOSE=2 -- $SAN python onedecode.py dense:1 --beam 5 > gss_R128_d1.log 2>&1
run r3 $G CT2_CUDA_GRAPHS_RESERVE=32 CT2_CUDA_GRAPHS_TIERS=+32 CT2_VERBOSE=2 -- $SAN python onedecode.py $PP --beam 5 > gss_pp.log 2>&1
for f in gss_R32_b5 gss_R32_b1 gss_R128_d1 gss_pp; do
  echo "--- $f: $(grep -E 'ERROR SUMMARY' $f.log)  tier-lines=$(grep -c 'CUDA graphs: tier' $f.log) falling-back=$(grep -c 'falling back' $f.log) NTOK=$(grep -o 'NTOK [0-9]*' $f.log | tr '\n' ' ')"
  echo "    phases: $(grep -o '(from [A-Za-z]*' $f.log | sort | uniq -c | tr '\n' ' ')"
done
echo "=== GSAFETYDONE $(date +%T) ==="
