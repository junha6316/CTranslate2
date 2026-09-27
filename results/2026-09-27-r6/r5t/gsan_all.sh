#!/bin/bash
# G-safety sanitizer, full stream-ordered race tracking ("all") with only the pre-existing
# encoder cuDNN conv kernels (regex cudnn: nchwToNhwcKernel, xmma fprop; flagged on 1a7abfb eager too) excluded
# from checking. Baselines: 1a7abfb eager and 1a7abfb graphs@128 under the same flags.
cd /opt/real/r6/r5t
run() {
  local tree=$1; shift
  ( source /opt/real/r6/r5t/env.sh "$tree"
    while [ "$1" != "--" ]; do export "$1"; shift; done; shift
    "$@" )
}
G="CT2_CUDA_GRAPHS=1 CT2_VERBOSE=2"
PP="pp:20 pp:21 pp:22 pp:23 pp:24 pp:25 pp:26 pp:27 pp:28 pp:29 pp:30 pp:31"
SAN="/usr/local/cuda-12.8/bin/compute-sanitizer --tool memcheck --track-stream-ordered-races all --kernel-name-exclude regex=cudnn --print-limit 1000"
report() {
  local f=$1
  echo "--- $f: $(grep -E 'ERROR SUMMARY' $f.log | head -1)  exit-errors=$(grep -c 'terminate called' $f.log) tier-lines=$(grep -c 'CUDA graphs: tier' $f.log) falling-back=$(grep -c 'falling back' $f.log) NTOK=$(grep -o 'NTOK [0-9]*' $f.log | tr '\n' ' ')"
  echo "    phases: $(grep -o 'step [0-9]* (from [A-Za-z]*' $f.log | sed 's/step [0-9]* //' | sort | uniq -c | tr '\n' ' ')"
  echo "    error kinds: $(grep -E '^========= [A-Z]' $f.log | grep -v 'ERROR SUMMARY\|COMPUTE-SANITIZER' | sed -E 's/ (on|at|of size).*//' | sort | uniq -c | tr '\n' ' ')"
}
echo "=== sanitizer all-races, excluding cuDNN-internal kernels (regex=cudnn) $(date +%T)"
run ref -- $SAN python onedecode.py std:0 --beam 5 > gsanA_ref_eager.log 2>&1; report gsanA_ref_eager
run ref $G CT2_CUDA_GRAPHS_RESERVE=128 -- $SAN python onedecode.py std:0 --beam 5 > gsanA_ref_g128.log 2>&1; report gsanA_ref_g128
run r3 $G CT2_CUDA_GRAPHS_RESERVE=32 CT2_CUDA_GRAPHS_TIERS=+32 -- $SAN python onedecode.py std:0 --beam 5 > gsanA_R32_b5.log 2>&1; report gsanA_R32_b5
run r3 $G CT2_CUDA_GRAPHS_RESERVE=32 CT2_CUDA_GRAPHS_TIERS=+32 -- $SAN python onedecode.py std:0 --beam 1 > gsanA_R32_b1.log 2>&1; report gsanA_R32_b1
run r3 $G CT2_CUDA_GRAPHS_RESERVE=128 CT2_CUDA_GRAPHS_TIERS=+64 -- $SAN python onedecode.py dense:1 --beam 5 > gsanA_R128_d1.log 2>&1; report gsanA_R128_d1
run r3 $G CT2_CUDA_GRAPHS_RESERVE=32 CT2_CUDA_GRAPHS_TIERS=+32 -- $SAN python onedecode.py $PP --beam 5 > gsanA_pp.log 2>&1; report gsanA_pp
echo "=== small beam5 padding tax b (std:0): eager / pad@128 / pad@448, two rounds $(date +%T)"
for round in 1 2; do
  run r3 -- python bench_long.py gb_eager gb_eager_$round.json std:0 --beams 5 2>&1 | grep -v "__file__\|^wrote"
  run r3 CT2_CUDA_PAD_KV=1 CT2_CUDA_GRAPHS_RESERVE=128 -- python bench_long.py gb_pad128 gb_pad128_$round.json std:0 --beams 5 2>&1 | grep -v "__file__\|^wrote"
  run r3 CT2_CUDA_PAD_KV=1 -- python bench_long.py gb_pad448 gb_pad448_$round.json std:0 --beams 5 2>&1 | grep -v "__file__\|^wrote"
done
python3 - <<'EOF'
import json, statistics
m = {c: statistics.median(r["median_ms"] for rd in (1, 2) for r in json.load(open(f"gb_{c}_{rd}.json"))) for c in ("eager", "pad128", "pad448")}
ntok = json.load(open("gb_pad128_1.json"))[0]["ntok"]
steps = range(2, 2 + ntok + 6)          # decode steps incl. the beam tail (~92 steps logged)
extra = sum(448 - t for t in steps) - sum(128 - t for t in steps)
b = (m["pad448"] - m["pad128"]) / extra * 1000
print(f"eager {m['eager']:.2f} pad@128 {m['pad128']:.2f} pad@448 {m['pad448']:.2f} ms; pad@448 - pad@128 = {m['pad448'] - m['pad128']:.2f} ms over {extra} extra slot-steps -> b = {b:.3f} us/slot/step")
EOF
echo "=== GSANALLDONE $(date +%T) ==="
