#!/bin/bash
# G1: default-path bit-identity of the tiers tip (r3) vs 1a7abfb (ref), same session.
cd /opt/real/r6/r5t
run() {  # run TREE LABEL [ENV=VAL ...] -- CMD...
  local tree=$1; shift
  ( source /opt/real/r6/r5t/env.sh "$tree"
    while [ "$1" != "--" ]; do export "$1"; shift; done; shift
    "$@" )
}
echo "=== G1.1 48-config matrix, all flags off (flash off / on), ref vs tip === $(date +%T)"
for tree in ref r3; do
  run $tree -- python matrix_ids.py g1_$tree g1_${tree}_off.json > g1_${tree}_off.log 2>&1
  run $tree -- python matrix_ids.py g1_$tree g1_${tree}_on.json flash > g1_${tree}_on.log 2>&1
done
python3 cmp_ids.py g1_ref_off.json g1_r3_off.json
python3 cmp_ids.py g1_ref_on.json g1_r3_on.json
echo "=== G1.1b 48-config matrix, CT2_CUDA_GRAPHS=1 RESERVE=128, TIERS unset (flash off) === $(date +%T)"
for tree in ref r3; do
  run $tree CT2_CUDA_GRAPHS=1 CT2_CUDA_GRAPHS_RESERVE=128 -- python matrix_ids.py g1g_$tree g1g_${tree}.json > g1g_${tree}.log 2>&1
done
python3 cmp_ids.py g1g_ref.json g1g_r3.json
echo "=== G1.2 RESERVE=64 without tiers: forced crossing, one decode, VERBOSE=2 === $(date +%T)"
for tree in ref r3; do
  run $tree CT2_CUDA_GRAPHS=1 CT2_CUDA_GRAPHS_RESERVE=64 CT2_VERBOSE=2 -- python onedecode.py std:0 --beam 5 --json g1x_$tree.json > g1x_$tree.log 2>&1
  echo "[$tree] does-not-fit: $(grep -c 'does not fit' g1x_$tree.log)  falling-back: $(grep -c 'falling back' g1x_$tree.log)  tier-lines: $(grep -c 'CUDA graphs: tier' g1x_$tree.log)"
  grep -E "does not fit|falling back|NTOK|lib =" g1x_$tree.log | cut -c1-220
done
python3 - <<'EOF'
import json
a = json.load(open("g1x_ref.json"))[0]; b = json.load(open("g1x_r3.json"))[0]
r4 = [50364, 639, 307, 257, 15834, 470, 53, 5230, 6613, 13, 1057, 15834, 470, 53, 5230, 25162, 366, 294, 264, 1908, 9274, 13, 50714, 50714, 1171, 544, 1589, 293, 281, 915, 484, 577, 291, 393, 13835, 11, 1767, 3441, 15834, 470, 53, 5230, 13, 4646, 13, 51014, 51014, 9647, 3357, 538, 3335, 4271, 479, 664, 446, 88, 13, 440, 5735, 295, 3630, 538, 6163, 314, 11728, 11, 16805, 538, 21704, 338, 460, 4680, 13, 51414, 51414, 4100, 502, 13, 20084, 278, 2149, 599, 51614, 51614]
print("ref vs tip tokens identical:", a["tok_ids"] == b["tok_ids"], " scores:", a["score"], b["score"], "identical:", a["score"] == b["score"])
print("tip tokens == round-4 g3_cross TOKS:", b["tok_ids"] == r4, " round-4 score -0.1947544664144516 ->", b["score"])
EOF
echo "=== G1DONE $(date +%T) ==="
