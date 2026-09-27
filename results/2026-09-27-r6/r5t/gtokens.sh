#!/bin/bash
# G-tokens: graphs+tiers vs graphs+tiers+CHECK (bit-identical, dscore 0) on the G-mech,
# G-safety and G-long workloads; tiers vs default eager parity.
cd /opt/real/r6/r5t
run() {
  local tree=$1; shift
  ( source /opt/real/r6/r5t/env.sh "$tree"
    while [ "$1" != "--" ]; do export "$1"; shift; done; shift
    "$@" )
}
G="CT2_CUDA_GRAPHS=1"
LONG="dense:0 dense:1 dense:2 dense:3 dense:4 dense:5 prev:64 prev:128 prev:192 prev:223"
echo "=== G-tokens $(date +%T)"
for cfg in "t64s:64:+64:std:0" "t32s:32:+32:std:0 std:1 std:2" "t128l:128:+64:$LONG"; do
  IFS=: read tag R T specs <<< "$cfg"
  run r3 $G CT2_CUDA_GRAPHS_RESERVE=$R CT2_CUDA_GRAPHS_TIERS=$T -- python check_windows.py gt_${tag} gt_${tag}.json $specs > gt_${tag}.log 2>&1
  run r3 $G CT2_CUDA_GRAPHS_RESERVE=$R CT2_CUDA_GRAPHS_TIERS=$T CT2_CUDA_GRAPHS_CHECK=1 -- python check_windows.py gt_${tag}_chk gt_${tag}_chk.json $specs > gt_${tag}_chk.log 2>&1
  echo "--- R=$R TIERS=$T [$specs]: CHECK mismatch lines (log): $(grep -c 'logits mismatch' gt_${tag}_chk.log)"
  python3 cmp_ids.py gt_${tag}.json gt_${tag}_chk.json
  python3 -c "
import json
rows=json.load(open('gt_${tag}_chk.json'))
print('   CHECK probe mismatches', sum(s['probe_mismatch_lines'] for r in rows for s in r['summaries'][:1]), 'transitions per row', [ (r['workload'], r['beam'], r['summaries'][-1]['transitions']) for r in rows ])
"
  run r3 -- python check_windows.py gt_${tag}_eager gt_${tag}_eager.json $specs > gt_${tag}_eager.log 2>&1
  python3 tokdiff.py gt_${tag}_eager.json gt_${tag}.json
done
echo "=== GTOKENSDONE $(date +%T) ==="
