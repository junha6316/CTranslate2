#!/bin/bash
# G-long: every config x workload, fresh process per config, two rounds (forward then
# reverse config order), median of 7 (2 warmup) per row.
cd /opt/real/r6/r5t
run() {
  local tree=$1; shift
  ( source /opt/real/r6/r5t/env.sh "$tree"
    while [ "$1" != "--" ]; do export "$1"; shift; done; shift
    "$@" )
}
MODEL=${MODEL:-small}
PFX=${PFX:-gl}
SPECS=${SPECS:-"dense:0 dense:1 dense:2 dense:3 dense:4 dense:5 prev:64 prev:128 prev:192 prev:223"}
BEAMS=${BEAMS:-5,1}
ROUNDS=${ROUNDS:-"1 2"}
G="CT2_CUDA_GRAPHS=1"
CFGS=("eager:"
      "sc128:$G CT2_CUDA_GRAPHS_RESERVE=128"
      "t64:$G CT2_CUDA_GRAPHS_RESERVE=128 CT2_CUDA_GRAPHS_TIERS=+64"
      "t64w:$G CT2_CUDA_GRAPHS_RESERVE=128 CT2_CUDA_GRAPHS_TIERS=+64 CT2_CUDA_GRAPHS_TIERS_REENTRY=warmup"
      "lad:$G CT2_CUDA_GRAPHS_RESERVE=128 CT2_CUDA_GRAPHS_TIERS=256,448"
      "sc256:$G CT2_CUDA_GRAPHS_RESERVE=256"
      "sc448:$G"
      "t64p:$G CT2_CUDA_GRAPHS_RESERVE=128 CT2_CUDA_GRAPHS_TIERS=+64 CT2_CUDA_POOL_RELEASE_THRESHOLD=512M"
      "sc128p:$G CT2_CUDA_GRAPHS_RESERVE=128 CT2_CUDA_POOL_RELEASE_THRESHOLD=512M")
if [ -n "$ONLY" ]; then  # e.g. ONLY="eager sc128 t64"
  keep=(); for c in "${CFGS[@]}"; do for o in $ONLY; do [ "${c%%:*}" = "$o" ] && keep+=("$c"); done; done
  CFGS=("${keep[@]}")
fi
n=${#CFGS[@]}
for round in $ROUNDS; do
  echo "=== $PFX round $round $(date +%T) model=$MODEL"
  for ((j = 0; j < n; j++)); do
    if [ $((round % 2)) = 1 ]; then i=$j; else i=$((n - 1 - j)); fi
    c=${CFGS[$i]}; tag=${c%%:*}; envs=${c#*:}
    run r3 $envs -- python bench_long.py ${PFX}_$tag ${PFX}_${tag}_$round.json --model $MODEL --beams $BEAMS $SPECS 2>&1
  done
done
echo "=== ${PFX} DONE $(date +%T) ==="
