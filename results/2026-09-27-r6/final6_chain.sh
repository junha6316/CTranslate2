#!/bin/bash
# Round-6 final paired bench chain (base vs r6). All timing under flock /tmp/gpu.lock.
cd /opt/real/r6
L=/tmp/gpu.lock
use() { bash /opt/real/r6/usepy.sh $1; }
stage=${1:-all}
if [ $stage = all ] || [ $stage = matrix ]; then
echo "#### MATRIX $(date +%T)"
for pair in "base r6 1" "r6 base 2"; do set -- $pair
  for t in $1 $2; do
    use $t
    flock $L bash matrix.sh $t final_${t}_$3 > final_${t}_$3_matrix.log 2>&1
    tail -2 final_${t}_$3_matrix.log
  done
done
fi
if [ $stage = all ] || [ $stage = graphs ]; then
echo "#### GRAPHS $(date +%T)"
G=CT2_CUDA_GRAPHS=1
CFGS=("eager:" "sc128:$G CT2_CUDA_GRAPHS_RESERVE=128" "sc448:$G CT2_CUDA_GRAPHS_RESERVE=448" "t64:$G CT2_CUDA_GRAPHS_RESERVE=128 CT2_CUDA_GRAPHS_TIERS=+64")
for pair in "base r6 1" "r6 base 2"; do set -- $pair; rd=$3
  for t in $1 $2; do
    use $t
    for c in "${CFGS[@]}"; do tag=${c%%:*}; envs=${c#*:}
      ( source /opt/real/r6/r5t/env.sh $t; for e in $envs; do export "$e"; done
        flock $L python bench_long.py g6_${t}_$tag /opt/real/r6/final_g6_${t}_${tag}_$rd.json --beams 5,1 std:0 dense:1 2>&1 | grep -v "^$" | tail -5 )
    done
  done
done
fi
if [ $stage = all ] || [ $stage = nsys ]; then
echo "#### NSYS $(date +%T)"
for t in base r6; do
  use $t
  flock $L bash nsys.sh $t final_$t --model small --ctype float16 --states cold > final_nsys_$t.log 2>&1
  tail -3 final_nsys_$t.log
done
fi
if [ $stage = all ] || [ $stage = san ]; then
echo "#### SAN $(date +%T)"
use r6
flock $L bash san2.sh r6 final_r6 flash_b5 b5_GRAPHS
use base
flock $L bash san2.sh base final_base flash_b5 b5_GRAPHS
use r6
fi
echo "FINAL6_DONE $(date +%T)"
