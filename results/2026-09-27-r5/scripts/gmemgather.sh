#!/bin/bash
# G-memory (peak process GPU MB, candidate vs single-cap R=448) and G-gather (nsys
# gather_rows_kernel at pad@128 vs pad@448, small beam5).
cd /opt/real/r6/r5t
run() {
  local tree=$1; shift
  ( source /opt/real/r6/r5t/env.sh "$tree"
    while [ "$1" != "--" ]; do export "$1"; shift; done; shift
    "$@" )
}
W="dense:0 dense:1 dense:2 dense:3 dense:4 dense:5 prev:64 prev:128 prev:192 prev:223"
echo "=== G-memory small $(date +%T)"
run r3 CT2_CUDA_GRAPHS=1 CT2_CUDA_GRAPHS_RESERVE=128 CT2_CUDA_GRAPHS_TIERS=+64 -- python mem_probe.py t64 $W 2>&1 | grep -E "MEM|running"
run r3 CT2_CUDA_GRAPHS=1 -- python mem_probe.py sc448 $W 2>&1 | grep -E "MEM|running"
run r3 CT2_CUDA_GRAPHS=1 CT2_CUDA_GRAPHS_RESERVE=128 -- python mem_probe.py sc128 $W 2>&1 | grep -E "MEM"
echo "=== G-memory large-v3 (W_prev 128/223 + dense:1/dense:4) $(date +%T)"
WL="prev:128 prev:223 dense:1 dense:4"
run r3 CT2_CUDA_GRAPHS=1 CT2_CUDA_GRAPHS_RESERVE=128 CT2_CUDA_GRAPHS_TIERS=+64 -- python mem_probe.py t64 --model large-v3 $WL 2>&1 | grep -E "MEM"
run r3 CT2_CUDA_GRAPHS=1 -- python mem_probe.py sc448 --model large-v3 $WL 2>&1 | grep -E "MEM"
echo "=== G-gather nsys gather_rows_kernel, pad@128 vs pad@448 (small beam5 std:0, 7 decodes) $(date +%T)"
for R in 128 448; do
  tag=gg_pad$R
  run r3 CT2_CUDA_PAD_KV=1 CT2_CUDA_GRAPHS_RESERVE=$R -- /usr/local/bin/nsys profile -o $tag --force-overwrite true -t cuda python nsys_gen.py 7 std:0 5 > $tag.log 2>&1
  /usr/local/bin/nsys stats --force-export=true --report cuda_gpu_kern_sum --format csv --force-overwrite true -o $tag $tag.nsys-rep > /dev/null 2>&1
  echo "[pad@$R]"; head -1 ${tag}_cuda_gpu_kern_sum.csv; grep -iE "gather_rows" ${tag}_cuda_gpu_kern_sum.csv | cut -c1-300
done
echo "=== GMEMGATHERDONE $(date +%T) ==="
