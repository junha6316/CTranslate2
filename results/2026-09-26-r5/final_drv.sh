#!/bin/bash
# Round-5 final paired bench driver. r4base = ~/w/bisect @ origin/perf/graphs-beam5, r5 = ~/w/r3.
set -u
cd /opt/real/r6
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH
source ~/venv/bin/activate
sw() {  # switch the python pkg to tree $1 and confirm
  local t=$1
  for v in $(env | grep -o '^CT2_[A-Z0-9_]*'); do unset "$v"; done
  export CTRANSLATE2_ROOT=$HOME/w/$t/install LD_LIBRARY_PATH=$HOME/w/$t/install/lib:/usr/local/cuda-12.8/lib64
  pip install -q -e ~/w/$t/python --force-reinstall --no-deps 2>&1 | tail -2
  python -c "import ctranslate2; print('SWITCH $t: ctranslate2.__file__ =', ctranslate2.__file__)"
  (cd ~/w/$t && echo "HEAD $(git rev-parse --short HEAD) dirty=$(git status --short -uno | wc -l)")
  CUR=$t
}
# run TREE-env'd command with extra CT2 env: rn ENV... -- cmd
rn() {
  ( for v in $(env | grep -o '^CT2_[A-Z0-9_]*'); do unset "$v"; done
    while [ "$1" != "--" ]; do export "$1"; shift; done; shift
    cd /opt/real/r6/r5t; "$@" )
}
G="CT2_CUDA_GRAPHS=1 CT2_CUDA_GRAPHS_RESERVE=128"
SPECS="std:0 dense:0 dense:1 dense:2 dense:3 dense:4 dense:5 prev:64 prev:128 prev:192 prev:223"
STAGE=${STAGE:-all}
if [ $STAGE = all ] || [ $STAGE = matrix ]; then
echo "##### MATRIX ABBA $(date +%T)"
sw bisect; ./matrix.sh bisect final_r4base_1
sw r3;     ./matrix.sh r3 final_r5_1
           ./matrix.sh r3 final_r5_2
sw bisect; ./matrix.sh bisect final_r4base_2
echo "##### MATRIX DONE $(date +%T)"
fi
if [ $STAGE = all ] || [ $STAGE = graphs ]; then
echo "##### GRAPHS $(date +%T)"
for round in 1 2; do
  if [ $round = 1 ]; then order="r4base r5tiers r5sc128 r5eager"; else order="r5eager r5sc128 r5tiers r4base"; fi
  for c in $order; do
    if [ $c = r4base ]; then [ "${CUR:-}" = bisect ] || sw bisect; else [ "${CUR:-}" = r3 ] || sw r3; fi
    case $c in
      r4base)  rn $G -- python bench_long.py final_g_r4base /opt/real/r6/final_g_r4base_$round.json $SPECS ;;
      r5tiers) rn $G CT2_CUDA_GRAPHS_TIERS=+64 -- python bench_long.py final_g_r5tiers /opt/real/r6/final_g_r5tiers_$round.json $SPECS ;;
      r5sc128) rn $G -- python bench_long.py final_g_r5sc128 /opt/real/r6/final_g_r5sc128_$round.json $SPECS ;;
      r5eager) rn -- python bench_long.py final_g_r5eager /opt/real/r6/final_g_r5eager_$round.json $SPECS ;;
    esac 2>&1 | grep -v '^wrote' 
  done
done
# verbose per-decode logs (crossings / fallbacks), 2 decodes each
[ "${CUR:-}" = bisect ] || sw bisect
for s in std:0 dense:1; do for b in 5 1; do
  rn $G CT2_VERBOSE=2 -- python onedecode.py $s --beam $b --n 2 > /opt/real/r6/final_v_r4base_${s/:/}_b$b.log 2>&1
done; done
sw r3
for s in std:0 dense:1; do for b in 5 1; do
  rn $G CT2_CUDA_GRAPHS_TIERS=+64 CT2_VERBOSE=2 -- python onedecode.py $s --beam $b --n 2 > /opt/real/r6/final_v_r5tiers_${s/:/}_b$b.log 2>&1
done; done
echo "##### GRAPHS DONE $(date +%T)"
fi
if [ $STAGE = all ] || [ $STAGE = nsys ]; then
echo "##### NSYS $(date +%T)"
for cfg in "r3:r5tiers:CT2_CUDA_GRAPHS_TIERS=+64" "r3:r5sc128:" "r3:r5eager:" "bisect:r4base:"; do
  IFS=: read tree tag extra <<< "$cfg"
  [ "${CUR:-}" = $tree ] || sw $tree
  if [ $tag = r5eager ]; then envs=""; else envs="$G $extra"; fi
  for b in 5 1; do
    out=/opt/real/r6/final_nsys_${tag}_d1_b$b
    rn $envs -- /usr/local/bin/nsys profile -o $out --force-overwrite true -t cuda python nsys_gen.py 7 dense:1 $b > $out.log 2>&1
    /usr/local/bin/nsys stats --force-export=true --report cuda_api_sum --format csv --force-overwrite true -o $out $out.nsys-rep > /dev/null 2>&1
    echo "[$tag b$b] $(grep -o 'lib = .*' $out.log) $(grep 'done ntok' $out.log)"
    grep -E ",cudaGraphLaunch$|,cudaGraphInstantiate[A-Za-z_]*$|,cudaMallocAsync[A-Za-z_]*$|,cudaFreeAsync[A-Za-z_]*$|,cudaLaunchKernel$|,cudaStreamBeginCapture[A-Za-z_]*$" ${out}_cuda_api_sum.csv | awk -F, '{gsub(/"/,""); print "   ", $NF, "calls=" $3}'
  done
done
echo "##### NSYS DONE $(date +%T)"
fi
if [ $STAGE = all ] || [ $STAGE = san ]; then
echo "##### SANITIZER $(date +%T)"
[ "${CUR:-}" = r3 ] || sw r3
S=/usr/local/cuda-12.8/bin/compute-sanitizer
for cfg in "R128_T64_b5:128:+64:5" "R128_T64_b1:128:+64:1" "R128_T32_b5:128:+32:5"; do
  IFS=: read tag R T b <<< "$cfg"
  out=/opt/real/r6/final_san_$tag.log
  rn CT2_CUDA_GRAPHS=1 CT2_CUDA_GRAPHS_RESERVE=$R CT2_CUDA_GRAPHS_TIERS=$T CT2_VERBOSE=2 -- $S --tool memcheck --print-limit 200 python onedecode.py dense:1 --beam $b > $out 2>&1
  echo "--- $tag: $(grep 'ERROR SUMMARY' $out | head -1) tier-lines=$(grep -c 'CUDA graphs: tier' $out) fallback=$(grep -c 'falling back' $out) $(grep -o 'NTOK [0-9]*' $out | head -1)"
  grep 'CUDA graphs: tier' $out | sed 's/^.*CUDA graphs:/    /'
done
echo "##### SANITIZER DONE $(date +%T)"
fi
echo "===== FINAL ALL DONE $(date +%T) ====="
