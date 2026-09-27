#!/bin/bash
# G-safety sanitizer: memcheck (round-4 pattern) and memcheck + stream-ordered
# use-after-free race tracking. "--track-stream-ordered-races all" is unusable here: its
# use-before-alloc check aborts the ENCODER's cuDNN conv (nchwToNhwcKernel) on the
# unmodified 1a7abfb build in plain eager mode too (gss_ref_eager_all.log).
cd /opt/real/r6/r5t
run() {
  local tree=$1; shift
  ( source /opt/real/r6/r5t/env.sh "$tree"
    while [ "$1" != "--" ]; do export "$1"; shift; done; shift
    "$@" )
}
G="CT2_CUDA_GRAPHS=1 CT2_VERBOSE=2"
PP="pp:20 pp:21 pp:22 pp:23 pp:24 pp:25 pp:26 pp:27 pp:28 pp:29 pp:30 pp:31"
S=/usr/local/cuda-12.8/bin/compute-sanitizer
report() {
  local f=$1
  echo "--- $f: $(grep -E 'ERROR SUMMARY' $f.log | head -1)  tier-lines=$(grep -c 'CUDA graphs: tier' $f.log) falling-back=$(grep -c 'falling back' $f.log) NTOK=$(grep -o 'NTOK [0-9]*' $f.log | tr '\n' ' ')"
  echo "    phases: $(grep -o 'step [0-9]* (from [A-Za-z]*' $f.log | sed 's/step [0-9]* //' | sort | uniq -c | tr '\n' ' ')"
}
for mode in uaf plain; do
  if [ $mode = uaf ]; then SAN="$S --tool memcheck --track-stream-ordered-races use-after-free"; else SAN="$S --tool memcheck"; fi
  echo "=== sanitizer mode=$mode: $SAN $(date +%T)"
  if [ $mode = uaf ]; then
    run ref -- $SAN python onedecode.py std:0 --beam 1 > gsan_${mode}_ref_eager.log 2>&1; report gsan_${mode}_ref_eager
  fi
  run r3 $G CT2_CUDA_GRAPHS_RESERVE=32 CT2_CUDA_GRAPHS_TIERS=+32 -- $SAN python onedecode.py std:0 --beam 5 > gsan_${mode}_R32_b5.log 2>&1; report gsan_${mode}_R32_b5
  run r3 $G CT2_CUDA_GRAPHS_RESERVE=32 CT2_CUDA_GRAPHS_TIERS=+32 -- $SAN python onedecode.py std:0 --beam 1 > gsan_${mode}_R32_b1.log 2>&1; report gsan_${mode}_R32_b1
  run r3 $G CT2_CUDA_GRAPHS_RESERVE=128 CT2_CUDA_GRAPHS_TIERS=+64 -- $SAN python onedecode.py dense:1 --beam 5 > gsan_${mode}_R128_d1.log 2>&1; report gsan_${mode}_R128_d1
  run r3 $G CT2_CUDA_GRAPHS_RESERVE=32 CT2_CUDA_GRAPHS_TIERS=+32 -- $SAN python onedecode.py $PP --beam 5 > gsan_${mode}_pp.log 2>&1; report gsan_${mode}_pp
done
echo "=== GSANDONE $(date +%T) ==="
