#!/bin/bash
# Writer+reader-instrumented initcheck totals, s2 vs base: -> /opt/real/r6/san/wrall.txt
cd /opt/real/r6
out=san/wrall.txt; : > $out
tot() { local f=$1; local r=$(grep -m1 -o "ERROR SUMMARY: [0-9]*" $f | grep -o "[0-9]*$"); local s=$(grep -o "[0-9]* errors were skipped" $f | grep -o "^[0-9]*"); echo $(( ${r:-0} + ${s:-0} )); }
cellenv() { case $1 in b5) echo "CT2_CUDA_GRAPHS=1 CT2_CUDA_GRAPHS_RESERVE=128";; b4) echo "CT2_CUDA_PAD_KV=1";; dense1) echo "CT2_CUDA_GRAPHS=1 CT2_CUDA_GRAPHS_RESERVE=128 CT2_CUDA_GRAPHS_TIERS=+64";; esac; }
export KVRE="regex=gather_rows_kernel|thrust|cub|copy_2d|fill|Fill|s161616gemm|s16816gemm_fp16|SoftMax"
for c in b4 dense1 b5; do
  for t in r6s2:-:s2 base:/opt/real/r6/pybase:base; do IFS=: read L P T <<< "$t"
    PL=30 SANX="--show-backtrace no" TMO=1800 bash sanone.sh $L $P wa_$T $c init_kv $(cellenv $c) > /dev/null 2>&1
    f=san/wa_${T}_${c}_init_kv.log
    echo "$T $c writers+attn total=$(tot $f) $(grep -h SANCELL $f) $(tail -1 $f) gather_printed=$(grep -A1 Uninitialized $f | grep -c gather_rows)" >> $out
  done
done
echo WRALL_DONE >> $out
