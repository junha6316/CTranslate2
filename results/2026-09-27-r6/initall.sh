#!/bin/bash
# init_gather totals (no backtraces) for all 5 G6 cells x {s1, s2, base}; then MHA init_kv
# (gather + attention GEMMs + softmax) totals for s2 vs base. -> /opt/real/r6/san/initall.txt
cd /opt/real/r6
out=san/initall.txt; : > $out
tot() { local f=$1; local r=$(grep -m1 -o "ERROR SUMMARY: [0-9]*" $f | grep -o "[0-9]*$"); local s=$(grep -o "[0-9]* errors were skipped" $f | grep -o "^[0-9]*"); echo $(( ${r:-0} + ${s:-0} )); }
cellenv() { case $1 in flash_b5|flash_b4) echo "";; b5) echo "CT2_CUDA_GRAPHS=1 CT2_CUDA_GRAPHS_RESERVE=128";; b4) echo "CT2_CUDA_PAD_KV=1";; dense1) echo "CT2_CUDA_GRAPHS=1 CT2_CUDA_GRAPHS_RESERVE=128 CT2_CUDA_GRAPHS_TIERS=+64";; esac; }
run() { local L=$1 P=$2 T=$3 c=$4 pass=$5; shift 5
  PL=30 SANX="--show-backtrace no" TMO=${TMO:-900} bash sanone.sh $L $P ia_$T $c $pass $(cellenv $c) > /dev/null 2>&1
  local f=san/ia_${T}_${c}_$pass.log
  echo "$T $c $pass ${KVRE:-} total=$(tot $f) $(grep -h SANCELL $f) $(tail -1 $f)" >> $out; }
for c in flash_b5 flash_b4 b5 b4 dense1; do
  run r6 - s1 $c init_gather; run r6s2 - s2 $c init_gather; run base /opt/real/r6/pybase base $c init_gather
done
export KVRE="regex=gather_rows_kernel|s161616gemm|s16816gemm_fp16|SoftMax"
for c in b5 b4 dense1; do run r6s2 - s2 $c init_kv; run base /opt/real/r6/pybase base $c init_kv; done
echo INITALL_DONE >> $out
