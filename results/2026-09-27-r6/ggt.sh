#!/bin/bash
# Stage-2 G3 tiers: nsys kern_sum under CT2_CUDA_GRAPHS=1 RESERVE=128 TIERS=+64,
# small fp16 beam5, std:0 (7 decodes) and dense:1 (5 decodes). Usage: ggt.sh TREE TAG
TREE=$1; TAG=$2
mkdir -p /opt/real/r6/gg
for W in std:0 dense:1; do
  ( source /opt/real/r6/r5t/env.sh $TREE
    export CT2_CUDA_GRAPHS=1 CT2_CUDA_GRAPHS_RESERVE=128 CT2_CUDA_GRAPHS_TIERS=+64
    ND=7; [ $W = dense:1 ] && ND=5
    out=/opt/real/r6/gg/${TAG}_tiers_${W/:/}
    /usr/local/bin/nsys profile -o $out --force-overwrite true -t cuda python nsys_gen.py $ND $W 5 > $out.log 2>&1
    /usr/local/bin/nsys stats --force-export=true --report cuda_gpu_kern_sum --format csv --force-overwrite true -o $out $out.nsys-rep > /dev/null 2>&1
    g=$(grep gather_rows_kernel ${out}_cuda_gpu_kern_sum.csv | cut -d, -f2,3,4)
    tot=$(awk -F, 'NR>1{s+=$2} END{print s}' ${out}_cuda_gpu_kern_sum.csv)
    echo "$TAG tiers $W decodes=$ND gather(total_ns,inst,avg_ns)=$g all_kernels_ns=$tot $(grep 'done ntok\|lib' $out.log | tr '\n' ' ')"
  )
done
