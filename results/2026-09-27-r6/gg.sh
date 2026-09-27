#!/bin/bash
# Stage-2 G3: nsys kern_sum of graphs pad@128 / pad@448 (small fp16 beam5 std:0 ts-on, 7 decodes,
# r5t/nsys_gen.py). Usage: gg.sh TREE TAG  -> /opt/real/r6/gg/TAG_pad{128,448}_cuda_gpu_kern_sum.csv
TREE=$1; TAG=$2
mkdir -p /opt/real/r6/gg
for R in 128 448; do
  ( source /opt/real/r6/r5t/env.sh $TREE
    export CT2_CUDA_GRAPHS=1 CT2_CUDA_GRAPHS_RESERVE=$R
    out=/opt/real/r6/gg/${TAG}_pad$R
    /usr/local/bin/nsys profile -o $out --force-overwrite true -t cuda python nsys_gen.py 7 std:0 5 > $out.log 2>&1
    /usr/local/bin/nsys stats --force-export=true --report cuda_gpu_kern_sum --format csv --force-overwrite true -o $out $out.nsys-rep > /dev/null 2>&1
    g=$(grep gather_rows_kernel ${out}_cuda_gpu_kern_sum.csv | cut -d, -f2,3,4)
    tot=$(awk -F, 'NR>1{s+=$2} END{print s}' ${out}_cuda_gpu_kern_sum.csv)
    echo "$TAG pad$R gather(total_ns,inst,avg_ns)=$g all_kernels_ns=$tot $(grep 'done ntok\|lib' $out.log | tr '\n' ' ')"
  )
done
echo GGDONE $TAG
