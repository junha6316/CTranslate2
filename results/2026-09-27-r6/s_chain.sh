#!/bin/bash
# Stage run on ~/w/r6: s_chain.sh TAG. G1 matrix + scores (prefix on and =0), G2, G3 nsys.
TAG=$1
cd /opt/real/r6
bash usepy.sh r6
bash matrix.sh r6 $TAG > ${TAG}_matrix.log 2>&1
( source r5t/env.sh r6; cd /opt/real/r6
  python scores.py $TAG ${TAG}_sc_off.json 0 | tail -1; python scores.py $TAG ${TAG}_sc_on.json 1 | tail -1
  export CT2_CUDA_GATHER_PREFIX=0
  python scores.py ${TAG}p0 ${TAG}p0_sc_off.json 0 | tail -1; python scores.py ${TAG}p0 ${TAG}p0_sc_on.json 1 | tail -1 )
bash nsys.sh r6 N_$TAG --states poisoned > nsys_$TAG.log 2>&1
bash g2.sh r6 $TAG > g2_$TAG.log 2>&1
source ~/venv/bin/activate
python cmp.py base_off.json ${TAG}_off.json; python cmp.py base_on.json ${TAG}_on.json
for f in off on; do python scmp.py base_sc_$f.json ${TAG}_sc_$f.json; python scmp.py base_sc_$f.json ${TAG}p0_sc_$f.json; done
grep gather_rows nsys_N_base_cuda_gpu_kern_sum.csv nsys_N_${TAG}_cuda_gpu_kern_sum.csv | cut -c1-120
python g2cmp.py base $TAG | tail -60
echo SCHAIN_DONE $TAG
