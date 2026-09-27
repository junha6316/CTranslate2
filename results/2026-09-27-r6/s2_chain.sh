#!/bin/bash
# Stage-2 locked timing chain: lib ~/w/r6s2/install with the r6 python pkg (stage 2 leaves
# the python ext ABI unchanged; RUNPATH lets LD_LIBRARY_PATH pick the lib). Tokens/scores
# were verified in s2_corr.sh. G2 walls interleaved with a fresh base pass, G3 gg (base,
# s1, s2) + anchor nsys, then the 96-config matrix.
TAG=s2
cd /opt/real/r6
echo "== s2_chain start $(date +%T)"
bash usepy.sh r6
bash g2.sh r6s2 $TAG > g2_$TAG.log 2>&1
bash usepy.sh base; bash g2.sh base base2 > g2_base2.log 2>&1
bash gg.sh base base; bash usepy.sh r6
bash gg.sh r6 s1
bash gg.sh r6s2 $TAG
source ~/venv/bin/activate
python g2cmp.py base2 $TAG | tail -70
python g2cmp.py base $TAG | tail -3
bash nsys.sh r6s2 N_$TAG --states poisoned > nsys_$TAG.log 2>&1
grep gather_rows nsys_N_base_cuda_gpu_kern_sum.csv nsys_N_s1_cuda_gpu_kern_sum.csv nsys_N_${TAG}_cuda_gpu_kern_sum.csv | cut -c1-120
bash matrix.sh r6s2 $TAG > ${TAG}_matrix.log 2>&1
python cmp.py base_off.json ${TAG}_off.json | tail -3; python cmp.py base_on.json ${TAG}_on.json | tail -3
python cmp.py s1_off.json ${TAG}_off.json | tail -3; python cmp.py s1_on.json ${TAG}_on.json | tail -3
echo "S2CHAIN_DONE $TAG $(date +%T)"
