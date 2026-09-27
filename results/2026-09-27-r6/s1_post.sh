#!/bin/bash
cd /opt/real/r6
bash san.sh r6 s1 > san_s1.log 2>&1
bash usepy.sh base
bash san.sh base base > san_base.log 2>&1
# large-v3 kern sum, base then s1
bash nsys.sh base N_base_large --states poisoned --model large-v3 > nsys_base_large.log 2>&1
bash usepy.sh r6
bash nsys.sh r6 N_s1_large --states poisoned --model large-v3 > nsys_s1_large.log 2>&1
# interleaved wall passes 2,3 (pass 1 = base / s1)
for i in 2 3; do
  bash usepy.sh base; bash matrix.sh base base$i > base${i}_matrix.log 2>&1
  bash usepy.sh r6; bash matrix.sh r6 s1_$i > s1_${i}_matrix.log 2>&1
done
echo S1POST_DONE
