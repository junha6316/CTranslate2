flock /tmp/s2corr.lock true  # wait for an unlocked correctness job (s2_corr.sh)
#!/bin/bash
# Stage-1 G3/G4 extras: large-v3 kern_sum base+s1, then interleaved wall passes 2,3.
cd /opt/real/r6
bash usepy.sh base
bash nsys.sh base N_base_large --states poisoned --model large-v3 > nsys_base_large.log 2>&1
bash usepy.sh r6
bash nsys.sh r6 N_s1_large --states poisoned --model large-v3 > nsys_s1_large.log 2>&1
for i in 2 3; do
  bash usepy.sh base; bash matrix.sh base base$i > base${i}_matrix.log 2>&1
  bash usepy.sh r6; bash matrix.sh r6 s1_$i > s1_${i}_matrix.log 2>&1
done
echo S1POST_DONE
