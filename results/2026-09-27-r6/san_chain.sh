flock /tmp/s2corr.lock true  # wait for an unlocked correctness job (s2_corr.sh)
cd /opt/real/r6
KV=0 bash san2.sh r6 s1 > san_s1.log 2>&1
bash usepy.sh base
KV=0 bash san2.sh base base > san_base.log 2>&1
bash usepy.sh r6
echo SANCHAIN_DONE
