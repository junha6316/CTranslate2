#!/bin/bash
# wait until s1_post2 and s2_chain are gone, then take gpu.lock for abab.sh
while pgrep -f 's1_post[2].sh|s2_chai[n].sh' >/dev/null; do sleep 30; done
cd /opt/real/r6 && flock /tmp/gpu.lock bash abab.sh 6
