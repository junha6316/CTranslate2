#!/bin/bash
# Stage-2 G6 on the MHA cells (s2 = r6 py + r6s2 lib), then base init_kv for comparison.
cd /opt/real/r6
bash usepy.sh r6
bash san2.sh r6s2 s2 b5_GRAPHS b4_PADKV dense1_TIERS
bash usepy.sh base
bash san2.sh base base b5_GRAPHS b4_PADKV dense1_TIERS
bash usepy.sh r6
echo S2SAN_DONE
