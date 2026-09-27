#!/bin/bash
# After the base matrix: dense windows, prev prompts, base scores, base G2.
cd /opt/real/r6/r5t
( source /opt/real/r6/r5t/env.sh base; python make_dense_windows.py > dense.log 2>&1; python make_prev_prompts.py > prev.log 2>&1 )
ls -la /opt/real/r6/r5t/*.npy /opt/real/r6/r5t/prev_prompts.json
( source /opt/real/r6/r5t/env.sh base; cd /opt/real/r6; python scores.py base base_sc_off.json 0 | tail -1; python scores.py base base_sc_on.json 1 | tail -1 )
bash /opt/real/r6/g2.sh base base
echo BASECHAIN_DONE
