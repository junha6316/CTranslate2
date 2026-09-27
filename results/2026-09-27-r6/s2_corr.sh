#!/bin/bash
# Stage-2 correctness only (tokens/scores), run while a sanitizer holds gpu.lock: its walls
# are NOT used. Holds /tmp/s2corr.lock; the queued gpu.lock chains wait on it first.
TAG=s2
cd /opt/real/r6
( source r5t/env.sh r6s2; cd /opt/real/r6
  python -c "import ctranslate2; print(ctranslate2.__file__, [l.split()[-1] for l in open('/proc/self/maps') if 'libctranslate2' in l][0])"
  python scores.py $TAG ${TAG}_sc_off.json 0 | tail -1; python scores.py $TAG ${TAG}_sc_on.json 1 | tail -1
  export CT2_CUDA_GATHER_PREFIX=0
  python scores.py ${TAG}p0 ${TAG}p0_sc_off.json 0 | tail -1; python scores.py ${TAG}p0 ${TAG}p0_sc_on.json 1 | tail -1 )
source ~/venv/bin/activate
for f in off on; do python scmp.py base_sc_$f.json ${TAG}_sc_$f.json; python scmp.py base_sc_$f.json ${TAG}p0_sc_$f.json; done
bash g2.sh r6s2 ${TAG}c > g2_${TAG}c.log 2>&1
python g2cmp.py base ${TAG}c | grep -v SAME | tail -30
python g2cmp.py base ${TAG}c | grep -c SAME
echo S2CORR_DONE
