#!/bin/bash
cd /opt/real/r6
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH
source ~/venv/bin/activate
for v in $(env | grep -o '^CT2_[A-Z0-9_]*'); do unset "$v"; done
for i in 1 2 3; do
  for t in bisect r3; do
    export CTRANSLATE2_ROOT=$HOME/w/$t/install LD_LIBRARY_PATH=$HOME/w/$t/install/lib:/usr/local/cuda-12.8/lib64
    pip install -q -e ~/w/$t/python --force-reinstall --no-deps 2>&1 | tail -2
    tag=$([ $t = bisect ] && echo r4base || echo r5)
    python final_one.py ${tag}_$i final_one_${tag}_$i.json
  done
done
python3 - <<'PY'
import json, statistics
for k in [(1, 4), (5, 1)]:
    a = [r["median_ms"] for i in (1,2,3) for r in json.load(open(f"final_one_r4base_{i}.json")) if (r["beam"], r["batch"]) == k]
    b = [r["median_ms"] for i in (1,2,3) for r in json.load(open(f"final_one_r5_{i}.json")) if (r["beam"], r["batch"]) == k]
    print(f"b{k[0]} n={k[1]}: r4base {[round(x,1) for x in a]} r5 {[round(x,1) for x in b]} delta-of-medians {(statistics.median(b)-statistics.median(a))/statistics.median(a)*100:+.2f}%")
PY
echo ONEDONE
