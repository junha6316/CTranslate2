#!/bin/bash
# nsys profile of the probe's timed window-0 reps only (cudaProfilerApi range).
# Usage: r6_nsys.sh <tree> <label> [probe args...]   (no checkout; profile tree as-is)
set -eo pipefail
TREE="$1"; LABEL="$2"; shift 2
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH
source ~/venv/bin/activate
export CTRANSLATE2_ROOT=$HOME/w/$TREE/install
export LD_LIBRARY_PATH=$HOME/w/$TREE/install/lib:/usr/local/cuda-12.8/lib64
python -c "import ctranslate2; print('ctranslate2.__file__ =', ctranslate2.__file__)"
echo "HEAD: $(cd ~/w/$TREE && git rev-parse --short HEAD)"
OUT=/opt/real/r6/nsys_$LABEL
R6_PROFILE_RANGE=1 /usr/local/bin/nsys profile -t cuda -c cudaProfilerApi --capture-range-end=stop \
  --force-overwrite true -o $OUT \
  python /opt/real/r6/probe.py "$LABEL" --no-windows "$@" | tee -a /opt/real/r6/nsys_probes.jsonl
/usr/local/bin/nsys stats --force-export=true --report cuda_api_sum --report cuda_gpu_kern_sum \
  --format csv --force-overwrite true -o $OUT $OUT.nsys-rep > /dev/null 2>&1
ls -la ${OUT}_*.csv
