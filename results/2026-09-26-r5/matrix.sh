#!/bin/bash
# Usage: r7_matrix.sh TREE TAG [ENV=VAL ...]
# Runs the full 48-config flash-off matrix then the flash-on matrix with the python pkg
# of ~/w/TREE, writing /opt/real/r6/TAG_off.json and TAG_on.json.
set -u
TREE=$1; TAG=$2; shift 2
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH
source ~/venv/bin/activate
export CTRANSLATE2_ROOT=$HOME/w/$TREE/install LD_LIBRARY_PATH=$HOME/w/$TREE/install/lib:/usr/local/cuda-12.8/lib64
for kv in "$@"; do export "$kv"; done
cd /opt/real
echo "== $TAG start $(date +%T) tree=$TREE env: $*"
python -c 'import ctranslate2; print("ctranslate2.__file__ =", ctranslate2.__file__)'
ls -la --time-style=+%H:%M $HOME/w/$TREE/install/lib/libctranslate2.so* | tail -1
(cd $HOME/w/$TREE && git log --oneline -1 && git status --short -uno | head)
env | grep '^CT2_' || true
python bench_batch.py ${TAG}_off /opt/real/r6/${TAG}_off.json
echo "== $TAG off done $(date +%T)"
python bench_batch_flash.py ${TAG}_on /opt/real/r6/${TAG}_on.json
echo "== $TAG on done $(date +%T)"
