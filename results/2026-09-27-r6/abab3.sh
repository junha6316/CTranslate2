#!/bin/bash
# abab3.sh ROUNDS: rotate base / s1 / s1 with CT2_CUDA_GATHER_PREFIX=0 (p0) per round.
source ~/venv/bin/activate
for v in $(env | grep -o '^CT2_[A-Z0-9_]*'); do unset "$v"; done
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH
cd /opt/real/r6; OUT=/opt/real/r6/abab3.jsonl; : > $OUT
run() { case $1 in
  base) PYTHONPATH=/opt/real/r6/pybase LD_LIBRARY_PATH=$HOME/w/base/install/lib:/usr/local/cuda-12.8/lib64 python probe_ab.py base $OUT;;
  s1) PYTHONPATH=$HOME/w/r6/python LD_LIBRARY_PATH=$HOME/w/r6/install/lib:/usr/local/cuda-12.8/lib64 python probe_ab.py s1 $OUT;;
  p0) CT2_CUDA_GATHER_PREFIX=0 PYTHONPATH=$HOME/w/r6/python LD_LIBRARY_PATH=$HOME/w/r6/install/lib:/usr/local/cuda-12.8/lib64 python probe_ab.py p0 $OUT;;
  esac; }
for i in $(seq 1 ${1:-6}); do
  case $((i%3)) in 1) o="base s1 p0";; 2) o="s1 p0 base";; 0) o="p0 base s1";; esac
  for t in $o; do run $t; done
done
echo ABAB3_DONE $(date +%T)
