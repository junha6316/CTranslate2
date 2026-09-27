#!/bin/bash
# abab.sh ROUNDS: alternate base/s1 processes (ABBA order) on the 3 worst small int8 flash-off cells.
source ~/venv/bin/activate
for v in $(env | grep -o '^CT2_[A-Z0-9_]*'); do unset "$v"; done
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH
cd /opt/real/r6; OUT=/opt/real/r6/abab.jsonl; : > $OUT
run() { if [ $1 = base ]; then PYTHONPATH=/opt/real/r6/pybase LD_LIBRARY_PATH=$HOME/w/base/install/lib:/usr/local/cuda-12.8/lib64 python probe_ab.py base $OUT;
        else PYTHONPATH=$HOME/w/r6/python LD_LIBRARY_PATH=$HOME/w/r6/install/lib:/usr/local/cuda-12.8/lib64 python probe_ab.py s1 $OUT; fi; }
for i in $(seq 1 ${1:-6}); do
  if [ $((i%2)) = 1 ]; then run base; run s1; else run s1; run base; fi
done
echo ABAB_DONE $(date +%T)
