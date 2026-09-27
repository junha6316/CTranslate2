#!/bin/bash
# One sanitizer pass with an explicit python package dir (no usepy switching).
# Usage: sanone.sh LIBTREE PYDIR TAG CELL PASS [ENV=VAL...]   PYDIR "-" = default r6 editable
LIB=$1; PYD=$2; TAG=$3; cell=$4; pass=$5; shift 5
source ~/venv/bin/activate
for v in $(env | grep -o '^CT2_[A-Z0-9_]*'); do unset "$v"; done
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=$HOME/w/$LIB/install/lib:/usr/local/cuda-12.8/lib64
[ "$PYD" != "-" ] && export PYTHONPATH=$PYD
for e in "$@"; do export "$e"; done
case $pass in
  memcheck) args="--tool memcheck";;
  init_gather) args="--tool initcheck --check-api-memory-access no --kernel-name regex=gather_rows_kernel";;
  init_wr) args="--tool initcheck --check-api-memory-access no --kernel-name regex=gather_rows_kernel|thrust|cub|copy_2d|fill|Fill --kernel-name-exclude regex=fprop|nchw|nhwc|cudnn";;
  init_kv) args="--tool initcheck --check-api-memory-access no --kernel-name ${KVRE:-regex=s161616gemm|s16816gemm_fp16|SoftMax} --kernel-name-exclude regex=fprop|nchw|nhwc|cudnn";;
esac
cd /opt/real/r6/san
out=${TAG}_${cell}_$pass.log
python -c "import ctranslate2; print('PKG', ctranslate2.__file__, [l.split()[-1] for l in open('/proc/self/maps') if 'libctranslate2' in l][0])" > $out
t0=$(date +%s)
timeout ${TMO:-2400} /usr/local/cuda-12.8/bin/compute-sanitizer $args --print-limit ${PL:-30} ${SANX} --error-exitcode 0 python /opt/real/r6/san.py $cell >> $out 2>&1; echo "rc=$?" >> $out
echo "$TAG $cell $pass ($(( $(date +%s)-t0 ))s) env[$*]: $(grep -h 'ERROR SUMMARY' $out | head -1) | $(grep -h SANCELL $out) | $(grep -h -A1 'Uninitialized\|Invalid' $out | grep -o 'at [^(]*' | sort | uniq -c | sort -rn | head -3 | tr '\n' ';')"
