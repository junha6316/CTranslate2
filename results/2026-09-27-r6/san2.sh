#!/bin/bash
# G6 v2: memcheck (all kernels) + initcheck on (a) the gather kernel only, (b) the KV-cache
# readers only: FA2 kernels on flash cells, cuBLAS/cutlass fp16 GEMMs + softmax on MHA cells
# (cuDNN conv excluded: pre-existing uninit reads in the encoder conv, and they make
# initcheck crawl). --check-api-memory-access no: host memcpy of never-written pool bytes
# is pre-existing noise. KV=0 skips pass (b) on MHA cells. Usage: san2.sh TREE TAG [cells]
TREE=$1; TAG=$2; shift 2
CELLS=${@:-"flash_b5 flash_b4 b5_GRAPHS b4_PADKV dense1_TIERS"}
mkdir -p /opt/real/r6/san; cd /opt/real/r6/san
CS=/usr/local/cuda-12.8/bin/compute-sanitizer
EXCL="--kernel-name-exclude regex=fprop|nchw|nhwc|cudnn"
for name in $CELLS; do
  case $name in
    flash_b5) cell=flash_b5; envs=""; kv="regex=gather_rows_kernel|flash_fwd";;
    flash_b4) cell=flash_b4; envs=""; kv="regex=gather_rows_kernel|flash_fwd";;
    b5_GRAPHS) cell=b5; envs="CT2_CUDA_GRAPHS=1 CT2_CUDA_GRAPHS_RESERVE=128"; kv="regex=gather_rows_kernel|s161616gemm|s16816gemm_fp16|SoftMax";;
    b4_PADKV) cell=b4; envs="CT2_CUDA_PAD_KV=1"; kv="regex=gather_rows_kernel|s161616gemm|s16816gemm_fp16|SoftMax";;
    dense1_TIERS) cell=dense1; envs="CT2_CUDA_GRAPHS=1 CT2_CUDA_GRAPHS_RESERVE=128 CT2_CUDA_GRAPHS_TIERS=+64"; kv="regex=gather_rows_kernel|s161616gemm|s16816gemm_fp16|SoftMax";;
  esac
  passes="memcheck init_gather init_kv"
  [ "${KV:-1}" = 0 ] && [[ $name != flash* ]] && passes="memcheck init_gather"
  for pass in $passes; do
    case $pass in
      memcheck) args="--tool memcheck";;
      init_gather) args="--tool initcheck --check-api-memory-access no --kernel-name regex=gather_rows_kernel";;
      init_kv) args="--tool initcheck --check-api-memory-access no --kernel-name $kv $EXCL";;
    esac
    t0=$(date +%s)
    ( source /opt/real/r6/r5t/env.sh $TREE; cd /opt/real/r6/san
      for e in $envs; do export "$e"; done
      timeout ${TMO:-2400} $CS $args --print-limit 30 --show-backtrace no --error-exitcode 0 python /opt/real/r6/san.py $cell > ${TAG}_${name}_$pass.log 2>&1; echo "rc=$?" >> ${TAG}_${name}_$pass.log )
    kern=$(grep -h -A1 "Uninitialized\|Invalid" ${TAG}_${name}_$pass.log | grep -o "at [^(]*" | sort | uniq -c | sort -rn | head -3 | tr '\n' ';')
    tot=$(( $(grep -m1 -o "ERROR SUMMARY: [0-9]*" ${TAG}_${name}_$pass.log | grep -o "[0-9]*$" || echo 0) + $(grep -o "[0-9]* errors were skipped" ${TAG}_${name}_$pass.log | grep -o "^[0-9]*" || echo 0) ))
    echo "$TAG $name $pass total=$tot ($(( $(date +%s)-t0 ))s): $(grep -h 'ERROR SUMMARY' ${TAG}_${name}_$pass.log | head -1) | $(grep -h SANCELL ${TAG}_${name}_$pass.log) | $(tail -1 ${TAG}_${name}_$pass.log) | $kern"
  done
done
echo SANDONE $TAG
