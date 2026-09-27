#!/bin/bash
# G6: memcheck (all kernels) + initcheck restricted to (a) the gather kernel and (b) the
# kernels that read the KV caches (flash FA2, GEMMs, softmax, copies). Full initcheck is
# unusable here: cuDNN's encoder conv (nchwToNhwc) reports pre-existing uninitialized
# reads and the tool then runs for >20 min per decode. Usage: san.sh TREE TAG [cells]
TREE=$1; TAG=$2; shift 2
CELLS=${@:-"flash_b5 flash_b4 b5_GRAPHS b4_PADKV dense1_TIERS"}
mkdir -p /opt/real/r6/san; cd /opt/real/r6/san
CS=/usr/local/cuda-12.8/bin/compute-sanitizer
for name in $CELLS; do
  case $name in
    flash_b5) cell=flash_b5; envs="";;
    flash_b4) cell=flash_b4; envs="";;
    b5_GRAPHS) cell=b5; envs="CT2_CUDA_GRAPHS=1 CT2_CUDA_GRAPHS_RESERVE=128";;
    b4_PADKV) cell=b4; envs="CT2_CUDA_PAD_KV=1";;
    dense1_TIERS) cell=dense1; envs="CT2_CUDA_GRAPHS=1 CT2_CUDA_GRAPHS_RESERVE=128 CT2_CUDA_GRAPHS_TIERS=+64";;
  esac
  for pass in memcheck init_gather init_kv; do
    case $pass in
      memcheck) args="--tool memcheck";;
      init_gather) args="--tool initcheck --check-api-memory-access no --kernel-name regex=gather_rows_kernel";;
      init_kv) args="--tool initcheck --check-api-memory-access no --kernel-name regex=flash|gemm|Kernel2|cutlass|softmax|SoftMax|copy|concat|Concat";;
    esac
    ( source /opt/real/r6/r5t/env.sh $TREE; cd /opt/real/r6/san
      for kv in $envs; do export "$kv"; done
      $CS $args --print-limit 30 --error-exitcode 0 python /opt/real/r6/san.py $cell > ${TAG}_${name}_$pass.log 2>&1 )
    kern=$(grep -h -A1 "Uninitialized\|Invalid" ${TAG}_${name}_$pass.log | grep -o "at [^(]*" | sort | uniq -c | sort -rn | head -3 | tr '\n' ';')
    echo "$TAG $name $pass: $(grep -h 'ERROR SUMMARY' ${TAG}_${name}_$pass.log) | $(grep -h SANCELL ${TAG}_${name}_$pass.log) | $kern"
  done
done
echo SANDONE $TAG
