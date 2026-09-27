#!/bin/bash
# G2 graphs/prealloc token+score+wall runs. Usage: g2.sh TREE TAG [ENV=VAL ...]
# Writes /opt/real/r6/g2/TAG_<mode>.json (bench_long rows) and TAG_<mode>_b4.json.
TREE=$1; TAG=$2; shift 2
mkdir -p /opt/real/r6/g2
cd /opt/real/r6/r5t
STD="std:0 std:1 std:2 std:3 std:0:nots"
LONG="dense:0 dense:1 dense:3 prev:192"
for mode in prealloc padkv g128 g448 tiers; do
  ( source /opt/real/r6/r5t/env.sh $TREE
    for kv in "$@"; do export "$kv"; done
    case $mode in
      prealloc) export CT2_CUDA_PREALLOC_KV=1; SPECS="$STD";;
      padkv) export CT2_CUDA_PAD_KV=1; SPECS="$STD";;
      g128) export CT2_CUDA_GRAPHS=1 CT2_CUDA_GRAPHS_RESERVE=128; SPECS="$STD";;
      g448) export CT2_CUDA_GRAPHS=1 CT2_CUDA_GRAPHS_RESERVE=448; SPECS="$STD";;
      tiers) export CT2_CUDA_GRAPHS=1 CT2_CUDA_GRAPHS_RESERVE=128 CT2_CUDA_GRAPHS_TIERS=+64; SPECS="$STD $LONG";;
    esac
    env | grep '^CT2_' | tr '\n' ' '; echo
    python bench_long.py ${TAG}_$mode /opt/real/r6/g2/${TAG}_$mode.json $SPECS 2>&1 | grep -v "^\[" | tail -40
    python /opt/real/r6/g2batch.py ${TAG}_${mode}_b4 /opt/real/r6/g2/${TAG}_${mode}_b4.json 2>&1 | tail -2
  )
done
echo "G2DONE $TAG"
