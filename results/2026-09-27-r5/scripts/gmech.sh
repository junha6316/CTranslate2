#!/bin/bash
# G-mech: forced transitions on the 84-token window (std:0).
cd /opt/real/r6/r5t
run() {
  local tree=$1; shift
  ( source /opt/real/r6/r5t/env.sh "$tree"
    while [ "$1" != "--" ]; do export "$1"; shift; done; shift
    "$@" )
}
G="CT2_CUDA_GRAPHS=1"
echo "=== G-mech logs (VERBOSE=2, 2 decodes per process) $(date +%T)"
for cfg in "64:+64:5" "64:+64:1" "32:+32:5" "32:+32:1"; do
  IFS=: read R T B <<< "$cfg"
  tag=gm_log_R${R}_b${B}
  run r3 $G CT2_CUDA_GRAPHS_RESERVE=$R CT2_CUDA_GRAPHS_TIERS=$T CT2_VERBOSE=2 -- python onedecode.py std:0 --beam $B --n 2 --json $tag.json > $tag.log 2>&1
  echo "--- R=$R TIERS=$T beam$B: tier-lines=$(grep -c 'CUDA graphs: tier' $tag.log) captures=$(grep -c 'captured step' $tag.log) falling-back=$(grep -c 'falling back' $tag.log) does-not-fit=$(grep -c 'does not fit' $tag.log) summaries=$(grep -c 'decode summary' $tag.log)"
  grep -E "CUDA graphs: tier|captured step|falling back|decode summary|NTOK" $tag.log | sed -E 's/^\[[^]]*\] \[ctranslate2\] \[thread [0-9]+\] //' | cut -c1-230
done
echo "=== G-mech nsys instantiate counts, 7 decodes $(date +%T)"
for cfg in "64:+64:5" "64:+64:1" "32:+32:5"; do
  IFS=: read R T B <<< "$cfg"
  tag=gm_nsys_R${R}_b${B}
  run r3 $G CT2_CUDA_GRAPHS_RESERVE=$R CT2_CUDA_GRAPHS_TIERS=$T -- /usr/local/bin/nsys profile -o $tag --force-overwrite true -t cuda python nsys_gen.py 7 std:0 $B > $tag.log 2>&1
  /usr/local/bin/nsys stats --force-export=true --report cuda_api_sum --format csv --force-overwrite true -o $tag $tag.nsys-rep > /dev/null 2>&1
  echo "[R=$R $T beam$B]"; grep -E "cudaGraphLaunch|cudaGraphInstantiate|cudaMallocAsync" ${tag}_cuda_api_sum.csv | awk -F, '{print "   ", $NF, "calls=" $3}'
done
echo "=== G-mech walls (std:0 ts-on, beam 5 and 1), two interleaved rounds $(date +%T)"
CFGS=("sc64:CT2_CUDA_GRAPHS_RESERVE=64"
      "t64:CT2_CUDA_GRAPHS_RESERVE=64 CT2_CUDA_GRAPHS_TIERS=+64"
      "t64w:CT2_CUDA_GRAPHS_RESERVE=64 CT2_CUDA_GRAPHS_TIERS=+64 CT2_CUDA_GRAPHS_TIERS_REENTRY=warmup"
      "sc32:CT2_CUDA_GRAPHS_RESERVE=32"
      "t32:CT2_CUDA_GRAPHS_RESERVE=32 CT2_CUDA_GRAPHS_TIERS=+32"
      "sc128:CT2_CUDA_GRAPHS_RESERVE=128")
for round in 1 2; do
  if [ $round = 1 ]; then order=(0 1 2 3 4 5); else order=(5 4 3 2 1 0); fi
  for i in "${order[@]}"; do
    c=${CFGS[$i]}; tag=${c%%:*}; envs=${c#*:}
    run r3 $G $envs -- python bench_long.py gm_$tag gm_${tag}_$round.json std:0 --beams 5,1 2>&1 | grep -v "^ctranslate2.__file__"
  done
done
python3 - <<'EOF'
import json, statistics
tags = ["sc64", "t64", "t64w", "sc32", "t32", "sc128"]
med = {}
for t in tags:
    rows = [r for rd in (1, 2) for r in json.load(open(f"gm_{t}_{rd}.json"))]
    for beam in (5, 1):
        rr = [r for r in rows if r["beam"] == beam]
        med[(t, beam)] = statistics.median(r["median_ms"] for r in rr)
        s = rr[-1]["summaries"][-1] if rr[-1]["summaries"] else {}
        print(f"{t:6s} b{beam}: {med[(t, beam)]:.2f} ms  runs={[r['median_ms'] for r in rr]} ntok={rr[-1]['ntok']} score={rr[-1]['score']:.8g} tr={s.get('transitions')} cap={s.get('captures')} rep={s.get('replays')} seg={s.get('segment_replays')}")
x = 0.256
for beam, thr in ((5, 1.5), (1, 1.0)):
    d = med[("sc64", beam)] - med[("t64", beam)]
    print(f"beam{beam}: tiers@64+64 {med[('t64', beam)]:.2f} vs single-cap@64 {med[('sc64', beam)]:.2f}: gain {d:+.2f} ms (need >= {thr}) -> {'PASS' if d >= thr else 'FAIL'};  K ~= 27x - 1.06 - gain = {27*x - 1.06 - d:.2f} ms (x={x})")
    dw = med[("t64", beam)] - med[("t64w", beam)]
    print(f"beam{beam}: fast {med[('t64', beam)]:.2f} vs warmup re-entry {med[('t64w', beam)]:.2f}: fast-warmup {dw:+.2f} ms (need <= +0.3) -> {'PASS' if dw <= 0.3 else 'FAIL'}")
    d32 = med[("sc32", beam)] - med[("t32", beam)]
    print(f"beam{beam}: tiers@32+32 {med[('t32', beam)]:.2f} vs single-cap@32 {med[('sc32', beam)]:.2f}: gain {d32:+.2f} ms; graphs@128 {med[('sc128', beam)]:.2f}")
EOF
echo "=== GMECHDONE $(date +%T) ==="
