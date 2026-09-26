#!/bin/bash
# G-large: large-v3 fp16 beam5 x (replay saving/step) and b (padding tax/slot/step) at
# R=128, then W_prev k in {128,223} + 2 dense windows, tiers vs single cap.
cd /opt/real/r6/r5t
run() {
  local tree=$1; shift
  ( source /opt/real/r6/r5t/env.sh "$tree"
    while [ "$1" != "--" ]; do export "$1"; shift; done; shift
    "$@" )
}
echo "=== G-large.1 x and b: eager / pad@128 / graphs@128 on std:0..2, beam5, two rounds $(date +%T)"
for round in 1 2; do
  run r3 -- python bench_long.py glx_eager glx_eager_$round.json --model large-v3 --beams 5 std:0 std:1 std:2 2>&1 | grep -v "__file__"
  run r3 CT2_CUDA_PAD_KV=1 CT2_CUDA_GRAPHS_RESERVE=128 -- python bench_long.py glx_pad128 glx_pad128_$round.json --model large-v3 --beams 5 std:0 std:1 std:2 2>&1 | grep -v "__file__"
  run r3 CT2_CUDA_GRAPHS=1 CT2_CUDA_GRAPHS_RESERVE=128 -- python bench_long.py glx_g128 glx_g128_$round.json --model large-v3 --beams 5 std:0 std:1 std:2 2>&1 | grep -v "__file__"
done
python3 - <<'EOF'
import json, statistics
m = {}
for c in ("eager", "pad128", "g128"):
    for rd in (1, 2):
        for r in json.load(open(f"glx_{c}_{rd}.json")):
            m.setdefault((c, r["workload"]), []).append(r)
xs, bs = [], []
for wl in ("std:0", "std:1", "std:2"):
    e, p, g = (statistics.median(r["median_ms"] for r in m[(c, wl)]) for c in ("eager", "pad128", "g128"))
    ntok = m[("g128", wl)][0]["ntok"]
    steps = ntok + 4          # decode steps incl. the beam tail (approximation)
    t0 = 2                    # first decode step (2-token prompt forward)
    slot_steps = sum(128 - t for t in range(t0, t0 + steps))
    x = (p - g) / steps
    b = (p - e) / slot_steps * 1000
    xs.append(x); bs.append(b)
    print(f"{wl}: ntok {ntok} eager {e:.1f} pad@128 {p:.1f} graphs@128 {g:.1f} | tax {p - e:+.1f} ms, replay saving {p - g:+.1f} ms | x={x:.3f} ms/step b={b:.3f} us/slot/step")
print(f"large-v3 beam5: mean x={sum(xs)/3:.3f} ms/step, mean b={sum(bs)/3:.3f} us/slot/step -> model stride sqrt(2K/b) needs K (see G-mech)")
EOF
echo "=== G-large.2 W_prev k=128,223 + dense:1, dense:4: tiers vs single cap, two rounds $(date +%T)"
MODEL=large-v3 PFX=glg SPECS="prev:128 prev:223 dense:1 dense:4" BEAMS=5 ONLY="sc128 t64" ./glong.sh 2>&1 | grep -v "__file__"
python3 glong_report.py glg
echo "=== GLARGEDONE $(date +%T) ==="
