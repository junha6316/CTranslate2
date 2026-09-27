#!/bin/bash
# G-short: tier code must be inert when no crossing happens (RESERVE=128, 84-tok window).
cd /opt/real/r6/r5t
run() {
  local tree=$1; shift
  ( source /opt/real/r6/r5t/env.sh "$tree"
    while [ "$1" != "--" ]; do export "$1"; shift; done; shift
    "$@" )
}
G="CT2_CUDA_GRAPHS=1 CT2_CUDA_GRAPHS_RESERVE=128"
echo "=== G-short walls: ABC x3 (A=ref graphs@128, B=tip +64, C=tip TIERS unset) $(date +%T)"
for i in 1 2 3; do
  run ref $G -- python bench_long.py gs_ref gs_ref_$i.json std:0 std:0:nots --beams 5,1
  run r3 $G CT2_CUDA_GRAPHS_TIERS=+64 -- python bench_long.py gs_t64 gs_t64_$i.json std:0 std:0:nots --beams 5,1
  run r3 $G -- python bench_long.py gs_tun gs_tun_$i.json std:0 std:0:nots --beams 5,1
done 2>&1 | grep -v "^ctranslate2.__file__"
run ref $G -- python -c "from common import print_lib_info; print_lib_info()"
run r3 $G -- python -c "from common import print_lib_info; print_lib_info()"
python3 - <<'EOF'
import json, statistics
cfg = {}
for tag in ["ref", "t64", "tun"]:
    for i in (1, 2, 3):
        for r in json.load(open(f"gs_{tag}_{i}.json")):
            cfg.setdefault((tag, r["workload"], r["beam"]), []).append(r)
ok = True
for wl in ["std:0", "std:0:nots"]:
    for beam in (5, 1):
        ref = statistics.median(r["median_ms"] for r in cfg[("ref", wl, beam)])
        line = f"{wl:11s} b{beam}: ref {ref:.2f}"
        for tag in ["t64", "tun"]:
            rows = cfg[(tag, wl, beam)]
            m = statistics.median(r["median_ms"] for r in rows)
            ratio = m / ref
            tok = all(r["tok_ids"] == cfg[("ref", wl, beam)][0]["tok_ids"] and r["score"] == cfg[("ref", wl, beam)][0]["score"] for r in rows)
            tr = sorted({s["transitions"] for r in rows for s in r["summaries"]})
            ok &= ratio <= 1.01 and tok and tr == [0]
            line += f" | {tag} {m:.2f} ({ratio:.4f}x) runs={[r['median_ms'] for r in rows]} tok+score={'same' if tok else 'DIFF'} transitions={tr}"
        print(line)
print("G-short wall/token verdict:", "PASS" if ok else "FAIL")
EOF
echo "=== G-short tokens: 9 standard windows ts on/off x beam 1/5 $(date +%T)"
run ref $G -- python check_windows.py cw_ref cw_ref.json > cw_ref.log 2>&1
run r3 $G CT2_CUDA_GRAPHS_TIERS=+64 -- python check_windows.py cw_t64 cw_t64.json > cw_t64.log 2>&1
run r3 $G -- python check_windows.py cw_tun cw_tun.json > cw_tun.log 2>&1
python3 cmp_ids.py cw_ref.json cw_t64.json
python3 cmp_ids.py cw_ref.json cw_tun.json
echo "tier transitions in cw_t64 / cw_tun summaries (expect only 0):"
python3 -c "
import json
for f in ['cw_t64.json','cw_tun.json']:
    rows=json.load(open(f)); print(f, sorted({s['transitions'] for r in rows for s in r['summaries']}), 'max ntok', max(r['ntok'] for r in rows), 'n_summaries', sum(len(r['summaries']) for r in rows))
"
echo "=== G-short nsys: beam5 ts-on, 7 decodes $(date +%T)"
for t in "ref:" "r3:CT2_CUDA_GRAPHS_TIERS=+64"; do
  tree=${t%%:*}; extra=${t#*:}
  tag=gs_nsys_${tree}
  run $tree $G $extra -- /usr/local/bin/nsys profile -o $tag --force-overwrite true -t cuda python nsys_gen.py 7 > $tag.log 2>&1
  /usr/local/bin/nsys stats --force-export=true --report cuda_api_sum --format csv --force-overwrite true -o $tag $tag.nsys-rep > /dev/null 2>&1
  echo "[$tree $extra]"; grep -E "cudaGraphLaunch|cudaGraphInstantiate|cudaMallocAsync|cudaFreeAsync|,cudaLaunchKernel$" ${tag}_cuda_api_sum.csv | awk -F, '{gsub(/"/,""); print "   ", $NF, "calls=" $3}'
done
echo "=== GSHORTDONE $(date +%T) ==="
